import Foundation
import Metal

/// **The blocks' elementwise ops, on the GPU** (Krea 2's, then Z-Image's SwiGLU) — on
/// the model of `AnimaGELU`.
///
/// On the CPU, at 1024², a block spent ~1.5 s per evaluation between two GEMMs: SwiGLU 0.76 s, norms
/// and residuals 0.47 s, gate 0.29 s — during which the GPU waited, since the next GEMM
/// depends on them. Here, three kernels, each **judged against the double**
/// before being wired in (the rule: never further from the exact than the path it replaces),
/// then by the guided trajectory against its fp64 references:
///
///   - `swiglu`  : `a ← a·b / (1 + e^{−a})` — `a·b` as a pair, `e^{−a}` as a pair (`exp_pair`), a
///     corrected quotient: one final rounding instead of the four of `Ops.siluGate`;
///   - `sigmoid_gate` (Swift: `carries`): `x ← x / (1 + e^{−g})`, the attention's sigmoid gate, same quotient;
///   - `residual`: `x ← x + gate[d] · y` in **two roundings**, the product then the sum, as torch computes
///     `x + gate * y` on the CPU (and `Ops.gatedResidual`): one `fma` was closer to the exact and moved
///     one value in nine by an ulp away from the reference. `.safe` does not forbid the contraction of
///     `a * b + c` into an FMA; the kernel turns it off (`#pragma METAL fp contract(off)`), and
///     `siliconed-dev elementwise` checks it to the bit against the CPU;
///
/// **The norms are not there, and it is a refutation**: an RMSNorm with compensated sum,
/// more accurate than the CPU's according to the judge, nonetheless moved the trajectory away from its fp64 reference (step 3:
/// 2.3·10⁻³ against 6.0·10⁻⁴ on the CPU). See `Krea2DiT.modulatedNorm`.
///
/// The division is the `precise::` version, the exponential is written out (`exp_pair`):
/// no fast-math (`mathMode = .safe`), like Anima's GELU.
package final class ElementwiseGPU {
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // Exact sum of two floats: a + b = s + e.
    static inline float2 two_sum(float a, float b) {
        float s = a + b, bp = s - a;
        return float2(s, (a - (s - bp)) + (b - bp));
    }

    // e^{x} in two floats (hi + lo): Cody–Waite reduction (ln 2 split, k·ln2_hi exact),
    // Taylor up to r⁷ on |r| ≤ ln2/2 (remainder < 6·10⁻⁹ relative), 1 + q kept as a pair.
    // `precise::exp` alone, then 1 + e and the division each rounded, ended up a little further
    // from the exact than `vvexpf` on average — the judge rejected it twice.
    static inline float2 exp_pair(float x) {
        const float log2e = 1.44269502162933350f, ln2_hi = 0.693115234375f, ln2_lo = 3.194618329871446e-05f;
        float k = rint(x * log2e);
        float r = fma(-k, ln2_lo, x - k * ln2_hi);
        float q = 1.0f/5040.0f;
        q = fma(q, r, 1.0f/720.0f); q = fma(q, r, 1.0f/120.0f); q = fma(q, r, 1.0f/24.0f);
        q = fma(q, r, 1.0f/6.0f);   q = fma(q, r, 0.5f);
        q = fma(q * r, r, r);                                  // e^r − 1
        float2 e = two_sum(1.0f, q);
        int n = int(k);
        return float2(ldexp(e.x, n), ldexp(e.y, n));
    }

    // p / (1 + e^{−v}), p given as a pair (hi, lo): the denominator as a pair, then a quotient
    // corrected by one step. Beyond v < −80, e^{−v} overflows: the value is ±0, as on the CPU.
    static inline float over_one_plus_exp(float2 p, float v) {
        if (v < -80.0f) return p.x / (1.0f + precise::exp(-v));
        float2 e = exp_pair(-v);
        float2 s = two_sum(1.0f, e.x); s.y += e.y;
        float qt = precise::divide(p.x, s.x);
        float remainder = fma(-qt, s.x, p.x) + p.y - qt * s.y;
        return qt + precise::divide(remainder, s.x);
    }

    kernel void swiglu(device float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                       constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        float v = a[i], w = b[i];
        float p = v * w;
        a[i] = over_one_plus_exp(float2(p, fma(v, w, -p)), v);
    }

    \(BFloatKernels.roundingSource)

    // The `_rounded` kernels write the fp32 value beside the bf16 copy only where a reader reads it (`RoundedOperand.writesFP32`):
    // two pipelines of one source, the values computed alike — the same bits in the copy either way.
    constant bool keeps_fp32 [[function_constant(0)]];

    // The same, and the value rounded to bf16 where the next product reads its operand (`WeightProducts.RoundedOperand`): two
    // values per thread, one 32-bit store, as `round_bf16` writes them. `n` is the count of pairs.
    kernel void swiglu_rounded(device float2 *a [[buffer(0)]], device const float2 *b [[buffer(1)]], device uint *r [[buffer(2)]],
                               constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        float2 v = a[i], w = b[i];
        float2 p = v * w;
        float2 y = float2(over_one_plus_exp(float2(p.x, fma(v.x, w.x, -p.x)), v.x),
                          over_one_plus_exp(float2(p.y, fma(v.y, w.y, -p.y)), v.y));
        if (keeps_fp32) a[i] = y;
        r[i] = rounded(y.x) | (rounded(y.y) << 16);
    }

    // `sigmoid_gate`, and the value rounded as `swiglu_rounded` writes it. `n` is the count of pairs.
    kernel void sigmoid_gate_rounded(device float2 *x [[buffer(0)]], device const float2 *g [[buffer(1)]], device uint *r [[buffer(2)]],
                                     constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        float2 v = x[i], w = g[i];
        float2 y = float2(over_one_plus_exp(float2(v.x, 0.0f), w.x), over_one_plus_exp(float2(v.y, 0.0f), w.y));
        if (keeps_fp32) x[i] = y;
        r[i] = rounded(y.x) | (rounded(y.y) << 16);
    }

    kernel void sigmoid_gate(device float *x [[buffer(0)]], device const float *g [[buffer(1)]],
                      constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        x[i] = over_one_plus_exp(float2(x[i], 0.0f), g[i]);
    }

    kernel void residual(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                       device const float *gate [[buffer(2)]], constant uint &cols [[buffer(3)]],
                       uint2 p [[thread_position_in_grid]]) {
        // Two roundings, as torch: no contraction of the product and the sum into one FMA.
        #pragma METAL fp contract(off)
        if (p.x >= cols) return;
        uint i = p.y * cols + p.x;
        float product = gate[p.x] * y[i];
        x[i] = x[i] + product;
    }

    // ── RMSNorm by rows, one threadgroup of 256 per row ─────────────────────────────────────────────────────────────────────
    // The sum of squares in plain fp32 (each lane its strided values, `simd_sum`, then the eight partials in order): not
    // `vDSP_svesq`'s order (`QKNormRope`), so not the CPU's bits — judged against the oracle. Then the CPU's roundings in its order.
    struct NormParams { uint cols; float eps; uint scaled; };

    static inline float row_scale(device const float *row, uint cols, float eps, uint t, uint lane, uint sg,
                                  threadgroup float *partial) {
        #pragma METAL fp contract(off)
        float ss = 0.0f;
        for (uint i = t; i < cols; i += 256) { float v = row[i]; float sq = v * v; ss = ss + sq; }
        float s = simd_sum(ss);
        if (lane == 0) partial[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint j = 0; j < 8; j++) total = total + partial[j];
        float mean = precise::divide(total, float(cols));
        return precise::divide(1.0f, precise::sqrt(mean + eps));
    }

    // out = ((x · r) · w) · scale — `Ops.rmsNorm` (`vsmul`, `vmul`) then `Ops.scaleRows`, each rounded.
    kernel void norm_scale(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                           device const float *scale [[buffer(2)]], device float *out [[buffer(3)]],
                           constant NormParams &p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                           uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float partial[8];
        device const float *source = x + ulong(row) * p.cols;
        device float *destination = out + ulong(row) * p.cols;
        float r = row_scale(source, p.cols, p.eps, t, lane, sg, partial);
        for (uint i = t; i < p.cols; i += 256) {
            float v = source[i] * r;
            v = v * w[i];
            if (p.scaled != 0) v = v * scale[i];
            destination[i] = v;
        }
    }

    // The same, and each value rounded to bf16 where the next product reads its operand (`WeightProducts.RoundedOperand`): a pair
    // of adjacent columns per thread and step, one 32-bit store (`cols` even), each value computed as above.
    kernel void norm_scale_rounded(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                                   device const float *scale [[buffer(2)]], device float *out [[buffer(3)]],
                                   device uint *rounded_out [[buffer(4)]],
                                   constant NormParams &p [[buffer(5)]], uint row [[threadgroup_position_in_grid]],
                                   uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float partial[8];
        device const float *source = x + ulong(row) * p.cols;
        device float *destination = out + ulong(row) * p.cols;
        device uint *pairs = rounded_out + ulong(row) * (p.cols / 2);
        float r = row_scale(source, p.cols, p.eps, t, lane, sg, partial);
        for (uint j = t; j < p.cols / 2; j += 256) {
            uint i = 2 * j;
            float v0 = source[i] * r, v1 = source[i + 1] * r;
            v0 = v0 * w[i]; v1 = v1 * w[i + 1];
            if (p.scaled != 0) { v0 = v0 * scale[i]; v1 = v1 * scale[i + 1]; }
            if (keeps_fp32) { destination[i] = v0; destination[i + 1] = v1; }
            pairs[j] = rounded(v0) | (rounded(v1) << 16);
        }
    }

    \(QuantizedKernels.rowSource)

    // The same, and the row quantized to int8 for W8A8 where the next products read their operand (`WeightProducts.RoundedOperand`):
    // the fp32 values computed as `norm_scale` computes them, each thread holding the columns `quantize_rows` would read into its
    // registers (chunk c of 256 in simdgroup c % 8, eight consecutive columns a lane), then `quantize_held` — `quantize_rows`'s own
    // code on the values it would have read back. One read of the row for the norm's sum, one for its values, one write of each form.
    // q: the rotation's stages, its norm's bits. Up to 8 · MAXC chunks of 256 (`ElementwiseGPU.quantizedColumns`).
    kernel void norm_scale_quantized(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                                     device const float *scale [[buffer(2)]], device float *out [[buffer(3)]],
                                     device char *codes [[buffer(4)]], device float *scales [[buffer(5)]],
                                     constant NormParams &p [[buffer(6)]], constant uint2 &q [[buffer(7)]],
                                     uint row [[threadgroup_position_in_grid]],
                                     uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float partial[8];
        threadgroup float part[32];
        device const float *source = x + ulong(row) * p.cols;
        device float *destination = out + ulong(row) * p.cols;
        float r = row_scale(source, p.cols, p.eps, t, lane, sg, partial);
        const uint chunks = (p.cols + 255) / 256;
        float v[MAXC][8];
        for (uint i = 0; i < MAXC; ++i) {
            const uint c = sg + i * 8;
            if (c < chunks) {
                const uint base = c * 256 + lane * 8;
                for (uint j = 0; j < 8; ++j) {
                    const uint col = base + j;
                    float y = 0.0f;
                    if (col < p.cols) {
                        y = source[col] * r;
                        y = y * w[col];
                        if (p.scaled != 0) y = y * scale[col];
                        destination[col] = y;
                    }
                    v[i][j] = y;
                }
            }
        }
        quantize_held(v, p.cols, q.x, as_type<float>(q.y), codes + ulong(row) * p.cols, scales + row, sg, 8, lane, part);
    }

    // `swiglu` quantized to int8 for W8A8 where the next product reads its operand (`WeightProducts.RoundedOperand`), and only
    // there: `quantize_rows`'s threadgroup (one a row, the same columns a lane), each value computed as `swiglu` computes it, in the
    // registers `quantize_rows` would read it into, then `quantize_held`. One read of each input, one write of the codes — neither
    // the fp32 row `swiglu` wrote nor the pass that read it back. p: the row's length, the rotation's stages, its norm's bits.
    kernel void swiglu_quantized(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                                 device char *codes [[buffer(2)]], device float *scales [[buffer(3)]],
                                 constant uint3 &p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint sgs [[simdgroups_per_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
        threadgroup float part[32];
        const uint K = p.x, chunks = (K + 255) / 256;
        device const float *ar = a + ulong(row) * K;
        device const float *br = b + ulong(row) * K;
        float v[MAXC][8];
        for (uint i = 0; i < MAXC; ++i) {
            const uint c = sg + i * sgs;
            if (c < chunks) {
                const uint base = c * 256 + lane * 8;
                if (K % 8 == 0) {                  // a lane's eight columns as two float4 each way: whole lines, not 32 strided floats
                    device const float4 *a4 = (device const float4 *)(ar + base);
                    device const float4 *b4 = (device const float4 *)(br + base);
                    for (uint h = 0; h < 2; ++h) {
                        const float4 x = a4[h], w = b4[h];
                        const float4 pr = x * w;
                        float4 y;
                        for (uint r = 0; r < 4; ++r) y[r] = over_one_plus_exp(float2(pr[r], fma(x[r], w[r], -pr[r])), x[r]);
                        for (uint r = 0; r < 4; ++r) v[i][4 * h + r] = y[r];
                    }
                } else {
                    for (uint r = 0; r < 8; ++r) {
                        const uint col = base + r;
                        float y = 0.0f;
                        if (col < K) {
                            const float x = ar[col], w = br[col];
                            const float pr = x * w;
                            y = over_one_plus_exp(float2(pr, fma(x, w, -pr)), x);
                        }
                        v[i][r] = y;
                    }
                }
            }
        }
        quantize_held(v, K, p.y, as_type<float>(p.z), codes + ulong(row) * K, scales + row, sg, sgs, lane, part);
    }

    // x += gate · ((y · r) · w) — `Ops.rmsNorm` then `Ops.gatedResidual`: the product, then the sum.
    kernel void norm_residual(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                              device const float *w [[buffer(2)]], device const float *gate [[buffer(3)]],
                              constant NormParams &p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                              uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float partial[8];
        device const float *source = y + ulong(row) * p.cols;
        device float *destination = x + ulong(row) * p.cols;
        float r = row_scale(source, p.cols, p.eps, t, lane, sg, partial);
        for (uint i = t; i < p.cols; i += 256) {
            float v = source[i] * r;
            v = v * w[i];
            float product = gate[i] * v;
            destination[i] = destination[i] + product;
        }
    }

    """

    private let queue: MTLCommandQueue
    private let swigluPipeline, carriesPipeline, residualPipeline, normScalePipeline, normResidualPipeline: MTLComputePipelineState
    /// Each `_rounded` kernel twice: the fp32 written beside the copy (`kept`), and the copy alone (`alone`).
    private let swigluRoundedPipeline, normScaleRoundedPipeline, carriesRoundedPipeline: (kept: MTLComputePipelineState,
                                                                                           alone: MTLComputePipelineState)
    private let normScaleQuantizedPipeline, swigluQuantizedPipeline: MTLComputePipelineState
    /// The longest row `norm_scale_quantized` holds: eight simdgroups of `QuantizedKernels.maxChunks` chunks of 256.
    package static let quantizedColumns = 8 * QuantizedKernels.maxChunks * 256

    package init(device: MTLDevice, queue: MTLCommandQueue) throws {
        self.queue = queue
        let options = MTLCompileOptions()
        options.mathMode = .safe          // no fast-math: `exp`, `fma` and the division as written
        let library = try device.makeLibrary(source: Self.source, options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw GEMM.Failure.noDevice }
            return try device.makeComputePipelineState(function: f)
        }
        func rounded(_ name: String) throws -> (kept: MTLComputePipelineState, alone: MTLComputePipelineState) {
            func built(_ keeps: Bool) throws -> MTLComputePipelineState {
                let values = MTLFunctionConstantValues()
                var value = keeps
                values.setConstantValue(&value, type: .bool, index: 0)
                return try device.makeComputePipelineState(function: try library.makeFunction(name: name, constantValues: values))
            }
            return (try built(true), try built(false))
        }
        swigluPipeline = try pipeline("swiglu"); carriesPipeline = try pipeline("sigmoid_gate")
        residualPipeline = try pipeline("residual")
        normScalePipeline = try pipeline("norm_scale"); normResidualPipeline = try pipeline("norm_residual")
        swigluRoundedPipeline = try rounded("swiglu_rounded"); normScaleRoundedPipeline = try rounded("norm_scale_rounded")
        carriesRoundedPipeline = try rounded("sigmoid_gate_rounded")
        normScaleQuantizedPipeline = try pipeline("norm_scale_quantized")
        swigluQuantizedPipeline = try pipeline("swiglu_quantized")
        let most = min(32, swigluQuantizedPipeline.maxTotalThreadsPerThreadgroup / 32)
        swigluQuantizedColumns = swigluQuantizedPipeline.threadExecutionWidth == 32 ? most * QuantizedKernels.maxChunks * 256 : 0
    }

    /// The longest row `swiglu_quantized` holds — `quantize_rows`'s threadgroup, up to 32 simdgroups as this device compiles it: what
    /// `WeightProducts.roundedOperand(…holding:)` is told, so that an operand its producer cannot quantize is quantized by its product.
    package let swigluQuantizedColumns: Int

    /// A buffer (at an offset in floats) — the form in which each kernel receives its operands.
    package typealias BufferSlice = (buffer: MTLBuffer, offset: Int)

    private func launch(_ pipeline: MTLComputePipelineState, _ slices: [BufferSlice],
                        constants: (MTLComputeCommandEncoder, Int) -> Void,
                        grid: MTLSize, group: MTLSize, byGroups: Bool = false) -> Double {
        guard case let commands = queue.checkedCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            // Nothing ran, and the output holds whatever was there: said and counted like a refusal.
            Attention.report("GPU elementwise (\(pipeline.label ?? "?")): no command buffer, so nothing ran")
            return 0
        }
        encoder.setComputePipelineState(pipeline)
        for (i, t) in slices.enumerated() { encoder.setBuffer(t.buffer, offset: t.offset * 4, index: i) }
        constants(encoder, slices.count)
        if byGroups {
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
        } else {
            encoder.dispatchThreads(grid, threadsPerThreadgroup: group)
        }
        encoder.endEncoding()
        let what = "GPU elementwise (\(pipeline.label ?? "?"))"
        return commands.commitSettled(label: "elementwise GPU", what: what)
    }

    private func linear(_ count: Int) -> (MTLSize, MTLSize) {
        (MTLSize(width: count, height: 1, depth: 1), MTLSize(width: 256, height: 1, depth: 1))
    }

    /// `a ← silu(a) · b` over `count` elements. Returns the GPU time. `rounded`: the result also written rounded to bf16 there,
    /// as the next product reads its operand (`WeightProducts.roundedOperand`; `count` even), and there alone where no reader reads
    /// the fp32 (`writesFP32`: `a` keeps its input) — or, an int8 operand its producer writes (`byProducer`), the result quantized
    /// there and nowhere else (`a` keeps its input).
    @discardableResult
    package func swiglu(_ a: BufferSlice, _ b: BufferSlice, count: Int, rounded: WeightProducts.RoundedOperand? = nil) -> Double {
        if let operand = rounded, let q = operand.int8, q.byProducer {
            return swiglu(a, b, rows: count / operand.columns, columns: operand.columns, quantized: q)
        }
        if let operand = rounded, let rounded = operand.bf16 {  // an int8 operand not the producer's is its first product's to quantize
            precondition(count % 2 == 0 && a.offset % 2 == 0 && b.offset % 2 == 0 && rounded.offset % 4 == 0)
            var n = UInt32(count / 2)
            let (g, t) = linear(count / 2)
            return launch(operand.writesFP32 ? swigluRoundedPipeline.kept : swigluRoundedPipeline.alone,
                          [a, b, (rounded.buffer, rounded.offset / 4)],
                          constants: { $0.setBytes(&n, length: 4, index: $1) }, grid: g, group: t)
        }
        var n = UInt32(count)
        let (g, t) = linear(count)
        return launch(swigluPipeline, [a, b], constants: { $0.setBytes(&n, length: 4, index: $1) },
                      grid: g, group: t)
    }

    /// `swiglu` over `rows` rows of `columns`, quantized to int8 into `q` (W8A8) — the codes and scales `quantize_rows` would write
    /// from it, bit for bit (`quantize_held`, the same code) — and written nowhere else: `a` keeps its input, as the product reads the
    /// codes alone (`roundedOperand` gives a producer's slot only to a product of one chunk, the GPU's alone, whose kernel takes the
    /// row). The fewest simdgroups a row (`maxChunks` chunks each): more rows in flight, the same codes (the row's maximum is exact in
    /// any order). `columns` ≤ `swigluQuantizedColumns`.
    @discardableResult
    package func swiglu(_ a: BufferSlice, _ b: BufferSlice, rows: Int, columns: Int, quantized q: WeightProducts.RoundedOperand.Quantized)
        -> Double {
        let chunks = (columns + 255) / 256
        let simdgroups = (chunks + QuantizedKernels.maxChunks - 1) / QuantizedKernels.maxChunks
        precondition(columns <= swigluQuantizedColumns && q.codesOffset % 4 == 0 && q.scalesOffset % 4 == 0)
        GEMM.producerQuantized.withLock { $0 += 1 }
        var p = SIMD3<UInt32>(UInt32(columns), UInt32(q.stages), q.norm.bitPattern)
        return launch(swigluQuantizedPipeline, [a, b, (q.buffer, q.codesOffset / 4), (q.buffer, q.scalesOffset / 4)],
                      constants: { $0.setBytes(&p, length: MemoryLayout<SIMD3<UInt32>>.stride, index: $1) },
                      grid: MTLSize(width: rows, height: 1, depth: 1), group: MTLSize(width: 32 * simdgroups, height: 1, depth: 1),
                      byGroups: true)
    }

    /// `x ← x · σ(g)`. `rounded`: as `swiglu`'s.
    @discardableResult
    package func carries(_ x: BufferSlice, _ g: BufferSlice, count: Int, rounded: WeightProducts.RoundedOperand? = nil) -> Double {
        if let operand = rounded, let rounded = operand.bf16 {  // an int8 operand is its first product's to quantize
            precondition(count % 2 == 0 && x.offset % 2 == 0 && g.offset % 2 == 0 && rounded.offset % 4 == 0)
            var n = UInt32(count / 2)
            let (grid, t) = linear(count / 2)
            return launch(operand.writesFP32 ? carriesRoundedPipeline.kept : carriesRoundedPipeline.alone,
                          [x, g, (rounded.buffer, rounded.offset / 4)],
                          constants: { $0.setBytes(&n, length: 4, index: $1) }, grid: grid, group: t)
        }
        var n = UInt32(count)
        let (grid, t) = linear(count)
        return launch(carriesPipeline, [x, g], constants: { $0.setBytes(&n, length: 4, index: $1) },
                      grid: grid, group: t)
    }

    /// `x[s, d] ← x[s, d] + gate[d] · y[s, d]` (`carries` holds the gate).
    @discardableResult
    package func residual(_ x: BufferSlice, plus y: BufferSlice, carries: BufferSlice, rows: Int, columns: Int) -> Double {
        var c = UInt32(columns)
        return launch(residualPipeline, [x, y, carries], constants: { $0.setBytes(&c, length: 4, index: $1) },
                      grid: MTLSize(width: columns, height: rows, depth: 1),
                      group: MTLSize(width: 256, height: 1, depth: 1))
    }

    private struct NormParams { var cols: UInt32; var eps: Float; var scaled: UInt32 }

    /// `out[s, :] ← (RMSNorm(x[s, :]) · w) · scale` (`scale` nil: the norm alone), one threadgroup per row. `rounded`: `out` also
    /// written rounded to bf16 there, as the next product reads its operand (`WeightProducts.roundedOperand`; `columns` even), and
    /// there alone where no reader reads the fp32 (`writesFP32`: `out` left as it was) — or
    /// quantized to int8 for W8A8, where the operand says the producer writes it (`byProducer`, `columns` ≤ `quantizedColumns`).
    @discardableResult
    package func normScale(_ x: BufferSlice, weight: BufferSlice, scale: BufferSlice?, into out: BufferSlice,
                           rows: Int, columns: Int, eps: Float, rounded operand: WeightProducts.RoundedOperand? = nil) -> Double {
        if let q = operand?.int8, q.byProducer {
            return normScale(x, weight: weight, scale: scale, into: out, rows: rows, columns: columns, eps: eps, quantized: q)
        }
        var p = NormParams(cols: UInt32(columns), eps: eps, scaled: scale == nil ? 0 : 1)
        if let operand, let rounded = operand.bf16 {
            precondition(columns % 2 == 0 && rounded.offset % 4 == 0)
            return launch(operand.writesFP32 ? normScaleRoundedPipeline.kept : normScaleRoundedPipeline.alone,
                          [x, weight, scale ?? weight, out, (rounded.buffer, rounded.offset / 4)],
                          constants: { $0.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: $1) },
                          grid: MTLSize(width: rows, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1), byGroups: true)
        }
        return launch(normScalePipeline, [x, weight, scale ?? weight, out],
                      constants: { $0.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: $1) },
                      grid: MTLSize(width: rows, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1), byGroups: true)
    }

    /// `normScale`, and `out` also quantized to int8 into `q` (W8A8) — its codes and scales `quantize_rows` would write from `out`,
    /// bit for bit (`quantize_held`, the same code). `columns` ≤ `quantizedColumns`.
    @discardableResult
    package func normScale(_ x: BufferSlice, weight: BufferSlice, scale: BufferSlice?, into out: BufferSlice,
                           rows: Int, columns: Int, eps: Float, quantized q: WeightProducts.RoundedOperand.Quantized) -> Double {
        precondition(columns <= Self.quantizedColumns && q.codesOffset % 4 == 0 && q.scalesOffset % 4 == 0)
        GEMM.producerQuantized.withLock { $0 += 1 }
        var p = NormParams(cols: UInt32(columns), eps: eps, scaled: scale == nil ? 0 : 1)
        var rotation = SIMD2<UInt32>(UInt32(q.stages), q.norm.bitPattern)
        return launch(normScaleQuantizedPipeline, [x, weight, scale ?? weight, out, (q.buffer, q.codesOffset / 4),
                                                   (q.buffer, q.scalesOffset / 4)],
                      constants: { encoder, index in
                          encoder.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: index)
                          encoder.setBytes(&rotation, length: MemoryLayout<SIMD2<UInt32>>.stride, index: index + 1)
                      },
                      grid: MTLSize(width: rows, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1), byGroups: true)
    }

    /// `x[s, :] ← x[s, :] + gate · (RMSNorm(y[s, :]) · w)`, one threadgroup per row.
    @discardableResult
    package func normResidual(_ x: BufferSlice, plus y: BufferSlice, weight: BufferSlice, gate: BufferSlice,
                              rows: Int, columns: Int, eps: Float) -> Double {
        var p = NormParams(cols: UInt32(columns), eps: eps, scaled: 1)
        return launch(normResidualPipeline, [x, y, weight, gate],
                      constants: { $0.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: $1) },
                      grid: MTLSize(width: rows, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1), byGroups: true)
    }
}
