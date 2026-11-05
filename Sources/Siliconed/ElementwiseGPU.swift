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

    package init(device: MTLDevice, queue: MTLCommandQueue) throws {
        self.queue = queue
        let options = MTLCompileOptions()
        options.mathMode = .safe          // no fast-math: `exp`, `fma` and the division as written
        let library = try device.makeLibrary(source: Self.source, options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw GEMM.Failure.noDevice }
            return try device.makeComputePipelineState(function: f)
        }
        swigluPipeline = try pipeline("swiglu"); carriesPipeline = try pipeline("sigmoid_gate")
        residualPipeline = try pipeline("residual")
        normScalePipeline = try pipeline("norm_scale"); normResidualPipeline = try pipeline("norm_residual")
    }

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

    /// `a ← silu(a) · b` over `count` elements. Returns the GPU time.
    @discardableResult
    package func swiglu(_ a: BufferSlice, _ b: BufferSlice, count: Int) -> Double {
        var n = UInt32(count)
        let (g, t) = linear(count)
        return launch(swigluPipeline, [a, b], constants: { $0.setBytes(&n, length: 4, index: $1) },
                      grid: g, group: t)
    }

    /// `x ← x · σ(g)`.
    @discardableResult
    package func carries(_ x: BufferSlice, _ g: BufferSlice, count: Int) -> Double {
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

    /// `out[s, :] ← (RMSNorm(x[s, :]) · w) · scale` (`scale` nil: the norm alone), one threadgroup per row.
    @discardableResult
    package func normScale(_ x: BufferSlice, weight: BufferSlice, scale: BufferSlice?, into out: BufferSlice,
                           rows: Int, columns: Int, eps: Float) -> Double {
        var p = NormParams(cols: UInt32(columns), eps: eps, scaled: scale == nil ? 0 : 1)
        return launch(normScalePipeline, [x, weight, scale ?? weight, out],
                      constants: { $0.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: $1) },
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
