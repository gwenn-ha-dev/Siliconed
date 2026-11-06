import Foundation
import Metal

/// **W8A8: the published int8 weight times the activation quantized to int8** (`gpu/int8x8`) — the forward of the quant's own author,
/// ComfyUI's `int8_linear` (comfy_kitchen `backends/eager/quantization.py`, frozen in `Tests/SiliconedTests/Fixtures/quant/comfy/`), on the
/// GPU's tensor units at the integers' rate. Compiled only when asked, apart from the default library.
///
///     x fp32 [rows, k] ─▶ quantize_rows  per group of G along k: y = x · H_G/√G   (convrot moved to the activation, as `rotate_groups`)
///          │     (`quantize_held`)       s_x = max(max|y| / 127, 1e-30)           (per row, the whole row: Comfy's `quantize_int8_rowwise`)
///          │                             q_x = clamp(rint(y / s_x), −128, 127)
///          │           ─▶ codes int8 [rows, k], s_x fp32 [rows]                   (the reserve's tail — or all of it, the weight read in
///          │                                                                       place in the map —, a chunk of rows at a time)
///          └─ or written by x's producer: a norm that holds the row calls `quantize_held` on the values it writes
///             (`WeightProducts.RoundedOperand.int8`), or the first product on x — the next products skip the pass (`prequantized`)
///     codes × q_w      matmul2d  int8_t × int8_t → int32                        (exact: |Σ| < 2³¹ for any k under 2¹⁷)
///     C int32 ─▶ scale_products  C = float(acc) · (s_x[row] · s_w[col])         (in place, the int32 read where the fp32 is written)
///
/// What sets it apart — what a port by analogy with the rounded route (`BFloatKernels`) would get wrong:
///
///   - **the activation is quantized per row, over the whole row**: one scale for k values, read only once the row's largest value is
///     known — a pass that cannot be cut along k. One threadgroup per row holds it whole, eight values per lane in registers (`maxChunks`
///     chunks of 256 per simdgroup), rotated by `simd_shuffle_xor` across lanes: one read of the row, one write of its codes;
///   - **the rotation is `rotate_groups`'s, operation for operation**: the regular Hadamard's radix-4 butterflies (H4 = 1 except −1 on its
///     anti-diagonal, Kronecker powers; Comfy's `_build_hadamard`), in the same order on the same operands — bits 0–1 of the column within
///     its 256 in a lane's registers, bit 2 there and bit 3 across a pair of lanes, bits 4–7 across lanes. So the rotated row is the one
///     the int8 route's pass writes, bit for bit, and its codes the CPU's (`W8A8.quantize`), which the probe and the tests read;
///   - **Comfy's recipe to the operation**: the scale divides (`x / scale`, correctly rounded: `precise::divide`), never a reciprocal's
///     product; `rint` is torch's `round` (half to even); the clamp is [−128, 127] though only ±127 is reached; the output scale is
///     `s_x · s_w` first, then times `float(acc)` (`chunk * (x_scale * weight_scale)`). Comfy rotates and divides in the layer's compute
///     dtype (bf16 in its renders): here in fp32, the activation as the engine holds it — nothing rounded that the recipe does not round;
///   - **the integer sum is exact, so the tile is bits-free**: every candidate tile (`MatmulTile`, `gpu/int8x8`) writes the same int32,
///     and the chunks of rows are independent (a row's scale is its own): the route's bits depend on neither;
///   - **no AMX, no ANE beside it** (the AMX reads the fp32 activation; the ANE's int8 × int8 is another lot): the GPU's whole product;
///   - **the quantization is one Metal function, wherever it runs** (`rowSource`'s `quantize_held`): `quantize_rows` and the norms that
///     quantize as they write (`norm_scale_quantized` and each family's) hold the row in the same registers and call the same code, so
///     an operand quantized by its producer is the pass's, bit for bit, though the norm runs 8 simdgroups and the pass up to 32 (the
///     row's maximum is exact in any order). A pass per operand, not per product: q, k and v read one, w1 and w3 one.
package final class QuantizedKernels {
    let quantize, product, scale: MTLComputePipelineState
    /// Chunks of 256 columns a simdgroup holds in registers (`maxChunks` × 8 floats a lane): with 32 simdgroups, a row of 32 768.
    static let maxChunks = 4

    /// **The rotation and the per-row quantization of a row held in registers**, as one Metal function both `quantize_rows` and a
    /// producer that holds the whole row call (`ElementwiseGPU`'s `norm_scale_quantized`, `WeightProducts.RoundedOperand`): the same
    /// code, hence the same codes and scale from the same fp32 row, whatever the threadgroup's count of simdgroups (the row's
    /// maximum is exact in any order). Interpolated into each library that quantizes; `MAXC` is `maxChunks`.
    static let rowSource = """
    #define MAXC \(maxChunks)

    // `rotate_groups`'s butterfly: the four outputs of one radix-4 stage, its operations in its order.
    inline float4 h4(float a, float b, float c, float d) {
        const float p = a + b, m = a - b, u = c + d, w = c - d;
        return float4(p + w, p - w, m + u, u - m);
    }

    // A stage on two bits of the lane index (`unit` = the lower one's value): the value at digit t lies `(digit ^ t) · unit` lanes away.
    inline float acrossLanes(float v, uint lane, ushort unit) {
        const float x1 = simd_shuffle_xor(v, unit), x2 = simd_shuffle_xor(v, ushort(2 * unit)), x3 = simd_shuffle_xor(v, ushort(3 * unit));
        const uint d = (lane / unit) & 3;
        const float a = d == 0 ? v : d == 1 ? x1 : d == 2 ? x2 : x3;
        const float b = d == 0 ? x1 : d == 1 ? v : d == 2 ? x3 : x2;
        const float c = d == 0 ? x2 : d == 1 ? x3 : d == 2 ? v : x1;
        const float e = d == 0 ? x3 : d == 1 ? x2 : d == 2 ? x1 : v;
        const float4 o = h4(a, b, c, e);
        return d == 0 ? o.x : d == 1 ? o.y : d == 2 ? o.z : o.w;
    }

    // Column `lane · 8 + r` of a chunk of 256: bits 0–2 are `r`, bits 3–7 the lane. `stages` of the four (log₄ G).
    inline void rotate(thread float *v, uint lane, uint stages) {
        if (stages > 0) {
            for (uint base = 0; base < 8; base += 4) {
                const float4 o = h4(v[base], v[base + 1], v[base + 2], v[base + 3]);
                v[base] = o.x; v[base + 1] = o.y; v[base + 2] = o.z; v[base + 3] = o.w;
            }
        }
        if (stages > 1) {
            // Bit 2 is the register's, bit 3 the lane pair's.
            const bool high = (lane & 1) != 0;
            for (uint r = 0; r < 4; ++r) {
                const float own0 = v[r], own1 = v[r + 4];
                const float other0 = simd_shuffle_xor(own0, ushort(1)), other1 = simd_shuffle_xor(own1, ushort(1));
                const float4 o = high ? h4(other0, other1, own0, own1) : h4(own0, own1, other0, other1);
                v[r] = high ? o.z : o.x;
                v[r + 4] = high ? o.w : o.y;
            }
        }
        if (stages > 2) { for (uint r = 0; r < 8; ++r) v[r] = acrossLanes(v[r], lane, 2); }
        if (stages > 3) { for (uint r = 0; r < 8; ++r) v[r] = acrossLanes(v[r], lane, 8); }
    }

    // A row of K held as `quantize_rows` holds it — chunk c of 256 columns in simdgroup c % sgs, its columns lane · 8 … lane · 8 + 7
    // in v[c / sgs], zero past K — rotated by `stages`, times `norm`, then quantized over the whole row: its codes to `qr`, its scale
    // to `*s`. Every simdgroup of the threadgroup calls it (a barrier); `part`: 32 floats of threadgroup memory nothing else uses.
    inline void quantize_held(thread float (*v)[8], uint K, uint stages, float norm, device char *qr, device float *s,
                              uint sg, uint sgs, uint lane, threadgroup float *part) {
        const uint chunks = (K + 255) / 256;
        float m = 0;
        for (uint i = 0; i < MAXC; ++i) {
            const uint c = sg + i * sgs;
            if (c < chunks) {          // the same for every lane of the simdgroup: the shuffles see all of them
                rotate(v[i], lane, stages);
                for (uint r = 0; r < 8; ++r) { v[i][r] *= norm; m = max(m, fabs(v[i][r])); }
            }
        }
        m = simd_max(m);
        if (lane == 0) part[sg] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        m = part[0];
        for (uint i = 1; i < sgs; ++i) m = max(m, part[i]);
        const float scale = max(precise::divide(m, 127.0f), 1e-30f);
        for (uint i = 0; i < MAXC; ++i) {
            const uint c = sg + i * sgs;
            if (c < chunks) {
                const uint base = c * 256 + lane * 8;
                for (uint r = 0; r < 8; ++r) {
                    if (base + r < K) qr[base + r] = char(clamp(rint(precise::divide(v[i][r], scale)), -128.0f, 127.0f));
                }
            }
        }
        if (sg == 0 && lane == 0) *s = scale;
    }
    """

    static let source = """
    #include <metal_stdlib>
    #include <metal_tensor>
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    using namespace metal;
    using namespace mpp::tensor_ops;

    \(rowSource)

    // x fp32 [rows, K] → codes int8 [rows, K], scales fp32 [rows]. One threadgroup a row; chunk c of 256 columns to simdgroup c % sgs.
    // p: K, the rotation's stages, its norm's bits.
    kernel void quantize_rows(device const float *x [[buffer(0)]], device char *q [[buffer(1)]], device float *s [[buffer(2)]],
                              constant uint3 &p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint sgs [[simdgroups_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
        threadgroup float part[32];
        const uint K = p.x, chunks = (K + 255) / 256;
        device const float *xr = x + size_t(row) * K;
        float v[MAXC][8];
        for (uint i = 0; i < MAXC; ++i) {
            const uint c = sg + i * sgs;
            if (c < chunks) {
                const uint base = c * 256 + lane * 8;
                for (uint r = 0; r < 8; ++r) v[i][r] = base + r < K ? xr[base + r] : 0.0f;
            }
        }
        quantize_held(v, K, p.y, as_type<float>(p.z), q + size_t(row) * K, s + row, sg, sgs, lane, part);
    }

    // C [rows, N] int32 → fp32 in place: float(acc) · (s_x[row] · s_w[col · stride]). p: N, the weight scale's stride (1 per column,
    // 0 for the tensor), the values.
    kernel void scale_products(device int *c [[buffer(0)]], device const float *sx [[buffer(1)]], device const float *sw [[buffer(2)]],
                               constant uint3 &p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
        if (i >= p.z) return;
        const uint r = i / p.x, j = i % p.x;
        c[i] = as_type<int>(float(c[i]) * (sx[r] * sw[j * p.y]));
    }

    \(MatmulTile(64, 64, simdgroups: 4).integerProduct)
    """

    /// `nil` when this macOS's compiler refuses the library or a kernel: the integers are then read on the rounded or fp32 activation.
    init?(device: MTLDevice, options: MTLCompileOptions) {
        guard let library = MetalCompile.library(device, Self.source, options, what: "W8A8 kernels") else { return nil }
        func pipeline(_ name: String) -> MTLComputePipelineState? {
            library.makeFunction(name: name).flatMap { MetalCompile.pipeline(device, $0, what: "W8A8 \(name)") }
        }
        guard let quantize = pipeline("quantize_rows"), let product = pipeline("linear_i8i8"), let scale = pipeline("scale_products"),
              quantize.maxTotalThreadsPerThreadgroup >= quantize.threadExecutionWidth else { return nil }
        self.quantize = quantize; self.product = product; self.scale = scale
    }

    /// The simdgroups `quantize_rows` runs a row of `k` on — one per chunk of 256 up to 32, never fewer than its registers need —, or
    /// `nil` when the row is longer than they hold.
    func simdgroups(k: Int) -> Int? {
        let width = quantize.threadExecutionWidth
        let most = min(32, quantize.maxTotalThreadsPerThreadgroup / width), chunks = (k + 255) / 256
        let least = (chunks + Self.maxChunks - 1) / Self.maxChunks
        guard width == 32, least <= most else { return nil }
        return max(least, min(most, chunks))
    }
}

/// **Comfy's per-row quantization of the activation, on the CPU** — what `quantize_rows` writes, bit for bit: the same butterflies in fp32
/// in `rotate_groups`'s order, the same correctly rounded division, half to even. The probe's integer truth reads it
/// (`Diagnostic.exactQuantizedRows`), and the tests.
package enum W8A8 {
    /// log₄ of a rotation group the kernel takes (4, 16, 64, 256; none: 0), `nil` for another.
    package static func stages(group: Int?) -> Int? {
        guard let group else { return 0 }
        switch group { case 4: return 1; case 16: return 2; case 64: return 3; case 256: return 4; default: return nil }
    }

    /// `x` (`k` values) rotated by groups of `group` (nil: none), then quantized: the codes and the row's scale.
    package static func quantize(_ x: UnsafePointer<Float>, k: Int, group: Int?) -> (codes: [Int8], scale: Float) {
        var y = Array(UnsafeBufferPointer(start: x, count: k))
        if let group {
            let norm = 1 / Float(group).squareRoot()
            for g0 in stride(from: 0, to: k, by: group) {
                var span = 1
                while span < group {
                    for base in stride(from: g0, to: g0 + group, by: 4 * span) {
                        for j in base..<(base + span) {
                            let (a, b, c, d) = (y[j], y[j + span], y[j + 2 * span], y[j + 3 * span])
                            let (p, m, u, w) = (a + b, a - b, c + d, c - d)
                            (y[j], y[j + span], y[j + 2 * span], y[j + 3 * span]) = (p + w, p - w, m + u, u - m)
                        }
                    }
                    span *= 4
                }
                for j in g0..<(g0 + group) { y[j] *= norm }
            }
        }
        let top = y.reduce(Float(0)) { max($0, abs($1)) }
        let scale = max(top / 127, 1e-30)
        let codes = y.map { Int8(max(-128, min(127, ($0 / scale).rounded(.toNearestOrEven)))) }
        return (codes, scale)
    }
}

extension TensorGEMM {
    /// Where an activation is quantized to: `rows` rows at a time, their codes at `codesOffset` (int8 `[rows, k]`) and their scales at
    /// `scalesOffset` (fp32 `[rows]`), in `buffer`. Offsets in bytes, 16-aligned.
    package typealias Quantizing = (buffer: MTLBuffer, codesOffset: Int, scalesOffset: Int, rows: Int)

    /// **`submit` on an activation quantized to int8** (W8A8): per chunk of `quantizing.rows` rows, the rotation and quantization of those
    /// rows into `quantizing`, their int8 × int8 product into `C`'s rows as int32, and the scales' pass that makes them fp32. Committed and
    /// not awaited; the whole product (no AMX columns). `nil` when the library is not compiled, the rotation's group is not one the
    /// kernel takes, or the row is longer than it holds: the caller reads the integers another way. `weightOffset`: where the integers
    /// start in `w` — 0 in the reserve, the tensor's offset in the wrapped map when read in place (16-aligned).
    ///
    /// `prequantized`: `quantizing` already holds the `m` rows' codes and scales, the ones this pass would write (by the operand's
    /// producer or the product before, `WeightProducts.RoundedOperand`): the products alone, in one chunk.
    func submitQuantized(a: MTLBuffer, int8 w: MTLBuffer, weightOffset: Int = 0, scales: MTLBuffer, scalesOffset: Int, perColumn: Bool,
                         group: Int?, quantizing: Quantizing, prequantized: Bool = false, c: MTLBuffer, m: Int, k: Int, n: Int)
        -> MTLCommandBuffer? {
        guard let kernels = quantized, let stages = W8A8.stages(group: group), let simdgroups = kernels.simdgroups(k: k) else { return nil }
        precondition(!prequantized || quantizing.rows >= m, "an operand quantized ahead holds all the rows")
        let kernel = CapacityModel.Kernel(.gpu, .int8x8)
        let tiled = tiles.products(kernel, rows: m, k: k, n: n)
        let tile = tiled?.tile ?? MatmulTile.standard(for: kernel)
        let pipeline = tiled?.int8 ?? kernels.product
        let commands = makeCommands()
        let encoder = commands.checkedEncoder()
        var norm = SIMD3<UInt32>(UInt32(k), UInt32(stages), (group.map { 1 / Float($0).squareRoot() } ?? 1).bitPattern)
        for first in stride(from: 0, to: m, by: quantizing.rows) {
            let count = min(quantizing.rows, m - first)
            if !prequantized {
                GEMM.quantizePasses.withLock { $0 += 1 }
                encoder.setComputePipelineState(kernels.quantize)
                encoder.setBuffer(a, offset: first * k * 4, index: 0)
                encoder.setBuffer(quantizing.buffer, offset: quantizing.codesOffset, index: 1)
                encoder.setBuffer(quantizing.buffer, offset: quantizing.scalesOffset, index: 2)
                encoder.setBytes(&norm, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
                encoder.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 32 * simdgroups, height: 1, depth: 1))
            }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(quantizing.buffer, offset: quantizing.codesOffset, index: 0)
            encoder.setBuffer(w, offset: weightOffset, index: 1)
            encoder.setBuffer(c, offset: first * n * 4, index: 2)
            var mkn = SIMD3<UInt32>(UInt32(count), UInt32(k), UInt32(n))
            encoder.setBytes(&mkn, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
            tile.dispatch(encoder, pipeline, rows: count, columns: n)
            encoder.setComputePipelineState(kernels.scale)
            encoder.setBuffer(c, offset: first * n * 4, index: 0)
            encoder.setBuffer(quantizing.buffer, offset: quantizing.scalesOffset, index: 1)
            encoder.setBuffer(scales, offset: scalesOffset, index: 2)
            var nStrideCount = SIMD3<UInt32>(UInt32(n), perColumn ? 1 : 0, UInt32(count * n))
            encoder.setBytes(&nStrideCount, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
            encoder.dispatchThreads(MTLSize(width: count * n, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        encoder.endEncoding()
        commands.commit()
        return commands
    }
}
