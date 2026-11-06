import Foundation
import Metal

/// **Anima's modulated LayerNorm and gated residual, on the GPU** — the stream's elementwise kernel for `AnimaDiT` (`SILICONED_STREAM`),
/// beside the CPU's `AnimaOps.modulate` and `Ops.gatedResidual`, which the block runs without the stream.
///
///     per row, one threadgroup of 256 (eight simdgroups):
///       x     = x + gate · y                          (residual)   the product, then the sum: two roundings, as torch and the CPU
///       μ     = Σ x / D                                (modulate)   fp32, each lane its strided values, `simd_sum`, the eight partials in order
///       c     = x − μ ;  σ² = Σ c² / D
///       out   = ((c · 1/√(σ² + ε)) · (1 + scale)) + shift          each operation rounded, in the CPU's order; `1 + scale` one rounding
///
/// One kernel, either half or both: in a block every gated residual is followed by the next sublayer's modulation of the same `x`, so
/// the two run on the row while it is in cache (one launch, one read of `x` instead of two); the first sublayer of a block modulates
/// alone, the last residual of the last block adds alone.
///
/// What sets it apart from Z-Image's `ElementwiseGPU.normScale` / `normResidual` — what a port by analogy would get wrong:
///
///   - **a LayerNorm, not an RMSNorm**: the mean is subtracted first, and the variance is that of the centred row; **no affine weight**
///     (`elementwise_affine=False`), eps **1e-6**;
///   - the modulation is `· (1 + scale) + shift`, a **shift** and the scale's `1 +` — Z-Image multiplies by its scale alone;
///   - the gate multiplies the sublayer's raw output (no norm on it, unlike Z-Image's `norm_residual`), **no tanh**;
///   - the residual comes **before** the norm in the fused kernel: Z-Image fuses the norm of the output into the residual, here the
///     residual is fused into the next norm of `x`.
///
/// **Not the CPU's bits, and why**: the residual is (the same product and sum, no contraction), the norm's sums are not — `vDSP_meanv`'s
/// and `vDSP_svesq`'s order is their own (`QKNormRope`: searched, not found), so the mean and the scale move by an ulp and every value
/// of the row with them. The route is judged against the oracle (`siliconed check anima dit` under `SILICONED_STREAM`).
package final class AnimaElementwiseGPU {
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params { uint cols; float eps; uint residual; uint modulate; };

    // The row's sum over 256 threads: each its strided values, `simd_sum`, then the eight partials in order.
    static inline float row_sum(float own, uint lane, uint sg, threadgroup float *partial) {
        #pragma METAL fp contract(off)
        float s = simd_sum(own);
        if (lane == 0) partial[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint j = 0; j < 8; j++) total = total + partial[j];
        return total;
    }

    kernel void residual_modulate(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                                  device const float *gate [[buffer(2)]], device const float *shift [[buffer(3)]],
                                  device const float *scale [[buffer(4)]], device float *out [[buffer(5)]],
                                  constant Params &p [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                                  uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
        // The products and the sums as written, each rounded: `.safe` alone does not forbid the contraction into an FMA.
        #pragma METAL fp contract(off)
        threadgroup float partialMean[8], partialSquares[8];
        device float *source = x + ulong(row) * p.cols;
        device const float *added = y + ulong(row) * p.cols;
        float sum = 0.0f;
        for (uint i = t; i < p.cols; i += 256) {
            float v = source[i];
            if (p.residual != 0) {
                float product = gate[i] * added[i];
                v = v + product;
                source[i] = v;              // read back below by this thread alone
            }
            sum = sum + v;
        }
        if (p.modulate == 0) return;        // uniform over the threadgroup: no barrier skipped by some threads only
        float mean = precise::divide(row_sum(sum, lane, sg, partialMean), float(p.cols));
        float squares = 0.0f;
        for (uint i = t; i < p.cols; i += 256) { float c = source[i] - mean; float sq = c * c; squares = squares + sq; }
        float variance = precise::divide(row_sum(squares, lane, sg, partialSquares), float(p.cols));
        float inverse = precise::divide(1.0f, precise::sqrt(variance + p.eps));
        device float *destination = out + ulong(row) * p.cols;
        for (uint i = t; i < p.cols; i += 256) {
            float c = source[i] - mean;
            c = c * inverse;
            float factor = 1.0f + scale[i];
            c = c * factor;
            destination[i] = c + shift[i];
        }
    }

    \(BFloatKernels.roundingSource)

    // The same, and each modulated value rounded to bf16 where the next products read their operand (`WeightProducts.RoundedOperand`):
    // one 16-bit store per value, in the same loops (a thread reads back only what it wrote).
    kernel void residual_modulate_rounded(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                                          device const float *gate [[buffer(2)]], device const float *shift [[buffer(3)]],
                                          device const float *scale [[buffer(4)]], device float *out [[buffer(5)]],
                                          device ushort *rounded_out [[buffer(6)]],
                                          constant Params &p [[buffer(7)]], uint row [[threadgroup_position_in_grid]],
                                          uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                          uint sg [[simdgroup_index_in_threadgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float partialMean[8], partialSquares[8];
        device float *source = x + ulong(row) * p.cols;
        device const float *added = y + ulong(row) * p.cols;
        float sum = 0.0f;
        for (uint i = t; i < p.cols; i += 256) {
            float v = source[i];
            if (p.residual != 0) {
                float product = gate[i] * added[i];
                v = v + product;
                source[i] = v;
            }
            sum = sum + v;
        }
        float mean = precise::divide(row_sum(sum, lane, sg, partialMean), float(p.cols));
        float squares = 0.0f;
        for (uint i = t; i < p.cols; i += 256) { float c = source[i] - mean; float sq = c * c; squares = squares + sq; }
        float variance = precise::divide(row_sum(squares, lane, sg, partialSquares), float(p.cols));
        float inverse = precise::divide(1.0f, precise::sqrt(variance + p.eps));
        device float *destination = out + ulong(row) * p.cols;
        device ushort *bf = rounded_out + ulong(row) * p.cols;
        for (uint i = t; i < p.cols; i += 256) {
            float c = source[i] - mean;
            c = c * inverse;
            float factor = 1.0f + scale[i];
            c = c * factor;
            float v = c + shift[i];
            destination[i] = v;
            bf[i] = ushort(rounded(v));
        }
    }

    \(QuantizedKernels.rowSource)

    // The same, and the row quantized to int8 for W8A8 where the next products read their operand (`WeightProducts.RoundedOperand`):
    // each value computed as above, by the thread that holds its column as `quantize_rows` would read it (chunk c of 256 in
    // simdgroup c % 8, eight consecutive columns a lane), then `quantize_held` — `quantize_rows`'s own code. q: the rotation's
    // stages, its norm's bits. Up to `ElementwiseGPU.quantizedColumns`.
    // The residual's sums are written by each thread at its strided columns and read back here by the thread that holds the column:
    // a barrier on the device's memory between.
    kernel void residual_modulate_quantized(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                                            device const float *gate [[buffer(2)]], device const float *shift [[buffer(3)]],
                                            device const float *scale [[buffer(4)]], device float *out [[buffer(5)]],
                                            constant Params &p [[buffer(6)]],
                                            device char *codes [[buffer(7)]], device float *scales [[buffer(8)]],
                                            constant uint2 &q [[buffer(9)]], uint row [[threadgroup_position_in_grid]],
                                            uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                            uint sg [[simdgroup_index_in_threadgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float partialMean[8], partialSquares[8], part[32];
        device float *source = x + ulong(row) * p.cols;
        device const float *added = y + ulong(row) * p.cols;
        float sum = 0.0f;
        for (uint i = t; i < p.cols; i += 256) {
            float v = source[i];
            if (p.residual != 0) {
                float product = gate[i] * added[i];
                v = v + product;
                source[i] = v;
            }
            sum = sum + v;
        }
        float mean = precise::divide(row_sum(sum, lane, sg, partialMean), float(p.cols));
        float squares = 0.0f;
        for (uint i = t; i < p.cols; i += 256) { float c = source[i] - mean; float sq = c * c; squares = squares + sq; }
        float variance = precise::divide(row_sum(squares, lane, sg, partialSquares), float(p.cols));
        float inverse = precise::divide(1.0f, precise::sqrt(variance + p.eps));
        device float *destination = out + ulong(row) * p.cols;
        threadgroup_barrier(mem_flags::mem_device);
        const uint chunks = (p.cols + 255) / 256;
        float held[MAXC][8];
        for (uint i = 0; i < MAXC; ++i) {
            const uint c = sg + i * 8;
            if (c < chunks) {
                for (uint j = 0; j < 8; ++j) {
                    const uint col = c * 256 + lane * 8 + j;
                    float y = 0.0f;
                    if (col < p.cols) {
                        float c = source[col] - mean;
                        c = c * inverse;
                        float factor = 1.0f + scale[col];
                        c = c * factor;
                        y = c + shift[col];
                        destination[col] = y;
                    }
                    held[i][j] = y;
                }
            }
        }
        quantize_held(held, p.cols, q.x, as_type<float>(q.y), codes + ulong(row) * p.cols, scales + row, sg, 8, lane, part);
    }
    """

    private let queue: MTLCommandQueue
    private let pipeline, roundedPipeline, quantizedPipeline: MTLComputePipelineState

    package init(device: MTLDevice, queue: MTLCommandQueue) throws {
        self.queue = queue
        let options = MTLCompileOptions()
        options.mathMode = .safe          // no fast-math: the division and the square root as written
        let library = try device.makeLibrary(source: Self.source, options: options)
        guard let f = library.makeFunction(name: "residual_modulate"), let g = library.makeFunction(name: "residual_modulate_rounded"),
              let h = library.makeFunction(name: "residual_modulate_quantized") else {
            throw GEMM.Failure.noDevice
        }
        pipeline = try device.makeComputePipelineState(function: f)
        roundedPipeline = try device.makeComputePipelineState(function: g)
        quantizedPipeline = try device.makeComputePipelineState(function: h)
        guard pipeline.maxTotalThreadsPerThreadgroup >= 256, roundedPipeline.maxTotalThreadsPerThreadgroup >= 256,
              quantizedPipeline.maxTotalThreadsPerThreadgroup >= 256 else {
            throw GEMM.Failure.noDevice
        }
    }

    /// A buffer at an offset in floats (`ElementwiseGPU.BufferSlice`).
    package typealias Slice = ElementwiseGPU.BufferSlice

    private struct Params { var cols: UInt32; var eps: Float; var residual: UInt32; var modulate: UInt32 }

    /// **`x[s, :] += gate · y[s, :]`** where `residual` is given, then **`out[s, :] = LN(x[s, :]) · (1 + scale) + shift`** where
    /// `modulate` is given — one threadgroup per row, one command buffer, settled on the queue's stream (`commitSettled`). Returns the
    /// GPU's seconds off the stream, 0 on it.
    @discardableResult
    /// `rounded` (with `modulate`): `out` also written rounded to bf16 there, as the next products read their operand
    /// (`WeightProducts.roundedOperand`) — or quantized to int8 for W8A8, where the operand says the producer writes it (`byProducer`).
    package func run(_ x: Slice, residual: (y: Slice, gate: Slice)?, modulate: (shift: Slice, scale: Slice, out: Slice)?,
                     rows: Int, columns: Int, eps: Float, rounded operand: WeightProducts.RoundedOperand? = nil) -> Double {
        guard rows > 0, residual != nil || modulate != nil else { return 0 }
        let commands = queue.checkedCommandBuffer()
        let encoder = commands.checkedEncoder()
        let rounded = modulate == nil ? nil : operand?.bf16
        let quantized = modulate == nil ? nil : operand?.int8.flatMap { $0.byProducer ? $0 : nil }
        encoder.setComputePipelineState(quantized != nil ? quantizedPipeline : rounded == nil ? pipeline : roundedPipeline)
        // Unused operands are bound to `x`, never read.
        var slices = [x, residual?.y ?? x, residual?.gate ?? x, modulate?.shift ?? x, modulate?.scale ?? x, modulate?.out ?? x]
        if let quantized { precondition(columns <= ElementwiseGPU.quantizedColumns); quantized.bind(encoder, at: slices.count + 1) }
        else if let rounded {
            precondition(rounded.offset % 4 == 0)
            slices.append((rounded.buffer, rounded.offset / 4))
        }
        for (i, s) in slices.enumerated() { encoder.setBuffer(s.buffer, offset: s.offset * 4, index: i) }
        var p = Params(cols: UInt32(columns), eps: eps, residual: residual == nil ? 0 : 1, modulate: modulate == nil ? 0 : 1)
        encoder.setBytes(&p, length: MemoryLayout<Params>.stride, index: slices.count)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        return commands.commitSettled(label: "elementwise GPU", what: "Anima residual + modulation (\(rows)×\(columns))")
    }
}
