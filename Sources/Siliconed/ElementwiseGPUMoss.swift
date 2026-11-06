import Foundation
import Metal

/// **MOSS-Image's modulated norm on the GPU** — the block's two pre-norms where the denoiser streams (`GPUStream`,
/// `MossImageDiT.blockStreamed`), in place of `Ops.rmsNorm` then `Ops.modulate` on the CPU, the GPU idle between them:
///
///     r    = 1 / √(Σ x² / d + ε)                  fp32, the sum in the GPU's order (below)
///     out  = ((x · r) · w) · (1 + scale) + shift   each product and the sum rounded, in this order — no contraction
///
/// What sets it apart from Z-Image's `ElementwiseGPU.normScale`, and what a port by analogy would miss:
///
///   - **a shift**: MOSS's adaLN is `x·(1 + scale) + shift` (`Ops.modulate`: `factor = scale + 1` in fp32, then `x·factor + shift`
///     in two roundings, as torch), where Z's is a scale alone; the `1 +` is one fp32 sum, commutative, so the CPU's factor's bits;
///   - **the shared modulation**: `scale` and `shift` are slices of the one modulation every block reads (`shift_a, scale_a, gate_a,
///     shift_f, scale_f, gate_f`, the shift first), not a per-block table;
///   - **no norm on the residual**: MOSS's gated residual adds the sublayer as it is (`ElementwiseGPU.residual`), where Z normalises
///     it first (`normResidual`) — so this kernel is the only norm of the block's stream besides the QK-norm.
///
/// **Not the CPU's bits**: the sum of squares is the plain fp32 sum of `ElementwiseGPU`'s `norm_scale` (each lane its strided values,
/// `simd_sum`, the eight partials in order), not `vDSP_svesq`'s order, which is not reproduced (`QKNormRope`). Judged against the oracle
/// (`siliconed check moss-image dit`), the rest of the arithmetic the CPU's to the rounding.
package final class MossImageNormGPU {
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct NormParams { uint cols; float eps; };

    // `ElementwiseGPU`'s `row_scale`, the same order: one threadgroup of 256 per row.
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

    // out = ((x · r) · w) · (1 + scale) + shift — `Ops.rmsNorm` (`vsmul`, `vmul`) then `Ops.modulate`, each rounded.
    kernel void norm_modulate(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                              device const float *scale [[buffer(2)]], device const float *shift [[buffer(3)]],
                              device float *out [[buffer(4)]], constant NormParams &p [[buffer(5)]],
                              uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float partial[8];
        device const float *source = x + ulong(row) * p.cols;
        device float *destination = out + ulong(row) * p.cols;
        float r = row_scale(source, p.cols, p.eps, t, lane, sg, partial);
        for (uint i = t; i < p.cols; i += 256) {
            float v = source[i] * r;
            v = v * w[i];
            float factor = scale[i] + 1.0f;
            float product = v * factor;
            destination[i] = product + shift[i];
        }
    }

    \(BFloatKernels.roundingSource)

    // The same, and each value rounded to bf16 where the next products read their operand (`WeightProducts.RoundedOperand`): a
    // pair of adjacent columns per thread and step (`cols` even), each value computed as above.
    kernel void norm_modulate_rounded(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                                      device const float *scale [[buffer(2)]], device const float *shift [[buffer(3)]],
                                      device float *out [[buffer(4)]], device uint *rounded_out [[buffer(5)]],
                                      constant NormParams &p [[buffer(6)]],
                                      uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
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
            float factor0 = scale[i] + 1.0f, factor1 = scale[i + 1] + 1.0f;
            float product0 = v0 * factor0, product1 = v1 * factor1;
            float y0 = product0 + shift[i], y1 = product1 + shift[i + 1];
            destination[i] = y0; destination[i + 1] = y1;
            pairs[j] = rounded(y0) | (rounded(y1) << 16);
        }
    }

    \(QuantizedKernels.rowSource)

    // The same, and the row quantized to int8 for W8A8 where the next products read their operand (`WeightProducts.RoundedOperand`):
    // each value computed as above, by the thread that holds its column as `quantize_rows` would read it (chunk c of 256 in
    // simdgroup c % 8, eight consecutive columns a lane), then `quantize_held` — `quantize_rows`'s own code. q: the rotation's
    // stages, its norm's bits. Up to `ElementwiseGPU.quantizedColumns`.
    kernel void norm_modulate_quantized(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                                        device const float *scale [[buffer(2)]], device const float *shift [[buffer(3)]],
                                        device float *out [[buffer(4)]], constant NormParams &p [[buffer(5)]],
                                        device char *codes [[buffer(6)]], device float *scales [[buffer(7)]],
                                        constant uint2 &q [[buffer(8)]],
                                        uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float partial[8], part[32];
        device const float *source = x + ulong(row) * p.cols;
        device float *destination = out + ulong(row) * p.cols;
        float r = row_scale(source, p.cols, p.eps, t, lane, sg, partial);
        const uint chunks = (p.cols + 255) / 256;
        float held[MAXC][8];
        for (uint i = 0; i < MAXC; ++i) {
            const uint c = sg + i * 8;
            if (c < chunks) {
                for (uint j = 0; j < 8; ++j) {
                    const uint col = c * 256 + lane * 8 + j;
                    float y = 0.0f;
                    if (col < p.cols) {
                        float v = source[col] * r;
                        v = v * w[col];
                        float factor = scale[col] + 1.0f;
                        float product = v * factor;
                        y = product + shift[col];
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
        options.mathMode = .safe          // the division and the square root as written
        let library = try device.makeLibrary(source: Self.source, options: options)
        guard let function = library.makeFunction(name: "norm_modulate"), let rounded = library.makeFunction(name: "norm_modulate_rounded"),
              let quantized = library.makeFunction(name: "norm_modulate_quantized")
        else { throw GEMM.Failure.noDevice }
        pipeline = try device.makeComputePipelineState(function: function)
        roundedPipeline = try device.makeComputePipelineState(function: rounded)
        quantizedPipeline = try device.makeComputePipelineState(function: quantized)
    }

    private struct NormParams { var cols: UInt32; var eps: Float }

    /// `out[s, :] ← RMSNorm(x[s, :]) · w · (1 + scale) + shift` over `rows × columns`, one threadgroup per row; each operand a buffer and
    /// an offset in floats. Returns the GPU's seconds — 0 on the stream, where it is queued and judged at the drain. `rounded`: `out`
    /// also written rounded to bf16 there, or quantized to int8 for W8A8 where the operand says the producer writes it (`byProducer`).
    @discardableResult
    package func normModulate(_ x: ElementwiseGPU.BufferSlice, weight: ElementwiseGPU.BufferSlice, scale: ElementwiseGPU.BufferSlice,
                              shift: ElementwiseGPU.BufferSlice, into out: ElementwiseGPU.BufferSlice,
                              rows: Int, columns: Int, eps: Float, rounded operand: WeightProducts.RoundedOperand? = nil) -> Double {
        let rounded = operand?.bf16
        let quantized = operand?.int8.flatMap { $0.byProducer ? $0 : nil }
        guard rows > 0, columns > 0, case let commands = queue.checkedCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else {
            Attention.report("GPU modulated norm (MOSS-Image): no command buffer, so nothing ran")
            return 0
        }
        var p = NormParams(cols: UInt32(columns), eps: eps)
        encoder.setComputePipelineState(quantized != nil ? quantizedPipeline : rounded == nil ? pipeline : roundedPipeline)
        var slices = [x, weight, scale, shift, out]
        if let quantized { precondition(columns <= ElementwiseGPU.quantizedColumns); quantized.bind(encoder, at: slices.count + 1) }
        else if let rounded {
            precondition(columns % 2 == 0 && rounded.offset % 4 == 0)
            slices.append((rounded.buffer, rounded.offset / 4))
        }
        for (i, slice) in slices.enumerated() { encoder.setBuffer(slice.buffer, offset: slice.offset * 4, index: i) }
        encoder.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: slices.count)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        return commands.commitSettled(label: "elementwise GPU", what: "GPU modulated norm (MOSS-Image)")
    }
}
