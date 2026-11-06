import Foundation
import Metal

/// **Krea 2's modulated norm on the GPU** — `RMSNorm(x)·w·(1 + scale) + shift`, the norm before each half of a block, queued on the
/// stream (`GPUStream`) between the residual that wrote `x` and the products that read the result, instead of a drain and a CPU pass.
///
///     per row, one threadgroup of 256 (8 simdgroups):
///       ss  = Σ x²                                  fp32, each lane its strided values, `simd_sum`, the eight partials in order
///       r   = 1 / √(ss/d + ε)                       `precise::` divide and square root, as the CPU's `/` and `squareRoot()`
///       out = ((x · r) · w) · factor + shift        four roundings in the CPU's order: `vDSP_vsmul`, `vDSP_vmul` (`Ops.rmsNorm`),
///                                                    then `x · a + b` in two (`Ops.modulate`): no contraction (`fp contract(off)`)
///
/// What sets it apart from Z-Image's `ElementwiseGPU.normScale` — what a port by analogy would get wrong:
///
///   - **a shift**: Krea 2's modulation is `(1 + scale)` then `+ shift`, two roundings as torch computes `x * a + b`; Z-Image's has
///     no shift;
///   - **`factor` is `1 + scale` rounded once, by the caller** (`vDSP_vadd` of the scale and ones, as `Ops.modulate` builds it): the
///     kernel multiplies by the rounded factor, never by `1 + scale` computed in the product;
///   - **the norm's weight already carries its `1 +`** (the forge folded the zero-centred weight): read as it is.
///
/// **Not the CPU's bits**: the sum of squares is a plain fp32 sum in another order than `vDSP_svesq`'s (`QKNormRope` says why that order
/// is not reproduced). Everything after the sum is the CPU's operations in its order. A refutation stands beside it: a compensated sum
/// moved Krea 2's guided trajectory away from fp64 (`Krea2DiT.modulatedNorm`); this one is the CPU's kind of sum, and the checks decide.
package final class ModulatedNormGPU {
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct NormParams { uint cols; float eps; };

    kernel void norm_modulate(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                              device const float *factor [[buffer(2)]], device const float *shift [[buffer(3)]],
                              device float *out [[buffer(4)]], constant NormParams &p [[buffer(5)]],
                              uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float partial[8];
        device const float *source = x + ulong(row) * p.cols;
        device float *destination = out + ulong(row) * p.cols;
        float ss = 0.0f;
        for (uint i = t; i < p.cols; i += 256) { float v = source[i]; float sq = v * v; ss = ss + sq; }
        float s = simd_sum(ss);
        if (lane == 0) partial[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint j = 0; j < 8; j++) total = total + partial[j];
        float mean = precise::divide(total, float(p.cols));
        float r = precise::divide(1.0f, precise::sqrt(mean + p.eps));
        for (uint i = t; i < p.cols; i += 256) {
            float v = source[i] * r;
            v = v * w[i];
            v = v * factor[i];
            v = v + shift[i];
            destination[i] = v;
        }
    }

    \(BFloatKernels.roundingSource)

    // The same, and each value rounded to bf16 where the next products read their operand (`WeightProducts.RoundedOperand`): a
    // pair of adjacent columns per thread and step (`cols` even), each value computed as above.
    kernel void norm_modulate_rounded(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                                      device const float *factor [[buffer(2)]], device const float *shift [[buffer(3)]],
                                      device float *out [[buffer(4)]], device uint *rounded_out [[buffer(5)]],
                                      constant NormParams &p [[buffer(6)]],
                                      uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float partial[8];
        device const float *source = x + ulong(row) * p.cols;
        device float *destination = out + ulong(row) * p.cols;
        device uint *pairs = rounded_out + ulong(row) * (p.cols / 2);
        float ss = 0.0f;
        for (uint i = t; i < p.cols; i += 256) { float v = source[i]; float sq = v * v; ss = ss + sq; }
        float s = simd_sum(ss);
        if (lane == 0) partial[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint j = 0; j < 8; j++) total = total + partial[j];
        float mean = precise::divide(total, float(p.cols));
        float r = precise::divide(1.0f, precise::sqrt(mean + p.eps));
        for (uint j = t; j < p.cols / 2; j += 256) {
            uint i = 2 * j;
            float v0 = source[i] * r, v1 = source[i + 1] * r;
            v0 = v0 * w[i]; v1 = v1 * w[i + 1];
            v0 = v0 * factor[i]; v1 = v1 * factor[i + 1];
            v0 = v0 + shift[i]; v1 = v1 + shift[i + 1];
            destination[i] = v0; destination[i + 1] = v1;
            pairs[j] = rounded(v0) | (rounded(v1) << 16);
        }
    }

    \(QuantizedKernels.rowSource)

    // The same, and the row quantized to int8 for W8A8 where the next products read their operand (`WeightProducts.RoundedOperand`):
    // each value computed as above, by the thread that holds its column as `quantize_rows` would read it (chunk c of 256 in
    // simdgroup c % 8, eight consecutive columns a lane), then `quantize_held` — `quantize_rows`'s own code. q: the rotation's
    // stages, its norm's bits. Up to `ElementwiseGPU.quantizedColumns`.
    kernel void norm_modulate_quantized(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                                        device const float *factor [[buffer(2)]], device const float *shift [[buffer(3)]],
                                        device float *out [[buffer(4)]], constant NormParams &p [[buffer(5)]],
                                        device char *codes [[buffer(6)]], device float *scales [[buffer(7)]],
                                        constant uint2 &q [[buffer(8)]],
                                        uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float partial[8], part[32];
        device const float *source = x + ulong(row) * p.cols;
        device float *destination = out + ulong(row) * p.cols;
        float ss = 0.0f;
        for (uint i = t; i < p.cols; i += 256) { float v = source[i]; float sq = v * v; ss = ss + sq; }
        float s = simd_sum(ss);
        if (lane == 0) partial[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint j = 0; j < 8; j++) total = total + partial[j];
        float mean = precise::divide(total, float(p.cols));
        float r = precise::divide(1.0f, precise::sqrt(mean + p.eps));
        const uint chunks = (p.cols + 255) / 256;
        float held[MAXC][8];
        for (uint i = 0; i < MAXC; ++i) {
            const uint c = sg + i * 8;
            if (c < chunks) {
                for (uint j = 0; j < 8; ++j) {
                    const uint col = c * 256 + lane * 8 + j;
                    float y = 0.0f;
                    if (col < p.cols) {
                        y = source[col] * r;
                        y = y * w[col];
                        y = y * factor[col];
                        y = y + shift[col];
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
        guard let f = library.makeFunction(name: "norm_modulate"), let g = library.makeFunction(name: "norm_modulate_rounded"),
              let h = library.makeFunction(name: "norm_modulate_quantized") else {
            throw GEMM.Failure.noDevice
        }
        pipeline = try device.makeComputePipelineState(function: f)
        roundedPipeline = try device.makeComputePipelineState(function: g)
        quantizedPipeline = try device.makeComputePipelineState(function: h)
    }

    private struct NormParams { var cols: UInt32; var eps: Float }

    /// `out[s, :] ← ((RMSNorm(x[s, :]) · w) · factor) + shift`, one threadgroup per row; `factor` is `1 + scale`, rounded by the caller.
    /// Settled on the queue's stream (`commitSettled`): waited for and its GPU seconds returned off the stream, queued and 0 on it.
    @discardableResult
    /// `rounded`: `out` also written rounded to bf16 there, as the next products read their operand (`WeightProducts.roundedOperand`) —
    /// or quantized to int8 for W8A8, where the operand says the producer writes it (`byProducer`).
    package func run(_ x: ElementwiseGPU.BufferSlice, weight: ElementwiseGPU.BufferSlice, factor: ElementwiseGPU.BufferSlice,
                     shift: ElementwiseGPU.BufferSlice, into out: ElementwiseGPU.BufferSlice, rows: Int, columns: Int, eps: Float,
                     rounded operand: WeightProducts.RoundedOperand? = nil) -> Double {
        let rounded = operand?.bf16
        let quantized = operand?.int8.flatMap { $0.byProducer ? $0 : nil }
        let commands = queue.checkedCommandBuffer()
        let encoder = commands.checkedEncoder()
        encoder.setComputePipelineState(quantized != nil ? quantizedPipeline : rounded == nil ? pipeline : roundedPipeline)
        var slices = [x, weight, factor, shift, out]
        if let quantized { precondition(columns <= ElementwiseGPU.quantizedColumns); quantized.bind(encoder, at: slices.count + 1) }
        else if let rounded {
            precondition(columns % 2 == 0 && rounded.offset % 4 == 0)
            slices.append((rounded.buffer, rounded.offset / 4))
        }
        for (i, t) in slices.enumerated() { encoder.setBuffer(t.buffer, offset: t.offset * 4, index: i) }
        var p = NormParams(cols: UInt32(columns), eps: eps)
        encoder.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: slices.count)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        return commands.commitSettled(label: "elementwise GPU", what: "GPU elementwise (norm_modulate)")
    }
}
