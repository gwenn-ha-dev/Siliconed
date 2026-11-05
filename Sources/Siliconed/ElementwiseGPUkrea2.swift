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
    """

    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    package init(device: MTLDevice, queue: MTLCommandQueue) throws {
        self.queue = queue
        let options = MTLCompileOptions()
        options.mathMode = .safe          // no fast-math: the division and the square root as written
        let library = try device.makeLibrary(source: Self.source, options: options)
        guard let f = library.makeFunction(name: "norm_modulate") else { throw GEMM.Failure.noDevice }
        pipeline = try device.makeComputePipelineState(function: f)
    }

    private struct NormParams { var cols: UInt32; var eps: Float }

    /// `out[s, :] ← ((RMSNorm(x[s, :]) · w) · factor) + shift`, one threadgroup per row; `factor` is `1 + scale`, rounded by the caller.
    /// Settled on the queue's stream (`commitSettled`): waited for and its GPU seconds returned off the stream, queued and 0 on it.
    @discardableResult
    package func run(_ x: ElementwiseGPU.BufferSlice, weight: ElementwiseGPU.BufferSlice, factor: ElementwiseGPU.BufferSlice,
                     shift: ElementwiseGPU.BufferSlice, into out: ElementwiseGPU.BufferSlice, rows: Int, columns: Int, eps: Float) -> Double {
        let commands = queue.checkedCommandBuffer()
        let encoder = commands.checkedEncoder()
        encoder.setComputePipelineState(pipeline)
        for (i, t) in [x, weight, factor, shift, out].enumerated() { encoder.setBuffer(t.buffer, offset: t.offset * 4, index: i) }
        var p = NormParams(cols: UInt32(columns), eps: eps)
        encoder.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        return commands.commitSettled(label: "elementwise GPU", what: "GPU elementwise (norm_modulate)")
    }
}
