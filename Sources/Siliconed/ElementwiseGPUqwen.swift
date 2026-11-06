import Foundation
import Metal
import Synchronization

/// **Qwen-Image-2.1's modulated LayerNorm, on the GPU** — `normed = LN(x) · (1 + s)`, the LayerNorm **without affine** of
/// `QwenImage21DiT.modulatedNorm` and `norm_out`, for the denoiser's stream (`GPUStream`), where the CPU's pass (`Ops.layerNorm`) left the GPU
/// idle between two products.
///
/// What sets it apart from the RMSNorms of `ElementwiseGPU` (Z-Image's stream), and what a port by analogy would miss:
///
///   - **two reductions per row, not one**: the mean first, then the sum of squares **of the centred row** — the CPU's order
///     (`vDSP_meanv`, `vDSP_vsadd` of −mean, `vDSP_svesq` of the difference), not `E[x²] − E[x]²`, which loses the digits a centred sum
///     keeps on rows far from zero;
///   - **no weight**: the scale is the modulation's `1 + s` (folded by `QwenImage21DiT.modulation`), of the band the rows belong to — the
///     conditions read `t = 0`, the target `t`: one dispatch per band, each its offset in the modulation slot;
///   - the roundings in the CPU's order after the sums: `x + (−mean)`, `· r`, `· scale`, each rounded (`fp contract(off)`), with
///     `r = 1 / √(ss / cols + ε)` as Swift writes it. The sums run in plain fp32 by lanes, `simd_sum`, then the eight partials in order:
///     not vDSP's order, so not the CPU's bits — judged against the oracle (`check qwen-image-2.1 dit`), as Z-Image's GPU norms are.
package final class ModulatedLayerNormGPU {
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct LayerNormParams { uint cols; float eps; };

    // The sum of `v` over the threadgroup of 256: each lane its strided values, `simd_sum`, the eight partials in order.
    static inline float block_sum(float v, uint lane, uint sg, threadgroup float *partial) {
        #pragma METAL fp contract(off)
        float s = simd_sum(v);
        if (lane == 0) partial[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint j = 0; j < 8; j++) total = total + partial[j];
        return total;
    }

    // out = ((x + (−mean)) · r) · scale, r = 1 / √(Σ(x − mean)² / cols + ε) — `Ops.layerNorm` with `scale`, one threadgroup per row.
    kernel void layer_norm_scale(device const float *x [[buffer(0)]], device const float *scale [[buffer(1)]],
                                 device float *out [[buffer(2)]], constant LayerNormParams &p [[buffer(3)]],
                                 uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float partialMean[8], partialSquares[8];
        device const float *source = x + ulong(row) * p.cols;
        device float *destination = out + ulong(row) * p.cols;
        float sum = 0.0f;
        for (uint i = t; i < p.cols; i += 256) sum = sum + source[i];
        float mean = precise::divide(block_sum(sum, lane, sg, partialMean), float(p.cols));
        float negativeMean = -mean;
        float squares = 0.0f;
        for (uint i = t; i < p.cols; i += 256) { float v = source[i] + negativeMean; float sq = v * v; squares = squares + sq; }
        float total = block_sum(squares, lane, sg, partialSquares);
        float r = precise::divide(1.0f, precise::sqrt(precise::divide(total, float(p.cols)) + p.eps));
        for (uint i = t; i < p.cols; i += 256) {
            float v = source[i] + negativeMean;
            v = v * r;
            v = v * scale[i];
            destination[i] = v;
        }
    }

    \(BFloatKernels.roundingSource)

    // The same, and each value rounded to bf16 where the next products read their operand (`WeightProducts.RoundedOperand`): a
    // pair of adjacent columns per thread and step (`cols` even), each value computed as above.
    kernel void layer_norm_scale_rounded(device const float *x [[buffer(0)]], device const float *scale [[buffer(1)]],
                                         device float *out [[buffer(2)]], device uint *rounded_out [[buffer(3)]],
                                         constant LayerNormParams &p [[buffer(4)]],
                                         uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float partialMean[8], partialSquares[8];
        device const float *source = x + ulong(row) * p.cols;
        device float *destination = out + ulong(row) * p.cols;
        device uint *pairs = rounded_out + ulong(row) * (p.cols / 2);
        float sum = 0.0f;
        for (uint i = t; i < p.cols; i += 256) sum = sum + source[i];
        float mean = precise::divide(block_sum(sum, lane, sg, partialMean), float(p.cols));
        float negativeMean = -mean;
        float squares = 0.0f;
        for (uint i = t; i < p.cols; i += 256) { float v = source[i] + negativeMean; float sq = v * v; squares = squares + sq; }
        float total = block_sum(squares, lane, sg, partialSquares);
        float r = precise::divide(1.0f, precise::sqrt(precise::divide(total, float(p.cols)) + p.eps));
        for (uint j = t; j < p.cols / 2; j += 256) {
            uint i = 2 * j;
            float v0 = source[i] + negativeMean, v1 = source[i + 1] + negativeMean;
            v0 = v0 * r; v1 = v1 * r;
            v0 = v0 * scale[i]; v1 = v1 * scale[i + 1];
            destination[i] = v0; destination[i + 1] = v1;
            pairs[j] = rounded(v0) | (rounded(v1) << 16);
        }
    }
    """

    private let queue: MTLCommandQueue
    private let pipeline, roundedPipeline: MTLComputePipelineState

    package init(device: MTLDevice, queue: MTLCommandQueue) throws {
        self.queue = queue
        let options = MTLCompileOptions()
        options.mathMode = .safe          // no fast-math: the division and the square root as written
        let library = try device.makeLibrary(source: Self.source, options: options)
        guard let f = library.makeFunction(name: "layer_norm_scale"), let g = library.makeFunction(name: "layer_norm_scale_rounded") else {
            throw GEMM.Failure.noDevice
        }
        pipeline = try device.makeComputePipelineState(function: f)
        roundedPipeline = try device.makeComputePipelineState(function: g)
    }

    /// **One per command queue**, compiled once: every layer of the DiT shares it.
    package static func shared(device: MTLDevice, queue: MTLCommandQueue) -> ModulatedLayerNormGPU? {
        let key = ObjectIdentifier(queue)
        return built.withLock { cache in
            if let k = cache[key] { return k }
            guard let made = try? ModulatedLayerNormGPU(device: device, queue: queue) else { return nil }
            cache[key] = made
            return made
        }
    }
    private static let built = Mutex<[ObjectIdentifier: ModulatedLayerNormGPU]>([:])

    private struct Params { var cols: UInt32; var eps: Float }

    /// `out[s, :] ← LN(x[s, :]) · scale` over `rows` rows of `columns`, one threadgroup per row, settled on the queue's stream
    /// (`commitSettled`): queued when the stream is on, waited for otherwise. Returns the GPU seconds when waited for.
    @discardableResult
    /// `rounded`: `out` also written rounded to bf16 there (its rows from the slot's), as the next products read their operand
    /// (`WeightProducts.roundedOperand`).
    package func run(_ x: ElementwiseGPU.BufferSlice, scale: ElementwiseGPU.BufferSlice, into out: ElementwiseGPU.BufferSlice,
                     rows: Int, columns: Int, eps: Float, rounded: WeightProducts.RoundedOperand? = nil) -> Double {
        guard rows > 0 else { return 0 }
        guard case let commands = queue.checkedCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            Attention.report("GPU LayerNorm: no command buffer, so nothing ran")
            return 0
        }
        var p = Params(cols: UInt32(columns), eps: eps)
        encoder.setComputePipelineState(rounded == nil ? pipeline : roundedPipeline)
        var slices = [x, scale, out]
        if let rounded {
            precondition(columns % 2 == 0 && rounded.offset % 4 == 0)
            slices.append((rounded.buffer, rounded.offset / 4))
        }
        for (i, t) in slices.enumerated() { encoder.setBuffer(t.buffer, offset: t.offset * 4, index: i) }
        encoder.setBytes(&p, length: MemoryLayout<Params>.stride, index: slices.count)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        return commands.commitSettled(label: "elementwise GPU", what: "GPU LayerNorm")
    }
}
