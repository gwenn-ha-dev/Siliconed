import Metal
import MetalPerformanceShadersGraph
import Foundation

/// The Qwen-Image VAE decoder (Anima's), in fp32, through `MPSGraph`.
///
///     latent [16, h, w] ─ ×std + mean ─ post_quant_conv ─ conv_in ─ mid(resnet, attention, resnet)
///       ─ up×4 (3 resnets, ×2 except the last) ─ RMSNorm ─ SiLU ─ conv_out ─ clamp ─ image [3, 8h, 8w]
///
/// ## Why there is no 3D convolution here — and what that implies
///
/// The VAE is Wan's: a **video** VAE, with **causal** 3D convolutions. But an image
/// is a video of **one** frame, and `AutoencoderKLQwenImage._decode` decodes it with an empty
/// frame cache. Two consequences, read in the diffusers code (8b3c707):
///
///   - `QwenImageCausalConv3d` pads time with **two frames of zeros in front** and nothing
///     behind (`_padding = (…, 2·p, 0)`). On one frame, a `3×3×3` kernel thus sees only
///     zeros through its `t = 0` and `t = 1` slices: it equals the 2D convolution of its slice
///     **`t = 2`**, and of it alone. The `1×1×1` are `1×1`.
///   - `QwenImageResample("upsample3d")` skips its `time_conv` on the first pass (`feat_cache`
///     empty → `"Rep"`): there is **no** temporal doubling, only the spatial ×2.
///
/// The decoder of an image is thus a 2D decoder, and this is not an approximation: it is the
/// same sum, minus the products by zero. The check is the oracle's image, which was
/// decoded by the real 3D module.
///
/// What sets it apart from Flux's VAE, which a port by analogy would miss:
///
///   - **RMSNorm** and not `GroupNorm`: `F.normalize` over the channels (L2 norm, floor 1e-12),
///     then `× √C × γ`, no bias;
///   - the denormalization is **per channel** (`latent × std + mean`) and there is a `post_quant_conv`;
///   - the upsampler **halves the channels** (`Conv2d(dim, dim/2)`), so each first
///     resnet of a block receives half of what the previous block output;
///   - the bottleneck attention draws q, k, v from **a single** 1×1 `to_qkv`, in that order;
///   - the output is **clamped to [-1, 1]** by the decoder itself.
package final class AnimaVAE {
    package struct Config: Sendable {
        package let mean: [Float], std: [Float]
        package let latentChannels = 16
        /// Anima-Base-v1.0-Diffusers' `vae/config.json`.
        package static let qwenImage = Config(
            mean: [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                   0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921],
            std: [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                  3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160])
    }

    /// A stage: one graph, one input, one output — `VAE`'s cut, for the same reason
    /// (measured: `MPSGraph` keeps alive the intermediates of the whole graph it executes).
    private struct Stage {
        let graph: MPSGraph
        let entry: MPSGraphTensor
        let output: MPSGraphTensor
        let shape: [NSNumber]
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let stages: [Stage]
    package let latentHeight, latentWidth: Int
    package var imageHeight: Int { latentHeight * 8 }
    package var imageWidth: Int { latentWidth * 8 }

    /// - Parameter path: the published `diffusion_pytorch_model.safetensors` (bf16), read as is.
    /// - Parameter latentHeight, latentWidth: the latent grid `[16, h, w]`.
    package init(path: String, latentHeight: Int, latentWidth: Int, config: Config = .qwenImage) throws {
        let weights = try Safetensors(path: path)
        guard weights.entries["decoder.conv_in.weight"]?.shape == [384, 16, 3, 3, 3] else {
            throw Safetensors.Failure.badHeader("\(path): not the Qwen-Image VAE")
        }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw GEMM.Failure.noDevice
        }
        self.device = device
        self.queue = queue
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
        self.stages = try (0..<4).map {
            try AnimaVAE.build(weights: weights, config: config, L: (latentHeight, latentWidth), stage: $0)
        }
    }

    private static func build(weights: Safetensors, config: Config, L: (h: Int, w: Int), stage: Int) throws -> Stage {
        let graph = MPSGraph()
        //   0 : [16, H, W]    1 : [192, 2H, 2W]    2 : [192, 4H, 4W]    3 : [96, 8H, 8W]
        let (channels, factor) = [(config.latentChannels, 1), (192, 2), (192, 4), (96, 8)][stage]
        let shape: [NSNumber] = [1, NSNumber(value: channels), NSNumber(value: factor * L.h),
                                 NSNumber(value: factor * L.w)]
        let L2 = (h: 2 * L.h, w: 2 * L.w), L4 = (h: 4 * L.h, w: 4 * L.w)
        let input = graph.placeholder(shape: shape, dataType: .float32, name: "input")

        let b = VAEBuildingBlocks(graph: graph, weights: weights, name: "VAE Qwen-Image")

        /// The bottleneck: one head, `H·W` tokens of `C` channels, scale `1/√C`, q, k, v drawn from a
        /// single 1×1 `to_qkv` — **without** query slices: its bits were never cut up.
        func attention(_ x: MPSGraphTensor, _ prefix: String, channels: Int, size: (h: Int, w: Int)) throws -> MPSGraphTensor {
            try b.attention(x, normalized: b.rmsNorm(x, prefix + ".norm", channels: channels), channels: channels, size: size,
                            qkv: { (try b.projection($0, prefix + ".to_qkv", rows: 0..<channels),
                                    try b.projection($0, prefix + ".to_qkv", rows: channels..<(2 * channels)),
                                    try b.projection($0, prefix + ".to_qkv", rows: (2 * channels)..<(3 * channels))) },
                            output: prefix + ".proj", requestBlock: 0)
        }

        /// `nearest-exact` ×2 (`VAEBuildingBlocks.double`), then `Conv2d(C, C/2, 3)`.
        func upsample(_ x: MPSGraphTensor, _ prefix: String, channels: Int, size: (h: Int, w: Int)) throws -> MPSGraphTensor {
            try b.conv(b.double(x, channels: channels, size: size), prefix + ".resample.1", padding: 1)
        }
        func constant(_ values: [Float], _ shape: [Int]) -> MPSGraphTensor { b.constant(values, shape) }

        var x = input
        switch stage {
        case 0:
            // The pipeline denormalizes BEFORE the decoder: `latents × std + mean`, per channel.
            let shape: [Int] = [1, config.latentChannels, 1, 1]
            x = graph.addition(graph.multiplication(x, constant(config.std, shape), name: nil),
                               constant(config.mean, shape), name: "denormalize")
            x = try b.conv(x, "post_quant_conv", padding: 0)
            x = try b.conv(x, "decoder.conv_in", padding: 1)
            x = try b.resnetQwen(x, "decoder.mid_block.resnets.0", inChannels: 384, outChannels: 384)
            x = try attention(x, "decoder.mid_block.attentions.0", channels: 384, size: L)
            x = try b.resnetQwen(x, "decoder.mid_block.resnets.1", inChannels: 384, outChannels: 384)
            for r in 0..<3 { x = try b.resnetQwen(x, "decoder.up_blocks.0.resnets.\(r)", inChannels: 384, outChannels: 384) }
            x = try upsample(x, "decoder.up_blocks.0.upsamplers.0", channels: 384, size: L)
        case 1:
            for r in 0..<3 {
                x = try b.resnetQwen(x, "decoder.up_blocks.1.resnets.\(r)", inChannels: r == 0 ? 192 : 384, outChannels: 384)
            }
            x = try upsample(x, "decoder.up_blocks.1.upsamplers.0", channels: 384, size: L2)
        case 2:
            for r in 0..<3 { x = try b.resnetQwen(x, "decoder.up_blocks.2.resnets.\(r)", inChannels: 192, outChannels: 192) }
            x = try upsample(x, "decoder.up_blocks.2.upsamplers.0", channels: 192, size: L4)
        default:
            for r in 0..<3 { x = try b.resnetQwen(x, "decoder.up_blocks.3.resnets.\(r)", inChannels: 96, outChannels: 96) }
            x = try b.silu(b.rmsNorm(x, "decoder.norm_out", channels: 96))
            x = try b.conv(x, "decoder.conv_out", padding: 1)
            x = graph.clamp(x, min: graph.constant(-1, dataType: .float32),
                            max: graph.constant(1, dataType: .float32), name: nil)
        }
        return Stage(graph: graph, entry: input, output: x, shape: shape)
    }

    /// `[O, I, T, kh, kw]` → `[O, I, kh, kw]`, the slice `t = T − 1` — the only one that a single
    /// frame, preceded by `T − 1` frames of zeros, exercises. A tensor that is not 5D
    /// passes as is. Internal and not private: it is the line that decides that the 3D VAE *is* a
    /// 2D VAE, and a wrong index (the central slice, by reflex) would give a plausible image.
    static func lastSlice(_ w: [Float], shape: [Int]) -> (values: [Float], shape: [Int]) {
        guard shape.count == 5 else { return (w, shape) }
        let (o, i, t, kh, kw) = (shape[0], shape[1], shape[2], shape[3], shape[4])
        let plan = kh * kw
        var slice = [Float](repeating: 0, count: o * i * plan)
        for oi in 0..<(o * i) {
            let source = (oi * t + (t - 1)) * plan
            for j in 0..<plan { slice[oi * plan + j] = w[source + j] }
        }
        return (slice, [o, i, kh, kw])
    }

    /// `latent` as `[16, h, w]` as the sampler returns it. Outputs `[3, 8h, 8w]` in `[-1, 1]`.
    package func decode(latent: UnsafePointer<Float>) -> (image: [Float], seconds: Double) {
        let count = 16 * latentHeight * latentWidth
        let inBuffer = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        inBuffer.contents().assumingMemoryBound(to: Float.self).update(from: latent, count: count)
        let started = Date()
        var data = MPSGraphTensorData(inBuffer, shape: stages[0].shape, dataType: .float32)
        for stage in stages {
            autoreleasepool {
                let results = stage.graph.run(with: queue, feeds: [stage.entry: data],
                                                 targetTensors: [stage.output], targetOperations: nil)
                data = results[stage.output]!
            }
        }
        var image = [Float](repeating: 0, count: 3 * imageHeight * imageWidth)
        image.withUnsafeMutableBytes { data.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
        return (image, Date().timeIntervalSince(started))
    }
}
