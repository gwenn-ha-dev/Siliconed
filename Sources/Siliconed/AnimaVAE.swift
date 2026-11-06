import Metal
import MetalPerformanceShadersGraph
import Foundation

/// The Qwen-Image VAE decoder (Anima's), in fp32, by passes of one kernel (`QwenImage21VAE.Stage`).
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

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let weights: Safetensors
    private let config: Config
    package let latentHeight, latentWidth: Int
    /// The Qwen-Image decoder is published in bf16 (`VAEPrecision`; Krea's fp32 copy is the same values widened, not installed).
    package static let publishedInBF16 = true
    package var imageHeight: Int { latentHeight * 8 }
    package var imageWidth: Int { latentWidth * 8 }

    /// - Parameter path: the published `diffusion_pytorch_model.safetensors` (bf16), read as is.
    /// - Parameter latentHeight, latentWidth: the latent grid `[16, h, w]`.
    package init(path: String, latentHeight: Int, latentWidth: Int, config: Config = .qwenImage) throws {
        let weights = try Safetensors(path: path)
        guard weights.entries["decoder.conv_in.weight"]?.shape == [384, 16, 3, 3, 3] else {
            throw Safetensors.Failure.badHeader("\(path): not the Qwen-Image VAE")
        }
        let (device, queue) = try SystemGPU.open()
        self.device = device
        self.queue = queue
        self.weights = weights
        self.config = config
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
    }

    /// **Passes of one kernel, in a heap of ours** (`VAEMemory`) — the cut of Qwen-Image-2.1's
    /// decoder (`QwenImage21VAE.run`): a graph of several operations put its intermediates beyond our heaps.
    ///
    ///     latent [16, h, w] → [384, h, w] → ×2 [192, 2h, 2w] → [384, 2h, 2w] → ×2 [192, 4h, 4w]
    ///       → ×2 [96, 8h, 8w] → image [3, 8h, 8w]
    ///
    /// **Judged against the fp64 witness** (the oracle's VAE recomputed in float64): not at the bit against the
    /// fp32 oracle — no GPU convolution sums in the order of torch's CPU one —, but closer to the exact
    /// than that oracle: worst plane 6.3·10⁻⁷ against its 1.0·10⁻⁶ at 512², 3.2·10⁻⁷ against 8.4·10⁻⁷
    /// at 1024² (`anima-vae`).
    private static func stages(config: Config, latentHeight h: Int, latentWidth w: Int) -> [QwenImage21VAE.Stage] {
        typealias V = QwenImage21VAE
        let L = (h: h, w: w), L2 = (h: 2 * h, w: 2 * w), L4 = (h: 4 * h, w: 4 * w), L8 = (h: 8 * h, w: 8 * w)
        let shape: [Int] = [1, config.latentChannels, 1, 1]
        // The pipeline denormalizes BEFORE the decoder: `latents × std + mean`, per channel.
        var passes: [V.Stage] = [
            // On the fp32 latent; rounded once into the activations in bf16 (`VAEMemory.half`).
            V.Stage(["x"], "x", [1, config.latentChannels, h, w]) { b, x in
                let y = b.graph.addition(b.graph.multiplication(x[0], b.constant(config.std, shape), name: nil),
                                         b.constant(config.mean, shape), name: "denormalize")
                return b.dataType == .float32 ? y : b.graph.cast(y, to: b.dataType, name: nil) },
            V.conv("post_quant_conv", 16, 16, size: L, kernel: 1),
            V.conv("decoder.conv_in", 16, 384, size: L),
        ]
        passes += V.resnet("decoder.mid_block.resnets.0", 384, 384, size: L, label: "mid.resnet0")
        // The bottleneck **without** query slices: its bits were never cut up.
        passes += V.attention("decoder.mid_block.attentions.0", 384, size: L, label: "mid.attention", requestBlock: 0)
        passes += V.resnet("decoder.mid_block.resnets.1", 384, 384, size: L, label: "mid")
        /// `nearest-exact` ×2, then `Conv2d(C, C/2, 3)`.
        func upsample(_ prefix: String, _ c: Int, size: (h: Int, w: Int)) -> [V.Stage] {
            [V.double(c, size: size),
             V.Stage(["x"], "x", [1, c / 2, 2 * size.h, 2 * size.w], weights: WeightList.conv(c / 2, c), label: prefix) { b, x in
                 try b.conv(x[0], prefix + ".resample.1", padding: 1) }]
        }
        for r in 0..<3 { passes += V.resnet("decoder.up_blocks.0.resnets.\(r)", 384, 384, size: L, label: "0.resnet\(r)") }
        passes += upsample("decoder.up_blocks.0.upsamplers.0", 384, size: L)
        for r in 0..<3 {
            passes += V.resnet("decoder.up_blocks.1.resnets.\(r)", r == 0 ? 192 : 384, 384, size: L2, label: "1.resnet\(r)")
        }
        passes += upsample("decoder.up_blocks.1.upsamplers.0", 384, size: L2)
        for r in 0..<3 { passes += V.resnet("decoder.up_blocks.2.resnets.\(r)", 192, 192, size: L4, label: "2.resnet\(r)") }
        passes += upsample("decoder.up_blocks.2.upsamplers.0", 192, size: L4)
        for r in 0..<3 { passes += V.resnet("decoder.up_blocks.3.resnets.\(r)", 96, 96, size: L8, label: "3.resnet\(r)") }
        var normOut = V.Stage(["x"], "x", [1, 96, L8.h, L8.w], weights: WeightList.rms(96)) { b, x in
            try b.silu(b.rmsNorm(x[0], "decoder.norm_out", channels: 96)) }
        normOut.norm = ("decoder.norm_out", 96, true)
        passes.append(normOut)
        passes.append(V.Stage(["x"], "x", [1, 3, L8.h, L8.w], weights: WeightList.conv(3, 96), label: "3") { b, x in
            b.graph.clamp(try b.conv(x[0], "decoder.conv_out", padding: 1),
                          min: b.graph.constant(-1, dataType: b.dataType), max: b.graph.constant(1, dataType: b.dataType), name: nil) })
        return passes
    }

    /// `[O, I, T, kh, kw]` → `[O, I, kh, kw]`, the slice `t = T − 1` — the only one that a single
    /// frame, preceded by `T − 1` frames of zeros, exercises. A tensor that is not 5D
    /// passes as is. Internal and not private: it is the line that decides that the 3D VAE *is* a
    /// 2D VAE, and a wrong index (the central slice, by reflex) would give a plausible image.
    /// (`VAEBuildingBlocks.layout` takes the same slice from the file, without this copy.)
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

    /// **Everything a decoding holds** for an `h × w` latent: the registers' heap (the latent, each
    /// stage's output, its weights), the largest stage's workspace, the image it returns — `decode`
    /// walked without a device.
    package static func plan(latentHeight h: Int, latentWidth w: Int, config: Config = .qwenImage, half: Bool = false) -> VAELayout {
        let memory = VAEMemory(half: half)
        do { _ = try decode(nil, latentHeight: h, latentWidth: w, config: config, weights: nil, memory: memory) } catch { preconditionFailure("VAE plan walk refused: \(error)") }
        return memory.layout
    }

    private static func decode(_ latent: UnsafePointer<Float>?, latentHeight h: Int, latentWidth w: Int, config: Config,
                               weights: Safetensors?, memory: VAEMemory) throws -> [Float] {
        let input = try memory.register([1, config.latentChannels, h, w], .float32)
        if let latent, memory.isLive { input.values.update(from: latent, count: input.count) }
        let image = try QwenImage21VAE.run(stages(config: config, latentHeight: h, latentWidth: w), weights: weights,
                                           name: "VAE Qwen-Image", memory: memory, registers: ["x": input])["x"]!
        memory.host(4 * image.count)
        let pixels = memory.isLive ? image.floats() : []
        memory.free(image)
        return pixels
    }

    /// `latent` as `[16, h, w]` as the sampler returns it. Outputs `[3, 8h, 8w]` in `[-1, 1]`.
    package func decode(latent: UnsafePointer<Float>) throws -> (image: [Float], seconds: Double) {
        let started = Date()
        let half = VAEPrecision.half(publishedInBF16: AnimaVAE.publishedInBF16, weights: weights)
        let memory = try VAEMemory(plan: AnimaVAE.plan(latentHeight: latentHeight, latentWidth: latentWidth, config: config, half: half),
                                   device: device, queue: queue, half: half)
        let image = try autoreleasepool {
            try AnimaVAE.decode(latent, latentHeight: latentHeight, latentWidth: latentWidth, config: config,
                                weights: weights, memory: memory)
        }
        memory.finish()
        return (image, Date().timeIntervalSince(started))
    }
}
