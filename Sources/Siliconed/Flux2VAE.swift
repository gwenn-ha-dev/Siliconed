import Metal
import MetalPerformanceShadersGraph
import Foundation

/// The FLUX.2 VAE decoder (`AutoencoderKLFlux2`) as MOSS-Image uses it, in fp32, by the banded passes of
/// `VAE.Bands` (one kernel per graph, every byte in the heaps of `VAEMemory`).
///
///     DiT latent [128, h/2, w/2] ─ ×√(var+ε) + mean (per channel, 128) ─ unpatchify 2×2 ─ z [32, h, w]
///       ─ post_quant_conv (1×1) ─ conv_in ─ mid(resnet, attention, resnet)
///       ─ up×4 (3 resnets, ×2 except the last) ─ GroupNorm ─ SiLU ─ conv_out ─ clamp ─ image [3, 8h, 8w]
///
/// Channels 384 → 384 → 192 → 96 (`Config.levels`): the ×2 sits between two levels, and the first
/// resnet of a level that narrows (384 → 192, 192 → 96) carries a 1×1 `conv_shortcut`. The factor
/// to the image is 16: 8 from the network, 2 from the patch (4096 DiT tokens at 1024²).
///
/// What sets it apart from Flux.1's decoder (`VAE`, Z-Image's), which a port by analogy would miss
/// (reference: diffusers 0.41 `autoencoder_kl_flux2.py`, `vae.py`; the denormalization is in MOSS's own
/// `pipeline_moss_image.py`, not in the VAE):
///
///   - **the decoder is narrower than the encoder**: `decoder_block_out_channels` 96/192/384/384
///     is read, and `block_out_channels` 128/256/512/512 is the ENCODER's. Flux.1's decoder is 128…512;
///     reading the wrong key gives a shape error at the first weight, or worse, none at the bottleneck;
///   - **the latent is normalized by a BatchNorm's running statistics, per channel of the PATCHED
///     latent** (128 of them, `bn.running_mean`, `bn.running_var`, `eps` 1e-4 from `batch_norm_eps`, not
///     torch's 1e-5): `z·√(var + ε) + mean`, applied BEFORE the unpatchify, on the channel order
///     the DiT produces. Flux.1's two scalars (`÷scale + shift`) and Qwen-Image's 16 means are not this;
///   - **the patch order is `(c, py, px)`**: channel `4c + 2py + px` of the patched latent is the pixel
///     `(2y + py, 2x + px)` of channel `c`. `reshape(c/4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)`: a
///     transposed order (`px` before `py`) gives an image of the right statistics with 2×2 cells
///     mirrored along the diagonal;
///   - **`post_quant_conv` exists** (`use_post_quant_conv`), a 1×1 on the 32 channels; Flux.1's config
///     has none. It cannot be folded into `conv_in`: that one pads with zeros after it, not before;
///   - **the output is clamped to [-1, 1] by the pipeline**, after the decoder: done here, as `AnimaVAE` does;
///   - the attention and the resnets are Flux.1's (`GroupNorm` 32 groups, eps 1e-6, a single head of
///     width C = 384, scale 1/√384), so the layers are `VAE.Bands`, whose widths come from the call and
///     not from the class: only the cut and the head are written here;
///   - `force_upcast`: the pipeline decodes in fp32 (`vae_dtype=torch.float32`); so do we, from the published weights.
///
/// Tiling (`vaeTile`) is not offered: it changes the image (`VAE.decodePerTiles`), and the decoder's
/// bands already hold its peak under B at every format.
package final class Flux2VAE {
    package struct Config: Hashable, Sendable {
        /// `decoder_block_out_channels` reversed to run from the bottleneck out.
        package let levels: [Int]
        package let groups: Int
        package let latentChannels: Int
        package let patch: Int
        package let batchNormEpsilon: Float
        package init(levels: [Int] = [384, 384, 192, 96], groups: Int = 32, latentChannels: Int = 32, patch: Int = 2,
                     batchNormEpsilon: Float = 1e-4) {
            self.levels = levels; self.groups = groups; self.latentChannels = latentChannels
            self.patch = patch; self.batchNormEpsilon = batchNormEpsilon
        }
        /// The channels of the DiT's latent: `latentChannels · patch²`.
        package var patchedChannels: Int { latentChannels * patch * patch }
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let weights: Safetensors
    private let config: Config
    /// `√(running_var + ε)` and `running_mean`, one per patched channel.
    private let scale: [Float], shift: [Float]
    /// The grid of the decoder's input (after the unpatchify): the image is `8×` that.
    package let latentHeight, latentWidth: Int
    package var imageHeight: Int { latentHeight * 8 }
    package var imageWidth: Int { latentWidth * 8 }

    /// - Parameter latentHeight, latentWidth: the DECODER's grid, `[32, h, w]` — twice the DiT's.
    package init(path: String, latentHeight: Int, latentWidth: Int, config: Config = Config()) throws {
        let weights = try Safetensors(path: path)
        guard weights.entries["decoder.conv_in.weight"]?.shape == [config.levels[0], config.latentChannels, 3, 3],
              weights.entries["post_quant_conv.weight"]?.shape == [config.latentChannels, config.latentChannels, 1, 1],
              let mean = weights.materialize("bn.running_mean"), let variance = weights.materialize("bn.running_var"),
              mean.count == config.patchedChannels, variance.count == config.patchedChannels else {
            throw Safetensors.Failure.badHeader("\(path): not the FLUX.2 VAE (decoder \(config.levels), \(config.patchedChannels) latent channels)")
        }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw GEMM.Failure.noDevice
        }
        self.device = device; self.queue = queue
        self.weights = weights; self.config = config
        self.scale = Flux2VAE.scale(variance: variance, epsilon: config.batchNormEpsilon)
        self.shift = mean
        self.latentHeight = latentHeight; self.latentWidth = latentWidth
    }

    // MARK: - The pipeline's part: denormalization and unpatchify (pure)

    /// `√(running_var + ε)`, in fp32 as torch computes it (`(bn.running_var + eps).sqrt()`).
    package static func scale(variance: [Float], epsilon: Float) -> [Float] { variance.map { ($0 + epsilon).squareRoot() } }

    /// `z · scale + shift` per channel of a `[C, h, w]` latent (`pipeline_moss_image.py`).
    package static func denormalize(_ z: [Float], scale: [Float], shift: [Float], plane: Int) -> [Float] {
        var out = z
        for c in 0..<scale.count {
            let (s, m) = (scale[c], shift[c])
            for i in (c * plane)..<((c + 1) * plane) { out[i] = z[i] * s + m }
        }
        return out
    }

    /// `[C·p², h, w]` → `[C, p·h, p·w]`: channel `p²c + p·py + px` goes to the pixel `(p·y + py, p·x + px)`
    /// (`reshape(C, p, p, h, w).permute(0, 1, 4, 2, 5, 3)`).
    package static func unpatchify(_ z: [Float], channels: Int, patch p: Int, height h: Int, width w: Int) -> [Float] {
        precondition(z.count == channels * p * p * h * w, "Flux2VAE.unpatchify: the latent does not have \(channels)·\(p)² channels")
        var out = [Float](repeating: 0, count: z.count)
        let (H, W) = (p * h, p * w)
        for c in 0..<channels {
            for py in 0..<p {
                for px in 0..<p {
                    let source = ((c * p + py) * p + px) * h * w
                    for y in 0..<h {
                        let target = (c * H + p * y + py) * W + px
                        for x in 0..<w { out[target + p * x] = z[source + y * w + x] }
                    }
                }
            }
        }
        return out
    }

    // MARK: - The decoder

    /// **The decoder's layers on a `[32, h, w]` latent (already denormalized and unpatchified)**, as
    /// `VAE.decodeBanded` cuts them — the head by passes, then layer by layer, each by bands; the
    /// ×2 comes after the third resnet of a level. Walked without a device with `nil`, it is the plan.
    private static func decodeBanded(_ latent: UnsafePointer<Float>?, height h: Int, width w: Int, config: Config,
                                     bands: VAE.Bands) throws -> [Float] {
        let memory = bands.memory
        let c = config.latentChannels, levels = config.levels
        // The two host arrays the pipeline's part makes (denormalized, unpatchified), alive together.
        memory.host(2 * 4 * c * h * w)
        let input = try memory.register([1, c, h, w])
        if let latent, memory.isLive { input.values.update(from: latent, count: input.count) }
        let projected = try bands.whole([input], input.shape, weights: WeightList.conv(c, c, kernel: 1),
                                        workspace: VAELayout.reformattedWeights(c * c) / 4) { b, p in
            try b.conv(p[0], "post_quant_conv", padding: 0)
        }
        memory.free(input)
        var x = try bands.conv(projected, "decoder.conv_in", outChannels: levels[0])
        memory.free(projected)
        x = try bands.resnet(x, "decoder.mid_block.resnets.0", inChannels: levels[0], outChannels: levels[0])
        try bands.attention(x, "decoder.mid_block.attentions.0")
        x = try bands.resnet(x, "decoder.mid_block.resnets.1", inChannels: levels[0], outChannels: levels[0])
        for r in 0..<3 {
            x = try bands.resnet(x, "decoder.up_blocks.0.resnets.\(r)", inChannels: levels[0], outChannels: levels[0])
        }
        for level in 1..<levels.count {
            let (entry, exit) = (levels[level - 1], levels[level])
            let doubled = try bands.upsample(x, "decoder.up_blocks.\(level - 1).upsamplers.0")
            memory.free(x)
            x = doubled
            for r in 0..<3 {
                x = try bands.resnet(x, "decoder.up_blocks.\(level).resnets.\(r)", inChannels: r == 0 ? entry : exit,
                                     outChannels: exit)
            }
        }
        // The pipeline's `.clamp(-1, 1)`: on the host, a min/max does not round.
        let image = try bands.normConv(x, norm: "decoder.conv_norm_out", conv: "decoder.conv_out", outChannels: 3,
                                       residual: nil, into: nil)
        memory.free(x)
        memory.host(4 * image.count)
        var pixels: [Float] = []
        if memory.isLive {
            pixels = Array(UnsafeBufferPointer(start: image.values, count: image.count))
            for i in pixels.indices { pixels[i] = min(1, max(-1, pixels[i])) }
        }
        memory.free(image)
        return pixels
    }

    /// **Everything a decoding holds** for a `[32, h, w]` decoder grid (the image is `8h × 8w`): both
    /// heaps and the host copies — `decodeBanded` walked without a device; what `MemoryNeed` charges
    /// (`MemoryNeed.vae`). Peak at 1024²: 1.0 GB registers; see the check for the measure.
    package static func plan(latentHeight h: Int, latentWidth w: Int, config: Config = Config(),
                             band: Int = VAE.bandElements) -> VAELayout {
        plans(VAEPlanKey(config, [h, w, band])) { walk(latentHeight: h, latentWidth: w, config: config, band: band) }
    }

    private static let plans = ShapeMemo<VAEPlanKey<Config>, VAELayout>()

    package static func walk(latentHeight h: Int, latentWidth w: Int, config: Config = Config(),
                             band: Int = VAE.bandElements) -> VAELayout {
        let memory = VAEMemory()
        let bands = VAE.Bands(memory: memory, weights: nil, name: "VAE Flux2", groups: config.groups, band: band)
        _ = try? decodeBanded(nil, height: h, width: w, config: config, bands: bands)
        return memory.layout
    }

    /// The pipeline's part alone on a DiT latent `[128, h/2, w/2]`: de-normalized, then de-patchified, `[32, h, w]`.
    package func pipelinePart(latent: [Float]) throws -> [Float] {
        let (h, w) = (latentHeight / 2, latentWidth / 2)
        guard latent.count == config.patchedChannels * h * w else {
            throw Safetensors.Failure.badHeader("Flux2VAE: a latent of \(latent.count) values, \(config.patchedChannels)·\(h)·\(w) expected")
        }
        return Flux2VAE.unpatchify(Flux2VAE.denormalize(latent, scale: scale, shift: shift, plane: h * w),
                                   channels: config.latentChannels, patch: config.patch, height: h, width: w)
    }

    /// `latent` as the DiT leaves it, `[128, h/2, w/2]`. Outputs `[3, 8h, 8w]` in `[-1, 1]`.
    package func decode(latent: UnsafePointer<Float>) throws -> (image: [Float], seconds: Double) {
        let started = Date()
        let (h, w) = (latentHeight, latentWidth)
        let z = try pipelinePart(latent: Array(UnsafeBufferPointer(start: latent, count: config.patchedChannels * (h / 2) * (w / 2))))
        let band = MemoryPlan.decoderBand(standard: VAE.bandElements, minimum: MemoryPlan.fluxMinimumBand, unit: "elements")
        let memory = try VAEMemory(plan: Flux2VAE.plan(latentHeight: h, latentWidth: w, config: config, band: band),
                                   device: device, queue: queue)
        let image = try autoreleasepool {
            try z.withUnsafeBufferPointer {
                try Flux2VAE.decodeBanded($0.baseAddress!, height: h, width: w, config: config,
                                          bands: VAE.Bands(memory: memory, weights: weights, name: "VAE Flux2",
                                                           groups: config.groups, band: band))
            }
        }
        memory.finish()
        return (image, Date().timeIntervalSince(started))
    }
}
