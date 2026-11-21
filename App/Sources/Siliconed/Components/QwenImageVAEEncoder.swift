import Foundation
import SiliCore

/// **The Qwen-Image VAE's encoder** (the same weights, whichever file holds them): a picture
/// to the mean of its latent, as diffusers' `AutoencoderKLQwenImage._encode` runs it in bf16 on one frame
/// (`autoencoder_kl_qwenimage.py`, `QwenImageEncoder3d`).
///
///     image [3, H, W] ─ conv_in (3 → 96) ─ 4 stages  96 → 96 ↓ → 192 ↓ → 384 ↓ → 384
///       ─ mid (resnet, attention, resnet) ─ RMS ─ SiLU ─ conv_out (384 → 32) ─ quant_conv ─ the first 16 channels
///
///     stage:  ↓(resnet₁(resnet₀(x)))        ↓: zero pad right and bottom, 3×3 at stride 2; the last stage has none
///
/// The resnet, the RMS norm, the bottleneck attention and the rounding of a convolution are the decoder's
/// (`QwenImageVAEDecoder`). What sets the encoder apart, and what a port by analogy would miss:
///
///   - **`down_blocks` is one flat list**: two resnets then the downsampler, so the stages' modules are numbered
///     0 1 (2) · 3 4 (5) · 6 7 (8) · 9 10, the downsamplers in brackets.
///   - **The downsampling pads one side only** (`ZeroPad2d((0, 1, 0, 1))`), then convolves without padding; its kernel
///     is stored 2-D (`resample.1`), the others 3-D and causal, read at their last time slice on one frame.
///   - **`time_conv` is published and never read on one frame** (`downsample3d` keeps the first frame as it is).
///   - **The latent is the distribution's mode**, the first 16 of the 32 channels (`sample_mode="argmax"`): no noise.
///   - **The normalization is `(z − mean) / std` in bf16**, each operation rounded (`normalized`).
///
/// As functions of the new engine (`QwenImageVAE`): conv_in and the first three stages by bands of rows, the last stage,
/// the mid block and the head out whole at the latent's size (`tail`), in f16 under the guard.
package enum QwenImageVAEEncoder {
    package static let factor = 8, latentChannels = 16
    /// The down path's widths.
    static let dims = [96, 96, 192, 384, 384]

    /// **`(z − mean) / std` in bf16**, per channel: `mean` and `std` bf16, the difference and the quotient each
    /// rounded (`encode_vae_image`). `[16, h·w]` in (bf16 values), the same out.
    package static func normalized(_ latent: [Float]) -> [Float] {
        let planeSize = latent.count / latentChannels
        var out = latent
        for c in 0..<latentChannels {
            let scale = BFloat.round(QwenImageVAEDecoder.std[c]), shift = BFloat.round(QwenImageVAEDecoder.mean[c])
            for i in 0..<planeSize {
                out[c * planeSize + i] = BFloat.round(BFloat.round(BFloat.round(latent[c * planeSize + i]) - shift) / scale)
            }
        }
        return out
    }

    /// **The run of a picture `height × width`** (multiples of 8): conv_in, three stages of two resnets and a ↓ by
    /// bands of at most `positions`, then the tail whole: the last stage, the mid block, RMS · SiLU · conv_out,
    /// quant_conv, the mean's 16 channels.
    static func plan(height: Int, width: Int, positions: Int) -> QwenImageVAE.Plan {
        QwenImageVAE.Plan.surveyed("qwen-image-vae-encoder", positions: positions) { p in
            var x = p.first(p.picture(c: 3, h: height, w: width), key: "encoder.conv_in", channels: dims[0])
            var module = 0
            for stage in 0..<3 {
                x = p.resnet(x, layer: "encoder.down_blocks.\(module)", channels: dims[stage + 1])
                x = p.resnet(x, layer: "encoder.down_blocks.\(module + 1)", channels: dims[stage + 1])
                x = p.down(x, key: "encoder.down_blocks.\(module + 2).resample.1")
                module += 3
            }
            let c = dims[4]
            p.tail([.resnet("encoder.down_blocks.\(module)", cin: dims[3], cout: c), .resnet("encoder.down_blocks.\(module + 1)", cin: c, cout: c),
                    .resnet("encoder.mid_block.resnets.0", cin: c, cout: c), .attention("encoder.mid_block.attentions.0", c: c),
                    .resnet("encoder.mid_block.resnets.1", cin: c, cout: c), .activate("encoder.norm_out.gamma", c: c),
                    .conv("encoder.conv_out", cin: c, cout: 2 * latentChannels, k: 3), .conv("quant_conv", cin: 2 * latentChannels, cout: 2 * latentChannels, k: 1),
                    .take(latentChannels)], x, channels: latentChannels)
        }
    }

    /// **The mean of a picture's latent**, `[16, H/8, W/8]`, not yet `normalized`: the encoder run once from the VAE at
    /// `path` under the machine's B, let go. `image` is `[3, H, W]` in [−1, 1].
    package static func mean(of image: [Float], height: Int, width: Int, path: String) throws -> [Float] {
        let file = try SafeTensors(path), budget = MemoryBudget.machineCeiling()
        return try blocking {
            let run = try await QwenImageVAERun.open("qwen-image-vae-encoder", file: file, budget: budget) { plan(height: height, width: width, positions: $0) }
            let mean = try await run.run(image)
            run.stage.close()
            return mean
        }
    }

    /// **A picture's latent as the denoiser reads it**, `[16, H/8, W/8]`: the encoder's mean, `normalized`. `image` is
    /// `[3, H, W]` in [−1, 1].
    package static func latent(of image: ImageRGB, path: String) throws -> [Float] {
        normalized(try mean(of: image.pixels, height: image.height, width: image.width, path: path))
    }
}
