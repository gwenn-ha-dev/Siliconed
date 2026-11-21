import Foundation
import SiliCore

/// **Qwen-Image-2.1's VAE encoder**: a reference picture, RGBA, to the mean of its latent, as
/// diffusers' `AutoencoderKLQwenImage21._encode` runs it in bf16 on one frame (`autoencoder_kl_qwenimage21.py`).
///
///     image [4, H, W] ─ conv_in (4 → 96) ─ 5 down blocks  96 → 96 ↓ → 192 ↓ → 384 ↓ → 768 ↓ → 768
///       ─ mid (resnet, attention, resnet) ─ RMS ─ SiLU ─ conv_out (768 → 128) ─ quant_conv ─ the first 64 channels
///
///     down block:  y = ↓(resnet₁(resnet₀(x))) + shortcut(x)        ↓: zero pad right and bottom, 3×3 at stride 2
///
/// The resnet, the RMS norm, the bottleneck attention and the rounding of a convolution are the decoder's
/// (`QwenImageVAEDecoder`). What sets the encoder apart, and what a port by analogy would miss:
///
///   - **The picture has four channels**: an opaque one is given an alpha of +1, a transparent one its own, straight
///     (`QwenImage21References.rgba`). Nothing is composited for the VAE.
///   - **The downsampling pads one side only** (`ZeroPad2d((0, 1, 0, 1))`), then convolves without padding.
///   - **Every block adds a shortcut without parameters** (`AvgDown3D`), whose temporal factor still acts on one frame:
///     the frame is padded with a frame of zeros *before* it, so of each input channel's two time slots the first is
///     zero. Block 0 averages each channel's 2×2; blocks 1 to 3 double the channels — output channel `2c` is **zero**,
///     `2c + 1` the 2×2 average of input channel `c`; the last block, which does not downsample, adds its input.
///   - **`downsampler.time_conv` is published and never read on one frame.**
///   - **The latent is the distribution's mode**, the first 64 of the 128 channels (`sample_mode="argmax"`): no noise.
///   - **The normalization is `(z − mean) / std` in bf16**, each operation rounded (`normalized`).
///
/// As functions of the new engine (`QwenImageVAE`): conv_in and the four downsampling blocks by bands of rows (each
/// block's input keeping its slab until its ↓ reads it again for the shortcut), the last block, the mid block and the
/// head out whole at the latent's size (`tail`), in f16 under the guard.
package enum QwenImage21VAEEncoder {
    /// The down path's widths.
    static let dims = [96, 96, 192, 384, 768, 768]

    /// **The run of a picture `height × width`** (multiples of 16): conv_in, four blocks of two resnets and a ↓ with
    /// the AvgDown3D of their input, by bands of at most `positions`, then the tail whole: the last block and its input
    /// added, the mid block, RMS · SiLU · conv_out, quant_conv, the mean's 64 channels.
    static func plan(height: Int, width: Int, positions: Int) -> QwenImageVAE.Plan {
        QwenImageVAE.Plan.surveyed("qwen-image-2.1-vae-encoder", positions: positions) { p in
            var x = p.first(p.picture(c: 4, h: height, w: width), key: "encoder.conv_in", channels: dims[0])
            for b in 0..<4 {
                let entry = x, output = dims[b + 1]
                x = p.resnet(x, layer: "encoder.down_blocks.\(b).resnets.0", channels: output, keeping: true)
                x = p.resnet(x, layer: "encoder.down_blocks.\(b).resnets.1", channels: output)
                x = p.down(x, key: "encoder.down_blocks.\(b).downsampler.resample.1", avg: (entry, output == 2 * dims[b] ? .doubling : .pool))
                p.release(entry)
            }
            let c = dims[5], channels = QwenImage21VAEDecoder.latentChannels
            p.tail([.keep, .resnet("encoder.down_blocks.4.resnets.0", cin: dims[4], cout: c), .resnet("encoder.down_blocks.4.resnets.1", cin: c, cout: c), .addKept,
                    .resnet("encoder.mid_block.resnets.0", cin: c, cout: c), .attention("encoder.mid_block.attentions.0", c: c),
                    .resnet("encoder.mid_block.resnets.1", cin: c, cout: c), .activate("encoder.norm_out.gamma", c: c),
                    .conv("encoder.conv_out", cin: c, cout: 2 * channels, k: 3), .conv("quant_conv", cin: 2 * channels, cout: 2 * channels, k: 1),
                    .take(channels)], x, channels: channels)
        }
    }

    /// **The mean of a picture's latent**, `[64, H/16, W/16]`, not yet `normalized`: the encoder run once from the VAE at
    /// `path` under the machine's B, let go. `image` is the RGBA `[4, H, W]` in [−1, 1] (`QwenImage21References.rgba`).
    package static func mean(of image: [Float], height: Int, width: Int, path: String) throws -> [Float] {
        let file = try SafeTensors(path), budget = MemoryBudget.machineCeiling()
        return try blocking {
            let run = try await QwenImageVAERun.open("qwen-image-2.1-vae-encoder", file: file, budget: budget) { plan(height: height, width: width, positions: $0) }
            defer { run.stage.close() }
            return try await run.run(image)
        }
    }

    /// **`(z − mean) / std` in bf16**, per channel: `mean` and `std` bf16, the difference and the quotient each
    /// rounded. `[64, h·w]` in (bf16 values), the same out.
    package static func normalized(_ latent: [Float]) -> [Float] {
        let channels = QwenImage21VAEDecoder.latentChannels, planeSize = latent.count / channels
        var out = latent
        for c in 0..<channels {
            let scale = BFloat.round(Float(QwenImage21Published.latentsStd[c])), shift = BFloat.round(Float(QwenImage21Published.latentsMean[c]))
            for i in 0..<planeSize {
                out[c * planeSize + i] = BFloat.round(BFloat.round(BFloat.round(latent[c * planeSize + i]) - shift) / scale)
            }
        }
        return out
    }
}
