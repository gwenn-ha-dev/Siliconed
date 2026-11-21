import Foundation
import SiliCore

/// **The Flux VAE's encoder**: a picture to the mean of its latent, as diffusers'
/// `AutoencoderKL.encode` runs it in bf16 (`Encoder`, `DownEncoderBlock2D`, `Downsample2D`).
///
///     image [3, H, W] ─ conv_in (3 → 128) ─ 4 blocks  128 ↓ → 256 ↓ → 512 ↓ → 512
///       ─ mid (resnet, attention, resnet) ─ GroupNorm ─ SiLU ─ conv_out (512 → 32) ─ the first 16 channels
///
///     block:  ↓(resnet₁(resnet₀(x)))        ↓: zero pad right and bottom, 3×3 at stride 2; the last block has none
///
/// The resnet, the group norm, the bottleneck attention and the convolution are the decoder's (`FluxVAEDecoder`). What
/// sets the encoder apart, and what a port by analogy would miss:
///
///   - **The downsampling pads one side only** (`F.pad(x, (0, 1, 0, 1))`), then convolves without padding.
///   - **No `quant_conv`**: this VAE publishes none (`use_quant_conv: false`); `conv_out`'s 32 channels are the moments.
///   - **The latent is the distribution's mode**, the first 16 of the 32 channels: the pipeline draws a sample, the
///     engine takes the mean, deterministic.
///   - **The shift before the scale**, `(z − 0.1159) · 0.3611`, each a bf16 operation (`prepare_latents`).
///
/// On the engine: the decoder's functions and arithmetic (`FluxVAE`), every level by bands, the last whole.
package enum FluxVAEEncoder {
    package static let factor = 8, latentChannels = 16

    /// **`(z − shift) · scaling` in bf16**: the two factors bf16, the difference and the product each rounded.
    package static func normalized(_ latent: [Float]) -> [Float] {
        let shift = BFloat.round(Float(FluxVAEDecoder.shift)), scale = BFloat.round(Float(FluxVAEDecoder.scaling))
        return latent.map { BFloat.round(BFloat.round(BFloat.round($0) - shift) * scale) }
    }

    /// **A picture's latent as the denoiser reads it**, `[16, H/8, W/8]`: the encoder's mean, `normalized`. `image` is
    /// `[3, H, W]` in [−1, 1], each side a multiple of 8.
    package static func latent(of image: ImageRGB, path: String) throws -> [Float] {
        let budget = MemoryBudget.machineCeiling()
        let mean = try blocking {
            let (pass, _) = try await FluxVAEDecoder.vae.open(.encode, path: path, height: image.height, width: image.width, budget: budget)
            defer { pass.stage.close() }
            return try await pass.encode(image.pixels)
        }
        return normalized(mean)
    }
}
