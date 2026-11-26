import Foundation
import SiliCore

/// **The FLUX.2 VAE's encoder (MOSS-Image's), in f32**: a picture to the DiT's latent, as
/// `Flux2Pipeline._encode_vae_image` runs it (diffusers 0.41) in the VAE's own dtype.
///
///     image [3, H, W] ─ conv_in (3 → 128) ─ 4 blocks  128 ↓ → 256 ↓ → 512 ↓ → 512
///       ─ mid (resnet, attention, resnet) ─ GroupNorm ─ SiLU ─ conv_out (512 → 64) ─ quant_conv (1×1)
///       ─ the first 32 channels ─ patchify 2×2 ─ [128, H/16, W/16] ─ (· − running_mean) / √(running_var + 10⁻⁴)
///
///     block:  ↓(resnet₁(resnet₀(x)))        ↓: zero pad right and bottom, 3×3 at stride 2; the last block has none
///
/// The resnet, the group norm, the bottleneck attention and the convolution are the decoder's (`Flux2VAEDecoder`). What
/// sets the encoder apart, and what a port by analogy would miss:
///
///   - **The widths are the encoder's own** (128, 256, 512, 512), not the decoder's (96, 192, 384, 384).
///   - **`quant_conv`** (1×1) after `conv_out`, and **the latent is the distribution's mode**: the first 32 of the 64 channels.
///   - **The patchify is the unpatchify's inverse**, channel first: `reshape(c, h, 2, w, 2).permute(0, 2, 4, 1, 3)`, 32
///     channels at 1/8 becoming 128 at 1/16, then the batch-norm statistics the decoder undoes.
///   - **f32 end to end**, as the decoder: the published weights are f32 and the reference encodes in them.
///
/// On the engine: the decoder's functions and arithmetic (`FluxVAE`), every level by bands, the last whole.
package enum Flux2VAEEncoder {
    /// **What `latent` would specialize** for a picture `height × width`, nothing allocated (`BandStage.prepare`).
    package static func prepare(height: Int, width: Int, path: String) throws -> [Preparation] {
        try Flux2VAEDecoder.vae.prepare(.encode, path: path, height: height, width: width, budget: MemoryBudget.machineCeiling())
    }

    /// **A picture's latent as the denoiser reads it**, `[128, H/16, W/16]`, batch-norm normalized. `image` is
    /// `[3, H, W]` in [−1, 1].
    package static func latent(of image: ImageRGB, path: String) throws -> [Float] {
        let budget = MemoryBudget.machineCeiling()
        return try blocking {
            let (pass, _) = try await Flux2VAEDecoder.vae.open(.encode, path: path, height: image.height, width: image.width, budget: budget)
            defer { pass.stage.close() }
            return try await pass.encode(image.pixels)
        }
    }
}
