import Foundation
import SiliCore

/// **The FLUX.2 VAE's decoder (MOSS-Image's), in f32**: the DiT's latent to pixels, as the pipeline's
/// `__call__` and `AutoencoderKLFlux2.decode` run it (diffusers 0.41; the pipeline's own docstring: "VAE decoding defaults to fp32").
///
///     z [128, h, w] ─ · √(running_var + 10⁻⁴) + running_mean ─ unpatchify 2×2 ─ [32, 2h, 2w] ─ post_quant_conv (1×1)
///       ─ conv_in 32→384 ─ mid (resnet, attention, resnet) ─ up 384 ×3 ↑ ─ up 384 ×3 ↑ ─ up 192 ×3 ↑ ─ up 96 ×3
///       ─ GroupNorm ─ SiLU ─ conv_out 96→3 ─ clamp(−1, 1) ─ pixels [3, 16h, 16w]
///
///     resnet:  y = shortcut(x) + conv₂(SiLU(GN₂(conv₁(SiLU(GN₁(x))))))       shortcut: a 1×1 where channels change
///     mid attention: one head over every position, scale 1/√C, on GN(x); x + to_out(·)
///
/// What sets it apart from the Flux VAE (`FluxVAEDecoder`), which a port by analogy would copy:
///
///   - **no scaling factor and shift**: the latent is the DiT's own space, **batch-norm denormalized** (`bn.running_mean`,
///     `bn.running_var`, eps `batch_norm_eps` 10⁻⁴ from `vae.json`) and then **unpatchified** (`reshape(b, c/4, 2, 2, h, w)
///     .permute(0, 1, 4, 2, 5, 3)`), 128 channels at 1/16 resolution becoming 32 at 1/8;
///   - **`post_quant_conv`** (1×1) before the decoder (`use_post_quant_conv`), and the decoder's widths are
///     `decoder_block_out_channels` (96, 192, 384, 384), not the encoder's (128 … 512);
///   - **the output is clamped to [−1, 1]** inside the pipeline (`.clamp(-1, 1)`);
///   - **f32 end to end**: the pipeline decodes in the VAE's own dtype, f32 (`force_upcast`), the published weights are f32; there is no bf16
///     reference of this stage and no reason to make one.
///
/// On the engine: Flux's VAE as functions (`FluxVAE`) in f32, the GPU's alone, every layer past the trunk by bands.
package final class Flux2VAEDecoder: LatentDecoder {
    /// The VAE as functions: f32, the up blocks 384, 384, 192, 96, the latent's 2×2 cells under batch-norm statistics.
    package static let vae = FluxVAE(name: "flux2-vae", half: false, decoderWidths: [384, 384, 192, 96], encoderWidths: [128, 256, 512, 512],
                                     latent: .patched(channels: 32, eps: 1e-4), clamped: true)

    private let pass: FluxVAEPass
    package let narrowed: Bool

    private init(_ pass: FluxVAEPass, narrowed: Bool) { (self.pass, self.narrowed) = (pass, narrowed) }

    package static func open(path: String, latentHeight: Int, latentWidth: Int, budget: Int) async throws -> Flux2VAEDecoder {
        let (pass, narrowed) = try await vae.open(.decode, path: path, height: latentHeight, width: latentWidth, budget: budget)
        return Flux2VAEDecoder(pass, narrowed: narrowed)
    }

    package static func admit(path: String, latentHeight: Int, latentWidth: Int, budget: Int) throws {
        try vae.admit(.decode, path: path, height: latentHeight, width: latentWidth, budget: budget)
    }

    /// `[128, h, w]` f32 to `[3, 16h, 16w]` f32, clamped to [−1, 1].
    package func pixels(_ latent: [Float]) async throws -> [Float] { try await pass.decode(latent) }

    /// **The boundaries the oracle writes** that the decoder hands out (`FluxVAEPlan.marks`): the trunk's end and the
    /// first two ×2's outputs. The oracle's `post_quant_conv_out` and `decoder_conv_in` lie inside the trunk's function.
    package static let tapNames = ["decoder_mid", "decoder_up0", "decoder_up1"]

    /// The same, with the boundaries `taps` names (`tapNames`), each `[C, H, W]` f32.
    package func pixels(_ latent: [Float], taps: Set<String>) async throws -> (pixels: [Float], taps: [String: [Float]]) {
        let (pixels, kept) = try await pass.decode(latent, keeping: taps)
        return (pixels, kept)
    }
}
