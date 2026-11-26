import Foundation
import SiliCore

/// **The Flux VAE's decoder**: the latent to pixels, as diffusers' `AutoencoderKL`
/// decodes in bf16 (`Decoder`, `UNetMidBlock2D`, `UpDecoderBlock2D`, `ResnetBlock2D`, `AttnProcessor2_0`).
///
///     z [16, h, w] / 0.3611 + 0.1159 ─ conv_in ─ mid (resnet, attention, resnet)
///       ─ up 512 ×3 ↑ ─ up 512 ×3 ↑ ─ up 256 ×3 ↑ ─ up 128 ×3 ─ GroupNorm ─ SiLU ─ conv_out ─ pixels [3, 8h, 8w]
///
///     resnet:  y = shortcut(x) + conv₂(SiLU(GN₂(conv₁(SiLU(GN₁(x))))))       shortcut: a 1×1 where channels change
///     mid attention: one head over every position, scale 1/√C, on GN(x); x + to_out(·)
///
/// What a port by analogy would miss:
///
///   - **The shift factor** (0.1159) after the scale (0.3611) — forgotten more often than the scale — each a bf16
///     operation in the reference on the latent brought to the decoder's dtype (`latents.to(vae.dtype)`,
///     `pipeline_z_image.py`); here both in f32, in the head's function (`FluxVAE`).
///   - **GroupNorm rounds once**: torch's `group_norm` on bf16 computes the statistics *and the affine* in f32 and
///     rounds the result; here the statistics are f32 and the affine is folded into one f16 product and sum on the
///     band (`FluxVAE`), below a bf16 input's error on the same. 32 groups, eps 10⁻⁶.
///   - **Nearest upsampling ×2, then a 3×3 convolution** (diffusers casts to f32 for `interpolate`: exact); written as a
///     broadcast against zeros, exact.
///
/// On the engine: f16 under the guard, the GPU's alone (`FluxVAE`, `FluxVAEBands`). Measured 18/11 on the mini
/// against `check z-image vae 768x512`'s fp64 witness, plane by plane: 2.5e-4 at worst, diffusers in bf16 3.0e-3.
package final class FluxVAEDecoder: LatentDecoder {
    package static let scaling = 0.3611, shift = 0.1159
    /// The VAE as functions: f16 under the guard, the up blocks 512, 512, 256, 128 from the trunk to the picture.
    package static let vae = FluxVAE(name: "flux-vae", half: true, decoderWidths: [512, 512, 256, 128], encoderWidths: [128, 256, 512, 512],
                                     latent: .scaled(channels: 16, scaling: scaling, shift: shift), clamped: false)

    private let pass: FluxVAEPass
    package let narrowed: Bool

    private init(_ pass: FluxVAEPass, narrowed: Bool) { (self.pass, self.narrowed) = (pass, narrowed) }

    package static func open(_ candidates: FluxVAE.Candidates) async throws -> FluxVAEDecoder {
        let (pass, narrowed) = try await vae.open(candidates)
        return FluxVAEDecoder(pass, narrowed: narrowed)
    }

    package static func admit(path: String, latentHeight: Int, latentWidth: Int, budget: Int) throws -> FluxVAE.Candidates {
        let c = try vae.candidates(.decode, path: path, height: latentHeight, width: latentWidth, budget: budget)
        try vae.admit(c)
        return c
    }

    package static func prepare(path: String, latentHeight: Int, latentWidth: Int, budget: Int) throws -> [Preparation] {
        try vae.prepare(.decode, path: path, height: latentHeight, width: latentWidth, budget: budget)
    }

    /// `[16, h, w]` f32 to `[3, 8h, 8w]` f32, around [−1, 1], not clamped.
    package func pixels(_ latent: [Float]) async throws -> [Float] { try await pass.decode(latent) }
}
