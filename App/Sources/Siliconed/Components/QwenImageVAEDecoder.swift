import Foundation
import SiliCore

/// **The Qwen-Image VAE's decoder, for one frame**: the normalized latent to
/// pixels, as diffusers' `AutoencoderKLQwenImage._decode` runs it in bf16 (`autoencoder_kl_qwenimage.py`).
///
///     z [16, h, w] ─ post_quant_conv ─ conv_in ─ mid (resnet, attention, resnet)
///       ─ up 384 ×3 ↑ ─ up 384 ×3 ↑ ─ up 192 ×3 ↑ ─ up 96 ×3 ─ RMS ─ SiLU ─ conv_out ─ clamp ─ pixels [3, 8h, 8w]
///
///     resnet:  y = shortcut(x) + conv₂(SiLU(RMS₂(conv₁(SiLU(RMS₁(x))))))      shortcut: a 1×1×1 where the channels change
///     mid attention: one head over every position, scale 1/√384, on RMS(x); x + proj(·)
///     ↑: nearest ×2, then a 2-D 3×3 convolution that halves the channels
///
/// What a port by analogy would miss (read in `autoencoder_kl_qwenimage.py`):
///
///   - **One frame through a causal 3-D VAE is a 2-D one**: `QwenImageCausalConv3d` pads two frames of zeros *before* the
///     frame, so a 3×3×3 kernel sees real data only in its last time slice; the program convolves with `weight[:, :, 2]`. The
///     "upsample3d" blocks' `time_conv` is skipped on the first frame (`feat_cache[idx] = "Rep"`): the frame count stays 1,
///     and `up_blocks.*.upsamplers.0.time_conv` is never read. `post_quant_conv` is a 1×1×1 convolution with a bias.
///   - **`QwenImageRMS_norm` is `F.normalize`**: in f32, `x / max(‖x‖₂ over the channels, 10⁻¹²)`, rounded to bf16 — *then* `× √C`
///     (a second rounding), `× γ` (a third). There is no ε in the variance and no GroupNorm; `γ` is `[C, 1, 1(, 1)]`.
///   - **The decoder output is clamped to [−1, 1]** (`torch.clamp` in `_decode`), unlike the Flux VAE's.
///   - **The normalization is outside the decoder, per channel and in bf16** (the pipelines' decoder step): `latents / (1/std) + mean`,
///     with `1/std` itself a bf16 quotient; `normalized(_:)` does it on the CPU as torch does, rounding after each operation.
///   - **A convolution sums in f32 with its bias inside and rounds once** — the bar the oracle sets
///     (`common.bf16_convolutions_in_fp32`). Here an f16 convolution under the guard: its result times its factor in
///     f32, the bias added in f32 (`QwenImageVAEFunctions`).
///   - **Attention is `F.scaled_dot_product_attention` on `[1, 1, h·w, 384]`**, q, k, v the three chunks of one `to_qkv`.
///
/// As functions of the new engine (`QwenImageVAE`): the head whole (post_quant_conv, conv_in, the mid block), every
/// layer past it by bands of rows, in f16 under the guard — closer to the fp64 witness than diffusers' bf16, whose
/// roundings the former engine copied one by one.
package final class QwenImageVAEDecoder: BandedDecoder {
    package static let latentChannels = 16, factor = 8

    /// The Qwen-Image VAE's `vae/config.json` (`latents_mean`, `latents_std`), as a diffusers repository of it publishes it.
    package static let mean: [Float] = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                                        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    package static let std: [Float] = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                                       3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]

    /// **`latents / (1 / std) + mean` in bf16**, per channel: the latent brought to bf16 (`latents.to(vae.dtype)`), `1 / std`
    /// a bf16 quotient of the bf16 `std`, the division and the sum each rounded to bf16. `[16, h·w]` in, the same out.
    package static func normalized(_ latent: [Float]) -> [Float] {
        let planeSize = latent.count / 16
        var out = latent
        for c in 0..<16 {
            let inverse = BFloat.round(1 / BFloat.round(std[c])), shift = BFloat.round(mean[c])
            for i in 0..<planeSize {
                let x = BFloat.round(latent[c * planeSize + i])
                out[c * planeSize + i] = BFloat.round(BFloat.round(x / inverse) + shift)
            }
        }
        return out
    }

    /// The up blocks' widths, from the trunk to the picture.
    static let blocks = [384, 384, 192, 96]

    /// **The run of a latent `h × w`**: the head, the up blocks' resnets and ×2 (each halving the channels), the head
    /// out, by bands of at most `positions`.
    static func plan(h: Int, w: Int, positions: Int) -> QwenImageVAE.Plan {
        QwenImageVAE.Plan.surveyed("qwen-image-vae-decoder", positions: positions) { p in
            let trunk = blocks[0]
            var x = p.head([.conv("post_quant_conv", cin: latentChannels, cout: latentChannels, k: 1),
                            .conv("decoder.conv_in", cin: latentChannels, cout: trunk, k: 3),
                            .resnet("decoder.mid_block.resnets.0", cin: trunk, cout: trunk),
                            .attention("decoder.mid_block.attentions.0", c: trunk),
                            .resnet("decoder.mid_block.resnets.1", cin: trunk, cout: trunk)],
                           latent: latentChannels, h: h, w: w, channels: trunk)
            for (b, out) in blocks.enumerated() {
                for r in 0..<3 { x = p.resnet(x, layer: "decoder.up_blocks.\(b).resnets.\(r)", channels: out) }
                if b < blocks.count - 1 { x = p.up(x, key: "decoder.up_blocks.\(b).upsamplers.0.resample.1", channels: out / 2) }
            }
            p.out(x, layer: "decoder", planes: 3)
        }
    }

    let run: QwenImageVAERun

    private init(_ run: QwenImageVAERun) { self.run = run }

    /// The decoder of a latent `latentHeight × latentWidth` from the VAE at `path`: its stage planned under `budget`
    /// (the tallest bands that hold), its arena filled, its functions loaded — the GPU's alone.
    package static func open(path: String, latentHeight: Int, latentWidth: Int, budget: Int) async throws -> QwenImageVAEDecoder {
        QwenImageVAEDecoder(try await QwenImageVAERun.open("qwen-image-vae-decoder", file: try SafeTensors(path), budget: budget) { plan(h: latentHeight, w: latentWidth, positions: $0) })
    }

    package static func admit(path: String, latentHeight: Int, latentWidth: Int, budget: Int) throws {
        try QwenImageVAERun.admit("qwen-image-vae-decoder", file: try SafeTensors(path), budget: budget) { plan(h: latentHeight, w: latentWidth, positions: $0) }
    }

    /// Whether B made the bands smaller than the run prefers.
    package var narrowed: Bool { run.bands.narrowed }

    /// The positions its bands hold at most, and its stage: its plan's P, and its close (`vae-bands`).
    package var positions: Int { run.plan.positions }
    package var stage: SiliCore.Stage { run.stage }

    /// A latent `[16, h, w]` already `normalized` to the pixels `[3, 8h, 8w]` in [−1, 1].
    package func pixels(_ latent: [Float]) async throws -> [Float] {
        try await run.run(latent)
    }
}
