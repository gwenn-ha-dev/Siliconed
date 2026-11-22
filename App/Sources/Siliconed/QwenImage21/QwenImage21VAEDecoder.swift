import Foundation
import SiliCore

/// **Qwen-Image-2.1's VAE decoder**: the de-normalized latent to RGBA pixels, as diffusers'
/// `AutoencoderKLQwenImage21._decode` runs it in bf16 (`autoencoder_kl_qwenimage21.py`).
///
///     z [64, h, w] ─ post_quant_conv ─ conv_in (64 → 1152) ─ mid (resnet, attention, resnet)
///       ─ 5 up blocks  1152 → 1152 ↑ → 1152 ↑ → 576 ↑ → 288 ↑ → 144 ─ RMS ─ SiLU ─ conv_out (144 → 4) ─ clamp ─ [4, 16h, 16w]
///
///     up block:  y = conv(↑(resnet₂(resnet₁(resnet₀(x))))) + shortcut(x)        ↑: nearest ×2, the conv keeps the channels
///
/// The resnet, the RMS norm, the bottleneck attention and the rounding of a convolution are Qwen-Image's
/// (`QwenImageVAEDecoder`, the same Wan family). What sets this one apart, and what a port by analogy would miss:
///
///   - **The reference itself is 2-D**: `QwenImage21CausalConv3d` is an `nn.Conv2d` that folds the single frame away; the
///     published kernels are `[O, I, k, k]`. `upsampler.time_conv` is published and never read on one frame.
///   - **16× and 64 channels, a decoder 144 wide** (`decoder_base_dim`; the encoder is 96): `144 × [8, 8, 8, 4, 2, 1]`.
///   - **The upsampler keeps its channels** (`upsample_out_dim = out_dim`), and it is `upsampler.resample.1`, singular.
///   - **Every upsampling block adds a shortcut without parameters from its input** (`DupUp3D`, `first_chunk`), whose
///     temporal factor still acts on one frame: of the repeated channels the view keeps the *last* time slot. Output
///     channel `o` at `(2y + i, 2x + j)` reads input channel `⌊(4·ft·o + 4·(ft − 1) + 2i + j) / r⌋` at `(y, x)`,
///     `r = 4·ft·out / in`: `o` for blocks 0 and 1, **`2o + 1`** for block 2, **`2o + i`** for block 3 — its rows
///     alternate between two channels (`shortcut`). The last block has none.
///   - **RGBA out**: four planes; the pipeline hands back an RGBA picture and the render keeps its RGB, uncomposited.
///   - **The de-normalization is `latents · std + mean` in bf16**, each operation rounded (`denormalized`), on the
///     latent's 64 channels as they are — the packing is a plain flatten, no 2 × 2.
///
/// As functions of the new engine (`QwenImageVAE`): the head whole (post_quant_conv, conv_in, the mid block), every
/// layer past it by bands of rows, in f16 under the guard; each upsampling block's input keeps its slab until its ×2
/// reads it again for the shortcut.
package final class QwenImage21VAEDecoder: BandedDecoder {
    package static let latentChannels = 64, factor = 16

    /// The blocks: channels in and out, and the shortcut's temporal factor where the block upsamples.
    static let blocks: [(input: Int, output: Int, temporal: Int?)] = [
        (1152, 1152, 2), (1152, 1152, 2), (1152, 576, 2), (576, 288, 1), (288, 144, nil)]

    /// **The run of a latent `h × w`**: the head, each block's three resnets and its ×2 with the DupUp3D of its input,
    /// the head out to RGBA, by bands of at most `positions`.
    static func plan(h: Int, w: Int, positions: Int) -> QwenImageVAE.Plan {
        QwenImageVAE.Plan.surveyed("qwen-image-2.1-vae-decoder", positions: positions) { p in
            let trunk = blocks[0].input
            var x = p.head([.conv("post_quant_conv", cin: latentChannels, cout: latentChannels, k: 1),
                            .conv("decoder.conv_in", cin: latentChannels, cout: trunk, k: 3),
                            .resnet("decoder.mid_block.resnets.0", cin: trunk, cout: trunk),
                            .attention("decoder.mid_block.attentions.0", c: trunk),
                            .resnet("decoder.mid_block.resnets.1", cin: trunk, cout: trunk)],
                           latent: latentChannels, h: h, w: w, channels: trunk)
            for (b, block) in blocks.enumerated() {
                let entry = x
                for r in 0..<3 {
                    x = p.resnet(x, layer: "decoder.up_blocks.\(b).resnets.\(r)", channels: block.output, keeping: r == 0 && block.temporal != nil)
                }
                if let temporal = block.temporal {
                    x = p.up(x, key: "decoder.up_blocks.\(b).upsampler.resample.1", channels: block.output, dup: (entry, temporal))
                    p.release(entry)
                }
            }
            p.out(x, layer: "decoder", planes: 4)
        }
    }

    let run: QwenImageVAERun

    private init(_ run: QwenImageVAERun) { self.run = run }

    /// The decoder of a latent `latentHeight × latentWidth` from the VAE at `path`: its stage planned under `budget`
    /// (the tallest bands that hold), its arena filled, its functions loaded — the GPU's alone.
    package static func open(path: String, latentHeight: Int, latentWidth: Int, budget: Int) async throws -> QwenImage21VAEDecoder {
        QwenImage21VAEDecoder(try await QwenImageVAERun.open("qwen-image-2.1-vae-decoder", file: try SafeTensors(path), budget: budget) { plan(h: latentHeight, w: latentWidth, positions: $0) })
    }

    package static func admit(path: String, latentHeight: Int, latentWidth: Int, budget: Int) throws {
        try QwenImageVAERun.admit("qwen-image-2.1-vae-decoder", file: try SafeTensors(path), budget: budget) { plan(h: latentHeight, w: latentWidth, positions: $0) }
    }

    /// Whether B made the bands smaller than the run prefers.
    package var narrowed: Bool { run.bands.narrowed }

    /// The positions its bands hold at most, and its stage: its plan's P, and its close (`vae-bands`).
    package var positions: Int { run.plan.positions }
    package var stage: SiliCore.Stage { run.stage }

    /// A latent `[64, h, w]` already `denormalized` to the pixels `[4, 16h, 16w]` (RGBA) in [−1, 1].
    package func pixels(_ latent: [Float]) async throws -> [Float] {
        try await run.run(latent)
    }

    /// **`latents · std + mean` in bf16**, per channel: the latent brought to bf16 (`latents.to(vae.dtype)`), `std` and
    /// `mean` bf16, the product and the sum each rounded. `[64, h·w]` in, the same out.
    package static func denormalized(_ latent: [Float]) -> [Float] {
        let planeSize = latent.count / latentChannels
        var out = latent
        for c in 0..<latentChannels {
            let scale = BFloat.round(Float(QwenImage21Published.latentsStd[c])), shift = BFloat.round(Float(QwenImage21Published.latentsMean[c]))
            for i in 0..<planeSize {
                out[c * planeSize + i] = BFloat.round(BFloat.round(BFloat.round(latent[c * planeSize + i]) * scale) + shift)
            }
        }
        return out
    }
}
