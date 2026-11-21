import Foundation

/// The Z-Image Turbo DiT, from the latent to the predicted velocity.
///
///     latent [C, H, W] ─ patchify ─ x_embedder ─ noise_refiner ×2 ─┐
///                                                                  ├─ concat ─ layers ×30 ─ FinalLayer ─ unpatchify
///     cap_feats [L, 2560] ─ cap_embedder ─ pad ─ context_refiner ×2┘
///
/// The image comes **first** in the unified sequence. The `context_refiner`s have no
/// modulation. SiLU only exists in `FinalLayer`. The RoPE positions of axis 0 are a segment
/// index: the text occupies `1…L`, the image the value `L+1`.
///
/// **Both sequences are padded to a multiple of 32**, not the caption alone (`_pad_with_ids`): the
/// image's missing rows (16 at 832×1216, none at 512², 1024², 768×512 or 1024×1536) are the last patch
/// repeated through `x_embedder`, then replaced by the learned `x_pad_token`, at RoPE position
/// `(0, 0, 0)`. Nothing masks them at batch 1: they are keys and queries in the noise refiners and the
/// thirty layers, and only the unpatchify drops them. The sequence is `[image, image pad, caption,
/// caption pad]`. A port that pads the caption alone computes another attention on those formats.
///
/// **A LoRA reaches every product of the graph**, the front and the head included (`x_embedder`, `cap_embedder.1`,
/// the final linear, the step embedder): `ZImageDiT.loraModules`.
extension ZImageDiT {
    package struct Config {
        package let dim, heads, hidden, layers, refiners, patch, channels, capDim, adalnDim: Int
        package let eps, theta, timeScale: Float
        package let axesDims: [Int]
        /// Both sequences are padded to a multiple of it (`SEQ_MULTI_OF`).
        package let seqMultiple = 32
        /// The positions the published RoPE table holds on each spatial axis (`axes_lens` `[1536, 512, 512]`): one per
        /// 16 px, so the longest side the family draws (`FamilyTraits.maximumSide`) — past it the table clamps
        /// without a word.
        package static let publishedAxisPositions = 512

        /// **The published model's** (`transformer/config.json` of `Tongyi-MAI/Z-Image-Turbo`): what a weights file of the
        /// family is read with — Comfy-Org's single files carry no configuration.
        package static let published = Config(dim: 3840, heads: 30, hidden: 10240, layers: 30, refiners: 2, patch: 2, channels: 16,
                                              capDim: 2560, adalnDim: 256, eps: 1e-5, theta: 256, timeScale: 1000,
                                              axesDims: [32, 48, 48])
    }
}
