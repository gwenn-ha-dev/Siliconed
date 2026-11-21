import Foundation

// **MOSS-Image's modules, and the model that assembles them.**
//
//     MossImageText ── Qwen3.5-9B, last hidden state ──▶ MossImageDenoising ── DiT + the DMD LoRA, Euler, CFG 1 ──▶ Flux2Decoding
//
// The recipe is the community's 8-step one: `MOSS-Image-SFT` under `lodestones`' DMD LoRA at strength 1, one evaluation
// per step, no negative branch. The official one (50 steps, CFG 3, block 22 skipped in the negative branch) is not
// ported.

extension TextFormat {
    package static let mossImage = TextFormat("qwen3.5-9b · chat · last hidden state after norm")
}

extension LatentSpace {
    /// FLUX.2's VAE as MOSS-Image's DiT sees it: the 32 channels patched 2×2 into **128**, a cell of 16 pixels.
    package static let flux2 = LatentSpace(name: "VAE FLUX.2", channels: 128, factor: 16, preview: .flux2)
}

/// The chat template (no system turn, the thinking block open), the Qwen2 tokenizer, Qwen3.5-9B: the last hidden
/// state after the final norm, every token of the template kept, one unpadded forward.
package struct MossImageTextModule: TextModule {
    package var map: String
    /// The `tokenizer/` folder of the published repository (its `tokenizer.json`).
    package var tokenizer: String
    package var output: TextFormat { .mossImage }
    package var name: String { "Qwen3.5-9B" }

    package init(map: String, tokenizer: String) { self.map = map; self.tokenizer = tokenizer }
}

/// The DiT under the family's LoRA — **always**: the 8-step recipe is the DMD LoRA at strength 1, applied beside the
/// products (`MossImageDiTProgram`), never merged into the bf16 weights.
package struct MossImageDenoisingModule: DenoisingModule {
    package var map: String
    /// The LoRA's file as published, in the family's components.
    package var turbo: String
    package var model: String { MossImagePublished.model }
    package var name: String { "DiT MOSS-Image" }
    package var entry: TextFormat { .mossImage }
    package var space: LatentSpace { .flux2 }
    package var defaultSteps: Int { 8 }
    /// Its seventh evaluation alone, σ 0.463: the eighth is at σ 0.024, where a lie changes nothing (`Detail.Bell`).
    package var detailBell: Detail.Bell? { Detail.Bell(start: 0.857, end: 1, bias: 0, more: 2, most: 3, followsSide: false) }

    package init(map: String, turbo: String) { self.map = map; self.turbo = turbo }

    /// Shift 5 on the 1000-step grid, rounded to it: at most `steps` σ, followed by the final 0.
    package func sigmas(steps: Int) -> [Float] { MossImageSchedule.sigmas(steps: steps) }

    /// The recipe as published: one evaluation per σ the schedule keeps, from the start to the end.
    package func plan(height: Int, width: Int, steps: Int, start: Int) -> DenoisingPlan {
        let σ = sigmas(steps: steps)
        return DenoisingPlan(startStep: start, startSigma: σ[start], evaluations: σ.count - 1 - start)
    }
}

/// **FLUX.2's encoder**, img2img's input on MOSS-Image: Flux's encoder, `quant_conv`, then the pipeline's patchify and
/// batch-norm normalization (`Flux2VAEEncoder`) — a `[128, H/16, W/16]` latent in the DiT's space.
package struct Flux2EncodingModule: ImageEncodingModule {
    package var path: String
    package var space: LatentSpace { .flux2 }
    package var name: String { "VAE FLUX.2" }
    package init(path: String) { self.path = path }
}

/// FLUX.2's decoder on the DiT's latent: the batch-norm denormalization, the 2×2 unpatchify, the decoder, the clamp.
package struct Flux2DecodingModule: DecodingModule {
    package var path: String
    package var space: LatentSpace { .flux2 }
    package var name: String { "VAE FLUX.2" }
    package init(path: String) { self.path = path }
}

// ── What the app reads ──────────────────────────────────────────────────────────────────────

/// Where MOSS-Image is published, at the commits an installation takes, and what its files are named by.
package enum MossImagePublished {
    /// The model's identifier.
    package static let model = "moss-image"
    /// The publisher's repository: the tokenizer, the VAE, the license.
    package static let openMOSS = Repository("OpenMOSS-Team/MOSS-Image-SFT", "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9")
    /// Comfy-Org's single files of the SFT's DiT and of its Qwen3.5-9B encoder: bf16, and int8 `convrot`.
    package static let comfy = Repository("Comfy-Org/MOSS-Image", "2929efe3cdd506c0a430bcfca70a9a8d805cc095")
    /// lodestones' 8-step DMD distillation, a LoRA.
    package static let dmd = Repository("lodestones/MOSS-Image-SFT-8-steps-dmd", "f95df83b5e90ae91ea412434b3f9b7f1c8c6314f")
    /// The family's own LoRA — the 8-step DMD distillation at strength 1 —: the published file, in the family's
    /// components folder under the repository's name.
    package static let turbo = "MOSS-Image-SFT-8-steps-dmd/lora.safetensors"

    /// The license and the LoRA's card, pinned.
    package static let licenses = [openMOSS.page("LICENSE"), dmd.page("README.md")]

    /// What both versions read: the tokenizer, the VAE, and the LoRA that makes the 8 steps.
    static let shared: [PublishedFile] = [
        PublishedFile(openMOSS, "tokenizer/tokenizer.json", at: "composants/moss-image/tokenizer/tokenizer.json",
                      bytes: 19_989_325, sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
        PublishedFile(openMOSS, "vae/diffusion_pytorch_model.safetensors", at: "composants/moss-image/vae.safetensors",
                      bytes: 249_521_372, sha256: "c3bdf94dc67ec69b30ea9fe1fa8340085679cae64d019f7c7e4a9921f1a7f17a"),
        PublishedFile(dmd, "lora.safetensors", at: "composants/moss-image/MOSS-Image-SFT-8-steps-dmd/lora.safetensors",
                      bytes: 1_189_176_800, sha256: "4abb14e39c1a59018907581e609f43616ec6810126144ec2b1d2e6f8969628fe"),
    ]
    package static let files: [Variant: [PublishedFile]] = [
        .standard: shared + [
            PublishedFile(comfy, "diffusion_models/moss_image_sft_bf16.safetensors", at: "moss_image_sft_bf16.safetensors",
                          bytes: 10_164_655_808, sha256: "f96738d8116d78b7316b57c663d8c98591ddafab7488f48e3099b7655e34ab6f"),
            PublishedFile(comfy, "text_encoders/qwen3.5_9b_bf16.safetensors", at: "qwen3.5_9b_bf16.safetensors",
                          bytes: 19_306_312_328, sha256: "7e6e9f08d598f829cb940e60ac0c698e1f1c27a47daffd7e598cd78c78b4cc53"),
        ],
        .compact: shared + [
            PublishedFile(comfy, "diffusion_models/moss_image_sft_int8_convrot.safetensors", at: "moss_image_sft_int8_convrot.safetensors",
                          bytes: 5_413_698_958, sha256: "b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f"),
            PublishedFile(comfy, "text_encoders/qwen3.5_9b_int8_convrot.safetensors", at: "qwen3.5_9b_int8_convrot.safetensors",
                          bytes: 9_748_121_532, sha256: "32e3787472f708fa5f83624acbf5ef23f7fc89c439e05065ea57cfe5a97f025a"),
        ],
    ]
}

extension ModelCard {
    /// **MOSS-Image under the community's 8-step LoRA** (OpenMOSS-Team's SFT model, lodestones' DMD distillation at
    /// strength 1). Apache 2.0.
    package static let mossImage = ModelCard(
        id: "moss-image", name: "MOSS-Image", family: .mossImage, map: nil,
        license: License(text: "Apache 2.0", commercial: true, requiredFilter: false, urls: MossImagePublished.licenses),
        defaultSteps: 8, space: .flux2, formats: recommendedFormats)
}

extension FamilyTraits {
    package static let mossImage = FamilyTraits(
        name: "MOSS-Image", ditKind: MossImageDiT.kind,
        ditMaps: [.standard: "moss_image_sft_bf16.safetensors", .compact: "moss_image_sft_int8_convrot.safetensors"],
        encoderMaps: [.standard: "qwen3.5_9b_bf16.safetensors", .compact: "qwen3.5_9b_int8_convrot.safetensors"],
        files: MossImagePublished.files,
        card: .mossImage,
        presentation: .mossImage,
        writerStyle: .mossImage,
        civitaiModelVersion: 3_409_867,
        // The 8 steps are lodestones' DMD LoRA (`MossImagePublished.turbo`), byte for byte tsolful's
        // "Moss_SFT_8_steps_dmd_v0.1" (`4abb14e39c1a…`). Without it, a Remix runs the SFT
        // alone at 8 steps and CFG 1, and gets mush.
        civitaiRecipeLoRA: 3_411_385,
        isBeta: true,
        startStrength: 0.80,
        loraModules: { MossImageDiTProgram.loraModules() },
        ditTensors: { MossImageDiTProgram.tensors() },
        model: { try Model.mossImage(in: $0, card: $1, dit: $2) },
        render: { try MossImageRender.render($0, $1, seeds: $2, model: $3, cancellation: $4, emit: $5) })
}

extension Model {
    /// MOSS-Image under its 8-step LoRA. Apache 2.0.
    package static func mossImage(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.mossImage.id) { try mossImage(in: b, card: .mossImage, dit: b.ditMap(.mossImage)) }
    }

    /// `encoder`: the text encoder's file when it is not the installed variant's (a check that judges one variant).
    package static func mossImage(in b: Library, card: ModelCard, dit: String, encoder: String? = nil) throws -> Model {
        // The LoRA lives with the components; without it the DiT would render an undistilled image in 8 steps.
        let turbo = try b.component(.mossImage, MossImagePublished.turbo)
        return Model(card: card, library: b, chain: try Chain(
            text: MossImageTextModule(map: try map(encoder ?? b.encoderMap(.mossImage), b), tokenizer: try b.component(.mossImage, "tokenizer")),
            denoising: MossImageDenoisingModule(map: try map(dit, b), turbo: turbo),
            decoding: Flux2DecodingModule(path: try b.component(.mossImage, "vae.safetensors")),
            encoding: Flux2EncodingModule(path: try b.component(.mossImage, "vae.safetensors"))))
    }
}
