// Qwen-Image-2.1 as the app reads it: its modules (text, denoising, decoding), its sheet, its traits.

import Foundation

// ── Qwen-Image-2.1 ──────────────────────────────────────────────────────────────────────────

extension TextFormat {
    package static let qwenImage21 = TextFormat("qwen3-vl-8b · raw template · last layer before norm", readsImages: true)
}

extension LatentSpace {
    /// Qwen-Image-2.1's own VAE: 64 channels, a cell of 16 pixels, no packing.
    package static let qwenImage21 = LatentSpace(name: "VAE Qwen-Image-2.1", channels: QwenImage21VAEDecoder.latentChannels,
                                                 factor: QwenImage21VAEDecoder.factor, preview: .qwenImage21)
}

/// The raw template, the Qwen tokenizer, Qwen3-VL-8B: the last layer's output before the final norm, the system
/// turn's rows dropped. **In an edit it reads the references with the prompt**, each at its `<imageN>` slot, through
/// its vision tower — the copy the VAE encodes too (`QwenImage21DenoisingModule.preparedReferences`).
package struct QwenImage21TextModule: ImageTextModule {
    package var map: String
    /// The folder of the published `processor/tokenizer.json`.
    package var tokenizer: String
    package var output: TextFormat { .qwenImage21 }
    package var name: String { "Qwen3-VL-8B" }

    package init(map: String, tokenizer: String) { self.map = map; self.tokenizer = tokenizer }
}

/// **The Qwen-Image-2.1-Turbo DiT**: no CFG, 8 steps on the published `sample_sigmas`; in an edit, up to three
/// references in its sequence.
package struct QwenImage21DenoisingModule: DenoisingModule {
    package var map: String
    /// `output_resolution`: the references' area (`R²`), and the edit format's.
    package var resolution: Int
    package var model: String { QwenImage21Published.model }
    package var name: String { "DiT Qwen-Image-2.1" }
    package var entry: TextFormat { .qwenImage21 }
    package var space: LatentSpace { .qwenImage21 }
    package var defaultSteps: Int { QwenImage21DiT.defaultSteps }

    package var maxReferences: Int { QwenImage21References.maximum }

    package init(map: String, resolution: Int = QwenImage21References.outputResolution) { self.map = map; self.resolution = resolution }

    /// Pillow's Lanczos to `calculate_dimensions(R², w/h)`, each at its own ratio.
    package func preparedReferences(_ images: [ImageRGB]) throws -> [ImageRGB] {
        images.map { QwenImage21References.prepared($0, resolution: resolution) }
    }

    package func editFormat(referenceWidth: Int, referenceHeight: Int) -> (width: Int, height: Int) {
        QwenImage21References.outputSize(referenceWidth: referenceWidth, referenceHeight: referenceHeight, resolution: resolution)
    }

    /// The published schedule, the same at every size. An unsupported count gives it too: `check(steps:)` has
    /// refused that count before.
    package func sigmas(steps: Int) -> [Float] { QwenImage21DiT.sigmas }

    /// The published recipe as it stands: one evaluation per step, from the start to the end.
    package func plan(height: Int, width: Int, steps: Int, start: Int) -> DenoisingPlan {
        DenoisingPlan(startStep: start, startSigma: sigmas(steps: steps)[start], evaluations: steps - start)
    }

    /// Another count has no published schedule; its pipeline has no img2img: the image encoder serves the references.
    package func check(steps: Int, startImage: Bool) throws {
        guard steps == defaultSteps else { throw EngineError.invalidSteps(steps: steps, allowed: [defaultSteps]) }
        if startImage { throw EngineError.imageToImageUnsupported(model: model) }
    }

    package var acceptsStartImage: Bool { false }
    /// Every `Solver`: its sampler walks them all on the published σ (`SolverWalk`).
    package var solvers: [Solver] { Solver.allCases }
    /// The last of its 8 published steps alone, σ 0.415: σ 0.705 still decides the face (`Detail.Bell`).
    package var detailBell: Detail.Bell? { .late }
}

/// Qwen-Image-2.1's decoder: the RGBA it decodes, **alpha dropped without compositing** — what the oracle's
/// `pil.convert("RGB")` writes.
package struct QwenImage21DecodingModule: DecodingModule {
    package var path: String
    package var space: LatentSpace { .qwenImage21 }
    package var name: String { "VAE Qwen-Image-2.1" }
    package init(path: String) { self.path = path }
}

/// Qwen-Image-2.1's encoder, for an edit's references: the picture made **RGBA** (`QwenImage21References.rgba`), the
/// latent's mean, then `(mean − mean_c) / std_c`.
package struct QwenImage21EncodingModule: ImageEncodingModule {
    package var path: String
    package var space: LatentSpace { .qwenImage21 }
    package var name: String { "VAE Qwen-Image-2.1" }
    package init(path: String) { self.path = path }
}

// ── What the app reads ──────────────────────────────────────────────────────────────────────

extension QwenImage21Published {
    /// The model's identifier.
    package static let model = "qwen-image-2.1"
    /// The publisher's repository: the tokenizer, the VAE, the license.
    package static let qwen = Repository("Qwen/Qwen-Image-2.1-Turbo", revision)
    /// Comfy-Org's single files of the Turbo's DiT and of its Qwen3-VL-8B encoder (the language model, the vision
    /// tower and the head in one): bf16, and int8 `convrot`.
    package static let comfy = Repository("Comfy-Org/Qwen-Image-2.1", "aacf31a6d48e707f352ecba714cee0680da3479b")
    /// The license and the model's card, pinned.
    package static let licenses = [qwen.page("LICENSE"), qwen.page("README.md")]

    /// What both versions read: the tokenizer and the VAE.
    static let shared: [PublishedFile] = [
        PublishedFile(qwen, "processor/tokenizer.json", at: "composants/qwen-image-2.1/tokenizer/tokenizer.json",
                      bytes: 11_422_650, sha256: "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"),
        PublishedFile(qwen, "vae/diffusion_pytorch_model.safetensors", at: "composants/qwen-image-2.1/vae.safetensors",
                      bytes: 675_508_656, sha256: "71879ffd5321e6d10c3c87513e2b474b1252efa7f3dec2969214a9bf06a6dd5c"),
    ]
    package static let files: [Variant: [PublishedFile]] = [
        .standard: shared + [
            PublishedFile(comfy, "diffusion_models/qwen_image_2.1_turbo_bf16.safetensors", at: "qwen_image_2.1_turbo_bf16.safetensors",
                          bytes: 14_230_280_616, sha256: "8b74237a7f505773e2c66726fa696444b4c704585c93a347cff9044937f7914a"),
            PublishedFile(comfy, "text_encoders/qwen3vl_8b_bf16.safetensors", at: "qwen3vl_8b_bf16.safetensors",
                          bytes: 17_534_334_616, sha256: "68bdc82bc1b66851162ae656225e7e2068166b603db19bd5d5a3b90eb12669a9"),
        ],
        .compact: shared + [
            PublishedFile(comfy, "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors", at: "qwen_image_2.1_turbo_int8_convrot.safetensors",
                          bytes: 7_256_787_829, sha256: "8144a0182964aa8403ad6dbc011db837304167d235161b9aabaa8532e967314c"),
            PublishedFile(comfy, "text_encoders/qwen3vl_8b_int8_convrot.safetensors", at: "qwen3vl_8b_int8_convrot.safetensors",
                          bytes: 9_350_798_360, sha256: "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f"),
        ],
    ]
}

extension ModelCard {
    /// ⚖️ Qwen Research License: non-commercial.
    package static let qwenImage21 = ModelCard(
        id: QwenImage21Published.model, name: "Qwen-Image-2.1 Turbo", family: .qwenImage21, map: nil,
        license: License(text: "Qwen Research (non-commercial)", commercial: false, requiredFilter: false,
                         urls: QwenImage21Published.licenses),
        defaultSteps: QwenImage21DiT.defaultSteps, space: .qwenImage21, formats: recommendedFormats)
}

extension FamilyTraits {
    package static let qwenImage21 = FamilyTraits(
        name: "Qwen-Image 2.1", ditKind: "qwen-image-2.1-turbo-dit",
        // As published, read where they lie: Comfy-Org's bf16 and int8 `convrot` files, the DiT's (`diffusion_models`)
        // and Qwen3-VL-8B's (`text_encoders`).
        ditMaps: [.standard: "qwen_image_2.1_turbo_bf16.safetensors", .compact: "qwen_image_2.1_turbo_int8_convrot.safetensors"],
        encoderMaps: [.standard: "qwen3vl_8b_bf16.safetensors", .compact: "qwen3vl_8b_int8_convrot.safetensors"],
        files: QwenImage21Published.files,
        card: .qwenImage21,
        presentation: .qwenImage21,
        writerStyle: .qwenImage21,
        civitaiModelVersion: 3_400_828,
        // Its VAE divides by 16 and the pipeline floors a side to 32 without a word (`_unpack_latents`).
        sideMultiple: 32,
        everySolver: true,
        loraModules: { QwenImage21Transformer.loraModules() },
        ditTensors: { QwenImage21Transformer.tensors() },
        // Sizes and digests read on the Hub: the undistilled models of this architecture.
        baseCheckpoints: [
            BaseCheckpoint("Comfy-Org/Qwen-Image-2.1@aacf31a6 qwen_image_2.1_bf16", 14230280616, "89f4158d066cc33906a199fca85634f766892dd78f49b6698dabf187ac86c4bc"),
            BaseCheckpoint("Comfy-Org/Qwen-Image-2.1@aacf31a6 qwen_image_2.1_int8_convrot", 7256783064, "cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d"),
            BaseCheckpoint("Comfy-Org/Qwen-Image-2.1@8150226f qwen_image_2.1_bf16", 14230284584, "b4c1ebf5167cbb568eddf9ddf0e1a25fa06888a7f057765f04fc9f9acc4bcc5e"),
            BaseCheckpoint("Comfy-Org/Qwen-Image-2.1@8150226f qwen_image_2.1_int8_convrot", 7256796872, "4ab3fd2516ac19231e76236ad78ac3d8d3b75e9dfedeca5a258d8c1ab60e9788"),
            BaseCheckpoint("unsloth/Qwen-Image-2.1 Qwen-Image-2.1-INT8-ConvRot", 7258370288, "f5f30176b7314cdc0574fe80778c56ef6651a4ac2b89aa28bbc897818fb497f0"),
            BaseCheckpoint("unsloth/Qwen-Image-2.1 Qwen-Image-2.1-INT8", 7258361376, "51ed9fe73c8780e43d91c2ff2f36bf609369a8af2761d617b9a10c83d6ea3c60"),
        ],
        // Qwen3-VL-8B's published file carries its final norm and its head: the encoder writes as it is installed.
        writer: { library in
            let map = library.map(library.encoderMap(.qwenImage21))
            guard Library.exists(map) else { throw MissingFile(map, .published) }
            return PromptWriter.Source(map: map, tokenizer: try library.component(.qwenImage21, "tokenizer"), config: .qwenImage21)
        },
        model: { try Model.qwenImage21(in: $0, card: $1, dit: $2) },
        render: { try QwenImage21Render.render($0, $1, seeds: $2, model: $3, cancellation: $4, emit: $5) })
}

extension Model {
    /// Qwen-Image-2.1 Turbo. ⚖️ Qwen Research License: non-commercial.
    package static func qwenImage21(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.qwenImage21.id) { try qwenImage21(in: b, card: .qwenImage21, dit: b.ditMap(.qwenImage21)) }
    }

    /// `encoder`: the text encoder's file when it is not the installed variant's (a check that judges one variant).
    /// `resolution`: the references' side (a check at the oracle's 512).
    package static func qwenImage21(in b: Library, card: ModelCard, dit: String, encoder: String? = nil,
                                    resolution: Int = QwenImage21References.outputResolution) throws -> Model {
        let vae = try b.component(.qwenImage21, "vae.safetensors")
        return Model(card: card, library: b, chain: try Chain(
            text: QwenImage21TextModule(map: try map(encoder ?? b.encoderMap(.qwenImage21), b),
                                        tokenizer: try b.component(.qwenImage21, "tokenizer")),
            denoising: QwenImage21DenoisingModule(map: try map(dit, b), resolution: resolution),
            decoding: QwenImage21DecodingModule(path: vae), encoding: QwenImage21EncodingModule(path: vae)))
    }
}

extension Qwen3Encoder.Config {
    /// **Qwen3-VL-8B's text model as Qwen-Image-2.1 reads it** (`text_encoder/config.json`'s `text_config`): all 36
    /// layers and no final norm; RoPE's θ is 5·10⁶. Its multimodal RoPE gives a text token the same position on its
    /// three axes: over text alone it is the plain one.
    package static let qwenImage21 = Qwen3Encoder.Config(width: 4096, heads: 32, kvHeads: 8, headDim: 128, hidden: 12288,
                                                         layers: 36, theta: 5e6, finalNorm: false)
}
