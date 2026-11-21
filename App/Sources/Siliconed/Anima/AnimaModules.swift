// Anima as the app reads it: its modules (text, denoising), its sheet, its traits.

import Foundation

extension TextFormat {
    package static let anima = TextFormat("qwen3-0.6b + t5 · conditioner anima")
}

// ── Anima ───────────────────────────────────────────────────────────────────────────────────

/// Two tokenizers, Qwen3-0.6B (`last_hidden_state`), then Anima's conditioner (T5 + adapter).
package struct AnimaTextModule: TextModule {
    package var map: String
    package var tokenizerQwen: String
    package var tokenizerT5: String
    /// The DiT's file, which carries the conditioner's weights (`llm_adapter.`).
    package var adapter: String
    package var output: TextFormat { .anima }
    package var name: String { "Qwen3-0.6B + T5" }

    package init(map: String, tokenizerQwen: String, tokenizerT5: String, adapter: String) {
        self.map = map; self.tokenizerQwen = tokenizerQwen; self.tokenizerT5 = tokenizerT5
        self.adapter = adapter
    }
}

/// Anima Turbo's Cosmos DiT: Euler, CFG 1.
package struct AnimaDenoisingModule: DenoisingModule {
    package var map: String
    package var model: String { "anima" }
    package var name: String { "DiT Anima" }
    package var entry: TextFormat { .anima }
    package var space: LatentSpace { .qwenImage }
    package var defaultSteps: Int { 8 }

    package init(map: String) { self.map = map }

    /// Every `Solver`: `euler` walks them on Anima's own σ, judged against diffusers' schedulers.
    package var solvers: [Solver] { Solver.allCases }
    /// The middle of its 8 evaluations, **the lie turned round**: told that more noise remains (σ 0.75 × 1.05, σ 0.643 × 1.1
    /// at More), it draws more — books, folds, hair — on the same person and pose. Told less, it gives nothing or mosaic
    /// blocks (`Detail.Bell`).
    package var detailBell: Detail.Bell? { Detail.Bell(start: 0.4, end: 0.9, more: -1, most: -1.5) }

    package func sigmas(steps: Int) -> [Float] { AnimaDiT.sigmas(steps: steps) }

    /// The reference's loop as it stands: one evaluation per step, from the start to the end.
    package func plan(height: Int, width: Int, steps: Int, start: Int) -> DenoisingPlan {
        DenoisingPlan(startStep: start, startSigma: sigmas(steps: steps)[start], evaluations: steps - start)
    }
}

// ── What the app reads ──────────────────────────────────────────────────────────────────────

/// Where Anima Turbo is published, at the commits an installation takes.
package enum AnimaPublished {
    /// CircleStone's repository: the Turbo's DiT (it carries the text conditioner) and Qwen3-0.6B, single files.
    package static let turbo = Repository("circlestone-labs/Anima", "f973fc41ec7545364ac9776c2440285f43ff2a30")
    /// Its Diffusers repository: both tokenizers and the VAE.
    package static let base = Repository("circlestone-labs/Anima-Base-v1.0-Diffusers", "073c3a9db359c31ad0e8aa268d15775473c2176c")
    /// Bedovyy's int8 `convrot` of the Turbo v1.1's DiT.
    package static let int8 = Repository("Bedovyy/Anima-INT8", "d7ba6686f161e9a829373106eb4b881327247ba9")
    package static let licenses = [turbo.page("LICENSE.md"), base.page("LICENSE.md")]

    /// What both versions read: the tokenizers, the VAE, and the text encoder — no int8 of Qwen3-0.6B is published.
    static let shared: [PublishedFile] = [
        PublishedFile(base, "tokenizer/tokenizer.json", at: "composants/anima/tokenizer/tokenizer.json",
                      bytes: 11_422_924, sha256: "476870a1f2fb6f6a2759a6ede2383bf9d5d738f17844563b65c91965b722ae09"),
        PublishedFile(base, "t5_tokenizer/tokenizer.json", at: "composants/anima/t5_tokenizer/tokenizer.json",
                      bytes: 2_424_069, gitBlob: "7093f0b3e6e8c7cc8a69b6b72d1cf2824803ae88"),
        PublishedFile(base, "vae/diffusion_pytorch_model.safetensors", at: "composants/anima/vae.safetensors",
                      bytes: 253_806_966, sha256: "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344"),
        PublishedFile(turbo, "split_files/text_encoders/qwen_3_06b_base.safetensors", at: "qwen_3_06b_base.safetensors",
                      bytes: 1_192_135_096, sha256: "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"),
    ]
    package static let files: [Variant: [PublishedFile]] = [
        .standard: shared + [
            PublishedFile(turbo, "split_files/diffusion_models/anima-turbo-v1.1.safetensors", at: "anima-turbo-v1.1.safetensors",
                          bytes: 4_182_230_656, sha256: "fba11953276b57edf59d1dc4f1857ac05aa079c56f982b4d7c20298d57d3f7eb"),
        ],
        .compact: shared + [
            PublishedFile(int8, "anima-turbo-v1.1-int8convrot.safetensors", at: "anima-turbo-v1.1-int8convrot.safetensors",
                          bytes: 2_649_848_838, sha256: "c462bea054de87ea03000e83f2b60e6d58fbcb6b925d9f607f8a470eeaad52b4"),
        ],
    ]
}

extension ModelCard {
    /// ⚠️ Non-commercial (CircleStone Labs Non-Commercial License; the model derives from NVIDIA's
    /// Cosmos-Predict2, NVIDIA Open Model License): never the default of a product that is sold. Its
    /// images, the license's « Outputs », may be used commercially (§ 2.e).
    package static let anima = ModelCard(
        id: "anima", name: "Anima Turbo", family: .anima, map: nil,
        license: License(text: "CircleStone Labs Non-Commercial", commercial: false, requiredFilter: false, urls: AnimaPublished.licenses),
        defaultSteps: 8, space: .qwenImage, formats: recommendedFormats)
}

extension FamilyTraits {
    /// The DiT as published, read where it lies: circlestone-labs' bf16 file (`split_files/diffusion_models`), Bedovyy's
    /// int8 `convrot` of it. Both carry the text conditioner. No int8 of Qwen3-0.6B is published: both versions read
    /// the bf16 one (`split_files/text_encoders`).
    package static let anima = FamilyTraits(
        name: "Anima", ditKind: "anima-turbo-dit",
        ditMaps: [.standard: "anima-turbo-v1.1.safetensors", .compact: "anima-turbo-v1.1-int8convrot.safetensors"],
        encoderMaps: [.standard: "qwen_3_06b_base.safetensors", .compact: "qwen_3_06b_base.safetensors"],
        files: AnimaPublished.files,
        card: .anima,
        presentation: .anima,
        writerStyle: .anima,
        civitaiModelVersion: 3_263_843,
        maximumSide: AnimaDiT.Config.publishedRopePositions * 2 * LatentSpace.qwenImage.factor,
        // CircleStone's license grants its « Outputs » any use, commercial included (§ 2.e).
        commercialImages: true,
        startStrength: 0.75,
        everySolver: true,
        loraModules: { AnimaCosmosDiT.loraModules() },
        ditTensors: { AnimaCosmosDiT.tensors() },
        // Sizes and digests read on the Hub: the undistilled models of this architecture.
        baseCheckpoints: [
            BaseCheckpoint("circlestone-labs/Anima anima-base-v1.0", 4182218328, "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e"),
            BaseCheckpoint("circlestone-labs/Anima anima-preview3-base", 4182218360, "14fffe8ad5116cd73b9a4696f6a89d7e5f6efdd24b2e4785603aa891a9b2295b"),
        ],
        model: { try Model.anima(in: $0, card: $1, dit: $2) },
        render: { try AnimaRender.render($0, $1, seeds: $2, model: $3, cancellation: $4, emit: $5) })
}

extension Qwen3Encoder.Config {
    /// **Qwen3-0.6B as Anima reads it**: `last_hidden_state`, all 28 layers and the model's final norm.
    package static let anima = Qwen3Encoder.Config(width: 1024, heads: 16, kvHeads: 8, headDim: 128, hidden: 3072,
                                                   layers: 28, finalNorm: true)
}

extension Model {
    /// Anima Turbo v1.1. ⚠️ Non-commercial license: never a product's default.
    package static func anima(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.anima.id) { try anima(in: b, card: .anima, dit: b.ditMap(.anima)) }
    }

    /// Anima on another DiT file (the Compact's, a check's): its text conditioner is that file's.
    package static func anima(in b: Library, dit: String) throws(EngineError) -> Model {
        try building(ModelCard.anima.id) { try anima(in: b, card: .anima, dit: dit) }
    }

    static func anima(in b: Library, card: ModelCard, dit: String) throws -> Model {
        let dit = try map(dit, b)
        return Model(card: card, library: b, chain: try Chain(
                   text: AnimaTextModule(map: try map(b.encoderMap(.anima), b),
                                     tokenizerQwen: try b.component(.anima, "tokenizer"),
                                     tokenizerT5: try b.component(.anima, "t5_tokenizer"),
                                     adapter: dit),
                   denoising: AnimaDenoisingModule(map: dit),
                   decoding: QwenImageDecodingModule(path: try b.component(.anima, "vae.safetensors")),
                   encoding: QwenImageEncodingModule(path: try b.component(.anima, "vae.safetensors"))))
    }
}
