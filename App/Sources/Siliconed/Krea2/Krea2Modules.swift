// Krea 2 as the app reads it: its modules (text, denoising), its sheet, its traits.

import Foundation

extension TextFormat {
    package static let krea2 = TextFormat("qwen3-vl-4b · 12 taps")
}

// ── Krea 2 ──────────────────────────────────────────────────────────────────────────────────

/// Template, Qwen tokenizer, Qwen3-VL-4B with its twelve layers taken. Not the fusion: that is the DiT's.
package struct Krea2TextModule: TextModule {
    package var map: String
    package var tokenizer: String
    /// The layers taken, read from `model_index.json`.
    package var sockets: [Int]
    package var output: TextFormat { .krea2 }
    package var name: String { "Qwen3-VL-4B" }

    package init(map: String, tokenizer: String, sockets: [Int]) {
        self.map = map; self.tokenizer = tokenizer; self.sockets = sockets
    }
}

/// The text fusion and `txt_in`, then the Krea 2 DiT, CFG 0. Both read the same map and the
/// LoRA touches both — which is why the fusion lives here and not in the text stage.
package struct Krea2DenoisingModule: DenoisingModule {
    package var map: String
    package var model: String { "krea2" }
    package var name: String { "DiT Krea 2" }
    package var entry: TextFormat { .krea2 }
    package var space: LatentSpace { .qwenImage }
    package var defaultSteps: Int { 8 }

    package init(map: String) { self.map = map }

    /// Every `Solver`: `euler` walks them on Krea 2's own σ (the exponential shift), judged against
    /// diffusers' schedulers (`SolverTests`).
    package var solvers: [Solver] { Solver.allCases }
    /// Its last two evaluations, σ 0.513 at half and σ 0.311 in full: earlier the lie redraws the face (`Detail.Bell`).
    package var detailBell: Detail.Bell? { Detail.Bell(start: 0.714, end: 1, bias: 1, more: 1, most: 2) }

    package func sigmas(steps: Int) -> [Float] { Krea2DiT.sigmas(steps: steps) }

    /// The published recipe as it stands: one evaluation per step, from the start to the end.
    package func plan(height: Int, width: Int, steps: Int, start: Int) -> DenoisingPlan {
        DenoisingPlan(startStep: start, startSigma: sigmas(steps: steps)[start], evaluations: steps - start)
    }
}

// ── What the app reads ──────────────────────────────────────────────────────────────────────

extension Krea2Published {
    /// Comfy-Org's repackaging, ungated: the DiT (Krea's own LFS object in bf16, and its int8 `convrot`), the text
    /// encoder (Krea's tensor for tensor, and its int8 `convrot`), and Krea's license file.
    package static let comfy = Repository("Comfy-Org/Krea-2", "588feaf955b08b9ac4723eeb751752ab045dba71")
    /// Krea's tokenizer is this repository's `processor/tokenizer.json` (the same LFS object).
    package static let tokenizer = Repository("Qwen/Qwen-Image-2.1-Turbo", "d65dbc9a7e8f6b5479e33dee6030eaab2a906509")
    /// Krea's VAE is Qwen-Image's (Krea's fp32 is this bf16 widened, value for value).
    package static let vae = Repository("Qwen/Qwen-Image", "75e0b4be04f60ec59a75f475837eced720f823b6")
    /// The license and the cards the model's sheet links to, pinned (Krea's own repository is gated).
    package static let licenses = [comfy.page("LICENSE.pdf"), comfy.page("README.md")]

    /// What both versions read: the tokenizer and the VAE.
    static let shared: [PublishedFile] = [
        PublishedFile(tokenizer, "processor/tokenizer.json", at: "composants/krea2/tokenizer/tokenizer.json",
                      bytes: 11_422_650, sha256: "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"),
        PublishedFile(vae, "vae/diffusion_pytorch_model.safetensors", at: "composants/krea2/vae.safetensors",
                      bytes: 253_806_966, sha256: "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344"),
    ]
    package static let files: [Variant: [PublishedFile]] = [
        .standard: shared + [
            PublishedFile(comfy, "diffusion_models/krea2_turbo_bf16.safetensors", at: "krea2_turbo_bf16.safetensors",
                          bytes: 26_283_332_608, sha256: "78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3"),
            PublishedFile(comfy, "text_encoders/qwen3vl_4b_bf16.safetensors", at: "qwen3vl_4b_bf16.safetensors",
                          bytes: 8_875_719_384, sha256: "36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34"),
        ],
        .compact: shared + [
            PublishedFile(comfy, "diffusion_models/krea2_turbo_int8_convrot.safetensors", at: "krea2_turbo_int8_convrot.safetensors",
                          bytes: 13_492_686_496, sha256: "8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1"),
            PublishedFile(comfy, "text_encoders/qwen3vl_4b_int8_convrot.safetensors", at: "qwen3vl_4b_int8_convrot.safetensors",
                          bytes: 4_858_559_240, sha256: "32de113cbf8282b02cbff0794b637b9d6676da09314a1db5a30bcea9bf4ee01c"),
        ],
    ]
}

extension ModelCard {
    /// ⚖️ Commercial under $1M annual revenue, mandatory content filter at deployment, "Krea" at
    /// the head of a distributed model (`Krea2License`).
    package static let krea2 = ModelCard(
        id: "krea2", name: "Krea 2 Turbo", family: .krea2, map: nil,
        license: License(text: "Krea 2 Community (commercial < $1M, mandatory content filter)",
                         commercial: true, requiredFilter: true, urls: Krea2Published.licenses),
        defaultSteps: 8, space: .qwenImage, formats: recommendedFormats)
}

extension FamilyTraits {
    package static let krea2 = FamilyTraits(
        name: "Krea 2",
        // As published, read where they lie: Comfy-Org's bf16 and int8 `convrot` files, the DiT's (`diffusion_models`)
        // and Qwen3-VL-4B's (`text_encoders`).
        ditMaps: [.standard: "krea2_turbo_bf16.safetensors", .compact: "krea2_turbo_int8_convrot.safetensors"],
        encoderMaps: [.standard: "qwen3vl_4b_bf16.safetensors", .compact: "qwen3vl_4b_int8_convrot.safetensors"],
        files: Krea2Published.files,
        card: .krea2,
        presentation: .krea2,
        writerStyle: .krea2,
        civitaiModelVersion: 3_064_584,
        // Its 8 steps open on σ 1 · 0.957 · 0.905 and decide who is in the picture: at 0.1 a variation gave another
        // woman (15.3 dB from the original), at 0.05 the same one, her smile changed (25.9 dB) — one pair of seeds, 512².
        subtleVariation: 0.05,
        agreement: Agreement(notice: Krea2License.attribution, text: Krea2License.text),
        everySolver: true,
        loraModules: { Krea2Transformer.loraModules() },
        ditTensors: { Krea2Transformer.tensors() },
        // Sizes and digests read on the Hub: the undistilled models of this architecture.
        baseCheckpoints: [
            BaseCheckpoint("Comfy-Org/Krea-2@588feaf9 krea2_raw_bf16", 26283332608, "f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7"),
            BaseCheckpoint("Comfy-Org/Krea-2@588feaf9 krea2_raw_int8_convrot", 13492686496, "5585a4a38c4bcfb6fde2d480a4aa6edf7f665721ebde56d30662c35a45f5fa5c"),
        ],
        model: { try Model.krea2(in: $0, card: $1, dit: $2) },
        render: RenderLoop.render(Krea2Render.Parts.self),
        prepare: RenderLoop.preparation(Krea2Render.Parts.self))
}

extension Model {
    /// Krea 2 Turbo. ⚖️ Krea 2 Community license: commercial under $1M annual revenue, mandatory
    /// content filter at deployment (§4.2), "Krea" at the head of a distributed model (§3.1).
    package static func krea2(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.krea2.id) { try krea2(in: b, card: .krea2, dit: b.ditMap(.krea2)) }
    }

    package static func krea2(in b: Library, card: ModelCard, dit: String, encoder: String? = nil) throws -> Model {
        Model(card: card, library: b, chain: try Chain(
                   text: Krea2TextModule(map: try map(encoder ?? b.encoderMap(.krea2), b),
                                     tokenizer: try b.component(.krea2, "tokenizer"),
                                     sockets: Krea2Text.sockets),
                   denoising: Krea2DenoisingModule(map: try map(dit, b)),
                   decoding: QwenImageDecodingModule(path: try b.component(.krea2, "vae.safetensors")),
                   encoding: QwenImageEncodingModule(path: try b.component(.krea2, "vae.safetensors"))))
    }
}

extension Qwen3Encoder.Config {
    /// **Qwen3-VL-4B's text model as Krea 2 reads it** (`text_encoder/config.json`'s `text_config`, `Krea2Published`):
    /// the last state drawn is `hidden_states[35]`, layer 34's output — 35 of its 36 layers, no final norm; RoPE's θ is
    /// 5·10⁶.
    package static let krea2 = Qwen3Encoder.Config(width: 2560, heads: 32, kvHeads: 8, headDim: 128, hidden: 9728,
                                                   layers: 35, theta: 5e6, finalNorm: false, prefix: "model.language_model.")
}
