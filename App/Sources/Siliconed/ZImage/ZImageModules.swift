import Foundation

// **Z-Image Turbo's chain**: its modules, its card, its model.
//
//                 text               image (img2img)            denoising               decoding
//   Z-Image    ZImageText ─────────── FluxEncoding ──────────── ZImageDenoising ──────── FluxDecoding

extension TextFormat {
    /// Qwen3-4B, chat template, second-to-last layer: `[tokens, 2560]`.
    package static let zImage = TextFormat("qwen3-4b · chat · penultimate layer")
}

package struct ZImageTextModule: TextModule {
    package var map: String
    /// The published repository's `tokenizer/` folder.
    package var tokenizer: String
    package var output: TextFormat { .zImage }
    package var name: String { "Qwen3-4B" }

    package init(map: String, tokenizer: String) { self.map = map; self.tokenizer = tokenizer }
}

/// Z-Image's S3 DiT under its sampler (negation).
package struct ZImageDenoisingModule: DenoisingModule {
    package var map: String

    package var model: String { "z-image" }
    package var name: String { "DiT Z-Image" }
    package var entry: TextFormat { .zImage }
    package var space: LatentSpace { .flux }
    package var defaultSteps: Int { 8 }
    /// Every `Solver`: its sampler walks them all (`SolverWalk`).
    package var solvers: [Solver] { Solver.allCases }
    /// The middle of its 8 evaluations: σ 0.75 at half and σ 0.643 in full (`Detail.Bell`).
    package var detailBell: Detail.Bell? { .middle }

    package init(map: String) { self.map = map }

    package func sigmas(steps: Int) -> [Float] { FlowMatchSchedule(steps: steps).sigmas }

    /// Evaluations are counted from the start to the end: every step moves (`FlowMatchSchedule`),
    /// hence `N − t_start` evaluations, `⌈N·s⌉` at a strength `s` in img2img.
    package func plan(height: Int, width: Int, steps: Int, start: Int) -> DenoisingPlan {
        let σ = FlowMatchSchedule(steps: steps).sigmas
        return DenoisingPlan(startStep: start, startSigma: σ[start], evaluations: steps - start)
    }
}

extension ModelCard {
    package static let zImage = ModelCard(
        id: "z-image", name: "Z-Image Turbo", family: .zImage, map: nil,
        license: License(text: "Apache 2.0", commercial: true, requiredFilter: false, urls: [ZImagePublished.card]),
        defaultSteps: 8, space: .flux, formats: recommendedFormats)
}

/// Where Z-Image Turbo is published, at the commits an installation takes.
package enum ZImagePublished {
    /// The publisher's repository: the text encoder, the tokenizer, the VAE — and the license, which its card declares.
    package static let tongyi = Repository("Tongyi-MAI/Z-Image-Turbo", "f332072aa78be7aecdf3ee76d5c247082da564a6")
    /// Comfy-Org's single files of the DiT: the publisher's weights in bf16, and their int8 `convrot`.
    package static let comfy = Repository("Comfy-Org/Z-Image-Turbo", "6fc90a3b1b653e935a0d175e260736de25b84df5")
    /// Disty0's SDNQ int8 of the text encoder.
    package static let sdnq = Repository("Disty0/Z-Image-Turbo-SDNQ-int8", "43d0947f91c14b93da75704be27565ee30694f21")

    /// The text encoder's index in the store: the repository's `text_encoder/` folder, kept with the family's other
    /// published files.
    package static let textEncoder = "composants/z-image/text_encoder/model.safetensors.index.json"
    /// Disty0's int8 of it (`text_encoder/model.safetensors`): a generic name, so a folder of its own.
    package static let textEncoderInt8 = "composants/z-image/text_encoder-sdnq-int8/model.safetensors"
    /// Z-Image publishes no license file: its model card declares Apache 2.0.
    package static let card = tongyi.page("README.md")

    /// What both versions read: the tokenizer and Flux's VAE.
    static let shared: [PublishedFile] = [
        PublishedFile(tongyi, "tokenizer/tokenizer.json", at: "composants/z-image/tokenizer/tokenizer.json",
                      bytes: 11_422_654, sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
        PublishedFile(tongyi, "vae/diffusion_pytorch_model.safetensors", at: "composants/z-image/vae.safetensors",
                      bytes: 167_666_902, sha256: "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3"),
    ]
    package static let files: [Variant: [PublishedFile]] = [
        .standard: shared + [
            PublishedFile(comfy, "split_files/diffusion_models/z_image_turbo_bf16.safetensors", at: "z_image_turbo_bf16.safetensors",
                          bytes: 12_309_866_400, sha256: "2407613050b809ffdff18a4ac99af83ea6b95443ecebdf80e064a79c825574a6"),
            PublishedFile(tongyi, "text_encoder/model.safetensors.index.json", at: "composants/z-image/text_encoder/model.safetensors.index.json",
                          bytes: 32_819, gitBlob: "95c0a0059df040d75dc6c396b174382cf61d2f91"),
            PublishedFile(tongyi, "text_encoder/model-00001-of-00003.safetensors", at: "composants/z-image/text_encoder/model-00001-of-00003.safetensors",
                          bytes: 3_957_900_840, sha256: "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223"),
            PublishedFile(tongyi, "text_encoder/model-00002-of-00003.safetensors", at: "composants/z-image/text_encoder/model-00002-of-00003.safetensors",
                          bytes: 3_987_450_520, sha256: "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5"),
            PublishedFile(tongyi, "text_encoder/model-00003-of-00003.safetensors", at: "composants/z-image/text_encoder/model-00003-of-00003.safetensors",
                          bytes: 99_630_640, sha256: "7ca841ee75b9c61267c0c6148fd8d096d3d21b6d3e161256a9b878154f91fc52"),
        ],
        .compact: shared + [
            PublishedFile(comfy, "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors", at: "z_image_turbo_int8_convrot.safetensors",
                          bytes: 6_201_001_296, sha256: "be517ebd47c912a5626a588e1aeea43e6be4a43c0cdcd2b48a2a780d9f358635"),
            PublishedFile(sdnq, "text_encoder/model.safetensors", at: "composants/z-image/text_encoder-sdnq-int8/model.safetensors",
                          bytes: 4_413_905_040, sha256: "21a5e09e4d05ddb2900dc5c51db5e2bc7bc8b3fa08ba10b372798fba3c594677"),
        ],
    ]
}

extension FamilyTraits {
    package static let zImage = FamilyTraits(
        name: "Z-Image", ditKind: "z-image-turbo-dit",
        // The DiT as published, read where it lies: Comfy-Org's bf16 and int8 files (`split_files/diffusion_models`).
        ditMaps: [.standard: "z_image_turbo_bf16.safetensors", .compact: "z_image_turbo_int8_convrot.safetensors"],
        // The text encoder as Tongyi publishes it: `text_encoder/`, three bf16 shards behind their index.
        // The Compact's is Disty0's SDNQ int8 of the same encoder, one file.
        encoderMaps: [.standard: ZImagePublished.textEncoder, .compact: ZImagePublished.textEncoderInt8],
        files: ZImagePublished.files,
        card: .zImage,
        presentation: .zImage,
        writerStyle: .zImage,
        civitaiModelVersion: 2_442_439,
        maximumSide: ZImageDiT.Config.publishedAxisPositions * ZImageDiT.Config.published.patch * LatentSpace.flux.factor,
        startStrength: 0.75,
        everySolver: true,
        loraModules: { ZImageDiT.loraModules() },
        ditTensors: { ZImageDiT.tensors() },
        // Sizes and digests read on the Hub: the undistilled models of this architecture.
        baseCheckpoints: [
            BaseCheckpoint("Comfy-Org/Z-Image@94d47a46 z_image_bf16", 12309866400, "996a67d3ff666946b1c25cbc16d1b1918b6cc0ac166309e23fe3b3d830263dee"),
            BaseCheckpoint("Comfy-Org/Z-Image@94d47a46 z_image_int8_convrot", 6201001296, "eb2b470694fbd9778f15efd309000ea0091312c018c10a352c120b1da9bf4279"),
        ],
        model: { try Model.zImage(in: $0, card: $1, dit: $2) },
        render: { try ZImageRender.render($0, $1, seeds: $2, model: $3, cancellation: $4, emit: $5) })
}

extension Model {
    /// Z-Image Turbo — the product's default.
    package static func zImage(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.zImage.id) { try zImage(in: b, card: .zImage, dit: b.ditMap(.zImage)) }
    }

    /// `encoder`: another text map than the installed version's (a check judging a version it names).
    package static func zImage(in b: Library, card: ModelCard, dit: String, encoder: String? = nil) throws -> Model {
        Model(card: card, library: b, chain: try Chain(
                   text: ZImageTextModule(map: try map(encoder ?? b.encoderMap(.zImage), b),
                                      tokenizer: try b.component(.zImage, "tokenizer")),
                   denoising: ZImageDenoisingModule(map: try map(dit, b)),
                   decoding: FluxDecodingModule(path: try b.component(.zImage, "vae.safetensors")),
                   encoding: FluxEncodingModule(path: try b.component(.zImage, "vae.safetensors"))))
    }
}

extension Qwen3Encoder.Config {
    /// **Qwen3-4B as Z-Image reads it** (`text_encoder/config.json`): `hidden_states[-2]`, so 35 of its 36 layers and
    /// no final norm — layer 35 and `model.norm` stay in the file, unread.
    package static let zImage = Qwen3Encoder.Config(width: 2560, heads: 32, kvHeads: 8, headDim: 128, hidden: 9728,
                                                    layers: 35, finalNorm: false)
}
