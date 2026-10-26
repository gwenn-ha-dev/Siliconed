import CryptoKit
import Foundation
import Synchronization

/// **Rebuilding a family from scratch**: download from the vendor what it publishes, forge, throw
/// away the sources. Siliconed ships no weights; it knows how to fetch them, and every file is
/// taken at a **pinned revision** (the commit that today's maps were forged from). Every weight
/// file (`.safetensors`, `.gguf`) is verified by its sha256 — the one pinned in the recipe, else the
/// one the Hub announces at that revision, and refused when there is neither; the small text files
/// (configs, tokenizers) are not LFS objects and have none. A rebuild yields the same maps.
///
///     store/composants/<famille>/   tokenizers, VAE, configs, DiT names: the small files the
///                                   engine reads as is (a few hundred MB)
///     store/<encoder>.silicon      the forged text encoder
///     store/<dit>.silicon           the forged vendor DiT (optional: a user who only brings their
///                                   own checkpoint does not need it)
///
/// Downloads live under `<root>/telechargements/` for the duration of a forge, then go away.
package struct Installer {
    package let library: Library
    /// Keep the downloaded sources (to reforge without redownloading).
    package var keepSources = false
    package var journal: @Sendable (String) -> Void = { print($0) }
    /// Once raised, it stops the installation between two files or in the middle of a download
    /// (`EngineError.cancelled`). The files already downloaded in full stay in `telechargements/` and
    /// are reused by the next attempt, sha256 checked; a file cut midway starts over from its first
    /// byte. A forge already begun runs to completion.
    package var cancellation: Cancellation?

    package init(library: Library) { self.library = library }

    // MARK: - What each family goes to fetch

    struct Repository { let name: String; let revision: String }
    /// `sha256`: the LFS object pinned in the recipe (a Compact's third-party files): the Hub must
    /// announce it, and the bytes must hash to it. Without it, the sha256 the Hub announces at the
    /// pinned revision is checked.
    /// `size`: the byte count the Hub announces for that LFS object, pinned beside the sha256 where
    /// a user's own copy can be recognized (`Adoption`): the cheap test before the hash.
    struct File { let repository: Repository; let path: String; let destination: String?; var sha256: String? = nil; var size: Int? = nil }

    static let zImage = Repository(name: "Tongyi-MAI/Z-Image-Turbo", revision: "f332072aa78be7aecdf3ee76d5c247082da564a6")
    static let animaBase = Repository(name: "circlestone-labs/Anima-Base-v1.0-Diffusers", revision: "073c3a9db359c31ad0e8aa268d15775473c2176c")
    static let anima = Repository(name: "circlestone-labs/Anima", revision: "f973fc41ec7545364ac9776c2440285f43ff2a30")
    // Krea 2 Turbo: `krea/Krea-2-Turbo` is gated (a token, a license accepted on the Hub), which a user
    // of the app does not have — **no file is downloaded from it**. Every byte comes from ungated
    // repositories, proven equal to Krea's: Comfy-Org's repackaging (the Standard's DiT, the very
    // LFS object of Krea's `turbo.safetensors`; its encoder, Krea's tensor for tensor; the Compact's int8
    // convrot), Qwen-Image's VAE (Krea's fp32 is its bf16 widened, value for value), the tokenizer
    // Qwen-Image-2.1-Turbo publishes (Krea's, the same LFS object), vantagewithai's GGUF for the Light;
    // Krea's three small files are in the code, verbatim (`Krea2Published`).
    static let krea2Comfy = Repository(name: "Comfy-Org/Krea-2", revision: "1171f4dc7e325bf3734586942e737872506c73e1")
    static let qwenImage = Repository(name: "Qwen/Qwen-Image", revision: "75e0b4be04f60ec59a75f475837eced720f823b6")
    /// The official Turbo, the Standard's every file: DiT, encoder, VAE, `processor/`, configs, schedule —
    /// and the Compact's and the Light's but their DiT and their encoder's weights. No recipe reads
    /// the base model (`Qwen/Qwen-Image-2.1`) any more.
    static let qwenImage21Turbo = Repository(name: "Qwen/Qwen-Image-2.1-Turbo", revision: "d65dbc9a7e8f6b5479e33dee6030eaab2a906509")
    // The Compacts' third parties: 8-bit weights of the same models, read by `Range` and
    // judged before being recipes.
    static let zImageGGUF = Repository(name: "unsloth/Z-Image-Turbo-GGUF", revision: "6c80814333b7b6a70a2e5b469a7c6437ce65de0f")
    static let zImageSDNQ = Repository(name: "Disty0/Z-Image-Turbo-SDNQ-int8", revision: "43d0947f91c14b93da75704be27565ee30694f21")
    // Qwen: the Compact's DiT is Comfy-Org's int8 convrot of the Turbo (the repository holds the base's
    // too); the 8-bit encoder is unsloth's of the base model's, which is the Turbo's to the bit (750
    // tensors) — no 8-bit encoder of the Turbo is published, and none is needed.
    static let qwenImage21Comfy = Repository(name: "Comfy-Org/Qwen-Image-2.1", revision: "df94239739eef7205973a145ffa6f441f7b64e84")
    static let qwenImage21Unsloth = Repository(name: "unsloth/Qwen-Image-2.1-FP8", revision: "bb21a8ba7f0371c19ad1f72c82a5e8fc10f39939")
    // The Light's DiT: unsloth's GGUF Q4_K_M, judged whole at the bit.
    // Z-Image's is in the Compact's repository (`zImageGGUF`), at the same revision.
    static let qwenImage21GGUF = Repository(name: "unsloth/Qwen-Image-2.1-Turbo-GGUF", revision: "5d1f9f759ea877ae556cda88dbec958b62ad3809")
    // Anima's Compact and Light DiTs: vanes430's GGUFs of Turbo v1.1 (ComfyUI-GGUF, `cosmos`), the
    // Compact's Q8_0 and the Light's Q4_K_M in one repository; its encoder is the Standard's.
    static let animaGGUF = Repository(name: "vanes430/Anima-Turbo-V1.1-GGUF", revision: "efdf070de980a52fc03027d2f6ed543cc1d87a8f")
    // Krea 2's Light DiT: vantagewithai's GGUF Q4_K_M. `Abiray/Krea-2-Turbo-GGUF`'s is the same tensors
    // at the same offsets, plus 48 bytes appended after the data (`L2P_bypass_…`): a copy, not taken.
    static let krea2GGUF = Repository(name: "vantagewithai/Krea-2-Turbo-GGUF", revision: "390847e2835c16bdf82303731e3d35b874760d89")

    // MOSS-Image: OpenMOSS-Team's own repository for every weight (DiT, Qwen3.5-9B encoder, FLUX.2 VAE, tokenizer),
    // Apache 2.0; and lodestones' DMD LoRA, which is the 8-step recipe (no official turbo as of these revisions).
    static let mossImage = Repository(name: "OpenMOSS-Team/MOSS-Image-SFT", revision: "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9")
    static let mossTurbo = Repository(name: "lodestones/MOSS-Image-SFT-8-steps-dmd", revision: "f95df83b5e90ae91ea412434b3f9b7f1c8c6314f")
    // Its Compact: Comfy-Org's int8 convrot of the SFT's DiT and of its encoder (the repository holds the Base's too).
    static let mossComfy = Repository(name: "Comfy-Org/MOSS-Image", revision: "530a4c94f92c35ef6818b1b035a983fe9dcf5f54")

    /// The small files read at render time, and their place under `store/composants/<famille>/`.
    static func components(_ f: Family) -> [File] {
        switch f {
        case .zImage:
            return [File(repository: zImage, path: "tokenizer/tokenizer.json", destination: "tokenizer/tokenizer.json"),
                    File(repository: zImage, path: "vae/diffusion_pytorch_model.safetensors", destination: "vae.safetensors",
                         sha256: "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3", size: 167666902),
                    File(repository: zImage, path: "transformer/config.json", destination: "transformer.json")]
        case .anima:
            return [File(repository: animaBase, path: "tokenizer/tokenizer.json", destination: "tokenizer/tokenizer.json"),
                    File(repository: animaBase, path: "t5_tokenizer/tokenizer.json", destination: "t5_tokenizer/tokenizer.json"),
                    File(repository: animaBase, path: "vae/diffusion_pytorch_model.safetensors", destination: "vae.safetensors",
                         sha256: "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344", size: 253806966),
                    File(repository: animaBase, path: "transformer/config.json", destination: "transformer.json")]
        case .krea2:
            // Krea's tokenizer is Qwen-Image-2.1-Turbo's `processor/tokenizer.json` (one LFS object,
            // `be756060`); its fp32 VAE is Qwen-Image's bf16 one widened, each of its 126.9 M values
            // (Anima's file, `0c8bc8b7`). No config: `Krea2Published` holds the two the engine reads.
            return [File(repository: qwenImage21Turbo, path: "processor/tokenizer.json", destination: "tokenizer/tokenizer.json"),
                    File(repository: qwenImage, path: "vae/diffusion_pytorch_model.safetensors", destination: "vae.safetensors",
                         sha256: "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344", size: 253806966)]
        case .mossImage:
            // The tokenizer is `Qwen2Tokenizer`: its pre-tokenizer's regex is the file's own (`Tokenizer`
            // reads it), the chat template is in the code (`Qwen35Prompt`). The VAE is published in fp32.
            return [File(repository: mossImage, path: "tokenizer/tokenizer.json", destination: "tokenizer/tokenizer.json",
                         sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523", size: 19989325),
                    File(repository: mossImage, path: "vae/diffusion_pytorch_model.safetensors", destination: "vae.safetensors",
                         sha256: "c3bdf94dc67ec69b30ea9fe1fa8340085679cae64d019f7c7e4a9921f1a7f17a", size: 249521372),
                    File(repository: mossImage, path: "vae/config.json", destination: "vae.json"),
                    File(repository: mossImage, path: "transformer/config.json", destination: "transformer.json")]
        case .qwenImage21:
            // `processor/` as published (transformers 5's serialization: `tokenizer.json` holds the
            // vocabulary, `processor_config.json` the image processor's values). The engine reads
            // `tokenizer.json` alone — the same tokens as the base's — and has the image values in code
            // (`Qwen3VLImages`). The VAE is published in bf16 (the base's fp32 rounded to nearest):
            // widened to fp32 at load, computed in fp32; its config carries `latents_mean`/`latents_std`.
            // The schedule: `sample_sigmas` in `model_index.json`, checked against the scheduler config.
            return ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json"]
                .map { File(repository: qwenImage21Turbo, path: "processor/" + $0, destination: "processor/" + $0) }
                + [File(repository: qwenImage21Turbo, path: "vae/diffusion_pytorch_model.safetensors", destination: "vae.safetensors",
                        sha256: "71879ffd5321e6d10c3c87513e2b474b1252efa7f3dec2969214a9bf06a6dd5c", size: 675508656),
                   File(repository: qwenImage21Turbo, path: "vae/config.json", destination: "vae.json"),
                   File(repository: qwenImage21Turbo, path: "transformer/config.json", destination: "transformer.json"),
                   File(repository: qwenImage21Turbo, path: "scheduler/scheduler_config.json", destination: "scheduler.json"),
                   File(repository: qwenImage21Turbo, path: "model_index.json", destination: "model_index.json")]
        }
    }

    /// **The license files a user reads before the first byte**, at the pinned revisions: the model's
    /// repository first, then each repository the installation also downloads from under a license
    /// file of its own. Checked by hand against the Hub (`/api/models/<repo>/revision/<sha>`):
    /// - Z-Image publishes no license file; Apache 2.0 is declared in the model card → `README.md`.
    /// - Anima's DiT comes from `circlestone-labs/Anima` (CircleStone Non-Commercial v1.2), its encoder,
    ///   VAE and tokenizers from the Diffusers repository, which carries its own file (v1.0).
    /// - Krea 2's `LICENSE.pdf` is read from Comfy-Org's repository, which carries Krea's own (the
    ///   same LFS object, `b82a2805`): Krea's is gated, it answers 401 until the Hub's gate is accepted.
    ///   Comfy-Org's card follows (the repackaging of the DiT and the encoder), then the cards of the
    ///   two Qwen repositories the VAE and the tokenizer are downloaded from (Qwen-Image,
    ///   Qwen-Image-2.1-Turbo): the bytes are Krea's own VAE and tokenizer (proven equal), under Krea's
    ///   license — as for the third parties, the license is the publisher's, the cards say whose bytes.
    /// - Qwen-Image-2.1-Turbo carries the base model's Qwen Research text (same git blob): one file.
    /// - **A Compact also downloads from third parties**, which publish no license file: their model
    ///   card (`README.md`) declares it — Apache 2.0 for unsloth's and Disty0's Z-Image, Qwen
    ///   Research (`license_link` to Qwen's file) for Comfy-Org's and unsloth's Qwen-Image-2.1 (the
    ///   Turbo's DiT; the encoder, the base's, the same tensors). They
    ///   come after the Standard's: the license is the publisher's, the cards say whose bytes.
    /// - **A Light** links the card of its DiT's repository (unsloth's GGUF: Apache 2.0 for Z-Image,
    ///   Qwen Research for Qwen-Image-2.1), then the card of its encoder's, the Compact's.
    /// - **MOSS-Image's Compact** adds Comfy-Org's card (`license: apache-2.0`, `base_model` the publisher's
    ///   two repositories): its DiT and its encoder are Comfy-Org's int8 convrot of the publisher's weights.
    /// - **Krea 2's Compact** adds nothing: its DiT and its encoder are Comfy-Org's, whose card is the
    ///   Standard's. **Its Light** adds vantagewithai's card (`license: other`, Krea's license link,
    ///   `base_model: krea/Krea-2-Turbo`).
    /// - **Anima's Compact and Light** link vanes430's card after the publisher's two files. The card
    ///   says `apache-2.0` and `base_model: circlestone-labs/Anima`: the bytes are CircleStone's DiT
    ///   requantized, under CircleStone's license, which the publisher's files say — as for the others,
    ///   the license is the publisher's, the card says whose bytes. No encoder card: the Standard's.
    static func licenseFiles(_ f: Family, _ v: Variant = .standard) -> [File] {
        switch (f, v) {
        case (.zImage, .compact), (.zImage, .light):
            return licenseFiles(f) + [File(repository: zImageGGUF, path: "README.md", destination: nil),
                                      File(repository: zImageSDNQ, path: "README.md", destination: nil)]
        case (.qwenImage21, .compact):
            return licenseFiles(f) + [File(repository: qwenImage21Comfy, path: "README.md", destination: nil),
                                      File(repository: qwenImage21Unsloth, path: "README.md", destination: nil)]
        case (.qwenImage21, .light):
            return licenseFiles(f) + [File(repository: qwenImage21GGUF, path: "README.md", destination: nil),
                                      File(repository: qwenImage21Unsloth, path: "README.md", destination: nil)]
        case (.anima, .compact), (.anima, .light):
            return licenseFiles(f) + [File(repository: animaGGUF, path: "README.md", destination: nil)]
        case (.krea2, .light):
            return licenseFiles(f) + [File(repository: krea2GGUF, path: "README.md", destination: nil)]
        case (.mossImage, .compact):
            return licenseFiles(f) + [File(repository: mossComfy, path: "README.md", destination: nil)]
        default: break
        }
        switch f {
        case .zImage: return [File(repository: zImage, path: "README.md", destination: nil)]
        case .anima: return [File(repository: anima, path: "LICENSE.md", destination: nil),
                             File(repository: animaBase, path: "LICENSE.md", destination: nil)]
        case .krea2: return [File(repository: krea2Comfy, path: "LICENSE.pdf", destination: nil),
                             File(repository: krea2Comfy, path: "README.md", destination: nil),
                             File(repository: qwenImage, path: "README.md", destination: nil),
                             File(repository: qwenImage21Turbo, path: "README.md", destination: nil)]
        case .qwenImage21: return [File(repository: qwenImage21Turbo, path: "LICENSE", destination: nil)]
        // Apache 2.0 for the model; the LoRA's repository declares its own in its card (Apache 2.0).
        case .mossImage: return [File(repository: mossImage, path: "LICENSE", destination: nil),
                                 File(repository: mossTurbo, path: "README.md", destination: nil)]
        }
    }

    /// `https://huggingface.co/<repo>/blob/<sha>/<file>` — the Hub's page for the file, at the revision
    /// the installation takes (`License.urls`).
    package static func licenseURLs(_ f: Family, _ v: Variant = .standard) -> [URL] {
        licenseFiles(f, v).map { Downloader.page($0.repository.name, revision: $0.repository.revision, path: $0.path) }
    }

    /// The published text encoder (a `text_encoder/` folder) — **its `config.json` first**. A
    /// Compact's comes from two repositories: the publisher's `config.json` (the third party's, when
    /// it publishes one, carries a `quantization_config` the forge has no use for), and the third
    /// party's 8-bit weights, which `ForgeText` keeps 8-bit. The Light's is the Compact's. Anima's
    /// three versions read the Standard's (`Family.encoderMap(_:)`). **Krea 2's has no `config.json`
    /// to download** (Krea's is gated: `Krea2Published.encoderConfig`): its files are the weights alone.
    static func encoder(_ f: Family, _ v: Variant = .standard) -> [File] {
        switch (f, v) {
        case (.zImage, .compact), (.zImage, .light):
            return [File(repository: zImage, path: "text_encoder/config.json", destination: nil),
                    File(repository: zImageSDNQ, path: "text_encoder/model.safetensors", destination: nil,
                         sha256: "21a5e09e4d05ddb2900dc5c51db5e2bc7bc8b3fa08ba10b372798fba3c594677", size: 4413905040)]
        case (.krea2, .compact), (.krea2, .light):
            // Comfy-Org's int8 convrot (`int8_tensorwise`, group 256): the 252 language-model linears and
            // the embedding table, per-row scales; the vision tower in bf16, not written.
            return [File(repository: krea2Comfy, path: "text_encoders/qwen3vl_4b_int8_convrot.safetensors", destination: nil,
                         sha256: "32de113cbf8282b02cbff0794b637b9d6676da09314a1db5a30bcea9bf4ee01c", size: 4858559240)]
        case (.qwenImage21, .compact), (.qwenImage21, .light):
            // The Turbo's `config.json` (the base's but `pad_token_id: null`), the base's 8-bit weights.
            return [File(repository: qwenImage21Turbo, path: "text_encoder/config.json", destination: nil),
                    File(repository: qwenImage21Unsloth, path: "Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors",
                         destination: nil, sha256: "0cb5ed0653e2ec8065f25a6cfb2ab6a89b6c8f143a500acea19f0e36214814b7", size: 9349769248)]
        case (.mossImage, .compact):
            // The publisher's `config.json`, Comfy-Org's int8 convrot (`int8_tensorwise`, group 256 on the language
            // model): its 216 linears and the embedding table, per-row scales; the vision tower, the multi-token head
            // and `lm_head` are not written.
            return [File(repository: mossImage, path: "text_encoder/config.json", destination: nil),
                    File(repository: mossComfy, path: "text_encoders/qwen3.5_9b_int8_convrot.safetensors", destination: nil,
                         sha256: "32e3787472f708fa5f83624acbf5ef23f7fc89c439e05065ea57cfe5a97f025a", size: 9748121532)]
        default: break
        }
        switch f {
        case .zImage:
            // The shards pinned (size, sha256) as a user's own copy is recognized by them (`Adoption`).
            return ["config.json", "model.safetensors.index.json"].map { File(repository: zImage, path: "text_encoder/" + $0, destination: nil) }
                + [("model-00001-of-00003.safetensors", "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223", 3957900840),
                   ("model-00002-of-00003.safetensors", "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5", 3987450520),
                   ("model-00003-of-00003.safetensors", "7ca841ee75b9c61267c0c6148fd8d096d3d21b6d3e161256a9b878154f91fc52", 99630640)]
                .map { File(repository: zImage, path: "text_encoder/" + $0.0, destination: nil, sha256: $0.1, size: $0.2) }
        case .anima:
            return [File(repository: animaBase, path: "text_encoder/config.json", destination: nil),
                    File(repository: animaBase, path: "text_encoder/model.safetensors", destination: nil,
                         sha256: "75e8db94adb374b61baca3d898fc2c0d5086d20689a96d4e74576546433a0526", size: 1192133232)]
        case .krea2:
            // Comfy-Org's bf16 file: Krea's `text_encoder/model.safetensors`, every tensor equal, under
            // `model.` (the `Qwen3VLForConditionalGeneration` layout).
            return [File(repository: krea2Comfy, path: "text_encoders/qwen3vl_4b_bf16.safetensors", destination: nil,
                         sha256: "36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34", size: 8875719384)]
        case .mossImage:
            // The whole `Qwen3_5Model` as published (language model and vision tower in the same four shards);
            // the forge writes the language model alone (`ForgeQwen35`).
            return ["config.json", "model.safetensors.index.json"].map { File(repository: mossImage, path: "text_encoder/" + $0, destination: nil) }
                + [("model-00001-of-00004.safetensors", "abfaa3d3a8612508c85ee8f6da9c734b862f10298b903765c4ad384177c32b0b", 4973910744),
                   ("model-00002-of-00004.safetensors", "9112637fa6bf4d07bafd6ae268852b18d3d0c1242f4ee901c5579006e5bd4894", 4988364288),
                   ("model-00003-of-00004.safetensors", "956f1ef51d0cfe610b0d22987c0b7821c36e3ce228abceccef523518bcf079dc", 4954202112),
                   ("model-00004-of-00004.safetensors", "9e21d20e8adde88c3e0b65d4a0d251f143d98ee9d52db36b4348361091e4e208", 1869001848)]
                .map { File(repository: mossImage, path: "text_encoder/" + $0.0, destination: nil, sha256: $0.1, size: $0.2) }
        case .qwenImage21:
            // One file, the base's four shards to the bit (750 tensors): the map's tensors are the same.
            return [File(repository: qwenImage21Turbo, path: "text_encoder/config.json", destination: nil),
                    File(repository: qwenImage21Turbo, path: "text_encoder/model.safetensors", destination: nil,
                         sha256: "a1eee8233f3063466dd3b06e857b0a2cf13a77c543908374771d51a546beb2e0", size: 17534340584)]
        }
    }

    /// The vendor's DiT (Krea 2's: Comfy-Org's copy of it) — or a Compact's single 8-bit file: unsloth's
    /// GGUF Q8_0 of Z-Image, Comfy-Org's int8 convrot of Qwen-Image-2.1-Turbo and of Krea 2 Turbo,
    /// vanes430's GGUF Q8_0 of Anima, all kept 8-bit by `ForgeDiT` — or a Light's single GGUF Q4_K_M
    /// (unsloth's; vanes430's for Anima, vantagewithai's for Krea 2), its blocks kept whole. No Q4_K_M carries a type under 4 bits: Z-Image's Q4_K, Q5_K, Q6_K;
    /// Qwen-Turbo's Q4_K, Q5_K, Q6_K and Q8_0 (the base's types and offsets, two metadata keys more);
    /// Anima's below.
    static func dit(_ f: Family, _ v: Variant = .standard) -> [File] {
        switch (f, v) {
        case (.zImage, .light):
            return [File(repository: zImageGGUF, path: "z-image-turbo-Q4_K_M.gguf", destination: nil,
                         sha256: "e6494f87de6abaf6a561924f50317a5f271fc34bb4222aabbd801197df8f7daa")]
        case (.qwenImage21, .light):
            return [File(repository: qwenImage21GGUF, path: "qwen-image-2.1-turbo-Q4_K_M.gguf", destination: nil,
                         sha256: "8940e8b522af886e0c0f5afddf720e82dd223cb9a43580b5f61c4d740452ac20")]
        case (.zImage, .compact):
            return [File(repository: zImageGGUF, path: "z-image-turbo-Q8_0.gguf", destination: nil,
                         sha256: "f163d60b0eb427469510b8226243d196574a18139a2e40c017409cfbda95ecfe")]
        case (.anima, .light):
            // Q4_K ×386, Q6_K ×124, F32 ×169, BF16 ×6: nothing under 4 bits. 118 of its tensors are the
            // text adapter (Q4_K ×50, Q6_K ×12, F32 ×56), written dequantized in fp32 beside the map.
            return [File(repository: animaGGUF, path: "anima-turbo-v1.1-Q4_K_M.gguf", destination: nil,
                         sha256: "709fceeec6b4f0eb155330c83fd2539406e86c01e431407276a1f31095484782")]
        case (.krea2, .light):
            // Q4_K ×201, Q5_K ×63, F32 ×167, F16 ×1 (`first`, published in fp32): nothing under 4 bits.
            // Two of its 432 tensors are Krea's easter egg, outside the model (`Recipes.kreaEasterEgg`).
            return [File(repository: krea2GGUF, path: "krea2_turbo-Q4_K_M.gguf", destination: nil,
                         sha256: "bc12f539de7a7a6ddf9bac17ec6e5bfecd42c3517190f491131d8464a1034b40")]
        case (.krea2, .compact):
            // int8 convrot ×224 (the 28 blocks' linears), the rest bf16 and fp32 as Comfy-Org writes it.
            return [File(repository: krea2Comfy, path: "diffusion_models/krea2_turbo_int8_convrot.safetensors", destination: nil,
                         sha256: "8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1")]
        case (.anima, .compact):
            // Q8_0 ×510, F32 ×169, BF16 ×6 (the adapter: Q8_0 ×62, F32 ×56).
            return [File(repository: animaGGUF, path: "anima-turbo-v1.1-Q8_0.gguf", destination: nil,
                         sha256: "74d6c9251f34bc4152fda460dd7befaa9cec79946c270a378fee3d33048153bd")]
        case (.qwenImage21, .compact):
            // Its 31 `comfy_attention.config` (ComfyUI's attention preference) are ignored by the forge.
            return [File(repository: qwenImage21Comfy, path: "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
                         destination: nil, sha256: "8144a0182964aa8403ad6dbc011db837304167d235161b9aabaa8532e967314c")]
        case (.mossImage, .compact):
            // int8 convrot ×168 (the 42 blocks' fused qkv, o, ffn_up_gate, ffn_down), the rest bf16 — the Standard's
            // bits — and its 42 `comfy_attention.config`, ignored by the forge.
            return [File(repository: mossComfy, path: "diffusion_models/moss_image_sft_int8_convrot.safetensors", destination: nil,
                         sha256: "b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f", size: 5413698958)]
        default: break
        }
        switch f {
        case .zImage:
            return (["diffusion_pytorch_model.safetensors.index.json"] + (1...3).map { "diffusion_pytorch_model-0000\($0)-of-00003.safetensors" })
                .map { File(repository: zImage, path: "transformer/" + $0, destination: nil) }
        case .krea2:
            // The original names (`blocks.N.attn.wq`, …), renamed by the forge: the map forged from it is
            // the diffusers shards' to the byte.
            return [File(repository: krea2Comfy, path: "diffusion_models/krea2_turbo_bf16.safetensors", destination: nil,
                         sha256: "78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3")]
        case .anima:
            return [File(repository: anima, path: "split_files/diffusion_models/anima-turbo-v1.1.safetensors", destination: nil)]
        case .mossImage:
            return (["diffusion_pytorch_model.safetensors.index.json"] + (1...3).map { "diffusion_pytorch_model-0000\($0)-of-00003.safetensors" })
                .map { name -> File in
                    let pinned: [String: (String, Int)] = [
                        "diffusion_pytorch_model-00001-of-00003.safetensors": ("e40d3444940a0b0fd7cd45ba7959a5bd3d5efd7ddf87aae5ce4d87ce8aa03d01", 4991978016),
                        "diffusion_pytorch_model-00002-of-00003.safetensors": ("d106da790ec2924ffd8af04b4ffa8f8de281a974b58e55649c3ec596dc8ef033", 4983137016),
                        "diffusion_pytorch_model-00003-of-00003.safetensors": ("11f637ab17c7093344a6110a1e87ffda8e56558dc55183d8e790222cfcdcf52d", 189549184)]
                    return File(repository: mossImage, path: "transformer/" + name, destination: nil,
                                sha256: pinned[name]?.0, size: pinned[name]?.1)
                }
        case .qwenImage21:
            // The Turbo's two shards, pinned: the base's names, shapes, dtypes and sizes, other values.
            let shards = ["6cccd922767f01694461bdcf1f34ea7b771f3442e4d5e338ea9aa55277431cc0",
                          "69f53ebb063d2f606bdaff7e22e0e2144f28d978d73dc75e6e4293fc5982eab5"]
            return [File(repository: qwenImage21Turbo, path: "transformer/diffusion_pytorch_model.safetensors.index.json", destination: nil)]
                + shards.enumerated().map {
                    File(repository: qwenImage21Turbo, path: "transformer/diffusion_pytorch_model-0000\($0.offset + 1)-of-00002.safetensors",
                         destination: nil, sha256: $0.element)
                }
        }
    }

    /// **What a family downloads beside its DiT and forges next to its components**: MOSS-Image's DMD LoRA (the
    /// 8-step recipe), `lora.safetensors` at strength 1 — fp32 in the map, **never merged into the bf16 weights**.
    static func companion(_ f: Family) -> [File] {
        guard f == .mossImage else { return [] }
        return [File(repository: mossTurbo, path: "lora.safetensors", destination: nil,
                     sha256: "4abb14e39c1a59018907581e609f43616ec6810126144ec2b1d2e6f8969628fe", size: 1189176800)]
    }

    /// Where the forged companion LoRA lives: the family's components folder.
    package func companionMap(_ f: Family) -> String? {
        f == .mossImage ? componentsFolder(f) + "/" + MossImageForge.turboMapName : nil
    }

    /// **The DiT's published config**, which the map's header copies (`ForgeDiT.forge`): the family's
    /// `transformer.json` among its components — Krea 2's is in the code (`Krea2Published`).
    package func ditConfig(_ f: Family) throws -> OrderedJSON {
        f == .krea2 ? Krea2Published.transformerConfig : try OrderedJSON.read(componentsFolder(f) + "/transformer.json")
    }

    /// The encoder's `config.json` when it is not downloaded beside its weights (`encoder(_:_:)`).
    package static func publishedEncoderConfig(_ f: Family) -> OrderedJSON? {
        f == .krea2 ? Krea2Published.encoderConfig : nil
    }

    // MARK: - Where things live

    package func componentsFolder(_ f: Family) -> String {
        library.root.appendingPathComponent("store/composants/\(f.rawValue)").path
    }
    // on-disk layout, kept for existing libraries (telechargements/, store/composants/, *.partiel)
    var downloads: String { library.root.appendingPathComponent("telechargements").path }
    func source(_ file: File) -> String {
        downloads + "/" + file.repository.name.replacingOccurrences(of: "/", with: "--") + "/" + file.path
    }

    /// Are a family's components there (without its DiT or its encoder)? A superseded installation's are not.
    package func presentComponents(_ f: Family) -> Bool {
        let fm = FileManager.default
        return Self.components(f).allSatisfy { fm.fileExists(atPath: componentsFolder(f) + "/" + $0.destination!) }
            && fm.fileExists(atPath: componentsFolder(f) + "/dit.json")
            && supersededFiles(f).isEmpty
    }

    /// **What a superseded installation of the family left**: Qwen-Image-2.1 installed from the base
    /// model (Standard, or a third party's 8-bit or GGUF DiT of it in Compact or Light), with a third
    /// party's turbo LoRA and its schedule — its DiT map, and its whole components folder, whose files
    /// have the names of the Turbo's (`vae.safetensors`, `scheduler.json`) and not their content.
    /// Marked by one of its DiT maps, the LoRA, its schedule or the base's `preprocessor_config.json`.
    /// Removed by the next installation before its first byte — that version no longer renders
    /// (`Family.supersededQwenImage21Kind`) — and by an uninstallation; nothing is kept. The encoder
    /// maps stay: the Turbo's encoder is the base's to the bit, and the Compact's 8-bit encoder is
    /// still the one the Compact and the Light read.
    package func supersededFiles(_ f: Family) -> [String] {
        guard f == .qwenImage21 else { return [] }
        let folder = componentsFolder(f), maps = Family.supersededQwenImage21Maps.map(library.map)
        let marks = maps + [folder + "/viggle-turbo-v0.3.lora.silicon", folder + "/scheduler-turbo.json",
                            folder + "/processor/preprocessor_config.json"]
        guard marks.contains(where: Library.exists) else { return [] }
        return (maps + [folder]).filter(Library.exists)
    }

    /// The version a superseded installation of the family was in, read from its DiT map — what its
    /// reinstallation keeps when none is asked (`install`): the `variant` file goes with the folder,
    /// and the Compact's encoder alone would make a Light look like a Compact.
    func supersededVariant(_ f: Family) -> Variant? {
        guard f == .qwenImage21 else { return nil }
        let names = Family.supersededQwenImage21Maps
        if Library.exists(library.map(names[2])) { return .light }
        if Library.exists(library.map(names[1])) { return .compact }
        if Library.exists(library.map(names[0])) { return .standard }
        return nil
    }

    /// What is missing for a family to be able to render with **a** DiT (its own or an imported one):
    /// the encoder of the version installed.
    package func isReady(_ f: Family) -> Bool {
        presentComponents(f) && FileManager.default.fileExists(atPath: library.map(library.encoderMap(f)))
    }

    // MARK: - Installer

    /// **Installs a family** in `variant` (`nil`: the version already there, Standard if none):
    /// components, encoder, and the DiT if `baseDiT`. What is already there is neither redownloaded
    /// nor reforged. `baseDiT` false never changes the version installed (`Library.resolvedVariant`).
    ///
    /// **One version at a time, and never none**: the other version's DiT and encoder (the encoder
    /// unless another family reads it) are removed **once the new version is complete** — until
    /// then the render reads the old one (`Library.variant(of:)`), and a full disk, a cut, a Stop or
    /// a refused file leave it whole. Only when the disk cannot hold both at once does the old one
    /// go before the first byte (`Library.InstallPlan.removesFirst`, said before the user accepts),
    /// after a check that counts the space it gives back.
    ///
    /// **What overlaps**: each large file's sha256 runs behind its download; the DiT downloads
    /// while the encoder is forged, when the disk holds both at once (else one after the other, the
    /// plan's peak). Not the DiT's forge with its download, shard by shard: the map is one file in
    /// execution order behind one SHA-256 in that order, and its forge now takes ~10 s against
    /// minutes of download — what it would hide is not worth a second order of things.
    package func install(_ f: Family, variant: Variant? = nil, baseDiT: Bool = true) throws {
        let fm = FileManager.default
        let superseded = supersededFiles(f)
        let variant = variant ?? (superseded.isEmpty ? nil : supersededVariant(f) ?? library.installedVariant(of: f))
        if !superseded.isEmpty {
            let bytes = superseded.reduce(0) { $0 + Library.size($1) }
            journal(String(format: "  the superseded installation removed first (base model and third-party turbo LoRA): %.2f GB freed",
                           Double(bytes) / 1e9))
            for path in superseded { try fm.removeItem(atPath: path) }
        }
        let plan = try library.installPlan(f, variant: variant, baseDiT: baseDiT, free: library.freeSpace())
        let v = plan.variant
        journal("── \(f.name)" + (f.variants.count > 1 ? " (\(v.rawValue))" : ""))
        try cancellation.check()
        // **Before the first byte**: the whole installation against the volume, from the measured
        // sizes of the forged maps. Each part is checked again, exactly, from the Hub's headers.
        if plan.removesFirst, let old = plan.replacing {
            journal(String(format: "  not enough room for both versions: the %@ version is removed first (%.2f GB)",
                           old.rawValue, Double(plan.freed) / 1e9))
            try library.release(f, keeping: v)
        }
        try library.checkSpace(plan.peak)
        try fm.createDirectory(atPath: componentsFolder(f), withIntermediateDirectories: true)
        // The version aimed at, for `Library.variant(of:)` to break a tie (both versions complete for
        // an instant, or neither after a stop that followed `removesFirst`).
        if f.variants.count > 1 {
            try Data(v.rawValue.utf8).write(to: URL(fileURLWithPath: library.intendedVariantPath(f)), options: .atomic)
        }
        for c in Self.components(f) {
            let target = componentsFolder(f) + "/" + c.destination!
            if fm.fileExists(atPath: target) { continue }
            // A user's copy taken by `Adoption` waits where a download would; the download verifies it.
            if fm.fileExists(atPath: source(c)) { try fm.moveItem(atPath: source(c), toPath: target) }
            try Downloader.download(c.repository.name, revision: c.repository.revision, path: c.path, to: target,
                                         pinned: c.sha256, cancellation: cancellation, journal: journal)
        }
        let reference = componentsFolder(f) + "/dit.json"
        if !fm.fileExists(atPath: reference) {
            try writeReference(f, to: reference)
        }
        let config = try ditConfig(f)

        let encoderName = f.encoderMap(v), ditName = f.ditMap(v)
        let encoderMap = library.map(encoderName), ditMap = library.map(ditName)
        let ditPresent = fm.fileExists(atPath: ditMap)
            && (f != .anima || fm.fileExists(atPath: adapterPath(fromMap: ditMap)))
            && (companionMap(f).map { fm.fileExists(atPath: $0) } ?? true)
        // MOSS-Image: the DiT's map may be there while its LoRA's is not (an interrupted installation).
        let ditMapPresent = fm.fileExists(atPath: ditMap)
        let ditFiles = f == .mossImage && ditMapPresent ? [] : Self.dit(f, v)
        /// The DiT's sources downloaded and verified while the encoder was forged.
        var verified: Set<String> = []
        if !fm.fileExists(atPath: encoderMap) {
            let files = Self.encoder(f, v)
            try checkSpace(files)
            for x in files { try download(x) }
            // **The DiT downloads while the encoder is forged** — when the disk holds, at once, the
            // encoder's sources, its map, and the DiT's sources and map (`checkSpace(_:alsoWriting:)`);
            // otherwise one after the other, as the plan the user accepted counts it. The forge
            // then says no percentages: the app shows the download, and that a map is being prepared.
            journal("forging the encoder → \(encoderName)")
            let begin = Date()
            let beside = baseDiT && !ditPresent
                ? startDownloads(ditFiles, alsoWriting: files.reduce(0) { $0 + Library.size(source($1)) }) : nil
            let forged = Result { () throws -> MapWriter.Tally in
                if f == .mossImage && v == .standard {
                    return try ForgeQwen35.forge(folder: (source(files[0]) as NSString).deletingLastPathComponent, to: encoderMap,
                                                 progressHandler: beside == nil ? progressHandler : nil)
                }
                if f == .mossImage {
                    // The publisher's config, Comfy-Org's 8-bit weights.
                    let weights = files[1]
                    return try ForgeQwen35.forge(source: TensorSource(paths: [source(weights)]), published: OrderedJSON.read(source(files[0])),
                                                 to: encoderMap, directory: weights.repository.name + "@" + weights.repository.revision,
                                                 progressHandler: beside == nil ? progressHandler : nil)
                }
                if let published = Self.publishedEncoderConfig(f) {
                    // The weights alone, their config in the code (Krea 2: `Krea2Published`).
                    return try ForgeText.forge(source: TensorSource(paths: files.map(source)), published: published, family: f,
                                               to: encoderMap, directory: files[0].repository.name + "@" + files[0].repository.revision,
                                               progressHandler: beside == nil ? progressHandler : nil)
                }
                if f.encoderMap(v) == f.encoderMap(.standard) {
                    return try ForgeText.forge(folder: (source(files[0]) as NSString).deletingLastPathComponent, family: f,
                                               to: encoderMap, progressHandler: beside == nil ? progressHandler : nil)
                }
                // Two repositories: the publisher's config, the third party's weights.
                let weights = files.dropFirst()
                return try ForgeText.forge(source: TensorSource(paths: weights.map(source)), published: OrderedJSON.read(source(files[0])),
                                           family: f, to: encoderMap,
                                           directory: weights[weights.startIndex].repository.name + "@" + weights[weights.startIndex].repository.revision,
                                           progressHandler: beside == nil ? progressHandler : nil)
            }
            if case .failure = forged, let beside { beside.stop(); _ = try? beside.wait() }
            let b = try forged.get()
            journal(String(format: "  %.2f GB, %d tensors, %d values rounded", Double(b.bytes) / 1e9, b.tensors, b.inexact))
            journal(String(format: "  ✓ forged %@ in %.0f s", encoderName, Date().timeIntervalSince(begin)))
            if !keepSources { for x in files { try? fm.removeItem(atPath: source(x)) } }
            if let beside { verified = try beside.wait() }
        }

        if baseDiT && !ditPresent && f == .mossImage {
            try installMossImage(variant: v, config: config, ditName: ditName, ditMap: ditMap, ditFiles: ditFiles, verified: verified)
        } else if baseDiT && !ditPresent {
            let files = ditFiles
            if verified.isEmpty { try checkSpace(files) }
            for x in files where !verified.contains(source(x)) { try download(x) }
            journal("forging the DiT → \(ditName)")
            let begin = Date()
            let sourceDiT: TensorSource
            let description: OrderedJSON
            if v != .standard || f == .krea2 {
                // A single third-party file, kept as published (`QuantizedLayouts`): GGUF Q8_0, int8
                // convrot (Compact), GGUF Q4_K_M blocks (Light) — or Comfy-Org's bf16 Krea 2 (Standard).
                sourceDiT = try TensorSource(paths: [source(files[0])])
                description = .object([.init("file", .string((files[0].path as NSString).lastPathComponent)),
                                       .init("repository", .string(files[0].repository.name)),
                                       .init("revision", .string(files[0].repository.revision))])
            } else if f == .anima {
                sourceDiT = try TensorSource(paths: [source(files[0])])
                description = .object([.init("file", .string((files[0].path as NSString).lastPathComponent))])
            } else {
                let folder = (source(files[0]) as NSString).deletingLastPathComponent
                sourceDiT = try TensorSource(folder: folder)
                description = .object([.init("directory", .string(files[0].repository.revision)),
                                      .init("shards", .list(files.dropFirst().map { .string(($0.path as NSString).lastPathComponent) }))])
            }
            let r = try ForgeDiT.forge(source: sourceDiT, family: f, config: config, to: ditMap,
                                        reference: try readReference(f), descriptionSource: description, progressHandler: progressHandler)
            r.rows.forEach { journal("  " + $0) }
            journal(String(format: "  ✓ forged %@ in %.0f s", ditName, Date().timeIntervalSince(begin)))
            if !keepSources { for x in files { try? fm.removeItem(atPath: source(x)) } }
        }
        // The new version is complete: only now does the other one go.
        let freed = try library.release(f, keeping: v)
        if freed > 0 { journal(String(format: "  the other version removed: %.2f GB freed", Double(freed) / 1e9)) }
        if !keepSources { cleanDownloads() }
        journal("✓ \(f.name) installed")
    }

    /// **MOSS-Image's DiT and its LoRA**: the DiT's three shards forged to one bf16 map (`MossImageForge.forgeDiT`), then the
    /// DMD LoRA forged to its own fp32 map in the components folder (`forgeLoRA`) — the two are one recipe, and the LoRA
    /// is never merged into the DiT's weights (a delta smaller than bf16's step would be lost).
    /// The Compact's DiT is Comfy-Org's single int8 file (`MossImageForge.forgeQuantizedDiT`); its LoRA is the Standard's.
    func installMossImage(variant v: Variant, config: OrderedJSON, ditName: String, ditMap: String, ditFiles: [File],
                          verified: Set<String>) throws {
        let fm = FileManager.default
        if !ditFiles.isEmpty {
            if verified.isEmpty { try checkSpace(ditFiles) }
            for x in ditFiles where !verified.contains(source(x)) { try download(x) }
            journal("forging the DiT → \(ditName)")
            let begin = Date()
            if v == .standard {
                let folder = (source(ditFiles[0]) as NSString).deletingLastPathComponent
                let tally = try MossImageForge.forgeDiT(source: TensorSource(folder: folder), config: config, to: ditMap)
                journal(String(format: "  %.2f GB, %d tensors, %d values rounded", Double(tally.bytes) / 1e9, tally.tensors, tally.inexact))
            } else {
                let file = ditFiles[0]
                let r = try MossImageForge.forgeQuantizedDiT(
                    source: TensorSource(paths: [source(file)]), config: config, to: ditMap,
                    description: .object([.init("file", .string((file.path as NSString).lastPathComponent)),
                                          .init("repository", .string(file.repository.name)),
                                          .init("revision", .string(file.repository.revision))]))
                r.rows.forEach { journal("  " + $0) }
            }
            journal(String(format: "  ✓ forged %@ in %.0f s", ditName, Date().timeIntervalSince(begin)))
            if !keepSources { for x in ditFiles { try? fm.removeItem(atPath: source(x)) } }
        }
        guard let turbo = companionMap(.mossImage), !fm.fileExists(atPath: turbo) else { return }
        let files = Self.companion(.mossImage)
        try checkSpace(files)
        for x in files { try download(x) }
        journal("forging the LoRA → \((turbo as NSString).lastPathComponent)")
        let begin = Date()
        let tally = try MossImageForge.forgeLoRA(source: TensorSource(paths: [source(files[0])]), config: MossImageDiT.Config(header: [:]),
                                                 to: turbo, name: MossImageForge.turboName)
        journal(String(format: "  %.2f GB, %d tensors", Double(tally.bytes) / 1e9, tally.tensors))
        journal(String(format: "  ✓ forged %@ in %.0f s", (turbo as NSString).lastPathComponent, Date().timeIntervalSince(begin)))
        if !keepSources { for x in files { try? fm.removeItem(atPath: source(x)) } }
    }

    /// **What an installation will occupy at its peak**, offline: the forged parts of `variant`
    /// still missing (`Family.installedSize`, measured), plus the sources of the largest of them,
    /// which live on disk while it is forged and are thrown away after. **A map is as large as its
    /// sources**: never wider, never narrower than published (bf16 stays bf16, 8-bit stays 8-bit)
    /// — so the sources weigh about their map (a little more when the forge drops tensors:
    /// `lm_head`, the layers past the one read). The exact figure is checked again before each
    /// download, from the Hub's headers (`checkSpace(_:)`). The DiT downloaded while the encoder is
    /// forged — both sources and both maps on disk at once — is not part of it: that happens only when
    /// the disk holds it then (`startDownloads`), never as a condition of the installation.
    package func spaceNeeded(_ f: Family, variant v: Variant = .standard, baseDiT: Bool) -> Int {
        let fm = FileManager.default
        let size = f.installedSize(v)
        var missing: [Int] = []
        if !presentComponents(f) { missing.append(size.components) }
        if !fm.fileExists(atPath: library.map(f.encoderMap(v))) { missing.append(size.encoder) }
        if baseDiT && !fm.fileExists(atPath: library.map(f.ditMap(v))) { missing.append(size.dit) }
        return missing.reduce(0, +) + (missing.max() ?? 0)
    }

    func download(_ x: File) throws {
        try cancellation.check()
        try Downloader.download(x.repository.name, revision: x.repository.revision, path: x.path, to: source(x),
                                     pinned: x.sha256, cancellation: cancellation, journal: journal)
    }

    /// **The space needed before downloading**: the sources (minus what is already there) and the
    /// forged map, no larger than its sources (never wider than published), plus its page alignment
    /// (16 KB per tensor: under 1 %, 2 % kept). Read from the Hub's headers. `alsoWriting`: another
    /// map being written meanwhile, by its sources' size (the encoder's, beside the DiT's download).
    func checkSpace(_ files: [File], alsoWriting: Int = 0) throws {
        var toDownload = 0, sources = 0
        for x in files where x.path.hasSuffix(".safetensors") || x.path.hasSuffix(".gguf") {
            let size = try Downloader.announcedFootprint(Downloader.url(x.repository.name, revision: x.repository.revision, path: x.path)).size ?? 0
            sources += size
            if Library.size(source(x)) != size { toDownload += size }
        }
        let other = alsoWriting + alsoWriting / 50
        try library.checkSpace(toDownload + sources + sources / 50 + other)
    }

    /// Downloads running on a thread of their own, beside a forge.
    final class Downloads: @unchecked Sendable {
        let stopping: Cancellation
        private let finished = DispatchSemaphore(value: 0)
        private var outcome: Result<Set<String>, Error> = .success([])
        init(stopping: Cancellation) { self.stopping = stopping }
        func run(_ body: @escaping @Sendable () throws -> Set<String>) {
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                outcome = Result { try body() }
                finished.signal()
            }
        }
        /// Stops them (the forge beside them failed); `wait` still has to be called.
        func stop() { stopping.cancel() }
        /// The sources downloaded and verified, once all are; the first failure otherwise.
        func wait() throws -> Set<String> { finished.wait(); return try outcome.get() }
    }

    /// **Starts downloading `files` beside the forge about to run**, when the disk holds them with
    /// the map that forge writes (`alsoWriting`); `nil` — and nothing started — when it does not.
    /// A Stop reaches them through the installation's token; a failed forge, through their own.
    func startDownloads(_ files: [File], alsoWriting: Int) -> Downloads? {
        guard (try? checkSpace(files, alsoWriting: alsoWriting)) != nil else { return nil }
        let downloads = Downloads(stopping: Cancellation(following: cancellation))
        var copy = self
        copy.cancellation = downloads.stopping
        let beside = copy
        downloads.run {
            for x in files { try beside.download(x) }
            return Set(files.map(beside.source))
        }
        return downloads
    }

    func cleanDownloads() {
        let fm = FileManager.default
        // Only empty folders: a kept download (`keepSources`) stays.
        guard let e = fm.enumerator(atPath: downloads) else { return }
        var folders: [String] = []
        while let x = e.nextObject() as? String {
            var d: ObjCBool = false
            if fm.fileExists(atPath: downloads + "/" + x, isDirectory: &d), d.boolValue { folders.append(x) }
        }
        for d in folders.sorted(by: { $0.count > $1.count }) {
            let p = downloads + "/" + d
            if (try? fm.contentsOfDirectory(atPath: p))?.isEmpty == true { try? fm.removeItem(atPath: p) }
        }
        if (try? fm.contentsOfDirectory(atPath: downloads))?.isEmpty == true { try? fm.removeItem(atPath: downloads) }
    }

    var progressHandler: (Int, Int, String) -> Void {
        let j = journal
        return { k, n, _ in if k % max(1, n / 10) == 0 && k > 0 { j("  … \(k * 100 / n) %") } }
    }

    // MARK: - The reference: the names and shapes of the vendor's DiT

    /// `dit.json`: `{map name: published shape}` of the vendor's DiT, read from the remote
    /// **headers** (a few hundred KB, not a weight). An imported checkpoint must match it exactly —
    /// this is what proves it really belongs to this family.
    func writeReference(_ f: Family, to path: String) throws {
        let catalog = try RemoteCatalog(files: Self.dit(f).filter { $0.path.hasSuffix(".safetensors") })
        let n = try Recipes.normalize(catalog, family: f)
        let pairs = n.dit.keys.sorted().map { OrderedJSON.Pair($0, .list(n.dit[$0]!.shape.map { .integer($0) })) }
        try Data(OrderedJSON.object(pairs).jsonText.utf8).write(to: URL(fileURLWithPath: path), options: .atomic)
        journal("  reference DiT names: \(pairs.count) tensors")
    }

    package func readReference(_ f: Family) throws -> [String: [Int]] {
        let path = componentsFolder(f) + "/dit.json"
        guard FileManager.default.fileExists(atPath: path) else { throw MissingFile(path, .published) }
        var r: [String: [Int]] = [:]
        for p in try OrderedJSON.read(path).pairs ?? [] { r[p.key] = p.value.elements?.compactMap(\.integer) }
        return r
    }
}

extension Family {
    /// **The license files an installation of `variant` links**, at the revisions it downloads: the
    /// publisher's (`ModelCard.license.urls`), then — for a Compact or a Light — each third party's model card.
    package func licenseURLs(_ variant: Variant) -> [URL] { Installer.licenseURLs(self, variant) }
}

/// Safetensors headers read remotely (`Range` requests): names, shapes, dtypes — not a weight.
final class RemoteCatalog: TensorCatalog {
    private(set) var names: [String] = []
    private var entries: [String: (dtype: String, shape: [Int])] = [:]

    init(files: [Installer.File]) throws {
        for f in files {
            let url = Downloader.url(f.repository.name, revision: f.repository.revision, path: f.path)
            let length = try Downloader.range(url, 0, 7)
            // The same bounds as `Safetensors`: a length that is not an `Int`, zero, or beyond the
            // specification's 100 MB is refused before a byte of it is requested.
            let declared = length.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            guard let n = Int(exactly: declared), n > 0, n <= Safetensors.maximumHeader else {
                throw Numerics.Failure(description: "\(f.path) : header length \(declared)")
            }
            let json = try Downloader.range(url, 8, 8 + n - 1)
            guard let object = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
                throw Numerics.Failure(description: "\(f.path) : unreadable header")
            }
            let sortedEntries = object.compactMap { (k, v) -> (String, Int, String, [Int])? in
                guard let e = v as? [String: Any], let o = e["data_offsets"] as? [Int], o.count == 2 else { return nil }
                return (k, o[0], e["dtype"] as? String ?? "", e["shape"] as? [Int] ?? [])
            }.sorted { $0.1 < $1.1 }
            for (k, _, d, s) in sortedEntries { names.append(k); entries[k] = (d, s) }
        }
    }

    func shape(_ name: String) -> [Int]? { entries[name]?.shape }
    func dtype(_ name: String) -> String? { entries[name]?.dtype }
    func read(_ name: String) throws -> [Float] { throw Numerics.Failure(description: "remote catalog: no values") }
}
