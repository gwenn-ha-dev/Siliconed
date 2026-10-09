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
///     store/composants/<famille>/<turbo>.lora.silicon
///                                   a LoRA the family cannot do without (Qwen-Image-2.1's turbo),
///                                   forged at installation like the maps
///
/// Downloads live under `<racine>/telechargements/` for the duration of a forge, then go away.
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
    struct File { let repository: Repository; let path: String; let destination: String?; var sha256: String? = nil }

    static let zImage = Repository(name: "Tongyi-MAI/Z-Image-Turbo", revision: "f332072aa78be7aecdf3ee76d5c247082da564a6")
    static let animaBase = Repository(name: "circlestone-labs/Anima-Base-v1.0-Diffusers", revision: "073c3a9db359c31ad0e8aa268d15775473c2176c")
    static let anima = Repository(name: "circlestone-labs/Anima", revision: "f973fc41ec7545364ac9776c2440285f43ff2a30")
    static let krea2 = Repository(name: "krea/Krea-2-Turbo", revision: "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7")
    static let klein4b = Repository(name: "black-forest-labs/FLUX.2-klein-4B", revision: "e7b7dc27f91deacad38e78976d1f2b499d76a294")
    static let ernie = Repository(name: "baidu/ERNIE-Image-Turbo", revision: "bc68c81e2a1730a394d5fc9fae70713dee940140")
    static let qwenImage21 = Repository(name: "Qwen/Qwen-Image-2.1", revision: "d26bb61231c349cf6b7896fa83353113880e1ba3")
    static let viggleTurbo = Repository(name: "Viggle/Qwen-Image-2.1-viggle-turbo", revision: "009a44a895ef85f7e643c80fdca9543795248867")
    // The Compacts' third parties: 8-bit weights of the same models, read by `Range` and
    // judged before being recipes.
    static let zImageGGUF = Repository(name: "unsloth/Z-Image-Turbo-GGUF", revision: "6c80814333b7b6a70a2e5b469a7c6437ce65de0f")
    static let zImageSDNQ = Repository(name: "Disty0/Z-Image-Turbo-SDNQ-int8", revision: "43d0947f91c14b93da75704be27565ee30694f21")
    static let qwenImage21Comfy = Repository(name: "Comfy-Org/Qwen-Image-2.1", revision: "cb504a4090723e43f17ad01cec0359490e2de613")
    static let qwenImage21Unsloth = Repository(name: "unsloth/Qwen-Image-2.1-FP8", revision: "bb21a8ba7f0371c19ad1f72c82a5e8fc10f39939")
    // The Light's DiT: unsloth's GGUF Q4_K_M, judged whole at the bit.
    // Z-Image's is in the Compact's repository (`zImageGGUF`), at the same revision.
    static let qwenImage21GGUF = Repository(name: "unsloth/Qwen-Image-2.1-GGUF", revision: "2c31ccd392b367a6637841a143813320a02dff55")

    /// The small files read at render time, and their place under `store/composants/<famille>/`.
    static func components(_ f: Family) -> [File] {
        switch f {
        case .zImage:
            return [File(repository: zImage, path: "tokenizer/tokenizer.json", destination: "tokenizer/tokenizer.json"),
                    File(repository: zImage, path: "vae/diffusion_pytorch_model.safetensors", destination: "vae.safetensors"),
                    File(repository: zImage, path: "transformer/config.json", destination: "transformer.json")]
        case .anima:
            return [File(repository: animaBase, path: "tokenizer/tokenizer.json", destination: "tokenizer/tokenizer.json"),
                    File(repository: animaBase, path: "t5_tokenizer/tokenizer.json", destination: "t5_tokenizer/tokenizer.json"),
                    File(repository: animaBase, path: "vae/diffusion_pytorch_model.safetensors", destination: "vae.safetensors"),
                    File(repository: animaBase, path: "transformer/config.json", destination: "transformer.json")]
        case .krea2:
            return [File(repository: krea2, path: "tokenizer/tokenizer.json", destination: "tokenizer/tokenizer.json"),
                    File(repository: krea2, path: "model_index.json", destination: "model_index.json"),
                    File(repository: krea2, path: "vae/diffusion_pytorch_model.safetensors", destination: "vae.safetensors"),
                    File(repository: krea2, path: "transformer/config.json", destination: "transformer.json")]
        case .klein4b:
            return [File(repository: klein4b, path: "tokenizer/tokenizer.json", destination: "tokenizer/tokenizer.json"),
                    File(repository: klein4b, path: "vae/diffusion_pytorch_model.safetensors", destination: "vae.safetensors"),
                    File(repository: klein4b, path: "transformer/config.json", destination: "transformer.json")]
        case .ernie:
            // The VAE is FLUX.2 [klein]'s, bit for bit (same sha256); only the pipeline differs.
            return [File(repository: ernie, path: "tokenizer/tokenizer.json", destination: "tokenizer/tokenizer.json"),
                    File(repository: ernie, path: "vae/diffusion_pytorch_model.safetensors", destination: "vae.safetensors"),
                    File(repository: ernie, path: "transformer/config.json", destination: "transformer.json")]
        case .qwenImage21:
            // `processor/` as published, minus what `tokenizer.json` already holds (`vocab.json`,
            // `merges.txt`, `added_tokens.json`, `special_tokens_map.json`) and the video. The VAE's
            // config carries `latents_mean`/`latents_std`. Two schedules: the base one, and Viggle's,
            // whose `shift_terminal` is null (the base's 0.02 wrecks the turbo's last step).
            return ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json"]
                .map { File(repository: qwenImage21, path: "processor/" + $0, destination: "processor/" + $0) }
                + [File(repository: qwenImage21, path: "vae/diffusion_pytorch_model.safetensors", destination: "vae.safetensors"),
                   File(repository: qwenImage21, path: "vae/config.json", destination: "vae.json"),
                   File(repository: qwenImage21, path: "transformer/config.json", destination: "transformer.json"),
                   File(repository: qwenImage21, path: "scheduler/scheduler_config.json", destination: "scheduler.json"),
                   File(repository: viggleTurbo, path: "scheduler/scheduler_config.json", destination: "scheduler-turbo.json")]
        }
    }

    /// The published LoRA that `Family.turboLoRA` is forged from.
    static func turbo(_ f: Family) -> File? {
        f == .qwenImage21
            ? File(repository: viggleTurbo, path: "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors", destination: nil)
            : nil
    }

    /// **The license files a user reads before the first byte**, at the pinned revisions: the model's
    /// repository first, then each repository the installation also downloads from under a license
    /// file of its own. Checked by hand against the Hub (`/api/models/<repo>/revision/<sha>`):
    /// - Z-Image publishes no license file; Apache 2.0 is declared in the model card → `README.md`.
    /// - Anima's DiT comes from `circlestone-labs/Anima` (CircleStone Non-Commercial v1.2), its encoder,
    ///   VAE and tokenizers from the Diffusers repository, which carries its own file (v1.0).
    /// - Krea 2 is gated: `LICENSE.pdf` answers 401 until the Hub's gate is accepted (the card's
    ///   `license_link` points to the same PDF outside the Hub).
    /// - Qwen-Image-2.1 and Viggle's turbo LoRA carry the same Qwen Research text (same git blob).
    /// - **A Compact also downloads from third parties**, which publish no license file: their model
    ///   card (`README.md`) declares it — Apache 2.0 for unsloth's and Disty0's Z-Image, Qwen
    ///   Research (`license_link` to Qwen's file) for Comfy-Org's and unsloth's Qwen-Image-2.1. They
    ///   come after the Standard's: the license is the publisher's, the cards say whose bytes.
    /// - **A Light** links the card of its DiT's repository (unsloth's GGUF: Apache 2.0 for Z-Image,
    ///   Qwen Research for Qwen-Image-2.1), then the card of its encoder's, the Compact's.
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
        default: break
        }
        switch f {
        case .zImage: return [File(repository: zImage, path: "README.md", destination: nil)]
        case .anima: return [File(repository: anima, path: "LICENSE.md", destination: nil),
                             File(repository: animaBase, path: "LICENSE.md", destination: nil)]
        case .krea2: return [File(repository: krea2, path: "LICENSE.pdf", destination: nil)]
        case .klein4b: return [File(repository: klein4b, path: "LICENSE.md", destination: nil)]
        case .ernie: return [File(repository: ernie, path: "LICENSE", destination: nil)]
        case .qwenImage21: return [File(repository: qwenImage21, path: "LICENSE", destination: nil),
                                   File(repository: viggleTurbo, path: "LICENSE", destination: nil)]
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
    /// party's 8-bit weights, which `ForgeText` keeps 8-bit. The Light's is the Compact's.
    static func encoder(_ f: Family, _ v: Variant = .standard) -> [File] {
        switch (f, v) {
        case (.zImage, .compact), (.zImage, .light):
            return [File(repository: zImage, path: "text_encoder/config.json", destination: nil),
                    File(repository: zImageSDNQ, path: "text_encoder/model.safetensors", destination: nil,
                         sha256: "21a5e09e4d05ddb2900dc5c51db5e2bc7bc8b3fa08ba10b372798fba3c594677")]
        case (.qwenImage21, .compact), (.qwenImage21, .light):
            return [File(repository: qwenImage21, path: "text_encoder/config.json", destination: nil),
                    File(repository: qwenImage21Unsloth, path: "Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors",
                         destination: nil, sha256: "0cb5ed0653e2ec8065f25a6cfb2ab6a89b6c8f143a500acea19f0e36214814b7")]
        default: break
        }
        switch f {
        case .zImage:
            return ["config.json", "model.safetensors.index.json", "model-00001-of-00003.safetensors",
                    "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]
                .map { File(repository: zImage, path: "text_encoder/" + $0, destination: nil) }
        case .anima:
            return ["config.json", "model.safetensors"].map { File(repository: animaBase, path: "text_encoder/" + $0, destination: nil) }
        case .krea2:
            return ["config.json", "model.safetensors"].map { File(repository: krea2, path: "text_encoder/" + $0, destination: nil) }
        case .klein4b:
            // Z-Image's Qwen3-4B, bit for bit: downloaded only if Z-Image has not already forged the map.
            return ["config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors"]
                .map { File(repository: klein4b, path: "text_encoder/" + $0, destination: nil) }
        case .ernie:
            return ["config.json", "model.safetensors"].map { File(repository: ernie, path: "text_encoder/" + $0, destination: nil) }
        case .qwenImage21:
            return (["config.json", "model.safetensors.index.json"] + (1...4).map { "model-0000\($0)-of-00004.safetensors" })
                .map { File(repository: qwenImage21, path: "text_encoder/" + $0, destination: nil) }
        }
    }

    /// The vendor's DiT — or a Compact's single 8-bit file: unsloth's GGUF Q8_0 of Z-Image,
    /// Comfy-Org's int8 convrot of Qwen-Image-2.1, both kept 8-bit by `ForgeDiT` — or a
    /// Light's single GGUF Q4_K_M from unsloth, its blocks kept whole. Neither
    /// Q4_K_M carries a type under 4 bits: Z-Image's Q4_K, Q5_K, Q6_K; Qwen's Q4_K, Q5_K, Q6_K and
    /// Q8_0 (its Q4_K_S, which carries Q3_K, would be refused whole).
    static func dit(_ f: Family, _ v: Variant = .standard) -> [File] {
        switch (f, v) {
        case (.zImage, .light):
            return [File(repository: zImageGGUF, path: "z-image-turbo-Q4_K_M.gguf", destination: nil,
                         sha256: "e6494f87de6abaf6a561924f50317a5f271fc34bb4222aabbd801197df8f7daa")]
        case (.qwenImage21, .light):
            return [File(repository: qwenImage21GGUF, path: "qwen-image-2.1-Q4_K_M.gguf", destination: nil,
                         sha256: "631d532e7ca71e8d90a87c71d3699761a812039d22e3370e87498d87754660fe")]
        case (.zImage, .compact):
            return [File(repository: zImageGGUF, path: "z-image-turbo-Q8_0.gguf", destination: nil,
                         sha256: "f163d60b0eb427469510b8226243d196574a18139a2e40c017409cfbda95ecfe")]
        case (.qwenImage21, .compact):
            return [File(repository: qwenImage21Comfy, path: "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
                         destination: nil, sha256: "cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d")]
        default: break
        }
        switch f {
        case .zImage:
            return (["diffusion_pytorch_model.safetensors.index.json"] + (1...3).map { "diffusion_pytorch_model-0000\($0)-of-00003.safetensors" })
                .map { File(repository: zImage, path: "transformer/" + $0, destination: nil) }
        case .krea2:
            return (["diffusion_pytorch_model.safetensors.index.json"] + (1...3).map { "diffusion_pytorch_model-0000\($0)-of-00003.safetensors" })
                .map { File(repository: krea2, path: "transformer/" + $0, destination: nil) }
        case .anima:
            return [File(repository: anima, path: "split_files/diffusion_models/anima-turbo-v1.1.safetensors", destination: nil)]
        case .klein4b:
            return [File(repository: klein4b, path: "transformer/diffusion_pytorch_model.safetensors", destination: nil)]
        case .ernie:
            return (["diffusion_pytorch_model.safetensors.index.json"] + (1...2).map { "diffusion_pytorch_model-0000\($0)-of-00002.safetensors" })
                .map { File(repository: ernie, path: "transformer/" + $0, destination: nil) }
        case .qwenImage21:
            return (["diffusion_pytorch_model.safetensors.index.json"] + (1...2).map { "diffusion_pytorch_model-0000\($0)-of-00002.safetensors" })
                .map { File(repository: qwenImage21, path: "transformer/" + $0, destination: nil) }
        }
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

    /// Are a family's components there (without its DiT or its encoder)? Its turbo LoRA counts.
    package func presentComponents(_ f: Family) -> Bool {
        let fm = FileManager.default
        return Self.components(f).allSatisfy { fm.fileExists(atPath: componentsFolder(f) + "/" + $0.destination!) }
            && fm.fileExists(atPath: componentsFolder(f) + "/dit.json")
            && (f.turboLoRA.map { fm.fileExists(atPath: componentsFolder(f) + "/" + $0) } ?? true)
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
    package func install(_ f: Family, variant: Variant? = nil, baseDiT: Bool = true) throws {
        let fm = FileManager.default
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
            try Downloader.download(c.repository.name, revision: c.repository.revision, path: c.path, to: target,
                                         cancellation: cancellation, journal: journal)
        }
        let reference = componentsFolder(f) + "/dit.json"
        if !fm.fileExists(atPath: reference) {
            try writeReference(f, to: reference)
        }
        let config = try OrderedJSON.read(componentsFolder(f) + "/transformer.json")

        // Checked against the family's DiT names (`dit.json`): needed even with an imported DiT.
        if let name = f.turboLoRA, let x = Self.turbo(f), !fm.fileExists(atPath: componentsFolder(f) + "/" + name) {
            try checkSpace([x])
            try download(x)
            journal("forging the turbo LoRA → \(name)")
            let r = try ForgeLoRA.forge(file: source(x), to: componentsFolder(f) + "/" + name, family: f,
                                         reference: { try readReference($0) })
            r.rows.forEach { journal("  " + $0) }
            if !keepSources { try? fm.removeItem(atPath: source(x)) }
        }

        let encoderName = f.encoderMap(v)
        let encoderMap = library.map(encoderName)
        if !fm.fileExists(atPath: encoderMap) {
            let files = Self.encoder(f, v)
            try checkSpace(files)
            for x in files { try download(x) }
            journal("forging the encoder → \(encoderName)")
            let b: MapWriter.Tally
            if v == .standard {
                b = try ForgeText.forge(folder: (source(files[0]) as NSString).deletingLastPathComponent, family: f,
                                        to: encoderMap, progressHandler: progressHandler)
            } else {
                // Two repositories: the publisher's config, the third party's weights.
                let weights = files.dropFirst()
                b = try ForgeText.forge(source: TensorSource(paths: weights.map(source)), published: OrderedJSON.read(source(files[0])),
                                        family: f, to: encoderMap,
                                        directory: weights[weights.startIndex].repository.name + "@" + weights[weights.startIndex].repository.revision,
                                        progressHandler: progressHandler)
            }
            journal(String(format: "  %.2f GB, %d tensors, %d values rounded", Double(b.bytes) / 1e9, b.tensors, b.inexact))
            if !keepSources { for x in files { try? fm.removeItem(atPath: source(x)) } }
        }

        let ditName = f.ditMap(v)
        let ditMap = library.map(ditName)
        let ditPresent = fm.fileExists(atPath: ditMap)
            && (f != .anima || fm.fileExists(atPath: adapterPath(fromMap: ditMap)))
        if baseDiT && !ditPresent {
            let files = Self.dit(f, v)
            try checkSpace(files)
            for x in files { try download(x) }
            journal("forging the DiT → \(ditName)")
            let sourceDiT: TensorSource
            let description: OrderedJSON
            if f == .anima {
                sourceDiT = try TensorSource(paths: [source(files[0])])
                description = .object([.init("file", .string((files[0].path as NSString).lastPathComponent))])
            } else if v != .standard {
                // A single third-party file, kept as published (`QuantizedLayouts`): GGUF Q8_0, int8
                // convrot (Compact), GGUF Q4_K_M blocks (Light).
                sourceDiT = try TensorSource(paths: [source(files[0])])
                description = .object([.init("file", .string((files[0].path as NSString).lastPathComponent)),
                                       .init("repository", .string(files[0].repository.name)),
                                       .init("revision", .string(files[0].repository.revision))])
            } else {
                let folder = (source(files[0]) as NSString).deletingLastPathComponent
                sourceDiT = try TensorSource(folder: folder)
                description = .object([.init("directory", .string(files[0].repository.revision)),
                                      .init("shards", .list(files.dropFirst().map { .string(($0.path as NSString).lastPathComponent) }))])
            }
            let r = try ForgeDiT.forge(source: sourceDiT, family: f, config: config, to: ditMap,
                                        reference: try readReference(f), descriptionSource: description, progressHandler: progressHandler)
            r.rows.forEach { journal("  " + $0) }
            if !keepSources { for x in files { try? fm.removeItem(atPath: source(x)) } }
        }
        // The new version is complete: only now does the other one go.
        let freed = try library.release(f, keeping: v)
        if freed > 0 { journal(String(format: "  the other version removed: %.2f GB freed", Double(freed) / 1e9)) }
        if !keepSources { cleanDownloads() }
        journal("✓ \(f.name) installed")
    }

    /// **What an installation will occupy at its peak**, offline: the forged parts of `variant`
    /// still missing (`Family.installedSize`, measured), plus the sources of the largest of them,
    /// which live on disk while it is forged and are thrown away after. **A map is as large as its
    /// sources**: never wider, never narrower than published (bf16 stays bf16, 8-bit stays 8-bit)
    /// — so the sources weigh about their map (a little more when the forge drops tensors:
    /// `lm_head`, the layers past the one read). The exact figure is checked again before each
    /// download, from the Hub's headers (`checkSpace(_:)`).
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
    /// (16 KB per tensor: under 1 %, 2 % kept). Read from the Hub's headers.
    func checkSpace(_ files: [File]) throws {
        var toDownload = 0, sources = 0
        for x in files where x.path.hasSuffix(".safetensors") || x.path.hasSuffix(".gguf") {
            let size = try Downloader.announcedFootprint(Downloader.url(x.repository.name, revision: x.repository.revision, path: x.path)).size ?? 0
            sources += size
            if Library.size(source(x)) != size { toDownload += size }
        }
        try library.checkSpace(toDownload + sources + sources / 50)
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
    public func licenseURLs(_ variant: Variant) -> [URL] { Installer.licenseURLs(self, variant) }
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

// MARK: - The download

/// **A Hugging Face file, at a pinned revision, verified.** The token (gated repos, like Krea 2)
/// comes from `HF_TOKEN` or from `~/.cache/huggingface/token` — the one `huggingface-cli login`
/// writes.
package enum Downloader {
    static func url(_ repository: String, revision: String, path: String) -> URL {
        URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)!)")!
    }

    /// The file's page on the Hub (`blob/`), for a human to read — not the bytes (`resolve/`).
    static func page(_ repository: String, revision: String, path: String) -> URL {
        URL(string: "https://huggingface.co/\(repository)/blob/\(revision)/\(path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)!)")!
    }

    static var token: String? {
        let env = ProcessInfo.processInfo.environment
        if let t = env["HF_TOKEN"] ?? env["HUGGING_FACE_HUB_TOKEN"], !t.isEmpty { return t }
        let path = (env["HF_HOME"] ?? NSHomeDirectory() + "/.cache/huggingface") + "/token"
        return (try? String(contentsOfFile: path, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func request(_ url: URL, method: String = "GET") -> URLRequest {
        var r = URLRequest(url: url)
        r.httpMethod = method
        if let t = token { r.setValue("Bearer \(t)", forHTTPHeaderField: "Authorization") }
        return r
    }

    /// The bytes `[begin, end]` of a remote file. Only a `206` with exactly those bytes is taken: a
    /// server that ignores `Range` answers `200` with the whole file — gigabytes read as a header.
    static func range(_ url: URL, _ begin: Int, _ end: Int) throws -> Data {
        var r = request(url)
        r.setValue("bytes=\(begin)-\(end)", forHTTPHeaderField: "Range")
        let frozen = r
        let (data, response) = try wait { try await URLSession.shared.data(for: frozen) }
        try checkRange(response, data, begin, end, file: url.lastPathComponent)
        return data
    }

    /// `range`'s judgment of a response, apart so that a test can hand it one.
    static func checkRange(_ response: URLResponse, _ data: Data, _ begin: Int, _ end: Int, file: String) throws {
        guard let h = response as? HTTPURLResponse, h.statusCode == 206 else {
            throw EngineError.downloadRefused(file: file, status: (response as? HTTPURLResponse)?.statusCode ?? 0)
        }
        guard data.count == end - begin + 1 else { throw EngineError.downloadInterrupted(file: file) }
    }

    /// The sha256 (LFS files) and size announced by the Hub, read from the response **before**
    /// redirection to the CDN.
    static func announcedFootprint(_ url: URL) throws -> (sha256: String?, size: Int?, status: Int) {
        final class NoRedirect: NSObject, URLSessionTaskDelegate {
            func urlSession(_ s: URLSession, task: URLSessionTask, willPerformHTTPRedirection r: HTTPURLResponse,
                            newRequest: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
                completionHandler(nil)
            }
        }
        let session = URLSession(configuration: .ephemeral, delegate: NoRedirect(), delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let r = request(url, method: "HEAD")
        let (_, response) = try wait { try await session.data(for: r) }
        guard let h = response as? HTTPURLResponse else { return (nil, nil, 0) }
        guard h.statusCode < 400 else {
            throw EngineError.downloadRefused(file: url.lastPathComponent, status: h.statusCode)
        }
        let etag = (h.value(forHTTPHeaderField: "X-Linked-Etag") ?? "").trimmingCharacters(in: CharacterSet(charactersIn: "\"W/"))
        let size = h.value(forHTTPHeaderField: "X-Linked-Size").flatMap(Int.init)
        return (etag.count == 64 ? etag : nil, size, h.statusCode)
    }

    /// **The sha256 a download is checked against**: the recipe's when it pins one (a Hub that
    /// announces another is refused before the first byte; one that announces none is checked
    /// against the pin alone), else the Hub's. A weight file with neither is refused, by name: an
    /// unverified checkpoint would be forged as if it had been verified.
    static func expectedSHA(path: String, pinned: String?, announced: String?, status: Int, file: String) throws -> String? {
        if let pinned {
            if let announced, announced != pinned { throw EngineError.downloadCorrupt(file: file) }
            return pinned
        }
        if announced == nil, path.hasSuffix(".safetensors") || path.hasSuffix(".gguf") {
            throw EngineError.downloadRefused(file: file, status: status)
        }
        return announced
    }

    /// Downloads `path` of `repository@revision` to `destination`. Already there and verified: nothing.
    /// `pinned`: the sha256 the recipe expects — a Hub that announces another one is refused before
    /// the first byte, and the bytes are checked against it.
    package static func download(_ repository: String, revision: String, path: String, to destination: String,
                                    pinned: String? = nil, cancellation: Cancellation? = nil,
                                    journal: @escaping @Sendable (String) -> Void) throws {
        let fm = FileManager.default
        let u = url(repository, revision: revision, path: path)
        let (announced, size, status) = try announcedFootprint(u)
        let label = "\(repository.split(separator: "/").last!)/\(path)"
        let sha = try expectedSHA(path: path, pinned: pinned, announced: announced, status: status, file: label)
        if fm.fileExists(atPath: destination) {
            let t = (try? fm.attributesOfItem(atPath: destination)[.size] as? Int) ?? -1
            if size == nil || t == size, sha == nil || (try? sha256(destination)) == sha { return }
            try fm.removeItem(atPath: destination)
        }
        try fm.createDirectory(atPath: (destination as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        journal("↓ \(label)" + (size.map { String(format: " (%.2f GB)", Double($0) / 1e9) } ?? ""))
        let begin = Date()
        if let size, size > chunkedFrom {
            try downloadInChunks(u, size: size, sha: sha, label: label, to: destination,
                                 cancellation: cancellation, journal: journal)
            let s = Date().timeIntervalSince(begin)
            journal(String(format: "  ✓ %.0f s, %.0f MB/s%@", s, Double(size) / s / 1e6, sha == nil ? "" : ", sha256 verified"))
            return
        }
        let tracking = Tracker(label: label, cancellation: cancellation, journal: journal)
        let session = URLSession(configuration: .default, delegate: tracking, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let temporary: URL, response: URLResponse
        do {
            (temporary, response) = try wait { try await session.download(for: request(u)) }
        } catch let e as URLError where e.code == .cancelled && cancellation?.isCancelled == true {
            throw EngineError.cancelled
        } catch is URLError {
            throw EngineError.downloadInterrupted(file: label)
        }
        guard let h = response as? HTTPURLResponse, h.statusCode == 200 else {
            try? fm.removeItem(at: temporary)
            let code = (response as? HTTPURLResponse)?.statusCode
            throw EngineError.downloadRefused(file: label, status: code ?? 0)
        }
        let partial = destination + ".partiel"
        try? fm.removeItem(atPath: partial)
        try fm.moveItem(at: temporary, to: URL(fileURLWithPath: partial))
        // Held until the file has its name: `Library.emptyDownloads` leaves a locked `.partiel` alone.
        let held = Library.holdPartial(partial)
        defer { if held >= 0 { close(held) } }
        if let size, (try fm.attributesOfItem(atPath: partial)[.size] as? Int) != size {
            try? fm.removeItem(atPath: partial)
            throw EngineError.downloadInterrupted(file: label)
        }
        if let sha {
            let received = try sha256(partial)
            guard received == sha else { try? fm.removeItem(atPath: partial); throw EngineError.downloadCorrupt(file: label) }
        }
        try fm.moveItem(atPath: partial, toPath: destination)
        let s = Date().timeIntervalSince(begin)
        if let size, size > 50_000_000 {
            journal(String(format: "  ✓ %.0f s, %.0f MB/s%@", s, Double(size) / s / 1e6, sha == nil ? "" : ", sha256 verified"))
        }
    }

    // MARK: The large files, in chunks

    /// Above this size a file comes in chunks; below, in one request (configs, tokenizers).
    static let chunkedFrom = 128 << 20
    static let chunkSize = 64 << 20
    static let parallelChunks = 8
    /// Attempts in a row **without a byte** before a chunk gives up; one that received bytes starts
    /// the count again.
    static let attemptsWithoutProgress = 6

    /// **Chunk bounds by endpoints**, never `size × index`: chunk i is `[i·size/n, (i+1)·size/n)`,
    /// n the fewest chunks of at most `chunk` bytes — contiguous, covering, never empty.
    package static func chunks(size: Int, chunk: Int = chunkSize) -> [Range<Int>] {
        let n = max(1, (size + chunk - 1) / chunk)
        return (0..<n).map { i in i * size / n ..< (i + 1) * size / n }
    }

    /// The ledger of a chunked download: its first line names the file (size, sha256), each next
    /// line a chunk written whole. A ledger of another file, or unreadable, resumes nothing.
    package static func resumedChunks(ledger: String, size: Int, sha: String?) -> Set<Int> {
        let lines = ledger.split(separator: "\n", omittingEmptySubsequences: false)
        // Only whole lines: the last element is "" after a final newline, or a line a crash cut.
        guard lines.count >= 2, lines[0] == "\(size) \(sha ?? "-")" else { return [] }
        return Set(lines.dropFirst().dropLast().compactMap { Int($0) })
    }

    /// **A large file in chunks of 64 MB, 8 at a time, each over a connection of its own.** One
    /// long request does not hold: measured 2026-10-09 on the Hub's CDN, the installer's
    /// single connection had fallen to 2–3 MB/s after half an hour, then stopped for over 60 s
    /// (the request's timeout: 3.75 GB of 4.41 lost), while a fresh connection to the same file
    /// read 30 MB/s and eight 51. So: every attempt opens a new session (`ephemeral`, nothing
    /// shared — with one session HTTP/2 would carry the eight on one connection); a chunk that
    /// stalls 30 s is asked again from its first missing byte; the bytes go to their offset in
    /// `.partiel` as they arrive (no chunk held in memory); the chunks written whole are kept in a
    /// ledger beside it (`.morceaux.partiel`, cleaned like any `.partiel`), so that a download
    /// cut short — the network, a stop, a crash — resumes where it was. The sha256 judges the
    /// whole file at the end, as before; a corrupt one starts over.
    static func downloadInChunks(_ u: URL, size: Int, sha: String?, label: String, to destination: String,
                                 cancellation: Cancellation?, journal: @escaping @Sendable (String) -> Void) throws {
        let fm = FileManager.default
        let partial = destination + ".partiel", ledgerPath = destination + ".morceaux.partiel"
        let all = chunks(size: size)
        var done: Set<Int> = []
        if (try? fm.attributesOfItem(atPath: partial)[.size] as? Int) == size,
           let text = try? String(contentsOfFile: ledgerPath, encoding: .utf8) {
            done = resumedChunks(ledger: text, size: size, sha: sha)
        }
        if done.isEmpty {
            try? fm.removeItem(atPath: partial)
            try "\(size) \(sha ?? "-")\n".write(toFile: ledgerPath, atomically: true, encoding: .utf8)
        } else {
            journal(String(format: "  resumed: %d of %d chunks already there", done.count, all.count))
        }
        let fd = open(partial, O_RDWR | O_CREAT, 0o644)
        guard fd >= 0, ftruncate(fd, off_t(size)) == 0 else {
            if fd >= 0 { close(fd) }
            throw EngineError.fileUnreadable(path: partial)
        }
        // Held while written: `Library.emptyDownloads` leaves a locked `.partiel` alone.
        _ = flock(fd, LOCK_EX | LOCK_NB)
        defer { close(fd) }
        guard let ledger = FileHandle(forWritingAtPath: ledgerPath) else { throw EngineError.fileUnreadable(path: ledgerPath) }
        _ = flock(ledger.fileDescriptor, LOCK_EX | LOCK_NB)
        defer { try? ledger.close() }
        _ = try? ledger.seekToEnd()

        let run = ChunkRun(fd: fd, ledger: ledger, size: size, label: label,
                           already: done.reduce(0) { $0 + all[$1].count }, cancellation: cancellation, journal: journal)
        let pending = all.indices.filter { !done.contains($0) }
        let next = Atomic<Int>(0)
        DispatchQueue.concurrentPerform(iterations: min(parallelChunks, pending.count)) { _ in
            while run.failure == nil {
                let k = next.wrappingAdd(1, ordering: .relaxed).oldValue
                guard k < pending.count else { return }
                run.fetch(u, chunk: pending[k], range: all[pending[k]])
            }
        }
        if let e = run.failure { throw e }
        try? ledger.close()
        try? fm.removeItem(atPath: ledgerPath)
        if let sha {
            let received = try sha256(partial)
            guard received == sha else { try? fm.removeItem(atPath: partial); throw EngineError.downloadCorrupt(file: label) }
        }
        try fm.moveItem(atPath: partial, toPath: destination)
    }

    /// The state the chunks of one file share: the descriptor they write to, the ledger, the bytes
    /// received (for the journal) and the first failure, which stops the others.
    final class ChunkRun: @unchecked Sendable {
        let fd: Int32, ledger: FileHandle, size: Int, label: String
        let cancellation: Cancellation?, journal: @Sendable (String) -> Void
        private let lock = NSLock()
        private var received: Int, reported = 0, _failure: EngineError?

        init(fd: Int32, ledger: FileHandle, size: Int, label: String, already: Int,
             cancellation: Cancellation?, journal: @escaping @Sendable (String) -> Void) {
            self.fd = fd; self.ledger = ledger; self.size = size; self.label = label
            self.received = already; self.cancellation = cancellation; self.journal = journal
            reported = Int(already * 100 / size) / 20 * 20
        }

        var failure: EngineError? { lock.withLock { _failure } }
        func fail(_ e: EngineError) { lock.withLock { if _failure == nil { _failure = e } } }
        var stopped: Bool { cancellation?.isCancelled == true || failure != nil }

        func add(_ n: Int) {
            let line: String? = lock.withLock {
                received += n
                let pc = Int(received * 100 / size)
                guard pc >= reported + 20 else { return nil }
                reported = pc - pc % 20
                return "  … \(reported) %"
            }
            if let line { journal(line) }
        }

        /// One chunk, asked again from its first missing byte until whole.
        func fetch(_ u: URL, chunk: Int, range: Range<Int>) {
            var at = range.lowerBound, idle = 0
            while at < range.upperBound {
                if cancellation?.isCancelled == true { fail(.cancelled); return }
                if failure != nil { return }
                let attempt = ChunkAttempt(run: self, at: at, end: range.upperBound)
                let configuration = URLSessionConfiguration.ephemeral
                configuration.timeoutIntervalForRequest = 30
                let session = URLSession(configuration: configuration, delegate: attempt, delegateQueue: nil)
                var r = Downloader.request(u)
                r.setValue("bytes=\(at)-\(range.upperBound - 1)", forHTTPHeaderField: "Range")
                session.dataTask(with: r).resume()
                attempt.finished.wait()
                session.finishTasksAndInvalidate()
                if let refused = attempt.refused { fail(.downloadRefused(file: label, status: refused)); return }
                if attempt.at > at { idle = 0 } else { idle += 1 }
                at = attempt.at
                if at < range.upperBound {
                    guard idle < Downloader.attemptsWithoutProgress else { fail(.downloadInterrupted(file: label)); return }
                    if idle > 0 { Thread.sleep(forTimeInterval: Double(1 << min(idle, 4))) }
                }
            }
            let line = Data("\(chunk)\n".utf8)
            lock.withLock { try? ledger.write(contentsOf: line) }
        }
    }

    /// One request for `[at, end)`: the bytes written at their offset as they arrive.
    final class ChunkAttempt: NSObject, URLSessionDataDelegate, @unchecked Sendable {
        let run: ChunkRun, end: Int
        var at: Int, refused: Int?
        let finished = DispatchSemaphore(value: 0)
        init(run: ChunkRun, at: Int, end: Int) { self.run = run; self.at = at; self.end = end }

        func urlSession(_ s: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void) {
            // Only a 206 starting at `at`: a 200 would be the whole file written at this offset.
            let h = response as? HTTPURLResponse
            guard h?.statusCode == 206,
                  h?.value(forHTTPHeaderField: "Content-Range")?.hasPrefix("bytes \(at)-") == true else {
                // A server error is passing (asked again); anything else is a refusal.
                if let c = h?.statusCode, !(500..<600).contains(c) { refused = c }
                completionHandler(.cancel); return
            }
            completionHandler(.allow)
        }

        func urlSession(_ s: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
            if run.stopped { dataTask.cancel(); return }
            let n = min(data.count, end - at)
            let written = data.withUnsafeBytes { pwrite(run.fd, $0.baseAddress, n, off_t(at)) }
            guard written == n else { run.fail(.fileUnreadable(path: run.label)); dataTask.cancel(); return }
            at += n
            run.add(n)
        }

        func urlSession(_ s: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            finished.signal()
        }
    }

    package static func sha256(_ path: String) throws -> String {
        guard let f = FileHandle(forReadingAtPath: path) else { throw EngineError.fileUnreadable(path: path) }
        defer { try? f.close() }
        var h = SHA256()
        // **A pool per chunk**: `read(upToCount:)` hands back an autoreleased buffer, and a caller
        // with no pool that drains (the CLI's main thread, one long `install` call) kept every
        // chunk — all the sources' bytes in dirty "Malloc Large", 31 GB and 25 GB of swap for
        // Z-Image (2026-10-06). With the pool: 69 MB, whatever the file's size.
        while try autoreleasepool(invoking: {
            let d = try f.read(upToCount: 64 << 20) ?? Data()
            h.update(data: d)
            return !d.isEmpty
        }) {}
        return h.finalize().map { String(format: "%02x", $0) }.joined()
    }

    final class Tracker: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
        let label: String
        let cancellation: Cancellation?
        let journal: @Sendable (String) -> Void
        var last = 0
        init(label: String, cancellation: Cancellation?, journal: @escaping @Sendable (String) -> Void) {
            self.label = label; self.cancellation = cancellation; self.journal = journal
        }
        func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
        func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didWriteData _: Int64,
                        totalBytesWritten written: Int64, totalBytesExpectedToWrite total: Int64) {
            if cancellation?.isCancelled == true { downloadTask.cancel(); return }
            guard total > 200_000_000 else { return }
            let pc = Int(written * 100 / total)
            if pc >= last + 20 { last = pc - pc % 20; journal("  … \(last) %") }
        }
    }
}

/// Waits for an asynchronous task from synchronous code (the forge and the CLI are).
private final class Box<T>: @unchecked Sendable { var r: Result<T, Error>? }

func wait<T: Sendable>(_ task: @escaping @Sendable () async throws -> T) throws -> T {
    let b = Box<T>(), s = DispatchSemaphore(value: 0)
    Task.detached {
        do { b.r = .success(try await task()) } catch { b.r = .failure(error) }
        s.signal()
    }
    s.wait()
    return try b.r!.get()
}
