import Foundation

// **The catalog: what an app displays before rendering — which models, which LoRAs, and for what.**
//
// It never throws: an app lists what is there, it does not fail because a map is missing. A
// model's availability is judged by constructing it (`Model.named(_:in:)`, which checks every
// file) — a single list of required files lives, the constructors'. A LoRA is read by its header
// alone (a few KB), never by its tensors.

/// A license, read to decide — not merely to be displayed.
public struct License: Sendable, Hashable, CustomStringConvertible {
    public let text: String
    /// Can an image or the model serve a product that is sold?
    public let commercial: Bool
    /// Does the license require a content filter at deployment? (Krea 2 Community, §4.2)
    public let requiredFilter: Bool
    /// **Where to read it**: the license file(s) on Hugging Face, at the revision the installation
    /// downloads (`https://huggingface.co/<repo>/blob/<sha>/<file>`) — the model's first, then each
    /// component under a license of its own (Qwen-Image-2.1: the model, then Viggle's turbo LoRA).
    /// Z-Image publishes no license file: its model card (`README.md`), which declares Apache 2.0.
    /// Krea 2's repository is gated: its file opens once the Hub's gate is accepted.
    /// An imported model: its family's, the architecture it derives from.
    public let urls: [URL]
    public var description: String { text }
}

/// **A recommended format, Draw Things style**: a name, width × height in pixels. All pass
/// `Format.check` (the test target judges it).
public struct RecommendedFormat: Sendable, Hashable, CustomStringConvertible {
    public enum Orientation: String, Sendable { case square, portrait, landscape }
    public let width: Int
    public let height: Int
    public var orientation: Orientation {
        width == height ? .square : (height > width ? .portrait : .landscape)
    }
    public init(_ width: Int, _ height: Int) { self.width = width; self.height = height }
    /// `832×1216 (portrait)`.
    public var description: String { "\(Format.label(width: width, height: height)) (\(orientation.rawValue))" }

    /// **The largest of `formats` whose need fits in `available` bytes**, or `nil`: what the app
    /// offers instead of a dead end when the machine's memory refuses the chosen size. Pure: the
    /// caller reads the budget **once** and passes it, rather than each format asking the machine.
    ///
    /// The orientation the user chose comes first — a portrait stays a portrait while one fits —
    /// then the area. Nothing under `Format.minimumSide` on a side: below 512² the models are out
    /// of their domain, and a format there is no way forward. `current` is never offered back.
    public static func largestFitting(
        _ formats: [RecommendedFormat], available: Int, current: RecommendedFormat,
        need: (RecommendedFormat) -> Int
    ) -> RecommendedFormat? {
        let fitting = formats.filter {
            $0 != current && min($0.width, $0.height) >= Format.minimumSide && need($0) <= available
        }
        let pool = fitting.contains { $0.orientation == current.orientation }
            ? fitting.filter { $0.orientation == current.orientation } : fitting
        // `max(by:)` keeps the first of equals: the first listed wins a tie (896×1152 and 768×1344
        // have the same area), the catalog's order.
        return pool.max { $0.width * $0.height < $1.width * $1.height }
    }
}

/// **What is known of a model without opening it** — name, license, steps, formats.
/// `Hashable`: a sheet serves as is as a `Picker`'s `tag`.
public struct ModelCard: Sendable, Identifiable, Hashable {
    /// `z-image`, `anima`, `krea2` — `Model.identifier`; `<famille>/<nom>` for an imported one.
    public let id: String
    public let name: String
    /// The architecture: the target a LoRA declares, and where tokenizers, encoder, VAE come from.
    public let family: Family
    /// The DiT map of an imported model (`nil`: the publisher's).
    public let map: String?
    /// Imported by the user, rather than published by the family's publisher.
    public var isImported: Bool { map != nil }
    public let license: License
    /// The denoiser's (Z-Image 8, Qwen-Image-2.1 6, FLUX.2 [klein] 4); the test target checks that they agree.
    public let defaultSteps: Int
    public let space: LatentSpace
    /// The square first, then portraits and landscapes, from stockiest to most elongated.
    public let formats: [RecommendedFormat]

    /// **Draw Things' formats, under the measured ceiling** (`Format.maxSurface`, 1024×1536):
    /// 1024², 896×1152, 832×1216, 768×1344 and their transposes, plus 512² for quick
    /// iteration. 1024×1536 is accepted but not recommended (double the time of a 1024² on
    /// Krea 2); 640×1536 is left aside: never measured.
    public static let recommendedFormats: [RecommendedFormat] = [
        RecommendedFormat(1024, 1024),
        RecommendedFormat(896, 1152), RecommendedFormat(832, 1216), RecommendedFormat(768, 1344),
        RecommendedFormat(1152, 896), RecommendedFormat(1216, 832), RecommendedFormat(1344, 768),
        RecommendedFormat(512, 512),
    ]

    public static let zImage = ModelCard(
        id: "z-image", name: "Z-Image Turbo", family: .zImage, map: nil,
        license: License(text: "Apache 2.0", commercial: true, requiredFilter: false, urls: Installer.licenseURLs(.zImage)),
        defaultSteps: 8, space: .flux, formats: recommendedFormats)
    /// ⚠️ Non-commercial: never the default of a product that is sold.
    public static let anima = ModelCard(
        id: "anima", name: "Anima Turbo", family: .anima, map: nil,
        license: License(text: "non-commercial (circlestone-labs)", commercial: false, requiredFilter: false,
                         urls: Installer.licenseURLs(.anima)),
        defaultSteps: 8, space: .qwenImage, formats: recommendedFormats)
    /// ⚖️ Commercial under $1M annual revenue, mandatory content filter at deployment, "Krea" at
    /// the head of a distributed model.
    public static let krea2 = ModelCard(
        id: "krea2", name: "Krea 2 Turbo", family: .krea2, map: nil,
        license: License(text: "Krea 2 Community (commercial < $1M, mandatory content filter)",
                         commercial: true, requiredFilter: true, urls: Installer.licenseURLs(.krea2)),
        defaultSteps: 8, space: .qwenImage, formats: recommendedFormats)

    /// FLUX.2 [klein] 4B, distilled: 4 steps.
    public static let klein4b = ModelCard(
        id: "klein-4b", name: "FLUX.2 [klein] 4B", family: .klein4b, map: nil,
        license: License(text: "Apache 2.0", commercial: true, requiredFilter: false, urls: Installer.licenseURLs(.klein4b)),
        defaultSteps: 4, space: .flux2, formats: recommendedFormats)

    /// ERNIE-Image Turbo (Baidu), distilled: 8 steps.
    public static let ernie = ModelCard(
        id: "ernie-image", name: "ERNIE-Image Turbo", family: .ernie, map: nil,
        license: License(text: "Apache 2.0", commercial: true, requiredFilter: false, urls: Installer.licenseURLs(.ernie)),
        defaultSteps: 8, space: .flux2, formats: recommendedFormats)

    /// **Qwen-Image-2.1 (Qwen) under Viggle's turbo LoRA v0.3**: 6 steps (5, 7, and a 9-step mode
    /// ending on the base model are accepted; any other count is refused), no CFG.
    ///
    /// - **What it is for**: generation, and **editing by instruction** — 1 to 3 reference images
    ///   (`maxReferences`) that its Qwen3-VL-8B encoder *sees*, so the prompt can name them
    ///   (“put the dog of image 2 next to the woman in image 1”). Image 1 is the one edited and sets
    ///   the output's format (`editFormat`). No img2img: a starting image is refused, pass it as image 1.
    /// - ⚠️ **License**: Qwen Research — non-commercial. Never a product's default.
    /// - **Weights** (`Family.installedSize(.standard)`, bf16 maps): DiT 14.2 GB, encoder with its vision tower
    ///   16.3 GB, components 2.7 GB (VAE, processor, the turbo LoRA 1.36 GB) — 33.2 GB on disk.
    /// - **Cost** (M1 Pro 16 GB): one DiT evaluation at 1024² takes ~19 s bare, ~22 s under the turbo
    ///   LoRA the product always applies. Measured in series: generation 1024²
    ///   134–137 s, an edit at 1248×832 190–193 s with one reference, 262–264 s with two; the worst
    ///   case (1024×1536, 3 references) 523 s at a 4.58 GB peak, zero swap-out.
    public static let qwenImage21 = ModelCard(
        id: "qwen-image-2.1", name: "Qwen-Image-2.1 Turbo", family: .qwenImage21, map: nil,
        license: License(text: "Qwen Research (non-commercial)", commercial: false, requiredFilter: false,
                         urls: Installer.licenseURLs(.qwenImage21)),
        defaultSteps: 6, space: .qwenImage21, formats: recommendedFormats)

    /// The repository's models, in the order of `Model.identifiers`.
    public static let allCards: [ModelCard] = [zImage, anima, krea2, klein4b, ernie, qwenImage21]

    /// A family's sheet — that of its published model.
    public static func of(_ family: Family) -> ModelCard {
        switch family {
        case .zImage: return zImage; case .anima: return anima; case .krea2: return krea2; case .klein4b: return klein4b
        case .ernie: return ernie
        case .qwenImage21: return qwenImage21
        }
    }


    public static func withID(_ id: String) -> ModelCard? { allCards.first { $0.id == id } }

    /// What is missing to build the model (the error's message), or `nil` if it is ready.
    /// The model itself: `Model.named(card.id, in:)`. Does not read the machine's profile.
    public func missing(in library: Library) -> String? {
        do { _ = try Model.make(id, in: library); return nil } catch { return "\(error)" }
    }
}

extension ModelCard {
    /// An imported model: its family's architecture, license and formats, its own name.
    init(isImported name: String, slug: String, family: Family, map: String) {
        let f = ModelCard.of(family)
        self.init(id: "\(family.rawValue)/\(slug)", name: name, family: family, map: map,
                  license: License(text: "that of the imported model (architecture \(f.name): \(f.license.text))",
                                   commercial: f.license.commercial, requiredFilter: f.license.requiredFilter,
                                   urls: f.license.urls),
                  defaultSteps: f.defaultSteps, space: f.space, formats: f.formats)
    }
}

/// **A forged LoRA, read by its header**: for which model, which rank, which name.
public struct LoRACard: Sendable, Identifiable, Hashable {
    public var id: String { path }
    public let path: String
    /// `Niji semi realism v5` — the header's `name`, underscores turned into spaces.
    public let name: String
    /// The name as the trainer wrote it.
    public let rawName: String
    /// The identifier of the targeted model (`cible.modele`, written by `ForgeLoRA` from the DiT of
    /// the family against which it checked every shape).
    public let target: String
    public let rank: Int
    /// The trainer's `modelspec.resolution`, if it said.
    public let resolution: String?
    /// The declared training model (`ss_sd_model_name`) — Krea-2-**Raw** for Incase.
    public let trainedOn: String?

    /// From a map's header; `nil` if it is not a LoRA, or if it does not state its target (it
    /// would be refused at render time: forge it again).
    public init?(header: [String: Any], path: String) {
        guard header["kind"] as? String == "lora",
              // on-disk key, kept for existing maps/profiles (.silicon header: cible, modele)
              let target = (header["cible"] as? [String: Any])?["modele"] as? String else { return nil }
        let raw = header["nom"] as? String
            ?? ((path as NSString).lastPathComponent as NSString).deletingPathExtension
        let metadata = (header["source"] as? [String: Any])?["metadata"] as? [String: Any]
        func meta(_ key: String) -> String? {
            (metadata?[key] as? String).flatMap { $0.isEmpty ? nil : $0 }
        }
        self.path = path; self.rawName = raw; self.name = LoRACard.readableName(raw)
        self.target = target; self.rank = header["rang"] as? Int ?? 0
        self.resolution = meta("modelspec.resolution"); self.trainedOn = meta("ss_sd_model_name")
    }

    /// `Niji_semi_realism_v5` → `Niji semi realism v5`; `.lora` and the extension removed.
    package static func readableName(_ raw: String) -> String {
        raw.replacingOccurrences(of: ".lora", with: "")
            .replacingOccurrences(of: "_", with: " ").replacingOccurrences(of: "-", with: " ")
            .split(separator: " ").joined(separator: " ")
    }

    /// `model`: a model or family identifier — a LoRA targets an architecture.
    public func compatible(with model: String) -> Bool {
        target == (model.split(separator: "/").first.map(String.init) ?? model)
    }

    /// A `Request`'s entry.
    public func entry(strength: Double = 1) -> LoRAEntry { LoRAEntry(path, strength: strength) }

    /// The "compatible with this model" filter, sorted by name.
    package static func filter(_ cards: [LoRACard], for model: String?) -> [LoRACard] {
        cards.filter { model == nil || $0.compatible(with: model!) }
            .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }
}

extension Library {
    /// **The models ready to render** — those whose maps and published files are all present.
    /// Never throws; `ModelCard.missing(in:)` says what the others are missing.
    public func models() -> [ModelCard] {
        cards().filter { $0.missing(in: self) == nil }
    }

    /// **All known models**: the publisher's, then the imported ones.
    public func cards() -> [ModelCard] { ModelCard.allCards + importedCards() }

    /// The DiTs of `store/importes/*.silicon`, read by their header (`name`, `kind`).
    public func importedCards() -> [ModelCard] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: imported.path)) ?? []
        return names.filter { $0.hasSuffix(".silicon") }.sorted().compactMap { name in
            let path = imported.appendingPathComponent(name).path
            guard let header = try? Artifact.header(path), let kind = header["kind"] as? String,
                  let family = Family(ditKind: kind) else { return nil }
            let slug = String(name.dropLast(".silicon".count))
            return ModelCard(isImported: header["nom"] as? String ?? slug, slug: slug, family: family, map: path)
        }
    }

    /// **The forged LoRAs of `store/*.lora.silicon`**, read by their header — `for`: only those
    /// targeting this model. An unreadable map, or one without a target, is skipped.
    public func loras(for model: String? = nil) -> [LoRACard] {
        let folder = root.appendingPathComponent("store")
        let names = (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []
        let cards = names.filter { $0.hasSuffix(".lora.silicon") }.compactMap { name -> LoRACard? in
            let path = folder.appendingPathComponent(name).path
            guard let header = try? Artifact.header(path) else { return nil }
            return LoRACard(header: header, path: path)
        }
        return LoRACard.filter(cards, for: model)
    }

    /// **A LoRA designated as one writes it**: its path, its file (`flat.lora.silicon`), its file
    /// without the extension (`flat`) or its name (`Flat color v2`), case ignored — among those
    /// targeting `model`. `nil` if none, or if the name designates several.
    public func lora(_ name: String, for model: String) -> LoRACard? {
        let compatible = loras(for: model)
        if let exact = compatible.first(where: { $0.path == name }) { return exact }
        let n = name.lowercased()
        let candidates = compatible.filter { f in
            let file = (f.path as NSString).lastPathComponent.lowercased()
            return file == n || file == n + ".lora.silicon"
                || f.name.lowercased() == n || f.rawName.lowercased() == n
        }
        return candidates.count == 1 ? candidates[0] : nil
    }
}
