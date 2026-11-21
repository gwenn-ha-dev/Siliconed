import Foundation

// **The catalog: what an app displays before rendering — which models, and for what.**
//
// It never throws: an app lists what is there, it does not fail because a file is missing. A
// model's availability is judged by constructing it (`Model.named(_:in:)`, which checks every
// file) — a single list of required files lives, the constructors'.

/// A license, read to decide — not merely to be displayed.
package struct License: Sendable, Hashable, CustomStringConvertible {
    package let text: String
    /// Can an image or the model serve a product that is sold?
    package let commercial: Bool
    /// Does the license require a content filter at deployment? (Krea 2 Community, §4.2)
    package let requiredFilter: Bool
    /// **Where to read it**: the license file(s) on Hugging Face, at the revision the installation
    /// downloads (`https://huggingface.co/<repo>/blob/<sha>/<file>`) — the model's first, then each
    /// component under a license of its own (Anima: its DiT's repository, then its Diffusers one).
    /// Z-Image publishes no license file: its model card (`README.md`), which declares Apache 2.0.
    /// Krea 2's `LICENSE.pdf` is read from Comfy-Org's copy (Krea's repository is gated).
    /// An imported model: its family's, the architecture it derives from.
    package let urls: [URL]
    package var description: String { text }
}

/// **A recommended format, Draw Things style**: a name, width × height in pixels. All pass
/// `Format.check` (the test target judges it).
package struct RecommendedFormat: Sendable, Hashable, CustomStringConvertible {
    package enum Orientation: String, Sendable { case square, portrait, landscape }
    package let width: Int
    package let height: Int
    package var orientation: Orientation {
        width == height ? .square : (height > width ? .portrait : .landscape)
    }
    package init(_ width: Int, _ height: Int) { self.width = width; self.height = height }
    /// `832×1216 (portrait)`.
    package var description: String { "\(Format.label(width: width, height: height)) (\(orientation.rawValue))" }
}

/// **What is known of a model without opening it** — name, license, steps, formats.
/// `Hashable`: a sheet serves as is as a `Picker`'s `tag`.
package struct ModelCard: Sendable, Identifiable, Hashable {
    /// `z-image`, `anima`, `krea2`, `qwen-image-2.1` — `Model.identifier`; `<family>/<name>` for an
    /// imported one.
    package let id: String
    package let name: String
    /// The architecture: the target a LoRA declares, and where tokenizers, encoder, VAE come from.
    package let family: Family
    /// The DiT map of an imported model (`nil`: the publisher's).
    package let map: String?
    /// Imported by the user, rather than published by the family's publisher.
    package var isImported: Bool { map != nil }
    package let license: License
    /// The denoiser's (Z-Image 8, Qwen-Image-2.1 8); the test target checks that they agree.
    package let defaultSteps: Int
    package let space: LatentSpace
    /// The square first, then portraits and landscapes, from stockiest to most elongated.
    package let formats: [RecommendedFormat]

    /// **Draw Things' formats**: 1024², 896×1152, 832×1216, 768×1344 and their transposes, plus
    /// 512² for quick iteration, and the large squares 1536² and 2048² for the Macs whose B holds
    /// them (the picker greys what this Mac's B refuses). Any other size is typed, and B decides
    /// (`Format`). A large square costs more than its area: attention is quadratic in the tokens, so
    /// 2048² is about 8 times 1024², not 4.
    package static let recommendedFormats: [RecommendedFormat] = [
        RecommendedFormat(1024, 1024), RecommendedFormat(1536, 1536), RecommendedFormat(2048, 2048),
        RecommendedFormat(896, 1152), RecommendedFormat(832, 1216), RecommendedFormat(768, 1344),
        RecommendedFormat(1152, 896), RecommendedFormat(1216, 832), RecommendedFormat(1344, 768),
        RecommendedFormat(512, 512),
    ]

    /// The repository's models, in the order of `Model.identifiers`. A family's card lives in its folder.
    package static let allCards: [ModelCard] = Family.allCases.map(\.traits.card)

    /// A family's sheet — that of its published model.
    package static func of(_ family: Family) -> ModelCard {
        family.traits.card
    }

    package static func withID(_ id: String) -> ModelCard? { allCards.first { $0.id == id } }

    /// What is missing to build the model (the error's message), or `nil` if it is ready: its chain constructed over
    /// this card, every file checked — no header read for its formats. The model itself: `Model.named(card.id, in:)`.
    package func missing(in library: Library) -> String? {
        do { _ = try Model.construct(self, in: library); return nil } catch { return "\(error)" }
    }
}

extension Library {
    /// **The models ready to render** — those whose maps and published files are all present.
    /// Never throws; `ModelCard.missing(in:)` says what the others are missing.
    package func models() -> [ModelCard] {
        cards().filter { $0.missing(in: self) == nil }
    }

    /// **All known models**: the publishers', then the imported ones (`importedCards`).
    package func cards() -> [ModelCard] { ModelCard.allCards + importedCards() }
}

/// **A LoRA the library holds, as an app lists it**: for which model, which rank, which name — read in its published
/// file (`Library.loras(for:)`, `LoRA.swift`).
package struct LoRACard: Sendable, Identifiable, Hashable {
    package var id: String { path }
    package let path: String
    /// `Niji semi realism v5` — `rawName`, underscores turned into spaces.
    package let name: String
    /// The file's name, without its extension.
    package let rawName: String
    /// The identifier of the targeted model's family.
    package let target: String
    /// The largest rank, and the smallest: equal unless the LoRA was trained with a rank a module.
    package let rank: Int
    package let smallestRank: Int
    /// What the file says of itself, when it does: the trainer's resolution (`modelspec.resolution`), the model it was
    /// trained on (`ss_sd_model_name`), the name the trainer wrote when it is not the one shown, its trigger words.
    package let resolution: String?
    package let trainedOn: String?
    package let trainerName: String?
    package let triggers: [String]

    package init(path: String, rawName: String, target: String, rank: Int, smallestRank: Int, resolution: String? = nil,
                 trainedOn: String? = nil, trainerName: String? = nil, triggers: [String] = []) {
        self.path = path; self.rawName = rawName; self.name = LoRACard.readableName(rawName)
        self.target = target; self.rank = rank; self.smallestRank = smallestRank
        self.resolution = resolution; self.trainedOn = trainedOn; self.trainerName = trainerName; self.triggers = triggers
    }

    /// **The name shown for the LoRA at `path`**, its card's, read from the path alone: what the window writes of an image's,
    /// a grid's or a sheet's LoRA does not depend on the model the rack holds — a LoRA of another model is not in its list.
    package static func name(path: String) -> String { readableName(rawName(path: path)) }

    /// The file's name without its extension: a card's `rawName`.
    package static func rawName(path: String) -> String { ((path as NSString).lastPathComponent as NSString).deletingPathExtension }

    /// `Niji_semi_realism_v5` → `Niji semi realism v5`: `.lora` removed wherever it appears, `_` and `-` spaces.
    package static func readableName(_ raw: String) -> String {
        raw.replacingOccurrences(of: ".lora", with: "")
            .replacingOccurrences(of: "_", with: " ").replacingOccurrences(of: "-", with: " ")
            .split(separator: " ").joined(separator: " ")
    }

    /// `model`: a model or family identifier — a LoRA targets an architecture.
    package func compatible(with model: String) -> Bool {
        target == (model.split(separator: "/").first.map(String.init) ?? model)
    }

    /// The "compatible with this model" filter, sorted by name.
    package static func filter(_ cards: [LoRACard], for model: String?) -> [LoRACard] {
        cards.filter { model == nil || $0.compatible(with: model!) }
            .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }
}

extension Library {
    /// **A LoRA designated as one writes it**: its path, its file, its file without the extension or its name, case
    /// ignored — among those targeting `model`. `nil` if none, or if the name designates several.
    package func lora(_ name: String, for model: String) -> LoRACard? {
        let compatible = loras(for: model)
        if let exact = compatible.first(where: { $0.path == name }) { return exact }
        let n = name.lowercased()
        let candidates = compatible.filter { f in
            let file = (f.path as NSString).lastPathComponent.lowercased()
            return file == n || (file as NSString).deletingPathExtension == n
                || f.name.lowercased() == n || f.rawName.lowercased() == n
        }
        return candidates.count == 1 ? candidates[0] : nil
    }
}
