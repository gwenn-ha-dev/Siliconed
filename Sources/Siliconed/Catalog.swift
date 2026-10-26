import Foundation

// **The catalog: what an app displays before rendering — which models, which LoRAs, and for what.**
//
// It never throws: an app lists what is there, it does not fail because a map is missing. A
// model's availability is judged by constructing it (`Model.named(_:in:)`, which checks every
// file) — a single list of required files lives, the constructors'. A LoRA is read by its header
// alone (a few KB), never by its tensors.

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

    /// **The largest of `formats` whose need fits in `budget` bytes** — B, this Mac's
    /// (`MemoryBudget.machineCeiling()`) — or `nil`: what the app offers instead of a dead end when B
    /// refuses the chosen size. Pure: the caller passes B and the need, never the machine's state.
    ///
    /// The orientation the user chose comes first — a portrait stays a portrait while one fits —
    /// then the area. Nothing under `Format.minimumSide` on a side: below 512² the models are out
    /// of their domain, and a format there is no way forward. `current` is never offered back.
    package static func largestFitting(
        _ formats: [RecommendedFormat], budget: Int, current: RecommendedFormat,
        need: (RecommendedFormat) -> Int
    ) -> RecommendedFormat? {
        let fitting = formats.filter {
            $0 != current && min($0.width, $0.height) >= Format.minimumSide && need($0) <= budget
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

    package static let zImage = ModelCard(
        id: "z-image", name: "Z-Image Turbo", family: .zImage, map: nil,
        license: License(text: "Apache 2.0", commercial: true, requiredFilter: false, urls: Installer.licenseURLs(.zImage)),
        defaultSteps: 8, space: .flux, formats: recommendedFormats)
    /// ⚠️ Non-commercial (CircleStone Labs Non-Commercial License; the model derives from NVIDIA's
    /// Cosmos-Predict2, NVIDIA Open Model License): never the default of a product that is sold. Its
    /// images, the license's « Outputs », may be used commercially (§ 2.e).
    package static let anima = ModelCard(
        id: "anima", name: "Anima Turbo", family: .anima, map: nil,
        license: License(text: "CircleStone Labs Non-Commercial", commercial: false, requiredFilter: false,
                         urls: Installer.licenseURLs(.anima)),
        defaultSteps: 8, space: .qwenImage, formats: recommendedFormats)
    /// ⚖️ Commercial under $1M annual revenue, mandatory content filter at deployment, "Krea" at
    /// the head of a distributed model.
    package static let krea2 = ModelCard(
        id: "krea2", name: "Krea 2 Turbo", family: .krea2, map: nil,
        license: License(text: "Krea 2 Community (commercial < $1M, mandatory content filter)",
                         commercial: true, requiredFilter: true, urls: Installer.licenseURLs(.krea2)),
        defaultSteps: 8, space: .qwenImage, formats: recommendedFormats)

    /// **Qwen-Image-2.1-Turbo (Qwen)**: 8 steps on its published schedule (any other count is
    /// refused), no CFG.
    ///
    /// - **What it is for**: generation, and **editing by instruction** — 1 to 3 reference images
    ///   (`maxReferences`) that its Qwen3-VL-8B encoder *sees*, so the prompt can name them
    ///   (“put the dog of image 2 next to the woman in image 1”). Image 1 is the one edited and sets
    ///   the output's format (`editFormat`). No img2img: a starting image is refused, pass it as image 1.
    /// - ⚠️ **License**: Qwen Research — non-commercial. Never a product's default.
    /// - **Weights** (`Family.installedSize(.standard)`, bf16 maps): DiT 14.2 GB, encoder with its vision tower
    ///   16.3 GB, components 0.69 GB (the VAE in bf16, the processor) — 31.2 GB on disk.
    /// - **Cost** (M1 Pro 16 GB): one DiT evaluation at 1024² takes ~17 s; a 1024² generation ~120 s
    ///   (8 steps, the first two at half size), an edit at 1248×832 with one reference ~210 s.
    package static let qwenImage21 = ModelCard(
        id: "qwen-image-2.1", name: "Qwen-Image-2.1 Turbo", family: .qwenImage21, map: nil,
        license: License(text: "Qwen Research (non-commercial)", commercial: false, requiredFilter: false,
                         urls: Installer.licenseURLs(.qwenImage21)),
        defaultSteps: 8, space: .qwenImage21, formats: recommendedFormats)

    /// **MOSS-Image under the community's 8-step LoRA** (OpenMOSS-Team's SFT model, lodestones' DMD distillation at
    /// strength 1; no official turbo exists as of the pinned revisions): single-stream DiT of 5.08 B, Qwen3.5-9B
    /// encoder (hybrid Gated DeltaNet), FLUX.2 VAE, 8 steps without CFG. Apache 2.0, commercial. Standard only.
    package static let mossImage = ModelCard(
        id: "moss-image", name: "MOSS-Image", family: .mossImage, map: nil,
        license: License(text: "Apache 2.0", commercial: true, requiredFilter: false, urls: Installer.licenseURLs(.mossImage)),
        defaultSteps: 8, space: .flux2, formats: recommendedFormats)

    /// The repository's models, in the order of `Model.identifiers`.
    package static let allCards: [ModelCard] = [zImage, anima, krea2, qwenImage21, mossImage]

    /// A family's sheet — that of its published model.
    package static func of(_ family: Family) -> ModelCard {
        switch family {
        case .zImage: return zImage; case .anima: return anima; case .krea2: return krea2
        case .qwenImage21: return qwenImage21
        case .mossImage: return mossImage
        }
    }

    package static func withID(_ id: String) -> ModelCard? { allCards.first { $0.id == id } }

    /// What is missing to build the model (the error's message), or `nil` if it is ready.
    /// The model itself: `Model.named(card.id, in:)`. Does not read the machine's profile.
    package func missing(in library: Library) -> String? {
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
package struct LoRACard: Sendable, Identifiable, Hashable {
    package var id: String { path }
    package let path: String
    /// `Niji semi realism v5` — the header's `name`, underscores turned into spaces.
    package let name: String
    /// The header's `name` as written: the imported file's name, without its extension (the
    /// trainer's own name for a map forged before an import took the file's).
    package let rawName: String
    /// The identifier of the targeted model (`cible.modele`, written by `ForgeLoRA` from the DiT of
    /// the family against which it checked every shape).
    package let target: String
    /// The largest rank. A LoRA trained with per-module ranks (Incase on Anima: 2 to 57) has its
    /// header's `rang` written as their sorted list, not one number.
    package let rank: Int
    /// The smallest rank: `rank` itself when every module has the same.
    package let smallestRank: Int
    /// The trainer's `modelspec.resolution`, if it said.
    package let resolution: String?
    /// The declared training model (`ss_sd_model_name`) — Krea-2-**Raw** for Incase.
    package let trainedOn: String?
    /// The name the trainer wrote (`trainer_name`, `ForgeLoRA.trainerName`), when it is not the one
    /// shown — case, underscores and dashes aside. `nil` for a map forged before the key.
    package let trainerName: String?
    /// The trigger words, when the file states them beyond doubt (`triggers`, `ForgeLoRA.triggers`);
    /// empty when it does not, and for a map forged before the key.
    package let triggers: [String]

    /// From a map's header; `nil` if it is not a LoRA, or if it does not state its target (it
    /// would be refused at render time: forge it again).
    package init?(header: [String: Any], path: String) {
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
        self.target = target
        let ranks = header["rang"] as? [Int] ?? (header["rang"] as? Int).map { [$0] } ?? []
        self.rank = ranks.max() ?? 0; self.smallestRank = ranks.min() ?? 0
        self.resolution = meta("modelspec.resolution"); self.trainedOn = meta("ss_sd_model_name")
        let trainer = (header["trainer_name"] as? String).flatMap { $0.isEmpty ? nil : $0 }, shown = name
        self.trainerName = trainer.flatMap {
            LoRACard.readableName($0).caseInsensitiveCompare(shown) == .orderedSame ? nil : $0
        }
        self.triggers = header["triggers"] as? [String] ?? []
    }

    /// `Niji_semi_realism_v5` → `Niji semi realism v5`; `.lora` and the extension removed.
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
    /// **The models ready to render** — those whose maps and published files are all present.
    /// Never throws; `ModelCard.missing(in:)` says what the others are missing.
    package func models() -> [ModelCard] {
        cards().filter { $0.missing(in: self) == nil }
    }

    /// **All known models**: the publisher's, then the imported ones.
    package func cards() -> [ModelCard] { ModelCard.allCards + importedCards() }

    /// The DiTs of `store/importes/*.silicon`, read by their header (`name`, `kind`). An import of a
    /// superseded kind is removed first (`removeSupersededImports`): no file stays on disk that the
    /// app does not show.
    package func importedCards() -> [ModelCard] {
        removeSupersededImports()
        let names = (try? FileManager.default.contentsOfDirectory(atPath: imported.path)) ?? []
        return names.filter { $0.hasSuffix(".silicon") }.sorted().compactMap { name in
            let path = imported.appendingPathComponent(name).path
            guard let header = try? Artifact.header(path), let kind = header["kind"] as? String,
                  let family = Family(ditKind: kind) else { return nil }
            let slug = String(name.dropLast(".silicon".count))
            return ModelCard(isImported: header["nom"] as? String ?? slug, slug: slug, family: family, map: path)
        }
    }

    /// **Removes the imported DiTs of a superseded kind** — Qwen-Image-2.1 checkpoints forged while the
    /// family rendered the base model under a third-party turbo LoRA (`Family.supersededQwenImage21Kind`,
    /// the kind of every Qwen map then, Standard, Compact, Light or imported). The engine refuses them
    /// and the catalog does not list them (`Family(ditKind:)` is nil): kept, they would occupy the disk
    /// out of sight. Their map and what goes with it (`adapterPath(fromMap:)`) are removed, one journal
    /// line each (stderr by default). Returns the maps removed.
    @discardableResult
    package func removeSupersededImports(journal: (String) -> Void = { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }) -> [String] {
        let fm = FileManager.default
        let names = (try? fm.contentsOfDirectory(atPath: imported.path)) ?? []
        var removed: [String] = []
        for name in names.sorted() where name.hasSuffix(".silicon") {
            let path = imported.appendingPathComponent(name).path
            guard let header = try? Artifact.header(path), header["kind"] as? String == Family.supersededQwenImage21Kind else { continue }
            let bytes = Self.size(path) + Self.size(adapterPath(fromMap: path))
            do { try fm.removeItem(atPath: path) } catch { continue }
            try? fm.removeItem(atPath: adapterPath(fromMap: path))
            removed.append(path)
            journal(String(format: "removed the imported model \"%@\" (%@): forged for Qwen-Image-2.1's base model and its "
                           + "third-party turbo LoRA, which the family no longer renders (%.2f GB freed)",
                           header["nom"] as? String ?? String(name.dropLast(".silicon".count)), name, Double(bytes) / 1e9))
        }
        return removed
    }

    /// **The forged LoRAs of `store/*.lora.silicon`**, read by their header — `for`: only those
    /// targeting this model. An unreadable map, or one without a target, is skipped.
    package func loras(for model: String? = nil) -> [LoRACard] {
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
    package func lora(_ name: String, for model: String) -> LoRACard? {
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
