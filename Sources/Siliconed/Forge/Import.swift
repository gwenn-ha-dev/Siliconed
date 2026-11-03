import Foundation

/// **Importing what a user brings**: a `.safetensors` or a `.gguf`, LoRA or model, without VAE or encoder —
/// this is how Civitai publishes. We recognize what it is and for which family, and forge it against
/// what the family's installation left (`dit.json`, the published DiT's names and shapes; its config,
/// encoder and VAE). **An import downloads nothing**: a family not installed is refused
/// (`EngineError.importNeedsFamily`) — its installation shows its license and its size first. The
/// file brought can then be thrown away: the render only reads `store/`.
///
/// **The one thing taken as a component is an encoder or a VAE identical to the one an installation
/// downloads** (`Adoption`): copied where the download would have put it, so that the installation
/// does not fetch it. Any other encoder is refused as before.
///
/// **A publisher's base model is refused where the family renders its Turbo** (`BaseModels`): it has
/// the Turbo's names and shapes, and would be forged and rendered on the Turbo's recipe.
///
/// This is Draw Things' idea (`BaseModelImporter`, `LoRAImporter`): the architecture is detected,
/// only the DiT is converted, the rest is the family's.
package enum ModelImport {
    package enum Kind: String, Sendable { case lora = "LoRA", model = "model", encoder = "encoder", vae = "VAE" }

    package struct Outcome: Sendable {
        package let kind: Kind
        package let family: Family
        /// The displayed name.
        package let name: String
        /// The forged map.
        package let path: String
        /// What happened, line by line (naming recognized, dtypes, roundings, installations).
        package let journal: [String]
        /// An encoder or a VAE only: the families whose installation expects this very file, and whether it
        /// was put where they will find it (`false`: each already has its encoder, nothing was copied).
        package var families: [Family] = []
        package var placed = false
        /// The identifier to pass to `Model.named` (a model) — `nil` for a LoRA.
        package var identifier: String? {
            kind == .model ? "\(family.rawValue)/" + ((path as NSString).lastPathComponent as NSString).deletingPathExtension : nil
        }
    }

    /// What the file is, without forging anything.
    package static func recognize(_ file: String) throws(EngineError) -> (Kind, Family?) {
        try refusingAs(file) {
            // Before the family is read: the app offers to install a family a model lacks.
            try BaseModels.refuse(file)
            let s = try TensorSource(paths: [file])
            if ForgeLoRA.isLoRA(s.names) { return (.lora, nil) }
            return (.model, Recipes.family(fromNames: s.names))
        }
    }

    /// **What the import would make of the file, read no further than its header**: a LoRA
    /// examined by the forge against the installed families (`ForgeLoRA.examine`), a model by its
    /// names. A LoRA for a family not installed is `importNeedsFamily`, as at the import.
    package static func examine(_ file: String, in b: Library) throws(EngineError) -> (Kind, Family) {
        try refusingAs(file) {
            let s = try TensorSource(paths: [file])
            if ForgeLoRA.isLoRA(s.names) {
                let installer = Installer(library: b)
                let e = try ForgeLoRA.examine(file: file, reference: { f in
                    installer.presentComponents(f) ? try installer.readReference(f) : nil
                })
                return (.lora, e.family)
            }
            guard let f = Recipes.family(fromNames: s.names) else {
                throw Numerics.Failure(description: "neither a LoRA nor a recognized DiT ("
                                       + Family.allCases.map { ModelCard.of($0).name }.joined(separator: ", ") + ")")
            }
            return (.model, f)
        }
    }

    /// The door of an import: what the forge or the `.safetensors` reader refuses in the user's file
    /// is `importRefused` (the forge's diagnostic as `detail`); everything else folds as anywhere.
    private static func refusingAs<T>(_ file: String, _ body: () throws -> T) throws(EngineError) -> T {
        try EngineError.boundary {
            do { return try body() } catch let e as Numerics.Failure {
                throw EngineError.importRefused(file: (file as NSString).lastPathComponent, detail: e.description)
            } catch let e as Safetensors.Failure {
                throw EngineError.importRefused(file: (file as NSString).lastPathComponent, detail: e.description)
            }
        }
    }

    /// `My Model v2.safetensors` → `my-model-v2`.
    package static func slug(_ name: String) -> String {
        let simple = name.folding(options: [.diacriticInsensitive, .caseInsensitive], locale: nil).lowercased()
        let pieces = simple.split { !($0.isLetter || $0.isNumber) || !$0.isASCII }
        return pieces.joined(separator: "-").prefix(80).description
    }

    /// **A path that does not exist yet**: `<slug><suffix>`, else `<slug>-2<suffix>`, `-3`… in
    /// `folder` — an import never replaces in silence a model or a LoRA imported before it.
    static func freePath(_ slug: String, suffix: String, in folder: String) -> String {
        let fm = FileManager.default
        var candidate = folder + "/" + slug + suffix, n = 2
        while fm.fileExists(atPath: candidate) || fm.fileExists(atPath: candidate + ".partiel") {
            candidate = folder + "/" + slug + "-\(n)" + suffix
            n += 1
        }
        return candidate
    }

    /// **Imports** a `.safetensors` or a `.gguf`. `name`: a model's displayed name (by default, the file).
    /// `journal` receives the progress. `cancellation`: kept for the callers; nothing downloads.
    package static func importFile(_ file: String, name: String? = nil, in b: Library,
                                cancellation: Cancellation? = nil,
                                journal: @escaping @Sendable (String) -> Void = { _ in }) throws(EngineError) -> Outcome {
        try refusingAs(file) {
            try importing(file, name: name, in: b, cancellation: cancellation, journal: journal)
        }
    }

    private static func importing(_ file: String, name: String?, in b: Library, cancellation: Cancellation?,
                                  journal: @escaping @Sendable (String) -> Void) throws -> Outcome {
        guard FileManager.default.fileExists(atPath: file) else { throw EngineError.fileMissing(path: file) }
        // The size is the cheap test; a file that is not an expected encoder goes on as before.
        if let a = try Adoption.adopt(file, in: b) { return adopted(a, file: file, journal: journal) }
        try BaseModels.refuse(file)
        var rows: [String] = []
        let note: @Sendable (String) -> Void = { journal($0) }
        var installer = Installer(library: b)
        installer.journal = note
        installer.cancellation = cancellation
        let fileName = (file as NSString).lastPathComponent
        let stack = Stack()
        let logLine: (String) -> Void = { l in stack.add(l); journal(l) }

        let source = try TensorSource(paths: [file])
        // **An 8-bit checkpoint is never stored wider than it came**: its bytes
        // and scales go into the map as they are (`QuantizedLayouts`); what cannot be read that
        // way — below 4 bits, a block-scaled fp8 — has already been refused by `TensorSource`, file and all.
        let base = ((file as NSString).lastPathComponent as NSString).deletingPathExtension
        try FileManager.default.createDirectory(atPath: b.map(""), withIntermediateDirectories: true)

        // Never `installer.install`: what the import needs is what the family's installation left.
        func prepare(_ f: Family) throws {
            guard installer.isReady(f) else { throw EngineError.importNeedsFamily(file: fileName, model: ModelCard.of(f).id) }
        }

        if ForgeLoRA.isLoRA(source.names) {
            // A name wholly outside ASCII slugs to nothing: `store/.lora.silicon`, a hidden file.
            let loraSlug = slug(base).isEmpty ? "lora" : slug(base)
            let path = freePath(loraSlug, suffix: ".lora.silicon", in: b.map(""))
            logLine("LoRA: \((file as NSString).lastPathComponent)")
            // A family not installed is not a candidate (`nil`); a real failure reading one that is propagates.
            let r = try ForgeLoRA.forge(file: file, to: path, reference: { f in
                installer.presentComponents(f) ? try installer.readReference(f) : nil
            })
            r.rows.forEach(logLine)
            rows = stack.rows
            return Outcome(kind: .lora, family: r.family, name: LoRACard.readableName(r.name), path: path, journal: rows)
        }

        guard let family = Recipes.family(fromNames: source.names) else {
            throw EngineError.importRefused(
                file: (file as NSString).lastPathComponent,
                detail: "neither a LoRA nor a recognized DiT (\(Family.allCases.map { ModelCard.of($0).name }.joined(separator: ", "))); "
                    + "an encoder, a VAE or a model of another architecture cannot be imported")
        }
        logLine("model \(ModelCard.of(family).name) : \((file as NSString).lastPathComponent)")
        try prepare(family)
        let slug = slug(name ?? base)
        guard !slug.isEmpty else {
            throw EngineError.importRefused(file: (file as NSString).lastPathComponent, detail: "empty name")
        }
        try FileManager.default.createDirectory(at: b.imported, withIntermediateDirectories: true)
        let path = freePath(slug, suffix: ".silicon", in: b.imported.path)
        let config = try installer.ditConfig(family)
        let r = try ForgeDiT.forge(source: source, family: family, config: config, to: path,
                                    reference: try installer.readReference(family), name: name ?? base,
                                    descriptionSource: .object([.init("file", .string((file as NSString).lastPathComponent))]),
                                    // The disk is checked against the map itself, once laid out:
                                    // 8-bit ≈ the file, bf16 from fp32 = half, fp32 kept = as much.
                                    reserve: { bytes in try b.checkSpace(bytes) },
                                    progressHandler: { k, n, _ in if k > 0 && k % max(1, n / 5) == 0 { journal("  … \(k * 100 / n) %") } })
        r.rows.forEach(logLine)
        rows = stack.rows
        return Outcome(kind: .model, family: family, name: name ?? base, path: path, journal: rows)
    }

    /// What `Adoption.adopt` did, told as an import's outcome.
    private static func adopted(_ a: Adoption.Result, file: String, journal: @Sendable (String) -> Void) -> Outcome {
        var families: [Family] = []
        for x in a.expected where !families.contains(x.family) { families.append(x.family) }
        let name = (file as NSString).lastPathComponent
        let kind: Kind = a.expected[0].file.destination == nil ? .encoder : .vae
        var rows = ["\(kind.rawValue): \(name), identical (size, sha256) to the one the installation of "
                    + families.map { ModelCard.of($0).name }.joined(separator: ", ") + " downloads"]
        if a.placed.isEmpty {
            rows.append("every version that expects it has it already: nothing copied")
        } else {
            rows.append(a.copied == 0 ? "already in place for the installation" : "copied: the installation will not download it")
        }
        rows.forEach(journal)
        var o = Outcome(kind: kind, family: families[0], name: name, path: a.placed.first ?? "", journal: rows)
        o.families = families
        o.placed = !a.placed.isEmpty
        return o
    }

    private final class Stack: @unchecked Sendable {
        private(set) var rows: [String] = []
        func add(_ l: String) { rows.append(l) }
    }

    /// Removes an imported model (its map, and an Anima's adapter).
    package static func remove(_ card: ModelCard) throws(EngineError) {
        guard let map = card.map else { throw .notAnImportedModel(model: card.name) }
        try EngineError.boundary { try FileManager.default.removeItem(atPath: map) }
        try? FileManager.default.removeItem(atPath: adapterPath(fromMap: map))
    }
}

extension Library {
    /// **Installs a family** from the vendor (downloads, forges, throws away the sources).
    /// `variant`: Standard, Compact or Light (`Family.variants`); `nil`, the version already there, Standard
    /// if none. Installing the other version **replaces** the one there once the new one is complete
    /// (before, only if the disk cannot hold both: `installPlan(_:variant:baseDiT:)` says so).
    /// `baseDiT`: also the vendor's DiT — useless if one only renders with an imported model; without
    /// it the version installed is kept, and another one asked is refused.
    package func install(_ family: Family, variant: Variant? = nil, baseDiT: Bool = true, cancellation: Cancellation? = nil,
                          journal: @escaping @Sendable (String) -> Void = { _ in }) throws(EngineError) {
        var i = Installer(library: self)
        i.journal = journal
        i.cancellation = cancellation
        try EngineError.boundary { try i.install(family, variant: variant, baseDiT: baseDiT) }
    }
}
