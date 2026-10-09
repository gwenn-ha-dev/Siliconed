import Foundation

/// **Importing what a user brings**: a `.safetensors`, LoRA or model, without VAE or encoder —
/// this is how Civitai publishes. We recognize what it is and for which family, install what is
/// missing from the family (tokenizers, encoder, VAE, from the vendor), and forge. The file
/// brought can then be thrown away: the render only reads `store/`.
///
/// This is Draw Things' idea (`BaseModelImporter`, `LoRAImporter`): the architecture is detected,
/// only the DiT is converted, the rest is the family's.
public enum ModelImport {
    public enum Kind: String, Sendable { case lora = "LoRA", model = "model" }

    public struct Outcome: Sendable {
        public let kind: Kind
        public let family: Family
        /// The displayed name.
        public let name: String
        /// The forged map.
        public let path: String
        /// What happened, line by line (naming recognized, dtypes, roundings, installations).
        public let journal: [String]
        /// The identifier to pass to `Model.named` (a model) — `nil` for a LoRA.
        public var identifier: String? {
            kind == .model ? "\(family.rawValue)/" + ((path as NSString).lastPathComponent as NSString).deletingPathExtension : nil
        }
    }

    /// What the file is, without forging anything.
    public static func recognize(_ file: String) throws(EngineError) -> (Kind, Family?) {
        try refusingAs(file) {
            let s = try TensorSource(paths: [file])
            if ForgeLoRA.isLoRA(s.names) { return (.lora, nil) }
            return (.model, Recipes.family(fromNames: s.names))
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
    public static func slug(_ name: String) -> String {
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

    /// **Imports** a `.safetensors`. `name`: a model's displayed name (by default, the file).
    /// `journal` receives the progress — a family installation downloads several GB;
    /// `cancellation` stops it during these downloads.
    public static func importFile(_ file: String, name: String? = nil, in b: Library,
                                cancellation: Cancellation? = nil,
                                journal: @escaping @Sendable (String) -> Void = { _ in }) throws(EngineError) -> Outcome {
        try refusingAs(file) {
            try importing(file, name: name, in: b, cancellation: cancellation, journal: journal)
        }
    }

    private static func importing(_ file: String, name: String?, in b: Library, cancellation: Cancellation?,
                                  journal: @escaping @Sendable (String) -> Void) throws -> Outcome {
        guard FileManager.default.fileExists(atPath: file) else { throw EngineError.fileMissing(path: file) }
        var rows: [String] = []
        let note: @Sendable (String) -> Void = { journal($0) }
        var installer = Installer(library: b)
        installer.journal = note
        installer.cancellation = cancellation
        let stack = Stack()
        let logLine: (String) -> Void = { l in stack.add(l); journal(l) }

        let source = try TensorSource(paths: [file])
        // **An 8-bit checkpoint is never stored wider than it came**: its bytes
        // and scales go into the map as they are (`QuantizedLayouts`); what cannot be read that
        // way — below 4 bits, a block-scaled fp8 — has already been refused by `TensorSource`, file and all.
        let base = ((file as NSString).lastPathComponent as NSString).deletingPathExtension
        try FileManager.default.createDirectory(atPath: b.map(""), withIntermediateDirectories: true)

        func prepare(_ f: Family) throws {
            if !installer.isReady(f) {
                logLine("\(f.name) : components absent — installing (tokenizers, encoder, VAE)")
                try installer.install(f, baseDiT: false)
            }
        }

        if ForgeLoRA.isLoRA(source.names) {
            // A name wholly outside ASCII slugs to nothing: `store/.lora.silicon`, a hidden file.
            let loraSlug = slug(base).isEmpty ? "lora" : slug(base)
            let path = freePath(loraSlug, suffix: ".lora.silicon", in: b.map(""))
            logLine("LoRA: \((file as NSString).lastPathComponent)")
            let r = try ForgeLoRA.forge(file: file, to: path, reference: { f in
                if !installer.presentComponents(f) { try prepare(f) }
                return try installer.readReference(f)
            })
            r.rows.forEach(logLine)
            rows = stack.rows
            return Outcome(kind: .lora, family: r.family, name: LoRACard.readableName(r.name), path: path, journal: rows)
        }

        guard let family = Recipes.family(fromNames: source.names) else {
            throw EngineError.importRefused(
                file: (file as NSString).lastPathComponent,
                detail: "neither a LoRA nor a recognized DiT (\(Family.allCases.map(\.name).joined(separator: ", "))); "
                    + "an encoder, a VAE or a model of another architecture cannot be imported")
        }
        logLine("model \(family.name) : \((file as NSString).lastPathComponent)")
        try prepare(family)
        let slug = slug(name ?? base)
        guard !slug.isEmpty else {
            throw EngineError.importRefused(file: (file as NSString).lastPathComponent, detail: "empty name")
        }
        try FileManager.default.createDirectory(at: b.imported, withIntermediateDirectories: true)
        let path = freePath(slug, suffix: ".silicon", in: b.imported.path)
        let config = try OrderedJSON.read(installer.componentsFolder(family) + "/transformer.json")
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

    private final class Stack: @unchecked Sendable {
        private(set) var rows: [String] = []
        func add(_ l: String) { rows.append(l) }
    }

    /// Removes an imported model (its map, and an Anima's adapter).
    public static func remove(_ card: ModelCard) throws(EngineError) {
        guard let map = card.map else { throw .notAnImportedModel(model: card.name) }
        try EngineError.boundary { try FileManager.default.removeItem(atPath: map) }
        try? FileManager.default.removeItem(atPath: adapterPath(fromMap: map))
    }
}

extension Library {
    /// The families whose encoder and components are installed — enough to render with an imported DiT.
    public func readyFamilies() -> [Family] {
        let i = Installer(library: self)
        return Family.allCases.filter { i.isReady($0) }
    }

    /// **Installs a family** from the vendor (downloads, forges, throws away the sources).
    /// `variant`: Standard or Compact (`Family.variants`); `nil`, the version already there, Standard
    /// if none. Installing the other version **replaces** the one there once the new one is complete
    /// (before, only if the disk cannot hold both: `installPlan(_:variant:baseDiT:)` says so).
    /// `baseDiT`: also the vendor's DiT — useless if one only renders with an imported model; without
    /// it the version installed is kept, and another one asked is refused.
    public func install(_ family: Family, variant: Variant? = nil, baseDiT: Bool = true, cancellation: Cancellation? = nil,
                          journal: @escaping @Sendable (String) -> Void = { _ in }) throws(EngineError) {
        var i = Installer(library: self)
        i.journal = journal
        i.cancellation = cancellation
        try EngineError.boundary { try i.install(family, variant: variant, baseDiT: baseDiT) }
    }
}
