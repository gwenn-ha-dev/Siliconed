import Foundation
import os
import CoreAI
import SiliCore

// **A LoRA, read in its published file and added beside the products it names** — never merged into the weights.
//
//     a LoRA's file ─ LoRAFile ─ its modules (down [rank, in], up [out, rank], alpha) by the DiT's module name
//     the stack of a request ─ LoRAStack ─ each product's factors, every module landing on it side by side
//     a family's function ─ Aicode.Block.lowRank ─ y + (x · down) · up, beside the product
//
// A module is named as the DiT's published file names its weight, without `.weight`: that is what the trainers write
// (`diffusion_model.<module>.lora_A.weight`), so a LoRA's file and a family's DiT meet on one name with no table
// between them. **What a family adapts is what its DiT adapts** (`FamilyTraits.loraModules`): a module believed adapted
// that is not yields a plausible, wrong image, so a file naming one is refused.
//
// **Beside, not merged** (`SiliCore.LowRank`, measured 18/11/2026): merged into the Compact's int8 codes, most of a
// published LoRA's change lies under a quantization step and is rounded away whole; merged into bf16, half of the
// products measured lose 63 to 95 % of it. One path for both versions, as the former engine had it.
//
// **How a family branches it**, in its DiT:
//   - `loraModules()` names every module with its shape, and where it lands when it is not a weight of the file
//     itself (`Shape.product`, `Shape.row`: a third of a fused `qkv`, a module under another trainer's name);
//   - for each product a LoRA may adapt (`LoRAProduct`: the function's argument, the file's key), its function gains two
//     arguments, `<argument>_down` `[in, R]` and `<argument>_up` `[R, out]`, and computes `b.lowRank(y, x, down:, up:)`
//     after the product, x as the product reads it before any rotation of the Compact's codes (the factors are in the
//     model's own basis); R the same for every layer the function is called on (`rank(of:on:)`), 0 when the stack
//     adapts none of them: the function as it is, no argument added. A function with a path is another asset: its name
//     says the ranks;
//   - its stage declares their tensors (`LoRAStack.tensors`, a prefix a layer), held all along: a rank of 32 on a
//     6-billion transformer is ~150 MB in f16, what P counts as any tensor; and fills them (`write`, through
//     `Stage.pointer`) before the first call.

// The formats read, the ones the trainers of these families write:
//
//   - `<prefix><module>.lora_A.weight` / `.lora_B.weight` (PEFT; DiffSynth's `.lora_A.default.weight`, the adapter's
//     name inside), and `.lora_down.weight` / `.lora_up.weight` with `.alpha` (kohya's names), under
//     `diffusion_model.`, `transformer.` or nothing;
//   - kohya's flat names, `lora_unet_<module with every dot an underscore>`: resolved against the family's modules.
//
// **The scale is `alpha / rank`**: the module's `.alpha` tensor, else PEFT's `lora_alpha` in the file's metadata
// (`lora_adapter_metadata`, as diffusers' loader reads it), else `alpha = rank` — a scale of 1, what diffusers'
// `get_peft_kwargs` takes for a file that says nothing.
// Refused by name: LoKr, LoHa, DoRA, a text encoder's LoRA, a convolution's, PEFT's per-module ranks or alphas and rsLoRA.

/// One adapted module of one file.
package struct LoRAModule: Sendable, Equatable {
    /// The factors' keys in the file.
    package let down, up: String
    package let rank, inn, out: Int
    /// `alpha / rank`, 1 when the file has no alpha for the module.
    package let scale: Double
}

/// **A LoRA's file, its modules by name.**
package final class LoRAFile {
    package let path: String
    package let map: MapWeights
    /// By the module's name as the file writes it, its prefix removed: dotted (`layers.0.attention.to_q`), or kohya's
    /// flat one (`blocks_0_self_attn_q_proj`) when `flat`.
    package let modules: [String: LoRAModule]
    package let flat: Bool
    /// **What the file says of itself** (`__metadata__` of its safetensors header), text by text: read once with the file.
    package let metadata: [String: String]

    static let prefixes = ["diffusion_model.", "transformer.", "model.diffusion_model."]
    static let flatPrefixes = ["lora_unet_", "lora_transformer_"]
    static let downSuffixes = [".lora_A.weight", ".lora_A.default.weight", ".lora_down.weight"]
    static let upSuffixes = [".lora_B.weight", ".lora_B.default.weight", ".lora_up.weight"]

    /// **The words that call the LoRA, when its file states them** — nothing is guessed from its name:
    ///
    ///   - `modelspec.trigger_phrase` (the Stability model spec, what Civitai's trainer and kohya write when told one)
    ///     and `trigger_word` (Draw Things), commas between the words;
    ///   - kohya's `ss_tag_frequency`: the tags **every** image of a dataset was captioned with (its count in
    ///     `ss_dataset_dirs`) — the word the trainer put in front of each caption. Three a dataset at most, the
    ///     shortest first: a tag shared by chance (`1girl`) is not what the list is for.
    ///
    /// ai-toolkit's files state none (`training_info`, `software`): their words are on their publisher's page.
    package static func triggers(_ said: [String: String]) -> [String] {
        var words: [String] = []
        for key in ["modelspec.trigger_phrase", "trigger_word"] {
            words += (said[key] ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        }
        func object(_ key: String) -> [String: Any] {
            (said[key].flatMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) } as? [String: Any]) ?? [:]
        }
        let datasets = object("ss_dataset_dirs")
        for (dataset, tags) in object("ss_tag_frequency").sorted(by: { $0.key < $1.key }) {
            guard let images = ((datasets[dataset] as? [String: Any])?["img_count"] as? NSNumber)?.intValue, images > 1,
                  let tags = tags as? [String: Any] else { continue }
            let everywhere = tags.filter { ($0.value as? NSNumber)?.intValue == images }.map { $0.key.trimmingCharacters(in: .whitespaces) }
            words += everywhere.sorted { ($0.count, $0) < ($1.count, $1) }.prefix(3)
        }
        var seen = Set<String>()
        return words.filter { !$0.isEmpty && seen.insert($0.lowercased()).inserted }
    }

    /// **PEFT's `lora_alpha`, when the metadata carries the adapter's configuration** (`lora_adapter_metadata`, a JSON
    /// text whose keys diffusers prefixes with the component's name). `nil` without it; throws on what this reader
    /// does not apply (a rank or an alpha a module, rsLoRA, DoRA).
    static func adapterAlpha(_ metadata: [String: String], file: String) throws -> Double? {
        guard let text = metadata["lora_adapter_metadata"],
              let said = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any] else { return nil }
        let config = Dictionary(said.map { (String($0.key.split(separator: ".").last ?? ""), $0.value) }) { a, _ in a }
        let patterned = ["rank_pattern", "alpha_pattern"].contains { !((config[$0] as? [String: Any])?.isEmpty ?? true) }
        if patterned || config["use_rslora"] as? Bool == true || config["use_dora"] as? Bool == true {
            throw EngineError.unsupportedLoRA(file: file, reason: .unsupportedKind)
        }
        return (config["lora_alpha"] as? NSNumber)?.doubleValue
    }

    package init(_ path: String) throws {
        self.path = path
        map = try MapWeights(path)
        metadata = (Safetensors.header(path)?["__metadata__"] as? [String: Any])?.compactMapValues { $0 as? String } ?? [:]
        let file = (path as NSString).lastPathComponent
        let keys = map.keys
        // What is not a plain LoRA, said before anything else is read of it.
        for marker in ["lokr_", "hada_", "dora_scale", "lora_te", "text_encoder"] where keys.contains(where: { $0.contains(marker) }) {
            throw EngineError.unsupportedLoRA(file: file, reason: .unsupportedKind)
        }
        var downs: [String: String] = [:], ups: [String: String] = [:], alphas: [String: String] = [:]
        for key in keys {
            if let s = Self.downSuffixes.first(where: key.hasSuffix) { downs[String(key.dropLast(s.count))] = key }
            else if let s = Self.upSuffixes.first(where: key.hasSuffix) { ups[String(key.dropLast(s.count))] = key }
            else if key.hasSuffix(".alpha") { alphas[String(key.dropLast(".alpha".count))] = key }
        }
        guard !downs.isEmpty else { throw EngineError.unsupportedLoRA(file: file, reason: .notALoRA) }
        guard Set(downs.keys) == Set(ups.keys) else { throw EngineError.unsupportedLoRA(file: file, reason: .inconsistent) }
        let adapterAlpha = try Self.adapterAlpha(metadata, file: file)
        let isFlat = downs.keys.allSatisfy { name in Self.flatPrefixes.contains(where: name.hasPrefix) }
        var modules: [String: LoRAModule] = [:]
        for (stem, down) in downs {
            guard let up = ups[stem], let d = map.shape(down), let u = map.shape(up), d.count == 2, u.count == 2, d[0] == u[1], d[0] > 0 else {
                throw EngineError.unsupportedLoRA(file: file, reason: .inconsistent)
            }
            var scale = adapterAlpha.map { $0 / Double(d[0]) } ?? 1
            if let key = alphas[stem] {
                guard let alpha = map.value(key) else { throw EngineError.unsupportedLoRA(file: file, reason: .inconsistent) }
                scale = alpha / Double(d[0])
            }
            let prefix = (isFlat ? Self.flatPrefixes : Self.prefixes).first(where: stem.hasPrefix) ?? ""
            modules[String(stem.dropFirst(prefix.count))] = LoRAModule(down: down, up: up, rank: d[0], inn: d[1], out: u[0], scale: scale)
        }
        self.modules = modules
        flat = isFlat
    }

    /// The file's module for the DiT's `name` (dotted), whichever way the file writes it.
    package func module(_ name: String) -> LoRAModule? {
        modules[flat ? name.replacingOccurrences(of: ".", with: "_") : name]
    }

    /// **The modules of the file a family does not reach**, given every module its DiT adapts with its shape: unknown
    /// names, and known ones of another shape. Empty: the family's DiT applies the whole file.
    package func unapplied(by family: [String: LoRAStack.Shape]) -> [String] {
        var reached = Set<String>()
        for (name, shape) in family {
            let key = flat ? name.replacingOccurrences(of: ".", with: "_") : name
            if let m = modules[key], m.out == shape.out, m.inn == shape.inn { reached.insert(key) }
        }
        return modules.keys.filter { !reached.contains($0) }.sorted()
    }

    /// **The family whose DiT applies the whole file**: recognized by its tensors (names and shapes), not by what
    /// its metadata claims. `nil` when none does, or — never seen — when several do.
    package func family() -> Family? {
        let fitting = Family.allCases.filter { unapplied(by: LoRAStack.modules(of: $0)).isEmpty }
        return fitting.count == 1 ? fitting[0] : nil
    }

    package var ranks: (largest: Int, smallest: Int) {
        let r = modules.values.map(\.rank)
        return (r.max() ?? 0, r.min() ?? 0)
    }
}

/// **A request's LoRAs as a DiT reads them**: each file with its strength, in the stack's order.
package struct LoRAStack {
    /// A module's product `[out, in]`, and where its change lands: the rows `row ..< row + out` of the file's weight
    /// `product` (its key without `.weight`), the module's own name when `nil`.
    package struct Shape: Sendable, Equatable {
        package let out, inn: Int
        package let product: String?
        package let row: Int
        package init(out: Int, inn: Int, product: String? = nil, row: Int = 0) {
            (self.out, self.inn, self.product, self.row) = (out, inn, product, row)
        }
    }

    package let entries: [(file: LoRAFile, strength: Double)]

    package static var none: LoRAStack { LoRAStack(entries: []) }

    package init(entries: [(file: LoRAFile, strength: Double)]) {
        self.entries = entries
    }

    /// The stack of a request, its files opened. A strength of 0 adapts nothing and is left out.
    package init(_ loras: [LoRAEntry]) throws {
        self.init(entries: try loras.filter { $0.strength != 0 }.map { (try LoRAFile($0.path), $0.strength) })
    }

    /// **The stack of a request on `family`'s DiT, each file recognized before any computation**: a file whose every
    /// module the family's DiT adapts. Another family's file is refused by that family's name, an unknown one by its own.
    package init(_ loras: [LoRAEntry], on family: Family) throws {
        try self.init(loras)
        let modules = Self.modules(of: family)
        for entry in entries where !entry.file.unapplied(by: modules).isEmpty {
            let name = (entry.file.path as NSString).lastPathComponent
            if let other = entry.file.family() {
                throw EngineError.loraForOtherModel(lora: name, target: other.rawValue, model: family.rawValue)
            }
            throw EngineError.unsupportedLoRA(file: name, reason: .noTarget)
        }
    }

    /// A family's adaptable modules, asked once a process.
    package static func modules(of family: Family) -> [String: Shape] {
        cache.withLock { known in
            if let m = known[family] { return m }
            let m = family.traits.loraModules()
            known[family] = m
            return m
        }
    }
    private static let cache = OSAllocatedUnfairLock<[Family: [String: Shape]]>(initialState: [:])

    package var isEmpty: Bool { entries.isEmpty }
}

// MARK: - The low-rank path in a family's functions

/// **A product a LoRA may adapt, as a family's function reads it**: the function's argument (`name`: its factors are
/// `<name>_down` and `<name>_up`), the weight of the DiT's file it reads (`key`, without `.weight`), its shape.
package struct LoRAProduct: Sendable, Equatable {
    package let name, key: String
    package let inn, out: Int
    package init(name: String, key: String, inn: Int, out: Int) { (self.name, self.key, self.inn, self.out) = (name, key, inn, out) }
}

extension LoRAStack {
    /// A module of a file of the stack landing on a product: its factors, its first row there, and its scale with the
    /// file's strength (`strength · alpha / rank`).
    struct Landing {
        let file: LoRAFile
        let module: LoRAModule
        let row: Int
        let scale: Double
    }

    /// **What lands on the product `key` of `family`'s DiT**, in the stack's order: every module whose shape is the one
    /// the family adapts (a file was recognized whole before, `init(_:on:)`).
    func landings(on key: String, modules: [String: Shape]) -> [Landing] {
        let targets = modules.filter { ($0.value.product ?? $0.key) == key }.sorted { $0.key < $1.key }
        return entries.flatMap { entry in
            targets.compactMap { name, shape -> Landing? in
                guard let m = entry.file.module(name), m.out == shape.out, m.inn == shape.inn else { return nil }
                return Landing(file: entry.file, module: m, row: shape.row, scale: entry.strength * m.scale)
            }
        }
    }

    /// **The rank a function is written at for the products `keys`** — the same argument in every layer that calls it:
    /// the largest sum of the ranks landing on one of them, padded (`LowRank.padded`); 0 when none is adapted.
    package func rank(of keys: [String], on family: Family) -> Int { rank(of: keys, modules: Self.modules(of: family)) }

    func rank(of keys: [String], modules: [String: Shape]) -> Int {
        LowRank.padded(keys.map { landings(on: $0, modules: modules).reduce(0) { $0 + $1.module.rank } }.max() ?? 0)
    }

    /// **The ranks a DiT's functions are written at**, by argument: each name of `products` (a block's, every block's;
    /// an edge's, its own) at the rank over every product of that name (`rank(of:on:)`); none where nothing lands.
    package func ranks(_ products: [LoRAProduct], on family: Family) -> [String: Int] {
        guard !isEmpty else { return [:] }
        let modules = Self.modules(of: family)
        return Dictionary(grouping: products, by: \.name).compactMapValues { same in
            let r = rank(of: same.map(\.key), modules: modules)
            return r > 0 ? r : nil
        }
    }

    /// **The tensors of the factors** of `products` at their function's ranks (`ranks`, by the argument's name; a rank
    /// of 0 has none), for the stage's declaration (`Stage.Declaration.tensors`): `<prefix><name>_down` `[in, R]` and
    /// `<prefix><name>_up` `[R, out]` in `element`, `prefix` telling a layer's from another's.
    package static func tensors(_ products: [LoRAProduct], ranks: [String: Int], element: NDArray.ScalarType,
                                prefix: String = "") -> [SiliCore.Stage.Tensor] {
        products.flatMap { p -> [SiliCore.Stage.Tensor] in
            let r = ranks[p.name] ?? 0
            guard r > 0 else { return [] }
            let b = LowRank.bytes(inn: p.inn, out: p.out, rank: r, element: element)
            return [.init(prefix + p.name + "_down", bytes: b.down), .init(prefix + p.name + "_up", bytes: b.up)]
        }
    }

    /// **The factors of `products` written into their tensors** (`tensors`, the same ranks, element and prefix; their
    /// bytes by `pointer`, the stage's): each module landing on a product side by side along the rank, its up rows at its
    /// own rows of the product, times its scale — a product nothing lands on, zeros. Before the first call.
    package func write(_ products: [LoRAProduct], on family: Family, ranks: [String: Int], element: NDArray.ScalarType,
                       prefix: String = "", into pointer: (String) -> UnsafeMutableRawPointer) throws {
        try write(products, modules: Self.modules(of: family), ranks: ranks, element: element, prefix: prefix, into: pointer)
    }

    func write(_ products: [LoRAProduct], modules: [String: Shape], ranks: [String: Int], element: NDArray.ScalarType,
               prefix: String = "", into pointer: (String) -> UnsafeMutableRawPointer) throws {
        for p in products {
            let places = ranks[p.name] ?? 0
            guard places > 0 else { continue }
            let landings = landings(on: p.key, modules: modules)
            let rank = landings.reduce(0) { $0 + $1.module.rank }
            guard rank <= places else { throw ProgramError("\(p.key): the LoRAs add a rank of \(rank) to a place of \(places)") }
            // One element at least: a product nothing lands on still writes its zeros through a pointer.
            var down = [Float](repeating: 0, count: max(1, rank * p.inn)), up = [Float](repeating: 0, count: max(1, p.out * rank))
            var column = 0
            for l in landings {
                let m = l.module
                guard let d = l.file.map.floats(m.down), let u = l.file.map.floats(m.up), l.row + m.out <= p.out, m.inn == p.inn else {
                    throw EngineError.unsupportedLoRA(file: (l.file.path as NSString).lastPathComponent, reason: .inconsistent)
                }
                for j in 0..<m.rank { for k in 0..<p.inn { down[(column + j) * p.inn + k] = d[j * m.inn + k] } }
                let s = Float(l.scale)
                for i in 0..<m.out { for j in 0..<m.rank { up[(l.row + i) * rank + column + j] = s * u[i * m.rank + j] } }
                column += m.rank
            }
            try down.withUnsafeBufferPointer { d in
                try up.withUnsafeBufferPointer { u in
                    try LowRank.write(down: d.baseAddress!, up: u.baseAddress!, rank: rank, inn: p.inn, out: p.out,
                                      into: pointer(prefix + p.name + "_down"), pointer(prefix + p.name + "_up"), rank: places,
                                      element: element)
                }
            }
        }
    }
}

// MARK: - The library's LoRAs

extension Library {
    /// Where the library keeps a family's LoRAs: `store/loras/<family>/`, each the file its publisher wrote.
    package func loraFolder(_ family: Family) -> String { map("loras/\(family.rawValue)") }

    /// **The LoRAs the library holds**, read in their files — `for`: only those targeting this model. A file that no
    /// longer reads is left out.
    package func loras(for model: String? = nil) -> [LoRACard] {
        let cards = Family.allCases.flatMap { family -> [LoRACard] in
            let folder = loraFolder(family)
            let names = (try? FileManager.default.contentsOfDirectory(atPath: folder)) ?? []
            return names.filter { $0.hasSuffix(".safetensors") }.compactMap { name in
                let path = (folder as NSString).appendingPathComponent(name)
                return (try? LoRAFile(path)).map { LoRACard(file: $0, family: family) }
            }
        }
        return LoRACard.filter(cards, for: model)
    }

    /// **Imports a LoRA: recognized, then copied as it is** into its family's folder — nothing is converted, the
    /// render reads the publisher's file. Refused by name when it is not a LoRA, or when no family's DiT applies
    /// every module it adapts (the closest family's missing modules said). A file of the same name is replaced.
    package func importLoRA(_ file: String, cancellation: Cancellation? = nil) throws(EngineError) -> LoRACard {
        try EngineError.boundary {
            let name = (file as NSString).lastPathComponent
            guard FileManager.default.fileExists(atPath: file) else { throw EngineError.fileMissing(path: file) }
            let lora: LoRAFile
            do { lora = try LoRAFile(file) } catch let e as EngineError { throw e } catch {
                throw EngineError.importRefused(file: name, detail: "\(error)")
            }
            guard let family = lora.family() else { throw EngineError.importRefused(file: name, detail: lora.refusal()) }
            let folder = loraFolder(family), target = (folder as NSString).appendingPathComponent(name)
            try FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)
            try ModelImport.copyIn(file, to: target, cancellation: cancellation)
            return LoRACard(file: try LoRAFile(target), family: family)
        }
    }

    /// Removes a LoRA of the library: its file.
    package func removeLoRA(_ card: LoRACard) throws {
        guard Family.allCases.contains(where: { card.path.hasPrefix(loraFolder($0) + "/") }) else {
            throw EngineError.fileMissing(path: card.path)
        }
        try FileManager.default.removeItem(atPath: card.path)
    }
}

extension Engine.Render.LoRAOrigin {
    /// **What a render's LoRA is known by outside the app**: its file's name without the extension, and the sha256 of
    /// the file — the publisher's bytes, copied as they are at the import (`ModelImport.copyIn`), which Civitai keeps
    /// for every file it scanned. Hashed once while the file keeps its size, date and inode (`ModelImport.Hashes`);
    /// `nil` when it does not read.
    package init(_ entry: LoRAEntry) {
        self.init(name: ((entry.path as NSString).lastPathComponent as NSString).deletingPathExtension,
                  sha256: try? ModelImport.Hashes.shared.sha256(entry.path))
    }
}

extension LoRACard {
    /// The card of a recognized file: its ranks, and what its trainer wrote of it.
    package init(file: LoRAFile, family: Family) {
        let said = file.metadata
        let raw = LoRACard.rawName(path: file.path)
        let trainer = said["ss_output_name"].flatMap { $0.isEmpty || $0 == raw ? nil : $0 }
        self.init(path: file.path, rawName: raw, target: family.rawValue, rank: file.ranks.largest, smallestRank: file.ranks.smallest,
                  resolution: said["modelspec.resolution"], trainedOn: said["ss_sd_model_name"], trainerName: trainer,
                  triggers: LoRAFile.triggers(said))
    }
}
