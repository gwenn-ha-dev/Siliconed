import CryptoKit
import Foundation

/// **The forge of a LoRA**: a `.safetensors` as published (Civitai, Hugging Face) → a mappable
/// map, checked module by module against its family's DiT. Replaces the earlier Python forge.
///
/// ## What it reads
///
/// Trainers agree on nothing, and every convention is brought down to the same pair
/// `down = A [r, k]`, `up = B [n, r]` (`ΔW = B·A`, PyTorch layout `[output, input]`):
///
///   · **prefixes** `diffusion_model.` (ai-toolkit, ComfyUI), `transformer.` (diffusers),
///     `base_model.model.` (PEFT), `lora_unet_` (kohya, musubi-tuner: names flattened with `_`);
///   · **suffixes** `lora_A/lora_B` (PEFT, with or without `.default`), `lora_down/lora_up`
///     (kohya), `lora.down/lora.up`;
///   · **original names** (`blocks.3.attn.wq`, `layers.0.attention.qkv`,
///     `blocks.3.self_attn.q_proj`) or diffusers (`transformer_blocks.3.attn.to_q`) — the rules of
///     `Recipes`, those of the DiTs;
///   · **fused qkv** (original Z-Image): shared `down`, `up` split in three by rows;
///   · **Qwen-Image-2.1 from ai-toolkit**: its vendored transformer keeps ComfyUI's fused
///     `img_mlp.gate_up` (`gate, up = gate_up(x).chunk(2)`, `out(silu(gate) * up)`), where the
///     map has diffusers' `gate_layer` (the SiLU side) and `proj`. `up` is cut by rows,
///     `[0, h)` → `gate_layer`, `[h, 2h)` → `proj`, `down` shared: exact. The ranges come from the
///     map's shapes, not from a constant: the split `ΔW` is exactly the fused `ΔW`, split.
///
/// ## The scale
///
/// One `.alpha` tensor per module (kohya): `ΔW = (α/r)·B·A`, folded into `up` — what diffusers
/// does (`get_alpha_scales`). Failing that, a PEFT **`lora_adapter_metadata`** (what diffusers
/// writes and reads back as the `LoraConfig`): `lora_alpha`, `alpha_pattern` (PEFT's match,
/// `(.*\.)?(key)$`), `use_rslora` (α/√r). Keys bare (`save_lora_adapter`) or prefixed
/// `transformer.` (a pipeline's `save_lora_weights`); those of a text encoder are not ours.
/// An `ss_network_alpha` only in the **metadata** is ignored, like diffusers
/// (`network_alphas=None`, hence α = r) and ComfyUI (which only reads the tensors). ai-toolkit in
/// PEFT format trains with α forced to r (`lora_special.py`) and saves no alpha: 1 is exact there.
/// The strength remains a setting of the caller.
///
/// ## What it refuses, rather than render a plausible and wrong image
///
/// Full deltas (non-zero `diff`, `diff_b`), DoRA (`dora_scale`), LoCon/LoHa/LoKr, and Anima's text
/// adapter: the engine only applies a LoRA to the DiT's `Linear`. A module that does not exist in
/// the family, or whose shape differs, causes a failure: the LoRA is not for this model.
///
/// ## What it keeps of the file
///
/// The **sha256 of the `.safetensors` as published** (`source.sha256`): the map's bytes are not
/// the file's, so once the file is thrown away this is the only thing that still names it to Civitai,
/// which credits an image's LoRA by a hash it stores for every file (`Infotext`). Hashed in a stream,
/// one fixed block: nothing proportional to the file is allocated.
package enum ForgeLoRA {
    package struct Report {
        package let tally: MapWriter.Tally
        package let family: Family
        package let name: String
        package let modules: Int
        package let ranks: [Int]
        package let notes: [String]
        package var rows: [String] {
            ["target: \(family.name) — \(modules) modules, rank \(ranks.count == 1 ? "\(ranks[0])" : "\(ranks)")"] + notes
                + [String(format: "%@ : %.1f MB", (tally.path as NSString).lastPathComponent, Double(tally.bytes) / 1e6)]
        }
    }

    /// **The sha256 of a whole file, in a stream**: one `block`-byte buffer, read with `pread`
    /// outside the page cache (`F_NOCACHE`) — a LoRA of 1 GB costs the same 8 MB as one of 10 MB.
    package static func sha256(file path: String, block: Int = 8 << 20) throws -> String {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw EngineError.fileUnreadable(path: path) }
        defer { close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: block, alignment: 16384)
        defer { buffer.deallocate() }
        var h = SHA256(), at: off_t = 0
        while true {
            let n = pread(fd, buffer, block, at)
            guard n >= 0 else { throw EngineError.fileUnreadable(path: path) }
            if n == 0 { break }
            h.update(bufferPointer: UnsafeRawBufferPointer(start: buffer, count: n))
            at += off_t(n)
        }
        return h.finalize().map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - What the trainer says of it

    /// **The name the trainer wrote**: `modelspec.title` (the SAI model spec's title, kohya), else
    /// `name` (ai-toolkit). `nil` when empty or a checkpoint counter — Civitai's trainer writes
    /// `epoch_1` into every epoch's file. Shown, never used to name the LoRA.
    package static func trainerName(_ metadata: [String: String]) -> String? {
        for key in ["modelspec.title", "name"] {
            if let v = metadata[key]?.trimmingCharacters(in: .whitespacesAndNewlines), !v.isEmpty,
               v.firstMatch(#"^epoch_\d+$"#) == nil { return v }
        }
        return nil
    }

    /// The most tags a dataset folder may hold for them to be read as its trigger.
    static let triggerTagLimit = 3

    /// **The trigger words — only what cannot be anything else.** Wrong ones would send the user
    /// typing words the LoRA never learned; none sends them to the LoRA's page.
    ///
    /// 1. An explicit field, first found: `modelspec.trigger_phrase` (SAI model spec),
    ///    `trigger_words`, `trigger_word`, `activation text` (some trainers and converters).
    ///    A JSON list of strings, else one phrase as written.
    /// 2. Failing that, `ss_tag_frequency`: kohya's captions counted per dataset folder,
    ///    `{folder: {tag: count}}`. A folder whose captions hold at most `triggerTagLimit` distinct
    ///    tags trained on those words alone: they are its trigger. ai-toolkit writes exactly that,
    ///    `{"1_<trigger>": {"<trigger>": 1}}`, and kohya a caption-less DreamBooth folder
    ///    `{"10_sks woman": {"sks woman": n}}`. A folder with more tags is a captioned dataset:
    ///    a trigger, if any, is one tag among the descriptive ones and nothing in the counts singles
    ///    it out (the most frequent is as often `1girl`), so it gives nothing. The folder's name is
    ///    not read: `<repeats>_<trigger>` is one convention, a dataset config names folders freely
    ///    (`Cleaned`). An empty tag is no trigger — ai-toolkit without one writes `{"1_": {"": 1}}`.
    ///
    /// Order: explicit fields as written; tags by folder name, then by falling count. No duplicates.
    package static func triggers(_ metadata: [String: String]) -> [String] {
        func clean(_ words: [String]) -> [String] {
            var seen = Set<String>()
            return words.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty && seen.insert($0).inserted }
        }
        for key in ["modelspec.trigger_phrase", "trigger_words", "trigger_word", "activation text"] {
            guard let v = metadata[key] else { continue }
            let list = (try? JSONSerialization.jsonObject(with: Data(v.utf8))) as? [String]
            let words = clean(list ?? [v])
            if !words.isEmpty { return words }
        }
        guard let raw = metadata["ss_tag_frequency"],
              let folders = (try? JSONSerialization.jsonObject(with: Data(raw.utf8))) as? [String: Any] else { return [] }
        var words: [String] = []
        for folder in folders.keys.sorted() {
            guard let tags = folders[folder] as? [String: Any] else { continue }
            let named = tags.filter { !$0.key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            guard named.count <= triggerTagLimit else { continue }
            let count = { (t: String) in (named[t] as? NSNumber)?.intValue ?? 0 }
            words += named.keys.sorted { (count($0), $1) > (count($1), $0) }
        }
        return clean(words)
    }

    static let prefixes = ["base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.", "unet."]

    struct Pair {
        var down: String?
        var up: String?
        var alpha: String?
    }

    /// A LoRA? — one key with the right form is enough. LyCORIS (`lokr_`, `hada_`) is one too, to
    /// be refused as such by `parseKey`: taken for a model, a LoKr on Krea 2's names passed for a
    /// checkpoint (a 1.5 GB one on the Hub).
    package static func isLoRA(_ names: [String]) -> Bool {
        names.contains { $0.contains(".lora_A") || $0.contains(".lora_B") || $0.contains(".lora_down")
            || $0.contains(".lora_up") || $0.contains(".lora.down") || $0.hasPrefix("lora_unet_")
            || $0.contains(".lokr_") || $0.contains(".hada_") }
    }

    /// Module and role of a published key, or the error saying why it is not read.
    static func parseKey(_ key: String) throws -> (module: String, role: String)? {
        var k = key
        for p in prefixes where k.hasPrefix(p) { k = String(k.dropFirst(p.count)); break }
        let roles: [(String, String)] = [
            (".lora_A.default.weight", "down"), (".lora_B.default.weight", "up"),
            (".lora_A.weight", "down"), (".lora_B.weight", "up"),
            (".lora_down.weight", "down"), (".lora_up.weight", "up"),
            (".lora.down.weight", "down"), (".lora.up.weight", "up"),
            (".alpha", "alpha"), (".diff_b", "diff_b"), (".diff", "diff"),
        ]
        for (suffix, role) in roles where k.hasSuffix(suffix) {
            return (String(k.dropLast(suffix.count)), role)
        }
        if k.contains("dora_scale") { throw Numerics.Failure(description: "\(key) : DoRA not supported") }
        if k.contains("hada_") || k.contains("lokr_") || k.contains("lora_mid") {
            throw Numerics.Failure(description: "\(key) : LyCORIS (LoHa, LoKr, LoCon) not supported")
        }
        throw Numerics.Failure(description: "\(key) : neither lora_A/lora_B nor lora_down/lora_up — unknown format")
    }

    /// PEFT's `LoraConfig`, as diffusers saves it in `lora_adapter_metadata`: the scale it implies.
    struct PEFTScale {
        var alpha: Double
        /// In the JSON's order: PEFT (`get_pattern_key`) takes the first key that matches.
        var alphaPattern: [(key: String, alpha: Double)] = []
        var rsLoRA = false

        /// From a safetensors' metadata; `nil` without `lora_adapter_metadata` or without `lora_alpha`.
        static func read(_ metadata: [String: String]) throws -> PEFTScale? {
            guard let raw = metadata["lora_adapter_metadata"] else { return nil }
            guard let pairs = (try? OrderedJSON.parse(Data(raw.utf8)))?.pairs else {
                throw Numerics.Failure(description: "lora_adapter_metadata : not a JSON object")
            }
            // Bare keys, or the transformer's of a pipeline; a text encoder's are not for the DiT.
            var config: [String: OrderedJSON] = [:]
            for p in pairs where p.key.hasPrefix("transformer.") { config[String(p.key.dropFirst(12))] = p.value }
            for p in pairs where !p.key.contains(".") && config[p.key] == nil { config[p.key] = p.value }
            guard let alpha = config["lora_alpha"]?.double else { return nil }
            // `NaN`, `Infinity` and `1e999` parse (Python's `json` writes them): a scale that is not a number.
            guard alpha.isFinite else {
                throw Numerics.Failure(description: "lora_adapter_metadata : lora_alpha \(alpha) is not finite")
            }
            if config["use_dora"]?.boolean == true {
                throw Numerics.Failure(description: "lora_adapter_metadata : use_dora — DoRA not supported")
            }
            var s = PEFTScale(alpha: alpha, rsLoRA: config["use_rslora"]?.boolean == true)
            s.alphaPattern = (config["alpha_pattern"]?.pairs ?? []).compactMap { p in p.value.double.map { (p.key, $0) } }
            for (key, value) in s.alphaPattern {
                try checkPattern(key)
                guard value.isFinite else {
                    throw Numerics.Failure(description: "lora_adapter_metadata : alpha_pattern \"\(key)\" \(value) is not finite")
                }
            }
            return s
        }

        /// The longest `alpha_pattern` key taken as a regular expression. Real keys name a module
        /// (`transformer_blocks.0.attn.to_q`, `.*to_k`): a few dozen characters.
        static let maximumPattern = 256

        /// A key of the file becomes a regular expression (PEFT's `re.match`), matched against every
        /// module during an import that cannot be stopped midway: a key too long, with a quantified
        /// group — `(a+)+`, the shape of catastrophic backtracking — or with more than one unbounded
        /// quantifier (`.*.*.*…z` backtracks exponentially in `^(.*\.)?(KEY)$`) is refused by name.
        /// diffusers writes literal module names (`get_peft_kwargs`); a hand-written PEFT key is a
        /// name or a suffix with one wildcard (`.*to_k`): both pass.
        static func checkPattern(_ key: String) throws {
            guard key.count <= maximumPattern else {
                throw Numerics.Failure(description: "lora_adapter_metadata : an alpha_pattern key of \(key.count) characters "
                                       + "(at most \(maximumPattern))")
            }
            guard key.firstMatch(#"\)[*+?{]"#) == nil else {
                throw Numerics.Failure(description: "lora_adapter_metadata : alpha_pattern key \"\(key)\" repeats a group — not read")
            }
            guard unboundedQuantifiers(key) <= 1 else {
                throw Numerics.Failure(description: "lora_adapter_metadata : alpha_pattern key \"\(key)\" has more than one "
                                       + "unbounded quantifier (*, +, {n,}) — not read")
            }
        }

        /// The `*`, `+` and `{n,}` of a regular expression, outside escapes and character classes.
        static func unboundedQuantifiers(_ pattern: String) -> Int {
            let c = Array(pattern)
            var n = 0, i = 0, inClass = false
            while i < c.count {
                let x = c[i]
                if x == "\\" { i += 2; continue }
                if inClass { if x == "]" { inClass = false }; i += 1; continue }
                switch x {
                case "[": inClass = true
                case "*", "+": n += 1
                case "{":
                    if let close = c[i...].firstIndex(of: "}"), String(c[(i + 1)..<close]).firstMatch(#"^\d*,$"#) != nil { n += 1 }
                default: break
                }
                i += 1
            }
            return n
        }

        /// `ΔW` multiplier for `module` (the published name, without prefix), of rank `r`. PEFT takes
        /// the first key, in the file's order, that matches `(.*\.)?(key)$`; a key that names the
        /// module literally (itself, or its dotted suffix) matches without a regular expression.
        func scale(_ module: String, r: Int) -> Float {
            let a = alphaPattern.first { p in
                module == p.key || module.hasSuffix("." + p.key)
                    || module.firstMatch(#"^(.*\.)?("# + p.key + #")$"#) != nil
            }?.alpha ?? alpha
            return Float(rsLoRA ? a / Double(r).squareRoot() : a / Double(r))
        }
    }

    /// Consecutive ranges of a fused weight, sized like each part in the map (`axis` 0: rows).
    static func consecutive(_ parts: [String], axis: Int, ref: [String: [Int]], family: Family) throws -> [(String, Range<Int>)] {
        var begin = 0
        return try parts.map { c in
            guard let f = ref[c + ".weight"], f.count == 2 else {
                throw Numerics.Failure(description: "\(c) : does not exist in \(family.name) — this LoRA is not for this model")
            }
            defer { begin += f[axis] }
            return (c, begin..<(begin + f[axis]))
        }
    }

    /// A target of the map, and what falls to it of the published `ΔW`: an equal share of the rows
    /// of `up` (`part`, i of n), or a range of rows (`rows`); none: everything.
    struct Target {
        let module: String
        var part: (Int, Int)? = nil
        var rows: Range<Int>? = nil
    }

    /// A family's original names, flattened: `lora_unet_layers_0_attention_qkv` → the module.
    static func candidates(_ family: Family, reference: [String: [Int]]) -> [String] {
        var c = reference.keys.filter { $0.hasSuffix(".weight") }.map { String($0.dropLast(7)) }
        switch family {
        case .zImage:
            for m in c where m.hasSuffix(".attention.to_q") { c.append(String(m.dropLast(5)) + ".qkv") }
            c += c.compactMap { m in m.hasSuffix(".attention.to_out.0") ? String(m.dropLast(9)) + ".out" : nil }
            c += c.compactMap { m in m.hasPrefix("all_final_layer.2-1.") ? "final_layer." + m.dropFirst(20) : nil }
            c += c.compactMap { m in m.hasPrefix("all_x_embedder.2-1") ? "x_embedder" + m.dropFirst(18) : nil }
        case .krea2:
            for n in 0..<64 {
                for x in ["wq", "wk", "wv", "wo", "gate"] { c.append("blocks.\(n).attn.\(x)") }
                for x in ["gate", "up", "down"] { c.append("blocks.\(n).mlp.\(x)") }
                for b in ["layerwise_blocks", "refiner_blocks"] where n < 8 {
                    for x in ["wq", "wk", "wv", "wo", "gate"] { c.append("txtfusion.\(b).\(n).attn.\(x)") }
                    for x in ["gate", "up", "down"] { c.append("txtfusion.\(b).\(n).mlp.\(x)") }
                }
            }
            c += ["first", "last.linear", "tmlp.0", "tmlp.2", "tproj.1", "txtmlp.1", "txtmlp.3", "txtfusion.projector"]
        case .anima:
            for n in 0..<64 {
                for (pattern, _) in Recipes.animaBlocks { c.append("blocks.\(n)." + pattern.dropLast()) }
            }
        case .qwenImage21, .mossImage:
            break
        }
        return c
    }

    /// The published module → its targets in the map. `ref`: the map's shapes, `[output, input]`.
    static func targets(_ module: String, family: Family, ref: [String: [Int]] = [:]) throws -> [Target] {
        switch family {
        case .qwenImage21:
            // Diffusers names, published as is (PEFT): nothing to split. Unlike Z-Image's,
            // the modulation (`modulation.1`) and the σ embedding (`time_text_embed.*`) are
            // **accepted**: a LoRA trained on this family may adapt both (turbo LoRAs of the base
            // model do), and the engine applies them, in double (`QwenImage21DiT.exactModules`).
            // ai-toolkit (and ComfyUI) fuse the SwiGLU's two inputs: `img_mlp.gate_up` is
            // `[gate_layer; proj]` by rows — `gate, up = gate_up(x).chunk(2)`, `silu(gate) * up`.
            if module.hasSuffix(".img_mlp.gate_up") {
                let root = String(module.dropLast(8))
                return try consecutive([root + ".gate_layer", root + ".proj"], axis: 0, ref: ref, family: family)
                    .map { Target(module: $0.0, rows: $0.1) }
            }
            return [Target(module: module)]
        case .zImage:
            var m = module
            if m.hasPrefix("final_layer.") { m = "all_final_layer.2-1." + m.dropFirst(12) }
            if m.hasPrefix("x_embedder") { m = "all_x_embedder.2-1" + m.dropFirst(10) }
            if m.hasSuffix(".attention.out") { m = String(m.dropLast(4)) + ".to_out.0" }
            if m.hasSuffix(".attention.qkv") {
                let r = String(m.dropLast(4))
                return [0, 1, 2].map { Target(module: r + [".to_q", ".to_k", ".to_v"][$0], part: ($0, 3)) }
            }
            return [Target(module: m)]
        case .krea2:
            var target = module
            if !(module.hasPrefix("transformer_blocks.") || module.hasPrefix("text_fusion.") || module.hasPrefix("txt_in.")
                || module.hasPrefix("img_in") || module.hasPrefix("time_") || module.hasPrefix("final_layer.")) {
                guard let (n, _) = Recipes.krea2(module + ".weight", shape: [1, 1]), n.hasSuffix(".weight") else {
                    throw Numerics.Failure(description: "\(module) : no Krea 2 renaming rule")
                }
                target = String(n.dropLast(7))
            }
            // diffusers adapts every `Linear` it names (`tmlp.*`, `tproj.1` included); the engine all
            // but the fusion's 12 → 1 projector — refused here rather than accepted and left out.
            guard Krea2DiT.appliesLoRA(target) else {
                throw Numerics.Failure(description: "\(module) → \(target) : "
                    + (target == "text_fusion.projector"
                       ? "Krea 2's 12 → 1 layer projector is a weighted sum the engine computes outside a GEMM, "
                         + "where a LoRA does not reach — refused rather than half-applied"
                       : "a Krea 2 module the engine does not adapt with a LoRA — refused rather than ignored"))
            }
            return [Target(module: target)]
        case .anima:
            if module.hasPrefix("llm_adapter.") {
                throw Numerics.Failure(description: "\(module) : this LoRA touches Anima's text adapter, "
                                      + "which the engine does not adapt — refused rather than half-applied")
            }
            if module.hasPrefix("transformer_blocks.") || module.hasPrefix("time_embed.") || module.hasPrefix("patch_embed.")
                || module.hasPrefix("norm_out.") || module == "proj_out" {
                return [Target(module: module)]
            }
            return [Target(module: String(try Recipes.anima(module + ".weight").dropLast(7)))]
        case .mossImage:
            throw Numerics.Failure(description: "\(module) : LoRAs of MOSS-Image are not imported — the family renders under its own 8-step LoRA")
        }
    }

    /// What the forge would make of a LoRA, judged on its names and shapes alone.
    package struct Examination {
        package let family: Family
        package let modules: Int
        package let ranks: [Int]
    }

    /// **Forges the LoRA.** `family`: imposed, or `nil` to recognize it from the names. `reference`:
    /// the published DiT's names and shapes of a family, `nil` when it is not installed — never a
    /// download; what it throws (a disk full, a damaged `dit.json`) is a real failure and propagates.
    package static func forge(file: String, to path: String, family imposed: Family? = nil,
                               reference: (Family) throws -> [String: [Int]]?) throws -> Report {
        try forging(file: file, to: path, family: imposed, reference: reference).1!
    }

    /// **The forge's verdict without a value read**: names, shapes, targets — what refuses a LoRA
    /// for another model, LyCORIS, DoRA, a text encoder. A file whose data is not there yet (a
    /// header from the Hub, the rest of the file sparse) is judged as the import will judge it,
    /// but for the values: a non-zero `diff` and a non-finite `alpha` are the import's to refuse.
    package static func examine(file: String, family imposed: Family? = nil,
                                 reference: (Family) throws -> [String: [Int]]?) throws -> Examination {
        try forging(file: file, to: nil, family: imposed, reference: reference).0
    }

    /// `path` `nil`: examine only, nothing read past the header, nothing written.
    private static func forging(file: String, to path: String?, family imposed: Family?,
                                reference: (Family) throws -> [String: [Int]]?) throws -> (Examination, Report?) {
        let source = try TensorSource(paths: [file])
        let metadata = source.metadata
        var notes: [String] = []

        // ── The pairs, per published module ─────────────────────────────────────────────────
        var pairs: [String: Pair] = [:]
        var moduleOrder: [String] = []
        for key in source.names {
            guard let (module, role) = try parseKey(key) else { continue }
            if pairs[module] == nil { moduleOrder.append(module) }
            var p = pairs[module, default: Pair()]
            switch role {
            case "down": p.down = key
            case "up": p.up = key
            case "alpha": p.alpha = key
            default:
                guard path != nil else { break }
                let v = try source.read(key)
                if v.contains(where: { $0 != 0 }) {
                    throw Numerics.Failure(description: "\(key) : a non-zero `\(role)` delta — the engine only applies "
                                          + "low-rank LoRAs on the weights; refused")
                }
                notes.append("\(key) : zero `\(role)` delta, ignored")
            }
            pairs[module] = p
        }
        let incomplete = pairs.filter { $0.value.down == nil || $0.value.up == nil }.map(\.key)
        let orphans = incomplete.filter { pairs[$0]?.down == nil && pairs[$0]?.up == nil }
        pairs = pairs.filter { $0.value.down != nil || $0.value.up != nil }
        moduleOrder = moduleOrder.filter { pairs[$0] != nil }
        let mismatched = incomplete.filter { !orphans.contains($0) }
        guard mismatched.isEmpty else {
            throw Numerics.Failure(description: "\(mismatched.count) mismatched modules (down without up), including \(mismatched.prefix(3))")
        }
        guard !pairs.isEmpty else { throw Numerics.Failure(description: "no down/up pair: not a LoRA") }
        // sd-scripts' `lora_te_`, `lora_te1_`…, diffusers' `text_encoder.`: the engine adapts the DiT only.
        if let te = moduleOrder.first(where: { $0.hasPrefix("lora_te") || $0.hasPrefix("text_encoder") }) {
            throw Numerics.Failure(description: "\(te) : this LoRA also adapts the text encoder, which the engine "
                                  + "does not adapt — refused rather than half-applied")
        }

        // ── The family, and the flattened kohya names ─────────────────────────────────────────
        let flattened = moduleOrder.contains { $0.hasPrefix("lora_unet_") }
        func familyOfNames(_ modules: [String]) -> Family? {
            func a(_ pattern: String) -> Bool { modules.contains { $0.firstMatch(pattern) != nil } }
            if a(#"^transformer_blocks\.\d+\.img_mlp\.(gate_layer|proj|gate_up|out)$"#) { return .qwenImage21 }
            if a(#"^(layers|noise_refiner|context_refiner)\.\d+\.(attention|feed_forward|adaLN_modulation)"#) { return .zImage }
            if a(#"^(blocks\.\d+\.(attn\.(wq|wk|wv|wo|gate)|mlp\.(gate|up|down))|txtfusion\.|text_fusion\.|transformer_blocks\.\d+\.(attn\.to_gate|ff\.(gate|up|down)))"#) { return .krea2 }
            if a(#"^(blocks\.\d+\.(self_attn|cross_attn|mlp\.layer|adaln_modulation)|llm_adapter\.|transformer_blocks\.\d+\.(attn1|attn2|ff\.net|norm[123]))"#) { return .anima }
            return nil
        }
        var family = imposed
        var renamed: [String: String] = [:]                 // published module → readable module
        if flattened {
            // Only the families installed are compared; a failure reading one is not a mismatch.
            for f in (imposed.map { [$0] } ?? Family.allCases) {
                guard let ref = try reference(f) else { continue }
                var table: [String: String] = [:]
                for c in candidates(f, reference: ref) { table[c.replacingOccurrences(of: ".", with: "_")] = c }
                var r: [String: String] = [:]
                for m in moduleOrder {
                    let flat = String(m.dropFirst("lora_unet_".count))
                    guard let c = table[flat] else { r = [:]; break }
                    r[m] = c
                }
                if r.count == moduleOrder.count { family = f; renamed = r; break }
            }
            guard family != nil, !renamed.isEmpty else {
                if let imposed, try reference(imposed) == nil {
                    throw EngineError.importNeedsFamily(file: (file as NSString).lastPathComponent, model: ModelCard.of(imposed).id)
                }
                throw Numerics.Failure(description: "kohya names (`lora_unet_…`) that no installed family recognizes — "
                                      + "install the target family, or the LoRA is not for a supported model")
            }
            notes.append("flattened kohya names reconstructed")
        } else {
            for m in moduleOrder { renamed[m] = m }
            family = family ?? familyOfNames(moduleOrder)
            // Names shared by several families (`transformer_blocks.N.attn.to_q`: a LoRA on the
            // attention alone, lightx2v's Qwen-Image-2.1 Lightning): the installed families' shapes
            // decide — exactly one must take every module.
            if family == nil {
                func fits(_ f: Family, _ ref: [String: [Int]]) -> Bool {
                    moduleOrder.allSatisfy { m in
                        guard let a = source.shape(pairs[m]!.down!), let b = source.shape(pairs[m]!.up!),
                              a.count == 2, b.count == 2,
                              let parts = try? targets(m, family: f, ref: ref), !parts.isEmpty else { return false }
                        return parts.allSatisfy { c in
                            guard let s = ref[c.module + ".weight"], s.count == 2, s[1] == a[1] else { return false }
                            return c.rows != nil || c.part != nil || s[0] == b[0]
                        }
                    }
                }
                let fitting = try Family.allCases.filter { f in try reference(f).map { fits(f, $0) } ?? false }
                if fitting.count == 1 {
                    family = fitting[0]
                    notes.append("family recognized by the installed shapes: \(fitting[0].name)")
                }
            }
        }
        guard let family else {
            throw Numerics.Failure(description: "family not recognized: neither " + Family.allCases.map(\.name).joined(separator: ", "))
        }
        guard let ref = try reference(family) else {
            throw EngineError.importNeedsFamily(file: (file as NSString).lastPathComponent, model: ModelCard.of(family).id)
        }

        // Z-Image: a fused qkv AND its to_q/to_k/to_v is a duplicate (Anime-Z) — diffusers skips
        // the qkv and the bare `out`; we do the same.
        if family == .zImage {
            let readableNames = Set(renamed.values)
            for (m, l) in renamed where l.hasSuffix(".attention.qkv") && readableNames.contains(String(l.dropLast(4)) + ".to_q")
                || l.hasSuffix(".attention.out") && readableNames.contains(String(l.dropLast(4)) + ".to_out.0") {
                pairs[m] = nil; notes.append("\(m) : duplicate of a split module, ignored")
            }
            moduleOrder = moduleOrder.filter { pairs[$0] != nil }
        }

        // ── Targets, shapes, scales ──────────────────────────────────────────────────────
        struct Plan { let target: String; let down: String; let up: String; let rows: Range<Int>?
                      let r: Int; let k: Int; let n: Int; let scale: Float }
        var plans: [String: Plan] = [:]
        var ranks = Set<Int>()
        var alphas = 0, peftScaled = 0
        let peft = try PEFTScale.read(metadata)
        for m in moduleOrder {
            let p = pairs[m]!
            let a = source.shape(p.down!)!, b = source.shape(p.up!)!
            guard a.count == 2, b.count == 2, a[0] == b[1] else {
                throw Numerics.Failure(description: "\(m) : shapes \(a) and \(b) incompatible")
            }
            let r = a[0], kTotal = a[1]
            var scale: Float = 1
            guard r > 0 else { throw Numerics.Failure(description: "\(m) : rank 0") }
            if p.alpha != nil, path == nil {
                alphas += 1
            } else if let al = p.alpha {
                let v = try source.scalar(al) ?? Float(r)
                guard v.isFinite else { throw Numerics.Failure(description: "\(m) : alpha \(v) is not finite") }
                scale = v / Float(r); alphas += 1
            } else if let peft {
                scale = peft.scale(m, r: r); peftScaled += 1
            }
            let parts = try targets(renamed[m]!, family: family, ref: ref)
            // Ranges: they must cover the whole fused weight, no more, no less.
            if let end = parts.compactMap(\.rows?.upperBound).max(), end != b[0] {
                throw Numerics.Failure(description: "\(m) : up has \(b[0]) rows, the map's parts make \(end) — "
                                      + "this LoRA is not for this model")
            }
            for c in parts {
                guard let shape = ref[c.module + ".weight"] else {
                    throw Numerics.Failure(description: "\(m) → \(c.module) : does not exist in \(family.name) — "
                                          + "this LoRA is not for this model")
                }
                let rows = c.rows ?? c.part.map { p in let n = b[0] / p.1; return (p.0 * n)..<((p.0 + 1) * n) }
                let n = rows?.count ?? b[0], k = kTotal
                guard shape == [n, k] else {
                    throw Numerics.Failure(description: "\(c.module) : ΔW \([n, k]) for a weight \(shape) — "
                                          + "this LoRA is not for this model")
                }
                guard plans[c.module] == nil else { throw Numerics.Failure(description: "\(c.module) targeted twice") }
                plans[c.module] = Plan(target: c.module, down: p.down!, up: p.up!, rows: rows,
                                       r: r, k: k, n: n, scale: scale)
                ranks.insert(r)
            }
        }
        let split = plans.values.filter { $0.target.hasSuffix(".img_mlp.gate_layer") && $0.rows != nil }.count
        if family == .qwenImage21, split > 0 {
            notes.append("fused img_mlp.gate_up (ai-toolkit) on \(split) modules: up split by rows [gate_layer; proj], down shared")
        }
        let scales = Set(plans.values.map(\.scale)).sorted().map { String(format: "%g", $0) }.joined(separator: ", ")
        if alphas > 0 {
            notes.append("alpha (kohya) on \(alphas) modules: ΔW × α/r = \(scales), folded into up")
        }
        if peftScaled > 0, let peft {
            notes.append("lora_adapter_metadata (PEFT) on \(peftScaled) modules: lora_alpha \(String(format: "%g", peft.alpha))"
                         + (peft.alphaPattern.isEmpty ? "" : ", alpha_pattern \(peft.alphaPattern.count) keys")
                         + (peft.rsLoRA ? ", rsLoRA α/√r" : "") + " → ΔW × \(scales), folded into up")
        } else if alphas == 0, let a = metadata["ss_network_alpha"] ?? metadata["alpha"] {
            notes.append("alpha \(a) declared in metadata only: ignored, like diffusers and ComfyUI")
        }

        let examination = Examination(family: family, modules: plans.count, ranks: ranks.sorted())
        guard let path else { return (examination, nil) }

        // ── The DiT map's order, then the writing ───────────────────────────────────
        let ditOrder = ForgeDiT.order(Array(ref.keys), family: family)
        let rank = Dictionary(uniqueKeysWithValues: ditOrder.enumerated().map { ($1, $0) })
        let modules = plans.keys.sorted { (rank[$0 + ".weight"] ?? .max, $0) < (rank[$1 + ".weight"] ?? .max, $1) }
        var tensors: [MapWriter.Tensor] = []
        var permuted = 0
        for m in modules {
            let p = plans[m]!
            tensors.append(.init(name: m + ".down", shape: [p.k, p.r], dtype: .bfloat16, transposed: nil) {
                let a = try source.read(p.down)
                var t = [Float](repeating: 0, count: a.count)
                a.withUnsafeBufferPointer { s in t.withUnsafeMutableBufferPointer { d in
                    Numerics.transpose(s.baseAddress!, rows: p.r, columns: p.k, to: d.baseAddress!) } }
                return t
            })
            let rope = family == .anima && m.firstMatch(#"\.attn1\.to_[qk]$"#) != nil
            if rope { permuted += 1 }
            tensors.append(.init(name: m + ".up", shape: [p.r, p.n], dtype: .bfloat16, transposed: nil) {
                var b = try source.read(p.up)
                if let l = p.rows { b = Array(b[(l.lowerBound * p.r)..<(l.upperBound * p.r)]) }
                if rope { b = ForgeDiT.interleave(b, rows: p.n, columns: p.r, head: ForgeDiT.animaHeadSize) }
                if p.scale != 1 { b.withUnsafeMutableBufferPointer { x in for i in x.indices { x[i] *= p.scale } } }
                var t = [Float](repeating: 0, count: b.count)
                b.withUnsafeBufferPointer { s in t.withUnsafeMutableBufferPointer { d in
                    Numerics.transpose(s.baseAddress!, rows: p.n, columns: p.r, to: d.baseAddress!) } }
                return t
            })
        }
        if permuted > 0 { notes.append("RoPE permutation: \(permuted) `up` (attn1.to_q, attn1.to_k)") }

        // Named after its file, which the user chose and sees in the Finder; what the trainer wrote
        // is often a leftover of its interface (`my_first_lora_v1_copy_copy_copy_copy`) and goes
        // to `trainer_name`, shown in a tooltip.
        let raw = ((file as NSString).lastPathComponent as NSString).deletingPathExtension
        let trainer = trainerName(metadata), triggerWords = triggers(metadata)
        let sortedRanks = ranks.sorted()
        let shortEntries = metadata.filter { $0.value.count < 200 }.sorted { $0.key < $1.key }
        let fileSHA256 = try sha256(file: file)
        let tally = try MapWriter.write(to: path, tensors: tensors, header: { _ in var h: [OrderedJSON.Pair] = [
            .init("format", 1), .init("kind", "lora"), .init("page", .integer(MapWriter.page)),
            .init("map_dtype", "bfloat16"), .init("linear_weights_transposed", true),
            // on-disk key, kept for existing maps/profiles (.silicon header: cible, modele, echelle, source, fichier, tenseurs…)
            // `carte` is the Standard DiT's name **on purpose, whatever version is installed**: a LoRA
            // is checked against the family's DiT names (`dit.json`, the publisher's), which the
            // Compact shares name for name (kept 8-bit) as an imported checkpoint does — it targets
            // the family, not one map, and is applied to whichever DiT renders. Nothing reads the key
            // (`LoRA` and the catalog read `modele`, the family); writing the installed version's
            // name would make the same LoRA forge to two different headers.
            .init("cible", .object([.init("modele", .string(family.rawValue)), .init("kind", .string(family.ditKind)),
                                   .init("carte", .string(family.ditMap))])),
            .init("nom", .string(raw)),
            .init("rang", sortedRanks.count == 1 ? .integer(sortedRanks[0]) : .list(sortedRanks.map { .integer($0) })),
            .init("echelle", .real(1)),
            .init("modules", .list(modules.map { .string($0) })),
            .init("source", .object([.init("fichier", .string((file as NSString).lastPathComponent)),
                                    .init("sha256", .string(fileSHA256)),
                                    .init("tenseurs", .integer(source.names.count)),
                                    .init("metadata", .object(shortEntries.map { .init($0.key, .string($0.value)) }))])),
        ]
            // Optional: a map forged before them has neither, and shows no tooltip.
            if let trainer { h.append(.init("trainer_name", .string(trainer))) }
            if !triggerWords.isEmpty { h.append(.init("triggers", .list(triggerWords.map { .string($0) }))) }
            h.append(.init("order", .list(tensors.map { .string($0.name) })))
            return h
        })
        return (examination, Report(tally: tally, family: family, name: raw, modules: modules.count, ranks: sortedRanks, notes: notes))
    }
}
