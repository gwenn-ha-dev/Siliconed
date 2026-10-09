import Foundation

/// **An architecture family**: what Siliconed knows how to render. An imported model (a Civitai
/// checkpoint) is a member of one — same architecture, other weights — and borrows from its family
/// everything the user does not bring: tokenizers, text encoder, VAE.
public enum Family: String, CaseIterable, Sendable, Codable {
    case zImage = "z-image"
    case anima
    case krea2
    /// FLUX.2 [klein] 4B. The 9B will have its own family: other shapes, other encoder (Qwen3-8B).
    case klein4b = "klein-4b"
    /// ERNIE-Image Turbo (Baidu): single-stream DiT, Ministral-3 encoder, FLUX.2 VAE.
    case ernie = "ernie-image"
    /// Qwen-Image-2.1 (Qwen) and Viggle's turbo LoRA: single-stream DiT, Qwen3-VL-8B encoder that
    /// sees the reference images, 64-channel VAE; generation and editing by instruction.
    case qwenImage21 = "qwen-image-2.1"

    /// The families offered to the user — `install all`, the app's library sheet. All of them
    /// since Qwen-Image-2.1 renders swap-free at its worst case; a family that
    /// cannot render yet is filtered out here, so that nobody downloads what cannot render.
    public static let offered: [Family] = allCases

    public var name: String {
        switch self {
        case .zImage: return "Z-Image"
        case .anima: return "Anima"
        case .krea2: return "Krea 2"
        case .klein4b: return "FLUX.2 [klein] 4B"
        case .ernie: return "ERNIE-Image"
        case .qwenImage21: return "Qwen-Image-2.1"
        }
    }

    /// A FLUX.2 [klein] DiT — what `KleinDiT` reads, size aside.
    package var isKlein: Bool { self == .klein4b }

    /// The `kind` of the DiT map — what the engine requires in order to read it.
    package var ditKind: String {
        switch self {
        case .zImage: return "z-image-turbo-dit"
        case .anima: return "anima-turbo-dit"
        case .krea2: return "krea2-turbo-dit"
        case .klein4b: return "flux2-klein-4b-dit"
        case .ernie: return "ernie-image-turbo-dit"
        // The base model: the turbo is a LoRA applied unmerged on top (`turboLoRA`).
        case .qwenImage21: return "qwen-image-2.1-dit"
        }
    }

    /// The base DiT map, in `store/`.
    package var ditMap: String {
        switch self {
        case .zImage: return "z-image-turbo-dit.v1.silicon"
        case .anima: return "anima-turbo-dit.v0.silicon"
        case .krea2: return "krea2-turbo-dit.v0.silicon"
        case .klein4b: return "flux2-klein-4b-dit.v0.silicon"
        case .ernie: return "ernie-image-turbo-dit.v0.silicon"
        case .qwenImage21: return "qwen-image-2.1-dit.v0.silicon"
        }
    }

    /// The text encoder map, in `store/`.
    package var encoderMap: String {
        switch self {
        case .zImage: return "qwen3-4b-encoder.v0.silicon"
        case .anima: return "qwen3-0.6b-encoder.v0.silicon"
        case .krea2: return "qwen3-vl-4b-encoder.v0.silicon"
        // Z-Image's Qwen3-4B, bit for bit (checked tensor by tensor): a single map for two families.
        case .klein4b: return "qwen3-4b-encoder.v0.silicon"
        // The language model of the published `Mistral3Model` (Ministral-3 3B), without its vision tower.
        case .ernie: return "ministral3-3b-encoder.v0.silicon"
        // The whole `Qwen3VLModel`: language model **and** vision tower — the references go through it.
        case .qwenImage21: return "qwen3-vl-8b-encoder.v0.silicon"
        }
    }

    /// **The versions a family installs in**: every family has its Standard;
    /// Z-Image and Qwen-Image-2.1 also have a Compact and a Light.
    public var variants: [Variant] {
        self == .zImage || self == .qwenImage21 ? [.standard, .compact, .light] : [.standard]
    }

    /// **The version preselected before an installation** (the welcome cards, « Install… »): the
    /// Standard (below the publisher's weights only by the user's choice) — except
    /// on a Mac of 8 GB or less, where the Light is preselected. Preselected,
    /// never imposed: the three versions are shown, and the user picks.
    public func preselectedVariant(physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> Variant {
        physicalMemory <= Variant.lightMemory && variants.contains(.light) ? .light : .standard
    }

    /// The DiT map of a version, in `store/`. The Standard's is `ditMap`; the others have their own
    /// name, so that two are never taken for each other (one is removed when another installs).
    package func ditMap(_ v: Variant) -> String {
        switch (self, v) {
        case (.zImage, .compact): return "z-image-turbo-dit.compact.v0.silicon"
        case (.qwenImage21, .compact): return "qwen-image-2.1-dit.compact.v0.silicon"
        case (.zImage, .light): return "z-image-turbo-dit.light.v0.silicon"
        case (.qwenImage21, .light): return "qwen-image-2.1-dit.light.v0.silicon"
        default: return ditMap
        }
    }

    /// The text encoder map of a version, in `store/`. **Z-Image's Compact encoder is not FLUX.2
    /// [klein]'s**: klein reads the publisher's Qwen3-4B (`qwen3-4b-encoder.v0`), the map Z-Image
    /// Standard shares with it; the Compact's is Disty0's 8-bit one, under a name of its own. **The
    /// Light reads the Compact's encoder, the same map**: no text encoder is published under 8 bits
    /// in a form the forge reads, and quantifying one ourselves is excluded (never below the publisher's
    /// weights unless the user chooses so). Switching between
    /// Compact and Light therefore keeps the encoder (`Library.releasable`).
    package func encoderMap(_ v: Variant) -> String {
        switch (self, v) {
        case (.zImage, .compact), (.zImage, .light): return "qwen3-4b-encoder.compact.v0.silicon"
        case (.qwenImage21, .compact), (.qwenImage21, .light): return "qwen3-vl-8b-encoder.compact.v0.silicon"
        default: return encoderMap
        }
    }

    /// **A LoRA that belongs to the family**, under `store/composants/<famille>/`: Qwen-Image-2.1 is
    /// a turbo only through Viggle's LoRA, applied **unmerged** (merged into bf16 weights it loses
    /// part of its update — Viggle's README) and switchable (the 9-step mode ends on the base
    /// model). Out of `store/*.lora.silicon`, so that it is not listed among the user's LoRAs.
    package var turboLoRA: String? {
        self == .qwenImage21 ? "viggle-turbo-v0.3.lora.silicon" : nil
    }
    /// Does the Standard version already download a part made by a third party (Viggle's turbo
    /// LoRA)? The license sheet must not say it all comes from the publisher. Public: the app's
    /// phrases are extracted without the package's symbols (`tools/translations.sh`).
    public var hasThirdPartyTurbo: Bool { turboLoRA != nil }

    package init?(ditKind: String) {
        guard let f = Family.allCases.first(where: { $0.ditKind == ditKind }) else { return nil }
        self = f
    }
}

/// **Which weights a family is installed with**. A single version of a family is
/// installed at a time: installing the other replaces it, and gives its space back.
public enum Variant: String, CaseIterable, Sendable, Codable {
    /// The publisher's weights, as published — the default (never below the
    /// publisher's weights unless the user chooses so).
    case standard
    /// 8-bit weights published by a third party, **DiT and text encoder**, kept 8-bit in the maps:
    /// about 40 % smaller on disk, an image slightly different from the Standard's, a step a little
    /// slower (the weights are widened at each evaluation). Tokenizers, VAE and the
    /// turbo LoRA are the Standard's.
    case compact
    /// **The DiT in 4 to 6 bits as a third party publishes it** (unsloth's GGUF Q4_K_M, its blocks
    /// kept whole), the Compact's 8-bit text encoder, the Standard's tokenizers, VAE and turbo
    /// LoRA: the smallest on disk and the least read from it at each step — another set of
    /// weights, an image that may compose differently. Preselected on a Mac of 8 GB
    /// (`Family.preselectedVariant`).
    case light

    /// The physical memory at or below which the Light is preselected: 8 GiB, what an 8 GB Mac reports.
    package static let lightMemory: UInt64 = 8 << 30
}

/// Anima's text adapter lives next to the DiT map it comes from.
package func adapterPath(fromMap map: String) -> String {
    // on-disk file name, kept for existing installations
    (map.hasSuffix(".silicon") ? String(map.dropLast(8)) : map) + ".adaptateur.safetensors"
}

// MARK: - From published names to map names

/// What the forge knows about each tensor of a source, without reading anything.
package protocol TensorCatalog: AnyObject {
    var names: [String] { get }
    func shape(_ name: String) -> [Int]?
    func dtype(_ name: String) -> String?
    func read(_ name: String) throws -> [Float]
    /// An 8-bit weight's kind and scale (`nil`: not 8-bit).
    func quantization(_ name: String) -> QuantizedKind?
    func readQuantized(_ name: String) throws -> QuantizedTensor
}

extension TensorCatalog {
    package func quantization(_ name: String) -> QuantizedKind? { nil }
    package func readQuantized(_ name: String) throws -> QuantizedTensor {
        throw Numerics.Failure(description: "\(name) is not 8-bit")
    }
}

extension TensorSource: TensorCatalog {}

/// A tensor under its map name (diffusers), and how to obtain it from the source.
package struct Provenance {
    /// The **published** shape (`[output, input]` for a `Linear`), before transposition.
    package let shape: [Int]
    package let dtypeSource: String
    package let read: () throws -> [Float]
    /// An 8-bit weight: its kind and scale, and its bytes after the same selection as `read` —
    /// rows and columns moved, never changed (`QuantizedTensor`). `nil`: not 8-bit.
    package var quantization: QuantizedKind? = nil
    package var readQuantized: (() throws -> QuantizedTensor)? = nil
    /// An 8-bit source tensor that a transformation forces into fp32 (`ForgeDiT` writes it so and
    /// says why): a column share that cuts a scale block.
    package var dequantizedBecause: String? = nil

    init(shape: [Int], dtypeSource: String, read: @escaping () throws -> [Float]) {
        self.shape = shape; self.dtypeSource = dtypeSource; self.read = read
    }

    /// The same provenance, with its 8-bit form when the source tensor is 8-bit.
    func quantized(_ source: TensorCatalog, _ name: String,
                   _ select: @escaping (QuantizedTensor) -> QuantizedTensor) -> Provenance {
        guard let q = source.quantization(name) else { return self }
        var p = self
        p.quantization = q
        p.readQuantized = { select(try source.readQuantized(name)) }
        return p
    }
}

/// The result of normalization: the DiT's tensors, and what is not part of it.
package struct NormalizedSource {
    package var dit: [String: Provenance] = [:]
    /// The keys of Anima's adapter (`llm_adapter.*`), original published names.
    package var adapter: [String] = []
    /// The embedded components we do not use (encoder, VAE) — we say so, we do not use them.
    package var ignored: [String] = []
    /// The naming recognized (`ComfyUI / original`, `diffusers`).
    package var naming = ""
}

package enum Recipes {
    /// The prefixes a checkpoint glues in front of the model's names.
    static let prefixes = ["model.diffusion_model.", "diffusion_model.", "net.", "transformer.", "model."]
    /// The components an "all-in-one" checkpoint embeds and that the family already provides.
    static let embedded = ["text_encoders.", "text_encoder.", "cond_stage_model.", "conditioner.",
                            "first_stage_model.", "vae.", "te.", "clip."]

    static func withoutPrefix(_ name: String) -> String {
        for p in prefixes where name.hasPrefix(p) { return String(name.dropFirst(p.count)) }
        return name
    }

    /// The rows `[begin, begin + number)` of a `[rows, columns]` tensor — an unequal share of a
    /// fused weight (FLUX.2's `to_qkv_mlp_proj`: q, k, v, gate, up).
    static func rows(_ source: TensorCatalog, _ name: String, begin: Int, number: Int) -> Provenance {
        let f = source.shape(name)!
        let columns = f.dropFirst().reduce(1, *)
        return Provenance(shape: [number] + f.dropFirst(), dtypeSource: source.dtype(name)!) {
            let all = try source.read(name)
            return Array(all[(begin * columns)..<((begin + number) * columns)])
        }.quantized(source, name) { $0.rows(begin: begin, number: number) }
    }

    /// The columns `[begin, begin + number)` of a `[rows, columns]` tensor — a share of a `Linear`'s
    /// **input** (`to_out` of a FLUX.2 single block: attention, then MLP). The caller has checked
    /// that the tensor is a matrix and that the columns exist (`matrix`).
    ///
    /// A packed GGUF weight's share must be whole blocks (32 inputs for
    /// Q4_0…Q5_1, 256 for a K-quant): a block's inputs share scales that nobody published for a part
    /// of them, and the map does not widen a 4-bit weight to fp32 — the file is refused, with the reason.
    static func columns(_ source: TensorCatalog, _ name: String, begin: Int, number: Int) throws -> Provenance {
        if let kind = source.quantization(name)?.kind, let block = kind.packedBlock,
           !QuantizedTensor.columnsKeepBlocks(begin: begin, number: number, block: block.values) {
            throw Numerics.Failure(description: "\(name) : \(source.dtype(name) ?? kind.rawValue) cut on its input at columns "
                                   + "[\(begin), \(begin + number)), inside a block of \(block.values) — "
                                   + "a GGUF block is kept whole, never widened: this file cannot be imported")
        }
        let f = source.shape(name)!
        let (rows, width) = (f[0], f[1])
        let p = Provenance(shape: [rows, number], dtypeSource: source.dtype(name)!) {
            let all = try source.read(name)
            var part = [Float](repeating: 0, count: rows * number)
            for l in 0..<rows {
                for j in 0..<number { part[l * number + j] = all[l * width + begin + j] }
            }
            return part
        }
        if let b = source.quantization(name)?.block, !QuantizedTensor.columnsKeepBlocks(begin: begin, number: number, block: b) {
            var q = p
            q.dequantizedBecause = " (columns [\(begin), \(begin + number)) cut its blocks of \(b) scales)"
            return q
        }
        // A convrot weight mixes the inputs of each group: a share of the input keeps it only on a
        // group boundary.
        if let g = source.quantization(name)?.rotation, !QuantizedTensor.columnsKeepBlocks(begin: begin, number: number, block: g) {
            var q = p
            q.dequantizedBecause = " (columns [\(begin), \(begin + number)) cut its convrot groups of \(g))"
            return q
        }
        return p.quantized(source, name) { $0.columns(begin: begin, number: number) }
    }

    /// The two halves of a tensor swapped on axis 0: BFL's `(shift, scale)` → diffusers'
    /// `(scale, shift)` (`swap_scale_shift`, FLUX.2's final layer).
    static func swappedHalves(_ source: TensorCatalog, _ name: String) -> Provenance {
        let f = source.shape(name)!
        return Provenance(shape: f, dtypeSource: source.dtype(name)!) {
            let all = try source.read(name), half = all.count / 2
            return Array(all[half...] + all[..<half])
        }.quantized(source, name) { q in q.selectingRows(Array((q.rows / 2)..<q.rows) + Array(0..<(q.rows / 2))) }
    }

    /// The rows `[begin, end)` of a `[rows, columns]` tensor — a share of a fused qkv. It runs while
    /// the names are normalized, **before** the shapes are checked against `dit.json`: a scalar or a
    /// row count the parts don't divide (a hostile or broken file) is refused here, not indexed.
    static func rows(_ source: TensorCatalog, _ name: String, part: Int, outOf parts: Int) throws -> Provenance {
        let f = source.shape(name)!
        guard f.count >= 1, f[0] > 0, f[0] % parts == 0 else {
            throw Numerics.Failure(description: "\(name) : shape \(f) does not split in \(parts) row blocks")
        }
        let n = f[0] / parts, columns = f.dropFirst().reduce(1, *)
        return Provenance(shape: [n] + f.dropFirst(), dtypeSource: source.dtype(name)!) {
            let all = try source.read(name)
            return Array(all[(part * n * columns)..<((part + 1) * n * columns)])
        }.quantized(source, name) { $0.rows(begin: part * n, number: n) }
    }

    static func asPublished(_ source: TensorCatalog, _ name: String, shape: [Int]? = nil) -> Provenance {
        Provenance(shape: shape ?? source.shape(name)!, dtypeSource: source.dtype(name)!) { try source.read(name) }
            .quantized(source, name) { $0 }
    }

    /// **Recognizes a DiT's family from its names**, prefixes removed. `nil`: none.
    package static func family(fromNames names: [String]) -> Family? {
        let n = Set(names.map(withoutPrefix))
        func a(_ pattern: String) -> Bool { n.contains { $0.range(of: pattern, options: .regularExpression) != nil } }
        if a(#"^(single_transformer_blocks\.\d+\.attn\.to_qkv_mlp_proj|double_stream_modulation_img\.lin)"#) { return .klein4b }
        if a(#"^layers\.\d+\.adaLN_sa_ln\.weight$"#) && a(#"^layers\.\d+\.self_attention\.to_q\."#) { return .ernie }
        if a(#"^transformer_blocks\.\d+\.img_mlp\.(gate_layer|gate_up)\."#) && a(#"^txt_in\.text_norm\."#) { return .qwenImage21 }
        if a(#"^(layers|noise_refiner|context_refiner)\.\d+\.attention\.(qkv|to_q)\."#) && a(#"^cap_embedder\."#) { return .zImage }
        if a(#"^(blocks\.\d+\.attn\.wq|txtfusion\.|transformer_blocks\.\d+\.attn\.to_gate)"#) { return .krea2 }
        if a(#"^(blocks\.\d+\.self_attn\.q_proj|llm_adapter\.|transformer_blocks\.\d+\.attn1\.to_q)"#) { return .anima }
        return nil
    }

    /// **Normalizes a source**: map names, qkv split, tables reshaped. Strict: a key that no rule
    /// takes causes a failure, rather than being silently forgotten (trap 3.10 — `strict=False`
    /// leaves a layer at its initialization).
    package static func normalize(_ source: TensorCatalog, family: Family) throws -> NormalizedSource {
        var r = NormalizedSource()
        var keys: [(published: String, name: String)] = []
        for published in source.names {
            let name = withoutPrefix(published)
            if embedded.contains(where: { published.hasPrefix($0) }) { r.ignored.append(published); continue }
            keys.append((published, name))
        }
        func place(_ name: String, _ p: Provenance) throws {
            guard r.dit[name] == nil else { throw Numerics.Failure(description: "\(name) produced twice") }
            r.dit[name] = p
        }
        switch family {
        case .zImage:
            let origin = keys.contains { $0.name.contains(".attention.qkv.") || $0.name.hasPrefix("final_layer.") || $0.name.hasPrefix("x_embedder.") }
            r.naming = origin ? "original Z-Image (ComfyUI)" : "diffusers"
            // `convert_z_image_transformer_checkpoint_to_diffusers`, in the same order.
            let renames = [("final_layer.", "all_final_layer.2-1."), ("x_embedder.", "all_x_embedder.2-1."),
                              (".attention.out.bias", ".attention.to_out.0.bias"),
                              (".attention.k_norm.weight", ".attention.norm_k.weight"),
                              (".attention.q_norm.weight", ".attention.norm_q.weight"),
                              (".attention.out.weight", ".attention.to_out.0.weight")]
            for (published, raw) in keys {
                var name = raw
                if origin { for (a, b) in renames { name = name.replacingOccurrences(of: a, with: b) } }
                if name.hasSuffix(".attention.qkv.weight") {
                    let root = String(name.dropLast(".qkv.weight".count))
                    for (k, x) in ["to_q", "to_k", "to_v"].enumerated() {
                        try place("\(root).\(x).weight", try rows(source, published, part: k, outOf: 3))
                    }
                } else if name == "cap_pad_token" || name == "x_pad_token", let f = source.shape(published), f.count == 1 {
                    // stable-diffusion.cpp's GGUFs (leejet) drop the trailing `ne` of 1: `[3840]` for the
                    // published `[1, 3840]`. Only a leading 1 is added back, bytes unchanged.
                    try place(name, asPublished(source, published, shape: [1] + f))
                } else {
                    try place(name, asPublished(source, published))
                }
            }
        case .krea2:
            let origin = keys.contains { $0.name.hasPrefix("blocks.") || $0.name.hasPrefix("txtfusion.") || $0.name.hasPrefix("first.") }
            r.naming = origin ? "original Krea 2 (krea-ai, ComfyUI)" : "diffusers"
            for (published, raw) in keys {
                guard origin else { try place(raw, asPublished(source, published)); continue }
                guard let (name, reshaped) = krea2(raw, shape: source.shape(published)!) else {
                    throw Numerics.Failure(description: "\(published) : no Krea 2 renaming rule")
                }
                try place(name, asPublished(source, published, shape: reshaped))
            }
        case .klein4b:
            let origin = keys.contains { $0.name.hasPrefix("double_blocks.") || $0.name.hasPrefix("single_blocks.") }
            r.naming = origin ? "original FLUX.2 (BFL, ComfyUI)" : "diffusers"
            for (published, raw) in keys {
                if origin, raw == "final_layer.adaLN_modulation.1.weight" {
                    try place("norm_out.linear.weight", swappedHalves(source, published)); continue
                }
                let name = origin ? try flux2(raw) : raw
                for (target, p) in try flux2Split(source, published, name) { try place(target, p) }
            }
        case .qwenImage21:
            // The diffusers naming, no bias — and in unsloth's GGUF (the ComfyUI layout) the MLP's two
            // input projections fused, `img_mlp.gate_up` = `[gate_layer; proj]` on the output axis:
            // the order ai-toolkit's LoRAs use too (`ForgeLoRA`), and the one the published rows show
            // (each half against the unsloth int8 checkpoint of the same base: correlation 1.0000
            // with its own projection, < 0.14 with the other — measured).
            let fused = keys.contains { $0.name.hasSuffix(".img_mlp.gate_up.weight") }
            r.naming = fused ? "diffusers, MLP gate_up fused (ComfyUI)" : "diffusers"
            for (published, name) in keys {
                if name.hasSuffix(".img_mlp.gate_up.weight") {
                    let root = String(name.dropLast("gate_up.weight".count))
                    try place(root + "gate_layer.weight", try rows(source, published, part: 0, outOf: 2))
                    try place(root + "proj.weight", try rows(source, published, part: 1, outOf: 2))
                } else {
                    try place(name, asPublished(source, published))
                }
            }
        case .ernie:
            // A single published naming (diffusers). `x_embedder` is a 1×1 convolution:
            // `[4096, 128, 1, 1]` reads as the `Linear` `[4096, 128]` that it is.
            r.naming = "diffusers"
            for (published, name) in keys {
                let f = source.shape(published)!
                if name == "x_embedder.proj.weight", f.count == 4, f[2] == 1, f[3] == 1 {
                    try place(name, asPublished(source, published, shape: [f[0], f[1]]))
                } else {
                    try place(name, asPublished(source, published))
                }
            }
        case .anima:
            let origin = keys.contains { $0.name.hasPrefix("blocks.") || $0.name.hasPrefix("x_embedder.") || $0.name.hasPrefix("llm_adapter.") }
            r.naming = origin ? "original Anima (ComfyUI / Cosmos)" : "diffusers"
            for (published, raw) in keys {
                if raw.hasPrefix("llm_adapter.") { r.adapter.append(published); continue }
                guard origin else { try place(raw, asPublished(source, published)); continue }
                try place(try anima(raw), asPublished(source, published))
            }
        }
        return r
    }

    // ── Krea 2: the `krea-ai/krea-2` repo → `Krea2Transformer2DModel` ─────────────────────
    static let kreaAttn = ["wq": "to_q", "wk": "to_k", "wv": "to_v", "wo": "to_out.0", "gate": "to_gate"]

    /// The diffusers name and the published shape it must have (`mod.lin`: `[6·D]` → `[6, D]`).
    static func krea2(_ name: String, shape: [Int]) -> (String, [Int])? {
        // The block: `blocks.N.` or `txtfusion.(layerwise|refiner)_blocks.N.`
        let head: String, remaining: String
        if let x = name.firstMatch(#"^blocks\.(\d+)\.(.+)$"#) {
            head = "transformer_blocks.\(x[1])."; remaining = x[2]
        } else if let x = name.firstMatch(#"^txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(.+)$"#) {
            head = "text_fusion.\(x[1]).\(x[2])."; remaining = x[3]
        } else {
            let singles: [String: String] = [
                "first.weight": "img_in.weight", "first.bias": "img_in.bias",
                "last.linear.weight": "final_layer.linear.weight", "last.linear.bias": "final_layer.linear.bias",
                "last.modulation.lin": "final_layer.scale_shift_table", "last.norm.scale": "final_layer.norm.weight",
                "tmlp.0.weight": "time_embed.linear_1.weight", "tmlp.0.bias": "time_embed.linear_1.bias",
                "tmlp.2.weight": "time_embed.linear_2.weight", "tmlp.2.bias": "time_embed.linear_2.bias",
                "tproj.1.weight": "time_mod_proj.weight", "tproj.1.bias": "time_mod_proj.bias",
                "txtmlp.0.scale": "txt_in.norm.weight",
                "txtmlp.1.weight": "txt_in.linear_1.weight", "txtmlp.1.bias": "txt_in.linear_1.bias",
                "txtmlp.3.weight": "txt_in.linear_2.weight", "txtmlp.3.bias": "txt_in.linear_2.bias",
                "txtfusion.projector.weight": "text_fusion.projector.weight",
            ]
            return singles[name].map { ($0, shape) }
        }
        if let x = remaining.firstMatch(#"^attn\.(wq|wk|wv|wo|gate)\.weight$"#) { return (head + "attn.\(kreaAttn[x[1]]!).weight", shape) }
        if let x = remaining.firstMatch(#"^attn\.qknorm\.(q|k)norm\.scale$"#) { return (head + "attn.norm_\(x[1]).weight", shape) }
        if let x = remaining.firstMatch(#"^mlp\.(gate|up|down)\.weight$"#) { return (head + "ff.\(x[1]).weight", shape) }
        if remaining == "prenorm.scale" { return (head + "norm1.weight", shape) }
        if remaining == "postnorm.scale" { return (head + "norm2.weight", shape) }
        if remaining == "mod.lin", shape.count == 1, shape[0] % 6 == 0 { return (head + "scale_shift_table", [6, shape[0] / 6]) }
        return nil
    }

    // ── FLUX.2: BFL → diffusers (`convert_flux2_transformer_checkpoint_to_diffusers`) ───────
    static let flux2Top: [String: String] = [
        "img_in.weight": "x_embedder.weight", "txt_in.weight": "context_embedder.weight",
        "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
        "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
        "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
        "final_layer.linear.weight": "proj_out.weight",
    ]
    static let flux2Double: [String: String] = [
        "img_attn.qkv.weight": "attn.qkv", "txt_attn.qkv.weight": "attn.added_qkv",
        "img_attn.norm.query_norm.scale": "attn.norm_q.weight", "img_attn.norm.key_norm.scale": "attn.norm_k.weight",
        "txt_attn.norm.query_norm.scale": "attn.norm_added_q.weight", "txt_attn.norm.key_norm.scale": "attn.norm_added_k.weight",
        "img_attn.proj.weight": "attn.to_out.0.weight", "txt_attn.proj.weight": "attn.to_add_out.weight",
        "img_mlp.0.weight": "ff.linear_in.weight", "img_mlp.2.weight": "ff.linear_out.weight",
        "txt_mlp.0.weight": "ff_context.linear_in.weight", "txt_mlp.2.weight": "ff_context.linear_out.weight",
    ]
    static let flux2Simple: [String: String] = [
        "linear1.weight": "attn.to_qkv_mlp_proj.weight", "linear2.weight": "attn.to_out.weight",
        "norm.query_norm.scale": "attn.norm_q.weight", "norm.key_norm.scale": "attn.norm_k.weight",
    ]

    /// A BFL name → the diffusers name (before splitting; `attn.qkv` and `attn.added_qkv` are the
    /// two fused qkv of a double block, which `flux2Split` splits).
    static func flux2(_ name: String) throws -> String {
        if let h = flux2Top[name] { return h }
        if let x = name.firstMatch(#"^double_blocks\.(\d+)\.(.+)$"#), let c = flux2Double[x[2]] { return "transformer_blocks.\(x[1]).\(c)" }
        if let x = name.firstMatch(#"^single_blocks\.(\d+)\.(.+)$"#), let c = flux2Simple[x[2]] { return "single_transformer_blocks.\(x[1]).\(c)" }
        throw Numerics.Failure(description: "\(name) : no FLUX.2 renaming rule")
    }

    /// **Splitting the fused weights**, so that each `KleinDiT` GEMM writes a contiguous slice:
    /// `to_qkv_mlp_proj` → `q, k, v, gate, up`; `linear_in` → `gate, up` (the half passed through
    /// SiLU first, `Flux2SwiGLU`); a single block's `to_out` → `attn, mlp` on its input; a BFL qkv
    /// → `to_q/to_k/to_v` or `add_{q,k,v}_proj`. The rest passes through as is.
    static func flux2Split(_ source: TensorCatalog, _ published: String, _ name: String) throws -> [(String, Provenance)] {
        let f = source.shape(published)!
        if name.hasSuffix(".attn.qkv") || name.hasSuffix(".attn.added_qkv") {
            let root = String(name.prefix(upTo: name.range(of: ".attn.", options: .backwards)!.upperBound))
            let parts = name.hasSuffix(".added_qkv") ? ["add_q_proj", "add_k_proj", "add_v_proj"] : ["to_q", "to_k", "to_v"]
            return try parts.enumerated().map { (root + $1 + ".weight", try rows(source, published, part: $0, outOf: 3)) }
        }
        // A file's header decides the shape: a 1-D tensor under a weight's name is refused by name,
        // not indexed past its end.
        func matrix() throws {
            guard f.count == 2 else { throw Numerics.Failure(description: "\(published) : a matrix expected, shape \(f)") }
        }
        if name.hasSuffix(".attn.to_qkv_mlp_proj.weight") {
            try matrix()
            let d = f[1], h = (f[0] - 3 * d) / 2
            guard h > 0, 3 * d + 2 * h == f[0] else { throw Numerics.Failure(description: "\(published) : unexpected shape \(f)") }
            let root = String(name.dropLast("weight".count))
            var begin = 0
            return zip(["q", "k", "v", "gate", "up"], [d, d, d, h, h]).map { part, n in
                defer { begin += n }
                return (root + part + ".weight", rows(source, published, begin: begin, number: n))
            }
        }
        if name.hasSuffix(".linear_in.weight") {
            let root = String(name.dropLast("weight".count))
            return [(root + "gate.weight", try rows(source, published, part: 0, outOf: 2)),
                    (root + "up.weight", try rows(source, published, part: 1, outOf: 2))]
        }
        if name.hasPrefix("single_transformer_blocks."), name.hasSuffix(".attn.to_out.weight") {
            try matrix()
            let d = f[0], root = String(name.dropLast("weight".count))
            guard f[1] > d else { throw Numerics.Failure(description: "\(published) : unexpected shape \(f)") }
            return [(root + "attn.weight", try columns(source, published, begin: 0, number: d)),
                    (root + "mlp.weight", try columns(source, published, begin: d, number: f[1] - d))]
        }
        return [(name, asPublished(source, published))]
    }

    // ── Anima: ComfyUI (Cosmos-Predict2) → diffusers — the rules of the earlier Python forge ───────
    static let animaBlocks: [(String, String)] = [
        ("adaln_modulation_self_attn.1.", "norm1.linear_1."), ("adaln_modulation_self_attn.2.", "norm1.linear_2."),
        ("adaln_modulation_cross_attn.1.", "norm2.linear_1."), ("adaln_modulation_cross_attn.2.", "norm2.linear_2."),
        ("adaln_modulation_mlp.1.", "norm3.linear_1."), ("adaln_modulation_mlp.2.", "norm3.linear_2."),
        ("self_attn.q_proj.", "attn1.to_q."), ("self_attn.k_proj.", "attn1.to_k."),
        ("self_attn.v_proj.", "attn1.to_v."), ("self_attn.output_proj.", "attn1.to_out.0."),
        ("self_attn.q_norm.", "attn1.norm_q."), ("self_attn.k_norm.", "attn1.norm_k."),
        ("cross_attn.q_proj.", "attn2.to_q."), ("cross_attn.k_proj.", "attn2.to_k."),
        ("cross_attn.v_proj.", "attn2.to_v."), ("cross_attn.output_proj.", "attn2.to_out.0."),
        ("cross_attn.q_norm.", "attn2.norm_q."), ("cross_attn.k_norm.", "attn2.norm_k."),
        ("mlp.layer1.", "ff.net.0.proj."), ("mlp.layer2.", "ff.net.2."),
    ]
    static let animaTop: [String: String] = [
        "x_embedder.proj.1.weight": "patch_embed.proj.weight",
        "t_embedder.1.linear_1.weight": "time_embed.t_embedder.linear_1.weight",
        "t_embedder.1.linear_2.weight": "time_embed.t_embedder.linear_2.weight",
        "t_embedding_norm.weight": "time_embed.norm.weight",
        "final_layer.adaln_modulation.1.weight": "norm_out.linear_1.weight",
        "final_layer.adaln_modulation.2.weight": "norm_out.linear_2.weight",
        "final_layer.linear.weight": "proj_out.weight",
    ]

    static func anima(_ name: String) throws -> String {
        if let h = animaTop[name] { return h }
        if let x = name.firstMatch(#"^blocks\.(\d+)\.(.+)$"#) {
            for (pattern, target) in animaBlocks where x[2].hasPrefix(pattern) {
                return "transformer_blocks.\(x[1])." + target + x[2].dropFirst(pattern.count)
            }
            throw Numerics.Failure(description: "Anima block key without a rule: \(name)")
        }
        throw Numerics.Failure(description: "Anima key without a rule: \(name)")
    }
}

extension String {
    /// The groups of an anchored regular expression, or `nil`.
    func firstMatch(_ pattern: String) -> [String]? {
        guard let re = try? NSRegularExpression(pattern: pattern),
              let x = re.firstMatch(in: self, range: NSRange(startIndex..., in: self)) else { return nil }
        return (0..<x.numberOfRanges).map { i in Range(x.range(at: i), in: self).map { String(self[$0]) } ?? "" }
    }
}
