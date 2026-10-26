import Foundation

/// **An architecture family**: what Siliconed knows how to render. An imported model (a Civitai
/// checkpoint) is a member of one — same architecture, other weights — and borrows from its family
/// everything the user does not bring: tokenizers, text encoder, VAE.
package enum Family: String, CaseIterable, Sendable, Codable {
    case zImage = "z-image"
    case anima
    case krea2
    /// Qwen-Image-2.1-Turbo (Qwen): single-stream DiT, Qwen3-VL-8B encoder that sees the reference
    /// images, 64-channel VAE; generation and editing by instruction, 8 steps.
    case qwenImage21 = "qwen-image-2.1"
    /// MOSS-Image-SFT (OpenMOSS-Team) under the community's DMD LoRA (lodestones) — the 8-step recipe:
    /// single-stream DiT of 5.08 B, Qwen3.5-9B hybrid encoder, FLUX.2 VAE. Standard only.
    case mossImage = "moss-image"

    /// The families offered to the user — `install all`, the app's library sheet. All of them
    /// since Qwen-Image-2.1 renders swap-free at its worst case; a family that
    /// cannot render yet is filtered out here, so that nobody downloads what cannot render.
    package static let offered: [Family] = allCases

    package var name: String {
        switch self {
        case .zImage: return "Z-Image"
        case .anima: return "Anima"
        case .krea2: return "Krea 2"
        case .qwenImage21: return "Qwen-Image-2.1"
        case .mossImage: return "MOSS-Image"
        }
    }

    /// The `kind` of the DiT map — what the engine requires in order to read it.
    package var ditKind: String {
        switch self {
        case .zImage: return "z-image-turbo-dit"
        case .anima: return "anima-turbo-dit"
        case .krea2: return "krea2-turbo-dit"
        // The official Turbo's checkpoint. Another kind than the base model's (`supersededQwenImage21Kind`),
        // which rendered under a third-party turbo LoRA: a map of it is not taken for this one.
        case .qwenImage21: return "qwen-image-2.1-turbo-dit"
        case .mossImage: return MossImageDiT.kind
        }
    }

    /// The base DiT map, in `store/`.
    package var ditMap: String {
        switch self {
        case .zImage: return "z-image-turbo-dit.v1.silicon"
        case .anima: return "anima-turbo-dit.v0.silicon"
        case .krea2: return "krea2-turbo-dit.v0.silicon"
        case .qwenImage21: return "qwen-image-2.1-turbo-dit.v0.silicon"
        case .mossImage: return "moss-image-sft-dit.v0.silicon"
        }
    }

    /// The text encoder map, in `store/`.
    package var encoderMap: String {
        switch self {
        case .zImage: return "qwen3-4b-encoder.v0.silicon"
        case .anima: return "qwen3-0.6b-encoder.v0.silicon"
        case .krea2: return "qwen3-vl-4b-encoder.v0.silicon"
        // The whole `Qwen3VLModel`: language model **and** vision tower — the references go through it.
        case .qwenImage21: return "qwen3-vl-8b-encoder.v0.silicon"
        case .mossImage: return "qwen3.5-9b-encoder.v0.silicon"
        }
    }

    /// **The versions a family installs in**: every family has its Standard, a Compact and a Light — but
    /// MOSS-Image, whose publisher (and no one else, as of the pinned revisions) offers the Standard alone: the
    /// community's 8-bit and GGUF copies of the DiT are not a version of *this* recipe (the DMD LoRA is
    /// applied to bf16 weights, and an 8-bit DiT would round the distillation away).
    package var variants: [Variant] { self == .mossImage ? [.standard] : [.standard, .compact, .light] }

    /// **The version preselected before an installation** (the welcome cards, « Install… »): the
    /// Compact, the one we recommend — as fast as the Standard on the Macs measured, half its disk,
    /// the Neural Engine on by default — except on a Mac of 8 GB or less, where the Light is
    /// preselected. Preselected, never imposed: the three versions are shown, and the user picks;
    /// the publisher's weights stay one click away.
    package func preselectedVariant(physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> Variant {
        if physicalMemory <= Variant.lightMemory && variants.contains(.light) { return .light }
        return variants.contains(.compact) ? .compact : .standard
    }

    /// The DiT map of a version, in `store/`. The Standard's is `ditMap`; the others have their own
    /// name, so that two are never taken for each other (one is removed when another installs).
    package func ditMap(_ v: Variant) -> String {
        switch (self, v) {
        case (.zImage, .compact): return "z-image-turbo-dit.compact.v0.silicon"
        case (.qwenImage21, .compact): return "qwen-image-2.1-turbo-dit.compact.v0.silicon"
        case (.zImage, .light): return "z-image-turbo-dit.light.v0.silicon"
        case (.qwenImage21, .light): return "qwen-image-2.1-turbo-dit.light.v0.silicon"
        case (.anima, .compact): return "anima-turbo-dit.compact.v0.silicon"
        case (.anima, .light): return "anima-turbo-dit.light.v0.silicon"
        case (.krea2, .compact): return "krea2-turbo-dit.compact.v0.silicon"
        case (.krea2, .light): return "krea2-turbo-dit.light.v0.silicon"
        default: return ditMap
        }
    }

    /// The text encoder map of a version, in `store/`. The Standard reads the publisher's encoder; the
    /// Compact's is a third party's 8-bit one, under a name of its own. **The
    /// Light reads the Compact's encoder, the same map**: no text encoder is published under 8 bits
    /// in a form the forge reads, and quantifying one ourselves is excluded (never below the publisher's
    /// weights unless the user chooses so). Switching between
    /// Compact and Light therefore keeps the encoder (`Library.releasable`). **Anima's three versions
    /// read the Standard's encoder** (Qwen3-0.6B, 1.2 GB in bf16): no 8-bit one is published by the
    /// third party of its DiT, and what an 8-bit one would save (~0.6 GB) is not worth a second map.
    package func encoderMap(_ v: Variant) -> String {
        switch (self, v) {
        case (.zImage, .compact), (.zImage, .light): return "qwen3-4b-encoder.compact.v0.silicon"
        case (.qwenImage21, .compact), (.qwenImage21, .light): return "qwen3-vl-8b-encoder.compact.v0.silicon"
        case (.krea2, .compact), (.krea2, .light): return "qwen3-vl-4b-encoder.compact.v0.silicon"
        default: return encoderMap
        }
    }

    /// The `kind` of the maps forged from Qwen-Image-2.1's **base** model, which rendered under a
    /// third-party turbo LoRA until the official Turbo replaced both: refused by the engine with a
    /// reason (reinstall), never read as the Turbo.
    package static let supersededQwenImage21Kind = "qwen-image-2.1-dit"
    /// Its DiT maps, in `store/` (`Installer.supersededFiles`): the Standard's, and the Compact's and
    /// the Light's — third-party 8-bit and GGUF DiTs of the base model, under the same LoRA.
    package static let supersededQwenImage21Maps = ["qwen-image-2.1-dit.v0.silicon", "qwen-image-2.1-dit.compact.v0.silicon",
                                                    "qwen-image-2.1-dit.light.v0.silicon"]

    package init?(ditKind: String) {
        guard let f = Family.allCases.first(where: { $0.ditKind == ditKind }) else { return nil }
        self = f
    }
}

/// **Which weights a family is installed with**. A single version of a family is
/// installed at a time: installing the other replaces it, and gives its space back.
package enum Variant: String, CaseIterable, Sendable, Codable {
    /// The publisher's weights, as published — the default (never below the
    /// publisher's weights unless the user chooses so).
    case standard
    /// 8-bit weights published by a third party, **DiT and text encoder**, kept 8-bit in the maps:
    /// about 40 % smaller on disk, an image slightly different from the Standard's, a step a little
    /// slower (the weights are widened at each evaluation). Tokenizers and VAE are the Standard's.
    /// Anima's Compact is its DiT alone in GGUF Q8_0 (vanes430's): its encoder stays the Standard's.
    /// Krea 2's are Comfy-Org's int8 convrot, DiT and encoder.
    case compact
    /// **The DiT in 4 to 6 bits as a third party publishes it** (a GGUF Q4_K_M — unsloth's, vanes430's
    /// for Anima, vantagewithai's for Krea 2 — its blocks kept whole), the Compact's text encoder, the Standard's tokenizers and VAE: the smallest
    /// on disk and the least read from it at each step — another set of weights, an image that may
    /// compose differently. Preselected on a Mac of 8 GB (`Family.preselectedVariant`).
    case light

    /// **The Neural Engine's default** (`Request.neuralEngine`): off on the Standard, whose promise is the
    /// publisher's weights computed in fp32, the image the checks compare with the reference; on on the 8-
    /// and 4-bit versions, whose image already differs from the Standard's, for the energy its fp16 products
    /// save (a fifth of the machine's, M1 Pro, 1024², 23/10/2026).
    package var neuralEngineByDefault: Bool { self != .standard }

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
    /// A 16-bit tensor's published bits, never widened (`TensorSource.read16`); `nil`: not available.
    func read16(_ name: String) throws -> [UInt16]?
}

extension TensorCatalog {
    package func quantization(_ name: String) -> QuantizedKind? { nil }
    package func readQuantized(_ name: String) throws -> QuantizedTensor {
        throw Numerics.Failure(description: "\(name) is not 8-bit")
    }
    package func read16(_ name: String) throws -> [UInt16]? { nil }
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
    /// The same selection on the published 16-bit bits (`TensorCatalog.read16`), when it only moves
    /// values — what lets the forge write a bf16 or fp16 weight without going through fp32. `nil`
    /// (or a `nil` result): the fp32 path.
    package var read16: (() throws -> [UInt16]?)? = nil

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

    /// The same provenance, with its 16-bit form (`read16`).
    func bits(_ read16: @escaping () throws -> [UInt16]?) -> Provenance {
        var p = self
        p.read16 = read16
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
            .bits { try source.read16(name).map { Array($0[(part * n * columns)..<((part + 1) * n * columns)]) } }
    }

    static func asPublished(_ source: TensorCatalog, _ name: String, shape: [Int]? = nil) -> Provenance {
        Provenance(shape: shape ?? source.shape(name)!, dtypeSource: source.dtype(name)!) { try source.read(name) }
            .quantized(source, name) { $0 }
            .bits { try source.read16(name) }
    }

    /// **Recognizes a DiT's family from its names**, prefixes removed. `nil`: none.
    package static func family(fromNames names: [String]) -> Family? {
        let n = Set(names.map(withoutPrefix))
        func a(_ pattern: String) -> Bool { n.contains { $0.range(of: pattern, options: .regularExpression) != nil } }
        if a(#"^transformer_blocks\.\d+\.img_mlp\.(gate_layer|gate_up)\."#) && a(#"^txt_in\.text_norm\."#) { return .qwenImage21 }
        if a(#"^(layers|noise_refiner|context_refiner)\.\d+\.attention\.(qkv|to_q)\."#) && a(#"^cap_embedder\."#) { return .zImage }
        if a(#"^(blocks\.\d+\.attn\.wq|txtfusion\.|transformer_blocks\.\d+\.attn\.to_gate)"#) { return .krea2 }
        // MOSS-Image is not recognized: a checkpoint of it is not imported (the family's DiT is forged by `MossImageForge`).
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
                if kreaEasterEgg.contains(raw) { r.ignored.append(published); continue }
                guard let (name, reshaped) = krea2(raw, shape: source.shape(published)!) else {
                    throw Numerics.Failure(description: "\(published) : no Krea 2 renaming rule")
                }
                try place(name, asPublished(source, published, shape: reshaped))
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
        case .mossImage:
            // The names as published; `rope.bands` is derived by the code, never a weight (`MossImageForge`).
            r.naming = "MOSS-Image"
            for (published, name) in keys {
                if MossImageForge.notWeights.contains(name) { r.ignored.append(published); continue }
                try place(name, asPublished(source, published))
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
    /// **Two tensors outside the model, Krea's easter egg**, which the GGUFs of Krea 2 Turbo carry
    /// (`egg_w`, `egg_h`, `egg_format: chw_m1p1_flat` in their metadata) and the published
    /// checkpoints do not: `last.down` `[6144, 6144]` is a picture in [−1, 1] (a hiring note),
    /// `last.up` is all zeros. Neither diffusers' `Krea2Transformer2DModel` nor the oracle has them:
    /// set aside like an embedded VAE, never placed — the strict recipe would refuse them otherwise.
    static let kreaEasterEgg: Set<String> = ["last.down.weight", "last.up.weight"]
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
