import Foundation

/// **The text encoder forge** — a published Qwen3 → the map that `TextEncoder` reads.
/// It replaces the earlier Python forge, whose three measured decisions it keeps:
///
///   1. **Keep only what is read.** Z-Image and Krea 2 take `hidden_states[-2]` (or taps up to the
///      input of layer 35): the output of layer 34 — layer 35, the final norm and `lm_head` are
///      not written. Anima reads `last_hidden_state` of a Qwen3-0.6B: all the layers and the
///      final norm (`final_norm: true`).
///   2. **`embed_tokens` is not transposed** — it is a lookup table, one row per token.
///   3. **`q_proj`, `k_proj`, `q_norm`, `k_norm` permuted per head** (`0, Dh/2, 1, Dh/2+1, …`):
///      `Ops.rope` pairs `(2p, 2p+1)`, `transformers` pairs `(j, j + Dh/2)`. The permutation makes
///      the interleaved rotation equal to `rotate_half`, with no second path at runtime.
///
/// **Never wider, never narrower than published**: each tensor goes through
/// `ForgeDiT.tensor`, so a bf16 encoder gives the bf16 map it always gave, byte for byte, and an
/// 8-bit one (Disty0's SDNQ int8 per row, unsloth's ComfyUI int8 convrot) stays 8-bit — bytes and
/// scales copied, transposed, the RoPE permutations done as row selections (each row keeps its scale;
/// a convrot rotates the *input*, which a row order leaves alone). `map_dtype` says "mixed" then.
///
/// ERNIE-Image reads the language model of a `Mistral3Model` (Ministral-3 3B): no QK-Norm, a yarn
/// RoPE (`TextEncoder.Config.yarn`), 24 layers kept out of 26.
///
/// Krea 2 reads a `Qwen3VLModel`: its text part has the shapes of Z-Image's Qwen3-4B, under
/// `language_model.`, next to a vision tower the text never goes through (dropped).
///
/// **Qwen-Image-2.1 reads a whole `Qwen3VLForConditionalGeneration`** (Qwen3-VL-8B): the
/// reference images go through its vision tower, so it is kept — and so is every layer of the
/// language model. What differs, and what a port by analogy would miss:
///
///   · **The output is the last layer, before the final RMSNorm** (`QwenImage21Pipeline` hooks
///     `norm` away: transformers 5 would return the normalized state). All 36 layers are written,
///     `norm` and `lm_head` are not (`lm_head` is untied: 622 M parameters nobody reads).
///   · **The vision tower** (27 blocks, `visual.`): its `qkv` stays fused, but its q and k thirds
///     are permuted per head (72) like the language model's — its RoPE is `rotate_half` too
///     (`apply_rotary_pos_emb_vision`, `[h, w, h, w]` angles): one rotation, `Ops.rope`'s, for the
///     whole encoder. `patch_embed` is a `Conv3d` whose stride equals its kernel: a `Linear`
///     `[3·2·16·16 = 1536 → 1152]` on the patch flattened as `(c, t, y, x)` — the order
///     `Qwen3VLVisionPatchEmbed.forward` views it in. `pos_embed` is a table (not transposed).
///   · **mRoPE** (`mrope_interleaved`, sections 24/20/20) assigns each frequency to an axis
///     (t, h, w); the per-head permutation keeps frequency `p` on the pair `(2p, 2p+1)`, so it
///     is the same map for text-only and image prompts.
package enum ForgeText {
    static let permuted = ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                           "self_attn.q_norm.weight", "self_attn.k_norm.weight"]
    /// The vision blocks' fused `qkv` (weight and bias): q and k thirds permuted per head.
    static let visionPermuted = ["attn.qkv.weight", "attn.qkv.bias"]

    /// What the forge keeps of a source, decided from the names and the config alone.
    package struct Plan {
        /// Map name → published name.
        package let kept: [String: String]
        /// Published names not written (the layers past the one read, `lm_head`, a vision tower).
        package let discarded: [String]
        /// The map's names, in execution order.
        package let order: [String]
        /// The config written in the header (`text_config` for a VL model).
        package let config: OrderedJSON
        package let vision: OrderedJSON?
        package let headDim: Int
        package let visionHead: Int?
        package let lastLine: Int
        package let finalNorm: Bool
    }

    package static func forge(folder: String, family: Family, to path: String,
                               progressHandler: ((Int, Int, String) -> Void)? = nil) throws -> MapWriter.Tally {
        try forge(source: TensorSource(folder: folder), published: OrderedJSON.read(folder + "/config.json"), family: family,
                  to: path, directory: (folder as NSString).lastPathComponent, progressHandler: progressHandler)
    }

    /// The plan of a source: which tensors, under which names, in which order.
    package static func plan(_ source: TensorCatalog, published: OrderedJSON, family: Family) throws -> Plan {
        var config = published
        if family == .krea2 || family == .ernie || family == .qwenImage21 {
            guard let t = config["text_config"] else { throw Numerics.Failure(description: "text_config absent") }
            config = t
        }
        guard let headDim = config["head_dim"]?.integer, let layers = config["num_hidden_layers"]?.integer else {
            throw Numerics.Failure(description: "config.json : head_dim or num_hidden_layers absent")
        }
        let vision = family == .qwenImage21 ? published["vision_config"] : nil
        if family == .qwenImage21 && vision == nil { throw Numerics.Failure(description: "vision_config absent") }
        let visionHead = try vision.map { v -> Int in
            guard let d = v["hidden_size"]?.integer, let h = v["num_heads"]?.integer, d % h == 0 else {
                throw Numerics.Failure(description: "vision_config : hidden_size or num_heads absent")
            }
            return d / h
        }
        let finalNorm = family == .anima
        // `hidden_states[-2]`: the output of the second-to-last layer (34 of 36 for Qwen3-4B,
        // 24 of 26 for ERNIE's Ministral-3 — checked against the oracle's text encoder). Qwen-Image-2.1:
        // the last layer, without the final norm.
        let lastLine = finalNorm || family == .qwenImage21 ? layers - 1 : family == .ernie ? layers - 2 : 34

        func wantedName(_ name: String) -> String? {
            let without: String
            if family == .qwenImage21 {
                if name.hasPrefix("model.visual.") { return String(name.dropFirst("model.".count)) }
                guard name.hasPrefix("model.language_model.") else { return nil }
                without = String(name.dropFirst("model.language_model.".count))
            } else if family == .krea2 {
                guard name.hasPrefix("language_model.") else { return nil }
                without = String(name.dropFirst("language_model.".count))
            } else if family == .ernie {
                // `Mistral3Model` published the old way: `language_model.model.*`, and a vision
                // tower, a projector, that a text never goes through.
                guard name.hasPrefix("language_model.model.") else { return nil }
                without = String(name.dropFirst("language_model.model.".count))
            } else if name.hasPrefix("model.") {
                without = String(name.dropFirst(6))
            } else if finalNorm && !name.hasPrefix("lm_head.") {
                without = name
            } else { return nil }
            if without == "norm.weight" { return finalNorm ? without : nil }
            if without.hasPrefix("layers.") { return Int(without.split(separator: ".")[1])! > lastLine ? nil : without }
            return without == "embed_tokens.weight" ? without : nil
        }
        var kept: [String: String] = [:], discarded: [String] = []
        for name in source.names {
            if let v = wantedName(name) { kept[v] = name } else { discarded.append(name) }
        }
        guard !kept.isEmpty else { throw Numerics.Failure(description: "no tensor kept — unexpected naming") }
        let ranked: [(key: (Int, Int, Int), name: String)] = kept.keys.map { (rank($0, vision: vision), $0) }
        let order = ranked.sorted { a, b -> Bool in
            a.key != b.key ? a.key < b.key : a.name < b.name
        }.map(\.name)
        return Plan(kept: kept, discarded: discarded, order: order, config: config, vision: vision, headDim: headDim,
                    visionHead: visionHead, lastLine: lastLine, finalNorm: finalNorm)
    }

    /// **One tensor of the map**, by `ForgeDiT.tensor`'s rule — never wider, never narrower than
    /// published: a bf16 weight stays bf16, an 8-bit one 8-bit (bytes and scales copied, rows moved,
    /// transposed), an fp16 one fp16, an fp32 one fp32 unless bf16 holds it exactly. The per-head
    /// permutations are row orders, so an 8-bit weight's rows move with their scales — a per-row scale
    /// and a convrot (a rotation of the *input*) both follow its rows.
    package static func tensor(_ name: String, published: String, source: TensorCatalog, plan: Plan) throws
        -> (MapWriter.Tensor, ForgeDiT.Outcome) {
        guard var shape = source.shape(published), let dtype = source.dtype(published) else {
            throw Numerics.Failure(description: "\(published) absent from the source")
        }
        // `Conv3d` with stride = kernel → the `Linear` it is, input flattened `(c, t, y, x)`.
        if name == "visual.patch_embed.proj.weight", shape.count == 5 {
            shape = [shape[0], shape.dropFirst().reduce(1, *)]
        }
        let table = name == "embed_tokens.weight" || name == "visual.pos_embed.weight"
        let rope = !name.hasPrefix("visual.") && permuted.contains { name.hasSuffix($0) }
        let visionRope = name.hasPrefix("visual.blocks.") && visionPermuted.contains { name.hasSuffix($0) }
        let order = rope ? ForgeDiT.interleaveOrder(rows: shape[0], head: plan.headDim)
            : visionRope ? plan.visionHead.map { qkOrder(rows: shape[0], head: $0) } : nil
        let p = Provenance(shape: shape, dtypeSource: dtype) { try source.read(published) }.quantized(source, published) { $0 }
        return try ForgeDiT.tensor(name, p, .init(transposed: shape.count == 2 && !table, named: .bfloat16, rowOrder: order))
    }

    /// Forges the map of a published encoder: its tensors, its config (`config.json`, whole).
    /// `directory`: the source's folder, as the header names it.
    package static func forge(source: TensorSource, published: OrderedJSON, family: Family, to path: String,
                               directory: String, progressHandler: ((Int, Int, String) -> Void)? = nil) throws -> MapWriter.Tally {
        let plan = try plan(source, published: published, family: family)
        let (config, vision, lastLine, finalNorm, order) = (plan.config, plan.vision, plan.lastLine, plan.finalNorm, plan.order)
        var tensors: [MapWriter.Tensor] = []
        var bfloat16Only = true
        for name in order {
            let (t, outcome) = try tensor(name, published: plan.kept[name]!, source: source, plan: plan)
            if outcome != .asNamed || t.dtype != .bfloat16 { bfloat16Only = false }
            tensors.append(t)
        }
        let parameters = tensors.reduce(0) { $0 + $1.shape.reduce(1, *) }
        let kind = family == .ernie ? "mistral3-text-encoder" : family == .krea2 ? "qwen3-vl-text-encoder"
            : family == .qwenImage21 ? "qwen3-vl-encoder" : finalNorm ? "qwen3-text-encoder-last-hidden" : "qwen3-4b-text-encoder"
        let files = source.files.map { ($0.path as NSString).lastPathComponent }
        return try MapWriter.write(to: path, tensors: tensors, header: { _ in
            var h: [OrderedJSON.Pair] = [
                .init("format", 1), .init("kind", .string(kind)), .init("page", .integer(MapWriter.page)),
                // A bf16 map keeps the value it always had (compared byte for byte); one that holds
                // 8-bit, fp16 or fp32 tensors says "mixed", as a DiT's does.
                .init("map_dtype", .string(bfloat16Only ? "bfloat16" : "mixed")), .init("linear_weights_transposed", true),
                .init("config", config.withoutPrivateKeys), .init("last_layer", .integer(lastLine)),
                .init("final_norm", .boolean(finalNorm)), .init("rope_interleaved", true),
                .init("rope_permuted", .list(permuted.map { .string($0) })),
            ]
            if let vision {
                h += [.init("vision_config", vision.withoutPrivateKeys),
                      .init("vision_rope_permuted", .list(visionPermuted.map { .string("q, k thirds of " + $0) })),
                      .init("vision_patch_embed_as_linear", "(c, t, y, x)"),
                      .init("tokens", .object(["image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"]
                          .compactMap { k in published[k].map { .init(k, $0) } }))]
            }
            h += [
                .init("dropped", .list(plan.discarded.sorted().map { .string($0) })),
                .init("source", .object([.init("directory", .string(directory)),
                                        .init("shards", .list(files.map { .string($0) })),
                                        .init("tensor_count", .integer(order.count)), .init("parameters", .integer(parameters))])),
                .init("order", .list(order.map { .string($0) })),
            ]
            return h
        }, progressHandler: progressHandler)
    }

    /// A fused `[q | k | v]` (axis 0, `rows` = 3 × width): q and k permuted per head like
    /// `ForgeDiT.interleave`, v untouched.
    static func interleaveQK(_ v: [Float], rows: Int, columns: Int, head: Int) -> [Float] {
        ForgeDiT.permutingRows(v, qkOrder(rows: rows, head: head), columns: columns)
    }

    /// The source row of each row of `interleaveQK`.
    static func qkOrder(rows: Int, head: Int) -> [Int] {
        let third = rows / 3, within = ForgeDiT.interleaveOrder(rows: third, head: head)
        return within + within.map { third + $0 } + Array((2 * third)..<rows)
    }

    /// The execution order: the vision tower first (the references are encoded before the text
    /// that holds them), each deepstack merger right after the block it taps, the final merger;
    /// then `embed_tokens`, the layers, the final norm. Ties broken by name: the same file at
    /// every forge.
    static func rank(_ n: String, vision: OrderedJSON?) -> (Int, Int, Int) {
        func index(_ k: Int) -> Int { Int(n.split(separator: ".")[k]) ?? 0 }
        if n.hasPrefix("visual.") {
            if n.hasPrefix("visual.blocks.") { return (0, index(2), 0) }
            if n.hasPrefix("visual.deepstack_merger_list.") {
                let taps = vision?["deepstack_visual_indexes"]?.elements?.compactMap(\.integer) ?? []
                let k = index(2)
                return (0, k < taps.count ? taps[k] : Int.max - 1, 1)
            }
            if n.hasPrefix("visual.merger.") { return (0, Int.max, 0) }
            return (0, -1, 0)                                          // patch_embed, pos_embed
        }
        if n.hasPrefix("layers.") { return (2, index(1), 0) }
        return n == "norm.weight" ? (3, 0, 0) : (1, 0, 0)
    }
}
