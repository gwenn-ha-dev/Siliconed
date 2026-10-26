import Foundation

/// **The Qwen3.5-9B forge** — MOSS-Image's text encoder, a published `Qwen3_5Model` → the map that
/// `Qwen35Encoder` reads. Text only: the vision tower (`visual.`) and the multi-token-prediction head
/// are not written. All 32 layers and the final norm are (the output is `last_hidden_state`).
///
/// What differs from `ForgeText` (Qwen3), and what a port by analogy would miss:
///
///   · **Every RMSNorm is zero-centered** (`Qwen3_5RMSNorm`: `x̂ · (1 + w)`) — the layer norms, the
///     QK-norms, the final norm. The `+ 1` is folded here (`Rule.foldsOne`), so the map holds fp32 and
///     the engine multiplies by the weight like any other. The ONE norm that is not zero-centered is the
///     delta rule's gated norm (`linear_attn.norm`, `Qwen3_5RMSNormGated`: `w · x̂ · silu(z)`): kept as is.
///   · **No RoPE permutation.** The rotation is partial (the first quarter of the head) and the engine
///     does `rotate_half` on it directly; `q_proj` is laid out per head as `[q | gate]` and stays so.
///   · **`conv1d.weight`** is the delta layer's depthwise causal convolution, `[8192, 1, 4]`: written
///     `[8192, 4]` (the middle axis is the group size, 1). `A_log` and `dt_bias` are the decay's
///     parameters; they stay fp32 (`mamba_ssm_dtype`), as do the small tensors a round to bf16 would touch.
///   · The Linear weights are laid out `[input, output]` like every encoder map (`in_proj_b`/`in_proj_a`
///     too: 4096 × 32).
package enum ForgeQwen35 {
    /// The tensors of one linear-attention (Gated DeltaNet) layer, in the order `Qwen35Encoder` widens them.
    package static let deltaOrder = [
        "input_layernorm.weight", "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight", "linear_attn.conv1d.weight",
        "linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.norm.weight", "linear_attn.out_proj.weight",
        "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    ]
    /// The tensors of one full-attention layer, in the order `Qwen35Encoder` widens them.
    package static let fullOrder = [
        "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.o_proj.weight",
        "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    ]

    /// The prefixes a checkpoint gives the language model: `Qwen3_5Model` (`language_model.`) or a
    /// `ForConditionalGeneration` (`model.language_model.`).
    static let prefixes = ["language_model.", "model.language_model."]

    package struct Plan {
        /// Map name → published name.
        package let kept: [String: String]
        package let discarded: [String]
        /// The map's names, in execution order.
        package let order: [String]
        package let config: OrderedJSON
        package let layers: Int
    }

    package static func plan(_ source: TensorCatalog, published: OrderedJSON) throws -> Plan {
        guard let config = published["text_config"] ?? (published["num_hidden_layers"] != nil ? published : nil),
              let layers = config["num_hidden_layers"]?.integer else {
            throw Numerics.Failure(description: "config.json: text_config or num_hidden_layers absent")
        }
        var kept: [String: String] = [:], discarded: [String] = []
        for name in source.names {
            guard let prefix = prefixes.first(where: name.hasPrefix) else { discarded.append(name); continue }
            let without = String(name.dropFirst(prefix.count))
            if without == "embed_tokens.weight" || without == "norm.weight" { kept[without] = name; continue }
            if without.hasPrefix("layers."), let layer = Int(without.split(separator: ".")[1]), layer < layers {
                kept[without] = name
            } else { discarded.append(name) }
        }
        guard kept["embed_tokens.weight"] != nil, kept["norm.weight"] != nil else {
            throw Numerics.Failure(description: "embed_tokens or norm absent: unexpected naming")
        }
        // Every layer must hold exactly the tensors of its type, none more: a missing one is a broken
        // download, an extra one a port that does not know the layer.
        let types = config["layer_types"]?.elements?.compactMap(\.text) ?? []
        guard types.count == layers else { throw Numerics.Failure(description: "layer_types: \(types.count) for \(layers) layers") }
        var order = ["embed_tokens.weight"]
        for l in 0..<layers {
            let suffixes = types[l] == "full_attention" ? fullOrder : deltaOrder
            let names = suffixes.map { "layers.\(l).\($0)" }
            let got = Set(kept.keys.filter { $0.hasPrefix("layers.\(l).") })
            guard got == Set(names) else {
                throw Numerics.Failure(description: "layer \(l) (\(types[l])): tensors \(got.symmetricDifference(names).sorted())")
            }
            order += names
        }
        order.append("norm.weight")
        return Plan(kept: kept, discarded: discarded, order: order, config: config, layers: layers)
    }

    /// What the forge decides for a tensor from its name alone.
    static func rule(_ name: String, shape: [Int]) -> ForgeDiT.Rule {
        let table = name == "embed_tokens.weight"
        let zeroCentered = name == "norm.weight" || ["input_layernorm.weight", "post_attention_layernorm.weight",
                                                     "q_norm.weight", "k_norm.weight"].contains { name.hasSuffix($0) }
        let small = name.hasSuffix("linear_attn.norm.weight") || name.hasSuffix("linear_attn.A_log")
            || name.hasSuffix("linear_attn.dt_bias") || name.hasSuffix("conv1d.weight")
        // The conv weight is 2-D by now but is no Linear: `[channels, kernel]` stays as it is.
        return ForgeDiT.Rule(transposed: shape.count == 2 && !table && !name.hasSuffix("conv1d.weight"), named: small || zeroCentered ? .float32 : .bfloat16,
                             rowOrder: nil, foldsOne: zeroCentered, measuresAbsMax: false)
    }

    package static func tensor(_ name: String, published: String, source: TensorCatalog) throws
        -> (MapWriter.Tensor, ForgeDiT.Outcome) {
        guard var shape = source.shape(published), let dtype = source.dtype(published) else {
            throw Numerics.Failure(description: "\(published) absent from the source")
        }
        // A depthwise `Conv1d` weight `[channels, 1, kernel]`: the group axis carries no values.
        if name.hasSuffix("conv1d.weight"), shape.count == 3, shape[1] == 1 { shape = [shape[0], shape[2]] }
        let p = Provenance(shape: shape, dtypeSource: dtype) { try source.read(published) }
            .quantized(source, published) { $0 }
            .bits { try source.read16(published) }
        return try ForgeDiT.tensor(name, p, rule(name, shape: shape))
    }

    package static func forge(folder: String, to path: String,
                              progressHandler: ((Int, Int, String) -> Void)? = nil) throws -> MapWriter.Tally {
        try forge(source: TensorSource(folder: folder), published: OrderedJSON.read(folder + "/config.json"), to: path,
                  directory: (folder as NSString).lastPathComponent, progressHandler: progressHandler)
    }

    package static func forge(source: TensorSource, published: OrderedJSON, to path: String, directory: String,
                              progressHandler: ((Int, Int, String) -> Void)? = nil) throws -> MapWriter.Tally {
        let plan = try plan(source, published: published)
        var tensors: [MapWriter.Tensor] = []
        var bfloat16Only = true
        for name in plan.order {
            let (t, outcome) = try tensor(name, published: plan.kept[name]!, source: source)
            if outcome != .asNamed || t.dtype != .bfloat16 { bfloat16Only = false }
            tensors.append(t)
        }
        let parameters = tensors.reduce(0) { $0 + $1.shape.reduce(1, *) }
        let files = source.files.map { ($0.path as NSString).lastPathComponent }
        return try MapWriter.write(to: path, tensors: tensors, header: { _ in
            [
                .init("format", 1), .init("kind", .string("qwen3.5-text-encoder")), .init("page", .integer(MapWriter.page)),
                // Norms and the delta layer's small tensors are fp32: the map is "mixed" by design.
                .init("map_dtype", .string(bfloat16Only ? "bfloat16" : "mixed")), .init("linear_weights_transposed", true),
                .init("config", plan.config.withoutPrivateKeys), .init("last_layer", .integer(plan.layers - 1)),
                .init("final_norm", true), .init("rope_interleaved", false),
                .init("dropped", .list(plan.discarded.sorted().map { .string($0) })),
                .init("source", .object([.init("directory", .string(directory)),
                                        .init("shards", .list(files.map { .string($0) })),
                                        .init("tensor_count", .integer(plan.order.count)), .init("parameters", .integer(parameters))])),
                .init("order", .list(plan.order.map { .string($0) })),
            ]
        }, progressHandler: progressHandler)
    }
}
