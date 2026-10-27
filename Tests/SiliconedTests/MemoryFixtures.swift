import XCTest
@testable import Siliconed

/// **Qwen-Image-2.1's maps as the formula reads them, written in memory** — the published configs and
/// a layer table with the published shapes and order, laid on 16 KB pages as the forge lays them (a
/// bf16 layer of the DiT runs 436 240 384 bytes, of the encoder 385 941 504: the installed maps'
/// headers, read on 12/10/2026), and the tokenizer's tables at the published file's counts (151 643
/// vocabulary entries in 3 989 982 bytes, 151 387 merges in 7 426 578: 28 213 248 bytes mapped). No
/// byte of weights, no file: the same figures on every Mac, the CI's included.
enum MemoryFixtures {
    // MARK: - The maps, built once per argument

    /// A family's maps cost tens of milliseconds to lay out, and the tests ask for the same ones hundreds of
    /// times: each is built once per set of arguments. Values, so a test that edits its copy edits nothing else.
    nonisolated(unsafe) private static var built: [String: Any] = [:]
    private static let lock = NSLock()

    private static func once<T>(_ key: String, _ make: () -> T) -> T {
        lock.lock(); defer { lock.unlock() }
        if let known = built[key] as? T { return known }
        let made = make()
        built[key] = made
        return made
    }

    static func zImageMaps(bytesPerValue: Int = 2, rotated: Bool = false) -> MemoryNeed.ZImageMaps {
        once("z-image \(bytesPerValue) \(rotated)") { buildZImageMaps(bytesPerValue: bytesPerValue, rotated: rotated) }
    }

    static func animaMaps(ditBytesPerValue: Int = 2, encoderBytesPerValue: Int = 2, rotated: Bool = false) -> MemoryNeed.AnimaMaps {
        once("anima \(ditBytesPerValue) \(encoderBytesPerValue) \(rotated)") {
            buildAnimaMaps(ditBytesPerValue: ditBytesPerValue, encoderBytesPerValue: encoderBytesPerValue, rotated: rotated)
        }
    }

    static func krea2Maps(ditBytesPerValue: Int = 2, encoderBytesPerValue: Int = 2, rotated: Bool = false) -> MemoryNeed.Krea2Maps {
        once("krea2 \(ditBytesPerValue) \(encoderBytesPerValue) \(rotated)") {
            buildKrea2Maps(ditBytesPerValue: ditBytesPerValue, encoderBytesPerValue: encoderBytesPerValue, rotated: rotated)
        }
    }

    static func qwenMaps(bytesPerValue: Int = 2) -> QwenImage21Maps {
        once("qwen-image-2.1 \(bytesPerValue)") { buildQwenMaps(bytesPerValue: bytesPerValue) }
    }

    // MARK: - Laying them out

    /// A layer table: `layers` blocks of `tensors` under `prefix(i)`, each tensor on its own pages.
    static func table(layers: Int, prefix: (Int) -> String, tensors: [(String, [Int])], bytesPerValue: Int,
                      page: Int = 16384) -> [String: Any] {
        var table: [String: Any] = [:], order: [String] = [], offset = page
        for layer in 0..<layers {
            for (suffix, shape) in tensors {
                let bytes = shape.reduce(1, *) * bytesPerValue + (bytesPerValue == 1 && shape.count == 2 ? 4 * shape[1] : 0)
                table[prefix(layer) + suffix] = ["offset": offset, "bytes": bytes, "shape": shape]
                order.append(prefix(layer) + suffix)
                offset += (bytes + page - 1) / page * page
            }
        }
        return ["page": page, "tensors": table, "order": order]
    }

    /// A map whose 2-D weights carry Comfy's int8 `convrot` (group 256), as an imported checkpoint keeps it.
    static func convrot(_ header: [String: Any]) -> [String: Any] {
        var h = header, t = header["tensors"] as! [String: [String: Any]]
        for (name, entry) in t where (entry["shape"] as? [Int])?.count == 2 { t[name]?["rotation"] = ["kind": "convrot", "group": 256] }
        h["tensors"] = t
        return h
    }

    /// **Z-Image's maps**: the DiT's 30 layers and the encoder's 35 with the published shapes and
    /// order, the published configs — `bytesPerValue` 2 (bf16) or 1 (an 8-bit map, plus a float scale
    /// per output column).
    private static func buildZImageMaps(bytesPerValue: Int, rotated: Bool) -> MemoryNeed.ZImageMaps {
        let (d, h) = (3840, 10240)
        let layer: [(String, [Int])] = [
            ("adaLN_modulation.0.bias", [4 * d]), ("adaLN_modulation.0.weight", [256, 4 * d]),
            ("attention.norm_k.weight", [128]), ("attention.norm_q.weight", [128]),
            ("attention.to_k.weight", [d, d]), ("attention.to_out.0.weight", [d, d]),
            ("attention.to_q.weight", [d, d]), ("attention.to_v.weight", [d, d]),
            ("attention_norm1.weight", [d]), ("attention_norm2.weight", [d]),
            ("feed_forward.w1.weight", [d, h]), ("feed_forward.w2.weight", [h, d]), ("feed_forward.w3.weight", [d, h]),
            ("ffn_norm1.weight", [d]), ("ffn_norm2.weight", [d]),
        ]
        var dit = table(layers: 30, prefix: DiT.layerPrefix, tensors: layer, bytesPerValue: bytesPerValue)
        dit["config"] = [String: Any]()   // the published defaults (`DiT.Config`)
        if rotated { dit = convrot(dit) }
        let (e, i) = (2560, 9728)
        let encoderLayer: [(String, [Int])] = [
            ("input_layernorm.weight", [e]), ("mlp.down_proj.weight", [i, e]), ("mlp.gate_proj.weight", [e, i]),
            ("mlp.up_proj.weight", [e, i]), ("post_attention_layernorm.weight", [e]), ("self_attn.k_norm.weight", [128]),
            ("self_attn.k_proj.weight", [e, 1024]), ("self_attn.o_proj.weight", [4096, e]),
            ("self_attn.q_norm.weight", [128]), ("self_attn.q_proj.weight", [e, 4096]), ("self_attn.v_proj.weight", [e, 1024]),
        ]
        var encoder = table(layers: 35, prefix: { "layers.\($0)." }, tensors: encoderLayer, bytesPerValue: bytesPerValue)
        encoder["last_layer"] = 34
        encoder["config"] = ["hidden_size": e, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
                             "intermediate_size": i]
        return MemoryNeed.ZImageMaps(dit: dit, encoder: encoder)
    }

    /// A Z-Image request on the fixture's maps, `threads` P cores, no process base.
    static func zImageRequest(_ width: Int, _ height: Int, tokens: Int = 50, threads: Int = 6) -> MemoryNeed.Request {
        var request = MemoryNeed.Request(card: .zImage, width: width, height: height, promptTokens: tokens)
        request.threads = threads
        request.zImageMaps = zImageMaps()
        return request
    }

    /// **Anima's maps**: the DiT's 28 blocks with the published shapes (Cosmos-Predict2-2B, `d` 2048,
    /// `h` 8192, the published config), and Qwen3-0.6B's 28 layers. `ditBytesPerValue` is the DiT's
    /// type — 2 (Standard, bf16), 1 (Compact, Q8_0), 0 for a 4-bit table (Light, Q4_K_M: half a byte a
    /// value, laid out as a quarter of bf16's) — and the formula must not read it; `encoderBytesPerValue`
    /// the encoder's, which it reads (its staging buffers).
    private static func buildAnimaMaps(ditBytesPerValue: Int, encoderBytesPerValue: Int, rotated: Bool) -> MemoryNeed.AnimaMaps {
        let (d, h, a) = (2048, 8192, 256)
        let block: [(String, [Int])] = [
            ("attn1.norm_k.weight", [128]), ("attn1.norm_q.weight", [128]), ("attn1.to_k.weight", [d, d]),
            ("attn1.to_out.0.weight", [d, d]), ("attn1.to_q.weight", [d, d]), ("attn1.to_v.weight", [d, d]),
            ("attn2.norm_k.weight", [128]), ("attn2.norm_q.weight", [128]), ("attn2.to_k.weight", [1024, d]),
            ("attn2.to_out.0.weight", [d, d]), ("attn2.to_q.weight", [d, d]), ("attn2.to_v.weight", [1024, d]),
            ("ff.net.0.proj.weight", [d, h]), ("ff.net.2.weight", [h, d]),
            ("norm1.linear_1.weight", [d, a]), ("norm1.linear_2.weight", [a, 3 * d]),
            ("norm2.linear_1.weight", [d, a]), ("norm2.linear_2.weight", [a, 3 * d]),
            ("norm3.linear_1.weight", [d, a]), ("norm3.linear_2.weight", [a, 3 * d]),
        ]
        let ditTensors = ditBytesPerValue > 0 ? block : block.map { ($0.0, $0.1.count == 2 ? [$0.1[0], $0.1[1] / 4] : $0.1) }
        var dit = table(layers: 28, prefix: { "transformer_blocks.\($0)." }, tensors: ditTensors,
                        bytesPerValue: max(1, ditBytesPerValue))
        dit["kind"] = "anima-turbo-dit"; dit["rope_layout"] = "interleaved"
        if rotated { dit = convrot(dit) }
        dit["config"] = ["num_attention_heads": 16, "attention_head_dim": 128, "mlp_ratio": 4.0, "num_layers": 28,
                         "in_channels": 16, "patch_size": [1, 2, 2], "adaln_lora_dim": 256, "text_embed_dim": 1024,
                         "rope_scale": [1.0, 4.0, 4.0]] as [String: Any]
        let (e, i) = (1024, 3072)
        let layer: [(String, [Int])] = [
            ("input_layernorm.weight", [e]), ("mlp.down_proj.weight", [i, e]), ("mlp.gate_proj.weight", [e, i]),
            ("mlp.up_proj.weight", [e, i]), ("post_attention_layernorm.weight", [e]), ("self_attn.k_norm.weight", [128]),
            ("self_attn.k_proj.weight", [e, 1024]), ("self_attn.o_proj.weight", [2048, e]),
            ("self_attn.q_norm.weight", [128]), ("self_attn.q_proj.weight", [e, 2048]), ("self_attn.v_proj.weight", [e, 1024]),
        ]
        var encoder = table(layers: 28, prefix: { "layers.\($0)." }, tensors: layer, bytesPerValue: encoderBytesPerValue)
        encoder["last_layer"] = 27
        encoder["config"] = ["hidden_size": e, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
                             "intermediate_size": i, "rope_parameters": ["rope_theta": 1_000_000.0, "rope_type": "default"]]
            as [String: Any]
        return MemoryNeed.AnimaMaps(dit: dit, encoder: encoder)!
    }

    /// An Anima request on the fixture's maps, `threads` P cores, no process base.
    static func animaRequest(_ width: Int, _ height: Int, tokens: Int = 50, ranks: [Int] = [],
                             references: [(width: Int, height: Int)] = [], threads: Int = 6,
                             maps: MemoryNeed.AnimaMaps = animaMaps()) -> MemoryNeed.Request {
        var request = MemoryNeed.Request(card: .anima, width: width, height: height, references: references,
                                         promptTokens: tokens, loraRanks: ranks)
        request.threads = threads
        request.animaMaps = maps
        return request
    }

    /// **Krea 2's maps**: the DiT's 28 single-stream blocks with the published shapes (`d` 6144, 48
    /// heads for 12 `k`/`v`, `h` 16384, the published config) — the norms, QK-norms and modulation
    /// tables in fp32 whatever the type, as the forge keeps them —, the timestep weights (`time_mod_proj`
    /// and the two `time_embed` Linears fp32, as published and kept), and Qwen3-VL-4B's text layers (`last_layer`
    /// 34: the taps stop at the input of layer 35). `ditBytesPerValue` is the DiT's Linears' type — 2
    /// (Standard, bf16), 1 (Compact, int8 and a float scale per output column), 0 for a 4-bit table
    /// (Light, Q4_K_M: a quarter of bf16's bytes) —, `encoderBytesPerValue` the encoder's.
    /// `rotated`: the 8-bit linears published rotated (Comfy-Org's int8 convrot, group 256).
    private static func buildKrea2Maps(ditBytesPerValue: Int, encoderBytesPerValue: Int, rotated: Bool) -> MemoryNeed.Krea2Maps {
        let (d, h, kv, page) = (6144, 16_384, 1536, 16384)
        let linears: [(String, [Int])] = [
            ("attn.to_q.weight", [d, d]), ("attn.to_k.weight", [d, kv]), ("attn.to_v.weight", [d, kv]),
            ("attn.to_gate.weight", [d, d]), ("attn.to_out.0.weight", [d, d]),
            ("ff.gate.weight", [d, h]), ("ff.up.weight", [d, h]), ("ff.down.weight", [h, d]),
        ]
        let floats: [(String, [Int])] = [("norm1.weight", [d]), ("norm2.weight", [d]), ("attn.norm_q.weight", [128]),
                                          ("attn.norm_k.weight", [128]), ("scale_shift_table", [6, d])]
        func bytes(_ shape: [Int], linear: Bool) -> Int {
            let count = shape.reduce(1, *)
            guard linear else { return 4 * count }
            switch ditBytesPerValue {
            case 0: return count / 2
            case 1: return count + 4 * shape[1]
            default: return count * ditBytesPerValue
            }
        }
        var table: [String: Any] = [:], order: [String] = [], offset = page
        func add(_ name: String, _ shape: [Int], linear: Bool, dtype: String? = nil) {
            let size = bytes(shape, linear: linear)
            var entry: [String: Any] = ["offset": offset, "bytes": size, "shape": shape]
            if let dtype { entry["dtype"] = dtype }
            if rotated, linear, dtype == "int8" { entry["rotation"] = ["kind": "convrot", "group": 256] }
            table[name] = entry
            order.append(name)
            offset += (size + page - 1) / page * page
        }
        let linearType = ["q4_k", "int8", "bfloat16"][min(ditBytesPerValue, 2)]
        add("time_embed.linear_1.weight", [256, d], linear: false, dtype: "float32")
        add("time_embed.linear_2.weight", [d, d], linear: false, dtype: "float32")
        add("time_mod_proj.weight", [d, 6 * d], linear: false, dtype: "float32")
        for layer in 0..<28 {
            for (suffix, shape) in floats { add(Krea2DiT.layerPrefix(layer) + suffix, shape, linear: false, dtype: "float32") }
            for (suffix, shape) in linears { add(Krea2DiT.layerPrefix(layer) + suffix, shape, linear: true, dtype: linearType) }
        }
        var dit: [String: Any] = ["page": page, "tensors": table, "order": order]
        dit["kind"] = "krea2-turbo-dit"; dit["rope_layout"] = "interleaved"
        dit["config"] = ["num_attention_heads": 48, "num_key_value_heads": 12, "attention_head_dim": 128,
                         "intermediate_size": h, "num_layers": 28, "in_channels": 64, "timestep_embed_dim": 256,
                         "axes_dims_rope": [32, 48, 48]] as [String: Any]
        let (e, i) = (2560, 9728)
        let layer: [(String, [Int])] = [
            ("input_layernorm.weight", [e]), ("mlp.down_proj.weight", [i, e]), ("mlp.gate_proj.weight", [e, i]),
            ("mlp.up_proj.weight", [e, i]), ("post_attention_layernorm.weight", [e]), ("self_attn.k_norm.weight", [128]),
            ("self_attn.k_proj.weight", [e, 1024]), ("self_attn.o_proj.weight", [4096, e]),
            ("self_attn.q_norm.weight", [128]), ("self_attn.q_proj.weight", [e, 4096]), ("self_attn.v_proj.weight", [e, 1024]),
        ]
        var encoder = self.table(layers: 35, prefix: { "layers.\($0)." }, tensors: layer, bytesPerValue: encoderBytesPerValue)
        if rotated, encoderBytesPerValue == 1, var t = encoder["tensors"] as? [String: [String: Any]] {
            for (name, entry) in t where (entry["shape"] as? [Int])?.count == 2 { t[name]?["rotation"] = ["kind": "convrot", "group": 256] }
            encoder["tensors"] = t
        }
        encoder["last_layer"] = 34
        encoder["config"] = ["hidden_size": e, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
                             "intermediate_size": i, "rope_parameters": ["rope_theta": 5_000_000.0, "rope_type": "default"]]
            as [String: Any]
        return MemoryNeed.Krea2Maps(dit: dit, encoder: encoder)!
    }

    /// A Krea 2 request on the fixture's maps, `threads` P cores, no process base. `tokens`: the
    /// encoder's rows (the template's 34 and the suffix's 5 included).
    static func krea2Request(_ width: Int, _ height: Int, tokens: Int = 59, ranks: [Int] = [],
                             references: [(width: Int, height: Int)] = [], threads: Int = 6,
                             maps: MemoryNeed.Krea2Maps = krea2Maps()) -> MemoryNeed.Request {
        var request = MemoryNeed.Request(card: .krea2, width: width, height: height, references: references,
                                         promptTokens: tokens, loraRanks: ranks)
        request.threads = threads
        request.krea2Maps = maps
        return request
    }

    /// The published tokenizer's tables (`Tokenizer.tableBytes(file:)` on it).
    static let qwenTokenizerTables = ByteTableFigures.mapped(entries: 151_643, keyBytes: 3_989_982)
        + ByteTableFigures.mapped(entries: 151_387, keyBytes: 7_426_578)

    /// `bytesPerValue` 2: the Standard (bf16); 1: an 8-bit map (a float scale per output column).
    private static func buildQwenMaps(bytesPerValue: Int) -> QwenImage21Maps {
        let (d, h) = (4096, 12_288)
        let block: [(String, [Int])] = [
            ("attn.norm_k.weight", [128]), ("attn.norm_q.weight", [128]),
            ("attn.to_k.weight", [d, d]), ("attn.to_out.0.weight", [d, d]), ("attn.to_q.weight", [d, d]),
            ("attn.to_v.weight", [d, d]), ("img_mlp.gate_layer.weight", [d, h]), ("img_mlp.out.weight", [h, d]),
            ("img_mlp.proj.weight", [d, h]),
        ]
        var dit = table(layers: 32, prefix: QwenImage21DiT.layerPrefix, tensors: block, bytesPerValue: bytesPerValue)
        dit["config"] = ["attention_head_dim": 128, "axes_dims_rope": [16, 56, 56], "context_in_dim": 4096, "eps": 1e-6,
                         "in_channels": 64, "mlp_ratio": 3, "num_attention_heads": 32, "num_layers": 32] as [String: Any]
        let layer: [(String, [Int])] = [
            ("input_layernorm.weight", [d]), ("mlp.down_proj.weight", [h, d]), ("mlp.gate_proj.weight", [d, h]),
            ("mlp.up_proj.weight", [d, h]), ("post_attention_layernorm.weight", [d]), ("self_attn.k_norm.weight", [128]),
            ("self_attn.k_proj.weight", [d, 1024]), ("self_attn.o_proj.weight", [d, d]), ("self_attn.q_norm.weight", [128]),
            ("self_attn.q_proj.weight", [d, d]), ("self_attn.v_proj.weight", [d, 1024]),
        ]
        var encoder = table(layers: 36, prefix: { "layers.\($0)." }, tensors: layer, bytesPerValue: bytesPerValue)
        encoder["last_layer"] = 35
        encoder["config"] = ["hidden_size": d, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
                             "intermediate_size": h] as [String: Any]
        encoder["vision_config"] = ["hidden_size": 1152, "num_heads": 16, "intermediate_size": 4304, "depth": 27,
                                    "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
                                    "num_position_embeddings": 2304, "out_hidden_size": 4096,
                                    "deepstack_visual_indexes": [8, 16, 24], "hidden_act": "gelu_pytorch_tanh"] as [String: Any]
        encoder["vision_patch_embed_as_linear"] = "(c, t, y, x)"
        return QwenImage21Maps(dit: dit, encoder: encoder, tokenizerTables: qwenTokenizerTables)!
    }

    /// A Qwen-Image-2.1 request on the fixture's maps, `threads` P cores, no process base.
    static func qwenRequest(_ width: Int, _ height: Int, references: [(width: Int, height: Int)] = [],
                            tokens: Int, ranks: [Int] = [], threads: Int = 6,
                            maps: QwenImage21Maps = qwenMaps()) -> MemoryNeed.Request {
        var request = MemoryNeed.Request(card: .qwenImage21, width: width, height: height, references: references,
                                         promptTokens: tokens, loraRanks: ranks)
        request.threads = threads
        request.qwenMaps = maps
        return request
    }
}

/// `ByteTable.mappedBytes`, reachable from the tests (the table is internal to the module).
enum ByteTableFigures {
    static func mapped(entries: Int, keyBytes: Int) -> Int { ByteTable.mappedBytes(entries: entries, keyBytes: keyBytes) }
}
