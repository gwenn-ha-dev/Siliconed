import CoreAI
import Foundation
import Metal
import SiliCore

/// **Anima's DiT as its block declares it**: what it reads, its LoRAs' products, its weights' entries, its forms
/// and edges.
extension AnimaEvaluation {
    /// The product whose element says the Compact (`Arithmetic.codes`): block 1's, block 0 left in bf16 by the Compact.
    static let codesMarker = "blocks.1.self_attn.q_proj.weight"

    /// Block i's products the file holds as codes (the Compact leaves blocks 0 and 27 and the MLP of blocks 1 to 3 in bf16).
    static func quantized(_ i: Int, in file: SafeTensors) -> Set<String> {
        Set(products.filter { file.tensors["blocks.\(i)." + $0.value + ".weight"]?.dtype == "I8" }.map(\.key))
    }

    /// A block's products: their arguments' names, their modules.
    static let products = ["q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.output_proj",
                           "cq": "cross_attn.q_proj", "ck": "cross_attn.k_proj", "cv": "cross_attn.v_proj", "co": "cross_attn.output_proj",
                           "m1": "mlp.layer1", "m2": "mlp.layer2"]

    /// Block i's function of `kind`: its own products on codes.
    static func function(_ kind: AnimaFunction.Kind, _ path: Path, block q: Set<String>, rows: Int = 1, keys: Int = 0,
                         context: Int = 0, taps: Bool = false, ranks: [String: Int] = [:]) -> AnimaFunction {
        let own = q.intersection(AnimaFunction.products(kind)), adapted = AnimaFunction.adapted(kind)
        return AnimaFunction(kind: kind, path: path, quantized: own, rows: rows, keys: keys, context: context, taps: taps,
                             ranks: ranks.filter { adapted.contains($0.key) })
    }

    // MARK: - The LoRAs

    /// The modules a LoRA adapts in a block, by their arguments' names: its products and its modulation's.
    static let adapted = products.merging(["sa1": "adaln_modulation_self_attn.1", "sa2": "adaln_modulation_self_attn.2",
                                           "ca1": "adaln_modulation_cross_attn.1", "ca2": "adaln_modulation_cross_attn.2",
                                           "ml1": "adaln_modulation_mlp.1", "ml2": "adaln_modulation_mlp.2"]) { a, _ in a }

    /// Block i's adapted products, as the LoRAs read them (the module's name, without `.weight`).
    static func loraProducts(block i: Int) -> [LoRAProduct] {
        adapted.sorted { $0.key < $1.key }.map { name, module in
            let (inn, out) = AnimaFunction.dims(name)
            return LoRAProduct(name: name, key: "blocks.\(i).\(module)", inn: inn, out: out)
        }
    }

    /// Each adapted product's rank in the functions: the same in every block (`LoRAStack.rank`), none where nothing lands.
    static func ranks(_ loras: LoRAStack) -> [String: Int] { loras.ranks(blocks.indices.flatMap { loraProducts(block: $0) }, on: .anima) }

    static func factorElement(_ path: Path) -> NDArray.ScalarType { path == .bf16 ? .bfloat16 : .float16 }

    // MARK: - The weights

    /// A product's entry: the codes (`<key>.weight`, `.weight_scale`), or the float weight transposed in the path's element.
    private static func product(_ name: String, _ key: String, _ path: Path, quantized: Bool) -> Weights.Entry {
        switch path {
        case .bf16: (name, key + ".weight", .bfloatTransposed)
        case .f16: quantized ? (name, key, .codes) : (name, key + ".weight", .halfTransposed)
        }
    }

    private static func gain(_ name: String, _ key: String, _ path: Path) -> Weights.Entry {
        (name, key, path == .bf16 ? .bfloat16(joined: []) : .half)
    }

    /// Block i's group in the guarded forms: its modulation, its self-attention, its cross-attention's query and output,
    /// its MLP.
    static func entries(block i: Int, _ path: Path, quantized: Set<String>) -> [Weights.Entry] {
        func product(_ name: String) -> Weights.Entry { Self.product(name, "blocks.\(i)." + products[name]!, path, quantized: quantized.contains(name)) }
        let p = "blocks.\(i)."
        return modulationEntries(block: i) + ["q", "k", "v", "o", "cq", "co", "m1", "m2"].map(product)
            + [gain("q_norm", p + "self_attn.q_norm.weight", path), gain("k_norm", p + "self_attn.k_norm.weight", path),
               gain("cq_norm", p + "cross_attn.q_norm.weight", path)]
    }

    /// Block i's modulation, bf16 on both paths: the one-row function widens it.
    static func modulationEntries(block i: Int) -> [Weights.Entry] {
        var all: [Weights.Entry] = []
        for (name, key) in [("sa", "adaln_modulation_self_attn"), ("ca", "adaln_modulation_cross_attn"), ("ml", "adaln_modulation_mlp")] {
            let k: String = "blocks.\(i)." + key
            all.append((name + "1", k + ".1.weight", .bfloatTransposed))
            all.append((name + "2", k + ".2.weight", .bfloatTransposed))
        }
        return all
    }

    /// Block i's cross keys' and values' products, read once a prompt: held, whatever B.
    static func contextEntries(block i: Int, _ path: Path, quantized q: Set<String>) -> [Weights.Entry] {
        let p = "blocks.\(i).cross_attn."
        return [product("ck", p + "k_proj", path, quantized: q.contains("ck")), product("cv", p + "v_proj", path, quantized: q.contains("cv")),
                gain("ck_norm", p + "k_norm.weight", path)]
    }

    /// The time path's, the embedder's and the final layer's.
    static func others(_ path: Path) -> [Weights.Entry] {
        [("t1", "t_embedder.1.linear_1.weight", .bfloatTransposed), ("t2", "t_embedder.1.linear_2.weight", .bfloatTransposed),
         ("t_norm", "t_embedding_norm.weight", .bfloat16(joined: [])), ("f1", "final_layer.adaln_modulation.1.weight", .bfloatTransposed),
         ("f2", "final_layer.adaln_modulation.2.weight", .bfloatTransposed),
         product("xe", "x_embedder.proj.1", path, quantized: false), product("fl", "final_layer.linear", path, quantized: false)]
    }

    // MARK: - The declaration

    /// The guarded form of a block whose products on codes are `q` (and block 0's under taps, its branches besides).
    static func guardedForm(_ q: Set<String>, taps: Bool) -> String {
        let own = q.subtracting(["ck", "cv"]).sorted()
        return "guarded" + (own.isEmpty ? "" : "-" + own.joined(separator: "+")) + (taps ? "-taps" : "")
    }

    /// The context's edge of a block whose products on codes are `q`.
    static func contextEdge(_ q: Set<String>) -> String {
        let own = q.intersection(["ck", "cv"]).sorted()
        return "context" + (own.isEmpty ? "" : "-" + own.joined(separator: "+"))
    }

    /// The modulation's rows (`self`, `cross`, `mlp`: shift, 1 + scale, gate) as the guarded forms read them, a row each.
    static let rows: [String: Block.Bind] = {
        var all: [String: Block.Bind] = [:]
        for (k, sublayer) in ["self", "cross", "mlp"].enumerated() {
            for (j, gain) in ["shift", "scale", "gate"].enumerated() { all["\(gain)\(k + 1)"] = .row(sublayer, j) }
        }
        return all
    }()

    /// **The DiT's declaration**: its per-row tensors (the stream and its factors, q k v, the RoPE tables; in the shared
    /// forms the steps' own; block 0's branches under taps), its other tensors (each block's cross keys and values its
    /// own), the edges' weights read once, a group a block (its context products held), its forms, the image's pass.
    static func declaration(_ g: Geometry, _ arithmetic: Arithmetic, quantized: [Set<String>], taps: Bool, ranks: [String: Int]) -> Block.Declaration {
        let path = Self.path(arithmetic), shared = sharedBlocks(arithmetic, quantized)
        let (k, c, e) = (g.keys, g.context, factorElement(path)), (bytes, gainRow) = (Layout.element(e), AnimaFunction.gainRow)
        var n = Block.Declaration("Anima's DiT (\(path)\(shared.isEmpty ? "" : ", shared"), \(g.height)×\(g.width), \(k) keys)", rows: k)
        n.perRow = [.init("x", width: d, e, next: "y", factor: path == .f16 ? "f" : nil), .init("q", width: d, e), .init("k", width: d, e),
                    .init("v", width: d, e), .init("cos", width: hd / 2, .float32), .init("sin", width: hd / 2, .float32)]
        if path == .f16 { n.perRow.append(.init("f", width: 1, .float32, next: "f_next")) }
        if !shared.isEmpty {
            n.perRow += [.init("x_factor", width: 1, .float16, next: "y_factor"), .init("qkv_factor", width: 1, .float16),
                         .init("attended", width: d, .float16), .init("mid", width: d, .float16, factor: "mid_factor"), .init("mid_factor", width: 1, .float16),
                         .init("mid2", width: d, .float16, factor: "mid2_factor"), .init("mid2_factor", width: 1, .float16),
                         .init("cqh", width: d, .float16), .init("cq_factor", width: 1, .float16)]
                + ["qh", "kh", "vh"].map { .init($0, width: d, .float16) }
        }
        // Block 0's branches: f32 from its guarded form, f16 from the shared forms (`AnimaShared`'s taps, a run tapped).
        let listening = taps && shared.contains(0)
        if taps { n.perRow += ["attn1", "attn2", "ff"].map { .init($0, width: d, listening ? .float16 : .float32) } }
        n.tapped = listening
        let latent = AnimaFunction.channels * g.height * g.width * 4
        var t: [(String, Int)] = [("latent", latent), ("t_freq", d * 4), ("temb", 3 * d * 4), ("embedded", d * 4), ("final", 2 * gainRow * bytes),
                                  ("mask", k * bytes), ("context", c * AnimaFunction.textDim * bytes), ("ctx_mask", c * bytes)]
        t += ["self", "cross", "mlp"].map { ($0, 3 * gainRow * bytes) }
        t += blocks.flatMap { b in ["ctx_k", "ctx_v"].map { ("\(b).\($0)", c * d * bytes) } }
        if !shared.isEmpty { t += AnimaShared.gains.map { ($0, d * 2) } + AnimaShared.gateFactors.map { ($0, 2) } }
        n.tensors = t.map { SiliCore.Stage.Tensor($0.0, bytes: $0.1) } + [SiliCore.Stage.Tensor("velocity", bytes: latent, .apart)]
            + blocks.indices.flatMap { LoRAStack.tensors(loraProducts(block: $0), ranks: ranks, element: e, prefix: "\(block($0)).") }
        n.residents = others(path)
        func form(_ i: Int) -> String { shared.contains(i) ? "shared" : guardedForm(quantized[i], taps: taps && i == 0) }
        n.groups = blocks.indices.map { i in
            Block.Group(block(i), form: form(i), weights: shared.contains(i) ? AnimaShared.layer(i, weights: arithmetic.weights) : entries(block: i, path, quantized: quantized[i]),
                        held: contextEntries(block: i, path, quantized: quantized[i]))
        }
        n.cycle = blocks
        // The guarded forms read the f32 factors; the activations' q beside the product's weight q.
        let factors: [String: Block.Bind] = path == .f16 ? ["x_factor": .rows("f"), "y_factor": .next("f")] : [:]
        var made = Set<String>()
        for i in blocks.indices where !shared.contains(i) && made.insert(form(i)).inserted {
            let (q, tapping) = (quantized[i], taps && i == 0)
            n.forms.append(Block.Form(form(i), steps: [.init("pre", on: .gpu), .init("post", on: .gpu, whole: ["k", "v"], bind: ["q": .rows("q")])],
                                      bind: rows.merging(factors) { a, _ in a }) { step, b in
                step == "pre" ? function(.pre, path, block: q, rows: b.rows, ranks: ranks).module()
                    : function(.post, path, block: q, rows: b.rows, keys: k, context: c, taps: tapping, ranks: ranks).module()
            })
        }
        if !shared.isEmpty {
            n.forms.append(Block.Form("shared", steps: [
                .init("pre"), .init("heads", on: .gpu), .init("attention", on: .gpu, whole: ["k", "v"], bind: ["q": .rows("q"), "o": .rows("attended")]),
                .init("out", bind: ["y": .rows("mid"), "y_factor": .rows("mid_factor")], taps: listening),
                .init("cross", on: .gpu, bind: ["o": .rows("attended")]),
                .init("crossout", bind: ["x": .rows("mid"), "x_factor": .rows("mid_factor"), "y": .rows("mid2"), "y_factor": .rows("mid2_factor")],
                      taps: listening),
                .init("ffn", bind: ["x": .rows("mid2"), "x_factor": .rows("mid2_factor")], taps: listening)]) { step, b in
                AnimaShared.module(step + (b.tapped ? "-taps" : ""), rows: b.rows, keys: k, context: c, weights: arithmetic.weights, ranks: ranks,
                                   neural: b.neural)
            })
        }
        n.passes = [Block.Pass("image", rows: 0..<g.tokens, shareable: 0..<g.tokens)]
        n.edges["time"] = { _ in AnimaFunction(kind: .time, path: path).module() }
        n.edges["modulation"] = { b in function(.modulation, path, block: [], ranks: ranks).module() }
        n.edges["embed"] = { _ in AnimaFunction(kind: .embed, path: path, rows: g.tokens, height: g.height, width: g.width).module() }
        n.edges["final"] = { _ in AnimaFunction(kind: .final, path: path, rows: g.tokens, height: g.height, width: g.width).module() }
        for q in Set(quantized.map { $0.intersection(["ck", "cv"]) }) {
            n.edges[contextEdge(q)] = { b in function(.context, path, block: q, context: c, ranks: ranks).module() }
        }
        return n
    }
}
