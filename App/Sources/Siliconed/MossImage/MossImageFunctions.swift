import CoreAI
import Foundation
import SiliCore

/// **MOSS-Image's DiT in functions of fixed shapes** (`MossImageTransformer.swift` declares them to the engine and holds the loop): the reference's
/// arithmetic (`pipeline_moss_image.py`, `MossImageDiTProgram`'s notes) written op by op on the engine's blocks, every
/// weight an argument read from the arena.
///
///     time          one row    time = mlp₂(SiLU(mlp₀ sinusoid)); modulation = time · W + b, [6, d]; head, [2, d]
///     embed-image   the image  tokens [HW, 128] · W + b: the stream's image rows
///     embed-text    the palier text [Tp, 4096] · W + b: the text's stream
///     pre           rows       n = RMS(x)·γ₁ (· (1 + scale_a) + shift_a); q k v = n · Wqkv (+ b); q, k: RoPE(RMS_head · γ)
///     post          rows       x₁ = x + gate_a · (sdpa(q, K, V) · Wo) (+ b); y = x₁ + gate_f · W_down(SITU(h, g)) (+ b)
///     final         the image  RMS(x)·γ · (1 + scale) + shift · W, the velocity [128, h, w]
///
/// `pre` and `post` come in two kinds: the 42 blocks' (`main`: modulated, gated, no bias, `ffn_up_gate` fused, value
/// first) and the text blocks' (`text`: no modulation, no gate, biases everywhere, `ffn_up` and `ffn_gate` apart). pre
/// runs on every row before post: post's attention reads the keys and values of the whole sequence.
///
/// **The Standard's path, bf16 on the GPU**: the products in bf16 (the published weights as they are, `[in, out]`), every
/// elementwise chain in f32 between them, rounded once to bf16 where a product, a norm or the attention reads it — the
/// reference rounds after every torch operation, and what is judged is the distance to the fp64 witness (the former
/// engine's finding: forcing each rounding cost time and moved the result away from the witness). The RoPE tables stay
/// f32 where the reference rounds them to bf16; the residual stream is bf16 between functions, as the reference's.
///
/// **The Compact's path** is the shared forms' (`MossShared`): the 42 blocks, their entry and the head; the time path,
/// the embedders and the text blocks are these, on the same weights.
enum MossFunctions {
    static let d = 3072, heads = 24, hd = 128, hidden = 8192, timeDim = 256, textDim = 4096, channels = 128
    /// A row of the shared modulation's, `[6, gainRow]`: each starts a page of its own, its readers taking one row each, a
    /// buffer of its own (`Layout.pageRow`).
    static let gainRow = Layout.pageRow(d, .bfloat16)
    static let eps = 1e-6
    typealias Argument = (name: String, shape: [Int], element: Aicode.Element)

    // MARK: - The arguments

    /// The function's arguments in the order it declares them. `rows`: a call's (pre, post), the image's (embed-image,
    /// final), the text's palier (embed-text); `keys`: post's.
    static func arguments(_ f: String, rows: Int, keys: Int, height h: Int = 0, width w: Int = 0, ranks: [String: Int] = [:]) -> [Argument] {
        plainArguments(f, rows: rows, keys: keys, height: h, width: w) + lowRankArguments(f, ranks: ranks, element: .bf16)
    }

    // MARK: - The low-rank path (`LoRA.swift`)

    /// The 42 blocks' products a LoRA may adapt, by the function that computes each: its argument, its in, its out.
    static let adapted: [String: [(name: String, inn: Int, out: Int)]] = [
        "pre": [("qkv", d, 3 * d)], "post": [("o", d, d), ("up_gate", d, 2 * hidden), ("down", hidden, d)],
        "out": [("o", d, d)], "ffn": [("up_gate", d, 2 * hidden), ("down", hidden, d)]]
    /// Each adapted product's key in the published file, block i's.
    static func key(_ product: String, _ i: Int) -> String {
        "layers.\(i).proj." + ["qkv": "qkv", "o": "o", "up_gate": "ffn_up_gate", "down": "ffn_down"][product]!
    }
    static func loraProducts(_ i: Int) -> [LoRAProduct] {
        [("qkv", d, 3 * d), ("o", d, d), ("up_gate", d, 2 * hidden), ("down", hidden, d)].map {
            LoRAProduct(name: $0.0, key: key($0.0, i), inn: $0.1, out: $0.2)
        }
    }

    /// `<product>_down` `[in, R]` and `<product>_up` `[R, out]` for each product of `f` a rank adapts (`ranks`, by the
    /// product: the same in every block).
    static func lowRankArguments(_ f: String, ranks: [String: Int], element: Aicode.Element) -> [Argument] {
        (adapted[f] ?? []).flatMap { p -> [Argument] in
            guard let r = ranks[p.name], r > 0 else { return [] }
            return [(p.name + "_down", [p.inn, r], element), (p.name + "_up", [r, p.out], element)]
        }
    }

    /// The function's arguments without the low-rank path's.
    static func plainArguments(_ f: String, rows: Int, keys: Int, height h: Int, width w: Int) -> [Argument] {
        let (d, hd, hidden) = (Self.d, Self.hd, Self.hidden)
        switch f {
        case "time":
            return [("t", [1, timeDim], .f32), ("t0", [timeDim, d], .bf16), ("t0_bias", [d], .f32), ("t2", [d, d], .bf16), ("t2_bias", [d], .f32),
                    ("mod", [d, 6 * d], .bf16), ("mod_bias", [6 * d], .f32), ("head", [d, 2 * d], .bf16), ("head_bias", [2 * d], .f32)]
        case "embed-image":
            return [("latent", [channels, h, w], .f32), ("video", [channels, d], .bf16), ("video_bias", [d], .f32)]
        case "embed-text":
            return [("text", [rows, textDim], .f32), ("text_proj", [textDim, d], .bf16), ("text_proj_bias", [d], .f32)]
        case "pre", "text-pre":
            let text = f == "text-pre"
            return [("x", [rows, d], .bf16)] + (text ? [] : [("shift", [1, d], .bf16), ("scale", [1, d], .bf16)])
                + [("norm1", [d], .f32), ("qkv", [d, 3 * d], .bf16)] + (text ? [("qkv_bias", [3 * d], .f32)] : [])
                + [("q_norm", [hd], .f32), ("k_norm", [hd], .f32), ("cos", [rows, 1, hd], .f32), ("sin", [rows, 1, hd], .f32)]
        case "post":
            return [("x", [rows, d], .bf16), ("q", [1, heads, rows, hd], .bf16), ("k", [1, keys, heads, hd], .bf16), ("v", [1, keys, heads, hd], .bf16),
                    ("mask", [1, keys], .bf16), ("gate_a", [1, d], .bf16), ("o", [d, d], .bf16), ("norm2", [d], .f32),
                    ("shift_f", [1, d], .bf16), ("scale_f", [1, d], .bf16), ("gate_f", [1, d], .bf16),
                    ("up_gate", [d, 2 * hidden], .bf16), ("down", [hidden, d], .bf16)]
        case "text-post":
            return [("x", [rows, d], .bf16), ("q", [1, heads, rows, hd], .bf16), ("k", [1, keys, heads, hd], .bf16), ("v", [1, keys, heads, hd], .bf16),
                    ("mask", [1, keys], .bf16), ("proj", [d, d], .bf16), ("proj_bias", [d], .f32), ("norm2", [d], .f32),
                    ("up", [d, hidden], .bf16), ("up_bias", [hidden], .f32), ("gate", [d, hidden], .bf16), ("gate_bias", [hidden], .f32),
                    ("down", [hidden, d], .bf16), ("down_bias", [d], .f32)]
        case "final":
            return [("x", [rows, d], .bf16), ("norm", [d], .f32), ("head", [2, d], .bf16), ("final_linear", [d, channels], .bf16)]
        default: preconditionFailure("no function \(f)")
        }
    }


    // MARK: - The functions

    static func module(_ f: String, rows: Int, keys: Int, height h: Int = 0, width w: Int = 0, tapped: Bool = false,
                       ranks: [String: Int] = [:]) -> Aicode.Module {
        let arguments = arguments(f, rows: rows, keys: keys, height: h, width: w, ranks: ranks)
        var m = Aicode.Module(arguments: arguments.map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        let b = m.body
        let names = m.argumentNames
        func a(_ name: String) -> Int { names.firstIndex(of: name)! }
        /// The product's result y with its low-rank path beside it, x as the product read it.
        func adapt(_ y: Int, _ x: Int, _ product: String) -> Int {
            (ranks[product] ?? 0) > 0 ? b.lowRank(y, x, down: a(product + "_down"), up: a(product + "_up")) : y
        }
        let (d, hd, hidden) = (Self.d, Self.hd, Self.hidden)
        func f32(_ x: Int) -> Int { b.castUnlessSame(x, to: .f32) }
        func bf16(_ x: Int) -> Int { b.castUnlessSame(x, to: .bf16) }
        /// `x · W` in bf16, `x` rounded once to bf16 first; with a bias, its sum in f32.
        func product(_ x: Int, _ w: String, bias: String? = nil) -> Int {
            let x = bf16(x)
            let n = Aicode.shape(b.types[x]).shape[0], out = Aicode.shape(b.types[a(w)]).shape[1]
            let p = b.matmul(x, a(w), result: Aicode.tensor([n, out], .bf16))
            return bias.map { b.binary(.add, f32(p), a($0)) } ?? p
        }
        /// `RMS(x)` rounded once to x's element as the reference's `y.to(x.dtype)`, times the gain in f32.
        func normed(_ x: Int, _ gain: String, axis: Int) -> Int { b.binary(.mul, f32(b.rmsNorm(x, axes: [axis], eps: eps)), a(gain)) }
        /// `n · (1 + scale) + shift` in f32.
        func modulated(_ n: Int, _ scale: String, _ shift: String) -> Int {
            b.binary(.add, b.binary(.mul, n, b.scale("add", f32(a(scale)), 1)), f32(a(shift)))
        }
        // RoPE on halves in f32 (`_rotate`: x · cos + cat(−b, a) · sin), the tables `[rows, 1, 128]` holding [cos | cos]
        // and [−sin | sin], so that the halves are only swapped.
        func rope(_ t: Int) -> Int {
            let swapped = b.concat([b.slice(t, axis: 2, start: hd / 2, size: hd / 2), b.slice(t, axis: 2, start: 0, size: hd / 2)], axis: 2)
            return b.binary(.add, b.binary(.mul, t, a("cos")), b.binary(.mul, swapped, a("sin")))
        }
        func headsFirst(_ t: Int) -> Int { b.reshape(b.transpose(t, [1, 0, 2]), [1, heads, rows, hd]) }
        // The keys and values token first, `[1, rows, heads, 128]`: a tile's a block of the sequence's, written whole, their
        // heads taken first inside the attention (measured 19/11 on Z: the same bits, the write twice faster).
        func tokenFirst(_ t: Int) -> Int { b.reshape(t, [1, rows, heads, hd]) }
        // `_situ`: (4 · tanh(g/4) · σ(g)) · (25 · tanh(h/25)), in f32.
        func situ(_ h: Int, _ g: Int) -> Int {
            let gate = b.binary(.mul, b.scale("mul", b.tanh(b.scale("mul", g, 0.25)), 4), b.sigmoid(g))
            return b.binary(.mul, gate, b.scale("mul", b.tanh(b.scale("mul", h, 0.04)), 25))
        }

        switch f {
        case "time":
            let s = b.silu(product(a("t"), "t0", bias: "t0_bias"))
            let time = bf16(product(s, "t2", bias: "t2_bias"))
            m.outputs = [("time", time), ("modulation", b.zeroPadded(b.reshape(bf16(product(time, "mod", bias: "mod_bias")), [6, d]), to: [6, Self.gainRow])),
                         ("head", b.reshape(bf16(product(time, "head", bias: "head_bias")), [2, d]))]
        case "embed-image":
            let tokens = b.transpose(b.reshape(a("latent"), [channels, h * w]), [1, 0])
            m.outputs = [("x", bf16(product(tokens, "video", bias: "video_bias")))]
        case "embed-text":
            m.outputs = [("x", bf16(product(a("text"), "text_proj", bias: "text_proj_bias")))]
        case "pre", "text-pre":
            let text = f == "text-pre"
            var n = normed(a("x"), "norm1", axis: 1)
            if !text { n = modulated(n, "scale", "shift") }
            let all = text ? bf16(product(n, "qkv", bias: "qkv_bias")) : adapt(product(n, "qkv"), bf16(n), "qkv")
            func third(_ i: Int) -> Int { b.reshape(b.slice(all, axis: 1, start: i * d, size: d), [rows, heads, hd]) }
            let q = normed(third(0), "q_norm", axis: 2), k = normed(third(1), "k_norm", axis: 2)
            m.outputs = [("q", headsFirst(bf16(rope(q)))), ("k", tokenFirst(bf16(rope(k)))), ("v", tokenFirst(third(2)))]
            if tapped { m.outputs += [("qkv_out", all), ("q_norm_out", b.reshape(bf16(q), [rows, d]))] }
        case "post", "text-post":
            let text = f == "text-post"
            let o = b.attention([a("q"), b.transpose(a("k"), [0, 2, 1, 3]), b.transpose(a("v"), [0, 2, 1, 3]), a("mask")], scale: 1 / Double(hd).squareRoot(), symbol: "sdpa_tile")
            let rowsFirst = b.reshape(b.transpose(b.reshape(o, [heads, rows, hd]), [1, 0, 2]), [rows, d])
            let attended = text ? product(rowsFirst, "proj", bias: "proj_bias") : adapt(product(rowsFirst, "o"), rowsFirst, "o")
            let x1 = b.binary(.add, f32(a("x")), text ? attended : b.binary(.mul, f32(attended), f32(a("gate_a"))))
            var z = normed(x1, "norm2", axis: 1)
            if !text { z = modulated(z, "scale_f", "shift_f") }
            let hValue: Int, gValue: Int, upGate: Int?
            if text {
                (hValue, gValue, upGate) = (product(z, "up", bias: "up_bias"), product(z, "gate", bias: "gate_bias"), nil)
            } else {
                let ug = adapt(product(z, "up_gate"), bf16(z), "up_gate")
                (hValue, gValue, upGate) = (f32(b.slice(ug, axis: 1, start: 0, size: hidden)), f32(b.slice(ug, axis: 1, start: hidden, size: hidden)), ug)
            }
            let s = situ(hValue, gValue)
            let down = text ? product(s, "down", bias: "down_bias") : adapt(product(s, "down"), bf16(s), "down")
            let y = b.binary(.add, x1, text ? down : b.binary(.mul, f32(down), f32(a("gate_f"))))
            m.outputs = [("y", bf16(y))]
            if tapped, let upGate { m.outputs += [("o_out", attended), ("up_gate_out", upGate), ("down_out", down)] }
        case "final":
            let n = normed(a("x"), "norm", axis: 1)
            let head = f32(a("head"))
            let shift = b.slice(head, axis: 0, start: 0, size: 1), scale = b.slice(head, axis: 0, start: 1, size: 1)
            let y = f32(product(b.binary(.add, b.binary(.mul, n, b.scale("add", scale, 1)), shift), "final_linear"))
            m.outputs = [("velocity", b.reshape(b.transpose(y, [1, 0]), [channels, h, w]))]
            if tapped { m.outputs += [("final_norm_out", bf16(n))] }
        default: preconditionFailure("no function \(f)")
        }
        return m
    }

    // MARK: - The weights

    /// One of the 42 blocks' in bf16: (argument, key, recipe).
    static func layer(_ i: Int) -> [Weights.Entry] {
        let p = "layers.\(i)."
        let products = [("qkv", "proj.qkv"), ("o", "proj.o"), ("up_gate", "proj.ffn_up_gate"), ("down", "proj.ffn_down")].map { name, key -> Weights.Entry in
            (name, p + key + ".weight", .bfloatTransposed)
        }
        return products + [("norm1", p + "norm1.weight", .float), ("norm2", p + "norm2.weight", .float),
                           ("q_norm", p + "q_norm.weight", .float), ("k_norm", p + "k_norm.weight", .float)]
    }

    /// A text block's, by the names of its functions' arguments.
    static func textBlock(_ i: Int) -> [Weights.Entry] {
        let p = "text_embedder.blocks.\(i)."
        var all: [Weights.Entry] = []
        for (name, key) in [("qkv", "qkv"), ("proj", "proj"), ("up", "ffn_up"), ("gate", "ffn_gate"), ("down", "ffn_down")] {
            all.append((name, p + key + ".weight", .bfloatTransposed))
            all.append((name + "_bias", p + key + ".bias", .float))
        }
        for name in ["norm1", "norm2", "q_norm", "k_norm"] { all.append((name, p + name + ".weight", .float)) }
        return all
    }

    /// Everything but the blocks, read once and kept: the time path, the embedders, the head.
    static var others: [Weights.Entry] {
        let products: [(String, String, Bool)] = [("t0", "time_embedding.0", true), ("t2", "time_embedding.2", true),
                                                   ("mod", "shared_block_modulation", true), ("head", "video_head_modulation", true),
                                                   ("video", "video_embedder", true), ("text_proj", "text_embedder.proj", true),
                                                   ("final_linear", "final_linear_video", false)]
        var all: [Weights.Entry] = []
        for (name, key, bias) in products {
            all.append((name, key + ".weight", .bfloatTransposed))
            if bias { all.append((name + "_bias", key + ".bias", .float)) }
        }
        all.append(("norm", "final_norm_video.weight", .float))
        return all
    }
}
