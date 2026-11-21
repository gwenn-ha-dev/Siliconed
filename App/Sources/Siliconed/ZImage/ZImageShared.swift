import Foundation
import SiliCore

/// **The Compact's Z-Image block, as the GPU and the ANE share it** — on the GPU alone (`exact`) or beside the ANE: the
/// same functions, two specializations of one asset. A block in f16 written as the Standard's (its guard and its norms
/// in f32, its `dequantize` reading a vector of scales, its post holding the attention) does not reach the ANE: the ANE's
/// compiler fails and Core AI runs it on the GPU, bit for bit (measured 17/11 on the M1 Pro). Here pre, out and ffn are
/// written in the forms both units take (`Neural.swift`: f16 at every border and inside, `[1, rows, n]`, q64 products,
/// norms by their peak, the guard's powers of two in f16); what reads across rows or computes in f32 is a function of
/// the GPU's:
///
///     modulation   GPU    `ZImageBlock`'s under the guard (one row), its four rows each an output
///     pre          both   n = RMS(x̂ · X) · γ₁ · s₁, rotated, brought under 2^e₀ by F: qkv̂ = q64, and F (e₀ an argument, `targets`)
///     heads        GPU    q, k: RMS_head(qkv̂) · γ (eps / F²), RoPE in f16; v = v̂ · F; q heads first, k and v
///                         token first into the whole sequence's rows
///     attention    GPU    o = sdpa(q, K, V, mask), rows first: a true value, under max|v|
///     out          both   x₁ = x + g₁ · RMS(o R · Wo) · γ₂, the stream kept as (x̂, X)
///     ffn          both   y = x₁ + the feed-forward half
///
/// Out and ffn are two functions, not one post, because the ANE fails to run them as one (`ANE inference operation
/// failed`, code −19, at its first call, from 1152 to 3456 rows; measured 17/11 on the M1 Pro): the post runs whole with
/// w₂'s k = 10 240 in one product, not with it cut under 4096 — the cut the feed-forward half alone runs.
///
/// Every factor is a power of two `[1, rows, 1]` in f16, the stream's included: X within [2⁻¹³, 2¹⁵].
///
/// **Two weights** (`SharedWeights`): the Compact's codes, each product's input rotated (q64); the Standard's bf16 rounded
/// to f16 as they are read, the input as it is (`product16`) — the Standard beside the ANE, the same functions else.
/// Its modulation reads `ada` in f16 under the guard of the edges (`ZImageFunctions`), the GPU's.
///
/// **The modulation gives the four rows the block reads each as its own output**: called by `encode` on the heap's
/// buffers, the ANE read a view at an offset into one wrong (measured 17/11 on the M1 Pro). What the ANE reads is a
/// surface of its own (`Arena`), which leaves no offset to read.
///
/// **A LoRA beside each product** (`ZImageBlock`'s rule): the factors f16 `[in, R]` and `[R, out]`, surfaces of their
/// own; each product's input before its rotation under the same factor. `taps`: the block's inner boundaries, true values
/// in f32, as the outputs in place of its own (`.norm1`, `.attentionIn`, `.q`, `.k`, `.v`; `.qNormed`, `.kNormed`;
/// `.sdpa`; `.attention`, `.norm2`; `.ffnNorm1`, `.ffnIn`, `.w1`, `.w3`, `.swiglu`, `.feedForward`, `.ffnNorm2`), what
/// `check z-image block` and `block-lora` read on the GPU; another asset.
///
/// **Against the CPU** (f32, on 4 tiles of 1152 rows, the ANE the first 3; measured 17/11 on the M1 Pro): row 0, the
/// ANE's, pre and heads 5.0e-4, out and ffn 8.64e-4; row 4223, the GPU's, 3.42e-4 and 7.82e-4; the f16 block written as
/// the Standard's, 3.84e-4 and 8.05e-4.
enum ZImageShared {
    typealias Argument = ZImageFunctions.Argument
    static var d: Int { ZImageBlock.d }
    static var heads: Int { ZImageBlock.heads }
    static var hd: Int { ZImageBlock.hd }
    static var hidden: Int { ZImageBlock.hidden }
    static var eps: Double { ZImageBlock.eps }
    /// What the ANE takes of the shared forms (the others: modulation, heads, attention).
    static let neural = ["pre", "out", "ffn"]

    /// **Each product's target, an argument** `[1, 1, 1]` f16 computed when the weights are read (`Recipe.target`), not
    /// `target16` in the graph: the ANE got it wrong inside the ffn (measured 18/11/2026 on the M5 Pro; the recipe says
    /// how). Its name, and the products whose codes it reads.
    static let targets: [(name: String, products: [String])] = [("qkv_target", ["qkv"]), ("out_target", ["out"]), ("w13_target", ["w1", "w3"]),
                                                                ("w2_target", ["w2"])]
    static func target(_ products: [String]) -> String { targets.first { $0.products == products }!.name }

    /// The products a LoRA adapts, by the function that computes each.
    static let adapted: [String: [String]] = ["pre": ["qkv"], "out": ["out"], "ffn": ["w1", "w3", "w2"]]

    static func arguments(_ function: String, rows: Int, keys: Int, weights: SharedWeights, ranks: [String: Int] = [:]) -> [Argument] {
        plainArguments(function, rows: rows, keys: keys, weights: weights) + lowRankArguments(function, ranks: ranks)
    }

    static func lowRankArguments(_ function: String, ranks: [String: Int]) -> [Argument] {
        let shapes = Dictionary(uniqueKeysWithValues: ZImageDiT.products("layers.0").map { ($0.name, (inn: $0.inn, out: $0.out)) })
        return (adapted[function] ?? []).flatMap { p -> [Argument] in
            guard let r = ranks[p], r > 0, let s = shapes[p] else { return [] }
            return [(p + "_down", [s.inn, r], .f16), (p + "_up", [r, s.out], .f16)]
        }
    }

    private static func plainArguments(_ function: String, rows: Int, keys: Int, weights: SharedWeights) -> [Argument] {
        let row = [1, 1, d], one = [1, 1, 1]
        func product(_ name: String, _ inn: Int, _ out: Int) -> [Argument] { weights.arguments(name, inn: inn, out: out) }
        switch function {
        case "pre":
            return [("x", [1, rows, d], .f16), ("x_factor", [1, rows, 1], .f16), ("scale1", row, .f16), ("norm1", row, .f16)]
                + product("qkv", d, 3 * d) + [("qkv_target", one, .f16)]
        case "heads":
            return [("qkv", [1, rows, 3 * d], .f16), ("qkv_factor", [1, rows, 1], .f16), ("q_norm", [hd], .f16), ("k_norm", [hd], .f16),
                    ("cos", [rows, 1, hd], .f32), ("sin", [rows, 1, hd], .f32)]
        case "attention":
            return [("q", [1, heads, rows, hd], .f16), ("k", [1, keys, heads, hd], .f16), ("v", [1, keys, heads, hd], .f16), ("mask", [1, keys], .f16)]
        case "out":
            return [("x", [1, rows, d], .f16), ("x_factor", [1, rows, 1], .f16), ("o", [1, rows, d], .f16), ("gate1", row, .f16), ("norm2", row, .f16)]
                + product("out", d, d) + [("out_target", one, .f16)]
        default:
            return [("x", [1, rows, d], .f16), ("x_factor", [1, rows, 1], .f16), ("scale2", row, .f16), ("gate2", row, .f16),
                    ("ffn_norm1", row, .f16), ("ffn_norm2", row, .f16)]
                + product("w1", d, hidden) + product("w3", d, hidden) + product("w2", hidden, d) + [("w13_target", one, .f16), ("w2_target", one, .f16)]
        }
    }

    /// The guarded modulation (`ZImageBlock`'s), the four rows the block reads: s₁, g₁, s₂, g₂, each `[1, 1, d]`; with
    /// `taps`, its own outputs replaced by them.
    static func modulation(weights: SharedWeights, ranks: [String: Int] = [:], taps: [ZImageDiT.BlockTap] = []) -> Aicode.Module {
        if !taps.isEmpty { return ZImageBlock.module("modulation", rows: 1, keys: 1, form: .shared, weights: weights, codes: weights.layout, ranks: ranks, taps: taps) }
        var m = ZImageBlock.module("modulation", rows: 1, keys: 1, form: .shared, weights: weights, codes: weights.layout, ranks: ranks)
        let b = m.body
        let (scales, gates) = (m.outputs[0].1, m.outputs[1].1)
        // A row of the block's gains, its page's padding cut (`ZImageBlock.gainRow`): the ANE's surfaces `[1, 1, d]`.
        func row(_ t: Int, _ i: Int) -> Int { b.reshape(b.slice(b.slice(t, axis: 0, start: i, size: 1), axis: 1, start: 0, size: d), [1, 1, d]) }
        m.outputs = [("scale1", row(scales, 0)), ("gate1", row(gates, 1)), ("scale2", row(scales, 2)), ("gate2", row(gates, 3))]
        return m
    }
    static let gains = ["scale1", "gate1", "scale2", "gate2"]

    /// `neural`: the ANE's function of a step both units take (`Block.Build.neural`): on `integer` codes, its products
    /// q64 where the GPU's multiply int8 × int8 (`Aicode.Module.product`).
    static func module(_ function: String, rows: Int, keys: Int, weights: SharedWeights, ranks: [String: Int] = [:],
                       taps: [ZImageDiT.BlockTap] = [], neural: Bool = false) -> Aicode.Module {
        var m = Aicode.Module(arguments: arguments(function, rows: rows, keys: keys, weights: weights, ranks: ranks).map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        let b = m.body
        let names = m.argumentNames
        func a(_ name: String) -> Int { names.firstIndex(of: name)! }
        var constants = Aicode.ScalingConstants()
        /// The product of a unit (rotated before codes), its low-rank path beside it reading `input` (the unit before
        /// its rotation, under the same factor).
        func product(_ x: Int, _ name: String, input: @autoclosure () -> Int) -> Int {
            m.product(x, name, input: input(), adapted: (ranks[name] ?? 0) > 0, weights: weights, neural: neural, constants: &constants)
        }
        func rotation(_ t: Int) -> Int { b.rotation(t, of: weights) }
        /// `t` under a factor `[1, rows, 1]` f16, a power of two: exact.
        func under(_ t: Int, _ f: Int) -> Int { b.binary(.divide, t, f) }
        var tapped: [(String, Int)] = []
        /// A true value in f32, `[rows, n]`: `t` times its factors (each `[1, rows, 1]` or `[rows, 1]`).
        func tap(_ t: ZImageDiT.BlockTap, _ value: @autoclosure () -> Int, _ factors: [Int] = []) {
            guard taps.contains(t) else { return }
            let v = value(), shape = Aicode.shape(b.types[v]).shape, n = shape.reduce(1, *) / rows
            var wide = b.castUnlessSame(v, to: .f32)
            for f in factors { wide = b.binary(.mul, wide, b.cast(f, to: .f32)) }
            tapped.append((t.rawValue, b.reshape(wide, [rows, n])))
        }
        func target(_ names: [String]) -> Int { a(Self.target(names)) }

        switch function {
        case "pre":
            let n1 = b.binary(.mul, b.norm(a("x"), eps: eps, factors: [a("x_factor")]), a("norm1"))
            let n = b.binary(.mul, n1, a("scale1"))
            let (unit, f) = b.ranged16(rotation(n), target: target(["qkv"]))
            let qkv = product(unit, "qkv", input: under(n, f))
            m.outputs = [("qkv", qkv), ("qkv_factor", f)]
            tap(.norm1, n1); tap(.attentionIn, n)
            for (i, t) in [ZImageDiT.BlockTap.q, .k, .v].enumerated() { tap(t, b.slice(qkv, axis: 2, start: i * d, size: d), [f]) }
        case "heads":
            // The guarded pre past its product: the GPU's. The RoPE in f16, on values the head norm bounds: in f32, heads
            // took 0.28 s of an evaluation at 1024², in f16 0.18; the DiT at 512² 1.596e-3 from the fp64 witness against
            // 1.578e-3 (the W8A8's 1.78e-2), measured 19/11 on the mini.
            let all = b.reshape(a("qkv"), [rows, 3 * d]), f = b.cast(b.reshape(a("qkv_factor"), [rows, 1]), to: .f32)
            func third(_ i: Int) -> Int { b.reshape(b.slice(all, axis: 1, start: i * d, size: d), [rows, heads, hd]) }
            let e = b.reshape(b.epsilon(f, eps), [rows, 1, 1])
            func headNormed(_ t: Int, _ gain: String) -> Int { b.binary(.mul, b.rmsNorm(t, axes: [2], epsilon: e), a(gain)) }
            let (cos, sin) = (b.cast(a("cos"), to: .f16), b.cast(a("sin"), to: .f16))
            func rope(_ t: Int) -> Int {
                let pairs = b.reshape(t, [rows, heads, hd / 2, 2])
                let swapped = b.reshape(b.concat([b.slice(pairs, axis: 3, start: 1, size: 1), b.slice(pairs, axis: 3, start: 0, size: 1)], axis: 3),
                                        [rows, heads, hd])
                return b.binary(.add, b.binary(.mul, t, cos), b.binary(.mul, swapped, sin))
            }
            func headsFirst(_ t: Int) -> Int { b.reshape(b.transpose(t, [1, 0, 2]), [1, heads, rows, hd]) }
            // The keys and values token first, a call's a block of the sequence's (`ZImageBlock`).
            func tokenFirst(_ t: Int) -> Int { b.reshape(t, [1, rows, heads, hd]) }
            let qn = headNormed(third(0), "q_norm")
            let q = headsFirst(rope(qn))
            let kn = headNormed(third(1), "k_norm")
            m.outputs = [("q", q), ("k", tokenFirst(rope(kn))), ("v", tokenFirst(b.binary(.mul, third(2), b.reshape(b.cast(f, to: .f16), [rows, 1, 1]))))]
            tap(.qNormed, qn); tap(.kNormed, kn)
        case "attention":
            let o = b.attention([a("q"), b.transpose(a("k"), [0, 2, 1, 3]), b.transpose(a("v"), [0, 2, 1, 3]), a("mask")], scale: 1 / Double(hd).squareRoot(), symbol: "sdpa_tile")
            m.outputs = [("o", b.reshape(b.transpose(b.reshape(o, [heads, rows, hd]), [1, 0, 2]), [1, rows, d]))]
            tap(.sdpa, m.outputs[0].1)
        case "out":
            let (u1, f1) = b.ranged16(rotation(a("o")), target: target(["out"]))
            let attention = product(u1, "out", input: under(a("o"), f1))
            let n2 = b.binary(.mul, b.norm(attention, eps: eps, factors: [f1]), a("norm2"))
            let (x1, x1f) = b.residual16(a("x"), a("x_factor"), plus: b.binary(.mul, a("gate1"), n2))
            m.outputs = [("y", x1), ("y_factor", x1f)]
            tap(.attention, attention, [f1]); tap(.norm2, n2)
        default:
            let (x1, x1f) = (a("x"), a("x_factor"))
            let normed = b.binary(.mul, b.norm(x1, eps: eps, factors: [x1f]), a("ffn_norm1"))
            let n3 = b.binary(.mul, normed, a("scale2"))
            let (r, f2) = b.ranged16(rotation(n3), target: target(["w1", "w3"]))
            lazy var ffnInput = under(n3, f2)
            let w1 = product(r, "w1", input: ffnInput), w3 = product(r, "w3", input: ffnInput)
            // SiLU(w₁ F₂) · w₃ F₂ = F₂² g₁ g₃ · (w₁/g₁) σ(w₁ F₂) (w₃/g₃), each under 2⁵: their product under 2¹⁰,
            // rotated under 2¹⁴. σ reads the true value clipped to ±16, where σ is 1 or 0 in f16.
            let (u1w, g1) = b.ranged16(w1, under: 5), (u3w, g3) = b.ranged16(w3, under: 5)
            let z = b.scale("minimum", b.scale("maximum", b.binary(.mul, w1, f2), -16), 16)
            let swiglu = b.binary(.mul, b.binary(.mul, u1w, b.sigmoid(z)), u3w)
            let (h, g4) = b.ranged16(rotation(swiglu), target: target(["w2"]))
            let ff = product(h, "w2", input: under(swiglu, g4))
            let n4 = b.binary(.mul, b.norm(ff, eps: eps, factors: [f2, f2, g1, g3, g4]), a("ffn_norm2"))
            let (y, yf) = b.residual16(x1, x1f, plus: b.binary(.mul, a("gate2"), n4))
            m.outputs = [("y", y), ("y_factor", yf)]
            tap(.ffnNorm1, normed); tap(.ffnIn, n3); tap(.w1, w1, [f2]); tap(.w3, w3, [f2]); tap(.swiglu, swiglu, [f2, f2, g1, g3])
            tap(.feedForward, ff, [f2, f2, g1, g3, g4]); tap(.ffnNorm2, n4)
        }
        if !taps.isEmpty { m.outputs = taps.map { t in tapped.first { $0.0 == t.rawValue }! } }
        return m
    }
}
