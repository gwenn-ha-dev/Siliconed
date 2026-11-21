import Foundation
import SiliCore

/// **A Z-Image block in three functions**, as diffusers runs it — the Standard's (bf16), and the Compact's modulation:
///
///     modulation   one row    m = adaLN(adaln) + b;  scales 1 + m, gates tanh m, `[4, d]` each: rows s₁ g₁ s₂ g₂
///     pre          rows       h = RMS(x)·γ₁·(1 + s₁);  q k v = the thirds of h · Wqkvᵀ;  q, k: RoPE(RMS_head · γ)
///     post         rows       x₁ = x + tanh g₁ · RMS(sdpa(q, K, V) · Woᵀ)·γ₂;  y = x₁ + the feed-forward half
///
/// Every call of pre runs before any of post: post's attention reads the keys and values of the whole sequence,
/// written by each call of pre, token first (`[1, keys, 30, 128]`). The keys past the sequence's length are masked by an
/// additive −10⁴ (`[1, keys]`): queries against all keys cost what the real keys do (measured 17/11 on the M1 Pro: 1024
/// queries against 4608 keys of which 4224 real, 21.8 ms; against the 4224 alone, 22.7; the composite holds +11 MiB,
/// no matrix of scores).
///
/// **bf16**, the Standard, is diffusers' arithmetic: the weights bf16, the norms' statistics in f32 rounded once, the
/// RoPE in f32, the residual stream bf16. The Compact's block is the shared forms' (`ZImageShared`) on the GPU alone or
/// beside the ANE; its modulation is this one's under the guard of f16 (`Guard.swift`): W8A16 on the Compact's codes,
/// adaln rotated by the file's Hadamard (`convrot`) under its factor; on the Standard's weights in f16 beside the ANE
/// (`SharedWeights.half`), adaln as it is, under the weight's target (the edges' guard, `ZImageFunctions`).
///
/// **A LoRA beside each product** (`LowRank`, `ranks` by product, `adapted`): `<product>_down` `[in, R]` and
/// `<product>_up` `[R, out]`, bf16 under bf16, f16 in the Compact's modulation — each product's own input before its
/// rotation, under the same factor as the rotated one. A rank of 0, no argument: the function of no LoRA, the same to
/// the byte.
///
/// `taps`: the oracle's inner boundaries of the block (`ZImageDiT.BlockTap`) as its outputs in place of its own, true
/// values in f32 — what `siliconed check z-image block` reads; another asset. Six at most a function.
enum ZImageBlock {
    typealias Argument = ZImageFunctions.Argument
    static let c = ZImageDiT.Config.published
    static var d: Int { c.dim }
    static var heads: Int { c.heads }
    static var hd: Int { c.dim / c.heads }
    static var hidden: Int { c.hidden }
    static var adalnDim: Int { c.adalnDim }
    static var eps: Double { Double(c.eps) }
    static let functions = ["modulation", "pre", "post"]
    /// A row of `scales` and `gates`, `[4, gainRow]`: each starts a page of its own, its readers taking one row each, a
    /// buffer of its own (`Layout.pageRow`).
    static var gainRow: Int { Layout.pageRow(d, .bfloat16) }

    /// The products a LoRA adapts, by the function that computes each (`ZImageDiT.products`' names).
    static let adapted: [String: [String]] = ["modulation": ["ada"], "pre": ["qkv"], "post": ["out", "w1", "w3", "w2"]]

    /// `<product>_down` `[in, R]` and `<product>_up` `[R, out]` for each product of `function` (`adapted`) a rank adapts.
    static func lowRankArguments(_ function: String, ranks: [String: Int], element: Aicode.Element) -> [Argument] {
        let shapes = Dictionary(uniqueKeysWithValues: ZImageDiT.products("layers.0").map { ($0.name, (inn: $0.inn, out: $0.out)) })
        return (adapted[function] ?? []).flatMap { p -> [Argument] in
            guard let r = ranks[p], r > 0, let s = shapes[p] else { return [] }
            return [(p + "_down", [s.inn, r], element), (p + "_up", [r, s.out], element)]
        }
    }

    /// The function's arguments in the order it declares them; the Compact's codes lie in `codes` (`CodesLayout`).
    static func arguments(_ function: String, rows: Int, keys: Int, form: ZImageFunctions.Form, weights: SharedWeights = .codes,
                          codes: CodesLayout = .inOut, ranks: [String: Int] = [:]) -> [Argument] {
        let e = form.element
        func product(_ name: String, _ k: Int, _ n: Int) -> [Argument] {
            if form == .bf16 { return [(name, [k, n], .bf16)] }
            return weights == .half ? [(name, [k, n], .f16)] : [(name, codes == .inOut ? [k, n] : [n, k], .si8), (name + "_scale", [n], .f16)]
        }
        var all: [Argument]
        switch function {
        case "modulation":
            all = [("adaln", [1, adalnDim], .f32)] + product("ada", adalnDim, 4 * d) + [("ada_bias", [4 * d], form == .bf16 ? .f32 : .f16)]
        case "pre":
            all = [("x", [rows, d], e), ("scale1", [1, d], e), ("norm1", [d], e)] + product("qkv", d, 3 * d)
                + [("q_norm", [hd], e), ("k_norm", [hd], e), ("cos", [rows, 1, hd], .f32), ("sin", [rows, 1, hd], .f32)]
        default:
            all = [("x", [rows, d], e), ("q", [1, heads, rows, hd], e), ("k", [1, keys, heads, hd], e), ("v", [1, keys, heads, hd], e),
                   ("mask", [1, keys], e), ("gate1", [1, d], e), ("norm2", [d], e)] + product("out", d, d)
                + [("scale2", [1, d], e), ("gate2", [1, d], e), ("ffn_norm1", [d], e), ("ffn_norm2", [d], e)]
                + product("w1", d, hidden) + product("w3", d, hidden) + product("w2", hidden, d)
        }
        return all + lowRankArguments(function, ranks: ranks, element: e)
    }

    /// `weights`: the shared modulation's (`SharedWeights`); nothing to the bf16 form.
    static func module(_ function: String, rows: Int, keys: Int, form: ZImageFunctions.Form, weights: SharedWeights = .codes,
                       codes: CodesLayout = .inOut, ranks: [String: Int] = [:], taps: [ZImageDiT.BlockTap] = []) -> Aicode.Module {
        precondition(taps.count <= 6, "six outputs at most")
        precondition(form == .bf16 || function == "modulation", "the Compact's block is `ZImageShared`'s")
        let arguments = arguments(function, rows: rows, keys: keys, form: form, weights: weights, codes: codes, ranks: ranks)
        var m = Aicode.Module(arguments: arguments.map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        let b = m.body
        let names = m.argumentNames
        func a(_ name: String) -> Int { names.firstIndex(of: name)! }
        var constants = Aicode.ScalingConstants()
        /// The product of `x` and its low-rank path beside it, reading `input`: under bf16 x itself; in the Compact's
        /// modulation x before its rotation, under the same factor.
        func product(_ x: Int, _ name: String, input: @autoclosure () -> Int? = nil) -> Int {
            let y = form == .bf16 ? b.matmul(x, a(name), result: Aicode.tensor([Aicode.shape(b.types[x]).shape[0], Aicode.shape(b.types[a(name)]).shape[1]], .bf16))
                : weights == .half ? b.product(x, a(name))
                : b.w8a16(x, codes: a(name), scale: a(name + "_scale"), layout: codes, constants: &constants)
            return (ranks[name] ?? 0) > 0 ? b.lowRank(y, input() ?? x, down: a(name + "_down"), up: a(name + "_up")) : y
        }
        func target(_ names: [String]) -> Int { b.target(codes: names.map { (a($0), a($0 + "_scale")) }, layout: codes) }
        func normed(_ x: Int, _ gain: String, axis: Int) -> Int { b.binary(.mul, b.rmsNorm(x, axes: [axis], eps: eps), a(gain)) }
        // RoPE on interleaved pairs in f32, rounded once (diffusers' `apply_rotary_emb`): x · cos + swap(x) · sin, the
        // tables `[rows, 1, 128]` holding each pair's angle twice and the sine's sign: (−sin, sin).
        func rope(_ t: Int) -> Int {
            let f = b.cast(t, to: .f32)
            let pairs = b.reshape(f, [rows, heads, hd / 2, 2])
            let swapped = b.reshape(b.concat([b.slice(pairs, axis: 3, start: 1, size: 1), b.slice(pairs, axis: 3, start: 0, size: 1)], axis: 3),
                                    [rows, heads, hd])
            return b.cast(b.binary(.add, b.binary(.mul, f, a("cos")), b.binary(.mul, swapped, a("sin"))), to: form.element)
        }
        func headsFirst(_ t: Int) -> Int { b.reshape(b.transpose(t, [1, 0, 2]), [1, heads, rows, hd]) }
        // The keys and values token first, `[1, rows, 30, 128]`: a call's a block of the sequence's, written whole, and the
        // attention takes their heads first inside (measured 19/11, Z at 1024²: the write 1.41 → 0.64 ms through a view
        // with gaps, the attention 4.18 → 4.21, the same bits).
        func tokenFirst(_ t: Int) -> Int { b.reshape(t, [1, rows, heads, hd]) }
        func keysHeadsFirst(_ t: Int) -> Int { b.transpose(t, [0, 2, 1, 3]) }
        /// A tap: a true value, `[rows, n]` (or `[rows, 30, 128]`) in f32, times its factor when it has one. Nothing of
        /// it is written without `taps`: the functions a render loads are the same to the byte.
        var tapped: [(String, Int)] = []
        func tap(_ t: ZImageDiT.BlockTap, _ v: @autoclosure () -> Int, factor f: @autoclosure () -> Int? = nil) {
            guard taps.contains(t) else { return }
            let wide = b.castUnlessSame(v(), to: .f32)
            tapped.append((t.rawValue, f().map { b.binary(.mul, wide, $0) } ?? wide))
        }
        func attended() -> Int {
            let o = b.attention([a("q"), keysHeadsFirst(a("k")), keysHeadsFirst(a("v")), a("mask")], scale: 1 / Double(hd).squareRoot(), symbol: "sdpa_tile")
            return b.reshape(b.transpose(b.reshape(o, [heads, rows, hd]), [1, 0, 2]), [rows, d])
        }

        switch (function, form) {
        case ("modulation", .bf16):
            // In f32, rounded once: diffusers rounds m, then 1 + m and tanh m, each in bf16.
            var ada = b.matmul(a("adaln"), b.cast(a("ada"), to: .f32), result: Aicode.tensor([1, 4 * d], .f32))
            if (ranks["ada"] ?? 0) > 0 { ada = b.lowRank(ada, a("adaln"), down: a("ada_down"), up: a("ada_up")) }
            let m6 = b.reshape(b.binary(.add, ada, a("ada_bias")), [4, d])
            tap(.adaln, b.reshape(m6, [1, 4 * d]))
            m.outputs = [("scales", b.zeroPadded(b.cast(b.binary(.add, b.scalar(1, .f32), m6), to: .bf16), to: [4, gainRow])),
                         ("gates", b.zeroPadded(b.cast(b.tanh(m6), to: .bf16), to: [4, gainRow]))]
        case ("modulation", _):
            // Multiplied out: 1 + m and tanh m read the true value. The codes' input rotated (`convrot`), the halves' as it is.
            let codes = weights == .codes
            let (unit, f) = b.ranged(codes ? b.convrot(a("adaln")) : a("adaln"), under: codes ? target(["ada"]) : b.target(weights: [a("ada")]))
            let m6 = b.reshape(b.binary(.add, b.binary(.mul, product(unit, "ada", input: codes ? b.binary(.divide, a("adaln"), f) : unit), b.cast(f, to: .f16)), a("ada_bias")), [4, d])
            tap(.adaln, b.reshape(m6, [1, 4 * d]))
            m.outputs = [("scales", b.zeroPadded(b.binary(.add, b.scalar(1, .f16), m6), to: [4, gainRow])), ("gates", b.zeroPadded(b.tanh(m6), to: [4, gainRow]))]
        case ("pre", .bf16):
            let n1 = normed(a("x"), "norm1", axis: 1)
            let h = b.binary(.mul, n1, a("scale1"))
            let all = product(h, "qkv")
            func third(_ i: Int) -> Int { b.reshape(b.slice(all, axis: 1, start: i * d, size: d), [rows, heads, hd]) }
            let (qn, kn) = (normed(third(0), "q_norm", axis: 2), normed(third(1), "k_norm", axis: 2))
            tap(.norm1, n1); tap(.attentionIn, h)
            tap(.q, b.reshape(third(0), [rows, d])); tap(.k, b.reshape(third(1), [rows, d])); tap(.v, b.reshape(third(2), [rows, d]))
            tap(.qNormed, qn); tap(.kNormed, kn)
            m.outputs = [("q", headsFirst(rope(qn))), ("k", tokenFirst(rope(kn))), ("v", tokenFirst(third(2)))]
        case (_, .bf16):
            let o = attended()
            let attention = product(o, "out")
            let n2 = normed(attention, "norm2", axis: 1)
            let x1 = b.binary(.add, a("x"), b.binary(.mul, a("gate1"), n2))
            let n3 = normed(x1, "ffn_norm1", axis: 1)
            let r = b.binary(.mul, n3, a("scale2"))
            let w1 = product(r, "w1"), w3 = product(r, "w3")
            let swiglu = b.binary(.mul, b.silu(w1), w3)
            let ff = product(swiglu, "w2")
            let n4 = normed(ff, "ffn_norm2", axis: 1)
            tap(.sdpa, o); tap(.attention, attention); tap(.norm2, n2); tap(.ffnNorm1, n3); tap(.ffnIn, r)
            tap(.w1, w1); tap(.w3, w3); tap(.swiglu, swiglu); tap(.feedForward, ff); tap(.ffnNorm2, n4)
            m.outputs = [("y", b.binary(.add, x1, b.binary(.mul, a("gate2"), n4)))]
        default:
            preconditionFailure("\(function) of the Compact: `ZImageShared`'s")
        }
        if !taps.isEmpty { m.outputs = taps.map { t in tapped.first { $0.0 == t.rawValue }! } }
        return m
    }
}
