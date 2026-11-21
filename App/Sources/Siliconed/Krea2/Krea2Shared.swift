import Foundation
import SiliCore

/// **A Krea 2 block as the GPU and the ANE share it** (Z-Image's design, `ZImageShared`): the Compact's block cut where
/// the ANE can take it, written in the forms both units run (`Neural.swift`: f16 at every border and inside, `[1, rows,
/// n]`, q64 products on the codes `[in, out]`, norms by their peak, every factor a power of two `[1, rows, 1]` in f16);
/// what reads across rows or computes in f32 is the GPU's:
///
///     modulation   GPU    temb_mod + mod.lin: (1 + prescale), preshift, pregate, (1 + postscale), postshift, postgate,
///                         each `[1, 1, d]`, each an output (a surface of its own for the ANE)
///     pre          both   n = RMS(x̂ · X) · (1 + γ₁) · s₁ + t₁, rotated, brought under 2^e₀ by F: q̂, k̂, v̂, ĝ = q64, and F
///     heads        GPU    q, k: RMS_head(q̂, k̂) · (1 + γ) (eps / F²), RoPE in f32, heads first into the sequence's rows;
///                         v = v̂ · F; σ(ĝ · F) the attention's gate
///     attention    GPU    o = sdpa(q, K, V, mask) · σ, rows first, the four query heads of a key head one of 4r rows
///     out          both   x₁ = x + g₁ · (o R · Wo), the stream kept as (x̂, X)
///     ffn          both   y = x₁ + g₂ · down(SiLU(gate h) · up h), h = RMS(x₁) · (1 + γ₂) · s₂ + t₂
///
/// Out and ffn are two functions as Z-Image's: the ANE fails to run its post as one (code −19, measured 17/11 on the M1
/// Pro). Where Z-Image's block normalizes after each product, Krea 2's adds the product's true value: the feed-forward's
/// is p · (F₂ g₁) · (F₂ g₃) · g₄, each pair a small factor by a large one, so that no partial product leaves f16.
///
/// **A LoRA beside a product** (`ranks`, by argument: `LoRA.swift`): `<name>_down` `[1, in, R]` and `<name>_up` `[1, R, out]`
/// in f16, `(x / F · down) · up` added to the product's result, x the product's input before its rotation (the factors
/// are in the model's own basis) under the same factor F as the codes read it: the change lands in the result's units.
/// `Aicode.Block.lowRank`'s formula in the shared forms' `[1, rows, n]`.
///
/// **Two weights** (`SharedWeights`): the Compact's codes, each product's input rotated (q64); the Standard's bf16
/// rounded to f16 as they are read, the input as it is (`product16`) — the Standard beside the ANE.
enum Krea2Shared {
    typealias Argument = Krea2Functions.Argument
    static let d = Krea2Functions.d, hd = Krea2Functions.hd, heads = Krea2Functions.heads, kvHeads = Krea2Functions.kvHeads
    static let kv = kvHeads * hd, hidden = Krea2Functions.hidden, eps = Krea2Functions.eps
    static let gains = ["s1", "t1", "g1", "s2", "t2", "g2"]

    /// Each product's target, an argument computed as the weights are read (`Recipe.target`, the ANE's lesson of 18/11),
    /// and the products whose codes it reads.
    static let targets: [(name: String, products: [String])] = [("pre_target", ["wq", "wk", "wv", "gate"]), ("wo_target", ["wo"]),
                                                                ("w13_target", ["mgate", "up"]), ("w2_target", ["down"])]

    /// The arguments of `function` over `rows`, `keys` the attention's.
    static func arguments(_ function: String, rows: Int, keys: Int, weights: SharedWeights, ranks: [String: Int] = [:]) -> [Argument] {
        let row = [1, 1, d], one = [1, 1, 1]
        let x: [Argument] = [("x", [1, rows, d], .f16), ("x_factor", [1, rows, 1], .f16)]
        func product(_ name: String, _ inn: Int, _ out: Int) -> [Argument] {
            weights.arguments(name, inn: inn, out: out)
                + (ranks[name].map { [(name + "_down", [1, inn, $0], .f16), (name + "_up", [1, $0, out], .f16)] } ?? [])
        }
        switch function {
        case "embed":
            return [("patches", [rows, Krea2Functions.patch], .f32), ("first", [d, Krea2Functions.patch], .bf16), ("first_bias", [d], .bf16)]
        case "final":
            return x + [("temb", [1, d], .bf16), ("last_mod", [2, d], .bf16), ("last_norm", [d], .f32),
                        ("last_linear", [Krea2Functions.patch, d], .bf16), ("last_linear_bias", [Krea2Functions.patch], .bf16)]
        case "modulation":
            return [("temb_mod", [6, d], .bf16), ("mod", [6, d], .bf16)]
        case "pre":
            return x + [("s1", row, .f16), ("t1", row, .f16), ("prenorm", row, .f16)] + product("wq", d, d) + product("wk", d, kv)
                + product("wv", d, kv) + product("gate", d, d) + [("pre_target", one, .f16)]
        case "heads":
            return [("qh", [1, rows, d], .f16), ("kh", [1, rows, kv], .f16), ("vh", [1, rows, kv], .f16), ("gh", [1, rows, d], .f16),
                    ("pre_factor", [1, rows, 1], .f16), ("qnorm", [hd], .f16), ("knorm", [hd], .f16),
                    ("cos", [rows, 1, hd], .f32), ("sin", [rows, 1, hd], .f32)]
        case "attention":
            return [("q", [1, heads, rows, hd], .f16), ("k", [1, keys, kvHeads, hd], .f16), ("v", [1, keys, kvHeads, hd], .f16),
                    ("mask", [1, keys], .f16), ("sg", [1, rows, d], .f16)]
        case "out", "out-taps":
            return x + [("o", [1, rows, d], .f16), ("g1", row, .f16)] + product("wo", d, d) + [("wo_target", one, .f16)]
        case "ffn", "ffn-taps":
            return x + [("s2", row, .f16), ("t2", row, .f16), ("g2", row, .f16), ("postnorm", row, .f16)] + product("mgate", d, hidden)
                + product("up", d, hidden) + product("down", hidden, d) + [("w13_target", one, .f16), ("w2_target", one, .f16)]
        default: preconditionFailure("krea2: no shared function \(function)")
        }
    }

    /// `function` written op by op; `out-taps` and `ffn-taps` (the GPU's, for a check's block 0) also give the attention's
    /// and the feed-forward's true values, `attention` and `feed` `[1, rows, d]` f16.
    static func module(_ function: String, rows: Int, keys: Int, weights: SharedWeights, ranks: [String: Int] = [:]) -> Aicode.Module {
        var m = Aicode.Module(arguments: arguments(function, rows: rows, keys: keys, weights: weights, ranks: ranks).map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        let b = m.body
        let names = m.argumentNames
        func a(_ name: String) -> Int { names.firstIndex(of: name)! }
        var constants = Aicode.ScalingConstants()
        /// The product of `name` on x (rotated, under F), its LoRA beside on `plain`, x before the rotation under F.
        func product(_ x: Int, _ name: String, plain: Int) -> Int {
            let y = weights == .codes ? b.q64(x, codes: a(name), scale: a(name + "_scale"), constants: &constants)
                : b.product16(x, weight: a(name))
            guard let r = ranks[name] else { return y }
            let out = Aicode.shape(b.types[y]).shape[2]
            let t = b.matmul(plain, a(name + "_down"), result: Aicode.tensor([1, rows, r], .f16))
            return b.binary(.add, y, b.matmul(t, a(name + "_up"), result: Aicode.tensor([1, rows, out], .f16)))
        }
        func rotation(_ t: Int) -> Int { b.rotation(t, of: weights) }
        func clamped(_ x: Int, _ c: Double) -> Int { b.scale("minimum", b.scale("maximum", x, -c), c) }
        /// `RMS(x̂ · X) · (1 + γ) · s + t`, in f16.
        func modulated(_ gain: String, _ s: String, _ t: String) -> Int {
            let n = b.norm(a("x"), eps: eps, factors: [a("x_factor")])
            return b.binary(.add, b.binary(.mul, b.binary(.mul, n, b.scale("add", a(gain), 1)), a(s)), a(t))
        }

        switch function {
        case "embed":
            // `first` in bf16 as the other paths, the stream then held as the shared forms hold it: under 2¹⁴.
            let x = b.binary(.add, b.matmul(b.cast(a("patches"), to: .bf16), b.transpose(a("first"), [1, 0]), result: Aicode.tensor([rows, d], .bf16)),
                             a("first_bias"))
            let (u, f) = b.ranged16(b.reshape(b.cast(x, to: .f16), [1, rows, d]), under: 14)
            m.outputs = [("x", u), ("x_factor", f)]

        case "final":
            // The f16 block's final layer, the stream's factor f16 `[1, rows, 1]`.
            let wide = b.cast(b.reshape(a("x"), [rows, d]), to: .f32), f = b.cast(b.reshape(a("x_factor"), [rows, 1]), to: .f32)
            let n = b.binary(.mul, b.rmsNorm(wide, axes: [1], epsilon: b.epsilon(f, eps)), b.scale("add", a("last_norm"), 1))
            let mod = b.binary(.add, a("last_mod"), a("temb"))
            let scale = b.slice(mod, axis: 0, start: 0, size: 1), shift = b.slice(mod, axis: 0, start: 1, size: 1)
            let h = b.binary(.add, b.binary(.mul, b.scale("add", scale, 1), b.cast(n, to: .bf16)), shift)
            let y = b.binary(.add, b.matmul(h, b.transpose(a("last_linear"), [1, 0]), result: Aicode.tensor([rows, Krea2Functions.patch], .bf16)),
                             a("last_linear_bias"))
            m.outputs = [("velocity", b.cast(y, to: .f32))]

        case "modulation":
            // In f32 then f16 once, as the f16 block's `modulated` reads its parts.
            let parts = b.cast(b.binary(.add, a("temb_mod"), a("mod")), to: .f32)
            func part(_ i: Int) -> Int { b.reshape(b.slice(parts, axis: 0, start: i, size: 1), [1, 1, d]) }
            func row(_ i: Int, plusOne: Bool) -> Int { b.cast(plusOne ? b.scale("add", part(i), 1) : part(i), to: .f16) }
            m.outputs = [("s1", row(0, plusOne: true)), ("t1", row(1, plusOne: false)), ("g1", row(2, plusOne: false)),
                         ("s2", row(3, plusOne: true)), ("t2", row(4, plusOne: false)), ("g2", row(5, plusOne: false))]

        case "pre":
            let n = modulated("prenorm", "s1", "t1")
            let (unit, f) = b.ranged16(rotation(n), target: a("pre_target"))
            let plain = b.binary(.divide, n, f)
            m.outputs = [("qh", product(unit, "wq", plain: plain)), ("kh", product(unit, "wk", plain: plain)), ("vh", product(unit, "wv", plain: plain)),
                         ("gh", product(unit, "gate", plain: plain)), ("pre_factor", f)]

        case "heads":
            // The f16 block's pre past its products: the GPU's, in f32 where it was.
            let f = b.cast(b.reshape(a("pre_factor"), [rows, 1]), to: .f32)
            let e = b.reshape(b.epsilon(f, eps), [rows, 1, 1])
            func headNormed(_ t: Int, _ count: Int, _ gain: String) -> Int {
                let h = b.reshape(t, [rows, count, hd])
                return b.binary(.mul, b.rmsNorm(b.cast(h, to: .f32), axes: [2], epsilon: e), b.cast(b.scale("add", a(gain), 1), to: .f32))
            }
            func rope(_ t: Int, _ count: Int) -> Int {
                let pairs = b.reshape(t, [rows, count, hd / 2, 2])
                let swapped = b.reshape(b.concat([b.slice(pairs, axis: 3, start: 1, size: 1), b.slice(pairs, axis: 3, start: 0, size: 1)], axis: 3),
                                        [rows, count, hd])
                return b.cast(b.binary(.add, b.binary(.mul, t, a("cos")), b.binary(.mul, swapped, a("sin"))), to: .f16)
            }
            func headsFirst(_ t: Int, _ count: Int) -> Int { b.reshape(b.transpose(t, [1, 0, 2]), [1, count, rows, hd]) }
            /// The keys and values token first (`Krea2Functions`').
            func tokenFirst(_ t: Int, _ count: Int) -> Int { b.reshape(t, [1, rows, count, hd]) }
            let factor = b.reshape(a("pre_factor"), [rows, 1, 1])
            let v = b.binary(.mul, b.reshape(a("vh"), [rows, kvHeads, hd]), factor)
            // σ reads the gate's true value, clipped to ±16, where σ is 1 or 0 in f16.
            let gate = b.binary(.mul, b.reshape(a("gh"), [rows, d]), b.reshape(a("pre_factor"), [rows, 1]))
            m.outputs = [("q", headsFirst(rope(headNormed(a("qh"), heads, "qnorm"), heads), heads)),
                         ("k", tokenFirst(rope(headNormed(a("kh"), kvHeads, "knorm"), kvHeads), kvHeads)),
                         ("v", tokenFirst(v, kvHeads)), ("sg", b.reshape(b.sigmoid(clamped(gate, 16)), [1, rows, d]))]

        case "attention":
            let q = b.reshape(a("q"), [1, kvHeads, 4 * rows, hd])
            let o = b.attention([q, b.transpose(a("k"), [0, 2, 1, 3]), b.transpose(a("v"), [0, 2, 1, 3]), a("mask")], scale: 1 / Double(hd).squareRoot(), symbol: "sdpa_tile")
            let rowsFirst = b.reshape(b.transpose(b.reshape(o, [heads, rows, hd]), [1, 0, 2]), [1, rows, d])
            m.outputs = [("o", b.binary(.mul, rowsFirst, a("sg")))]

        case "out", "out-taps":
            let (u, f) = b.ranged16(rotation(a("o")), target: a("wo_target"))
            let attention = b.binary(.mul, product(u, "wo", plain: b.binary(.divide, a("o"), f)), f)
            let (y, yf) = b.residual16(a("x"), a("x_factor"), plus: b.binary(.mul, a("g1"), attention))
            m.outputs = [("y", y), ("y_factor", yf)] + (function == "out-taps" ? [("attention", attention)] : [])

        default:
            let n = modulated("postnorm", "s2", "t2")
            let (r, f2) = b.ranged16(rotation(n), target: a("w13_target"))
            let plain = b.binary(.divide, n, f2)
            let w1 = product(r, "mgate", plain: plain), w3 = product(r, "up", plain: plain)
            // SiLU(w₁ F₂) · w₃ F₂ = F₂² g₁ g₃ · (w₁/g₁) σ(w₁ F₂) (w₃/g₃), each under 2⁵ (Z-Image's).
            let (u1, g1) = b.ranged16(w1, under: 5), (u3, g3) = b.ranged16(w3, under: 5)
            let z = clamped(b.binary(.mul, w1, f2), 16)
            let inner = b.binary(.mul, b.binary(.mul, u1, b.sigmoid(z)), u3)
            let (h, g4) = b.ranged16(rotation(inner), target: a("w2_target"))
            let p = product(h, "down", plain: b.binary(.divide, inner, g4))
            let feed = b.binary(.mul, b.binary(.mul, b.binary(.mul, p, b.binary(.mul, f2, g1)), b.binary(.mul, f2, g3)), g4)
            let (y, yf) = b.residual16(a("x"), a("x_factor"), plus: b.binary(.mul, a("g2"), feed))
            m.outputs = [("y", y), ("y_factor", yf)] + (function == "ffn-taps" ? [("feed", feed)] : [])
        }
        return m
    }

    /// A block's weights in the shared forms: the codes `[in, out]` (surfaces) and their scales, or the f16 weights
    /// (`weights`); the norms' gains in f16, the table in bf16, the targets after the products they read.
    static func block(_ prefix: String, weights: SharedWeights) -> [Weights.Entry] {
        return Krea2Functions.products.map { weights.entry($0.0, key: prefix + $0.1) }
            + [("prenorm", prefix + "prenorm.scale", .half), ("postnorm", prefix + "postnorm.scale", .half),
               ("qnorm", prefix + "attn.qknorm.qnorm.scale", .half), ("knorm", prefix + "attn.qknorm.knorm.scale", .half),
               ("mod", prefix + "mod.lin", .bfloat16(joined: []))]
            + targets.map { ($0.name, "", Recipe.target($0.products)) }
    }
}
