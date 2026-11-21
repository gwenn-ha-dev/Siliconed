import CoreAI
import Foundation
import SiliCore

/// **A Qwen-Image-2.1 layer as the GPU and the ANE share it** (Z's `ZImageShared`, the engine's forms in
/// `Neural.swift`): f16 at every border and inside, `[1, rows, n]`, q64 products on the codes `[in, out]`, norms by their
/// peak, the guard's powers of two in f16; what reads across rows or computes in f32 is the GPU's:
///
///     pre          both   n = LN(x̂ · X) · (1 + s₁), rotated, under 2^e₀ by F: q̂, k̂, v̂ = q64, and F
///     heads        GPU    q, k: RoPE(RMS_head(q̂) · γ) (eps / F², `headNorm`) in f32; v = v̂ · F; heads first, into the pass's keys
///     attention    GPU    o = sdpa(q, [kₚ | kₜ], [vₚ | vₜ], mask), rows first: a true value
///     out          both   x₁ = x + tanh g₁ · to_out(o R), the stream kept as (x̂, X)
///     ffn          both   y = x₁ + tanh g₂ · down(SiLU(gate) · up), gate, up = each half of gate_up on LN(x₁) · (1 + s₂) R
///
/// Out and ffn are two functions, as Z's: the ANE failed Z's post whole (code −19, measured 17/11 on the M1 Pro).
/// Qwen-Image-2.1 has no sandwich norm: an update enters the stream as (Δ̂, G) (`residual16(_:_:plus:factors:)`).
///
/// **The LayerNorm**, without its affine, is the engine's (`layerNorm16`): x̂ lies within 2¹⁴ (the stream's bound), so its
/// mean along a row of 4096 stays in f16 on either unit.
///
/// **Under a LoRA**, each product's change is computed beside it in f16, the ANE's element (`Aicode.Module.product`, the
/// factors balanced by `LowRank.write`), on its input before the rotation over the factor F the rotated input was
/// ranged by: the change then lies under F as the product's result does. That input lies within 2¹⁰ (the rotation of a
/// block of 256 moves a peak by 16 at most, and the published codes' targets are 2⁶ at most).
enum QwenImage21Shared {
    static let c = QwenImage21DiT.Config.published
    static var d: Int { c.dim }
    static var hd: Int { c.headDim }
    static var heads: Int { c.heads }
    typealias Argument = QwenImage21Functions.Argument

    /// Each product's target, an argument `[1, 1, 1]` f16 computed as the weights are read (`Recipe.target`): the ANE
    /// computed it wrong inside a shared feed-forward (Z's, measured 18/11 on the M5 Pro).
    static let targets: [(name: String, products: [String])] = [("qkv_target", ["to_q", "to_k", "to_v"]), ("out_target", ["to_out"]),
                                                                ("gate_up_target", ["gate_up"]), ("down_target", ["down"])]
    /// The modulation's rows, each a tensor `[1, 1, d]` (`timestep`'s outputs in the shared forms).
    static let gains = ["scale1", "gate1", "scale2", "gate2"]

    static func arguments(_ f: String, rows: Int, prefix: Int, target: Int, weights: SharedWeights) -> [Argument] {
        let (d, hd, h, hidden) = (Self.d, Self.hd, Self.heads, c.hidden)
        let row = [1, 1, d], one = [1, 1, 1]
        func stream(_ name: String) -> [Argument] { [(name, [1, rows, d], .f16), (name + "_factor", [1, rows, 1], .f16)] }
        func weight(_ name: String, _ inn: Int, _ out: Int) -> [Argument] { weights.arguments(name, inn: inn, out: out) }
        switch f {
        case "pre":
            return stream("x") + [("scale1", row, .f16)] + weight("to_q", d, d) + weight("to_k", d, d) + weight("to_v", d, d)
                + [("qkv_target", one, .f16)]
        case "heads":
            return [("qh", [1, rows, d], .f16), ("kh", [1, rows, d], .f16), ("vh", [1, rows, d], .f16), ("qkv_factor", [1, rows, 1], .f16),
                    ("norm_q", [hd], .f16), ("norm_k", [hd], .f16), ("cos", [rows, 1, hd], .f32), ("sin", [rows, 1, hd], .f32)]
        case "attention":
            return [("q", [1, h, rows, hd], .f16), ("kp", [1, prefix, h, hd], .f16), ("vp", [1, prefix, h, hd], .f16),
                    ("kt", [1, target, h, hd], .f16), ("vt", [1, target, h, hd], .f16), ("mask", [rows, prefix + target], .f16)]
        case "out":
            return stream("x") + [("o", [1, rows, d], .f16), ("gate1", row, .f16)] + weight("to_out", d, d) + [("out_target", one, .f16)]
        case "ffn":
            return stream("x") + [("scale2", row, .f16), ("gate2", row, .f16)] + weight("gate_up", d, 2 * hidden) + weight("down", hidden, d)
                + [("gate_up_target", one, .f16), ("down_target", one, .f16)]
        default:
            preconditionFailure("\(f): no shared function")
        }
    }

    static func module(_ f: String, rows: Int, prefix: Int, target: Int, weights: SharedWeights, ranks: QwenImage21Functions.Ranks = [:]) -> Aicode.Module {
        let all = arguments(f, rows: rows, prefix: prefix, target: target, weights: weights) + QwenImage21Functions.factors(f, path: .shared, ranks: ranks)
        var m = Aicode.Module(arguments: all.map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        let b = m.body
        let names = m.argumentNames
        func a(_ name: String) -> Int { names.firstIndex(of: name)! }
        let (d, hd, h, eps) = (Self.d, Self.hd, Self.heads, c.eps)
        var constants = Aicode.ScalingConstants()
        /// The product `name` on x, the rotation of `unrotated` over its factor, with its LoRA's change when one adapts it.
        func product(_ x: Int, _ name: String, unrotated: Int, columns: Range<Int>? = nil) -> Int {
            m.product(x, name, input: unrotated, adapted: (ranks[name] ?? 0) > 0, columns: columns, constants: &constants)
        }
        func layerNorm(_ x: Int, _ xf: Int) -> Int { b.layerNorm16(x, xf, eps: eps) }
        func rotation(_ t: Int) -> Int { b.rotation(t, of: weights) }

        switch f {
        case "pre":
            let n = b.binary(.mul, layerNorm(a("x"), a("x_factor")), a("scale1"))
            let (unit, factor) = b.ranged16(rotation(n), target: a("qkv_target"))
            let x = b.binary(.divide, n, factor)
            m.outputs = [("qh", product(unit, "to_q", unrotated: x)), ("kh", product(unit, "to_k", unrotated: x)),
                         ("vh", product(unit, "to_v", unrotated: x)), ("qkv_factor", factor)]
        case "heads":
            // The GPU's half path past its products, in f32 where it was.
            func split(_ t: Int) -> Int { b.reshape(t, [rows, h, hd]) }
            func headNormed(_ t: Int, _ gain: String) -> Int { b.headNorm(t, factor: a("qkv_factor"), gain: a(gain), eps: eps, heads: h) }
            func rope(_ x: Int) -> Int {
                let pairs = b.reshape(x, [rows, h, hd / 2, 2])
                let swapped = b.reshape(b.concat([b.slice(pairs, axis: 3, start: 1, size: 1), b.slice(pairs, axis: 3, start: 0, size: 1)], axis: 3),
                                        [rows, h, hd])
                return b.cast(b.binary(.add, b.binary(.mul, x, a("cos")), b.binary(.mul, swapped, a("sin"))), to: .f16)
            }
            func headsFirst(_ t: Int) -> Int { b.reshape(b.transpose(t, [1, 0, 2]), [1, h, rows, hd]) }
            /// The keys and values token first (`QwenImage21Functions`').
            func tokenFirst(_ t: Int) -> Int { b.reshape(t, [1, rows, h, hd]) }
            let v = b.binary(.mul, split(a("vh")), b.reshape(a("qkv_factor"), [rows, 1, 1]))
            m.outputs = [("q", headsFirst(rope(headNormed(a("qh"), "norm_q")))), ("k", tokenFirst(rope(headNormed(a("kh"), "norm_k")))),
                         ("v", tokenFirst(v))]
        case "attention":
            let keys = b.transpose(b.concat([a("kp"), a("kt")], axis: 1), [0, 2, 1, 3]), values = b.transpose(b.concat([a("vp"), a("vt")], axis: 1), [0, 2, 1, 3])
            let o = b.attention([a("q"), keys, values, a("mask")], scale: 1 / Double(hd).squareRoot(), symbol: "sdpa_tile")
            m.outputs = [("o", b.reshape(b.transpose(b.reshape(o, [h, rows, hd]), [1, 0, 2]), [1, rows, d]))]
        case "out":
            let (unit, f1) = b.ranged16(rotation(a("o")), target: a("out_target"))
            let attended = product(unit, "to_out", unrotated: b.binary(.divide, a("o"), f1))
            let (y, yf) = b.residual16(a("x"), a("x_factor"), plus: b.binary(.mul, a("gate1"), attended), factors: [f1])
            m.outputs = [("y", y), ("y_factor", yf)]
        default:
            let n = b.binary(.mul, layerNorm(a("x"), a("x_factor")), a("scale2"))
            let (r, f2) = b.ranged16(rotation(n), target: a("gate_up_target"))
            // The gate and up, each a product of its own on its half of the fused weight: the ANE's compiler refuses
            // the whole product sliced after (`Aicode.Module.product`'s `columns`).
            let input = b.binary(.divide, n, f2)
            let w1 = product(r, "gate_up", unrotated: input, columns: 0..<c.hidden)
            let w3 = product(r, "gate_up", unrotated: input, columns: c.hidden..<(2 * c.hidden))
            // SiLU(w₁ F₂) · w₃ F₂ = F₂² g₁ g₃ · (w₁/g₁) σ(w₁ F₂) (w₃/g₃), each under 2⁵ (the half path's): their product
            // under 2¹⁰, rotated under 2¹⁴. σ reads the true value clipped to ±16, where σ is 1 or 0 in f16.
            let (u1, g1) = b.ranged16(w1, under: 5), (u3, g3) = b.ranged16(w3, under: 5)
            let z = b.scale("minimum", b.scale("maximum", b.binary(.mul, w1, f2), -16), 16)
            let gated = b.binary(.mul, b.binary(.mul, u1, b.sigmoid(z)), u3)
            let (hidden, g4) = b.ranged16(rotation(gated), target: a("down_target"))
            let delta = b.binary(.mul, a("gate2"), product(hidden, "down", unrotated: b.binary(.divide, gated, g4)))
            let (y, yf) = b.residual16(a("x"), a("x_factor"), plus: delta, factors: [f2, f2, g1, g3, g4])
            m.outputs = [("y", y), ("y_factor", yf)]
        }
        return m
    }
}
