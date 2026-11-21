import CoreAI
import Foundation
import SiliCore

/// **A MOSS-Image block as the GPU and the ANE share it** (Z's `ZImageShared`, the engine's forms in `Neural.swift`): f16
/// at every border and inside, `[1, rows, n]`, q64 products on the codes `[in, out]`, norms by their peak, the guard's
/// powers of two in f16; what reads across rows or computes in f32 is the GPU's:
///
///     gains       GPU    the modulation's six rows each its own `[1, 1, d]`: shift, 1 + scale, a gate as ĝ = g / G
///     enter       GPU    the embedders' bf16 rows as the stream (x̂, X), X `[1, rows, 1]`
///     pre         both   n = RMS(x̂ · X) · γ₁ · (1 + s₁) + t₁, rotated, under 2^e₀ by F: qkv̂ = q64, and F
///     heads       GPU    q, k: RMS_head(qkv̂) · γ (eps / F²), RoPE in f32; v = v̂ · F; heads first, into the sequence's keys
///     attention   GPU    o = sdpa(q, K, V, mask), rows first: a true value
///     out         both   x₁ = x + g₁ · (o R) · Wo, the stream kept as (x̂, X)
///     ffn         both   y = x₁ + g₂ · W_down(SITU(h, g) R), [h | g] = W_up_gate(RMS(x₁) · γ₂ · (1 + s₂) + t₂) R
///     final       GPU    the Compact's head, on the stream (x̂, X)
///
/// What MOSS asks beyond Z's forms:
///   - **its gates are unbounded** (no tanh; 8.75 and 30.5 at σ = 1, the goldens' `shared_modulation`): each is held as
///     (ĝ, G), max |ĝ| ≤ 1, G the power of two the loop reads off the modulation's row (a function returns six outputs at
///     most, `Run.call`), so that ĝ times a product's result under its target stays in f16;
///     the update enters the stream as (Δ̂, F · G) (`residual16(_:_:plus:factors:)`): there is no sandwich norm to bound it;
///   - **the norm's input is modulated with a shift** and its gains reach 12.75 (`norm1`, the 42 blocks' largest): n is
///     brought under 2¹⁰ before its rotation (16 × at most), not rotated as it is;
///   - **SITU** reads its two halves true: h · F clipped to ±128 (25 tanh(h / 25) is 25 in f16 past it), g · F to ±32
///     (σ and tanh(g / 4) are 1 past it); its result within ±100 by construction, rotated under 2¹⁴.
///
/// **Two weights** (`SharedWeights`): the Compact's codes, each product's input rotated (q64); the Standard's bf16
/// rounded to f16 as they are read, the input as it is (`product16`) — the Standard beside the ANE.
enum MossShared {
    typealias Argument = MossFunctions.Argument
    static let d = MossFunctions.d, heads = MossFunctions.heads, hd = MossFunctions.hd, hidden = MossFunctions.hidden
    static let eps = MossFunctions.eps

    /// The gains `gains` writes, each `[1, 1, d]`, and the gates' factors `[1, 1, 1]`.
    static let gains = ["shift1", "scale1", "gate1", "shift2", "scale2", "gate2"]
    static let gateFactors = ["gate1_factor", "gate2_factor"]

    static func arguments(_ f: String, rows: Int, keys: Int, weights: SharedWeights, ranks: [String: Int] = [:]) -> [Argument] {
        plainArguments(f, rows: rows, keys: keys, weights: weights) + MossFunctions.lowRankArguments(f, ranks: ranks, element: .f16)
    }

    static func plainArguments(_ f: String, rows: Int, keys: Int, weights: SharedWeights) -> [Argument] {
        let row = [1, 1, d], one = [1, 1, 1]
        let stream: [Argument] = [("x", [1, rows, d], .f16), ("x_factor", [1, rows, 1], .f16)]
        func weight(_ name: String, _ inn: Int, _ out: Int) -> [Argument] {
            weights.arguments(name, inn: inn, out: out) + [(name + "_target", one, .f16)]
        }
        switch f {
        case "gains": return [("modulation", [6, MossFunctions.gainRow], .bf16), ("gate1_factor", one, .f16), ("gate2_factor", one, .f16)]
        case "enter": return [("x", [rows, d], .bf16)]
        case "pre": return stream + [("shift1", row, .f16), ("scale1", row, .f16), ("norm1", row, .f16)] + weight("qkv", d, 3 * d)
        case "heads":
            return [("qkv", [1, rows, 3 * d], .f16), ("qkv_factor", [1, rows, 1], .f16), ("q_norm", [hd], .f16), ("k_norm", [hd], .f16),
                    ("cos", [rows, 1, hd], .f32), ("sin", [rows, 1, hd], .f32)]
        case "attention":
            return [("q", [1, heads, rows, hd], .f16), ("k", [1, keys, heads, hd], .f16), ("v", [1, keys, heads, hd], .f16), ("mask", [1, keys], .f16)]
        case "out": return stream + [("attended", [1, rows, d], .f16), ("gate1", row, .f16), ("gate1_factor", one, .f16)] + weight("o", d, d)
        case "ffn":
            return stream + [("shift2", row, .f16), ("scale2", row, .f16), ("gate2", row, .f16), ("gate2_factor", one, .f16), ("norm2", row, .f16)]
                + weight("up_gate", d, 2 * hidden) + weight("down", hidden, d)
        case "final":
            return stream + [("norm", [d], .f32), ("head", [2, d], .bf16),
                    ("final_linear", [d, MossFunctions.channels], .bf16)]
        default: preconditionFailure("\(f): no shared function")
        }
    }

    static func module(_ f: String, rows: Int, keys: Int, weights: SharedWeights, height h: Int = 0, width w: Int = 0, tapped: Bool = false,
                       ranks: [String: Int] = [:]) -> Aicode.Module {
        let arguments = arguments(f, rows: rows, keys: keys, weights: weights, ranks: ranks)
        var m = Aicode.Module(arguments: arguments.map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        let b = m.body
        let names = m.argumentNames
        func a(_ name: String) -> Int { names.firstIndex(of: name)! }
        var constants = Aicode.ScalingConstants()
        func product(_ x: Int, _ name: String, input: Int) -> Int {
            m.product(x, name, input: input, adapted: (ranks[name] ?? 0) > 0, constants: &constants)
        }
        func f32(_ x: Int) -> Int { b.castUnlessSame(x, to: .f32) }
        func bf16(_ x: Int) -> Int { b.castUnlessSame(x, to: .bf16) }
        func f16(_ x: Int) -> Int { b.castUnlessSame(x, to: .f16) }
        func rotated(_ t: Int, for w: String) -> (unit: Int, factor: Int, input: Int) { b.rotated16(t, target: a(w + "_target"), of: weights) }
        /// RMS(x̂ · X) · γ · (1 + s) + t, rotated under the product's target.
        func modulatedInput(_ x: Int, _ xf: Int, gain: String, scale: String, shift: String, for w: String) -> (unit: Int, factor: Int, input: Int) {
            rotated(b.binary(.add, b.binary(.mul, b.binary(.mul, b.norm(x, eps: eps, factors: [xf]), a(gain)), a(scale)), a(shift)), for: w)
        }
        /// A true value at 1 at most (the gains' clip, the SITU's ±100): x̂ · F clipped to ±`limit`, F `[1, rows, 1]`.
        func clipped(_ t: Int, _ f: Int, _ limit: Double) -> Int { b.scale("minimum", b.scale("maximum", b.binary(.mul, t, f), -limit), limit) }

        switch f {
        case "gains":
            let all = f32(a("modulation"))
            func row(_ i: Int) -> Int { b.reshape(b.slice(b.slice(all, axis: 0, start: i, size: 1), axis: 1, start: 0, size: d), [1, 1, d]) }
            func gate(_ i: Int, _ factor: String) -> Int { f16(b.binary(.divide, row(i), f32(a(factor)))) }
            m.outputs = [("shift1", f16(row(0))), ("scale1", f16(b.scale("add", row(1), 1))), ("gate1", gate(2, "gate1_factor")),
                         ("shift2", f16(row(3))), ("scale2", f16(b.scale("add", row(4), 1))), ("gate2", gate(5, "gate2_factor"))]
        case "enter":
            let (u, factor) = b.ranged16(b.reshape(f16(a("x")), [1, rows, d]), under: 14)
            m.outputs = [("y", u), ("y_factor", factor)]
        case "pre":
            let r = modulatedInput(a("x"), a("x_factor"), gain: "norm1", scale: "scale1", shift: "shift1", for: "qkv")
            m.outputs = [("qkv", product(r.unit, "qkv", input: r.input)), ("qkv_factor", r.factor)]
        case "heads":
            let all = b.reshape(a("qkv"), [rows, 3 * d]), factor = f32(b.reshape(a("qkv_factor"), [rows, 1]))
            func third(_ i: Int) -> Int { b.reshape(b.slice(all, axis: 1, start: i * d, size: d), [rows, heads, hd]) }
            func headNormed(_ t: Int, _ gain: String) -> Int { b.headNorm(t, factor: a("qkv_factor"), gain: a(gain), eps: eps, heads: heads) }
            // MOSS's RoPE on halves (`_rotate`), the tables [cos | cos] and [−sin | sin]: the halves only swapped.
            func rope(_ t: Int) -> Int {
                let swapped = b.concat([b.slice(t, axis: 2, start: hd / 2, size: hd / 2), b.slice(t, axis: 2, start: 0, size: hd / 2)], axis: 2)
                return f16(b.binary(.add, b.binary(.mul, t, a("cos")), b.binary(.mul, swapped, a("sin"))))
            }
            func headsFirst(_ t: Int) -> Int { b.reshape(b.transpose(t, [1, 0, 2]), [1, heads, rows, hd]) }
            /// The keys and values token first (`MossFunctions`').
            func tokenFirst(_ t: Int) -> Int { b.reshape(t, [1, rows, heads, hd]) }
            let q = headNormed(third(0), "q_norm")
            m.outputs = [("q", headsFirst(rope(q))), ("k", tokenFirst(rope(headNormed(third(1), "k_norm")))),
                         ("v", tokenFirst(b.binary(.mul, third(2), b.reshape(f16(factor), [rows, 1, 1]))))]
            if tapped { m.outputs.append(("q_norm_out", b.reshape(bf16(q), [rows, d]))) }
        case "attention":
            let o = b.attention([a("q"), b.transpose(a("k"), [0, 2, 1, 3]), b.transpose(a("v"), [0, 2, 1, 3]), a("mask")], scale: 1 / Double(hd).squareRoot(), symbol: "sdpa_tile")
            m.outputs = [("attended", b.reshape(b.transpose(b.reshape(o, [heads, rows, hd]), [1, 0, 2]), [1, rows, d]))]
        case "out":
            let r = rotated(a("attended"), for: "o")
            let p = product(r.unit, "o", input: r.input)
            let (y, yf) = b.gated16(a("x"), a("x_factor"), plus: p, gate: a("gate1"), gateFactor: a("gate1_factor"), factors: [r.factor])
            m.outputs = [("y", y), ("y_factor", yf)]
            if tapped { m.outputs.append(("o_out", b.reshape(bf16(b.binary(.mul, f32(p), f32(r.factor))), [rows, d]))) }
        case "ffn":
            let r = modulatedInput(a("x"), a("x_factor"), gain: "norm2", scale: "scale2", shift: "shift2", for: "up_gate"), f2 = r.factor
            let ug = product(r.unit, "up_gate", input: r.input)
            let hv = clipped(b.slice(ug, axis: 2, start: 0, size: hidden), f2, 128), gv = clipped(b.slice(ug, axis: 2, start: hidden, size: hidden), f2, 32)
            let gate = b.binary(.mul, b.scale("mul", b.tanh(b.scale("mul", gv, 0.25)), 4), b.sigmoid(gv))
            let situ = b.binary(.mul, gate, b.scale("mul", b.tanh(b.scale("mul", hv, 0.04)), 25))
            let (u3, f3) = b.ranged16(b.rotation(situ, of: weights), target: a("down_target"))
            let p = product(u3, "down", input: b.binary(.divide, situ, f3))
            let (y, yf) = b.gated16(a("x"), a("x_factor"), plus: p, gate: a("gate2"), gateFactor: a("gate2_factor"), factors: [f3])
            m.outputs = [("y", y), ("y_factor", yf)]
            if tapped {
                m.outputs += [("up_gate_out", b.reshape(bf16(b.binary(.mul, f32(ug), f32(f2))), [rows, 2 * hidden])),
                              ("down_out", b.reshape(bf16(b.binary(.mul, f32(p), f32(f3))), [rows, d]))]
            }
        case "final":
            let x = b.reshape(b.binary(.mul, f32(a("x")), f32(a("x_factor"))), [rows, d])
            let n = b.binary(.mul, b.rmsNorm(x, axes: [1], eps: eps), a("norm"))
            let head = f32(a("head"))
            let shift = b.slice(head, axis: 0, start: 0, size: 1), scale = b.slice(head, axis: 0, start: 1, size: 1)
            let modulated = bf16(b.binary(.add, b.binary(.mul, n, b.scale("add", scale, 1)), shift))
            let y = f32(b.matmul(modulated, a("final_linear"), result: Aicode.tensor([rows, MossFunctions.channels], .bf16)))
            m.outputs = [("velocity", b.reshape(b.transpose(y, [1, 0]), [MossFunctions.channels, h, w]))]
            if tapped { m.outputs.append(("final_norm_out", bf16(n))) }
        default: preconditionFailure("\(f): no shared function")
        }
        return m
    }

    /// One of the 42 blocks' weights in the shared forms: the codes `[in, out]` (the stage's surfaces) or the f16
    /// weights (`weights`), each product's target after it, the gains in f16.
    static func layer(_ i: Int, weights: SharedWeights) -> [Weights.Entry] {
        let p = "layers.\(i)."
        var all: [Weights.Entry] = []
        // Each product's target computed when the weights are read (`Recipe.target`): the ANE computed it wrong inside a
        // shared feed-forward (Z's, measured 18/11 on the M5 Pro).
        for (name, key) in [("qkv", "proj.qkv"), ("o", "proj.o"), ("up_gate", "proj.ffn_up_gate"), ("down", "proj.ffn_down")] {
            all.append(weights.entry(name, key: p + key))
            all.append((name + "_target", "", .target([name])))
        }
        for name in ["norm1", "norm2", "q_norm", "k_norm"] { all.append((name, p + name + ".weight", .half)) }
        return all
    }
}
