import CoreAI
import Foundation
import SiliCore

/// **An Anima block as the GPU and the ANE share it** (MOSS's `MossShared`, Qwen-Image-2.1's `QwenImage21Shared`, the
/// engine's forms in `Neural.swift`): f16 at every border and inside, `[1, rows, n]`, products on the Compact's codes
/// (Bedovyy's `convrot`: the input rotated by groups of 256) int8 × int8 on the GPU, q64 on the ANE's copy of them
/// (`SharedWeights`), norms by their peak, the guard's powers of two in f16; what reads
/// across rows, or computes in f32, is the GPU's:
///
///     pre        both   n = LN(x̂ · X) · (1 + s₁) + t₁, rotated under 2^e₀ by F: q̂, k̂, v̂ = the products, and F
///     heads      GPU    q, k: RMS_head(·) · γ (eps / F²) in f32, RoPE; v = v̂ · F; heads first, into the image's keys
///     attention  GPU    o = sdpa(q, K, V, mask) over every row: a true value
///     out        both   x₁ = x + g₁ · (o R) · W_o; then n₂ = LN(x₁) · (1 + s₂) + t₂ rotated: ĉq = the product, and its F
///     cross      GPU    RMS_head(cq) · γ (eps / F²) in f32, sdpa against the prompt's keys: a true value
///     crossout   both   x₂ = x₁ + g₂ · (o R) · W_co
///     ffn        both   y = x₂ + g₃ · W₂(GELU(W₁(LN(x₂) · (1 + s₃) + t₃ R)) R)
///
/// What Anima asks beyond MOSS's forms:
///   - **a LayerNorm without affine** (`CosmosAdaLayerNormZero`, ε = 10⁻⁶), Qwen-Image-2.1's: u = x̂ · 2⁻¹⁴ (x̂ within
///     2¹⁴, u within 1, exact), the RMS of u − ū as the true value (u − ū) · 2¹⁴ · X;
///   - **three updates a block**, gates unbounded (no tanh) as MOSS's: each gate held as (ĝ, G) by the loop off its row;
///     the stream through `mid` and `mid2`, then the other buffer;
///   - **the exact GELU** (erf) of the true value z = ĥ · F: GELU(clip(z, ±16)) + max(z − 16, 0), Φ being 1 or 0 in
///     f16 past ±16, so that the primitive never reads a value its range does not hold.
///
/// **Each product on what the file holds**: its codes where the file holds codes, else its weight in f16
/// (`SharedWeights.half`, the publisher's bf16 rounded once), no input rotated (`product16`) — the Compact's blocks 0 and
/// 27 all in f16, the MLP of 1 to 3 too, the others on codes; every product of the Standard beside the ANE in f16. Every
/// block of both in these forms, the ANE beside on every one. (The Compact's five blocks with float products were the
/// GPU's alone in guarded forms till 26/11/2026: 0.96 s of a step of 4.14 at 1024² on the M1 Pro, the ANE idle.) Block 0's
/// branches written by `out-taps`, `crossout-taps` and `ffn-taps` (the GPU's, for a check listening, `Block.run`'s
/// `tapped`): each product's true value before its gate, in f16.
enum AnimaShared {
    typealias Argument = AnimaFunction.Argument
    static let d = AnimaFunction.d, heads = AnimaFunction.heads, hd = AnimaFunction.hd, hidden = AnimaFunction.hidden

    /// The products each function reads, by their arguments' names, and their targets (`Recipe.target`, computed as the
    /// weights are read: the ANE computed it wrong inside a shared feed-forward, Z's, measured 18/11 on the M5 Pro).
    static let products: [String: [String]] = ["pre": ["q", "k", "v"], "out": ["o", "cq"], "crossout": ["co"], "ffn": ["m1", "m2"]]
    static let targets: [(name: String, products: [String])] = [("qkv_target", ["q", "k", "v"]), ("o_target", ["o"]), ("cq_target", ["cq"]),
                                                                ("co_target", ["co"]), ("m1_target", ["m1"]), ("m2_target", ["m2"])]
    /// The modulation's rows as the forms read them, each `[1, 1, d]` (a gate over its factor), and the gates' factors.
    static let gains = ["shift1", "scale1", "gate1", "shift2", "scale2", "gate2", "shift3", "scale3", "gate3"]
    static let gateFactors = ["gate1_factor", "gate2_factor", "gate3_factor"]

    /// What product `name` reads: its codes where the block's file holds codes (`codes`), else its weight in f16.
    static func weights(_ name: String, codes: Set<String>) -> SharedWeights { codes.contains(name) ? .codes : .half }

    /// `codes`: the block's products the file holds as codes (`AnimaEvaluation.quantized`).
    static func arguments(_ f: String, rows: Int, keys: Int, context: Int, codes: Set<String>, ranks: [String: Int], neural: Bool = false) -> [Argument] {
        let row = [1, 1, d], one = [1, 1, 1]
        let stream: [Argument] = [("x", [1, rows, d], .f16), ("x_factor", [1, rows, 1], .f16)]
        func weight(_ name: String) -> [Argument] {
            let (inn, out) = AnimaFunction.dims(name)
            return weights(name, codes: codes).arguments(name, inn: inn, out: out, neural: neural)
        }
        func gains(_ names: String...) -> [Argument] { names.map { ($0, $0.hasSuffix("_factor") || $0.hasSuffix("_target") ? one : row, .f16) } }
        let own: [Argument]
        switch f {
        case "pre": own = stream + gains("shift1", "scale1") + weight("q") + weight("k") + weight("v") + gains("qkv_target")
        case "heads":
            own = [("qh", [1, rows, d], .f16), ("kh", [1, rows, d], .f16), ("vh", [1, rows, d], .f16), ("qkv_factor", [1, rows, 1], .f16),
                   ("q_norm", [hd], .f16), ("k_norm", [hd], .f16), ("cos", [rows, 1, hd / 2], .f32), ("sin", [rows, 1, hd / 2], .f32)]
        case "attention":
            own = [("q", [1, heads, rows, hd], .f16), ("k", [1, keys, heads, hd], .f16), ("v", [1, keys, heads, hd], .f16), ("mask", [1, keys], .f16)]
        case "out":
            own = stream + [("attended", [1, rows, d], .f16)] + gains("gate1", "gate1_factor", "shift2", "scale2") + weight("o") + weight("cq")
                + gains("o_target", "cq_target")
        case "cross":
            own = [("cqh", [1, rows, d], .f16), ("cq_factor", [1, rows, 1], .f16), ("cq_norm", [hd], .f16),
                   ("ctx_k", [1, heads, context, hd], .f16), ("ctx_v", [1, heads, context, hd], .f16), ("ctx_mask", [1, context], .f16)]
        case "crossout": own = stream + [("attended", [1, rows, d], .f16)] + gains("gate2", "gate2_factor") + weight("co") + gains("co_target")
        case "ffn":
            own = stream + gains("shift3", "scale3", "gate3", "gate3_factor") + weight("m1") + weight("m2") + gains("m1_target", "m2_target")
        default: preconditionFailure("\(f): no shared function")
        }
        return own + (products[f] ?? []).compactMap { p in ranks[p].map { (p, $0) } }.flatMap { p, r -> [Argument] in
            let (inn, out) = AnimaFunction.dims(p)
            return [(p + "_down", [inn, r], .f16), (p + "_up", [r, out], .f16)]
        }
    }

    /// `neural`: the ANE's function of a step both units take: its products q64 on the ANE's copy of the codes, the
    /// GPU's int8 × int8 (`Aicode.Module.product`).
    static func module(_ function: String, rows: Int, keys: Int, context: Int, codes: Set<String>, ranks: [String: Int],
                       neural: Bool = false) -> Aicode.Module {
        let tapping = function.hasSuffix("-taps"), f = tapping ? String(function.dropLast(5)) : function
        var m = Aicode.Module(arguments: arguments(f, rows: rows, keys: keys, context: context, codes: codes, ranks: ranks, neural: neural).map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        let b = m.body
        let names = m.argumentNames
        func a(_ name: String) -> Int { names.firstIndex(of: name)! }
        var constants = Aicode.ScalingConstants()
        func f32(_ x: Int) -> Int { b.castUnlessSame(x, to: .f32) }
        func f16(_ x: Int) -> Int { b.castUnlessSame(x, to: .f16) }
        func product(_ x: Int, _ name: String, input: Int) -> Int {
            m.product(x, name, input: input, adapted: (ranks[name] ?? 0) > 0, neural: neural, constants: &constants)
        }
        /// t under the target of its products, rotated before codes (the products of one target all on codes or none).
        func rotated(_ t: Int, for target: String) -> (unit: Int, factor: Int, input: Int) {
            let of = targets.first { $0.name == target }!.products.map { weights($0, codes: codes) }
            precondition(Set(of).count == 1, "\(target): products on codes and on floats")
            return b.rotated16(t, target: a(target), of: of[0])
        }
        /// LN(x̂ · X) · (1 + s) + t, the LN without affine (ε = 10⁻⁶).
        func modulated(_ x: Int, _ xf: Int, _ shift: String, _ scale: String) -> Int {
            b.binary(.add, b.binary(.mul, b.layerNorm16(x, xf, eps: 1e-6), a(scale)), a(shift))
        }
        func updated(_ x: Int, _ xf: Int, _ p: Int, gate: String, factors: [Int]) -> (Int, Int) {
            b.gated16(x, xf, plus: p, gate: a(gate), gateFactor: a(gate + "_factor"), factors: factors)
        }
        /// A head's RMS norm (ε = 10⁻⁵).
        func headNormed(_ t: Int, _ factor: String, _ gain: String) -> Int { b.headNorm(t, factor: a(factor), gain: a(gain), eps: 1e-5, heads: heads) }
        func headsFirst(_ t: Int) -> Int { b.reshape(b.transpose(t, [1, 0, 2]), [1, heads, rows, hd]) }
        /// The self-attention's keys and values token first (`AnimaFunction`'s).
        func tokenFirst(_ t: Int) -> Int { b.reshape(t, [1, rows, heads, hd]) }
        func rowsFirst(_ o: Int) -> Int { b.reshape(b.transpose(b.reshape(o, [heads, rows, hd]), [1, 0, 2]), [1, rows, d]) }
        /// `apply_rotary_emb(use_real_unbind_dim=-2)` (`AnimaFunction`'s): a head's halves rotated in f32, rounded once.
        func rope(_ t: Int) -> Int {
            let half = hd / 2
            let x0 = b.slice(t, axis: 2, start: 0, size: half), x1 = b.slice(t, axis: 2, start: half, size: half)
            let (c, s) = (a("cos"), a("sin"))
            let first = b.binary(.sub, b.binary(.mul, x0, c), b.binary(.mul, x1, s))
            let second = b.binary(.add, b.binary(.mul, x1, c), b.binary(.mul, x0, s))
            return f16(b.concat([first, second], axis: 2))
        }

        switch f {
        case "pre":
            let r = rotated(modulated(a("x"), a("x_factor"), "shift1", "scale1"), for: "qkv_target")
            m.outputs = [("qh", product(r.unit, "q", input: r.input)), ("kh", product(r.unit, "k", input: r.input)),
                         ("vh", product(r.unit, "v", input: r.input)), ("qkv_factor", r.factor)]
        case "heads":
            let factor = b.reshape(a("qkv_factor"), [rows, 1, 1])
            m.outputs = [("q", headsFirst(rope(headNormed(a("qh"), "qkv_factor", "q_norm")))),
                         ("k", tokenFirst(rope(headNormed(a("kh"), "qkv_factor", "k_norm")))),
                         ("v", tokenFirst(b.binary(.mul, b.reshape(a("vh"), [rows, heads, hd]), factor)))]
        case "attention":
            let o = b.attention([a("q"), b.transpose(a("k"), [0, 2, 1, 3]), b.transpose(a("v"), [0, 2, 1, 3]), a("mask")], scale: 1 / Double(hd).squareRoot(), symbol: "sdpa_self")
            m.outputs = [("o", rowsFirst(o))]
        case "out":
            let r = rotated(a("attended"), for: "o_target")
            let p = product(r.unit, "o", input: r.input)
            let (x1, x1f) = updated(a("x"), a("x_factor"), p, gate: "gate1", factors: [r.factor])
            let r2 = rotated(modulated(x1, x1f, "shift2", "scale2"), for: "cq_target")
            m.outputs = [("y", x1), ("y_factor", x1f), ("cqh", product(r2.unit, "cq", input: r2.input)), ("cq_factor", r2.factor)]
                + (tapping ? [("attn1", b.binary(.mul, p, r.factor))] : [])
        case "cross":
            let q = headsFirst(f16(headNormed(a("cqh"), "cq_factor", "cq_norm")))
            let o = b.attention([q, a("ctx_k"), a("ctx_v"), a("ctx_mask")], scale: 1 / Double(hd).squareRoot(), symbol: "sdpa_cross")
            m.outputs = [("o", rowsFirst(o))]
        case "crossout":
            let r = rotated(a("attended"), for: "co_target")
            let p = product(r.unit, "co", input: r.input)
            let (y, yf) = updated(a("x"), a("x_factor"), p, gate: "gate2", factors: [r.factor])
            m.outputs = [("y", y), ("y_factor", yf)] + (tapping ? [("attn2", b.binary(.mul, p, r.factor))] : [])
        case "ffn":
            let r = rotated(modulated(a("x"), a("x_factor"), "shift3", "scale3"), for: "m1_target")
            let z = b.binary(.mul, product(r.unit, "m1", input: r.input), r.factor)
            let gelu = b.gelu16(z)
            let (u, f2) = b.ranged16(b.rotation(gelu, of: weights("m2", codes: codes)), target: a("m2_target"))
            let p = product(u, "m2", input: b.binary(.divide, gelu, f2))
            let (y, yf) = updated(a("x"), a("x_factor"), p, gate: "gate3", factors: [f2])
            m.outputs = [("y", y), ("y_factor", yf)] + (tapping ? [("ff", b.binary(.mul, p, f2))] : [])
        default: preconditionFailure("\(f): no shared function")
        }
        return m
    }

    /// Block i's weights in these forms: its modulation's (the GPU's guarded function reads them), each product's codes
    /// (where the file holds codes, `codes`) or f16 weight and their target after them, the heads' gains in f16.
    static func layer(_ i: Int, codes: Set<String>) -> [Weights.Entry] {
        let p = "blocks.\(i)."
        var all = AnimaEvaluation.modulationEntries(block: i)
        for name in ["q", "k", "v", "o", "cq", "co", "m1", "m2"] { all.append(weights(name, codes: codes).entry(name, key: p + AnimaEvaluation.products[name]!)) }
        for t in targets { all.append((t.name, "", .target(t.products))) }
        for (name, key) in [("q_norm", "self_attn.q_norm"), ("k_norm", "self_attn.k_norm"), ("cq_norm", "cross_attn.q_norm")] {
            all.append((name, p + key + ".weight", .half))
        }
        return all
    }
}
