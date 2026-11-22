import Foundation

/// **The guard of f16**: the rules every family's functions
/// are written with, whatever the model — no value whose size the activations decide is written in f16 as it is. It
/// travels as (t, F): t in f16, F = 2^e a row (f32, `[rows, 1]`), computed by the graph, its true value t · F:
///   - **a product** reads t brought under 2^e₀, e₀ its own: K · max|w| · 2^e₀ ≤ 65 504 bounds its result whatever its
///     input (an int8 code is at most 128 times its scale), e₀ computed by the graph from the weight itself (`target`),
///     so that no file is assumed. Its result keeps the factor;
///   - **a norm** (RMS, LayerNorm) reads t, its eps eps / F² a row (`epsilon`): F leaves nothing else;
///   - **x · φ(x)** (SiLU, GELU) passes F on: F · t · φ(t F), φ reading the true value clipped where it is flat; a
///     product of two kept values multiplies their factors, each brought low enough that their product holds;
///   - **an op a row factor cannot pass** (a bias, a sum of rows under different factors) reads the true value, in f32;
///   - **the residual stream is a kept value** (`residual`): no product reads it, only norms, so it is held high, in
///     [2¹³, 2¹⁴] (`stream`), where its small channels stay among f16's normals beside a massive one.
/// Written true, multiplied out: what the attention reads across rows, where one factor a row cannot go (v), and a
/// value bounded by construction (a normalized row times weights: a modulation's `1 + m`, `tanh m`). Which unit's
/// attention holds which scores is a fact of that unit on that Mac, not a rule written here.
extension Aicode.Block {
    /// The exponent of the residual stream: its rows' largest in [2¹³, 2¹⁴].
    public var stream: Int { scalar(14, .f32) }

    /// `t` `[rows, n]` (f16 or f32) as t / F in f16 and F `[rows, 1]` f32: e = round(log₂ max|t| + ½) − e₀, so that the
    /// row's largest is in [2^(e₀−1), 2^e₀]. F = 2^max(e, 0) / 2^max(−e, 0), each rounded: exact whatever `pow` gives,
    /// within ½. From f16, e within [−13, 15] so that F is one of f16's normals and divides t in f16 exactly (a row past
    /// 2¹⁵ · 2^e₀ comes out over 2^e₀, under 2^(e₀+1)); from f32, t is divided in f32, then rounded once.
    ///
    /// `over`: the axes a factor spans (the last by default: a row); the factor keeps the others, 1 along these. A feature
    /// map's row is its channels and its columns (`[rows, C, W]` over `[1, 2]`); what a convolution reads, the whole window
    /// (every axis: one factor, since its 3×3 mixes the rows).
    public func ranged(_ t: Int, over axes: [Int]? = nil, under e0: Int? = nil) -> (unit: Int, factor: Int) {
        let (shape, element) = Aicode.shape(types[t])
        let top = castUnlessSame(scale("maximum", reduceMax(primitive("abs", t), axes: axes ?? [shape.count - 1]), 0x1p-14), to: .f32)
        var e = primitive("round", scale("add", scale("mul", primitive("log", top), 1 / Foundation.log(2.0)), 0.5))
        if var e0 {
            // A target of another rank (a kernel's `[1, 1, 1, 1]` for a window `[rows, C, W]`) brought to t's: ones. A
            // scalar's broadcasts as it is (the stream's: the assets written before unchanged).
            let r0 = Aicode.shape(types[e0]).shape.count
            if r0 > 0 && r0 != shape.count { e0 = reshape(e0, Array(repeating: 1, count: shape.count)) }
            e = binary(.sub, e, e0)
        }
        let (low, high) = element == .f32 ? (-120.0, 120.0) : (-13.0, 15.0)
        e = scale("minimum", scale("maximum", e, low), high)
        func power(_ k: Int) -> Int { primitive("round", scale("pow", scale("maximum", k, 0), 2, scalarFirst: true)) }
        let f = binary(.divide, power(e), power(scale("mul", e, -1)))
        let unit = element == .f32 ? cast(binary(.divide, t, f), to: .f16) : binary(.divide, t, cast(f, to: .f16))
        return (unit, f)
    }

    /// The exponent e₀ the W8A16 products of `codes` (int8 in `layout`, each with its scales along `out`) read their one
    /// input under: the largest with K · 128 · max s · 2^e₀ ≤ 65 504 for each, K the products' inputs (the codes' first
    /// axis `[in, out]`, their second `[out, in]`).
    public func target(codes: [(codes: Int, scale: Int)], layout: CodesLayout = .inOut) -> Int {
        minimum(codes.map { q, s in
            let top = castUnlessSame(reduceMax(primitive("abs", s), axes: [0]), to: .f32)
            let k = Aicode.shape(types[q]).shape[layout == .inOut ? 0 : 1]
            return exponent(scale("mul", top, Double(k * 128)))
        })
    }

    /// The same for products of f16 weights `[in, out]`: K · max|w| · 2^e₀ ≤ 65 504.
    public func target(weights: [Int]) -> Int {
        minimum(weights.map { w in
            let top = castUnlessSame(reduceMax(primitive("abs", w), axes: [0, 1]), to: .f32)
            return exponent(scale("mul", top, Double(Aicode.shape(types[w]).shape[0])))
        })
    }

    /// The same for convolutions of f16 kernels `[out, in, kh, kw]`: each output reads in · kh · kw inputs, so that
    /// in · kh · kw · max|w| · 2^e₀ ≤ 65 504 bounds it whatever the window holds.
    public func target(kernels: [Int]) -> Int {
        minimum(kernels.map { w in
            let s = Aicode.shape(types[w]).shape
            let top = castUnlessSame(reduceMax(primitive("abs", w), axes: Array(s.indices)), to: .f32)
            return exponent(scale("mul", top, Double(s.dropFirst().reduce(1, *))))
        })
    }

    /// floor(log₂(65 504 / bound)), as round(x − ½) — halves away from zero: a floor, one less at 0 and the negative
    /// integers, never above it.
    private func exponent(_ bound: Int) -> Int {
        let room = scale("divide", bound, 65504, scalarFirst: true)
        return primitive("round", scale("sub", scale("mul", primitive("log", room), 1 / Foundation.log(2.0)), 0.5))
    }

    private func minimum(_ each: [Int]) -> Int { each.dropFirst().reduce(each[0]) { elementwise("minimum", $0, $1) } }

    /// **A gated SiLU of two products kept under one factor** (SwiGLU): `gate` and `up` `[rows, n]` f16, the results of
    /// products whose input was kept under F `[rows, 1]` f32. SiLU(g F) · u F = F² g₁ g₃ · (g/g₁) sigmoid(g F) (u/g₃), g
    /// and u each brought under 2⁵ so that their product holds under 2¹⁰, the sigmoid reading the true value clipped to
    /// ±16 (sigmoid(16) is 1 in f16). `entering` brings that product under what reads it (its rotation, the next product's target) and returns
    /// its unit and factor g₄; the result is that unit, under F² g₁ g₃ g₄ — powers of two, so their product is exact.
    public func gatedSiLU(_ gate: Int, _ up: Int, factor f: Int, entering: (Int) -> (unit: Int, factor: Int)) -> (unit: Int, factor: Int) {
        let five = scalar(5, .f32)
        let (u1, g1) = ranged(gate, under: five), (u3, g3) = ranged(up, under: five)
        let z = scale("minimum", scale("maximum", binary(.mul, gate, cast(f, to: .f16)), -16), 16)
        let (h, g4) = entering(binary(.mul, binary(.mul, u1, sigmoid(z)), u3))
        return (h, binary(.mul, binary(.mul, binary(.mul, f, f), binary(.mul, g1, g3)), g4))
    }

    /// **The same in f16 alone**, for a function the ANE takes: `gate` and `up` `[1, rows, n]` f16 under F `[1, rows, 1]`
    /// f16 (`ranged16`'s), each brought under 2⁵ (`ranged16`, g₁ and g₃), the sigmoid reading the true value clipped to
    /// ±16. The product (g/g₁) sigmoid(g F) (u/g₃), under 2¹⁰, is the true value over F² g₁ g₃: the factors are left to the
    /// caller, which multiplies them where its update needs them (a norm's, a residual's, in its own order).
    public func gatedSiLU16(_ gate: Int, _ up: Int, factor f: Int) -> (value: Int, gate: Int, up: Int) {
        let (u1, g1) = ranged16(gate, under: 5), (u3, g3) = ranged16(up, under: 5)
        let z = scale("minimum", scale("maximum", binary(.mul, gate, f), -16), 16)
        return (binary(.mul, binary(.mul, u1, sigmoid(z)), u3), g1, g3)
    }

    /// The eps a norm reads of a row kept under F: eps / F² (0 past f32's range, where eps is nothing).
    public func epsilon(_ f: Int, _ eps: Double) -> Int { scale("divide", binary(.mul, f, f), eps, scalarFirst: true) }

    /// The residual stream's update: x̂ (under 2¹⁴) kept under X, plus Δ, a true value in f16 (under 2¹⁶: Δ/4 under 2¹⁴,
    /// kept under 4; a Δ kept under its own G enters as Δ̂ with G), both brought to half the larger factor C (under 2¹³
    /// each), summed, the sum brought back to [2¹³, 2¹⁴]: (x̂', C · C').
    public func residual(_ x: Int, _ xf: Int, plus delta: Int) -> (value: Int, factor: Int) {
        let c = scale("mul", scale("maximum", xf, 4), 2)
        let sum = binary(.add, binary(.mul, x, cast(binary(.divide, xf, c), to: .f16)),
                         binary(.mul, delta, cast(scale("divide", c, 1, scalarFirst: true), to: .f16)))
        let (s, f) = ranged(sum, under: stream)
        return (s, binary(.mul, c, f))
    }

    /// The same with Δ kept as (Δ̂, G) — a product's result under its target (under 2¹⁶) and its factor `[rows, 1]` f32,
    /// where no norm bounds Δ before the stream (an encoder's products, 18/11): C = 2 · max(X, 4G), each term under 2¹³.
    public func residual(_ x: Int, _ xf: Int, plus delta: Int, factor g: Int) -> (value: Int, factor: Int) {
        let c = scale("mul", elementwise("maximum", xf, scale("mul", g, 4)), 2)
        let sum = binary(.add, binary(.mul, x, cast(binary(.divide, xf, c), to: .f16)),
                         binary(.mul, delta, cast(binary(.divide, g, c), to: .f16)))
        let (s, f) = ranged(sum, under: stream)
        return (s, binary(.mul, c, f))
    }

    /// `x · W` with W the codes int8 dequantized in f16 by their scales along `out` (W8A16): the scales follow the output,
    /// never the contraction. `layout`: how the codes lie (`CodesLayout`, the stage's plan's), `[in, out]` read as they
    /// lie, `[out, in]` transposed after the dequantize, as `linear` writes it.
    public func w8a16(_ x: Int, codes: Int, scale s: Int, layout: CodesLayout = .inOut, constants: inout Aicode.ScalingConstants) -> Int {
        let rows = Aicode.shape(types[x]).shape[0]
        switch layout {
        case .inOut:
            let w = dequantize(codes, scale: s, axis: 1, constants: &constants)
            return matmul(x, w, result: Aicode.tensor([rows, Aicode.shape(types[w]).shape[1]], .f16))
        case .outIn:
            let w = dequantize(codes, scale: s, axis: 0, constants: &constants)
            return matmul(x, transpose(w, [1, 0]), result: Aicode.tensor([rows, Aicode.shape(types[w]).shape[0]], .f16))
        }
    }

    /// `x · W` with W f16 `[in, out]`.
    public func product(_ x: Int, _ w: Int) -> Int {
        let rows = Aicode.shape(types[x]).shape[0], out = Aicode.shape(types[w]).shape[1]
        return matmul(x, w, result: Aicode.tensor([rows, out], .f16))
    }

    /// A product's result kept under F, multiplied out in f32 with its bias (`[out]` f32): what an op a row factor cannot
    /// pass reads.
    public func multipliedOut(_ r: Int, _ f: Int, bias: Int) -> Int { binary(.add, binary(.mul, cast(r, to: .f32), f), bias) }
}

/// **A row's exponent on the CPU, as a function would compute it**, for the rows written before a function reads them
/// (a stream's first rows, a check's inputs): F = 2^e with the row's largest |v| = `top` brought under 2^e₀ (e₀ = 14,
/// the stream's). Two rules, as the graph has two: written for a function that reads them, the rows are the ones it
/// would have written itself.
public enum RowExponent {
    /// The rule a row's exponent is computed by: the graph's `ranged` (an f16 or an f32 factor) or `ranged16`.
    public enum Rule: Sendable { case ranged(f32Factor: Bool), ranged16 }

    /// `Aicode.Block.ranged`'s: round(log₂ top + ½) − e₀, the largest in [2^(e₀−1), 2^e₀]; within [−13, 15] for an f16
    /// factor, [−120, 120] for an f32 one. `top` floored at 2⁻¹⁴ as the graph floors it: a row of zeros has a factor.
    public static func ranged(largest top: Double, under e0: Int = 14, f32Factor: Bool) -> Int {
        let (low, high) = f32Factor ? (-120, 120) : (-13, 15)
        return min(max(Int((Foundation.log2(max(top, 0x1p-14)) + 0.5).rounded(.toNearestOrAwayFromZero)) - e0, low), high)
    }

    /// `Aicode.Block.ranged16`'s (the ANE's, no `round`): ⌈log₂ top⌉ − e₀, the largest in (2^(e₀−1), 2^e₀], within [−13, 15];
    /// `top` floored at 2⁻¹⁴ (`peak`'s).
    public static func ranged16(largest top: Double, under e0: Int = 14) -> Int {
        min(max(Int(Foundation.log2(max(top, 0x1p-14)).rounded(.up)) - e0, -13), 15)
    }

    /// **A row of true values as the stream holds it**: its factor F = 2^e by `rule` (the row's largest under 2^e₀), and
    /// the row over F, in f32 — a power of two: v / F is exact, rounded once where its client writes it in f16.
    public static func split<Row: Collection<Float>>(_ row: Row, _ rule: Rule, under e0: Int = 14) -> (unit: [Float], factor: Float) {
        let top = max(row.reduce(0) { max($0, abs($1)) }, 0x1p-14)
        let e = switch rule {
        case .ranged(let f32Factor): ranged(largest: Double(top), under: e0, f32Factor: f32Factor)
        case .ranged16: ranged16(largest: Double(top), under: e0)
        }
        let factor = Float(sign: .plus, exponent: e, significand: 1)
        return (row.map { $0 / factor }, factor)
    }
}
