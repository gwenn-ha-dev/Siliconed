import Foundation

/// **The shared forms' pieces that several transformers write the same** (`Neural.swift`'s forms put together): a
/// product's input rotated under its target, the q64 product with its low-rank path, a head's RMS norm of a guarded
/// value, a LayerNorm without affine in f16, an update through a gate held as (ĝ, G), the exact GELU of a true value
/// within f16's range. Each the very graph its clients wrote: the same values to the bit.
extension Aicode.Block {
    /// **t rotated under a product's target**: t under 2^`e0` (F₀), rotated, under the target (F₁) — (the unit, F₀ · F₁,
    /// the unrotated t / (F₀ · F₁), what a low-rank path beside the product reads, `Aicode.Module.product`). Both factors
    /// powers of two: the rotation of t / F is the rotation of t, over F. Before halves (`SharedWeights.half`), nothing to
    /// rotate: t under the target alone, the unit what the path reads.
    public func rotated16(_ t: Int, under e0: Double = 10, target: Int, of weights: SharedWeights) -> (unit: Int, factor: Int, input: Int) {
        guard weights.isCodes else {
            let (u, f) = ranged16(t, target: target)
            return (u, f, u)
        }
        let (u, f0) = ranged16(t, under: e0)
        let (r, f1) = ranged16(rotation(u), target: target)
        return (r, binary(.mul, f0, f1), binary(.divide, u, f1))
    }

    /// **A head's RMS norm of a guarded value** t · F, in f32: t read as `[rows, heads, hd]`, F a row's factor
    /// (`[rows, 1]` in any shape of as many values), the norm's ε divided by F² (`epsilon`), times γ (`gain`, `[hd]`).
    public func headNorm(_ t: Int, factor: Int, gain: Int, eps: Double, heads: Int) -> Int {
        let rows = Aicode.shape(types[factor]).shape.reduce(1, *), hd = Aicode.shape(types[t]).shape.reduce(1, *) / rows / heads
        let e = reshape(epsilon(castUnlessSame(reshape(factor, [rows, 1]), to: .f32), eps), [rows, 1, 1])
        return binary(.mul, rmsNorm(castUnlessSame(reshape(t, [rows, heads, hd]), to: .f32), axes: [2], epsilon: e), castUnlessSame(gain, to: .f32))
    }

    /// **A LayerNorm without affine of the stream (x̂, X)** in f16: u = x̂ · 2⁻¹⁴ (x̂ within 2¹⁴, u within 1, exact), the
    /// norm of u − ū read as the true value (u − ū) · 2¹⁴ · X (`norm`).
    public func layerNorm16(_ x: Int, _ xf: Int, eps: Double) -> Int {
        let u = scale("mul", x, 0x1p-14)
        return norm(binary(.sub, u, meanLast(u)), eps: eps, factors: [xf, scalar(0x1p14, .f16)])
    }

    /// **An update through a gate held as (ĝ, G)** — ĝ = g / G within 1, G a power of two (`Gate.factor`) — into the
    /// stream (x̂, X): the update p under `factors`, ĝ · p under them and G (`residual16(_:_:plus:factors:)`).
    public func gated16(_ x: Int, _ xf: Int, plus p: Int, gate: Int, gateFactor: Int, factors: [Int]) -> (value: Int, factor: Int) {
        residual16(x, xf, plus: binary(.mul, p, gate), factors: factors + [gateFactor])
    }

    /// **The exact GELU (erf) of a true value z in f16**: GELU(clip(z, ±16)) + max(z − 16, 0) — Φ is 1 or 0 in f16 past
    /// ±16, so that the primitive never reads a value its range does not hold.
    public func gelu16(_ z: Int) -> Int {
        let clipped = scale("minimum", scale("maximum", z, -16), 16)
        return binary(.add, gelu(clipped, tanh: false), scale("maximum", scale("sub", z, 16), 0))
    }
}

extension Aicode.Module {
    /// **The shared forms' product of `name`** on the unit x (`[1, rows, in]`, its input rotated before codes): `codes`
    /// on its codes `name` and scales `name_scale`, or `product16` on its f16 weight `name` — which, its argument's
    /// element says (`SharedWeights`). `adapted`, its low-rank path beside it (`name_down`, `name_up`,
    /// `Aicode.Block.lowRank`) reading `input`, the unit before its rotation under the same factor (`rotated16`'s): the
    /// change kept under the same. The rows `[1, rows, n]` read as `[rows, n]` by the low-rank path.
    ///
    /// `columns`: the product of those columns of the weight alone, its weight, scales and low-rank `up` sliced along
    /// their last axis in the function. The ANE's compiler refuses a product of 24 576 columns followed by its two
    /// halves (a fused SwiGLU input, 4 096 → 2 × 12 288): `ANECCompile() FAILED` at every load, the function falling
    /// back to the GPU without a word (a preferred unit's fallback) and the ANE's share computed at the GPU's pace; each
    /// half a product of its own compiles (20/11, the M1 Pro).
    ///
    /// `neural`: on codes, the GPU's function multiplies int8 × int8 on the codes by pieces `[p, out, in / p]`
    /// (`w8a8`), the ANE's `q64` on its copy `[1, in, out]`; on halves, both units the same.
    public func product(_ x: Int, _ name: String, input: @autoclosure () -> Int, adapted: Bool, columns: Range<Int>? = nil,
                        neural: Bool = false, constants: inout Aicode.ScalingConstants) -> Int {
        let b = body
        /// The columns of `t` along `axis` (its last, unless told).
        func part(_ t: Int, axis: Int? = nil) -> Int {
            guard let columns else { return t }
            return b.slice(t, axis: axis ?? Aicode.shape(b.types[t]).shape.count - 1, start: columns.lowerBound, size: columns.count)
        }
        let codes = Aicode.shape(b.types[argument(name)]).element == .si8
        let w = part(argument(name), axis: codes && !neural ? 1 : nil), s = { part(self.argument(name + "_scale")) }
        let y = codes ? b.codes(x, w, scale: s(), neural: neural, constants: &constants) : b.product16(x, weight: w)
        guard adapted else { return y }
        let input = input(), shape = Aicode.shape(b.types[y]).shape, inn = Aicode.shape(b.types[input]).shape.last!, out = shape.last!
        let rows = shape.reduce(1, *) / out
        let low = b.lowRank(b.reshape(y, [rows, out]), b.reshape(input, [rows, inn]), down: argument(name + "_down"), up: part(argument(name + "_up")))
        return b.reshape(low, shape)
    }
}

/// **A gate held as (ĝ, G)**: G the least power of two at or past the gate's largest magnitude, within f16's normal
/// exponents (2⁻¹⁴ to 2¹⁵), 1 for a gate of zeros — so that ĝ = g / G is within 1 and exact past G's division.
public enum Gate {
    public static func factor(largest top: Float) -> Float {
        let e = top > 0 ? min(max(Int(Foundation.log2(top).rounded(.up)), -14), 15) : 0
        return Float(sign: .plus, exponent: e, significand: 1)
    }
}
