import Foundation

extension Aicode.Block {
    /// The residual stream's update in f16 (the guard's `residual`): x̂ kept under X `[1, rows, 1]`, plus Δ a true value,
    /// both brought to half the larger factor C = 2 · max(X, 4), summed, the sum brought back to [2¹³, 2¹⁴]: (x̂', C · F).
    /// Every factor a power of two, every quotient exact; C past 2¹⁵ (a stream past 2²⁹) leaves f16.
    public func residual16(_ x: Int, _ xf: Int, plus delta: Int) -> (value: Int, factor: Int) {
        let c = scale("mul", scale("maximum", xf, 4), 2)
        let sum = binary(.add, binary(.mul, x, binary(.divide, xf, c)), binary(.mul, delta, scale("divide", c, 1, scalarFirst: true)))
        let (u, f) = ranged16(sum, under: 14)
        return (u, binary(.mul, c, f))
    }
}

/// **The shared forms' residual where no norm bounds the update** (`residual16(_:_:plus:)`, above, reads Δ as a true value
/// in f16, which a transformer without a sandwich norm leaves: its update is a product's result under its target, times
/// the factors of every input it came from). Δ is kept as (Δ̂, G), G = ∏ Gᵢ, each Gᵢ a power of two `[1, rows, 1]`
/// in f16 (`ranged16`'s). The product of the Gᵢ is never formed: a feed-forward's five factors can leave f16 on the way
/// where their product does not. Every power is counted by its exponent instead (`binade`, exact on both units, no
/// `round`), summed, and applied as a power of two.
extension Aicode.Block {
    /// k for f = 2^k, f a power of two within f16's normals: `binade` + 1, an exact integer in f16.
    public func exponent16(_ f: Int) -> Int { scale("add", binade(f), 1) }

    /// 2^e for e an integer within [−14, 15] in f16: a quotient of two powers of a non-negative exponent, `pow` exact at an
    /// integer on both units (`ranged16`'s).
    public func power16(_ e: Int) -> Int {
        func power(_ k: Int) -> Int { scale("pow", scale("maximum", k, 0), 2, scalarFirst: true) }
        return binary(.divide, power(e), power(scale("mul", e, -1)))
    }

    /// t · 2^e for e ≤ 0 an integer, possibly under −14: in two steps, each a normal power of f16, the first leaving t
    /// within its normals; a term brought past 2⁻²⁸ is nothing beside the one it is summed with, which lies above 2⁻¹.
    public func scaled16(_ t: Int, byPower e: Int) -> Int {
        let first = scale("maximum", e, -14), second = scale("maximum", binary(.sub, e, first), -14)
        return binary(.mul, binary(.mul, t, power16(first)), power16(second))
    }

    /// x̂ under X `[1, rows, 1]` plus Δ̂ under G = ∏ `factors`, both brought under C = 2 · max(X, 4, 4G) (C's exponent
    /// counted: kc = 1 + max(kX, 2, kG + 2)), summed — each term under 2¹³ —, the sum brought back to [2¹³, 2¹⁴]:
    /// (x̂', C · F). Δ̂ within f16 (a product's result under its target), X and C within f16's normals (a stream under
    /// 2²⁹, as `residual16`).
    public func residual16(_ x: Int, _ xf: Int, plus delta: Int, factors: [Int]) -> (value: Int, factor: Int) {
        let kx = exponent16(xf)
        let kg = factors.dropFirst().reduce(exponent16(factors[0])) { binary(.add, $0, exponent16($1)) }
        let kc = scale("add", elementwise("maximum", scale("maximum", kx, 2), scale("add", kg, 2)), 1)
        let sum = binary(.add, scaled16(x, byPower: binary(.sub, kx, kc)), scaled16(delta, byPower: binary(.sub, kg, kc)))
        let (u, f) = ranged16(sum, under: 14)
        return (u, binary(.mul, power16(kc), f))
    }
}
