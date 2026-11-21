import Foundation

/// **What the ANE takes, written so the GPU takes it too** (the former engine's `NeuralStage` and `NeuralProduct`, 0.22): the
/// forms of a product, a norm and the rotation that the ANE runs whole, on `[1, rows, columns]` values in f16, no f32
/// anywhere a function's graph shows it.
///
///     peak       max(max|t| along the last axis, 2⁻¹⁴), kept `[1, rows, 1]`: what brings a row to 1
///     q64        x · (codes over 64) · 64 s: a `dequantize` by the constant 2⁻⁶, the scales after the sum
///     norm       u = t / peak, u · rsqrt(mean(u²) + (√eps / (peak · F))²): RMS of the true value t · F
///     rotation   ConvRot's Hadamard by groups of 256 as a product at rank 4, `[1, rows, n / 256, 256] · K`
///
/// What a port by analogy would miss (measured 17/11 on the M1 Pro, `proto units`, 3072 rows × 3840 × 10240 on the
/// ANE beside 1024 on the GPU): **the ANE takes the codes as its own int8 weights only from a `dequantize` by a scalar
/// constant** — 7.6 TFLOP/s from the arena's `MTLBuffer` through a stream, 10.1 from IOSurfaces; the codes cast and
/// multiplied by 2⁻⁶ run at 4.8 and 5.6, and a `dequantize` by a vector of scales falls whole to the GPU (wrong on the
/// ANE besides, measured 12/11). Core AI's GPU runs q64 at its rate for today's form (22.1 ms for 22.0), nearer the
/// CPU (2.1e-4 for 3.0e-4: the scales multiply the sums, never a code), and both units compute at once (0.95 – 0.99 ×
/// the longer alone). The codes as the file lays them, transposed in the function, cost the GPU 1.7 ×.
/// **What the shared forms' products read**: ConvRot's int8 codes `[in, out]` and their scales (the Compact: the input
/// rotated by groups of 256, `q64`), or f16 weights `[in, out]` (the Standard beside the ANE: the publisher's bf16 rounded
/// once to f16 as they are read, `Recipe.halfTransposed`; the input as it is, `product16`). The ANE takes no bf16 at its
/// borders (DONE, « L'ANE par Core AI »), so the Standard's rows there are f16 under the guard, the GPU's beside them the
/// same functions on the same bytes: one copy of the weights. **bf16 → f16 is exact** within f16's normals (8 bits of
/// significand into 11); measured 20/11/2026 on five published transformers (2 to 12.5 billion weights each): no weight
/// past 65 504 (the largest 123), 0.03 to 0.72 % under 2⁻¹⁴, each rounded to f16's subnormals within 2⁻²⁵. The ANE
/// flushes those: at most 5 · 10⁻⁶ of a product's energy (the product with the most under 2⁻¹⁴: 3.7 % of its weights).
public enum SharedWeights: Sendable, Hashable {
    case codes, half

    /// A product's arguments `[1, in, out]`: the codes and their scales `[1, 1, out]`, or the f16 weight.
    public func arguments(_ name: String, inn: Int, out: Int) -> [(name: String, shape: [Int], element: Aicode.Element)] {
        self == .codes ? [(name, [1, inn, out], .si8), (name + "_scale", [1, 1, out], .f16)] : [(name, [1, inn, out], .f16)]
    }

    /// A product's entry from its file's key (`<key>.weight`, `[out, in]`): its codes and scales (`Recipe.codes`), or the
    /// weight rounded to f16 `[in, out]` (`Recipe.halfTransposed`).
    public func entry(_ name: String, key: String) -> Weights.Entry {
        self == .codes ? (name, key, .codes) : (name, key + ".weight", .halfTransposed)
    }
}

extension Aicode.Block {
    /// `[1, rows, 1]`: each row's largest magnitude, f16's smallest normal at least.
    public func peak(_ t: Int) -> Int {
        let rank = Aicode.shape(types[t]).shape.count
        return scale("maximum", reduceMax(primitive("abs", t), axes: [rank - 1]), 0x1p-14)
    }

    /// The mean along the last axis, kept: the composite `reduce_mean`, in x's element at its borders.
    public func meanLast(_ t: Int) -> Int {
        let (shape, element) = Aicode.shape(types[t])
        return reduceMean(t, integers([shape.count - 1], .si32), result: Aicode.tensor(Array(shape.dropLast()) + [1], element))
    }

    /// `x · W` for W int8 codes `[1, k, n]` with their scales `[1, 1, n]`: x times the codes over 64, the sums times 64 s.
    /// The sums stay in f16 while k · 2 · max|x| does: x brought to 1 first (`peak`). k cut as `pieces` cuts it.
    public func q64(_ x: Int, codes: Int, scale s: Int, constants: inout Aicode.ScalingConstants) -> Int {
        let w = dequantize(codes, scale: scalar(1.0 / 64, .f16), axis: 0, constants: &constants)
        return binary(.mul, pieces(x, w), scale("mul", s, 64))
    }

    /// `x · W` for W f16 `[1, k, n]` (`SharedWeights.half`): the same product, its sums the result, k cut as `q64`'s.
    public func product16(_ x: Int, weight w: Int) -> Int { pieces(x, w) }

    /// **The widest input a product of the shared forms reads at once**: past it, the ANE's rate falls. Measured by the
    /// former engine, then again on 20/11 on the M1 Pro, a transformer block of width 6144 at 1024 rows (the ANE alone):
    /// its feed-forward 194 ms read whole, 82 cut in two pieces of 3072 (3.2 → 7.5 TFLOP/s), the block 245 → 118 ms; the
    /// GPU's the same (171 ms either way). Every product: one that reads no more is one piece, unchanged.
    static let widest = 4096

    /// `x · w` in f16 `[1, rows, n]`, k cut in equal pieces of at most `widest` (multiples of 64), each a product, summed.
    private func pieces(_ x: Int, _ w: Int) -> Int {
        let rows = Aicode.shape(types[x]).shape[1], (k, n) = (Aicode.shape(types[w]).shape[1], Aicode.shape(types[w]).shape[2])
        var pieces = (k + Self.widest - 1) / Self.widest
        while k % pieces != 0 || (k / pieces) % 64 != 0 { pieces += 1 }
        let piece = k / pieces
        var sum: Int? = nil
        for i in 0..<pieces {
            let xp = pieces == 1 ? x : slice(x, axis: 2, start: i * piece, size: piece)
            let wp = pieces == 1 ? w : slice(w, axis: 1, start: i * piece, size: piece)
            let y = matmul(xp, wp, result: Aicode.tensor([1, rows, n], .f16))
            sum = sum.map { binary(.add, $0, y) } ?? y
        }
        return sum!
    }

    /// The input of a product of `weights`: x rotated (`rotation`) where the weights are ConvRot's codes, as it is where
    /// they are halves.
    public func rotation(_ x: Int, of weights: SharedWeights) -> Int { weights == .codes ? rotation(x) : x }

    /// The RMS norm of the true value t · F along the last axis (no gain), t brought to 1 by its peak: only the eps
    /// reads the size. `factors`: F's, each `[1, rows, 1]` (none: F = 1). √eps is divided by each in turn, so that no
    /// product of them leaves f16 (past it, eps is nothing and the quotient 0).
    public func norm(_ t: Int, eps: Double, factors: [Int] = []) -> Int {
        let p = peak(t), u = binary(.divide, t, p)
        var e = scale("divide", p, eps.squareRoot(), scalarFirst: true)
        for f in factors { e = binary(.divide, e, f) }
        return binary(.mul, u, rsqrt(binary(.add, meanLast(binary(.mul, u, u)), binary(.mul, e, e))))
    }

    /// ConvRot's rotation of `[1, rows, n]` by groups of 256, at rank 4: `[1, rows, n / 256, 256] · Kᵀ`, K's entries ±2⁻⁴.
    public func rotation(_ x: Int) -> Int {
        let (shape, element) = Aicode.shape(types[x])
        let m = 4, g = 1 << (2 * m)
        let (order, entries) = Aicode.kronecker(Aicode.regular, m)
        precondition(order == g && shape.count == 3 && shape[2] % g == 0, "rotation: \(shape) is not [1, rows, n] by groups of \(g)")
        let one = Aicode.roundedScalar(Double(sign: .plus, exponent: -m, significand: 1), element)
        let minus = Aicode.roundedScalar(-Double(sign: .plus, exponent: -m, significand: 1), element)
        let bytes = (0..<g * g).flatMap { i in entries[(i % g) * g + i / g] > 0 ? one : minus }
        let matrix = constant(Aicode.dense(bytes, splat: false), Aicode.tensor([g, g], element))
        let grouped = [1, shape[1], shape[2] / g, g]
        return reshape(matmul(reshape(x, grouped), matrix, result: Aicode.tensor(grouped, element)), shape)
    }
}

extension Aicode.Block {
    /// The guard's `ranged` in f16 alone, for a function the ANE takes: `t` `[1, rows, n]` f16 as t / F and F `[1, rows, 1]`
    /// f16, F = 2^e a row, e = m + 1 − e₀ within [−13, 15] for the row's peak p in (2^m, 2^(m+1)] (`binade`), so that
    /// the row's largest is in (2^(e₀−1), 2^e₀]; F = 2^max(e, 0) / 2^max(−e, 0), `pow` exact at an integer on both units.
    ///
    /// Measured 17/11 on the M1 Pro (`proto guard-units`, 1024 rows of 3840 spanning 2⁻¹⁴ to 2¹⁴; `proto probe`, the
    /// shared ffn): **a `round` anywhere in a function takes the ANE's product from 35 to 89 ms** (3072 rows × 3840 ×
    /// 10 240; `log`, `exp`, `pow`, `sqrt`, `maximum`, a division cost nothing), and e = round(log₂ p + ½) − e₀ left
    /// 44 rows of 1024 outside the bound, log₂ in f16 erring at a power of two; f16's magic add, (x + 1536) − 1536, does
    /// not round on either unit (875 factors not a power of two). t / F is exact on the GPU, and on the ANE but for 147
    /// values in 3.9 million — a −0 given as +0, a quotient under f16's smallest normal rounded to a step of its own.
    /// The engine's division by the peak, not a power of two, is inexact on the ANE in nearly every row.
    public func ranged16(_ t: Int, under e0: Double) -> (unit: Int, factor: Int) { ranged16(t, exponent: scale("sub", binade(peak(t)), e0 - 1)) }

    /// The same with e₀ a value of the graph (an argument, `Recipe.target`): a label of its own, so that a literal exponent is never
    /// read as a value's index.
    public func ranged16(_ t: Int, target e0: Int) -> (unit: Int, factor: Int) {
        ranged16(t, exponent: scale("add", binary(.sub, binade(peak(t)), e0), 1))
    }

    private func ranged16(_ t: Int, exponent e: Int) -> (unit: Int, factor: Int) {
        let e = scale("minimum", scale("maximum", e, -13), 15)
        func power(_ k: Int) -> Int { scale("pow", scale("maximum", k, 0), 2, scalarFirst: true) }
        let f = binary(.divide, power(e), power(scale("mul", e, -1)))
        return (binary(.divide, t, f), f)
    }

    /// m for p in (2^m, 2^(m+1)], p in [2⁻¹⁴, 65 504] (a peak), an exact integer in f16, with no `round`: the powers
    /// 2^k, k in [−16, 15], that p exceeds, counted, less 17. A step is (p − 2^k) over 2^k's ulp, clipped to [0, 1]: 0 or 1
    /// exactly, p past 2^k being past it by that ulp at least; the 16 under 1 compare p · 2¹⁰ with 2^(k+10), so that no
    /// difference is a subnormal (which the ANE may flush), the scales capped at 2¹⁵ (p · 2¹⁰ past f16, ∞, counts them
    /// all). Each half's mean over its 16 steps is exact.
    public func binade(_ p: Int) -> Int {
        let element = Aicode.shape(types[p]).element
        func vector(_ v: [Double]) -> Int {
            constant(Aicode.dense(v.flatMap { Aicode.roundedScalar($0, element) }, splat: false), Aicode.tensor([1, 1, v.count], element))
        }
        func count(_ x: Int, _ ks: ClosedRange<Int>, shift: Int) -> Int {
            let thresholds = vector(ks.map { Foundation.pow(2, Double($0 + shift)) })
            let scales = vector(ks.map { Foundation.pow(2, Double(min(10 - $0 - shift, 15))) })
            let steps = scale("minimum", scale("maximum", binary(.mul, binary(.sub, x, thresholds), scales), 0), 1)
            return scale("mul", meanLast(steps), Double(ks.count))
        }
        let high = count(p, 0...15, shift: 0), low = count(scale("mul", p, 1024), -16 ... -1, shift: 10)
        return scale("sub", binary(.add, high, low), 17)
    }
}

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
