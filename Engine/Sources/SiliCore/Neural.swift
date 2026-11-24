import CoreAI
import Foundation
import Metal

/// **What the ANE takes, written so the GPU takes it too** (the former engine's `NeuralStage` and `NeuralProduct`, 0.22): the
/// forms of a product, a norm and the rotation that the ANE runs whole, on `[1, rows, columns]` values in f16, no f32
/// anywhere a function's graph shows it.
///
///     peak       max(max|t| along the last axis, 2⁻¹⁴), kept `[1, rows, 1]`: what brings a row to 1
///     q64        x · (codes over 64) · 64 s: a `dequantize` by the constant 2⁻⁶, the scales after the sum (W8A16, the
///                ANE's, on its copy of the codes `[in, out]`)
///     w8a8       x's rows brought to int8 by their peaks, int8 × int8 summed on the GPU, the sums · peak · 64 s (W8A8,
///                the GPU's, on the codes by pieces of k)
///     norm       u = t / peak, u · rsqrt(mean(u²) + (√eps / (peak · F))²): RMS of the true value t · F
///     rotation   ConvRot's Hadamard by groups of 256 as a product at rank 4, `[1, rows, n / 256, 256] · K`
///
/// What a port by analogy would miss (measured 17/11 on the M1 Pro, `proto units`, 3072 rows × 3840 × 10240 on the
/// ANE beside 1024 on the GPU): **the ANE takes the codes as its own int8 weights only from a `dequantize` by a scalar
/// constant** — 7.6 TFLOP/s from the arena's `MTLBuffer` through a stream, 10.1 from IOSurfaces; the codes cast and
/// multiplied by 2⁻⁶ run at 4.8 and 5.6, and a `dequantize` by a vector of scales falls whole to the GPU (wrong on the
/// ANE besides, measured 12/11). Core AI's GPU runs q64 at the rate of its f16 product (22.1 ms for 22.0), nearer the
/// CPU (2.1e-4 for 3.0e-4: the scales multiply the sums, never a code), and both units compute at once (0.95 – 0.99 ×
/// the longer alone).
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

    /// **`x · W` on the ANE** for W int8 codes `[1, k, n]` with their scales `[1, 1, n]` (the ANE's copy of a group's codes,
    /// `Block`): x times the codes over 64, the sums times 64 s. The sums stay in f16: x comes under the product's
    /// target 2^e₀ (`rotated16`, `Recipe.target`), each term within 2 · 2^e₀ (|code| / 64 ≤ 2), k · 2 · 2^e₀ ≤ 65 504 by
    /// the target's cap (`Weights.target16`); times 64 s, the result the guard bounds. k cut as `Weights.pieces` cuts it. Measured 22/11/2026 on the M5 Pro (1024 × 3840 → 10 240):
    /// 17.8 TFLOP/s on `[in, out]`; the same codes `[out, in]` transposed in the function 15.7 (a fixed cost a call, the
    /// same from 512 to 3072 rows: W transposed at every call; on the M1 Pro 0.54 to 0.87 ×), transposed before their
    /// dequantize 15.7, read as W · xᵀ 11.8 (11.9 with x and y given channels first), as a 1 × 1 convolution 0.37.
    public func q64(_ x: Int, codes: Int, scale s: Int, constants: inout Aicode.ScalingConstants) -> Int {
        let w = dequantize(codes, scale: scalar(1.0 / 64, .f16), axis: 0, constants: &constants)
        return binary(.mul, pieces(x, w), scale("mul", s, 64))
    }

    /// **`x · W` int8 × int8 on the GPU** (W8A8), W the codes `[p, n, k / p]` by pieces of k (`Weights.pieces`) with their
    /// scales `[1, 1, n]` or `[n]`, x `[1, rows, k]` or `[rows, k]`: each row of x brought to its codes by its peak p over
    /// the whole row (`round(x · 127 / p)`, a half away from zero: the Compact's witness, int8 per row), each piece of x
    /// quantized under it and read under the constant 1/127, each piece of W under 1/64, each product NT, the pieces'
    /// sums summed, times p, times 64 s. Both operands of a product a `dequantize` by a scalar constant is the form Core
    /// AI's GPU runs as int8 × int8 summed in int32 (MPS's affine int8 product); a scale a row of x in its `dequantize`
    /// runs at 0.74 TFLOP/s and wrong (1.13 relative, measured 21/11/2026 on the M5 Pro), so the row's scale is taken out
    /// of the product and multiplied after: the same codes, the same sums. x's codes sliced after their quantization, or
    /// W's rows sliced, lose Core AI's fusion (≈ 24 TFLOP/s against 45, 22/11 on the M5 Pro); x's pieces quantized each
    /// and W's pieces whole keep it. **The sums stay in f16**: each term within 2 (|code| / 127 ≤ 1, |code| / 64 ≤ 2),
    /// k · 2 at most; times p · 64 s, the result the guard bounds (k · 128 · max s · p ≤ 65 504, `target`). Scales in f32 take MPS's fallback (0.3 – 1.1 TFLOP/s): every one here is f16. **The row's peak at 127 · 2⁻¹⁴
    /// at least**, not `peak`'s 2⁻¹⁴: its scale p / 127 then stays among f16's normals (2⁻¹⁴ / 127 is a subnormal, which a
    /// unit may flush to 0: a row of zeros quantized by 0 / 0). A row whose peak lies under it takes fewer than 127 codes,
    /// its error still within half its scale.
    public func w8a8(_ x: Int, codes: Int, scale s: Int, constants: inout Aicode.ScalingConstants) -> Int {
        let flat = Aicode.shape(types[x]).shape.count == 2
        let u = flat ? reshape(x, [1] + Aicode.shape(types[x]).shape) : x
        let shape = Aicode.shape(types[u]).shape, rows = shape[1], k = shape[2]
        let ws = Aicode.shape(types[codes]).shape, (count, piece, n) = (ws[0], ws[2], ws[1])
        precondition(count * piece == k, "w8a8: codes \(ws) for \(k) inputs")
        precondition(2 * k <= 65504, "w8a8: \(k) inputs, past what f16's sums hold (each term within 2)")
        let p = scale("maximum", reduceMax(primitive("abs", u), axes: [2]), 127 * 0x1p-14)
        let rs = reshape(scale("divide", p, 127), [rows])
        var sums: Int? = nil
        for i in 0..<count {
            let xi = count == 1 ? u : slice(u, axis: 2, start: i * piece, size: piece)
            let xq = dequantize(quantize(xi, scale: rs, axis: 1, constants: &constants), scale: scalar(1.0 / 127, .f16), axis: 0, constants: &constants)
            let wi = count == 1 ? codes : slice(codes, axis: 0, start: i, size: 1)
            let w = dequantize(wi, scale: scalar(1.0 / 64, .f16), axis: 0, constants: &constants)
            let y = matmul(xq, transpose(w, [0, 2, 1]), result: Aicode.tensor([1, rows, n], .f16))
            sums = sums.map { binary(.add, $0, y) } ?? y
        }
        // Under the guard (each row's peak p in [2^(e₀−1), 2^e₀], K · 128 · max s · 2^e₀ ≤ 65 504, `target`), p · 64 s is
        // within 65 504 / 2K, and its product with the sums is the result the guard bounds: no step past f16. The sums
        // times p (to 2K · 2^e₀) pass it where max s < 1/64; the sums times 64 s (to 128 K max s) where e₀ < 0.
        let y = binary(.mul, sums!, binary(.mul, p, scale("mul", s, 64)))
        return flat ? reshape(y, [rows, n]) : y
    }

    /// **A product on the Compact's codes**: int8 × int8 on the GPU (`w8a8`, the codes by pieces `[p, n, k / p]`), q64 on
    /// the ANE (`neural`, its copy of the codes `[1, k, n]`).
    public func codes(_ x: Int, _ w: Int, scale s: Int, neural: Bool, constants: inout Aicode.ScalingConstants) -> Int {
        neural ? q64(x, codes: w, scale: s, constants: &constants) : w8a8(x, codes: w, scale: s, constants: &constants)
    }

    /// `x · W` for W f16 `[1, k, n]` (`SharedWeights.half`): the same product, its sums the result, k cut as `q64`'s.
    public func product16(_ x: Int, weight w: Int) -> Int { pieces(x, w) }

    /// `x · w` in f16 `[1, rows, n]`, k cut as `Weights.pieces` cuts it, each piece a product, summed.
    private func pieces(_ x: Int, _ w: Int) -> Int {
        let rows = Aicode.shape(types[x]).shape[1], (k, n) = (Aicode.shape(types[w]).shape[1], Aicode.shape(types[w]).shape[2])
        let (pieces, piece) = Weights.written(k)
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
    public func rotation(_ x: Int, of weights: SharedWeights) -> Int { weights.isCodes ? rotation(x) : x }

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
        let (shape, element) = Aicode.shape(types[x]), g = Aicode.rotationGroup
        precondition(shape.count == 3 && shape[2] % g == 0, "rotation: \(shape) is not [1, rows, n] by groups of \(g)")
        let matrix = rotationMatrix(element)
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
    ///
    /// **The GPU's function computes F in f32, by `round`** (`ranged`, the f32 guard's: the largest in [2^(e₀−1), 2^e₀],
    /// within [−13, 15], F cast to f16 at the border, exact): `binade`'s count and `power16` are dozens of small ops a
    /// row between two passes, which the GPU runs one after the other. Measured 24/11/2026 on the M5 Pro, a shared block
    /// of 2048 channels at 4096 rows on the GPU alone: its three updates 4.70 → 4.42 ms, 2.55 → 2.16, 12.94 → 12.74, the
    /// same functions without any guard 3.43, 1.77, 11.32. The two units' rows may then differ in their bits, never in range.
    public func ranged16(_ t: Int, under e0: Double) -> (unit: Int, factor: Int) {
        neural ? ranged16(t, exponent: scale("sub", binade(peak(t)), e0 - 1)) : rounded16(t, under: scalar(e0, .f32))
    }

    /// The same with e₀ a value of the graph (an argument, `Recipe.target`): a label of its own, so that a literal exponent is never
    /// read as a value's index.
    public func ranged16(_ t: Int, target e0: Int) -> (unit: Int, factor: Int) {
        neural ? ranged16(t, exponent: scale("add", binary(.sub, binade(peak(t)), e0), 1)) : rounded16(t, under: castUnlessSame(e0, to: .f32))
    }

    /// The GPU's `ranged16`: the f32 guard's factor (`ranged`, t f16: F within f16's normals), in f16 at the border.
    private func rounded16(_ t: Int, under e0: Int) -> (unit: Int, factor: Int) {
        let (u, f) = ranged(t, under: e0)
        return (u, cast(f, to: .f16))
    }

    private func ranged16(_ t: Int, exponent e: Int) -> (unit: Int, factor: Int) {
        let f = power16(scale("minimum", scale("maximum", e, -13), 15))
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
