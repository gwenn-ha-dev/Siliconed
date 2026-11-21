import CoreAI
import Foundation

/// **A low-rank path beside a product**: `y + (x · down) · up`, the change `Δ = down · up` (`[in, out]`, the arena's
/// layout of a product's weight) never added into the weight.
///
/// **Why beside and not merged** (measured 18/11/2026 on the published weights of three transformers and low-rank changes
/// trained for them, `.build/coeur/glue/requant.py` and `fuse16.py`, the change's own error ‖Δ_kept − Δ‖ / ‖Δ‖ for a
/// Gaussian input): merged into int8 codes and requantized per row, a change of 0.1 % to 3 % of a quantization step —
/// most of the products measured — is rounded away whole (error 1.00, nothing of it along Δ), and one of 41 % of a step
/// keeps an error of 0.47; merged into bf16, the change loses 0.63 to 0.95 of itself on half the products measured
/// (‖Δ‖/‖W‖ of 10⁻⁴ to 10⁻³: under half an ulp of most entries), 0.08 to 0.15 on the others; merged into f16, 0.19 to
/// 0.68 on the small ones. Beside the product, the change is computed as it is, whatever the weight's element; what it
/// costs is two thin products, `rows · r · (in + out)`, against the product's `rows · in · out`.
///
/// **The element**: the factors' (`write`). bf16 or f32 factors compute in f32, x widened, the sum rounded once to y's
/// element; f16 factors compute in f16 — the element the ANE computes in —, kept in range by `write` (below).
extension Aicode.Block {
    /// `y + (x · down) · up`: x `[rows, in]` as the product reads it (an input kept under a factor gives a change kept
    /// under the same), down `[in, r]`, up `[r, out]`, y `[rows, out]`.
    public func lowRank(_ y: Int, _ x: Int, down: Int, up: Int) -> Int {
        let factors = Aicode.shape(types[down]).element, inner: Aicode.Element = factors == .f16 ? .f16 : .f32
        let rows = Aicode.shape(types[x]).shape[0], r = Aicode.shape(types[down]).shape[1], out = Aicode.shape(types[up]).shape[1]
        let t = matmul(castUnlessSame(x, to: inner), castUnlessSame(down, to: inner), result: Aicode.tensor([rows, r], inner))
        let d = matmul(t, castUnlessSame(up, to: inner), result: Aicode.tensor([rows, out], inner))
        return binary(.add, y, castUnlessSame(d, to: Aicode.shape(types[y]).element))
    }
}

public enum LowRank {
    /// **The rank a function is written at**: `rank` rounded up to 32, so that a row of `down` (`r` elements) is a
    /// multiple of 64 bytes in f16 as in f32 — what Core AI's GPU binds as it lies (`Arena.aligned`). The columns past
    /// the rank are zeros: they add nothing.
    public static func padded(_ rank: Int) -> Int { rank == 0 ? 0 : (rank + 31) / 32 * 32 }

    /// **The factors written into their places**: `down` f32 `[r, in]` and `up` f32 `[out, r]` (a change `up · down`
    /// `[out, in]`, as a file stores its factors; any scale already in `up`) into `[in, R]` and `[R, out]` of `element`
    /// (f16, bf16 or f32), R ≥ r, the columns past r zeros.
    ///
    /// **In f16, each column of the path is balanced by a power of two** (exact: no rounding of its own): down's column j
    /// times c = 2^⌊log₂(2 / ‖down_j‖₁)⌋, up's row j divided by it — so that |(x · down)_j| ≤ 2 · max |x| whatever x, the
    /// product's input in f16 under 2¹⁴ (`Guard`) leaving `x · down` under 2¹⁵. Throws when a factor still passes f16's
    /// range.
    public static func write(down: UnsafePointer<Float>, up: UnsafePointer<Float>, rank r: Int, inn: Int, out: Int,
                             into downPlace: UnsafeMutableRawPointer, _ upPlace: UnsafeMutableRawPointer, rank R: Int,
                             element: NDArray.ScalarType) throws {
        precondition(R >= r, "a rank of \(r) into a place of \(R)")
        var scale = [Float](repeating: 1, count: r)
        if element == .float16 {
            for j in 0..<r {
                var l1: Float = 0
                for k in 0..<inn { l1 += abs(down[j * inn + k]) }
                if l1 > 0 { scale[j] = Float(sign: .plus, exponent: Int((2 / l1).exponent), significand: 1) }
            }
        }
        func store(_ to: UnsafeMutableRawPointer, _ count: Int, _ value: (Int) -> Float) throws {
            switch element {
            case .float32:
                let p = to.bindMemory(to: Float.self, capacity: count)
                for i in 0..<count { p[i] = value(i) }
            case .float16:
                let p = to.bindMemory(to: Float16.self, capacity: count)
                for i in 0..<count {
                    let v = value(i)
                    guard abs(v) <= 65504 else { throw Failure("a low-rank factor of \(v) past f16's range") }
                    p[i] = Float16(v)
                }
            case .bfloat16:
                // Rounded to nearest, ties to even, as a cast to bf16 rounds.
                let p = to.bindMemory(to: UInt16.self, capacity: count)
                for i in 0..<count { p[i] = BFloat16Bits.rounded(value(i)) }
            default: preconditionFailure("low-rank factors in \(element)")
            }
        }
        try store(downPlace, inn * R) { i in
            let (k, j) = i.quotientAndRemainder(dividingBy: R)
            return j < r ? down[j * inn + k] * scale[j] : 0
        }
        try store(upPlace, R * out) { i in
            let (j, o) = i.quotientAndRemainder(dividingBy: out)
            return j < r ? up[o * r + j] / scale[j] : 0
        }
    }

    /// What the factors of a product of `inn` × `out` take at rank R in `element`: `[in, R]` and `[R, out]`.
    public static func bytes(inn: Int, out: Int, rank R: Int, element: NDArray.ScalarType) -> (down: Int, up: Int) {
        let size = element == .float32 ? 4 : 2
        return (inn * R * size, R * out * size)
    }
}
