import CoreAI
import Foundation
import Metal

/// **What the shared forms' products read**: ConvRot's int8 codes and their scales, the input rotated by groups of 256
/// (the Compact), or f16 weights `[in, out]` (the Standard beside the ANE: the publisher's bf16 rounded once to f16 as
/// they are read, `Recipe.halfTransposed`; the input as it is, `product16`). The ANE takes no bf16 at its borders, so the
/// Standard's rows there are f16 under the guard, the GPU's beside them the same functions on the same bytes: one copy
/// of the weights. **bf16 → f16 is exact** within f16's normals (8 bits of significand into 11); measured 20/11/2026 on
/// five published transformers (2 to 12.5 billion weights each): no weight past 65 504 (the largest 123), 0.03 to 0.72 %
/// under 2⁻¹⁴, each rounded to f16's subnormals within 2⁻²⁵. The ANE flushes those: at most 5 · 10⁻⁶ of a product's
/// energy (the product with the most under 2⁻¹⁴: 3.7 % of its weights).
///
/// **The codes, one layout for both units**: the GPU's `[p, out, in / p]`, the file's rows by pieces of k
/// (`Weights.pieces`), int8 × int8 (`w8a8`); the ANE's a copy `[1, in, out]` of the codes, one each, made on the CPU while
/// the ANE is idle (`Block`), q64. Each unit reads its own layout at its rate: Core AI's GPU takes the integer product NT only (the
/// codes `[in, out]`, it runs at q64's rate), the ANE `[in, out]` only (`[out, in]` transposed at every call: a fixed
/// cost, 0.54 to 0.87 × on the M1 Pro, 0.89 × on the M5 Pro at 1024 rows). Measured 22/11/2026, 4096 rows, TFLOP/s, q64
/// against int8 × int8 by pieces: the M5 Pro 24.3 – 24.8 against 41.2 – 47.6 (k/v 22.2 against 31.2); the M1 Pro 3.64 –
/// 3.95 against 3.55 – 3.98.
public enum SharedWeights: Sendable, Hashable {
    case codes, half

    /// The weights are the Compact's int8 codes.
    public var isCodes: Bool { self == .codes }

    /// A product's arguments: the codes by pieces `[p, out, in / p]` on the GPU, the ANE's copy `[1, in, out]` (`neural`),
    /// and their scales `[1, 1, out]`; or the f16 weight `[1, in, out]`.
    public func arguments(_ name: String, inn: Int, out: Int, neural: Bool) -> [(name: String, shape: [Int], element: Aicode.Element)] {
        switch self {
        case .codes:
            return [(name, neural ? [1, inn, out] : Weights.codesShape(inn: inn, out: out), .si8), (name + "_scale", [1, 1, out], .f16)]
        case .half: return [(name, [1, inn, out], .f16)]
        }
    }

    /// A product's entry from its file's key (`<key>.weight`, `[out, in]`): its codes and scales (`Recipe.codes`), or the
    /// weight rounded to f16 `[in, out]` (`Recipe.halfTransposed`).
    public func entry(_ name: String, key: String) -> Weights.Entry {
        isCodes ? (name, key, .codes) : (name, key + ".weight", .halfTransposed)
    }
}
