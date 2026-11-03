import Foundation
import Siliconed

// **The judge — one rule for every boundary of every model.**
//
// A boundary is a tensor the engine computes and the reference also writes (a block's output, a
// latent, a decoded image). Its figure is the worst **per-channel** relative L2 error (`channelError`),
// or per latent plane (`errorPerPlane`): a global error would average a broken channel away.
//
// The rule, the same for all:
//
//     without a witness:   e(engine, oracle)  ≤ floor
//     with a witness:      e(engine, witness) ≤ max(witness floor, 1.25 · e(reference, witness))
//
// `oracle` is the reference implementation in fp32 on the CPU; `witness`, when the stage has one, is
// the same computation in fp64, and then it decides: the engine may be no further from the exact value
// than the reference is, plus 25 %. `reference` is the fp32 oracle — or, on a half-precision path, the
// worse of it and the reference run in bf16, since a bf16 path is judged against bf16.
//
// The witness floor is the noise of the statistic, not a tolerance. Per channel it is the floor
// (5·10⁻⁴): the worst position of an ill-conditioned boundary puts honest fp32 computations anywhere
// from 1× to 3.5× the oracle's error. Per latent plane there is none: thousands of values per measure,
// both sides near 10⁻⁶, and a floor would hide a hundredfold drift.

/// The figure of one boundary and its verdict.
package struct Measure: Sendable {
    package let boundary: String
    /// Against the fp32 oracle: worst channel, median channel, index of the worst.
    package let worst: Double
    package let median: Double
    package let channel: Int
    /// Against the fp64 witness, when one judged: the engine's distance and the reference's.
    package let atWitness: (engine: Double, reference: Double)?
    package let threshold: Double
    package let passed: Bool
    /// What a stage adds to the line (a shape mismatch, the bf16 reference, a stage-specific arbiter).
    package let note: String
    /// How the line reads: a tensor against its oracle (`channelError`), a figure against a bound in its
    /// own unit (ulp, dB, a relative error), or an identity (bit for bit).
    package var kind: Kind = .tensor

    /// The figure the verdict compared with `threshold`: the distance to the witness when one judged.
    package var judged: Double { atWitness?.engine ?? worst }

    package enum Kind: Sendable, Equatable {
        case tensor
        case bound(unit: String, above: Bool)
        case identity
    }
}

/// The rule above, for one boundary.
package enum Judge {
    /// The default floor: below it, a deviation from the fp32 oracle is accepted without a witness.
    package static let floor = 5e-4

    /// `engine` against `oracle`, channels on the last axis (`columns` wide). `witness` is the fp64
    /// computation of the same boundary, `half` the reference in bf16 when the engine runs a
    /// half-precision path; either may be absent. A shape mismatch fails, said.
    package static func boundary(_ name: String, engine: [Float], oracle: UnsafeBufferPointer<Float>,
                                 witness: UnsafeBufferPointer<Float>? = nil, half: [Float]? = nil,
                                 columns: Int, floor: Double = floor, witnessFloor: Double? = nil) -> Measure {
        guard oracle.count == engine.count else {
            return Measure(boundary: name, worst: .infinity, median: .infinity, channel: 0, atWitness: nil,
                           threshold: floor, passed: false,
                           note: "shape: \(engine.count) values against \(oracle.count)")
        }
        let vsOracle = channelError(engine, oracle, columns: columns)
        let half = half.flatMap { $0.count == engine.count ? $0 : nil }
        guard let witness, witness.count == engine.count else {
            // No witness: against the oracle; on a half-precision path, bf16's own distance + 25 %.
            let reference = half.map { channelError($0, oracle, columns: columns).worst } ?? 0
            let threshold = max(floor, 1.25 * reference)
            return Measure(boundary: name, worst: vsOracle.worst, median: vsOracle.median, channel: vsOracle.channel,
                           atWitness: nil, threshold: threshold, passed: vsOracle.worst <= threshold,
                           note: half == nil ? "" : String(format: "bf16 %.2e", reference))
        }
        let engineExact = channelError(engine, witness, columns: columns).worst
        var reference = channelError(Array(oracle), witness, columns: columns).worst
        if let half { reference = max(reference, channelError(half, witness, columns: columns).worst) }
        let threshold = max(witnessFloor ?? floor, 1.25 * reference)
        return Measure(boundary: name, worst: vsOracle.worst, median: vsOracle.median, channel: vsOracle.channel,
                       atWitness: (engineExact, reference), threshold: threshold, passed: engineExact <= threshold,
                       note: String(format: "oracle at fp64 %.2e", reference))
    }

    /// An exact comparison: bit for bit, or the count of differing values.
    package static func identical(_ name: String, _ engine: [Float], _ oracle: UnsafeBufferPointer<Float>) -> Measure {
        let differing = engine.count == oracle.count
            ? zip(engine, oracle).reduce(0) { $0 + ($1.0.bitPattern == $1.1.bitPattern ? 0 : 1) } : -1
        return identity(name, differing: differing, of: engine.count, against: oracle.count)
    }

    /// An identity already counted: `differing` values out of `count` (−1: the shapes differ).
    package static func identity(_ name: String, differing: Int, of count: Int, against other: Int? = nil,
                                 unit: String = "value") -> Measure {
        let units = count == 1 ? unit : unit + "s"
        return Measure(boundary: name, worst: differing == 0 ? 0 : .infinity, median: 0, channel: 0, atWitness: nil,
                       threshold: 0, passed: differing == 0,
                       note: differing == 0 ? "bit for bit, \(count) \(units)" : differing < 0
                           ? "shape: \(count) \(units) against \(other ?? 0)" : "\(differing) of \(count) \(units) differ",
                       kind: .identity)
    }

    /// A figure judged by a stage's own bound (a kernel against its definition, a counter-test that must
    /// fail): `value ≤ bound`, or `value > bound` when `above` (a counter-test proves the check can see).
    package static func bound(_ name: String, _ value: Double, _ bound: Double, unit: String = "",
                              above: Bool = false, note: String = "") -> Measure {
        Measure(boundary: name, worst: value, median: value, channel: 0, atWitness: nil, threshold: bound,
                passed: !value.isNaN && (above ? value > bound : value <= bound), note: note,
                kind: .bound(unit: unit, above: above))
    }

    /// The sampler's σ schedule against the reference's: a badly transcribed shift gives a plausible, wrong trajectory.
    package static func schedule(_ sigmas: [Float], _ reference: UnsafeBufferPointer<Float>) -> Measure {
        bound("σ", Double(zip(sigmas, reference).map { abs($0 - $1) }.max() ?? 0), 1e-6, unit: "max |Δ|")
    }

    /// A property that holds or not (a slice is page-aligned, a duplicate is refused).
    package static func holds(_ name: String, _ ok: Bool, note: String = "") -> Measure {
        Measure(boundary: name, worst: ok ? 0 : .infinity, median: 0, channel: 0, atWitness: nil, threshold: 0,
                passed: ok, note: note, kind: .identity)
    }
}

/// The sentences the DiT stages of the four models share, in `check list` and in the table of VERIFY.md.
enum Wording {
    static let rectangle = "on a rectangle, the transposed golden must deviate"
    static let halfPrecision = "with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention, against bf16's own error + 25 %"
    static let publishers = "(the Compact's or the Light's published weights, dequantized by their publisher's library)"
}
