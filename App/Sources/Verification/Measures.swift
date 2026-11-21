import Foundation
import Siliconed

// **The measures the stages share**, beside `channelError` (`ChannelError.swift`):
// the error per latent plane, an image's PSNR, the
// orientation counter-test of a rectangle.

/// **The relative L2 error per plane** of a `[planes, h, w]` tensor. `channelError` takes its channels on
/// the **last** axis, which would make each pixel of a `[16, h, w]` latent a channel; here a plane is
/// a channel of the latent. Same floor (1 % of the median norm) and same handling of `NaN`: a
/// non-finite value counts as infinity and fails any threshold.
package func errorPerPlane(_ got: [Float], _ expected: UnsafeBufferPointer<Float>, planes: Int)
        -> (worst: Double, plane: Int, median: Double) {
    let size = expected.count / planes
    var numerators = [Double](repeating: 0, count: planes), norms = [Double](repeating: 0, count: planes)
    for p in 0..<planes {
        for i in (p * size)..<((p + 1) * size) {
            let d = Double(got[i]) - Double(expected[i])
            numerators[p] += d * d
            norms[p] += Double(expected[i]) * Double(expected[i])
        }
    }
    let sortedNorms = norms.map { $0.squareRoot() }.sorted()
    let floor = 0.01 * sortedNorms[planes / 2]
    var worst = 0.0, plane = 0, all: [Double] = []
    for p in 0..<planes {
        var r = numerators[p].squareRoot() / max(norms[p].squareRoot(), floor)
        if r.isNaN { r = .infinity }
        all.append(r)
        if r > worst { worst = r; plane = p }
    }
    all.sort()
    return (worst, plane, all[all.count / 2])
}

/// An image's PSNR against its reference (values in [−1, 1]), **and the count of non-finite values**:
/// `max(0, nan)` is `0` in Swift, and a `NaN` once read as "max deviation 0" beside a `nan` PSNR.
package func imageError(_ image: [Float], _ expected: UnsafeBufferPointer<Float>)
        -> (psnr: Double, rmse: Double, worst: Double, notFinite: Int) {
    var squared = 0.0, worst = 0.0, notFinite = 0
    for i in 0..<image.count {
        guard image[i].isFinite else { notFinite += 1; continue }
        let delta = Double(image[i]) - Double(expected[i])
        squared += delta * delta
        worst = max(worst, abs(delta))
    }
    let rmse = (squared / Double(image.count)).squareRoot()
    return (20 * log10(2.0 / rmse), rmse, worst, notFinite)
}

extension Judge {
    /// The rule of `boundary`, per latent plane (`errorPerPlane`): for a `[planes, h, w]` tensor. No
    /// witness floor by default: a plane is a stable statistic (see the rule, `Judge.swift`).
    package static func planes(_ name: String, engine: [Float], oracle: UnsafeBufferPointer<Float>,
                               witness: UnsafeBufferPointer<Float>? = nil, reference: [Float]? = nil, planes: Int,
                               floor: Double = floor, witnessFloor: Double = 0) -> Measure {
        guard oracle.count == engine.count else {
            return Measure(boundary: name, worst: .infinity, median: .infinity, channel: 0, atWitness: nil,
                           threshold: floor, passed: false, note: "shape: \(engine.count) values against \(oracle.count)")
        }
        let vsOracle = errorPerPlane(engine, oracle, planes: planes)
        guard let witness, witness.count == engine.count else {
            return Measure(boundary: name, worst: vsOracle.worst, median: vsOracle.median, channel: vsOracle.plane,
                           atWitness: nil, threshold: floor, passed: vsOracle.worst <= floor, note: "")
        }
        let engineExact = errorPerPlane(engine, witness, planes: planes).worst
        // `reference`: the bf16 run's values on a half-precision path, the fp32 oracle's otherwise.
        let referenceExact = errorPerPlane(reference ?? Array(oracle), witness, planes: planes).worst
        let threshold = max(witnessFloor, 1.25 * referenceExact)
        return Measure(boundary: name, worst: vsOracle.worst, median: vsOracle.median, channel: vsOracle.plane,
                       atWitness: (engineExact, referenceExact), threshold: threshold, passed: engineExact <= threshold,
                       note: String(format: "oracle at fp64 %.2e", referenceExact))
    }

    /// **A Compact's render is looked at, not held to a bar.** Eight free steps on int8 products part from the fp32
    /// picture as any other arithmetic's do (Z-Image: 22.12 dB at 512, a sound picture), and a bar in dB would judge
    /// where the trajectory forks, not the products: those are judged at the DiT, beside the publisher's own int8 run.
    /// What the render still proves is that the whole chain ends on a finite picture; its PSNR is said, not judged.
    package static func compactRender(_ pixels: [Float], _ expected: UnsafeBufferPointer<Float>) -> Measure {
        let (psnr, _, _, notFinite) = imageError(pixels, expected)
        return holds("image", notFinite == 0, note: notFinite > 0 ? "\(notFinite) non-finite values"
                        : String(format: "finite; %.2f dB from the fp32 picture, said and not judged", psnr))
    }

    /// Integers (identifiers, positions, masks) against the golden's: identical, or how many differ. A
    /// golden that lacks them fails: the oracle wrote what the stage reads.
    package static func integers(_ name: String, _ ours: [Int], _ golden: UnsafeBufferPointer<Int32>?) -> Measure {
        guard let golden else { return holds(name, false, note: "the golden lacks it") }
        guard golden.count == ours.count else { return identity(name, differing: -1, of: ours.count, against: golden.count) }
        return identity(name, differing: zip(ours, golden).filter { $0 != Int($1) }.count, of: ours.count)
    }

    /// Floats that must be the golden's bit for bit; a golden that lacks them fails.
    package static func identical(_ name: String, _ engine: [Float], _ oracle: UnsafeBufferPointer<Float>?) -> Measure {
        guard let oracle else { return holds(name, false, note: "the golden lacks it") }
        return identical(name, engine, oracle)
    }

    /// A boundary the engine should have produced and did not: a failure, said.
    package static func missing(_ name: String) -> Measure {
        Measure(boundary: name, worst: .infinity, median: .infinity, channel: 0, atWitness: nil, threshold: floor,
                passed: false, note: "not produced by the engine")
    }

    /// **The orientation counter-test of a rectangle.** A check at `h ≠ w` catches a height/width swap only
    /// if the golden tells the two apart: `got` `[C, h, w]` against the golden's transpose, plane by plane,
    /// must deviate clearly (> 0.1). `nil` for a square.
    package static func orientation(_ got: [Float], _ expected: UnsafeBufferPointer<Float>,
                                    height: Int, width: Int) -> Measure? {
        guard height != width else { return nil }
        let plane = height * width, planes = expected.count / plane
        var numerator = 0.0, denominator = 0.0
        for c in 0..<planes {
            // The transpose `[w, h]` of plane `c`, stored by rows: element (x, y) is golden[y, x].
            for x in 0..<width {
                for y in 0..<height {
                    let t = Double(expected[c * plane + y * width + x])
                    let g = Double(got[c * plane + x * height + y])
                    numerator += (g - t) * (g - t); denominator += t * t
                }
            }
        }
        return bound("counter-test: transposed", (numerator / max(denominator, 1e-30)).squareRoot(), 0.1,
                     unit: "relative", above: true, note: "must deviate, or the check is blind to orientation")
    }
}
