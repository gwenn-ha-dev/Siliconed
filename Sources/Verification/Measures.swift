import Foundation
import Siliconed

// **The measures the stages share**, beside `channelError` (in the library, `Diagnostic.swift`, since a
// user's diagnostic reads the same figure): the error per latent plane, an image's PSNR, the
// orientation counter-test of a rectangle, and the bf16 reference of a half-precision path.

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

/// **`--neural-engine`: the DiT's products shared with the Neural Engine**, half the rows of every widened product
/// of 512 rows or more, whatever this Mac's map or profile says: the same split on every Mac, and more of the ANE than a
/// map gives — the strictest case. The ANE computes in fp16, so the stage is judged against the reference
/// run in bf16 (`HalfPrecisionReference`). Set through the environment, which has the last word over the
/// profile, before anything reads the engine's settings; the tally then says what the ANE computed.
package enum NeuralEngineRun {
    package static func enable() throws {
        setenv("SILICONED_GEMM_ANE", "1", 1)
        setenv("SILICONED_ANE_FRACTION", "0.5", 1)
        setenv("SILICONED_ANE_SHARES", "", 1)
        setenv("SILICONED_ANE_AMX", "", 1)   // the GPU alone beside the ANE: the pair judged, not the three
        // Every product of 512 rows or more, not only those the render's threshold takes (1024): the
        // conditions of an edit (1 021 rows for a short prompt) and its steps (988 at 512²) included.
        setenv("SILICONED_ANE_MIN", "\(2 * NeuralEngine.rowUnit)", 1)
        let settings = EngineSettings.effective
        guard settings.gemmANE, settings.aneFraction == 0.5, settings.aneShares.isEmpty, settings.aneAMX.isEmpty,
              settings.aneMinimumRows == 2 * NeuralEngine.rowUnit else {
            throw UsageError("--neural-engine: the engine's settings were read before it")
        }
    }

    /// How many of the DiT's products the ANE computed. None (no ANE on this Mac, every product refused):
    /// skipped, since what would be judged is the GPU's fp32 against a looser bar.
    package static func account(_ report: Report) throws {
        let (handed, givenBack) = GEMM.neuralEngineTally.withLock { $0 }
        guard handed > givenBack else {
            throw Skip("--neural-engine: the Neural Engine computed none of the DiT's products on this Mac"
                       + " (\(handed) handed, \(givenBack) given back to the GPU)")
        }
        report.add(Judge.holds("Neural Engine", true,
                               note: "half the rows of \(handed - givenBack) products"
                                   + (givenBack > 0 ? ", \(givenBack) given back to the GPU" : "")))
    }
}

/// **`--bf16-activations`: the DiT's products on an activation rounded to bf16**, the weights as published, the
/// sums and the output fp32, every product of 256 rows or more (the engine's own rule). The strictest case of
/// the map's setting: on any Mac, whatever its map says. Judged like the ANE's, against the reference run in
/// bf16 (`HalfPrecisionReference`): invariant 3, no further from the fp64 witness than diffusers in bf16, + 25 %.
package enum BF16ActivationRun {
    package static func enable() throws {
        setenv("SILICONED_GEMM_BF16_ACT", "1", 1)
        setenv("SILICONED_BF16_ACT_MIN", "256", 1)
        guard EngineSettings.effective.gemmBF16Activations else {
            throw UsageError("--bf16-activations: the engine's settings were read before it")
        }
    }

    /// How many products rounded their activation. None (no Metal 4, a refused compiler): skipped, since what
    /// would be judged is the fp32 path against a looser bar.
    package static func account(_ report: Report) throws {
        let rounded = GEMM.roundedProducts.withLock { $0 }
        guard rounded > 0 else {
            throw Skip("--bf16-activations: no product rounded its activation on this Mac (Metal 4 missing, or the kernel refused)")
        }
        report.add(Judge.holds("bf16 activations", true, note: "\(rounded) products on a rounded operand"))
    }
}

/// **Invariant 3 for a half-precision path** (the ANE's products in fp16): the reference that sets the
/// bar is the reference run in bf16 on the golden's inputs, beside the golden as `witness-bf16-X`.
/// `nil` when the engine's path is fp32, or the bf16 reference is missing (said: the fp32 rule applies).
package struct HalfPrecisionReference {
    let tensors: Safetensors

    package init?(goldens: Goldens, file: String) {
        guard EngineSettings.effective.gemmANE || EngineSettings.effective.gemmBF16Activations else { return nil }
        // `witness-bf16-X`, or Qwen-Image-2.1's own name for it (`goldens-qwen21-dit-bf16-R`).
        let names = [file.replacingOccurrences(of: "goldens-", with: "witness-bf16-"),
                     file.replacingOccurrences(of: "-dit-", with: "-dit-bf16-")]
        guard let found = names.lazy.compactMap({ goldens.optional($0) }).first else {
            print("a half-precision path is on and no bf16 reference beside \(file): the fp32 rule applies")
            return nil
        }
        tensors = found
    }

    /// The bf16 reference's values at `name`, if it holds that boundary.
    package func values(_ name: String) -> [Float]? { tensors.float32(name).map(Array.init) }
}
