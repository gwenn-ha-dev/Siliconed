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

/// **The strictest case of a half-precision route, on any Mac whatever its map says** — designated as the process's plan
/// (`CapacityPlan.forcing`) before anything reads the engine's settings, so that the same rows run on every Mac:
///
///   - **`--neural-engine`**: half the rows of every widened product of 512 rows or more go to the Neural Engine, the GPU
///     beside it alone (the pair judged, not the three). The ANE computes in fp16, so the stage is judged against the reference
///     run in bf16 (`HalfPrecisionReference`); the tally then says what the ANE computed.
///   - **`--bf16-activations`**: the DiT's products on an activation rounded to bf16, the weights as published, the sums and the
///     output fp32, every product of 256 rows or more. Judged like the ANE's, against the reference run in bf16: invariant 3, no
///     further from the fp64 witness than diffusers in bf16, + 25 %. **With `--neural-engine`**, the pair a Standard's map may choose:
///     half the rows of every bf16 product of 512 rows or more to the ANE's fp16 program on the same bf16 copy, beside the GPU's
///     rounded rows (`gpu+ane/bf16act`).
///   - **`--bf16-attention`**: the DiT's attention on bf16 operands (Q, K, V and the probabilities rounded once, the softmax and the
///     sums fp32), in the fused form at the first candidate tile — every attention the plan speaks of. Judged the same way.
///   - **`--fp16-attention`**: the same on fp16 operands (`tiles16`) — what a render takes at every version where its map chooses
///     it, the Standard's included (`CapacitySolver.Constraints.allowsHalf`, decided 3/11/2026). Judged the same way: against the
///     reference run in bf16, + 25 %, as the ANE's fp16 products.
package enum HalfPrecisionRun {
    package static func enable(neuralEngine: Bool, bf16Activations: Bool, bf16Attention: Bool = false, fp16Attention: Bool = false) throws {
        do {
            try EngineSettings.designate(plan: .forcing(neuralEngine: neuralEngine, bf16Activations: bf16Activations,
                                                        tiles16: fp16Attention, tilesbf16: bf16Attention,
                                                        fused: bf16Attention || fp16Attention))
        } catch {
            let flags = [neuralEngine ? "--neural-engine" : nil, bf16Activations ? "--bf16-activations" : nil,
                         bf16Attention ? "--bf16-attention" : nil, fp16Attention ? "--fp16-attention" : nil].compactMap { $0 }
            throw UsageError("\(flags.joined(separator: " ")): the engine's settings were read before it")
        }
    }
}

/// **`--no-accelerators`: a render's plan with the « Accelerators » switch off.** The check runs under this Mac's map (`SILICONED_MAP`,
/// a report of this Mac and this build) restricted to fp32 (`CapacityPlan.restricted(accelerators: false)`): the routes the map
/// chooses — the AMX's rows, a weight read as published, the tiled attention — but no ANE and no operand rounded below fp32, judged by
/// the fp32 rule. Without such a map the defaults run, which are fp32 already: the flag changes nothing, and says so.
package enum FP32Run {
    package static func enable() throws {
        guard let named = ProcessInfo.processInfo.environment["SILICONED_MAP"], !named.isEmpty, named != "1",
              let data = FileManager.default.contents(atPath: named), let report = Diagnostic.Report.read(data), report.isCurrent
        else {
            print("--no-accelerators: no map of this Mac and this build in SILICONED_MAP — the defaults run, fp32 already")
            return
        }
        let plan = report.plan().restricted(accelerators: false)
        do { try EngineSettings.designate(plan: plan) }
        catch { throw UsageError("--no-accelerators: the engine's settings were read before it") }
        print("--no-accelerators: this Mac's map, fp32 throughout (\(plan.identity))")
    }
}

/// **The int8 route of a published int8 weight** (`gemm_int8`). A render of a map holding weights published as int8 with a scale
/// per column or per tensor (a Compact from Comfy-Org's int8 convrot) reads them as published wherever its map says the int8
/// kernel wins, else widened; `siliconed check` runs the defaults, which widen. A DiT stage on such a map therefore evaluates along
/// the int8 route (`plan`), says how many products read the integers (`account`), and judges its output once more widened,
/// by the same rule: both routes a render can take. Under `--bf16-activations` the process's plan reads them already, on the rounded
/// activation (`gpu/int8act`), as it reads a GGUF Q8_0's blocks (`gpu/q8act`): this route then stands aside.
package struct Int8Route {
    /// The plan the DiT under test runs under: `nil` when the process's plan already reads the integers (`SILICONED_PLAN=int8`).
    package let plan: CapacityPlan?
    /// The process's plan does not read the integers: the stage judges the widened route too.
    package var judgesWidened: Bool { plan != nil }
    /// The engine's neural-engine flag for the DiT built on `plan`.
    package var neuralEngine: Bool { (plan ?? EngineSettings.effective.plan).usesANE }
    private let before: Int

    /// `nil` when `artifact` publishes no int8 weight `gemm_int8` reads.
    package init?(_ artifact: Artifact, _ invocation: Invocation) {
        guard artifact.tensors.values.contains(where: { $0.dtype == .int8 && $0.scale != nil && $0.scale?.block == nil }) else {
            return nil
        }
        plan = Self.plan(invocation)
        before = GEMM.int8Products.withLock { $0 }
    }

    /// The process's plan with `int8`: the invocation's half-precision flags, or the developer's `SILICONED_PLAN` (which has the
    /// last word), plus the published integers read as they are, whatever this Mac's map says.
    private static func plan(_ invocation: Invocation) -> CapacityPlan? {
        if EngineSettings.effective.plan.resources.int8Weights { return nil }
        if let named = ProcessInfo.processInfo.environment["SILICONED_PLAN"], !named.isEmpty,
           let forced = CapacityPlan.forcing(named: named + ",int8") { return forced }
        return .forcing(neuralEngine: invocation.neuralEngine, bf16Activations: invocation.bf16Activations,
                        tiles16: invocation.fp16Attention, tilesbf16: invocation.bf16Attention,
                        fused: invocation.bf16Attention || invocation.fp16Attention, int8: true)
    }

    /// The products that read the published integers since the route was taken: none fails, the route was not crossed.
    package func account(_ report: Report) {
        let read = GEMM.int8Products.withLock { $0 } - before
        report.add(Judge.holds("gemm_int8", read > 0, note: "\(read) products on the published integers"))
    }
}

package enum NeuralEngineRun {
    /// How many of the DiT's products the ANE computed. None (no ANE on this Mac, every product refused):
    /// skipped, since what would be judged is the GPU's fp32 against a looser bar.
    package static func account(_ report: Report) throws {
        let (handed, givenBack) = GEMM.neuralEngineTally.withLock { $0 }
        guard handed > givenBack else {
            throw Skip("--neural-engine: the Neural Engine computed none of the DiT's products on this Mac"
                       + " (\(handed) handed, \(givenBack) given back to the GPU)")
        }
        // Under `SILICONED_PLAN=aneint8` the integers of a published int8 weight are the program's input: the verdict says how many
        // of the products read them so.
        // With `--bf16-activations` the ANE reads a bf16 weight's copy beside the GPU's rounded rows (`gpu+ane/bf16act`): said too.
        let integers = NeuralEngine.int8Tally.withLock { $0 }, copies = NeuralEngine.bf16Tally.withLock { $0 }
        report.add(Judge.holds("Neural Engine", true,
                               note: "half the rows of \(handed - givenBack) products"
                                   + (integers > 0 ? ", \(integers) of them on the weight's integers" : "")
                                   + (copies > 0 ? ", \(copies) of them on the bf16 weight beside the GPU's rounded rows" : "")
                                   + (givenBack > 0 ? ", \(givenBack) given back to the GPU" : "")))
    }
}

package enum BF16AttentionRun {
    /// How many attentions ran on bf16 operands. None (no Metal 4, a refused compiler, a shape the kernel does not take): skipped.
    package static func account(_ report: Report) throws {
        let runs = TiledAttention.bf16Runs.withLock { $0 }
        guard runs > 0 else {
            throw Skip("--bf16-attention: no attention ran on bf16 operands on this Mac (Metal 4 missing, or the kernel refused)")
        }
        report.add(Judge.holds("bf16 attention", true, note: "\(runs) attentions on bf16 operands"))
    }
}

package enum FP16AttentionRun {
    /// How many attentions ran on fp16 operands. None (no Metal 4, a refused compiler, a shape the kernel does not take): skipped.
    package static func account(_ report: Report) throws {
        let runs = TiledAttention.fp16Runs.withLock { $0 }
        guard runs > 0 else {
            throw Skip("--fp16-attention: no attention ran on fp16 operands on this Mac (Metal 4 missing, or the kernel refused)")
        }
        report.add(Judge.holds("fp16 attention", true, note: "\(runs) attentions on fp16 operands"))
    }
}

package enum BF16ActivationRun {
    /// How many products rounded their activation, and how many read a published 8-bit weight (a Compact's or a Light's: under
    /// `--bf16-activations` they are read as published on the rounded activation, `gpu/int8act` or `gpu/q8act`, and a Light's Q4_K and
    /// Q5_K blocks `gpu/q4kact` (its Q6_K widened) — a rounded activation
    /// has no widened route). None rounded (no Metal 4, a refused compiler): skipped, since what would be judged is the fp32 path against
    /// a looser bar.
    package static func account(_ report: Report) throws {
        let rounded = GEMM.roundedProducts.withLock { $0 }
        guard rounded > 0 else {
            throw Skip("--bf16-activations: no product rounded its activation on this Mac (Metal 4 missing, or the kernel refused)")
        }
        let integers = GEMM.int8Products.withLock { $0 }, kquants = GEMM.kquantProducts.withLock { $0 }
        report.add(Judge.holds("bf16 activations", true, note: "\(rounded) products on a rounded operand"
                                   + (integers > 0 ? ", \(integers) products on published 8-bit weights" : "")
                                   + (kquants > 0 ? ", \(kquants) on published K-quant blocks" : "")))
    }
}

/// **Invariant 3 for a half-precision path** (the ANE's products in fp16): the reference that sets the
/// bar is the reference run in bf16 on the golden's inputs, beside the golden as `witness-bf16-X`.
/// `nil` when the engine's path is fp32, or the bf16 reference is missing (said: the fp32 rule applies).
package struct HalfPrecisionReference {
    let tensors: Safetensors

    package init?(goldens: Goldens, file: String) {
        guard EngineSettings.effective.plan.resources.halfPrecision else { return nil }
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
