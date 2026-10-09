import Accelerate
import Foundation

/// The flow-matching sampler of Z-Image Turbo.
///
/// Three traps, all verified against the reference at step 1:
///
///   - **`sigma_min` is forced to zero by the pipeline** (`pipeline_z_image.py:477`), not by the
///     scheduler. Hence a final sigma of zero, a final step that moves nothing, and **`N` steps =
///     `N−1` evaluations**. The bare scheduler leaves 0.008929 and would cost one more evaluation.
///   - **Time is inverted**: the DiT receives `(1000 − t)/1000 = 1 − σ`.
///   - **The output is negated**: `noise_pred = −v`.
package struct FlowMatchSchedule {
    package let sigmas: [Float]          // `steps + 1` values, ending in zero

    package init(steps: Int, shift: Float = 3.0) {
        // `linspace(1, 0, steps)` — `steps` points, endpoints included — then the static shift.
        // `use_dynamic_shifting` is false in the published config, so `mu` is computed for nothing.
        var values: [Float] = (0..<steps).map { i in
            let raw = steps == 1 ? 1 : 1 - Float(i) / Float(steps - 1)
            return shift * raw / (1 + (shift - 1) * raw)
        }
        values.append(0)
        sigmas = values
    }

    /// What the DiT receives: the inverted time.
    package func modelTime(at step: Int) -> Float { 1 - sigmas[step] }

    /// The steps that actually move the latent. At 8 steps, there are 7.
    package var effectiveSteps: Int {
        (0..<(sigmas.count - 1)).filter { sigmas[$0 + 1] != sigmas[$0] }.count
    }
}

package final class Sampler {
    package let dit: DiT
    package let schedule: FlowMatchSchedule

    /// What each step produced, to compare against the golden trajectory.
    package private(set) var velocities: [[Float]] = []
    package private(set) var latents: [[Float]] = []
    package var recordTrajectory = false

    package init(dit: DiT, steps: Int) {
        self.dit = dit
        self.schedule = FlowMatchSchedule(steps: steps)
    }

    /// **The spectral schedule**: the first `spectralSteps` evaluations run at half-size.
    ///
    ///     x_{k+1} = x_k + Δσ · 𝓘 v_θ(𝓟 x_k, σ_k)
    ///
    /// In rectified flow, the SNR at frequency `f` reaches 1 at `f* = (1−σ)/σ`. On this schedule `f*`
    /// is [0; 0.056; 0.134; 0.25; 0.445; 0.834; 2.0]: **the first four evaluations
    /// carry less than 15 % of the last one's band**, and yet we compute at 1024² to
    /// produce what 512² would carry just as well.
    ///
    /// **The state never goes down** — only the evaluation does. A scheme that downsampled the state
    /// then upsampled it back would pay the interpolation cumulatively; here every step restarts from the
    /// full-resolution latent, and the resampling leaves no trace in what is
    /// kept.
    ///
    /// Measured on the spectral oracle (09-21): `k=0` reproduces the reference **to the
    /// bit** — without this reference nothing that follows could be attributed to the trick rather than to the
    /// harness; `k=3` removes **32.1 % of the tokens** and the image stays sharp; `k=4` covers it
    /// with speckles. **The verdict is in the image**: the three splits render 10.83 / 10.91 /
    /// 10.72 dB, a saturated PSNR that no longer separates anything.
    package var spectralSteps = 0

    /// **The size divisor, step by step** — the generalization of `spectralSteps`.
    ///
    /// `spectralSteps = k` says "the first `k` steps at half-size". It is a special case of
    /// `divisors = [2, 2, 1, 1, …]`, and the special case costs a step: **at the first step,
    /// `σ = 1` and `f*(σ) = (1 − σ)/σ` is ZERO** — no frequency reaches an SNR of 1 there, not
    /// even the lowest. Evaluating that step at half-size is still computing four times too much.
    ///
    /// The product's calibrated rule says: half-size as long as `f* ≤ 0.10`. A grid divided by
    /// `d` carries `d` times less band, so the same argument gives **`f* ≤ 0.10/d`** — and on the
    /// eight-step schedule, `f* = [0; 0.056; 0.134; …]` would give `[4, 2, 1, 1, 1, 1, 1]`. A quarter
    /// of a 1024² is 256², below the 512² floor (`Spectral.minimumSide`): `run` refuses it.
    ///
    /// ⚠️ **The quarter has no oracle**, and that must be said: the spectral oracle only measured
    /// factor-of-two descents. The ÷4 path **composes the verified operators twice**
    /// (𝓟 then 𝓟, 𝓘 then 𝓘) — that is not the same as a ÷4 in one pass, and it is a
    /// convention we choose rather than one we read off. Its verdict is therefore in the image, like
    /// that of `k`.
    package var divisors: [Int] = []

    /// **More detail** (`Detail`): the σ the DiT is told, lowered mid-trajectory; the Euler step
    /// keeps the true σ. `normal`: the schedule's own, to the bit.
    package var detail: Detail = .normal
    /// `Detail.scale` of the render's size.
    package var detailScale: Double = 1

    /// **After each evaluation** (zero steps are not evaluations): `(absolute step, seconds, evaluated
    /// grid h, w, latent after the update, derivative dx/dσ = negated)`. This is how the product's
    /// module numbers its steps, computes the preview and checks cancellation; it may
    /// throw, and the loop stops. `onStep`, for its part, remains the instrument of the checks.
    package var afterEvaluation: ((Int, Double, Int, Int, UnsafePointer<Float>, UnsafePointer<Float>) throws -> Void)?

    /// **Can the descent run these divisors on this grid?** Each a power of two, and the largest
    /// dividing both sides — then every level halves exactly. All ones: nothing to divide.
    package static func divides(_ divisors: [Int], height: Int, width: Int) -> Bool {
        let largest = divisors.max() ?? 1
        guard largest > 1 else { return true }
        return divisors.allSatisfy { $0 >= 1 && $0 & ($0 - 1) == 0 }
            && height > 0 && width > 0 && height % largest == 0 && width % largest == 0
    }

    /// The effective divisor of a step: the one that was set, otherwise the one `spectralSteps` implies.
    func divisor(_ step: Int) -> Int {
        if divisors.indices.contains(step) { return max(1, divisors[step]) }
        return step < spectralSteps ? 2 : 1
    }

    /// `latent` as `[C, H, W]`, modified in place. Returns the number of evaluations actually done.
    ///
    /// - Parameter latentHeight, latentWidth: `H` and `W`. Required as soon as `spectralSteps` is not
    ///   zero — without them, we cannot decimate.
    /// - Parameter start: the first step executed of the full calendar — img2img's `t_start`
    ///   (`Strength.startStep`), 0 in txt2img. The calendar does not change; only steps
    ///   `start … N − 1` run, under their absolute number (`modelTime(at:)` included).
    /// - Parameter onStep: `(step, σ, seconds, height and width of the evaluated latent)`.
    @discardableResult
    package func run(latent: UnsafeMutablePointer<Float>, count: Int,
                    caps: UnsafePointer<Float>, latentHeight: Int = 0, latentWidth: Int = 0,
                    start: Int = 0, onStep: ((Int, Float, Double, Int, Int) -> Void)? = nil) throws -> Int {
        let (height, width) = (latentHeight, latentWidth)
        let maxDivisor = (0..<(schedule.sigmas.count - 1)).map { divisor($0) }.max() ?? 1
        guard maxDivisor == 1 || (height > 0 && width > 0 && count % (height * width) == 0) else {
            throw Failure.missingSide
        }
        // **512 px per side is the floor, and it is imposed here, not recommended.** Below a latent
        // of 64, the model is out of its domain: what is evaluated there is neither an image nor a
        // measurement. Per side: the smaller of the two decides.
        guard maxDivisor == 1 || min(height, width) / maxDivisor >= Spectral.minimumSide else {
            throw Failure.belowFloor(side: min(height, width) / maxDivisor)
        }
        // **Every divisor a power of two, dividing both sides.** The descent is `𝓟` applied
        // `log₂ d` times, and its buffers are sized by halving `height` down to `height / maxDivisor`:
        // a divisor of 3 (or 2 on an odd side), possible under `SILICONED_DIVISORS`, would decimate
        // into a level that does not exist, or upsample past the end of `latent`.
        let all = (0..<(schedule.sigmas.count - 1)).map { divisor($0) }
        guard Self.divides(all, height: height, width: width) else {
            throw Failure.indivisible(height: height, width: width, divisor: maxDivisor)
        }
        let channels = height > 0 && width > 0 ? count / (height * width) : 0
        // **One buffer per descent level**, reserved once for all the low
        // evaluations. At 1024² they weigh 4 MB for the half and 1 MB for the quarter — we do not
        // count them, but reallocating them at every step would be seven times the work of the first.
        // **Stable allocations, not Swift arrays.** An `UnsafePointer(array)` taken
        // outside its `withUnsafeBufferPointer` is only valid for the duration of the call — and the descent
        // needs to hold a pointer from one level to the next. Two simultaneous exclusive accesses to
        // two elements of the same array would be one more fault.
        var levels: [UnsafeMutablePointer<Float>] = []
        var levelGrids: [(height: Int, width: Int)] = []
        var grid = (height: height, width: width)
        while grid.height > height / maxDivisor {
            grid = (grid.height / 2, grid.width / 2)
            levels.append(UnsafeMutablePointer<Float>.allocate(
                capacity: channels * grid.height * grid.width))
            levelGrids.append(grid)
        }
        defer { for level in levels { level.deallocate() } }

        // What the DiT is told: `1 − σ'` (`FlowMatchSchedule.modelTime`) with σ' the detail's σ.
        let told = detail.modelSigmas(schedule.sigmas, start: start, scale: detailScale)
        var evaluations = 0
        for step in start..<(schedule.sigmas.count - 1) {
            // Between two steps, not in the middle of one: see `Cancellation`.
            try dit.cancellation.check()
            let sigma = schedule.sigmas[step], next = schedule.sigmas[step + 1]
            var delta = next - sigma
            if delta == 0 {
                // The step moves nothing: we do not pay for the evaluation. That is M1, and it is 12.5 %.
                if recordTrajectory {
                    velocities.append(velocities.last ?? [Float](repeating: 0, count: count))
                    latents.append(Array(UnsafeBufferPointer(start: latent, count: count)))
                }
                onStep?(step, sigma, 0, height, width)
                continue
            }
            let started = Date()
            let d = divisor(step)
            let spectral = d > 1
            let (evaluatedHeight, evaluatedWidth) = (height / d, width / d)
            // `noise_pred = −v`, then Euler: `x ← x + Δσ · noise_pred`, hence `x − Δσ · v`.
            // The negation precedes the upsampling, as in the reference — 𝓘 is linear, so
            // the order has no effect on the bits, but reading it in the same order removes the
            // question.
            var negated: [Float]
            if spectral {
                // **The descent, one factor of two at a time.** `𝓟` only exists as a factor of two —
                // that is the convention read off `torch` and verified to the bit (the resampling
                // check). A ÷4 is therefore `𝓟 ∘ 𝓟`, and the upsampling `𝓘 ∘ 𝓘`: two correct
                // operators composed, and not a third one that would need verifying.
                let depth = Int(log2(Double(d)).rounded())
                var source = UnsafePointer<Float>(latent)
                for level in 0..<depth {
                    Resample.decimate(source, into: levels[level], channels: channels,
                                      height: levelGrids[level].height,
                                      width: levelGrids[level].width)
                    source = UnsafePointer(levels[level])
                }
                var speed = try dit.forward(latent: UnsafePointer(levels[depth - 1]),
                                              caps: caps, sigma: 1 - told[step],
                                              latentHeight: evaluatedHeight, latentWidth: evaluatedWidth)
                var minusOne: Float = -1
                speed.withUnsafeMutableBufferPointer {
                    vDSP_vsmul($0.baseAddress!, 1, &minusOne, $0.baseAddress!, 1,
                               vDSP_Length($0.count))
                }
                // The upsampling, symmetric: from the lowest level up to the full size, one factor
                // of two at a time. `speed` always carries the current level.
                for level in stride(from: depth - 1, through: 0, by: -1) {
                    let source = levelGrids[level]
                    var largest = [Float](repeating: 0,
                                            count: channels * (source.height * 2) * (source.width * 2))
                    speed.withUnsafeBufferPointer { down in
                        largest.withUnsafeMutableBufferPointer { up in
                            Resample.bilinearDouble(down.baseAddress!, into: up.baseAddress!,
                                                    channels: channels, height: source.height,
                                                    width: source.width)
                        }
                    }
                    speed = largest
                }
                negated = speed
                negated.withUnsafeMutableBufferPointer { buffer in
                    vDSP_vsma(buffer.baseAddress!, 1, &delta, latent, 1, latent, 1, vDSP_Length(count))
                }
            } else {
                negated = try dit.forward(latent: latent, caps: caps,
                                          sigma: 1 - told[step])
                negated.withUnsafeMutableBufferPointer { buffer in
                    var minusOne: Float = -1
                    vDSP_vsmul(buffer.baseAddress!, 1, &minusOne, buffer.baseAddress!, 1, vDSP_Length(count))
                    vDSP_vsma(buffer.baseAddress!, 1, &delta, latent, 1, latent, 1, vDSP_Length(count))
                }
            }
            evaluations += 1
            // **Where does the trajectory go wrong?** An entirely `NaN` image does not say at which step
            // it became so, and there are seven. The check costs 262,144 comparisons per
            // step — nothing next to an evaluation — and it turns "the render is broken"
            // into "the velocity of step 3 contains 412 non-finite values". It announces itself (a warning,
            // hence an event during a render) and does not stop: the rest of the trajectory says whether it propagates or
            // resolves, and the two do not have the same cause.
            if let bad = Sampler.countNotFinite(negated, count) {
                Warnings.emit("✗ step \(step): the VELOCITY carries \(bad.count) non-finite values "
                    + "(first at index \(bad.first), evaluation at \(evaluatedWidth)×\(evaluatedHeight))")
            }
            if let bad = Sampler.countNotFinite(UnsafeBufferPointer(start: latent, count: count), count) {
                Warnings.emit("✗ step \(step): the LATENT carries \(bad.count) non-finite values "
                    + "(first at index \(bad.first))")
            }
            if recordTrajectory {
                velocities.append(negated)
                latents.append(Array(UnsafeBufferPointer(start: latent, count: count)))
            }
            let seconds = Date().timeIntervalSince(started)
            if let afterEvaluation {
                try negated.withUnsafeBufferPointer {
                    try afterEvaluation(step, seconds, evaluatedHeight, evaluatedWidth, UnsafePointer(latent), $0.baseAddress!)
                }
            }
            onStep?(step, sigma, seconds, evaluatedHeight, evaluatedWidth)
        }
        return evaluations
    }

    /// The non-finite values of a buffer, or `nil` if there are none — the common case, and the one
    /// that must cost the least. A single pass, bailing out as soon as we know there are some.
    private static func countNotFinite(_ values: some Collection<Float>, _ count: Int)
            -> (count: Int, first: Int)? {
        var bad = 0, first = -1
        for (i, value) in values.enumerated() where !value.isFinite {
            bad += 1
            if first < 0 { first = i }
        }
        return bad > 0 ? (bad, first) : nil
    }

    package enum Failure: Error, CustomStringConvertible {
        case missingSide
        case belowFloor(side: Int)
        /// A divisor that is not a power of two, or a grid it does not divide (`SILICONED_DIVISORS`):
        /// the descent halves exactly, level by level, or reads past its buffers.
        case indivisible(height: Int, width: Int, divisor: Int)
        package var description: String {
            switch self {
            case let .indivisible(height, width, divisor):
                return "spectral divisor \(divisor) on a latent of \(width)×\(height): a power of two dividing both sides is needed"
            case .missingSide:
                return "the spectral schedule requires the latent grid: `run(..., latentHeight:, latentWidth:)`"
            case .belowFloor(let side):
                return "an evaluation would go down to \(side * 8) px on a side: the floor is "
                    + "\(Spectral.minimumSide * 8) px per side, the model is out of its domain below"
            }
        }
    }
}
