import Accelerate
import Foundation

/// The flow-matching schedule of Z-Image Turbo: **`N` steps are `N` evaluations**.
///
/// diffusers 0.41 (`get_default_z_image_sigmas`, PR #13730) and the official recipe — Tongyi's
/// model card asks the older pipeline for 9 steps, « this actually results in 8 DiT forwards »:
///
///     σᵢ = shift(linspace(1, 1/N, N))ᵢ,  shift(s) = 3s / (1 + 2s),  then a terminal 0
///     N = 8:  1, 0.954545, 0.9, 0.833333, 0.75, 0.642857, 0.5, 0.3, 0
///
/// What a port by analogy would miss:
///
///   - **The 0.36 pipeline was not this.** It forced `scheduler.sigma_min = 0` and walked
///     `linspace(1, 0, N)`: its last two σ were `0, 0`, a final step that moves nothing, and `N` steps
///     were `N − 1` evaluations. The engine carried that until 0.8.2 — 7 evaluations where the
///     model was distilled for 8. Every step now moves the latent; no step is skipped.
///   - **The ramp is torch's `linspace` in float32**, not NumPy's in double: `step = (end − 1)/(N − 1)`
///     in float, `1 + step·i` on the first half and `end − step·(N − 1 − i)` on the second, **each a
///     single rounding** (the CPU kernel fuses them), `end = Float(1/N)` with `1/N` a Python double.
///     Then the shift in float32 (`set_timesteps` casts the list to `np.float32` first).
///   - **Time is inverted**: the DiT receives `(1000 − t)/1000` with `t = σ·1000`, **in float32,
///     operation by operation** (`modelTime`) — not `1 − σ`, which is exact where the pipeline rounds
///     twice: at σ₁ = 0.954545 the two are 8 ulps apart, at σ₂ 3 and σ₅ 1 (until 18/10 the engine took
///     `1 − σ`).
///   - **The output is negated**: `noise_pred = −v`.
package struct FlowMatchSchedule {
    package let sigmas: [Float]          // `steps + 1` values, strictly decreasing, ending in zero

    package init(steps: Int, shift: Float = 3.0) {
        let end = Float(1.0 / Double(steps))
        let step = steps > 1 ? (end - 1) / Float(steps - 1) : 0
        var values: [Float] = (0..<steps).map { i in
            let raw = i < steps / 2 ? fmaf(step, Float(i), 1) : fmaf(-step, Float(steps - 1 - i), end)
            return shift * raw / (1 + (shift - 1) * raw)
        }
        values.append(0)
        sigmas = values
    }

    /// **What the DiT receives: the inverted time.** `timestep = (1000 − t) / 1000`, `t = σ · 1000` (`scheduler.timesteps`), each operation a float32
    /// rounding as torch does them — `ZImagePipeline.__call__`, and its img2img's.
    package static func modelTime(sigma: Float) -> Float {
        let t = sigma * 1000
        return (1000 - t) / 1000
    }
}

package final class Sampler {
    /// The full-size DiT, or `nil` until the half-size start is over (`init(half:…)`).
    private var dit: DiT?
    private var buildFull: (() throws -> DiT)?
    /// The half-size start's DiT and its evaluation count, released when they are done.
    private var half: DiT?
    private let halfSteps: Int
    package let schedule: FlowMatchSchedule

    /// What each step produced, to compare against the golden trajectory.
    package private(set) var velocities: [[Float]] = []
    package private(set) var latents: [[Float]] = []
    package var recordTrajectory = false

    package init(dit: DiT, steps: Int) {
        self.dit = dit
        self.halfSteps = 0
        self.schedule = FlowMatchSchedule(steps: steps)
    }

    /// **The half-size start**: the first `halfSteps` evaluations by `half`, a DiT sized for the half
    /// grid, then `full` builds the full-size one — one DiT at a time, never both.
    ///
    ///     x_low = 𝓑x + σ·𝓑ε            (𝓑 the 2×2 mean, ε the starting noise)
    ///     x̂₀ = x_low − σ·v_θ(x_low, σ)  (the small grid's estimate)
    ///     dx/dσ = (x − 𝓛x̂₀) / σ         (𝓛 Lanczos-3 ×2)
    ///
    /// What a port of the old half-size steps (`x ← x + Δσ·𝓘v(𝓟x)`, removed in 0.9) would miss, and
    /// what made them leave grain on the eight evaluations of the recipe:
    ///
    ///   - **The upsampled velocity carries no high band**: the latent's high-frequency noise was
    ///     never removed, 100 % of it left at σ = 0.833 where 83 % belongs, which the full steps read as
    ///     texture (sand on the skin). Upsampling x̂₀ instead, the noise part of the derivative is the
    ///     full latent's own: the state stays exactly `σε + (1 − σ)S`, whatever the interpolator.
    ///   - **Decimation hands the model an alias**: `𝓟ε` is white, but its low band is the full
    ///     latent's high band folded; the model's x̂₀ explains a noise the full grid does not carry
    ///     at those frequencies, and the full steps reconcile the two with a crackle. Because the state is
    ///     exactly `σε + (1 − σ)S`, `𝓑x + σ𝓑ε = 2σ·𝓑ε + (1 − σ)·𝓑S`: noise of variance σ² that *is* the
    ///     latent's low band, signal antialiased.
    ///   - **Lanczos, not bilinear, on x̂₀**: bilinear passes the spectral images of the small grid's
    ///     upper band into the full one.
    ///
    /// Only from the first step (`start` 0): ε is the starting latent.
    package init(half: DiT, halfSteps: Int, steps: Int, full: @escaping () throws -> DiT) {
        self.half = half
        self.halfSteps = halfSteps
        self.buildFull = full
        self.schedule = FlowMatchSchedule(steps: steps)
    }

    /// **More detail** (`Detail`): the σ the DiT is told, lowered mid-trajectory; the Euler step
    /// keeps the true σ. `normal`: the schedule's own, to the bit.
    package var detail: Detail = .normal
    /// `Detail.scale` of the render's size.
    package var detailScale: Double = 1

    /// **The update rule** (`Solver`): Euler, the distillation's own, unless the user picked another.
    /// The schedule, the steps and what the DiT is told do not change.
    package var solver: Solver = .euler
    /// The ancestral rule's noise: one stream from this seed, step after step — the render's seed,
    /// so the same seed gives the same image.
    package var solverSeed: UInt64 = 0

    /// **After each evaluation** (every step is one): `(absolute step, seconds, evaluated
    /// grid h, w, x, derivative dx/dσ = negated, σₓ)`, the estimate x̂₀ being `x − σₓ·negated`. This
    /// is how the product's module numbers its steps, computes the preview and checks cancellation; it
    /// may throw, and the loop stops. `onStep`, for its part, remains the instrument of the checks.
    ///
    /// Under Euler, `x` is the latent after the update and `σₓ` the next σ — as it always was, so a
    /// render keeps its bits. Under another solver the latent after the update is not on the line
    /// `x̂₀ + σ'·m` (the ancestral rule has fresh noise in it, the multistep rules a correction), so
    /// `x` is the walk's own x̂₀ (`SolverWalk.estimate`) and `σₓ` is 0.
    package var afterEvaluation: ((Int, Double, Int, Int, UnsafePointer<Float>, UnsafePointer<Float>, Float) throws -> Void)?

    /// `latent` as `[C, H, W]`, modified in place. Returns the number of evaluations actually done.
    ///
    /// - Parameter latentHeight, latentWidth: `H` and `W`, the grid every step evaluates — handed to
    ///   `afterEvaluation` (the preview reads it); 0 when nobody reads it.
    /// - Parameter start: the first step executed of the full schedule — img2img's `t_start`
    ///   (`Strength.startStep`), 0 in txt2img. The schedule does not change; only steps
    ///   `start … N − 1` run, under their absolute number (`modelTime(at:)` included).
    /// - Parameter onStep: `(step, σ, seconds)`.
    @discardableResult
    package func run(latent: UnsafeMutablePointer<Float>, count: Int,
                    caps: UnsafePointer<Float>, latentHeight: Int = 0, latentWidth: Int = 0,
                    start: Int = 0, onStep: ((Int, Float, Double) -> Void)? = nil) throws -> Int {
        let (height, width) = (latentHeight, latentWidth)
        // What the DiT is told: `(1000 − 1000σ')/1000` (`FlowMatchSchedule.modelTime`) with σ' the detail's σ.
        let told = detail.modelSigmas(schedule.sigmas, start: start, bell: .zImage, scale: detailScale)
        // The update, step by step, on the true σ (`Solver`). Its history lives in `Pages`.
        var stream = Noise(seed: solverSeed ^ 0x5EED_A9CE_57A1_0000)
        let walk = SolverWalk(solver: solver, schedule: schedule.sigmas, start: start, count: count,
                              noise: solver == .eulerAncestral ? { _, e in stream.fill(e, count: count) } : nil)
        var evaluations = 0
        // The half-size start's buffers: ε, the small grid, its estimate, the upsampled estimate.
        let channels = half != nil ? count / (height * width) : 0
        let (lowHeight, lowWidth) = (height / 2, width / 2)
        let lowCount = channels * lowHeight * lowWidth
        let noise = half != nil ? Array(UnsafeBufferPointer(start: latent, count: count)) : []
        var small = [Float](repeating: 0, count: lowCount), estimate = small
        var upsampled = [Float](repeating: 0, count: half != nil ? count : 0)
        for step in start..<(schedule.sigmas.count - 1) {
            // Between two steps, not in the middle of one: see `Cancellation`.
            let reduced = half != nil && step - start < halfSteps
            if !reduced, dit == nil {
                half = nil
                dit = try buildFull?()
            }
            guard let current = reduced ? half : dit else { break }
            try current.cancellation.check()
            let sigma = schedule.sigmas[step], next = schedule.sigmas[step + 1]
            let started = Date()
            // `noise_pred = −v`, then the solver — Euler: `x ← x + Δσ · noise_pred`, hence `x − Δσ · v`.
            // One autorelease pool per evaluation: what MPS autoreleases (descriptors, NDArrays and
            // their buffers) dies with the evaluation, not with the render's queue item.
            // Under DPM++ and UniPC, `(1000 − k)/1000` with diffusers' integer timestep (`Solver.integerTimestep`).
            let time = solver.integerTimestep(told[step], first: step == start).map { Float(1000 - $0) / 1000 }
                ?? FlowMatchSchedule.modelTime(sigma: told[step])
            var negated: [Float]
            if reduced {
                noise.withUnsafeBufferPointer { ε in
                    Resample.average(latent, plus: sigma, times: ε.baseAddress!, into: &small,
                                     channels: channels, height: lowHeight, width: lowWidth)
                }
                let low = try small.withUnsafeBufferPointer { s in
                    try autoreleasepool { try current.forward(latent: s.baseAddress!, caps: caps, sigma: time) }
                }
                // `low` is the model's output, `−dx/dσ`: x̂₀ = x_low + σ·low.
                for i in 0..<lowCount { estimate[i] = small[i] - sigma * -low[i] }
                estimate.withUnsafeBufferPointer { e in
                    Resample.lanczosDouble(e.baseAddress!, into: &upsampled, channels: channels,
                                           height: lowHeight, width: lowWidth)
                }
                let inverse = 1 / sigma
                negated = (0..<count).map { (latent[$0] - upsampled[$0]) * inverse }
            } else {
                negated = try autoreleasepool { try current.forward(latent: latent, caps: caps, sigma: time) }
                negated.withUnsafeMutableBufferPointer { buffer in
                    var minusOne: Float = -1
                    vDSP_vsmul(buffer.baseAddress!, 1, &minusOne, buffer.baseAddress!, 1, vDSP_Length(count))
                }
            }
            negated.withUnsafeBufferPointer { walk.step(latent: latent, derivative: $0.baseAddress!) }
            evaluations += 1
            // **Where does the trajectory go wrong?** An entirely `NaN` image does not say at which step
            // it became so, and there are eight. The check costs 262,144 comparisons per
            // step — nothing next to an evaluation — and it turns "the render is broken"
            // into "the velocity of step 3 contains 412 non-finite values". It announces itself (a
            // warning, hence an event during a render) and does not stop: the rest of the trajectory
            // says whether it propagates or resolves, and the two do not have the same cause.
            if let bad = Sampler.countNotFinite(negated, count) {
                Warnings.emit("✗ step \(step): the VELOCITY carries \(bad.count) non-finite values "
                    + "(first at index \(bad.first), evaluation at \(width)×\(height))")
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
                    if let estimate = walk.estimate {
                        try afterEvaluation(step, seconds, height, width, estimate, $0.baseAddress!, 0)
                    } else {
                        try afterEvaluation(step, seconds, height, width, UnsafePointer(latent),
                                            $0.baseAddress!, next)
                    }
                }
            }
            onStep?(step, sigma, seconds)
        }
        return evaluations
    }

    /// The non-finite values of a buffer (how many, and the first one's index), or `nil` if there are
    /// none — the common case. A single pass.
    static func countNotFinite(_ values: some Collection<Float>, _ count: Int)
            -> (count: Int, first: Int)? {
        var bad = 0, first = -1
        for (i, value) in values.enumerated() where !value.isFinite {
            bad += 1
            if first < 0 { first = i }
        }
        return bad > 0 ? (bad, first) : nil
    }
}

/// **NumPy's `linspace(start, stop, num)`** in double, as `function_base.linspace` computes it:
/// `step = (stop − start)/(num − 1)`, `i·step + start`, the last value set to `stop`. Anima's and
/// Krea 2's pipelines hand it to the scheduler, which casts it to float32.
package enum NumPyRamp {
    package static func linspace(_ start: Double, _ stop: Double, _ count: Int) -> [Double] {
        guard count > 1 else { return count == 1 ? [start] : [] }
        let step = (stop - start) / Double(count - 1)
        var values = (0..<count).map { Double($0) * step + start }
        values[count - 1] = stop
        return values
    }
}
