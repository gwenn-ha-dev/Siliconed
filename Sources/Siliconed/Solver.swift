import Accelerate
import Foundation

/// **The update rule of a step**: what the sampler does with the DiT's derivative once it has it.
/// The σ schedule is the model's own and does not change — only how a step walks it.
///
///     euler            x' = x + (σ' − σ)·m                                    (the distillation's own)
///     euler-ancestral  x' = (1 − σ')·x̂₀ + σ'·ε,         x̂₀ = x − σ·m       (fresh noise every step)
///     dpmpp-2m         λ = log(1 − σ) − log σ,  x' = (σ'/σ)·x − (1 − σ')(e^−h − 1)·(D₀ + ½D₁)
///     unipc            UniP-bh2 predictor + UniC corrector, order 2
///
/// `m = dx/dσ` is the engine's `negated` (`−v`), diffusers' `model_output`. Each rule is diffusers'
/// scheduler read line by line (`FlowMatchEulerDiscreteScheduler`, its `stochastic_sampling`,
/// `DPMSolverMultistepScheduler` and `UniPCMultistepScheduler`, both in `flow_prediction` with
/// `use_flow_sigmas`), checked against them on a synthetic model (`SolverTests`) — **to the bit** since
/// 18/10: every per-element operation is torch's, two roundings where torch writes `a·b + c` (it never
/// fuses on the CPU; `vDSP_vsma` did), the corrector's 2×2 system solved as LAPACK does (`solve`). The
/// scalar coefficients go through libm's `logf`/`expf`/`expm1f` where torch runs SLEEF's: one bit apart
/// on rare σ (Krea 2's DPM++ at σ₂), the one residue.
///
/// What a port by analogy would miss:
///
///   - **The walk is the schedule's tail as is**: every step of every schedule here moves (Z-Image's
///     too since 0.8.2 — before, it ended `…, 0, 0` and the walk had to drop that step, where a
///     multistep rule would take `log 0 − log 0`). The orders, the warm-up and the « last step »
///     are counted from the first step run.
///   - **σ = 1 makes `λ = −∞`.** DPM++ lives with it: `h₀ = +∞`, `r₀ = +∞`, `D₁ = 0`, so its first
///     second-order step is a first-order one — IEEE arithmetic, not a special case. UniPC does not
///     (its corrector solves a system with a `−∞` in it): diffusers moves σ = 1 to `1 − 10⁻⁶` in
///     `set_timesteps`. Only the solver's coefficients see it here (`solverSigmas`).
///   - **Under DPM++ and UniPC the DiT is told an integer timestep**: both schedulers cast `timesteps`
///     to **int64** (`set_timesteps`, `.to(torch.int64)`), so diffusers' DiT is told `⌊1000σ⌋/1000` —
///     `σ·1000` in float32, truncated, after UniPC's nudge: 999 for σ = 1 under UniPC, 1000 under
///     DPM++, 954 for 0.954545 — where Euler's (`FlowMatchEulerDiscreteScheduler`) keeps the float.
///     The engine tells it the same (`integerTimestep`, since 18/10); the update keeps the true σ.
///   - **UniPC corrects after the evaluation, not before**: the DiT sees the predictor's sample,
///     `x̂₀` is computed on that same sample, and only then is the sample corrected — the corrected
///     one feeds the next predictor, never the DiT.
///   - **Both multistep rules drop to first order at the last step** (`lower_order_final`,
///     `final_sigmas_type = "zero"`): σ' = 0 makes `λ' = +∞`, and the step lands on `x̂₀`, like Euler.
///   - **The ancestral rule renoises entirely** (diffusers' `stochastic_sampling`, not k-diffusion's
///     `eta`): the whole `σ'` part of the next sample is fresh noise.
package enum Solver: String, CaseIterable, Sendable, Codable {
    case euler
    case eulerAncestral = "euler-ancestral"
    case dpmpp2m = "dpmpp-2m"
    case unipc

    /// **The scheduler's timestep under this rule** — what the pipeline divides by 1000 before the DiT
    /// (see above): `σ·1000` in float32, truncated, UniPC's first σ = 1 nudged to `1 − 10⁻⁶` first.
    /// `nil` under Euler's rules, whose scheduler keeps the float.
    package func integerTimestep(_ sigma: Float, first: Bool) -> Int? {
        switch self {
        case .euler, .eulerAncestral: return nil
        case .dpmpp2m: return Int(sigma * 1000)
        case .unipc: return Int((first && abs(sigma - 1) < 1e-6 ? sigma - 1e-6 : sigma) * 1000)
        }
    }

    /// The σ to hand a DiT that computes its time as `(σ·1000)/1000` (Qwen-Image-2.1, Anima, Krea 2):
    /// `k/1000` for an integer timestep `k` — not always `k` again once times 1000 in float32 (10 of
    /// 0…1000 are not), but always `k/1000` again once divided — σ itself under Euler's rules.
    package func toldSigma(_ sigma: Float, first: Bool) -> Float {
        integerTimestep(sigma, first: first).map { Float($0) / 1000 } ?? sigma
    }

    /// How many latents a render under this rule keeps besides its own: the previous `x̂₀` for both
    /// multistep rules; for UniPC also the predictor's sample, its `D₁` (the corrector's) and the
    /// current `x̂₀`; one noise draw and its `x̂₀` for the ancestral one.
    package var history: Int {
        switch self {
        case .euler: return 0
        case .dpmpp2m: return 1
        case .eulerAncestral: return 2
        case .unipc: return 4
        }
    }

    /// The bytes of `history` for a latent of `count` floats — the sampler's term of `MemoryNeed`.
    package func historyBytes(count: Int) -> Int { history * Arena.capacity(for: count * 4) }
}

/// One render's walk under a `Solver`: the σ it walks and the latents it keeps between two steps.
package final class SolverWalk {
    package let solver: Solver
    /// The σ from the first step run, ending in 0.
    package let sigmas: [Float]
    let count: Int
    /// `ε` for the ancestral rule: `(step of the walk, buffer of count)`.
    let noise: ((Int, UnsafeMutablePointer<Float>) -> Void)?

    // UniPC's σ, σ = 1 moved off (see `Solver`). The other rules read `sigmas`.
    private let solverSigmas: [Float]
    private var pages: [Pages] = []
    /// The previous step's `x̂₀` (multistep), or the noise draw (ancestral).
    private var previous: UnsafeMutablePointer<Float>?
    /// The ancestral rule's `x̂₀`.
    private var landing: UnsafeMutablePointer<Float>?

    /// **The last step's estimate x̂₀ = x − σ·m**, on the sample the DiT saw — what a preview
    /// shows. `nil` under Euler, whose `x' − σ'·m` is the same estimate (`Sampler.afterEvaluation`).
    package var estimate: UnsafePointer<Float>? {
        switch solver {
        case .euler: return nil
        case .eulerAncestral: return landing.map { UnsafePointer($0) }
        case .dpmpp2m, .unipc: return previous.map { UnsafePointer($0) }
        }
    }
    /// UniPC: the sample the last predictor started from, its `D₁`, the current `x̂₀`.
    private var lastSample: UnsafeMutablePointer<Float>?
    private var lastD1: UnsafeMutablePointer<Float>?
    private var current: UnsafeMutablePointer<Float>?
    private var lastOrder = 0
    private var done = 0

    /// - Parameter schedule: the full schedule, `steps + 1` values (`FlowMatchSchedule.sigmas`).
    /// - Parameter start: the first step run (img2img's `t_start`).
    package init(solver: Solver, schedule: [Float], start: Int, count: Int,
                 noise: ((Int, UnsafeMutablePointer<Float>) -> Void)? = nil) {
        self.solver = solver
        self.count = count
        self.noise = noise
        sigmas = Array(schedule[start...])
        var nudged = sigmas
        if solver == .unipc, let first = nudged.first, abs(first - 1) < 1e-6 { nudged[0] = first - 1e-6 }
        solverSigmas = nudged
        if solver.history > 0 {
            pages = (0..<solver.history).map { _ in Pages(bytes: count * 4) }
            previous = pages[0].floats()
            if solver == .eulerAncestral { landing = pages[1].floats() }
            if solver == .unipc { (lastSample, lastD1, current) = (pages[1].floats(), pages[2].floats(), pages[3].floats()) }
        }
    }

    /// `λ = log α − log σ`, `α = 1 − σ`, in float like the reference's 0-d tensors.
    private static func lambda(_ sigma: Float) -> Float { logf(1 - sigma) - logf(sigma) }

    /// One step: `latent` (the sample the DiT just saw) becomes the next sample, from the
    /// derivative `m = dx/dσ` the DiT returned on it.
    package func step(latent x: UnsafeMutablePointer<Float>, derivative m: UnsafePointer<Float>) {
        let k = done
        precondition(k < sigmas.count - 1, "the walk has no step left")
        defer { done += 1 }
        let (s, t) = (sigmas[k], sigmas[k + 1])
        let n = count
        switch solver {
        case .euler:
            // `prev_sample = sample + dt * model_output`: two roundings, as torch computes it on the CPU.
            // Not `vDSP_vsma`, which fuses them (one rounding): 6.5 % of the values a bit apart per step.
            let delta = t - s
            for i in 0..<n { x[i] = x[i] + delta * m[i] }

        case .eulerAncestral:
            // `x0 = sample − σ·m; prev = (1 − σ')·x0 + σ'·noise`, the noise drawn every step, even the
            // last (where σ' = 0 throws it away), as `randn_tensor` is.
            let e = previous!, x0 = landing!
            noise?(k, e)
            for i in 0..<n {
                x0[i] = x[i] - s * m[i]
                x[i] = (1 - t) * x0[i] + t * e[i]
            }

        case .dpmpp2m:
            // `convert_model_output`: x̂₀ = x − σ·m. First and last steps first order.
            let prior = previous!
            let (alphaT, lambdaT, lambdaS) = (1 - t, Self.lambda(t), Self.lambda(s))
            let h = lambdaT - lambdaS
            let ratio = t / s, c = alphaT * (expf(-h) - 1)
            if k == 0 || k == sigmas.count - 2 {
                for i in 0..<n {
                    let d0 = x[i] - s * m[i]
                    x[i] = ratio * x[i] - c * d0
                    prior[i] = d0
                }
            } else {
                // `r₀ = h₀/h`, `D₁ = (1/r₀)(m₀ − m₁)`, midpoint: `− c·D₀ − ½c·D₁`.
                let h0 = lambdaS - Self.lambda(sigmas[k - 1])
                let inverse = 1 / (h0 / h), half = 0.5 * c
                for i in 0..<n {
                    let d0 = x[i] - s * m[i]
                    let d1 = inverse * (d0 - prior[i])
                    x[i] = ratio * x[i] - c * d0 - half * d1
                    prior[i] = d0
                }
            }

        case .unipc:
            stepUniPC(k, x, m)
        }
    }

    /// **`torch.linalg.solve([[1, 1], [rk, 1]], [b1, b2])` as torch computes it on the CPU**: LAPACK's
    /// LU with partial pivoting (the rows swap when `|rk| > 1`), then the two triangular solves — the
    /// multiplier a division, the back substitution by the pivot's reciprocal, no fused product. Read off
    /// 3 000 random systems against torch 2.14 (all equal); the closed form `(b1 − b2)/(1 − rk)` is
    /// another rounding, a few coefficients one bit apart. Returns `(ρ₁, ρ₂)`, the corrector's `(r, t)`.
    static func solve(rk: Float, _ b1: Float, _ b2: Float) -> (r: Float, t: Float) {
        let swapped = abs(rk) > 1
        let (pivot, below) = swapped ? (rk, Float(1)) : (Float(1), rk)
        let (top, bottom) = swapped ? (b2, b1) : (b1, b2)
        let l = below / pivot
        let u22 = 1 - l * 1
        let y2 = bottom - l * top
        let x2 = y2 * (1 / u22)
        let x1 = (top - 1 * x2) * (1 / pivot)
        return (x1, x2)
    }

    /// `UniPCMultistepScheduler.step`, `predict_x0`, `bh2`, `solver_order = 2`, `lower_order_final`.
    private func stepUniPC(_ k: Int, _ x: UnsafeMutablePointer<Float>, _ m: UnsafePointer<Float>) {
        let sig = solverSigmas, n = count
        let prior = previous!, last = lastSample!, older = lastD1!, x0 = current!
        // `x̂₀` on the sample the DiT saw, before any correction.
        for i in 0..<n { x0[i] = x[i] - sig[k] * m[i] }

        // ── UniC: correct the sample the last predictor produced, now that its x̂₀ is known ──
        if k > 0 {
            let (sT, sS0) = (sig[k], sig[k - 1])
            let h = Self.lambda(sT) - Self.lambda(sS0)
            let hh = -h, phi1 = expm1f(hh), bh = expm1f(hh)
            let b1 = (phi1 / hh - 1) / bh
            var rho: (r: Float, t: Float)
            if lastOrder == 1 {
                rho = (0, 0.5)
            } else {
                let rk = (Self.lambda(sig[k - 2]) - Self.lambda(sS0)) / h
                let b2 = ((phi1 / hh - 1) / hh - 0.5) * 2 / bh
                rho = Self.solve(rk: rk, b1, b2)
            }
            let (ratio, c) = (sT / sS0, (1 - sT) * phi1), cb = (1 - sT) * bh
            for i in 0..<n {
                // `m0` = the previous x̂₀, `D1_t = model_t − m0`; order 2 adds `(m₋₂ − m0)/rk` — the
                // previous predictor's own `D₁`: same σ, same `h`, same `rk`.
                var correction = rho.t * (x0[i] - prior[i])
                if lastOrder == 2 { correction += rho.r * older[i] }
                x[i] = ratio * last[i] - c * prior[i] - cb * correction
            }
        }

        // ── UniP: predict the next sample from the (corrected) one ─────────────────────────────
        let order = min(2, sig.count - 1 - k, done + 1)
        lastOrder = order
        for i in 0..<n { last[i] = x[i] }
        let (sT, sS0) = (sig[k + 1], sig[k])
        let h = Self.lambda(sT) - Self.lambda(sS0)
        let hh = -h, phi1 = expm1f(hh), bh = expm1f(hh)
        let (ratio, c, cb) = (sT / sS0, (1 - sT) * phi1, (1 - sT) * bh)
        if order == 1 {
            for i in 0..<n { x[i] = ratio * x[i] - c * x0[i] }
        } else {
            let rk = (Self.lambda(sig[k - 1]) - Self.lambda(sS0)) / h
            for i in 0..<n {
                let d1 = (prior[i] - x0[i]) / rk
                older[i] = d1
                x[i] = ratio * x[i] - c * x0[i] - cb * (0.5 * d1)
            }
        }
        for i in 0..<n { prior[i] = x0[i] }
    }
}
