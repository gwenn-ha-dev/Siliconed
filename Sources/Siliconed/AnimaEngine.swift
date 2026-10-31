import Foundation

// **Anima's σ schedule.** The Turbo's `FlowMatchEulerDiscreteScheduler` under `use_dynamic_shifting = false`, and nothing else of
// the model: the DiT itself is `Anima.swift`, its sampling loop the shared `Sampler.swift`. What sets it apart from the other
// families' schedules is that it is computed **in float32, operation by operation**, as the reference's `set_timesteps`
// does — the same formula in double, rounded once, is one bit off on half of the σ of N = 1…50 (see `sigmas`).

extension AnimaDiT {
    /// The Anima Turbo schedule: `sigmas = linspace(1, 1/N, N)` shifted by `shift = 3`
    /// (`σ' = 3σ / (1 + 2σ)`), then a zero terminal σ — `FlowMatchEulerDiscreteScheduler` under
    /// `use_dynamic_shifting = false`, as the modular `anima` pipeline calls it.
    ///
    /// **In float32, operation by operation**, as `set_timesteps` computes it: NumPy's double ramp
    /// cast to float32 first, then `shift * s / (1 + (shift − 1) * s)` on a float32 array (NumPy 2 keeps
    /// a Python float weak). The same formula in double, rounded once, is one bit off on 673 of the
    /// 1 275 σ of N = 1…50 (not at N = 8). To the bit against diffusers (`SolverTests`).
    package static func sigmas(steps: Int, shift: Double = 3) -> [Float] {
        let k = Float(shift), k1 = Float(shift - 1)
        return NumPyRamp.linspace(1, 1 / Double(steps), steps).map { s -> Float in
            let x = Float(s)
            return k * x / (1 + k1 * x)
        } + [0]
    }
}
