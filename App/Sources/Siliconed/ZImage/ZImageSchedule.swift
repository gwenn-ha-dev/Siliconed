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
///     twice: at σ₁ = 0.954545 the two are 8 ulps apart, at σ₂ 3 and σ₅ 1.
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
