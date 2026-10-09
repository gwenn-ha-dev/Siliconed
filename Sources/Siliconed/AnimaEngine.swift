import Foundation

extension AnimaDiT {
    /// The Anima Turbo schedule: `sigmas = linspace(1, 1/N, N)` shifted by `shift = 3`
    /// (`σ' = 3σ / (1 + 2σ)`), then a zero terminal σ — `FlowMatchEulerDiscreteScheduler` under
    /// `use_dynamic_shifting = false`, as the modular `anima` pipeline calls it. Checked to
    /// 10⁻⁶ against the oracle's by the trajectory check.
    package static func sigmas(steps: Int, shift: Double = 3) -> [Float] {
        (0..<steps).map { i -> Float in
            // At a single step, the ramp has no slope (0/0): σ₀ = 1, like Krea 2 and Z-Image.
            let s = steps == 1 ? 1 : 1 - Double(i) * (1 - 1 / Double(steps)) / Double(steps - 1)
            return Float(shift * s / (1 + (shift - 1) * s))
        } + [0]
    }
}
