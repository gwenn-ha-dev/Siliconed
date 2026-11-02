// The two switches of the app's DiT box, as pure rules: « Faster Rendering » and « Accelerators ».
// They live here, not in the views, so that what a switch shows, what it greys out and what a render
// takes are one decision that the tests can read (`DiTSwitchesTests`).

/// Why a switch of the DiT box is greyed out; `nil` where it is available.
package enum DiTSwitchUnavailable: Equatable, Sendable {
    /// This Mac is not measured yet — none kept, or the one kept is of another Mac, macOS or build (an update): the first render
    /// measures it. Until then nobody knows whether an accelerator will get work, and « nothing to do » would be a guess.
    case notMeasured
    /// This Mac's measure chooses no accelerator — no row for the Neural Engine, no rounded operand on the GPU: the denoiser is fp32
    /// throughout, nothing to choose.
    case noAccelerator
    /// The size has a side under 1024 px: the half grid would fall under the 512-px floor.
    case sizeUnder1024
    /// A LoRA, Detail or an edit: the half-size start was never judged under them.
    case withLoRADetailOrEdit
    /// Not the recipe's step count: the half-size start is judged on the recipe's schedule only.
    case stepsOffRecipe
}

package enum DiTSwitches {
    /// **The Faster Rendering a render takes.** The Beta option is the master: off, no render is faster whatever
    /// the per-model choice says (the choices are kept, and come back with the option). On, the model's own
    /// choice, else its version's default (`Variant.fasterRenderingByDefault`, off everywhere since the option
    /// became a Beta). Whether the plan then applies it to this size is the engine's rule (`Spectral.reducedEvaluations`).
    package static func fasterRendering(beta: Bool, choice: Bool?, versionDefault: Bool) -> Bool {
        beta && (choice ?? versionDefault)
    }

    /// Why Faster Rendering would do nothing for the render being set up, or `nil` when it would apply: the
    /// same conditions as `Spectral.reducedEvaluations` (both sides at twice the 512-px floor, the recipe's steps, no
    /// LoRA / Detail / edit), asked in pixels for the box to grey the switch with its reason.
    package static func fasterRenderingUnavailable(width: Int, height: Int, withLoRA: Bool, edit: Bool,
                                                   atRecipeSteps: Bool) -> DiTSwitchUnavailable? {
        if withLoRA || edit { return .withLoRADetailOrEdit }
        if min(width, height) < 2 * Format.minimumSide { return .sizeUnder1024 }
        if !atRecipeSteps { return .stepsOffRecipe }
        return nil
    }

    /// The « Accelerators » switch: available where this Mac's map chooses one (`AcceleratorOffer`: the Neural Engine's rows, the GPU's
    /// tensor units on rounded operands), whatever the version.
    package static func acceleratorsUnavailable(measured: Bool, offered: Bool) -> DiTSwitchUnavailable? {
        guard measured else { return .notMeasured }
        return offered ? nil : .noAccelerator
    }

    /// The accelerators a render takes: the choice only where the map offers them.
    package static func accelerators(offered: Bool, choice: Bool?, versionDefault: Bool) -> Bool {
        offered && (choice ?? versionDefault)
    }
}
