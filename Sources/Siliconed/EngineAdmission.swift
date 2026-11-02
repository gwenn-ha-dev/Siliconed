import Foundation

/// **What a request is, once the chain has judged it — before any computation, any memory and any file.**
/// Pure: a request, a chain and one fact about the plan (does it bring up the Neural Engine?) go in; the grid in latent cells, the
/// steps and the starting step, the denoiser's plan and the two switches the render reads (Neural Engine, « Faster Rendering ») come out,
/// each **resolved here once** — the render, its memory need and its `Context` all read this value, never the settings again.
///
///     refusals, in this order    empty prompt · steps outside 1…maximumSteps · img2img (encoder, strength) · references ·
///                                the denoiser's own `check` · solver · detail · a strength that leaves no evaluation
///     then                       the plan (`DenoisingModule.plan`): a LoRA or a Detail turns the half-size start off, and so do
///                                references (`DenoisingPlan.unreduced`: the plan cannot see them, the chain can)
///
/// A request the model cannot render is refused for what it is, not for its memory: the preflight (`Preflight`) comes after.
package struct Admission: Sendable {
    package let space: LatentSpace
    /// The latent grid (`request` pixels over the space's factor).
    package let height: Int, width: Int
    package let steps: Int
    /// The first step executed (0 in txt2img).
    package let start: Int
    package let plan: DenoisingPlan
    /// **What the denoiser may compute on beyond fp32**: the switch asked (`Request.accelerators`), the Neural Engine where the plan
    /// offers it (`usesANE`); off, fp32 throughout.
    package let accelerators: Accelerators
    /// The denoiser's products may take the Neural Engine: asked and offered by the plan.
    package var neuralEngine: Bool { accelerators.neuralEngine }
    /// « Faster Rendering » as the request asks it (`Request.fasterRendering`): the plan above already honours it.
    package let fasterRendering: Bool

    package static func judge(_ request: Request, chain: Chain, usesANE: Bool) throws -> Admission {
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Request.Failure.emptyPrompt
        }
        let denoising = chain.denoising, space = denoising.space
        let height = request.height / space.factor, width = request.width / space.factor
        let steps = request.steps ?? denoising.defaultSteps
        guard (1...Request.maximumSteps).contains(steps) else { throw Request.Failure.steps(steps) }
        // **img2img is judged before any computation, too**: an out-of-range strength, a chain
        // without an encoder, a strength that leaves no evaluation all throw here.
        var start = 0
        // The strength's N is the schedule's length as diffusers' `get_timesteps` reads it (`len(timesteps)`): the
        // steps asked, but fewer where a family's schedule merges σs (MOSS-Image's, snapped to its training grid).
        let scheduled = denoising.sigmas(steps: steps).count - 1
        if request.image != nil {
            guard chain.encoding != nil else { throw Strength.Failure.withoutEncoder(model: denoising.model) }
            guard request.strength > 0, request.strength <= 1 else { throw Strength.Failure.outOfBounds(request.strength) }
            start = Strength.startStep(steps: scheduled, strength: request.strength)
        }
        if !request.references.isEmpty {
            guard chain.encoding != nil, request.references.count <= denoising.maxReferences else {
                throw Request.Failure.references(request.references.count,
                                                 max: chain.encoding == nil ? 0 : denoising.maxReferences)
            }
        }
        try denoising.check(steps: steps, startImage: request.image != nil)
        guard denoising.solvers.contains(request.solver) else {
            throw Request.Failure.solver(request.solver, model: denoising.model)
        }
        guard request.detail == .normal || denoising.acceptsDetail else {
            throw Request.Failure.detail(model: denoising.model)
        }
        // Detail, like a LoRA, turns Qwen-Image-2.1's spectral schedule off; an edit too.
        let faster = request.fasterRendering
        var plan = denoising.plan(height: height, width: width, steps: steps, start: start,
                                  withLoRA: !request.loras.isEmpty || request.detail != .normal, faster: faster)
        if !request.references.isEmpty { plan = plan.unreduced }
        guard plan.evaluations > 0 else {
            // The floor: the last start that keeps an evaluation, `d`, is reached as soon as
            // `N − N·s < d + 1`, i.e. `s > (N − d − 1)/N`.
            let last = (0..<scheduled).last {
                denoising.plan(height: height, width: width, steps: steps, start: $0,
                               withLoRA: !request.loras.isEmpty, faster: faster).evaluations > 0
            } ?? 0
            throw Strength.Failure.noEvaluation(strength: request.strength, steps: steps,
                                                floor: Double(scheduled - last - 1) / Double(scheduled))
        }
        return Admission(space: space, height: height, width: width, steps: steps, start: start, plan: plan,
                         accelerators: Accelerators(requested: request.accelerators, usesANE: usesANE), fasterRendering: faster)
    }
}
