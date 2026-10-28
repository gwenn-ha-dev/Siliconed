// The Anima modules (text, denoising), the Qwen-Image encoding and decoding modules Anima
// shares with Krea 2, and Krea 2's `HalfSizeStart`. Split from `Modules.swift`.

import Foundation

// ── Anima ───────────────────────────────────────────────────────────────────────────────────

/// Two tokenizers, Qwen3-0.6B (`last_hidden_state`), then Anima's conditioner (T5 + adapter).
package struct AnimaTextModule: TextModule {
    package var map: String
    package var tokenizerQwen: String
    package var tokenizerT5: String
    /// The published Turbo file, which carries the adapter's weights.
    package var adapter: String
    package var output: TextFormat { .anima }
    package var name: String { "Qwen3-0.6B + T5" }

    package init(map: String, tokenizerQwen: String, tokenizerT5: String, adapter: String) {
        self.map = map; self.tokenizerQwen = tokenizerQwen; self.tokenizerT5 = tokenizerT5
        self.adapter = adapter
    }

    package func encoder(_ prompt: String, context: Context) throws -> Conditioning {
        let begin = Date()
        // `padding="longest"` for a single prompt: no padding. Where the reference truncates (512,
        // T5 keeping its `</s>`) the engine reads on, up to `Request.maximumPromptTokens` per tokenizer,
        // and refuses past it. T5 first: its vocabulary is the finer one, it reaches the bound first on a
        // Latin-script prompt.
        let t5Ids = try T5Tokenizer(directory: tokenizerT5).encode(prompt, maxLength: .max)
        try Request.admitPrompt(tokens: t5Ids.count - 1, max: Request.maximumPromptTokens)
        let qwenIds = try Tokenizer(directory: tokenizerQwen).encode(prompt)
        try Request.admitPrompt(tokens: qwenIds.count, max: Request.maximumPromptTokens)
        // The engine already refuses a blank prompt; this guards a direct call of the module.
        guard !qwenIds.isEmpty else { throw Request.Failure.emptyPrompt }
        context.timings.tokenizer = Date().timeIntervalSince(begin)
        context.footprints.tokenizer = Arena.processFootprint()
        try context.check()
        let encoder = try TextEncoder(artifact: try Artifact(path: map), sequence: qwenIds.count,
                                       freezeCut: context.reproducible)
        encoder.cancellation = context.cancellation
        let hidden = try encoder.encode(ids: qwenIds, realTokens: qwenIds.count)
        context.footprints.encoder = Arena.processFootprint()
        // `hidden` points into the encoder's arena: the encoder must outlive the conditioner's read
        // of it, which ARC does not see through an unsafe pointer. So both arenas are alive at once —
        // what the memory need charges in the text phase (`MemoryNeed.anima`).
        let values = try withExtendedLifetime(encoder) {
            try AnimaConditioner(path: adapter).condition(t5: t5Ids, qwen: hidden, qwenRows: qwenIds.count)
        }
        return Conditioning(format: .anima, rows: values.count / 1024, width: 1024,
                               values: values, tokens: qwenIds.count)
    }
}

/// Anima Turbo's Cosmos DiT: Euler, CFG 1.
package struct AnimaDenoisingModule: DenoisingModule {
    package var map: String
    package var model: String { "anima" }
    package var name: String { "DiT Anima" }
    package var entry: TextFormat { .anima }
    package var space: LatentSpace { .qwenImage }
    package var defaultSteps: Int { 8 }

    package init(map: String) { self.map = map }

    /// Every `Solver`: `euler` walks them on Anima's own σ, judged against diffusers' schedulers.
    package var solvers: [Solver] { Solver.allCases }
    /// No Detail: the lie about σ makes mosaic blocks or nothing visible (`Detail`).
    package var acceptsDetail: Bool { false }

    package func sigmas(steps: Int) -> [Float] { AnimaDiT.sigmas(steps: steps) }

    /// **By default every evaluation at full size**, the reference's loop as it stands.
    ///
    /// **The half-size start** (`halfSizeStart`, off by default; Z-Image's recipe, `Sampler.init(half:…)`):
    /// the first `halfSizeSteps` evaluations on the half grid (`denoise`). Only at the recipe's eight,
    /// from the first step, on a grid that halves above the floor (8-px cells, and each half side a
    /// multiple of the ×2 patch: 1024², 1024×1536 and its transpose; a 512² or 832×1216 never halves),
    /// never under a user LoRA or another schedule (`withLoRA`: never judged under them; Anima refuses
    /// Detail). `SILICONED_SPECTRAL` forces `k` (a measurement), within the same rules.
    package func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan {
        let halvable = Spectral.divisible(height: height, width: width, by: 2)
        let rule = Spectral.halfSizeStart && steps == defaultSteps ? Self.halfSizeSteps : 0
        let k = start > 0 || !halvable || withLoRA ? 0 : min(max(0, Spectral.forced ?? rule), steps - 1)
        return DenoisingPlan(startStep: start, startSigma: sigmas(steps: steps)[start], evaluations: steps - start,
                             reduced: k, reducedHeight: k > 0 ? height / 2 : height,
                             reducedWidth: k > 0 ? width / 2 : width)
    }

    /// Its half-size evaluations (σ 1 → 0.833 at 3, the schedule's shift 3 as Z-Image's).
    package static let halfSizeSteps = 3

    /// **The denoising**: one DiT, every σ's modulations tabulated at once, Euler (or the `Solver`).
    ///
    /// Under the half-size start (`plan`), the first `k` evaluations by a DiT built for the half grid
    /// — Z-Image's recipe (`Sampler.init(half:…)`, where the why is written):
    ///
    ///     x_low = 𝓑x + σ·𝓑ε             (𝓑 the 2×2 mean, ε the starting noise)
    ///     x̂₀ = x_low − σ·v_θ(x_low, σ)   (the DiT returns dx/dσ itself, as Qwen's: no negation)
    ///     dx/dσ = (x − 𝓛x̂₀) / σ          (𝓛 Lanczos-3 ×2) — the state stays exactly σε + (1 − σ)S
    ///
    /// What a port by analogy would miss:
    ///   - **the latent is `[16, H, W]` at 8 px, unpacked**: the ×2 patch is the DiT's own
    ///     (`AnimaOps.patchify`), so 𝓑 and 𝓛 work on the true spatial grid, and commute with the VAE's
    ///     per-channel normalization (affine, weights summing to one);
    ///   - **the RoPE needs no rescaling**: `CosmosRotaryPosEmbed` slices `arange` positions 0 … n − 1
    ///     per axis with fixed NTK bases (`rope_scale`), whatever the grid — a half-size evaluation is
    ///     exactly a native 512² one, and the padding mask stays zero;
    ///   - **the DiT and its arena are sized for one grid** (`AnimaDiT.slices`): the half grid needs its
    ///     own DiT, its own modulation table (the first `k` σ; the full one tabulates the rest). One DiT
    ///     at a time: the half-size one is released before the full one is built, so the need
    ///     (`MemoryNeed.anima`, the full DiT) holds — the half-size one is smaller by far more than
    ///     the start's few latents (ε, the small grid, the upsampled estimate), released with it.
    package func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                          context: Context) throws {
        let (height, width) = (latent.height, latent.width)
        let k = plan(height: height, width: width, steps: steps, start: start,
                     withLoRA: lora != nil || context.detail != .normal).reduced
        // Every σ the DiT will be told (`euler` tells it `context.told`), its modulations computed at once.
        let schedule = sigmas(steps: steps)
        let told = context.told(schedule, start: start)
        let build = { (h: Int, w: Int, range: Range<Int>) throws -> AnimaDiT in
            let dit = try AnimaDiT(artifact: try Artifact(path: map), latentHeight: h, latentWidth: w,
                                   textLength: text.rows, freezeCut: context.reproducible,
                                   neuralEngine: context.neuralEngine, lora: lora)
            dit.cancellation = context.cancellation
            try dit.tabulate(sigmas: Array(told[range]))
            return dit
        }
        guard k > 0 else {
            let dit = try build(height, width, start..<(schedule.count - 1))
            try euler(&latent, sigmas: schedule, start: start, context: context) { x, sigma in
                try text.values.withUnsafeBufferPointer {
                    try dit.forward(latent: x, context: $0.baseAddress!, sigma: sigma)
                }
            }
            context.footprints.denoising = Arena.processFootprint()
            return
        }
        // One DiT at a time: the half-size one goes before the full one is built (`HalfSizeStart.handOver`).
        var small: AnimaDiT? = try build(height / 2, width / 2, 0..<k)
        var full: AnimaDiT?
        let half = HalfSizeStart(steps: k, speed: { x, sigma in
            try text.values.withUnsafeBufferPointer {
                try small!.forward(latent: x, context: $0.baseAddress!, sigma: sigma)
            }
        }, handOver: {
            small = nil
            full = try build(height, width, k..<(schedule.count - 1))
        })
        try euler(&latent, sigmas: schedule, start: start, context: context, half: half) { x, sigma in
            try text.values.withUnsafeBufferPointer {
                try full!.forward(latent: x, context: $0.baseAddress!, sigma: sigma)
            }
        }
        context.footprints.denoising = Arena.processFootprint()
    }
}

/// The Qwen-Image VAE's encoder (img2img), for Anima and Krea 2 — same weights, two files: the
/// image at the render's format → `(mean − mean_c) / std_c`.
package struct QwenImageEncodingModule: ImageEncodingModule {
    package var path: String
    package var space: LatentSpace { .qwenImage }
    package var name: String { "VAE Qwen-Image" }
    package init(path: String) { self.path = path }

    package func encoder(_ image: ImageRGB, context: Context) throws -> Latent {
        try encoderImage(image, path: path, family: .qwenImage, space: space, context: context)
    }
}

package struct QwenImageDecodingModule: DecodingModule {
    package var path: String
    package var space: LatentSpace { .qwenImage }
    package var name: String { "VAE Qwen-Image" }
    package init(path: String) { self.path = path }

    package func decode(_ latent: Latent, context: Context) throws -> ImageRGB {
        let vae = try AnimaVAE(path: path, latentHeight: latent.height, latentWidth: latent.width)
        let (image, seconds) = try latent.values.withUnsafeBufferPointer { try vae.decode(latent: $0.baseAddress!) }
        context.timings.decoding = seconds
        return ImageRGB(pixels: image, height: vae.imageHeight, width: vae.imageWidth)
    }
}

/// **Euler in flow matching, without negation**: `x += (σᵢ₊₁ − σᵢ) · v`, one evaluation per step.
/// The step of Anima and Krea 2; Z-Image has its own (`Sampler`), which negates the velocity.
/// `start`: the first step executed of the full schedule (img2img). `v` is
/// `dx/dσ` as is, hence x̂₀ = x − σ·v for the preview. Cancellation is checked before each step,
/// and between layers in the DiT.
///
/// **Another `Solver`** (`context.solver`, Anima's since 0.10, and Krea 2's) walks the
/// same σ through `SolverWalk` — `v` is already its `m = dx/dσ`, no negation — and the preview and a
/// sketch read the walk's own x̂₀ with σₓ = 0, as in `Sampler`. Euler keeps the loop below: its two
/// roundings (`dt` in float, then `x + dt·v`) are the reference's `scheduler.step`, to the bit.
///
/// **The half-size start** (`half`, Anima's and Krea 2's; from `start` 0 only): its first `half.steps`
/// evaluations by `half.speed` on the half grid, Z-Image's recipe (`Sampler.init(half:…)`, where the
/// why is written) for a DiT that returns `dx/dσ` itself:
///
///     x_low = 𝓑x + σ·𝓑ε             (𝓑 the 2×2 mean, ε the starting noise)
///     x̂₀ = x_low − σ·v_θ(x_low, σ)   (no negation)
///     dx/dσ = (x − 𝓛x̂₀) / σ          (𝓛 Lanczos-3 ×2) — the state stays exactly σε + (1 − σ)S
///
/// σ is the state's (`sigmas[i]`), never the told one, which only conditions the DiT. Its buffers —
/// ε, the small grid, the estimate, the upsampled estimate — are given back before the first full
/// evaluation, then `half.handOver` runs, off the step's clock: the half-size DiT goes, the full one
/// is built (one at a time).
func euler(_ latent: inout Latent, sigmas: [Float], start: Int = 0, context: Context,
           half: HalfSizeStart? = nil, speed: (UnsafePointer<Float>, Float) throws -> [Float]) throws {
    context.beginDenoising(evaluations: sigmas.count - 1 - start)
    // What the DiT is told (`Detail`: Anima and Krea 2 refuse it, so their σ themselves); the step
    // below keeps the true σ.
    let told = context.told(sigmas, start: start)
    let count = latent.values.count
    var stream = Noise(seed: context.solverSeed ^ 0x5EED_A9CE_57A1_0000)
    let walk = context.solver == .euler ? nil
        : SolverWalk(solver: context.solver, schedule: sigmas, start: start, count: count,
                     noise: context.solver == .eulerAncestral ? { _, e in stream.fill(e, count: count) } : nil)
    let halfSteps = start == 0 ? half?.steps ?? 0 : 0
    let (lowHeight, lowWidth) = (latent.height / 2, latent.width / 2)
    let lowCount = halfSteps > 0 ? latent.space.channels * lowHeight * lowWidth : 0
    var noise = halfSteps > 0 ? latent.values : []
    var small = [Float](repeating: 0, count: lowCount), estimate = small
    var upsampled = [Float](repeating: 0, count: halfSteps > 0 ? count : 0)
    for i in start..<(sigmas.count - 1) {
        try context.check()
        let reduced = i - start < halfSteps
        if !reduced, !noise.isEmpty {
            noise = []; small = []; estimate = []; upsampled = []
            try half?.handOver()
        }
        let tick = Date()
        // Under DPM++ and UniPC, diffusers' integer timestep over 1000 (`Solver.toldSigma`).
        let toldSigma = context.solver.toldSigma(told[i], first: i == start)
        let v: [Float]
        if reduced, let half {
            let sigma = sigmas[i]
            latent.values.withUnsafeBufferPointer { x in
                noise.withUnsafeBufferPointer { ε in
                    Resample.average(x.baseAddress!, plus: sigma, times: ε.baseAddress!, into: &small,
                                     channels: latent.space.channels, height: lowHeight, width: lowWidth)
                }
            }
            let low = try autoreleasepool {
                try small.withUnsafeBufferPointer { try half.speed($0.baseAddress!, toldSigma) }
            }
            for j in 0..<lowCount { estimate[j] = small[j] - sigma * low[j] }
            estimate.withUnsafeBufferPointer { e in
                Resample.lanczosDouble(e.baseAddress!, into: &upsampled, channels: latent.space.channels,
                                       height: lowHeight, width: lowWidth)
            }
            let inverse = 1 / sigma
            v = (0..<count).map { (latent.values[$0] - upsampled[$0]) * inverse }
        } else {
            // A pool per evaluation: what MPS autoreleases inside the DiT (graph runs' arrays, command
            // buffers) is freed every step, not at the render's end, whatever the family.
            v = try autoreleasepool {
                try latent.values.withUnsafeBufferPointer { try speed($0.baseAddress!, toldSigma) }
            }
        }
        if let walk {
            latent.values.withUnsafeMutableBufferPointer { x in
                v.withUnsafeBufferPointer { walk.step(latent: x.baseAddress!, derivative: $0.baseAddress!) }
            }
        } else {
            let dt = sigmas[i + 1] - sigmas[i]
            latent.values.withUnsafeMutableBufferPointer { x in
                for j in 0..<x.count { x[j] += dt * v[j] }
            }
        }
        let seconds = Date().timeIntervalSince(tick)
        try latent.values.withUnsafeBufferPointer { x in
            try v.withUnsafeBufferPointer { v in
                try context.stepDone(sigma: sigmas[i], seconds: seconds,
                                     evaluatedHeight: reduced ? lowHeight : latent.height,
                                     evaluatedWidth: reduced ? lowWidth : latent.width, space: latent.space,
                                     x: walk?.estimate ?? x.baseAddress!,
                                     v: v.baseAddress!, nextSigma: walk == nil ? sigmas[i + 1] : 0,
                                     fullHeight: latent.height, fullWidth: latent.width)
            }
        }
    }
}

/// `euler`'s half-size start: how many evaluations, and the speed on the half grid (`[C, h/2, w/2]`
/// in, `dx/dσ` on that grid out). The speed owns its DiT: the one `euler` calls next, at full size,
/// builds its own only once this one is given back (`Krea2DenoisingModule.denoise`).
struct HalfSizeStart {
    let steps: Int
    /// The half-size DiT's `dx/dσ` on the small grid, at the told σ.
    let speed: (UnsafePointer<Float>, Float) throws -> [Float]
    /// Before the first full step, off its clock: the half-size DiT goes, the full one is built.
    let handOver: () throws -> Void
}
