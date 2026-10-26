import Foundation

// **The repository's modules, and the models that assemble them.**
//
// (each name below is a `…Module`)
//                 text               image (img2img)            denoising               decoding
//   Z-Image    ZImageText ─────────── FluxEncoding ──────────── ZImageDenoising ──────── FluxDecoding
//   Anima      AnimaText ──────────── QwenImageEncoding ─┬───── AnimaDenoising ─┬─────── QwenImageDecoding
//   Krea 2     Krea2Text ──────────── QwenImageEncoding ─┘      Krea2Denoising ─┘
//   Qwen 2.1   QwenImage21Text ◀─ refs ── QwenImage21Encoding ── QwenImage21Denoising ── QwenImage21Decoding
//              (the encoder sees the references: one Lanczos copy feeds it and the VAE)
//
// Each module wraps a stage's classes without changing a single computation. Their checks
// (`forward`, `qwen21-*`, `anima-*`, `krea2-*`, …) verify those classes; the close-out verifies the whole chain.

// ── Z-Image ─────────────────────────────────────────────────────────────────────────────────

package struct ZImageTextModule: TextModule {
    package var map: String
    /// The published repository's `tokenizer/` folder.
    package var tokenizer: String
    package var output: TextFormat { .zImage }
    package var name: String { "Qwen3-4B" }

    package init(map: String, tokenizer: String) { self.map = map; self.tokenizer = tokenizer }

    package func encoder(_ prompt: String, context: Context) throws -> Conditioning {
        let begin = Date()
        var identifiers: [Int] = [], realCount = 0
        // **Two scopes, not one**: a 151 k-entry dictionary and a Metal arena do not leave the same
        // residue, and measuring them together makes it impossible to know which to fix.
        do {
            identifiers = try Tokenizer(directory: tokenizer).admitChat(user: prompt)
            realCount = identifiers.count
        }
        context.timings.tokenizer = Date().timeIntervalSince(begin)
        context.footprints.tokenizer = Arena.processFootprint()
        try context.check()
        let encoder = try TextEncoder(artifact: try Artifact(path: map), sequence: realCount,
                                       freezeCut: context.reproducible)
        encoder.cancellation = context.cancellation
        let caps = Array(try encoder.encode(ids: identifiers, realTokens: realCount))
        context.footprints.encoder = Arena.processFootprint()
        return Conditioning(format: .zImage, rows: realCount, width: caps.count / max(1, realCount),
                               values: caps, tokens: realCount)
    }
}

/// Z-Image's S3 DiT under its sampler (negation; every evaluation at full size unless the half-size
/// start is on).
package struct ZImageDenoisingModule: DenoisingModule {
    package var map: String

    package var model: String { "z-image" }
    package var name: String { "DiT Z-Image" }
    package var entry: TextFormat { .zImage }
    package var space: LatentSpace { .flux }
    package var defaultSteps: Int { 8 }
    /// Every `Solver`: its sampler walks them all (`SolverWalk`).
    package var solvers: [Solver] { Solver.allCases }

    package init(map: String) { self.map = map }

    package func sigmas(steps: Int) -> [Float] { FlowMatchSchedule(steps: steps).sigmas }

    /// **By default every evaluation at full size** (since 0.8.2). The former half-size first steps
    /// were judged clean on the seven-evaluation schedule of before 0.8.2; on the recipe's eight
    /// (first full step at σ = 0.9, six full steps after), the high-frequency noise the half-size steps
    /// leave in the full latent (the up-sampled velocity does not carry it) is no longer resorbed —
    /// sand on the skin and grain on the walls at 1024², at `k = 2` and at `k = 1`. Qwen-Image-2.1's
    /// half-size start is this one's recipe (`QwenImage21DenoisingModule.plan`), its `k` judged on its own.
    ///
    /// Evaluations are counted from the start to the end: every step moves (`FlowMatchSchedule`),
    /// hence `N − t_start` evaluations, `⌈N·s⌉` at a strength `s` in img2img.
    ///
    /// **The half-size start** (`Spectral.halfSizeStart`, the app's « Faster Rendering »): the first
    /// three evaluations (σ 1 → 0.833) on the half grid, by `Sampler.init(half:…)` — not the steps
    /// removed in 0.8.2, whose grain it was written against. Only from the first step, at the
    /// recipe's eight, on a grid that halves above the floor (1024², 1024×1536 and its transpose),
    /// without a LoRA or Detail (never judged under either). `SILICONED_SPECTRAL` forces the count (a
    /// measurement), within the same rules.
    package func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan {
        let σ = FlowMatchSchedule(steps: steps).sigmas
        let halvable = Spectral.divisible(height: height, width: width, by: 2)
        let rule = Spectral.halfSizeStart && steps == defaultSteps ? Self.halfSizeSteps : 0
        let k = start > 0 || !halvable || withLoRA ? 0 : min(max(0, Spectral.forced ?? rule), steps - 1)
        return DenoisingPlan(startStep: start, startSigma: σ[start], evaluations: steps - start,
                             reduced: k, reducedHeight: k > 0 ? height / 2 : height,
                             reducedWidth: k > 0 ? width / 2 : width)
    }

    /// Its half-size evaluations: 3, judged clean on two prompts; at 4, fine dots came back on skin.
    package static let halfSizeSteps = 3

    package func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                          context: Context) throws {
        let (height, width) = (latent.height, latent.width)
        let planned = plan(height: height, width: width, steps: steps, start: start,
                           withLoRA: lora != nil || context.detail != .normal)
        let full = { () throws -> DiT in
            if let kept = context.keptDenoiser as? DiT { return kept }
            let dit = try DiT(artifact: try Artifact(path: map), latentHeight: height, latentWidth: width,
                              capLength: text.rows, freezeCut: context.reproducible,
                              neuralEngine: context.neuralEngine, lora: lora)
            dit.cancellation = context.cancellation
            context.keptDenoiser = dit
            return dit
        }
        let sampler: Sampler
        if planned.reduced > 0 {
            // One DiT at a time: a batch's kept full DiT goes before the half-size one is built.
            context.keptDenoiser = nil
            let half = try DiT(artifact: try Artifact(path: map), latentHeight: planned.reducedHeight,
                               latentWidth: planned.reducedWidth, capLength: text.rows,
                               freezeCut: context.reproducible, neuralEngine: context.neuralEngine, lora: lora)
            half.cancellation = context.cancellation
            sampler = Sampler(half: half, halfSteps: planned.reduced, steps: steps, full: full)
        } else {
            let dit = try full()
            dit.cancellation = context.cancellation
            sampler = Sampler(dit: dit, steps: steps)
        }
        sampler.detail = context.detail
        sampler.detailScale = context.detailScale
        sampler.solver = context.solver
        sampler.solverSeed = context.solverSeed
        context.beginDenoising(evaluations: planned.evaluations)
        // `negated` is `dx/dσ` (the DiT's output negated, see `Sampler`): x̂₀ = x − σ·negated.
        let σ = sampler.schedule.sigmas
        sampler.afterEvaluation = { step, seconds, h, w, x, derived, sigmaX in
            try context.stepDone(sigma: σ[step], seconds: seconds, evaluatedHeight: h, evaluatedWidth: w,
                                 space: .flux, x: x, v: derived, nextSigma: sigmaX,
                                 fullHeight: height, fullWidth: width)
        }
        try latent.values.withUnsafeMutableBufferPointer { l in
            try text.values.withUnsafeBufferPointer { caps in
                _ = try sampler.run(latent: l.baseAddress!, count: l.count,
                                            caps: caps.baseAddress!, latentHeight: height,
                                            latentWidth: width, start: start)
            }
        }
        context.footprints.denoising = Arena.processFootprint()   // before the release
    }
}

package struct FluxDecodingModule: DecodingModule {
    package var path: String
    package var space: LatentSpace { .flux }
    package var name: String { "VAE Flux" }
    package init(path: String) { self.path = path }

    package func decode(_ latent: Latent, context: Context) throws -> ImageRGB {
        let vae = try VAE(path: path, latentHeight: latent.height, latentWidth: latent.width)
        let (image, seconds) = try latent.values.withUnsafeBufferPointer { try vae.decode(latent: $0.baseAddress!) }
        context.timings.decoding = seconds
        return ImageRGB(pixels: image, height: vae.imageHeight, width: vae.imageWidth)
    }
}

/// The Flux VAE's encoder (img2img): the image at the render's format → `(mean − 0.1159) × 0.3611`.
package struct FluxEncodingModule: ImageEncodingModule {
    package var path: String
    package var space: LatentSpace { .flux }
    package var name: String { "VAE Flux" }
    package init(path: String) { self.path = path }

    package func encoder(_ image: ImageRGB, context: Context) throws -> Latent {
        try encoderImage(image, path: path, family: .flux, space: space, context: context)
    }
}

/// **A VAE encoder in its own scope**: it is built, encodes, notes its footprint before returning
/// its memory (a peak is read before the release), and dies on exit.
func encoderImage(_ image: ImageRGB, path: String, family: VAEEncoder.Family, space: LatentSpace,
                  context: Context) throws -> Latent {
    let encoder = try VAEEncoder(path: path, family: family, height: image.height, width: image.width)
    let output = try image.pixels.withUnsafeBufferPointer { try encoder.encode(image: $0.baseAddress!) }
    context.footprints.encoding = Arena.processFootprint()
    return Latent(space: space, height: encoder.latentGridHeight, width: encoder.latentGridWidth,
                  values: output.latent)
}

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
                     withLoRA: lora != nil || context.detail != .normal, schedule: context.schedule).reduced
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

// ── Krea 2 ──────────────────────────────────────────────────────────────────────────────────

/// Template, Qwen tokenizer, Qwen3-VL-4B with its twelve layers taken. Not the fusion: that is the DiT's.
package struct Krea2TextModule: TextModule {
    package var map: String
    package var tokenizer: String
    /// The layers taken, read from `model_index.json`.
    package var sockets: [Int]
    package var output: TextFormat { .krea2 }
    package var name: String { "Qwen3-VL-4B" }

    package init(map: String, tokenizer: String, sockets: [Int]) {
        self.map = map; self.tokenizer = tokenizer; self.sockets = sockets
    }

    package func encoder(_ prompt: String, context: Context) throws -> Conditioning {
        let begin = Date()
        let ids = try Krea2Text.admitted(prompt, tokenizer: try Tokenizer(directory: tokenizer))
        let tokens = ids.count - Krea2Text.templateTokens
        context.timings.tokenizer = Date().timeIntervalSince(begin)
        context.footprints.tokenizer = Arena.processFootprint()
        try context.check()
        let hiddenValues = try Krea2Text.hiddenStates(ids, encoder: map, sockets: sockets,
                                                freezeCut: context.reproducible,
                                                cancellation: context.cancellation)
        context.footprints.encoder = Arena.processFootprint()
        return Conditioning(format: .krea2, rows: tokens, width: hiddenValues.count / max(1, tokens),
                               values: hiddenValues, tokens: tokens)
    }
}

/// The text fusion and `txt_in`, then the Krea 2 DiT: Euler, CFG 0. Both read the same map and the
/// LoRA touches both — which is why the fusion lives here and not in the text stage.
package struct Krea2DenoisingModule: DenoisingModule {
    package var map: String
    package var model: String { "krea2" }
    package var name: String { "DiT Krea 2" }
    package var entry: TextFormat { .krea2 }
    package var space: LatentSpace { .qwenImage }
    package var defaultSteps: Int { 8 }

    package init(map: String) { self.map = map }

    /// Every `Solver`: `euler` walks them on Krea 2's own σ (the exponential shift), judged against
    /// diffusers' schedulers (`SolverTests`).
    package var solvers: [Solver] { Solver.allCases }
    /// No Detail: the node's bell changes the pose and the face, not the texture (`Detail`).
    package var acceptsDetail: Bool { false }

    package func sigmas(steps: Int) -> [Float] { Krea2DiT.sigmas(steps: steps) }

    /// **By default every evaluation at full size**: the published recipe as it stands.
    ///
    /// **The half-size start** (`halfSizeStart`, off by default; Z-Image's recipe, by `euler`): the
    /// first `halfSizeSteps` evaluations on the half grid. Only at the recipe's eight, from the first
    /// step, on a grid that halves above the floor (`Spectral.floor` in 8-px cells: 1024², 1024×1536
    /// and its transpose; a 512² or 832×1216 never halves), never under a user LoRA or another
    /// schedule (`withLoRA`: never judged under them). `SILICONED_SPECTRAL` forces `k` (a
    /// measurement), within the same rules.
    package func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan {
        let halvable = Spectral.divisible(height: height, width: width, by: 2,
                                          floor: Spectral.floor(factor: space.factor))
        let rule = Spectral.halfSizeStart && steps == defaultSteps ? Self.halfSizeSteps : 0
        let k = start > 0 || !halvable || withLoRA ? 0 : min(max(0, Spectral.forced ?? rule), steps - 1)
        return DenoisingPlan(startStep: start, startSigma: sigmas(steps: steps)[start], evaluations: steps - start,
                             reduced: k, reducedHeight: k > 0 ? height / 2 : height,
                             reducedWidth: k > 0 ? width / 2 : width)
    }

    /// Its half-size evaluations: 4 (σ 1 → 0.760), judged by eye on two prompts at 1024² — one more than
    /// Z-Image's and Qwen-Image-2.1's, Krea 2's shifted schedule lingering longer at high σ. Not the
    /// largest that renders: at 5 the freckles start to fade, at 6 the shelves split into strips and an
    /// eyelid doubles, at 7 the faces ghost.
    package static let halfSizeSteps = 4

    /// **The half-size start builds two DiTs, one at a time**: the half-grid one (its own RoPE grid,
    /// its own arena and sequence `s/4 + T`, exactly a native evaluation at half the size), given
    /// back before the full one maps its arena — never both, the full one is the larger and is what
    /// the memory need charges (`MemoryNeed.krea2`). The two share the map, its staging stream and
    /// the modulation table (a function of σ alone); the fused text is the same rows.
    package func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                          context: Context) throws {
        let artifact = try Artifact(path: map)
        let tokens = text.rows
        let merged = try fusion(text, artifact: artifact, lora: lora, context: context)
        let (height, width) = (latent.height, latent.width)
        let planned = plan(height: height, width: width, steps: steps, start: start,
                           withLoRA: lora != nil || context.detail != .normal, schedule: context.schedule)
        // The modulation of only the σs evaluated — from `start` to the last non-null one —, in
        // one pass. It depends only on σ: the following images of a batch reuse the first's table.
        let σ = sigmas(steps: steps)
        let modulationKey = "krea2.modulation " + map
        func build(height: Int, width: Int) throws -> Krea2DiT {
            let dit = try Krea2DiT(artifact: artifact, latentHeight: height, latentWidth: width, maxText: tokens,
                                   freezeCut: context.reproducible, neuralEngine: context.neuralEngine, lora: lora)
            dit.cancellation = context.cancellation
            try context.check()
            if let table = context.reserve[modulationKey] as? Krea2DiT.ModulationTable { dit.modulations = table }
            try dit.tabulateModulation(sigmas: Array(context.told(σ, start: start)[start..<(σ.count - 1)]))
            context.reserve[modulationKey] = dit.modulations
            return dit
        }
        var small = planned.reduced > 0
            ? try build(height: planned.reducedHeight, width: planned.reducedWidth) : nil
        var full = small == nil ? try build(height: height, width: width) : nil
        let half = small.map { _ in
            HalfSizeStart(steps: planned.reduced, speed: { x, sigma in
                try merged.withUnsafeBufferPointer {
                    try small!.forward(latent: x, text: $0.baseAddress!, textRows: tokens, sigma: sigma)
                }
            }, handOver: {
                small = nil
                full = try build(height: height, width: width)
            })
        }
        try euler(&latent, sigmas: σ, start: start, context: context, half: half) { x, sigma in
            try merged.withUnsafeBufferPointer {
                try full!.forward(latent: x, text: $0.baseAddress!, textRows: tokens, sigma: sigma)
            }
        }
        context.footprints.denoising = Arena.processFootprint()
    }

    /// A completed fusion, and what determined it.
    private final class FusionDone {
        let text: [Float], lora: LoRA?, merged: [Float]
        init(text: [Float], lora: LoRA?, merged: [Float]) { self.text = text; self.lora = lora; self.merged = merged }
    }

    /// **The text fusion, once per batch**: it depends only on the text and the LoRA, not on the
    /// seed (1.5 to 2 s per image at 512², measured). It dies on exit, before the
    /// DiT asks for its arenas; only its result — tokens × 6,144 — stays in the context.
    private func fusion(_ text: Conditioning, artifact: Artifact, lora: LoRA?,
                        context: Context) throws -> [Float] {
        let key = "krea2.fusion " + map
        if let mergeDone = context.reserve[key] as? FusionDone, mergeDone.lora === lora, mergeDone.text == text.values {
            return mergeDone.merged
        }
        let fusion = try Krea2TextFusion(artifact: artifact, maxTokens: text.rows,
                                         freezeCut: context.reproducible, neuralEngine: context.neuralEngine,
                                         lora: lora)
        fusion.cancellation = context.cancellation
        let merged = try text.values.withUnsafeBufferPointer {
            try fusion.fuse($0.baseAddress!, tokens: text.rows)
        }
        context.reserve[key] = FusionDone(text: text.values, lora: lora, merged: merged)
        return merged
    }
}

// ── Qwen-Image-2.1 ─────────────────────────────────────────────────────────────────────────

/// **Qwen3-VL-8B on the prompt and the references together** — the raw template, each image its
/// `<imageN>` slot, the vision tower, the last layer before the final norm. The references arrive at
/// the DiT's size (`QwenImage21DenoisingModule.preparedReferences`): the copy the VAE encodes too.
package struct QwenImage21TextModule: ImageTextModule {
    package var map: String
    /// The published `processor/` folder (its `tokenizer.json`).
    package var tokenizer: String
    package var output: TextFormat { .qwenImage21 }
    package var name: String { "Qwen3-VL-8B" }

    package init(map: String, tokenizer: String) { self.map = map; self.tokenizer = tokenizer }

    package func encoder(_ prompt: String, images: [ImageRGB], context: Context) throws -> Conditioning {
        let begin = Date()
        let tokenizer = try Tokenizer(directory: tokenizer)
        // Already at their size: `pixels` refuses one that `smart_resize` would resample.
        let pixels = try images.map { try Qwen3VLImages.pixels(QwenImage21Pipeline.rgb8($0)) }
        context.timings.tokenizer = Date().timeIntervalSince(begin)
        context.footprints.tokenizer = Arena.processFootprint()
        try context.check()
        let artifact = try Artifact(path: map)
        let encoder = try Qwen3VLEncoder(artifact: artifact, tokenizer: tokenizer, freezeCut: context.reproducible)
        encoder.cancellation = context.cancellation
        // Its own autorelease pool: the attention's autoreleased tensor data would otherwise keep
        // Metal buffers alive past the encoder on a thread that never drains (see `QwenImage21DiT.pass`).
        let out = try autoreleasepool { try encoder.encode(prompt: prompt, images: pixels) }
        // 16 GB read once: first in line for reclaim, before the DiT's weights come in.
        artifact.dropFromCache()
        context.footprints.encoder = Arena.processFootprint()
        context.reserve["qwen21.encoder"] = encoder.timings   // vision / language, for a check that measures
        return Conditioning(format: .qwenImage21, rows: out.rows, width: out.hidden, values: out.embeddings,
                            tokens: out.rows, imageSlots: out.imagePadMask)
    }
}

/// **The Qwen-Image-2.1-Turbo DiT**: Euler without CFG, 8 steps on the published `sample_sigmas`, one
/// prefill per render. The user's LoRAs, if any, apply to the Turbo's map itself. See `QwenImage21Pipeline`.
package struct QwenImage21DenoisingModule: DenoisingModule {
    package var map: String
    package var scheduler: QwenImage21Pipeline.Scheduler
    /// `output_resolution`: the references' area (`R²`), and the edit format's.
    package var resolution: Int
    /// Where the K/V cache lives — `nil`: the policy (`QwenImage21CachePolicy`). Set by the checks that measure.
    package var storage: QwenImage21DiT.CacheStorage? = nil
    package var model: String { Family.qwenImage21.rawValue }
    package var name: String { "DiT Qwen-Image-2.1" }
    package var entry: TextFormat { .qwenImage21 }
    package var space: LatentSpace { .qwenImage21 }
    package var defaultSteps: Int { QwenImage21Pipeline.defaultSteps }
    /// The pipeline's edits: 1 to 3 references.
    package var maxReferences: Int { 3 }

    package init(map: String, scheduler: QwenImage21Pipeline.Scheduler,
                resolution: Int = QwenImage21Pipeline.outputResolution) {
        self.map = map; self.scheduler = scheduler; self.resolution = resolution
    }

    /// The published schedule, the same at every size. An unsupported count gives it too:
    /// `check(steps:)` has refused that count before.
    package func sigmas(steps: Int) -> [Float] { scheduler.sigmas }

    /// The official σ, or ComfyUI's for this architecture (`SigmaSchedule`) — a comparison.
    package var schedules: [SigmaSchedule] { SigmaSchedule.allCases }
    package func sigmas(steps: Int, schedule: SigmaSchedule) -> [Float] {
        switch schedule {
        case .official: return sigmas(steps: steps)
        case .comfyUI: return QwenImage21Pipeline.comfySigmas(steps: steps)
        }
    }

    /// **By default every evaluation at full size** (since 0.10): the published recipe as it stands —
    /// a half-size phase changes the image, and a recipe compared with another (ComfyUI's,
    /// `SigmaSchedule`) must be the recipe itself.
    ///
    /// **The half-size start** (`halfSizeStart`, off by default; Z-Image's recipe): the first
    /// `halfSizeSteps` evaluations on the half grid, by the half-size phase of `QwenImage21Sampler` —
    /// not the `x += dt·𝓘v(𝓟x)` steps it ran until 0.10 (two, judged clean then), which left the
    /// latent's high-frequency noise in place and handed the model an alias. Only at the recipe's
    /// eight, from the first step, on a grid whose half stays at 512 px on each side (`Spectral.floor`
    /// in 16-px cells: 1024², 1024×1536 and its transpose; a 512² or 832×1216 never halves), never under
    /// a user LoRA, Detail or another schedule (`withLoRA`: never judged under them) and never in an
    /// edit — the plan cannot see the references, the chain zeroes it (`DenoisingPlan.unreduced`) and
    /// the denoiser ignores it with references. `SILICONED_SPECTRAL` forces `k` (a measurement),
    /// within the same rules.
    package func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan {
        let halvable = Spectral.divisible(height: height, width: width, by: 2,
                                          floor: Spectral.floor(factor: space.factor))
        let rule = Spectral.halfSizeStart && steps == defaultSteps ? Self.halfSizeSteps : 0
        let k = start > 0 || !halvable || withLoRA ? 0 : min(max(0, Spectral.forced ?? rule), steps - 1)
        return DenoisingPlan(startStep: start, startSigma: sigmas(steps: steps)[start],
                             evaluations: steps - start, reduced: k,
                             reducedHeight: k > 0 ? height / 2 : height, reducedWidth: k > 0 ? width / 2 : width)
    }

    /// Its half-size evaluations: 3 (σ 1 → 0.927), judged by eye on two prompts at 1024², as Z-Image's.
    /// Not the largest that renders: at 4 the freckles fade and the shelves start to block, at 5 the
    /// books are a 2×2 patchwork.
    package static let halfSizeSteps = 3

    /// **Whether a batch's next image may reuse this one's DiT** (`Context.keptDenoiser`): only when its
    /// denoising builds no half-size DiT first (`reduced` 0) — kept beside the half-size one, the full
    /// DiT would be two DiTs at once, which the memory need never charges (`QwenImage21Need`). A batch
    /// with the spectral schedule rebuilds its full DiT per image: the same bits (an isolated render
    /// builds it too), a few milliseconds and, for a generation, a prefill fused into the first step.
    package static func keepsDenoiser(reduced: Int) -> Bool { reduced == 0 }

    package func check(steps: Int, startImage: Bool) throws {
        try QwenImage21Pipeline.check(steps: steps)
        if startImage { throw QwenImage21Pipeline.Failure.imageToImage }
    }

    /// Its pipeline has no img2img: the image encoder serves the references only.
    package var acceptsStartImage: Bool { false }
    /// Its noise leaves in the last two evaluations: an early x̂₀ is blurred and grainy (`Sketch`).
    package var sketches: Bool { false }
    /// Every `Solver`: its sampler walks them all (`SolverWalk`), Euler in its own loop.
    package var solvers: [Solver] { Solver.allCases }

    /// Pillow's Lanczos to `calculate_dimensions(R², w/h)`, each at its own ratio.
    package func preparedReferences(_ images: [ImageRGB]) throws -> [ImageRGB] {
        images.map { QwenImage21Pipeline.preparedReference($0, resolution: resolution) }
    }

    package func editFormat(referenceWidth: Int, referenceHeight: Int) -> (width: Int, height: Int) {
        QwenImage21Pipeline.outputSize(referenceWidth: referenceWidth, referenceHeight: referenceHeight,
                                       resolution: resolution)
    }

    package func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                        context: Context) throws {
        try denoise(&latent, text: text, steps: steps, start: start, references: [], lora: lora, context: context)
    }

    package func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int,
                        references: [Latent], lora: LoRA?, context: Context) throws {
        guard references.allSatisfy({ $0.space == space }), references.count <= maxReferences else {
            throw Request.Failure.references(references.count, max: maxReferences)
        }
        guard text.format == entry, text.imageSlots.count == text.rows else {
            throw Chain.Failure.incompatibleText(output: text.format, entry: entry)
        }
        try check(steps: steps, startImage: start > 0)
        var sampler = QwenImage21Sampler(artifact: try Artifact(path: map), lora: lora)
        sampler.freezeCut = context.reproducible
        sampler.neuralEngine = context.neuralEngine
        sampler.cancellation = context.cancellation
        sampler.storage = storage
        sampler.solver = context.solver
        sampler.solverSeed = context.solverSeed
        let (height, width) = (latent.height, latent.width)
        // Detail, like a LoRA, turns it off (`plan`), and so does another schedule; an edit has none.
        let reduced = references.isEmpty
            ? plan(height: height, width: width, steps: steps, start: start,
                   withLoRA: lora != nil || context.detail != .normal, schedule: context.schedule).reduced : 0
        // **The previous image's DiT is reused only when no half-size DiT comes first**
        // (`keepsDenoiser`): the memory need charges one DiT at a time — the half-size one is built and
        // released before the full one —, and the full DiT of image n kept beside the half-size one of
        // image n + 1 was both (+1.1 GB at 1024²). Between keeping and rebuilding, rebuilding: B.
        if !QwenImage21DenoisingModule.keepsDenoiser(reduced: reduced) { context.keptDenoiser = nil }
        sampler.reusable = context.keptDenoiser as? QwenImage21DiT
        // **The last edit's K/V, kept** (`RenderCache`): only where the policy puts them on disk anyway
        // (a prefix with images); a text-only prefix is computed inside the first step for nearly
        // nothing. The map and the user's stack change the conditions' K/V: both are in the key.
        if let cache = context.cache, storage == nil {
            let prefix = QwenImage21CachePolicy.prefix(conditioningRows: text.rows,
                                                       imageSlots: text.imageSlots.filter { $0 }.count)
            if QwenImage21CachePolicy.storage(prefix: prefix) == .file {
                let key = kvKey(stack: lora?.layers ?? [], text: text, references: references, latent: latent,
                                reproducible: context.reproducible, neuralEngine: context.neuralEngine)
                cache.keepOnlyKV([key])
                sampler.kept = (cache.kvPath(key), key.hex)
                sampler.keptReserve = cache.keptReserve
            }
        }
        let σ = sigmas(steps: steps, schedule: context.schedule)
        context.beginDenoising(evaluations: steps)
        try sampler.run(latent: &latent.values, height: height, width: width, hidden: text.values,
                        slots: text.imageSlots, references: references, sigmas: σ,
                        told: context.told(σ, start: start, bell: .late), reduced: reduced) { step in
            try context.stepDone(sigma: step.sigma, seconds: step.seconds, evaluatedHeight: step.evaluatedHeight,
                                 evaluatedWidth: step.evaluatedWidth, space: space, x: step.estimate, v: step.velocity,
                                 nextSigma: step.estimateSigma, fullHeight: height, fullWidth: width)
        }
        context.keptDenoiser = QwenImage21DenoisingModule.keepsDenoiser(reduced: reduced) ? sampler.reused : nil
        context.footprints.denoising = Arena.processFootprint()
        // What a check that measures reads back: the prefill/steps split and the cache's storage.
        context.reserve["qwen21.denoising"] = (sampler.timings, sampler.used)
    }

    /// **What the conditions' K/V depend on**: the binary and the settings that change
    /// bits, the DiT's map, the user's LoRA stack (each map's revision and strength), the encoder's
    /// states and slots, each reference latent, and the target's grid (it closes the joint sequence).
    /// Not the seed, not σ: the conditions read `t = 0` and never see the target.
    func kvKey(stack: [LoRA.Layer], text: Conditioning, references: [Latent], latent: Latent,
               reproducible: Bool, neuralEngine: Bool) -> RenderCache.Key {
        var k = RenderCache.KeyBuilder("qwen21 kv v2")
        k.add(RenderCache.build); k.add(RenderCache.numerics(reproducible: reproducible, neuralEngine: neuralEngine))
        k.add(RenderCache.stamp(map))
        k.add(stack.count)
        for layer in stack { k.add(RenderCache.stamp(layer.artifact.path)); k.add(Int(layer.strength.bitPattern)) }
        k.add(text.format.name); k.add(text.rows); k.add(text.width)
        k.add(floats: text.values); k.add(flags: text.imageSlots)
        k.add(references.count)
        for reference in references { k.add(reference) }
        k.add(latent.height); k.add(latent.width)
        return k.finish()
    }
}

extension QwenImage21DenoisingModule {
    /// The Turbo's schedule: `model_index.json`'s `sample_sigmas`, checked against `scheduler.json`.
    static func product(map: String, in b: Library) throws -> QwenImage21DenoisingModule {
        QwenImage21DenoisingModule(
            map: map, scheduler: try QwenImage21Pipeline.Scheduler.files(config: try b.component(.qwenImage21, "scheduler.json"),
                                                                         modelIndex: try b.component(.qwenImage21, "model_index.json")))
    }
}

// ── the models: preset chains ────────────────────────────────────────────────────

/// **A model is a preset chain, a name and a license.** Its files are read from a `Library`
/// and checked at construction: a missing file throws here, before a minute of computation, with
/// its full path.
///
/// The constructors and `named` take the library under the same label, `in:` — that of
/// `ModelCard.missing(in:)`. A custom chain becomes a model through
/// `Model(card:chain:)`: it is the only thing `render`, `renderBatch` and `events` take.
package struct Model: Sendable {
    /// The name, the license, the recommended formats: what the catalog knows without opening anything.
    package let card: ModelCard
    package let chain: Chain
    /// The library it was built from — where its renders keep what they do not redo
    /// (`Library.cacheFolder`). `nil` for a hand-composed chain: nothing is cached.
    package let library: Library?

    /// A hand-composed chain, under the sheet of the model whose denoiser it holds.
    package init(card: ModelCard, chain: Chain) { self.card = card; self.chain = chain; self.library = nil }

    package init(card: ModelCard, library: Library, chain: Chain) {
        self.card = card; self.chain = chain; self.library = library
    }

    /// `z-image`, `anima`, `krea2`, `qwen-image-2.1` — what `--model` accepts; an imported model is
    /// called `<family>/<name>` (`z-image/my-model`).
    package var identifier: String { card.id }
    package var name: String { card.name }
    /// The architecture — the target a LoRA declares.
    package var family: Family { card.family }
    /// To be recalled to anyone distributing an image or the model.
    package var license: License { card.license }

    package static let identifiers = ModelCard.allCards.map(\.id)

    package enum Failure: Error, CustomStringConvertible {
        case unknown(String)
        package var description: String {
            switch self {
            case .unknown(let name):
                return "unknown model '\(name)' — \(Model.identifiers.joined(separator: ", ")), or <family>/<name> of an imported model"
            }
        }
    }

    /// The model `identifier` of the library — `EngineError.modelNotInstalled` if a file is missing
    /// (`ModelCard.missing(in:)` says which), `unknownModel` if there is no such model.
    package static func named(_ identifier: String, in b: Library) throws(EngineError) -> Model {
        try building(identifier) { try make(identifier, in: b) }
    }

    /// **The door of a model's construction**: a missing file is a model not installed.
    private static func building(_ identifier: String, _ body: () throws -> Model) throws(EngineError) -> Model {
        do { return try body() } catch is MissingFile {
            throw .modelNotInstalled(model: identifier)
        } catch {
            throw EngineError(error)
        }
    }

    /// `named`, with the internal failure that says which file is missing.
    package static func make(_ identifier: String, in b: Library) throws -> Model {
        switch identifier {
        case "z-image": return try zImage(in: b, card: .zImage, dit: b.ditMap(.zImage))
        case "anima": return try anima(in: b, card: .anima, dit: b.ditMap(.anima))
        case "krea2": return try krea2(in: b, card: .krea2, dit: b.ditMap(.krea2))
        case "qwen-image-2.1": return try qwenImage21(in: b, card: .qwenImage21, dit: b.ditMap(.qwenImage21))
        case "moss-image": return try mossImage(in: b, card: .mossImage, dit: b.ditMap(.mossImage))
        default:
            guard let card = b.importedCards().first(where: { $0.id == identifier }), let dit = card.map else {
                throw Failure.unknown(identifier)
            }
            switch card.family {
            case .zImage: return try zImage(in: b, card: card, dit: dit)
            case .anima: return try anima(in: b, card: card, dit: dit)
            case .krea2: return try krea2(in: b, card: card, dit: dit)
            case .qwenImage21: return try qwenImage21(in: b, card: card, dit: dit)
            case .mossImage: return try mossImage(in: b, card: card, dit: dit)
            }
        }
    }

    /// The library's `store/<name>` map, if it is there (`MissingFile` otherwise, folded into
    /// `modelNotInstalled` by `building`).
    static func map(_ name: String, _ b: Library) throws -> String {
        let path = name.hasPrefix("/") ? name : b.map(name)
        guard FileManager.default.fileExists(atPath: path) else { throw MissingFile(path, .map) }
        return path
    }

    /// Z-Image Turbo — the product's default. `denoising`: a denoiser on another map.
    package static func zImage(in b: Library, denoising: ZImageDenoisingModule? = nil) throws(EngineError) -> Model {
        try building(ModelCard.zImage.id) { try zImage(in: b, card: .zImage, dit: b.ditMap(.zImage), denoising: denoising) }
    }

    static func zImage(in b: Library, card: ModelCard, dit: String,
                       denoising: ZImageDenoisingModule? = nil) throws -> Model {
        Model(card: card, library: b, chain: try Chain(
                   text: ZImageTextModule(map: try map(b.encoderMap(.zImage), b),
                                      tokenizer: try b.component(.zImage, "tokenizer")),
                   denoising: denoising ?? ZImageDenoisingModule(map: try map(dit, b)),
                   decoding: FluxDecodingModule(path: try b.component(.zImage, "vae.safetensors")),
                   encoding: FluxEncodingModule(path: try b.component(.zImage, "vae.safetensors"))))
    }

    /// Anima Turbo v1.1. ⚠️ Non-commercial license: never a product's default.
    package static func anima(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.anima.id) { try anima(in: b, card: .anima, dit: b.ditMap(.anima)) }
    }

    /// Anima's text adapter comes from the same file as the DiT: it lives next to its map.
    static func anima(in b: Library, card: ModelCard, dit: String) throws -> Model {
        let dit = try map(dit, b)
        let adapter = adapterPath(fromMap: dit)
        guard FileManager.default.fileExists(atPath: adapter) else { throw MissingFile(adapter, .map) }
        return Model(card: card, library: b, chain: try Chain(
                   text: AnimaTextModule(map: try map(b.encoderMap(.anima), b),
                                     tokenizerQwen: try b.component(.anima, "tokenizer"),
                                     tokenizerT5: try b.component(.anima, "t5_tokenizer"),
                                     adapter: adapter),
                   denoising: AnimaDenoisingModule(map: dit),
                   decoding: QwenImageDecodingModule(path: try b.component(.anima, "vae.safetensors")),
                   encoding: QwenImageEncodingModule(path: try b.component(.anima, "vae.safetensors"))))
    }

    /// Krea 2 Turbo. ⚖️ Krea 2 Community license: commercial under $1M annual revenue, mandatory
    /// content filter at deployment (§4.2), "Krea" at the head of a distributed model (§3.1).
    package static func krea2(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.krea2.id) { try krea2(in: b, card: .krea2, dit: b.ditMap(.krea2)) }
    }

    static func krea2(in b: Library, card: ModelCard, dit: String) throws -> Model {
        Model(card: card, library: b, chain: try Chain(
                   text: Krea2TextModule(map: try map(b.encoderMap(.krea2), b),
                                     tokenizer: try b.component(.krea2, "tokenizer"),
                                     sockets: Krea2Text.sockets),
                   denoising: Krea2DenoisingModule(map: try map(dit, b)),
                   decoding: QwenImageDecodingModule(path: try b.component(.krea2, "vae.safetensors")),
                   encoding: QwenImageEncodingModule(path: try b.component(.krea2, "vae.safetensors"))))
    }

    /// Qwen-Image-2.1-Turbo: 8 steps, generation and editing by instruction (1 to 3
    /// references, which its encoder sees). ⚠️ Qwen Research license: non-commercial.
    package static func qwenImage21(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.qwenImage21.id) { try qwenImage21(in: b, card: .qwenImage21, dit: b.ditMap(.qwenImage21)) }
    }

    static func qwenImage21(in b: Library, card: ModelCard, dit: String) throws -> Model {
        let vae = try b.component(.qwenImage21, "vae.safetensors")
        return Model(card: card, library: b, chain: try Chain(
                   text: QwenImage21TextModule(map: try map(b.encoderMap(.qwenImage21), b),
                                               tokenizer: try b.component(.qwenImage21, "processor")),
                   denoising: try QwenImage21DenoisingModule.product(map: try map(dit, b), in: b),
                   decoding: QwenImage21DecodingModule(path: vae),
                   encoding: QwenImage21EncodingModule(path: vae)))
    }

    /// MOSS-Image under its 8-step LoRA. Apache 2.0.
    package static func mossImage(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.mossImage.id) { try mossImage(in: b, card: .mossImage, dit: b.ditMap(.mossImage)) }
    }

    static func mossImage(in b: Library, card: ModelCard, dit: String) throws -> Model {
        // The LoRA lives with the components; without it the DiT would render an undistilled image in 8 steps.
        guard let turbo = Installer(library: b).companionMap(.mossImage) else { throw MissingFile("turbo LoRA", .map) }
        guard FileManager.default.fileExists(atPath: turbo) else { throw MissingFile(turbo, .map) }
        return Model(card: card, library: b, chain: try Chain(
            text: MossImageTextModule(map: try map(b.encoderMap(.mossImage), b), tokenizer: try b.component(.mossImage, "tokenizer")),
            denoising: MossImageDenoisingModule(map: try map(dit, b), turbo: turbo),
            decoding: Flux2DecodingModule(path: try b.component(.mossImage, "vae.safetensors"))))
    }
}
