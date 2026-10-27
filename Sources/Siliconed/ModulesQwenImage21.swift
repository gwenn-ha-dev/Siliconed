// The Qwen-Image-2.1 modules (text, denoising). Split from `Modules.swift`.

import Foundation

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
