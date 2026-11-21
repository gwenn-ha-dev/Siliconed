import Accelerate
import Foundation
import SiliCore

/// **MOSS-Image's render** (`MOSSImagePipeline` of the publisher's `pipeline_moss_image.py`, the community's 8-step
/// recipe: the DMD LoRA at strength 1, one evaluation per step, CFG 1), translated into three sessions, the sampler Siliconed's own.
///
///     prompt ─ chat template ─ tokenizer ─ ids [T] ─▶ ┌ encoder session ┐ ─ hidden [T, 4096] bf16                          once a batch
///     seed ─ torch.randn ─ latent [128, h, w] f32 ─▶ ┌ DiT session (one evaluation, the LoRA apart) ┐ ─ v ─ solver (CPU)       per step
///     latent ─▶ ┌ decoder session (f32) ┐ ─ pixels [3, 16h, 16w], clamped                                                      per image
///
/// What a port by analogy would miss (`pipeline_moss_image.py`):
///
///   - **The latent stays f32 between steps** (`latents` fp32, the Euler update in f32); only the DiT's input is brought to bf16 and its
///     output widened (`.sample.float()`). The velocity is **not negated**: `scheduler.step` adds `(σ_next − σ) · v`;
///   - **the DiT is told `σ·1000`** in f32 (`scheduler.timesteps`), σ being the snapped grid's (`MossImageSchedule`);
///   - **the decoder reads the latent in f32** and decodes in f32 (the VAE's own dtype), the output clamped to [−1, 1];
///   - **the caption is encoded once, unpadded**, real tokens only; no negative branch at CFG 1.
///
/// An image to start from is encoded once, before the DiT's session comes (`Flux2VAEEncoder`), and mixed with each seed's noise at the
/// first step run. Not yet, refused before any computation: a solver other than Euler (the recipe's).
package enum MossImageRender {
    static func render(_ request: Request, _ admitted: Admission, seeds: [UInt64], model: Model,
                       cancellation: Cancellation?, emit: (Engine.Event) -> Void) throws -> [Engine.Render] {
        let chain = model.chain
        guard let text = chain.text as? MossImageTextModule, let denoising = chain.denoising as? MossImageDenoisingModule,
              let decoding = chain.decoding as? Flux2DecodingModule else { throw EngineError.notOnEngine(model: model.identifier) }
        // What the engine cannot run yet, before the first computation.
        let user = try LoRAStack(request.loras, on: model.card.family)
        guard request.solver == .euler else { throw EngineError.notOnEngine(model: model.identifier) }
        let plan = admitted.plan
        let space = denoising.space
        let (height, width, steps, start) = (admitted.height, admitted.width, admitted.steps, admitted.start)
        let totalStart = Date()
        emit(.start(Engine.Plan(denoising: plan, seeds: seeds, stages: request.image == nil ? [.text, .denoising, .decoding] : [.text, .image, .denoising, .decoding])))

        // ── the text, once ──
        try cancellation.check()
        emit(.stage(.text, image: 0))
        var timings = Engine.Timings()
        let textStart = Date()
        let (ids, promptTokens) = try Qwen35Prompt.admitted(request.prompt, tokenizer: try Tokenizer(directory: text.tokenizer))
        timings.tokenizer = Date().timeIntervalSince(textStart)
        try cancellation.check()
        var key = RenderCache.Key("MossImageRender.encode")
        try key.file(text.map)
        key.integers(ids)
        let (caps, cached) = try RenderCache.conditioning(key, of: model) { try encode(ids, map: text.map) }
        timings.text = Date().timeIntervalSince(textStart)
        timings.encoder = timings.text - timings.tokenizer
        emit(.text(tokens: promptTokens, tokenizer: timings.tokenizer, encoder: timings.encoder, cached: cached))

        // ── the image to start from, once: its latent, the encoder's session let go before the DiT's comes ──
        var origin: Latent?
        if let image = request.image {
            guard let encoding = chain.encoding as? Flux2EncodingModule else { throw EngineError.notOnEngine(model: model.identifier) }
            try cancellation.check()
            emit(.stage(.image, image: 0))
            let encodingStart = Date()
            origin = Latent(space: space, height: height, width: width, values: try Flux2VAEEncoder.latent(of: image, path: encoding.path))
            timings.encoding = Date().timeIntervalSince(encodingStart)
            emit(.encoding(seconds: timings.encoding))
        }

        // ── the denoising, image by image, one DiT session for the batch ──
        // Announced before its DiT is planned: the plan's seconds are the denoising's, not the text's.
        try cancellation.check()
        emit(.stage(.denoising, image: 0))
        let ditStart = Date()
        var dit: Denoiser? = try Denoiser(map: denoising.map, loras: try MossImageRender.recipe(denoising.turbo, then: user), latentHeight: height, latentWidth: width, textLength: ids.count)
        dit!.text = caps
        let sigmas = MossImageSchedule.sigmas(steps: steps)
        let timesteps = MossImageSchedule.timesteps(request.detail.modelSigmas(
            sigmas, start: start, bell: denoising.detailBell, scale: Detail.scale(width: request.width, height: request.height)))
        let count = space.channels * height * width
        var pending: [(latent: [Float], timings: Engine.Timings, elapsed: TimeInterval)] = []
        var imageStart = totalStart
        for (n, seed) in seeds.enumerated() {
            try cancellation.check()
            if n > 0 { timings = Engine.Timings(); imageStart = Date() }
            let noise = Latent.noise(space, height: height, width: width, seed: seed, variations: request.variations)
            // img2img: σ_s · ε + (1 − σ_s) · z₀ at the first step run, then the txt2img schedule from there, unchanged.
            var latent = (origin.map { Latent.start(image: $0, noise: noise, sigma: sigmas[start]) } ?? noise).values
            if n > 0 { emit(.stage(.denoising, image: n)) }
            let denoisingStart = n == 0 ? ditStart : Date()
            let walk = SolverWalk(solver: request.solver, schedule: sigmas, start: start, count: count, seed: seed)
            var done = 0
            for step in start..<(sigmas.count - 1) {
                try cancellation.check()
                let stepStart = Date()
                let velocity = try dit!.velocity(latent, time: timesteps[step])
                latent.withUnsafeMutableBufferPointer { x in
                    velocity.withUnsafeBufferPointer { walk.step(latent: x.baseAddress!, derivative: $0.baseAddress!) }
                }
                done += 1
                emit(.step(image: n, index: done, total: plan.evaluations, sigma: sigmas[step],
                           seconds: Date().timeIntervalSince(stepStart), latentGridHeight: height, latentGridWidth: width))
                if request.previews {
                    let projection = space.preview
                    let rgba = velocity.withUnsafeBufferPointer { v in
                        if let estimate = walk.estimate {
                            return projection.apply(x: estimate, v: v.baseAddress!, sigma: 0, height: height, width: width)
                        }
                        return latent.withUnsafeBufferPointer {
                            projection.apply(x: $0.baseAddress!, v: v.baseAddress!, sigma: sigmas[step + 1],
                                             height: height, width: width)
                        }
                    }
                    emit(.preview(Engine.Preview(image: n, index: done, total: plan.evaluations, height: height, width: width, rgba: rgba)))
                }
            }
            timings.denoising = Date().timeIntervalSince(denoisingStart)
            pending.append((latent, timings, Date().timeIntervalSince(imageStart)))
        }
        // The DiT leaves before the decoder comes (`DecoderSession`).
        dit = nil

        // ── the decoding, image by image, one decoder session ──
        // Announced before its session is planned: the plan's seconds are the first image's decoding.
        try cancellation.check()
        emit(.stage(.decoding, image: 0))
        var decodingStart = Date()
        let decoder = try DecoderSession<Flux2VAEDecoder>(path: decoding.path, latentHeight: height, latentWidth: width)
        if let warning = decoder.warning { emit(.warning(warning)) }
        var renders: [Engine.Render] = []
        for (n, image) in pending.enumerated() {
            if n > 0 {
                try cancellation.check()
                emit(.stage(.decoding, image: n))
                decodingStart = Date()
            }
            guard FiniteImage.isFinite(image.latent) else { continue }
            let pixels = try decoder.pixels(image.latent)
            var t = image.timings
            t.decoding = Date().timeIntervalSince(decodingStart)
            t.total = image.elapsed + t.decoding
            emit(.decoding(seconds: t.decoding))
            guard FiniteImage.isFinite(pixels) else { continue }
            let render = Engine.Render(
                model: chain.denoising.model, prompt: request.prompt, seed: seeds[n], steps: steps, loras: request.loras,
                image: ImageRGB(pixels: pixels, height: space.factor * height, width: space.factor * width),
                evaluations: plan.evaluations, startStep: plan.startStep, startSigma: plan.startSigma,
                timings: t, footprints: Engine.Footprints(), tokens: promptTokens, detail: request.detail, solver: request.solver,
                variations: request.variations.filter { $0.effectiveStrength > 0 }, modelName: model.card.name,
                isImportedModel: model.card.isImported)
            emit(.image(index: n, render))
            renders.append(render)
        }
        if renders.isEmpty { throw EngineError.nonFiniteImage }
        return renders
    }

    // MARK: - The three sessions

    /// **The text**: Qwen3.5-9B over the real tokens, `last_hidden_state` back as bf16 values `[L, 4096]`; its rotary
    /// tables are `Qwen35Encoder.rotaryTables`.
    static func encode(_ ids: [Int], map path: String) throws -> HostTensor { try Qwen35Encoder.encode(ids, map: path) }

    /// **The recipe's stack**: the DMD LoRA at strength 1, then the request's own.
    package static func recipe(_ turbo: String, then user: LoRAStack = .none) throws -> LoRAStack {
        LoRAStack(entries: [(try LoRAFile(turbo), 1)] + user.entries)
    }

    /// **One DiT evaluation**, opened once for a batch: the caption given once (`text`), refined by the text blocks at the
    /// first evaluation, then the latent and the told timestep at every one.
    package final class Denoiser {
        package var text: HostTensor? { didSet { prompted = false } }
        private let dit: MossDiT
        private var prompted = false

        package init(map path: String, loras: LoRAStack, latentHeight: Int, latentWidth: Int, textLength: Int) throws {
            let file = try SafeTensors(path)
            let g = MossDiT.Geometry(height: latentHeight, width: latentWidth, text: textLength)
            dit = try blocking { try await MossDiT.open(file: file, geometry: g, budget: MemoryBudget.machineCeiling(), loras: loras) }
        }

        deinit { dit.close() }

        /// What the DiT is: its arithmetic and its plan under B.
        package var described: String { dit.described }

        /// `model_out` for the latent at the told timestep (σ·1000), widened to f32.
        package func velocity(_ latent: [Float], time: Float) throws -> [Float] {
            guard let text else { preconditionFailure("the caption is given before the first evaluation") }
            let dit = UncheckedSendable(dit)
            // Prompted once the caption is refined: a refinement that threw (a cancellation) is made again.
            if !prompted {
                try blocking { try await dit.value.prompt(text.floats) }
                prompted = true
            }
            return try blocking { try await dit.value.evaluate(latent, timestep: time) }
        }
    }

    /// **One evaluation with every boundary of the reference read** (`check moss-image dit`): the DiT at `latent`
    /// `[128, h, w]`, the caption `text` `[T, 4096]`, the told timestep; the velocity and the boundaries by the oracle's names.
    package static func probe(map path: String, latent: [Float], latentHeight: Int, latentWidth: Int, text: [Float], textLength: Int,
                              timestep: Float, loras: LoRAStack = .none) throws -> (velocity: [Float], boundaries: [String: [Float]]) {
        let file = try SafeTensors(path)
        let g = MossDiT.Geometry(height: latentHeight, width: latentWidth, text: textLength)
        return try blocking {
            let dit = try await MossDiT.open(file: file, geometry: g, budget: MemoryBudget.machineCeiling(), tapped: true, loras: loras)
            try await dit.prompt(text)
            let v = try await dit.evaluate(latent, timestep: timestep)
            let boundaries = dit.boundaries
            dit.close()
            return (v, boundaries)
        }
    }
}
