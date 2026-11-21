import Accelerate
import Foundation

/// **Anima Turbo's render**: `AnimaModularPipeline` translated into four stages, the sampler Siliconed's own.
///
///     prompt ─ Qwen tokenizer ─ ids [Lq] ─▶ ┌ encoder session ┐ ─ hidden [Lq, 1024] bf16 ─┐
///            └ T5 tokenizer ─ ids [Lt] ───────────────────────────────────────────────────┴▶ ┌ conditioner session ┐ ─ context [≥512, 1024]
///     seed ─ torch.randn ─ latent f32 ─▶ ┌ DiT session (one evaluation) ┐ ─ velocity ─ Euler step (CPU) ─▶ ×N       per image
///                                        └ latent → bf16, t_freq (CPU), cos/sin (CPU) ┘
///     latent ─ ÷(1/std) + mean (CPU, bf16) ─▶ ┌ decoder session ┐ ─ pixels [3, H, W]                                 per image
///
/// **The sampler stays on the CPU, between two runs**: a step is a few vector operations on 16·h·w floats against
/// seconds of evaluation, and the four solvers, the previews and the cancellation read the latent at every step.
///
/// What a port by analogy would miss (`modular_pipelines/anima`, in bf16):
///
///   - **The step is `x += (σᵢ₊₁ − σᵢ) · v`, no negation**: the velocity is the derivative the
///     solver wants. The latent stays f32 between steps; only the DiT's input is brought to bf16 (`latents.to(dtype)`).
///   - **The DiT is told `bf16(bf16(t) / 1000)`** (`AnimaDenoiserInputStep`: `t.expand(...).to(dtype)`, then `/ num_train_timesteps`),
///     with `t = σ · 1000` in f32 (the scheduler's `timesteps`), or the integer timestep under DPM++ and UniPC: `modelTime`.
///   - **The decoder's normalization is bf16 and per channel** (`AnimaVaeDecoderStep`): `QwenImageVAEDecoder.normalized`.
///   - **Four sessions, one after the other**, each planned under B and let go before the next: a plan holds its weights.
///
/// An image to start from goes through the VAE's encoder first (`QwenImageVAEEncoder`), then the same loop from its step.
package enum AnimaRender {
    static func render(_ request: Request, _ admitted: Admission, seeds: [UInt64], model: Model,
                       cancellation: Cancellation?, emit: (Engine.Event) -> Void) throws -> [Engine.Render] {
        let chain = model.chain
        guard let text = chain.text as? AnimaTextModule, let denoising = chain.denoising as? AnimaDenoisingModule,
              let decoding = chain.decoding as? QwenImageDecodingModule else { throw EngineError.notOnEngine(model: model.identifier) }
        let loras = try LoRAStack(request.loras, on: model.card.family)
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
        let (qwenIds, t5Ids) = try tokens(request.prompt, text)
        timings.tokenizer = Date().timeIntervalSince(textStart)
        try cancellation.check()
        var key = RenderCache.Key("AnimaRender.encode")
        try key.file(text.map)
        try key.file(text.adapter)
        key.integers(qwenIds)
        key.integers(t5Ids)
        let (context, cached) = try RenderCache.conditioning(key, of: model) {
            try encode(qwenIds: qwenIds, t5Ids: t5Ids, encoder: text.map, adapter: text.adapter)
        }
        timings.text = Date().timeIntervalSince(textStart)
        timings.encoder = timings.text - timings.tokenizer
        emit(.text(tokens: qwenIds.count, tokenizer: timings.tokenizer, encoder: timings.encoder, cached: cached))

        // ── the image to start from, once: its latent, the encoder's session let go before the DiT's comes ──
        var origin: Latent?
        if let image = request.image {
            guard let encoding = chain.encoding as? QwenImageEncodingModule else { throw EngineError.notOnEngine(model: model.identifier) }
            try cancellation.check()
            emit(.stage(.image, image: 0))
            let encodingStart = Date()
            origin = Latent(space: space, height: height, width: width, values: try QwenImageVAEEncoder.latent(of: image, path: encoding.path))
            timings.encoding = Date().timeIntervalSince(encodingStart)
            emit(.encoding(seconds: timings.encoding))
        }

        // ── the denoising, image by image, one DiT session for the batch ──
        // Announced before its DiT is planned: the plan's seconds are the denoising's, not the text's.
        try cancellation.check()
        emit(.stage(.denoising, image: 0))
        let ditStart = Date()
        var dit: Denoiser? = try Denoiser(map: denoising.map, loras: loras, latentHeight: height, latentWidth: width, textLength: context.shape[0])
        try dit!.give(context)
        let schedule = denoising.sigmas(steps: steps)
        let told = request.detail.modelSigmas(schedule, start: start, bell: denoising.detailBell,
                                              scale: Detail.scale(width: request.width, height: request.height))
        let count = space.channels * height * width
        var pending: [(latent: [Float], timings: Engine.Timings, elapsed: TimeInterval)] = []
        var imageStart = totalStart
        for (n, seed) in seeds.enumerated() {
            try cancellation.check()
            if n > 0 { timings = Engine.Timings(); imageStart = Date() }
            let noise = Latent.noise(space, height: height, width: width, seed: seed, variations: request.variations)
            // img2img: σ_s · ε + (1 − σ_s) · z₀ at the first step run, then the txt2img schedule from there, unchanged.
            var latent = (origin.map { Latent.start(image: $0, noise: noise, sigma: schedule[start]) } ?? noise).values
            if n > 0 { emit(.stage(.denoising, image: n)) }
            let denoisingStart = n == 0 ? ditStart : Date()
            let walk = SolverWalk(solver: request.solver, schedule: schedule, start: start, count: count, seed: seed)
            var done = 0
            for step in start..<(schedule.count - 1) {
                try cancellation.check()
                let stepStart = Date()
                let velocity = try dit!.velocity(latent, time: modelTime(sigma: told[step], solver: request.solver, first: step == start))
                latent.withUnsafeMutableBufferPointer { x in
                    velocity.withUnsafeBufferPointer { walk.step(latent: x.baseAddress!, derivative: $0.baseAddress!) }
                }
                done += 1
                emit(.step(image: n, index: done, total: plan.evaluations, sigma: schedule[step],
                           seconds: Date().timeIntervalSince(stepStart), latentGridHeight: height, latentGridWidth: width))
                if request.previews {
                    let projection = space.preview
                    let rgba = velocity.withUnsafeBufferPointer { v in
                        if let estimate = walk.estimate {
                            return projection.apply(x: estimate, v: v.baseAddress!, sigma: 0, height: height, width: width)
                        }
                        return latent.withUnsafeBufferPointer {
                            projection.apply(x: $0.baseAddress!, v: v.baseAddress!, sigma: schedule[step + 1], height: height, width: width)
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
        let decoder = try DecoderSession<QwenImageVAEDecoder>(path: decoding.path, latentHeight: height, latentWidth: width)
        if let warning = decoder.warning { emit(.warning(warning)) }
        var renders: [Engine.Render] = []
        for (n, image) in pending.enumerated() {
            if n > 0 {
                try cancellation.check()
                emit(.stage(.decoding, image: n))
                decodingStart = Date()
            }
            guard FiniteImage.isFinite(image.latent) else { continue }
            let pixels = try decoder.pixels(QwenImageVAEDecoder.normalized(image.latent))
            var t = image.timings
            t.decoding = Date().timeIntervalSince(decodingStart)
            t.total = image.elapsed + t.decoding
            emit(.decoding(seconds: t.decoding))
            guard FiniteImage.isFinite(pixels) else { continue }
            let render = Engine.Render(
                model: chain.denoising.model, prompt: request.prompt, seed: seeds[n], steps: steps, loras: request.loras,
                image: ImageRGB(pixels: pixels, height: space.factor * height, width: space.factor * width),
                evaluations: plan.evaluations, startStep: plan.startStep, startSigma: plan.startSigma,
                timings: t, footprints: Engine.Footprints(), tokens: qwenIds.count, detail: request.detail, solver: request.solver,
                variations: request.variations.filter { $0.effectiveStrength > 0 }, modelName: model.card.name,
                isImportedModel: model.card.isImported)
            emit(.image(index: n, render))
            renders.append(render)
        }
        if renders.isEmpty { throw EngineError.nonFiniteImage }
        return renders
    }

    // MARK: - The time

    /// **What the DiT receives** (`AnimaDenoiserInputStep`): `t = σ · 1000` in f32 (the integer timestep under DPM++ and UniPC),
    /// brought to bf16, divided by 1000 in bf16.
    package static func modelTime(sigma: Float, solver: Solver, first: Bool) -> Float {
        let t = solver.integerTimestep(sigma, first: first).map { Float($0) } ?? sigma * 1000
        return BFloat.round(BFloat.round(t) / 1000)
    }

    // MARK: - The text

    /// The two tokenizers' identifiers: no template, no padding for one prompt. Where the reference truncates (512, T5 keeping its
    /// `</s>`) the render reads on, up to `Request.maximumPromptTokens` per tokenizer, and refuses past it. T5 first: its
    /// vocabulary is the finer one.
    package static func tokens(_ prompt: String, _ text: AnimaTextModule) throws -> (qwen: [Int], t5: [Int]) {
        let t5Ids = try T5Tokenizer(directory: text.tokenizerT5).encode(prompt, maxLength: .max)
        try Request.admitPrompt(tokens: t5Ids.count - 1, max: Request.maximumPromptTokens)
        let qwenIds = try Tokenizer(directory: text.tokenizerQwen).encode(prompt)
        try Request.admitPrompt(tokens: qwenIds.count, max: Request.maximumPromptTokens)
        guard !qwenIds.isEmpty else { throw Request.Failure.emptyPrompt }
        return (qwenIds, t5Ids)
    }

    /// **The text**: Qwen3-0.6B's `last_hidden_state` `[Lq, 1024]`, then the conditioner over it (`AnimaConditioner`,
    /// its rotary tables `AnimaConditioner.rotaryTables`); the context `[max(Lt, 512), 1024]`.
    static func encode(qwenIds: [Int], t5Ids: [Int], encoder path: String, adapter: String) throws -> HostTensor {
        guard let hidden = try Qwen3Encoder(config: .anima, path: path).states(qwenIds)["hidden"] else {
            throw EngineError.internalFailure(component: "Qwen3-0.6B", detail: "no hidden states")
        }
        return try blocking { try await AnimaConditioner.context(source: hidden, t5Ids: t5Ids, path: adapter, budget: MemoryBudget.machineCeiling()).context }
    }

    // MARK: - The DiT

    /// **The DiT for a batch** (`AnimaEvaluation`): its stage planned under B and opened once, the prompt given once
    /// (`give`), an evaluation a step; let go with it.
    package final class Denoiser {
        private let dit: AnimaEvaluation
        private var given = false

        package init(map path: String, loras: LoRAStack = .none, latentHeight: Int, latentWidth: Int, textLength: Int) throws {
            let geometry = AnimaEvaluation.Geometry(height: latentHeight, width: latentWidth, context: textLength)
            dit = try blocking { try await AnimaEvaluation.open(path, geometry: geometry, loras: loras, budget: MemoryBudget.machineCeiling()) }
        }

        deinit { dit.close() }

        /// The conditioner's context `[rows, 1024]`, once for the batch: each block's cross keys and values.
        package func give(_ context: HostTensor) throws {
            let dit = dit
            try blocking { try await dit.prompt(context.floats, rows: context.shape[0]) }
            given = true
        }

        /// `model_out` for the latent at the told time (`modelTime`), in f32.
        package func velocity(_ latent: [Float], time: Float) throws -> [Float] {
            precondition(given, "the DiT's context is given before its first evaluation")
            let dit = dit
            return try blocking { try await dit.evaluate(latent, time: time) }
        }
    }
}
