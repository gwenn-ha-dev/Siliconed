import Accelerate
import Foundation
import SiliCore

/// **Z-Image Turbo's render**: `ZImagePipeline` translated into three stages, the sampler Siliconed's own.
///
///     prompt ─ chat template ─ tokenizer ─ ids [L] ─▶ ┌ encoder session ┐ ─ caps [L, 2560] bf16            once a batch
///     seed ─ torch.randn ─ latent f32 ─▶ ┌ DiT session (one evaluation) ┐ ─ velocity ─ −v ─ solver (CPU) ─▶ ×N   per image
///                                       └ latent → bf16, t_freq (CPU), cos/sin (CPU) ┘
///     latent ─▶ ┌ decoder session ┐ ─ pixels [3, H, W]                                                     per image
///
/// **The sampler stays on the CPU, between two runs**: the four solvers, Detail's told σ, the previews and the
/// cancellation read the latent at every step, and a step is a few vector operations on 16·h·w floats against seconds of
/// evaluation. So the DiT session is one evaluation, run once a step with the latent given and the velocity read back.
///
/// What a port by analogy would miss (`pipeline_z_image.py`, in bf16):
///
///   - **The latent stays f32 between steps**; only the DiT's input is brought to bf16 (`latents.to(transformer.dtype)`),
///     its output widened (`.float()`) and **negated** (`noise_pred = −model_out`) before the scheduler's step in f32.
///   - **The DiT is told the inverted time** `(1000 − σ·1000)/1000` in f32 (`FlowMatchSchedule.modelTime`), or the integer
///     timestep under DPM++ and UniPC (`Solver.integerTimestep`), never `1 − σ`.
///   - **The reference's decoder reads the final latent in bf16** (`latents.to(vae.dtype)`) and scales it there, in
///     bf16; here the latent goes in f32 and is scaled in f32 in the decoder's head (`FluxVAE`).
///   - **The three sessions are planned one after the other**, each under B, and let go before the next: a plan holds
///     its weights; three at once would count three models.
///
/// An image to start from is encoded once, before the DiT's session comes (`FluxVAEEncoder`), and mixed with each
/// seed's noise at the first step run.
package enum ZImageRender {
    static func render(_ request: Request, _ admitted: Admission, seeds: [UInt64], model: Model,
                       cancellation: Cancellation?, emit: (Engine.Event) -> Void) throws -> [Engine.Render] {
        let chain = model.chain
        guard let text = chain.text as? ZImageTextModule, let denoising = chain.denoising as? ZImageDenoisingModule,
              let decoding = chain.decoding as? FluxDecodingModule else { throw EngineError.notOnEngine(model: model.identifier) }
        // The LoRAs recognized before the first computation.
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
        let (ids, promptTokens) = try Tokenizer(directory: text.tokenizer).admitChat(user: request.prompt)
        timings.tokenizer = Date().timeIntervalSince(textStart)
        try cancellation.check()
        var key = RenderCache.Key("ZImageRender.encode")
        try key.file(text.map)
        key.integers(ids)
        let (caps, cached) = try RenderCache.conditioning(key, of: model) { try encode(ids, map: text.map) }
        timings.text = Date().timeIntervalSince(textStart)
        timings.encoder = timings.text - timings.tokenizer
        emit(.text(tokens: promptTokens, tokenizer: timings.tokenizer, encoder: timings.encoder, cached: cached))

        // ── the image to start from, once: its latent, the encoder's session let go before the DiT's comes ──
        var origin: Latent?
        if let image = request.image {
            guard let encoding = chain.encoding as? FluxEncodingModule else { throw EngineError.notOnEngine(model: model.identifier) }
            try cancellation.check()
            emit(.stage(.image, image: 0))
            let encodingStart = Date()
            origin = Latent(space: space, height: height, width: width, values: try FluxVAEEncoder.latent(of: image, path: encoding.path))
            timings.encoding = Date().timeIntervalSince(encodingStart)
            emit(.encoding(seconds: timings.encoding))
        }

        // ── the denoising, image by image, one DiT session for the batch ──
        // Announced before its DiT is planned: the plan's seconds are the denoising's, not the text's.
        try cancellation.check()
        emit(.stage(.denoising, image: 0))
        let ditStart = Date()
        var dit: Denoiser? = try Denoiser(map: denoising.map, loras: loras, latentHeight: height, latentWidth: width, capLength: ids.count)
        dit!.caps = caps
        let schedule = FlowMatchSchedule(steps: steps)
        let told = request.detail.modelSigmas(schedule.sigmas, start: start, bell: denoising.detailBell,
                                              scale: Detail.scale(width: request.width, height: request.height))
        let count = space.channels * height * width
        var pending: [(latent: [Float], timings: Engine.Timings, elapsed: TimeInterval)] = []
        var imageStart = totalStart
        for (n, seed) in seeds.enumerated() {
            try cancellation.check()
            if n > 0 { timings = Engine.Timings(); imageStart = Date() }
            let noise = Latent.noise(space, height: height, width: width, seed: seed, variations: request.variations)
            // img2img: σ_s · ε + (1 − σ_s) · z₀ at the first step run, then the txt2img schedule from there, unchanged.
            var latent = (origin.map { Latent.start(image: $0, noise: noise, sigma: schedule.sigmas[start]) } ?? noise).values
            if n > 0 { emit(.stage(.denoising, image: n)) }
            let denoisingStart = n == 0 ? ditStart : Date()
            let walk = SolverWalk(solver: request.solver, schedule: schedule.sigmas, start: start, count: count, seed: seed)
            var done = 0
            for step in start..<(schedule.sigmas.count - 1) {
                try cancellation.check()
                let stepStart = Date()
                let time = request.solver.integerTimestep(told[step], first: step == start).map { Float(1000 - $0) / 1000 }
                    ?? FlowMatchSchedule.modelTime(sigma: told[step])
                var negated = try dit!.velocity(latent, time: time)
                vDSP.multiply(-1, negated, result: &negated)
                latent.withUnsafeMutableBufferPointer { x in
                    negated.withUnsafeBufferPointer { walk.step(latent: x.baseAddress!, derivative: $0.baseAddress!) }
                }
                done += 1
                emit(.step(image: n, index: done, total: plan.evaluations, sigma: schedule.sigmas[step],
                           seconds: Date().timeIntervalSince(stepStart), latentGridHeight: height, latentGridWidth: width))
                if request.previews {
                    let projection = space.preview
                    let rgba = negated.withUnsafeBufferPointer { v in
                        if let estimate = walk.estimate {
                            return projection.apply(x: estimate, v: v.baseAddress!, sigma: 0, height: height, width: width)
                        }
                        return latent.withUnsafeBufferPointer {
                            projection.apply(x: $0.baseAddress!, v: v.baseAddress!, sigma: schedule.sigmas[step + 1],
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
        let decoder = try DecoderSession<FluxVAEDecoder>(path: decoding.path, latentHeight: height, latentWidth: width)
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

    // MARK: - The three stages

    /// **The text**: the encoder over the real tokens, `hidden_states[-2]` back as bf16 values, `[L, 2560]`.
    package static func encode(_ ids: [Int], map path: String) throws -> HostTensor {
        guard let caps = try Qwen3Encoder(config: .zImage, prefix: "model.", path: path).states(ids)["hidden"] else {
            throw ProgramError("the encoder gave no hidden states")
        }
        return caps
    }

    /// **One DiT evaluation**, opened once for a batch: its stage planned under B (`ZImageTransformer.open`), the
    /// caption refined at the first evaluation, the latent and the time's sinusoid given at each. The arithmetic is the
    /// map's and the precision's (`ZImageTransformer.arithmetic`). Let go with it (`Stage.close`).
    package final class Denoiser {
        package var caps: HostTensor?
        let dit: ZImageTransformer
        private var prompted = false

        package init(map path: String, loras: LoRAStack = .none, latentHeight: Int, latentWidth: Int, capLength: Int) throws {
            let file = try ZImageDiT.file(path)
            let geometry = try ZImageTransformer.Geometry(latentHeight: latentHeight, latentWidth: latentWidth, capLength: capLength)
            nonisolated(unsafe) let loras = loras
            dit = try blocking { try await ZImageTransformer.open(file: file, geometry: geometry, precision: CoreAI.inForce, loras: loras) }
        }

        deinit { dit.close() }

        /// What the DiT is: its arithmetic and its plan under B.
        package var described: String { "\(dit.arithmetic), " + dit.stage.plan.line }

        /// What a check reads of an evaluation: the oracle's boundaries (`ZImageDiT.Tap`), true values in f32, written
        /// as they pass into the check's buffers (`tapCounts` values each, made before the DiT is planned: its floor
        /// counts them) — the caption's at the first evaluation.
        package typealias Taps = [ZImageDiT.Tap: UnsafeMutableBufferPointer<Float>]

        /// Each boundary's values for a latent `[16, h, w]` and a caption of `capLength` tokens.
        package static func tapCounts(latentHeight: Int, latentWidth: Int, capLength: Int) throws -> [ZImageDiT.Tap: Int] {
            ZImageTransformer.tapCounts(try ZImageTransformer.Geometry(latentHeight: latentHeight, latentWidth: latentWidth, capLength: capLength))
        }

        /// `model_out` for the latent `[16, h, w]` at the told time, widened to f32 (not yet negated).
        package func velocity(_ latent: [Float], time: Float, taps: Taps? = nil) throws -> [Float] {
            let dit = dit
            if !prompted {
                guard let caps else { preconditionFailure("the caption before the first evaluation") }
                try blocking { try await dit.prompt(caps.floats, taps: taps) }
                prompted = true
            }
            return try blocking { try await dit.evaluate(latent, time: time, taps: taps) }
        }
    }
}
