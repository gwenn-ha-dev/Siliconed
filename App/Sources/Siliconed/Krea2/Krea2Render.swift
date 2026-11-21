import Accelerate
import Foundation
import SiliCore

/// **Krea 2 Turbo's render**: `Krea2Pipeline` translated into three stages, the sampler Siliconed's own.
///
///     prompt ─ template + Qwen tokenizer ─ ids [34 + L + 5] ─▶ ┌ encoder session ┐ ─ 12 taps [L + 5, 12, 2560] bf16   once a batch
///     seed ─ torch.randn ─ latent f32 ─▶ ┌ DiT session (one evaluation, fusion included) ┐ ─ velocity ─ Euler step (CPU) ─▶ ×N
///                                        └ latent → bf16, t_freq (CPU), cos/sin (CPU) ┘                                 per image
///     latent ─ ÷(1/std) + mean (CPU, bf16) ─▶ ┌ decoder session ┐ ─ pixels [3, H, W]                                   per image
///
/// What a port by analogy would miss (`pipeline_krea2.py`, in bf16):
///
///   - **The text is twelve taps of Qwen3-VL, not one state**: `hidden_states[2, 5, …, 35]` (`text_encoder_select_layers`), i.e. the
///     outputs of layers 1, 4, …, 34, the template's 34 tokens removed after the encoder (`get_text_hidden_states`). The reference
///     pads in the middle and masks; the render runs the real tokens alone, its positions (`cumsum(mask) − 1`) being the plain ones.
///   - **The step is `x += (σᵢ₊₁ − σᵢ) · v`, no negation**, the latent f32 between steps; only the DiT's input is brought to bf16.
///   - **The DiT is told `(t / 1000).to(bf16)`** (`pipeline_krea2.py:644`), `t = σ · 1000` in f32, or the integer timestep under
///     DPM++ and UniPC: `Krea2Transformer.modelTime`. Turbo has no guidance: one evaluation a step.
///   - **The decoder's normalization is bf16 and per channel** (`latents / latents_std + latents_mean`): `QwenImageVAEDecoder.normalized`.
///   - **Three sessions, one after the other**, each planned under B and let go before the next.
///
/// An image to start from goes through the VAE's encoder first (`QwenImageVAEEncoder`), then the same loop from its step.
package enum Krea2Render {
    static func render(_ request: Request, _ admitted: Admission, seeds: [UInt64], model: Model,
                       cancellation: Cancellation?, emit: (Engine.Event) -> Void) throws -> [Engine.Render] {
        let chain = model.chain
        guard let text = chain.text as? Krea2TextModule, let denoising = chain.denoising as? Krea2DenoisingModule,
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
        let (ids, promptTokens) = try Krea2Text.admitted(request.prompt, tokenizer: try Tokenizer(directory: text.tokenizer))
        timings.tokenizer = Date().timeIntervalSince(textStart)
        try cancellation.check()
        var key = RenderCache.Key("Krea2Render.encode")
        try key.file(text.map)
        key.integers(ids)
        key.integers(text.sockets)
        let (taps, cached) = try RenderCache.conditioning(key, of: model) { try encode(ids, map: text.map, sockets: text.sockets) }
        timings.text = Date().timeIntervalSince(textStart)
        timings.encoder = timings.text - timings.tokenizer
        emit(.text(tokens: promptTokens, tokenizer: timings.tokenizer, encoder: timings.encoder, cached: cached))

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
        var dit: Denoiser? = try Denoiser(map: denoising.map, loras: loras, taps: taps, latentHeight: height, latentWidth: width)
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
                timings: t, footprints: Engine.Footprints(), tokens: promptTokens, detail: request.detail, solver: request.solver,
                variations: request.variations.filter { $0.effectiveStrength > 0 }, modelName: model.card.name,
                isImportedModel: model.card.isImported)
            emit(.image(index: n, render))
            renders.append(render)
        }
        if renders.isEmpty { throw EngineError.nonFiniteImage }
        return renders
    }

    // MARK: - The time

    /// **What the DiT receives**: `t = σ · 1000` in f32 (the integer timestep under DPM++ and UniPC), `/ 1000`, brought to bf16.
    package static func modelTime(sigma: Float, solver: Solver, first: Bool) -> Float {
        guard let integer = solver.integerTimestep(sigma, first: first) else { return Krea2Transformer.modelTime(sigma: sigma) }
        return BFloat.round(Float(integer) / 1000)
    }

    // MARK: - The text

    /// **The text**: Qwen3-VL's twelve taps over the real tokens (template included), the template's rows dropped
    /// (`assemble`): `[L, 12, 2560]` bf16 values.
    package static func encode(_ ids: [Int], map path: String, sockets: [Int]) throws -> HostTensor {
        let encoder = Qwen3Encoder(config: .krea2, prefix: "model.language_model.", path: path)
        let states = try encoder.states(ids, taps: Set(sockets.map { $0 - 1 }))
        return try assemble(states, sockets: sockets, rows: ids.count, width: encoder.config.width)
    }

    /// `torch.stack([hidden_states[i] for i in sockets], dim=2)[:, 34:]`: `hidden_states[i]` is the output of layer `i − 1`.
    package static func assemble(_ states: [String: HostTensor], sockets: [Int], rows: Int, width: Int) throws -> HostTensor {
        let kept = rows - Krea2Text.templateTokens
        var values = [Float](repeating: 0, count: kept * sockets.count * width)
        for (slot, socket) in sockets.enumerated() {
            guard let state = states["layer\(socket - 1)"]?.floats else { throw ProgramError("the encoder gave no layer \(socket - 1)") }
            for row in 0..<kept {
                let from = (Krea2Text.templateTokens + row) * width, to = (row * sockets.count + slot) * width
                values.replaceSubrange(to..<(to + width), with: state[from..<(from + width)])
            }
        }
        return HostTensor([kept, sockets.count, width], floats: values)
    }

    // MARK: - The DiT

    /// **The DiT, opened once for a batch** (`Krea2Session`): the text fused once (`Krea2Fusion`, its arena let go before
    /// the DiT's comes), then one evaluation a step. The file says the path: `convrot` codes are the Compact's, its shared
    /// forms (`Krea2Shared`) on the GPU, the ANE taking its share under the precision `ane` (a control's rows, else found in
    /// flight: `Block`); bf16 weights the Standard's, bf16 end to end. A LoRA lands on the fusion's blocks and the DiT's (`LoRA.swift`).
    package final class Denoiser {
        let session: Krea2Session
        let file: SafeTensors
        /// The fused text and its rows in the stream (`text_fused`, `txt_in_out`): a check's (`boundaries`), nil otherwise.
        package let text: (fused: [Float], rows: [Float])?
        /// The velocity, its storage made before the stage is planned, so that its floor counts it: each evaluation writes
        /// it in place, and its caller holds none past the next.
        private var velocities: [Float]

        /// `boundaries`: the functions that write block 0's inner boundaries loaded too, and the fused text kept (a check's).
        package init(map path: String, loras: LoRAStack = .none, taps: HostTensor, latentHeight: Int, latentWidth: Int,
                     boundaries: Bool = false) throws {
            // A fine-tune saved as a whole checkpoint keeps the DiT under its prefix: read without it, as published.
            let marker = "blocks.0.attn.wq.weight"
            let file = try MapWeights.dit(path, marker: marker)
            self.file = file
            let codes = file.tensors[marker]?.dtype == "I8"
            let budget = MemoryBudget.machineCeiling()
            let fused = try blocking { try await Krea2Fusion.text(taps, file: file, loras: loras, budget: budget, fused: boundaries) }
            text = boundaries ? (fused.fused, fused.rows) : nil
            velocities = [Float](repeating: 0, count: 16 * latentHeight * latentWidth)
            let geometry = Krea2Session.Geometry(latentHeight: latentHeight, latentWidth: latentWidth, text: taps.shape[0])
            let rows = fused.rows
            let ranks = Krea2Functions.ranks(loras, blocks: Krea2Session.blocks.map { $0 + "." })
            let arithmetic = try Arithmetic.opened(codes: codes, precision: CoreAI.inForce) {
                Krea2Session.declaration(geometry, $0, ranks: ranks, taps: boundaries)
            }
            session = try blocking {
                let session = try await Krea2Session.open(file: file, geometry: geometry, arithmetic: arithmetic, loras: loras, taps: boundaries, budget: budget)
                await session.setText(rows)
                return session
            }
        }

        /// What the DiT is: its arithmetic and its plan under B.
        package var described: String { "\(session.arithmetic), " + session.stage.plan.line }

        /// The stage let go before the next one comes, the footprint waited for (`Stage.close`).
        deinit { session.close() }

        /// What a check reads of an evaluation: each boundary in the reference's rows, text first (`Krea2Transformer.Tap`),
        /// true values in f32 written into the check's own buffers (`tapCounts` values each) — made before the DiT is
        /// planned, so that its floor counts them.
        package typealias Taps = [Krea2Transformer.Tap: UnsafeMutableBufferPointer<Float>]

        /// Each boundary's values the DiT writes at this geometry (`Krea2Session.tapCounts`).
        package static func tapCounts(latentHeight: Int, latentWidth: Int, text: Int) -> [Krea2Transformer.Tap: Int] {
            Krea2Session.tapCounts(Krea2Session.Geometry(latentHeight: latentHeight, latentWidth: latentWidth, text: text))
        }

        /// `model_out` for the latent at the told time (`modelTime`); `taps`, the boundaries a check reads too (the
        /// denoiser opened with `boundaries`).
        package func velocity(_ latent: [Float], time: Float, taps: Taps? = nil) throws -> [Float] {
            let s = UncheckedSendable(session), t = UncheckedSendable(taps)
            try velocities.withUnsafeMutableBufferPointer { v in
                let out = UncheckedSendable(v)
                try blocking { try await s.value.evaluate(latent, time: time, into: out.value, taps: t.value) }
            }
            return velocities
        }
    }
}
