import Foundation

/// **Qwen-Image-2.1 Turbo's render**: `QwenImage21Pipeline` translated into sessions, the sampler
/// Siliconed's own.
///
///     prompt ─ raw template + Qwen tokenizer ─ ids [14 + L] ─▶ ┌ encoder session ┐ ─ states [L, 4096] bf16              once a batch
///     seed ─ torch.randn ─ latent [64, h, w] f32 ─▶ ┌ DiT session (the whole sequence once, then the target) ┐ ─ v ─ solver (CPU)   ×8
///     latent ─ · std + mean (CPU, bf16) ─▶ ┌ decoder session ┐ ─ pixels RGBA [4, 16h, 16w], the alpha dropped           per image
///
/// **An edit** (`Request.references`, one to three) adds, once a batch:
///
///     references ─ `QwenImage21References.prepared` (~R², Pillow's Lanczos) ─┬─ patches ─▶ ┌ vision tower, a picture ┐ ─ rows at the
///                                                                            │             picture's slots of the encoder's sequence
///                                                                            └─ RGBA ─▶ ┌ VAE encoder, a picture ┐ ─ (z − mean)/std
///                                                                               ─ tokens at those slots of the DiT's sequence
///
/// What a port by analogy would miss (`pipeline_qwenimage21.py`, in bf16):
///
///   - **The text is the last layer's output before the final norm**, the system turn's 14 rows dropped after the
///     encoder; the prompt is never padded nor cut (the product bounds it, `Request.maximumPromptTokens`).
///   - **The schedule is published, eight σ** (`sample_sigmas`): no shift computed from the size, no other step count.
///   - **The step is `x += (σᵢ₊₁ − σᵢ) · v`, no negation**, the latent f32 between steps; the DiT is told
///     `(t / 1000)` in bf16, `t = σ · 1000` itself a bf16 value (`QwenImage21Transformer.modelTime`).
///   - **The noise is drawn `(1, 1, 64, h, w)`**, planar, and packed by a plain flatten: no 2 × 2.
///   - **The decoder gives RGBA**; the picture is its RGB, uncomposited (the oracle's `convert("RGB")`).
///   - **Three sessions, one after the other**, each planned under B and let go before the next.
///
///   - **In an edit the DiT's sequence is the encoder's**: the text rows where they were, each reference's latent
///     tokens at its picture's slots (four a slot, row-major), the target last; the encoder's own rows at the slots
///     never reach the DiT.
///   - **The conditions are computed once a render** (the reference's K/V cache, generation included): text and
///     references see neither the target nor the time, so the first step runs the whole sequence and keeps their keys
///     and values, the others run the target alone (`Denoiser`).
///
/// A request's LoRAs are added beside the DiT's products (`LoRAStack`).
package enum QwenImage21Render {
    static func render(_ request: Request, _ admitted: Admission, seeds: [UInt64], model: Model,
                       cancellation: Cancellation?, emit: (Engine.Event) -> Void) throws -> [Engine.Render] {
        let chain = model.chain
        guard let text = chain.text as? QwenImage21TextModule, let denoising = chain.denoising as? QwenImage21DenoisingModule,
              let decoding = chain.decoding as? QwenImage21DecodingModule else { throw EngineError.notOnEngine(model: model.identifier) }
        let references = try denoising.preparedReferences(request.references)
        guard references.isEmpty || chain.encoding is QwenImage21EncodingModule else { throw EngineError.notOnEngine(model: model.identifier) }
        let loras = try LoRAStack(request.loras, on: model.card.family)
        let plan = admitted.plan
        let (height, width, steps, start) = (admitted.height, admitted.width, admitted.steps, admitted.start)
        let space = denoising.space
        let totalStart = Date()
        emit(.start(Engine.Plan(denoising: plan, seeds: seeds, stages: references.isEmpty ? [.text, .denoising, .decoding] : [.text, .image, .denoising, .decoding])))

        // ── the text, once ──
        try cancellation.check()
        emit(.stage(.text, image: 0))
        var timings = Engine.Timings()
        let textStart = Date()
        let tokenizer = try Tokenizer(directory: text.tokenizer)
        // A latent cell is a vision patch: a reference's grid is both, and it takes a quarter of it in slots.
        let grids = references.map { (height: $0.height / space.factor, width: $0.width / space.factor) }
        let (ids, rows, dropped, promptTokens) = try QwenImage21Text.identifiers(request.prompt, pictures: grids.map { $0.height * $0.width / 4 },
                                                                          tokenizer: tokenizer)
        timings.tokenizer = Date().timeIntervalSince(textStart)
        try cancellation.check()
        let slots = Array(rows.dropFirst(dropped))
        let pictures = try references.map { try Qwen3VLImages.pixels(QwenImage21References.rgb8($0)) }
        var key = RenderCache.Key("QwenImage21Render.encode")
        try key.file(text.map)
        key.integers(ids)
        key.integers([dropped])
        for picture in pictures {
            key.integers([picture.gridHeight, picture.gridWidth])
            key.floats(picture.values)
        }
        let (states, cached) = try RenderCache.conditioning(key, of: model) {
            try spoken(encode(ids, dropped: dropped, pictures: pictures, slots: rows, map: text.map), slots: slots)
        }
        timings.text = Date().timeIntervalSince(textStart)
        timings.encoder = timings.text - timings.tokenizer
        emit(.text(tokens: promptTokens, tokenizer: timings.tokenizer, encoder: timings.encoder, cached: cached))

        // ── an edit's references, once: each picture's latent, its tokens row-major ──
        var tokens: [Float] = []
        if let encoding = chain.encoding as? QwenImage21EncodingModule, !references.isEmpty {
            try cancellation.check()
            emit(.stage(.image, image: 0))
            let encodingStart = Date()
            for reference in references { tokens += try Self.tokens(latent(of: reference, path: encoding.path)) }
            timings.encoding = Date().timeIntervalSince(encodingStart)
            emit(.encoding(seconds: timings.encoding))
        }

        // ── the denoising, image by image, one DiT session for the batch ──
        let sequence = try QwenImage21DiT.Sequence(slots: slots, references: grids, height: height, width: width)
        // Announced before its DiT is planned: the plan's seconds are the denoising's, not the text's.
        try cancellation.check()
        emit(.stage(.denoising, image: 0))
        let ditStart = Date()
        var dit: Denoiser? = try Denoiser(map: denoising.map, loras: loras, sequence: sequence, text: states,
                               references: tokens.isEmpty ? nil : HostTensor([sequence.referenceRows, space.channels], floats: tokens))
        let schedule = denoising.sigmas(steps: steps)
        let told = request.detail.modelSigmas(schedule, start: start, bell: denoising.detailBell,
                                              scale: Detail.scale(width: space.factor * width, height: space.factor * height))
        let count = space.channels * height * width
        var pending: [(latent: [Float], timings: Engine.Timings, elapsed: TimeInterval)] = []
        var imageStart = totalStart
        for (n, seed) in seeds.enumerated() {
            try cancellation.check()
            if n > 0 { timings = Engine.Timings(); imageStart = Date() }
            var latent = Latent.noise(space, height: height, width: width, seed: seed, variations: request.variations).values
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
        let decoder = try DecoderSession<QwenImage21VAEDecoder>(path: decoding.path, latentHeight: height, latentWidth: width)
        if let warning = decoder.warning { emit(.warning(warning)) }
        var renders: [Engine.Render] = []
        for (n, image) in pending.enumerated() {
            if n > 0 {
                try cancellation.check()
                emit(.stage(.decoding, image: n))
                decodingStart = Date()
            }
            guard FiniteImage.isFinite(image.latent) else { continue }
            // The decoded RGBA's alpha dropped without compositing: the picture's RGB planes.
            let rgba = try decoder.pixels(QwenImage21VAEDecoder.denormalized(image.latent))
            let pixels = Array(rgba.prefix(rgba.count / 4 * 3))
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

    /// **What the DiT receives**: `σ · 1000` rounded to bf16, `/ 1000` rounded again (the integer timestep under DPM++
    /// and UniPC, `/ 1000` in bf16).
    package static func modelTime(sigma: Float, solver: Solver, first: Bool) -> Float {
        guard let integer = solver.integerTimestep(sigma, first: first) else { return QwenImage21Transformer.modelTime(sigma: sigma) }
        return BFloat.round(Float(integer) / 1000)
    }

    // MARK: - The text

    /// **The text**: Qwen3-VL-8B over the real tokens (template included) — the output of its last layer, before the
    /// model's final norm (the pipeline hooks that norm to hand back its input), `hidden` `[L, 4096]` bf16 values; and
    /// `layer<i>` for each layer of `taps`.
    package static func states(_ ids: [Int], map path: String, taps: Set<Int> = []) throws -> [String: HostTensor] {
        try Qwen3Encoder(config: .qwenImage21, prefix: "model.", path: path).states(ids, taps: taps)
    }

    /// **The text with pictures**: each picture through the vision tower (`Qwen3VLVision`, its tables
    /// `Qwen3VLVision.tables`), then the language model over the whole sequence — the tower's merged rows at the
    /// picture's `<|image_pad|>` rows of `ids`, its three deepstack rows added there after layers 0, 1 and 2, the
    /// positions on three axes (`Qwen3VLImages.rotaryPositions`, `Qwen3VLImages.rotaryTable`). `slots`: the rows of
    /// `ids` that are a picture's. Without a picture, `states(_:map:taps:)`. `entries`: `layer<i>_in`.
    package static func states(_ ids: [Int], pictures: [Qwen3VLImages.Pixels], slots: [Bool], map path: String, taps: Set<Int> = [],
                               entries: Set<Int> = []) throws -> [String: HostTensor] {
        guard !pictures.isEmpty else { return try states(ids, map: path, taps: taps) }
        let rows = try Qwen3VLVision.rows(pictures, config: .eightB, map: path)
        let read = try Qwen3VLGenerator.sequence(slots: slots, pictures: pictures, rows: rows, config: .qwenImage21, tower: .eightB)
        return try Qwen3Encoder(config: .qwenImage21, prefix: "model.", path: path).states(ids, pictures: read, taps: taps, entries: entries)
    }

    /// `prompt_embeds`: the states without the system turn's rows (`_drop_idx`).
    package static func encode(_ ids: [Int], dropped: Int, pictures: [Qwen3VLImages.Pixels] = [], slots: [Bool] = [],
                               map path: String) throws -> HostTensor {
        guard let hidden = try states(ids, pictures: pictures, slots: slots, map: path)["hidden"] else { throw ProgramError("the encoder gave no states") }
        let width = Qwen3Encoder.Config.qwenImage21.width
        return HostTensor([ids.count - dropped, width], floats: Array(hidden.floats[(dropped * width)...]))
    }

    /// **The rows the DiT's `txt_in` reads**: the encoder's rows that are not a picture's slots, in order.
    package static func spoken(_ states: HostTensor, slots: [Bool]) -> HostTensor {
        guard slots.contains(true) else { return states }
        let width = states.shape[1], all = states.floats
        var rows: [Float] = []
        for (i, slot) in slots.enumerated() where !slot { rows += all[(i * width)..<((i + 1) * width)] }
        return HostTensor([rows.count / width, width], floats: rows)
    }

    // MARK: - The references

    /// **One reference's latent**, `[64, h, w]`: the prepared picture as RGBA (`QwenImage21References.rgba`) through the
    /// VAE's encoder, the mean normalized in bf16 (`QwenImage21VAEEncoder.normalized`).
    package static func latent(of reference: ImageRGB, path: String) throws -> [Float] {
        QwenImage21VAEEncoder.normalized(try QwenImage21VAEEncoder.mean(of: QwenImage21References.rgba(reference), height: reference.height,
                                                                         width: reference.width, path: path))
    }

    /// `_pack_latents`: planes `[64, h·w]` to tokens `[h·w, 64]`, row-major.
    package static func tokens(_ latent: [Float]) -> [Float] {
        let channels = LatentSpace.qwenImage21.channels, plane = latent.count / channels
        var out = [Float](repeating: 0, count: latent.count)
        for t in 0..<plane { for c in 0..<channels { out[t * channels + c] = latent[c * plane + t] } }
        return out
    }

    // MARK: - The DiT

    /// **The DiT**, opened once for a batch (`QwenImage21Loop`): its functions loaded, its weights kept under B or read
    /// through a ring, then the prefix — the text, an edit's reference tokens — computed once, each layer's keys and
    /// values kept. Each step then runs the target alone over them. The path is the file's: bf16 weights on the bf16
    /// path, int8 codes in W8A16 under the guard of f16. A request's LoRAs beside the products they adapt.
    package final class Denoiser {
        let loop: QwenImage21Loop

        /// `text`: the encoder's text rows, in order (`spoken`); `references`: `[R, 64]`, each reference's tokens.
        package init(map path: String, loras: LoRAStack = .none, sequence: QwenImage21DiT.Sequence, text: HostTensor, references: HostTensor?) throws {
            let (text, references) = (text.floats, references?.floats ?? [])
            nonisolated(unsafe) let loras = loras
            loop = try blocking {
                let loop = try await QwenImage21Loop.open(map: path, sequence: sequence, loras: loras)
                try await loop.prompt(text: text, references: references)
                return UncheckedSendable(loop)
            }.value
        }

        /// `model_out` for the latent at the told time (`modelTime`), in f32.
        package func velocity(_ latent: [Float], time: Float) throws -> [Float] {
            nonisolated(unsafe) let loop = loop
            return try blocking { try await loop.evaluate(latent, time: time) }
        }
    }
}
