import Foundation

/// **A family's part of a render**: what differs from one family to the other — its text, its image stage, its DiT
/// and the time it is told, what its decoder reads and gives. Everything else is `RenderLoop`'s, the same for all.
///
/// The stages are called in this order, each once a batch: `tokens`, `encode`, `image` (when `encodes`), `denoiser`,
/// then `decoderInput` and `picture` once an image. What one stage makes for a later one (ids, a conditioning, an
/// edit's references) the family keeps.
package protocol RenderParts {
    associatedtype Decoder: LatentDecoder
    /// The family's part of `request` on `model`, as admitted: what it refuses before any computation (a module of
    /// another family, a LoRA it does not recognize) thrown here.
    init(_ request: Request, _ admitted: Admission, model: Model) throws
    /// Whether the render has an image stage: an image to start from, or an edit's references.
    var encodes: Bool { get }
    /// The prompt tokenized: the tokens the render reports.
    mutating func tokens() throws -> Int
    /// The text encoded, or read from `RenderCache`: whether it was.
    mutating func encode() throws -> Bool
    /// The image stage (`encodes`): the latent to start from, or nil when what it encodes goes to the DiT instead.
    mutating func image() throws -> Latent?
    /// The DiT for the batch, opened under B and given what it reads once (the text, an edit's references).
    func denoiser(height: Int, width: Int) throws -> any RenderDenoiser
    /// The VAE the decoder reads.
    var decoderPath: String { get }
    /// What the decoder reads of a final latent (the family's normalization).
    func decoderInput(_ latent: [Float]) -> [Float]
    /// The picture's planes `[3, H, W]` of what the decoder gave.
    func picture(_ decoded: [Float]) -> [Float]
}

extension RenderParts {
    package func decoderInput(_ latent: [Float]) -> [Float] { latent }
    package func picture(_ decoded: [Float]) -> [Float] { decoded }
}

/// **What a family's part reads of the render alike**: the request, its model and its admission; the chain's modules in
/// the family's types; the image to start from beside its encoder's module; the request's LoRAs, recognized before the
/// first computation. A chain of other modules is not the family's (`EngineError.notOnEngine`).
package struct RenderInputs<Text: TextModule, Denoising: DenoisingModule, Encoding: ImageEncodingModule, Decoding: DecodingModule> {
    package let request: Request, model: Model, admitted: Admission
    package let text: Text, denoising: Denoising, decoding: Decoding
    package let start: (image: ImageRGB, encoding: Encoding)?
    package let loras: LoRAStack

    package init(_ request: Request, _ admitted: Admission, model: Model) throws {
        let chain = model.chain
        guard let text = chain.text as? Text, let denoising = chain.denoising as? Denoising,
              let decoding = chain.decoding as? Decoding else { throw EngineError.notOnEngine(model: model.identifier) }
        if let image = request.image {
            guard let encoding = chain.encoding as? Encoding else { throw EngineError.notOnEngine(model: model.identifier) }
            start = (image, encoding)
        } else {
            start = nil
        }
        loras = try LoRAStack(request.loras, on: model.card.family)
        (self.request, self.model, self.admitted, self.text, self.denoising, self.decoding) = (request, model, admitted, text, denoising, decoding)
    }

    /// The image stage: the latent of the image to start from as `encode` gives it (the image, its encoder's module),
    /// nil without one.
    package func image(_ encode: (ImageRGB, Encoding) throws -> [Float]) throws -> Latent? {
        guard let start else { return nil }
        return Latent(space: denoising.space, height: admitted.height, width: admitted.width, values: try encode(start.image, start.encoding))
    }
}

/// **A family's DiT as the loop runs it**: let go when the loop drops it, before the decoder comes.
package protocol RenderDenoiser: AnyObject {
    /// The derivative the solver walks for `latent` at the told σ: the family's time (the solver's integer timestep
    /// at `first`, or its own), its sign.
    func derivative(_ latent: [Float], sigma: Float, solver: Solver, first: Bool) throws -> [Float]
}

/// **The render every family runs**: the stages announced and timed, the text and the image stage once a batch, one
/// DiT for the batch and an image a seed — its noise, its start, the solver's walk, a step's event and preview, the
/// cancellation read at every one —, the DiT let go, one decoder for the batch, the pictures checked and handed back.
///
/// **The sampler stays on the CPU, between two runs**: the four solvers, Detail's told σ, the previews and the
/// cancellation read the latent at every step, and a step is a few vector operations on the latent's floats against
/// seconds of evaluation. **The sessions come one after the other**, each planned under B and let go before the next.
package enum RenderLoop {
    /// **A family's render** (`FamilyTraits.render`): its part made of the request (`RenderParts.init`), then `run`.
    package static func render<Parts: RenderParts & SendableMetatype>(
        _ parts: Parts.Type
    ) -> @Sendable (Request, Admission, [UInt64], Model, Cancellation?, (Engine.Event) -> Void) throws -> [Engine.Render] {
        { request, admitted, seeds, model, cancellation, emit in
            try run(Parts.self, request, admitted, seeds: seeds, model: model, cancellation: cancellation, emit: emit)
        }
    }

    package static func run<Parts: RenderParts>(_: Parts.Type, _ request: Request, _ admitted: Admission, seeds: [UInt64], model: Model,
                                                cancellation: Cancellation?, emit: (Engine.Event) -> Void) throws -> [Engine.Render] {
        var parts = try Parts(request, admitted, model: model)
        let chain = model.chain, denoising = chain.denoising
        let plan = admitted.plan
        let space = denoising.space
        let (height, width, steps, start) = (admitted.height, admitted.width, admitted.steps, admitted.start)
        let totalStart = Date()
        // A decoding taken again (`keeping`): the latents its DiT left, decoded alone.
        if let kept = keeping.flatMap(KeptLatents.read) {
            return try decode(parts, kept, request, admitted, seeds: seeds, model: model, cancellation: cancellation, emit: emit)
        }
        emit(.start(Engine.Plan(denoising: plan, seeds: seeds, stages: parts.encodes ? [.text, .image, .denoising, .decoding] : [.text, .denoising, .decoding])))

        // ── the text, once ──
        try cancellation.check()
        emit(.stage(.text, image: 0))
        var timings = Engine.Timings()
        let textStart = Date()
        let tokens = try parts.tokens()
        timings.tokenizer = Date().timeIntervalSince(textStart)
        try cancellation.check()
        let cached = try parts.encode()
        timings.text = Date().timeIntervalSince(textStart)
        timings.encoder = timings.text - timings.tokenizer
        emit(.text(tokens: tokens, tokenizer: timings.tokenizer, encoder: timings.encoder, cached: cached))

        // ── the image stage, once: its encoder's session let go before the DiT's comes ──
        var origin: Latent?
        if parts.encodes {
            try cancellation.check()
            emit(.stage(.image, image: 0))
            let encodingStart = Date()
            origin = try parts.image()
            timings.encoding = Date().timeIntervalSince(encodingStart)
            emit(.encoding(seconds: timings.encoding))
        }

        // ── the denoising, image by image, one DiT session for the batch ──
        // Announced before its DiT is planned: the plan's seconds are the denoising's, not the text's.
        try cancellation.check()
        emit(.stage(.denoising, image: 0))
        let ditStart = Date()
        // The decoder judged before the DiT, over the floor the DiT will give back: refused now, not after its minutes.
        try Parts.Decoder.admit(path: parts.decoderPath, latentHeight: height, latentWidth: width, budget: MemoryBudget.machineCeiling())
        var dit: (any RenderDenoiser)? = try parts.denoiser(height: height, width: width)
        let schedule = denoising.sigmas(steps: steps)
        let told = request.detail.modelSigmas(schedule, start: start, bell: denoising.detailBell,
                                              scale: Detail.scale(width: request.width, height: request.height))
        let count = space.channels * height * width
        var pending: [KeptLatents.Image] = []
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
                let derivative = try dit!.derivative(latent, sigma: told[step], solver: request.solver, first: step == start)
                latent.withUnsafeMutableBufferPointer { x in
                    derivative.withUnsafeBufferPointer { walk.step(latent: x.baseAddress!, derivative: $0.baseAddress!) }
                }
                done += 1
                emit(.step(image: n, index: done, total: plan.evaluations, sigma: schedule[step],
                           seconds: Date().timeIntervalSince(stepStart), latentGridHeight: height, latentGridWidth: width))
                if request.previews {
                    let projection = space.preview
                    let rgba = derivative.withUnsafeBufferPointer { v in
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
            // A non-finite latent is the DiT's (or what it read), said here: decoded, it would fail at the VAE's stage.
            guard FiniteImage.isFinite(latent) else { emit(.unusable(image: n, stage: .denoising)); continue }
            pending.append(KeptLatents.Image(index: n, latent: latent, timings: timings, elapsed: Date().timeIntervalSince(imageStart)))
        }
        // The DiT leaves before the decoder comes (`DecoderSession`).
        dit = nil
        guard !pending.isEmpty else { throw EngineError.nonFiniteImage(stage: .denoising) }
        let kept = KeptLatents(tokens: tokens, images: pending)
        if let keeping { try kept.write(to: keeping) }
        return try decode(parts, kept, request, admitted, seeds: seeds, model: model, cancellation: cancellation, emit: emit)
    }

    /// **Where the worker's render keeps its latents between its DiT and its decoder** (`RenderOrder.keeping`), the
    /// render's own file, nil outside the worker. The fuse stops the process the moment it passes B, and a decoder that
    /// passes it would lose the DiT's minutes with it: the next worker, handed the same call, finds them here and decodes
    /// them alone (`EngineWorker`), in a form that holds (the fuse marked the one it stopped under, `ThisMac.fuse`).
    /// Each image leaves the file once handed over. Set by the worker for each call, which runs one at a time.
    nonisolated(unsafe) package static var keeping: URL?

    /// **The decoding, image by image, one decoder session**: the latents of `kept`, those its file still holds when
    /// taken again (`keeping`).
    private static func decode<Parts: RenderParts>(_ parts: Parts, _ kept: KeptLatents, _ request: Request, _ admitted: Admission,
                                                   seeds: [UInt64], model: Model, cancellation: Cancellation?,
                                                   emit: (Engine.Event) -> Void) throws -> [Engine.Render] {
        let denoising = model.chain.denoising, space = denoising.space, plan = admitted.plan
        let (height, width, steps, tokens) = (admitted.height, admitted.width, admitted.steps, kept.tokens)
        var left = kept
        let pending = kept.images
        // Never empty: `run` decodes only once a latent came out finite, and the worker hands a render on to the next
        // worker only while its file still holds a latent (`EngineWorker.Pending.handedOn`).
        precondition(!pending.isEmpty, "a decoding with no latent to decode")
        let first = pending[0]

        // Announced before its session is planned: the plan's seconds are the first image's decoding.
        try cancellation.check()
        emit(.stage(.decoding, image: first.index))
        var decodingStart = Date()
        let decoder = try DecoderSession<Parts.Decoder>(path: parts.decoderPath, latentHeight: height, latentWidth: width)
        if let warning = decoder.warning { emit(.warning(warning)) }
        var renders: [Engine.Render] = []
        for (k, image) in pending.enumerated() {
            let n = image.index
            if k > 0 {
                try cancellation.check()
                emit(.stage(.decoding, image: n))
                decodingStart = Date()
            }
            let pixels = parts.picture(try decoder.pixels(parts.decoderInput(image.latent)))
            var t = image.timings
            t.decoding = Date().timeIntervalSince(decodingStart)
            t.total = image.elapsed + t.decoding
            emit(.decoding(seconds: t.decoding))
            defer { left.hand(n, to: keeping) }
            guard FiniteImage.isFinite(pixels) else { emit(.unusable(image: n, stage: .decoding)); continue }
            let render = Engine.Render(
                model: denoising.model, prompt: request.prompt, seed: seeds[n], steps: steps, loras: request.loras,
                image: ImageRGB(pixels: pixels, height: space.factor * height, width: space.factor * width),
                evaluations: plan.evaluations, startStep: plan.startStep, startSigma: plan.startSigma,
                timings: t, footprints: Engine.Footprints(), tokens: tokens, detail: request.detail, solver: request.solver,
                variations: request.variations.filter { $0.effectiveStrength > 0 }, modelName: model.card.name,
                isImportedModel: model.card.isImported)
            emit(.image(index: n, render))
            renders.append(render)
        }
        if renders.isEmpty && kept.handed == 0 { throw EngineError.nonFiniteImage(stage: .decoding) }
        return renders
    }
}
