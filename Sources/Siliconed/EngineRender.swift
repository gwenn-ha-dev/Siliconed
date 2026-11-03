import Foundation
import os

// **The engine's render, stage by stage** (`Engine.render`, `renderBatch`): text, [image encoding,]
// denoising, decoding, each in its own scope. The modules and the chain are in `Chain.swift`.

extension Engine {
    /// **Renders an image. Blocks, and serializes concurrent calls** — the form for the CLI and
    /// tools; an app takes the `async` facade (`Facade.swift`), with the same parameters in the
    /// same order. `cancellation`: raised from another thread, the render stops at the next layer
    /// boundary and throws `EngineError.cancelled`, memory returned. `onProgress` is called on the
    /// render thread. A custom chain goes through a `Model(card:chain:)`. **Throws only
    /// `EngineError`**, the preflights first (`Preflight`).
    package func render(_ request: Request, model: Model, cancellation: Cancellation? = nil,
                       onProgress: (@Sendable (Event) -> Void)? = nil) throws(EngineError) -> Render {
        try renderBatch(request, seeds: [request.seed], model: model, cancellation: cancellation,
                      onProgress: onProgress)[0]
    }

    /// **A batch: the same request under several seeds**, the text (and the input image) encoded
    /// **only once**. `request.seed` is ignored. Each image has its events (`stage(_, image: n)`,
    /// `step(image: n, …)`, `image(index: n, …)`) and its `Render`; cancellation is checked between
    /// two images as everywhere else.
    package func renderBatch(_ request: Request, seeds: [UInt64], model: Model, cancellation: Cancellation? = nil,
                          onProgress: (@Sendable (Event) -> Void)? = nil) throws(EngineError) -> [Render] {
        // Called from the engine's own queue (a progress closure, a warning receiver), `sync` below would wait for itself for ever.
        dispatchPrecondition(condition: .notOnQueue(Engine.file))
        return try EngineError.boundary {
            try Engine.file.sync { try execute(request, seeds, model, cancellation, onProgress) }
        }
    }

    /// The same batch, called **from** the engine queue (the `async` facade is already on it).
    func renderBatchOnQueue(_ request: Request, seeds: [UInt64], model: Model, cancellation: Cancellation?,
                            onProgress: (@Sendable (Event) -> Void)?) throws -> [Render] {
        dispatchPrecondition(condition: .onQueue(Engine.file))
        return try execute(request, seeds, model, cancellation, onProgress)
    }

    /// **The render, stage by stage — a batch of one image or more.**
    ///
    /// The text and the input image each live in their own scope and return their memory before
    /// the DiT; only what they produce survives (the `Conditioning`, a few MB; the latent z₀).
    /// Then, **for each seed, exactly what an isolated render would do**: noise, start,
    /// denoising (the module builds its DiT and returns it, or keeps it for the next seed when the
    /// need counts it: `Context.keptDenoiser`, never beside a half-size start), decoding. What was
    /// costly and does not depend on the seed — Krea 2's fusion and modulation — is kept in
    /// `Context.reserve`. Image n of a batch stays **bit-identical** to an isolated render of its
    /// seed (verified). Between two
    /// renders — two processes, the CLI and the app —, the text, the encoded images and Qwen's last
    /// edit's K/V are read back from the library's cache (`RenderCache`), the same floats.
    private func execute(_ request: Request, _ seeds: [UInt64], _ model: Model,
                          _ cancellation: Cancellation?,
                          _ onProgress: (@Sendable (Event) -> Void)?) throws -> [Render] {
        // One render, one plan: a measure that ends meanwhile applies to the next one (`EngineSettings.adopt`).
        EngineSettings.renderBegan()
        defer { EngineSettings.renderEnded() }
        // A render assumes the whole of B: the Prompt Lab's resident weights go back first.
        PromptWriter.unloadResident()
        let chain = model.chain, cache = model.library?.renderCache
        guard !seeds.isEmpty else { return [] }
        // A request cancelled while waiting on the queue pays for nothing: neither `start`
        // nor the opening of its LoRA stack.
        try cancellation.check()
        // **The request's own refusals first, before any computation and before the memory is judged**
        // (`Admission`, pure): a request the model cannot render is refused for what it is, not for its memory.
        let admitted = try Admission.judge(request, chain: chain, usesANE: EngineSettings.effective.plan.usesANE)
        let (space, height, width, steps, start, plan) = (admitted.space, admitted.height, admitted.width, admitted.steps,
                                                           admitted.start, admitted.plan)
        // Installed, license, format, memory, no other render: the first that fails throws. The
        // lock is held until this function returns, the whole batch.
        // B, this Mac's budget (`MemoryBudget`): the preflight judges the floor against it, and the
        // plan is chosen in it (`MemoryPlan.choose`) — nothing read from the machine's state.
        let ceiling = MemoryBudget.machineCeiling()
        // **The need, from the request's own inputs** (`MemoryNeed`): each reference at its size, the
        // prompt's tokens bounded by its bytes, the batch, what this process holds now (`base`: B covers
        // the whole process, the app's interface included); the LoRA ranks join once the stack is open
        // (below). The base is what the process holds once its frameworks are up
        // (`OperatorFootprint.bringUp`), and alive: what `free` left queued goes back first
        // (`Arena.returnFreedPages`).
        OperatorFootprint.bringUp()
        Arena.returnFreedPages()
        var memoryRequest = model.card.memoryRequest(
            width: request.width, height: request.height,
            references: (request.image.map { [$0] } ?? request.references).map { ($0.width, $0.height) },
            promptTokens: MemoryNeed.Request.promptTokens(request.prompt, family: model.card.family,
                                                           references: request.references.count))
        memoryRequest.solver = request.solver
        memoryRequest.accelerators = admitted.accelerators
        memoryRequest.batch = seeds.count
        memoryRequest.library = model.library ?? .processWide
        let lock = try Preflight.run(model, width: request.width, height: request.height,
                                     references: request.references.count, ceiling: ceiling, memory: memoryRequest)
        defer { withExtendedLifetime(lock) {} }
        // The stack opens BEFORE any computation: an unreadable map throws here, not after a
        // minute — and so does a LoRA forged for another model. The DiTs apply a LoRA only to the
        // modules it names: Anima's on Krea 2 would touch none, and the render would come out
        // without it, without a word.
        let stack = request.loras.isEmpty
            ? nil : try LoRA(paths: request.loras.map { ($0.path, Float($0.strength)) })
        try stack?.checkTarget(model: chain.denoising.model)
        if let stack, model.library != nil {
            // Each LoRA's rank, not the cumulative one alone: two or more stage a `down` (`LoRA.downBytes`).
            memoryRequest.loraRanks = stack.layerRanks
            try model.card.checkMemory(memoryRequest, ceiling: ceiling)
        }
        // The machine's profile is read HERE, at render time — never at request construction.
        let context = Context(cancellation: cancellation,
                                previews: request.previews, onProgress: onProgress)
        context.cache = cache
        context.accelerators = admitted.accelerators
        context.fasterRendering = admitted.fasterRendering
        // A sketch that reaches the plan's last evaluation is the whole render: no stop, no estimate.
        let sketch = request.sketch.map { max(1, $0) }.flatMap { $0 < plan.evaluations ? $0 : nil }
        context.stopAfter = sketch
        context.detail = request.detail
        context.detailScale = Detail.scale(width: request.width, height: request.height)
        context.solver = request.solver
        // The library's warnings become events for the duration of the render (and go to
        // `Warnings.outsideRender` without `onProgress`: a "✗ step 3: the VELOCITY carries …" is not lost),
        // and memory is returned to the system on exit — **including on a cancellation**: the
        // arenas are unmapped by the scope unwinding, this returns what malloc kept.
        let warn: @Sendable (String) -> Void = { message in
            if let onProgress { onProgress(.warning(message)) } else { Warnings.outsideRender?(message) }
        }
        // The plan in B, and its need: what this render may take at most, the figure its own peak is
        // checked against at its end (`MemoryCheck`).
        let memoryPlan = MemoryPlan.choose(memoryRequest, ceiling: ceiling)
        // Never a render of a library's model without its need: the preflight refused an unknown one,
        // and a map replaced since would make it unknown here. A hand-composed chain (no library) is
        // outside the memory contract, like the installation, the license and the lock: its modules
        // are its caller's — the lean plan, no check.
        let need = memoryPlan.need
        if need == nil, model.library != nil { throw EngineError.memoryNeedUnknown(model: model.card.id) }
        // A hand-composed chain is the caller's: said in the render's warnings, so that nobody takes it for the product's.
        let outsideContract = model.library == nil
        PhaseBoundary.begin()
        defer { _ = PhaseBoundary.end() }
        return try MemoryPlan.during(memoryPlan) { try Warnings.during(warn) {
            // A GPU or AMX submission that failed reaches this render's journal, even when its cause
            // was already said by an earlier render (`Attention.report` says each cause once).
            let gpuFailures = Attention.failures
            defer { Attention.summarize(since: gpuFailures) }
            if outsideContract { Warnings.emit("⚠ a hand-composed chain: outside the memory contract (no need, no lock, no license check)") }
            let totalStart = Date()
            context.footprints.start = Arena.processFootprint()
            PhaseBoundary.mark("start")
            // A sketch announces the evaluations it will do: the progress reaches 1 at its image.
            context.emit(.start(Plan(
                denoising: sketch.map { plan.sketched($0) } ?? plan, seeds: seeds,
                stages: request.image != nil || !request.references.isEmpty ? [.text, .image, .denoising, .decoding]
                                                                            : [.text, .denoising, .decoding])))

            // ── The references at the denoiser's size, once: a text stage that reads images and the
            //    image encoder below receive this same copy. ──
            let prepared = request.references.isEmpty ? [] : try chain.denoising.preparedReferences(request.references)

            // ── The text, once for the whole batch. The module returns its memory on exit. ──
            try context.check()
            context.emit(.stage(.text, image: 0))
            let begin = Date()
            // Read back from the cache when the same module has already encoded the same prompt and
            // the same images (`RenderCache`): the floats it wrote, so the same bits.
            // The images enter the key only if the encoder sees them.
            let textKey = (chain.text as? any CacheIdentified).flatMap { module in
                cache.map { _ in RenderCache.textKey(module, prompt: request.prompt,
                                                     images: chain.text is any ImageTextModule ? prepared : []) }
            }
            let text: Conditioning
            var cached = false
            if let textKey, let kept = cache?.conditioning(textKey) {
                text = kept; cached = true
            } else {
                // Each phase in its own autorelease pool: the render is one item of the engine's
                // queue, whose pool drains only at its end — what MPS autoreleases in a phase
                // (descriptors, NDArrays and their buffers) would otherwise live beside the next.
                text = try autoreleasepool {
                    try (chain.text as? any ImageTextModule)?.encoder(request.prompt, images: prepared, context: context)
                        ?? chain.text.encoder(request.prompt, context: context)
                }
                if let textKey { cache?.store(text, key: textKey) }
            }
            context.timings.text = Date().timeIntervalSince(begin)
            context.timings.encoder = context.timings.text - context.timings.tokenizer
            context.emit(.text(tokens: text.tokens, tokenizer: context.timings.tokenizer,
                                    encoder: context.timings.encoder, cached: cached))
            context.footprints.text = Arena.processFootprint()
            PhaseBoundary.returned()
            PhaseBoundary.mark("text")

            // ── The image (img2img), once: the encoder lives in its own scope, before the DiT. ──
            var z₀: Latent? = nil
            if let image = request.image, let encoding = chain.encoding {
                try context.check()
                context.emit(.stage(.image, image: 0))
                let encodingStart = Date()
                // Already at the format (the request fitted it on receipt): `fitted` returns it as is.
                z₀ = try autoreleasepool {
                    try encode(image.fitted(width: request.width, height: request.height), with: encoding, context: context)
                }
                context.timings.encoding = Date().timeIntervalSince(encodingStart)
                context.emit(.encoding(seconds: context.timings.encoding))
                PhaseBoundary.mark("image")
            }

            // ── The references (editing), once: same encoder, each at its own size. ──
            var references: [Latent] = []
            if !prepared.isEmpty, let encoding = chain.encoding {
                try context.check()
                context.emit(.stage(.image, image: 0))
                let referencesStart = Date()
                for image in prepared {
                    try context.check()
                    references.append(try autoreleasepool { try encode(image, with: encoding, context: context) })
                    PhaseBoundary.mark("reference \(references.count)")
                }
                context.timings.encoding += Date().timeIntervalSince(referencesStart)
                context.emit(.encoding(seconds: context.timings.encoding))
            }

            // ── Every image's denoising, then every decoding. ──
            // **Not denoise-decode, denoise-decode** (measured: Z-Image 1408×704, a batch of 3 under an idle
            // ballast swapped 18 008 pages where 1 image left the compressor 5.8 GB of room). Each switch
            // between the DiT and the decoder is a wave of fresh allocations — the arenas released, the
            // decoder's graph, the arenas again — and each wave sends the others' anonymous memory into
            // the compressor, which keeps it: its floor climbed ~2.3 GB at every image boundary, never
            // within a denoising. In this order a batch has the one switch of a single render; the
            // latents waiting meanwhile are kilobytes. Same bits: each image is the same computation.
            var pending: [(latent: Latent, timings: Timings, elapsed: TimeInterval, classes: [ProductLedger.Tally], fallbacks: [ProductLedger.Fallback])] = []
            var imageStart = totalStart
            for (n, seed) in seeds.enumerated() {
                try context.check()
                context.image = n
                context.solverSeed = seed
                if n > 0 {
                    // The text and the input image were paid for by the first.
                    context.timings = Timings()
                    imageStart = Date()
                }
                var latent = Latent.noise(space, height: height, width: width, seed: seed,
                                          variations: request.variations)
                if let z₀ { latent = Latent.start(image: z₀, noise: latent, sigma: plan.startSigma) }

                context.emit(.stage(.denoising, image: n))
                let denoisingStart = Date()
                // The classes this image's evaluations ran, by route (`ProductLedger`): armed for the denoising alone.
                let ledger = ProductLedger()
                do {
                    try autoreleasepool {
                        try ProductLedger.recording(into: ledger) {
                            try chain.denoising.denoise(&latent, text: text, steps: steps, start: start,
                                                        references: references, lora: stack, context: context)
                        }
                    }
                } catch is Context.Sketched {
                    latent.values = context.estimate!
                }
                context.timings.denoising = Date().timeIntervalSince(denoisingStart)
                PhaseBoundary.mark("denoising")
                pending.append((latent, context.timings, Date().timeIntervalSince(imageStart), ledger.tallies, ledger.fallbacks))
            }

            context.keptDenoiser = nil
            PhaseBoundary.returned()
            PhaseBoundary.mark("denoiser released")

            var renderResults: [Render] = []
            var refused = false
            for (n, (latent, timings, elapsed, classes, fallbacks)) in pending.enumerated() {
                try context.check()
                context.image = n
                context.timings = timings
                context.emit(.stage(.decoding, image: n))
                // **An image that is not finite is never delivered as done** (`FiniteImage`): the latent
                // before paying for its decoding, the pixels after it — a black image after 200 s, said
                // as such, rather than handed to the history as a success. In a batch it is skipped, not
                // the batch: the images after it are still decoded and delivered.
                guard FiniteImage.isFinite(latent.values, "the final latent", image: n) else { refused = true; continue }
                let image = try autoreleasepool { try chain.decoding.decode(latent, context: context) }
                guard FiniteImage.isFinite(image.pixels, "the decoded image", image: n) else { refused = true; continue }
                // An image's time: its own text, encoding and denoising, then its decoding.
                context.timings.total = elapsed + context.timings.decoding
                context.footprints.end = Arena.processFootprint()
                PhaseBoundary.mark("decoding")
                context.emit(.decoding(seconds: context.timings.decoding))
                let seed = seeds[n]

                var renderResult = Render(model: chain.denoising.model, prompt: request.prompt, seed: seed,
                                  steps: steps, loras: request.loras, loraSummary: stack?.summary, image: image,
                                  evaluations: sketch ?? plan.evaluations, sketch: sketch, spectral: plan.reduced,
                                  strength: request.image != nil ? request.strength : nil, startStep: plan.startStep,
                                  startSigma: plan.startSigma,
                                  timings: context.timings, footprints: context.footprints, tokens: text.tokens,
                                  detail: request.detail, solver: request.solver,
                                  variations: request.variations.filter { $0.effectiveStrength > 0 },
                                  modelName: model.card.name, isImportedModel: model.card.isImported,
                                  loraOrigins: stack?.origins ?? [], neuralEngine: context.accelerators.neuralEngine,
                                  accelerators: context.accelerators.isOn,
                                  offer: AcceleratorOffer(plan: EngineSettings.effective.plan.forDenoiser(context.accelerators)),
                                  fasterRendering: admitted.fasterRendering, classes: classes,
                                  loraRank: stack?.maxRank ?? 0)
                renderResult.fallbacks = fallbacks
                context.emit(.image(index: n, renderResult))
                renderResults.append(renderResult)
            }
            if refused, renderResults.isEmpty { throw EngineError.nonFiniteImage }
            // **The render's own peak against its need**: a phase above its need is a
            // defect of the code, logged and carried by every image of the batch.
            guard let need else { return renderResults }
            let check = MemoryCheck(need: need, marks: PhaseBoundary.end())
            for defect in check.defects {
                MemoryCheck.log.error("\(defect, privacy: .public) — \(model.card.id, privacy: .public) \(request.width)×\(request.height), \(request.references.count) reference(s)")
            }
            for slack in check.slack {
                MemoryCheck.log.info("\(slack, privacy: .public) — \(model.card.id, privacy: .public) \(request.width)×\(request.height)")
            }
            return renderResults.map { var r = $0; r.memory = check; return r }
        } }
    }

    /// An image through the chain's encoder — or its latent read back from the cache (`RenderCache`).
    private func encode(_ image: ImageRGB, with encoding: any ImageEncodingModule, context: Context) throws -> Latent {
        guard let cache = context.cache, let module = encoding as? any CacheIdentified else {
            return try encoding.encoder(image, context: context)
        }
        let key = RenderCache.imageKey(module, image: image)
        if let kept = cache.latent(key) { return kept }
        let latent = try encoding.encoder(image, context: context)
        cache.store(latent, key: key)
        return latent
    }
}

/// **A render's last guard**: a value that is not finite (NaN, ±∞) in the final latent or the decoded
/// pixels makes an image that comes out black or blank — whatever the family, whatever went wrong upstream
/// (a velocity gone non-finite, a GPU fault). That image gets no `Render`, no `.image` event, nothing in
/// the history; the render throws `EngineError.nonFiniteImage` when no image of it is left. One pass over each, at the end
/// of a render only (a 1024² latent is 65 536 to 262 144 floats, its pixels 3 Mi): never inside a step.
package enum FiniteImage {
    /// How many values of `values` are not finite.
    package static func nonFiniteCount(_ values: [Float]) -> Int {
        values.withUnsafeBufferPointer { b in
            var n = 0
            for v in b where !v.isFinite { n += 1 }
            return n
        }
    }

    /// Whether `values` is finite; when not, the count, and `what` it was found in, go to the render's
    /// warnings — the error the render may end with says nothing technical.
    package static func isFinite(_ values: [Float], _ what: String, image: Int) -> Bool {
        let bad = nonFiniteCount(values)
        guard bad > 0 else { return true }
        Warnings.emit("✗ image \(image + 1): \(what) carries \(bad) non-finite values out of \(values.count): not delivered")
        return false
    }

}
