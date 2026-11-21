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
        try RenderLoop.run(try Parts(request, admitted, model: model), request, admitted, seeds: seeds, model: model,
                           cancellation: cancellation, emit: emit)
    }

    /// **MOSS-Image's part of the render** (`RenderLoop`): the chat template and Qwen3.5-9B's states, Flux 2's encoder for
    /// an image to start from, the DiT with the recipe's LoRA told `σ·1000`, Flux 2's decoder in f32.
    struct Parts: RenderParts {
        typealias Decoder = Flux2VAEDecoder
        private let request: Request, model: Model, admitted: Admission
        private let text: MossImageTextModule, denoising: MossImageDenoisingModule
        private let start: (image: ImageRGB, encoding: Flux2EncodingModule)?
        private let user: LoRAStack
        let decoderPath: String
        private var ids: [Int] = [], caps: HostTensor?

        init(_ request: Request, _ admitted: Admission, model: Model) throws {
            let chain = model.chain
            guard let text = chain.text as? MossImageTextModule, let denoising = chain.denoising as? MossImageDenoisingModule,
                  let decoding = chain.decoding as? Flux2DecodingModule else { throw EngineError.notOnEngine(model: model.identifier) }
            if let image = request.image {
                guard let encoding = chain.encoding as? Flux2EncodingModule else { throw EngineError.notOnEngine(model: model.identifier) }
                start = (image, encoding)
            } else {
                start = nil
            }
            // What the engine cannot run yet, before the first computation.
            user = try LoRAStack(request.loras, on: model.card.family)
            guard request.solver == .euler else { throw EngineError.notOnEngine(model: model.identifier) }
            (self.request, self.model, self.admitted, self.text, self.denoising) = (request, model, admitted, text, denoising)
            decoderPath = decoding.path
        }

        var encodes: Bool { start != nil }

        mutating func tokens() throws -> Int {
            let (ids, promptTokens) = try Qwen35Prompt.admitted(request.prompt, tokenizer: try Tokenizer(directory: text.tokenizer))
            self.ids = ids
            return promptTokens
        }

        mutating func encode() throws -> Bool {
            let (ids, map) = (ids, text.map)
            var key = RenderCache.Key("MossImageRender.encode")
            try key.file(map)
            key.integers(ids)
            let (caps, cached) = try RenderCache.conditioning(key, of: model) { try MossImageRender.encode(ids, map: map) }
            self.caps = caps
            return cached
        }

        func image() throws -> Latent? {
            guard let start else { return nil }
            return Latent(space: denoising.space, height: admitted.height, width: admitted.width,
                          values: try Flux2VAEEncoder.latent(of: start.image, path: start.encoding.path))
        }

        func denoiser(height: Int, width: Int) throws -> any RenderDenoiser {
            let dit = try Denoiser(map: denoising.map, loras: try MossImageRender.recipe(denoising.turbo, then: user),
                                   latentHeight: height, latentWidth: width, textLength: ids.count)
            dit.text = caps
            return dit
        }
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

extension MossImageRender.Denoiser: RenderDenoiser {
    /// The velocity itself, at the told timestep `σ·1000` (`MossImageSchedule.timesteps`): `scheduler.step` adds
    /// `(σ_next − σ) · v`. Euler alone, so neither the solver nor the first step changes it.
    package func derivative(_ latent: [Float], sigma: Float, solver: Solver, first: Bool) throws -> [Float] {
        try velocity(latent, time: sigma * Float(MossImageSchedule.trainTimesteps))
    }
}
