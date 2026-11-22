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
/// first step run.
package enum MossImageRender {
    /// **MOSS-Image's part of the render** (`RenderLoop`): the chat template and Qwen3.5-9B's states, Flux 2's encoder for
    /// an image to start from, the DiT with the recipe's LoRA told `σ·1000`, Flux 2's decoder in f32.
    struct Parts: RenderParts {
        typealias Decoder = Flux2VAEDecoder
        private let inputs: RenderInputs<MossImageTextModule, MossImageDenoisingModule, Flux2EncodingModule, Flux2DecodingModule>
        private var ids: [Int] = [], caps: HostTensor?

        init(_ request: Request, _ admitted: Admission, model: Model) throws { inputs = try RenderInputs(request, admitted, model: model) }

        var encodes: Bool { inputs.start != nil }
        var decoderPath: String { inputs.decoding.path }

        mutating func tokens() throws -> Int {
            let (ids, promptTokens) = try Qwen35Prompt.admitted(inputs.request.prompt, tokenizer: try Tokenizer(directory: inputs.text.tokenizer))
            self.ids = ids
            return promptTokens
        }

        mutating func encode() throws -> Bool {
            let (ids, map) = (ids, inputs.text.map)
            var key = RenderCache.Key("MossImageRender.encode")
            try key.file(map)
            key.integers(ids)
            let (caps, cached) = try RenderCache.conditioning(key, of: inputs.model) { try MossImageRender.encode(ids, map: map) }
            self.caps = caps
            return cached
        }

        func image() throws -> Latent? { try inputs.image { try Flux2VAEEncoder.latent(of: $0, path: $1.path) } }

        func denoiser(height: Int, width: Int) throws -> any RenderDenoiser {
            let dit = try Denoiser(map: inputs.denoising.map, loras: try MossImageRender.recipe(inputs.denoising.turbo, then: inputs.loras),
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
            let file = try MossDiT.file(path)
            let g = MossDiT.Geometry(height: latentHeight, width: latentWidth, text: textLength)
            dit = try blocking { try await MossDiT.open(file: file, geometry: g, budget: MemoryBudget.machineCeiling(), loras: loras) }
        }

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
        let file = try MossDiT.file(path)
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
