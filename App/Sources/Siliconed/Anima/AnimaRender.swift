import Foundation
import SiliCore

/// **Anima Turbo's render**: `AnimaModularPipeline` translated into four stages, the sampler Siliconed's own.
///
///     prompt ─ Qwen tokenizer ─ ids [Lq] ─▶ ┌ encoder session ┐ ─ hidden [Lq, 1024] bf16 ─┐
///            └ T5 tokenizer ─ ids [Lt] ───────────────────────────────────────────────────┴▶ ┌ conditioner session ┐ ─ context [≥512, 1024]
///     seed ─ torch.randn ─ latent f32 ─▶ ┌ DiT session (one evaluation) ┐ ─ velocity ─ Euler step (CPU) ─▶ ×N       per image
///                                        └ latent → bf16, t_freq (CPU), cos/sin (CPU) ┘
///     latent ─ ÷(1/std) + mean (CPU, bf16) ─▶ ┌ decoder session ┐ ─ pixels [3, H, W]                                 per image
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
    /// **Anima's part of the render** (`RenderLoop`): two tokenizers, Qwen3-0.6B and the conditioner, the VAE's encoder for
    /// an image to start from, the DiT told `modelTime`, the decoder reading the latent normalized in bf16.
    struct Parts: RenderParts {
        typealias Decoder = QwenImageVAEDecoder
        private let inputs: RenderInputs<AnimaTextModule, AnimaDenoisingModule, QwenImageEncodingModule, QwenImageDecodingModule>
        private var qwenIds: [Int] = [], t5Ids: [Int] = [], context: HostTensor?

        init(_ request: Request, _ admitted: Admission, model: Model) throws { inputs = try RenderInputs(request, admitted, model: model) }

        var encodes: Bool { inputs.start != nil }
        var decoderPath: String { inputs.decoding.path }

        mutating func tokens() throws -> Int {
            (qwenIds, t5Ids) = try AnimaRender.tokens(inputs.request.prompt, inputs.text)
            return qwenIds.count
        }

        mutating func encode() throws -> Bool {
            let (qwenIds, t5Ids, text) = (qwenIds, t5Ids, inputs.text)
            var key = RenderCache.Key("AnimaRender.encode")
            try key.file(text.map)
            try key.file(text.adapter)
            key.integers(qwenIds)
            key.integers(t5Ids)
            let (context, cached) = try RenderCache.conditioning(key, of: inputs.model) {
                try AnimaRender.encode(qwenIds: qwenIds, t5Ids: t5Ids, encoder: text.map, adapter: text.adapter)
            }
            self.context = context
            return cached
        }

        func image() throws -> Latent? { try inputs.image { try QwenImageVAEEncoder.latent(of: $0, path: $1.path) } }

        func denoiser(height: Int, width: Int) throws -> any RenderDenoiser {
            guard let context else { preconditionFailure("the text before the DiT") }
            let dit = try Denoiser(map: inputs.denoising.map, loras: inputs.loras, latentHeight: height, latentWidth: width, textLength: context.shape[0])
            try dit.give(context)
            return dit
        }

        func prepare(height: Int, width: Int) throws -> [Functions.Specialization] {
            let budget = MemoryBudget.machineCeiling(), text = inputs.text
            let c = AnimaConditioner(targetLength: t5Ids.count, sourceLength: qwenIds.count)
            let geometry = AnimaEvaluation.Geometry(height: height, width: width, context: c.rows)
            return try Qwen3Encoder(config: .anima, path: text.map).prepare(tokens: qwenIds.count)
                + AnimaConditioner.prepare(targetLength: t5Ids.count, sourceLength: qwenIds.count, path: text.adapter, budget: budget)
                + (inputs.start.map { try QwenImageVAEEncoder.prepare(height: $0.image.height, width: $0.image.width, path: $0.encoding.path) } ?? [])
                + AnimaEvaluation.prepare(inputs.denoising.map, geometry: geometry, loras: inputs.loras, budget: budget)
        }

        func decoderInput(_ latent: [Float]) -> [Float] { QwenImageVAEDecoder.normalized(latent) }
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

        /// What the DiT is: its arithmetic and its plan under B.
        package var described: String { dit.described }

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

extension AnimaRender.Denoiser: RenderDenoiser {
    /// The velocity itself, at the told time (`AnimaRender.modelTime`): the step is `x += (σᵢ₊₁ − σᵢ) · v`.
    package func derivative(_ latent: [Float], sigma: Float, solver: Solver, first: Bool) throws -> [Float] {
        try velocity(latent, time: AnimaRender.modelTime(sigma: sigma, solver: solver, first: first))
    }

    /// One cycle of its block an evaluation.
    package func expect(evaluations: Int) { dit.block.expect(cycles: evaluations) }
}
