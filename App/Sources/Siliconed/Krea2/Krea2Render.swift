import Foundation
import Metal
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
        try RenderLoop.run(try Parts(request, admitted, model: model), request, admitted, seeds: seeds, model: model,
                           cancellation: cancellation, emit: emit)
    }

    /// **Krea 2's part of the render** (`RenderLoop`): the template and Qwen3-VL's twelve taps, the VAE's encoder for an
    /// image to start from, the DiT (the text fused once) told `modelTime`, the decoder reading the latent normalized in bf16.
    struct Parts: RenderParts {
        typealias Decoder = QwenImageVAEDecoder
        private let request: Request, model: Model, admitted: Admission
        private let text: Krea2TextModule, denoising: Krea2DenoisingModule
        private let start: (image: ImageRGB, encoding: QwenImageEncodingModule)?
        private let loras: LoRAStack
        let decoderPath: String
        private var ids: [Int] = [], taps: HostTensor?

        init(_ request: Request, _ admitted: Admission, model: Model) throws {
            let chain = model.chain
            guard let text = chain.text as? Krea2TextModule, let denoising = chain.denoising as? Krea2DenoisingModule,
                  let decoding = chain.decoding as? QwenImageDecodingModule else { throw EngineError.notOnEngine(model: model.identifier) }
            if let image = request.image {
                guard let encoding = chain.encoding as? QwenImageEncodingModule else { throw EngineError.notOnEngine(model: model.identifier) }
                start = (image, encoding)
            } else {
                start = nil
            }
            loras = try LoRAStack(request.loras, on: model.card.family)
            (self.request, self.model, self.admitted, self.text, self.denoising) = (request, model, admitted, text, denoising)
            decoderPath = decoding.path
        }

        var encodes: Bool { start != nil }

        mutating func tokens() throws -> Int {
            let (ids, promptTokens) = try Krea2Text.admitted(request.prompt, tokenizer: try Tokenizer(directory: text.tokenizer))
            self.ids = ids
            return promptTokens
        }

        mutating func encode() throws -> Bool {
            let (ids, text) = (ids, text)
            var key = RenderCache.Key("Krea2Render.encode")
            try key.file(text.map)
            key.integers(ids)
            key.integers(text.sockets)
            let (taps, cached) = try RenderCache.conditioning(key, of: model) { try Krea2Render.encode(ids, map: text.map, sockets: text.sockets) }
            self.taps = taps
            return cached
        }

        func image() throws -> Latent? {
            guard let start else { return nil }
            return Latent(space: denoising.space, height: admitted.height, width: admitted.width,
                          values: try QwenImageVAEEncoder.latent(of: start.image, path: start.encoding.path))
        }

        func denoiser(height: Int, width: Int) throws -> any RenderDenoiser {
            guard let taps else { preconditionFailure("the text before the DiT") }
            return try Denoiser(map: denoising.map, loras: loras, taps: taps, latentHeight: height, latentWidth: width)
        }

        func decoderInput(_ latent: [Float]) -> [Float] { QwenImageVAEDecoder.normalized(latent) }
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
    /// forms (`Krea2Shared`) on the GPU, the ANE taking its share with the Neural Engine on (a control's rows, else found in
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
            let ranks = Krea2Session.ranks(loras)
            guard let device = MTLCreateSystemDefaultDevice() else { throw ProgramError("no Metal device") }
            let offered = try blocking { try await Arithmetic.offered(codes: codes, device) }
            let arithmetic = try Arithmetic.opened(codes: codes, neuralEngine: CoreAI.neuralEngineInForce, offered: offered) {
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

extension Krea2Render.Denoiser: RenderDenoiser {
    /// The velocity itself, at the told time (`Krea2Render.modelTime`): the step is `x += (σᵢ₊₁ − σᵢ) · v`.
    package func derivative(_ latent: [Float], sigma: Float, solver: Solver, first: Bool) throws -> [Float] {
        try velocity(latent, time: Krea2Render.modelTime(sigma: sigma, solver: solver, first: first))
    }
}
