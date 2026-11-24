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
    /// **Z-Image's part of the render** (`RenderLoop`): the chat template and Qwen3-4B's states, Flux's encoder for an
    /// image to start from, the DiT told the inverted time and its velocity negated, Flux's decoder.
    struct Parts: RenderParts {
        typealias Decoder = FluxVAEDecoder
        private let inputs: RenderInputs<ZImageTextModule, ZImageDenoisingModule, FluxEncodingModule, FluxDecodingModule>
        private var ids: [Int] = [], caps: HostTensor?

        init(_ request: Request, _ admitted: Admission, model: Model) throws { inputs = try RenderInputs(request, admitted, model: model) }

        var encodes: Bool { inputs.start != nil }
        var decoderPath: String { inputs.decoding.path }

        mutating func tokens() throws -> Int {
            let (ids, promptTokens) = try Tokenizer(directory: inputs.text.tokenizer).admitChat(user: inputs.request.prompt)
            self.ids = ids
            return promptTokens
        }

        mutating func encode() throws -> Bool {
            let (ids, map) = (ids, inputs.text.map)
            var key = RenderCache.Key("ZImageRender.encode")
            try key.file(map)
            key.integers(ids)
            let (caps, cached) = try RenderCache.conditioning(key, of: inputs.model) { try ZImageRender.encode(ids, map: map) }
            self.caps = caps
            return cached
        }

        func image() throws -> Latent? { try inputs.image { try FluxVAEEncoder.latent(of: $0, path: $1.path) } }

        func denoiser(height: Int, width: Int) throws -> any RenderDenoiser {
            let dit = try Denoiser(map: inputs.denoising.map, loras: inputs.loras, latentHeight: height, latentWidth: width, capLength: ids.count)
            dit.caps = caps
            return dit
        }
    }

    /// **What the DiT is told**: the inverted time `(1000 − σ·1000)/1000` in f32 (`FlowMatchSchedule.modelTime`), or the
    /// integer timestep's under DPM++ and UniPC.
    package static func modelTime(sigma: Float, solver: Solver, first: Bool) -> Float {
        solver.integerTimestep(sigma, first: first).map { Float(1000 - $0) / 1000 } ?? FlowMatchSchedule.modelTime(sigma: sigma)
    }

    // MARK: - The three stages

    /// **The text**: the encoder over the real tokens, `hidden_states[-2]` back as bf16 values, `[L, 2560]`.
    package static func encode(_ ids: [Int], map path: String) throws -> HostTensor {
        guard let caps = try Qwen3Encoder(config: .zImage, path: path).states(ids)["hidden"] else {
            throw ProgramError("the encoder gave no hidden states")
        }
        return caps
    }

    /// **One DiT evaluation**, opened once for a batch: its stage planned under B (`ZImageTransformer.open`), the
    /// caption refined at the first evaluation, the latent and the time's sinusoid given at each. The arithmetic is the
    /// map's and the Neural Engine switch's (`ZImageTransformer.arithmetic`). Let go with it (`Stage.close`).
    package final class Denoiser {
        package var caps: HostTensor?
        let dit: ZImageTransformer
        private var prompted = false

        package init(map path: String, loras: LoRAStack = .none, latentHeight: Int, latentWidth: Int, capLength: Int) throws {
            let file = try ZImageDiT.file(path)
            let geometry = try ZImageTransformer.Geometry(latentHeight: latentHeight, latentWidth: latentWidth, capLength: capLength)
            nonisolated(unsafe) let loras = loras
            dit = try blocking { try await ZImageTransformer.open(file: file, geometry: geometry, loras: loras) }
        }

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

extension ZImageRender.Denoiser: RenderDenoiser {
    /// The pipeline's `noise_pred = −model_out`, at the told time (`ZImageRender.modelTime`).
    package func derivative(_ latent: [Float], sigma: Float, solver: Solver, first: Bool) throws -> [Float] {
        var negated = try velocity(latent, time: ZImageRender.modelTime(sigma: sigma, solver: solver, first: first))
        vDSP.multiply(-1, negated, result: &negated)
        return negated
    }

    /// One cycle of its block an evaluation.
    package func expect(evaluations: Int) { dit.block.expect(cycles: evaluations) }
}
