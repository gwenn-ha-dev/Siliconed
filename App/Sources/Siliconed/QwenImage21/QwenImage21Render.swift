import Foundation
import SiliCore

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
///   - **Three sessions, each planned under B**, one opened beside the one before it only where B holds both at their
///     peaks (`RenderLoop`).
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
    /// **Qwen-Image-2.1's part of the render** (`RenderLoop`): the raw template and Qwen3-VL-8B's states, an edit's
    /// references through the vision tower and the VAE's encoder (their tokens to the DiT, no latent to start from), the
    /// DiT told `modelTime`, the decoder reading the latent denormalized in bf16, its RGBA's alpha dropped.
    struct Parts: RenderParts {
        typealias Decoder = QwenImage21VAEDecoder
        private let request: Request, model: Model, admitted: Admission
        private let text: QwenImage21TextModule, denoising: QwenImage21DenoisingModule
        private let references: [ImageRGB], encoding: QwenImage21EncodingModule?
        private let loras: LoRAStack
        let decoderPath: String
        private var grids: [(height: Int, width: Int)] = [], ids: [Int] = [], rows: [Bool] = [], dropped = 0
        private var slots: [Bool] = [], states: HostTensor?, referenceTokens: [Float] = []

        init(_ request: Request, _ admitted: Admission, model: Model) throws {
            let chain = model.chain
            guard let text = chain.text as? QwenImage21TextModule, let denoising = chain.denoising as? QwenImage21DenoisingModule,
                  let decoding = chain.decoding as? QwenImage21DecodingModule else { throw EngineError.notOnEngine(model: model.identifier) }
            references = try denoising.preparedReferences(request.references)
            encoding = chain.encoding as? QwenImage21EncodingModule
            guard references.isEmpty || encoding != nil else { throw EngineError.notOnEngine(model: model.identifier) }
            loras = try LoRAStack(request.loras, on: model.card.family)
            (self.request, self.model, self.admitted, self.text, self.denoising) = (request, model, admitted, text, denoising)
            decoderPath = decoding.path
        }

        var encodes: Bool { !references.isEmpty }

        mutating func tokens() throws -> Int {
            let tokenizer = try Tokenizer(directory: text.tokenizer)
            let factor = denoising.space.factor
            // A latent cell is a vision patch: a reference's grid is both, and it takes a quarter of it in slots.
            grids = references.map { (height: $0.height / factor, width: $0.width / factor) }
            let identified = try QwenImage21Text.identifiers(request.prompt, pictures: grids.map { $0.height * $0.width / 4 },
                                                             tokenizer: tokenizer)
            (ids, rows, dropped) = (identified.ids, identified.slots, identified.dropped)
            return identified.prompt
        }

        mutating func encode(ahead: Stage.Ahead?) throws -> Bool {
            let (ids, rows, dropped, map) = (ids, self.rows, self.dropped, text.map)
            let slots = Array(rows.dropFirst(dropped))
            let pictures = try references.map { try Qwen3VLImages.pixels(QwenImage21References.rgb8($0)) }
            var key = RenderCache.Key("QwenImage21Render.encode")
            try key.file(map)
            key.integers(ids)
            key.integers([dropped])
            for picture in pictures {
                key.integers([picture.gridHeight, picture.gridWidth])
                key.floats(picture.values)
            }
            let (states, cached) = try RenderCache.conditioning(key, of: model) {
                // Without pictures its one stage, the DiT opening beside it (with them the references' stage comes last).
                ahead?.next()
                return try QwenImage21Render.spoken(QwenImage21Render.encode(ids, dropped: dropped, pictures: pictures, slots: rows, map: map), slots: slots)
            }
            (self.slots, self.states) = (slots, states)
            return cached
        }

        /// Each reference's latent, its tokens row-major, kept for the DiT's sequence.
        mutating func image(ahead: Stage.Ahead?) throws -> Latent? {
            guard let encoding else { return nil }
            for (i, reference) in references.enumerated() {
                // The last reference's encoder, the last stage before the DiT: the DiT opens beside it.
                if i == references.count - 1 { ahead?.next() }
                referenceTokens += try QwenImage21Render.tokens(planes: QwenImage21Render.latent(of: reference, path: encoding.path))
            }
            return nil
        }

        func denoiser(height: Int, width: Int) throws -> any RenderDenoiser {
            let sequence = try QwenImage21DiT.Sequence(slots: Array(rows.dropFirst(dropped)), references: grids, height: height, width: width)
            return try Denoiser(map: denoising.map, loras: loras, sequence: sequence)
        }

        func give(_ dit: any RenderDenoiser) throws {
            guard let states, let dit = dit as? Denoiser else { preconditionFailure("the text before Qwen-Image-2.1's DiT") }
            try dit.prompt(text: states, references: referenceTokens.isEmpty ? nil
                           : HostTensor([dit.loop.g.sequence.referenceRows, denoising.space.channels], floats: referenceTokens))
        }

        /// An edit's vision tower is not prepared: it compiles as it loads, as any function the cache does not hold.
        func prepare(height: Int, width: Int) throws -> [Functions.Specialization] {
            let sequence = try QwenImage21DiT.Sequence(slots: Array(rows.dropFirst(dropped)), references: grids, height: height, width: width)
            return try Qwen3Encoder(config: .qwenImage21, path: text.map).prepare(tokens: ids.count)
                + (encoding.map { e in try references.flatMap { try QwenImage21VAEEncoder.prepare(height: $0.height, width: $0.width, path: e.path) } } ?? [])
                + QwenImage21Loop.prepare(map: denoising.map, sequence: sequence, loras: loras)
        }

        func decoderInput(_ latent: [Float]) -> [Float] { QwenImage21VAEDecoder.denormalized(latent) }

        /// The decoded RGBA's alpha dropped without compositing: the picture's RGB planes.
        func picture(_ decoded: [Float]) -> [Float] { Array(decoded.prefix(decoded.count / 4 * 3)) }
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
        try Qwen3Encoder(config: .qwenImage21, path: path).states(ids, taps: taps)
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
        return try Qwen3Encoder(config: .qwenImage21, path: path).states(ids, pictures: read, taps: taps, entries: entries)
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

    /// `_pack_latents`: planes `[C, n]` to tokens `[n, C]`, row-major — the latent's 64 channels by default.
    package static func tokens(planes: [Float], channels: Int = LatentSpace.qwenImage21.channels) -> [Float] {
        let n = planes.count / channels
        var out = [Float](repeating: 0, count: planes.count)
        for t in 0..<n { for c in 0..<channels { out[t * channels + c] = planes[c * n + t] } }
        return out
    }

    /// `_unpack_latents`: tokens `[n, C]` back to planes `[C, n]`.
    package static func planes(tokens: [Float], channels: Int = LatentSpace.qwenImage21.channels) -> [Float] {
        let n = tokens.count / channels
        var out = [Float](repeating: 0, count: tokens.count)
        for t in 0..<n { for c in 0..<channels { out[c * n + t] = tokens[t * channels + c] } }
        return out
    }

    // MARK: - The DiT

    /// **The DiT**, opened once for a batch (`QwenImage21Loop`): its functions loaded, its weights kept under B or read
    /// through a ring, then the prefix — the text, an edit's reference tokens — computed once, each layer's keys and
    /// values kept. Each step then runs the target alone over them. The path is the file's: bf16 weights on the bf16
    /// path, int8 codes in the shared forms under the guard of f16 (int8 × int8 on the GPU, q64 on
    /// the ANE's copy of the codes). A request's LoRAs beside the products they adapt.
    package final class Denoiser {
        let loop: QwenImage21Loop

        package init(map path: String, loras: LoRAStack = .none, sequence: QwenImage21DiT.Sequence) throws {
            nonisolated(unsafe) let loras = loras
            loop = try blocking { UncheckedSendable(try await QwenImage21Loop.open(map: path, sequence: sequence, loras: loras)) }.value
        }

        /// **What it reads once**: `text`, the encoder's text rows, in order (`spoken`); `references`: `[R, 64]`, each
        /// reference's tokens.
        package func prompt(text: HostTensor, references: HostTensor?) throws {
            let (text, references) = (text.floats, references?.floats ?? [])
            nonisolated(unsafe) let loop = loop
            try blocking { try await loop.prompt(text: text, references: references) }
        }

        /// What the DiT is: its arithmetic and its plan under B.
        package var described: String { loop.described }

        /// `model_out` for the latent at the told time (`modelTime`), in f32.
        package func velocity(_ latent: [Float], time: Float) throws -> [Float] {
            nonisolated(unsafe) let loop = loop
            return try blocking { try await loop.evaluate(latent, time: time) }
        }
    }
}

extension QwenImage21Render.Denoiser: RenderDenoiser {
    /// The velocity itself, at the told time (`QwenImage21Render.modelTime`): the step is `x += (σᵢ₊₁ − σᵢ) · v`.
    package func derivative(_ latent: [Float], sigma: Float, solver: Solver, first: Bool) throws -> [Float] {
        try velocity(latent, time: QwenImage21Render.modelTime(sigma: sigma, solver: solver, first: first))
    }

    /// One cycle of its block an evaluation.
    package func expect(evaluations: Int) { loop.block.expect(cycles: evaluations) }
}
