import Foundation

// **The repository's modules, and the models that assemble them.**
//
// (each name below is a `…Module`)
//                 text               image (img2img)            denoising               decoding
//   Z-Image    ZImageText ─────────── FluxEncoding ──────────── ZImageDenoising ──────── FluxDecoding
//   Anima      AnimaText ──────────── QwenImageEncoding ─┬───── AnimaDenoising ─┬─────── QwenImageDecoding
//   Krea 2     Krea2Text ──────────── QwenImageEncoding ─┘      Krea2Denoising ─┘
//   Qwen 2.1   QwenImage21Text ◀─ refs ── QwenImage21Encoding ── QwenImage21Denoising ── QwenImage21Decoding
//              (the encoder sees the references: one Lanczos copy feeds it and the VAE)
//
// Each module wraps a stage's classes without changing a single computation. Their checks
// (`forward`, `qwen21-*`, `anima-*`, `krea2-*`, …) verify those classes; the close-out verifies the whole chain.

// ── Z-Image ─────────────────────────────────────────────────────────────────────────────────

package struct ZImageTextModule: TextModule {
    package var map: String
    /// The published repository's `tokenizer/` folder.
    package var tokenizer: String
    package var output: TextFormat { .zImage }
    package var name: String { "Qwen3-4B" }

    package init(map: String, tokenizer: String) { self.map = map; self.tokenizer = tokenizer }

    package func encoder(_ prompt: String, context: Context) throws -> Conditioning {
        let begin = Date()
        var identifiers: [Int] = [], realCount = 0
        // **Two scopes, not one**: a 151 k-entry dictionary and a Metal arena do not leave the same
        // residue, and measuring them together makes it impossible to know which to fix.
        do {
            identifiers = try Tokenizer(directory: tokenizer).admitChat(user: prompt)
            realCount = identifiers.count
        }
        context.timings.tokenizer = Date().timeIntervalSince(begin)
        context.footprints.tokenizer = Arena.processFootprint()
        try context.check()
        let encoder = try TextEncoder(artifact: try Artifact(path: map), sequence: realCount,
                                       freezeCut: context.reproducible)
        encoder.cancellation = context.cancellation
        let caps = Array(try encoder.encode(ids: identifiers, realTokens: realCount))
        context.footprints.encoder = Arena.processFootprint()
        return Conditioning(format: .zImage, rows: realCount, width: caps.count / max(1, realCount),
                               values: caps, tokens: realCount)
    }
}

/// Z-Image's S3 DiT under its sampler (negation; every evaluation at full size unless the half-size
/// start is on).
package struct ZImageDenoisingModule: DenoisingModule {
    package var map: String

    package var model: String { "z-image" }
    package var name: String { "DiT Z-Image" }
    package var entry: TextFormat { .zImage }
    package var space: LatentSpace { .flux }
    package var defaultSteps: Int { 8 }
    /// Every `Solver`: its sampler walks them all (`SolverWalk`).
    package var solvers: [Solver] { Solver.allCases }

    package init(map: String) { self.map = map }

    package func sigmas(steps: Int) -> [Float] { FlowMatchSchedule(steps: steps).sigmas }

    /// **By default every evaluation at full size** (since 0.8.2). The former half-size first steps
    /// were judged clean on the seven-evaluation schedule of before 0.8.2; on the recipe's eight
    /// (first full step at σ = 0.9, six full steps after), the high-frequency noise the half-size steps
    /// leave in the full latent (the up-sampled velocity does not carry it) is no longer resorbed —
    /// sand on the skin and grain on the walls at 1024², at `k = 2` and at `k = 1`. Qwen-Image-2.1's
    /// half-size start is this one's recipe (`QwenImage21DenoisingModule.plan`), its `k` judged on its own.
    ///
    /// Evaluations are counted from the start to the end: every step moves (`FlowMatchSchedule`),
    /// hence `N − t_start` evaluations, `⌈N·s⌉` at a strength `s` in img2img.
    ///
    /// **The half-size start** (`Spectral.halfSizeStart`, the app's « Faster Rendering »): the first
    /// three evaluations (σ 1 → 0.833) on the half grid, by `Sampler.init(half:…)` — not the steps
    /// removed in 0.8.2, whose grain it was written against. Only from the first step, at the
    /// recipe's eight, on a grid that halves above the floor (1024², 1024×1536 and its transpose),
    /// without a LoRA or Detail (never judged under either). `SILICONED_SPECTRAL` forces the count (a
    /// measurement), within the same rules.
    package func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan {
        let σ = FlowMatchSchedule(steps: steps).sigmas
        let halvable = Spectral.divisible(height: height, width: width, by: 2)
        let rule = Spectral.halfSizeStart && steps == defaultSteps ? Self.halfSizeSteps : 0
        let k = start > 0 || !halvable || withLoRA ? 0 : min(max(0, Spectral.forced ?? rule), steps - 1)
        return DenoisingPlan(startStep: start, startSigma: σ[start], evaluations: steps - start,
                             reduced: k, reducedHeight: k > 0 ? height / 2 : height,
                             reducedWidth: k > 0 ? width / 2 : width)
    }

    /// Its half-size evaluations: 3, judged clean on two prompts; at 4, fine dots came back on skin.
    package static let halfSizeSteps = 3

    package func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                          context: Context) throws {
        let (height, width) = (latent.height, latent.width)
        let planned = plan(height: height, width: width, steps: steps, start: start,
                           withLoRA: lora != nil || context.detail != .normal)
        let full = { () throws -> DiT in
            if let kept = context.keptDenoiser as? DiT { return kept }
            let dit = try DiT(artifact: try Artifact(path: map), latentHeight: height, latentWidth: width,
                              capLength: text.rows, freezeCut: context.reproducible,
                              neuralEngine: context.neuralEngine, lora: lora)
            dit.cancellation = context.cancellation
            context.keptDenoiser = dit
            return dit
        }
        let sampler: Sampler
        if planned.reduced > 0 {
            // One DiT at a time: a batch's kept full DiT goes before the half-size one is built.
            context.keptDenoiser = nil
            let half = try DiT(artifact: try Artifact(path: map), latentHeight: planned.reducedHeight,
                               latentWidth: planned.reducedWidth, capLength: text.rows,
                               freezeCut: context.reproducible, neuralEngine: context.neuralEngine, lora: lora)
            half.cancellation = context.cancellation
            sampler = Sampler(half: half, halfSteps: planned.reduced, steps: steps, full: full)
        } else {
            let dit = try full()
            dit.cancellation = context.cancellation
            sampler = Sampler(dit: dit, steps: steps)
        }
        sampler.detail = context.detail
        sampler.detailScale = context.detailScale
        sampler.solver = context.solver
        sampler.solverSeed = context.solverSeed
        context.beginDenoising(evaluations: planned.evaluations)
        // `negated` is `dx/dσ` (the DiT's output negated, see `Sampler`): x̂₀ = x − σ·negated.
        let σ = sampler.schedule.sigmas
        sampler.afterEvaluation = { step, seconds, h, w, x, derived, sigmaX in
            try context.stepDone(sigma: σ[step], seconds: seconds, evaluatedHeight: h, evaluatedWidth: w,
                                 space: .flux, x: x, v: derived, nextSigma: sigmaX,
                                 fullHeight: height, fullWidth: width)
        }
        try latent.values.withUnsafeMutableBufferPointer { l in
            try text.values.withUnsafeBufferPointer { caps in
                _ = try sampler.run(latent: l.baseAddress!, count: l.count,
                                            caps: caps.baseAddress!, latentHeight: height,
                                            latentWidth: width, start: start)
            }
        }
        context.footprints.denoising = Arena.processFootprint()   // before the release
    }
}

package struct FluxDecodingModule: DecodingModule {
    package var path: String
    package var space: LatentSpace { .flux }
    package var name: String { "VAE Flux" }
    package init(path: String) { self.path = path }

    package func decode(_ latent: Latent, context: Context) throws -> ImageRGB {
        let vae = try VAE(path: path, latentHeight: latent.height, latentWidth: latent.width)
        let (image, seconds) = try latent.values.withUnsafeBufferPointer { try vae.decode(latent: $0.baseAddress!) }
        context.timings.decoding = seconds
        return ImageRGB(pixels: image, height: vae.imageHeight, width: vae.imageWidth)
    }
}

/// The Flux VAE's encoder (img2img): the image at the render's format → `(mean − 0.1159) × 0.3611`.
package struct FluxEncodingModule: ImageEncodingModule {
    package var path: String
    package var space: LatentSpace { .flux }
    package var name: String { "VAE Flux" }
    package init(path: String) { self.path = path }

    package func encoder(_ image: ImageRGB, context: Context) throws -> Latent {
        try encoderImage(image, path: path, family: .flux, space: space, context: context)
    }
}

/// **A VAE encoder in its own scope**: it is built, encodes, notes its footprint before returning
/// its memory (a peak is read before the release), and dies on exit.
func encoderImage(_ image: ImageRGB, path: String, family: VAEEncoder.Family, space: LatentSpace,
                  context: Context) throws -> Latent {
    let encoder = try VAEEncoder(path: path, family: family, height: image.height, width: image.width)
    let output = try image.pixels.withUnsafeBufferPointer { try encoder.encode(image: $0.baseAddress!) }
    context.footprints.encoding = Arena.processFootprint()
    return Latent(space: space, height: encoder.latentGridHeight, width: encoder.latentGridWidth,
                  values: output.latent)
}

// ── the models: preset chains ────────────────────────────────────────────────────

/// **A model is a preset chain, a name and a license.** Its files are read from a `Library`
/// and checked at construction: a missing file throws here, before a minute of computation, with
/// its full path.
///
/// The constructors and `named` take the library under the same label, `in:` — that of
/// `ModelCard.missing(in:)`. A custom chain becomes a model through
/// `Model(card:chain:)`: it is the only thing `render`, `renderBatch` and `events` take.
package struct Model: Sendable {
    /// The name, the license, the recommended formats: what the catalog knows without opening anything.
    package let card: ModelCard
    package let chain: Chain
    /// The library it was built from — where its renders keep what they do not redo
    /// (`Library.cacheFolder`). `nil` for a hand-composed chain: nothing is cached.
    package let library: Library?

    /// A hand-composed chain, under the sheet of the model whose denoiser it holds.
    package init(card: ModelCard, chain: Chain) { self.card = card; self.chain = chain; self.library = nil }

    package init(card: ModelCard, library: Library, chain: Chain) {
        self.card = card; self.chain = chain; self.library = library
    }

    /// `z-image`, `anima`, `krea2`, `qwen-image-2.1` — what `--model` accepts; an imported model is
    /// called `<family>/<name>` (`z-image/my-model`).
    package var identifier: String { card.id }
    package var name: String { card.name }
    /// The architecture — the target a LoRA declares.
    package var family: Family { card.family }
    /// To be recalled to anyone distributing an image or the model.
    package var license: License { card.license }

    package static let identifiers = ModelCard.allCards.map(\.id)

    package enum Failure: Error, CustomStringConvertible {
        case unknown(String)
        package var description: String {
            switch self {
            case .unknown(let name):
                return "unknown model '\(name)' — \(Model.identifiers.joined(separator: ", ")), or <family>/<name> of an imported model"
            }
        }
    }

    /// The model `identifier` of the library — `EngineError.modelNotInstalled` if a file is missing
    /// (`ModelCard.missing(in:)` says which), `unknownModel` if there is no such model.
    package static func named(_ identifier: String, in b: Library) throws(EngineError) -> Model {
        try building(identifier) { try make(identifier, in: b) }
    }

    /// **The door of a model's construction**: a missing file is a model not installed.
    private static func building(_ identifier: String, _ body: () throws -> Model) throws(EngineError) -> Model {
        do { return try body() } catch is MissingFile {
            throw .modelNotInstalled(model: identifier)
        } catch {
            throw EngineError(error)
        }
    }

    /// `named`, with the internal failure that says which file is missing.
    package static func make(_ identifier: String, in b: Library) throws -> Model {
        switch identifier {
        case "z-image": return try zImage(in: b, card: .zImage, dit: b.ditMap(.zImage))
        case "anima": return try anima(in: b, card: .anima, dit: b.ditMap(.anima))
        case "krea2": return try krea2(in: b, card: .krea2, dit: b.ditMap(.krea2))
        case "qwen-image-2.1": return try qwenImage21(in: b, card: .qwenImage21, dit: b.ditMap(.qwenImage21))
        case "moss-image": return try mossImage(in: b, card: .mossImage, dit: b.ditMap(.mossImage))
        default:
            guard let card = b.importedCards().first(where: { $0.id == identifier }), let dit = card.map else {
                throw Failure.unknown(identifier)
            }
            switch card.family {
            case .zImage: return try zImage(in: b, card: card, dit: dit)
            case .anima: return try anima(in: b, card: card, dit: dit)
            case .krea2: return try krea2(in: b, card: card, dit: dit)
            case .qwenImage21: return try qwenImage21(in: b, card: card, dit: dit)
            case .mossImage: return try mossImage(in: b, card: card, dit: dit)
            }
        }
    }

    /// The library's `store/<name>` map, if it is there (`MissingFile` otherwise, folded into
    /// `modelNotInstalled` by `building`).
    static func map(_ name: String, _ b: Library) throws -> String {
        let path = name.hasPrefix("/") ? name : b.map(name)
        guard FileManager.default.fileExists(atPath: path) else { throw MissingFile(path, .map) }
        return path
    }

    /// Z-Image Turbo — the product's default. `denoising`: a denoiser on another map.
    package static func zImage(in b: Library, denoising: ZImageDenoisingModule? = nil) throws(EngineError) -> Model {
        try building(ModelCard.zImage.id) { try zImage(in: b, card: .zImage, dit: b.ditMap(.zImage), denoising: denoising) }
    }

    static func zImage(in b: Library, card: ModelCard, dit: String,
                       denoising: ZImageDenoisingModule? = nil) throws -> Model {
        Model(card: card, library: b, chain: try Chain(
                   text: ZImageTextModule(map: try map(b.encoderMap(.zImage), b),
                                      tokenizer: try b.component(.zImage, "tokenizer")),
                   denoising: denoising ?? ZImageDenoisingModule(map: try map(dit, b)),
                   decoding: FluxDecodingModule(path: try b.component(.zImage, "vae.safetensors")),
                   encoding: FluxEncodingModule(path: try b.component(.zImage, "vae.safetensors"))))
    }

    /// Anima Turbo v1.1. ⚠️ Non-commercial license: never a product's default.
    package static func anima(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.anima.id) { try anima(in: b, card: .anima, dit: b.ditMap(.anima)) }
    }

    /// Anima's text adapter comes from the same file as the DiT: it lives next to its map.
    static func anima(in b: Library, card: ModelCard, dit: String) throws -> Model {
        let dit = try map(dit, b)
        let adapter = adapterPath(fromMap: dit)
        guard FileManager.default.fileExists(atPath: adapter) else { throw MissingFile(adapter, .map) }
        return Model(card: card, library: b, chain: try Chain(
                   text: AnimaTextModule(map: try map(b.encoderMap(.anima), b),
                                     tokenizerQwen: try b.component(.anima, "tokenizer"),
                                     tokenizerT5: try b.component(.anima, "t5_tokenizer"),
                                     adapter: adapter),
                   denoising: AnimaDenoisingModule(map: dit),
                   decoding: QwenImageDecodingModule(path: try b.component(.anima, "vae.safetensors")),
                   encoding: QwenImageEncodingModule(path: try b.component(.anima, "vae.safetensors"))))
    }

    /// Krea 2 Turbo. ⚖️ Krea 2 Community license: commercial under $1M annual revenue, mandatory
    /// content filter at deployment (§4.2), "Krea" at the head of a distributed model (§3.1).
    package static func krea2(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.krea2.id) { try krea2(in: b, card: .krea2, dit: b.ditMap(.krea2)) }
    }

    static func krea2(in b: Library, card: ModelCard, dit: String) throws -> Model {
        Model(card: card, library: b, chain: try Chain(
                   text: Krea2TextModule(map: try map(b.encoderMap(.krea2), b),
                                     tokenizer: try b.component(.krea2, "tokenizer"),
                                     sockets: Krea2Text.sockets),
                   denoising: Krea2DenoisingModule(map: try map(dit, b)),
                   decoding: QwenImageDecodingModule(path: try b.component(.krea2, "vae.safetensors")),
                   encoding: QwenImageEncodingModule(path: try b.component(.krea2, "vae.safetensors"))))
    }

    /// Qwen-Image-2.1-Turbo: 8 steps, generation and editing by instruction (1 to 3
    /// references, which its encoder sees). ⚠️ Qwen Research license: non-commercial.
    package static func qwenImage21(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.qwenImage21.id) { try qwenImage21(in: b, card: .qwenImage21, dit: b.ditMap(.qwenImage21)) }
    }

    static func qwenImage21(in b: Library, card: ModelCard, dit: String) throws -> Model {
        let vae = try b.component(.qwenImage21, "vae.safetensors")
        return Model(card: card, library: b, chain: try Chain(
                   text: QwenImage21TextModule(map: try map(b.encoderMap(.qwenImage21), b),
                                               tokenizer: try b.component(.qwenImage21, "processor")),
                   denoising: try QwenImage21DenoisingModule.product(map: try map(dit, b), in: b),
                   decoding: QwenImage21DecodingModule(path: vae),
                   encoding: QwenImage21EncodingModule(path: vae)))
    }

    /// MOSS-Image under its 8-step LoRA. Apache 2.0.
    package static func mossImage(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.mossImage.id) { try mossImage(in: b, card: .mossImage, dit: b.ditMap(.mossImage)) }
    }

    static func mossImage(in b: Library, card: ModelCard, dit: String) throws -> Model {
        // The LoRA lives with the components; without it the DiT would render an undistilled image in 8 steps.
        guard let turbo = Installer(library: b).companionMap(.mossImage) else { throw MissingFile("turbo LoRA", .map) }
        guard FileManager.default.fileExists(atPath: turbo) else { throw MissingFile(turbo, .map) }
        return Model(card: card, library: b, chain: try Chain(
            text: MossImageTextModule(map: try map(b.encoderMap(.mossImage), b), tokenizer: try b.component(.mossImage, "tokenizer")),
            denoising: MossImageDenoisingModule(map: try map(dit, b), turbo: turbo),
            decoding: Flux2DecodingModule(path: try b.component(.mossImage, "vae.safetensors"))))
    }
}
