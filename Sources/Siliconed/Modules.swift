import Foundation

// **The repository's modules, and the models that assemble them.**
//
// (each name below is a `…Module`)
//                 text               image (img2img)            denoising               decoding
//   Z-Image    ZImageText ─────────── FluxEncoding ──────────── ZImageDenoising ──────── FluxDecoding
//   Anima      AnimaText ──────────── QwenImageEncoding ─┬───── AnimaDenoising ─┬─────── QwenImageDecoding
//   Krea 2     Krea2Text ──────────── QwenImageEncoding ─┘      Krea2Denoising ─┘
//   Klein 4B   KleinText ──────────── Flux2Encoding ─────────── KleinDenoising ───────── Flux2Decoding
//   ERNIE      ErnieText ──────────── Flux2Encoding (eps 1e-5) ─ ErnieDenoising ──────── Flux2Decoding (eps 1e-5)
//   Qwen 2.1   QwenImage21Text ◀─ refs ── QwenImage21Encoding ── QwenImage21Denoising ── QwenImage21Decoding
//              (the encoder sees the references: one Lanczos copy feeds it and the VAE)
//
// Each module wraps a stage's classes without changing a single computation. Their checks
// (`forward`, `qwen21-*`, `anima-*`, `krea2-*`, …) verify those classes; the close-out verifies the whole chain.

// ── Z-Image ─────────────────────────────────────────────────────────────────────────────────

public struct ZImageTextModule: TextModule {
    public var map: String
    /// The published repository's `tokenizer/` folder.
    public var tokenizer: String
    public var output: TextFormat { .zImage }
    public var name: String { "Qwen3-4B" }

    public init(map: String, tokenizer: String) { self.map = map; self.tokenizer = tokenizer }

    public func encoder(_ prompt: String, context: Context) throws -> Conditioning {
        let begin = Date()
        var identifiers: [Int] = [], realCount = 0
        // **Two scopes, not one**: a 151 k-entry dictionary and a Metal arena do not leave the same
        // residue, and measuring them together makes it impossible to know which to fix.
        do {
            let (ids, mask) = try Tokenizer(directory: tokenizer).encodeChat(user: prompt)
            identifiers = ids
            realCount = mask.reduce(0, +)
        }
        context.timings.tokenizer = Date().timeIntervalSince(begin)
        Arena.releaseToSystem()
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

/// Z-Image's S3 DiT under its sampler (negation, spectral schedule, null steps skipped).
public struct ZImageDenoisingModule: DenoisingModule {
    public var map: String
    /// The number of half-size evaluations. `nil`: `SILICONED_SPECTRAL` if set (read at render
    /// time, by `EngineSettings`), otherwise the product's rule (`Spectral.steps`).
    public var spectral: Int?
    /// The per-step side divisor. Empty: derived from `spectral`.
    public var divisors: [Int]
    /// The caption positions to erase — a measuring instrument. See `DiT.capsDrop`.
    public var capsDrop: [Int]

    public var model: String { "z-image" }
    public var name: String { "DiT Z-Image" }
    public var entry: TextFormat { .zImage }
    public var space: LatentSpace { .flux }
    public var defaultSteps: Int { 8 }

    public init(map: String, spectral: Int? = nil,
                divisors: [Int] = [], capsDrop: [Int] = []) {
        self.map = map; self.spectral = spectral; self.divisors = divisors
        self.capsDrop = capsDrop
    }

    /// **The product's denoiser** — the defaults, plus what the environment forces.
    ///
    /// It is here, and nowhere lower down, that `SILICONED_DIVISORS` and `SILICONED_CAPS_DROP` are
    /// read: the environment keeps the last word because a measurement is taken by forcing a
    /// setting, but it says so **at the edge**. `SILICONED_SPECTRAL`, for its part, goes through
    /// `EngineSettings` and is read at render time (`plan`): building a model must not freeze the
    /// machine's profile.
    static func product(map: String) -> ZImageDenoisingModule {
        let env = ProcessInfo.processInfo.environment
        return ZImageDenoisingModule(
            map: map,
            divisors: env["SILICONED_DIVISORS"]?.split(separator: ",").compactMap { Int($0) } ?? [],
            capsDrop: env["SILICONED_CAPS_DROP"]?.split(separator: ",").compactMap { Int($0) } ?? [])
    }

    public func sigmas(steps: Int) -> [Float] { FlowMatchSchedule(steps: steps).sigmas }

    /// `SILICONED_SPECTRAL`, via `EngineSettings` — read at render time, never at construction.
    private var forcedSpectral: Int? {
        EngineSettings.effective.provenance["spectral"] != nil ? EngineSettings.effective.spectral : nil
    }

    /// **In img2img (`start > 0`), the spectral schedule is off: `k = 0`**, even when forced.
    ///
    /// `Spectral.steps` counts the **leading** σs of the full schedule, calibrated on pure-noise
    /// starts; truncated, it would reduce steps that no longer run. Related to the σs
    /// actually evaluated, it would give `k = 1` only at start 1 (σ = 0.947) and `k = 0` from
    /// start 2 on (σ = 0.882, `f* = 0.134 > 0.10`) — nothing to gain at useful strengths
    /// (≤ 0.75), and a never-measured regime: the state already carries the image's high
    /// frequencies. At `k = 0`, the trajectory reproduces the oracle. Forced divisors turn
    /// off for the same reason.
    ///
    /// Evaluations are counted from the start to the end, **null step skipped**: the last step
    /// (σ = 0 → 0) is always in the kept tail, hence `⌈N·s⌉ − 1` evaluations.
    ///
    /// **Under a LoRA, the product's rule returns `k = 0`**; a set or forced `k` keeps the last
    /// word, so one can still measure. The half-size steps leave the full latent's high-frequency
    /// noise intact (the up-sampled velocity does not carry it): ~13 % too much at σ = 0.88,
    /// which the full steps must resorb. `k = 2` was judged clean without a LoRA;
    /// under a LoRA, nothing has judged it, and grain and blotches were seen at 1024². **Detail goes the same way** (the engine passes `withLoRA` for it too): it tells
    /// those full steps that less noise remains than does, and the excess stays — sand on walls and
    /// skin at 1024² (measured: amount 1 at k = 2, confetti; at k = 0, +19 %, clean).
    public func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan {
        let schedule = FlowMatchSchedule(steps: steps)
        let σ = schedule.sigmas
        let rule = withLoRA ? 0 : Spectral.steps(height: height, width: width, schedule: schedule)
        return DenoisingPlan(startStep: start, startSigma: σ[start],
                                evaluations: (start..<steps).filter { σ[$0 + 1] != σ[$0] }.count,
                                reduced: start > 0 ? 0 : spectral ?? forcedSpectral ?? rule,
                                reducedHeight: height / 2, reducedWidth: width / 2)
    }

    public func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                          context: Context) throws {
        let (height, width) = (latent.height, latent.width)
        let dit: DiT
        if let kept = context.keptDenoiser as? DiT {
            dit = kept
        } else {
            dit = try DiT(artifact: try Artifact(path: map), latentHeight: height, latentWidth: width,
                          capLength: text.rows, freezeCut: context.reproducible, lora: lora)
            context.keptDenoiser = dit
        }
        dit.capsDrop = capsDrop
        dit.cancellation = context.cancellation
        let sampler = Sampler(dit: dit, steps: steps)
        let planned = plan(height: height, width: width, steps: steps, start: start,
                           withLoRA: lora != nil || context.detail != .normal)
        sampler.spectralSteps = planned.reduced
        sampler.divisors = start > 0 ? [] : divisors
        sampler.detail = context.detail
        sampler.detailScale = context.detailScale
        context.beginDenoising(evaluations: planned.evaluations)
        // `negated` is `dx/dσ` (the DiT's output negated, see `Sampler`): x̂₀ = x − σ·negated.
        let σ = sampler.schedule.sigmas
        sampler.afterEvaluation = { step, seconds, h, w, x, derived in
            try context.stepDone(sigma: σ[step], seconds: seconds, evaluatedHeight: h, evaluatedWidth: w,
                                 space: .flux, x: x, v: derived, nextSigma: σ[step + 1],
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

public struct FluxDecodingModule: DecodingModule {
    public var path: String
    public var space: LatentSpace { .flux }
    public var name: String { "VAE Flux" }
    public init(path: String) { self.path = path }

    public func decode(_ latent: Latent, context: Context) throws -> ImageRGB {
        let vae = try VAE(path: path, latentHeight: latent.height, latentWidth: latent.width)
        let (image, seconds) = try latent.values.withUnsafeBufferPointer { try vae.decode(latent: $0.baseAddress!) }
        context.timings.decoding = seconds
        return ImageRGB(pixels: image, height: vae.imageHeight, width: vae.imageWidth)
    }
}

/// The Flux VAE's encoder (img2img): the image at the render's format → `(mean − 0.1159) × 0.3611`.
public struct FluxEncodingModule: ImageEncodingModule {
    public var path: String
    public var space: LatentSpace { .flux }
    public var name: String { "VAE Flux" }
    public init(path: String) { self.path = path }

    public func encoder(_ image: ImageRGB, context: Context) throws -> Latent {
        try encoderImage(image, path: path, family: .flux, space: space, context: context)
    }
}

/// **A VAE encoder in its own scope**: it is built, encodes, notes its footprint before returning
/// its memory (a peak is read before the release), and dies on exit.
func encoderImage(_ image: ImageRGB, path: String, family: VAEEncoder.Family, space: LatentSpace,
                  context: Context) throws -> Latent {
    let encoder = try VAEEncoder(path: path, family: family, height: image.height, width: image.width)
    let output = image.pixels.withUnsafeBufferPointer { encoder.encode(image: $0.baseAddress!) }
    context.footprints.encoding = Arena.processFootprint()
    return Latent(space: space, height: encoder.latentGridHeight, width: encoder.latentGridWidth,
                  values: output.latent)
}

// ── Anima ───────────────────────────────────────────────────────────────────────────────────

/// Two tokenizers, Qwen3-0.6B (`last_hidden_state`), then Anima's conditioner (T5 + adapter).
public struct AnimaTextModule: TextModule {
    public var map: String
    public var tokenizerQwen: String
    public var tokenizerT5: String
    /// The published Turbo file, which carries the adapter's weights.
    public var adapter: String
    public var output: TextFormat { .anima }
    public var name: String { "Qwen3-0.6B + T5" }

    public init(map: String, tokenizerQwen: String, tokenizerT5: String, adapter: String) {
        self.map = map; self.tokenizerQwen = tokenizerQwen; self.tokenizerT5 = tokenizerT5
        self.adapter = adapter
    }

    public func encoder(_ prompt: String, context: Context) throws -> Conditioning {
        let begin = Date()
        // `padding="longest", max_length=512, truncation=True` for a single prompt: no padding,
        // and Qwen truncated to 512 — T5 keeps its `</s>` under truncation.
        let qwenIds = Array(try Tokenizer(directory: tokenizerQwen).encode(prompt).prefix(512))
        let t5Ids = try T5Tokenizer(directory: tokenizerT5).encode(prompt)
        // The engine already refuses a blank prompt; this guards a direct call of the module.
        guard !qwenIds.isEmpty else { throw Request.Failure.emptyPrompt }
        context.timings.tokenizer = Date().timeIntervalSince(begin)
        context.footprints.tokenizer = Arena.processFootprint()
        try context.check()
        let encoder = try TextEncoder(artifact: try Artifact(path: map), sequence: qwenIds.count,
                                       freezeCut: context.reproducible)
        encoder.cancellation = context.cancellation
        let hidden = try encoder.encode(ids: qwenIds, realTokens: qwenIds.count)
        context.footprints.encoder = Arena.processFootprint()
        let values = try AnimaConditioner(path: adapter)
            .condition(t5: t5Ids, qwen: hidden, qwenRows: qwenIds.count)
        return Conditioning(format: .anima, rows: values.count / 1024, width: 1024,
                               values: values, tokens: qwenIds.count)
    }
}

/// Anima Turbo's Cosmos DiT: Euler, CFG 1.
public struct AnimaDenoisingModule: DenoisingModule {
    public var map: String
    public var model: String { "anima" }
    public var name: String { "DiT Anima" }
    public var entry: TextFormat { .anima }
    public var space: LatentSpace { .qwenImage }
    public var defaultSteps: Int { 8 }

    public init(map: String) { self.map = map }

    public func sigmas(steps: Int) -> [Float] { AnimaDiT.sigmas(steps: steps) }

    public func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan {
        DenoisingPlan(startStep: start, startSigma: sigmas(steps: steps)[start], evaluations: steps - start,
                         reduced: 0, reducedHeight: height, reducedWidth: width)
    }

    public func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                          context: Context) throws {
        let dit = try AnimaDiT(artifact: try Artifact(path: map), latentHeight: latent.height,
                               latentWidth: latent.width,
                               textLength: text.rows, freezeCut: context.reproducible, lora: lora)
        dit.cancellation = context.cancellation
        try euler(&latent, sigmas: sigmas(steps: steps), start: start, context: context) { x, sigma in
            try text.values.withUnsafeBufferPointer {
                try dit.forward(latent: x, context: $0.baseAddress!, sigma: sigma)
            }
        }
        context.footprints.denoising = Arena.processFootprint()
    }
}

/// The Qwen-Image VAE's encoder (img2img), for Anima and Krea 2 — same weights, two files: the
/// image at the render's format → `(mean − mean_c) / std_c`.
public struct QwenImageEncodingModule: ImageEncodingModule {
    public var path: String
    public var space: LatentSpace { .qwenImage }
    public var name: String { "VAE Qwen-Image" }
    public init(path: String) { self.path = path }

    public func encoder(_ image: ImageRGB, context: Context) throws -> Latent {
        try encoderImage(image, path: path, family: .qwenImage, space: space, context: context)
    }
}

public struct QwenImageDecodingModule: DecodingModule {
    public var path: String
    public var space: LatentSpace { .qwenImage }
    public var name: String { "VAE Qwen-Image" }
    public init(path: String) { self.path = path }

    public func decode(_ latent: Latent, context: Context) throws -> ImageRGB {
        let vae = try AnimaVAE(path: path, latentHeight: latent.height, latentWidth: latent.width)
        let (image, seconds) = latent.values.withUnsafeBufferPointer { vae.decode(latent: $0.baseAddress!) }
        context.timings.decoding = seconds
        return ImageRGB(pixels: image, height: vae.imageHeight, width: vae.imageWidth)
    }
}

/// **Euler in flow matching, without negation**: `x += (σᵢ₊₁ − σᵢ) · v`, one evaluation per step.
/// The step of Anima and Krea 2; Z-Image has its own (`Sampler`), which negates the velocity and
/// skips null steps. `start`: the first step executed of the full schedule (img2img). `v` is
/// `dx/dσ` as is, hence x̂₀ = x − σ·v for the preview. Cancellation is checked before each step,
/// and between layers in the DiT.
func euler(_ latent: inout Latent, sigmas: [Float], start: Int = 0, context: Context,
           speed: (UnsafePointer<Float>, Float) throws -> [Float]) throws {
    context.beginDenoising(evaluations: sigmas.count - 1 - start)
    // What the DiT is told (`Detail`); the step below keeps the true σ.
    let told = context.told(sigmas, start: start)
    for i in start..<(sigmas.count - 1) {
        try context.check()
        let tick = Date()
        let v = try latent.values.withUnsafeBufferPointer { try speed($0.baseAddress!, told[i]) }
        let dt = sigmas[i + 1] - sigmas[i]
        latent.values.withUnsafeMutableBufferPointer { x in
            for j in 0..<x.count { x[j] += dt * v[j] }
        }
        let seconds = Date().timeIntervalSince(tick)
        try latent.values.withUnsafeBufferPointer { x in
            try v.withUnsafeBufferPointer { v in
                try context.stepDone(sigma: sigmas[i], seconds: seconds, evaluatedHeight: latent.height,
                                     evaluatedWidth: latent.width, space: latent.space, x: x.baseAddress!,
                                     v: v.baseAddress!, nextSigma: sigmas[i + 1],
                                     fullHeight: latent.height, fullWidth: latent.width)
            }
        }
    }
}

// ── Krea 2 ──────────────────────────────────────────────────────────────────────────────────

/// Template, Qwen tokenizer, Qwen3-VL-4B with its twelve layers taken. Not the fusion: that is the DiT's.
public struct Krea2TextModule: TextModule {
    public var map: String
    public var tokenizer: String
    /// The layers taken, read from `model_index.json`.
    public var sockets: [Int]
    public var output: TextFormat { .krea2 }
    public var name: String { "Qwen3-VL-4B" }

    public init(map: String, tokenizer: String, sockets: [Int]) {
        self.map = map; self.tokenizer = tokenizer; self.sockets = sockets
    }

    public func encoder(_ prompt: String, context: Context) throws -> Conditioning {
        let begin = Date()
        let ids = Krea2Text.identifiers(prompt, tokenizer: try Tokenizer(directory: tokenizer))
        let tokens = ids.count - Krea2Text.templateTokens
        context.timings.tokenizer = Date().timeIntervalSince(begin)
        context.footprints.tokenizer = Arena.processFootprint()
        try context.check()
        let hiddenValues = try Krea2Text.hiddenStates(ids, encoder: map, sockets: sockets,
                                                freezeCut: context.reproducible,
                                                cancellation: context.cancellation)
        context.footprints.encoder = Arena.processFootprint()
        return Conditioning(format: .krea2, rows: tokens, width: hiddenValues.count / max(1, tokens),
                               values: hiddenValues, tokens: tokens)
    }
}

/// The text fusion and `txt_in`, then the Krea 2 DiT: Euler, CFG 0. Both read the same map and the
/// LoRA touches both — which is why the fusion lives here and not in the text stage.
public struct Krea2DenoisingModule: DenoisingModule {
    public var map: String
    public var model: String { "krea2" }
    public var name: String { "DiT Krea 2" }
    public var entry: TextFormat { .krea2 }
    public var space: LatentSpace { .qwenImage }
    public var defaultSteps: Int { 8 }

    public init(map: String) { self.map = map }

    public func sigmas(steps: Int) -> [Float] { Krea2DiT.sigmas(steps: steps) }

    public func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan {
        DenoisingPlan(startStep: start, startSigma: sigmas(steps: steps)[start], evaluations: steps - start,
                         reduced: 0, reducedHeight: height, reducedWidth: width)
    }

    public func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                          context: Context) throws {
        let artifact = try Artifact(path: map)
        let tokens = text.rows
        let merged = try fusion(text, artifact: artifact, lora: lora, context: context)
        Arena.releaseToSystem()
        let dit = try Krea2DiT(artifact: artifact, latentHeight: latent.height,
                               latentWidth: latent.width, maxText: tokens,
                               freezeCut: context.reproducible, lora: lora)
        dit.cancellation = context.cancellation
        try context.check()
        // The modulation of only the σs evaluated — from `start` to the last non-null one —, in
        // one pass. It depends only on σ: the following images of a batch reuse the first's table.
        let σ = sigmas(steps: steps)
        let modulationKey = "krea2.modulation " + map
        if let table = context.reserve[modulationKey] as? Krea2DiT.ModulationTable { dit.modulations = table }
        try dit.tabulateModulation(sigmas: Array(context.told(σ, start: start)[start..<(σ.count - 1)]))
        context.reserve[modulationKey] = dit.modulations
        try euler(&latent, sigmas: σ, start: start, context: context) { x, sigma in
            try merged.withUnsafeBufferPointer {
                try dit.forward(latent: x, text: $0.baseAddress!, textRows: tokens, sigma: sigma)
            }
        }
        context.footprints.denoising = Arena.processFootprint()
    }

    /// A completed fusion, and what determined it.
    private final class FusionDone {
        let text: [Float], lora: LoRA?, merged: [Float]
        init(text: [Float], lora: LoRA?, merged: [Float]) { self.text = text; self.lora = lora; self.merged = merged }
    }

    /// **The text fusion, once per batch**: it depends only on the text and the LoRA, not on the
    /// seed (1.5 to 2 s per image at 512², measured). It dies on exit, before the
    /// DiT asks for its arenas; only its result — tokens × 6,144 — stays in the context.
    private func fusion(_ text: Conditioning, artifact: Artifact, lora: LoRA?,
                        context: Context) throws -> [Float] {
        let key = "krea2.fusion " + map
        if let mergeDone = context.reserve[key] as? FusionDone, mergeDone.lora === lora, mergeDone.text == text.values {
            return mergeDone.merged
        }
        let fusion = try Krea2TextFusion(artifact: artifact, maxTokens: text.rows,
                                         freezeCut: context.reproducible, lora: lora)
        fusion.cancellation = context.cancellation
        let merged = try text.values.withUnsafeBufferPointer {
            try fusion.fuse($0.baseAddress!, tokens: text.rows)
        }
        context.reserve[key] = FusionDone(text: text.values, lora: lora, merged: merged)
        return merged
    }
}

// ── FLUX.2 [klein] ──────────────────────────────────────────────────────────────────────────

/// Chat template, Qwen tokenizer padded to 512, Qwen3-4B with its three layers taken — Z-Image's
/// map, bit for bit. The padding is computed (`KleinText`): the DiT receives it without a mask.
public struct KleinTextModule: TextModule {
    public var map: String
    public var tokenizer: String
    public var output: TextFormat { .klein }
    public var name: String { "Qwen3-4B" }

    public init(map: String, tokenizer: String) { self.map = map; self.tokenizer = tokenizer }

    public func encoder(_ prompt: String, context: Context) throws -> Conditioning {
        let begin = Date()
        let (ids, realCount) = KleinText.identifiers(prompt, tokenizer: try Tokenizer(directory: tokenizer))
        context.timings.tokenizer = Date().timeIntervalSince(begin)
        Arena.releaseToSystem()
        context.footprints.tokenizer = Arena.processFootprint()
        try context.check()
        let hiddenValues = try KleinText.hiddenStates(ids, realCount: realCount, encoder: map,
                                                freezeCut: context.reproducible, cancellation: context.cancellation)
        context.footprints.encoder = Arena.processFootprint()
        return Conditioning(format: .klein, rows: ids.count, width: hiddenValues.count / ids.count,
                               values: hiddenValues, tokens: realCount)
    }
}

/// The FLUX.2 [klein] DiT: Euler, CFG 1 (distilled), schedule shifted according to the grid.
public struct KleinDenoisingModule: DenoisingModule {
    public var map: String
    public var family: Family
    public var model: String { family.rawValue }
    public var name: String { "DiT FLUX.2 [klein]" }
    public var entry: TextFormat { .klein }
    public var space: LatentSpace { .flux2 }
    /// The model's card: `num_inference_steps=4`.
    public var defaultSteps: Int { 4 }

    public init(map: String, family: Family = .klein4b) { self.map = map; self.family = family }

    /// Outside a grid, that of 1024² (4,096 tokens) — the reference format. A render always takes
    /// that of its grid (`sigmas(step:height:width:)`).
    public func sigmas(steps: Int) -> [Float] { KleinDiT.sigmas(steps: steps, imageTokens: 4096) }

    public func sigmas(steps: Int, height: Int, width: Int) -> [Float] {
        KleinDiT.sigmas(steps: steps, imageTokens: height * width)
    }

    public func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan {
        DenoisingPlan(startStep: start, startSigma: sigmas(steps: steps, height: height, width: width)[start],
                         evaluations: steps - start, reduced: 0, reducedHeight: height, reducedWidth: width)
    }

    /// The multi-reference editing of `Flux2KleinPipeline`: four, as ComfyUI offers it — the
    /// sequence grows by that many images, and the time with it.
    public var maxReferences: Int { 4 }

    public func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                          context: Context) throws {
        try denoise(&latent, text: text, steps: steps, start: start, references: [], lora: lora, context: context)
    }

    public func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int,
                          references: [Latent], lora: LoRA?, context: Context) throws {
        guard references.allSatisfy({ $0.space == space }), references.count <= maxReferences else {
            throw Request.Failure.references(references.count, max: maxReferences)
        }
        let dit = try KleinDiT(artifact: try Artifact(path: map), latentHeight: latent.height,
                               latentWidth: latent.width, references: references.map { ($0.height, $0.width) },
                               textRows: text.rows, freezeCut: context.reproducible, lora: lora)
        dit.cancellation = context.cancellation
        try text.values.withUnsafeBufferPointer { try dit.prepareText($0.baseAddress!) }
        try dit.prepareReferences(references.map(\.values))
        try context.check()
        // The modulation of only the σs evaluated, in one pass; the following images of a batch reuse it.
        let σ = sigmas(steps: steps, height: latent.height, width: latent.width)
        let key = "klein.modulation " + map
        if let table = context.reserve[key] as? KleinDiT.ModulationTable { dit.modulations = table }
        try dit.tabulateModulation(sigmas: Array(context.told(σ, start: start)[start..<(σ.count - 1)]))
        context.reserve[key] = dit.modulations
        try euler(&latent, sigmas: σ, start: start, context: context) { x, sigma in
            try dit.forward(latent: x, sigma: sigma)
        }
        context.footprints.denoising = Arena.processFootprint()
    }
}

/// The FLUX.2 VAE's decoder: the latent's `BatchNorm` undone, the 2×2 cells unpacked, then Flux's
/// 32-channel decoder (`VAE.Config.flux2`).
public struct Flux2DecodingModule: DecodingModule {
    public var path: String
    public var space: LatentSpace { .flux2 }
    public var name: String { "VAE FLUX.2" }
    /// The `BatchNorm`'s eps: `batch_norm_eps` from the published `vae/config.json` (1e-4) for
    /// FLUX.2, **1e-5 hard-coded** in `ErnieImagePipeline` — same VAE, bit for bit, other pipeline.
    public var eps: Float
    public init(path: String, eps: Float = 1e-4) { self.path = path; self.eps = eps }

    public func decode(_ latent: Latent, context: Context) throws -> ImageRGB {
        let z = try Flux2DecodingModule.unpack(latent, path: path, eps: eps)
        let vae = try VAE(path: path, latentHeight: 2 * latent.height, latentWidth: 2 * latent.width,
                          config: .flux2)
        let (image, seconds) = try z.withUnsafeBufferPointer { try vae.decode(latent: $0.baseAddress!) }
        context.timings.decoding = seconds
        return ImageRGB(pixels: image, height: vae.imageHeight, width: vae.imageWidth)
    }

    /// The VAE's `BatchNorm` statistics: `(mean, √(var + eps))` per packed channel.
    static func normPerBatch(_ path: String, eps: Float) throws -> (mean: [Float], deviation: [Float]) {
        let weights = try Safetensors(path: path)
        guard let mean = weights.materialize("bn.running_mean"), let variance = weights.materialize("bn.running_var"),
              mean.count == LatentSpace.flux2.channels, variance.count == LatentSpace.flux2.channels else {
            throw Safetensors.Failure.missingTensor(file: path, name: "bn.running_mean / bn.running_var")
        }
        return (mean, variance.map { ($0 + eps).squareRoot() })
    }

    /// `[128, h, w]` → `x · √(var + eps) + mean`, then `_unpatchify_latents` → `[32, 2h, 2w]`:
    /// channel `c·4 + ph·2 + pw` goes to `(c, 2y + ph, 2x + pw)`.
    package static func unpack(_ latent: Latent, path: String, eps: Float = 1e-4) throws -> [Float] {
        let (mean, deviations) = try normPerBatch(path, eps: eps)
        let (h, w) = (latent.height, latent.width), channels = latent.space.channels / 4
        var z = [Float](repeating: 0, count: latent.values.count)
        for c in 0..<channels {
            for ph in 0..<2 {
                for pw in 0..<2 {
                    let k = c * 4 + ph * 2 + pw
                    let deviation = deviations[k], m = mean[k]
                    for y in 0..<h {
                        for x in 0..<w {
                            z[(c * 2 * h + 2 * y + ph) * 2 * w + 2 * x + pw] = latent.values[(k * h + y) * w + x] * deviation + m
                        }
                    }
                }
            }
        }
        return z
    }
}

/// The FLUX.2 VAE's encoder (img2img): the mean `[32, H/8, W/8]`, packed 2×2
/// (`_patchify_latents`) then normalized by the `BatchNorm` — the exact inverse of `Flux2DecodingModule`.
public struct Flux2EncodingModule: ImageEncodingModule {
    public var path: String
    public var space: LatentSpace { .flux2 }
    public var name: String { "VAE FLUX.2" }
    /// That of the decoder it inverts (`Flux2DecodingModule.eps`).
    public var eps: Float
    public init(path: String, eps: Float = 1e-4) { self.path = path; self.eps = eps }

    public func encoder(_ image: ImageRGB, context: Context) throws -> Latent {
        let raw = try encoderImage(image, path: path, family: .flux2, space: .flux2, context: context)
        let (mean, deviations) = try Flux2DecodingModule.normPerBatch(path, eps: eps)
        let (h, w) = (raw.height / 2, raw.width / 2)
        var z = [Float](repeating: 0, count: raw.values.count)
        for c in 0..<(space.channels / 4) {
            for ph in 0..<2 {
                for pw in 0..<2 {
                    let k = c * 4 + ph * 2 + pw
                    for y in 0..<h {
                        for x in 0..<w {
                            z[(k * h + y) * w + x] = (raw.values[(c * 2 * h + 2 * y + ph) * 2 * w + 2 * x + pw] - mean[k]) / deviations[k]
                        }
                    }
                }
            }
        }
        return Latent(space: .flux2, height: h, width: w, values: z)
    }
}

// ── ERNIE-Image ────────────────────────────────────────────────────────────────────────────

/// Mistral tokenizer (`<s>` in front, **no template**), Ministral-3 up to `hidden_states[-2]`:
/// real tokens only, `[T, 3072]` — the pipeline does not pad.
public struct ErnieTextModule: TextModule {
    public var map: String
    public var tokenizer: String
    public var output: TextFormat { .ernie }
    public var name: String { "Ministral-3 3B" }

    public init(map: String, tokenizer: String) { self.map = map; self.tokenizer = tokenizer }

    public func encoder(_ prompt: String, context: Context) throws -> Conditioning {
        let begin = Date()
        let ids = ErnieText.identifiers(prompt, tokenizer: try Tokenizer(directory: tokenizer))
        context.timings.tokenizer = Date().timeIntervalSince(begin)
        Arena.releaseToSystem()
        context.footprints.tokenizer = Arena.processFootprint()
        try context.check()
        let hiddenValues = try ErnieText.hiddenStates(ids, encoder: map, freezeCut: context.reproducible,
                                                cancellation: context.cancellation)
        context.footprints.encoder = Arena.processFootprint()
        return Conditioning(format: .ernie, rows: ids.count, width: hiddenValues.count / ids.count,
                               values: hiddenValues, tokens: ids.count)
    }
}

/// The ERNIE-Image Turbo DiT: Euler, CFG 1 (distilled), 8 steps, fixed schedule (`shift = 4`).
public struct ErnieDenoisingModule: DenoisingModule {
    public var map: String
    public var family: Family
    public var model: String { family.rawValue }
    public var name: String { "DiT ERNIE-Image" }
    public var entry: TextFormat { .ernie }
    public var space: LatentSpace { .flux2 }
    /// The model's card: `num_inference_steps=8, guidance_scale=1.0`.
    public var defaultSteps: Int { 8 }

    public init(map: String, family: Family = .ernie) { self.map = map; self.family = family }

    public func sigmas(steps: Int) -> [Float] { ErnieDiT.sigmas(steps: steps) }

    public func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan {
        DenoisingPlan(startStep: start, startSigma: sigmas(steps: steps)[start],
                         evaluations: steps - start, reduced: 0, reducedHeight: height, reducedWidth: width)
    }

    public func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                          context: Context) throws {
        let dit = try ErnieDiT(artifact: try Artifact(path: map), latentHeight: latent.height,
                               latentWidth: latent.width, textRows: text.rows,
                               freezeCut: context.reproducible, lora: lora)
        dit.cancellation = context.cancellation
        try text.values.withUnsafeBufferPointer { try dit.prepareText($0.baseAddress!) }
        try context.check()
        // The modulation of only the σs evaluated, in one pass; the following images of a batch reuse it.
        let σ = sigmas(steps: steps)
        let key = "ernie.modulation " + map
        if let table = context.reserve[key] as? ErnieDiT.ModulationTable { dit.modulations = table }
        try dit.tabulateModulation(sigmas: Array(context.told(σ, start: start)[start..<(σ.count - 1)]))
        context.reserve[key] = dit.modulations
        try euler(&latent, sigmas: σ, start: start, context: context) { x, sigma in
            try dit.forward(latent: x, sigma: sigma)
        }
        context.footprints.denoising = Arena.processFootprint()
    }
}

// ── Qwen-Image-2.1 ─────────────────────────────────────────────────────────────────────────

/// **Qwen3-VL-8B on the prompt and the references together** — the raw template, each image its
/// `<imageN>` slot, the vision tower, the last layer before the final norm. The references arrive at
/// the DiT's size (`QwenImage21DenoisingModule.preparedReferences`): the copy the VAE encodes too.
public struct QwenImage21TextModule: ImageTextModule {
    public var map: String
    /// The published `processor/` folder (its `tokenizer.json`).
    public var tokenizer: String
    public var output: TextFormat { .qwenImage21 }
    public var name: String { "Qwen3-VL-8B" }

    public init(map: String, tokenizer: String) { self.map = map; self.tokenizer = tokenizer }

    public func encoder(_ prompt: String, images: [ImageRGB], context: Context) throws -> Conditioning {
        let begin = Date()
        let tokenizer = try Tokenizer(directory: tokenizer)
        // Already at their size: `pixels` refuses one that `smart_resize` would resample.
        let pixels = try images.map { try Qwen3VLImages.pixels(QwenImage21Pipeline.rgb8($0)) }
        context.timings.tokenizer = Date().timeIntervalSince(begin)
        Arena.releaseToSystem()
        context.footprints.tokenizer = Arena.processFootprint()
        try context.check()
        let artifact = try Artifact(path: map)
        let encoder = try Qwen3VLEncoder(artifact: artifact, tokenizer: tokenizer, freezeCut: context.reproducible)
        encoder.cancellation = context.cancellation
        // Its own autorelease pool: the attention's autoreleased tensor data would otherwise keep
        // Metal buffers alive past the encoder on a thread that never drains (see `QwenImage21DiT.pass`).
        let out = try autoreleasepool { try encoder.encode(prompt: prompt, images: pixels) }
        // 16 GB read once: first in line for reclaim, before the DiT's weights come in.
        artifact.dropFromCache()
        context.footprints.encoder = Arena.processFootprint()
        context.reserve["qwen21.encoder"] = encoder.timings   // vision / language, for a check that measures
        return Conditioning(format: .qwenImage21, rows: out.rows, width: out.hidden, values: out.embeddings,
                            tokens: out.rows, imageSlots: out.imagePadMask)
    }
}

/// **The Qwen-Image-2.1 DiT under Viggle's turbo**: Euler without CFG, 6 steps on the raw nodes
/// shifted by μ of the target, one prefill per phase — the 9-step mode (`steps: 9`) ends on two steps
/// of the base model, its own prefill. The turbo is part of the model: it is stacked under the user's
/// LoRAs. See `QwenImage21Pipeline`.
public struct QwenImage21DenoisingModule: DenoisingModule {
    public var map: String
    /// Viggle's turbo LoRA, always applied unmerged (`Family.turboLoRA`).
    public var turbo: String
    package var scheduler: QwenImage21Pipeline.Scheduler
    /// `output_resolution`: the references' area (`R²`), and the edit format's.
    public var resolution: Int
    /// Where the K/V cache lives — `nil`: the policy (`QwenImage21CachePolicy`). Set by the checks that measure.
    package var storage: QwenImage21DiT.CacheStorage? = nil
    public var model: String { Family.qwenImage21.rawValue }
    public var name: String { "DiT Qwen-Image-2.1" }
    public var entry: TextFormat { .qwenImage21 }
    public var space: LatentSpace { .qwenImage21 }
    public var defaultSteps: Int { QwenImage21Pipeline.defaultSteps }
    /// Viggle's card: edits with 1 to 3 references.
    public var maxReferences: Int { 3 }

    package init(map: String, turbo: String, scheduler: QwenImage21Pipeline.Scheduler,
                resolution: Int = QwenImage21Pipeline.outputResolution) {
        self.map = map; self.turbo = turbo; self.scheduler = scheduler; self.resolution = resolution
    }

    /// Outside a grid, that of 1024² (4,096 tokens). A render takes its grid's.
    public func sigmas(steps: Int) -> [Float] { sigmas(steps: steps, height: 64, width: 64) }

    /// An unsupported count gives the 6-step schedule here: `check(steps:)` has refused it before.
    public func sigmas(steps: Int, height: Int, width: Int) -> [Float] {
        (try? QwenImage21Pipeline.sigmas(steps: steps, imageTokens: height * width, scheduler: scheduler))
            ?? QwenImage21Pipeline.sigmas(nodes: (try? QwenImage21Pipeline.nodes(steps: 6)) ?? [1],
                                          mu: scheduler.mu(imageTokens: height * width))
    }

    /// **No spectral schedule — refused by eye.** Z-Image's threshold (`f* ≤ 0.10`) gives
    /// `k = 3` here at 1024², −34 % (146 → 96 s), and it breaks every image: faces made of pasted
    /// fragments, patchwork clothes, shelves shattered into collage (seeds 42, 7, 0). `k = 2` breaks
    /// a corner on seed 7. The threshold does not transfer: a 16-px, 64-channel cell carries far
    /// more of the image than Z-Image's 8-px one. The half-size phase is kept out of the engine.
    public func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan {
        DenoisingPlan(startStep: start, startSigma: sigmas(steps: steps, height: height, width: width)[start],
                      evaluations: steps - start, reduced: 0, reducedHeight: height, reducedWidth: width)
    }

    public func check(steps: Int, startImage: Bool) throws {
        _ = try QwenImage21Pipeline.nodes(steps: steps)
        if startImage { throw QwenImage21Pipeline.Failure.imageToImage }
    }

    /// Its pipeline has no img2img: the image encoder serves the references only.
    public var acceptsStartImage: Bool { false }
    /// Its noise leaves in the last two evaluations: an early x̂₀ is blurred and grainy (`Sketch`).
    public var sketches: Bool { false }

    /// Pillow's Lanczos to `calculate_dimensions(R², w/h)`, each at its own ratio.
    public func preparedReferences(_ images: [ImageRGB]) throws -> [ImageRGB] {
        images.map { QwenImage21Pipeline.image(QwenImage21Pipeline.reference($0, resolution: resolution)) }
    }

    public func editFormat(referenceWidth: Int, referenceHeight: Int) -> (width: Int, height: Int) {
        QwenImage21Pipeline.outputSize(referenceWidth: referenceWidth, referenceHeight: referenceHeight,
                                       resolution: resolution)
    }

    public func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                        context: Context) throws {
        try denoise(&latent, text: text, steps: steps, start: start, references: [], lora: lora, context: context)
    }

    public func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int,
                        references: [Latent], lora: LoRA?, context: Context) throws {
        guard references.allSatisfy({ $0.space == space }), references.count <= maxReferences else {
            throw Request.Failure.references(references.count, max: maxReferences)
        }
        guard text.format == entry, text.imageSlots.count == text.rows else {
            throw Chain.Failure.incompatibleText(output: text.format, entry: entry)
        }
        try check(steps: steps, startImage: start > 0)
        let turboLayer = try LoRA(paths: [(path: turbo, strength: 1)]).layers
        let user = lora?.layers ?? []
        var sampler = QwenImage21Sampler(artifact: try Artifact(path: map),
                                         turbo: try LoRA(layers: turboLayer + user),
                                         base: user.isEmpty ? nil : try LoRA(layers: user))
        sampler.freezeCut = context.reproducible
        sampler.cancellation = context.cancellation
        sampler.storage = storage
        sampler.reusable = context.keptDenoiser as? QwenImage21DiT
        // **The last edit's K/V, kept** (`RenderCache`): only where the policy puts them on disk anyway
        // (a prefix with images); a text-only prefix is computed inside the first step for nearly
        // nothing. One key per phase — its stack changes the conditions' K/V.
        if let cache = context.cache, storage == nil {
            let prefix = text.rows + 3 * text.imageSlots.filter { $0 }.count
            if QwenImage21CachePolicy.storage(prefix: prefix) == .file {
                var stacks: [[LoRA.Layer]] = [turboLayer + user]
                if !user.isEmpty { stacks.append(user) }
                let keys = stacks.map { kvKey(stack: $0, text: text, references: references, latent: latent,
                                              reproducible: context.reproducible) }
                cache.keepOnlyKV(keys)
                sampler.kept = keys.map { (cache.kvPath($0), $0.hex) }
                sampler.keptReserve = cache.keptReserve
            }
        }
        let σ = sigmas(steps: steps, height: latent.height, width: latent.width)
        context.beginDenoising(evaluations: steps)
        let (height, width) = (latent.height, latent.width)
        try sampler.run(latent: &latent.values, height: height, width: width, hidden: text.values,
                        slots: text.imageSlots, references: references, sigmas: σ,
                        told: context.told(σ, start: start, bell: .late),
                        turboSteps: QwenImage21Pipeline.turboSteps(steps)) { step in
            try context.stepDone(sigma: step.sigma, seconds: step.seconds, evaluatedHeight: height,
                                 evaluatedWidth: width, space: space, x: step.latent, v: step.velocity,
                                 nextSigma: σ[step.index + 1], fullHeight: height, fullWidth: width)
        }
        context.keptDenoiser = sampler.reused
        context.footprints.denoising = Arena.processFootprint()
        // What a check that measures reads back: the prefill/steps split and the cache's storage.
        context.reserve["qwen21.denoising"] = (sampler.timings, sampler.used)
    }

    /// **What the conditions' K/V of one phase depend on**: the binary and the settings that change
    /// bits, the DiT's map, the phase's LoRA stack (each map's revision and strength), the encoder's
    /// states and slots, each reference latent, and the target's grid (it closes the joint sequence).
    /// Not the seed, not σ: the conditions read `t = 0` and never see the target.
    func kvKey(stack: [LoRA.Layer], text: Conditioning, references: [Latent], latent: Latent,
               reproducible: Bool) -> RenderCache.Key {
        var k = RenderCache.KeyBuilder("qwen21 kv v1")
        k.add(RenderCache.build); k.add(RenderCache.numerics(reproducible: reproducible))
        k.add(RenderCache.stamp(map))
        k.add(stack.count)
        for layer in stack { k.add(RenderCache.stamp(layer.artifact.path)); k.add(Int(layer.strength.bitPattern)) }
        k.add(text.format.name); k.add(text.rows); k.add(text.width)
        k.add(floats: text.values); k.add(flags: text.imageSlots)
        k.add(references.count)
        for reference in references { k.add(reference) }
        k.add(latent.height); k.add(latent.width)
        return k.finish()
    }
}

extension QwenImage21DenoisingModule {
    /// The library's turbo LoRA and Viggle's scheduler config (`scheduler-turbo.json`).
    static func product(map: String, in b: Library) throws -> QwenImage21DenoisingModule {
        QwenImage21DenoisingModule(
            map: map, turbo: try b.component(.qwenImage21, Family.qwenImage21.turboLoRA!),
            scheduler: try QwenImage21Pipeline.Scheduler.file(try b.component(.qwenImage21, "scheduler-turbo.json")))
    }
}

// ── the models: preset chains ────────────────────────────────────────────────────

/// **A model is a preset chain, a name and a license.** Its files are read from a `Library`
/// and checked at construction: a missing file throws here, before a minute of computation, with
/// its full path.
///
/// The three constructors and `named` take the library under the same label, `in:` — that of
/// `ModelCard.missing(in:)`. A custom chain becomes a model through
/// `Model(card:chain:)`: it is the only thing `render`, `renderBatch` and `events` take.
public struct Model: Sendable {
    /// The name, the license, the recommended formats: what the catalog knows without opening anything.
    public let card: ModelCard
    public let chain: Chain
    /// The library it was built from — where its renders keep what they do not redo
    /// (`Library.cacheFolder`). `nil` for a hand-composed chain: nothing is cached.
    package let library: Library?

    /// A hand-composed chain, under the sheet of the model whose denoiser it holds.
    public init(card: ModelCard, chain: Chain) { self.card = card; self.chain = chain; self.library = nil }

    package init(card: ModelCard, library: Library, chain: Chain) {
        self.card = card; self.chain = chain; self.library = library
    }

    /// `z-image`, `anima`, `krea2` — what `--model` accepts; an imported model is called
    /// `<family>/<nom>` (`z-image/mon-modele`).
    public var identifier: String { card.id }
    public var name: String { card.name }
    /// The architecture — the target a LoRA declares.
    public var family: Family { card.family }
    /// To be recalled to anyone distributing an image or the model.
    public var license: License { card.license }

    public static let identifiers = ModelCard.allCards.map(\.id)

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
    public static func named(_ identifier: String, in b: Library) throws(EngineError) -> Model {
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
        case "klein-4b": return try klein4b(in: b, card: .klein4b, dit: b.ditMap(.klein4b))
        case "ernie-image": return try ernie(in: b, card: .ernie, dit: b.ditMap(.ernie))
        case "qwen-image-2.1": return try qwenImage21(in: b, card: .qwenImage21, dit: b.ditMap(.qwenImage21))
        default:
            guard let card = b.importedCards().first(where: { $0.id == identifier }), let dit = card.map else {
                throw Failure.unknown(identifier)
            }
            switch card.family {
            case .zImage: return try zImage(in: b, card: card, dit: dit)
            case .anima: return try anima(in: b, card: card, dit: dit)
            case .krea2: return try krea2(in: b, card: card, dit: dit)
            case .klein4b: return try klein4b(in: b, card: card, dit: dit)
            case .ernie: return try ernie(in: b, card: card, dit: dit)
            case .qwenImage21: return try qwenImage21(in: b, card: card, dit: dit)
            }
        }
    }

    /// The library's `store/<nom>` map, if it is there (`MissingFile` otherwise, folded into
    /// `modelNotInstalled` by `building`).
    static func map(_ name: String, _ b: Library) throws -> String {
        let path = name.hasPrefix("/") ? name : b.map(name)
        guard FileManager.default.fileExists(atPath: path) else { throw MissingFile(path, .map) }
        return path
    }

    /// Z-Image Turbo — the product's default. `denoising`: a denoiser tuned differently from the
    /// product's (a forced spectral schedule, for instance).
    public static func zImage(in b: Library, denoising: ZImageDenoisingModule? = nil) throws(EngineError) -> Model {
        try building(ModelCard.zImage.id) { try zImage(in: b, card: .zImage, dit: b.ditMap(.zImage), denoising: denoising) }
    }

    static func zImage(in b: Library, card: ModelCard, dit: String,
                       denoising: ZImageDenoisingModule? = nil) throws -> Model {
        Model(card: card, library: b, chain: try Chain(
                   text: ZImageTextModule(map: try map(b.encoderMap(.zImage), b),
                                      tokenizer: try b.component(.zImage, "tokenizer")),
                   denoising: denoising ?? .product(map: try map(dit, b)),
                   decoding: FluxDecodingModule(path: try b.component(.zImage, "vae.safetensors")),
                   encoding: FluxEncodingModule(path: try b.component(.zImage, "vae.safetensors"))))
    }

    /// Anima Turbo v1.1. ⚠️ Non-commercial license: never a product's default.
    public static func anima(in b: Library) throws(EngineError) -> Model {
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
    public static func krea2(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.krea2.id) { try krea2(in: b, card: .krea2, dit: b.ditMap(.krea2)) }
    }

    static func krea2(in b: Library, card: ModelCard, dit: String) throws -> Model {
        Model(card: card, library: b, chain: try Chain(
                   text: Krea2TextModule(map: try map(b.encoderMap(.krea2), b),
                                     tokenizer: try b.component(.krea2, "tokenizer"),
                                     sockets: try Krea2Text.sockets(index: try b.component(.krea2, "model_index.json"))),
                   denoising: Krea2DenoisingModule(map: try map(dit, b)),
                   decoding: QwenImageDecodingModule(path: try b.component(.krea2, "vae.safetensors")),
                   encoding: QwenImageEncodingModule(path: try b.component(.krea2, "vae.safetensors"))))
    }

    /// FLUX.2 [klein] 4B, distilled to 4 steps. Apache 2.0.
    public static func klein4b(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.klein4b.id) { try klein4b(in: b, card: .klein4b, dit: b.ditMap(.klein4b)) }
    }

    static func klein4b(in b: Library, card: ModelCard, dit: String) throws -> Model {
        Model(card: card, library: b, chain: try Chain(
                   text: KleinTextModule(map: try map(b.encoderMap(.klein4b), b),
                                     tokenizer: try b.component(.klein4b, "tokenizer")),
                   denoising: KleinDenoisingModule(map: try map(dit, b), family: .klein4b),
                   decoding: Flux2DecodingModule(path: try b.component(.klein4b, "vae.safetensors")),
                   encoding: Flux2EncodingModule(path: try b.component(.klein4b, "vae.safetensors"))))
    }

    /// ERNIE-Image Turbo, distilled to 8 steps. Apache 2.0.
    public static func ernie(in b: Library) throws(EngineError) -> Model {
        try building(ModelCard.ernie.id) { try ernie(in: b, card: .ernie, dit: b.ditMap(.ernie)) }
    }

    static func ernie(in b: Library, card: ModelCard, dit: String) throws -> Model {
        let vae = try b.component(.ernie, "vae.safetensors")
        return Model(card: card, library: b, chain: try Chain(
                   text: ErnieTextModule(map: try map(b.encoderMap(.ernie), b),
                                     tokenizer: try b.component(.ernie, "tokenizer")),
                   denoising: ErnieDenoisingModule(map: try map(dit, b), family: .ernie),
                   decoding: Flux2DecodingModule(path: vae, eps: 1e-5),
                   encoding: Flux2EncodingModule(path: vae, eps: 1e-5)))
    }

    /// Qwen-Image-2.1 under Viggle's turbo: 6 steps, generation and editing by instruction (1 to 3
    /// references, which its encoder sees). ⚠️ Qwen Research license: non-commercial.
    public static func qwenImage21(in b: Library) throws(EngineError) -> Model {
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
}
