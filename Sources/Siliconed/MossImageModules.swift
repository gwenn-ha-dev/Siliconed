import Foundation

// **MOSS-Image's modules, and the model that assembles them.**
//
//     MossImageText ── Qwen3.5-9B, last hidden state ──▶ MossImageDenoising ── DiT + the DMD LoRA, Euler, CFG 1 ──▶ Flux2Decoding
//
// The recipe is the community's 8-step one: `MOSS-Image-SFT` under `lodestones`' DMD LoRA at strength 1, one evaluation
// per step, no negative branch. The official one (50 steps, CFG 3, block 22 skipped in the negative branch) is the DiT's
// `forward(skipping:)`, judged by `check moss-image dit`, and is not offered here.

/// The chat template (no system turn, the thinking block open), the Qwen2 tokenizer, Qwen3.5-9B: the last hidden
/// state after the final norm, every token of the template kept, one unpadded forward.
package struct MossImageTextModule: TextModule {
    package var map: String
    /// The `tokenizer/` folder of the published repository (its `tokenizer.json`).
    package var tokenizer: String
    package var output: TextFormat { .mossImage }
    package var name: String { "Qwen3.5-9B" }

    package init(map: String, tokenizer: String) { self.map = map; self.tokenizer = tokenizer }

    package func encoder(_ prompt: String, context: Context) throws -> Conditioning {
        let begin = Date()
        let tokenizer = try Tokenizer(directory: tokenizer)
        // The template's own tokens are not the user's: the bound counts the prompt's.
        let ids = Qwen35Prompt.ids(prompt, tokenizer: tokenizer, maxLength: .max)
        let template = Qwen35Prompt.ids("", tokenizer: tokenizer, maxLength: .max).count
        try Request.admitPrompt(tokens: ids.count - template, max: Request.maximumPromptTokens)
        context.timings.tokenizer = Date().timeIntervalSince(begin)
        context.footprints.tokenizer = Arena.processFootprint()
        try context.check()
        let artifact = try Artifact(path: map)
        let hidden = try autoreleasepool { () throws -> [Float] in
            let encoder = try Qwen35Encoder(artifact: artifact, sequence: ids.count, freezeCut: context.reproducible)
            encoder.cancellation = context.cancellation
            return try encoder.encode(ids: ids)
        }
        // 16 GB read once: first in line for reclaim, before the DiT's weights come in.
        artifact.dropFromCache()
        context.footprints.encoder = Arena.processFootprint()
        return Conditioning(format: .mossImage, rows: ids.count, width: hidden.count / max(1, ids.count), values: hidden,
                            tokens: ids.count - template)
    }
}

/// The DiT under the family's LoRA — **always**: the 8-step recipe is the DMD LoRA at strength 1, applied to the
/// bf16 weights beside the products (`CacheLoRA`), never merged into them. A request's own LoRAs stack after it.
package struct MossImageDenoisingModule: DenoisingModule {
    package var map: String
    /// The LoRA's map (`MossImageForge.forgeLoRA`), in the family's components.
    package var turbo: String
    package var model: String { MossImageForge.model }
    package var name: String { "DiT MOSS-Image" }
    package var entry: TextFormat { .mossImage }
    package var space: LatentSpace { .flux2 }
    package var defaultSteps: Int { 8 }
    /// No img2img: the family has no image encoder (the product has none).
    package var acceptsStartImage: Bool { false }
    /// The lie about σ was never judged on this model.
    package var acceptsDetail: Bool { false }

    package init(map: String, turbo: String) { self.map = map; self.turbo = turbo }

    /// Shift 5 on the 1000-step grid, rounded to it: at most `steps` σ, followed by the final 0.
    package func sigmas(steps: Int) -> [Float] { MossImageSchedule.sigmas(steps: steps) }

    /// **By default every evaluation at full size**: the recipe as published.
    ///
    /// **The half-size start** (`Spectral.halfSizeStart`, the app's Faster Rendering; `euler`, as Krea 2): the first
    /// `halfSizeSteps` evaluations on the half grid. Only at the recipe's eight, from the first step, on a grid that
    /// halves above the 512-px floor (16-px cells: 1024², 1024×1536 and its transpose), never under a request's LoRA —
    /// the turbo one is the recipe, not a request's. `SILICONED_SPECTRAL` forces `k` (a measurement), within the same rules.
    package func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan {
        let σ = sigmas(steps: steps)
        let halvable = Spectral.divisible(height: height, width: width, by: 2, floor: Spectral.floor(factor: space.factor))
        let rule = Spectral.halfSizeStart && steps == defaultSteps ? Self.halfSizeSteps : 0
        let evaluations = σ.count - 1 - start
        let k = start > 0 || !halvable || withLoRA ? 0 : min(max(0, Spectral.forced ?? rule), evaluations - 1)
        return DenoisingPlan(startStep: start, startSigma: σ[start], evaluations: evaluations,
                             reduced: k, reducedHeight: k > 0 ? height / 2 : height, reducedWidth: k > 0 ? width / 2 : width)
    }

    /// Its half-size evaluations: **1** (σ 1 → 0.968), judged by eye on the library and surfer prompts at 1024² — the
    /// fewest of the five: MOSS takes the half grid worse than the others. At 2 the skin smooths and the book spines
    /// soften, at 3 the skin turns waxy and hands go soft, at 5 the faces blotch. One saves 6 to 11 % at 1024².
    package static let halfSizeSteps = 1

    /// The turbo LoRA at strength 1, then the request's.
    package func stack(_ user: LoRA?) throws -> LoRA {
        let turbo = try LoRA(paths: [(turbo, 1)])
        try turbo.checkTarget(model: model)
        return try LoRA(layers: turbo.layers + (user?.layers ?? []))
    }

    package func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                         context: Context) throws {
        let artifact = try Artifact(path: map)
        let tokens = text.rows
        let σ = sigmas(steps: steps)
        // What the DiT is told: `σ · 1000` in fp32 (the scheduler's timesteps); no bell, the σ themselves.
        let told = context.told(σ, start: start)
        let timesteps = told[start..<(σ.count - 1)].map { $0 * Float(MossImageSchedule.trainTimesteps) }
        let stack = try stack(lora)
        let planned = plan(height: latent.height, width: latent.width, steps: steps, start: start,
                           withLoRA: lora != nil || context.detail != .normal)
        // The modulation depends on σ alone: both grids, and the following images of a batch, read the first's table.
        let key = "moss-image.modulation " + map
        // **Two DiTs, one at a time** (Krea 2's rule): the half grid's — its own RoPE grid, arena and sequence
        // `S/4 + T` — given back before the full one maps its arena; the full one is what the need charges. The
        // full one takes over the small one's expanded LoRA, the one thing that outlives it.
        func build(height: Int, width: Int, expanded: CacheLoRA? = nil) throws -> MossImageDiT {
            let dit = try MossImageDiT(artifact: artifact, gridHeight: height, gridWidth: width, maxText: tokens,
                                       freezeCut: context.reproducible, neuralEngine: context.neuralEngine, lora: stack,
                                       expanded: expanded)
            dit.cancellation = context.cancellation
            try context.check()
            if let table = context.reserve[key] as? MossImageDiT.ModulationTable { dit.modulations = table }
            try dit.tabulateModulation(timesteps: Array(timesteps))
            context.reserve[key] = dit.modulations
            return dit
        }
        var small = planned.reduced > 0
            ? try build(height: planned.reducedHeight, width: planned.reducedWidth) : nil
        var full = small == nil ? try build(height: latent.height, width: latent.width) : nil
        let (height, width) = (latent.height, latent.width)
        func speed(_ dit: MossImageDiT, _ x: UnsafePointer<Float>, _ sigma: Float) throws -> [Float] {
            try text.values.withUnsafeBufferPointer {
                try dit.forward(latent: x, text: $0.baseAddress!, textRows: tokens, timestep: sigma * Float(MossImageSchedule.trainTimesteps))
            }
        }
        let half = small.map { _ in
            HalfSizeStart(steps: planned.reduced, speed: { x, sigma in try speed(small!, x, sigma) }, handOver: {
                let expanded = small?.expandedLoRA
                small = nil
                full = try build(height: height, width: width, expanded: expanded)
            })
        }
        try euler(&latent, sigmas: σ, start: start, context: context, half: half) { x, sigma in try speed(full!, x, sigma) }
        context.footprints.denoising = Arena.processFootprint()
    }
}

/// FLUX.2's decoder on the DiT's latent: the batch-norm denormalization, the 2×2 unpatchify, the decoder, the clamp.
package struct Flux2DecodingModule: DecodingModule {
    package var path: String
    package var space: LatentSpace { .flux2 }
    package var name: String { "VAE FLUX.2" }
    package init(path: String) { self.path = path }

    package func decode(_ latent: Latent, context: Context) throws -> ImageRGB {
        // The decoder's grid is twice the DiT's (`[32, 2h, 2w]` after the unpatchify).
        let vae = try Flux2VAE(path: path, latentHeight: latent.height * 2, latentWidth: latent.width * 2)
        let (image, seconds) = try latent.values.withUnsafeBufferPointer { try vae.decode(latent: $0.baseAddress!) }
        context.timings.decoding = seconds
        return ImageRGB(pixels: image, height: vae.imageHeight, width: vae.imageWidth)
    }
}

extension MossImageTextModule: CacheIdentified {
    package var cacheIdentity: [String] {
        [String(describing: type(of: self)), output.name] + [map, tokenizer].map(RenderCache.stamp)
    }
}
