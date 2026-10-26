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

    /// Every evaluation at full size: the recipe as published.
    package func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan {
        let σ = sigmas(steps: steps)
        return DenoisingPlan(startStep: start, startSigma: σ[start], evaluations: σ.count - 1 - start,
                             reduced: 0, reducedHeight: height, reducedWidth: width)
    }

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
        let dit = try MossImageDiT(artifact: artifact, gridHeight: latent.height, gridWidth: latent.width, maxText: tokens,
                                   freezeCut: context.reproducible, neuralEngine: context.neuralEngine, lora: try stack(lora))
        dit.cancellation = context.cancellation
        try context.check()
        // The modulation depends on σ alone: the following images of a batch reuse the first's table.
        let key = "moss-image.modulation " + map
        if let table = context.reserve[key] as? MossImageDiT.ModulationTable { dit.modulations = table }
        try dit.tabulateModulation(timesteps: Array(timesteps))
        context.reserve[key] = dit.modulations
        try euler(&latent, sigmas: σ, start: start, context: context) { x, sigma in
            try text.values.withUnsafeBufferPointer {
                try dit.forward(latent: x, text: $0.baseAddress!, textRows: tokens, timestep: sigma * Float(MossImageSchedule.trainTimesteps))
            }
        }
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
