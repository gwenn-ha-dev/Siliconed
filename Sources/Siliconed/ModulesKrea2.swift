// The Krea 2 modules (text, denoising). Split from `Modules.swift`.

import Foundation

// ── Krea 2 ──────────────────────────────────────────────────────────────────────────────────

/// Template, Qwen tokenizer, Qwen3-VL-4B with its twelve layers taken. Not the fusion: that is the DiT's.
package struct Krea2TextModule: TextModule {
    package var map: String
    package var tokenizer: String
    /// The layers taken, read from `model_index.json`.
    package var sockets: [Int]
    package var output: TextFormat { .krea2 }
    package var name: String { "Qwen3-VL-4B" }

    package init(map: String, tokenizer: String, sockets: [Int]) {
        self.map = map; self.tokenizer = tokenizer; self.sockets = sockets
    }

    package func encoder(_ prompt: String, context: Context) throws -> Conditioning {
        let begin = Date()
        let ids = try Krea2Text.admitted(prompt, tokenizer: try Tokenizer(directory: tokenizer))
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
package struct Krea2DenoisingModule: DenoisingModule {
    package var map: String
    package var model: String { "krea2" }
    package var name: String { "DiT Krea 2" }
    package var entry: TextFormat { .krea2 }
    package var space: LatentSpace { .qwenImage }
    package var defaultSteps: Int { 8 }

    package init(map: String) { self.map = map }

    /// Every `Solver`: `euler` walks them on Krea 2's own σ (the exponential shift), judged against
    /// diffusers' schedulers (`SolverTests`).
    package var solvers: [Solver] { Solver.allCases }
    /// No Detail: the node's bell changes the pose and the face, not the texture (`Detail`).
    package var acceptsDetail: Bool { false }

    package func sigmas(steps: Int) -> [Float] { Krea2DiT.sigmas(steps: steps) }

    /// **By default every evaluation at full size**: the published recipe as it stands.
    ///
    /// **The half-size start** (`halfSizeStart`, off by default; Z-Image's recipe, by `euler`): the
    /// first `halfSizeSteps` evaluations on the half grid. Only at the recipe's eight, from the first
    /// step, on a grid that halves above the floor (`Spectral.floor` in 8-px cells: 1024², 1024×1536
    /// and its transpose; a 512² or 832×1216 never halves), never under a user LoRA or another
    /// schedule (`withLoRA`: never judged under them). `SILICONED_SPECTRAL` forces `k` (a
    /// measurement), within the same rules.
    package func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan {
        let halvable = Spectral.divisible(height: height, width: width, by: 2,
                                          floor: Spectral.floor(factor: space.factor))
        let rule = Spectral.halfSizeStart && steps == defaultSteps ? Self.halfSizeSteps : 0
        let k = start > 0 || !halvable || withLoRA ? 0 : min(max(0, Spectral.forced ?? rule), steps - 1)
        return DenoisingPlan(startStep: start, startSigma: sigmas(steps: steps)[start], evaluations: steps - start,
                             reduced: k, reducedHeight: k > 0 ? height / 2 : height,
                             reducedWidth: k > 0 ? width / 2 : width)
    }

    /// Its half-size evaluations: 4 (σ 1 → 0.760), judged by eye on two prompts at 1024² — one more than
    /// Z-Image's and Qwen-Image-2.1's, Krea 2's shifted schedule lingering longer at high σ. Not the
    /// largest that renders: at 5 the freckles start to fade, at 6 the shelves split into strips and an
    /// eyelid doubles, at 7 the faces ghost.
    package static let halfSizeSteps = 4

    /// **The half-size start builds two DiTs, one at a time**: the half-grid one (its own RoPE grid,
    /// its own arena and sequence `s/4 + T`, exactly a native evaluation at half the size), given
    /// back before the full one maps its arena — never both, the full one is the larger and is what
    /// the memory need charges (`MemoryNeed.krea2`). The two share the map, its staging stream and
    /// the modulation table (a function of σ alone); the fused text is the same rows.
    package func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                          context: Context) throws {
        let artifact = try Artifact(path: map)
        let tokens = text.rows
        let merged = try fusion(text, artifact: artifact, lora: lora, context: context)
        let (height, width) = (latent.height, latent.width)
        let planned = plan(height: height, width: width, steps: steps, start: start,
                           withLoRA: lora != nil || context.detail != .normal, schedule: context.schedule)
        // The modulation of only the σs evaluated — from `start` to the last non-null one —, in
        // one pass. It depends only on σ: the following images of a batch reuse the first's table.
        let σ = sigmas(steps: steps)
        let modulationKey = "krea2.modulation " + map
        func build(height: Int, width: Int) throws -> Krea2DiT {
            let dit = try Krea2DiT(artifact: artifact, latentHeight: height, latentWidth: width, maxText: tokens,
                                   freezeCut: context.reproducible, neuralEngine: context.neuralEngine, lora: lora)
            dit.cancellation = context.cancellation
            try context.check()
            if let table = context.reserve[modulationKey] as? Krea2DiT.ModulationTable { dit.modulations = table }
            try dit.tabulateModulation(sigmas: Array(context.told(σ, start: start)[start..<(σ.count - 1)]))
            context.reserve[modulationKey] = dit.modulations
            return dit
        }
        var small = planned.reduced > 0
            ? try build(height: planned.reducedHeight, width: planned.reducedWidth) : nil
        var full = small == nil ? try build(height: height, width: width) : nil
        let half = small.map { _ in
            HalfSizeStart(steps: planned.reduced, speed: { x, sigma in
                try merged.withUnsafeBufferPointer {
                    try small!.forward(latent: x, text: $0.baseAddress!, textRows: tokens, sigma: sigma)
                }
            }, handOver: {
                small = nil
                full = try build(height: height, width: width)
            })
        }
        try euler(&latent, sigmas: σ, start: start, context: context, half: half) { x, sigma in
            try merged.withUnsafeBufferPointer {
                try full!.forward(latent: x, text: $0.baseAddress!, textRows: tokens, sigma: sigma)
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
                                         freezeCut: context.reproducible, neuralEngine: context.neuralEngine,
                                         lora: lora)
        fusion.cancellation = context.cancellation
        let merged = try text.values.withUnsafeBufferPointer {
            try fusion.fuse($0.baseAddress!, tokens: text.rows)
        }
        context.reserve[key] = FusionDone(text: text.values, lora: lora, merged: merged)
        return merged
    }
}
