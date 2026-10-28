import Foundation

/// **Anima's need, phase by phase** — nothing but the constructors' shape functions.
///
///     crossing: text [L, 1024] · latent [16, H/8, W/8] · (img2img) the image as handed, and z₀
///
///     text        TextEncoder arena(T) + its staging buffers × the largest encoder layer          ┐
///                 + causal SDPA: mask T·T (ours) ⊕ queries' copy (MPSGraph) · its GEMM           │ one at
///                 + the conditioner's arena (L T5 rows reading T Qwen rows, in double)            │ a time,
///     references  (img2img, Détail) the `fitted` copy when the size differs                     │ each
///                 + the Qwen-Image encoder's two heaps and host copies (`VAEEncoder.plan`)        │ handing
///     denoising   the DiT's arena(s, L, r) + the velocity   + LoRA `down` staging                 │ back its
///                 + two SDPAs (self s×s, cross s×L), cut by heads or not · its GEMM               │ memory
///     decoding    the Qwen-Image decoder's two heaps and the image it returns (`AnimaVAE.plan`)    ┘
///                 (a batch: the images already decoded, and every latent crossing until then)
///
///     peak = max(phases) + crossing + process base      s = (H/16)(W/16), T ≤ 1 024, L = 1 025, r = Σ ranks
///
/// Ours, each the function its allocator calls: `TextEncoder.arenaBytes`, `TextEncoder.stagingSlots`,
/// `Artifact.largestRun`, `AnimaConditioner.arenaBytes`, `AnimaDiT.arenaBytes`, `LoRA.downBytes`,
/// `HeadSplitAttention.hostBytes`, `AnimaDiT.tableBytes`,
/// `Attention.maskBytes`, `Conductor.workspaceBytes`, `GEMM.tierBytes`, `VAEEncoder.plan`, `AnimaVAE.plan`.
/// Outside our hand, from the shapes: MPSGraph's copy of the queries (`OperatorFootprint.sdpaQueries`),
/// the VAEs' SDPA copies (`VAELayout.outside`). The frameworks' constants are not charged.
///
/// What the formula has to know and a port by analogy with Z-Image would miss:
///
///   - **the DiT's map is never in the need, whatever its type** — Standard (bf16), Compact (Q8_0),
///     Light (Q4_K_M): `AnimaDiT` reads it through the page cache (the prefetcher's `madvise`, no
///     streamed tail, no staging buffer) and widens each weight into one fp32 `reserve` of `d·h`. The
///     three versions have **the same denoising need**; the map's type shows only through the text
///     encoder's staging, read from that map's header (a version's own encoder, if it has one);
///   - **the DiT's text is charged at its bound, `L` = 1 025 rows**: the conditioner pads its output to
///     512 with zeros that count in the softmax, and past 512 it is the T5 tokens and `</s>`, which the
///     prompt's bytes do not bound (NFKC, below) — so the DiT's text k/v cache (`[28, L, 2048]` twice,
///     470 MB) is charged whole, whatever the prompt;
///   - **two encoders' worth of text, both alive**: the conditioner reads the Qwen states in the
///     encoder's own arena (`hidden` is a pointer into it), so the encoder's arena, its staging buffers
///     and its SDPA are alive beside the conditioner's arena (`AnimaTextModule`);
///   - **the T5 rows are bounded by the cut, not by the prompt's bytes**: T5's normalizer is NFKC,
///     which expands (one 3-byte ligature gives 18 letters) — the conditioner is charged at `L` rows;
///     the Qwen rows are bounded by the bytes (`Request.promptTokens`, no template, refused past 1 024);
///   - **`modulation` and `temb` are computed in double** (`AnimaExact`): `t_embedder.linear_2`'s
///     `d × 3d` weight widened to double, 100 MB of the DiT's arena (`AnimaDiT.exactProducts`) — and
///     **tabulated for the render's σ before the first evaluation**, in an arena of their own
///     (`AnimaDiT.tableBytes`, ~17 MB at its `tableRows` = 8 σ, whatever the steps: more are tabulated
///     8 at a time);
///   - **two attentions per DiT, both cut by heads where `HeadSplitAttention.applies`** (self at `s`,
///     cross at `s × 512`: two host arenas, each with its MPSGraph copy of the GPU heads' queries),
///     alive with the DiT; no K/V on disk. **The half-size start** (off by default) builds a DiT for the
///     half grid first and releases it, with the start's few latents (ε, the small grid, the upsampled
///     estimate), before the full one: smaller by far than the full DiT, it is not charged;
///   - **a LoRA lives in the DiT's arena** (`loraBas`, `loraHaut`, `loraMid`, at the stack's cumulative
///     rank and the widest module it may touch, `AnimaDiT.loraBounds`), no cache of widened weights as
///     Z-Image's; with two LoRAs or more, `down` is staged through `LoRA.downBytes`;
///   - **the Qwen-Image decoder is one pass per operation**, no bands (`AnimaVAE.stages`): the lean
///     plan does not shrink it.
extension MemoryNeed {

    /// **What Anima's formula reads beside the request**: the DiT's config and the encoder's config and
    /// tensor table (their headers, never a tensor) — each file read once, by its stamp. A header that
    /// does not parse makes no maps: the need is unknown and the render refused
    /// (`EngineError.memoryNeedUnknown`).
    package struct AnimaMaps: Sendable {
        package let dit: AnimaDiT.Config
        package let encoder: TextEncoder.Config
        /// The largest tensor's run in the encoder's layers (`Artifact.largestRun`): what one of its buffers holds.
        package let encoderLayerStaging: Int
        /// What widening the DiT's largest int8 convrot weight allocates outside every arena
        /// (`Widen.rotationScratchBytes`): an imported Comfy-style convrot DiT; 0 otherwise.
        package let ditRotationScratch: Int

        /// From the headers — pure: the tests hand it headers written in memory. `nil` when the
        /// encoder's config does not parse or its map has no tensor table.
        package init?(dit: [String: Any], encoder: [String: Any], encoderSize: Int = .max) {
            guard let config = try? TextEncoder.Config(header: encoder) else { return nil }
            let staging = Artifact.largestRun(header: encoder, prefixes: TextEncoder.layerPrefixes(config),
                                                fileSize: encoderSize)
            guard staging > 0 else { return nil }
            self.dit = AnimaDiT.Config(header: dit); self.encoder = config; encoderLayerStaging = staging
            ditRotationScratch = Widen.rotationScratchBytes(header: dit)
        }

        /// The maps this card renders with: its own DiT (an imported one) or the library's installed
        /// version (Standard, Compact or Light), and that version's encoder. `nil` when one cannot be read.
        package init?(card: ModelCard, library: Library = .processWide) {
            let ditPath = card.map ?? library.map(library.ditMap(.anima))
            let encoderPath = library.map(library.encoderMap(.anima))
            let maps = AnimaMaps.installed(InstalledFiles(paths: [ditPath, encoderPath])) { () -> AnimaMaps? in
                guard let dit = try? Artifact.header(ditPath), let encoder = try? Artifact.header(encoderPath) else { return nil }
                return AnimaMaps(dit: dit, encoder: encoder, encoderSize: InstalledFiles.size(encoderPath))
            }
            guard let maps else { return nil }
            self = maps
        }

        /// The maps read, by their files' stamps.
        private static let installed = ShapeMemo<InstalledFiles, AnimaMaps?>(capacity: 8)
    }

    /// **Anima's need** on the installed maps.
    static func anima(_ request: Request) -> MemoryNeed? {
        guard let maps = request.animaMaps ?? AnimaMaps(card: request.card, library: request.library) else { return nil }
        return anima(request, maps: maps)
    }

    /// The decoder's plan, walked once per latent grid (`AnimaVAE.plan` walks every pass).
    private static let animaDecoderPlans = ShapeMemo<[Int], VAELayout>()

    /// **Anima's need on these maps** — pure: the tests hand it headers written in memory.
    package static func anima(_ request: Request, maps: AnimaMaps) -> MemoryNeed {
        let dit = maps.dit, encoder = maps.encoder
        let f = MemoryLayout<Float>.size
        // The Qwen ids are refused past the bound (`AnimaTextModule`): the encoder never sees more; the
        // request's tokens bound them by the prompt's bytes.
        let tokens = min(max(1, request.promptTokens), Siliconed.Request.maximumPromptTokens)
        // T5: charged at its bound whatever the prompt (NFKC expands, the bytes do not bound it),
        // padded to 512 at least.
        let t5 = AnimaConditioner.Shape.maximumTokens
        let textRows = max(AnimaConditioner.Shape.minimumLength, t5)
        // The Qwen-Image VAE: 8 pixels per latent cell, `dit.channels` (16) per cell.
        let (latentHeight, latentWidth) = (request.height / 8, request.width / 8)
        let latent = dit.channels * latentHeight * latentWidth * f
        let rank = request.loraRanks.reduce(0, +)
        let (width, height) = (request.width, request.height)

        // ── what crosses: the conditioning (`[L, 1024]`), the latent, img2img's image as handed
        //    (kept by the request through the render) and its z₀ at the render's grid ─────────────
        let text = textRows * AnimaConditioner.Shape.dim * f
        let starts = request.references.reduce(0) { $0 + 3 * $1.width * $1.height * f + latent }
        let crossing = text + latent + starts + batchLatents(request, latent: latent)

        // ── text: the encoder's arena, its staging buffers, its causal SDPA, its GEMM — and, the
        //    encoder still alive, the conditioner's arena ─────────────────────────────────────────
        let (h, a, kv, i) = (encoder.hidden, encoder.attentionWidth, encoder.kvWidth, encoder.intermediate)
        let textPhase = Phase(kind: .text, terms: [
            Term("encoder arena", TextEncoder.arenaBytes(config: encoder, sequence: tokens)),
            Term("encoder staging", TextEncoder.stagingSlots(granted: request.encoderSlots) * maps.encoderLayerStaging),
            Term("conditioner arena", AnimaConditioner.arenaBytes(t5: t5, qwen: tokens)),
        ] + attention("encoder", heads: encoder.heads, rows: tokens, keys: tokens, headDim: encoder.headDim,
                      causal: true, split: false, threads: request.threads)
          + gemm("encoder", bounded: [(tokens, h, a), (tokens, h, kv), (tokens, a, h), (tokens, h, i), (tokens, i, h)]))

        // ── img2img's starting image, encoded at the render's format (`fitted`: a copy when the size
        //    handed differs), one encoder at a time: its heaps and the latent it returns ─────────
        let references = request.references.map { image in
            Phase(kind: .references, terms: [
                Term("fitted copy", image.width == width && image.height == height ? 0 : 3 * width * height * f),
            ] + vae("Qwen-Image encoder", VAEEncoder.plan(family: .qwenImage, height: height, width: width)))
        }

        // ── denoising: the arena, the velocity, the two SDPAs, the GEMM ─────────────────────────────
        let s = AnimaDiT.imageTokens(latentHeight: latentHeight, latentWidth: latentWidth, config: dit)
        let (d, hidden) = (dit.dim, dit.hidden)
        let inDim = (dit.channels + 1) * dit.patch * dit.patch, outDim = dit.channels * dit.patch * dit.patch
        let bounds = AnimaDiT.loraBounds(config: dit)
        let denoising = Phase(kind: .denoising, terms: [
            Term("DiT arena", AnimaDiT.arenaBytes(config: dit, latentHeight: latentHeight, latentWidth: latentWidth,
                                                  textLength: textRows,
                                                  lora: rank > 0 ? (rank, bounds.entry, bounds.output) : nil)),
            // Every modulation of up to `tableRows` σ, whatever the steps (`AnimaDiT.tabulate`).
            Term("modulation table", AnimaDiT.tableBytes(config: dit, count: AnimaDiT.tableRows)),
            Term("LoRA down staging", request.loraRanks.count > 1
                ? LoRA.downBytes(k: bounds.entry, rank: request.loraRanks.max() ?? 0) : 0),
            // The velocity the DiT returns (`forward`'s `[C, h, w]`); Euler keeps no history.
            Term("sampler", latent + request.solver.historyBytes(count: latent / f)),
            Term("DiT rotation scratch", maps.ditRotationScratch),
        ] + attention("self-attention", heads: dit.heads, rows: s, keys: s, headDim: dit.headDim, causal: false,
                      split: HeadSplitAttention.applies(sequence: s, heads: dit.heads, headDim: dit.headDim), threads: request.threads)
          + attention("cross-attention", heads: dit.heads, rows: s, keys: textRows, headDim: dit.headDim, causal: false,
                      split: HeadSplitAttention.applies(sequence: s, heads: dit.heads, headDim: dit.headDim), threads: request.threads)
          // Every height is exact: the image's tokens, and the text's `L`.
          + gemm("DiT", products: [(s, inDim, d), (s, d, d), (s, d, hidden), (s, hidden, d), (s, d, outDim),
                                   (textRows, dit.textDim, d)],
                 rank: rank, ane: request.neuralEngine))

        // ── decoding: the decoder's two heaps and the image it returns; the latent it reads crosses ──
        let plan = animaDecoderPlans([latentHeight, latentWidth]) {
            AnimaVAE.plan(latentHeight: latentHeight, latentWidth: latentWidth)
        }
        let decoding = Phase(kind: .decoding, terms: vae("Qwen-Image decoder", plan) + [batchImages(request)])

        return MemoryNeed(phases: [textPhase] + references + [denoising, decoding], crossing: crossing, base: request.base)
    }
}
