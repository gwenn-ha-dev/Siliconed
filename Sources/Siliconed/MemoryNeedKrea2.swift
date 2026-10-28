import Foundation

/// **Krea 2's need, phase by phase** — nothing but the constructors' shape functions.
///
///     crossing: the 12 taps [T, 12, 2560] · the fused text [T, 6144] and the modulation table (kept by
///               the context to the render's end) · latent [16, H/8, W/8] · (img2img) the image, z₀
///
///     text        TextEncoder arena(S) + its staging buffers × the largest encoder layer          ┐
///                 + causal SDPA: mask S·S (ours) ⊕ queries' copy (MPSGraph) · its GEMM           │ one at
///                 + the taps' stack [S, 12, 2560] (`encodeTaps`, in `Pages`)                      │ a time,
///     references  (img2img) the `fitted` copy when the size differs                              │ each
///                 + the Qwen-Image encoder's two heaps and host copies (`VAEEncoder.plan`)        │ handing
///     fusion      Krea2TextFusion arena(12T, r) + its CPU attentions' scores · its GEMM           │ back its
///     denoising   Krea2DiT arena(S + T, r) + the stream's 3 buffers + CacheLoRA(r)           │ memory
///                 + the GQA SDPA (queries' copy, and the slices' concatenation past `maxMatrix`)  │
///                 + the velocity · its GEMM — or, before the first evaluation, the modulation      │
///                 tabulated for every σ (the larger of the two moments)                          │
///     decoding    the Qwen-Image decoder's two heaps and the image it returns (`AnimaVAE.plan`)    ┘
///                 (a batch: the images already decoded, and every latent crossing until then)
///
///     peak = max(phases) + crossing + process base
///     S = 34 + T + 5 encoder rows, T ≤ 512 text rows, s = (H/16)(W/16) image rows, r = Σ ranks
///
/// Ours, each the function its allocator calls: `TextEncoder.arenaBytes`, `TextEncoder.stagingSlots`,
/// `Artifact.largestRun`, `Krea2TextFusion.arenaBytes`, `Krea2DiT.arenaBytes`, `CacheLoRA.bytes`,
/// `LoRA.downBytes`, `Krea2DiT.tableBytes`, `Krea2DiT.tabulationBytes`, `Attention.maskBytes`,
/// `Conductor.workspaceBytes`, `GEMM.tierBytes`, `VAEEncoder.plan`, `AnimaVAE.plan`. Outside our hand,
/// from the shapes: MPSGraph's copy of the queries (`OperatorFootprint.sdpaQueries`) and, past
/// `Attention.maxMatrix`, the slices it concatenates; the VAEs' SDPA copies (`VAELayout.outside`).
/// The frameworks' constants are not charged.
///
/// What the formula has to know and a port by analogy with Anima or Qwen-Image-2.1 would miss:
///
///   - **the map is 26 GB in bf16, more than the machine**: it streams beside the page cache, the whole
///     map, through `TailStream.unitSlots` buffers of its largest tensor (`Krea2DiT.generationTail`, Qwen-Image-2.1's
///     rule; the tensor is 868 MB at most in bf16 per layer, about half in 8 bits, a quarter in 4) — the map's type shows there and
///     in the encoder's staging, nowhere else. A map B holds whole (a 4-bit one on a larger Mac) is
///     left to the cache: its buffers are charged, not allocated;
///   - **the text is not padded**: the DiT's sequence is `s + T`, the prompt's rows and not 512
///     (`Krea2DiT`'s header), so the arena follows the prompt; the encoder sees the template too — 34
///     rows before, 5 after (`Krea2Text`), dropped from the 12 taps it hands on;
///   - **the fusion is the denoising's, not the text's** (the LoRA touches it): it runs once per render,
///     in its own arena, which dies before the DiT maps its own — two moments, never side by side.
///     Its per-layer blocks run at `12 T` rows; its `normed`, `merged`, `o` are sized at `12 T × 6144`
///     though only `T` rows of `txt_in` reach that width — charged as reserved;
///   - **what the context keeps to the render's end** (`Context.reserve`): the fused text `[T, 6144]`
///     and the modulation table, one `(temb, temb_mod)` per σ — they cross the decoding;
///   - **the modulation is tabulated for every σ at once, in double** (`Krea2DiT.tabulateModulation`):
///     its transient vectors at `Request.maximumSteps`, and a timestep weight the map does not publish in
///     fp32 widened for the call (`Krea2Exact.widenedBytes`: `time_mod_proj`, 906 MB, if a version
///     quantized it) — alive beside the arena, before any attention: the larger moment is charged;
///   - **one SDPA, GQA 48/12, without repetition**: MPSGraph's copy of the 48 heads' queries; past
///     `Attention.maxMatrix` (1024×1536) it runs by query slices and concatenates them;
///   - **a LoRA costs three things in the DiT**: the `mid [s + T, r]` slice, the `CacheLoRA` of every
///     module it may touch (the blocks' 8 Linears and the final one, ~460 MB at rank 32), and, with two
///     LoRAs or more, one `down` staged while the cache fills; in the fusion, its three slices;
///   - **the half-size start's DiT** (`Krea2DenoisingModule.denoise`, off by default, never under a LoRA)
///     is built and given back before the full one: never both. Beside it, `euler` keeps ε, the small
///     grid, its estimate and the upsampled estimate (2.5 latents) — a quarter of the image rows saves
///     far more than that in its arena — and gives them back before the first full evaluation: the
///     full DiT's phase, as charged, is the larger.
extension MemoryNeed {

    /// **What Krea 2's formula reads beside the request**: the DiT's config and tensor table, the
    /// encoder's config and tensor table (their headers, never a tensor) — each file read once, by its
    /// stamp. A header that does not parse makes no maps: the need is unknown and the render refused
    /// (`EngineError.memoryNeedUnknown`).
    package struct Krea2Maps: Sendable {
        package let dit: Krea2DiT.Config
        package let encoder: TextEncoder.Config
        /// The largest layer's staging run in the DiT's map (`Artifact.stagingBytes`): the map's type, read by the tests.
        package let ditLayerStaging: Int
        /// The largest tensor's staging run in the DiT's map (`Artifact.largestRun`): what one of the stream's
        /// buffers holds (`TailStream.unitSlots` of them).
        package let ditUnitStaging: Int
        /// The largest encoder layer's staging run: what one of its buffers holds.
        package let encoderLayerStaging: Int
        /// The largest timestep weight the tabulation widens (`Krea2Exact.linears`: a weight not
        /// published in fp32); 0 when the three are fp32.
        package let timestepWidening: Int
        /// What widening the largest rotated (int8 convrot) weight allocates beside the arenas
        /// (`Widen.rotationScratchBytes`): the Compact's encoder and DiT, the Light's encoder; 0 otherwise.
        package let encoderRotationScratch: Int, ditRotationScratch: Int

        /// From the headers — pure: the tests hand it headers written in memory. `nil` when the
        /// encoder's config does not parse or a map has no tensor table.
        package init?(dit: [String: Any], ditSize: Int = .max, encoder: [String: Any], encoderSize: Int = .max) {
            guard let encoderConfig = try? TextEncoder.Config(header: encoder) else { return nil }
            let config = Krea2DiT.Config(header: dit)
            let ditStaging = Artifact.stagingBytes(header: dit, prefixes: (0..<config.layers).map(Krea2DiT.layerPrefix),
                                                   fileSize: ditSize)
            let encoderStaging = Artifact.largestRun(header: encoder, prefixes: TextEncoder.layerPrefixes(encoderConfig),
                                                       fileSize: encoderSize)
            guard ditStaging > 0, encoderStaging > 0 else { return nil }
            self.dit = config; self.encoder = encoderConfig
            ditUnitStaging = Artifact.largestRun(header: dit, prefixes: (0..<config.layers).map(Krea2DiT.layerPrefix),
                                                 fileSize: ditSize)
            ditLayerStaging = ditStaging; encoderLayerStaging = encoderStaging
            timestepWidening = Krea2DiT.timestepWidening(header: dit, config: config)
            encoderRotationScratch = Widen.rotationScratchBytes(header: encoder, prefixes: TextEncoder.layerPrefixes(encoderConfig))
            ditRotationScratch = Widen.rotationScratchBytes(header: dit, prefixes: (0..<config.layers).map(Krea2DiT.layerPrefix))
        }

        /// The maps this card renders with: its own DiT (an imported one) or the library's installed
        /// version (Standard, Compact or Light), and that version's encoder. `nil` when one cannot be read.
        package init?(card: ModelCard, library: Library = .processWide) {
            let ditPath = card.map ?? library.map(library.ditMap(.krea2))
            let encoderPath = library.map(library.encoderMap(.krea2))
            let maps = Krea2Maps.installed(InstalledFiles(paths: [ditPath, encoderPath])) { () -> Krea2Maps? in
                guard let dit = try? Artifact.header(ditPath), let encoder = try? Artifact.header(encoderPath) else { return nil }
                return Krea2Maps(dit: dit, ditSize: InstalledFiles.size(ditPath), encoder: encoder,
                                 encoderSize: InstalledFiles.size(encoderPath))
            }
            guard let maps else { return nil }
            self = maps
        }

        /// The maps read, by their files' stamps.
        private static let installed = ShapeMemo<InstalledFiles, Krea2Maps?>(capacity: 8)
    }

    /// **Krea 2's need** on the installed maps.
    static func krea2(_ request: Request) -> MemoryNeed? {
        guard let maps = request.krea2Maps ?? Krea2Maps(card: request.card, library: request.library) else { return nil }
        return krea2(request, maps: maps)
    }

    /// The decoder's plan, walked once per latent grid (`AnimaVAE.plan` walks every pass).
    private static let krea2DecoderPlans = ShapeMemo<[Int], VAELayout>()

    /// **Krea 2's need on these maps** — pure: the tests hand it headers written in memory.
    package static func krea2(_ request: Request, maps: Krea2Maps) -> MemoryNeed {
        let dit = maps.dit, encoder = maps.encoder
        let f = MemoryLayout<Float>.size
        // The encoder's rows: the template, the prompt capped where `Krea2Text.admitted` refuses, the
        // suffix (`Request.promptTokens` bounds them by the prompt's bytes); the DiT's, without the template.
        let sequence = min(max(Krea2Text.templateTokens + 1, request.promptTokens), Krea2Text.maximumIdentifiers)
        let t = sequence - Krea2Text.templateTokens
        let taps = Krea2TextFusion.Shape.layers
        // The Qwen-Image VAE: 8 pixels per latent cell, 16 channels; the DiT patches 2×2 cells a token.
        let (latentHeight, latentWidth) = (request.height / 8, request.width / 8)
        let latent = dit.latentChannels * latentHeight * latentWidth * f
        let s = Krea2DiT.imageTokens(latentHeight: latentHeight, latentWidth: latentWidth)
        let n = s + t
        let rank = request.loraRanks.reduce(0, +)
        let (width, height) = (request.width, request.height)
        let (d, hidden, kv) = (dit.dim, dit.hidden, dit.kvWidth)

        // ── what crosses: the taps (`Conditioning`), and from the fusion on, the fused text and the
        //    modulation table (`Context.reserve`, to the render's end); the latent; img2img's image as
        //    handed and its z₀ at the render's grid ─────────────────────────────────────────────────
        let text = t * taps * encoder.hidden * f
        let kept = t * d * f + Krea2DiT.tableBytes(config: dit, count: Siliconed.Request.maximumSteps)
        let starts = request.references.reduce(0) { $0 + 3 * $1.width * $1.height * f + latent }
        let crossing = text + kept + latent + starts + batchLatents(request, latent: latent)

        // ── text: the encoder's arena, its staging buffers, its causal SDPA, its GEMM, and the taps'
        //    stack it gathers (the array kept from it is the conditioning, crossing) ─────────────────
        let (h, a, kvE, i) = (encoder.hidden, encoder.attentionWidth, encoder.kvWidth, encoder.intermediate)
        let textPhase = Phase(kind: .text, terms: [
            Term("encoder arena", TextEncoder.arenaBytes(config: encoder, sequence: sequence)),
            Term("encoder staging", TextEncoder.stagingSlots(granted: request.encoderSlots) * maps.encoderLayerStaging),
            Term("taps stack", Arena.capacity(for: sequence * taps * h * f)),
            Term("encoder rotation scratch", maps.encoderRotationScratch),
        ] + attention("encoder", heads: encoder.heads, rows: sequence, keys: sequence, headDim: encoder.headDim,
                      causal: true, split: false, threads: request.threads)
          + gemm("encoder", bounded: [(sequence, h, a), (sequence, h, kvE), (sequence, a, h), (sequence, h, i),
                                      (sequence, i, h)]))

        // ── img2img's starting image, encoded at the render's format (`fitted`: a copy when the size
        //    handed differs): the Qwen-Image encoder's heaps ──────────────────────────────────────
        let references = request.references.map { image in
            Phase(kind: .references, terms: [
                Term("fitted copy", image.width == width && image.height == height ? 0 : 3 * width * height * f),
            ] + vae("Qwen-Image encoder", VAEEncoder.plan(family: .qwenImage, height: height, width: width)))
        }

        // ── the fusion, its arena gone before the DiT's: its scores on the CPU, its GEMM ────────────
        let fusion = Krea2TextFusion.Shape.self
        let rows = taps * t
        let (fd, fh) = (fusion.dim, fusion.hidden)
        let fusionPhase = Phase(kind: .denoising, terms: [
            Term("fusion arena", Krea2TextFusion.arenaBytes(outDim: d, maxTokens: t, rank: rank)),
            Term("fusion scores", Krea2TextFusion.scoreBytes(tokens: t)),
            Term("LoRA down staging", request.loraRanks.count > 1 ? LoRA.downBytes(k: fh, rank: request.loraRanks.max() ?? 0) : 0),
        ] + gemm("fusion", bounded: [(rows, fd, fd), (rows, fd, fh), (rows, fh, fd), (t, fd, fd), (t, fd, fh), (t, fh, fd),
                                     (t, fd, d), (t, d, d)],
                 rank: rank, ane: request.neuralEngine))

        // ── the DiT: its arena, the map's staging blocks, the LoRA cache, the velocity; then either the
        //    modulation's tabulation or the SDPA and the GEMM, the larger ─────────────────────────────
        let modules = Krea2DiT.loraModules(config: dit)
        let streamed = request.tail.fraction > 0 && request.tail.streamed
        var terms = [
            Term("DiT arena", Krea2DiT.arenaBytes(config: dit, latentHeight: latentHeight, latentWidth: latentWidth,
                                                  maxText: t, rank: rank)),
            Term("map staging", streamed ? TailStream.unitSlots * maps.ditUnitStaging : 0),
            Term("LoRA cache", rank > 0 ? CacheLoRA.arenaBytes(CacheLoRA.bytes(modules: modules.map { ($0.k, $0.n, rank) })) : 0),
            Term("LoRA down staging", request.loraRanks.count > 1
                ? LoRA.downBytes(k: modules.map(\.k).max() ?? 0, rank: request.loraRanks.max() ?? 0) : 0),
            // The velocity `forward` returns (`[16, h, w]`); Euler keeps no history.
            Term("sampler", latent + request.solver.historyBytes(count: latent / f)),
            Term("DiT rotation scratch", maps.ditRotationScratch),
        ]
        // Under the head split, MPSGraph sees the GPU's groups only: its queries' copy and its slices are theirs.
        let split = Krea2DiT.headSplit(sequence: n, heads: dit.heads, headDim: dit.headDim)
        let running = MemoryNeed.attention("DiT", heads: dit.heads, rows: n, keys: n, headDim: dit.headDim, causal: false,
                                           split: split, threads: request.threads, group: dit.heads / dit.kvHeads)
            // The image's rows are exact; the sequence's follow the prompt's bound.
            + gemm("DiT", products: [(s, dit.channels, d), (s, d, dit.channels)],
                   bounded: [(n, d, d), (n, d, kv), (n, d, hidden), (n, hidden, d)], rank: rank, ane: request.neuralEngine)
        terms += running
        // The tabulation runs right after the DiT is built and before its first evaluation: the arena is
        // mapped and not yet written (an untouched page is not resident) and the stream does not exist
        // yet. It is a moment of its own — the tabulation, the widened timestep weight, what a LoRA keeps —
        // and the phase is the larger of the two, not their sum.
        let tabulating = Krea2DiT.tabulationBytes(config: dit, count: Siliconed.Request.maximumSteps, rank: rank)
            + maps.timestepWidening + terms.filter { $0.name.hasPrefix("LoRA") }.reduce(0) { $0 + $1.bytes }
        terms.append(Term("modulation tabulation beyond the evaluation",
                          max(0, tabulating - terms.reduce(0) { $0 + $1.bytes })))
        let denoising = Phase(kind: .denoising, terms: terms)

        // ── decoding: the decoder's two heaps and the image it returns; the latent it reads crosses ──
        let plan = krea2DecoderPlans([latentHeight, latentWidth]) {
            AnimaVAE.plan(latentHeight: latentHeight, latentWidth: latentWidth)
        }
        let decoding = Phase(kind: .decoding, terms: vae("Qwen-Image decoder", plan) + [batchImages(request)])

        return MemoryNeed(phases: [textPhase] + references + [fusionPhase, denoising, decoding], crossing: crossing,
                          base: request.base)
    }
}

// ── the sizes, one source: the constructors allocate them, `MemoryNeed.krea2` charges them ────

extension Krea2Text {
    /// The suffix's tokens (`<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`): what `identifiers`
    /// appends after the cut.
    package static let suffixTokens = 5
    /// The most identifiers a render encodes (`admitted`): the template, the prompt at its bound, the suffix.
    package static var maximumIdentifiers: Int { templateTokens + Request.maximumPromptTokens + suffixTokens }
}

extension Krea2DiT {
    /// A layer's tensors in the map: what a staging block holds (`MemoryPlan.ditTail`, `Krea2Maps`).
    package static func layerPrefix(_ layer: Int) -> String { "transformer_blocks.\(layer)." }

    /// **The whole map streams beside the page cache**, as Qwen-Image-2.1's generation does
    /// (`QwenImage21DiT.generationTail`): 26 GB of bf16 swept at every evaluation would miss every
    /// cached page of a 16 GB Mac, and push the others' memory into the compressor. A map B holds whole
    /// beside the denoising is left to the cache (`MemoryPlan.released`). Which bytes go through the
    /// cache, never the bytes.
    package static let generationTail = 1.0

    /// The image's tokens: one per 2×2 latent cells.
    package static func imageTokens(latentHeight: Int, latentWidth: Int) -> Int { (latentHeight / 2) * (latentWidth / 2) }

    /// **The DiT's slices, in floats** — what `init` reserves, each on its page: at `n = s + T` rows
    /// `x`, `normed`, `q`, `gate`, `merged`, `o` (`d`), `k`, `v` (`kvHeads · headDim`), `h1`, `h2`
    /// (`hidden`), the RoPE table (`headDim`); the six modulation vectors; one reserve for the largest
    /// weight (`d · hidden`); the patches and the final projection at the image's rows; under a LoRA,
    /// its `mid [n, r]`.
    package static func slices(config: Config, imageTokens s: Int, maxText: Int, rank: Int) -> [(name: String, count: Int)] {
        let n = s + maxText, d = config.dim, h = config.hidden, kv = config.kvWidth
        return [("x", n * d), ("normed", n * d), ("q", n * d), ("k", n * kv), ("v", n * kv), ("gate", n * d),
                ("merged", n * d), ("o", n * d), ("h2", n * h), ("modulation", 6 * d), ("reserve", d * h),
                ("tokens", s * config.channels), ("projected", s * config.channels), ("freqs", n * config.headDim)]
            + (rank > 0 ? [("loraMid", n * rank)] : [])
    }

    /// **The arena `init` maps** for this grid, these text rows and this cumulative rank.
    package static func arenaBytes(config: Config, latentHeight: Int, latentWidth: Int, maxText: Int, rank: Int) -> Int {
        Arena.capacity(slices: slices(config: config, imageTokens: imageTokens(latentHeight: latentHeight, latentWidth: latentWidth),
                                      maxText: maxText, rank: rank).map { $0.count * 4 })
    }

    /// **Every module a LoRA may touch in the DiT** (`CacheLoRA`'s retained ones, `[k, n]` as the map
    /// lays them out): `loraTargets`, the list the cache reads — the worst case of a stack.
    package static func loraModules(config: Config) -> [(k: Int, n: Int)] {
        loraTargets(config: config).map { (k: $0.k, n: $0.n) }
    }

    /// **The modulation table the context keeps** (`ModulationTable`): per σ, `temb [d]` and
    /// `temb_mod [6d]` in fp32.
    package static func tableBytes(config: Config, count: Int) -> Int { count * 7 * config.dim * 4 }

    /// **What the tabulation of `count` σ holds at once, in double** (`modulation(sigmas:)`), at its
    /// widest — `time_mod_proj`'s call: the hidden vectors, `temb`, its activation, the inputs flattened,
    /// the outputs accumulated and returned (`Krea2Exact.linears`), then the fp32 table they become;
    /// under a LoRA, that call's `down` and `up` (`Krea2Exact.loraBytes`).
    package static func tabulationBytes(config: Config, count: Int, rank: Int = 0) -> Int {
        let d = config.dim
        return count * (4 * d + 2 * 6 * d) * 8 + tableBytes(config: config, count: count)
            + Krea2Exact.loraBytes(k: d, n: 6 * d, rank: rank)
    }

    /// The timestep weights `modulation(sigmas:)` reads, `[input, output]`.
    static func timestepWeights(_ config: Config) -> [(name: String, inputs: Int, outputs: Int)] {
        [("time_embed.linear_1.weight", config.timeDim, config.dim), ("time_embed.linear_2.weight", config.dim, config.dim),
         ("time_mod_proj.weight", config.dim, 6 * config.dim)]
    }

    /// **The largest timestep weight a map makes `Krea2Exact.linears` widen**: one not published in
    /// fp32 there (read in place otherwise) — its `[input, output]` in fp32 for the call. A weight the
    /// header does not list is charged as if widened.
    package static func timestepWidening(header: [String: Any], config: Config) -> Int {
        let table = header["tensors"] as? [String: [String: Any]] ?? [:]
        return timestepWeights(config).map { weight -> Int in
            guard let entry = table[weight.name], entry["dtype"] as? String == Artifact.DType.float32.rawValue else {
                let shape = (table[weight.name]?["shape"] as? [Int]).flatMap { $0.count == 2 ? $0 : nil }
                return Krea2Exact.widenedBytes(inputs: shape?[0] ?? weight.inputs, outputs: shape?[1] ?? weight.outputs)
            }
            return 0
        }.max() ?? 0
    }
}

extension Krea2TextFusion {
    /// The fusion's own shapes — not the DiT's config: 2560 wide (Qwen3-VL-4B's), 20 heads of 128, a
    /// SwiGLU of 6912, the encoder's 12 taps.
    package enum Shape {
        package static let dim = 2560, heads = 20, headDim = 128, hidden = 6912, layers = 12
    }

    /// **The fusion's slices, in floats**, as `init` reserves them, at `12 T` rows: `x`, `q`, `k`, `v`,
    /// `gate` at 2560; `normed`, `merged`, `o` at the wider of 2560 and the DiT's 6144 (`txt_in` writes
    /// them at `T` rows); `h1`, `h2`, the SwiGLU's scratch at 6912; one reserve for the widest weight
    /// (`txt_in.linear_2`, 6144²); under a LoRA, its `down`, `up` and `mid`.
    package static func slices(outDim: Int, maxTokens: Int, rank: Int) -> [(name: String, count: Int)] {
        let r = maxTokens * Shape.layers, d = Shape.dim, h = Shape.hidden, wide = max(d, outDim)
        let widest = max(outDim * outDim, d * outDim, d * h)
        return [("x", r * d), ("normed", r * wide), ("q", r * d), ("k", r * d), ("v", r * d), ("gate", r * d),
                ("merged", r * wide), ("o", r * wide), ("h1", r * h), ("h2", r * h), ("scratch", r * h), ("reserve", widest)]
            + (rank > 0 ? [("loraBas", h * rank), ("loraHaut", rank * h), ("loraMid", r * rank)] : [])
    }

    /// **The arena `init` maps** for `maxTokens` text rows and this cumulative rank.
    package static func arenaBytes(outDim: Int, maxTokens: Int, rank: Int) -> Int {
        Arena.capacity(slices: slices(outDim: outDim, maxTokens: maxTokens, rank: rank).map { $0.count * 4 })
    }

    /// **The scores of the CPU attentions** (`Krea2Ops.attention`): one `[L, L]` array per (sequence,
    /// head) task — every task's at once, the bound: the per-layer blocks' `T · 20` tasks of 12², or the
    /// refiners' 20 of `T²`, never both.
    package static func scoreBytes(tokens t: Int) -> Int {
        max(t * Shape.heads * Shape.layers * Shape.layers, Shape.heads * t * t) * 4
    }
}
