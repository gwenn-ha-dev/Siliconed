import Foundation

/// **Z-Image's need, phase by phase** — nothing but the constructors' shape functions.
///
///     crossing: caps [T, 2560] · latent [16, H/8, W/8] · (img2img) the image as handed, and z₀
///
///     text        TextEncoder arena(T) + its staging buffers × the largest encoder layer          ┐
///                 + causal SDPA: mask T·T (ours) ⊕ queries' copy (MPSGraph) · its GEMM           │ one at
///     references  (img2img, Détail) the `fitted` copy when the size differs                     │ a time,
///                 + the Flux encoder's two heaps and host copies (`VAEEncoder.plan`)              │ each
///     denoising   4 DiT arenas(S, r) + CacheLoRA(r) + the tail's buffers + the sampler's latents │ handing
///                 + the full-size attentions the 3 blocks keep (a head split from 1024 rows:     │ back its
///                   host arena ⊕ the GPU heads' queries; MPSGraph below) · its GEMM               │ memory
///     decoding    the Flux decoder's two heaps and the image it returns (`VAE.plan`)              ┘
///                 (a batch: the images already decoded, and every latent crossing until then)
///
///     peak = max(phases) + crossing + process base     S = ⌈(H/16)(W/16)⌉₃₂ + ⌈T⌉₃₂, r = Σ ranks
///
/// Ours, each the function its allocator calls: `TextEncoder.arenaBytes`, `TextEncoder.stagingSlots`,
/// `Artifact.largestRun`, `DiT.arenaBytes`, `CacheLoRA.bytes`, `LoRA.downBytes`,
/// `HeadSplitAttention.hostBytes`, `Attention.maskBytes`, `Conductor.workspaceBytes`, `GEMM.tierBytes`,
/// `VAEEncoder.plan`, `VAE.plan`. Outside our hand, from the shapes: MPSGraph's copy of the queries
/// (`OperatorFootprint.sdpaQueries`), the VAE's SDPA copies (`VAELayout.outside`); the process's base is
/// the request's, read from the process (`Request.base`). The frameworks' constants are not charged.
///
/// What the formula has to know and a port by analogy with Qwen-Image-2.1 would miss:
///
///   - **the prompt enters twice**: `T` rows in the encoder (its arena and a causal SDPA whose mask is
///     `T²`, in our pages), and `⌈T⌉₃₂` caption rows in every DiT block — the caption padding
///     is the DiT's, the encoder computes exactly `T` positions (cut at 512);
///   - **the tokenizer is not in the text phase's peak**: `ZImageTextModule` drops it in its own scope
///     before the encoder maps its arena (`tableBytes`, ~20 MB, never beside it);
///   - **the tail's buffers are the map's, not the model's**: their size is the largest streamed
///     layer's run in the file (362 MB for the bf16 map, about half for the 8-bit Compact), read from
///     the map's header — the card's type is there and nowhere else;
///   - **one attention per block, at its one length**: every evaluation runs at the full grid (Z-Image
///     has no half-size steps, unlike Qwen-Image-2.1), so each block keeps a single head split or graph;
///   - **a LoRA costs three things**: the `mid [S, r]` slice in each block's arena and the DiT's own, the
///     `CacheLoRA` (every module a stack can touch, `DiT.loraTargets` — the worst case), and, with more than one LoRA, the `down`
///     staging of one module while the cache fills (`LoRA.downBytes`);
///   - **the VAEs hand everything back before they return** (two heaps of ours; `finish` waits until
///     Metal's ledger is back to its level before them, under 0.1 s): the encoder's memory is not
///     carried into the denoising any more, nor the DiT's into the decoding (`Context.keptDenoiser`
///     released first);
///   - Z-Image has no reference image: `references` are img2img's (and Détail's) starting image as
///     handed to the engine, kept by the request through the render (crossing), encoded at the render's
///     format (`fitted`, a copy when the size differs) before the DiT; its z₀ crosses at the render's grid.
extension MemoryNeed {

    /// **What Z-Image's formula reads beside the request**: its maps' headers — configs and tensor
    /// tables, a few hundred KB, never a tensor. The largest tensor's run of each layer is read once,
    /// here (`Artifact.largestRun` bridges the whole tensor table at each call, 2 ms a map in a debug
    /// build); a set of layers holds the largest of theirs.
    ///
    /// `@unchecked Sendable`: the headers are JSON values, read once and never written.
    package struct ZImageMaps: @unchecked Sendable {
        package let dit: [String: Any]
        package let ditSize: Int
        package let encoder: [String: Any]
        package let encoderSize: Int
        /// `Artifact.largestRun` of each DiT layer alone: what a stream's buffer holds if that layer is streamed.
        private let ditLayerRun: [Int]
        /// `Artifact.largestRun` of the encoder's layers (0 when its config does not parse).
        package let encoderLayerStaging: Int
        /// What widening the DiT's largest int8 convrot weight allocates outside every arena
        /// (`Widen.rotationScratchBytes`): an imported Comfy-style convrot DiT; 0 otherwise.
        package let ditRotationScratch: Int

        package init(dit: [String: Any], ditSize: Int = .max, encoder: [String: Any], encoderSize: Int = .max) {
            self.dit = dit; self.ditSize = ditSize; self.encoder = encoder; self.encoderSize = encoderSize
            ditLayerRun = (0..<DiT.Config(header: dit).layers).map {
                Artifact.largestRun(header: dit, prefixes: [DiT.layerPrefix($0)], fileSize: ditSize)
            }
            encoderLayerStaging = (try? TextEncoder.Config(header: encoder)).map { config in
                Artifact.largestRun(header: encoder, prefixes: TextEncoder.layerPrefixes(config), fileSize: encoderSize)
            } ?? 0
            ditRotationScratch = Widen.rotationScratchBytes(header: dit)
        }

        /// What one buffer of the DiT's stream holds over `layers`: the largest tensor's run (`Artifact.largestRun`).
        package func ditUnitStaging(_ layers: Range<Int>) -> Int { ditLayerRun[layers].max() ?? 0 }

        /// The maps this card renders with: its own DiT (an imported one) or the library's installed
        /// version (Standard, Compact or Light), and that version's encoder. `nil` when they are not
        /// installed — the render is refused before, by the preflight's first check. Each file is
        /// read once: kept by its path, size, date and inode (`installed`), read again when replaced.
        package init?(card: ModelCard, library: Library = .processWide) {
            let ditPath = card.map ?? library.map(library.ditMap(.zImage))
            let encoderPath = library.map(library.encoderMap(.zImage))
            let key = InstalledFiles(paths: [ditPath, encoderPath])
            let maps = ZImageMaps.installed(key) { () -> ZImageMaps? in
                guard let dit = try? Artifact.header(ditPath), let encoder = try? Artifact.header(encoderPath) else { return nil }
                return ZImageMaps(dit: dit, ditSize: InstalledFiles.size(ditPath), encoder: encoder,
                                  encoderSize: InstalledFiles.size(encoderPath))
            }
            guard let maps else { return nil }
            self = maps
        }

        /// The maps read, by their files' stamps.
        private static let installed = ShapeMemo<InstalledFiles, ZImageMaps?>(capacity: 8)
    }

    /// **The files a formula reads, by their stamps** — path, size, date and inode: a file replaced
    /// (reinstalled, another version) is read again, an unchanged one never twice.
    package struct InstalledFiles: Hashable, Sendable {
        var paths: [String]
        var stamps: [Int]
        init(paths: [String]) {
            self.paths = paths
            stamps = paths.flatMap { path -> [Int] in
                var info = stat()
                guard stat(path, &info) == 0 else { return [-1] }
                return [Int(info.st_size), Int(info.st_mtimespec.tv_sec), Int(info.st_mtimespec.tv_nsec), Int(info.st_ino)]
            }
        }
        static func size(_ path: String) -> Int {
            ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? NSNumber)?.intValue ?? .max
        }
    }

    /// **Z-Image's need** on the installed maps.
    static func zImage(_ request: Request) -> MemoryNeed? {
        guard let maps = request.zImageMaps ?? ZImageMaps(card: request.card, library: request.library) else { return nil }
        return zImage(request, maps: maps)
    }

    /// **Z-Image's need on these maps** — pure: the tests hand it headers written in memory.
    package static func zImage(_ request: Request, maps: ZImageMaps) -> MemoryNeed? {
        let dit = DiT.Config(header: maps.dit)
        guard let encoder = try? TextEncoder.Config(header: maps.encoder) else { return nil }
        let f = MemoryLayout<Float>.size
        // `Tokenizer.admitChat` refuses a longer prompt: the encoder never sees more.
        let tokens = min(max(1, request.promptTokens),
                         Request.promptTokens(text: Siliconed.Request.maximumPromptTokens, family: .zImage))
        // The Flux VAE: 8 pixels per latent cell, `dit.channels` (16) per cell.
        let (latentHeight, latentWidth) = (request.height / 8, request.width / 8)
        let latent = dit.channels * latentHeight * latentWidth * f
        let rank = request.loraRanks.reduce(0, +)
        let (width, height) = (request.width, request.height)

        // ── what crosses: the caption features (`Conditioning`), the latent, img2img's image as handed
        //    (kept by the request through the render) and its z₀ at the render's grid ────────────
        let caps = tokens * encoder.hidden * f
        let starts = request.references.reduce(0) { $0 + 3 * $1.width * $1.height * f + latent }
        let crossing = caps + latent + starts + batchLatents(request, latent: latent)

        // ── text: the encoder's arena, its staging buffers, its causal SDPA, its GEMM ─────────────
        let encoderLayer = maps.encoderLayerStaging
        let (h, a, kv, i) = (encoder.hidden, encoder.attentionWidth, encoder.kvWidth, encoder.intermediate)
        let text = Phase(kind: .text, terms: [
            Term("encoder arena", TextEncoder.arenaBytes(config: encoder, sequence: tokens)),
            Term("encoder staging", TextEncoder.stagingSlots(granted: request.encoderSlots) * encoderLayer),
        ] + attention("encoder", heads: encoder.heads, rows: tokens, keys: tokens, headDim: encoder.headDim,
                      causal: true, split: false, threads: request.threads)
          + gemm("encoder", bounded: [(tokens, h, a), (tokens, h, kv), (tokens, a, h), (tokens, h, i), (tokens, i, h)]))

        // ── img2img's starting image, encoded at the render's format (`fitted`: a copy when the size
        //    handed differs), one encoder at a time: its heaps and the latent it returns ─────────
        let references = request.references.map { image in
            Phase(kind: .references, terms: [
                Term("fitted copy", image.width == width && image.height == height ? 0 : 3 * width * height * f),
            ] + vae("Flux encoder", VAEEncoder.plan(family: .flux, height: height, width: width)))
        }

        // ── denoising ────────────────────────────────────────────────────────────────────────────
        let capPadded = DiT.capPadded(tokens, config: dit)
        let released = request.tail.released(layers: dit.layers)
        let streamed = released < dit.layers && request.tail.streamed
        let staging = !streamed ? 0 : TailStream.unitSlots * maps.ditUnitStaging(released..<dit.layers)
        // The attentions alive at the peak: one per block at its full length (image refiners at the
        // image's tokens, layers at the unified sequence, context refiners at the padded caption) —
        // cut by heads where `HeadSplitAttention.applies`, MPSGraph otherwise.
        let headDim = dit.dim / dit.heads
        // The image's rows padded to 32 like the caption's (`x_pad_token`): 3 968 at 832×1216, not 3 952.
        let image = DiT.imagePadded(latentHeight: latentHeight, latentWidth: latentWidth, config: dit)
        let sequence = image + capPadded
        let attentions = [("image refiners", image), ("layers", sequence), ("context refiners", capPadded)].flatMap {
            attention($0.0, heads: dit.heads, rows: $0.1, keys: $0.1, headDim: headDim, causal: false,
                      split: HeadSplitAttention.applies(sequence: $0.1, heads: dit.heads, headDim: headDim),
                      threads: request.threads)
        }
        let (d, hidden) = (dit.dim, dit.hidden), inDim = dit.patch * dit.patch * dit.channels
        let loraModules = DiT.loraModules(config: dit)
        let denoising = Phase(kind: .denoising, terms: [
            Term("DiT arenas", DiT.arenaBytes(config: dit, latentHeight: latentHeight, latentWidth: latentWidth,
                                              capLength: tokens, loraRank: rank)),
            Term("LoRA cache", rank > 0
                ? CacheLoRA.arenaBytes(CacheLoRA.bytes(modules: loraModules.map { ($0.k, $0.n, rank) })) : 0),
            Term("LoRA down staging", request.loraRanks.count > 1
                ? LoRA.downBytes(k: loraModules.map(\.k).max() ?? 0, rank: request.loraRanks.max() ?? 0) : 0),
            Term("map staging", staging),
            // The DiT's velocity `[C, h, w]`.
            Term("sampler", latent + request.solver.historyBytes(count: latent / f)),
            Term("DiT rotation scratch", maps.ditRotationScratch),
        ] + attentions
          // The image's rows are exact; the caption's, and the unified sequence's, follow the tokens' bound.
          + gemm("DiT", products: [(image, inDim, d)],
                 bounded: [(sequence, d, d), (sequence, d, hidden), (sequence, hidden, d), (capPadded, dit.capDim, d),
                           (sequence, d, inDim)],
                 rank: rank, ane: request.neuralEngine))

        // ── decoding: the decoder's two heaps and the image it returns; the latent it reads crosses ──
        let band = request.minimumBands ? min(VAE.bandElements, MemoryPlan.fluxMinimumBand) : VAE.bandElements
        let decoding = Phase(kind: .decoding,
                             terms: vae("Flux decoder", VAE.plan(latentHeight: latentHeight, latentWidth: latentWidth,
                                                                 band: band)) + [batchImages(request)])

        return MemoryNeed(phases: [text] + references + [denoising, decoding], crossing: crossing, base: request.base)
    }
}
