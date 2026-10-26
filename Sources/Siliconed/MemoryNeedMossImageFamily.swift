import Foundation

/// **MOSS-Image's need, phase by phase** — the family's formula, which composes the three constructors' own
/// shape functions: the DiT's (`MemoryNeed.mossImageDenoising`, its file), the encoder's and the decoder's.
///
///     crossing: the encoded text [T, 4096] · the modulation table (kept by the context to the render's end)
///               · latent [128, H/16, W/16]
///
///     text        Qwen35Encoder arena(T) + its staging units × the largest tensor of the stream
///                 + the full layers' causal SDPA (mask T·T ⊕ queries' copy) · its GEMM
///     denoising   MossImageDiT arena(S + T, r) + the stream's units + CacheLoRA(r) … (`mossImageDenoising`)
///     decoding    the FLUX.2 decoder's two heaps and the image it returns (`Flux2VAE.plan`)
///
///     peak = max(phases) + crossing + process base
///
/// What the formula has to know and a port by analogy with Krea 2 would miss:
///
///   - **the encoder is 16.8 GB in bf16 — the language model alone, with its norms in fp32, 16 GB —, more than B**:
///     it streams layer by layer, tensor by tensor (`Artifact.streamUnits(ordered:)`), through `TailStream.unitSlots`
///     buffers of its largest tensor (the 12288 × 4096 MLP matrices, 100 MB) — the regime of Qwen-Image-2.1's;
///   - **the DiT map is 10.2 GB**, streamed too (`MossImageDiT.generationTail`), unless B holds it;
///   - **the DMD LoRA is always there**: rank 128 in fp32, 1.19 GB once cached (`CacheLoRA`), whatever the
///     request carries. The formula takes its rank from the LoRA's own map (`MossMaps.turboRank`) and puts it
///     before any rank the request adds — the stack the denoiser opens is `[turbo] + request.loras`;
///   - **the text is not padded and not cut at 512**: the encoder's rows are the prompt's tokens and the template's
///     ten, the DiT's text rows the same (`Qwen35Prompt`); `Request.promptTokens` bounds them by the prompt's bytes.
extension MemoryNeed {

    /// **What MOSS-Image's formula reads beside the request**: the DiT's header and size, the encoder's config and
    /// its staging unit, the LoRA's rank — headers only, never a tensor. A header that does not parse makes no
    /// maps: the need is unknown and the render refused (`EngineError.memoryNeedUnknown`).
    package struct MossMaps: @unchecked Sendable {
        package let ditHeader: [String: Any]
        package let ditSize: Int
        package let dit: MossImageDiT.Config
        package let encoderConfig: Qwen35Encoder.Config
        /// The largest tensor's staging length in the encoder's layers (`Artifact.largestRun`): one stream buffer.
        package let encoderUnit: Int
        /// The LoRA map's rank (its `rang`), the one every render opens.
        package let turboRank: Int

        package init?(dit: [String: Any], ditSize: Int = .max, encoder: [String: Any], encoderSize: Int = .max, turboRank: Int) {
            guard let encoderConfig = try? Qwen35Encoder.Config(header: encoder), turboRank > 0 else { return nil }
            let config = MossImageDiT.Config(header: dit)
            guard Artifact.largestRun(header: dit, prefixes: (0..<config.layers).map(MossImageDiT.layerPrefix), fileSize: ditSize) > 0
            else { return nil }
            let unit = Artifact.largestRun(header: encoder, prefixes: (0..<encoderConfig.layers).map { "layers.\($0)." },
                                           fileSize: encoderSize)
            guard unit > 0 else { return nil }
            ditHeader = dit; self.ditSize = ditSize; self.dit = config; self.encoderConfig = encoderConfig
            encoderUnit = unit; self.turboRank = turboRank
        }

        /// The maps this card renders with: the library's installed Standard — DiT, encoder, the DMD LoRA.
        package init?(card: ModelCard, library: Library = .processWide) {
            let ditPath = card.map ?? library.map(library.ditMap(.mossImage))
            let encoderPath = library.map(library.encoderMap(.mossImage))
            let turboPath = Installer(library: library).companionMap(.mossImage) ?? ""
            let maps = MossMaps.installed(InstalledFiles(paths: [ditPath, encoderPath, turboPath])) { () -> MossMaps? in
                guard let dit = try? Artifact.header(ditPath), let encoder = try? Artifact.header(encoderPath),
                      let turbo = try? Artifact.header(turboPath), let rank = turbo["rang"] as? Int else { return nil }
                return MossMaps(dit: dit, ditSize: InstalledFiles.size(ditPath), encoder: encoder,
                                encoderSize: InstalledFiles.size(encoderPath), turboRank: rank)
            }
            guard let maps else { return nil }
            self = maps
        }

        private static let installed = ShapeMemo<InstalledFiles, MossMaps?>(capacity: 8)
    }

    /// **MOSS-Image's need** on the installed maps.
    static func mossImage(_ request: Request) -> MemoryNeed? {
        guard let maps = request.mossMaps ?? MossMaps(card: request.card, library: request.library) else { return nil }
        return mossImage(request, maps: maps)
    }

    /// The decoder's plan is memoized by `Flux2VAE.plan` itself.

    /// **MOSS-Image's need on these maps** — pure: the tests hand it headers written in memory.
    package static func mossImage(_ request: Request, maps: MossMaps) -> MemoryNeed? {
        let encoder = maps.encoderConfig, dit = maps.dit
        let f = MemoryLayout<Float>.size
        // The rows: the template's and the prompt's, up to where the reference cuts (`Qwen35Prompt.maxTokens`) and
        // the app refuses (`Request.maximumPromptTokens`, past which the text module throws before the encoder).
        let sequence = min(max(1, request.promptTokens), Qwen35Prompt.maxTokens)
        let t = sequence
        let (gridHeight, gridWidth) = (request.height / 16, request.width / 16)
        let latent = dit.channels * gridHeight * gridWidth * f

        let crossing = t * dit.textDim * f + MossImageDiT.tableBytes(config: dit, count: Siliconed.Request.maximumSteps)
            + latent + batchLatents(request, latent: latent)

        // ── text: the encoder's arena, its stream's buffers, the full layers' SDPA, its GEMM ───────────
        let (h, i, a, kv) = (encoder.hidden, encoder.intermediate, encoder.attentionWidth, encoder.kvWidth)
        let channels = encoder.convChannels, valueWidth = encoder.valueWidth
        let text = Phase(kind: .text, terms: [
            Term("encoder arena", Qwen35Encoder.arenaBytes(config: encoder, sequence: sequence)),
            Term("encoder staging", TextEncoder.stagingSlots(granted: request.encoderSlots) * maps.encoderUnit),
        ] + attention("encoder", heads: encoder.heads, rows: sequence, keys: sequence, headDim: encoder.headDim,
                      causal: true, split: false, threads: request.threads)
          + gemm("encoder", bounded: [(sequence, h, channels), (sequence, h, valueWidth), (sequence, valueWidth, h),
                                      (sequence, h, 2 * a), (sequence, h, kv), (sequence, a, h),
                                      (sequence, h, i), (sequence, i, h)]))

        // ── denoising: the DiT's own formula, under the stack the denoiser opens: the turbo LoRA, then the request's ──
        var denoisingRequest = request
        denoisingRequest.loraRanks = [maps.turboRank] + request.loraRanks
        guard let denoising = mossImageDenoising(denoisingRequest, textRows: t, header: maps.ditHeader, fileSize: maps.ditSize,
                                                 velocities: 0) else { return nil }

        // ── decoding: the decoder's two heaps and the image it returns; the latent it reads crosses ──────
        let band = request.minimumBands ? min(VAE.bandElements, MemoryPlan.fluxMinimumBand) : VAE.bandElements
        let decoding = Phase(kind: .decoding,
                             terms: vae("FLUX.2 decoder", Flux2VAE.plan(latentHeight: request.height / 8, latentWidth: request.width / 8,
                                                                       band: band)) + [batchImages(request)])
        return MemoryNeed(phases: [text, denoising, decoding], crossing: crossing, base: request.base)
    }
}
