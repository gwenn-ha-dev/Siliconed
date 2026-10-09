import Accelerate
import Foundation

/// The Z-Image Turbo DiT, from the latent to the predicted velocity.
///
///     latent [C, H, W] ─ patchify ─ x_embedder ─ noise_refiner ×2 ─┐
///                                                                  ├─ concat ─ layers ×30 ─ FinalLayer ─ unpatchify
///     cap_feats [L, 2560] ─ cap_embedder ─ pad ─ context_refiner ×2┘
///
/// The image comes **first** in the unified sequence. The `context_refiner`s have no
/// modulation. SiLU only exists in `FinalLayer`. The RoPE positions of axis 0 are a segment
/// index: the text occupies `1…L`, the image the value `L+1`.
package final class DiT {
    package struct Config {
        package let dim, heads, hidden, layers, refiners, patch, channels, capDim, adalnDim: Int
        package let eps, theta, timeScale: Float
        package let axesDims, axesLens: [Int]
        package let seqMultiple = 32

        package init(header: [String: Any]) {
            let c = header["config"] as? [String: Any] ?? [:]
            dim = c["dim"] as? Int ?? 3840
            heads = c["n_heads"] as? Int ?? 30
            hidden = Int(Double(dim) / 3 * 8)
            layers = c["n_layers"] as? Int ?? 30
            refiners = c["n_refiner_layers"] as? Int ?? 2
            patch = 2
            channels = c["in_channels"] as? Int ?? 16
            capDim = c["cap_feat_dim"] as? Int ?? 2560
            adalnDim = 256
            eps = Float(c["norm_eps"] as? Double ?? 1e-5)
            theta = Float(c["rope_theta"] as? Double ?? 256)
            timeScale = Float(c["t_scale"] as? Double ?? 1000)
            axesDims = (c["axes_dims"] as? [Int]) ?? [32, 48, 48]
            axesLens = (c["axes_lens"] as? [Int]) ?? [1536, 512, 512]
        }
    }

    package let config: Config
    package private(set) var boundaries: [String: [Float]] = [:]
    package var recordBoundaries = false
    /// The current render's token, consulted between two layers (see `Cancellation`).
    package var cancellation: Cancellation?
    /// **The ablation of a prompt attribute**: the caption positions to replace with the learned
    /// padding token. A measurement instrument, never a render one — it changes the image, and it
    /// announces itself on the error output. Empty by default. See `forward`.
    package var capsDrop: [Int] = []
    /// This request's LoRA stack, or `nil`. Exposed so that the caller can announce it:
    /// a LoRA believed applied that is not returns a plausible and wrong image.
    package let lora: LoRA?
    /// To isolate what the prefetcher costs the GPU: it runs on four threads and touches pages
    /// continuously during the GEMMs.
    package var prefetchEnabled: Bool {
        get { prefetcher.enabled }
        set { prefetcher.enabled = newValue }
    }

    private let artifact: Artifact
    private let prefetcher: Prefetcher
    private let gemm: GEMM
    private let rope: RopeTables
    private let arena: Arena
    /// The time per phase, aggregated over the three sets of blocks.
    package var timings: [String: Double] {
        var total: [String: Double] = [:]
        for block in [imageBlock, capBlock, unifiedBlock] {
            for (phase, seconds) in block.timings { total[phase, default: 0] += seconds }
        }
        return total
    }
    package func resetTimings() { for b in [imageBlock, capBlock, unifiedBlock] { b.resetTimings() } }
    package var census: [String: (count: Int, seconds: Double, best: Double)] { gemm.census }
    /// What the per-token cut returned, when it is on — `nil` otherwise, and that is the
    /// difference between "it did nothing" and "it was not there".
    package var conductor: Conductor? { gemm.conductor }
    package var widener: WidenGPU? { gemm.widener }
    package var traceGEMM: Int { get { gemm.trace } set { gemm.trace = newValue } }

    private let imageBlock: Block          // sized for the image tokens alone
    private let capBlock: Block            // for the text tokens alone
    private let unifiedBlock: Block        // for the whole sequence
    /// The **maxima**: what the arenas are sized for. An evaluation may ask for less
    /// (the spectral schedule does), never more.
    private let latentHeight, latentWidth, imageTokens, capLength, capPadded, sequence: Int

    /// - Parameter freezeCut: the reproducible mode — the GPU/AMX cut is not servo-controlled, so
    ///   two evaluations of the same inputs return the same bits. See `Conductor.driven`.
    /// - Parameter lora: this request's stack. The arenas are sized for **its** cumulative
    ///   rank — the engine receives the request before building the DiT, so there is no
    ///   maximum to fix in advance and nothing to reload when the stack changes.
    package init(artifact: Artifact, latentHeight: Int, latentWidth: Int, capLength: Int,
                freezeCut: Bool = EngineSettings.effective.frozenCut, lora: LoRA? = nil) throws {
        self.lora = lora
        self.artifact = artifact
        self.prefetcher = Prefetcher(artifact: artifact)
        self.config = Config(header: artifact.header)
        self.gemm = try GEMM(freezeCut: freezeCut)
        self.rope = RopeTables(axesDims: config.axesDims, axesLens: config.axesLens, theta: config.theta)
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
        self.capLength = capLength
        imageTokens = (latentHeight / config.patch) * (latentWidth / config.patch)
        capPadded = capLength + ((-capLength) % config.seqMultiple + config.seqMultiple) % config.seqMultiple
        sequence = imageTokens + capPadded

        let d = config.dim, h = config.hidden
        // Tight sizing, because at 1024² anonymous memory is the only item that can swap.
        // A block reserves **seven** `s×d` slices (`x`, `normed`, `qkv`, `q`, `k`, `v`, `merged`),
        // two `s×h` (`h1`, `h3`; `scratch` is only `4d` since the SwiGLU lives on the GPU,
        // a `d×h` weight reserve and the modulation. There used to be
        // eleven `s×d`: the four CPU-transposition slices went off into the SDPA's
        // graph. This formula must follow `Block`'s `slot(...)`s, otherwise the arena refuses to
        // serve at startup — intended behaviour, but it costs a minute.
        // The LoRA stack adds the `mid [s, Σr]` slice to each block; its widened weights live
        // in the shared cache (`CacheLoRA`, ~340 MB for PopArt at rank 32).
        let r = lora?.maxRank ?? 0
        func blockBudget(_ s: Int) -> Int {
            (7 * s * d + 2 * s * h + d * h + 8 * d + s * r) * 4 + (8 << 20)
        }
        let elementwise = try ElementwiseGPU(device: gemm.device, queue: gemm.queue)
        // The stack's widened weights, kept from one evaluation to the next and shared by the
        // three sets of blocks (`CacheLoRA`).
        let cacheLoRA = r > 0 ? try CacheLoRA(lora: lora!) : nil
        // The DiT itself: a `max(d·h, s·h)` scratch, five `s×d` slices, and odds and ends.
        let own = (max(d * h, sequence * h) + 5 * sequence * d + 4 * sequence * config.patch * config.patch
                   * config.channels + 8 * capPadded * max(config.capDim, d) + 8 * d) * 4 + (16 << 20)
        arena = try Arena(capacity: own)
        imageBlock = try Block(artifact: artifact,
                               shapes: .init(sequence: imageTokens, dim: d, heads: config.heads, hidden: h),
                               eps: config.eps, gemm: gemm, elementwise: elementwise,
                               arena: try Arena(capacity: blockBudget(imageTokens)), lora: lora,
                               cacheLoRA: cacheLoRA)
        capBlock = try Block(artifact: artifact,
                             shapes: .init(sequence: capPadded, dim: d, heads: config.heads, hidden: h),
                             eps: config.eps, gemm: gemm, elementwise: elementwise,
                             arena: try Arena(capacity: blockBudget(capPadded)), lora: lora,
                             cacheLoRA: cacheLoRA)
        capBlock.modulation = false
        unifiedBlock = try Block(artifact: artifact,
                                 shapes: .init(sequence: sequence, dim: d, heads: config.heads, hidden: h),
                                 eps: config.eps, gemm: gemm, elementwise: elementwise,
                                 arena: try Arena(capacity: blockBudget(sequence)), lora: lora,
                                 cacheLoRA: cacheLoRA)
    }

    private func record(_ name: String, _ p: UnsafePointer<Float>, _ n: Int) {
        guard recordBoundaries else { return }
        boundaries[name] = Array(UnsafeBufferPointer(start: p, count: n))
    }

    /// `latent` as `[C, H, W]`, `caps` as `[L, capDim]`, `sigma` as the scheduler gives it.
    /// Returns the model output as `[C, H, W]` — **before** the negation that the pipeline applies.
    ///
    /// - Parameter latentHeight, latentWidth: the latent grid **of this evaluation**, at most
    ///   the sizing one. This is what makes the spectral schedule possible without a second instance:
    ///   the arenas are sized once for the full resolution, and a smaller evaluation
    ///   only occupies the beginning of each slice. The model, for its part, only sees a smaller image —
    ///   the RoPE positions are rebuilt on the grid it is given, as the reference does.
    ///
    ///   **A second instance would have cost ~1 GB of arenas** for a path that never runs at the
    ///   same time as the other. An arena belongs to a phase, and the two resolutions are the
    ///   same phase.
    package func forward(latent: UnsafePointer<Float>, caps: UnsafePointer<Float>,
                        sigma: Float, latentHeight: Int? = nil, latentWidth: Int? = nil) throws -> [Float] {
        let d = config.dim, patch = config.patch, channels = config.channels
        let inDim = patch * patch * channels
        let height = latentHeight ?? self.latentHeight, width = latentWidth ?? self.latentWidth
        guard height <= self.latentHeight, width <= self.latentWidth,
              height % patch == 0, width % patch == 0 else {
            throw Artifact.Failure.misuse(
                "latent \(width)×\(height): at most \(self.latentWidth)×\(self.latentHeight) "
                    + "and a multiple of \(patch)")
        }
        // Tokens are stored row by row: `t / tilesWide` is the row, `t % tilesWide` the
        // column — for the patchify, the RoPE and the unpatchify.
        let tilesWide = width / patch
        let imageTokens = (height / patch) * tilesWide
        let sequence = imageTokens + capPadded
        let scratch = try slot("scratch", max(d * config.hidden, self.sequence * config.hidden))

        // The tail of the map is read beside the page cache and handed back as it is computed, so
        // that the head stays cached for the next evaluation (`Prefetcher.firstReleasedLayer`).
        // How much, and with how many staging buffers (~0.36 GB each, anonymous), the render decides
        // at every evaluation (`MemoryPlan.ditTail`): half by default, the whole map in the lean
        // plan. A switch between two evaluations changes which bytes go through the cache,
        // never the bytes. **Decided before the evaluation's first request**: the refiners ask for
        // `layers.0.` ahead, and a block requested before its stream exists is paged in through the
        // cache, then read a second time by the stream. The former guard (a tail only up to the area of a
        // 1024²) is gone: the 1024×1536 swap it answered was the decoder's, and 1024×1536
        // renders with the tail and its buffers without swap (measured: 3 774 MB, 0 swapout).
        let layerBlocks = (0..<config.layers).map { prefetcher.namesOfBlock(prefix: "layers.\($0).") }
        let (released, slots) = MemoryPlan.ditTail(artifact: artifact, layerBlocks: layerBlocks)
        try artifact.streamTail(blocks: slots == 0 ? [] : Array(layerBlocks[released...]), slots: slots)

        // The front end first: while the step embedding is computed, the disk brings in
        // the embedders and the refiners.
        prefetcher.request(artifact.order.filter {
            $0.hasPrefix("t_embedder.") || $0.hasPrefix("all_x_embedder.") || $0.hasPrefix("cap_embedder.")
        })

        // ── t_embedder ───────────────────────────────────────────────────────────────────
        let adaln = try slot("adaln", config.adalnDim)
        let mid = try slot("mid", 1024)
        var frequencies = [Float](repeating: 0, count: config.adalnDim)
        frequencies.withUnsafeMutableBufferPointer {
            Ops.timestepEmbedding(sigma * config.timeScale, into: $0.baseAddress!, dim: config.adalnDim)
        }
        try linearCPU("t_embedder.mlp.0", x: frequencies, into: mid,
                      outputs: 1024, inputs: config.adalnDim, scratch: scratch)
        Ops.siluInPlace(mid, count: 1024, scratch: scratch)
        try linearCPU("t_embedder.mlp.2", x: UnsafeBufferPointer(start: mid, count: 1024), into: adaln,
                      outputs: config.adalnDim, inputs: 1024, scratch: scratch)
        record("t_embedder_out", adaln, config.adalnDim)

        // ── image: patchify, x_embedder, refiners ────────────────────────────────────────
        let tokens = try slot("tokens", self.imageTokens * inDim)
        Ops.patchify(latent, into: tokens, channels: channels,
                     height: height, width: width, patch: patch)
        let image = try slot("image", self.sequence * d)
        try linearGPU("all_x_embedder.\(patch)-1", a: tokens, into: image,
                      m: imageTokens, k: inDim, n: d, scratch: scratch,
                      wrapRows: self.imageTokens)
        record("x_embedder_out", image, imageTokens * d)

        // ── text: cap_embedder, learned padding token, refiners ──────────────────────────
        let capWork = try slot("capWork", capPadded * max(config.capDim, d))
        for row in 0..<capPadded {
            (capWork + row * config.capDim).update(from: caps + min(row, capLength - 1) * config.capDim,
                                                   count: config.capDim)
        }
        try widen("cap_embedder.0.weight", count: config.capDim, into: scratch)
        Ops.rmsNorm(capWork, weight: scratch, into: capWork,
                    rows: capPadded, columns: config.capDim, eps: config.eps)
        let cap = try slot("cap", capPadded * d)
        try linearGPU("cap_embedder.1", a: capWork, into: cap,
                      m: capPadded, k: config.capDim, n: d, scratch: scratch)
        record("cap_embedder_out", cap, capPadded * d)
        var padToken = [Float](repeating: 0, count: d)
        try padToken.withUnsafeMutableBufferPointer { _ = try artifact.materialize("cap_pad_token", into: $0) }
        for row in capLength..<capPadded { (cap + row * d).update(from: padToken, count: d) }

        // **`SILICONED_CAPS_DROP=5,6,7,8` — the ablation of a prompt attribute.**
        //
        // A measuring instrument: which caption positions carry what (it was written for "why is
        // the thirty-year-old woman a child?", which was settled otherwise).
        //
        // This flag replaces the designated captions with the **learned padding token**, and
        // that is what makes the ablation clean: the model saw this token at this place during
        // training, so we stay within its distribution. Putting zeros, or removing the
        // rows, would feed it an input it has never seen — and the image would change for a reason
        // that had nothing to do with the attribute.
        //
        // The indices are those of the templated prompt: `0 <|im_start|>`, `1 user`, `2 \n`, `3 a`,
        // `4 ' '`, `5 3`, `6 0`, `7 ' year'`, `8 ' old'`, `9 ' woman'`, … `13 ' library'`.
        // So `5,6,7,8` removes "30 year old" and nothing else.
        //
        // **Measurement, never a render.** A flag that changes the image must not be able to slip
        // into a verdict: it announces itself on the error output at every evaluation.
        //
        // It is set through `capsDrop`, not read from the environment here (that would be 238 reads
        // per evaluation, and a setting the caller could not see): `ZImageDenoisingModule.product`
        // fills it from `SILICONED_CAPS_DROP`, once, at the edge.
        if !capsDrop.isEmpty {
            let dropped = capsDrop.filter { $0 < capLength }
            for row in dropped { (cap + row * d).update(from: padToken, count: d) }
            Warnings.emit("⚠ caps_drop: captions \(dropped) replaced by the padding token")
        }
        record("context_refiner_in", cap, capPadded * d)

        // ── RoPE tables ─────────────────────────────────────────────────────────────────
        let complex = rope.complexPerToken
        let freqs = try slot("freqs", self.sequence * complex * 2)
        for t in 0..<imageTokens {
            rope.write(ids: [capPadded + 1, t / tilesWide, t % tilesWide], into: freqs + t * complex * 2)
        }
        for i in 0..<capPadded {
            rope.write(ids: [1 + i, 0, 0], into: freqs + (imageTokens + i) * complex * 2)
        }
        record("block_freqs_cis", freqs, sequence * complex * 2)

        // The prefetcher runs ahead: it requests the next block while this one is computed.
        // That is all it needs, the access order being the file order.
        prefetcher.request(prefetcher.namesOfBlock(prefix: "noise_refiner.0."))
        for i in 0..<config.refiners {
            try cancellation.check()
            prefetcher.request(prefetcher.namesOfBlock(
                prefix: i + 1 < config.refiners ? "noise_refiner.\(i + 1)." : "context_refiner.0."))
            try imageBlock.forward(x: image, freqs: freqs, adaln: adaln,
                                   adalnDim: config.adalnDim, prefix: "noise_refiner.\(i).",
                                   sequence: imageTokens)
        }
        record("noise_refiner_out", image, imageTokens * d)
        for i in 0..<config.refiners {
            try cancellation.check()
            prefetcher.request(prefetcher.namesOfBlock(
                prefix: i + 1 < config.refiners ? "context_refiner.\(i + 1)." : "layers.0."))
            try capBlock.forward(x: cap, freqs: freqs + imageTokens * complex * 2, adaln: adaln,
                                 adalnDim: config.adalnDim, prefix: "context_refiner.\(i).")
        }
        record("context_refiner_out", cap, capPadded * d)

        // ── concatenation: image FIRST ──────────────────────────────────────────────────
        let unified = try slot("unified", self.sequence * d)
        unified.update(from: image, count: imageTokens * d)
        (unified + imageTokens * d).update(from: cap, count: capPadded * d)
        record("unified_in", unified, sequence * d)

        for layer in 0..<config.layers {
            try cancellation.check()   // between two layers, never inside one (see `Cancellation`)
            // Two blocks ahead: a single one would let the computation catch up with the disk at
            // small resolutions, where a block only lasts a few tens of milliseconds.
            for ahead in 1...2 where layer + ahead < config.layers {
                prefetcher.request(prefetcher.namesOfBlock(prefix: "layers.\(layer + ahead)."))
            }
            if layer == config.layers - 1 {
                prefetcher.request(artifact.order.filter { $0.hasPrefix("all_final_layer.") })
            }
            try unifiedBlock.forward(x: unified, freqs: freqs, adaln: adaln,
                                     adalnDim: config.adalnDim, prefix: "layers.\(layer).",
                                     sequence: sequence)
            if layer >= released { artifact.dropFromCache(prefetcher.namesOfBlock(prefix: "layers.\(layer).")) }
        }
        record("layers_29_out", unified, sequence * d)

        // ── FinalLayer: SiLU is here and nowhere else ───────────────────────────────────
        let key = "all_final_layer.\(patch)-1."
        var silu = Array(UnsafeBufferPointer(start: adaln, count: config.adalnDim))
        silu.withUnsafeMutableBufferPointer { Ops.siluInPlace($0.baseAddress!, count: $0.count, scratch: scratch) }
        let scale = try slot("scale", d)
        try linearCPU(key + "adaLN_modulation.1", x: silu, into: scale,
                      outputs: d, inputs: config.adalnDim, scratch: scratch)
        Ops.addScalar(scale, 1, count: d)
        let head = try slot("head", self.sequence * d)
        Ops.layerNorm(unified, into: head, rows: sequence, columns: d, eps: 1e-6)
        Ops.scaleRows(head, by: scale, rows: sequence, columns: d)
        let projected = try slot("projected", self.sequence * inDim)
        try linearGPU(key + "linear", a: head, into: projected,
                      m: sequence, k: d, n: inDim, scratch: scratch, wrapRows: self.sequence)
        record("final_layer_out", projected, sequence * inDim)

        var out = [Float](repeating: 0, count: channels * height * width)
        out.withUnsafeMutableBufferPointer {
            Ops.unpatchify(projected, into: $0.baseAddress!, latentHeight: height,
                           latentWidth: width, patch: patch, channels: channels)
        }
        return out
    }

    // ── plumbing ────────────────────────────────────────────────────────────────────────

    private var reserved: Set<String> = []
    private func slot(_ name: String, _ count: Int) throws -> UnsafeMutablePointer<Float> {
        if reserved.contains(name) { return arena.pointer(name)!.assumingMemoryBound(to: Float.self) }
        reserved.insert(name)
        return try arena.reserve(name, bytes: count * 4).assumingMemoryBound(to: Float.self)
    }

    /// A weight of exactly `count` values into `destination` — refused before writing if the map's
    /// is larger (`Artifact.materialize`), after if it is smaller: either is another map.
    private func widen(_ name: String, count: Int, into destination: UnsafeMutablePointer<Float>) throws {
        let got = try artifact.materialize(name, into: destination, capacity: count)
        guard got == count else {
            throw Artifact.Failure.badHeader("\(name): \(got) values, \(count) expected")
        }
    }

    private func linearCPU(_ key: String, x: some Collection<Float>, into out: UnsafeMutablePointer<Float>,
                           outputs: Int, inputs: Int, scratch: UnsafeMutablePointer<Float>) throws {
        try widen(key + ".weight", count: outputs * inputs, into: scratch)
        var bias = [Float](repeating: 0, count: outputs)
        try bias.withUnsafeMutableBufferPointer { _ = try artifact.materialize(key + ".bias", into: $0) }
        Array(x).withUnsafeBufferPointer {
            Ops.gemv(weight: scratch, bias: bias, x: $0.baseAddress!, into: out,
                     outputs: outputs, inputs: inputs, transposed: artifact.linearWeightsTransposed)
        }
    }

    /// - Parameter wrapRows: the number of **reserved** rows, when the call uses fewer.
    ///   The Metal wrappers are memoized by address; sizing them on the call would make the
    ///   `MTLBuffer` be rebuilt at every change of resolution, i.e. the page wiring that is paid for once.
    private func linearGPU(_ key: String, a: UnsafeMutablePointer<Float>, into out: UnsafeMutablePointer<Float>,
                           m: Int, k: Int, n: Int, scratch: UnsafeMutablePointer<Float>,
                           wrapRows: Int? = nil) throws {
        try widen(key + ".weight", count: n * k, into: scratch)
        let rows = wrapRows ?? m
        let ab = try gemm.wrap(UnsafeMutableRawPointer(a), bytes: rows * k * 4, name: "a")
        let wb = try gemm.wrap(UnsafeMutableRawPointer(scratch), bytes: n * k * 4, name: "w")
        let ob = try gemm.wrap(UnsafeMutableRawPointer(out), bytes: rows * n * 4, name: "o")
        _ = gemm.linear(a: ab, b: wb, c: ob, m: m, k: k, n: n,
                        weightIsTransposed: artifact.linearWeightsTransposed)
        if artifact.tensors[key + ".bias"] != nil {
            var bias = [Float](repeating: 0, count: n)
            try bias.withUnsafeMutableBufferPointer { _ = try artifact.materialize(key + ".bias", into: $0) }
            for row in 0..<m { vDSP_vadd(out + row * n, 1, bias, 1, out + row * n, 1, vDSP_Length(n)) }
        }
    }
}
