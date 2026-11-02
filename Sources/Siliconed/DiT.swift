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
///
/// **Both sequences are padded to a multiple of 32**, not the caption alone (`_pad_with_ids`): the
/// image's missing rows (16 at 832×1216, none at 512², 1024², 768×512 or 1024×1536) are the last patch
/// repeated through `x_embedder`, then replaced by the learned `x_pad_token`, at RoPE position
/// `(0, 0, 0)`. Nothing masks them at batch 1: they are keys and queries in the noise refiners and the
/// thirty layers, and only the unpatchify drops them. The sequence is `[image, image pad, caption,
/// caption pad]`. A port that pads the caption alone computes another attention on those formats.
///
/// **Every product by a weight goes through `WeightProducts`**, the front and the head included
/// (`x_embedder`, `cap_embedder.1`, the final linear): a LoRA on them is applied, as is one on the
/// `M = 1` products (`t_embedder.mlp.*`, the final modulation), on the CPU beside their `gemv`.
package final class DiT: BoundaryRecording {
    package struct Config {
        package let dim, heads, hidden, layers, refiners, patch, channels, capDim, adalnDim: Int
        package let eps, theta, timeScale: Float
        package let axesDims, axesLens: [Int]
        package let seqMultiple = 32
        /// The published RoPE table: 1 536 segment indexes, 512 positions on each spatial axis
        /// (one per 16 px: `Format.maximumSide`).
        package static let publishedAxesLens = [1536, 512, 512]

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
            axesLens = (c["axes_lens"] as? [Int]) ?? Config.publishedAxesLens
        }
    }

    package let config: Config
    package internal(set) var boundaries: [String: [Float]] = [:]
    package var recordBoundaries = false
    /// The current render's token, consulted between two layers (see `Cancellation`).
    package var cancellation: Cancellation?
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
    /// The floats of each slice of `arena`, from `ownSlices`.
    private let ownSizes: [String: Int]
    /// The image, caption and unified blocks' arenas, in that order.
    private let blockArenas: [Arena]
    private let cacheLoRA: CacheLoRA?
    /// The time per phase, aggregated over the three sets of blocks.
    package var timings: [String: Double] {
        var total: [String: Double] = [:]
        for block in [imageBlock, capBlock, unifiedBlock] {
            for (phase, seconds) in block.timings { total[phase, default: 0] += seconds }
        }
        for (phase, seconds) in products.timings { total[phase, default: 0] += seconds }
        return total
    }
    package func resetTimings() {
        for b in [imageBlock, capBlock, unifiedBlock] { b.resetTimings() }
        products.resetTimings()
    }
    package var census: [String: (count: Int, seconds: Double, best: Double)] { gemm.census }
    /// What the per-token cut returned, when it is on — `nil` otherwise, and that is the
    /// difference between "it did nothing" and "it was not there".
    package var conductor: Conductor? { gemm.conductor }
    package var traceGEMM: Int { get { gemm.trace } set { gemm.trace = newValue } }

    private let imageBlock: Block          // sized for the image tokens alone
    private let capBlock: Block            // for the text tokens alone
    private let unifiedBlock: Block        // for the whole sequence
    /// The grid the arenas are sized for, which every evaluation uses.
    /// `imageTokens` the latent's patches, `imagePadded` them up to `seqMultiple` (the rows the
    /// image blocks and the unified sequence carry).
    private let latentHeight, latentWidth, imageTokens, imagePadded, capLength, capPadded, sequence: Int
    /// The products by a weight outside the blocks (`x_embedder`, `cap_embedder.1`, the final linear),
    /// in the own arena's `scratch`, with the stack's `loraMid`.
    private let products: WeightProducts

    /// - Parameter lora: this request's stack. The arenas are sized for **its** cumulative
    ///   rank — the engine receives the request before building the DiT, so there is no
    ///   maximum to fix in advance and nothing to reload when the stack changes.
    package init(artifact: Artifact, latentHeight: Int, latentWidth: Int, capLength: Int,
                accelerators: Accelerators = .process, lora: LoRA? = nil) throws {
        self.lora = lora
        self.artifact = artifact
        self.prefetcher = Prefetcher(artifact: artifact)
        self.config = Config(header: artifact.header)
        self.gemm = try GEMM(accelerators: accelerators)
        self.rope = RopeTables(axesDims: config.axesDims, axesLens: config.axesLens, theta: config.theta)
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
        self.capLength = capLength
        imageTokens = DiT.imageTokens(latentHeight: latentHeight, latentWidth: latentWidth, config: config)
        imagePadded = DiT.padded(imageTokens, config: config)
        capPadded = DiT.capPadded(capLength, config: config)
        sequence = imagePadded + capPadded

        let d = config.dim, h = config.hidden
        // The arenas are sized by `blockArenaBytes` and `ownArenaBytes`, the functions the memory need
        // charges (`MemoryNeed.zImage`): the formula equals the allocation by construction.
        // The LoRA stack adds the `mid [s, Σr]` slice to each block; its widened weights live
        // in the shared cache (`CacheLoRA`, ~340 MB for PopArt at rank 32).
        let r = lora?.maxRank ?? 0
        if let lora, r > 0 {
            let unapplied = lora.rankPerTarget.keys.filter { !DiT.appliesLoRA($0) }.sorted()
            if let first = unapplied.first {
                throw LoRA.Failure.inconsistentShape(target: first, reason: "a Z-Image module the engine does not adapt "
                    + "(\(unapplied.count) in all) — refused rather than left out")
            }
        }
        let elementwise = try ElementwiseGPU(device: gemm.device, queue: gemm.queue)
        // The stack's widened weights, kept from one evaluation to the next and shared by the
        // three sets of blocks (`CacheLoRA`).
        let cacheLoRA = r > 0 ? try CacheLoRA(lora: lora!) : nil
        arena = try Arena(capacity: DiT.ownArenaBytes(imageTokens: imagePadded, capPadded: capPadded, config: config,
                                                      loraRank: r))
        let ownSlices = DiT.ownSlices(imageTokens: imagePadded, capPadded: capPadded, config: config, loraRank: r)
        ownSizes = Dictionary(uniqueKeysWithValues: ownSlices.map { ($0.name, $0.count) })
        // `scratch` is the products' weight reserve, sized for the widest weight they read.
        let reserve = try arena.reserve("scratch", bytes: ownSizes["scratch"]! * 4).assumingMemoryBound(to: Float.self)
        let mid = r > 0 ? try arena.reserve("loraMid", bytes: ownSizes["loraMid"]! * 4).assumingMemoryBound(to: Float.self)
                        : nil
        reserved = ["scratch", "loraMid"]
        products = WeightProducts(gemm: gemm, artifact: artifact, reserve: reserve, capacity: ownSizes["scratch"]!,
                                  transposed: artifact.linearWeightsTransposed, lora: cacheLoRA, mid: mid,
                                  midFloats: ownSizes["loraMid"] ?? 0, roundsActivations: true)
        let config = self.config   // a local: the closure below must not capture a half-built `self`
        // **One arena for the three sets of blocks**: the image refiners, the caption refiners and the layers
        // run one after the other and each writes its scratch whole before reading it, so they share the
        // pages — the largest set's, not the sum (`blocksArenaBytes`).
        let shared = try Arena(capacity: DiT.blocksArenaBytes(image: imagePadded, cap: capPadded, config: config, loraRank: r))
        let arenas = [shared, Arena(sharing: shared), Arena(sharing: shared)]
        blockArenas = [shared]
        imageBlock = try Block(artifact: artifact,
                               shapes: .init(sequence: imagePadded, dim: d, heads: config.heads, hidden: h),
                               eps: config.eps, gemm: gemm, elementwise: elementwise,
                               arena: arenas[0], lora: lora,
                               cacheLoRA: cacheLoRA)
        capBlock = try Block(artifact: artifact,
                             shapes: .init(sequence: capPadded, dim: d, heads: config.heads, hidden: h),
                             eps: config.eps, gemm: gemm, elementwise: elementwise,
                             arena: arenas[1], lora: lora,
                             cacheLoRA: cacheLoRA)
        capBlock.modulation = false
        unifiedBlock = try Block(artifact: artifact,
                                 shapes: .init(sequence: sequence, dim: d, heads: config.heads, hidden: h),
                                 eps: config.eps, gemm: gemm, elementwise: elementwise,
                                 arena: arenas[2], lora: lora,
                                 cacheLoRA: cacheLoRA)
        self.cacheLoRA = cacheLoRA
    }

    /// The bytes this DiT's two arenas and its LoRA cache hold — what the tests compare with
    /// `MemoryNeed`'s formula.
    package var arenaCapacity: Int { arena.capacity + blockArenas.reduce(0) { $0 + $1.capacity } }
    package var cacheLoRACapacity: Int { cacheLoRA?.capacity ?? 0 }

    /// `latent` as `[C, H, W]`, `caps` as `[L, capDim]`, `sigma` as the scheduler gives it.
    /// Returns the model output as `[C, H, W]` — **before** the negation that the pipeline applies.
    /// Every evaluation is at the grid the arenas were sized for: the RoPE positions are rebuilt
    /// on it, as the reference does.
    package func forward(latent: UnsafePointer<Float>, caps: UnsafePointer<Float>,
                        sigma: Float) throws -> [Float] {
        let d = config.dim, patch = config.patch, channels = config.channels
        let inDim = patch * patch * channels
        let height = latentHeight, width = latentWidth
        guard height % patch == 0, width % patch == 0 else {
            throw Artifact.Failure.misuse("latent \(width)×\(height): a multiple of \(patch)")
        }
        // Tokens are stored row by row: `t / tilesWide` is the row, `t % tilesWide` the
        // column — for the patchify, the RoPE and the unpatchify.
        let tilesWide = width / patch
        let scratch = try slot("scratch")

        // The tail of the map is read beside the page cache and handed back as it is computed, so
        // that the head stays cached for the next evaluation (`Prefetcher.firstReleasedLayer`).
        // How much, and with how many staging buffers (units of a few tensors, the largest one's size each: `TailStream`), the plan chosen in B
        // says (`MemoryPlan.ditTail`): a head of what B leaves, at most half; none on the leanest; the
        // whole map cached when B holds it. Which bytes go through the cache, never the bytes.
        // **Decided before the evaluation's first request**: the refiners ask for
        // `layers.0.` ahead, and a block requested before its stream exists is paged in through the
        // cache, then read a second time by the stream. The former guard (a tail only up to the area of a
        // 1024²) is gone: the 1024×1536 swap it answered was the decoder's, and 1024×1536
        // renders with the tail and its buffers without swap (measured: 3 774 MB, 0 swapout).
        let layerBlocks = (0..<config.layers).map { prefetcher.namesOfBlock(prefix: DiT.layerPrefix($0)) }
        let (released, streamed) = MemoryPlan.ditTail(artifact: artifact, layerBlocks: layerBlocks)
        if !streamed {
            try artifact.streamTail(blocks: [], slots: 0)
        } else {
            try artifact.streamUnits(layers: layerBlocks, from: released, prefix: DiT.layerPrefix, suffixes: DiT.consumptionOrder)
            artifact.setStreamPhase(family: .zImage, tokens: imagePadded, textRows: capPadded)
        }

        // The front end first: while the step embedding is computed, the disk brings in
        // the embedders and the refiners.
        prefetcher.request(artifact.order.filter {
            $0.hasPrefix("t_embedder.") || $0.hasPrefix("all_x_embedder.") || $0.hasPrefix("cap_embedder.")
        })

        // ── t_embedder ───────────────────────────────────────────────────────────────────
        let adaln = try slot("adaln")
        let mid = try slot("mid")
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
        let tokens = try slot("tokens")
        Ops.patchify(latent, into: tokens, channels: channels,
                     height: height, width: width, patch: patch)
        // The reference pads with the last patch (`feat[-1:].repeat`), embeds, then overwrites those
        // rows with `x_pad_token`: the embedding of the copies is what its probe records.
        for row in imageTokens..<imagePadded {
            (tokens + row * inDim).update(from: tokens + (imageTokens - 1) * inDim, count: inDim)
        }
        let image = try slot("image")
        try linearGPU("all_x_embedder.\(patch)-1", a: tokens, into: image,
                      m: imagePadded, k: inDim, n: d)
        record("x_embedder_out", image, imagePadded * d)
        if imagePadded > imageTokens {
            var padToken = [Float](repeating: 0, count: d)
            try padToken.withUnsafeMutableBufferPointer { _ = try artifact.materialize("x_pad_token", into: $0) }
            for row in imageTokens..<imagePadded { (image + row * d).update(from: padToken, count: d) }
        }

        // ── text: cap_embedder, learned padding token, refiners ──────────────────────────
        let capWork = try slot("capWork")
        for row in 0..<capPadded {
            (capWork + row * config.capDim).update(from: caps + min(row, capLength - 1) * config.capDim,
                                                   count: config.capDim)
        }
        try widen("cap_embedder.0.weight", count: config.capDim, into: scratch)
        Ops.rmsNorm(capWork, weight: scratch, into: capWork,
                    rows: capPadded, columns: config.capDim, eps: config.eps)
        let cap = try slot("cap")
        try linearGPU("cap_embedder.1", a: capWork, into: cap,
                      m: capPadded, k: config.capDim, n: d)
        record("cap_embedder_out", cap, capPadded * d)
        var padToken = [Float](repeating: 0, count: d)
        try padToken.withUnsafeMutableBufferPointer { _ = try artifact.materialize("cap_pad_token", into: $0) }
        for row in capLength..<capPadded { (cap + row * d).update(from: padToken, count: d) }
        record("context_refiner_in", cap, capPadded * d)

        // ── RoPE tables ─────────────────────────────────────────────────────────────────
        let complex = rope.complexPerToken
        let freqs = try slot("freqs")
        for t in 0..<imageTokens {
            rope.write(ids: [capPadded + 1, t / tilesWide, t % tilesWide], into: freqs + t * complex * 2)
        }
        for t in imageTokens..<imagePadded {   // the image's padding: position (0, 0, 0)
            rope.write(ids: [0, 0, 0], into: freqs + t * complex * 2)
        }
        for i in 0..<capPadded {
            rope.write(ids: [1 + i, 0, 0], into: freqs + (imagePadded + i) * complex * 2)
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
                                   adalnDim: config.adalnDim, prefix: "noise_refiner.\(i).")
        }
        record("noise_refiner_out", image, imagePadded * d)
        for i in 0..<config.refiners {
            try cancellation.check()
            prefetcher.request(prefetcher.namesOfBlock(
                prefix: i + 1 < config.refiners ? "context_refiner.\(i + 1)." : "layers.0."))
            try capBlock.forward(x: cap, freqs: freqs + imagePadded * complex * 2, adaln: adaln,
                                 adalnDim: config.adalnDim, prefix: "context_refiner.\(i).")
        }
        record("context_refiner_out", cap, capPadded * d)

        // ── concatenation: image FIRST ──────────────────────────────────────────────────
        let unified = try slot("unified")
        unified.update(from: image, count: imagePadded * d)
        (unified + imagePadded * d).update(from: cap, count: capPadded * d)
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
                                     adalnDim: config.adalnDim, prefix: "layers.\(layer).")
            if layer >= released { artifact.dropFromCache(prefetcher.namesOfBlock(prefix: "layers.\(layer).")) }
        }
        record("layers_29_out", unified, sequence * d)

        // ── FinalLayer: SiLU is here and nowhere else ───────────────────────────────────
        let key = "all_final_layer.\(patch)-1."
        var silu = Array(UnsafeBufferPointer(start: adaln, count: config.adalnDim))
        silu.withUnsafeMutableBufferPointer { Ops.siluInPlace($0.baseAddress!, count: $0.count, scratch: scratch) }
        let scale = try slot("scale")
        try linearCPU(key + "adaLN_modulation.1", x: silu, into: scale,
                      outputs: d, inputs: config.adalnDim, scratch: scratch)
        Ops.addScalar(scale, 1, count: d)
        let head = try slot("head")
        Ops.layerNorm(unified, into: head, rows: sequence, columns: d, eps: 1e-6)
        Ops.scaleRows(head, by: scale, rows: sequence, columns: d)
        let projected = try slot("projected")
        try linearGPU(key + "linear", a: head, into: projected,
                      m: sequence, k: d, n: inDim)
        record("final_layer_out", projected, sequence * inDim)

        // The first `imageTokens` rows: the image's padding and the caption are dropped here.
        var out = [Float](repeating: 0, count: channels * height * width)
        out.withUnsafeMutableBufferPointer {
            Ops.unpatchify(projected, into: $0.baseAddress!, latentHeight: height,
                           latentWidth: width, patch: patch, channels: channels)
        }
        return out
    }

    // ── plumbing ────────────────────────────────────────────────────────────────────────

    private var reserved: Set<String> = []
    /// The slice `name` of the DiT's own arena, at the size `ownSlices` gives it (reserved on first use).
    private func slot(_ name: String) throws -> UnsafeMutablePointer<Float> {
        if reserved.contains(name) { return arena.pointer(name)!.assumingMemoryBound(to: Float.self) }
        guard let count = ownSizes[name] else { throw Artifact.Failure.misuse("DiT: no slice \(name)") }
        reserved.insert(name)
        return try arena.reserve(name, bytes: count * 4).assumingMemoryBound(to: Float.self)
    }

    /// A weight of exactly `count` values into `destination` — refused before writing if the map's
    /// is larger (`Artifact.materialize`), after if it is smaller: either is another map.
    private func widen(_ name: String, count: Int, into destination: UnsafeMutablePointer<Float>) throws {
        try artifact.materializeExactly(name, count: count, into: destination)
    }

    /// A `M = 1` product on the CPU (`gemv`), then the stack of `key`: `y += (x · down) · up`, two
    /// `gemv`s too — as `Block` adapts `adaLN_modulation.0`: two GPU submissions for a few M FLOPs
    /// would cost more than the product.
    private func linearCPU(_ key: String, x: some Collection<Float>, into out: UnsafeMutablePointer<Float>,
                           outputs: Int, inputs: Int, scratch: UnsafeMutablePointer<Float>) throws {
        try widen(key + ".weight", count: outputs * inputs, into: scratch)
        var bias = [Float](repeating: 0, count: outputs)
        try bias.withUnsafeMutableBufferPointer { _ = try artifact.materialize(key + ".bias", into: $0) }
        let input = Array(x)
        input.withUnsafeBufferPointer {
            Ops.gemv(weight: scratch, bias: bias, x: $0.baseAddress!, into: out,
                     outputs: outputs, inputs: inputs, transposed: artifact.linearWeightsTransposed)
        }
        guard let cacheLoRA, let (down, up, r) = try cacheLoRA.module(key, k: inputs, n: outputs) else { return }
        var middle = [Float](repeating: 0, count: r), delta = [Float](repeating: 0, count: outputs)
        input.withUnsafeBufferPointer { x in
            Ops.gemv(weight: down, bias: nil, x: x.baseAddress!, into: &middle,
                     outputs: r, inputs: inputs, transposed: true)
        }
        Ops.gemv(weight: up, bias: nil, x: middle, into: &delta, outputs: outputs, inputs: r, transposed: true)
        vDSP_vadd(out, 1, delta, 1, out, 1, vDSP_Length(outputs))
    }

    /// `out [m, n] = a [m, k] · W (+ the stack of `key`) + bias`, by `WeightProducts`. `a` and `out` are
    /// wrapped at their whole slice (`GEMM.wrap` memoizes by address).
    private func linearGPU(_ key: String, a: UnsafeMutablePointer<Float>, into out: UnsafeMutablePointer<Float>,
                           m: Int, k: Int, n: Int) throws {
        let ab = try gemm.wrap(UnsafeMutableRawPointer(a), bytes: m * k * 4, name: "a")
        let ob = try gemm.wrap(UnsafeMutableRawPointer(out), bytes: m * n * 4, name: "o")
        try products.linear(key + ".weight", a: ab, c: ob, m: m, k: k, n: n)
        if artifact.tensors[key + ".bias"] != nil {
            var bias = [Float](repeating: 0, count: n)
            try bias.withUnsafeMutableBufferPointer { _ = try artifact.materialize(key + ".bias", into: $0) }
            for row in 0..<m { vDSP_vadd(out + row * n, 1, bias, 1, out + row * n, 1, vDSP_Length(n)) }
        }
    }
}

// ── sizing: the shapes the constructor allocates, and the memory need charges ──────────────────

extension DiT {
    /// The image tokens of a latent grid: one per `patch × patch` cell.
    package static func imageTokens(latentHeight: Int, latentWidth: Int, config: Config) -> Int {
        (latentHeight / config.patch) * (latentWidth / config.patch)
    }

    /// `rows` up to the next multiple of `seqMultiple` (`_pad_with_ids`: both sequences).
    package static func padded(_ rows: Int, config: Config) -> Int {
        rows + ((-rows) % config.seqMultiple + config.seqMultiple) % config.seqMultiple
    }

    /// The caption rows, padded up to `seqMultiple` (the reference pads with its learned token).
    package static func capPadded(_ capLength: Int, config: Config) -> Int { padded(capLength, config: config) }

    /// The image rows the blocks carry: the patches, padded up to `seqMultiple` with `x_pad_token`.
    package static func imagePadded(latentHeight: Int, latentWidth: Int, config: Config) -> Int {
        padded(imageTokens(latentHeight: latentHeight, latentWidth: latentWidth, config: config), config: config)
    }

    /// **One `Block`'s arena at sequence `s`**: its slices (`Block.slices`), each on its page, not a
    /// byte more — tight, because at 1024² anonymous memory is the only item that can swap, and because
    /// the memory need charges this capacity while `phys_footprint` sees only the pages written.
    package static func blockArenaBytes(sequence s: Int, config: Config, loraRank r: Int) -> Int {
        let shapes = Block.Shapes(sequence: s, dim: config.dim, heads: config.heads, hidden: config.hidden)
        return Arena.capacity(slices: Block.slices(shapes, loraRank: r).map { $0.count * 4 })
    }

    /// **The slices of the DiT's own arena, in floats** — what `forward` reserves (`slot(_:)` reads its
    /// sizes here), each written whole by an evaluation at the sizing grid:
    ///   - `scratch`: the widest weight `forward` widens itself — `cap_embedder.1` (`d·capDim`), the
    ///     x embedder and the final linear (`d·p²C`), the step embedder (`1024·256`), the final
    ///     modulation (`d·256`). It used to be `max(d·h, S·h)` — 157 to 170 MB at 1024², 39 written;
    ///   - `image` at the padded image rows (`x_embedder`'s output, the image refiners' stream), not `S`;
    ///   - `capWork` at `capDim` per caption row (`cap_embedder.0`'s norm in place), not `max(capDim, d)`;
    ///   - `unified` and `head` at `S·d`, `cap` at `capPadded·d`, `tokens` and `projected` at `p²C` per
    ///     row, the RoPE table at `S · Σ axes/2 · 2`, and three small rows;
    ///   - with a stack of rank `r`, `loraMid [S, r]`: the products outside the blocks (`x_embedder`,
    ///     `cap_embedder.1`, the final linear), `S` their longest.
    /// `imageTokens` is the image's rows **padded** (`imagePadded`), as the blocks carry them.
    /// The former formula counted five `S·d` (three are written), four `S·p²C` (two) and
    /// `8 · capPadded · max(capDim, d)` (two caption slices), plus 16 MB.
    package static func ownSlices(imageTokens: Int, capPadded: Int, config: Config,
                                  loraRank: Int = 0) -> [(name: String, count: Int)] {
        let d = config.dim, sequence = imageTokens + capPadded
        let inDim = config.patch * config.patch * config.channels
        let complex = config.axesDims.reduce(0) { $0 + $1 / 2 }   // `RopeTables.complexPerToken`
        let scratch = [config.capDim * d, inDim * d, 1024 * config.adalnDim, d * config.adalnDim].max()!
        return [("scratch", scratch), ("adaln", config.adalnDim), ("mid", 1024),
                ("tokens", imageTokens * inDim), ("image", imageTokens * d), ("capWork", capPadded * config.capDim),
                ("cap", capPadded * d), ("freqs", sequence * complex * 2), ("unified", sequence * d),
                ("scale", d), ("head", sequence * d), ("projected", sequence * inDim)]
            + (loraRank > 0 ? [("loraMid", sequence * loraRank)] : [])
    }

    /// **The DiT's own arena**: its slices (`ownSlices`), each on its page.
    package static func ownArenaBytes(imageTokens: Int, capPadded: Int, config: Config, loraRank: Int = 0) -> Int {
        Arena.capacity(slices: ownSlices(imageTokens: imageTokens, capPadded: capPadded, config: config,
                                         loraRank: loraRank).map { $0.count * 4 })
    }

    /// **The blocks' shared arena**: the largest of the image, caption and unified blocks' (`blockArenaBytes`).
    package static func blocksArenaBytes(image: Int, cap: Int, config: Config, loraRank: Int) -> Int {
        [image, cap, image + cap].map { blockArenaBytes(sequence: $0, config: config, loraRank: loraRank) }.max()!
    }

    /// **The two arenas a DiT maps** — its own, and the blocks' shared one (the image, caption and unified
    /// blocks take turns in it) — each as `Arena` rounds it: `arenaCapacity` of the DiT built on these shapes,
    /// to the byte.
    package static func arenaBytes(config: Config, latentHeight: Int, latentWidth: Int,
                                   capLength: Int, loraRank: Int) -> Int {
        let image = imagePadded(latentHeight: latentHeight, latentWidth: latentWidth, config: config)
        let cap = capPadded(capLength, config: config)
        return Arena.capacity(for: ownArenaBytes(imageTokens: image, capPadded: cap, config: config, loraRank: loraRank))
            + Arena.capacity(for: blocksArenaBytes(image: image, cap: cap, config: config, loraRank: loraRank))
    }

    /// A block's `Linear`s a stack adapts (`Block`), `(k, n)` as the map lays them out: `to_q`, `to_k`, `to_v`,
    /// `to_out.0` (`d → d`), `w1`, `w3` (`d → h`), `w2` (`h → d`); and the modulation (`256 → 4d`) on the
    /// modulated blocks — the noise refiners and the layers, not the context refiners.
    private static let blockLinears = ["attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0",
                                       "feed_forward.w1", "feed_forward.w3", "feed_forward.w2"]

    /// **Every module a LoRA may touch, `(name, k, n)`**: each block's `Linear`s and modulation (`Block`),
    /// then the DiT's own — the step embedder's two (`M = 1`, CPU), `x_embedder`, `cap_embedder.1`, the final
    /// modulation (`M = 1`, CPU) and the final linear (`forward`). One list for the need that charges the
    /// cache (`loraModules`) and the check that refuses what is not here (`appliesLoRA`).
    package static func loraTargets(config: Config) -> [(name: String, k: Int, n: Int)] {
        let d = config.dim, h = config.hidden, a = config.adalnDim
        let inDim = config.patch * config.patch * config.channels
        let shapes = [(d, d), (d, d), (d, d), (d, d), (d, h), (d, h), (h, d)]
        func block(_ prefix: String, modulated: Bool) -> [(name: String, k: Int, n: Int)] {
            zip(blockLinears, shapes).map { (name: prefix + $0, k: $1.0, n: $1.1) }
                + (modulated ? [(name: prefix + "adaLN_modulation.0", k: a, n: 4 * d)] : [])
        }
        let head = "all_final_layer.\(config.patch)-1."
        // In typed parts: one chain of `+` over tuple arrays outlasts the type checker of an older Swift.
        var targets: [(name: String, k: Int, n: Int)] = []
        targets += (0..<config.refiners).flatMap { block("noise_refiner.\($0).", modulated: true) }
        targets += (0..<config.layers).flatMap { block(layerPrefix($0), modulated: true) }
        targets += (0..<config.refiners).flatMap { block("context_refiner.\($0).", modulated: false) }
        let outer: [(name: String, k: Int, n: Int)] = [
            (name: "t_embedder.mlp.0", k: a, n: 1024), (name: "t_embedder.mlp.2", k: 1024, n: a),
            (name: "all_x_embedder.\(config.patch)-1", k: inDim, n: d),
            (name: "cap_embedder.1", k: config.capDim, n: d),
            (name: head + "adaLN_modulation.1", k: a, n: d), (name: head + "linear", k: d, n: inDim)]
        return targets + outer
    }

    /// **The modules a LoRA can touch, `(k, n)` each** (`loraTargets`): a stack that touches all of them is
    /// the worst case of `CacheLoRA`.
    package static func loraModules(config: Config) -> [(k: Int, n: Int)] {
        loraTargets(config: config).map { (k: $0.k, n: $0.n) }
    }

    /// **Does the engine apply a LoRA on this module** (the map's name, without `.weight`)? The DiT refuses
    /// a stack with any other before computing: a module believed adapted that is not yields a plausible,
    /// wrong image — what `x_embedder` and the final layer were until the products took them.
    package static func appliesLoRA(_ module: String) -> Bool {
        let fixed: Set = ["t_embedder.mlp.0", "t_embedder.mlp.2", "all_x_embedder.2-1", "cap_embedder.1",
                          "all_final_layer.2-1.adaLN_modulation.1", "all_final_layer.2-1.linear"]
        if fixed.contains(module) { return true }
        for (prefix, modulated) in [("noise_refiner.", true), ("layers.", true), ("context_refiner.", false)]
            where module.hasPrefix(prefix) {
            let rest = module.dropFirst(prefix.count)
            guard let dot = rest.firstIndex(of: "."), Int(rest[..<dot]) != nil else { return false }
            let leaf = String(rest[rest.index(after: dot)...])
            return blockLinears.contains(leaf) || (modulated && leaf == "adaLN_modulation.0")
        }
        return false
    }

    /// **A layer's tensors in the order `Block.forward` widens them** — the order its stream reads them in
    /// (`Artifact.streamUnits`): the modulation, the attention's norms and projections, the feed-forward.
    package static let consumptionOrder = ["adaLN_modulation.0.weight", "adaLN_modulation.0.bias", "attention_norm1.weight",
                                           "attention.to_q.weight", "attention.to_k.weight", "attention.to_v.weight",
                                           "attention.norm_q.weight", "attention.norm_k.weight",
                                           "attention.to_out.0.weight", "attention_norm2.weight", "ffn_norm1.weight",
                                           "feed_forward.w1.weight", "feed_forward.w3.weight", "feed_forward.w2.weight",
                                           "ffn_norm2.weight"]

    /// The names of a DiT layer's tensors are `layers.<i>.…`: the tail's staging buffers are sized on
    /// the largest of them (`Artifact.largestRun`).
    package static func layerPrefix(_ layer: Int) -> String { "layers.\(layer)." }
}
