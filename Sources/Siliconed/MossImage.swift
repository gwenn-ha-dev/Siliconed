import Metal
import Accelerate
import Foundation

/// The MOSS-Image-SFT DiT — `MOSSImageTransformer`, 5.08 G parameters —, in fp32 over the weights as
/// published (bf16).
///
///     Qwen3.5 last hidden state [T, 4096] ─ Linear ─ 2 text blocks (own weights, biases, no modulation) ─┐
///     latent [128, H, W] (VAE patched 2×2, 16 px per cell) ─ tokens [H·W, 128] ─ Linear ────────────────┤
///                                                                                                       ▼
///                                                          one sequence: image rows, then text rows
///     t ─ sinusoid 256 [cos | sin] (fp64) ─ Linear ─ SiLU ─ Linear ─ time ─┬─ Linear(6d) ─▶ 42 blocks
///                                                                           └─ Linear(2d) ─▶ head: shift, scale
///     image rows ─ RMSNorm(final) · (1 + scale) + shift ─ Linear(128, no bias) ─ velocity [128, H, W]
///
/// **One block** — every block reads the same six vectors `[shift_a, scale_a, gate_a, shift_f, scale_f, gate_f]`:
///
///     x += gate_a · Wo( SDPA(RoPE(RMS_h(Wq n)), RoPE(RMS_h(Wk n)), Wv n) )   n = RMS(x)·w·(1 + scale_a) + shift_a
///     x += gate_f · W_down( SITU(h, g) )              [h | g] = W_up_gate m,  m = RMS(x)·w·(1 + scale_f) + shift_f
///
/// What sets it apart from the other targets, and what a port by analogy would miss:
///
///   - **the modulation is shared by all 42 blocks, and has no table**: a single `shared_block_modulation`
///     (3072 → 6·3072) per evaluation, read as it comes — the blocks add nothing to it (≠ Krea 2's
///     `temb_mod + table`). It is **not** passed through a SiLU first (the usual adaLN does), and its order is
///     `shift, scale, gate` twice — the shift first. The gates multiply the sublayers as they are, no `tanh`;
///   - **the activation is SITU, not SwiGLU**: `4·tanh(g/4)·σ(g) · 25·tanh(h/25)` — gate and value both capped.
///     `ffn_up_gate` is one fused Linear whose output is `[hidden | gate]` (the **value first**, the gate
///     second, `chunk(2)`); the text blocks keep `ffn_up`/`ffn_gate` apart, with biases;
///   - **the RoPE has four axes** `(r, f, h, w)`, 16 bands each (`head_dim / 8`), θ = 10⁴, `head_dim` = 128 split
///     **in halves** (`rotate_half`, not interleaved pairs): the table is `[sin(64) | cos(64)]`, entry
///     `axis·16 + band`. The image sits at `(0, T, h − (H−1)/2, w − (W−1)/2)` — **centred** on zero, and its frame
///     axis is the **text length** —, the text at `(0, i, 0, 0)`. The bands are fp32 (`1 / θ^(i/16)`), the
///     angles fp32 products: never rounded to bf16 (the reference's own `_apply` guards it);
///   - **the text goes through its own two blocks first** — a Linear 4096 → 3072, then RMSNorm / QKV / attention
///     over the text rows alone (their RoPE rows are the text's) / `proj` / SITU FFN, biases everywhere —, and
///     only then joins the image. No mask: every token of the sequence sees every other;
///   - **the sequence is image then text in the reference; here the text rows come first** (a Metal wrapper wants a
///     page-aligned address and the text blocks work on rows `[0, T)`). With no mask and every row's RoPE
///     attached to the row, the SDPA is invariant to the order of the keys: the same function, the sums in
///     another order. The boundaries are recorded in the reference's order;
///   - **the head**: `final_norm_video` (a plain RMSNorm, weight kept) then `x·(1 + scale) + shift` with
///     `video_head_modulation(time) → [shift | scale]`; `final_linear_video` has **no bias**;
///   - the time vector is the sinusoid **in fp64** over the fp32 timestep the scheduler hands (`σ·1000`, rounded
///     once to fp32), `[cos | sin]` with `10⁴^(−j/128)`, cast to fp32 before the MLP. The MLP and both
///     modulations are done here in double (`Krea2Exact`), tabulated once per σ;
///   - **the official recipe skips block 22 in the negative branch** (`skip_layers`, zero-based): the
///     pass takes `skipping:`; the block is not computed, not read, and the residual passes through it;
///   - the RMSNorms are plain (`x·rsqrt(mean(x²) + 10⁻⁶)·w`, eps 10⁻⁶ in the code, not in the config), no
///     `1 + w` anywhere; `q_norm`/`k_norm` act per head over 128;
///   - **a LoRA reaches** `layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}` (the DMD LoRA: rank 128, fp32, one alpha
///     per module, `diffusion_model.` prefix), and `time_embedding.*`, `shared_block_modulation`,
///     `video_head_modulation` through the double path. The distilled model's own time embedding
///     (`time_embedding.safetensors`, a different sampler) is not this route.
package final class MossImageDiT {
    package struct Config {
        package let dim, heads, headDim, hidden, layers, textDim, timeDim, channels, textBlocks: Int
        package let ropeBase: Double

        package init(header: [String: Any]) {
            let c = header["config"] as? [String: Any] ?? [:]
            dim = c["dim"] as? Int ?? 3072
            heads = c["num_heads"] as? Int ?? 24
            headDim = dim / heads
            hidden = c["ffn_dim"] as? Int ?? 8192
            layers = c["num_layers"] as? Int ?? 42
            textDim = c["text_dim"] as? Int ?? 4096
            timeDim = c["freq_dim"] as? Int ?? 256
            channels = c["latent_channels"] as? Int ?? 128
            ropeBase = c["rope_base"] as? Double ?? 10000
            textBlocks = 2
        }
        /// Bands per axis (`head_dim / 8`): four axes, sin and cos, make `head_dim`.
        package var bands: Int { headDim / 8 }
    }

    /// The RMSNorm's eps: written in the model's code, not its config.
    package static let eps: Float = 1e-6

    package let config: Config
    package private(set) var boundaries: [String: [Float]] = [:]
    package var recordBoundaries = false
    /// The blocks whose output `boundaries` keeps (`block<N>_out`).
    package var recordedBlocks: Set<Int> = [0, 1, 21, 41]
    /// The token of the render in progress, consulted between two layers (see `Cancellation`).
    package var cancellation: Cancellation?
    package var timings: [String: Double] { ownTimings.merging(products.timings, uniquingKeysWith: +) }
    private var ownTimings: [String: Double] = [:]
    package func resetTimings() { ownTimings.removeAll(); products.resetTimings() }
    package var census: [String: (count: Int, seconds: Double, best: Double)] { gemm.census }

    private let artifact: Artifact
    private let prefetcher: Prefetcher
    private let gemm: GEMM
    private let arena: Arena
    private let kernels: MossImageKernels
    private let elementwise: ElementwiseGPU
    package let gridHeight, gridWidth, imageTokens, maxText, maxRows: Int
    private var attentions: [Int: Attention] = [:]
    private var splits: [Int: HeadSplitAttention] = [:]

    /// Where the SDPA is cut by heads between the GPU and the CPU: Krea 2's rule (`HeadSplitAttention.applies`,
    /// from `headSplitRows` rows) — the memory need reads the same condition.
    package static func headSplit(sequence: Int, heads: Int, headDim: Int) -> Bool {
        HeadSplitAttention.applies(sequence: sequence, heads: heads, headDim: headDim) && sequence >= headSplitRows
    }
    package static let headSplitRows = 2048

    private let x, normed, q, k, v, merged, o, h1, h2: UnsafeMutablePointer<Float>
    private let reserve, tokens, projected, freqs, textIn, modulationSlot: UnsafeMutablePointer<Float>
    /// The floats each slice reserves, by address: a Metal wrapper is always made at the slice's size.
    private let reserved: [UnsafeMutablePointer<Float>: Int]
    private var reserveCapacity: Int { config.dim * 2 * config.hidden }

    package let lora: LoRA?
    private let cacheLoRA: CacheLoRA?
    private let products: WeightProducts

    /// - Parameters:
    ///   - gridHeight, gridWidth: the latent's cells (`image / 16`); one token each.
    ///   - maxText: the most text rows an evaluation passes.
    ///   - expanded: the stack's expanded weights from a DiT given back, taken over rather than filled again (the
    ///     half-size start's, handed to the full one: the turbo LoRA's 1.19 GB of fp32 read once).
    package init(artifact: Artifact, gridHeight: Int, gridWidth: Int, maxText: Int = 512,
                 freezeCut: Bool = EngineSettings.effective.frozenCut, neuralEngine: Bool = EngineSettings.effective.gemmANE,
                 lora: LoRA? = nil, expanded: CacheLoRA? = nil) throws {
        guard artifact.header["kind"] as? String == MossImageDiT.kind else {
            throw Artifact.Failure.badHeader("map \(artifact.header["kind"] ?? "?"): \(MossImageDiT.kind) expected")
        }
        guard artifact.linearWeightsTransposed else {
            throw Artifact.Failure.badHeader("MOSS-Image map forged without transposed Linears")
        }
        self.artifact = artifact
        self.config = Config(header: artifact.header)
        self.prefetcher = Prefetcher(artifact: artifact)
        self.gemm = try GEMM(freezeCut: freezeCut, neuralEngine: neuralEngine)
        self.elementwise = try ElementwiseGPU(device: gemm.device, queue: gemm.queue)
        self.kernels = try MossImageKernels(device: gemm.device, queue: gemm.queue)
        self.gridHeight = gridHeight; self.gridWidth = gridWidth
        self.maxText = maxText
        imageTokens = gridHeight * gridWidth
        maxRows = imageTokens + maxText

        let r = lora?.maxRank ?? 0
        let sizes = MossImageDiT.slices(config: config, imageTokens: imageTokens, maxText: maxText, rank: r)
        let arena = try Arena(capacity: Arena.capacity(slices: sizes.map { $0.count * 4 }))
        self.arena = arena
        var slots: [String: UnsafeMutablePointer<Float>] = [:]
        var floats: [UnsafeMutablePointer<Float>: Int] = [:]
        for (name, count) in sizes {
            let pointer = try arena.reserve(name, bytes: count * 4).assumingMemoryBound(to: Float.self)
            slots[name] = pointer; floats[pointer] = count
        }
        reserved = floats
        func slot(_ name: String) -> UnsafeMutablePointer<Float> { slots[name]! }
        x = slot("x"); normed = slot("normed"); q = slot("q"); k = slot("k"); v = slot("v")
        merged = slot("merged"); o = slot("o"); h1 = slot("h1"); h2 = slot("h2")
        modulationSlot = slot("modulation"); reserve = slot("reserve"); tokens = slot("tokens")
        projected = slot("projected"); freqs = slot("freqs"); textIn = slot("textIn")
        self.lora = r > 0 ? lora : nil
        if let lora, r > 0 {
            let unapplied = lora.rankPerTarget.keys.filter { !MossImageDiT.appliesLoRA($0) }.sorted()
            if let first = unapplied.first {
                throw LoRA.Failure.inconsistentShape(target: first, reason: "a MOSS-Image module the engine does not adapt "
                    + "(\(unapplied.count) in all) — refused rather than let it be ignored")
            }
            let served = Set(MossImageDiT.loraTargets(config: config).map(\.name))
            if let expanded, expanded.lora === lora { cacheLoRA = expanded }
            else { cacheLoRA = try CacheLoRA(lora: lora, retain: served.contains) }
        } else {
            cacheLoRA = nil
        }
        products = WeightProducts(gemm: gemm, artifact: artifact, reserve: reserve, capacity: config.dim * 2 * config.hidden,
                                  transposed: true, lora: cacheLoRA, mid: r > 0 ? slot("loraMid") : nil,
                                  midFloats: maxRows * r, roundsActivations: true)
    }

    package static let kind = "moss-image-sft-dit"

    package var arenaCapacity: Int { arena.capacity }
    package var loraCacheCapacity: Int { cacheLoRA?.capacity ?? 0 }
    package var expandedLoRA: CacheLoRA? { cacheLoRA }

    // MARK: - what the pass reads, in the order it reads it

    package static func layerPrefix(_ layer: Int) -> String { "layers.\(layer)." }

    /// **A block's tensors in the order `block` widens them** — the order its stream reads them in.
    package static let consumptionOrder = ["norm1.weight", "proj.q.weight", "proj.k.weight", "proj.v.weight", "q_norm.weight",
                                           "k_norm.weight", "proj.o.weight", "norm2.weight", "proj.ffn_up_gate.weight",
                                           "proj.ffn_down.weight"]

    // MARK: - the modules a LoRA adapts

    /// A block's six Linears, in the map's order.
    package static let blockLinears = ["proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down"]

    /// **Every module a LoRA may touch in the DiT's GEMMs**, `[k, n]` as the map lays them out.
    package static func loraTargets(config: Config) -> [(name: String, k: Int, n: Int)] {
        let d = config.dim, f = config.hidden
        let shapes = [(d, d), (d, d), (d, d), (d, d), (d, 2 * f), (f, d)]
        return (0..<config.layers).flatMap { layer in
            zip(blockLinears, shapes).map { (name: layerPrefix(layer) + $0, k: $1.0, n: $1.1) }
        }
    }

    /// **The timestep's and modulations' `Linear`s**, adapted outside the GEMMs: unmerged, in double.
    package static let exactModules = ["time_embedding.0", "time_embedding.2", "shared_block_modulation", "video_head_modulation"]

    /// **Does the engine apply a LoRA on this module** (the map's name, without `.weight`)? The blocks' six
    /// Linears and the timestep path — not the text blocks, the embedders or the head's projection. What
    /// a map's LoRA is refused on: a module believed adapted that is not yields a plausible, wrong image.
    package static func appliesLoRA(_ module: String) -> Bool {
        if exactModules.contains(module) { return true }
        guard module.hasPrefix("layers.") else { return false }
        let rest = module.dropFirst("layers.".count)
        guard let dot = rest.firstIndex(of: "."), Int(rest[..<dot]) != nil else { return false }
        return blockLinears.contains(String(rest[rest.index(after: dot)...]))
    }

    private func record(_ name: String, _ p: UnsafePointer<Float>, _ n: Int) {
        guard recordBoundaries else { return }
        boundaries[name] = Array(UnsafeBufferPointer(start: p, count: n))
    }

    /// Rows `[text T | image S]` of a tensor `width` wide, recorded in the reference's order: image rows, then text rows.
    private func recordRows(_ name: String, _ p: UnsafePointer<Float>, width: Int, text t: Int) {
        guard recordBoundaries else { return }
        let s = imageTokens
        var out = [Float](repeating: 0, count: (s + t) * width)
        out.withUnsafeMutableBufferPointer { o in
            o.baseAddress!.update(from: p + t * width, count: s * width)
            (o.baseAddress! + s * width).update(from: p, count: t * width)
        }
        boundaries[name] = out
    }

    private func timed<T>(_ phase: String, _ body: () throws -> T) rethrows -> T {
        let started = DispatchTime.now().uptimeNanoseconds
        defer { ownTimings[phase, default: 0] += Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9 }
        return try body()
    }

    // MARK: - one evaluation

    /// **One evaluation.** `latent` as `[128, H, W]`, `text` as `[T, 4096]` (the encoder's last hidden
    /// state, no padding), `timestep` as the pipeline hands it to the transformer: the scheduler's
    /// `σ·1000` in fp32 (`MossImageSchedule.timesteps`). `skipping`: the zero-based blocks not computed
    /// (the official recipe's negative branch skips 22). Returns the velocity as `[128, H, W]`.
    package func forward(latent: UnsafePointer<Float>, text: UnsafePointer<Float>, textRows t: Int,
                         timestep: Float, skipping skipped: Set<Int> = []) throws -> [Float] {
        let d = config.dim, c = config.channels
        guard t >= 1, t <= maxText else {
            throw Artifact.Failure.badHeader("\(t) text tokens, between 1 and \(maxText)")
        }
        guard skipped.allSatisfy({ (0..<config.layers).contains($0) }) else {
            throw Artifact.Failure.misuse("MOSS-Image: skipped blocks \(skipped.sorted()) outside 0..<\(config.layers)")
        }
        let s = imageTokens, n = t + s

        let tabulated = modulations[timestep.bitPattern]
        if tabulated == nil {
            prefetcher.request(MossImageDiT.exactModules.flatMap { [$0 + ".weight", $0 + ".bias"] })
        }
        prefetcher.request(["video_embedder.weight", "video_embedder.bias", "text_embedder.proj.weight", "text_embedder.proj.bias"])
        prefetcher.request(prefetcher.namesOfBlock(prefix: "text_embedder.blocks."))
        // The blocks are read beside the page cache, the whole map (`generationTail`), through the staging
        // blocks the plan chosen in B gives — unless B holds the map whole. **A skipped block is released
        // without being computed**: the stream's sequence is the same at every evaluation.
        let layerBlocks = (0..<config.layers).map { prefetcher.namesOfBlock(prefix: MossImageDiT.layerPrefix($0)) }
        let (released, streamed) = MemoryPlan.ditTail(artifact: artifact, layerBlocks: layerBlocks,
                                                   fraction: MossImageDiT.generationTail)
        if !streamed {
            try artifact.streamTail(blocks: [], slots: 0)
        } else {
            try artifact.streamUnits(layers: layerBlocks, from: released, prefix: MossImageDiT.layerPrefix,
                                     suffixes: MossImageDiT.consumptionOrder)
        }
        prefetcher.request(layerBlocks[0])

        // ── the timestep: the six modulation vectors and the head's two, shared by every block ───
        let (mod, head) = try tabulated ?? timed("modulation") { try modulation(timesteps: [timestep])[0] }
        record("shared_modulation", mod, 6 * d)
        record("head_modulation", head, 2 * d)
        modulationSlot.update(from: mod, count: 6 * d)
        (modulationSlot + 6 * d).update(from: head, count: 2 * d)

        MossImageRope.write(into: freqs, textRows: t, tilesHigh: gridHeight, tilesWide: gridWidth,
                            headDim: config.headDim, base: config.ropeBase)
        recordRows("rope", freqs, width: config.headDim, text: t)

        // ── the text: Linear 4096 → 3072, then its two blocks, on rows [0, T) ───────────────────
        textIn.update(from: text, count: t * config.textDim)
        try linearGPU("text_embedder.proj.weight", a: textIn, into: x, m: t, k: config.textDim, n: d)
        try addBias("text_embedder.proj.bias", to: x, rows: t, columns: d)
        record("text_proj_out", x, t * d)
        for block in 0..<config.textBlocks {
            try autoreleasepool { try textBlock(block, rows: t) }
            record("text_block\(block)_out", x, t * d)
        }

        // ── the image: tokens [H·W, 128] → Linear → rows [T, T + S) (through `o`: page alignment) ──
        vDSP_mtrans(latent, 1, tokens, 1, vDSP_Length(s), vDSP_Length(c))   // [C, S] → [S, C]
        try linearGPU("video_embedder.weight", a: tokens, into: o, m: s, k: c, n: d)
        (x + t * d).update(from: o, count: s * d)
        try addBias("video_embedder.bias", to: x + t * d, rows: s, columns: d)
        record("video_embed", x + t * d, s * d)

        for layer in 0..<config.layers {
            try cancellation.check()   // between two layers, never inside one (see `Cancellation`)
            if skipped.contains(layer) {
                // Staged, then released: the stream's sequence (and so its read-ahead) stays the same.
                if layer >= released { prefetcher.request(layerBlocks[layer]); artifact.dropFromCache(layerBlocks[layer]) }
                continue
            }
            for ahead in 1...2 where layer + ahead < config.layers {
                prefetcher.request(layerBlocks[layer + ahead])
            }
            // An autorelease pool per layer, as Krea 2's: the head split's graph run wraps our buffers.
            try autoreleasepool { try block(layer, rows: n, text: t) }
            if layer >= released { artifact.dropFromCache(layerBlocks[layer]) }
            if recordedBlocks.contains(layer) { recordRows("block\(layer)_out", x, width: config.dim, text: t) }
        }

        // ── the head, on the image rows: RMSNorm · (1 + scale) + shift, then 3072 → 128 without bias ──
        let m = modulationSlot + 6 * d
        try artifact.materialize("final_norm_video.weight", into: reserve, capacity: reserveCapacity)
        timed("norm") {
            Ops.rmsNorm(x + t * d, weight: reserve, into: normed, rows: s, columns: d, eps: MossImageDiT.eps)
        }
        record("final_norm_out", normed, s * d)
        timed("norm") { Krea2Ops.modulate(normed, scale: m + d, shift: m, rows: s, columns: d) }
        try linearGPU("final_linear_video.weight", a: normed, into: projected, m: s, k: d, n: c)

        var out = [Float](repeating: 0, count: c * s)
        out.withUnsafeMutableBufferPointer {
            vDSP_mtrans(projected, 1, $0.baseAddress!, 1, vDSP_Length(c), vDSP_Length(s))   // [S, C] → [C, S]
        }
        record("model_out", out, out.count)
        return out
    }

    /// The whole map streams beside the page cache, as Krea 2's and Qwen-Image-2.1's generations do.
    package static let generationTail = 1.0

    // MARK: - the modulation

    /// The time vector's consumers for one timestep: `mod [6d]` and `head [2d]`, in fp32.
    package typealias ModulationTable = [UInt32: (mod: [Float], head: [Float])]
    package var modulations: ModulationTable = [:]

    /// **Precomputes the modulation of the timesteps that will be evaluated** — one read of the three
    /// weights for the whole render (`shared_block_modulation` is 56.6 M values).
    package func tabulateModulation(timesteps: [Float]) throws {
        let newKeys = Array(Set(timesteps.map(\.bitPattern)).subtracting(modulations.keys)).sorted()
        guard !newKeys.isEmpty else { return }
        let values = try timed("modulation") { try modulation(timesteps: newKeys.map(Float.init(bitPattern:))) }
        for (key, value) in zip(newKeys, values) { modulations[key] = value }
    }

    /// `time → (mod, head)` for several timesteps, in double, each weight read once for all.
    private func modulation(timesteps: [Float]) throws -> [(mod: [Float], head: [Float])] {
        let sinusoids = timesteps.map { MossImageExact.sinusoid(timestep: $0, dim: config.timeDim).map(Double.init) }
        var hidden = try Krea2Exact.linears(artifact, "time_embedding.0", xs: sinusoids, lora: lora)
        for i in hidden.indices { MossImageExact.silu(&hidden[i]) }
        let times = try Krea2Exact.linears(artifact, "time_embedding.2", xs: hidden, lora: lora)
        if recordBoundaries, let first = times.first, let sinusoid = sinusoids.first {
            boundaries["time_sinusoid"] = sinusoid.map(Float.init)
            boundaries["time_embedding_out"] = first.map(Float.init)
        }
        let mods = try Krea2Exact.linears(artifact, "shared_block_modulation", xs: times, lora: lora)
        let heads = try Krea2Exact.linears(artifact, "video_head_modulation", xs: times, lora: lora)
        return zip(mods, heads).map { ($0.map(Float.init), $1.map(Float.init)) }
    }

    // MARK: - the blocks

    /// A main block: attention then SITU feed-forward, modulated by the shared vectors.
    private func block(_ layer: Int, rows n: Int, text t: Int) throws {
        let d = config.dim, f = config.hidden, heads = config.heads, dh = config.headDim
        let prefix = MossImageDiT.layerPrefix(layer)
        let tag: String? = recordBoundaries && layer == 0 ? "block0_" : nil
        let m = modulationSlot
        let mb = try slice(m), xb = try slice(x), ob = try slice(o)
        // ── attention: shift_a m[0], scale_a m[1], gate_a m[2] ───────────────────────────────
        try modulatedNorm(prefix + "norm1.weight", x, rows: n, scale: m + d, shift: m)
        try linearGPU(prefix + "proj.q.weight", a: normed, into: q, m: n, k: d, n: d)
        try linearGPU(prefix + "proj.k.weight", a: normed, into: k, m: n, k: d, n: d)
        try linearGPU(prefix + "proj.v.weight", a: normed, into: v, m: n, k: d, n: d)
        if tag != nil {
            recordRows("block0_q_out", q, width: d, text: t)
            recordRows("block0_k_out", k, width: d, text: t)
            recordRows("block0_v_out", v, width: d, text: t)
        }
        try headNorm(prefix + "q_norm.weight", q, rows: n * heads)
        try headNorm(prefix + "k_norm.weight", k, rows: n * heads)
        if tag != nil { recordRows("block0_q_norm_out", q, width: d, text: t) }
        timed("rope") {
            MossImageRope.apply(q, freqs: freqs, sequence: n, heads: heads, headDim: dh)
            MossImageRope.apply(k, freqs: freqs, sequence: n, heads: heads, headDim: dh)
        }
        try attend(rows: n)
        try linearGPU(prefix + "proj.o.weight", a: merged, into: o, m: n, k: d, n: d)
        if tag != nil { recordRows("block0_o_out", o, width: d, text: t) }
        timed("norm") {
            ownTimings["elementwise GPU", default: 0] += elementwise.residual((xb, 0), plus: (ob, 0), carries: (mb, 2 * d),
                                                                          rows: n, columns: d)
        }
        // ── feed-forward: shift_f m[3], scale_f m[4], gate_f m[5] ─────────────────────────────
        try modulatedNorm(prefix + "norm2.weight", x, rows: n, scale: m + 4 * d, shift: m + 3 * d)
        try linearGPU(prefix + "proj.ffn_up_gate.weight", a: normed, into: h1, m: n, k: d, n: 2 * f)
        if tag != nil { recordRows("block0_ffn_up_gate_out", h1, width: 2 * f, text: t) }
        let (h1b, h2b) = (try slice(h1), try slice(h2))
        timed("situ") {
            ownTimings["elementwise GPU", default: 0] += kernels.situ(hidden: (h1b, 0), gate: (h1b, f), into: (h2b, 0),
                                                                   rows: n, columns: f, strides: (2 * f, 2 * f, f))
        }
        try linearGPU(prefix + "proj.ffn_down.weight", a: h2, into: o, m: n, k: f, n: d)
        if tag != nil { recordRows("block0_ffn_down_out", o, width: d, text: t) }
        timed("norm") {
            ownTimings["elementwise GPU", default: 0] += elementwise.residual((xb, 0), plus: (ob, 0), carries: (mb, 5 * d),
                                                                          rows: n, columns: d)
        }
    }

    /// A text block: plain pre-norm transformer block with biases, over the text rows `[0, T)` alone.
    private func textBlock(_ index: Int, rows t: Int) throws {
        let d = config.dim, f = config.hidden, heads = config.heads
        let prefix = "text_embedder.blocks.\(index)."
        // ── attention: the fused QKV, split row by row into q, k, v ─────────────────────────
        try plainNorm(prefix + "norm1.weight", x, rows: t)
        try linearGPU(prefix + "qkv.weight", a: normed, into: h1, m: t, k: d, n: 3 * d)
        try addBias(prefix + "qkv.bias", to: h1, rows: t, columns: 3 * d)
        let (qkv, qOut, kOut, vOut) = (h1, q, k, v)
        Parallel.rows(t, width: 3 * d) { first, howMany in
            for row in first..<(first + howMany) {
                let source = qkv + row * 3 * d
                (qOut + row * d).update(from: source, count: d)
                (kOut + row * d).update(from: source + d, count: d)
                (vOut + row * d).update(from: source + 2 * d, count: d)
            }
        }
        try headNorm(prefix + "q_norm.weight", q, rows: t * heads)
        try headNorm(prefix + "k_norm.weight", k, rows: t * heads)
        timed("rope") {
            MossImageRope.apply(q, freqs: freqs, sequence: t, heads: heads, headDim: config.headDim)
            MossImageRope.apply(k, freqs: freqs, sequence: t, heads: heads, headDim: config.headDim)
        }
        try attend(rows: t)
        try linearGPU(prefix + "proj.weight", a: merged, into: o, m: t, k: d, n: d)
        try addBias(prefix + "proj.bias", to: o, rows: t, columns: d)
        vDSP_vadd(x, 1, o, 1, x, 1, vDSP_Length(t * d))
        // ── feed-forward: ffn_up and ffn_gate apart, SITU(up, gate) ──────────────────────────
        try plainNorm(prefix + "norm2.weight", x, rows: t)
        try linearGPU(prefix + "ffn_up.weight", a: normed, into: h1, m: t, k: d, n: f)
        try addBias(prefix + "ffn_up.bias", to: h1, rows: t, columns: f)
        try linearGPU(prefix + "ffn_gate.weight", a: normed, into: h2, m: t, k: d, n: f)
        try addBias(prefix + "ffn_gate.bias", to: h2, rows: t, columns: f)
        let (h1b, h2b) = (try slice(h1), try slice(h2))
        timed("situ") {
            ownTimings["elementwise GPU", default: 0] += kernels.situ(hidden: (h1b, 0), gate: (h2b, 0), into: (h1b, 0),
                                                                   rows: t, columns: f, strides: (f, f, f))
        }
        try linearGPU(prefix + "ffn_down.weight", a: h1, into: o, m: t, k: f, n: d)
        try addBias(prefix + "ffn_down.bias", to: o, rows: t, columns: d)
        vDSP_vadd(x, 1, o, 1, x, 1, vDSP_Length(t * d))
    }

    // ── plumbing ────────────────────────────────────────────────────────────────────────

    /// A slice of the arena seen from the GPU — wrapped at its reserved size.
    private func slice(_ p: UnsafeMutablePointer<Float>) throws -> MTLBuffer {
        try gemm.wrap(UnsafeMutableRawPointer(p), bytes: reserved[p]! * 4, name: "elementwise")
    }

    /// `normed = RMSNorm(src)·w` — the text blocks' norms, no modulation.
    private func plainNorm(_ name: String, _ source: UnsafePointer<Float>, rows: Int) throws {
        try artifact.materialize(name, into: reserve, capacity: reserveCapacity)
        timed("norm") { Ops.rmsNorm(source, weight: reserve, into: normed, rows: rows, columns: config.dim, eps: MossImageDiT.eps) }
    }

    /// `normed = RMSNorm(src)·w·(1 + scale) + shift`, on the CPU (see `Krea2DiT.modulatedNorm`).
    private func modulatedNorm(_ name: String, _ source: UnsafePointer<Float>, rows: Int,
                               scale: UnsafePointer<Float>, shift: UnsafePointer<Float>) throws {
        try plainNorm(name, source, rows: rows)
        timed("norm") { Krea2Ops.modulate(normed, scale: scale, shift: shift, rows: rows, columns: config.dim) }
    }

    /// QK-Norm: RMSNorm over `head_dim`.
    private func headNorm(_ name: String, _ tensor: UnsafeMutablePointer<Float>, rows: Int) throws {
        try artifact.materialize(name, into: reserve, capacity: reserveCapacity)
        timed("norm") { Ops.rmsNorm(tensor, weight: reserve, into: tensor, rows: rows,
                                    columns: config.headDim, eps: MossImageDiT.eps) }
    }

    private func addBias(_ name: String, to c: UnsafeMutablePointer<Float>, rows: Int, columns: Int) throws {
        try artifact.materialize(name, into: reserve, capacity: reserveCapacity)
        let bias = UnsafePointer(reserve)
        Parallel.rows(rows, width: columns) { first, howMany in
            for row in first..<(first + howMany) {
                vDSP_vadd(c + row * columns, 1, bias, 1, c + row * columns, 1, vDSP_Length(columns))
            }
        }
    }

    private func attend(rows n: Int) throws {
        let qb = try slice(q), kb = try slice(k), vb = try slice(v), ob = try slice(merged)
        if MossImageDiT.headSplit(sequence: n, heads: config.heads, headDim: config.headDim) {
            let split: HeadSplitAttention
            if let s = splits[n] { split = s } else {
                split = try HeadSplitAttention(device: gemm.device, queue: gemm.queue, heads: config.heads, sequence: n,
                                               headDim: config.headDim)
                splits[n] = split
            }
            timed("sdpa wall") {
                let (gpu, cpu) = split.run(q: qb, k: kb, v: vb, into: ob, q: q, k: k, v: v, out: merged)
                ownTimings["sdpa GPU", default: 0] += gpu
                ownTimings["sdpa CPU", default: 0] += cpu
            }
            return
        }
        let attention: Attention
        if let a = attentions[n] { attention = a } else {
            attention = Attention(device: gemm.device, queue: gemm.queue, heads: config.heads,
                                  sequence: n, headDim: config.headDim)
            attentions[n] = attention
        }
        timed("sdpa wall") { ownTimings["sdpa GPU", default: 0] += attention.run(q: qb, k: kb, v: vb, into: ob) }
    }

    /// `C[m, n] = A[m, k] · W`, W laid out `[k, n]` by the forge. Both buffers wrapped at their slice's size.
    private func linearGPU(_ name: String, a: UnsafeMutablePointer<Float>, into c: UnsafeMutablePointer<Float>,
                           m: Int, k: Int, n: Int) throws {
        let ab = try gemm.wrap(UnsafeMutableRawPointer(a), bytes: reserved[a]! * 4, name: "a")
        let cb = try gemm.wrap(UnsafeMutableRawPointer(c), bytes: reserved[c]! * 4, name: "c")
        try products.linear(name, a: ab, c: cb, m: m, k: k, n: n)
    }
}

// MARK: - the scheduler

/// **The sigmas of `MOSSImageScheduler`** — Euler flow matching on a grid the model was trained on.
///
///     grid(count, shift)  = shift·v / (1 + (shift − 1)·v),  v = linspace(1, shift/(N + shift − 1), count)   fp32
///     σ                   = grid(steps, 5), each σ moved to the nearest value of grid(1000, 1) reversed
///                           (`searchsorted`, then the nearer of the two neighbours, ties down), then
///                           duplicates removed — **fewer σ than steps** when the schedule is dense
///
/// What a port by analogy would miss: the **rounding to the training grid** (a shifted `linspace` alone gives
/// 0.9679 where the reference has 0.968), the dedup (250 steps give 246 evaluations, 1000 give 615), the
/// shift applied **once** (the scheduler resets its own to 1 before `set_timesteps`), `training_shift` 1, and
/// timesteps = `σ·1000` in fp32 (463.00003 for 0.463), the last σ being followed by 0.
package enum MossImageSchedule {
    package static let trainTimesteps = 1000

    /// `torch.linspace(start, end, count)` in fp32 as torch computes it on this machine: from the start by `step`
    /// for the first half, from the end by `step` backwards for the rest — each value **one fused
    /// multiply-add** (`start + step·i`), which a separate product and sum misses on 239 of the training
    /// grid's 1000 values.
    static func linspace(_ start: Float, _ end: Float, _ count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Float(count - 1), halfway = count / 2
        return (0..<count).map {
            $0 < halfway ? start.addingProduct(step, Float($0)) : end.addingProduct(-step, Float(count - $0 - 1))
        }
    }

    static func grid(_ count: Int, shift: Double, total: Int = trainTimesteps) -> [Float] {
        let lower = Float(shift / (Double(total) + shift - 1))
        let s = Float(shift)
        return linspace(1, lower, count).map { s * $0 / (1 + (s - 1) * $0) }
    }

    /// The grid the model was trained on, ascending: `0.001 … 1`.
    package static var trainingGrid: [Float] { Array(grid(trainTimesteps, shift: 1).reversed()) }

    /// The σ the scheduler walks for `steps`, **followed by the final 0**. `shift` 5 and `trainingShift` 1
    /// are `scheduler_config.json`'s.
    package static func sigmas(steps: Int, shift: Double = 5, trainingShift: Double = 1) -> [Float] {
        precondition(steps >= 1 && shift > 0 && trainingShift > 0, "steps and shifts must be positive")
        let total = trainTimesteps
        let wanted = grid(steps, shift: shift, total: total)
        let training = Array(grid(total, shift: trainingShift, total: total).reversed())   // ascending
        var snapped: [Float] = []
        for sigma in wanted {
            // searchsorted, left: the first index whose value is not below σ.
            var low = 0, high = training.count
            while low < high {
                let mid = (low + high) / 2
                if training[mid] < sigma { low = mid + 1 } else { high = mid }
            }
            let upper = min(low, total - 1), lower = max(upper - 1, 0)
            let nearest = abs(sigma - training[lower]) <= abs(training[upper] - sigma) ? lower : upper
            let value = training[nearest]
            if snapped.last != value { snapped.append(value) }
        }
        return snapped + [0]
    }

    /// What the transformer is told at each evaluation: `σ·1000`, fp32 (the scheduler's `timesteps`).
    package static func timesteps(_ sigmas: [Float]) -> [Float] {
        sigmas.dropLast().map { $0 * Float(trainTimesteps) }
    }
}

// MARK: - the position embedding

/// **MOSS-Image's RoPE**, four axes `(r, f, h, w)`, `head_dim / 8` bands each, **half-split**.
///
/// Entry `axis·bands + band` of the table's first half is `sin(coordinate[axis] · bands[band])`, of its second
/// half the cosine, with `bands[i] = 1 / θ^(i/16)` in fp32; `x' = x·cos + rotate_half(x)·sin`, where the
/// rotated pairs are `(j, j + head_dim/2)`.
package enum MossImageRope {
    /// The four coordinates of every row, text rows first then the image's `tilesHigh · tilesWide` in
    /// row-major order. The reference lays out the image first; the coordinates travel with the row.
    package static func coordinates(textRows t: Int, tilesHigh: Int, tilesWide: Int) -> [[Float]] {
        var rows: [[Float]] = (0..<t).map { [0, Float($0), 0, 0] }
        let centreH = Float(tilesHigh - 1) / 2, centreW = Float(tilesWide - 1) / 2
        for h in 0..<tilesHigh {
            for w in 0..<tilesWide { rows.append([0, Float(t), Float(h) - centreH, Float(w) - centreW]) }
        }
        return rows
    }

    /// `1 / θ^(i/bands)`, fp32 (`_default_bands`).
    package static func bands(headDim: Int, base: Double) -> [Float] {
        let count = headDim / 8
        return (0..<count).map { 1 / powf(Float(base), Float($0) / Float(count)) }
    }

    /// Writes the `[rows, head_dim]` table: `[sin(head_dim/2) | cos(head_dim/2)]` per row.
    package static func write(into out: UnsafeMutablePointer<Float>, textRows t: Int, tilesHigh: Int, tilesWide: Int,
                              headDim: Int, base: Double) {
        let bands = bands(headDim: headDim, base: base), count = bands.count, half = headDim / 2
        let rows = coordinates(textRows: t, tilesHigh: tilesHigh, tilesWide: tilesWide)
        nonisolated(unsafe) let out = out
        Parallel.rows(rows.count, width: headDim) { first, howMany in
            for r in first..<(first + howMany) {
                let row = out + r * headDim
                for axis in 0..<4 {
                    for b in 0..<count {
                        let phase = rows[r][axis] * bands[b]
                        row[axis * count + b] = sinf(phase)
                        row[half + axis * count + b] = cosf(phase)
                    }
                }
            }
        }
    }

    /// `x[s, h, :] ← x·cos + rotate_half(x)·sin` over `[S, H, Dh]`, in place. The products are two roundings
    /// each, the sum of the two (torch's `x * cos + cat(-b, a) * sin`).
    package static func apply(_ x: UnsafeMutablePointer<Float>, freqs: UnsafePointer<Float>,
                              sequence: Int, heads: Int, headDim: Int) {
        let half = headDim / 2
        Parallel.rows(sequence, width: heads * headDim) { first, howMany in
            for s in first..<(first + howMany) {
                let sin = freqs + s * headDim, cos = sin + half
                for h in 0..<heads {
                    let row = x + (s * heads + h) * headDim
                    for j in 0..<half {
                        let a = row[j], b = row[j + half]
                        let ac = a * cos[j], bs = (-b) * sin[j]
                        let bc = b * cos[j], asn = a * sin[j]
                        row[j] = ac + bs
                        row[j + half] = bc + asn
                    }
                }
            }
        }
    }
}

// MARK: - the pieces in double and the activation

package enum MossImageExact {
    /// The sinusoid, **computed in double then rounded once to fp32**: `[cos | sin]` of `t·10⁴^(−j/half)`,
    /// `t` the fp32 timestep the scheduler hands.
    package static func sinusoid(timestep: Float, dim: Int) -> [Float] {
        let half = dim / 2
        var out = [Float](repeating: 0, count: dim)
        for j in 0..<half {
            let angle = Double(timestep) * pow(10000, -Double(j) / Double(half))
            out[j] = Float(cos(angle))
            out[half + j] = Float(sin(angle))
        }
        return out
    }

    static func silu(_ x: inout [Double]) {
        for i in x.indices { x[i] = x[i] / (1 + exp(-x[i])) }
    }

    /// **SITU**, `hidden` the value and `gate` the gate: `4·tanh(g/4)·σ(g) · 25·tanh(h/25)` — both capped
    /// (the gate's first factor in ±4, the value in ±25), then multiplied. In double: the definition the
    /// kernel is judged against.
    package static func situ(hidden: Double, gate: Double) -> Double {
        (4 * tanh(gate / 4) * (1 / (1 + exp(-gate)))) * (25 * tanh(hidden / 25))
    }
}

// MARK: - the SITU kernel

/// **SITU on the GPU**: `out = 4·tanh(g/4)·σ(g) · 25·tanh(h/25)`, the order of the reference's operations
/// (`_situ`), `precise::` functions, no fast-math. Reads `hidden` and `gate` from one buffer (the fused
/// `ffn_up_gate`'s `[hidden | gate]` rows) or two, at their row strides; `out` may be `hidden` itself.
package final class MossImageKernels {
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void situ(device const float *h [[buffer(0)]], device const float *g [[buffer(1)]],
                     device float *out [[buffer(2)]], constant uint &cols [[buffer(3)]],
                     constant uint4 &strides [[buffer(4)]], constant uint2 &offsets [[buffer(5)]],
                     uint2 p [[thread_position_in_grid]]) {
        #pragma METAL fp contract(off)
        if (p.x >= cols) return;
        float hv = h[offsets.x + p.y * strides.x + p.x];
        float gv = g[offsets.y + p.y * strides.y + p.x];
        float gate = 4.0f * precise::tanh(gv / 4.0f) * precise::divide(1.0f, 1.0f + precise::exp(-gv));
        float value = 25.0f * precise::tanh(hv / 25.0f);
        out[p.y * strides.z + p.x] = gate * value;
    }
    """

    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    package init(device: MTLDevice, queue: MTLCommandQueue) throws {
        self.queue = queue
        let options = MTLCompileOptions()
        options.mathMode = .safe
        let library = try device.makeLibrary(source: Self.source, options: options)
        guard let function = library.makeFunction(name: "situ") else { throw GEMM.Failure.noDevice }
        pipeline = try device.makeComputePipelineState(function: function)
    }

    /// `out[r, j] = SITU(hidden[r, j], gate[r, j])` for `rows × columns`; each operand is a buffer and an offset
    /// in floats, `strides` the row strides in floats of `(hidden, gate, out)`. Returns the GPU time.
    @discardableResult
    package func situ(hidden: ElementwiseGPU.BufferSlice, gate: ElementwiseGPU.BufferSlice, into out: ElementwiseGPU.BufferSlice,
                      rows: Int, columns: Int, strides: (Int, Int, Int)) -> Double {
        guard rows > 0, columns > 0, let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else {
            Attention.report("GPU SITU: no command buffer, so nothing ran")
            return 0
        }
        var cols = UInt32(columns)
        var s = SIMD4<UInt32>(UInt32(strides.0), UInt32(strides.1), UInt32(strides.2), 0)
        var offsets = SIMD2<UInt32>(UInt32(hidden.offset), UInt32(gate.offset))
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(hidden.buffer, offset: 0, index: 0)
        encoder.setBuffer(gate.buffer, offset: 0, index: 1)
        encoder.setBuffer(out.buffer, offset: out.offset * 4, index: 2)
        encoder.setBytes(&cols, length: 4, index: 3)
        encoder.setBytes(&s, length: MemoryLayout<SIMD4<UInt32>>.size, index: 4)
        encoder.setBytes(&offsets, length: MemoryLayout<SIMD2<UInt32>>.size, index: 5)
        encoder.dispatchThreads(MTLSize(width: columns, height: rows, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        return Attention.verdict(commands, what: "GPU SITU")
    }
}
