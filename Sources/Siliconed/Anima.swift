import Metal
import Accelerate
import Foundation

/// The Anima Turbo DiT — Cosmos-Predict2-2B lineage —, from the latent to the predicted velocity, in fp32.
///
///     latent [16, H, W] + padding mask (zeros) ─ patchify (SLOW channel) ─ patch_embed ───────┐
///                                                                                            │
///     σ ─ sinusoid 2048 ──┬─ linear_1 · SiLU · linear_2 ─ temb [6144] ──────────────┐        │
///                         └─ RMSNorm ─ embedded [2048] ─ SiLU ──────────────────────┤        │
///                                                                                   ▼        ▼
///     text [512, 1024] ─────────────────────────────────────────────────▶ blocks ×28 ─ norm_out ─ proj_out ─ unpatchify
///
/// One block, three sublayers, **each with its own modulation**:
///
///     shift, scale, gate = linear_2ᵢ(linear_1ᵢ(SiLU(embedded))) + temb     (i = 1, 2, 3)
///     x += gate₁ · attn1( LN(x)·(1+scale₁) + shift₁ )     self-attention, QK-Norm, RoPE
///     x += gate₂ · attn2( LN(x)·(1+scale₂) + shift₂ , text )   cross, QK-Norm, NO RoPE
///     x += gate₃ · W₂ GELU( W₁ (LN(x)·(1+scale₃) + shift₃) )
///
/// What sets it apart from the Z-Image block, and what a port by analogy would miss:
///
///   - **a shift** in the modulation, and **no tanh** on the gate;
///   - a **low-rank** modulation (`adaln_lora_dim = 256`) *plus* `temb`, the same for the
///     three sublayers;
///   - an **exact GELU** FFN (`erf`), ungated — not a SwiGLU;
///   - patchify lays out the **slowest channel** (`c·4 + ph·2 + pw`), and it carries a seven-
///     teenth channel, the padding mask, zero for a whole image;
///   - Cosmos RoPE is *split*: the forge converted it to interleaved (`ForgeDiT`),
///     so `Ops.rope` applies as-is on a table specific to Cosmos (`AnimaRope`);
///   - that table is **bounded**: `max(max_size // patch_size)` = 128 positions per side, 2048 px —
///     the reference cannot compute past it, so `Format.maximumSide(for: .anima)` refuses it
///     (`Config.ropePositions`), where the formula alone would extrapolate without a word;
///   - the **residuals climb to 2.5·10⁵** at block 13 (golden tensors): four times the fp16
///     ceiling. Anima requires fp32 as much as Z-Image does;
///   - **every modulation depends on σ alone** (not on the latent, not on the text): the 170 `gemv`s of
///     an evaluation are computed for all the render's σ before the first one (`tabulate`), each weight
///     read once a render;
///   - the cross-attention's text k/v do not change between evaluations either: cached (`textKCache`).
///
/// Where it runs, at 1024² (measured on the M1 Pro, an evaluation ~5.4 s): the GEMMs on the GPU and the AMX
/// (`Conductor`, ~3.0 s, at the pair's ceiling), the self- and cross-attention cut by heads between the GPU
/// and the CPU (`HeadSplitAttention`, ~1.9 s), the GELU on the GPU (`AnimaGELU`); the norms, QK-Norm with
/// RoPE (one pass), the residuals and the modulations (in double) on the CPU, ~0.4 s.
package final class AnimaDiT {
    package struct Config {
        package let dim, heads, headDim, hidden, layers, channels, patch, adalnLoRA, textDim: Int
        package let ropeScale: [Double]
        /// **The positions the reference's RoPE table holds, per side**: `CosmosRotaryPosEmbed` builds
        /// `seq = arange(max(max_size // patch_size))` and slices `seq[:tokens]` on each axis — beyond it,
        /// the slice comes back short and the reference fails on the concatenation. Note the `max` over
        /// the three axes: with the published `max_size = (128, 240, 240)`, `patch_size = (1, 2, 2)` it is
        /// the *time* axis' 128 that bounds the image's sides, not 240 // 2 = 120.
        package let ropePositions: Int

        /// `max(max_size // patch_size)` (`CosmosRotaryPosEmbed.__init__` then `forward`).
        package static func ropePositions(maxSize: [Int], patchSize: [Int]) -> Int {
            zip(maxSize, patchSize).map { $0 / $1 }.max() ?? 0
        }
        /// The published configuration's (`max_size = [128, 240, 240]`, `patch_size = [1, 2, 2]`, read in
        /// the forged map's header): 128 tokens, 2048 px on a side through the VAE ×8 and the patch ×2.
        package static let publishedRopePositions = ropePositions(maxSize: [128, 240, 240], patchSize: [1, 2, 2])

        package init(header: [String: Any]) {
            let c = header["config"] as? [String: Any] ?? [:]
            heads = c["num_attention_heads"] as? Int ?? 16
            headDim = c["attention_head_dim"] as? Int ?? 128
            dim = heads * headDim
            hidden = Int(Double(dim) * (c["mlp_ratio"] as? Double ?? 4))
            layers = c["num_layers"] as? Int ?? 28
            channels = c["in_channels"] as? Int ?? 16
            patch = (c["patch_size"] as? [Int])?.last ?? 2
            adalnLoRA = c["adaln_lora_dim"] as? Int ?? 256
            textDim = c["text_embed_dim"] as? Int ?? 1024
            ropeScale = (c["rope_scale"] as? [Double]) ?? [1, 4, 4]
            ropePositions = Config.ropePositions(maxSize: (c["max_size"] as? [Int]) ?? [128, 240, 240],
                                                 patchSize: (c["patch_size"] as? [Int]) ?? [1, 2, 2])
        }
    }

    package let config: Config
    package private(set) var boundaries: [String: [Float]] = [:]
    package var recordBoundaries = false
    /// The token of the render in progress, consulted between two layers (see `Cancellation`).
    package var cancellation: Cancellation?
    /// The weight products' phases (`WeightProducts`) added to the DiT's own.
    package var timings: [String: Double] { ownTimings.merging(products.timings, uniquingKeysWith: +) }
    private var ownTimings: [String: Double] = [:]
    package var conductor: Conductor? { gemm.conductor }
    package var census: [String: (count: Int, seconds: Double, best: Double)] { gemm.census }

    private let artifact: Artifact
    private let prefetcher: Prefetcher
    private let gemm: GEMM
    private let arena: Arena
    private let transposed: Bool
    /// The sizing grid, which is that of every evaluation: the half-size start builds a DiT of its own
    /// (`AnimaDenoisingModule.denoise`).
    private let latentHeight, latentWidth, imageTokens, textLength: Int
    /// **The LoRA stack** (`ForgeLoRA`, Anima's names and RoPE): `ΔW = B·A` per module, the
    /// GEMMs through `gemm.lora` (two thin products accumulated into the output), the modulation
    /// `gemv`s in double like their main weight. The cached text k/v carry it too:
    /// they are computed once, LoRA included.
    package let lora: LoRA?
    private let loraDown, loraUp, loraMid: UnsafeMutablePointer<Float>?
    /// The products by a weight, and their LoRA stack (expanded at each call into `loraBas`/`loraHaut`).
    private let products: WeightProducts
    /// The GELU on the GPU.
    private let gelu: AnimaGELU

    // Arena slices, reserved once.
    private let x, normed, q, k, v, merged, o, h1, reserve: UnsafeMutablePointer<Float>
    /// The floats `reserve` holds (`d·h`, the largest weight): what `materialize` may write into it.
    private var reserveCapacity: Int { config.dim * config.hidden }
    private let tokens, projected, freqs, text: UnsafeMutablePointer<Float>
    /// **The text `k`/`v`, per layer, computed once per context**.
    ///
    /// Cross-attention projects the same text at every step: neither σ nor the latent enter into it,
    /// and neither does `norm_k`. The 8 evaluations of a render thus recomputed the same thing 8 times —
    /// 0.08–0.09 s per evaluation, measured by a probe (≈ 1.2%). Kept here, they cost
    /// 28 × 2 × 512 × 2048 × 4 = 235 MB of arena, and the computation is the same, so the bits are too.
    /// The cache is valid as long as the received context is **identical byte for byte** to the
    /// previous one: another prompt in the same DiT recomputes it, it does not reuse it.
    private let textKCache, textVCache: UnsafeMutablePointer<Float>
    private var textCacheValid = false
    private let scaleOne: UnsafeMutablePointer<Float>
    /// `AnimaExact`'s double copy of a modulation `gemv`'s weight (`gemv`): the widest is
    /// `time_embed.t_embedder.linear_2`, `d × 3d` — 100 MB of doubles, sized by `exactProducts`. Its input
    /// and output, `m` rows of them, are the modulation table's (`tableSlices`).
    private let exactW: UnsafeMutablePointer<Double>
    /// The doubles `exactW` holds: the largest weight of `exactProducts`.
    private let exactWeight: Int

    /// - Parameter textLength: the length of the text context — 512 at least, the conditioner
    ///   padding its output with zeros up to that. The zeros **count**: the reference does not
    ///   mask cross-attention, so each zero row takes its share of the softmax.
    package init(artifact: Artifact, latentHeight: Int, latentWidth: Int, textLength: Int = 512,
                freezeCut: Bool = EngineSettings.effective.frozenCut, neuralEngine: Bool = EngineSettings.effective.gemmANE, lora: LoRA? = nil) throws {
        guard artifact.header["kind"] as? String == "anima-turbo-dit" else {
            throw Artifact.Failure.badHeader("map \(artifact.header["kind"] ?? "?"): anima-turbo-dit expected")
        }
        guard artifact.header["rope_layout"] as? String == "interleaved" else {
            throw Artifact.Failure.badHeader("RoPE not converted by the forge — `Ops.rope` would be wrong")
        }
        self.artifact = artifact
        self.config = Config(header: artifact.header)
        // `Format.check(…, family: .anima)` refuses these sides first; this guards a hand-made call.
        guard latentHeight / config.patch <= config.ropePositions, latentWidth / config.patch <= config.ropePositions else {
            let side = config.ropePositions * config.patch * LatentSpace.qwenImage.factor
            throw EngineError.formatRefused(width: latentWidth * LatentSpace.qwenImage.factor,
                                            height: latentHeight * LatentSpace.qwenImage.factor,
                                            reason: .sideTooLong, bound: side)
        }
        self.prefetcher = Prefetcher(artifact: artifact)
        self.gemm = try GEMM(freezeCut: freezeCut, neuralEngine: neuralEngine)
        self.gelu = try AnimaGELU(device: gemm.device, queue: gemm.queue)
        self.transposed = artifact.linearWeightsTransposed
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
        self.textLength = textLength
        imageTokens = AnimaDiT.imageTokens(latentHeight: latentHeight, latentWidth: latentWidth, config: config)

        let r = lora?.maxRank ?? 0
        // The arena is the sum of its slices (`slices`), each on its page — the function the memory
        // need charges (`MemoryNeed.anima`): the formula equals the allocation by construction. It
        // used to be the floats counted inline, plus 64 pages and 8 MB of margin no formula could see.
        let sized = AnimaDiT.slices(config: config, imageTokens: imageTokens, textLength: textLength,
                                    lora: r > 0 ? (r, lora!.maxEntry, lora!.maxOutput) : nil)
        let bytes = Dictionary(uniqueKeysWithValues: sized.map { ($0.name, $0.bytes) })
        // LOCAL binding: a nested function cannot capture `self` before the end of
        // initialization (same constraint as in `Attention`).
        let arena = try Arena(capacity: Arena.capacity(slices: sized.map(\.bytes)))
        self.arena = arena
        func raw(_ name: String) throws -> UnsafeMutableRawPointer { try arena.reserve(name, bytes: bytes[name]!) }
        func slot(_ name: String) throws -> UnsafeMutablePointer<Float> { try raw(name).assumingMemoryBound(to: Float.self) }
        func wide(_ name: String) throws -> UnsafeMutablePointer<Double> { try raw(name).assumingMemoryBound(to: Double.self) }
        x = try slot("x"); normed = try slot("normed")
        q = try slot("q"); k = try slot("k"); v = try slot("v")
        merged = try slot("merged"); o = try slot("o")
        // `h1` lies over `q`…`merged`, dead by the feed-forward (adjacent in the arena, `4d = h` floats a row).
        h1 = q; reserve = try slot("reserve")
        guard UnsafeMutableRawPointer(h1 + imageTokens * config.hidden) <= UnsafeMutableRawPointer(o) else {
            throw Artifact.Failure.badHeader("Anima: h1 does not fit over the attention's scratch")
        }
        tokens = try slot("tokens"); projected = try slot("projected")
        freqs = try slot("freqs")
        text = try slot("text")
        textKCache = try slot("textK")
        textVCache = try slot("textV")
        scaleOne = try slot("scaleOne")
        exactW = try wide("exactW")
        exactWeight = AnimaDiT.exactProducts(config: config).map { $0.k * $0.n }.max()!
        self.lora = r > 0 ? lora : nil
        if r > 0 {
            loraDown = try slot("loraBas")
            loraUp = try slot("loraHaut")
            loraMid = try slot("loraMid")
        } else {
            loraDown = nil; loraUp = nil; loraMid = nil
        }
        products = WeightProducts(gemm: gemm, artifact: artifact, reserve: reserve, capacity: config.dim * config.hidden,
                                  transposed: transposed,
                                  lora: r > 0 ? ExpandedLoRA(lora: lora!, down: loraDown!, up: loraUp!) : nil,
                                  mid: loraMid, midFloats: max(imageTokens, textLength) * r, roundsActivations: true)
    }

    /// The bytes this DiT's arena maps — what the tests compare with `AnimaDiT.arenaBytes`.
    package var arenaCapacity: Int { arena.capacity }

    private func record(_ name: String, _ p: UnsafePointer<Float>, _ n: Int) {
        guard recordBoundaries else { return }
        boundaries[name] = Array(UnsafeBufferPointer(start: p, count: n))
    }

    private func timed<T>(_ phase: String, _ body: () throws -> T) rethrows -> T {
        let started = DispatchTime.now().uptimeNanoseconds
        defer { ownTimings[phase, default: 0] += Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9 }
        return try body()
    }

    /// `latent` as `[16, H, W]`, `context` as `[textLength, 1024]` (the conditioner output,
    /// padded), `sigma` as the scheduler gives it — the pipeline passes `t / 1000`, `t = σ·1000`, which
    /// the timestep table retraces (`tabulate`). Returns the velocity as `[16, H, W]`.
    package func forward(latent: UnsafePointer<Float>, context: UnsafePointer<Float>,
                        sigma: Float) throws -> [Float] {
        let d = config.dim, p = config.patch, c = config.channels
        let (height, width) = (latentHeight, latentWidth)
        guard height % p == 0, width % p == 0 else {
            throw Artifact.Failure.badHeader("latent \(width)×\(height): multiple of \(p) expected")
        }
        let s = imageTokens
        let inDim = (c + 1) * p * p, outDim = c * p * p

        prefetcher.request(artifact.order.filter { !$0.hasPrefix("transformer_blocks.") && !$0.hasPrefix("norm_out.") && !$0.hasPrefix("proj_out.") })
        prefetcher.request(prefetcher.namesOfBlock(prefix: "transformer_blocks.0."))

        // ── the timestep: temb, embedded and every modulation, from the table (`tabulate`) ──────
        try ensureTabulated(sigma)
        let row = try modulationRow(sigma)
        record("temb", row + AnimaDiT.ModulationRow.temb(config), 3 * d)
        record("embedded_timestep", row + AnimaDiT.ModulationRow.embedded(config), d)

        // ── patchify, padding mask included, then patch_embed ────────────────────────────────
        AnimaOps.patchify(latent, into: tokens, channels: c, height: height, width: width, patch: p)
        try linearGPU("patch_embed.proj.weight", a: tokens, into: x, m: s, k: inDim, n: d,
                      rowsA: imageTokens, rowsC: imageTokens)
        record("patch_embed_out", x, s * d)

        AnimaRope.write(into: freqs, tilesHigh: height / p, tilesWide: width / p,
                        headDim: config.headDim, scale: config.ropeScale)
        if !textCacheValid || memcmp(text, context, textLength * config.textDim * 4) != 0 {
            text.update(from: context, count: textLength * config.textDim)
            textCacheValid = false
        }

        for layer in 0..<config.layers {
            try cancellation.check()   // between two layers, never inside one (see `Cancellation`)
            for ahead in 1...2 where layer + ahead < config.layers {
                prefetcher.request(prefetcher.namesOfBlock(prefix: "transformer_blocks.\(layer + ahead)."))
            }
            if layer == config.layers - 1 {
                prefetcher.request(artifact.order.filter { $0.hasPrefix("norm_out.") || $0.hasPrefix("proj_out.") })
            }
            try block(layer, sequence: s, modulations: row + AnimaDiT.ModulationRow.sublayer(layer, 1, config))
            record("block\(layer)_out", x, s * d)
        }
        textCacheValid = true

        // ── norm_out: shift, scale — no gate — and temb[:2d] ─────────────────────────────────
        let final = UnsafePointer(row + AnimaDiT.ModulationRow.normOut(config))
        timed("norm") { AnimaOps.modulate(x, into: normed, shift: final, scale: final + d,
                                          scaleOne: scaleOne, rows: s, columns: d) }
        try linearGPU("proj_out.weight", a: normed, into: projected, m: s, k: d, n: outDim,
                      rowsA: imageTokens, rowsC: imageTokens)

        var out = [Float](repeating: 0, count: c * height * width)
        out.withUnsafeMutableBufferPointer {
            Ops.unpatchify(projected, into: $0.baseAddress!, latentHeight: height, latentWidth: width,
                           patch: p, channels: c)
        }
        record("model_out", out, out.count)
        return out
    }

    /// One block: three sublayers, three modulations.
    /// - Parameter modulations: the block's three `shift, scale, gate` rows, `3 × 3d` floats, from the table.
    private func block(_ layer: Int, sequence s: Int, modulations: UnsafePointer<Float>) throws {
        let d = config.dim, h = config.hidden, heads = config.heads, dh = config.headDim
        let prefix = "transformer_blocks.\(layer)."
        let tag = layer == 0 ? "block0_" : nil
        var modulation = modulations

        /// `shift, scale, gate` of sublayer `i` — `3d` floats of the table, in that order.
        func modulate(_ i: Int) {
            modulation = modulations + (i - 1) * 3 * d
            let m = modulation
            timed("norm") { AnimaOps.modulate(x, into: normed, shift: m, scale: m + d,
                                              scaleOne: scaleOne, rows: s, columns: d) }
        }
        var gate: UnsafePointer<Float> { modulation + 2 * d }

        // ── 1. self-attention ────────────────────────────────────────────────────────────────
        modulate(1)
        try linearGPU(prefix + "attn1.to_q.weight", a: normed, into: q, m: s, k: d, n: d)
        try linearGPU(prefix + "attn1.to_k.weight", a: normed, into: k, m: s, k: d, n: d)
        try linearGPU(prefix + "attn1.to_v.weight", a: normed, into: v, m: s, k: d, n: d)
        // QK-Norm then RoPE, one pass per row (`Ops.rmsNormRope`): the same operations in the same order
        // as the two passes, so the same bits (measured).
        for (name, tensor) in [("attn1.norm_q.weight", q), ("attn1.norm_k.weight", k)] {
            try artifact.materialize(prefix + name, into: reserve, capacity: reserveCapacity)
            timed("qk-norm + rope") { Ops.rmsNormRope(tensor, weight: reserve, freqs: freqs, sequence: s,
                                                      heads: heads, headDim: dh, eps: 1e-5) }
        }
        try attend(selfAttention(s), "sdpa self", q: q, k: k, v: v, keyRows: imageTokens)
        try linearGPU(prefix + "attn1.to_out.0.weight", a: merged, into: o, m: s, k: d, n: d)
        if let tag { record(tag + "attn1_out", o, s * d) }
        timed("residual") { Ops.gatedResidual(x, plus: o, gate: gate, rows: s, columns: d) }

        // ── 2. cross-attention to the text — no RoPE ────────────────────────────────────────
        modulate(2)
        try linearGPU(prefix + "attn2.to_q.weight", a: normed, into: q, m: s, k: d, n: d)
        let textK = textKCache + layer * textLength * d, textV = textVCache + layer * textLength * d
        if !textCacheValid {
            try linearGPU(prefix + "attn2.to_k.weight", a: text, into: textK, m: textLength, k: config.textDim, n: d,
                          rowsA: textLength, rowsC: textLength)
            try linearGPU(prefix + "attn2.to_v.weight", a: text, into: textV, m: textLength, k: config.textDim, n: d,
                          rowsA: textLength, rowsC: textLength)
            try headNorm(prefix + "attn2.norm_k.weight", textK, rows: textLength * heads)
        }
        try headNorm(prefix + "attn2.norm_q.weight", q, rows: s * heads)
        try attend(crossAttention(s), "sdpa cross", q: q, k: textK, v: textV, keyRows: textLength)
        try linearGPU(prefix + "attn2.to_out.0.weight", a: merged, into: o, m: s, k: d, n: d)
        if let tag { record(tag + "attn2_out", o, s * d) }
        timed("residual") { Ops.gatedResidual(x, plus: o, gate: gate, rows: s, columns: d) }

        // ── 3. GELU FFN ───────────────────────────────────────────────────────────────────────
        modulate(3)
        try linearGPU(prefix + "ff.net.0.proj.weight", a: normed, into: h1, m: s, k: d, n: h)
        let buffer = try gemm.wrap(UnsafeMutableRawPointer(h1), bytes: imageTokens * h * 4, name: "c")
        timed("gelu") { ownTimings["gelu GPU", default: 0] += gelu.run(buffer, count: s * h) }
        try linearGPU(prefix + "ff.net.2.weight", a: h1, into: o, m: s, k: h, n: d)
        if let tag { record(tag + "ff_out", o, s * d) }
        timed("residual") { Ops.gatedResidual(x, plus: o, gate: gate, rows: s, columns: d) }
    }

    // ── plumbing ────────────────────────────────────────────────────────────────────────

    /// QK-Norm: RMSNorm over `head_dim`, eps 1e-5 (diffusers' `Attention(qk_norm="rms_norm")`).
    private func headNorm(_ name: String, _ tensor: UnsafeMutablePointer<Float>, rows: Int) throws {
        try artifact.materialize(name, into: reserve, capacity: reserveCapacity)
        timed("qk-norm") { Ops.rmsNorm(tensor, weight: reserve, into: tensor, rows: rows,
                                    columns: config.headDim, eps: 1e-5) }
    }

    /// One attention path, chosen once per DiT: the matrix flash kernel (`flash`, self-attention only),
    /// the cut by heads (`HeadSplitAttention.applies`), or one MPSGraph SDPA — as `Block` chooses, so the
    /// capacity map's `flash` and `head_split` (`CapacityRules`) decide Anima's attention as Z-Image's.
    ///
    /// **The cut by heads, self and cross** (measured on the M1 Pro): 6 of the 16 heads on the CPU (`cpuFraction` 0.4, the
    /// shared one — 7 measured the same, 5 slower), the cross-attention's 512 text keys included. At 1024²
    /// the SDPA went 2.69 → 1.90 s an evaluation (self 2.37 → 1.67, cross 0.33 → 0.23), the render
    /// 52.6 → 45.7 s; at 512² 0.37 → 0.26 s, the render level. The bits change: `model_out` at 1.43·10⁻⁵ of
    /// the fp64 witness (MPSGraph alone 1.03, the fp32 oracle 1.47); flash, under the map, 1.67.
    private enum AttentionPath {
        case graph(Attention)
        case split(HeadSplitAttention)
        case flash(FlashMatrix)
    }
    private var selfPath: AttentionPath?
    private var crossPath: AttentionPath?

    private func selfAttention(_ s: Int) throws -> AttentionPath {
        if let p = selfPath { return p }
        let p: AttentionPath
        var flash: FlashMatrix?
        if HeadSplitAttention.allowsFlash(sequence: s, heads: config.heads, headDim: config.headDim) {
            do {
                flash = try FlashMatrix(device: gemm.device, queue: gemm.queue, heads: config.heads, sequence: s,
                                        headDim: config.headDim, rowsPerSimdgroup: 8, dimsPerSimdgroup: 64,
                                        keysPerTile: 16, blocksPerGroup: 8)
            } catch {
                Warnings.emit("⚠ SILICONED_FLASH: falling back at S=\(s) — \(error)")
            }
        }
        if let flash {
            p = .flash(flash)
        } else if HeadSplitAttention.applies(sequence: s) {
            p = .split(try HeadSplitAttention(device: gemm.device, queue: gemm.queue, heads: config.heads,
                                              sequence: s, headDim: config.headDim))
        } else {
            p = .graph(Attention(device: gemm.device, queue: gemm.queue, heads: config.heads,
                                 sequence: s, headDim: config.headDim))
        }
        selfPath = p
        return p
    }

    private func crossAttention(_ s: Int) throws -> AttentionPath {
        if let p = crossPath { return p }
        let p: AttentionPath
        if HeadSplitAttention.applies(sequence: s) {
            p = .split(try HeadSplitAttention(device: gemm.device, queue: gemm.queue, heads: config.heads,
                                              sequence: s, headDim: config.headDim, keys: textLength))
        } else {
            p = .graph(Attention(device: gemm.device, queue: gemm.queue, heads: config.heads,
                                 sequence: s, headDim: config.headDim, keySequence: textLength))
        }
        crossPath = p
        return p
    }

    private func attend(_ path: AttentionPath, _ phase: String, q: UnsafeMutablePointer<Float>,
                        k: UnsafeMutablePointer<Float>, v: UnsafeMutablePointer<Float>, keyRows: Int) throws {
        let d = config.dim
        let qb = try gemm.wrap(UnsafeMutableRawPointer(q), bytes: imageTokens * d * 4, name: "q")
        let kb = try gemm.wrap(UnsafeMutableRawPointer(k), bytes: keyRows * d * 4, name: "k")
        let vb = try gemm.wrap(UnsafeMutableRawPointer(v), bytes: keyRows * d * 4, name: "v")
        let ob = try gemm.wrap(UnsafeMutableRawPointer(merged), bytes: imageTokens * d * 4, name: "merged")
        timed("sdpa wall") {
            timed(phase) {
                switch path {
                case .graph(let a): ownTimings["sdpa GPU", default: 0] += a.run(q: qb, k: kb, v: vb, into: ob)
                case .flash(let f): ownTimings["sdpa GPU", default: 0] += f.run(q: qb, k: kb, v: vb, into: ob)
                case .split(let h):
                    let (gpu, cpu) = h.run(q: qb, k: kb, v: vb, into: ob, q: q, k: k, v: v, out: merged)
                    ownTimings["sdpa GPU", default: 0] += gpu
                    ownTimings["sdpa CPU", default: 0] += cpu
                }
            }
        }
    }

    /// `y = W·x`, no bias — no Cosmos `Linear` carries one — **accumulated in double**.
    ///
    /// ## Why not `Ops.gemv` — measured on 2026-09-22 against an fp64 reference
    ///
    /// `cblas_sgemv` is true fp32, but on a weight laid out `[input, output]` it accumulates
    /// like a naive loop: 1.5·10⁻⁶ relative error on `2048 × 2048`, **five times** that of
    /// the inverse layout. These `gemv`s carry `temb` and the 84 modulations, which multiply the
    /// whole residual: Anima amplifies the deviation up to `model_out`. Measured against the fp64
    /// evaluation of the oracle:
    ///
    ///                          `temb`     `block13_out`   `model_out` (medians)
    ///     `cblas_sgemv`       9.6·10⁻⁷      9.9·10⁻⁵        1.4·10⁻⁵     6× farther than the oracle
    ///     double              0            7.8·10⁻⁶        1.9·10⁻⁶     closer than the fp32 oracle
    ///     fp32 oracle         1.2·10⁻⁷      1.2·10⁻⁵        2.6·10⁻⁶
    ///
    /// Swapping the GEMM and the SDPA for their double computation moved nothing: it was here.
    /// The cost is nil on the clock (7.31 versus 7.35 s) — these products have `M = 1`.
    ///
    /// **`m` rows at once, one per σ of the table (`tabulate`)**: the weight is read and widened to double
    /// once for every σ of a render, instead of once per evaluation. `cblas_dgemm` at `m` rows does not sum
    /// in the order it does at one row — the doubles differ in their last bits — but each product is
    /// exact to ~10⁻¹⁶ either way and is narrowed to fp32 at once, so the floats come out the same
    /// (measured: rendered pixels identical to the one-row computation).
    private func gemv(_ name: String, x: UnsafePointer<Float>, into out: UnsafeMutablePointer<Float>,
                      rows m: Int, outputs: Int, inputs: Int,
                      scratch: (a: UnsafeMutablePointer<Double>, c: UnsafeMutablePointer<Double>)) throws {
        let got = try artifact.materialize(name, into: reserve, capacity: reserveCapacity)
        guard got == outputs * inputs else {
            throw Artifact.Failure.badHeader("\(name): \(got) values, \(outputs * inputs) expected")
        }
        guard inputs * outputs <= exactWeight else {
            throw Artifact.Failure.badHeader("\(name): \(inputs) × \(outputs) beyond the double scratch (`exactProducts`)")
        }
        timed("modulation") {
            AnimaExact.linear(a: x, w: reserve, into: out, m: m, k: inputs, n: outputs,
                              weightIsTransposed: transposed, scratch: (scratch.a, exactW, scratch.c))
        }
        // The modulation LoRA, in double like the weight: `y += (x·down)·up`, row by row.
        let target = String(name.dropLast(".weight".count))
        if let lora, let down = loraDown, let up = loraUp, lora.rank(target) > 0 {
            let r = try timed("lora weights") { try lora.materialize(target, k: inputs, n: outputs, down: down, up: up) }
            timed("lora modulation") {
                for i in 0..<m {
                    let x = x + i * inputs, out = out + i * outputs
                    var mid = [Double](repeating: 0, count: r)
                    for j in 0..<inputs {
                        let xj = Double(x[j]), row = down + j * r
                        for q in 0..<r { mid[q] += xj * Double(row[q]) }
                    }
                    for o in 0..<outputs {
                        var sum = 0.0
                        for q in 0..<r { sum += mid[q] * Double(up[q * outputs + o]) }
                        out[o] = Float(Double(out[o]) + sum)
                    }
                }
            }
        }
    }

    // ── the modulation table ────────────────────────────────────────────────────────────────

    /// **Every modulation of a render, computed before its first evaluation**.
    ///
    /// Anima's 84 sublayer modulations, `norm_out`'s and `temb` depend on σ alone — not on the latent,
    /// not on the text. Computed per evaluation, each of the 170 `gemv`s read its weight, widened it to
    /// fp32 then to double, for one row: 8 times a render, the same 350 MB of bf16 for 8 vectors. Here
    /// each weight is read once for all the render's σ (`tabulate(sigmas:)`, `m` rows); an evaluation
    /// whose σ is not in the table (the checks, which call `forward` alone) tabulates its own, one row —
    /// one path either way. Same bits (`gemv`).
    private var table: (arena: Arena, index: [UInt32: Int], rows: UnsafeMutablePointer<Float>)?

    /// Where each vector sits in a σ's row of the table, in floats.
    package enum ModulationRow {
        static func temb(_ c: Config) -> Int { 0 }
        static func embedded(_ c: Config) -> Int { 3 * c.dim }
        /// `shift, scale, gate` of sublayer `i` (1…3) of `layer`, `3d` floats.
        static func sublayer(_ layer: Int, _ i: Int, _ c: Config) -> Int { 4 * c.dim + (layer * 3 + i - 1) * 3 * c.dim }
        /// `norm_out`'s `shift, scale`, `2d` floats.
        static func normOut(_ c: Config) -> Int { 4 * c.dim + c.layers * 9 * c.dim }
        /// One σ's row: `temb` 3d, `embedded` d, the sublayers', `norm_out`'s.
        package static func floats(_ c: Config) -> Int { normOut(c) + 2 * c.dim }
    }

    /// **The table's slices, bytes, in the order `tabulate` reserves them**, for `count` σ: the rows, then
    /// the computation's scratch — the sinusoids and the hidden `[count, d]`, SiLU's scratch, the
    /// low-rank middle `[count, 256]`, `AnimaExact`'s double input and output at `count` rows.
    package static func tableSlices(config: Config, count: Int) -> [Int] {
        let d = config.dim, f = MemoryLayout<Float>.size, w = MemoryLayout<Double>.size
        let exact = exactProducts(config: config)
        return [count * ModulationRow.floats(config) * f, count * d * f, count * d * f, count * 3 * d * f,
                count * config.adalnLoRA * f, count * exact.map(\.k).max()! * w, count * exact.map(\.n).max()! * w]
    }

    /// **The table's arena for `count` σ** — what `tabulate` maps, to the byte; the memory need charges it
    /// at the render's evaluations (`MemoryNeed.anima`).
    package static func tableBytes(config: Config, count: Int) -> Int {
        Arena.capacity(slices: tableSlices(config: config, count: count))
    }

    /// **The σ a table holds at most**: the official recipe's 8 evaluations. A render of more steps
    /// tabulates them 8 at a time (`schedule`), so the table's size never follows the request — the
    /// memory need charges these rows, whatever the steps.
    package static let tableRows = 8
    /// The render's σ, in order, as `tabulate(sigmas:)` received them: a σ missing from the table is
    /// looked up here, and the next `tableRows` from it are tabulated.
    private var schedule: [Float] = []

    /// **The render's σ**: the first `tableRows` distinct ones are tabulated now, the rest when reached.
    package func tabulate(sigmas: [Float]) throws {
        schedule = sigmas
        try tabulate(from: 0)
    }

    /// `σ`'s row, the table refilled from `σ` on if it does not hold it; a σ outside the schedule (the checks,
    /// which call `forward` alone) is tabulated alone.
    private func ensureTabulated(_ sigma: Float) throws {
        guard table?.index[sigma.bitPattern] == nil else { return }
        if let i = schedule.firstIndex(where: { $0.bitPattern == sigma.bitPattern }) {
            try tabulate(from: i)
        } else {
            schedule = [sigma]
            try tabulate(from: 0)
        }
    }

    /// Computes the modulations of the `tableRows` distinct σ of `schedule` from `first`, replacing the table.
    private func tabulate(from first: Int) throws {
        var unique: [Float] = []
        for s in schedule[first...] where !unique.contains(where: { $0.bitPattern == s.bitPattern }) {
            guard unique.count < AnimaDiT.tableRows else { break }
            unique.append(s)
        }
        let m = max(1, unique.count), d = config.dim, a = config.adalnLoRA
        table = nil   // the previous arena is unmapped before the new one is mapped
        let sizes = AnimaDiT.tableSlices(config: config, count: m)
        let arena = try Arena(capacity: Arena.capacity(slices: sizes))
        func floats(_ name: String, _ i: Int) throws -> UnsafeMutablePointer<Float> {
            try arena.reserve(name, bytes: sizes[i]).assumingMemoryBound(to: Float.self)
        }
        let rows = try floats("rows", 0), sinusoid = try floats("sinusoid", 1), hidden = try floats("hidden", 2)
        let siluScratch = try floats("silu", 3), mid = try floats("mid", 4)
        let exactA = try arena.reserve("exactA", bytes: sizes[5]).assumingMemoryBound(to: Double.self)
        let exactC = try arena.reserve("exactC", bytes: sizes[6]).assumingMemoryBound(to: Double.self)
        let scratch = (a: exactA, c: exactC), stride = ModulationRow.floats(config)
        /// The `n` floats of `slot` for each σ, copied into each σ's row at `offset`.
        func scatter(_ slot: UnsafePointer<Float>, _ n: Int, at offset: Int) {
            for i in 0..<m { (rows + i * stride + offset).update(from: slot + i * n, count: n) }
        }

        // temb [3d] and embedded [d]. The pipeline tells the DiT `t / 1000` with `t = σ · 1000`, two float32
        // roundings (`AnimaLoopBeforeDenoiser`): σ back to the bit at 8 steps, not always — at 19 steps
        // σ₅ and σ₁₆, 13 σ of N = 1…50 move by one bit (18/10).
        for i in 0..<m { Ops.timestepEmbedding((unique[i] * 1000) / 1000, into: sinusoid + i * d, dim: d) }
        try gemv("time_embed.t_embedder.linear_1.weight", x: sinusoid, into: hidden, rows: m, outputs: d, inputs: d,
                 scratch: scratch)
        Ops.siluInPlace(hidden, count: m * d, scratch: siluScratch)
        // temb lands in `siluScratch` ([m, 3d]), then into the rows.
        try gemv("time_embed.t_embedder.linear_2.weight", x: hidden, into: siluScratch, rows: m, outputs: 3 * d,
                 inputs: d, scratch: scratch)
        scatter(siluScratch, 3 * d, at: ModulationRow.temb(config))
        try artifact.materialize("time_embed.norm.weight", into: reserve, capacity: reserveCapacity)
        Ops.rmsNorm(sinusoid, weight: reserve, into: hidden, rows: m, columns: d, eps: 1e-6)
        scatter(hidden, d, at: ModulationRow.embedded(config))
        // SiLU(embedded) is the input of ALL the modulations: `sinusoid` holds it from here on.
        sinusoid.update(from: hidden, count: m * d)
        Ops.siluInPlace(sinusoid, count: m * d, scratch: hidden)

        /// One low-rank modulation `d → 256 → outputs`, plus `temb[:outputs]`, into each σ's row at `offset`.
        func modulation(_ prefix: String, outputs: Int, at offset: Int) throws {
            try gemv(prefix + "linear_1.weight", x: sinusoid, into: mid, rows: m, outputs: a, inputs: d, scratch: scratch)
            try gemv(prefix + "linear_2.weight", x: mid, into: siluScratch, rows: m, outputs: outputs, inputs: a,
                     scratch: scratch)
            for i in 0..<m {
                let row = rows + i * stride
                vDSP_vadd(siluScratch + i * outputs, 1, row + ModulationRow.temb(config), 1, row + offset, 1,
                          vDSP_Length(outputs))
            }
        }
        for layer in 0..<config.layers {
            for sub in 1...3 {
                try modulation("transformer_blocks.\(layer).norm\(sub).", outputs: 3 * d,
                               at: ModulationRow.sublayer(layer, sub, config))
            }
        }
        try modulation("norm_out.", outputs: 2 * d, at: ModulationRow.normOut(config))
        table = (arena, Dictionary(uniqueKeysWithValues: unique.enumerated().map { ($1.bitPattern, $0) }), rows)
    }

    /// σ's row of the table.
    private func modulationRow(_ sigma: Float) throws -> UnsafeMutablePointer<Float> {
        guard let table, let i = table.index[sigma.bitPattern] else {
            throw Artifact.Failure.misuse("Anima: σ \(sigma) not tabulated")
        }
        return table.rows + i * ModulationRow.floats(config)
    }

    /// `C[m, n] = A[m, k] · W`, on the GPU (and the AMX if the driver cuts out).
    ///
    /// - Parameters rowsA, rowsC: the **reserved** rows of the two slices. Metal wrappers
    ///   are memoized by address and at their maximum size; so we always wrap the whole slice,
    ///   never the part a call uses.
    private func linearGPU(_ name: String, a: UnsafeMutablePointer<Float>, into c: UnsafeMutablePointer<Float>,
                           m: Int, k: Int, n: Int, rowsA: Int? = nil, rowsC: Int? = nil) throws {
        let ab = try gemm.wrap(UnsafeMutableRawPointer(a), bytes: (rowsA ?? imageTokens) * k * 4, name: "a")
        let cb = try gemm.wrap(UnsafeMutableRawPointer(c), bytes: (rowsC ?? imageTokens) * n * 4, name: "c")
        try products.linear(name, a: ab, c: cb, m: m, k: k, n: n)
    }
}

/// Cosmos 3D RoPE, **in the interleaved layout** that the forge made possible.
///
/// `CosmosRotaryPosEmbed` splits `head_dim = 128` into `t = 44`, `h = 42`, `w = 42` dimensions, i.e.
/// 22 + 21 + 21 = 64 frequencies, and builds `cat([t, h, w] × 2)`: the split pair `(p, p+64)`
/// rotates by the angle of index `p`. After the forge's permutation, this pair is `(2p, 2p+1)`,
/// and the table is simply `[S, 64, (cos, sin)]` with the reference's angle `p`.
///
/// The bases are **stretched by NTK**: `θ_h = 10000 · 4^(42/40)`, likewise for `w`, and `θ_t = 10000`
/// (`rope_scale = [1, 4, 4]`). An image has only one frame: the `t` axis is zero everywhere, so its
/// 22 pairs do not rotate. The reference's table stops at `Config.ropePositions` per side; the
/// formula here would go on, so the callers bound it (`AnimaDiT.init`, `Format.maximumSide(for:)`).
///
/// **The reference's arithmetic, rounding for rounding** (`CosmosRotaryPosEmbed.forward`): θ a Python
/// double, which `θ ** dim_range` casts to **float32** (a scalar against a float32 tensor), `1.0 / …` in
/// float32, `torch.outer(seq, freqs)` with `seq` a float32 `arange` — one float32 product —, then
/// `torch.cos`/`torch.sin` (SLEEF in torch: a one-bit residue, `Ops.polar`). Until 18/10 the table was
/// in double, rounded once — the witness's function rather than the reference's.
package enum AnimaRope {
    package static func write(into out: UnsafeMutablePointer<Float>, tilesHigh: Int, tilesWide: Int,
                             headDim: Int, scale: [Double]) {
        let dimH = headDim / 6 * 2, dimW = dimH, dimT = headDim - dimH - dimW
        func frequencies(_ dims: Int, _ factor: Double) -> [Float] {
            let theta = Float(10000.0 * pow(factor, Double(dims) / Double(dims - 2)))
            return (0..<(dims / 2)).map { 1 / powf(theta, Float(2 * $0) / Float(dims)) }
        }
        let ft = frequencies(dimT, scale[0]), fh = frequencies(dimH, scale[1]), fw = frequencies(dimW, scale[2])
        let pairs = headDim / 2
        precondition(ft.count + fh.count + fw.count == pairs)
        for row in 0..<tilesHigh {
            for column in 0..<tilesWide {
                let token = out + (row * tilesWide + column) * pairs * 2
                var index = 0
                for _ in ft { token[2 * index] = 1; token[2 * index + 1] = 0; index += 1 }
                for f in fh {
                    let angle = Float(row) * f
                    token[2 * index] = cosf(angle); token[2 * index + 1] = sinf(angle); index += 1
                }
                for f in fw {
                    let angle = Float(column) * f
                    token[2 * index] = cosf(angle); token[2 * index + 1] = sinf(angle); index += 1
                }
            }
        }
    }
}

/// The elementwise operations specific to Anima.
package enum AnimaOps {
    /// `[C, H, W]` + a zero mask channel → `[(H/p)·(W/p), (C+1)·p·p]`, the **slowest
    /// channel**: `CosmosPatchEmbed` does `permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7)`, so
    /// the index within the token is `(c·p + ph)·p + pw`. It is the inverse of Z-Image.
    package static func patchify(_ latent: UnsafePointer<Float>, into out: UnsafeMutablePointer<Float>,
                                channels: Int, height: Int, width: Int, patch: Int) {
        let tilesHigh = height / patch, tilesWide = width / patch
        let perToken = (channels + 1) * patch * patch
        for th in 0..<tilesHigh {
            for tw in 0..<tilesWide {
                let token = out + (th * tilesWide + tw) * perToken
                for ch in 0...channels {
                    for ph in 0..<patch {
                        for pw in 0..<patch {
                            token[(ch * patch + ph) * patch + pw] = ch == channels ? 0
                                : latent[(ch * height + th * patch + ph) * width + tw * patch + pw]
                        }
                    }
                }
            }
        }
    }

    /// `out = LayerNorm(x) · (1 + scale) + shift` — LayerNorm without affine, eps 1e-6.
    ///
    /// **One pass, each row while it is in cache**: `Ops.layerNorm`'s operations, then the `vma`, on the
    /// same row — the same calls in the same order as the two passes it replaces, so the same bits,
    /// without reading `[s, d]` back from memory a second time.
    package static func modulate(_ x: UnsafePointer<Float>, into out: UnsafeMutablePointer<Float>,
                                shift: UnsafePointer<Float>, scale: UnsafePointer<Float>,
                                scaleOne: UnsafeMutablePointer<Float>, rows: Int, columns: Int) {
        var one: Float = 1
        vDSP_vsadd(scale, 1, &one, scaleOne, 1, vDSP_Length(columns))
        let n = vDSP_Length(columns)
        let factor = UnsafePointer(scaleOne)
        Parallel.rows(rows, width: columns) { first, howMany in
            for row in first..<(first + howMany) {
                let source = x + row * columns, r = out + row * columns
                var mean: Float = 0
                vDSP_meanv(source, 1, &mean, n)
                var negativeMean = -mean
                vDSP_vsadd(source, 1, &negativeMean, r, 1, n)
                var sumOfSquares: Float = 0
                vDSP_svesq(r, 1, &sumOfSquares, n)
                var inverse = 1 / (sumOfSquares / Float(columns) + 1e-6).squareRoot()
                vDSP_vsmul(r, 1, &inverse, r, 1, n)
                Ops.multiplyAdd(r, times: factor, plus: shift, count: columns)   // `* (1 + scale) + shift`: two roundings
            }
        }
    }

    /// **Exact** GELU: `x · Φ(x) = ½ x (1 + erf(x / √2))` — `nn.GELU()` without approximation.
    package static func gelu(_ x: UnsafeMutablePointer<Float>, count: Int) {
        let chunk = 1 << 16
        let pieces = (count + chunk - 1) / chunk
        nonisolated(unsafe) let x = x   // disjoint slices, one per thread
        DispatchQueue.concurrentPerform(iterations: pieces) { piece in
            let start = piece * chunk, end = min(start + chunk, count)
            for i in start..<end {
                let value = x[i]
                x[i] = 0.5 * value * (1 + erff(value * 0.70710678118654752))
            }
        }
    }
}

/// The double computation of the modulation `gemv` (see `AnimaDiT.gemv`).
enum AnimaExact {
    /// `C[m, n] = A[m, k] · W`, W laid out `[k, n]` if the forge transposed it, `[n, k]` otherwise —
    /// widened to double in the caller's `scratch` (`a` holds `m·k`, `w` `k·n`, `c` `m·n`: the DiT's
    /// arena slices `exactA`, `exactW`, `exactC`), multiplied, narrowed back. The same `vDSP`
    /// widenings and the same `cblas_dgemm` call as when the three were fresh arrays; `c` is cleared
    /// first, as a fresh array was (with β = 0 it is not read).
    static func linear(a: UnsafePointer<Float>, w: UnsafePointer<Float>, into c: UnsafeMutablePointer<Float>,
                       m: Int, k: Int, n: Int, weightIsTransposed: Bool,
                       scratch: (a: UnsafeMutablePointer<Double>, w: UnsafeMutablePointer<Double>,
                                 c: UnsafeMutablePointer<Double>)) {
        vDSP_vspdp(a, 1, scratch.a, 1, vDSP_Length(m * k))
        vDSP_vspdp(w, 1, scratch.w, 1, vDSP_Length(k * n))
        vDSP_vclrD(scratch.c, 1, vDSP_Length(m * n))
        cblas_dgemm(CblasRowMajor, CblasNoTrans, weightIsTransposed ? CblasNoTrans : CblasTrans,
                    Int32(m), Int32(n), Int32(k), 1, scratch.a, Int32(k), scratch.w,
                    Int32(weightIsTransposed ? n : k), 0, scratch.c, Int32(n))
        vDSP_vdpsp(scratch.c, 1, c, 1, vDSP_Length(m * n))
    }
}

// ── sizing: the shapes the constructor allocates, and the memory need charges ──────────────────

extension AnimaDiT {
    /// The image tokens of a latent grid: one per `patch × patch` cell — every evaluation's (the arena
    /// is sized once, for its DiT's grid; the half-size start's DiT is another, released before the full one).
    package static func imageTokens(latentHeight: Int, latentWidth: Int, config: Config) -> Int {
        (latentHeight / config.patch) * (latentWidth / config.patch)
    }

    /// **The `gemv`s computed in double, `(k inputs, n outputs)`** — `temb` (`t_embedder.linear_1`,
    /// then `linear_2` to `3d`), each sublayer's low-rank modulation (`d → 256 → 3d`) and `norm_out`'s
    /// (`256 → 2d`): what `AnimaExact`'s scratch must hold, the largest `d × 3d` (100 MB of doubles).
    package static func exactProducts(config: Config) -> [(k: Int, n: Int)] {
        let d = config.dim, a = config.adalnLoRA
        return [(d, d), (d, 3 * d), (d, a), (a, 3 * d), (a, 2 * d)]
    }

    /// **The widest input and output a LoRA can bring to this DiT** — `LoRA.maxEntry` and `maxOutput`
    /// are a module's `k` and `n`; the DiT's widest are the FFN's (`ff.net.2` reads `h`, `ff.net.0`
    /// writes `h`), the cross-attention's text (`textDim`) or the modulation's `3d`: what the memory
    /// need charges a stack whose modules it does not see (`MemoryNeed.anima`).
    package static func loraBounds(config: Config) -> (entry: Int, output: Int) {
        (max(config.hidden, config.dim, config.textDim), max(config.hidden, 3 * config.dim))
    }

    /// **The DiT's slices, in bytes, in the order `init` reserves them** — each written by an
    /// evaluation at the sizing grid (`s` image tokens) or by the text it caches:
    ///   - seven `[s, d]` (`x`, `normed`, `q`, `k`, `v`, `merged`, `o`), the FFN's `[s, h]`, the
    ///     patches `[s, (C+1)p²]` (the mask channel), the output `[s, Cp²]`, the RoPE table `[s, Dh]`;
    ///   - `reserve`, one widened weight, `d·h` fp32 **whatever the map's type** — bf16, Q8_0, Q4_K: the
    ///     map is read through the page cache and widened here, never staged (`materialize`);
    ///   - the text `[T, 1024]` and its k/v for **every layer** (`textK`, `textV`, `[layers, T, d]`,
    ///     235 MB at T = 512): computed once per context;
    ///   - `scaleOne` (2d), and `AnimaExact`'s double copy of the widest modulation weight (`exactProducts`);
    ///     the modulations themselves live in their own table (`tableBytes`), sized by the render's σ;
    ///   - under a stack of cumulative rank `r`: its `down [max(k, d), r]`, `up [r, max(n, d)]`, and the
    ///     `mid [max(s, T), r]` the GPU's two products share.
    /// The former count charged `4d` for the `d` of `embedded`, and added 64 pages and 8 MB of margin.
    package static func slices(config: Config, imageTokens s: Int, textLength t: Int,
                               lora: (rank: Int, entry: Int, output: Int)?) -> [(name: String, bytes: Int)] {
        let d = config.dim, h = config.hidden, f = MemoryLayout<Float>.size, w = MemoryLayout<Double>.size
        let inDim = (config.channels + 1) * config.patch * config.patch
        let outDim = config.channels * config.patch * config.patch
        let exact = exactProducts(config: config)
        var slices: [(name: String, bytes: Int)] = [
            ("x", s * d * f), ("normed", s * d * f), ("q", s * d * f), ("k", s * d * f), ("v", s * d * f),
            ("merged", s * d * f), ("o", s * d * f), ("reserve", d * h * f),
            ("tokens", s * inDim * f), ("projected", s * outDim * f), ("freqs", s * config.headDim * f),
            ("text", t * config.textDim * f), ("textK", config.layers * t * d * f), ("textV", config.layers * t * d * f),
            ("scaleOne", 2 * d * f), ("exactW", exact.map { $0.k * $0.n }.max()! * w),
        ]
        if let lora, lora.rank > 0 {
            slices += [("loraBas", max(lora.entry, d) * lora.rank * f), ("loraHaut", lora.rank * max(lora.output, d) * f),
                       ("loraMid", max(s, t) * lora.rank * f)]
        }
        return slices
    }

    /// **The DiT's arena**: its slices, each on its page (`Arena.capacity(slices:)`) — `arenaCapacity`
    /// of the DiT built on these shapes, to the byte.
    package static func arenaBytes(config: Config, latentHeight: Int, latentWidth: Int, textLength: Int,
                                   lora: (rank: Int, entry: Int, output: Int)?) -> Int {
        Arena.capacity(slices: slices(config: config,
                                      imageTokens: imageTokens(latentHeight: latentHeight, latentWidth: latentWidth,
                                                               config: config),
                                      textLength: textLength, lora: lora).map(\.bytes))
    }
}
