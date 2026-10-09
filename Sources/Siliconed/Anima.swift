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
///   - the **residuals climb to 2.5·10⁵** at block 13 (golden tensors): four times the fp16
///     ceiling. Anima requires fp32 as much as Z-Image does.
package final class AnimaDiT {
    package struct Config {
        package let dim, heads, headDim, hidden, layers, channels, patch, adalnLoRA, textDim: Int
        package let ropeScale: [Double]

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
        }
    }

    package let config: Config
    package private(set) var boundaries: [String: [Float]] = [:]
    package var recordBoundaries = false
    /// The token of the render in progress, consulted between two layers (see `Cancellation`).
    package var cancellation: Cancellation?
    package private(set) var timings: [String: Double] = [:]
    package func resetTimings() { timings.removeAll() }
    package var conductor: Conductor? { gemm.conductor }

    private let artifact: Artifact
    private let prefetcher: Prefetcher
    private let gemm: GEMM
    private let arena: Arena
    private let transposed: Bool
    /// The sizing grid, which is that of every evaluation (no spectral schedule here).
    private let latentHeight, latentWidth, imageTokens, textLength: Int
    /// **`SILICONED_ANIMA_EXACT=gemm,sdpa` — a diagnostic instrument, never a render path.**
    /// Replaces the GEMM and/or the SDPA with their DOUBLE computation on the CPU, to attribute a
    /// deviation from fp64 to one or the other. Slow (several minutes per evaluation), and announced.
    private static let exact: Set<String> = Set(
        (ProcessInfo.processInfo.environment["SILICONED_ANIMA_EXACT"] ?? "")
            .split(separator: ",").map(String.init))
    /// **The LoRA stack** (`ForgeLoRA`, Anima's names and RoPE): `ΔW = B·A` per module, the
    /// GEMMs through `gemm.lora` (two thin products accumulated into the output), the modulation
    /// `gemv`s in double like their main weight. The cached text k/v carry it too:
    /// they are computed once, LoRA included.
    package let lora: LoRA?
    private let loraDown, loraUp, loraMid: UnsafeMutablePointer<Float>?
    private var loraBuffers: (down: MTLBuffer, up: MTLBuffer, mid: MTLBuffer)?
    /// The GELU on the GPU. `SILICONED_ANIMA_GELU_CPU=1` restores the old path — the reference.
    private let gelu: AnimaGELU?
    private var selfAttentions: [Int: Attention] = [:]
    private var crossAttentions: [Int: Attention] = [:]

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
    private let temb, embedded, modulation, scaleOne: UnsafeMutablePointer<Float>

    /// - Parameter textLength: the length of the text context — 512 for Anima, the conditioner
    ///   padding its output with zeros up to that. The zeros **count**: the reference does not
    ///   mask cross-attention, so each zero row takes its share of the softmax.
    package init(artifact: Artifact, latentHeight: Int, latentWidth: Int, textLength: Int = 512,
                freezeCut: Bool = EngineSettings.effective.frozenCut, lora: LoRA? = nil) throws {
        guard artifact.header["kind"] as? String == "anima-turbo-dit" else {
            throw Artifact.Failure.badHeader("map \(artifact.header["kind"] ?? "?"): anima-turbo-dit expected")
        }
        guard artifact.header["rope_layout"] as? String == "interleaved" else {
            throw Artifact.Failure.badHeader("RoPE not converted by the forge — `Ops.rope` would be wrong")
        }
        self.artifact = artifact
        self.config = Config(header: artifact.header)
        self.prefetcher = Prefetcher(artifact: artifact)
        self.gemm = try GEMM(freezeCut: freezeCut)
        self.gelu = ProcessInfo.processInfo.environment["SILICONED_ANIMA_GELU_CPU"] == "1"
            ? nil : try AnimaGELU(device: gemm.device, queue: gemm.queue)
        self.transposed = artifact.linearWeightsTransposed
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
        self.textLength = textLength
        imageTokens = (latentHeight / config.patch) * (latentWidth / config.patch)

        let s = imageTokens, d = config.dim, h = config.hidden
        let inDim = (config.channels + 1) * config.patch * config.patch
        let outDim = config.channels * config.patch * config.patch
        let r = lora?.maxRank ?? 0
        let loraFloats = r > 0 ? max(lora!.maxEntry, d) * r + r * max(lora!.maxOutput, d)
                                 + max(s, textLength) * r : 0
        let floats = loraFloats + 7 * s * d + s * h + d * h + s * inDim + s * outDim + s * config.headDim
            + textLength * config.textDim + 2 * config.layers * textLength * d + 3 * d + 4 * d + 3 * d + 2 * d
        // LOCAL binding: a nested function cannot capture `self` before the end of
        // initialization (same constraint as in `Attention`).
        let arena = try Arena(capacity: floats * 4 + 64 * Arena.alignment + (8 << 20))
        self.arena = arena
        func slot(_ name: String, _ count: Int) throws -> UnsafeMutablePointer<Float> {
            try arena.reserve(name, bytes: count * 4).assumingMemoryBound(to: Float.self)
        }
        x = try slot("x", s * d); normed = try slot("normed", s * d)
        q = try slot("q", s * d); k = try slot("k", s * d); v = try slot("v", s * d)
        merged = try slot("merged", s * d); o = try slot("o", s * d)
        h1 = try slot("h1", s * h); reserve = try slot("reserve", d * h)
        tokens = try slot("tokens", s * inDim); projected = try slot("projected", s * outDim)
        freqs = try slot("freqs", s * config.headDim)
        text = try slot("text", textLength * config.textDim)
        textKCache = try slot("textK", config.layers * textLength * d)
        textVCache = try slot("textV", config.layers * textLength * d)
        temb = try slot("temb", 3 * d); embedded = try slot("embedded", d)
        modulation = try slot("modulation", 3 * d); scaleOne = try slot("scaleOne", 2 * d)
        self.lora = r > 0 ? lora : nil
        if let lora, r > 0 {
            loraDown = try slot("loraBas", max(lora.maxEntry, d) * r)
            loraUp = try slot("loraHaut", r * max(lora.maxOutput, d))
            loraMid = try slot("loraMid", max(s, textLength) * r)
        } else {
            loraDown = nil; loraUp = nil; loraMid = nil
        }
    }

    private func record(_ name: String, _ p: UnsafePointer<Float>, _ n: Int) {
        guard recordBoundaries else { return }
        boundaries[name] = Array(UnsafeBufferPointer(start: p, count: n))
    }

    private func timed<T>(_ phase: String, _ body: () throws -> T) rethrows -> T {
        let started = DispatchTime.now().uptimeNanoseconds
        defer { timings[phase, default: 0] += Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9 }
        return try body()
    }

    /// `latent` as `[16, H, W]`, `context` as `[textLength, 1024]` (the conditioner output,
    /// padded), `sigma` as the scheduler gives it — the pipeline passes `t / 1000`, which
    /// **is** σ. Returns the velocity as `[16, H, W]`.
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

        // ── the timestep: temb [3d] and embedded [d] ───────────────────────────────────────────
        var sinusoid = [Float](repeating: 0, count: d)
        sinusoid.withUnsafeMutableBufferPointer {
            Ops.timestepEmbedding(sigma, into: $0.baseAddress!, dim: d)
        }
        try sinusoid.withUnsafeBufferPointer { proj in
            try gemv("time_embed.t_embedder.linear_1.weight", x: proj.baseAddress!, into: normed, outputs: d, inputs: d)
            Ops.siluInPlace(normed, count: d, scratch: merged)
            try gemv("time_embed.t_embedder.linear_2.weight", x: normed, into: temb, outputs: 3 * d, inputs: d)
            try artifact.materialize("time_embed.norm.weight", into: reserve, capacity: reserveCapacity)
            Ops.rmsNorm(proj.baseAddress!, weight: reserve, into: embedded, rows: 1, columns: d, eps: 1e-6)
        }
        record("temb", temb, 3 * d)
        record("embedded_timestep", embedded, d)
        // SiLU(embedded) is the input of ALL the modulations: we compute it once.
        let siluEmbedded = [Float](unsafeUninitializedCapacity: d) { buffer, n in
            buffer.baseAddress!.update(from: embedded, count: d)
            Ops.siluInPlace(buffer.baseAddress!, count: d, scratch: merged)
            n = d
        }

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
            try block(layer, sequence: s, silu: siluEmbedded)
            record("block\(layer)_out", x, s * d)
        }
        textCacheValid = true

        // ── norm_out: shift, scale — no gate — and temb[:2d] ─────────────────────────────────
        try siluEmbedded.withUnsafeBufferPointer { e in
            try gemv("norm_out.linear_1.weight", x: e.baseAddress!, into: merged,
                     outputs: config.adalnLoRA, inputs: d)
            try gemv("norm_out.linear_2.weight", x: merged, into: modulation,
                     outputs: 2 * d, inputs: config.adalnLoRA)
        }
        vDSP_vadd(modulation, 1, temb, 1, modulation, 1, vDSP_Length(2 * d))
        timed("norm") { AnimaOps.modulate(x, into: normed, shift: modulation, scale: modulation + d,
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
    private func block(_ layer: Int, sequence s: Int, silu: [Float]) throws {
        let d = config.dim, h = config.hidden, heads = config.heads, dh = config.headDim
        let prefix = "transformer_blocks.\(layer)."
        let tag = layer == 0 ? "block0_" : nil

        /// `shift, scale, gate` of sublayer `i` — in `modulation`, in that order.
        func modulate(_ i: Int) throws {
            try silu.withUnsafeBufferPointer { e in
                try gemv(prefix + "norm\(i).linear_1.weight", x: e.baseAddress!, into: merged,
                         outputs: config.adalnLoRA, inputs: d)
                try gemv(prefix + "norm\(i).linear_2.weight", x: merged, into: modulation,
                         outputs: 3 * d, inputs: config.adalnLoRA)
            }
            vDSP_vadd(modulation, 1, temb, 1, modulation, 1, vDSP_Length(3 * d))
            timed("norm") { AnimaOps.modulate(x, into: normed, shift: modulation, scale: modulation + d,
                                              scaleOne: scaleOne, rows: s, columns: d) }
        }
        var gate: UnsafePointer<Float> { UnsafePointer(modulation + 2 * d) }

        // ── 1. self-attention ────────────────────────────────────────────────────────────────
        try modulate(1)
        try linearGPU(prefix + "attn1.to_q.weight", a: normed, into: q, m: s, k: d, n: d)
        try linearGPU(prefix + "attn1.to_k.weight", a: normed, into: k, m: s, k: d, n: d)
        try linearGPU(prefix + "attn1.to_v.weight", a: normed, into: v, m: s, k: d, n: d)
        try headNorm(prefix + "attn1.norm_q.weight", q, rows: s * heads)
        try headNorm(prefix + "attn1.norm_k.weight", k, rows: s * heads)
        timed("rope") {
            Ops.rope(q, freqs: freqs, sequence: s, heads: heads, headDim: dh)
            Ops.rope(k, freqs: freqs, sequence: s, heads: heads, headDim: dh)
        }
        try attend(selfAttention(s), q: q, k: k, v: v, keyRows: imageTokens)
        try linearGPU(prefix + "attn1.to_out.0.weight", a: merged, into: o, m: s, k: d, n: d)
        if let tag { record(tag + "attn1_out", o, s * d) }
        timed("residual") { Ops.gatedResidual(x, plus: o, gate: gate, rows: s, columns: d) }

        // ── 2. cross-attention to the text — no RoPE ────────────────────────────────────────
        try modulate(2)
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
        try attend(crossAttention(s), q: q, k: textK, v: textV, keyRows: textLength)
        try linearGPU(prefix + "attn2.to_out.0.weight", a: merged, into: o, m: s, k: d, n: d)
        if let tag { record(tag + "attn2_out", o, s * d) }
        timed("residual") { Ops.gatedResidual(x, plus: o, gate: gate, rows: s, columns: d) }

        // ── 3. GELU FFN ───────────────────────────────────────────────────────────────────────
        try modulate(3)
        try linearGPU(prefix + "ff.net.0.proj.weight", a: normed, into: h1, m: s, k: d, n: h)
        if let gelu {
            let buffer = try gemm.wrap(UnsafeMutableRawPointer(h1), bytes: imageTokens * h * 4, name: "c")
            timed("gelu") { timings["gelu GPU", default: 0] += gelu.run(buffer, count: s * h) }
        } else {
            timed("gelu") { AnimaOps.gelu(h1, count: s * h) }
        }
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

    private func selfAttention(_ s: Int) -> Attention {
        if let a = selfAttentions[s] { return a }
        let a = Attention(device: gemm.device, queue: gemm.queue, heads: config.heads,
                          sequence: s, headDim: config.headDim)
        selfAttentions[s] = a
        return a
    }

    private func crossAttention(_ s: Int) -> Attention {
        if let a = crossAttentions[s] { return a }
        let a = Attention(device: gemm.device, queue: gemm.queue, heads: config.heads,
                          sequence: s, headDim: config.headDim, keySequence: textLength)
        crossAttentions[s] = a
        return a
    }

    private func attend(_ attention: Attention, q: UnsafeMutablePointer<Float>, k: UnsafeMutablePointer<Float>,
                        v: UnsafeMutablePointer<Float>, keyRows: Int) throws {
        let d = config.dim
        if AnimaDiT.exact.contains("sdpa") {
            timed("sdpa exact") {
                AnimaExact.attention(q: q, k: k, v: v, into: merged, queries: attention.sequence,
                                     keys: attention.keySequence, heads: config.heads, headDim: config.headDim)
            }
            return
        }
        let qb = try gemm.wrap(UnsafeMutableRawPointer(q), bytes: imageTokens * d * 4, name: "q")
        let kb = try gemm.wrap(UnsafeMutableRawPointer(k), bytes: keyRows * d * 4, name: "k")
        let vb = try gemm.wrap(UnsafeMutableRawPointer(v), bytes: keyRows * d * 4, name: "v")
        let ob = try gemm.wrap(UnsafeMutableRawPointer(merged), bytes: imageTokens * d * 4, name: "merged")
        timed("sdpa wall") { timings["sdpa GPU", default: 0] += attention.run(q: qb, k: kb, v: vb, into: ob) }
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
    private func gemv(_ name: String, x: UnsafePointer<Float>, into out: UnsafeMutablePointer<Float>,
                      outputs: Int, inputs: Int) throws {
        let got = try artifact.materialize(name, into: reserve, capacity: reserveCapacity)
        guard got == outputs * inputs else {
            throw Artifact.Failure.badHeader("\(name): \(got) values, \(outputs * inputs) expected")
        }
        timed("modulation") {
            AnimaExact.linear(a: x, w: reserve, into: out, m: 1, k: inputs, n: outputs,
                              weightIsTransposed: transposed)
        }
        // The modulation LoRA, in double like the weight: `y += (x·down)·up`.
        let target = String(name.dropLast(".weight".count))
        if let lora, let down = loraDown, let up = loraUp, lora.rank(target) > 0 {
            let r = try timed("lora weights") { try lora.materialize(target, k: inputs, n: outputs, down: down, up: up) }
            timed("lora modulation") {
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

    /// `C[m, n] = A[m, k] · W`, on the GPU (and the AMX if the driver cuts out).
    ///
    /// - Parameters rowsA, rowsC: the **reserved** rows of the two slices. Metal wrappers
    ///   are memoized by address and at their maximum size; so we always wrap the whole slice,
    ///   never the part a call uses.
    private func linearGPU(_ name: String, a: UnsafeMutablePointer<Float>, into c: UnsafeMutablePointer<Float>,
                           m: Int, k: Int, n: Int, rowsA: Int? = nil, rowsC: Int? = nil) throws {
        let got = try timed("widening") { try artifact.materialize(name, into: reserve, capacity: reserveCapacity) }
        guard got == k * n else {
            throw Artifact.Failure.badHeader("\(name): \(got) values, \(k * n) expected")
        }
        if AnimaDiT.exact.contains("gemm") {
            timed("gemm exact") { AnimaExact.linear(a: a, w: reserve, into: c, m: m, k: k, n: n,
                                                    weightIsTransposed: transposed) }
            return
        }
        let ab = try gemm.wrap(UnsafeMutableRawPointer(a), bytes: (rowsA ?? imageTokens) * k * 4, name: "a")
        let wb = try gemm.wrap(UnsafeMutableRawPointer(reserve), bytes: config.dim * config.hidden * 4, name: "w")
        let cb = try gemm.wrap(UnsafeMutableRawPointer(c), bytes: (rowsC ?? imageTokens) * n * 4, name: "c")
        timed("gemm wall") {
            timings["gemm GPU", default: 0] += gemm.linear(a: ab, b: wb, c: cb, m: m, k: k, n: n,
                                                            weightIsTransposed: transposed)
        }
        let target = String(name.dropLast(".weight".count))
        if let lora, let down = loraDown, let up = loraUp, let mid = loraMid, lora.rank(target) > 0 {
            if loraBuffers == nil {
                let rmax = lora.maxRank, d = config.dim
                loraBuffers = (try gemm.wrap(UnsafeMutableRawPointer(down), bytes: max(lora.maxEntry, d) * rmax * 4, name: "loraBas"),
                               try gemm.wrap(UnsafeMutableRawPointer(up), bytes: rmax * max(lora.maxOutput, d) * 4, name: "loraHaut"),
                               try gemm.wrap(UnsafeMutableRawPointer(mid), bytes: max(imageTokens, textLength) * rmax * 4, name: "loraMid"))
            }
            let r = try timed("lora weights") { try lora.materialize(target, k: k, n: n, down: down, up: up) }
            let t = loraBuffers!
            timed("lora wall") {
                timings["lora GPU", default: 0] += gemm.lora(x: ab, down: t.down, mid: t.mid, up: t.up, c: cb,
                                                             m: m, k: k, r: r, n: n)
            }
        }
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
/// 22 pairs do not rotate.
package enum AnimaRope {
    package static func write(into out: UnsafeMutablePointer<Float>, tilesHigh: Int, tilesWide: Int,
                             headDim: Int, scale: [Double]) {
        let dimH = headDim / 6 * 2, dimW = dimH, dimT = headDim - dimH - dimW
        func frequencies(_ dims: Int, _ factor: Double) -> [Double] {
            let theta = 10000.0 * pow(factor, Double(dims) / Double(dims - 2))
            return (0..<(dims / 2)).map { 1 / pow(theta, Double(2 * $0) / Double(dims)) }
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
                    let angle = Double(row) * f
                    token[2 * index] = Float(cos(angle)); token[2 * index + 1] = Float(sin(angle)); index += 1
                }
                for f in fw {
                    let angle = Double(column) * f
                    token[2 * index] = Float(cos(angle)); token[2 * index + 1] = Float(sin(angle)); index += 1
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
    package static func modulate(_ x: UnsafePointer<Float>, into out: UnsafeMutablePointer<Float>,
                                shift: UnsafePointer<Float>, scale: UnsafePointer<Float>,
                                scaleOne: UnsafeMutablePointer<Float>, rows: Int, columns: Int) {
        var one: Float = 1
        vDSP_vsadd(scale, 1, &one, scaleOne, 1, vDSP_Length(columns))
        Ops.layerNorm(x, into: out, rows: rows, columns: columns, eps: 1e-6)
        let n = vDSP_Length(columns)
        let factor = UnsafePointer(scaleOne)
        Parallel.rows(rows, width: columns) { first, howMany in
            for row in first..<(first + howMany) {
                let r = out + row * columns
                vDSP_vma(r, 1, factor, 1, shift, 1, r, 1, n)
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

/// The double computations: those of the `SILICONED_ANIMA_EXACT` diagnostic, and the modulation
/// `gemv`, which borrows them permanently (see `AnimaDiT.gemv`).
enum AnimaExact {
    /// `C[m, n] = A[m, k] · W`, W laid out `[k, n]` if the forge transposed it, `[n, k]` otherwise.
    static func linear(a: UnsafePointer<Float>, w: UnsafePointer<Float>, into c: UnsafeMutablePointer<Float>,
                       m: Int, k: Int, n: Int, weightIsTransposed: Bool) {
        var a64 = [Double](repeating: 0, count: m * k), w64 = [Double](repeating: 0, count: k * n)
        vDSP_vspdp(a, 1, &a64, 1, vDSP_Length(m * k))
        vDSP_vspdp(w, 1, &w64, 1, vDSP_Length(k * n))
        var c64 = [Double](repeating: 0, count: m * n)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, weightIsTransposed ? CblasNoTrans : CblasTrans,
                    Int32(m), Int32(n), Int32(k), 1, a64, Int32(k), w64,
                    Int32(weightIsTransposed ? n : k), 0, &c64, Int32(n))
        vDSP_vdpsp(c64, 1, c, 1, vDSP_Length(m * n))
    }

    /// SDPA without mask, `[S, H·Dh]` in and out, head by head, in double.
    static func attention(q: UnsafePointer<Float>, k: UnsafePointer<Float>, v: UnsafePointer<Float>,
                          into out: UnsafeMutablePointer<Float>, queries: Int, keys: Int,
                          heads: Int, headDim: Int) {
        let width = heads * headDim, scale = 1 / Double(headDim).squareRoot()
        for h in 0..<heads {
            func gather(_ x: UnsafePointer<Float>, _ rows: Int) -> [Double] {
                var out = [Double](repeating: 0, count: rows * headDim)
                for r in 0..<rows { for j in 0..<headDim { out[r * headDim + j] = Double(x[r * width + h * headDim + j]) } }
                return out
            }
            let qh = gather(q, queries), kh = gather(k, keys), vh = gather(v, keys)
            var scores = [Double](repeating: 0, count: queries * keys)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(queries), Int32(keys), Int32(headDim),
                        scale, qh, Int32(headDim), kh, Int32(headDim), 0, &scores, Int32(keys))
            scores.withUnsafeMutableBufferPointer { s in
                nonisolated(unsafe) let s = s.baseAddress!   // one row per thread
                DispatchQueue.concurrentPerform(iterations: queries) { r in
                    let row = s + r * keys
                    var peak = -Double.infinity
                    for j in 0..<keys { peak = max(peak, row[j]) }
                    var total = 0.0
                    for j in 0..<keys { row[j] = exp(row[j] - peak); total += row[j] }
                    for j in 0..<keys { row[j] /= total }
                }
            }
            var result = [Double](repeating: 0, count: queries * headDim)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(queries), Int32(headDim), Int32(keys),
                        1, scores, Int32(keys), vh, Int32(headDim), 0, &result, Int32(headDim))
            for r in 0..<queries { for j in 0..<headDim { out[r * width + h * headDim + j] = Float(result[r * headDim + j]) } }
        }
    }
}
