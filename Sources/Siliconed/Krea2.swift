import Metal
import Accelerate
import Foundation

/// The Krea 2 Turbo DiT — `Krea2Transformer2DModel`, 12.9 G parameters —, in fp32.
///
///     12 Qwen3-VL hidden states [T, 12, 2560] ─ fusion (2 blocks per token, projector, 2 blocks)
///                                                ─ txt_in ─ text [T, 6144] ───────────────────┐
///     latent [16, H, W] ─ patchify (SLOW channel) ─ img_in ─ image [S, 6144] ──── concat ──────┤
///                                                                                            ▼
///     σ ─ sinusoid 256 (cos first, ×1000) ─ MLP ─ temb [6144] ─ GELU ─ time_mod_proj ─▶ blocks ×28
///                                                        │                                    │
///                                                        └──────────── final layer ◀── image only
///
/// One block, single-stream — text and image in **one** sequence, one attention:
///
///     m = temb_mod + block_table          (6 × 6144: prescale, preshift, pregate, postscale, …)
///     x += pregate · Wo( SDPA(RoPE(n(q)), RoPE(n(k)), v) ⊙ σ(Wg·h) ),  h = RMSNorm(x)(1+prescale)+preshift
///     x += postgate · SwiGLU( RMSNorm(x)(1+postscale) + postshift )
///
/// What sets it apart from the other two targets, and what a port by analogy would miss:
///
///   - **the attention is an ordinary softmax SDPA**, followed by a **sigmoid gate on its
///     output**, element by element. Our first notes announced a "sigmoid attention": that was
///     a reading of the paper, the code says `dispatch_attention_fn` then `* torch.sigmoid(gate)`.
///     No new kernel, then;
///   - **GQA 48/12**: four `q` heads per `k`/`v` head, read without repetition — the SDPA lays out the
///     four `q` heads of a group on the query axis (`Attention(kvHeads:)`);
///   - the modulation is **shared**: a single `time_mod_proj` per evaluation, and each block only
///     adds a learned table to it — 3 `gemv` per evaluation instead of 84 for Anima;
///   - the RMSNorms carry a **zero-centered** weight (`1 + w`): the forge folded it in (fp32);
///   - the text goes through a learned **fusion** of the twelve drawn layers (`Krea2TextFusion`),
///     computed once per render;
///   - the RoPE is **already interleaved** (Flux convention, θ = 1000, axes 32/48/48), and the text is
///     at position `(0, 0, 0)` — it does not rotate.
///
/// **The text padding is removed, and this is exact.** The reference pads the text to 512 tokens
/// and masks the padding tokens *as keys*, everywhere; their own rows are
/// never read again (sliced off before the final layer). Without mask or padding, each useful row
/// sees exactly the same keys: same sum. The golden tensors, computed WITH padding, are
/// the check.
package final class Krea2DiT {
    package struct Config {
        package let dim, heads, kvHeads, headDim, hidden, layers, channels, timeDim: Int
        package let axes: [Int]
        package let theta, eps: Double

        package init(header: [String: Any]) {
            let c = header["config"] as? [String: Any] ?? [:]
            heads = c["num_attention_heads"] as? Int ?? 48
            kvHeads = c["num_key_value_heads"] as? Int ?? 12
            headDim = c["attention_head_dim"] as? Int ?? 128
            dim = heads * headDim
            hidden = c["intermediate_size"] as? Int ?? 16384
            layers = c["num_layers"] as? Int ?? 28
            channels = c["in_channels"] as? Int ?? 64
            timeDim = c["timestep_embed_dim"] as? Int ?? 256
            axes = c["axes_dims_rope"] as? [Int] ?? [32, 48, 48]
            theta = c["rope_theta"] as? Double ?? 1000
            eps = c["norm_eps"] as? Double ?? 1e-5
        }
        /// The latent channels: `in_channels = 16 · 2 · 2`.
        package var latentChannels: Int { channels / 4 }
        package var kvWidth: Int { kvHeads * headDim }
    }

    package let config: Config
    package private(set) var boundaries: [String: [Float]] = [:]
    package var recordBoundaries = false
    /// The token of the render in progress, consulted between two layers (see `Cancellation`).
    package var cancellation: Cancellation?
    package private(set) var timings: [String: Double] = [:]
    package func resetTimings() { timings.removeAll() }

    private let artifact: Artifact
    private let prefetcher: Prefetcher
    private let gemm: GEMM
    private let arena: Arena
    /// The sizing grid, which is that of every evaluation (no spectral schedule here).
    private let latentHeight, latentWidth, imageTokens, maxText, maxRows: Int
    /// The GQA SDPA, per sequence length. The flash kernel (`FlashMatrix`) is no longer
    /// wired in: it wants repeated `k`/`v`, and it tied on Krea 2.
    private var attentions: [Int: Attention] = [:]
    /// **`SILICONED_KREA2_EXACT=gemm,sdpa` — a diagnostic instrument, never a render path.**
    /// The GEMM and/or the SDPA in DOUBLE on the CPU (`AnimaExact`), to attribute a deviation to one or
    /// the other — or to the fp32 oracle. Slow, and announced.
    private static let exact: Set<String> = Set(
        (ProcessInfo.processInfo.environment["SILICONED_KREA2_EXACT"] ?? "")
            .split(separator: ",").map(String.init))

    private let x, normed, q, k, v, gate, merged, o, h1, h2: UnsafeMutablePointer<Float>
    private let reserve, tokens, projected, freqs: UnsafeMutablePointer<Float>
    /// The floats `reserve` holds (`d·h`, the largest weight): what `materialize` may write into it.
    private var reserveCapacity: Int { config.dim * config.hidden }
    /// The six modulation vectors of a block (`temb_mod + table`), where the GPU reads them.
    private let modulationSlot: UnsafeMutablePointer<Float>
    /// The block's elementwise operations — gate, SwiGLU, residuals — on the GPU. The norms stay on the
    /// CPU: on the GPU, the trajectory's fp64 reference rejected them (see `ElementwiseGPU`).
    private let elementwise: ElementwiseGPU
    /// **The LoRA stack** (`ForgeLoRA`, Krea 2 family): `ΔW = B·A` per module, applied by
    /// `gemm.lora` — two thin products accumulated into the main GEMM's output, like Anima.
    /// Its expanded weights are kept from one evaluation to the next (`CacheLoRA`, the DiT's only).
    package let lora: LoRA?
    private let cacheLoRA: CacheLoRA?
    private let loraMid: UnsafeMutablePointer<Float>?
    private var loraMidBuffer: MTLBuffer?

    package init(artifact: Artifact, latentHeight: Int, latentWidth: Int, maxText: Int = 512,
                freezeCut: Bool = EngineSettings.effective.frozenCut, lora: LoRA? = nil) throws {
        guard artifact.header["kind"] as? String == "krea2-turbo-dit" else {
            throw Artifact.Failure.badHeader("map \(artifact.header["kind"] ?? "?"): krea2-turbo-dit expected")
        }
        guard artifact.header["rope_layout"] as? String == "interleaved",
              artifact.header["rmsnorm_plus_one_folded"] as? Bool == true,
              artifact.linearWeightsTransposed else {
            throw Artifact.Failure.badHeader("Krea 2 map forged without interleaved RoPE, folded `1 + w` or transposed Linears")
        }
        self.artifact = artifact
        self.config = Config(header: artifact.header)
        self.prefetcher = Prefetcher(artifact: artifact)
        self.gemm = try GEMM(freezeCut: freezeCut)
        self.elementwise = try ElementwiseGPU(device: gemm.device, queue: gemm.queue)
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
        self.maxText = maxText
        imageTokens = (latentHeight / 2) * (latentWidth / 2)
        maxRows = imageTokens + maxText

        let n = maxRows, d = config.dim, h = config.hidden, kv = config.kvWidth
        let r = lora?.maxRank ?? 0
        let floats = 6 * n * d + 2 * n * kv + 2 * n * h + d * h + 6 * d
            + imageTokens * config.channels * 2 + n * config.headDim + n * r
        let arena = try Arena(capacity: floats * 4 + 32 * Arena.alignment + (8 << 20))
        self.arena = arena
        func slot(_ name: String, _ count: Int) throws -> UnsafeMutablePointer<Float> {
            try arena.reserve(name, bytes: count * 4).assumingMemoryBound(to: Float.self)
        }
        x = try slot("x", n * d); normed = try slot("normed", n * d)
        q = try slot("q", n * d); k = try slot("k", n * kv); v = try slot("v", n * kv)
        gate = try slot("gate", n * d); merged = try slot("merged", n * d); o = try slot("o", n * d)
        h1 = try slot("h1", n * h); h2 = try slot("h2", n * h)
        modulationSlot = try slot("modulation", 6 * d)
        reserve = try slot("reserve", d * h)
        tokens = try slot("tokens", imageTokens * config.channels)
        projected = try slot("projected", imageTokens * config.channels)
        freqs = try slot("freqs", n * config.headDim)
        self.lora = r > 0 ? lora : nil
        if let lora, r > 0 {
            loraMid = try slot("loraMid", n * r)
            cacheLoRA = try CacheLoRA(lora: lora, retain: { $0.hasPrefix("transformer_blocks.") || $0.hasPrefix("final_layer.") })
        } else {
            loraMid = nil; cacheLoRA = nil
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

    /// `latent` as `[16, H, W]`, `text` as `[T, 6144]` (the output of `txt_in`, without padding), `sigma`
    /// as the pipeline passes it (`t / 1000`, which **is** σ). Returns the velocity as `[16, H, W]`.
    package func forward(latent: UnsafePointer<Float>, text: UnsafePointer<Float>, textRows t: Int,
                        sigma: Float) throws -> [Float] {
        let d = config.dim, c = config.latentChannels
        let (height, width) = (latentHeight, latentWidth)
        guard t <= maxText else {
            throw Artifact.Failure.badHeader("\(t) text tokens, at most \(maxText)")
        }
        let s = imageTokens, n = t + s

        let tabulated = modulations[sigma.bitPattern]
        if tabulated == nil {
            prefetcher.request(["time_embed.linear_1.weight", "time_embed.linear_1.bias",
                                "time_embed.linear_2.weight", "time_embed.linear_2.bias",
                                "time_mod_proj.weight", "time_mod_proj.bias"])
        }
        prefetcher.request(["img_in.weight", "img_in.bias"])
        prefetcher.request(prefetcher.namesOfBlock(prefix: "transformer_blocks.0."))

        // ── the timestep: temb [d], then the shared modulation temb_mod [6d] ─────────────────
        let (temb, tembMod) = try tabulated ?? timed("modulation") { try modulation(sigmas: [sigma])[0] }
        record("temb", temb, d)
        record("temb_mod", tembMod, 6 * d)

        // ── the sequence: [text T, image S] ─────────────────────────────────────────────────
        x.update(from: text, count: t * d)
        Krea2Ops.patchify(latent, into: tokens, channels: c, height: height, width: width)
        // Through `o`: a Metal wrapper wants a page-aligned address, and `x + t·d` is
        // not.
        try linearGPU("img_in.weight", a: tokens, into: o, m: s, k: config.channels, n: d,
                      rowsA: imageTokens)
        (x + t * d).update(from: o, count: s * d)
        try addBias("img_in.bias", to: x + t * d, rows: s, columns: d)
        record("img_in_out", x + t * d, s * d)

        Krea2Rope.write(into: freqs, textRows: t, tilesHigh: height / 2, tilesWide: width / 2,
                        axes: config.axes, theta: config.theta)

        for layer in 0..<config.layers {
            try cancellation.check()   // between two layers, never inside one (see `Cancellation`)
            for ahead in 1...2 where layer + ahead < config.layers {
                prefetcher.request(prefetcher.namesOfBlock(prefix: "transformer_blocks.\(layer + ahead)."))
            }
            if layer == config.layers - 1 {
                prefetcher.request(prefetcher.namesOfBlock(prefix: "final_layer."))
            }
            try tembMod.withUnsafeBufferPointer {
                try block(layer, rows: n, tembMod: $0.baseAddress!)
            }
            if [0, 1, 13, 27].contains(layer) { record("block\(layer)_out", x, n * d) }
        }

        // ── the final layer, on the image only: scale = temb + table[0], shift = temb + table[1]
        let m = modulationSlot
        try artifact.materialize("final_layer.scale_shift_table", into: m, capacity: 6 * config.dim)
        for j in 0..<d { m[j] += temb[j]; m[d + j] += temb[j] }
        try modulatedNorm("final_layer.norm.weight", x + t * d, rows: s, scale: m, shift: m + d)
        try linearGPU("final_layer.linear.weight", a: normed, into: projected, m: s, k: d, n: config.channels,
                      rowsA: maxRows, rowsC: imageTokens)
        try addBias("final_layer.linear.bias", to: projected, rows: s, columns: config.channels)

        var out = [Float](repeating: 0, count: c * height * width)
        out.withUnsafeMutableBufferPointer {
            Krea2Ops.unpatchify(projected, into: $0.baseAddress!, channels: c, height: height, width: width)
        }
        record("model_out", out, out.count)
        return out
    }

    /// **The tabulated modulation**: σ (its bits) → `(temb, temb_mod)`. Empty until
    /// `tabulateModulation` has been called — `forward` then computes its own, as before.
    package var modulations: ModulationTable = [:]
    /// The bits of σ → `(temb, temb_mod)`. A table taken over from another DiT of the same map is as good as
    /// its own: neither the latent, nor the text, nor the LoRA enter into it.
    package typealias ModulationTable = [UInt32: (temb: [Float], mod: [Float])]

    /// **Precomputes the modulation of the σ that will be evaluated** — a single read of
    /// `time_mod_proj` (906 MB in fp32) for the whole render, instead of one per evaluation.
    ///
    /// Neither the latent nor the text enter into `temb` and `temb_mod`: they are functions of σ
    /// alone. The schedule is known at the start; in img2img only the **evaluated** σ are passed (the
    /// tail from `t_start`). Each vector is accumulated in the same order as in an isolated
    /// pass (`Krea2Exact.linears`): the table gives **the same bits** as the per-step computation.
    /// A σ absent from the table is computed on the fly, as before.
    package func tabulateModulation(sigmas: [Float]) throws {
        let newKeys = Array(Set(sigmas.map(\.bitPattern)).subtracting(modulations.keys)).sorted()
        guard !newKeys.isEmpty else { return }
        let values = try timed("modulation") { try modulation(sigmas: newKeys.map(Float.init(bitPattern:))) }
        for (key, value) in zip(newKeys, values) { modulations[key] = value }
    }

    /// `temb` and `temb_mod` for several σ, **in double**, each weight read once for all.
    private func modulation(sigmas: [Float]) throws -> [(temb: [Float], mod: [Float])] {
        let sinusoids = sigmas.map { Krea2Exact.sinusoid(sigma: $0, dim: config.timeDim).map(Double.init) }
        var hiddenValues = try Krea2Exact.linears(artifact, "time_embed.linear_1", xs: sinusoids)
        for i in hiddenValues.indices { Krea2Exact.geluTanh(&hiddenValues[i]) }
        let tembs = try Krea2Exact.linears(artifact, "time_embed.linear_2", xs: hiddenValues)
        var enabled = tembs
        for i in enabled.indices { Krea2Exact.geluTanh(&enabled[i]) }
        let mods = try Krea2Exact.linears(artifact, "time_mod_proj", xs: enabled)
        return zip(tembs, mods).map { ($0.map(Float.init), $1.map(Float.init)) }
    }

    /// A single-stream block.
    private func block(_ layer: Int, rows n: Int, tembMod: UnsafePointer<Float>) throws {
        let d = config.dim, hh = config.hidden, kv = config.kvWidth
        let heads = config.heads, dh = config.headDim
        let prefix = "transformer_blocks.\(layer)."
        let tag = layer == 0 ? "block0_" : nil

        // m = temb_mod + table, six vectors of d, in the slice the GPU reads
        let m = modulationSlot
        try artifact.materialize(prefix + "scale_shift_table", into: m, capacity: 6 * config.dim)
        vDSP_vadd(m, 1, tembMod, 1, m, 1, vDSP_Length(6 * d))
        let mb = try slice(m), xb = try slice(x), ob = try slice(o)
        let pregate = (mb, 2 * d), postgate = (mb, 5 * d)
        // ── attention ─────────────────────────────────────────────────────────────────────
        try modulatedNorm(prefix + "norm1.weight", x, rows: n, scale: m, shift: m + d)
        try linearGPU(prefix + "attn.to_q.weight", a: normed, into: q, m: n, k: d, n: d)
        try linearGPU(prefix + "attn.to_k.weight", a: normed, into: k, m: n, k: d, n: kv)
        try linearGPU(prefix + "attn.to_v.weight", a: normed, into: v, m: n, k: d, n: kv)
        try linearGPU(prefix + "attn.to_gate.weight", a: normed, into: gate, m: n, k: d, n: d)
        try headNorm(prefix + "attn.norm_q.weight", q, rows: n * heads)
        try headNorm(prefix + "attn.norm_k.weight", k, rows: n * config.kvHeads)
        timed("rope") {
            Ops.rope(q, freqs: freqs, sequence: n, heads: heads, headDim: dh)
            Ops.rope(k, freqs: freqs, sequence: n, heads: config.kvHeads, headDim: dh)
        }
        try attend(rows: n)
        let (mergedb, gateb) = (try slice(merged), try slice(gate))
        timed("gate") { timings["elementwise GPU", default: 0] += elementwise.carries((mergedb, 0), (gateb, 0), count: n * d) }
        try linearGPU(prefix + "attn.to_out.0.weight", a: merged, into: o, m: n, k: d, n: d)
        if let tag { record(tag + "attn_out", o, n * d) }
        timed("norm") {
            timings["elementwise GPU", default: 0] += elementwise.residual((xb, 0), plus: (ob, 0), carries: pregate,
                                                                          rows: n, columns: d)
        }

        // ── SwiGLU ────────────────────────────────────────────────────────────────────────
        try modulatedNorm(prefix + "norm2.weight", x, rows: n, scale: m + 3 * d, shift: m + 4 * d)
        try linearGPU(prefix + "ff.gate.weight", a: normed, into: h1, m: n, k: d, n: hh)
        try linearGPU(prefix + "ff.up.weight", a: normed, into: h2, m: n, k: d, n: hh)
        let (h1b, h2b) = (try slice(h1), try slice(h2))
        timed("swiglu") { timings["elementwise GPU", default: 0] += elementwise.swiglu((h1b, 0), (h2b, 0), count: n * hh) }
        try linearGPU(prefix + "ff.down.weight", a: h1, into: o, m: n, k: hh, n: d)
        if let tag { record(tag + "ff_out", o, n * d) }
        timed("norm") {
            timings["elementwise GPU", default: 0] += elementwise.residual((xb, 0), plus: (ob, 0), carries: postgate,
                                                                          rows: n, columns: d)
        }
    }

    // ── plumbing ────────────────────────────────────────────────────────────────────────

    /// A slice of the arena seen from the GPU — wrapped **at its reserved size** (wrappers are
    /// memoized by address, see `linearGPU`): the modulation block, or a slice of rows.
    private func slice(_ p: UnsafeMutablePointer<Float>) throws -> MTLBuffer {
        let d = config.dim
        let floats: Int
        switch p {
        case modulationSlot: floats = 6 * d
        case k, v: floats = maxRows * config.kvWidth
        case h1, h2: floats = maxRows * config.hidden
        default: floats = maxRows * d
        }
        return try gemm.wrap(UnsafeMutableRawPointer(p), bytes: floats * 4, name: "elementwise")
    }

    /// `normed = RMSNorm(src) · (1 + scale) + shift` — the norm's weight already carries its `1 +`.
    ///
    /// **On the CPU, and a measurement decided it.** On the GPU, with a compensated sum, the
    /// norm is closer to double than `vDSP_svesq` at the judge (`elementwise`, on drawn
    /// inputs) — and yet the guided trajectory, against its fp64 references, moves away from it:
    /// step 3 at 2.3·10⁻³ from fp64 at the worst channel (median 6.3·10⁻⁵) against 6.0·10⁻⁴ (3.4·10⁻⁵) on
    /// the CPU, and the oracle at 6.8·10⁻⁴. The QK-Norm alone is enough to cause it (1.35·10⁻³). A better
    /// rounding per operation is not a better trajectory; the trajectory is what judges.
    private func modulatedNorm(_ name: String, _ source: UnsafePointer<Float>, rows: Int,
                               scale: UnsafePointer<Float>, shift: UnsafePointer<Float>) throws {
        let d = config.dim
        try artifact.materialize(name, into: reserve, capacity: reserveCapacity)
        timed("norm") {
            Ops.rmsNorm(source, weight: reserve, into: normed, rows: rows, columns: d, eps: Float(config.eps))
            Krea2Ops.modulate(normed, scale: scale, shift: shift, rows: rows, columns: d)
        }
    }

    /// QK-Norm: RMSNorm over `head_dim`, eps 1e-5, folded weight. On the CPU, like `modulatedNorm`.
    private func headNorm(_ name: String, _ tensor: UnsafeMutablePointer<Float>, rows: Int) throws {
        try artifact.materialize(name, into: reserve, capacity: reserveCapacity)
        timed("norm") { Ops.rmsNorm(tensor, weight: reserve, into: tensor, rows: rows,
                                    columns: config.headDim, eps: Float(config.eps)) }
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
        let d = config.dim, kv = config.kvWidth, group = config.heads / config.kvHeads
        if Krea2DiT.exact.contains("sdpa") {
            // The double instrument wants repeated `k`/`v` heads: it repeats them for itself.
            var kWide = [Float](repeating: 0, count: n * d), vWide = kWide
            Krea2Ops.expandKV(k, into: &kWide, rows: n, kvHeads: config.kvHeads, group: group, headDim: config.headDim)
            Krea2Ops.expandKV(v, into: &vWide, rows: n, kvHeads: config.kvHeads, group: group, headDim: config.headDim)
            timed("sdpa exact") {
                AnimaExact.attention(q: q, k: kWide, v: vWide, into: merged, queries: n, keys: n,
                                     heads: config.heads, headDim: config.headDim)
            }
            return
        }
        let attention: Attention
        if let a = attentions[n] { attention = a } else {
            attention = Attention(device: gemm.device, queue: gemm.queue, heads: config.heads,
                                  sequence: n, headDim: config.headDim, kvHeads: config.kvHeads)
            attentions[n] = attention
        }
        let qb = try gemm.wrap(UnsafeMutableRawPointer(q), bytes: maxRows * d * 4, name: "q")
        let kb = try gemm.wrap(UnsafeMutableRawPointer(k), bytes: maxRows * kv * 4, name: "k")
        let vb = try gemm.wrap(UnsafeMutableRawPointer(v), bytes: maxRows * kv * 4, name: "v")
        let ob = try gemm.wrap(UnsafeMutableRawPointer(merged), bytes: maxRows * d * 4, name: "merged")
        timed("sdpa wall") { timings["sdpa GPU", default: 0] += attention.run(q: qb, k: kb, v: vb, into: ob) }
    }

    /// `C[m, n] = A[m, k] · W`, W laid out `[k, n]` by the forge.
    ///
    /// - Parameters rowsA, rowsC: the **reserved** rows of the two slices (default: `maxRows`).
    ///   Metal wrappers are memoized by address, at their maximum size.
    private func linearGPU(_ name: String, a: UnsafeMutablePointer<Float>, into c: UnsafeMutablePointer<Float>,
                           m: Int, k: Int, n: Int, rowsA: Int? = nil, rowsC: Int? = nil) throws {
        let got = try timed("widening") { try artifact.materialize(name, into: reserve, capacity: reserveCapacity) }
        guard got == k * n else {
            throw Artifact.Failure.badHeader("\(name): \(got) values, \(k * n) expected")
        }
        if Krea2DiT.exact.contains("gemm") {
            timed("gemm exact") { AnimaExact.linear(a: a, w: reserve, into: c, m: m, k: k, n: n,
                                                    weightIsTransposed: true) }
            return
        }
        let ab = try gemm.wrap(UnsafeMutableRawPointer(a), bytes: (rowsA ?? maxRows) * k * 4, name: "a")
        let wb = try gemm.wrap(UnsafeMutableRawPointer(reserve), bytes: config.dim * config.hidden * 4, name: "w")
        let cb = try gemm.wrap(UnsafeMutableRawPointer(c), bytes: (rowsC ?? maxRows) * n * 4, name: "c")
        timed("gemm wall") {
            timings["gemm GPU", default: 0] += gemm.linear(a: ab, b: wb, c: cb, m: m, k: k, n: n,
                                                            weightIsTransposed: true)
        }
        let target = String(name.dropLast(".weight".count))
        if let lora, let cache = cacheLoRA, let mid = loraMid,
           let stack = try timed("lora weights", { try cache.module(target, k: k, n: n) }) {
            if loraMidBuffer == nil {
                loraMidBuffer = try gemm.wrap(UnsafeMutableRawPointer(mid), bytes: maxRows * lora.maxRank * 4,
                                              name: "loraMid")
            }
            let down = try gemm.wrap(UnsafeMutableRawPointer(stack.down), bytes: k * stack.rank * 4, name: "loraBas")
            let up = try gemm.wrap(UnsafeMutableRawPointer(stack.up), bytes: stack.rank * n * 4, name: "loraHaut")
            timed("lora wall") {
                timings["lora GPU", default: 0] += gemm.lora(x: ab, down: down, mid: loraMidBuffer!, up: up, c: cb,
                                                             m: m, k: k, r: stack.rank, n: n)
            }
        }
    }
}

/// **The Krea 2 text fusion**: the twelve hidden states drawn from Qwen3-VL → `[T, 6144]`.
///
///     [T, 12, 2560] ─ 2 "per-layer" blocks (attention over the axis of the 12 layers, token by token)
///                   ─ projector (12 → 1) ─ [T, 2560] ─ 2 "refiner" blocks (attention over T)
///                   ─ txt_in : RMSNorm · Linear · GELU(tanh) · Linear ─ [T, 6144]
///
/// Neither σ nor the latent enter into it: it runs **once per render**, before the DiT reserves
/// its arena — its own is returned beforehand. Its blocks are pre-norm blocks without RoPE or
/// modulation, with 20 heads of 128 without GQA, and with the same sigmoid gate as the DiT. Their
/// attentions are small (12 × 12 per token, T × T at most 512²): they run on the CPU.
package final class Krea2TextFusion {
    package let dim = 2560, heads = 20, headDim = 128, hidden = 6912, layers = 12
    package let outDim: Int
    package private(set) var boundaries: [String: [Float]] = [:]
    package var recordBoundaries = false
    /// The token of the render in progress, consulted between two layers (see `Cancellation`).
    package var cancellation: Cancellation?

    private let artifact: Artifact
    private let gemm: GEMM
    private let arena: Arena
    private let rows: Int
    private let x, normed, q, k, v, gate, merged, o, h1, h2, scratch, reserve: UnsafeMutablePointer<Float>
    /// The floats `reserve` holds (`widest`): what `materialize` may write into it.
    private var reserveCapacity: Int { reserved[reserve]! / 4 }
    /// The reserved size of each slice, per address: a Metal wrapper is always made at
    /// the size of the slice, never that of the call (they are memoized by address).
    private let reserved: [UnsafeMutablePointer<Float>: Int]
    private let eps: Float

    /// - Parameter maxTokens: the number of text tokens (`T`) — the "per-layer" blocks
    ///   process `12 T` rows.
    /// The LoRA stack: a Krea 2 LoRA also touches the fusion (32 modules out of 256 for Incase).
    package let lora: LoRA?
    private let loraDown, loraUp, loraMid: UnsafeMutablePointer<Float>?

    package init(artifact: Artifact, maxTokens: Int, freezeCut: Bool = EngineSettings.effective.frozenCut,
                lora: LoRA? = nil) throws {
        guard artifact.header["kind"] as? String == "krea2-turbo-dit" else {
            throw Artifact.Failure.badHeader("the text fusion lives in the Krea 2 DiT map")
        }
        let config = Krea2DiT.Config(header: artifact.header)
        self.artifact = artifact
        self.outDim = config.dim
        self.eps = Float(config.eps)
        self.gemm = try GEMM(freezeCut: freezeCut)
        let r = maxTokens * 12
        self.rows = r
        let d = 2560, h = 6912, widest = max(config.dim * config.dim, d * config.dim, d * h)
        let rank = lora?.maxRank ?? 0
        let floats = 7 * r * d + 3 * r * h + widest + r * config.dim * 2 + (rank > 0 ? 2 * h * rank + r * rank : 0)
        let arena = try Arena(capacity: floats * 4 + 16 * Arena.alignment + (8 << 20))
        self.arena = arena
        var sizes: [UnsafeMutablePointer<Float>: Int] = [:]
        func slot(_ name: String, _ count: Int) throws -> UnsafeMutablePointer<Float> {
            let p = try arena.reserve(name, bytes: count * 4).assumingMemoryBound(to: Float.self)
            sizes[p] = count * 4
            return p
        }
        x = try slot("x", r * d); normed = try slot("normed", r * max(d, config.dim))
        q = try slot("q", r * d); k = try slot("k", r * d); v = try slot("v", r * d)
        gate = try slot("gate", r * d); merged = try slot("merged", r * max(d, config.dim))
        o = try slot("o", r * max(d, config.dim))
        h1 = try slot("h1", r * h); h2 = try slot("h2", r * h); scratch = try slot("scratch", r * h)
        reserve = try slot("reserve", widest)
        self.lora = rank > 0 ? lora : nil
        if rank > 0 {
            loraDown = try slot("loraBas", h * rank); loraUp = try slot("loraHaut", rank * h)
            loraMid = try slot("loraMid", r * rank)
        } else {
            loraDown = nil; loraUp = nil; loraMid = nil
        }
        reserved = sizes
    }

    private func record(_ name: String, _ p: UnsafePointer<Float>, _ n: Int) {
        guard recordBoundaries else { return }
        boundaries[name] = Array(UnsafeBufferPointer(start: p, count: n))
    }

    /// `hidden` en `[T, 12, 2560]` → `[T, 6144]`.
    package func fuse(_ hidden: UnsafePointer<Float>, tokens t: Int) throws -> [Float] {
        let d = dim
        precondition(t * 12 <= rows, "\(t) tokens, arena sized for \(rows / 12)")
        x.update(from: hidden, count: t * 12 * d)
        for i in 0..<2 {
            try cancellation.check()
            try fusionBlock("text_fusion.layerwise_blocks.\(i).", rows: t * 12, batches: t, length: 12)
        }
        // the projector: out[t, c] = Σ_l x[t, l, c] · w[l]
        var w = [Float](repeating: 0, count: 12)
        try w.withUnsafeMutableBufferPointer { _ = try artifact.materialize("text_fusion.projector.weight", into: $0) }
        for token in 0..<t {
            let out = normed + token * d
            for c in 0..<d { out[c] = 0 }
            for l in 0..<12 {
                var weight = w[l]
                vDSP_vsma(x + (token * 12 + l) * d, 1, &weight, out, 1, out, 1, vDSP_Length(d))
            }
        }
        x.update(from: normed, count: t * d)
        record("projected", x, t * d)
        for i in 0..<2 {
            try cancellation.check()
            try fusionBlock("text_fusion.refiner_blocks.\(i).", rows: t, batches: 1, length: t)
        }
        record("text_fused", x, t * d)

        // ── txt_in ─────────────────────────────────────────────────────────────────────────
        let wide = outDim
        try artifact.materialize("txt_in.norm.weight", into: reserve, capacity: reserveCapacity)
        Ops.rmsNorm(x, weight: reserve, into: normed, rows: t, columns: d, eps: eps)
        try linear("txt_in.linear_1.weight", a: normed, into: merged, m: t, k: d, n: wide)
        try addBias("txt_in.linear_1.bias", merged, rows: t, columns: wide)
        Krea2Ops.geluTanh(merged, count: t * wide)
        try linear("txt_in.linear_2.weight", a: merged, into: o, m: t, k: wide, n: wide)
        try addBias("txt_in.linear_2.bias", o, rows: t, columns: wide)
        record("txt_in_out", o, t * wide)
        return Array(UnsafeBufferPointer(start: o, count: t * wide))
    }

    /// `x += attn(norm1(x)) ; x += ff(norm2(x))` — without RoPE, without modulation, without mask.
    private func fusionBlock(_ prefix: String, rows r: Int, batches: Int, length: Int) throws {
        let d = dim
        try artifact.materialize(prefix + "norm1.weight", into: reserve, capacity: reserveCapacity)
        Ops.rmsNorm(x, weight: reserve, into: normed, rows: r, columns: d, eps: eps)
        try linear(prefix + "attn.to_q.weight", a: normed, into: q, m: r, k: d, n: d)
        try linear(prefix + "attn.to_k.weight", a: normed, into: k, m: r, k: d, n: d)
        try linear(prefix + "attn.to_v.weight", a: normed, into: v, m: r, k: d, n: d)
        try linear(prefix + "attn.to_gate.weight", a: normed, into: gate, m: r, k: d, n: d)
        try artifact.materialize(prefix + "attn.norm_q.weight", into: reserve, capacity: reserveCapacity)
        Ops.rmsNorm(q, weight: reserve, into: q, rows: r * heads, columns: headDim, eps: eps)
        try artifact.materialize(prefix + "attn.norm_k.weight", into: reserve, capacity: reserveCapacity)
        Ops.rmsNorm(k, weight: reserve, into: k, rows: r * heads, columns: headDim, eps: eps)
        Krea2Ops.attention(q: q, k: k, v: v, into: merged, batches: batches, length: length,
                           heads: heads, headDim: headDim)
        Krea2Ops.sigmoidGate(merged, gate: gate, count: r * d, scratch: scratch)
        try linear(prefix + "attn.to_out.0.weight", a: merged, into: o, m: r, k: d, n: d)
        vDSP_vadd(x, 1, o, 1, x, 1, vDSP_Length(r * d))

        try artifact.materialize(prefix + "norm2.weight", into: reserve, capacity: reserveCapacity)
        Ops.rmsNorm(x, weight: reserve, into: normed, rows: r, columns: d, eps: eps)
        try linear(prefix + "ff.gate.weight", a: normed, into: h1, m: r, k: d, n: hidden)
        try linear(prefix + "ff.up.weight", a: normed, into: h2, m: r, k: d, n: hidden)
        Ops.siluGate(h1, h2, into: h1, count: r * hidden, scratch: scratch)
        try linear(prefix + "ff.down.weight", a: h1, into: o, m: r, k: hidden, n: d)
        vDSP_vadd(x, 1, o, 1, x, 1, vDSP_Length(r * d))
    }

    private func addBias(_ name: String, _ c: UnsafeMutablePointer<Float>, rows: Int, columns: Int) throws {
        try artifact.materialize(name, into: reserve, capacity: reserveCapacity)
        for row in 0..<rows {
            vDSP_vadd(c + row * columns, 1, reserve, 1, c + row * columns, 1, vDSP_Length(columns))
        }
    }

    /// Each slice is wrapped at its reserved size (the slice's whole arena). The
    /// refiners and `txt_in` run at m = T (12 to 20 tokens for a typical prompt): it is
    /// `GEMM` that submits them to 64 rows, not this arena.
    private func linear(_ name: String, a: UnsafeMutablePointer<Float>, into c: UnsafeMutablePointer<Float>,
                        m: Int, k: Int, n: Int) throws {
        let got = try artifact.materialize(name, into: reserve, capacity: reserveCapacity)
        guard got == k * n else {
            throw Artifact.Failure.badHeader("\(name): \(got) values, \(k * n) expected")
        }
        let ab = try gemm.wrap(UnsafeMutableRawPointer(a), bytes: reserved[a]!, name: "a")
        let wb = try gemm.wrap(UnsafeMutableRawPointer(reserve), bytes: reserved[reserve]!, name: "w")
        let cb = try gemm.wrap(UnsafeMutableRawPointer(c), bytes: reserved[c]!, name: "c")
        _ = gemm.linear(a: ab, b: wb, c: cb, m: m, k: k, n: n, weightIsTransposed: true)
        let target = String(name.dropLast(".weight".count))
        if let lora, let down = loraDown, let up = loraUp, let mid = loraMid, lora.rank(target) > 0 {
            let r = try lora.materialize(target, k: k, n: n, down: down, up: up)
            _ = gemm.lora(x: ab, down: try gemm.wrap(UnsafeMutableRawPointer(down), bytes: reserved[down]!, name: "loraBas"),
                          mid: try gemm.wrap(UnsafeMutableRawPointer(mid), bytes: reserved[mid]!, name: "loraMid"),
                          up: try gemm.wrap(UnsafeMutableRawPointer(up), bytes: reserved[up]!, name: "loraHaut"),
                          c: cb, m: m, k: k, r: r, n: n)
        }
    }
}

/// The Krea 2 RoPE: `Krea2RotaryPosEmbed` (Flux's), axes `(t, h, w)` = 32/48/48 dims,
/// θ = 1000, **adjacent pairs** (`repeat_interleave_real=True`) — hence the
/// `[N, 64, (cos, sin)]` table of `Ops.rope`, without permutation. The angles in double, like the reference
/// (`freqs_dtype = float64`). The text is at `(0, 0, 0)`: cos 1, sin 0, it does not rotate.
package enum Krea2Rope {
    package static func write(into out: UnsafeMutablePointer<Float>, textRows: Int, tilesHigh: Int,
                             tilesWide: Int, axes: [Int], theta: Double) {
        let pairs = axes.reduce(0, +) / 2
        func frequencies(_ dims: Int) -> [Double] {
            (0..<(dims / 2)).map { 1 / pow(theta, Double(2 * $0) / Double(dims)) }
        }
        let f = axes.map(frequencies)
        for row in 0..<textRows {
            let token = out + row * pairs * 2
            for p in 0..<pairs { token[2 * p] = 1; token[2 * p + 1] = 0 }
        }
        for h in 0..<tilesHigh {
            for w in 0..<tilesWide {
                let token = out + (textRows + h * tilesWide + w) * pairs * 2
                var index = 0
                for (axis, position) in [0, h, w].enumerated() {
                    for frequency in f[axis] {
                        let angle = Double(position) * frequency
                        token[2 * index] = Float(cos(angle)); token[2 * index + 1] = Float(sin(angle))
                        index += 1
                    }
                }
            }
        }
    }
}

/// The elementwise operations specific to Krea 2.
package enum Krea2Ops {
    /// `[C, H, W]` → `[(H/2)·(W/2), C·2·2]`, **slowest channel**: `_pack_latents` does
    /// `view(C, H/2, 2, W/2, 2).permute(2, 4, 1, 3, 5)`, so the index within the token is
    /// `(c·2 + ph)·2 + pw`. The opposite of Z-Image, like Anima (without its mask channel).
    package static func patchify(_ latent: UnsafePointer<Float>, into out: UnsafeMutablePointer<Float>,
                                channels: Int, height: Int, width: Int) {
        let tilesHigh = height / 2, tilesWide = width / 2, perToken = channels * 4
        for th in 0..<tilesHigh {
            for tw in 0..<tilesWide {
                let token = out + (th * tilesWide + tw) * perToken
                for ch in 0..<channels {
                    for ph in 0..<2 {
                        for pw in 0..<2 {
                            token[(ch * 2 + ph) * 2 + pw] = latent[(ch * height + th * 2 + ph) * width + tw * 2 + pw]
                        }
                    }
                }
            }
        }
    }

    /// The exact inverse of `patchify` — `_unpack_latents`.
    package static func unpatchify(_ tokens: UnsafePointer<Float>, into out: UnsafeMutablePointer<Float>,
                                  channels: Int, height: Int, width: Int) {
        let tilesHigh = height / 2, tilesWide = width / 2, perToken = channels * 4
        for th in 0..<tilesHigh {
            for tw in 0..<tilesWide {
                let token = tokens + (th * tilesWide + tw) * perToken
                for ch in 0..<channels {
                    for ph in 0..<2 {
                        for pw in 0..<2 {
                            out[(ch * height + th * 2 + ph) * width + tw * 2 + pw] = token[(ch * 2 + ph) * 2 + pw]
                        }
                    }
                }
            }
        }
    }

    /// `x = x · (1 + scale) + shift`, in place, per row.
    package static func modulate(_ x: UnsafeMutablePointer<Float>, scale: UnsafePointer<Float>,
                                shift: UnsafePointer<Float>, rows: Int, columns: Int) {
        var factor = [Float](repeating: 1, count: columns)
        vDSP_vadd(scale, 1, factor, 1, &factor, 1, vDSP_Length(columns))
        factor.withUnsafeBufferPointer { f in
            let factor = f.baseAddress!
            Parallel.rows(rows, width: columns) { first, howMany in
                for row in first..<(first + howMany) {
                    let r = x + row * columns
                    vDSP_vma(r, 1, factor, 1, shift, 1, r, 1, vDSP_Length(columns))
                }
            }
        }
    }

    /// `x ⊙ σ(gate)` in place — the attention's output gate.
    package static func sigmoidGate(_ x: UnsafeMutablePointer<Float>, gate: UnsafePointer<Float>, count: Int,
                                   scratch: UnsafeMutablePointer<Float>) {
        let slices = Parallel.threads * 2
        Parallel.rows(slices, width: max(1, count / slices)) { first, howMany in
            let start = first * count / slices, end = (first + howMany) * count / slices
            let n = end - start
            guard n > 0 else { return }
            var minusOne: Float = -1, one: Float = 1
            var length = Int32(n)
            let s = scratch + start
            vDSP_vsmul(gate + start, 1, &minusOne, s, 1, vDSP_Length(n))
            vvexpf(s, s, &length)
            vDSP_vsadd(s, 1, &one, s, 1, vDSP_Length(n))
            vDSP_vdiv(s, 1, x + start, 1, x + start, 1, vDSP_Length(n))
        }
    }

    /// GELU, `tanh` approximation: `½ x (1 + tanh(√(2/π) (x + 0.044715 x³)))`.
    package static func geluTanh(_ x: UnsafeMutablePointer<Float>, count: Int) {
        let c = Float((2 / Double.pi).squareRoot())
        nonisolated(unsafe) let x = x   // disjoint slices, one per thread
        DispatchQueue.concurrentPerform(iterations: max(1, count / 65536 + 1)) { piece in
            let start = piece * 65536, end = min(start + 65536, count)
            guard start < end else { return }
            for i in start..<end {
                let value = x[i]
                x[i] = 0.5 * value * (1 + tanhf(c * (value + 0.044715 * value * value * value)))
            }
        }
    }

    /// `kvHeads` heads repeated `group` times, consecutively: `repeat_interleave(group, dim=2)`,
    /// so the `q` head of index `h` reads the `k`/`v` head of index `h / group`.
    /// Only serves the double instrument now (`SILICONED_KREA2_EXACT=sdpa`): the render's SDPA reads
    /// `k`/`v` without repetition.
    package static func expandKV(_ source: UnsafePointer<Float>, into destination: UnsafeMutablePointer<Float>,
                                rows: Int, kvHeads: Int, group: Int, headDim: Int) {
        Parallel.rows(rows, width: kvHeads * group * headDim) { first, howMany in
            for s in first..<(first + howMany) {
                for kvHead in 0..<kvHeads {
                    let from = source + (s * kvHeads + kvHead) * headDim
                    for repetition in 0..<group {
                        (destination + (s * kvHeads * group + kvHead * group + repetition) * headDim)
                            .update(from: from, count: headDim)
                    }
                }
            }
        }
    }

    /// SDPA without mask, on the CPU, `[B, L, H·Dh]` → same shape: `B` independent sequences of
    /// length `L`. The fusion's "per-layer" blocks have `B = T`, `L = 12`; the refiners
    /// `B = 1`, `L = T`. The scores accumulate in fp32 through `cblas_sgemm`.
    package static func attention(q: UnsafePointer<Float>, k: UnsafePointer<Float>, v: UnsafePointer<Float>,
                                 into out: UnsafeMutablePointer<Float>, batches: Int, length: Int,
                                 heads: Int, headDim: Int) {
        let width = heads * headDim, scale = 1 / Float(headDim).squareRoot()
        // Each task (batch, head) reads q, k, v and writes its own columns of `out`.
        nonisolated(unsafe) let (q, k, v, out) = (q, k, v, out)
        DispatchQueue.concurrentPerform(iterations: batches * heads) { job in
            let b = job / heads, h = job % heads
            let base = b * length * width + h * headDim
            var scores = [Float](repeating: 0, count: length * length)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(length), Int32(length), Int32(headDim),
                        scale, q + base, Int32(width), k + base, Int32(width), 0, &scores, Int32(length))
            for r in 0..<length {
                scores.withUnsafeMutableBufferPointer { s in
                    let row = s.baseAddress! + r * length
                    var peak: Float = 0
                    vDSP_maxv(row, 1, &peak, vDSP_Length(length))
                    var minusPeak = -peak
                    vDSP_vsadd(row, 1, &minusPeak, row, 1, vDSP_Length(length))
                    var n = Int32(length)
                    vvexpf(row, row, &n)
                    var total: Float = 0
                    vDSP_sve(row, 1, &total, vDSP_Length(length))
                    vDSP_vsdiv(row, 1, &total, row, 1, vDSP_Length(length))
                }
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(length), Int32(headDim), Int32(length),
                        1, scores, Int32(length), v + base, Int32(width), 0, out + base, Int32(width))
        }
    }
}

/// The timestep `gemv`s, **in double** — Anima's lesson (`AnimaDiT.gemv`): `temb` and the
/// modulation multiply the whole residual, and an `sgemv` on an `[input, output]` weight cost a
/// factor of six there against the fp64 reference. Three products, including `time_mod_proj` (6144 → 36,864,
/// **published fp32**, read in place in the map without a copy) — once per render for all the σ
/// (`Krea2DiT.tabulateModulation`), plus once per evaluation.
enum Krea2Exact {
    /// `yᵥ = W·xᵥ + b` for each vector `xᵥ`, W laid out `[input, output]` (forge), accumulated in
    /// double. **Each row of the weight is read once for all the vectors** — this is what
    /// makes the modulation tabulable —, and each `yᵥ` accumulates in the order of an isolated pass
    /// (`i` increasing, then `j`): the result does not depend on the number of vectors, to the bit.
    static func linears(_ artifact: Artifact, _ module: String, xs: [[Double]]) throws -> [[Double]] {
        let name = module + ".weight"
        guard let tensor = artifact.tensors[name], tensor.shape.count == 2,
              xs.allSatisfy({ $0.count == tensor.shape[0] }) else {
            throw Artifact.Failure.badHeader("\(name): shape \(artifact.tensors[name]?.shape ?? []) for \(xs.first?.count ?? 0) inputs")
        }
        let inputs = tensor.shape[0], outputs = tensor.shape[1], count = xs.count
        var bias = [Float](repeating: 0, count: outputs)
        // Without a published bias (FLUX.2 has none), the bias is zero.
        if artifact.tensors[module + ".bias"] != nil {
            try bias.withUnsafeMutableBufferPointer { _ = try artifact.materialize(module + ".bias", into: $0) }
        }
        var y = [Double](repeating: 0, count: count * outputs)
        for v in 0..<count { for j in 0..<outputs { y[v * outputs + j] = Double(bias[j]) } }

        let x = xs.flatMap { $0 }
        func accumulate(_ weight: UnsafePointer<Float>) {
            let chunk = 1024, pieces = (outputs + chunk - 1) / chunk
            x.withUnsafeBufferPointer { x in
                y.withUnsafeMutableBufferPointer { out in
                    // Each thread writes its own columns `start..<end` of each `yᵥ`.
                    nonisolated(unsafe) let (weight, x, out) = (weight, x.baseAddress!, out.baseAddress!)
                    DispatchQueue.concurrentPerform(iterations: pieces) { piece in
                        let start = piece * chunk, end = min(start + chunk, outputs)
                        for i in 0..<inputs {
                            let row = weight + i * outputs
                            for v in 0..<count where x[v * inputs + i] != 0 {
                                let xi = x[v * inputs + i], o = out + v * outputs
                                for j in start..<end { o[j] += xi * Double(row[j]) }
                            }
                        }
                    }
                }
            }
        }
        if tensor.dtype == .float32, let p = artifact.pointer(name) {
            accumulate(p.assumingMemoryBound(to: Float.self))          // read in place, without a copy
        } else {
            var widened = [Float](repeating: 0, count: inputs * outputs)
            try widened.withUnsafeMutableBufferPointer {
                _ = try artifact.materialize(name, into: $0)
                accumulate(UnsafePointer($0.baseAddress!))
            }
        }
        return (0..<count).map { Array(y[($0 * outputs)..<(($0 + 1) * outputs)]) }
    }

    /// The sinusoid of `Krea2TimestepEmbedding`, **computed in double then rounded once**.
    ///
    /// Its arguments climb to 1000 radians (`σ · 1000 · f`, f₀ = 1). The reference computes them in
    /// fp32: the rounded angle alone carries 3·10⁻⁵ rad near 1000, and torch's fp32 `exp`
    /// is not correctly rounded (15 frequencies out of 128 at one ulp). Its sinusoid is thus
    /// accurate only to 2·10⁻⁵, and `temb` — which cancels 6144 terms of 10⁻² for certain
    /// components of 10⁻⁵ — shows it. No fp32 reproduces it to the bit, not even
    /// diffusers on another device: the target is the **true** function, that of the fp64 reference.
    /// σ retraces the pipeline's path in fp32 — `t = σ·1000` (the scheduler), `t / 1000` (the
    /// pipeline) —: it is the input the reference receives.
    static func sinusoid(sigma: Float, dim: Int) -> [Float] {
        let half = dim / 2
        let t: Float = sigma * 1000, timestep = Double(t / 1000)
        var out = [Float](repeating: 0, count: dim)
        for j in 0..<half {
            let angle = timestep * 1000 * exp(-log(1e4) * Double(j) / Double(half))
            out[j] = Float(cos(angle))
            out[half + j] = Float(sin(angle))
        }
        return out
    }

    static func geluTanh(_ x: inout [Double]) {
        let c = (2 / Double.pi).squareRoot()
        for i in x.indices {
            let value = x[i]
            x[i] = 0.5 * value * (1 + tanh(c * (value + 0.044715 * value * value * value)))
        }
    }
}
