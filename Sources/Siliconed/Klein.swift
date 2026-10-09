import Metal
import Accelerate
import Foundation

/// The FLUX.2 [klein] 4B DiT — `Flux2Transformer2DModel`, 3.9 G parameters —, in fp32.
///
///     3 Qwen3-4B hidden states [512, 3 × 2560] ─ context_embedder ─ text [512, 3072] ─────────┐
///     latent [128, H/16, W/16] ─ tokens [S, 128] ─ x_embedder ─ image [S, 3072] ── concat ─────┤
///                                                                                            ▼
///     σ ─ sinusoid 256 (cos first) ─ MLP ─ temb ─ SiLU ─┬─ 3 modulations ─▶ 5 double blocks
///                                                           │                   20 single blocks
///                                                           └─ norm_out ◀──────── image only
///
/// **A double block** — two streams, one attention: the text and the image each have their own weights
/// (`add_*_proj`, `to_add_out`, `ff_context` for the text) and their modulation, and attend to
/// each other together over `[text, image]`.
///
///     n = LN(x)·(1 + scale) + shift          LN without affine, eps 1e-6, stream modulation
///     x += gate · Wo( SDPA(RoPE(n(q)), RoPE(n(k)), v) )
///     x += gate' · W₂( SiLU(W₁ᵍ n') ⊙ W₁ᵘ n' )
///
/// **A single block** — one sequence, a single set of weights, attention and MLP **in parallel**:
///
///     n = LN(x)·(1 + scale) + shift
///     x += gate · W_out [ SDPA(q, k, v) | SiLU(gate) ⊙ up ]      q, k, v, gate, up = W_in · n
///
/// What sets it apart from the other three targets, and what a port by analogy would miss:
///
///   - **no bias**, anywhere, and `LayerNorm`s without affine for the modulation (the QK-Norm
///     is an RMSNorm with an ordinary weight, not `1 + w`);
///   - the modulation is **shared by all the blocks**: three `Linear`s on `SiLU(temb)` per
///     evaluation (image, text, singles), plus that of `norm_out` — tabulated by σ (`tabulateModulation`);
///   - **the text keeps its 512 positions, padding included, and without mask**: `Flux2KleinPipeline`
///     pads to 512 and passes no mask to the DiT. The padding rows are visible
///     keys; removing them would change the sum (unlike Krea 2). See `KleinText`;
///   - the RoPE has **four** axes of 32 (`t, h, w, l`), θ = 2000, adjacent pairs: the image is
///     at `(0, h, w, 0)`, the text at `(0, 0, 0, l)` — **the text rotates**, on its fourth axis;
///   - the fused weights are **split by the forge**: `to_qkv_mlp_proj` into `q, k, v, gate, up`,
///     `ff.linear_in` into `gate, up`, a single block's `to_out` into `attn` and `mlp` (on its input,
///     hence the accumulating GEMM). Each GEMM thus writes a contiguous slice.
///
/// **The references (editing, FLUX.2's "Kontext")**: the tokens of encoded images follow
/// those of the generated image — `[text | image | ref. 1 | ref. 2 …]` —, go through the SAME weights
/// as the image (`x_embedder`, the image stream of the double blocks) and the same modulation, at RoPE
/// position `(10·i, h, w, 0)`. Only the generated rows come out. Nothing else changes: it is the
/// sequence that gets longer, not the model.
///
/// The `[text | image]` cut falls on a page: 512 × 3072 × 4 bytes = 384 pages. The text and
/// image GEMMs thus write into the same slice, at their row, without copying.
package final class KleinDiT {
    package struct Config {
        package let dim, heads, headDim, hidden, doubles, singles, channels, timeDim, context: Int
        package let axes: [Int]
        package let theta, eps: Double

        package init(header: [String: Any]) {
            let c = header["config"] as? [String: Any] ?? [:]
            heads = c["num_attention_heads"] as? Int ?? 24
            headDim = c["attention_head_dim"] as? Int ?? 128
            dim = heads * headDim
            hidden = Int(Double(dim) * (c["mlp_ratio"] as? Double ?? 3))
            doubles = c["num_layers"] as? Int ?? 5
            singles = c["num_single_layers"] as? Int ?? 20
            channels = c["in_channels"] as? Int ?? 128
            timeDim = c["timestep_guidance_channels"] as? Int ?? 256
            context = c["joint_attention_dim"] as? Int ?? 7680
            axes = c["axes_dims_rope"] as? [Int] ?? [32, 32, 32, 32]
            theta = c["rope_theta"] as? Double ?? 2000
            eps = c["eps"] as? Double ?? 1e-6
        }
        /// The modulation floats of an evaluation: 6 d (image) + 6 d (text) + 3 d (singles).
        package var modulation: Int { 15 * dim }
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
    private let elementwise: ElementwiseGPU
    /// The token grid (that of the latent: patch 1) and the fixed text length.
    /// `imageTokens`: the generated tokens; `fluxImage`: those of the image stream, references included.
    private let height, width, imageTokens, fluxImage, textRows, maxRows: Int
    private let references: [(height: Int, width: Int)]
    private var readyReferences: Bool
    private var attention: Attention?

    private let x, normed, q, k, v, merged, o, h1, h2, reserve, tokens, freqs: UnsafeMutablePointer<Float>
    /// The floats `reserve` holds (`d·h`, the largest weight): what `materialize` may write into it.
    private var reserveCapacity: Int { config.dim * config.hidden }
    /// `[text 512, 3072]`: the output of `context_embedder`, computed once (`prepareText`).
    private let text: UnsafeMutablePointer<Float>
    private var readyText = false
    /// The fifteen modulation vectors of an evaluation, where the GPU reads the gates.
    private let modulationSlot: UnsafeMutablePointer<Float>
    /// The reserved sizes, per address: a Metal wrapper is made at the size of the slice.
    private var reserved: [UnsafeMutablePointer<Float>: Int] = [:]
    /// **The LoRA stack** (`ForgeLoRA`, FLUX.2 [klein] family): `ΔW = B·A` per split slice,
    /// applied by `gemm.lora` — two thin products accumulated into the main GEMM's output,
    /// like Krea 2. Its expanded weights are kept from one evaluation to the next (`CacheLoRA`).
    package let lora: LoRA?
    private let cacheLoRA: CacheLoRA?
    private let loraMid: UnsafeMutablePointer<Float>?

    /// - Parameters latentHeight, latentWidth: the latent grid `[128, H/16, W/16]`, which is
    ///   that of the tokens.
    /// - Parameter references: the grids of the reference latents (editing), in order;
    ///   their values come from `prepareReferences`.
    package init(artifact: Artifact, latentHeight: Int, latentWidth: Int,
                references: [(height: Int, width: Int)] = [], textRows: Int = KleinText.length,
                freezeCut: Bool = EngineSettings.effective.frozenCut, lora: LoRA? = nil) throws {
        guard let kind = artifact.header["kind"] as? String, Family(ditKind: kind)?.isKlein == true else {
            throw Artifact.Failure.badHeader("map \(artifact.header["kind"] ?? "?"): a FLUX.2 [klein] DiT expected")
        }
        guard artifact.header["fused_split"] as? Bool == true, artifact.linearWeightsTransposed else {
            throw Artifact.Failure.badHeader("FLUX.2 map forged without split fused weights or transposed Linears")
        }
        self.artifact = artifact
        self.config = Config(header: artifact.header)
        self.prefetcher = Prefetcher(artifact: artifact)
        self.gemm = try GEMM(freezeCut: freezeCut)
        self.elementwise = try ElementwiseGPU(device: gemm.device, queue: gemm.queue)
        self.height = latentHeight
        self.width = latentWidth
        self.textRows = textRows
        imageTokens = latentHeight * latentWidth
        self.references = references
        readyReferences = references.isEmpty
        fluxImage = imageTokens + references.reduce(0) { $0 + $1.height * $1.width }
        maxRows = fluxImage + textRows
        guard textRows * config.dim % (Arena.alignment / 4) == 0,
              textRows * config.hidden % (Arena.alignment / 4) == 0 else {
            throw Artifact.Failure.badHeader("\(textRows) text rows: the text/image cut does not fall on a page")
        }

        let n = maxRows, d = config.dim, h = config.hidden
        let r = lora?.maxRank ?? 0
        let floats = 7 * n * d + 2 * n * h + d * h + textRows * d + fluxImage * config.channels
            + n * config.headDim + config.modulation + n * r
        let arena = try Arena(capacity: floats * 4 + 32 * Arena.alignment + (8 << 20))
        self.arena = arena
        var sizes: [UnsafeMutablePointer<Float>: Int] = [:]
        func slot(_ name: String, _ count: Int) throws -> UnsafeMutablePointer<Float> {
            let p = try arena.reserve(name, bytes: count * 4).assumingMemoryBound(to: Float.self)
            sizes[p] = count * 4
            return p
        }
        x = try slot("x", n * d); normed = try slot("normed", n * d)
        q = try slot("q", n * d); k = try slot("k", n * d); v = try slot("v", n * d)
        merged = try slot("merged", n * d); o = try slot("o", n * d)
        h1 = try slot("h1", n * h); h2 = try slot("h2", n * h)
        reserve = try slot("reserve", d * h)
        text = try slot("text", textRows * d)
        tokens = try slot("tokens", fluxImage * config.channels)
        freqs = try slot("freqs", n * config.headDim)
        modulationSlot = try slot("modulation", config.modulation)
        self.lora = r > 0 ? lora : nil
        if let lora, r > 0 {
            loraMid = try slot("loraMid", n * r)
            cacheLoRA = try CacheLoRA(lora: lora)
        } else {
            loraMid = nil; cacheLoRA = nil
        }
        reserved = sizes
        KleinRope.write(into: freqs, textRows: textRows, tilesHigh: height, tilesWide: width,
                        references: references, axes: config.axes, theta: config.theta)
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

    /// **The text, once per DiT**: `context_embedder` over the 512 rows `[512, 7680]`. Neither σ
    /// nor the latent enter into it; each evaluation copies the result at the head of the sequence.
    package func prepareText(_ hiddenInput: UnsafePointer<Float>) throws {
        let t = textRows, d = config.dim
        // Through `normed`: it is a slice of the wanted width (7680 ≤ n·d), at a page address.
        (UnsafeMutablePointer(mutating: normed)).update(from: hiddenInput, count: t * config.context)
        try linearGPU("context_embedder.weight", a: normed, into: text, m: t, k: config.context, n: d)
        record("context_embedder_out", text, t * d)
        readyText = true
    }

    /// **The references, once per DiT**: each latent `[128, h, w]` (encoded and normalized by
    /// `Flux2EncodingModule`) becomes its tokens, laid out after those of the generated image. Constant from one step
    /// to the next: neither σ nor the current latent enter into them.
    package func prepareReferences(_ latents: [[Float]]) throws {
        guard latents.count == references.count,
              zip(latents, references).allSatisfy({ $0.count == config.channels * $1.height * $1.width }) else {
            throw Artifact.Failure.badHeader("FLUX.2: \(latents.count) references for \(references.count) announced grids")
        }
        var rowLine = imageTokens
        for (values, grid) in zip(latents, references) {
            let n = grid.height * grid.width
            values.withUnsafeBufferPointer {
                vDSP_mtrans($0.baseAddress!, 1, tokens + rowLine * config.channels, 1, vDSP_Length(n), vDSP_Length(config.channels))
            }
            rowLine += n
        }
        readyReferences = true
    }

    /// `latent` as `[128, h, w]`, `sigma` as the scheduler holds it. Returns the velocity `[128, h, w]`.
    package func forward(latent: UnsafePointer<Float>, sigma: Float) throws -> [Float] {
        guard readyText, readyReferences else {
            throw Artifact.Failure.badHeader("FLUX.2: `prepareText` (and `prepareReferences`) before `forward`")
        }
        let d = config.dim, c = config.channels, t = textRows, s = imageTokens, n = maxRows

        let tabulated = modulations[sigma.bitPattern]
        prefetcher.request(["x_embedder.weight"])
        prefetcher.request(prefetcher.namesOfBlock(prefix: "transformer_blocks.0."))
        let m = try tabulated ?? timed("modulation") { try modulation(sigmas: [sigma])[0] }
        m.withUnsafeBufferPointer { modulationSlot.update(from: $0.baseAddress!, count: config.modulation) }
        record("mod_img", modulationSlot, 6 * d)
        record("mod_txt", modulationSlot + 6 * d, 6 * d)
        record("mod_single", modulationSlot + 12 * d, 3 * d)

        // ── the sequence: [text T, image S] ─────────────────────────────────────────────────
        x.update(from: text, count: t * d)
        vDSP_mtrans(latent, 1, tokens, 1, vDSP_Length(s), vDSP_Length(c))   // [C, S] → [S, C]
        // The references follow, already as tokens: a single GEMM for the whole image stream.
        try linearGPU("x_embedder.weight", a: tokens, into: x + t * d, m: fluxImage, k: c, n: d)
        record("x_embedder_out", x + t * d, s * d)

        for layer in 0..<config.doubles {
            try cancellation.check()   // between two layers, never inside one (see `Cancellation`)
            prefetcher.request(prefetcher.namesOfBlock(prefix: layer + 1 < config.doubles
                ? "transformer_blocks.\(layer + 1)." : "single_transformer_blocks.0."))
            try double(layer)
            if layer == 0 || layer == config.doubles - 1 {
                let tag = layer == 0 ? "double0" : "double_last"
                record(tag + "_txt", x, t * d); record(tag + "_img", x + t * d, s * d)
            }
        }
        for layer in 0..<config.singles {
            try cancellation.check()
            for ahead in 1...2 where layer + ahead < config.singles {
                prefetcher.request(prefetcher.namesOfBlock(prefix: "single_transformer_blocks.\(layer + ahead)."))
            }
            if layer == config.singles - 1 { prefetcher.request(["norm_out.linear.weight", "proj_out.weight"]) }
            try single(layer)
            if layer == 0 { record("single0_out", x, n * d) }
            if layer == config.singles - 1 { record("single_last_out", x, n * d) }
        }

        // ── norm_out, on the image only: (scale, shift) — diffusers' order, not BFL's
        let end = m.withUnsafeBufferPointer { Array($0[config.modulation...]) }
        end.withUnsafeBufferPointer { f in
            timed("norm") {
                Ops.layerNorm(x + t * d, into: normed, rows: s, columns: d, eps: Float(config.eps))
                Krea2Ops.modulate(normed, scale: f.baseAddress!, shift: f.baseAddress! + d, rows: s, columns: d)
            }
        }
        try linearGPU("proj_out.weight", a: normed, into: o, m: s, k: d, n: c)
        var out = [Float](repeating: 0, count: c * s)
        out.withUnsafeMutableBufferPointer { vDSP_mtrans(o, 1, $0.baseAddress!, 1, vDSP_Length(c), vDSP_Length(s)) }
        record("model_out", out, out.count)
        return out
    }

    // ── the modulation ──────────────────────────────────────────────────────────────────

    /// The bits of σ → the modulations of an evaluation: `[image 6d | text 6d | singles 3d |
    /// norm_out 2d]`. Neither the latent nor the text enter into it.
    package typealias ModulationTable = [UInt32: [Float]]
    package var modulations: ModulationTable = [:]

    /// **Precomputes the modulation of the σ that will be evaluated** — each modulation weight read
    /// once for the whole render. Same accumulation as an isolated pass (`Krea2Exact.linears`):
    /// the table gives the same bits as the per-step computation.
    package func tabulateModulation(sigmas: [Float]) throws {
        let newKeys = Array(Set(sigmas.map(\.bitPattern)).subtracting(modulations.keys)).sorted()
        guard !newKeys.isEmpty else { return }
        let values = try timed("modulation") { try modulation(sigmas: newKeys.map(Float.init(bitPattern:))) }
        for (key, value) in zip(newKeys, values) { modulations[key] = value }
    }

    /// `temb`, then the four `Linear`s on `SiLU(temb)`, **in double**, each weight read once
    /// for all the σ.
    private func modulation(sigmas: [Float]) throws -> [[Float]] {
        let sinusoids = sigmas.map { KleinDiT.sinusoid(sigma: $0, dim: config.timeDim) }
        var hiddenValues = try Krea2Exact.linears(artifact, "time_guidance_embed.timestep_embedder.linear_1", xs: sinusoids)
        for i in hiddenValues.indices { KleinDiT.silu(&hiddenValues[i]) }
        var tembs = try Krea2Exact.linears(artifact, "time_guidance_embed.timestep_embedder.linear_2", xs: hiddenValues)
        if recordBoundaries, let first = tembs.first { record("temb", first.map(Float.init), first.count) }
        for i in tembs.indices { KleinDiT.silu(&tembs[i]) }
        let parts = try ["double_stream_modulation_img", "double_stream_modulation_txt",
                           "single_stream_modulation", "norm_out"].map {
            try Krea2Exact.linears(artifact, $0 + ".linear", xs: tembs)
        }
        return sigmas.indices.map { i in parts.flatMap { $0[i] }.map(Float.init) }
    }

    /// **The sinusoid of `Timesteps(256, flip_sin_to_cos=True)`, computed in double** on the
    /// timestep the reference receives: `t = σ·1000` (the scheduler), `t / 1000` (the pipeline),
    /// `× 1000` (the DiT), each in fp32. Same reason as `Krea2Exact.sinusoid`: the angles
    /// climb to 1000 rad, where an fp32 computation is accurate only to 10⁻⁵.
    package static func sinusoid(sigma: Float, dim: Int) -> [Double] {
        let half = dim / 2
        let t: Float = sigma * 1000, pipeline: Float = t / 1000, entry = Double(pipeline * 1000)
        var out = [Double](repeating: 0, count: dim)
        for j in 0..<half {
            let angle = entry * exp(-log(10000) * Double(j) / Double(half))
            out[j] = Double(Float(cos(angle)))
            out[half + j] = Double(Float(sin(angle)))
        }
        return out
    }

    static func silu(_ x: inout [Double]) {
        for i in x.indices { x[i] = x[i] / (1 + exp(-x[i])) }
    }

    // ── the blocks ──────────────────────────────────────────────────────────────────────

    /// A double block: text in rows `[0, T)`, image in `[T, n)`.
    private func double(_ layer: Int) throws {
        // `s`: the whole image stream — the references take the image's weights and modulation.
        let d = config.dim, hh = config.hidden, t = textRows, s = fluxImage, n = maxRows
        let p = "transformer_blocks.\(layer)."
        let img = modulationSlot, txt = modulationSlot + 6 * d
        let mb = try slice(modulationSlot)
        let xb = try slice(x), ob = try slice(o)
        let (xi, oi) = (t * d, t * d)

        // ── attention ─────────────────────────────────────────────────────────────────────
        try modulatedNorm(x, rows: t, into: normed, scale: txt + d, shift: txt)
        try modulatedNorm(x + t * d, rows: s, into: normed + t * d, scale: img + d, shift: img)
        for (image, text, output) in [("to_q", "add_q_proj", q), ("to_k", "add_k_proj", k), ("to_v", "add_v_proj", v)] {
            try linearGPU(p + "attn.\(text).weight", a: normed, into: output, m: t, k: d, n: d)
            try linearGPU(p + "attn.\(image).weight", a: normed + t * d, into: output + t * d, m: s, k: d, n: d)
        }
        try headNorm(p + "attn.norm_added_q.weight", q, rows: t)
        try headNorm(p + "attn.norm_added_k.weight", k, rows: t)
        try headNorm(p + "attn.norm_q.weight", q + t * d, rows: s)
        try headNorm(p + "attn.norm_k.weight", k + t * d, rows: s)
        try rope(rows: n)
        try attend(rows: n)
        try linearGPU(p + "attn.to_add_out.weight", a: merged, into: o, m: t, k: d, n: d)
        try linearGPU(p + "attn.to_out.0.weight", a: merged + t * d, into: o + t * d, m: s, k: d, n: d)
        if layer == 0 { record("double0_attn_img", o + t * d, imageTokens * d) }   // the output of `Flux2Attention`, after `to_out`
        try residuals(xb, ob, t: t, s: s, carriesText: (mb, 6 * d + 2 * d), carriesImage: (mb, 2 * d), xi: xi, oi: oi)

        // ── SwiGLU ────────────────────────────────────────────────────────────────────────
        try modulatedNorm(x, rows: t, into: normed, scale: txt + 4 * d, shift: txt + 3 * d)
        try modulatedNorm(x + t * d, rows: s, into: normed + t * d, scale: img + 4 * d, shift: img + 3 * d)
        try linearGPU(p + "ff_context.linear_in.gate.weight", a: normed, into: h1, m: t, k: d, n: hh)
        try linearGPU(p + "ff_context.linear_in.up.weight", a: normed, into: h2, m: t, k: d, n: hh)
        try linearGPU(p + "ff.linear_in.gate.weight", a: normed + t * d, into: h1 + t * hh, m: s, k: d, n: hh)
        try linearGPU(p + "ff.linear_in.up.weight", a: normed + t * d, into: h2 + t * hh, m: s, k: d, n: hh)
        let (h1b, h2b) = (try slice(h1), try slice(h2))
        timed("swiglu") { timings["elementwise GPU", default: 0] += elementwise.swiglu((h1b, 0), (h2b, 0), count: n * hh) }
        try linearGPU(p + "ff_context.linear_out.weight", a: h1, into: o, m: t, k: hh, n: d)
        try linearGPU(p + "ff.linear_out.weight", a: h1 + t * hh, into: o + t * d, m: s, k: hh, n: d)
        try residuals(xb, ob, t: t, s: s, carriesText: (mb, 6 * d + 5 * d), carriesImage: (mb, 5 * d), xi: xi, oi: oi)
    }

    /// The two residuals of a double block, each stream under its gate.
    private func residuals(_ xb: MTLBuffer, _ ob: MTLBuffer, t: Int, s: Int, carriesText: ElementwiseGPU.BufferSlice,
                         carriesImage: ElementwiseGPU.BufferSlice, xi: Int, oi: Int) throws {
        let d = config.dim
        timed("residual") {
            timings["elementwise GPU", default: 0] += elementwise.residual((xb, 0), plus: (ob, 0), carries: carriesText,
                                                                          rows: t, columns: d)
            timings["elementwise GPU", default: 0] += elementwise.residual((xb, xi), plus: (ob, oi), carries: carriesImage,
                                                                          rows: s, columns: d)
        }
    }

    /// A single block, over the whole sequence.
    private func single(_ layer: Int) throws {
        let d = config.dim, hh = config.hidden, n = maxRows
        let p = "single_transformer_blocks.\(layer)."
        let m = modulationSlot + 12 * d
        try modulatedNorm(x, rows: n, into: normed, scale: m + d, shift: m)
        for (part, output, width) in [("q", q, d), ("k", k, d), ("v", v, d), ("gate", h1, hh), ("up", h2, hh)] {
            try linearGPU(p + "attn.to_qkv_mlp_proj.\(part).weight", a: normed, into: output, m: n, k: d, n: width)
        }
        try headNorm(p + "attn.norm_q.weight", q, rows: n)
        try headNorm(p + "attn.norm_k.weight", k, rows: n)
        try rope(rows: n)
        try attend(rows: n)
        let (h1b, h2b) = (try slice(h1), try slice(h2))
        timed("swiglu") { timings["elementwise GPU", default: 0] += elementwise.swiglu((h1b, 0), (h2b, 0), count: n * hh) }
        try linearGPU(p + "attn.to_out.attn.weight", a: merged, into: o, m: n, k: d, n: d)
        try linearGPU(p + "attn.to_out.mlp.weight", a: h1, into: o, m: n, k: hh, n: d, accumulate: true)
        let (xb, ob, mb) = (try slice(x), try slice(o), try slice(modulationSlot))
        timed("residual") {
            timings["elementwise GPU", default: 0] += elementwise.residual((xb, 0), plus: (ob, 0), carries: (mb, 12 * d + 2 * d),
                                                                          rows: n, columns: d)
        }
    }

    // ── plumbing ────────────────────────────────────────────────────────────────────────

    /// A slice of the arena seen from the GPU, wrapped **to the end of its reservation** — including
    /// from an interior row (the text/image cut). Wrappers are memoized
    /// by address: a size that changed from one call to the next would redo them at every block.
    private func slice(_ p: UnsafeMutablePointer<Float>) throws -> MTLBuffer {
        guard let (begin, bytes) = reserved.first(where: { $0.key <= p && p < $0.key + $0.value / 4 }) else {
            throw Artifact.Failure.badHeader("FLUX.2: an address outside the arena")
        }
        return try gemm.wrap(UnsafeMutableRawPointer(p), bytes: bytes - (p - begin) * 4, name: "slice")
    }

    /// `destination = LN(source) · (1 + scale) + shift` — LayerNorm without affine, eps 1e-6, on the CPU.
    private func modulatedNorm(_ source: UnsafePointer<Float>, rows: Int, into destination: UnsafeMutablePointer<Float>,
                               scale: UnsafePointer<Float>, shift: UnsafePointer<Float>) throws {
        let d = config.dim
        timed("norm") {
            Ops.layerNorm(source, into: destination, rows: rows, columns: d, eps: Float(config.eps))
            Krea2Ops.modulate(destination, scale: scale, shift: shift, rows: rows, columns: d)
        }
    }

    /// QK-Norm: RMSNorm over `head_dim`, eps 1e-6, ordinary weight.
    private func headNorm(_ name: String, _ tensor: UnsafeMutablePointer<Float>, rows: Int) throws {
        try artifact.materialize(name, into: reserve, capacity: reserveCapacity)
        timed("norm") { Ops.rmsNorm(tensor, weight: reserve, into: tensor, rows: rows * config.heads,
                                    columns: config.headDim, eps: Float(config.eps)) }
    }

    private func rope(rows n: Int) throws {
        timed("rope") {
            Ops.rope(q, freqs: freqs, sequence: n, heads: config.heads, headDim: config.headDim)
            Ops.rope(k, freqs: freqs, sequence: n, heads: config.heads, headDim: config.headDim)
        }
    }

    private func attend(rows n: Int) throws {
        let attention: Attention
        if let a = self.attention { attention = a } else {
            attention = Attention(device: gemm.device, queue: gemm.queue, heads: config.heads,
                                  sequence: n, headDim: config.headDim)
            self.attention = attention
        }
        let (qb, kb, vb, ob) = (try slice(q), try slice(k), try slice(v), try slice(merged))
        timed("sdpa wall") { timings["sdpa GPU", default: 0] += attention.run(q: qb, k: kb, v: vb, into: ob) }
    }

    /// `C[m, n] = A[m, k] · W` (or `C +=`), W laid out `[k, n]` by the forge. `a` and `c` may be
    /// an interior row of a slice (the text/image cut): it falls on a page.
    private func linearGPU(_ name: String, a: UnsafeMutablePointer<Float>, into c: UnsafeMutablePointer<Float>,
                           m: Int, k: Int, n: Int, accumulate: Bool = false) throws {
        let got = try timed("widening") { try artifact.materialize(name, into: reserve, capacity: reserveCapacity) }
        guard got == k * n else {
            throw Artifact.Failure.badHeader("\(name): \(got) values, \(k * n) expected")
        }
        let (ab, wb, cb) = (try slice(a), try slice(reserve), try slice(c))
        timed("gemm wall") {
            timings["gemm GPU", default: 0] += accumulate
                ? gemm.accumulatedLinear(a: ab, b: wb, c: cb, m: m, k: k, n: n)
                : gemm.linear(a: ab, b: wb, c: cb, m: m, k: k, n: n, weightIsTransposed: true)
        }
        // The stack, accumulated into the same output — after the main GEMM, whether it writes or accumulates.
        let target = String(name.dropLast(".weight".count))
        if let cache = cacheLoRA, let mid = loraMid,
           let stack = try timed("lora weights", { try cache.module(target, k: k, n: n) }) {
            let down = try gemm.wrap(UnsafeMutableRawPointer(stack.down), bytes: k * stack.rank * 4, name: "loraBas")
            let up = try gemm.wrap(UnsafeMutableRawPointer(stack.up), bytes: stack.rank * n * 4, name: "loraHaut")
            let mb = try slice(mid)
            timed("lora wall") {
                timings["lora GPU", default: 0] += gemm.lora(x: ab, down: down, mid: mb, up: up, c: cb,
                                                             m: m, k: k, r: stack.rank, n: n)
            }
        }
    }
}

/// The FLUX.2 RoPE: `Flux2PosEmbed`, **four** axes `(t, h, w, l)` of 32 dimensions, θ = 2000,
/// adjacent pairs (`repeat_interleave_real=True`) — the `[N, 64, (cos, sin)]` table of `Ops.rope`.
/// The angles in double, like the reference (`freqs_dtype = float64`). The text is at
/// `(0, 0, 0, l)` — it rotates on its fourth axis —, the image at `(0, h, w, 0)`.
package enum KleinRope {
    /// The references `i = 1, 2…` are at `(10·i, h, w, 0)` (`_prepare_image_ids`, `scale = 10`).
    package static func write(into out: UnsafeMutablePointer<Float>, textRows: Int, tilesHigh: Int,
                             tilesWide: Int, references: [(height: Int, width: Int)] = [],
                             axes: [Int], theta: Double) {
        let pairs = axes.reduce(0, +) / 2
        let f = axes.map { dims in (0..<(dims / 2)).map { 1 / pow(theta, Double(2 * $0) / Double(dims)) } }
        func write(_ row: Int, _ position: [Int]) {
            let token = out + row * pairs * 2
            var index = 0
            for (axis, p) in position.enumerated() {
                for frequency in f[axis] {
                    let angle = Double(p) * frequency
                    token[2 * index] = Float(cos(angle)); token[2 * index + 1] = Float(sin(angle))
                    index += 1
                }
            }
        }
        for l in 0..<textRows { write(l, [0, 0, 0, l]) }
        for h in 0..<tilesHigh {
            for w in 0..<tilesWide { write(textRows + h * tilesWide + w, [0, h, w, 0]) }
        }
        var rowLine = textRows + tilesHigh * tilesWide
        for (i, r) in references.enumerated() {
            for h in 0..<r.height {
                for w in 0..<r.width { write(rowLine, [10 * (i + 1), h, w, 0]); rowLine += 1 }
            }
        }
    }
}

/// **The FLUX.2 [klein] text**, as `Flux2KleinPipeline._get_qwen3_prompt_embeds` prepares it.
///
///     chat template (no thinking) ─ tokenizer, padded to 512 on the right ─ Qwen3-4B
///       ─ hidden_states[9, 18, 27] (each layer's INPUT) ─ [512, 3 × 2560]
///
/// **The padding is computed, not removed**: the DiT receives the 512 rows without mask. Its
/// rows are those of `transformers` under the mask `causal ∧ attention_mask` — a padding
/// row, at its position `i`, attends only to the real tokens (`TextEncoder(visibleKeys:)`).
/// The encoder is Z-Image's, to the bit: same weights (checked tensor by tensor), same map.
package enum KleinText {
    package static let prefix = "<|im_start|>user\n"
    /// `apply_chat_template(add_generation_prompt=True, enable_thinking=False)`.
    package static let suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    package static let length = 512
    package static let sockets = [9, 18, 27]

    /// The 512 identifiers (truncated then padded on the right) and the number of real tokens.
    package static func identifiers(_ prompt: String, tokenizer: Tokenizer) -> (ids: [Int], realCount: Int) {
        let (ids, mask) = Tokenizer.pad(tokenizer.encode(prefix + prompt + suffix), maxLength: length)
        return (ids, mask.reduce(0, +))
    }

    /// `[512, 3 × 2560]`.
    package static func hiddenStates(_ ids: [Int], realCount: Int, encoder: String,
                                   freezeCut: Bool = EngineSettings.effective.frozenCut,
                                   cancellation: Cancellation? = nil) throws -> [Float] {
        let encoder = try TextEncoder(artifact: try Artifact(path: encoder), sequence: ids.count,
                                      freezeCut: freezeCut, visibleKeys: realCount)
        encoder.cancellation = cancellation
        return try encoder.encodeTaps(ids: ids, taps: sockets)
    }
}

extension KleinDiT {
    /// **The FLUX.2 [klein] schedule**: `sigmas = linspace(1, 1/N, N)`, shifted by the
    /// exponential shift `σ' = e^μ / (e^μ + (1/σ − 1))` at a μ **that depends on the grid and the
    /// number of steps** (`compute_empirical_mu`), then a zero terminal σ.
    package static func sigmas(steps: Int, imageTokens: Int) -> [Float] {
        let mu = empiricalMu(imageTokens: imageTokens, steps: steps)
        return (0..<steps).map { i -> Float in
            let s = steps == 1 ? 1 : 1 - Double(i) * (1 - 1 / Double(steps)) / Double(steps - 1)
            return Float(exp(mu) / (exp(mu) + (1 / s - 1)))
        } + [0]
    }

    /// `compute_empirical_mu`: two lines in the sequence length, interpolated over the steps
    /// under 4,300 tokens; beyond, the second alone.
    package static func empiricalMu(imageTokens: Int, steps: Int) -> Double {
        let (a1, b1) = (8.73809524e-05, 1.89833333), (a2, b2) = (0.00016927, 0.45666666)
        let s = Double(imageTokens)
        if imageTokens > 4300 { return a2 * s + b2 }
        let m200 = a2 * s + b2, m10 = a1 * s + b1
        let a = (m200 - m10) / 190, b = m200 - 200 * a
        return a * Double(steps) + b
    }
}
