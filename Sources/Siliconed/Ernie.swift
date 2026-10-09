import Metal
import Accelerate
import Foundation

/// The ERNIE-Image Turbo DiT — `ErnieImageTransformer2DModel`, 7.9 G parameters —, in fp32.
///
///     latent [128, H/16, W/16] ─ tokens [S, 128] ─ x_embedder (+ bias) ─ image [S, 4096] ──┐
///     Ministral-3 hidden_states[-2] [T, 3072] ─ text_proj ─ text [T, 4096] ─── concat ────┤
///                                                                                          ▼
///     σ ─ sinusoid 4096 (sin first) ─ MLP ─ c ─┬─ SiLU ─ adaLN_modulation ─▶ 36 layers
///                                                 └─ final_norm.linear ◀────── image only
///
/// **One layer** — a single `[image | text]` sequence, a single set of weights, in series:
///
///     n = RMS(x)·(1 + scale) + shift               weighted RMSNorm, eps 1e-6
///     x += gate · Wo( SDPA(RoPE(RMS(q)), RoPE(RMS(k)), v) )
///     n' = RMS'(x)·(1 + scale') + shift'
///     x += gate' · W₂( up(n') ⊙ GELU(gate(n')) )   exact GELU (`erf`), the gate under GELU
///
/// What sets it apart from the other targets, and what a port by analogy would miss:
///
///   - **the image first, the text after** in the sequence (FLUX.2 does the opposite), and the text
///     is **not padded**: the pipeline passes its `T` real tokens, without useful mask;
///   - the modulation is **a single one**, shared by the 36 layers: `Linear(SiLU(c))` into six
///     vectors `(shift, scale, gate)` × (attention, MLP) — but `final_norm` reads `c` **without** SiLU,
///     and returns `(scale, shift)` in that order;
///   - the RoPE has three axes `(32, 48, 48)`, θ = 256: the image at `(T, y, x)` — the text
///     length on the first axis —, the text at `(t, 0, 0)`. Its angles are **repeated in pairs**
///     (`[θ₀, θ₀, θ₁, θ₁, …]`) but applied by `rotate_half` (`j` with `j + 64`): it is neither
///     the interleaved RoPE nor the split one, and the pair `(j, j + 64)` rotates by **two different
///     angles**. See `ErnieRope`;
///   - biases on `x_embedder` (a 1×1 convolution) and `final_linear`, nowhere else.
///
/// `d = 4096` floats make a 16 KB page: the image/text cut falls on a page whatever
/// the text length, and the GEMMs write each part at its row without copying.
package final class ErnieDiT {
    package struct Config {
        package let dim, heads, headDim, hidden, layers, channels, context: Int
        package let axes: [Int]
        package let theta, eps: Double

        package init(header: [String: Any]) {
            let c = header["config"] as? [String: Any] ?? [:]
            dim = c["hidden_size"] as? Int ?? 4096
            heads = c["num_attention_heads"] as? Int ?? 32
            headDim = dim / heads
            hidden = c["ffn_hidden_size"] as? Int ?? 12288
            layers = c["num_layers"] as? Int ?? 36
            channels = c["in_channels"] as? Int ?? 128
            context = c["text_in_dim"] as? Int ?? 3072
            axes = c["rope_axes_dim"] as? [Int] ?? [32, 48, 48]
            theta = (c["rope_theta"] as? Double) ?? (c["rope_theta"] as? Int).map(Double.init) ?? 256
            eps = c["eps"] as? Double ?? 1e-6
        }
        /// The modulation floats of an evaluation: 6 d (the layers) + 2 d (`final_norm`).
        package var modulation: Int { 8 * dim }
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
    private let gelu: AnimaGELU
    /// The tokens of the image (the latent grid: patch 1), of the text, and of the sequence.
    private let imageTokens, textRows, maxRows: Int
    private var attention: Attention?

    private let x, normed, q, k, v, merged, o, h1, h2, reserve, tokens, freqs: UnsafeMutablePointer<Float>
    /// The floats `reserve` holds (`d·h`, the largest weight): what `materialize` may write into it.
    private var reserveCapacity: Int { config.dim * config.hidden }
    /// `[T, 4096]`: the output of `text_proj`, computed once (`prepareText`).
    private let text: UnsafeMutablePointer<Float>
    private var readyText = false
    /// The eight modulation vectors of an evaluation, where the GPU reads the gates.
    private let modulationSlot: UnsafeMutablePointer<Float>
    /// The reserved sizes, per address: a Metal wrapper is made at the size of the slice.
    private var reserved: [UnsafeMutablePointer<Float>: Int] = [:]
    /// **The LoRA stack** (`ForgeLoRA`, ERNIE-Image family): `ΔW = B·A` per `Linear`, applied
    /// by `gemm.lora` — two thin products accumulated into the main GEMM's output, like
    /// Krea 2 and FLUX.2 [klein]. Its expanded weights are kept from one evaluation to the next (`CacheLoRA`).
    package let lora: LoRA?
    private let cacheLoRA: CacheLoRA?
    private let loraMid: UnsafeMutablePointer<Float>?

    /// - Parameters latentHeight, latentWidth: the latent grid `[128, H/16, W/16]`, which is
    ///   that of the tokens.
    /// - Parameter textRows: the real text tokens — `hidden_states[-2]` is not padded.
    package init(artifact: Artifact, latentHeight: Int, latentWidth: Int, textRows: Int,
                freezeCut: Bool = EngineSettings.effective.frozenCut, lora: LoRA? = nil) throws {
        guard let kind = artifact.header["kind"] as? String, Family(ditKind: kind) == .ernie else {
            throw Artifact.Failure.badHeader("map \(artifact.header["kind"] ?? "?"): an ERNIE-Image DiT expected")
        }
        guard artifact.linearWeightsTransposed else {
            throw Artifact.Failure.badHeader("ERNIE-Image map forged without transposed Linears")
        }
        self.artifact = artifact
        self.config = Config(header: artifact.header)
        self.prefetcher = Prefetcher(artifact: artifact)
        self.gemm = try GEMM(freezeCut: freezeCut)
        self.elementwise = try ElementwiseGPU(device: gemm.device, queue: gemm.queue)
        self.gelu = try AnimaGELU(device: gemm.device, queue: gemm.queue)
        self.textRows = textRows
        imageTokens = latentHeight * latentWidth
        maxRows = imageTokens + textRows
        guard textRows > 0, config.dim % (Arena.alignment / 4) == 0, config.hidden % (Arena.alignment / 4) == 0 else {
            throw Artifact.Failure.badHeader("ERNIE-Image: \(textRows) text tokens, or a width that does not fall on a page")
        }

        let n = maxRows, d = config.dim, h = config.hidden
        let r = lora?.maxRank ?? 0
        let floats = 7 * n * d + 2 * n * h + d * h + textRows * d + imageTokens * config.channels
            + n * 2 * config.headDim + config.modulation + n * r
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
        tokens = try slot("tokens", imageTokens * config.channels)
        freqs = try slot("freqs", n * 2 * config.headDim)
        modulationSlot = try slot("modulation", config.modulation)
        self.lora = r > 0 ? lora : nil
        if let lora, r > 0 {
            loraMid = try slot("loraMid", n * r)
            cacheLoRA = try CacheLoRA(lora: lora)
        } else {
            loraMid = nil; cacheLoRA = nil
        }
        reserved = sizes
        ErnieRope.write(into: freqs, textRows: textRows, tilesHigh: latentHeight, tilesWide: latentWidth,
                        axes: config.axes, theta: Float(config.theta))
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

    /// **The text, once per DiT**: `text_proj` over the `T` rows `[T, 3072]`. Neither σ nor the
    /// latent enter into it; each evaluation copies the result after the image.
    package func prepareText(_ hiddenInput: UnsafePointer<Float>) throws {
        let t = textRows, d = config.dim
        // Through `normed`: a wide enough slice (3072 ≤ n·d), at a page address.
        normed.update(from: hiddenInput, count: t * config.context)
        try linearGPU("text_proj.weight", a: normed, into: text, m: t, k: config.context, n: d)
        record("text_proj_out", text, t * d)
        readyText = true
    }

    /// `latent` as `[128, h, w]`, `sigma` as the scheduler holds it. Returns the velocity `[128, h, w]`.
    package func forward(latent: UnsafePointer<Float>, sigma: Float) throws -> [Float] {
        guard readyText else { throw Artifact.Failure.badHeader("ERNIE-Image: `prepareText` before `forward`") }
        let d = config.dim, c = config.channels, t = textRows, s = imageTokens, n = maxRows

        let tabulated = modulations[sigma.bitPattern]
        prefetcher.request(["x_embedder.proj.weight"])
        prefetcher.request(prefetcher.namesOfBlock(prefix: "layers.0."))
        let m = try tabulated ?? timed("modulation") { try modulation(sigmas: [sigma])[0] }
        m.withUnsafeBufferPointer { modulationSlot.update(from: $0.baseAddress!, count: config.modulation) }
        record("mod", modulationSlot, 6 * d)
        record("mod_final", modulationSlot + 6 * d, 2 * d)

        // ── the sequence: [image S, text T] ─────────────────────────────────────────────────
        vDSP_mtrans(latent, 1, tokens, 1, vDSP_Length(s), vDSP_Length(c))   // [C, S] → [S, C]
        try linearGPU("x_embedder.proj.weight", a: tokens, into: x, m: s, k: c, n: d)
        try addBias("x_embedder.proj.bias", at: x, rows: s)
        record("x_embedder_out", x, s * d)
        (x + s * d).update(from: text, count: t * d)

        for layer in 0..<config.layers {
            try cancellation.check()   // between two layers, never inside one (see `Cancellation`)
            for ahead in 1...2 where layer + ahead < config.layers {
                prefetcher.request(prefetcher.namesOfBlock(prefix: "layers.\(layer + ahead)."))
            }
            if layer == config.layers - 1 {
                prefetcher.request(["final_linear.weight", "final_linear.bias"])
            }
            try layerPass(layer)
            if layer == 0 { record("layer0_out", x, n * d) }
            if layer == 1 { record("layer1_out", x, n * d) }
            if layer == config.layers - 1 { record("layer_last_out", x, n * d) }
        }

        // ── final_norm, on the image only: LN without affine, (scale, shift) ───────────────
        let end = modulationSlot + 6 * d
        timed("norm") {
            Ops.layerNorm(x, into: normed, rows: s, columns: d, eps: Float(config.eps))
            Krea2Ops.modulate(normed, scale: end, shift: end + d, rows: s, columns: d)
        }
        try linearGPU("final_linear.weight", a: normed, into: o, m: s, k: d, n: c)
        try addBias("final_linear.bias", at: o, rows: s, columns: c)
        var out = [Float](repeating: 0, count: c * s)
        out.withUnsafeMutableBufferPointer { vDSP_mtrans(o, 1, $0.baseAddress!, 1, vDSP_Length(c), vDSP_Length(s)) }
        record("model_out", out, out.count)
        return out
    }

    // ── the modulation ──────────────────────────────────────────────────────────────────

    /// The bits of σ → the modulation of an evaluation: `[layers 6d | final_norm 2d]`.
    package typealias ModulationTable = [UInt32: [Float]]
    package var modulations: ModulationTable = [:]

    /// **Precomputes the modulation of the σ that will be evaluated** — each modulation weight read
    /// once for the whole render, with the same accumulation as an isolated pass.
    package func tabulateModulation(sigmas: [Float]) throws {
        let newKeys = Array(Set(sigmas.map(\.bitPattern)).subtracting(modulations.keys)).sorted()
        guard !newKeys.isEmpty else { return }
        let values = try timed("modulation") { try modulation(sigmas: newKeys.map(Float.init(bitPattern:))) }
        for (key, value) in zip(newKeys, values) { modulations[key] = value }
    }

    /// `c = time_embedding(sinusoid)`, then `adaLN_modulation(SiLU(c))` and `final_norm.linear(c)`,
    /// **in double**, each weight read once for all the σ.
    private func modulation(sigmas: [Float]) throws -> [[Float]] {
        let sinusoids = sigmas.map { ErnieDiT.sinusoid(sigma: $0, dim: config.dim) }
        var hiddenValues = try Krea2Exact.linears(artifact, "time_embedding.linear_1", xs: sinusoids)
        for i in hiddenValues.indices { KleinDiT.silu(&hiddenValues[i]) }
        let cs = try Krea2Exact.linears(artifact, "time_embedding.linear_2", xs: hiddenValues)
        if recordBoundaries, let first = cs.first { record("temb", first.map(Float.init), first.count) }
        var enabled = cs
        for i in enabled.indices { KleinDiT.silu(&enabled[i]) }
        let layers = try Krea2Exact.linears(artifact, "adaLN_modulation.1", xs: enabled)
        let end = try Krea2Exact.linears(artifact, "final_norm.linear", xs: cs)
        return sigmas.indices.map { i in (layers[i] + end[i]).map(Float.init) }
    }

    /// **The sinusoid of `Timesteps(4096, flip_sin_to_cos=False)`, computed in double** — `[sin |
    /// cos]` — on the timestep the reference receives: `t = σ·1000` in fp32 (the scheduler),
    /// passed as is to the DiT. Same reason as `KleinDiT.sinusoid`: the angles climb to 1000 rad.
    package static func sinusoid(sigma: Float, dim: Int) -> [Double] {
        let half = dim / 2
        let entry = Double(sigma * 1000)
        var out = [Double](repeating: 0, count: dim)
        for j in 0..<half {
            let angle = entry * exp(-log(10000) * Double(j) / Double(half))
            out[j] = Double(Float(sin(angle)))
            out[half + j] = Double(Float(cos(angle)))
        }
        return out
    }

    // ── the layer ───────────────────────────────────────────────────────────────────────

    private func layerPass(_ layer: Int) throws {
        let d = config.dim, hh = config.hidden, n = maxRows
        let p = "layers.\(layer)."
        let mod = modulationSlot   // shift, scale, gate (attention) · shift, scale, gate (MLP)

        // ── attention ─────────────────────────────────────────────────────────────────────
        try modulatedNorm(p + "adaLN_sa_ln.weight", into: normed, scale: mod + d, shift: mod)
        for (name, output) in [("to_q", q), ("to_k", k), ("to_v", v)] {
            try linearGPU(p + "self_attention.\(name).weight", a: normed, into: output, m: n, k: d, n: d)
        }
        try headNorm(p + "self_attention.norm_q.weight", q)
        try headNorm(p + "self_attention.norm_k.weight", k)
        timed("rope") {
            ErnieRope.apply(q, table: freqs, rows: n, heads: config.heads, headDim: config.headDim)
            ErnieRope.apply(k, table: freqs, rows: n, heads: config.heads, headDim: config.headDim)
        }
        try attend()
        try linearGPU(p + "self_attention.to_out.0.weight", a: merged, into: o, m: n, k: d, n: d)
        let (xb, ob, mb) = (try slice(x), try slice(o), try slice(modulationSlot))
        timed("residual") {
            timings["elementwise GPU", default: 0] += elementwise.residual((xb, 0), plus: (ob, 0), carries: (mb, 2 * d),
                                                                          rows: n, columns: d)
        }

        // ── MLP: up ⊙ GELU(gate) ───────────────────────────────────────────────────────────
        try modulatedNorm(p + "adaLN_mlp_ln.weight", into: normed, scale: mod + 4 * d, shift: mod + 3 * d)
        try linearGPU(p + "mlp.gate_proj.weight", a: normed, into: h1, m: n, k: d, n: hh)
        try linearGPU(p + "mlp.up_proj.weight", a: normed, into: h2, m: n, k: d, n: hh)
        let (h1b, h2b) = (try slice(h1), try slice(h2))
        timed("geglu") { timings["elementwise GPU", default: 0] += gelu.geglu(h1b, h2b, count: n * hh) }
        try linearGPU(p + "mlp.linear_fc2.weight", a: h1, into: o, m: n, k: hh, n: d)
        timed("residual") {
            timings["elementwise GPU", default: 0] += elementwise.residual((xb, 0), plus: (ob, 0), carries: (mb, 5 * d),
                                                                          rows: n, columns: d)
        }
    }

    // ── plumbing ────────────────────────────────────────────────────────────────────────

    /// A slice of the arena seen from the GPU, wrapped **to the end of its reservation**.
    private func slice(_ p: UnsafeMutablePointer<Float>) throws -> MTLBuffer {
        guard let (begin, bytes) = reserved.first(where: { $0.key <= p && p < $0.key + $0.value / 4 }) else {
            throw Artifact.Failure.badHeader("ERNIE-Image: an address outside the arena")
        }
        return try gemm.wrap(UnsafeMutableRawPointer(p), bytes: bytes - (p - begin) * 4, name: "slice")
    }

    /// `destination = RMS(x) · weight · (1 + scale) + shift` — diffusers' RMSNorm, on the CPU.
    private func modulatedNorm(_ weights: String, into destination: UnsafeMutablePointer<Float>,
                               scale: UnsafePointer<Float>, shift: UnsafePointer<Float>) throws {
        let d = config.dim
        try artifact.materialize(weights, into: reserve, capacity: reserveCapacity)
        timed("norm") {
            Ops.rmsNorm(x, weight: reserve, into: destination, rows: maxRows, columns: d, eps: Float(config.eps))
            Krea2Ops.modulate(destination, scale: scale, shift: shift, rows: maxRows, columns: d)
        }
    }

    /// QK-Norm: RMSNorm over `head_dim`, eps 1e-6, ordinary weight.
    private func headNorm(_ name: String, _ tensor: UnsafeMutablePointer<Float>) throws {
        try artifact.materialize(name, into: reserve, capacity: reserveCapacity)
        timed("norm") { Ops.rmsNorm(tensor, weight: reserve, into: tensor, rows: maxRows * config.heads,
                                    columns: config.headDim, eps: Float(config.eps)) }
    }

    private func attend() throws {
        let attention: Attention
        if let a = self.attention { attention = a } else {
            attention = Attention(device: gemm.device, queue: gemm.queue, heads: config.heads,
                                  sequence: maxRows, headDim: config.headDim)
            self.attention = attention
        }
        let (qb, kb, vb, ob) = (try slice(q), try slice(k), try slice(v), try slice(merged))
        timed("sdpa wall") { timings["sdpa GPU", default: 0] += attention.run(q: qb, k: kb, v: vb, into: ob) }
    }

    /// `c[s, j] += bias[j]` on the CPU.
    private func addBias(_ name: String, at c: UnsafeMutablePointer<Float>, rows: Int, columns: Int? = nil) throws {
        let n = columns ?? config.dim
        try artifact.materialize(name, into: reserve, capacity: reserveCapacity)
        for r in 0..<rows { vDSP_vadd(c + r * n, 1, reserve, 1, c + r * n, 1, vDSP_Length(n)) }
    }

    /// `C[m, n] = A[m, k] · W`, W laid out `[k, n]` by the forge.
    ///
    /// The bf16 → fp32 widening stays on the CPU, behind the prefetcher: on the GPU (`WidenGPU`), the
    /// 16 GB map — larger than what the cache keeps — gets wired cold tensor by
    /// tensor, 13.3 s per 512² evaluation against 7.3–8.0 (as measured on Krea 2).
    private func linearGPU(_ name: String, a: UnsafeMutablePointer<Float>, into c: UnsafeMutablePointer<Float>,
                           m: Int, k: Int, n: Int) throws {
        let got = try timed("widening") { try artifact.materialize(name, into: reserve, capacity: reserveCapacity) }
        guard got == k * n else {
            throw Artifact.Failure.badHeader("\(name): \(got) values, \(k * n) expected")
        }
        let (ab, wb, cb) = (try slice(a), try slice(reserve), try slice(c))
        timed("gemm wall") {
            timings["gemm GPU", default: 0] += gemm.linear(a: ab, b: wb, c: cb, m: m, k: k, n: n, weightIsTransposed: true)
        }
        // The stack, accumulated into the same output, after the main GEMM (and before any bias).
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

/// **ERNIE-Image's RoPE**: `ErnieImageEmbedND3` then `apply_rotary_emb`, reproduced as is.
///
/// Three axes `(32, 48, 48)` give 16 + 24 + 24 = 64 angles `e`, each **repeated twice**
/// side by side: `F = [e₀, e₀, e₁, e₁, …, e₆₃, e₆₃]` (128). Then `rotate_half`: component `j`
/// is paired with `j + 64`, not `j + 1`. For `j < 64`:
///
///     out[j]      = x[j]·cos F[j]           − x[j+64]·sin F[j]           F[j]      = e[j/2]
///     out[j + 64] = x[j+64]·cos F[j+64]     + x[j]·sin F[j+64]           F[j + 64] = e[32 + j/2]
///
/// The two outputs of a pair rotate by two different angles: it is not a rotation, and
/// no permutation of the weights brings it back to `Ops.rope`. The table keeps the four values per
/// pair: `[cos F[j], sin F[j], cos F[j+64], sin F[j+64]]`.
///
/// The angles in fp32 like the reference: `ω = 1 / θ^(2i/d)` (`torch.pow` on fp32), the angle
/// `position · ω` as one product, cos and sin in fp32.
package enum ErnieRope {
    /// The image `(y, x)` as `(T, y, x)`, rows `[0, S)`; the text `t` as `(t, 0, 0)`, rows `[S, S + T)`.
    package static func write(into out: UnsafeMutablePointer<Float>, textRows: Int, tilesHigh: Int,
                             tilesWide: Int, axes: [Int], theta: Float) {
        let ω = axes.map { dims in (0..<(dims / 2)).map { i in 1 / powf(theta, Float(2 * i) / Float(dims)) } }
        let pairs = axes.reduce(0, +) / 2          // 64
        func write(_ row: Int, _ position: [Int]) {
            var e = [Float](); e.reserveCapacity(pairs)
            for (axis, p) in position.enumerated() { for f in ω[axis] { e.append(Float(p) * f) } }
            let rowLine = out + row * 4 * pairs
            for j in 0..<pairs {
                let a = e[j / 2], b = e[pairs / 2 + j / 2]
                rowLine[4 * j] = cosf(a); rowLine[4 * j + 1] = sinf(a)
                rowLine[4 * j + 2] = cosf(b); rowLine[4 * j + 3] = sinf(b)
            }
        }
        for y in 0..<tilesHigh {
            for x in 0..<tilesWide { write(y * tilesWide + x, [textRows, y, x]) }
        }
        let s = tilesHigh * tilesWide
        for t in 0..<textRows { write(s + t, [t, 0, 0]) }
    }

    /// In place on `[rows, heads, headDim]`, the two products rounded then summed like
    /// `x * cos_ + x_rotated * sin_`.
    package static func apply(_ x: UnsafeMutablePointer<Float>, table: UnsafePointer<Float>,
                              rows: Int, heads: Int, headDim: Int) {
        let half = headDim / 2
        Parallel.rows(rows, width: heads * headDim) { first, howMany in
            for r in first..<(first + howMany) {
                let f = table + r * 4 * half
                for h in 0..<heads {
                    let v = x + (r * heads + h) * headDim
                    for j in 0..<half {
                        let a = v[j], b = v[j + half]
                        let ca = f[4 * j], sa = f[4 * j + 1], cb = f[4 * j + 2], sb = f[4 * j + 3]
                        let p1 = a * ca, p2 = -b * sa, p3 = b * cb, p4 = a * sb
                        v[j] = p1 + p2
                        v[j + half] = p3 + p4
                    }
                }
            }
        }
    }
}

extension ErnieDiT {
    /// **The ERNIE-Image schedule**: `linspace(1, 0, N + 1)[:-1]` (in fp32, like
    /// `torch.linspace`), shifted by `σ' = s·σ / (1 + (s − 1)·σ)` with `s = 4` (the scheduler's
    /// `shift`, without dynamic shifting), then a zero terminal σ. Does not depend on the grid.
    package static func sigmas(steps: Int, shift: Float = 4) -> [Float] {
        let step = Float(0 - 1) / Float(steps), half = (steps + 1) / 2
        return (0..<steps).map { i -> Float in
            let s: Float = i < half ? 1 + step * Float(i) : 0 - step * Float(steps - i)
            return shift * s / (1 + (shift - 1) * s)
        } + [0]
    }
}

/// **The ERNIE-Image text**, as `ErnieImagePipeline.encode_prompt` prepares it.
///
///     tokenizer (`add_special_tokens=True`: `<s>` in front, **no template**, truncated to 2048)
///       ─ Ministral-3 ─ hidden_states[-2] (the output of layer 24 of 26) ─ [T, 3072]
///
/// No padding: the DiT receives the `T` real tokens.
package enum ErnieText {
    /// `model_max_length` of the published tokenizer.
    package static let maximumLength = 2048

    package static func identifiers(_ prompt: String, tokenizer: Tokenizer) -> [Int] {
        let ids = tokenizer.headTokens + tokenizer.encode(prompt)
        return Array(ids.prefix(maximumLength))
    }

    /// `[T, 3072]`.
    package static func hiddenStates(_ ids: [Int], encoder: String, freezeCut: Bool = EngineSettings.effective.frozenCut,
                                   cancellation: Cancellation? = nil) throws -> [Float] {
        let encoder = try TextEncoder(artifact: try Artifact(path: encoder), sequence: ids.count, freezeCut: freezeCut)
        encoder.cancellation = cancellation
        return Array(try encoder.encode(ids: ids, realTokens: ids.count))
    }
}
