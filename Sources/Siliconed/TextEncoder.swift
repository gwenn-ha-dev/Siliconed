import Foundation
import Metal

/// The decoder-only text encoder: first Z-Image Turbo's Qwen3-4B, stopped at `hidden_states[-2]`,
/// then the language models of the other families on the same code (Krea 2, FLUX.2 [klein],
/// ERNIE-Image, and Qwen-Image-2.1's Qwen3-VL-8B through `Qwen3VLEncoder`). The notes below are
/// Z-Image's; what each family adds is in its own file. The tokenizer is in `Tokenizer.swift`.
///
/// ## Three things not to rediscover
///
/// **There are only 35 layers out of 36.** `hidden_states[-2]` was compared against the golden tensors rather than
/// deduced from the `transformers` code: it is **identical to `layer34_out`**, to the last bit.
/// `all_hidden_states` collects the *input* of each layer, then the output of the final norm —
/// with 36 layers it has 37 entries, so `[-2]` is the input of layer 35. Layer 35, the
/// final norm and the `lm_head` are not in the map: `ForgeText` does not write them.
///
/// **RoPE is interleaved because the forge laid it out that way.** `transformers` pairs
/// `(j, j + Dh/2)`, `Ops.rope` pairs `(2p, 2p+1)`. The forge permutes the outputs of `q_proj` and
/// `k_proj` — and the `q_norm`/`k_norm` weights with them — so that the interleaved rotation
/// computes exactly `rotate_half`. Verified numerically with zero error. **Do not "fix" the
/// RoPE here**: that would be fixing it twice.
///
/// **Attention is causal, and the padding is on the right.** A token `i < realCount` therefore only attends to
/// real tokens: its hidden states do not depend on the padded length. That is what
/// allows `sequence = realCount` instead of 512 — **check before relying on it**, by comparing
/// `cap_feats` at both lengths, because if the reference were bidirectional the
/// reasoning would collapse entirely.
package final class TextEncoder {

    /// What the artifact declares. Nothing is hard-coded: a map forged from a Qwen3-0.6B
    /// (Anima) would carry other numbers and the same code.
    package struct Config {
        package let hidden: Int, layers: Int, heads: Int, kvHeads: Int, headDim: Int
        package let intermediate: Int, eps: Float, theta: Float
        /// The inverse frequencies of RoPE when they are not `θ^(−2j/Dh)`: the **yarn** RoPE
        /// of ERNIE-Image's Ministral-3 (`rope_type: yarn`). `nil`: ordinary RoPE.
        package let frequencies: [Float]?
        /// Qwen3-VL's **interleaved mRoPE** sections `(t, h, w)` — `(24, 20, 20)` —, `nil` elsewhere.
        /// Only `setPositions` reads them: without a call, the 1D table stays (Krea 2's verified bits).
        package let mropeSections: [Int]?

        init(_ header: [String: Any], lastLayer: Int) throws {
            guard let c = header["config"] as? [String: Any],
                  let hidden = c["hidden_size"] as? Int,
                  let heads = c["num_attention_heads"] as? Int,
                  let kvHeads = c["num_key_value_heads"] as? Int,
                  let headDim = c["head_dim"] as? Int,
                  let intermediate = c["intermediate_size"] as? Int else {
                throw Artifact.Failure.badHeader("incomplete encoder config")
            }
            self.hidden = hidden
            self.layers = lastLayer + 1          // 0…lastLayer inclusive: the map only has those
            self.heads = heads
            self.kvHeads = kvHeads
            self.headDim = headDim
            self.intermediate = intermediate
            self.eps = Float(c["rms_norm_eps"] as? Double ?? 1e-6)
            // `transformers` 5 puts θ under `rope_parameters` (Anima's Qwen3-0.6B), 4 at the root.
            self.theta = Float(c["rope_theta"] as? Double
                               ?? (c["rope_parameters"] as? [String: Any])?["rope_theta"] as? Double
                               ?? 1_000_000)
            let rope = c["rope_parameters"] as? [String: Any] ?? c["rope_scaling"] as? [String: Any] ?? [:]
            if let sections = rope["mrope_section"] as? [Int] {
                // The chunked layout (`[T…T H…H W…W]`, Qwen2-VL) would be another table.
                guard rope["mrope_interleaved"] as? Bool == true, sections.count == 3,
                      sections.reduce(0, +) == headDim / 2 else {
                    throw Artifact.Failure.badHeader("mRoPE \(sections): only the interleaved layout is ported")
                }
                self.mropeSections = sections
            } else {
                self.mropeSections = nil
            }
            switch rope["rope_type"] as? String ?? rope["type"] as? String ?? "default" {
            case "default":
                self.frequencies = nil
            case "yarn":
                self.frequencies = try Config.yarn(rope, theta: Double(self.theta), dim: headDim)
            case let other:
                throw Artifact.Failure.badHeader("RoPE \"\(other)\": this port only knows ordinary RoPE and yarn")
            }
            // GQA is stated here, once: `heads` queries for `kvHeads` keys. The DiT has 30/30,
            // Qwen3-4B has 32/8, the other families their own. A non-integer ratio does not exist.
            guard heads % kvHeads == 0 else {
                throw Artifact.Failure.badHeader("GQA: \(heads) q heads for \(kvHeads) k/v")
            }
        }
        /// **The yarn RoPE frequencies**, as `_compute_yarn_parameters` in `transformers`
        /// computes them — in fp32, operation by operation: `θ^(2j/d)`, the extrapolated inverse and
        /// the interpolated inverse (`/ factor`), blended by a ramp between two correction
        /// dimensions (`beta_fast`, `beta_slow`, truncated). The attention factor (`mscale`)
        /// must be 1: this port does not multiply cos and sin.
        static func yarn(_ r: [String: Any], theta: Double, dim: Int) throws -> [Float] {
            func real(_ k: String) -> Double? { (r[k] as? Double) ?? (r[k] as? Int).map(Double.init) }
            guard let factor = real("factor"), let origin = real("original_max_position_embeddings") else {
                throw Artifact.Failure.badHeader("yarn RoPE without `factor` or `original_max_position_embeddings`")
            }
            func mscale(_ m: Double) -> Double { factor <= 1 ? 1 : 0.1 * m * log(factor) + 1 }
            let attention: Double
            if let a = real("attention_factor") { attention = a }
            else if let m = real("mscale"), let mad = real("mscale_all_dim"), m != 0, mad != 0 { attention = mscale(m) / mscale(mad) }
            else { attention = mscale(1) }
            guard attention == 1 else {
                throw Artifact.Failure.badHeader("yarn RoPE: attention factor \(attention) ≠ 1, not supported")
            }
            let fast = real("beta_fast") ?? 32, lent = real("beta_slow") ?? 1
            func dimension(_ turns: Double) -> Double {
                Double(dim) * log(origin / (turns * 2 * .pi)) / (2 * log(theta))
            }
            var down = dimension(fast), up = dimension(lent)
            if (r["truncate"] as? Bool) ?? true { down = down.rounded(.down); up = up.rounded(.up) }
            down = max(down, 0); up = min(up, Double(dim - 1))
            if down == up { up += 0.001 }
            let (b, h) = (Float(down), Float(up)), base = Float(theta), f = Float(factor)
            return (0..<(dim / 2)).map { j in
                let power = powf(base, Float(2 * j) / Float(dim))
                let extrapolated = 1 / power, interpolated = 1 / (f * power)
                let ramp = min(max((Float(j) - b) / (h - b), 0), 1)
                let part = 1 - ramp
                return interpolated * (1 - part) + extrapolated * part
            }
        }

        /// How many `q` heads share one `k`/`v` head.
        package var groupSize: Int { heads / kvHeads }
        /// The width of the attention projections — `heads · headDim`, which is **not** `hidden`.
        package var attentionWidth: Int { heads * headDim }
        package var kvWidth: Int { kvHeads * headDim }
    }

    package let config: Config
    package let sequence: Int
    private let artifact: Artifact
    private let gemm: GEMM
    private let arena: Arena
    private let attention: Attention
    private let prefetcher: Prefetcher
    /// The layers requested ahead of the one computing (see `encode`).
    static let layersAhead = 3

    // Arena slices, reserved once.
    private let x, normed, projected, q, k, v, kWide, vWide, attended: UnsafeMutablePointer<Float>
    private let gate, up, swiglu, freqs, scratch, reserve: UnsafeMutablePointer<Float>
    private let xBuf, normedBuf, projectedBuf, qBuf, kBuf, vBuf: MTLBuffer
    private let kWideBuf, vWideBuf, attendedBuf, gateBuf, upBuf, reserveBuf: MTLBuffer

    /// - Parameter sequence: the number of positions computed. **Not necessarily 512** — see
    ///   the note on causality at the top of this file.
    /// - Parameter freezeCut: the reproducible mode. **The encoder needs it as much as the
    ///   DiT**: its blocks run at 512 tokens, hence above `amx_min`, hence the GPU/AMX split
    ///   applies there and is slaved to it. An engine that froze only the DiT would render
    ///   different captions on every run, and the image behind them too — it is
    ///   exactly the kind of half-fix that reads like a fix.
    /// - Parameter visibleKeys: the real positions, when the padding is **computed** and not
    ///   removed (FLUX.2 [klein]: its 512 rows enter the DiT without a mask). The rows beyond
    ///   only attend to real tokens; the real rows do not change by a single bit.
    package init(artifact: Artifact, sequence: Int,
                freezeCut: Bool = EngineSettings.effective.frozenCut, visibleKeys: Int? = nil) throws {
        guard let lastLayer = artifact.header["last_layer"] as? Int else {
            throw Artifact.Failure.badHeader("`last_layer` missing: this artifact is not an encoder")
        }
        let config = try Config(artifact.header, lastLayer: lastLayer)
        // **LOCAL bindings until all properties are set.** A nested function
        // that reads `self.arena` captures `self`, and Swift forbids that before full
        // initialization. Same constraint as in `Attention.swift` and `VAE.swift`, and it always
        // costs the same way: "variable used before being initialized" on a line that
        // seems to use only locals.
        let gemm = try GEMM(freezeCut: freezeCut)
        let s = sequence, h = config.hidden, a = config.attentionWidth
        let kv = config.kvWidth, i = config.intermediate
        // The largest weight of a block: `hidden × intermediate`. The attention projections are
        // smaller, so a single reserve is enough — same decision as `Block`.
        let budget = (5 * s * h + 3 * s * a + 2 * s * kv + 3 * s * i
                      + s * config.headDim + h * i + 4 * h) * 4 + (32 << 20)
        let arena = try Arena(capacity: budget)
        func slot(_ name: String, _ count: Int) throws -> UnsafeMutablePointer<Float> {
            try arena.reserve(name, bytes: count * 4).assumingMemoryBound(to: Float.self)
        }
        let x = try slot("x", s * h)
        let normed = try slot("normed", s * h)
        let projected = try slot("projected", s * h)
        let q = try slot("q", s * a)
        let k = try slot("k", s * kv)
        let v = try slot("v", s * kv)
        let kWide = try slot("kWide", s * a)
        let vWide = try slot("vWide", s * a)
        let attended = try slot("attended", s * a)
        let gate = try slot("gate", s * i)
        let up = try slot("up", s * i)
        // `siluGate` wants a scratch buffer the size of the count: it computes the sigmoid
        // separately rather than in place, and the parallel split requires disjoint slices.
        let swiglu = try slot("swiglu", s * i)
        let freqs = try slot("freqs", s * config.headDim)      // [S, Dh/2, 2]
        let scratch = try slot("scratch", 4 * h)               // the norm weights, widened
        let reserve = try slot("reserve", h * i)

        self.config = config
        self.sequence = sequence
        self.artifact = artifact
        self.gemm = gemm
        self.arena = arena
        self.prefetcher = Prefetcher(artifact: artifact)
        self.x = x; self.normed = normed; self.projected = projected
        self.q = q; self.k = k; self.v = v
        self.kWide = kWide; self.vWide = vWide; self.attended = attended
        self.gate = gate; self.up = up; self.swiglu = swiglu
        self.freqs = freqs; self.scratch = scratch; self.reserve = reserve

        self.attention = Attention(device: gemm.device, queue: gemm.queue,
                                   heads: config.heads, sequence: s, headDim: config.headDim,
                                   causal: true, visibleKeys: visibleKeys)

        xBuf = try gemm.wrap(x, bytes: s * h * 4, name: "x")
        normedBuf = try gemm.wrap(normed, bytes: s * h * 4, name: "normed")
        projectedBuf = try gemm.wrap(projected, bytes: s * h * 4, name: "projected")
        qBuf = try gemm.wrap(q, bytes: s * a * 4, name: "q")
        kBuf = try gemm.wrap(k, bytes: s * kv * 4, name: "k")
        vBuf = try gemm.wrap(v, bytes: s * kv * 4, name: "v")
        kWideBuf = try gemm.wrap(kWide, bytes: s * a * 4, name: "kWide")
        vWideBuf = try gemm.wrap(vWide, bytes: s * a * 4, name: "vWide")
        attendedBuf = try gemm.wrap(attended, bytes: s * a * 4, name: "attended")
        gateBuf = try gemm.wrap(gate, bytes: s * i * 4, name: "gate")
        upBuf = try gemm.wrap(up, bytes: s * i * 4, name: "up")
        reserveBuf = try gemm.wrap(reserve, bytes: h * i * 4, name: "reserve")

        // A 1D table: `rope_theta` is 10⁶ here against 256 for the DiT, and the position is
        // the token index. Written once — it depends on neither the prompt nor the layer.
        if let frequencies = config.frequencies {
            // yarn: `inv_freq @ position` in fp32 (a single product, hence one rounding), then cos, sin.
            for position in 0..<s {
                let rowLine = freqs + position * config.headDim
                for (j, f) in frequencies.enumerated() {
                    let angle = Float(position) * f
                    rowLine[2 * j] = cosf(angle); rowLine[2 * j + 1] = sinf(angle)
                }
            }
        } else {
            let tables = RopeTables(axesDims: [config.headDim], axesLens: [s], theta: config.theta)
            for position in 0..<s {
                tables.write(ids: [position], into: freqs + position * config.headDim)
            }
        }
    }

    /// **Qwen3-VL's interleaved mRoPE**: rewrites the table from three positions per token,
    /// `(t, h, w)` — `Qwen3VLModel.get_rope_index`, see `Qwen3VLPrompt.positions`.
    ///
    /// `Qwen3VLTextRotaryEmbedding` computes `inv_freq · position` for each of the three axes, then
    /// keeps, frequency by frequency, ONE axis: `h` for `j ≡ 1 (mod 3)`, `j < 3·20`; `w` for
    /// `j ≡ 2`, `j < 3·20`; `t` everywhere else (`recomposition_frequencies`). The forge's per-head
    /// permutation puts frequency `j` on the pair `(2j, 2j+1)`, so this is the same table layout as
    /// the 1D one. A text-only prompt has `t = h = w = index`: the 1D RoPE, by construction.
    ///
    /// The frequencies are `transformers`' own, `1 / θ^(2j/Dh)` in fp32 (`inverseFrequencies`),
    /// and NOT `θ^(−2j/Dh)` as in the 1D path: one rounding apart, and the angle multiplies it by
    /// positions in the hundreds.
    package func setPositions(_ positions: [[Int]]) {
        guard let sections = config.mropeSections else {
            preconditionFailure("setPositions: this map declares no mRoPE")
        }
        precondition(positions.count == 3 && positions.allSatisfy { $0.count >= sequence },
                     "setPositions: three axes of \(sequence) positions are needed")
        let half = config.headDim / 2
        let inverse = TextEncoder.inverseFrequencies(theta: config.theta, dim: config.headDim)
        let axes = (0..<half).map { TextEncoder.mropeAxis(frequency: $0, sections: sections) }
        for s in 0..<sequence {
            let row = freqs + s * config.headDim
            for j in 0..<half {
                let angle = Float(positions[axes[j]][s]) * inverse[j]
                row[2 * j] = cosf(angle); row[2 * j + 1] = sinf(angle)
            }
        }
    }

    /// `1.0 / (θ ** (arange(0, Dh, 2, dtype=float) / Dh))`, fp32 operation by operation.
    package static func inverseFrequencies(theta: Float, dim: Int) -> [Float] {
        (0..<(dim / 2)).map { j in 1 / powf(theta, Float(2 * j) / Float(dim)) }
    }

    /// The axis (0 = t, 1 = h, 2 = w) whose position frequency `j` reads, in the interleaved layout.
    package static func mropeAxis(frequency j: Int, sections: [Int]) -> Int {
        if j % 3 == 1 && j < 3 * sections[1] { return 1 }
        if j % 3 == 2 && j < 3 * sections[2] { return 2 }
        return 0
    }

    /// The boundaries recorded, when asked — the same mechanism as `Block`, so that
    /// the encoder is checked **layer by layer** and not only at its output. The goldens
    /// carry `layer33_out`, `layer34_out`, `layer35_out` and `final_norm_out`.
    package var recordBoundaries = false
    /// When set, only these boundaries are recorded — and the INPUTS of the layers (`layerN_in`)
    /// become recordable. At 577 tokens a boundary weighs 9.5 MB, and there would be 72 of them.
    package var recordedNames: Set<String>?
    /// Called once on the embeddings, before layer 0: Qwen3-VL writes its image features over the
    /// `<|image_pad|>` rows there (`masked_scatter`).
    package var embeddingsHook: ((UnsafeMutablePointer<Float>) -> Void)?
    /// Called on the residual stream after each layer, once `layerN_out` is recorded: Qwen3-VL's
    /// **deepstack** adds its features to the image rows after layers 0, 1 and 2.
    package var layerOutputHook: ((Int, UnsafeMutablePointer<Float>) -> Void)?
    /// The current render's token, consulted between two layers (see `Cancellation`).
    package var cancellation: Cancellation?
    package private(set) var boundaries: [String: [Float]] = [:]

    private func record(_ name: String, _ pointer: UnsafePointer<Float>, _ count: Int) {
        guard recordBoundaries, recordedNames?.contains(name) ?? true else { return }
        boundaries[name] = Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func widen(_ name: String, count: Int, into destination: UnsafeMutablePointer<Float>) throws {
        let got = try artifact.materialize(name, into: destination, capacity: count)
        guard got == count else {
            throw Artifact.Failure.badHeader("\(name): \(got) values, \(count) expected")
        }
    }

    /// `RMSNorm` over the `Dh` axis, head by head — Qwen3's QK-Norm, **before** RoPE.
    ///
    /// It is a norm per *head* and not per token: the mean square is taken over the 128
    /// components of a head, not over the 4096 of the whole vector. Confusing the two gives plausible
    /// vectors and wrong attention.
    private func normHeads(_ t: UnsafeMutablePointer<Float>, weight: UnsafePointer<Float>,
                           heads: Int) {
        let d = config.headDim, eps = config.eps
        Parallel.rows(sequence, width: heads * d) { first, howMany in
            for s in first..<(first + howMany) {
                for h in 0..<heads {
                    let row = t + (s * heads + h) * d
                    var sum: Float = 0
                    for j in 0..<d { sum += row[j] * row[j] }
                    let scale = 1 / (sum / Float(d) + eps).squareRoot()
                    for j in 0..<d { row[j] = row[j] * scale * weight[j] }
                }
            }
        }
    }

    /// `kvHeads` heads repeated `groupSize` times — what GQA asks for and what
    /// `MPSGraph.scaledDotProductAttention` does not expose.
    ///
    /// Copy rather than broadcast: 512 tokens × 4096 floats = 8 MB per tensor and per
    /// layer, i.e. 0.6 GB over the 35 layers. Measurable, but very far from the 7.84 GB of weights
    /// that the same pass reads — and a graph that broadcast would cost one more transposition.
    /// **To revisit if the encoder becomes anything other than a single pass.**
    private func expandKV(_ source: UnsafePointer<Float>, into destination: UnsafeMutablePointer<Float>) {
        let d = config.headDim, group = config.groupSize, kvHeads = config.kvHeads
        Parallel.rows(sequence, width: config.attentionWidth) { first, howMany in
            for s in first..<(first + howMany) {
                for kvHead in 0..<kvHeads {
                    let from = source + (s * kvHeads + kvHead) * d
                    for repetition in 0..<group {
                        let to = destination + (s * (kvHeads * group) + kvHead * group + repetition) * d
                        to.update(from: from, count: d)
                    }
                }
            }
        }
    }

    /// **Krea 2's taps**: `hidden_states[i]` for each `i` requested — the INPUT of
    /// layer `i`, verified against the oracle's text encoder (zero error), and not its output. Returns
    /// `[sequence, sockets.count, hidden]`, the `dim=2` stacking of `Krea2Pipeline`. The highest
    /// tap may be `config.layers`: it is then the output of the last kept layer.
    package func encodeTaps(ids: [Int], taps: [Int]) throws -> [Float] {
        guard let highest = taps.max(), highest <= config.layers, taps.allSatisfy({ $0 >= 0 }) else {
            throw Artifact.Failure.badHeader("taps \(taps): the map stops at the input of layer \(config.layers)")
        }
        let h = config.hidden
        var stacked = [Float](repeating: 0, count: sequence * taps.count * h)
        tap = { [sequence] layer, x in
            for (slot, wanted) in taps.enumerated() where wanted == layer {
                stacked.withUnsafeMutableBufferPointer { out in
                    for position in 0..<sequence {
                        (out.baseAddress! + (position * taps.count + slot) * h)
                            .update(from: x + position * h, count: h)
                    }
                }
            }
        }
        defer { tap = nil }
        _ = try encode(ids: ids, realTokens: sequence, stopAfter: highest)
        return stacked
    }

    /// Called with the input of each layer (and, after the last, with its output).
    private var tap: ((Int, UnsafePointer<Float>) -> Void)?

    /// The identifiers → `[realCount, hidden]`, that is `cap_feats`.
    ///
    /// The reference's masked extraction (`pipeline_z_image.py` 217-245) reduces here to the first
    /// `realCount` rows: the padding is on the right and attention is causal.
    package func encode(ids: [Int], realTokens: Int) throws -> UnsafeBufferPointer<Float> {
        try encode(ids: ids, realTokens: realTokens, stopAfter: config.layers)
    }

    /// - Parameter stopAfter: the number of layers to compute — Krea 2's taps stop
    ///   at the input of layer 35, that is after 35 layers, like Z-Image.
    private func encode(ids: [Int], realTokens: Int, stopAfter: Int) throws -> UnsafeBufferPointer<Float> {
        precondition(ids.count >= sequence, "at least \(sequence) identifiers are needed")
        precondition(realTokens <= sequence, "\(realTokens) real tokens for \(sequence) positions")
        let h = config.hidden, a = config.attentionWidth, kv = config.kvWidth
        let i = config.intermediate

        // The embedding table is 778 MB and only `sequence` rows of it are touched. It is
        // mapped: a lookup only pages in what it reads, so ~19 pages instead of
        // 47,500. That is why the forge does NOT transpose it.
        guard let table = artifact.tensors["embed_tokens.weight"] else {
            throw Artifact.Failure.badHeader("embed_tokens.weight missing from the map")
        }
        guard table.shape.count == 2, table.shape[1] == h else {
            throw Artifact.Failure.badHeader("embed_tokens: \(table.shape) instead of [V, \(h)]")
        }
        for position in 0..<sequence {
            let token = ids[position]
            // `-1` is the tokenizer's own refusal (`Tokenizer.bpe`): a piece of the prompt that has no
            // entry in its vocabulary, which the byte alphabet makes impossible with the file this
            // port was validated on. Beyond the table, the tokenizer and the map disagree.
            guard token >= 0 else {
                throw Tokenizer.Failure.malformed("a piece of the prompt has no entry in the vocabulary "
                                                  + "(position \(position)): not the tokenizer this port was validated on")
            }
            guard token < table.shape[0] else {
                throw Artifact.Failure.badHeader("token \(token) beyond the embedding table's \(table.shape[0]) rows: "
                                                 + "the tokenizer and the map are not of the same model")
            }
            // One row, whatever the map's dtype (bf16 here today; an 8-bit table reads its scale).
            try artifact.materializeRows("embed_tokens.weight", first: token, count: 1, into: x + position * h)
        }
        embeddingsHook?(x)
        record("embed_out", x, sequence * h)

        // **Three layers ahead, from layer 0 on.** With one ahead (and layer 0 read cold), a short
        // prompt — where the layers cost their read, not their compute — streamed Qwen3-VL-8B's 16 GB
        // at ~2.7 GB/s: 6.3–6.8 s; two ahead 5.6–5.8 s; three 5.4 s (3.5 GB/s), same bits. The pages
        // are clean and file-backed: in flight, never swapped.
        let ahead = TextEncoder.layersAhead
        // **Every layer streamed around the page cache** (`TailStream`): an encoder's map is
        // read once per render and never stays cached, so it is read with `pread` + `F_NOCACHE` into
        // `ahead + 1` staging buffers instead of paged in. Qwen3-VL-8B text only: 4.63 → 3.1–3.4 s,
        // disk 13.9 → 6–7 GB (the rest was still cached); +1.5 GB of footprint while it runs. And the
        // encoder no longer evicts what the cache holds of the DiT. In a render, the budget decides how
        // many buffers (`MemoryPlan.encoderSlots`, 1…4, none = every layer through the map); the layers
        // requested ahead stay three — a block with no free buffer waits for one (`TailStream.stage`).
        let layerBlocks = (0..<min(stopAfter, config.layers)).map { prefetcher.namesOfBlock(prefix: "layers.\($0).") }
        try artifact.streamTail(blocks: layerBlocks,
                                slots: min(ahead + 1, MemoryPlan.encoderSlots(artifact: artifact, layerBlocks: layerBlocks)))
        for l in 0..<min(ahead, config.layers) { prefetcher.request(prefetcher.namesOfBlock(prefix: "layers.\(l).")) }
        for layer in 0..<min(stopAfter, config.layers) {
            try cancellation.check()   // between two layers, never inside one (see `Cancellation`)
            tap?(layer, x)
            if recordedNames != nil { record("layer\(layer)_in", x, sequence * h) }
            let p = "layers.\(layer)."
            prefetcher.request(prefetcher.namesOfBlock(prefix: "layers.\(layer + ahead)."))

            // ── attention ───────────────────────────────────────────────────────────────────
            try widen(p + "input_layernorm.weight", count: h, into: scratch)
            Ops.rmsNorm(x, weight: scratch, into: normed, rows: sequence, columns: h, eps: config.eps)
            if layer == 0 { record("l0_normed", normed, sequence * h) }

            try widen(p + "self_attn.q_proj.weight", count: h * a, into: reserve)
            _ = gemm.linear(a: normedBuf, b: reserveBuf, c: qBuf, m: sequence, k: h, n: a,
                            weightIsTransposed: true)
            try widen(p + "self_attn.k_proj.weight", count: h * kv, into: reserve)
            _ = gemm.linear(a: normedBuf, b: reserveBuf, c: kBuf, m: sequence, k: h, n: kv,
                            weightIsTransposed: true)
            try widen(p + "self_attn.v_proj.weight", count: h * kv, into: reserve)
            _ = gemm.linear(a: normedBuf, b: reserveBuf, c: vBuf, m: sequence, k: h, n: kv,
                            weightIsTransposed: true)

            if layer == 0 { record("l0_q", q, sequence * a); record("l0_v", v, sequence * kv) }
            // Qwen3's QK-Norm; Mistral has none.
            if artifact.tensors[p + "self_attn.q_norm.weight"] != nil {
                try widen(p + "self_attn.q_norm.weight", count: config.headDim, into: scratch)
                normHeads(q, weight: scratch, heads: config.heads)
                try widen(p + "self_attn.k_norm.weight", count: config.headDim, into: scratch)
                normHeads(k, weight: scratch, heads: config.kvHeads)
            }

            if layer == 0 { record("l0_qnormed", q, sequence * a) }
            Ops.rope(q, freqs: freqs, sequence: sequence, heads: config.heads, headDim: config.headDim)
            Ops.rope(k, freqs: freqs, sequence: sequence, heads: config.kvHeads, headDim: config.headDim)

            if layer == 0 { record("l0_qroped", q, sequence * a) }
            expandKV(k, into: kWide)
            expandKV(v, into: vWide)
            _ = attention.run(q: qBuf, k: kWideBuf, v: vWideBuf, into: attendedBuf)

            if layer == 0 { record("l0_attended", attended, sequence * a) }
            try widen(p + "self_attn.o_proj.weight", count: a * h, into: reserve)
            _ = gemm.linear(a: attendedBuf, b: reserveBuf, c: projectedBuf, m: sequence, k: a, n: h,
                            weightIsTransposed: true)
            for j in 0..<(sequence * h) { x[j] += projected[j] }

            // ── feed-forward network ────────────────────────────────────────────────────────
            try widen(p + "post_attention_layernorm.weight", count: h, into: scratch)
            Ops.rmsNorm(x, weight: scratch, into: normed, rows: sequence, columns: h, eps: config.eps)

            try widen(p + "mlp.gate_proj.weight", count: h * i, into: reserve)
            _ = gemm.linear(a: normedBuf, b: reserveBuf, c: gateBuf, m: sequence, k: h, n: i,
                            weightIsTransposed: true)
            try widen(p + "mlp.up_proj.weight", count: h * i, into: reserve)
            _ = gemm.linear(a: normedBuf, b: reserveBuf, c: upBuf, m: sequence, k: h, n: i,
                            weightIsTransposed: true)
            Ops.siluGate(gate, up, into: gate, count: sequence * i, scratch: swiglu)

            try widen(p + "mlp.down_proj.weight", count: i * h, into: reserve)
            _ = gemm.linear(a: gateBuf, b: reserveBuf, c: projectedBuf, m: sequence, k: i, n: h,
                            weightIsTransposed: true)
            for j in 0..<(sequence * h) { x[j] += projected[j] }

            record("layer\(layer)_out", x, sequence * h)
            layerOutputHook?(layer, x)
            artifact.dropFromCache(prefetcher.namesOfBlock(prefix: p))   // its staging buffer back
        }
        tap?(min(stopAfter, config.layers), x)

        // `hidden_states[-2]` IS the output of the last kept layer. No final norm:
        // it comes after, and `cap_feats` is taken before.
        //
        // **Except when the map carries it**: Anima reads `last_hidden_state`, which is the output of
        // the final norm. The forge writes it under `--all` and declares it — we do not guess.
        if artifact.header["final_norm"] as? Bool == true {
            try widen("norm.weight", count: h, into: scratch)
            Ops.rmsNorm(x, weight: scratch, into: x, rows: sequence, columns: h, eps: config.eps)
            record("final_norm_out", x, sequence * h)
        }
        return UnsafeBufferPointer(start: x, count: realTokens * h)
    }
}
