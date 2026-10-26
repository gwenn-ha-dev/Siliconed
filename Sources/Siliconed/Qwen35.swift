import Accelerate
import Foundation
import Metal

/// **MOSS-Image's text encoder: Qwen3.5-9B**, text only — `Qwen3_5Model` as
/// `MOSSImagePipeline.encode_prompt` drives it, in fp32: one unpadded forward per prompt, the output is
/// `last_hidden_state` (AFTER the final norm), every row of it.
///
///     prompt ─ chat template (user turn, generation prompt, `<think>\n`) ─ tokenizer ─ ids (≤ 4096)
///     embed_tokens(ids) ─ [S, 4096]
///     32 layers, three of every four a Gated DeltaNet, the fourth a full attention (3, 7, …, 31):
///       x += mixer(RMS₁(x))        x += SwiGLU(RMS₂(x))        RMS = x̂ · (1 + w), eps 1e-6
///     RMS(x) (the final norm) ─ [S, 4096]
///
/// **A DeltaNet layer** (`Qwen3_5GatedDeltaNet`, 16 key heads and 32 value heads of 128):
///
///     qkv = W_qkv n  (8192 = 2048 | 2048 | 4096)     z = W_z n  (4096)     b = W_b n, a = W_a n  (32)
///     q|k|v = SiLU(causal depthwise conv₄(qkv))                              per channel, kernel 4
///     q̂ = l2(q)/√128, k̂ = l2(k)                                              per head, eps 1e-6
///     β = σ(b)      g = −exp(A_log) · softplus(a + dt_bias)                  per value head, fp32
///     per value head h (reads key head h/2), state S ∈ ℝ^{128×128}, from S = 0:
///         S ← exp(g_t) · S       δ = β_t · (v_t − Sᵀ k̂_t)       S ← S + k̂_t δᵀ       o_t = Sᵀ q̂_t
///     mixer = W_out( w ⊙ RMS(o) ⊙ SiLU(z) )                                  per value head, w NOT zero-centred
///
/// **A full-attention layer** (`Qwen3_5Attention`, 16 query heads, 4 key/value heads, head 256):
///
///     [q | gate] = W_q n  (per head: 256 q then 256 gate)    k = W_k n    v = W_v n
///     q = RMS_head(q)·(1 + w_q)    k = RMS_head(k)·(1 + w_k)    RoPE on the first 64 of the 256 dims
///     mixer = W_o( SDPA_causal(q, k, v) ⊙ σ(gate) )
///
/// ## What sets it apart, and what a port by analogy would miss
///
///   - **The recurrence IS the layer.** The reference runs the sequence through `torch_chunk_gated_delta_rule`
///     (chunks of 64, a triangular solve), which is the same function as the token-by-token recurrence
///     above — in exact arithmetic. This port runs the recurrence (a few `sgemv`/`sger` of 128 × 128 per
///     token and head, 32 heads in parallel): no padding to a chunk, no solve, and a test holds the
///     chunked formulation against it (`Qwen35Tests`). The state is per layer and starts at zero: a prompt
///     is one sequence, nothing is cached across renders.
///   - **Every RMSNorm is zero-centred** (`x̂ · (1 + w)`) except the delta rule's gated norm
///     (`w · x̂ · SiLU(z)`, `Qwen3_5RMSNormGated`): the forge folds the `+ 1` (`ForgeQwen35`), so the engine
///     multiplies by the weight everywhere and the gated norm is the exception that is NOT folded.
///   - **`q_proj` is twice as wide as the attention** and its output is `[q | gate]` PER HEAD
///     (`view(S, heads, 2·headDim)` then `chunk(2)`), not `[all q | all gates]`.
///   - **The RoPE is partial and not interleaved**: `partial_rotary_factor` 0.25 rotates the first 64 of
///     256 dims of q and k with `rotate_half` (pair `j` with `j + 32`), the rest passes through; θ = 10⁷
///     (`rope_parameters`, nested in the config); `inv_freq = 1/θ^(2j/64)` — the divisor is the ROTARY
///     dimension, not the head's. The mRoPE (`mrope_section` 11/11/10, interleaved) is dead weight for
///     text: the three position axes are equal (`arange`), so the recomposition changes nothing.
///   - **The output gate σ(gate) multiplies BEFORE `o_proj`**, on the attention output, per channel.
///   - **The conv is causal with zero history**: `F.conv1d(padding = 3)` truncated to `S` — row `t` reads
///     rows `t − 3 … t`, the missing ones are zeros. It runs on the concatenation `[q | k | v]`, so the SiLU
///     is applied to q, k and v alike, BEFORE the l2 norms.
///   - **The key heads are repeated by `repeat_interleave`** (value head `h` ← key head `h / 2`), not tiled.
///   - **`softplus` has torch's threshold** (x > 20 is the identity), `g` and `β` are fp32 whatever the model's dtype.
///   - **The tokenizer is `Qwen2Tokenizer`** (`tokenizer.json`'s pre-tokenizer, `\p{L}+`), not the
///     `Qwen3_5Tokenizer` class of transformers 5 (`\p{L}\p{M}+`): MOSS-Image publishes the former. The
///     template has NO system turn and ends `<|im_start|>assistant\n<think>\n` (`enable_thinking=True`);
///     nothing is dropped, `truncation=True, max_length=4096` cuts the END of the sequence.
///   - **The checkpoint is a whole `Qwen3_5Model`** (`language_model.` + `visual.`): the vision tower is never read.
///
/// The 16.8 GB of bf16 weights exceed the budget on 16 GB: they stream layer by layer through
/// `TailStream`'s units, in the order `ForgeQwen35.deltaOrder`/`fullOrder` — the regime of `TextEncoder`.
package final class Qwen35Encoder {

    package struct Config {
        package let hidden, layers, heads, kvHeads, headDim, intermediate: Int
        package let eps: Float, theta: Float
        /// The rotated dimensions of a head (`head_dim · partial_rotary_factor`).
        package let rotaryDim: Int
        package let keyHeads, valueHeads, keyHeadDim, valueHeadDim, convKernel: Int
        /// Per layer: `true` for a full attention, `false` for a Gated DeltaNet.
        package let fullLayer: [Bool]

        package init(header: [String: Any]) throws {
            guard let c = header["config"] as? [String: Any],
                  let hidden = c["hidden_size"] as? Int, let layers = c["num_hidden_layers"] as? Int,
                  let heads = c["num_attention_heads"] as? Int, let kvHeads = c["num_key_value_heads"] as? Int,
                  let headDim = c["head_dim"] as? Int, let intermediate = c["intermediate_size"] as? Int,
                  let keyHeads = c["linear_num_key_heads"] as? Int, let valueHeads = c["linear_num_value_heads"] as? Int,
                  let keyDim = c["linear_key_head_dim"] as? Int, let valueDim = c["linear_value_head_dim"] as? Int,
                  let kernel = c["linear_conv_kernel_dim"] as? Int, let types = c["layer_types"] as? [String],
                  types.count == layers else {
                throw Artifact.Failure.badHeader("incomplete Qwen3.5 config")
            }
            guard c["attn_output_gate"] as? Bool ?? true else {
                throw Artifact.Failure.badHeader("Qwen3.5: only the attention with an output gate is ported")
            }
            guard c["attention_bias"] as? Bool != true, c["hidden_act"] as? String ?? "silu" == "silu" else {
                throw Artifact.Failure.badHeader("Qwen3.5: attention bias or an activation other than SiLU")
            }
            guard types.allSatisfy({ $0 == "linear_attention" || $0 == "full_attention" }) else {
                throw Artifact.Failure.badHeader("Qwen3.5: layer types \(Set(types))")
            }
            guard heads % kvHeads == 0, valueHeads % keyHeads == 0, kernel >= 1 else {
                throw Artifact.Failure.badHeader("Qwen3.5: GQA \(heads)/\(kvHeads), delta heads \(keyHeads)/\(valueHeads)")
            }
            let rope = c["rope_parameters"] as? [String: Any] ?? [:]
            guard (rope["rope_type"] as? String ?? "default") == "default" else {
                throw Artifact.Failure.badHeader("Qwen3.5 RoPE \"\(rope["rope_type"] ?? "")\": only the default is ported")
            }
            let factor = rope["partial_rotary_factor"] as? Double ?? c["partial_rotary_factor"] as? Double ?? 1
            let rotary = Int(Double(headDim) * factor)
            guard rotary > 0, rotary % 2 == 0, rotary <= headDim else {
                throw Artifact.Failure.badHeader("Qwen3.5: rotary dimension \(rotary) of a head of \(headDim)")
            }
            self.hidden = hidden; self.layers = layers; self.heads = heads; self.kvHeads = kvHeads
            self.headDim = headDim; self.intermediate = intermediate
            self.eps = Float(c["rms_norm_eps"] as? Double ?? 1e-6)
            self.theta = Float(rope["rope_theta"] as? Double ?? c["rope_theta"] as? Double ?? 10_000_000)
            self.rotaryDim = rotary
            self.keyHeads = keyHeads; self.valueHeads = valueHeads
            self.keyHeadDim = keyDim; self.valueHeadDim = valueDim; self.convKernel = kernel
            self.fullLayer = types.map { $0 == "full_attention" }
        }

        package var attentionWidth: Int { heads * headDim }
        package var kvWidth: Int { kvHeads * headDim }
        package var groupSize: Int { heads / kvHeads }
        package var keyWidth: Int { keyHeads * keyHeadDim }
        package var valueWidth: Int { valueHeads * valueHeadDim }
        /// The channels the delta layer convolves: `[q | k | v]`.
        package var convChannels: Int { 2 * keyWidth + valueWidth }
    }

    package let config: Config
    package let sequence: Int
    private let artifact: Artifact
    private let gemm: GEMM
    private let arena: Arena
    private let attention: Attention
    private let prefetcher: Prefetcher
    private let products: WeightProducts
    static let layersAhead = 3

    private let x, normed, projected, mixed, convolved, z, core, q, k, v, kWide, vWide, attended: UnsafeMutablePointer<Float>
    private let gate, up, swiglu, rope, scratch, reserve, decay, beta, rawA, rawB, state: UnsafeMutablePointer<Float>
    private let xBuf, normedBuf, projectedBuf, mixedBuf, zBuf, coreBuf, qBuf, kBuf, vBuf: MTLBuffer
    private let kWideBuf, vWideBuf, attendedBuf, gateBuf, upBuf: MTLBuffer

    /// The probes, on request (checks): `embed_out`, `layerN_in`, `layerN_input_norm`, `layerN_mixer`, `layerN_out`, `final_norm_out`, and inside a layer
    /// `layerN_qkv`/`_core`/`_gated` (delta) or `layerN_q_proj`/`_attn_gated` (full).
    package var recordedNames: Set<String>?
    package private(set) var boundaries: [String: [Float]] = [:]
    package var cancellation: Cancellation?

    /// - Parameter sequence: the number of positions computed (the prompt's tokens, ≤ 4096).
    package init(artifact: Artifact, sequence: Int, freezeCut: Bool = EngineSettings.effective.frozenCut) throws {
        let config = try Config(header: artifact.header)
        let gemm = try GEMM(freezeCut: freezeCut, neuralEngine: false)
        let s = sequence, h = config.hidden, i = config.intermediate
        let a = config.attentionWidth, kv = config.kvWidth
        let arena = try Arena(capacity: Qwen35Encoder.arenaBytes(config: config, sequence: s))
        var slices: [String: UnsafeMutablePointer<Float>] = [:]
        for (name, count) in Qwen35Encoder.slices(config: config, sequence: s) {
            slices[name] = try arena.reserve(name, bytes: count * 4).assumingMemoryBound(to: Float.self)
        }
        func slice(_ name: String) -> UnsafeMutablePointer<Float> { slices[name]! }
        self.config = config
        self.sequence = sequence
        self.artifact = artifact
        self.gemm = gemm
        self.arena = arena
        self.prefetcher = Prefetcher(artifact: artifact)
        x = slice("x"); normed = slice("normed"); projected = slice("projected"); mixed = slice("mixed")
        convolved = slice("convolved"); z = slice("z"); core = slice("core"); q = slice("q"); k = slice("k"); v = slice("v")
        kWide = slice("kWide"); vWide = slice("vWide"); attended = slice("attended")
        gate = slice("gate"); up = slice("up"); swiglu = slice("swiglu"); rope = slice("rope")
        scratch = slice("scratch"); reserve = slice("reserve"); decay = slice("decay"); beta = slice("beta")
        rawA = slice("rawA"); rawB = slice("rawB"); state = slice("state")
        attention = Attention(device: gemm.device, queue: gemm.queue, heads: config.heads, sequence: s,
                              headDim: config.headDim, causal: true)
        xBuf = try gemm.wrap(x, bytes: s * h * 4, name: "x")
        normedBuf = try gemm.wrap(normed, bytes: s * h * 4, name: "normed")
        projectedBuf = try gemm.wrap(projected, bytes: s * h * 4, name: "projected")
        mixedBuf = try gemm.wrap(mixed, bytes: s * max(config.convChannels, 2 * a) * 4, name: "mixed")
        coreBuf = try gemm.wrap(core, bytes: s * config.valueWidth * 4, name: "core")
        zBuf = try gemm.wrap(z, bytes: s * config.valueWidth * 4, name: "z")
        qBuf = try gemm.wrap(q, bytes: s * a * 4, name: "q")
        kBuf = try gemm.wrap(k, bytes: s * kv * 4, name: "k")
        vBuf = try gemm.wrap(v, bytes: s * kv * 4, name: "v")
        kWideBuf = try gemm.wrap(kWide, bytes: s * a * 4, name: "kWide")
        vWideBuf = try gemm.wrap(vWide, bytes: s * a * 4, name: "vWide")
        attendedBuf = try gemm.wrap(attended, bytes: s * a * 4, name: "attended")
        gateBuf = try gemm.wrap(gate, bytes: s * i * 4, name: "gate")
        upBuf = try gemm.wrap(up, bytes: s * i * 4, name: "up")
        products = WeightProducts(gemm: gemm, artifact: artifact, reserve: reserve, capacity: h * i, transposed: true)
        Qwen35Encoder.writeRope(into: rope, count: s, theta: config.theta, rotaryDim: config.rotaryDim)
    }

    // MARK: - Sizes

    /// **The arena's bytes for `sequence` positions**, as `Arena` maps it — what `init` sizes it with and what a
    /// memory need charges: the sum of `slices`, each on its page.
    package static func arenaBytes(config: Config, sequence s: Int) -> Int {
        Arena.capacity(slices: slices(config: config, sequence: s).map { $0.count * 4 })
    }

    /// **The slices, in floats, in the order `init` reserves them.** The delta layer's `mixed` (`[q|k|v]`
    /// before the convolution) is the full attention's `[q | gate]` (`2 · heads · headDim`, the same 8192
    /// here); `z` holds the delta layer's gate and the full attention's output gate; `scratch` takes the
    /// largest small tensor (`in_proj_b`: hidden × 32); `reserve` the largest weight (`hidden × intermediate`).
    package static func slices(config c: Config, sequence s: Int) -> [(name: String, count: Int)] {
        let h = c.hidden, a = c.attentionWidth, kv = c.kvWidth, i = c.intermediate
        let wide = max(c.convChannels, 2 * a)
        let small = max(h * c.valueHeads, c.convChannels * c.convKernel, h)
        return [("x", s * h), ("normed", s * h), ("projected", s * h), ("mixed", s * wide),
                ("convolved", s * c.convChannels), ("z", s * max(c.valueWidth, a)), ("core", s * c.valueWidth),
                ("q", s * a), ("k", s * kv), ("v", s * kv), ("kWide", s * a), ("vWide", s * a), ("attended", s * a),
                ("gate", s * i), ("up", s * i), ("swiglu", s * i), ("rope", s * c.rotaryDim),
                ("scratch", small), ("reserve", h * i),
                ("decay", s * c.valueHeads), ("beta", s * c.valueHeads), ("rawA", s * c.valueHeads),
                ("rawB", s * c.valueHeads), ("state", c.valueHeads * c.keyHeadDim * c.valueHeadDim)]
    }

    /// The names of layer `l` in the order `encode` widens them (the stream's order).
    package static func consumption(layer l: Int, config: Config) -> [String] {
        (config.fullLayer[l] ? ForgeQwen35.fullOrder : ForgeQwen35.deltaOrder).map { "layers.\(l).\($0)" }
    }

    // MARK: - The pure parts

    /// `[count, rotaryDim/2, (cos, sin)]` — `Qwen3_5TextRotaryEmbedding` for a text prompt (the three axes equal
    /// to the index): `inv_freq` in fp32 over the ROTARY dimension, `position.float() · inv_freq`, then cos and sin.
    package static func writeRope(into out: UnsafeMutablePointer<Float>, count: Int, theta: Float, rotaryDim: Int) {
        let half = rotaryDim / 2
        let inverse = TextEncoder.inverseFrequencies(theta: theta, dim: rotaryDim)
        for s in 0..<count {
            let row = out + s * rotaryDim
            for j in 0..<half {
                let angle = Float(s) * inverse[j]
                row[2 * j] = cosf(angle); row[2 * j + 1] = sinf(angle)
            }
        }
    }

    /// `rotate_half` on the first `rotaryDim` dims of each head: `(x₁, x₂) → (x₁ cos − x₂ sin, x₂ cos + x₁ sin)`,
    /// `x₁ = x[j]`, `x₂ = x[j + rotaryDim/2]`; the other dims untouched. `x`: `[S, heads, headDim]`.
    package static func applyRope(_ x: UnsafeMutablePointer<Float>, table: UnsafePointer<Float>, sequence: Int,
                                  heads: Int, headDim: Int, rotaryDim: Int) {
        let half = rotaryDim / 2
        Parallel.rows(sequence, width: heads * headDim) { first, howMany in
            for s in first..<(first + howMany) {
                let t = table + s * rotaryDim
                for h in 0..<heads {
                    let row = x + (s * heads + h) * headDim
                    for j in 0..<half {
                        let c = t[2 * j], sn = t[2 * j + 1]
                        let a = row[j], b = row[j + half]
                        row[j] = a * c - b * sn
                        row[j + half] = b * c + a * sn
                    }
                }
            }
        }
    }

    /// `softplus(x)` as torch computes it: the identity above 20.
    @inline(__always) package static func softplus(_ x: Float) -> Float { x > 20 ? x : log1pf(expf(x)) }

    /// **The causal depthwise convolution and its SiLU**, `x`/`out`: `[S, channels]`, `weight`: `[channels, kernel]`.
    /// Row `t` reads rows `t − kernel + 1 … t` (zeros before the first), tap `j` of the kernel on row `t − kernel + 1 + j`.
    package static func causalConvolution(_ x: UnsafePointer<Float>, weight: UnsafePointer<Float>, into out: UnsafeMutablePointer<Float>,
                                          sequence: Int, channels: Int, kernel: Int) {
        Parallel.rows(sequence, width: channels * kernel) { first, howMany in
            for t in first..<(first + howMany) {
                let o = out + t * channels
                for c in 0..<channels {
                    var acc: Float = 0
                    for j in 0..<kernel {
                        let r = t - kernel + 1 + j
                        if r >= 0 { acc += weight[c * kernel + j] * x[r * channels + c] }
                    }
                    o[c] = acc / (1 + expf(-acc))
                }
            }
        }
    }

    /// `x ← x · rsqrt(Σx² + eps)` on each head of `[S, heads, dim]` (`l2norm`, eps INSIDE the root), times `scale`.
    package static func l2NormalizeHeads(_ x: UnsafeMutablePointer<Float>, rowStride: Int, sequence: Int, heads: Int,
                                         dim: Int, scale: Float, eps: Float = 1e-6) {
        Parallel.rows(sequence, width: heads * dim) { first, howMany in
            for s in first..<(first + howMany) {
                for h in 0..<heads {
                    let p = x + s * rowStride + h * dim
                    var sum: Float = 0
                    vDSP_svesq(p, 1, &sum, vDSP_Length(dim))
                    var factor = scale / (sum + eps).squareRoot()
                    vDSP_vsmul(p, 1, &factor, p, 1, vDSP_Length(dim))
                }
            }
        }
    }

    /// **The gated delta rule, token by token** (`torch_recurrent_gated_delta_rule`), for the whole sequence.
    ///
    /// `q`, `k`: `kHeads` heads of `keyDim` at `rowStride` floats per row (already normalised, `q` already scaled);
    /// `v`: `vHeads` heads of `valueDim` at the same stride; `decay` (`g`, log space) and `beta`: `[S, vHeads]`;
    /// `out`: `[S, vHeads · valueDim]`; `state`: `[vHeads, keyDim, valueDim]`, zeroed here (a prompt starts from
    /// nothing). Value head `h` reads key head `h / (vHeads / kHeads)`. The heads are independent: they run in parallel.
    package static func gatedDeltaRule(q: UnsafePointer<Float>, k: UnsafePointer<Float>, v: UnsafePointer<Float>,
                                       decay: UnsafePointer<Float>, beta: UnsafePointer<Float>,
                                       out: UnsafeMutablePointer<Float>, state: UnsafeMutablePointer<Float>,
                                       rowStride: Int, sequence: Int, kHeads: Int, vHeads: Int, keyDim: Int, valueDim: Int) {
        let share = vHeads / kHeads
        let size = keyDim * valueDim
        state.update(repeating: 0, count: vHeads * size)
        // `width` is a floor on the work per head, so that the heads are always split across the cores.
        Parallel.rows(vHeads, width: 1 << 20) { first, howMany in
            let memory = UnsafeMutablePointer<Float>.allocate(capacity: 2 * valueDim)
            defer { memory.deallocate() }
            let delta = memory + valueDim
            for head in first..<(first + howMany) {
                let S = state + head * size
                let keyHead = head / share
                for t in 0..<sequence {
                    let qt = q + t * rowStride + keyHead * keyDim
                    let kt = k + t * rowStride + keyHead * keyDim
                    let vt = v + t * rowStride + head * valueDim
                    let ot = out + (t * vHeads + head) * valueDim
                    var forget = expf(decay[t * vHeads + head])
                    vDSP_vsmul(S, 1, &forget, S, 1, vDSP_Length(size))
                    // kv_mem = Sᵀ k
                    cblas_sgemv(CblasRowMajor, CblasTrans, Int32(keyDim), Int32(valueDim), 1, S, Int32(valueDim), kt, 1, 0, memory, 1)
                    // δ = β (v − kv_mem)   (`vDSP_vsub(b, a)` is a − b)
                    vDSP_vsub(memory, 1, vt, 1, delta, 1, vDSP_Length(valueDim))
                    var b = beta[t * vHeads + head]
                    vDSP_vsmul(delta, 1, &b, delta, 1, vDSP_Length(valueDim))
                    // S += k δᵀ
                    cblas_sger(CblasRowMajor, Int32(keyDim), Int32(valueDim), 1, kt, 1, delta, 1, S, Int32(valueDim))
                    // o = Sᵀ q
                    cblas_sgemv(CblasRowMajor, CblasTrans, Int32(keyDim), Int32(valueDim), 1, S, Int32(valueDim), qt, 1, 0, ot, 1)
                }
            }
        }
    }

    /// `w ⊙ RMS(core) ⊙ SiLU(z)` per head of `valueDim` (`Qwen3_5RMSNormGated`): the weight is used as is.
    package static func gatedNorm(_ core: UnsafeMutablePointer<Float>, z: UnsafePointer<Float>, weight: UnsafePointer<Float>,
                                  rows: Int, dim: Int, eps: Float) {
        Parallel.rows(rows, width: dim) { first, howMany in
            for r in first..<(first + howMany) {
                let c = core + r * dim, g = z + r * dim
                var sum: Float = 0
                vDSP_svesq(c, 1, &sum, vDSP_Length(dim))
                let scale = 1 / (sum / Float(dim) + eps).squareRoot()
                for j in 0..<dim {
                    let gate = g[j]
                    c[j] = weight[j] * (c[j] * scale) * (gate / (1 + expf(-gate)))
                }
            }
        }
    }

    // MARK: - The forward pass

    private func record(_ name: String, _ pointer: UnsafePointer<Float>, _ count: Int) {
        guard let names = recordedNames, names.contains(name) else { return }
        boundaries[name] = Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func widen(_ name: String, count: Int, into destination: UnsafeMutablePointer<Float>) throws {
        let got = try artifact.materialize(name, into: destination, capacity: count)
        guard got == count else { throw Artifact.Failure.badHeader("\(name): \(got) values, \(count) expected") }
    }

    /// `RMSNorm` over each head's `headDim`, weight already `1 + w` (the forge folded it).
    private func normHeads(_ t: UnsafeMutablePointer<Float>, weight: UnsafePointer<Float>, heads: Int) {
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

    private func expandKV(_ source: UnsafePointer<Float>, into destination: UnsafeMutablePointer<Float>) {
        let d = config.headDim, group = config.groupSize, kvHeads = config.kvHeads
        Parallel.rows(sequence, width: config.attentionWidth) { first, howMany in
            for s in first..<(first + howMany) {
                for kvHead in 0..<kvHeads {
                    let from = source + (s * kvHeads + kvHead) * d
                    for repetition in 0..<group {
                        (destination + (s * (kvHeads * group) + kvHead * group + repetition) * d).update(from: from, count: d)
                    }
                }
            }
        }
    }

    /// `y[S, n] = x[S, k] · W[k, n]` for a weight too thin for the GPU path (`in_proj_b`, `in_proj_a`: n = 32),
    /// widened on the CPU into `scratch`.
    private func thinProduct(_ name: String, a: UnsafePointer<Float>, k: Int, n: Int, into out: UnsafeMutablePointer<Float>) throws {
        try widen(name, count: k * n, into: scratch)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(sequence), Int32(n), Int32(k), 1, a, Int32(k),
                    scratch, Int32(n), 0, out, Int32(n))
    }

    /// The Gated DeltaNet layer's mixer: `normed` in, `projected` out.
    private func deltaMixer(_ p: String, layer: Int) throws {
        let c = config, s = sequence, h = c.hidden
        let channels = c.convChannels, vw = c.valueWidth
        try products.linear(p + "linear_attn.in_proj_qkv.weight", a: normedBuf, c: mixedBuf, m: s, k: h, n: channels)
        try products.linear(p + "linear_attn.in_proj_z.weight", a: normedBuf, c: zBuf, m: s, k: h, n: vw)
        record("layer\(layer)_qkv", mixed, s * channels)
        try thinProduct(p + "linear_attn.in_proj_b.weight", a: normed, k: h, n: c.valueHeads, into: rawB)
        try thinProduct(p + "linear_attn.in_proj_a.weight", a: normed, k: h, n: c.valueHeads, into: rawA)

        try widen(p + "linear_attn.conv1d.weight", count: channels * c.convKernel, into: scratch)
        Qwen35Encoder.causalConvolution(mixed, weight: scratch, into: convolved, sequence: s, channels: channels, kernel: c.convKernel)

        // q = l2(q)/√d, k = l2(k), per head, in place in the convolved rows.
        Qwen35Encoder.l2NormalizeHeads(convolved, rowStride: channels, sequence: s, heads: c.keyHeads, dim: c.keyHeadDim,
                                       scale: 1 / Float(c.keyHeadDim).squareRoot())
        Qwen35Encoder.l2NormalizeHeads(convolved + c.keyWidth, rowStride: channels, sequence: s, heads: c.keyHeads,
                                       dim: c.keyHeadDim, scale: 1)

        // β = σ(b), g = −exp(A_log) · softplus(a + dt_bias)
        try widen(p + "linear_attn.A_log", count: c.valueHeads, into: scratch)
        try widen(p + "linear_attn.dt_bias", count: c.valueHeads, into: scratch + c.valueHeads)
        for t in 0..<s {
            for hv in 0..<c.valueHeads {
                let i = t * c.valueHeads + hv
                beta[i] = 1 / (1 + expf(-rawB[i]))
                decay[i] = -expf(scratch[hv]) * Qwen35Encoder.softplus(rawA[i] + scratch[c.valueHeads + hv])
            }
        }
        Qwen35Encoder.gatedDeltaRule(q: convolved, k: convolved + c.keyWidth, v: convolved + 2 * c.keyWidth, decay: decay,
                                     beta: beta, out: core, state: state, rowStride: channels, sequence: s,
                                     kHeads: c.keyHeads, vHeads: c.valueHeads, keyDim: c.keyHeadDim, valueDim: c.valueHeadDim)

        record("layer\(layer)_core", core, s * vw)
        try widen(p + "linear_attn.norm.weight", count: c.valueHeadDim, into: scratch)
        Qwen35Encoder.gatedNorm(core, z: z, weight: scratch, rows: s * c.valueHeads, dim: c.valueHeadDim, eps: c.eps)
        record("layer\(layer)_gated", core, s * vw)
        try products.linear(p + "linear_attn.out_proj.weight", a: coreBuf, c: projectedBuf, m: s, k: vw, n: h)
    }

    /// The full attention layer's mixer: `normed` in, `projected` out.
    private func fullMixer(_ p: String, layer: Int) throws {
        let c = config, s = sequence, h = c.hidden, a = c.attentionWidth, kv = c.kvWidth, d = c.headDim
        try products.linear(p + "self_attn.q_proj.weight", a: normedBuf, c: mixedBuf, m: s, k: h, n: 2 * a)
        try products.linear(p + "self_attn.k_proj.weight", a: normedBuf, c: kBuf, m: s, k: h, n: kv)
        try products.linear(p + "self_attn.v_proj.weight", a: normedBuf, c: vBuf, m: s, k: h, n: kv)
        record("layer\(layer)_q_proj", mixed, s * 2 * a)
        // `view(S, heads, 2·d).chunk(2)`: each head's row is `[q (d) | gate (d)]`.
        for t in 0..<s {
            for head in 0..<c.heads {
                let row = mixed + (t * c.heads + head) * 2 * d
                (q + (t * c.heads + head) * d).update(from: row, count: d)
                (z + (t * c.heads + head) * d).update(from: row + d, count: d)
            }
        }
        try widen(p + "self_attn.q_norm.weight", count: d, into: scratch)
        normHeads(q, weight: scratch, heads: c.heads)
        try widen(p + "self_attn.k_norm.weight", count: d, into: scratch)
        normHeads(k, weight: scratch, heads: c.kvHeads)
        Qwen35Encoder.applyRope(q, table: rope, sequence: s, heads: c.heads, headDim: d, rotaryDim: c.rotaryDim)
        Qwen35Encoder.applyRope(k, table: rope, sequence: s, heads: c.kvHeads, headDim: d, rotaryDim: c.rotaryDim)
        expandKV(k, into: kWide)
        expandKV(v, into: vWide)
        _ = attention.run(q: qBuf, k: kWideBuf, v: vWideBuf, into: attendedBuf)
        // The output gate multiplies the attention, before `o_proj`.
        Parallel.rows(s, width: a) { first, howMany in
            for t in first..<(first + howMany) {
                for j in 0..<a {
                    let g = self.z[t * a + j]
                    self.attended[t * a + j] *= 1 / (1 + expf(-g))
                }
            }
        }
        record("layer\(layer)_attn_gated", attended, s * a)
        try products.linear(p + "self_attn.o_proj.weight", a: attendedBuf, c: projectedBuf, m: s, k: a, n: h)
    }

    /// **The prompt's identifiers → `last_hidden_state`**, `[sequence, hidden]` (after the final norm; with
    /// `stopAfter` layers only, the residual stream after that many layers, un-normalised — the checks' cut).
    package func encode(ids: [Int], stopAfter: Int? = nil) throws -> [Float] {
        precondition(ids.count >= sequence, "at least \(sequence) identifiers are needed")
        let c = config, s = sequence, h = c.hidden, i = c.intermediate
        let layers = min(stopAfter ?? c.layers, c.layers)
        boundaries = [:]

        guard let table = artifact.tensors["embed_tokens.weight"] else {
            throw Artifact.Failure.badHeader("embed_tokens.weight missing from the map")
        }
        guard table.shape.count == 2, table.shape[1] == h else {
            throw Artifact.Failure.badHeader("embed_tokens: \(table.shape) instead of [V, \(h)]")
        }
        for position in 0..<s {
            let token = ids[position]
            guard token >= 0 else {
                throw Tokenizer.Failure.malformed("a piece of the prompt has no entry in the vocabulary (position \(position))")
            }
            guard token < table.shape[0] else {
                throw Artifact.Failure.badHeader("token \(token) beyond the embedding table's \(table.shape[0]) rows: "
                                                 + "the tokenizer and the map are not of the same model")
            }
            try artifact.materializeRows("embed_tokens.weight", first: token, count: 1, into: x + position * h)
        }
        record("embed_out", x, s * h)

        // The stream, as `TextEncoder` does: units in the order the layers widen them.
        let blocks = (0..<layers).map { prefetcher.namesOfBlock(prefix: "layers.\($0).") }
        let granted = TextEncoder.stagingSlots(granted: MemoryPlan.encoderSlots(artifact: artifact, layerBlocks: blocks))
        if granted == 0 {
            try artifact.streamTail(blocks: [], slots: 0)
        } else {
            try artifact.streamUnits(ordered: (0..<layers).map { Qwen35Encoder.consumption(layer: $0, config: c) }
                                     + (layers == c.layers ? [["norm.weight"]] : []))
        }
        for l in 0..<min(Qwen35Encoder.layersAhead, layers) { prefetcher.request(prefetcher.namesOfBlock(prefix: "layers.\(l).")) }
        for layer in 0..<layers {
            try cancellation.check()
            if recordedNames != nil { record("layer\(layer)_in", x, s * h) }
            let p = "layers.\(layer)."
            prefetcher.request(prefetcher.namesOfBlock(prefix: "layers.\(layer + Qwen35Encoder.layersAhead)."))

            try widen(p + "input_layernorm.weight", count: h, into: scratch)
            Ops.rmsNorm(x, weight: scratch, into: normed, rows: s, columns: h, eps: c.eps)
            record("layer\(layer)_input_norm", normed, s * h)
            if c.fullLayer[layer] { try fullMixer(p, layer: layer) } else { try deltaMixer(p, layer: layer) }
            record("layer\(layer)_mixer", projected, s * h)
            for j in 0..<(s * h) { x[j] += projected[j] }

            try widen(p + "post_attention_layernorm.weight", count: h, into: scratch)
            Ops.rmsNorm(x, weight: scratch, into: normed, rows: s, columns: h, eps: c.eps)
            try products.linear(p + "mlp.gate_proj.weight", a: normedBuf, c: gateBuf, m: s, k: h, n: i)
            try products.linear(p + "mlp.up_proj.weight", a: normedBuf, c: upBuf, m: s, k: h, n: i)
            Ops.siluGate(gate, up, into: gate, count: s * i, scratch: swiglu)
            try products.linear(p + "mlp.down_proj.weight", a: gateBuf, c: projectedBuf, m: s, k: i, n: h)
            for j in 0..<(s * h) { x[j] += projected[j] }

            record("layer\(layer)_out", x, s * h)
            artifact.dropFromCache(prefetcher.namesOfBlock(prefix: p))
        }
        if layers == c.layers {
            try widen("norm.weight", count: h, into: scratch)
            Ops.rmsNorm(x, weight: scratch, into: x, rows: s, columns: h, eps: c.eps)
            record("final_norm_out", x, s * h)
        }
        return Array(UnsafeBufferPointer(start: x, count: s * h))
    }
}

/// **The prompt side of MOSS-Image's encoder**: the chat template and the cut. Pure functions.
package enum Qwen35Prompt {
    /// `MOSSImagePipeline.encode_prompt`'s default `max_sequence_length`, and its bound (`1 ≤ n ≤ 4096`).
    package static let maxTokens = 4096

    /// `apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=True)`
    /// of the published `chat_template.jinja`: no system turn, and the generation prompt opens the thinking block.
    package static func template(_ prompt: String) -> String {
        "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n"
    }

    /// The identifiers the encoder reads: the template, tokenized, cut to `maxLength` from the END
    /// (`truncation=True`, right side) — the reference's behaviour, reproduced, not corrected.
    package static func ids(_ prompt: String, tokenizer: Tokenizer, maxLength: Int = maxTokens) -> [Int] {
        Array(tokenizer.encode(template(prompt)).prefix(maxLength))
    }
}
