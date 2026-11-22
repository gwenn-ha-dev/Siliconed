import CoreAI
import Foundation
import Metal
import SiliCore

/// **Qwen3.5-9B as MOSS-Image's text encoder** (`Qwen3_5Model`, text only): the chat template's token
/// identifiers in, `last_hidden_state` (after the final norm) out. Transcribed from transformers'
/// `models/qwen3_5/modeling_qwen3_5.py` as it runs in bf16 (`torch_dtype=bfloat16`, the pipeline's own text encoder).
///
///     ids [T] ─ embed_tokens ─ 32 layers ─ norm ─ hidden [T, 4096]
///       layer i:  x += mixer(RMS(x));   x += down(SiLU(gate n) ⊙ up n),  n = RMS(x)      mixer = full attention if (i+1) % 4 == 0
///                                                                                        else Gated DeltaNet
///     Gated DeltaNet (16 key heads, 32 value heads, 128 wide):
///       qkv ─ causal depthwise conv₄ ─ SiLU ─ q | k | v ─ l2 norm q, k ─ gated delta rule ─ RMSNormGated(·, z) ─ out_proj
///       β = σ(b)      g = −exp(A_log) · softplus(a + dt_bias)
///     Full attention (16 heads, 4 key/value heads, 256 wide):
///       [q | gate] per head ─ RMS(q), RMS(k) ─ RoPE on 64 of 256 ─ causal sdpa ─ · σ(gate) ─ o_proj
///
/// **Three functions a palier of rows, the weights their arguments** (the bf16 path, the GPU alone): one for the 24
/// DeltaNet layers, one for the 8 attention layers, one for the final norm. The prompt is padded to its palier — the
/// attention is causal, the conv causal, the delta rule causal, every other op row by row: the rows past the prompt
/// change none of its own. The token table (2 GB) stays in the file: the CPU reads the row of each token. The layers
/// (`declaration`, `SiliCore.Layers`) hold a group of weights a layer, every one kept where B holds them, else the
/// DeltaNet layers by the ring; each fused product's parts side by side along `out` as the file lays them (`[out, in]`: one read after
/// the other, `Recipe.bfloat16`).
///
/// What a port by analogy (Qwen3) would miss, and where the reference rounds:
///
///   - **Every `Qwen3_5RMSNorm` is zero-centred** (`output = norm(x.float()) * (1.0 + weight.float())`, `.type_as(x)`): one
///     rounding, after the product in f32; the file holds `w` in bf16, so the function computes `cast(norm(f32 x) · (1 + f32 w), bf16)`.
///     `Qwen3RMSNorm` (`Qwen3Encoder`) rounds twice;
///   - **`Qwen3_5RMSNormGated` is the other kind** (the delta rule's norm, `weight` plain): statistics in f32, the
///     normalized value rounded to bf16, then `weight * that` in bf16, then `* silu(gate.float())` in f32, rounded once;
///   - **`q_proj` is twice as wide**: `view(.., heads, 2·256).chunk(2)`, the gate is per head after the query, and
///     multiplies `sigmoid(gate)` into the attention output BEFORE `o_proj` (`Qwen3_5Attention.forward`);
///   - **partial, half-split RoPE**: the first 64 of 256 dims, θ = 10⁷, `inv_freq = 1 / θ^(arange(0, 64, 2) / 64)` in f32 on
///     the CPU; the mRoPE sections only move columns between three equal position axes (text: all equal), so the table is
///     the plain one (`Qwen3_5TextRotaryEmbedding.forward`); `cos.to(bf16)`, the rotation in bf16 (`apply_rotary_pos_emb`).
///     The table is a constant of the palier's function, computed on the CPU in f32 (`rotaryTables`);
///   - **the conv is causal with zero history** (`causal_conv1d_fn` falls back to `F.silu(conv1d(x)[..., :T])`, padding 3), over
///     `[q | k | v]` together, bf16 weights, the SiLU BEFORE the l2 norms: four shifted rows summed in f32, rounded once,
///     then the SiLU rounded once; key heads are `repeat_interleave`d (value head h reads key head h / 2);
///   - **decay and β**: `g = -A_log.float().exp() * softplus(a.float() + dt_bias)` with `A_log`, `dt_bias` the bf16
///     parameters widened (the file holds `A_log` and the gated norm's weight in f32: rounded to bf16 when read, as
///     `from_pretrained` does);
///   - **the gated delta rule** is `torch_chunk_gated_delta_rule` in f32 (chunk 64, `use_qk_l2norm_in_kernel`, query scaled by
///     `Dk^-½` after the l2 norm). Inside a chunk the output is `((q kᵀ) ⊙ D) (I + L)⁻¹ (β v)`,
///     `L = strictlylower((β k kᵀ) ⊙ D)` (`solve_triangular(..., unitriangular=True)` ignores the diagonal),
///     `D[i,j] = exp(cum_i − cum_j)`, the decay summed from the chunk's own start. Without a triangular solve, the
///     inverse is built block by block, 1, 2, 4… rows — every intermediate is an entry of a true
///     inverse. The series `(I − L)(I + L²)(I + L⁴)…` is the same matrix on paper and is not it in f32: on a chunk of 64
///     its powers grow and cancel (layer 20 of the long prompt off by 0.38, an `inf` five layers on). From one chunk to
///     the next a state `S` `[Hv, Dk, Dv]` is carried, unrolled in the function; the first chunk's is zero and its terms
///     vanish. The palier is a whole number of chunks: the reference pads the last chunk with zeros, the palier with
///     the pad's rows, and the real rows are the same either way (the rule is causal).
package struct Qwen35Encoder: Sendable {
    package struct Config: Sendable {
        package let width, heads, kvHeads, headDim, hidden, layers, vocabulary: Int
        package let keyHeads, valueHeads, keyHeadDim, valueHeadDim, convKernel, rotaryDim: Int
        package let theta, eps: Double

        /// Qwen3.5-9B's text model (`config.json`'s `text_config`): what the published file holds, every layer read.
        package static let published = Config(width: 4096, heads: 16, kvHeads: 4, headDim: 256, hidden: 12288, layers: 32,
                                              vocabulary: 248_320, keyHeads: 16, valueHeads: 32, keyHeadDim: 128, valueHeadDim: 128,
                                              convKernel: 4, rotaryDim: 64, theta: 1e7, eps: 1e-6)
        var keyWidth: Int { keyHeads * keyHeadDim }
        var valueWidth: Int { valueHeads * valueHeadDim }
        var convChannels: Int { 2 * keyWidth + valueWidth }
        /// The DeltaNet's input products fused: `[qkv | z | b | a]`.
        var deltaInputs: Int { convChannels + valueWidth + 2 * valueHeads }
        /// The attention's: `[q·gate | k | v]`.
        var attentionInputs: Int { 2 * heads * headDim + 2 * kvHeads * headDim }
        /// `layer_types`: every fourth layer is a full attention.
        func isFull(_ layer: Int) -> Bool { (layer + 1) % 4 == 0 }
    }

    /// `chunk_size` of `torch_chunk_gated_delta_rule`: the reference's, and part of its arithmetic (the decay is summed
    /// per chunk).
    package static let chunkTokens = 64

    /// The paliers a prompt is padded to, whole chunks: a render admits 1024 tokens of its own and the template's 14.
    package static let paliers = [64, 128, 256, 512, 1088]

    /// What the file puts before each name of the text model.
    package static let prefix = "model.language_model."

    package let config: Config

    package init(config: Config = .published) {
        self.config = config
    }

    /// The smallest palier that holds `n` tokens.
    package static func palier(_ n: Int) throws -> Int {
        guard let p = paliers.first(where: { $0 >= n }) else { throw Failure("\(n) tokens, past the largest palier \(paliers.last!)") }
        return p
    }

    /// `cos`, `sin` `[length, rotaryDim]` as the reference rounds them: f32 angles from f32 frequencies, then bf16 (returned as
    /// f32 values already on the bf16 grid, duplicated into halves).
    package func rotaryTables(length: Int) -> (cos: [Float], sin: [Float]) {
        let half = config.rotaryDim / 2
        let theta = Float(config.theta)
        let freqs = (0..<half).map { 1 / powf(theta, Float(2 * $0) / Float(config.rotaryDim)) }
        var cos: [Float] = [], sin: [Float] = []
        for position in 0..<length {
            let angles = freqs.map { Float(position) * $0 }
            for _ in 0..<2 { for a in angles { cos.append(BFloat.round(cosSin(Double(a)).cos)) } }
            for _ in 0..<2 { for a in angles { sin.append(BFloat.round(cosSin(Double(a)).sin)) } }
        }
        return (cos, sin)
    }

    /// The masks of a chunk of `t` rows, `[t, t]` (row i, column j): what the delta rule's cumulative sums, its
    /// strictly lower part and its block-by-block inverse select. `levels`: the lower-left blocks of 1, 2, 4… rows within
    /// blocks of twice that, in the order the inverse doubles.
    static func chunkMasks(_ t: Int) -> (onAndAbove: [Float], above: [Float], below: [Float], levels: [[Float]]) {
        func mask(_ f: (Int, Int) -> Float) -> [Float] { (0..<(t * t)).map { f($0 / t, $0 % t) } }
        var levels: [[Float]] = [], side = 1
        while side < t {
            let s = side
            levels.append(mask { i, j in i / (2 * s) == j / (2 * s) && i % (2 * s) >= s && j % (2 * s) < s ? 1 : 0 })
            side *= 2
        }
        return (mask { i, j in i <= j ? 1 : 0 }, mask { i, j in j > i ? -1e30 : 0 }, mask { i, j in j < i ? 1 : 0 }, levels)
    }

    // MARK: - The layers

    /// The path a file asks for: int8 `convrot` codes by 256 the half's, bf16 the Standard's.
    static func path(of file: SafeTensors) throws -> Path {
        let probe = prefix + "layers.0.mlp.gate_proj"
        guard file.tensors[probe + ".weight"]?.dtype == "I8" else { return .bfloat }
        let raw = try file.tensor(probe + ".comfy_quant", "U8", nil)
        var note = [UInt8](repeating: 0, count: raw.end - raw.start)
        try note.withUnsafeMutableBytes { try file.read(raw, into: $0.baseAddress!) }
        let quant = try JSONSerialization.jsonObject(with: Data(note)) as? [String: Any]
        guard quant?["convrot"] as? Bool == true, quant?["convrot_groupsize"] as? Int == 256 else {
            throw Failure("\(probe): codes without ConvRot's rotation by 256 (\(String(decoding: note, as: UTF8.self)))")
        }
        return .half
    }

    /// **What the encoder's layers need** on `rows` (`SiliCore.Layers`): the stream (and its factors on the half) and the
    /// final norm's output apart; the final norm's gain resident; a group a layer, used once each in order, a DeltaNet
    /// layer's the slots' layout — the ring reads the DeltaNet layers B does not keep, the 8 attention layers (another
    /// layout) are kept whatever B —; a kind of layer each mixer, and the final norm.
    func declaration(rows: Int, path: Path) -> Layers.Declaration {
        let c = config, e = self
        var d = Layers.Declaration("Qwen3.5 on \(rows) rows (\(path.suffix))", rows: rows)
        d.perRow = [.init("x", width: c.width, path.isBFloat ? .bfloat16 : .float16, next: "y", factor: path.isBFloat ? nil : "x_factor")]
        d.tensors = [.init("hidden", bytes: rows * c.width * 2, .apart)]
        d.residents = [("norm", Self.prefix + "norm.weight", .bfloat16(joined: []))]
        d.layers = (0..<c.layers).map { Block.Group("\($0)", form: c.isFull($0) ? "attention" : "delta", weights: path.isBFloat ? entries($0) : halfEntries($0)) }
        switch path {
        case .bfloat:
            d.kinds = ["delta": { b in e.deltaModule(rows: b.rows) }, "attention": { b in e.attentionModule(rows: b.rows) }]
            d.edges["norm"] = { _ in e.normModule(rows: rows) }
        case .half:
            d.perRow.append(.init("x_factor", width: 1, .float32, next: "y_factor"))
            d.kinds = ["delta": { b in e.halfDeltaModule(rows: b.rows) }, "attention": { b in e.halfAttentionModule(rows: b.rows) }]
            d.edges["norm"] = { _ in e.halfNormModule(rows: rows) }
        }
        return d
    }

    // MARK: - The run

    /// **The encoder over `ids`** (their template applied): `hidden` `[T, width]` (the final norm's output) and
    /// `layer<i>` (layer i's output, before it) for each of `taps`, as f32 values. The path is the file's (`path(of:)`).
    package func states(_ ids: [Int], map mapPath: String, taps: Set<Int> = []) throws -> [String: HostTensor] {
        precondition(!ids.isEmpty, "no token")
        let rows = try Self.palier(ids.count), e = self
        return try blocking {
            let file = try SafeTensors(mapPath)
            let path = try Self.path(of: file)
            guard let device = MTLCreateSystemDefaultDevice() else { throw Failure("no Metal device") }
            let declaration = e.declaration(rows: rows, path: path)
            let layers = try await Layers.open(device, declaration, files: [file], budget: MemoryBudget.machineCeiling())
            defer { layers.close() }
            return try await e.encode(ids, layers, file: file, rows: rows, path: path, taps: taps)
        }
    }

    /// `last_hidden_state`: what MOSS-Image's DiT reads.
    package static func encode(_ ids: [Int], map path: String) throws -> HostTensor {
        guard let hidden = try Qwen35Encoder().states(ids, map: path)["hidden"] else { throw ProgramError("the encoder gave no hidden states") }
        return hidden
    }

    private func encode(_ ids: [Int], _ layers: Layers, file: SafeTensors, rows: Int, path: Path, taps: Set<Int>) async throws -> [String: HostTensor] {
        let c = config, key = Self.prefix + "embed_tokens"
        // The token table's rows, the palier's rest the last token's.
        let padded = (0..<rows).map { $0 < ids.count ? ids[$0] : ids.last! }
        if let wrong = padded.first(where: { !(0..<c.vocabulary).contains($0) }) { throw Failure("token \(wrong) past the vocabulary") }
        let to = await layers.rows("x")
        switch path {
        case .bfloat:
            // bf16 as they lie.
            let table = try file.tensor(key + ".weight", "BF16", [c.vocabulary, c.width])
            let row = c.width * 2
            for (r, id) in padded.enumerated() { try file.read(table, at: id * row, count: row, into: to + r * row) }
        case .half:
            let table = try file.tensor(key + ".weight", "I8", [c.vocabulary, c.width])
            let scales = try file.tensor(key + ".weight_scale", "F32", [c.vocabulary, 1])
            let values = try Self.rotatedRows(padded, table: table, scales: scales, file: file, width: c.width)
            Self.writeStream(values, width: c.width, to: to, factors: await layers.rows("x_factor").assumingMemoryBound(to: Float.self))
        }
        var out: [String: HostTensor] = [:]
        try await layers.forward(after: { i in
            if taps.contains(i) { out["layer\(i)"] = HostTensor([ids.count, c.width], floats: await layers.values("x", rows: 0..<ids.count)) }
        })
        try await layers.call("norm")
        // The norm's output: true values, bf16 or f16.
        let n = ids.count * c.width, p = await layers.pointer("hidden")
        out["hidden"] = HostTensor([ids.count, c.width], floats: path.isBFloat ? (0..<n).map { BFloat16Bits.widened(p.assumingMemoryBound(to: UInt16.self)[$0]) }
                                                                                : (0..<n).map { Float(p.assumingMemoryBound(to: Float16.self)[$0]) })
        return out
    }
}
