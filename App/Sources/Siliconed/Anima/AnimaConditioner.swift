import CoreAI
import Foundation
import Metal
import SiliCore

/// **Anima's text conditioner (the "LLM adapter")**: Qwen3-0.6B's states and the T5 token identifiers
/// in, the DiT's `[max(L, 512), 1024]` context out. Transcribed from diffusers' `AnimaTextConditioner`
/// (`condition_embedder_anima.py`) as it runs in bf16.
///
///     t5 ids [Lt] ─ embed ─ blocks ×6 ─ out_proj ─ RMSNorm ─ zero rows up to 512 ─ prompt_embeds
///
///     block:  h += self_attn( RMS(h) )                       q, k from the T5 side, RoPE on both, positions 0 … Lt−1
///             h += cross_attn( RMS(h), qwen [Lq, 1024] )     q on 0 … Lt−1, k on 0 … Lq−1 (the Qwen side's own positions)
///             h += W₂ GELU( W₁ RMS(h) + b₁ ) + b₂
///
/// What a port by analogy would miss (read in `condition_embedder_anima.py`):
///
///   - **The queries are the T5 token embeddings; the Qwen states are only keys and values.** The output has the T5
///     sequence's length (padded to 512 with zero rows), not the Qwen one's.
///   - **Every norm is torch's `nn.RMSNorm`, not diffusers' `RMSNorm`**: `F.rms_norm` on bf16 rounds once — the
///     normalized value times γ in f32, then bf16 — where diffusers' (and transformers') round the normalized value first
///     and then multiply (measured: 1 bf16 ulp apart on random rows). QK-norm per head (64) included, ε = 10⁻⁶.
///   - **The rotary table is `rotate_half`'s, on the original q/k layout** (the published weights are used as
///     they are): `x · cos + rotate_half(x) · sin` on halves of the 64-wide head, in bf16 — three
///     roundings. cos and sin are f32 (`inv_freq` stays f32 under `from_pretrained(torch_dtype=bf16)`) brought to bf16
///     (`cos.to(hidden_states.dtype)`); they are computed here on the CPU and given.
///   - **Cross-attention takes its keys' RoPE from the source positions** (`source_position_embeddings`), the queries' from
///     the target's; with one prompt both sequences are whole and no mask exists (`attention_mask` all true is `None`).
///   - **A plain GELU** (`nn.GELU()`, erf) between two `Linear`s *with* bias; the attention projections have none.
///   - **No `in_proj`** in the published adapter (`model_dim == target_dim`): the embeddings enter the blocks as they are.
package struct AnimaConditioner {
    package let targetLength, sourceLength: Int
    package let dim = 1024, heads = 16, layers = 6, vocabulary = 32128, minimumLength = 512, eps = 1e-6
    package var headDim: Int { dim / heads }
    /// The context's width.
    package static let width = 1024

    package init(targetLength: Int, sourceLength: Int) {
        self.targetLength = targetLength
        self.sourceLength = sourceLength
    }

    /// The conditioner's names are the module paths of the `AnimaTextConditioner`; the DiT's file, read under its prefix
    /// (`AnimaEvaluation.file`), files them under `llm_adapter.`.
    package static func mapKey(_ name: String) -> String { "llm_adapter." + name }

    /// The rows the context has: the T5 sequence, at least 512.
    package var rows: Int { max(targetLength, minimumLength) }

    // MARK: - The CPU's tables

    /// **`AnimaRotaryEmbedding`'s cos and sin for the positions `0 ..< length`**, `[length, 64]` each: `inv_freq =
    /// 1/θ^(2i/64)` (θ = 10⁴), the angle `position · inv_freq` in float32, `[angles, angles]`, then cos and sin, brought to
    /// bf16 (`cos.to(hidden_states.dtype)`) — as float values.
    package static func rotaryTables(length: Int, headDim: Int = 64) -> (cos: [Float], sin: [Float]) {
        let frequencies = (0..<(headDim / 2)).map { Float(1 / pow(10_000.0, Double(2 * $0) / Double(headDim))) }
        var cos: [Float] = [], sin: [Float] = []
        cos.reserveCapacity(length * headDim); sin.reserveCapacity(length * headDim)
        for position in 0..<length {
            let angles = frequencies.map { Float(position) * $0 }
            for a in angles + angles {
                let (c, s) = cosSin(Double(a))
                cos.append(BFloat.round(c))
                sin.append(BFloat.round(s))
            }
        }
        return (cos, sin)
    }

    // MARK: - On the engine

    /// The function's sides: the T5 and Qwen sequences at a palier of 64, their padding masked (−10⁴ on its keys).
    package static let palier = 64
    package var targetRows: Int { (targetLength + Self.palier - 1) / Self.palier * Self.palier }
    package var sourceRows: Int { (sourceLength + Self.palier - 1) / Self.palier * Self.palier }

    /// The adapter's weights as the function reads them: products `[in, out]` bf16, norms and biases as stored.
    static func entries() -> [Weights.Entry] {
        var all: [Weights.Entry] = []
        for i in 0..<6 {
            let p = "blocks.\(i)."
            for n in ["norm_self_attn", "norm_cross_attn", "norm_mlp"] { all.append(("b\(i).\(n)", mapKey(p + n + ".weight"), .bfloat16(joined: []))) }
            for a in ["self_attn", "cross_attn"] {
                for w in ["q_proj", "k_proj", "v_proj", "o_proj"] { all.append(("b\(i).\(a).\(w)", mapKey(p + a + "." + w + ".weight"), .bfloatTransposed)) }
                for n in ["q_norm", "k_norm"] { all.append(("b\(i).\(a).\(n)", mapKey(p + a + "." + n + ".weight"), .bfloat16(joined: []))) }
            }
            for l in ["0", "2"] {
                all.append(("b\(i).mlp\(l)", mapKey(p + "mlp.\(l).weight"), .bfloatTransposed))
                all.append(("b\(i).mlp\(l)_bias", mapKey(p + "mlp.\(l).bias"), .bfloat16(joined: [])))
            }
        }
        return all + [("out_proj", mapKey("out_proj.weight"), .bfloatTransposed), ("out_proj_bias", mapKey("out_proj.bias"), .bfloat16(joined: [])),
                      ("norm", mapKey("norm.weight"), .bfloat16(joined: []))]
    }

    /// The asset's name: its sides.
    func asset(taps: Bool) -> String { "anima-conditioner-\(targetRows)x\(sourceRows)\(taps ? "-taps" : "")-bf16" }

    /// **The conditioner in one function**, in bf16 as the reference runs it: `h` the T5 embeddings `[T, 1024]` (gathered
    /// on the CPU), `source` Qwen's states `[S, 1024]`, the masks of their padding, the rotary tables; `context` `[T, 1024]`
    /// f32 out (bf16 values), with `cond_block0_out` and `cond_block5_out` under `taps`.
    func module(taps: Bool) -> Aicode.Module {
        let (t, s, hd) = (targetRows, sourceRows, headDim)
        var arguments: [(String, Aicode.Entry)] = [("h", Aicode.tensor([t, dim], .bf16)), ("source", Aicode.tensor([s, dim], .bf16)),
                                                   ("t_mask", Aicode.tensor([1, t], .bf16)), ("s_mask", Aicode.tensor([1, s], .bf16)),
                                                   ("cos_t", Aicode.tensor([t, 1, hd], .bf16)), ("sin_t", Aicode.tensor([t, 1, hd], .bf16)),
                                                   ("cos_s", Aicode.tensor([s, 1, hd], .bf16)), ("sin_s", Aicode.tensor([s, 1, hd], .bf16))]
        for (name, _, _) in Self.entries() {
            let shape: [Int]
            switch name.split(separator: ".").last! {
            case "q_proj", "k_proj", "v_proj", "o_proj", "out_proj": shape = [dim, dim]
            case "mlp0": shape = [dim, 4 * dim]
            case "mlp2": shape = [4 * dim, dim]
            case "mlp0_bias": shape = [4 * dim]
            case "q_norm", "k_norm": shape = [hd]
            default: shape = [dim]
            }
            arguments.append((name, Aicode.tensor(shape, .bf16)))
        }
        var m = Aicode.Module(arguments: arguments)
        let b = m.body, names = m.argumentNames
        func a(_ name: String) -> Int { names.firstIndex(of: name)! }
        func linear(_ x: Int, _ w: String, bias: String? = nil) -> Int {
            let rows = Aicode.shape(b.types[x]).shape[0], out = Aicode.shape(b.types[a(w)]).shape[1]
            let y = b.matmul(x, a(w), result: Aicode.tensor([rows, out], .bf16))
            return bias.map { b.binary(.add, y, a($0)) } ?? y
        }
        /// `nn.RMSNorm` on bf16: the normalized value times γ in f32, rounded once.
        func rms(_ x: Int, _ gain: String, axis: Int) -> Int {
            b.cast(b.binary(.mul, b.rmsNorm(b.cast(x, to: .f32), axes: [axis], eps: eps), b.cast(a(gain), to: .f32)), to: .bf16)
        }
        /// `x · cos + rotate_half(x) · sin`, each operation a bf16 rounding.
        func rotated(_ x: Int, _ cos: String, _ sin: String) -> Int {
            let half = hd / 2
            let x0 = b.slice(x, axis: 2, start: 0, size: half), x1 = b.slice(x, axis: 2, start: half, size: half)
            let turned = b.concat([b.scale("mul", x1, -1), x0], axis: 2)
            return b.binary(.add, b.binary(.mul, x, a(cos)), b.binary(.mul, turned, a(sin)))
        }
        func headsFirst(_ x: Int, _ l: Int) -> Int { b.reshape(b.transpose(x, [1, 0, 2]), [1, heads, l, hd]) }
        func attention(_ x: Int, _ p: String, source: Int?, _ l: Int) -> Int {
            let (src, lk) = source.map { ($0, s) } ?? (x, t)
            let (kc, ks, mask) = source == nil ? ("cos_t", "sin_t", "t_mask") : ("cos_s", "sin_s", "s_mask")
            let q = rotated(rms(b.reshape(linear(x, p + ".q_proj"), [l, heads, hd]), p + ".q_norm", axis: 2), "cos_t", "sin_t")
            let k = rotated(rms(b.reshape(linear(src, p + ".k_proj"), [lk, heads, hd]), p + ".k_norm", axis: 2), kc, ks)
            let v = b.reshape(linear(src, p + ".v_proj"), [lk, heads, hd])
            let o = b.attention([headsFirst(q, l), headsFirst(k, lk), headsFirst(v, lk), a(mask)], scale: 1 / Double(hd).squareRoot(),
                                symbol: (source == nil ? "sdpa_self_" : "sdpa_cross_") + p.prefix { $0 != "." })
            return linear(b.reshape(b.transpose(b.reshape(o, [heads, l, hd]), [1, 0, 2]), [l, dim]), p + ".o_proj")
        }
        var h = a("h")
        var outputs: [(String, Int)] = []
        for i in 0..<layers {
            let p = "b\(i)"
            h = b.binary(.add, h, attention(rms(h, p + ".norm_self_attn", axis: 1), p + ".self_attn", source: nil, t))
            h = b.binary(.add, h, attention(rms(h, p + ".norm_cross_attn", axis: 1), p + ".cross_attn", source: a("source"), t))
            let up = linear(rms(h, p + ".norm_mlp", axis: 1), p + ".mlp0", bias: p + ".mlp0_bias")
            h = b.binary(.add, h, linear(b.gelu(up, tanh: false), p + ".mlp2", bias: p + ".mlp2_bias"))
            if taps && (i == 0 || i == layers - 1) { outputs.append(("cond_block\(i)_out", b.cast(h, to: .f32))) }
        }
        m.outputs = [("context", b.cast(rms(linear(h, "out_proj", bias: "out_proj_bias"), "norm", axis: 1), to: .f32))] + outputs
        return m
    }

    /// **What `context` would specialize** for `targetLength` T5 identifiers over `sourceLength` of Qwen's states, nothing
    /// allocated (`Stage.prepare`).
    package static func prepare(targetLength: Int, sourceLength: Int, path: String, budget: Int) throws -> [Preparation] {
        let c = AnimaConditioner(targetLength: targetLength, sourceLength: sourceLength)
        return try SiliCore.Stage.prepare(declaration(c, taps: false).declaration, files: [try AnimaEvaluation.file(path)], budget: budget)
    }

    /// The stage `context` opens and `prepare` specializes, and the outputs it writes.
    private static func declaration(_ c: AnimaConditioner, taps: Bool) -> (declaration: SiliCore.Stage.Declaration, outputs: [String]) {
        let (t, s, d) = (c.targetRows, c.sourceRows, c.dim)
        let outputs = ["context"] + (taps ? ["cond_block0_out", "cond_block\(c.layers - 1)_out"] : [])
        var declaration = SiliCore.Stage.Declaration("anima conditioner")
        declaration.tensors = [.init("h", bytes: t * d * 2), .init("source", bytes: s * d * 2), .init("t_mask", bytes: t * 2), .init("s_mask", bytes: s * 2)]
            + ["cos_t", "sin_t"].map { .init($0, bytes: t * c.headDim * 2) } + ["cos_s", "sin_s"].map { .init($0, bytes: s * c.headDim * 2) }
            + outputs.map { .init($0, bytes: t * d * 4, .apart) }
        declaration.residents = entries()
        let asset = c.asset(taps: taps)
        declaration.functions = ["conditioner": SiliCore.Stage.Function(asset, unit: .gpu) { c.module(taps: taps) }]
        return (declaration, outputs)
    }

    /// **The context for a prompt**: Qwen's states `source` `[Lq, 1024]` and the T5 identifiers through the conditioner,
    /// on a stage of its own (the adapter's weights resident), let go after; `[max(Lt, 512), 1024]`, zero rows past Lt.
    /// `taps`: the first and last blocks' outputs besides, `[Lt, 1024]`, by the oracle's names.
    package static func context(source: HostTensor, t5Ids: [Int], path: String, taps: Bool = false,
                                budget: Int, say: (String) -> Void = { _ in }) async throws -> (context: HostTensor, tapped: [String: [Float]]) {
        guard let device = MTLCreateSystemDefaultDevice() else { throw Failure("no Metal device") }
        let c = AnimaConditioner(targetLength: t5Ids.count, sourceLength: source.shape[0])
        let file = try AnimaEvaluation.file(path)
        let (t, s, lt, lq, d) = (c.targetRows, c.sourceRows, t5Ids.count, source.shape[0], c.dim)
        let (declaration, outputs) = Self.declaration(c, taps: taps)
        func bf16(_ x: Float) -> UInt16 { UInt16(BFloat.round(x).bitPattern >> 16) }
        func fill(_ stage: SiliCore.Stage) throws {
            let h = stage.pointer("h").assumingMemoryBound(to: UInt16.self)
            let table = try file.tensor(mapKey("embed.weight"), "BF16", [c.vocabulary, d])
            for (r, id) in t5Ids.enumerated() { try file.read(table, at: id * d * 2, count: d * 2, into: h + r * d) }
            let src = stage.pointer("source").assumingMemoryBound(to: UInt16.self)
            for i in 0..<(lq * d) { src[i] = bf16(source.floats[i]) }
            for (name, n, l) in [("t_mask", t, lt), ("s_mask", s, lq)] {
                let p = stage.pointer(name).assumingMemoryBound(to: UInt16.self)
                for i in 0..<n { p[i] = bf16(i < l ? 0 : -1e4) }
            }
            for (side, l) in [("t", lt), ("s", lq)] {
                let (cos, sin) = rotaryTables(length: l, headDim: c.headDim)
                let pc = stage.pointer("cos_" + side).assumingMemoryBound(to: UInt16.self), ps = stage.pointer("sin_" + side).assumingMemoryBound(to: UInt16.self)
                for i in cos.indices { pc[i] = bf16(cos[i]); ps[i] = bf16(sin[i]) }
            }
        }
        func call(_ f: InferenceFunction, _ stage: SiliCore.Stage) async throws {
            var inputs: [String: View] = [:]
            let shapes: [String: [Int]] = ["h": [t, d], "source": [s, d], "t_mask": [1, t], "s_mask": [1, s],
                                           "cos_t": [t, 1, c.headDim], "sin_t": [t, 1, c.headDim], "cos_s": [s, 1, c.headDim], "sin_s": [s, 1, c.headDim]]
            for (name, shape) in shapes { inputs[name] = stage.view(name, shape, .bfloat16) }
            for (name, placed) in stage.residents { inputs[name] = stage.view(placed) }
            try await Run.call(f, stage.stream, inputs: inputs, outputs: outputs.map { ($0, stage.view($0, [t, d], .float32)) })
        }
        let stage = try await SiliCore.Stage.open(device, declaration, files: [file], budget: budget)
        say(stage.plan.line)
        defer { stage.close() }
        try fill(stage)
        try await call(stage.functions["conditioner"]!, stage)
        func rows(_ name: String) -> [Float] { Array(UnsafeBufferPointer(start: stage.pointer(name).assumingMemoryBound(to: Float.self), count: lt * d)) }
        var tapped: [String: [Float]] = [:]
        for name in outputs.dropFirst() { tapped[name] = rows(name) }
        let context = rows("context") + [Float](repeating: 0, count: (c.rows - lt) * d)
        return (HostTensor([c.rows, d], floats: context), tapped)
    }
}
