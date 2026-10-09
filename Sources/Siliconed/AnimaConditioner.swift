import Accelerate
import Foundation

/// Anima's **conditioner** (`llm_adapter`): learned T5 queries that go and read Qwen.
///
///     T5 ids ─ embed [32128, 1024] ─┬─ ×6 : x += SelfAttn(RMS(x))              RoPE on both
///                                   │       x += CrossAttn(RMS(x), Qwen)        RoPE on both, own positions
///     last_hidden_state Qwen ───────┘       x += W₂ GELU(W₁ RMS(x) + b₁) + b₂
///                                   ─ out_proj (+b) ─ RMSNorm ─ zero-padded to 512
///
/// What a port by analogy would miss:
///
///   - the **cross**-attention **carries a RoPE** — the queries rotate at their T5 position, the
///     keys at their Qwen position: it is not the DiT's cross-attention, which has none;
///   - the RoPE is **split** (`rotate_half`, pairs `(j, j+32)`), `θ = 10,000`, heads of 64;
///     the weights do not go through a forge, so it is applied as-is here;
///   - per-head QK-Norm, eps 1e-6; the MLPs and `out_proj` **have a bias**, the attentions do not.
///
/// **Everything is computed in double**, and it is free: forty tokens, six blocks, once per
/// render. This is the rule — more precision, never less — with nothing to trade off.
///
/// The weights are read from the published Turbo file, without a forge: 0.13 G parameters read once,
/// of which only the `embed` table rows of the prompt tokens are touched.
package final class AnimaConditioner {
    package let dim = 1024, heads = 16, headDim = 64, blocks = 6, hidden = 4096
    package let minimumLength = 512
    private let weights: Safetensors
    private let prefix: String
    package var recordBoundaries = false
    package private(set) var boundaries: [String: [Float]] = [:]

    /// - Parameter path: the ComfyUI Turbo file (`anima-turbo-v1.1.safetensors`), which mixes the
    ///   DiT and the adapter under `model.diffusion_model.llm_adapter.`.
    package init(path: String) throws {
        weights = try Safetensors(path: path)
        prefix = "model.diffusion_model.llm_adapter."
        guard weights.entries[prefix + "embed.weight"]?.shape == [32128, 1024] else {
            throw Safetensors.Failure.badHeader("\(path): no Anima adapter")
        }
    }

    private func load(_ name: String) throws -> [Double] {
        guard let values = weights.materialize(prefix + name) else {
            throw Safetensors.Failure.badHeader("adapter: \(name) missing")
        }
        var out = [Double](repeating: 0, count: values.count)
        vDSP_vspdp(values, 1, &out, 1, vDSP_Length(values.count))
        return out
    }

    /// `y[r, o] = Σ x[r, i] · W[o, i] (+ b[o])` — `W` in `nn.Linear` layout, `[output, input]`.
    private func linear(_ x: [Double], rows: Int, _ name: String, inputs: Int, outputs: Int,
                        bias: Bool = false) throws -> [Double] {
        let w = try load(name + ".weight")
        var y = [Double](repeating: 0, count: rows * outputs)
        if bias {
            let b = try load(name + ".bias")
            for r in 0..<rows { for o in 0..<outputs { y[r * outputs + o] = b[o] } }
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(rows), Int32(outputs), Int32(inputs),
                    1, x, Int32(inputs), w, Int32(inputs), bias ? 1 : 0, &y, Int32(outputs))
        return y
    }

    /// `nn.RMSNorm`: `x · rsqrt(mean(x²) + eps) · w`, per slice of `width`.
    private func rmsNorm(_ x: [Double], width: Int, weight: [Double], eps: Double = 1e-6) -> [Double] {
        var y = x
        for start in stride(from: 0, to: x.count, by: width) {
            var sum = 0.0
            for j in 0..<width { sum += x[start + j] * x[start + j] }
            let scale = 1 / (sum / Double(width) + eps).squareRoot()
            for j in 0..<width { y[start + j] = x[start + j] * scale * weight[j] }
        }
        return y
    }

    /// The split RoPE of `AnimaRotaryEmbedding`: the angle `p · θ^(−2j/64)` rotates `(j, j+32)`.
    private func rope(_ x: inout [Double], rows: Int) {
        let half = headDim / 2
        let inverse = (0..<half).map { 1 / pow(10000.0, Double(2 * $0) / Double(headDim)) }
        for r in 0..<rows {
            for h in 0..<heads {
                let base = (r * heads + h) * headDim
                for j in 0..<half {
                    let angle = Double(r) * inverse[j]
                    let (c, s) = (cos(angle), sin(angle))
                    let (a, b) = (x[base + j], x[base + j + half])
                    x[base + j] = a * c - b * s
                    x[base + j + half] = b * c + a * s
                }
            }
        }
    }

    private func attention(_ name: String, query x: [Double], rows: Int, context: [Double],
                           contextRows: Int) throws -> [Double] {
        var q = try linear(x, rows: rows, name + ".q_proj", inputs: dim, outputs: dim)
        var k = try linear(context, rows: contextRows, name + ".k_proj", inputs: dim, outputs: dim)
        let v = try linear(context, rows: contextRows, name + ".v_proj", inputs: dim, outputs: dim)
        q = rmsNorm(q, width: headDim, weight: try load(name + ".q_norm.weight"))
        k = rmsNorm(k, width: headDim, weight: try load(name + ".k_norm.weight"))
        rope(&q, rows: rows)
        rope(&k, rows: contextRows)
        let scale = 1 / Double(headDim).squareRoot()
        var out = [Double](repeating: 0, count: rows * dim)
        var scores = [Double](repeating: 0, count: contextRows)
        for h in 0..<heads {
            for r in 0..<rows {
                var peak = -Double.infinity
                for c in 0..<contextRows {
                    var dot = 0.0
                    for j in 0..<headDim { dot += q[r * dim + h * headDim + j] * k[c * dim + h * headDim + j] }
                    scores[c] = dot * scale
                    peak = max(peak, scores[c])
                }
                var total = 0.0
                for c in 0..<contextRows { scores[c] = exp(scores[c] - peak); total += scores[c] }
                for c in 0..<contextRows {
                    let p = scores[c] / total
                    for j in 0..<headDim { out[r * dim + h * headDim + j] += p * v[c * dim + h * headDim + j] }
                }
            }
        }
        return try linear(out, rows: rows, name + ".o_proj", inputs: dim, outputs: dim)
    }

    private func record(_ name: String, _ x: [Double]) {
        guard recordBoundaries else { return }
        boundaries[name] = x.map { Float($0) }
    }

    /// - Parameters:
    ///   - t5: the T5 identifiers, `</s>` included.
    ///   - qwen: Qwen3-0.6B's `last_hidden_state`, `[qwenRows, 1024]`.
    /// - Returns: `[max(512, t5.count), 1024]`, the rows beyond the T5 tokens zero — they
    ///   **count** in the softmax of the DiT's cross-attention, which does not mask them.
    package func condition(t5: [Int], qwen: UnsafeBufferPointer<Float>, qwenRows: Int) throws -> [Float] {
        let rows = t5.count
        guard qwen.count == qwenRows * dim else {
            throw Safetensors.Failure.badHeader("Qwen: \(qwen.count) values for \(qwenRows) × \(dim)")
        }
        var source = [Double](repeating: 0, count: qwen.count)
        vDSP_vspdp(qwen.baseAddress!, 1, &source, 1, vDSP_Length(qwen.count))

        // The table has 32,128 rows: we only expand those of the prompt.
        // The adapter is written by the forge in its published type (bf16, fp16, fp32), or fp32 when
        // it came 8-bit (`ForgeDiT.writeAdapter` stores a dequantized tensor in fp32, never rounded).
        guard let table = weights.pointer(prefix + "embed.weight"),
              let tableType = weights.entries[prefix + "embed.weight"]?.dtype, ["BF16", "F16", "F32"].contains(tableType) else {
            throw Safetensors.Failure.badHeader("embed: bf16, fp16 or fp32 expected")
        }
        var row = [Float](repeating: 0, count: dim)
        var x = [Double](repeating: 0, count: rows * dim)
        for (r, id) in t5.enumerated() {
            guard id >= 0, id < 32128 else { throw Safetensors.Failure.badHeader("T5 token \(id) outside the vocabulary") }
            row.withUnsafeMutableBytes {
                if tableType == "F32" {
                    $0.copyMemory(from: UnsafeRawBufferPointer(start: table.advanced(by: id * dim * 4), count: dim * 4))
                } else if tableType == "F16" {
                    Widen.float16ToFloat32(table.advanced(by: id * dim * 2), count: dim,
                                           into: $0.baseAddress!.assumingMemoryBound(to: Float.self))
                } else {
                    Widen.bfloat16ToFloat32(source: table.advanced(by: id * dim * 2), destination: $0.baseAddress!, count: dim)
                }
            }
            for j in 0..<dim { x[r * dim + j] = Double(row[j]) }
        }

        for b in 0..<blocks {
            let p = "blocks.\(b)."
            let selfIn = rmsNorm(x, width: dim, weight: try load(p + "norm_self_attn.weight"))
            let a = try attention(p + "self_attn", query: selfIn, rows: rows, context: selfIn, contextRows: rows)
            for i in 0..<x.count { x[i] += a[i] }
            let crossIn = rmsNorm(x, width: dim, weight: try load(p + "norm_cross_attn.weight"))
            let c = try attention(p + "cross_attn", query: crossIn, rows: rows, context: source, contextRows: qwenRows)
            for i in 0..<x.count { x[i] += c[i] }
            let mlpIn = rmsNorm(x, width: dim, weight: try load(p + "norm_mlp.weight"))
            var h = try linear(mlpIn, rows: rows, p + "mlp.0", inputs: dim, outputs: hidden, bias: true)
            for i in 0..<h.count { h[i] = 0.5 * h[i] * (1 + erf(h[i] / 2.0.squareRoot())) }
            let m = try linear(h, rows: rows, p + "mlp.2", inputs: hidden, outputs: dim, bias: true)
            for i in 0..<x.count { x[i] += m[i] }
            record("cond_block\(b)_out", x)
        }
        let projected = try linear(x, rows: rows, "out_proj", inputs: dim, outputs: dim, bias: true)
        let normed = rmsNorm(projected, width: dim, weight: try load("norm.weight"))

        var out = [Float](repeating: 0, count: max(minimumLength, rows) * dim)
        for i in 0..<normed.count { out[i] = Float(normed[i]) }
        return out
    }
}
