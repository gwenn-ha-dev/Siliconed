import Foundation
import XCTest
@testable import Siliconed

/// **MOSS-Image's Qwen3.5 encoder, the parts that need no weight**: the delta rule's recurrence against the
/// reference's chunked formulation (transcribed from `torch_chunk_gated_delta_rule`, in double), the causal
/// convolution, the partial RoPE, the template, the forge's plan, the arena's sizes.
final class Qwen35Tests: XCTestCase {

    // MARK: - A deterministic source of numbers

    private struct Generator {
        var state: UInt64
        mutating func next() -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Double(state >> 11) / Double(1 << 53)
        }
        mutating func normal() -> Double {
            let u = max(next(), 1e-12), v = next()
            return (-2 * log(u)).squareRoot() * cos(2 * .pi * v)
        }
    }

    // MARK: - The delta rule

    /// `torch_chunk_gated_delta_rule` for ONE value head, chunk `size`, on inputs already normalised (q scaled).
    /// Line for line: pad, cumulative decay, the unit-triangular systems, the scan over the chunks.
    private func chunked(q: [[Double]], k: [[Double]], v: [[Double]], g: [Double], beta: [Double], size: Int) -> [[Double]] {
        let n = q.count, dk = q[0].count, dv = v[0].count
        let padded = (n + size - 1) / size * size
        func pad(_ rows: [[Double]], _ width: Int) -> [[Double]] { rows + Array(repeating: Array(repeating: 0, count: width), count: padded - n) }
        let q = pad(q, dk), k = pad(k, dk), v = pad(v, dv)
        let beta = beta + Array(repeating: 0, count: padded - n), g = g + Array(repeating: 0, count: padded - n)
        var S = Array(repeating: Array(repeating: 0.0, count: dv), count: dk)
        var out: [[Double]] = []
        func dot(_ a: [Double], _ b: [Double]) -> Double { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }
        for c in 0..<(padded / size) {
            let r = (c * size)..<((c + 1) * size)
            let qc = Array(q[r]), kc = Array(k[r])
            let kBeta = zip(kc, beta[r]).map { row, b in row.map { $0 * b } }
            let vBeta = zip(v[r], beta[r]).map { row, b in row.map { $0 * b } }
            var cum: [Double] = [], run = 0.0
            for x in g[r] { run += x; cum.append(run) }
            func pair(_ i: Int, _ j: Int) -> Double { j <= i ? exp(cum[i] - cum[j]) : 0 }
            var ut = Array(repeating: Array(repeating: 0.0, count: size), count: size)
            var intra = ut
            for i in 0..<size { for j in 0..<size {
                ut[i][j] = dot(kBeta[i], kc[j]) * pair(i, j)
                intra[i][j] = dot(qc[i], kc[j]) * pair(i, j)
            } }
            let decayedKBeta = (0..<size).map { i in kBeta[i].map { $0 * exp(cum[i]) } }
            // (I + strict_lower(ut)) X = B, forward substitution.
            func solve(_ b: [[Double]]) -> [[Double]] {
                var x = b
                for i in 0..<size { for j in 0..<i { for col in 0..<b[0].count { x[i][col] -= ut[i][j] * x[j][col] } } }
                return x
            }
            let newValues = solve(vBeta), kCumdecay = solve(decayedKBeta)
            let last = cum[size - 1]
            let qDecayed = (0..<size).map { i in qc[i].map { $0 * exp(cum[i]) } }
            let kDecayed = (0..<size).map { i in kc[i].map { $0 * exp(last - cum[i]) } }
            var vNew = newValues
            for i in 0..<size { for col in 0..<dv {
                var s = 0.0
                for a in 0..<dk { s += kCumdecay[i][a] * S[a][col] }
                vNew[i][col] -= s
            } }
            for i in 0..<size {
                var row = [Double](repeating: 0, count: dv)
                for col in 0..<dv {
                    var inter = 0.0
                    for a in 0..<dk { inter += qDecayed[i][a] * S[a][col] }
                    var within = 0.0
                    for j in 0..<size { within += intra[i][j] * vNew[j][col] }
                    row[col] = inter + within
                }
                out.append(row)
            }
            let chunkDecay = exp(last)
            for a in 0..<dk { for col in 0..<dv {
                var s = 0.0
                for j in 0..<size { s += kDecayed[j][a] * vNew[j][col] }
                S[a][col] = S[a][col] * chunkDecay + s
            } }
        }
        return Array(out[0..<n])
    }

    /// The recurrence (the engine's `gatedDeltaRule`) equals the chunked formulation, across a chunk boundary
    /// (S = 150 → three chunks of 64, the last padded), with GQA (2 key heads for 4 value heads), strongly
    /// negative decays included.
    func testRecurrenceEqualsChunkedFormulation() {
        var rng = Generator(state: 7)
        let (S, kHeads, vHeads, dk, dv) = (150, 2, 4, 16, 12)
        let stride = 2 * kHeads * dk + vHeads * dv
        var packed = [Float](repeating: 0, count: S * stride)
        var q = [[[Double]]](repeating: [], count: kHeads), k = q
        for h in 0..<kHeads {
            for t in 0..<S {
                var a = (0..<dk).map { _ in rng.normal() }, b = a
                b = (0..<dk).map { _ in rng.normal() }
                let na = a.reduce(0) { $0 + $1 * $1 }.squareRoot(), nb = b.reduce(0) { $0 + $1 * $1 }.squareRoot()
                a = a.map { $0 / na / Double(dk).squareRoot() }; b = b.map { $0 / nb }   // q: l2 then / √dk, k: l2
                q[h].append(a); k[h].append(b)
                for j in 0..<dk { packed[t * stride + h * dk + j] = Float(a[j]); packed[t * stride + kHeads * dk + h * dk + j] = Float(b[j]) }
            }
        }
        var v = [[[Double]]](repeating: [], count: vHeads)
        for h in 0..<vHeads { for t in 0..<S {
            let row = (0..<dv).map { _ in rng.normal() }
            v[h].append(row)
            for j in 0..<dv { packed[t * stride + 2 * kHeads * dk + h * dv + j] = Float(row[j]) }
        } }
        var g = [Float](repeating: 0, count: S * vHeads), beta = g
        for i in 0..<g.count {
            g[i] = Float(i % 7 == 0 ? -3 * rng.next() : -0.3 * rng.next())
            beta[i] = Float(0.05 + 0.9 * rng.next())
        }
        var out = [Float](repeating: 0, count: S * vHeads * dv), state = [Float](repeating: 0, count: vHeads * dk * dv)
        packed.withUnsafeBufferPointer { p in
            Qwen35Encoder.gatedDeltaRule(q: p.baseAddress!, k: p.baseAddress! + kHeads * dk, v: p.baseAddress! + 2 * kHeads * dk,
                                         decay: g, beta: beta, out: &out, state: &state, rowStride: stride, sequence: S,
                                         kHeads: kHeads, vHeads: vHeads, keyDim: dk, valueDim: dv)
        }
        var worst = 0.0, scale = 0.0
        for h in 0..<vHeads {
            let kh = h / (vHeads / kHeads)
            let reference = chunked(q: q[kh], k: k[kh], v: v[h], g: (0..<S).map { Double(g[$0 * vHeads + h]) },
                                    beta: (0..<S).map { Double(beta[$0 * vHeads + h]) }, size: 64)
            for t in 0..<S { for j in 0..<dv {
                let mine = Double(out[(t * vHeads + h) * dv + j])
                worst = max(worst, abs(mine - reference[t][j])); scale = max(scale, abs(reference[t][j]))
            } }
        }
        XCTAssertGreaterThan(scale, 0.1, "the test must exercise a non-trivial output")
        XCTAssertLessThan(worst / scale, 2e-5, "recurrent vs chunked, relative to the largest output")
    }

    /// The delta rule forgets: with β = 0 nothing is written, the output is zero; with decay −∞ the state
    /// is erased at each token, so `o_t = β (k̂·q̂) v_t` exactly.
    func testDeltaRuleLimits() {
        let (S, dk, dv) = (5, 4, 3)
        var rng = Generator(state: 3)
        let row = 2 * dk + dv
        let packed = (0..<(S * row)).map { _ in Float(rng.normal()) }
        var out = [Float](repeating: 9, count: S * dv), state = [Float](repeating: 0, count: dk * dv)
        let zero = [Float](repeating: 0, count: S), half = [Float](repeating: 0.5, count: S), erase = [Float](repeating: -200, count: S)
        packed.withUnsafeBufferPointer { p in
            Qwen35Encoder.gatedDeltaRule(q: p.baseAddress!, k: p.baseAddress! + dk, v: p.baseAddress! + 2 * dk, decay: zero, beta: zero,
                                         out: &out, state: &state, rowStride: row, sequence: S, kHeads: 1, vHeads: 1, keyDim: dk, valueDim: dv)
        }
        XCTAssertEqual(out.map(abs).max(), 0)
        packed.withUnsafeBufferPointer { p in
            Qwen35Encoder.gatedDeltaRule(q: p.baseAddress!, k: p.baseAddress! + dk, v: p.baseAddress! + 2 * dk, decay: erase, beta: half,
                                         out: &out, state: &state, rowStride: row, sequence: S, kHeads: 1, vHeads: 1, keyDim: dk, valueDim: dv)
        }
        for t in 0..<S {
            let q = (0..<dk).map { packed[t * row + $0] }, k = (0..<dk).map { packed[t * row + dk + $0] }
            let qk = zip(q, k).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            for j in 0..<dv {
                XCTAssertEqual(out[t * dv + j], 0.5 * qk * packed[t * row + 2 * dk + j], accuracy: 1e-5)
            }
        }
    }

    // MARK: - Convolution, RoPE, softplus, norms

    func testCausalConvolutionHasZeroHistoryAndSiLU() {
        var rng = Generator(state: 11)
        let (S, C, K) = (9, 5, 4)
        let x = (0..<(S * C)).map { _ in Float(rng.normal()) }, w = (0..<(C * K)).map { _ in Float(rng.normal()) }
        var out = [Float](repeating: 0, count: S * C)
        Qwen35Encoder.causalConvolution(x, weight: w, into: &out, sequence: S, channels: C, kernel: K)
        for t in 0..<S { for c in 0..<C {
            var acc = 0.0
            for j in 0..<K where t - (K - 1) + j >= 0 { acc += Double(w[c * K + j]) * Double(x[(t - (K - 1) + j) * C + c]) }
            XCTAssertEqual(Double(out[t * C + c]), acc / (1 + exp(-acc)), accuracy: 1e-5)
        } }
        // The last tap reads the current row: zeroing the others leaves a pointwise SiLU.
        var pointwise = [Float](repeating: 0, count: C * K)
        for c in 0..<C { pointwise[c * K + K - 1] = 1 }
        Qwen35Encoder.causalConvolution(x, weight: pointwise, into: &out, sequence: S, channels: C, kernel: K)
        XCTAssertEqual(Double(out[0]), Double(x[0]) / (1 + exp(-Double(x[0]))), accuracy: 1e-6)
    }

    /// Partial `rotate_half`: pairs `(j, j + 32)` of the first 64 dims, the other 192 untouched, norm kept, position 0 the identity.
    func testPartialRopeRotatesTheFirstQuarterOnly() {
        let (S, heads, d, rotary) = (6, 2, 256, 64)
        var table = [Float](repeating: 0, count: S * rotary)
        Qwen35Encoder.writeRope(into: &table, count: S, theta: 1e7, rotaryDim: rotary)
        XCTAssertEqual(table[0], 1); XCTAssertEqual(table[1], 0)
        var rng = Generator(state: 5)
        let original = (0..<(S * heads * d)).map { _ in Float(rng.normal()) }
        var x = original
        Qwen35Encoder.applyRope(&x, table: table, sequence: S, heads: heads, headDim: d, rotaryDim: rotary)
        for s in 0..<S { for h in 0..<heads {
            let base = (s * heads + h) * d
            for j in 0..<32 {
                let angle = Double(s) * pow(1e7, -Double(2 * j) / 64)
                let a = Double(original[base + j]), b = Double(original[base + j + 32])
                XCTAssertEqual(Double(x[base + j]), a * cos(angle) - b * sin(angle), accuracy: 2e-5)
                XCTAssertEqual(Double(x[base + j + 32]), b * cos(angle) + a * sin(angle), accuracy: 2e-5)
            }
            for j in 64..<d { XCTAssertEqual(x[base + j], original[base + j]) }
        } }
        for h in 0..<heads { for j in 0..<d { XCTAssertEqual(x[h * d + j], original[h * d + j]) } }   // position 0
    }

    func testSoftplusKeepsTorchsThreshold() {
        XCTAssertEqual(Qwen35Encoder.softplus(25), 25)
        XCTAssertEqual(Qwen35Encoder.softplus(0), log(2), accuracy: 1e-7)
        XCTAssertEqual(Qwen35Encoder.softplus(-30), 0, accuracy: 1e-12)
    }

    func testGatedNormUsesTheWeightAsIsAndSiLUOfTheGate() {
        let core: [Float] = [1, 2, 3, 4], z: [Float] = [0.5, -1, 2, 0], w: [Float] = [1, 2, 0.5, 3]
        var out = core
        Qwen35Encoder.gatedNorm(&out, z: z, weight: w, rows: 1, dim: 4, eps: 1e-6)
        let scale = 1 / (Double(1 + 4 + 9 + 16) / 4 + 1e-6).squareRoot()
        for j in 0..<4 {
            let gate = Double(z[j]) / (1 + exp(-Double(z[j])))
            XCTAssertEqual(Double(out[j]), Double(w[j]) * Double(core[j]) * scale * gate, accuracy: 1e-5)
        }
    }

    // MARK: - The prompt

    func testTemplateHasNoSystemTurnAndOpensTheThinkingBlock() {
        XCTAssertEqual(Qwen35Prompt.template("woman posing in a library"),
                       "<|im_start|>user\nwoman posing in a library<|im_end|>\n<|im_start|>assistant\n<think>\n")
    }

    // MARK: - Config, plan, sizes

    static var header: [String: Any] { ["config": [
        "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
        "intermediate_size": 12288, "rms_norm_eps": 1e-6, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "attn_output_gate": true,
        "layer_types": (0..<32).map { ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention" },
        "rope_parameters": ["rope_theta": 10_000_000.0, "partial_rotary_factor": 0.25, "rope_type": "default",
                            "mrope_interleaved": true, "mrope_section": [11, 11, 10]],
    ] as [String: Any]] }

    func testConfigReadsTheNestedRopeAndTheLayerTypes() throws {
        let c = try Qwen35Encoder.Config(header: Self.header)
        XCTAssertEqual(c.rotaryDim, 64)
        XCTAssertEqual(c.theta, 1e7)
        XCTAssertEqual(c.convChannels, 8192)
        XCTAssertEqual(c.fullLayer.enumerated().filter(\.element).map(\.offset), [3, 7, 11, 15, 19, 23, 27, 31])
        var wrong = Self.header
        var config = wrong["config"] as! [String: Any]
        config["rope_parameters"] = ["rope_type": "yarn"]
        wrong["config"] = config
        XCTAssertThrowsError(try Qwen35Encoder.Config(header: wrong))
    }

    func testArenaAtTheLongestPromptStaysUnderTwoGigabytes() throws {
        let c = try Qwen35Encoder.Config(header: Self.header)
        let slices = Qwen35Encoder.slices(config: c, sequence: 4096)
        XCTAssertEqual(Set(slices.map(\.name)).count, slices.count)
        let bytes = Qwen35Encoder.arenaBytes(config: c, sequence: 4096)
        XCTAssertLessThan(bytes, 2 << 30)
        XCTAssertGreaterThan(Qwen35Encoder.arenaBytes(config: c, sequence: 4096), Qwen35Encoder.arenaBytes(config: c, sequence: 256))
    }

    /// A catalog of names and shapes only.
    private final class Names: TensorCatalog {
        let names: [String]
        let shapes: [String: [Int]]
        init(_ shapes: [String: [Int]]) { self.shapes = shapes; names = shapes.keys.sorted() }
        func shape(_ name: String) -> [Int]? { shapes[name] }
        func dtype(_ name: String) -> String? { "BF16" }
        func read(_ name: String) throws -> [Float] { throw Numerics.Failure(description: "names only") }
    }

    func testForgePlanKeepsTheLanguageModelAndRefusesAnUnknownLayerShape() throws {
        let types = ["linear_attention", "full_attention"]
        let config = OrderedJSON.object([.init("num_hidden_layers", .integer(2)),
                                         .init("layer_types", .list(types.map { .string($0) }))])
        var shapes: [String: [Int]] = ["language_model.embed_tokens.weight": [8, 4], "language_model.norm.weight": [4],
                                       "visual.blocks.0.attn.qkv.weight": [3, 3]]
        for (l, type) in types.enumerated() {
            for suffix in type == "full_attention" ? ForgeQwen35.fullOrder : ForgeQwen35.deltaOrder {
                shapes["language_model.layers.\(l).\(suffix)"] = [4, 4]
            }
        }
        let plan = try ForgeQwen35.plan(Names(shapes), published: OrderedJSON.object([.init("text_config", config)]))
        XCTAssertEqual(plan.order.first, "embed_tokens.weight")
        XCTAssertEqual(plan.order.last, "norm.weight")
        XCTAssertEqual(plan.order.count, 2 + ForgeQwen35.deltaOrder.count + ForgeQwen35.fullOrder.count)
        XCTAssertEqual(plan.discarded, ["visual.blocks.0.attn.qkv.weight"])
        shapes["language_model.layers.1.self_attn.extra.weight"] = [4, 4]
        XCTAssertThrowsError(try ForgeQwen35.plan(Names(shapes), published: OrderedJSON.object([.init("text_config", config)])))
    }

    func testForgeRuleFoldsTheZeroCentredNormsButNotTheGatedOne() {
        for name in ["layers.0.input_layernorm.weight", "layers.3.self_attn.q_norm.weight", "norm.weight"] {
            let r = ForgeQwen35.rule(name, shape: [4096])
            XCTAssertTrue(r.foldsOne, name)
            XCTAssertEqual(r.named, .float32)
        }
        let gated = ForgeQwen35.rule("layers.0.linear_attn.norm.weight", shape: [128])
        XCTAssertFalse(gated.foldsOne)
        XCTAssertEqual(gated.named, .float32)
        XCTAssertTrue(ForgeQwen35.rule("layers.0.mlp.up_proj.weight", shape: [12288, 4096]).transposed)
        XCTAssertFalse(ForgeQwen35.rule("embed_tokens.weight", shape: [248320, 4096]).transposed)
        XCTAssertFalse(ForgeQwen35.rule("layers.0.linear_attn.conv1d.weight", shape: [8192, 4]).foldsOne)
        XCTAssertFalse(ForgeQwen35.rule("layers.0.linear_attn.conv1d.weight", shape: [8192, 4]).transposed, "a depthwise conv is no Linear")
    }
}
