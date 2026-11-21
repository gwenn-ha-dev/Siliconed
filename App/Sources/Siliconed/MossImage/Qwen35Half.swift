import CoreAI
import Foundation
import SiliCore

/// **The Compact's path: Qwen3.5-9B in W8A16, f16 under the guard** (`Guard.swift`). The file is ComfyUI's int8
/// `convrot` (`qwen3.5_9b_int8_convrot.safetensors`): every product of the 32 layers int8 `[out, in]` with a scale a row,
/// its codes rotated by the Hadamard by 256 (`W = (q · s) · R`, so `x · Wᵀ = (x · R) · (q · s)ᵀ`); the embedding table
/// too; `in_proj_b`, `in_proj_a` (32 rows), the conv, the norms, `A_log` and `dt_bias` as the Standard's.
///
/// The stream is (x̂, X): x̂ f16 in [2¹³, 2¹⁴] a row, X its factor (f32). Each product reads its input rotated then
/// brought under its own target, its result kept under the factor of that input; the products that read one value
/// read it under the smallest of their targets, once rotated. What the guard writes alone and the DeltaNet adds:
///   - **the delta rule reads true values, in f32**: q, k, v, z multiplied out of their factor before the conv, then
///     conv, SiLU, the l2 norms, the decay, the rule and the gated norm as the Standard computes them, without its bf16
///     roundings; the gated norm's output (f32) rotated in f32, then brought under `out_proj`'s target;
///   - **β and the decay's `a` are products of f32 weights**: their 32 rows are not codes, and read the normed row
///     unrotated, in f32;
///   - **a zero-centred norm reads x̂ with eps / X²**, its gain `1 + w` rounded once to f16;
///   - **the attention's gate and v are true values**: σ(gate · F) in f32, v · F in f16 (bounded: a product of a
///     normed row), what the attention reads across rows; the causal mask −10⁴ in f16 (a score is under 2⁸ there).
extension Qwen35Encoder {
    /// How a layer's function computes: the Standard's bf16, or the Compact's W8A16 in f16; its products' weights, bf16
    /// or int8 codes, in the layout the engine reads as the layers open (`Block.Build`).
    package enum Path: Equatable, Sendable {
        case bfloat
        case half

        /// Its name in the layers'.
        var suffix: String { self == .bfloat ? "bf16" : "w8a16" }

        var isBFloat: Bool { if case .bfloat = self { true } else { false } }
    }

    /// A half layer's arguments: the stream and its factor, then the weights; the products `[in, out]` or `[out, in]`
    /// int8 as `codes` lays them, each with its scales `[out]` f16 (`Recipe.codes`).
    func halfArguments(full: Bool, rows: Int, codes: CodesLayout) -> [Argument] {
        let c = config
        func product(_ name: String, _ out: Int, _ inn: Int) -> [Argument] {
            [(name, codes == .inOut ? [inn, out] : [out, inn], .si8), (name + "_scale", [out], .f16)]
        }
        let mixer: [Argument] = full
            ? product("q", 2 * c.heads * c.headDim, c.width) + product("k", c.kvHeads * c.headDim, c.width)
                + product("v", c.kvHeads * c.headDim, c.width) + [("q_norm", [c.headDim], .bf16), ("k_norm", [c.headDim], .bf16)]
                + product("out", c.width, c.heads * c.headDim)
            : product("qkv", c.convChannels, c.width) + product("z", c.valueWidth, c.width)
                + [("b", [c.valueHeads, c.width], .bf16), ("a", [c.valueHeads, c.width], .bf16), ("conv", [c.convChannels * c.convKernel], .bf16),
                   ("a_log", [c.valueHeads], .bf16), ("dt_bias", [c.valueHeads], .bf16), ("gnorm", [c.valueHeadDim], .bf16)]
                + product("out", c.width, c.valueWidth)
        return [("x", [rows, c.width], .f16), ("x_factor", [1, rows], .f32), ("norm1", [c.width], .bf16)] + mixer
            + [("norm2", [c.width], .bf16)] + product("gate", c.hidden, c.width) + product("up", c.hidden, c.width)
            + product("down", c.width, c.hidden)
    }

    /// Layer `i`'s weights on the half path: the products' codes, the rest bf16 as the Standard reads it.
    func halfEntries(_ i: Int) -> [Weights.Entry] {
        let p = Self.prefix + "layers.\(i)."
        func codes(_ name: String, _ key: String) -> Weights.Entry { (name, p + key, .codes) }
        func plain(_ name: String, _ key: String) -> Weights.Entry { (name, p + key, .bfloat16(joined: [])) }
        let mixer: [Weights.Entry] = config.isFull(i)
            ? [codes("q", "self_attn.q_proj"), codes("k", "self_attn.k_proj"), codes("v", "self_attn.v_proj"),
               plain("q_norm", "self_attn.q_norm.weight"), plain("k_norm", "self_attn.k_norm.weight"), codes("out", "self_attn.o_proj")]
            : [codes("qkv", "linear_attn.in_proj_qkv"), codes("z", "linear_attn.in_proj_z"), plain("b", "linear_attn.in_proj_b.weight"),
               plain("a", "linear_attn.in_proj_a.weight"), plain("conv", "linear_attn.conv1d.weight"), plain("a_log", "linear_attn.A_log"),
               plain("dt_bias", "linear_attn.dt_bias"), plain("gnorm", "linear_attn.norm.weight"), codes("out", "linear_attn.out_proj")]
        return [plain("norm1", "input_layernorm.weight")] + mixer
            + [plain("norm2", "post_attention_layernorm.weight"), codes("gate", "mlp.gate_proj"), codes("up", "mlp.up_proj"),
               codes("down", "mlp.down_proj")]
    }

    /// A half function's writer: the arguments by name, the guard's products and norms.
    final class Guarded {
        let b: Aicode.Block, names: [String], eps: Double, codes: CodesLayout, rows: Int
        private var constants = Aicode.ScalingConstants()

        init(_ m: Aicode.Module, eps: Double, codes: CodesLayout, rows: Int) {
            (b, names, self.eps, self.codes, self.rows) = (m.body, m.argumentNames, eps, codes, rows)
        }

        func a(_ name: String) -> Int { names.firstIndex(of: name)! }
        func f32(_ x: Int) -> Int { b.castUnlessSame(x, to: .f32) }
        /// A factor argument `[1, rows]` as the rows' column.
        func column(_ name: String) -> Int { b.reshape(a(name), [rows, 1]) }
        /// A zero-centred norm's gain `1 + w`, rounded once to f16.
        func gain(_ name: String) -> Int { b.cast(b.scale("add", f32(a(name)), 1), to: .f16) }
        /// `Qwen3_5RMSNorm` of x̂ kept under X along `axis`: F leaves nothing but its eps.
        func normed(_ x: Int, _ xf: Int, _ gain: String) -> Int {
            b.binary(.mul, b.rmsNorm(x, axes: [1], epsilon: b.epsilon(xf, eps)), self.gain(gain))
        }
        /// `t` (f16 or f32) rotated as the codes were, then brought under the smallest target of the products that read it.
        func entering(_ t: Int, _ products: [String]) -> (unit: Int, factor: Int) {
            func target(_ name: String) -> Int { b.target(codes: [(a(name), a(name + "_scale"))], layout: codes) }
            let e0 = products.dropFirst().reduce(target(products[0])) { b.elementwise("minimum", $0, target($1)) }
            return b.ranged(b.convrot(t), under: e0)
        }
        func product(_ x: Int, _ name: String) -> Int {
            b.w8a16(x, codes: a(name), scale: a(name + "_scale"), layout: codes, constants: &constants)
        }
        /// The feed-forward half and its residual (`Qwen3Functions.swift`'s): SiLU(g F₂) · u F₂ by `gatedSiLU`.
        func feedForward(_ x: Int, _ xf: Int, hidden: Int) -> (value: Int, factor: Int) {
            let (r, f2) = entering(normed(x, xf, "norm2"), ["gate", "up"])
            let (h, fh) = b.gatedSiLU(product(r, "gate"), product(r, "up"), factor: f2) { entering($0, ["down"]) }
            return b.residual(x, xf, plus: product(h, "down"), factor: fh)
        }
    }

    private func halfModule(full: Bool, rows: Int, codes: CodesLayout) -> (Aicode.Module, Guarded) {
        let m = Aicode.Module(arguments: halfArguments(full: full, rows: rows, codes: codes).map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        return (m, Guarded(m, eps: config.eps, codes: codes, rows: rows))
    }

    /// **A full-attention layer on `rows`, half**: outputs y, y_factor.
    func halfAttentionModule(rows n: Int, codes: CodesLayout) -> Aicode.Module {
        let c = config, hd = c.headDim, r = c.rotaryDim
        let (made, g) = halfModule(full: true, rows: n, codes: codes)
        var m = made
        let tables = rotaryTables(length: n)
        let cos = m.constant("rope_cos", Self.bytes(tables.cos), Aicode.tensor([n, 1, r], .f32))
        let sin = m.constant("rope_sin", Self.bytes(tables.sin), Aicode.tensor([n, 1, r], .f32))
        let causal = (0..<(n * n)).map { $0 % n > $0 / n ? Float16(-1e4) : 0 }
        let mask = m.constant("causal", Self.bytes(causal), Aicode.tensor([n, n], .f16))
        let zeros = m.constant("zeros", [UInt8](repeating: 0, count: 2 * c.heads / c.kvHeads), Aicode.tensor([1, 1, c.heads / c.kvHeads, 1, 1], .f16))
        let b = g.b, x = g.a("x"), xf = g.column("x_factor")
        let (u, f) = g.entering(g.normed(x, xf, "norm1"), ["q", "k", "v"])
        let qg = b.reshape(g.product(u, "q"), [n, c.heads, 2 * hd])
        let q0 = b.slice(qg, axis: 2, start: 0, size: hd)
        let gate = b.reshape(b.slice(qg, axis: 2, start: hd, size: hd), [n, c.heads * hd])
        let k0 = b.reshape(g.product(u, "k"), [n, c.kvHeads, hd])
        let v = b.binary(.mul, b.reshape(g.product(u, "v"), [n, c.kvHeads, hd]), b.reshape(b.cast(f, to: .f16), [n, 1, 1]))
        let e = b.reshape(b.epsilon(f, c.eps), [n, 1, 1])
        func headNormed(_ t: Int, _ gain: String) -> Int { b.binary(.mul, b.rmsNorm(t, axes: [2], epsilon: e), g.gain(gain)) }
        // RoPE on the first 64 of each head, in f32 on the f32 tables, rounded once.
        func rotated(_ t: Int) -> Int {
            let t32 = g.f32(t)
            let rot = b.slice(t32, axis: 2, start: 0, size: r), pass = b.slice(t, axis: 2, start: r, size: hd - r)
            let half = b.concat([b.scale("mul", b.slice(rot, axis: 2, start: r / 2, size: r / 2), -1), b.slice(rot, axis: 2, start: 0, size: r / 2)], axis: 2)
            return b.concat([b.cast(b.binary(.add, b.binary(.mul, rot, cos), b.binary(.mul, half, sin)), to: .f16), pass], axis: 2)
        }
        func headsFirst(_ t: Int, _ heads: Int) -> Int { b.reshape(b.transpose(t, [1, 0, 2]), [1, heads, n, hd]) }
        func repeated(_ t: Int) -> Int { b.reshape(b.binary(.add, b.reshape(t, [1, c.kvHeads, 1, n, hd]), zeros), [1, c.heads, n, hd]) }
        let q = headsFirst(rotated(headNormed(q0, "q_norm")), c.heads)
        let k = repeated(headsFirst(rotated(headNormed(k0, "k_norm")), c.kvHeads))
        let o = b.attention([q, k, repeated(headsFirst(v, c.kvHeads)), mask], scale: 1 / Double(hd).squareRoot(), symbol: "sdpa_qwen35")
        let merged = b.reshape(b.transpose(b.reshape(o, [c.heads, n, hd]), [1, 0, 2]), [n, c.heads * hd])
        let opened = b.cast(b.sigmoid(b.binary(.mul, g.f32(gate), f)), to: .f16)
        let (ou, of) = g.entering(b.binary(.mul, merged, opened), ["out"])
        let (x1, x1f) = b.residual(x, xf, plus: g.product(ou, "out"), factor: of)
        let (y, yf) = g.feedForward(x1, x1f, hidden: c.hidden)
        m.outputs = [("y", y), ("y_factor", b.reshape(yf, [1, n]))]
        return m
    }

    /// **A Gated DeltaNet layer on `rows`, half**: outputs y, y_factor.
    func halfDeltaModule(rows n: Int, codes: CodesLayout) -> Aicode.Module {
        let c = config
        let (made, g) = halfModule(full: false, rows: n, codes: codes)
        var m = made
        let b = g.b, x = g.a("x"), xf = g.column("x_factor")
        let n1 = g.normed(x, xf, "norm1")
        let (u, f) = g.entering(n1, ["qkv", "z"])
        let qkv = b.binary(.mul, g.f32(g.product(u, "qkv")), f), z = b.binary(.mul, g.f32(g.product(u, "z")), f)
        let n1f = g.f32(n1)
        let beta = b.sigmoid(b.linear(n1f, g.f32(g.a("b")))), a = b.linear(n1f, g.f32(g.a("a")))

        // The causal conv with zero history, in f32: the rows shifted down by 3 − k, times the kernel's column k.
        let history = c.convKernel - 1
        let zeros = m.constant("history", [UInt8](repeating: 0, count: history * c.convChannels * 4), Aicode.tensor([history, c.convChannels], .f32))
        let padded = b.concat([zeros, qkv], axis: 0)
        // The kernel given flat: `[channels, 4]` bf16 rows of 8 bytes are what Core AI binds padded to 64 (copied at
        // every call, 19/11); flat, as it lies.
        let kernel = g.f32(b.transpose(b.reshape(g.a("conv"), [c.convChannels, c.convKernel]), [1, 0]))
        var sum: Int?
        for k in 0..<c.convKernel {
            let term = b.binary(.mul, b.slice(padded, axis: 0, start: k, size: n), b.slice(kernel, axis: 0, start: k, size: 1))
            sum = sum.map { b.binary(.add, $0, term) } ?? term
        }
        let mixed = b.silu(sum!)
        func heads(_ start: Int, _ count: Int, _ dim: Int) -> Int { b.reshape(b.slice(mixed, axis: 1, start: start, size: count * dim), [n, count, dim]) }
        let q = heads(0, c.keyHeads, c.keyHeadDim), k = heads(c.keyWidth, c.keyHeads, c.keyHeadDim)
        let v = heads(2 * c.keyWidth, c.valueHeads, c.valueHeadDim)
        let raw = b.binary(.add, a, g.f32(g.a("dt_bias")))
        let softplus = b.binary(.add, b.primitive("log", b.scale("add", b.exp(b.scale("minimum", raw, 20)), 1)),
                                b.scale("maximum", b.scale("sub", raw, 20), 0))
        let decay = b.binary(.mul, b.scale("mul", b.exp(g.f32(g.a("a_log"))), -1), softplus)

        let w = Writer(b: b, names: g.names, eps: c.eps)
        let core = gatedDelta(&m, w, q: q, k: k, v: v, decay: decay, beta: beta, rows: n)
        let normed = b.binary(.mul, b.rmsNorm(b.reshape(core, [n * c.valueHeads, c.valueHeadDim]), axes: [1], eps: c.eps), g.f32(g.a("gnorm")))
        let gated = b.reshape(b.binary(.mul, normed, b.silu(b.reshape(z, [n * c.valueHeads, c.valueHeadDim]))), [n, c.valueWidth])
        let (gu, gf) = g.entering(gated, ["out"])
        let (x1, x1f) = b.residual(x, xf, plus: g.product(gu, "out"), factor: gf)
        let (y, yf) = g.feedForward(x1, x1f, hidden: c.hidden)
        m.outputs = [("y", y), ("y_factor", b.reshape(yf, [1, n]))]
        return m
    }

    /// The final norm of (x̂, X) on `rows`, into `hidden`: f16 out (a normed row times `1 + w`).
    func halfNormModule(rows n: Int) -> Aicode.Module {
        var m = Aicode.Module(arguments: [("x", Aicode.tensor([n, config.width], .f16)), ("x_factor", Aicode.tensor([1, n], .f32)),
                                          ("norm", Aicode.tensor([config.width], .bf16))])
        let g = Guarded(m, eps: config.eps, codes: .inOut, rows: n)
        m.outputs = [("hidden", g.normed(g.a("x"), g.column("x_factor"), "norm"))]
        return m
    }

    // MARK: - The embedding

    /// The embedding rows of `ids` from int8 `convrot` codes: `(q · s) · R` a row, R = H₂₅₆ / 16 by groups of 256, in f32.
    /// H₂₅₆ = H₄ ⊗ H₄ ⊗ H₄ ⊗ H₄ (`Aicode.convrotMatrix`): applied as four 4-point transforms, one a digit of the index.
    static func rotatedRows(_ ids: [Int], table: SafeTensors.Tensor, scales: SafeTensors.Tensor, file: SafeTensors, width d: Int) throws -> [Float] {
        var out = [Float](repeating: 0, count: ids.count * d), raw = [Int8](repeating: 0, count: d)
        for (r, id) in ids.enumerated() {
            try raw.withUnsafeMutableBytes { try file.read(table, at: id * d, count: d, into: $0.baseAddress!) }
            var s: Float = 0
            try withUnsafeMutableBytes(of: &s) { try file.read(scales, at: id * 4, count: 4, into: $0.baseAddress!) }
            out.withUnsafeMutableBufferPointer { o in
                for j in 0..<d { o[r * d + j] = Float(raw[j]) * s }
                for g in stride(from: r * d, to: (r + 1) * d, by: 256) { hadamard256(o.baseAddress! + g) }
            }
        }
        return out
    }

    /// `x · H₂₅₆ / 16` in place (H symmetric): the regular H₄ `[[1, 1, 1, −1], [1, 1, −1, 1], [1, −1, 1, 1], [−1, 1, 1, 1]]`
    /// along each base-4 digit of the index, each output summed from 0 in the order of its column (the same bits as the
    /// product written out, without the ~4 200 arrays a row of 4 096 allocated).
    static func hadamard256(_ x: UnsafeMutablePointer<Float>) {
        var stride = 64
        while stride >= 1 {
            for base in 0..<256 where (base / stride) % 4 == 0 {
                let (a, b, c, d) = (x[base], x[base + stride], x[base + 2 * stride], x[base + 3 * stride])
                x[base] = 0 + a + b + c - d
                x[base + stride] = 0 + a + b - c + d
                x[base + 2 * stride] = 0 + a - b + c + d
                x[base + 3 * stride] = 0 - a + b + c + d
            }
            stride /= 4
        }
        for i in 0..<256 { x[i] /= 16 }
    }

    /// Rows written as the stream: x̂ = v / X in f16, X = 2^e with the row's largest in (2¹³, 2¹⁴].
    static func writeStream(_ values: [Float], width d: Int, to: UnsafeMutableRawPointer, factors: UnsafeMutablePointer<Float>) {
        let h = to.assumingMemoryBound(to: Float16.self)
        for r in 0..<(values.count / d) {
            let row = values[(r * d)..<((r + 1) * d)]
            let top = max(row.reduce(0) { max($0, abs($1)) }, 0x1p-14)
            let e = RowExponent.ranged(largest: Double(top), f32Factor: true)
            let f = Float(pow(2.0, Double(e)))
            factors[r] = f
            for (j, v) in row.enumerated() { h[r * d + j] = Float16(v / f) }
        }
    }
}
