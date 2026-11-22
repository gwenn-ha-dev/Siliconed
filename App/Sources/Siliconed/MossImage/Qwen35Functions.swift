import CoreAI
import Foundation
import SiliCore

/// **Qwen3.5's functions**, written op by op: a layer's arguments and weights, the full attention, the gated delta
/// rule by chunks, the final norm.
extension Qwen35Encoder {
    typealias Argument = (name: String, shape: [Int], element: Aicode.Element)

    /// A layer's function's arguments: the stream, then its weights, each a place of the slot (`slot`); the products
    /// `[out, in]`, as the file lays them (`Recipe.product`).
    func arguments(full: Bool, rows: Int) -> [Argument] {
        let c = config
        func product(_ name: String, _ out: Int, _ inn: Int) -> Argument { (name, [out, inn], .bf16) }
        let mixer: [Argument] = full
            ? [product("in", c.attentionInputs, c.width), ("q_norm", [c.headDim], .bf16), ("k_norm", [c.headDim], .bf16)]
            : [product("in", c.deltaInputs, c.width), ("conv", [c.convChannels * c.convKernel], .bf16), ("a_log", [c.valueHeads], .bf16),
               ("dt_bias", [c.valueHeads], .bf16), ("gnorm", [c.valueHeadDim], .bf16)]
        return [("x", [rows, c.width], .bf16), ("norm1", [c.width], .bf16)] + mixer
            + [product("out", c.width, full ? c.heads * c.headDim : c.valueWidth), ("norm2", [c.width], .bf16),
               product("gate_up", 2 * c.hidden, c.width), product("down", c.width, c.hidden)]
    }

    /// The file's tensors each place of layer `i` holds, side by side.
    func keys(_ i: Int) -> [(place: String, keys: [String])] {
        let p = Self.prefix + "layers.\(i)."
        let mixer: [(String, [String])] = config.isFull(i)
            ? [("in", ["q_proj", "k_proj", "v_proj"].map { p + "self_attn.\($0).weight" }),
               ("q_norm", [p + "self_attn.q_norm.weight"]), ("k_norm", [p + "self_attn.k_norm.weight"]), ("out", [p + "self_attn.o_proj.weight"])]
            : [("in", ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"].map { p + "linear_attn.\($0).weight" }),
               ("conv", [p + "linear_attn.conv1d.weight"]), ("a_log", [p + "linear_attn.A_log"]), ("dt_bias", [p + "linear_attn.dt_bias"]),
               ("gnorm", [p + "linear_attn.norm.weight"]), ("out", [p + "linear_attn.out_proj.weight"])]
        return [("norm1", [p + "input_layernorm.weight"])] + mixer
            + [("norm2", [p + "post_attention_layernorm.weight"]), ("gate_up", [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"]),
               ("down", [p + "mlp.down_proj.weight"])]
    }

    /// The places a layer's products hold.
    static let products: Set<String> = ["in", "out", "gate_up", "down"]

    /// Layer `i`'s weights as the stage reads them, bf16: each product's parts joined along `out` (`Recipe.product`, by
    /// their keys without `.weight`), the others as they lie.
    func entries(_ i: Int) -> [Weights.Entry] {
        keys(i).map { place, keys in
            guard Self.products.contains(place) else { return (place, keys[0], .bfloat16(joined: Array(keys.dropFirst()))) }
            let bare = keys.map { String($0.dropLast(".weight".count)) }
            return (place, bare[0], .product(joined: Array(bare.dropFirst())))
        }
    }

    /// A function's writer: the arguments by name, the norms and products as the reference rounds them.
    struct Writer {
        let b: Aicode.Block, names: [String], eps: Double
        func a(_ name: String) -> Int { names.firstIndex(of: name)! }
        func f32(_ x: Int) -> Int { b.castUnlessSame(x, to: .f32) }
        func bf16(_ x: Int) -> Int { b.castUnlessSame(x, to: .bf16) }
        /// `Qwen3_5RMSNorm` along `axis`: `norm(x.float()) * (1.0 + weight.float())`, rounded once.
        func zeroCentred(_ x: Int, _ gain: Int, axis: Int) -> Int {
            bf16(b.binary(.mul, b.rmsNorm(f32(x), axes: [axis], eps: eps), b.scale("add", f32(gain), 1)))
        }
        /// `act(x)` of a bf16 value as torch computes it: in f32, rounded once.
        func silu(_ x: Int) -> Int { bf16(b.silu(f32(x))) }
        func sigmoid(_ x: Int) -> Int { bf16(b.sigmoid(f32(x))) }
        /// `a @ b` over the last two axes, in f32.
        func product(_ x: Int, _ y: Int) -> Int {
            let (xs, ys) = (Aicode.shape(b.types[x]).shape, Aicode.shape(b.types[y]).shape)
            return b.matmul(x, y, result: Aicode.tensor(Array(xs.dropLast()) + [ys.last!], .f32))
        }
        func swapLast(_ x: Int) -> Int {
            let rank = Aicode.shape(b.types[x]).shape.count
            return b.transpose(x, Array(0..<(rank - 2)) + [rank - 1, rank - 2])
        }
        /// The feed-forward half and its residual: `x + down(SiLU(gate n) ⊙ up n)`, n = RMS(x).
        func feedForward(_ x: Int, hidden: Int) -> Int {
            let gu = b.linear(zeroCentred(x, a("norm2"), axis: 1), a("gate_up"))
            let gate = b.slice(gu, axis: 1, start: 0, size: hidden), up = b.slice(gu, axis: 1, start: hidden, size: hidden)
            return b.binary(.add, x, b.linear(b.binary(.mul, silu(gate), up), a("down")))
        }
    }

    private func module(full: Bool, rows: Int) -> (Aicode.Module, Writer) {
        let m = Aicode.Module(arguments: arguments(full: full, rows: rows).map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        return (m, Writer(b: m.body, names: m.argumentNames, eps: config.eps))
    }

    static func bytes<T>(_ values: [T]) -> [UInt8] { values.withUnsafeBytes { Array($0) } }

    /// **A full-attention layer on `rows`**, its RoPE tables and causal mask constants.
    func attentionModule(rows n: Int) -> Aicode.Module {
        let c = config, hd = c.headDim, r = c.rotaryDim
        let (made, w) = module(full: true, rows: n)
        var m = made
        let tables = rotaryTables(length: n)
        let cos = m.constant("rope_cos", Self.bytes(tables.cos.map(BFloat16Bits.rounded)), Aicode.tensor([n, 1, r], .bf16))
        let sin = m.constant("rope_sin", Self.bytes(tables.sin.map(BFloat16Bits.rounded)), Aicode.tensor([n, 1, r], .bf16))
        // −10⁹ past the diagonal: Core AI's kernel subtracts each row's largest score, which is a real one.
        let causal = (0..<(n * n)).map { $0 % n > $0 / n ? BFloat16Bits.rounded(-1e9) : 0 }
        let mask = m.constant("causal", Self.bytes(causal), Aicode.tensor([n, n], .bf16))
        let zeros = m.constant("zeros", [UInt8](repeating: 0, count: 2 * c.heads / c.kvHeads), Aicode.tensor([1, 1, c.heads / c.kvHeads, 1, 1], .bf16))
        let b = w.b, x = w.a("x")
        let all = b.linear(w.zeroCentred(x, w.a("norm1"), axis: 1), w.a("in"))
        let qg = b.reshape(b.slice(all, axis: 1, start: 0, size: 2 * c.heads * hd), [n, c.heads, 2 * hd])
        let q0 = b.slice(qg, axis: 2, start: 0, size: hd)
        let gate = b.reshape(b.slice(qg, axis: 2, start: hd, size: hd), [n, c.heads * hd])
        func part(_ start: Int) -> Int { b.reshape(b.slice(all, axis: 1, start: start, size: c.kvHeads * hd), [n, c.kvHeads, hd]) }
        let k0 = part(2 * c.heads * hd), v = part(2 * c.heads * hd + c.kvHeads * hd)
        // `apply_rotary_pos_emb` on the first 64 of each head, bf16 throughout.
        func rotated(_ t: Int) -> Int {
            let rot = b.slice(t, axis: 2, start: 0, size: r), pass = b.slice(t, axis: 2, start: r, size: hd - r)
            let half = b.concat([b.scale("mul", b.slice(rot, axis: 2, start: r / 2, size: r / 2), -1), b.slice(rot, axis: 2, start: 0, size: r / 2)], axis: 2)
            return b.concat([b.binary(.add, b.binary(.mul, rot, cos), b.binary(.mul, half, sin)), pass], axis: 2)
        }
        func headsFirst(_ t: Int, _ heads: Int) -> Int { b.reshape(b.transpose(t, [1, 0, 2]), [1, heads, n, hd]) }
        func repeated(_ t: Int) -> Int { b.reshape(b.binary(.add, b.reshape(t, [1, c.kvHeads, 1, n, hd]), zeros), [1, c.heads, n, hd]) }
        let q = headsFirst(rotated(w.zeroCentred(q0, w.a("q_norm"), axis: 2)), c.heads)
        let k = repeated(headsFirst(rotated(w.zeroCentred(k0, w.a("k_norm"), axis: 2)), c.kvHeads))
        let o = b.attention([q, k, repeated(headsFirst(v, c.kvHeads)), mask], scale: 1 / Double(hd).squareRoot(), symbol: "sdpa_qwen35")
        let merged = b.reshape(b.transpose(b.reshape(o, [c.heads, n, hd]), [1, 0, 2]), [n, c.heads * hd])
        let x1 = b.binary(.add, x, b.linear(b.binary(.mul, merged, w.sigmoid(gate)), w.a("out")))
        m.outputs = [("y", w.feedForward(x1, hidden: c.hidden))]
        return m
    }

    /// **A Gated DeltaNet layer on `rows`** (a whole number of chunks), the chunks' masks constants.
    func deltaModule(rows n: Int) -> Aicode.Module {
        let c = config
        let (made, w) = module(full: false, rows: n)
        var m = made
        let b = w.b, x = w.a("x")
        let all = b.linear(w.zeroCentred(x, w.a("norm1"), axis: 1), w.a("in"))
        let qkv = b.slice(all, axis: 1, start: 0, size: c.convChannels)
        let z = b.slice(all, axis: 1, start: c.convChannels, size: c.valueWidth)
        let beta = b.slice(all, axis: 1, start: c.convChannels + c.valueWidth, size: c.valueHeads)
        let a = b.slice(all, axis: 1, start: c.convChannels + c.valueWidth + c.valueHeads, size: c.valueHeads)

        // F.conv1d(padding = 3)[..., :T] with zero history: the rows shifted down by 3 − k, times the kernel's column k,
        // summed in f32 and rounded once; the SiLU after, rounded once.
        let history = c.convKernel - 1
        let zeros = m.constant("history", [UInt8](repeating: 0, count: history * c.convChannels * 2), Aicode.tensor([history, c.convChannels], .bf16))
        let padded = w.f32(b.concat([zeros, qkv], axis: 0))
        // The kernel given flat: `[channels, 4]` bf16 rows of 8 bytes are what Core AI binds padded to 64 (copied at
        // every call, 19/11); flat, as it lies.
        let kernel = w.f32(b.transpose(b.reshape(w.a("conv"), [c.convChannels, c.convKernel]), [1, 0]))
        var sum: Int?
        for k in 0..<c.convKernel {
            let term = b.binary(.mul, b.slice(padded, axis: 0, start: k, size: n), b.slice(kernel, axis: 0, start: k, size: 1))
            sum = sum.map { b.binary(.add, $0, term) } ?? term
        }
        let mixed = w.silu(w.bf16(sum!))
        func heads(_ start: Int, _ count: Int, _ dim: Int) -> Int { b.reshape(w.f32(b.slice(mixed, axis: 1, start: start, size: count * dim)), [n, count, dim]) }
        let q = heads(0, c.keyHeads, c.keyHeadDim), k = heads(c.keyWidth, c.keyHeads, c.keyHeadDim)
        let v = heads(2 * c.keyWidth, c.valueHeads, c.valueHeadDim)

        let betaF = w.f32(w.sigmoid(beta))
        // softplus as torch's (beyond 20 the identity): log(1 + exp(min(x, 20))) + max(x − 20, 0).
        let raw = b.binary(.add, w.f32(a), w.f32(w.a("dt_bias")))
        let softplus = b.binary(.add, b.primitive("log", b.scale("add", b.exp(b.scale("minimum", raw, 20)), 1)),
                                b.scale("maximum", b.scale("sub", raw, 20), 0))
        let decay = b.binary(.mul, b.scale("mul", b.exp(w.f32(w.a("a_log"))), -1), softplus)

        let core = gatedDelta(&m, w, q: q, k: k, v: v, decay: decay, beta: betaF, rows: n)
        // Qwen3_5RMSNormGated.forward
        let rounded = b.reshape(w.bf16(core), [n * c.valueHeads, c.valueHeadDim])
        let normed = b.binary(.mul, b.rmsNorm(rounded, axes: [1], eps: c.eps), w.a("gnorm"))
        let gated = b.reshape(w.bf16(b.binary(.mul, w.f32(normed), b.silu(w.f32(b.reshape(z, [n * c.valueHeads, c.valueHeadDim]))))), [n, c.valueWidth])
        let x1 = b.binary(.add, x, b.linear(gated, w.a("out")))
        m.outputs = [("y", w.feedForward(x1, hidden: c.hidden))]
        return m
    }

    /// `torch_chunk_gated_delta_rule`: q, k `[T, Hk, Dk]`, v `[T, Hv, Dv]`, decay and beta `[T, Hv]`, f32; T a whole number
    /// of chunks. Returns `[T, Hv, Dv]` f32.
    ///
    ///     per chunk (all at once, the chunk folded into the batch):
    ///       u = (I + L)⁻¹ (β v)     kc = (I + L)⁻¹ (β k · e^cum)     A = (q kᵀ) ⊙ D
    ///     the scan, S = 0 before the first chunk:
    ///       u' = u − kc S          out = (q · e^cum) S + A u'          S ← S · e^cum_last + (k · e^(cum_last − cum))ᵀ u'
    func gatedDelta(_ m: inout Aicode.Module, _ w: Writer, q: Int, k: Int, v: Int, decay: Int, beta: Int, rows T: Int) -> Int {
        let c = config, b = w.b, H = c.valueHeads, t = Self.chunkTokens, n = T / t, B = H * n
        let masks = Self.chunkMasks(t)
        func constant(_ key: String, _ values: [Float]) -> Int { m.constant(key, Self.bytes(values), Aicode.tensor([t, t], .f32)) }
        let onAndAbove = constant("on_and_above", masks.onAndAbove), above = constant("above", masks.above), below = constant("below", masks.below)
        let levels = masks.levels.enumerated().map { constant("level\($0.offset)", $0.element) }
        let identity = constant("identity", (0..<(t * t)).map { $0 / t == $0 % t ? 1 : 0 })
        let pair = m.constant("repeat", [UInt8](repeating: 0, count: 4 * c.valueHeads / c.keyHeads), Aicode.tensor([1, 1, c.valueHeads / c.keyHeads, 1], .f32))

        func l2(_ x: Int) -> Int { b.binary(.mul, x, b.primitive("rsqrt", b.scale("add", b.sum(b.binary(.mul, x, x), axes: [2]), 1e-6))) }
        func perValueHead(_ x: Int) -> Int {
            let wide = b.binary(.add, b.reshape(x, [T, c.keyHeads, 1, c.keyHeadDim]), pair)
            return b.transpose(b.reshape(wide, [T, H, c.keyHeadDim]), [1, 0, 2])                       // [Hv, T, Dk]
        }
        let qh = perValueHead(b.scale("mul", l2(q), 1 / Double(c.keyHeadDim).squareRoot())), kh = perValueHead(l2(k))
        let betaH = b.reshape(b.transpose(beta, [1, 0]), [H, T, 1])
        let vBeta = b.binary(.mul, b.transpose(v, [1, 0, 2]), betaH), kBeta = b.binary(.mul, kh, betaH)
        func chunks(_ x: Int) -> Int { b.reshape(x, [B, t, Aicode.shape(b.types[x]).shape.last!]) }
        let (qc, kc, vb, kb) = (chunks(qh), chunks(kh), chunks(vBeta), chunks(kBeta))

        // cum[i] = Σ_{j ≤ i} decay[j] within its chunk; D[i, j] = exp(cum_i − cum_j) on and below the diagonal, 0 above.
        let cum = w.product(b.reshape(b.transpose(decay, [1, 0]), [B, t]), onAndAbove)                  // [B, t]
        let pairwise = b.exp(b.binary(.add, b.binary(.sub, b.reshape(cum, [B, t, 1]), b.reshape(cum, [B, 1, t])), above))
        let lower = b.binary(.mul, b.binary(.mul, w.product(kb, w.swapLast(kc)), pairwise), below)
        // (I + L)⁻¹ by doubling blocks: X exact on diagonal blocks of s, then of 2s — [[A, 0], [C, D]]⁻¹ is
        // [[A⁻¹, 0], [−D⁻¹ C A⁻¹, D⁻¹]], that is X − X C̃ X with C̃ the lower-left blocks of L alone.
        var inverse = b.binary(.sub, identity, b.binary(.mul, lower, levels[0]))
        for level in levels.dropFirst() { inverse = b.binary(.sub, inverse, w.product(w.product(inverse, b.binary(.mul, lower, level)), inverse)) }
        let newValues = w.product(inverse, vb)
        let intra = b.binary(.mul, w.product(qc, w.swapLast(kc)), pairwise)
        guard n > 1 else { return b.transpose(b.reshape(w.product(intra, newValues), [H, T, c.valueHeadDim]), [1, 0, 2]) }

        let grown = b.reshape(b.exp(cum), [B, t, 1])
        let stateKeys = w.product(inverse, b.binary(.mul, kb, grown)), stateQueries = b.binary(.mul, qc, grown)
        let last = b.slice(cum, axis: 1, start: t - 1, size: 1)                                         // [B, 1]
        let writtenKeys = w.swapLast(b.binary(.mul, kc, b.reshape(b.exp(b.binary(.sub, last, cum)), [B, t, 1])))
        let carried = b.reshape(b.exp(last), [B, 1, 1])
        /// Chunk `i` of a batch of chunks: `[Hv, a, b]`.
        func pick(_ x: Int, _ i: Int) -> Int {
            let s = Aicode.shape(b.types[x]).shape
            return b.reshape(b.slice(b.reshape(x, [H, n] + s.dropFirst()), axis: 1, start: i, size: 1), [H] + s.dropFirst())
        }
        var state: Int?, outputs: [Int] = []
        for i in 0..<n {
            let u = pick(newValues, i)
            let fresh = state.map { b.binary(.sub, u, w.product(pick(stateKeys, i), $0)) } ?? u
            let within = w.product(pick(intra, i), fresh)
            outputs.append(state.map { b.binary(.add, w.product(pick(stateQueries, i), $0), within) } ?? within)
            guard i < n - 1 else { break }
            let written = w.product(pick(writtenKeys, i), fresh)
            state = state.map { b.binary(.add, b.binary(.mul, $0, pick(carried, i)), written) } ?? written
        }
        return b.transpose(b.concat(outputs, axis: 1), [1, 0, 2])
    }

    /// The final norm on `rows`, into `hidden`.
    func normModule(rows n: Int) -> Aicode.Module {
        var m = Aicode.Module(arguments: [("x", Aicode.tensor([n, config.width], .bf16)), ("norm", Aicode.tensor([config.width], .bf16))])
        let w = Writer(b: m.body, names: m.argumentNames, eps: config.eps)
        m.outputs = [("hidden", w.zeroCentred(w.a("x"), w.a("norm"), axis: 1))]
        return m
    }
}
