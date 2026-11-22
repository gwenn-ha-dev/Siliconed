import CoreAI
import Foundation
import SiliCore

/// **Qwen3's functions**, written op by op (`Aicode.Block`), their weights arguments: one layer over a palier of rows
/// (the encoder; the prefill, which also hands back its keys and values), one layer over a token and the K/V cache (the
/// decode), the head over a stretch of the vocabulary: every size, bf16 and ComfyUI's rotated codes.
///
///     y₁ = x + o(sdpa(RoPE(RMS_head(q) γ_q), RoPE(RMS_head(k) γ_k), v, causal))     q k v = RMS(x) γ₁ · Wqkv
///     y  = y₁ + down(SiLU(gate(n)) ⊙ up(n))                                           n = RMS(y₁) γ₂
///
/// The RoPE tables are arguments (`cos`, `sin` `[rows, 1, headDim]` f32): Qwen3-VL's positions are three-axis where a
/// picture sits, and the decode's token moves. The causal mask is a constant of the palier's asset (−10⁴ past the
/// diagonal: Core AI's kernel subtracts each row's largest score); the decode's is an argument, the cache's rows past the
/// token masked the same way.
/// **The arithmetic a file's weights are computed in**, read from the file: bf16 end to end on bf16 weights; f16
/// (W8A8, `w8a8`) on int8 codes, SDNQ's (`<key>.scale`, bf16 `[out, 1]`) or ComfyUI's (`<key>.weight_scale`, f32 `[out,
/// 1]`, `comfy_quant` saying whether the codes carry the `convrot` rotation), laid by pieces of k (`Weights.codesShape`).
enum Qwen3Arithmetic: Equatable {
    case bf16
    case w8a8(scale: String, rotated: Bool)

    init(_ file: SafeTensors, prefix: String) throws {
        let probe = prefix + "layers.0.self_attn.q_proj"
        guard let t = file.tensors[probe + ".weight"] else { throw ProgramError("\(file.path): no tensor \(probe).weight") }
        switch t.dtype {
        case "BF16": self = .bf16
        case "I8":
            if file.tensors[probe + ".scale"] != nil { self = .w8a8(scale: ".scale", rotated: false); return }
            guard file.tensors[probe + ".weight_scale"] != nil else { throw ProgramError("\(probe): int8 codes without their scales") }
            self = .w8a8(scale: ".weight_scale", rotated: try Self.rotated(file, probe))
        default: throw ProgramError("\(probe).weight: \(t.dtype), neither bf16 nor int8")
        }
    }

    /// Whether ComfyUI's note on `key` says its codes carry the `convrot` rotation (by 256, the only group it writes).
    static func rotated(_ file: SafeTensors, _ key: String) throws -> Bool {
        guard file.tensors[key + ".comfy_quant"] != nil else { return false }
        let note = try JSONSerialization.jsonObject(with: Data(try file.bytes(key + ".comfy_quant"))) as? [String: Any]
        guard note?["convrot"] as? Bool == true else { return false }
        guard note?["convrot_groupsize"] as? Int == 256 else { throw ProgramError("\(key): a rotation by other groups than 256") }
        return true
    }

    var half: Bool { if case .w8a8 = self { true } else { false } }
    var rotated: Bool { if case .w8a8(_, let r) = self { r } else { false } }
    /// The element of the stream, the gains and (bf16) the weights.
    var element: Aicode.Element { half ? .f16 : .bf16 }
    var scalar: NDArray.ScalarType { half ? .float16 : .bfloat16 }
    /// Its name in the layers'.
    var name: String {
        switch self { case .bf16: "bf16"; case .w8a8(_, let r): r ? "w8a8r" : "w8a8" }
    }
}

/// One argument of a function: its name, shape and element.
typealias Qwen3Argument = (name: String, shape: [Int], element: Aicode.Element)

extension Qwen3Encoder {
    /// **A layer's weights as its function reads them**: the products that share an input fused side by side along
    /// `out` (q k v; gate up, `Recipe.product`), the codes by pieces of k (`Weights.codesShape`), the bf16 weights `[out,
    /// in]`; the gains in the stream's element.
    func weightArguments(_ a: Qwen3Arithmetic) -> [Qwen3Argument] {
        let c = config, d = c.width, hd = c.headDim, e = a.element
        func lies(_ inn: Int, _ out: Int) -> [Int] { Weights.codesShape(inn: inn, out: out) }
        guard a.half else {
            return [("norm1", [d], e), ("qkv", [c.qkvWidth, d], e), ("q_norm", [hd], e), ("k_norm", [hd], e),
                    ("o", [d, c.heads * hd], e), ("norm2", [d], e), ("gate_up", [2 * c.hidden, d], e), ("down", [d, c.hidden], e)]
        }
        return [("norm1", [d], e), ("qkv", lies(d, c.qkvWidth), .si8), ("qkv_scale", [c.qkvWidth], e), ("q_norm", [hd], e), ("k_norm", [hd], e),
                ("o", lies(c.heads * hd, d), .si8), ("o_scale", [d], e), ("norm2", [d], e),
                ("gate_up", lies(d, 2 * c.hidden), .si8), ("gate_up_scale", [2 * c.hidden], e), ("down", lies(c.hidden, d), .si8), ("down_scale", [d], e)]
    }

    /// **Layer `i`'s weights as the stage reads them**, by the names `weightArguments` gives: the gains rounded once to
    /// f16 on codes (bf16 into f16: exact within f16's normals), kept as they lie in bf16; each fused product's parts
    /// joined along `out`, int8 codes with their scales or bf16.
    func entries(_ i: Int, _ a: Qwen3Arithmetic) -> [Weights.Entry] {
        let p = prefix + "layers.\(i).", gain: Recipe = a.half ? .half : .bfloat16(joined: [])
        return [("norm1", p + "input_layernorm.weight", gain), ("norm2", p + "post_attention_layernorm.weight", gain),
                ("q_norm", p + "self_attn.q_norm.weight", gain), ("k_norm", p + "self_attn.k_norm.weight", gain),
                ("qkv", p + "self_attn.q_proj", .product(joined: [p + "self_attn.k_proj", p + "self_attn.v_proj"])),
                ("o", p + "self_attn.o_proj", .product(joined: [])),
                ("gate_up", p + "mlp.gate_proj", .product(joined: [p + "mlp.up_proj"])), ("down", p + "mlp.down_proj", .product(joined: []))]
    }

    /// What the cache attention reads beside the token: the rows already written, `[capacity, kvHeads, headDim]`.
    enum Attending { case causal(rows: Int), cached(capacity: Int) }

    /// **One layer over `rows` rows** (the encoder; with `cache`, the prefill: `k` and `v` `[rows, kvHeads, headDim]` out
    /// beside the stream, rotated keys and true values), or **over one token reading the cache** (`decode`).
    func layerModule(_ a: Qwen3Arithmetic, _ attending: Attending, cache: Bool) -> Aicode.Module {
        let c = config, d = c.width, hd = c.headDim, e = a.element
        let rows: Int
        var head: [Qwen3Argument]
        switch attending {
        case .causal(let r):
            rows = r
            head = [("x", [rows, d], e)] + (a.half ? [("x_factor", [1, rows], .f32)] : []) + [("cos", [rows, 1, hd], .f32), ("sin", [rows, 1, hd], .f32)]
        case .cached(let capacity):
            rows = 1
            head = [("x", [1, d], e)] + (a.half ? [("x_factor", [1, 1], .f32)] : []) + [("cos", [1, 1, hd], .f32), ("sin", [1, 1, hd], .f32),
                    ("mask", [capacity + 1], e), ("cache_k", [capacity, c.kvHeads, hd], e), ("cache_v", [capacity, c.kvHeads, hd], e)]
        }
        head += weightArguments(a)
        var m = Aicode.Module(arguments: head.map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        var mask: Int?
        if case .causal = attending {
            // −10⁴ past the diagonal, in the stream's element (bf16 rounds it to −9984: what matters is how far below).
            var bits = [UInt16](repeating: 0, count: rows * rows)
            let minus: UInt16 = a.half ? Float16(-1e4).bitPattern : UInt16(truncatingIfNeeded: BFloat.round(-1e4).bitPattern >> 16)
            for i in 0..<rows { for j in (i + 1)..<rows { bits[i * rows + j] = minus } }
            mask = m.constant("causal", bits.withUnsafeBytes { Array($0) }, Aicode.tensor([rows, rows], e))
        }
        let group = c.heads / c.kvHeads
        let zeros = group > 1 ? m.constant("zeros", [UInt8](repeating: 0, count: group * 2), Aicode.tensor([1, 1, group, 1, 1], e)) : nil
        let b = m.body, names = m.argumentNames
        func arg(_ name: String) -> Int { names.firstIndex(of: name)! }

        func headsFirst(_ t: Int, _ n: Int, _ r: Int) -> Int { b.reshape(b.transpose(t, [1, 0, 2]), [1, n, r, hd]) }
        func repeated(_ t: Int, _ r: Int) -> Int {
            guard let zeros else { return t }
            return b.reshape(b.binary(.add, b.reshape(t, [1, c.kvHeads, 1, r, hd]), zeros), [1, c.heads, r, hd])
        }
        /// q `[1, H, rows, hd]`; k, v `[rows, kvH, hd]`: the attention's output `[1, H, rows, hd]`.
        func attend(_ q: Int, _ k: Int, _ v: Int) -> Int {
            let scale = 1 / Double(hd).squareRoot()
            switch attending {
            case .causal:
                return b.attention([q, repeated(headsFirst(k, c.kvHeads, rows), rows), repeated(headsFirst(v, c.kvHeads, rows), rows), mask!],
                                   scale: scale, symbol: "sdpa_qwen3")
            case .cached(let capacity):
                let keys = b.concat([arg("cache_k"), k], axis: 0), values = b.concat([arg("cache_v"), v], axis: 0)
                let n = capacity + 1
                // The mask given flat: a row of `capacity + 1` elements is bound padded to 64 bytes (its descriptor, 19/11).
                return b.attention([q, repeated(headsFirst(keys, c.kvHeads, n), n), repeated(headsFirst(values, c.kvHeads, n), n), b.reshape(arg("mask"), [1, n])],
                                   scale: scale, symbol: "sdpa_qwen3_cached")
            }
        }
        /// `rotate_half`: x · cos + (−x₂, x₁) · sin over `[rows, n, hd]`.
        func rotatedHalves(_ t: Int, in element: Aicode.Element) -> Int {
            let w = b.castUnlessSame(t, to: element)
            let cs = b.castUnlessSame(arg("cos"), to: element), sn = b.castUnlessSame(arg("sin"), to: element)
            let half = b.concat([b.scale("mul", b.slice(w, axis: 2, start: hd / 2, size: hd / 2), -1), b.slice(w, axis: 2, start: 0, size: hd / 2)], axis: 2)
            return b.binary(.add, b.binary(.mul, w, cs), b.binary(.mul, half, sn))
        }

        let outputs: [(String, Int)]
        if a.half {
            outputs = f16Layer(b, a, rows: rows, arg: arg, attend: attend, rotatedHalves: rotatedHalves)
        } else {
            outputs = bf16Layer(b, a, rows: rows, arg: arg, attend: attend, rotatedHalves: rotatedHalves)
        }
        m.outputs = outputs.filter { cache || !["k", "v"].contains($0.0) || isDecode(attending) }
        return m
    }

    private func isDecode(_ a: Attending) -> Bool { if case .cached = a { true } else { false } }

    /// **The f16 layer**: the stream (x̂, X), each product's input under its own target, a
    /// rotated file's input rotated first (`convrot`: x · Wᵀ = (x · R) · (q · s)ᵀ). Outputs y, y_factor, k, v.
    private func f16Layer(_ b: Aicode.Block, _ a: Qwen3Arithmetic, rows: Int, arg: (String) -> Int, attend: (Int, Int, Int) -> Int,
                          rotatedHalves: (Int, Aicode.Element) -> Int) -> [(String, Int)] {
        let c = config, hd = c.headDim, eps = c.eps
        var constants = Aicode.ScalingConstants()
        func product(_ x: Int, _ name: String) -> Int { b.w8a8(x, codes: arg(name), scale: arg(name + "_scale"), constants: &constants) }
        func target(_ name: String) -> Int { b.target(codes: [(arg(name), arg(name + "_scale"))]) }
        func entering(_ t: Int, _ name: String) -> (unit: Int, factor: Int) { b.ranged(a.rotated ? b.convrot(t) : t, under: target(name)) }
        func column(_ f: Int) -> Int { b.reshape(f, [rows, 1]) }

        let x = arg("x"), xf = column(arg("x_factor"))
        let n1 = b.binary(.mul, b.rmsNorm(x, axes: [1], epsilon: b.epsilon(xf, eps)), arg("norm1"))
        let (u, f) = entering(n1, "qkv")
        let all = product(u, "qkv")
        func part(_ start: Int, _ n: Int) -> Int { b.reshape(b.slice(all, axis: 1, start: start * hd, size: n * hd), [rows, n, hd]) }
        let e = b.reshape(b.epsilon(f, eps), [rows, 1, 1])
        func headNormed(_ t: Int, _ gain: String) -> Int { b.binary(.mul, b.rmsNorm(t, axes: [2], epsilon: e), arg(gain)) }
        func rotated(_ t: Int) -> Int { b.cast(rotatedHalves(t, .f32), to: .f16) }
        let q = headsFirst(b, rotated(headNormed(part(0, c.heads), "q_norm")), c.heads, rows)
        let k = rotated(headNormed(part(c.heads, c.kvHeads), "k_norm"))
        let v = b.binary(.mul, part(c.heads + c.kvHeads, c.kvHeads), b.reshape(b.cast(f, to: .f16), [rows, 1, 1]))
        let o = attend(q, k, v)
        let rowsFirst = b.reshape(b.transpose(b.reshape(o, [c.heads, rows, hd]), [1, 0, 2]), [rows, c.heads * hd])
        let (ou, of) = entering(rowsFirst, "o")
        let (x1, x1f) = b.residual(x, xf, plus: product(ou, "o"), factor: of)

        let n2 = b.binary(.mul, b.rmsNorm(x1, axes: [1], epsilon: b.epsilon(x1f, eps)), arg("norm2"))
        let (r, f2) = entering(n2, "gate_up")
        let gu = product(r, "gate_up")
        let gate = b.slice(gu, axis: 1, start: 0, size: c.hidden), up = b.slice(gu, axis: 1, start: c.hidden, size: c.hidden)
        // Under 2¹⁰ (`gatedSiLU`); rotated, under 2¹⁴ before its own target.
        let (h, fh) = b.gatedSiLU(gate, up, factor: f2) { entering($0, "down") }
        let (y, yf) = b.residual(x1, x1f, plus: product(h, "down"), factor: fh)
        return [("y", y), ("y_factor", b.reshape(yf, [1, rows])), ("k", k), ("v", v)]
    }

    /// **The bf16 layer**, transformers' roundings op by op: RMSNorm's normalized value rounded, then × γ; RoPE in bf16 on
    /// tables brought to bf16 (`cos.to(x.dtype)`); SiLU in f32, rounded once (torch's kernel); every sum in bf16.
    private func bf16Layer(_ b: Aicode.Block, _ a: Qwen3Arithmetic, rows: Int, arg: (String) -> Int, attend: (Int, Int, Int) -> Int,
                           rotatedHalves: (Int, Aicode.Element) -> Int) -> [(String, Int)] {
        let c = config, hd = c.headDim, eps = c.eps
        func product(_ x: Int, _ name: String) -> Int { b.linear(x, arg(name)) }
        func rms(_ t: Int, _ gain: String, axis: Int) -> Int { b.binary(.mul, b.rmsNorm(t, axes: [axis], eps: eps), arg(gain)) }
        let x = arg("x")
        let all = product(rms(x, "norm1", axis: 1), "qkv")
        func part(_ start: Int, _ n: Int) -> Int { b.reshape(b.slice(all, axis: 1, start: start * hd, size: n * hd), [rows, n, hd]) }
        let q = headsFirst(b, rotatedHalves(rms(part(0, c.heads), "q_norm", axis: 2), .bf16), c.heads, rows)
        let k = rotatedHalves(rms(part(c.heads, c.kvHeads), "k_norm", axis: 2), .bf16)
        let v = part(c.heads + c.kvHeads, c.kvHeads)
        let o = attend(q, k, v)
        let rowsFirst = b.reshape(b.transpose(b.reshape(o, [c.heads, rows, hd]), [1, 0, 2]), [rows, c.heads * hd])
        let x1 = b.binary(.add, x, product(rowsFirst, "o"))
        let gu = product(rms(x1, "norm2", axis: 1), "gate_up")
        let gate = b.slice(gu, axis: 1, start: 0, size: c.hidden), up = b.slice(gu, axis: 1, start: c.hidden, size: c.hidden)
        let h = b.binary(.mul, b.cast(b.silu(b.cast(gate, to: .f32)), to: .bf16), up)
        let y = b.binary(.add, x1, product(h, "down"))
        return [("y", y), ("k", k), ("v", v)]
    }

    private func headsFirst(_ b: Aicode.Block, _ t: Int, _ n: Int, _ rows: Int) -> Int {
        b.reshape(b.transpose(t, [1, 0, 2]), [1, n, rows, config.headDim])
    }


    // MARK: - The head

    /// **The head over a stretch of the vocabulary** (`lm_head(norm(x))`), its logits written flat (`[rows · stretch]`: a
    /// row of 9496 f32 is bound padded to 9504, its descriptor read 19/11): the final norm, then the product by `stretch`
    /// rows of `lm_head` as the file lays it (`[stretch, width]`: a stretch of rows is one piece of the arena, and a
    /// head reads one row a token), summed and kept in f32 — a bf16 logit near 20 is 0.125 wide, more than the gap between
    /// the two best tokens at many a step. In f16 on codes: the normed row under the stretch's target, the logits its
    /// factor times the product.
    func headModule(_ a: Qwen3Arithmetic, rows: Int, stretch: Int) -> Aicode.Module {
        let d = config.width, e = a.element
        var args: [Qwen3Argument] = [("x", [rows, d], e)]
        if a.half { args += [("x_factor", [1, rows], .f32), ("norm", [d], e), ("w", Weights.codesShape(inn: d, out: stretch), .si8), ("w_scale", [stretch], e)] }
        else { args += [("norm", [d], e), ("w", [stretch, d], e)] }
        var m = Aicode.Module(arguments: args.map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        let b = m.body, names = m.argumentNames
        func arg(_ name: String) -> Int { names.firstIndex(of: name)! }
        guard a.half else {
            let n = b.binary(.mul, b.rmsNorm(arg("x"), axes: [1], eps: config.eps), arg("norm"))
            m.outputs = [("logits", b.reshape(b.linear(b.cast(n, to: .f32), b.cast(arg("w"), to: .f32)), [rows * stretch]))]
            return m
        }
        var constants = Aicode.ScalingConstants()
        let xf = b.reshape(arg("x_factor"), [rows, 1])
        let n = b.binary(.mul, b.rmsNorm(arg("x"), axes: [1], epsilon: b.epsilon(xf, config.eps)), arg("norm"))
        let e0 = b.target(codes: [(arg("w"), arg("w_scale"))])
        let (u, f) = b.ranged(a.rotated ? b.convrot(n) : n, under: e0)
        let y = b.w8a8(u, codes: arg("w"), scale: arg("w_scale"), constants: &constants)
        m.outputs = [("logits", b.reshape(b.binary(.mul, b.cast(y, to: .f32), f), [rows * stretch]))]
        return m
    }
}
