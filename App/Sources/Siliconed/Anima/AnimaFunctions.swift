import Foundation
import SiliCore

/// **Anima's DiT in functions** (`AnimaCosmosDiT` says the math): each a Core AI function at fixed shapes, every weight
/// an argument, written op by op. Two paths, one per variant:
///
///   - **bf16**, the Standard: every value in bf16 as diffusers runs it, the GPU alone;
///   - **f16**, the Compact: W8A16 on Bedovyy's int8 `convrot` codes (the activation rotated by the file's Hadamard,
///     then the codes dequantized in f16), the products the file leaves in bf16 (blocks 0 and 27, the MLP of blocks 1 to 3) rounded once to f16, under the guard
///     of f16 (`Guard.swift`): the residual stream kept as (x̂, X), each product's input brought under its target.
///
/// The one-row functions (`time`, `modulation`) compute in f32 on both paths, the bf16 weights widened inside (exact):
/// their values are the reference's, rounded once at their outputs.
///
///     time         t_freq → temb [1, 3d], embedded [1, d] (f32); the final layer's shift and 1 + scale [2, d]
///     modulation   embedded, temb → each sublayer's shift, 1 + scale, gate: self, cross, mlp [3, d]
///     context      the conditioner's context → each block's cross keys and values `ctx_k`, `ctx_v`, heads first (once a prompt)
///     embed        latent [16, h, w] → patchify → x_embedder → the residual stream's rows
///     pre          a tile: LN · (1 + scale₁) + shift₁ → q, k, v (QK norm, RoPE); k, v into the whole sequence's
///     post         a tile: self-attention · output_proj → x₁; cross-attention → x₂; GELU MLP → y
///     final        LN · (1 + scale) + shift → final_layer.linear → unpatchify → velocity [16, h, w] f32
///
/// Every tile's pre runs before any tile's post: post's attention reads the keys and values of every row. A block is
/// pre and post on the same weights: its modulation once, then the tiles.
package struct AnimaFunction: Sendable {
    enum Kind: String { case time, modulation, context, embed, pre, post, final }
    enum Path: String { case bf16, f16 }

    let kind: Kind
    let path: Path
    /// The function's products on int8 codes (the Compact's, which leaves blocks 0 and 27 and the MLP of blocks 1 to 3 on
    /// their bf16 weights); the others on float weights. Only the function's own (`products`).
    var quantized: Set<String> = []
    /// pre, post: the tile's rows; embed, final: the image's tokens.
    var rows = 1
    /// post: the self-attention's keys (the image's rows, padded); context, post: the cross-attention's keys.
    var keys = 0, context = 0
    /// embed, final: the latent's sides.
    var height = 0, width = 0
    /// post: block 0's three branches as outputs besides (the oracle's boundaries; a check's alone).
    var taps = false
    /// How the codes lie (the stage's plan's, this Mac's measure): `[in, out]` or the file's `[out, in]`.
    var codes: CodesLayout = .inOut
    /// The low-rank path beside the products a LoRA adapts (`LoRAStack`): each one's rank, the same for every block that
    /// calls the function; none, the function as it is.
    var ranks: [String: Int] = [:]

    typealias Argument = (name: String, shape: [Int], element: Aicode.Element)

    package static let d = 2048, heads = 16, hd = 128, hidden = 8192, low = 256, textDim = 1024, channels = 16, patch = 2
    /// A row of the modulation's (`final`, `self`, `cross`, `mlp`): each starts a page of its own, its readers taking one
    /// row each, a buffer of its own (`Layout.pageRow`).
    package static let gainRow = Layout.pageRow(d, .bfloat16)
    /// The patch's values: the latent's channels and the padding mask's (zeros), `patch²` each.
    static let patchValues = (channels + 1) * patch * patch
    static let outputValues = patch * patch * channels

    /// The element the block's values travel in.
    var element: Aicode.Element { path == .bf16 ? .bf16 : .f16 }
    var guarded: Bool { path == .f16 }

    /// The products a function of `kind` reads, by their arguments' names.
    static func products(_ kind: Kind) -> [String] {
        switch kind {
        case .context: ["ck", "cv"]
        case .pre: ["q", "k", "v"]
        case .post: ["o", "cq", "co", "m1", "m2"]
        case .time, .modulation, .embed, .final: []
        }
    }

    /// The products a LoRA may adapt in a function of `kind`: its own, and the modulation's.
    static func adapted(_ kind: Kind) -> [String] {
        kind == .modulation ? ["sa1", "sa2", "ca1", "ca2", "ml1", "ml2"] : products(kind)
    }

    /// A product's `[in, out]`, by its argument's name.
    static func dims(_ name: String) -> (inn: Int, out: Int) {
        switch name {
        case "ck", "cv": (textDim, d)
        case "m1": (d, hidden)
        case "m2": (hidden, d)
        case "sa1", "ca1", "ml1": (d, low)
        case "sa2", "ca2", "ml2": (low, 3 * d)
        default: (d, d)
        }
    }

    /// The LoRA factors' element: the path's.
    var factors: Aicode.Element { path == .bf16 ? .bf16 : .f16 }

    // MARK: - Arguments

    /// A product's weight as the function reads it: int8 codes `[in, out]` and their scales `[out]`, or the float weight.
    private func product(_ name: String, _ inn: Int, _ out: Int) -> [Argument] {
        if guarded && quantized.contains(name) { return [(name, codes == .inOut ? [inn, out] : [out, inn], .si8), (name + "_scale", [out], .f16)] }
        return [(name, [inn, out], element)]
    }

    /// The function's arguments in the order it declares them: its own, then each adapted product's factors.
    var arguments: [Argument] {
        own + ranks.sorted { $0.key < $1.key }.flatMap { name, r -> [Argument] in
            let (inn, out) = Self.dims(name)
            return [(name + "_down", [inn, r], factors), (name + "_up", [r, out], factors)]
        }
    }

    private var own: [Argument] {
        let (d, hd, e) = (Self.d, Self.hd, element)
        let row = [1, d]
        let stream: [Argument] = guarded ? [("x", [rows, d], .f16), ("x_factor", [1, rows], .f32)] : [("x", [rows, d], e)]
        switch kind {
        case .time:
            return [("t_freq", [1, d], .f32), ("t1", [d, d], .bf16), ("t2", [d, 3 * d], .bf16), ("t_norm", [d], .bf16),
                    ("f1", [d, Self.low], .bf16), ("f2", [Self.low, 2 * d], .bf16)]
        case .modulation:
            return [("embedded", row, .f32), ("temb", [1, 3 * d], .f32)]
                + ["sa", "ca", "ml"].flatMap { [($0 + "1", [d, Self.low], Aicode.Element.bf16), ($0 + "2", [Self.low, 3 * d], .bf16)] }
        case .context:
            return [("context", [context, Self.textDim], e)] + product("ck", Self.textDim, d) + product("cv", Self.textDim, d) + [("ck_norm", [hd], e)]
        case .embed:
            return [("latent", [Self.channels, height, width], .f32), ("xe", [Self.patchValues, d], e)]
        case .final:
            return stream + [("shift", row, e), ("scale", row, e), ("fl", [d, Self.outputValues], e)]
        case .pre:
            return stream + [("shift1", row, e), ("scale1", row, e)] + product("q", d, d) + product("k", d, d) + product("v", d, d)
                + [("q_norm", [hd], e), ("k_norm", [hd], e), ("cos", [rows, 1, hd / 2], .f32), ("sin", [rows, 1, hd / 2], .f32)]
        case .post:
            let attention: [Argument] = [("q", [1, Self.heads, rows, hd], e), ("k", [1, keys, Self.heads, hd], e), ("v", [1, keys, Self.heads, hd], e),
                                         ("mask", [1, keys], e)]
            let gains: [Argument] = ["gate1", "shift2", "scale2", "gate2", "shift3", "scale3", "gate3"].map { ($0, row, e) }
            let cross: [Argument] = [("ctx_k", [1, Self.heads, context, hd], e), ("ctx_v", [1, Self.heads, context, hd], e), ("ctx_mask", [1, context], e)]
            return stream + attention + gains + product("o", d, d) + product("cq", d, d) + [("cq_norm", [hd], e)] + cross
                + product("co", d, d) + product("m1", d, Self.hidden) + product("m2", Self.hidden, d)
        }
    }

    // MARK: - The module

    func module() -> Aicode.Module {
        var m = Aicode.Module(arguments: arguments.map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        let b = m.body
        let names = m.argumentNames
        func a(_ name: String) -> Int { names.firstIndex(of: name)! }
        var constants = Aicode.ScalingConstants()
        let (d, heads, hd, e) = (Self.d, Self.heads, Self.hd, element)

        /// `x · W` with W `[in, out]` in x's element.
        func linear(_ x: Int, _ w: Int) -> Int {
            let (xs, element) = Aicode.shape(b.types[x])
            return b.matmul(x, w, result: Aicode.tensor(Array(xs.dropLast()) + [Aicode.shape(b.types[w]).shape.last!], element))
        }
        /// `y` with the LoRA's low-rank path of the product `name` beside it, x what the product read: y as it is when
        /// nothing adapts it.
        func adapted(_ y: Int, _ x: Int, _ name: String) -> Int {
            ranks[name] == nil ? y : b.lowRank(y, x, down: a(name + "_down"), up: a(name + "_up"))
        }
        /// The bf16 path's product of `name`, its LoRA beside.
        func product(_ x: Int, _ name: String) -> Int { adapted(linear(x, a(name)), x, name) }
        /// A one-row product in f32: the bf16 weight widened (exact), its LoRA beside.
        func wide(_ x: Int, _ w: String) -> Int { adapted(linear(x, b.cast(a(w), to: .f32)), x, w) }
        func rowsOf(_ t: Int, _ i: Int, _ n: Int) -> Int { b.slice(t, axis: 0, start: i, size: n) }
        func columns(_ t: Int, _ i: Int, _ n: Int) -> Int { b.slice(t, axis: 1, start: i, size: n) }
        /// diffusers' `RMSNorm` with a gain over a head's last axis: the normalized value rounded to the element, then × γ.
        func headNorm(_ t: Int, _ gain: String, epsilon: Int? = nil) -> Int {
            let n = epsilon.map { b.rmsNorm(t, axes: [2], epsilon: $0) } ?? b.rmsNorm(t, axes: [2], eps: 1e-5)
            return b.binary(.mul, n, a(gain))
        }
        func headsFirst(_ t: Int, _ l: Int) -> Int { b.reshape(b.transpose(t, [1, 0, 2]), [1, heads, l, hd]) }
        // The self-attention's keys and values token first, `[1, l, heads, 128]`: a tile's a block of the sequence's, written
        // whole, their heads taken first inside the attention (measured 19/11 on Z: the same bits, the write twice faster).
        func tokenFirst(_ t: Int, _ l: Int) -> Int { b.reshape(t, [1, l, heads, hd]) }
        /// `apply_rotary_emb(use_real_unbind_dim=-2)`: a head's halves `(x[p], x[64 + p])` rotated in f32, rounded once.
        func rope(_ t: Int, _ l: Int) -> Int {
            let f = b.cast(t, to: .f32), half = hd / 2
            let x0 = b.slice(f, axis: 2, start: 0, size: half), x1 = b.slice(f, axis: 2, start: half, size: half)
            let (c, s) = (a("cos"), a("sin"))
            let first = b.binary(.sub, b.binary(.mul, x0, c), b.binary(.mul, x1, s))
            let second = b.binary(.add, b.binary(.mul, x1, c), b.binary(.mul, x0, s))
            return b.cast(b.concat([first, second], axis: 2), to: e)
        }
        func attend(_ q: Int, _ k: Int, _ v: Int, _ mask: Int, _ l: Int, symbol: String) -> Int {
            let o = b.attention([q, k, v, mask], scale: 1 / Double(hd).squareRoot(), symbol: symbol)
            return b.reshape(b.transpose(b.reshape(o, [heads, l, hd]), [1, 0, 2]), [l, d])
        }
        /// `CosmosAdaLayerNormZero`'s norm: LN without affine (ε = 10⁻⁶) · (1 + scale) + shift, both rows given. Guarded,
        /// LN reads x̂ with ε / X² (`epsilon`): LN(x̂ X) = LN(x̂) but for ε, so its value is true.
        func modulated(_ x: Int, _ shift: String, _ scale: String, epsilon: Int? = nil) -> Int {
            let n = b.layerNorm(x, axes: [1], epsilon: epsilon ?? b.scalar(1e-6, .f32))
            return b.binary(.add, b.binary(.mul, n, a(scale)), a(shift))
        }
        /// Guarded: x̂'s ε a row, from its factor `[1, rows]` (or `[rows, 1]`).
        func epsilon(_ f: Int, _ l: Int) -> Int { b.epsilon(b.reshape(f, [l, 1]), 1e-6) }

        /// **A product of the guarded path**: x (a true value, f16 or f32) rotated by the file's Hadamard when the weight is
        /// codes, brought under the product's target, multiplied: (result, its factor `[rows, 1]` f32). `names`: the products
        /// reading the same input (one rotation, one target, the smallest).
        func guardedProducts(_ x: Int, _ names: [String]) -> (results: [Int], factors: [Int]) {
            let (results, f): ([Int], Int)
            if names.allSatisfy(quantized.contains) {
                let target = b.target(codes: names.map { (a($0), a($0 + "_scale")) }, layout: codes)
                let (unit, g) = b.ranged(b.convrot(x), under: target)
                (results, f) = (names.map { b.w8a16(unit, codes: a($0), scale: a($0 + "_scale"), layout: codes, constants: &constants) }, g)
            } else {
                precondition(!names.contains(where: quantized.contains), "\(kind): products of one input on codes and on floats")
                let target = b.target(weights: names.map { a($0) })
                let (unit, g) = b.ranged(x, under: target)
                (results, f) = (names.map { b.product(unit, a($0)) }, g)
            }
            // A product a LoRA adapts: its true value with the low-rank path beside (x as the product read it, before any
            // rotation), brought under a factor of its own.
            var factors = [Int](repeating: f, count: names.count), out = results
            for (i, n) in names.enumerated() where ranks[n] != nil {
                (out[i], factors[i]) = b.ranged(adapted(trueValue(results[i], f), x, n))
            }
            return (out, factors)
        }
        /// A guarded product's result multiplied out, in f32: what an op its row factor cannot pass reads.
        func trueValue(_ r: Int, _ f: Int) -> Int { b.binary(.mul, b.cast(r, to: .f32), f) }
        /// The residual stream's update by gate · branch (the branch true, f32): (x̂', X') `[rows, 1]`. The update is held
        /// as high as the stream (`stream`), not under 1: `residual` brings both under C = 2 · max(X, 4G) by X / C and G / C
        /// in f16, and a G of the update's size beside Anima's small stream (X ≈ 2⁻¹² at the embedder) put X / C under
        /// f16's normals, which the GPU flushes: block 0 lost its stream (measured 18/11 on the M5 Pro, the Compact's
        /// block0_out 1.11 from the oracle, the bf16 reference 0.16). Held high, X / C leaves the normals only for an
        /// update 2¹³ times the stream.
        func updated(_ x: Int, _ xf: Int, _ gate: String, _ branch: Int) -> (Int, Int) {
            let (delta, g) = b.ranged(b.binary(.mul, branch, b.cast(a(gate), to: .f32)), under: b.stream)
            return b.residual(x, xf, plus: delta, factor: g)
        }

        switch kind {
        case .time:
            // `CosmosEmbedding` in f32: temb = linear_2(SiLU(linear_1(t))), embedded = RMSNorm(t) · γ (ε = 10⁻⁶); the
            // final layer's modulation, linear(linear(SiLU(embedded))) + temb[:2d], its shift and 1 + scale.
            let t = a("t_freq")
            let temb = wide(b.silu(wide(t, "t1")), "t2")
            let embedded = b.binary(.mul, b.rmsNorm(t, axes: [1], eps: 1e-6), b.cast(a("t_norm"), to: .f32))
            let mf = b.binary(.add, wide(wide(b.silu(embedded), "f1"), "f2"), columns(temb, 0, 2 * d))
            let final = b.concat([columns(mf, 0, d), b.scale("add", columns(mf, d, d), 1)], axis: 0)
            m.outputs = [("temb", temb), ("embedded", embedded), ("final", b.zeroPadded(b.cast(final, to: e), to: [2, Self.gainRow]))]
        case .modulation:
            // Each sublayer's [shift, scale, gate] = linear(linear(SiLU(embedded))) + temb, as rows: shift, 1 + scale, gate.
            let s = b.silu(a("embedded"))
            m.outputs = [("self", "sa"), ("cross", "ca"), ("mlp", "ml")].map { name, p in
                let m3 = b.reshape(b.binary(.add, wide(wide(s, p + "1"), p + "2"), a("temb")), [3, d])
                let rows3 = b.concat([rowsOf(m3, 0, 1), b.scale("add", rowsOf(m3, 1, 1), 1), rowsOf(m3, 2, 1)], axis: 0)
                return (name, b.zeroPadded(b.cast(rows3, to: e), to: [3, Self.gainRow]))
            }
        case .context:
            let l = context, c = a("context")
            if guarded {
                let (r, f) = guardedProducts(c, ["ck", "cv"])
                let k = headNorm(b.reshape(r[0], [l, heads, hd]), "ck_norm", epsilon: b.reshape(b.epsilon(f[0], 1e-5), [l, 1, 1]))
                let v = b.cast(trueValue(r[1], f[1]), to: .f16)
                m.outputs = [("ctx_k", headsFirst(k, l)), ("ctx_v", headsFirst(b.reshape(v, [l, heads, hd]), l))]
            } else {
                let k = headNorm(b.reshape(product(c, "ck"), [l, heads, hd]), "ck_norm")
                m.outputs = [("ctx_k", headsFirst(k, l)), ("ctx_v", headsFirst(b.reshape(product(c, "cv"), [l, heads, hd]), l))]
            }
        case .embed:
            // `CosmosPatchEmbed`: "c (h ph) (w pw) → (h w) (c ph pw)", the padding mask's channel (zeros) appended.
            let (h, w, p) = (height / Self.patch, width / Self.patch, Self.patch)
            let latent = guarded ? a("latent") : b.cast(a("latent"), to: e)
            let zeros = b.scale("mul", b.slice(latent, axis: 0, start: 0, size: 1), 0)
            let patches = b.reshape(b.transpose(b.reshape(b.concat([latent, zeros], axis: 0), [Self.channels + 1, h, p, w, p]), [1, 3, 0, 2, 4]),
                                    [h * w, Self.patchValues])
            if guarded {
                let (x, f) = b.ranged(linear(patches, b.cast(a("xe"), to: .f32)), under: b.stream)
                m.outputs = [("x", x), ("x_factor", b.reshape(f, [1, h * w]))]
            } else {
                m.outputs = [("x", linear(patches, a("xe")))]
            }
        case .final:
            let (h, w, p, l) = (height / Self.patch, width / Self.patch, Self.patch, rows)
            let y: Int
            if guarded {
                let n = modulated(a("x"), "shift", "scale", epsilon: epsilon(a("x_factor"), l))
                let (r, f) = guardedProducts(n, ["fl"])
                y = trueValue(r[0], f[0])
            } else {
                y = b.cast(linear(modulated(a("x"), "shift", "scale"), a("fl")), to: .f32)
            }
            // The rows `[h·w, p·p·c]` back to `[c, h·p, w·p]`: `(ph, pw, c)`-major.
            let image = b.reshape(b.transpose(b.reshape(y, [h, w, p, p, Self.channels]), [4, 0, 2, 1, 3]), [Self.channels, height, width])
            m.outputs = [("velocity", image)]
        case .pre:
            let l = rows
            if guarded {
                let n = modulated(a("x"), "shift1", "scale1", epsilon: epsilon(a("x_factor"), l))
                let (r, f) = guardedProducts(n, ["q", "k", "v"])
                func normed(_ i: Int, _ gain: String) -> Int {
                    headNorm(b.reshape(r[i], [l, heads, hd]), gain, epsilon: b.reshape(b.epsilon(f[i], 1e-5), [l, 1, 1]))
                }
                let v = b.cast(trueValue(r[2], f[2]), to: .f16)
                m.outputs = [("q", headsFirst(rope(normed(0, "q_norm"), l), l)), ("k", tokenFirst(rope(normed(1, "k_norm"), l), l)),
                             ("v", tokenFirst(b.reshape(v, [l, heads, hd]), l))]
            } else {
                let n = modulated(a("x"), "shift1", "scale1")
                func normed(_ w: String, _ gain: String) -> Int { headNorm(b.reshape(product(n, w), [l, heads, hd]), gain) }
                m.outputs = [("q", headsFirst(rope(normed("q", "q_norm"), l), l)), ("k", tokenFirst(rope(normed("k", "k_norm"), l), l)),
                             ("v", tokenFirst(b.reshape(product(n, "v"), [l, heads, hd]), l))]
            }
        case .post:
            let l = rows
            let o = attend(a("q"), b.transpose(a("k"), [0, 2, 1, 3]), b.transpose(a("v"), [0, 2, 1, 3]), a("mask"), l, symbol: "sdpa_self")
            var taps: [(String, Int)] = []
            if guarded {
                let (r1, f1) = guardedProducts(o, ["o"])
                let o1 = trueValue(r1[0], f1[0])
                let (x1, x1f) = updated(a("x"), b.reshape(a("x_factor"), [l, 1]), "gate1", o1)
                let n2 = modulated(x1, "shift2", "scale2", epsilon: b.epsilon(x1f, 1e-6))
                let (rq, fq) = guardedProducts(n2, ["cq"])
                let q2 = headNorm(b.reshape(rq[0], [l, heads, hd]), "cq_norm", epsilon: b.reshape(b.epsilon(fq[0], 1e-5), [l, 1, 1]))
                let cross = attend(headsFirst(q2, l), a("ctx_k"), a("ctx_v"), a("ctx_mask"), l, symbol: "sdpa_cross")
                let (r2, f2) = guardedProducts(cross, ["co"])
                let o2 = trueValue(r2[0], f2[0])
                let (x2, x2f) = updated(x1, x1f, "gate2", o2)
                let n3 = modulated(x2, "shift3", "scale3", epsilon: b.epsilon(x2f, 1e-6))
                let (r3, f3) = guardedProducts(n3, ["m1"])
                // GELU of the true value, in f32: x · Φ(x) passes no row factor through Φ.
                let (r4, f4) = guardedProducts(b.gelu(trueValue(r3[0], f3[0]), tanh: false), ["m2"])
                let ff = trueValue(r4[0], f4[0])
                let (y, yf) = updated(x2, x2f, "gate3", ff)
                m.outputs = [("y", y), ("y_factor", b.reshape(yf, [1, l]))]
                taps = [("attn1", o1), ("attn2", o2), ("ff", ff)]
            } else {
                let o1 = product(o, "o")
                let x1 = b.binary(.add, a("x"), b.binary(.mul, a("gate1"), o1))
                let n2 = modulated(x1, "shift2", "scale2")
                let q2 = headNorm(b.reshape(product(n2, "cq"), [l, heads, hd]), "cq_norm")
                let o2 = product(attend(headsFirst(q2, l), a("ctx_k"), a("ctx_v"), a("ctx_mask"), l, symbol: "sdpa_cross"), "co")
                let x2 = b.binary(.add, x1, b.binary(.mul, a("gate2"), o2))
                let ff = product(b.gelu(product(modulated(x2, "shift3", "scale3"), "m1"), tanh: false), "m2")
                m.outputs = [("y", b.binary(.add, x2, b.binary(.mul, a("gate3"), ff)))]
                taps = [("attn1", b.cast(o1, to: .f32)), ("attn2", b.cast(o2, to: .f32)), ("ff", b.cast(ff, to: .f32))]
            }
            if self.taps { m.outputs += taps }
        }
        return m
    }
}
