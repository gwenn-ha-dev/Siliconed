import CoreAI
import Foundation
import SiliCore

/// **Qwen-Image-2.1's DiT in functions at fixed shapes** (the loop is `QwenImage21Loop`), on one of three paths:
///
///     bfloat   the Standard's bf16 weights, bf16 products on the GPU, the residual stream in f32
///     half     the Compact's int8 codes in W8A16: dequantized in f16 inside the function, every activation under the
///              guard of f16 (`Guard.swift`), the stream as (x̂ f16, a row's factor f32); its input rotated by ConvRot's
///              Hadamard (256) before each product, as the codes were
///     shared   the same codes in the forms the GPU and the ANE both take (`QwenImage21Shared`): f16 at every border,
///              the stream as (x̂ f16, a row's factor f16), the layer in five functions of which the ANE takes three
///
///     timestep        one row, f32   temb = linear_2(SiLU(linear_1 t_freq)); m = modulation(SiLU temb) as the rows
///                                    1 + s₁, tanh g₁, 1 + s₂, tanh g₂; final_scale = 1 + norm_out(SiLU temb)
///     embed-prefix    the prefix     text rows: txt_in (zero-centred RMS · Linear · GELU tanh · Linear); reference
///                                    rows: img_in — each row one or the other (`which`), the stream
///     embed-target    the target     img_in on its tokens, the stream
///     pre             a tile         LN(x)·(1 + s₁) · to_q, to_k, to_v; q, k: RoPE(RMS_head · γ), as heads
///     post            a tile         x₁ = x + tanh g₁ · to_out(sdpa(q, [kₚ | kₜ], [vₚ | vₜ]) + mask);
///                                    y = x₁ + tanh g₂ · out(SiLU(gate) · up), [gate | up] = gate_up(LN(x₁)·(1 + s₂))
///     final           the target     LN(x) · final_scale · proj_out
///
/// The outer functions compute in f32 on both paths, their weights widened from the file's bf16 (exact): their cost is
/// nothing beside the layers', and one path for them is one less to judge. The layers have no bias and no sandwich
/// norm: under the guard, a product's update enters the stream as (Δ̂, G) (`residual(_:_:plus:factor:)`).
///
/// **The keys**: a post reads the layer's prefix keys and values (written once a render, `kp`, `vp`) and the target's
/// (written by each step's pre, `kt`, `vt`), concatenated inside the function, under a mask `[rows, keys]`: the prefix's
/// block-causal one for its rows (text causal, a picture a block, every row seeing what precedes it), the target's
/// every real key. One post serves both passes; the target's keys are zero while the prefix runs, masked by −10⁴.
///
/// **Under a LoRA** (`LoRAStack`), a product a LoRA adapts reads its factors beside its weight (`Aicode.Block.lowRank`),
/// at the function's ranks (`Ranks`: one function serves every layer, at the largest rank landing on its product in any
/// of them), on the product's input before any rotation — the model's own basis, where the factors were trained. Under
/// the guard, that input is taken under the factor of the product's (the value before the rotation over the factor
/// `ranged` found for it): the change then comes out under the product's own factor, added to its result as it lies.
package enum QwenImage21Functions {
    package enum Path: String, Sendable {
        case bfloat, half, shared

        /// The element of what a layer reads and writes.
        var element: Aicode.Element { self == .bfloat ? .bf16 : .f16 }
        var scalar: NDArray.ScalarType { self == .bfloat ? .bfloat16 : .float16 }
        /// The element of the stream's row factors: f32 under the GPU's guard, f16 in the shared forms.
        var factor: Aicode.Element { self == .shared ? .f16 : .f32 }
        /// The modulation's rows as the layers read them: f32 beside the bf16 path's f32 stream, f16 under the guard.
        var gains: Aicode.Element { self == .bfloat ? .f32 : .f16 }
        var gainsScalar: NDArray.ScalarType { self == .bfloat ? .float32 : .float16 }
        /// A row of `mod`, `[4, gainRow]`: each starts a page of its own, its readers taking one row each, a buffer of its
        /// own (`Layout.pageRow`).
        var gainRow: Int { Layout.pageRow(QwenImage21Functions.d, gainsScalar) }
        var gainBytes: Int { gainRow * Layout.element(gainsScalar) }
    }

    static let c = QwenImage21DiT.Config.published
    static var d: Int { c.dim }
    static var heads: Int { c.heads }
    static var hd: Int { c.headDim }

    typealias Argument = (name: String, shape: [Int], element: Aicode.Element)
    /// The rank of each adapted product's factors, by its argument's name (`ranks(_:)`); none: no LoRA.
    package typealias Ranks = [String: Int]

    // MARK: - The LoRAs

    /// The products outside the layers: (argument, module, in, out).
    static let outerProducts: [(String, String, Int, Int)] = [
        ("linear_1", "time_text_embed.timestep_embedder.linear_1", c.timeDim, c.dim), ("linear_2", "time_text_embed.timestep_embedder.linear_2", c.dim, c.dim),
        ("modulation", "modulation.1", c.dim, 4 * c.dim), ("norm_out", "norm_out.linear", c.dim, c.dim),
        ("txt_in1", "txt_in.in_layer", c.context, c.dim), ("txt_in2", "txt_in.out_layer", c.dim, c.dim),
        ("img_in", "img_in", c.channels, c.dim), ("proj_out", "proj_out", c.dim, c.channels)]

    /// The products function `f` computes (its arguments, in order).
    static func adapted(_ f: String) -> [String] {
        switch f {
        case "timestep": return ["linear_1", "linear_2", "modulation", "norm_out"]
        case "embed-prefix": return ["txt_in1", "txt_in2", "img_in"]
        case "embed-target": return ["img_in"]
        case "final": return ["proj_out"]
        case "pre": return ["to_q", "to_k", "to_v"]
        case "post": return ["to_out", "gate_up", "down"]
        case "out": return ["to_out"]
        case "ffn": return ["gate_up", "down"]
        default: return []
        }
    }

    /// The products outside the layers as a LoRA lands on them.
    static func outerLoRA() -> [LoRAProduct] { outerProducts.map { LoRAProduct(name: $0.0, key: $0.1, inn: $0.2, out: $0.3) } }

    /// Layer `i`'s products as a LoRA lands on them.
    static func layerLoRA(_ i: Int) -> [LoRAProduct] {
        let (d, hidden) = (c.dim, c.hidden)
        let shapes: [String: (Int, Int)] = ["to_q": (d, d), "to_k": (d, d), "to_v": (d, d), "to_out": (d, d), "gate_up": (d, 2 * hidden), "down": (hidden, d)]
        return products.map { LoRAProduct(name: $0.0, key: "transformer_blocks.\(i).\($0.1)", inn: shapes[$0.0]!.0, out: shapes[$0.0]!.1) }
    }

    /// **The ranks `stack` asks of the functions**: each product's, the largest landing on it in any layer.
    package static func ranks(_ stack: LoRAStack) -> Ranks {
        guard !stack.isEmpty else { return [:] }
        var ranks = Ranks()
        for p in outerLoRA() { ranks[p.name] = stack.rank(of: [p.key], on: .qwenImage21) }
        let layers = (0..<c.layers).map(layerLoRA)
        for (k, p) in layers[0].enumerated() { ranks[p.name] = stack.rank(of: layers.map { $0[k].key }, on: .qwenImage21) }
        return ranks.filter { $0.value > 0 }
    }

    /// The factors' element: f32 outside the layers (their functions compute in f32); a layer's, its path's.
    static func factorElement(_ product: String, path: Path) -> NDArray.ScalarType {
        outerProducts.contains { $0.0 == product } ? .float32 : path.scalar
    }

    /// The factors' arguments of function `f`'s adapted products, `[in, R]` and `[R, out]`.
    static func factors(_ f: String, path: Path, ranks: Ranks) -> [Argument] {
        let outer = outerLoRA(), layer = layerLoRA(0)
        return adapted(f).flatMap { name -> [Argument] in
            guard let r = ranks[name], r > 0, let p = (outer + layer).first(where: { $0.name == name }) else { return [] }
            let e: Aicode.Element = factorElement(name, path: path) == .float32 ? .f32 : path.element
            return [(name + "_down", [p.inn, r], e), (name + "_up", [r, p.out], e)]
        }
    }

    // MARK: - The weights

    /// A layer's products: (argument, module).
    static let products = [("to_q", "attn.to_q"), ("to_k", "attn.to_k"), ("to_v", "attn.to_v"), ("to_out", "attn.to_out.0"),
                           ("gate_up", "img_mlp.gate_up"), ("down", "img_mlp.out")]
    static let gains = [("norm_q", "attn.norm_q.weight"), ("norm_k", "attn.norm_k.weight")]

    /// Layer `i`'s weights, by the file's keys.
    static func layer(_ i: Int, path: Path, weights: SharedWeights = .codes) -> [Weights.Entry] {
        let p = "transformer_blocks.\(i)."
        switch path {
        case .half: return products.map { ($0.0, p + $0.1, Recipe.codes) } + gains.map { ($0.0, p + $0.1, Recipe.half) }
        case .shared:
            return products.map { weights.entry($0.0, key: p + $0.1) } + gains.map { ($0.0, p + $0.1, Recipe.half) }
                + QwenImage21Shared.targets.map { ($0.name, "", Recipe.target($0.products)) }
        case .bfloat: return products.map { ($0.0, p + $0.1 + ".weight", Recipe.bfloatTransposed) } + gains.map { ($0.0, p + $0.1, Recipe.bfloat16(joined: [])) }
        }
    }

    /// Everything else, in f32 on both paths.
    static func others() -> [Weights.Entry] {
        [("linear_1", "time_text_embed.timestep_embedder.linear_1.weight", .floatTransposed),
         ("linear_2", "time_text_embed.timestep_embedder.linear_2.weight", .floatTransposed),
         ("modulation", "modulation.1.weight", .floatTransposed), ("norm_out", "norm_out.linear.weight", .floatTransposed),
         ("txt_norm", "txt_in.text_norm.weight", .float), ("txt_in1", "txt_in.in_layer.weight", .floatTransposed),
         ("txt_in2", "txt_in.out_layer.weight", .floatTransposed), ("img_in", "img_in.weight", .floatTransposed),
         ("proj_out", "proj_out.weight", .floatTransposed)]
    }

    /// **Every tensor the DiT reads, by its published name and shape** (`[out, in]` for a product): what a file is
    /// recognized against.
    static func tensors() -> [String: [Int]] {
        let d = c.dim
        var all: [String: [Int]] = [
            "time_text_embed.timestep_embedder.linear_1.weight": [d, c.timeDim], "time_text_embed.timestep_embedder.linear_2.weight": [d, d],
            "modulation.1.weight": [4 * d, d], "norm_out.linear.weight": [d, d], "txt_in.text_norm.weight": [c.context],
            "txt_in.in_layer.weight": [d, c.context], "txt_in.out_layer.weight": [d, d], "img_in.weight": [d, c.channels],
            "proj_out.weight": [c.channels, d]]
        for (_, shape, name) in layerModules() { all[name + ".weight"] = shape }
        for i in 0..<c.layers {
            for (_, key) in gains { all["transformer_blocks.\(i).\(key)"] = [c.headDim] }
        }
        return all
    }

    /// Each layer's products, `[out, in]`, by module name.
    static func layerModules() -> [(Int, [Int], String)] {
        let (d, hidden) = (c.dim, c.hidden)
        let shapes: [String: [Int]] = ["to_q": [d, d], "to_k": [d, d], "to_v": [d, d], "to_out": [d, d], "gate_up": [2 * hidden, d], "down": [d, hidden]]
        return (0..<c.layers).flatMap { i in products.map { (i, shapes[$0.0]!, "transformer_blocks.\(i).\($0.1)") } }
    }

    /// **Every module a LoRA may adapt**: each product by its published name, and the two halves of the fused SwiGLU
    /// input under diffusers' names (`gate_layer`, `proj`), which a LoRA trained on diffusers' module adapts each.
    static func loraModules() -> [String: LoRAStack.Shape] {
        let (d, hidden) = (c.dim, c.hidden)
        var all: [String: LoRAStack.Shape] = [
            "time_text_embed.timestep_embedder.linear_1": .init(out: d, inn: c.timeDim),
            "time_text_embed.timestep_embedder.linear_2": .init(out: d, inn: d), "modulation.1": .init(out: 4 * d, inn: d),
            "txt_in.in_layer": .init(out: d, inn: c.context), "txt_in.out_layer": .init(out: d, inn: d), "img_in": .init(out: d, inn: c.channels),
            "norm_out.linear": .init(out: d, inn: d), "proj_out": .init(out: c.channels, inn: d)]
        for (i, shape, name) in layerModules() {
            all[name] = .init(out: shape[0], inn: shape[1])
            if name.hasSuffix("gate_up") {
                // [gate | up]: the gate's rows first (`split`'s order in the layer), each half landing on its own rows.
                for (k, half) in ["gate_layer", "proj"].enumerated() {
                    all["transformer_blocks.\(i).img_mlp.\(half)"] = .init(out: hidden, inn: d, product: name, row: k * hidden)
                }
            }
        }
        return all
    }

    // MARK: - The arguments

    /// The functions outside the layers: one form on every path but bf16's.
    static let outer: Set<String> = ["timestep", "embed-prefix", "embed-target", "final"]

    /// A function's arguments in the order it declares them. `rows`: a tile's, or the prefix's or the target's keys for
    /// the outer functions; `prefix`, `target`: the keys of each (a post's).
    static func arguments(_ f: String, path: Path, weights: SharedWeights = .codes, codes: CodesLayout, rows: Int, prefix: Int = 0,
                          target: Int = 0, ranks: Ranks = [:]) -> [Argument] {
        own(f, path: path, weights: weights, codes: codes, rows: rows, prefix: prefix, target: target) + factors(f, path: path, ranks: ranks)
    }

    private static func own(_ f: String, path: Path, weights: SharedWeights, codes: CodesLayout, rows: Int, prefix: Int, target: Int) -> [Argument] {
        if path == .shared, !outer.contains(f) { return QwenImage21Shared.arguments(f, rows: rows, prefix: prefix, target: target, weights: weights) }
        let (d, e, hd, h) = (Self.d, path.element, Self.hd, Self.heads)
        func stream(_ name: String) -> [Argument] {
            path == .bfloat ? [(name, [rows, d], .f32)] : [(name, [rows, d], .f16), (name + "_factor", [1, rows], path.factor)]
        }
        func weight(_ name: String, _ inn: Int, _ out: Int) -> [Argument] {
            path == .half ? [(name, codes == .inOut ? [inn, out] : [out, inn], .si8), (name + "_scale", [out], .f16)] : [(name, [inn, out], .bf16)]
        }
        switch f {
        case "timestep":
            return [("t_freq", [1, c.timeDim], .f32), ("linear_1", [c.timeDim, d], .f32), ("linear_2", [d, d], .f32),
                    ("modulation", [d, 4 * d], .f32), ("norm_out", [d, d], .f32)]
        case "embed-prefix":
            return [("text", [rows, c.context], .f32), ("references", [rows, c.channels], .f32), ("which", [1, rows], .f32),
                    ("txt_norm", [c.context], .f32), ("txt_in1", [c.context, d], .f32), ("txt_in2", [d, d], .f32), ("img_in", [c.channels, d], .f32)]
        case "embed-target":
            return [("tokens", [rows, c.channels], .f32), ("img_in", [c.channels, d], .f32)]
        case "final":
            return stream("x") + [("final_scale", [1, d], .f32), ("proj_out", [d, c.channels], .f32)]
        case "pre":
            return stream("x") + [("scale1", [1, d], path.gains)] + weight("to_q", d, d) + weight("to_k", d, d) + weight("to_v", d, d)
                + [("norm_q", [hd], e), ("norm_k", [hd], e), ("cos", [rows, 1, hd], .f32), ("sin", [rows, 1, hd], .f32)]
        default:
            return stream("x") + [("q", [1, h, rows, hd], e), ("kp", [1, prefix, h, hd], e), ("vp", [1, prefix, h, hd], e),
                                  ("kt", [1, target, h, hd], e), ("vt", [1, target, h, hd], e), ("mask", [rows, prefix + target], e),
                                  ("gate1", [1, d], path.gains), ("scale2", [1, d], path.gains), ("gate2", [1, d], path.gains)]
                + weight("to_out", d, d) + weight("gate_up", d, 2 * c.hidden) + weight("down", c.hidden, d)
        }
    }

    // MARK: - The modules

    /// `codes`: the int8 codes' layout on this Mac (`CodesLayout`), what the half path's products read.
    /// `weights`: the shared forms' (`SharedWeights`).
    /// `tapped`: `final` also gives `normed`, the head's norm, which only the check reads (a render left it in P,
    /// `[targetKeys, d]` in f32 written at every step).
    /// `neural`: the ANE's function of a step both units take (`Block.Build.neural`, `QwenImage21Shared.module`).
    static func module(_ f: String, path: Path, weights: SharedWeights = .codes, codes: CodesLayout, rows: Int, prefix: Int = 0, target: Int = 0,
                       ranks: Ranks = [:], tapped: Bool = false, neural: Bool = false) -> Aicode.Module {
        if path == .shared, !outer.contains(f) {
            return QwenImage21Shared.module(f, rows: rows, prefix: prefix, target: target, weights: weights, ranks: ranks, neural: neural)
        }
        let arguments = arguments(f, path: path, codes: codes, rows: rows, prefix: prefix, target: target, ranks: ranks)
        var m = Aicode.Module(arguments: arguments.map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        let b = m.body
        let names = m.argumentNames
        func a(_ name: String) -> Int { names.firstIndex(of: name)! }
        let (d, hd, h, eps, e) = (Self.d, Self.hd, Self.heads, c.eps, path.element)
        var constants = Aicode.ScalingConstants()
        /// `y`, the product `w`'s result on `x`, with its LoRA's change when one adapts it.
        func adapt(_ y: Int, _ x: Int, _ w: String) -> Int {
            (ranks[w] ?? 0) > 0 ? b.lowRank(y, x, down: a(w + "_down"), up: a(w + "_up")) : y
        }
        func f32(_ x: Int, _ w: String) -> Int {
            adapt(b.matmul(x, a(w), result: Aicode.tensor([Aicode.shape(b.types[x]).shape[0], Aicode.shape(b.types[a(w)]).shape[1]], .f32)), x, w)
        }
        /// A layer's product: W8A16 on the codes, or bf16. `unrotated`: the input before its rotation, under the same
        /// factor (the half path's).
        func product(_ x: Int, _ w: String, unrotated: Int? = nil) -> Int {
            switch path {
            case .half, .shared: return adapt(b.w8a16(x, codes: a(w), scale: a(w + "_scale"), layout: codes, constants: &constants), unrotated!, w)
            case .bfloat: return adapt(b.matmul(x, a(w), result: Aicode.tensor([Aicode.shape(b.types[x]).shape[0], Aicode.shape(b.types[a(w)]).shape[1]], .bf16)), x, w)
            }
        }
        /// A value before its rotation over the row factor `factor` `[rows, 1]` the rotated one was ranged by: a power of
        /// two, exact.
        func under(_ t: Int, _ factor: Int) -> Int { b.binary(.divide, t, b.castUnlessSame(factor, to: Aicode.shape(b.types[t]).element)) }
        func aim(_ ws: [String]) -> Int { b.target(codes: ws.map { (a($0), a($0 + "_scale")) }, layout: codes) }
        func column(_ f: Int) -> Int { b.reshape(f, [rows, 1]) }
        func line(_ f: Int) -> Int { b.reshape(f, [1, rows]) }
        func layerNorm(_ x: Int, epsilon: Int) -> Int { b.layerNorm(x, axes: [1], epsilon: epsilon) }
        /// The stream's true values in f32 (the outer functions' input).
        func trueStream() -> Int {
            path == .bfloat ? a("x") : b.binary(.mul, b.cast(a("x"), to: .f32), column(b.castUnlessSame(a("x_factor"), to: .f32)))
        }
        /// The stream out of an outer function, from its true values in f32. In the shared forms the factor is f16's,
        /// within [2⁻¹³, 2¹⁵] as `ranged16` keeps it: a row under 2 (a padding row, zero) then lies under 2¹³.
        func streamOut(_ x: Int) -> [(String, Int)] {
            switch path {
            case .bfloat: return [("x", x)]
            case .half:
                let (s, sf) = b.ranged(x, under: b.stream)
                return [("x", s), ("x_factor", line(sf))]
            case .shared:
                let sf = b.scale("minimum", b.scale("maximum", b.ranged(x, under: b.stream).factor, 0x1p-13), 0x1p15)
                return [("x", b.cast(b.binary(.divide, x, sf), to: .f16)), ("x_factor", line(b.cast(sf, to: .f16)))]
            }
        }
        // RoPE on adjacent pairs in f32, rounded once (`apply_rotary_emb_qwen(use_real=False)`): x · cos + swap(x) · sin,
        // the tables `[rows, 1, 128]` holding each pair's angle twice and the sine's sign (−sin, sin).
        func rope(_ t: Int) -> Int {
            let x = b.cast(t, to: .f32)
            let pairs = b.reshape(x, [rows, h, hd / 2, 2])
            let swapped = b.reshape(b.concat([b.slice(pairs, axis: 3, start: 1, size: 1), b.slice(pairs, axis: 3, start: 0, size: 1)], axis: 3),
                                    [rows, h, hd])
            return b.cast(b.binary(.add, b.binary(.mul, x, a("cos")), b.binary(.mul, swapped, a("sin"))), to: e)
        }
        func headsFirst(_ t: Int) -> Int { b.reshape(b.transpose(t, [1, 0, 2]), [1, h, rows, hd]) }
        // The keys and values token first, `[1, rows, 32, 128]`: a tile's a block of the pass's, written whole, their heads
        // taken first inside the attention (measured 19/11 on Z: the same bits, the write twice faster).
        func tokenFirst(_ t: Int) -> Int { b.reshape(t, [1, rows, h, hd]) }
        func split(_ t: Int) -> Int { b.reshape(t, [rows, h, hd]) }

        switch (f, path) {
        case ("timestep", _):
            let temb = f32(b.silu(f32(a("t_freq"), "linear_1")), "linear_2")
            let s = b.silu(temb)
            let m4 = b.reshape(f32(s, "modulation"), [4, d])
            func quarter(_ k: Int) -> Int { b.slice(m4, axis: 0, start: k, size: 1) }
            let one = b.scalar(1, .f32)
            let rows = b.concat([b.binary(.add, one, quarter(0)), b.tanh(quarter(1)), b.binary(.add, one, quarter(2)), b.tanh(quarter(3))], axis: 0)
            m.outputs = [("temb", temb), ("modulation", b.reshape(m4, [1, 4 * d])), ("mod", b.zeroPadded(b.castUnlessSame(rows, to: path.gains), to: [4, path.gainRow])),
                         ("final_scale", b.binary(.add, one, f32(s, "norm_out")))]
        case ("embed-prefix", _):
            // txt_in: the zero-centred RMSNorm in f32 (`x · rrms · (1 + w)`), Linear, GELU tanh, Linear.
            let n = b.binary(.mul, b.rmsNorm(a("text"), axes: [1], eps: eps), b.scale("add", a("txt_norm"), 1))
            let u = f32(n, "txt_in1")
            let inner = b.scale("mul", b.binary(.add, u, b.scale("mul", b.binary(.mul, b.binary(.mul, u, u), u), 0.044715)), (2 / Double.pi).squareRoot())
            let gelu = b.binary(.mul, b.scale("mul", u, 0.5), b.scale("add", b.tanh(inner), 1))
            let text = f32(gelu, "txt_in2"), image = f32(a("references"), "img_in")
            let which = b.reshape(a("which"), [rows, 1])
            let x = b.binary(.add, b.binary(.mul, which, text), b.binary(.mul, b.scale("sub", which, 1, scalarFirst: true), image))
            m.outputs = streamOut(x)
        case ("embed-target", _):
            m.outputs = streamOut(f32(a("tokens"), "img_in"))
        case ("final", _):
            let n = b.binary(.mul, b.layerNorm(trueStream(), axes: [1], epsilon: b.scalar(eps, .f32)), a("final_scale"))
            m.outputs = [("velocity", f32(n, "proj_out"))] + (tapped ? [("normed", n)] : [])

        case (_, .shared):
            preconditionFailure("\(f): the shared forms' (`QwenImage21Shared`)")
        case ("pre", .half):
            let n1 = b.binary(.mul, layerNorm(a("x"), epsilon: b.epsilon(column(a("x_factor")), eps)), a("scale1"))
            let (unit, f) = b.ranged(b.convrot(n1), under: aim(["to_q", "to_k", "to_v"]))
            let x1 = under(n1, f)
            let eh = b.reshape(b.epsilon(f, eps), [rows, 1, 1])
            func headNormed(_ t: Int, _ gain: String) -> Int { b.binary(.mul, b.rmsNorm(split(t), axes: [2], epsilon: eh), a(gain)) }
            // v multiplied out: the attention reads it across rows, where no row's factor can go.
            let v = b.binary(.mul, split(product(unit, "to_v", unrotated: x1)), b.reshape(b.cast(f, to: .f16), [rows, 1, 1]))
            m.outputs = [("q", headsFirst(rope(headNormed(product(unit, "to_q", unrotated: x1), "norm_q")))),
                         ("k", tokenFirst(rope(headNormed(product(unit, "to_k", unrotated: x1), "norm_k")))), ("v", tokenFirst(v))]
        case ("pre", .bfloat):
            let n1 = b.cast(b.binary(.mul, layerNorm(a("x"), epsilon: b.scalar(eps, .f32)), a("scale1")), to: .bf16)
            // `diffusers.RMSNorm` rounds twice: the normalized value to bf16, then × γ.
            func headNormed(_ t: Int, _ gain: String) -> Int { b.binary(.mul, b.rmsNorm(split(t), axes: [2], eps: eps), a(gain)) }
            m.outputs = [("q", headsFirst(rope(headNormed(product(n1, "to_q"), "norm_q")))),
                         ("k", tokenFirst(rope(headNormed(product(n1, "to_k"), "norm_k")))), ("v", tokenFirst(split(product(n1, "to_v"))))]

        case (_, .half):
            let o = attention(b, a, rows: rows)
            let (attended, f1) = b.ranged(b.convrot(o), under: aim(["to_out"]))
            let (x1, x1f) = b.residual(a("x"), column(a("x_factor")), plus: b.binary(.mul, a("gate1"), product(attended, "to_out", unrotated: under(o, f1))),
                                       factor: f1)
            let n3 = b.binary(.mul, layerNorm(x1, epsilon: b.epsilon(x1f, eps)), a("scale2"))
            let (r, f2) = b.ranged(b.convrot(n3), under: aim(["gate_up"]))
            let both = product(r, "gate_up", unrotated: under(n3, f2))
            let w1 = b.slice(both, axis: 1, start: 0, size: c.hidden), w3 = b.slice(both, axis: 1, start: c.hidden, size: c.hidden)
            // Under 2¹⁰ (`gatedSiLU`), its rotation under 2¹⁴; the LoRA's path reads it unrotated, under the same g₄.
            var (gated, g4) = (0, 0)
            let (hidden, factor) = b.gatedSiLU(w1, w3, factor: f2) { t in
                let r = b.ranged(b.convrot(t), under: aim(["down"]))
                (gated, g4) = (t, r.factor)
                return r
            }
            let (y, yf) = b.residual(x1, x1f, plus: b.binary(.mul, a("gate2"), product(hidden, "down", unrotated: under(gated, g4))), factor: factor)
            m.outputs = [("y", y), ("y_factor", line(yf))]
        case (_, .bfloat):
            let o = attention(b, a, rows: rows)
            let x1 = b.binary(.add, a("x"), b.binary(.mul, a("gate1"), b.cast(product(o, "to_out"), to: .f32)))
            let n3 = b.cast(b.binary(.mul, layerNorm(x1, epsilon: b.scalar(eps, .f32)), a("scale2")), to: .bf16)
            let both = product(n3, "gate_up")
            let gate = b.slice(both, axis: 1, start: 0, size: c.hidden), up = b.slice(both, axis: 1, start: c.hidden, size: c.hidden)
            let y = b.binary(.add, x1, b.binary(.mul, a("gate2"), b.cast(product(b.binary(.mul, b.silu(gate), up), "down"), to: .f32)))
            m.outputs = [("y", y)]
        }
        return m
    }

    /// A tile's attention over the prefix's keys and the target's (token first, their heads taken first here), rows
    /// first `[rows, d]`.
    private static func attention(_ b: Aicode.Block, _ a: (String) -> Int, rows: Int) -> Int {
        func headsFirst(_ t: Int) -> Int { b.transpose(t, [0, 2, 1, 3]) }
        let keys = headsFirst(b.concat([a("kp"), a("kt")], axis: 1)), values = headsFirst(b.concat([a("vp"), a("vt")], axis: 1))
        let o = b.attention([a("q"), keys, values, a("mask")], scale: 1 / Double(hd).squareRoot(), symbol: "sdpa_tile")
        return b.reshape(b.transpose(b.reshape(o, [heads, rows, hd]), [1, 0, 2]), [rows, d])
    }
}
