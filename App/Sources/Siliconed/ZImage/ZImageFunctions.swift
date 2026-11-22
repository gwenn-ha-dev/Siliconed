import Foundation
import SiliCore

/// **Z-Image's DiT in Core AI functions of fixed shapes**, written op by op (`Aicode`), the weights arguments. Two
/// forms of the same functions, one a render:
///
///     bf16     the Standard, bf16 end to end, the GPU alone: diffusers' arithmetic, each norm's statistics in f32
///     shared   the Compact, on the GPU alone (the Neural Engine off) or shared with the ANE (on): int8 × int8 on the GPU, q64
///              on the ANE's copy of the codes (`SharedWeights`), on the int8 `convrot` codes (or the
///              Standard's weights in f16, `SharedWeights.half`) under the guard of f16 (`Guard.swift`), the forms both units take (`Neural.swift`), the residual
///              stream and its products' inputs as (t, F), F f16 `[1, rows, 1]`
///
/// The functions besides a block's (`ZImageFunctions`), the block's (`ZImageBlock`, `ZImageShared`):
///
///     timestep        one row, f32   adaln = mlp₂(SiLU(mlp₀ t_freq)), final_scale = 1 + adaLN_final(SiLU(adaln))
///     embed-image     the latent     patchify · x_embedder + b, the pad token past the image's rows: the stream
///     embed-caption   the palier     RMS · γ · cap_embedder + b, the pad token past the caption: the stream
///     final           the image      LayerNorm (no affine, eps 10⁻⁶) · final_scale · linear + b, unpatchify
///
/// One-row functions compute in f32 (their cost is nothing; a unit without f32 leaves them to the GPU). In the shared
/// forms everything a call of rows reads is f16 under the guard; the biases are added in f32, where a row factor cannot pass.
///
/// **A LoRA beside each product** (`LowRank`, `ranks` by product): the timestep's factors f32, the others' the form's
/// element; each product's input as the product reads it (no rotation here: under the guard, the unit under its factor).
enum ZImageFunctions {
    enum Form: Hashable, Sendable {
        case bf16, shared

        var element: Aicode.Element { self == .bf16 ? .bf16 : .f16 }
        init(_ a: Arithmetic) { self = a.shared ? .shared : .bf16 }
    }

    typealias Argument = (name: String, shape: [Int], element: Aicode.Element)
    static let c = ZImageDiT.Config.published
    static var d: Int { c.dim }
    static var patch: Int { c.patch }
    static var channels: Int { c.channels }
    static var adalnDim: Int { c.adalnDim }
    static var capDim: Int { c.capDim }
    static var eps: Double { Double(c.eps) }

    // MARK: - The functions besides the block's

    /// The products a LoRA adapts, by the function that computes each (`ZImageDiT.edges`' names).
    static let adapted: [String: [String]] = ["timestep": ["mlp0", "mlp2", "final_ada"], "embed-image": ["x_embedder"],
                                              "embed-caption": ["cap_embedder"], "final": ["final_linear"]]

    /// The element of a function's low-rank factors: f32 in the timestep, which computes in f32; the form's otherwise.
    static func lowRankElement(_ function: String, _ form: Form) -> Aicode.Element { function == "timestep" ? .f32 : form.element }

    static func lowRankArguments(_ function: String, form: Form, ranks: [String: Int]) -> [Argument] {
        let shapes = Dictionary(uniqueKeysWithValues: ZImageDiT.edges().map { ($0.name, (inn: $0.inn, out: $0.out)) })
        let e = lowRankElement(function, form)
        return (adapted[function] ?? []).flatMap { p -> [Argument] in
            guard let r = ranks[p], r > 0, let s = shapes[p] else { return [] }
            return [(p + "_down", [s.inn, r], e), (p + "_up", [r, s.out], e)]
        }
    }

    /// `rows`: embed-image and final, the image's rows padded to 32 (the image's own rows are `h/2 · w/2`); embed-caption,
    /// its palier. Under `shared`, the stream's factors as the shared block holds them, f16 `[1, rows, 1]`; none under
    /// `bf16`.
    static func arguments(_ function: String, height h: Int, width w: Int, rows: Int, form: Form, ranks: [String: Int] = [:]) -> [Argument] {
        plainArguments(function, height: h, width: w, rows: rows, form: form) + lowRankArguments(function, form: form, ranks: ranks)
    }

    private static func plainArguments(_ function: String, height h: Int, width w: Int, rows: Int, form: Form) -> [Argument] {
        let p = patch * patch * channels
        if form == .bf16 {
            switch function {
            case "timestep": return plainArguments(function, height: h, width: w, rows: rows, form: .shared)
            case "embed-image":
                return [("latent", [channels, h, w], .f32), ("x_embedder", [p, d], .bf16), ("x_embedder_bias", [d], .bf16), ("x_pad_token", [1, d], .bf16)]
            case "embed-caption":
                return [("caps", [rows, capDim], .f32), ("real", [1, rows], .f32), ("cap_norm", [capDim], .bf16),
                        ("cap_embedder", [capDim, d], .bf16), ("cap_embedder_bias", [d], .bf16), ("cap_pad_token", [1, d], .bf16)]
            default:
                return [("x", [rows, d], .bf16), ("final_scale", [1, d], .f32), ("final_linear", [d, p], .bf16), ("final_linear_bias", [p], .f32)]
            }
        }
        switch function {
        case "timestep":
            return [("t_freq", [1, adalnDim], .f32), ("mlp0", [adalnDim, 1024], .f32), ("mlp0_bias", [1024], .f32),
                    ("mlp2", [1024, adalnDim], .f32), ("mlp2_bias", [adalnDim], .f32),
                    ("final_ada", [adalnDim, d], .f32), ("final_ada_bias", [d], .f32)]
        case "embed-image":
            return [("latent", [channels, h, w], .f32), ("x_embedder", [p, d], .f16), ("x_embedder_bias", [d], .f32), ("x_pad_token", [1, d], .f32)]
        case "embed-caption":
            return [("caps", [rows, capDim], .f32), ("real", [1, rows], .f32), ("cap_norm", [capDim], .f32),
                    ("cap_embedder", [capDim, d], .f16), ("cap_embedder_bias", [d], .f32), ("cap_pad_token", [1, d], .f32)]
        default:
            return [("x", [rows, d], .f16), ("x_factor", [1, rows, 1], .f16),
                    ("final_scale", [1, d], .f32), ("final_linear", [d, p], .f16), ("final_linear_bias", [p], .f32)]
        }
    }

    static func module(_ function: String, height h: Int, width w: Int, rows: Int, form: Form, ranks: [String: Int] = [:]) -> Aicode.Module {
        var m = Aicode.Module(arguments: arguments(function, height: h, width: w, rows: rows, form: form, ranks: ranks).map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        let b = m.body
        let names = m.argumentNames
        func a(_ name: String) -> Int { names.firstIndex(of: name)! }
        // The stream's factor out, and in: a power of two within f16's normals (X in [2⁻¹³, 2¹⁵] under the shared block).
        func factorOut(_ f: Int) -> Int { b.cast(b.reshape(f, [1, rows, 1]), to: .f16) }
        func factorIn() -> Int { b.cast(b.reshape(a("x_factor"), [rows, 1]), to: .f32) }
        /// `y`, the product of `x` and W, with its low-rank path beside it where a rank adapts W.
        func adapted(_ y: Int, _ x: Int, _ w: String) -> Int {
            (ranks[w] ?? 0) > 0 ? b.lowRank(y, x, down: a(w + "_down"), up: a(w + "_up")) : y
        }
        func f32(_ x: Int, _ w: String, bias: String) -> Int {
            let y = b.matmul(x, a(w), result: Aicode.tensor([Aicode.shape(b.types[x]).shape[0], Aicode.shape(b.types[a(w)]).shape[1]], .f32))
            return b.binary(.add, adapted(y, x, w), a(bias))
        }
        /// `x · W + b` in bf16, `[rows, out]`.
        func bf16(_ x: Int, _ w: String, bias: String) -> Int {
            let out = Aicode.shape(b.types[a(w)]).shape[1]
            return b.binary(.add, adapted(b.matmul(x, a(w), result: Aicode.tensor([Aicode.shape(b.types[x]).shape[0], out], .bf16)), x, w), a(bias))
        }
        let (hp, wp, tokens) = (h / patch, w / patch, (h / patch) * (w / patch))
        // "c (h ph) (w pw) → (h w) (ph pw c)"
        func patches() -> Int {
            b.reshape(b.transpose(b.reshape(a("latent"), [channels, hp, patch, wp, patch]), [1, 3, 2, 4, 0]), [tokens, patch * patch * channels])
        }
        // "(h w) (ph pw c) → c (h ph) (w pw)", the padding dropped first.
        func unpatchified(_ y: Int) -> Int {
            let image = b.slice(y, axis: 0, start: 0, size: tokens)
            return b.reshape(b.transpose(b.reshape(image, [hp, wp, patch, patch, channels]), [4, 0, 2, 1, 3]), [channels, h, w])
        }
        switch (function, form) {
        case ("timestep", _):
            let adaln = f32(b.silu(f32(a("t_freq"), "mlp0", bias: "mlp0_bias")), "mlp2", bias: "mlp2_bias")
            m.outputs = [("adaln", adaln), ("final_scale", b.binary(.add, b.scalar(1, .f32), f32(b.silu(adaln), "final_ada", bias: "final_ada_bias")))]
        case ("embed-image", .bf16):
            var x = bf16(b.cast(patches(), to: .bf16), "x_embedder", bias: "x_embedder_bias")
            if rows > tokens { x = b.concat([x, b.repeated(a("x_pad_token"), rows - tokens)], axis: 0) }
            m.outputs = [("x", x)]
        case ("embed-image", _):
            let (unit, f) = b.ranged(patches(), under: b.target(weights: [a("x_embedder")]))
            var x = b.multipliedOut(adapted(b.product(unit, a("x_embedder")), unit, "x_embedder"), f, bias: a("x_embedder_bias"))
            if rows > tokens { x = b.concat([x, b.repeated(a("x_pad_token"), rows - tokens)], axis: 0) }
            let (s, sf) = b.ranged(x, under: b.stream)
            m.outputs = [("x", s), ("x_factor", factorOut(sf))]
        case ("embed-caption", .bf16):
            let n = b.binary(.mul, b.rmsNorm(b.cast(a("caps"), to: .bf16), axes: [1], eps: eps), a("cap_norm"))
            let x = bf16(n, "cap_embedder", bias: "cap_embedder_bias")
            let real = b.cast(b.reshape(a("real"), [rows, 1]), to: .bf16)
            let pad = b.binary(.mul, b.scale("sub", real, 1, scalarFirst: true), a("cap_pad_token"))
            m.outputs = [("x", b.binary(.add, b.binary(.mul, x, real), pad))]
        case ("embed-caption", _):
            let n = b.binary(.mul, b.rmsNorm(a("caps"), axes: [1], eps: eps), a("cap_norm"))
            let (unit, f) = b.ranged(n, under: b.target(weights: [a("cap_embedder")]))
            let x = b.multipliedOut(adapted(b.product(unit, a("cap_embedder")), unit, "cap_embedder"), f, bias: a("cap_embedder_bias"))
            // The caption's rows, the pad token past them: real · x + (1 − real) · pad, `real` a row of 1 and 0 (a
            // column `[rows, 1]` Core AI asks with 64 bytes a row, and copies).
            let real = b.reshape(a("real"), [rows, 1])
            let pad = b.binary(.mul, b.scale("sub", real, 1, scalarFirst: true), a("cap_pad_token"))
            let (s, sf) = b.ranged(b.binary(.add, b.binary(.mul, x, real), pad), under: b.stream)
            m.outputs = [("x", s), ("x_factor", factorOut(sf))]
        case (_, .bf16):
            // The norm and its scale in f32, rounded once for the product (diffusers rounds the norm, then the scale).
            let n = b.binary(.mul, b.layerNorm(b.cast(a("x"), to: .f32), axes: [1], epsilon: b.scalar(1e-6, .f32)), a("final_scale"))
            let nb = b.cast(n, to: .bf16)
            let y = adapted(b.matmul(nb, a("final_linear"), result: Aicode.tensor([rows, patch * patch * channels], .bf16)), nb, "final_linear")
            m.outputs = [("velocity", unpatchified(b.binary(.add, b.cast(y, to: .f32), a("final_linear_bias"))))]
        default:
            let e = b.epsilon(factorIn(), 1e-6)
            let n = b.binary(.mul, b.layerNorm(b.cast(a("x"), to: .f32), axes: [1], epsilon: e), a("final_scale"))
            let (unit, f) = b.ranged(n, under: b.target(weights: [a("final_linear")]))
            m.outputs = [("velocity", unpatchified(b.multipliedOut(adapted(b.product(unit, a("final_linear")), unit, "final_linear"), f, bias: a("final_linear_bias"))))]
        }
        return m
    }

    // MARK: - The weights

    /// A block's: (argument, key, recipe). `shared`: each product's codes or f16 weight
    /// (`weights`), the targets the shared forms read (`ZImageShared.targets`) after the products they come from.
    static func entries(_ block: String, form: Form, weights: SharedWeights = .codes) -> [Weights.Entry] {
        let p = block + "."
        let products = ZImageDiT.products(block), vectors = ZImageDiT.vectors(block)
        if form == .bf16 {
            return products.map { ($0.name, p + $0.key + ".weight", Recipe.bfloatTransposed) }
                + vectors.map { ($0.name, p + $0.key, $0.name == "ada_bias" ? Recipe.float : .bfloat16(joined: [])) }
        }
        return products.map { weights.entry($0.name, key: p + $0.key) } + vectors.map { ($0.name, p + $0.key, Recipe.half) }
            + ZImageShared.targets.map { ($0.name, "", Recipe.target($0.products)) }
    }

    /// Everything else, by the functions' names for it.
    static func others(form: Form) -> [Weights.Entry] {
        let float: [Weights.Entry] = [
            ("mlp0", "t_embedder.mlp.0.weight", .floatTransposed), ("mlp0_bias", "t_embedder.mlp.0.bias", .float),
            ("mlp2", "t_embedder.mlp.2.weight", .floatTransposed), ("mlp2_bias", "t_embedder.mlp.2.bias", .float),
            ("final_ada", "final_layer.adaLN_modulation.1.weight", .floatTransposed), ("final_ada_bias", "final_layer.adaLN_modulation.1.bias", .float),
            ("final_linear_bias", "final_layer.linear.bias", .float)]
        if form == .bf16 {
            return float + [
                ("x_embedder", "x_embedder.weight", .bfloatTransposed), ("x_embedder_bias", "x_embedder.bias", .bfloat16(joined: [])), ("x_pad_token", "x_pad_token", .bfloat16(joined: [])),
                ("cap_norm", "cap_embedder.0.weight", .bfloat16(joined: [])), ("cap_embedder", "cap_embedder.1.weight", .bfloatTransposed),
                ("cap_embedder_bias", "cap_embedder.1.bias", .bfloat16(joined: [])), ("cap_pad_token", "cap_pad_token", .bfloat16(joined: [])),
                ("final_linear", "final_layer.linear.weight", .bfloatTransposed)]
        }
        return float + [
            ("x_embedder", "x_embedder.weight", .halfTransposed), ("x_embedder_bias", "x_embedder.bias", .float), ("x_pad_token", "x_pad_token", .float),
            ("cap_norm", "cap_embedder.0.weight", .float), ("cap_embedder", "cap_embedder.1.weight", .halfTransposed),
            ("cap_embedder_bias", "cap_embedder.1.bias", .float), ("cap_pad_token", "cap_pad_token", .float),
            ("final_linear", "final_layer.linear.weight", .halfTransposed)]
    }
}
