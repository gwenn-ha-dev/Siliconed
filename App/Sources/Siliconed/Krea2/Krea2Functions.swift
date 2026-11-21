import Foundation
import SiliCore

/// **Krea 2's functions** (`Krea2Transformer2DModel`, read in `transformer_krea2.py`), each a Core AI function of fixed
/// shapes, its weights arguments read from the arena; the loop that calls them is `Krea2Session`, the text fusion's
/// `Krea2Fusion`.
///
///     timestep    one row     temb = tmlp₂(GELU(tmlp₀ t_freq)), temb_mod = tproj(GELU(temb)) `[6, d]`
///     embed       the image   first · patches + b: the stream's image rows
///     pre         a call      h = (1 + prescale) · RMS₁(x) + preshift;  q, k, v = h · W;  q, k: RoPE(RMS_head)
///     post        a call      x₁ = x + pregate · Wo(SDPA(q, K, V) · σ(Wg h));  y = x₁ + postgate · down(SiLU(gate h′) · up h′)
///     final       the image   (1 + scale) · RMS(x) + shift, last · + b: the velocity's patches
///     layerwise   32 tokens   the fusion's first stage: a block whose sequence is one token's 12 taps (batch 32, length 12)
///     projector   32 tokens   the 12 taps summed by their weights, a feature at a time
///     refiner     the palier  the fusion's second stage: a block over the tokens, the keys past them masked
///     txtmlp      the palier  RMS · Linear · GELU · Linear: the stream's text rows
///
/// **bf16 end to end on the GPU** (the Standard): every value rounded where the reference rounds it — a norm once from f32
/// with its `1 + w`, the RoPE once from f32. The Compact's block is `Krea2Shared`'s; its timestep, text fusion, embedding
/// and final layer are these, in bf16: the Compact's file keeps their weights in bf16, as the reference (`witness-w8a8`)
/// runs them; once a step or once a prompt, their cost is nothing.
///
/// **A LoRA beside a product** (`ranks`, by argument: `LoRA.swift`): `<name>_down` `[in, R]` and `<name>_up` `[R, out]` in
/// bf16, `y + (x · down) · up` on x as the product reads it (`Aicode.Block.lowRank`).
///
/// What a port by analogy would miss:
///   - **GQA 48/12 without a repetition**: the four query heads of a key head lie next to each other (`repeat_interleave`),
///     so q `[1, 48, r, 128]` is `[1, 12, 4r, 128]` as it lies; the attention reads the 12 key heads as they are.
///   - **The gate of the attention is a sigmoid on its output**, before `wo`; the modulation is shared — `temb_mod` once an
///     evaluation, each block adding its own table `mod.lin` `[6 · d]` (bf16 + bf16, as the reference sums them).
package enum Krea2Functions {
    package static let d = 6144, heads = 48, kvHeads = 12, hd = 128, hidden = 16384, timeDim = 256, patch = 64, eps = 1e-5
    package static let textWidth = 2560, textHeads = 20, textHidden = 6912, taps = 12
    /// A tile of the fusion's first stage, in tokens.
    static let tokens = 32
    package typealias Argument = (name: String, shape: [Int], element: Aicode.Element)

    // MARK: - The arguments

    /// The arguments of `function` in the order it declares them. `rows`: a tile's, the image's, the text's palier; `keys`:
    /// the sequence's length the attention reads; `ranks`: the LoRAs' by product (none: the function as it is).
    package static func arguments(_ function: String, rows: Int, keys: Int, ranks: [String: Int] = [:]) -> [Argument] {
        func product(_ name: String, _ inn: Int, _ out: Int) -> [Argument] {
            [(name, [out, inn], .bf16)] + (ranks[name].map { [(name + "_down", [inn, $0], .bf16), (name + "_up", [$0, out], .bf16)] } ?? [])
        }
        let x: [Argument] = [("x", [rows, d], .bf16)]
        let modulation: [Argument] = [("temb_mod", [6, d], .bf16), ("mod", [6, d], .bf16), ("prenorm", [d], .f32)]
        switch function {
        case "timestep":
            return [("t_freq", [1, timeDim], .f32), ("tmlp0", [d, timeDim], .bf16), ("tmlp0_bias", [d], .bf16),
                    ("tmlp2", [d, d], .bf16), ("tmlp2_bias", [d], .bf16), ("tproj", [6 * d, d], .bf16), ("tproj_bias", [6 * d], .bf16)]
        case "embed":
            return [("patches", [rows, patch], .f32), ("first", [d, patch], .bf16), ("first_bias", [d], .bf16)]
        case "final":
            return x + [("temb", [1, d], .bf16), ("last_mod", [2, d], .bf16), ("last_norm", [d], .f32),
                        ("last_linear", [patch, d], .bf16), ("last_linear_bias", [patch], .bf16)]
        case "pre":
            return x + modulation + product("wq", d, d) + product("wk", d, kvHeads * hd) + product("wv", d, kvHeads * hd)
                + [("qnorm", [hd], .f32), ("knorm", [hd], .f32), ("cos", [rows, 1, hd], .f32), ("sin", [rows, 1, hd], .f32)]
        case "post", "post-taps":
            return x + [("q", [1, heads, rows, hd], .bf16), ("k", [1, keys, kvHeads, hd], .bf16), ("v", [1, keys, kvHeads, hd], .bf16),
                        ("mask", [1, keys], .bf16)] + modulation + product("gate", d, d) + product("wo", d, d)
                + [("postnorm", [d], .f32)] + product("up", d, hidden) + product("mgate", d, hidden) + product("down", hidden, d)
        case "layerwise", "refiner":
            let w = textWidth
            let block: [Argument] = [("x", [rows, w], .bf16), ("prenorm", [w], .f32)] + product("wq", w, w) + product("wk", w, w)
                + product("wv", w, w) + [("qnorm", [hd], .f32), ("knorm", [hd], .f32)] + product("gate", w, w) + product("wo", w, w)
                + [("postnorm", [w], .f32)] + product("up", w, textHidden) + product("mgate", w, textHidden) + product("down", textHidden, w)
            return function == "refiner" ? block + [("mask", [1, rows], .bf16)] : block
        case "projector":
            return [("x", [taps * rows, textWidth], .bf16), ("projector", [taps], .bf16)]
        case "txtmlp":
            return [("x", [rows, textWidth], .bf16), ("txt_norm", [textWidth], .f32), ("txt1", [d, textWidth], .bf16), ("txt1_bias", [d], .bf16),
                    ("txt3", [d, d], .bf16), ("txt3_bias", [d], .bf16)]
        default: preconditionFailure("krea2: no function \(function)")
        }
    }

    // MARK: - The functions

    /// `function` written op by op. `layerwise`'s rows are its tokens' taps (`taps` · `tokens`), `projector`'s its tokens.
    package static func module(_ function: String, rows: Int, keys: Int, ranks: [String: Int] = [:]) -> Aicode.Module {
        let arguments = arguments(function, rows: rows, keys: keys, ranks: ranks)
        var m = Aicode.Module(arguments: arguments.map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        let b = m.body
        let names = m.argumentNames
        func a(_ name: String) -> Int { names.firstIndex(of: name)! }
        func shape(_ v: Int) -> [Int] { Aicode.shape(b.types[v]).shape }

        /// `x · Wᵀ`, bf16, x `[rows, in]`, W `[out, in]` as the file lays it (`Recipe.bfloat16`), its LoRA beside.
        func mm(_ x: Int, _ w: String) -> Int {
            let y = b.matmul(x, b.transpose(a(w), [1, 0]), result: Aicode.tensor([shape(x)[0], shape(a(w))[0]], .bf16))
            return ranks[w] == nil ? y : b.lowRank(y, x, down: a(w + "_down"), up: a(w + "_up"))
        }
        func linear(_ x: Int, _ w: String) -> Int { b.binary(.add, mm(x, w), a(w + "_bias")) }
        /// `Krea2RMSNorm`: f32 statistics, `1 + w` in f32, the result f32 (its caller rounds it once).
        func normed(_ x: Int, _ gain: String, axis: Int) -> Int {
            b.binary(.mul, b.rmsNorm(b.castUnlessSame(x, to: .f32), axes: [axis], eps: eps), b.scale("add", a(gain), 1))
        }
        func bf16(_ x: Int) -> Int { b.castUnlessSame(x, to: .bf16) }
        // The block's six rows, `temb_mod + mod.lin` in bf16 (the reference's sum): prescale, preshift, pregate, postscale,
        // postshift, postgate.
        let parts = names.contains("mod") ? b.binary(.add, a("temb_mod"), a("mod")) : -1
        func part(_ i: Int) -> Int { b.slice(parts, axis: 0, start: i, size: 1) }
        /// `(1 + scale) · n + shift`, in bf16 from n rounded once (the reference).
        func modulated(_ n: Int, _ scale: Int, _ shift: Int) -> Int {
            b.binary(.add, b.binary(.mul, b.scale("add", part(scale), 1), bf16(n)), part(shift))
        }
        /// RoPE on adjacent pairs in f32, rounded once (`apply_rotary_emb`): x · cos + swap(x) · sin, the tables `[rows, 1,
        /// 128]` holding each pair's angle twice and the sine's sign, (−sin, sin).
        func rope(_ t: Int, _ count: Int) -> Int {
            let f = b.castUnlessSame(t, to: .f32), r = shape(t)[0]
            let pairs = b.reshape(f, [r, count, hd / 2, 2])
            let swapped = b.reshape(b.concat([b.slice(pairs, axis: 3, start: 1, size: 1), b.slice(pairs, axis: 3, start: 0, size: 1)], axis: 3), [r, count, hd])
            return b.cast(b.binary(.add, b.binary(.mul, f, a("cos")), b.binary(.mul, swapped, a("sin"))), to: .bf16)
        }
        func headsFirst(_ t: Int) -> Int { let s = shape(t); return b.reshape(b.transpose(t, [1, 0, 2]), [1, s[1], s[0], s[2]]) }
        // The keys and values token first, `[1, rows, 12, 128]`: a tile's a block of the sequence's, written whole, their
        // heads taken first inside the attention (measured 19/11 on Z: the same bits, the write twice faster).
        func tokenFirst(_ t: Int) -> Int { b.reshape(t, [1] + shape(t)) }
        /// The attention of a tile's q against every key, the four query heads of a key head as one of 4r rows.
        func attended(_ r: Int) -> Int {
            let q = b.reshape(a("q"), [1, kvHeads, 4 * r, hd])
            let o = b.attention([q, b.transpose(a("k"), [0, 2, 1, 3]), b.transpose(a("v"), [0, 2, 1, 3]), a("mask")], scale: 1 / Double(hd).squareRoot(), symbol: "sdpa_tile")
            return b.reshape(b.transpose(b.reshape(o, [heads, r, hd]), [1, 0, 2]), [r, d])
        }

        switch function {
        case "timestep":
            let temb = linear(b.gelu(linear(bf16(a("t_freq")), "tmlp0"), tanh: true), "tmlp2")
            m.outputs = [("temb", temb), ("temb_mod", b.reshape(linear(b.gelu(temb, tanh: true), "tproj"), [6, d]))]

        case "embed":
            m.outputs = [("x", linear(bf16(a("patches")), "first"))]

        case "final":
            let mod = b.binary(.add, a("last_mod"), a("temb"))
            let scale = b.slice(mod, axis: 0, start: 0, size: 1), shift = b.slice(mod, axis: 0, start: 1, size: 1)
            let n = bf16(normed(a("x"), "last_norm", axis: 1))
            let y = linear(b.binary(.add, b.binary(.mul, b.scale("add", scale, 1), n), shift), "last_linear")
            m.outputs = [("velocity", b.cast(y, to: .f32))]

        case "pre":
            let r = rows
            let h = modulated(normed(a("x"), "prenorm", axis: 1), 0, 1)
            let q = b.reshape(mm(h, "wq"), [r, heads, hd]), k = b.reshape(mm(h, "wk"), [r, kvHeads, hd]), v = b.reshape(mm(h, "wv"), [r, kvHeads, hd])
            m.outputs = [("q", headsFirst(rope(bf16(normed(q, "qnorm", axis: 2)), heads))),
                         ("k", tokenFirst(rope(bf16(normed(k, "knorm", axis: 2)), kvHeads))), ("v", tokenFirst(v))]

        case "post", "post-taps":
            let o = attended(rows)
            let h = modulated(normed(a("x"), "prenorm", axis: 1), 0, 1)
            let attention = mm(b.binary(.mul, o, b.sigmoid(mm(h, "gate"))), "wo")
            let x1 = b.binary(.add, a("x"), b.binary(.mul, part(2), attention))
            let f = modulated(normed(x1, "postnorm", axis: 1), 3, 4)
            let ff = mm(b.binary(.mul, b.silu(mm(f, "mgate")), mm(f, "up")), "down")
            m.outputs = [("y", b.binary(.add, x1, b.binary(.mul, part(5), ff)))]
            if function == "post-taps" { m.outputs += [("attention", b.cast(attention, to: .f32)), ("feed", b.cast(ff, to: .f32))] }

        case "layerwise", "refiner":
            // A block of the fusion, no modulation, no RoPE: `layerwise` over one token's 12 taps (batch `rows / 12`,
            // length 12), `refiner` over the tokens (batch 1), the keys past them masked.
            let n = rows, (batch, length) = function == "layerwise" ? (rows / taps, taps) : (1, rows)
            let h = bf16(normed(a("x"), "prenorm", axis: 1))
            func split(_ t: Int) -> Int { b.reshape(t, [n, textHeads, hd]) }
            func layout(_ t: Int) -> Int { b.transpose(b.reshape(t, [batch, length, textHeads, hd]), [0, 2, 1, 3]) }
            let q = layout(bf16(normed(split(mm(h, "wq")), "qnorm", axis: 2))), k = layout(bf16(normed(split(mm(h, "wk")), "knorm", axis: 2)))
            let v = layout(mm(h, "wv"))
            let o = b.attention([q, k, v] + (function == "refiner" ? [a("mask")] : []), scale: 1 / Double(hd).squareRoot(), symbol: "sdpa_text")
            let rowsFirst = b.reshape(b.transpose(o, [0, 2, 1, 3]), [n, textWidth])
            let x1 = b.binary(.add, a("x"), mm(b.binary(.mul, rowsFirst, b.sigmoid(mm(h, "gate"))), "wo"))
            let f = bf16(normed(x1, "postnorm", axis: 1))
            m.outputs = [("y", b.binary(.add, x1, mm(b.binary(.mul, b.silu(mm(f, "mgate")), mm(f, "up")), "down")))]

        case "projector":
            // `permute(0, 1, 3, 2)` then `Linear(12, 1)`: each feature a sum over the taps, in f32, rounded once.
            let x = b.cast(b.reshape(a("x"), [rows, taps, textWidth]), to: .f32), w = b.reshape(b.cast(a("projector"), to: .f32), [1, taps, 1])
            m.outputs = [("y", bf16(b.reshape(b.sum(b.binary(.mul, x, w), axes: [1]), [rows, textWidth])))]

        case "txtmlp":
            let n = bf16(normed(a("x"), "txt_norm", axis: 1))
            m.outputs = [("y", linear(b.gelu(linear(n, "txt1"), tanh: true), "txt3"))]

        default: preconditionFailure("krea2: no function \(function)")
        }
        return m
    }

    // MARK: - The weights

    /// A transformer block's: (argument, key, recipe), `blocks.<i>.` its prefix; the products bf16 `[out, in]`, the norms
    /// in f32 (`_keep_in_fp32_modules`), the table in bf16.
    static func block(_ prefix: String) -> [Weights.Entry] {
        products.map { ($0.0, prefix + $0.1 + ".weight", Recipe.bfloat16(joined: [])) } + norms(prefix) + [("mod", prefix + "mod.lin", .bfloat16(joined: []))]
    }

    /// An asset's name for the LoRAs' ranks of its function: none, nothing.
    package static func named(_ ranks: [String: Int]) -> String {
        ranks.isEmpty ? "" : "-lora" + ranks.sorted { $0.key < $1.key }.map { "-\($0.key)\($0.value)" }.joined()
    }

    /// Each product's rank over the blocks `prefixes` (the same argument in every block a function is called on): the
    /// largest landing on one of them (`LoRAStack.rank`), none where nothing lands.
    static func ranks(_ loras: LoRAStack, blocks prefixes: [String]) -> [String: Int] {
        var ranks: [String: Int] = [:]
        for (name, module) in products {
            let r = loras.rank(of: prefixes.map { $0 + module }, on: .krea2)
            if r > 0 { ranks[name] = r }
        }
        return ranks
    }

    /// A block's products: (argument, module), the module as the file and a LoRA name it.
    static let products = [("wq", "attn.wq"), ("wk", "attn.wk"), ("wv", "attn.wv"), ("gate", "attn.gate"), ("wo", "attn.wo"),
                           ("up", "mlp.up"), ("mgate", "mlp.gate"), ("down", "mlp.down")]

    /// A block of the text fusion, `txtfusion.layerwise_blocks.<i>.` or `txtfusion.refiner_blocks.<i>.`: bf16 in both files.
    static func fusionBlock(_ prefix: String) -> [Weights.Entry] {
        ["wq", "wk", "wv", "gate", "wo"].map { ($0, prefix + "attn.\($0).weight", Recipe.bfloat16(joined: [])) }
            + [("up", prefix + "mlp.up.weight", .bfloat16(joined: [])), ("mgate", prefix + "mlp.gate.weight", .bfloat16(joined: [])),
               ("down", prefix + "mlp.down.weight", .bfloat16(joined: []))] + norms(prefix)
    }

    static func norms(_ prefix: String) -> [Weights.Entry] {
        [("prenorm", prefix + "prenorm.scale", .float), ("postnorm", prefix + "postnorm.scale", .float),
         ("qnorm", prefix + "attn.qknorm.qnorm.scale", .float), ("knorm", prefix + "attn.qknorm.knorm.scale", .float)]
    }

    /// The DiT's other weights, read once an evaluation's arena: the timestep's, the embedding's, the final layer's.
    static let others: [Weights.Entry] = [
        ("tmlp0", "tmlp.0.weight", .bfloat16(joined: [])), ("tmlp0_bias", "tmlp.0.bias", .bfloat16(joined: [])),
        ("tmlp2", "tmlp.2.weight", .bfloat16(joined: [])), ("tmlp2_bias", "tmlp.2.bias", .bfloat16(joined: [])),
        ("tproj", "tproj.1.weight", .bfloat16(joined: [])), ("tproj_bias", "tproj.1.bias", .bfloat16(joined: [])),
        ("first", "first.weight", .bfloat16(joined: [])), ("first_bias", "first.bias", .bfloat16(joined: [])),
        ("last_mod", "last.modulation.lin", .bfloat16(joined: [])), ("last_norm", "last.norm.scale", .float),
        ("last_linear", "last.linear.weight", .bfloat16(joined: [])), ("last_linear_bias", "last.linear.bias", .bfloat16(joined: []))]

    /// The fusion's besides its blocks: the projector, `txtmlp`.
    static let fusionOthers: [Weights.Entry] = [
        ("projector", "txtfusion.projector.weight", .bfloat16(joined: [])), ("txt_norm", "txtmlp.0.scale", .float),
        ("txt1", "txtmlp.1.weight", .bfloat16(joined: [])), ("txt1_bias", "txtmlp.1.bias", .bfloat16(joined: [])),
        ("txt3", "txtmlp.3.weight", .bfloat16(joined: [])), ("txt3_bias", "txtmlp.3.bias", .bfloat16(joined: []))]
}
