import Foundation
import SiliCore

/// **Z-Image Turbo's DiT**: one evaluation, from the latent to the velocity, as diffusers'
/// `ZImageTransformer2DModel` (0.41) runs it. The configuration is `Config` (`ZImage.swift`); the functions are
/// `ZImageFunctions.swift`, the loop `ZImageTransformer.swift`.
/// **Its weights are read as published** (Comfy-Org's single files, the original names): a block's `attention.qkv`
/// `[3d, d]` is one product whose result is cut in three, where diffusers runs `to_q`, `to_k`, `to_v`.
///
///     latent [16, h, w] ─ patchify ─ x_embedder ─ ⟨x_pad_token⟩ ─ noise_refiner ×2 ─┐
///                                                                                   ├─ [image | caption] ─ layers ×30 ─ final ─ unpatchify
///     cap_feats [L, 2560] ─ RMS · γ ─ cap_embedder ─ ⟨cap_pad_token⟩ ─ context_refiner ×2 ┘
///     t_freq [256] (CPU) ─ t_embedder ─ adaln ──▶ every modulated block and the final layer
///     cos, sin [S, 128] (CPU) ──▶ every block's RoPE
///
///     block:  x += tanh(g₁) · RMS₂( out( sdpa( RoPE(RMS_h(q)), RoPE(RMS_h(k)), v ) ) )      on RMS₁(x) · (1 + s₁)
///             x += tanh(g₂) · RMS₄( w2( SiLU(w1 n) ⊙ w3 n ) )                               n = RMS₃(x) · (1 + s₂)
///             [s₁, g₁, s₂, g₂] = adaLN_modulation(adaln)   (no SiLU before it; none at all in the context refiners)
///
/// What a port by analogy would miss (read in `transformer_z_image.py` line by line):
///
///   - **The image comes first** in the unified sequence, and **both sequences are padded to a multiple of 32** with
///     learned tokens (`x_pad_token`, `cap_pad_token`) that nothing masks at batch 1.
///   - **The positions**: the caption's `(1 … L₃₂, 0, 0)` on every row, its padding included; the image's
///     `(L₃₂ + 1, h, w)`, its padding `(0, 0, 0)`; θ = 256, axes `[32, 48, 48]`, pairs interleaved (`view_as_complex`).
///   - **The RoPE table is f64 rounded to f32** (`precompute_freqs_cis`: `1/θ^(2i/d)` and the outer product in float64,
///     then `.float()`), **and the rotation runs in f32** (`x_in.float()`, complex64), rounded once (`type_as`). Both
///     tables are computed here on the CPU as the reference does (`rope`), not as `exp` in f32.
///   - **The timestep's sinusoid is f32** (`timestep_embedding` under disabled autocast: `−log(10⁴) · arange / 128`,
///     `exp`, `t · 1000 · freqs`, `[cos, sin]`). It is computed on the CPU (`timestepFrequencies`); the model is handed
///     `t` and multiplies it by `t_scale` = 1000 itself.
///   - **RMSNorm rounds before γ** (diffusers' `RMSNorm`: `hidden.to(weight.dtype) * weight`); the final layer's
///     LayerNorm has no affine, eps 10⁻⁶; gates are `tanh`, scales `1 + s`; SiLU exists only in `FinalLayer`.
///   - **The Compact's products** (ComfyUI's int8 `convrot`): `W = (q · s) · R`, so `x · Wᵀ = (x · R) · (q · s)ᵀ` — the
///     activation rotated by the file's Hadamard (by 256), once per value however many products read it (w1, w3).
package struct ZImageDiT {
    package let config: Config
    package let latentHeight, latentWidth, capLength: Int

    package init(config: Config, latentHeight: Int, latentWidth: Int, capLength: Int) {
        self.config = config
        (self.latentHeight, self.latentWidth, self.capLength) = (latentHeight, latentWidth, capLength)
    }

    package var headDim: Int { config.dim / config.heads }
    /// The image tokens of the latent grid: one per `patch × patch` cell.
    package var imageTokens: Int { (latentHeight / config.patch) * (latentWidth / config.patch) }
    package var imagePadded: Int { padded(imageTokens) }
    package var capPadded: Int { padded(capLength) }
    func padded(_ n: Int) -> Int { (n + config.seqMultiple - 1) / config.seqMultiple * config.seqMultiple }

    // MARK: - What the DiT reads of its file

    /// Every block, by the prefix its file gives it.
    package static func blocks(_ c: Config = .published) -> [String] {
        (0..<c.refiners).map { "noise_refiner.\($0)" } + (0..<c.refiners).map { "context_refiner.\($0)" } + (0..<c.layers).map { "layers.\($0)" }
    }

    /// A block's products, by their name in the functions and in the file, and their shape `[out, in]`. The context
    /// refiner's have no modulation.
    static func products(_ block: String, _ c: Config = .published) -> [(name: String, key: String, out: Int, inn: Int)] {
        let (d, h) = (c.dim, c.hidden)
        var all = [("qkv", "attention.qkv", 3 * d, d), ("out", "attention.out", d, d), ("w1", "feed_forward.w1", h, d),
                   ("w3", "feed_forward.w3", h, d), ("w2", "feed_forward.w2", d, h)]
        if !block.hasPrefix("context_refiner") { all.insert(("ada", "adaLN_modulation.0", 4 * d, c.adalnDim), at: 0) }
        return all
    }

    /// A block's vectors: (name in the functions, key in the file, length).
    static func vectors(_ block: String, _ c: Config = .published) -> [(name: String, key: String, count: Int)] {
        let (d, hd) = (c.dim, c.dim / c.heads)
        var all = [("norm1", "attention_norm1.weight", d), ("norm2", "attention_norm2.weight", d), ("q_norm", "attention.q_norm.weight", hd),
                   ("k_norm", "attention.k_norm.weight", hd), ("ffn_norm1", "ffn_norm1.weight", d), ("ffn_norm2", "ffn_norm2.weight", d)]
        if !block.hasPrefix("context_refiner") { all.append(("ada_bias", "adaLN_modulation.0.bias", 4 * d)) }
        return all
    }

    /// The products outside the blocks: (name in the functions, module in the file, out, in, its bias's name).
    static func edges(_ c: Config = .published) -> [(name: String, key: String, out: Int, inn: Int, bias: String)] {
        let (d, p) = (c.dim, c.patch * c.patch * c.channels)
        return [("mlp0", "t_embedder.mlp.0", 1024, c.adalnDim, "mlp0_bias"), ("mlp2", "t_embedder.mlp.2", c.adalnDim, 1024, "mlp2_bias"),
                ("final_ada", "final_layer.adaLN_modulation.1", d, c.adalnDim, "final_ada_bias"), ("x_embedder", "x_embedder", d, p, "x_embedder_bias"),
                ("cap_embedder", "cap_embedder.1", d, c.capDim, "cap_embedder_bias"), ("final_linear", "final_layer.linear", p, d, "final_linear_bias")]
    }

    /// Every tensor the DiT reads, by the name its published file gives it: what a file is recognized against. The
    /// Compact's int8 file keeps the same names and shapes for its codes, its scales beside.
    package static func tensors() -> [String: [Int]] {
        let c = Config.published
        var all: [String: [Int]] = ["x_pad_token": [1, c.dim], "cap_pad_token": [1, c.dim], "cap_embedder.0.weight": [c.capDim]]
        for e in edges(c) { all[e.key + ".weight"] = [e.out, e.inn]; all[e.key + ".bias"] = [e.out] }
        for b in blocks(c) {
            for p in products(b, c) { all["\(b).\(p.key).weight"] = [p.out, p.inn] }
            for v in vectors(b, c) { all["\(b).\(v.key)"] = [v.count] }
        }
        return all
    }

    /// **Every module a LoRA may adapt**: each product of the DiT, by the name Comfy-Org's file gives it and by
    /// diffusers' (the three it renames, landing on the product of the file's name), and each third of a block's `qkv`
    /// (`to_q`, `to_k`, `to_v`) landing on its rows, with its shape `[out, in]`. The front and the head included:
    /// `x_embedder`, `cap_embedder.1`, the final linear, the step embedder.
    package static func loraModules() -> [String: LoRAStack.Shape] {
        let c = Config.published
        var all: [String: LoRAStack.Shape] = [:]
        func note(_ product: String, _ out: Int, _ inn: Int) {
            for n in loraNames(product, c) { all[n] = LoRAStack.Shape(out: out, inn: inn, product: n == product ? nil : product) }
        }
        for e in edges(c) { note(e.key, e.out, e.inn) }
        for b in blocks(c) {
            for p in products(b, c) { note("\(b).\(p.key)", p.out, p.inn) }
            for (i, t) in ["to_q", "to_k", "to_v"].enumerated() {
                all["\(b).attention.\(t)"] = LoRAStack.Shape(out: c.dim, inn: c.dim, product: "\(b).attention.qkv", row: i * c.dim)
            }
        }
        return all
    }

    /// **A module's names in a LoRA's file**: Comfy-Org's (the file's own), and diffusers' for the three it renames.
    static func loraNames(_ name: String, _ c: Config) -> [String] {
        if name.hasSuffix(".attention.out") { return [name, String(name.dropLast("out".count)) + "to_out.0"] }
        if name == "x_embedder" { return [name, "all_x_embedder.\(c.patch)-1"] }
        if name.hasPrefix("final_layer.") { return [name, "all_final_layer.\(c.patch)-1." + name.dropFirst("final_layer.".count)] }
        return [name]
    }

    /// **The products a LoRA adapts, by the functions' names for them**: block `block`'s (`products`), the keys under
    /// the file's own names; the edges' (`edges`) for `block` nil.
    static func loraProducts(_ block: String?, _ c: Config = .published) -> [LoRAProduct] {
        guard let block else { return edges(c).map { LoRAProduct(name: $0.name, key: $0.key, inn: $0.inn, out: $0.out) } }
        return products(block, c).map { LoRAProduct(name: $0.name, key: "\(block).\($0.key)", inn: $0.inn, out: $0.out) }
    }

    /// **The ranks of a stack's low-rank paths**, by product name: the largest any block's takes (one function serves
    /// every block, `LoRAStack.rank`), an edge's its own; none for an empty stack.
    static func loraRanks(_ loras: LoRAStack, _ c: Config = .published) -> [String: Int] {
        guard !loras.isEmpty else { return [:] }
        var keys: [String: [String]] = [:]
        for p in loraProducts(nil, c) + blocks(c).flatMap({ loraProducts($0, c) }) { keys[p.name, default: []].append(p.key) }
        return keys.compactMapValues { k in
            let r = loras.rank(of: k, on: .zImage)
            return r > 0 ? r : nil
        }
    }

    /// **The DiT's file**, read under the prefix it keeps the DiT's names under (a checkpoint saved whole keeps
    /// `model.diffusion_model.`), found where it puts `x_pad_token`, a tensor of the DiT alone (`MapWeights.dit`).
    static func file(_ path: String) throws -> SafeTensors { try MapWeights.dit(path, marker: "x_pad_token") }

    // MARK: - The CPU's tables

    /// **The positions, `[S, 3]`**: the image's rows first, then the caption's (above).
    package func positions() -> [(Int, Int, Int)] {
        let w = latentWidth / config.patch
        var p: [(Int, Int, Int)] = (0..<imageTokens).map { (capPadded + 1, $0 / w, $0 % w) }
        p += Array(repeating: (0, 0, 0), count: imagePadded - imageTokens)
        p += (0..<capPadded).map { (1 + $0, 0, 0) }
        return p
    }

    /// **The RoPE tables as the block reads them**, `[rows, 128]` f32 each: `precompute_freqs_cis`'s angles (per axis
    /// 1/θ^(2i/d) in f64, position · freq rounded to f32, then cos and sin in f64, rounded to f32), each pair's angle
    /// twice, the sine's sign (−sin, sin). Rows past the positions at (0, 0, 0): cos 1, sin 0.
    package func rope(rows: Int) -> (cos: [Float], sin: [Float]) {
        let theta = Double(config.theta), hd = headDim
        let freqs = config.axesDims.map { n in (0..<(n / 2)).map { 1 / Foundation.pow(theta, Double(2 * $0) / Double(n)) } }
        var c = [Float](repeating: 1, count: rows * hd), s = [Float](repeating: 0, count: rows * hd)
        for (r, (p0, p1, p2)) in positions().enumerated() {
            var j = 0
            for (axis, position) in [p0, p1, p2].enumerated() {
                for f in freqs[axis] {
                    let angle = Double(Float(Double(position) * f))
                    let (cs, sn) = cosSin(angle)
                    c[r * hd + 2 * j] = cs; c[r * hd + 2 * j + 1] = cs
                    s[r * hd + 2 * j] = -sn; s[r * hd + 2 * j + 1] = sn
                    j += 1
                }
            }
        }
        return (c, s)
    }

    /// **`TimestepEmbedder.timestep_embedding` of `t · t_scale`, in float32 as torch computes it**: `[cos | sin]` of
    /// `t·1000 · exp(−log(10⁴) · i / 128)`, each operation one float rounding.
    package static func timestepFrequencies(_ t: Float, config: Config) -> [Float] {
        timestepSinusoid(t * config.timeScale, half: config.adalnDim / 2)
    }

    // MARK: - The oracle's boundaries

    /// The evaluation's boundaries the oracle writes (`probe_dit`), by its names.
    package enum Tap: String, CaseIterable, Sendable {
        case tEmbedder = "t_embedder_out", xEmbedder = "x_embedder_out", capEmbedder = "cap_embedder_out"
        case contextRefinerIn = "context_refiner_in", freqs = "block_freqs_cis"
        case noiseRefiner = "noise_refiner_out", contextRefiner = "context_refiner_out", unified = "unified_in"
        case layer0 = "layers_0_out", layer1 = "layers_1_out", layer15 = "layers_15_out", layer29 = "layers_29_out"
        case finalLayer = "final_layer_out"

        static let layers: [Int: Tap] = [0: .layer0, 1: .layer1, 15: .layer15, 29: .layer29]
    }

    /// Layer 0's inner boundaries (`probe_dit`), by the oracle's names.
    package enum BlockTap: String, CaseIterable, Sendable {
        case adaln = "layer0_adaln_raw", norm1 = "layer0_attn_norm1_out", attentionIn = "layer0_attn_in"
        case q = "layer0_q", k = "layer0_k", v = "layer0_v", qNormed = "layer0_q_normed", kNormed = "layer0_k_normed"
        case sdpa = "layer0_sdpa_out", attention = "layer0_attention_out", norm2 = "layer0_attn_norm2_out"
        case ffnNorm1 = "layer0_ffn_norm1_out", ffnIn = "layer0_ffn_in", w1 = "layer0_w1_out", w3 = "layer0_w3_out"
        case swiglu = "layer0_swiglu_out", feedForward = "layer0_feed_forward_out", ffnNorm2 = "layer0_ffn_norm2_out"
        case out = "layers_0_out"
    }
}
