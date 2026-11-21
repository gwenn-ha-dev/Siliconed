import Foundation

/// **The Krea 2 Turbo DiT**: one evaluation, from the latent and the twelve Qwen3-VL taps to the
/// velocity, as diffusers' `Krea2Transformer2DModel` (`transformer_krea2.py`) runs it in bf16. The configuration and the
/// LoRA targets are `Krea2DiT` (`Krea2.swift`). **Its weights are read as published** (Comfy-Org's single files, krea-ai's
/// names): `blocks.<i>.attn.wq` where diffusers says `transformer_blocks.<i>.attn.to_q`, a block's table `mod.lin`
/// `[6 · d]` flat, a norm's `scale` without its `1 +`.
///
///     taps [T, 12, 2560] ─ layerwise blocks ×2 (attention across the 12 taps, per token) ─ projector 12→1
///                        ─ refiner blocks ×2 (attention across the tokens) ─ txtmlp ─ text [T, 6144] ──┐
///     latent [16, h, w] ─ patchify (C slowest) ─ first ─ image [N, 6144] ────────────────────── concat ┤
///     t_freq [256] (CPU) ─ tmlp ─ temb ─ GELU ─ tproj ─ temb_mod [6·6144] ────────────────▶ every block │
///     cos, sin [T+N, 64] (CPU) ──▶ every block's RoPE                                                   ▼
///                                                    blocks ×28 ─ image rows ─ final layer ─ unpatchify
///
///     block:  m = temb_mod + table                          (6 × 6144: prescale, preshift, pregate, postscale, postshift, postgate)
///             x += pregate · Wo( SDPA(RoPE(n(q)), RoPE(n(k)), v) · σ(Wg h) ),  h = (1 + prescale) · RMS₁(x) + preshift
///             x += postgate · down( SiLU(gate h') · up h' ),                    h' = (1 + postscale) · RMS₂(x) + postshift
///
/// What a port by analogy would miss (read in `transformer_krea2.py` and `pipeline_krea2.py`, line by line):
///
///   - **`Krea2RMSNorm` rounds once**: `F.rms_norm(x.float(), weight = 1 + w)` in f32, weight included, then `.to(dtype)`.
///     The file holds `w` in f32; the `1 +` is an f32 sum here. diffusers' own `RMSNorm` rounds twice,
///     this one does not.
///   - **The attention is an ordinary softmax SDPA, then a sigmoid gate** on its output, element by element, before `to_out`
///     (`Krea2AttnProcessor`: `hidden_states * torch.sigmoid(gate)`).
///   - **GQA 48/12** by `repeat_interleave(4, dim=2)` on k and v; the heads repeated before the attention.
///   - **The fusion's two stages differ in what the batch is**: the layerwise blocks see the 12 taps of one token as a sequence
///     (batch T, length 12), the refiners see the tokens (batch 1, length T). Neither has RoPE or modulation; the engine
///     pads the text to its palier and masks the keys past it, as the reference masks its padding keys.
///   - **`time_embed` rounds its sinusoid to bf16** (`emb.to(dtype)`) after an f32 `cos | sin` of `t · 1000 · freqs`; `t` is
///     itself the pipeline's `(t / 1000).to(bf16)` (`pipeline_krea2.py:644`). The final layer reads `temb`, not `gelu(temb)`.
///   - **The RoPE tables are f64 rounded to f32** (`get_1d_rotary_pos_embed` with `freqs_dtype = float64`: `1 / θ^(2i/d)` and
///     the outer product in double, `cos` and `sin` in double, then `.float()`), axes `[32, 48, 48]`, θ = 1000, pairs
///     adjacent; the rotation `x.float()·cos + rot.float()·sin` runs in f32 and is rounded once. The text sits at (0, 0, 0),
///     the image starts at (0, 0, 0) too: no offset.
///   - **No product's rounding is forced.** Core AI's GPU fuses the operation that follows a product onto its f32 sum,
///     where torch rounds the product to bf16 first; what is judged is the distance to the witness, not torch's bits. A
///     pad-and-slice barrier behind every product (264 an evaluation) changed neither the step (5.53 and 5.50 s at 1024²,
///     M5 Pro) nor `dit`'s judgment (`model_out` 2.004e-2 with, 2.009e-2 without), and the Compact's `dit`
///     is green without (1.684e-2 for 1.98e-2; 2.133e-2 with).
///   - **The Compact's products** (ComfyUI's int8 `convrot`): activation rotated by the Hadamard (256), once per value, then
///     cast into int8 codes against the weight's own (W8A8). **A weight the file keeps in f32** (`first`, `tmlp`, `txtmlp`, the tables) is
///     loaded in bf16, as `from_pretrained(torch_dtype=bfloat16)` loads it; the norms stay f32 (`_keep_in_fp32_modules`).
package enum Krea2Transformer {
    /// The products of the DiT by their published names, `[out, in]`: the 28 blocks', the text fusion's four blocks', its
    /// projector, `txtmlp`'s, the embedders' and the final layer's.
    static func products(_ c: Krea2DiT.Config = .published) -> [String: LoRAStack.Shape] {
        let (d, kv, w, th) = (c.dim, c.kvHeads * c.headDim, 2560, 6912)
        var all: [String: LoRAStack.Shape] = [:]
        func put(_ name: String, _ out: Int, _ inn: Int) { all[name] = LoRAStack.Shape(out: out, inn: inn) }
        for i in 0..<c.layers {
            let p = "blocks.\(i)."
            for name in ["wq", "gate", "wo"] { put(p + "attn." + name, d, d) }
            put(p + "attn.wk", kv, d); put(p + "attn.wv", kv, d)
            put(p + "mlp.up", c.hidden, d); put(p + "mlp.gate", c.hidden, d); put(p + "mlp.down", d, c.hidden)
        }
        for p in (0..<2).map({ "txtfusion.layerwise_blocks.\($0)." }) + (0..<2).map({ "txtfusion.refiner_blocks.\($0)." }) {
            for name in ["wq", "wk", "wv", "gate", "wo"] { put(p + "attn." + name, w, w) }
            put(p + "mlp.up", th, w); put(p + "mlp.gate", th, w); put(p + "mlp.down", w, th)
        }
        put("txtfusion.projector", 1, 12)
        put("txtmlp.1", d, w); put("txtmlp.3", d, d)
        put("tmlp.0", d, c.timeDim); put("tmlp.2", d, d); put("tproj.1", 6 * d, d)
        put("first", d, c.channels); put("last.linear", c.channels, d)
        return all
    }

    /// Every tensor the DiT reads, by the name its published file gives it: what a file is recognized against.
    package static func tensors() -> [String: [Int]] {
        let c = Krea2DiT.Config.published, d = c.dim
        var all: [String: [Int]] = [:]
        for (name, s) in products(c) { all[name + ".weight"] = [s.out, s.inn] }
        for bias in ["tmlp.0", "tmlp.2", "first", "txtmlp.1", "txtmlp.3"] { all[bias + ".bias"] = [d] }
        all["tproj.1.bias"] = [6 * d]
        all["last.linear.bias"] = [c.channels]
        all["last.modulation.lin"] = [2, d]
        all["last.norm.scale"] = [d]
        all["txtmlp.0.scale"] = [2560]
        for i in 0..<c.layers {
            let p = "blocks.\(i)."
            all[p + "mod.lin"] = [6 * d]
            all[p + "prenorm.scale"] = [d]; all[p + "postnorm.scale"] = [d]
            all[p + "attn.qknorm.qnorm.scale"] = [c.headDim]; all[p + "attn.qknorm.knorm.scale"] = [c.headDim]
        }
        for p in (0..<2).map({ "txtfusion.layerwise_blocks.\($0)." }) + (0..<2).map({ "txtfusion.refiner_blocks.\($0)." }) {
            all[p + "prenorm.scale"] = [2560]; all[p + "postnorm.scale"] = [2560]
            all[p + "attn.qknorm.qnorm.scale"] = [c.headDim]; all[p + "attn.qknorm.knorm.scale"] = [c.headDim]
        }
        return all
    }

    /// **Every module a LoRA may adapt**: the products of the DiT's 28 blocks and of the text fusion's four, the
    /// timestep's three and the final layer's (`edges`), by their published names and by diffusers' (landing on the
    /// product of the published name) — what the trainers adapt, and what the functions add a low rank beside
    /// (`Krea2Functions`, `Krea2Shared`). Read 21/11: a distillation LoRA (lvladikov's 4-step, rank 64) adapts the 224
    /// products of the blocks and the four edges, in either naming; two style LoRAs read 19/11, the 288 products of the
    /// blocks and of the fusion. A file naming the projector, `txtmlp` or `first` is refused by name, never computed
    /// without it.
    package static func loraModules() -> [String: LoRAStack.Shape] {
        let adapted = products().filter { name, _ in
            ["blocks.", "txtfusion.layerwise_blocks.", "txtfusion.refiner_blocks."].contains(where: name.hasPrefix) || edges.contains(name)
        }
        var all = adapted
        for (name, s) in adapted { all[diffusersName(name)] = LoRAStack.Shape(out: s.out, inn: s.inn, product: name) }
        return all
    }

    /// The products besides the blocks' a LoRA reaches, computed once an evaluation: the timestep's and the final layer's.
    static let edges = ["tmlp.0", "tmlp.2", "tproj.1", "last.linear"]

    /// **A product's name in diffusers' `Krea2Transformer2DModel`**, what a trainer on diffusers writes: the block
    /// `transformer_blocks.<i>.` (the fusion's `text_fusion.`), the attention's `to_q` `to_k` `to_v` `to_out.0` `to_gate`,
    /// the feed-forward `ff.`; the edges by their own names (the former engine's table, v0.21 `Recipes.krea2`).
    static func diffusersName(_ name: String) -> String {
        let edges = ["tmlp.0": "time_embed.linear_1", "tmlp.2": "time_embed.linear_2", "tproj.1": "time_mod_proj", "last.linear": "final_layer.linear"]
        if let e = edges[name] { return e }
        var n = name.hasPrefix("blocks.") ? "transformer_blocks." + name.dropFirst("blocks.".count) : "text_fusion." + name.dropFirst("txtfusion.".count)
        for (comfy, diffusers) in [("wq", "to_q"), ("wk", "to_k"), ("wv", "to_v"), ("wo", "to_out.0"), ("gate", "to_gate")] where n.hasSuffix(".attn." + comfy) {
            n = String(n.dropLast(comfy.count)) + diffusers
        }
        return n.replacingOccurrences(of: ".mlp.", with: ".ff.")
    }

    // MARK: - The CPU's tables

    /// **`Krea2TimestepEmbedding`'s sinusoid in float32 as torch computes it**: `[cos | sin]` of `(t · 1000) · exp(−ln(10⁴) · i / 128)`,
    /// each operation one float rounding. `t` is the pipeline's bf16 timestep.
    package static func timestepFrequencies(_ t: Float, config: Krea2DiT.Config) -> [Float] {
        timestepSinusoid(t * 1000, half: config.timeDim / 2)
    }

    /// **What the DiT is told** (`pipeline_krea2.py:644`): `(t / 1000)` in f32 from `t = σ · 1000`, then `.to(bf16)`.
    package static func modelTime(sigma: Float) -> Float { BFloat.round(sigma * 1000 / 1000) }

    // MARK: - The oracle's boundaries

    package enum Tap: String, CaseIterable, Sendable {
        case temb, tembMod = "temb_mod", textFused = "text_fused", txtIn = "txt_in_out", imgIn = "img_in_out"
        case block0Attention = "block0_attn_out", block0Feed = "block0_ff_out", block0 = "block0_out"
        case block1 = "block1_out", block13 = "block13_out", block27 = "block27_out"
    }
}
