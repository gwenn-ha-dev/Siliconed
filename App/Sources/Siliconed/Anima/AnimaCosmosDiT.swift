import Foundation

/// **Anima Turbo's DiT**: one evaluation, from the latent to the velocity, as diffusers'
/// `CosmosTransformer3DModel` (`transformer_cosmos.py`) runs it in bf16. The configuration and the σ schedule are
/// `AnimaDiT` (`Anima.swift`), the block's formula is written there too. **Its weights are read as published**
/// (circlestone-labs' single file, Cosmos-Predict2's names under `model.diffusion_model.`): `blocks.<i>.self_attn.q_proj`
/// where diffusers says `transformer_blocks.<i>.attn1.to_q`. The text conditioner's weights lie in the same file
/// (`llm_adapter.`, `AnimaConditioner`).
///
///     latent [16, h, w] ⊕ padding mask (zeros) ─ patchify ─ x_embedder ───────────────────────────────┐
///     t_freq [2048] (CPU) ─ [cos | sin] bf16 ─┬─ linear_1 · SiLU · linear_2 ─ temb [6144]               │
///                                              └─ RMSNorm ─ embedded [2048]                             ▼
///     context [512, 1024] ──────────────────────────────────────────────────────▶ blocks ×28 ─ final_layer ─ unpatchify
///     cos, sin [S, 64] (CPU) ──▶ every self-attention's RoPE
///
///     block:  [shift, scale, gate]ᵢ = adaln_modulationᵢ.2( adaln_modulationᵢ.1( SiLU(embedded) ) ) + temb   (self_attn, cross_attn, mlp)
///             x += gate₁ · self_attn( LN(x) · (1 + scale₁) + shift₁ )           QK-norm, RoPE
///             x += gate₂ · cross_attn( LN(x) · (1 + scale₂) + shift₂ ; context ) QK-norm, no RoPE
///             x += gate₃ · W₂ GELU( W₁ ( LN(x) · (1 + scale₃) + shift₃ ) )
///
/// What a port by analogy would miss (read in `transformer_cosmos.py`, `embeddings.py`, `normalization.py`):
///
///   - **The patch is channel-major**: `c·4 + ph·2 + pw` (`CosmosPatchEmbed`: `permute(0, 2, 4, 6, 1, 3, 5, 7)`), a
///     seventeenth channel (`concat_padding_mask`, zeros here) included; the output is `(ph, pw, c)`-major
///     (`unflatten(2, (p_h, p_w, p_t, -1))`) — not the patch layout's inverse.
///   - **The timestep's sinusoid is f32** (`get_timestep_embedding`: `exp(−ln(10⁴) · i / 1024)`, `t · freq`, `[cos | sin]` after
///     `flip_sin_to_cos`) brought to bf16 (`type_as(hidden_states)`): on the CPU (`timestepFrequencies`), given. The model is
///     told `t` as the pipeline gives it, `bf16(bf16(σ·1000) / 1000)` (`AnimaRender.modelTime`).
///   - **diffusers' `RMSNorm`** (`normalization.py`: the normalized value rounded to bf16, then × γ) for the QK norms (ε = 10⁻⁵,
///     `Attention`'s default, not 10⁻⁶) and for `time_embed.norm` (10⁻⁶).
///   - **The time path computes in f32 and rounds once, at its outputs** (`AnimaFunction`, `time` and `modulation`): its
///     values are then the reference's taken further from rounding, not its bf16 products one by one. No product's
///     rounding is forced.
///   - **The RoPE is the split `rotate_half` of `apply_rotary_emb(use_real_unbind_dim=-2)`**, run in f32 and rounded once to
///     bf16: a head's pair `p` is `(x[p], x[64 + p])`, its two halves, not two neighbours. The table holds one angle per pair,
///     64 of them: time 22, height 21, width 21 (`CosmosRotaryPosEmbed`: `dim_t = 44`, `dim_h = dim_w = 42` at a head of 128;
///     the reference repeats them over both halves). The table is f32 (`seq`, `freqs`, `cos`, `sin`) and computed on the CPU.
///   - **LayerNorm without affine, ε = 10⁻⁶; `(1 + scale)` is a bf16 sum**; the gate multiplies without a tanh.
///   - **The GELU is exact** (`approximate="none"`), the FFN ungated; no linear has a bias (`bias=False` everywhere).
///   - **The final layer's norm takes `shift` first, then `scale`** (two chunks, `CosmosAdaLayerNorm`), from `temb[..., :2d]`.
///   - **The Compact's products** (Bedovyy's int8 `convrot` on the attention projections and the MLP of the inner blocks): the
///     activation rotated by the map's Hadamard (256), once per value however many products read it (the context too, rotated
///     once for all 28 blocks), then cast into int8 codes against the weight's own (`int8Linear`, W8A8).
package struct AnimaCosmosDiT {
    package let config: AnimaDiT.Config
    package let latentHeight, latentWidth: Int
    /// What the published file puts before each of the DiT's names.
    package static let prefix = "model.diffusion_model."

    /// Every tensor the DiT reads, by the name its published file gives it: what a file is recognized against. The
    /// products by their weight `[out, in]`, the norms' gains by theirs.
    package static func tensors() -> [String: [Int]] {
        let c = AnimaDiT.Config.published
        var all = Dictionary(uniqueKeysWithValues: products().map { (prefix + $0.key + ".weight", [$0.value.out, $0.value.inn]) })
        for i in 0..<c.layers {
            for norm in ["self_attn.q_norm", "self_attn.k_norm", "cross_attn.q_norm", "cross_attn.k_norm"] {
                all[prefix + "blocks.\(i).\(norm).weight"] = [c.headDim]
            }
        }
        all[prefix + "t_embedding_norm.weight"] = [c.dim]
        return all
    }

    /// **Every module a LoRA may adapt**: each block's products and its modulation's, by their published names
    /// (`AnimaEvaluation.loraProducts`); the embedders, the time path, the final layer and the text adapter are not.
    package static func loraModules() -> [String: LoRAStack.Shape] {
        Dictionary(uniqueKeysWithValues: (0..<AnimaDiT.Config.published.layers).flatMap(AnimaEvaluation.loraProducts).map {
            ($0.key, LoRAStack.Shape(out: $0.out, inn: $0.inn))
        })
    }

    /// The DiT's products by module, their weights' shapes `[out, in]` (none has a bias).
    static func products() -> [String: LoRAStack.Shape] {
        let c = AnimaDiT.Config.published, (d, a) = (c.dim, c.adalnLoRA)
        let patch = (c.channels + 1) * c.patch * c.patch, out = c.patch * c.patch * c.channels
        var all: [String: LoRAStack.Shape] = [
            "x_embedder.proj.1": .init(out: d, inn: patch), "t_embedder.1.linear_1": .init(out: d, inn: d),
            "t_embedder.1.linear_2": .init(out: 3 * d, inn: d), "final_layer.linear": .init(out: out, inn: d),
            "final_layer.adaln_modulation.1": .init(out: a, inn: d), "final_layer.adaln_modulation.2": .init(out: 2 * d, inn: a)]
        for i in 0..<c.layers {
            let p = "blocks.\(i)."
            for m in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj", "cross_attn.q_proj", "cross_attn.output_proj"] {
                all[p + m] = .init(out: d, inn: d)
            }
            all[p + "cross_attn.k_proj"] = .init(out: d, inn: c.textDim)
            all[p + "cross_attn.v_proj"] = .init(out: d, inn: c.textDim)
            all[p + "mlp.layer1"] = .init(out: c.hidden, inn: d)
            all[p + "mlp.layer2"] = .init(out: d, inn: c.hidden)
            for m in ["self_attn", "cross_attn", "mlp"] {
                all[p + "adaln_modulation_\(m).1"] = .init(out: a, inn: d)
                all[p + "adaln_modulation_\(m).2"] = .init(out: 3 * d, inn: a)
            }
        }
        return all
    }

    package init(config: AnimaDiT.Config, latentHeight: Int, latentWidth: Int) {
        self.config = config
        (self.latentHeight, self.latentWidth) = (latentHeight, latentWidth)
    }

    package var patchesHigh: Int { latentHeight / config.patch }
    package var patchesWide: Int { latentWidth / config.patch }
    package var tokens: Int { patchesHigh * patchesWide }

    // MARK: - The CPU's tables

    /// **`CosmosRotaryPosEmbed`'s table**, one angle per rotated pair: `[tokens, headDim / 2]` cos and sin, f32 as torch
    /// computes them — the frequencies `1 / θ^(2i/dim)` with `θ = 10⁴ · ntk`, `ntk = scale^(dim / (dim − 2))`, the angle
    /// `position · frequency`, laid out time, height, width.
    package func rope() -> (cos: [Float], sin: [Float]) {
        let hd = config.headDim
        let dimH = hd / 6 * 2, dimW = dimH, dimT = hd - dimH - dimW
        func frequencies(_ dim: Int, _ scale: Double) -> [Float] {
            let theta = Float(10_000.0 * pow(scale, Double(dim) / Double(dim - 2)))
            return (0..<(dim / 2)).map { 1 / powf(theta, Float(2 * $0) / Float(dim)) }
        }
        let ft = frequencies(dimT, config.ropeScale[0]), fh = frequencies(dimH, config.ropeScale[1]), fw = frequencies(dimW, config.ropeScale[2])
        var cos: [Float] = [], sin: [Float] = []
        cos.reserveCapacity(tokens * hd / 2); sin.reserveCapacity(tokens * hd / 2)
        for row in 0..<patchesHigh {
            for column in 0..<patchesWide {
                let angles = ft.map { Float(0) * $0 } + fh.map { Float(row) * $0 } + fw.map { Float(column) * $0 }
                for a in angles { let (c, s) = cosSin(Double(a)); cos.append(c); sin.append(s) }
            }
        }
        return (cos, sin)
    }

    /// **`get_timestep_embedding` of the model's `timestep`, in float32 as torch computes it**: `[cos | sin]` of
    /// `t · exp(−ln(10⁴) · i / 1024)`, each operation one float rounding.
    package static func timestepFrequencies(_ t: Float, config: AnimaDiT.Config) -> [Float] {
        timestepSinusoid(t, half: config.dim / 2)
    }

    // MARK: - The oracle's boundaries

    package enum Tap: String, CaseIterable, Sendable {
        case temb, embeddedTimestep = "embedded_timestep", patchEmbed = "patch_embed_out"
        case block0 = "block0_out", block1 = "block1_out", block13 = "block13_out", block27 = "block27_out"
        case block0Attention1 = "block0_attn1_out", block0Attention2 = "block0_attn2_out", block0FeedForward = "block0_ff_out"

        package static let blocks: [Int: Tap] = [0: .block0, 1: .block1, 13: .block13, 27: .block27]
    }

    /// Block 0's branches (`probe_dit`) and its output, by the oracle's names.
    package enum BlockTap: String, CaseIterable, Sendable {
        case attention1 = "block0_attn1_out", attention2 = "block0_attn2_out", feedForward = "block0_ff_out", out = "block0_out"
    }
}
