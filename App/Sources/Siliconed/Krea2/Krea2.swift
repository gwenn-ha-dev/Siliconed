import Foundation

/// The Krea 2 Turbo DiT — `Krea2Transformer2DModel`, 12.8 G parameters —, in bf16 (`Krea2Functions`, run by `Krea2Session`).
///
///     12 Qwen3-VL hidden states [T, 12, 2560] ─ fusion (2 blocks per token, projector, 2 blocks)
///                                                ─ txt_in ─ text [T, 6144] ───────────────────┐
///     latent [16, H, W] ─ patchify (SLOW channel) ─ img_in ─ image [S, 6144] ──── concat ──────┤
///                                                                                            ▼
///     σ ─ sinusoid 256 (cos first, ×1000) ─ MLP ─ temb [6144] ─ GELU ─ time_mod_proj ─▶ blocks ×28
///                                                        │                                    │
///                                                        └──────────── final layer ◀── image only
///
/// One block, single-stream — text and image in **one** sequence, one attention:
///
///     m = temb_mod + block_table          (6 × 6144: prescale, preshift, pregate, postscale, …)
///     x += pregate · Wo( SDPA(RoPE(n(q)), RoPE(n(k)), v) ⊙ σ(Wg·h) ),  h = RMSNorm(x)(1+prescale)+preshift
///     x += postgate · SwiGLU( RMSNorm(x)(1+postscale) + postshift )
///
/// What sets it apart from the other targets, and what a port by analogy would miss:
///
///   - **the attention is an ordinary softmax SDPA**, followed by a **sigmoid gate on its
///     output**, element by element. Our first notes announced a "sigmoid attention": that was
///     a reading of the paper, the code says `dispatch_attention_fn` then `* torch.sigmoid(gate)`.
///     No new kernel, then;
///   - **GQA 48/12**: four `q` heads per `k`/`v` head (`repeat_interleave`); Core AI attends equal heads, so the
///     four query heads of a key head are read as one head of four times the rows, no key repeated (`Krea2Functions`);
///   - the modulation is **shared**: a single `time_mod_proj` per evaluation, and each block only
///     adds a learned table to it — 3 `gemv` per evaluation instead of a modulation per block;
///   - the RMSNorms carry a **zero-centered** weight (`1 + w`), the sum made in the graph (fp32);
///   - the text goes through a learned **fusion** of the twelve drawn layers (`Krea2Fusion`), once a
///     prompt, before the evaluations;
///   - the RoPE is **already interleaved** (Flux convention, θ = 1000, axes 32/48/48), and the text is
///     at position `(0, 0, 0)` — it does not rotate. The image starts at `(0, 0, 0)` too: its first
///     token shares the text's position, with no offset (Qwen-Image shifts its image, Krea does not);
///   - the **final layer reads `temb` as it is**, not `gelu(temb)` as `time_mod_proj` does, and adds its
///     own `[2, d]` table (scale, then shift);
///   - **σ is rounded to bf16 before the sinusoid**: the official pipeline casts the timestep to the latents'
///     dtype (`.to(latents.dtype)`), and the render tells the DiT that value (`Krea2Transformer.modelTime`);
///   - **the text rows of block 0's output decide most of the velocity's error**: exact text rows from there
///     leave a sixth of it, and a random error of rounding size on them, in any direction, as much as the
///     engine's or more. The engine and the fp32 reference are two draws of that ill-conditioning; one draw
///     against one draw does not judge a pass, the latent planes do;
///   - **a LoRA reaches the products of the 28 blocks and of the fusion's four, the timestep's and the final layer's**
///     (`Krea2Transformer.loraModules`), under Comfy's names or diffusers'; one naming the projector, `txtmlp` or `first`
///     is refused.
///
/// For a port of the undistilled Krea-2-Raw, two things this Turbo port does not need: μ then depends on
/// the image's length (`calculate_shift`, `max_image_seq_len` 6400, not Flux's 4096), and the guidance is
/// `cond + g · (cond − uncond)`, i.e. a scale of `1 + g` in the usual convention.
///
/// **The text padding is the engine's, and this is exact.** The reference pads the text to 512 tokens
/// and masks the padding tokens *as keys*, everywhere; their own rows are
/// never read again (sliced off before the final layer). Here the text is padded to its palier (512, then
/// by 512: `Krea2Session.Geometry.palier`) and the keys past it are masked (−10⁴): each useful row
/// sees exactly the same keys, same sum. The golden tensors, computed with the reference's padding, are
/// the check.
package enum Krea2DiT {
    package struct Config {
        package let dim, heads, kvHeads, headDim, hidden, layers, channels, timeDim: Int
        package let axes: [Int]
        package let theta, eps: Double

        /// **The published model's** (`transformer/config.json`, `Krea2Published`): what a weights file of the family is
        /// read with — Comfy-Org's single files carry no configuration.
        package static let published = Config(dim: 6144, heads: 48, kvHeads: 12, headDim: 128, hidden: 16384, layers: 28, channels: 64,
                                              timeDim: 256, axes: [32, 48, 48], theta: 1000, eps: 1e-5)
    }
}


/// **The Krea 2 text**, as `Krea2Pipeline.get_text_hidden_states` prepares it.
///
///     system template + prompt ─ tokenizer, 1 024 at most ───┐
///     "assistant" suffix (5 tokens) ───────────────────────┴─ Qwen3-VL ─ hidden_states[2, 5, …, 35]
///                                                            ─ the 34 template tokens removed
///
/// The reference pads in the middle — the prompt, **then** the padding, then the suffix — and masks.
/// Here nothing is padded: Qwen's attention is causal and the padding is never a visible
/// key, and the reference's positions (`cumsum(mask) − 1`) are precisely those of a
/// sequence without padding. Same sum, minus the products by zero.
package enum Krea2Text {
    package static let template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        + "texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
        + "<|im_start|>user\n"
    package static let suffix = "<|im_end|>\n<|im_start|>assistant\n"
    /// `prompt_template_encode_start_idx`: the template tokens, removed after the encoder.
    package static let templateTokens = 34
    package static let length = 512

    /// The layers drawn from the encoder, read from the published `model_index.json` — not copied:
    /// the file itself is in the code (`Krea2Published`, its git blob checked by a test), `[2, 5, …, 35]`.
    package static let sockets: [Int] =
        (Krea2Published.modelIndex["text_encoder_select_layers"]?.elements ?? []).compactMap(\.integer)

    /// The identifiers the encoder receives, template included, without padding.
    package static func identifiers(_ prompt: String, tokenizer: Tokenizer) -> [Int] {
        let head = Array(tokenizer.encode(template + prompt)
            .prefix(length + templateTokens - tokenizer.encode(suffix).count))
        return head + tokenizer.encode(suffix)
    }

    /// `identifiers` for a render: the whole prompt, **refused past `Request.maximumPromptTokens`**,
    /// never truncated; `prompt`, the prompt's own tokens. `identifiers` keeps the reference's cut (507 tokens of
    /// prompt), for the checks.
    package static func admitted(_ prompt: String, tokenizer: Tokenizer) throws -> (ids: [Int], prompt: Int) {
        let head = tokenizer.encode(template + prompt)
        try Request.admitPrompt(tokens: head.count - templateTokens, max: Request.maximumPromptTokens)
        return (head + tokenizer.encode(suffix), head.count - templateTokens)
    }
}

extension Krea2DiT {
    /// The Krea 2 Turbo schedule: `sigmas = linspace(1, 1/N, N)`, shifted by the **exponential
    /// shift** at `μ = 1.15` (`is_distilled`: fixed μ, whatever the resolution) —
    /// `σ' = e^μ / (e^μ + (1/σ − 1))` —, then a zero terminal σ.
    ///
    /// **In float32, operation by operation**, as `set_timesteps` computes it: NumPy's double ramp cast
    /// to float32, then `math.exp(μ) / (math.exp(μ) + (1 / t − 1) ** 1.0)` on a float32 array — `e^μ`
    /// rounded to float32 (a weak Python float), `** 1.0` the identity. The same formula in double,
    /// rounded once, is one bit off on 4 of the 8 σ at N = 8 (402 of 1 275 for N = 1…50). To the bit
    /// against diffusers (`SolverTests`).
    package static func sigmas(steps: Int, mu: Double = 1.15) -> [Float] {
        let e = Float(exp(mu))
        return NumPyRamp.linspace(1, 1 / Double(steps), steps).map { s -> Float in
            let t = Float(s)
            return e / (e + (1 / t - 1))
        } + [0]
    }
}
