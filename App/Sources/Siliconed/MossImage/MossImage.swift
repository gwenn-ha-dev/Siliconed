import Foundation

/// The MOSS-Image-SFT DiT — `MOSSImageTransformer`, 5.08 G parameters —, in bf16 as published
/// (`MossImageDiTProgram`).
///
///     Qwen3.5 last hidden state [T, 4096] ─ Linear ─ 2 text blocks (own weights, biases, no modulation) ─┐
///     latent [128, H, W] (VAE patched 2×2, 16 px per cell) ─ tokens [H·W, 128] ─ Linear ────────────────┤
///                                                                                                       ▼
///                                                          one sequence: image rows, then text rows
///     t ─ sinusoid 256 [cos | sin] (fp64) ─ Linear ─ SiLU ─ Linear ─ time ─┬─ Linear(6d) ─▶ 42 blocks
///                                                                           └─ Linear(2d) ─▶ head: shift, scale
///     image rows ─ RMSNorm(final) · (1 + scale) + shift ─ Linear(128, no bias) ─ velocity [128, H, W]
///
/// **One block** — every block reads the same six vectors `[shift_a, scale_a, gate_a, shift_f, scale_f, gate_f]`:
///
///     x += gate_a · Wo( SDPA(RoPE(RMS_h(Wq n)), RoPE(RMS_h(Wk n)), Wv n) )   n = RMS(x)·w·(1 + scale_a) + shift_a
///     x += gate_f · W_down( SITU(h, g) )              [h | g] = W_up_gate m,  m = RMS(x)·w·(1 + scale_f) + shift_f
///
/// What sets it apart from the other targets, and what a port by analogy would miss:
///
///   - **the modulation is shared by all 42 blocks, and has no table**: a single `shared_block_modulation`
///     (3072 → 6·3072) per evaluation, read as it comes — the blocks add nothing to it (≠ Krea 2's
///     `temb_mod + table`). It is **not** passed through a SiLU first (the usual adaLN does), and its order is
///     `shift, scale, gate` twice — the shift first. The gates multiply the sublayers as they are, no `tanh`;
///   - **the activation is SITU, not SwiGLU**: `4·tanh(g/4)·σ(g) · 25·tanh(h/25)` — gate and value both capped.
///     `ffn_up_gate` is one fused Linear whose output is `[hidden | gate]` (the **value first**, the gate
///     second, `chunk(2)`); the text blocks keep `ffn_up`/`ffn_gate` apart, with biases;
///   - **the RoPE has four axes** `(r, f, h, w)`, 16 bands each (`head_dim / 8`), θ = 10⁴, `head_dim` = 128 split
///     **in halves** (`rotate_half`, not interleaved pairs): the table is `[sin(64) | cos(64)]`, entry
///     `axis·16 + band`. The image sits at `(0, T, h − (H−1)/2, w − (W−1)/2)` — **centred** on zero, and its frame
///     axis is the **text length** —, the text at `(0, i, 0, 0)`. The bands are fp32 (`1 / θ^(i/16)`), the
///     angles fp32 products: never rounded to bf16 (the reference's own `_apply` guards it);
///   - **the text goes through its own two blocks first** — a Linear 4096 → 3072, then RMSNorm / QKV / attention
///     over the text rows alone (their RoPE rows are the text's) / `proj` / SITU FFN, biases everywhere —, and
///     only then joins the image. No mask: every token of the sequence sees every other;
///   - **the sequence is image rows, then text rows**, as the reference's; with no mask and every row's RoPE
///     attached to the row, every position sees every other;
///   - **the head**: `final_norm_video` (a plain RMSNorm, weight kept) then `x·(1 + scale) + shift` with
///     `video_head_modulation(time) → [shift | scale]`; `final_linear_video` has **no bias**;
///   - the time vector is the sinusoid **in fp64** over the fp32 timestep the scheduler hands (`σ·1000`, rounded
///     once to fp32), `[cos | sin]` with `10⁴^(−j/128)`, cast to fp32 before the MLP, computed on the CPU. The MLP
///     and both modulations are the graph's, in bf16;
///   - **the official recipe skips block 22 in the negative branch** (`skip_layers`, zero-based): the block is not
///     computed, not read, and the residual passes through it. Not ported: the render runs no negative branch;
///   - the RMSNorms are plain (`x·rsqrt(mean(x²) + 10⁻⁶)·w`, eps 10⁻⁶ in the code, not in the config), no
///     `1 + w` anywhere; `q_norm`/`k_norm` act per head over 128;
///   - **a LoRA reaches** `layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}` (the DMD LoRA: rank 128, fp32, one alpha
///     per module, `diffusion_model.` prefix); one naming the time path (`time_embedding.*`, `shared_block_modulation`,
///     `video_head_modulation`) is refused (`MossImageDiTProgram.loraModules`). The distilled model's own time embedding
///     (`time_embedding.safetensors`, a different sampler) is not this route.
package enum MossImageDiT {
    package struct Config {
        package let dim, heads, headDim, hidden, layers, textDim, timeDim, channels, textBlocks: Int
        package let ropeBase: Double

        /// MOSS-Image-SFT as published (`transformer/config.json`); two text blocks.
        package static let published = Config(dim: 3072, heads: 24, headDim: 128, hidden: 8192, layers: 42, textDim: 4096, timeDim: 256,
                                              channels: 128, textBlocks: 2, ropeBase: 10000)
        /// Bands per axis (`head_dim / 8`): four axes, sin and cos, make `head_dim`.
        package var bands: Int { headDim / 8 }
    }
}

// MARK: - the scheduler

/// **The sigmas of `MOSSImageScheduler`** — Euler flow matching on a grid the model was trained on.
///
///     grid(count, shift)  = shift·v / (1 + (shift − 1)·v),  v = linspace(1, shift/(N + shift − 1), count)   fp32
///     σ                   = grid(steps, 5), each σ moved to the nearest value of grid(1000, 1) reversed
///                           (`searchsorted`, then the nearer of the two neighbours, ties down), then
///                           duplicates removed — **fewer σ than steps** when the schedule is dense
///
/// What a port by analogy would miss: the **rounding to the training grid** (a shifted `linspace` alone gives
/// 0.9679 where the reference has 0.968), the dedup (250 steps give 246 evaluations, 1000 give 615), the
/// shift applied **once** (the scheduler resets its own to 1 before `set_timesteps`), `training_shift` 1, and
/// timesteps = `σ·1000` in fp32 (463.00003 for 0.463), the last σ being followed by 0.
package enum MossImageSchedule {
    package static let trainTimesteps = 1000

    /// `torch.linspace(start, end, count)` in fp32 as torch computes it on this machine: from the start by `step`
    /// for the first half, from the end by `step` backwards for the rest — each value **one fused
    /// multiply-add** (`start + step·i`), which a separate product and sum misses on 239 of the training
    /// grid's 1000 values.
    static func linspace(_ start: Float, _ end: Float, _ count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Float(count - 1), halfway = count / 2
        return (0..<count).map {
            $0 < halfway ? start.addingProduct(step, Float($0)) : end.addingProduct(-step, Float(count - $0 - 1))
        }
    }

    static func grid(_ count: Int, shift: Double, total: Int = trainTimesteps) -> [Float] {
        let lower = Float(shift / (Double(total) + shift - 1))
        let s = Float(shift)
        return linspace(1, lower, count).map { s * $0 / (1 + (s - 1) * $0) }
    }

    /// The grid the model was trained on, ascending: `0.001 … 1`.
    package static var trainingGrid: [Float] { Array(grid(trainTimesteps, shift: 1).reversed()) }

    /// The σ the scheduler walks for `steps`, **followed by the final 0**. `shift` 5 and `trainingShift` 1
    /// are `scheduler_config.json`'s.
    package static func sigmas(steps: Int, shift: Double = 5, trainingShift: Double = 1) -> [Float] {
        precondition(steps >= 1 && shift > 0 && trainingShift > 0, "steps and shifts must be positive")
        let total = trainTimesteps
        let wanted = grid(steps, shift: shift, total: total)
        let training = Array(grid(total, shift: trainingShift, total: total).reversed())   // ascending
        var snapped: [Float] = []
        for sigma in wanted {
            // searchsorted, left: the first index whose value is not below σ.
            var low = 0, high = training.count
            while low < high {
                let mid = (low + high) / 2
                if training[mid] < sigma { low = mid + 1 } else { high = mid }
            }
            let upper = min(low, total - 1), lower = max(upper - 1, 0)
            let nearest = abs(sigma - training[lower]) <= abs(training[upper] - sigma) ? lower : upper
            let value = training[nearest]
            if snapped.last != value { snapped.append(value) }
        }
        return snapped + [0]
    }

    /// What the transformer is told at each evaluation: `σ·1000`, fp32 (the scheduler's `timesteps`).
    package static func timesteps(_ sigmas: [Float]) -> [Float] { sigmas.dropLast().map(timestep) }

    /// What the transformer is told at σ, the render's (`MossImageRender.Denoiser.derivative`): one function, so that
    /// the trajectory's identity on `timesteps` judges it.
    package static func timestep(_ sigma: Float) -> Float { sigma * Float(trainTimesteps) }
}

/// **The prompt side of MOSS-Image's encoder**: the chat template and the cut. Pure functions.
package enum Qwen35Prompt {
    /// `MOSSImagePipeline.encode_prompt`'s default `max_sequence_length`, and its bound (`1 ≤ n ≤ 4096`).
    package static let maxTokens = 4096

    /// `apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=True)`
    /// of the published `chat_template.jinja`: no system turn, and the generation prompt opens the thinking block.
    package static func template(_ prompt: String) -> String {
        "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n"
    }

    /// The identifiers the encoder reads: the template, tokenized, cut to `maxLength` from the END
    /// (`truncation=True`, right side) — the reference's behaviour, reproduced, not corrected.
    package static func ids(_ prompt: String, tokenizer: Tokenizer, maxLength: Int = maxTokens) -> [Int] {
        Array(tokenizer.encode(template(prompt)).prefix(maxLength))
    }

    /// `ids` for a render: the whole prompt in its template, **refused past `Request.maximumPromptTokens`**, never
    /// truncated — the publisher's bound (4096) is beyond the product's; `prompt`, the prompt's own tokens. `ids` keeps
    /// the reference's cut, for the checks.
    package static func admitted(_ prompt: String, tokenizer: Tokenizer) throws -> (ids: [Int], prompt: Int) {
        let ids = tokenizer.encode(template(prompt))
        let tokens = ids.count - tokenizer.encode(template("")).count
        try Request.admitPrompt(tokens: tokens, max: Request.maximumPromptTokens)
        return (ids, tokens)
    }
}
