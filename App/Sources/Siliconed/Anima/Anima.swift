import Foundation

/// The Anima Turbo DiT — Cosmos-Predict2-2B lineage —, from the latent to the predicted velocity (`AnimaCosmosDiT`, in bf16).
///
///     latent [16, H, W] + padding mask (zeros) ─ patchify (SLOW channel) ─ patch_embed ───────┐
///                                                                                            │
///     σ ─ sinusoid 2048 ──┬─ linear_1 · SiLU · linear_2 ─ temb [6144] ──────────────┐        │
///                         └─ RMSNorm ─ embedded [2048] ─ SiLU ──────────────────────┤        │
///                                                                                   ▼        ▼
///     text [512, 1024] ─────────────────────────────────────────────────▶ blocks ×28 ─ norm_out ─ proj_out ─ unpatchify
///
/// One block, three sublayers, **each with its own modulation**:
///
///     shift, scale, gate = linear_2ᵢ(linear_1ᵢ(SiLU(embedded))) + temb     (i = 1, 2, 3)
///     x += gate₁ · attn1( LN(x)·(1+scale₁) + shift₁ )     self-attention, QK-Norm, RoPE
///     x += gate₂ · attn2( LN(x)·(1+scale₂) + shift₂ , text )   cross, QK-Norm, NO RoPE
///     x += gate₃ · W₂ GELU( W₁ (LN(x)·(1+scale₃) + shift₃) )
///
/// What sets it apart from the usual adaLN block, and what a port by analogy would miss:
///
///   - **a shift** in the modulation, and **no tanh** on the gate;
///   - a **low-rank** modulation (`adaln_lora_dim = 256`) *plus* `temb`, the same for the
///     three sublayers;
///   - an **exact GELU** FFN (`erf`), ungated — not a SwiGLU;
///   - patchify lays out the **slowest channel** (`c·4 + ph·2 + pw`), and it carries a seven-
///     teenth channel, the padding mask, zero for a whole image;
///   - Cosmos RoPE is *split*: a pair is a head's two halves (`AnimaFunction.module`'s `rope`);
///   - that table is **bounded**: `max(max_size // patch_size)` = 128 positions per side, 2048 px —
///     the reference cannot compute past it, so `Format.maximumSide(for: .anima)` refuses it
///     (`Config.ropePositions`), where the formula alone would extrapolate without a word;
///   - the **residuals climb to 2.5·10⁵** at block 13 (golden tensors): four times the fp16
///     ceiling, within bf16's range — nothing of the DiT runs in fp16.
package enum AnimaDiT {
    package struct Config {
        package let dim, heads, headDim, hidden, layers, channels, patch, adalnLoRA, textDim: Int
        package let ropeScale: [Double]
        /// **The positions the reference's RoPE table holds, per side**: `CosmosRotaryPosEmbed` builds
        /// `seq = arange(max(max_size // patch_size))` and slices `seq[:tokens]` on each axis — beyond it,
        /// the slice comes back short and the reference fails on the concatenation. Note the `max` over
        /// the three axes: with the published `max_size = (128, 240, 240)`, `patch_size = (1, 2, 2)` it is
        /// the *time* axis' 128 that bounds the image's sides, not 240 // 2 = 120.
        package let ropePositions: Int

        /// `max(max_size // patch_size)` (`CosmosRotaryPosEmbed.__init__` then `forward`).
        package static func ropePositions(maxSize: [Int], patchSize: [Int]) -> Int {
            zip(maxSize, patchSize).map { $0 / $1 }.max() ?? 0
        }
        /// The published configuration's (`max_size = [128, 240, 240]`, `patch_size = [1, 2, 2]`):
        /// 128 tokens, 2048 px on a side through the VAE ×8 and the patch ×2.
        package static let publishedRopePositions = ropePositions(maxSize: [128, 240, 240], patchSize: [1, 2, 2])

        /// **The published model's** (`transformer/config.json` of `circlestone-labs/Anima-Base-v1.0-Diffusers`): what a
        /// weights file of the family is read with — the single file carries no configuration.
        package static let published = Config(dim: 2048, heads: 16, headDim: 128, hidden: 8192, layers: 28, channels: 16, patch: 2,
                                              adalnLoRA: 256, textDim: 1024, ropeScale: [1, 4, 4], ropePositions: publishedRopePositions)
    }

}

// **Anima's σ schedule.** The Turbo's `FlowMatchEulerDiscreteScheduler` under `use_dynamic_shifting = false`, and nothing else of
// the model: the DiT itself is `AnimaCosmosDiT.swift`, its sampling loop `AnimaRender.swift`. What sets it apart from the other
// families' schedules is that it is computed **in float32, operation by operation**, as the reference's `set_timesteps`
// does — the same formula in double, rounded once, is one bit off on half of the σ of N = 1…50 (see `sigmas`).

extension AnimaDiT {
    /// The Anima Turbo schedule: `sigmas = linspace(1, 1/N, N)` shifted by `shift = 3`
    /// (`σ' = 3σ / (1 + 2σ)`), then a zero terminal σ — `FlowMatchEulerDiscreteScheduler` under
    /// `use_dynamic_shifting = false`, as the modular `anima` pipeline calls it.
    ///
    /// **In float32, operation by operation**, as `set_timesteps` computes it: NumPy's double ramp
    /// cast to float32 first, then `shift * s / (1 + (shift − 1) * s)` on a float32 array (NumPy 2 keeps
    /// a Python float weak). The same formula in double, rounded once, is one bit off on 673 of the
    /// 1 275 σ of N = 1…50 (not at N = 8). To the bit against diffusers (`SolverTests`).
    package static func sigmas(steps: Int, shift: Double = 3) -> [Float] {
        let k = Float(shift), k1 = Float(shift - 1)
        return NumPyRamp.linspace(1, 1 / Double(steps), steps).map { s -> Float in
            let x = Float(s)
            return k * x / (1 + k1 * x)
        } + [0]
    }
}
