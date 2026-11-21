import Foundation

/// **The Qwen-Image-2.1-Turbo DiT**: one evaluation over the joint sequence, from the text states,
/// the reference latents and the target latent to the velocity, as diffusers' `QwenImage21Transformer2DModel`
/// (`transformer_qwenimage21.py`, 0.41) runs it in bf16 without a K/V cache. The configuration and the sequence are
/// `QwenImage21DiT` (`QwenImage21.swift`). **Its weights are read as published** (Comfy-Org's single files).
///
///     text [Tt, 4096] ─ txt_in (zero-centred RMS · Linear · GELU tanh · Linear) ─┐
///     references [R, 64] ─ img_in ───────────────────────────────────────────────┤ joint [P + S, 4096], the target last
///     latent [64, h, w] ─ row-major tokens ─ img_in ─────────────────────────────┘
///     t_freq [256] (CPU: t a step, 0 a render) ─ linear_1 · SiLU · linear_2 = temb [d] ─ SiLU ─ modulation.1 ─ [4d]
///     cos, sin [P + S, 64] (CPU) ──▶ every layer's RoPE
///
///     layer:  n = LN(x) · (1 + s₁);  x += tanh(g₁) · to_out( SDPA_blockcausal(RoPE(RMS q), RoPE(RMS k), v) )
///             m = LN(x) · (1 + s₂);  x += tanh(g₂) · out( SiLU(gate) · up ),  [gate | up] = gate_up(m)
///     norm_out: LN(x) · (1 + linear(SiLU(temb)));  proj_out;  the target's rows
///
/// What a port by analogy would miss (read in `transformer_qwenimage21.py` and `pipeline_qwenimage21.py`):
///
///   - **Each row reads its own timestep's modulation**: the target the row of `t`, the prefix the row of `0`
///     (`_select_modulation_rows`), in the layers and in `norm_out`. The modulation being shared, `1 + s` and `tanh(g)`
///     are computed once for the 32 layers, a row each: the prefix's (0) once a render, the target's (t) once a step.
///   - **The attention is the reference's exact processor** (`QwenImage21AttnProcessor`): one SDPA per prefix segment on
///     the keys `[0, end)` — a text segment under `[ones | tril]` —, then the target on every key.
///   - **`diffusers.RMSNorm` rounds twice** on q and k (the normalized value to bf16, then × γ); **the zero-centred
///     RMSNorm of `txt_in` rounds once** (`x.float() · rrms · (w.float() + 1)`, then `.to(dtype)`).
///   - **RoPE is a complex product in f32** (`use_real=False`: `view_as_complex(x.float()) · freqs_cis`), rounded once.
///   - **The sinusoid is rounded to bf16** before `linear_1` (`timesteps_proj.to(dtype=hidden_states.dtype)`); `t` is
///     itself the pipeline's `(t.to(bf16) / 1000)` (`modelTime`).
///   - **No product's rounding is forced.** Core AI's GPU fuses the operation that follows a product onto its f32 sum,
///     where torch rounds the product to bf16 first; what is judged is the distance to the witness, not torch's bits. A
///     pad-and-slice barrier behind every product gave the same step within a run's drift (3.31 and 3.22 s at 1024²,
///     M5 Pro) and a `dit` further from the witness (`model_out` 1.373e-2 with, 9.298e-3 without).
///   - **The Compact's products** (ComfyUI's int8 `convrot`): the activation rotated by the Hadamard (256), once per
///     value, then int8 × int8 (`int8Linear`).
package struct QwenImage21Transformer {
    package let config: QwenImage21DiT.Config
    package let sequence: QwenImage21DiT.Sequence

    package init(config: QwenImage21DiT.Config = .published, sequence: QwenImage21DiT.Sequence) {
        self.config = config
        self.sequence = sequence
    }

    /// Every tensor the DiT reads, by the name its published file gives it: what a file is recognized against.
    package static func tensors() -> [String: [Int]] { QwenImage21Functions.tensors() }

    /// **Every module a LoRA may adapt**: each product of the DiT, by its published name.
    package static func loraModules() -> [String: LoRAStack.Shape] { QwenImage21Functions.loraModules() }

    // MARK: - The CPU's tables

    /// **`QwenImage21Rope`'s table over the joint sequence**, `[P + S, 64]`, in float32 as torch computes it:
    /// `1 / θ^(2i/d)` (`pow` and the division in f32), `position · freq` in f32 (`torch.outer`), then `cos` and `sin`
    /// (`torch.polar`). Axes `(16, 56, 56)`: 8, 28 and 28 complex frequencies.
    package func rope() -> (cos: [Float], sin: [Float]) {
        let freqs = config.axes.map { d in (0..<(d / 2)).map { 1 / powf(Float(config.theta), Float(2 * $0) / Float(d)) } }
        var cos: [Float] = [], sin: [Float] = []
        cos.reserveCapacity(sequence.rows * config.headDim / 2); sin.reserveCapacity(sequence.rows * config.headDim / 2)
        for (frame, y, x) in sequence.positions {
            for (axis, p) in [frame, y, x].enumerated() {
                for f in freqs[axis] {
                    let angle = Float(p) * f
                    let (c, s) = cosSin(Double(angle))
                    cos.append(c); sin.append(s)
                }
            }
        }
        return (cos, sin)
    }

    /// **`QwenImage21TemporalTimesteps`' sinusoid in float32 as torch computes it**, one row `[256]`: `[cos | sin]` of
    /// `(1000 · t) · exp(−ln(10⁴) · i / 128)`, each operation one float rounding.
    package static func timestepFrequencies(_ t: Float, config: QwenImage21DiT.Config = .published) -> [Float] {
        timestepSinusoid(1000 * t, half: config.timeDim / 2)
    }

    /// **What the DiT is told** (`pipeline_qwenimage21.py`): `t = σ · 1000` in f32 brought to the latents' bf16
    /// (`t.expand(…).to(latents.dtype)`), then `/ 1000` in bf16.
    package static func modelTime(sigma: Float) -> Float { BFloat.round(BFloat.round(sigma * 1000) / 1000) }

    // MARK: - The oracle's boundaries

    package enum Tap: String, CaseIterable, Sendable {
        case temb, modulation, txtIn = "txt_in_out", imgIn = "img_in_out", jointIn = "joint_in"
        case layer0 = "layer0_out", layer1 = "layer1_out", layerLast = "layer_last_out", normOut = "norm_out"
    }
}
