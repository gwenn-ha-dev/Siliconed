import Foundation
import Metal
import CoreAI
import SiliCore

/// **MOSS-Image-SFT's DiT**: a latent to its velocity, as `MOSSImageTransformer.forward` of the
/// publisher's `pipeline_moss_image.py` runs it under `torch_dtype=bfloat16`. The structure and what sets the model apart are
/// `MossImageDiT` (`MossImage.swift`); this file keeps the shapes and the CPU tables the engine's DiT
/// (`MossDiT`, `MossImageTransformer.swift`) reads.
///
///     latent [128,H,W] bf16, sin(t·f) | cos(t·f) [256] f32 (fp64 on the CPU), text [T,4096] bf16, rope [S,128] f32 (CPU)
///       time:    Linear(256,d)+b ─ SiLU ─ Linear(d,d)+b ─▶ time ─ Linear(d,6d)+b ─▶ 6 × [d] ─ ┐ shared by the 42 blocks
///                                                          └ Linear(d,2d)+b ─▶ head [shift | scale]
///       image:   tokens [H·W,128] ─ Linear+b ─ video [H·W,d]
///       text:    Linear(4096,d)+b ─ 2 × text block (rope rows of the text) ─ text [T,d]
///       x = image rows, then text rows [S = H·W + T, d] ─ 42 × block(x, modulation) ─ image rows ─ head ─ velocity [128,H,W]
///
/// **The reference rounds to bf16 after every torch operation; the program does not force those roundings.** Core AI's GPU
/// fuses a chain of elementwise operations and carries f32 through it: a value is rounded where a chain ends (a product, a
/// norm, the attention), not after each operation. What is judged is the distance to the fp64 witness, not torch's bits.
/// Forcing each rounding (a pad by one element and the slice that drops it, 1 450 per evaluation) cost 3.90 s a step against
/// 2.59 s at 1024² (M5 Pro) for the same judgment of `dit` (23 boundaries of 23 at 512² and at 1024²), and a
/// first velocity of the trajectory at 1.0e-1 of the witness against 4.7e-2 without. The reference's arithmetic, as written here:
///
///   - `_RMSNorm.forward`: `y = x.float() * rsqrt(x.float().square().mean(-1) + 1e-6)`; `y.to(x.dtype) * self.weight`;
///   - `_Block.forward`: `norm1(x) * (1 + scale_a) + shift_a`, the sublayer, `× gate`, `x + …`. q and k are
///     `q_norm(proj_q(y).unflatten(heads))` per head over 128;
///   - `_rotate`: `x * cos + cat(-b, a) * sin` on bf16 values, where other references rotate in f32. The table is
///     `[sin(64) | cos(64)]` in f32 (`_RoPE.forward`, fp32 bands and phases), then `.to(video.dtype)`;
///   - `_situ`: `(4 · tanh(g/4) · σ(g)) · (25 · tanh(h/25))`;
///   - the time path: the sinusoid in fp64 then `.float()`, `.to(bf16)` for the MLP (`time_embedding(embedding.to(weight.dtype))`),
///     `.float()` and `.to(bf16)` back for the modulations (the identity on a bf16 value): all bf16, SiLU between;
///   - **the LoRA is kept apart** (the reference's demo, `apart_hook` of the oracle): `y = base(x) + (alpha/rank) · up(down(x))`.
///     Merged into bf16 weights it loses 92 % of its delta (`witness-moss-bf16-512`: 0.22 from the f32 merge, 0.019 kept apart).
///     The demo brings the factors and `down(x)` to bf16; here the low-rank path stays f32 up to its cast to bf16 (see
///     `product`): the published factors are f32.
///
/// The sequence is image rows then text rows as the reference's; no mask, so every position sees every other.
package struct MossImageDiTProgram {
    package let config: MossImageDiT.Config
    package let latentHeight, latentWidth, textLength: Int

    /// Every tensor the DiT reads, by the name its published file gives it and its shape there (`[out, in]` for a
    /// product, the Compact's codes the same): what a file is recognized against.
    package static func tensors() -> [String: [Int]] {
        var all: [String: [Int]] = [:]
        for (name, shape) in products() { all[name + ".weight"] = shape }
        let c = MossImageDiT.Config.published, d = c.dim
        for b in ["time_embedding.0", "time_embedding.2", "shared_block_modulation", "video_head_modulation", "video_embedder", "text_embedder.proj"] {
            all[b + ".bias"] = [all[b + ".weight"]![0]]
        }
        for i in 0..<c.textBlocks {
            let p = "text_embedder.blocks.\(i)."
            for m in ["qkv", "proj", "ffn_up", "ffn_gate", "ffn_down"] { all[p + m + ".bias"] = [all[p + m + ".weight"]![0]] }
            for n in ["norm1", "norm2"] { all[p + n + ".weight"] = [d] }
            for n in ["q_norm", "k_norm"] { all[p + n + ".weight"] = [c.headDim] }
        }
        for i in 0..<c.layers {
            for n in ["norm1", "norm2"] { all["layers.\(i).\(n).weight"] = [d] }
            for n in ["q_norm", "k_norm"] { all["layers.\(i).\(n).weight"] = [c.headDim] }
        }
        all["final_norm_video.weight"] = [d]
        return all
    }

    /// Every product of the DiT by its published name (without `.weight`), `[out, in]`.
    static func products() -> [String: [Int]] {
        let c = MossImageDiT.Config.published, d = c.dim, h = c.hidden
        var all: [String: [Int]] = [
            "time_embedding.0": [d, c.timeDim], "time_embedding.2": [d, d], "shared_block_modulation": [6 * d, d],
            "video_head_modulation": [2 * d, d], "video_embedder": [d, c.channels], "text_embedder.proj": [d, c.textDim],
            "final_linear_video": [c.channels, d]]
        for i in 0..<c.textBlocks {
            let p = "text_embedder.blocks.\(i)."
            all[p + "qkv"] = [3 * d, d]; all[p + "proj"] = [d, d]; all[p + "ffn_up"] = [h, d]; all[p + "ffn_gate"] = [h, d]; all[p + "ffn_down"] = [d, h]
        }
        for i in 0..<c.layers {
            let p = "layers.\(i).proj."
            all[p + "qkv"] = [3 * d, d]; all[p + "o"] = [d, d]; all[p + "ffn_up_gate"] = [2 * h, d]; all[p + "ffn_down"] = [d, h]
        }
        return all
    }

    /// **Every module a LoRA may adapt**: the 42 blocks' products, by their published names, and each third of a block's
    /// `qkv` under its own (`proj.q`, `proj.k`, `proj.v`: the DMD LoRA's names) landing on its rows. The products outside
    /// the blocks (the time path, the embedders, the text blocks, the head) take no low-rank path: a file adapting one is
    /// refused rather than half applied.
    package static func loraModules() -> [String: LoRAStack.Shape] {
        let c = MossImageDiT.Config.published, d = c.dim, h = c.hidden
        var all: [String: LoRAStack.Shape] = [:]
        for i in 0..<c.layers {
            let p = "layers.\(i).proj."
            all[p + "qkv"] = .init(out: 3 * d, inn: d)
            for (j, t) in ["q", "k", "v"].enumerated() { all[p + t] = .init(out: d, inn: d, product: p + "qkv", row: j * d) }
            all[p + "o"] = .init(out: d, inn: d)
            all[p + "ffn_up_gate"] = .init(out: 2 * h, inn: d); all[p + "ffn_down"] = .init(out: d, inn: h)
        }
        return all
    }

    package init(config: MossImageDiT.Config = .published, latentHeight: Int, latentWidth: Int, textLength: Int) {
        self.config = config
        (self.latentHeight, self.latentWidth, self.textLength) = (latentHeight, latentWidth, textLength)
    }

    package var tokens: Int { latentHeight * latentWidth }
    package var sequence: Int { tokens + textLength }

    /// The boundaries of the oracle's `probe_dit`, beside `model_out`.
    package static let tapNames = ["time_embedding_out", "shared_modulation", "head_modulation", "video_embed", "rope", "text_proj_out",
                                   "text_block0_out", "text_block1_out", "block0_q_out", "block0_k_out", "block0_v_out", "block0_o_out",
                                   "block0_ffn_up_gate_out", "block0_ffn_down_out", "block0_q_norm_out", "block0_out", "block1_out",
                                   "block13_out", "block27_out", "block41_out", "final_norm_out"]

    // MARK: - The CPU's tables

    /// `sin(t·f) | cos(t·f)`… the reference's `cat((phase.cos(), phase.sin()))`: `[cos | sin]` of `t · 10⁴^(−j/128)` in fp64, as f32.
    package static func timeSinusoid(_ t: Float, freqDim: Int = 256) -> [Float] {
        let half = freqDim / 2
        let phases = (0..<half).map { Double(t) * pow(10000, -Double($0) / Double(half)) }
        return phases.map { cosSin($0).cos } + phases.map { cosSin($0).sin }
    }

    /// `_RoPE.forward`: the f32 table `[S, 128] = [sin(64) | cos(64)]`, image rows `(0, T, h − (H−1)/2, w − (W−1)/2)` then the text
    /// rows `(0, i, 0, 0)`; entry `axis · 16 + band`, `bands = 1 / 10⁴^(i/16)` in f32.
    package func rope() -> [Float] {
        let n = config.bands, base = Float(config.ropeBase)
        let bands = (0..<n).map { 1 / powf(base, Float($0) / Float(n)) }
        var rows: [[Float]] = []
        for h in 0..<latentHeight {
            for w in 0..<latentWidth {
                rows.append([0, Float(textLength), Float(h) - Float(latentHeight - 1) / 2, Float(w) - Float(latentWidth - 1) / 2])
            }
        }
        for i in 0..<textLength { rows.append([0, Float(i), 0, 0]) }
        var table: [Float] = []
        table.reserveCapacity(rows.count * config.headDim)
        for row in rows {
            let phases = row.flatMap { c in bands.map { c * $0 } }
            table += phases.map { cosSin(Double($0)).sin } + phases.map { cosSin(Double($0)).cos }
        }
        return table
    }
}
