import Foundation

/// The Qwen-Image-2.1-Turbo DiT — `QwenImage21Transformer2DModel`, 7.1 G parameters. The Turbo's checkpoint has the
/// base's architecture to the name, shape and dtype, and other values.
///
///     Qwen3-VL last layer [T, 4096] (image slots included) ─ text rows only ─ txt_in ─┐
///     reference latents [64, hᵢ, wᵢ] ─ tokens ─ img_in ──────────────────────────────┤ prefix P
///     target latent [64, h, w] ─ tokens ─ img_in ────────────────────────────────────┐│ target S
///                                                                                     ▼▼
///       joint sequence: text, each reference's h·w tokens AT ITS SLOTS, the target last
///                                                                                     │
///     t ─ sinusoid 256 [cos | sin] ─ MLP ─ temb(t) ─┬─ SiLU ─ modulation ──▶ 32 layers ─▶ target
///     0 ─ sinusoid 256 ─ MLP ─ temb(0) ─────────────┘   (conditions read the t = 0 row)
///                                                   └─ SiLU ─ norm_out.linear ─ final scale
///
/// **One layer** — a single sequence, a single set of weights, LayerNorms without affine:
///
///     x += tanh(g₁) · Wo( SDPA_blockcausal(RoPE(RMS(q)), RoPE(RMS(k)), v) )   on LN(x)·(1 + s₁)
///     x += tanh(g₂) · W_out( SiLU(W_gate n) ⊙ W_proj n )                       n = LN(x)·(1 + s₂)
///
/// What sets it apart from the other families, and what a port by analogy would miss:
///
///   - **block-causal attention** (`(q ≥ kv) ∨ same image`): text is strictly causal, each image block bidirectional
///     in itself, and everything sees what precedes it. It is done as the reference's exact processor does
///     (`QwenImage21AttnProcessor`): one SDPA per prefix segment on the keys `[0, end)`, a causal triangle for a text
///     segment, one SDPA for the target on all the keys. Two **adjacent** references stay two blocks;
///   - **the conditions are modulated by `t = 0`** (`causal_condition`, `_select_modulation_rows`): the modulation is
///     computed for two timesteps, and the text and the references take the second row — in every layer AND in
///     `norm_out`. They never see the target either: their rows do not depend on the step;
///   - the modulation is **shared by the 32 layers** and has **no shift**: `[scale₁, gate₁, scale₂, gate₂]`, gates
///     through **tanh**; `norm_out` is scale only (`linear` → d, not 2d);
///   - the image tokens are **substituted into the encoder's sequence**: each Qwen3-VL image slot stands for a 2×2
///     group, so each run of slots becomes `4·slots` rows in which the latent goes **row-major** (`_pack_latents` is a
///     plain flatten). The encoder rows at the slots are overwritten in the reference: here only the text rows go
///     through `txt_in`. The target's slots are appended by the pipeline;
///   - the sinusoid is **`[cos | sin]`** (cos first), over 256 channels, on `1000 · t`;
///   - `txt_in` is a zero-centred RMSNorm (`1 + w`, in f32) → Linear → **GELU tanh** → Linear;
///   - the RoPE has three axes `(16, 56, 56)`, θ = 10 000, adjacent pairs, **computed in float32** (`torch.outer` of
///     the positions and `1 / θ^(2i/d)`, then `torch.polar`): text at `(p, p, p)` with `p` advancing token by token; an
///     image freezes the frame axis at the current `p`, lays out `(y, x)` **centred on zero**
///     (`y ∈ [−(h − h/2), h/2)`), then advances `p` by `max(h, w)`;
///   - no bias anywhere; `img_in` and `proj_out` work at 64 channels, `patch_size` 1;
///   - **Comfy-Org's files fuse the SwiGLU's two inputs**: `img_mlp.gate_up` is `[gate_layer; proj]` by rows
///     (`gate, up = gate_up(x).chunk(2)`), every other name diffusers'.
package enum QwenImage21DiT {
    package struct Config: Sendable {
        package let dim, heads, headDim, hidden, layers, channels, context, timeDim: Int
        package let axes: [Int]
        package let theta, eps: Double

        /// **The published model's** (`transformer/config.json` of `Qwen/Qwen-Image-2.1-Turbo`): Comfy-Org's single
        /// files carry no configuration. θ is written in the model (`QwenImage21Rope(theta=10000, …)`).
        package static let published = Config(dim: 4096, heads: 32, headDim: 128, hidden: 12288, layers: 32, channels: 64,
                                              context: 4096, timeDim: 256, axes: [16, 56, 56], theta: 10_000, eps: 1e-6)
    }

    /// **The joint sequence**: the prefix (runs of text, reference blocks, in the encoder's order) and the target last.
    package struct Sequence: Sendable, Equatable {
        package enum Segment: Sendable, Equatable {
            case text(Int)
            /// A reference's latent grid.
            case image(height: Int, width: Int)

            package var rows: Int {
                switch self {
                case .text(let n): return n
                case .image(let h, let w): return h * w
                }
            }
        }
        package let prefix: [Segment]
        package let height: Int
        package let width: Int

        /// A generation: the text, then the target.
        package init(text: Int, height: Int, width: Int) {
            self.init(prefix: [.text(text)], height: height, width: width)
        }

        package init(prefix: [Segment], height: Int, width: Int) {
            self.prefix = prefix.filter { $0.rows > 0 }
            self.height = height
            self.width = width
        }

        /// **From the encoder's rows** (`image_pad_mask`, the system turn dropped) and each reference's latent grid:
        /// a slot stands for a 2×2 group, so reference `i` takes `hᵢ·wᵢ / 4` slots. The blocks come from the grids,
        /// not from the runs of slots: two references side by side stay two blocks (`build_token_metadata`).
        package init(slots: [Bool], references: [(height: Int, width: Int)], height: Int, width: Int) throws {
            var prefix: [Segment] = [], text = 0, index = 0, next = 0
            while index < slots.count {
                guard slots[index] else { text += 1; index += 1; continue }
                guard next < references.count else { throw ProgramError("an image slot without its reference") }
                let r = references[next]
                next += 1
                let count = r.height * r.width / 4
                guard index + count <= slots.count, slots[index..<(index + count)].allSatisfy({ $0 }) else {
                    throw ProgramError("reference \(next): \(count) slots expected at row \(index)")
                }
                if text > 0 { prefix.append(.text(text)); text = 0 }
                prefix.append(.image(height: r.height, width: r.width))
                index += count
            }
            guard next == references.count else { throw ProgramError("\(references.count) references for \(next) runs of slots") }
            if text > 0 { prefix.append(.text(text)) }
            self.init(prefix: prefix, height: height, width: width)
        }

        package var textRows: Int { prefix.reduce(0) { if case .text(let n) = $1 { return $0 + n } else { return $0 } } }
        package var referenceRows: Int { prefixRows - textRows }
        package var prefixRows: Int { prefix.reduce(0) { $0 + $1.rows } }
        package var targetRows: Int { height * width }
        package var rows: Int { prefixRows + targetRows }

        /// The prefix's segments as the attention reads them: `[start, end)` and whether it is text.
        package var segments: [(start: Int, end: Int, isText: Bool)] {
            var out: [(Int, Int, Bool)] = [], start = 0
            for s in prefix {
                if case .text = s { out.append((start, start + s.rows, true)) } else { out.append((start, start + s.rows, false)) }
                start += s.rows
            }
            return out
        }

        /// **`QwenImage21Rope.forward`'s positions**, `(frame, height, width)` per row: text counts up on the three
        /// axes; an image block freezes the frame at the position reached, lays its rows and columns centred on
        /// zero, then advances the position by `max(h, w)`.
        package var positions: [(Int, Int, Int)] {
            var out: [(Int, Int, Int)] = [], position = 0
            func block(_ h: Int, _ w: Int) {
                for y in -(h - h / 2)..<(h / 2) { for x in -(w - w / 2)..<(w / 2) { out.append((position, y, x)) } }
                position += max(h, w)
            }
            for s in prefix {
                switch s {
                case .text(let n):
                    for _ in 0..<n { out.append((position, position, position)); position += 1 }
                case .image(let h, let w): block(h, w)
                }
            }
            block(height, width)
            return out
        }
    }

    /// **The Turbo's schedule**: `sample_sigmas` of its `model_index.json` (`QwenImage21Published`), as
    /// `set_timesteps` keeps them — each a Python double made float32, the shift the identity, then a zero σ. The
    /// same at every size, in a generation and in an edit; the last step jumps from 0.4146 to 0.
    package static var sigmas: [Float] { QwenImage21Published.sampleSigmas.map { Float($0) } + [0] }

    /// The step count the Turbo publishes a schedule for.
    package static var defaultSteps: Int { QwenImage21Published.sampleSigmas.count }
}

/// **The prompt side**: template, identifiers, `_drop_idx`. Pure functions.
///
///     <|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n          ← dropped after the encoder
///     <|im_start|>user\n[<image1><|vision_start|><|image_pad|>…<|vision_end|> …]{prompt}<|im_end|>\n<|im_start|>assistant\n
///
/// The template is a raw string (not `apply_chat_template`); `<image1>` is literal text before `<|vision_start|>`, the
/// following pictures are each preceded by a space, and the prompt is glued after the last `<|vision_end|>`.
package enum QwenImage21Text {
    package static let system = "Comprehend and analyze the provided prompt."

    /// `prompt_template_t2i` / `prompt_template_ti2i` of `QwenImage21Pipeline`, ONE `<|image_pad|>` per image (the
    /// processor expands them).
    package static func template(prompt: String, images: Int = 0) -> String {
        // "Qwen has no bos token, so an empty string leaves the encoder with nothing to read."
        let text = prompt.isEmpty ? " " : prompt
        let slots = (0..<images).map { "<image\($0 + 1)><|vision_start|><|image_pad|><|vision_end|>" }.joined(separator: " ")
        return systemTurn + "<|im_start|>user\n\(slots)\(text)<|im_end|>\n<|im_start|>assistant\n"
    }

    /// The system turn as the template writes it — its length in tokens is `_drop_idx` (14).
    package static var systemTurn: String { "<|im_start|>system\n\(system)<|im_end|>\n" }

    /// **The identifiers the encoder reads**, the rows of them that are a picture's, and how many of them are the system
    /// turn's. The whole prompt, **refused past `Request.maximumPromptTokens`**, never truncated — the reference does not
    /// bound it, so the bound is the product's. `pictures`: each reference's slots (a quarter of its latent cells), in
    /// order — the processor repeats the template's one `<|image_pad|>` a picture that many times. A prompt that spells
    /// `<|image_pad|>`, the place of a reference image, has no meaning for the reference either (Qwen3-VL raises in an
    /// edit, the DiT on the slot no latent fills in a generation): refused.
    package static func identifiers(_ prompt: String, pictures: [Int] = [], tokenizer: Tokenizer) throws
        -> (ids: [Int], slots: [Bool], dropped: Int, prompt: Int) {
        let own = tokenizer.encode(prompt)
        if own.contains(try Qwen3VLChat.padIdentifier(tokenizer)) { throw EngineError.promptReservedText(text: Qwen3VLChat.imagePad) }
        try Request.admitPrompt(tokens: own.count, max: Request.maximumPromptTokens)
        let (ids, slots) = try Qwen3VLChat.expanded(template(prompt: prompt, images: pictures.count), rows: pictures, tokenizer: tokenizer)
        return (ids, slots, tokenizer.encode(systemTurn).count, own.count)
    }
}
