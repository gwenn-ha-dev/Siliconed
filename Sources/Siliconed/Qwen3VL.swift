import Accelerate
import Foundation
import Metal

/// **Qwen-Image-2.1's encoder: Qwen3-VL-8B**, text and images together — `Qwen3VLForConditionalGeneration`
/// as `QwenImage21Pipeline._get_qwen_prompt_embeds` drives it, in fp32.
///
///     prompt ─ hard-coded template ─ tokenizer ─ ids (each <|image_pad|> × gh·gw/4)
///                                                  │
///     image ─ Lanczos (PIL) ─ processor ─ pixel_values [N, 1536] (2×2 merge-block order)
///             │
///             ▼  vision tower, per image: patch_embed (a Linear) + learned positions (48², bilinear)
///             27 blocks d 1152, 16 heads × 72, 2D RoPE, attention within the image, GELU-tanh MLP
///             ├─ after blocks 8, 16, 24: deepstack merger k ─ [N/4, 4096] ─┐
///             └─ merger: LayerNorm(1152) → view 4 patches → MLP (exact GELU) ─ [N/4, 4096]
///                                                  │                       │
///     embed_tokens(ids), image rows overwritten ◀──┘                       │
///             ▼                                                            │
///     36 layers (`TextEncoder`), mRoPE (t, h, w) ── after layers 0, 1, 2: + deepstack k on the image rows
///             ▼
///     output of layer 35, NOT normalized ─ rows from `dropCount` on ─ prompt_embeds [T, 4096]
///                                                                     image_pad_mask [T]
///
/// **A vision block** (pre-norm, LayerNorm with bias, eps 1e-6, every Linear with a bias):
///
///     q, k, v = split₃(W_qkv · LN₁(x) + b)          16 heads × 72
///     x += W_o · SDPA(RoPE₂ᴰ(q), RoPE₂ᴰ(k), v) + b    bidirectional, one image at a time
///     x += W₂ · GELU_tanh(W₁ · LN₂(x) + b₁) + b₂     4304
///
/// The 36 language layers are the ones `TextEncoder` already runs for Z-Image (Qwen3, QK-Norm,
/// GQA — 32/8 here): it is reused as is, with three hooks — the image embeddings, the deepstack
/// additions, the 3D positions.
///
/// ## What sets it apart, and what a port by analogy would miss
///
///   - **The output is the last layer BEFORE the final RMSNorm.** transformers 5 returns the
///     normalized state as `hidden_states[-1]`; the pipeline hooks `norm` to return its input. The
///     map does not even carry `norm.weight`. Measured by the oracle: rms 19.0 un-normalized against
///     a third of that normalized.
///   - **The template is written by hand, not by `apply_chat_template`** (they tokenize
///     differently): images go first in the user turn, `<imageN>` is plain text, a SPACE separates
///     two images, none separates the last image from the prompt. An empty prompt becomes `" "`.
///   - **The first 14 tokens are dropped** (`_drop_idx`: the system turn, tokenized by the chat
///     template); everything after stays, the assistant header included.
///   - **Two different GELUs**: `gelu_pytorch_tanh` in the vision MLP, the EXACT `nn.GELU()` in
///     the four mergers.
///   - **The deepstack mergers normalize AFTER the 2×2 shuffle** (`use_postshuffle_norm`:
///     LayerNorm over 4608), the final merger BEFORE it (over 1152). Same module, a flag apart.
///   - **Deepstack is added to the OUTPUT of layers 0, 1, 2** (`_deepstack_process`), not to their
///     input, and only on the `<|image_pad|>` rows.
///   - **mRoPE is interleaved**: frequency `j` reads `h` if `j ≡ 1 (mod 3)`, `w` if `j ≡ 2`
///     (both below 60), `t` otherwise. Image tokens sit at `(s, s + row, s + col)` with `s` the
///     position after the preceding text; the text after an image resumes at
///     `s + max(gh, gw)/2`, not at `s + gh·gw/4` — positions are shorter than the sequence.
///   - **The vision RoPE is 2D over a 72-wide head**: 18 frequencies `θ = 10⁴` over 36 dims, the
///     first 18 angles read the row, the next 18 the column, and the 36 are duplicated
///     (`rotate_half` pairs `j` with `j + 36`). The forge permutes `qkv`'s q and k thirds per head
///     so that `Ops.rope`'s pairs `(2p, 2p+1)` compute it.
///   - **Learned positions are interpolated** from a 48 × 48 table: `align_corners=True` bilinear,
///     source `i · 47 / (g − 1)` in fp32, 4 taps clamped to the border, in the 2×2 merge-block
///     order of the patches — not raster order.
///   - **`patch_embed` is a `Conv3d` whose stride equals its kernel** `(2, 16, 16)`: a Linear
///     1536 → 1152 on the patch flattened `(c, t, y, x)`. An image is a two-frame video of itself:
///     the processor DUPLICATES each patch in time.
package final class Qwen3VLEncoder {
    /// What the DiT and the pipeline receive.
    package struct Output {
        /// `prompt_embeds`, `[rows, hidden]` row-major: the output of the last layer, not normalized,
        /// from token `dropCount` on.
        package let embeddings: [Float]
        /// `image_pad_mask`, `[rows]`: `true` on the `<|image_pad|>` rows — the condition image slots.
        package let imagePadMask: [Bool]
        package let rows: Int
        package let hidden: Int
    }

    package let artifact: Artifact
    package let tokenizer: Tokenizer
    package let tokens: Qwen3VLPrompt.Tokens
    package let vision: Qwen3VLVision.Config
    private let freezeCut: Bool

    /// The probes, on request (checks): those of the language model (`layerN_in`, `layerN_out`,
    /// `embed_out`) and of the vision tower (`vit_*`, see `Qwen3VLVision`).
    package var recordedNames: Set<String>?
    package private(set) var boundaries: [String: [Float]] = [:]
    package var cancellation: Cancellation?
    /// Seconds, by phase — `vision`, `language`.
    package private(set) var timings: [String: Double] = [:]

    package init(artifact: Artifact, tokenizer: Tokenizer,
                 freezeCut: Bool = EngineSettings.effective.frozenCut) throws {
        self.artifact = artifact
        self.tokenizer = tokenizer
        self.tokens = try Qwen3VLPrompt.Tokens(header: artifact.header, tokenizer: tokenizer)
        self.vision = try Qwen3VLVision.Config(header: artifact.header)
        self.freezeCut = freezeCut
        guard artifact.header["final_norm"] as? Bool != true else {
            throw Artifact.Failure.badHeader("Qwen3-VL: the output is taken BEFORE the final norm")
        }
    }

    /// The prompt and its condition images (already brought to their size by the pipeline, see
    /// `Qwen3VLImages`) → `prompt_embeds` and `image_pad_mask`.
    package func encode(prompt: String, images: [Qwen3VLImages.Pixels] = []) throws -> Output {
        boundaries = [:]
        timings = [:]
        let prepared = try Qwen3VLPrompt.prepare(prompt: prompt, grids: images.map { ($0.gridHeight, $0.gridWidth) },
                                             tokenizer: tokenizer, tokens: tokens, merge: vision.merge)
        let sequence = prepared.ids.count

        // ── the vision tower, image by image (its attention never crosses two images) ───────
        var features: [Qwen3VLVision.Features] = []
        if !images.isEmpty {
            let started = Date()
            let tower = try Qwen3VLVision(artifact: artifact, config: vision,
                                          maximumPatches: images.map(\.patches).max()!, freezeCut: freezeCut)
            tower.cancellation = cancellation
            for (index, image) in images.enumerated() {
                tower.recordedNames = recordedNames.map { names in
                    Set(names.filter { $0.hasSuffix("_\(index)") }.map { String($0.dropLast("_\(index)".count)) })
                }
                features.append(try tower.encode(image))
                for (name, value) in tower.boundaries { boundaries["\(name)_\(index)"] = value }
            }
            timings["vision"] = Date().timeIntervalSince(started)
        }
        let imageRows = prepared.ids.indices.filter { prepared.ids[$0] == tokens.image }
        let featureRows = features.reduce(0) { $0 + $1.tokens }
        guard imageRows.count == featureRows else {
            throw Artifact.Failure.misuse("\(imageRows.count) <|image_pad|> tokens for \(featureRows) image features")
        }

        // ── the language model ──────────────────────────────────────────────────────────────
        let started = Date()
        let encoder = try TextEncoder(artifact: artifact, sequence: sequence, freezeCut: freezeCut)
        let h = encoder.config.hidden
        guard h == vision.output else {
            throw Artifact.Failure.badHeader("vision output \(vision.output) for a language model of \(h)")
        }
        encoder.cancellation = cancellation
        encoder.setPositions(prepared.positions)
        if let names = recordedNames {
            encoder.recordBoundaries = true
            encoder.recordedNames = names
        }
        if !features.isEmpty {
            // `masked_scatter`: the image features, concatenated in image order, fill the
            // `<|image_pad|>` rows in sequence order.
            func scatter(_ rows: [Int], _ select: (Qwen3VLVision.Features) -> [Float],
                         into x: UnsafeMutablePointer<Float>, add: Bool) {
                var next = 0
                for feature in features {
                    let values = select(feature)
                    values.withUnsafeBufferPointer { v in
                        for r in 0..<feature.tokens {
                            let destination = x + rows[next] * h, source = v.baseAddress! + r * h
                            if add { vDSP_vadd(destination, 1, source, 1, destination, 1, vDSP_Length(h)) }
                            else { destination.update(from: source, count: h) }
                            next += 1
                        }
                    }
                }
            }
            encoder.embeddingsHook = { x in scatter(imageRows, \.embeddings, into: x, add: false) }
            let levels = features[0].deepstack.count
            encoder.layerOutputHook = { layer, x in
                guard layer < levels else { return }
                scatter(imageRows, { $0.deepstack[layer] }, into: x, add: true)
            }
        }
        let all = try encoder.encode(ids: prepared.ids, realTokens: sequence)
        timings["language"] = Date().timeIntervalSince(started)
        for (name, value) in encoder.boundaries { boundaries[name] = value }

        let rows = sequence - prepared.dropCount
        let embeddings = Array(UnsafeBufferPointer(start: all.baseAddress! + prepared.dropCount * h, count: rows * h))
        let mask = prepared.ids[prepared.dropCount...].map { $0 == tokens.image }
        return Output(embeddings: embeddings, imagePadMask: mask, rows: rows, hidden: h)
    }
}

/// **The prompt side**: template, identifiers, `_drop_idx`, 3D positions. Pure functions.
package enum Qwen3VLPrompt {
    package static let system = "Comprehend and analyze the provided prompt."

    /// The special identifiers, read from the map's header and checked against the tokenizer:
    /// a tokenizer from another revision would put the image features on the wrong rows.
    package struct Tokens {
        package let image: Int, visionStart: Int, visionEnd: Int

        package init(image: Int, visionStart: Int, visionEnd: Int) {
            self.image = image; self.visionStart = visionStart; self.visionEnd = visionEnd
        }

        init(header: [String: Any], tokenizer: Tokenizer) throws {
            guard let t = header["tokens"] as? [String: Any], let image = t["image_token_id"] as? Int,
                  let start = t["vision_start_token_id"] as? Int, let end = t["vision_end_token_id"] as? Int else {
                throw Artifact.Failure.badHeader("`tokens` missing: not a Qwen3-VL encoder map")
            }
            for (text, id) in [("<|image_pad|>", image), ("<|vision_start|>", start), ("<|vision_end|>", end)] {
                guard tokenizer.encode(text) == [id] else {
                    throw Artifact.Failure.badHeader("tokenizer: \(text) is \(tokenizer.encode(text)), the map says \(id)")
                }
            }
            self.init(image: image, visionStart: start, visionEnd: end)
        }
    }

    /// `prompt_template_t2i` / `prompt_template_ti2i` of `QwenImage21Pipeline`, ONE `<|image_pad|>`
    /// per image (the processor expands them).
    package static func template(prompt: String, images: Int) -> String {
        // "Qwen has no bos token, so an empty string leaves the encoder with nothing to read."
        let text = prompt.isEmpty ? " " : prompt
        let slots = (0..<images).map { "<image\($0 + 1)><|vision_start|><|image_pad|><|vision_end|>" }
            .joined(separator: " ")
        return "<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n\(slots)\(text)<|im_end|>\n"
            + "<|im_start|>assistant\n"
    }

    /// The system turn as `apply_chat_template` writes it — its length is `_drop_idx`.
    package static var systemTurn: String { "<|im_start|>system\n\(system)<|im_end|>\n" }

    package struct Prepared {
        package let ids: [Int]
        package let dropCount: Int
        /// `[t, h, w]`, each `[ids.count]`.
        package let positions: [[Int]]
    }

    /// **What the prompt side refuses** — before the encoder reads a weight.
    package enum Failure: Error, CustomStringConvertible, Equatable {
        /// The prompt spells `<|image_pad|>`: the tokenizer splits it as the image slot's special
        /// token, and **the reference has no meaning for it either**: in an edit Qwen3-VL raises ("Image
        /// features and image tokens do not match"); in a generation the encoder reads it as a text token
        /// but marks its row in `image_pad_mask`, and the DiT raises on the slot no latent fills
        /// (`QwenImage21Transformer2DModel.build_token_metadata`, diffusers). Refused by name, before any
        /// weight is read, rather than crashing on `expand`'s count.
        case reservedText(String)
        /// More tokens than `maxPromptTokens` (see there).
        case tooLong(tokens: Int, max: Int)
        /// `expand` given more or fewer slots than images — the template's own count, never the user's.
        case slots(images: Int, slots: Int)

        package var description: String {
            switch self {
            case .reservedText(let text):
                return "the prompt contains \(text), a control token of Qwen-Image-2.1's encoder "
                    + "(the place of a reference image): remove it from the prompt"
            case let .tooLong(tokens, max):
                return "the prompt is \(tokens) tokens long, Qwen-Image-2.1 reads at most \(max): shorten it"
            case let .slots(images, slots): return "\(images) images for \(slots) <|image_pad|> slots"
            }
        }
    }

    /// **The longest prompt Qwen-Image-2.1 renders: 512 tokens** (of the prompt alone, before the
    /// template and the image slots).
    ///
    /// The reference does not bound it — `QwenImage21Pipeline._get_qwen_prompt_embeds` tokenizes the
    /// whole prompt, no `max_sequence_length` (its predecessor Qwen-Image cut at 1024) — so a bound
    /// here is the product's, not a port's: it refuses, it never truncates, and a prompt it accepts
    /// is encoded exactly as the reference does. 512 is the cut of every other text the engine reads
    /// (Z-Image, FLUX.2 [klein] pad or cut at 512), and what it costs is computed, not measured:
    ///
    ///   - encoder: `TextEncoder`'s arena takes `(5h + 3a + 2kv + 3i + Dh) · 4` B = 287 KB per token
    ///     (Qwen3-VL-8B: h = a = 4096, kv = 1024, i = 12288, Dh = 128): 147 MB for 512 tokens, plus the
    ///     causal mask, `S²` floats twice (array and graph constant): 2 MB at S = 512, 77 MB at
    ///     the worst edit's S ≈ 3 100;
    ///   - DiT: every text row is a row of the prefix — `x`, `keys`, `values`, `freqs`, ~49 KB per row
    ///     (25 MB), and the prefix K/V `2 · 32 layers · 4096 · 4` B = 1 MB per row when it is kept in
    ///     memory (512 MB; `MemoryPlan` puts it on disk when the budget says so).
    ///
    /// The measured floors (`ModelCard.measuredPeaks`) were taken with the library
    /// prompt, a dozen tokens: a 512-token prompt is beyond them by the figures above, which the
    /// reserve of `MemoryBudget` has to absorb — a measurement at 512 tokens is still owed.
    package static let maxPromptTokens = 512

    /// The prompt as a user may write it for this encoder: no image slot spelled out, at most
    /// `maxPromptTokens` tokens.
    package static func admit(_ prompt: String, tokenizer: Tokenizer, tokens: Tokens) throws {
        try admit(ids: tokenizer.encode(prompt), image: tokens.image)
    }

    /// `admit` on the prompt's identifiers alone.
    package static func admit(ids: [Int], image: Int) throws {
        if ids.contains(image) { throw Failure.reservedText("<|image_pad|>") }
        guard ids.count <= maxPromptTokens else { throw Failure.tooLong(tokens: ids.count, max: maxPromptTokens) }
    }

    /// - Parameter grids: each image's patch grid `(gh, gw)` — `image_grid_thw[1:]`, `t = 1`.
    package static func prepare(prompt: String, grids: [(Int, Int)], tokenizer: Tokenizer, tokens: Tokens,
                                merge: Int = 2) throws -> Prepared {
        try admit(prompt, tokenizer: tokenizer, tokens: tokens)
        let raw = tokenizer.encode(template(prompt: prompt, images: grids.count))
        let ids = try expand(raw, image: tokens.image, counts: grids.map { $0.0 * $0.1 / (merge * merge) })
        return Prepared(ids: ids, dropCount: tokenizer.encode(systemTurn).count,
                        positions: positions(ids: ids, image: tokens.image, grids: grids, merge: merge))
    }

    /// Each `<|image_pad|>` repeated as many times as its image has merged tokens — what
    /// `Qwen3VLProcessor` does on the string before tokenizing (the added token splits first, so
    /// expanding the identifiers is the same).
    package static func expand(_ ids: [Int], image: Int, counts: [Int]) throws -> [Int] {
        let slots = ids.reduce(0) { $0 + ($1 == image ? 1 : 0) }
        guard slots == counts.count else { throw Failure.slots(images: counts.count, slots: slots) }
        var out: [Int] = [], next = 0
        for id in ids {
            if id == image {
                out += Array(repeating: image, count: counts[next]); next += 1
            } else { out.append(id) }
        }
        return out
    }

    /// **`Qwen3VLModel.get_rope_index`** for one unpadded sequence: runs of text count up on the
    /// three axes; a run of image tokens (`mm_token_type_ids == 1`) is the merged grid
    /// `(gh/2) × (gw/2)` in row-major order at `(s, s + row, s + col)`, after which the text resumes at
    /// `s + max(gh, gw) / 2`.
    package static func positions(ids: [Int], image: Int, grids: [(Int, Int)], merge: Int = 2) -> [[Int]] {
        var t: [Int] = [], h: [Int] = [], w: [Int] = []
        var current = 0, index = 0, nextGrid = 0
        while index < ids.count {
            if ids[index] != image {
                t.append(current); h.append(current); w.append(current)
                current += 1; index += 1
                continue
            }
            let (gh, gw) = grids[nextGrid]; nextGrid += 1
            let (rows, columns) = (gh / merge, gw / merge)
            for r in 0..<rows {
                for c in 0..<columns {
                    t.append(current); h.append(current + r); w.append(current + c)
                }
            }
            index += rows * columns
            current += max(gh, gw) / merge
        }
        return [t, h, w]
    }
}
