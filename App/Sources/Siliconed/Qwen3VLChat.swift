import Foundation

/// **A chat for Qwen3-VL to continue**: the Instruct's `apply_chat_template(messages, add_generation_prompt=True)`,
/// tokenized, each picture's slot repeated over its merged tokens. Not the image pipeline's template (its own system
/// turn, its `<imageN>` tags): this is what the prompt writer reads.
package enum Qwen3VLChat {
    package static let imagePad = "<|image_pad|>"
    package static let imageSlot = "<|vision_start|>" + imagePad + "<|vision_end|>"

    /// An optional `system` turn, then one `user` turn: the picture slots one after the other, the text right after
    /// them. A system turn is written exactly when one is given.
    package static func template(system: String? = nil, user: String, images: Int) -> String {
        (system.map { "<|im_start|>system\n" + $0 + "<|im_end|>\n" } ?? "")
            + "<|im_start|>user\n" + String(repeating: imageSlot, count: images) + user
            + "<|im_end|>\n<|im_start|>assistant\n"
    }

    package enum Failure: Error, CustomStringConvertible, Equatable {
        /// A text spells one of the template's control tokens: the tokenizer would read it as the token, not as text.
        case reservedText(String)
        package var description: String {
            switch self { case .reservedText(let t): return "the text contains \(t), a control token of Qwen3-VL's chat template" }
        }
    }

    /// The control tokens of the template that a text must not spell.
    package static let reserved = [imagePad, "<|video_pad|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
                                   "<|im_start|>", "<|im_end|>", "<|endoftext|>"]

    /// `generation_config.eos_token_id`, in its order: `<|im_end|>`, `<|endoftext|>`.
    package static func stops(_ tokenizer: Tokenizer) -> [Int] {
        ["<|im_end|>", "<|endoftext|>"].compactMap { text in
            let ids = tokenizer.encode(text)
            return ids.count == 1 ? ids[0] : nil
        }
    }

    /// The template tokenized, each `<|image_pad|>` repeated over its picture's merged tokens; `slots`: the rows that
    /// are a picture's. `grids`: each picture's grid in patches.
    package static func prepare(system: String? = nil, user: String, grids: [(height: Int, width: Int)], tokenizer: Tokenizer,
                                merge: Int = 2) throws -> (ids: [Int], slots: [Bool]) {
        for text in reserved where user.contains(text) || (system?.contains(text) ?? false) { throw Failure.reservedText(text) }
        return try expanded(template(system: system, user: user, images: grids.count),
                            rows: grids.map { $0.height * $0.width / (merge * merge) }, tokenizer: tokenizer)
    }

    /// `<|image_pad|>`'s identifier: the rows a picture fills.
    package static func padIdentifier(_ tokenizer: Tokenizer) throws -> Int {
        let ids = tokenizer.encode(imagePad)
        guard ids.count == 1 else { throw ProgramError("the tokenizer has no \(imagePad) token") }
        return ids[0]
    }

    /// **A template tokenized, its `i`-th `<|image_pad|>` repeated `rows[i]` times** — what Qwen3-VL's processor does to
    /// a picture's one slot, whichever template holds it; `slots`: the rows that are a picture's.
    package static func expanded(_ template: String, rows: [Int], tokenizer: Tokenizer) throws -> (ids: [Int], slots: [Bool]) {
        let pad = try padIdentifier(tokenizer)
        var ids: [Int] = [], slots: [Bool] = [], next = 0
        for id in tokenizer.encode(template) {
            guard id == pad else { ids.append(id); slots.append(false); continue }
            guard next < rows.count else { throw ProgramError("the template holds more picture slots than pictures") }
            ids.append(contentsOf: [Int](repeating: id, count: rows[next]))
            slots.append(contentsOf: [Bool](repeating: true, count: rows[next]))
            next += 1
        }
        guard next == rows.count else { throw ProgramError("\(rows.count) pictures for \(next) slots in the template") }
        return (ids, slots)
    }

    /// **At most 2²⁰ pixels a picture**, so at most 1 024 picture tokens — the largest condition Qwen-Image-2.1 hands
    /// this tower. The processor's own bound (2²⁴) would let a 4K scan cost 16 384 tokens; a description does not
    /// need them.
    package static let maxPixels = 1 << 20

    /// A picture as the writer reads it: `smart_resize(factor 32, max_pixels)`, brought there by Pillow's Lanczos so
    /// that the processor's own resize is the identity (the oracle does the same), then the processor.
    package static func pixels(_ image: Qwen3VLImages.RGB8, maxPixels: Int = maxPixels) throws -> Qwen3VLImages.Pixels {
        let size = Qwen3VLImages.smartResize(height: image.height, width: image.width, factor: 32, maxPixels: maxPixels)
        return try Qwen3VLImages.pixels(PillowResample.lanczos(image, width: size.width, height: size.height))
    }

    /// The text written, as `batch_decode(skip_special_tokens=True)` gives it.
    package static func text(_ generated: [Int], tokenizer: Tokenizer) -> String {
        tokenizer.decode(generated.filter { !tokenizer.specialIdentifiers.contains($0) })
    }
}
