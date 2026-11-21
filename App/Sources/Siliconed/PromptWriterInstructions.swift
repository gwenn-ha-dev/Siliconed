import Foundation

/// **Our instructions to the lab's writer: one system prompt per family it writes for, per gesture.**
///
///     Rewrite   user's text ──▶ system: common rules + the family's style + output contract
///     Describe  image       ──▶ system: common rules + the family's caption style + output contract
///
/// The texts are ours. Each family's rules are the style its publisher states for its own model, turned into short
/// instructions for the Qwen3-VL-8B that writes; they live in the family's folder (`FamilyTraits.writerStyle`), each
/// under its sources. No publisher's text is copied (their licences, and Qwen's rewriters were trained for other
/// weights, `Qwen-Image-2.1-PE-*`).
///
/// What a port by analogy would get wrong, and these do not:
///   - **Nothing is removed, softened, sanitized or refused**, and no rating tag is added. Krea 2's published
///     expansion prompt has a rule that dresses every person ("Respect the Human Form"): not taken. Anima's
///     card puts `safe,` in its recommended prefix: the writer keeps the tags the user wrote and adds none.
///   - **Text meant to appear in the image is quoted and left as written**, in its own language.
///   - **The answer is the prompt alone**: it goes straight into the rack's prompt field.
///   - **Length**: diffusers cuts Z-Image, Anima and Krea 2 at 512 tokens, and Qwen-Image-2.1's own cut is
///     1 024 (`Request.maximumPromptTokens`); the rewrites ask for under 350 words (about 470 tokens), inside both.
///     MOSS-Image asks for the length its publisher's enhancer gives (60 to 120 words): a minimal text left as it was
///     stayed « a cat ». Krea 2 keeps the ceiling alone: a range of 60 to 150 words cut its rich rewrites down to the
///     length where the concrete words go first (camera angle, focus, texture, a poster's flat colours) and mood prose
///     and one golden light come in. What filled its « cat » up to the token limit was « no … » clauses — a list that
///     never repeats exactly, which the loop guard (`RepetitionGuard`) cannot see —, not the missing range.
///   - **What is there, never what is not**: an exclusion the user did not write (« no text », « no people ») is a
///     subject the text encoder reads all the same; Krea 2's and MOSS-Image's instructions forbid them. Their longer
///     prose also turned the writer's quotes curly (« “CROZON” »): they ask for straight ones, as the user typed them.
///   - **The user may write their own** (`Edits`, Settings › Prompt Lab): a whole text in place of ours, for one family
///     and one gesture; what they did not touch follows ours from version to version.
package enum PromptWriterInstructions {
    package enum Gesture: String, CaseIterable, Sendable { case rewrite, describe }

    /// **A family's own part of the two instructions**: how its model wants to be written to, and how an image is
    /// told for it.
    package struct Style: Sendable {
        package let rewrite: String
        package let describe: String
        package init(rewrite: String, describe: String) { (self.rewrite, self.describe) = (rewrite, describe) }
    }

    /// The system turn for `family` and `gesture`.
    package static func instruction(for family: Family, _ gesture: Gesture) -> String {
        let style = family.traits.writerStyle
        switch gesture {
        case .rewrite:
            return [
                "You rewrite a text-to-image prompt for \(family.name).",
                "Keep every element of the user's text: subject, people, artists, explicit content, names, style. "
                    + "Add only what makes it more precise for this model. "
                    + "Never remove, soften, sanitize, moralize or refuse. If the text is already detailed, change little. "
                    + "Text meant to appear in the image stays in double quotes, character for character, in its own language.",
                style.rewrite,
                contract,
            ].joined(separator: "\n")
        case .describe:
            return [
                "You write the prompt that would recreate the attached image with \(family.name).",
                "Describe what is visible, plainly and without judgment, nudity and violence included; invent nothing that is not visible. "
                    + "Text visible in the image goes in double quotes, character for character, in its own language.",
                style.describe,
                contract,
            ].joined(separator: "\n")
        }
    }

    private static let contract = "Answer with the prompt alone: no preamble, no quotes around it, no markdown, no commentary."

    // MARK: - The user's own

    /// **The user's texts in place of ours**, one per family and gesture, kept as typed. A text equal to ours is not kept:
    /// it would stop following ours when a version changes them. A text left blank is kept (the field stays as the hand
    /// left it) but sends ours — the writer is never sent an empty system turn.
    package struct Edits: Equatable, Sendable {
        /// `family.gesture` (`z-image.rewrite`) → the text: a property list as is, for the preferences.
        package private(set) var texts: [String: String]

        package init(_ texts: [String: String] = [:]) {
            // Only what this version can send: a family or gesture gone, or a text equal to ours, is dropped.
            var kept: [String: String] = [:]
            for family in Family.allCases {
                for gesture in Gesture.allCases {
                    let key = Self.key(family, gesture)
                    if let text = texts[key], text != PromptWriterInstructions.instruction(for: family, gesture) { kept[key] = text }
                }
            }
            self.texts = kept
        }

        package static func key(_ family: Family, _ gesture: Gesture) -> String { "\(family.rawValue).\(gesture.rawValue)" }

        /// The text the field shows: theirs if they wrote one, blank included; ours otherwise.
        package func text(_ family: Family, _ gesture: Gesture) -> String {
            texts[Self.key(family, gesture)] ?? PromptWriterInstructions.instruction(for: family, gesture)
        }

        /// Whether the field is theirs: « Restore » brings ours back.
        package func isEdited(_ family: Family, _ gesture: Gesture) -> Bool { texts[Self.key(family, gesture)] != nil }

        /// Their text, when it is the one sent: written, and not blank.
        package func sent(_ family: Family, _ gesture: Gesture) -> String? {
            texts[Self.key(family, gesture)].flatMap { $0.allSatisfy(\.isWhitespace) ? nil : $0 }
        }

        /// **The system turn sent** for `family` and `gesture`: theirs if it is sent, ours otherwise.
        package func instruction(for family: Family, _ gesture: Gesture) -> String {
            sent(family, gesture) ?? PromptWriterInstructions.instruction(for: family, gesture)
        }

        /// Their text for `family` and `gesture`; ours, or `nil`, restores ours.
        package mutating func set(_ text: String?, for family: Family, _ gesture: Gesture) {
            let key = Self.key(family, gesture)
            if let text, text != PromptWriterInstructions.instruction(for: family, gesture) { texts[key] = text }
            else { texts[key] = nil }
        }
    }
}

/// **A dropped image, as the writer reads it**: ImageIO's decoding, the EXIF orientation applied, transparency over
/// white — and for an image that has transparency, Pillow's rule for an RGBA picture: Lanczos on the premultiplied
/// copy, then `white.paste(img, mask=A)` (compositing before resizing is what an analogy would do), brought straight
/// to the size the writer reads so that its own resize is the identity. An opaque one goes in as decoded:
/// `Qwen3VLChat.pixels` brings it to that size itself, once.
package enum PromptWriterImage {
    package static func rgb8(from data: Data) throws(EngineError) -> Qwen3VLImages.RGB8 {
        let image = try ImageRGB(data: data, name: "image")
        guard let rgba = image.rgba else {
            let bytes = image.rgba8()
            var rgb = [UInt8](repeating: 0, count: 3 * image.width * image.height)
            for p in 0..<(image.width * image.height) { for c in 0..<3 { rgb[3 * p + c] = bytes[4 * p + c] } }
            return Qwen3VLImages.RGB8(bytes: rgb, width: image.width, height: image.height)
        }
        let size = Qwen3VLImages.smartResize(height: image.height, width: image.width, factor: 32, maxPixels: Qwen3VLChat.maxPixels)
        let resized = PillowAlpha.lanczos(rgba, width: image.width, height: image.height, toWidth: size.width, toHeight: size.height)
        return PillowAlpha.overWhite(resized, width: size.width, height: size.height)
    }
}
