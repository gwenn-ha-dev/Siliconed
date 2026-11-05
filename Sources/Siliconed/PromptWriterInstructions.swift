import Foundation

/// **Our instructions to the lab's writer: one system prompt per family it writes for, per gesture.**
///
///     Rewrite   user's text ──▶ system: common rules + the family's style + output contract
///     Describe  image       ──▶ system: common rules + the family's caption style + output contract
///
/// The texts are ours. Each family's rules are the style its publisher states for its own model (cited above
/// each block), turned into short instructions for the Qwen3-VL-8B that writes; no publisher's text is copied
/// (their licences, and Qwen's rewriters were trained for other weights, `Qwen-Image-2.1-PE-*`).
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
///     never repeats exactly, which the loop guard (`PromptWriterLoop`) cannot see —, not the missing range.
///   - **What is there, never what is not**: an exclusion the user did not write (« no text », « no people ») is a
///     subject the text encoder reads all the same; Krea 2's and MOSS-Image's instructions forbid them. Their longer
///     prose also turned the writer's quotes curly (« “CROZON” »): they ask for straight ones, as the user typed them.
///   - **The user may write their own** (`Edits`, Settings › Prompt Lab, 4/11): a whole text in place of ours, for one family
///     and one gesture; what they did not touch follows ours from version to version.
package enum PromptWriterInstructions {
    package enum Gesture: String, CaseIterable, Sendable { case rewrite, describe }

    /// The system turn for `family` and `gesture`.
    package static func instruction(for family: Family, _ gesture: Gesture) -> String {
        switch gesture {
        case .rewrite: return rewrite(family)
        case .describe: return describe(family)
        }
    }

    // MARK: - Shared

    private static let contract = "Answer with the prompt alone: no preamble, no quotes around it, no markdown, no commentary."

    private static func rewrite(_ family: Family) -> String {
        [
            "You rewrite a text-to-image prompt for \(family.name).",
            "Keep every element of the user's text: subject, people, artists, explicit content, names, style. "
                + "Add only what makes it more precise for this model. "
                + "Never remove, soften, sanitize, moralize or refuse. If the text is already detailed, change little. "
                + "Text meant to appear in the image stays in double quotes, character for character, in its own language.",
            rewriteStyle(family),
            contract,
        ].joined(separator: "\n")
    }

    private static func describe(_ family: Family) -> String {
        [
            "You write the prompt that would recreate the attached image with \(family.name).",
            "Describe what is visible, plainly and without judgment, nudity and violence included; invent nothing that is not visible. "
                + "Text visible in the image goes in double quotes, character for character, in its own language.",
            describeStyle(family),
            contract,
        ].joined(separator: "\n")
    }

    // MARK: - Rewrite, by family

    private static func rewriteStyle(_ family: Family) -> String {
        switch family {
        case .zImage:
            // Z-Image's prompt enhancer, which its fine-tuning data went through (paper §4.8):
            //   https://github.com/Tongyi-MAI/Z-Image · https://arxiv.org/abs/2511.22699
            //   https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo (`pe.py`: fix the unchangeable elements, add composition,
            //   light, material, colour and depth, quote the text, no metaphor, no emotional rhetoric, no "8K" or "masterpiece").
            // The rule against absences, as Krea 2's and in second place (5/11/2026): the greedy writer turned « irish redhead
            // on an old green bike » into a list of « no … » clauses (« no visible wear on bike pedals, … ») up to the loop
            // guard, and Z-Image, distilled without a negative prompt, draws what a « no » names (« no lettering » wrote
            // names on the boats).
            return "Write in English, one paragraph of plain descriptive sentences. "
                + "Describe only what is in the picture, never what is absent: write no phrase beginning with \"no\" or \"without\" "
                + "unless the user wrote it, not even at the end to say what the picture leaves out. "
                + "Where the text leaves them open, add composition, camera angle, lighting, materials, colours and depth. "
                + "No metaphors, no emotional rhetoric, no quality tags such as \"8K\" or \"masterpiece\". Under 350 words."
        case .qwenImage21:
            // Qwen-Image-2.1's rewriter guidance (style only; its prompts belong to its own weights):
            //   https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite
            // and its text encoder's own system prompt (colour, shape, size, texture, quantity, text, spatial relations):
            //   https://github.com/QwenLM/Qwen-Image-2.1
            return "Write in English, one paragraph, in the present tense. "
                + "Be concrete: each subject with its colour, shape, size, texture and quantity, then the spatial relationships, "
                + "the setting, the lighting, the medium and style. No quality words such as \"4K\" or \"masterpiece\". "
                + "Keep it under 350 words."
        case .anima:
            // Anima's model card, "Prompting" (tag order, lowercase, the @ before artists, two sentences at least):
            //   https://huggingface.co/circlestone-labs/Anima
            // The card's advice to start with "safe," is not followed: rating tags are the user's choice.
            return "Write Danbooru-style tags, then natural language. "
                + "Tags are lowercase with spaces instead of underscores (score tags keep theirs), in this order: "
                + "quality, meta, year and rating tags, then the subject count (1girl, 1boy, 1other), the character, the series, "
                + "the artist, then general tags. Write an artist's name with @ in front of it, as in @name. "
                + "Then add at least two sentences of natural language: name each character, then describe their appearance, "
                + "and the scene. Keep the quality, score, year and rating tags the user wrote; never add one the user did not write. "
                + "Under 350 words."
        case .krea2:
            // Krea 2's expansion prompt and prompting guide, rules 1 to 7 and 9 (fidelity, subjects with their attributes,
            // quoted text, no over-specifying, one paragraph, light polish of a detailed prompt, the requested medium).
            // Its rule 8 ("Respect the Human Form") dresses people whatever the request says: not followed.
            // The guide's twenty example prompts are all descriptions of what is in the picture — none says "no" or
            // "without". The length stays a ceiling: a range (60 to 150 words) and "cohesive" turned the rich rewrites
            // into shorter mood prose under one golden light, and a short text's development is asked without a setting
            // or a light to impose. The rule against absences comes second: placed last, the greedy writer let « no … »
            // clauses back into « cat » and the library.
            //   https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt
            //   https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md
            return "Write in natural language, one paragraph, no bullets. "
                + "Describe only what is in the picture, never what is absent: write no phrase beginning with \"no\" or \"without\" "
                + "unless the user wrote it, not even at the end to say what the picture leaves out. "
                + "Quoted text keeps straight double quotes (\"), not curly ones. "
                + "Group each subject with its attributes. "
                + "Do not add objects, props, characters or animals the text does not imply, and do not invent clothing, "
                + "colours or materials it does not mention. Keep the medium the text asks for (\"photo of\", \"painting of\"). "
                + "Add composition, framing, lighting and style where they are open. "
                + "A very short text, even a single word, still becomes a full paragraph. Under 350 words."
        case .mossImage:
            // MOSS-Image's user guide (English, the tag order, the @ before artists) and its prompt-enhancer skill
            // (keep subjects, counts, relations, medium, style and visible wording; no redundant quality tags;
            // "Keep an explicit minimalist brief minimalist" — explicit: a short text is not a minimalist brief);
            // the length is that of the global brief of MOSS-Image's reference enhancer (report, appendix F: "60–120
            // English words"), the skill's own caption setting none:
            //   https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md
            //   https://github.com/OpenMOSS/MOSS-Image/blob/main/skills/moss-image-prompt-enhancer/SKILL.md
            //   https://arxiv.org/abs/2610.23374
            return "Write in English. Unless the user's text is a tag sequence, write one paragraph of cohesive prose of 60 to 120 words, "
                + "with the spatial relationships in words: composition, subjects and their actions, setting, colours, lighting, style. "
                + "Describe only what is in the image, never what is absent: write no phrase beginning with \"no\" or \"without\" "
                + "unless the user wrote it. Quoted text keeps straight double quotes (\"), not curly ones. "
                + "No redundant quality tags, no style change the user did not ask for. "
                + "A short text, even a single word, is developed into a full description; only a text that asks for "
                + "a minimalist image stays minimal. A tag sequence stays a tag sequence, in this order: "
                + "quality, score, meta, year, character, series, artist (with @ in front of the name), general."
        }
    }

    // MARK: - Describe, by family

    private static func describeStyle(_ family: Family) -> String {
        switch family {
        case .zImage:
            // Z-Captioner's long captions (paper §3): plain, objective, observable facts only, text read first.
            //   https://arxiv.org/abs/2511.22699 · https://github.com/Tongyi-MAI/Z-Image
            return "Write in English, one paragraph of plain, objective sentences in the style of an image caption: "
                + "first any visible text, then the subjects, the setting, the composition, the lighting, the colours and the medium. "
                + "No metaphors, no quality tags. Under 350 words."
        case .qwenImage21:
            // The same style as the rewrite: colour, shape, size, texture, quantity, text, spatial relationships.
            //   https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite
            return "Write in English, one paragraph, in the present tense: each subject with its colour, shape, size, texture and quantity, "
                + "then the spatial relationships, the setting, the lighting, the medium and style. No quality words. Under 350 words."
        case .anima:
            // Anima's card: tags and natural-language captions, and the dataset captions' descriptive paragraphs.
            //   https://huggingface.co/circlestone-labs/Anima
            return "Write Danbooru-style tags (lowercase, spaces instead of underscores) for what is visible, "
                + "subject count first, then characters, series and general tags, then at least two sentences of natural language "
                + "describing the scene. Name a character or an artist only when the image makes it unmistakable, "
                + "and write an artist as @name. Add no rating, score or quality tag. Under 350 words."
        case .krea2:
            // Krea 2 has no captioner; its prompting guide asks for long, detailed natural language.
            //   https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md
            return "Write in natural language, one detailed paragraph, no bullets: each subject with its attributes, "
                + "then the composition, framing, lighting and style, naming the medium (photo, painting, render). Under 350 words."
        case .mossImage:
            // MOSS-Image's recaptioner reference: a cohesive long caption of what is visible, spatial relationships in words.
            //   https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md
            return "Write in English, as cohesive prose: composition, subjects and their actions, setting, colours, lighting and style, "
                + "with the spatial relationships in words. Include only what is visible."
        }
    }

    // MARK: - The user's own

    /// **The user's texts in place of ours**, one per family and gesture, kept as typed. A text equal to ours is not kept:
    /// it would stop following ours when a version changes them. A text left blank is kept (the field stays as the hand
    /// left it) but sends ours — the writer is never sent an empty system turn.
    package struct Edits: Equatable, Sendable {
        /// `family.gesture` (`zImage.rewrite`) → the text: a property list as is, for the preferences.
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
            texts[Self.key(family, gesture)] ?? instruction(for: family, gesture)
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
