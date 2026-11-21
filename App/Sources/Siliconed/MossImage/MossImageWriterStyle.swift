// How the prompt writer writes for MOSS-Image (`PromptWriterInstructions`).

extension PromptWriterInstructions.Style {
    package static let mossImage = PromptWriterInstructions.Style(
        // MOSS-Image's user guide (English, the tag order, the @ before artists) and its prompt-enhancer skill
        // (keep subjects, counts, relations, medium, style and visible wording; no redundant quality tags;
        // "Keep an explicit minimalist brief minimalist" — explicit: a short text is not a minimalist brief);
        // the length is that of the global brief of MOSS-Image's reference enhancer (report, appendix F: "60–120
        // English words"), the skill's own caption setting none:
        //   https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md
        //   https://github.com/OpenMOSS/MOSS-Image/blob/main/skills/moss-image-prompt-enhancer/SKILL.md
        //   https://arxiv.org/abs/2610.23374
        rewrite: "Write in English. Unless the user's text is a tag sequence, write one paragraph of cohesive prose of 60 to 120 words, "
            + "with the spatial relationships in words: composition, subjects and their actions, setting, colours, lighting, style. "
            + "Describe only what is in the image, never what is absent: write no phrase beginning with \"no\" or \"without\" "
            + "unless the user wrote it. Quoted text keeps straight double quotes (\"), not curly ones. "
            + "No redundant quality tags, no style change the user did not ask for. "
            + "A short text, even a single word, is developed into a full description; only a text that asks for "
            + "a minimalist image stays minimal. A tag sequence stays a tag sequence, in this order: "
            + "quality, score, meta, year, character, series, artist (with @ in front of the name), general.",
        // MOSS-Image's recaptioner reference: a cohesive long caption of what is visible, spatial relationships in words.
        //   https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md
        describe: "Write in English, as cohesive prose: composition, subjects and their actions, setting, colours, lighting and style, "
            + "with the spatial relationships in words. Include only what is visible.")
}
