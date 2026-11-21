// How the prompt writer writes for Qwen-Image-2.1 (`PromptWriterInstructions`).

extension PromptWriterInstructions.Style {
    package static let qwenImage21 = PromptWriterInstructions.Style(
        // Qwen-Image-2.1's rewriter guidance (style only; its prompts belong to its own weights):
        //   https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite
        // and its text encoder's own system prompt (colour, shape, size, texture, quantity, text, spatial relations):
        //   https://github.com/QwenLM/Qwen-Image-2.1
        rewrite: "Write in English, one paragraph, in the present tense. "
            + "Be concrete: each subject with its colour, shape, size, texture and quantity, then the spatial relationships, "
            + "the setting, the lighting, the medium and style. No quality words such as \"4K\" or \"masterpiece\". "
            + "Keep it under 350 words.",
        // The same style as the rewrite: colour, shape, size, texture, quantity, text, spatial relationships.
        //   https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite
        describe: "Write in English, one paragraph, in the present tense: each subject with its colour, shape, size, texture and quantity, "
            + "then the spatial relationships, the setting, the lighting, the medium and style. No quality words. Under 350 words.")
}
