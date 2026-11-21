// How the prompt writer writes for Anima (`PromptWriterInstructions`).

extension PromptWriterInstructions.Style {
    package static let anima = PromptWriterInstructions.Style(
        // Anima's model card, "Prompting" (tag order, lowercase, the @ before artists, two sentences at least):
        //   https://huggingface.co/circlestone-labs/Anima
        // The card's advice to start with "safe," is not followed: rating tags are the user's choice.
        rewrite: "Write Danbooru-style tags, then natural language. "
            + "Tags are lowercase with spaces instead of underscores (score tags keep theirs), in this order: "
            + "quality, meta, year and rating tags, then the subject count (1girl, 1boy, 1other), the character, the series, "
            + "the artist, then general tags. Write an artist's name with @ in front of it, as in @name. "
            + "Then add at least two sentences of natural language: name each character, then describe their appearance, "
            + "and the scene. Keep the quality, score, year and rating tags the user wrote; never add one the user did not write. "
            + "Under 350 words.",
        // Anima's card: tags and natural-language captions, and the dataset captions' descriptive paragraphs.
        //   https://huggingface.co/circlestone-labs/Anima
        describe: "Write Danbooru-style tags (lowercase, spaces instead of underscores) for what is visible, "
            + "subject count first, then characters, series and general tags, then at least two sentences of natural language "
            + "describing the scene. Name a character or an artist only when the image makes it unmistakable, "
            + "and write an artist as @name. Add no rating, score or quality tag. Under 350 words.")
}
