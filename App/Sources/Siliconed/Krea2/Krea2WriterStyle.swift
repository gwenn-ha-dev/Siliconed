// How the prompt writer writes for Krea 2 (`PromptWriterInstructions`).

extension PromptWriterInstructions.Style {
    package static let krea2 = PromptWriterInstructions.Style(
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
        rewrite: "Write in natural language, one paragraph, no bullets. "
            + "Describe only what is in the picture, never what is absent: write no phrase beginning with \"no\" or \"without\" "
            + "unless the user wrote it, not even at the end to say what the picture leaves out. "
            + "Quoted text keeps straight double quotes (\"), not curly ones. "
            + "Group each subject with its attributes. "
            + "Do not add objects, props, characters or animals the text does not imply, and do not invent clothing, "
            + "colours or materials it does not mention. Keep the medium the text asks for (\"photo of\", \"painting of\"). "
            + "Add composition, framing, lighting and style where they are open. "
            + "A very short text, even a single word, still becomes a full paragraph. Under 350 words.",
        // Krea 2 has no captioner; its prompting guide asks for long, detailed natural language.
        //   https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md
        describe: "Write in natural language, one detailed paragraph, no bullets: each subject with its attributes, "
            + "then the composition, framing, lighting and style, naming the medium (photo, painting, render). Under 350 words.")
}
