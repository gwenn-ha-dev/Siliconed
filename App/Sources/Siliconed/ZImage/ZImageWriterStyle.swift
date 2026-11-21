// How the prompt writer writes for Z-Image (`PromptWriterInstructions`).

extension PromptWriterInstructions.Style {
    package static let zImage = PromptWriterInstructions.Style(
        // Z-Image's prompt enhancer, which its fine-tuning data went through (paper §4.8):
        //   https://github.com/Tongyi-MAI/Z-Image · https://arxiv.org/abs/2511.22699
        //   https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo (`pe.py`: fix the unchangeable elements, add composition,
        //   light, material, colour and depth, quote the text, no metaphor, no emotional rhetoric, no "8K" or "masterpiece").
        // The rule against absences, in second place: the greedy writer turned « irish redhead on an old
        // green bike » into a list of « no … » clauses (« no visible wear on bike pedals, … ») up to the loop guard, and
        // Z-Image, distilled without a negative prompt, draws what a « no » names (« no lettering » wrote names on the boats).
        rewrite: "Write in English, one paragraph of plain descriptive sentences. "
            + "Describe only what is in the picture, never what is absent: write no phrase beginning with \"no\" or \"without\" "
            + "unless the user wrote it, not even at the end to say what the picture leaves out. "
            + "Where the text leaves them open, add composition, camera angle, lighting, materials, colours and depth. "
            + "No metaphors, no emotional rhetoric, no quality tags such as \"8K\" or \"masterpiece\". Under 350 words.",
        // Z-Captioner's long captions (paper §3): plain, objective, observable facts only, text read first.
        //   https://arxiv.org/abs/2511.22699 · https://github.com/Tongyi-MAI/Z-Image
        // Not « in the style of an image caption »: with « first any visible text » it made the writer open on a double
        // quote and wrap the whole answer in a pair (two images of two, once behind a « Z-Image: » too); without it, none
        // of three, and the text read on a wetsuit still comes first.
        describe: "Write in English, one paragraph of plain, objective sentences: "
            + "first any visible text, then the subjects, the setting, the composition, the lighting, the colours and the medium. "
            + "No metaphors, no quality tags. Under 350 words.")
}
