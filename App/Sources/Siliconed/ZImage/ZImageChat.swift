import Foundation

// **Z-Image's chat template, and what its pipeline does with it**: the Qwen3 template of one user turn, the padding to
// a fixed length and the mask (`pipeline_z_image.py`), beside the shared tokenizer.

extension Tokenizer {
    /// `apply_chat_template([{user, prompt}], add_generation_prompt=True, enable_thinking=True)`
    /// for the Qwen3 template, reduced to the only form the pipeline ever asks for.
    ///
    /// **No Jinja engine**: with a single user turn and thinking enabled, the template is
    /// this literal and nothing else. `goldens-text.json` records the string `transformers`
    /// produces — the shortcut is **pinned**, not assumed.
    package static let chatPrefix = "<|im_start|>user\n"
    package static let chatSuffix = "<|im_end|>\n<|im_start|>assistant\n"

    /// The padding token: `<|endoftext|>`. Read from `goldens-text.safetensors`, not deduced.
    package static let padToken = 151643

    /// What the engine gives the text encoder: `[maxLength]` identifiers and their mask.
    ///
    /// **The pipeline pads to fixed length** (`padding="max_length"`, 512) and does not cut
    /// short sequences — the mask carries the information, not the length. `cap_feats` is then
    /// the masked extraction: 19 rows out of 512 for the bench prompt (`pipeline_z_image.py`
    /// 217-245).
    ///
    /// > **Truncation is silent and it costs the assistant header**: at 512, a
    /// > long prompt is cut mid-sentence and loses `<|im_end|>\n<|im_start|>assistant\n`
    /// > entirely. That is the reference's behavior, so it is **reproduced and not corrected** —
    /// > the useful budget is 504 tokens, and saying so at admission is the caller's job.
    package func encodeChat(user prompt: String, maxLength: Int = 512) -> (ids: [Int], mask: [Int]) {
        Self.pad(encode(Self.chatPrefix + prompt + Self.chatSuffix), maxLength: maxLength)
    }

    /// `encodeChat` for a render: the whole prompt in its template, unpadded (the encoder reads the
    /// real tokens alone), **refused past `Request.maximumPromptTokens`**, never truncated; `prompt`, the
    /// prompt's own tokens. `encodeChat` keeps the reference's cut at 512, for the checks.
    package func admitChat(user prompt: String) throws -> (ids: [Int], prompt: Int) {
        let ids = encode(Self.chatPrefix + prompt + Self.chatSuffix)
        let tokens = ids.count - encode(Self.chatPrefix + Self.chatSuffix).count
        try Request.admitPrompt(tokens: tokens, max: Request.maximumPromptTokens)
        return (ids, tokens)
    }

    /// **Truncation and padding, separated from encoding.** This is the half of `encodeChat` that
    /// does not depend on the vocabulary — hence the half that can be checked in a millisecond, without
    /// the 7 MB of JSON that the tokenizer parses at startup.
    ///
    /// This is not a test convenience: it is the rule that carries **the documented trap
    /// above**. At 512, a long prompt is cut mid-sentence and loses
    /// `<|im_end|>\n<|im_start|>assistant\n` entirely — the reference's behavior, reproduced
    /// and not corrected. A rule described in a comment and never executed is
    /// a rule one believes is upheld.
    static func pad(_ identifiers: [Int], maxLength: Int) -> (ids: [Int], mask: [Int]) {
        var ids = identifiers
        if ids.count > maxLength { ids = Array(ids.prefix(maxLength)) }
        let realCount = ids.count
        ids += Array(repeating: padToken, count: maxLength - realCount)
        return (ids, Array(repeating: 1, count: realCount)
                   + Array(repeating: 0, count: maxLength - realCount))
    }
}
