import Foundation

/// **The Krea 2 text**, as `Krea2Pipeline.get_text_hidden_states` prepares it.
///
///     system template + prompt ─ tokenizer, 1 024 at most ───┐
///     "assistant" suffix (5 tokens) ───────────────────────┴─ Qwen3-VL ─ hidden_states[2, 5, …, 35]
///                                                            ─ the 34 template tokens removed
///
/// The reference pads in the middle — the prompt, **then** the padding, then the suffix — and masks.
/// Here nothing is padded: Qwen's attention is causal and the padding is never a visible
/// key, and the reference's positions (`cumsum(mask) − 1`) are precisely those of a
/// sequence without padding. Same sum, minus the products by zero.
package enum Krea2Text {
    package static let template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        + "texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
        + "<|im_start|>user\n"
    package static let suffix = "<|im_end|>\n<|im_start|>assistant\n"
    /// `prompt_template_encode_start_idx`: the template tokens, removed after the encoder.
    package static let templateTokens = 34
    package static let length = 512

    /// The layers drawn from the encoder, read from the published `model_index.json` — not copied:
    /// the file itself is in the code (`Krea2Published`, its git blob checked by a test), `[2, 5, …, 35]`.
    package static var sockets: [Int] {
        (Krea2Published.modelIndex["text_encoder_select_layers"]?.elements ?? []).compactMap(\.integer)
    }

    /// The identifiers the encoder receives, template included, without padding.
    package static func identifiers(_ prompt: String, tokenizer: Tokenizer) -> [Int] {
        let head = Array(tokenizer.encode(template + prompt)
            .prefix(length + templateTokens - tokenizer.encode(suffix).count))
        return head + tokenizer.encode(suffix)
    }

    /// `identifiers` for a render: the whole prompt, **refused past `Request.maximumPromptTokens`**,
    /// never truncated. `identifiers` keeps the reference's cut (507 tokens of prompt), for the checks.
    package static func admitted(_ prompt: String, tokenizer: Tokenizer) throws -> [Int] {
        let head = tokenizer.encode(template + prompt)
        try Request.admitPrompt(tokens: head.count - templateTokens, max: Request.maximumPromptTokens)
        return head + tokenizer.encode(suffix)
    }

    /// `[T, 12, 2560]`, `T` = identifiers − template.
    package static func hiddenStates(_ ids: [Int], encoder: String, sockets: [Int],
                                   cancellation: Cancellation? = nil) throws -> [Float] {
        let encoder = try TextEncoder(artifact: try KeptEncoder.artifact(encoder), sequence: ids.count)
        encoder.cancellation = cancellation
        return try encoder.encodeTaps(ids: ids, taps: sockets, dropping: templateTokens)
    }
}

extension Krea2DiT {
    /// The Krea 2 Turbo schedule: `sigmas = linspace(1, 1/N, N)`, shifted by the **exponential
    /// shift** at `μ = 1.15` (`is_distilled`: fixed μ, whatever the resolution) —
    /// `σ' = e^μ / (e^μ + (1/σ − 1))` —, then a zero terminal σ.
    ///
    /// **In float32, operation by operation**, as `set_timesteps` computes it: NumPy's double ramp cast
    /// to float32, then `math.exp(μ) / (math.exp(μ) + (1 / t − 1) ** 1.0)` on a float32 array — `e^μ`
    /// rounded to float32 (a weak Python float), `** 1.0` the identity. The same formula in double,
    /// rounded once, is one bit off on 4 of the 8 σ at N = 8 (402 of 1 275 for N = 1…50). To the bit
    /// against diffusers (`SolverTests`).
    package static func sigmas(steps: Int, mu: Double = 1.15) -> [Float] {
        let e = Float(exp(mu))
        return NumPyRamp.linspace(1, 1 / Double(steps), steps).map { s -> Float in
            let t = Float(s)
            return e / (e + (1 / t - 1))
        } + [0]
    }
}
