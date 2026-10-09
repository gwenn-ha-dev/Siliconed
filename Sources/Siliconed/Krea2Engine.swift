import Foundation

/// **The Krea 2 text**, as `Krea2Pipeline.get_text_hidden_states` prepares it.
///
///     system template + prompt ─ tokenizer, truncated to 541 ─┐
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

    /// The layers drawn from the encoder, read from the published `model_index.json` — not copied.
    package static func sockets(index: String) throws -> [Int] {
        let data = try Data(contentsOf: URL(fileURLWithPath: index))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let sockets = root["text_encoder_select_layers"] as? [Int] else {
            throw Safetensors.Failure.badHeader("\(index): `text_encoder_select_layers` missing from model_index.json")
        }
        return sockets
    }

    /// The identifiers the encoder receives, template included, without padding.
    package static func identifiers(_ prompt: String, tokenizer: Tokenizer) -> [Int] {
        let head = Array(tokenizer.encode(template + prompt)
            .prefix(length + templateTokens - tokenizer.encode(suffix).count))
        return head + tokenizer.encode(suffix)
    }

    /// `[T, 12, 2560]`, `T` = identifiers − template.
    package static func hiddenStates(_ ids: [Int], encoder: String, sockets: [Int],
                                   freezeCut: Bool = EngineSettings.effective.frozenCut,
                                   cancellation: Cancellation? = nil) throws -> [Float] {
        let encoder = try TextEncoder(artifact: try Artifact(path: encoder), sequence: ids.count,
                                      freezeCut: freezeCut)
        encoder.cancellation = cancellation
        let all = try encoder.encodeTaps(ids: ids, taps: sockets)
        return Array(all[(templateTokens * sockets.count * encoder.config.hidden)...])
    }
}

extension Krea2DiT {
    /// The Krea 2 Turbo schedule: `sigmas = linspace(1, 1/N, N)`, shifted by the **exponential
    /// shift** at `μ = 1.15` (`is_distilled`: fixed μ, whatever the resolution) —
    /// `σ' = e^μ / (e^μ + (1/σ − 1))` —, then a zero terminal σ. Checked against the oracle's
    /// by the trajectory check.
    package static func sigmas(steps: Int, mu: Double = 1.15) -> [Float] {
        (0..<steps).map { i -> Float in
            let s = steps == 1 ? 1 : 1 - Double(i) * (1 - 1 / Double(steps)) / Double(steps - 1)
            return Float(exp(mu) / (exp(mu) + (1 / s - 1)))
        } + [0]
    }
}
