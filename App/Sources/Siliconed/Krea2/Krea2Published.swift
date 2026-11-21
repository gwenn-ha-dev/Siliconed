import Foundation

/// **`model_index.json` of `krea/Krea-2-Turbo`, as published** — the only bytes of that repository the engine still reads, and
/// the Hub serves them behind a gate (a token, a license accepted): a user of the app has neither. The weights come from
/// ungated repositories; this file is written here **verbatim**, byte for byte, at revision `98e0fe11`. A test hashes the
/// literal as git does (`blob <n>\0` + bytes, SHA-1) and finds the blob id the Hub lists for the file at that revision: the
/// text here *is* the file.
///
///     model_index.json           → `text_encoder_select_layers`, the twelve layers drawn (`Krea2Text`)
package enum Krea2Published {
    /// The revision these bytes are from, and the file's git blob id there (`/api/models/…/tree`).
    package static let revision = "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"
    package static let blobs: [(path: String, blob: String, text: String)] = [
        ("model_index.json", "889dbf53a26b78303ec0d9782c9bd3a1a13e2483", modelIndexText),
    ]

    /// `model_index.json`, parsed in its order.
    package static let modelIndex = parse(modelIndexText)

    private static func parse(_ text: String) -> OrderedJSON {
        // The literals are fixed and tested: a parse failure is a programming error.
        try! OrderedJSON.parse(Data(text.utf8))
    }

    static let modelIndexText = #"""
        {
          "_class_name": "Krea2Pipeline",
          "_diffusers_version": "0.39.0.dev0",
          "scheduler": [
            "diffusers",
            "FlowMatchEulerDiscreteScheduler"
          ],
          "text_encoder": [
            "transformers",
            "Qwen3VLModel"
          ],
          "text_encoder_select_layers": [
            2,
            5,
            8,
            11,
            14,
            17,
            20,
            23,
            26,
            29,
            32,
            35
          ],
          "tokenizer": [
            "transformers",
            "Qwen2Tokenizer"
          ],
          "transformer": [
            "diffusers",
            "Krea2Transformer2DModel"
          ],
          "vae": [
            "diffusers",
            "AutoencoderKLQwenImage"
          ],
          "is_distilled": true,
          "patch_size": 2
        }
        """#
}
