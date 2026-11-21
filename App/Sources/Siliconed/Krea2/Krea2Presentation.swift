// What the interface says of Krea 2: English sentences, the keys of the app's catalog (`tools/translations.sh` reads them here).

import Foundation

extension FamilyPresentation {
    package static let krea2 = FamilyPresentation(
        symbol: "camera.aperture",
        // The largest DiT of the five (12.8 B), and the slowest render.
        blurb: "Aesthetic images, from photography to painting. The largest model, and the slowest.",
        // Krea's own repository is gated: every byte comes from copies proven equal to its files.
        details: [.standard: "The image model and the text encoder in the publisher’s precision (bf16), downloaded from Comfy-Org’s copy; the VAE and the tokenizer from Qwen’s. Each is the publisher’s file, value for value.",
                  .compact: "8-bit weights published by Comfy-Org: the image model and the text encoder (int8 convrot). VAE and tokenizer as in the Standard version."],
        // § 4.2 asks whoever deploys the model for content filtering; the app filters nothing (no prompt is
        // censored): said once, plainly, where the license is accepted.
        licenseNotes: ["The license asks whoever deploys the model to filter the content it makes (§ 4.2). Siliconed applies no filter: by installing the model and accepting its license, you take on that responsibility."],
        // Krea's repository is gated (a Hugging Face account, its form): its files come from ungated copies, each
        // proven equal to Krea's.
        sources: [.standard: "Siliconed ships no weights. The publisher’s repository asks for a Hugging Face account, so its files are downloaded from copies proven identical: the image model and the text encoder from Comfy-Org, the VAE and the tokenizer from Qwen. The publisher’s license applies, and each model card says whose files they are."],
        licenseName: "Krea 2 Community (commercial under $1M, mandatory content filter)")
}
