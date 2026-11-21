// What the interface says of Anima: English sentences, the keys of the app's catalog (`tools/translations.sh` reads them here).

import Foundation

extension FamilyPresentation {
    package static let anima = FamilyPresentation(
        symbol: "paintbrush",
        // Its publisher trained it for illustration and keeps it there: no photorealism, by choice.
        blurb: "Illustration, painting and anime. No photorealism: its publisher’s choice.",
        // Its two versions share one text encoder, the publisher's: only its image model changes.
        summaries: [.compact: "Smaller on disk: the image model in 8 bits, the publisher’s text encoder; images slightly different."],
        details: [.compact: "The image model in 8 bits, published by Bedovyy (int8 convrot). The text encoder (bf16), tokenizers and VAE are the publisher’s."],
        licenseNotes: ["Anima derives from NVIDIA’s Cosmos-Predict2, under the NVIDIA Open Model License."],
        // The Compact is Bedovyy's, the image model alone.
        sources: [.compact: "Siliconed ships no weights. The Compact version’s image model is the publisher’s, republished in 8 bits by Bedovyy and downloaded from there; everything else comes from the publisher. The publisher’s license applies, and Bedovyy’s model card says what it changed."],
        licenseName: "CircleStone Labs Non-Commercial")
}
