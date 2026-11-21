// What the interface says of Z-Image: English sentences, the keys of the app's catalog (`tools/translations.sh` reads them here).

import Foundation

extension FamilyPresentation {
    package static let zImage = FamilyPresentation(
        blurb: "Fast and photographic. The one to start with.",
        details: [.compact: "8-bit weights published by third parties: the image model from Comfy-Org (int8 convrot), the text encoder from Disty0 (SDNQ int8). Tokenizer and VAE are the publisher’s."])
}
