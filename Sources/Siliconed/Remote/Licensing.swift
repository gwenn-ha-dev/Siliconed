import CryptoKit
import Foundation

/// **A license the lender asks of a client**, whose job waits for it (`RemoteQueue.awaitingLicense`).
/// The sheet opens on the borrowing Mac, built from the same card (both apps are one build); what it
/// cannot know is the version installed on the lender, which decides the links (`variant`).
///
/// **The acceptance is the app's, never `silicontrol`'s**, and carries the digest of what was shown: the
/// lender accepts only if it is the digest of its own text, so that a click can never accept words other
/// than the ones in front of the user. The first answer wins: the lender publishes `nil` when none waits.
package struct RemoteLicense: Codable, Equatable, Sendable {
    package var model: String
    /// `Variant.rawValue` of the family's installed version.
    package var variant: String
    package var digest: Data
    package init(model: String, variant: String, digest: Data) { self.model = model; self.variant = variant; self.digest = digest }
}

/// The borrower's word. `digest` is the borrower's own, of the text it showed; none when declining.
package struct RemoteLicenseAnswer: Codable, Equatable, Sendable {
    package var model: String
    package var accepted: Bool
    package var digest: Data?
    package init(model: String, accepted: Bool, digest: Data?) { self.model = model; self.accepted = accepted; self.digest = digest }
}

package enum LicenseDigest {
    /// **SHA-256 of what the license sheet states**, language-independent: the card's license (its name,
    /// the commercial and filter flags, every link at its pinned revision), the family and version it
    /// applies to, whether the DiT is imported, and Krea 2's Agreement whole when it is shown. The sheet's
    /// own sentences are fixed by the build, which both Macs share; these are the parts that vary.
    package static func of(_ card: ModelCard, variant: Variant) -> Data {
        var lines = ["license/1", card.id, card.family.rawValue, variant.rawValue, card.isImported ? "imported" : "published",
                     card.license.text, "commercial=\(card.license.commercial)", "filter=\(card.license.requiredFilter)"]
        lines += card.family.licenseURLs(variant).map(\.absoluteString)
        if card.family == .krea2 { lines += [Krea2License.attribution, Krea2License.text] }
        return Data(SHA256.hash(data: Data(lines.joined(separator: "\n").utf8)))
    }
}
