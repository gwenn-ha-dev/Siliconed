// **A model's license, shown before it is used**.
//
// Two moments, one rule — nothing happens before the user has read and accepted:
//   - **before the first byte of an installation**: the Models sheet shows the license and the place
//     on disk inline (`LicenseTerms`, with a link to each license file at its pinned revision), and « Accept and Install » records the acceptance, then downloads;
//   - **before the first render of a model installed otherwise** (by the developer's command line, by an older app):
//     the engine's preflight refuses it (`EngineError.licenseNotAccepted`), the job waits, and this
//     sheet asks. Accepting records it (`Library.acceptLicense`) and the job runs again.
//
// The app holds no door of its own: it is the engine that refuses a render whose license the library
// does not record. The app only presents.

import Siliconed
import SwiftUI

/// **What the license says, in a few lines**: its name as the publisher gives it, then what it means.
struct LicenseTerms: View {
    let card: ModelCard
    /// The version installed or about to be: a Compact or a Light downloads from more repositories — the
    /// publisher's license files, then the third parties' model cards (`Family.licenseURLs`) — and
    /// its weights are not the publisher's bytes.
    var variant: Variant = .standard

    /// The license files to link, at the revisions the installation takes.
    var urls: [URL] { card.isImported ? card.license.urls : card.family.licenseURLs(variant) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label {
                Text(card.licenseName).font(.headline)
            } icon: {
                Image(systemName: card.license.commercial ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                    .foregroundStyle(card.license.commercial ? .green : .orange)
            }
            VStack(alignment: .leading, spacing: 4) {
                if card.license.commercial {
                    Text("Commercial use is allowed, under the license's terms.")
                } else {
                    Text("Non-commercial: personal use and research only. Images made with it can't be sold or used in a commercial product.")
                }
                if card.license.requiredFilter {
                    Text("The license requires a content filter wherever the model is deployed.")
                }
                Group {
                    if variant == .compact && !card.isImported {
                        Text("Siliconed ships no weights. The Compact version is the publisher's model republished in 8 bits by third parties, and downloaded from them; the publisher's license applies, and each third party's model card says what it changed.")
                    } else if variant == .light && !card.isImported {
                        Text("Siliconed ships no weights. The Light version is the publisher's model republished in 4 to 8 bits by third parties, and downloaded from them; the publisher's license applies, and each third party's model card says what it changed.")
                    } else if card.family.hasThirdPartyTurbo && !card.isImported {
                        Text("Siliconed ships no weights: the model is downloaded from its publisher, and its turbo LoRA from the third party that made it, each under its own license.")
                    } else {
                        Text("Siliconed ships no weights: the model is downloaded from its publisher, under the publisher's license.")
                    }
                }
                .foregroundStyle(.secondary)
                // The license file(s) on Hugging Face, at the revision the installation takes. Several:
                // each named by its repository (Qwen-Image-2.1: the model, then the turbo LoRA).
                let urls = self.urls
                ForEach(urls, id: \.self) { url in
                    Group {
                        if urls.count == 1 {
                            Link("Read the License", destination: url)
                        } else {
                            Link("Read the License: \(Self.repository(url))", destination: url)
                        }
                    }
                    .help(url.absoluteString)
                }
            }
            .font(.callout)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// `Qwen/Qwen-Image-2.1` from `https://huggingface.co/Qwen/Qwen-Image-2.1/blob/<sha>/LICENSE`.
    static func repository(_ url: URL) -> String {
        url.pathComponents.dropFirst().prefix(2).joined(separator: "/")
    }
}

/// **The sheet the engine's refusal opens**: accept, and the waiting render runs.
struct LicenseSheet: View {
    let app: AppState
    let card: ModelCard

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("\(card.name): accept its license to render").font(.title3.weight(.semibold))
            Text("This model is installed, but its license hasn't been accepted in this library yet. The engine renders nothing before.")
                .font(.callout).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            // The links and words of the version on disk: a Compact's third parties too.
            LicenseTerms(card: card, variant: app.library.variant(of: card.family))
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
            HStack {
                Spacer()
                Button("Not Now") { app.declineLicense(card) }
                    .keyboardShortcut(.cancelAction)
                Button("Accept and Render") { app.acceptLicense(card) }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 480)
    }
}
