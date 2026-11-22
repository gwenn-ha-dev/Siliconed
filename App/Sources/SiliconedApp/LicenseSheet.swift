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
    /// The version installed or about to be: a Compact downloads from more repositories — the
    /// publisher's license files, then the third parties' model cards (`Installer.licensePages`) — and
    /// its weights are not the publisher's bytes.
    var variant: Variant = .standard

    /// The license files to link, at the revisions the installation takes. An imported model's are its
    /// family's installed version's: its encoder and VAE are that version's files, a Compact's from
    /// third parties.
    var urls: [URL] { Installer.licensePages(card.family, variant) }
    private var presentation: FamilyPresentation { card.family.traits.presentation }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label {
                Text(card.licenseName).font(.headline)
            } icon: {
                Image(systemName: card.license.commercial ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                    .foregroundStyle(card.license.commercial ? .green : Palette.caution)
            }
            VStack(alignment: .leading, spacing: 4) {
                if card.license.commercial {
                    Text("Commercial use is allowed, under the license’s terms.")
                } else if card.imagesCommercial {
                    // The model is non-commercial, its images are not (`FamilyTraits.commercialImages`).
                    Text("Non-commercial: the model is for personal use and research only. The images it makes may be used for any purpose, commercial use included.")
                } else {
                    Text("Non-commercial: personal use and research only. Images made with it can’t be sold or used in a commercial product.")
                }
                // What the family's license adds: a parent license, a duty on whoever deploys the model.
                ForEach(Array(presentation.licenseNotes.enumerated()), id: \.offset) { Text($0.element) }
                Group {
                    if card.isImported {
                        // Nothing is downloaded: the DiT is the user's file, the rest the family's.
                        Text("The image model is the file you imported, under its author’s license. Its text encoder and VAE are those of the installed version, under the licenses linked below.")
                    } else if let own = presentation.sources[variant] {
                        Text(own)
                    } else if variant == .compact {
                        Text("Siliconed ships no weights. The Compact version is the publisher’s model republished in 8 bits by third parties, and downloaded from them; the publisher’s license applies, and each third party’s model card says what it changed.")
                    } else {
                        Text("Siliconed ships no weights: the model is downloaded from its publisher, under the publisher’s license.")
                    }
                }
                .foregroundStyle(Palette.secondary)
                // The license file(s) on Hugging Face, at the revision the installation takes. Several:
                // each named by its repository (a DiT's, then its Diffusers one).
                let urls = self.urls
                ForEach(urls, id: \.self) { url in
                    Group {
                        if urls.count == 1 {
                            Button("Read the License") { NSWorkspace.shared.open(url) }.buttonStyle(.chartLink)
                        } else {
                            Button("Read the License: \(Self.repository(url))") { NSWorkspace.shared.open(url) }.buttonStyle(.chartLink)
                        }
                    }
                    .help(url.absoluteString)
                }
            }
            .font(.callout)
            .fixedSize(horizontal: false, vertical: true)
            // A family whose license is an Agreement: the text itself, whole, and its attribution notice word for word
            // (`FamilyTraits.agreement`).
            if let agreement = card.family.traits.agreement { WholeAgreement(agreement: agreement) }
        }
    }

    /// `Qwen/Qwen-Image-2.1` from `https://huggingface.co/Qwen/Qwen-Image-2.1/blob/<sha>/LICENSE`.
    static func repository(_ url: URL) -> String {
        url.pathComponents.dropFirst().prefix(2).joined(separator: "/")
    }
}

/// **A family's Agreement, in full, where it is accepted** (`FamilyTraits.agreement`): its attribution notice, then the
/// whole text in a box that scrolls. Quoted, never translated: it is the license.
private struct WholeAgreement: View {
    let agreement: Agreement
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(verbatim: agreement.notice)
                .font(.callout).italic()
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            ScrollView(.vertical) {
                Text(verbatim: agreement.text)
                    .font(.caption)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
            }
            .frame(height: 220)
            .background(Palette.well, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).strokeBorder(Palette.hairline))
        }
    }
}

/// **The sheet the engine's refusal opens**: accept, and the waiting render runs.
struct LicenseSheet: View {
    let app: AppState
    let card: ModelCard

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("\(card.name): accept its license to render").font(.title3.weight(.semibold))
            Text(app.mirror.map { String(localized: "This model is installed on “\($0.macName)”, but its license hasn’t been accepted there yet. Accept it to render with it.") }
                    ?? String(localized: "This model is installed, but its license hasn’t been accepted in this library yet. Accept it to render with it."))
                .font(.callout).foregroundStyle(Palette.secondary)
                .fixedSize(horizontal: false, vertical: true)
            // The links and words of the version on disk: a Compact's third parties too.
            LicenseTerms(card: card, variant: app.licenseVariant(of: card))
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
            HStack {
                Spacer()
                Button("Not Now") { app.declineLicense(card) }
                    .keyboardShortcut(.cancelAction)
                // No default key: a Return typed by habit must not accept a license.
                Button("Accept and Render") { app.acceptLicense(card) }
            }
        }
        .padding(20)
        .frame(width: 480)
    }
}
