// **The sheet a « Use this model » link opens** (`AppState+Hub.swift`): the repository, its license, its files each
// with what the import would make of it — judged on its header while the sheet is open — and one chosen. Nothing is
// downloaded before « Import ». A file the import would refuse is shown, greyed, with its reason: a repository often
// mixes models.

import Siliconed
import SwiftUI

struct HubSheet: View {
    let app: AppState

    var body: some View {
        let offer = app.hubOffer
        VStack(alignment: .leading, spacing: 14) {
            Text("From Hugging Face").font(.title3.weight(.semibold))
            if let offer {
                Button { NSWorkspace.shared.open(URL(string: "https://huggingface.co/\(offer.link.repository)")!) } label: {
                    Text(verbatim: offer.link.repository).font(.body.monospaced())
                }
                .buttonStyle(.chartLink)
                if let p = offer.problem {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(p.title)
                        if let s = p.suggestion { Text(s).foregroundStyle(.secondary) }
                    }
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
                } else if let r = offer.repository {
                    repository(r, offer)
                } else {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Reading the repository…").foregroundStyle(.secondary)
                    }
                }
            }
            HStack {
                if offer?.repository != nil, !app.managementPossible {
                    Text("Import is possible once the render or installation in progress is done.")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                Button("Cancel") { app.hubOffer = nil }
                    .keyboardShortcut(.cancelAction)
                Button(actionTitle(offer)) { app.importFromHub() }
                    .keyboardShortcut(.defaultAction).tint(Palette.control)
                    .unavailable(app.managementRefusal
                                 ?? (offer.map { AppState.HubOffer.choosable($0.choice?.verdict) } == true ? nil
                                     : String(localized: "Choose one of the files the import takes.")))
            }
        }
        .padding(20)
        .frame(width: 560)
    }

    private func actionTitle(_ offer: AppState.HubOffer?) -> String {
        guard let choice = offer?.choice else { return String(localized: "Import") }
        if case .modelNeedsFamily(let family) = choice.verdict { return String(localized: "Install \(family.name)…") }
        return String(localized: "Import (\(Figures.disk(choice.file.size)))")
    }

    @ViewBuilder
    private func repository(_ r: HubRepository, _ offer: AppState.HubOffer) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            if let l = r.license {
                Text("License on its page: \(l). The file is its author’s, under that license.")
            } else {
                Text("No license declared on its page: the file is its author’s.")
            }
            if r.gated {
                Text("Its author restricts access on Hugging Face, and Siliconed has no account there: download the file from its page, then import it with File › Import a Model or LoRA…")
            }
        }
        .font(.callout).foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)

        if r.gated {
        } else if offer.files.isEmpty {
            Text("Nothing in this repository can be imported: Siliconed takes a LoRA or an image model as one .safetensors file.")
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            ScrollView(.vertical) {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(offer.files, id: \.file.path) { row in
                        fileRow(row.file, row.verdict, chosen: row.file.path == offer.chosen)
                    }
                }
                .padding(6)
            }
            .frame(maxHeight: 260)
            .fixedSize(horizontal: false, vertical: true)
            .background(Palette.well, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).strokeBorder(Palette.hairline))
            Text("Nothing is downloaded before Import. The file is checked against the fingerprint Hugging Face lists, imported, then deleted.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func fileRow(_ f: HubRepository.File, _ v: HubImport.Verdict?, chosen: Bool) -> some View {
        let choosable = AppState.HubOffer.choosable(v)
        return Button {
            app.hubOffer?.chosen = f.path
        } label: {
            HStack(spacing: 8) {
                Image(systemName: chosen ? "largecircle.fill.circle" : "circle")
                    .foregroundStyle(chosen ? Palette.live : .secondary)
                VStack(alignment: .leading, spacing: 1) {
                    Text(verbatim: f.path).lineLimit(1).truncationMode(.middle)
                    verdict(v).font(.caption).foregroundStyle(.secondary)
                }
                Spacer(minLength: 8)
                Text(Figures.disk(f.size))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            .padding(.vertical, 4).padding(.horizontal, 6)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .opacity(choosable || v == nil ? 1 : 0.5)
        .help(f.path)
        .unavailable(choosable ? nil : refusal(v) ?? String(localized: "Checking…"))
    }

    @ViewBuilder
    private func verdict(_ v: HubImport.Verdict?) -> some View {
        switch v {
        case nil: Text("Checking…")
        // The catalog's name: a LoRA applies to the model the family renders.
        case .lora(let f): Text("LoRA for \(ModelCard.of(f).name)")
        case .model(let f): Text("Image model, \(f.name) family")
        case .modelNeedsFamily(let f): Text("Image model, \(f.name) family, which is to install first")
        case .unexamined: Text("Checked once downloaded")
        case .refused: Text("Not supported — hover for the reason")
        }
    }

    private func refusal(_ v: HubImport.Verdict?) -> String? {
        if case .refused(let why) = v { why } else { nil }
    }
}
