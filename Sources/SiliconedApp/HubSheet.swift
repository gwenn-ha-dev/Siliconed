// **The sheet a « Use this model » link opens** (`AppState+Hub.swift`): the repository, its license,
// its files each with what the import would make of it — judged on its header while the sheet is
// open — and one chosen. Nothing is downloaded before « Import ». A file the engine would refuse is
// shown, greyed, with the forge's reason: a repository often mixes models.

import Siliconed
import SwiftUI

struct HubSheet: View {
    let app: AppState

    var body: some View {
        let offer = app.hubOffer
        VStack(alignment: .leading, spacing: 14) {
            Text("From Hugging Face").font(.title3.weight(.semibold))
            if let offer {
                Link(destination: URL(string: "https://huggingface.co/\(offer.link.repository)")!) {
                    Text(verbatim: offer.link.repository).font(.body.monospaced())
                }
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
                if offer?.repository != nil, !app.libraryManageable {
                    Text("Import is possible once the render or installation in progress is done.")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                Button("Cancel") { app.hubOffer = nil }
                    .keyboardShortcut(.cancelAction)
                Button(importTitle(offer)) { app.importFromHub() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!(offer.map { AppState.HubOffer.importable($0.choice?.verdict) } ?? false) || !app.libraryManageable)
            }
        }
        .padding(20)
        .frame(width: 560)
    }

    private func importTitle(_ offer: AppState.HubOffer?) -> String {
        guard let f = offer?.choice?.file else { return String(localized: "Import") }
        return String(localized: "Import (\(ByteCountFormatter.string(fromByteCount: Int64(f.size), countStyle: .file)))")
    }

    @ViewBuilder
    private func repository(_ r: HubRepository, _ offer: AppState.HubOffer) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            if let l = r.license {
                Text("License on its page: \(l). The file is its author's, under that license.")
            } else {
                Text("No license declared on its page: the file is its author's.")
            }
            if r.gated {
                Text("Its author restricts access on Hugging Face: the download needs your account's access to it.")
            }
        }
        .font(.callout).foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)

        if offer.files.isEmpty {
            Text("Nothing in this repository can be imported: Siliconed takes a LoRA or an image model as one .safetensors or .gguf file.")
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
        let importable = AppState.HubOffer.importable(v)
        return Button {
            app.hubOffer?.chosen = f.path
        } label: {
            HStack(spacing: 8) {
                Image(systemName: chosen ? "largecircle.fill.circle" : "circle")
                    .foregroundStyle(chosen ? Color.accentColor : .secondary)
                VStack(alignment: .leading, spacing: 1) {
                    Text(verbatim: f.path).lineLimit(1).truncationMode(.middle)
                    verdict(v).font(.caption).foregroundStyle(.secondary)
                }
                Spacer(minLength: 8)
                Text(ByteCountFormatter.string(fromByteCount: Int64(f.size), countStyle: .file))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            .padding(.vertical, 4).padding(.horizontal, 6)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!importable)
        .opacity(importable || v == nil ? 1 : 0.5)
        .help(refusal(v) ?? f.path)
    }

    @ViewBuilder
    private func verdict(_ v: HubImport.Verdict?) -> some View {
        switch v {
        case nil: Text("Checking…")
        case .lora(let f): Text("LoRA for \(f.name)")
        case .loraNeedsFamily(let f): Text("LoRA for \(f.name), which will be installed first")
        case .model(let f): Text("Image model, \(f.name) family")
        case .unexamined: Text("Checked once downloaded")
        case .refused: Text("Not supported — hover for the reason")
        }
    }

    private func refusal(_ v: HubImport.Verdict?) -> String? {
        if case .refused(let why) = v { why } else { nil }
    }
}
