// **The "Models" sheet: what the app has installed, what it weighs, and what it takes to render it.**
//
// The app lives in its own space (`Library.standard`) and fills it itself: a family
// is installed from its publisher (downloaded at a pinned revision, forged, sources discarded), a model or
// a LoRA from Civitai is imported, and everything can be removed. Each row states its place on disk — or,
// before installing, the one it will take. Everything that goes can be redone: that is the rule that allows
// removing without remorse.
//
// Nothing is removed during a render or a forge (both read and write GBs);
// an installation stops mid-download, keeping the files already downloaded (the one in progress starts over).
//
// **Nothing comes preinstalled, and nothing is downloaded before the user has read**: « Install »
// opens, under the model's row, its license and the place it will take; « Accept and Install » records
// the acceptance in the library (`Library.acceptLicense`), then downloads. The families shown are the
// visible ones (Z-Image, Qwen-Image-2.1; all six in Developer Mode) — the library itself knows them all.

import AppKit
import Siliconed
import SwiftUI

struct ModelManagement: View {
    @Bindable var app: AppState
    @Environment(\.dismiss) private var shutDown
    @State private var removal: Removal?

    /// What we are about to remove, for the duration of the confirmation.
    enum Removal: Identifiable {
        case family(Family), model(ModelCard), lora(LoRACard)
        var id: String {
            switch self {
            case .family(let f): "family:" + f.rawValue
            case .model(let m): "model:" + m.id
            case .lora(let l): "lora:" + l.path
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            headerView
            Divider()
            Form {
                Section {
                    ForEach(families) { familyLine($0) }
                } header: {
                    Text("Publishers' models")
                } footer: {
                    Text("Downloaded at a pinned revision and verified (sha256), then forged without changing their precision. Installing takes a little more space while forging.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("Imported models") {
                    if !imported.isEmpty {
                        ForEach(imported, id: \.card.id) { importedLine($0.card, $0.bytes) }
                    } else {
                        Text("None. A Civitai checkpoint of a known family imports as is (⌘O); it borrows its family's encoder and VAE.")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                }
                Section("LoRA") {
                    if !loras.isEmpty {
                        ForEach(loras, id: \.card.id) { loraLine($0.card, $0.bytes) }
                    } else {
                        Text("None. A LoRA (.safetensors, as published) imports like a model.")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                }
                if let t = app.occupancy?.downloads, t > 0 {
                    Section {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Interrupted downloads")
                                Text("Resumed at the next installation, or discarded.").font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Text(Figures.disk(t)).monospacedDigit().foregroundStyle(.secondary)
                            Button("Discard") { app.emptyDownloads() }.disabled(!app.managementPossible)
                        }
                    }
                }
            }
            .formStyle(.grouped)
            Divider()
            footer
        }
        .frame(minWidth: 640, idealWidth: 680, minHeight: 560, idealHeight: 700)
        .onAppear { app.refreshCatalog() }
        .onDisappear { app.installOffer = nil }
        .confirmationDialog(removalTitle, isPresented: Binding(get: { removal != nil }, set: { if !$0 { removal = nil } }),
                            presenting: removal) { r in
            Button(removalLabel(r), role: .destructive) { remove(r) }
            Button("Cancel", role: .cancel) {}
        } message: { r in
            Text(removalMessage(r))
        }
    }

    // ── What the sheet lists: the visible families, and what belongs to them ──

    private var families: [Library.Occupancy.FamilyPart] {
        (app.occupancy?.families ?? []).filter { app.isVisible($0.family) }
    }
    private var imported: [(card: ModelCard, bytes: Int)] {
        (app.occupancy?.imported ?? []).filter { app.isVisible($0.card.family) }
    }
    private var loras: [(card: LoRACard, bytes: Int)] {
        (app.occupancy?.loras ?? []).filter { Family(rawValue: $0.card.target).map(app.isVisible) ?? app.developerMode }
    }

    // ── Header and footer ──

    private var headerView: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Models").font(.title2.weight(.semibold))
                    Text(verbatim: app.library.displayPath)
                        .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        .lineLimit(1).truncationMode(.middle)
                }
                Spacer()
                Button("Show in Finder") { app.showInFinder() }
                Button("Report My Configuration…") { app.openDiagnostic() }
                    .help("Measures the installed models on this Mac, then opens a prefilled GitHub issue")
            }
            if app.visibleReady.isEmpty {
                Text("Nothing comes with Siliconed. Choose a model: its license and the space it takes are shown before anything is downloaded.")
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
    }

    private var footer: some View {
        HStack(spacing: 12) {
            if let title = app.forgeInProgress {
                ProgressView().controlSize(.small)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.callout).lineLimit(1)
                    Text(ForgeProgress.line(app.forgeJournal) ?? "…")
                        .font(.caption).monospacedDigit().foregroundStyle(.secondary).lineLimit(1)
                }
                .help(Library.withoutHome(app.forgeJournal.joined(separator: "\n")))
                Spacer()
                Button("Stop") { app.cancelForge() }
                    .help("Stops the installation: the files already downloaded are kept, the one in progress will start over")
            } else {
                if let o = app.occupancy {
                    Text(summary(o)).font(.callout).foregroundStyle(.secondary).monospacedDigit()
                }
                Spacer()
                Button("Import…") { app.chooseToImport() }
                    .disabled(!app.managementPossible)
                    .help("A model (.safetensors or .gguf) or a LoRA (.safetensors), as published on Civitai")
            }
            Button("Done") { shutDown() }
                .keyboardShortcut(.defaultAction)
        }
        .padding(16)
        .background(.bar)
    }

    private func summary(_ o: Library.Occupancy) -> String {
        let free = o.free.map { String(localized: "\(Figures.disk($0)) free on disk") }
        let used = o.total == 0 ? String(localized: "Nothing installed") : String(localized: "\(Figures.disk(o.total)) used")
        return [used, free].compactMap { $0 }.joined(separator: " · ")
    }

    // ── A publisher family ──

    @ViewBuilder private func familyLine(_ p: Library.Occupancy.FamilyPart) -> some View {
        familyRow(p)
        if let offer = app.installOffer, offer.family == p.family { installPanel(p, offer: offer) }
    }

    /// **Before the first byte**: the version (when the family has two), the license, and the place on
    /// disk, then the user's answer.
    private func installPanel(_ p: Library.Occupancy.FamilyPart, offer: AppState.InstallOffer) -> some View {
        let card = ModelCard.of(p.family)
        // What `install` will do, read from the disk now (`Library.installPlan`): the version, what it
        // adds, the version it replaces and whether that one must go before the first byte.
        let plan = try? app.library.installPlan(p.family, variant: offer.variant, baseDiT: offer.baseDiT)
        let bytes = plan?.added ?? toDownload(p, baseDiT: offer.baseDiT, variant: offer.variant)
        let choice = Binding(get: { app.installOffer?.variant ?? offer.variant }, set: { app.installOffer?.variant = $0 })
        return VStack(alignment: .leading, spacing: 12) {
            // What an imported model lacks installs in the version there: no choice (`baseDiT` false).
            if p.family.variants.count > 1, offer.baseDiT {
                VersionChoice(family: p.family, variant: choice)
                if let plan, let replaced = plan.replacing {
                    if plan.removesFirst {
                        Label("Not enough room to keep both: the \(VersionChoice.name(replaced)) version will be removed before the download (\(Figures.disk(plan.freed)) freed). Until the installation is done, \(card.name) cannot render.",
                              systemImage: "exclamationmark.triangle")
                            .font(.callout).foregroundStyle(.orange)
                    } else {
                        Label("Replaces the \(VersionChoice.name(replaced)) version now installed, once the new one is ready: \(Figures.disk(plan.freed)) freed then.",
                              systemImage: "arrow.triangle.2.circlepath")
                            .font(.callout)
                    }
                }
                Divider()
            }
            LicenseTerms(card: card, variant: offer.variant)
            Divider()
            HStack(alignment: .firstTextBaseline) {
                Label {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("\(Figures.disk(bytes)) on disk once installed")
                        if let free = app.occupancy?.free {
                            Text("\(Figures.disk(free)) free on this disk")
                                .font(.caption)
                                .foregroundStyle(plan.map { !$0.fits } ?? (free < bytes) ? Color.red : Color.secondary)
                        }
                    }
                } icon: { Image(systemName: "internaldrive") }
                Spacer()
            }
            .font(.callout)
            // Not enough room even after freeing the other version: the installer would refuse anyway
            // (`diskFull`, nothing removed) — say it here instead of letting the user accept.
            let tooBig = plan.map { !$0.fits } ?? false
            HStack {
                if tooBig {
                    Text("Not enough free space on this disk to install it.")
                        .font(.caption).foregroundStyle(.red)
                }
                Spacer()
                Button("Cancel") { app.installOffer = nil }
                Button("Accept and Install") { app.acceptAndInstall(p.family, variant: offer.variant, baseDiT: offer.baseDiT) }
                    .buttonStyle(.borderedProminent)
                    .disabled(!app.managementPossible || tooBig)
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.accentColor.opacity(0.07)))
    }

    private func familyRow(_ p: Library.Occupancy.FamilyPart) -> some View {
        let card = ModelCard.of(p.family)
        return HStack(alignment: .center, spacing: 12) {
            Image(systemName: icon(p)).foregroundStyle(p.dit > 0 && p.isReady ? .green : .secondary).frame(width: 18)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(card.name).fontWeight(.medium)
                    Text("\(card.defaultSteps) steps").font(.caption).foregroundStyle(.secondary)
                    // The version on disk, said where there is a choice.
                    if p.family.variants.count > 1, p.dit > 0, let v = p.variant {
                        Text(verbatim: VersionChoice.name(v)).font(.caption).foregroundStyle(.secondary)
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .overlay(Capsule().strokeBorder(Palette.hairline))
                            .help(VersionChoice.summary(v))
                    }
                    if !card.license.commercial {
                        Text("non-commercial").font(.caption).foregroundStyle(.orange)
                    }
                }
                Text(state(p)).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if p.present {
                Text(Figures.disk(p.total)).monospacedDigit().foregroundStyle(.secondary)
            }
            familyButtons(p, card: card)
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder private func familyButtons(_ p: Library.Occupancy.FamilyPart, card: ModelCard) -> some View {
        let installed = p.isReady && p.dit > 0
        if installed {
            Button("Use") { app.identifier = card.id; shutDown() }
                .disabled(app.identifier == card.id)
        } else {
            Button(p.present ? "Finish Installing…" : "Install…") { app.offerInstall(p.family) }
                .disabled(!app.managementPossible || app.installOffer?.family == p.family)
                .help(String(localized: "Shows the license and the space it takes; nothing is downloaded before you accept"))
        }
        if p.present {
            Menu {
                // The other version, offered like a first installation: its size, then « Accept and Install ».
                if installed, let current = p.variant {
                    ForEach(p.family.variants.filter { $0 != current }, id: \.self) { other in
                        Button("Switch to \(VersionChoice.name(other))…") { app.offerInstall(p.family, variant: other) }
                    }
                }
                Button("Uninstall…", role: .destructive) { removal = .family(p.family) }
            } label: { Image(systemName: "ellipsis") }
                .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
                .disabled(!app.managementPossible)
        }
    }

    private func icon(_ p: Library.Occupancy.FamilyPart) -> String {
        if p.isReady && p.dit > 0 { return "checkmark.circle.fill" }
        return p.present ? "circle.lefthalf.filled" : "arrow.down.circle"
    }

    /// What the family is on this disk, in one line.
    private func state(_ p: Library.Occupancy.FamilyPart) -> String {
        let card = ModelCard.of(p.family)
        if p.isReady && p.dit > 0 {
            return app.library.isLicenseAccepted(card)
                ? String(localized: "\(card.licenseName) · license accepted")
                : String(localized: "\(card.licenseName) · license not accepted yet: asked at the first render")
        }
        if p.isReady { return String(localized: "Encoder and VAE only — enough to render with an imported model") }
        if p.present { return String(localized: "Incomplete installation — it will resume where it stopped") }
        return String(localized: "≈ \(Figures.disk(toDownload(p))) once installed · \(card.licenseName)")
    }

    /// The space the installation of `variant` will add once done: the parts missing, read from the
    /// disk (`Library.InstallPlan.added`) — an encoder already there counts for nothing, also
    /// Z-Image's Standard one kept by FLUX.2 [klein] while Z-Image is in Compact. The other version
    /// counts for nothing either: it goes once the new one is ready.
    private func toDownload(_ p: Library.Occupancy.FamilyPart, baseDiT: Bool = true, variant: Variant = .standard) -> Int {
        if let plan = try? app.library.installPlan(p.family, variant: variant, baseDiT: baseDiT) { return plan.added }
        let t = p.family.installedSize(variant)
        return (baseDiT ? t.dit : 0) + t.encoder + (p.components > 0 ? 0 : t.components)
    }

    // ── An imported model, a LoRA ──

    private func importedLine(_ f: ModelCard, _ bytes: Int) -> some View {
        HStack(spacing: 12) {
            Image(systemName: app.readySet.contains(f.id) ? "checkmark.circle.fill" : "exclamationmark.circle")
                .foregroundStyle(app.readySet.contains(f.id) ? .green : .orange).frame(width: 18)
            VStack(alignment: .leading, spacing: 3) {
                Text(f.name).fontWeight(.medium)
                Text(app.readySet.contains(f.id)
                     ? String(localized: "\(f.family.name) architecture")
                     : String(localized: "\(f.family.name) architecture — needs its family's encoder and VAE"))
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Text(Figures.disk(bytes)).monospacedDigit().foregroundStyle(.secondary)
            if app.readySet.contains(f.id) {
                Button("Use") { app.identifier = f.id; shutDown() }
                    .disabled(app.identifier == f.id)
            } else {
                Button("Install \(f.family.name)…") { app.offerInstall(f.family, baseDiT: false) }
                    .disabled(!app.managementPossible)
            }
            Button { removal = .model(f) } label: { Image(systemName: "trash") }
                .buttonStyle(.borderless).help("Remove this model")
                .disabled(!app.managementPossible)
        }
        .padding(.vertical, 2)
    }

    private func loraLine(_ l: LoRACard, _ bytes: Int) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "slider.horizontal.3").foregroundStyle(.secondary).frame(width: 18)
            VStack(alignment: .leading, spacing: 3) {
                Text(l.name).fontWeight(.medium).lineLimit(1)
                Text("for \(Family(rawValue: l.target)?.name ?? l.target) · rank \(l.rank)")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Text(Figures.disk(bytes)).monospacedDigit().foregroundStyle(.secondary)
            Button { removal = .lora(l) } label: { Image(systemName: "trash") }
                .buttonStyle(.borderless).help("Remove this LoRA")
                .disabled(!app.managementPossible)
        }
        .padding(.vertical, 2)
    }

    // ── Remove ──

    private var removalTitle: String {
        switch removal {
        case .family(let f)?: String(localized: "Uninstall \(f.name)?")
        case .model(let m)?: String(localized: "Remove \(m.name)?")
        case .lora(let l)?: String(localized: "Remove the LoRA \(l.name)?")
        case nil: ""
        }
    }

    private func removalLabel(_ r: Removal) -> String {
        if case .family = r { return String(localized: "Uninstall") }
        return String(localized: "Remove")
    }

    private func removalMessage(_ r: Removal) -> String {
        switch r {
        case .family(let f):
            let sharing = app.occupancy?.families.first { $0.family == f }?.encoderSharedWith ?? []
            var m = String(localized: "The DiT, encoder and VAE are deleted; installing again downloads them again.")
            if let other = sharing.first {
                m += " " + String(localized: "The text encoder stays: \(other.name) uses it too.")
            }
            m += " " + String(localized: "This family's imported models and LoRAs stay.")
            return m
        case .model:
            return String(localized: "Its forged card is deleted. To get it back, import its file again.")
        case .lora:
            return String(localized: "Its forged card is deleted. To get it back, import its file again.")
        }
    }

    private func remove(_ r: Removal) {
        switch r {
        case .family(let f): app.uninstall(f)
        case .model(let m): app.remove(m)
        case .lora(let l): app.remove(l)
        }
    }
}

/// **Standard, Compact or Light**, before « Install… »: each version with its place
/// on disk, and what the chosen one changes, in one line. The Standard is preselected
/// (below the publisher's weights only by the user's choice), the Light on a Mac of 8 GB
/// (`Family.preselectedVariant`) — preselected, all three shown; the bits live in the tooltip.
struct VersionChoice: View {
    let family: Family
    @Binding var variant: Variant

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Picker("Version", selection: $variant) {
                ForEach(family.variants, id: \.self) { v in
                    let size = family.installedSize(v)
                    Text("\(Self.name(v)) · \(Figures.disk(size.dit + size.encoder + size.components)) on disk").tag(v)
                }
            }
            .pickerStyle(.radioGroup)
            .labelsHidden()
            Text(Self.summary(variant))
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .help(Self.details(variant, family))
        }
    }

    static func name(_ v: Variant) -> String {
        switch v {
        case .standard: String(localized: "Standard")
        case .compact: String(localized: "Compact")
        case .light: String(localized: "Light")
        }
    }

    /// What the version changes, without jargon.
    static func summary(_ v: Variant) -> String {
        switch v {
        case .standard: String(localized: "The publisher's weights, as published.")
        case .compact: String(localized: "Smaller on disk; images slightly different, about as fast.")
        case .light: String(localized: "The smallest on disk, made for Macs with 8 GB; another set of weights, so an image may come out composed differently.")
        }
    }

    /// Whose bytes, in which format — for whoever wants to know.
    static func details(_ v: Variant, _ f: Family) -> String {
        switch (v, f) {
        case (.compact, .zImage):
            String(localized: "8-bit weights published by third parties: the image model from unsloth (GGUF Q8_0), the text encoder from Disty0 (SDNQ int8). Tokenizer and VAE are the publisher's.")
        case (.compact, .qwenImage21):
            String(localized: "8-bit weights published by third parties: the image model from Comfy-Org (int8 convrot), the text encoder from unsloth (int8 convrot). Processor, VAE and turbo LoRA are the publisher's.")
        case (.light, .zImage):
            String(localized: "The image model in 4 to 6 bits, published by unsloth (GGUF Q4_K_M); the Compact's 8-bit text encoder, from Disty0 (SDNQ int8). Tokenizer and VAE are the publisher's.")
        case (.light, .qwenImage21):
            String(localized: "The image model in 4 to 8 bits, published by unsloth (GGUF Q4_K_M); the Compact's 8-bit text encoder, from unsloth (int8 convrot). Processor, VAE and turbo LoRA are the publisher's.")
        default:
            String(localized: "The image model and the text encoder in the publisher's precision (bf16).")
        }
    }
}
