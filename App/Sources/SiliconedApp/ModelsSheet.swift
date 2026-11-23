// **The "Models" sheet: what the app has installed, what it weighs, and what it takes to render it.**
//
// The app lives in its own space (`Library.standard`) and fills it itself: a family is installed from its
// publishers' repositories — each file downloaded at a pinned revision and verified, as published — in one
// version or both, and everything can be removed. Each row states its place on disk — or, before
// installing, the one it will take. Everything that goes can be downloaded again: that is the rule that
// allows removing without remorse.
//
// Nothing is removed during a render or an installation (both read and write GBs); an installation stops
// mid-download, keeping what it downloaded: the next attempt resumes from it.
//
// **Nothing comes preinstalled, and nothing is downloaded before the user has read**: « Install »
// opens, under the model's row, its license and the place it will take; « Accept and Install » records
// the acceptance in the library (`Library.acceptLicense`), then downloads. The families are shown in the
// order of `AppState.frontFamilies`.

import AppKit
import Siliconed
import SwiftUI

struct ModelManagement: View {
    @Bindable var app: AppState
    @Environment(\.dismiss) private var shutDown
    @State private var removal: Removal?
    @State private var importedRemoval: ModelCard?
    @State private var loraRemoval: LoRACard?
    @State private var legacyRemoval = false

    /// What we are about to remove, for the duration of the confirmation: a family, or one of its versions.
    struct Removal: Identifiable {
        let family: Family
        var variant: Variant?
        var id: String { family.rawValue + "/" + (variant?.rawValue ?? "") }
    }

    var body: some View {
        VStack(spacing: 0) {
            headerView
            Divider()
            Form {
                Section {
                    ForEach(families) { familyLine($0) }
                } header: {
                    Text("Publishers’ models")
                } footer: {
                    Text("Each file is downloaded at a pinned revision and verified (sha256): the weights are the publishers’ files, as published.")
                        .font(.caption).foregroundStyle(Palette.secondary)
                }
                if !imported.isEmpty {
                    Section("Imported models") {
                        ForEach(imported) { importedLine($0.card, $0.bytes) }
                    }
                }
                if !app.installedLoRAs.isEmpty {
                    Section("LoRA") {
                        ForEach(app.installedLoRAs, id: \.card.path) { loraLine($0.card, $0.bytes) }
                    }
                }
                if let t = app.occupancy?.legacy, t > 0 {
                    Section {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Files of version 0.21")
                                Text("Siliconed no longer reads them: it renders with the publishers’ files, as published.")
                                    .font(.caption).foregroundStyle(Palette.secondary)
                            }
                            Spacer()
                            Text(Figures.disk(t)).monospacedDigit().foregroundStyle(Palette.secondary)
                            Button("Free…") { legacyRemoval = true }
                                .unavailable(app.mirror.map { String(localized: "Only “\($0.macName)” frees the files of version 0.21 of its library.") }
                                             ?? app.managementRefusal)
                        }
                    }
                }
                if let t = app.occupancy?.downloads, t > 0 {
                    Section {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Interrupted downloads")
                                Text("Resumed at the next installation, or discarded.").font(.caption).foregroundStyle(Palette.secondary)
                            }
                            Spacer()
                            Text(Figures.disk(t)).monospacedDigit().foregroundStyle(Palette.secondary)
                            Button("Discard") { app.emptyDownloads() }.unavailable(app.libraryRefusal)
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
        .confirmationDialog("Delete the files of version 0.21?", isPresented: $legacyRemoval) {
            Button("Remove", role: .destructive) { app.releaseLegacy() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Nothing is migrated: what this version renders with is downloaded from its publisher.")
        }
        .confirmationDialog(importedRemoval.map { String(localized: "Remove \($0.name)?") } ?? "",
                            isPresented: Binding(get: { importedRemoval != nil }, set: { if !$0 { importedRemoval = nil } }),
                            presenting: importedRemoval) { card in
            Button("Remove", role: .destructive) { app.remove(card) }
            Button("Cancel", role: .cancel) {}
        } message: { _ in
            Text("Its file is deleted from the library. To get it back, import it again.")
        }
        .confirmationDialog(loraRemoval.map { String(localized: "Remove the LoRA \($0.name)?") } ?? "",
                            isPresented: Binding(get: { loraRemoval != nil }, set: { if !$0 { loraRemoval = nil } }),
                            presenting: loraRemoval) { lora in
            Button("Remove", role: .destructive) { app.remove(lora) }
            Button("Cancel", role: .cancel) {}
        } message: { _ in
            Text("Its file is deleted from the library. To get it back, import it again.")
        }
        .confirmationDialog(removalTitle, isPresented: Binding(get: { removal != nil }, set: { if !$0 { removal = nil } }),
                            presenting: removal) { r in
            Button(r.variant == nil ? "Uninstall" : "Remove", role: .destructive) { app.uninstall(r.family, variant: r.variant) }
            Button("Cancel", role: .cancel) {}
        } message: { r in
            Text(r.variant == nil ? "The DiT, encoder and VAE are deleted; installing again downloads them again."
                                  : "The files this version alone reads are deleted; installing it again downloads them again.")
        }
    }

    // ── An imported model ──

    private var imported: [Library.Occupancy.ImportedPart] {
        AppState.inDisplayOrder(app.occupancy?.imported ?? [], family: \.card.family)
    }

    private func importedLine(_ card: ModelCard, _ bytes: Int) -> some View {
        let ready = app.readySet.contains(card.id)
        return HStack(spacing: 12) {
            Image(systemName: ready ? "checkmark.circle.fill" : "exclamationmark.circle")
                .foregroundStyle(ready ? .green : Palette.caution).frame(width: 18)
            VStack(alignment: .leading, spacing: 3) {
                Text(card.name).fontWeight(.medium)
                Text(ready ? String(localized: "\(card.family.name) architecture")
                           : String(localized: "\(card.family.name) architecture — needs its family’s encoder and VAE"))
                    .font(.caption).foregroundStyle(Palette.secondary)
            }
            Spacer()
            Text(Figures.disk(bytes)).monospacedDigit().foregroundStyle(Palette.secondary)
            if ready {
                useButton(card)
            } else {
                Button("Install…") { app.offerInstall(card.family) }
                    .unavailable(app.libraryRefusal)
            }
            Button { importedRemoval = card } label: { Image(systemName: "trash") }
                .buttonStyle(.borderless).help("Remove this model")
                .accessibilityLabel(Text("Remove this model"))
                .unavailable(app.libraryRefusal)
        }
        .padding(.vertical, 2)
    }

    // ── A LoRA ──

    private func loraLine(_ l: LoRACard, _ bytes: Int) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "slider.horizontal.3").foregroundStyle(Palette.secondary).frame(width: 18)
            VStack(alignment: .leading, spacing: 3) {
                Text(verbatim: l.name).fontWeight(.medium).lineLimit(1)
                let family = Family(rawValue: l.target)?.name ?? l.target
                Text(l.smallestRank == l.rank ? String(localized: "for \(family) · rank \(l.rank)")
                     : String(localized: "for \(family) · ranks \(l.smallestRank)–\(l.rank)"))
                    .font(.caption).foregroundStyle(Palette.secondary)
            }
            .help(l.fromItsMetadata.joined(separator: "\n"))
            Spacer()
            Text(Figures.disk(bytes)).monospacedDigit().foregroundStyle(Palette.secondary)
            Button { loraRemoval = l } label: { Image(systemName: "trash") }
                .buttonStyle(.borderless).help("Remove this LoRA")
                .accessibilityLabel(Text("Remove this LoRA"))
                .unavailable(app.libraryRefusal)
        }
        .padding(.vertical, 2)
    }

    // ── What the sheet lists: the families, in the order every list shows them ──

    private var families: [Library.Occupancy.FamilyPart] {
        AppState.inDisplayOrder(app.occupancy?.families ?? [], family: \.family)
    }

    // ── Header and footer ──

    private var headerView: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Models").font(.title2.weight(.semibold))
                    Text(verbatim: app.libraryTitle)
                        .font(.caption).foregroundStyle(Palette.secondary).textSelection(.enabled)
                        .lineLimit(1).truncationMode(.middle)
                }
                Spacer()
                Button("Import…") { app.chooseToImport() }
                    .help("A model or a LoRA, as published.")
                    .unavailable(app.importRefusal)
                Button("Show in Finder") { app.showInFinder() }
                    // The library's disk absent, the Finder has nothing to show: greyed, as in Settings.
                    .unavailable(app.localOnlyHelp ?? app.libraryBlocker)
            }
            if app.visibleReady.isEmpty {
                Text("Nothing comes with Siliconed. Choose a model: its license and the space it takes are shown before anything is downloaded.")
                    .font(.callout).foregroundStyle(Palette.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
    }

    private var footer: some View {
        HStack(spacing: 12) {
            if let title = app.libraryJob {
                // The installation's bytes, of those to come; the wheel alone before its first file, and for a move.
                if let fraction = app.libraryJobFraction {
                    ProgressView(value: fraction).progressViewStyle(.circular).controlSize(.small)
                } else {
                    ProgressView().controlSize(.small)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.callout).lineLimit(1)
                    Text(app.libraryJobLine ?? "…")
                        .font(.caption).monospacedDigit().foregroundStyle(Palette.secondary).lineLimit(1)
                }
                .help(app.libraryJobHelp)
                Spacer()
                Button("Stop") { app.cancelLibraryJob() }
                    .help(app.libraryStopHelp)
            } else if app.remoteConfirming {
                ProgressView().controlSize(.small)
                Text("Waiting for confirmation on “\(app.libraryTitle)”").font(.callout).foregroundStyle(Palette.secondary)
                Spacer()
            } else if let problem = app.problem {
                // **What went wrong, where the eye is**: the sheet covers the column's foot, where the problem is said too — an
                // interrupted download, a full disk. It stands in place of the summary until it is closed or another job starts.
                ProblemView(problem: problem) { app.problem = nil }
            } else {
                if let o = app.occupancy {
                    Text(summary(o)).font(.callout).foregroundStyle(Palette.secondary).monospacedDigit()
                }
                Spacer()
            }
            Button("Done") { shutDown() }
                .keyboardShortcut(.defaultAction).tint(Palette.control)
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

    /// **Before the first byte**: the version (when the family has several), the license, and the place on
    /// disk, then the user's answer.
    private func installPanel(_ p: Library.Occupancy.FamilyPart, offer: AppState.InstallOffer) -> some View {
        let card = ModelCard.of(p.family)
        // What the installation will do, read from the disk now (`Installer.plan`): what is left to download,
        // and whether the disk takes it only in place of the family's other version.
        let plan = app.installPlan(p.family, variant: offer.variant)
        let bytes = plan?.toDownload ?? p.family.installedSize(offer.variant)
        let choice = Binding(get: { app.installOffer?.variant ?? offer.variant }, set: { app.installOffer?.variant = $0 })
        return VStack(alignment: .leading, spacing: 12) {
            if p.family.variants.count > 1 {
                VersionChoice(family: p.family, variant: choice)
                if let plan, plan.fitsReplacing, let replaced = plan.others.first {
                    Label("Not enough room to keep both: the \(VersionChoice.name(replaced)) version will be removed before the download (\(Figures.disk(plan.releasable)) freed). Until the installation is done, \(card.name) cannot render.",
                          systemImage: "exclamationmark.triangle")
                        .font(.callout).foregroundStyle(Palette.caution)
                }
                Divider()
            }
            LicenseTerms(card: card, variant: offer.variant)
            Divider()
            // Not enough room even in place of the other version: the installer would refuse anyway
            // (`diskFull`, nothing removed) — said here instead of letting the user accept.
            let tooBig = plan.map { !$0.fits && !$0.fitsReplacing } ?? false
            HStack(alignment: .firstTextBaseline) {
                Label {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("\(Figures.disk(bytes)) to download")
                        if let free = app.occupancy?.free {
                            Text("\(Figures.disk(free)) free on this disk")
                                .font(.caption)
                                .foregroundStyle(tooBig ? Palette.failure : Palette.secondary)
                        }
                    }
                } icon: { Image(systemName: "internaldrive") }
                Spacer()
            }
            .font(.callout)
            HStack {
                if tooBig {
                    Text("Not enough free space on this disk to install it.")
                        .font(.caption).foregroundStyle(Palette.failure)
                }
                Spacer()
                Button("Cancel") { app.installOffer = nil }
                Button("Accept and Install") {
                    app.acceptAndInstall(p.family, variant: offer.variant, replacing: plan?.fitsReplacing ?? false)
                }
                .buttonStyle(.prominent)
                .unavailable(app.libraryRefusal ?? (tooBig ? String(localized: "Not enough free space on this disk to install it.") : nil))
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8).fill(Palette.live.opacity(0.07)))
    }

    private func familyRow(_ p: Library.Occupancy.FamilyPart) -> some View {
        let card = ModelCard.of(p.family)
        return HStack(alignment: .center, spacing: 12) {
            Image(systemName: icon(p)).foregroundStyle(p.isReady ? Color.green : Palette.secondary).frame(width: 18)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(card.name).fontWeight(.medium)
                        .help(Welcome.blurb(p.family))
                    if p.family.isBeta { BetaBadge() }
                    Text("\(card.defaultSteps) steps").font(.caption).foregroundStyle(Palette.secondary)
                    // The version rendered, said where there is a choice.
                    if p.family.variants.count > 1, let v = p.variant {
                        Text(verbatim: VersionChoice.name(v)).font(.caption).foregroundStyle(Palette.secondary)
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .overlay(Capsule().strokeBorder(Palette.hairline))
                            .help(VersionChoice.summary(v, p.family))
                    }
                    if !card.license.commercial {
                        Text("non-commercial").font(.caption).foregroundStyle(Palette.caution)
                    }
                }
                Text(state(p)).font(.caption).foregroundStyle(Palette.secondary)
            }
            Spacer()
            if p.present {
                Text(Figures.disk(p.bytes)).monospacedDigit().foregroundStyle(Palette.secondary)
            }
            familyButtons(p, card: card)
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder private func familyButtons(_ p: Library.Occupancy.FamilyPart, card: ModelCard) -> some View {
        if p.isReady {
            useButton(card)
        } else {
            Button(p.present ? "Finish Installing…" : "Install…") { app.offerInstall(p.family) }
                .help(String(localized: "Shows the license and the space it takes; nothing is downloaded before you accept"))
                .unavailable(app.libraryRefusal
                             ?? (app.installOffer?.family == p.family ? String(localized: "Its license and the space it takes are shown below.") : nil))
        }
        if p.present {
            Menu {
                // The other version: rendered at once when it is whole on disk, else offered like a first
                // installation — its size, then « Accept and Install ».
                if let current = p.variant {
                    ForEach(p.family.variants.filter { $0 != current }, id: \.self) { other in
                        Button("Switch to \(VersionChoice.name(other))…") {
                            if p.installed.contains(other) { app.use(p.family, variant: other) } else { app.offerInstall(p.family, variant: other) }
                        }
                    }
                }
                // Both on disk: either goes alone, what the other reads stays.
                if p.installed.count > 1 {
                    ForEach(p.installed, id: \.self) { v in
                        Button("Remove the \(VersionChoice.name(v)) Version…", role: .destructive) { removal = Removal(family: p.family, variant: v) }
                    }
                }
                Button("Uninstall…", role: .destructive) { removal = Removal(family: p.family) }
            } label: { Image(systemName: "ellipsis") }
                .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
                .help("More Actions")
                .accessibilityLabel(Text("More Actions"))
                .unavailable(app.libraryRefusal)
        }
    }

    /// « Use »: the rack takes the model, and the sheet closes — greyed on the rack's own.
    private func useButton(_ card: ModelCard) -> some View {
        Button("Use") { app.identifier = card.id; shutDown() }
            .unavailable(app.identifier == card.id ? String(localized: "The rack already renders with this model.") : nil)
    }

    private func icon(_ p: Library.Occupancy.FamilyPart) -> String {
        if p.isReady { return "checkmark.circle.fill" }
        return p.present ? "circle.lefthalf.filled" : "arrow.down.circle"
    }

    /// What the family is on this disk, in one line.
    private func state(_ p: Library.Occupancy.FamilyPart) -> String {
        let card = ModelCard.of(p.family)
        if p.isReady {
            return app.licenseAccepted(card)
                ? String(localized: "\(card.licenseName) · license accepted")
                : String(localized: "\(card.licenseName) · license not accepted yet: asked at the first render")
        }
        if p.present { return String(localized: "Incomplete installation — it will resume where it stopped") }
        return String(localized: "≈ \(Figures.disk(p.family.installedSize(p.family.preselectedVariant))) once installed · \(card.licenseName)")
    }

    // ── Remove ──

    private var removalTitle: String {
        guard let r = removal else { return "" }
        return r.variant.map { String(localized: "Remove the \(VersionChoice.name($0)) version of \(r.family.name)?") }
            ?? String(localized: "Uninstall \(r.family.name)?")
    }
}

/// **Standard or Compact**, before « Install… »: each version with its place on disk, and what the chosen
/// one changes, in one line. The Compact is preselected and marked recommended
/// (`Family.preselectedVariant`); the bits live in the tooltip.
struct VersionChoice: View {
    let family: Family
    @Binding var variant: Variant

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Picker("Version", selection: $variant) {
                ForEach(family.variants, id: \.self) { v in
                    let disk = Figures.disk(family.installedSize(v))
                    Group {
                        if v == family.preselectedVariant {
                            Text("\(Self.name(v)) · \(disk) on disk · recommended")
                        } else {
                            Text("\(Self.name(v)) · \(disk) on disk")
                        }
                    }.tag(v)
                }
            }
            .pickerStyle(.radioGroup).tint(Palette.control)
            .labelsHidden()
            // A version's line wraps rather than loses its end where the column is narrow (the welcome's cards).
            .fixedSize(horizontal: false, vertical: true)
            Text(Self.summary(variant, family))
                .font(.caption).foregroundStyle(Palette.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .help(Self.details(variant, family))
        }
    }

    static func name(_ v: Variant) -> String {
        switch v {
        case .standard: String(localized: "Standard")
        case .compact: String(localized: "Compact")
        }
    }

    /// What the version changes, without jargon: the family's own sentence where the general one is not true of it
    /// (`FamilyPresentation.summaries`).
    static func summary(_ v: Variant, _ f: Family) -> String {
        if let own = f.traits.presentation.summaries[v] { return String(localized: own) }
        switch v {
        case .standard: return String(localized: "The publisher’s weights, as published.")
        case .compact: return String(localized: "Smaller on disk; images slightly different, about as fast.")
        }
    }

    /// Whose bytes, in which format — for whoever wants to know (`FamilyPresentation.details`).
    static func details(_ v: Variant, _ f: Family) -> String {
        if let own = f.traits.presentation.details[v] { return String(localized: own) }
        return String(localized: "The image model and the text encoder in the publisher’s precision (bf16).")
    }
}

/// « Beta », same outlined capsule as the version badge, in the orange of « non-commercial » (a caveat, not an error).
/// Shown where a model with a community recipe is chosen (`Family.isBeta`).
struct BetaBadge: View {
    /// What the hover says: by default, why a model is in beta.
    var help: LocalizedStringKey = "Beta: the 8-step recipe is the community’s distillation, not the publisher’s. It leaves beta when the publisher releases its own fast version."
    var body: some View {
        Text("Beta").font(.caption).foregroundStyle(Palette.caution)
            .padding(.horizontal, 5).padding(.vertical, 1)
            .overlay(Capsule().strokeBorder(Palette.caution.opacity(0.6)))
            .help(help)
    }
}
