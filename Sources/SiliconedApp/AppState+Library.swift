// **`AppState`: the library** — importing, installing, removing models, the forge in progress. The
// state itself is in `AppState.swift`.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

extension AppState {
    // MARK: Import, install

    /// Re-reads the library: after an import, an installation, a removal.
    func refreshCatalog() {
        cards = library.cards()
        readySet = Set(library.models().map(\.id))
        occupancy = library.occupancy()
        modelsRead = [:]
        readModelCatalog()
        readEditors()
    }

    /// The chosen model's family, if it must be installed to render — an imported model requires
    /// only its components and the encoder.
    var familyToInstall: Family? { missing != nil ? card?.family : nil }

    /// A forge, a render, a diagnostic: one waits for the other (the first two read or write GBs, the
    /// diagnostic would time them).
    /// The library's disk must be there: an installation never lands anywhere else.
    var managementPossible: Bool { mirror == nil && forgeInProgress == nil && !busy && !diagnosticRunning && libraryReachable }

    /// **No render while an installation or an import runs**: one big job at a time on 16 GB — the
    /// forge reads and writes GBs, a render maps as many. Why « Generate », « Explore » and the
    /// variations are greyed out, and what `silicontrol add` answers (`busy`); the queue waits too.
    var installBlocker: String? {
        forgeInProgress == nil ? nil : String(localized: "An installation is running: renders can start once it is done.")
    }

    /// What an import takes: a `.safetensors` (a LoRA, or a DiT in bf16, fp16, fp32 or 8 bits as
    /// published) or a `.gguf` (a DiT, 4 bits at least). The forge refuses the rest and says why.
    static let importExtensions: Set<String> = ["safetensors", "gguf"]
    static func importable(_ url: URL) -> Bool { importExtensions.contains(url.pathExtension.lowercased()) }

    /// Choose a `.safetensors` or a `.gguf` (LoRA or model, as published on Civitai) and import it.
    func chooseToImport() {
        guard managementPossible else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = Self.importExtensions.sorted().map { UTType(filenameExtension: $0) ?? .data }
        panel.allowsMultipleSelection = true
        panel.prompt = String(localized: "Import")
        panel.message = String(localized: "A model or a LoRA, as published.")
        guard panel.runModal() == .OK else { return }
        perform(panel.urls)
    }

    /// What an import run comes back with: the files imported, or the family a model among them
    /// needs first — then nothing was imported.
    enum ImportRun: Sendable {
        case imported([ModelImport.Outcome])
        case needs(Family)
    }

    /// Imports `.safetensors` and `.gguf` files, one after the other. The file brought in may be discarded afterwards.
    ///
    /// **A checkpoint dropped before its family**: the engine refuses it (`importNeedsFamily`, the
    /// encoder and VAE are the family's). The app reads the file's family first and, if missing,
    /// offers the installation of what the model lacks — the family's license, its encoder and VAE,
    /// never its DiT — then imports the files once it is done (`InstallOffer.thenImport`).
    func perform(_ urls: [URL]) {
        let files = urls.filter(Self.importable)
        guard !files.isEmpty, managementPossible else { return }
        let names = files.map(\.lastPathComponent).joined(separator: ", ")
        launchForge(String(localized: "Importing \(names)")) { b, cancellation, journal in
            let installer = Installer(library: b)
            for f in files {
                if case (.model, let family?) = try ModelImport.recognize(f.path), !installer.isReady(family) {
                    return ImportRun.needs(family)
                }
            }
            // A file downloaded from the Hub (`importFromHub`) was only there for this import —
            // imported or refused, it goes; one waiting for its family above stays.
            let hub = HubImport.downloads(in: b) + "/"
            defer { for f in files where f.path.hasPrefix(hub) { try? FileManager.default.removeItem(at: f) } }
            return ImportRun.imported(try files.map {
                try ModelImport.importFile($0.path, in: b, cancellation: cancellation, journal: journal)
            })
        } end: { [weak self] (run: ImportRun) in
            guard let self else { return }
            guard case .imported(let results) = run else {
                if case .needs(let family) = run { offerInstall(family, baseDiT: false, thenImport: files) }
                return
            }
            for r in results {
                if r.kind == .encoder || r.kind == .vae {
                    let names = r.families.map(\.name).joined(separator: ", ")
                    if r.kind == .vae {
                        flash(r.placed ? String(localized: "VAE kept for \(names): its installation will not download it")
                                       : String(localized: "VAE already installed for \(names)"))
                    } else {
                        flash(r.placed ? String(localized: "Encoder kept for \(names): its installation will not download it")
                                       : String(localized: "Encoder already installed for \(names)"))
                    }
                } else if let id = r.identifier {
                    identifier = id
                    // One acceptance: the family's, read before its encoder and VAE were installed,
                    // covers the model built on them — no second sheet at the first render.
                    if library.isLicenseAccepted(ModelCard.of(r.family)), let card = cards.first(where: { $0.id == id }) {
                        try? library.acceptLicense(card)
                    }
                    flash(String(localized: "Model imported: \(r.name)"))
                } else if compatibleLoras.contains(where: { $0.path == r.path }) && loras.count < Self.maxLoRA {
                    loras.append(LoRASlot(path: r.path))
                    flash(String(localized: "LoRA imported: \(r.name)"))
                } else {
                    flash(String(localized: "LoRA imported for \(r.family.name): \(r.name)"))
                }
            }
        }
    }

    /// **Offers to install a family**: the Models sheet opens on it, with its license and the place it
    /// takes on disk. Nothing is downloaded before « Accept and Install » (`acceptAndInstall`).
    /// `variant`: the version preselected in the panel — by default the one installed, else
    /// `Family.preselectedVariant` (the Compact unless the user picks another; the
    /// Light preselected on a Mac of 8 GB, shown among the three).
    /// `baseDiT` false (what an imported model lacks) never changes the version: the one installed is
    /// imposed, Standard if none, whatever `variant` says (`Library.installPlan`).
    /// `thenImport`: the files to import once installed (`perform`).
    func offerInstall(_ family: Family, baseDiT: Bool = true, variant: Variant? = nil, thenImport: [URL] = []) {
        let installed = library.installedVariant(of: family)
        installOffer = InstallOffer(family: family, baseDiT: baseDiT,
                                    variant: baseDiT ? variant ?? installed ?? family.preselectedVariant() : installed ?? .standard,
                                    thenImport: thenImport)
        managementOpen = true
    }

    /// **The user read the license and the size, and accepted**: the acceptance is recorded in the
    /// library first (the engine will ask for it at every render), then the first byte is downloaded.
    /// `baseDiT` false: what an imported model of the family lacks (encoder, VAE), not its DiT.
    /// The other version of the family, if installed, is replaced (`Library.install`).
    func acceptAndInstall(_ family: Family, variant: Variant, baseDiT: Bool = true) {
        guard managementPossible else { return }
        do { try library.acceptLicense(ModelCard.of(family)) } catch {
            problem = Problem(error)
            return
        }
        let thenImport = installOffer?.thenImport ?? []
        installOffer = nil
        // The plan the panel showed: when the old version must go first, a Stop leaves the family
        // without a complete version — the flash says so instead of « Stopped — … kept » alone.
        let plan = try? library.installPlan(family, variant: variant, baseDiT: baseDiT)
        let stopped = plan.flatMap { p in p.removesFirst ? p.replacing : nil }.map { old in
            String(localized: "Stopped — \(family.name) can render again after a new installation: the \(VersionChoice.name(old)) version was removed to make room. The files already downloaded are kept; the one in progress will start over.")
        }
        launchForge(String(localized: "Installing \(family.name)"), stopped: stopped) { b, cancellation, journal in
            try b.install(family, variant: variant, baseDiT: baseDiT, cancellation: cancellation, journal: journal)
        } end: { [weak self] (_: Void) in
            guard let self else { return }
            // The model just installed becomes the form's, if the form's does not render.
            if !readySet.contains(identifier), baseDiT { identifier = ModelCard.of(family).id }
            flash(String(localized: "\(family.name) installed"))
            if !thenImport.isEmpty { perform(thenImport) }
        }
    }

    /// The model chosen in the form is not installed: offer its family.
    func installSelectedModel() {
        guard let f = familyToInstall else { return }
        offerInstall(f, baseDiT: card?.isImported != true)
    }

    /// Stops the installation or import in progress; the files already downloaded are kept.
    func cancelForge() { cancellationForge?.cancel() }

    /// Uninstalls a family (DiT, components, encoder if not shared). Its imported
    /// models and its LoRAs remain.
    func uninstall(_ family: Family) {
        guard managementPossible else { return }
        do {
            try library.uninstall(family)
            refreshCatalog()
            flash(String(localized: "\(family.name) uninstalled"))
        } catch { problem = Problem(error) }
    }

    /// Removes an imported model (its map; the original file was already no longer read).
    func remove(_ f: ModelCard) {
        guard f.isImported, managementPossible else { return }
        do {
            try ModelImport.remove(f)
            refreshCatalog()
            if identifier == f.id { identifier = ModelCard.of(f.family).id }
            flash(String(localized: "Model \(f.name) removed"))
        } catch { problem = Problem(error) }
    }

    /// Removes a forged LoRA, and from the stack if it was there.
    func remove(_ lora: LoRACard) {
        guard managementPossible else { return }
        do {
            try ModelImport.remove(lora)
            loras.removeAll { $0.path == lora.path }
            refreshCatalog()
            flash(String(localized: "LoRA \(lora.name) removed"))
        } catch { problem = Problem(error) }
    }

    func emptyDownloads() {
        guard managementPossible else { return }
        do { try library.emptyDownloads(); refreshCatalog() } catch { problem = Problem(error) }
    }

    /// The library folder, in the Finder.
    func showInFinder() {
        NSWorkspace.shared.activateFileViewerSelecting([library.root])
    }

    /// A forge in the background: it reads and writes GBs, the main thread does not wait.
    /// `stopped`: what « Stop » flashes, when « the files already downloaded are kept » is not all.
    func launchForge<T: Sendable>(_ title: String, stopped: String? = nil,
                                          _ job: @escaping @Sendable (Library, Cancellation, @escaping @Sendable (String) -> Void) throws -> T,
                                          end: @escaping @MainActor (T) -> Void) {
        forgeInProgress = title
        forgeJournal = []
        problem = nil
        // A copy to this Mac would read the library beside it: the model's choice will start it again.
        warmCancellation?.cancel()
        let b = library
        let cancellation = Cancellation()
        cancellationForge = cancellation
        Task.detached(priority: .userInitiated) { [weak self] in
            let journal: @Sendable (String) -> Void = { rowLine in Task { @MainActor in self?.forgeJournal.append(rowLine) } }
            let result = Result { try job(b, cancellation, journal) }
            await MainActor.run {
                guard let self else { return }
                self.forgeInProgress = nil
                self.cancellationForge = nil
                self.refreshCatalog()
                switch result {
                case .success(let r): end(r)
                case .failure(let e):
                    // « Stop » cancels the token: the engine throws `EngineError.cancelled`.
                    if EngineError(e) == .cancelled {
                        self.flash(stopped ?? String(localized: "Stopped — the files already downloaded are kept; the one in progress will start over"))
                    } else {
                        self.problem = Problem(e)
                    }
                }
                // Jobs that entered the queue meanwhile waited for the forge (`next()`).
                self.next()
                self.warm()
            }
        }
    }
}
