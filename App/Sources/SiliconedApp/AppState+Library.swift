// **`AppState`: the library** — installing and removing a family's versions, the job in progress. The
// state itself is in `AppState.swift`.

import AppKit
import Siliconed
import SwiftUI
import Synchronization
import UniformTypeIdentifiers

extension AppState {
    // MARK: Install

    /// Re-reads the library: after an installation, a removal.
    func refreshCatalog() {
        // A mirror's library is the other Mac's, published (`receiveLibrary`); this disk is not its.
        guard mirror == nil else { return }
        cards = library.cards()
        let before = readySet
        readySet = Set(library.models().map(\.id))
        prepareInBackground(cards.map(\.id).filter { readySet.contains($0) && !before.contains($0) })
        readOccupancy()
        installedLoRAs = library.loras().map { ($0, Library.fileSize($0.path) ?? 0) }
        libraryRead += 1
        modelsRead = [:]
        readModelCatalog()
        readEditors()
    }

    /// **What the library occupies, read off the main thread**: a walk of its store, as long as its disk takes — another disk,
    /// a network share. A read that returns after a later one, or for a library since left, is dropped.
    func readOccupancy() {
        occupancyReads += 1
        let read = occupancyReads, library = library
        Task {
            let o = await Task.detached(priority: .utility) { library.occupancy() }.value
            guard read == occupancyReads, library.root == self.library.root else { return }
            occupancy = o
        }
    }

    /// The chosen model's family, if it must be installed to render.
    var familyToInstall: Family? { missing != nil ? card?.family : nil }

    /// The library's disk must be there: an installation never lands anywhere else.
    var managementPossible: Bool { managementRefusal == nil }

    /// **Why the library cannot change now**, said where a greyed button would be silent; `nil` when it can.
    var managementRefusal: String? {
        if let localOnlyHelp { return localOnlyHelp }
        return libraryBusy ?? libraryBlocker
    }

    /// **Why the library in use cannot be changed for another now**: a job on it, or a render — its disk may be away.
    var libraryBusy: String? {
        if let libraryJob { return String(localized: "\(libraryJob): this waits until it is done.") }
        if busy { return String(localized: "A render is in progress: the library changes once the queue is done.") }
        return nil
    }

    /// **No render while an installation runs**: one big job at a time on 16 GB — a download writes GBs, a
    /// render maps as many. Why « Generate » and the variations are greyed out, and what `silicontrol add`
    /// answers (`busy`); the queue waits too.
    /// The job named as the footer names it: a move, an import or an upload also hold the queue, and each said « An
    /// installation is running ».
    var installBlocker: String? {
        libraryJob.map { String(localized: "\($0): renders can start once it is done.") }
    }

    /// **Offers to install a family**: the Models sheet opens on it, with its license and the place it
    /// takes on disk. Nothing is downloaded before « Accept and Install » (`acceptAndInstall`).
    /// `variant`: the version preselected in the panel — by default the one rendered, else
    /// `Family.preselectedVariant` (the Compact unless the user picks another).
    func offerInstall(_ family: Family, variant: Variant? = nil) {
        installOffer = InstallOffer(family: family, variant: variant ?? renderedVariant(of: family) ?? family.preselectedVariant)
        managementOpen = true
    }

    /// **The user read the license and the size, and accepted**: the acceptance is recorded in the
    /// library first (the engine will ask for it at every render), then the first byte is downloaded.
    /// `replacing`: the panel said the disk takes this version only in place of the family's others
    /// (`Installer.Plan.fitsReplacing`); they go before the first byte.
    /// `asked`: a borrowing Mac's request (`MirrorLibraryHost`), whose license digest was checked; this
    /// Mac's own offer is not touched.
    func acceptAndInstall(_ family: Family, variant: Variant, replacing: Bool = false, asked: Bool = false) {
        if mirror != nil { installRemotely(family, variant: variant, replacing: replacing); return }
        guard managementPossible else { return }
        do { try library.acceptLicense(ModelCard.of(family)) } catch {
            problem = Problem(error)
            return
        }
        if !asked { installOffer = nil }
        // When the other version goes first, a Stop leaves the family without a whole version — the flash says so.
        let removed = replacing ? installPlan(family, variant: variant).flatMap { $0.fitsReplacing ? $0.others.first : nil } : nil
        let stopped = removed.map { old in
            String(localized: "Stopped — \(family.name) can render again once its installation is finished: the \(VersionChoice.name(old)) version was removed to make room. What was downloaded is kept.")
        }
        launchLibraryJob(String(localized: "Installing \(family.name)"), stopped: stopped, task: .installing(family)) { b, cancellation, journal, progress in
            var installer = Installer(library: b)
            installer.journal = journal
            installer.cancellation = cancellation
            installer.progress = progress
            try installer.install(family, variant: variant, replacing: replacing)
        } end: { [weak self] (_: Void) in
            guard let self else { return }
            // The model just installed becomes the form's, if the form's does not render.
            if !readySet.contains(identifier) { identifier = ModelCard.of(family).id }
            flash(String(localized: "\(family.name) installed"))
        }
    }

    // MARK: Import

    /// What an import takes: a `.safetensors`, a LoRA or a model, as its author published it.
    static func importable(_ url: URL) -> Bool { url.pathExtension.lowercased() == "safetensors" }

    /// Choose `.safetensors` files (LoRA or model, as published on Civitai) and import them.
    func chooseToImport() {
        guard importPossible else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "safetensors") ?? .data]
        panel.allowsMultipleSelection = true
        panel.prompt = String(localized: "Import")
        panel.message = String(localized: "A model or a LoRA, as published.")
        guard panel.runModal() == .OK else { return }
        perform(panel.urls)
    }

    /// What an import run comes back with: the files imported, or the family a model among them needs first —
    /// then nothing was imported.
    enum ImportRun: Sendable {
        case imported([ModelImport.Outcome])
        case needs(Family)
    }

    /// **The gesture « import these files »** (a drop, the open panel, `silicontrol import`): `false`, and the reason said,
    /// when the library cannot take them now — a render or an installation running, its disk away.
    @discardableResult
    func perform(_ urls: [URL]) -> Bool {
        // In a mirror the file is not imported here: it goes up to the other Mac (`MirrorUpload.swift`).
        if mirror != nil { upload(urls); return true }
        let files = urls.filter(Self.importable)
        guard !files.isEmpty else { return false }
        guard managementPossible else {
            problem = libraryReachable
                ? Problem(String(localized: "Import is possible once the render or installation in progress is done."))
                : Problem(EngineError.libraryAbsent(path: library.root.path, volume: library.volumeName))
            return false
        }
        importFiles(files)
        return true
    }

    /// **Imports `.safetensors` files, one after the other**: each recognized, then copied as it is. A model dropped
    /// before its family is installed imports nothing: the family's installation is offered, and the file is brought
    /// again once it is done (an import downloads nothing). « Stop » stops the copy under way (`ModelImport.copyIn`):
    /// nothing half-copied stays in the library, the files already imported stay, and the user's own are only read.
    func importFiles(_ files: [URL]) {
        let names = files.map(\.lastPathComponent).joined(separator: ", ")
        let stopped = files.count == 1
            ? String(localized: "Import stopped: nothing was copied, and your file is untouched.")
            : String(localized: "Import stopped: the files already imported stay, nothing half-copied is kept, and your files are untouched.")
        launchLibraryJob(String(localized: "Importing \(names)"), stopped: stopped, task: .importing) { b, cancellation, _, _ in
            let installer = Installer(library: b)
            // A file another Mac uploaded (`UploadHub`) was only there for this import: alone in a folder of its
            // own, which goes with it — imported, refused, or waiting for its family.
            let uploaded = UploadStore.directory(in: b).path + "/"
            defer {
                for f in files where f.path.hasPrefix(uploaded) { try? FileManager.default.removeItem(at: f.deletingLastPathComponent()) }
            }
            for f in files {
                if case (.model, let family) = try ModelImport.recognize(f.path), installer.installed(family).isEmpty { return ImportRun.needs(family) }
            }
            return ImportRun.imported(try files.map { try ModelImport.importFile($0.path, in: b, cancellation: cancellation) })
        } end: { [weak self] (run: ImportRun) in
            guard let self else { return }
            guard case .imported(let results) = run else {
                if case .needs(let family) = run { offerInstall(family) }
                return
            }
            // One toast for the run, each file said in it: a toast per file replaced the one before, and only the last was read.
            var said: [String] = []
            for r in results {
                if let id = r.identifier {
                    identifier = id
                    // One acceptance: the family's covers the model built on its encoder and VAE.
                    if library.isLicenseAccepted(ModelCard.of(r.family)), let card = cards.first(where: { $0.id == id }) {
                        try? library.acceptLicense(card)
                    }
                    said.append(String(localized: "Model imported: \(r.name)"))
                } else if compatibleLoras.contains(where: { $0.path == r.path }) && loras.count < Self.maxLoRA {
                    loras.append(LoRASlot(path: r.path))
                    said.append(String(localized: "LoRA imported: \(r.name)"))
                } else {
                    said.append(String(localized: "LoRA imported for \(ModelCard.of(r.family).name): \(r.name)"))
                }
            }
            if !said.isEmpty { flash(said.joined(separator: " · ")) }
        }
    }

    /// **A gesture on the library, at once**: in a mirror, asked of the lender (`ask`; `nil`, a gesture of this Mac's only);
    /// here, done when the library can change now — `act` changes it and says what to flash, if anything —, the catalog
    /// read again; an error said.
    private func libraryGesture(asking ask: RemoteLibraryRequest?, _ act: () throws -> String?) {
        if mirror != nil { if let ask { askLibrary(ask) }; return }
        guard managementPossible else { return }
        do {
            let said = try act()
            refreshCatalog()
            if let said { flash(said) }
        } catch { problem = Problem(error) }
    }

    /// Removes an imported model: its file. The form goes back to its family's own model.
    func remove(_ card: ModelCard) {
        libraryGesture(asking: .removeModel(card.id)) {
            try ModelImport.remove(card, in: library)
            if identifier == card.id { identifier = ModelCard.of(card.family).id }
            return String(localized: "Model \(card.name) removed")
        }
    }

    /// **Removes a LoRA of the library**: its file, and the LoRA from the stack and from a comparison where it was.
    /// In a mirror the lender removes it once its user agrees (`MirrorLibraryHost`); its library news takes it off this sheet.
    func remove(_ lora: LoRACard) {
        libraryGesture(asking: .removeLoRA(lora.path)) {
            try library.removeLoRA(lora)
            loras.removeAll { $0.path == lora.path }
            for i in brackets.indices { brackets[i].loras.removeAll { $0 == lora.path } }
            return String(localized: "LoRA \(lora.name) removed")
        }
    }

    /// The model chosen in the form is not installed: offer its family.
    func installSelectedModel() {
        guard let f = familyToInstall else { return }
        offerInstall(f)
    }

    /// **What « Stop » does to the job under way**, its tooltip: an installation keeps what it downloaded, an import keeps
    /// nothing half-copied and never touches the user's file.
    var libraryStopHelp: String {
        switch libraryJobTask {
        case .importing: String(localized: "Stops the import: nothing half-copied enters the library, and no file of yours is touched")
        case .installing: String(localized: "Stops the installation: what was downloaded is kept, and the next installation resumes from it")
        // The one job of its kind: the library's move (`moveLibrary`).
        case .other: String(localized: "Stops the move: the library stays where it was, untouched")
        }
    }

    /// Stops the installation in progress; what was downloaded is kept, and the next attempt resumes from it. An import
    /// stops its copy (`importFiles`).
    func cancelLibraryJob() {
        // In a mirror, Stop stops this Mac's own upload if one runs (`MirrorUpload.swift`), else the lender's job.
        if let m = mirror, m.uploads.current == nil {
            if m.uploads.waiting { stopWaitingUploads() } else { askLibrary(.cancel) }
            return
        }
        libraryJobCancellation?.cancel()
    }

    /// **Renders the family with another of its versions whole on disk** (`Installer.use`).
    func use(_ family: Family, variant: Variant) {
        libraryGesture(asking: .use(family, variant)) {
            try Installer(library: library).use(family, variant: variant)
            return String(localized: "\(family.name) now renders with its \(VersionChoice.name(variant)) version")
        }
    }

    /// Uninstalls a version of a family — what the other installed version reads stays — or the whole family (`nil`).
    func uninstall(_ family: Family, variant: Variant? = nil) {
        libraryGesture(asking: .uninstall(family, variant)) {
            try Installer(library: library).uninstall(family, variant: variant)
            return variant.map { String(localized: "\(VersionChoice.name($0)) version of \(family.name) removed") }
                ?? String(localized: "\(family.name) uninstalled")
        }
    }

    func emptyDownloads() {
        libraryGesture(asking: .discard) {
            try library.emptyDownloads()
            return nil
        }
    }

    /// Deletes what version 0.21 left in the library and this one never reads (`Library.legacyFiles`) — on this Mac only.
    func releaseLegacy() {
        libraryGesture(asking: nil) {
            let freed = try library.releaseLegacy()
            return String(localized: "\(Figures.disk(freed)) freed")
        }
    }

    /// The library folder, in the Finder.
    func showInFinder() {
        guard mirror == nil else { return }
        NSWorkspace.shared.activateFileViewerSelecting([library.root])
    }

    /// A job on the library in the background: it reads and writes GBs, the main thread does not wait.
    /// `stopped`: what « Stop » flashes, when « what was downloaded is kept » is not all.
    /// The job is handed the library, its token, its journal (a terminal's rows: the tooltip) and where an
    /// installation tells its progress — heard here ten times a second at most.
    func launchLibraryJob<T: Sendable>(
        _ title: String, stopped: String? = nil, task: RemoteLibraryJob.Task = .other,
        _ job: @escaping @Sendable (Library, Cancellation, @escaping @Sendable (String) -> Void,
                                    @escaping @Sendable (Installer.Progress) -> Void) throws -> T,
        end: @escaping @MainActor (T) -> Void
    ) {
        libraryJob = title
        libraryJobTask = task
        libraryJobJournal = []
        libraryJobProgress = nil
        problem = nil
        let b = library
        let cancellation = Cancellation()
        libraryJobCancellation = cancellation
        Task.detached(priority: .userInitiated) { [weak self] in
            let journal: @Sendable (String) -> Void = { rowLine in Task { @MainActor in self?.libraryJobJournal.append(rowLine) } }
            let heard = Mutex(Date.distantPast)
            let progress: @Sendable (Installer.Progress) -> Void = { p in
                let now = Date()
                // The first and the last word of a file always pass: its name, then its bytes whole.
                let due = heard.withLock { last in
                    guard now.timeIntervalSince(last) >= 0.1 || p.received == 0 || p.received == p.total else { return false }
                    last = now
                    return true
                }
                if due { Task { @MainActor in if self?.libraryJob != nil { self?.libraryJobProgress = p } } }
            }
            // Off the cooperative pool: a job waits on its downloads (`blocking`).
            let result: Result<T, Error>
            do { result = .success(try await offThePool { try job(b, cancellation, journal, progress) }) } catch { result = .failure(error) }
            await MainActor.run {
                guard let self else { return }
                self.libraryJob = nil
                self.libraryJobProgress = nil
                self.libraryJobCancellation = nil
                self.refreshCatalog()
                switch result {
                case .success(let r): end(r)
                case .failure(let e):
                    // « Stop » cancels the token: the engine throws `EngineError.cancelled`.
                    if EngineError(e) == .cancelled {
                        self.flash(stopped ?? String(localized: "Stopped — what was downloaded is kept: the installation resumes where it stopped."))
                    } else {
                        self.problem = Problem(e)
                    }
                }
                // Jobs that entered the queue meanwhile waited for the library's (`next()`).
                self.next()
            }
        }
    }
}
