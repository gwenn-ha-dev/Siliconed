// **The lender's library, in the Models sheet of a borrowing Mac** (`Remote/Installing.swift`).
//
// - **Borrower** (`AppState(mirroring:)`): the sheet is the ordinary one, reading what the lender
//   published — `occupancy` is rebuilt from `RemoteLibrary`, the install plans and the accepted licenses
//   come from it, an installation under way is `RemoteLibraryJob` said here. Each gesture is a message instead
//   of an action (`if mirror != nil` at the head of the action); what has no meaning for another Mac's
//   disk (opening or moving a library, the Finder) is greyed, with its reason.
// - **Lender** (`MirrorLibraryHost`): publishes the library and the job to every client — the library is
//   common — and judges each request again as the Models sheet would: the family and version known, the
//   disk, nothing else running, the digest of the license text shown. The download is its own
//   (`launchLibraryJob`); a failure is told to the client that asked.
// - **The license** is accepted on the borrower's sheet, which shows the lender's words (one build on both
//   Macs, `LicenseDigest`); the click goes with the digest and the lender accepts only its own text.

import AppKit
import Foundation
import Siliconed

// MARK: - Borrower

extension AppState {
    /// This library's name where the sheets say where it is.
    var libraryTitle: String { mirror?.macName ?? library.shownPath }

    /// Why a gesture on this library's disk is greyed, in a mirror (`nil`: it is this Mac's).
    var localOnlyHelp: String? {
        mirror.map { String(localized: "This is the library of “\($0.macName)”: only that Mac can open it, move it or show it in the Finder.") }
    }

    /// **May the library change now**: this Mac's (`managementPossible`), or the lender's — nothing of
    /// ours running, no installation under way there; the lender judges again.
    var libraryManageable: Bool { libraryRefusal == nil }

    /// **Why the library cannot change now**, this Mac's or the lender's, said on what it greys; `nil` when it can.
    var libraryRefusal: String? {
        guard let m = mirror else { return managementRefusal }
        if let linkBlocker { return linkBlocker }
        // Not published yet: the link is still being set up.
        guard remoteLibrary != nil else { return String(localized: "Connecting to “\(m.macName)”…") }
        // `libraryJob`, in a mirror: the lender's job as published, or this Mac's own upload (`MirrorUpload.swift`).
        if let job = libraryJob ?? remoteLibraryJob.map({ libraryJobTitle($0.task) }) {
            return String(localized: "\(job): this waits until it is done.")
        }
        if busy { return String(localized: "A render is in progress: the library changes once the queue is done.") }
        if remoteConfirming { return String(localized: "Waiting for confirmation on “\(m.macName)”") }
        return nil
    }

    func askLibrary(_ r: RemoteLibraryRequest) { mirror?.send(.libraryAsk(r)) }

    /// The version the family renders with — the lender's in a mirror; `nil` while none is whole on disk.
    func renderedVariant(of f: Family) -> Variant? { occupancy?.families.first { $0.family == f }?.variant }

    func licenseAccepted(_ card: ModelCard) -> Bool {
        mirror != nil ? remoteLibrary?.part(card.family)?.accepted ?? false : library.isLicenseAccepted(card)
    }

    /// What installing `variant` would do against the disk that receives it.
    func installPlan(_ f: Family, variant: Variant) -> RemotePlan? {
        if mirror != nil { return remoteLibrary?.part(f)?.plan(variant) }
        return RemotePlan(Installer(library: library).plan(f, variant))
    }

    /// The job under way, said in this Mac's language (`LibraryJobProgress`): this Mac's, or the lender's as it publishes it.
    var libraryJobLine: String? {
        // This Mac's own upload is told here, in its journal; the lender's job as it publishes it.
        if let m = mirror, m.uploads.current == nil { return remoteLibraryJob.flatMap(LibraryJobProgress.line) }
        return libraryJobProgress.map(LibraryJobProgress.line) ?? LibraryJobProgress.line(libraryJobJournal)
    }

    /// How far the installation under way is, of 1; `nil` before its first file, and for a job that tells no bytes.
    var libraryJobFraction: Double? {
        if mirror != nil {
            guard case .downloading(_, _, _, let received, let total)? = remoteLibraryJob?.phase, total > 0 else { return nil }
            return min(1, Double(received) / Double(total))
        }
        return libraryJobProgress?.fraction
    }

    /// The tooltip of that line: the bytes received, of those to come.
    var libraryJobHelp: String {
        if mirror != nil {
            guard case .downloading(_, _, _, let received, let total)? = remoteLibraryJob?.phase else { return "" }
            return String(localized: "\(Figures.disk(received)) of \(Figures.disk(total))")
        }
        if let p = libraryJobProgress { return String(localized: "\(Figures.disk(p.received)) of \(Figures.disk(p.total))") }
        return Library.withoutHome(libraryJobJournal.joined(separator: "\n"))
    }

    func receiveLibrary(_ n: RemoteLibraryNews) {
        switch n {
        case .library(let l):
            remoteLibrary = l
            let o = l.occupancy()
            occupancy = o
            // The lender's imported models are models of the form too.
            cards = ModelCard.allCards + o.imported.map(\.card)
            installedLoRAs = l.loras.map { ($0.card, $0.bytes) }
        case .job(let j):
            remoteLibraryJob = j
            // An upload of this Mac's holds the line until it ends (`endUpload` hands it back).
            guard mirror?.uploads.current == nil else { return }
            // A queue that waits keeps its line (« Waiting for… ») until the lender's job is over.
            if mirror?.uploads.waiting == true {
                if j == nil { lenderJobEnded() }
                return
            }
            libraryJob = j.map { libraryJobTitle($0.task) }
            // What « Stop » says it does follows the lender's job (`libraryStopHelp`).
            if let j { libraryJobTask = j.task }
            libraryJobJournal = []
        case .confirming(let waiting):
            remoteConfirming = waiting
        }
    }

    func libraryJobTitle(_ t: RemoteLibraryJob.Task) -> String {
        switch t {
        case .installing(let f): String(localized: "Installing \(f.name)")
        case .importing: String(localized: "Importing on “\(mirror?.macName ?? "")”")
        case .other: String(localized: "Working on the library of “\(mirror?.macName ?? "")”")
        }
    }

    // MARK: Gestures

    /// « Accept and Install » on the borrower's sheet: the click, with the digest of the license it showed.
    func installRemotely(_ family: Family, variant: Variant, replacing: Bool) {
        guard libraryManageable else { return }
        askLibrary(.install(RemoteInstall(family: family, variant: variant, replacing: replacing,
                                          digest: LicenseDigest.of(ModelCard.of(family), variant: variant))))
        installOffer = nil
    }
}

extension LibraryJobProgress {
    /// The lender's job, from its structured phase: the same line as this Mac's own (`line(_:)` of a progress).
    static func line(_ j: RemoteLibraryJob) -> String? {
        guard case .downloading(let file, _, _, let received, let total) = j.phase else { return nil }
        return line(file: file, received: received, total: total)
    }
}

// MARK: - Lender

@MainActor
final class MirrorLibraryHost {
    private let app: AppState
    private let host: RemoteHost
    private let names: (Data) -> String?
    private var library: RemoteLibrary?
    private var job: RemoteLibraryJob?
    private var libraryScheduled = false, jobScheduled = false
    /// The client whose installation runs, and the lender's problem before it: its failure is told to it alone.
    private var requester: Data?
    private var problemBefore: Problem?
    private var working = false
    /// **The removal waiting for a click**: one alert at a time, in this Mac's window, for the client that asked.
    /// Nothing is deleted without the click; no answer for `waitLimit` seconds is a refusal.
    private struct Asking {
        let token: Int
        let identity: Data
        let alert: NSAlert
        var sheet: NSWindow?
        var timeout: Task<Void, Never>?
        let then: () -> Void
    }
    private var asking: Asking?
    private var asked = 0
    private static let waitLimit = 120

    init(app: AppState, host: RemoteHost, names: @escaping (Data) -> String?) {
        self.app = app; self.host = host; self.names = names
        watchLibrary()
        watchJob()
    }

    func joined(_ identity: Data) {
        if library == nil { library = snapshot() }
        if let library { host.send(.libraryNews(.library(library)), to: identity) }
        // Said when none runs too: the line a client kept across a cut (« Installing… ») ends with it.
        host.send(.libraryNews(.job(job)), to: identity)
    }

    // MARK: Publishing

    private func snapshot() -> RemoteLibrary? {
        app.occupancy.map { RemoteLibrary($0, loras: app.installedLoRAs, in: app.library) }
    }

    private func watchLibrary() {
        withObservationTracking {
            _ = app.occupancy; _ = app.cards; _ = app.installedLoRAs
        } onChange: { [weak self] in
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.scheduleLibrary() } }
        }
    }

    private func scheduleLibrary() {
        guard !libraryScheduled else { return }
        libraryScheduled = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            MainActor.assumeIsolated { self?.fireLibrary() }
        }
    }

    private func fireLibrary() {
        libraryScheduled = false
        if let l = snapshot(), l != library {
            library = l
            host.send(.libraryNews(.library(l)))
        }
        watchLibrary()
    }

    private func watchJob() {
        withObservationTracking {
            _ = app.libraryJob; _ = app.libraryJobProgress; _ = app.libraryJobTask
        } onChange: { [weak self] in
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.scheduleJob() } }
        }
    }

    private func scheduleJob() {
        guard !jobScheduled else { return }
        jobScheduled = true
        // A download tells its bytes ten times a second: a client hears them twice.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            MainActor.assumeIsolated { self?.fireJob() }
        }
    }

    private func fireJob() {
        jobScheduled = false
        let j = app.libraryJob == nil ? nil : RemoteLibraryJob(task: app.libraryJobTask, progress: app.libraryJobProgress)
        if j != job {
            job = j
            host.send(.libraryNews(.job(j)))
        }
        if j != nil {
            working = true
        } else if working {
            working = false
            told()
        }
        watchJob()
    }

    /// The installation that client asked for is over: if it failed, the client is told (the lender's own window shows it).
    private func told() {
        let who = requester
        requester = nil
        if let who, let p = app.problem, p != problemBefore, p.engine != nil { host.send(.refused(RemoteRefusal(p)), to: who) }
    }

    // MARK: Confirming a removal

    /// A client is gone: its question goes with it.
    func left(_ identity: Data) {
        if asking?.identity == identity { resolve(asking!.token, nil) }
    }

    /// Asks this Mac's user before `then` deletes anything for `identity`. A second question while one waits is refused.
    private func confirm(_ identity: Data, _ sentence: String, then: @escaping () -> Void) {
        guard asking == nil else { refuse(.app(.removalPending), identity); return }
        asked += 1
        let token = asked
        let alert = NSAlert()
        alert.messageText = sentence
        alert.informativeText = String(localized: "Nothing is removed without your click.")
        alert.addButton(withTitle: String(localized: "Remove")).hasDestructiveAction = true
        alert.addButton(withTitle: String(localized: "Decline"))
        var a = Asking(token: token, identity: identity, alert: alert, then: then)
        a.timeout = Task { [weak self] in
            try? await Task.sleep(for: .seconds(Self.waitLimit))
            guard !Task.isCancelled else { return }
            self?.resolve(token, .removalUnanswered)
        }
        let answered: (NSApplication.ModalResponse) -> Void = { [weak self] r in
            MainActor.assumeIsolated { self?.resolve(token, r == .alertFirstButtonReturn ? .allowed : .removalDeclined) }
        }
        NSApp.activate()
        if let w = NSApp.keyWindow ?? NSApp.mainWindow ?? NSApp.windows.first(where: { $0.isVisible && $0.canBecomeMain }) {
            a.sheet = w
            asking = a
            alert.beginSheetModal(for: w, completionHandler: answered)
        } else {
            asking = a
            Task { @MainActor in answered(alert.runModal()) }
        }
        host.send(.libraryNews(.confirming(true)), to: identity)
    }

    private enum Outcome { case allowed, removalDeclined, removalUnanswered }

    /// The alert is over: by a click, the time, or the client leaving (`nil`: nobody to tell). Once only.
    private func resolve(_ token: Int, _ outcome: Outcome?) {
        guard let a = asking, a.token == token else { return }
        asking = nil
        a.timeout?.cancel()
        if let w = a.sheet { if w.attachedSheet != nil { w.endSheet(a.alert.window) } } else { NSApp.abortModal() }
        guard let outcome else { return }
        host.send(.libraryNews(.confirming(false)), to: a.identity)
        switch outcome {
        case .allowed: a.then()
        case .removalDeclined: refuse(.app(.removalDeclined), a.identity)
        case .removalUnanswered: refuse(.app(.removalUnanswered), a.identity)
        }
    }

    private func who(_ identity: Data) -> String { names(identity) ?? "" }

    // MARK: Requests

    func handle(_ r: RemoteLibraryRequest, from identity: Data) {
        switch r {
        case .cancel:
            // Only the job that client asked for: this Mac's own, or another Mac's, is not its to stop.
            if requester == identity { app.cancelLibraryJob() }
        case .install(let i):
            install(i, from: identity)
        case .use(let family, let variant):
            // The disk itself, not `occupancy`: a read may still be on its way after the last change.
            guard app.library.part(family).installed.contains(variant) else {
                refuse(.app(.familyUnknown("\(family.rawValue)/\(variant.rawValue)")), identity); return
            }
            guard manageable(identity) else { return }
            changing(identity) { app.use(family, variant: variant) }
        case .uninstall(let family, let variant):
            guard variant.map(family.variants.contains) ?? true else {
                refuse(.app(.familyUnknown("\(family.rawValue)/\(variant!.rawValue)")), identity); return
            }
            guard manageable(identity) else { return }
            let sentence = variant.map { String(localized: "“\(who(identity))” wants to remove the \(VersionChoice.name($0)) version of \(family.name).") }
                ?? String(localized: "“\(who(identity))” wants to uninstall \(family.name).")
            confirm(identity, sentence) { [self] in
                if manageable(identity) { changing(identity) { app.uninstall(family, variant: variant) } }
            }
        case .removeModel(let id):
            guard let card = RemoteLibraryRequest.importedModel(id, among: app.cards) else {
                refuse(.engine(.notAnImportedModel(model: id)), identity); return
            }
            guard manageable(identity) else { return }
            confirm(identity, String(localized: "“\(who(identity))” wants to remove the model \(card.name).")) { [self] in
                // Named again after the click: the library may have changed while the alert waited.
                guard manageable(identity), let card = RemoteLibraryRequest.importedModel(id, among: app.cards) else { return }
                changing(identity) { app.remove(card) }
            }
        case .removeLoRA(let path):
            guard let lora = RemoteLibraryRequest.lora(path, among: app.installedLoRAs.map(\.card)) else {
                refuse(.app(.loraNotInstalled(name: ((path as NSString).lastPathComponent as NSString).deletingPathExtension)), identity)
                return
            }
            guard manageable(identity) else { return }
            confirm(identity, String(localized: "“\(who(identity))” wants to remove the LoRA \(lora.name).")) { [self] in
                guard manageable(identity), let lora = RemoteLibraryRequest.lora(path, among: app.installedLoRAs.map(\.card)) else { return }
                changing(identity) { app.remove(lora) }
            }
        case .discard:
            guard manageable(identity) else { return }
            // Nothing to empty deletes nothing: no question. The store is walked off the main thread, the disk as it is now.
            let library = app.library
            Task {
                let left = await Task.detached(priority: .userInitiated) { library.downloadsLeft() }.value
                guard manageable(identity) else { return }
                guard r.deletes(downloadsLeft: left) else { changing(identity) { app.emptyDownloads() }; return }
                confirm(identity, String(localized: "“\(who(identity))” wants to empty the interrupted downloads of this Mac.")) { [self] in
                    if manageable(identity) { changing(identity) { app.emptyDownloads() } }
                }
            }
        }
    }

    private func refuse(_ r: RemoteRefusal, _ identity: Data) { host.send(.refused(r), to: identity) }

    /// Nothing else is running and the disk is there; the client is told otherwise.
    private func manageable(_ identity: Data) -> Bool {
        guard app.libraryReachable else { refuse(.app(.libraryUnreachable), identity); return false }
        guard app.managementPossible else { refuse(.app(.libraryBusy), identity); return false }
        return true
    }

    /// A removal made on the lender as its own window makes it: its failure is the client's too.
    private func changing(_ identity: Data, _ action: () -> Void) {
        let kept = app.problem
        app.problem = nil
        action()
        if let p = app.problem, p.engine != nil { refuse(RemoteRefusal(p), identity) } else { app.problem = kept }
    }

    private func install(_ i: RemoteInstall, from identity: Data) {
        guard i.family.variants.contains(i.variant) else {
            refuse(.app(.familyUnknown("\(i.family.rawValue)/\(i.variant.rawValue)")), identity)
            return
        }
        guard manageable(identity) else { return }
        let card = ModelCard.of(i.family)
        // Only the text this Mac would have shown: a click never accepts other words.
        guard i.digest == LicenseDigest.of(card, variant: i.variant) else {
            refuse(.app(.licenseMismatch(model: card.id)), identity)
            return
        }
        let plan = Installer(library: app.library).plan(i.family, i.variant)
        let replaces = i.replacing && plan.fitsReplacing
        guard plan.fits || replaces else {
            refuse(.engine(.diskFull(needed: plan.needed, available: plan.free)), identity)
            return
        }
        // An installation that takes the other version's place removes files: this Mac's user clicks first,
        // as for a removal — the borrower's click is not enough to take a version off this disk.
        guard replaces, let old = plan.others.first else { begin(i, replacing: false, from: identity); return }
        confirm(identity, String(localized: "“\(who(identity))” wants to install \(i.family.name) (\(VersionChoice.name(i.variant))), which removes the \(VersionChoice.name(old)) version first: until it is installed, \(i.family.name) cannot render.")) { [self] in
            if manageable(identity) { begin(i, replacing: true, from: identity) }
        }
    }

    /// The library's job that starts now is `identity`'s: an installation it asked for, the import of a file it sent.
    func asked(by identity: Data) {
        requester = identity
        problemBefore = app.problem
    }

    /// `replacing`: what this Mac's user confirmed, never what the client asked — only a click here removes a version.
    private func begin(_ i: RemoteInstall, replacing: Bool, from identity: Data) {
        asked(by: identity)
        app.acceptAndInstall(i.family, variant: i.variant, replacing: replacing, asked: true)
    }
}
