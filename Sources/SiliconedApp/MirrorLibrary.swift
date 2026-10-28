// **The lender's library, in the Models and Hub sheets of a borrowing Mac** (`Remote/Installing.swift`).
//
// - **Borrower** (`AppState(mirroring:)`): the sheets are the ordinary ones, reading what the lender
//   published — `occupancy` is rebuilt from `RemoteLibrary`, the install plans and the accepted licenses
//   come from it, an installation under way is `RemoteForge` said here. Each gesture is a message instead
//   of an action (`if mirror != nil` at the head of the action); what has no meaning for another Mac's
//   disk (opening or moving a library, the Finder) is greyed, with its reason.
// - **Lender** (`MirrorLibraryHost`): publishes the library and the forge to every client — the library is
//   common — and judges each request again as the Models sheet would: the family and version known, the
//   disk, nothing else running, the digest of the license text shown. The download and the forge are its
//   own (`launchForge`); a failure is told to the client that asked.
// - **The license** is accepted on the borrower's sheet, which shows the lender's words (one build on both
//   Macs, `LicenseDigest`); the click goes with the digest and the lender accepts only its own text.

import Foundation
import Siliconed

// MARK: - Borrower

extension AppState {
    /// This library's name where the sheets say where it is.
    var libraryTitle: String { mirror?.macName ?? library.displayPath }

    /// Why a gesture on this library's disk is greyed, in a mirror (`nil`: it is this Mac's).
    var localOnlyHelp: String? {
        mirror.map { String(localized: "This is the library of “\($0.macName)”: only that Mac can open it, move it or show it in the Finder.") }
    }

    /// **May the library change now**: this Mac's (`managementPossible`), or the lender's — nothing of
    /// ours running, no installation under way there; the lender judges again.
    var libraryManageable: Bool {
        mirror == nil ? managementPossible : remoteLibrary != nil && remoteForge == nil && !busy
    }

    func askLibrary(_ r: RemoteLibraryRequest) { mirror?.send(.libraryAsk(r)) }

    /// The version installed, the lender's in a mirror.
    func installedVariant(of f: Family) -> Variant? {
        mirror != nil ? remoteLibrary?.part(f)?.installed : library.installedVariant(of: f)
    }

    /// Compact, Light on a lender of 8 GB: the lender's memory decides.
    func preselectedVariant(of f: Family) -> Variant {
        mirror != nil ? remoteLibrary?.part(f)?.preselected ?? f.preselectedVariant() : f.preselectedVariant()
    }

    func licenseAccepted(_ card: ModelCard) -> Bool {
        mirror != nil ? remoteLibrary?.part(card.family)?.accepted ?? false : library.isLicenseAccepted(card)
    }

    /// What installing `variant` would do against the disk that receives it.
    func installPlan(_ f: Family, variant: Variant, baseDiT: Bool) -> RemotePlan? {
        if mirror != nil { return remoteLibrary?.part(f)?.plan(variant, baseDiT: baseDiT) }
        return (try? library.installPlan(f, variant: variant, baseDiT: baseDiT)).map { RemotePlan($0, baseDiT: baseDiT) }
    }

    /// The installation under way, said in this Mac's language (`ForgeProgress`).
    var forgeLine: String? {
        // This Mac's own upload is told here, in its journal; the lender's forge as it publishes it.
        guard let m = mirror, m.uploads.current == nil else { return ForgeProgress.line(forgeJournal) }
        return remoteForge.flatMap(ForgeProgress.line)
    }

    /// The tooltip of that line: the engine's rows here, the bytes received there.
    var forgeHelp: String {
        guard mirror != nil else { return Library.withoutHome(forgeJournal.joined(separator: "\n")) }
        if case .downloading(_, _, let bytes?, let total?, _)? = remoteForge?.phase {
            return String(localized: "\(Figures.disk(bytes)) of \(Figures.disk(total))")
        }
        return ""
    }

    func receiveLibrary(_ n: RemoteLibraryNews) {
        switch n {
        case .library(let l):
            remoteLibrary = l
            let o = l.occupancy()
            occupancy = o
            // The lender's imported models are models of the form too.
            cards = ModelCard.allCards + o.imported.map(\.card)
        case .forge(let f):
            remoteForge = f
            // An upload of this Mac's holds the line until it ends (`endUpload` hands it back).
            guard mirror?.uploads.current == nil else { return }
            forgeInProgress = f.map { forgeTitle($0.task) }
            forgeJournal = []
        case .hubVerdict(let file, let v):
            guard hubOffer?.link.repository == file.repository,
                  let k = hubOffer?.files.firstIndex(where: { $0.file.path == file.path }) else { return }
            let verdict = HubImport.Verdict(v)
            hubOffer?.files[k].verdict = verdict
            // The link's file if it is importable, else the first that is (as `open` does on the disk).
            if let link = hubOffer?.link, hubOffer?.chosen == nil, HubOffer.importable(verdict),
               link.file == nil || link.file == file.path || !(hubOffer?.files.contains { $0.file.path == link.file } ?? false) {
                hubOffer?.chosen = file.path
            }
        }
    }

    func forgeTitle(_ t: RemoteForge.Task) -> String {
        switch t {
        case .installing(let f): String(localized: "Installing \(f.name)")
        case .downloading(let file): String(localized: "Downloading \(file)")
        case .importing(let files): String(localized: "Importing \(files.joined(separator: ", "))")
        case .other: String(localized: "Working on the library of “\(mirror?.macName ?? "")”")
        }
    }

    // MARK: Gestures

    /// « Accept and Install » on the borrower's sheet: the click, with the digest of the license it showed.
    func installRemotely(_ family: Family, variant: Variant, baseDiT: Bool) {
        guard libraryManageable else { return }
        askLibrary(.install(RemoteInstall(family: family, variant: variant, baseDiT: baseDiT,
                                          digest: LicenseDigest.of(ModelCard.of(family), variant: variant))))
        installOffer = nil
    }

    /// **A « Use this model » link, with a lender**: the repository is read from this Mac's network, but
    /// each file is judged by the lender, whose maps alone say what a LoRA fits.
    func openRemotely(_ link: HubLink) {
        guard let mirror else { return }
        let offer = HubOffer(link: link)
        hubOffer = offer
        Task.detached(priority: .userInitiated) { [weak self] in
            let r: HubRepository
            do { r = try HubRepository.fetch(link.repository) } catch {
                await MainActor.run { if self?.hubOffer?.id == offer.id { self?.hubOffer?.problem = Problem(error) } }
                return
            }
            let candidates = Array(r.candidates(preferring: link.file).prefix(Self.hubExamined))
            await MainActor.run {
                guard let self, self.hubOffer?.id == offer.id else { return }
                if let f = r.installs {
                    self.hubOffer = nil
                    self.offerInstall(f)
                    return
                }
                self.hubOffer?.repository = r
                self.hubOffer?.files = candidates.map { ($0, nil) }
                for f in candidates { mirror.send(.libraryAsk(.hubJudge(RemoteHubFile(repository: r.id, path: f.path)))) }
            }
        }
    }

    /// « Import » in a mirror: the lender judges the file again and downloads it. A LoRA for a family
    /// not installed there is not imported behind the user's back: the family's installation is offered.
    func importFromHubRemotely() {
        guard libraryManageable, let offer = hubOffer, let r = offer.repository, let (f, judged) = offer.choice,
              let verdict = judged, HubOffer.importable(verdict) else { return }
        hubOffer = nil
        if case .loraNeedsFamily(let family) = verdict, remoteLibrary?.part(family)?.isReady != true {
            offerInstall(family)
            return
        }
        askLibrary(.hubInstall(RemoteHubFile(repository: r.id, path: f.path)))
    }
}

extension HubImport.Verdict {
    init(_ v: RemoteHubVerdict) {
        switch v {
        case .lora(let f): self = .lora(f)
        case .model(let f): self = .model(f)
        case .loraNeedsFamily(let f): self = .loraNeedsFamily(f)
        case .unexamined: self = .unexamined
        case .refused(let r): self = .refused(Problem(r).text)
        }
    }
}

extension ForgeProgress {
    /// The same lines as the engine's journal gives (`line(_:)`), from the structured phase.
    static func line(_ f: RemoteForge) -> String? {
        switch f.phase {
        case .starting:
            nil
        case .installed:
            String(localized: "Installed")
        case .downloading(let file, let percent, _, _, let preparing):
            switch (percent.map(Figures.percent), preparing) {
            case let (p?, name?): String(localized: "Downloading \(file): \(p) — preparing \(name)")
            case let (nil, name?): String(localized: "Downloading \(file) — preparing \(name)")
            case let (p?, nil): String(localized: "Downloading \(file): \(p)")
            case (nil, nil): String(localized: "Downloading \(file)")
            }
        case .preparing(let name, let converted):
            converted.map { String(localized: "Converting: \(Figures.percent($0))") } ?? String(localized: "Preparing \(name)")
        }
    }
}

// MARK: - Lender

@MainActor
final class MirrorLibraryHost {
    private let app: AppState
    private let host: RemoteHost
    private var library: RemoteLibrary?
    private var forge: RemoteForge?
    private var libraryScheduled = false, forgeScheduled = false
    /// The client whose installation runs, and the lender's problem before it: its failure is told to it alone.
    private var requester: Data?
    private var problemBefore: Problem?
    private var forging = false
    /// The Hub repositories read lately, so that the files of one are not asked for 24 times.
    private var repositories: [String: HubRepository] = [:]
    /// The Hub files are judged one after the other: each reads a few hundred KB from the Hub.
    private var judging: Task<Void, Never>?

    init(app: AppState, host: RemoteHost) {
        self.app = app; self.host = host
        watchLibrary()
        watchForge()
    }

    func joined(_ identity: Data) {
        if library == nil { library = snapshot() }
        if let library { host.send(.libraryNews(.library(library)), to: identity) }
        if let forge { host.send(.libraryNews(.forge(forge)), to: identity) }
    }

    // MARK: Publishing

    private func snapshot() -> RemoteLibrary? {
        app.occupancy.map { RemoteLibrary($0, in: app.library) }
    }

    private func watchLibrary() {
        withObservationTracking {
            _ = app.occupancy; _ = app.cards
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

    private func watchForge() {
        withObservationTracking {
            _ = app.forgeInProgress; _ = app.forgeJournal; _ = app.forgeTask
        } onChange: { [weak self] in
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.scheduleForge() } }
        }
    }

    private func scheduleForge() {
        guard !forgeScheduled else { return }
        forgeScheduled = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            MainActor.assumeIsolated { self?.fireForge() }
        }
    }

    private func fireForge() {
        forgeScheduled = false
        let f = app.forgeInProgress == nil ? nil : RemoteForge(task: app.forgeTask, journal: app.forgeJournal)
        if f != forge {
            forge = f
            host.send(.libraryNews(.forge(f)))
        }
        if f != nil {
            forging = true
        } else if forging {
            forging = false
            told()
        }
        watchForge()
    }

    /// The installation that client asked for is over: if it failed, the client is told (the lender's own window shows it).
    private func told() {
        defer { requester = nil }
        guard let who = requester, let p = app.problem, p != problemBefore, p.engine != nil else { return }
        host.send(.refused(RemoteRefusal(p)), to: who)
    }

    // MARK: Requests

    func handle(_ r: RemoteLibraryRequest, from identity: Data) {
        switch r {
        case .cancel:
            app.cancelForge()
        case .install(let i):
            install(i, from: identity)
        case .uninstall(let family):
            guard Family.offered.contains(family), manageable(identity) else { return }
            changing(identity) { app.uninstall(family) }
        case .removeModel(let id):
            guard let card = app.cards.first(where: { $0.id == id && $0.isImported }) else {
                refuse(.app(.modelUnknown(id)), identity); return
            }
            guard manageable(identity) else { return }
            changing(identity) { app.remove(card) }
        case .removeLoRA(let path):
            guard let card = app.occupancy?.loras.first(where: { $0.card.path == path })?.card else {
                refuse(.app(.loraNotInstalled(name: ((path as NSString).lastPathComponent as NSString).deletingPathExtension)), identity)
                return
            }
            guard manageable(identity) else { return }
            changing(identity) { app.remove(card) }
        case .discard:
            guard manageable(identity) else { return }
            changing(identity) { app.emptyDownloads() }
        case .hubJudge(let file):
            judge(file, for: identity)
        case .hubInstall(let file):
            hubInstall(file, for: identity)
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
        guard Family.offered.contains(i.family), i.family.variants.contains(i.variant) else {
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
        do {
            let plan = try app.library.installPlan(i.family, variant: i.variant, baseDiT: i.baseDiT)
            guard plan.fits else { throw EngineError.diskFull(needed: plan.peak, available: app.library.freeSpace() ?? 0) }
        } catch {
            refuse(RemoteRefusal(error), identity)
            return
        }
        started(by: identity)
        app.acceptAndInstall(i.family, variant: i.variant, baseDiT: i.baseDiT, asked: true)
    }

    private func started(by identity: Data) {
        requester = identity
        problemBefore = app.problem
    }

    // MARK: The Hub

    private enum Judged {
        case verdict(HubRepository, HubRepository.File, HubImport.Verdict)
        case unknown
        case failed(EngineError)
    }

    private func judged(_ file: RemoteHubFile) async -> Judged {
        guard HubLink.isRepository(file.repository), HubLink.isFile(file.path) else { return .unknown }
        let cached = repositories[file.repository]
        let b = app.library
        let result: (Judged, HubRepository?) = await Task.detached(priority: .userInitiated) {
            do {
                let r: HubRepository
                if let cached { r = cached } else { r = try HubRepository.fetch(file.repository) }
                guard let f = r.files.first(where: { $0.path == file.path }) else { return (Judged.unknown, r) }
                return (.verdict(r, f, try HubImport.examine(r, f, in: b)), r)
            } catch let e as EngineError {
                return (.failed(e), nil)
            } catch {
                return (.failed(EngineError(error)), nil)
            }
        }.value
        if let r = result.1 {
            if repositories.count >= 4 { repositories = [:] }
            repositories[file.repository] = r
        }
        return result.0
    }

    private func judge(_ file: RemoteHubFile, for identity: Data) {
        let previous = judging
        judging = Task { @MainActor [weak self] in
            await previous?.value
            guard let self else { return }
            let verdict: RemoteHubVerdict = switch await judged(file) {
            case .verdict(_, _, let v): RemoteHubVerdict(v)
            case .unknown: .refused(.app(.hubFileUnknown(file.path)))
            case .failed(let e): .refused(.engine(e))
            }
            host.send(.libraryNews(.hubVerdict(file, verdict)), to: identity)
        }
    }

    /// Judged again here — the Hub may have moved, the borrower's verdict is never the last word —, then
    /// downloaded and imported as a file dropped on this Mac's window.
    private func hubInstall(_ file: RemoteHubFile, for identity: Data) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            let found = await judged(file)
            switch found {
            case .unknown: refuse(.app(.hubFileUnknown(file.path)), identity)
            case .failed(let e): refuse(.engine(e), identity)
            case .verdict(let r, let f, let v):
                switch v {
                case .refused(let why): refuse(.text(why), identity)
                case .loraNeedsFamily(let family) where !Installer(library: app.library).isReady(family):
                    refuse(.app(.hubNeedsFamily(family)), identity)
                default:
                    guard manageable(identity) else { return }
                    started(by: identity)
                    app.downloadFromHub(r, f, verdict: v)
                }
            }
        }
    }
}
