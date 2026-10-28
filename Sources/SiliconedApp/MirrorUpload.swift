// **Importing a file of the Neo on the mini** — both ends of the upload (`Siliconed/Remote/Upload.swift`
// is the protocol and the streaming; this is what the app does with it).
//
// - **Borrowing side** (`UploadBorrower`, `AppState.upload`): « Import… » in a mirror asks for the file
//   on this Mac, offers it, and — if the other Mac takes it — streams it from the disk as the network
//   goes. One file at a time; the others of a selection follow once the one before is verified.
//   Nothing resumes: a cut link ends it, and says so.
//   **The queue survives a `busy`**: the other Mac still imports the file before (it takes the forge once
//   verified), so the next is offered again when its forge is published as over — or, a render being the
//   cause, every few seconds. « Stop » empties the queue, whatever it waits on.
// - **Lending side** (`UploadHub`): judges the offer before the first byte (an import, a forge or a
//   render under way refuses it: one big job at a time), receives the stream into a temporary folder of
//   the library, and — verified — hands the file to `AppState.perform`, the very path of a local import
//   (family recognized, forge, the family's installation offered if it lacks). The temporary file goes
//   with the import.
// - **Progress is the forge's**: both sides show it as an installation in progress (`forgeInProgress`,
//   its journal, « Stop »), the lender's renders waiting behind it as they wait behind a forge.

import AppKit
import Foundation
import Siliconed
import UniformTypeIdentifiers

// MARK: - Borrowing

/// What the borrowing mirror needs of its link (`App.compute`): is it joined, and the stream out.
struct UploadPort {
    var linked: () -> Bool
    var send: (UInt32, any StreamSource) -> Void
    var cancel: (UInt32) -> Void
}

/// What a mirror remembers of the uploads it asked for.
@MainActor
final class UploadBorrower {
    struct Current {
        let id: UInt32
        let url: URL
        let name: String
        let size: Int64
        let source: UploadSource
        let cancellation: Cancellation
        var answered = false
    }
    let port: UploadPort?
    var queue: [URL] = []
    var current: Current?
    /// The queue holds files and none is out: the lender is busy, they wait for it.
    var waiting = false
    var retry: Task<Void, Never>?
    private(set) var next: UInt32 = 1

    init(port: UploadPort?) { self.port = port }

    func newID() -> UInt32 { defer { next += 1 }; return next }
}

extension AppState {
    /// « Import… » is possible: here, as the library allows; in a mirror, when the link is joined and no
    /// upload runs.
    var importPossible: Bool {
        guard let m = mirror else { return managementPossible }
        return m.uploads.port?.linked() == true && forgeInProgress == nil
    }

    /// Offers the files to the other Mac, one after the other (`receive(upload:)` goes on).
    func upload(_ urls: [URL]) {
        guard let link = mirror, importPossible else { return }
        link.uploads.queue = urls.filter(Self.importable)
        uploadNext()
    }

    private func uploadNext() {
        guard let link = mirror else { return }
        let u = link.uploads
        u.waiting = false
        u.retry?.cancel()
        while let url = u.queue.first {
            u.queue.removeFirst()
            guard let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize, size > 0 else {
                problem = Problem(Self.sentence(.notImportable, mac: link.macName))
                continue
            }
            let id = u.newID()
            let cancellation = Cancellation()
            let source = UploadSource(
                url: url, size: Int64(size), cancelled: { cancellation.isCancelled },
                progress: { [weak self] sent in
                    DispatchQueue.main.async { MainActor.assumeIsolated { self?.uploadMoved(id, sent) } }
                },
                finished: { [weak self] result in
                    DispatchQueue.main.async { MainActor.assumeIsolated { self?.uploadSent(id, result) } }
                })
            u.current = .init(id: id, url: url, name: url.lastPathComponent, size: Int64(size), source: source, cancellation: cancellation)
            forgeInProgress = String(localized: "Importing on “\(link.macName)”")
            forgeJournal = ["⇡ \(url.lastPathComponent)"]
            cancellationForge = cancellation
            problem = nil
            link.send(.uploadOffer(UploadOffer(id: id, name: url.lastPathComponent, size: Int64(size))))
            return
        }
    }

    /// The other Mac's word on an offer, then on the file.
    func receive(upload m: SessionMessage) {
        guard let link = mirror, let cur = link.uploads.current else { return }
        switch m {
        case .uploadAnswer(let id, let answer) where id == cur.id:
            switch answer {
            case .accepted:
                link.uploads.current?.answered = true
                // « Stop » pressed while the offer was out: the first pull would say so, but nothing pulls yet.
                if cur.cancellation.isCancelled { stopUpload(.cancelled) } else { link.uploads.port?.send(id, cur.source) }
            case .refused(.busy):
                waitForLender(cur)
            case .refused(let r):
                endUpload(problem: Self.sentence(r, mac: link.macName))
            }
        case .uploadResult(let id, let result) where id == cur.id:
            switch result {
            case .verified:
                endUpload(flash: String(localized: "“\(cur.name)” is on “\(link.macName)”: it is being imported there."), keepingQueue: true)
                uploadNext()
            case .failed(let f):
                endUpload(problem: Self.sentence(f, mac: link.macName))
            }
        default:
            break
        }
    }

    private func uploadMoved(_ id: UInt32, _ sent: Int64) {
        guard let cur = mirror?.uploads.current, cur.id == id else { return }
        forgeJournal.append("… \(Int(sent * 100 / max(cur.size, 1)))%")
    }

    /// The source is done: its digest went out (the other Mac now checks), or it could not go on.
    private func uploadSent(_ id: UInt32, _ result: Result<Void, UploadFailure>) {
        guard let cur = mirror?.uploads.current, cur.id == id else { return }
        switch result {
        case .success: forgeJournal.append("✓ sent")
        case .failure(let f): stopUpload(f)
        }
    }

    /// Ends the upload here and tells the other Mac, which deletes the partial file.
    private func stopUpload(_ f: UploadFailure) {
        guard let link = mirror, let cur = link.uploads.current else { return }
        link.send(.uploadCancel(cur.id))
        link.uploads.port?.cancel(cur.id)
        cur.source.close()
        endUpload(problem: f == .cancelled ? nil : Self.sentence(f, mac: link.macName),
                  flash: f == .cancelled ? Self.sentence(f, mac: link.macName) : nil)
    }

    /// The lender refused this offer as busy: the file goes back to the head of the queue, and waits.
    private func waitForLender(_ cur: UploadBorrower.Current) {
        guard let link = mirror else { return }
        let u = link.uploads
        u.queue.insert(cur.url, at: 0)
        endUpload(keepingQueue: true)
        u.waiting = true
        forgeInProgress = String(localized: "Waiting for “\(link.macName)” to finish before sending \(cur.name)")
        u.retry?.cancel()
        u.retry = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(5))
                guard !Task.isCancelled, let self, let m = self.mirror, m.uploads.waiting, m.uploads.current == nil else { return }
                self.uploadNext()
            }
        }
    }

    /// « Stop » on a queue that waits: the files are forgotten; the lender's own import, if any, goes on
    /// (a second « Stop » reaches it).
    func stopWaitingUploads() {
        guard let link = mirror, link.uploads.waiting else { return }
        endUpload()
    }

    /// The lender's forge is over: the file that waited for it may go.
    func lenderForgeEnded() {
        guard let link = mirror, link.uploads.waiting, link.uploads.current == nil else { return }
        uploadNext()
    }

    private func endUpload(problem sentence: String? = nil, flash message: String? = nil, keepingQueue: Bool = false) {
        guard let link = mirror else { return }
        link.uploads.current = nil
        link.uploads.retry?.cancel()
        if !keepingQueue { link.uploads.queue = []; link.uploads.waiting = false }
        // The lender may be forging already — the file just sent, imported there: its line comes back.
        forgeInProgress = remoteForge.map { forgeTitle($0.task) }
        forgeJournal = []
        cancellationForge = nil
        if let sentence { problem = Problem(sentence) }
        if let message { flash(message) }
    }

    /// The link left the joined state: nothing resumes.
    func uploadLinkChanged(_ s: RemoteLink.State) {
        guard let link = mirror else { return }
        if case .connected = s { return }
        guard let cur = link.uploads.current else {
            if link.uploads.waiting { endUpload() }
            return
        }
        link.uploads.port?.cancel(cur.id)
        cur.source.close()
        endUpload(problem: Self.sentence(.interrupted, mac: link.macName))
    }

    static func sentence(_ r: UploadRefusal, mac: String) -> String {
        switch r {
        case .notImportable: String(localized: "Only a .safetensors or a .gguf file can be imported.")
        case .busy: String(localized: "“\(mac)” is busy: an installation, an import or a render is running.")
        case .noSpace(let needed, let available):
            String(localized: "Not enough disk space: \(Figures.disk(Int(needed))) needed, \(Figures.disk(Int(available))) available.")
        case .cannotWrite: sentence(UploadFailure.cannotWrite, mac: mac)
        }
    }

    static func sentence(_ f: UploadFailure, mac: String) -> String {
        switch f {
        case .size, .fingerprint: String(localized: "The file arrived damaged on “\(mac)”: nothing was kept. Send it again.")
        case .cannotWrite: String(localized: "“\(mac)” could not write the file.")
        case .cannotRead: String(localized: "The file could not be read.")
        case .cancelled: String(localized: "Import stopped: the part already sent was deleted.")
        case .interrupted: String(localized: "The link to “\(mac)” was lost: the part already sent was deleted. Send it again.")
        }
    }
}

/// The journal's two upload markers, as the sentence a footer shows (`ForgeProgress.line`):
/// `⇡ file` sending, `⇣ file` receiving, `… N%` the percent, `✓ …` the bytes all out (the other end checks).
enum UploadProgress {
    static func line(_ journal: [String]) -> String? {
        guard let start = journal.lastIndex(where: { $0.hasPrefix("⇡ ") || $0.hasPrefix("⇣ ") }) else { return nil }
        let sending = journal[start].hasPrefix("⇡ ")
        let file = String(journal[start].dropFirst(2))
        let rest = journal[(start + 1)...]
        if rest.contains(where: { $0.hasPrefix("✓ ") }) { return String(localized: "Checking \(file)") }
        guard let percent = rest.last(where: { $0.hasPrefix("… ") }).flatMap({ Int($0.dropFirst(2).prefix { $0.isNumber }) }) else {
            return sending ? String(localized: "Sending \(file)") : String(localized: "Receiving \(file)")
        }
        let p = Figures.percent(percent)
        return sending ? String(localized: "Sending \(file): \(p)") : String(localized: "Receiving \(file): \(p)")
    }
}

// MARK: - Lending

/// The lending Mac's side: one upload at a time from one Mac, received into the library's temporary folder.
@MainActor
final class UploadHub {
    private let app: AppState
    private let send: (SessionMessage, Data) -> Void
    let inbox: UploadInbox
    private var cancellation: Cancellation?
    private static var purged = false

    init(app: AppState, send: @escaping (SessionMessage, Data) -> Void) {
        self.app = app; self.send = send
        let directory = UploadStore.directory(in: app.library)
        // What a previous run left: nothing waits for it now (a file waiting for its family is in memory only).
        if !Self.purged { Self.purged = true; UploadStore.purge(directory) }
        inbox = UploadInbox(directory: directory, extensions: AppState.importExtensions)
    }

    func offer(_ o: UploadOffer, from identity: Data) {
        func answer(_ a: UploadAnswer) { send(.uploadAnswer(id: o.id, a), identity) }
        guard app.managementPossible else { answer(.refused(.busy)); return }
        let c = Cancellation()
        if let refusal = inbox.begin(identity: identity, o, cancelled: { c.isCancelled }) { answer(.refused(refusal)); return }
        cancellation = c
        let name = UploadStore.cleanName(o.name) ?? o.name
        app.forgeInProgress = String(localized: "Importing from another Mac")
        app.forgeJournal = ["⇣ \(name)"]
        app.cancellationForge = c
        app.problem = nil
        answer(.accepted)
    }

    /// A stream step, on the main actor: only the whole percents and the end come this way.
    func stepped(_ identity: Data, _ offer: UploadOffer, _ step: UploadReceiver.Step) {
        switch step {
        case .more(let received, _):
            app.forgeJournal.append("… \(Int(received * 100 / max(offer.size, 1)))%")
        case .done(let file):
            release()
            send(.uploadResult(id: offer.id, .verified), identity)
            if app.managementPossible { app.perform([file]) }
            else {
                try? FileManager.default.removeItem(at: file.deletingLastPathComponent())
                app.next()
            }
        case .failed(let f):
            release()
            send(.uploadResult(id: offer.id, .failed(f)), identity)
            app.flash(String(localized: "Import from another Mac stopped: nothing was kept."))
            app.next()
        }
    }

    /// The Mac cancelled (`id`), or left (`nil`): the partial file goes.
    func abort(_ identity: Data, id: UInt32? = nil) {
        guard inbox.abort(identity: identity, id: id) != nil else { return }
        release()
        app.flash(String(localized: "Import from another Mac stopped: nothing was kept."))
        app.next()
    }

    /// The forge's place is given back (the import, if any, takes it again).
    private func release() {
        guard cancellation != nil else { return }
        cancellation = nil
        app.forgeInProgress = nil
        app.forgeJournal = []
        app.cancellationForge = nil
    }
}
