// **This Mac's state, published to the Macs it lends itself to** — the lending side of the mirror
// (`AppState+Mirror.swift` is the other). It reads the app's state as the window does and sends what
// changed; what comes back is a request it judges as `silicontrol add` would, then queues.
//
// - **Observed, not hooked**: `withObservationTracking` over what a snapshot reads — the next change
//   sends a new one, 50 ms later at most so that a burst of changes (a step: fraction, step, preview)
//   goes as one. A snapshot equal to the last is not sent.
// - **The catalog is read apart** (it reads LoRAs from the disk): each time the library is read again
//   (`AppState.libraryRead`) — a LoRA that comes or goes changes neither the cards nor the ready models.
// - **Every request is judged again**: the model ready, its schedule, the format's rules, B, the LoRAs
//   installed for it. The borrowing Mac's form judged first, with this Mac's B and traits, but its word is
//   never the last.
// - **Each Mac sees only its own** (`Entry.owner`, `Job.owner`): its jobs, its images, its preview, its
//   series — published per client (`Seen`). This Mac's own renders and other Macs' are a count and
//   a duration (`RemoteQueue.others`), and a client's actions reach only what it asked for. This Mac's
//   window sees everything: it is its machine.
// - **An edit's images come as blobs before their render**, but a message never waits for a blob
//   (`Frames`): the render may come first, and is held until they are all there — in its order, the
//   same Mac's renders after it held too, another Mac's not. The bytes live in memory only, until the job
//   is made (incognito), and within a bound per Mac (`uploadBytes`, `uploadCount`).

import AppKit
import Foundation
import ImageIO
import Siliconed

@MainActor
final class MirrorServer {
    let app: AppState
    let host: RemoteHost
    /// What each Mac connected was sent, by identity.
    var seen: [Data: Seen] = [:]
    struct Seen {
        var queue: RemoteQueue?
        var entries: Set<Int> = []
        /// Its series, as last sent.
        var series: [Int: RemoteSeries] = [:]
        /// Its jobs that ended, told once their images are sent.
        var ended: [RemoteEnded] = []
        var statistics: RemoteStatistics?
        /// The license asked (`licensing`), whether it was sent, and what.
        var license: RemoteLicense?
        /// The first `licensing` of a session is sent even when it is `nil`: a sheet the client left open
        /// across a cut closes with it.
        var licenseSent = false
    }
    /// Each client's last measured render, and its session (images, compute seconds) — in memory, kept
    /// across a reconnection (`statisticsSnapshot`).
    var measured: [Data: Measures] = [:]
    var sessions: [Data: (images: Int, seconds: Double)] = [:]
    /// The images each Mac removed, in memory until this Mac quits — kept across a cut, as its series
    /// is, so that ⌘Z still works after the link came back (`remove`, `restore`).
    private var trash: [Data: SessionTrash<Entry>] = [:]
    var lastCatalog: MirrorCatalog?
    /// The machine (`MirrorMachine.swift`).
    private let machine: MachinePublisher
    /// The library, common to all clients (`MirrorLibrary.swift`).
    private let library: MirrorLibraryHost
    var lastPreview: CGImage?
    var nextBlob: UInt32 = 1
    var queueScheduled = false, catalogScheduled = false
    /// The images each Mac sent, by blob, until a render takes them.
    var uploads: [Data: [UInt32: Data]] = [:]
    /// **What one Mac's images may hold here at once**: the queue full of edits (`AppState.maxQueueLength`), each
    /// with a few images of some megabytes. Past it the Mac is asking for more than the queue could take: its
    /// waiting renders are refused and its images dropped — a paired Mac does not pin the lender's memory.
    private static let uploadBytes = 512 << 20
    private static let uploadCount = 4 * AppState.maxQueueLength
    /// Renders and grids waiting for their images, in their order.
    var held: [(identity: Data, asked: Asked)] = []
    /// This Mac's signals, heard until lending stops (`stop`).
    private var listening: UUID?
    /// The files other Macs import here (`MirrorUpload.swift`).
    private lazy var importHub = UploadHub(app: app) { [weak self] m, identity in self?.host.send(m, to: identity) }
        importing: { [weak self] identity in self?.library.asked(by: identity) }
    enum Asked {
        case render(RemoteJob), grid(RemoteGridRequest)
        var references: [RemoteReference] {
            switch self {
            case .render(let r): r.references
            case .grid(let g): g.job.references
            }
        }
        /// The number the borrower waits for an answer under.
        var request: Int? {
            switch self {
            case .render(let r): r.ordinal
            case .grid(let g): g.request
            }
        }
    }

    /// `names`: the name a paired Mac was lent to under (`Pairings.Lent`), for the alerts that name it.
    init(app: AppState, host: RemoteHost, names: @escaping (Data) -> String?) {
        self.app = app; self.host = host
        machine = MachinePublisher(app: app, host: host)
        library = MirrorLibraryHost(app: app, host: host, names: names)
        let inbox = importHub.inbox
        // Pieces are written on the network's queue; only the whole percents and the end reach the main actor.
        host.receive(streams: { [weak self] identity, id, bytes, last in
            guard let (offer, step) = inbox.piece(identity: identity, id: id, bytes, last: last) else { return }
            if case .more(_, false) = step { return }
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.importHub.stepped(identity, offer, step) } }
        })
        listening = app.subscribe { [weak self] s in self?.signal(s) }
        watchQueue()
        watchCatalog()
    }

    /// Lending stops: this Mac's signals are no longer heard (the observations end with the server).
    func stop() {
        if let listening { app.unsubscribe(listening) }
        listening = nil
    }

    /// A client's job that ended is told to it alone, after its images.
    private func signal(_ s: Signal) {
        switch s {
        case .finished(let job, let issue):
            guard let owner = job.owner, seen[owner] != nil else { return }
            let done = app.completed[job.ordinal]
            // Only images the client will be sent: an end waits for its images.
            let sent = Set(app.history.map(\.ordinal))
            seen[owner]?.ended.append(RemoteEnded(ordinal: job.ordinal, issue: Self.remote(issue),
                                                  images: (done?.images ?? []).filter(sent.contains), warnings: done?.warnings ?? []))
            scheduleQueue()
        case .image(let job, let entry):
            guard let owner = job.owner else { return }
            var session = sessions[owner] ?? (0, 0)
            session.images += 1
            session.seconds += entry.timings.total
            sessions[owner] = session
        default:
            break
        }
    }

    // MARK: - A Mac joins, leaves, asks

    /// `theirs`: the client's hello, with the images it still holds (`Hello.held`).
    func joined(_ identity: Data, _ theirs: Hello) {
        // Joined again with no leave between (the old session died unseen): what it had under way went with it.
        if seen[identity] != nil { left(identity) }
        // The images the client names are not sent again — if this very launch numbered them.
        var first = Seen()
        first.entries = RemoteResume.alreadyHeld(by: theirs, launch: Hello.thisLaunch,
                                                 lenderHas: Set(app.history.filter { $0.owner == identity }.map(\.ordinal)))
        seen[identity] = first
        let catalog = lastCatalog ?? catalogSnapshot()
        lastCatalog = catalog
        host.send(.catalog(catalog), to: identity)
        machine.joined(identity)
        host.send(.synced(sync(for: identity)), to: identity)
        library.joined(identity)
        // The preview of a render under way is the client's alone; none otherwise — a stale one is cleared.
        sendPreview(to: identity)
        // The ends of its jobs, again: one that finished during a cut was told to a `Seen` that is gone.
        seen[identity]?.ended = endedJobs(of: identity)
        publish(to: identity, Tick(app))
    }

    func left(_ identity: Data) {
        library.left(identity)
        importHub.abort(identity)
        seen[identity] = nil
        uploads[identity] = nil
        held.removeAll { $0.identity == identity }
        // Another Mac's request may have waited behind one of this Mac's.
        release()
    }

    func received(blob id: UInt32, _ data: Data, from identity: Data) {
        guard seen[identity] != nil else { return }
        let mine = uploads[identity] ?? [:]
        guard mine.count < Self.uploadCount, mine.values.reduce(data.count, { $0 + $1.count }) <= Self.uploadBytes else {
            uploads[identity] = nil
            let refused = held.filter { $0.identity == identity }
            held.removeAll { $0.identity == identity }
            for r in refused { refuse(.app(.queueFull), r.asked, to: identity) }
            return
        }
        uploads[identity, default: [:]][id] = data
        release()
    }

    func handle(_ m: SessionMessage, from identity: Data) {
        switch m {
        case .render(let r):
            hold(.render(r), from: identity)
        case .grid(let g):
            hold(.grid(g), from: identity)
        case .quote(let q):
            host.send(.quoted(answer(q)), to: identity)
        case .enlarge(let o, let download):
            func refuse(_ r: RemoteRefusal) { host.send(.enlargeRefused(o, r), to: identity) }
            guard let e = own(o, identity) else { refuse(.app(.imageGone(o))); return }
            Task {
                do { try await app.enlarged(e, download: download) }
                catch EnlargeFailure.downloadRequired { host.send(.enlargeNeedsDownload(o), to: identity) }
                catch EnlargeFailure.refused(let sentence) { refuse(.text(sentence)) }
                catch EnlargeFailure.download { refuse(.app(.enlargeModelDownload)) }
                catch { refuse(.app(.enlargeFailed)) }
            }
        case .crop(let o, let box):
            guard let e = own(o, identity) else { host.send(.refused(.app(.imageGone(o))), to: identity); return }
            if let refusal = app.cropRefusalCode(e) { host.send(.refused(.app(refusal)), to: identity); return }
            // The editor's rules (`CropSpec.resolve`), as for a frame typed here: the box comes from another Mac.
            switch CropSpec.box(box).resolve(width: e.width, height: e.height) {
            case .success: Task { await app.crop(e, box: box) }
            case .failure(let f):
                let refusal: AppRefusal = switch f {
                case .cannotCrop(let side): .cropSmallest(side: side)
                case .belowFloor(let side): .cropNeeds(side: side)
                case .outside, .unsupportedRatio: .cropOutside(width: e.width, height: e.height)
                case .whole: .cropWhole
                }
                host.send(.refused(.app(refusal)), to: identity)
            }
        case .cancel:
            if app.currentJob?.owner == identity { app.cancel() }
        case .removeJob(let n):
            if let j = (app.file + app.awaitingLicense).first(where: { $0.ordinal == n && $0.owner == identity }) { app.remove(j) }
        case .stopAll:
            removeWaiting(of: identity)
            if app.currentJob?.owner == identity { app.cancel() }
        case .clearQueue:
            removeWaiting(of: identity)
        case .deleteImages(let ordinals):
            remove(ordinals, of: identity)
        case .restoreImages(let ordinals):
            restore(ordinals, for: identity)
        case .answerLicense(let a):
            answerLicense(a, from: identity)
        case .libraryAsk(let r):
            library.handle(r, from: identity)
        case .uploadOffer(let o):
            importHub.offer(o, from: identity)
        case .uploadCancel(let id):
            importHub.abort(identity, id: id)
        default:
            break
        }
    }

    /// One of the client's own images; another's is as if it did not exist.
    func own(_ ordinal: Int, _ identity: Data) -> Entry? {
        app.history.first { $0.ordinal == ordinal && $0.owner == identity }
    }

    private func removeWaiting(of identity: Data, where keep: (Job) -> Bool = { _ in true }) {
        for j in app.file + app.awaitingLicense where j.owner == identity && keep(j) { app.remove(j) }
    }

    // MARK: - Removal undone, and the join's sync

    /// A client's removal: its own images only, kept in its trash (at their place: the history is ordered by
    /// ordinal) until it leaves. This Mac's ⌘Z is not offered the removal — it was not this Mac's.
    private func remove(_ ordinals: [Int], of identity: Data) {
        let gone = Set(ordinals)
        let mine = app.history.filter { gone.contains($0.ordinal) && $0.owner == identity }
        for e in mine { trash[identity, default: SessionTrash()].put(e.ordinal, e) }
        app.undoManager?.disableUndoRegistration()
        app.delete(mine)
        app.undoManager?.enableUndoRegistration()
    }

    /// « Undo »: the client's images back, same ordinal, same owner, each where its ordinal belongs; they
    /// go back to it as any new image does (`publish(to:)`). What the trash no longer holds is said.
    private func restore(_ ordinals: [Int], for identity: Data) {
        let back = trash[identity]?.take(ordinals) ?? [:]
        for (o, e) in back.sorted(by: { $0.key < $1.key }) where !app.history.contains(where: { $0.ordinal == o }) {
            app.history.insert(e, at: app.history.firstIndex { $0.ordinal < o } ?? app.history.count)
        }
        if back.count < Set(ordinals).count { host.send(.refused(.app(.imagesNotRestorable)), to: identity) }
    }

    /// What this Mac still holds of `identity`'s: the images, and the jobs in the queue or ended.
    private func sync(for identity: Data) -> RemoteSync {
        var jobs = Set(app.completed.filter { $0.value.job.owner == identity }.keys)
        for j in [app.currentJob].compactMap({ $0 }) + app.file + app.awaitingLicense where j.owner == identity { jobs.insert(j.ordinal) }
        return RemoteSync(entries: app.history.filter { $0.owner == identity }.map(\.ordinal), jobs: jobs.sorted())
    }

    /// The ends of the client's jobs this session, their images the ones still in the history.
    private func endedJobs(of identity: Data) -> [RemoteEnded] {
        let kept = Set(app.history.filter { $0.owner == identity }.map(\.ordinal))
        return app.completed.filter { $0.value.job.owner == identity }.sorted { $0.key < $1.key }.map { o, done in
            RemoteEnded(ordinal: o, issue: Self.remote(done.issue), images: done.images.filter(kept.contains), warnings: done.warnings)
        }
    }
}
