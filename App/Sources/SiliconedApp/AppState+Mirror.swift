// **The other Mac's state, shown here** — the borrowing side of the remote Mac (`OtherMacs.swift`). A
// second `AppState`, `init(mirroring:)`, beside this Mac's own, which it never touches: back in local,
// everything is where it was.
//
//     the lender publishes (`Mirror.swift`)      the mirror keeps
//       catalog        ready models, traits, B     readySet, compatibleLoras, missing, memoryCeiling
//       queue          job running, waiting        currentJob, file, fraction, stage, steps, problem
//       entry + PNG    a history image             history (the thumbnail made here, from the PNG)
//       preview        a small PNG per step        preview
//
// - **It computes nothing**: its derived values (`busy`, `canRender`…) are the same
//   code reading what was published — the lender checks every job again before
//   queueing it.
// - **Its actions are sent** (`enqueue`, `cancel`, `remove`, `stopAll`, `clearQueue`, `delete`: each
//   starts with `if let mirror`). A job keeps the form's every choice, its seeds included: the lender
//   renders the same bits as if it had been asked on its own screen.
// - **Only this Mac's own** comes back: its jobs, its images, its series; what the other Mac renders
//   for itself or for other Macs is a count and a duration (`othersQueued`), said in the footer.
// - **Ordinals are the identities** (`t3`, `i7`), as for `silicontrol`: a job or an image keeps its
//   `UUID` here from one snapshot to the next, so that the selection holds.
// - **An edit's images go first**: a file's own bytes, an image in memory as a PNG, an image of the
//   lender's history by its ordinal (`RemoteReference`); the lender holds the render until they are there.
//   What the form cannot compute without the models — an edit's format — it asks
//   (`quote`), and waits for the answer before « Generate ».
// - **« Variations »** of an image: its own variations travel with the job, and an edit's images are
//   named by the image (`ofImage`) — they never came here.
// - **Enlarging and cropping** happen there, on its own pixels (`enlarge`, `crop` by ordinal); the
//   consent to download Apple's model is asked here. An enlargement's curtain lays under it its source
//   when that is in the history (`RemoteEntry.enlargedFrom`); else there is no curtain.
// - **Named subjects stay here** (incognito): a render that cites them sends the model's prompt and the
//   subjects' framed pictures after the edit's images (`RemoteJob.modelPrompt`, `subjectImages`), and
//   nothing else of them.
// - **A series** is asked whole (`grid`) and made there; the canvas shows the one it publishes, its cells'
//   images arriving as any history image, laid in their cells by their job.
// - **The statistics and the license sheet** are the other Mac's, for this Mac's renders (`MirrorStats.swift`).
// - **⌘Z after a removal**: the other Mac keeps the images removed in a trash of its session, per Mac;
//   « Undo » asks them back by ordinal (`restoreImages`) and « Redo » removes them again.
// - **After a cut** the other Mac says at each join what it still holds of ours (`synced`): what is not in
//   it is removed here, and a job it no longer knows ends as failed — nothing waits for ever.
// - **Its refusals are codes** (`RemoteRefusal`), said here in this Mac's language.

import AppKit
import Foundation
import ImageIO
import Siliconed

/// What the mirror sends through, and what it remembers between two snapshots.
@MainActor
final class MirrorLink {
    let macName: String
    private let sender: (SessionMessage) -> Void
    /// Jobs and images already built here, by ordinal: their `UUID` stays. A job stays once gone from the
    /// queue: its end (`ended`) comes after.
    var jobs: [Int: Job] = [:]
    private(set) var nextRequest = 1
    /// Who waits for the lender's word on a render (`silicontrol add`), by request number.
    var waiters: [Int: (RemoteAnswer) -> Void] = [:]
    /// The job each request asked for, until the lender numbers it.
    var requested: [Int: Job] = [:]
    var entries: [Int: Entry] = [:] {
        didSet { held.set(Set(entries.keys)) }
    }
    /// The ordinals of `entries`, for the link's next join: the lender does not send those again.
    let held = HeldImages()
    /// **Why a request would be lost**: the link is not joined (`AppState.linkBlocker`). Then `render` and
    /// `explore` answer a refusal at once, so that nothing waits for ever for a word that cannot come.
    var down: String?
    /// The lender's launch (`Hello.launch`) as its last `welcome` said it, and whether it changed since
    /// the images and jobs were numbered: its ordinals start again (`settle` forgets the old ones).
    var launch: UInt64?
    var restarted = false
    /// An entry waits for its PNG, the blob that follows it.
    var pending: [UInt32: RemoteEntry] = [:]
    /// Ends told before their images' PNGs came (messages never wait for blobs): held until they all have.
    var endings: [RemoteEnded] = []
    var previewBlob: UInt32?
    private let blobSender: (Data, UInt32) -> Void
    private var nextBlob: UInt32 = 1
    /// The quotes asked and not yet answered: each is asked once.
    var asked: Set<RemoteQuote> = []
    /// The edit's images the window's last series was asked with: its cells' own, as the canvas says them — and whether
    /// that series, not yet published back, opens on the canvas when it is.
    var exploreReferences: [ReferenceImage] = []
    var opensNextSeries = false
    /// The installed version of the family whose license is asked: its links, on the other Mac.
    var licenseVariant: Variant?
    /// **What was in flight when the link was cut**: the jobs not ended, the renders not answered. The next
    /// `synced` settles them — those it does not know will never end (`settle`).
    var cut: (jobs: Set<Int>, requests: Set<Int>)?
    /// Crops asked and not yet seen back: the one that comes registers its « Undo » (`cropped`).
    var cropsAsked = 0
    /// **Who waits for an enlargement** (`silicontrol enlarge`), by the ordinal of the image enlarged — the other
    /// Mac's word names it —, first come first served. Without one, that word goes to the window (the consent
    /// alert, the flash). A cut answers them all: their word will not come.
    var enlargeWaiters: [Int: [(Enlarged) -> Void]] = [:]
    enum Enlarged { case done(Entry), downloadRequired, refused(String) }

    /// The first who waits for the enlargement of image `source`, taken off the list; `nil`: nobody does.
    func enlargementWaiter(of source: Int) -> ((Enlarged) -> Void)? {
        guard var waiting = enlargeWaiters[source], !waiting.isEmpty else { return nil }
        let first = waiting.removeFirst()
        enlargeWaiters[source] = waiting.isEmpty ? nil : waiting
        return first
    }

    /// The link is cut: every enlargement waited for is refused, `why`.
    func failEnlargements(_ why: String) {
        let all = enlargeWaiters.values.flatMap { $0 }
        enlargeWaiters = [:]
        for w in all { w(.refused(why)) }
    }
    /// **The images being read off the main thread** (`AppState.receive(blob:)`), one after the other: each waits for the
    /// one before, so that they enter the history, and are signalled, in the order they came.
    var receiving: Task<Void, Never>?
    /// The grids the other Mac made for `silicontrol grid`: their cell jobs in order.
    var grids: [Int: [Int]] = [:]
    /// The files this Mac offers to import there (`MirrorUpload.swift`).
    let uploads: UploadBorrower

    init(macName: String, send: @escaping (SessionMessage) -> Void, sendBlob: @escaping (Data, UInt32) -> Void,
         uploadPort: UploadPort? = nil) {
        self.macName = macName; self.sender = send; self.blobSender = sendBlob
        uploads = UploadBorrower(port: uploadPort)
    }

    func send(_ m: SessionMessage) { sender(m) }

    func ask(_ q: RemoteQuote) {
        guard asked.insert(q).inserted else { return }
        sender(.quote(q))
    }

    /// The job, its images sent first. A file that can no longer be read refuses it here. `format: false`:
    /// the lender derives an edit's format from image 1. `answered`: the lender's word, when it comes.
    func render(_ job: Job, format: Bool = true, answered: ((RemoteAnswer) -> Void)? = nil) throws(EngineError) {
        if let down { answered?(.refused(.text(down))); return }
        var r = MirrorProjection.remote(job)
        r.references = try remote(job.references)
        if !format { r.width = 0; r.height = 0 }
        r.ordinal = nextRequest
        nextRequest += 1
        if let answered { waiters[r.ordinal] = answered }
        requested[r.ordinal] = job
        sender(.render(r))
    }

    /// A series, or a grid: the cells are made there (`enqueueGrid`), each judged as a render is, and the series
    /// published back (`series`). `answered` (`silicontrol grid`): the other Mac's word on the grid, its cell jobs
    /// numbered; without, the window's series, which opens on the canvas once published.
    func explore(_ grid: RenderGrid, settings: RenderSettings, references: [ReferenceImage], previews: Bool = true,
                 answered: ((RemoteAnswer) -> Void)? = nil) throws(EngineError) {
        if let down { answered?(.refused(.text(down))); return }
        var job = MirrorProjection.remote(settings, ordinal: 0, seeds: [], previews: previews, defaultSteps: nil)
        job.references = try remote(references)
        var request: Int?
        if let answered {
            request = nextRequest
            nextRequest += 1
            waiters[request!] = answered
        } else {
            exploreReferences = references
            opensNextSeries = true
        }
        sender(.grid(RemoteGridRequest(spec: RemoteGridSpec(grid), job: job, request: request)))
    }

    /// A grid answered: its cells are known, to follow.
    func made(_ grid: Int, cells: [Int]) { grids[grid] = cells }

    /// The word of the other Mac on every request numbered from `first` (those a gesture of the window,
    /// `vary`, has just sent): `answered` gets the request's rank among them.
    func awaiting(from first: Int, _ answered: @escaping (Int, RemoteAnswer) -> Void) {
        for r in first..<nextRequest { waiters[r] = { answered(r - first, $0) } }
    }

    private func remote(_ references: [ReferenceImage]) throws(EngineError) -> [RemoteReference] {
        try references.map { ref throws(EngineError) in
            RemoteReference(origin: try origin(ref), name: ref.name, width: ref.width, height: ref.height)
        }
    }

    private func origin(_ ref: ReferenceImage) throws(EngineError) -> RemoteReference.Origin {
        if let o = ref.ordinal { return .history(o) }
        let data: Data?, file: Bool
        switch ref.source {
        case .remote(let o): return o
        case .file(let url): data = try? Data(contentsOf: url); file = true
        case .bytes(let d): data = d; file = true
        case .image(let cg), .subject(let cg?, _): data = MirrorServer.png(cg); file = false
        case .subject(nil, _): data = nil; file = false
        }
        guard let data else { throw .imageUnreadable(file: ref.name) }
        let id = nextBlob
        nextBlob = nextBlob &+ 1 == 0 ? 1 : nextBlob &+ 1
        blobSender(data, id)
        return file ? .file(blob: id) : .image(blob: id)
    }
}

/// Between the app's types and what crosses (`Mirror.swift`), both ways.
enum MirrorProjection {
    static func remote(_ j: Job) -> RemoteJob {
        remote(j.settings, ordinal: j.ordinal, seeds: j.seeds, previews: j.previews, defaultSteps: j.defaultSteps)
    }

    static func remote(_ s: RenderSettings, ordinal: Int, seeds: [UInt64], previews: Bool, defaultSteps: Int?) -> RemoteJob {
        RemoteJob(ordinal: ordinal, model: s.identifier, modelName: s.modelName, prompt: s.prompt,
                  width: s.format.width, height: s.format.height, steps: s.steps, defaultSteps: defaultSteps,
                  loras: s.loras.map { RemoteLoRA(path: $0.path, strength: $0.strength) }, seeds: seeds,
                  detail: s.detail, solver: s.solver, previews: previews,
                  variations: s.variations, modelPrompt: s.modelPrompt, subjectImages: s.subjectImages > 0 ? s.subjectImages : nil)
    }

    static func settings(_ r: RemoteJob) -> RenderSettings {
        var s = RenderSettings(identifier: r.model, modelName: r.modelName, prompt: r.prompt,
                               format: RecommendedFormat(r.width, r.height), steps: r.steps,
                               loras: r.loras.map { LoRASlot(path: $0.path, strength: $0.strength) },
                               detail: r.detail, solver: r.solver)
        s.variations = r.variations
        s.modelPrompt = r.modelPrompt
        s.subjectImages = r.subjectImages ?? 0
        s.references = r.references.count
        return s
    }

    static func job(_ r: RemoteJob, ordinal: Int, plan: DenoisingPlan? = nil) -> Job {
        Job(ordinal: ordinal, settings: settings(r), seeds: r.seeds, references: [], previews: r.previews,
            defaultSteps: r.defaultSteps, plan: plan)
    }

    static let stageStates: [StageState: String] = [.idle: "idle", .active: "active", .done: "done", .failed: "failed"]
}

extension AppState {
    /// The chosen model's traits, as the other Mac published them.
    var traits: RemoteTraits? { mirrorCatalog?.traits[identifier] }

    /// The other Mac's answer, or `nil` while it is asked.
    func quoted(_ q: RemoteQuote) -> RemoteQuoteAnswer? {
        if let a = mirrorQuotes[q] { return a }
        mirror?.ask(q)
        return nil
    }

    /// The edit's quote: the format image 1 gives — only with an image to edit (`framingImage`).
    var editQuote: RemoteQuote? {
        guard mirror != nil, let first = framingImage else { return nil }
        return RemoteQuote(model: identifier, width: format.width, height: format.height,
                           first: [first.width, first.height], images: references.count)
    }

    /// The rack's three module names: from the chain here, from the traits in a mirror.
    var moduleNames: (text: String?, encoding: String?, decoding: String?) {
        if mirror != nil { return (traits?.text, traits?.encoding, traits?.decoding) }
        return (chain?.text.name, chain?.encoding?.name, chain?.decoding.name)
    }

    /// How the weights of the model on the rack are stored: read from its installed maps here, published by the lender in a mirror.
    var weightFormats: WeightFormats? {
        mirror != nil ? traits?.weights : model(identifier)?.formats
    }

    func readMirroredCatalog() {
        guard let c = mirrorCatalog else { compatibleLoras = []; missing = nil; return }
        // The lender names its LoRAs; their ranks stay there.
        compatibleLoras = (c.loras[identifier] ?? []).map { l in
            LoRACard(path: l.path, rawName: c.loraNames[l.path] ?? l.path, target: identifier, rank: 0, smallestRank: 0)
        }
        missing = readySet.contains(identifier) ? nil : String(localized: "Not installed on \(mirror?.macName ?? "")")
    }

    // MARK: - What the other Mac publishes

    func receive(_ m: SessionMessage) {
        guard let link = mirror else { return }
        switch m {
        case .catalog(let c):
            mirrorCatalog = c
            // What a quote answered may have changed with the models or B.
            mirrorQuotes = [:]; link.asked = []
            readySet = Set(c.ready)
            memoryCeiling = c.ceiling
            if !readySet.contains(identifier),
               let first = visibleReady.first?.id { identifier = first }
            readMirroredCatalog()
        case .queue(let q):
            func job(_ r: RemoteJob) -> Job {
                if let j = link.jobs[r.ordinal] { return j }
                let j = MirrorProjection.job(r, ordinal: r.ordinal)
                link.jobs[r.ordinal] = j
                return j
            }
            let running = q.current.map(job)
            let was = (job: currentJob?.ordinal, stage: stage, step: currentStep)
            if running?.ordinal != currentJob?.ordinal { batchSize = running?.seeds.count ?? 1 }
            currentJob = running
            file = q.waiting.map(job)
            awaitingLicense = q.awaitingLicense.map(job)
            fraction = q.fraction
            stage = q.stage.flatMap(Engine.Stage.init(rawValue:))
            currentStep = q.currentStep; totalSteps = q.totalSteps
            batchImage = q.batchImage; batchSize = q.batchSize
            let states = Dictionary(uniqueKeysWithValues: MirrorProjection.stageStates.map { ($1, $0) })
            stageStates = Dictionary(uniqueKeysWithValues: q.stageStates.compactMap { k, v in
                Engine.Stage(rawValue: k).flatMap { s in states[v].map { (s, $0) } }
            })
            stageSeconds = Dictionary(uniqueKeysWithValues: q.stageSeconds.compactMap { k, v in Engine.Stage(rawValue: k).map { ($0, v) } })
            stageModel = q.current?.model
            estimatedRemaining = q.remaining
            remainingInstant = Date()
            othersQueued = q.others; othersSeconds = q.othersSeconds
            enlarging = q.enlarging.flatMap { link.entries[$0]?.id }
            if running == nil { prerenders = [:] }
            // What the engine signals here, said from what the other Mac published (`silicontrol follow`).
            if let running {
                if running.ordinal != was.job { signal(.started(running)) }
                if let s = stage, s != was.stage || running.ordinal != was.job { signal(.stage(running, s, image: batchImage)) }
                if currentStep > 0, currentStep != was.step {
                    signal(.step(running, image: batchImage, index: currentStep, total: totalSteps, seconds: 0))
                }
            }
        case .entry(let e):
            link.pending[e.png] = e
        case .removed(let ordinals):
            let gone = Set(ordinals)
            let ids = Set(history.filter { gone.contains($0.ordinal) }.map(\.id))
            guard !ids.isEmpty else { return }
            let first = history.firstIndex { ids.contains($0.id) } ?? 0
            history.removeAll { ids.contains($0.id) }
            for o in ordinals { link.entries[o] = nil }
            marked.subtract(ids)
            if let s = selection, ids.contains(s) {
                selection = history.isEmpty ? nil : history[min(first, history.count - 1)].id
                anchor = selection
                loadSelectedImage()
            }
        case .preview(let id):
            link.previewBlob = id
            if id == nil, let c = currentJob { prerenders[c.ordinal] = nil }
        case .series(let x):
            guard let grid = try? x.spec.grid() else { return }
            let issues = x.issues.mapValues { i -> Issue in
                switch i {
                case .finished: .finished
                case .stopped: .stopped
                case .removed: .removed
                case .failed(let r): .failed(Problem(r))
                }
            }
            if series[x.ordinal] != nil { series[x.ordinal]!.issues = issues; return }
            // A series not seen yet: the window's last, or a `silicontrol grid`'s.
            let opens = link.opensNextSeries
            let refs = opens ? link.exploreReferences : []
            var run = GridRun(ordinal: x.ordinal, grid: grid, settings: MirrorProjection.settings(x.job),
                              jobs: x.cells.map(\.ordinal), references: x.job.references.count)
            run.cellJobs = zip(grid.cells, x.cells).map { cell, r in
                let j = MirrorProjection.job(r, ordinal: r.ordinal)
                // The cell's images are the ones this Mac sent.
                let used = cell.image.map { refs.indices.contains($0) ? [refs[$0]] : [] } ?? refs
                return Job(ordinal: j.ordinal, settings: j.settings, seeds: j.seeds, references: used, previews: j.previews,
                           defaultSteps: j.defaultSteps, plan: nil)
            }
            run.issues = issues
            series[x.ordinal] = run
            // Its cells are history images, come or to come as any other: found by their job.
            for (i, j) in run.jobs.enumerated() { gridCells[j] = (x.ordinal, i) }
            if opens {
                link.opensNextSeries = false
                followsRender = false
                shownSeries = x.ordinal
            }
        case .enlargeNeedsDownload(let o):
            // The consent is asked where the user is; the other Mac downloads Apple's model.
            if let waiter = link.enlargementWaiter(of: o) { waiter(.downloadRequired); return }
            if link.entries[o] != nil, Self.consentsToDownload() { link.send(.enlarge(o, download: true)) }
        case .enlargeRefused(let o, let r):
            let p = Problem(r)
            // An enlargement asked by `silicontrol` is answered there; its refusal is not the window's.
            if let waiter = link.enlargementWaiter(of: o) { waiter(.refused(p.text)); return }
            problem = p
            flash(p.text)
        case .answer(let request, let a):
            if case let .queued(o, _, w, h) = a, let j = link.requested[request], link.jobs[o] == nil {
                var settings = j.settings
                settings.format = RecommendedFormat(w, h)
                let known = Job(ordinal: o, settings: settings, seeds: j.seeds, references: j.references, previews: j.previews,
                                defaultSteps: j.defaultSteps, plan: nil)
                link.jobs[o] = known
            }
            // A grid's cells are known by their answer, as a render is: they may end before a snapshot shows them.
            if case let .gridQueued(grid, cells, _) = a {
                link.made(grid, cells: cells.map(\.ordinal))
                for r in cells where link.jobs[r.ordinal] == nil { link.jobs[r.ordinal] = MirrorProjection.job(r, ordinal: r.ordinal) }
            }
            link.requested[request] = nil
            if let waiter = link.waiters.removeValue(forKey: request) { waiter(a); return }
            if case .refused(let r) = a { let p = Problem(r); problem = p; flash(p.text) }
        case .ended(let e):
            link.endings.append(e)
            endHeld()
        case .quoted(let a):
            link.asked.remove(a.quote)
            mirrorQuotes[a.quote] = a
        case .statistics(let s):
            receiveStatistics(s)
        case .licensing(let l):
            receiveLicensing(l)
        case .refused(let r):
            let p = Problem(r)
            problem = p
            flash(p.text)
        case .synced(let s):
            settle(s)
        case .machine(let m):
            receiveMachine(m)
        case .libraryNews(let n):
            receiveLibrary(n)
        case .uploadAnswer, .uploadResult:
            receive(upload: m)
        default:
            break
        }
    }

    /// The held ends whose images all came: told, after them, as here (`silicontrol wait` emits each image first).
    func endHeld() {
        guard let link = mirror else { return }
        link.endings.removeAll { e in
            // Told again at a join (`synced`): one already ended here is not ended twice.
            if completed[e.ordinal] != nil { return true }
            guard e.images.allSatisfy({ link.entries[$0] != nil }) else { return false }
            // Known from the queue, or from the answer to its `render` (a job that ended between two snapshots).
            guard let job = link.jobs[e.ordinal] else { return true }
            let issue: Issue = switch e.issue {
            case .finished: .finished
            case .stopped: .stopped
            case .removed: .removed
            case .failed(let r): .failed(Problem(r))
            }
            if case .failed(let p) = issue { problem = p }
            complete(job, issue, images: e.images, warnings: e.warnings)
            return true
        }
    }

    func receive(blob id: UInt32, _ data: Data) {
        guard let link = mirror else { return }
        if id == link.previewBlob {
            // Not decoded here: ImageIO reads it where it is drawn.
            // The other Mac sends the running job's alone: kept by job, a cell keeps it once the next one runs.
            if let c = currentJob { prerenders[c.ordinal] = Self.decode(data) }
            return
        }
        guard let e = link.pending.removeValue(forKey: id), link.entries[e.ordinal] == nil else { return }
        // An enlargement's waiter is taken now, in the order the answers come: a refusal of the next enlargement that
        // arrives while this image is decoded must not take it. A new enlargement only: one sent again at a join
        // answers no request.
        let waiter = e.enlarged != nil && e.ordinal > lastImage ? e.enlargedFrom.flatMap(link.enlargementWaiter(of:)) : nil
        let before = link.receiving
        link.receiving = Task { [weak self] in
            await before?.value
            // The PNG decoded and its vignette reduced off the main thread, as a render's are (`FinishedImage`).
            let made = await Task.detached(priority: .userInitiated) { () -> (Decoded, Decoded)? in
                guard let image = Self.decode(data) else { return nil }
                return (Decoded(image), Decoded(FinishedImage.reduce(image, side: 240) ?? image))
            }.value
            guard let self, self.mirror === link, link.entries[e.ordinal] == nil else { return }
            guard let image = made?.0.image, let vignette = made?.1.image else {
                waiter?(.refused(String(localized: "This image could not be read.")))
                return
            }
            self.received(e, png: data, image: image, vignette: vignette, link: link, waiter: waiter)
        }
    }

    /// An image of the other Mac, decoded: into the history in its place, the newest shown while the canvas follows.
    private func received(_ e: RemoteEntry, png data: Data, image: CGImage, vignette: CGImage, link: MirrorLink,
                          waiter: ((MirrorLink.Enlarged) -> Void)?) {
        var entry = Self.entry(e, png: data, vignette: vignette)
        // Its edit images stay on the other Mac: named, for « Variations » (`RemoteReference.ofImage`).
        entry.references = e.job.references.map {
            ReferenceImage(source: .remote($0.origin), vignette: NSImage(systemSymbolName: "photo", accessibilityDescription: nil) ?? NSImage(),
                           name: $0.name, width: $0.width, height: $0.height)
        }
        // An enlargement's source, when it is still in the history: the curtain lays it under the enlarged
        // image. None otherwise — no curtain, as for an edit whose images stayed on the other Mac.
        entry.enlargedFrom = e.enlargedFrom.flatMap { link.entries[$0]?.png }
        link.entries[e.ordinal] = entry
        if e.cropped != nil, link.cropsAsked > 0, e.ordinal > lastImage {
            link.cropsAsked -= 1
            registerRemoteUndo(of: e.ordinal, name: String(localized: "Crop"))
        }
        waiter?(.done(entry))
        if let job = link.jobs[e.job.ordinal] { signal(.image(job, entry)) }
        // The history is newest first; entries arrive in their order, the first connection's oldest first.
        let at = history.firstIndex { $0.ordinal < e.ordinal } ?? history.count
        lastImage = max(lastImage, e.ordinal)
        admit(entry, image: image, at: at, shows: at == 0 && followsRender)
        endHeld()
    }

    private static func entry(_ e: RemoteEntry, png: Data, vignette: CGImage) -> Entry {
        let settings = MirrorProjection.settings(e.job)
        return Entry(ordinal: e.ordinal, job: e.job.ordinal, png: png,
                     vignette: vignette, width: e.width, height: e.height,
                     seed: e.seed, timings: e.timings, footprints: e.footprints, evaluations: e.evaluations,
                     settings: settings, metadata: e.metadata, enlarged: e.enlarged, cropped: e.cropped)
    }

    nonisolated static func decode(_ png: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(png as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
}

extension AppState {
    /// The session with the other Mac changed: said where the problems are said; joined again, the
    /// problem goes.
    func linkChanged(_ s: RemoteLink.State, name: String) {
        switch s {
        case .connected(let hello):
            linkBlocker = nil; mirror?.down = nil
            if problem?.code == "remote" { problem = nil }
            // Asked before a cut: asked again.
            mirror?.asked = []
            if let link = mirror, let launch = hello.launch {
                if let known = link.launch, known != launch { link.restarted = true }
                link.launch = launch
            }
        case .connecting, .off: break
        case .waiting:
            let lost = String(localized: "Lost “\(name)”: trying again…")
            linkBlocker = lost; mirror?.down = lost
            remoteConfirming = false; uploadLinkChanged(s); noteCut()
            mirror?.failEnlargements(lost)
            problem = { var p = Problem(lost); p.code = "remote"; return p }()
        case .refused:
            let why = OtherMacsSection.state(s, found: true)
            linkBlocker = why; mirror?.down = why
            remoteConfirming = false; uploadLinkChanged(s)
            mirror?.failEnlargements(why)
            problem = { var p = Problem(why); p.code = "remote"; return p }()
        }
    }
}

// MARK: - Removal undone, and the join's sync

extension AppState {
    /// A removal in a mirror, asked of the other Mac, which keeps the images in its session trash; « Undo »
    /// asks them back and « Redo » removes them again. After the switch back to this Mac the undo is vain.
    func removeRemotely(_ ordinals: [Int]) {
        guard let mirror, !ordinals.isEmpty else { return }
        mirror.send(.deleteImages(ordinals))
        undoManager?.registerUndo(withTarget: self) { $0.restoreRemotely(ordinals) }
        undoManager?.setActionName(ordinals.count == 1 ? String(localized: "Remove Image")
                                                       : String(localized: "Remove \(ordinals.count) Images"))
    }

    private func restoreRemotely(_ ordinals: [Int]) {
        guard let mirror else { return }
        mirror.send(.restoreImages(ordinals))
        undoManager?.registerUndo(withTarget: self) { $0.removeRemotely(ordinals) }
        undoManager?.setActionName(ordinals.count == 1 ? String(localized: "Remove Image")
                                                       : String(localized: "Remove \(ordinals.count) Images"))
    }

    /// An image the other Mac made from a crop asked here: undone by removing it, like any removal.
    private func registerRemoteUndo(of ordinal: Int, name: String) {
        undoManager?.registerUndo(withTarget: self) { $0.removeRemotely([ordinal]) }
        undoManager?.setActionName(name)
    }

    /// The link was cut: what is in flight is noted, for `settle`.
    func noteCut() {
        guard let link = mirror else { return }
        let held = Set(link.endings.map(\.ordinal))
        let jobs = Set(link.jobs.keys.filter { completed[$0] == nil && !held.contains($0) })
        let requests = Set(link.waiters.keys).union(link.requested.keys)
        // Lost again before the join's `synced` came: what the first cut noted is still in flight, and what
        // was asked in between too.
        if let before = link.cut { link.cut = (before.jobs.union(jobs), before.requests.union(requests)) }
        else { link.cut = (jobs, requests) }
    }

    /// **The other Mac's word on what it still holds of ours**, at each join: the history and the jobs are
    /// exactly those. An image or an end it does not list is dropped; a render or a job from before the
    /// cut that it does not know is over — said, so that no `wait` hangs.
    fileprivate func settle(_ s: RemoteSync) {
        guard let link = mirror else { return }
        let keep = Set(s.entries)
        let stale = Set(history.map(\.ordinal)).union(link.entries.keys).subtracting(keep).sorted()
        if !stale.isEmpty {
            receive(.removed(stale))
            for o in stale { link.entries[o] = nil }
        }
        // Ends waiting for an image that is gone would wait for ever.
        link.endings = link.endings.map { var e = $0; e.images = e.images.filter(keep.contains); return e }
        let known = Set(s.jobs)
        if let cut = link.cut {
            let lost = Problem(String(localized: "“\(link.macName)” no longer knows this render."))
            let held = Set(link.endings.map(\.ordinal))
            for o in cut.jobs.sorted() where !known.contains(o) && completed[o] == nil && !held.contains(o) {
                guard let job = link.jobs[o] else { continue }
                problem = lost
                complete(job, .failed(lost))
            }
            for r in cut.requests.sorted() {
                link.requested[r] = nil
                link.waiters.removeValue(forKey: r)?(.refused(.text(lost.text)))
            }
        }
        link.cut = nil
        // The lender restarted: it numbers from 1 again, so the jobs and ends kept under the old numbers
        // would be taken for the new ones (a job shown with another's prompt, an end swallowed).
        if link.restarted {
            link.restarted = false
            link.jobs = [:]; link.grids = [:]; link.endings = []
            series = [:]; gridCells = [:]; shownSeries = nil
            forgetCompletedJobs()
        }
        endHeld()
    }
}

extension AppState {
    /// **The form that crosses a switch of Mac** (`MainWindow.compute(on:)`): what the user has typed and chosen,
    /// carried from the state left to the state shown. The one place that lists it — a setting added to the
    /// form is added here. The LoRAs, the edit's images and the starting image are not: they name files of one Mac's library
    /// or live in one Mac's memory — nor the brackets that name a LoRA.
    func copyForm(from other: AppState) {
        // The model first: changing it resets the steps and a start strength chosen for another family.
        prompt = other.prompt; identifier = other.identifier; steps = other.steps
        orientation = other.orientation; format = other.format; batch = other.batch
        // The seed before the variations: setting it drops them (`seed.didSet`).
        fixedSeed = other.fixedSeed; seed = other.seed; variations = other.variations
        detail = other.detail; solver = other.solver; chosenStartStrength = other.chosenStartStrength
        previews = other.previews
        brackets = other.brackets.filter { b in
            switch b.key {
            case .loraStrength, .comparedLoRAs: false
            default: true
            }
        }
        seriesPlaced = other.seriesPlaced
    }
}
