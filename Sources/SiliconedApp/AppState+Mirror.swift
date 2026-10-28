// **The other Mac's state, shown here** — the borrowing side of the remote Mac (`OtherMacs.swift`). A
// second `AppState`, `init(mirroring:)`, beside this Mac's own, which it never touches: back in local,
// everything is where it was.
//
//     the mini publishes (`Mirror.swift`)        the mirror keeps
//       catalog        ready models, traits, B     readySet, compatibleLoras, missing, memoryCeiling
//       queue          job running, waiting        currentJob, file, fraction, stage, steps, problem
//       entry + PNG    a history image             history (the thumbnail made here, from the PNG)
//       preview        a small PNG per step        preview
//
// - **It computes nothing**: its derived values (`busy`, `canRender`, `memoryRefusal`…) are the same
//   code reading what was published — B is the mini's, and the mini checks every job again before
//   queueing it.
// - **Its actions are sent** (`enqueue`, `cancel`, `remove`, `stopAll`, `clearQueue`, `delete`: each
//   starts with `if let mirror`). A job keeps the form's every choice, its seeds included: the mini
//   renders the same bits as if it had been asked on its own screen.
// - **Only this Mac's own** comes back: its jobs, its images, its exploration; what the other Mac renders
//   for itself or for other Macs is a count and a duration (`othersQueued`), said in the footer.
// - **Ordinals are the identities** (`t3`, `i7`), as for `silicontrol`: a job or an image keeps its
//   `UUID` here from one snapshot to the next, so that the selection holds.
// - **An edit's images go first**: a file's own bytes, an image in memory as a PNG, an image of the
//   mini's history by its ordinal (`RemoteReference`); the mini holds the render until they are there.
//   What the form cannot compute without the maps — an edit's format, a need with images — it asks
//   (`quote`), and waits for the answer before « Generate ».
// - **« Variations »** of an image: its own variations travel with the job, and an edit's images are
//   named by the image (`ofImage`) — they never came here.
// - **Enlarging and cropping** happen there, on its own pixels (`enlarge`, `crop` by ordinal); the
//   consent to download Apple's model is asked here. An enlargement's curtain lays under it its source
//   when that is in the history (`RemoteEntry.enlargedFrom`); else there is no curtain.
// - **Named subjects stay here** (incognito): a render that cites them sends the model's prompt and the
//   subjects' framed pictures after the edit's images (`RemoteJob.modelPrompt`, `subjectImages`), and
//   nothing else of them.
// - **The exploration** is asked whole (`grid`) and made there; the panel shows the one it publishes,
//   its cells' images arriving one by one (`cellImage`). « Finish This Image » is an ordinary render.
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
    /// Who waits for the mini's word on a render (`silicontrol add`), by request number.
    var waiters: [Int: (RemoteAnswer) -> Void] = [:]
    /// The job each request asked for, until the mini numbers it.
    var requested: [Int: Job] = [:]
    var entries: [Int: Entry] = [:]
    /// An entry waits for its PNG, the blob that follows it.
    var pending: [UInt32: RemoteEntry] = [:]
    /// Ends told before their images' PNGs came (messages never wait for blobs): held until they all have.
    var endings: [RemoteEnded] = []
    var previewBlob: UInt32?
    private let blobSender: (Data, UInt32) -> Void
    private var nextBlob: UInt32 = 1
    /// The quotes asked and not yet answered: each is asked once.
    var asked: Set<RemoteQuote> = []
    /// The exploration's cell images, by grid then cell; a cell image waits for its PNG.
    var cells: [Int: [Int: Entry]] = [:]
    var pendingCells: [UInt32: (grid: Int, cell: Int, entry: RemoteEntry)] = [:]
    /// The edit's images the last exploration was asked with: what « Use These Settings » puts back.
    var exploreReferences: [ReferenceImage] = []
    /// The installed version of the family whose license is asked: its links, on the other Mac.
    var licenseVariant: Variant?
    /// **What was in flight when the link was cut**: the jobs not ended, the renders not answered. The next
    /// `synced` settles them — those it does not know will never end (`settle`).
    var cut: (jobs: Set<Int>, requests: Set<Int>)?
    /// Crops asked and not yet seen back: the one that comes registers its « Undo » (`cropped`).
    var cropsAsked = 0
    /// **Who waits for an enlargement** (`silicontrol enlarge`), first come first served: the other Mac
    /// enlarges one image at a time. Without one, the word of the other Mac goes to the window (the
    /// consent alert, the flash).
    var enlargeWaiters: [(Enlarged) -> Void] = []
    enum Enlarged { case done(Entry), downloadRequired, refused(String) }
    /// The grids the other Mac made for `silicontrol grid`: their cell jobs in order, and whether they are sketches.
    var grids: [Int: (jobs: [Int], sketch: Bool)] = [:]
    private var sketches: [Int: Bool] = [:]

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
    /// the mini derives an edit's format from image 1. `answered`: the mini's word, when it comes.
    func render(_ job: Job, format: Bool = true, answered: ((RemoteAnswer) -> Void)? = nil) throws(EngineError) {
        var r = MirrorProjection.remote(job)
        r.references = try remote(job.references)
        if !format { r.width = 0; r.height = 0 }
        r.ordinal = nextRequest
        nextRequest += 1
        if let answered { waiters[r.ordinal] = answered }
        requested[r.ordinal] = job
        sender(.render(r))
    }

    /// An exploration: the cells are made there (`enqueueGrid`), where a sketch's stop is known.
    /// `answered` (`silicontrol grid`): the other Mac's word on the grid, its cell jobs numbered; `sketch:
    /// false` is a grid whose cells join the history, and its sheet.
    func explore(_ grid: RenderGrid, settings: RenderSettings, references: [ReferenceImage], stopAfter: Int,
                 neuralEngine: Bool?, sketch: Bool = true, previews: Bool = true,
                 answered: ((RemoteAnswer) -> Void)? = nil) throws(EngineError) {
        var job = MirrorProjection.remote(settings, ordinal: 0, seeds: [], previews: previews, defaultSteps: nil,
                                          neuralEngine: neuralEngine)
        job.references = try remote(references)
        if sketch { exploreReferences = references }
        var request: Int?
        if let answered {
            request = nextRequest
            nextRequest += 1
            waiters[request!] = answered
            sketches[request!] = sketch
        }
        sender(.grid(RemoteGridRequest(spec: RemoteGridSpec(grid), job: job, sketch: sketch, stopAfter: stopAfter,
                                       request: request)))
    }

    /// A grid answered: its cells are known, to follow.
    func made(_ grid: Int, cells: [Int], request: Int) {
        grids[grid] = (cells, sketches.removeValue(forKey: request) ?? true)
    }

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
        remote(j.settings, ordinal: j.ordinal, seeds: j.seeds, previews: j.previews, defaultSteps: j.defaultSteps,
               neuralEngine: j.neuralEngine)
    }

    static func remote(_ s: RenderSettings, ordinal: Int, seeds: [UInt64], previews: Bool, defaultSteps: Int?,
                       neuralEngine: Bool?) -> RemoteJob {
        RemoteJob(ordinal: ordinal, model: s.identifier, modelName: s.modelName, prompt: s.prompt,
                  width: s.format.width, height: s.format.height, steps: s.steps, defaultSteps: defaultSteps,
                  loras: s.loras.map { RemoteLoRA(path: $0.path, strength: $0.strength) }, seeds: seeds,
                  detail: s.detail, solver: s.solver, previews: previews, neuralEngine: neuralEngine,
                  variations: s.variations, modelPrompt: s.modelPrompt, subjectImages: s.subjectImages > 0 ? s.subjectImages : nil)
    }

    static func settings(_ r: RemoteJob) -> RenderSettings {
        var s = RenderSettings(identifier: r.model, modelName: r.modelName, prompt: r.prompt,
                               format: RecommendedFormat(r.width, r.height), steps: r.steps,
                               loras: r.loras.map { LoRASlot(path: $0.path, strength: $0.strength) },
                               detail: r.detail, solver: r.solver)
        s.neuralEngine = r.neuralEngine ?? false
        s.variations = r.variations
        s.modelPrompt = r.modelPrompt
        s.subjectImages = r.subjectImages ?? 0
        s.references = r.references.count
        return s
    }

    static func job(_ r: RemoteJob, ordinal: Int, plan: DenoisingPlan? = nil) -> Job {
        var j = Job(ordinal: ordinal, settings: settings(r), seeds: r.seeds, references: [], previews: r.previews,
                    defaultSteps: r.defaultSteps, plan: plan)
        j.neuralEngine = r.neuralEngine
        return j
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

    /// The edit's quote: the format image 1 gives, and the need with every image read.
    var editQuote: RemoteQuote? {
        guard mirror != nil, let first = references.first else { return nil }
        return RemoteQuote(model: identifier, width: format.width, height: format.height,
                           first: [first.width, first.height], images: references.count + subjectImageCount)
    }

    /// The rack's three module names: from the chain here, from the traits in a mirror.
    var moduleNames: (text: String?, encoding: String?, decoding: String?) {
        if mirror != nil { return (traits?.text, traits?.encoding, traits?.decoding) }
        return (chain?.text.name, chain?.encoding?.name, chain?.decoding.name)
    }

    func readMirroredCatalog() {
        guard let c = mirrorCatalog else { compatibleLoras = []; missing = nil; return }
        compatibleLoras = (c.loras[identifier] ?? []).compactMap { l in
            LoRACard(header: ["kind": "lora", "cible": ["modele": identifier], "nom": c.loraNames[l.path] ?? l.path],
                     path: l.path)
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
            // Not `fasterRendering`: its setter writes this Mac's preference and the engine's switch.
            settingsRead = c.settingsRead
            if !isVisible(identifier) || !readySet.contains(identifier),
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
            if running == nil { preview = nil }
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
            if id == nil { preview = nil }
        case .explored(let x):
            guard let x, let grid = try? x.spec.grid() else { explored = nil; return }
            let refs = link.exploreReferences
            var run = GridRun(ordinal: x.ordinal, grid: grid, settings: MirrorProjection.settings(x.job),
                              jobs: x.cells.map(\.ordinal), references: x.job.references.count, sketch: x.sketch)
            run.cellJobs = zip(grid.cells, x.cells).map { cell, r in
                let j = MirrorProjection.job(r, ordinal: r.ordinal)
                // The cell's images are the ones this Mac sent: « Use These Settings » puts them back.
                let used = cell.image.map { refs.indices.contains($0) ? [refs[$0]] : [] } ?? refs
                return Job(ordinal: j.ordinal, settings: j.settings, seeds: j.seeds, references: used, previews: j.previews,
                           defaultSteps: j.defaultSteps, plan: nil)
            }
            run.entries = link.cells[x.ordinal] ?? [:]
            run.issues = x.issues.mapValues { i -> Issue in
                switch i {
                case .finished: .finished
                case .stopped: .stopped
                case .removed: .removed
                case .failed(let r): .failed(Problem(r))
                }
            }
            link.cells = link.cells.filter { $0.key == x.ordinal }
            explored = run
        case .cellImage(let grid, let cell, let e):
            link.pendingCells[e.png] = (grid, cell, e)
        case .enlargeNeedsDownload(let o):
            // The consent is asked where the user is; the other Mac downloads Apple's model.
            if !link.enlargeWaiters.isEmpty { link.enlargeWaiters.removeFirst()(.downloadRequired); return }
            if link.entries[o] != nil, Self.consentsToDownload() { link.send(.enlarge(o, download: true)) }
        case .answer(let request, let a):
            if case let .queued(o, _, w, h) = a, let j = link.requested[request], link.jobs[o] == nil {
                var settings = j.settings
                settings.format = RecommendedFormat(w, h)
                var known = Job(ordinal: o, settings: settings, seeds: j.seeds, references: j.references, previews: j.previews,
                                defaultSteps: j.defaultSteps, plan: nil)
                known.neuralEngine = j.neuralEngine
                link.jobs[o] = known
            }
            // A grid's cells are known by their answer, as a render is: they may end before a snapshot shows them.
            if case let .gridQueued(grid, cells, _) = a {
                link.made(grid, cells: cells.map(\.ordinal), request: request)
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
            measureRefused(r)
            let p = Problem(r)
            // An enlargement asked by `silicontrol` is answered there; its refusal is not the window's.
            if !link.enlargeWaiters.isEmpty { link.enlargeWaiters.removeFirst()(.refused(p.text)); return }
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
            preview = Self.decode(data)
            return
        }
        if let (grid, cell, e) = link.pendingCells.removeValue(forKey: id) {
            guard let image = Self.decode(data) else { return }
            var entry = Self.entry(e, png: data, image: image)
            entry.sketch = e.evaluations
            link.cells[grid, default: [:]][cell] = entry
            if explored?.ordinal == grid { explored?.entries[cell] = entry }
            // As here, a sketch's image is told to whoever follows its job (`silicontrol grid --wait`).
            if let run = explored, run.ordinal == grid, run.cellJobs.indices.contains(cell) { signal(.image(run.cellJobs[cell], entry)) }
            return
        }
        guard let e = link.pending.removeValue(forKey: id), link.entries[e.ordinal] == nil,
              let image = Self.decode(data) else { return }
        var entry = Self.entry(e, png: data, image: image)
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
        if entry.enlarged != nil, !link.enlargeWaiters.isEmpty { link.enlargeWaiters.removeFirst()(.done(entry)) }
        if let job = link.jobs[e.job.ordinal] { signal(.image(job, entry)) }
        // The history is newest first; entries arrive in their order, the first connection's oldest first.
        let at = history.firstIndex { $0.ordinal < e.ordinal } ?? history.count
        history.insert(entry, at: at)
        lastImage = max(lastImage, e.ordinal)
        if at == 0, followsRender {
            selectedImage = (entry.id, image)
            selection = entry.id
            marked = []; anchor = entry.id
        }
        endHeld()
    }

    private static func entry(_ e: RemoteEntry, png: Data, image: CGImage) -> Entry {
        var settings = MirrorProjection.settings(e.job)
        settings.neuralEngine = e.neuralEngine
        let grid = e.sheet.flatMap { s in (try? s.spec.grid()).map { GridInfo(ordinal: s.ordinal, grid: $0, empty: s.empty) } }
        return Entry(ordinal: e.ordinal, job: e.job.ordinal, png: png,
                     vignette: FinishedImage.reduce(image, side: 240) ?? image, width: e.width, height: e.height,
                     seed: e.seed, timings: e.timings, footprints: e.footprints, evaluations: e.evaluations,
                     settings: settings, metadata: e.metadata, grid: grid, enlarged: e.enlarged, cropped: e.cropped)
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
        case .connected:
            if problem?.code == "remote" { problem = nil }
            // Asked before a cut: asked again.
            mirror?.asked = []
        case .connecting, .off: break
        case .waiting: uploadLinkChanged(s); noteCut(); problem = { var p = Problem(String(localized: "Lost “\(name)”: trying again…")); p.code = "remote"; return p }()
        case .refused: uploadLinkChanged(s); problem = { var p = Problem(OtherMacsSection.state(s, found: true)); p.code = "remote"; return p }()
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
        guard let link = mirror, link.cut == nil else { return }
        let held = Set(link.endings.map(\.ordinal))
        link.cut = (Set(link.jobs.keys.filter { completed[$0] == nil && !held.contains($0) }),
                    Set(link.waiters.keys).union(link.requested.keys))
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
        endHeld()
    }
}
