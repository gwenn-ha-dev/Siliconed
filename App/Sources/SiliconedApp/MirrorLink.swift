// The borrowing side's link to the other Mac (`MirrorLink`), and the projection of a job onto what crosses (`MirrorProjection`).

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
    /// The edit's images the last series was asked with (the window's or `silicontrol grid`'s): its cells' own, as the canvas
    /// says them — and whether
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
    /// published back (`series`), which opens on the canvas once published — the window's series and `silicontrol grid`'s
    /// alike, as on this Mac. `answered` (`silicontrol grid`): the other Mac's word on the grid, its cell jobs numbered.
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
        }
        exploreReferences = references
        opensNextSeries = true
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
