// `MirrorServer`: what a client is sent — its queue, its series, the catalog, its images.

import AppKit
import Foundation
import ImageIO
import Siliconed

extension MirrorServer {
    /// **What every client's snapshot reads, read once per publication**: a burst of changes publishes to every
    /// client, and each would otherwise walk the whole history and estimate the whole queue.
    struct Tick {
        /// The history's images by owner, newest first.
        let history: [Data: [Entry]]
        let remaining: Double?
        let estimates: [Int: Double]

        @MainActor init(_ app: AppState) {
            history = Dictionary(grouping: app.history.lazy.filter { $0.owner != nil }, by: { $0.owner! })
            remaining = app.remaining(at: Date())
            var e: [Int: Double] = [:]
            for j in app.file + app.awaitingLicense { if let s = app.estimation(j) { e[j.ordinal] = s } }
            estimates = e
        }
    }

    /// The queue as `identity` sees it: its jobs, the running one's progress if it is its own; the rest a
    /// count, and what it takes.
    func queueSnapshot(for identity: Data, _ tick: Tick) -> RemoteQueue {
        var q = RemoteQueue()
        let mine = { (j: Job) in j.owner == identity }
        q.waiting = app.file.filter(mine).map(MirrorProjection.remote)
        q.awaitingLicense = app.awaitingLicense.filter(mine).map(MirrorProjection.remote)
        q.enlarging = app.enlarging.flatMap { id in tick.history[identity]?.first { $0.id == id }?.ordinal }
        var seconds: Double? = 0
        if let c = app.currentJob, !mine(c) {
            q.others += 1
            seconds = tick.remaining
        }
        for t in app.file where !mine(t) {
            q.others += 1
            seconds = seconds.flatMap { s in tick.estimates[t.ordinal].map { s + $0 } }
        }
        q.othersSeconds = q.others > 0 ? seconds : nil
        guard let c = app.currentJob, mine(c) else { return q }
        q.current = MirrorProjection.remote(c)
        q.fraction = app.fraction
        q.stage = app.stage?.rawValue
        q.currentStep = app.currentStep; q.totalSteps = app.totalSteps
        q.batchImage = app.batchImage; q.batchSize = app.batchSize
        q.stageStates = Dictionary(uniqueKeysWithValues: app.stageStates.compactMap { k, v in
            MirrorProjection.stageStates[v].map { (k.rawValue, $0) }
        })
        q.stageSeconds = Dictionary(uniqueKeysWithValues: app.stageSeconds.map { ($0.key.rawValue, $0.value) })
        q.remaining = tick.remaining
        return q
    }

    static func remote(_ i: Issue) -> RemoteIssue {
        switch i {
        case .finished: .finished
        case .stopped: .stopped
        case .removed: .removed
        case .failed(let p): .failed(RemoteRefusal(p))
        }
    }

    func remote(_ r: GridRun) -> RemoteSeries {
        RemoteSeries(
            ordinal: r.ordinal, spec: RemoteGridSpec(r.grid),
            job: MirrorProjection.remote(r.settings, ordinal: 0, seeds: [], previews: true, defaultSteps: nil),
            cells: r.cellJobs.map(MirrorProjection.remote),
            issues: r.issues.mapValues(Self.remote))
    }

    func catalogSnapshot() -> MirrorCatalog {
        var traits: [String: RemoteTraits] = [:], loras: [String: [RemoteLoRA]] = [:], names: [String: String] = [:]
        for id in app.readySet {
            if let m = app.model(id) {
                let d = m.chain.denoising
                traits[id] = RemoteTraits(maxReferences: d.maxReferences, readsImages: m.chain.text.output.readsImages,
                                          acceptsDetail: d.acceptsDetail,
                                          steps: d.scheduledSteps,
                                          text: m.chain.text.name, encoding: m.chain.encoding?.name,
                                          decoding: m.chain.decoding.name, weights: m.formats)
            }
            let held = app.library.loras(for: id)
            loras[id] = held.map { RemoteLoRA(path: $0.path, strength: 0) }
            for l in held { names[l.path] = l.rawName }
        }
        return MirrorCatalog(ready: app.readySet.sorted(), traits: traits, loras: loras, loraNames: names,
                             ceiling: app.memoryCeiling,
                             enlarges: SuperResolution.available)
    }

    func send(_ e: Entry, to identity: Data) {
        let id = blob()
        host.send(.entry(remote(e, png: id)), to: identity)
        host.send(blob: e.png, id: id, to: identity)
    }

    /// A history image, as the other Mac reads it: its job, its edit's images named by this image.
    private func remote(_ e: Entry, png id: UInt32) -> RemoteEntry {
        var job = MirrorProjection.remote(e.settings, ordinal: e.job, seeds: [e.seed], previews: false, defaultSteps: nil)
        job.references = e.references.enumerated().map { i, ref in
            RemoteReference(origin: .ofImage(e.ordinal, index: i), name: ref.name, width: ref.width, height: ref.height)
        }
        let r = RemoteEntry(ordinal: e.ordinal, job: job,
                            width: e.width, height: e.height, seed: e.seed, timings: e.timings, footprints: e.footprints,
                            evaluations: e.evaluations, metadata: e.metadata, png: id,
                            enlarged: e.enlarged, cropped: e.cropped,
                            enlargedFrom: e.enlargedFrom.flatMap { from in app.history.first { $0.owner == e.owner && $0.png == from }?.ordinal })
        return r
    }

    func blob() -> UInt32 {
        defer { nextBlob &+= 1; if nextBlob == 0 { nextBlob = 1 } }
        return nextBlob
    }

    static func png(_ image: CGImage) -> Data? {
        let d = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(d, "public.png" as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, nil)
        return CGImageDestinationFinalize(dest) ? d as Data : nil
    }
}
