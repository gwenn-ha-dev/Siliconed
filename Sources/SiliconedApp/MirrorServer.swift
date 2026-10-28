// **This Mac's state, published to the Macs it lends itself to** — the lending side of the mirror
// (`AppState+Mirror.swift` is the other). It reads the app's state as the window does and sends what
// changed; what comes back is a request it judges as `silicontrol add` would, then queues.
//
// - **Observed, not hooked**: `withObservationTracking` over what a snapshot reads — the next change
//   sends a new one, 50 ms later at most so that a burst of changes (a step: fraction, step, preview)
//   goes as one. A snapshot equal to the last is not sent.
// - **The catalog is read apart** (it reads LoRAs from the disk): only when the ready models, the map or
//   « Faster Rendering » change.
// - **Every request is judged again**: the model ready, its schedule, the format's rules, B, the LoRAs
//   forged for it. The borrowing Mac's form judged first, with this Mac's B and traits, but its word is
//   never the last.
// - **An edit's images come as blobs before their render**, but a message never waits for a blob
//   (`Frames`): the render may come first, and is held until they are all there — in its order, the
//   renders after it held too. The bytes live in memory only, until the job is made (incognito).

import AppKit
import Foundation
import ImageIO
import Siliconed

@MainActor
final class MirrorServer {
    private let app: AppState
    private let host: RemoteHost
    private var clients: Set<Data> = []
    private var lastQueue: RemoteQueue?
    private var lastCatalog: MirrorCatalog?
    private var sentEntries: Set<Int> = []
    private var lastPreview: CGImage?
    private var nextBlob: UInt32 = 1
    private var queueScheduled = false, catalogScheduled = false
    /// The images each Mac sent, by blob, until a render takes them.
    private var uploads: [Data: [UInt32: Data]] = [:]
    /// Renders and grids waiting for their images, in their order.
    private var held: [(identity: Data, asked: Asked)] = []
    private enum Asked {
        case render(RemoteJob), grid(RemoteGridRequest)
        var references: [RemoteReference] {
            switch self {
            case .render(let r): r.references
            case .grid(let g): g.job.references
            }
        }
    }
    private var lastExploration: RemoteExploration?
    /// The exploration's cells whose image was sent: [grid, cell].
    private var sentCells: Set<[Int]> = []

    init(app: AppState, host: RemoteHost) {
        self.app = app; self.host = host
        // What is there now goes to each Mac as it joins (`joined`); what comes after, to all.
        sentEntries = Set(app.history.map(\.ordinal))
        watchQueue()
        watchCatalog()
    }

    // MARK: - A Mac joins, leaves, asks

    func joined(_ identity: Data) {
        clients.insert(identity)
        let catalog = lastCatalog ?? catalogSnapshot()
        lastCatalog = catalog
        host.send(.catalog(catalog), to: identity)
        host.send(.queue(lastQueue ?? queueSnapshot()), to: identity)
        // Oldest first: the mirror inserts each at its place.
        for e in app.history.reversed() { send(e, to: identity) }
        host.send(.explored(explorationSnapshot()), to: identity)
        if let r = app.explored {
            for (cell, e) in r.entries { send(e, cell: [r.ordinal, cell], to: identity); sentCells.insert([r.ordinal, cell]) }
        }
    }

    func left(_ identity: Data) {
        clients.remove(identity)
        uploads[identity] = nil
        held.removeAll { $0.identity == identity }
    }

    func received(blob id: UInt32, _ data: Data, from identity: Data) {
        guard clients.contains(identity) else { return }
        uploads[identity, default: [:]][id] = data
        release()
    }

    func handle(_ m: SessionMessage, from identity: Data) {
        switch m {
        case .render(let r):
            held.append((identity, .render(r)))
            release()
        case .grid(let g):
            held.append((identity, .grid(g)))
            release()
        case .addSheet(let n):
            if let r = app.explored, r.ordinal == n { app.addSheet(r) }
        case .quote(let q):
            host.send(.quoted(answer(q)), to: identity)
        case .enlarge(let o, let download):
            guard let e = app.history.first(where: { $0.ordinal == o }) else { return }
            Task {
                do { try await app.enlarged(e, download: download) }
                catch EnlargeFailure.downloadRequired { host.send(.enlargeNeedsDownload(o), to: identity) }
                catch EnlargeFailure.refused(let sentence) { host.send(.refused(sentence), to: identity) }
                catch EnlargeFailure.download { host.send(.refused(String(localized: "The enlargement model could not be downloaded")), to: identity) }
                catch { host.send(.refused(String(localized: "This image could not be enlarged")), to: identity) }
            }
        case .crop(let o, let box):
            guard let e = app.history.first(where: { $0.ordinal == o }) else { return }
            if let refusal = app.cropRefusal(e) { host.send(.refused(refusal), to: identity); return }
            Task { await app.crop(e, box: box) }
        case .cancel:
            app.cancel()
        case .removeJob(let n):
            if let j = (app.file + app.awaitingLicense).first(where: { $0.ordinal == n }) { app.remove(j) }
        case .stopAll:
            app.stopAll()
        case .clearQueue:
            app.clearQueue()
        case .deleteImages(let ordinals):
            let gone = Set(ordinals)
            app.delete(app.history.filter { gone.contains($0.ordinal) })
        default:
            break
        }
    }

    /// The renders whose images are all here, in order: the first still waiting holds the others.
    private func release() {
        while let (identity, asked) = held.first {
            let sent = uploads[identity] ?? [:]
            guard asked.references.allSatisfy({ $0.blob.map { sent[$0] != nil } ?? true }) else { return }
            held.removeFirst()
            do {
                switch asked {
                case .render(let r): app.enqueue([try admit(r, images: sent)])
                case .grid(let g): try admit(g, images: sent)
                }
            } catch let e as Refused {
                host.send(.refused(e.text), to: identity)
            } catch {
                host.send(.refused(Problem(EngineError(error)).text), to: identity)
            }
            for b in asked.references.compactMap(\.blob) { uploads[identity]?[b] = nil }
        }
    }

    /// **What the borrowing Mac's form cannot compute**: an edit's format (the denoiser's rule, capped as
    /// the app caps it), the need with images — from this Mac's maps, as its own form would.
    private func answer(_ q: RemoteQuote) -> RemoteQuoteAnswer {
        var f = RecommendedFormat(q.width, q.height)
        var need: Int?
        if app.readySet.contains(q.model), let card = app.cards.first(where: { $0.id == q.model }) {
            if let first = q.first, first.count == 2, let m = app.model(q.model) {
                let e = m.chain.denoising.editFormat(referenceWidth: first[0], referenceHeight: first[1])
                f = AppState.cappedForEditing(RecommendedFormat(e.width, e.height))
            }
            need = card.memoryNeed(width: f.width, height: f.height, references: q.images)
        }
        return RemoteQuoteAnswer(quote: q, width: f.width, height: f.height, need: need)
    }

    /// An edit's image as this Mac reads it: one of its history, or the bytes sent — a file's decoded as
    /// the file (`ImageRGB(data:)`), an image's PNG to the same `CGImage`.
    private func reference(_ r: RemoteReference, images: [UInt32: Data]) throws -> ReferenceImage {
        switch r.origin {
        case .history(let o):
            guard let e = app.history.first(where: { $0.ordinal == o }), let ref = app.fromHistory(e) else {
                throw Refused(text: String(localized: "Image i\(String(o)) is no longer in the history."))
            }
            return ref
        case .ofImage(let o, let i):
            guard let e = app.history.first(where: { $0.ordinal == o }), e.references.indices.contains(i) else {
                throw Refused(text: String(localized: "Image i\(String(o)) is no longer in the history."))
            }
            return e.references[i]
        case .file(let b):
            guard let d = images[b], let size = ReferenceImage.size(of: d), let vignette = NSImage(data: d) else {
                throw EngineError.imageUnreadable(file: r.name)
            }
            return ReferenceImage(source: .bytes(d), vignette: vignette, name: r.name, width: size.width, height: size.height)
        case .image(let b):
            guard let d = images[b], let cg = AppState.decode(d) else { throw EngineError.imageUnreadable(file: r.name) }
            return ReferenceImage(source: .image(cg), vignette: NSImage(cgImage: cg, size: .zero), name: r.name,
                                  width: cg.width, height: cg.height)
        }
    }

    /// **A request becomes a job here**, as if typed on this Mac: what the engine would refuse later
    /// is refused now, with its sentence.
    private func admit(_ r: RemoteJob, images: [UInt32: Data]) throws -> Job {
        guard app.readySet.contains(r.model), let card = app.cards.first(where: { $0.id == r.model }) else {
            throw EngineError.modelNotInstalled(model: r.model)
        }
        let model = try Model.named(r.model, in: app.library)
        let denoising = model.chain.denoising
        try denoising.check(steps: r.steps, startImage: false)
        try Format.check(width: r.width, height: r.height, family: card.family)
        let references = try r.references.map { try reference($0, images: images) }
        if !references.isEmpty {
            guard references.count <= denoising.maxReferences else {
                throw Refused(text: String(localized: "\(references.count) references: \(card.name) reads \(denoising.maxReferences) at most"))
            }
            // The app edits up to 1024² of area (`AppState.editingSurface`), as on this Mac's own form.
            guard r.width * r.height <= AppState.editingSurface else {
                throw Refused(text: String(localized: "the app edits up to 1024² of area, not \(String(r.width))x\(String(r.height))"))
            }
        }
        try card.checkMemory(width: r.width, height: r.height, references: references.count, ceiling: app.memoryCeiling)
        let forged = Set(app.library.loras(for: r.model).map(\.path))
        if let stray = r.loras.first(where: { !forged.contains($0.path) }) {
            throw Refused(text: String(localized: "This LoRA isn't installed on this Mac: \(RemoteControl.shortName(stray.path))"))
        }
        guard (1...8).contains(r.seeds.count) else { throw Refused(text: String(localized: "A render takes 1 to 8 images.")) }
        let f = denoising.space.factor
        let full = denoising.plan(height: r.height / f, width: r.width / f, steps: r.steps, start: 0,
                                  withLoRA: !r.loras.isEmpty || r.detail != .normal, schedule: r.schedule)
        // An edit's plan is unreduced (`AppState.expectedPlan`).
        let plan = references.isEmpty ? full : full.unreduced
        var job = MirrorProjection.job(r, ordinal: app.nextJobNumber(), plan: plan)
        job = Job(ordinal: job.ordinal,
                  settings: { var s = job.settings; s.modelName = card.name; s.references = references.count; return s }(),
                  seeds: job.seeds, references: references, previews: job.previews, defaultSteps: card.defaultSteps, plan: plan)
        job.neuralEngine = r.neuralEngine
        return job
    }

    /// **An exploration becomes cells here**, by `enqueueGrid` as on this Mac's own form: each cell's
    /// model ready, its format and memory judged, the stack forged; a sketch's stop is this Mac's to find.
    private func admit(_ g: RemoteGridRequest, images: [UInt32: Data]) throws {
        let grid = try g.spec.grid()
        let references = try g.job.references.map { try reference($0, images: images) }
        var settings = MirrorProjection.settings(g.job)
        for id in Set(grid.cells.map { $0.model ?? g.job.model }) {
            guard app.readySet.contains(id), let card = app.cards.first(where: { $0.id == id }) else {
                throw EngineError.modelNotInstalled(model: id)
            }
            try Format.check(width: g.job.width, height: g.job.height, family: card.family)
            try card.checkMemory(width: g.job.width, height: g.job.height, references: references.count, ceiling: app.memoryCeiling)
            if id == g.job.model { settings.modelName = card.name }
        }
        let forged = Set(app.library.loras(for: g.job.model).map(\.path))
        if let stray = g.job.loras.first(where: { !forged.contains($0.path) }) {
            throw Refused(text: String(localized: "This LoRA isn't installed on this Mac: \(RemoteControl.shortName(stray.path))"))
        }
        if g.sketch { app.stopExploring() }
        app.enqueueGrid(grid, settings: settings, references: references, previews: g.job.previews, sketch: g.sketch,
                        stopAfter: g.stopAfter, neuralEngine: g.job.neuralEngine)
    }

    private struct Refused: Error { let text: String }

    // MARK: - Snapshots

    private func queueSnapshot() -> RemoteQueue {
        var q = RemoteQueue()
        q.current = app.currentJob.map(MirrorProjection.remote)
        q.waiting = app.file.map(MirrorProjection.remote)
        q.awaitingLicense = app.awaitingLicense.map(MirrorProjection.remote)
        q.fraction = app.fraction
        q.stage = app.stage?.rawValue
        q.currentStep = app.currentStep; q.totalSteps = app.totalSteps
        q.batchImage = app.batchImage; q.batchSize = app.batchSize
        q.stageStates = Dictionary(uniqueKeysWithValues: app.stageStates.compactMap { k, v in
            MirrorProjection.stageStates[v].map { (k.rawValue, $0) }
        })
        q.stageSeconds = Dictionary(uniqueKeysWithValues: app.stageSeconds.map { ($0.key.rawValue, $0.value) })
        q.remaining = app.remaining(at: Date())
        q.problem = app.problem?.text
        q.enlarging = app.enlarging.flatMap { id in app.history.first { $0.id == id }?.ordinal }
        return q
    }

    private func explorationSnapshot() -> RemoteExploration? {
        guard let r = app.explored else { return nil }
        return RemoteExploration(
            ordinal: r.ordinal, spec: RemoteGridSpec(r.grid),
            job: MirrorProjection.remote(r.settings, ordinal: 0, seeds: [], previews: true, defaultSteps: nil, neuralEngine: nil),
            cells: r.cellJobs.map(MirrorProjection.remote),
            issues: r.issues.mapValues { i in
                switch i {
                case .finished: .finished
                case .stopped: .stopped
                case .removed: .removed
                case .failed(let p): .failed(p.text)
                }
            },
            sketch: r.sketch)
    }

    private func catalogSnapshot() -> MirrorCatalog {
        var traits: [String: RemoteTraits] = [:], loras: [String: [RemoteLoRA]] = [:], names: [String: String] = [:]
        var needs: [String: [String: Int]] = [:]
        for id in app.readySet {
            if let m = app.model(id) {
                let d = m.chain.denoising
                traits[id] = RemoteTraits(maxReferences: d.maxReferences, readsImages: m.chain.text.output.readsImages,
                                          acceptsDetail: d.acceptsDetail, comfyUI: d.schedules.contains(.comfyUI),
                                          steps: (1...50).filter { (try? d.check(steps: $0, startImage: false)) != nil },
                                          text: m.chain.text.name, encoding: m.chain.encoding?.name,
                                          decoding: m.chain.decoding.name)
            }
            if let card = app.cards.first(where: { $0.id == id }) {
                for f in card.formats {
                    needs[id, default: [:]][MirrorCatalog.key(f.width, f.height)] = card.memoryNeed(width: f.width, height: f.height)
                }
            }
            let forged = app.library.loras(for: id)
            loras[id] = forged.map { RemoteLoRA(path: $0.path, strength: 0) }
            for l in forged { names[l.path] = l.rawName }
        }
        return MirrorCatalog(ready: app.readySet.sorted(), traits: traits, loras: loras, loraNames: names, needs: needs,
                             ceiling: app.memoryCeiling, settingsRead: app.settingsRead,
                             neuralEngineOffered: app.neuralEngineOffered, fasterRendering: app.fasterRendering,
                             enlarges: SuperResolution.available)
    }

    private func send(_ e: Entry, to identity: Data? = nil) {
        let id = blob()
        host.send(.entry(remote(e, png: id)), to: identity)
        host.send(blob: e.png, id: id, to: identity)
    }

    /// An exploration's cell image: `at` is [grid, cell].
    private func send(_ e: Entry, cell at: [Int], to identity: Data? = nil) {
        let id = blob()
        host.send(.cellImage(grid: at[0], cell: at[1], remote(e, png: id)), to: identity)
        host.send(blob: e.png, id: id, to: identity)
    }

    private func remote(_ e: Entry, png id: UInt32) -> RemoteEntry {
        var job = MirrorProjection.remote(e.settings, ordinal: e.job, seeds: [e.seed], previews: false, defaultSteps: nil,
                                          neuralEngine: e.settings.neuralEngine)
        job.references = e.references.enumerated().map { i, ref in
            RemoteReference(origin: .ofImage(e.ordinal, index: i), name: ref.name, width: ref.width, height: ref.height)
        }
        let r = RemoteEntry(ordinal: e.ordinal, job: job,
                            width: e.width, height: e.height, seed: e.seed, timings: e.timings, footprints: e.footprints,
                            evaluations: e.evaluations, metadata: e.metadata, neuralEngine: e.settings.neuralEngine, png: id,
                            enlarged: e.enlarged, cropped: e.cropped,
                            sheet: e.grid.map { RemoteSheet(ordinal: $0.ordinal, spec: RemoteGridSpec($0.grid), empty: $0.empty) })
        return r
    }

    private func blob() -> UInt32 {
        defer { nextBlob &+= 1; if nextBlob == 0 { nextBlob = 1 } }
        return nextBlob
    }

    // MARK: - Watching

    private func watchQueue() {
        withObservationTracking {
            _ = queueSnapshot()
            _ = app.history.map(\.ordinal)
            _ = app.preview
            _ = explorationSnapshot()
            _ = app.explored?.entries.keys
        } onChange: { [weak self] in
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.scheduleQueue() } }
        }
    }

    private func scheduleQueue() {
        guard !queueScheduled else { return }
        queueScheduled = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { [weak self] in
            MainActor.assumeIsolated { self?.fireQueue() }
        }
    }

    private func fireQueue() {
        queueScheduled = false
        publishQueue()
        watchQueue()
    }

    private func publishQueue() {
        let q = queueSnapshot()
        if q != lastQueue {
            lastQueue = q
            if !clients.isEmpty { host.send(.queue(q)) }
        }
        let ordinals = Set(app.history.map(\.ordinal))
        let gone = sentEntries.subtracting(ordinals)
        if !gone.isEmpty, !clients.isEmpty { host.send(.removed(gone.sorted())) }
        for e in app.history.reversed() where !sentEntries.contains(e.ordinal) && !clients.isEmpty { send(e) }
        sentEntries = ordinals
        let x = explorationSnapshot()
        if x != lastExploration {
            if x?.ordinal != lastExploration?.ordinal { sentCells = [] }
            lastExploration = x
            if !clients.isEmpty { host.send(.explored(x)) }
        }
        if let r = app.explored, !clients.isEmpty {
            for (cell, e) in r.entries where !sentCells.contains([r.ordinal, cell]) {
                sentCells.insert([r.ordinal, cell])
                send(e, cell: [r.ordinal, cell])
            }
        }
        if app.preview !== lastPreview {
            lastPreview = app.preview
            guard !clients.isEmpty else { return }
            if let p = app.preview, let png = Self.png(p) {
                let id = blob()
                host.send(.preview(id))
                host.send(blob: png, id: id)
            } else {
                host.send(.preview(nil))
            }
        }
    }

    private func watchCatalog() {
        withObservationTracking {
            _ = app.readySet; _ = app.cards; _ = app.settingsRead; _ = app.fasterRendering
        } onChange: { [weak self] in
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.fireCatalog() } }
        }
    }

    private func fireCatalog() {
        let c = catalogSnapshot()
        if c != lastCatalog {
            lastCatalog = c
            if !clients.isEmpty { host.send(.catalog(c)) }
        }
        watchCatalog()
    }

    static func png(_ image: CGImage) -> Data? {
        let d = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(d, "public.png" as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, nil)
        return CGImageDestinationFinalize(dest) ? d as Data : nil
    }
}
