// `AppState` as a mirror: what the other Mac publishes, received — its catalog, its queue, its images, its blobs.

import AppKit
import Foundation
import ImageIO
import Siliconed

extension AppState {
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
