// `MirrorServer`: a client's request, judged as `silicontrol add` would — held, refused, released, admitted.

import AppKit
import Foundation
import ImageIO
import Siliconed

extension MirrorServer {
    /// A client that waits for more renders than the queue holds is asking for nothing the queue could give.
    func hold(_ asked: Asked, from identity: Data) {
        guard held.filter({ $0.identity == identity }).count < AppState.maxQueueLength else {
            refuse(.app(.queueFull), asked, to: identity)
            return
        }
        held.append((identity, asked))
        release()
    }

    /// A render is answered with its request number; a series' refusal is said.
    func refuse(_ text: RemoteRefusal, _ asked: Asked, to identity: Data) {
        if let n = asked.request { host.send(.answer(request: n, .refused(text)), to: identity) }
        else { host.send(.refused(text), to: identity) }
    }

    /// The renders whose images are all here, in order: one still waiting holds the same Mac's after it — never
    /// another Mac's, whose queue would otherwise hang on an image that does not come.
    func release() {
        var waiting = Set<Data>()
        var i = 0
        while i < held.count {
            let (identity, asked) = held[i]
            let sent = uploads[identity] ?? [:]
            guard !waiting.contains(identity), asked.references.allSatisfy({ $0.blob.map { sent[$0] != nil } ?? true }) else {
                waiting.insert(identity)
                i += 1
                continue
            }
            held.remove(at: i)
            do {
                switch asked {
                case .render(let r):
                    let job = try admit(r, images: sent, from: identity)
                    app.enqueue([job])
                    host.send(.answer(request: r.ordinal, .queued(ordinal: job.ordinal, estimate: app.estimation(job),
                                                                  width: job.settings.format.width,
                                                                  height: job.settings.format.height)), to: identity)
                case .grid(let g):
                    let cells = try admit(g, images: sent, from: identity)
                    if let n = g.request {
                        host.send(.answer(request: n, .gridQueued(grid: cells.run.ordinal, cells: cells.jobs.map(MirrorProjection.remote),
                                                                  estimate: app.estimation(of: cells.jobs))),
                                  to: identity)
                    }
                }
            } catch let e as Refused {
                refuse(.app(e.refusal), asked, to: identity)
            } catch {
                refuse(.engine(EngineError(error)), asked, to: identity)
            }
            for b in asked.references.compactMap(\.blob) { uploads[identity]?[b] = nil }
        }
    }

    /// **What the borrowing Mac's form cannot compute**: an edit's format (the denoiser's rule, capped as
    /// the app caps it), from the models this Mac holds, as its own form would.
    func answer(_ q: RemoteQuote) -> RemoteQuoteAnswer {
        // Numbers from another Mac reach arithmetic that traps on overflow: out of bounds, no format of ours.
        guard q.isSane else { return RemoteQuoteAnswer(quote: q, width: q.width, height: q.height, need: nil) }
        var f = RecommendedFormat(q.width, q.height)
        if app.readySet.contains(q.model), let first = q.first, first.count == 2, let m = app.model(q.model) {
            let e = m.chain.denoising.editFormat(referenceWidth: first[0], referenceHeight: first[1])
            f = AppState.cappedForEditing(RecommendedFormat(e.width, e.height))
        }
        return RemoteQuoteAnswer(quote: q, width: f.width, height: f.height, need: nil)
    }

    /// An edit's image as this Mac reads it: one of its history, or the bytes sent — a file's decoded as
    /// the file (`ImageRGB(data:)`), an image's PNG to the same `CGImage`.
    private func reference(_ r: RemoteReference, images: [UInt32: Data], from identity: Data) throws -> ReferenceImage {
        switch r.origin {
        case .history(let o):
            guard let e = own(o, identity), let ref = app.fromHistory(e) else {
                throw Refused(.imageGone(o))
            }
            return ref
        case .ofImage(let o, let i):
            guard let e = own(o, identity), e.references.indices.contains(i) else {
                throw Refused(.imageGone(o))
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
    private func admit(_ asked: RemoteJob, images: [UInt32: Data], from identity: Data) throws -> Job {
        var r = asked
        guard app.readySet.contains(r.model), let card = app.cards.first(where: { $0.id == r.model }) else {
            throw EngineError.modelNotInstalled(model: r.model)
        }
        let model = try Model.named(r.model, in: app.library)
        let denoising = model.chain.denoising
        try denoising.check(steps: r.steps, startImage: false)
        // What the borrower's form offers for this model, from its traits: its word is not the last.
        if r.solver != .euler, !AppState.solverFamilies.contains(card.family) {
            throw Refused(.solverNotOffered(model: card.id, solver: r.solver.rawValue))
        }
        if r.detail != .normal, !denoising.acceptsDetail { throw Refused(.detailNotOffered(model: card.id)) }
        // What costs nothing to judge, before anything is decoded.
        guard app.file.count < AppState.maxQueueLength else { throw Refused(.queueFull) }
        guard r.references.count <= denoising.maxReferences else {
            throw Refused(.tooManyReferences(count: r.references.count, model: card.id, max: denoising.maxReferences))
        }
        guard (1...8).contains(r.seeds.count) else { throw Refused(.seedCount) }
        let references = try r.references.map { try reference($0, images: images, from: identity) }
        // A job that cites named subjects (which stayed on the borrower): the rewrite the model reads, and their
        // pictures last among the references — rebuilt as if the subjects were here.
        let subjects = r.subjectImages ?? 0
        if subjects < 0 || subjects > references.count || (r.modelPrompt != nil) != (subjects > 0)
            || r.modelPrompt?.allSatisfy(\.isWhitespace) == true {
            throw Refused(.subjectsIncomplete)
        }
        // No format with an edit (`silicontrol add --ref`): image 1's, as this Mac's form derives it.
        if r.width == 0, r.height == 0, references.count > subjects, let first = references.first {
            let f = denoising.editFormat(referenceWidth: first.width, referenceHeight: first.height)
            let capped = AppState.cappedForEditing(RecommendedFormat(f.width, f.height))
            r.width = capped.width; r.height = capped.height
        }
        try Format.check(width: r.width, height: r.height, family: card.family)
        if references.count > subjects {
            // The app edits up to 1024² of area (`AppState.editingSurface`), as on this Mac's own form; subjects alone
            // make a new image at the size chosen.
            guard r.width * r.height <= AppState.editingSurface else {
                throw Refused(.editSurface(width: r.width, height: r.height))
            }
        }
        let installed = Set(app.library.loras(for: r.model).map(\.path))
        if let stray = r.loras.first(where: { !installed.contains($0.path) }) {
            throw Refused(.loraNotInstalled(name: RemoteControl.shortName(stray.path)))
        }
        let f = denoising.space.factor
        let plan = denoising.plan(height: r.height / f, width: r.width / f, steps: r.steps, start: 0)
        var job = MirrorProjection.job(r, ordinal: app.nextJobNumber(), plan: plan)
        job = Job(ordinal: job.ordinal,
                  settings: { var s = job.settings; s.modelName = card.name; s.references = references.count; return s }(),
                  seeds: job.seeds, references: references, previews: job.previews, defaultSteps: card.defaultSteps, plan: plan)
        job.owner = identity
        return job
    }

    /// **A series becomes cells here**, by `enqueueGrid` as on this Mac's own form: each cell's model ready, its format
    /// and memory judged, the stack's LoRAs installed. The window's series (no request number) is the one published to that Mac.
    private func admit(_ g: RemoteGridRequest, images: [UInt32: Data], from identity: Data) throws -> (run: GridRun, jobs: [Job]) {
        let grid = try g.spec.grid()
        // Before anything is decoded: the cells the queue has room for, and no more images than any model reads.
        guard app.file.count + grid.cells.count <= AppState.maxQueueLength else { throw Refused(.queueFull) }
        guard g.job.references.count <= Self.gridReferenceBound else {
            throw Refused(.tooManyReferences(count: g.job.references.count, model: g.job.model, max: Self.gridReferenceBound))
        }
        let references = try g.job.references.map { try reference($0, images: images, from: identity) }
        var settings = MirrorProjection.settings(g.job)
        for id in Set(grid.cells.map { $0.model ?? g.job.model }) {
            guard app.readySet.contains(id), let card = app.cards.first(where: { $0.id == id }) else {
                throw EngineError.modelNotInstalled(model: id)
            }
            try Format.check(width: g.job.width, height: g.job.height, family: card.family)
            // Each cell's step count against its own model's schedules, as `silicontrol grid` judges it.
            let denoising = try Model.named(id, in: app.library).chain.denoising
            if id == g.job.model {
                settings.modelName = card.name
                if g.job.detail != .normal, !denoising.acceptsDetail { throw Refused(.detailNotOffered(model: card.id)) }
                // A solver swept with no model axis is the base model's to walk (`silicontrol grid`'s rule).
                if !grid.cells.contains(where: { $0.model != nil }), !AppState.solverFamilies.contains(card.family),
                   let solver = grid.cells.lazy.map({ $0.solver ?? g.job.solver }).first(where: { $0 != .euler }) {
                    throw Refused(.solverNotOffered(model: card.id, solver: solver.rawValue))
                }
            }
            for cell in grid.cells where (cell.model ?? g.job.model) == id {
                try denoising.check(steps: cell.steps ?? card.defaultSteps, startImage: false)
            }
        }
        let installed = Set(app.library.loras(for: g.job.model).map(\.path))
        if let stray = g.job.loras.first(where: { !installed.contains($0.path) }) {
            throw Refused(.loraNotInstalled(name: RemoteControl.shortName(stray.path)))
        }
        // Not opened in this Mac's window: it is that Mac's.
        return app.enqueueGrid(grid, settings: settings, references: references, previews: g.job.previews,
                               named: false, owner: identity, shows: false)
    }

    /// No model reads more than 3 images (`maxReferences`); each cell is judged against its own model's later.
    private static let gridReferenceBound = 8

    private struct Refused: Error {
        let refusal: AppRefusal
        init(_ refusal: AppRefusal) { self.refusal = refusal }
    }
}
