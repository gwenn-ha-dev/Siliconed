// **`silicontrol` follows the switch** (Render › Compute On): it talks only to this Mac's socket, never
// to the network, but once the window shows another Mac, the commands act on what it shows — the
// other Mac's queue and history (`AppState+Mirror.swift`), as the buttons do. Split from `RemoteControl.swift`.
//
// - `add` is asked of the other Mac, which numbers the jobs and answers each (`RemoteAnswer`): the
//   answer comes before the verdict, so a refusal is the command's own. An edit with no format typed
//   leaves its size to the other Mac (image 1's, by its denoiser's rule).
// - `wait` and `follow` read the signals the mirror gives from what the other Mac publishes.
// - What acts on this Mac — `install` — refuses; `set beta` writes
//   this Mac's preferences and the mirror's copy of them; the window says where it computes.
// - `grid` is asked whole of the other Mac, which
//   makes the cells (`MirrorLink.explore`) and answers with them (`gridQueued`); `wait g2` follows it
//   from what it publishes. `vary` is the window's « Variations » (the same jobs, each answered by the
//   other Mac); `enlarge` asks the other Mac, which enlarges its own pixels and says if it needs Apple's
//   model (`--download` is the consent, as here). A cell's, a variation's or an enlargement's answer has
//   the fields it has here, plus `computing_on`.
// - `add` and `grid` fill the rack as a click would, and the canvas shows the series once published, as on this Mac.
// - **Named subjects stay on this Mac** (`subjects` acts on them, mirrored or not; no refusal): a job
//   that cites them goes with its rewrite and their framed pictures (`namedJob`, `RemoteJob.modelPrompt`).

import AppKit
import Foundation
import Siliconed

extension RemoteControl {
    /// The commands a mirror does not take, with the sentence that says why.
    func refuseOnMirror(_ command: String, _ arguments: [String] = []) throws {
        guard let mirror = app.mirror else { return }
        let name = mirror.macName
        switch command {
        // « Stop » works on the lender's installation as on this Mac's, with no click on the other.
        case "install" where arguments == ["stop"]:
            break
        // Files go up to the other Mac, whose owner confirms there.
        case "start-image":
            throw Refusal("remote", String(localized: "a starting image is not offered while the window computes on “\(name)”"),
                          help: String(localized: "Render › Compute On › This Mac, then retry"))
        case "install", "import":
            throw Refusal("remote", String(localized: "\(command) acts on this Mac, and the window computes on “\(name)”"),
                          help: String(localized: "Render › Compute On › This Mac, then retry"))
        default:
            break
        }
    }

    /// **`models` on a mirror**: the other Mac's ready models, as it published them (`MirrorCatalog`).
    func mirroredModels() -> [String: Any] {
        let catalog = app.mirrorCatalog
        let list = app.visibleCards.map { f -> [String: Any] in
            var o: [String: Any] = ["id": f.id, "name": f.name, "default_steps": f.defaultSteps,
                                    "formats": f.formats.map { "\($0.width)x\($0.height)" },
                                    "commercial_license": f.license.commercial, "license": f.license.text,
                                    "default": f.id == ModelCard.of(.default).id, "ready": app.readySet.contains(f.id)]
            if let t = catalog?.traits[f.id] {
                o["edit"] = t.maxReferences > 0
                o["max_references"] = t.maxReferences
                o["loras"] = (catalog?.loras[f.id] ?? []).map { l in
                    ["lora": Self.shortName(l.path), "name": catalog?.loraNames[l.path] ?? Self.shortName(l.path)]
                }
            }
            return o
        }
        return ["ok": true, "models": list, "computing_on": app.mirror?.macName ?? NSNull()]
    }

    /// **`add` on a mirror**: the request read as `add` reads it, judged with what the other Mac
    /// published, then asked of it — one job per alternative of the prompt, at the same seeds.
    func addOnMirror(_ mirror: MirrorLink, _ arguments: [String], cwd: String, _ c: Connection, grid: Bool = false) throws {
        let d = try parse(arguments)
        // A request while the session is down would be dropped, and its command would wait for an answer
        // that cannot come.
        if let down = app.linkBlocker { throw Refusal("link", down) }
        // The starting image does not travel to the other Mac yet: said, never dropped in silence.
        if d.image != nil || d.strength != nil {
            throw Refusal("remote", String(localized: "a starting image (--image) is not offered while the window computes on “\(mirror.macName)”"),
                          help: String(localized: "Render › Compute On › This Mac, then retry"))
        }
        // The rules of `add` here, with what the other Mac published of its models — which judges again.
        try checkUsage(d, grid: grid)
        guard app.readySet.contains(d.model), let card = app.cards.first(where: { $0.id == d.model }) else {
            let ready = app.visibleReady.map(\.id).joined(separator: ", ")
            throw Refusal("model", String(localized: "model \"\(d.model)\" is not ready on “\(mirror.macName)”"),
                          help: String(localized: "ready: \(ready) — silicontrol models"))
        }
        let traits = app.mirrorCatalog?.traits[d.model]
        try checkSolver(d.solver, family: card.family, name: card.name)
        if let traits { try checkDetail(d.detail, accepted: traits.acceptsDetail, name: card.name) }
        guard let prompt = d.prompt, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Refusal("usage", String(localized: "empty prompt"))
        }
        let alternatives: PromptAlternatives
        do { alternatives = try PromptAlternatives(prompt) } catch {
            let p = Problem(error)
            throw Refusal("prompt", p.title, help: p.suggestion)
        }

        // The references: files here, or images of the other Mac's history (`i7`), sent by their ordinal.
        var references: [ReferenceImage] = []
        if !d.references.isEmpty {
            let max = traits?.maxReferences ?? 0
            guard max > 0 else {
                throw Refusal("edit", String(localized: "\(card.name) does no editing by reference (--ref)"),
                              help: String(localized: "models that do have \"edit\": true in \"silicontrol models\""))
            }
            guard d.references.count <= max else {
                let n = d.references.count
                throw Refusal("edit", String(localized: "\(n) references: \(card.name) reads \(max) at most"),
                              help: String(localized: "\"max_references\" in \"silicontrol models\""))
            }
            references = try d.references.map { try referenceImage($0, cwd: cwd) }
        }
        // An edit with no format typed: image 1's, derived on the other Mac.
        let typed = d.format.map { RecommendedFormat($0.0, $0.1) }
        let format = typed ?? RecommendedFormat(1024, 1024)
        if typed != nil || references.isEmpty {
            do { try Format.check(width: format.width, height: format.height, family: card.family) } catch {
                let p = Problem(error)
                throw Refusal("format", p.title, help: p.suggestion)
            }
        }
        let steps = d.steps ?? card.defaultSteps
        if let t = traits { try checkSteps(steps, scheduled: t.steps, defaultSteps: card.defaultSteps) }
        try checkLoRAs(d)
        let available = app.mirrorCatalog?.loras[d.model] ?? []
        let stack = try d.loras.map { entry -> LoRASlot in
            guard let l = availableLoRA(entry.name, among: available) else {
                let name = entry.name, list = available.map { Self.shortName($0.path) }.joined(separator: ", ")
                throw Refusal("lora", String(localized: "LoRA \"\(name)\" not found for \(d.model)"),
                              help: available.isEmpty ? String(localized: "no LoRA installed for this model")
                                                   : String(localized: "compatible: \(list)"))
            }
            return LoRASlot(path: l.path, strength: entry.strength)
        }
        var folder: String?
        if let raw = d.folder {
            folder = absolute(raw, cwd)
            try createFolder(folder!)
        }

        let firstNumber = d.seed ?? Request.drawnSeed()
        let variants = alternatives.variants
        // Named subjects (they stay here): each expanded prompt rewritten — refused whole if one is —, and
        // a model that reads no images has none. A grid cites none, as here.
        let capacity = (traits?.readsImages ?? false) ? traits?.maxReferences ?? 0 : 0
        for v in variants {
            if case .failure(let f)? = app.rewrite(v.prompt, capacity: capacity, editImages: references.count) {
                if grid { break }
                let p = Problem(f)
                throw Refusal("subject", p.title, help: p.suggestion ?? "silicontrol subjects")
            }
        }
        if grid {
            if variants.contains(where: { app.rewrite($0.prompt, capacity: capacity) != nil }) {
                throw Refusal("subject", String(localized: "a grid does not cite named subjects yet"), help: "silicontrol add")
            }
            try addGridOnMirror(mirror, d, prompt: prompt, card: card, format: format, steps: steps, stack: stack,
                                references: references, seed: firstNumber, folder: folder, c)
            return
        }
        let jobs = variants.map { v -> Job in
            var settings = RenderSettings(identifier: card.id, modelName: card.name, prompt: v.prompt, format: format,
                                          steps: steps, loras: stack, references: references.count, detail: d.detail,
                                          solver: d.solver)
            let used = app.namedJob(v.prompt, settings: &settings, references: references, capacity: capacity)
            return Job(ordinal: 0, settings: settings, seeds: (0..<d.batch).map { firstNumber &+ UInt64($0) },
                          references: used, previews: d.previews ?? app.previews, defaultSteps: card.defaultSteps, plan: nil)
        }

        // Each job is answered by the other Mac; the command answers once all are.
        var answers: [RemoteAnswer?] = Array(repeating: nil, count: jobs.count)
        func answered() {
            guard answers.allSatisfy({ $0 != nil }) else { return }
            let refused = answers.compactMap { a -> String? in if case .refused(let r)? = a { Problem(r).text } else { nil } }
            if let first = refused.first {
                finish(c, Response.failure("refused", first))
                return
            }
            let added = zip(answers, variants).map { a, v -> [String: Any] in
                guard case let .queued(o, estimate, _, _)? = a else { return [:] }
                var j = mirror.jobs[o].map { json($0) } ?? ["job": "t\(o)"]
                j["estimate_s"] = estimate.map { Response.number($0, 1) } ?? NSNull()
                if alternatives.hasAlternatives { j["choices"] = v.choices }
                j["computing_on"] = mirror.macName
                return j
            }
            let ordinals = answers.compactMap { a -> Int? in if case let .queued(o, _, _, _)? = a { o } else { nil } }
            if d.wait {
                for var o in added { o["type"] = "added"; emit(c, o) }
                followToEnd(ordinals, folder: folder, c)
            } else if added.count == 1 {
                var o = added[0]
                o["ok"] = true
                finish(c, o)
            } else {
                finish(c, ["ok": true, "jobs": added, "computing_on": mirror.macName])
            }
        }
        // The canvas follows the render, as after « Generate ».
        app.followsRender = true
        for (i, job) in jobs.enumerated() {
            do {
                // An edit's own images decide the format; the subjects' pictures never do.
                try mirror.render(job, format: typed != nil || references.isEmpty) { a in
                    answers[i] = a
                    answered()
                }
            } catch {
                throw Refusal("ref", Problem(error).text)
            }
        }
        // The form takes the request, as after a click, as `add` on this Mac does.
        app.takeRequest(jobs[0].settings, prompt: prompt, references: references, start: nil, strength: nil,
                        seed: firstNumber, fixed: d.seed != nil)
    }

    /// A LoRA of the other Mac, by the name `--lora` was given: its short name, its path, or its own name.
    private func availableLoRA(_ name: String, among available: [RemoteLoRA]) -> RemoteLoRA? {
        let names = app.mirrorCatalog?.loraNames ?? [:]
        return available.first { Self.shortName($0.path) == name || $0.path == name || names[$0.path] == name }
    }

    // MARK: - grid, vary, enlarge on a mirror

    /// **`grid` on a mirror**: the request `add` checked, its axes read and the grid checked here by the
    /// library (`RenderGrid`, which the other Mac checks again), every cell's step count against the
    /// steps its model published — then the cells are made there, and the command answers with them.
    private func addGridOnMirror(_ mirror: MirrorLink, _ d: RenderOrder, prompt: String, card: ModelCard, format: RecommendedFormat,
                                 steps: Int, stack: [LoRASlot], references: [ReferenceImage], seed: UInt64,
                                 folder: String?, _ c: Connection) throws {
        let available = app.mirrorCatalog?.loras[d.model] ?? []
        let g = try readGrid(d, prompt: prompt, seed: seed, steps: steps, stack: stack, images: references.count, start: false) {
            availableLoRA($0, among: available)?.path
        }
        try checkSolverAxis(g, family: card.family, name: card.name)
        for cell in g.cells {
            guard let n = cell.steps, let t = app.mirrorCatalog?.traits[cell.model ?? d.model] else { continue }
            try checkSteps(n, scheduled: t.steps, defaultSteps: nil)
        }
        guard app.file.count + g.count <= AppState.maxQueueLength else {
            let n = g.count, max = AppState.maxQueueLength, room = max - app.file.count
            throw Refusal("queue_full", String(localized: "\(n) jobs for the grid: the queue has room for \(room) (\(max) jobs)"),
                          help: "silicontrol wait, then retry")
        }
        let settings = RenderSettings(identifier: card.id, modelName: card.name, prompt: prompt, format: format, steps: steps,
                                      loras: stack, references: references.count, detail: d.detail, solver: d.solver)
        let on = mirror.macName
        do {
            try mirror.explore(g, settings: settings, references: references,
                               previews: d.previews ?? app.previews) { [self] answer in
                switch answer {
                case .refused(let text):
                    finish(c, Response.failure("refused", Problem(text).text))
                case let .gridQueued(grid, cells, estimate):
                    let added = cells.enumerated().map { i, r -> [String: Any] in
                        var o = json(mirror.jobs[r.ordinal] ?? MirrorProjection.job(r, ordinal: r.ordinal))
                        o["grid"] = "g\(grid)"; o["column"] = i % g.columns; o["row"] = i / g.columns
                        o["computing_on"] = on
                        return o
                    }
                    if d.wait {
                        for var o in added { o["type"] = "added"; emit(c, o) }
                        followMirroredGrid(mirror, grid, folder: folder, sheet: d.sheet, c)
                    } else {
                        finish(c, ["ok": true, "grid": "g\(grid)", "columns": g.columns, "rows": g.rows, "jobs": added,
                                   "estimate_s": estimate.map { Response.number($0, 1) } ?? NSNull(), "computing_on": on])
                    }
                default:
                    finish(c, Response.failure("internal", "unexpected answer"))
                }
            }
        } catch {
            throw Refusal("ref", Problem(error).text)
        }
        // The form takes the grid, as after a click, and the canvas shows it once the other Mac publishes it
        // (`MirrorLink.explore`), as `grid` on this Mac does.
        app.takeRequest(settings, prompt: prompt, references: references, start: nil, strength: nil, seed: seed,
                        fixed: d.seed != nil)
        _ = app.setSeries(from: g)
        app.followsRender = false
    }

    /// **`wait g2` and `grid --wait` on a mirror**: each cell's image as it comes (written into `folder` if given), then —
    /// `sheet`, into the same folder — the sheet, drawn here from the cells' images, then the verdict.
    func followMirroredGrid(_ mirror: MirrorLink, _ g: Int, folder: String?, sheet: Bool = true, _ c: Connection) {
        // The state followed, once: the window may switch Mac meanwhile, and the listener must leave the one it joined.
        let app = self.app
        guard let jobs = mirror.grids[g] else {
            finish(c, Response.failure("unknown_job", String(localized: "no grid \("g\(g)") in this session"), help: "silicontrol status"))
            return
        }
        var tallies: [Int: [String: Any]] = [:]
        var told = Set<Int>()
        var concluded = false
        var token: UUID?
        func conclude() {
            guard !concluded, tallies.count == jobs.count else { return }
            concluded = true
            if let token { app.unsubscribe(token) }
            let allFinished = jobs.allSatisfy { tallies[$0]?["outcome"] as? String == "done" }
            var o: [String: Any] = ["ok": allFinished, "grid": "g\(g)", "jobs": jobs.compactMap { tallies[$0] },
                                    "computing_on": mirror.macName]
            if !allFinished {
                o["error"] = "render"
                o["message"] = String(localized: "at least one cell did not render: see \"outcome\" in \"jobs\"; the sheet marks it empty")
            }
            guard sheet, let folder, let run = app.series[g] else { finish(c, o); return }
            Task {
                do {
                    let path = try await app.writeSheet(run, to: folder)
                    if let path { emit(c, ["type": "sheet", "grid": "g\(g)", "path": path]) }
                    o["sheet"] = path ?? NSNull()
                } catch {
                    o["sheet"] = NSNull()
                    o["write_error"] = "\(error)"
                }
                finish(c, o)
            }
        }
        func image(_ job: Int, _ e: Entry) {
            if jobs.contains(job), told.insert(job).inserted { emitImage(e, folder: folder, c) }
        }
        // What already came (a grid waited for later).
        for n in jobs {
            guard let t = app.completed[n] else { continue }
            tallies[n] = Self.json(t.issue, job: n, images: t.images, warnings: t.warnings)
            if let e = t.images.compactMap({ k in app.history.first { $0.ordinal == k } }).first { image(n, e) }
        }
        conclude()
        guard !concluded else { return }
        token = app.subscribe { [weak self] signal in
            guard self != nil else { return }
            if c.broken { if let token { app.unsubscribe(token) }; return }
            switch signal {
            case let .image(t, e): image(t.ordinal, e)
            case let .finished(t, issue) where jobs.contains(t.ordinal):
                tallies[t.ordinal] = Self.json(issue, job: t.ordinal, images: app.completed[t.ordinal]?.images ?? [],
                                               warnings: app.completed[t.ordinal]?.warnings ?? [])
            default:
                break
            }
            conclude()
        }
    }

    /// **`vary` on a mirror**: « Variations » as the window does it — jobs with the image's settings and a
    /// variation seed each, an edit's images named by the image (`ofImage`) —, each answered by the other Mac.
    func varyOnMirror(_ mirror: MirrorLink, _ arguments: [String], _ c: Connection) throws {
        let (e, amount, count) = try varyRequest(arguments)
        guard app.canVary(e) else {
            throw Refusal("model", String(localized: "\(e.name) has no variations: an enlargement or a crop, or its model is not ready"),
                          help: "silicontrol models")
        }
        guard app.file.count + count <= AppState.maxQueueLength else {
            throw Refusal("queue_full", String(localized: "the queue is full (\(AppState.maxQueueLength) jobs)"),
                          help: "silicontrol wait, then retry")
        }
        let first = mirror.nextRequest
        let jobs = app.vary(e, amount, count: count)
        guard mirror.nextRequest - first == jobs.count else {
            throw Refusal("ref", app.problem?.text ?? String(localized: "This image could not be varied"))
        }
        answerQueued(mirror, from: first, count: jobs.count, c)
    }

    /// **The word of the other Mac on `count` jobs a gesture just sent** (from request `first`): the command answers once all are in, with
    /// the jobs as the other Mac numbered them, or the first refusal.
    func answerQueued(_ mirror: MirrorLink, from first: Int, count: Int, _ c: Connection) {
        var answers: [RemoteAnswer?] = Array(repeating: nil, count: count)
        mirror.awaiting(from: first) { [self] i, a in
            answers[i] = a
            guard answers.allSatisfy({ $0 != nil }) else { return }
            if case .refused(let text)? = answers.first(where: { if case .refused? = $0 { true } else { false } }) {
                finish(c, Response.failure("refused", Problem(text).text))
                return
            }
            let added = answers.map { a -> [String: Any] in
                guard case let .queued(o, _, _, _)? = a else { return [:] }
                var j = mirror.jobs[o].map { json($0) } ?? ["job": "t\(o)"]
                j["computing_on"] = mirror.macName
                return j
            }
            finish(c, ["ok": true, "jobs": added, "computing_on": mirror.macName])
        }
    }

    /// **`enlarge` on a mirror**: the other Mac enlarges its own pixels. Its answer is the new image, or
    /// the need for Apple's model — which `--download` consents to, as here —, or a refusal.
    func enlargeOnMirror(_ mirror: MirrorLink, _ arguments: [String], _ c: Connection) throws {
        guard (1...2).contains(arguments.count), arguments.count == 1 || arguments[1] == "--download" else {
            throw Refusal("usage", "enlarge <i7> [--download]")
        }
        guard let e = image(arguments[0]) else {
            throw Refusal("unknown_image", String(localized: "no image \"\(arguments[0])\" in the history"), help: "silicontrol history")
        }
        if let refusal = app.enlargeRefusal(e) {
            finish(c, Response.failure("refused", refusal))
            return
        }
        mirror.enlargeWaiters[e.ordinal, default: []].append { [self] outcome in
            switch outcome {
            case .done(let result):
                var o = json(result)
                o["computing_on"] = mirror.macName
                finish(c, ["ok": true, "image": o, "computing_on": mirror.macName])
            case .downloadRequired:
                finish(c, Response.failure("download_required", String(localized: "macOS does not have Apple’s enlargement model yet"),
                                           help: "silicontrol enlarge \(e.name) --download"))
            case .refused(let sentence):
                finish(c, Response.failure("refused", sentence))
            }
        }
        mirror.send(.enlarge(e.ordinal, download: arguments.count == 2))
    }
}
