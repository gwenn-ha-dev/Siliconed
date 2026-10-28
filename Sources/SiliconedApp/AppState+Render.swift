// **`AppState`: the render** — the queue, the jobs it launches on the engine, the grids, what each
// event of the engine moves. The state itself is in `AppState.swift`.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

extension AppState {
    // MARK: Render

    /// **Puts a render in the queue**, and launches it if nothing is running. `imposedSeed`: "Redo"
    /// (the seed of a history image). `imposedFormat`: an exploration cell's, finished at its sketch's
    /// size — with edit images, `outputFormat` would derive another one from image 1.
    func render(imposedSeed: UInt64? = nil, imposedFormat: RecommendedFormat? = nil) {
        guard canRender else { return }
        followsRender = true
        // Everything that reads the state is taken here: the job no longer depends on the form.
        let firstNumber = imposedSeed ?? (fixedSeed ? seed : UInt64.random(in: 0...UInt64(UInt32.max)))
        let format = imposedFormat ?? outputFormat
        guard let prompts = try? alternatives.get().prompts else { return }
        // **One job per alternative, all at the same seeds**: a series compares the prompts, so
        // nothing else may differ. Each keeps the batch. Its history images carry the expanded prompt
        // — and, citing named subjects, the model's prompt and their pictures after the edit's.
        enqueue(prompts.map { expanded in
            var settings = RenderSettings(identifier: identifier, modelName: card?.name ?? identifier, prompt: expanded,
                                          format: format, steps: steps, loras: loras.filter { !$0.path.isEmpty },
                                          references: references.count, variations: variations, detail: effectiveDetail,
                                          solver: effectiveSolver)
            let used = namedJob(expanded, settings: &settings, references: references)
            let plan = expectedPlan(steps: steps, format: format, withLoRA: loras.contains { !$0.path.isEmpty } || !plainSampling,
                                    references: used.count)
            return Job(ordinal: nextJobNumber(), settings: settings,
                       seeds: (0..<max(1, batch)).map { firstNumber &+ UInt64($0) },
                       references: used, previews: previews,
                       defaultSteps: card?.defaultSteps, plan: plan)
        })
    }

    /// **A grid enters the queue**: one ordinary job per cell, in the grid's order, and the run that
    /// will lay their images out — the path of the exploration and of `silicontrol grid`. `settings`:
    /// what the cells share, the prompt as typed. `sketch`: each cell stops as a sketch (`Sketch`), its
    /// image kept by the run, not the history, and no sheet is drawn unless asked (`addSheet`).
    @discardableResult
    /// `owner`: another Mac's (`MirrorServer`), whose exploration is not shown here (`shows`).
    func enqueueGrid(_ grid: RenderGrid, settings: RenderSettings, references: [ReferenceImage], previews: Bool,
                     sketch: Bool = false, stopAfter: Int = 0, neuralEngine: Bool? = nil, owner: Data? = nil,
                     shows: Bool = true) -> (run: GridRun, jobs: [Job]) {
        lastGrid += 1
        let jobs = grid.cells.map { cell in
            var s = settings
            s.prompt = cell.prompt
            let card = cards.first { $0.id == (cell.model ?? settings.identifier) }
            if let m = cell.model { s.identifier = m; s.modelName = card?.name ?? m }
            s.steps = cell.steps ?? card?.defaultSteps ?? settings.steps
            for (i, f) in cell.loraStrengths.enumerated() where s.loras.indices.contains(i) { s.loras[i].strength = f }
            if let l = cell.addedLoRA { s.loras.append(LoRASlot(path: l, strength: cell.addedStrength)) }
            if let solver = cell.solver { s.solver = solver }
            let used = cell.image.map { references.indices.contains($0) ? [references[$0]] : [] } ?? references
            s.references = used.count
            let model = try? Model.named(s.identifier, in: library)
            // A models axis: a cell whose sampler walks Euler only renders Euler (its caption says none).
            if let model, !model.chain.denoising.solvers.contains(s.solver) { s.solver = .euler }
            // …and a cell whose model offers no Detail renders Normal.
            if let model, !model.chain.denoising.acceptsDetail { s.detail = .normal }
            var plan = model.map { m in
                let f = m.chain.denoising.space.factor
                let plan = m.chain.denoising.plan(height: s.format.height / f, width: s.format.width / f, steps: s.steps,
                                                  start: 0, withLoRA: !s.loras.isEmpty || s.detail != .normal)
                return s.references > 0 ? plan.unreduced : plan
            }
            // A stop at or past the plan's count is a finished image — kept by the exploration all the same.
            let stop = !sketch ? nil : stopAfter > 0 ? min(stopAfter, plan?.evaluations ?? stopAfter)
                : model?.sketchEvaluations(steps: s.steps, width: s.format.width, height: s.format.height)
            if let stop { plan = plan?.sketched(stop) }
            var job = Job(ordinal: nextJobNumber(), settings: s, seeds: [cell.seed], references: used,
                          previews: previews, defaultSteps: card?.defaultSteps, plan: plan)
            job.sketch = stop
            job.neuralEngine = neuralEngine
            job.owner = owner
            return job
        }
        var run = GridRun(ordinal: lastGrid, grid: grid, settings: settings, jobs: jobs.map(\.ordinal),
                          references: references.count, sketch: sketch)
        run.cellJobs = jobs
        run.owner = owner
        gridRuns[run.ordinal] = run
        for (i, job) in jobs.enumerated() { gridCells[job.ordinal] = (run.ordinal, i) }
        if sketch, shows { explored = run }
        enqueue(jobs)
        return (run, jobs)
    }

    /// The grid and the cell a job renders, if it is one.
    func gridPlace(of job: Int) -> (grid: Int, cell: Int)? { gridCells[job] }

    /// **A grid's last cell ended**: its sheet is drawn off the main thread from the cells' PNGs —
    /// an empty cell says why —, then joins the history like an image. No cell rendered: no sheet.
    func finishGrid(_ ordinal: Int) {
        guard let run = gridRuns.removeValue(forKey: ordinal) else { return }
        if explored?.ordinal == ordinal { explored = run }
        // An exploration's sheet is drawn only when asked (`addSheet`): the window is its sheet.
        guard !run.sketch else {
            endedGrids[ordinal] = (run.jobs, nil)
            signal(.sheet(run, nil))
            return
        }
        addSheet(run)
    }

    /// **A grid's sheet joins the history**: drawn off the main thread from the cells' PNGs — an empty
    /// cell says why. No cell rendered: no sheet.
    func addSheet(_ run: GridRun) {
        // In a mirror, the other Mac draws it, from its own cells: the sheet joins its history.
        if let mirror { mirror.send(.addSheet(run.ordinal)); return }
        let ordinal = run.ordinal
        let entries = run.grid.cells.indices.compactMap { run.entries[$0] }
        guard let first = entries.first else {
            endedGrids[ordinal] = (run.jobs, nil)
            signal(.sheet(run, nil))
            return
        }
        let drawing = sheetDrawing(run)
        drawingSheets[ordinal] = run.jobs
        Task {
            do {
                let (png, vignette) = try await Task.detached(priority: .userInitiated) { try drawing.draw() }.value
                // What the cells cost together: their times summed, their memory peaks the highest.
                var timings = first.timings, footprints = first.footprints
                for e in entries.dropFirst() {
                    let t = e.timings, f = e.footprints
                    timings.tokenizer += t.tokenizer; timings.encoder += t.encoder; timings.text += t.text
                    timings.encoding += t.encoding; timings.denoising += t.denoising
                    timings.decoding += t.decoding; timings.total += t.total
                    footprints.start = max(footprints.start, f.start); footprints.tokenizer = max(footprints.tokenizer, f.tokenizer)
                    footprints.encoder = max(footprints.encoder, f.encoder); footprints.text = max(footprints.text, f.text)
                    footprints.encoding = max(footprints.encoding, f.encoding)
                    footprints.denoising = max(footprints.denoising, f.denoising); footprints.end = max(footprints.end, f.end)
                }
                lastImage += 1
                let entry = Entry(ordinal: lastImage, job: run.jobs.last ?? 0, png: png, vignette: vignette,
                                  width: drawing.layout.width, height: drawing.layout.height, seed: run.grid.base.seed,
                                  timings: timings, footprints: footprints,
                                  evaluations: entries.map(\.evaluations).reduce(0, +), settings: run.settings,
                                  metadata: drawing.metadata,
                                  grid: GridInfo(ordinal: run.ordinal, grid: run.grid, empty: run.jobs.count - entries.count),
                                  owner: run.owner)
                history.insert(entry, at: 0)
                if followsRender {
                    selection = entry.id
                    marked = []; anchor = entry.id
                }
                drawingSheets[ordinal] = nil
                endedGrids[ordinal] = (run.jobs, entry.ordinal)
                signal(.sheet(run, entry))
                next()
            } catch {
                problem = Problem(EngineError(error))
                drawingSheets[ordinal] = nil
                endedGrids[ordinal] = (run.jobs, nil)
                signal(.sheet(run, nil))
                next()
            }
        }
    }

    /// **What the sheet says**, in the user's language: the prompt as typed, what the cells share,
    /// the axes' values.
    private func sheetDrawing(_ run: GridRun) -> SheetDrawing {
        let g = run.grid, r = run.settings
        let stack = r.loras
        func lora(_ axis: RenderGrid.Axis?) -> Int? { if case .loraStrength(let slot, _)? = axis { slot } else { nil } }
        func labels(_ axis: RenderGrid.Axis) -> [String] {
            let name = lora(axis).flatMap { stack.indices.contains($0) ? loraName(stack[$0].path) : nil } ?? "LoRA"
            return g.values(of: axis).map { GridWords.label($0, loraName: name, modelName: modelName, addedName: loraName) }
        }
        func varies(_ test: (RenderGrid.Axis) -> Bool) -> Bool { test(g.x) || g.y.map(test) == true }
        // What every cell shares: what no axis varies. A model axis gives each its own steps.
        let models = varies { if case .models = $0 { true } else { false } }
        var shared = (models ? [] : [r.modelName]) + ["\(r.format.width) × \(r.format.height)"]
        if !models, !varies({ if case .steps = $0 { true } else { false } }) { shared.append(String(localized: "\(r.steps) steps")) }
        if !varies({ switch $0 { case .seeds, .consecutiveSeeds: true; default: false } }) {
            shared.append(String(localized: "seed \(String(g.base.seed))"))
        }
        for (i, l) in stack.enumerated() where lora(g.x) != i && lora(g.y) != i {
            shared.append("\(loraName(l.path)) \(Figures.strength(l.strength))")
        }
        if r.solver != .euler, !varies({ if case .solvers = $0 { true } else { false } }) { shared.append(r.solver.choice) }
        if run.references > 0, !varies({ if case .images = $0 { true } else { false } }) {
            shared.append(String(localized: "edit, \(run.references) images"))
        }
        let axes = [g.x] + (g.y.map { [$0] } ?? [])
        let names = axes.map { GridWords.name($0, grid: g, stack: stack, loraName: loraName) }
        let layout = g.sheet(imageWidth: r.format.width, imageHeight: r.format.height)
        let cells = g.cells.enumerated().map { i, cell in
            SheetDrawing.Cell(column: cell.column, row: cell.row, png: run.entries[i]?.png,
                              note: GridWords.note(run.issues[i]))
        }
        return SheetDrawing(layout: layout, cells: cells, title: g.title, subtitle: shared.joined(separator: " · "),
                            columnLabels: labels(g.x), rowLabels: g.y.map(labels) ?? [],
                            metadata: ["Software": "Siliconed grid", "Title": g.title,
                                       "Description": "x: \(names[0])" + (names.count > 1 ? " · y: \(names[1])" : "")
                                           + " · " + shared.joined(separator: " · "),
                                       "grid": "\(g.columns)x\(g.rows)"])
    }

    /// Ready-made jobs enter the queue, in order — the path of the button and of the remote control.
    func enqueue(_ jobs: [Job]) {
        guard !jobs.isEmpty else { return }
        if let mirror {
            // The form is blocked while the link is down; what reaches here by another door is said, not dropped.
            if let down = linkBlocker { problem = Problem(down); flash(down); return }
            do { for j in jobs { try mirror.render(j) } } catch { problem = Problem(error); flash(Problem(error).text) }
            return
        }
        // A measure that failed before the first render is retried by the next click, not by itself.
        if !settingsRead, case .failed = diagnostic { diagnostic = .ready }
        file.append(contentsOf: jobs)
        if inProgress { flash(String(localized: "Added to the queue (\(file.count) waiting)")) }
        // An exploration across models: all of them copied to this Mac before its first cell, if needed.
        let models = Set(jobs.map(\.settings.identifier))
        if models.count > 1 || models.first != identifier {
            warm(models.sorted(), loras: Array(Set(jobs.flatMap { $0.settings.loras.map(\.path) })).filter { !$0.isEmpty })
        }
        next()
    }

    /// Launches the queue's first job if nothing is running. Called again at the end of each render,
    /// succeeded, stopped or failed: the queue empties by itself.
    /// The diagnostic occupies the GPU and the CPU like a render: the queue waits for it (and it for the
    /// queue) — and for an installation, which reads and writes GBs (`installBlocker`). **No render
    /// without this Mac's measure** (`settingsRead`): the first job starts it, and its end calls `next`.
    /// An « Enlarge ×4 » under way and a grid's sheet being drawn are waited for too — memory a render
    /// would add to, never counted in its plan —; each calls `next` when it ends (`enlarged`, `addSheet`).
    func next() {
        guard mirror == nil else { return }
        guard currentJob == nil, !file.isEmpty, !diagnosticRunning, forgeInProgress == nil else {
            if currentJob == nil, let a = activity {
                ProcessInfo.processInfo.endActivity(a); activity = nil
                DockTile.finished()
            }
            return
        }
        guard enlarging == nil, drawingSheets.isEmpty else { return }
        guard settingsRead else {
            if case .ready = diagnostic { measure() }
            return
        }
        if activity == nil {
            activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .latencyCritical],
                                                             reason: "Image render")
        }
        let job = file.removeFirst()
        currentJob = job
        problem = nil; fraction = 0; estimatedRemaining = nil; stage = nil; preview = nil
        stageStates = [:]; stageSeconds = [:]; stageModel = job.settings.identifier
        journal = []   // the warnings shown are this render's, not the last one that had any
        currentStep = 0; totalSteps = 0; batchImage = 0; batchSize = job.seeds.count
        measurements.beginJob(job)
        jobImages = []
        signal(.started(job))
        task = Task {
            let issue = await execute(job)
            currentJob = nil; preview = nil; task = nil
            if case .failed(let p) = issue, p.code == EngineError.licenseNotAccepted(model: "").code {
                // **The engine holds the door, the app presents**: the job waits for the answer.
                problem = nil
                awaitingLicense.append(job)
                if licenseRequest == nil, let card = cards.first(where: { $0.id == job.settings.identifier }) {
                    managementOpen = false
                    licenseRequest = LicenseRequest(card: card)
                }
            } else {
                complete(job, issue, images: jobImages, warnings: journal)
            }
            next()
        }
    }

    /// **The user accepted the license**: the library records it (the engine reads it at every render),
    /// and the jobs that waited for it go back to the head of the queue, in their order.
    func acceptLicense(_ card: ModelCard) {
        if let mirror { answerLicense(card, accepted: true, via: mirror); return }
        do { try library.acceptLicense(card) } catch {
            problem = Problem(error)
            return
        }
        licenseRequest = nil
        let waiting = awaitingLicense.filter { $0.settings.identifier == card.id }
        awaitingLicense.removeAll { $0.settings.identifier == card.id }
        file.insert(contentsOf: waiting, at: 0)
        presentNextLicense()
        next()
    }

    /// **Declined**: the jobs that waited end with the engine's refusal, as they would have.
    func declineLicense(_ card: ModelCard) {
        if let mirror { answerLicense(card, accepted: false, via: mirror); return }
        licenseRequest = nil
        let refused = awaitingLicense.filter { $0.settings.identifier == card.id }
        awaitingLicense.removeAll { $0.settings.identifier == card.id }
        let p = Problem(EngineError.licenseNotAccepted(model: card.id))
        for job in refused { complete(job, .failed(p)) }
        problem = p
        presentNextLicense()
    }

    private func presentNextLicense() {
        guard licenseRequest == nil, let job = awaitingLicense.first,
              let card = cards.first(where: { $0.id == job.settings.identifier }) else { return }
        licenseRequest = LicenseRequest(card: card)
    }

    private func execute(_ job: Job) async -> Issue {
        let settings = job.settings
        var loraEntries = settings.loras.map { LoRAEntry($0.path, strength: $0.strength) }
        do {
            // A library on another disk is read from its copy on this Mac (`WorkingCopy`): brought up to
            // date first — nothing to do when the model's choice already did it.
            var read = library
            if let copy = workingCopy {
                let cancellation = Cancellation()
                try await withTaskCancellationHandler {
                    try await prepareCopy(models: [settings.identifier], loras: settings.loras.map(\.path), cancellation: cancellation)
                } onCancel: { cancellation.cancel() }
                read = copy.library
                loraEntries = settings.loras.map { LoRAEntry(copy.path(of: $0.path), strength: $0.strength) }
            } else {
                try library.checkReachable()
            }
            let model = try Model.named(settings.identifier, in: read)
            // The references are decoded off the main thread (3072 px at most), in their order; the
            // engine then brings each to its size: the full-size photos do not live during the render.
            let sources = job.references
            let references = try await Task.detached(priority: .userInitiated) {
                try sources.map { try $0.imageRGB() }
            }.value
            var request = Request(settings.modelPrompt ?? settings.prompt, width: settings.format.width,
                                  height: settings.format.height, seed: job.seeds[0],
                                  steps: settings.steps == job.defaultSteps ? nil : settings.steps,
                                  loras: loraEntries, previews: job.previews)
            request.references = references
            request.sketch = job.sketch
            request.detail = settings.detail
            request.solver = settings.solver
            request.variations = settings.variations
            request.neuralEngine = job.neuralEngine ?? neuralEngine(for: settings.identifier)

            var progress = Engine.Progress()
            for try await item in Engine().events(request, seeds: job.seeds, model: model) {
                progress.receive(item)
                measurements.receive(item)
                if progress.fraction != fraction { DockTile.show(fraction: progress.fraction, waiting: file.count) }
                fraction = progress.fraction
                if progress.estimatedRemaining != estimatedRemaining {
                    estimatedRemaining = progress.estimatedRemaining
                    remainingInstant = Date()
                }
                switch item {
                case .stage(let e, let image):
                    // The stage that was working is done; this one lights up.
                    if let previous = stage, stageStates[previous] == .active { stageStates[previous] = .done }
                    // The denoising ends where the decoding begins: its steps, for this image.
                    if e == .decoding, batchImage < measurements.stepSeconds.count {
                        stageSeconds[.denoising] = measurements.stepSeconds[batchImage].reduce(0, +)
                    }
                    stageStates[e] = .active
                    stage = e
                    batchImage = image
                    signal(.stage(job, e, image: image))
                case let .text(_, tokenizer, encoder, _):
                    stageSeconds[.text] = tokenizer + encoder
                case .encoding(let s):
                    stageSeconds[.image] = s
                case .decoding(let s):
                    stageSeconds[.decoding] = s
                case let .step(image, index, total, _, seconds, _, _):
                    batchImage = image; currentStep = index; totalSteps = total
                    signal(.step(job, image: image, index: index, total: total, seconds: seconds))
                case .preview(let a):
                    preview = a.cgImage()
                case .image(_, let renderResult):
                    stageStates[.decoding] = .done
                    // The render against the map's prediction (`RenderLoop`): kept for the report and the next launch.
                    checkMap(renderResult, model: model, references: references.count)
                    // The PNG and the thumbnail are made off the main thread; the engine has already
                    // moved on to the next image of the batch.
                    // Named subjects: the PNG keeps the prompt typed, the model's beside it. Nobody
                    // cited (the switch off among others): the engine's metadata, untouched.
                    let named = settings.modelPrompt.map {
                        NamedSubjects.metadata(renderResult.metadata, typed: settings.prompt, modelPrompt: $0)
                    }
                    let finished = try await Task.detached(priority: .userInitiated) {
                        try FinishedImage(renderResult, metadata: named)
                    }.value
                    lastImage += 1
                    let entry = Entry(ordinal: lastImage, job: job.ordinal, png: finished.png, vignette: finished.vignette,
                                        width: renderResult.image.width, height: renderResult.image.height,
                                        seed: renderResult.seed, timings: renderResult.timings, footprints: renderResult.footprints,
                                        evaluations: renderResult.evaluations,
                                        settings: { var s = settings; s.neuralEngine = renderResult.neuralEngine; return s }(),
                                        metadata: named ?? renderResult.metadata, references: job.references,
                                        sketch: renderResult.sketch, owner: job.owner)
                    // A sketch belongs to its exploration, not to the history.
                    if job.sketch != nil, let at = gridCells[job.ordinal] {
                        gridRuns[at.grid]?.entries[at.cell] = entry
                        if explored?.ordinal == at.grid { explored = gridRuns[at.grid] }
                        preview = nil
                        sessionSeconds += renderResult.timings.total
                        jobImages.append(entry.ordinal)
                        signal(.image(job, entry))
                        continue
                    }
                    history.insert(entry, at: 0)
                    if followsRender {
                        selectedImage = (entry.id, finished.image)
                        selection = entry.id
                        marked = []; anchor = entry.id
                    }
                    preview = nil
                    sessionImages += 1
                    sessionSeconds += renderResult.timings.total
                    jobImages.append(entry.ordinal)
                    if let at = gridCells[job.ordinal] { gridRuns[at.grid]?.entries[at.cell] = entry }
                    signal(.image(job, entry))
                case .warning(let message):
                    journal.append(message)
                default:
                    break
                }
            }
            if var v = measurements.speed {
                // A cached prompt teaches nothing about the encoder: the cost learned before stays.
                if v.text == nil { v.text = speeds[job.speedKey]?.text }
                speeds[job.speedKey] = v
            }
            // Cancelled, the `Task` leaves the loop WITHOUT an error (AsyncThrowingStream).
            if Task.isCancelled {
                for (s, v) in stageStates where v == .active { stageStates[s] = .idle }
                flash(String(localized: "Render stopped"))
                return .stopped
            }
            return .finished
        } catch {
            // Every refusal of the engine is one `EngineError` case (`EngineError(_:)` folds the rest).
            let e = EngineError(error)
            if let s = stage, stageStates[s] == .active { stageStates[s] = e == .cancelled ? .idle : .failed }
            if e == .cancelled {
                flash(String(localized: "Render stopped"))
                return .stopped
            }
            let p = Problem(e)
            problem = p
            return .failed(p)
        }
    }

    /// The per-image plan that the chosen model's denoiser foresees for these settings.
    /// `withLoRA`: a LoRA or a Detail — either turns the spectral schedule off; so do references (an edit).
    func expectedPlan(steps: Int, format: RecommendedFormat, withLoRA: Bool, references: Int) -> DenoisingPlan? {
        // A plan reads the settings (`Spectral.forced`): none before they are (`settingsRead`).
        guard settingsRead, let chain else { return nil }
        let f = chain.denoising.space.factor
        let plan = chain.denoising.plan(height: format.height / f, width: format.width / f,
                                        steps: steps, start: 0, withLoRA: withLoRA)
        return references > 0 ? plan.unreduced : plan
    }

    /// The remaining time of the running render at instant `date`, counted down from the last event.
    /// Before its first full step the engine has no pace of its own yet: the pace learned for this
    /// model and format stands in, counted from the job's start — else the render weighed nothing.
    func remaining(at date: Date) -> Double? {
        if let e = estimatedRemaining { return max(0, e - date.timeIntervalSince(remainingInstant)) }
        guard let job = currentJob, let planned = estimation(job), let begin = measurements.begin else { return nil }
        return max(0, planned - date.timeIntervalSince(begin))
    }

    /// **The time remaining before the queue is empty**, and whether it is complete: the running render
    /// (its own estimate), then each wait at the learned pace for its model and format.
    /// A wait never measured makes the estimate partial — a minimum, said as such.
    func remainingInQueue(at date: Date) -> (seconds: Double, isComplete: Bool) {
        let inProgress = remaining(at: date)
        var total = inProgress ?? 0
        var isComplete = !self.inProgress || inProgress != nil
        for t in file {
            if let e = estimation(t) { total += e } else { isComplete = false }
        }
        return (total, isComplete)
    }

    /// The planned duration of a waiting job, or `nil` if it was never measured.
    func estimation(_ t: Job) -> Double? {
        if mirror != nil { return mirrorEstimates[t.ordinal] }
        guard let v = speeds[t.speedKey], let plan = t.plan else { return nil }
        return v.duration(plan, images: t.seeds.count)
    }

    /// Stops the running render; the queue continues. The engine checks for cancellation between the
    /// layers: stopping takes 0.03 to 0.41 s.
    func cancel() { if let mirror { mirror.send(.cancel) } else { task?.cancel() } }

    /// Removes a render that is still waiting — in the queue, or for its model's license.
    func remove(_ job: Job) {
        if let mirror { mirror.send(.removeJob(job.ordinal)); return }
        if awaitingLicense.contains(where: { $0.id == job.id }) {
            awaitingLicense.removeAll { $0.id == job.id }
            if let id = licenseRequest?.card.id, !awaitingLicense.contains(where: { $0.settings.identifier == id }) {
                licenseRequest = nil
                presentNextLicense()
            }
            complete(job, .removed)
            return
        }
        guard file.contains(where: { $0.id == job.id }) else { return }
        file.removeAll { $0.id == job.id }
        complete(job, .removed)
    }

    /// Empties the queue and stops the running render.
    func stopAll() {
        if let mirror { mirror.send(.stopAll); return }
        clearQueue()
        task?.cancel()
    }

    func clearQueue() {
        if let mirror { mirror.send(.clearQueue); return }
        let removedNames = file + awaitingLicense
        file.removeAll()
        awaitingLicense.removeAll()
        licenseRequest = nil
        for t in removedNames { complete(t, .removed) }
    }

    /// Whether « Variations » can take this image: an ordinary history image of an installed model,
    /// no installation running.
    func canVary(_ entry: Entry) -> Bool {
        installBlocker == nil && entry.grid == nil && entry.sketch == nil && entry.enlarged == nil
            && entry.cropped == nil
            && readySet.contains(entry.settings.identifier)
    }

    /// **« Variations »**: cousins of a history image — exactly its settings (model, prompt, LoRA,
    /// size, steps, seed, edit images), each with a variation seed drawn at random at `strength`
    /// (`Variation.Amount`, its strength for the model's family), ordinary jobs whose images join the history.
    /// **As many as the rack's « Images »** (`batch`, 1 by default): at Qwen's ~200 s an edit, a fixed
    /// four made one click a quarter of an hour.
    ///
    /// **A variation of a variation turns the clicked image's noise**, not its parent's: the new
    /// variation is appended to the image's own (`Request.variations` applies them in order). The
    /// strength is the new request's; the origin is the image clicked — what the user sees and likes.
    /// The form is not touched: this is a gesture on an image, not a change of settings.
    @discardableResult
    func vary(_ entry: Entry, _ amount: Variation.Amount, count: Int? = nil) -> [Job] {
        guard canVary(entry), let family = cards.first(where: { $0.id == entry.settings.identifier })?.family else { return [] }
        let strength = amount.strength(for: family)
        followsRender = true
        let card = cards.first { $0.id == entry.settings.identifier }
        let plan = (try? Model.named(entry.settings.identifier, in: library)).map { m in
            let f = m.chain.denoising.space.factor
            let plan = m.chain.denoising.plan(height: entry.settings.format.height / f, width: entry.settings.format.width / f,
                                              steps: entry.settings.steps, start: 0,
                                              withLoRA: !entry.settings.loras.isEmpty || entry.settings.detail != .normal)
            return entry.settings.references > 0 ? plan.unreduced : plan
        }
        let jobs = (0..<(count ?? batch)).map { _ in
            var settings = entry.settings
            settings.variations.append(Variation(seed: UInt64.random(in: 0...UInt64(UInt32.max)), strength: strength))
            return Job(ordinal: nextJobNumber(), settings: settings, seeds: [entry.seed], references: entry.references,
                       previews: previews, defaultSteps: card?.defaultSteps, plan: plan)
        }
        enqueue(jobs)
        return jobs
    }

    /// Redo the chosen image: same settings, same seed.
    func redo(_ entry: Entry) {
        restoreSettings(entry)
        // A sheet: its exploration again, at its seed.
        if entry.grid != nil { explore(seed: entry.seed); return }
        render(imposedSeed: entry.seed)
    }

    func restoreSettings(_ entry: Entry) {
        restoreSettings(entry.settings)
        // Fixed, or the next render draws a new seed and the settings reused are not the image's.
        seed = entry.seed
        fixedSeed = true
        // After the seed: setting it drops a variation (`seed.didSet`).
        variations = entry.settings.variations
        if let g = entry.grid?.grid {
            // A sheet's prompt is the one typed, its groups the grid's axes: not escaped.
            setGridForm(g, prompt: entry.settings.prompt)
            MainWindow.shared.showExploration()
        }
    }

    /// The form takes these settings — those of a history image, or of a job that the
    /// remote control has just queued: the app shows what it computes, as after a click.
    func restoreSettings(_ r: RenderSettings) {
        identifier = r.identifier
        // The job's prompt is an expanded one: its braces are characters, not a series.
        prompt = PromptAlternatives.escaping(r.prompt)
        orientation = r.format.orientation
        format = r.format
        steps = r.steps
        loras = r.loras.map(allowedStrength)
        detail = r.detail
        solver = r.solver
    }

    /// **A Siliconed PNG dropped on the canvas**: the form takes what its metadata says
    /// (`Engine.Render.Recipe`), through `restoreSettings` as "Reuse These Settings" does, and the
    /// seed is fixed. What cannot come back is said (a `Problem`), and the rest is restored anyway.
    /// `false` when the file is not a PNG written by Siliconed: the drop keeps its usual meaning.
    ///
    /// Incognito: the file is only read — no panel, no recent item, no copy.
    func restoreSettings(fromPNG url: URL) -> Bool {
        guard url.isFileURL, url.pathExtension.lowercased() == "png",
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let recipe = Engine.Render.Recipe(metadata: PNG.text(data)) else { return false }
        let file = url.lastPathComponent
        var notes: [String] = []

        // The PNG names a family, not a card (a fine-tune writes its family's name): the form's
        // model if it is of that family, otherwise the publisher's card, otherwise an imported one.
        let ready = visibleCards.filter { $0.family.rawValue == recipe.model && readySet.contains($0.id) }
        let target = ready.first { $0.id == identifier } ?? ready.first { !$0.isImported } ?? ready.first
        if target == nil, let model = recipe.model {
            // Steps and LoRAs belong to that model: the form's stay.
            let name = cards.first { $0.family.rawValue == model }?.name ?? model
            notes.append(String(localized: "\(name) isn't installed: its steps and LoRAs are left out."))
        }

        var stack: [LoRASlot] = loras
        if let target {
            stack = []
            var lost: [String] = []
            for entry in recipe.loras {
                if stack.count < Self.maxLoRA, let found = library.lora(entry.path, for: target.id) {
                    stack.append(LoRASlot(path: found.path, strength: entry.strength))
                } else {
                    lost.append(entry.path)
                }
            }
            if !lost.isEmpty {
                let names = lost.formatted(.list(type: .and))
                notes.append(String(localized: "LoRA not found: \(names)."))
            }
            if !negativeLoRAStrengths, stack.contains(where: { $0.strength < 0 }) {
                notes.append(String(localized: "A negative LoRA strength comes back as 0: turn on Experimental → Negative LoRA Strengths."))
            }
        }

        // The size if the model it goes to renders it (`Format.check(…, family:)`), the form's otherwise.
        var restoredFormat = format
        let family = (target ?? card)?.family ?? .zImage   // no model at all: the general rules (Z-Image adds none)
        if let w = recipe.width, let h = recipe.height, (try? Format.check(width: w, height: h, family: family)) != nil {
            restoredFormat = RecommendedFormat(w, h)
        }
        if recipe.sourceImageMissing {
            notes.append(String(localized: "The image it started from isn't in the file (img2img)."))
        }
        if !recipe.unreadable.isEmpty {
            let keys = recipe.unreadable.formatted(.list(type: .and))
            notes.append(String(localized: "Unreadable in the file: \(keys)."))
        }

        let model = target ?? card
        restoreSettings(RenderSettings(identifier: model?.id ?? identifier, modelName: model?.name ?? "",
                                       prompt: recipe.prompt ?? prompt, format: restoredFormat,
                                       steps: target == nil ? steps : (recipe.steps ?? target!.defaultSteps),
                                       loras: stack, detail: recipe.detail, solver: recipe.solver))
        if recipe.solver != .euler, !offeredSolvers.contains(recipe.solver) {
            notes.append(String(localized: "Made with the \(recipe.solver.choice) solver, offered in Developer Mode: Euler will be used."))
        }
        if recipe.detail != .normal, !offersDetail {
            notes.append(String(localized: "\(model?.name ?? identifier) doesn't offer Detail: Normal will be used."))
        }
        if let s = recipe.seed { seed = s; fixedSeed = true }
        // A variation is its seed's: without the seed, there is nothing to vary.
        variations = recipe.seed == nil ? [] : recipe.variations
        flash(String(localized: "Settings of \(file) restored"))
        if !notes.isEmpty {
            problem = Problem(String(localized: "Not everything came back from \(file)."),
                              suggestion: notes.joined(separator: "\n"))
        }
        return true
    }
}
