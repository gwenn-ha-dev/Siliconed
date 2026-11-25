// **`AppState`: the render** — the queue, the grids, the estimates, a job removed or cancelled. A job's run on the engine
// is in `AppState+Execute.swift`; the state itself is in `AppState.swift`.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

extension AppState {
    // MARK: Render

    /// **Puts a render in the queue**, and launches it if nothing is running. `imposedSeed`: "Redo" (the seed of a history
    /// image). `imposedFormat`: "Redo" (its size) — with edit images, `outputFormat` would derive another one from image 1.
    /// **Something varies** (`Series.swift`): the series' grid, one finished image per cell.
    func render(imposedSeed: UInt64? = nil, imposedFormat: RecommendedFormat? = nil) {
        guard canRender else { return }
        // Everything that reads the state is taken here: the job no longer depends on the form.
        let firstNumber = imposedSeed ?? (fixedSeed ? seed : Request.drawnSeed())
        // The seed drawn shown in the rack's field, greyed: Fixed then keeps it.
        if imposedSeed == nil, !fixedSeed { seed = firstNumber }
        let format = imposedFormat ?? outputFormat
        guard let prompts = try? alternatives.get().prompts, (try? carried.get()) != nil else { return }
        if case .success(let g?) = seriesGrid(seed: firstNumber) {
            renderSeries(g, format: format)
            return
        }
        followsRender = true
        // The edit's images as the render reads them, decoded once for the whole series.
        let edit = renderReferences
        // One job, one image: a prompt with groups, or a lot, is a series.
        enqueue(prompts.map { expanded in
            var settings = RenderSettings(identifier: identifier, modelName: card?.name ?? identifier, prompt: expanded,
                                          format: format, steps: steps, loras: activeLoRAs,
                                          references: edit.count, variations: variations, detail: effectiveDetail,
                                          solver: effectiveSolver)
            // **The subjects the text cites, translated here** (`namedJob`): their pictures after the edit's, the model
            // reading the rewrite, the text kept as typed.
            let used = namedJob(expanded, settings: &settings, references: edit, model: model(identifier),
                                capacity: mirror != nil ? imageCapacity : nil)
            let start = usedStart
            if start != nil { settings.strength = startStrength }
            let plan = expectedPlan(steps: steps, format: format, strength: settings.strength)
            var job = Job(ordinal: nextJobNumber(), settings: settings,
                          seeds: [firstNumber],
                          references: used, previews: previews,
                          defaultSteps: card?.defaultSteps, plan: plan)
            job.start = start
            return job
        })
    }

    /// **The series « Generate » renders**: its grid, each cell a finished image into the history under the series'
    /// tile, the canvas showing the grid as the cells come (the history does not take the canvas from it: `followsRender`
    /// off). In a mirror, the other Mac makes the cells, as it makes a render, and publishes the series back.
    private func renderSeries(_ g: RenderGrid, format: RecommendedFormat) {
        var settings = RenderSettings(identifier: identifier, modelName: card?.name ?? identifier, prompt: prompt,
                                      format: format, steps: steps, loras: activeLoRAs, references: renderReferences.count,
                                      variations: variations, detail: effectiveDetail, solver: effectiveSolver)
        if usedStart != nil { settings.strength = startStrength }
        followsRender = false
        if let mirror {
            do { try mirror.explore(g, settings: settings, references: renderReferences, start: usedStart) }
            catch { problem = Problem(error); flash(Problem(error).text) }
        } else {
            enqueueGrid(g, settings: settings, references: renderReferences, previews: previews, named: true, start: usedStart)
        }
    }

    /// **A grid enters the queue**: one ordinary job per cell, in the grid's order, and the run that
    /// will lay their images out — the path of a series and of `silicontrol grid`. `settings`:
    /// what the cells share, the prompt as typed. `named`: each cell's `@name`s translated as a render's
    /// are (`namedJob`). `owner`: another Mac's (`MirrorServer`), whose series this window does not open (`shows`).
    /// `start`: the starting image every cell renders from (img2img, at `settings.strength`) — dropped by a cell whose model
    /// does not start from one (a models axis).
    @discardableResult
    func enqueueGrid(_ grid: RenderGrid, settings: RenderSettings, references: [ReferenceImage], previews: Bool,
                     named: Bool = false, owner: Data? = nil,
                     shows: Bool = true, start: ReferenceImage? = nil) -> (run: GridRun, jobs: [Job]) {
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
            if let detail = cell.detail { s.detail = detail }
            if let strength = cell.startStrength { s.strength = strength }
            var used = cell.image.map { references.indices.contains($0) ? [references[$0]] : [] } ?? references
            s.references = used.count
            let model = try? Model.named(s.identifier, in: library)
            if named {
                used = namedJob(cell.prompt, settings: &s, references: used, model: model,
                                capacity: mirror != nil ? imageCapacity : nil)
            }
            // A models axis: a cell whose sampler walks Euler only renders Euler (its caption says none).
            if let model, !model.chain.denoising.solvers.contains(s.solver) { s.solver = .euler }
            // …and a cell whose model offers no Detail renders Normal.
            if let model, !model.chain.denoising.acceptsDetail { s.detail = .normal }
            // …and a cell whose model does not start from an image renders from noise.
            let cellStart = start != nil && model?.chain.acceptsStartImage == true ? start : nil
            if cellStart == nil { s.strength = nil }
            let plan = model.map { Self.plan($0.chain.denoising, steps: s.steps, format: s.format, strength: s.strength) }
            var job = Job(ordinal: nextJobNumber(), settings: s, seeds: [cell.seed], references: used,
                          previews: previews, defaultSteps: card?.defaultSteps, plan: plan)
            job.start = cellStart
            job.owner = owner
            return job
        }
        var run = GridRun(ordinal: lastGrid, grid: grid, settings: settings, jobs: jobs.map(\.ordinal),
                          references: references.count)
        run.cellJobs = jobs
        run.owner = owner
        series[run.ordinal] = run
        for (i, job) in jobs.enumerated() { gridCells[job.ordinal] = (run.ordinal, i) }
        if shows { shownSeries = run.ordinal }
        enqueue(jobs)
        return (run, jobs)
    }

    /// The grid and the cell a job renders, if it is one.
    func gridPlace(of job: Int) -> (grid: Int, cell: Int)? { gridCells[job] }

    /// Ready-made jobs enter the queue, in order — the path of the button and of the remote control.
    func enqueue(_ jobs: [Job]) {
        guard !jobs.isEmpty else { return }
        if let mirror {
            // The form is blocked while the link is down; what reaches here by another door is said, not dropped.
            if let down = linkBlocker { problem = Problem(down); flash(down); return }
            do { for j in jobs { try mirror.render(j) } } catch { problem = Problem(error); flash(Problem(error).text) }
            return
        }
        file.append(contentsOf: jobs)
        if inProgress { flash(String(localized: "Added to the queue (\(file.count) waiting)")) }
        next()
    }

    /// Launches the queue's first job if nothing is running. Called again at the end of each render,
    /// succeeded, stopped or failed: the queue empties by itself.
    /// The queue waits for an installation, which reads and writes GBs (`installBlocker`).
    /// An « Enlarge ×4 » under way is waited for too — memory a render would add to, never counted in its
    /// plan —; it calls `next` when it ends (`enlarged`).
    func next() {
        guard mirror == nil else { return }
        guard currentJob == nil, !file.isEmpty, libraryJob == nil else {
            // The activity and the Dock's count end with the queue, not while its jobs wait for the library.
            if currentJob == nil, file.isEmpty, let a = activity {
                ProcessInfo.processInfo.endActivity(a); activity = nil
                DockTile.finished()
            }
            return
        }
        guard enlarging == nil else { return }
        if activity == nil {
            activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .latencyCritical],
                                                             reason: "Image render")
        }
        let job = file.removeFirst()
        train = [job] + riders(of: job)
        delivered = []; started = [job.id]
        currentJob = job
        problem = nil; fraction = 0; estimatedRemaining = nil; stage = nil; prerenders = [:]
        stageStates = [:]; stageSeconds = [:]; stageModel = job.settings.identifier
        journal = []   // the warnings shown are this render's, not the last one that had any
        currentStep = 0; totalSteps = 0; batchImage = 0; batchSize = job.seeds.count
        measurements.beginJob(job)
        jobImages = []
        signal(.started(job))
        task = Task {
            let issue = await execute(job)
            let (last, riders) = (currentJob, train)
            currentJob = nil; prerenders = [:]; task = nil; train = []
            if case .failed(let p) = issue, p.code == EngineError.licenseNotAccepted(model: "").code {
                // **The engine holds the door, the app presents**: the job waits for the answer.
                problem = nil
                awaitingLicense.append(job)
                if licenseRequest == nil, let card = cards.first(where: { $0.id == job.settings.identifier }) {
                    managementOpen = false
                    licenseRequest = LicenseRequest(card: card)
                }
            } else if riders.count > 1 {
                settle(riders, last: last, issue)
            } else {
                complete(job, issue, images: jobImages, warnings: journal)
            }
            next()
        }
    }

    /// **The cells that render with `job`**: the queue's next ones, of its series, that are the same render under
    /// another seed (`Job.rendersAs`). They stay in the queue.
    private func riders(of job: Job) -> [Job] {
        guard let grid = gridCells[job.ordinal]?.grid else { return [] }
        return Array(file.prefix(Self.maxTrain - 1).prefix { gridCells[$0.ordinal]?.grid == grid && $0.rendersAs(job) })
    }

    /// **The job of an image of the running call**, made the running one: the one that was goes back to the queue, at
    /// its rank among the train's, unless its image came. `nil`: that job was removed from the queue meanwhile — the
    /// engine still computes its image (a call is not cut short), and nothing is made of it.
    func turn(to image: Int) -> Job? {
        guard train.count > 1, train.indices.contains(image) else { return currentJob }
        let rider = train[image]
        if currentJob?.id == rider.id { return rider }
        guard let at = file.firstIndex(where: { $0.id == rider.id }) else { return nil }
        file.remove(at: at)
        if let was = currentJob, !delivered.contains(was.id), let rank = train.firstIndex(where: { $0.id == was.id }) {
            let ranks = Dictionary(uniqueKeysWithValues: train.enumerated().map { ($1.id, $0) })
            file.insert(was, at: file.firstIndex { (ranks[$0.id] ?? .max) > rank } ?? file.endIndex)
        }
        currentJob = rider
        if started.insert(rider.id).inserted { signal(.started(rider)) }
        return rider
    }

    /// **A train's call ended**: the jobs whose image came have ended already (`execute`); the others, by how it ended.
    /// - stopped: « Stop » stops the running render and the queue continues — the cells not decoded yet with it;
    /// - refused or failed: one request, one answer — every job still open ends with it, none is tried again alone;
    /// - finished: a job still open is one the engine neither gave an image nor said lost (`Event.unusable`), a fault.
    private func settle(_ riders: [Job], last: Job?, _ issue: Issue) {
        if case .stopped = issue {
            if let last, !delivered.contains(last.id) { complete(last, .stopped, warnings: journal) }
            return
        }
        var verdict = issue
        if case .finished = issue {
            verdict = .failed(Problem(EngineError.internalFailure(component: "engine", detail: "the render ended without this job's image")))
        }
        for rider in riders where !delivered.contains(rider.id) {
            guard rider.id == last?.id || file.contains(where: { $0.id == rider.id }) else { continue }
            file.removeAll { $0.id == rider.id }
            if case .failed(let p) = verdict { problem = p }
            complete(rider, verdict, warnings: journal)
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

    /// The per-image plan that the chosen model's denoiser foresees for these settings.
    func expectedPlan(steps: Int, format: RecommendedFormat, strength: Double? = nil) -> DenoisingPlan? {
        chain.map { Self.plan($0.denoising, steps: steps, format: format, strength: strength) }
    }

    /// **The per-image plan a denoiser foresees** — the rack's render, a grid's cell, a variation. `strength`: img2img's,
    /// when the render starts from an image — it starts that many steps in (`Strength.startStep`).
    static func plan(_ denoising: any DenoisingModule, steps: Int, format: RecommendedFormat, strength: Double?) -> DenoisingPlan {
        let f = denoising.space.factor
        let start = strength.map { min(steps, Strength.startStep(steps: steps, strength: $0)) } ?? 0
        return denoising.plan(height: format.height / f, width: format.width / f, steps: steps, start: start)
    }

    /// The remaining time of the running render at instant `date`, counted down from the last event.
    /// Before its first step the engine has no pace of its own yet: the pace learned for this
    /// model and format stands in, counted from the job's start — else the render weighed nothing.
    func remaining(at date: Date) -> Double? {
        if let e = estimatedRemaining { return max(0, e - date.timeIntervalSince(remainingInstant)) }
        guard let job = currentJob, let planned = estimation(job), let begin = measurements.begin else { return nil }
        return max(0, planned - date.timeIntervalSince(begin))
    }

    /// **Core AI compiles**: a function the cache of the render under way does not hold (`Engine.Held.compiling`) — a
    /// model's first render, a new size, another share of the Neural Engine —, or what a preparation compiles ahead
    /// (`compilingAhead`), which a render asked meanwhile waits for. What it waits for then is paid once, and no
    /// countdown says it.
    var compilingCoreAI: Bool {
        aheadCompiles || inProgress && renderCompiles
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

    /// The planned duration of these jobs together, or `nil` if one was never measured.
    func estimation(of jobs: [Job]) -> Double? {
        var total = 0.0
        for j in jobs {
            guard let e = estimation(j) else { return nil }
            total += e
        }
        return total
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

    /// The window closes: nothing must keep occupying the GPU with no view to show it.
    func unmount() {
        toastTask?.cancel()
        clearQueue()
        task?.cancel()
    }
}
