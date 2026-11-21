// **`AppState`: the render** — the queue, the jobs it launches on the engine, the grids, what each
// event of the engine moves. The state itself is in `AppState.swift`.

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
        let firstNumber = imposedSeed ?? (fixedSeed ? seed : UInt64.random(in: 0...UInt64(UInt32.max)))
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
            do { try mirror.explore(g, settings: settings, references: renderReferences) }
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
            let cellStart = start != nil && model?.chain.entries.contains(where: { $0.kind == .image }) == true ? start : nil
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
    private func turn(to image: Int) -> Job? {
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

    private func execute(_ job: Job) async -> Issue {
        let settings = job.settings
        let loraEntries = settings.loras.map { LoRAEntry($0.path, strength: $0.strength) }
        do {
            try library.checkReachable()
            // The references are decoded off the main thread (3072 px at most), in their order; the
            // engine then brings each to its size: the full-size photos do not live during the render.
            let sources = job.references
            let references = try await Task.detached(priority: .userInitiated) {
                try sources.map { try $0.imageRGB() }
            }.value
            // The starting image (img2img), decoded off the main thread too, as `siliconed-dev render --image` reads it; the
            // request fills and crops it to the format at once (`Request.image`).
            let startSource = settings.strength == nil ? nil : job.start
            let start = try await Task.detached(priority: .userInitiated) { try startSource?.imageRGB() }.value
            var request = Request(settings.modelPrompt ?? settings.prompt, width: settings.format.width,
                                  height: settings.format.height, seed: job.seeds[0],
                                  steps: settings.steps == job.defaultSteps ? nil : settings.steps,
                                  loras: loraEntries, image: start, strength: settings.strength ?? Strength.defaultValue,
                                  previews: job.previews)
            request.references = references
            request.detail = settings.detail
            request.solver = settings.solver
            request.variations = settings.variations

            var progress = Engine.Progress()
            // A train: its jobs' seeds, one image each, in order.
            let together = train.count > 1
            // When the denoising under way began: its box says the seconds from there to the decoding, its DiT's plan
            // included.
            var denoisingBegan: Date?
            // The engine renders in its own process (`EngineWorker`): the model is read there, under B.
            for try await item in EngineWorker.shared.render(request, seeds: together ? train.map { $0.seeds[0] } : job.seeds,
                                                            model: settings.identifier, library: library) {
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
                    // The denoising ends where the decoding begins.
                    if e == .denoising { denoisingBegan = Date() }
                    if e == .decoding, stage == .denoising, let began = denoisingBegan {
                        stageSeconds[.denoising] = Date().timeIntervalSince(began)
                    }
                    stageStates[e] = .active
                    stage = e
                    batchImage = image
                    if let rider = turn(to: image) { signal(.stage(rider, e, image: together ? 0 : image)) }
                case let .text(_, tokenizer, encoder, _):
                    stageSeconds[.text] = tokenizer + encoder
                case .encoding(let s):
                    stageSeconds[.image] = s
                case .decoding(let s):
                    stageSeconds[.decoding] = s
                case let .step(image, index, total, _, seconds, _, _):
                    batchImage = image; currentStep = index; totalSteps = total
                    if let rider = turn(to: image) {
                        signal(.step(rider, image: together ? 0 : image, index: index, total: total, seconds: seconds))
                    }
                case .preview(let a):
                    // The image's own job: in a train, the cell it is of, whichever the running one is now.
                    let of = together && train.indices.contains(a.image) ? train[a.image] : job
                    prerenders[of.ordinal] = a.cgImage()
                case .image(let image, let renderResult):
                    stageStates[.decoding] = .done
                    guard let job = turn(to: image) else { break }
                    // The PNG and the thumbnail are made off the main thread; the engine has already
                    // moved on to the next image of the batch.
                    // A job `silicontrol add "@name"` translated: the PNG keeps the prompt typed, the model's beside it.
                    // Any other job (the rack's, which reads what it holds): the engine's metadata, untouched.
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
                                        settings: settings,
                                        metadata: named ?? renderResult.metadata, references: job.references,
                                        start: start == nil ? nil : job.start, owner: job.owner)
                    admit(entry, image: finished.image, shows: followsRender)
                    prerenders[job.ordinal] = nil
                    sessionImages += 1
                    sessionSeconds += renderResult.timings.total
                    jobImages.append(entry.ordinal)
                    signal(.image(job, entry))
                    // In a train, a job ends with its image: the call goes on for the others.
                    if together {
                        delivered.insert(job.id)
                        complete(job, .finished, images: [entry.ordinal], warnings: journal)
                    }
                case let .unusable(image, lost):
                    // In a train, a job whose image is lost ends with it: the call goes on for the others. Alone, its
                    // call ends in `nonFiniteImage`, below.
                    guard together, let job = turn(to: image) else { break }
                    delivered.insert(job.id)
                    let p = Problem(EngineError.nonFiniteImage(stage: lost))
                    problem = p
                    complete(job, .failed(p), warnings: journal)
                case .warning(let message):
                    journal.append(message)
                default:
                    break
                }
            }
            // What this render teaches about speed: its own pace (the first render on a Mac finds the Neural Engine's share
            // in flight, its pace within 3 % of the next, measured 20/11 on the mini).
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

    /// **The render under way waits for Core AI to compile** a function its cache does not hold (`Engine.Held.compiling`):
    /// a model's first render, a new size, another share of the Neural Engine. What it waits for then is paid once, and no
    /// countdown says it.
    var compilingCoreAI: Bool {
        inProgress && measurements.compiling()
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
    func canVary(_ entry: Entry) -> Bool { varyRefusal(entry) == nil }

    /// **Why « Variations » cannot take this image**, said on the greyed button; `nil` when it can.
    func varyRefusal(_ entry: Entry) -> String? {
        if let installBlocker { return installBlocker }
        if entry.enlarged != nil { return String(localized: "An enlargement has no seed of its own: vary the image it was made from") }
        if entry.cropped != nil { return String(localized: "A crop has no seed of its own: vary the image it was cut from") }
        if !readySet.contains(entry.settings.identifier) {
            return String(localized: "\(entry.settings.modelName) isn’t installed: install it to vary this image")
        }
        return nil
    }

    /// **« Variations »**: cousins of a history image — exactly its settings (model, prompt, LoRA,
    /// size, steps, seed, edit images), each with a variation seed drawn at random at `strength`
    /// (`Variation.Amount`, its strength for the model's family), ordinary jobs whose images join the history.
    /// **As many as the rack's « Seeds »** (`batch`, 1 by default): at Qwen's ~200 s an edit, a fixed
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
        let plan = (try? Model.named(entry.settings.identifier, in: library)).map {
            Self.plan($0.chain.denoising, steps: entry.settings.steps, format: entry.settings.format, strength: entry.settings.strength)
        }
        let jobs = (0..<(count ?? batch)).map { _ in
            var settings = entry.settings
            settings.variations.append(Variation(seed: UInt64.random(in: 0...UInt64(UInt32.max)), strength: strength))
            var job = Job(ordinal: nextJobNumber(), settings: settings, seeds: [entry.seed], references: entry.references,
                          previews: previews, defaultSteps: card?.defaultSteps, plan: plan)
            job.start = entry.start
            return job
        }
        enqueue(jobs)
        return jobs
    }

    /// Redo the chosen image: same settings, same seed — and its size: an edit's would otherwise be derived again from image 1
    /// (`outputFormat`), and one typed on the command line (`add … 1024 --ref …`) came back at image 1's proportions.
    func redo(_ entry: Entry) {
        restoreSettings(entry)
        render(imposedSeed: entry.seed, imposedFormat: entry.settings.format)
    }

    func restoreSettings(_ entry: Entry) {
        restoreSettings(entry.settings)
        // An image's settings are one render's: no knob bracketed.
        brackets = []
        seriesPlaced = [nil, nil]
        // Fixed, or the next render draws a new seed and the settings reused are not the image's.
        seed = entry.seed
        fixedSeed = true
        // After the seed: setting it drops a variation (`seed.didSet`).
        variations = entry.settings.variations
        // The images it edited, or none: image 1 is the image's own, never the one the lab held before — otherwise an edit
        // comes back without what it edits, and an image made from noise comes back editing what was left there.
        setEditImages(entry.editImages)
        // The image it started from, with its strength — or none: « Redo » renders the same image, not an img2img of another.
        if let s = entry.settings.strength, let start = entry.start, startsFromImage {
            setStart(start)
            startStrength = s
        } else {
            setStart(nil)
        }
    }

    /// **« Reuse This Grid »**, on a series' tile: the form takes what the cells share and the grid's axes, as the window
    /// brackets them — the prompt as typed, its groups the grid's.
    func restoreSettings(_ run: GridRun) {
        restoreSettings(run.settings)
        prompt = run.settings.prompt
        fixedSeed = true
        seed = run.grid.base.seed
        let left = setSeries(from: run.grid)
        if !left.isEmpty { flash(String(localized: "The window can’t vary models, steps or edit images: those axes were left out.")) }
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
              let data = try? Data(contentsOf: url, options: .mappedIfSafe) else { return false }
        return restoreSettings(fromPNG: data, file: url.lastPathComponent)
    }

    /// Whether these bytes are a PNG Siliconed wrote: what offers « Reuse These Settings » on an image of the Prompt Lab.
    static func isRecipe(_ data: Data) -> Bool { Engine.Render.Recipe(metadata: PNG.text(data)) != nil }

    /// The same, from the bytes — an image of the Prompt Lab; `file` names it in the notes.
    func restoreSettings(fromPNG data: Data, file: String) -> Bool {
        guard let recipe = Engine.Render.Recipe(metadata: PNG.text(data)) else { return false }
        var notes: [String] = []

        // The PNG names a family, not a card (a fine-tune writes its family's name): the form's
        // model if it is of that family, otherwise the publisher's card, otherwise an imported one.
        let ready = visibleCards.filter { $0.family.rawValue == recipe.model && readySet.contains($0.id) }
        let target = ready.first { $0.id == identifier } ?? ready.first { !$0.isImported } ?? ready.first
        if target == nil, let model = recipe.model {
            // Steps and LoRAs belong to that model: the form's stay.
            let name = cards.first { $0.family.rawValue == model }?.name ?? model
            notes.append(String(localized: "\(name) isn’t installed: its steps and LoRAs are left out."))
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
                notes.append(String(localized: "A negative LoRA strength comes back as 0: turn on Settings › Advanced › Negative LoRA Strengths."))
            }
        }

        // The size if the model it goes to renders it (`Format.check(…, family:)`), the form's otherwise.
        var restoredFormat = format
        let family = (target ?? card)?.family ?? .default   // no model at all: the default family's rules
        if let w = recipe.width, let h = recipe.height, (try? Format.check(width: w, height: h, family: family)) != nil {
            restoredFormat = RecommendedFormat(w, h)
        }
        if recipe.sourceImageMissing {
            notes.append(String(localized: "The image it started from isn’t in the file (img2img)."))
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
            notes.append(String(localized: "Made with the \(recipe.solver.choice) solver, offered when More Solvers is on: Euler will be used."))
        }
        if recipe.detail != .normal, !offersDetail {
            notes.append(String(localized: "\(model?.name ?? identifier) doesn’t offer Detail: Normal will be used."))
        }
        if let s = recipe.seed { seed = s; fixedSeed = true }
        // An img2img's strength comes back to the rack; its image is not in the file (said above): the one carried, if any, stays.
        if let s = recipe.strength { startStrength = StartImage.snapped(s) }
        // A variation is its seed's: without the seed, there is nothing to vary.
        variations = recipe.seed == nil ? [] : recipe.variations
        flash(String(localized: "Settings of \(file) restored"))
        if !notes.isEmpty {
            problem = Problem(String(localized: "Not everything came back from \(file)."),
                              suggestion: notes.joined(separator: "\n"))
        }
        return true
    }

    /// The window closes: nothing must keep occupying the GPU with no view to show it.
    func unmount() {
        toastTask?.cancel()
        clearQueue()
        task?.cancel()
    }
}
