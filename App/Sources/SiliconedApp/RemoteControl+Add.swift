// `add` and `grid`: the order read from the command line (`AddOrder`), checked, queued.

import AppKit
import Foundation
import Measurement
import Observation
import Siliconed
import SilicontrolHelp

extension RemoteControl {
    /// **A command's flags, read as every command reads them** (`Flags.scan`): one it does not take is refused by name.
    /// `text`: the command takes a free text (a prompt), which may begin with `--` — the refusal then says where such a text goes.
    static func flags(_ arguments: [String], valued: Set<String> = [], switches: Set<String> = [], text: Bool = false) throws -> Flags {
        switch Flags.scan(arguments, valued: valued, switches: switches) {
        case .success(let flags): return flags
        case .failure(.unknownFlag(let a)): throw unknownFlag(a, text: text)
        case .failure(.flagWithoutValue(let a)): throw Refusal("usage", String(localized: "\(a) expects a value"))
        }
    }

    /// A flag the command does not take. A hint about the lone `--` only where a text may begin with `--`: elsewhere the
    /// command's page says what it takes (`handle`).
    static func unknownFlag(_ a: String, text: Bool) -> Refusal {
        Refusal("usage", String(localized: "unknown flag: \(a)"),
                help: text ? String(localized: "a text that begins with -- goes after a lone --") : nil)
    }

    /// `add` and `grid` read (`AddOrder`, the library's), their mistakes said in the app's words.
    func parse(_ arguments: [String]) throws -> AddOrder {
        switch AddOrder.parse(arguments) {
        case .success(let order): return order
        case .failure(let failure): throw Self.refusal(failure)
        }
    }

    static func refusal(_ failure: AddOrder.Failure) -> Refusal {
        switch failure {
        case .unknownFlag(let a): return unknownFlag(a, text: true)
        case .flagWithoutValue(let a): return Refusal("usage", String(localized: "\(a) expects a value"))
        case .badValue(.steps, let v):
            return Refusal("usage", String(localized: "--steps expects an integer from 1 to \(Request.maximumSteps), not \"\(v)\""))
        case .badValue(.detail, let v): return Refusal("usage", String(localized: "--detail expects normal, more or most, not \"\(v)\""))
        case .badValue(.solver, let v):
            return Refusal("usage", String(localized: "--solver expects euler, euler-ancestral, dpmpp-2m or unipc, not \"\(v)\""))
        case .badValue(.batch, let v): return Refusal("usage", String(localized: "--batch expects an integer from 1 to 8, not \"\(v)\""))
        case .badValue(.lora, let v): return Refusal("usage", String(localized: "--lora expects a finite strength, not \"\(v)\""))
        case .badValue(.strength, let v):
            return Refusal("usage", String(localized: "--strength expects a number above 0 and at most 1 (0.65 or 65%), not \"\(v)\""))
        case .promptMissing:
            return Refusal("usage", String(localized: "the prompt is missing"),
                           help: "silicontrol add [--model id] \"<prompt>\" [512|1024|WxH|P:Q@1K|2K] [seed] — silicontrol help add")
        case .unknownRatio(let r):
            return Refusal("format", String(localized: "no such proportion: \"\(r)\""),
                           help: String(localized: "proportions: \(SizeScales.ratioList) — silicontrol help add"))
        case .unknownResolution(let r):
            return Refusal("format", String(localized: "no such resolution: \"\(r)\""), help: String(localized: "1K or 2K, e.g. 3:4@2K"))
        case .unreadableScale(let p):
            return Refusal("format", String(localized: "unreadable size: \"\(p)\""), help: String(localized: "proportion@resolution, e.g. 3:4@2K"))
        case .unreadableFormat(let p):
            return Refusal("format", String(localized: "unreadable format: \"\(p)\""), help: String(localized: "width x height, e.g. 832x1216"))
        case .extraArgument(let p):
            return Refusal("usage", String(localized: "extra argument: \"\(p)\""),
                           help: String(localized: "the prompt is ONE argument: put it in quotes — silicontrol help add"))
        }
    }

    func absolute(_ path: String, _ cwd: String) -> String {
        URL(fileURLWithPath: (path as NSString).expandingTildeInPath, relativeTo: URL(fileURLWithPath: cwd, isDirectory: true))
            .standardizedFileURL.path
    }

    /// **The starting image `add` and `grid` read** (`--image`, img2img), on this Mac or on the one the window computes on: a
    /// model whose chain starts from one, never beside `--ref` — one or the other. With no format typed, the model's
    /// recommended format nearest its proportions, as the rack takes one.
    func startRequest(_ d: inout AddOrder, startsFromImage: Bool, edits: Bool, name: String, formats: [RecommendedFormat],
                      cwd: String) throws -> ReferenceImage? {
        if d.strength != nil, d.image == nil {
            throw Refusal("usage", String(localized: "--strength goes with --image"), help: "silicontrol help add")
        }
        guard let word = d.image else { return nil }
        guard startsFromImage else {
            throw Refusal("img2img", String(localized: "\(name) doesn’t start from an image (--image)"),
                          help: edits ? String(localized: "--ref <image> to edit it (help edit)")
                                      : String(localized: "models that do have \"start_image\": true in \"silicontrol models\""))
        }
        guard d.references.isEmpty else {
            throw Refusal("usage", String(localized: "--image and --ref don’t go together"), help: "silicontrol help add")
        }
        let start = try startImage(word, cwd: cwd)
        if d.format == nil,
           let f = StartImage.recommendedFormat(width: start.width, height: start.height, among: formats, current: app.format) {
            d.format = (f.width, f.height)
        }
        return start
    }

    /// The strength a start renders at — `--strength`, or its family's —; one that leaves no step is refused, as the rack waits.
    func startStrength(_ d: AddOrder, start: ReferenceImage?, family: Family?, steps: Int) throws -> Double? {
        guard start != nil else { return nil }
        let strength = StartImage.snapped(d.strength ?? StartImage.defaultStrength(for: family))
        if Strength.startStep(steps: steps, strength: strength) >= steps {
            throw Refusal("img2img", String(localized: "At this strength the image would come back unchanged: raise it."),
                          help: "--strength 0.65")
        }
        return strength
    }

    /// `add`, and `grid` (`grid: true`): the same request, checked the same way; a grid queues one
    /// job per cell (`AppState.enqueueGrid`) instead of one per alternative.
    func add(_ arguments: [String], cwd: String, _ c: Connection, grid: Bool = false) throws {
        var d = try parse(arguments)
        // One big job at a time on 16 GB: no render while an installation or an import runs.
        if let reason = app.libraryBlocker { throw Refusal("library_absent", reason, help: String(localized: "connect the disk, then retry")) }
        if let reason = app.installBlocker { throw Refusal("busy", reason, help: String(localized: "retry once the installation is done (silicontrol models)")) }
        try checkUsage(d, grid: grid)
        let b = app.library
        let model: Model
        do { model = try Model.named(d.model, in: b) } catch {
            let detail = Problem(error).title, readySet = app.visibleReady.map(\.id).joined(separator: ", ")
            throw Refusal("model", String(localized: "model \"\(d.model)\" unavailable: \(detail)"),
                        help: String(localized: "ready: \(readySet) — silicontrol models"))
        }
        let card = b.cards().first { $0.id == model.identifier }
        try checkSolver(d.solver, family: card?.family, name: card?.name ?? model.name)
        try checkDetail(d.detail, accepted: model.chain.denoising.acceptsDetail, name: card?.name ?? model.name)
        guard let prompt = d.prompt, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Refusal("usage", String(localized: "empty prompt"))
        }
        // `{a|b}`: one job per expanded prompt, parsed like the app's field — before anything else
        // is read, so that a typo costs nothing.
        let alternatives: PromptAlternatives
        do { alternatives = try PromptAlternatives(prompt) } catch {
            let p = Problem(error)
            throw Refusal("prompt", p.title, help: p.suggestion)
        }

        let start = try startRequest(&d, startsFromImage: model.chain.acceptsStartImage,
                                     edits: model.chain.denoising.maxReferences > 0, name: card?.name ?? model.name,
                                     formats: card?.formats ?? ModelCard.recommendedFormats, cwd: cwd)

        // The references, in order: files, or images of the history (`i7`). The first is image 1,
        // the one edited: with no format typed, the output takes the denoiser's (`editFormat`) —
        // the app's rule and the CLI's, so that a command and a click give the same image.
        var references: [ReferenceImage] = []
        if !d.references.isEmpty {
            let max = model.chain.denoising.maxReferences
            guard max > 0 else {
                let name = model.name
                throw Refusal("edit", String(localized: "\(name) does no editing by reference (--ref)"),
                            help: String(localized: "models that do have \"edit\": true in \"silicontrol models\""))
            }
            guard d.references.count <= max else {
                let n = d.references.count, name = model.name
                throw Refusal("edit", String(localized: "\(n) references: \(name) reads \(max) at most"),
                            help: String(localized: "\"max_references\" in \"silicontrol models\""))
            }
            references = try d.references.map { try referenceImage($0, cwd: cwd) }
            if d.format == nil, let first = references.first {
                let f = model.chain.denoising.editFormat(referenceWidth: first.width, referenceHeight: first.height)
                let capped = AppState.cappedForEditing(RecommendedFormat(f.width, f.height))
                d.format = (capped.width, capped.height)
            }
            // The app edits up to `AppState.editingSurface` of area; beyond is the CLI's.
            if let (w, h) = d.format, w * h > AppState.editingSurface {
                throw Refusal("format", AppState.editSurfaceRefusal(width: w, height: h),
                            help: String(localized: "leave the format out (the output follows image 1)"))
            }
        }

        let (width, height) = d.format ?? (1024, 1024)
        do { try Format.check(width: width, height: height, family: model.card.family) } catch {
            let p = Problem(error)
            throw Refusal("format", p.title, help: p.suggestion)
        }
        try checkLoRAs(d)
        let stack = try d.loras.map { entry -> LoRASlot in
            guard let f = b.lora(entry.name, for: model.identifier) ?? b.lora(absolute(entry.name, cwd), for: model.identifier) else {
                let compatible = b.loras(for: model.identifier).map { Self.shortName($0.path) }
                let name = entry.name, id = model.identifier, list = compatible.joined(separator: ", ")
                throw Refusal("lora", String(localized: "LoRA \"\(name)\" not found for \(id)"),
                            help: compatible.isEmpty ? String(localized: "no LoRA installed for this model")
                                                      : String(localized: "compatible: \(list)"))
            }
            return LoRASlot(path: f.path, strength: entry.strength)
        }
        guard grid || app.file.count + alternatives.count <= AppState.maxQueueLength else {
            let max = AppState.maxQueueLength
            if alternatives.count > 1 {
                let n = alternatives.count, room = max - app.file.count
                throw Refusal("queue_full", String(localized: "\(n) jobs for the alternatives: the queue has room for \(room) (\(max) jobs)"),
                            help: "silicontrol wait, then retry")
            }
            throw Refusal("queue_full", String(localized: "the queue is full (\(max) jobs)"),
                        help: "silicontrol wait, then retry")
        }
        var folder: String?
        if let raw = d.folder {
            folder = absolute(raw, cwd)
            try createFolder(folder!)
        }

        let defaultSteps = card?.defaultSteps ?? model.chain.denoising.defaultSteps
        let steps = d.steps ?? defaultSteps
        // A step count the denoiser has no schedule for would fail in the queue: refused here, like everything
        // else an accepted request cannot fail on.
        try checkSteps(steps, scheduled: model.chain.denoising.scheduledSteps, defaultSteps: defaultSteps)
        let f = model.chain.denoising.space.factor
        let strength = try startStrength(d, start: start, family: card?.family, steps: steps)
        let first = strength.map { min(steps, Strength.startStep(steps: steps, strength: $0)) } ?? 0
        let plan = model.chain.denoising.plan(height: height / f, width: width / f, steps: steps, start: first)
        // One seed for the whole series, drawn once: the alternatives are compared at the same seeds.
        let firstNumber = d.seed ?? Request.drawnSeed()
        if grid {
            if alternatives.variants.contains(where: { app.rewrite($0.prompt, model: model) != nil }) {
                throw Refusal("subject", String(localized: "a grid does not cite named subjects yet"),
                              help: "silicontrol add")
            }
            let format = RecommendedFormat(width, height)
            try addGrid(d, prompt: prompt, model: model, card: card, format: format, steps: steps, stack: stack,
                        references: references, start: start, strength: strength,
                        seed: firstNumber, defaultSteps: defaultSteps, folder: folder, cwd: cwd, c)
            return
        }
        let variants = alternatives.variants
        // Named subjects (`Subjects.swift`): each expanded prompt rewritten, its
        // subjects' pictures after the --ref images — refused whole if one expansion is.
        for v in variants {
            if case .failure(let f)? = app.rewrite(v.prompt, model: model, editImages: references.count) {
                let p = Problem(f)
                throw Refusal("subject", p.title, help: p.suggestion ?? "silicontrol subjects")
            }
        }
        let jobs = variants.map { v in
            var settings = RenderSettings(identifier: model.identifier, modelName: card?.name ?? model.name,
                                          prompt: v.prompt, format: RecommendedFormat(width, height), steps: steps,
                                          loras: stack, references: references.count, detail: d.detail,
                                          solver: d.solver)
            settings.strength = strength
            let used = app.namedJob(v.prompt, settings: &settings, references: references, model: model)
            var job = Job(ordinal: app.nextJobNumber(), settings: settings,
                          seeds: (0..<d.batch).map { firstNumber &+ UInt64($0) },
                          references: used, previews: d.previews ?? app.previews,
                          defaultSteps: defaultSteps, plan: plan)
            job.start = start
            return job
        }
        // The canvas follows the render, as after « Generate » — a series shown before keeps it otherwise.
        app.followsRender = true
        app.enqueue(jobs)
        app.takeRequest(jobs[0].settings, prompt: prompt, references: references, start: start, strength: strength,
                        seed: firstNumber, fixed: d.seed != nil, batch: d.batch)

        let added = zip(jobs, variants).map { job, v -> [String: Any] in
            var o = queued(job)
            if alternatives.hasAlternatives { o["choices"] = v.choices }
            return o
        }
        if d.wait {
            for var o in added { o["type"] = "added"; emit(c, o) }
            followToEnd(jobs.map(\.ordinal), folder: folder, c)
        } else if jobs.count == 1 {
            // One job: the answer of a prompt without alternatives, unchanged.
            var o = added[0]
            o["ok"] = true
            finish(c, o)
        } else {
            finish(c, ["ok": true, "jobs": added, "estimate_s": estimate(jobs)])
        }
    }

    /// A job queued on this Mac, as `add` and `grid` answer it: its place (0: rendering, else from 1) and its estimate.
    func queued(_ job: Job) -> [String: Any] {
        var o = json(job)
        o["position"] = app.currentJob?.ordinal == job.ordinal
            ? 0 : (app.file.firstIndex { $0.ordinal == job.ordinal }.map { $0 + 1 } ?? 0)
        o["estimate_s"] = app.estimation(job).map { Response.number($0, 1) } ?? NSNull()
        return o
    }

    /// The jobs' estimate, all together; `null` when one was never measured.
    func estimate(_ jobs: [Job]) -> Any { app.estimation(of: jobs).map { Response.number($0, 1) } ?? NSNull() }
}
