// `add` and `grid`: the order read from the command line (`RenderOrder`), checked, queued.

import AppKit
import Foundation
import Measurement
import Observation
import Siliconed
import SilicontrolHelp

extension RemoteControl {
    /// **A command's flags, read as every command reads them** (`Flags.scan`): one it does not take is refused by name, and
    /// a text that begins with `--` goes after a lone `--`.
    static func flags(_ arguments: [String], valued: Set<String>, switches: Set<String> = []) throws -> Flags {
        switch Flags.scan(arguments, valued: valued, switches: switches) {
        case .success(let flags): return flags
        case .failure(.unknownFlag(let a)):
            throw Refusal("usage", String(localized: "unknown flag: \(a)"),
                          help: String(localized: "a text that begins with -- goes after a lone --"))
        case .failure(.flagWithoutValue(let a)): throw Refusal("usage", String(localized: "\(a) expects a value"))
        }
    }

    /// The flags of `add` and `grid`, by kind: those that take a value, and the switches.
    static let addValued: Set<String> = ["--model", "--steps", "--detail", "--solver", "--batch", "--x", "--y", "--lora", "--ref",
                                         "--out", "--image", "--strength"]
    static let addSwitches: Set<String> = ["--preview", "--no-preview", "--wait", "--no-sheet"]

    /// The arguments of the developer's render command, plus `--wait` and `--out`.
    struct RenderOrder {
        var model = ModelCard.of(.default).id
        var prompt: String?
        var format: (Int, Int)?
        var seed: UInt64?
        var batch = 1
        var steps: Int?
        /// `--detail more|most`: the rack's Detail choice (`Request.detail`).
        var detail = Detail.normal
        /// `--solver NAME`: the rack's Solver (`Request.solver`), More Solvers and `AppState.solverFamilies` only.
        var solver = Solver.euler
        var loras: [(name: String, strength: Double)] = []
        /// The `--ref`, in their order: the first is image 1, the one edited.
        var references: [String] = []
        /// `--image IMAGE|i7`: the starting image of an img2img (a model with "start_image": true); `--strength`: its strength,
        /// diffusers' (the family's, `StartImage.defaultStrength`, when left out).
        var image: String?
        var strength: Double?
        /// `nil`: like the DiT box's « Preview every step » switch.
        var previews: Bool?
        var wait = false
        /// `grid --no-sheet`: a series as « Generate » renders it — the cells, the sheet only on `sheet g2`.
        var sheet = true
        var folder: String?
        /// `grid`: the axes as typed (`RenderGrid.axis`), and whether `--batch` was given (refused).
        var x: String?, y: String?
        var batchGiven = false
    }

    func parse(_ arguments: [String]) throws -> RenderOrder {
        var d = RenderOrder()
        let flags = try Self.flags(arguments, valued: Self.addValued, switches: Self.addSwitches)
        // A flag given twice: its last value, except those that add up (`--lora`, `--ref`).
        if let model = flags.value("--model") { d.model = model }
        if let v = flags.value("--steps") {
            guard let n = Int(v), (1...Request.maximumSteps).contains(n) else {
                throw Refusal("usage", String(localized: "--steps expects an integer from 1 to \(Request.maximumSteps), not \"\(v)\""))
            }
            d.steps = n
        }
        if let v = flags.value("--detail") {
            guard let level = Detail(rawValue: v.lowercased()) else {
                throw Refusal("usage", String(localized: "--detail expects normal, more or most, not \"\(v)\""))
            }
            d.detail = level
        }
        if let v = flags.value("--solver") {
            guard let rule = Solver(rawValue: v.lowercased()) else {
                throw Refusal("usage", String(localized: "--solver expects euler, euler-ancestral, dpmpp-2m or unipc, not \"\(v)\""))
            }
            d.solver = rule
        }
        if let v = flags.value("--batch") {
            guard let n = Int(v), (1...8).contains(n) else { throw Refusal("usage", String(localized: "--batch expects an integer from 1 to 8, not \"\(v)\"")) }
            d.batch = n
            d.batchGiven = true
        }
        d.x = flags.value("--x")
        d.y = flags.value("--y")
        for v in flags.values["--lora"] ?? [] {
            // `name:strength` — the last colon separates the strength (a path has none).
            if let colonIndex = v.lastIndex(of: ":"), let f = Double(v[v.index(after: colonIndex)...].replacingOccurrences(of: ",", with: ".")) {
                // `Double` reads « nan » and « inf »: a strength that would make every weight NaN.
                guard f.isFinite else { throw Refusal("usage", String(localized: "--lora expects a finite strength, not \"\(v)\"")) }
                d.loras.append((String(v[..<colonIndex]), f))
            } else {
                d.loras.append((v, 1))
            }
        }
        d.references = flags.values["--ref"] ?? []
        d.folder = flags.value("--out")
        if let last = flags.last(of: ["--preview", "--no-preview"]) { d.previews = last == "--preview" }
        if flags.has("--wait") { d.wait = true }
        if flags.has("--no-sheet") { d.sheet = false }
        d.image = flags.value("--image")
        if let v = flags.value("--strength") {
            guard let s = StartImage.strength(v) else {
                throw Refusal("usage", String(localized: "--strength expects a number above 0 and at most 1 (0.65 or 65%), not \"\(v)\""))
            }
            d.strength = s
        }
        // A prompt that begins with `--` is written after a lone `--`: everything there is a word.
        let positional = flags.allWords
        guard let prompt = positional.first else {
            throw Refusal("usage", String(localized: "the prompt is missing"), help: "silicontrol add [--model id] \"<prompt>\" [512|1024|WxH|P:Q@1K|2K] [seed] — silicontrol help add")
        }
        d.prompt = prompt
        for p in positional.dropFirst() {
            if d.format == nil, SizeScales.looksLikeScale(p) {
                // `3:4@2K`: the pixels of the rack's two-scale Size (`SizeScales`), whether or not that Beta option is on.
                // The model's own limits are judged below, with `WxH`'s.
                do { let c = try SizeScales.parse(p); d.format = (c.width, c.height) } catch {
                    switch error {
                    case .unknownRatio(let r):
                        throw Refusal("format", String(localized: "no such proportion: \"\(r)\""),
                                      help: String(localized: "proportions: \(SizeScales.ratioList) — silicontrol help add"))
                    case .unknownResolution(let r):
                        throw Refusal("format", String(localized: "no such resolution: \"\(r)\""), help: String(localized: "1K or 2K, e.g. 3:4@2K"))
                    case .unreadable:
                        throw Refusal("format", String(localized: "unreadable size: \"\(p)\""), help: String(localized: "proportion@resolution, e.g. 3:4@2K"))
                    }
                }
            } else if d.format == nil, p.lowercased().contains("x") || p.contains("×") {
                guard let f = Format.parse(p) else { throw Refusal("format", String(localized: "unreadable format: \"\(p)\""), help: String(localized: "width x height, e.g. 832x1216")) }
                d.format = (f.width, f.height)
            } else if d.format == nil, p == "512" || p == "1024" {
                d.format = (Int(p)!, Int(p)!)
            } else if d.seed == nil, let g = UInt64(p) {
                d.seed = g
            } else {
                throw Refusal("usage", String(localized: "extra argument: \"\(p)\""),
                            help: String(localized: "the prompt is ONE argument: put it in quotes — silicontrol help add"))
            }
        }
        return d
    }

    func absolute(_ path: String, _ cwd: String) -> String {
        URL(fileURLWithPath: (path as NSString).expandingTildeInPath, relativeTo: URL(fileURLWithPath: cwd, isDirectory: true))
            .standardizedFileURL.path
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

        // The starting image (img2img): a model whose chain starts from one, never with references — one or the other.
        var start: ReferenceImage?
        if d.strength != nil, d.image == nil {
            throw Refusal("usage", String(localized: "--strength goes with --image"), help: "silicontrol help add")
        }
        if let word = d.image {
            guard model.chain.acceptsStartImage else {
                let name = card?.name ?? model.name
                throw Refusal("img2img", String(localized: "\(name) doesn’t start from an image (--image)"),
                              help: model.chain.denoising.maxReferences > 0
                                  ? String(localized: "--ref <image> to edit it (help edit)")
                                  : String(localized: "models that do have \"start_image\": true in \"silicontrol models\""))
            }
            guard d.references.isEmpty else {
                throw Refusal("usage", String(localized: "--image and --ref don’t go together"), help: "silicontrol help add")
            }
            start = try startImage(word, cwd: cwd)
            if d.format == nil, let start {
                let fitting = card?.formats ?? ModelCard.recommendedFormats
                if let f = StartImage.recommendedFormat(width: start.width, height: start.height, among: fitting, current: app.format) {
                    d.format = (f.width, f.height)
                }
            }
        }

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
        let strength = start == nil ? nil : StartImage.snapped(d.strength ?? StartImage.defaultStrength(for: card?.family))
        let first = strength.map { min(steps, Strength.startStep(steps: steps, strength: $0)) } ?? 0
        if strength != nil, first >= steps {
            throw Refusal("img2img", String(localized: "At this strength the image would come back unchanged: raise it."),
                          help: "--strength 0.65")
        }
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
                        seed: firstNumber, fixed: d.seed != nil)

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
