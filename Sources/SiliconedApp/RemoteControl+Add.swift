// `add` and its argument parsing, single images and XY grids: the command that
// builds a `RenderOrder` from the command line. Split from `RemoteControl.swift`.

import AppKit
import Foundation
import Observation
import Siliconed
import SilicontrolHelp

extension RemoteControl {
    func parse(_ arguments: [String]) throws -> RenderOrder {
        var d = RenderOrder()
        var positional: [String] = []
        var i = 0
        func value(_ flag: String) throws -> String {
            guard i + 1 < arguments.count else { throw Refusal("usage", String(localized: "\(flag) expects a value")) }
            i += 1
            return arguments[i]
        }
        while i < arguments.count {
            let a = arguments[i]
            switch a {
            case "--model": d.model = try value(a)
            case "--steps":
                let v = try value(a)
                guard let n = Int(v), (1...Request.maximumSteps).contains(n) else {
                    throw Refusal("usage", String(localized: "--steps expects an integer from 1 to \(Request.maximumSteps), not \"\(v)\""))
                }
                d.steps = n
            case "--detail":
                let v = try value(a)
                guard let level = Detail(rawValue: v.lowercased()) else {
                    throw Refusal("usage", String(localized: "--detail expects normal, more or most, not \"\(v)\""))
                }
                d.detail = level
            case "--solver":
                let v = try value(a)
                guard let rule = Solver(rawValue: v.lowercased()) else {
                    throw Refusal("usage", String(localized: "--solver expects euler, euler-ancestral, dpmpp-2m or unipc, not \"\(v)\""))
                }
                d.solver = rule
            case "--batch":
                let v = try value(a)
                guard let n = Int(v), (1...8).contains(n) else { throw Refusal("usage", String(localized: "--batch expects an integer from 1 to 8, not \"\(v)\"")) }
                d.batch = n
                d.batchGiven = true
            case "--x": d.x = try value(a)
            case "--y": d.y = try value(a)
            case "--lora":
                let v = try value(a)
                // `name:strength` — the last colon separates the strength (a path has none).
                if let colonIndex = v.lastIndex(of: ":"), let f = Double(v[v.index(after: colonIndex)...].replacingOccurrences(of: ",", with: ".")) {
                    // `Double` reads « nan » and « inf »: a strength that would make every weight NaN.
                    guard f.isFinite else { throw Refusal("usage", String(localized: "--lora expects a finite strength, not \"\(v)\"")) }
                    d.loras.append((String(v[..<colonIndex]), f))
                } else {
                    d.loras.append((v, 1))
                }
            case "--ref": d.references.append(try value(a))
            case "--out": d.folder = try value(a)
            case "--ane": d.neuralEngine = true
            case "--no-ane": d.neuralEngine = false
            case "--preview": d.previews = true
            case "--no-preview": d.previews = false
            case "--wait": d.wait = true
            case "--sketch": d.sketch = true
            case "--image", "--strength":
                throw Refusal("img2img", String(localized: "the app does no img2img (--image, --strength)"),
                            help: String(localized: "--ref <image> for editing (a model with \"edit\": true)"))
            default:
                if a.hasPrefix("--") { throw Refusal("usage", String(localized: "unknown flag: \(a)")) }
                positional.append(a)
            }
            i += 1
        }
        guard let prompt = positional.first else {
            throw Refusal("usage", String(localized: "the prompt is missing"), help: "silicontrol add [--model id] \"<prompt>\" [512|1024|WxH] [seed] — silicontrol help add")
        }
        d.prompt = prompt
        for p in positional.dropFirst() {
            if d.format == nil, p.lowercased().contains("x") || p.contains("×") {
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
        if !grid, d.sketch {
            throw Refusal("usage", String(localized: "--sketch belongs to « silicontrol grid »"), help: "silicontrol help grid")
        }
        if !grid, d.x != nil || d.y != nil {
            throw Refusal("usage", String(localized: "--x and --y belong to « silicontrol grid »"), help: "silicontrol help grid")
        }
        if grid, d.x == nil {
            throw Refusal("usage", String(localized: "a grid needs --x (and optionally --y)"), help: "silicontrol help grid")
        }
        if grid, d.batchGiven {
            throw Refusal("usage", String(localized: "a grid's cell is one image: --batch does not apply"),
                          help: String(localized: "--x seeds=N gives N seeds — silicontrol help grid"))
        }
        let b = app.library
        let model: Model
        do { model = try Model.named(d.model, in: b) } catch {
            let detail = Problem(error).title, readySet = app.visibleReady.map(\.id).joined(separator: ", ")
            throw Refusal("model", String(localized: "model \"\(d.model)\" unavailable: \(detail)"),
                        help: String(localized: "ready: \(readySet) — silicontrol models"))
        }
        let card = b.cards().first { $0.id == model.identifier }
        // The app renders what it shows: the other families need Developer Mode (the developer's
        // command line renders them all).
        guard card.map({ app.isVisible($0.family) }) ?? true else {
            let visible = visibleCards(b).map(\.id).joined(separator: ", ")
            throw Refusal("model", String(localized: "model \"\(d.model)\" is shown only in Developer Mode"),
                        help: String(localized: "models: \(visible) — or turn on Developer Mode in Settings (⌘,)"))
        }
        // A solver other than Euler: the rack's, so Developer Mode, and the models in front only.
        if d.solver != .euler {
            guard app.developerMode else {
                throw Refusal("solver", String(localized: "--solver \(d.solver.rawValue) is offered only in Developer Mode"),
                              help: String(localized: "turn on Developer Mode in Settings (⌘,)"))
            }
            guard card.map({ AppState.solverFamilies.contains($0.family) }) == true else {
                throw Refusal("solver", String(localized: "\(card?.name ?? model.name) doesn't offer the \(d.solver.rawValue) solver"),
                              help: String(localized: "Euler only, or --model z-image"))
            }
        }
        // Detail: not on a model that offers none (Anima, Krea 2) — the rack hides it there.
        if d.detail != .normal, !model.chain.denoising.acceptsDetail {
            throw Refusal("detail", String(localized: "\(card?.name ?? model.name) doesn't offer Detail"),
                          help: String(localized: "leave out --detail"))
        }
        // The Neural Engine: only where this Mac's map turned it on (the rack shows no choice elsewhere).
        if d.neuralEngine == true, !app.neuralEngineOffered {
            throw Refusal("neural_engine", String(localized: "this Mac's capacity map leaves the Neural Engine off"),
                          help: String(localized: "leave out --ane"))
        }
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
            references = try d.references.map { r -> ReferenceImage in
                if let entry = image(r), let reference = app.fromHistory(entry) { return reference }
                let url = URL(fileURLWithPath: absolute(r, cwd))
                guard let vignette = NSImage(contentsOf: url), vignette.isValid,
                      let size = ReferenceImage.size(of: url) else {
                    let path = url.path
                    throw Refusal("ref", String(localized: "image unreadable or missing: \(path)"))
                }
                return ReferenceImage(source: .file(url), vignette: vignette, name: url.lastPathComponent,
                                      width: size.width, height: size.height)
            }
            if d.format == nil, let first = references.first {
                let f = model.chain.denoising.editFormat(referenceWidth: first.width, referenceHeight: first.height)
                let capped = AppState.cappedForEditing(RecommendedFormat(f.width, f.height))
                d.format = (capped.width, capped.height)
            }
            // The app edits up to 1024² of area (`AppState.editingSurface`); beyond is the CLI's.
            if let (w, h) = d.format, w * h > AppState.editingSurface {
                throw Refusal("format", String(localized: "the app edits up to 1024² of area, not \(String(w))x\(String(h))"),
                            help: String(localized: "leave the format out (the output follows image 1)"))
            }
        }

        let (width, height) = d.format ?? (1024, 1024)
        do { try Format.check(width: width, height: height, family: model.card.family) } catch {
            let p = Problem(error)
            throw Refusal("format", p.title, help: p.suggestion)
        }
        guard d.loras.count <= AppState.maxLoRA else {
            let n = d.loras.count, max = AppState.maxLoRA
            throw Refusal("lora", String(localized: "\(n) LoRAs: the app stacks \(max) at most"))
        }
        guard app.negativeLoRAStrengths || !d.loras.contains(where: { $0.strength < 0 }) else {
            throw Refusal("lora", String(localized: "negative LoRA strengths are off"),
                          help: String(localized: "turn on Experimental → Negative LoRA Strengths in Settings (⌘,)"))
        }
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
        // A step count the denoiser has no schedule for (Qwen-Image-2.1-Turbo: 8 alone) would
        // fail in the queue: refused here, like everything else an accepted request cannot fail on.
        if (try? model.chain.denoising.check(steps: steps, startImage: false)) == nil {
            let accepted = (1...50).filter { (try? model.chain.denoising.check(steps: $0, startImage: false)) != nil }
                .map(String.init).formatted(.list(type: .or))
            throw Refusal("steps", String(localized: "This model has no schedule for \(String(steps)) steps: \(accepted)."),
                        help: "--steps \(defaultSteps) (the model's \"default_steps\")")
        }
        let f = model.chain.denoising.space.factor
        let plan = model.chain.denoising.plan(height: height / f, width: width / f, steps: steps, start: 0,
                                                 withLoRA: !stack.isEmpty || d.detail != .normal)
        // One seed for the whole series, drawn once: the alternatives are compared at the same seeds.
        let firstNumber = d.seed ?? UInt64.random(in: 0...UInt64(UInt32.max))
        if grid {
            if alternatives.variants.contains(where: { app.rewrite($0.prompt, model: model) != nil }) {
                throw Refusal("subject", String(localized: "a grid does not cite named subjects yet"),
                              help: "silicontrol add")
            }
            let format = RecommendedFormat(width, height)
            try addGrid(d, prompt: prompt, model: model, card: card, format: format, steps: steps, stack: stack,
                        references: references, seed: firstNumber, defaultSteps: defaultSteps, folder: folder, cwd: cwd, c)
            return
        }
        let variants = alternatives.variants
        // Named subjects (experimental, `Subjects.swift`): each expanded prompt rewritten, its
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
            let used = app.namedJob(v.prompt, settings: &settings, references: references, model: model)
            var job = Job(ordinal: app.nextJobNumber(), settings: settings,
                          seeds: (0..<d.batch).map { firstNumber &+ UInt64($0) },
                          references: used, previews: d.previews ?? app.previews,
                          // Subjects' pictures make the plan unreduced (as `AppState.expectedPlan`); without
                          // them, the plan as it was.
                          defaultSteps: defaultSteps, plan: used.count > references.count ? plan.unreduced : plan)
            job.neuralEngine = d.neuralEngine
            return job
        }
        app.enqueue(jobs)
        if let on = d.neuralEngine { app.setNeuralEngine(on, for: model.identifier) }
        // The form takes what the job computes, as after a click: settings, edit images, seed — and
        // the prompt as typed, so that « Generate » would queue the same series.
        app.restoreSettings(jobs[0].settings)
        app.prompt = prompt
        app.references = references
        // A seed drawn here is a Random render, as after a click: the rack must not keep showing an old Fixed one.
        if let seed = d.seed { app.fixedSeed = true; app.seed = seed } else { app.fixedSeed = false }

        let added = zip(jobs, variants).map { job, v -> [String: Any] in
            var o = json(job)
            o["position"] = app.currentJob?.ordinal == job.ordinal
                ? 0 : (app.file.firstIndex { $0.ordinal == job.ordinal }.map { $0 + 1 } ?? 0)
            o["estimate_s"] = app.estimation(job).map { Response.number($0, 1) } ?? NSNull()
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
            let estimates = jobs.map { app.estimation($0) }
            let total: Any = estimates.contains { $0 == nil } ? NSNull()
                : Response.number(estimates.compactMap { $0 }.reduce(0, +), 1)
            finish(c, ["ok": true, "jobs": added, "estimate_s": total])
        }
    }

    // ── subjects ──

    /// **`grid`**: the request `add` checked, its axes read and the grid checked by the library
    /// (`RenderGrid`), every step count of it checked by the denoiser, the queue's room — then one job
    /// per cell, and the sheet when the last one ends.
    private func addGrid(_ d: RenderOrder, prompt: String, model: Model, card: ModelCard?, format: RecommendedFormat,
                         steps: Int, stack: [LoRASlot], references: [ReferenceImage], seed: UInt64,
                         defaultSteps: Int, folder: String?, cwd: String, _ c: Connection) throws {
        let g: RenderGrid
        do {
            let names = d.loras.map(\.name)
            // A LoRA axis names its LoRAs as `--lora` does (a name, else a path from the client's
            // folder): their paths for the cells.
            func resolved(_ a: RenderGrid.Axis) throws(EngineError) -> RenderGrid.Axis {
                guard case let .addedLoRAs(l, f) = a else { return a }
                var paths: [String] = []
                for name in l {
                    if name.isEmpty { paths.append(""); continue }
                    guard let card = app.library.lora(name, for: model.identifier)
                            ?? app.library.lora(absolute(name, cwd), for: model.identifier) else {
                        throw .gridRefused(reason: .noSuchLoRA)
                    }
                    paths.append(card.path)
                }
                return .addedLoRAs(paths, strength: f)
            }
            let x = try resolved(RenderGrid.axis(d.x ?? "", loras: names))
            let y = try d.y.map { try resolved(RenderGrid.axis($0, loras: names)) }
            g = try RenderGrid(.init(prompt: prompt, seed: seed, steps: steps, loraStrengths: stack.map(\.strength),
                                     images: references.count), x: x, y: y)
            guard !app.refusesNegativeStrength(g) else { throw EngineError.gridRefused(reason: .invalidValue) }
        } catch {
            let p = Problem(error)
            throw Refusal("grid", p.title, help: p.suggestion ?? "silicontrol help grid")
        }
        // A solver axis is the rack's Solver swept, so as `--solver`: Developer Mode, and the models in
        // front (a model axis: a cell of a model walking Euler alone renders Euler, as in the app).
        if g.cells.contains(where: { ($0.solver ?? .euler) != .euler }) {
            guard app.developerMode else {
                throw Refusal("solver", String(localized: "a solver axis is offered only in Developer Mode"),
                              help: String(localized: "turn on Developer Mode in Settings (⌘,)"))
            }
            guard g.cells.contains(where: { $0.model != nil }) || card.map({ AppState.solverFamilies.contains($0.family) }) == true else {
                throw Refusal("solver", String(localized: "\(card?.name ?? model.name) walks Euler only"),
                              help: String(localized: "Euler only, or --model z-image"))
            }
        }
        // Each cell's step count against its own model's schedules (a model axis: each model's).
        for cell in g.cells {
            guard let n = cell.steps else { continue }
            let denoising = cell.model.flatMap { try? Model.named($0, in: app.library).chain.denoising } ?? model.chain.denoising
            guard (try? denoising.check(steps: n, startImage: false)) == nil else { continue }
            let accepted = (1...50).filter { (try? denoising.check(steps: $0, startImage: false)) != nil }
                .map(String.init).formatted(.list(type: .or))
            throw Refusal("steps", String(localized: "This model has no schedule for \(String(n)) steps: \(accepted)."))
        }
        guard app.file.count + g.count <= AppState.maxQueueLength else {
            let n = g.count, max = AppState.maxQueueLength, room = max - app.file.count
            throw Refusal("queue_full", String(localized: "\(n) jobs for the grid: the queue has room for \(room) (\(max) jobs)"),
                          help: "silicontrol wait, then retry")
        }
        let settings = RenderSettings(identifier: model.identifier, modelName: card?.name ?? model.name, prompt: prompt,
                                      format: format, steps: steps, loras: stack, references: references.count, detail: d.detail,
                                      solver: d.solver)
        if d.sketch { app.stopExploring() }
        let (run, jobs) = app.enqueueGrid(g, settings: settings, references: references, previews: d.previews ?? app.previews,
                                          sketch: d.sketch, neuralEngine: d.neuralEngine)
        // The rack's tag of each model the cells render, as after a click.
        if let on = d.neuralEngine { for id in Set(jobs.map(\.settings.identifier)) { app.setNeuralEngine(on, for: id) } }
        // The form takes the grid, as after a click: settings, prompt as typed, edit images, axes.
        app.restoreSettings(settings)
        app.prompt = prompt
        app.references = references
        if let s = d.seed { app.fixedSeed = true; app.seed = s } else { app.fixedSeed = false }
        app.setGridForm(g, prompt: prompt)

        let added = jobs.map { job -> [String: Any] in
            var o = json(job)
            o["position"] = app.currentJob?.ordinal == job.ordinal
                ? 0 : (app.file.firstIndex { $0.ordinal == job.ordinal }.map { $0 + 1 } ?? 0)
            o["estimate_s"] = app.estimation(job).map { Response.number($0, 1) } ?? NSNull()
            return o
        }
        if d.wait {
            for var o in added { o["type"] = "added"; emit(c, o) }
            followGrid(run.ordinal, folder: folder, c)
        } else {
            let estimates = jobs.map { app.estimation($0) }
            let total: Any = estimates.contains { $0 == nil } ? NSNull()
                : Response.number(estimates.compactMap { $0 }.reduce(0, +), 1)
            finish(c, ["ok": true, "grid": run.name, "columns": g.columns, "rows": g.rows, "jobs": added,
                       "estimate_s": total])
        }
    }
}
