// **The app's remote control**: the open app listens on a Unix socket, and the `silicontrol`
// command (shipped inside Siliconed.app) drives it — the same queue, the same history, the same
// buttons as the mouse. The protocol and the commands are in `docs/REMOTE-CONTROL.md`; this file
// is its server.
//
// **The protocol**: one connection per command. The client writes ONE JSON line,
// `{"argv": ["add", "--model", "z-image", "…"], "cwd": "/client/folder"}`, then reads JSON lines
// until the connection closes. Intermediate lines carry `"type"` (an event); the last one carries
// `"ok"` (the verdict). The arguments are those of the command line, as is: all the parsing is
// here, the client is only a pipe — and `nc -U` is enough to talk to the app. The help
// (`help`, `help <topic>`, `<command> --help`) is answered here too, in the user's language:
// `{"ok": true, "help": "<text>"}`. Its pages and its routing live in `Sources/SilicontrolHelp`,
// shared with `silicontrol`, which answers them itself without launching the app.
//
// **Nobody else**: the socket is `<library>/silicontrol.sock`, mode 0600 — only the account that
// launched the app can open it. No network port. **Incognito kept**: nothing is written to disk
// unless asked (`--out`, `save`), like « Save » in the app.
//
// The plumbing (socket, reads, writes) never touches the main thread; the commands do: they read
// and change the `AppState`, like a button.

import AppKit
import Foundation
import Observation
import Siliconed
import SilicontrolHelp

// MARK: - The commands, on the main thread

@MainActor @Observable
final class RemoteControl {
    @ObservationIgnored let app: AppState
    @ObservationIgnored private var source: DispatchSourceRead?
    /// Why the remote control is not listening (another Siliconed, a path too long), or `nil` —
    /// shown greyed in the Siliconed menu, under « Install the silicontrol command… ».
    private(set) var failureMessage: String?

    init(app: AppState) { self.app = app }

    var path: String { app.library.socket }

    func boot() {
        do {
            source = try Socket.startListening(path) { [weak self] argv, cwd, connection in
                Task { @MainActor in self?.handle(argv, cwd: cwd, connection) }
            }
        } catch {
            failureMessage = "\(error)"
        }
    }

    /// When the app quits: the socket goes with it.
    func stop() {
        guard let source else { return }
        source.cancel()
        self.source = nil
        unlink(path)
    }

    // ── Answering ──

    private func finish(_ c: Connection, _ object: [String: Any]) {
        c.send(Response.rowLine(object))
        c.shutDown()
    }

    private func emit(_ c: Connection, _ object: [String: Any]) { c.send(Response.rowLine(object)) }

    /// An error a command raises: its code is the `error` field of the answer.
    struct Refusal: Error {
        let code: String, message: String, help: String?
        init(_ code: String, _ message: String, help: String? = nil) { self.code = code; self.message = message; self.help = help }
    }

    private func handle(_ argv: [String], cwd: String, _ c: Connection) {
        // `help`, `<command> --help`: the pages `silicontrol` also answers on its own, without touching the app.
        if let answer = RemoteControlHelp.answer(argv) { finish(c, answer); return }
        let command = argv.first ?? ""
        let arguments = Array(argv.dropFirst())
        do {
            switch command {
            case "status": finish(c, state())
            case "models": finish(c, models())
            case "install": finish(c, try install(arguments))
            case "diagnose": try diagnose(arguments, c)
            case "add": try add(arguments, cwd: cwd, c)
            case "grid": try add(arguments, cwd: cwd, c, grid: true)
            case "wait": try wait(arguments, cwd: cwd, c)
            case "follow": follow(c)
            case "cancel": finish(c, try cancel(arguments))
            case "clear": finish(c, clear())
            case "set": finish(c, try set(arguments))
            case "history": finish(c, try history(arguments))
            case "save": finish(c, try save(arguments, cwd: cwd))
            case "vary": finish(c, try vary(arguments))
            case "enlarge": try enlarge(arguments, c)
            case "subjects": finish(c, try subjects(arguments, cwd: cwd))
            default:
                throw Refusal("usage", String(localized: "unknown command: \"\(command)\""),
                            help: "silicontrol help")
            }
        } catch let r as Refusal {
            // A usage mistake with no advice of its own points to its command's help.
            let help = r.help ?? (r.code == "usage" ? "silicontrol help \(command)" : nil)
            finish(c, Response.failure(r.code, r.message, help: help))
        } catch {
            finish(c, Response.failure("internal", "\(error)"))
        }
    }

    // ── The objects returned ──

    private func json(_ t: Job) -> [String: Any] {
        let r = t.settings
        return ["job": t.name, "model": r.identifier, "prompt": r.prompt,
                "width": r.format.width, "height": r.format.height, "steps": r.steps,
                // A subject's pictures are references, not images to edit (`RenderSettings.editing`).
                "seeds": t.seeds.map { NSNumber(value: $0) }, "edit": r.editing,
                "references": t.references.count,
                "loras": r.loras.map { ["lora": Self.shortName($0.path), "strength": Response.number($0.strength)] }]
            .merging(gridPlace(t)) { a, _ in a }
            // Only a job that cites named subjects has it: with the switch off, the answer is unchanged.
            .merging(r.modelPrompt.map { ["model_prompt": $0, "subject_images": r.subjectImages] } ?? [:]) { a, _ in a }
            .merging(r.isVariation ? ["variation": Self.json(r.variations)] : [:]) { a, _ in a }
            .merging(r.detail == .normal ? [:] : ["detail": r.detail.rawValue]) { a, _ in a }
            .merging(r.solver == .euler ? [:] : ["solver": r.solver.rawValue]) { a, _ in a }
            .merging(r.schedule == .official ? [:] : ["schedule": r.schedule.rawValue]) { a, _ in a }
            // What the job will ask, on a Mac whose map offers the Neural Engine: elsewhere, the answer is unchanged.
            .merging(app.neuralEngineOffered ? ["neural_engine": t.neuralEngine ?? app.neuralEngine(for: r.identifier)] : [:]) { a, _ in a }
    }

    /// The variation seeds in order, each with its strength — the seed stays the origin's.
    private static func json(_ variations: [Variation]) -> [[String: Any]] {
        variations.map { ["seed": NSNumber(value: $0.seed), "strength": Response.number($0.strength)] }
    }

    /// A grid's cell: its grid, its column and its row (from 0) — empty for any other job.
    private func gridPlace(_ t: Job) -> [String: Any] {
        guard let at = app.gridPlace(of: t.ordinal), let run = app.gridRuns[at.grid] else { return [:] }
        return ["grid": run.name, "column": at.cell % run.grid.columns, "row": at.cell / run.grid.columns]
    }

    private func json(_ e: Entry, path: String? = nil) -> [String: Any] {
        let r = e.settings
        var o: [String: Any] = [
            "image": e.name, "job": "t\(e.job)", "model": r.identifier, "prompt": r.prompt,
            "width": e.width, "height": e.height, "seed": NSNumber(value: e.seed), "steps": r.steps,
            "evaluations": e.evaluations, "seconds": Response.number(e.seconds), "edit": r.editing,
            "references": r.references,
            "loras": r.loras.map { ["lora": Self.shortName($0.path), "strength": Response.number($0.strength)] }]
        if r.isVariation { o["variation"] = Self.json(r.variations) }
        if r.detail != .normal { o["detail"] = r.detail.rawValue }
        if r.solver != .euler { o["solver"] = r.solver.rawValue }
        if r.schedule != .official { o["schedule"] = r.schedule.rawValue }
        if r.neuralEngine { o["neural_engine"] = true }
        if let g = e.grid {
            // A grid's sheet: no job made it; its seconds and evaluations are its cells' sum.
            o["job"] = nil
            o["grid"] = g.name
            o["sheet"] = true
            o["columns"] = g.grid.columns
            o["rows"] = g.grid.rows
            o["empty_cells"] = g.empty
        }
        if let n = e.sketch { o["sketch"] = n }
        if let n = e.enlarged { o["enlarged"] = n }
        if let path { o["path"] = path }
        return o
    }

    /// A job's end, as the remote control writes it: `outcome`, and for an error its translated
    /// `message` and the engine's stable `code` (`EngineError.code`, `"license_not_accepted"`…);
    /// `warnings`, the engine's own lines during its render (English), when it wrote any.
    private static func json(_ issue: Issue, job n: Int, images: [Int], warnings: [String] = []) -> [String: Any] {
        var o: [String: Any] = ["job": "t\(n)", "images": images.map { "i\($0)" }]
        if !warnings.isEmpty { o["warnings"] = warnings }
        switch issue {
        case .finished: o["outcome"] = "done"
        case .stopped: o["outcome"] = "stopped"
        case .removed: o["outcome"] = "removed"
        case .failed(let p):
            o["outcome"] = "error"
            o["message"] = p.text
            if let code = p.code { o["code"] = code }
        }
        return o
    }

    /// `flat` for `…/store/flat.lora.silicon`: what one types after `--lora`.
    nonisolated static func shortName(_ path: String) -> String {
        let file = (path as NSString).lastPathComponent
        return file.hasSuffix(".lora.silicon") ? String(file.dropLast(".lora.silicon".count)) : file
    }

    // ── status ──

    private func state() -> [String: Any] {
        var o: [String: Any] = ["ok": true, "library": app.library.root.path,
                                "history_images": app.history.count,
                                "faster_rendering": app.fasterRendering]
        // The settings renders run on: a series compared across launches compares maps too.
        o["settings"] = ["read": app.settingsRead, "map": app.mapInUse?.date ?? NSNull(),
                         "warnings": app.profileWarnings] as [String: Any]
        if let t = app.currentJob {
            var e = json(t)
            e["stage"] = app.stage.map(Self.stageName) ?? NSNull()
            e["batch_image"] = app.batchImage + 1
            e["step"] = app.currentStep
            e["steps_total"] = app.totalSteps
            e["fraction"] = Response.number(app.fraction, 3)
            e["remaining_s"] = app.remaining(at: Date()).map { Response.number($0, 1) } ?? NSNull()
            o["running"] = e
        } else {
            o["running"] = NSNull()
        }
        o["queue"] = app.file.map { t -> [String: Any] in
            var e = json(t)
            e["estimate_s"] = app.estimation(t).map { Response.number($0, 1) } ?? NSNull()
            return e
        }
        // Refused by the engine until the user accepts the model's license in the app (a sheet shows it):
        // out of the queue, not lost — accepted, they go back to its head; declined, they end in "error".
        o["waiting_for_license"] = app.awaitingLicense.map { t -> [String: Any] in
            var e = json(t)
            e["state"] = "waiting_for_license"
            let license = app.cards.first { $0.id == t.settings.identifier }?.license
            e["license"] = license?.text ?? NSNull()
            e["license_urls"] = license?.urls.map(\.absoluteString) ?? NSNull()
            return e
        }
        // An installation or an import blocks renders (`busy`): say it, and where it is.
        o["installing"] = app.forgeInProgress.map { title -> [String: Any] in
            ["title": title, "last": app.forgeJournal.last ?? NSNull()]
        } ?? NSNull()
        let remaining = app.remainingInQueue(at: Date())
        o["queue_remaining_s"] = Response.number(remaining.seconds, 1)
        o["remaining_complete"] = remaining.isComplete
        return o
    }

    private static func stageName(_ item: Engine.Stage) -> String {
        switch item {
        case .text: "text"
        case .image: "image"
        case .denoising: "denoising"
        case .decoding: "decoding"
        }
    }

    // ── models ──

    /// The models the app shows — Z-Image, Qwen-Image-2.1, Anima and Krea 2, in the app's order
    /// (`AppState.frontFamilies`).
    private func models() -> [String: Any] {
        let b = app.library
        let list = visibleCards(b).map { f -> [String: Any] in
            var o: [String: Any] = ["id": f.id, "name": f.name, "default_steps": f.defaultSteps,
                                    "formats": f.formats.map { "\($0.width)x\($0.height)" },
                                    "commercial_license": f.license.commercial,
                                    "license": f.license.text,
                                    // A Compact or a Light also links its third parties' model cards (`Family.licenseURLs`);
                                    // an imported model, its installed version's (its encoder and VAE).
                                    "license_urls": f.family.licenseURLs(b.variant(of: f.family))
                                        .map(\.absoluteString),
                                    "license_accepted": b.isLicenseAccepted(f)]
            o["default"] = f.id == "z-image"   // the default of `add` without --model
            // Stage C: the versions a publisher's model installs in, each with its place on disk, and
            // the one on disk once it is ready (an imported model has its own DiT: no version).
            if !f.isImported {
                o["variants"] = f.family.variants.map { v -> [String: Any] in
                    let s = f.family.installedSize(v)
                    return ["variant": v.rawValue, "disk_bytes": s.dit + s.encoder + s.components]
                }
            }
            if let missingList = f.missing(in: b) {
                o["ready"] = false
                o["missing"] = missingList
                // The user installs, in the app: the license and the size are read there first.
                o["install"] = f.isImported ? "silicontrol install \(f.family.rawValue)" : "silicontrol install \(f.id)"
                // One only knows by opening the installed model's chain.
                o["edit"] = NSNull()
                o["max_references"] = NSNull()
                o["variant"] = NSNull()
            } else {
                o["ready"] = true
                o["variant"] = f.isImported ? NSNull() : b.variant(of: f.family).rawValue as Any
                let max = (try? Model.named(f.id, in: b))?.chain.denoising.maxReferences ?? 0
                o["edit"] = max > 0
                o["max_references"] = max
                o["loras"] = b.loras(for: f.id).map { ["lora": Self.shortName($0.path), "name": $0.name] }
                // The rack's fp32 tag, per installed version: only on a Mac whose map offers the Neural Engine.
                if app.neuralEngineOffered { o["neural_engine"] = app.neuralEngine(for: f.id) }
            }
            return o
        }
        return ["ok": true, "models": list]
    }

    /// The library's models the app shows, in its order (`AppState.inDisplayOrder`).
    private func visibleCards(_ b: Library) -> [ModelCard] {
        AppState.inDisplayOrder(b.cards().filter { app.isVisible($0.family) }, family: \.family)
    }

    // ── install ──

    /// `install <id> [--standard|--compact|--light]`: **the app shows the model's license and the place it
    /// takes, and the user accepts there** — the command downloads nothing by itself. It opens the
    /// Models sheet on the model and returns at once; `models` then says "ready" once the
    /// installation is done. **Without a flag the version installed is kept** (Standard if none): a
    /// command that names no version never switches. `--standard`/`--compact`/`--light` preselect that
    /// version — the one installed is offered for replacement, like « Switch to … » in the sheet. An
    /// imported model's prerequisites (`baseDiT` false) never switch: a flag that contradicts the
    /// family's version installed is refused, with the reason.
    private func install(_ arguments: [String]) throws -> [String: Any] {
        let options = arguments.filter { $0.hasPrefix("--") }
        let flags = Variant.allCases.map { "--" + $0.rawValue }
        if let unknown = options.first(where: { !flags.contains($0) }) {
            throw Refusal("usage", String(localized: "unknown flag: \(unknown)"), help: "silicontrol help install")
        }
        let named = Variant.allCases.filter { options.contains("--" + $0.rawValue) }
        if named.count > 1 {
            throw Refusal("usage", String(localized: "one version at most: --standard, --compact or --light"), help: "silicontrol help install")
        }
        let ids = arguments.filter { !$0.hasPrefix("--") }
        guard ids.count == 1, let id = ids.first else {
            throw Refusal("usage", String(localized: "install expects one model id"), help: "silicontrol models")
        }
        let b = app.library
        guard let card = b.cards().first(where: { $0.id == id || $0.family.rawValue == id }), app.isVisible(card.family) else {
            let visible = visibleCards(b).map(\.id).joined(separator: ", ")
            throw Refusal("model", String(localized: "unknown model: \"\(id)\""), help: String(localized: "models: \(visible)"))
        }
        let asked = named.first
        if let asked, !card.family.variants.contains(asked) {
            let version = VersionChoice.name(asked)
            throw Refusal("usage", String(localized: "\(card.name) has no \(version) version"), help: "silicontrol models")
        }
        let installed = b.installedVariant(of: card.family)
        // Nothing installed: the version the app preselects (Standard; Light on a Mac of 8 GB).
        let variant = asked ?? installed ?? (card.isImported ? .standard : card.family.preselectedVariant())
        // What an imported model lacks installs in its family's version, never another one.
        if card.isImported, let asked, asked != (installed ?? .standard) {
            let name = card.name, family = card.family.name, kept = VersionChoice.name(installed ?? .standard)
            let command = "silicontrol install \(card.family.rawValue) --\(asked.rawValue)"
            throw Refusal("usage", String(localized: "\(name) needs \(family) in the version installed (\(kept)): an imported model does not switch versions"),
                          help: String(localized: "switch the publisher's model first: \(command)"))
        }
        // Ready, and in the version asked (an imported model keeps its family's version): nothing to do.
        guard card.missing(in: b) != nil || (!card.isImported && b.variant(of: card.family) != variant) else {
            return ["ok": true, "model": card.id, "installed": true, "variant": b.variant(of: card.family).rawValue,
                    "message": String(localized: "already installed")]
        }
        guard app.managementPossible else {
            throw Refusal("busy", String(localized: "a render, an installation or the diagnostic is running"),
                        help: "silicontrol wait, then retry")
        }
        app.offerInstall(card.family, baseDiT: !card.isImported, variant: variant)
        NSApp.activate()
        return ["ok": true, "model": card.id, "installed": false, "variant": variant.rawValue,
                "message": String(localized: "the app shows the license and the space on disk: the user accepts there")]
    }

    // ── diagnose ──

    /// `diagnose [--issue]`: **« Measure This Mac », from the terminal** — the sheet opens in the
    /// app and the diagnostic runs (~1 min, no weights, the queue held meanwhile); the answer is the
    /// report, the same JSON the issue carries. `--issue` then does what the sheet's
    /// « Open the Issue » does: GitHub's form opens in the browser, prefilled — the user submits it.
    private func diagnose(_ arguments: [String], _ c: Connection) throws {
        let issue = arguments.contains("--issue")
        if let unknown = arguments.first(where: { $0 != "--issue" }) {
            throw Refusal("usage", String(localized: "unknown flag: \(unknown)"), help: "silicontrol help diagnose")
        }
        if let reason = app.diagnosticBlocker { throw Refusal("busy", reason, help: "silicontrol wait, then retry") }
        app.openDiagnostic()
        app.startDiagnostic()
        NSApp.activate()
        emit(c, ["type": "started", "stages": Diagnostic.Stage.allCases.map(\.rawValue)])
        Task { @MainActor [weak self] in
            while let self, self.app.diagnosticRunning, !c.broken {
                try? await Task.sleep(for: .milliseconds(500))
            }
            guard let self else { return }
            if case .failed(let problem) = self.app.diagnostic {
                self.finish(c, Response.failure(problem.code ?? "failed", problem.text))
                return
            }
            guard case .finished(let report) = self.app.diagnostic,
                  let data = try? report.json(pretty: false),
                  let object = try? JSONSerialization.jsonObject(with: data) else {
                self.finish(c, Response.failure("stopped", String(localized: "the diagnostic was stopped")))
                return
            }
            var o: [String: Any] = ["ok": true, "report": object]
            if issue {
                o["issue_url_length"] = (try? report.issueURL(includingReport: false).absoluteString.count) ?? NSNull()
                if let notice = self.app.openIssue(report) { o["message"] = notice }
            }
            self.finish(c, o)
        }
    }

    // ── add ──

    /// The arguments of the developer's render command, plus `--wait` and `--out`.
    private struct RenderOrder {
        var model = "z-image"
        var prompt: String?
        var format: (Int, Int)?
        var seed: UInt64?
        var batch = 1
        var steps: Int?
        /// `--detail more|most`: the rack's Detail choice (`Request.detail`).
        var detail = Detail.normal
        /// `--solver NAME`: the rack's Solver (`Request.solver`), Developer Mode and `AppState.solverFamilies` only.
        var solver = Solver.euler
        /// `--schedule official|comfyui`: the rack's ComfyUI schedule switch (`Request.schedule`), Developer
        /// Mode and the models that offer it only.
        var schedule = SigmaSchedule.official
        /// `--ane`, `--no-ane`: the rack's fp32 tag (`Job.neuralEngine`); `nil`, the model's choice.
        var neuralEngine: Bool?
        var loras: [(name: String, strength: Double)] = []
        /// The `--ref`, in their order: the first is image 1, the one edited.
        var references: [String] = []
        /// `nil`: like the app's « Preview at every step » switch.
        var previews: Bool?
        var wait = false
        var folder: String?
        /// `grid`: the axes as typed (`RenderGrid.axis`), and whether `--batch` was given (refused).
        var x: String?, y: String?
        var batchGiven = false
        /// `grid --sketch`: the exploration's cells (`Sketch`), shown in its window, not in the history.
        var sketch = false
    }

    private func parse(_ arguments: [String]) throws -> RenderOrder {
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
            case "--schedule":
                let v = try value(a)
                guard let s = SigmaSchedule(rawValue: v.lowercased()) else {
                    throw Refusal("usage", String(localized: "--schedule expects official or comfyui, not \"\(v)\""))
                }
                d.schedule = s
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

    private func absolute(_ path: String, _ cwd: String) -> String {
        URL(fileURLWithPath: (path as NSString).expandingTildeInPath, relativeTo: URL(fileURLWithPath: cwd, isDirectory: true))
            .standardizedFileURL.path
    }

    /// `add`, and `grid` (`grid: true`): the same request, checked the same way; a grid queues one
    /// job per cell (`AppState.enqueueGrid`) instead of one per alternative.
    private func add(_ arguments: [String], cwd: String, _ c: Connection, grid: Bool = false) throws {
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
        // ComfyUI's schedule: the rack's switch, so Developer Mode, and the models that offer it.
        if d.schedule != .official {
            guard app.developerMode else {
                throw Refusal("schedule", String(localized: "--schedule \(d.schedule.rawValue) is offered only in Developer Mode"),
                              help: String(localized: "turn on Developer Mode in Settings (⌘,)"))
            }
            guard model.chain.denoising.schedules.contains(d.schedule) else {
                throw Refusal("schedule", String(localized: "\(card?.name ?? model.name) doesn't offer the ComfyUI schedule"),
                              help: String(localized: "leave out --schedule, or --model qwen-image-2.1"))
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
                                                 withLoRA: !stack.isEmpty || d.detail != .normal, schedule: d.schedule)
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
                                          solver: d.solver, schedule: d.schedule)
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

    /// **`subjects`** (experimental, `Subjects.swift`): list, name, delete — the Subjects mode's
    /// gestures, in the app's memory. Citing one is `add "@alice in a library"`. Refused while the app's switch is off.
    ///
    ///     subjects                                              the library
    ///     subjects add IMAGE --name N --kind K [--frame x,y,w,h] [--second IMAGE2]
    ///     subjects remove NAME
    private func subjects(_ arguments: [String], cwd: String) throws -> [String: Any] {
        guard app.subjects.enabled else {
            throw Refusal("disabled", String(localized: "named subjects are off"),
                          help: String(localized: "turn on Experimental → Named Subjects in Settings (⌘,)"))
        }
        // In memory, like the history: a subject lives as long as the app (incognito) — no path to give.
        func json(_ s: NamedSubjects.Subject) -> [String: Any] {
            ["name": s.name, "kind": s.kind.rawValue,
             "pictures": s.pictures.map { p -> [String: Any] in
                 ["frame": [p.frame.x, p.frame.y, p.frame.width, p.frame.height].map { Response.number($0, 4) }]
             }]
        }
        switch arguments.first {
        case nil:
            return ["ok": true, "subjects": app.subjects.all.map(json)]
        case "remove":
            guard arguments.count == 2, let s = app.subjects.subject(named: arguments[1]) else {
                let name = arguments.dropFirst().first ?? ""
                throw Refusal("subject", String(localized: "No subject is called @\(name)."), help: "silicontrol subjects")
            }
            app.subjects.delete(s)
            return ["ok": true, "removed": s.name]
        case "add":
            var images: [String] = [], first: String?, name: String?, kind: NamedSubjects.Kind?
            var frames: [NamedSubjects.Frame] = []
            var i = 1
            func value(_ flag: String) throws -> String {
                guard i + 1 < arguments.count else { throw Refusal("usage", String(localized: "\(flag) expects a value")) }
                i += 1
                return arguments[i]
            }
            while i < arguments.count {
                let a = arguments[i]
                switch a {
                case "--name": name = try value(a)
                case "--kind":
                    let v = try value(a)
                    guard let k = NamedSubjects.Kind(rawValue: v.lowercased()) else {
                        let kinds = NamedSubjects.Kind.allCases.map(\.rawValue).joined(separator: ", ")
                        throw Refusal("usage", String(localized: "--kind expects one of \(kinds), not \"\(v)\""))
                    }
                    kind = k
                case "--frame":
                    let v = try value(a)
                    let n = v.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
                    guard n.count == 4, n.allSatisfy({ (0...1).contains($0) }), n[2] > 0, n[3] > 0 else {
                        throw Refusal("usage", String(localized: "--frame expects x,y,width,height in fractions of the image, not \"\(v)\""))
                    }
                    frames.append(NamedSubjects.Frame(x: n[0], y: n[1], width: n[2], height: n[3]))
                case "--second": images.append(try value(a))
                default:
                    if a.hasPrefix("--") { throw Refusal("usage", String(localized: "unknown flag: \(a)")) }
                    // One IMAGE, first wherever it is written; a second one is `--second`'s.
                    guard first == nil else {
                        throw Refusal("usage", String(localized: "subjects add IMAGE --name NAME --kind KIND"), help: "silicontrol help subjects")
                    }
                    first = a
                }
                i += 1
            }
            if let first { images.insert(first, at: 0) }
            guard let name, let kind, first != nil, (1...NamedSubjects.maxPictures).contains(images.count) else {
                throw Refusal("usage", String(localized: "subjects add IMAGE --name NAME --kind KIND"), help: "silicontrol help subjects")
            }
            let sources = try images.enumerated().map { n, path in
                let url = URL(fileURLWithPath: absolute(path, cwd))
                // Checked as the sheet does (it decodes the picture): a file ImageIO cannot read would
                // otherwise make a subject that fails every render citing it.
                guard ReferenceImage.size(of: url) != nil else {
                    let path = url.path
                    throw Refusal("ref", String(localized: "image unreadable or missing: \(path)"))
                }
                return (source: SubjectSource.file(url),
                        frame: frames.indices.contains(n) ? frames[n] : NamedSubjects.Frame.whole)
            }
            if let p = app.subjects.create(name: name, kind: kind, pictures: sources) {
                throw Refusal("subject", p.title, help: p.suggestion)
            }
            guard let s = app.subjects.subject(named: name) else { throw Refusal("subject", String(localized: "No subject is called @\(name).")) }
            return ["ok": true, "subject": json(s)]
        default:
            throw Refusal("usage", String(localized: "unknown subcommand: \(arguments[0])"), help: "silicontrol help subjects")
        }
    }

    // ── grid ──

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
                                      solver: d.solver, schedule: d.schedule)
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

    /// **Follows a grid to its sheet**: each cell's image as it comes (written into `folder` if
    /// given), then the sheet, then the verdict. A grid already ended answers at once.
    private func followGrid(_ g: Int, folder: String?, _ c: Connection) {
        let jobs = app.gridRuns[g]?.jobs ?? app.drawingSheets[g] ?? app.endedGrids[g]?.jobs ?? []
        var tallies: [Int: [String: Any]] = [:]
        func verdict(sheet: Int?) {
            let list = jobs.compactMap { tallies[$0] }
            let allFinished = list.count == jobs.count && list.allSatisfy { $0["outcome"] as? String == "done" }
            var o: [String: Any] = ["ok": allFinished, "grid": "g\(g)", "sheet": sheet.map { "i\($0)" } ?? NSNull(), "jobs": list]
            if !allFinished {
                o["error"] = "render"
                o["message"] = String(localized: "at least one cell did not render: see \"outcome\" in \"jobs\"; the sheet marks it empty")
            }
            finish(c, o)
        }
        for n in jobs {
            guard let t = app.completed[n] else { continue }
            for k in t.images { if let e = app.history.first(where: { $0.ordinal == k }) { emitImage(e, folder: folder, c) } }
            tallies[n] = Self.json(t.issue, job: n, images: t.images, warnings: t.warnings)
        }
        if app.drawingSheets[g] == nil, let ended = app.endedGrids[g] {
            if let s = ended.sheet, let e = app.history.first(where: { $0.ordinal == s }) {
                emitImage(e, folder: folder, type: "sheet", c)
            }
            verdict(sheet: ended.sheet)
            return
        }
        var token: UUID?
        token = app.subscribe { [weak self] signal in
            guard let self else { return }
            if c.broken { if let token { self.app.unsubscribe(token) }; return }
            switch signal {
            case .image(let t, let e) where jobs.contains(t.ordinal):
                self.emitImage(e, folder: folder, c)
            case .finished(let t, let issue) where jobs.contains(t.ordinal):
                tallies[t.ordinal] = Self.json(issue, job: t.ordinal, images: self.app.completed[t.ordinal]?.images ?? [],
                                               warnings: self.app.completed[t.ordinal]?.warnings ?? [])
            case .sheet(let run, let entry) where run.ordinal == g:
                if let entry { self.emitImage(entry, folder: folder, type: "sheet", c) }
                if let token { self.app.unsubscribe(token) }
                verdict(sheet: entry?.ordinal)
            default:
                break
            }
        }
    }

    /// One `"type": "image"` line (or `"sheet"`), the PNG written into `folder` if given.
    private func emitImage(_ e: Entry, folder: String?, type: String = "image", _ c: Connection) {
        var path: String?
        var o: [String: Any] = [:]
        if let folder {
            do { path = try write(e, in: folder) } catch { o["write_error"] = "\(error)" }
        }
        o.merge(json(e, path: path)) { a, _ in a }
        o["type"] = type
        emit(c, o)
    }

    // ── wait ──

    private func wait(_ arguments: [String], cwd: String, _ c: Connection) throws {
        var ordinals: [Int] = []
        var grids: [Int] = []
        var folder: String?
        var i = 0
        while i < arguments.count {
            let a = arguments[i]
            if a == "--out" {
                guard i + 1 < arguments.count else { throw Refusal("usage", String(localized: "--out expects a folder")) }
                folder = absolute(arguments[i + 1], cwd)
                i += 2
                continue
            }
            // `g2`: a grid, followed to its sheet.
            if a.first == "g", let n = Int(a.dropFirst()) {
                guard app.gridRuns[n] != nil || app.drawingSheets[n] != nil || app.endedGrids[n] != nil else {
                    throw Refusal("unknown_job", String(localized: "no grid \(a) in this session"), help: "silicontrol status")
                }
                grids.append(n)
                i += 1
                continue
            }
            guard let n = Self.ordinal(a, prefix: "t") else {
                throw Refusal("usage", String(localized: "job expected (t3), not \"\(a)\""))
            }
            guard app.completed[n] != nil || app.currentJob?.ordinal == n
                    || app.file.contains(where: { $0.ordinal == n })
                    || app.awaitingLicense.contains(where: { $0.ordinal == n }) else {
                throw Refusal("unknown_job", String(localized: "no job \(a) in this session"), help: "silicontrol status")
            }
            ordinals.append(n)
            i += 1
        }
        if !grids.isEmpty {
            guard grids.count == 1, ordinals.isEmpty else {
                throw Refusal("usage", String(localized: "wait follows one grid alone: wait g2 [--out D]"))
            }
            if let folder { try createFolder(folder) }
            followGrid(grids[0], folder: folder, c)
            return
        }
        if ordinals.isEmpty {
            // With no job named: everything running or waiting now.
            ordinals = (app.currentJob.map { [$0.ordinal] } ?? []) + app.file.map(\.ordinal) + app.awaitingLicense.map(\.ordinal)
        }
        if let folder { try createFolder(folder) }
        followToEnd(ordinals, folder: folder, c)
    }

    /// Emits each image (written into `folder` if given), then the verdict when all the jobs
    /// are done. Those already done answer at once.
    private func followToEnd(_ ordinals: [Int], folder: String?, _ c: Connection) {
        var remainingSet = Set(ordinals)
        var tallies: [Int: [String: Any]] = [:]

        func image(_ e: Entry) { emitImage(e, folder: folder, c) }
        func tally(_ n: Int, _ issue: Issue, _ images: [Int]) {
            tallies[n] = Self.json(issue, job: n, images: images, warnings: app.completed[n]?.warnings ?? [])
            remainingSet.remove(n)
        }
        func verdict() {
            let jobs = ordinals.compactMap { tallies[$0] }
            let allFinished = jobs.allSatisfy { $0["outcome"] as? String == "done" }
            var o: [String: Any] = ["ok": allFinished, "jobs": jobs]
            if !allFinished {
                o["error"] = "render"
                o["message"] = String(localized: "at least one job did not finish: see \"outcome\" in \"jobs\"")
            }
            finish(c, o)
        }

        for n in ordinals {
            guard let t = app.completed[n] else { continue }
            for k in t.images { if let e = app.history.first(where: { $0.ordinal == k }) { image(e) } }
            tally(n, t.issue, t.images)
        }
        if remainingSet.isEmpty { verdict(); return }

        var token: UUID?
        token = app.subscribe { [weak self] signal in
            guard let self else { return }
            if c.broken { if let token { self.app.unsubscribe(token) }; return }
            switch signal {
            case .image(let t, let e) where remainingSet.contains(t.ordinal):
                image(e)
            case .finished(let t, let issue) where remainingSet.contains(t.ordinal):
                tally(t.ordinal, issue, self.app.completed[t.ordinal]?.images ?? [])
                if remainingSet.isEmpty {
                    if let token { self.app.unsubscribe(token) }
                    verdict()
                }
            default:
                break
            }
        }
    }

    // ── follow ──

    private func follow(_ c: Connection) {
        guard app.busy || !app.awaitingLicense.isEmpty else {
            finish(c, ["ok": true, "message": String(localized: "nothing is running or waiting")])
            return
        }
        var token: UUID?
        // Idle once nothing is queued, rendering, waiting **or drawing a sheet**: the last cell of
        // a grid ends before its sheet is drawn (`finishGrid`), and `follow` promises the sheet.
        func quitIfIdle() {
            guard app.file.isEmpty, app.currentJob == nil, app.awaitingLicense.isEmpty,
                  app.gridRuns.isEmpty, app.drawingSheets.isEmpty else { return }
            if let token { app.unsubscribe(token) }
            finish(c, ["ok": true, "message": String(localized: "queue empty")])
        }
        token = app.subscribe { [weak self] signal in
            guard let self else { return }
            if c.broken { if let token { self.app.unsubscribe(token) }; return }
            switch signal {
            case .started(let t):
                var o = self.json(t); o["type"] = "start"; self.emit(c, o)
            case let .stage(t, item, image):
                self.emit(c, ["type": "stage", "job": t.name, "stage": Self.stageName(item), "batch_image": image + 1])
            case let .step(t, image, index, total, seconds):
                self.emit(c, ["type": "step", "job": t.name, "batch_image": image + 1, "step": index, "steps_total": total,
                                 "seconds": Response.number(seconds)])
            case let .image(_, e):
                var o = self.json(e); o["type"] = "image"; self.emit(c, o)
            case let .sheet(run, entry):
                var o: [String: Any] = entry.map { self.json($0) } ?? ["grid": run.name, "image": NSNull()]
                o["type"] = "sheet"
                self.emit(c, o)
                quitIfIdle()
            case let .finished(t, issue):
                var o = Self.json(issue, job: t.ordinal, images: self.app.completed[t.ordinal]?.images ?? [],
                                  warnings: self.app.completed[t.ordinal]?.warnings ?? [])
                o["type"] = "end"
                self.emit(c, o)
                // `.finished` goes out before the next one is started: the queue says whether any is left.
                quitIfIdle()
            }
        }
    }

    // ── cancel, clear ──

    private func cancel(_ arguments: [String]) throws -> [String: Any] {
        var stopped: String?, removedNames: [String] = [], alreadyFinished: [String] = []
        let ordinals = try arguments.map { a -> Int in
            guard let n = Self.ordinal(a, prefix: "t") else { throw Refusal("usage", String(localized: "job expected (t3), not \"\(a)\"")) }
            return n
        }
        if ordinals.isEmpty {
            if let t = app.currentJob { app.cancel(); stopped = t.name }
        }
        for n in ordinals {
            if app.currentJob?.ordinal == n {
                app.cancel(); stopped = "t\(n)"
            } else if let t = (app.file + app.awaitingLicense).first(where: { $0.ordinal == n }) {
                app.remove(t); removedNames.append(t.name)
            } else if app.completed[n] != nil {
                alreadyFinished.append("t\(n)")
            } else {
                throw Refusal("unknown_job", String(localized: "no job t\(n) in this session"), help: "silicontrol status")
            }
        }
        return ["ok": true, "stopped": stopped ?? NSNull(), "removed": removedNames, "already_finished": alreadyFinished]
    }

    private func clear() -> [String: Any] {
        let stopped = app.currentJob?.name
        let removedNames = (app.file + app.awaitingLicense).map(\.name)
        app.stopAll()
        return ["ok": true, "stopped": stopped ?? NSNull(), "removed": removedNames]
    }

    // ── set ──

    /// **`set faster-rendering on|off`: the Settings switch, from the terminal** — `AppState.fasterRendering`,
    /// so the preference is written and the window's switch follows, exactly as a click. Only with the queue
    /// empty: the engine reads it at every plan, and a job started under one value and planned under the
    /// other would mix two recipes in one series. Setting the value it already has is not a change.
    private func set(_ arguments: [String]) throws -> [String: Any] {
        guard arguments.count == 2, arguments[0] == "faster-rendering", ["on", "off"].contains(arguments[1]) else {
            throw Refusal("usage", "set faster-rendering on|off")
        }
        guard !app.busy, app.forgeInProgress == nil, !app.diagnosticRunning, app.awaitingLicense.isEmpty else {
            throw Refusal("busy", String(localized: "a render, an installation or the diagnostic is running"),
                          help: "silicontrol wait, then retry")
        }
        let previous = app.fasterRendering
        app.fasterRendering = arguments[1] == "on"
        return ["ok": true, "faster_rendering": app.fasterRendering, "previous": previous]
    }

    // ── history, save ──

    private func history(_ arguments: [String]) throws -> [String: Any] {
        var n = Int.max
        if let a = arguments.first {
            guard let v = Int(a), v >= 1 else { throw Refusal("usage", String(localized: "history [N]: N an integer ≥ 1, not \"\(a)\"")) }
            n = v
        }
        return ["ok": true, "images": app.history.prefix(n).map { json($0) }]
    }

    private func save(_ arguments: [String], cwd: String) throws -> [String: Any] {
        guard arguments.count >= 2, let destination = arguments.last else {
            throw Refusal("usage", "save <i7> [i8 …] <file.png|file.jpg|folder>")
        }
        let entries = try arguments.dropLast().map { a -> Entry in
            guard let e = image(a) else {
                throw Refusal("unknown_image", String(localized: "no image \"\(a)\" in the history"), help: "silicontrol history")
            }
            return e
        }
        let target = absolute(destination, cwd)
        let ext = (target as NSString).pathExtension.lowercased()
        var files: [[String: Any]] = []
        if entries.count == 1, ["png", "jpg", "jpeg"].contains(ext) {
            // A named file: written, or replaced — it is « Save as ».
            try createFolder((target as NSString).deletingLastPathComponent)
            try writeFile(entries[0], target)
            files.append(["image": entries[0].name, "path": target])
        } else {
            try createFolder(target)
            for e in entries { files.append(["image": e.name, "path": try write(e, in: target)]) }
        }
        return ["ok": true, "files": files]
    }

    // ── vary ──

    /// **`vary i7 subtle|strong [--batch N]`**: « Variations » on a history image, as its menu does — N
    /// jobs (the rack's « Images » by default) with its settings and seed, each turned by a variation
    /// seed drawn at random.
    private func vary(_ arguments: [String]) throws -> [String: Any] {
        var count = app.batch
        var rest = arguments
        if rest.count == 4, rest[2] == "--batch" {
            guard let n = Int(rest[3]), (1...8).contains(n) else {
                throw Refusal("usage", String(localized: "--batch expects an integer from 1 to 8, not \"\(rest[3])\""))
            }
            count = n
            rest.removeLast(2)
        }
        guard rest.count == 2, let amount = Variation.Amount(rawValue: rest[1]) else {
            throw Refusal("usage", "vary <i7> subtle|strong [--batch N]")
        }
        let arguments = rest
        guard let e = image(arguments[0]) else {
            throw Refusal("unknown_image", String(localized: "no image \"\(arguments[0])\" in the history"), help: "silicontrol history")
        }
        if let reason = app.libraryBlocker { throw Refusal("library_absent", reason, help: String(localized: "connect the disk, then retry")) }
        if let reason = app.installBlocker { throw Refusal("busy", reason, help: String(localized: "retry once the installation is done (silicontrol models)")) }
        guard app.canVary(e) else {
            throw Refusal("model", String(localized: "\(e.name) has no variations: a grid's sheet, or its model is not ready"),
                          help: "silicontrol models")
        }
        guard app.file.count + count <= AppState.maxQueueLength else {
            throw Refusal("queue_full", String(localized: "the queue is full (\(AppState.maxQueueLength) jobs)"),
                          help: "silicontrol wait, then retry")
        }
        return ["ok": true, "jobs": app.vary(e, amount, count: count).map { json($0) }]
    }

    /// `enlarge i7 [--download]`: « Enlarge ×4 », answered once the enlarged image is in the history.
    private func enlarge(_ arguments: [String], _ c: Connection) throws {
        guard (1...2).contains(arguments.count), arguments.count == 1 || arguments[1] == "--download" else {
            throw Refusal("usage", "enlarge <i7> [--download]")
        }
        guard let e = image(arguments[0]) else {
            throw Refusal("unknown_image", String(localized: "no image \"\(arguments[0])\" in the history"), help: "silicontrol history")
        }
        let download = arguments.count == 2
        Task {
            do {
                let result = try await app.enlarged(e, download: download)
                finish(c, ["ok": true, "image": json(result)])
            } catch EnlargeFailure.refused(let sentence) {
                finish(c, Response.failure("refused", sentence))
            } catch EnlargeFailure.downloadRequired {
                finish(c, Response.failure("download_required",
                                           String(localized: "macOS does not have Apple's enlargement model yet"),
                                           help: "silicontrol enlarge \(e.name) --download"))
            } catch EnlargeFailure.download {
                finish(c, Response.failure("download_failed", String(localized: "The enlargement model could not be downloaded")))
            } catch {
                finish(c, Response.failure("internal", String(localized: "This image could not be enlarged")))
            }
        }
    }

    // ── Tools ──

    /// `t3` → 3; `3` too.
    private static func ordinal(_ text: String, prefix: Character) -> Int? {
        Int(text.first == prefix ? String(text.dropFirst()) : text)
    }

    private func image(_ name: String) -> Entry? {
        guard let n = Self.ordinal(name, prefix: "i"), name.first == "i" else { return nil }
        return app.history.first { $0.ordinal == n }
    }

    private func createFolder(_ path: String) throws {
        do { try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true) } catch {
            let detail = error.localizedDescription
            throw Refusal("write", String(localized: "folder cannot be created: \(path) (\(detail))"))
        }
    }

    /// In a folder, under the app's export name (`AppState.exportName`: the model, the LoRAs, the steps
    /// if they are not the model's, the format, the seed). Never over an existing file (`-2`, `-3`…).
    private func write(_ e: Entry, in folder: String) throws -> String {
        let path = AppState.freeURL(in: URL(fileURLWithPath: folder), base: app.exportName(e), ext: "png").path
        try writeFile(e, path)
        return path
    }

    private func writeFile(_ e: Entry, _ path: String) throws {
        do { try app.write(e, to: URL(fileURLWithPath: path)) } catch {
            throw Refusal("write", "\(path): \(error.localizedDescription)")
        }
    }
}
