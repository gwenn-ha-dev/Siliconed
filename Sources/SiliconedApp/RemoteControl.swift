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
    /// This Mac's state, whose library holds the socket.
    @ObservationIgnored let local: AppState
    /// What the window shows: another Mac's mirror once switched (`RemoteControl+Mirror.swift`), else this Mac.
    var app: AppState { MainWindow.shared.carrier.mirror ?? local }
    @ObservationIgnored private var source: DispatchSourceRead?
    /// Why the remote control is not listening (another Siliconed, a path too long), or `nil` —
    /// shown greyed in the Siliconed menu, under « Install the silicontrol command… ».
    private(set) var failureMessage: String?

    init(app: AppState) { local = app }

    var path: String { local.library.socket }

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

    func finish(_ c: Connection, _ object: [String: Any]) {
        c.send(Response.rowLine(object))
        c.shutDown()
    }

    func emit(_ c: Connection, _ object: [String: Any]) { c.send(Response.rowLine(object)) }

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
            try refuseOnMirror(command)
            switch command {
            case "status": finish(c, state())
            case "models": finish(c, app.mirror == nil ? models() : mirroredModels())
            case "install": finish(c, try install(arguments))
            case "diagnose": try diagnose(arguments, c)
            case "add":
                if let mirror = app.mirror { try addOnMirror(mirror, arguments, cwd: cwd, c) } else { try add(arguments, cwd: cwd, c) }
            case "grid":
                if let mirror = app.mirror { try addOnMirror(mirror, arguments, cwd: cwd, c, grid: true) } else { try add(arguments, cwd: cwd, c, grid: true) }
            case "wait": try wait(arguments, cwd: cwd, c)
            case "follow": follow(c)
            case "cancel": finish(c, try cancel(arguments))
            case "clear": finish(c, clear())
            case "set":
                if let mirror = app.mirror { try setOnMirror(mirror, arguments, c) } else { finish(c, try set(arguments)) }
            case "history": finish(c, try history(arguments))
            case "save": finish(c, try save(arguments, cwd: cwd))
            case "vary":
                if let mirror = app.mirror { try varyOnMirror(mirror, arguments, c) } else { finish(c, try vary(arguments)) }
            case "enlarge":
                if let mirror = app.mirror { try enlargeOnMirror(mirror, arguments, c) } else { try enlarge(arguments, c) }
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

    func json(_ t: Job) -> [String: Any] {
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

    func json(_ e: Entry, path: String? = nil) -> [String: Any] {
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
    static func json(_ issue: Issue, job n: Int, images: [Int], warnings: [String] = []) -> [String: Any] {
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
                                "faster_rendering": app.fasterRenderingShown]
        // The settings renders run on: a series compared across launches compares maps too.
        o["settings"] = ["read": app.settingsRead, "map": app.mapFacts.inUse?.date ?? NSNull(),
                         "provisional": app.mapFacts.inUse?.provisional ?? NSNull(),
                         "remeasuring": app.mapFacts.remeasuring,
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
        o["computing_on"] = app.mirror?.macName ?? NSNull()
        return o
    }

    static func stageName(_ item: Engine.Stage) -> String {
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
    func visibleCards(_ b: Library) -> [ModelCard] {
        AppState.inDisplayOrder(b.cards().filter { app.isVisible($0.family) }, family: \.family)
    }

    // ── install ──
}
