// The simpler remote-control commands (install, diagnose, subjects, cancel, set, save,
// vary, enlarge…), each answering one `silicontrol` verb. Split from `RemoteControl.swift`.

import AppKit
import Foundation
import Observation
import Siliconed
import SilicontrolHelp

extension RemoteControl {
    /// `install <id> [--standard|--compact|--light]`: **the app shows the model's license and the place it
    /// takes, and the user accepts there** — the command downloads nothing by itself. It opens the
    /// Models sheet on the model and returns at once; `models` then says "ready" once the
    /// installation is done. **Without a flag the version installed is kept** (Standard if none): a
    /// command that names no version never switches. `--standard`/`--compact`/`--light` preselect that
    /// version — the one installed is offered for replacement, like « Switch to … » in the sheet. An
    /// imported model's prerequisites (`baseDiT` false) never switch: a flag that contradicts the
    /// family's version installed is refused, with the reason.
    func install(_ arguments: [String]) throws -> [String: Any] {
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
        guard let card = b.cards().first(where: { $0.id == id || $0.family.rawValue == id }) else {
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
                          help: String(localized: "switch the publisher’s model first: \(command)"))
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

    /// `diagnose [--issue|--stop]`: **« Measure This Mac », from the terminal** — the sheet opens in the
    /// app and the diagnostic runs (~1 min, no weights, the queue held meanwhile); the answer is the
    /// report, the same JSON the issue carries. `--issue` then does what the sheet's
    /// « Open a GitHub Issue » does: GitHub's form opens in the browser, prefilled — the user submits it.
    func diagnose(_ arguments: [String], _ c: Connection) throws {
        // The sheet's « Stop », from another connection: the diagnose that waits on the measure answers `stopped`.
        if arguments == ["--stop"] {
            let running = app.diagnosticRunning
            app.cancelDiagnostic()
            finish(c, ["ok": true, "stopped": running])
            return
        }
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
                o["issue_url_length"] = (try? report.issueURL().absoluteString.count) ?? NSNull()
                if let notice = self.app.openIssue(report) { o["message"] = notice }
            }
            self.finish(c, o)
        }
    }

    // ── add ──

    /// The arguments of the developer's render command, plus `--wait` and `--out`.
    struct RenderOrder {
        var model = "z-image"
        var prompt: String?
        var format: (Int, Int)?
        var seed: UInt64?
        var batch = 1
        var steps: Int?
        /// `--detail more|most`: the rack's Detail choice (`Request.detail`).
        var detail = Detail.normal
        /// `--solver NAME`: the rack's Solver (`Request.solver`), More Solvers and `AppState.solverFamilies` only.
        var solver = Solver.euler
        /// `--accelerators`, `--no-accelerators` (`--ane`, `--no-ane`, their former names): the DiT box's switch (`Job.accelerators`);
        /// `nil`, the model's choice.
        var accelerators: Bool?
        var loras: [(name: String, strength: Double)] = []
        /// The `--ref`, in their order: the first is image 1, the one edited.
        var references: [String] = []
        /// `--image IMAGE|i7`: the starting image of an img2img (a model with "start_image": true); `--strength`: its strength,
        /// diffusers' (`AppState.defaultStartStrength` when left out).
        var image: String?
        var strength: Double?
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

    /// **`subjects`** (`Subjects.swift`): list, name, delete — the Prompt Lab's library, in the app's memory.
    /// Citing one is `add "@alice in a library"`.
    ///
    ///     subjects                                              the library
    ///     subjects add IMAGE --name N --kind K [--frame x,y,w,h] [--second IMAGE2]
    ///     subjects remove NAME
    func subjects(_ arguments: [String], cwd: String) throws -> [String: Any] {
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

    func cancel(_ arguments: [String]) throws -> [String: Any] {
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

    func clear() -> [String: Any] {
        let stopped = app.currentJob?.name
        let removedNames = (app.file + app.awaitingLicense).map(\.name)
        app.stopAll()
        return ["ok": true, "stopped": stopped ?? NSNull(), "removed": removedNames]
    }

    // ── set ──

    /// **`set faster-rendering|accelerators on|off [--model ID]` and `set beta [OPTION on|off]`: Settings' switches, from the
    /// terminal.** Each goes through the function its switch calls — `AppState.setFasterRendering(_:for:)`, `setAccelerators(_:for:)`,
    /// the Beta properties — so the choice is written and the window follows, exactly as a click. A model's choice is per
    /// installed version; without `--model`, the model in the rack. Only with the queue empty, so that a series is not started
    /// under one recipe and finished under the other (a job keeps the value it was queued with:
    /// `RenderSettings.fasterRendering`). Setting the value it already has is not a change. The Beta options are the window's
    /// switches, which have no such wait, and nor do these.
    func set(_ arguments: [String]) throws -> [String: Any] {
        let order: SetOrder
        switch SetOrder.parse(arguments) {
        case .success(let parsed): order = parsed
        case .failure(.usage(let line)): throw Refusal("usage", line, help: "silicontrol help set")
        case .failure(.unknownBeta(let name)):
            throw Refusal("usage", String(localized: "unknown Beta option: \"\(name)\""),
                          help: String(localized: "Beta options: \(BetaOption.names)"))
        }
        switch order {
        case .betaList: return ["ok": true, "beta": betaState()]
        case .beta(let option, let on): return setBeta(option, on)
        case .perModel(let kind, let on, let model):
            let id = model ?? app.identifier
            guard let card = app.cards.first(where: { $0.id == id }) else {
                let visible = visibleCards(app.library).map(\.id).joined(separator: ", ")
                throw Refusal("model", String(localized: "unknown model: \"\(id)\""), help: String(localized: "models: \(visible)"))
            }
            guard !app.busy, app.forgeInProgress == nil, !app.diagnosticRunning, app.awaitingLicense.isEmpty else {
                throw Refusal("busy", String(localized: "a render, an installation or the diagnostic is running"),
                              help: "silicontrol wait, then retry")
            }
            switch kind {
            case .fasterRendering:
                // A Beta option, like --solver's More Solvers: turning it on is refused while the option is off (off stays possible:
                // it is what a render takes anyway, and `measure` puts a value back through here).
                if on, !app.fasterRenderingBeta {
                    throw Refusal("faster_rendering", String(localized: "Faster Rendering is offered only when its Beta option is on"),
                                  help: String(localized: "turn on Faster Rendering in Settings (⌘,), under Advanced › Beta"))
                }
                let previous = app.fasterRendering(for: card.id)
                app.setFasterRendering(on, for: card.id)
                return ["ok": true, "model": card.id, "faster_rendering": app.fasterRendering(for: card.id), "previous": previous]
            case .accelerators:
                // The switch is greyed where this Mac is not measured yet, or its measure chooses no accelerator (`DiTSwitches`):
                // nothing to choose, either way.
                if let why = DiTSwitches.acceleratorsUnavailable(measured: app.settingsRead, offered: app.acceleratorsOffered) {
                    throw Refusal("accelerators", AcceleratorWords.unavailable(why), help: "silicontrol diagnose")
                }
                let previous = app.accelerators(for: card.id)
                app.setAccelerators(on, for: card.id)
                return ["ok": true, "model": card.id, "accelerators": app.accelerators(for: card.id), "previous": previous]
            }
        }
    }

    /// The four Beta options as the window's switches hold them (this Mac's preferences, shown by Settings whichever Mac computes).
    func betaState() -> [String: Any] {
        var o: [String: Any] = [:]
        // JSON keys are snake_case; the name typed after `set beta` keeps its hyphens.
        for option in BetaOption.allCases { o[option.key] = local.beta(option) }
        return o
    }

    /// **`set beta OPTION on|off`**: the property the switch is bound to, on this Mac's state — and on the one the window shows,
    /// when it computes elsewhere, whose copy of the preference would otherwise stay as it was at the switch (the rack there
    /// reads it). Each option's own effect is its property's (`AppState.developerMode`'s `didSet` takes a solver the models no
    /// longer offer out of the form; `negativeLoRAStrengths`' brings the strengths back to 0).
    func setBeta(_ option: BetaOption, _ on: Bool) -> [String: Any] {
        let previous = local.beta(option)
        local.setBeta(option, on)
        if app !== local { app.setBeta(option, on) }
        return ["ok": true, "option": option.rawValue, "enabled": local.beta(option), "previous": previous, "beta": betaState()]
    }

    // ── history, save ──

    func history(_ arguments: [String]) throws -> [String: Any] {
        var n = Int.max
        if let a = arguments.first {
            guard let v = Int(a), v >= 1 else { throw Refusal("usage", String(localized: "history [N]: N an integer ≥ 1, not \"\(a)\"")) }
            n = v
        }
        return ["ok": true, "images": app.history.prefix(n).map { json($0) }]
    }

    /// **`save i7 [i8 …] DEST [--format png|jpeg|heic]`**: « Save… » and « Export… » without the panel. One image to a file takes its format from the
    /// file's extension (.png, .jpg/.jpeg, .heic), as the panel's Format pop-up does; into a folder, each image under its export name, in `--format`
    /// (PNG by default). Only PNG keeps the prompt, the seed and the model.
    func save(_ arguments: [String], cwd: String) throws -> [String: Any] {
        var arguments = arguments
        var kind = ExportFormat.png
        if let i = arguments.firstIndex(of: "--format") {
            guard i + 1 < arguments.count else { throw Refusal("usage", String(localized: "--format expects a value")) }
            guard let k = ExportFormat(extension: arguments[i + 1]) else {
                let v = arguments[i + 1]
                throw Refusal("usage", String(localized: "--format expects png, jpeg or heic, not \"\(v)\""))
            }
            kind = k
            arguments.removeSubrange(i...(i + 1))
        }
        guard arguments.count >= 2, let destination = arguments.last else {
            throw Refusal("usage", "save <i7> [i8 …] <file.png|file.jpg|file.heic|folder> [--format png|jpeg|heic]")
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
        if entries.count == 1, ExportFormat(extension: ext) != nil {
            // A named file: written, or replaced — it is « Save as ».
            try createFolder((target as NSString).deletingLastPathComponent)
            try writeFile(entries[0], target)
            files.append(["image": entries[0].name, "path": target])
        } else {
            try createFolder(target)
            for e in entries { files.append(["image": e.name, "path": try write(e, in: target, as: kind)]) }
        }
        return ["ok": true, "files": files]
    }

    // ── vary ──

    /// **`vary i7 subtle|strong [--batch N]`**: « Variations » on a history image, as its menu does — N
    /// jobs (the rack's « Images » by default) with its settings and seed, each turned by a variation
    /// seed drawn at random.
    func vary(_ arguments: [String]) throws -> [String: Any] {
        let (e, amount, count) = try varyRequest(arguments)
        if let reason = app.libraryBlocker { throw Refusal("library_absent", reason, help: String(localized: "connect the disk, then retry")) }
        if let reason = app.installBlocker { throw Refusal("busy", reason, help: String(localized: "retry once the installation is done (silicontrol models)")) }
        guard app.canVary(e) else {
            throw Refusal("model", String(localized: "\(e.name) has no variations: a grid’s sheet, or its model is not ready"),
                          help: "silicontrol models")
        }
        guard app.file.count + count <= AppState.maxQueueLength else {
            throw Refusal("queue_full", String(localized: "the queue is full (\(AppState.maxQueueLength) jobs)"),
                          help: "silicontrol wait, then retry")
        }
        return ["ok": true, "jobs": app.vary(e, amount, count: count).map { json($0) }]
    }

    /// What `vary` is asked: the image, the amount, how many — read the same on this Mac and on a mirror.
    func varyRequest(_ arguments: [String]) throws -> (entry: Entry, amount: Variation.Amount, count: Int) {
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
        return (e, amount, count)
    }

    /// `enlarge i7 [--download]`: « Enlarge ×4 », answered once the enlarged image is in the history.
    func enlarge(_ arguments: [String], _ c: Connection) throws {
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
                                           String(localized: "macOS does not have Apple’s enlargement model yet"),
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
    static func ordinal(_ text: String, prefix: Character) -> Int? {
        Int(text.first == prefix ? String(text.dropFirst()) : text)
    }

    func image(_ name: String) -> Entry? {
        guard let n = Self.ordinal(name, prefix: "i"), name.first == "i" else { return nil }
        return app.history.first { $0.ordinal == n }
    }

    func createFolder(_ path: String) throws {
        do { try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true) } catch {
            let detail = error.localizedDescription
            throw Refusal("write", String(localized: "folder cannot be created: \(path) (\(detail))"))
        }
    }

    /// In a folder, under the app's export name (`AppState.exportName`: the model, the LoRAs, the steps
    /// if they are not the model's, the format, the seed). Never over an existing file (`-2`, `-3`…).
    func write(_ e: Entry, in folder: String, as kind: ExportFormat = .png) throws -> String {
        let path = AppState.freeURL(in: URL(fileURLWithPath: folder), base: app.exportName(e), ext: kind.fileExtension).path
        try writeFile(e, path)
        return path
    }

    private func writeFile(_ e: Entry, _ path: String) throws {
        do { try app.write(e, to: URL(fileURLWithPath: path)) } catch {
            throw Refusal("write", "\(path): \(error.localizedDescription)")
        }
    }
}
