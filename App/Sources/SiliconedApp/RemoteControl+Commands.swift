// The simpler remote-control commands (install, subjects, cancel, set, save,
// vary, enlarge…), each answering one `silicontrol` verb. Split from `RemoteControl.swift`.

import AppKit
import Foundation
import Observation
import Siliconed
import SilicontrolHelp

extension RemoteControl {
    // ── import, install ──

    /// **`import <file.safetensors>…`**: the window's import (`AppState.perform`). Every file is recognized before
    /// anything is copied, so that a refusal — not a model nor a LoRA of a family, a model whose family is not
    /// installed — comes back as the command's failure and not as a sentence in a window nobody looks at.
    func importing(_ arguments: [String], cwd: String) throws -> [String: Any] {
        guard !arguments.isEmpty, !arguments.contains(where: { $0.hasPrefix("--") }) else {
            throw Refusal("usage", String(localized: "import expects one or more .safetensors files"), help: "silicontrol help import")
        }
        let urls = arguments.map { URL(fileURLWithPath: absolute($0, cwd)) }
        if let other = urls.first(where: { !AppState.importable($0) }) {
            throw Refusal("usage", String(localized: "not a .safetensors file: \(other.lastPathComponent)"), help: "silicontrol help import")
        }
        guard app.managementPossible else {
            throw Refusal("busy", String(localized: "a render or an installation is running"), help: "silicontrol wait, then retry")
        }
        let installer = Installer(library: app.library)
        var files: [[String: Any]] = []
        for url in urls {
            do {
                let (kind, family) = try ModelImport.recognize(url.path)
                if kind == .model, installer.installed(family).isEmpty {
                    throw EngineError.importNeedsFamily(file: url.lastPathComponent, model: family.rawValue)
                }
                files.append(["file": url.lastPathComponent, "kind": kind.rawValue, "model": ModelCard.of(family).id])
            } catch {
                let problem = Problem(error)
                throw Refusal(problem.code ?? "import_refused", problem.text)
            }
        }
        app.perform(urls)
        return ["ok": true, "importing": files]
    }

    /// `install <id> [--standard|--compact]`: **the app shows the model's license and the place it
    /// takes, and the user accepts there** — the command downloads nothing by itself. It opens the
    /// Models sheet on the model and returns at once; `models` then says "ready" once the
    /// installation is done. **Without a flag the version rendered is kept** (the Compact if none is
    /// installed): a command that names no version never switches. `--standard`/`--compact` name a
    /// version: whole on disk beside the one rendered, the family renders with it at once, like
    /// « Switch to … » in the sheet; else its installation is offered.
    func install(_ arguments: [String]) throws -> [String: Any] {
        let options = arguments.filter { $0.hasPrefix("--") }
        let flags = Variant.allCases.map { "--" + $0.rawValue }
        if let unknown = options.first(where: { !flags.contains($0) }) {
            throw Refusal("usage", String(localized: "unknown flag: \(unknown)"), help: "silicontrol help install")
        }
        let named = Variant.allCases.filter { options.contains("--" + $0.rawValue) }
        if named.count > 1 {
            throw Refusal("usage", String(localized: "one version at most: --standard or --compact"), help: "silicontrol help install")
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
        let part: Library.Occupancy.FamilyPart? = app.library.part(card.family)
        // No version named: the one rendered, else the one the app preselects (the Compact where there is one).
        let variant = asked ?? part?.variant ?? card.family.preselectedVariant
        // Whole on disk and the one rendered: nothing to do.
        if part?.variant == variant {
            return ["ok": true, "model": card.id, "installed": true, "variant": variant.rawValue,
                    "message": String(localized: "already installed")]
        }
        guard app.managementPossible else {
            throw Refusal("busy", String(localized: "a render or an installation is running"),
                        help: "silicontrol wait, then retry")
        }
        // Whole on disk beside the one rendered: the sheet's « Switch to … » renders with it at once.
        if part?.installed.contains(variant) == true {
            app.problem = nil
            app.use(card.family, variant: variant)
            if let p = app.problem { throw Refusal(p.code ?? "write", p.text) }
            return ["ok": true, "model": card.id, "installed": true, "variant": variant.rawValue]
        }
        app.offerInstall(card.family, variant: variant)
        NSApp.activate()
        return ["ok": true, "model": card.id, "installed": false, "variant": variant.rawValue,
                "message": String(localized: "the app shows the license and the space on disk: the user accepts there")]
    }

    // ── add, grid: the order ──

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

    // ── cancel, clear ──

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

    // ── compiled ──

    /// **`compiled [erase]`: Settings ▸ Macs ▸ This Mac's « Compiled for This Mac »**: what Core AI compiled for this Mac
    /// weighs (`CoreAI.compiled`, read off the main thread); `erase` has the engine erase it, as « Erase… » does, and
    /// answers once it is done — after the render under way, if one is.
    func compiled(_ arguments: [String], _ c: Connection) throws {
        switch arguments {
        case []:
            Task { finish(c, ["ok": true, "compiled_bytes": await app.weighCompiled(), "erasing": app.erasingCompiled]) }
        case ["erase"]:
            guard !app.erasingCompiled else {
                throw Refusal("busy", String(localized: "an erasure of the compilations is already under way"), help: "silicontrol compiled, then retry")
            }
            app.eraseCompiled { [weak self] result in
                guard let self else { return }
                switch result {
                case .success(let freed):
                    finish(c, ["ok": true, "freed_bytes": freed, "compiled_bytes": app.compiledBytes ?? 0])
                case .failure(let e):
                    let error = EngineError(e)
                    finish(c, Response.failure(error.code, error.localizedDescription))
                }
            }
        default:
            throw Refusal("usage", String(localized: "compiled expects nothing, or erase"), help: "silicontrol help compiled")
        }
    }

    // ── set ──

    /// **`set beta [OPTION on|off]`: Settings' switches, and the DiT box's « Preview every step », from the terminal.** Each goes through the property its switch is
    /// bound to, so the choice is written and the window follows, exactly as a click. Setting the value it already has is not a
    /// change. `set save-images FOLDER|off` is Settings ▸ General's Save Every Image To… (`EveryImage`), this Mac's: an
    /// existing folder, never created, as the panel only offers one.
    func set(_ arguments: [String], cwd: String) throws -> [String: Any] {
        // `neural-engine` is the app's own preference: the library's `SetOrder` does not know it.
        if arguments.first == "neural-engine" { return try setNeuralEngine(Array(arguments.dropFirst())) }
        if arguments.first == "glass" { return try setGlass(Array(arguments.dropFirst())) }
        if arguments.first == "updates" { return try setUpdates(Array(arguments.dropFirst())) }
        if arguments.first == "memory" { return try setMemory(Array(arguments.dropFirst())) }
        if arguments.first == "preview" { return try setPreview(Array(arguments.dropFirst())) }
        let order: SetOrder
        switch SetOrder.parse(arguments) {
        case .success(let parsed): order = parsed
        case .failure(.usage(let line)): throw Refusal("usage", line, help: "silicontrol help set")
        case .failure(.unknownBeta(let name)):
            throw Refusal("usage", String(localized: "unknown Beta option: \"\(name)\""),
                          help: String(localized: "Beta options: \(BetaOption.names)"))
        }
        switch order {
        case .saveImagesState: return ["ok": true, "save_images": EveryImage.shared.folder?.path ?? NSNull()]
        case .saveImages(let typed):
            let previous: Any = EveryImage.shared.folder?.path ?? NSNull()
            if let typed {
                let path = absolute(typed, cwd)
                var isFolder: ObjCBool = false
                guard FileManager.default.fileExists(atPath: path, isDirectory: &isFolder), isFolder.boolValue else {
                    throw Refusal("write", String(localized: "not a folder: \(path)"), help: String(localized: "give an existing folder"))
                }
                EveryImage.shared.folder = URL(fileURLWithPath: path, isDirectory: true)
            } else {
                EveryImage.shared.folder = nil
            }
            return ["ok": true, "save_images": EveryImage.shared.folder?.path ?? NSNull(), "previous": previous]
        case .betaList: return ["ok": true, "beta": betaState()]
        case .beta(let option, let on): return setBeta(option, on)
        }
    }

    /// **`set glass [on|off]`**: Settings ▸ General's Glass, as a click; alone, it is read.
    func setGlass(_ arguments: [String]) throws -> [String: Any] {
        let previous = local.glass
        if let typed = arguments.first {
            guard arguments.count == 1, typed == "on" || typed == "off" else {
                throw Refusal("usage", String(localized: "set glass [on|off]"), help: "silicontrol help set")
            }
            local.glass = typed == "on"
        }
        return ["ok": true, "glass": local.glass, "previous": previous]
    }

    /// **`set preview [on|off]`**: the DiT box's « Preview every step », as a click — the window's state, this Mac's or the one it
    /// computes on, whose renders it changes from the next one on (`add --preview` and `--no-preview` still decide for one job). Alone,
    /// it is read.
    func setPreview(_ arguments: [String]) throws -> [String: Any] {
        let previous = app.previews
        if let typed = arguments.first {
            guard arguments.count == 1, typed == "on" || typed == "off" else {
                throw Refusal("usage", String(localized: "set preview [on|off]"), help: "silicontrol help set")
            }
            app.previews = typed == "on"
        }
        return ["ok": true, "preview": app.previews, "previous": previous]
    }

    /// **`set updates [on|off]`**: Settings ▸ General's Check at Launch, as a click; alone, it is read.
    func setUpdates(_ arguments: [String]) throws -> [String: Any] {
        let previous = Updates.shared.atLaunch
        if let typed = arguments.first {
            guard arguments.count == 1, typed == "on" || typed == "off" else {
                throw Refusal("usage", String(localized: "set updates [on|off]"), help: "silicontrol help set")
            }
            Updates.shared.atLaunch = typed == "on"
        }
        return ["ok": true, "updates": Updates.shared.atLaunch, "previous": previous]
    }

    /// **`set memory [GB|default]`**: Settings ▸ Macs ▸ This Mac's Memory Limit — B, in gigabytes (GiB), within this
    /// Mac's range (`MemoryBudget.range`), through the property its slider is bound to; alone, it is read. The engine's
    /// next process takes it.
    func setMemory(_ arguments: [String]) throws -> [String: Any] {
        let previous = memoryState()
        if let typed = arguments.first {
            let range = MemoryBudget.range
            guard arguments.count == 1 else {
                throw Refusal("usage", String(localized: "set memory [GB|default]"), help: "silicontrol help set")
            }
            if typed == "default" {
                local.memoryBudget = nil
            } else {
                guard let gb = Double(typed), gb.isFinite, Int(gb * Double(1 << 30)) >= range.lowerBound,
                      Int(gb * Double(1 << 30)) <= range.upperBound else {
                    throw Refusal("usage", String(localized: "set memory [GB|default]: from \(range.lowerBound >> 30) to \(range.upperBound >> 30) on this Mac"),
                                  help: "silicontrol help set")
                }
                local.memoryBudget = Int(gb * Double(1 << 30))
            }
        }
        return ["ok": true, "memory": memoryState(), "previous": previous]
    }

    /// B as Settings shows it: what it is, whether the user chose it, the default and the range.
    func memoryState() -> [String: Any] {
        let range = MemoryBudget.range
        return ["budget_bytes": MemoryBudget.budget, "chosen": local.memoryBudget != nil,
                "default_bytes": MemoryBudget.standard, "range_bytes": [range.lowerBound, range.upperBound]]
    }

    /// **`set neural-engine [on|off]`**: Settings ▸ General's Neural Engine switch, through the property it is bound to (it
    /// writes the choice and tells the engine). This Mac's preference, for what this Mac computes: a render it lends follows
    /// it, one it borrows follows the lender's. Alone, it is read.
    func setNeuralEngine(_ arguments: [String]) throws -> [String: Any] {
        let previous = local.neuralEngine
        if let typed = arguments.first {
            guard arguments.count == 1, typed == "on" || typed == "off" else {
                throw Refusal("usage", String(localized: "set neural-engine [on|off]"), help: "silicontrol help set")
            }
            local.neuralEngine = typed == "on"
        }
        return ["ok": true, "neural_engine": local.neuralEngine, "previous": previous]
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
    /// jobs (the rack's « Seeds » by default) with its settings and seed, each turned by a variation
    /// seed drawn at random.
    func vary(_ arguments: [String]) throws -> [String: Any] {
        let (e, amount, count) = try varyRequest(arguments)
        if let reason = app.libraryBlocker { throw Refusal("library_absent", reason, help: String(localized: "connect the disk, then retry")) }
        if let reason = app.installBlocker { throw Refusal("busy", reason, help: String(localized: "retry once the installation is done (silicontrol models)")) }
        guard app.canVary(e) else {
            throw Refusal("model", String(localized: "\(e.name) has no variations: an enlargement or a crop, or its model is not ready"),
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
