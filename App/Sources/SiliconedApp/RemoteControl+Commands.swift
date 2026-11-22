// The remote control's commands of the library and the queue (import, install, subjects, cancel, clear, compiled),
// each answering one `silicontrol` verb.

import AppKit
import Foundation
import Measurement
import Observation
import Siliconed
import SilicontrolHelp

extension RemoteControl {
    // ── import, install ──

    /// **`import <file.safetensors>…`**: the window's import (`AppState.perform`). Every file is recognized before
    /// anything is copied, so that a refusal — not a model nor a LoRA of a family, a model whose family is not
    /// installed — comes back as the command's failure and not as a sentence in a window nobody looks at.
    func importing(_ arguments: [String], cwd: String) throws -> [String: Any] {
        // A file whose name begins with `--` goes after a lone `--`.
        let paths = try Self.flags(arguments).allWords
        guard !paths.isEmpty else {
            throw Refusal("usage", String(localized: "import expects one or more .safetensors files"), help: "silicontrol help import")
        }
        let urls = paths.map { URL(fileURLWithPath: absolute($0, cwd)) }
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
        let flags = try Self.flags(arguments, switches: Set(Variant.allCases.map { "--" + $0.rawValue }))
        let named = Variant.allCases.filter { flags.has("--" + $0.rawValue) }
        if named.count > 1 {
            throw Refusal("usage", String(localized: "one version at most: --standard or --compact"), help: "silicontrol help install")
        }
        let ids = flags.allWords
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
            let flags = try Self.flags(Array(arguments.dropFirst()), valued: ["--name", "--kind", "--frame", "--second"])
            let name = flags.value("--name")
            var kind: NamedSubjects.Kind?
            if let v = flags.value("--kind") {
                guard let k = NamedSubjects.Kind(rawValue: v.lowercased()) else {
                    let kinds = NamedSubjects.Kind.allCases.map(\.rawValue).joined(separator: ", ")
                    throw Refusal("usage", String(localized: "--kind expects one of \(kinds), not \"\(v)\""))
                }
                kind = k
            }
            let frames = try (flags.values["--frame"] ?? []).map { v -> NamedSubjects.Frame in
                let n = v.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
                guard n.count == 4, n.allSatisfy({ (0...1).contains($0) }), n[2] > 0, n[3] > 0 else {
                    throw Refusal("usage", String(localized: "--frame expects x,y,width,height in fractions of the image, not \"\(v)\""))
                }
                return NamedSubjects.Frame(x: n[0], y: n[1], width: n[2], height: n[3])
            }
            // One IMAGE, first wherever it is written; a second one is `--second`'s.
            guard flags.allWords.count <= 1 else {
                throw Refusal("usage", String(localized: "subjects add IMAGE --name NAME --kind KIND"), help: "silicontrol help subjects")
            }
            let first = flags.allWords.first
            let images = (first.map { [$0] } ?? []) + (flags.values["--second"] ?? [])
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
        let ordinals = try Self.flags(arguments).allWords.map { a -> Int in
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
        switch try Self.flags(arguments).allWords {
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
}
