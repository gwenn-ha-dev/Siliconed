// **The window's other gestures, from the terminal** — the history's (`remove`, `undo`, `again`, `reuse`, `crop`), the series'
// (`sheet`) and the models' (`import`, `uninstall`, `install stop`). Each calls the function the button or the menu item calls
// (`AppState.delete`, the window's `UndoManager`, `redo`, `restoreSettings`, `addSheet`, `perform`, `uninstall`, `remove`,
// `cancelForge`), with the guard the window puts on it: a gesture that is greyed out in the window is a refusal here, with the same
// sentence. The arguments are read in the library (`GestureOrder`), where the tests read them. Compute On and the library are in
// `RemoteControl+Place.swift`.
//
// What destroys something (`uninstall`) acts only with `--yes`; without it, the answer says what would go and the room it frees.

import AppKit
import Foundation
import Siliconed

extension RemoteControl {
    /// The order, read — or the usage refusal the library's failure becomes.
    func order(_ command: String, _ arguments: [String]) throws -> GestureOrder {
        switch GestureOrder.parse(command, arguments) {
        case .success(let order): return order
        case .failure(.usage(let line)): throw Refusal("usage", line, help: "silicontrol help \(command)")
        case .failure(.unknownFlag(let flag)):
            throw Refusal("usage", String(localized: "unknown flag: \(flag)"), help: "silicontrol help \(command)")
        case .failure(.flagWithoutValue(let flag)): throw Refusal("usage", String(localized: "\(flag) expects a value"))
        case .failure(.badValue(let what, let value)):
            switch what {
            case "cell": throw Refusal("usage", String(localized: "cell expected (g2c3 or t5), not \"\(value)\""), help: "silicontrol help finish")
            case "grid": throw Refusal("usage", String(localized: "grid expected (g2), not \"\(value)\""), help: "silicontrol help finish")
            case "--strength":
                throw Refusal("usage", String(localized: "--strength expects a number above 0 and at most 1 (0.65 or 65%), not \"\(value)\""),
                              help: "silicontrol help start-image")
            default:
                throw Refusal("usage", String(localized: "\(what) does not read: \"\(value)\""), help: "silicontrol help \(command)")
            }
        }
    }

    /// **Routes an order to the gesture it names.**
    func gesture(_ order: GestureOrder, cwd: String, _ c: Connection) throws {
        switch order {
        case .remove(let names): finish(c, try removeImages(names))
        case .undo(let redo): finish(c, try undo(redo: redo))
        case .again(let word): try again(word, cwd: cwd, c)
        case .reuse(let word): finish(c, try reuse(word, cwd: cwd))
        case .crop(let image, let spec): try crop(image, spec, c)
        case .sheet(let grid): try addSheet(grid: grid, c)
        case .importFiles(let paths): try importFiles(paths, cwd: cwd, c)
        case .uninstall(let what, let confirmed): finish(c, try uninstall(what, confirmed: confirmed))
        case .macs(let action): finish(c, try macs(action))
        case .compute(let target): try compute(target, c)
        case .library(let action, let confirmed): try library(action, confirmed: confirmed, cwd: cwd, c)
        case .startImage(let action): finish(c, try startImage(action, cwd: cwd))
        case .lab(let action): try lab(action, cwd: cwd, c)
        case .settings(let action): finish(c, try settings(action))
        case .bracket(let action): finish(c, try bracket(action))
        case .rack(let place): finish(c, rack(place))
        }
    }

    // MARK: - A knob's bubble, the rack's scroll

    /// **`bracket KNOB`**: a click on that knob's ± (or right-click › « Bracket… »), its bubble open beside it — refused where the
    /// window draws no ± that would open (Detail on a model without it, no starting image, « Compare LoRAs » with nothing to
    /// compare, a LoRA not in the stack or switched off). **`bracket close`**: the bubble closed, its values left as they are.
    func bracket(_ action: GestureOrder.BracketAction) throws -> [String: Any] {
        guard case .open(let knob) = action else {
            app.bracketing = nil
            return ["ok": true, "bracketing": NSNull()]
        }
        let key: SeriesKey, name: String
        switch knob {
        case .seeds: (key, name) = (.seeds, "seed")
        case .detail: (key, name) = (.detail, "detail")
        case .solver: (key, name) = (.solver, "solver")
        case .start: (key, name) = (.startStrength, "start")
        case .compareLoRAs: (key, name) = (.comparedLoRAs, "loras")
        case .lora(let word):
            let named = app.loras.filter { !$0.path.isEmpty }
            guard let slot = named.first(where: { $0.path == word || Self.shortName($0.path) == word || app.loraName($0.path) == word }) else {
                let list = named.map { Self.shortName($0.path) }
                throw Refusal("lora", String(localized: "No LoRA \"\(word)\" in the rack's stack") + (list.isEmpty ? "" : ": " + list.joined(separator: ", ")))
            }
            (key, name) = (.loraStrength(slot.path), "lora " + Self.shortName(slot.path))
        }
        guard app.applies(key), app.canBracket(key) else {
            throw Refusal("bracket", String(localized: "This setting has no ± to open under the current model and settings."), help: "silicontrol help bracket")
        }
        app.bracketing = key
        return ["ok": true, "bracketing": name]
    }

    /// **`rack top|dit`**: the rack's column scrolled to its head, or to the DiT's box — the chain then pinned in small at the top of
    /// the column, as when the hand scrolls down to the denoiser's settings.
    func rack(_ place: GestureOrder.RackPlace) -> [String: Any] {
        app.rackScroll = (place, (app.rackScroll?.count ?? 0) + 1)
        return ["ok": true, "rack": place.rawValue]
    }

    // MARK: - The history

    /// **`remove i7 [i8 …]`**: « Remove from History » — undone by `undo`, as ⌘Z undoes it.
    func removeImages(_ names: [String]) throws -> [String: Any] {
        var entries: [Entry] = []
        for name in names {
            guard let e = image(name) else {
                throw Refusal("unknown_image", String(localized: "no image \"\(name)\" in the history"), help: "silicontrol history")
            }
            if !entries.contains(where: { $0.id == e.id }) { entries.append(e) }
        }
        app.delete(entries)
        return ["ok": true, "removed": entries.map(\.name), "history_images": app.history.count, "undo": undoState()]
    }

    /// What ⌘Z and ⇧⌘Z would do now: the window's own undo manager, whose actions are removals, crops and a prompt carried to the rack.
    func undoState() -> [String: Any] {
        let m = app.undoManager
        return ["can_undo": m?.canUndo ?? false, "action": m?.canUndo == true ? m?.undoActionName ?? "" : "",
                "can_redo": m?.canRedo ?? false, "redo_action": m?.canRedo == true ? m?.redoActionName ?? "" : ""]
    }

    /// **`undo [--redo]`**: Edit › Undo (⌘Z) or Redo (⇧⌘Z) — whatever the window would undo, nothing more.
    func undo(redo: Bool) throws -> [String: Any] {
        guard let m = app.undoManager, redo ? m.canRedo : m.canUndo else {
            throw Refusal("nothing_to_undo", redo ? String(localized: "nothing to redo") : String(localized: "nothing to undo"),
                          help: "silicontrol help undo")
        }
        let name = redo ? m.redoActionName : m.undoActionName
        if redo { m.redo() } else { m.undo() }
        return ["ok": true, "done": redo ? "redo" : "undo", "action": name, "history_images": app.history.count, "undo": undoState()]
    }

    /// An image of the history, or a Siliconed PNG on disk (`again` and `reuse` take both, as the window does: the image's menu, or a PNG dropped).
    private enum Source { case entry(Entry), file(URL) }

    private func source(_ word: String, cwd: String) throws -> Source {
        if word.first == "i", Int(word.dropFirst()) != nil {
            guard let e = image(word) else {
                throw Refusal("unknown_image", String(localized: "no image \"\(word)\" in the history"), help: "silicontrol history")
            }
            return .entry(e)
        }
        return .file(URL(fileURLWithPath: absolute(word, cwd)))
    }

    /// **`reuse <i7|file.png>`**: « Reuse These Settings » — the rack takes the image's model, prompt, size, steps, LoRAs and seed. Or a PNG
    /// Siliconed wrote, as when it is dropped on the canvas: what could not come back is said in `notes`.
    func reuse(_ word: String, cwd: String) throws -> [String: Any] {
        var notes: String?
        switch try source(word, cwd: cwd) {
        case .entry(let e): app.restoreSettings(e)
        case .file(let url): notes = try restore(fromPNG: url)
        }
        var o: [String: Any] = ["ok": true, "rack": rackSettings()]
        if let notes { o["notes"] = notes }
        return o
    }

    /// The PNG dropped: `restoreSettings(fromPNG:)`, and the notes it leaves in the window's alert.
    private func restore(fromPNG url: URL) throws -> String? {
        let before = app.problem?.title
        guard app.restoreSettings(fromPNG: url) else {
            let path = url.path
            throw Refusal("ref", String(localized: "not a PNG that Siliconed wrote, or unreadable: \(path)"))
        }
        guard let p = app.problem, p.title != before else { return nil }
        return p.suggestion
    }

    /// What the rack holds after a gesture that fills it.
    func rackSettings() -> [String: Any] {
        ["model": app.identifier, "prompt": app.prompt, "steps": app.steps,
         "width": app.format.width, "height": app.format.height,
         "seed": app.fixedSeed ? NSNumber(value: app.seed) : NSNull(),
         "loras": app.loras.filter { !$0.path.isEmpty }.map { ["lora": Self.shortName($0.path), "strength": Response.number($0.strength), "on": $0.on] },
         "start_image": startJSON() ?? NSNull(), "series": seriesJSON()]
    }

    /// **The rack's series** — what « Generate » varies: each axis as `grid` reads it, and what is beyond two.
    private func seriesJSON() -> [String: Any] {
        let layout = app.seriesLayout
        let names = app.activeLoRAs.map { Self.shortName($0.path) }
        func axis(_ k: SeriesKey) -> String? { app.gridAxis(k).map { GridWords.command($0, loras: names) } }
        return ["x": layout.columns.flatMap(axis) ?? NSNull(), "y": layout.rows.flatMap(axis) ?? NSNull(),
                "excess": layout.excess.compactMap(axis), "renders": app.seriesRenders]
    }

    /// **`again <i7|file.png>`**: « Render Again (Same Seed) » (⌘R) — `redo` for an image of the history, which fills the rack with its
    /// settings and renders at its seed; for a PNG Siliconed wrote, the file's recipe the way a drop reads it, then « Generate ». An image
    /// made another day comes out the same, bit for bit, if the model and the LoRAs are the ones it was made with.
    func again(_ word: String, cwd: String, _ c: Connection) throws {
        let gesture: () -> Void
        var sheet = false
        switch try source(word, cwd: cwd) {
        case .entry(let e):
            sheet = e.grid != nil
            gesture = { self.app.redo(e) }
        case .file(let url):
            _ = try restore(fromPNG: url)
            gesture = { self.app.render() }
        }
        if sheet, app.mirror != nil {
            // The other Mac makes the cells of a sheet's series: its answer is not a job's.
            gesture()
            finish(c, ["ok": true, "message": String(localized: "the series is asked of the other Mac"), "computing_on": app.mirror?.macName ?? NSNull()])
            return
        }
        queueing(c, gesture)
    }

    /// **A gesture that queues jobs, answered with them**: those the function queued, or the refusal the window would have shown greyed out.
    /// On a mirror each job is answered by the other Mac (`RemoteAnswer`), as `vary` does.
    func queueing(_ c: Connection, _ gesture: () -> Void) {
        if let mirror = app.mirror {
            let first = mirror.nextRequest
            gesture()
            let count = mirror.nextRequest - first
            guard count > 0 else { finish(c, renderRefusal()); return }
            answerQueued(mirror, from: first, count: count, c)
        } else {
            let before = app.lastJob
            gesture()
            let jobs = ([app.currentJob].compactMap { $0 } + app.file).filter { $0.ordinal > before }
            guard !jobs.isEmpty else { finish(c, renderRefusal()); return }
            finish(c, ["ok": true, "jobs": jobs.map { json($0) }])
        }
    }

    /// Why nothing was queued: what greys « Generate » out, with the code the same reason has in `add`.
    private func renderRefusal() -> [String: Any] {
        let text = app.blocker ?? String(localized: "nothing was queued")
        let code = app.linkBlocker != nil ? "link" : app.libraryBlocker != nil ? "library_absent" : app.installBlocker != nil ? "busy"
            : app.missing != nil ? "model" : app.file.count >= AppState.maxQueueLength ? "queue_full" : "refused"
        return Response.failure(code, text, help: code == "refused" ? "silicontrol help again" : nil)
    }

    /// **`crop <i7> --box X,Y,W,H | --ratio P:Q`**: « Crop… », Return: the frame's pixels become a new image at the head of the history, the
    /// original stays (`undo` removes the crop). The frame obeys the editor's rules (`Crop.Frame`): never under `Format.minimumSide` a side,
    /// never the whole image, and not a grid's sheet.
    func crop(_ image: String, _ spec: CropSpec, _ c: Connection) throws {
        guard let e = self.image(image) else {
            throw Refusal("unknown_image", String(localized: "no image \"\(image)\" in the history"), help: "silicontrol history")
        }
        if let why = app.cropRefusal(e) { throw Refusal("crop", why) }
        if app.cropping != nil { throw Refusal("busy", String(localized: "the crop editor is open in the window"), help: String(localized: "finish or cancel the crop there")) }
        let box: Crop.Box
        switch spec.resolve(width: e.width, height: e.height) {
        case .success(let b): box = b
        case .failure(.whole):
            throw Refusal("crop", String(localized: "the whole image is not a crop"))
        case .failure(.outside):
            let w = String(e.width), h = String(e.height)
            throw Refusal("crop", String(localized: "the frame leaves the image (\(w) × \(h))"))
        case .failure(.belowFloor(let side)):
            let px = String(side)
            throw Refusal("crop", String(localized: "a side of the frame is under \(px) px"))
        case .failure(.cannotCrop(let side)):
            let px = String(side)
            throw Refusal("crop", String(localized: "this image cannot be cropped: both sides must be \(px) px or more, and one above"))
        case .failure(.unsupportedRatio):
            throw Refusal("crop", String(localized: "this image cannot hold that proportion at 512 px or more"))
        }
        let before = app.lastImage
        let mirror = app.mirror
        Task {
            await app.commitCrop(e, box: box)
            // The other Mac cuts its own pixels: its image comes through the history's sync.
            let deadline = Date().addingTimeInterval(mirror == nil ? 15 : 60)
            while app.history.first(where: { $0.cropped != nil && $0.ordinal > before }) == nil, Date() < deadline, !c.broken {
                try? await Task.sleep(for: .milliseconds(100))
            }
            guard let made = app.history.first(where: { $0.cropped != nil && $0.ordinal > before }) else {
                finish(c, Response.failure("crop", String(localized: "This image could not be cropped")))
                return
            }
            var o = json(made)
            o["crop"] = ["x": box.x, "y": box.y, "width": box.width, "height": box.height]
            finish(c, ["ok": true, "image": o, "undo": undoState()])
        }
    }

    // MARK: - The series

    /// **`sheet g2`**: « Add Sheet to History » — the series' images side by side, labelled, an image of the history. Answered once it is there.
    func addSheet(grid: Int, _ c: Connection) throws {
        guard let run = app.explored, run.ordinal == grid else {
            let shown = app.explored?.name, wanted = "g\(grid)"
            throw Refusal("unknown_job", shown.map { String(localized: "\(wanted) is not the series shown (\($0))") } ?? String(localized: "no series is shown"),
                          help: "silicontrol status")
        }
        // The window greys the button while cells are on their way or none rendered.
        if run.entries.isEmpty { throw Refusal("cell", String(localized: "no cell has a picture yet, so there is no sheet to draw")) }
        if app.exploring { throw Refusal("busy", String(localized: "cells are still on their way"), help: "silicontrol wait g\(grid)") }
        let known = Set(app.history.filter { $0.grid?.ordinal == grid }.map(\.id))
        app.addSheet(run)
        Task {
            let deadline = Date().addingTimeInterval(120)
            while app.history.first(where: { $0.grid?.ordinal == grid && !known.contains($0.id) }) == nil, Date() < deadline, !c.broken {
                try? await Task.sleep(for: .milliseconds(150))
            }
            guard let sheet = app.history.first(where: { $0.grid?.ordinal == grid && !known.contains($0.id) }) else {
                finish(c, Response.failure("render", String(localized: "the sheet was not drawn")))
                return
            }
            finish(c, ["ok": true, "image": json(sheet)])
        }
    }

    // MARK: - The models

    /// **`import FILE …`**: « Import a Model or LoRA… » (⌘O), the files given instead of chosen. The same guard (`importPossible`), the same
    /// engine, the same result in the library. **A model whose family is not installed** opens the Models sheet on that family, with its
    /// license and the space it takes, and imports once the user has accepted there — the command answers at once, `"needs_family"` —
    /// it never accepts for anyone. On a mirror the files would go up to the other Mac, whose owner confirms: refused here (`remote`).
    func importFiles(_ paths: [String], cwd: String, _ c: Connection) throws {
        let urls = paths.map { URL(fileURLWithPath: absolute($0, cwd)) }
        for url in urls {
            let path = url.path
            guard FileManager.default.isReadableFile(atPath: path) else { throw Refusal("ref", String(localized: "file unreadable or missing: \(path)")) }
            guard AppState.importable(url) else {
                throw Refusal("import", String(localized: "not a .safetensors or a .gguf: \(path)"), help: "silicontrol help import")
            }
        }
        if let reason = app.libraryBlocker { throw Refusal("library_absent", reason, help: String(localized: "connect the disk, then retry")) }
        guard app.importPossible else {
            throw Refusal("busy", String(localized: "a render, an installation or the diagnostic is running"), help: "silicontrol wait, then retry")
        }
        let before = libraryContents()
        app.problem = nil
        app.perform(urls)
        NSApp.activate()
        Task {
            while app.forgeInProgress != nil, !c.broken { try? await Task.sleep(for: .milliseconds(300)) }
            if let p = app.problem {
                finish(c, Response.failure(p.code ?? "import", p.text))
                return
            }
            if let offer = app.installOffer, !offer.thenImport.isEmpty {
                finish(c, ["ok": true, "imported": [], "needs_family": offer.family.rawValue, "install": "silicontrol install \(offer.family.rawValue)",
                           "message": String(localized: "the app shows the family’s license and the space it takes: the user accepts there, and the import follows")])
                return
            }
            let after = libraryContents()
            finish(c, ["ok": true, "files": urls.map(\.lastPathComponent),
                       "models": after.models.subtracting(before.models).sorted(), "loras": after.loras.subtracting(before.loras).sorted()])
        }
    }

    /// The imported models and the LoRAs in the library, by id and by short name.
    private func libraryContents() -> (models: Set<String>, loras: Set<String>) {
        let o = app.library.occupancy()
        return (Set(o.imported.map(\.card.id)), Set(o.loras.map { Self.shortName($0.card.path) }))
    }

    /// **`uninstall ID|lora NAME|downloads [--yes]`**: the Models sheet's « Uninstall… » (a publisher's model, whose imported models and LoRAs stay),
    /// the trash of an imported model or of a LoRA, and « Discard » for the interrupted downloads — each after the window's confirmation, which
    /// here is `--yes`. Without it nothing happens: the answer says what would go and the room it would free.
    /// Refused as the sheet greys it out: a render, an installation, the measure running (`busy`), the library's disk absent (`library_absent`).
    /// **On a Mac followed** the sheet asks the other Mac and waits for a click there: refused here (`remote`).
    func uninstall(_ what: GestureOrder.Removal, confirmed: Bool) throws -> [String: Any] {
        if let mirror = app.mirror {
            throw Refusal("remote", String(localized: "removing from “\(mirror.macName)” waits for a click on that Mac"),
                          help: String(localized: "Render › Compute On › This Mac, then retry"))
        }
        if let reason = app.libraryBlocker { throw Refusal("library_absent", reason, help: String(localized: "connect the disk, then retry")) }
        guard app.managementPossible else {
            throw Refusal("busy", String(localized: "a render, an installation or the diagnostic is running"), help: "silicontrol wait, then retry")
        }
        let o = app.library.occupancy()
        var answer: [String: Any]
        var act: () -> Void
        switch what {
        case .downloads:
            guard o.downloads > 0 else { throw Refusal("model", String(localized: "no interrupted download to discard"), help: "silicontrol models") }
            answer = ["target": "downloads", "kind": "downloads", "frees_bytes": o.downloads]
            act = { self.app.emptyDownloads() }
        case .lora(let name):
            guard let l = o.loras.first(where: { Self.shortName($0.card.path) == name }) else {
                let names = o.loras.map { Self.shortName($0.card.path) }.joined(separator: ", ")
                throw Refusal("lora", String(localized: "LoRA \"\(name)\" not found"),
                              help: names.isEmpty ? String(localized: "no LoRA installed") : String(localized: "installed: \(names)"))
            }
            answer = ["target": name, "kind": "lora", "frees_bytes": l.bytes]
            act = { self.app.remove(l.card) }
        case .model(let id):
            let cards = app.cards
            guard let card = cards.first(where: { $0.id == id }) ?? cards.first(where: { !$0.isImported && $0.family.rawValue == id }) else {
                throw Refusal("model", String(localized: "unknown model: \"\(id)\""), help: "silicontrol models")
            }
            if card.isImported {
                guard let m = o.imported.first(where: { $0.card.id == card.id }) else {
                    throw Refusal("model", String(localized: "unknown model: \"\(id)\""), help: "silicontrol models")
                }
                answer = ["target": card.id, "kind": "model", "frees_bytes": m.bytes]
                act = { self.app.remove(card) }
            } else {
                guard let part = o.families.first(where: { $0.family == card.family }), part.present else {
                    let name = card.name
                    throw Refusal("model", String(localized: "nothing of \(name) is installed"), help: "silicontrol models")
                }
                answer = ["target": card.id, "kind": "family", "frees_bytes": part.total,
                          // Theirs to keep: they render again once the family is installed.
                          "stays": o.imported.filter { $0.card.family == card.family }.map(\.card.id)
                              + o.loras.filter { $0.card.target == card.family.rawValue }.map { Self.shortName($0.card.path) }]
                act = { self.app.uninstall(card.family) }
            }
        }
        answer["ok"] = true
        guard confirmed else {
            answer["dry_run"] = true
            answer["message"] = String(localized: "nothing was removed: add --yes to remove it")
            return answer
        }
        app.problem = nil
        act()
        if let p = app.problem { throw Refusal(p.code ?? "write", p.text) }
        answer["removed"] = true
        return answer
    }

    /// **`install stop`**: « Stop » in the Models sheet and the sidebar — the installation or the import in progress stops, the files already
    /// downloaded are kept. On a Mac followed it stops this Mac's own upload, or the lender's forge, as the button does.
    func stopInstallation() -> [String: Any] {
        let running = app.forgeInProgress != nil || app.remoteForge != nil
        if running { app.cancelForge() }
        return ["ok": true, "stopped": running]
    }
}
