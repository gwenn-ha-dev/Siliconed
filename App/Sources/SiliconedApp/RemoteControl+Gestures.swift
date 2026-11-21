// **The window's other gestures, from the terminal** — the history's (`remove`, `undo`, `again`, `reuse`, `crop`), the series'
// (`sheet`, `remove g2`, `reuse g2`) and the models' (`uninstall`, `install stop`). Each calls the function the button or the menu
// item calls (`AppState.delete`, the window's `UndoManager`, `redo`, `restoreSettings`, `writeSheet`, `removeSeries`, `uninstall`,
// `remove`, `releaseLegacy`, `cancelLibraryJob`), with the guard the window puts on it: a gesture that is greyed out in the window is a refusal here, with the same
// sentence. The arguments are read in the library (`GestureOrder`), where the tests read them. Compute On and the library are in
// `RemoteControl+Place.swift`.
//
// What destroys something (`uninstall`, the files of version 0.21 included) acts only with `--yes`; without it, the answer says what would go and the room it frees.

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
            case "grid": throw Refusal("usage", String(localized: "grid expected (g2), not \"\(value)\""), help: "silicontrol help contract")
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
        case .removeSeries(let grid): finish(c, try removeSeries(grid))
        case .reuseSeries(let grid): finish(c, try reuseSeries(grid))
        case .undo(let redo): finish(c, try undo(redo: redo))
        case .again(let word): try again(word, cwd: cwd, c)
        case .reuse(let word): finish(c, try reuse(word, cwd: cwd))
        case .crop(let image, let spec): try crop(image, spec, c)
        case .sheet(let grid, let path): try sheet(grid: grid, to: absolute(path, cwd), c)
        case .uninstall(let what, let confirmed): finish(c, try uninstall(what, confirmed: confirmed))
        case .macs(let action): finish(c, try macs(action))
        case .compute(let target): try compute(target, c)
        case .library(let action, let confirmed): try library(action, confirmed: confirmed, cwd: cwd, c)
        case .startImage(let action): finish(c, try startImage(action, cwd: cwd))
        case .lab(let action): try lab(action, cwd: cwd, c)
        case .settings(let action): finish(c, try settings(action))
        case .bracket(let action): finish(c, try bracket(action))
        case .rack(let place): finish(c, rack(place))
        case .snapshot(let file, let window, let appearance): try snapshot(file, window: window, appearance: appearance, cwd: cwd, c)
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

    /// **`rack top|dit|fold|unfold|detach|dock`**: the rack's column scrolled to its head, or to the DiT's box — the chain then pinned
    /// in small at the top of the column, as when the hand scrolls down to the denoiser's settings —, or folded to its strip and back
    /// (`RackColumn`), or the toolbar's « Detach the Chain » and « Dock the Chain » (`RackColumnModel.setDetached`). A scroll unfolds a
    /// folded column first, and brings a detached one's window forward.
    func rack(_ place: GestureOrder.RackPlace) -> [String: Any] {
        switch place {
        case .fold: RackColumnModel.shared.fold()
        case .unfold: RackColumnModel.shared.unfold()
        case .detach: RackColumnModel.shared.setDetached(true)
        case .dock: RackColumnModel.shared.setDetached(false)
        case .top, .dit: app.rackScroll = (place, (app.rackScroll?.count ?? 0) + 1)
        }
        return ["ok": true, "rack": place.rawValue, "folded": RackColumnModel.shared.folded, "detached": RackColumnModel.shared.detached]
    }

    // MARK: - A window, as a picture

    /// **`snapshot FILE.png`**: one of the app's windows drawn by the app itself into a PNG — its frame, title bar and toolbar
    /// included —, at the screen's scale. Not a screen capture: nothing but this app's own drawing is read, so no Screen Recording
    /// permission is asked, and it works from a session that has no display (ssh), where `screencapture` cannot. The main
    /// window unless `--window` names another by a word of its title or by its number in `windows`, the list every answer
    /// carries (a sheet, a popover and a bubble are windows too, untitled: by number). `--appearance light|dark` draws it in
    /// that appearance, whatever the system's, and puts the window's own back. **What the window server composes is not
    /// in it**: glass and materials come as their plain fill, a shadow not at all. Each listed window carries its `id`, the
    /// window server's number, which `screencapture -l` takes for a real capture of the glass.
    func snapshot(_ file: String, window word: String?, appearance: GestureOrder.SnapshotAppearance?, cwd: String, _ c: Connection) throws {
        let shown = NSApp.windows.filter { $0.isVisible && $0.contentView != nil }
        let main = shown.first { $0.delegate === MainWindow.shared }
        let listed: [[String: Any]] = shown.enumerated().map { n, w in
            ["n": n + 1, "title": w.title, "width": Int(w.frame.width), "height": Int(w.frame.height), "main": w === main,
             "id": w.windowNumber]
        }
        let target: NSWindow?
        if let word {
            if let n = Int(word) { target = shown.indices.contains(n - 1) ? shown[n - 1] : nil }
            else { target = shown.first { $0.title.localizedCaseInsensitiveContains(word) } }
        } else {
            target = main
        }
        guard let target, let view = target.contentView?.superview ?? target.contentView else {
            let names = listed.map { "\($0["n"]!): \(($0["title"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "—")" }.joined(separator: ", ")
            throw Refusal("window", word.map { String(localized: "no window \"\($0)\" is shown") } ?? String(localized: "the main window is not shown"),
                          help: names.isEmpty ? nil : names)
        }
        var path = absolute(file, cwd)
        if !path.lowercased().hasSuffix(".png") { path += ".png" }
        let own = target.appearance
        if let appearance { target.appearance = NSAppearance(named: appearance == .dark ? .darkAqua : .aqua) }
        Task {
            // The views take the appearance at their next drawing: SwiftUI's, a turn of the run loop later.
            if appearance != nil { try? await Task.sleep(for: .milliseconds(400)) }
            defer { if appearance != nil { target.appearance = own } }
            view.layoutSubtreeIfNeeded()
            guard let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) else {
                finish(c, Response.failure("snapshot", String(localized: "the window could not be drawn")))
                return
            }
            view.cacheDisplay(in: view.bounds, to: bitmap)
            guard let png = bitmap.representation(using: .png, properties: [:]) else {
                finish(c, Response.failure("snapshot", String(localized: "the window could not be drawn")))
                return
            }
            do {
                try png.write(to: URL(fileURLWithPath: path), options: .atomic)
            } catch {
                finish(c, Response.failure("write", error.localizedDescription))
                return
            }
            finish(c, ["ok": true, "path": path, "window": target.title, "pixels": [bitmap.pixelsWide, bitmap.pixelsHigh],
                       "appearance": appearance?.rawValue ?? NSNull(), "windows": listed])
        }
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

    /// **`reuse <i7|file.png>`**: « Reuse These Settings » — the rack takes the image's model, prompt, size, steps, LoRAs, edit
    /// images and seed. Or a PNG Siliconed wrote, as when it is dropped on the canvas: what could not come back is said in `notes`.
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
    /// made another day comes out the same, bit for bit, if the model and the LoRAs are the ones it was made with, the Neural Engine
    /// is off (on, the engine may move work to it between renders) and macOS has not changed.
    func again(_ word: String, cwd: String, _ c: Connection) throws {
        let gesture: () -> Void
        switch try source(word, cwd: cwd) {
        case .entry(let e):
            gesture = { self.app.redo(e) }
        case .file(let url):
            _ = try restore(fromPNG: url)
            gesture = { self.app.render() }
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
            let px = String(Format.minimumSide)
            throw Refusal("crop", String(localized: "this image cannot hold that proportion at \(px) px or more"))
        }
        let app = self.app
        let before = app.lastImage
        Task {
            await app.commitCrop(e, box: box)
            @MainActor func made() -> Entry? { app.history.first { $0.cropped != nil && $0.ordinal > before } }
            // Here the crop is in the history once `commitCrop` returns; the other Mac cuts its own pixels, and its
            // image comes through the history's sync.
            if app.mirror != nil {
                let deadline = Date().addingTimeInterval(60)
                while made() == nil, Date() < deadline, !c.broken { try? await Task.sleep(for: .milliseconds(100)) }
            }
            guard let made = made() else {
                finish(c, Response.failure("crop", String(localized: "This image could not be cropped")))
                return
            }
            var o = json(made)
            o["crop"] = ["x": box.x, "y": box.y, "width": box.width, "height": box.height]
            finish(c, ["ok": true, "image": o, "undo": undoState()])
        }
    }

    // MARK: - The series

    /// **`sheet g2 <file.png|folder>`**: « Export Sheet… » — the series' images side by side, labelled, written there (in a
    /// folder, under the sheet's name). Any series of the session, ended or not: a cell not rendered yet is drawn empty.
    func sheet(grid: Int, to path: String, _ c: Connection) throws {
        let run = try run(grid)
        // The window greys the button while no cell has an image.
        if app.cells(of: run).isEmpty { throw Refusal("cell", String(localized: "no cell has a picture yet, so there is no sheet to draw")) }
        Task {
            do {
                guard let written = try await app.writeSheet(run, to: path) else {
                    finish(c, Response.failure("cell", String(localized: "no cell has a picture yet, so there is no sheet to draw")))
                    return
                }
                finish(c, ["ok": true, "grid": run.name, "path": written])
            } catch {
                finish(c, Response.failure("write", Problem(EngineError(error)).text))
            }
        }
    }

    /// A series of the session, by its number — the refusal `sheet` gives too.
    private func run(_ grid: Int) throws -> GridRun {
        guard let run = app.series[grid] else {
            throw Refusal("unknown_job", String(localized: "no grid \("g\(grid)") in this session"), help: "silicontrol status")
        }
        return run
    }

    /// **`remove g2`**: the series' tile › « Remove Series from History » (`removeSeries`) — its cells leave the history, which `undo`
    /// brings back as ⌘Z does, and its cells still waiting leave the queue; one running stops.
    func removeSeries(_ grid: Int) throws -> [String: Any] {
        let run = try run(grid)
        let removed = app.cells(of: run).values.map(\.name)
        app.removeSeries(run)
        return ["ok": true, "grid": run.name, "removed": removed.sorted(), "history_images": app.history.count, "undo": undoState()]
    }

    /// **`reuse g2`**: the series' tile › « Reuse This Grid » (`restoreSettings(_: GridRun)`) — the rack takes what the cells share and
    /// the grid's axes, the prompt as typed. What the window cannot bracket (the grid's models, steps or edit images) is left out, and said.
    func reuseSeries(_ grid: Int) throws -> [String: Any] {
        let run = try run(grid)
        let before = app.toastTask
        app.restoreSettings(run)
        var o: [String: Any] = ["ok": true, "grid": run.name, "rack": rackSettings()]
        // What the window flashes, read as `reuse` reads a PNG's notes.
        if app.toastTask != before, let said = app.toast { o["notes"] = said }
        return o
    }

    // MARK: - The models

    /// **`uninstall ID [--variant V]|downloads|legacy [--yes]`**: the Models sheet's « Uninstall… » (a family whole), « Remove the … Version… »
    /// (one of its versions: what the other installed one reads stays), « Remove » (an imported model: its file alone), « Discard » for the
    /// interrupted downloads, and « Free… » for the files of version 0.21 (`releaseLegacy`) — each after the
    /// window's confirmation, which here is `--yes`. Without it nothing happens: the answer says what would go and the room it would free.
    /// Refused as the sheet greys it out: a render or an installation running (`busy`), the library's disk absent (`library_absent`).
    /// **On a Mac followed** the sheet asks the other Mac and waits for a click there: refused here (`remote`).
    func uninstall(_ what: GestureOrder.Removal, confirmed: Bool) throws -> [String: Any] {
        if let mirror = app.mirror {
            throw Refusal("remote", String(localized: "removing from “\(mirror.macName)” waits for a click on that Mac"),
                          help: String(localized: "Render › Compute On › This Mac, then retry"))
        }
        if let reason = app.libraryBlocker { throw Refusal("library_absent", reason, help: String(localized: "connect the disk, then retry")) }
        guard app.managementPossible else {
            throw Refusal("busy", String(localized: "a render or an installation is running"), help: "silicontrol wait, then retry")
        }
        // What each target needs, read on the disk now — not the whole walk of `occupancy()`.
        let l = app.library
        var answer: [String: Any]
        var act: () -> Void
        switch what {
        case .downloads:
            let left = l.downloadsLeft()
            guard left > 0 else { throw Refusal("model", String(localized: "no interrupted download to discard"), help: "silicontrol models") }
            answer = ["target": "downloads", "kind": "downloads", "frees_bytes": left]
            act = { self.app.emptyDownloads() }
        case .legacy:
            // The sheet shows the row only when there is something: none, no « Free… ».
            let files = l.legacyFiles()
            guard !files.isEmpty else { throw Refusal("model", String(localized: "nothing of version 0.21 is left"), help: "silicontrol models") }
            answer = ["target": "legacy", "kind": "legacy", "files": files.count, "frees_bytes": files.reduce(0) { $0 + $1.bytes }]
            act = { self.app.releaseLegacy() }
        case .model(let id, let version):
            let cards = app.cards
            guard let card = cards.first(where: { $0.id == id }) ?? cards.first(where: { !$0.isImported && $0.family.rawValue == id }) else {
                throw Refusal("model", String(localized: "unknown model: \"\(id)\""), help: "silicontrol models")
            }
            // An imported model is its file alone (the sheet's « Remove »): its family, which other models read, stays.
            if card.isImported {
                guard version == nil else {
                    throw Refusal("model", String(localized: "\(card.name) is an imported model: it has no versions"), help: "silicontrol uninstall \(card.id)")
                }
                answer = ["target": card.id, "kind": "imported", "frees_bytes": card.map.flatMap { Library.fileSize(l.map($0)) } ?? 0]
                act = { self.app.remove(card) }
                break
            }
            let part = l.part(card.family)
            guard part.present else {
                let name = card.name
                throw Refusal("model", String(localized: "nothing of \(name) is installed"), help: "silicontrol models")
            }
            let family = card.family
            guard let version else {
                answer = ["target": card.id, "kind": "family", "frees_bytes": part.bytes]
                act = { self.app.uninstall(family) }
                break
            }
            guard let variant = Variant(rawValue: version.lowercased()), part.installed.contains(variant) else {
                let installed = part.installed.map(\.rawValue).joined(separator: ", ")
                throw Refusal("model", String(localized: "\(card.name) has no \(version) version installed"),
                              help: installed.isEmpty ? "silicontrol models" : String(localized: "installed: \(installed)"))
            }
            // What the family's other installed version reads stays: the room freed is this version's own files.
            let kept = Set(part.installed.filter { $0 != variant }.flatMap { family.files($0) }.map(\.place))
            let frees = family.files(variant).filter { !kept.contains($0.place) }.reduce(0) { $0 + $1.bytes }
            answer = ["target": card.id, "kind": "version", "variant": variant.rawValue, "frees_bytes": frees]
            act = { self.app.uninstall(family, variant: variant) }
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

    /// **`install stop`**: « Stop » in the Models sheet and the sidebar — the installation in progress stops, what was downloaded is
    /// kept. On a Mac followed it stops this Mac's own upload, or the lender's installation, as the button does.
    func stopInstallation() -> [String: Any] {
        // In a mirror `libraryJob` is this Mac's own upload too (`MirrorUpload.swift`).
        let running = app.libraryJob != nil || app.remoteLibraryJob != nil
        if running { app.cancelLibraryJob() }
        return ["ok": true, "stopped": running]
    }
}
