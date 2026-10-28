// What `add --wait`, `wait` and `follow` stream back to the client: the events of a job
// or a grid, then its images. Split from `RemoteControl.swift`.

import AppKit
import Foundation
import Observation
import Siliconed
import SilicontrolHelp

extension RemoteControl {
    /// **Follows a grid to its sheet**: each cell's image as it comes (written into `folder` if
    /// given), then the sheet, then the verdict. A grid already ended answers at once.
    func followGrid(_ g: Int, folder: String?, _ c: Connection) {
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
    func emitImage(_ e: Entry, folder: String?, type: String = "image", _ c: Connection) {
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

    func wait(_ arguments: [String], cwd: String, _ c: Connection) throws {
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
                guard app.gridRuns[n] != nil || app.drawingSheets[n] != nil || app.endedGrids[n] != nil
                        || app.mirror?.grids[n] != nil else {
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
            if let mirror = app.mirror { followMirroredGrid(mirror, grids[0], folder: folder, c) } else { followGrid(grids[0], folder: folder, c) }
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
    func followToEnd(_ ordinals: [Int], folder: String?, _ c: Connection) {
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

    func follow(_ c: Connection) {
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
}
