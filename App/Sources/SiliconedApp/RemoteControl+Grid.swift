// `grid`: its axes read and checked by the library, then one job per cell.

import AppKit
import Foundation
import Measurement
import Observation
import Siliconed
import SilicontrolHelp

extension RemoteControl {
    /// **`grid`**: the request `add` checked, its axes read and the grid checked by the library
    /// (`RenderGrid`), every step count of it checked by the denoiser, the queue's room — then one job
    /// per cell; with `--wait --out D`, the sheet written into D when the last one ends.
    func addGrid(_ d: AddOrder, prompt: String, model: Model, card: ModelCard?, format: RecommendedFormat,
                         steps: Int, stack: [LoRASlot], references: [ReferenceImage], start: ReferenceImage?, strength: Double?,
                         seed: UInt64,
                         defaultSteps: Int, folder: String?, cwd: String, _ c: Connection) throws {
        // A LoRA axis names its LoRAs as `--lora` does (a name, else a path from the client's folder).
        let g = try readGrid(d, prompt: prompt, seed: seed, steps: steps, stack: stack, images: references.count, start: start != nil) { name in
            (app.library.lora(name, for: model.identifier) ?? app.library.lora(absolute(name, cwd), for: model.identifier))?.path
        }
        try checkSolverAxis(g, family: card?.family, name: card?.name ?? model.name)
        // Each cell's step count against its own model's schedules (a model axis: each model's).
        for cell in g.cells {
            guard let n = cell.steps else { continue }
            let denoising = cell.model.flatMap { try? Model.named($0, in: app.library).chain.denoising } ?? model.chain.denoising
            try checkSteps(n, scheduled: denoising.scheduledSteps, defaultSteps: nil)
        }
        guard app.file.count + g.count <= AppState.maxQueueLength else {
            let n = g.count, max = AppState.maxQueueLength, room = max - app.file.count
            throw Refusal("queue_full", String(localized: "\(n) jobs for the grid: the queue has room for \(room) (\(max) jobs)"),
                          help: "silicontrol wait, then retry")
        }
        var settings = RenderSettings(identifier: model.identifier, modelName: card?.name ?? model.name, prompt: prompt,
                                      format: format, steps: steps, loras: stack, references: references.count, detail: d.detail,
                                      solver: d.solver)
        settings.strength = strength
        let (run, jobs) = app.enqueueGrid(g, settings: settings, references: references, previews: d.previews ?? app.previews,
                                          start: start)
        // The form takes the grid, as after a click, and the axes the window brackets (a models, steps, listed seeds or images
        // axis stays silicontrol's).
        app.takeRequest(settings, prompt: prompt, references: references, start: start, strength: strength, seed: seed,
                        fixed: d.seed != nil)
        _ = app.setSeries(from: g)
        // …and the canvas shows the grid filling in, as « Generate » shows a series (`renderSeries`).
        app.followsRender = false
        app.shownSeries = run.ordinal

        let added = jobs.map(queued)
        if d.wait {
            for var o in added { o["type"] = "added"; emit(c, o) }
            followGrid(run.ordinal, folder: folder, sheet: d.sheet, c)
        } else {
            finish(c, ["ok": true, "grid": run.name, "columns": g.columns, "rows": g.rows, "jobs": added,
                       "estimate_s": estimate(jobs)])
        }
    }

    /// **The grid of `grid`**: its axes read, a LoRA axis's names turned into paths by `lora` (`nil`: no such LoRA),
    /// and the grid checked by the library (`RenderGrid`) — and by the rack's rule on negative strengths.
    func readGrid(_ d: AddOrder, prompt: String, seed: UInt64, steps: Int, stack: [LoRASlot], images: Int, start: Bool,
                  lora: (String) -> String?) throws -> RenderGrid {
        do {
            let names = d.loras.map(\.name)
            func resolved(_ a: RenderGrid.Axis) throws(EngineError) -> RenderGrid.Axis {
                guard case let .addedLoRAs(l, f) = a else { return a }
                var paths: [String] = []
                for name in l {
                    if name.isEmpty { paths.append(""); continue }
                    guard let path = lora(name) else { throw .gridRefused(reason: .noSuchLoRA) }
                    paths.append(path)
                }
                return .addedLoRAs(paths, strength: f)
            }
            let x = try resolved(RenderGrid.axis(d.x ?? "", loras: names))
            let y = try d.y.map { try resolved(RenderGrid.axis($0, loras: names)) }
            let g = try RenderGrid(.init(prompt: prompt, seed: seed, steps: steps, loraStrengths: stack.map(\.strength),
                                         images: images, start: start), x: x, y: y)
            guard !app.refusesNegativeStrength(g) else { throw EngineError.gridRefused(reason: .invalidValue) }
            return g
        } catch {
            let p = Problem(error)
            throw Refusal("grid", p.title, help: p.suggestion ?? "silicontrol help grid")
        }
    }
}
