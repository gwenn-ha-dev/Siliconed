// `AppState`: an image made again — varied, redone — and its settings brought back to the rack, from the history, a series or a PNG.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

extension AppState {
    /// Whether « Variations » can take this image: an ordinary history image of an installed model,
    /// no installation running.
    func canVary(_ entry: Entry) -> Bool { varyRefusal(entry) == nil }

    /// **Why « Variations » cannot take this image**, said on the greyed button; `nil` when it can.
    func varyRefusal(_ entry: Entry) -> String? {
        if let installBlocker { return installBlocker }
        if entry.enlarged != nil { return String(localized: "An enlargement has no seed of its own: vary the image it was made from") }
        if entry.cropped != nil { return String(localized: "A crop has no seed of its own: vary the image it was cut from") }
        if !readySet.contains(entry.settings.identifier) {
            return String(localized: "\(entry.settings.modelName) isn’t installed: install it to vary this image")
        }
        return nil
    }

    /// **« Variations »**: cousins of a history image — exactly its settings (model, prompt, LoRA,
    /// size, steps, seed, edit images), each with a variation seed drawn at random at `strength`
    /// (`Variation.Amount`, its strength for the model's family), ordinary jobs whose images join the history.
    /// **As many as the rack's « Seeds »** (`batch`, 1 by default): at Qwen's ~200 s an edit, a fixed
    /// four made one click a quarter of an hour.
    ///
    /// **A variation of a variation turns the clicked image's noise**, not its parent's: the new
    /// variation is appended to the image's own (`Request.variations` applies them in order). The
    /// strength is the new request's; the origin is the image clicked — what the user sees and likes.
    /// The form is not touched: this is a gesture on an image, not a change of settings.
    @discardableResult
    func vary(_ entry: Entry, _ amount: Variation.Amount, count: Int? = nil) -> [Job] {
        guard canVary(entry), let family = cards.first(where: { $0.id == entry.settings.identifier })?.family else { return [] }
        let strength = amount.strength(for: family)
        followsRender = true
        let card = cards.first { $0.id == entry.settings.identifier }
        let plan = (try? Model.named(entry.settings.identifier, in: library)).map {
            Self.plan($0.chain.denoising, steps: entry.settings.steps, format: entry.settings.format, strength: entry.settings.strength)
        }
        let jobs = (0..<(count ?? batch)).map { _ in
            var settings = entry.settings
            settings.variations.append(Variation(seed: Request.drawnSeed(), strength: strength))
            var job = Job(ordinal: nextJobNumber(), settings: settings, seeds: [entry.seed], references: entry.references,
                          previews: previews, defaultSteps: card?.defaultSteps, plan: plan)
            job.start = entry.start
            return job
        }
        enqueue(jobs)
        return jobs
    }

    /// Redo the chosen image: same settings, same seed — and its size: an edit's would otherwise be derived again from image 1
    /// (`outputFormat`), and one typed on the command line (`add … 1024 --ref …`) came back at image 1's proportions.
    func redo(_ entry: Entry) {
        restoreSettings(entry)
        render(imposedSeed: entry.seed, imposedFormat: entry.settings.format)
    }

    func restoreSettings(_ entry: Entry) {
        restoreSettings(entry.settings)
        // An image's settings are one render's: no knob bracketed.
        brackets = []
        seriesPlaced = [nil, nil]
        // Fixed, or the next render draws a new seed and the settings reused are not the image's.
        seed = entry.seed
        fixedSeed = true
        // After the seed: setting it drops a variation (`seed.didSet`).
        variations = entry.settings.variations
        // The images it edited, or none: image 1 is the image's own, never the one the lab held before — otherwise an edit
        // comes back without what it edits, and an image made from noise comes back editing what was left there.
        setEditImages(entry.editImages)
        // The image it started from, with its strength — or none: « Redo » renders the same image, not an img2img of another.
        if let s = entry.settings.strength, let start = entry.start, startsFromImage {
            setStart(start)
            startStrength = s
        } else {
            setStart(nil)
        }
    }

    /// **« Reuse This Grid »**, on a series' tile: the form takes what the cells share and the grid's axes, as the window
    /// brackets them — the prompt as typed, its groups the grid's.
    func restoreSettings(_ run: GridRun) {
        restoreSettings(run.settings)
        prompt = run.settings.prompt
        fixedSeed = true
        seed = run.grid.base.seed
        let left = setSeries(from: run.grid)
        if !left.isEmpty { flash(String(localized: "The window can’t vary models, steps or edit images: those axes were left out.")) }
    }

    /// **The form takes a request the remote control has queued** (`silicontrol add` and `grid`, on this Mac or another), as
    /// after a click: its settings, the prompt as typed — so that « Generate » would queue the same series —, the edit's images,
    /// the starting image or none, and the seed: one drawn there is a Random render, the rack showing it greyed, not an old
    /// Fixed one. Nothing bracketed is left of the series before, and `batch` seeds are the lot (`add --batch`); a grid's axes
    /// are set after, by `setSeries`.
    func takeRequest(_ settings: RenderSettings, prompt: String, references: [ReferenceImage], start: ReferenceImage?,
                     strength: Double?, seed: UInt64, fixed: Bool, batch: Int = 1) {
        restoreSettings(settings)
        self.prompt = prompt
        brackets = []
        seriesPlaced = [nil, nil]
        self.batch = batch
        setEditImages(references)
        setStart(start)
        if let strength { startStrength = strength }
        fixedSeed = fixed
        self.seed = seed
        // After the seed, which drops them only when it changes: a variation the rack held stays otherwise, on a seed the
        // command gave that is the rack's.
        variations = settings.variations
    }

    /// The form takes these settings — those of a history image, or of a job that the
    /// remote control has just queued: the app shows what it computes, as after a click.
    func restoreSettings(_ r: RenderSettings) {
        identifier = r.identifier
        // The job's prompt is an expanded one: its braces are characters, not a series.
        prompt = PromptAlternatives.escaping(r.prompt)
        orientation = r.format.orientation
        format = r.format
        steps = r.steps
        loras = r.loras.map(allowedStrength)
        detail = r.detail
        solver = r.solver
    }

    /// **A Siliconed PNG dropped on the canvas**: the form takes what its metadata says
    /// (`Engine.Render.Recipe`), through `restoreSettings` as "Reuse These Settings" does, and the
    /// seed is fixed. What cannot come back is said (a `Problem`), and the rest is restored anyway.
    /// `false` when the file is not a PNG written by Siliconed: the drop keeps its usual meaning.
    ///
    /// Incognito: the file is only read — no panel, no recent item, no copy.
    func restoreSettings(fromPNG url: URL) -> Bool {
        guard url.isFileURL, url.pathExtension.lowercased() == "png",
              let data = try? Data(contentsOf: url, options: .mappedIfSafe) else { return false }
        return restoreSettings(fromPNG: data, file: url.lastPathComponent)
    }

    /// Whether these bytes are a PNG Siliconed wrote: what offers « Reuse These Settings » on an image of the Prompt Lab.
    static func isRecipe(_ data: Data) -> Bool { Engine.Render.Recipe(metadata: PNG.text(data)) != nil }

    /// The same, from the bytes — an image of the Prompt Lab; `file` names it in the notes.
    func restoreSettings(fromPNG data: Data, file: String) -> Bool {
        guard let recipe = Engine.Render.Recipe(metadata: PNG.text(data)) else { return false }
        var notes: [String] = []

        // The PNG names a family, not a card (a fine-tune writes its family's name): the form's
        // model if it is of that family, otherwise the publisher's card, otherwise an imported one.
        let ready = visibleCards.filter { $0.family.rawValue == recipe.model && readySet.contains($0.id) }
        let target = ready.first { $0.id == identifier } ?? ready.first { !$0.isImported } ?? ready.first
        if target == nil, let model = recipe.model {
            // Steps and LoRAs belong to that model: the form's stay.
            let name = cards.first { $0.family.rawValue == model }?.name ?? model
            notes.append(String(localized: "\(name) isn’t installed: its steps and LoRAs are left out."))
        }

        var stack: [LoRASlot] = loras
        if let target {
            stack = []
            var lost: [String] = []
            for entry in recipe.loras {
                if stack.count < Self.maxLoRA, let found = library.lora(entry.path, for: target.id) {
                    stack.append(LoRASlot(path: found.path, strength: entry.strength))
                } else {
                    lost.append(entry.path)
                }
            }
            if !lost.isEmpty {
                let names = lost.formatted(.list(type: .and))
                notes.append(String(localized: "LoRA not found: \(names)."))
            }
            if !negativeLoRAStrengths, stack.contains(where: { $0.strength < 0 }) {
                notes.append(String(localized: "A negative LoRA strength comes back as 0: turn on Settings › Advanced › Negative LoRA Strengths."))
            }
        }

        // The size if the model it goes to renders it (`Format.check(…, family:)`), the form's otherwise.
        var restoredFormat = format
        let family = (target ?? card)?.family ?? .default   // no model at all: the default family's rules
        if let w = recipe.width, let h = recipe.height, (try? Format.check(width: w, height: h, family: family)) != nil {
            restoredFormat = RecommendedFormat(w, h)
        }
        if recipe.sourceImageMissing {
            notes.append(String(localized: "The image it started from isn’t in the file (img2img)."))
        }
        if !recipe.unreadable.isEmpty {
            let keys = recipe.unreadable.formatted(.list(type: .and))
            notes.append(String(localized: "Unreadable in the file: \(keys)."))
        }

        let model = target ?? card
        restoreSettings(RenderSettings(identifier: model?.id ?? identifier, modelName: model?.name ?? "",
                                       prompt: recipe.prompt ?? prompt, format: restoredFormat,
                                       steps: target == nil ? steps : (recipe.steps ?? target!.defaultSteps),
                                       loras: stack, detail: recipe.detail, solver: recipe.solver))
        if recipe.solver != .euler, !offeredSolvers.contains(recipe.solver) {
            notes.append(String(localized: "Made with the \(recipe.solver.choice) solver, offered when More Solvers is on: Euler will be used."))
        }
        if recipe.detail != .normal, !offersDetail {
            notes.append(String(localized: "\(model?.name ?? identifier) doesn’t offer Detail: Normal will be used."))
        }
        if let s = recipe.seed { seed = s; fixedSeed = true }
        // An img2img's strength comes back to the rack; its image is not in the file (said above): the one carried, if any, stays.
        if let s = recipe.strength { startStrength = StartImage.snapped(s) }
        // A variation is its seed's: without the seed, there is nothing to vary.
        variations = recipe.seed == nil ? [] : recipe.variations
        flash(String(localized: "Settings of \(file) restored"))
        if !notes.isEmpty {
            problem = Problem(String(localized: "Not everything came back from \(file)."),
                              suggestion: notes.joined(separator: "\n"))
        }
        return true
    }
}
