// The chosen model's catalog and what it decides about the form: the memory need, the
// refusals, the XY grid defaults and the per-stage estimates. Split from `AppState.swift`.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

extension AppState {
    // MARK: Catalog

    var card: ModelCard? { cards.first { $0.id == identifier } }
    var formats: [RecommendedFormat] { card?.formats ?? ModelCard.recommendedFormats }
    var orientationFormats: [RecommendedFormat] { formats.filter { $0.orientation == orientation } }
    func loraCard(_ path: String) -> LoRACard? { compatibleLoras.first { $0.path == path } }
    var canAddLoRA: Bool { loras.count < Self.maxLoRA && !compatibleLoras.isEmpty }
    /// How many references the chosen model reads (0: it does no editing).
    var maxReferences: Int { mirror != nil ? traits?.maxReferences ?? 0 : chain?.denoising.maxReferences ?? 0 }
    var editingPossible: Bool { maxReferences > 0 }
    var editors: [ModelCard] { visibleReady.filter { editingModels.contains($0.id) } }
    func model(_ id: String) -> Model? {
        // A mirror reads nothing from this Mac's disk: its models are the other Mac's.
        guard mirror == nil else { return nil }
        if let known = modelsRead[id] { return known }
        let m = try? Model.named(id, in: library)
        modelsRead[id] = .some(m)
        return m
    }
    var canAddReference: Bool { references.count < maxReferences }
    /// The model's encoder sees the images (`TextFormat.readsImages`), so the prompt can name them
    /// (`<image1>`, `<image2>`… in Qwen-Image-2.1's template).
    var promptNamesImages: Bool { mirror != nil ? traits?.readsImages ?? false : chain?.text.output.readsImages ?? false }

    /// **The format the render will have**: when editing, the one the denoiser derives from image 1
    /// (`DenoisingModule.editFormat`, the rule the developer's command line applies too), the chosen one
    /// otherwise.
    var outputFormat: RecommendedFormat {
        // In a mirror, the other Mac's answer; the chosen format while it is asked (« Generate » waits).
        if let q = editQuote { return quoted(q).map { RecommendedFormat($0.width, $0.height) } ?? format }
        guard let first = references.first, let denoising = chain?.denoising else { return format }
        let f = denoising.editFormat(referenceWidth: first.width, referenceHeight: first.height)
        return Self.cappedForEditing(RecommendedFormat(f.width, f.height))
    }

    /// **An edit's area, in the app: 1024² at most**. The engine renders up
    /// to 1024×1536 with three references, but that is 523 s: it stays with the developer's command line.
    /// The denoiser's own rule already lands near 1024² (about 1 megapixel at image 1's proportions);
    /// only an image elongated enough that its short side is raised to 512 goes beyond — its long side
    /// then gives way, to a multiple of 32, the short side kept.
    static let editingSurface = 1024 * 1024

    static func cappedForEditing(_ f: RecommendedFormat) -> RecommendedFormat {
        guard f.width * f.height > editingSurface else { return f }
        if f.width >= f.height { return RecommendedFormat(editingSurface / f.height / 32 * 32, f.height) }
        return RecommendedFormat(f.width, editingSurface / f.width / 32 * 32)
    }

    /// **The memory this render needs**: its floor computed from the request's shapes, with what the
    /// app already holds (`ModelCard.memoryNeed`) — what the engine's preflight compares to B. `nil` when it cannot be computed.
    var memoryNeed: Int? {
        guard settingsRead else { return nil }
        let f = outputFormat
        return need(f, images: references.count + subjectImageCount)
    }

    /// Why "Generate" is grayed out, or `nil`: the app does not try what the request would refuse.
    /// The footer reads its two halves once per drawing (`formBlocker`, then `memoryBlocker`).
    var blocker: String? { formBlocker ?? memoryBlocker }

    /// What the form itself refuses: everything but the memory.
    var formBlocker: String? {
        if let reason = linkBlocker { return reason }
        if let reason = libraryBlocker { return reason }
        if let reason = installBlocker { return reason }
        if missing != nil { return String(localized: "This model isn't installed.") }
        if prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return String(localized: "Write a prompt.") }
        let n: Int
        switch alternatives {
        case .failure(let e): return Problem(e).text
        case .success(let a): n = a.count
        }
        if let refusal = subjectsBlocker { return refusal }
        if file.count >= Self.maxQueueLength { return String(localized: "The queue is full.") }
        if file.count + n > Self.maxQueueLength {
            let room = Self.maxQueueLength - file.count
            return String(localized: "These alternatives make \(String(n)) renders: the queue has room for \(String(room)).")
        }
        if let q = editQuote, quoted(q) == nil, let mirror {
            return String(localized: "Asking “\(mirror.macName)” for the edit’s size…")
        }
        if mirror != nil, let t = traits, !t.steps.contains(steps) {
            let accepted = t.steps.map(String.init).formatted(.list(type: .or))
            return String(localized: "This model has no schedule for \(String(steps)) steps: \(accepted).")
        }
        if let denoising = chain?.denoising, (try? denoising.check(steps: steps, startImage: false)) == nil {
            let accepted = acceptedSteps.map(String.init).formatted(.list(type: .or))
            return String(localized: "This model has no schedule for \(String(steps)) steps: \(accepted).")
        }
        return nil
    }

    /// **The need of the chosen model at `f`**, with `images` references: the engine's, from the maps'
    /// headers — or, in a mirror, the one the other Mac published (`MirrorCatalog.needs`: its formats,
    /// without references). A format it did not publish has no need here: the other Mac judges it.
    func need(_ f: RecommendedFormat, images: Int) -> Int? {
        if mirror != nil {
            if images == 0 { return mirrorCatalog?.needs[identifier]?[MirrorCatalog.key(f.width, f.height)] }
            if let e = editQuote, let a = mirrorQuotes[e], a.width == f.width, a.height == f.height, e.images == images { return a.need }
            return quoted(RemoteQuote(model: identifier, width: f.width, height: f.height, first: nil, images: images))?.need
        }
        return card?.memoryNeed(width: f.width, height: f.height, references: images)
    }

    /// The engine's check against B (`ModelCard.checkMemory`); in a mirror, the published need against
    /// the other Mac's B.
    func checkNeed(_ f: RecommendedFormat, images: Int) throws(EngineError) {
        guard mirror != nil else {
            try card?.checkMemory(width: f.width, height: f.height, references: images, ceiling: memoryCeiling)
            return
        }
        if let n = need(f, images: images), n > memoryCeiling { throw .insufficientMemory(needed: n, budget: memoryCeiling) }
    }

    /// Rule 7: a render that cannot fit is refused before it starts — by the engine's own check
    /// (`ModelCard.checkMemory`, against B), not after the click.
    var memoryBlocker: String? { memoryRefusal?.text }

    /// **The smallest Mac that renders `format` with `images` references**, in GB — `nil` when this
    /// one does. The engine's own check (`ModelCard.checkMemory`), against B.
    func macNeeded(_ format: RecommendedFormat, images: Int) -> Int? {
        guard settingsRead, let card else { return nil }
        do { try checkNeed(format, images: images) }
        catch {
            guard case let .insufficientMemory(needed, _) = error else { return nil }
            return MemoryBudget.macNeeded(for: needed)
        }
        return nil
    }

    /// « Needs a Mac with 24 GB of memory. » — the one sentence of a memory refusal, wherever it shows.
    nonisolated static func needsMac(_ needed: Int) -> String {
        String(localized: "Needs a Mac with \(String(MemoryBudget.macNeeded(for: needed))) GB of memory.")
    }

    /// **A memory refusal, never a dead end**: what greys « Generate » out, and the way forward — the
    /// largest of the model's formats this Mac renders. Reached only by what the pickers cannot grey
    /// out ahead: a size typed by hand before a model change, subjects cited by the prompt.
    struct MemoryRefusal {
        let text: String
        /// The largest format that fits in B (`RecommendedFormat.largestFitting`), `nil` when none
        /// does — or when editing, where image 1 sets the size and the picker does not.
        let smaller: RecommendedFormat?
    }

    var memoryRefusal: MemoryRefusal? {
        guard settingsRead, let card else { return nil }
        let f = outputFormat
        let images = references.count + subjectImageCount
        do { try checkNeed(f, images: images) } catch {
            // A need that cannot be computed refuses too, with its own sentence (`memoryNeedUnknown`).
            guard case let .insufficientMemory(needed, budget) = error else { return MemoryRefusal(text: Problem(error).text, smaller: nil) }
            let smaller = references.isEmpty
                ? RecommendedFormat.largestFitting(formats, budget: budget, current: f) {
                    need($0, images: images) ?? .max
                  }
                : nil
            return MemoryRefusal(text: Self.needsMac(needed), smaller: smaller)
        }
        return nil
    }

    /// **What one more image would need** (after `adding` others): `nil` when this Mac renders it,
    /// else the sentence. Image 1 sets an edit's size (about a megapixel, 1024² at most): before it,
    /// judged at 1024².
    func nextReferenceRefusal(adding: Int = 0) -> String? {
        let f = references.isEmpty ? RecommendedFormat(1024, 1024) : outputFormat
        let images = references.count + adding + 1
        guard let card, macNeeded(f, images: images) != nil else { return nil }
        return need(f, images: images).map(Self.needsMac)
    }

    /// The way forward the footer offers: the size that fits, chosen as the picker would.
    func useFormat(_ f: RecommendedFormat) {
        orientation = f.orientation
        format = f
    }

    /// **The step counts the chosen model has a schedule for**, asked of its denoiser
    /// (`DenoisingModule.check`) rather than known here: all of them for most, 8 alone for
    /// Qwen-Image-2.1-Turbo. Read when "Generate" is greyed out for that reason, never at each drawing.
    var acceptedSteps: [Int] {
        if mirror != nil { return traits?.steps ?? [] }
        guard let denoising = chain?.denoising else { return [] }
        return (1...50).filter { (try? denoising.check(steps: $0, startImage: false)) != nil }
    }
    var canRender: Bool { blocker == nil }

    /// **The prompt's alternatives** (`{library|greenhouse}`), parsed at each reading: a few
    /// microseconds, and the form never holds a stale parse. A prompt without a group is one prompt.
    var alternatives: Result<PromptAlternatives, EngineError> {
        Result { () throws(EngineError) in try PromptAlternatives(prompt) }
    }

    /// How many renders "Generate" queues: one per expanded prompt (1 when the prompt does not parse —
    /// the button is greyed out then, and `blocker` says why).
    var variantCount: Int { (try? alternatives.get())?.count ?? 1 }

    // ── The XY grid ──

    /// A model as the user knows it: its card's name, or its identifier.
    func modelName(_ identifier: String) -> String { cards.first { $0.id == identifier }?.name ?? identifier }

    /// The LoRAs the render will take, in the stack's order — what a LoRA axis indexes.
    var activeLoRAs: [LoRASlot] { loras.filter { !$0.path.isEmpty } }

    /// The exploration prompt's `{…}` groups — what the grid's axes offer (none when it does not parse).
    var promptGroups: [[String]] { (try? PromptAlternatives(explorePrompt, limit: .max).groups) ?? [] }

    /// A LoRA as the user knows it: its card's name, or its short name.
    func loraName(_ path: String) -> String { loraCard(path)?.name ?? RemoteControl.shortName(path) }

    /// **The grid the exploration's form describes**, checked by the library. `seed`: the base seed
    /// (the rack's; the one drawn at the click for a random seed). A model axis drops the stack: a
    /// LoRA is forged for one model (`loraAcrossModels` otherwise).
    func grid(seed: UInt64) -> Result<RenderGrid, EngineError> {
        let models = [gridX, gridY].contains { $0.kind == .model }
        let stack = models ? [] : activeLoRAs
        return Result { () throws(EngineError) in
            guard let x = try gridX.axis(stack: stack) else { throw .gridRefused(reason: .emptyAxis) }
            let g = try RenderGrid(.init(prompt: explorePrompt, seed: seed, steps: steps,
                                         loraStrengths: stack.map(\.strength), images: references.count),
                                   x: x, y: try gridY.axis(stack: stack))
            guard !refusesNegativeStrength(g) else { throw .gridRefused(reason: .invalidValue) }
            // A solver the form's model does not offer (Developer Mode off) renders nothing new.
            guard g.cells.allSatisfy({ $0.solver.map(offeredSolvers.contains) ?? true }) else { throw .gridRefused(reason: .invalidValue) }
            return g
        }
    }
    var grid: Result<RenderGrid, EngineError> { grid(seed: seed) }

    /// Axes that make sense for this prompt — its groups first, the seed otherwise.
    func gridDefaults() {
        guard (try? grid.get()) == nil else { return }
        let groups = promptGroups.count
        gridX = GridAxisForm(kind: groups > 0 ? .prompt(0) : .seed)
        gridY = GridAxisForm(kind: groups > 1 ? .prompt(1) : groups == 1 ? .seed : .none)
        if groups == 1 { gridY.seedCount = 2 }
    }

    /// **The exploration's form takes a grid** — a sheet's settings reused, or a grid the remote
    /// control queued: its prompt as typed, its axes.
    func setGridForm(_ g: RenderGrid, prompt typed: String) {
        let stack = activeLoRAs
        explorePrompt = typed
        gridX = GridAxisForm(g.x, stack: stack)
        gridY = GridAxisForm(g.y, stack: stack)
    }

    /// The series, one expanded prompt per line — the tooltip of the render count.
    var alternativesHelp: String {
        guard let a = try? alternatives.get(), a.hasAlternatives else { return "" }
        let list = a.prompts.prefix(12).joined(separator: "\n")
        return a.count > 12 ? list + "\n…" : list
    }

    /// **What the render will cost as it is set**, if it has already been measured in the session at
    /// this format: never a constant, never another machine.
    var plannedCost: Double? {
        let withLoRA = loras.contains { !$0.path.isEmpty }
        let format = outputFormat
        let images = references.count + subjectImageCount
        guard let v = speeds[Speed.key(identifier, format, references: images, lora: withLoRA)] else { return nil }
        guard let plan = expectedPlan(steps: steps, format: format, withLoRA: withLoRA || !plainSampling,
                                      references: images) else { return nil }
        // One render per alternative, each the cost of one: same model, format, plan and batch.
        return Double(variantCount) * v.duration(plan, images: max(1, batch))
    }

    /// **One stage's share of that cost**, per image — what the rack writes under each box. Learned
    /// like the whole (`speeds`): `nil` until this model has rendered at this size in the session.
    func stageEstimate(_ stage: Engine.Stage) -> Double? {
        let withLoRA = loras.contains { !$0.path.isEmpty }
        let format = outputFormat
        let images = references.count + subjectImageCount
        guard let v = speeds[Speed.key(identifier, format, references: images, lora: withLoRA)] else { return nil }
        switch stage {
        case .text: return v.text
        case .image: return v.image
        case .decoding: return v.decoding
        case .denoising:
            guard let plan = expectedPlan(steps: steps, format: format, withLoRA: withLoRA || !plainSampling,
                                      references: images) else { return nil }
            let reduced = min(plan.reduced, plan.evaluations)
            return Double(plan.evaluations - reduced) * v.full + Double(reduced) * (v.reducedAverage ?? v.full)
        }
    }

    /// **Under a box, its time**: what the stage just took when it is done in the render on screen;
    /// otherwise the estimate learned on this Mac. `measured` says which.
    func stageCost(_ stage: Engine.Stage) -> (seconds: Double?, measured: Bool) {
        if state(of: stage) == .done, let s = stageSeconds[stage] { return (s, true) }
        return (stageEstimate(stage), false)
    }

    /// The rack's light for a stage of the chosen model.
    func state(of stage: Engine.Stage) -> StageState {
        stageModel == identifier ? stageStates[stage] ?? .idle : .idle
    }

    func modelChanged() {
        readModelCatalog()
        warm()
        // A LoRA from another model would be refused at render (`LoRA.Failure.wrongTarget`).
        let valid = Set(compatibleLoras.map(\.path))
        loras.removeAll { !$0.path.isEmpty && !valid.contains($0.path) }
        steps = card?.defaultSteps ?? 8
        if references.count > maxReferences { references = Array(references.prefix(maxReferences)) }
        // A size typed by hand stays while the new model takes it (`Format.check(…, family:)`: Qwen-Image-2.1
        // wants multiples of 32, Anima 2048 px a side at most) — and this Mac's memory, or the footer
        // says so and offers the largest that fits.
        if (try? checkFormat(width: format.width, height: format.height)) == nil {
            format = formats[0]; orientation = format.orientation
        }
    }

    /// **The chosen model's format rules** (`Format.check(…, family:)`, the engine's preflight); the
    /// general ones when no model is chosen.
    func checkFormat(width: Int, height: Int) throws(EngineError) {
        if let family = card?.family { try Format.check(width: width, height: height, family: family) }
        else { try Format.check(width: width, height: height) }
    }

    /// The multiple a typed side must be, and the longest side, for the chosen model — what the size
    /// field announces, so that it never suggests a size the model refuses.
    var formatMultiple: Int { card.map { Format.multiple(for: $0.family) } ?? Format.multiple }
    var formatMaximumSide: Int { card.map { Format.maximumSide(for: $0.family) } ?? Format.maximumSide }

    /// **A size typed by hand** (`832x1216`, `1024`): read by `Format.parse`, judged by `Format.check` —
    /// the engine's own rules (sides ≥ 512, multiples of 16, at most 8192 px a side; no area ceiling, B decides), and the
    /// model's (Qwen-Image-2.1 multiples of 32, Anima 2048 px a side at most). The refusal, translated.
    func setCustomFormat(_ text: String) -> Problem? {
        guard let (w, h) = Format.parse(text.trimmingCharacters(in: .whitespaces)) else {
            return Problem(EngineError.formatUnreadable(text: text))
        }
        do { try checkFormat(width: w, height: h) } catch { return Problem(error) }
        // Beyond B, refused here as the picker greys it out: the same check as the engine's.
        if settingsRead {
            do { try checkNeed(RecommendedFormat(w, h), images: subjectImageCount) }
            catch { return Problem(error) }
        }
        let f = RecommendedFormat(w, h)
        orientation = f.orientation
        format = f
        return nil
    }

    /// The sizes the picker offers: the model's for this orientation, and the one typed by hand.
    var pickerFormats: [RecommendedFormat] {
        orientationFormats.contains(format) ? orientationFormats : orientationFormats + [format]
    }

    /// The orientation's first size this Mac renders (`macNeeded`), its first at all when none does —
    /// the footer then says why.
    func orientationChanged() {
        guard format.orientation != orientation, let first = orientationFormats.first else { return }
        format = orientationFormats.first { macNeeded($0, images: subjectImageCount) == nil } ?? first
    }
}
