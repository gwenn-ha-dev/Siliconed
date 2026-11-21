// The chosen model's catalog and what it decides about the form: the
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
    /// The model's encoder sees the images (`TextFormat.readsImages`), so the prompt can name them
    /// (`<image1>`, `<image2>`… in Qwen-Image-2.1's template).
    var promptNamesImages: Bool { mirror != nil ? traits?.readsImages ?? false : chain?.text.output.readsImages ?? false }

    /// **The format the render will have**: when editing, the one the denoiser derives from image 1
    /// (`DenoisingModule.editFormat`, the rule the developer's command line applies too), the chosen one
    /// otherwise — references or cited subjects without an image to edit included (`framingImage`).
    var outputFormat: RecommendedFormat {
        // In a mirror, the other Mac's answer; the chosen format while it is asked (« Generate » waits).
        if let q = editQuote { return quoted(q).map { RecommendedFormat($0.width, $0.height) } ?? format }
        guard let first = framingImage, let denoising = chain?.denoising else { return format }
        let f = denoising.editFormat(referenceWidth: first.width, referenceHeight: first.height)
        return Self.cappedForEditing(RecommendedFormat(f.width, f.height))
    }

    /// **An edit's area, in the app: 1024² at most**. The engine renders up
    /// to 1024×1536 with three references, but that is 523 s: it stays with the developer's command line.
    /// The denoiser's own rule already lands near 1024² (about 1 megapixel at image 1's proportions);
    /// only an image elongated enough that its short side is raised to 512 goes beyond — its long side
    /// then gives way, to a multiple of 32, the short side kept.
    nonisolated static let editingSurface = 1024 * 1024

    static func cappedForEditing(_ f: RecommendedFormat) -> RecommendedFormat {
        guard f.width * f.height > editingSurface else { return f }
        if f.width >= f.height { return RecommendedFormat(editingSurface / f.height / 32 * 32, f.height) }
        return RecommendedFormat(f.width, editingSurface / f.width / 32 * 32)
    }

    /// Why "Generate" is grayed out, or `nil`: the app does not try what the request would refuse
    /// (the memory is the engine's to refuse, `EngineError.insufficientMemory`).
    var blocker: String? { blocker(reading: seriesReading) }

    /// `blocker`, on a reading of the series the caller already made (`seriesReading`): a body that also shows how many
    /// images the click makes reads the series once, not once per sentence — each reading builds the grid again.
    func blocker(reading series: @autoclosure () -> (blocker: String?, renders: Int)) -> String? {
        if let reason = linkBlocker { return reason }
        if let reason = libraryBlocker { return reason }
        if let reason = installBlocker { return reason }
        if missing != nil { return String(localized: "This model isn’t installed.") }
        if prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            // Describe fills the text only when it ends: while it writes, the text is empty and the prompt is coming.
            let lab = LabModel.shared
            return lab.isWriting && lab.rack === self ? String(localized: "The writer is writing the prompt.") : String(localized: "Write a prompt.")
        }
        if case .failure(let e) = alternatives { return Problem(e).text }
        // An unknown `@name`, too many images, a mention inside a group: said before the click (`LabInputs.carry`).
        if case .failure(let refusal) = carried { return refusal.sentence }
        // Three things varying, a bracket the library refuses (`Series.swift`).
        let series = series()
        if let reason = series.blocker { return reason }
        if file.count >= Self.maxQueueLength { return String(localized: "The queue is full.") }
        let n = series.renders
        if file.count + n > Self.maxQueueLength {
            let room = Self.maxQueueLength - file.count
            return String(localized: "This series makes \(String(n)) renders: the queue has room for \(String(room)).")
        }
        if let q = editQuote, quoted(q) == nil, let mirror {
            return String(localized: "Asking “\(mirror.macName)” for the edit’s size…")
        }
        if mirror != nil, let t = traits, !t.steps.contains(steps) {
            let accepted = t.steps.map(String.init).formatted(.list(type: .or))
            return String(localized: "This model renders in \(accepted) steps, not \(steps).")
        }
        if let denoising = chain?.denoising, (try? denoising.check(steps: steps, startImage: false)) == nil {
            let accepted = acceptedSteps.map(String.init).formatted(.list(type: .or))
            return String(localized: "This model renders in \(accepted) steps, not \(steps).")
        }
        if let reason = startBlocker { return reason }
        return nil
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

    // ── The XY grid ──

    /// A model as the user knows it: its card's name, or its identifier.
    func modelName(_ identifier: String) -> String { cards.first { $0.id == identifier }?.name ?? identifier }

    /// The LoRAs the render will take — chosen, and switched on —, in the stack's order: what a LoRA axis indexes.
    var activeLoRAs: [LoRASlot] { loras.filter { !$0.path.isEmpty && $0.on } }

    /// The prompt's `{…}` groups — each an axis of the series (none when it does not parse).
    var promptGroups: [[String]] { (try? PromptAlternatives(prompt, limit: .max).groups) ?? [] }

    /// A LoRA as the user knows it: its card's name (`LoRACard.name(path:)`), whatever model the rack holds — not looked up
    /// among the rack's LoRAs, where an Anima grid still shown after reusing a MOSS-Image would find none and print the file.
    func loraName(_ path: String) -> String { LoRACard.name(path: path) }

    /// The series, one axis per line — the tooltip of the render count.
    var seriesHelp: String {
        let layout = seriesLayout
        return zip([String(localized: "Columns"), String(localized: "Rows")], layout.axes)
            .map { "\($0): \(axisWords($1))" }.joined(separator: "\n")
    }

    /// **What the render will cost as it is set**, if it has already been measured in the session at
    /// this format: never a constant, never another machine. `renders`: the series' (`seriesRenders`), read by the caller.
    func plannedCost(renders: Int) -> Double? {
        let withLoRA = !activeLoRAs.isEmpty
        let format = outputFormat
        let images = sentImages
        guard let v = speeds[Speed.key(identifier, format, references: images, lora: withLoRA)] else { return nil }
        guard let plan = expectedPlan(steps: steps, format: format, strength: usedStart.map { _ in startStrength }) else { return nil }
        // A series: one image per cell, each the cost of one (same model, format and plan, near enough).
        return Double(renders) * v.duration(plan, images: 1)
    }

    /// **One stage's share of that cost**, per image — what the rack writes under each box. Learned
    /// like the whole (`speeds`): `nil` until this model has rendered at this size in the session.
    func stageEstimate(_ stage: Engine.Stage) -> Double? {
        let withLoRA = !activeLoRAs.isEmpty
        let format = outputFormat
        let images = sentImages
        guard let v = speeds[Speed.key(identifier, format, references: images, lora: withLoRA)] else { return nil }
        switch stage {
        case .text: return v.text
        case .image: return v.image
        case .decoding: return v.decoding
        case .denoising:
            guard let plan = expectedPlan(steps: steps, format: format, strength: usedStart.map { _ in startStrength }) else { return nil }
            return Double(plan.evaluations) * v.full
        }
    }

    /// **Under a box, its time**: what the stage just took when it is done in the render on screen;
    /// otherwise the estimate learned on this Mac. `measured` says which.
    /// No image goes: the image encoder costs nothing, and says no figure.
    func stageCost(_ stage: Engine.Stage) -> (seconds: Double?, measured: Bool) {
        if stage == .image, sentImages == 0 { return (nil, false) }
        if state(of: stage) == .done, let s = stageSeconds[stage] { return (s, true) }
        return (stageEstimate(stage), false)
    }

    /// The rack's light for a stage of the chosen model.
    func state(of stage: Engine.Stage) -> StageState {
        stageModel == identifier ? stageStates[stage] ?? .idle : .idle
    }

    func modelChanged() {
        readModelCatalog()
        // A LoRA from another model would be refused at render (`LoRA.Failure.wrongTarget`).
        let valid = Set(compatibleLoras.map(\.path))
        loras.removeAll { !$0.path.isEmpty && !valid.contains($0.path) }
        steps = card?.defaultSteps ?? 8
        // A role the new model does not offer becomes neutral, never another: the image stays, waiting (`LabInputs.conforming`).
        conformInputs()
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
        let f = RecommendedFormat(w, h)
        orientation = f.orientation
        format = f
        return nil
    }

    // ── Size in two scales (Beta): the cells are `SizeScales`'; this is only what the rack asks of them ──

    /// The table cell the form's size is, `nil` for a size typed by hand or from the default menu that is not in the table.
    var sizeCell: SizeCell? { SizeScales.cell(matching: format) }

    /// **Why a cell is greyed in the rack**, `nil` where it can be chosen: the model's own limit (Anima's 2048 px a side),
    /// as the default menu checks it.
    func scaleUnavailable(_ cell: SizeCell) -> String? {
        if let family = card?.family, let refusal = SizeScales.refusal(cell, family: family) {
            if case let .formatRefused(_, _, _, bound) = refusal {
                return String(localized: "\(Figures.size(cell.width, cell.height)) — \(String(bound)) px a side at most")
            }
            return Problem(refusal).text
        }
        return nil
    }

    /// Takes the cell as the form's size, as a click on the default menu would.
    func selectScale(_ ratio: SizeRatio, _ resolution: SizeResolution) {
        guard let cell = SizeScales.cell(ratio, resolution) else { return }
        useFormat(cell.format)
    }

    /// The sizes the picker offers: the model's for this orientation, and the one typed by hand.
    var pickerFormats: [RecommendedFormat] {
        orientationFormats.contains(format) ? orientationFormats : orientationFormats + [format]
    }

    func orientationChanged() {
        guard format.orientation != orientation, let first = orientationFormats.first else { return }
        format = first
    }

    /// The way forward the footer offers: the size chosen as the picker would.
    func useFormat(_ f: RecommendedFormat) {
        orientation = f.orientation
        format = f
    }
}
