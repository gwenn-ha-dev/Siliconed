// **The app's state**: what the user sets, what the render shows, the history.
//
// **Incognito by default**: the history lives in memory and dies with the
// process. Nothing is written to disk without an explicit gesture — "Save", "Copy" —,
// no temporary file, no preference, no log: nothing that Spotlight or recent items
// could find. **Three exceptions, and only three**, each a gesture of the user: two preferences —
// the « Developer Mode » switch and the comparison's curtain or side by side (`Curtain.preferred`),
// like the system language the app follows — and the licenses accepted, which the library keeps
// itself (`<root>/accepted-licenses.json`) because its engine refuses a render without them.
//
// **Two models in front**: Z-Image Turbo and Qwen-Image-2.1 Turbo. The four
// other families (Anima, Krea 2, FLUX.2 [klein] 4B, ERNIE-Image) exist and render, but the list, the
// Models sheet, the diagnostic and `silicontrol models` show them only in Developer Mode. The filter
// is here, in the app: the library is the same.
//
// **The main thread does no computing**: the engine runs on its queue, and what the app does with
// the pixels — decoding the reference image, encoding the PNG, reducing the thumbnail — goes to a
// detached task. The catalog (ready models, compatible LoRAs, what is missing) is read from disk when
// it changes, never at each drawing of the view.
//
// **All the phrases go through `String(localized:)`** or through a `Text` key: the table is
// `Localizable.xcstrings` (English source, fr de es it), which `tools/app.sh` synchronizes and compiles.
// The engine's refusals are translated case by case in `Problems.swift`.
//
// A pure client of `Siliconed` (`import`, never `@testable`): everything called here is
// public, exactly what a macOS 15+ app depending on the `Siliconed` product would see. The view is
// in `MainView.swift`, the window and the app in `App.swift`.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

/// A LoRA of the stack: empty until one is chosen (the row is added, then chosen).
struct LoRASlot: Identifiable, Equatable {
    let id = UUID()
    var path = ""
    var strength = 0.8
}

/// An edit's reference: a chosen or dragged file, or an image from the history. We keep the source,
/// not an `ImageRGB` (75 MB of floats for a 3072 px photo): it is decoded off the main thread at
/// render time. Its size in pixels (EXIF orientation applied, as `ImageRGB(contentsOf:)` does) is
/// read from the header alone: image 1 sets the output's format (`DenoisingModule.editFormat`).
///
/// `@unchecked Sendable`: immutable; the thumbnail (`NSImage`) is read only by the view.
struct ReferenceImage: Identifiable, @unchecked Sendable {
    enum Source { case file(URL), image(CGImage) }
    let id = UUID()
    let source: Source
    let vignette: NSImage
    let name: String
    let width: Int, height: Int

    /// Decoded for display, upright, at `width × height` — the original under an edit's curtain.
    func displayed(width: Int, height: Int) -> CGImage? {
        let image: CGImage?
        switch source {
        case .image(let cg): image = cg
        case .file(let url):
            let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                            kCGImageSourceThumbnailMaxPixelSize: max(self.width, self.height),
                                            kCGImageSourceCreateThumbnailWithTransform: true]
            image = CGImageSourceCreateWithURL(url as CFURL, nil)
                .flatMap { CGImageSourceCreateThumbnailAtIndex($0, 0, options as CFDictionary) }
        }
        guard let image, let c = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                           space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        c.interpolationQuality = .high
        c.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return c.makeImage()
    }

    func imageRGB() throws -> ImageRGB {
        switch source {
        case .file(let url): return try ImageRGB(contentsOf: url)
        case .image(let cg): return try ImageRGB(cgImage: cg)
        }
    }

    /// A file's size as it will be decoded, upright — or `nil` if ImageIO cannot read its header.
    static func size(of url: URL) -> (width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let p = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let w = p[kCGImagePropertyPixelWidth] as? Int, let h = p[kCGImagePropertyPixelHeight] as? Int,
              w > 0, h > 0 else { return nil }
        // Orientations 5 to 8 turn the image a quarter: the upright image has its sides swapped.
        let quarter = ((p[kCGImagePropertyOrientation] as? Int) ?? 1) >= 5
        return quarter ? (h, w) : (w, h)
    }
}

/// **An image as it leaves the app** by a drag or Share: its PNG bytes, with the name the receiver
/// gives the file. Only data: no temporary file is written on the way (incognito).
struct PNGImage: Transferable {
    let data: Data
    let name: String
    static var transferRepresentation: some TransferRepresentation {
        DataRepresentation(exportedContentType: .png) { $0.data }
            .suggestedFileName { $0.name }
    }
}

/// **What the Edit box accepts**: a file (the Finder), or PNG bytes — a strip thumbnail, which leaves
/// as data only (`PNGImage`). The file first: a Finder PNG is read from its place, not copied.
enum DroppedImage: Transferable {
    case file(URL), png(Data)
    static var transferRepresentation: some TransferRepresentation {
        ProxyRepresentation(importing: { (url: URL) in DroppedImage.file(url) })
        DataRepresentation(importedContentType: .png) { DroppedImage.png($0) }
    }
}

/// What redoes an image: taken at launch, kept with each history image.
struct RenderSettings {
    var identifier: String
    var modelName: String
    var prompt: String
    var format: RecommendedFormat
    var steps: Int
    var loras: [LoRASlot]
    /// How many references the edit read (0: no edit).
    var references = 0
    /// **The variations its noise was turned by** (`Request.variations`), in order: empty for an
    /// ordinary image. The seed stays the origin's.
    var variations: [Variation] = []
    var editing: Bool { references > 0 }
    /// **More detail** (`Request.detail`): the rack's Normal · More · Most.
    var detail: Detail = .normal
    var isVariation: Bool { !variations.isEmpty }
}

/// **A history image**: the PNG (8-bit sRGB, with its metadata) and a thumbnail. No full-size
/// `CGImage` per image: at 1024², it cost 4 MB more each, next to a
/// model of 8 to 12 GB. The chosen image is decoded from its PNG (`AppState.selectedImage`).
struct Entry: Identifiable {
    let id = UUID()
    /// `i7`: the short name the remote control gives and receives (`RemoteControl.swift`).
    let ordinal: Int
    /// The job that rendered it (`t3`).
    let job: Int
    let png: Data
    let vignette: CGImage
    let width: Int, height: Int
    let seed: UInt64
    let timings: Engine.Timings
    let footprints: Engine.Footprints
    let evaluations: Int
    let settings: RenderSettings
    let metadata: [String: String]
    /// The edit's references it was rendered with — what « Variations » renders again. Shared with
    /// its job's other images (the same sources), never decoded here.
    var references: [ReferenceImage] = []
    /// **A grid's sheet** (`Grid.swift`): its `settings` are what the cells share — the prompt as
    /// typed, braces included —, its `timings` the cells' sum, its `footprints` their peaks.
    var grid: GridInfo? = nil
    /// **A sketch** (an exploration's cell): the evaluations it stopped after. Never in the history.
    var sketch: Int? = nil

    var seconds: Double { timings.total }
    var name: String { "i\(ordinal)" }

    func image() -> CGImage? { Self.decode(png) }

    /// `image()` off the main thread — what the canvas, the comparison and the large cells wait for:
    /// a 1024² PNG takes tens of milliseconds to decode, enough to stutter the strip under ← →.
    static func decoded(_ png: Data) async -> Decoded {
        await Task.detached(priority: .userInitiated) { Decoded(decode(png)) }.value
    }

    private static func decode(_ png: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(png as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    var legend: String {
        if let g = grid?.grid {
            let f = settings.format
            return String(localized: "Grid \(String(g.columns))×\(String(g.rows)) · \(settings.modelName) · \(String(f.width))×\(String(f.height)) · \(Figures.duration(seconds))")
        }
        let base = settings.isVariation
            ? String(localized: "\(settings.modelName) · \(String(width))×\(String(height)) · seed \(String(seed)) · variation · \(Figures.duration(seconds))")
            : String(localized: "\(settings.modelName) · \(String(width))×\(String(height)) · seed \(String(seed)) · \(Figures.duration(seconds))")
        // Only when it changes the image: Normal is every render's, it says nothing.
        return settings.detail == .normal ? base : base + " · " + settings.detail.label
    }
}

extension Detail {
    /// The rack's segment, and the history caption's « Detail: More ».
    var choice: String {
        switch self {
        case .normal: String(localized: "Normal", comment: "Detail level")
        case .more: String(localized: "More", comment: "Detail level")
        case .most: String(localized: "Most", comment: "Detail level")
        }
    }
    var label: String { String(localized: "Detail: \(choice)") }
}

/// An image decoded by a detached task, handed back to the main thread. `@unchecked Sendable`: a
/// `CGImage` is immutable.
struct Decoded: @unchecked Sendable {
    let image: CGImage?
    init(_ image: CGImage?) { self.image = image }
}

/// What the detached task makes of a `Render`: the PNG, the image for immediate display, the
/// thumbnail. `@unchecked Sendable`: three immutable values, built then only read.
struct FinishedImage: @unchecked Sendable {
    let png: Data
    let image: CGImage
    let vignette: CGImage

    init(_ renderResult: Engine.Render) throws {
        png = try renderResult.png()
        image = renderResult.image.cgImage()
        vignette = Self.reduce(image, side: 240) ?? image
    }

    /// Reduced for the strip (240 px on the longest side, ×2 for the Retina screen).
    static func reduce(_ image: CGImage, side: Int) -> CGImage? {
        let scale = min(1, Double(side) / Double(max(image.width, image.height)))
        let l = max(1, Int(Double(image.width) * scale)), h = max(1, Int(Double(image.height) * scale))
        guard let c = CGContext(data: nil, width: l, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        c.interpolationQuality = .high
        c.draw(image, in: CGRect(x: 0, y: 0, width: l, height: h))
        return c.makeImage()
    }
}

/// **A requested render**: everything needed to make it, taken at the moment of the click —
/// changing the prompt or the model afterwards does not affect it. It waits in the queue, in memory.
struct Job: Identifiable {
    let id = UUID()
    /// `t3`: numbered in the session, whether it comes from a click or from the remote control.
    let ordinal: Int
    let settings: RenderSettings
    let seeds: [UInt64]
    /// The edit's references, in order — image 1 is the one edited: re-read, not re-noised.
    let references: [ReferenceImage]
    let previews: Bool
    /// The model's default number of steps: `nil` in the request if it was not changed.
    let defaultSteps: Int?
    /// The denoiser's plan for one image (evaluations, including those reduced by the spectral), LoRA
    /// included — `nil` if the model is not installed. What, with the learned speed, estimates
    /// its duration before it runs. A sketch's plan is already cut to its evaluations.
    let plan: DenoisingPlan?
    /// **A sketch** (`Request.sketch`): the evaluations it stops after; its image is the estimate,
    /// kept by its exploration and never put in the history. `nil`: a finished image.
    var sketch: Int? = nil

    /// The key of the learned speed: same model, same format, same number of references (each one
    /// lengthens the sequence: one 1024² reference takes Klein from 10.7 to 26.6 s per evaluation,
    /// measured), with or without LoRA (the spectral turns off under a LoRA, and the LoRA weighs on
    /// every evaluation).
    var speedKey: String {
        Speed.key(settings.identifier, settings.format, references: references.count, lora: !settings.loras.isEmpty)
    }

    var name: String { "t\(ordinal)" }

    var legend: String {
        let n = seeds.count > 1 ? " ×\(seeds.count)" : ""
        return "\(settings.modelName) · \(settings.format.width)×\(settings.format.height)\(n)"
    }
}

/// **The learned speed** of a model at a format, drawn from a finished render: each stage in
/// seconds, and two step speeds — full and half-size (the spectral, ~0.35 of a full step).
struct Speed {
    var full: Double
    /// `nil` if the measured render had no reduced steps.
    var reducedAverage: Double?
    var text: Double
    var image: Double
    var decoding: Double

    /// One key per number of references: a speed learned with one reference says nothing of three
    /// (the sequence, the encoder's images, the condition cache all grow with each).
    static func key(_ identifier: String, _ format: RecommendedFormat, references: Int, lora: Bool) -> String {
        "\(identifier)|\(format.width)×\(format.height)\(references > 0 ? "|edit\(references)" : "")\(lora ? "|LoRA" : "")"
    }

    /// The duration of a render of `images` images on this plan. `image` (encoding the references)
    /// was measured with the same number of references: the key says so.
    func duration(_ plan: DenoisingPlan, images: Int) -> Double {
        let reduced = min(plan.reduced, plan.evaluations)
        let perImage = Double(plan.evaluations - reduced) * full
            + Double(reduced) * (reducedAverage ?? full) + decoding
        return text + image + Double(images) * perImage
    }
}

/// **How a job ended** — what `wait` returns to the remote control.
enum Issue {
    case finished
    /// Stopped while it was running ("Stop", `cancel`): the images already made remain.
    case stopped
    /// Removed from the queue before having run.
    case removed
    /// The engine refused or failed: its translated sentence, and its `EngineError.code`.
    case failed(Problem)
}

/// **Where a stage of the chain stands** — what the rack lights: waiting, working, done, or where the
/// render broke.
enum StageState { case idle, active, done, failed }

/// **The license to accept before a render** — the engine refused it (`licenseNotAccepted`): the jobs
/// of that model wait, the sheet shows the license, and accepting puts them back at the head of the queue.
struct LicenseRequest: Identifiable {
    let card: ModelCard
    var id: String { card.id }
}

/// **The diagnostic** (`Diagnostic.run`, ~15 s per model): what it measures, while it runs, then its report.
enum DiagnosticPhase {
    /// Not started: the sheet explains what it does, and how long.
    case ready
    /// Running: the models measured so far out of `total`, and the stage under way.
    case running(done: Int, total: Int, line: String)
    case finished(Diagnostic.Report)
}

/// **What the remote control listens to**: the lifecycle of each job, rendered by it or not.
enum Signal {
    case started(Job)
    case stage(Job, Engine.Stage, image: Int)
    case step(Job, image: Int, index: Int, total: Int, seconds: Double)
    case image(Job, Entry)
    case finished(Job, Issue)
    /// A grid's last cell ended: its sheet, or `nil` when no cell rendered (or the drawing failed).
    case sheet(GridRun, Entry?)
}

/// **The figures of a render**, drawn from its events — what the statistics window
/// shows. In memory, like the rest.
struct Measures {
    private(set) var job: Job?
    private(set) var begin: Date?
    private(set) var plan: Engine.Plan?
    private(set) var tokens: Int?
    private(set) var tokenizer: Double?, encoder: Double?
    private(set) var encoding: Double?
    /// Per image of the batch, the duration of each DiT evaluation.
    private(set) var stepSeconds: [[Double]] = []
    private(set) var decodings: [Double] = []
    /// The timers and footprints of the last finished image.
    private(set) var timings: Engine.Timings?
    private(set) var footprints: Engine.Footprints?

    mutating func beginJob(_ t: Job) {
        self = Measures()
        job = t
        begin = Date()
    }

    mutating func receive(_ item: Engine.Event) {
        switch item {
        case .start(let p):
            plan = p
            stepSeconds = Array(repeating: [], count: p.seeds.count)
        case let .text(j, t, e):
            tokens = j; tokenizer = t; encoder = e
        case .encoding(let s):
            encoding = s
        case let .step(image, _, _, _, seconds, _, _) where image < stepSeconds.count:
            stepSeconds[image].append(seconds)
        case .decoding(let s):
            decodings.append(s)
        case .image(_, let renderResult):
            timings = renderResult.timings
            footprints = renderResult.footprints
        default:
            break
        }
    }

    /// The average of the evaluations done; the first step re-reads the map, it counts like the others
    /// (it is also what the next render will pay).
    var secondsPerEvaluation: Double? {
        let combined = stepSeconds.flatMap { $0 }
        return combined.isEmpty ? nil : combined.reduce(0, +) / Double(combined.count)
    }

    /// The half-size steps of the spectral schedule: the first ones of each image.
    var reduced: Int { plan?.denoising.reduced ?? 0 }

    /// What this render learns about speed; `nil` without a full step.
    var speed: Speed? {
        let k = reduced
        let fullCount = stepSeconds.flatMap { $0.dropFirst(k) }, reduced = stepSeconds.flatMap { $0.prefix(k) }
        func mean(_ v: [Double]) -> Double? { v.isEmpty ? nil : v.reduce(0, +) / Double(v.count) }
        guard let full = mean(fullCount) else { return nil }
        return Speed(full: full, reducedAverage: mean(reduced), text: (tokenizer ?? 0) + (encoder ?? 0),
                       image: encoding ?? 0, decoding: mean(decodings) ?? Engine.Progress.decodingWeights * full)
    }
}

@MainActor @Observable
final class AppState {
    let library: Library
    /// The warnings of the machine profile (a profile measured elsewhere, for example).
    let profileWarnings: [String]
    /// All the known models — the publisher's, then the imported ones (`Library.cards()`).
    private(set) var cards: [ModelCard] = []
    /// The ready models (all their files present).
    private(set) var readySet: Set<String> = []

    // ── What the app shows ──
    /// **The families in front**: the two the product is. The others need
    /// Developer Mode.
    static let frontFamilies: Set<Family> = [.zImage, .qwenImage21]
    private static let developerModeKey = "developerMode"
    /// **Developer Mode**: the four other families.
    /// One of the app's two preferences (the other is `Curtain.preferred`) — written without a render
    /// gesture, because the user switched it.
    var developerMode = UserDefaults.standard.bool(forKey: AppState.developerModeKey) {
        didSet {
            guard developerMode != oldValue else { return }
            UserDefaults.standard.set(developerMode, forKey: Self.developerModeKey)
            // A model that leaves the list leaves the form too.
            if !isVisible(identifier) { identifier = defaultIdentifier }
        }
    }
    func isVisible(_ family: Family) -> Bool { developerMode || Self.frontFamilies.contains(family) }
    func isVisible(_ id: String) -> Bool { cards.first { $0.id == id }.map { isVisible($0.family) } ?? false }
    /// The models the list offers: the front families' (and their imported fine-tunes), or all of them.
    var visibleCards: [ModelCard] { cards.filter { isVisible($0.family) } }
    /// The visible models ready to render — empty at first launch: the Models sheet opens by itself.
    var visibleReady: [ModelCard] { visibleCards.filter { readySet.contains($0.id) } }
    /// Z-Image first (Apache 2.0); a non-commercial model is never the default unless it is the only one.
    private var defaultIdentifier: String { visibleReady.first?.id ?? ModelCard.zImage.id }

    // ── What the user sets ──
    var identifier = "" { didSet { if identifier != oldValue { modelChanged() } } }
    var prompt = ""
    var orientation = RecommendedFormat.Orientation.square { didSet { orientationChanged() } }
    var format = ModelCard.recommendedFormats[0]
    var loras: [LoRASlot] = []
    /// The edit's references, numbered from 1 in this order: **image 1 is the one edited** (it sets
    /// the output's format), the others are what the prompt borrows from (« the dog of image 2 »).
    /// At most the denoiser's `maxReferences`.
    var references: [ReferenceImage] = []
    var steps = 8
    /// **More detail** (`Detail`): Normal is the render as the model makes it.
    var detail = Detail.normal
    /// Fixed seed (the field) or drawn at random at each render, like Draw Things. A variation
    /// belongs to its seed: changing the seed, or letting it be drawn, drops it.
    var fixedSeed = false { didSet { if !fixedSeed { variations = [] } } }
    var seed: UInt64 = 42 { didSet { if seed != oldValue { variations = [] } } }
    /// **The variations the next render takes** (`Request.variations`) — only ever set by reusing a
    /// variation's settings (its menu, a dropped PNG), shown in the rack with a button that removes them.
    var variations: [Variation] = []
    var batch = 1
    var previews = true
    /// **The exploration's form** (`Exploration.swift`): its own prompt, a small format, the two axes.
    var explorePrompt = ""
    var exploreShape = ExploreShape.square
    var gridX = GridAxisForm(kind: .seed)
    var gridY = GridAxisForm(kind: .none)
    /// **The exploration shown**: its cells as they arrive, kept once ended — until the next one.
    private(set) var explored: GridRun?
    /// The canvas shows the exploration's grid rather than an image (`ExplorationCanvas`).
    var showsExploration = false
    /// A grid cell's long side on the canvas, in points: the slider, the pinch, ⌘+ ⌘−.
    var exploreCellSide: CGFloat = 220
    /// **Where the exploration's sketches stop**: 0, where the image is readable (`Sketch`); n, after n
    /// evaluations — more for a LoRA, whose style can settle late; at the plan's count, finished images.
    var exploreStop = 0
    static let cellSides: ClosedRange<CGFloat> = 96...1024
    static func clampedCellSide(_ s: CGFloat) -> CGFloat { min(max(s, cellSides.lowerBound), cellSides.upperBound) }

    // ── The chosen model's catalog: read from disk when it changes, not at each drawing ──
    /// The chosen model's chain — what the boxes draw. `nil` if it is not installed.
    private(set) var chain: Chain?
    /// The LoRAs forged for the chosen model.
    private(set) var compatibleLoras: [LoRACard] = []
    /// What the chosen model is missing (the library's message), or `nil` if it is ready.
    private(set) var missing: String?

    // ── What the render shows ──
    /// The running render, then those waiting — in the order they were requested.
    private(set) var currentJob: Job?
    /// The figures of the running render and of the last finished one — the statistics window.
    private(set) var measurements = Measures()
    /// **Seconds per evaluation**, per model and format, learned in this session (never
    /// written): that is what estimates the queue and the "Generate" button. A pair never rendered has
    /// no estimate.
    private(set) var speeds: [String: Speed] = [:]
    /// What the session cost: images, and seconds of computing.
    private(set) var sessionImages = 0
    private(set) var sessionSeconds = 0.0
    private(set) var file: [Job] = []
    var inProgress: Bool { currentJob != nil }
    /// A render is running or waiting: forges (several GB) and removing a model wait.
    var busy: Bool { inProgress || !file.isEmpty }
    private(set) var fraction = 0.0
    /// The time remaining at the last step or decoding, and when: between two events (a Krea 2
    /// step lasts 35 s), `remaining(at:)` counts it down instead of letting the predicted end slide.
    private(set) var estimatedRemaining: Double?
    private var remainingInstant = Date()
    private(set) var stage: Engine.Stage?
    private(set) var currentStep = 0
    private(set) var totalSteps = 0
    private(set) var batchImage = 0
    private(set) var batchSize = 1
    private(set) var preview: CGImage?
    private(set) var history: [Entry] = []
    var selection: Entry.ID? {
        didSet {
            guard selection != oldValue else { return }
            loadSelectedImage()
            // An image chosen (or a finished render followed): the canvas shows it, not the grid.
            if selection != nil { showsExploration = false }
        }
    }
    /// **The images marked in the strip**, beyond the one shown: ⌘-click adds or removes one, ⇧-click
    /// and ⇧-arrow mark a run, ⌘A marks all. Meaningful from two on — what Export, Copy, Share and
    /// Remove then act on (`targets`); a plain click or a new image clears it.
    private(set) var marked: Set<Entry.ID> = [] { didSet { if marked != oldValue { loadCompared() } } }
    /// **Two to four images marked are shown side by side** on the canvas — to choose between seeds —,
    /// decoded once per marking, like the chosen image.
    private(set) var compared: [(entry: Entry, image: CGImage)] = []
    static let maxCompared = 4
    /// **An edit against its image 1** (the bar's curtain button): the original under the output, at
    /// the output's size. Stays on from one edited image to the next; an image that is no edit shows alone.
    var comparesOriginal = false { didSet { loadOriginal() } }
    private(set) var original: (id: Entry.ID, image: CGImage)?
    var showsOriginal: Bool {
        comparesOriginal && marked.count <= 1 && original != nil && original?.id == selectedEntry?.id
    }
    func canCompareOriginal(_ e: Entry) -> Bool { e.grid == nil && e.settings.editing && !e.references.isEmpty }
    /// The thumbnails' height in the strip (the slider at its end); in memory, like the rest.
    var thumbnailSide: Double = 64
    /// Where a ⇧-run starts: the last image clicked without ⇧.
    private var anchor: Entry.ID?
    /// **The images that left the app** — saved, exported, written by `silicontrol save`. The others
    /// exist only in this process: quitting asks before losing them.
    private(set) var saved: Set<Entry.ID> = []
    var unsaved: [Entry] { history.filter { !saved.contains($0.id) } }
    /// The window's undo manager: removing images from the history is undone with ⌘Z.
    weak var undoManager: UndoManager?
    /// The canvas's zoom on the chosen image (1: fitted) — the pinch, and ⌘+ ⌘− ⌘0.
    var zoom: CGFloat = 1
    /// **The canvas follows the running render** — its preview, then each image it makes —, until a
    /// history image is chosen during it (`show`): the history is browsed while the render goes on, its
    /// images join the strip without taking the canvas. « Generate » and the live thumbnail follow it
    /// again.
    var followsRender = true
    /// The chosen image, decoded from its PNG — the only full-size image the app keeps.
    private(set) var selectedImage: (id: Entry.ID, image: CGImage)?
    private(set) var journal: [String] = []
    /// The last refusal or failure, translated (`Problems.swift`); closed by the user.
    var problem: Problem?
    /// **Each stage of the last render, as the engine's events moved it** — the rack's lights, for the
    /// model of that render. Reset when a job starts.
    private(set) var stageStates: [Engine.Stage: StageState] = [:]
    /// The model the lights belong to: those of another model, chosen since, stay off.
    private(set) var stageModel: String?
    /// **The license the engine asked for** — the sheet that shows it (`LicenseSheet.swift`).
    var licenseRequest: LicenseRequest?
    /// The jobs whose model's license the engine refused: they wait for the user's answer — out of the
    /// queue, so `silicontrol status` lists them apart (`"waiting_for_license"`).
    private(set) var awaitingLicense: [Job] = []
    /// **The family whose installation the Models sheet offers**: its license and its place on disk
    /// shown, waiting for « Accept and Install » — nothing is downloaded before.
    var installOffer: InstallOffer?
    struct InstallOffer: Equatable {
        let family: Family
        /// False: only what an imported model of the family lacks (encoder, VAE), not its DiT.
        var baseDiT = true
        /// The version to install, chosen in the panel: the one installed (else the preselected one)
        /// unless the user picks another — never another one when `baseDiT` is false (`offerInstall`).
        var variant: Variant = .standard
    }
    /// The diagnostic sheet, and where the diagnostic stands.
    var diagnosticOpen = false
    private(set) var diagnostic = DiagnosticPhase.ready
    private var diagnosticCancellation: Cancellation?
    var diagnosticRunning: Bool { if case .running = diagnostic { true } else { false } }
    /// The chosen image full screen, in the window (never a file opened elsewhere).
    var fullScreen = false
    private(set) var toast: String?
    /// An installation or an import in progress (several minutes, several GB) — and what it says.
    private(set) var forgeInProgress: String?
    private(set) var forgeJournal: [String] = []
    /// The token of the forge in progress: "Stop" cuts a download in the middle.
    private var cancellationForge: Cancellation?
    /// What the library occupies — the "Models" sheet; re-read with the catalog.
    private(set) var occupancy: Library.Occupancy?
    /// The "Models" sheet is open (⇧⌘M; by itself when nothing is installed).
    var managementOpen = false

    private var task: Task<Void, Never>?
    private var toastTask: Task<Void, Never>?
    /// **No App Nap during a render**: in the background (another app in the foreground, or
    /// the app opened by `silicontrol open`), macOS throttles the app — 64 s instead of 44
    /// for a measured 512². Held from the first job launched until the queue is empty.
    private var activity: NSObjectProtocol?

    // ── What the remote control reads (`RemoteControl.swift`) ──
    private var lastJob = 0
    private var lastImage = 0
    /// The session's finished jobs, their outcome and their images (even removed from the history
    /// since): `wait t3` afterwards still returns the answer.
    private(set) var completed: [Int: (job: Job, issue: Issue, images: [Int])] = [:]
    private var jobImages: [Int] = []
    /// **The grids whose cells are still on their way**, and which cell each job is.
    private(set) var gridRuns: [Int: GridRun] = [:]
    private var gridCells: [Int: (grid: Int, cell: Int)] = [:]
    /// The session's ended grids: their cells' jobs, and their sheet (`i9`), if one was drawn.
    private(set) var endedGrids: [Int: (jobs: [Int], sheet: Int?)] = [:]
    /// The grids whose cells have all ended and whose sheet is being drawn, with their cells' jobs:
    /// out of `gridRuns`, not yet in `endedGrids` — `wait g2` and `follow` must still see them.
    private(set) var drawingSheets: [Int: [Int]] = [:]
    private var lastGrid = 0
    /// Who listens to the signals, by token. Called on the main thread, in order.
    private var subscribers: [UUID: (Signal) -> Void] = [:]

    func subscribe(_ listener: @escaping (Signal) -> Void) -> UUID {
        let token = UUID()
        subscribers[token] = listener
        return token
    }
    func unsubscribe(_ token: UUID) { subscribers[token] = nil }
    private func signal(_ s: Signal) { for listener in subscribers.values { listener(s) } }

    private func complete(_ t: Job, _ issue: Issue, images: [Int] = []) {
        completed[t.ordinal] = (t, issue, images)
        signal(.finished(t, issue))
        if let at = gridCells.removeValue(forKey: t.ordinal), gridRuns[at.grid] != nil {
            gridRuns[at.grid]!.issues[at.cell] = issue
            if explored?.ordinal == at.grid { explored = gridRuns[at.grid] }
            if gridRuns[at.grid]!.isComplete { finishGrid(at.grid) }
        }
    }

    func nextJobNumber() -> Int { lastJob += 1; return lastJob }

    static let maxLoRA = 3
    /// A prompt's alternatives queue one render each, up to `PromptAlternatives.maxCombinations`:
    /// the queue holds that many, so that the largest series fits in an empty queue.
    static let maxQueueLength = PromptAlternatives.maxCombinations

    init(library: Library) {
        self.library = library
        // The machine profile, read in THIS library, once per process and before
        // any render: the app builds only one `AppState` (see `App.swift`).
        do {
            try EngineSettings.load(from: library)
            profileWarnings = EngineSettings.effective.warnings.map(Library.withoutHome)
        } catch {
            profileWarnings = [Library.withoutHome(error.localizedDescription)]
        }
        cards = library.cards()
        readySet = Set(library.models().map(\.id))
        occupancy = library.occupancy()
        identifier = defaultIdentifier
        readEditors()
    }

    // MARK: Catalog

    var card: ModelCard? { cards.first { $0.id == identifier } }
    var formats: [RecommendedFormat] { card?.formats ?? ModelCard.recommendedFormats }
    var orientationFormats: [RecommendedFormat] { formats.filter { $0.orientation == orientation } }
    func loraCard(_ path: String) -> LoRACard? { compatibleLoras.first { $0.path == path } }
    var canAddLoRA: Bool { loras.count < Self.maxLoRA && !compatibleLoras.isEmpty }
    /// How many references the chosen model reads (0: it does no editing).
    var maxReferences: Int { chain?.denoising.maxReferences ?? 0 }
    var editingPossible: Bool { maxReferences > 0 }
    /// The installed models that edit, read once per catalog: where « Edit with… » sends an image.
    private(set) var editingModels: Set<String> = []
    var editors: [ModelCard] { visibleReady.filter { editingModels.contains($0.id) } }
    private func readEditors() {
        editingModels = readySet.filter { (model($0)?.chain.denoising.maxReferences ?? 0) > 0 }
    }

    /// **The models, read once per catalog**: `Model.named` reads the map headers on disk, and the
    /// exploration's footer asks for each cell's model at every drawing. Emptied by `refreshCatalog`.
    @ObservationIgnored private var modelsRead: [String: Model?] = [:]
    func model(_ id: String) -> Model? {
        if let known = modelsRead[id] { return known }
        let m = try? Model.named(id, in: library)
        modelsRead[id] = .some(m)
        return m
    }
    var canAddReference: Bool { references.count < maxReferences }
    /// The model's encoder sees the images (`TextFormat.readsImages`), so the prompt can name them
    /// (`<image1>`, `<image2>`… in Qwen-Image-2.1's template). FLUX.2 [klein]'s is blind: its
    /// references only enter the DiT.
    var promptNamesImages: Bool { chain?.text.output.readsImages ?? false }

    /// **The format the render will have**: when editing, the one the denoiser derives from image 1
    /// (`DenoisingModule.editFormat`, the rule the developer's command line applies too), the chosen one
    /// otherwise.
    var outputFormat: RecommendedFormat {
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

    /// **The memory this render needs**: the model's measured peak at this size (`ModelCard.memoryNeed`),
    /// what the engine's preflight compares to the machine's memory.
    var memoryNeed: Int? {
        let f = outputFormat
        return card.map { $0.memoryNeed(width: f.width, height: f.height) }.flatMap { $0 > 0 ? $0 : nil }
    }

    /// Why "Generate" is grayed out, or `nil`: the app does not try what the request would refuse.
    /// The footer reads its two halves once per drawing (`formBlocker`, then `memoryBlocker`).
    var blocker: String? { formBlocker ?? memoryBlocker }

    /// Bumped every 2 s by the footer while « Generate » waits for memory, and only then: the machine's
    /// free memory is not observable, and without it « available now » stayed frozen — the button
    /// greyed out after the user had closed what took the memory.
    var memoryTick = 0

    /// What the form itself refuses: everything but the memory.
    var formBlocker: String? {
        if let reason = installBlocker { return reason }
        if missing != nil { return String(localized: "This model isn't installed.") }
        if prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return String(localized: "Write a prompt.") }
        let n: Int
        switch alternatives {
        case .failure(let e): return Problem(e).text
        case .success(let a): n = a.count
        }
        if file.count >= Self.maxQueueLength { return String(localized: "The queue is full.") }
        if file.count + n > Self.maxQueueLength {
            let room = Self.maxQueueLength - file.count
            return String(localized: "These alternatives make \(String(n)) renders: the queue has room for \(String(room)).")
        }
        if let denoising = chain?.denoising, (try? denoising.check(steps: steps, startImage: false)) == nil {
            let accepted = acceptedSteps.map(String.init).formatted(.list(type: .or))
            return String(localized: "This model has no schedule for \(String(steps)) steps: \(accepted).")
        }
        return nil
    }

    /// Rule 7: a render that cannot fit is refused before it starts — by the engine's own check
    /// (`ModelCard.checkMemory`, the machine's state read now), not after the click.
    var memoryBlocker: String? { memoryRefusal?.text }

    /// **A memory refusal, never a dead end**: what « Generate » waits for, and the ways forward —
    /// the largest of the model's formats that fits now, and who holds the memory. An ordinary
    /// desktop (a browser, mail, a few idle apps) is enough to refuse 1024²: a bare « needs X, Y is
    /// available » left the user nothing to do but guess.
    struct MemoryRefusal {
        let text: String
        /// The largest format that fits in the memory read now (`RecommendedFormat.largestFitting`),
        /// `nil` when none does — or when editing, where image 1 sets the size and the picker does not.
        let smaller: RecommendedFormat?
        /// « Safari, Mail and Messages are using 3.1 GB. » — `nil` when no other application weighs.
        let holders: String?
    }

    /// The budget is read **once** (`MemoryBudget.current()`): the check and every candidate format
    /// are judged against the same reading. The applications are listed only here, when the memory
    /// refuses: never at a drawing where « Generate » is free.
    var memoryRefusal: MemoryRefusal? {
        _ = memoryTick
        guard let card else { return nil }
        let f = outputFormat
        let budget = MemoryBudget.current()
        do { try card.checkMemory(width: f.width, height: f.height, budget: budget) } catch {
            guard case let .insufficientMemory(needed, available) = error else { return nil }
            let smaller = references.isEmpty
                ? RecommendedFormat.largestFitting(formats, available: available, current: f) {
                    card.memoryNeed(width: $0.width, height: $0.height)
                  }
                : nil
            return MemoryRefusal(
                text: String(localized: "This size needs \(Figures.memory(needed)) of memory, and \(Figures.memory(available)) is free right now. Generate comes back on its own as memory frees up."),
                smaller: smaller, holders: MemoryHolders.line())
        }
        return nil
    }

    /// The way forward the footer offers: the size that fits, chosen as the picker would.
    func useFormat(_ f: RecommendedFormat) {
        orientation = f.orientation
        format = f
    }

    /// **The step counts the chosen model has a schedule for**, asked of its denoiser
    /// (`DenoisingModule.check`) rather than known here: all of them for most, Viggle's 5, 6, 7 and 9
    /// for Qwen-Image-2.1. Read when "Generate" is greyed out for that reason, never at each drawing.
    var acceptedSteps: [Int] {
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
            return try RenderGrid(.init(prompt: explorePrompt, seed: seed, steps: steps,
                                        loraStrengths: stack.map(\.strength), images: references.count),
                                  x: x, y: try gridY.axis(stack: stack))
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
        guard let v = speeds[Speed.key(identifier, format, references: references.count, lora: withLoRA)] else { return nil }
        guard let plan = expectedPlan(steps: steps, format: format, withLoRA: withLoRA || detail != .normal) else { return nil }
        // One render per alternative, each the cost of one: same model, format, plan and batch.
        return Double(variantCount) * v.duration(plan, images: max(1, batch))
    }

    /// **One stage's share of that cost**, per image — what the rack writes under each box. Learned
    /// like the whole (`speeds`): `nil` until this model has rendered at this size in the session.
    func stageEstimate(_ stage: Engine.Stage) -> Double? {
        let withLoRA = loras.contains { !$0.path.isEmpty }
        let format = outputFormat
        guard let v = speeds[Speed.key(identifier, format, references: references.count, lora: withLoRA)] else { return nil }
        switch stage {
        case .text: return v.text
        case .image: return v.image
        case .decoding: return v.decoding
        case .denoising:
            guard let plan = expectedPlan(steps: steps, format: format, withLoRA: withLoRA || detail != .normal) else { return nil }
            let reduced = min(plan.reduced, plan.evaluations)
            return Double(plan.evaluations - reduced) * v.full + Double(reduced) * (v.reducedAverage ?? v.full)
        }
    }

    /// The rack's light for a stage of the chosen model.
    func state(of stage: Engine.Stage) -> StageState {
        stageModel == identifier ? stageStates[stage] ?? .idle : .idle
    }

    private func modelChanged() {
        readModelCatalog()
        // A LoRA from another model would be refused at render (`LoRA.Failure.wrongTarget`).
        let valid = Set(compatibleLoras.map(\.path))
        loras.removeAll { !$0.path.isEmpty && !valid.contains($0.path) }
        steps = card?.defaultSteps ?? 8
        if references.count > maxReferences { references = Array(references.prefix(maxReferences)) }
        // A size typed by hand stays: every model takes any size `Format.check` accepts.
        if (try? Format.check(width: format.width, height: format.height)) == nil {
            format = formats[0]; orientation = format.orientation
        }
    }

    /// **A size typed by hand** (`832x1216`, `1024`): read by `Format.parse`, judged by `Format.check` —
    /// the engine's own rules (sides ≥ 512, multiples of 16, area ≤ 1024×1536). The refusal, translated.
    func setCustomFormat(_ text: String) -> Problem? {
        guard let (w, h) = Format.parse(text.trimmingCharacters(in: .whitespaces)) else {
            return Problem(EngineError.formatUnreadable(text: text))
        }
        do { try Format.check(width: w, height: h) } catch { return Problem(error) }
        let f = RecommendedFormat(w, h)
        orientation = f.orientation
        format = f
        return nil
    }

    /// The sizes the picker offers: the model's for this orientation, and the one typed by hand.
    var pickerFormats: [RecommendedFormat] {
        orientationFormats.contains(format) ? orientationFormats : orientationFormats + [format]
    }

    /// Re-reads from disk what depends on the chosen model: its chain, its LoRAs, what it is missing.
    private func readModelCatalog() {
        chain = model(identifier)?.chain
        compatibleLoras = library.loras(for: identifier)
        missing = readySet.contains(identifier) ? nil : card?.missing(in: library)
    }

    private func orientationChanged() {
        if format.orientation != orientation, let first = orientationFormats.first { format = first }
    }

    // MARK: Reference (editing)

    // **No img2img in the app** (decided 2026-09-25): the re-noised starting image
    // (SDEdit) does not follow an instruction prompt and keeps a style only by losing the likeness.
    // The engine and the CLI keep it (`Request.image`, `--image`); the app offers only editing
    // by reference, where the model supports it.

    private func read(_ url: URL) -> ReferenceImage? {
        guard let vignette = NSImage(contentsOf: url), vignette.isValid, let size = ReferenceImage.size(of: url) else {
            problem = Problem(EngineError.imageUnreadable(file: url.path))
            return nil
        }
        return ReferenceImage(source: .file(url), vignette: vignette, name: url.lastPathComponent,
                              width: size.width, height: size.height)
    }

    func fromHistory(_ entry: Entry) -> ReferenceImage? {
        guard let cg = entry.image() else { return nil }
        return ReferenceImage(source: .image(cg), vignette: NSImage(cgImage: entry.vignette, size: .zero),
                             name: String(localized: "image from history (seed \(String(entry.seed)))"),
                             width: entry.width, height: entry.height)
    }

    /// The open panel: as many images as there are free slots, added after those already set.
    func chooseReferences() {
        guard canAddReference else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = maxReferences - references.count > 1
        panel.prompt = String(localized: "Choose")
        panel.message = references.isEmpty
            ? String(localized: "The image to edit: the prompt says what changes (“change her jacket to red”).")
            : String(localized: "Another image the prompt can draw from (“the dog of image 2”).")
        guard panel.runModal() == .OK else { return }
        addReferences(panel.urls)
    }

    /// Files chosen or dropped, appended in their order; beyond the model's maximum, the rest is
    /// left out and the toast says so.
    func addReferences(_ urls: [URL]) {
        // Files only: a dragged thumbnail (text) must not come back as a URL to read.
        let urls = urls.filter(\.isFileURL)
        guard editingPossible, !urls.isEmpty else { return }
        let free = max(0, maxReferences - references.count)
        references += urls.prefix(free).compactMap(read)
        if urls.count > free {
            flash(String(localized: "This model reads \(maxReferences) images at most"))
        }
    }

    /// A file chosen to take the place of the reference `id`, at its number.
    func replaceReference(_ id: ReferenceImage.ID) {
        guard references.contains(where: { $0.id == id }) else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.prompt = String(localized: "Choose")
        guard panel.runModal() == .OK, let url = panel.url, let image = read(url),
              let i = references.firstIndex(where: { $0.id == id }) else { return }
        references[i] = image
    }

    func removeReference(_ id: ReferenceImage.ID) {
        references.removeAll { $0.id == id }
    }

    /// Moves a reference to the place `index` (0: image 1), the others shifting. Image 1 is the one
    /// edited: putting another image first changes what is edited, and the output's format.
    func moveReference(_ id: ReferenceImage.ID, to index: Int) {
        guard let i = references.firstIndex(where: { $0.id == id }), i != index,
              references.indices.contains(index) else { return }
        references.insert(references.remove(at: i), at: index)
    }

    /// One place left (−1, towards image 1) or right (+1).
    func moveReference(_ id: ReferenceImage.ID, by offset: Int) {
        guard let i = references.firstIndex(where: { $0.id == id }) else { return }
        moveReference(id, to: i + offset)
    }

    /// "Edit this image": a history image becomes **image 1** — the one edited, whose format the
    /// output takes. It replaces the previous image 1 (editing a result again is the usual next
    /// gesture); the other references stay.
    func placeReference(_ entry: Entry) {
        guard editingPossible, let image = fromHistory(entry) else { return }
        if references.isEmpty { references = [image] } else { references[0] = image }
        flash(String(localized: "Image 1 set: say what changes"))
    }

    /// « Edit with Qwen-Image-2.1 »: the rack takes that model, and the image becomes its image 1.
    func edit(_ entry: Entry, with model: String) {
        if identifier != model { identifier = model }
        placeReference(entry)
    }

    /// PNG bytes dropped on the Edit box: a strip image is recognized by its bytes (its name, its
    /// curtain); any other PNG is taken as it is.
    func addReference(png data: Data) {
        guard editingPossible else { return }
        guard canAddReference else { flash(String(localized: "This model reads \(maxReferences) images at most")); return }
        if let entry = history.first(where: { $0.png == data }), let image = fromHistory(entry) {
            references.append(image)
        } else if let source = CGImageSourceCreateWithData(data as CFData, nil),
                  let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) {
            let vignette = FinishedImage.reduce(cg, side: 240) ?? cg
            references.append(ReferenceImage(source: .image(cg), vignette: NSImage(cgImage: vignette, size: .zero),
                                             name: String(localized: "dropped image"), width: cg.width, height: cg.height))
        } else {
            flash(String(localized: "This image can't be read"))
        }
    }

    // MARK: Render

    /// **Puts a render in the queue**, and launches it if nothing is running. `imposedSeed`: "Redo"
    /// (the seed of a history image). `imposedFormat`: an exploration cell's, finished at its sketch's
    /// size — with edit images, `outputFormat` would derive another one from image 1.
    func render(imposedSeed: UInt64? = nil, imposedFormat: RecommendedFormat? = nil) {
        guard canRender else { return }
        followsRender = true
        // Everything that reads the state is taken here: the job no longer depends on the form.
        let firstNumber = imposedSeed ?? (fixedSeed ? seed : UInt64.random(in: 0...UInt64(UInt32.max)))
        let format = imposedFormat ?? outputFormat
        guard let prompts = try? alternatives.get().prompts else { return }
        let plan = expectedPlan(steps: steps, format: format, withLoRA: loras.contains { !$0.path.isEmpty } || detail != .normal)
        // **One job per alternative, all at the same seeds**: a series compares the prompts, so
        // nothing else may differ. Each keeps the batch. Its history images carry the expanded prompt.
        enqueue(prompts.map { expanded in
            let settings = RenderSettings(identifier: identifier, modelName: card?.name ?? identifier, prompt: expanded,
                                          format: format, steps: steps, loras: loras.filter { !$0.path.isEmpty },
                                          references: references.count, variations: variations, detail: detail)
            return Job(ordinal: nextJobNumber(), settings: settings,
                       seeds: (0..<max(1, batch)).map { firstNumber &+ UInt64($0) },
                       references: references, previews: previews,
                       defaultSteps: card?.defaultSteps, plan: plan)
        })
    }

    /// **A grid enters the queue**: one ordinary job per cell, in the grid's order, and the run that
    /// will lay their images out — the path of the exploration and of `silicontrol grid`. `settings`:
    /// what the cells share, the prompt as typed. `sketch`: each cell stops as a sketch (`Sketch`), its
    /// image kept by the run, not the history, and no sheet is drawn unless asked (`addSheet`).
    @discardableResult
    func enqueueGrid(_ grid: RenderGrid, settings: RenderSettings, references: [ReferenceImage], previews: Bool,
                     sketch: Bool = false, stopAfter: Int = 0) -> (run: GridRun, jobs: [Job]) {
        lastGrid += 1
        let jobs = grid.cells.map { cell in
            var s = settings
            s.prompt = cell.prompt
            let card = cards.first { $0.id == (cell.model ?? settings.identifier) }
            if let m = cell.model { s.identifier = m; s.modelName = card?.name ?? m }
            s.steps = cell.steps ?? card?.defaultSteps ?? settings.steps
            for (i, f) in cell.loraStrengths.enumerated() where s.loras.indices.contains(i) { s.loras[i].strength = f }
            if let l = cell.addedLoRA { s.loras.append(LoRASlot(path: l, strength: cell.addedStrength)) }
            let used = cell.image.map { references.indices.contains($0) ? [references[$0]] : [] } ?? references
            s.references = used.count
            let model = try? Model.named(s.identifier, in: library)
            var plan = model.map { m in
                let f = m.chain.denoising.space.factor
                return m.chain.denoising.plan(height: s.format.height / f, width: s.format.width / f, steps: s.steps,
                                              start: 0, withLoRA: !s.loras.isEmpty || s.detail != .normal)
            }
            // A stop at or past the plan's count is a finished image — kept by the exploration all the same.
            let stop = !sketch ? nil : stopAfter > 0 ? min(stopAfter, plan?.evaluations ?? stopAfter)
                : model?.sketchEvaluations(steps: s.steps, width: s.format.width, height: s.format.height)
            if let stop { plan = plan?.sketched(stop) }
            var job = Job(ordinal: nextJobNumber(), settings: s, seeds: [cell.seed], references: used,
                          previews: previews, defaultSteps: card?.defaultSteps, plan: plan)
            job.sketch = stop
            return job
        }
        var run = GridRun(ordinal: lastGrid, grid: grid, settings: settings, jobs: jobs.map(\.ordinal),
                          references: references.count, sketch: sketch)
        run.cellJobs = jobs
        gridRuns[run.ordinal] = run
        for (i, job) in jobs.enumerated() { gridCells[job.ordinal] = (run.ordinal, i) }
        if sketch { explored = run }
        enqueue(jobs)
        return (run, jobs)
    }

    /// The grid and the cell a job renders, if it is one.
    func gridPlace(of job: Int) -> (grid: Int, cell: Int)? { gridCells[job] }

    /// **A grid's last cell ended**: its sheet is drawn off the main thread from the cells' PNGs —
    /// an empty cell says why —, then joins the history like an image. No cell rendered: no sheet.
    private func finishGrid(_ ordinal: Int) {
        guard let run = gridRuns.removeValue(forKey: ordinal) else { return }
        if explored?.ordinal == ordinal { explored = run }
        // An exploration's sheet is drawn only when asked (`addSheet`): the window is its sheet.
        guard !run.sketch else {
            endedGrids[ordinal] = (run.jobs, nil)
            signal(.sheet(run, nil))
            return
        }
        addSheet(run)
    }

    /// **A grid's sheet joins the history**: drawn off the main thread from the cells' PNGs — an empty
    /// cell says why. No cell rendered: no sheet.
    func addSheet(_ run: GridRun) {
        let ordinal = run.ordinal
        let entries = run.grid.cells.indices.compactMap { run.entries[$0] }
        guard let first = entries.first else {
            endedGrids[ordinal] = (run.jobs, nil)
            signal(.sheet(run, nil))
            return
        }
        let drawing = sheetDrawing(run)
        drawingSheets[ordinal] = run.jobs
        Task {
            do {
                let (png, vignette) = try await Task.detached(priority: .userInitiated) { try drawing.draw() }.value
                // What the cells cost together: their times summed, their memory peaks the highest.
                var timings = first.timings, footprints = first.footprints
                for e in entries.dropFirst() {
                    let t = e.timings, f = e.footprints
                    timings.tokenizer += t.tokenizer; timings.encoder += t.encoder; timings.text += t.text
                    timings.encoding += t.encoding; timings.denoising += t.denoising
                    timings.decoding += t.decoding; timings.total += t.total
                    footprints.start = max(footprints.start, f.start); footprints.tokenizer = max(footprints.tokenizer, f.tokenizer)
                    footprints.encoder = max(footprints.encoder, f.encoder); footprints.text = max(footprints.text, f.text)
                    footprints.encoding = max(footprints.encoding, f.encoding)
                    footprints.denoising = max(footprints.denoising, f.denoising); footprints.end = max(footprints.end, f.end)
                }
                lastImage += 1
                let entry = Entry(ordinal: lastImage, job: run.jobs.last ?? 0, png: png, vignette: vignette,
                                  width: drawing.layout.width, height: drawing.layout.height, seed: run.grid.base.seed,
                                  timings: timings, footprints: footprints,
                                  evaluations: entries.map(\.evaluations).reduce(0, +), settings: run.settings,
                                  metadata: drawing.metadata,
                                  grid: GridInfo(ordinal: run.ordinal, grid: run.grid, empty: run.jobs.count - entries.count))
                history.insert(entry, at: 0)
                if followsRender {
                    selection = entry.id
                    marked = []; anchor = entry.id
                }
                drawingSheets[ordinal] = nil
                endedGrids[ordinal] = (run.jobs, entry.ordinal)
                signal(.sheet(run, entry))
            } catch {
                problem = Problem(EngineError(error))
                drawingSheets[ordinal] = nil
                endedGrids[ordinal] = (run.jobs, nil)
                signal(.sheet(run, nil))
            }
        }
    }

    /// **What the sheet says**, in the user's language: the prompt as typed, what the cells share,
    /// the axes' values.
    private func sheetDrawing(_ run: GridRun) -> SheetDrawing {
        let g = run.grid, r = run.settings
        let stack = r.loras
        func lora(_ axis: RenderGrid.Axis?) -> Int? { if case .loraStrength(let slot, _)? = axis { slot } else { nil } }
        func labels(_ axis: RenderGrid.Axis) -> [String] {
            let name = lora(axis).flatMap { stack.indices.contains($0) ? loraName(stack[$0].path) : nil } ?? "LoRA"
            return g.values(of: axis).map { GridWords.label($0, loraName: name, modelName: modelName, addedName: loraName) }
        }
        func varies(_ test: (RenderGrid.Axis) -> Bool) -> Bool { test(g.x) || g.y.map(test) == true }
        // What every cell shares: what no axis varies. A model axis gives each its own steps.
        let models = varies { if case .models = $0 { true } else { false } }
        var shared = (models ? [] : [r.modelName]) + ["\(r.format.width) × \(r.format.height)"]
        if !models, !varies({ if case .steps = $0 { true } else { false } }) { shared.append(String(localized: "\(r.steps) steps")) }
        if !varies({ switch $0 { case .seeds, .consecutiveSeeds: true; default: false } }) {
            shared.append(String(localized: "seed \(String(g.base.seed))"))
        }
        for (i, l) in stack.enumerated() where lora(g.x) != i && lora(g.y) != i {
            shared.append("\(loraName(l.path)) \(Figures.strength(l.strength))")
        }
        if run.references > 0, !varies({ if case .images = $0 { true } else { false } }) {
            shared.append(String(localized: "edit, \(run.references) images"))
        }
        let axes = [g.x] + (g.y.map { [$0] } ?? [])
        let names = axes.map { GridWords.name($0, grid: g, stack: stack, loraName: loraName) }
        let layout = g.sheet(imageWidth: r.format.width, imageHeight: r.format.height)
        let cells = g.cells.enumerated().map { i, cell in
            SheetDrawing.Cell(column: cell.column, row: cell.row, png: run.entries[i]?.png,
                              note: GridWords.note(run.issues[i]))
        }
        return SheetDrawing(layout: layout, cells: cells, title: g.title, subtitle: shared.joined(separator: " · "),
                            columnLabels: labels(g.x), rowLabels: g.y.map(labels) ?? [],
                            metadata: ["Software": "Siliconed grid", "Title": g.title,
                                       "Description": "x: \(names[0])" + (names.count > 1 ? " · y: \(names[1])" : "")
                                           + " · " + shared.joined(separator: " · "),
                                       "grid": "\(g.columns)x\(g.rows)"])
    }

    /// Ready-made jobs enter the queue, in order — the path of the button and of the remote control.
    func enqueue(_ jobs: [Job]) {
        guard !jobs.isEmpty else { return }
        file.append(contentsOf: jobs)
        if inProgress { flash(String(localized: "Added to the queue (\(file.count) waiting)")) }
        next()
    }

    /// Launches the queue's first job if nothing is running. Called again at the end of each render,
    /// succeeded, stopped or failed: the queue empties by itself.
    /// The diagnostic loads weights like a render: the queue waits for it (and it for the queue) — and
    /// for an installation, which reads and writes GBs (`installBlocker`).
    private func next() {
        guard currentJob == nil, !file.isEmpty, !diagnosticRunning, forgeInProgress == nil else {
            if currentJob == nil, let a = activity {
                ProcessInfo.processInfo.endActivity(a); activity = nil
                DockTile.finished()
            }
            return
        }
        if activity == nil {
            activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .latencyCritical],
                                                             reason: "Image render")
        }
        let job = file.removeFirst()
        currentJob = job
        problem = nil; fraction = 0; estimatedRemaining = nil; stage = nil; preview = nil
        stageStates = [:]; stageModel = job.settings.identifier
        journal = []   // the warnings shown are this render's, not the last one that had any
        currentStep = 0; totalSteps = 0; batchImage = 0; batchSize = job.seeds.count
        measurements.beginJob(job)
        jobImages = []
        signal(.started(job))
        task = Task {
            let issue = await execute(job)
            currentJob = nil; preview = nil; task = nil
            if case .failed(let p) = issue, p.code == EngineError.licenseNotAccepted(model: "").code {
                // **The engine holds the door, the app presents**: the job waits for the answer.
                problem = nil
                awaitingLicense.append(job)
                if licenseRequest == nil, let card = cards.first(where: { $0.id == job.settings.identifier }) {
                    managementOpen = false
                    licenseRequest = LicenseRequest(card: card)
                }
            } else {
                complete(job, issue, images: jobImages)
            }
            next()
        }
    }

    /// **The user accepted the license**: the library records it (the engine reads it at every render),
    /// and the jobs that waited for it go back to the head of the queue, in their order.
    func acceptLicense(_ card: ModelCard) {
        do { try library.acceptLicense(card) } catch {
            problem = Problem(error)
            return
        }
        licenseRequest = nil
        let waiting = awaitingLicense.filter { $0.settings.identifier == card.id }
        awaitingLicense.removeAll { $0.settings.identifier == card.id }
        file.insert(contentsOf: waiting, at: 0)
        presentNextLicense()
        next()
    }

    /// **Declined**: the jobs that waited end with the engine's refusal, as they would have.
    func declineLicense(_ card: ModelCard) {
        licenseRequest = nil
        let refused = awaitingLicense.filter { $0.settings.identifier == card.id }
        awaitingLicense.removeAll { $0.settings.identifier == card.id }
        let p = Problem(EngineError.licenseNotAccepted(model: card.id))
        for job in refused { complete(job, .failed(p)) }
        problem = p
        presentNextLicense()
    }

    private func presentNextLicense() {
        guard licenseRequest == nil, let job = awaitingLicense.first,
              let card = cards.first(where: { $0.id == job.settings.identifier }) else { return }
        licenseRequest = LicenseRequest(card: card)
    }

    private func execute(_ job: Job) async -> Issue {
        let settings = job.settings
        let loraEntries = settings.loras.map { LoRAEntry($0.path, strength: $0.strength) }
        do {
            let model = try Model.named(settings.identifier, in: library)
            // The references are decoded off the main thread (3072 px at most), in their order; the
            // engine then brings each to its size: the full-size photos do not live during the render.
            let sources = job.references
            let references = try await Task.detached(priority: .userInitiated) {
                try sources.map { try $0.imageRGB() }
            }.value
            var request = Request(settings.prompt, width: settings.format.width,
                                  height: settings.format.height, seed: job.seeds[0],
                                  steps: settings.steps == job.defaultSteps ? nil : settings.steps,
                                  loras: loraEntries, previews: job.previews)
            request.references = references
            request.sketch = job.sketch
            request.detail = settings.detail
            request.variations = settings.variations

            var progress = Engine.Progress()
            for try await item in Engine().events(request, seeds: job.seeds, model: model) {
                progress.receive(item)
                measurements.receive(item)
                if progress.fraction != fraction { DockTile.show(fraction: progress.fraction, waiting: file.count) }
                fraction = progress.fraction
                if progress.estimatedRemaining != estimatedRemaining {
                    estimatedRemaining = progress.estimatedRemaining
                    remainingInstant = Date()
                }
                switch item {
                case .stage(let e, let image):
                    // The stage that was working is done; this one lights up.
                    if let previous = stage, stageStates[previous] == .active { stageStates[previous] = .done }
                    stageStates[e] = .active
                    stage = e
                    batchImage = image
                    signal(.stage(job, e, image: image))
                case let .step(image, index, total, _, seconds, _, _):
                    batchImage = image; currentStep = index; totalSteps = total
                    signal(.step(job, image: image, index: index, total: total, seconds: seconds))
                case .preview(let a):
                    preview = a.cgImage()
                case .image(_, let renderResult):
                    stageStates[.decoding] = .done
                    // The PNG and the thumbnail are made off the main thread; the engine has already
                    // moved on to the next image of the batch.
                    let finished = try await Task.detached(priority: .userInitiated) { try FinishedImage(renderResult) }.value
                    lastImage += 1
                    let entry = Entry(ordinal: lastImage, job: job.ordinal, png: finished.png, vignette: finished.vignette,
                                        width: renderResult.image.width, height: renderResult.image.height,
                                        seed: renderResult.seed, timings: renderResult.timings, footprints: renderResult.footprints,
                                        evaluations: renderResult.evaluations, settings: settings,
                                        metadata: renderResult.metadata, references: job.references,
                                        sketch: renderResult.sketch)
                    // A sketch belongs to its exploration, not to the history.
                    if job.sketch != nil, let at = gridCells[job.ordinal] {
                        gridRuns[at.grid]?.entries[at.cell] = entry
                        if explored?.ordinal == at.grid { explored = gridRuns[at.grid] }
                        preview = nil
                        sessionSeconds += renderResult.timings.total
                        jobImages.append(entry.ordinal)
                        signal(.image(job, entry))
                        continue
                    }
                    history.insert(entry, at: 0)
                    if followsRender {
                        selectedImage = (entry.id, finished.image)
                        selection = entry.id
                        marked = []; anchor = entry.id
                    }
                    preview = nil
                    sessionImages += 1
                    sessionSeconds += renderResult.timings.total
                    jobImages.append(entry.ordinal)
                    if let at = gridCells[job.ordinal] { gridRuns[at.grid]?.entries[at.cell] = entry }
                    signal(.image(job, entry))
                case .warning(let message):
                    journal.append(message)
                default:
                    break
                }
            }
            if let v = measurements.speed { speeds[job.speedKey] = v }
            // Cancelled, the `Task` leaves the loop WITHOUT an error (AsyncThrowingStream).
            if Task.isCancelled {
                for (s, v) in stageStates where v == .active { stageStates[s] = .idle }
                flash(String(localized: "Render stopped"))
                return .stopped
            }
            return .finished
        } catch {
            // Every refusal of the engine is one `EngineError` case (`EngineError(_:)` folds the rest).
            let e = EngineError(error)
            if let s = stage, stageStates[s] == .active { stageStates[s] = e == .cancelled ? .idle : .failed }
            if e == .cancelled {
                flash(String(localized: "Render stopped"))
                return .stopped
            }
            let p = Problem(e)
            problem = p
            return .failed(p)
        }
    }

    /// The per-image plan that the chosen model's denoiser foresees for these settings.
    /// `withLoRA`: a LoRA or a Detail — either turns Z-Image's spectral schedule off.
    private func expectedPlan(steps: Int, format: RecommendedFormat, withLoRA: Bool) -> DenoisingPlan? {
        guard let chain else { return nil }
        let f = chain.denoising.space.factor
        return chain.denoising.plan(height: format.height / f, width: format.width / f,
                                      steps: steps, start: 0, withLoRA: withLoRA)
    }

    /// The remaining time of the running render at instant `date`, counted down from the last event.
    func remaining(at date: Date) -> Double? {
        estimatedRemaining.map { max(0, $0 - date.timeIntervalSince(remainingInstant)) }
    }

    /// **The time remaining before the queue is empty**, and whether it is complete: the running render
    /// (its own estimate), then each wait at the learned pace for its model and format.
    /// A wait never measured makes the estimate partial — a minimum, said as such.
    func remainingInQueue(at date: Date) -> (seconds: Double, isComplete: Bool) {
        let inProgress = remaining(at: date)
        var total = inProgress ?? 0
        var isComplete = !self.inProgress || inProgress != nil
        for t in file {
            if let e = estimation(t) { total += e } else { isComplete = false }
        }
        return (total, isComplete)
    }

    /// The planned duration of a waiting job, or `nil` if it was never measured.
    func estimation(_ t: Job) -> Double? {
        guard let v = speeds[t.speedKey], let plan = t.plan else { return nil }
        return v.duration(plan, images: t.seeds.count)
    }

    /// Stops the running render; the queue continues. The engine checks for cancellation between the
    /// layers: stopping takes 0.03 to 0.41 s.
    func cancel() { task?.cancel() }

    /// Removes a render that is still waiting — in the queue, or for its model's license.
    func remove(_ job: Job) {
        if awaitingLicense.contains(where: { $0.id == job.id }) {
            awaitingLicense.removeAll { $0.id == job.id }
            if let id = licenseRequest?.card.id, !awaitingLicense.contains(where: { $0.settings.identifier == id }) {
                licenseRequest = nil
                presentNextLicense()
            }
            complete(job, .removed)
            return
        }
        guard file.contains(where: { $0.id == job.id }) else { return }
        file.removeAll { $0.id == job.id }
        complete(job, .removed)
    }

    /// Empties the queue and stops the running render.
    func stopAll() {
        clearQueue()
        task?.cancel()
    }

    private func clearQueue() {
        let removedNames = file + awaitingLicense
        file.removeAll()
        awaitingLicense.removeAll()
        licenseRequest = nil
        for t in removedNames { complete(t, .removed) }
    }

    /// Whether « Variations » can take this image: an ordinary history image of an installed model,
    /// no installation running.
    func canVary(_ entry: Entry) -> Bool {
        installBlocker == nil && entry.grid == nil && entry.sketch == nil && readySet.contains(entry.settings.identifier)
    }

    /// **« Variations »**: cousins of a history image — exactly its settings (model, prompt, LoRA,
    /// size, steps, seed, edit images), each with a variation seed drawn at random at `strength`
    /// (`Variation.Amount`, its strength for the model's family), ordinary jobs whose images join the history.
    /// **As many as the rack's « Images »** (`batch`, 1 by default): at ERNIE's ~200 s an image, a fixed
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
        let plan = (try? Model.named(entry.settings.identifier, in: library)).map { m in
            let f = m.chain.denoising.space.factor
            return m.chain.denoising.plan(height: entry.settings.format.height / f, width: entry.settings.format.width / f,
                                          steps: entry.settings.steps, start: 0, withLoRA: !entry.settings.loras.isEmpty || entry.settings.detail != .normal)
        }
        let jobs = (0..<(count ?? batch)).map { _ in
            var settings = entry.settings
            settings.variations.append(Variation(seed: UInt64.random(in: 0...UInt64(UInt32.max)), strength: strength))
            return Job(ordinal: nextJobNumber(), settings: settings, seeds: [entry.seed], references: entry.references,
                       previews: previews, defaultSteps: card?.defaultSteps, plan: plan)
        }
        enqueue(jobs)
        return jobs
    }

    /// Redo the chosen image: same settings, same seed (same bits if `reproducible`).
    func redo(_ entry: Entry) {
        restoreSettings(entry)
        // A sheet: its exploration again, at its seed.
        if entry.grid != nil { explore(seed: entry.seed); return }
        render(imposedSeed: entry.seed)
    }

    func restoreSettings(_ entry: Entry) {
        restoreSettings(entry.settings)
        // Fixed, or the next render draws a new seed and the settings reused are not the image's.
        seed = entry.seed
        fixedSeed = true
        // After the seed: setting it drops a variation (`seed.didSet`).
        variations = entry.settings.variations
        if let g = entry.grid?.grid {
            // A sheet's prompt is the one typed, its groups the grid's axes: not escaped.
            setGridForm(g, prompt: entry.settings.prompt)
            MainWindow.shared.showExploration()
        }
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
        loras = r.loras
        detail = r.detail
    }

    /// **A Siliconed PNG dropped on the canvas**: the form takes what its metadata says
    /// (`Engine.Render.Recipe`), through `restoreSettings` as "Reuse These Settings" does, and the
    /// seed is fixed. What cannot come back is said (a `Problem`), and the rest is restored anyway.
    /// `false` when the file is not a PNG written by Siliconed: the drop keeps its usual meaning.
    ///
    /// Incognito: the file is only read — no panel, no recent item, no copy.
    func restoreSettings(fromPNG url: URL) -> Bool {
        guard url.isFileURL, url.pathExtension.lowercased() == "png",
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let recipe = Engine.Render.Recipe(metadata: PNG.text(data)) else { return false }
        let file = url.lastPathComponent
        var notes: [String] = []

        // The PNG names a family, not a card (a fine-tune writes its family's name): the form's
        // model if it is of that family, otherwise the publisher's card, otherwise an imported one.
        let ready = visibleCards.filter { $0.family.rawValue == recipe.model && readySet.contains($0.id) }
        let target = ready.first { $0.id == identifier } ?? ready.first { !$0.isImported } ?? ready.first
        if target == nil, let model = recipe.model {
            // Steps and LoRAs belong to that model: the form's stay.
            let name = cards.first { $0.family.rawValue == model }?.name ?? model
            notes.append(String(localized: "\(name) isn't installed: its steps and LoRAs are left out."))
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
        }

        var restoredFormat = format
        if let w = recipe.width, let h = recipe.height, (try? Format.check(width: w, height: h)) != nil {
            restoredFormat = RecommendedFormat(w, h)
        }
        if recipe.sourceImageMissing {
            notes.append(String(localized: "The image it started from isn't in the file (img2img)."))
        }
        if !recipe.unreadable.isEmpty {
            let keys = recipe.unreadable.formatted(.list(type: .and))
            notes.append(String(localized: "Unreadable in the file: \(keys)."))
        }

        let model = target ?? card
        restoreSettings(RenderSettings(identifier: model?.id ?? identifier, modelName: model?.name ?? "",
                                       prompt: recipe.prompt ?? prompt, format: restoredFormat,
                                       steps: target == nil ? steps : (recipe.steps ?? target!.defaultSteps),
                                       loras: stack, detail: recipe.detail))
        if let s = recipe.seed { seed = s; fixedSeed = true }
        // A variation is its seed's: without the seed, there is nothing to vary.
        variations = recipe.seed == nil ? [] : recipe.variations
        flash(String(localized: "Settings of \(file) restored"))
        if !notes.isEmpty {
            problem = Problem(String(localized: "Not everything came back from \(file)."),
                              suggestion: notes.joined(separator: "\n"))
        }
        return true
    }

    // MARK: History and outputs

    var selectedEntry: Entry? {
        history.first { $0.id == selection } ?? history.first
    }

    // **The decodings are off the main thread** (`Entry.decodedImage`): each loader keeps its task, a
    // newer request cancels the older, and a result lands only if it is still the one asked for. The
    // canvas keeps what it showed until then — tens of milliseconds.
    @ObservationIgnored private var comparedLoad: Task<Void, Never>?
    @ObservationIgnored private var selectedLoad: Task<Void, Never>?
    @ObservationIgnored private var originalLoad: Task<Void, Never>?

    private func loadCompared() {
        comparedLoad?.cancel()
        let entries = marked.count > 1 && marked.count <= Self.maxCompared ? history.filter { marked.contains($0.id) } : []
        guard !entries.isEmpty else { compared = []; return }
        let kept = Dictionary(compared.map { ($0.entry.id, $0.image) }, uniquingKeysWith: { a, _ in a })
        let marking = marked
        comparedLoad = Task { [weak self] in
            var loaded: [(entry: Entry, image: CGImage)] = []
            for e in entries {
                let image: CGImage? = if let k = kept[e.id] { k } else { await Entry.decoded(e.png).image }
                if let image { loaded.append((e, image)) }
            }
            guard let self, !Task.isCancelled, self.marked == marking else { return }
            self.compared = loaded
        }
    }

    /// The full-size image of the chosen entry, decoded once per choice.
    private func loadSelectedImage() {
        defer { loadOriginal() }
        selectedLoad?.cancel()
        guard let e = selectedEntry else { selectedImage = nil; return }
        guard selectedImage?.id != e.id else { return }
        let id = e.id, png = e.png
        selectedLoad = Task { [weak self] in
            let image = await Entry.decoded(png).image
            guard let self, !Task.isCancelled, self.selectedEntry?.id == id else { return }
            self.selectedImage = image.map { (id, $0) }
        }
    }

    /// The chosen edit's image 1, decoded only while its curtain is asked for — up to 3072 px read.
    private func loadOriginal() {
        originalLoad?.cancel()
        guard comparesOriginal, let e = selectedEntry, canCompareOriginal(e) else { original = nil; return }
        guard original?.id != e.id else { return }
        let id = e.id, reference = e.references[0], width = e.width, height = e.height
        originalLoad = Task { [weak self] in
            let image = await Task.detached(priority: .userInitiated) {
                Decoded(reference.displayed(width: width, height: height))
            }.value.image
            guard let self, !Task.isCancelled, self.selectedEntry?.id == id, self.comparesOriginal else { return }
            self.original = image.map { (id, $0) }
        }
    }

    /// A history image chosen by the user: during a render, the canvas leaves the render for it.
    func show(_ id: Entry.ID) {
        selection = id
        marked = []
        anchor = id
        if inProgress { followsRender = false }
    }

    /// **A click on a thumbnail**, read with its modifiers as the Finder reads them: alone, it shows
    /// the image; with ⌘, it adds it to the marked ones or takes it out; with ⇧, it marks the run
    /// from the last plain click.
    func click(_ id: Entry.ID, modifiers: NSEvent.ModifierFlags) {
        if modifiers.contains(.shift), let a = anchor ?? selection {
            markRun(from: a, to: id)
        } else if modifiers.contains(.command) {
            var m = marked.count > 1 ? marked : Set([selection].compactMap { $0 })
            if m.contains(id), m.count > 1 {
                m.remove(id)
                if selection == id, let other = history.first(where: { m.contains($0.id) }) { reveal(other.id) }
            } else {
                m.insert(id)
                reveal(id)
            }
            marked = m.count > 1 ? m : []
            anchor = id
        } else {
            show(id)
        }
    }

    /// Shows an image without touching what is marked.
    private func reveal(_ id: Entry.ID) {
        selection = id
        if inProgress { followsRender = false }
    }

    private func markRun(from a: Entry.ID, to b: Entry.ID) {
        guard let i = history.firstIndex(where: { $0.id == a }), let j = history.firstIndex(where: { $0.id == b }) else { return }
        let run = Set(history[min(i, j)...max(i, j)].map(\.id))
        marked = run.count > 1 ? run : []
        reveal(b)
    }

    /// ⌘A in the strip: every image.
    func markAll() {
        guard history.count > 1 else { return }
        marked = Set(history.map(\.id))
        if selection == nil { selection = history.first?.id }
    }

    /// Esc: back to the one image shown.
    func unmark() { marked = [] }

    /// Whether a thumbnail wears the selection's edge: the image shown, or one of those marked.
    func isMarked(_ id: Entry.ID) -> Bool {
        marked.count > 1 ? marked.contains(id) : id == selectedEntry?.id
    }

    /// **What the image commands act on**: the marked images, in the strip's order — or the one shown.
    var targets: [Entry] {
        marked.count > 1 ? history.filter { marked.contains($0.id) } : selectedEntry.map { [$0] } ?? []
    }

    /// The neighbouring image in the strip: −1 to the left (more recent), +1 to the right (older), as
    /// the arrow keys move along it. Watching a render, it starts from the most recent image. With
    /// `extending` (⇧), the run from the anchor is marked.
    func chooseAdjacent(_ offset: Int, extending: Bool = false) {
        guard let e = selectedEntry, let i = history.firstIndex(where: { $0.id == e.id }) else { return }
        if inProgress && followsRender { show(history[offset > 0 ? i : 0].id); return }
        let j = min(max(i + offset, 0), history.count - 1)
        if extending, let a = anchor { markRun(from: a, to: history[j].id) } else { show(history[j].id) }
    }

    /// Home, End: the most recent image, the oldest.
    func chooseEnd(oldest: Bool) {
        guard let e = oldest ? history.last : history.first else { return }
        show(e.id)
    }

    func delete(_ entry: Entry) { delete([entry]) }

    /// **Removes images from the history** — undone by ⌘Z, each back at its place. The image shown
    /// passes to the nearest one left.
    func delete(_ entries: [Entry]) {
        let ids = Set(entries.map(\.id))
        let removed = history.enumerated().filter { ids.contains($0.element.id) }
        guard !removed.isEmpty else { return }
        let first = removed[0].offset
        history.removeAll { ids.contains($0.id) }
        marked = []
        if let s = selection, ids.contains(s) {
            selection = history.isEmpty ? nil : history[min(first, history.count - 1)].id
        }
        anchor = selection
        loadSelectedImage()
        undoManager?.registerUndo(withTarget: self) { app in
            for (i, e) in removed { app.history.insert(e, at: min(i, app.history.count)) }
            app.show(removed[0].element.id)
            app.marked = removed.count > 1 ? ids : []
            app.undoManager?.registerUndo(withTarget: app) { $0.delete(removed.map(\.element)) }
        }
        undoManager?.setActionName(removed.count == 1 ? String(localized: "Remove Image")
                                                       : String(localized: "Remove \(removed.count) Images"))
    }

    /// Everything, after a confirmation that says how many were never saved.
    func clearHistory() {
        guard !history.isEmpty else { return }
        let alert = NSAlert()
        alert.messageText = String(localized: "Clear the history?")
        let n = unsaved.count
        alert.informativeText = n == 0
            ? String(localized: "Every image has been saved. You can undo this with ⌘Z.")
            : String(localized: "\(n) of these images were never saved. You can undo this with ⌘Z while the app is open.")
        alert.addButton(withTitle: String(localized: "Clear History"))
        alert.addButton(withTitle: String(localized: "Cancel"))
        alert.buttons[0].hasDestructiveAction = true
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        delete(history)
    }

    /// Save: PNG by default (it carries the prompt, the seed, the model), JPEG optionally.
    func save(_ entry: Entry) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png, .jpeg]
        panel.nameFieldStringValue = exportName(entry) + ".png"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try write(entry, to: url)
            flash(String(localized: "Saved: \(url.lastPathComponent)"))
        } catch {
            problem = Problem((error as? EngineError) ?? .imageWriteFailed(file: url.path))
        }
    }

    /// **Export**: one image is « Save As »; several go into a chosen folder, each under its own name
    /// (`exportName`), as PNG or JPEG — `true` once written.
    @discardableResult
    func export(_ entries: [Entry]) -> Bool {
        guard !entries.isEmpty else { return false }
        if entries.count == 1 {
            save(entries[0])
            return saved.contains(entries[0].id)
        }
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.prompt = String(localized: "Export")
        panel.message = String(localized: "Export \(entries.count) images to a folder. Each keeps its prompt, seed and model in its metadata.")
        let kind = NSPopUpButton(frame: .zero, pullsDown: false)
        kind.addItems(withTitles: ["PNG", "JPEG"])
        let label = NSTextField(labelWithString: String(localized: "Format:"))
        let accessory = NSStackView(views: [label, kind])
        accessory.edgeInsets = NSEdgeInsets(top: 10, left: 16, bottom: 10, right: 16)
        panel.accessoryView = accessory
        panel.isAccessoryViewDisclosed = true
        guard panel.runModal() == .OK, let folder = panel.url else { return false }
        let ext = kind.indexOfSelectedItem == 1 ? "jpg" : "png"
        do {
            for e in entries { try write(e, to: Self.freeURL(in: folder, base: exportName(e), ext: ext)) }
        } catch {
            problem = Problem((error as? EngineError) ?? .imageWriteFailed(file: folder.path))
            return false
        }
        flash(String(localized: "\(entries.count) images exported to \(folder.lastPathComponent)"))
        return true
    }

    /// **An image's file name**: the model, its LoRAs, the steps if they are not the model's, the
    /// format, the seed — two different settings never share a name (`silicontrol save` writes the same).
    func exportName(_ e: Entry) -> String {
        let r = e.settings
        if let g = e.grid?.grid {
            // `grid-z-image-2x2-512x512-42`: the grid, the cells' format, the base seed (the first of
            // `seeds=N`) — none when the seeds are listed: no cell need have it.
            let listed = [g.x, g.y].contains { if case .seeds? = $0 { true } else { false } }
            return (["grid", r.identifier.replacingOccurrences(of: "/", with: "-"), "\(g.columns)x\(g.rows)",
                     "\(r.format.width)x\(r.format.height)"] + (listed ? [] : ["\(e.seed)"])).joined(separator: "-")
        }
        let modelSteps = cards.first { $0.id == r.identifier }?.defaultSteps
        return ([r.identifier.replacingOccurrences(of: "/", with: "-")]
                + r.loras.map { "lora-" + RemoteControl.shortName($0.path) + ($0.strength == 1 ? "" : String(format: "@%.2f", $0.strength)) }
                + (r.steps != modelSteps ? ["p\(r.steps)"] : [])
                + (r.editing ? ["edit"] : [])
                // A sketch never takes a finished image's name.
                + (e.sketch.map { ["sketch\($0)"] } ?? [])
                + ["\(e.width)x\(e.height)", "\(e.seed)"]
                // A variation is another image of the same seed: `v<seed>x<strength in hundredths>`, in
                // order — the CLI's name for it.
                + r.variations.map { "v\($0.seed)x\(Int(($0.strength * 100).rounded()))" }).joined(separator: "-")
    }

    /// `base.png` in the folder, or `base-2.png`, `base-3.png`… never over an existing file.
    static func freeURL(in folder: URL, base: String, ext: String) -> URL {
        var url = folder.appendingPathComponent("\(base).\(ext)")
        var k = 2
        while FileManager.default.fileExists(atPath: url.path) {
            url = folder.appendingPathComponent("\(base)-\(k).\(ext)")
            k += 1
        }
        return url
    }

    /// Writes an image as its extension says (JPEG at 0.9, or the PNG with its metadata), and counts it
    /// as saved.
    func write(_ entry: Entry, to url: URL) throws {
        if ["jpg", "jpeg"].contains(url.pathExtension.lowercased()) {
            guard let cg = entry.image(),
                  let jpeg = NSBitmapImageRep(cgImage: cg).representation(using: .jpeg, properties: [.compressionFactor: 0.9]) else {
                throw EngineError.imageWriteFailed(file: url.path)
            }
            try jpeg.write(to: url, options: .atomic)
        } else {
            try PNG.write(entry.png, to: url.path)
        }
        saved.insert(entry.id)
    }

    func copyImage(_ entry: Entry) { copyImages([entry]) }

    /// One pasteboard item per image (PNG): pasted into Finder, Photos or a message, each arrives.
    func copyImages(_ entries: [Entry]) {
        guard !entries.isEmpty else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.writeObjects(entries.map { e in
            let item = NSPasteboardItem()
            item.setData(e.png, forType: .png)
            return item
        })
        flash(entries.count == 1 ? String(localized: "Image copied") : String(localized: "\(entries.count) images copied"))
    }

    /// What a drag out of the strip or the canvas carries, and what Share sends: the PNG, under its
    /// export name. Nothing is written before the drop: the receiver writes it.
    func transferable(_ e: Entry) -> PNGImage { PNGImage(data: e.png, name: exportName(e) + ".png") }

    /// The `silicontrol add` command that redoes what the form describes — the equivalent of Draw
    /// Things Headless's "Copy as curl", for the `silicontrol` command shipped in the bundle. The
    /// references go as `--ref`, in order; one that is not a file has no command form: the command
    /// says so in a comment.
    func command() -> String {
        func quotes(_ s: String) -> String { "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'" }
        var pieces = ["silicontrol", "add", "--model", identifier]
        for l in loras where !l.path.isEmpty {
            let file = ((l.path as NSString).lastPathComponent as NSString).deletingPathExtension
            pieces += ["--lora", quotes("\((file as NSString).deletingPathExtension):\(String(format: "%.2f", l.strength))")]
        }
        if steps != card?.defaultSteps { pieces += ["--steps", "\(steps)"] }
        var notes: [String] = []
        for (n, reference) in references.enumerated() {
            if case .file(let url) = reference.source {
                pieces += ["--ref", quotes(url.path)]
            } else {
                notes.append(String(localized: "image \(n + 1) comes from the history: save it, then --ref"))
            }
        }
        if batch > 1 { pieces += ["--batch", "\(batch)"] }
        pieces.append(previews ? "--preview" : "--no-preview")
        let format = outputFormat
        pieces += [quotes(prompt), "\(format.width)x\(format.height)",
                     "\(fixedSeed ? seed : (selectedEntry?.seed ?? seed))"]
        // The app's own library needs no mention; another one is the app's environment.
        let root = library.root == Library.standard.root ? ""
            : "SILICONED_ROOT=" + quotes(library.root.path) + " "
        return root + pieces.joined(separator: " ") + notes.map { "\n# \($0)" }.joined()
    }

    func copyCommand() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(command(), forType: .string)
        flash(String(localized: "Command copied"))
    }

    /// A transient confirmation ("Image copied") that clears by itself.
    func flash(_ message: String) {
        toast = message
        toastTask?.cancel()
        toastTask = Task {
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled else { return }
            toast = nil
        }
    }

    // MARK: Import, install

    /// Re-reads the library: after an import, an installation, a removal.
    func refreshCatalog() {
        cards = library.cards()
        readySet = Set(library.models().map(\.id))
        occupancy = library.occupancy()
        modelsRead = [:]
        readModelCatalog()
        readEditors()
    }

    /// The chosen model's family, if it must be installed to render — an imported model requires
    /// only its components and the encoder.
    var familyToInstall: Family? { missing != nil ? card?.family : nil }

    /// A forge, a render, a diagnostic: one waits for the other (all read or write GBs).
    var managementPossible: Bool { forgeInProgress == nil && !busy && !diagnosticRunning }

    /// **No render while an installation or an import runs**: one big job at a time on 16 GB — the
    /// forge reads and writes GBs, a render maps as many. Why « Generate », « Explore » and the
    /// variations are greyed out, and what `silicontrol add` answers (`busy`); the queue waits too.
    var installBlocker: String? {
        forgeInProgress == nil ? nil : String(localized: "An installation is running: renders can start once it is done.")
    }

    /// What an import takes: a `.safetensors` (a LoRA, or a DiT in bf16, fp16, fp32 or 8 bits as
    /// published) or a `.gguf` (a DiT, 4 bits at least). The forge refuses the rest and says why.
    static let importExtensions: Set<String> = ["safetensors", "gguf"]
    static func importable(_ url: URL) -> Bool { importExtensions.contains(url.pathExtension.lowercased()) }

    /// Choose a `.safetensors` or a `.gguf` (LoRA or model, as published on Civitai) and import it.
    func chooseToImport() {
        guard managementPossible else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = Self.importExtensions.sorted().map { UTType(filenameExtension: $0) ?? .data }
        panel.allowsMultipleSelection = true
        panel.prompt = String(localized: "Import")
        panel.message = String(localized: "A model or a LoRA of a known family, as published (Civitai, Hugging Face). The encoder and VAE come with the family: there is nothing else to provide.")
        guard panel.runModal() == .OK else { return }
        perform(panel.urls)
    }

    /// Imports `.safetensors` and `.gguf` files, one after the other. The file brought in may be discarded afterwards.
    func perform(_ urls: [URL]) {
        let files = urls.filter(Self.importable)
        guard !files.isEmpty, managementPossible else { return }
        let names = files.map(\.lastPathComponent).joined(separator: ", ")
        launchForge(String(localized: "Importing \(names)")) { b, cancellation, journal in
            try files.map { try ModelImport.importFile($0.path, in: b, cancellation: cancellation, journal: journal) }
        } end: { [weak self] (results: [ModelImport.Outcome]) in
            guard let self else { return }
            for r in results {
                if let id = r.identifier {
                    identifier = id
                    flash(String(localized: "Model imported: \(r.name)"))
                } else if compatibleLoras.contains(where: { $0.path == r.path }) && loras.count < Self.maxLoRA {
                    loras.append(LoRASlot(path: r.path))
                    flash(String(localized: "LoRA imported: \(r.name)"))
                } else {
                    flash(String(localized: "LoRA imported for \(r.family.name): \(r.name)"))
                }
            }
        }
    }

    /// **Offers to install a family**: the Models sheet opens on it, with its license and the place it
    /// takes on disk. Nothing is downloaded before « Accept and Install » (`acceptAndInstall`).
    /// `variant`: the version preselected in the panel — by default the one installed, else
    /// `Family.preselectedVariant` (the publisher's weights unless the user picks another; the
    /// Light preselected on a Mac of 8 GB, shown among the three).
    /// `baseDiT` false (what an imported model lacks) never changes the version: the one installed is
    /// imposed, Standard if none, whatever `variant` says (`Library.installPlan`).
    func offerInstall(_ family: Family, baseDiT: Bool = true, variant: Variant? = nil) {
        let installed = library.installedVariant(of: family)
        installOffer = InstallOffer(family: family, baseDiT: baseDiT,
                                    variant: baseDiT ? variant ?? installed ?? family.preselectedVariant() : installed ?? .standard)
        managementOpen = true
    }

    /// **The user read the license and the size, and accepted**: the acceptance is recorded in the
    /// library first (the engine will ask for it at every render), then the first byte is downloaded.
    /// `baseDiT` false: what an imported model of the family lacks (encoder, VAE), not its DiT.
    /// The other version of the family, if installed, is replaced (`Library.install`).
    func acceptAndInstall(_ family: Family, variant: Variant, baseDiT: Bool = true) {
        guard managementPossible else { return }
        do { try library.acceptLicense(ModelCard.of(family)) } catch {
            problem = Problem(error)
            return
        }
        installOffer = nil
        // The plan the panel showed: when the old version must go first, a Stop leaves the family
        // without a complete version — the flash says so instead of « Stopped — … kept » alone.
        let plan = try? library.installPlan(family, variant: variant, baseDiT: baseDiT)
        let stopped = plan.flatMap { p in p.removesFirst ? p.replacing : nil }.map { old in
            String(localized: "Stopped — \(family.name) can render again after a new installation: the \(VersionChoice.name(old)) version was removed to make room. The files already downloaded are kept; the one in progress will start over.")
        }
        launchForge(String(localized: "Installing \(family.name)"), stopped: stopped) { b, cancellation, journal in
            try b.install(family, variant: variant, baseDiT: baseDiT, cancellation: cancellation, journal: journal)
        } end: { [weak self] (_: Void) in
            guard let self else { return }
            // The model just installed becomes the form's, if the form's does not render.
            if !readySet.contains(identifier), baseDiT { identifier = ModelCard.of(family).id }
            flash(String(localized: "\(family.name) installed"))
        }
    }

    /// The model chosen in the form is not installed: offer its family.
    func installSelectedModel() {
        guard let f = familyToInstall else { return }
        offerInstall(f, baseDiT: card?.isImported != true)
    }

    /// Stops the installation or import in progress; the files already downloaded are kept.
    func cancelForge() { cancellationForge?.cancel() }

    /// Uninstalls a family (DiT, components, encoder if not shared). Its imported
    /// models and its LoRAs remain.
    func uninstall(_ family: Family) {
        guard managementPossible else { return }
        do {
            try library.uninstall(family)
            refreshCatalog()
            flash(String(localized: "\(family.name) uninstalled"))
        } catch { problem = Problem(error) }
    }

    /// Removes an imported model (its map; the original file was already no longer read).
    func remove(_ f: ModelCard) {
        guard f.isImported, managementPossible else { return }
        do {
            try ModelImport.remove(f)
            refreshCatalog()
            if identifier == f.id { identifier = ModelCard.of(f.family).id }
            flash(String(localized: "Model \(f.name) removed"))
        } catch { problem = Problem(error) }
    }

    /// Removes a forged LoRA, and from the stack if it was there.
    func remove(_ lora: LoRACard) {
        guard managementPossible else { return }
        do {
            try ModelImport.remove(lora)
            loras.removeAll { $0.path == lora.path }
            refreshCatalog()
            flash(String(localized: "LoRA \(lora.name) removed"))
        } catch { problem = Problem(error) }
    }

    func emptyDownloads() {
        guard managementPossible else { return }
        do { try library.emptyDownloads(); refreshCatalog() } catch { problem = Problem(error) }
    }

    /// The library folder, in the Finder.
    func showInFinder() {
        NSWorkspace.shared.activateFileViewerSelecting([library.root])
    }

    /// A forge in the background: it reads and writes GBs, the main thread does not wait.
    /// `stopped`: what « Stop » flashes, when « the files already downloaded are kept » is not all.
    private func launchForge<T: Sendable>(_ title: String, stopped: String? = nil,
                                          _ job: @escaping @Sendable (Library, Cancellation, @escaping @Sendable (String) -> Void) throws -> T,
                                          end: @escaping @MainActor (T) -> Void) {
        forgeInProgress = title
        forgeJournal = []
        problem = nil
        let b = library
        let cancellation = Cancellation()
        cancellationForge = cancellation
        Task.detached(priority: .userInitiated) { [weak self] in
            let journal: @Sendable (String) -> Void = { rowLine in Task { @MainActor in self?.forgeJournal.append(rowLine) } }
            let result = Result { try job(b, cancellation, journal) }
            await MainActor.run {
                guard let self else { return }
                self.forgeInProgress = nil
                self.cancellationForge = nil
                self.refreshCatalog()
                switch result {
                case .success(let r): end(r)
                case .failure(let e):
                    // « Stop » cancels the token: the engine throws `EngineError.cancelled`.
                    if EngineError(e) == .cancelled {
                        self.flash(stopped ?? String(localized: "Stopped — the files already downloaded are kept; the one in progress will start over"))
                    } else {
                        self.problem = Problem(e)
                    }
                }
                // Jobs that entered the queue meanwhile waited for the forge (`next()`).
                self.next()
            }
        }
    }

    // MARK: Diagnostic

    /// **What the diagnostic measures**: the visible models installed, the publishers' (an imported
    /// DiT has no golden, and its timings are its family's).
    var diagnosticCards: [ModelCard] { visibleReady.filter { !$0.isImported } }

    /// Why the diagnostic cannot start now, or `nil`. It loads weights like a render — measured
    /// beside one, it would measure both.
    var diagnosticBlocker: String? {
        if busy { return String(localized: "A render is running: the diagnostic can start once the queue is empty.") }
        if forgeInProgress != nil { return String(localized: "An installation is running: the diagnostic can start once it is done.") }
        if diagnosticCards.isEmpty { return String(localized: "Install a model first: the diagnostic measures the models installed.") }
        return nil
    }

    func openDiagnostic() {
        managementOpen = false
        if case .finished = diagnostic {} else if !diagnosticRunning { diagnostic = .ready }
        diagnosticOpen = true
    }

    /// **Runs `Diagnostic.run`** off the main thread (~15 s per model), the queue held meanwhile.
    func startDiagnostic() {
        guard diagnosticBlocker == nil else { return }
        let cards = diagnosticCards, b = library
        let cancellation = Cancellation()
        diagnosticCancellation = cancellation
        diagnostic = .running(done: 0, total: cards.count, line: String(localized: "Preparing…"))
        Task.detached(priority: .userInitiated) { [weak self] in
            let report = Diagnostic.run(cards, in: b, cancellation: cancellation) { card, stage in
                Task { @MainActor in self?.diagnosticProgress(card, stage, cards: cards) }
            }
            await MainActor.run {
                guard let self else { return }
                self.diagnosticCancellation = nil
                self.diagnostic = cancellation.isCancelled ? .ready : .finished(report)
                self.next()
            }
        }
    }

    func cancelDiagnostic() { diagnosticCancellation?.cancel() }

    /// The library names the model and the stage (`Diagnostic.Stage`): the app says it in the user's language.
    private func diagnosticProgress(_ card: ModelCard, _ stage: Diagnostic.Stage, cards: [ModelCard]) {
        guard case .running = diagnostic, let i = cards.firstIndex(where: { $0.id == card.id }) else { return }
        let name = switch stage {
        case .textEncoder: String(localized: "text encoder")
        case .denoiser: String(localized: "two DiT evaluations")
        case .decoder: String(localized: "VAE decoder")
        }
        diagnostic = .running(done: i, total: cards.count, line: "\(card.name) · \(name)")
    }

    /// **« Open the issue »**: the GitHub form, prefilled by the report. A URL beyond ~8 000 characters is
    /// cut by browsers and GitHub: then the JSON goes to the clipboard, and the form opens with every
    /// field but the report's.
    /// Returns what to tell the user, or `nil` when the link carried everything.
    func openIssue(_ report: Diagnostic.Report) -> String? {
        do {
            let url = try report.issueURL()
            if url.absoluteString.count <= 8000 {
                NSWorkspace.shared.open(url)
                return nil
            }
            copyJSON(report)
            NSWorkspace.shared.open(try report.issueURL(includingReport: false))
            return String(localized: "The report is too long for a link: it is in the clipboard. Paste it into the form's “Report” field.")
        } catch {
            return Problem(error).text
        }
    }

    func copyJSON(_ report: Diagnostic.Report) {
        guard let data = try? report.json() else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(String(decoding: data, as: UTF8.self), forType: .string)
        flash(String(localized: "Report copied (JSON)"))
    }

    /// The window closes: nothing must keep occupying the GPU with no view to show it.
    func unmount() {
        toastTask?.cancel()
        clearQueue()
        task?.cancel()
        diagnosticCancellation?.cancel()
    }
}

// MARK: - The figures, in the user's language

/// Durations, seconds, strengths and memory, formatted according to the system language (`0,85 s` in
/// French, `0.85 s` in English). The developer's command line keeps its decimal point: it is a
/// syntax, not a display.
/// **Who holds the memory**: the three other applications with the largest footprint, named in one
/// line under a memory refusal. Their `phys_footprint` (`proc_pid_rusage`), the figure the engine's
/// own peaks are measured on — not their resident size, which counts
/// shared pages. Only the regular applications, the ones the user sees in the Dock and can close:
/// a daemon named here would be a riddle. An application's helper processes (Safari's web content,
/// a browser's tabs) are not added to it: the figure is a floor, the names are what matter.
enum MemoryHolders {
    static func line() -> String? {
        let me = ProcessInfo.processInfo.processIdentifier
        let heaviest = NSWorkspace.shared.runningApplications
            .filter { $0.activationPolicy == .regular && $0.processIdentifier != me }
            .compactMap { app -> (name: String, bytes: Int)? in
                guard let name = app.localizedName, let bytes = footprint(app.processIdentifier), bytes > 0
                else { return nil }
                return (name, bytes)
            }
            .sorted { $0.bytes > $1.bytes }
            .prefix(3)
        guard let first = heaviest.first else { return nil }
        let total = Figures.memory(heaviest.map(\.bytes).reduce(0, +))
        if heaviest.count == 1 { return String(localized: "\(first.name) is using \(total).") }
        let names = heaviest.map(\.name).formatted(.list(type: .and))
        return String(localized: "\(names) are using \(total).")
    }

    /// `ri_phys_footprint`, or `nil` when the process is gone or not readable.
    private static func footprint(_ pid: pid_t) -> Int? {
        var info = rusage_info_v4()
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(pid, RUSAGE_INFO_V4, $0) }
        }
        return status == 0 ? Int(info.ri_phys_footprint) : nil
    }
}

enum Figures {
    /// `0.85 s`, `12.3 s` — the precision of the engine's timers.
    static func seconds(_ s: Double) -> String {
        s.formatted(.number.precision(.fractionLength(s < 10 ? 2 : 1))) + " s"
    }

    /// `42 s`, `1 min 5 s`, `1 h 2 min` — a waiting time ("42 sec", "1 min, 5 sec" in English).
    static func duration(_ s: Double) -> String {
        let d = Duration.seconds(max(0, s.rounded()))
        return s < 3600
            ? d.formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated))
            : d.formatted(.units(allowed: [.hours, .minutes], width: .abbreviated))
    }

    static func strength(_ f: Double) -> String { f.formatted(.number.precision(.fractionLength(2))) }

    /// `40 %` in French, `40%` in English — `n` out of 100.
    static func percent(_ n: Int) -> String { (Double(n) / 100).formatted(.percent.precision(.fractionLength(0))) }

    static func hour(_ d: Date) -> String { d.formatted(date: .omitted, time: .shortened) }

    static func memory(_ bytes: Int) -> String { Int64(bytes).formatted(.byteCount(style: .memory)) }

    /// The space on disk, as the Finder counts it (`20.4 GB`).
    static func disk(_ bytes: Int) -> String { Int64(bytes).formatted(.byteCount(style: .file)) }
}

extension Engine.Stage {
    var label: String {
        switch self {
        case .text: String(localized: "Encoding the prompt")
        case .image: String(localized: "Encoding the image")
        case .denoising: String(localized: "Denoising")
        case .decoding: String(localized: "Decoding")
        }
    }
}

extension RecommendedFormat.Orientation {
    var label: String {
        switch self {
        case .square: String(localized: "Square")
        case .portrait: String(localized: "Portrait")
        case .landscape: String(localized: "Landscape")
        }
    }
    /// A symbol whose silhouette follows the aspect ratio.
    var symbol: String {
        switch self {
        case .square: "square"
        case .portrait: "rectangle.portrait"
        case .landscape: "rectangle"
        }
    }
}
