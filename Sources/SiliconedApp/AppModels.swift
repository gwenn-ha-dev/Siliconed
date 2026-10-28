// **The app's value types**: what `AppState` keeps and the views read — the LoRA stack's slots,
// the edit's references, the history's entries, the jobs, the render's figures —, then how a figure
// is written in the user's language (`Figures`). The state itself is in `AppState.swift`.

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
    /// `subject`: a named subject's picture, cropped to its frame (`Subjects.swift`) — `nil` if its
    /// bytes did not decode, which fails the render naming it. `bytes`: a file's bytes sent by another
    /// Mac (`MirrorServer`), decoded as the file was there. `remote`: an image that stays on the other
    /// Mac, named only (a mirrored history image's edit images) — nothing to show or decode here.
    enum Source { case file(URL), image(CGImage), subject(CGImage?, name: String), bytes(Data), remote(RemoteReference.Origin) }
    let id = UUID()
    let source: Source
    let vignette: NSImage
    let name: String
    let width: Int, height: Int
    /// Taken from the history: its `i` ordinal — what a mirror sends instead of the bytes.
    var ordinal: Int? = nil

    /// Decoded for display, upright, at `width × height` — the original under an edit's curtain.
    func displayed(width: Int, height: Int) -> CGImage? {
        let image: CGImage?
        switch source {
        case .image(let cg): image = cg
        case .subject(let cg, _): image = cg
        case .remote: image = nil
        case .file, .bytes:
            let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                            kCGImageSourceThumbnailMaxPixelSize: max(self.width, self.height),
                                            kCGImageSourceCreateThumbnailWithTransform: true]
            image = imageSource.flatMap { CGImageSourceCreateThumbnailAtIndex($0, 0, options as CFDictionary) }
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
        case .bytes(let data): return try ImageRGB(data: data, name: name)
        case .remote: throw EngineError.imageUnreadable(file: name)
        case .image(let cg): return try ImageRGB(cgImage: cg)
        case .subject(let cg, let name):
            guard let cg else { throw EngineError.imageUnreadable(file: name) }
            return try ImageRGB(cgImage: cg)
        }
    }

    private var imageSource: CGImageSource? {
        switch source {
        case .file(let url): CGImageSourceCreateWithURL(url as CFURL, nil)
        case .bytes(let data): CGImageSourceCreateWithData(data as CFData, nil)
        case .image, .subject, .remote: nil
        }
    }

    /// A file's size as it will be decoded, upright — or `nil` if ImageIO cannot read its header.
    static func size(of url: URL) -> (width: Int, height: Int)? {
        CGImageSourceCreateWithURL(url as CFURL, nil).flatMap(size)
    }

    static func size(of data: Data) -> (width: Int, height: Int)? {
        CGImageSourceCreateWithData(data as CFData, nil).flatMap(size)
    }

    private static func size(_ source: CGImageSource) -> (width: Int, height: Int)? {
        guard let p = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
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

/// **What the Images box accepts**: a file (the Finder), or PNG bytes — a strip thumbnail, which leaves
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
    /// How many references the render read — the edit's and the named subjects' (0: none).
    var references = 0
    /// **The prompt the model read**, when it is not `prompt`: the named subjects' rewrite
    /// (`NamedSubjects.rewrite`). `prompt` stays the one typed — what « Reuse These Settings » gives back.
    var modelPrompt: String? = nil
    /// Of `references`, those that are named subjects' pictures, after the edit's own.
    var subjectImages = 0
    /// **The variations its noise was turned by** (`Request.variations`), in order: empty for an
    /// ordinary image. The seed stays the origin's.
    var variations: [Variation] = []
    /// An edit: images of its own, not only subjects.
    var editing: Bool { references - subjectImages > 0 }
    /// **More detail** (`Request.detail`): the rack's Normal · More · Most.
    var detail: Detail = .normal
    /// **The solver** (`Request.solver`): Euler unless picked in Developer Mode.
    var solver: Solver = .euler
    /// **Its denoiser could take the Neural Engine** (`Engine.Render.neuralEngine`): set from the render, for
    /// the image's panel — the choice itself is read when the job starts (`Job.neuralEngine`).
    var neuralEngine = false
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
    /// **An enlargement** (`Enlarge.swift`): its factor. Its settings stay the render's — what « Reuse
    /// These Settings » and « Redo » bring back is the image it was made from.
    var enlarged: Int? = nil
    /// The image it was enlarged from, as its PNG — what the curtain lays under it, even once that
    /// image has left the history.
    var enlargedFrom: Data? = nil
    /// **A crop** (`CropEditor.swift`): the pixels of the image it was cut from. Its settings stay the render's,
    /// like an enlargement's; it was not rendered, so it has no time.
    var cropped: Crop.Box? = nil
    /// **Who asked for it**: another Mac's identity (`MirrorServer`), `nil` for this Mac. A client sees
    /// only its own; what is made from an image (crop, enlargement, sheet) keeps its owner.
    var owner: Data? = nil

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
        if cropped != nil {
            return String(localized: "\(settings.modelName) · \(String(width))×\(String(height)) · seed \(String(seed)) · cropped")
        }
        if let g = grid?.grid {
            let f = settings.format
            return String(localized: "Grid \(String(g.columns))×\(String(g.rows)) · \(settings.modelName) · \(String(f.width))×\(String(f.height)) · \(Figures.duration(seconds))")
        }
        let base = settings.isVariation
            ? String(localized: "\(settings.modelName) · \(String(width))×\(String(height)) · seed \(String(seed)) · variation · \(Figures.duration(seconds))")
            : String(localized: "\(settings.modelName) · \(String(width))×\(String(height)) · seed \(String(seed)) · \(Figures.duration(seconds))")
        // Only when it changes the image: Normal and Euler are every render's, they say nothing.
        return base + (settings.detail == .normal ? "" : " · " + settings.detail.label)
            + (settings.solver == .euler ? "" : " · " + settings.solver.label)
            + (enlarged.map { " · " + String(localized: "enlarged ×\(String($0))") } ?? "")
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

extension Solver {
    /// The rack's menu item: the names samplers go by everywhere, not translated.
    var choice: String {
        switch self {
        case .euler: "Euler"
        case .eulerAncestral: "Euler a"
        case .dpmpp2m: "DPM++ 2M"
        case .unipc: "UniPC"
        }
    }
    /// The history caption's « Solver: UniPC ».
    var label: String { String(localized: "Solver: \(choice)") }
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

    /// `metadata`: the render's own unless named subjects rewrote its prompt (`NamedSubjects.metadata`).
    init(_ renderResult: Engine.Render, metadata: [String: String]? = nil) throws {
        png = try metadata.map { try renderResult.image.png(metadata: $0) } ?? renderResult.png()
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
    /// **The Neural Engine as the remote control asked it** (`add --ane`, `--no-ane`), frozen in the job;
    /// `nil`, the model's choice when the job starts (`AppState.neuralEngine(for:)`), like a click.
    var neuralEngine: Bool? = nil
    /// Who asked for it (`Entry.owner`).
    var owner: Data? = nil

    /// The key of the learned speed: same model, same format, same number of references (each one
    /// lengthens the sequence and every evaluation with it), with or without LoRA (the spectral turns off under a LoRA, and the LoRA weighs on
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
    /// `nil` while the text encoder has only been read back from the cache: ~0 s for that prompt,
    /// nothing learned for the next one.
    var text: Double?
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
        return (text ?? 0) + image + Double(images) * perImage
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

/// **The diagnostic** (`Diagnostic.run`, ~1 min, no weights): what it measures, while it runs, then its report.
enum DiagnosticPhase {
    /// Not started: the sheet explains what it does, and how long.
    case ready
    /// Running: the stages done so far out of `total`, and the one under way.
    case running(done: Int, total: Int, line: String)
    case finished(Diagnostic.Report)
    /// Refused or failed before its report (another process rendering: `renderAlreadyRunning`).
    case failed(Problem)
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
    /// The conditioning came from `RenderCache`: its time is not the encoder's cost.
    private(set) var textCached = false
    /// Per image of the batch, the duration of each DiT evaluation.
    private(set) var stepSeconds: [[Double]] = []
    private(set) var decodings: [Double] = []
    /// The timers and footprints of the last finished image.
    private(set) var timings: Engine.Timings?
    private(set) var footprints: Engine.Footprints?
    /// A mirror has no `plan`: the other Mac's evaluations and half-size steps (`RemoteMeasures`).
    private(set) var remotePlan: (evaluations: Int, reduced: Int)?

    init() {}

    /// What the other Mac measured for this Mac's render, dated here: `begin` is `elapsed` ago.
    init(remote r: RemoteMeasures, job: Job, now: Date = Date()) {
        self.job = job
        begin = now.addingTimeInterval(-r.elapsed)
        if let e = r.evaluations { remotePlan = (e, r.reduced ?? 0) }
        tokens = r.tokens; tokenizer = r.tokenizer; encoder = r.encoder; encoding = r.encoding
        textCached = r.textCached; stepSeconds = r.stepSeconds; decodings = r.decodings
        timings = r.timings; footprints = r.footprints
    }

    /// The evaluations planned per image, here or on the Mac that renders.
    var planned: Int { plan?.denoising.evaluations ?? remotePlan?.evaluations ?? 0 }

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
        case let .text(j, t, e, cached):
            tokens = j; tokenizer = t; encoder = e; textCached = cached
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

    /// The half-size steps of the spectral schedule: the first ones of each image.
    var reduced: Int { plan?.denoising.reduced ?? remotePlan?.reduced ?? 0 }

    /// What this render learns about speed; `nil` without a full step.
    var speed: Speed? {
        let k = reduced
        let fullCount = stepSeconds.flatMap { $0.dropFirst(k) }, reduced = stepSeconds.flatMap { $0.prefix(k) }
        func mean(_ v: [Double]) -> Double? { v.isEmpty ? nil : v.reduce(0, +) / Double(v.count) }
        guard let full = mean(fullCount) else { return nil }
        return Speed(full: full, reducedAverage: mean(reduced), text: textCached ? nil : (tokenizer ?? 0) + (encoder ?? 0),
                       image: encoding ?? 0, decoding: mean(decodings) ?? Engine.Progress.decodingWeights * full)
    }
}

// MARK: - The figures, in the user's language

/// Durations, seconds, strengths and memory, formatted according to the system language (`0,85 s` in
/// French, `0.85 s` in English). The developer's command line keeps its decimal point: it is a
/// syntax, not a display.
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
