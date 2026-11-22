// **The app's value types**: what `AppState` keeps and the views read — the LoRA stack's slots, the settings that redo an
// image, the history's entries, the jobs, how they end and what they signal. The edit's references are in
// `ReferenceImage.swift`, the render's figures in `Measures.swift`, how a figure is written in `Figures.swift`; the state
// itself is in `AppState.swift`.

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
    /// Off: kept in the stack with its strength, out of the render — to see the image without it, and back.
    var on = true
}

/// What redoes an image: taken at launch, kept with each history image.
struct RenderSettings: Equatable {
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
    /// **The solver** (`Request.solver`): Euler unless picked with More Solvers on.
    var solver: Solver = .euler
    /// **img2img's strength** (`Request.strength`, diffusers'): set when the render starts from an image (`Job.start`), `nil` for
    /// any other. What the PNG writes and « Redo » gives back with the image.
    var strength: Double? = nil
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
    /// **The edit's own images**, in their numbers: its references without the named subjects' pictures that follow them
    /// (`NamedSubjects.editImages`), which come back with the text that cites them.
    var editImages: [ReferenceImage] { NamedSubjects.editImages(references, subjectImages: settings.subjectImages) }
    /// **The image it started from** (img2img, `RenderSettings.strength`): what « Variations » and « Redo » render again.
    var start: ReferenceImage? = nil
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
    /// only its own; what is made from an image (crop, enlargement) keeps its owner.
    var owner: Data? = nil

    var seconds: Double { timings.total }
    var name: String { "i\(ordinal)" }

    func image() -> CGImage? { Self.decode(png) }

    /// `image()` off the main thread — what the canvas, the comparison and the large cells wait for:
    /// a 1024² PNG takes tens of milliseconds to decode, enough to stutter the strip under ← →.
    /// Decoded there for good (`kCGImageSourceShouldCacheImmediately`): an image ImageIO decodes lazily would be decoded at
    /// its first drawing, on the main thread.
    static func decoded(_ png: Data) async -> Decoded {
        await Task.detached(priority: .userInitiated) { Decoded(decode(png, now: true)) }.value
    }

    static func decode(_ png: Data, now: Bool = false) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(png as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, now ? [kCGImageSourceShouldCacheImmediately: true] as CFDictionary : nil)
    }

    var legend: String {
        if cropped != nil {
            return String(localized: "\(settings.modelName) · \(Figures.size(width, height)) · seed \(String(seed)) · cropped")
        }
        let base = settings.isVariation
            ? String(localized: "\(settings.modelName) · \(Figures.size(width, height)) · seed \(String(seed)) · variation · \(Figures.duration(seconds))")
            : String(localized: "\(settings.modelName) · \(Figures.size(width, height)) · seed \(String(seed)) · \(Figures.duration(seconds))")
        // Only when it changes the image: Normal and Euler are every render's, they say nothing.
        return base + (settings.strength.map { " · " + StartWords.legend($0) } ?? "")
            + (settings.detail == .normal ? "" : " · " + settings.detail.label)
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
    var settings: RenderSettings
    let seeds: [UInt64]
    /// The edit's references, in order — image 1 is the one edited: re-read, not re-noised.
    let references: [ReferenceImage]
    /// **The image it starts from** (img2img): re-noised to `settings.strength`, then denoised. `nil`: from noise.
    var start: ReferenceImage? = nil
    let previews: Bool
    /// The model's default number of steps: `nil` in the request if it was not changed.
    let defaultSteps: Int?
    /// The denoiser's plan for one image (its evaluations) — `nil` if the model is not installed. What, with the learned speed, estimates
    /// its duration before it runs.
    let plan: DenoisingPlan?
    /// Who asked for it (`Entry.owner`).
    var owner: Data? = nil

    /// The key of the learned speed: same model, same format, same number of references (each one
    /// lengthens the sequence and every evaluation with it), with or without LoRA (the LoRA weighs on
    /// every evaluation).
    var speedKey: String {
        Speed.key(settings.identifier, settings.format, references: references.count + (start == nil ? 0 : 1), lora: !settings.loras.isEmpty)
    }

    var name: String { "t\(ordinal)" }

    /// **The same render under another seed**: one image each, everything else the engine reads equal. What lets the
    /// queue render two cells of a series in one pass (`AppState.train`).
    func rendersAs(_ other: Job) -> Bool {
        seeds.count == 1 && other.seeds.count == 1 && settings == other.settings
            && references.map(\.id) == other.references.map(\.id) && start?.id == other.start?.id
            && previews == other.previews && defaultSteps == other.defaultSteps
    }

    var legend: String {
        let n = seeds.count > 1 ? " ×\(seeds.count)" : ""
        return "\(settings.modelName) · \(Figures.size(settings.format.width, settings.format.height))\(n)"
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

/// **What the remote control listens to**: the lifecycle of each job, rendered by it or not.
enum Signal {
    case started(Job)
    case stage(Job, Engine.Stage, image: Int)
    case step(Job, image: Int, index: Int, total: Int, seconds: Double)
    case image(Job, Entry)
    case finished(Job, Issue)
    /// A series' last cell ended.
    case gridEnded(GridRun)
}
