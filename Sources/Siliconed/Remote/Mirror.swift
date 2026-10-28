import Foundation

/// **What a lending Mac publishes, and what a borrowing one asks**, once their session is joined
/// (`SessionMessage`). Plain values — strings, numbers, the engine's `Codable` enums — so that both ends
/// decode exactly what the other encoded; the app projects its `Job`, `Entry` and state into them and
/// back.
///
/// **Each Mac sees only its own**: the queue, the history, the preview and the exploration are published
/// per client — what the mini renders for itself or for another Mac is only a count (`others`).
///
///     mini → Neo   catalog     the ready models, what each accepts, its LoRAs, B — whenever it changes
///                  queue       the job running and its progress, the jobs waiting — whenever it changes
///                  entry + PNG one history image, its PNG as the blob that follows; `removed` the reverse
///                  preview     the denoising's latent preview as a small PNG blob, once per step
///                  quoted      the answer to a `quote`; `answer` the one to a `render`
///                  ended       one of the Neo's jobs ended, after its images (what `silicontrol wait` reads)
///                  explored    the exploration under way, without its images; `cellImage` + PNG one cell's
///     Neo → mini   render      one job from the Neo's form; the mini checks it again (B, steps) before
///                              queueing it — the Neo's judgment is never the last word; an edit's images
///                              go first as blobs, or by their history ordinal
///                  quote       what an edit's format and a need with images are: the denoiser's rule
///                              and the memory need come from maps only the mini holds
///                  grid        an exploration (or a grid) from the Neo's form: its cells made on the mini,
///                              which alone knows where a model's sketch stops; `addSheet` draws its sheet
///                  enlarge, crop  an image of the mini's history, made there; `enlargeNeedsDownload`
///                              asks the Neo's consent to Apple's model
///                  cancel, removeJob, stopAll, clearQueue, deleteImages
///
/// **No `Date` crosses**: the two clocks differ. What remains of a render is a duration.

package struct RemoteLoRA: Codable, Equatable, Sendable {
    package var path: String
    package var strength: Double
    package init(path: String, strength: Double) { self.path = path; self.strength = strength }
}

/// **An edit's image**, in a `render`: the mini decodes it to the pixels the Neo would have rendered.
package struct RemoteReference: Codable, Equatable, Sendable {
    package enum Origin: Codable, Equatable, Sendable {
        /// An image of the mini's history, by its `i` ordinal: no byte crosses.
        case history(Int)
        /// A file's own bytes, the blob sent before the `render`, decoded as the file (`ImageRGB(data:)`).
        case file(blob: UInt32)
        /// An image held in memory on the Neo (pasted, a subject's picture), as a lossless PNG blob.
        case image(blob: UInt32)
        /// The `index`-th edit image a history image was rendered with (« Variations » of an edit): it
        /// stays on the mini, the Neo only names it.
        case ofImage(Int, index: Int)
    }
    package var origin: Origin
    package var name: String
    package var width: Int, height: Int
    package init(origin: Origin, name: String, width: Int, height: Int) {
        self.origin = origin; self.name = name; self.width = width; self.height = height
    }

    package var blob: UInt32? {
        switch origin {
        case .history, .ofImage: nil
        case .file(let b), .image(let b): b
        }
    }
}

/// **A question the Neo's form asks** — what it cannot compute without the maps' headers.
package struct RemoteQuote: Codable, Hashable, Sendable {
    package var model: String
    /// The format chosen; with `first`, ignored: an edit's format follows image 1.
    package var width: Int, height: Int
    /// Image 1's upright size, when editing.
    package var first: [Int]?
    /// The images the render reads (the edit's and the named subjects').
    package var images: Int
    package init(model: String, width: Int, height: Int, first: [Int]?, images: Int) {
        self.model = model; self.width = width; self.height = height; self.first = first; self.images = images
    }
}

/// The mini's answer: the format the render will have, and its memory need there (`nil`: not computable).
package struct RemoteQuoteAnswer: Codable, Equatable, Sendable {
    package var quote: RemoteQuote
    package var width: Int, height: Int
    package var need: Int?
    package init(quote: RemoteQuote, width: Int, height: Int, need: Int?) {
        self.quote = quote; self.width = width; self.height = height; self.need = need
    }
}

/// A job, as the queue shows it and as the Neo asks for it.
package struct RemoteJob: Codable, Equatable, Sendable {
    /// The mini's `t` ordinal; in a `render` request, the Neo's request number, which the mini's
    /// `answer` carries back (the mini numbers the job).
    package var ordinal: Int
    package var model: String
    package var modelName: String
    package var prompt: String
    /// 0 × 0 in an edit's `render`: the mini derives the format from image 1 (`silicontrol add --ref`).
    package var width: Int, height: Int
    package var steps: Int
    package var defaultSteps: Int?
    package var loras: [RemoteLoRA]
    package var seeds: [UInt64]
    package var detail: Detail
    package var solver: Solver
    package var schedule: SigmaSchedule
    package var previews: Bool
    /// The Neural Engine as the rack chose it; `nil`: the mini's choice for the model.
    package var neuralEngine: Bool?
    /// An edit's images, image 1 first.
    package var references: [RemoteReference]
    /// « Variations »: applied in order over the seed's noise.
    package var variations: [Variation]

    package init(ordinal: Int, model: String, modelName: String, prompt: String, width: Int, height: Int, steps: Int,
                 defaultSteps: Int?, loras: [RemoteLoRA], seeds: [UInt64], detail: Detail, solver: Solver,
                 schedule: SigmaSchedule, previews: Bool, neuralEngine: Bool?, references: [RemoteReference] = [],
                 variations: [Variation] = []) {
        self.references = references; self.variations = variations
        self.ordinal = ordinal; self.model = model; self.modelName = modelName; self.prompt = prompt
        self.width = width; self.height = height; self.steps = steps; self.defaultSteps = defaultSteps
        self.loras = loras; self.seeds = seeds; self.detail = detail; self.solver = solver
        self.schedule = schedule; self.previews = previews; self.neuralEngine = neuralEngine
    }
}

/// The queue and the render under way.
package struct RemoteQueue: Codable, Equatable, Sendable {
    package var current: RemoteJob?
    package var waiting: [RemoteJob] = []
    /// Jobs held until their model's license is accepted on the mini (lot 4 brings the sheet here).
    package var awaitingLicense: [RemoteJob] = []
    package var fraction = 0.0
    /// `Engine.Stage.rawValue`.
    package var stage: String?
    package var currentStep = 0, totalSteps = 0
    package var batchImage = 0, batchSize = 1
    package var stageStates: [String: String] = [:]
    package var stageSeconds: [String: Double] = [:]
    /// Seconds left in the render running, as the mini estimated them when it sent this.
    package var remaining: Double?
    /// The last refusal or failure, in the mini's words.
    package var problem: String?
    /// The history image being enlarged (`i` ordinal).
    package var enlarging: Int?
    /// **What others asked of the mini**, never shown: its own renders and other Macs', as a count, and
    /// the time they take before it is free (`nil`: one of them never measured).
    package var others = 0
    package var othersSeconds: Double?
    package init() {}
}

/// One history image; its PNG is the blob `png` that follows the message.
package struct RemoteEntry: Codable, Equatable, Sendable {
    package var ordinal: Int
    package var job: RemoteJob
    package var width: Int, height: Int
    package var seed: UInt64
    package var timings: Engine.Timings
    package var footprints: Engine.Footprints
    package var evaluations: Int
    package var metadata: [String: String]
    package var neuralEngine: Bool
    package var png: UInt32
    /// An enlargement's factor; a crop's box in the image it was cut from.
    package var enlarged: Int?
    package var cropped: Crop.Box?
    package var sheet: RemoteSheet?

    package init(ordinal: Int, job: RemoteJob, width: Int, height: Int, seed: UInt64, timings: Engine.Timings,
                 footprints: Engine.Footprints, evaluations: Int, metadata: [String: String], neuralEngine: Bool, png: UInt32,
                 enlarged: Int? = nil, cropped: Crop.Box? = nil, sheet: RemoteSheet? = nil) {
        self.enlarged = enlarged; self.cropped = cropped; self.sheet = sheet
        self.ordinal = ordinal; self.job = job; self.width = width; self.height = height; self.seed = seed
        self.timings = timings; self.footprints = footprints; self.evaluations = evaluations
        self.metadata = metadata; self.neuralEngine = neuralEngine; self.png = png
    }
}

/// What a ready model accepts — what the Neo's form reads from a chain it cannot build (`Model.named`
/// reads the maps' headers on the mini's disk).
package struct RemoteTraits: Codable, Equatable, Sendable {
    package var maxReferences: Int
    package var readsImages: Bool
    package var acceptsDetail: Bool
    package var comfyUI: Bool
    /// The step counts the denoiser has a schedule for, among 1…50.
    package var steps: [Int]
    /// The rack's module names.
    package var text: String?, encoding: String?, decoding: String?
    package init(maxReferences: Int, readsImages: Bool, acceptsDetail: Bool, comfyUI: Bool, steps: [Int],
                 text: String?, encoding: String?, decoding: String?) {
        self.maxReferences = maxReferences; self.readsImages = readsImages; self.acceptsDetail = acceptsDetail
        self.comfyUI = comfyUI; self.steps = steps; self.text = text; self.encoding = encoding; self.decoding = decoding
    }
}

package struct MirrorCatalog: Codable, Equatable, Sendable {
    /// The ready models' ids (the publisher's cards: both apps are one build).
    package var ready: [String]
    package var traits: [String: RemoteTraits]
    /// The LoRAs forged for each model: path on the mini, name.
    package var loras: [String: [RemoteLoRA]]
    package var loraNames: [String: String]
    /// The memory each ready model needs at each of its formats, without references (`key`), in bytes —
    /// computed from maps the borrowing Mac does not hold (`ModelCard.memoryNeed`).
    package var needs: [String: [String: Int]]
    /// The mini's B, in bytes: the Neo judges memory against it, the mini checks again.
    package var ceiling: Int
    package var settingsRead: Bool
    package var neuralEngineOffered: Bool
    package var fasterRendering: Bool
    /// macOS's super-resolution is offered there (`SuperResolution.available`).
    package var enlarges: Bool
    package init(ready: [String], traits: [String: RemoteTraits], loras: [String: [RemoteLoRA]], loraNames: [String: String],
                 needs: [String: [String: Int]] = [:], ceiling: Int, settingsRead: Bool, neuralEngineOffered: Bool, fasterRendering: Bool,
                 enlarges: Bool = false) {
        self.enlarges = enlarges
        self.ready = ready; self.traits = traits; self.loras = loras; self.loraNames = loraNames; self.needs = needs
        self.ceiling = ceiling
        self.settingsRead = settingsRead; self.neuralEngineOffered = neuralEngineOffered; self.fasterRendering = fasterRendering
    }

    package static func key(_ width: Int, _ height: Int) -> String { "\(width)x\(height)" }
}

/// **An XY grid as it crosses**: its base and axes, rebuilt — and so checked again — by
/// `RenderGrid.init` on the other side.
package struct RemoteGridSpec: Codable, Equatable, Sendable {
    package var base: RenderGrid.Base
    package var x: RenderGrid.Axis
    package var y: RenderGrid.Axis?
    package init(_ g: RenderGrid) { base = g.base; x = g.x; y = g.y }
    package func grid() throws(EngineError) -> RenderGrid { try RenderGrid(base, x: x, y: y) }
}

/// An exploration asked from the Neo's form. `job`: what the cells share — the prompt as typed, the
/// format, the stack, the edit's images —; its seeds are none, the grid's are.
package struct RemoteGridRequest: Codable, Equatable, Sendable {
    package var spec: RemoteGridSpec
    package var job: RemoteJob
    package var sketch: Bool
    /// The evaluations a sketch stops after; 0: where the model's sketch stops by itself.
    package var stopAfter: Int
    package init(spec: RemoteGridSpec, job: RemoteJob, sketch: Bool, stopAfter: Int) {
        self.spec = spec; self.job = job; self.sketch = sketch; self.stopAfter = stopAfter
    }
}

/// How a cell's job ended, as the panel says it.
package enum RemoteIssue: Codable, Equatable, Sendable {
    case finished, stopped, removed
    case failed(String)
}

/// **The exploration on the mini**, published whole at each change — but its images, which go once
/// each (`cellImage`).
package struct RemoteExploration: Codable, Equatable, Sendable {
    package var ordinal: Int
    package var spec: RemoteGridSpec
    package var job: RemoteJob
    /// Each cell's job, in the grid's order: its ordinal, and its settings whole.
    package var cells: [RemoteJob]
    package var issues: [Int: RemoteIssue]
    package var sketch: Bool
    package init(ordinal: Int, spec: RemoteGridSpec, job: RemoteJob, cells: [RemoteJob], issues: [Int: RemoteIssue], sketch: Bool) {
        self.ordinal = ordinal; self.spec = spec; self.job = job; self.cells = cells; self.issues = issues; self.sketch = sketch
    }
}

/// A sheet in the history: which grid it drew, and how many cells stayed empty.
package struct RemoteSheet: Codable, Equatable, Sendable {
    package var ordinal: Int
    package var spec: RemoteGridSpec
    package var empty: Int
    package init(ordinal: Int, spec: RemoteGridSpec, empty: Int) { self.ordinal = ordinal; self.spec = spec; self.empty = empty }
}

/// **The mini's word on a `render`**, carried back with the Neo's request number.
package enum RemoteAnswer: Codable, Equatable, Sendable {
    /// Queued as this job, with what the mini estimates it takes (`nil`: never measured).
    case queued(ordinal: Int, estimate: Double?, width: Int, height: Int)
    case refused(String)
}

/// **A job of the Neo's that ended** on the mini — sent after its images, so that whoever follows it
/// has them all when the verdict comes.
package struct RemoteEnded: Codable, Equatable, Sendable {
    package var ordinal: Int
    package var issue: RemoteIssue
    package var images: [Int]
    package var warnings: [String]
    package init(ordinal: Int, issue: RemoteIssue, images: [Int], warnings: [String]) {
        self.ordinal = ordinal; self.issue = issue; self.images = images; self.warnings = warnings
    }
}
