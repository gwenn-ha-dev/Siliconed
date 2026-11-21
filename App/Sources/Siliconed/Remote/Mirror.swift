import Foundation

/// **What a lending Mac publishes, and what a borrowing one asks**, once their session is joined
/// (`SessionMessage`). Plain values — strings, numbers, the engine's `Codable` enums — so that both ends
/// decode exactly what the other encoded; the app projects its `Job`, `Entry` and state into them and
/// back.
///
/// **Each Mac sees only its own**: the queue, the history, the preview and the series are published
/// per client — what the lender renders for itself or for another Mac is only a count (`others`).
///
///     lender → borrower
///                  catalog     the ready models, what each accepts, its LoRAs, B — whenever it changes
///                  queue       the job running and its progress, the jobs waiting — whenever it changes
///                  entry + PNG one history image, its PNG as the blob that follows; `removed` the reverse
///                  preview     the denoising's latent preview as a small PNG blob, once per step
///                  quoted      the answer to a `quote`; `answer` the one to a `render`
///                  ended       one of the borrower's jobs ended, after its images (what `silicontrol wait` reads)
///                  series      each of the borrower's series when it changes, without its images: its cells are
///                              history images, laid out by their job
///                  synced      at each join: the images and jobs the lender still knows of the borrower's, so that
///                              what was removed during a cut is removed here too
///                  statistics  what the Statistics window reads, for the borrower's renders only (`Statistics.swift`)
///                  licensing   the license a waiting job of the borrower's needs, accepted on the borrower's screen
///     borrower → lender
///                  render      one job from the borrower's form; the lender checks it again (B, steps) before
///                              queueing it — the borrower's judgment is never the last word; an edit's images
///                              go first as blobs, or by their history ordinal
///                  quote       what an edit's format and a need with images are: the denoiser's rule
///                              and the memory need come from maps only the lender holds
///                  grid        a series (or a grid) from the borrower's form: its cells made on the lender, each
///                              judged there as a render is; asked with a request number (`silicontrol grid`), it
///                              is answered by `gridQueued`. Its sheet is the borrower's to draw, from its images
///                  enlarge, crop  an image of the lender's history, made there; `enlargeNeedsDownload`
///                              asks the borrower's consent to Apple's model, `enlargeRefused` says why not
///                  cancel, removeJob, stopAll, clearQueue, deleteImages
///                  restoreImages  the images just deleted, asked back from the lender's session trash
///                  answerLicense  the click on that sheet, with the digest of the text shown (`Licensing.swift`)
///
/// **No `Date` crosses**: the two clocks differ. What remains of a render is a duration. **No sentence
/// crosses** where a code can: a refusal is `RemoteRefusal` (`Refusal.swift`), said in the reader's language.

package struct RemoteLoRA: Codable, Equatable, Sendable {
    package var path: String
    package var strength: Double
    package init(path: String, strength: Double) { self.path = path; self.strength = strength }
}

/// **An edit's image**, in a `render`: the lender decodes it to the pixels the borrower would have rendered.
package struct RemoteReference: Codable, Equatable, Sendable {
    package enum Origin: Codable, Equatable, Sendable {
        /// An image of the lender's history, by its `i` ordinal: no byte crosses.
        case history(Int)
        /// A file's own bytes, the blob sent before the `render`, decoded as the file (`ImageRGB(data:)`).
        case file(blob: UInt32)
        /// An image held in memory on the borrower (pasted, a subject's picture), as a lossless PNG blob.
        case image(blob: UInt32)
        /// The `index`-th edit image a history image was rendered with (« Variations » of an edit): it
        /// stays on the lender, the borrower only names it.
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

/// **A question the borrower's form asks** — what it cannot compute without the maps' headers.
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

extension RemoteQuote {
    /// **Whether a lender may compute this**: the numbers come from another Mac and reach products and sums
    /// that trap on overflow (`editFormat`). A side up to what any model draws, an image up to
    /// what a camera makes, a handful of images.
    package var isSane: Bool {
        let sides = 1...Format.maximumSide
        guard sides.contains(width), sides.contains(height), (0...32).contains(images) else { return false }
        guard let first else { return true }
        return first.count == 2 && first.allSatisfy { (1...(1 << 16)).contains($0) }
    }
}

/// The lender's answer: the format the render will have, and its memory need there (`nil`: not computable).
package struct RemoteQuoteAnswer: Codable, Equatable, Sendable {
    package var quote: RemoteQuote
    package var width: Int, height: Int
    package var need: Int?
    package init(quote: RemoteQuote, width: Int, height: Int, need: Int?) {
        self.quote = quote; self.width = width; self.height = height; self.need = need
    }
}

/// A job, as the queue shows it and as the borrower asks for it.
package struct RemoteJob: Codable, Equatable, Sendable {
    /// The lender's `t` ordinal; in a `render` request, the borrower's request number, which the lender's
    /// `answer` carries back (the lender numbers the job).
    package var ordinal: Int
    package var model: String
    package var modelName: String
    package var prompt: String
    /// 0 × 0 in an edit's `render`: the lender derives the format from image 1 (`silicontrol add --ref`).
    package var width: Int, height: Int
    package var steps: Int
    package var defaultSteps: Int?
    package var loras: [RemoteLoRA]
    package var seeds: [UInt64]
    package var detail: Detail
    package var solver: Solver
    package var previews: Bool
    /// An edit's images, image 1 first.
    package var references: [RemoteReference]
    /// « Variations »: applied in order over the seed's noise.
    package var variations: [Variation]
    /// **Named subjects**, which stay on the borrower (incognito): `prompt` is the one typed, `modelPrompt` the
    /// rewrite the model reads, and the last `subjectImages` of `references` are the subjects' framed
    /// pictures, sent as `.image` blobs after the edit's own. `nil`: the job cites nobody.
    package var modelPrompt: String?
    package var subjectImages: Int?
    /// **The precision of the Mac that asks**: whoever receives the image approves its margin, not the Mac that
    /// computes it. Set as the job is sent; `nil` in what the lender says of its own jobs.
    package var precision: Precision?

    package init(ordinal: Int, model: String, modelName: String, prompt: String, width: Int, height: Int, steps: Int,
                 defaultSteps: Int?, loras: [RemoteLoRA], seeds: [UInt64], detail: Detail, solver: Solver,
                 previews: Bool, references: [RemoteReference] = [],
                 variations: [Variation] = [], modelPrompt: String? = nil, subjectImages: Int? = nil) {
        self.references = references; self.variations = variations
        self.modelPrompt = modelPrompt; self.subjectImages = subjectImages
        self.ordinal = ordinal; self.model = model; self.modelName = modelName; self.prompt = prompt
        self.width = width; self.height = height; self.steps = steps; self.defaultSteps = defaultSteps
        self.loras = loras; self.seeds = seeds; self.detail = detail; self.solver = solver
        self.previews = previews
    }
}

/// The queue and the render under way.
package struct RemoteQueue: Codable, Equatable, Sendable {
    package var current: RemoteJob?
    package var waiting: [RemoteJob] = []
    /// Jobs held until their model's license is accepted on the lender.
    package var awaitingLicense: [RemoteJob] = []
    package var fraction = 0.0
    /// `Engine.Stage.rawValue`.
    package var stage: String?
    package var currentStep = 0, totalSteps = 0
    package var batchImage = 0, batchSize = 1
    package var stageStates: [String: String] = [:]
    package var stageSeconds: [String: Double] = [:]
    /// Seconds left in the render running, as the lender estimated them when it sent this.
    package var remaining: Double?
    /// The last refusal or failure.
    package var problem: RemoteRefusal?
    /// The history image being enlarged (`i` ordinal).
    package var enlarging: Int?
    /// **What others asked of the lender**, never shown: its own renders and other Macs', as a count, and
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
    package var png: UInt32
    /// An enlargement's factor; a crop's box in the image it was cut from.
    package var enlarged: Int?
    package var cropped: Crop.Box?
    /// An enlargement's source, by its history ordinal (`nil`: it left the history): the borrowing Mac lays
    /// it under the enlarged image, as the curtain does here — it has no original of its own.
    package var enlargedFrom: Int?

    package init(ordinal: Int, job: RemoteJob, width: Int, height: Int, seed: UInt64, timings: Engine.Timings,
                 footprints: Engine.Footprints, evaluations: Int, metadata: [String: String], png: UInt32,
                 enlarged: Int? = nil, cropped: Crop.Box? = nil, enlargedFrom: Int? = nil) {
        self.enlarged = enlarged; self.cropped = cropped; self.enlargedFrom = enlargedFrom
        self.ordinal = ordinal; self.job = job; self.width = width; self.height = height; self.seed = seed
        self.timings = timings; self.footprints = footprints; self.evaluations = evaluations
        self.metadata = metadata; self.png = png
    }
}

/// What a ready model accepts — what the borrower's form reads from a chain it cannot build (`Model.named`
/// reads the maps' headers on the lender's disk).
package struct RemoteTraits: Codable, Equatable, Sendable {
    package var maxReferences: Int
    package var readsImages: Bool
    package var acceptsDetail: Bool
    /// The step counts the denoiser has a schedule for, among 1…50.
    package var steps: [Int]
    /// The rack's module names.
    package var text: String?, encoding: String?, decoding: String?
    /// How this model's weights are stored on the lending Mac (`WeightFormats`): the rack's labels.
    package var weights: WeightFormats
    package init(maxReferences: Int, readsImages: Bool, acceptsDetail: Bool, steps: [Int],
                 text: String?, encoding: String?, decoding: String?, weights: WeightFormats) {
        self.weights = weights
        self.maxReferences = maxReferences; self.readsImages = readsImages; self.acceptsDetail = acceptsDetail
        self.steps = steps; self.text = text; self.encoding = encoding; self.decoding = decoding
    }
}

package struct MirrorCatalog: Codable, Equatable, Sendable {
    /// The ready models' ids (the publisher's cards: both apps are one build).
    package var ready: [String]
    package var traits: [String: RemoteTraits]
    /// The LoRAs installed for each model: path on the lender, name.
    package var loras: [String: [RemoteLoRA]]
    package var loraNames: [String: String]
    /// The lender's B, in bytes.
    package var ceiling: Int
    /// macOS's super-resolution is offered there (`SuperResolution.available`).
    package var enlarges: Bool
    package init(ready: [String], traits: [String: RemoteTraits], loras: [String: [RemoteLoRA]], loraNames: [String: String],
                 ceiling: Int, enlarges: Bool = false) {
        self.enlarges = enlarges
        self.ready = ready; self.traits = traits; self.loras = loras; self.loraNames = loraNames
        self.ceiling = ceiling
    }
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

/// A series asked from the borrower's form, or a grid from its `silicontrol grid`. `job`: what the cells share — the prompt
/// as typed, the format, the stack, the edit's images —; its seeds are none, the grid's are.
package struct RemoteGridRequest: Codable, Equatable, Sendable {
    package var spec: RemoteGridSpec
    package var job: RemoteJob
    /// The borrower's request number when it waits for the lender's word (`silicontrol grid`): answered by
    /// `RemoteAnswer.gridQueued`. `nil`: the window, which reads the series published.
    package var request: Int?
    package init(spec: RemoteGridSpec, job: RemoteJob, request: Int? = nil) {
        self.spec = spec; self.job = job; self.request = request
    }
}

/// How a cell's job ended, as the panel says it.
package enum RemoteIssue: Codable, Equatable, Sendable {
    case finished, stopped, removed
    case failed(RemoteRefusal)
}

/// **A series of the borrower's on the lender** — every one of the session, the window's and `silicontrol grid`'s —,
/// published whole at each change but its images, which are history images sent as any other (`entry`): the borrower
/// lays them in their cells by their job.
package struct RemoteSeries: Codable, Equatable, Sendable {
    package var ordinal: Int
    package var spec: RemoteGridSpec
    package var job: RemoteJob
    /// Each cell's job, in the grid's order: its ordinal, and its settings whole.
    package var cells: [RemoteJob]
    package var issues: [Int: RemoteIssue]
    package init(ordinal: Int, spec: RemoteGridSpec, job: RemoteJob, cells: [RemoteJob], issues: [Int: RemoteIssue]) {
        self.ordinal = ordinal; self.spec = spec; self.job = job; self.cells = cells; self.issues = issues
    }
}

/// **The lender's word on a `render`**, carried back with the borrower's request number.
package enum RemoteAnswer: Codable, Equatable, Sendable {
    /// Queued as this job, with what the lender estimates it takes (`nil`: never measured).
    case queued(ordinal: Int, estimate: Double?, width: Int, height: Int)
    case refused(RemoteRefusal)
    /// A grid queued as this grid, its cell jobs in the grid's order, whole (`RemoteGridRequest.request`).
    case gridQueued(grid: Int, cells: [RemoteJob], estimate: Double?)
}

/// **A job of the borrower's that ended** on the lender — sent after its images, so that whoever follows it
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
