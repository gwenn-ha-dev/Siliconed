import Foundation

/// **What a lending Mac publishes, and what a borrowing one asks**, once their session is joined
/// (`SessionMessage`). Plain values — strings, numbers, the engine's `Codable` enums — so that both ends
/// decode exactly what the other encoded; the app projects its `Job`, `Entry` and state into them and
/// back.
///
///     mini → Neo   catalog     the ready models, what each accepts, its LoRAs, B — whenever it changes
///                  queue       the job running and its progress, the jobs waiting — whenever it changes
///                  entry + PNG one history image, its PNG as the blob that follows; `removed` the reverse
///                  preview     the denoising's latent preview as a small PNG blob, once per step
///                  quoted      the answer to a `quote`
///     Neo → mini   render      one job from the Neo's form; the mini checks it again (B, steps) before
///                              queueing it — the Neo's judgment is never the last word; an edit's images
///                              go first as blobs, or by their history ordinal
///                  quote       what an edit's format and a need with images are: the denoiser's rule
///                              and the memory need come from maps only the mini holds
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
    }
    package var origin: Origin
    package var name: String
    package var width: Int, height: Int
    package init(origin: Origin, name: String, width: Int, height: Int) {
        self.origin = origin; self.name = name; self.width = width; self.height = height
    }

    package var blob: UInt32? {
        switch origin {
        case .history: nil
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
    /// The mini's `t` ordinal; 0 in a `render` request (the mini numbers it).
    package var ordinal: Int
    package var model: String
    package var modelName: String
    package var prompt: String
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

    package init(ordinal: Int, model: String, modelName: String, prompt: String, width: Int, height: Int, steps: Int,
                 defaultSteps: Int?, loras: [RemoteLoRA], seeds: [UInt64], detail: Detail, solver: Solver,
                 schedule: SigmaSchedule, previews: Bool, neuralEngine: Bool?, references: [RemoteReference] = []) {
        self.references = references
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

    package init(ordinal: Int, job: RemoteJob, width: Int, height: Int, seed: UInt64, timings: Engine.Timings,
                 footprints: Engine.Footprints, evaluations: Int, metadata: [String: String], neuralEngine: Bool, png: UInt32) {
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
    package init(ready: [String], traits: [String: RemoteTraits], loras: [String: [RemoteLoRA]], loraNames: [String: String],
                 needs: [String: [String: Int]] = [:], ceiling: Int, settingsRead: Bool, neuralEngineOffered: Bool, fasterRendering: Bool) {
        self.ready = ready; self.traits = traits; self.loras = loras; self.loraNames = loraNames; self.needs = needs
        self.ceiling = ceiling
        self.settingsRead = settingsRead; self.neuralEngineOffered = neuralEngineOffered; self.fasterRendering = fasterRendering
    }

    package static func key(_ width: Int, _ height: Int) -> String { "\(width)x\(height)" }
}
