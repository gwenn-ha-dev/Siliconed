import Foundation

// **The API: closed modules, typed wires, a chain that refuses to be wired wrongly.**
//
// A render is always the same sequence of stages — text, [image encoding,] denoising, decoding —
// and what distinguishes one model from another is *which* module occupies each stage. This
// file names the stages (four protocols), what flows between them (`Conditioning`, `Latent`,
// `ImageRGB`) and the wiring rule: **two modules connect only if the wire is of the same type**.
// Text encoded for Anima does not condition Krea 2's DiT, a Flux VAE latent is not decoded with
// Qwen-Image's — and `Chain` refuses it at construction, not after a minute of encoder.
//
// The system is **closed** (the Reason vision, not ComfyUI): the modules are the repository's,
// ported and verified here. The protocols are public so a caller can compose its chains, not so
// it can write new links.
//
// What does not change, and must be read as a constraint: **each stage lives in its own scope**.
// A module builds its model, computes, and returns its memory on exit; the next stage begins only
// afterwards. That is what holds the 16 GB ceiling.

// ── the wires ──────────────────────────────────────────────────────────────────────────────

/// **What a text encoder produces, and therefore what a denoiser accepts.** Two formats are equal
/// only if they have the same name: the name describes the encoder, the layer taken and the shape.
public struct TextFormat: Hashable, Sendable, CustomStringConvertible {
    public let name: String
    /// **The text stage reads the reference images too** (`ImageTextModule`): the encoder sees them
    /// with the prompt, and the denoiser expects the slots they left in the sequence.
    public let readsImages: Bool
    public init(_ name: String, readsImages: Bool = false) { self.name = name; self.readsImages = readsImages }
    public var description: String { name }

    /// Qwen3-4B, chat template, second-to-last layer: `[tokens, 2560]`.
    public static let zImage = TextFormat("qwen3-4b · chat · penultimate layer")
    /// Qwen3-0.6B + T5 → Anima's conditioner: `[rows, 1024]`.
    public static let anima = TextFormat("qwen3-0.6b + t5 · conditioner anima")
    /// Qwen3-VL-4B, twelve layers taken, template stripped: `[tokens, 12 × 2560]`. The fusion and
    /// `txt_in` belong to the DiT (the LoRA touches them), hence to the denoiser.
    public static let krea2 = TextFormat("qwen3-vl-4b · 12 taps")
    /// Qwen3-4B, chat template without thinking, three layers taken (9, 18, 27), **512 rows
    /// padding included**: `[512, 3 × 2560]`. `context_embedder` belongs to the DiT.
    public static let klein = TextFormat("qwen3-4b · chat · taps 9, 18, 27 · 512 rows")
    /// Ministral-3 3B, no template, `<s>` in front, second-to-last layer, real tokens only:
    /// `[tokens, 3072]`. `text_proj` belongs to the DiT.
    public static let ernie = TextFormat("ministral-3 · no template · penultimate layer")
    /// Qwen3-VL-8B on the prompt **and the reference images** (raw template, `<imageN>` slots), the
    /// last layer before the final norm, system turn dropped: `[T, 4096]` and the `image_pad_mask` of
    /// its `T` rows (`Conditioning.imageSlots`). `txt_in` and the images' latents belong to the DiT.
    public static let qwenImage21 = TextFormat("qwen3-vl-8b · prompt + images · last layer before norm",
                                               readsImages: true)
}

/// **A latent's space**: the VAE that defined it. The same number of channels does not mean the
/// same space — Flux and Qwen-Image both have sixteen channels and do not decode each other.
public struct LatentSpace: Hashable, Sendable, CustomStringConvertible {
    public let name: String
    public let channels: Int
    /// Pixels per latent cell, along one side.
    public let factor: Int
    public var description: String { name }

    public static let flux = LatentSpace(name: "VAE Flux", channels: 16, factor: 8)
    public static let qwenImage = LatentSpace(name: "VAE Qwen-Image", channels: 16, factor: 8)
    /// **The FLUX.2 DiT's space**: the VAE's 32 channels, **packed 2×2** (128 channels, one cell
    /// per 16 pixels) and normalized by the VAE's `BatchNorm` — what `Flux2KleinPipeline` draws
    /// and denoises. The decoder denormalizes and unpacks (`Flux2DecodingModule`).
    public static let flux2 = LatentSpace(name: "VAE FLUX.2", channels: 128, factor: 16)
    /// Qwen-Image-2.1's VAE (`AutoencoderKLQwenImage21`): 64 channels, 16 pixels per cell, **no
    /// packing** — planar `[64, H/16, W/16]`, normalized per channel `(z − mean_c) / std_c`; the DiT's
    /// tokens are its transpose `[h·w, 64]` (`QwenImage21VAE.pack` / `unpack`). The VAE itself reads and
    /// writes RGBA (`QwenImage21EncodingModule`, `QwenImage21DecodingModule`).
    public static let qwenImage21 = LatentSpace(name: "VAE Qwen-Image-2.1", channels: 64, factor: 16)
}

/// The encoded text, ready for a denoiser of the same format.
public struct Conditioning: Sendable {
    public let format: TextFormat
    /// `values` is `[rows, width]`, row by row.
    public let rows: Int
    public let width: Int
    public let values: [Float]
    /// The prompt's tokens, for display — not always `rows` (Anima).
    public let tokens: Int
    /// **Which rows stand for an image** (`image_pad_mask`), one entry per row — empty for a format
    /// that does not read images (`TextFormat.readsImages`).
    public let imageSlots: [Bool]

    /// An already-encoded text — a golden tensor's, so a check can drive one of the product's
    /// denoisers without going back through the encoder.
    package init(format: TextFormat, rows: Int, width: Int, values: [Float], tokens: Int, imageSlots: [Bool] = []) {
        self.format = format; self.rows = rows; self.width = width; self.values = values
        self.tokens = tokens; self.imageSlots = imageSlots
    }
}

/// A planar latent `[channels, height, width]`, and the space that gives it meaning.
public struct Latent: Sendable {
    public let space: LatentSpace
    public let height: Int
    public let width: Int
    public var values: [Float]

    public init(space: LatentSpace, height: Int, width: Int, values: [Float]) {
        self.space = space; self.height = height; self.width = width; self.values = values
    }

    /// The starting noise. `Noise` does not reproduce `torch.randn`: a seed gives an image of the
    /// model, not diffusers'. It is drawn in planar order: a square therefore yields the same bits
    /// as before rectangles arrived. **Qwen-Image-2.1 is the exception**: its noise is
    /// `torch.randn((1, 1, 64, h, w))` to the bit (`TorchNoise`), so a seed gives diffusers' image.
    public static func noise(_ space: LatentSpace, height: Int, width: Int, seed: UInt64) -> Latent {
        var values = [Float](repeating: 0, count: space.channels * height * width)
        if space == .qwenImage21 {
            var noise = TorchNoise(seed: seed)
            values.withUnsafeMutableBufferPointer { noise.fill($0.baseAddress!, count: $0.count) }
        } else {
            var noise = Noise(seed: seed)
            values.withUnsafeMutableBufferPointer { noise.fill($0.baseAddress!, count: $0.count) }
        }
        return Latent(space: space, height: height, width: width, values: values)
    }
}

/// `[3, height, width]`, planar, in `[-1, 1]` — what `png()` and `cgImage()` read, and what an
/// image encoder receives. Reading a file or a `CGImage` and fitting to the format:
/// `ImageToImage.swift`.
public struct ImageRGB: Sendable, Equatable {
    public let pixels: [Float]
    public let height: Int
    public let width: Int

    public init(pixels: [Float], height: Int, width: Int) {
        precondition(height > 0 && width > 0, "ImageRGB: \(width)×\(height), an empty image")
        precondition(pixels.count == 3 * height * width,
                     "ImageRGB: \(pixels.count) values for 3 × \(height) × \(width)")
        self.pixels = pixels; self.height = height; self.width = width
    }
}

/// What a denoiser announces before starting: where it starts, how many evaluations, of which how
/// many reduced.
public struct DenoisingPlan: Sendable {
    /// The first step executed of the full schedule: 0 in txt2img, `t_start` in img2img.
    public let startStep: Int
    /// The starting σ, `σ[startStep]`: 1 in txt2img.
    public let startSigma: Float
    /// The evaluations actually paid for, from the starting step to the end.
    public let evaluations: Int
    /// Half-size evaluations (Z-Image's spectral schedule); 0 elsewhere.
    public let reduced: Int
    /// The latent grid of the reduced evaluations.
    public let reducedHeight: Int
    public let reducedWidth: Int

    /// **The same plan, stopped as a sketch** (`Request.sketch`): its first `n` evaluations — what an
    /// app estimates a sketch's duration from. `n` at or past `evaluations`: the plan itself.
    public func sketched(_ n: Int) -> DenoisingPlan {
        let kept = min(max(1, n), evaluations)
        return DenoisingPlan(startStep: startStep, startSigma: startSigma, evaluations: kept,
                             reduced: min(reduced, kept), reducedHeight: reducedHeight, reducedWidth: reducedWidth)
    }
}

/// **Where a sketch stops** (`Request.sketch`): after the first evaluation that brings σ to
/// `threshold` or below — the evaluations it takes depend on the schedule, not on a count.
///
/// Measured (512², estimate x̂₀ against the final image reduced ×4, by eye on the sheets):
///
///     Z-Image, 7 evaluations      σ after:  .947  .882  .800  .692 …      stop at 3 → 21 dB, pose,
///                                                                         clothes and light final
///     Qwen-Image-2.1, 6           σ after:  .963  .923  .837  .632 …      stop at 4 → 20 dB,
///                                                                         composition, soft
///
/// A step count is no rule — Qwen's schedule keeps σ high longer, and at 3 of 6 its estimate is a
/// blur (17 dB); σ is. Under a LoRA, Z-Image's pose still moved between 3 and 4: a sketch shows
/// where the image goes, the finished render is the judge.
///
/// **Qwen-Image-2.1 does not sketch** (`DenoisingModule.sketches`, 2026-10-06): its « soft » at 4 is,
/// seen through the app, blurred and grainy, and still at 5 (22 dB) — it removes the noise in its
/// last two evaluations. A sketch of it saves 15 s of 51 for an image no one can judge: its cells
/// are finished images.
public enum Sketch {
    public static let threshold: Float = 0.8

    /// The evaluations a sketch of `sigmas` (a full schedule, ending in 0) takes; null steps
    /// (σ unchanged) are not evaluations. Never fewer than 1, never more than the schedule's.
    public static func evaluations(sigmas: [Float]) -> Int {
        var n = 0
        for i in 0..<max(0, sigmas.count - 1) where sigmas[i + 1] != sigmas[i] {
            n += 1
            if sigmas[i + 1] <= threshold { return n }
        }
        return max(1, n)
    }
}

public extension Model {
    /// The evaluations a sketch of this model takes at this format (`Sketch`); `steps` nil: the
    /// model's default.
    func sketchEvaluations(steps: Int?, width: Int, height: Int) -> Int {
        chain.denoising.sketchEvaluations(steps: steps ?? chain.denoising.defaultSteps,
                                          height: height / chain.denoising.space.factor,
                                          width: width / chain.denoising.space.factor)
    }
}

// ── the modules ─────────────────────────────────────────────────────────────────────────────

/// **A render's thread, lent to each stage**: what holds for the whole request, the progress, the
/// cancellation token, and the items only the stage can measure (the tokenizer in the text stage,
/// the footprint before release in denoising).
///
/// **Not `Sendable`, and deliberately so**: it is mutable without a lock (`timings`, `footprints`,
/// the step counter), which is safe only because a single thread — the engine queue's — touches
/// it. What goes out to the UI is a copy of it (`Render`, the events).
///
/// **Public in name only**: the module protocols cite it, but nothing in it is usable outside the
/// package (`package`) — only the engine and the CLI checks construct it.
public final class Context {
    package let reproducible: Bool
    /// The render's token; the modules pass it to their DiTs and encoders (between layers).
    package let cancellation: Cancellation?
    /// Compute each step's preview (`Request.previews`).
    package let previews: Bool
    private let onProgress: (@Sendable (Engine.Event) -> Void)?
    package var timings = Engine.Timings()
    package var footprints = Engine.Footprints()
    /// **What a stage computes once for the whole batch** — what does not depend on the seed
    /// (Krea 2's text fusion and modulation table). Lives as long as the context, hence one
    /// batch; each module prefixes its keys with its name and checks what it takes back.
    package var reserve: [String: Any] = [:]
    /// **The render cache, between renders** (`RenderCache`): set by the engine for a model built from
    /// a library, when the settings allow it. `nil` for the checks, which build their own context —
    /// a check that read a previous render's floats would judge nothing.
    package var cache: RenderCache?
    /// **The denoiser kept from one image of a batch to the next**: every image of a request has the
    /// same format, text and stack, so its DiT — arenas and map — is built once; building it again
    /// is a fresh allocation wave per image, which sends the others' memory into the compressor for
    /// good. Released by the chain before the decoding (the decoder never shares the room with it).
    package var keptDenoiser: AnyObject?
    /// The current batch image, and the evaluations done on it / planned.
    var image = 0
    private var stepsDone = 0
    private(set) var plannedEvaluations = 0
    /// **A sketch** (`Request.sketch`): after this many evaluations `stepDone` keeps x̂₀ in
    /// `estimate` and throws `Sketched` — the way out that cancellation already proved through every
    /// denoiser, arenas unwound by the scopes. `nil`: the whole render.
    package var stopAfter: Int?
    /// **More detail** (`Request.detail`): every sampler asks it what σ to tell its DiT (`told`).
    package var detail: Detail = .normal
    /// `Detail.scale` of the render's size: the standard bell's amount shrinks as the image grows.
    package var detailScale: Double = 1

    /// The σs the DiT is told, for a full schedule of which steps `start…` run.
    package func told(_ sigmas: [Float], start: Int, bell: Detail.Bell = .standard) -> [Float] {
        detail.modelSigmas(sigmas, start: start, bell: bell, scale: detailScale)
    }
    /// x̂₀ at the sketch's last evaluation, `[channels, fullHeight, fullWidth]`.
    package private(set) var estimate: [Float]?
    /// Thrown by `stepDone` when the sketch is done; caught by the engine, never seen by a caller.
    package struct Sketched: Error {}

    /// For the checks, which drive one of the product's modules outside a render.
    package init(reproducible: Bool, cancellation: Cancellation? = nil, previews: Bool = false,
                onProgress: (@Sendable (Engine.Event) -> Void)?) {
        self.reproducible = reproducible; self.cancellation = cancellation; self.previews = previews
        self.onProgress = onProgress
    }
    func emit(_ event: Engine.Event) { onProgress?(event) }

    /// Throws `EngineError.cancelled` if the render is cancelled.
    package func check() throws { try cancellation.check() }

    /// The denoiser announces its evaluations before the first: that is the events' `total`.
    func beginDenoising(evaluations: Int) {
        stepsDone = 0; estimate = nil
        plannedEvaluations = min(evaluations, stopAfter ?? evaluations)
    }

    /// **An evaluation is done**: the numbered `step` event (1 … `total`), then the preview if
    /// requested. `x` is the latent **after** the update, `v` the step's derivative `dx/dσ`,
    /// `nextSigma` the σ reached: x̂₀ = x − σ_next·v (see `Preview`). Then checks cancellation.
    /// `evaluatedHeight`: the grid this step evaluated (half-size under the spectral schedule);
    /// `fullHeight`: the render's, which is that of `x` and `v`.
    func stepDone(sigma: Float, seconds: Double, evaluatedHeight: Int, evaluatedWidth: Int,
                 space: LatentSpace, x: UnsafePointer<Float>, v: UnsafePointer<Float>,
                 nextSigma: Float, fullHeight: Int, fullWidth: Int) throws {
        stepsDone += 1
        let total = max(plannedEvaluations, stepsDone)
        emit(.step(image: image, index: stepsDone, total: total, sigma: sigma, seconds: seconds,
                     latentGridHeight: evaluatedHeight, latentGridWidth: evaluatedWidth))
        if previews, let projection = space.preview {
            let rgba = projection.apply(x: x, v: v, sigma: nextSigma,
                                            height: fullHeight, width: fullWidth)
            emit(.preview(Engine.Preview(image: image, index: stepsDone, total: total, height: fullHeight,
                                          width: fullWidth, rgba: rgba)))
        }
        if let stopAfter, stepsDone >= stopAfter {
            let count = space.channels * fullHeight * fullWidth
            var x0 = [Float](repeating: 0, count: count)
            for i in 0..<count { x0[i] = x[i] - nextSigma * v[i] }
            estimate = x0
            throw Sketched()
        }
        try check()
    }
}

/// The text stage: a prompt goes in, a `Conditioning` comes out. It returns its memory on exit.
public protocol TextModule: Sendable {
    /// What an app displays on the module's box ("Qwen3-VL-4B").
    var name: String { get }
    var output: TextFormat { get }
    func encoder(_ prompt: String, context: Context) throws -> Conditioning
}

/// **A text stage that sees the reference images** (Qwen-Image-2.1's Qwen3-VL): the prompt and the
/// references go in together — already at the denoiser's size (`DenoisingModule.preparedReferences`),
/// the very copy the image encoder then reads. Without references, it is an ordinary text stage.
public protocol ImageTextModule: TextModule {
    func encoder(_ prompt: String, images: [ImageRGB], context: Context) throws -> Conditioning
}

public extension ImageTextModule {
    func encoder(_ prompt: String, context: Context) throws -> Conditioning {
        try encoder(prompt, images: [], context: context)
    }
}

/// The denoising stage: it turns noise — or a noised image — into a latent, under a text of its
/// format.
///
/// **The starting step** (`start`) is the whole of img2img on the denoiser's side: it unrolls its
/// full schedule, computed for `step` (shift and μ included), but executes only steps
/// `start … step − 1`. The latent it receives is already noised to `σ[start]` (`Latent.start`).
public protocol DenoisingModule: Sendable {
    /// The identifier of the model whose DiT it carries (`z-image`, `anima`, `krea2`): the target
    /// a LoRA must declare for it to be applied. See `LoRA.checkTarget`.
    var model: String { get }
    var name: String { get }
    var entry: TextFormat { get }
    var space: LatentSpace { get }
    var defaultSteps: Int { get }
    /// The full schedule of `step` steps: `step + 1` σ values, ending in 0. It does not depend on
    /// the strength — img2img slices it, it does not recompute it.
    func sigmas(steps: Int) -> [Float]
    /// The same, for a latent grid — FLUX.2 shifts its schedule according to the image size. By
    /// default, that of `sigmas(step:)`.
    func sigmas(steps: Int, height: Int, width: Int) -> [Float]
    /// `height`, `width`: the latent grid, not the pixels. `withLoRA`: the request carries a
    /// stack — Z-Image turns its spectral schedule off by default there (`ZImageDenoisingModule.plan`).
    func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan
    func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                   context: Context) throws
    /// **How many reference images it can read** (editing, FLUX.2's "Kontext"): 0 for a denoiser
    /// without. A reference is not a starting point (img2img): it enters the sequence, next to the
    /// generated image, and the denoiser re-reads it at each layer.
    var maxReferences: Int { get }
    /// Denoising with references already encoded in its space. By default: none.
    func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int,
                   references: [Latent], lora: LoRA?, context: Context) throws
    /// **The reference images as this denoiser reads them**, prepared once for the whole render: the
    /// text stage (if it reads images) and the image encoder receive this same copy. By default,
    /// FLUX.2's sizing (`ImageRGB.forReference`).
    func preparedReferences(_ images: [ImageRGB]) throws -> [ImageRGB]
    /// **The format of an edit whose request names none**, from its first reference's size. By
    /// default FLUX.2's: the reference's own size under 1024², raised to the floor.
    func editFormat(referenceWidth: Int, referenceHeight: Int) -> (width: Int, height: Int)
    /// **What the denoiser refuses before any computation**: a step count it has no schedule for, a
    /// starting image (img2img) when its pipeline has none. By default, nothing.
    func check(steps: Int, startImage: Bool) throws
    /// Whether the denoiser starts from an encoded image (img2img). By default yes; a pipeline
    /// without img2img (Qwen-Image-2.1) says no, and the chain then offers no image entry.
    var acceptsStartImage: Bool { get }
    /// Whether its estimate x̂₀ is readable early (`Sketch`). By default yes; a denoiser that clears
    /// its noise only at the end says no, and its sketches are finished images.
    var sketches: Bool { get }
}

/// The image encoding stage (img2img): an image in the render's format goes in, a latent of its
/// space comes out — normalized as the denoiser expects. It returns its memory on exit.
public protocol ImageEncodingModule: Sendable {
    var name: String { get }
    var space: LatentSpace { get }
    func encoder(_ image: ImageRGB, context: Context) throws -> Latent
}

/// The decoding stage: a latent of its space goes in, an image comes out.
public protocol DecodingModule: Sendable {
    var name: String { get }
    var space: LatentSpace { get }
    func decode(_ latent: Latent, context: Context) throws -> ImageRGB
}

public extension DenoisingModule {
    func sigmas(steps: Int, height: Int, width: Int) -> [Float] { sigmas(steps: steps) }
    /// The plan of a request without a LoRA.
    func plan(height: Int, width: Int, steps: Int, start: Int) -> DenoisingPlan {
        plan(height: height, width: width, steps: steps, start: start, withLoRA: false)
    }
    var maxReferences: Int { 0 }
    func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int,
                   references: [Latent], lora: LoRA?, context: Context) throws {
        guard references.isEmpty else { throw Request.Failure.references(references.count, max: maxReferences) }
        try denoise(&latent, text: text, steps: steps, start: start, lora: lora, context: context)
    }
    func preparedReferences(_ images: [ImageRGB]) throws -> [ImageRGB] { images.map { $0.forReference() } }
    func editFormat(referenceWidth: Int, referenceHeight: Int) -> (width: Int, height: Int) {
        let size = ImageRGB.referenceSize(width: referenceWidth, height: referenceHeight)
        var (w, h) = (max(Format.minimumSide, size.width), max(Format.minimumSide, size.height))
        // Raising the short side to the floor can push a long strip past the area cap (10:1 gave
        // 3232×512): the long side gives way, as in Qwen-Image-2.1's rule.
        if w * h > Format.maxSurface {
            if w >= h { w = Format.maxSurface / h / Format.multiple * Format.multiple }
            else { h = Format.maxSurface / w / Format.multiple * Format.multiple }
        }
        return (w, h)
    }
    func check(steps: Int, startImage: Bool) throws {}
    var acceptsStartImage: Bool { true }
    var sketches: Bool { true }
    /// The evaluations a sketch takes on the latent grid `height × width`: where σ reaches the
    /// threshold, or all of them for a denoiser that does not sketch.
    func sketchEvaluations(steps: Int, height: Int, width: Int) -> Int {
        let all = plan(height: height, width: width, steps: steps, start: 0).evaluations
        guard sketches else { return all }
        return min(all, Sketch.evaluations(sigmas: sigmas(steps: steps, height: height, width: width)))
    }
}

/// A module without a name of its own carries its type's: a hand-composed chain can still be
/// drawn.
public extension TextModule { var name: String { String(describing: Self.self) } }
public extension DenoisingModule { var name: String { String(describing: Self.self) } }
public extension ImageEncodingModule { var name: String { String(describing: Self.self) } }
public extension DecodingModule { var name: String { String(describing: Self.self) } }

// ── the chain ───────────────────────────────────────────────────────────────────────────────

/// **Three modules wired, plus the image encoder if there is one — and verified wirable.**
///
/// The encoder is optional **in the type** (a chain without it only does txt2img) and present in
/// every `Model` of the catalog. Its wire is the same as the decoder's: it must speak the
/// denoiser's space.
public struct Chain: Sendable {
    public let text: any TextModule
    public let denoising: any DenoisingModule
    public let decoding: any DecodingModule
    public let encoding: (any ImageEncodingModule)?

    package enum Failure: Error, CustomStringConvertible {
        case incompatibleText(output: TextFormat, entry: TextFormat)
        case latentIncompatible(output: LatentSpace, entry: LatentSpace)
        case imageIncompatible(output: LatentSpace, entry: LatentSpace)
        case textWithoutImages(TextFormat)
        package var description: String {
            switch self {
            case let .incompatibleText(output, entry):
                return "wire refused: the text comes out as « \(output) », the denoiser expects « \(entry) »"
            case let .textWithoutImages(format):
                return "wire refused: « \(format) » reads the reference images, and the text module does not take them"
            case let .latentIncompatible(output, entry):
                return "wire refused: the denoiser returns a latent of \(output), the decoder expects \(entry)"
            case let .imageIncompatible(output, entry):
                return "wire refused: the image encoder returns a latent of \(output), the denoiser expects \(entry)"
            }
        }
    }

    public init(text: any TextModule, denoising: any DenoisingModule,
                decoding: any DecodingModule, encoding: (any ImageEncodingModule)? = nil) throws(EngineError) {
        guard text.output == denoising.entry else {
            throw EngineError(Failure.incompatibleText(output: text.output, entry: denoising.entry))
        }
        // A format that reads images must come from a module that takes them: the denoiser would
        // otherwise find no slot for its references.
        guard !text.output.readsImages || text is any ImageTextModule else {
            throw EngineError(Failure.textWithoutImages(text.output))
        }
        guard denoising.space == decoding.space else {
            throw EngineError(Failure.latentIncompatible(output: denoising.space, entry: decoding.space))
        }
        if let encoding, encoding.space != denoising.space {
            throw EngineError(Failure.imageIncompatible(output: encoding.space, entry: denoising.space))
        }
        self.text = text; self.denoising = denoising; self.decoding = decoding
        self.encoding = encoding
    }

    /// **An input of the chain**: what an app must offer the user to feed it.
    public struct Entry: Sendable, Hashable {
        public enum Kind: Sendable, Hashable {
            /// The prompt (`Request.prompt`).
            case text
            /// The starting image (`Request.image`, `Request.strength`).
            case image
            /// An editing reference image (`Request.references`) — as many as
            /// `DenoisingModule.maxReferences`.
            case reference
        }
        public let kind: Kind
        /// Without it, the render is refused; optional, it only changes what it does.
        public let isRequired: Bool
        /// The module that receives it ("Qwen3-VL-4B", "VAE Flux").
        public let module: String
    }

    /// **What the chain accepts, in signal order** — what an app draws as inputs instead of a
    /// fixed form. The text always, the image if there is an image
    /// encoder and the denoiser starts from one, the references if it reads them.
    public var entries: [Entry] {
        [Entry(kind: .text, isRequired: true, module: text.name)]
            + (encoding.flatMap { denoising.acceptsStartImage
                                     ? [Entry(kind: .image, isRequired: false, module: $0.name)] : nil } ?? [])
            + (encoding != nil && denoising.maxReferences > 0
                ? [Entry(kind: .reference, isRequired: false, module: denoising.name)] : [])
    }
}

// ── the request ──────────────────────────────────────────────────────────────────────────────

/// A LoRA of the stack: a forged map (`ForgeLoRA`) and its strength.
///
/// **A strength is a `Double` everywhere in the API** (a LoRA's as well as img2img's); the LoRA
/// brings it down to `Float` at merge time, where it multiplies fp32 weights.
public struct LoRAEntry: Sendable, Hashable {
    public var path: String
    public var strength: Double
    public init(_ path: String, strength: Double = 1) { self.path = path; self.strength = strength }

    /// `carte.silicon:0.8,autre.silicon:0.5` — the syntax of `SILICONED_LORA`. A missing or
    /// unreadable strength is 1. An entry made only of colons (`:`) names no file: it is kept as
    /// written, so the render refuses it by name (`EngineError.fileMissing(path: ":")`) rather than
    /// the parse trapping on a piece that is not there.
    package static func stack(_ text: String) -> [LoRAEntry] {
        text.split(separator: ",").map { entry in
            let pieces = entry.split(separator: ":")
            guard let path = pieces.first else { return LoRAEntry(String(entry)) }
            return LoRAEntry(String(path),
                              strength: pieces.count > 1 ? Double(Float(pieces[1]) ?? 1) : 1)
        }
    }

    /// `popart100` — the short form a file name carries.
    package var shortTag: String {
        let base = (((path as NSString).lastPathComponent) as NSString).deletingPathExtension
            .replacingOccurrences(of: ".lora", with: "").lowercased().prefix(12)
        return "\(base)\(Int((strength * 100).rounded()))"
    }
}

/// **A request, and everything that distinguishes it from another — whatever the model.** What is
/// specific to a model (Z-Image's spectral schedule, for instance) lives on its module.
///
/// **Building a request reads nothing** — neither the machine's profile nor a file: what depends
/// on them (`reproducible` left at `nil`) is resolved at render time. An app can therefore hold
/// one in its state before having chosen its library and loaded its profile.
public struct Request: Sendable, Equatable {
    public var prompt: String
    /// The image's format, in pixels. See `Format` for what is accepted. Changing it refits the
    /// input image — from the **already-fitted** image: to change format without losing edges,
    /// set the source image again (`image = …`) after the format.
    public var width: Int { didSet { fit() } }
    public var height: Int { didSet { fit() } }
    public var seed: UInt64
    /// `nil`: the denoiser's default (`ModelCard.defaultSteps`). At least 1, otherwise the render
    /// throws `EngineError.invalidSteps` before any computation.
    public var steps: Int?
    public var loras: [LoRAEntry]
    /// **Same seed, same bits**: the GPU/AMX split is not servo-controlled. See `Conductor.driven`.
    /// `nil`: the machine's setting (`EngineSettings.effective.frozenCut`, true by default), **read at
    /// render time**, not at construction.
    public var reproducible: Bool?
    /// **The img2img starting image.** Set at any size, it is **immediately** filled and
    /// center-cropped to the request's format (`ImageRGB.fitted`): a request never keeps a
    /// full-size photo (48 Mpx would weigh 576 MB of floats for the whole render). `nil`: txt2img
    /// — one way of doing things, not two APIs.
    public var image: ImageRGB? { didSet { fit() } }
    /// diffusers' `strength` (`Strength`), in `]0 ; 1]`. No effect without `image`.
    public var strength: Double
    /// **The editing reference images** ("change her jacket to red"): the denoiser re-reads them
    /// at each layer, and the generated image starts from noise — nothing to do with `image`,
    /// which is its starting point. Each keeps **its own** aspect ratio, at the size the denoiser
    /// gives it (`DenoisingModule.preparedReferences`): FLUX.2 [klein] under 1024² at multiples of 16,
    /// Qwen-Image-2.1 at ~1024² at multiples of 32 (Pillow's Lanczos) — whatever the request's format.
    /// Qwen-Image-2.1's encoder also SEES them: the prompt names them `<image1>`, `<image2>`… in this
    /// order, and the first is the image being edited. Empty: no editing. Only the denoisers that
    /// announce it read them (`maxReferences`: 4 for FLUX.2 [klein], 3 for Qwen-Image-2.1).
    public var references: [ImageRGB] = []
    /// **A preview at each step** (`Event.preview`, `Engine.Preview`). False by default: nothing
    /// is then computed or copied.
    public var previews: Bool
    /// **A sketch: stop after this many evaluations** and decode the denoiser's estimate of the
    /// final image, x̂₀ = x − σ·v, instead of the image. `nil`: the whole render. The estimate is free
    /// (the evaluation computed `v` anyway) and speaks early: on Z-Image at 512², after 3 of 7
    /// evaluations the pose, the clothes and the light are those of the final image. The
    /// evaluations done are **the first ones of the whole render, bit for bit** — a sketch is a
    /// prefix, not another render: the same request without `sketch` finishes the image it showed.
    /// Under 1, one evaluation; at or past the plan's count, the whole render (`Render.sketch` nil).
    public var sketch: Int?
    /// **More detail** (`Detail`): the denoiser is told, mid-trajectory, that a little less noise
    /// remains than really does, and leaves finer texture. Same evaluations, same cost; `normal` is
    /// the render without it, to the bit. Every denoiser honours it (they all step a σ schedule).
    public var detail: Detail = .normal
    /// **Variations** (`Variation`): the seed's starting noise turned towards other seeds', in order —
    /// cousins of the image `seed` gives, its composition kept in proportion to `strength`
    /// (`Variation.Amount`). Empty: the seed's noise, to the bit. Only the noise changes:
    /// the text, the images and the render cache's keys do not see it. In a batch, each seed is turned
    /// by the same variations.
    public var variations: [Variation] = []

    /// The most steps a request may ask for. The Turbo models are distilled for 6 to 8; the cap is
    /// there for what does not come from the stepper — a PNG's metadata, `silicontrol --steps`, the
    /// API — where 2·10⁹ steps would allocate the schedule (8 GB) before the first evaluation.
    public static let maximumSteps = 50

    /// What a request refuses, before any computation.
    package enum Failure: Error, CustomStringConvertible, Equatable {
        /// An empty or blank prompt: every model refuses it (Anima would have no token to
        /// encode; the others would render an image nothing asked for).
        case emptyPrompt
        /// A number of steps outside `1...maximumSteps`.
        case steps(Int)
        /// More references than the denoiser reads (0: it reads none).
        case references(Int, max: Int)
        package var description: String {
            switch self {
            case let .references(n, max) where max == 0:
                return "\(n) reference image(s): this model does no reference editing (FLUX.2 [klein] and Qwen-Image-2.1 do)"
            case let .references(n, max):
                return "\(n) reference images: this model reads at most \(max)"
            case .emptyPrompt: return "empty prompt: there is nothing to render"
            case .steps(let n): return "\(n) steps: a render takes between 1 and \(Request.maximumSteps)"
            }
        }
    }

    public init(_ prompt: String, width: Int, height: Int, seed: UInt64 = 42, steps: Int? = nil,
                loras: [LoRAEntry] = [], image: ImageRGB? = nil, strength: Double = Strength.defaultValue,
                previews: Bool = false, reproducible: Bool? = nil) {
        self.prompt = prompt; self.width = width; self.height = height; self.seed = seed
        self.steps = steps; self.loras = loras; self.image = image; self.strength = strength
        self.previews = previews; self.reproducible = reproducible
        fit()   // `didSet` does not run inside an `init`
    }

    /// The `resolution × resolution` square — the common case, written short.
    public init(_ prompt: String, resolution: Int = 1024, seed: UInt64 = 42, steps: Int? = nil,
                loras: [LoRAEntry] = [], image: ImageRGB? = nil, strength: Double = Strength.defaultValue,
                previews: Bool = false, reproducible: Bool? = nil) {
        self.init(prompt, width: resolution, height: resolution, seed: seed, steps: steps,
                  loras: loras, image: image, strength: strength, previews: previews, reproducible: reproducible)
    }

    /// The image at the current format. A format refused by `Format` is not fitted (the render
    /// will throw on the format, not here); an image already at the format comes back as is, bit
    /// for bit.
    private mutating func fit() {
        guard let source = image, source.width != width || source.height != height,
              (try? Format.check(width: width, height: height)) != nil else { return }
        image = source.fitted(width: width, height: height)
    }
}

/// **The formats a render accepts — three rules, each with its reason.**
///
/// 1. **Each side ≥ 512 px** (latent ≥ 64). This is the repository's floor
///    (`Spectral.minimumSide`), read **per side** and not per area: 832×1216 has the area of
///    1024², but a 416×608 would have that of 512² with one side outside the models' domain. The
///    floor applies to *every* evaluation — hence the spectral schedule's rule, which goes down
///    only if both half-sizes stay at the floor (`Spectral.steps`).
/// 2. **Each side a multiple of 16**: the VAE divides by 8, the DiTs' patch by 2.
/// 3. **Area at most `maxSurface`** — measured, not deduced. See its note.
public enum Format {
    /// The smallest side, in pixels: `Spectral.minimumSide` latent cells of 8 px.
    public static let minimumSide = Spectral.minimumSide * 8
    public static let multiple = 16
    /// **The area ceiling, in pixels: that of 1024×1536, measured.**
    ///
    /// The 16 GB criterion had only been measured at 1024², and an app lets users type 2048×2048.
    /// A measurement rendered the three models at 832×1216 and 1216×832 (the area of 1024² within 3.5 %):
    /// peak of 4.8 GB (Z-Image), 5.3 GB (Anima), 5.2 GB (Krea 2). Then at 1024×1536, ×1.5:
    /// Z-Image and Anima passed, but **Krea 2 died** in its first SDPA — 48 heads over 6,156
    /// tokens, a 7.3 GB matrix that `MPSGraph` split itself by committing into our buffer. Since
    /// then Krea 2's SDPA splits its own queries beyond `Attention.maxMatrix`, and the three
    /// models fit at 1024×1536 — measured peaks (`/usr/bin/time -l`, footprint): **Krea 2
    /// 7.8 GB** (439 s), **Z-Image 7.5 GB** (with PopArt, 174 s), **Anima 7.6 GB** (91 s), all at
    /// decoding. Beyond that, nothing is measured: the ceiling is the last format that was.
    public static let maxSurface = 1024 * 1536

    /// Throws if the format is not renderable. The order of the rules is that of the most useful error.
    public static func check(width: Int, height: Int) throws(EngineError) {
        guard width >= minimumSide, height >= minimumSide else {
            throw .imageTooSmall(side: min(width, height), minimum: minimumSide)
        }
        guard width % multiple == 0, height % multiple == 0 else {
            throw .formatRefused(width: width, height: height, reason: .notMultiple)
        }
        // Each side bounded BEFORE the product: `99999999984 × 99999999984` would overflow.
        let maximumSide = maxSurface / minimumSide
        guard width <= maximumSide, height <= maximumSide, width * height <= maxSurface else {
            throw .formatRefused(width: width, height: height, reason: .tooLarge)
        }
    }

    /// **`LxH`: width × height**, the order of Draw Things and of screens — `832x1216` is a
    /// portrait. `×` counts as `x`; a lone integer is a square. Checks nothing but the syntax.
    public static func parse(_ text: String) -> (width: Int, height: Int)? {
        let pieces = text.lowercased().replacingOccurrences(of: "×", with: "x")
            .split(separator: "x", omittingEmptySubsequences: false)
        switch pieces.count {
        case 1: return Int(pieces[0]).map { ($0, $0) }
        case 2:
            guard let l = Int(pieces[0]), let h = Int(pieces[1]) else { return nil }
            return (l, h)
        default: return nil
        }
    }

    /// `512` for a square, `832x1216` otherwise — what a file name carries. Square names do not
    /// change: the close-out and the journal find them again.
    package static func fileLabel(width: Int, height: Int) -> String {
        width == height ? "\(width)" : "\(width)x\(height)"
    }

    /// `512²` or `832×1216` — what is displayed.
    package static func label(width: Int, height: Int) -> String {
        width == height ? "\(width)²" : "\(width)×\(height)"
    }
}

// ── the engine ───────────────────────────────────────────────────────────────────────────

extension Engine {
    /// **Renders an image. Blocks, and serializes concurrent calls** — the form for the CLI and
    /// tools; an app takes the `async` facade (`Facade.swift`), with the same parameters in the
    /// same order. `cancellation`: raised from another thread, the render stops at the next layer
    /// boundary and throws `EngineError.cancelled`, memory returned. `onProgress` is called on the
    /// render thread. A custom chain goes through a `Model(card:chain:)`. **Throws only
    /// `EngineError`**, the preflights first (`Preflight`).
    public func render(_ request: Request, model: Model, cancellation: Cancellation? = nil,
                       onProgress: (@Sendable (Event) -> Void)? = nil) throws(EngineError) -> Render {
        try renderBatch(request, seeds: [request.seed], model: model, cancellation: cancellation,
                      onProgress: onProgress)[0]
    }

    /// **A batch: the same request under several seeds**, the text (and the input image) encoded
    /// **only once**. `request.seed` is ignored. Each image has its events (`stage(_, image: n)`,
    /// `step(image: n, …)`, `image(index: n, …)`) and its `Render`; cancellation is checked between
    /// two images as everywhere else.
    public func renderBatch(_ request: Request, seeds: [UInt64], model: Model, cancellation: Cancellation? = nil,
                          onProgress: (@Sendable (Event) -> Void)? = nil) throws(EngineError) -> [Render] {
        try EngineError.boundary {
            try Engine.file.sync { try execute(request, seeds, model, cancellation, onProgress) }
        }
    }

    /// The same batch, called **from** the engine queue (the `async` facade is already on it).
    func renderBatchOnQueue(_ request: Request, seeds: [UInt64], model: Model, cancellation: Cancellation?,
                            onProgress: (@Sendable (Event) -> Void)?) throws -> [Render] {
        dispatchPrecondition(condition: .onQueue(Engine.file))
        return try execute(request, seeds, model, cancellation, onProgress)
    }

    /// **The render, stage by stage — a batch of one image or more.**
    ///
    /// The text and the input image each live in their own scope and return their memory before
    /// the DiT; only what they produce survives (the `Conditioning`, a few MB; the latent z₀).
    /// Then, **for each seed, exactly what an isolated render would do**: noise, start,
    /// denoising (the module builds its DiT and returns it), decoding. Keeping the DiT from one
    /// image to the next would gain nothing: its construction costs a few milliseconds, the
    /// weights being re-read at every evaluation. What was costly and does not depend on
    /// the seed — Krea 2's fusion and modulation — is kept in `Context.reserve`. Image n of a
    /// batch stays **bit-identical** to an isolated render of its seed (verified). Between two
    /// renders — two processes, the CLI and the app —, the text, the encoded images and Qwen's last
    /// edit's K/V are read back from the library's cache (`RenderCache`), the same floats.
    private func execute(_ request: Request, _ seeds: [UInt64], _ model: Model,
                          _ cancellation: Cancellation?,
                          _ onProgress: (@Sendable (Event) -> Void)?) throws -> [Render] {
        let chain = model.chain, cache = model.library?.renderCache
        let space = chain.denoising.space
        guard !seeds.isEmpty else { return [] }
        // A request cancelled while waiting on the queue pays for nothing: neither `start`
        // nor the opening of its LoRA stack.
        try cancellation.check()
        // Installed, license, format, memory, no other render: the first that fails throws. The
        // lock is held until this function returns, the whole batch.
        // The budget is read once, here, at the render's launch (`MemoryBudget`): the preflight judges
        // the floor against it, and the settings that depend on memory derive from it (`MemoryPlan`).
        let budget = MemoryBudget.current()
        let lock = try Preflight.run(model, width: request.width, height: request.height, budget: budget)
        defer { withExtendedLifetime(lock) {} }
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Request.Failure.emptyPrompt
        }
        // The stack opens BEFORE any computation: an unreadable map throws here, not after a
        // minute — and so does a LoRA forged for another model. The DiTs apply a LoRA only to the
        // modules it names: Anima's on Krea 2 would touch none, and the render would come out
        // without it, without a word.
        let stack = request.loras.isEmpty
            ? nil : try LoRA(paths: request.loras.map { ($0.path, Float($0.strength)) })
        try stack?.checkTarget(model: chain.denoising.model)
        let height = request.height / space.factor, width = request.width / space.factor
        let steps = request.steps ?? chain.denoising.defaultSteps
        guard (1...Request.maximumSteps).contains(steps) else { throw Request.Failure.steps(steps) }
        // **img2img is judged before any computation, too**: an out-of-range strength, a chain
        // without an encoder, a strength that leaves no evaluation all throw here.
        var start = 0
        if request.image != nil {
            guard chain.encoding != nil else { throw Strength.Failure.withoutEncoder(model: chain.denoising.model) }
            guard request.strength > 0, request.strength <= 1 else { throw Strength.Failure.outOfBounds(request.strength) }
            start = Strength.startStep(steps: steps, strength: request.strength)
        }
        if !request.references.isEmpty {
            guard chain.encoding != nil, request.references.count <= chain.denoising.maxReferences else {
                throw Request.Failure.references(request.references.count,
                                              max: chain.encoding == nil ? 0 : chain.denoising.maxReferences)
            }
        }
        try chain.denoising.check(steps: steps, startImage: request.image != nil)
        // Detail, like a LoRA, turns Z-Image's spectral schedule off (`ZImageDenoisingModule.plan`).
        let plan = chain.denoising.plan(height: height, width: width, steps: steps, start: start,
                                          withLoRA: stack != nil || request.detail != .normal)
        guard plan.evaluations > 0 else {
            // The floor: the last start that keeps an evaluation, `d`, is reached as soon as
            // `N − N·s < d + 1`, i.e. `s > (N − d − 1)/N`.
            let last = (0..<steps).last {
                chain.denoising.plan(height: height, width: width, steps: steps, start: $0,
                                       withLoRA: stack != nil).evaluations > 0
            } ?? 0
            throw Strength.Failure.noEvaluation(strength: request.strength, steps: steps,
                                               floor: Double(steps - last - 1) / Double(steps))
        }
        // A forced spectral schedule (`ZImageDenoisingModule(spectral:)`, `SILICONED_SPECTRAL`) that
        // would go below the floor is refused here, not in `Sampler.run` after a minute of text.
        if plan.reduced > 0, min(plan.reducedHeight, plan.reducedWidth) < Spectral.minimumSide {
            throw Sampler.Failure.belowFloor(side: min(plan.reducedHeight, plan.reducedWidth))
        }
        // The machine's profile is read HERE, at render time — never at request construction.
        let reproducible = request.reproducible ?? EngineSettings.effective.frozenCut
        let context = Context(reproducible: reproducible, cancellation: cancellation,
                                previews: request.previews, onProgress: onProgress)
        context.cache = cache
        // A sketch that reaches the plan's last evaluation is the whole render: no stop, no estimate.
        let sketch = request.sketch.map { max(1, $0) }.flatMap { $0 < plan.evaluations ? $0 : nil }
        context.stopAfter = sketch
        context.detail = request.detail
        context.detailScale = Detail.scale(width: request.width, height: request.height)
        // The library's warnings become events for the duration of the render (and go to
        // `Warnings.outsideRender` without `onProgress`: a "✗ VITESSE non finie" is not lost),
        // and memory is returned to the system on exit — **including on a cancellation**: the
        // arenas are unmapped by the scope unwinding, this returns what malloc kept.
        let warn: @Sendable (String) -> Void = { message in
            if let onProgress { onProgress(.warning(message)) } else { Warnings.outsideRender?(message) }
        }
        let floor = model.card.memoryNeed(width: request.width, height: request.height)
        let comfortable = model.card.memoryComfort(width: request.width, height: request.height)
        return try MemoryPlan.during(budget: budget, floor: floor, comfortable: comfortable) { try Warnings.during(warn) {
            defer { Arena.releaseToSystem() }
            // A GPU or AMX submission that failed reaches this render's journal, even when its cause
            // was already said by an earlier render (`Attention.report` says each cause once).
            let gpuFailures = Attention.failures
            defer { Attention.summarize(since: gpuFailures) }
            let totalStart = Date()
            context.footprints.start = Arena.processFootprint()
            // A sketch announces the evaluations it will do: the progress reaches 1 at its image.
            context.emit(.start(Plan(
                denoising: sketch.map { plan.sketched($0) } ?? plan, seeds: seeds,
                stages: request.image != nil || !request.references.isEmpty ? [.text, .image, .denoising, .decoding]
                                                                            : [.text, .denoising, .decoding])))

            // ── The references at the denoiser's size, once: a text stage that reads images and the
            //    image encoder below receive this same copy. ──
            let prepared = request.references.isEmpty ? [] : try chain.denoising.preparedReferences(request.references)

            // ── The text, once for the whole batch. The module returns its memory on exit. ──
            try context.check()
            context.emit(.stage(.text, image: 0))
            let begin = Date()
            // Read back from the cache when the same module has already encoded the same prompt and
            // the same images (`RenderCache`): the floats it wrote, so the same bits.
            // The images enter the key only if the encoder sees them (FLUX.2 [klein]'s is blind).
            let textKey = (chain.text as? any CacheIdentified).flatMap { module in
                cache.map { _ in RenderCache.textKey(module, prompt: request.prompt,
                                                     images: chain.text is any ImageTextModule ? prepared : [],
                                                     reproducible: reproducible) }
            }
            let text: Conditioning
            if let textKey, let kept = cache?.conditioning(textKey) {
                text = kept
            } else {
                text = try (chain.text as? any ImageTextModule)?.encoder(request.prompt, images: prepared, context: context)
                    ?? chain.text.encoder(request.prompt, context: context)
                if let textKey { cache?.store(text, key: textKey) }
            }
            context.timings.text = Date().timeIntervalSince(begin)
            context.timings.encoder = context.timings.text - context.timings.tokenizer
            context.emit(.text(tokens: text.tokens, tokenizer: context.timings.tokenizer,
                                    encoder: context.timings.encoder))
            Arena.releaseToSystem()
            context.footprints.text = Arena.processFootprint()

            // ── The image (img2img), once: the encoder lives in its own scope, before the DiT. ──
            var z₀: Latent? = nil
            if let image = request.image, let encoding = chain.encoding {
                try context.check()
                context.emit(.stage(.image, image: 0))
                let encodingStart = Date()
                // Already at the format (the request fitted it on receipt): `fitted` returns it as is.
                z₀ = try encode(image.fitted(width: request.width, height: request.height), with: encoding,
                                context: context)
                context.timings.encoding = Date().timeIntervalSince(encodingStart)
                Arena.releaseToSystem()
                context.emit(.encoding(seconds: context.timings.encoding))
            }

            // ── The references (editing), once: same encoder, each at its own size. ──
            var references: [Latent] = []
            if !prepared.isEmpty, let encoding = chain.encoding {
                try context.check()
                context.emit(.stage(.image, image: 0))
                let referencesStart = Date()
                for image in prepared {
                    try context.check()
                    references.append(try encode(image, with: encoding, context: context))
                    Arena.releaseToSystem()
                }
                context.timings.encoding += Date().timeIntervalSince(referencesStart)
                context.emit(.encoding(seconds: context.timings.encoding))
            }

            // ── Every image's denoising, then every decoding. ──
            // **Not denoise-decode, denoise-decode** (measured: Z-Image 1408×704, a batch of 3 under an idle
            // ballast swapped 18 008 pages where 1 image left the compressor 5.8 GB of room). Each switch
            // between the DiT and the decoder is a wave of fresh allocations — the arenas released, the
            // decoder's graph, the arenas again — and each wave sends the others' anonymous memory into
            // the compressor, which keeps it: its floor climbed ~2.3 GB at every image boundary, never
            // within a denoising. In this order a batch has the one switch of a single render; the
            // latents waiting meanwhile are kilobytes. Same bits: each image is the same computation.
            var pending: [(latent: Latent, timings: Timings, elapsed: TimeInterval)] = []
            var imageStart = totalStart
            for (n, seed) in seeds.enumerated() {
                try context.check()
                context.image = n
                if n > 0 {
                    // The text and the input image were paid for by the first.
                    context.timings = Timings()
                    imageStart = Date()
                }
                var latent = Latent.noise(space, height: height, width: width, seed: seed,
                                          variations: request.variations)
                if let z₀ { latent = Latent.start(image: z₀, noise: latent, sigma: plan.startSigma) }

                context.emit(.stage(.denoising, image: n))
                let denoisingStart = Date()
                do {
                    try chain.denoising.denoise(&latent, text: text, steps: steps, start: start,
                                                    references: references, lora: stack, context: context)
                } catch is Context.Sketched {
                    latent.values = context.estimate!
                }
                context.timings.denoising = Date().timeIntervalSince(denoisingStart)
                Arena.releaseToSystem()
                pending.append((latent, context.timings, Date().timeIntervalSince(imageStart)))
            }

            context.keptDenoiser = nil
            Arena.releaseToSystem()

            var renderResults: [Render] = []
            for (n, (latent, timings, elapsed)) in pending.enumerated() {
                try context.check()
                context.image = n
                context.timings = timings
                context.emit(.stage(.decoding, image: n))
                MemoryPlan.readForDecoding()   // after the DiT is gone: the decoders' bands derive from it
                let image = try chain.decoding.decode(latent, context: context)
                // An image's time: its own text, encoding and denoising, then its decoding.
                context.timings.total = elapsed + context.timings.decoding
                context.footprints.end = Arena.processFootprint()
                context.emit(.decoding(seconds: context.timings.decoding))
                let seed = seeds[n]

                let renderResult = Render(model: chain.denoising.model, prompt: request.prompt, seed: seed,
                                  steps: steps, loras: request.loras, loraSummary: stack?.summary, image: image,
                                  reproducible: reproducible,
                                  evaluations: sketch ?? plan.evaluations, sketch: sketch, spectral: plan.reduced,
                                  strength: request.image != nil ? request.strength : nil, startStep: plan.startStep,
                                  startSigma: plan.startSigma,
                                  timings: context.timings, footprints: context.footprints, tokens: text.tokens,
                                  detail: request.detail,
                                  variations: request.variations.filter { $0.effectiveStrength > 0 })
                context.emit(.image(index: n, renderResult))
                renderResults.append(renderResult)
            }
            return renderResults
        } }
    }

    /// An image through the chain's encoder — or its latent read back from the cache (`RenderCache`).
    private func encode(_ image: ImageRGB, with encoding: any ImageEncodingModule, context: Context) throws -> Latent {
        guard let cache = context.cache, let module = encoding as? any CacheIdentified else {
            return try encoding.encoder(image, context: context)
        }
        let key = RenderCache.imageKey(module, image: image, reproducible: context.reproducible)
        if let kept = cache.latent(key) { return kept }
        let latent = try encoding.encoder(image, context: context)
        cache.store(latent, key: key)
        return latent
    }
}
