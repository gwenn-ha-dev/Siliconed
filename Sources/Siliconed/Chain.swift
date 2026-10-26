import Foundation
import os

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
// ported and verified here. The protocols are open to the package so a caller can compose its chains, not so
// it can write new links.
//
// What does not change, and must be read as a constraint: **each stage lives in its own scope**.
// A module builds its model, computes, and returns its memory on exit; the next stage begins only
// afterwards. That is what holds the 16 GB ceiling.

// ── the wires ──────────────────────────────────────────────────────────────────────────────

/// **What a text encoder produces, and therefore what a denoiser accepts.** Two formats are equal
/// only if they have the same name: the name describes the encoder, the layer taken and the shape.
package struct TextFormat: Hashable, Sendable, CustomStringConvertible {
    package let name: String
    /// **The text stage reads the reference images too** (`ImageTextModule`): the encoder sees them
    /// with the prompt, and the denoiser expects the slots they left in the sequence.
    package let readsImages: Bool
    package init(_ name: String, readsImages: Bool = false) { self.name = name; self.readsImages = readsImages }
    package var description: String { name }

    /// Qwen3-4B, chat template, second-to-last layer: `[tokens, 2560]`.
    package static let zImage = TextFormat("qwen3-4b · chat · penultimate layer")
    /// Qwen3-0.6B + T5 → Anima's conditioner: `[rows, 1024]`.
    package static let anima = TextFormat("qwen3-0.6b + t5 · conditioner anima")
    /// Qwen3-VL-4B, twelve layers taken, template stripped: `[tokens, 12 × 2560]`. The fusion and
    /// `txt_in` belong to the DiT (the LoRA touches them), hence to the denoiser.
    package static let krea2 = TextFormat("qwen3-vl-4b · 12 taps")
    /// Qwen3-VL-8B on the prompt **and the reference images** (raw template, `<imageN>` slots), the
    /// last layer before the final norm, system turn dropped: `[T, 4096]` and the `image_pad_mask` of
    /// its `T` rows (`Conditioning.imageSlots`). `txt_in` and the images' latents belong to the DiT.
    package static let qwenImage21 = TextFormat("qwen3-vl-8b · prompt + images · last layer before norm",
                                               readsImages: true)
    /// Qwen3.5-9B on the chat template (no system turn, the thinking block open), last hidden state after the final
    /// norm, every token kept: `[T, 4096]`. The two text blocks and the projection belong to the DiT.
    package static let mossImage = TextFormat("qwen3.5-9b · chat · last hidden state after norm")
}

/// **A latent's space**: the VAE that defined it. The same number of channels does not mean the
/// same space — Flux and Qwen-Image both have sixteen channels and do not decode each other.
package struct LatentSpace: Hashable, Sendable, CustomStringConvertible {
    package let name: String
    package let channels: Int
    /// Pixels per latent cell, along one side.
    package let factor: Int
    package var description: String { name }

    package static let flux = LatentSpace(name: "VAE Flux", channels: 16, factor: 8)
    package static let qwenImage = LatentSpace(name: "VAE Qwen-Image", channels: 16, factor: 8)
    /// Qwen-Image-2.1's VAE (`AutoencoderKLQwenImage21`): 64 channels, 16 pixels per cell, **no
    /// packing** — planar `[64, H/16, W/16]`, normalized per channel `(z − mean_c) / std_c`; the DiT's
    /// tokens are its transpose `[h·w, 64]` (`QwenImage21VAE.pack` / `unpack`). The VAE itself reads and
    /// writes RGBA (`QwenImage21EncodingModule`, `QwenImage21DecodingModule`).
    package static let qwenImage21 = LatentSpace(name: "VAE Qwen-Image-2.1", channels: 64, factor: 16)
    /// FLUX.2's VAE (`AutoencoderKLFlux2`) as MOSS-Image's DiT sees it: the 32 channels patched 2×2 into **128**,
    /// 16 pixels per cell, planar `[128, H/16, W/16]`; the pipeline de-normalizes it with the VAE's batch-norm
    /// statistics and un-patchifies it before the decoder (`Flux2VAE`). Noise as `torch.randn` draws it.
    package static let flux2 = LatentSpace(name: "VAE FLUX.2", channels: 128, factor: 16)
}

/// The encoded text, ready for a denoiser of the same format.
package struct Conditioning: Sendable {
    package let format: TextFormat
    /// `values` is `[rows, width]`, row by row.
    package let rows: Int
    package let width: Int
    package let values: [Float]
    /// The prompt's tokens, for display — not always `rows` (Anima).
    package let tokens: Int
    /// **Which rows stand for an image** (`image_pad_mask`), one entry per row — empty for a format
    /// that does not read images (`TextFormat.readsImages`).
    package let imageSlots: [Bool]

    /// An already-encoded text — a golden tensor's, so a check can drive one of the product's
    /// denoisers without going back through the encoder.
    package init(format: TextFormat, rows: Int, width: Int, values: [Float], tokens: Int, imageSlots: [Bool] = []) {
        self.format = format; self.rows = rows; self.width = width; self.values = values
        self.tokens = tokens; self.imageSlots = imageSlots
    }
}

/// A planar latent `[channels, height, width]`, and the space that gives it meaning.
package struct Latent: Sendable {
    package let space: LatentSpace
    package let height: Int
    package let width: Int
    package var values: [Float]

    package init(space: LatentSpace, height: Int, width: Int, values: [Float]) {
        self.space = space; self.height = height; self.width = width; self.values = values
    }

    /// The starting noise. `Noise` does not reproduce `torch.randn`: a seed gives an image of the
    /// model, not diffusers'. It is drawn in planar order: a square therefore yields the same bits
    /// as before rectangles arrived. **Qwen-Image-2.1 is the exception**: its noise is
    /// `torch.randn((1, 1, 64, h, w))` to the bit (`TorchNoise`), so a seed gives diffusers' image.
    package static func noise(_ space: LatentSpace, height: Int, width: Int, seed: UInt64) -> Latent {
        var values = [Float](repeating: 0, count: space.channels * height * width)
        if space == .qwenImage21 || space == .flux2 {
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
///
/// **An image with transparency carries it twice**: `pixels` is always the image laid over white
/// (what every model but one reads), and `rgba` the source's own bytes, unpremultiplied, alpha
/// included — what Qwen-Image-2.1's edit reads, as diffusers hands its references in RGBA
/// (`QwenImage21Pipeline.preparedReference`). An opaque image has no `rgba`: nothing changes for it.
package struct ImageRGB: Sendable, Equatable {
    package let pixels: [Float]
    package let height: Int
    package let width: Int
    /// `[height, width, 4]` interleaved 8-bit RGBA, **straight** (not premultiplied: a transparent
    /// pixel keeps its colour), when some pixel's alpha is under 255; `nil` for an opaque image.
    package let rgba: [UInt8]?

    package init(pixels: [Float], height: Int, width: Int) {
        self.init(pixels: pixels, height: height, width: width, rgba: nil)
    }

    package init(pixels: [Float], height: Int, width: Int, rgba: [UInt8]?) {
        precondition(height > 0 && width > 0, "ImageRGB: \(width)×\(height), an empty image")
        precondition(pixels.count == 3 * height * width,
                     "ImageRGB: \(pixels.count) values for 3 × \(height) × \(width)")
        precondition(rgba.map { $0.count == 4 * height * width } ?? true, "ImageRGB: RGBA of another size")
        self.pixels = pixels; self.height = height; self.width = width; self.rgba = rgba
    }
}

/// What a denoiser announces before starting: where it starts, how many evaluations, of which how
/// many reduced.
package struct DenoisingPlan: Sendable {
    /// The first step executed of the full schedule: 0 in txt2img, `t_start` in img2img.
    package let startStep: Int
    /// The starting σ, `σ[startStep]`: 1 in txt2img.
    package let startSigma: Float
    /// The evaluations actually paid for, from the starting step to the end.
    package let evaluations: Int
    /// Half-size evaluations (Qwen-Image-2.1's spectral schedule); 0 elsewhere.
    package let reduced: Int
    /// The latent grid of the reduced evaluations.
    package let reducedHeight: Int
    package let reducedWidth: Int

    /// The same plan at full size from its first evaluation: an edit's (no spectral schedule).
    package var unreduced: DenoisingPlan {
        DenoisingPlan(startStep: startStep, startSigma: startSigma, evaluations: evaluations,
                      reduced: 0, reducedHeight: reducedHeight, reducedWidth: reducedWidth)
    }

    /// **The same plan, stopped as a sketch** (`Request.sketch`): its first `n` evaluations — what an
    /// app estimates a sketch's duration from. `n` at or past `evaluations`: the plan itself.
    package func sketched(_ n: Int) -> DenoisingPlan {
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
///     Z-Image, 7 (before 0.8.2)   σ after:  .947  .882  .800  .692 …      stop at 3 → 21 dB, pose,
///                                                                         clothes and light final
///     Z-Image, 8 (since 0.8.2)    σ after:  .955  .900  .833  .750 …      stop at 4 (rule, not re-measured)
///     Qwen-Image-2.1, 6           σ after:  .963  .923  .837  .632 …      stop at 4 → 20 dB,
///                                                                         composition, soft
///
/// A step count is no rule — Qwen's schedule keeps σ high longer, and at 3 of 6 its estimate is a
/// blur (17 dB); σ is. (Qwen's line was measured on the 6-step third-party turbo the family carried
/// before the official Turbo; the Turbo's 8 fixed steps reach 0.8 only after the 6th evaluation,
/// σ 0.705.) Under a LoRA, Z-Image's pose still moved between 3 and 4: a sketch shows
/// where the image goes, the finished render is the judge.
///
/// **Qwen-Image-2.1 does not sketch** (`DenoisingModule.sketches`, 2026-10-06): its « soft » at 4 is,
/// seen through the app, blurred and grainy, and still at 5 (22 dB) — it removes the noise in its
/// last two evaluations. A sketch of it saves 15 s of 51 for an image no one can judge: its cells
/// are finished images. Re-judged on the Turbo's 8 steps (512², seeds 42, 7, 1000, against the
/// finished image reduced ×4): after 6 evaluations a blur (24.5–26 dB), after 7 nearly finished
/// (30–32 dB) for 3 s saved at 512², 14 s at 1024² — the threshold's stop, and not worth a sketch.
///
/// **0.5 since 0.12, 0.8 before** (the user, 18/10/2026): at 0.8 a LoRA's style had not landed — Anima
/// under inCase, 4 of 8, showed the strengths 0.6 and 1 alike, the fine line still to come. 0.5 costs
/// 6 of 8 on Z-Image and Anima; the panel's stepper still stops anywhere.
package enum Sketch {
    package static let threshold: Float = 0.5

    /// The evaluations a sketch of `sigmas` (a full schedule, ending in 0, one evaluation per step)
    /// takes. Never fewer than 1, never more than the schedule's.
    package static func evaluations(sigmas: [Float]) -> Int {
        for i in 0..<max(0, sigmas.count - 1) where sigmas[i + 1] <= threshold { return i + 1 }
        return max(1, sigmas.count - 1)
    }
}

package extension Model {
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
package final class Context {
    package let reproducible: Bool
    /// **The denoiser's products may take the Neural Engine** (`Request.neuralEngine`), where this Mac's map
    /// turned it on: resolved at render time. The encoders never do.
    package var neuralEngine = EngineSettings.effective.gemmANE
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
    /// `Detail.scale` of the render's size: the bell's amount shrinks as the image grows.
    package var detailScale: Double = 1
    /// **The update rule** (`Request.solver`), and the seed of the image being denoised (the ancestral
    /// rule's noise).
    package var solver: Solver = .euler
    package var solverSeed: UInt64 = 0
    /// **Whose σ** (`Request.schedule`): the official recipe unless asked otherwise.
    package var schedule: SigmaSchedule = .official

    /// The σs the DiT is told, for a full schedule of which steps `start…` run; no bell, the σ themselves.
    package func told(_ sigmas: [Float], start: Int, bell: Detail.Bell? = nil) -> [Float] {
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
package protocol TextModule: Sendable {
    /// What an app displays on the module's box ("Qwen3-VL-4B").
    var name: String { get }
    var output: TextFormat { get }
    func encoder(_ prompt: String, context: Context) throws -> Conditioning
}

/// **A text stage that sees the reference images** (Qwen-Image-2.1's Qwen3-VL): the prompt and the
/// references go in together — already at the denoiser's size (`DenoisingModule.preparedReferences`),
/// the very copy the image encoder then reads. Without references, it is an ordinary text stage.
package protocol ImageTextModule: TextModule {
    func encoder(_ prompt: String, images: [ImageRGB], context: Context) throws -> Conditioning
}

package extension ImageTextModule {
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
package protocol DenoisingModule: Sendable {
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
    /// **The σ schedules it walks** (`SigmaSchedule`). By default the official one alone; the chain
    /// refuses another before any computation.
    var schedules: [SigmaSchedule] { get }
    /// The full schedule under `schedule`. By default `sigmas(steps:)`, the official one.
    func sigmas(steps: Int, schedule: SigmaSchedule) -> [Float]
    /// `height`, `width`: the latent grid, not the pixels. `withLoRA`: the request carries a
    /// stack — Qwen-Image-2.1 turns its spectral schedule off there (`QwenImage21DenoisingModule.plan`).
    func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan
    func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                   context: Context) throws
    /// **How many reference images it can read** (editing): 0 for a denoiser
    /// without. A reference is not a starting point (img2img): it enters the sequence, next to the
    /// generated image, and the denoiser re-reads it at each layer.
    var maxReferences: Int { get }
    /// Denoising with references already encoded in its space. By default: none.
    func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int,
                   references: [Latent], lora: LoRA?, context: Context) throws
    /// **The reference images as this denoiser reads them**, prepared once for the whole render: the
    /// text stage (if it reads images) and the image encoder receive this same copy. By default,
    /// as given (a denoiser without references never receives any).
    func preparedReferences(_ images: [ImageRGB]) throws -> [ImageRGB]
    /// **The format of an edit whose request names none**, from its first reference's size. By
    /// default the reference's own size under 1024² (its ratio kept), raised to the floor and kept
    /// under the area cap.
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
    /// The update rules its sampler walks (`Solver`). By default Euler alone: the chain refuses
    /// another before any computation rather than render Euler under another name.
    var solvers: [Solver] { get }
    /// Whether it honours `Request.detail`. By default yes; a model on which the lie about σ changes
    /// nothing visible (Anima) says no, and the chain refuses More and Most rather than offer
    /// a choice that does nothing.
    var acceptsDetail: Bool { get }
}

/// The image encoding stage (img2img): an image in the render's format goes in, a latent of its
/// space comes out — normalized as the denoiser expects. It returns its memory on exit.
package protocol ImageEncodingModule: Sendable {
    var name: String { get }
    var space: LatentSpace { get }
    func encoder(_ image: ImageRGB, context: Context) throws -> Latent
}

/// The decoding stage: a latent of its space goes in, an image comes out.
package protocol DecodingModule: Sendable {
    var name: String { get }
    var space: LatentSpace { get }
    func decode(_ latent: Latent, context: Context) throws -> ImageRGB
}

package extension DenoisingModule {
    /// The plan of a request without a LoRA.
    func plan(height: Int, width: Int, steps: Int, start: Int) -> DenoisingPlan {
        plan(height: height, width: width, steps: steps, start: start, withLoRA: false)
    }
    /// The plan under a schedule: another than the official turns Qwen-Image-2.1's half-size steps
    /// off, like a LoRA — they were judged on its own σ only.
    func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool,
              schedule: SigmaSchedule) -> DenoisingPlan {
        plan(height: height, width: width, steps: steps, start: start, withLoRA: withLoRA || schedule != .official)
    }
    var schedules: [SigmaSchedule] { [.official] }
    func sigmas(steps: Int, schedule: SigmaSchedule) -> [Float] { sigmas(steps: steps) }
    var maxReferences: Int { 0 }
    func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int,
                   references: [Latent], lora: LoRA?, context: Context) throws {
        guard references.isEmpty else { throw Request.Failure.references(references.count, max: maxReferences) }
        try denoise(&latent, text: text, steps: steps, start: start, lora: lora, context: context)
    }
    func preparedReferences(_ images: [ImageRGB]) throws -> [ImageRGB] { images }
    func editFormat(referenceWidth: Int, referenceHeight: Int) -> (width: Int, height: Int) {
        // The area brought under 1024² at the reference's ratio, each side floored to the multiple.
        let m = Format.multiple
        let scale = min(1, (Double(1024 * 1024) / Double(referenceWidth * referenceHeight)).squareRoot())
        var (w, h) = (max(Format.minimumSide, Int(Double(referenceWidth) * scale) / m * m),
                      max(Format.minimumSide, Int(Double(referenceHeight) * scale) / m * m))
        // Raising the short side to the floor can push a long strip past the edit's area (10:1 gave
        // 3232×512): the long side gives way, as in Qwen-Image-2.1's rule.
        if w * h > Format.editSurface {
            if w >= h { w = Format.editSurface / h / Format.multiple * Format.multiple }
            else { h = Format.editSurface / w / Format.multiple * Format.multiple }
        }
        return (w, h)
    }
    func check(steps: Int, startImage: Bool) throws {}
    var acceptsStartImage: Bool { true }
    var sketches: Bool { true }
    var solvers: [Solver] { [.euler] }
    var acceptsDetail: Bool { true }
    /// The evaluations a sketch takes on the latent grid `height × width`: where σ reaches the
    /// threshold, or all of them for a denoiser that does not sketch.
    func sketchEvaluations(steps: Int, height: Int, width: Int) -> Int {
        let all = plan(height: height, width: width, steps: steps, start: 0).evaluations
        guard sketches else { return all }
        return min(all, Sketch.evaluations(sigmas: sigmas(steps: steps)))
    }
}

/// A module without a name of its own carries its type's: a hand-composed chain can still be
/// drawn.
package extension TextModule { var name: String { String(describing: Self.self) } }
package extension DenoisingModule { var name: String { String(describing: Self.self) } }
package extension ImageEncodingModule { var name: String { String(describing: Self.self) } }
package extension DecodingModule { var name: String { String(describing: Self.self) } }

// ── the chain ───────────────────────────────────────────────────────────────────────────────

/// **Three modules wired, plus the image encoder if there is one — and verified wirable.**
///
/// The encoder is optional **in the type** (a chain without it only does txt2img) and present in
/// every `Model` of the catalog. Its wire is the same as the decoder's: it must speak the
/// denoiser's space.
package struct Chain: Sendable {
    package let text: any TextModule
    package let denoising: any DenoisingModule
    package let decoding: any DecodingModule
    package let encoding: (any ImageEncodingModule)?

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

    package init(text: any TextModule, denoising: any DenoisingModule,
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
    package struct Entry: Sendable, Hashable {
        package enum Kind: Sendable, Hashable {
            /// The prompt (`Request.prompt`).
            case text
            /// The starting image (`Request.image`, `Request.strength`).
            case image
            /// An editing reference image (`Request.references`) — as many as
            /// `DenoisingModule.maxReferences`.
            case reference
        }
        package let kind: Kind
        /// Without it, the render is refused; optional, it only changes what it does.
        package let isRequired: Bool
        /// The module that receives it ("Qwen3-VL-4B", "VAE Flux").
        package let module: String
    }

    /// **What the chain accepts, in signal order** — what an app draws as inputs instead of a
    /// fixed form. The text always, the image if there is an image
    /// encoder and the denoiser starts from one, the references if it reads them.
    package var entries: [Entry] {
        [Entry(kind: .text, isRequired: true, module: text.name)]
            + (encoding.flatMap { denoising.acceptsStartImage
                                     ? [Entry(kind: .image, isRequired: false, module: $0.name)] : nil } ?? [])
            + (encoding != nil && denoising.maxReferences > 0
                ? [Entry(kind: .reference, isRequired: false, module: denoising.name)] : [])
    }
}
