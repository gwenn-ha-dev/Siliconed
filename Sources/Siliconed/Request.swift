import Foundation

// **The request: what a render is asked for** — the prompt, the format, the seed, the LoRA stack
// (`LoRAEntry`), the images — whatever the model. The chain that runs it is in `Chain.swift`.

// ── the request ──────────────────────────────────────────────────────────────────────────────

/// A LoRA of the stack: a forged map (`ForgeLoRA`) and its strength.
///
/// **A strength is a `Double` everywhere in the API** (a LoRA's as well as img2img's); the LoRA
/// brings it down to `Float` at merge time, where it multiplies fp32 weights.
package struct LoRAEntry: Sendable, Hashable {
    package var path: String
    package var strength: Double
    package init(_ path: String, strength: Double = 1) { self.path = path; self.strength = strength }

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
/// specific to a model (Qwen-Image-2.1's spectral schedule, for instance) lives on its module.
///
/// **Building a request reads nothing** — neither the machine's profile nor a file: what depends
/// on them (`reproducible` left at `nil`) is resolved at render time. An app can therefore hold
/// one in its state before having chosen its library and loaded its profile.
package struct Request: Sendable, Equatable {
    package var prompt: String
    /// The image's format, in pixels. See `Format` for what is accepted. Changing it refits the
    /// input image — from the **already-fitted** image: to change format without losing edges,
    /// set the source image again (`image = …`) after the format.
    package var width: Int { didSet { fit() } }
    package var height: Int { didSet { fit() } }
    package var seed: UInt64
    /// `nil`: the denoiser's default (`ModelCard.defaultSteps`). At least 1, otherwise the render
    /// throws `EngineError.invalidSteps` before any computation.
    package var steps: Int?
    package var loras: [LoRAEntry]
    /// **Same seed, same bits**: the GPU/AMX split is not servo-controlled. See `Conductor.driven`.
    /// `nil`: the machine's setting (`EngineSettings.effective.frozenCut`, true by default), **read at
    /// render time**, not at construction.
    package var reproducible: Bool?
    /// **The img2img starting image.** Set at any size, it is **immediately** filled and
    /// center-cropped to the request's format (`ImageRGB.fitted`): a request never keeps a
    /// full-size photo (48 Mpx would weigh 576 MB of floats for the whole render). `nil`: txt2img
    /// — one way of doing things, not two APIs.
    package var image: ImageRGB? { didSet { fit() } }
    /// diffusers' `strength` (`Strength`), in `]0 ; 1]`. No effect without `image`.
    package var strength: Double
    /// **The editing reference images** ("change her jacket to red"): the denoiser re-reads them
    /// at each layer, and the generated image starts from noise — nothing to do with `image`,
    /// which is its starting point. Each keeps **its own** aspect ratio, at the size the denoiser
    /// gives it (`DenoisingModule.preparedReferences`): Qwen-Image-2.1 at ~1024² at multiples of 32
    /// (Pillow's Lanczos) — whatever the request's format.
    /// Qwen-Image-2.1's encoder also SEES them: the prompt names them `<image1>`, `<image2>`… in this
    /// order, and the first is the image being edited. Empty: no editing. Only the denoisers that
    /// announce it read them (`maxReferences`: 3 for Qwen-Image-2.1).
    package var references: [ImageRGB] = []
    /// **A preview at each step** (`Event.preview`, `Engine.Preview`). False by default: nothing
    /// is then computed or copied.
    package var previews: Bool
    /// **A sketch: stop after this many evaluations** and decode the denoiser's estimate of the
    /// final image, x̂₀ = x − σ·v, instead of the image. `nil`: the whole render. The estimate is free
    /// (the evaluation computed `v` anyway) and speaks early: on Z-Image at 512², after 3 of its
    /// former 7 evaluations (4 of 8 since 0.8.2) the pose, the clothes and the light are those of
    /// the final image. The
    /// evaluations done are **the first ones of the whole render, bit for bit** — a sketch is a
    /// prefix, not another render: the same request without `sketch` finishes the image it showed.
    /// Under 1, one evaluation; at or past the plan's count, the whole render (`Render.sketch` nil).
    package var sketch: Int?
    /// **More detail** (`Detail`): the denoiser is told, mid-trajectory, that a little less noise
    /// remains than really does, and leaves finer texture. Same evaluations, same cost; `normal` is
    /// the render without it, to the bit. The models whose `DenoisingModule.acceptsDetail` says so
    /// honour it — Z-Image and Qwen-Image-2.1, not Anima nor Krea 2.
    package var detail: Detail = .normal
    /// **The update rule of each step** (`Solver`): Euler, the one the Turbo models were distilled
    /// for, unless another is asked — the same steps, the same σ, another walk between them. Only the
    /// models whose `DenoisingModule.solvers` name it accept it; the app offers them in Developer Mode.
    package var solver: Solver = .euler
    /// **Variations** (`Variation`): the seed's starting noise turned towards other seeds', in order —
    /// cousins of the image `seed` gives, its composition kept in proportion to `strength`
    /// (`Variation.Amount`). Empty: the seed's noise, to the bit. Only the noise changes:
    /// the text, the images and the render cache's keys do not see it. In a batch, each seed is turned
    /// by the same variations.
    package var variations: [Variation] = []
    /// **The Neural Engine beside the GPU in the denoiser**, where this Mac's map turned it on (`gemm_ane`):
    /// the products it takes run in fp16, each weight column and activation row lifted into fp16's range and
    /// undone in fp32 (`NeuralEngine`) — less energy, and another image than fp32's, close to it (at 1024²,
    /// Z-Image's and Qwen-Image-2.1's within a tenth of a level on average, Krea 2's finest details moved).
    /// False: the denoiser in fp32 throughout, the image the checks compare with the reference. It can only
    /// turn the branch off: where the map left it off, true changes nothing. The encoders never take it.
    package var neuralEngine = true

    /// The most steps a request may ask for. The Turbo models are distilled for 6 to 8; the cap is
    /// there for what does not come from the stepper — a PNG's metadata, `silicontrol --steps`, the
    /// API — where 2·10⁹ steps would allocate the schedule (8 GB) before the first evaluation.
    package static let maximumSteps = 50

    /// What a request refuses, before any computation.
    package enum Failure: Error, CustomStringConvertible, Equatable {
        /// An empty or blank prompt: every model refuses it (Anima would have no token to
        /// encode; the others would render an image nothing asked for).
        case emptyPrompt
        /// A number of steps outside `1...maximumSteps`.
        case steps(Int)
        /// More references than the denoiser reads (0: it reads none).
        case references(Int, max: Int)
        /// A solver the model's sampler does not walk.
        case solver(Solver, model: String)
        /// More or Most detail asked of a model that does not offer it.
        case detail(model: String)
        /// A prompt longer than its encoder reads, counted without the template (`admitPrompt`).
        case promptTooLong(tokens: Int, max: Int)
        package var description: String {
            switch self {
            case let .references(n, max) where max == 0:
                return "\(n) reference image(s): this model does no reference editing (Qwen-Image-2.1 does)"
            case let .references(n, max):
                return "\(n) reference images: this model reads at most \(max)"
            case .emptyPrompt: return "empty prompt: there is nothing to render"
            case .steps(let n): return "\(n) steps: a render takes between 1 and \(Request.maximumSteps)"
            case let .solver(solver, model): return "\(model) does not offer the \(solver.rawValue) solver"
            case .detail(let model): return "\(model) does not offer Detail"
            case let .promptTooLong(tokens, max):
                return "the prompt is \(tokens) tokens long, this model reads at most \(max): shorten it"
            }
        }
    }

    /// **The longest prompt any family renders: 1 024 tokens** of the prompt alone, the template's
    /// left out — per tokenizer when a family has two (Anima). diffusers cuts Z-Image, Anima and Krea 2
    /// at 512 (`max_sequence_length`, a pipeline default, not a bound of the models: their DiTs and
    /// RoPE take longer texts) and does not bound Qwen-Image-2.1; ComfyUI bounds none of them, and the
    /// long prompts users share are written there. 1 024 is Qwen-Image's own cut; past 512 the models
    /// read texts longer than diffusers ever hands them.
    package static let maximumPromptTokens = 1024

    /// **No prompt is ever cut in silence**: past its bound a prompt is refused, so an image always
    /// answers the whole prompt. `tokens` and `max` count the prompt's own tokens, the template's left
    /// out: what the user can shorten.
    package static func admitPrompt(tokens: Int, max: Int) throws {
        guard tokens <= max else { throw Failure.promptTooLong(tokens: tokens, max: max) }
    }

    package init(_ prompt: String, width: Int, height: Int, seed: UInt64 = 42, steps: Int? = nil,
                loras: [LoRAEntry] = [], image: ImageRGB? = nil, strength: Double = Strength.defaultValue,
                previews: Bool = false, reproducible: Bool? = nil) {
        self.prompt = prompt; self.width = width; self.height = height; self.seed = seed
        self.steps = steps; self.loras = loras; self.image = image; self.strength = strength
        self.previews = previews; self.reproducible = reproducible
        fit()   // `didSet` does not run inside an `init`
    }

    /// The `resolution × resolution` square — the common case, written short.
    package init(_ prompt: String, resolution: Int = 1024, seed: UInt64 = 42, steps: Int? = nil,
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
