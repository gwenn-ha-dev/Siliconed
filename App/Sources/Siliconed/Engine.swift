import Foundation

/// **The product, as a library: a prompt goes in, pixels come out.**
///
/// The engine knows no word of the image; this type is Siliconed's side of the seam. It holds what a render emits and
/// the queue renders go through (`Facade.swift`); a family's pipeline — text, [image encoding,] denoising,
/// decoding — is computed by the engine.
///
/// 1. **It prints nothing and writes nothing.** It renders pixels and a trace; the PNG, an index of the outputs and the
///    on-screen timings remain the caller's business.
/// 2. **Everything that varies from one request to the next is in `Request`**: two requests can differ within one process.
/// 3. **One render at a time, and it is enforced, not recommended** (`Engine.file`).
///
/// **The naming rule**: what the engine *emits* is nested in `Engine` (`Render`, `Event`, `Plan`, `Stage`, `Preview`,
/// `Progress`, `Timings`, `Footprints`); what the caller *builds* is at the top level (`Request`, `Model`, `ImageRGB`,
/// `Library`, `Cancellation`). The one error that leaves the engine is `EngineError` (`EngineError.swift`); inside the
/// package, each reader keeps its `Failure`, folded at the door.
package final class Engine: Sendable {

    // ── what comes out ────────────────────────────────────────────────────────────────────

    /// What each phase cost. `text`, `encoding`, `denoising` and `decoding` partition
    /// `total`; `tokenizer` and `encoder` are indented within `text` — the lesson learned on
    /// decompositions that partition nothing. `encoding`: the img2img image, 0 without one.
    package struct Timings: Sendable, Codable, Equatable {
        package var tokenizer = 0.0, encoder = 0.0, text = 0.0
        package var encoding = 0.0, denoising = 0.0, decoding = 0.0, total = 0.0
        package init() {}
    }

    /// **What a stage of a render held**, as the engine's stages told it (`SiliCore.Stages`): the engine's stages opened
    /// while it was the one under way, never a sample of the family's own. A family announces a stage before it opens
    /// its engine's stages. A stage whose result came from `RenderCache` has none.
    package struct Held: Sendable, Codable, Equatable {
        package var stage: Stage
        /// What the footprint held at its most above the process's floor while the stage's engine stages lived, and
        /// what their plans counted for it above that floor (P − floor: the arena and Core AI's shares), under which B:
        /// the largest of its engine stages.
        package var held = 0, counted = 0, budget = 0
        /// The process's footprint at its most while the stage lived.
        package var process = 0
        /// The ring's slots the largest of its plans read through; 0 when every group was kept.
        package var ring = 0
        /// The groups of weights of its largest engine stage, and those it kept in memory (from the start and in
        /// flight): the others were read from the disk at each use, through the ring.
        package var groups = 0, kept = 0
        /// What Core AI compiled for it, in seconds (functions its cache did not hold), and whether it compiles now.
        package var compilation = 0.0
        package var compiling = false
        /// The share of the stage's work the Neural Engine computed (as the family counts it); `nil` when it ran none.
        package var neuralEngine: Double?
        /// **Its block's cycles as they ran** (a DiT's evaluations, every image of a batch in turn: one block for the
        /// batch), in every render; empty for a stage without a block.
        package var cycles: [Cycle] = []
        package init(stage: Stage) { self.stage = stage }
    }

    /// **A cycle of a block, as it ran** (`SiliCore.Stages.Cycle`): one DiT evaluation. The Neural Engine's rows of those
    /// it could take, by its unit (0: the GPU alone); whether it visits the share of none (`Block.zero`: this Mac's κ
    /// unknown, or the share predicted the fastest); its seconds; and **its rates, in TFLOP/s**: the block's — what its
    /// calls executed over its own seconds, what its start loaded for a move left out —, and each unit's over its own
    /// calls (nil: not timed — the GPU alone, a block's last cycle).
    package struct Cycle: Sendable, Codable, Equatable {
        package var rows, shareable, unit: Int
        package var zero: Bool
        package var seconds: Double
        package var tflops: Double
        package var neuralEngine: Double?, gpu: Double?
        package init(rows: Int, shareable: Int, unit: Int, zero: Bool = false, seconds: Double, tflops: Double = 0,
                     neuralEngine: Double? = nil, gpu: Double? = nil) {
            (self.rows, self.shareable, self.unit, self.zero, self.seconds) = (rows, shareable, unit, zero, seconds)
            (self.tflops, self.neuralEngine, self.gpu) = (tflops, neuralEngine, gpu)
        }
    }

    /// **What a render held**, stage by stage, and what the Neural Engine took of it.
    package struct Footprints: Sendable, Codable, Equatable {
        /// In the order of the render's stages; a stage that opened no engine stage is absent.
        package var stages: [Held] = []
        /// **The Neural Engine's share** of the render's work, 0 to 1: each stage's share weighted by the seconds its
        /// engine stages lived. `nil` for a render whose stages ran none.
        package var neuralEngine: Double?
        package init() {}
    }

    /// An image and what it took to get it. **No file has been written.**
    ///
    /// In a batch, each image has its own `Render`; the text (and the input image) are paid for only
    /// once, so `timings.text` and `timings.encoding` hold only for the first, and the `total`s
    /// of a batch's images partition the batch's time.
    package struct Render: Sendable {
        /// What requested it — for the PNG metadata and an app's history.
        package let model: String
        package let prompt: String
        package let seed: UInt64
        package let steps: Int
        package let loras: [LoRAEntry]
        /// The image, planar `[3, height, width]`, fp32 around `[-1, 1]` (not clamped).
        package let image: ImageRGB
        /// The evaluations done.
        package let evaluations: Int
        /// The img2img strength; `nil` in txt2img. Stamped as the render hands the image over (`renderBatchOnQueue`).
        package var strength: Double? = nil
        /// The first step executed and its σ (0 and 1 in txt2img).
        package let startStep: Int
        package let startSigma: Float
        package let timings: Timings
        package var footprints: Footprints
        /// The prompt's own tokens, its template's left out.
        package let tokens: Int
        /// **More detail** (`Request.detail`); `normal` writes nothing in the metadata.
        package var detail: Detail = .normal
        /// **The update rule** (`Request.solver`); `euler` writes nothing in the metadata.
        package var solver: Solver = .euler
        /// **The variations its noise was turned by** (`Request.variations`), those of null
        /// strength left out; empty for an ordinary render.
        package var variations: [Variation] = []
        /// **The model's displayed name** (`ModelCard.name`): `Z-Image Turbo`, or an imported fine-tune's
        /// own. `nil`: the family identifier `model` stands for it.
        package var modelName: String? = nil
        /// Rendered with an imported DiT rather than the publisher's — then no Civitai page is claimed
        /// for the model (`Infotext.civitaiModelVersion`).
        package var isImportedModel = false
        /// Per LoRA of `loras`, in order: its name and the sha256 of its file (`LoRAOrigin(_:)`), stamped on each
        /// image as the render hands it over (`renderBatchOnQueue`); empty for a `Render` made otherwise (a test's).
        package var loraOrigins: [LoRAOrigin] = []

        /// **The generation metadata**, as `png()` writes it (`iTXt` chunks). ASCII keys, readable
        /// values; **no date** — two identical renders must remain two identical files.
        ///
        /// Enough to redo a **txt2img** on the same machine and the same version. **Not an img2img**: the source image is not included
        /// (the `image` key says so), and a LoRA appears only by its file name. `parameters` says the
        /// same render again for Civitai (`Infotext`), with the app's version: identical renders of
        /// one version stay identical files.
        package var metadata: [String: String] {
            var m = ["prompt": prompt, "seed": "\(seed)", "model": model, "steps": "\(steps)",
                     "format": "\(image.width)x\(image.height)",
                     "Software": "Siliconed"]
            if let strength {
                m["strength"] = String(format: "%.2f", strength)
                m["image"] = "source not included"
            }
            if detail != .normal { m["detail"] = detail.rawValue }
            if solver != .euler { m["solver"] = solver.rawValue }
            // `seed` stays the origin's: the variations turn its noise, in order (`Variation.text`).
            if !variations.isEmpty { m["variation"] = Variation.text(variations) }
            if !loras.isEmpty {
                m["lora"] = loras.map { "\((($0.path as NSString).lastPathComponent)):\(String(format: "%.2f", $0.strength))" }
                    .joined(separator: ",")
            }
            // Beside the keys above, which the app reads back (`Recipe`): Civitai reads this one only.
            m["parameters"] = parameters()
            return m
        }

        /// **The strength a render of `request` is recorded with**: the starting image's, `nil` without one — an edit's
        /// references are not a starting image.
        package static func strength(of request: Request) -> Double? { request.image != nil ? request.strength : nil }

        /// What a LoRA is known by outside the app: its name and the sha256 of the
        /// `.safetensors` it was imported from, `nil` when it is not known.
        package struct LoRAOrigin: Sendable, Equatable {
            package let name: String
            package let sha256: String?
            package init(name: String, sha256: String?) { self.name = name; self.sha256 = sha256 }
        }

        /// **The `parameters` chunk** — the same render in A1111's text, the format Civitai reads
        /// (`Infotext`). `version`: the app's, `Infotext.appVersion` by default.
        package func parameters(version: String? = Infotext.appVersion) -> String {
            let lora = loras.enumerated().map { i, entry in
                let origin = i < loraOrigins.count ? loraOrigins[i] : nil
                return Infotext.LoRA(name: origin?.name ?? (entry.path as NSString).lastPathComponent,
                                     strength: entry.strength, sha256: origin?.sha256)
            }
            return Infotext.text(prompt: prompt, steps: steps, sampler: solver, seed: seed,
                                 width: image.width, height: image.height, model: modelName ?? model,
                                 strength: strength, loras: lora,
                                 civitaiModelVersion: isImportedModel ? nil : Infotext.civitaiModelVersion(model: model),
                                 version: version)
        }

        /// The sRGB PNG, metadata included.
        package func png() throws(EngineError) -> Data { try image.png(metadata: metadata) }

        /// **`metadata` read back**: what a Siliconed PNG says about its render (`PNG.text`, then
        /// this). Pure: no library is consulted, so nothing here says whether the model or a LoRA is
        /// installed — that is the caller's to resolve, and to say.
        ///
        /// What it cannot give back, by construction of `metadata`:
        /// - **the model card**, only its family (`z-image`): an imported fine-tune writes the
        ///   family's name, not its own;
        /// - **a LoRA's path**, only its file name (`flat.safetensors`);
        /// - **an img2img's source image** (`image = source not included`), nor an edit's references:
        ///   an edit writes nothing that tells it from a txt2img.
        ///
        /// Each field is `nil` when its key is absent **or unreadable**; `unreadable` lists the keys
        /// that were there but did not parse, so the caller can say so instead of guessing.
        package struct Recipe: Sendable, Equatable {
            /// The family, as `Render.model` writes it (`z-image`, `qwen-image-2.1`…).
            package var model: String?
            package var prompt: String?
            package var seed: UInt64?
            package var steps: Int?
            package var width: Int?, height: Int?
            /// The stack in its order; `path` is **the file name only**, as written.
            package var loras: [LoRAEntry] = []
            /// The img2img strength; `nil` in txt2img.
            package var strength: Double?
            /// An img2img: the image it started from is not in the file.
            package var sourceImageMissing = false
            /// `Request.detail`; `normal` when the key is absent (every render before it).
            package var detail: Detail = .normal
            /// `Request.solver`; `euler` when the key is absent (every render before it).
            package var solver: Solver = .euler
            /// The variations its noise was turned by, in order (`Request.variations`); empty for an
            /// ordinary render.
            package var variations: [Variation] = []
            /// Keys present whose value did not parse, sorted.
            package var unreadable: [String] = []

            /// `nil` unless `Software` is `Siliconed`: another program's keys (ComfyUI's `prompt`
            /// is a JSON graph) are not interpreted.
            package init?(metadata m: [String: String]) {
                guard m["Software"] == "Siliconed" else { return nil }
                var bad: [String] = []
                func read<T>(_ key: String, _ parse: (String) -> T?) -> T? {
                    guard let raw = m[key] else { return nil }
                    if let v = parse(raw.trimmingCharacters(in: .whitespaces)) { return v }
                    bad.append(key)
                    return nil
                }
                model = read("model") { $0.isEmpty ? nil : $0 }
                prompt = m["prompt"]
                seed = read("seed") { UInt64($0) }
                steps = read("steps") { Int($0).flatMap { (1...Request.maximumSteps).contains($0) ? $0 : nil } }
                if let f = read("format", { Format.parse($0).flatMap { $0.width > 0 && $0.height > 0 ? $0 : nil } }) {
                    width = f.width; height = f.height
                }
                strength = read("strength") { Double($0).flatMap { $0 > 0 && $0 <= 1 ? $0 : nil } }
                sourceImageMissing = strength != nil || m["image"] != nil
                detail = read("detail") { Detail(rawValue: $0) } ?? .normal
                solver = read("solver") { Solver(rawValue: $0) } ?? .euler
                variations = read("variation") { Variation.parse($0) } ?? []
                if let raw = m["lora"] {
                    // `name:0.80,…`, split at the LAST colon: the strength is always written.
                    for entry in raw.split(separator: ",") {
                        let e = entry.trimmingCharacters(in: .whitespaces)
                        guard let colon = e.lastIndex(of: ":"), let s = Double(e[e.index(after: colon)...]),
                              s.isFinite, !e[..<colon].isEmpty else {
                            if !bad.contains("lora") { bad.append("lora") }
                            continue
                        }
                        loras.append(LoRAEntry(String(e[..<colon]), strength: s))
                    }
                }
                unreadable = bad.sorted()
            }
        }
    }

    /// **What is about to be paid for**, announced before any computation (`Event.start`).
    package struct Plan: Sendable {
        /// The denoiser's plan, for each image: evaluations (the last `total` of the `step`
        /// events), starting step and σ.
        package let denoising: DenoisingPlan
        /// The batch's seeds, in image order; just one outside a batch.
        package let seeds: [UInt64]
        /// The stages that will run, in order — `image` only in img2img. The text and the input
        /// image run only once per batch; denoising and decoding, per image.
        package let stages: [Stage]
    }

    /// A render's stages, as `Event.stage` announces them.
    package enum Stage: String, Sendable, Codable, CaseIterable {
        case text, image, denoising, decoding
    }

    /// **Progress, for whoever wants to display it.** The engine emits, it does not format.
    ///
    /// The order of a successful render: `start`, then `stage(.text)` … `text`, [`stage(.image)`
    /// … `encoding`], then for each image of the batch `stage(.denoising)` and its `step`s (and
    /// `preview`s if requested), then for each `stage(.decoding)`, `decoding`, `image`; an image lost
    /// ends in `unusable` instead (one whose latent is not finite, right after its steps, never decoded). A stage is
    /// announced before its sessions are planned: what a plan takes is its stage's. The `warning`s
    /// can fall anywhere. The closure is called **on the render thread**, synchronously: it must
    /// return quickly (the engine's process writes each event to the app's pipe there, `WorkerServer`).
    package enum Event: Sendable {
        case start(Plan)
        /// **The start of a stage** — emitted BEFORE it begins, so the bar moves during the text
        /// encoder and the decoder. `image`: the index in the batch (0 for the text and the input
        /// image, which run only once).
        case stage(Stage, image: Int)
        /// `tokens`: the prompt's own, its template's left out (`Render.tokens`).
        /// `cached`: the conditioning was read back from `RenderCache` — its seconds are the cache's, not
        /// the encoder's, and say nothing of the next prompt.
        case text(tokens: Int, tokenizer: Double, encoder: Double, cached: Bool = false)
        /// The img2img image is encoded and mixed with the noise.
        case encoding(seconds: Double)
        /// **One denoiser evaluation**, numbered the same way for every model: `index` from
        /// 1 to `total`, `total` = the evaluations actually paid for (`Plan.evaluations`).
        /// `sigma`: the σ evaluated. The grid is the one this step evaluated, in latent cells.
        case step(image: Int, index: Int, total: Int, sigma: Float, seconds: Double,
                 latentGridHeight: Int, latentGridWidth: Int)
        /// The x̂₀ prediction of the step that just finished, projected (`Request.previews`).
        case preview(Preview)
        case decoding(seconds: Double)
        /// **An image of the batch is ready** — the last event of each image.
        case image(index: Int, Render)
        /// **An image of the batch is lost** — the last event of that image, in place of `image`: a non-finite value
        /// in its latent (`stage` `.denoising`, said before any decoding is announced) or in its pixels (`.decoding`).
        /// A batch that loses them all ends in `EngineError.nonFiniteImage`: `.denoising` when no latent was finite,
        /// `.decoding` otherwise.
        case unusable(image: Int, stage: Stage)
        /// What the library would otherwise write to standard error: a fallback, a non-finite value…
        case warning(String)
        /// **What the stages held so far**, read in the engine's ledger: after each stage's start, each step and each
        /// result, so that a display follows the memory as it is taken.
        case held([Held])
        /// **The footprint's buckets since the last of these** (`MemorySeries`, a delta), sent with each `held`; the app
        /// keeps the series whole.
        case memory(MemorySeries)
    }

    /// **An honest progress fraction, and the time remaining**, drawn from the events alone.
    ///
    /// The unit is **the DiT evaluation**, which dominates everything: a render is worth
    /// `text + [image] + images × (evaluations + decoding)` units. The non-DiT stages weigh what
    /// they weighed in measured 512² renders, rounded up: text 0.5 evaluation (Z-Image 0.54 · Anima
    /// 0.22 · Krea 2 0.25), input image 0.15 (0.06 to 0.14), decoding 0.2 (0.05 to 0.17). The
    /// fraction never goes backwards and is 1 once the last image is delivered or lost.
    ///
    /// The time remaining comes **from the steps already done**, never from a constant: the speed
    /// depends on the machine, the heat and the page cache. The time remaining therefore exists only
    /// **after the first step**; decoding counts what it cost in this render, or 0.2 of a step before
    /// the first. An image lost at the DiT (`unusable` at `denoising`) is never decoded: its decoding counts as done.
    package struct Progress: Sendable {
        package static let textWeights = 0.5, imageWeights = 0.15, decodingWeights = 0.2

        private var total = 1.0, done = 0.0
        private var images = 0, evaluations = 0
        private var secondsSteps = 0.0, stepsDone = 0
        private var decoding: Double?, decodingsDone = 0

        package init() {}

        /// `[0, 1]`.
        package var fraction: Double { min(1, done / total) }

        /// In seconds; `nil` before the first step.
        package var estimatedRemaining: Double? {
            guard stepsDone > 0 else { return nil }
            let step = secondsSteps / Double(stepsDone)
            let remainingSteps = max(0, images * evaluations - stepsDone)
            let remainingDecodings = max(0, images - decodingsDone)
            return Double(remainingSteps) * step
                + Double(remainingDecodings) * (decoding ?? Progress.decodingWeights * step)
        }

        package mutating func receive(_ event: Event) {
            switch event {
            case .start(let plan):
                // A single `Progress` for an app's whole life: a render's time remaining is drawn
                // only from ITS steps, not from the previous render's (another model, another format).
                self = Progress()
                images = plan.seeds.count
                evaluations = plan.denoising.evaluations
                total = Progress.textWeights + (plan.stages.contains(.image) ? Progress.imageWeights : 0)
                    + Double(images) * (Double(evaluations) + Progress.decodingWeights)
            case .text: done += Progress.textWeights
            case .encoding: done += Progress.imageWeights
            case let .step(_, _, _, _, seconds, _, _):
                done += 1
                secondsSteps += seconds; stepsDone += 1
            case .decoding(let seconds):
                done += Progress.decodingWeights
                decoding = seconds; decodingsDone += 1
            case .unusable(_, .denoising):
                // Lost after its steps, never decoded: its decoding counts as done, so that the fraction still reaches 1
                // and the time remaining no longer counts it.
                done += Progress.decodingWeights
                decodingsDone += 1
            case .stage, .preview, .image, .unusable, .warning, .held, .memory: break
            }
        }
    }

    // ── the engine ─────────────────────────────────────────────────────────────────────────

    /// **The queue that enforces "one render at a time"**: two renders would not share a machine, they would fight over
    /// it, and the memory peak would double, which B forbids. Static: one queue for the whole process, however many
    /// `Engine`s are created. **QoS `.userInitiated`**: without it, a render launched from a queue of lower QoS would inherit
    /// it, and the system would treat it as background work.
    static let file = DispatchQueue(label: "siliconed.engine", qos: .userInitiated)

    package init() {}
}
