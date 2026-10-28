import Foundation

/// **The product, as a library: a prompt goes in, pixels come out.**
///
/// This file holds the engine and what it renders; the chain itself — modules, wires, `render` —
/// is in `Chain.swift`, `EngineRender.swift` and `Modules.swift`. It first pulled the chain out of the CLI's
/// render command, where it lived mixed in with `print` calls, to make it an object any caller can
/// hold: the CLI, the app (`SiliconedApp`), a server if one is ever needed.
///
/// Three rules govern it, and each answers a flaw the CLI path had by nature.
///
/// 1. **The engine prints nothing and writes nothing.** It renders pixels and a trace; the PNG,
///    an index of the outputs and the on-screen timings remain the caller's business. A library that
///    imposes a working directory cannot be embedded.
/// 2. **Everything that varies from one request to the next is in `Request`**, or on the module it
///    concerns. `EngineSettings.effective` stays what the *machine* can do (co-execution, decoder slices);
///    the prompt, the seed, the resolution, the spectral schedule and reproducibility are call
///    parameters. Without that, two requests cannot differ within one process.
/// 3. **One render at a time, and it is enforced, not recommended.** See `Engine.render`.
///
/// What this file does not change, and must be read as a constraint: **the scope of the arenas
/// per phase**. Text, image (img2img), denoising and decoding each live in their own block, and
/// each phase returns its memory before the next one asks for its own. That is what holds the
/// 16 GB ceiling, and why `render` rebuilds everything on every call instead of keeping a
/// warm DiT: the cost is building each stage and re-reading its weights **on every render** (and
/// on every image of a batch), the benefit is never swapping.
///
/// **The naming rule**: what the engine *emits* is nested in `Engine` (`Render`, `Event`,
/// `Plan`, `Stage`, `Preview`, `Progress`, `Timings`, `Footprints`); what the caller *builds or
/// wires* is at the top level (`Request`, `Model`, `Chain`, the modules and their wires,
/// `ImageRGB`, `Library`, `Cancellation`). The one error that leaves the engine is `EngineError` (`Errors.swift`);
/// inside the package, each reader keeps its `Failure`, folded at the door.
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

    /// The footprints at the checkpoints. What is measured here cannot be deduced from any total:
    /// a measurement found 520 MB of residue after the text phase because someone sampled between the two,
    /// not because a calculation predicted it.
    package struct Footprints: Sendable, Codable, Equatable {
        package var start = 0, tokenizer = 0, encoder = 0, text = 0
        /// The image encoder's peak, taken before it returns its memory; 0 without an image.
        package var encoding = 0
        package var denoising = 0, end = 0
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
        /// What the LoRA stack did, in one line — `nil` when there was none.
        package let loraSummary: String?
        /// The image, planar `[3, height, width]`, fp32 around `[-1, 1]` (not clamped).
        package let image: ImageRGB
        /// Was the GPU/AMX split frozen (`Request.reproducible`, resolved at render time)?
        package let reproducible: Bool
        /// The evaluations done — for a sketch, those it stopped after.
        package let evaluations: Int
        /// **A sketch** (`Request.sketch`): the evaluations it stopped after, the image being the
        /// estimate x̂₀ there. `nil`: the whole render.
        package let sketch: Int?
        /// The `k` actually applied, which is not the settings' one (a profile is not allowed to
        /// write it) and which a file name must carry on pain of lying.
        package let spectral: Int
        /// The img2img strength; `nil` in txt2img.
        package let strength: Double?
        /// The first step executed and its σ (0 and 1 in txt2img).
        package let startStep: Int
        package let startSigma: Float
        package let timings: Timings
        package let footprints: Footprints
        package let tokens: Int
        /// **More detail** (`Request.detail`); `normal` writes nothing in the metadata.
        package var detail: Detail = .normal
        /// **The update rule** (`Request.solver`); `euler` writes nothing in the metadata.
        package var solver: Solver = .euler
        /// **Whose σ** (`Request.schedule`); `official` writes nothing in the metadata.
        /// **The variations its noise was turned by** (`Request.variations`), those of null
        /// strength left out; empty for an ordinary render.
        package var variations: [Variation] = []
        /// **The model's displayed name** (`ModelCard.name`): `Z-Image Turbo`, or an imported fine-tune's
        /// own. `nil`: the family identifier `model` stands for it.
        package var modelName: String? = nil
        /// Rendered with an imported DiT rather than the publisher's — then no Civitai page is claimed
        /// for the model (`Infotext.civitaiModelVersion`).
        package var isImportedModel = false
        /// Per LoRA of `loras`, in order: its name and the sha256 of its published file
        /// (`LoRA.origins`); empty when the engine did not open the stack (a test's `Render`).
        package var loraOrigins: [LoRAOrigin] = []
        /// **The render's memory: its computed need against its own measured peak**, phase by phase
        /// (`MemoryCheck`); a phase above its need is a defect of the engine (`MemoryCheck.exceeded`).
        /// `nil` when no need was computed (a model whose maps are not installed).
        package var memory: MemoryCheck? = nil
        /// **Its denoiser's products could take the Neural Engine** (`Request.neuralEngine`, resolved against
        /// the map): fp16 products, another image than fp32's. False writes nothing in the metadata.
        package var neuralEngine = false
        /// **What the denoising computed, by class and route** (`ProductLedger`): every product by a weight and every
        /// attention of this image's evaluations, with the executions and wall seconds of each — what the capacity
        /// map's prediction is compared with (`RenderLoop`). Empty for a render that was not recorded (a test's).
        package var classes: [ProductLedger.Tally] = []

        /// **The generation metadata**, as `png()` writes it (`iTXt` chunks). ASCII keys, readable
        /// values; **no date** — two identical renders must remain two identical files.
        ///
        /// Enough to redo a **txt2img** on the same machine and the same version (`reproducible`
        /// says whether the bits will match). **Not an img2img**: the source image is not included
        /// (the `image` key says so), and a LoRA appears only by its file name. `parameters` says the
        /// same render again for Civitai (`Infotext`), with the app's version: identical renders of
        /// one version stay identical files.
        package var metadata: [String: String] {
            var m = ["prompt": prompt, "seed": "\(seed)", "model": model, "steps": "\(steps)",
                     "format": "\(image.width)x\(image.height)", "reproducible": "\(reproducible)",
                     "Software": "Siliconed"]
            if let strength {
                m["strength"] = String(format: "%.2f", strength)
                m["image"] = "source not included"
            }
            if let sketch { m["sketch"] = "\(sketch)" }
            if detail != .normal { m["detail"] = detail.rawValue }
            if solver != .euler { m["solver"] = solver.rawValue }
            if neuralEngine { m["neural-engine"] = "true" }
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

        /// What a LoRA is known by outside the app: its name (the map's `nom`) and the sha256 of the
        /// `.safetensors` it was imported from, `nil` for a map forged before that was kept.
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
        /// - **a LoRA's path**, only its file name (`flat.lora.silicon`);
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
            package var reproducible: Bool?
            /// `Request.detail`; `normal` when the key is absent (every render before it).
            package var detail: Detail = .normal
            /// `Request.solver`; `euler` when the key is absent (every render before it).
            package var solver: Solver = .euler
            /// `Request.neuralEngine`; false when the key is absent (every render before it, and every one
            /// computed in fp32).
            package var neuralEngine = false
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
                reproducible = read("reproducible") { Bool($0) }
                detail = read("detail") { Detail(rawValue: $0) } ?? .normal
                solver = read("solver") { Solver(rawValue: $0) } ?? .euler
                neuralEngine = read("neural-engine") { Bool($0) } ?? false
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
        /// events), reduced ones (the spectral), starting step and σ. Grids in latent cells.
        package let denoising: DenoisingPlan
        /// The batch's seeds, in image order; just one outside a batch.
        package let seeds: [UInt64]
        /// The stages that will run, in order — `image` only in img2img. The text and the input
        /// image run only once per batch; denoising and decoding, per image.
        package let stages: [Stage]
    }

    /// A render's stages, as `Event.stage` announces them.
    package enum Stage: String, Sendable, CaseIterable {
        case text, image, denoising, decoding
    }

    /// **Progress, for whoever wants to display it.** The engine emits, it does not format.
    ///
    /// The order of a successful render: `start`, then `stage(.text)` … `text`, [`stage(.image)`
    /// … `encoding`], then for each image of the batch `stage(.denoising)`, the `step` (and the
    /// `preview` if requested), `stage(.decoding)`, `decoding`, `image`. The `warning`s can
    /// fall anywhere. The closure is called **on the render thread**, synchronously: it must
    /// return quickly (a UI does its `MainActor` hop there, or goes through the `async` stream).
    package enum Event: Sendable {
        case start(Plan)
        /// **The start of a stage** — emitted BEFORE it begins, so the bar moves during the text
        /// encoder and the decoder. `image`: the index in the batch (0 for the text and the input
        /// image, which run only once).
        case stage(Stage, image: Int)
        /// `cached`: the conditioning was read back from `RenderCache` — its seconds are the cache's, not
        /// the encoder's, and say nothing of the next prompt.
        case text(tokens: Int, tokenizer: Double, encoder: Double, cached: Bool = false)
        /// The img2img image is encoded and mixed with the noise.
        case encoding(seconds: Double)
        /// **One denoiser evaluation**, numbered the same way for every model: `index` from
        /// 1 to `total`, `total` = the evaluations actually paid for (`Plan.evaluations`).
        /// `sigma`: the σ evaluated. The grid is the one this step
        /// evaluated — smaller than the render under the spectral schedule.
        case step(image: Int, index: Int, total: Int, sigma: Float, seconds: Double,
                 latentGridHeight: Int, latentGridWidth: Int)
        /// The x̂₀ prediction of the step that just finished, projected (`Request.previews`).
        case preview(Preview)
        case decoding(seconds: Double)
        /// **An image of the batch is ready** — the last event of each image.
        case image(index: Int, Render)
        /// What the library used to write to standard error: a fallback, a non-finite value…
        case warning(String)
    }

    /// **An honest progress fraction, and the time remaining**, drawn from the events alone.
    ///
    /// The unit is **the DiT evaluation**, which dominates everything: a render is worth
    /// `text + [image] + images × (evaluations + decoding)` units. The non-DiT stages weigh what
    /// they weighed in measured 512² renders, rounded up: text 0.5 evaluation (Z-Image 0.54 · Anima
    /// 0.22 · Krea 2 0.25), input image 0.15 (0.06 to 0.14), decoding 0.2 (0.05 to 0.17). The
    /// fraction never goes backwards and is 1 at the last `image`.
    ///
    /// The time remaining comes **from the steps already done**, never from a constant: the speed
    /// depends on the machine, the heat and the page cache. Two speeds, because two kinds of step:
    /// the half-size evaluations of the spectral schedule (the first ones; ~0.35 of a full step on
    /// Z-Image 1024² when Z-Image still had them) and the full ones. A common average, drawn from the reduced steps,
    /// promised a render twice too short, then went backwards at every step. The time remaining
    /// therefore exists only **after the first full step**; decoding counts what it cost in this
    /// render, or 0.2 of a full step before the first.
    package struct Progress: Sendable {
        package static let textWeights = 0.5, imageWeights = 0.15, decodingWeights = 0.2

        private var total = 1.0, done = 0.0
        private var images = 0, evaluations = 0, reduced = 0
        private var secondsFull = 0.0, fullCount = 0
        private var secondsReduced = 0.0, reducedDone = 0
        private var decoding: Double?, decodingsDone = 0

        package init() {}

        /// `[0, 1]`.
        package var fraction: Double { min(1, done / total) }

        /// In seconds; `nil` before the first full step.
        package var estimatedRemaining: Double? {
            guard fullCount > 0 else { return nil }
            let full = secondsFull / Double(fullCount)
            let reducedAverage = reducedDone > 0 ? secondsReduced / Double(reducedDone) : full
            let remainingFull = max(0, images * (evaluations - reduced) - fullCount)
            let remainingReduced = max(0, images * reduced - reducedDone)
            let remainingDecodings = max(0, images - decodingsDone)
            return Double(remainingFull) * full + Double(remainingReduced) * reducedAverage
                + Double(remainingDecodings) * (decoding ?? Progress.decodingWeights * full)
        }

        package mutating func receive(_ event: Event) {
            switch event {
            case .start(let plan):
                // A single `Progress` for an app's whole life: a render's time remaining is drawn
                // only from ITS steps, not from the previous render's (another model, another format).
                self = Progress()
                images = plan.seeds.count
                evaluations = plan.denoising.evaluations
                reduced = min(plan.denoising.reduced, evaluations)
                total = Progress.textWeights + (plan.stages.contains(.image) ? Progress.imageWeights : 0)
                    + Double(images) * (Double(evaluations) + Progress.decodingWeights)
            case .text: done += Progress.textWeights
            case .encoding: done += Progress.imageWeights
            case let .step(_, index, _, _, seconds, _, _):
                done += 1
                if index <= reduced { secondsReduced += seconds; reducedDone += 1 }
                else { secondsFull += seconds; fullCount += 1 }
            case .decoding(let seconds):
                done += Progress.decodingWeights
                decoding = seconds; decodingsDone += 1
            case .stage, .preview, .image, .warning: break
            }
        }
    }

    // ── the engine ─────────────────────────────────────────────────────────────────────────

    /// **The queue that enforces "one render at a time".**
    ///
    /// Nothing in this engine is reentrant, and that is no oversight: `GEMM` memoizes its
    /// `MTLBuffer`s and its MPS operators in bare dictionaries, `Conductor` holds raw pointers in
    /// instance state and a single thread, `Block` reuses its arena's slices. Two simultaneous
    /// renders in one process would not crash: they would silently corrupt each other, which is
    /// worse.
    ///
    /// **And it is also the right physical answer**: there is one GPU, two AMX blocks and 16 GB.
    /// Two concurrent renders would not share a machine, they would fight over it — and the memory
    /// peak would double, which criterion 4 forbids. The queue is therefore not a stopgap until
    /// the engine is made reentrant; it is the contract.
    ///
    /// **And it is static: one queue for the whole process, not one per `Engine`.** Everything it
    /// protects is global — `GEMM`'s memoizations, `Arena.live`, the GPU, the 16 GB — whereas a
    /// `Engine` holds nothing. While it was an instance property, two views of an app that each
    /// created their own `Engine()` rendered in parallel: the contract was asserted, not
    /// enforced. It is now, however many `Engine`s are created.
    ///
    /// **QoS `.userInitiated`**: without it, a render launched by the `async` facade would inherit
    /// the default QoS, and the system would treat it as background work.
    static let file = DispatchQueue(label: "siliconed.engine", qos: .userInitiated)

    package init() {}
}

// ── the spectral schedule, which is a product rule ────────────────────────────────────

/// **The spectral schedule is a PRODUCT choice, not a machine one** — the half-size start of Z-Image,
/// Qwen-Image-2.1, Anima and Krea 2, one switch for all (`Spectral.halfSizeStart`; `Sampler.init(half:…)`,
/// `QwenImage21Sampler`, `euler`).
///
/// It is therefore not tuned like the AMX or the VAE slices: a profile cannot write it
/// (`EngineSettings` refuses), because it does not change the speed at constant image — **it changes
/// the image**. What judges it is the eye, on free draws, not a stopwatch: a verdict taken on a
/// single draw is not a verdict, and a rule judged clean on one schedule does not carry over to
/// another (Z-Image's, clean on its former seven evaluations, left grain on the eight of its recipe).
/// What lives here are the floors every half-size evaluation obeys, and the measurement's override.
package enum Spectral {

    /// **The half-size start, one switch for every family** — the app's « Faster Rendering », on by
    /// default there (`AppState.fasterRendering`); off here, so the CLI, the checks and the judge render
    /// the recipe at full size. Each family keeps its own count (`halfSizeSteps`: 3, Krea 2 4), judged by
    /// eye on its own schedule; its plan says where it applies.
    nonisolated(unsafe) package static var halfSizeStart = false

    /// `SILICONED_SPECTRAL` — read at render time, never at construction; `nil` when nothing forces it.
    /// It forces the half-size start's `k` over its rule, within the floor: a measurement, never the product.
    /// **Read from the environment, not through `EngineSettings.effective`**: only the environment may set
    /// it (a profile may not), and the app asks for plans before the capacity map is measured — through
    /// `effective`, that first plan froze the process on the defaults and the map was then refused, the
    /// AMX off for the whole session (+20 % per MOSS step).
    package static var forced: Int? {
        ProcessInfo.processInfo.environment["SILICONED_SPECTRAL"].flatMap { Int($0) }
    }

    /// **The 512-px floor in the cells of a latent of `factor` px per cell**: 64 for the 8-px VAEs,
    /// 32 for Qwen-Image-2.1's 16-px one. The floor is a pixel rule (the model's domain), not a
    /// cell count: read in 8-px cells, a 1024² Qwen latent (64²) would never go down.
    package static func floor(factor: Int) -> Int { minimumSide * 8 / factor }

    /// **The smallest latent side an evaluation is allowed to reach: 64, i.e. 512 px.**
    /// Project owner's decision, 2026-09-22: below 512² the model is out of its domain, so nothing
    /// evaluated there — neither image nor timing — means anything. **Per side**, since
    /// rectangles (2026-09-23): an 832×1216 has the area of 1024², its half 416×608 does not have
    /// that of a 512² on one side. The spectral plan never halves under it (`divisible`), and
    /// `Format` refuses a render that would cross it.
    package static let minimumSide = 64

    /// **Is the half-size allowed on this grid?** An evaluation divided by `d` must stay at the
    /// floor **on both sides** — `minimumSide` is a per-side floor, not a per-area one — and land
    /// on a grid that the ×2 patch tiles: each latent side a multiple of `2d`.
    ///
    /// A consequence to know: **no usual portrait or landscape format has a spectral schedule.**
    /// 832×1216 has the area of 1024², but its half-size would be 416×608 — one side under the
    /// floor, hence `k = 0`, and every evaluation at full size. The spectral schedule applies only
    /// from 1024 px **on each side**: that leaves 1024², 1024×1536, its transpose (half grid `h ≠ w`),
    /// the sizes between and above them whose latent sides divide (1088², 1152², 1024×1280, 2048²…)
    /// — none of them a usual portrait or landscape format.
    ///
    /// `floor`: the per-side floor **in this latent's cells** (`floor(factor:)`).
    static func divisible(height: Int, width: Int, by d: Int, floor: Int = minimumSide) -> Bool {
        height / d >= floor && width / d >= floor
            && height % (2 * d) == 0 && width % (2 * d) == 0
    }
}
