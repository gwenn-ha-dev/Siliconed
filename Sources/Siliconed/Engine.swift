import Foundation

/// **The product, as a library: a prompt goes in, pixels come out.**
///
/// This file holds the engine and what it renders; the chain itself — modules, wires, `render` —
/// is in `Chain.swift` and `Modules.swift`. It first pulled the chain out of the CLI's
/// `renderCommand`, where it lived mixed in with `print` calls, to make it an object any caller can
/// hold: the CLI, the app (`SiliconedApp`), a server if one is ever needed.
///
/// Three rules govern it, and each answers a flaw the CLI path had by nature.
///
/// 1. **The engine prints nothing and writes nothing.** It renders pixels and a trace; the PNG,
///    `out/INDEX.md` and the on-screen timings remain the caller's business. A library that
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
/// `ImageRGB`, `Library`, `Cancellation`). The one public error is `EngineError` (`Errors.swift`);
/// inside the package, each reader keeps its `Failure`, folded at the door.
public final class Engine: Sendable {

    // ── what comes out ────────────────────────────────────────────────────────────────────

    /// What each phase cost. `text`, `encoding`, `denoising` and `decoding` partition
    /// `total`; `tokenizer` and `encoder` are indented within `text` — the lesson learned on
    /// decompositions that partition nothing. `encoding`: the img2img image, 0 without one.
    public struct Timings: Sendable {
        public var tokenizer = 0.0, encoder = 0.0, text = 0.0
        public var encoding = 0.0, denoising = 0.0, decoding = 0.0, total = 0.0
    }

    /// The footprints at the checkpoints. What is measured here cannot be deduced from any total:
    /// a measurement found 520 MB of residue after the text phase because someone sampled between the two,
    /// not because a calculation predicted it.
    public struct Footprints: Sendable {
        public var start = 0, tokenizer = 0, encoder = 0, text = 0
        /// The image encoder's peak, taken before it returns its memory; 0 without an image.
        public var encoding = 0
        public var denoising = 0, end = 0
    }

    /// An image and what it took to get it. **No file has been written.**
    ///
    /// In a batch, each image has its own `Render`; the text (and the input image) are paid for only
    /// once, so `timings.text` and `timings.encoding` hold only for the first, and the `total`s
    /// of a batch's images partition the batch's time.
    public struct Render: Sendable {
        /// What requested it — for the PNG metadata and an app's history.
        public let model: String
        public let prompt: String
        public let seed: UInt64
        public let steps: Int
        public let loras: [LoRAEntry]
        /// What the LoRA stack did, in one line — `nil` when there was none.
        public let loraSummary: String?
        /// The image, planar `[3, height, width]`, fp32 around `[-1, 1]` (not clamped).
        public let image: ImageRGB
        /// Was the GPU/AMX split frozen (`Request.reproducible`, resolved at render time)?
        public let reproducible: Bool
        /// The evaluations done — for a sketch, those it stopped after.
        public let evaluations: Int
        /// **A sketch** (`Request.sketch`): the evaluations it stopped after, the image being the
        /// estimate x̂₀ there. `nil`: the whole render.
        public let sketch: Int?
        /// The `k` actually applied, which is not the settings' one (a profile is not allowed to
        /// write it) and which a file name must carry on pain of lying.
        public let spectral: Int
        /// The img2img strength; `nil` in txt2img.
        public let strength: Double?
        /// The first step executed and its σ (0 and 1 in txt2img).
        public let startStep: Int
        public let startSigma: Float
        public let timings: Timings
        public let footprints: Footprints
        public let tokens: Int
        /// **More detail** (`Request.detail`); `normal` writes nothing in the metadata.
        public var detail: Detail = .normal
        /// **The variations its noise was turned by** (`Request.variations`), those of null
        /// strength left out; empty for an ordinary render.
        public var variations: [Variation] = []

        /// **The generation metadata**, as `png()` writes it (`iTXt` chunks). ASCII keys, readable
        /// values; **no date** — two identical renders must remain two identical files.
        ///
        /// Enough to redo a **txt2img** on the same machine and the same version (`reproducible`
        /// says whether the bits will match). **Not an img2img**: the source image is not included
        /// (the `image` key says so), and a LoRA appears only by its file name.
        public var metadata: [String: String] {
            var m = ["prompt": prompt, "seed": "\(seed)", "model": model, "steps": "\(steps)",
                     "format": "\(image.width)x\(image.height)", "reproducible": "\(reproducible)",
                     "Software": "Siliconed"]
            if let strength {
                m["strength"] = String(format: "%.2f", strength)
                m["image"] = "source not included"
            }
            if let sketch { m["sketch"] = "\(sketch)" }
            if detail != .normal { m["detail"] = detail.rawValue }
            // `seed` stays the origin's: the variations turn its noise, in order (`Variation.text`).
            if !variations.isEmpty { m["variation"] = Variation.text(variations) }
            if !loras.isEmpty {
                m["lora"] = loras.map { "\((($0.path as NSString).lastPathComponent)):\(String(format: "%.2f", $0.strength))" }
                    .joined(separator: ",")
            }
            return m
        }

        /// The sRGB PNG, metadata included.
        public func png() throws(EngineError) -> Data { try image.png(metadata: metadata) }

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
        public struct Recipe: Sendable, Equatable {
            /// The family, as `Render.model` writes it (`z-image`, `qwen-image-2.1`…).
            public var model: String?
            public var prompt: String?
            public var seed: UInt64?
            public var steps: Int?
            public var width: Int?, height: Int?
            /// The stack in its order; `path` is **the file name only**, as written.
            public var loras: [LoRAEntry] = []
            /// The img2img strength; `nil` in txt2img.
            public var strength: Double?
            /// An img2img: the image it started from is not in the file.
            public var sourceImageMissing = false
            public var reproducible: Bool?
            /// `Request.detail`; `normal` when the key is absent (every render before it).
            public var detail: Detail = .normal
            /// The variations its noise was turned by, in order (`Request.variations`); empty for an
            /// ordinary render.
            public var variations: [Variation] = []
            /// Keys present whose value did not parse, sorted.
            public var unreadable: [String] = []

            /// `nil` unless `Software` is `Siliconed`: another program's keys (ComfyUI's `prompt`
            /// is a JSON graph) are not interpreted.
            public init?(metadata m: [String: String]) {
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
    public struct Plan: Sendable {
        /// The denoiser's plan, for each image: evaluations (the last `total` of the `step`
        /// events), reduced ones (the spectral), starting step and σ. Grids in latent cells.
        public let denoising: DenoisingPlan
        /// The batch's seeds, in image order; just one outside a batch.
        public let seeds: [UInt64]
        /// The stages that will run, in order — `image` only in img2img. The text and the input
        /// image run only once per batch; denoising and decoding, per image.
        public let stages: [Stage]
    }

    /// A render's stages, as `Event.stage` announces them.
    public enum Stage: String, Sendable, CaseIterable {
        case text, image, denoising, decoding
    }

    /// **Progress, for whoever wants to display it.** The engine emits, it does not format.
    ///
    /// The order of a successful render: `start`, then `stage(.text)` … `text`, [`stage(.image)`
    /// … `encoding`], then for each image of the batch `stage(.denoising)`, the `step` (and the
    /// `preview` if requested), `stage(.decoding)`, `decoding`, `image`. The `warning`s can
    /// fall anywhere. The closure is called **on the render thread**, synchronously: it must
    /// return quickly (a UI does its `MainActor` hop there, or goes through the `async` stream).
    public enum Event: Sendable {
        case start(Plan)
        /// **The start of a stage** — emitted BEFORE it begins, so the bar moves during the text
        /// encoder and the decoder. `image`: the index in the batch (0 for the text and the input
        /// image, which run only once).
        case stage(Stage, image: Int)
        case text(tokens: Int, tokenizer: Double, encoder: Double)
        /// The img2img image is encoded and mixed with the noise.
        case encoding(seconds: Double)
        /// **One denoiser evaluation**, numbered the same way for every model: `index` from
        /// 1 to `total`, `total` = the evaluations actually paid for (`Plan.evaluations`) — Z-Image's
        /// null steps are not steps. `sigma`: the σ evaluated. The grid is the one this step
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
    /// the half-size evaluations of the spectral schedule (the first ones, ~0.35 of a full step
    /// on Z-Image 1024²) and the full ones. A common average, drawn from the reduced steps,
    /// promised a render twice too short, then went backwards at every step. The time remaining
    /// therefore exists only **after the first full step**; decoding counts what it cost in this
    /// render, or 0.2 of a full step before the first.
    public struct Progress: Sendable {
        public static let textWeights = 0.5, imageWeights = 0.15, decodingWeights = 0.2

        private var total = 1.0, done = 0.0
        private var images = 0, evaluations = 0, reduced = 0
        private var secondsFull = 0.0, fullCount = 0
        private var secondsReduced = 0.0, reducedDone = 0
        private var decoding: Double?, decodingsDone = 0

        public init() {}

        /// `[0, 1]`.
        public var fraction: Double { min(1, done / total) }

        /// In seconds; `nil` before the first full step.
        public var estimatedRemaining: Double? {
            guard fullCount > 0 else { return nil }
            let full = secondsFull / Double(fullCount)
            let reducedAverage = reducedDone > 0 ? secondsReduced / Double(reducedDone) : full
            let remainingFull = max(0, images * (evaluations - reduced) - fullCount)
            let remainingReduced = max(0, images * reduced - reducedDone)
            let remainingDecodings = max(0, images - decodingsDone)
            return Double(remainingFull) * full + Double(remainingReduced) * reducedAverage
                + Double(remainingDecodings) * (decoding ?? Progress.decodingWeights * full)
        }

        public mutating func receive(_ event: Event) {
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

    public init() {}

}

// ── the spectral schedule, which is a product rule ────────────────────────────────────

/// **The spectral schedule is a PRODUCT choice, not a machine one.**
///
/// It is therefore not tuned like the AMX or the VAE slices: a profile cannot write it
/// (`EngineSettings` refuses), because it does not change the speed at constant image — **it changes
/// the image**. What judges it is the spectral check, against the spectral oracle and by eye,
/// not a stopwatch. That is why its rule is in the library's code, and not in the CLI it came
/// from: it belongs to the engine as much as the schedule itself.
///
/// ## ⚠️ `k = 2`, not `k = 3` — corrected on 2026-09-22
///
/// A first study had chosen `k=3` ("sharp") over `k=4` ("confetti"), **on a single draw**: the oracle's,
/// the reference's noise and golden captions. The first **free** render — our noise, seed 42 —
/// shows at `k=3` speckles on the T-shirt and debris floating in the air, that is, exactly the
/// failure mode that study attributed to `k=4`. Reproduced at seed 7, and **without** AMX co-execution,
/// so it is neither the draw nor an interaction with the AMX.
///
/// ```
///   k=0   140.5 s   clean           k=2   118.9 s   clean           k=3   100.2 s   debris
/// ```
///
/// The mechanism is the one that study already named for `k=4`: an excess of high frequency that the
/// remaining steps no longer have time to resorb. **The breaking point depends on the draw** — it
/// was between 3 and 4 on the oracle's, it is between 2 and 3 on these. An image verdict taken on
/// a single draw is not a verdict.
package enum Spectral {

    /// **The threshold is calibrated, not derived** — and that is the only honesty possible here.
    ///
    /// In rectified flow, the SNR at frequency `f` reaches 1 at `f*(σ) = (1 − σ)/σ`; an evaluation
    /// whose useful band fits under the threshold can be done at half-size without losing anything
    /// but what the noise already covered.
    ///
    /// ```
    ///   σ      1      0.947   0.882   0.800   0.692   0.545   0.333
    ///   f*     0      0.056   0.134   0.250   0.444   0.833   2.000
    ///                    ↑ calibrated threshold between the two ↑
    /// ```
    ///
    /// Theory says where the SNR equals 1; it does not say how much energy above `f*` a model
    /// **distilled to seven steps** can still resorb. Measurement does: `k=2` (so `f* ≤ 0.056`)
    /// is clean on four draws, `k=3` (`f* ≤ 0.134`) carries debris on two of three. The
    /// threshold is therefore taken **between the two**, at 0.10.
    ///
    /// What the rule brings over a hard-coded number: it **follows the schedule**. Moving to 12
    /// steps, or changing the `shift`, moves the σs — and so `k` — without redoing the campaign.
    package static let threshold: Float = 0.10

    /// **The smallest latent side an evaluation is allowed to reach: 64, i.e. 512 px.**
    /// Project owner's decision, 2026-09-22: below 512² the model is out of its domain, so nothing
    /// evaluated there — neither image nor timing — means anything. **Per side**, since
    /// rectangles (2026-09-23): an 832×1216 has the area of 1024², its half 416×608 does not have
    /// that of a 512² on one side. `Sampler.run` refuses a divisor that would cross it, and
    /// `Format` refuses a render that would cross it.
    package static let minimumSide = 64

    /// **Is the half-size allowed on this grid?** An evaluation divided by `d` must stay at the
    /// floor **on both sides** — `minimumSide` is a per-side floor, not a per-area one — and land
    /// on a grid that the ×2 patch tiles: each latent side a multiple of `2d`.
    ///
    /// A consequence to know: **no usual portrait or landscape format has a spectral schedule.**
    /// 832×1216 has the area of 1024², but its half-size would be 416×608 — one side under the
    /// floor, hence `k = 0`, and Z-Image pays its seven full evaluations there. The spectral
    /// schedule applies only from 1024 px **on each side**: under the area ceiling
    /// (`Format.maxSurface`, 1024×1536), that leaves only 1024², 1024×1536 and its
    /// transpose — the latter through the rectangular path (half-grid `h ≠ w`), `k = 2`, rendered
    /// clean by Z-Image.
    static func divisible(height: Int, width: Int, by d: Int) -> Bool {
        height / d >= minimumSide && width / d >= minimumSide
            && height % (2 * d) == 0 && width % (2 * d) == 0
    }

    /// The number of half-size evaluations, derived from the schedule. `height`, `width`: the
    /// latent grid.
    ///
    /// **It applies only from 1024 px on each side.** `k` half-size evaluations are 512² under a
    /// 1024² render — within the model's domain — but **256² under a 512² render**, and we removed
    /// 256² from the bench precisely because the model is out of its domain there. See `divisible`
    /// for rectangles.
    package static func steps(height: Int, width: Int, schedule: FlowMatchSchedule) -> Int {
        guard divisible(height: height, width: width, by: 2) else { return 0 }
        var k = 0
        for sigma in schedule.sigmas {
            let band = sigma > 0 ? (1 - sigma) / sigma : .infinity
            if band <= threshold { k += 1 } else { break }
        }
        return k
    }

    /// **The per-step divisors, derived from the same threshold** — `f*(σ) ≤ 2·0.10 / d`.
    ///
    /// A grid divided by `d` carries `d` times less band: if half-size is safe as long as
    /// `f* ≤ 0.10`, quarter-size is safe as long as `f* ≤ 0.05`. On the eight-step schedule, `f*`
    /// is `[0 ; 0.056 ; 0.134 ; …]`, so the rule returns **`[4, 2, 1, 1, 1, 1, 1]`**: the first
    /// step at a quarter, the second at a half, the rest at full size.
    ///
    /// ## ⚠️ The `2·` is not a comfort factor — corrected on 2026-09-22
    ///
    /// The formula was written `f* ≤ seuil / d` in the original comment, **and that is not what
    /// the sentence just above describes**. `threshold` is calibrated for the **half**-size, not for
    /// the full one: that is exactly what `steps(...)` applies, which counts the steps whose
    /// `f* ≤ threshold` and evaluates them at `d = 2`. The right anchor is therefore `d = 2 ↦ threshold`,
    /// which gives `seuil · 2 / d` and returns `d = 4 ↦ seuil/2`, the previous paragraph's
    /// sentence.
    ///
    /// Written `seuil / d`, the rule returns **`[4, 1, 1, …]`**: it refuses half-size at step 1,
    /// where `f* = 0.056 > 0.050`, that is, it contradicts `steps(...)` on the only step where the
    /// two functions rule together. Nobody had seen it because this path never ran:
    /// `SILICONED_DIVISORS` was always set by hand. **It is the first defect found by the test
    /// target**, and it was in a formula commented over ten lines — comment density is no
    /// substitute for an execution.
    ///
    /// ⚠️ **And theory does not see the model's domain**: at a quarter, a 1024² render would
    /// evaluate at **256²**, where the model is out of its domain. Since 2026-09-22, 512² is an
    /// enforced floor (`minimumSide`): the rule therefore returns `[2, 2, 1, …]` at 1024², the
    /// quarter being allowed only from 2048². **Called by no render, and kept internal**: the
    /// tests hold it as the rule's safeguard — it must stay in agreement with `steps(...)` on the
    /// half-size.
    static func divisors(height: Int, width: Int, schedule: FlowMatchSchedule) -> [Int] {
        guard divisible(height: height, width: width, by: 2) else {
            return Array(repeating: 1, count: max(0, schedule.sigmas.count - 1))
        }
        return (0..<(schedule.sigmas.count - 1)).map { step in
            let sigma = schedule.sigmas[step]
            let band = sigma > 0 ? (1 - sigma) / sigma : .infinity
            // The largest `d` of {4, 2, 1} whose threshold still holds. `threshold` is calibrated at
            // half-size, hence the `2 /`: at `d = 2` the bound is `threshold`, at `d = 4` it is half.
            // And never under `minimumSide`: at 1024², the quarter would be 256².
            for d in [4, 2] where band <= threshold * 2 / Float(d)
                && divisible(height: height, width: width, by: d) { return d }
            return 1
        }
    }
}
