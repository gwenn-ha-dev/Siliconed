import Accelerate
import Foundation

/// **Qwen-Image-2.1-Turbo's pipeline** — `QwenImage21Pipeline.__call__` with the Turbo's checkpoint
/// (`Qwen/Qwen-Image-2.1-Turbo@d65dbc9a`), in fp32: what happens between the request and the DiT, and
/// the loop around it.
///
///     references ─ RGB8 ─ Pillow Lanczos to ~R² (sides ×32) ─┬─ Qwen3-VL (with the prompt) ─ hidden [T, 4096] + slots
///                  (one copy, `preparedReferences`)          └─ VAE encoder (RGBA) ─ reference latents [64, hᵢ, wᵢ]
///     seed ─ torch.randn(1, 1, 64, h, w) (`TorchNoise`) ─ target latent [64, h, w]
///     model_index.json `sample_sigmas` ─ float32, as is ─ σ₀ … σ₇, then 0
///
///     prefill once (text + references, K/V kept) ─▶ 8 × { v = DiT(x, σᵢ);  x += (σᵢ₊₁ − σᵢ)·v }   no CFG
///     half-size start (off by default): steps 0 … k−1 are  x̂₀ = DiT½-estimate(𝓑x + σᵢ𝓑ε),  v = (x − 𝓛x̂₀)/σᵢ
///
/// What the reference does that a port by analogy would miss (read in `pipeline_qwenimage21.py` of
/// diffusers 0.41.0 — PR #14950, which differs from `80c7ed26` by `sample_sigmas` alone in the path
/// rendered —, the Turbo's `model_index.json`, `scheduler_config.json` and card):
///
///   - **the schedule lives in `model_index.json`** (`sample_sigmas`, 8 nodes), not in the scheduler's
///     config: the pipeline passes it as `sigmas=` and `set_timesteps` keeps it **as is** —
///     `np.array(sigmas).astype(np.float32)` (Python's doubles, then float32: two roundings), then
///     `shift·s/(1 + (shift − 1)·s)` with `shift = 1`, the identity to the bit, then **a 0 appended**.
///     The last step jumps from 0.4146 to 0;
///   - **`use_dynamic_shifting: false`**: the pipeline still computes μ from the target's tokens and
///     passes it, and the scheduler ignores it. No dependence on the size or on the references — a
///     generation and an edit at any size get the same eight σ. `max_image_seq_len`/`max_shift` are
///     still in the config and serve nothing;
///   - `num_inference_steps` alone does not override the schedule: 8 is the only count published,
///     any other is refused (`acceptedSteps`);
///   - the DiT is the turbo — a full distillation from the base (`time_text_embed` moved the most,
///     2.3·10⁻²), **not the base plus a LoRA**: no LoRA under the user's, the conditions' K/V depend on
///     the map (and on a user LoRA), one prefill per render;
///   - the DiT receives `t = σ·1000` (float32) **divided by 1000**: for these eight nodes the round
///     trip gives σ back to the bit, kept anyway (`QwenImage21DiT.timestep`);
///   - **the noise is drawn `(1, 1, 64, h, w)`** — frame axis before channels — and packed raster
///     `[h·w, 64]`: planar `[64, h, w]` is exactly its draw order (`TorchNoise`);
///   - the scheduler steps in float32, `x + (σ_next − σ)·v`, two roundings (no FMA);
///   - the references are sized by **`calculate_dimensions(R², w/h)`** — round half to EVEN at 32 —
///     with `R = output_resolution`, whatever the output size; one Lanczos (Pillow's, to the bit,
///     `PillowResample`) feeds both the encoder and the VAE;
///   - **every reference is made RGBA** (`convert("RGBA")`): a transparent one keeps its alpha — Pillow
///     premultiplies around the Lanczos, the VAE reads the four channels as they are, and only the vision
///     copy is laid over white (`preparedReference`, `PillowAlpha`; `qwen21-preprocess-alpha`);
///   - the output follows the **last** reference without `height`/`width`. **The product's rule is
///     the first** (`outputSize`: the image one edits), at ~R², each side ≥ 512, under `Format`'s
///     ceiling; an explicit size wins. The checks pass the oracle's sizes explicitly;
///   - an explicit size is **floored to 32** without a word (`multiple_of = vae_scale_factor · 2`); the
///     product refuses a side not a multiple of 32 at the preflight instead (`Format.multiple(for:)`);
///   - the VAE returns RGBA; the product writes the RGB planes, alpha dropped (`QwenImage21DecodingModule`).
///     The Turbo publishes its VAE in bf16, exactly the base's fp32 rounded to nearest: the product
///     reads that file as published, widened to fp32 at load (never below the precision published).
///     Its encoder is the base's to the bit, its `processor/` another set of files with the same tokens.
///
/// What is ours, not the reference's — **the half-size start** (`QwenImage21DenoisingModule.plan`),
/// off by default: a generation evaluates its first `k` steps on the half grid, Z-Image's recipe
/// (`Sampler.init(half:…)`). Never under a user LoRA, Detail, another schedule, a start past 0, **in
/// an edit**, nor wherever the half grid would fall under 512 px on a side (`Spectral.floor`, counted
/// in 16-px cells): every check at 512² runs without it, the engine reproduces the reference there.
/// How the half-size phase is built, and why it is a phase and not a flag, is `QwenImage21Sampler`'s.
///
/// ## Memory: what the K/V cache costs, and where it lives
///
/// The cache holds `2 · 32 · P · 4096` floats for `P` condition rows (text + references): 0.52 MB per
/// row, 4.3 GB for one ~1 MP reference, 12.9 GB for three (`QwenImage21CachePolicy`).
package enum QwenImage21Pipeline {
    /// `output_resolution`: the pipeline's default, and the product's — references at ~1 MP.
    package static let outputResolution = 1024
    /// The Turbo's step count: the length of its published schedule.
    package static let defaultSteps = 8

    // ── the schedule ────────────────────────────────────────────────────────────────────

    /// **The Turbo's schedule**: `model_index.json`'s `sample_sigmas`, checked against the parts of
    /// `FlowMatchEulerDiscreteScheduler`'s config (`scheduler.json` in the store) that would change
    /// them. Defaults: as published.
    package struct Scheduler: Equatable, Sendable {
        /// `sample_sigmas` as `set_timesteps` keeps them: each a Python double made float32.
        package var nodes: [Float] = Scheduler.published
        /// `model_index.json` of `d65dbc9a`, as written there.
        package static let publishedNodes: [Double] = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
        package static let published: [Float] = publishedNodes.map { Float($0) }
        package init() {}

        /// - Parameters:
        ///   - config: `scheduler/scheduler_config.json`. Anything that would move the published σ —
        ///     a dynamic or non-unit shift, a terminal stretch, inverted or Karras/exponential/beta σ,
        ///     stochastic steps — is refused: the engine has no such schedule.
        ///   - modelIndex: `model_index.json`, whose `sample_sigmas` are the schedule.
        package init(config: Data, modelIndex: Data) throws {
            guard let c = try JSONSerialization.jsonObject(with: config) as? [String: Any] else {
                throw Artifact.Failure.badHeader("scheduler config: not a JSON object")
            }
            func flag(_ key: String) -> Bool { c[key] as? Bool ?? false }
            func number(_ key: String) -> Double? { (c[key] as? NSNumber)?.doubleValue }
            guard !flag("use_dynamic_shifting") else {
                throw Artifact.Failure.badHeader("scheduler config: use_dynamic_shifting — the Turbo's schedule is fixed")
            }
            guard number("shift") ?? 1 == 1, number("shift_terminal") ?? 0 == 0 else {
                throw Artifact.Failure.badHeader("scheduler config: a shift (\(c["shift"] ?? "?"), terminal \(c["shift_terminal"] ?? "null")) — the Turbo's is 1, terminal null")
            }
            for key in ["invert_sigmas", "use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas", "stochastic_sampling"]
            where flag(key) {
                throw Artifact.Failure.badHeader("scheduler config: \(key) — not the Turbo's schedule")
            }
            guard let index = try JSONSerialization.jsonObject(with: modelIndex) as? [String: Any],
                  let raw = index["sample_sigmas"] as? [NSNumber] else {
                throw Artifact.Failure.badHeader("model_index.json: no sample_sigmas — not Qwen-Image-2.1-Turbo's")
            }
            let nodes = raw.map { Float($0.doubleValue) }
            guard nodes.count == defaultSteps, nodes.first == 1, zip(nodes, nodes.dropFirst()).allSatisfy({ $0 > $1 }),
                  nodes.last! > 0 else {
                throw Artifact.Failure.badHeader("model_index.json: sample_sigmas \(nodes) — \(defaultSteps) decreasing σ from 1 expected")
            }
            self.nodes = nodes
        }

        package static func files(config: String, modelIndex: String) throws -> Scheduler {
            try Scheduler(config: Data(contentsOf: URL(fileURLWithPath: config)),
                          modelIndex: Data(contentsOf: URL(fileURLWithPath: modelIndex)))
        }

        /// **`set_timesteps(sigmas=sample_sigmas, mu=μ)`**: the nodes as they are (μ ignored), then a zero σ.
        package var sigmas: [Float] { nodes + [0] }
    }

    /// The step counts the Turbo has a schedule for.
    package static let acceptedSteps = [defaultSteps]

    /// Another count has no published schedule: refused (`QwenImage21Pipeline.Failure.steps`).
    package static func check(steps: Int) throws {
        guard acceptedSteps.contains(steps) else { throw Failure.steps(steps) }
    }

    /// `scheduler.timesteps`: `σ · 1000` in float32 — what the oracle records.
    package static func timesteps(_ sigmas: [Float]) -> [Float] { sigmas.dropLast().map { $0 * 1000 } }

    // ── the sizes ───────────────────────────────────────────────────────────────────────

    /// **The product's output size** for an edit without an explicit size: the first reference's
    /// aspect ratio at ~R² (`calculate_dimensions`, sides at 32), then the product's bounds — each side
    /// ≥ `Format.minimumSide` (raised keeping the ratio), the area ≤ `Format.editSurface` (the long
    /// side cut). Always a multiple of 32, as the pipeline floors it.
    package static func outputSize(referenceWidth: Int, referenceHeight: Int,
                                   resolution: Int = outputResolution) -> (width: Int, height: Int) {
        let ratio = Double(referenceWidth) / Double(referenceHeight)
        var (w, h) = Qwen3VLImages.conditionSize(width: referenceWidth, height: referenceHeight, resolution: resolution)
        let floor = (Format.minimumSide + 31) / 32 * 32
        if w < floor { w = floor; h = max(floor, Int((Double(w) / ratio / 32).rounded()) * 32) }
        if h < floor { h = floor; w = max(floor, Int((Double(h) * ratio / 32).rounded()) * 32) }
        if w * h > Format.editSurface {
            if w >= h { w = Format.editSurface / h / 32 * 32 } else { h = Format.editSurface / w / 32 * 32 }
        }
        return (w, h)
    }

    /// **One reference as the pipeline reads it**: RGB8 again (the `[-1, 1]` floats of an 8-bit image
    /// round back to their bytes), Pillow's Lanczos to `calculate_dimensions(R², w/h)`. The same copy
    /// then goes to the encoder (`Qwen3VLImages.pixels`) and to the VAE (`ImageRGB`, `x/255·2 − 1`).
    package static func reference(_ image: ImageRGB, resolution: Int = outputResolution) -> Qwen3VLImages.RGB8 {
        let bytes = rgb8(image)
        let size = Qwen3VLImages.conditionSize(width: image.width, height: image.height, resolution: resolution)
        return PillowResample.lanczos(bytes, width: size.width, height: size.height)
    }

    /// **One reference as the VAE and the vision tower read it** — `reference`'s copy for an opaque
    /// image; with transparency (`ImageRGB.rgba`), what `__call__` does to an RGBA one: Pillow's Lanczos
    /// premultiplied (`PillowAlpha.lanczos`, a copy at the same size: a transparent pixel keeps its
    /// RGB), `rgba` kept for the VAE (four channels, the real alpha), `pixels` laid over white by
    /// `white.paste(img, mask=A)` for the vision tower. The alpha is not composited for the VAE, and the
    /// RGB is not composited before the resize — what an analogy with the other models would do.
    package static func preparedReference(_ image: ImageRGB, resolution: Int = outputResolution) -> ImageRGB {
        guard let rgba = image.rgba else { return self.image(reference(image, resolution: resolution)) }
        let size = Qwen3VLImages.conditionSize(width: image.width, height: image.height, resolution: resolution)
        let resized = PillowAlpha.lanczos(rgba, width: image.width, height: image.height,
                                          toWidth: size.width, toHeight: size.height)
        let white = PillowAlpha.overWhite(resized, width: size.width, height: size.height)
        return ImageRGB(pixels: self.image(white).pixels, height: size.height, width: size.width, rgba: resized)
    }

    /// `[3, h, w]` in `[-1, 1]` → interleaved 8-bit RGB: `round((x + 1)/2 · 255)`, the exact inverse of
    /// `ImageRGB`'s `b/255·2 − 1` on 8-bit sources.
    package static func rgb8(_ image: ImageRGB) -> Qwen3VLImages.RGB8 {
        let plane = image.width * image.height
        var bytes = [UInt8](repeating: 0, count: 3 * plane)
        for c in 0..<3 {
            for p in 0..<plane {
                let v = (image.pixels[c * plane + p] + 1) / 2 * 255
                bytes[p * 3 + c] = UInt8(max(0, min(255, v.rounded())))
            }
        }
        return Qwen3VLImages.RGB8(bytes: bytes, width: image.width, height: image.height)
    }

    /// 8-bit RGB → `[3, h, w]` in `[-1, 1]`, as `VaeImageProcessor` makes it (`x/255`, then `2x − 1`).
    package static func image(_ rgb: Qwen3VLImages.RGB8) -> ImageRGB {
        let plane = rgb.width * rgb.height
        var pixels = [Float](repeating: 0, count: 3 * plane)
        for p in 0..<plane { for c in 0..<3 { pixels[c * plane + p] = Float(rgb.bytes[p * 3 + c]) / 255 * 2 - 1 } }
        return ImageRGB(pixels: pixels, height: rgb.height, width: rgb.width)
    }

    package enum Failure: Error, CustomStringConvertible, Equatable {
        case steps(Int)
        case imageToImage
        package var description: String {
            switch self {
            case .steps(let n):
                return "\(n) steps: Qwen-Image-2.1-Turbo publishes a schedule for \(QwenImage21Pipeline.defaultSteps) steps only"
            case .imageToImage:
                return "Qwen-Image-2.1 does no img2img (its pipeline has none): pass the image as a reference (--ref) "
                    + "and say what to change"
            }
        }
    }
}

/// **The denoising of one render** — the loop of `__call__` around `QwenImage21DiT`: one DiT (the user's
/// LoRAs, if any, size its arenas), one prefill, the steps.
///
/// **The half-size start** (`reduced`, generation only; the header) is one more phase in front: the
/// first `k` steps on a DiT built for the half grid. The state stays at full size, only the
/// evaluation goes down — Z-Image's recipe (`Sampler.init(half:…)`, where the why is written):
///
///     x_low = 𝓑x + σ·𝓑ε             (𝓑 the 2×2 mean, ε the starting noise)
///     x̂₀ = x_low − σ·v_θ(x_low, σ)   (here the DiT returns dx/dσ itself, no negation)
///     dx/dσ = (x − 𝓛x̂₀) / σ          (𝓛 Lanczos-3 ×2) — the state stays exactly σε + (1 − σ)S
///
/// It replaces `x += dt·𝓘v(𝓟x)` (decimation, bilinear velocity; until 0.10), which left the latent's
/// high band at its starting noise and fed the model an alias of it. What makes it a phase rather
/// than a flag on the DiT, and what a port by analogy would miss:
///
///   - **the target's grid is built into the joint sequence** (`QwenImage21Sequence`: its slots, its
///     RoPE rows, the arenas' `S`): the half-size steps need their own DiT, hence their own prefill;
///   - **that prefill gives the same K/V as the full one**: the conditions come first and never see
///     the target (block-causal), and the RoPE lays the target out *after* them, its `(y, x)` centred
///     on its own grid (`QwenImage21Rope.forward` in diffusers). A half-size evaluation is exactly a
///     native 512² evaluation of the same conditions; for a generation the prefix is a few dozen
///     text rows, fused into the first step: the second prefill costs nearly nothing;
///   - **the latent has no 2×2 packing** (`_pack_latents` is a flatten, 64 true channels at 16 px):
///     𝓑 and 𝓛 work on the planar `[64, h, w]`, and commute with the VAE's per-channel normalization
///     (both are affine with weights summing to one);
///   - **the schedule stays the full grid's**: the half-size evaluation is told the same σ;
///   - **the floor counts 16-px cells**: a 1024² latent is 64², its half 32² = 512² (`Spectral.floor`).
package struct QwenImage21Sampler {
    /// What a step hands back: its absolute index, σᵢ, the seconds, the latent after the update and
    /// the velocity, both planar `[64, h, w]` at full size (the velocity up-sampled after a half-size
    /// evaluation), the grid the DiT evaluated, and where the estimate x̂₀ is read: `x − σₓ·v` with `x`
    /// the latent and `σₓ` the next σ under Euler, `x` the walk's own x̂₀ and `σₓ = 0` under another
    /// solver (`Sampler.afterEvaluation`, the same rule).
    package typealias Step = (index: Int, sigma: Float, seconds: Double, latent: UnsafePointer<Float>,
                              velocity: UnsafePointer<Float>, evaluatedHeight: Int, evaluatedWidth: Int,
                              estimate: UnsafePointer<Float>, estimateSigma: Float)

    package let artifact: Artifact
    /// The user's LoRA stack, `nil` without: the Turbo is the map itself.
    package let lora: LoRA?
    /// Where the K/V cache lives: `nil`, the policy's choice for the render's prefix
    /// (`QwenImage21CachePolicy`); forced by the checks that measure the three.
    package var storage: QwenImage21DiT.CacheStorage?
    /// The storage the last `run` used.
    package private(set) var used: QwenImage21DiT.CacheStorage = .memory
    /// **The render cache's K/V file** (`RenderCache.kvPath`): when the conditions go to a file, it is
    /// this one — read back without a prefill if a previous render left it whole, kept for the next one
    /// otherwise. `nil`: the unlinked file, gone with the render.
    package var kept: (path: String, key: String)?
    /// What a kept file must leave free on its volume (`RenderCache.keptReserve`).
    package var keptReserve: Int64 = RenderCache.defaultKeptReserve
    package var freezeCut = EngineSettings.effective.frozenCut
    /// The DiT's products may take the Neural Engine (`Request.neuralEngine`).
    package var neuralEngine = EngineSettings.effective.gemmANE
    package var cancellation: Cancellation?
    /// Seconds: `prefill`, `steps`, and the DiT's own breakdown (`dit …`).
    package private(set) var timings: [String: Double] = [:]
    /// **The DiT of the previous image of a batch, reused**: same sequence, LoRA stack and storage —
    /// every image of a request has them —, so building it again would only be a fresh allocation wave
    /// (`Context.keptDenoiser`). A user LoRA included: the stack is the request's, the same per image.
    package var reusable: QwenImage21DiT?
    /// The DiT the run used, for the next image.
    package private(set) var reused: QwenImage21DiT?

    /// **The update rule** (`Solver`): Euler, the distillation's own, unless another is asked; the
    /// ancestral rule's noise from `solverSeed`. The schedule and what the DiT is told do not change.
    package var solver: Solver = .euler
    package var solverSeed: UInt64 = 0

    package init(artifact: Artifact, lora: LoRA?) {
        self.artifact = artifact; self.lora = lora
    }

    /// - Parameters:
    ///   - latent: the target `[64, h, w]`, updated in place.
    ///   - hidden: the encoder's `[T, 4096]`; `slots`: its `image_pad_mask` (`T` entries).
    ///   - references: each `[64, hᵢ, wᵢ]`, in prompt order.
    ///   - sigmas: the full schedule (`steps + 1` values).
    ///   - told: the σ the DiT is told at each step (`Detail.modelSigmas`); `nil`: `sigmas`.
    ///   - reduced: the first steps evaluated at half size (the spectral schedule); 0 with references.
    package mutating func run(latent: inout [Float], height: Int, width: Int,
                              hidden: [Float], slots: [Bool], references: [Latent],
                              sigmas: [Float], told: [Float]? = nil, reduced: Int = 0,
                              onStep: (Step) throws -> Void) throws {
        let steps = sigmas.count - 1
        // Under DPM++ and UniPC, diffusers' integer timestep over 1000 (`Solver.toldSigma`).
        let told = (told ?? sigmas).enumerated().map { solver.toldSigma($1, first: $0 == 0) }
        let half = references.isEmpty ? min(max(0, reduced), steps) : 0
        guard height % 2 == 0, width % 2 == 0, half == 0 || (height % 4 == 0 && width % 4 == 0) else {
            throw Artifact.Failure.misuse("Qwen-Image-2.1: a target grid \(width)×\(height) not made of 2×2 slots"
                                          + (half > 0 ? " at half size" : ""))
        }
        func sequence(_ d: Int) throws -> QwenImage21Sequence {
            try QwenImage21Sequence(
                slots: slots + [Bool](repeating: true, count: height / d * (width / d) / 4),
                images: references.map { .init(height: $0.height, width: $0.width) }
                    + [.init(height: height / d, width: width / d)])
        }
        let channels = latent.count / (height * width)
        // The references' latents at a stable address for the whole denoising, in `Pages`.
        let copies = references.map { reference -> Pages in
            let pages = Pages(bytes: reference.values.count * 4)
            pages.floats().update(from: reference.values, count: reference.values.count)
            return pages
        }
        let pointers = copies.map { $0.floats() }
        defer { withExtendedLifetime(copies) {} }
        timings = [:]
        // Another solver walks the whole schedule across the phases (`SolverWalk`, its history in
        // `Pages`); Euler keeps the loop below, its two roundings to the bit.
        var stream = Noise(seed: solverSeed ^ 0x5EED_A9CE_57A1_0000)
        let count = latent.count
        let walk = solver == .euler ? nil
            : SolverWalk(solver: solver, schedule: sigmas, start: 0, count: count,
                         noise: solver == .eulerAncestral ? { _, e in stream.fill(e, count: count) } : nil)
        // ε, the starting noise the half-size start reads at every one of its steps (`run` always
        // starts at σ₀: no img2img here).
        let noise = half > 0 ? latent : []
        let phases: [(range: Range<Int>, divisor: Int)] = [(0..<half, 2), (half..<steps, 1)].filter { !$0.range.isEmpty }
        for (range, divisor) in phases {
            try cancellation.check()
            let sequence = try sequence(divisor)
            // Its own scope: the half-size DiT (arenas, cache) is gone before the full one asks. It is
            // never kept for a batch's next image, nor does it take the render cache's K/V file.
            used = storage ?? QwenImage21CachePolicy.storage(prefix: sequence.prefix)
            let dit = try (divisor == 1 ? reusable : nil)
                ?? QwenImage21DiT(artifact: artifact, sequence: sequence, lora: lora, storage: used,
                                  kept: divisor == 1 ? kept.map { ($0.path, $0.key, keptReserve) } : nil,
                                  freezeCut: freezeCut, neuralEngine: neuralEngine)
            if divisor == 1 { reused = dit }
            if dit.reusedPrefill { timings["kv reused", default: 0] += 1 }
            if dit.keepsPrefill { timings["kv kept", default: 0] += 1 }
            dit.cancellation = cancellation
            try dit.tabulateModulation(sigmas: Array(told[0..<steps]))
            var started = Date()
            try hidden.withUnsafeBufferPointer {
                // The conditions' layers run inside the first step (`QwenImage21DiT.prefillPending`): its
                // seconds include them, `prefill` keeps only `txt_in` and `img_in`.
                try dit.prefill(hidden: $0.baseAddress!, references: pointers.map { UnsafePointer($0) }, intoFirstStep: true)
            }
            timings["prefill", default: 0] += Date().timeIntervalSince(started)
            let (evaluatedHeight, evaluatedWidth) = (height / divisor, width / divisor)
            let lowCount = divisor > 1 ? channels * evaluatedHeight * evaluatedWidth : 0
            var small = [Float](repeating: 0, count: lowCount), estimate = small
            var upsampled = [Float](repeating: 0, count: divisor > 1 ? count : 0)
            for i in range {
                try cancellation.check()
                started = Date()
                let v: [Float]
                if divisor > 1 {
                    // 𝓑x + σ𝓑ε on the state's own σ (not the told one, which only conditions the DiT).
                    let sigma = sigmas[i]
                    latent.withUnsafeBufferPointer { x in
                        noise.withUnsafeBufferPointer { ε in
                            Resample.average(x.baseAddress!, plus: sigma, times: ε.baseAddress!, into: &small,
                                             channels: channels, height: evaluatedHeight, width: evaluatedWidth)
                        }
                    }
                    let down = try small.withUnsafeBufferPointer { try dit.forward(latent: $0.baseAddress!, sigma: told[i]) }
                    for j in 0..<lowCount { estimate[j] = small[j] - sigma * down[j] }
                    estimate.withUnsafeBufferPointer { e in
                        Resample.lanczosDouble(e.baseAddress!, into: &upsampled, channels: channels,
                                               height: evaluatedHeight, width: evaluatedWidth)
                    }
                    let inverse = 1 / sigma
                    v = (0..<count).map { (latent[$0] - upsampled[$0]) * inverse }
                } else {
                    v = try latent.withUnsafeBufferPointer { try dit.forward(latent: $0.baseAddress!, sigma: told[i]) }
                }
                if let walk {
                    latent.withUnsafeMutableBufferPointer { x in
                        v.withUnsafeBufferPointer { walk.step(latent: x.baseAddress!, derivative: $0.baseAddress!) }
                    }
                } else {
                    // `scheduler.step`: `dt = σ_next − σ` in float32, then `x + dt·v` — two roundings.
                    let dt = sigmas[i + 1] - sigmas[i]
                    latent.withUnsafeMutableBufferPointer { x in
                        for j in 0..<x.count { x[j] += dt * v[j] }
                    }
                }
                // Z-Image's warning (`Sampler.run`), here too: a non-finite velocity otherwise comes out
                // as an entirely black image, with no word of which step it became so.
                if let bad = Sampler.countNotFinite(v, v.count) {
                    Warnings.emit("✗ step \(i): the VELOCITY carries \(bad.count) non-finite values "
                        + "(first at index \(bad.first), evaluation at \(evaluatedWidth)×\(evaluatedHeight))")
                }
                let seconds = Date().timeIntervalSince(started)
                timings["steps", default: 0] += seconds
                try latent.withUnsafeBufferPointer { x in
                    try v.withUnsafeBufferPointer {
                        try onStep((i, sigmas[i], seconds, x.baseAddress!, $0.baseAddress!, evaluatedHeight, evaluatedWidth,
                                    walk?.estimate ?? x.baseAddress!, walk == nil ? sigmas[i + 1] : 0))
                    }
                }
            }
            for (name, seconds) in dit.timings { timings["dit " + name, default: 0] += seconds }
        }
    }
}

/// **The LoRA stack expanded one layer ahead, not all at once.** `CacheLoRA` keeps the whole stack in
/// fp32 for the render: a rank-256 LoRA on the 224 modules of the blocks is 2.68 GB resident, more than
/// any other item of the denoising stage. Here only two layers live (2 × 84 MB): layer `l` being read by the GPU
/// and layer `l + 1` being expanded on a background thread meanwhile — the same `LoRA.materialize`, so
/// the same bits, redone at each evaluation (~5 s of CPU per evaluation, hidden behind the GEMMs).
///
/// The two slots are **fixed arenas with a fixed layout** (every layer has the same modules and
/// shapes): a module's address is that of its slot, so `GEMM.wrap`'s memo by address stays valid —
/// freeing and remapping a slot would hand the GPU a stale mapping. Slot `l mod 2` is refilled for
/// layer `l + 2` only once layer `l + 1` starts, when the GPU has finished with layer `l` (every GEMM
/// waits for its command buffer). The modules outside the blocks (`txt_in`, `img_in`, `proj_out`,
/// read once per evaluation) stay in a small `CacheLoRA`.
package final class StreamedLoRA {
    package typealias Module = (down: UnsafeMutablePointer<Float>, up: UnsafeMutablePointer<Float>, rank: Int)
    private let lora: LoRA
    private let layers: Int
    private let blockPrefix = "transformer_blocks."
    /// Per module suffix (`attn.to_q`): `(down offset, up offset, k, n)` in bytes within a slot.
    private var layout: [String: (down: Int, up: Int, k: Int, n: Int)] = [:]
    private var slots: [Arena] = []
    private var filled: [Int?] = [nil, nil]
    private var pending: [DispatchGroup?] = [nil, nil]
    private var failures: [Error?] = [nil, nil]
    private let others: CacheLoRA
    package let bytes: Int

    /// - Parameter retain: the modules served (the exact ones are computed elsewhere, in double).
    package init(lora: LoRA, layers: Int, retain: @escaping (String) -> Bool) throws {
        self.lora = lora; self.layers = layers
        var shapes: [String: (k: Int, n: Int)] = [:]
        for layer in lora.layers {
            for name in layer.artifact.order where name.hasSuffix(".down") {
                let target = String(name.dropLast(5))
                guard target.hasPrefix(blockPrefix + "0."), retain(target), let down = layer.artifact.tensors[name],
                      let up = layer.artifact.tensors[target + ".up"] else { continue }
                shapes[String(target.dropFirst((blockPrefix + "0.").count))] = (down.shape[0], up.shape[1])
            }
        }
        var modules: [String: (k: Int, n: Int, rank: Int)] = [:]
        for (suffix, shape) in shapes { modules[suffix] = (shape.k, shape.n, lora.rank(blockPrefix + "0." + suffix)) }
        let laid = StreamedLoRA.layout(modules)
        layout = laid.layout
        let offset = laid.bytes
        // Every block must carry the same modules at the same rank as block 0.
        for target in lora.rankPerTarget.keys where target.hasPrefix(blockPrefix) && retain(target) {
            let parts = target.dropFirst(blockPrefix.count).split(separator: ".", maxSplits: 1)
            guard parts.count == 2, let suffix = parts.last.map(String.init), layout[suffix] != nil,
                  lora.rank(target) == lora.rank(blockPrefix + "0." + suffix) else {
                throw LoRA.Failure.inconsistentShape(target: target, reason: "a block whose LoRA modules differ from block 0's")
            }
        }
        bytes = offset
        if offset > 0 { slots = [try Arena(capacity: offset), try Arena(capacity: offset)] }
        others = try CacheLoRA(lora: lora, retain: { !$0.hasPrefix("transformer_blocks.") && retain($0) })
    }

    /// **One slot's layout** — each module's `down` then `up`, page-rounded, in suffix order — and its
    /// bytes. Two slots of it live during the denoising; `MemoryNeed.qwenImage21` charges them from here.
    package static func layout(_ modules: [String: (k: Int, n: Int, rank: Int)])
            -> (layout: [String: (down: Int, up: Int, k: Int, n: Int)], bytes: Int) {
        let page = Arena.alignment
        func rounded(_ b: Int) -> Int { (b + page - 1) / page * page }
        var layout: [String: (down: Int, up: Int, k: Int, n: Int)] = [:]
        var offset = 0
        for suffix in modules.keys.sorted() {
            let (k, n, r) = modules[suffix]!
            layout[suffix] = (offset, offset + rounded(k * r * 4), k, n)
            offset += rounded(k * r * 4) + rounded(r * n * 4)
        }
        return (layout, offset)
    }

    /// The stack of module `target`, expanded — `nil` if no LoRA touches it.
    package func module(_ target: String, k: Int, n: Int) throws -> Module? {
        guard target.hasPrefix(blockPrefix) else { return try others.module(target, k: k, n: n) }
        let parts = target.dropFirst(blockPrefix.count).split(separator: ".", maxSplits: 1)
        guard parts.count == 2, let layer = Int(parts[0]), let place = layout[String(parts[1])] else {
            return lora.rank(target) > 0 ? try others.module(target, k: k, n: n) : nil
        }
        precondition(place.k == k && place.n == n, "StreamedLoRA: \(target) is [\(k), \(n)], laid out [\(place.k), \(place.n)]")
        let slot = layer % 2
        if filled[slot] != layer { try fill(layer, sync: true) }
        try collect(slot)
        // The other slot held layer − 1, which the GPU has finished: refill it with layer + 1.
        let next = (layer + 1) % layers
        if filled[next % 2] != next { try fill(next, sync: false) }
        let base = slots[slot].pointer("slot")!
        return ((base + place.down).assumingMemoryBound(to: Float.self), (base + place.up).assumingMemoryBound(to: Float.self),
                lora.rank(target))
    }

    private func collect(_ slot: Int) throws {
        if let group = pending[slot] { group.wait(); pending[slot] = nil }
        if let failure = failures[slot] { failures[slot] = nil; filled[slot] = nil; throw failure }
    }

    /// Expands every module of `layer` into its slot — now, or on a background thread.
    private func fill(_ layer: Int, sync: Bool) throws {
        let slot = layer % 2
        if let group = pending[slot] { group.wait(); pending[slot] = nil }
        if slots[slot].pointer("slot") == nil { _ = try slots[slot].reserve("slot", bytes: bytes) }
        filled[slot] = layer
        failures[slot] = nil
        let base = slots[slot].pointer("slot")!
        let work = { [lora, layout, blockPrefix] () -> Error? in
            do {
                for (suffix, place) in layout {
                    _ = try lora.materialize(blockPrefix + "\(layer)." + suffix, k: place.k, n: place.n,
                                             down: (base + place.down).assumingMemoryBound(to: Float.self),
                                             up: (base + place.up).assumingMemoryBound(to: Float.self))
                }
                return nil
            } catch { return error }
        }
        if sync {
            failures[slot] = work()
            return
        }
        let group = DispatchGroup()
        nonisolated(unsafe) let task = work
        nonisolated(unsafe) let owner = self
        DispatchQueue.global(qos: .userInitiated).async(group: group) { owner.failures[slot] = task() }
        pending[slot] = group
    }

    deinit { for group in pending { group?.wait() } }
}

/// **Where the conditions' K/V live**, by the size of the prefix. Measured on a 1-reference edit
/// (P = 4,092, 4.3 GB of cache, 768×512 then 1248×832): in memory, the render swapped (+167 k pages
/// out, 8.3 GB peak); on disk (`QwenImage21KVFile`), the same bits, no swap, and faster (10.5 against
/// 12.0 s per step: the read of layer `l + 1` hides behind layer `l`, where the memory copy did not);
/// recomputed (a full pass per step), 29.8 s per step and other bits. So: a text-only prefix (a
/// generation, tens of rows) in memory, anything with an image on disk.
package enum QwenImage21CachePolicy {
    /// `2 · layers · P · d · 4` bytes.
    package static func bytes(prefix: Int, layers: Int = 32, dim: Int = 4096) -> Int { 2 * layers * prefix * dim * 4 }
    package static let memoryBudget = 256 << 20

    /// **The one placement rule** — the sampler's, the module's and the memory need's (`QwenImage21Need`).
    package static func storage(prefix: Int) -> QwenImage21DiT.CacheStorage {
        bytes(prefix: prefix) <= memoryBudget ? .memory : .file
    }

    /// The longest prefix whose K/V stay in memory (256 rows).
    package static var largestMemoryPrefix: Int { memoryBudget / bytes(prefix: 1) }

    /// **The DiT's prefix of a conditioning**: its rows, each image slot being 4 rows of the DiT (one 2×2
    /// group of latent tokens) where the encoder has one — what `QwenImage21Sequence.prefix` counts.
    package static func prefix(conditioningRows rows: Int, imageSlots: Int) -> Int { rows + 3 * imageSlots }
}

/// **The conditions' K/V on disk** — `QwenImage21DiT.CacheStorage.file`: one unlinked temporary file
/// (it disappears with the process, even killed), written once by the prefill, read back layer by layer
/// at each step. `F_NOCACHE`: the reads and writes go straight between the SSD and the DiT's own
/// `keys`/`values` rows, never through the unified buffer cache — the file never occupies memory,
/// neither ours nor the system's. Every offset and length is a multiple of a 16 KB page (`d = 4096`
/// floats), as direct I/O requires.
///
/// The step reads layer `l + 1` while layer `l` finishes (`prefetch` after its attention, `wait` before
/// the next): the prefix rows `[0, P)` of `keys`/`values` are free from the attention on, the target
/// writes only rows `[P, P + S)`.
///
/// **Kept between renders** (`kept(at:…)`, the render cache's `kv/<key>.kv`): the same file, written
/// under a temporary name, then — once the prefill has written its last layer — synced, closed by a
/// trailer page (magic, key, shape) and renamed. The next render with the same key opens it
/// `complete`: no prefill, the steps read it as they read the file they would have written.
package final class QwenImage21KVFile {
    package private(set) var path: String
    package let bytesPerLayer: Int
    package let layers: Int
    private let fd: Int32
    private let lanes = 8
    private var pending: (layer: Int, group: DispatchGroup, failure: ErrorBox)?
    /// The conditions' K/V are all there (a kept file read back): the prefill has nothing to do.
    package private(set) var complete = false
    /// This render writes the file the next one may read back (`kept`, not complete yet).
    package let writesKept: Bool
    /// Where a file being kept goes once complete, and its trailer's key; `nil` for the unlinked one.
    private var destination: (path: String, key: String)?
    /// The trailer: one page after the layers, `SLKV1` + JSON (key, layers, bytes per layer).
    package static let trailerBytes = Arena.alignment

    private final class ErrorBox: @unchecked Sendable {
        private let lock = NSLock()
        private var error: Int32 = 0
        func fail(_ e: Int32) { lock.lock(); if error == 0 { error = e }; lock.unlock() }
        var value: Int32 { lock.lock(); defer { lock.unlock() }; return error }
    }

    package struct Failure: Error, CustomStringConvertible {
        package let description: String
    }

    /// - Parameter halfBytes: `P · d · 4`, one of K or V for one layer.
    package init(halfBytes: Int, layers: Int, directory: String = NSTemporaryDirectory()) throws {
        precondition(halfBytes % Arena.alignment == 0, "QwenImage21KVFile: a half that is not whole pages")
        bytesPerLayer = 2 * halfBytes; self.layers = layers; writesKept = false
        let total = Int64(bytesPerLayer) * Int64(layers)
        let free = (try? URL(fileURLWithPath: directory).resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage ?? 0
        guard free > total + (2 << 30) else {
            // The K/V cache of an edit, with 2 GB to spare.
            throw EngineError.diskFull(needed: Int(total) + (2 << 30), available: Int(free))
        }
        let path = (directory as NSString).appendingPathComponent("siliconed-kv-\(getpid())-\(UUID().uuidString)")
        let fd = open(path, O_RDWR | O_CREAT | O_EXCL, 0o600)
        guard fd >= 0 else { throw Failure(description: "K/V cache: cannot create \(path) (errno \(errno))") }
        unlink(path)
        _ = fcntl(fd, F_NOCACHE, 1)
        self.path = path; self.fd = fd
    }

    private init(fd: Int32, path: String, halfBytes: Int, layers: Int, complete: Bool,
                 destination: (path: String, key: String)?) {
        self.fd = fd; self.path = path; self.bytesPerLayer = 2 * halfBytes; self.layers = layers
        self.complete = complete; self.destination = destination; self.writesKept = destination != nil
        _ = fcntl(fd, F_NOCACHE, 1)
    }

    /// **The kept file of `key`**: read back if it is there and whole (`complete`), otherwise one to
    /// write next to it, renamed to `path` by `seal`. `nil` when keeping it would leave less than
    /// `reserve` bytes free (or the folder cannot be written): the caller then takes the unlinked file.
    package static func kept(at path: String, key: String, halfBytes: Int, layers: Int, reserve: Int64) -> QwenImage21KVFile? {
        precondition(halfBytes % Arena.alignment == 0, "QwenImage21KVFile: a half that is not whole pages")
        let total = Int64(2 * halfBytes) * Int64(layers)
        let fd = open(path, O_RDONLY)
        if fd >= 0 {
            var s = stat()
            if fstat(fd, &s) == 0, Int64(s.st_size) == total + Int64(trailerBytes),
               let trailer = readTrailer(fd, offset: off_t(total)),
               trailer.key == key, trailer.layers == layers, trailer.bytesPerLayer == 2 * halfBytes {
                utimes(path, nil)   // its date says when it last served
                return QwenImage21KVFile(fd: fd, path: path, halfBytes: halfBytes, layers: layers, complete: true,
                                         destination: nil)
            }
            close(fd)
            unlink(path)
        }
        let folder = (path as NSString).deletingLastPathComponent
        let free = (try? URL(fileURLWithPath: folder).resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage ?? 0
        guard free > total + Int64(trailerBytes) + reserve else { return nil }
        let partial = path + ".\(getpid())-\(UUID().uuidString).partial"
        let writing = open(partial, O_RDWR | O_CREAT | O_EXCL, 0o600)
        guard writing >= 0 else { return nil }
        return QwenImage21KVFile(fd: writing, path: partial, halfBytes: halfBytes, layers: layers, complete: false,
                                 destination: (path, key))
    }

    private static func readTrailer(_ fd: Int32, offset: off_t) -> (key: String, layers: Int, bytesPerLayer: Int)? {
        var page = [UInt8](repeating: 0, count: trailerBytes)
        guard page.withUnsafeMutableBytes({ pread(fd, $0.baseAddress!, trailerBytes, offset) }) == trailerBytes,
              Array(page.prefix(5)) == Array("SLKV1".utf8) else { return nil }
        let json = page.dropFirst(5).prefix { $0 != 0 }
        guard let h = (try? JSONSerialization.jsonObject(with: Data(json))) as? [String: Any],
              let key = h["key"] as? String, let layers = h["layers"] as? Int, let bytes = h["bytesPerLayer"] as? Int
        else { return nil }
        return (key, layers, bytes)
    }

    /// **The prefill is done**: the layers to the disk (`fsync`), the trailer after them, the rename to
    /// the kept name — the file is now what the next render with the same key reads. Best effort: a
    /// failure here (a full disk, the cache emptied meanwhile) only loses the keeping; the render
    /// goes on reading its open file.
    package func seal() {
        guard let (target, key) = destination else { return }
        destination = nil
        pending?.group.wait()
        let header = (try? JSONSerialization.data(withJSONObject: ["key": key, "layers": layers, "bytesPerLayer": bytesPerLayer] as [String: Any],
                                                  options: [.sortedKeys])) ?? Data()
        guard 5 + header.count < Self.trailerBytes else { unlink(path); return }
        let page = UnsafeMutableRawPointer.allocate(byteCount: Self.trailerBytes, alignment: Arena.alignment)
        defer { page.deallocate() }
        page.initializeMemory(as: UInt8.self, repeating: 0, count: Self.trailerBytes)
        Array("SLKV1".utf8).withUnsafeBytes { page.copyMemory(from: $0.baseAddress!, byteCount: 5) }
        header.withUnsafeBytes { (page + 5).copyMemory(from: $0.baseAddress!, byteCount: header.count) }
        guard fsync(fd) == 0,
              pwrite(fd, page, Self.trailerBytes, off_t(layers * bytesPerLayer)) == Self.trailerBytes,
              rename(path, target) == 0 else {
            unlink(path)
            return
        }
        path = target
    }

    /// **`seal`, only behind a finite output**: the first evaluation's velocity, which every condition's
    /// K/V reaches (the joint attention reads them all from every target row) — a single non-finite one
    /// makes it non-finite. Otherwise the file is not kept (`deinit` unlinks it) and this render goes on
    /// reading its open file: a kept file is read back by every later render of the same edit, without a
    /// prefill, and would hand them its NaN (`RenderCache.allFinite`). Returns `false` when a file
    /// being kept was given up for a non-finite output.
    @discardableResult
    package func seal(after output: [Float]) -> Bool {
        guard destination != nil else { return true }
        guard RenderCache.allFinite(output) else { return false }
        seal()
        return true
    }

    deinit {
        pending?.group.wait()
        close(fd)
        // A kept file left unsealed (a cancelled render, a failure in the first step) is not kept.
        if destination != nil { unlink(path) }
    }

    /// Both halves of one layer, split into `lanes` page-aligned pieces: `(offset in the file, pointer, bytes)`.
    private func pieces(layer: Int, keys: UnsafeMutablePointer<Float>, values: UnsafeMutablePointer<Float>)
            -> [(offset: off_t, pointer: UnsafeMutableRawPointer, bytes: Int)] {
        let half = bytesPerLayer / 2, pages = half / Arena.alignment, perHalf = lanes / 2
        var out: [(offset: off_t, pointer: UnsafeMutableRawPointer, bytes: Int)] = []
        for (h, base) in [UnsafeMutableRawPointer(keys), UnsafeMutableRawPointer(values)].enumerated() {
            for piece in 0..<perHalf {
                let first = piece * pages / perHalf * Arena.alignment, end = (piece + 1) * pages / perHalf * Arena.alignment
                guard end > first else { continue }
                out.append((off_t(layer * bytesPerLayer + h * half + first), base + first, end - first))
            }
        }
        return out
    }

    private func run(_ write: Bool, layer: Int, keys: UnsafeMutablePointer<Float>, values: UnsafeMutablePointer<Float>,
                     group: DispatchGroup, failure: ErrorBox) {
        let fd = self.fd
        for piece in pieces(layer: layer, keys: keys, values: values) {
            nonisolated(unsafe) let pointer = piece.pointer
            let (offset, bytes) = (piece.offset, piece.bytes)
            DispatchQueue.global(qos: .userInitiated).async(group: group) {
                var done = 0
                while done < bytes {
                    let n = write ? pwrite(fd, pointer + done, bytes - done, offset + off_t(done))
                                  : pread(fd, pointer + done, bytes - done, offset + off_t(done))
                    if n <= 0 { failure.fail(n == 0 ? EIO : errno); return }
                    done += n
                }
            }
        }
    }

    /// Writes layer `layer`'s K/V (`keys`, `values`: `[P, d]` each), and returns once they are on disk.
    package func write(layer: Int, keys: UnsafeMutablePointer<Float>, values: UnsafeMutablePointer<Float>) throws {
        let group = DispatchGroup(), failure = ErrorBox()
        run(true, layer: layer, keys: keys, values: values, group: group, failure: failure)
        group.wait()
        if failure.value != 0 { throw Failure(description: "K/V cache: write failed (errno \(failure.value))") }
    }

    /// Starts reading layer `layer` into `keys`/`values`; `wait` collects it.
    package func prefetch(layer: Int, keys: UnsafeMutablePointer<Float>, values: UnsafeMutablePointer<Float>) {
        pending?.group.wait()
        let group = DispatchGroup(), failure = ErrorBox()
        run(false, layer: layer, keys: keys, values: values, group: group, failure: failure)
        pending = (layer, group, failure)
    }

    /// Layer `layer` in `keys`/`values`: the prefetched one, or read now.
    package func wait(layer: Int, keys: UnsafeMutablePointer<Float>, values: UnsafeMutablePointer<Float>) throws {
        if pending?.layer != layer { prefetch(layer: layer, keys: keys, values: values) }
        guard let (_, group, failure) = pending else { return }
        group.wait()
        pending = nil
        if failure.value != 0 { throw Failure(description: "K/V cache: read failed (errno \(failure.value))") }
    }
}
