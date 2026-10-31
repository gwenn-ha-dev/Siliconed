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
