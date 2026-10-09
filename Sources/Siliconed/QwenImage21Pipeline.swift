import Accelerate
import Foundation

/// **Qwen-Image-2.1's pipeline** — `QwenImage21Pipeline.__call__` under Viggle's turbo, in fp32: what
/// happens between the request and the DiT, and the loop around it.
///
///     references ─ RGB8 ─ Pillow Lanczos to ~R² (sides ×32) ─┬─ Qwen3-VL (with the prompt) ─ hidden [T, 4096] + slots
///                  (one copy, `preparedReferences`)          └─ VAE encoder (RGBA) ─ reference latents [64, hᵢ, wᵢ]
///     seed ─ torch.randn(1, 1, 64, h, w) (`TorchNoise`) ─ target latent [64, h, w]
///     μ(h·w) ─ Viggle's raw σ nodes, exponential shift, no terminal stretch ─ σ₀ … σₙ₋₁, 0
///
///     prefill once (text + references, K/V kept) ─▶ n × { v = DiT(x, σᵢ);  x += (σᵢ₊₁ − σᵢ)·v }   no CFG
///     9-step mode: 7 steps under the turbo, then the turbo off, a NEW prefill, the last 2 steps
///
/// What the reference does that a port by analogy would miss (read in `pipeline_qwenimage21.py`,
/// diffusers `80c7ed26`, Viggle's README and scheduler, and the oracle's 19 traps):
///
///   - **the σ nodes are raw**: `[1, 0.9375, 0.875, 0.75, 0.5, 0.25]` are passed as `sigmas=` and the
///     scheduler SHIFTS them, `e^μ / (e^μ + (1/σ − 1))`, in **float32** (NumPy: `e^μ` rounded to
///     float32 first). Viggle's scheduler has `shift_terminal: null` — the base one's 0.02 wrecks the
///     last step — and `max_image_seq_len` 8192, `max_shift` 0.9 (not FLUX's 4096 / 1.15);
///   - **μ counts the TARGET's tokens only** (`latents.shape[1]`): the references, prepended to the
///     sequence, do not move the schedule;
///   - the DiT receives `t = σ·1000` (float32) **divided by 1000** — a round trip that does not give
///     σ back to the bit (`QwenImage21DiT.timestep`);
///   - **the noise is drawn `(1, 1, 64, h, w)`** — frame axis before channels — and packed raster
///     `[h·w, 64]`: planar `[64, h, w]` is exactly its draw order (`TorchNoise`);
///   - the scheduler steps in float32, `x + (σ_next − σ)·v`, two roundings (no FMA);
///   - the references are sized by **`calculate_dimensions(R², w/h)`** — round half to EVEN at 32 —
///     with `R = output_resolution`, whatever the output size; one Lanczos (Pillow's, to the bit,
///     `PillowResample`) feeds both the encoder and the VAE;
///   - the output follows the **last** reference without `height`/`width`. **The product's rule is
///     the first** (`outputSize`: the image one edits), at ~R², each side ≥ 512, under `Format`'s
///     ceiling; an explicit size wins. The checks pass the oracle's sizes explicitly;
///   - the VAE returns RGBA; the product writes the RGB planes, alpha dropped (`QwenImage21DecodingModule`);
///   - the turbo is a LoRA on the modulation path too: **the cache depends on it**. The 9-step
///     mode's base tail (`disable_lora()` in Viggle's README) recomputes the prefill.
///
/// ## Memory: what the K/V cache costs, and where it lives
///
/// The cache holds `2 · 32 · P · 4096` floats for `P` condition rows (text + references): 0.52 MB per
/// row, 4.3 GB for one ~1 MP reference, 12.9 GB for three (`QwenImage21CachePolicy`).
package enum QwenImage21Pipeline {
    /// `output_resolution`: the pipeline's default, and the product's — references at ~1 MP.
    package static let outputResolution = 1024
    /// The turbo's step count (Viggle's card) and the 9-step mode's turbo part.
    package static let defaultSteps = 6
    package static let nineStepTurbo = 7

    // ── the schedule ────────────────────────────────────────────────────────────────────

    /// The parts of `FlowMatchEulerDiscreteScheduler`'s config the turbo reads — Viggle's
    /// `scheduler/scheduler_config.json` (`scheduler-turbo.json` in the store). Defaults: as published.
    package struct Scheduler: Equatable, Sendable {
        package var baseSequence = 256, maxSequence = 8192
        package var baseShift = 0.5, maxShift = 0.9
        package var terminal: Double? = nil
        package init() {}

        package init(json: Data) throws {
            guard let c = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
                throw Artifact.Failure.badHeader("scheduler config: not a JSON object")
            }
            guard c["use_dynamic_shifting"] as? Bool ?? true, c["time_shift_type"] as? String ?? "exponential" == "exponential",
                  c["invert_sigmas"] as? Bool != true else {
                throw Artifact.Failure.badHeader("scheduler config: the turbo expects a dynamic exponential shift")
            }
            func number(_ key: String) -> Double? { (c[key] as? NSNumber)?.doubleValue }
            baseSequence = number("base_image_seq_len").map(Int.init) ?? baseSequence
            maxSequence = number("max_image_seq_len").map(Int.init) ?? maxSequence
            baseShift = number("base_shift") ?? baseShift
            maxShift = number("max_shift") ?? maxShift
            terminal = number("shift_terminal")
            // Viggle's README: "the base config's `shift_terminal: 0.02` wrecks the last step".
            guard terminal == nil || terminal == 0 else {
                throw Artifact.Failure.badHeader("scheduler config: shift_terminal \(terminal!) — the turbo's is null")
            }
        }

        package static func file(_ path: String) throws -> Scheduler {
            try Scheduler(json: Data(contentsOf: URL(fileURLWithPath: path)))
        }

        /// `calculate_shift`: linear in the target's token count, extrapolated beyond `maxSequence`.
        package func mu(imageTokens: Int) -> Double {
            let m = (maxShift - baseShift) / Double(maxSequence - baseSequence)
            let b = baseShift - m * Double(baseSequence)
            return Double(imageTokens) * m + b
        }
    }

    /// The step counts Viggle's turbo has a schedule for (and the 9-step mode).
    package static let acceptedSteps = [5, 6, 7, 9]

    /// **Viggle's raw nodes.** 6 steps by default; 5 and 7 add or remove nodes at the high-noise end
    /// only (README, "Rules that matter"); 9 is the 9-step mode, whose last two nodes run without the
    /// turbo. Another count has no published schedule: refused (`QwenImage21Pipeline.Failure.steps`).
    package static func nodes(steps: Int) throws -> [Float] {
        switch steps {
        case 5: return [1, 0.875, 0.75, 0.5, 0.25]
        case 6: return [1, 0.9375, 0.875, 0.75, 0.5, 0.25]
        case 7: return [1, 0.9583, 0.9167, 0.875, 0.75, 0.5, 0.25]
        // `[1.0, 0.9583, 0.9167, 0.875, 0.75, 0.5, 0.25, 1 / 6, 1 / 12]`: Python's `1/6` is a double, made
        // float32 by `np.array(sigmas).astype(np.float32)`.
        case 9: return [1, 0.9583, 0.9167, 0.875, 0.75, 0.5, 0.25, Float(1.0 / 6.0), Float(1.0 / 12.0)]
        default: throw Failure.steps(steps)
        }
    }

    /// The steps that run under the turbo: all but the 9-step mode's last two.
    package static func turboSteps(_ steps: Int) -> Int { steps == 9 ? nineStepTurbo : steps }

    /// **`set_timesteps(sigmas=nodes, mu=μ)`**: the exponential shift in float32 as NumPy does it
    /// (`math.exp(μ)` is a Python float, made float32 against the float32 array), then a zero σ.
    package static func sigmas(nodes: [Float], mu: Double) -> [Float] {
        let e = Float(exp(mu))
        return nodes.map { t in
            let r: Float = 1 / t - 1
            return e / (e + r)
        } + [0]
    }

    /// The full schedule of `steps` steps for a target of `imageTokens` latent cells.
    package static func sigmas(steps: Int, imageTokens: Int, scheduler: Scheduler) throws -> [Float] {
        sigmas(nodes: try nodes(steps: steps), mu: scheduler.mu(imageTokens: imageTokens))
    }

    /// `scheduler.timesteps`: `σ · 1000` in float32 — what the oracle records.
    package static func timesteps(_ sigmas: [Float]) -> [Float] { sigmas.dropLast().map { $0 * 1000 } }

    // ── the sizes ───────────────────────────────────────────────────────────────────────

    /// **The product's output size** for an edit without an explicit size: the first reference's
    /// aspect ratio at ~R² (`calculate_dimensions`, sides at 32), then the product's bounds — each side
    /// ≥ `Format.minimumSide` (raised keeping the ratio), the area ≤ `Format.maxSurface` (the long
    /// side cut). Always a multiple of 32, as the pipeline floors it.
    package static func outputSize(referenceWidth: Int, referenceHeight: Int,
                                   resolution: Int = outputResolution) -> (width: Int, height: Int) {
        let ratio = Double(referenceWidth) / Double(referenceHeight)
        var (w, h) = Qwen3VLImages.conditionSize(width: referenceWidth, height: referenceHeight, resolution: resolution)
        let floor = (Format.minimumSide + 31) / 32 * 32
        if w < floor { w = floor; h = max(floor, Int((Double(w) / ratio / 32).rounded()) * 32) }
        if h < floor { h = floor; w = max(floor, Int((Double(h) * ratio / 32).rounded()) * 32) }
        if w * h > Format.maxSurface {
            if w >= h { w = Format.maxSurface / h / 32 * 32 } else { h = Format.maxSurface / w / 32 * 32 }
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
                return "\(n) steps: Qwen-Image-2.1's turbo has Viggle's schedules for 5, 6 (default) and 7 steps, "
                    + "and the 9-step mode (7 turbo + 2 base)"
            case .imageToImage:
                return "Qwen-Image-2.1 does no img2img (its pipeline has none): pass the image as a reference (--ref) "
                    + "and say what to change"
            }
        }
    }
}

/// **The denoising of one render** — the loop of `__call__` around `QwenImage21DiT`, phase by phase:
/// one phase under the turbo, plus the base tail in the 9-step mode, each with its own DiT (the LoRA
/// stack sizes its arenas) and its own prefill (the cache depends on the stack).
package struct QwenImage21Sampler {
    /// What a step hands back: its absolute index, σᵢ, the seconds, the latent after the update and
    /// the velocity, both planar `[64, h, w]`.
    package typealias Step = (index: Int, sigma: Float, seconds: Double, latent: UnsafePointer<Float>,
                              velocity: UnsafePointer<Float>)

    package let artifact: Artifact
    /// The stack of the turbo steps (the turbo, then the user's LoRAs) and of the base tail (the
    /// user's alone, `nil` without).
    package let turbo: LoRA?
    package let base: LoRA?
    /// Where the K/V cache lives: `nil`, the policy's choice for the render's prefix
    /// (`QwenImage21CachePolicy`); forced by the checks that measure the three.
    package var storage: QwenImage21DiT.CacheStorage?
    /// The storage the last `run` used.
    package private(set) var used: QwenImage21DiT.CacheStorage = .memory
    /// **The render cache's K/V files, one per phase** (`RenderCache.kvPath`): when the conditions
    /// go to a file, it is this one — read back without a prefill if a previous render left it whole,
    /// kept for the next one otherwise. `nil`: the unlinked file, gone with the render.
    package var kept: [(path: String, key: String)]?
    /// What a kept file must leave free on its volume (`RenderCache.keptReserve`).
    package var keptReserve: Int64 = RenderCache.defaultKeptReserve
    package var freezeCut = EngineSettings.effective.frozenCut
    package var cancellation: Cancellation?
    /// Seconds per phase: `prefill`, `steps`, and the DiT's own breakdown (`dit …`).
    package private(set) var timings: [String: Double] = [:]
    /// **The DiT of the previous image of a batch, reused** when the run has one phase (no user LoRA):
    /// same sequence, stack and storage, so building it again would only be a fresh allocation wave
    /// (`Context.keptDenoiser`). With two phases each image already alternates two DiTs; nothing is kept.
    package var reusable: QwenImage21DiT?
    /// The DiT a one-phase run used, for the next image; `nil` after a two-phase run.
    package private(set) var reused: QwenImage21DiT?

    package init(artifact: Artifact, turbo: LoRA?, base: LoRA?) {
        self.artifact = artifact; self.turbo = turbo; self.base = base
    }

    /// - Parameters:
    ///   - latent: the target `[64, h, w]`, updated in place.
    ///   - hidden: the encoder's `[T, 4096]`; `slots`: its `image_pad_mask` (`T` entries).
    ///   - references: each `[64, hᵢ, wᵢ]`, in prompt order.
    ///   - sigmas: the full schedule (`steps + 1` values); `turboSteps`: how many run under `turbo`.
    ///   - told: the σ the DiT is told at each step (`Detail.modelSigmas`); `nil`: `sigmas`.
    package mutating func run(latent: inout [Float], height: Int, width: Int,
                              hidden: [Float], slots: [Bool], references: [Latent],
                              sigmas: [Float], told: [Float]? = nil, turboSteps: Int,
                              onStep: (Step) throws -> Void) throws {
        let steps = sigmas.count - 1
        let told = told ?? sigmas
        let target = height * width
        guard height % 2 == 0, width % 2 == 0 else {
            throw Artifact.Failure.misuse("Qwen-Image-2.1: a target grid \(width)×\(height) not made of 2×2 slots")
        }
        let sequence = try QwenImage21Sequence(
            slots: slots + [Bool](repeating: true, count: target / 4),
            images: references.map { .init(height: $0.height, width: $0.width) } + [.init(height: height, width: width)])
        let phases = [(0..<min(turboSteps, steps), turbo), (min(turboSteps, steps)..<steps, base)].filter { !$0.0.isEmpty }
        timings = [:]
        for (phase, (range, lora)) in phases.enumerated() {
            try cancellation.check()
            // Its own scope: the previous phase's DiT (arenas, cache, LoRA) is gone before this one asks.
            used = storage ?? QwenImage21CachePolicy.storage(prefix: sequence.prefix)
            let file = kept.flatMap { phase < $0.count ? $0[phase] : nil }
            let dit = try (phases.count == 1 ? reusable : nil)
                ?? QwenImage21DiT(artifact: artifact, sequence: sequence, lora: lora, storage: used,
                                  kept: file.map { ($0.path, $0.key, keptReserve) }, freezeCut: freezeCut)
            reused = phases.count == 1 ? dit : nil
            if dit.reusedPrefill { timings["kv reused", default: 0] += 1 }
            if dit.keepsPrefill { timings["kv kept", default: 0] += 1 }
            dit.cancellation = cancellation
            try dit.tabulateModulation(sigmas: Array(told[range]))
            var started = Date()
            let pointers = references.map { reference -> UnsafeMutablePointer<Float> in
                let p = UnsafeMutablePointer<Float>.allocate(capacity: reference.values.count)
                p.update(from: reference.values, count: reference.values.count)
                return p
            }
            defer { pointers.forEach { $0.deallocate() } }
            try hidden.withUnsafeBufferPointer {
                // The conditions' layers run inside the first step (`QwenImage21DiT.prefillPending`): its
                // seconds include them, `prefill` keeps only `txt_in` and `img_in`.
                try dit.prefill(hidden: $0.baseAddress!, references: pointers.map { UnsafePointer($0) }, intoFirstStep: true)
            }
            timings["prefill", default: 0] += Date().timeIntervalSince(started)
            for i in range {
                try cancellation.check()
                started = Date()
                let v = try latent.withUnsafeBufferPointer { try dit.forward(latent: $0.baseAddress!, sigma: told[i]) }
                // `scheduler.step`: `dt = σ_next − σ` in float32, then `x + dt·v` — two roundings.
                let dt = sigmas[i + 1] - sigmas[i]
                latent.withUnsafeMutableBufferPointer { x in
                    for j in 0..<x.count { x[j] += dt * v[j] }
                }
                let seconds = Date().timeIntervalSince(started)
                timings["steps", default: 0] += seconds
                try latent.withUnsafeBufferPointer { x in
                    try v.withUnsafeBufferPointer { try onStep((i, sigmas[i], seconds, x.baseAddress!, $0.baseAddress!)) }
                }
            }
            for (name, seconds) in dit.timings { timings["dit " + name, default: 0] += seconds }
        }
    }
}

/// **The LoRA stack expanded one layer ahead, not all at once.** `CacheLoRA` keeps the whole stack in
/// fp32 for the render: Viggle's rank-256 turbo on 224 modules is 2.68 GB resident, the largest single
/// item of the denoising stage. Here only two layers live (2 × 84 MB): layer `l` being read by the GPU
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
        let page = Arena.alignment
        func rounded(_ b: Int) -> Int { (b + page - 1) / page * page }
        var offset = 0
        for suffix in shapes.keys.sorted() {
            let (k, n) = shapes[suffix]!, r = lora.rank(blockPrefix + "0." + suffix)
            layout[suffix] = (offset, offset + rounded(k * r * 4), k, n)
            offset += rounded(k * r * 4) + rounded(r * n * 4)
        }
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

    package static func storage(prefix: Int) -> QwenImage21DiT.CacheStorage {
        bytes(prefix: prefix) <= memoryBudget ? .memory : .file
    }
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
