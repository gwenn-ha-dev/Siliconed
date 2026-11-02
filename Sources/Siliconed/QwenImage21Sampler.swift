import Accelerate
import Foundation

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
    /// What the DiT's products may compute on beyond fp32 (`Request.accelerators`, resolved).
    package var accelerators = Accelerators.process
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
                                  accelerators: accelerators)
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
