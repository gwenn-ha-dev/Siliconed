import Foundation
import Siliconed

// **img2img, one stage for each family that starts from an image** — `siliconed check <model> img2img`.
//
//     picture [3, H, W] ─ the product's encoder ─ z₀ ──────────┐
//     seed ─ ε (`torch.randn`, bit for bit) ───────────────────┼─ x_s = σ_s·ε + (1 − σ_s)·z₀ ─ steps t_start … N − 1
//     strength ─ the product's admission ─ t_start, σ_s = σ[t_start] ┘
//
// The oracle (`tools/oracle/<model>.py img2img`) runs diffusers' img2img on the reference's modules: the picture of the
// golden encoded to its posterior mean, `get_timesteps` over the full schedule, `scale_noise`, then the loop from `x_s`.
// The engine is judged at each piece, in the order a render meets them:
//
//   - the seed's noise, bit for bit, and the schedule;
//   - **the request through the product's admission** (`Admission.judge`): its `t_start` must be diffusers', its σ_s
//     within 1e-6 — the strength's semantics are diffusers' exactly (`Strength`);
//   - **the start's mix, bit for bit**: `Latent.start` on the ORACLE's z₀, the noise and σ_s must give the oracle's own
//     `x_s` to the bit — the same fp32 products and sum as `scale_noise`;
//   - the product's encoder on the golden's picture (z₀, per latent plane), then the start it makes (x_s);
//   - each step t_start … N − 1 from the oracle's latent at that step, the velocity judged as the family's trajectory
//     judges it (at the step's fp64 witness where the oracle wrote one; on Krea 2, a step failing there is judged
//     again by clouds of summation orders, `Clouds`, when the oracle drew that step's orders: `img2img-draw`);
//   - **the product's denoiser run free from the engine's own start**: it must evaluate exactly σ[t_start] … σ[N − 1]
//     (judged); its final latent against the oracle's is shown with its gain, not judged — the trajectory amplifies.
//
// Not judged here, by design: the preprocessing (`ImageRGB.fitted`, the app's centered crop — diffusers stretches) and
// the encoder's mean instead of diffusers' sample (`VAEEncoder`, the engine's deterministic choice): the oracle takes the
// mean too, and hands the engine its picture already at the format.
//
// Qwen-Image-2.1 has no stage: its denoiser refuses a start image (`QwenImage21Pipeline.Failure.imageToImage`).

enum Img2ImgStages {
    static let zImage = stage(Family(
        model: "z-image", golden: "goldens-img2img", family: .zImage, space: .flux, ditMap: ZImageStages.ditMap,
        oracle: "tools/oracle/z-image.py img2img --size {size}",
        chain: { map in
            let vae = try Library.processWide.component(.zImage, "vae.safetensors")
            return try Chain(text: ZImageTextModule(map: "", tokenizer: ""), denoising: ZImageDenoisingModule(map: map),
                             decoding: FluxDecodingModule(path: vae), encoding: FluxEncodingModule(path: vae))
        },
        conditioning: { goldens in
            let t = try goldens.open("goldens-text.safetensors")
            guard let caps = t.float32("cap_feats"), let e = t.entries["cap_feats"] else { throw Skip("goldens-text: cap_feats missing") }
            let rows = e.shape.count == 3 ? e.shape[1] : e.shape[0]
            return Conditioning(format: .zImage, rows: rows, width: caps.count / rows, values: Array(caps), tokens: rows)
        },
        guided: { route in
            // The sampler, as the trajectory stage runs it: the oracle's latent laid back after each step.
            let dit = try DiT(artifact: try Artifact(path: route.map), latentHeight: route.height, latentWidth: route.width,
                              capLength: route.text.rows)
            let sampler = Sampler(dit: dit, steps: route.sigmas.count - 1)
            sampler.recordTrajectory = true
            guard var current = route.oracle(route.start) else { return [:] }
            try current.withUnsafeMutableBufferPointer { l in
                let (buffer, n) = (l.baseAddress!, l.count)
                _ = try route.text.values.withUnsafeBufferPointer { caps in
                    try sampler.run(latent: buffer, count: n, caps: caps.baseAddress!, latentHeight: route.height,
                                    latentWidth: route.width, start: route.start) { i, _, _ in
                        if let next = route.oracle(i + 1) { buffer.update(from: next, count: n) }
                    }
                }
            }
            return Dictionary(uniqueKeysWithValues: sampler.velocities.enumerated().map { (route.start + $0.offset, $0.element) })
        },
        judge: { name, engine, oracle, witness, columns, _ in
            Judge.boundary(name, engine: engine, oracle: oracle, witness: witness, columns: columns)
        }))

    static let anima = stage(Family(
        model: AnimaStages.model, golden: "anima/goldens-anima-img2img", family: .anima, space: .qwenImage,
        ditMap: AnimaStages.ditMap, oracle: "tools/oracle/anima.py img2img --size {size}",
        chain: { map in
            let vae = try Library.processWide.component(.anima, "vae.safetensors")
            return try Chain(text: AnimaTextModule(map: "", tokenizerQwen: "", tokenizerT5: "", adapter: ""),
                             denoising: AnimaDenoisingModule(map: map),
                             decoding: QwenImageDecodingModule(path: vae), encoding: QwenImageEncodingModule(path: vae))
        },
        conditioning: { goldens in
            let t = try goldens.open("anima/goldens-anima-text.safetensors")
            guard let embeds = t.float32("prompt_embeds"), let e = t.entries["prompt_embeds"] else {
                throw Skip("goldens-anima-text: prompt_embeds missing")
            }
            return Conditioning(format: .anima, rows: e.shape[0], width: e.shape[1], values: Array(embeds), tokens: e.shape[0])
        },
        guided: { route in
            let dit = try AnimaDiT(artifact: try Artifact(path: route.map), latentHeight: route.height, latentWidth: route.width,
                                   textLength: route.text.rows)
            return try route.steps { latent, i in
                try latent.withUnsafeBufferPointer { l in
                    try route.text.values.withUnsafeBufferPointer {
                        try dit.forward(latent: l.baseAddress!, context: $0.baseAddress!, sigma: route.sigmas[i])
                    }
                }
            }
        },
        judge: { name, engine, oracle, witness, columns, _ in
            Judge.boundary(name, engine: engine, oracle: oracle, witness: witness, columns: columns)
        }))

    static let krea2 = stage(Family(
        model: Krea2Stages.model, golden: "krea2/goldens-krea2-img2img", family: .krea2, space: .qwenImage,
        ditMap: Krea2Stages.ditMap, variants: true,
        oracle: "tools/oracle/krea2.py img2img --size {size}; img2img-witness --size {size} --step 3 4 5 6 7; img2img-draw --size {size} --step i --permute 1 2 3 4 5 6 7 8 for a step i failing at its witness; with --variant compact, the same with --variant \(Wording.publishers)",
        chain: { map in
            let vae = try Library.processWide.component(.krea2, "vae.safetensors")
            return try Chain(text: Krea2TextModule(map: "", tokenizer: "", sockets: Krea2Text.sockets),
                             denoising: Krea2DenoisingModule(map: map),
                             decoding: QwenImageDecodingModule(path: vae), encoding: QwenImageEncodingModule(path: vae))
        },
        conditioning: { goldens in
            // The 12 taps of the valid rows: the product's denoiser fuses them itself.
            let t = try goldens.open("krea2/goldens-krea2-text.safetensors")
            guard let hidden = t.float32("hidden_valid"), let e = t.entries["hidden_valid"] else {
                throw Skip("goldens-krea2-text: hidden_valid missing")
            }
            return Conditioning(format: .krea2, rows: e.shape[0], width: hidden.count / e.shape[0], values: Array(hidden),
                                tokens: e.shape[0])
        },
        guidedText: { goldens, version in
            // The guided steps read the oracle's fused text (`txt_in`), as the trajectory stage does — the version's:
            // `txt_in` is a DiT layer, and a Compact's differs from the Standard's (|Δ| 4e-5 of the largest value).
            let stem = variantStem("krea2/goldens-krea2-dit-512", version)
            let dit512 = try goldens.open(stem + ".safetensors")
            guard let text = dit512.float32("txt_in_out"), let e = dit512.entries["txt_in_out"] else {
                throw Skip("\(stem): txt_in_out missing")
            }
            return Conditioning(format: .krea2, rows: e.shape[0], width: e.shape[1], values: Array(text), tokens: e.shape[0])
        },
        guided: { route in
            let dit = try Krea2DiT(artifact: try Artifact(path: route.map), latentHeight: route.height, latentWidth: route.width,
                                   maxText: route.text.rows, accelerators: route.accelerators, plan: route.plan)
            // Like the product: the modulation of the σ evaluated, in one pass, before the first step.
            try dit.tabulateModulation(sigmas: Array(route.sigmas[route.start..<(route.sigmas.count - 1)]))
            return try route.steps { latent, i in
                try latent.withUnsafeBufferPointer { l in
                    try route.text.values.withUnsafeBufferPointer {
                        try dit.forward(latent: l.baseAddress!, text: $0.baseAddress!, textRows: route.text.rows, sigma: route.sigmas[i])
                    }
                }
            }
        },
        witness: { goldens, size, version, step in
            goldens.optional(variantStem("krea2/witness-krea2-f64-img2img-\(size)", version) + "-step\(step).safetensors")?
                .float32("velocity")
        },
        judge: { name, engine, oracle, witness, columns, _ in
            Judge.boundary(name, engine: engine, oracle: oracle, witness: witness, columns: columns)
        },
        rejudge: { route, goldens, size, version, guided in
            // The trajectory's clouds (`Clouds`): a step failing at its witness, with the oracle's order draws of it
            // (`draw-krea2-img2img-R-step<i>-permgroupN`), is judged by the engine's draws of the same seeds — one map
            // permuted per seed (`Krea2PermutedMap.draws`, seed outside), every failing step evaluated on it.
            let stem = variantStem("krea2/draw-krea2-img2img-\(size)", version)
            var drawn: [Int: [UInt64: [String: [Float]]]] = [:]
            var clouds = guided.map { g in
                Clouds(draws: [], orders: Clouds.orders(goldens, stem + "-step\(g.step)"), engineOrder: { seed in
                    guard let v = drawn[g.step]?[seed] else { throw Artifact.Failure.misuse("step \(g.step): order \(seed) not drawn") }
                    return v
                }) { _ in [:] }
            }
            let ordered = guided.indices.filter { clouds[$0].judgesByOrders(guided[$0].measure, exact: guided[$0].exact, as: "model_out") }
            let text = route.text
            let orders = try text.values.withUnsafeBufferPointer { t in
                try Krea2PermutedMap.draws(source: route.map, requests: ordered.map { clouds[$0].orders.map(\.seed) }, text: t) {
                    k, map, moved in
                    let g = guided[ordered[k]]
                    let dit = try Krea2DiT(artifact: map, latentHeight: route.height, latentWidth: route.width, maxText: text.rows,
                                           accelerators: route.accelerators, plan: route.plan)
                    return ["model_out": try g.input.withUnsafeBufferPointer {
                        try dit.forward(latent: $0.baseAddress!, text: moved.baseAddress!, textRows: text.rows, sigma: route.sigmas[g.step])
                    }]
                }
            }
            for (k, i) in ordered.enumerated() { drawn[guided[i].step] = orders[k] }
            return try guided.indices.map {
                try clouds[$0].arbitrate(guided[$0].measure, exact: guided[$0].exact, columns: route.height * route.width, as: "model_out")
            }
        }))

    static let mossImage = stage(Family(
        model: MossImageStages.model, golden: "moss/goldens-moss-img2img", family: .mossImage, space: .flux2,
        ditMap: MossImageStages.ditMap, variants: true,
        oracle: "tools/oracle/moss-image.py img2img --size {size}; with --variant compact, the same with --variant \(Wording.publishers)",
        chain: { map in
            let vae = try Library.processWide.component(.mossImage, "vae.safetensors")
            return try Chain(text: MossImageTextModule(map: "", tokenizer: ""),
                             denoising: MossImageDenoisingModule(map: map, turbo: try turboMap()),
                             decoding: Flux2DecodingModule(path: vae), encoding: Flux2EncodingModule(path: vae))
        },
        conditioning: { goldens in
            let dit512 = try goldens.open("moss/goldens-moss-dit-512.safetensors")
            guard let text = dit512.float32("text_in"), let e = dit512.entries["text_in"] else {
                throw Skip("goldens-moss-dit-512: text_in missing")
            }
            return Conditioning(format: .mossImage, rows: e.shape[0], width: e.shape[1], values: Array(text), tokens: e.shape[0])
        },
        guided: { route in
            let module = MossImageDenoisingModule(map: route.map, turbo: try turboMap())
            let dit = try MossImageDiT(artifact: try Artifact(path: route.map), gridHeight: route.height, gridWidth: route.width,
                                       maxText: route.text.rows, accelerators: route.accelerators, lora: try module.stack(nil),
                                       plan: route.plan)
            let timesteps = MossImageSchedule.timesteps(route.sigmas)
            try dit.tabulateModulation(timesteps: timesteps)
            return try route.steps { latent, i in
                try latent.withUnsafeBufferPointer { l in
                    try route.text.values.withUnsafeBufferPointer {
                        try dit.forward(latent: l.baseAddress!, text: $0.baseAddress!, textRows: route.text.rows, timestep: timesteps[i])
                    }
                }
            }
        },
        judge: { name, engine, oracle, witness, _, planes in
            Judge.planes(name + " per plane", engine: engine, oracle: oracle, witness: witness, planes: planes)
        }))

    /// The turbo LoRA's map, the one every MOSS-Image render opens.
    private static func turboMap() throws -> String {
        let path = Installer(library: .processWide).companionMap(.mossImage) ?? ""
        guard FileManager.default.fileExists(atPath: path) else { throw Skip("LoRA map missing: \(path) — siliconed install moss-image") }
        return path
    }

    // MARK: - What differs between families

    /// What one family hands the common run: its golden, its product chain, its text, its DiT on a guided step.
    struct Family: Sendable {
        let model: String
        /// The golden's stem without its size: `krea2/goldens-krea2-img2img` → `…-img2img-512`, `…-img2img-compact-512`.
        let golden: String
        let family: Siliconed.Family
        let space: LatentSpace
        let ditMap: String
        /// Whether `--variant compact|light` exists: the oracle writes a variant's golden from the Standard's start.
        var variants = false
        let oracle: String
        /// The product's chain on a DiT map: its admission, its encoder and its denoiser are what a render runs.
        let chain: @Sendable (String) throws -> Chain
        /// The golden text as the product's denoiser takes it (its free pass).
        let conditioning: @Sendable (Goldens) throws -> Conditioning
        /// The text the guided steps' DiT reads, when it is not the product's (Krea 2: the version's fused text).
        var guidedText: (@Sendable (Goldens, Variant?) throws -> Conditioning)? = nil
        /// The velocities of steps `start …`, each from the oracle's latent at that step.
        let guided: @Sendable (Route) throws -> [Int: [Float]]
        /// A step's fp64 witness, where the oracle wrote one.
        var witness: (@Sendable (Goldens, String, Variant?, Int) -> UnsafeBufferPointer<Float>?)? = nil
        /// How a guided velocity is judged: the family's trajectory rule (`columns` = h·w, `planes` = the latent's channels).
        let judge: @Sendable (String, [Float], UnsafeBufferPointer<Float>, UnsafeBufferPointer<Float>?, Int, Int) -> Measure
        /// The guided verdicts judged again where the family's trajectory rule does (Krea 2: clouds of summation orders for
        /// a step failing at its witness); without it, each single verdict stands. Given the route, the goldens, the size.
        var rejudge: (@Sendable (Route, Goldens, String, Variant?, [Guided]) throws -> [Measure])? = nil
    }

    /// One guided step's verdict, with what a rejudging needs: its fp64 witness and the oracle's latent it started from.
    struct Guided {
        let step: Int
        let measure: Measure
        let exact: UnsafeBufferPointer<Float>?
        let input: [Float]
    }

    /// What a guided pass is given: the map, the grid, the schedule and the start, the text, the oracle's latents, the route.
    struct Route {
        let map: String
        let height: Int, width: Int
        let sigmas: [Float]
        let start: Int
        let text: Conditioning
        /// The oracle's latent entering step `i` (its `x_s` at `start`).
        let oracle: (Int) -> [Float]?
        let accelerators: Accelerators
        let plan: CapacityPlan?

        /// One evaluation per step `start … N − 1`, each on the oracle's latent.
        func steps(_ evaluate: ([Float], Int) throws -> [Float]) throws -> [Int: [Float]] {
            var velocities: [Int: [Float]] = [:]
            for i in start..<(sigmas.count - 1) {
                guard let latent = oracle(i) else { continue }
                velocities[i] = try evaluate(latent, i)
            }
            return velocities
        }
    }

    // MARK: - The stage

    static func stage(_ f: Family) -> Stage {
        Stage(model: f.model, name: "img2img",
              proves: "the start of an img2img at the oracle's strength (0.6): the seed's noise and the schedule, t_start and σ_s through the product's admission, the start's mix bit for bit, the encoder's z₀ and x_s, each step's velocity from the oracle's latent, then the product's denoiser from its own start over exactly the steps diffusers runs",
              options: f.variants ? ["--map", "--variant"] : ["--map"], oracle: f.oracle, weights: ["VAE", "DiT"],
              rule: "noise and the mix from the oracle's z₀ bit for bit; t_start equal, σ_s within 1e-6; z₀ and x_s per latent plane, floor 5e-4; each guided velocity by the family's trajectory rule (floor 5e-4, at the step's fp64 witness where it exists; Krea 2: clouds of summation orders when the oracle drew the failing step's); the product's steps exactly σ[t_start] … σ[N − 1]; its free final latent shown, not judged"
                + (f.variants ? "; with --variant, the Standard's start through the version's DiT, along the int8 route when it publishes int8 and once widened" : "")) { invocation in
            try run(f, invocation)
        }
    }

    static func run(_ f: Family, _ invocation: Invocation) throws -> Report {
        let version = f.variants ? try invocation.quantizedVariant() : nil
        let stem = variantStem("\(f.golden)-\(invocation.size)", version)
        let report = Report(title: "\(f.model) img2img\(version.map { " " + $0.rawValue } ?? "") \(invocation.size)")
        let golden = try invocation.goldens.open(stem + ".safetensors")
        let manifest = try invocation.goldens.manifest(stem)
        guard let seed = (manifest["seed"] as? NSNumber)?.uint64Value, let strength = (manifest["strength"] as? NSNumber)?.doubleValue,
              let imageEntry = golden.entries["image"], let image = golden.float32("image"),
              let epsEntry = golden.entries["eps"], let eps = golden.float32("eps"), let z0Oracle = golden.float32("z0"),
              let xOracle = golden.float32("x_s"), let sigmas = golden.float32("sigmas"), let tStart = golden.float32("t_start"),
              let sigmaS = golden.float32("sigma_s"), let final = golden.float32("latent_final") else {
            throw Skip("\(golden.path): the img2img's inputs (seed, strength, image, eps, z0, x_s, sigmas, t_start, sigma_s, latent_final) are missing")
        }
        let (channels, h, w) = (epsEntry.shape[0], epsEntry.shape[1], epsEntry.shape[2])
        let (height, width) = (imageEntry.shape[1], imageEntry.shape[2])
        let map = try invocation.ditMap(f.ditMap, family: f.family, version: version)
        let chain = try f.chain(map)
        let denoising = chain.denoising, steps = denoising.defaultSteps
        guard let encoding = chain.encoding else { throw Skip("\(f.model): the chain has no image encoder") }

        // ── the noise and the schedule ─────────────────────────────────────────────────────────────────────
        let noise = Latent.noise(f.space, height: h, width: w, seed: seed)
        report.add(Judge.identical("noise (seed \(seed))", noise.values, eps))
        let σ = denoising.sigmas(steps: steps)
        report.add(Judge.schedule(σ, sigmas))

        // ── the request through the product's admission: t_start and σ_s are diffusers' ─────────────────────
        let picture = ImageRGB(pixels: Array(image), height: height, width: width)
        let request = Request("img2img", width: width, height: height, seed: seed, image: picture, strength: strength)
        let admitted = try Admission.judge(request, chain: chain, usesANE: false)
        let start = admitted.start
        report.add(Judge.holds("t_start", start == Int(tStart[0]),
                               note: "engine \(start), diffusers \(Int(tStart[0])); \(admitted.plan.evaluations) evaluations at strength \(strength)"))
        report.add(Judge.bound("σ_s", Double(abs(admitted.plan.startSigma - sigmaS[0])), 1e-6, unit: "|Δ|",
                               note: String(format: "%.6f", admitted.plan.startSigma)))

        // ── the start: the mix bit for bit on the oracle's z₀, then the product's encoder and its own start ──
        let z0 = Latent(space: f.space, height: h, width: w, values: Array(z0Oracle))
        report.add(Judge.identical("x_s from the oracle's z₀", Latent.start(image: z0, noise: noise, sigma: admitted.plan.startSigma).values,
                                   xOracle))
        let encoded = try encoding.encoder(picture, context: Context(onProgress: nil))
        report.add(Judge.planes("z₀ (the encoder) per plane", engine: encoded.values, oracle: z0Oracle, planes: channels))
        let ours = Latent.start(image: encoded, noise: noise, sigma: admitted.plan.startSigma)
        report.add(Judge.planes("x_s per plane", engine: ours.values, oracle: xOracle, planes: channels))

        // ── the guided steps: each from the oracle's latent, judged by the family's trajectory rule ──────────
        let text = try f.guidedText?(invocation.goldens, version) ?? f.conditioning(invocation.goldens)
        let oracle = { (i: Int) -> [Float]? in i == start ? Array(xOracle) : golden.float32("latent_\(i)").map(Array.init) }
        let int8 = version == nil ? nil : Int8Route(try Artifact(path: map), invocation)
        let route = { (plan: CapacityPlan?, neuralEngine: Bool) in
            Route(map: map, height: h, width: w, sigmas: σ, start: start, text: text, oracle: oracle,
                  accelerators: .on(neuralEngine: neuralEngine), plan: plan)
        }
        let judgeSteps = { (r: Route, suffix: String) throws in
            let velocities = try f.guided(r)
            var guided: [Guided] = []
            for i in start..<(σ.count - 1) {
                guard let expected = golden.float32("velocity_\(i)"), let v = velocities[i], let input = oracle(i) else {
                    report.add(Judge.missing("v_\(i)\(suffix)")); continue
                }
                let exact = f.witness?(invocation.goldens, invocation.size, version, i)
                guided.append(Guided(step: i, measure: f.judge("v_\(i)\(suffix)", v, expected, exact, h * w, channels),
                                     exact: exact, input: input))
            }
            for m in try f.rejudge?(r, invocation.goldens, invocation.size, version, guided) ?? guided.map(\.measure) { report.add(m) }
        }
        if let int8 {
            try judgeSteps(route(int8.plan, int8.neuralEngine), "")
            int8.account(report)
            if int8.judgesWidened { try judgeSteps(route(nil, EngineSettings.effective.plan.usesANE), " (widened)") }
        } else {
            try judgeSteps(Route(map: map, height: h, width: w, sigmas: σ, start: start, text: text, oracle: oracle,
                                 accelerators: .process, plan: nil), "")
        }

        // ── the product's denoiser, free from its own start: exactly the steps diffusers runs ────────────────
        var latent = ours
        let told = Told()
        let context = Context(onProgress: { event in
            if case let .step(_, _, _, sigma, _, _, _) = event { told.add(sigma) }
        })
        try denoising.denoise(&latent, text: try f.conditioning(invocation.goldens), steps: steps, start: start, lora: nil, context: context)
        let expected = Array(σ[start..<(σ.count - 1)]), evaluated = told.sigmas
        let same = evaluated.count == expected.count && zip(evaluated, expected).allSatisfy { abs($0 - $1) <= 1e-6 }
        report.add(Judge.holds("the product's steps", same,
                               note: "\(evaluated.count) evaluations, σ " + evaluated.map { String(format: "%.4f", $0) }.joined(separator: " ")))
        let drift = errorPerPlane(latent.values, final, planes: channels)
        print(String(format: "free pass (the product's denoiser from its own start): final latent against the oracle's, worst plane %.2e, median %.2e — shown, not judged",
                     drift.worst, drift.median))
        return report
    }

    /// The σ the product's denoiser reports, in order.
    private final class Told: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [Float] = []
        func add(_ sigma: Float) { lock.withLock { values.append(sigma) } }
        var sigmas: [Float] { lock.withLock { values } }
    }
}
