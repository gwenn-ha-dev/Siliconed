import Foundation
import Siliconed

// **Krea 2 Turbo, stage by stage** — `siliconed check krea2 <stage>`.
//
//     text → dit → trajectory → vae
//
// What sets it apart: the text is not one encoder's output but **twelve taps** of Qwen3-VL (layers
// spread over its depth), fused by refiners and `txt_in`; the reference pads the prompt in the middle
// and masks, the engine removes the padding — the taps are judged against states computed WITH it. The
// VAE is Qwen-Image's, with Krea's weights (Anima's to the bit).
//
// **Clouds.** Some boundaries are ill-conditioned in fp32: block 0's text rows amplify a rounding about
// a thousandfold, and one fp32 draw against one says which summation order the engine took, not how
// close it is. When the single verdict at the witness fails and the oracle has **draws** of the same
// inputs, the engine draws the same way: its median no further from the exact than the oracle's worst
// draw, + 25 %. Without draws, the single verdict stands.
//
// **Two families of draws, and the orders judge where they exist.** The first moves the latent by one ulp
// (`krea2.py draw --jitter N`): the exact value moves by ~10⁻⁷ and the witness still judges, but every product
// still sums its terms in the one order its BLAS or kernel chose, so the cloud samples the inputs, not the
// arithmetic. The second draws the arithmetic itself (`--permute N --permute-kind group`, `Krea2PermutedMap`):
// the residual stream's channels reordered everywhere at once, by whole groups of 256 — the same model, the
// same exact result, its sums over the hidden dimension in another order. Groups of 256 because it is the only
// family a convrot map admits: its rotation is one Hadamard per group, which a permutation of groups commutes
// with and nothing finer does; the oracle and the engine then draw the same seeds, so the same orders.
// Measured on the Compact at 512², `block27_out` against fp64 (·10⁻⁴): the oracle's golden is 4.60, the low
// tail of its own orders (4.96–6.18 over seeds 1–8), and its identity taken by another loading path gives 6.27;
// the engine's widened route spans 5.00–7.02 (median 5.69), its int8 route 5.05–6.90 (median 6.03). The two
// routes overlap, int8 − widened per seed +0.27 ± 0.25: the order, not the route, moved the int8 past the
// ulp clouds' bar (6.35, which the widened route itself fails on one order in eight). So when the oracle has
// order draws of a boundary, the engine draws the same seeds (its nominal + 8 orders, the median judged), and
// the oracle's cloud is its nominal, its ulp draws and its orders; the ulp moves judge otherwise. The same rule
// holds for each guided step of `trajectory`, whose velocity the oracle draws in orders at every step: there the
// permuted map is the Standard's, ~26 GB rewritten per draw — paid only when a step fails at its witness, and drawn
// seed outside: one map per seed, on which every failing step is evaluated, 8 maps at worst whatever their number.

enum Krea2Stages {
    static let all: [Stage] = [text, dit, trajectory, vae]

    static let model = "krea2"
    static let ditMap = "krea2-turbo-dit.v0.silicon"

    static let text = Stage(
        model: model, name: "text",
        proves: "the prompt's identifiers, the 12 Qwen3-VL taps (padding removed, judged against states computed with it), then the fusion and txt_in on the oracle's taps (with --long, a prompt past 512 tokens read whole, as the app admits it: no fusion golden for it)",
        options: ["--long"], oracle: "tools/oracle/krea2.py text [--long]", weights: ["text encoder", "DiT (the fusion)"],
        rule: "identifiers bit for bit; the rule, floor 5e-4"
    ) { invocation in
        let stem = invocation.long ? "krea2/goldens-krea2-text-long" : "krea2/goldens-krea2-text"
        let report = Report(title: "\(model) text\(invocation.long ? " --long" : "")")
        let goldens = try invocation.goldens.open(stem + ".safetensors")
        guard let prompt = try invocation.goldens.manifest(stem)["prompt"] as? String,
              let paddedIds = goldens.int32("ids"), let mask = goldens.int32("mask"),
              let validEntry = goldens.entries["hidden_valid"], let oracleHidden = goldens.float32("hidden_valid") else {
            throw Skip("\(goldens.path): prompt, ids, mask or hidden_valid missing")
        }
        // The reference pads in the middle and masks: its valid tokens are the engine's whole prompt.
        let valid = zip(paddedIds, mask).filter { $0.1 != 0 }.map { $0.0 }
        // `identifiers` keeps the reference's default cut; past it, the app's admission is what a render reads.
        let tokenizer = try Tokenizer(directory: try Library.processWide.component(.krea2, "tokenizer"))
        let ids = invocation.long ? try Krea2Text.admitted(prompt, tokenizer: tokenizer) : Krea2Text.identifiers(prompt, tokenizer: tokenizer)
        let tokens = valid.withUnsafeBufferPointer { Judge.integers("tokens", ids, $0) }
        report.add(tokens)
        guard tokens.passed else { return report }

        let sockets = Krea2Text.sockets, width = 2560, t = validEntry.shape[0]
        let hidden = try Krea2Text.hiddenStates(ids, encoder: MapCheck.use(Library.processWide.map("qwen3-vl-4b-encoder.v0.silicon")),
                                                sockets: sockets)
        report.add(goldens, "Qwen3-VL taps", hidden, name: "hidden_valid", columns: width)
        // Each tap alone: a deviation that grows with depth shows here.
        let taps: Int = sockets.count
        let tap = { (values: UnsafeBufferPointer<Float>, slot: Int) -> [Float] in
            var picked: [Float] = []
            picked.reserveCapacity(t * width)
            for row in 0..<t {
                let start: Int = (row * taps + slot) * width
                picked.append(contentsOf: values[start..<(start + width)])
            }
            return picked
        }
        for slot in [0, 5, 11] {
            let oracle = tap(oracleHidden, slot)
            let engine = hidden.withUnsafeBufferPointer { tap($0, slot) }
            report.add(oracle.withUnsafeBufferPointer {
                Judge.boundary("tap \(sockets[slot])", engine: engine, oracle: $0, columns: width)
            })
        }
        // The fusion's golden is the 512² DiT evaluation's, on the short prompt.
        guard !invocation.long else { return report }
        let dit512 = try invocation.goldens.open("krea2/goldens-krea2-dit-512.safetensors")
        let fusion = try Krea2TextFusion(artifact: try Artifact(path: try invocation.ditMap(ditMap)), maxTokens: t)
        fusion.recordBoundaries = true
        let fused = try fusion.fuse(oracleHidden.baseAddress!, tokens: t)
        report.add(dit512, "fusion (refiners)", fusion.boundaries["text_fused"], name: "text_fused", columns: width)
        report.add(dit512, "txt_in", fused, name: "txt_in_out", columns: fusion.outDim)
        return report
    }

    static let dit = Stage(
        model: model, name: "dit",
        proves: "one evaluation of the DiT at σ = 1, boundary by boundary, judged at the fp64 witness where it exists",
        sizes: ["512", "1024", "768x512"], options: ["--map", "--neural-engine", "--bf16-activations", "--bf16-attention", "--fp16-attention", "--no-accelerators", "--variant"],
        oracle: "tools/oracle/krea2.py dit --size {size}; witness --size {size}; witness-step --size {size} --step 0; draw --size {size} --step 0 --jitter 1 2 3; with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention, bf16 --size {size}; with --variant compact or light, dit, bf16, witness, witness-step --step 0, draw --step 0 --jitter 1 2 3 and draw --size {size} --step 0 --permute 1 2 3 4 5 6 7 8 --permute-kind group with --variant, at 512 \(Wording.publishers)", weights: ["DiT"],
        rule: "the rule, floor 5e-4, at the fp64 witness where it exists, clouds of summation orders when the oracle has order draws (the same seeds drawn by the engine), of one-ulp moves when it has only those; model_out per latent plane at the witness; \(Wording.rectangle); \(Wording.halfPrecision)"
    ) { invocation in
        let version = try invocation.quantizedVariant()
        let report = Report(title: "\(model) dit\(version.map { " " + $0.rawValue } ?? "") \(invocation.size)")
        let stem = variantStem("krea2/goldens-krea2-dit-\(invocation.size)", version)
        let goldens = try invocation.goldens.open(stem + ".safetensors")
        guard let latentEntry = goldens.entries["latent_in"], let latent = goldens.float32("latent_in"),
              let text = goldens.float32("txt_in_out"), let textEntry = goldens.entries["txt_in_out"],
              let sigma = goldens.float32("timestep") else {
            throw Skip("\(goldens.path): the evaluation's inputs are missing")
        }
        let (height, width, t) = (latentEntry.shape[1], latentEntry.shape[2], textEntry.shape[0])
        let artifact = try Artifact(path: try invocation.ditMap(ditMap, family: .krea2, version: version))
        // `witness-krea2-f64-R` stops after block 1; `-step0` is the whole evaluation in double.
        // A version's witnesses are the oracle's own, written from its weights (`witness-krea2-f64-<variant>-R`).
        let witnessStem = stem.replacingOccurrences(of: "goldens-krea2-dit-", with: "witness-krea2-f64-")
        let witnesses = [witnessStem, witnessStem + "-step0"].compactMap { invocation.goldens.optional($0 + ".safetensors") }
        let exact = { (name: String) in witnesses.lazy.compactMap { $0.float32(name) }.first }

        // Published int8 weights (the Compact): the evaluation along the int8 kernel, `model_out` once widened below (`Int8Route`).
        let int8 = Int8Route(artifact, invocation)
        let accelerators = Accelerators.on(neuralEngine: int8?.neuralEngine ?? EngineSettings.effective.plan.usesANE)
        let dit = try Krea2DiT(artifact: artifact, latentHeight: height, latentWidth: width, maxText: t,
                               accelerators: accelerators, plan: int8?.plan)
        dit.recordBoundaries = true
        let started = Date()
        let out = try dit.forward(latent: latent.baseAddress!, text: text.baseAddress!, textRows: t, sigma: sigma[0])
        if invocation.profile { printTimings(dit.timings, seconds: Date().timeIntervalSince(started)) }
        let recorded = dit.boundaries.merging(["model_out": out]) { $1 }
        let half = try invocation.halfReference(stem + ".safetensors", script: "tools/oracle/krea2.py bf16 --size \(invocation.size)\(version.map { " --variant " + $0.rawValue } ?? "")")
        // The oracle's draws of these inputs: `draw-krea2-R-step0-jitterN`.
        let draws = stem.replacingOccurrences(of: "goldens-krea2-dit-", with: "draw-krea2-") + "-step0"
        // Their summation-order draws, `draw-krea2-R-step0-permgroupN`, and the engine's of the same seeds: the map rewritten
        // with its residual stream permuted (`Krea2PermutedMap.draw`), evaluated by a DiT built as `make` builds it.
        let orders = Clouds.orders(invocation.goldens, draws)
        let orderDraw = { (make: @escaping (Artifact) throws -> Krea2DiT) -> (UInt64) throws -> [String: [Float]] in
            { seed in
                try Krea2PermutedMap.draw(source: artifact.path, seed: seed, text: text) { map, movedText in
                    let permuted = try make(map)
                    permuted.recordBoundaries = true
                    let v = try permuted.forward(latent: latent.baseAddress!, text: movedText.baseAddress!, textRows: t, sigma: sigma[0])
                    return permuted.boundaries.merging(["model_out": v]) { $1 }
                }
            }
        }
        var clouds = Clouds(draws: (1...8).compactMap { invocation.goldens.optional(draws + "-jitter\($0).safetensors") },
                            orders: orders, engineOrder: orderDraw {
                                try Krea2DiT(artifact: $0, latentHeight: height, latentWidth: width, maxText: t,
                                             accelerators: accelerators, plan: int8?.plan)
                            }) { seed in
            var moved = Array(latent)
            moveByOneULP(&moved, seed: seed)
            let v = try moved.withUnsafeBufferPointer {
                try dit.forward(latent: $0.baseAddress!, text: text.baseAddress!, textRows: t, sigma: sigma[0])
            }
            return dit.boundaries.merging(["model_out": v]) { $1 }
        }
        let d = dit.config.dim
        let widths = ["temb": d, "temb_mod": d, "model_out": height * width]
        for name in ["temb", "temb_mod", "img_in_out", "block0_attn_out", "block0_ff_out", "block0_out",
                     "block1_out", "block13_out", "block27_out", "model_out"] {
            guard let engine = recorded[name] else { report.add(Judge.missing(name)); continue }
            guard let oracle = goldens.float32(name) else { continue }
            let columns = widths[name] ?? d
            let m = Judge.boundary(name, engine: engine, oracle: oracle, witness: exact(name),
                                   half: half?.values(name), columns: columns)
            report.add(try clouds.arbitrate(m, exact: exact(name), columns: columns))
        }
        if let expected = goldens.float32("model_out") {
            if let witness = exact("model_out") {
                let m = Judge.planes("model_out per plane", engine: out, oracle: expected, witness: witness,
                                     reference: half?.values("model_out"), planes: 16)
                report.add(try clouds.arbitrate(m, exact: witness, columns: height * width, as: "model_out", planes: 16))
            }
            if let m = Judge.orientation(out, expected, height: height, width: width) { report.add(m) }
        }
        if let int8 {
            int8.account(report)
            if int8.judgesWidened, let expected = goldens.float32("model_out") {
                // The widened route by the same rule, its clouds drawn by its own DiT.
                let widened = try Krea2DiT(artifact: artifact, latentHeight: height, latentWidth: width, maxText: t)
                let forward = { (l: UnsafePointer<Float>) in try widened.forward(latent: l, text: text.baseAddress!, textRows: t, sigma: sigma[0]) }
                let other = try forward(latent.baseAddress!)
                var widenedClouds = Clouds(draws: (1...8).compactMap { invocation.goldens.optional(draws + "-jitter\($0).safetensors") },
                                           orders: orders, engineOrder: orderDraw {
                                               try Krea2DiT(artifact: $0, latentHeight: height, latentWidth: width, maxText: t)
                                           }) { seed in
                    var moved = Array(latent)
                    moveByOneULP(&moved, seed: seed)
                    return ["model_out": try moved.withUnsafeBufferPointer { try forward($0.baseAddress!) }]
                }
                if let witness = exact("model_out") {
                    let m = Judge.planes("model_out per plane (widened)", engine: other, oracle: expected, witness: witness,
                                         reference: half?.values("model_out"), planes: 16)
                    report.add(try widenedClouds.arbitrate(m, exact: witness, columns: height * width, as: "model_out", planes: 16))
                } else {
                    report.add(Judge.boundary("model_out (widened)", engine: other, oracle: expected, half: half?.values("model_out"),
                                              columns: height * width))
                }
            }
        }
        if invocation.neuralEngine { try NeuralEngineRun.account(report) }
        if invocation.bf16Activations { try BF16ActivationRun.account(report) }
        if invocation.bf16Attention { try BF16AttentionRun.account(report) }
        if invocation.fp16Attention { try FP16AttentionRun.account(report) }
        return report
    }

    static let trajectory = Stage(
        model: model, name: "trajectory",
        proves: "the seed's noise bit for bit, the schedule, then the 8 Euler steps: each step's velocity from the oracle's latent at that step, at the step's fp64 witness where it exists",
        sizes: ["512", "1024"], options: ["--map"],
        oracle: "tools/oracle/krea2.py trajectory --size {size}; witness-step --size {size} --step 0 1 2 3 4 5 6 7; draw --size {size} --step i --permute 1 2 3 4 5 6 7 8 --permute-kind group, one run per step i from 0 to 7", weights: ["DiT"],
        rule: "σ within 1e-6; the rule on each guided velocity, floor 5e-4, at the step's fp64 witness where it exists, clouds of summation orders when the oracle has order draws of the step (the same seeds drawn by the engine), of one-ulp moves when it has only those; the free pass is shown, not judged"
    ) { invocation in
        let report = Report(title: "\(model) trajectory \(invocation.size)")
        let trajectory = try invocation.goldens.open("krea2/goldens-krea2-trajectory-\(invocation.size).safetensors")
        // The text rows do not depend on the size: the 512² evaluation's are every trajectory's.
        let dit512 = try invocation.goldens.open("krea2/goldens-krea2-dit-512.safetensors")
        guard let initEntry = trajectory.entries["latent_init"], let initial = trajectory.float32("latent_init"),
              let reference = trajectory.float32("sigmas"), let text = dit512.float32("txt_in_out"),
              let textEntry = dit512.entries["txt_in_out"] else {
            throw Skip("\(trajectory.path): the trajectory's inputs are missing")
        }
        // The starting noise first: `torch.randn` at the manifest's seed, to the bit (`TorchNoise`).
        guard let seed = (try? invocation.goldens.manifest("krea2/goldens-krea2-trajectory-\(invocation.size)"))?["seed"] as? NSNumber else {
            throw Skip("the trajectory's manifest holds no seed")
        }
        report.add(Judge.identical("noise (seed \(seed))",
                                   Latent.noise(.qwenImage, height: initEntry.shape[initEntry.shape.count - 2],
                                                width: initEntry.shape[initEntry.shape.count - 1], seed: seed.uint64Value).values,
                                   initial))
        let steps = reference.count - 1
        let sigmas = Krea2DiT.sigmas(steps: steps)
        report.add(Judge.schedule(sigmas, reference))
        let side = initEntry.shape[1], t = textEntry.shape[0], columns = side * side
        let map = try invocation.ditMap(ditMap)
        let dit = try Krea2DiT(artifact: try Artifact(path: map), latentHeight: side, latentWidth: side, maxText: t)
        // Like the product: the modulation of every σ in one pass, before the first step.
        try dit.tabulateModulation(sigmas: Array(sigmas.dropLast()))
        let velocity = { (latent: [Float], i: Int) in
            try latent.withUnsafeBufferPointer {
                try dit.forward(latent: $0.baseAddress!, text: text.baseAddress!, textRows: t, sigma: sigmas[i])
            }
        }
        // The guided pass: each step from the oracle's latent, its velocity measured at its witness; then the order draws of
        // every step that fails there, map by map (below); then each step judged.
        var pending: [(step: Int, measure: Measure, exact: UnsafeBufferPointer<Float>?, clouds: Clouds, input: [Float])] = []
        var drawn: [Int: [UInt64: [String: [Float]]]] = [:]
        for i in 0..<steps {
            let input = i == 0 ? Array(initial) : trajectory.float32("latent_\(i)").map(Array.init) ?? []
            guard !input.isEmpty, let expected = trajectory.float32("velocity_\(i)") else { continue }
            let step = "krea2/witness-krea2-f64-\(invocation.size)-step\(i)"
            let exact = invocation.goldens.optional(step + ".safetensors")?.float32("velocity")
            // The step's draws: one-ulp moves (`-jitterN`) and summation orders (`-permgroupN`), the engine's orders read
            // from `drawn`, filled below.
            let stem = "krea2/draw-krea2-\(invocation.size)-step\(i)"
            let clouds = Clouds(draws: (1...8).compactMap { invocation.goldens.optional(stem + "-jitter\($0).safetensors") },
                                orders: Clouds.orders(invocation.goldens, stem), engineOrder: { seed in
                guard let v = drawn[i]?[seed] else { throw Artifact.Failure.misuse("step \(i): order \(seed) not drawn") }
                return v
            }) { seed in
                var moved = input
                moveByOneULP(&moved, seed: seed)
                return ["model_out": try velocity(moved, i)]
            }
            let m = Judge.boundary("v_\(i)", engine: try velocity(input, i), oracle: expected, witness: exact, columns: columns)
            pending.append((i, m, exact, clouds, input))
        }
        // **Seed outside**: one map rewritten with its residual stream permuted per seed (the Standard's, ~26 GB), on which
        // every failing step of that seed is evaluated from its latent — 8 maps at worst, not 8 per failing step.
        let ordered = pending.filter { $0.clouds.judgesByOrders($0.measure, exact: $0.exact, as: "model_out") }
        let orders = try Krea2PermutedMap.draws(source: map, requests: ordered.map { $0.clouds.orders.map(\.seed) }, text: text) {
            k, permutedMap, movedText in
            let p = ordered[k]
            let permuted = try Krea2DiT(artifact: permutedMap, latentHeight: side, latentWidth: side, maxText: t)
            return ["model_out": try p.input.withUnsafeBufferPointer {
                try permuted.forward(latent: $0.baseAddress!, text: movedText.baseAddress!, textRows: t, sigma: sigmas[p.step])
            }]
        }
        for (k, p) in ordered.enumerated() { drawn[p.step] = orders[k] }
        var guided: [Measure] = []
        for p in pending {
            var clouds = p.clouds
            guided.append(try clouds.arbitrate(p.measure, exact: p.exact, columns: columns, as: "model_out"))
        }
        // The free pass: the latents chain, the error amplifies; shown beside each step.
        var latent = Array(initial)
        for i in 0..<steps {
            let v = try velocity(latent, i)
            let dt = sigmas[i + 1] - sigmas[i]
            for j in 0..<latent.count { latent[j] += dt * v[j] }
            guard i < guided.count else { continue }
            let drift = trajectory.float32("latent_\(i + 1)").map { channelError(latent, $0, columns: columns).worst }
            report.add(drift.map { guided[i].with(note: String(format: "free latent %.2e", $0)) } ?? guided[i])
        }
        return report
    }

    static let vae = Stage(
        model: model, name: "vae",
        proves: "the Qwen-Image decoder with Krea's weights on the final latent of the oracle's trajectory, plane by plane",
        sizes: ["512", "1024", "768x512"], oracle: "tools/oracle/krea2.py vae --size {size}; vae-witness --size {size}", weights: ["VAE"],
        rule: "the rule per plane, floor 5e-4, at the fp64 witness (no floor there); on a rectangle, the transposed image must deviate"
    ) { invocation in
        let report = Report(title: "\(model) vae \(invocation.size)")
        let goldens = try invocation.goldens.open("krea2/goldens-krea2-vae-\(invocation.size).safetensors")
        let witness = invocation.goldens.optional("krea2/witness-krea2-vae-f64-\(invocation.size).safetensors")
        guard let entry = goldens.entries["latent_final"], let latent = goldens.float32("latent_final"),
              let expected = goldens.float32("image") else {
            throw Skip("\(goldens.path): latent_final or image missing")
        }
        let decoder = try AnimaVAE(path: try Library.processWide.component(.krea2, "vae.safetensors"),
                                   latentHeight: entry.shape[1], latentWidth: entry.shape[2])
        let (image, _) = try decoder.decode(latent: latent.baseAddress!)
        report.add(Judge.planes("image RGB", engine: image, oracle: expected, witness: witness?.float32("image"), planes: 3))
        if let m = Judge.orientation(image, expected, height: decoder.imageHeight, width: decoder.imageWidth) { report.add(m) }
        return report
    }
}

/// Every value moved by one ulp, up or down by a hash of (seed, index) — the oracle's `--jitter`.
func moveByOneULP(_ x: inout [Float], seed: UInt64) {
    for i in 0..<x.count {
        let h = (UInt64(i) &* 2654435761 &+ seed &* 40503) >> 7
        x[i] = h & 1 == 1 ? x[i].nextUp : x[i].nextDown
    }
}

/// **The clouds' arbiter** (see the header): a measure that fails at the witness, and only such a one, is
/// judged again by draws when the oracle has at least two of the same inputs. Two families: the summation orders
/// (`orders`, the oracle's `-permgroupN` with the engine's own draws of the same seeds, `engineOrder`), which judge
/// whenever both sides have them, and the one-ulp moves of the input (`draws`, `engineDraw`) otherwise. The oracle's
/// cloud is its nominal, its ulp draws and its order draws; the engine's, its nominal and the draws of the judging family.
struct Clouds {
    let draws: [Safetensors]
    /// One engine evaluation on the inputs moved by one ulp (seed n): its boundaries by name.
    let engineDraw: (UInt64) throws -> [String: [Float]]
    /// The oracle's summation-order draws, by seed (`krea2.py draw --permute N --permute-kind group`).
    let orders: [(seed: UInt64, file: Safetensors)]
    /// One engine evaluation with its residual stream permuted by the group permutation of a seed
    /// (`Krea2PermutedMap.draw`): its boundaries by name, in the reference's channel order.
    let engineOrder: ((UInt64) throws -> [String: [Float]])?
    private var engine: [[String: [Float]]] = []

    init(draws: [Safetensors], orders: [(seed: UInt64, file: Safetensors)] = [],
         engineOrder: ((UInt64) throws -> [String: [Float]])? = nil, engineDraw: @escaping (UInt64) throws -> [String: [Float]]) {
        self.draws = draws
        self.engineDraw = engineDraw
        self.engineOrder = engineOrder
        self.orders = engineOrder == nil ? [] : orders
    }

    /// The oracle's order draws of `stem` (`-permgroup1` … `-permgroup8`): not `permgroup0`, the identity taken by
    /// another loading path than the golden's, nor the full permutations, which a convrot map cannot draw.
    static func orders(_ goldens: Goldens, _ stem: String) -> [(seed: UInt64, file: Safetensors)] {
        (1...8).compactMap { n in goldens.optional(stem + "-permgroup\(n).safetensors").map { (UInt64(n), $0) } }
    }

    /// **Will `arbitrate` judge `m` by summation orders**, and so ask `engineOrder` for every seed of `orders`? What a
    /// caller asks before drawing, to draw the orders of all its failing measures map by map (`Krea2PermutedMap.draws`)
    /// rather than measure by measure: a measure that fails at its witness, with at least two oracle order draws of it.
    func judgesByOrders(_ m: Measure, exact: UnsafeBufferPointer<Float>?, as name: String? = nil) -> Bool {
        guard !m.passed, let exact, m.atWitness != nil else { return false }
        let name = name ?? m.boundary
        return orders.filter { $0.file.float32(name)?.count == exact.count }.count >= 2
    }

    /// `m` itself, or its verdict by clouds. `name`: the boundary's name in the draws, `m`'s by default.
    /// `planes`: the measure is per plane of a latent (`Judge.planes`), not per channel of `columns`.
    mutating func arbitrate(_ m: Measure, exact: UnsafeBufferPointer<Float>?, columns: Int, as name: String? = nil,
                            planes: Int? = nil) throws -> Measure {
        guard !m.passed, let exact, let atWitness = m.atWitness else { return m }
        let name = name ?? m.boundary
        let worst = { (v: [Float], exact: UnsafeBufferPointer<Float>) -> Double in
            planes.map { errorPerPlane(v, exact, planes: $0).worst } ?? channelError(v, exact, columns: columns).worst
        }
        let judged = { (files: [Safetensors]) in
            files.compactMap { d -> Double? in
                guard let v = d.float32(name), v.count == exact.count else { return nil }
                return worst(Array(v), exact)
            }
        }
        let moves = judged(draws), permuted = judged(orders.map(\.file))
        // The orders judge when the oracle drew them: the engine draws the same seeds.
        let byOrders = judgesByOrders(m, exact: exact, as: name), engineOrder = byOrders ? self.engineOrder : nil
        guard byOrders || moves.count >= 2 else { return m.with(note: "no oracle draws to arbitrate") }
        if engine.isEmpty {
            engine = try engineOrder.map { draw in try orders.map { try draw($0.seed) } }
                ?? (1...4).map { try engineDraw(UInt64($0)) }
        }
        let engineCloud = [atWitness.engine] + engine.compactMap { b -> Double? in
            guard let v = b[name], v.count == exact.count else { return nil }
            return worst(v, exact)
        }
        let oracleCloud = [atWitness.reference] + moves + permuted
        let median = engineCloud.sorted()[engineCloud.count / 2], threshold = 1.25 * (oracleCloud.max() ?? 0)
        let note = byOrders
            ? String(format: "clouds of summation orders: engine median of %d (nominal + %d orders), oracle worst of %d (nominal + %d ulp + %d orders)",
                     engineCloud.count, engineCloud.count - 1, oracleCloud.count, moves.count, permuted.count)
            : String(format: "clouds of one-ulp moves: engine median of %d draws, oracle worst of %d", engineCloud.count, oracleCloud.count)
        var verdict = Measure(boundary: m.boundary, worst: m.worst, median: m.median, channel: m.channel,
                              atWitness: (median, oracleCloud.max() ?? 0), threshold: threshold, passed: median <= threshold, note: note)
        verdict.kind = m.kind
        return verdict
    }
}
