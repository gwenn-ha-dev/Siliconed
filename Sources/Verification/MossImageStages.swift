import Foundation
import Siliconed

// **MOSS-Image, stage by stage** — `siliconed check moss-image <stage>`.
//
//     tokenizer → text-delta → text-full → text → schedule → dit → dit-lora → trajectory → vae
//
// (the text stages are `MossImageEncoderStages`, the decoder's `MossImageDecoderStages`; one model, one id.) The DiT's goldens are the reference's
// (`pipeline_moss_image.py`, fp32 on the CPU, `tools/oracle/moss-image.py`): the prompt « a 30 year old woman
// posing in a library », seed 42, 512² (a 32 × 32 grid of 128-channel cells). The `dit` stage starts from the
// oracle's own inputs (the noise, the timestep 1000, the encoder's last hidden state), so a deviation is
// located where it is born; every boundary the reference exposes is judged — the time path, the modulations, the
// RoPE table, the embedded image, the text path (projection and both text blocks), block 0 inside out, blocks
// 1, 13, 27 and 41, the head — at the fp64 witness where it exists.
//
// What sets it apart: the sequence is **image then text** in every golden; the engine keeps the text first
// (`MossImageDiT`) and records in the reference's order. The LoRA golden has the DMD LoRA **merged in fp32**
// at strength 1; the engine applies it unmerged after each product (the same algebra, the sums in another order).
//
// The maps are the installed family's (`store/`): the DiT, and for the LoRA stages the DMD LoRA's map in the
// family's components (`--lora PATH` judges another).

package enum MossImageStages {
    package static let all: [Stage] = MossImageEncoderStages.all + [schedule, dit, ditLoRA, trajectory] + MossImageDecoderStages.all

    package static let model = "moss-image"
    package static let ditMap = Family.mossImage.ditMap
    static let steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]

    // MARK: - The scheduler

    static let schedule = Stage(
        model: model, name: "schedule",
        proves: "the sigmas of MOSSImageScheduler (shift 5 on the fp32 grid, snapped to the 1000-step training grid) and the timesteps the DiT receives, for 1 to 10, 20 and 50 steps",
        oracle: "tools/oracle/moss-image.py schedule", weights: [],
        rule: "bit for bit"
    ) { invocation in
        let report = Report(title: "\(model) schedule")
        let goldens = try invocation.goldens.open("moss/goldens-moss-schedule.safetensors")
        if let grid = goldens.float32("training_grid") {
            report.add(Judge.identical("training grid", MossImageSchedule.trainingGrid, grid))
        }
        for n in steps {
            guard let sigmas = goldens.float32("sigmas_\(n)"), let timesteps = goldens.float32("timesteps_\(n)") else { continue }
            let ours = MossImageSchedule.sigmas(steps: n)
            report.add(Judge.identical("σ, \(n) steps", ours, sigmas))
            report.add(Judge.identical("timesteps, \(n) steps", MossImageSchedule.timesteps(ours), timesteps))
        }
        return report
    }

    // MARK: - The DiT

    static let dit = Stage(
        model: model, name: "dit",
        proves: "one evaluation of the DiT at σ = 1, boundary by boundary — time, modulations, RoPE, text path, block 0 inside out, blocks 1, 13, 27, 41, the head —, judged at the fp64 witness",
        options: ["--map", "--neural-engine", "--bf16-activations", "--variant"],
        oracle: "tools/oracle/moss-image.py dit; witness; with --neural-engine or --bf16-activations, dit-bf16; with --variant compact, dit, witness and dit-bf16 with --variant \(Wording.publishers)",
        weights: ["DiT"],
        rule: "the rule, floor 5e-4, at the fp64 witness where it exists; model_out per latent plane at the witness; \(Wording.halfPrecision)"
    ) { invocation in try evaluate(invocation, lora: false) }

    static let ditLoRA = Stage(
        model: model, name: "dit-lora",
        proves: "the same evaluation with the DMD 8-step LoRA at strength 1 (kohya `lora.safetensors`, rank 128, fp32), the golden having it merged in fp32",
        options: ["--map", "--lora"],
        oracle: "tools/oracle/moss-image.py dit --lora; witness --lora",
        weights: ["DiT", "LoRA"],
        rule: "the rule, floor 5e-4, at the fp64 witness where it exists; model_out per latent plane at the witness"
    ) { invocation in try evaluate(invocation, lora: true) }

    // MARK: - The 8-step trajectory

    static let trajectory = Stage(
        model: model, name: "trajectory",
        proves: "the 8 steps of the recipe: the seed's noise bit for bit, the σ, each step's velocity from the oracle's latent at that step (the DMD LoRA at strength 1), then the product's denoiser run free from the noise on the oracle's text — its latents shown against the oracle's, and against the free loop the stage runs itself",
        options: ["--map", "--lora"],
        oracle: "tools/oracle/moss-image.py trajectory",
        weights: ["DiT", "LoRA"],
        rule: "noise and σ bit for bit; each guided velocity per latent plane, floor 5e-4; the product's free run against the stage's own loop, 1e-5 (the same arithmetic); the free latents against the oracle's are shown, not judged (eight steps amplify the evaluation's rounding)"
    ) { invocation in
        let report = Report(title: "\(model) trajectory \(invocation.size)")
        let trajectory = try invocation.goldens.open("moss/goldens-moss-trajectory-\(invocation.size).safetensors")
        // The text rows do not depend on the size: the 512² evaluation's input is every trajectory's.
        let dit512 = try invocation.goldens.open("moss/goldens-moss-dit-512.safetensors")
        guard let initEntry = trajectory.entries["latent_init"], let initial = trajectory.float32("latent_init"),
              let sigmas = trajectory.float32("sigmas"), let text = dit512.float32("text_in"),
              let textEntry = dit512.entries["text_in"],
              let manifest = try? invocation.goldens.manifest("moss/goldens-moss-trajectory-\(invocation.size)"),
              let seed = (manifest["seed"] as? NSNumber)?.uint64Value else {
            throw Skip("\(trajectory.path): the trajectory's inputs (latent_init, sigmas, text_in, the seed) are missing")
        }
        let steps = sigmas.count - 1
        let (channels, height, width) = (initEntry.shape[initEntry.shape.count - 3], initEntry.shape[initEntry.shape.count - 2],
                                         initEntry.shape[initEntry.shape.count - 1])
        let rows = textEntry.shape[0]
        report.add(Judge.identical("noise (seed \(seed))",
                                   Latent.noise(.flux2, height: height, width: width, seed: seed).values, initial))
        let ours = MossImageSchedule.sigmas(steps: steps)
        report.add(Judge.schedule(ours, sigmas))
        let path = invocation.extras["--lora"] ?? Installer(library: .processWide).companionMap(.mossImage) ?? ""
        guard FileManager.default.fileExists(atPath: path) else {
            throw Skip("LoRA map missing: \(path) — siliconed install moss-image (or --lora PATH)")
        }
        let module = MossImageDenoisingModule(map: try invocation.ditMap(ditMap), turbo: path)
        let timesteps = MossImageSchedule.timesteps(ours)
        let dit = try MossImageDiT(artifact: try Artifact(path: module.map), gridHeight: height, gridWidth: width, maxText: rows,
                                   lora: try module.stack(nil))
        try dit.tabulateModulation(timesteps: timesteps)
        let velocity = { (latent: [Float], i: Int) -> [Float] in
            try latent.withUnsafeBufferPointer {
                try dit.forward(latent: $0.baseAddress!, text: text.baseAddress!, textRows: rows, timestep: timesteps[i])
            }
        }
        // The guided pass: each step from the oracle's latent, its velocity judged.
        for i in 0..<steps {
            let input = i == 0 ? Array(initial) : trajectory.float32("latent_\(i)").map(Array.init) ?? []
            guard !input.isEmpty, let expected = trajectory.float32("velocity_\(i)") else { continue }
            report.add(Judge.planes("v_\(i) per plane", engine: try velocity(input, i), oracle: expected, planes: channels))
        }
        // The free pass, the stage's own loop (Euler: `x += (σᵢ₊₁ − σᵢ)·v`, the scheduler's two roundings): the latents chain.
        var latent = Array(initial)
        var drift: [Double] = []
        for i in 0..<steps {
            let v = try velocity(latent, i)
            let dt = ours[i + 1] - ours[i]
            for j in 0..<latent.count { latent[j] += dt * v[j] }
            drift.append(trajectory.float32("latent_\(i + 1)").map { errorPerPlane(latent, $0, planes: channels).worst } ?? .nan)
        }
        print("  free latents against the oracle's, per plane, by step: " + drift.enumerated()
            .map { String(format: "%d %.1e", $0.offset + 1, $0.element) }.joined(separator: " · "))
        // The product's denoiser on the same inputs: the module's wiring (timesteps, modulation, Euler) is the loop's.
        var produced = Latent(space: .flux2, height: height, width: width, values: Array(initial))
        let conditioning = Conditioning(format: .mossImage, rows: rows, width: textEntry.shape[1], values: Array(text), tokens: rows)
        let started = Date()
        try module.denoise(&produced, text: conditioning, steps: steps, start: 0, lora: nil,
                           context: Context(reproducible: false, onProgress: nil))
        if invocation.profile { print(String(format: "denoising %.1f s for %d steps", Date().timeIntervalSince(started), steps)) }
        report.add(latent.withUnsafeBufferPointer {
            Judge.planes("product vs the stage's loop (latent_final)", engine: produced.values, oracle: $0, planes: channels, floor: 1e-5)
        })
        if let final = trajectory.float32("latent_final") {
            print(String(format: "  the product's final latent against the oracle's, per plane: %.2e",
                     errorPerPlane(produced.values, final, planes: channels).worst))
        }
        return report
    }

    /// The boundaries of the DiT in the reference's tensor names, with the width of their channels.
    static func boundaries(dim d: Int, headDim: Int, ffn: Int) -> [(name: String, columns: Int)] {
        [("time_embedding_out", d), ("shared_modulation", d), ("head_modulation", d), ("video_embed", d),
         ("rope", headDim), ("text_proj_out", d), ("text_block0_out", d), ("text_block1_out", d),
         ("block0_q_out", d), ("block0_k_out", d), ("block0_v_out", d), ("block0_q_norm_out", d), ("block0_o_out", d),
         ("block0_ffn_up_gate_out", 2 * ffn), ("block0_ffn_down_out", d),
         ("block0_out", d), ("block1_out", d), ("block13_out", d), ("block27_out", d), ("block41_out", d),
         ("final_norm_out", d)]
    }

    static func evaluate(_ invocation: Invocation, lora useLoRA: Bool) throws -> Report {
        // The Compact's golden is the reference on Comfy-Org's int8 convrot dequantized; `dit-lora` judges the Standard's.
        let version = useLoRA ? nil : try invocation.quantizedVariant()
        let title = "\(model) \(useLoRA ? "dit-lora" : "dit")\(version.map { " " + $0.rawValue } ?? "") \(invocation.size)"
        let report = Report(title: title)
        let stem = useLoRA ? "moss/goldens-moss-dit-lora-\(invocation.size)"
            : variantStem("moss/goldens-moss-dit-\(invocation.size)", version)
        let goldens = try invocation.goldens.open(stem + ".safetensors")
        guard let latentEntry = goldens.entries["latent_in"], let latent = goldens.float32("latent_in"),
              let text = goldens.float32("text_in"), let textEntry = goldens.entries["text_in"],
              let timestep = goldens.float32("timestep") else {
            throw Skip("\(goldens.path): the evaluation's inputs (latent_in, text_in, timestep) are missing")
        }
        let (height, width, t) = (latentEntry.shape[1], latentEntry.shape[2], textEntry.shape[0])
        let artifact = try Artifact(path: try invocation.ditMap(ditMap, family: .mossImage, version: version))
        var stack: LoRA?
        if useLoRA {
            let path = invocation.extras["--lora"] ?? Installer(library: .processWide).companionMap(.mossImage) ?? ""
            guard FileManager.default.fileExists(atPath: path) else {
                throw Skip("LoRA map missing: \(path) — siliconed install moss-image (or --lora PATH)")
            }
            let loaded = try LoRA(paths: [(path, 1)])
            try loaded.checkTarget(model: MossImageForge.model)
            stack = loaded
        }
        let witnessStem = stem.replacingOccurrences(of: "goldens-moss-dit", with: "witness-moss-dit")
            .replacingOccurrences(of: "512", with: "f64-512")
        let witness = invocation.goldens.optional(witnessStem + ".safetensors")
        // `dit-lora` lists no half-precision option: `half` is nil there.
        let half = useLoRA ? nil : try invocation.halfReference(
            stem + ".safetensors", script: "tools/oracle/moss-image.py dit-bf16\(version.map { " --variant " + $0.rawValue } ?? "")")
        let dit = try MossImageDiT(artifact: artifact, gridHeight: height, gridWidth: width, maxText: t, lora: stack)
        dit.recordBoundaries = true
        dit.recordedBlocks = [0, 1, 13, 27, 41]
        let started = Date()
        let out = try dit.forward(latent: latent.baseAddress!, text: text.baseAddress!, textRows: t, timestep: timestep[0])
        if invocation.profile { printTimings(dit.timings, seconds: Date().timeIntervalSince(started)) }
        let recorded = dit.boundaries
        for (name, columns) in boundaries(dim: dit.config.dim, headDim: dit.config.headDim, ffn: dit.config.hidden) {
            guard let oracle = goldens.float32(name) else { continue }
            guard let engine = recorded[name] else { report.add(Judge.missing(name)); continue }
            report.add(Judge.boundary(name, engine: engine, oracle: oracle, witness: witness?.float32(name),
                                      half: half?.values(name), columns: columns))
        }
        if let expected = goldens.float32("model_out") {
            report.add(Judge.planes("model_out per plane", engine: out, oracle: expected,
                                    witness: witness?.float32("model_out"), reference: half?.values("model_out"),
                                    planes: dit.config.channels))
        }
        // The official recipe's negative branch: block 22 not computed (zero-based), the same inputs — judged
        // when the oracle has written it (`goldens-moss-dit-skip22-512`, no LoRA), on the fp32 path only: the
        // oracle has no bf16 run of it to set a half-precision bar (the turbo recipe, CFG 1, never runs it).
        if half != nil {
            print("model_out, block 22 skipped: judged on the fp32 path only (no bf16 reference of that branch)")
        } else if !useLoRA, version == nil, let skipped = invocation.goldens.optional("moss/goldens-moss-dit-skip22-\(invocation.size).safetensors"),
           let expected = skipped.float32("model_out") {
            dit.recordBoundaries = false
            let skippedOut = try dit.forward(latent: latent.baseAddress!, text: text.baseAddress!, textRows: t,
                                             timestep: timestep[0], skipping: [22])
            report.add(Judge.planes("model_out, block 22 skipped", engine: skippedOut, oracle: expected, planes: dit.config.channels))
        }
        if invocation.neuralEngine { try NeuralEngineRun.account(report) }
        if invocation.bf16Activations { try BF16ActivationRun.account(report) }
        return report
    }
}
