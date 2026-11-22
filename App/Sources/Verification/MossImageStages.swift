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
    package static let all: [Stage] = MossImageEncoderStages.all + [schedule, dit, ditLoRA, pace, trajectory, vae, img2img, render]

    package static let model = "moss-image"
    package static let ditMap = Family.mossImage.ditMap(.standard)
    static let compactMap = Family.mossImage.ditMap(.compact)
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
        proves: "one evaluation of the DiT (the base weights, no LoRA) at σ = 1, boundary by boundary: the time path, the embedders, the text blocks, block 0 inside out, blocks 1, 13, 27 and 41, the head, and the velocity per latent plane (with --variant compact, the Compact's int8 codes)",
        sizes: ["512", "1024"], options: ["--map", "--variant"], variants: [[], ["--variant", "compact"]],
        oracle: "tools/oracle/moss-image.py dit --size {size}; witness --size {size}; dit-bf16 --size {size}; with --variant compact, dit, witness and w8a8 with --variant compact at 512",
        weights: ["DiT"],
        rule: "the rule, floor 5e-4, with the model in bf16 beside the oracle (the W8A8 run for a Compact); model_out per latent plane at the fp64 witness"
    ) { invocation in
        let compact = try invocation.compact()
        let stem = { (s: String) in "moss/\(s)\(compact ? "-compact" : "")-\(invocation.size)" }
        let report = Report(title: "\(model) dit\(compact ? " compact" : "") \(invocation.size)", half: compact ? "W8A8" : "bf16")
        let path = try invocation.ditMap(ditMap, compactMap, compact: compact)
        let goldens = try invocation.goldens.open(stem("goldens-moss-dit") + ".safetensors")
        let witness = invocation.goldens.optional("moss/witness-moss-dit\(compact ? "-compact" : "")-f64-\(invocation.size).safetensors")
        let half = compact
            ? try invocation.goldens.int8(stem("witness-w8a8-moss-dit") + ".safetensors", script: "tools/oracle/moss-image.py w8a8 --variant compact")
            : try invocation.goldens.half(stem("witness-bf16-moss-dit") + ".safetensors", script: "tools/oracle/moss-image.py dit-bf16 --size \(invocation.size)")
        guard let latent = goldens.float32("latent_in"), let text = goldens.float32("text_in"), let entry = goldens.entries["text_in"],
              let timestep = goldens.float32("timestep"), let latentEntry = goldens.entries["latent_in"] else {
            throw Skip("\(goldens.path): the evaluation's inputs are missing")
        }
        let program = MossImageDiTProgram(latentHeight: latentEntry.shape[1], latentWidth: latentEntry.shape[2], textLength: entry.shape[0])
        let c = program.config, (h, w) = (program.latentHeight, program.latentWidth)
        let (velocity, out) = try MossImageRender.probe(map: path, latent: Array(latent), latentHeight: h, latentWidth: w,
                                                        text: Array(text), textLength: program.textLength, timestep: timestep[0])
        var boundaries = out
        boundaries["rope"] = program.rope()
        let widths = ["time_embedding_out": c.dim, "shared_modulation": 6 * c.dim, "head_modulation": 2 * c.dim, "rope": c.headDim,
                      "block0_ffn_up_gate_out": 2 * c.hidden, "block0_q_norm_out": c.headDim]
        for name in MossImageDiTProgram.tapNames {
            report.add(goldens, name, boundaries[name], witness: witness?.float32(name), half: half.float32(name).map(Array.init),
                       columns: widths[name] ?? c.dim)
        }
        if let expected = goldens.float32("model_out") {
            if let exact = witness?.float32("model_out") {
                report.add(Judge.planes("model_out per plane", engine: velocity, oracle: expected, witness: exact,
                                        reference: half.float32("model_out").map(Array.init), planes: c.channels))
            } else {
                report.add(goldens, "model_out", velocity, half: half.float32("model_out").map(Array.init), columns: h * w)
            }
            if let m = Judge.orientation(velocity, expected, height: h, width: w) { report.add(m) }
        }
        return report
    }

    // MARK: - The DiT under the DMD LoRA

    static let ditLoRA = Stage(
        model: model, name: "dit-lora",
        proves: "the DiT under the family's DMD LoRA kept apart (y = base(x) + scale·up(down(x)), the model card's demo), against the reference's LoRA-merged fp32 evaluation, at the fp64 witness, with the bf16 reference carrying the LoRA apart beside it",
        oracle: "tools/oracle/moss-image.py dit --lora; witness --lora; bf16 (model_out_apart)", weights: ["DiT", "LoRA"],
        rule: "model_out per latent plane at the fp64 witness: no further than the bf16 reference with the LoRA kept apart, + 25 %"
    ) { invocation in
        let report = Report(title: "\(model) dit-lora \(invocation.size)")
        let path = try invocation.ditMap(ditMap)
        let loraPath = try Library.processWide.component(.mossImage, MossImagePublished.turbo)
        let goldens = try invocation.goldens.open("moss/goldens-moss-dit-lora-\(invocation.size).safetensors")
        let witness = invocation.goldens.optional("moss/witness-moss-dit-lora-f64-\(invocation.size).safetensors")
        let half = try invocation.goldens.half("moss/witness-moss-bf16-\(invocation.size).safetensors", script: "tools/oracle/moss-image.py bf16 --size \(invocation.size)")
        guard let latent = goldens.float32("latent_in"), let text = goldens.float32("text_in"), let entry = goldens.entries["text_in"],
              let timestep = goldens.float32("timestep"), let latentEntry = goldens.entries["latent_in"],
              let expected = goldens.float32("model_out") else { throw Skip("\(goldens.path): the evaluation's inputs are missing") }
        let program = MossImageDiTProgram(latentHeight: latentEntry.shape[1], latentWidth: latentEntry.shape[2], textLength: entry.shape[0])
        let (velocity, _) = try MossImageRender.probe(map: path, latent: Array(latent), latentHeight: program.latentHeight, latentWidth: program.latentWidth,
                                                      text: Array(text), textLength: program.textLength, timestep: timestep[0],
                                                      loras: try MossImageRender.recipe(loraPath))
        report.add(Judge.planes("model_out per plane", engine: velocity, oracle: expected, witness: witness?.float32("model_out"),
                                reference: half.float32("model_out_apart").map(Array.init), planes: program.config.channels))
        return report
    }

    // MARK: - The DiT's pace

    static let pace = Stage(
        model: model, name: "pace", proves: Pace.proves + " (the DMD LoRA kept apart, as a render; with --variant compact, the Compact's)",
        sizes: ["1024", "512"], options: ["--renders", "--evaluations", "--map", "--variant"],
        oracle: nil, weights: ["DiT", "LoRA"], rule: Pace.rule
    ) { invocation in
        let compact = try invocation.compact()
        let path = try invocation.ditMap(ditMap, compactMap, compact: compact)
        let (h, w) = (invocation.height / LatentSpace.flux2.factor, invocation.width / LatentSpace.flux2.factor), rows = 48
        let text = HostTensor([rows, MossImageDiT.Config.published.textDim], floats: Pace.noise(rows * MossImageDiT.Config.published.textDim, seed: 1))
        let loras = try MossImageRender.recipe(try Library.processWide.component(.mossImage, MossImagePublished.turbo))
        return try Pace.run(invocation, title: "\(model) pace\(compact ? " compact" : "") \(invocation.size)") {
            let d = try MossImageRender.Denoiser(map: path, loras: loras, latentHeight: h, latentWidth: w, textLength: rows)
            print("\(model) pace: " + d.described)
            d.text = text
            return d
        } evaluate: { d, e, time in
            try d.velocity(Pace.noise(LatentSpace.flux2.channels * h * w, seed: UInt64(e + 2)), time: 1000 * time)
        }
    }

    // MARK: - The decoder

    static let vae = Stage(
        model: model, name: "vae",
        proves: "the FLUX.2 VAE decoder (f32, as the pipeline decodes) on the reference's latent: batch-norm denormalization, unpatchify, post_quant_conv, the decoder's boundaries and the clamped image",
        oracle: "tools/oracle/moss-image.py vae; vae-witness", weights: ["VAE"],
        rule: "per plane, at the fp64 witness: no further from it than the f32 reference, + 25 %"
    ) { invocation in
        let report = Report(title: "\(model) vae \(invocation.size)")
        let goldens = try invocation.goldens.open("moss/goldens-moss-vae-\(invocation.size).safetensors")
        let witness = invocation.goldens.optional("moss/witness-moss-vae-f64-\(invocation.size).safetensors")
        guard let entry = goldens.entries["latent"], let latent = goldens.float32("latent"), let expected = goldens.float32("image") else {
            throw Skip("\(goldens.path): latent or image missing")
        }
        let (h, w) = (entry.shape[1], entry.shape[2])
        let path = try Library.processWide.component(.mossImage, "vae.safetensors")
        let names = Flux2VAEDecoder.tapNames
        let (image, taps) = try blocking {
            let decoder = try await Flux2VAEDecoder.open(path: path, latentHeight: h, latentWidth: w, budget: MemoryBudget.machineCeiling())
            return try await decoder.pixels(Array(latent), taps: Set(names))
        }
        for name in names {
            guard let expected = goldens.float32(name) else { continue }
            guard let got = taps[name] else { report.add(Judge.missing(name)); continue }
            report.add(Judge.planes(name, engine: got, oracle: expected, planes: goldens.entries[name]?.shape[0] ?? 1))
        }
        report.add(Judge.planes("image RGB", engine: image, oracle: expected, witness: witness?.float32("image"), planes: 3))
        if let m = Judge.orientation(image, expected, height: 16 * h, width: 16 * w) { report.add(m) }
        return report
    }

    // MARK: - The 8-step trajectory

    static let trajectory = Stage(
        model: model, name: "trajectory",
        proves: "the seed's noise bit for bit, the 8-step σ schedule and the timesteps, then each step's velocity (the DMD LoRA kept apart) from the reference's own latent at that step",
        options: ["--map"],
        oracle: "tools/oracle/moss-image.py trajectory; trajectory-bf16", weights: ["DiT", "LoRA"],
        rule: "the noise and σ exact; each step, position by position (the 128 channels of a cell), by the rule against the model in bf16 (LoRA apart), floor 5e-4"
    ) { invocation in
        let report = Report(title: "\(model) trajectory \(invocation.size)")
        let path = try invocation.ditMap(ditMap)
        let reference = try invocation.goldens.open("moss/goldens-moss-trajectory-\(invocation.size).safetensors")
        let dit = try invocation.goldens.open("moss/goldens-moss-dit-\(invocation.size).safetensors")
        let half = try invocation.goldens.half("moss/witness-bf16-moss-trajectory-\(invocation.size).safetensors",
                                               script: "tools/oracle/moss-image.py trajectory-bf16 --size \(invocation.size)")
        let manifest = try invocation.goldens.manifest("moss/goldens-moss-trajectory-\(invocation.size)")
        guard let initEntry = reference.entries["latent_init"], let initial = reference.float32("latent_init"),
              let sigmas = reference.float32("sigmas"), let timesteps = reference.float32("timesteps"),
              let text = dit.float32("text_in"), let textEntry = dit.entries["text_in"], let seed = manifest["seed"] as? NSNumber else {
            throw Skip("\(reference.path): the trajectory's inputs are missing")
        }
        let (height, width) = (initEntry.shape[1], initEntry.shape[2])
        report.add(Judge.identical("noise (seed \(seed))", Latent.noise(.flux2, height: height, width: width, seed: seed.uint64Value).values, initial))
        let ours = MossImageSchedule.sigmas(steps: sigmas.count - 1)
        report.add(Judge.schedule(ours, sigmas))
        report.add(Judge.identical("timesteps", MossImageSchedule.timesteps(ours), timesteps))
        let denoiser = try MossImageRender.Denoiser(map: path, loras: try MossImageRender.recipe(try Library.processWide.component(.mossImage, MossImagePublished.turbo)),
                                                    latentHeight: height, latentWidth: width, textLength: textEntry.shape[0])
        denoiser.text = HostTensor(textEntry.shape, floats: Array(text))
        try report.trajectory(reference, steps: sigmas.count - 1, half: half, columns: height * width, name: { "velocity_\($0)" },
                          label: { "v_\($0)" }) { step, latent in
            // The render's own call (`RenderLoop`) at our σ, its timestep `MossImageSchedule.timestep`'s, judged above.
            try denoiser.derivative(latent, sigma: ours[step], solver: .euler, first: step == 0)
        }
        return report
    }

    // MARK: - An image to start from

    static let img2img = Stage(
        model: model, name: "img2img",
        proves: "an image to start from: the FLUX.2 VAE's encoder (f32, as the reference encodes) on the oracle's picture — the posterior mean, patchified, batch-norm normalized —, the first step run at strength 0.6 and its σ, the starting latent σ·ε + (1 − σ)·z₀ from the reference's own z₀ and noise, then the app's render from the picture",
        sizes: ["512"], options: ["--png"],
        oracle: "tools/oracle/moss-image.py text; img2img --size {size}", weights: ["text encoder", "DiT", "LoRA", "VAE"],
        rule: "z₀ per plane against the f32 reference, at the floor; the first step and σ exact; the starting latent bit for bit; the render finite, its PSNR against the reference's final latent decoded here said and not judged (a free loop in bf16)"
    ) { invocation in
        let report = Report(title: "\(model) img2img \(invocation.size)")
        let vae = try Library.processWide.component(.mossImage, "vae.safetensors")
        let reference = try StartImageReference(
            report, invocation, stem: "moss/goldens-moss-img2img-\(invocation.size)", witness: nil, space: .flux2,
            schedule: { MossImageSchedule.sigmas(steps: $0) }, encode: { try Flux2VAEEncoder.latent(of: $0, path: vae) })
        guard let prompt = try invocation.goldens.manifest("moss/goldens-moss-img2img-\(invocation.size)")["prompt"] as? String else {
            throw Skip("moss/goldens-moss-img2img-\(invocation.size): the prompt is missing")
        }
        let model = try Model.mossImage(in: Library.processWide, card: .mossImage, dit: try invocation.ditMap(ditMap),
                                        encoder: Family.mossImage.encoderMap(.standard))
        let render = try Engine().render(reference.request(prompt), model: model)
        if let png = invocation.extras["--png"] { try render.png().write(to: URL(fileURLWithPath: png)) }
        let (h, w) = (reference.latentHeight, reference.latentWidth)
        let decoded = try DecoderSession<Flux2VAEDecoder>(path: vae, latentHeight: h, latentWidth: w).pixels(reference.final)
        reference.judge(report, render, decoded: decoded)
        return report
    }

    // MARK: - The whole render

    static let render = Stage(
        model: model, name: "render",
        proves: "the app's render (Engine.render: tokenizer, Qwen3.5-9B, 8 DMD steps, FLUX.2 decoder) from the reference prompt and seed against the reference's fp32 picture",
        sizes: ["512"], options: ["--variant", "--png"],
        oracle: "tools/oracle/moss-image.py trajectory; vae-final", weights: ["text encoder", "DiT", "LoRA", "VAE"],
        rule: "the same picture: PSNR past 25 dB against the fp32 reference (a bound written before the run, as Z-Image's); with --variant compact, a finite picture, its PSNR said and not judged: the Compact is judged at the DiT"
    ) { invocation in
        let compact = try invocation.compact()
        let report = Report(title: "\(model) render\(compact ? " compact" : "") \(invocation.size)")
        let final = try invocation.goldens.open("moss/goldens-moss-vae-final-\(invocation.size).safetensors")
        let manifest = try invocation.goldens.manifest("moss/goldens-moss-trajectory-\(invocation.size)")
        guard let expected = final.float32("image"), let seed = manifest["seed"] as? NSNumber, let prompt = manifest["prompt"] as? String else {
            throw Skip("\(final.path): image, seed or prompt missing")
        }
        let library = Library.processWide
        let variant: Variant = compact ? .compact : .standard
        let path = try invocation.ditMap(ditMap, compactMap, compact: compact)
        let model = try Model.mossImage(in: library, card: .mossImage, dit: path, encoder: Family.mossImage.encoderMap(variant))
        let started = Date()
        let render = try Engine().render(Request(prompt, width: invocation.width, height: invocation.height, seed: seed.uint64Value, steps: 8),
                                         model: model)
        try report.rendered(render, since: started, invocation)
        report.judgePicture(render.image.pixels, expected, compact: compact)
        return report
    }
}
