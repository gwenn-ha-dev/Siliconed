import Foundation
import Siliconed

// **Anima Turbo, stage by stage** — `siliconed check anima <stage>`.
//
//     tokenizer → text → block → dit → pace → trajectory → vae → render
//
// What sets it apart: two tokenizers (Qwen's and T5's) feed Qwen3-0.6B and a conditioner that maps its
// states onto T5's; the step is `x += (σᵢ₊₁ − σᵢ) · v`, without the negation Z-Image's pipeline applies;
// the VAE is Qwen-Image's. The goldens are the reference prompt's, seed 42, at 512² and 1024², the DiT's
// evaluation and trajectory also at 1536² (9,216 image tokens).
//
// Every stage but the tokenizer runs on the engine, in the arithmetic Siliconed declares (bf16 as the reference runs in
// bf16, int8 on the Compact's codes, judged beside the publisher's own int8 products). So each is judged by the rule with the
// reference's own bf16 run beside the f32 oracle (`half`). A stage without its bf16 reference is skipped, with the script that
// writes it.

enum AnimaStages {
    static let all: [Stage] = [tokenizer, text, block, dit, ditLoRA, pace, trajectory, vae, img2img, render]

    static let model = "anima"
    static let ditMap = Family.anima.ditMap(.standard)
    static let compactMap = Family.anima.ditMap(.compact)
    static let encoderMap = Family.anima.encoderMap(.standard)

    // MARK: - The text

    static let tokenizer = Stage(
        model: model, name: "tokenizer",
        proves: "the T5 tokenizer against `tokenizers` on a corpus of edge cases: normalization, whitespace, special tokens, truncation",
        oracle: "tools/oracle/anima.py tokenizer", weights: ["tokenizer"], rule: "every case bit for bit"
    ) { invocation in
        let report = Report(title: "\(model) tokenizer")
        let path = invocation.goldens.path("anima/t5-corpus.json")
        guard FileManager.default.fileExists(atPath: path) else { throw Skip("corpus missing: \(path)") }
        guard let cases = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [[String: Any]] else {
            throw Skip("unreadable corpus: \(path)")
        }
        let t5 = try T5Tokenizer(directory: try Library.processWide.component(.anima, "t5_tokenizer"))
        var differing: [String] = []
        for entry in cases {
            guard let text = entry["text"] as? String, let expected = entry["ids"] as? [Int] else { continue }
            if t5.encode(text) != expected { differing.append(String(text.debugDescription.prefix(40))) }
        }
        report.add(Judge.identity("T5 corpus", differing: differing.count, of: cases.count, unit: "case"))
        for text in differing.prefix(8) { print("  differs: \(text)") }
        return report
    }

    static let text = Stage(
        model: model, name: "text",
        proves: "the prompt through both tokenizers, Qwen3-0.6B (bf16, final norm included) and the conditioner, to the DiT's [512, 1024] context",
        oracle: "tools/oracle/anima.py text; text-bf16", weights: ["text encoder", "DiT (the conditioner)"],
        rule: "identifiers bit for bit; the rule against transformers and diffusers in bf16"
    ) { invocation in
        let report = Report(title: "\(model) text")
        // The tokenizers and Qwen3-0.6B, then the conditioner from the reference's own Qwen states: one thing judged at a time.
        try Qwen3TextStages.anima(invocation, report: report)
        let goldens = try invocation.goldens.open("anima/goldens-anima-text.safetensors")
        let half = try invocation.goldens.half("anima/witness-bf16-anima-text.safetensors", script: "tools/oracle/anima.py text-bf16")
        guard let hiddenEntry = goldens.entries["qwen_last_hidden"], let hidden = goldens.float32("qwen_last_hidden"),
              let t5 = goldens.int32("t5_ids") else {
            throw Skip("\(goldens.path): the conditioner's inputs are missing")
        }
        let source = HostTensor(hiddenEntry.shape, floats: Array(hidden)), ids = (0..<t5.count).map { Int(t5[$0]) }
        // The conditioner's weights are the DiT file's (`llm_adapter`).
        let path = try invocation.ditMap(ditMap)
        let (context, tapped) = try blocking {
            try await AnimaConditioner.context(source: source, t5Ids: ids, path: path, taps: true, budget: MemoryBudget.machineCeiling()) { print($0) }
        }
        for name in ["cond_block0_out", "cond_block5_out"] {
            report.add(goldens, name, tapped[name], half: half.float32(name).map(Array.init), columns: AnimaConditioner.width)
        }
        report.add(goldens, "prompt_embeds", context.floats, half: half.float32("prompt_embeds").map(Array.init), columns: AnimaConditioner.width)
        return report
    }

    // MARK: - The DiT

    static let block = Stage(
        model: model, name: "block",
        proves: "block 0 alone, from the oracle's own input, at its four boundaries: self-attention, cross-attention, feed-forward, output",
        sizes: ["512", "1024", "768x512"], options: ["--map"],
        oracle: "tools/oracle/anima.py dit --size {size}; block-bf16 --size {size}", weights: ["DiT"],
        rule: "the rule against diffusers' block 0 in bf16 on the same input"
    ) { invocation in
        let report = Report(title: "\(model) block \(invocation.size)")
        let path = try invocation.ditMap(ditMap)
        let goldens = try invocation.goldens.open("anima/goldens-anima-dit-\(invocation.size).safetensors")
        let half = try invocation.goldens.half("anima/witness-bf16-anima-block-\(invocation.size).safetensors",
                                               script: "tools/oracle/anima.py block-bf16 --size \(invocation.size)")
        guard let latentEntry = goldens.entries["latent_in"], let x = goldens.float32("patch_embed_out"),
              let temb = goldens.float32("temb"), let embedded = goldens.float32("embedded_timestep"),
              let context = goldens.float32("prompt_embeds"), let contextEntry = goldens.entries["prompt_embeds"] else {
            throw Skip("\(goldens.path): the block's inputs are missing")
        }
        let rows = contextEntry.shape[0]
        let geometry = AnimaEvaluation.Geometry(height: latentEntry.shape[1], width: latentEntry.shape[2], context: rows)
        let tapped = try blocking {
            let dit = try await AnimaEvaluation.open(path, geometry: geometry, taps: true, budget: MemoryBudget.machineCeiling())
            print("\(model) block: " + dit.described)
            defer { print("let go: \(dit.close().line)") }
            try await dit.prompt(Array(context), rows: rows)
            try await dit.blockZero(x: Array(x), temb: Array(temb), embedded: Array(embedded))
            return dit.tapped
        }
        for name in AnimaCosmosDiT.BlockTap.allCases.map(\.rawValue) {
            report.add(goldens, name, tapped[name], half: half.float32(name).map(Array.init), columns: AnimaFunction.d)
        }
        return report
    }

    static let dit = Stage(
        model: model, name: "dit",
        proves: "one evaluation of the DiT at σ = 1, boundary by boundary: the embedders, blocks 0, 1, 13 and 27, and the velocity per latent plane (with --variant compact, the Compact's int8 codes)",
        sizes: ["512", "1024", "768x512", "1536"], options: ["--map", "--variant"],
        oracle: "tools/oracle/anima.py dit --size {size}; witness --size {size} --step 0; bf16 --size {size}; with --variant compact, dit and w8a8 with --variant compact at 512",
        weights: ["DiT"],
        rule: "the rule, floor 5e-4, with diffusers in bf16 beside the oracle (the W8A8 run for a Compact); model_out per latent plane at the fp64 witness; at h ≠ w, the orientation counter-test"
    ) { invocation in
        let compact = try invocation.compact()
        // `goldens-anima-dit-compact-512`: the variant before the size.
        let stem = { (s: String) in "anima/\(s)-\(compact ? "compact-" : "")\(invocation.size)" }
        let report = Report(title: "\(model) dit\(compact ? " compact" : "") \(invocation.size)", half: compact ? "W8A8" : "bf16")
        let path = try invocation.ditMap(ditMap, compactMap, compact: compact)
        let goldens = try invocation.goldens.open(stem("goldens-anima-dit") + ".safetensors")
        // The Standard's fp64 witness is the Standard's weights': a variant has none.
        let witness = compact ? nil : invocation.goldens.optional("anima/witness-anima-f64-\(invocation.size)-step0.safetensors")
        let half = compact
            ? try invocation.goldens.int8(stem("witness-w8a8-anima-dit") + ".safetensors", script: "tools/oracle/anima.py w8a8 --size \(invocation.size) --variant compact")
            : try invocation.goldens.half(stem("witness-bf16-anima-dit") + ".safetensors", script: "tools/oracle/anima.py bf16 --size \(invocation.size)")
        guard let latentEntry = goldens.entries["latent_in"], let latent = goldens.float32("latent_in"),
              let contextEntry = goldens.entries["prompt_embeds"], let context = goldens.float32("prompt_embeds"),
              let sigma = goldens.float32("timestep") else {
            throw Skip("\(goldens.path): the evaluation's inputs are missing")
        }
        let (h, w, rows, d) = (latentEntry.shape[1], latentEntry.shape[2], contextEntry.shape[0], AnimaFunction.d)
        let geometry = AnimaEvaluation.Geometry(height: h, width: w, context: rows)
        let (velocity, tapped) = try blocking {
            let dit = try await AnimaEvaluation.open(path, geometry: geometry, taps: true, budget: MemoryBudget.machineCeiling())
            print("\(model) dit: " + dit.described)
            defer { print("let go: \(dit.close().line)") }
            try await dit.prompt(Array(context), rows: rows)
            // The bf16 reference is told its timestep in bf16 (`timestep.to(half)`).
            let v = try await dit.evaluate(Array(latent), time: BFloat.round(sigma[0]), taps: Set(AnimaCosmosDiT.Tap.blocks.keys))
            var tapped = dit.tapped
            tapped["temb"] = await dit.read("temb", count: 3 * d, as: .float32)
            tapped["embedded_timestep"] = await dit.read("embedded", count: d, as: .float32)
            return (v, tapped)
        }
        let widths: [String: Int] = ["temb": 3 * d, "embedded_timestep": d]
        for name in AnimaCosmosDiT.Tap.allCases.map(\.rawValue) where goldens.entries[name] != nil {
            report.add(goldens, name, tapped[name], witness: witness?.float32(name), half: half.float32(name).map(Array.init),
                       columns: widths[name] ?? d)
        }
        // Per latent plane against the witness where there is one (squares): the worst POSITION is a noisy statistic here (an
        // honest fp32 computation lands from 1× to 3.5× the oracle's). A rectangle has none: per position, with bf16 beside.
        if let expected = goldens.float32("model_out") {
            if let exact = witness?.float32("model_out") {
                report.add(Judge.planes("model_out per plane", engine: velocity, oracle: expected, witness: exact,
                                        reference: half.float32("model_out").map(Array.init), planes: AnimaFunction.channels))
            } else {
                report.add(goldens, "model_out", velocity, half: half.float32("model_out").map(Array.init), columns: h * w)
            }
            if let m = Judge.orientation(velocity, expected, height: h, width: w) { report.add(m) }
        }
        return report
    }

    // MARK: - The DiT's pace

    static let pace = Stage(
        model: model, name: "pace", proves: Pace.proves + " (with --variant compact, the Compact's)", sizes: ["1024", "512"],
        options: ["--renders", "--evaluations", "--map", "--variant"], oracle: nil, weights: ["DiT"], rule: Pace.rule
    ) { invocation in
        let compact = try invocation.compact()
        let path = try invocation.ditMap(ditMap, compactMap, compact: compact)
        let (h, w) = (invocation.height / LatentSpace.qwenImage.factor, invocation.width / LatentSpace.qwenImage.factor), rows = 512
        let context = HostTensor([rows, AnimaFunction.textDim], floats: Pace.noise(rows * AnimaFunction.textDim, seed: 1))
        return try Pace.run(invocation, title: "\(model) pace\(compact ? " compact" : "") \(invocation.size)") {
            let d = try AnimaRender.Denoiser(map: path, latentHeight: h, latentWidth: w, textLength: rows)
            print("\(model) pace: " + d.described)
            try d.give(context)
            return d
        } evaluate: { d, e, time in
            try d.velocity(Pace.noise(AnimaFunction.channels * h * w, seed: UInt64(e + 2)), time: time)
        }
    }

    // MARK: - The sampler

    static let trajectory = Stage(
        model: model, name: "trajectory",
        proves: "the seed's noise bit for bit, the 8-step σ schedule, then each step's velocity from the reference's own latent at that step (the render's DiT session)",
        sizes: ["512", "1024"], options: ["--map"],
        oracle: "tools/oracle/anima.py trajectory --size {size}; trajectory-bf16 --size {size}; witness --size {size} --step N for a step at its threshold", weights: ["DiT"],
        rule: "the noise and σ exact; each step by the rule against diffusers' same step in bf16, against its fp64 witness where one is written"
    ) { invocation in
        let report = Report(title: "\(model) trajectory \(invocation.size)")
        let path = try invocation.ditMap(ditMap)
        let reference = try invocation.goldens.open("anima/goldens-anima-trajectory-\(invocation.size).safetensors")
        let text = try invocation.goldens.open("anima/goldens-anima-text.safetensors")
        let half = try invocation.goldens.half("anima/witness-bf16-anima-trajectory-\(invocation.size).safetensors",
                                               script: "tools/oracle/anima.py trajectory-bf16 --size \(invocation.size)")
        guard let initEntry = reference.entries["latent_init"], let initial = reference.float32("latent_init"),
              let sigmas = reference.float32("sigmas"), let context = text.float32("prompt_embeds"),
              let contextEntry = text.entries["prompt_embeds"],
              let seed = (try invocation.goldens.manifest("anima/goldens-anima-trajectory-\(invocation.size)"))["seed"] as? NSNumber else {
            throw Skip("\(reference.path): the trajectory's inputs are missing")
        }
        let (height, width) = (initEntry.shape[1], initEntry.shape[2])
        report.add(Judge.identical("noise (seed \(seed))", Latent.noise(.qwenImage, height: height, width: width, seed: seed.uint64Value).values,
                                   initial))
        let steps = sigmas.count - 1
        let schedule = AnimaDiT.sigmas(steps: steps)
        report.add(Judge.schedule(schedule, sigmas))
        // Then each step from the reference's own latent: each velocity is computed on the input the reference saw.
        let denoiser = try AnimaRender.Denoiser(map: path, latentHeight: height, latentWidth: width, textLength: contextEntry.shape[0])
        try denoiser.give(HostTensor(contextEntry.shape, floats: Array(context)))
        try report.trajectory(reference, steps: steps, half: half, columns: height * width, name: { "velocity_\($0)" }, witness: { step in
            invocation.goldens.optional("anima/witness-anima-f64-\(invocation.size)-step\(step).safetensors")?.float32("velocity")
        }) { step, latent in
            // The render's own call (`RenderLoop`), under the reference's Euler (the render's default): Anima's derivative is its velocity.
            try denoiser.derivative(latent, sigma: schedule[step], solver: .euler, first: step == 0)
        }
        return report
    }

    // MARK: - The decoder

    static let img2img = Stage(
        model: model, name: "img2img",
        proves: "an image to start from: the VAE's encoder in bf16 on the oracle's picture (the posterior mean, normalized), the first step run at strength 0.6 and its σ, the starting latent σ·ε + (1 − σ)·z₀ from the reference's own z₀ and noise, then the app's render from the picture",
        sizes: ["512"], options: ["--png"],
        oracle: "tools/oracle/anima.py text; img2img --size {size}; img2img-bf16 --size {size}", weights: ["text encoder", "DiT", "VAE"],
        rule: "z₀ per latent plane by the rule against diffusers encoding in bf16; the first step and σ exact; the starting latent bit for bit; the render finite, its PSNR against the reference's final latent decoded here said and not judged (a free loop in bf16)"
    ) { invocation in
        let report = Report(title: "\(model) img2img \(invocation.size)")
        let vae = try Library.processWide.component(.anima, "vae.safetensors")
        let reference = try StartImageReference(
            report, invocation, stem: "anima/goldens-anima-img2img-\(invocation.size)",
            witness: ("anima/witness-bf16-anima-img2img-\(invocation.size)", "tools/oracle/anima.py img2img-bf16 --size \(invocation.size)"),
            space: .qwenImage, schedule: { AnimaDiT.sigmas(steps: $0) }, encode: { try QwenImageVAEEncoder.latent(of: $0, path: vae) })
        guard let prompt = try invocation.goldens.manifest("anima/goldens-anima-text")["prompt"] as? String else {
            throw Skip("anima/goldens-anima-text: the prompt is missing")
        }
        let render = try Engine().render(reference.request(prompt), model: try Model.anima(in: Library.processWide, dit: try invocation.ditMap(ditMap)))
        if let png = invocation.extras["--png"] { try render.png().write(to: URL(fileURLWithPath: png)) }
        reference.judge(report, render, decoded: try qwenImageDecoded(reference.final, height: reference.latentHeight, width: reference.latentWidth, vae: vae))
        return report
    }

    static let vae = Stage(
        model: model, name: "vae",
        proves: "the Qwen-Image decoder in bf16 on the final latent of the oracle's trajectory (a centered crop of the 1024² one at 768x512), plane by plane",
        sizes: ["512", "1024", "768x512"],
        oracle: "tools/oracle/anima.py vae --size {size}; vae-witness --size {size}; vae-bf16 --size {size}", weights: ["VAE"],
        rule: "per plane, at the fp64 witness: no further from it than diffusers decoding in bf16, + 25 %; on a rectangle, the transposed image must deviate"
    ) { invocation in
        let report = Report(title: "\(model) vae \(invocation.size)")
        let goldens = try invocation.goldens.open("anima/goldens-anima-vae-\(invocation.size).safetensors")
        let witness = invocation.goldens.optional("anima/witness-anima-vae-f64-\(invocation.size).safetensors")
        let half = try invocation.goldens.half("anima/witness-bf16-anima-vae-\(invocation.size).safetensors",
                                               script: "tools/oracle/anima.py vae-bf16 --size \(invocation.size)")
        guard let entry = goldens.entries["latent_final"], let latent = goldens.float32("latent_final"),
              let expected = goldens.float32("image") else {
            throw Skip("\(goldens.path): latent_final or image missing")
        }
        let (h, w) = (entry.shape[1], entry.shape[2])
        let image = try qwenImageDecoded(Array(latent), height: h, width: w, vae: try Library.processWide.component(.anima, "vae.safetensors"))
        report.add(Judge.planes("image RGB", engine: image, oracle: expected, witness: witness?.float32("image"),
                                reference: half.float32("image").map(Array.init), planes: 3))
        if let m = Judge.orientation(image, expected, height: 8 * h, width: 8 * w) { report.add(m) }
        return report
    }

    // MARK: - The whole render

    static let render = Stage(
        model: model, name: "render",
        proves: "the app's render (Engine.render: both tokenizers, encoder, conditioner, 8 Euler steps, decoder) from the reference prompt and seed, against the reference's float32 picture, with diffusers' own bf16 render beside it",
        sizes: ["512", "1024"], options: ["--variant", "--map", "--png"],
        oracle: "tools/oracle/anima.py trajectory --size {size}; vae --size {size}; render-bf16 --size {size}",
        weights: ["text encoder", "DiT", "VAE"],
        rule: "the same picture: PSNR against diffusers in fp32 no more than 3 dB under diffusers' own bf16 render's (a free 8-step loop in bf16 is as far from the fp32 picture as it is); with --variant compact, a finite picture, its PSNR said and not judged: the Compact is judged at the DiT"
    ) { invocation in
        let compact = try invocation.compact()
        let report = Report(title: "\(model) render\(compact ? " compact" : "") \(invocation.size)")
        let image = try invocation.goldens.open("anima/goldens-anima-vae-\(invocation.size).safetensors")
        // The bf16 render is the Standard's bar; a Compact's render has none.
        let half = compact ? nil : try invocation.goldens.half("anima/witness-bf16-anima-render-\(invocation.size).safetensors",
                                                               script: "tools/oracle/anima.py render-bf16 --size \(invocation.size)")
        guard let prompt = try invocation.goldens.manifest("anima/goldens-anima-text")["prompt"] as? String,
              let seed = (try invocation.goldens.manifest("anima/goldens-anima-trajectory-\(invocation.size)"))["seed"] as? NSNumber,
              let expected = image.float32("image"), compact || half?.float32("image") != nil else {
            throw Skip("\(image.path): the prompt, seed or image is missing")
        }
        let path = try invocation.ditMap(ditMap, compactMap, compact: compact)
        let model = try Model.anima(in: Library.processWide, dit: path)
        let started = Date()
        let render = try Engine().render(Request(prompt, width: invocation.width, height: invocation.height,
                                                 seed: seed.uint64Value, steps: 8), model: model)
        try report.rendered(render, since: started, invocation)
        report.judgePicture(render.image.pixels, expected, compact: compact, half: half?.float32("image").map(Array.init))
        return report
    }

    // MARK: - The DiT under a LoRA

    static let ditLoRA = Stage(
        model: model, name: "dit-lora",
        proves: "one evaluation of the DiT at σ = 1 under a user's LoRA read in its published file (the oracle's inCase, kohya's flat names, 448 modules, strength 1), the low-rank path added beside each product: the velocity per latent plane",
        sizes: ["512"], options: ["--map", "--lora"],
        oracle: "tools/oracle/anima.py lora --size {size} --lora FILE; witness --size {size} --lora FILE; lora-bf16 --size {size} --lora FILE", weights: ["DiT", "the LoRA"],
        rule: "the file recognized as this family's, every module applied; model_out per latent plane at the fp64 witness: no further than the bf16 reference with the LoRA kept apart, + 25 %"
    ) { invocation in
        let report = Report(title: "\(model) dit-lora \(invocation.size)")
        let stem = "anima/goldens-anima-dit-lora-\(invocation.size)"
        let goldens = try invocation.goldens.open(stem + ".safetensors")
        let witness = invocation.goldens.optional("anima/witness-anima-f64-lora-\(invocation.size)-step0.safetensors")
        let half = try invocation.goldens.half("anima/witness-bf16-anima-dit-lora-\(invocation.size).safetensors",
                                               script: "tools/oracle/anima.py lora-bf16 --size \(invocation.size) --lora FILE")
        let info = try invocation.goldens.manifest(stem)
        guard let latentEntry = goldens.entries["latent_in"], let latent = goldens.float32("latent_in"),
              let contextEntry = goldens.entries["prompt_embeds"], let context = goldens.float32("prompt_embeds"),
              let sigma = goldens.float32("timestep"), let expected = goldens.float32("model_out") else {
            throw Skip("\(goldens.path): the evaluation's inputs are missing")
        }
        guard let file = invocation.extras["--lora"] else { throw Skip("the oracle's LoRA is not given: --lora <\(info["lora"] as? String ?? "its file")>") }
        let lora = try LoRAFile(file)
        let unapplied = lora.unapplied(by: AnimaCosmosDiT.loraModules())
        report.add(Judge.holds("recognized", lora.family() == .anima && unapplied.isEmpty,
                               note: "\(lora.modules.count) modules, rank \(lora.ranks.largest); \(unapplied.count) not applied"))
        let stack = LoRAStack(entries: [(lora, 1)])
        let path = try invocation.ditMap(ditMap)
        let (h, w, rows) = (latentEntry.shape[1], latentEntry.shape[2], contextEntry.shape[0])
        let (velocity, tapped) = try blocking {
            let dit = try await AnimaEvaluation.open(path, geometry: AnimaEvaluation.Geometry(height: h, width: w, context: rows), loras: stack,
                                                     budget: MemoryBudget.machineCeiling())
            print("\(model) dit-lora: " + dit.described)
            defer { dit.close() }
            try await dit.prompt(Array(context), rows: rows)
            // The bf16 reference is told its timestep in bf16 (`timestep.to(half)`).
            let v = try await dit.evaluate(Array(latent), time: BFloat.round(sigma[0]), taps: Set(AnimaCosmosDiT.Tap.blocks.keys))
            return (v, dit.tapped)
        }
        for name in AnimaCosmosDiT.Tap.blocks.sorted(by: { $0.key < $1.key }).map(\.value.rawValue) {
            report.add(goldens, name, tapped[name], half: half.float32(name).map(Array.init), columns: AnimaFunction.d)
        }
        report.add(Judge.planes("model_out per plane", engine: velocity, oracle: expected, witness: witness?.float32("model_out"),
                                reference: half.float32("model_out").map(Array.init), planes: AnimaFunction.channels))
        return report
    }
}
