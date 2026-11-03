import Foundation
import Siliconed

// **Anima Turbo, stage by stage** — `siliconed check anima <stage>`.
//
//     tokenizer → text → dit → trajectory → vae
//
// What sets it apart: two tokenizers (Qwen's and T5's) feed Qwen3-0.6B and a conditioner that maps its
// states onto T5's; the step is `x += (σᵢ₊₁ − σᵢ) · v`, without the negation Z-Image's pipeline applies;
// the VAE is Qwen-Image's. The goldens are the reference prompt's, seed 42, at 512² and 1024².

enum AnimaStages {
    static let all: [Stage] = [tokenizer, text, dit, trajectory, vae]

    static let model = "anima"
    static let ditMap = "anima-turbo-dit.v0.silicon"

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
        proves: "the prompt through both tokenizers, Qwen3-0.6B (final norm included) and the conditioner, to the DiT's [512, 1024] context (with --long, a prompt past 512 tokens: more rows)",
        options: ["--long"], oracle: "tools/oracle/anima.py text [--long]", weights: ["text encoder", "DiT (the conditioner)"],
        rule: "identifiers bit for bit; the rule, floor 5e-4"
    ) { invocation in
        let stem = invocation.long ? "anima/goldens-anima-text-long" : "anima/goldens-anima-text"
        let report = Report(title: "\(model) text\(invocation.long ? " --long" : "")")
        let goldens = try invocation.goldens.open(stem + ".safetensors")
        guard let prompt = try invocation.goldens.manifest(stem)["prompt"] as? String else {
            throw Skip("\(stem).json: no prompt")
        }
        let qwenIds = try Tokenizer(directory: try Library.processWide.component(.anima, "tokenizer")).encode(prompt)
        // Uncut, as the app reads it (`AnimaModules.encoder`); the reference cuts at its `max_sequence_length`.
        let t5Ids = try T5Tokenizer(directory: try Library.processWide.component(.anima, "t5_tokenizer")).encode(prompt, maxLength: .max)
        report.add(Judge.integers("Qwen tokens", qwenIds, goldens.int32("qwen_ids")))
        report.add(Judge.integers("T5 tokens", t5Ids, goldens.int32("t5_ids")))
        let encoder = try TextEncoder(artifact: try Artifact(path: MapCheck.use(Library.processWide.map("qwen3-0.6b-encoder.v0.silicon"))),
                                      sequence: qwenIds.count)
        encoder.recordBoundaries = true
        let hidden = try encoder.encode(ids: qwenIds, realTokens: qwenIds.count)
        for layer in [0, 13, 26] {
            report.add(goldens, "qwen layer \(layer)", encoder.boundaries["layer\(layer)_out"], name: "qwen_layer\(layer)_out", columns: 1024)
        }
        report.add(goldens, "qwen last_hidden", Array(hidden), name: "qwen_last_hidden", columns: 1024)
        let conditioner = try AnimaConditioner(path: adapterPath(fromMap: Library.processWide.map(Family.anima.ditMap)))
        conditioner.recordBoundaries = true
        // `hidden` points into the encoder's arena, which ARC does not see through.
        let embeds = try withExtendedLifetime(encoder) { try conditioner.condition(t5: t5Ids, qwen: hidden, qwenRows: qwenIds.count) }
        for b in [0, 5] { report.add(goldens, "conditioner block \(b)", conditioner.boundaries["cond_block\(b)_out"], name: "cond_block\(b)_out", columns: 1024) }
        report.add(goldens, "prompt_embeds", embeds, columns: 1024)
        return report
    }

    static let dit = Stage(
        model: model, name: "dit",
        proves: "one evaluation of the DiT at σ = 1, boundary by boundary, and its output per latent plane against the fp64 witness",
        sizes: ["512", "1024", "768x512"], options: ["--map", "--neural-engine", "--bf16-activations", "--bf16-attention", "--fp16-attention", "--no-accelerators", "--variant"],
        oracle: "tools/oracle/anima.py dit --size {size}; witness --size {size} --step 0; with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention, bf16 --size {size}; with --variant compact or light, dit and bf16 with --variant, at 512 \(Wording.publishers)", weights: ["DiT"],
        rule: "the rule, floor 5e-4; model_out per latent plane at the fp64 witness; \(Wording.rectangle); \(Wording.halfPrecision)"
    ) { invocation in
        let version = try invocation.quantizedVariant()
        let report = Report(title: "\(model) dit\(version.map { " " + $0.rawValue } ?? "") \(invocation.size)")
        let file = variantStem("anima/goldens-anima-dit-\(invocation.size)", version) + ".safetensors"
        let goldens = try invocation.goldens.open(file)
        // The Standard's fp64 witness is the Standard's weights': a version has none unless the oracle wrote its own.
        let witness = version == nil ? invocation.goldens.optional("anima/witness-anima-f64-\(invocation.size)-step0.safetensors") : nil
        guard let latentEntry = goldens.entries["latent_in"], let latent = goldens.float32("latent_in"),
              let context = goldens.float32("prompt_embeds"), let contextEntry = goldens.entries["prompt_embeds"],
              let sigma = goldens.float32("timestep") else {
            throw Skip("\(goldens.path): the evaluation's inputs are missing")
        }
        let (height, width) = (latentEntry.shape[1], latentEntry.shape[2])
        let dit = try AnimaDiT(artifact: try Artifact(path: try invocation.ditMap(ditMap, family: .anima, version: version)),
                               latentHeight: height, latentWidth: width, textLength: contextEntry.shape[0])
        dit.recordBoundaries = true
        let started = Date()
        _ = try dit.forward(latent: latent.baseAddress!, context: context.baseAddress!, sigma: sigma[0])
        if invocation.profile { printTimings(dit.timings, seconds: Date().timeIntervalSince(started)) }
        let half = try invocation.halfReference(file, script: "tools/oracle/anima.py bf16 --size \(invocation.size)\(version.map { " --variant " + $0.rawValue } ?? "")")
        let d = dit.config.dim
        let widths = ["temb": 3 * d, "embedded_timestep": d, "model_out": height * width]
        // Per channel against the oracle: the witness's worst POSITION is a noisy statistic here (an
        // honest fp32 computation lands from 1× to 3.5× the oracle's), so it judges per plane, below.
        for name in ["temb", "embedded_timestep", "patch_embed_out", "block0_attn1_out", "block0_attn2_out",
                     "block0_ff_out", "block0_out", "block1_out", "block13_out", "block27_out", "model_out"] {
            report.add(goldens, name, dit.boundaries[name], half: half?.values(name), columns: widths[name] ?? d)
        }
        if let out = dit.boundaries["model_out"], let expected = goldens.float32("model_out") {
            if let exact = witness?.float32("model_out") {
                report.add(Judge.planes("model_out per plane", engine: out, oracle: expected, witness: exact,
                                        reference: half?.values("model_out"), planes: 16))
            }
            if let m = Judge.orientation(out, expected, height: height, width: width) { report.add(m) }
        }
        if invocation.neuralEngine { try NeuralEngineRun.account(report) }
        if invocation.bf16Activations { try BF16ActivationRun.account(report) }
        if invocation.bf16Attention { try BF16AttentionRun.account(report) }
        if invocation.fp16Attention { try FP16AttentionRun.account(report) }
        return report
    }

    static let trajectory = Stage(
        model: model, name: "trajectory",
        proves: "the seed's noise bit for bit, the schedule, then the 8 Euler steps: each step's velocity from the oracle's latent at that step",
        sizes: ["512", "1024"], options: ["--map"],
        oracle: "tools/oracle/anima.py trajectory --size {size}", weights: ["DiT"],
        rule: "σ within 1e-6; the rule on each guided velocity, floor 5e-4; the free pass is shown with its gain, not judged"
    ) { invocation in
        let report = Report(title: "\(model) trajectory \(invocation.size)")
        let trajectory = try invocation.goldens.open("anima/goldens-anima-trajectory-\(invocation.size).safetensors")
        let text = try invocation.goldens.open("anima/goldens-anima-text.safetensors")
        guard let initEntry = trajectory.entries["latent_init"], let initial = trajectory.float32("latent_init"),
              let reference = trajectory.float32("sigmas"), let context = text.float32("prompt_embeds"),
              let contextEntry = text.entries["prompt_embeds"] else {
            throw Skip("\(trajectory.path): the trajectory's inputs are missing")
        }
        // The starting noise first: `torch.randn` at the manifest's seed, to the bit (`TorchNoise`).
        guard let seed = (try? invocation.goldens.manifest("anima/goldens-anima-trajectory-\(invocation.size)"))?["seed"] as? NSNumber else {
            throw Skip("the trajectory's manifest holds no seed")
        }
        report.add(Judge.identical("noise (seed \(seed))",
                                   Latent.noise(.qwenImage, height: initEntry.shape[initEntry.shape.count - 2],
                                                width: initEntry.shape[initEntry.shape.count - 1], seed: seed.uint64Value).values,
                                   initial))
        let steps = reference.count - 1
        let sigmas = AnimaDiT.sigmas(steps: steps, shift: 3)
        // A badly transcribed shift gives a plausible and wrong trajectory.
        report.add(Judge.schedule(sigmas, reference))
        let side = initEntry.shape[1]
        let dit = try AnimaDiT(artifact: try Artifact(path: try invocation.ditMap(ditMap)), latentHeight: side, latentWidth: side,
                               textLength: contextEntry.shape[0])
        // The guided pass, as before it the oracle's latent at each step; then the free pass, shown. The
        // witness of a step is not used: at 1024², steps 3 and 7, a hot point of the exact function's
        // conditioning fails torch's own redraws per plane.
        var guided: [Measure] = []
        var latent = Array(initial)
        for i in 0..<steps {
            if i > 0, let truth = trajectory.float32("latent_\(i)") { latent = Array(truth) }
            let v = try latent.withUnsafeBufferPointer { try dit.forward(latent: $0.baseAddress!, context: context.baseAddress!, sigma: sigmas[i]) }
            if let expected = trajectory.float32("velocity_\(i)") {
                guided.append(Judge.boundary("v_\(i)", engine: v, oracle: expected, columns: side * side))
            }
        }
        latent = Array(initial)
        for i in 0..<steps {
            let v = try latent.withUnsafeBufferPointer { try dit.forward(latent: $0.baseAddress!, context: context.baseAddress!, sigma: sigmas[i]) }
            let dt = sigmas[i + 1] - sigmas[i]
            for j in 0..<latent.count { latent[j] += dt * v[j] }
            guard i < guided.count else { continue }
            let drift = trajectory.float32("latent_\(i + 1)").map { channelError(latent, $0, columns: side * side).worst }
            report.add(drift.map { guided[i].with(note: String(format: "free latent %.2e", $0)) } ?? guided[i])
        }
        return report
    }

    static let vae = Stage(
        model: model, name: "vae",
        proves: "the Qwen-Image decoder on the final latent of the oracle's trajectory, plane by plane",
        sizes: ["512", "1024", "768x512"], oracle: "tools/oracle/anima.py vae --size {size}; vae-witness --size {size}", weights: ["VAE"],
        rule: "the rule per plane, floor 5e-4, at the fp64 witness (no floor there); on a rectangle, the transposed image must deviate"
    ) { invocation in
        let report = Report(title: "\(model) vae \(invocation.size)")
        let goldens = try invocation.goldens.open("anima/goldens-anima-vae-\(invocation.size).safetensors")
        let witness = invocation.goldens.optional("anima/witness-anima-vae-f64-\(invocation.size).safetensors")
        guard let entry = goldens.entries["latent_final"], let latent = goldens.float32("latent_final"),
              let expected = goldens.float32("image") else {
            throw Skip("\(goldens.path): latent_final or image missing")
        }
        let decoder = try AnimaVAE(path: try Library.processWide.component(.anima, "vae.safetensors"),
                                   latentHeight: entry.shape[1], latentWidth: entry.shape[2])
        let (image, _) = try decoder.decode(latent: latent.baseAddress!)
        report.add(Judge.planes("image RGB", engine: image, oracle: expected, witness: witness?.float32("image"), planes: 3))
        if let m = Judge.orientation(image, expected, height: decoder.imageHeight, width: decoder.imageWidth) { report.add(m) }
        return report
    }
}
