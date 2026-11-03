import Foundation
import Siliconed

// **Z-Image Turbo, stage by stage** — `siliconed check z-image <stage>`.
//
//     tokenizer → text → block → dit → trajectory → vae
//
// The goldens are the reference prompt's (`goldens-text.json` holds it), seed 42, written by diffusers and
// transformers in fp32 on the CPU. Each stage starts from the oracle's own inputs, so a deviation is
// located where it is born rather than inherited from the stage before.

enum ZImageStages {
    static let all: [Stage] = [tokenizer, text, block, dit, trajectory, vae]

    static let ditMap = "z-image-turbo-dit.v1.silicon"

    // MARK: - The text

    static let tokenizer = Stage(
        model: "z-image", name: "tokenizer",
        proves: "the chat template and the tokenizer give transformers' 512 identifiers and mask, padding included, and the app's admission the same real tokens (with --long, past 512: 1024)",
        options: ["--long"], oracle: "tools/oracle/z-image.py text [--long]", weights: ["tokenizer"], rule: "bit for bit"
    ) { invocation in
        let stem = invocation.long ? "goldens-text-long" : "goldens-text"
        let report = Report(title: "z-image tokenizer\(invocation.long ? " --long" : "")")
        let tokenizer = try Tokenizer(directory: try Library.processWide.component(.zImage, "tokenizer"))
        let goldens = try invocation.goldens.open(stem + ".safetensors")
        guard let truth = goldens.int32("input_ids"), let mask = goldens.int32("attention_mask") else {
            throw Skip("\(goldens.path): no input_ids or attention_mask")
        }
        // The prompt lives in one place, the oracle's, which writes it beside the goldens.
        guard let prompt = try invocation.goldens.manifest(stem)["prompt"] as? String else { throw Skip("\(stem).json holds no prompt") }
        let (ids, ours) = tokenizer.encodeChat(user: prompt, maxLength: truth.count)
        // All 512, padding included: the wrong padding token gives right features (the extraction is
        // masked) and a wrong encoder, found three stages later.
        report.add(Judge.identity("input_ids", differing: (0..<ids.count).filter { ids[$0] != Int(truth[$0]) }.count, of: ids.count))
        report.add(Judge.identity("attention_mask", differing: (0..<ours.count).filter { ours[$0] != Int(mask[$0]) }.count, of: ours.count))
        // What a render reads: the same real tokens, admitted whole, unpadded.
        let real = (0..<mask.count).filter { mask[$0] != 0 }.map { truth[$0] }
        do {
            let admitted = try tokenizer.admitChat(user: prompt)
            report.add(real.withUnsafeBufferPointer { Judge.integers("admitted (the app's path)", admitted, $0) })
        } catch { report.add(Judge.holds("admitted (the app's path)", false, note: "\(error)")) }
        return report
    }

    static let text = Stage(
        model: "z-image", name: "text",
        proves: "the Qwen3-4B encoder gives the reference's hidden states, layer by layer, and its cap_feats (with --long, on a prompt past 512 tokens)",
        options: ["--long"], oracle: "tools/oracle/z-image.py text [--long]", weights: ["text encoder"], rule: "the rule, floor 5e-4"
    ) { invocation in
        let report = Report(title: "z-image text\(invocation.long ? " --long" : "")")
        let artifact = try Artifact(path: MapCheck.use(Library.processWide.map("qwen3-4b-encoder.v0.silicon")))
        let goldens = try invocation.goldens.open(invocation.long ? "goldens-text-long.safetensors" : "goldens-text.safetensors")
        guard let truthIds = goldens.int32("input_ids"), let mask = goldens.int32("attention_mask") else {
            throw Skip("\(goldens.path): no input_ids or attention_mask")
        }
        // The identifiers come from the golden, not the tokenizer: one thing judged at a time. Causal
        // attention with right padding: the real tokens alone are computed, no token depends on padding.
        let real = (0..<mask.count).reduce(0) { $0 + Int(mask[$1]) }
        let encoder = try TextEncoder(artifact: artifact, sequence: real)
        encoder.recordBoundaries = true
        let caps = try encoder.encode(ids: (0..<real).map { Int(truthIds[$0]) }, realTokens: real)
        let width = encoder.config.hidden
        for layer in [33, 34] where layer < encoder.config.layers {
            let name = "layer\(layer)_out"
            guard let expected = goldens.float32(name) else { continue }
            guard let got = encoder.boundaries[name] else { report.add(Judge.missing(name)); continue }
            // The reference is padded (512, or 1024 with --long): the computed positions only.
            let kept = min(got.count, real * width)
            report.add(Judge.boundary(name, engine: Array(got.prefix(kept)), oracle: UnsafeBufferPointer(rebasing: expected.prefix(kept)),
                                      columns: width))
        }
        if let expected = goldens.float32("cap_feats") {
            // `caps` points into the encoder's arena, which ARC does not see through.
            withExtendedLifetime(encoder) {
                report.add(Judge.boundary("cap_feats", engine: Array(caps), oracle: expected, columns: width))
            }
        }
        return report
    }

    // MARK: - The DiT

    static let block = Stage(
        model: "z-image", name: "block",
        proves: "block 0 matches the reference at its 19 internal boundaries: norms, projections, attention, SwiGLU",
        sizes: ["512", "1024"], options: ["--map"],
        oracle: "tools/oracle/z-image.py dit --size {size}", weights: ["DiT"], rule: "the rule, floor 5e-4"
    ) { invocation in
        let report = Report(title: "z-image block \(invocation.size)")
        let artifact = try Artifact(path: try invocation.ditMap(ditMap))
        let goldens = try invocation.goldens.open("goldens-dit-\(invocation.size).safetensors")
        guard let unified = goldens.entries["unified_in"], let freqs = goldens.float32("block_freqs_cis"),
              let adaln = goldens.float32("block_adaln_input") else {
            throw Skip("\(goldens.path): the block's inputs are missing")
        }
        // diffusers passes no attention mask at batch 1 (every sequence has the full length): a mask in
        // the golden would mean a batch of unequal lengths, which the engine does not run.
        guard goldens.entries["block_attn_mask"] == nil else {
            throw Skip("\(goldens.path): the oracle passed an attention mask, the engine runs none")
        }
        let s = unified.shape[1], d = unified.shape[2]
        let config = artifact.header["config"] as? [String: Any] ?? [:]
        let heads = config["n_heads"] as? Int ?? 30
        let hidden = 10240
        let arena = try Arena(capacity: (8 * s * d + 3 * s * hidden + d * hidden + 4 * d) * 4 + (64 << 20))
        let x = try arena.reserve("x", bytes: s * d * 4).assumingMemoryBound(to: Float.self)
        let gemm = try GEMM()
        let block = try Block(artifact: artifact, shapes: .init(sequence: s, dim: d, heads: heads, hidden: hidden),
                              eps: Float(config["norm_eps"] as? Double ?? 1e-5), gemm: gemm,
                              elementwise: try ElementwiseGPU(device: gemm.device, queue: gemm.queue), arena: arena)
        block.recordBoundaries = true
        // The stream is the caller's: `Block` works on it in place.
        x.update(from: goldens.float32("unified_in")!.baseAddress!, count: s * d)
        try block.forward(x: x, freqs: freqs.baseAddress!, adaln: adaln.baseAddress!, adalnDim: adaln.count, prefix: "layers.0.")
        // Each boundary carries its width: 4·D for the modulation, the hidden size in the FFN, D otherwise.
        let widths: [String: Int] = ["layer0_adaln_raw": 4 * d, "layer0_w1_out": hidden,
                                     "layer0_w3_out": hidden, "layer0_swiglu_out": hidden]
        for name in ["layer0_adaln_raw", "layer0_attn_norm1_out", "layer0_attn_in",
                     "layer0_q", "layer0_k", "layer0_v", "layer0_q_normed", "layer0_k_normed",
                     "layer0_sdpa_out", "layer0_attention_out", "layer0_attn_norm2_out",
                     "layer0_ffn_norm1_out", "layer0_ffn_in", "layer0_w1_out", "layer0_w3_out",
                     "layer0_swiglu_out", "layer0_feed_forward_out", "layer0_ffn_norm2_out", "layers_0_out"] {
            guard let expected = goldens.float32(name) else { continue }
            guard let got = block.boundaries[name] else { report.add(Judge.missing(name)); continue }
            report.add(Judge.boundary(name, engine: got, oracle: expected, columns: widths[name] ?? d))
        }
        withExtendedLifetime(arena) {}
        return report
    }

    static let dit = Stage(
        model: "z-image", name: "dit",
        proves: "one evaluation of the DiT, latent to velocity: embedders, refiners, 30 blocks, head, unpatchify",
        sizes: ["512", "1024", "768x512", "832x1216"], options: ["--map", "--neural-engine", "--bf16-activations", "--bf16-attention", "--fp16-attention", "--no-accelerators", "--variant"],
        oracle: "tools/oracle/z-image.py dit --size {size}; witness --size {size}; with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention, bf16 --size {size}; with --variant compact or light, the same three with --variant, at 512 \(Wording.publishers)", weights: ["DiT"],
        rule: "the rule, floor 5e-4; \(Wording.rectangle); \(Wording.halfPrecision)"
    ) { invocation in
        let version = try invocation.quantizedVariant()
        let report = Report(title: "z-image dit\(version.map { " " + $0.rawValue } ?? "") \(invocation.size)")
        let file = variantStem("goldens-dit-\(invocation.size)", version) + ".safetensors"
        let goldens = try invocation.goldens.open(file)
        let witness = invocation.goldens.optional(variantStem("witness-dit-f64-\(invocation.size)", version) + ".safetensors")
        guard let latentEntry = goldens.entries["in_latent"], let latent = goldens.float32("in_latent"),
              let capsEntry = goldens.entries["in_caps"], let caps = goldens.float32("in_caps"),
              let sigma = goldens.float32("in_sigma") else {
            throw Skip("\(goldens.path): the evaluation's inputs are missing")
        }
        // `in_latent` is `[16, 1, h, w]`.
        let (height, width) = (latentEntry.shape[2], latentEntry.shape[3])
        let dit = try DiT(artifact: try Artifact(path: try invocation.ditMap(ditMap, family: .zImage, version: version)),
                          latentHeight: height, latentWidth: width, capLength: capsEntry.shape[0])
        dit.recordBoundaries = true
        let started = Date()
        let out = try dit.forward(latent: latent.baseAddress!, caps: caps.baseAddress!, sigma: sigma[0])
        if invocation.profile { printTimings(dit.timings, seconds: Date().timeIntervalSince(started)) }

        var produced = dit.boundaries
        produced["model_out"] = out
        let half = try invocation.halfReference(file, script: "tools/oracle/z-image.py bf16 --size \(invocation.size)\(version.map { " --variant " + $0.rawValue } ?? "")")
        let widths: [String: Int] = ["t_embedder_out": dit.config.adalnDim, "block_freqs_cis": 2,
                                     "final_layer_out": dit.config.patch * dit.config.patch * dit.config.channels,
                                     "model_out": height * width]
        for name in ["t_embedder_out", "x_embedder_out", "cap_embedder_out", "context_refiner_in", "block_freqs_cis",
                     "noise_refiner_out", "context_refiner_out", "unified_in", "layers_29_out",
                     "final_layer_out", "model_out"] {
            guard let expected = goldens.float32(name) else { continue }
            guard let got = produced[name] else { report.add(Judge.missing(name)); continue }
            report.add(Judge.boundary(name, engine: got, oracle: expected, witness: witness?.float32(name),
                                      half: half?.values(name), columns: widths[name] ?? dit.config.dim))
        }
        if let expected = goldens.float32("model_out"), let m = Judge.orientation(out, expected, height: height, width: width) {
            report.add(m)
        }
        if invocation.neuralEngine { try NeuralEngineRun.account(report) }
        if invocation.bf16Activations { try BF16ActivationRun.account(report) }
        if invocation.bf16Attention { try BF16AttentionRun.account(report) }
        if invocation.fp16Attention { try FP16AttentionRun.account(report) }
        return report
    }

    static let trajectory = Stage(
        model: "z-image", name: "trajectory",
        proves: "the seed's noise bit for bit, then the 8-step sampler: each step's velocity and latent, from the reference's own latent at that step",
        sizes: ["512", "1024"], options: ["--map"],
        oracle: "tools/oracle/z-image.py trajectory --size {size}", weights: ["DiT"],
        rule: "the rule on the guided pass, floor 5e-4; the free pass is shown with its gain, not judged"
    ) { invocation in
        let report = Report(title: "z-image trajectory \(invocation.size)")
        let goldens = try invocation.goldens.open("goldens-dit-\(invocation.size).safetensors")
        let reference = try invocation.goldens.open("goldens-trajectory-\(invocation.size).safetensors")
        guard let capsEntry = goldens.entries["in_caps"], let caps = goldens.float32("in_caps"),
              let initEntry = reference.entries["latent_init"], let initial = reference.float32("latent_init") else {
            throw Skip("\(reference.path): the trajectory's inputs are missing")
        }
        // The starting noise first: `torch.randn` at the manifest's seed, to the bit (`TorchNoise`).
        guard let seed = (try? invocation.goldens.manifest("goldens-trajectory-\(invocation.size)"))?["seed"] as? NSNumber else {
            throw Skip("the trajectory's manifest holds no seed")
        }
        report.add(Judge.identical("noise (seed \(seed))",
                                   Latent.noise(.flux, height: initEntry.shape[initEntry.shape.count - 2],
                                                width: initEntry.shape[initEntry.shape.count - 1], seed: seed.uint64Value).values,
                                   initial))
        // Then each step from the reference's `latent_init`.
        let size = initEntry.shape[2], channels = initEntry.shape[1]
        let count = channels * size * size
        let dit = try DiT(artifact: try Artifact(path: try invocation.ditMap(ditMap)), latentHeight: size, latentWidth: size,
                          capLength: capsEntry.shape[0])
        let latent = UnsafeMutablePointer<Float>.allocate(capacity: count)
        defer { latent.deallocate() }
        /// One pass; `guided` overwrites the latent with the reference's after each step, so that each
        /// velocity is computed on the input the reference saw: it measures the code. The free pass
        /// measures how much this trajectory amplifies a perturbation, a property of the prompt.
        func pass(guided: Bool) throws -> Sampler {
            let sampler = Sampler(dit: dit, steps: 8)
            sampler.recordTrajectory = true
            latent.update(from: initial.baseAddress!, count: count)
            _ = try sampler.run(latent: latent, count: count, caps: caps.baseAddress!) { step, _, _ in
                if guided, let truth = reference.float32("latent_\(step + 1)") { latent.update(from: truth.baseAddress!, count: count) }
            }
            return sampler
        }
        let free = try pass(guided: false), held = try pass(guided: true)
        for step in 0..<held.latents.count {
            for (label, guided, unguided) in [("v_\(step)", held.velocities[step], free.velocities[step]),
                                              ("latent_\(step + 1)", held.latents[step], free.latents[step])] {
                guard let expected = reference.float32(label) else { continue }
                let m = Judge.boundary(label, engine: guided, oracle: expected, columns: size * size)
                let drift = channelError(unguided, expected, columns: size * size).worst
                report.add(m.with(note: String(format: "free %.2e (×%.1f)", drift, m.worst > 0 ? drift / m.worst : .nan)))
            }
        }
        return report
    }

    // MARK: - The decoder

    static let vae = Stage(
        model: "z-image", name: "vae",
        proves: "Flux's decoder on a rectangular latent (the centered crop of the final 1024² latent), plane by plane: what a square cannot see",
        sizes: ["768x512"],
        oracle: "tools/oracle/z-image.py vae --size {size}; vae-witness --size {size}", weights: ["VAE"],
        rule: "the rule per plane, floor 5e-4, at the fp64 witness (no floor there); on a rectangle, the transposed image must deviate"
    ) { invocation in
        let report = Report(title: "z-image vae \(invocation.size)")
        let goldens = try invocation.goldens.open("goldens-vae-\(invocation.size).safetensors")
        let witness = invocation.goldens.optional("witness-vae-f64-\(invocation.size).safetensors")
        guard let entry = goldens.entries["latent"], let latent = goldens.float32("latent"),
              let expected = goldens.float32("image") else {
            throw Skip("\(goldens.path): latent or image missing")
        }
        let decoder = try VAE(path: try Library.processWide.component(.zImage, "vae.safetensors"),
                              latentHeight: entry.shape[1], latentWidth: entry.shape[2])
        let (image, _) = try decoder.decode(latent: latent.baseAddress!)
        report.add(Judge.planes("image RGB", engine: image, oracle: expected, witness: witness?.float32("image"), planes: 3))
        if let m = Judge.orientation(image, expected, height: decoder.imageHeight, width: decoder.imageWidth) { report.add(m) }
        return report
    }
}

/// `--profile`: an evaluation's phases, the wall ones partitioning the time, each GPU one under its own.
func printTimings(_ timings: [String: Double], seconds: Double) {
    print(String(format: "one evaluation in %.2f s, footprint %.0f MB", seconds, Double(Arena.processFootprint()) / 1e6))
    for (phase, value) in timings.filter({ !$0.key.hasSuffix("GPU") }).sorted(by: { $0.value > $1.value }) {
        print("  " + phase.padding(toLength: 16, withPad: " ", startingAt: 0) + String(format: "%6.2f s  %4.1f %%", value, 100 * value / seconds))
        if let gpu = timings[phase.replacingOccurrences(of: " wall", with: " GPU")], phase.hasSuffix(" wall") {
            print("    of which GPU   " + String(format: "%6.2f s", gpu))
        }
    }
}

extension Measure {
    /// The same measure, its note replaced.
    func with(note: String) -> Measure {
        var m = Measure(boundary: boundary, worst: worst, median: median, channel: channel, atWitness: atWitness,
                        threshold: threshold, passed: passed, note: [self.note, note].filter { !$0.isEmpty }.joined(separator: "  "))
        m.kind = kind
        return m
    }
}
