import CryptoKit
import Foundation
import os
import Siliconed

// **Z-Image Turbo, stage by stage** — `siliconed check z-image <stage>`.
//
//     tokenizer → text → block → dit → trajectory → vae
//
// The goldens are the reference prompt's (`goldens-text.json` holds it), seed 42, written by diffusers and
// transformers in fp32 on the CPU. Each stage starts from the oracle's own inputs, so a deviation is
// located where it is born rather than inherited from the stage before.
//
// Every stage but the tokenizer runs on the engine, in the arithmetic Siliconed declares (bf16 as the reference runs in
// bf16, int8 on the Compact's codes, judged beside the publisher's own int8 products). So each is judged by the rule
// with the reference's own bf16 run beside the f32 oracle (`half`): no further from the exact value than diffusers in
// bf16 is, plus 25 %. A stage without its bf16 reference is skipped, with the script that writes it.

enum ZImageStages {
    static let all: [Stage] = [tokenizer, text, block, blockLoRA, dit, pace, trajectory, vae, img2img, render, renderLoRA]

    static let model = "z-image"
    static let ditMap = Family.zImage.ditMap(.standard)
    static let compactMap = Family.zImage.ditMap(.compact)
    static let encoderMap = Family.zImage.encoderMap(.standard)

    // MARK: - The text

    static let tokenizer = Stage(
        model: model, name: "tokenizer",
        proves: "the chat template and the tokenizer give transformers' 512 identifiers and mask, padding included, and the app's admission the same real tokens (with --long, past 512: 1024)",
        options: ["--long"], oracle: "tools/oracle/z-image.py text [--long]", weights: ["tokenizer"], rule: "bit for bit"
    ) { invocation in
        let stem = invocation.long ? "goldens-text-long" : "goldens-text"
        let report = Report(title: "\(model) tokenizer\(invocation.long ? " --long" : "")")
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
            let admitted = try tokenizer.admitChat(user: prompt).ids
            report.add(real.withUnsafeBufferPointer { Judge.integers("admitted (the app's path)", admitted, $0) })
        } catch { report.add(Judge.holds("admitted (the app's path)", false, note: "\(error)")) }
        return report
    }

    // MARK: - The text

    static let text = Stage(
        model: model, name: "text",
        proves: "the Qwen3-4B encoder (35 of its 36 layers, bf16) gives the reference's layer 33 and its cap_feats, hidden_states[-2], on the real tokens",
        options: ["--variant"], oracle: "tools/oracle/z-image.py text; text-bf16; with --variant compact, text and text-bf16 with --variant compact",
        weights: ["text encoder"], rule: "the rule against transformers in bf16 (a Compact's: on its SDNQ int8 dequantized)"
    ) { invocation in
        try Qwen3TextStages.zImage(invocation)
    }

    // MARK: - The DiT

    static let block = Stage(
        model: model, name: "block",
        proves: "layer 0 alone, from the oracle's own input, at its 19 inner boundaries: modulation, norms, projections, QK-norm, attention, SwiGLU (with --variant compact, the Compact's int8 codes)",
        options: ["--map", "--variant"],
        oracle: "tools/oracle/z-image.py dit --size {size}; block-bf16 --size {size}; with --variant compact, dit and w8a8 with --variant compact at 512", weights: ["DiT"],
        rule: "the rule against diffusers' layer 0 in bf16 on the same input"
    ) { invocation in
        let compact = try invocation.compact()
        let stem = { (s: String) in "\(s)-\(compact ? "compact-" : "")\(invocation.size)" }
        let report = Report(title: "\(model) block\(compact ? " compact" : "") \(invocation.size)", half: compact ? "W8A8" : "bf16")
        let path = try invocation.ditMap(ditMap, compactMap, compact: compact)
        let goldens = try invocation.goldens.open(stem("goldens-dit") + ".safetensors")
        // The Compact's layer 0 beside the publisher's int8 products (its W8A8 run holds layer 0's boundaries); the
        // Standard's beside diffusers' layer 0 in bf16 on the same input.
        let half = compact
            ? try invocation.goldens.int8(stem("witness-w8a8-dit") + ".safetensors", script: "tools/oracle/z-image.py w8a8 --size \(invocation.size) --variant compact")
            : try invocation.goldens.half("witness-bf16-block-\(invocation.size).safetensors", script: "tools/oracle/z-image.py block-bf16 --size \(invocation.size)")
        guard let unified = goldens.entries["unified_in"], let x = goldens.float32("unified_in"),
              let adaln = goldens.float32("block_adaln_input"), let caps = goldens.entries["in_caps"],
              let latent = goldens.entries["in_latent"] else {
            throw Skip("\(goldens.path): the block's inputs are missing")
        }
        let (h, w) = (latent.shape[2], latent.shape[3])
        let geometry = ZImageDiT(config: .published, latentHeight: h, latentWidth: w, capLength: caps.shape[0])
        guard geometry.imagePadded + geometry.capPadded == unified.shape[1] else {
            throw Skip("the golden's sequence is \(unified.shape[1]), the DiT's \(geometry.imagePadded + geometry.capPadded)")
        }
        let c = ZImageDiT.Config.published
        let widths: [ZImageDiT.BlockTap: Int] = [.adaln: 4 * c.dim, .w1: c.hidden, .w3: c.hidden, .swiglu: c.hidden]
        // The boundaries' arrays made and written before the layer's stage is planned, so that its floor counts them
        // (360 MB at 512), and judged once it is closed: made while it lived, they passed P by up to 1070 MiB.
        let taps = try ZImageLayerZero.tapCounts(latentHeight: h, latentWidth: w, capLength: caps.shape[0]).mapValues { n in
            let b = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
            b.initialize(repeating: .nan)
            return b
        }
        defer { taps.values.forEach { $0.deallocate() } }
        let described = try ZImageLayerZero.run(map: path, x: Array(x), adaln: Array(adaln), latentHeight: h, latentWidth: w,
                                                capLength: caps.shape[0], taps: taps)
        print("\(model) block: " + described)
        for tap in ZImageDiT.BlockTap.allCases {
            report.add(goldens, tap.rawValue, taps[tap].map { Array($0) }, half: half.float32(tap.rawValue).map(Array.init), columns: widths[tap] ?? c.dim)
        }
        return report
    }

    static let blockLoRA = Stage(
        model: model, name: "block-lora",
        proves: "each block function's low-rank path (a LoRA beside each product) against the CPU, product by product, on the render's layer 0 over the oracle's stream; with --variant compact, the Compact's shared forms; under --neural-engine on, the Standard's shared forms in f16 too, and the shared forms' ANE steps against the GPU's (the ANE's share told, as a render takes it)",
        options: ["--map", "--variant"],
        oracle: "tools/oracle/z-image.py dit --size {size}; with --variant compact, dit --variant compact (the inputs only)", weights: ["DiT"],
        rule: "each change (with its factors less without) against the CPU's x · down · up: ≤ 4 · 2⁻⁸ in bf16, ≤ 4 · 2⁻¹¹ in f16 (the result's and the path's roundings, the change as large as the result); each product's input and each function with every factor zero the same bits as without; on the ANE, the adapted function against the GPU's ≤ 2 × the same without + 2⁻⁹"
    ) { invocation in
        let compact = try invocation.compact()
        let report = Report(title: "\(model) block-lora\(compact ? " compact" : "")")
        let path = try invocation.ditMap(ditMap, compactMap, compact: compact)
        let goldens = try invocation.goldens.open("goldens-dit-\(compact ? "compact-" : "")512.safetensors")
        guard let x = goldens.float32("unified_in"), let adaln = goldens.float32("block_adaln_input"), let caps = goldens.entries["in_caps"],
              let latent = goldens.entries["in_latent"] else {
            throw Skip("\(goldens.path): the block's inputs are missing")
        }
        let (described, results) = try ZImageLowRankCheck.run(map: path, x: Array(x), adaln: Array(adaln), latentHeight: latent.shape[2],
                                                              latentWidth: latent.shape[3], capLength: caps.shape[0])
        print("\(model) block-lora: " + described)
        for r in results {
            report.add(r.bar.map { Judge.bound(r.name, r.value, $0, note: r.note) }
                       ?? Judge.holds(r.name, r.value == 0, note: r.value == 0 ? "bit for bit, " + r.note : "\(Int(r.value)) values differ, " + r.note))
        }
        return report
    }

    static let dit = Stage(
        model: model, name: "dit",
        proves: "one evaluation of the DiT, latent to velocity: the embedders, refiners, 30 layers, head and unpatchify (with --variant compact, the Compact's int8 codes)",
        sizes: ["512", "1024", "768x512", "832x1216"], options: ["--map", "--variant"],
        oracle: "tools/oracle/z-image.py dit --size {size}; witness --size {size}; bf16 --size {size}; with --variant compact, dit, witness and w8a8 with --variant compact at 512",
        weights: ["DiT"], rule: "the rule against the fp64 witness, with diffusers in bf16 beside the oracle (the W8A8 run for a Compact); at h ≠ w, the orientation counter-test"
    ) { invocation in
        let compact = try invocation.compact()
        // `goldens-dit-compact-512`, `witness-dit-f64-compact-512`: the variant before the size.
        let stem = { (s: String) in "\(s)-\(compact ? "compact-" : "")\(invocation.size)" }
        let report = Report(title: "\(model) dit\(compact ? " compact" : "") \(invocation.size)", half: compact ? "W8A8" : "bf16")
        let path = try invocation.ditMap(ditMap, compactMap, compact: compact)
        let goldens = try invocation.goldens.open(stem("goldens-dit") + ".safetensors")
        let witness = invocation.goldens.optional(stem("witness-dit-f64") + ".safetensors")
        let half = compact
            ? try invocation.goldens.int8(stem("witness-w8a8-dit") + ".safetensors", script: "tools/oracle/z-image.py w8a8 --size \(invocation.size) --variant compact")
            : try invocation.goldens.half(stem("witness-bf16-dit") + ".safetensors", script: "tools/oracle/z-image.py bf16 --size \(invocation.size)")
        guard let latentEntry = goldens.entries["in_latent"], let latent = goldens.float32("in_latent"),
              let capsEntry = goldens.entries["in_caps"], let caps = goldens.float32("in_caps"),
              let sigma = goldens.float32("in_sigma") else {
            throw Skip("\(goldens.path): the evaluation's inputs are missing")
        }
        let c = ZImageDiT.Config.published
        let (h, w, capLength) = (latentEntry.shape[2], latentEntry.shape[3], capsEntry.shape[0])
        // The boundaries' arrays made and written before the DiT is planned, so that its floor counts them, and judged
        // once its stage is closed: made while it lived, a tap's copy and its judging passed P by 90 MiB at 512 in bf16
        // (measured 19/11/2026).
        let taps = try ZImageRender.Denoiser.tapCounts(latentHeight: h, latentWidth: w, capLength: capLength).mapValues { n in
            let b = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
            b.initialize(repeating: .nan)
            return b
        }
        defer { taps.values.forEach { $0.deallocate() } }
        let velocity: [Float] = try {
            let denoiser = try ZImageRender.Denoiser(map: path, latentHeight: h, latentWidth: w, capLength: capLength)
            print("\(model) dit: " + denoiser.described)
            denoiser.caps = HostTensor([capLength, c.capDim], floats: Array(caps))
            return try denoiser.velocity(Array(latent), time: sigma[0], taps: taps)
        }()
        let widths: [ZImageDiT.Tap: Int] = [.tEmbedder: c.adalnDim, .freqs: 2, .finalLayer: c.patch * c.patch * c.channels]
        for tap in ZImageDiT.Tap.allCases {
            let name = tap.rawValue, got = taps[tap].map { Array($0) } ?? []
            guard let expected = goldens.float32(name) else { continue }
            let rows = { (b: UnsafeBufferPointer<Float>) in UnsafeBufferPointer(rebasing: b.prefix(got.count)) }
            report.add(Judge.boundary(name, engine: got, oracle: rows(expected), witness: witness?.float32(name).map(rows),
                                      half: half.float32(name).map { Array($0.prefix(got.count)) }, halfName: report.half, columns: widths[tap] ?? c.dim))
        }
        report.add(goldens, "model_out", velocity, witness: witness?.float32("model_out"),
                   half: half.float32("model_out").map(Array.init), columns: h * w)
        if let expected = goldens.float32("model_out"), let m = Judge.orientation(velocity, expected, height: h, width: w) {
            report.add(m)
        }
        // The velocity's global relative error beside the judge's channels, the reference's run beside (said, not judged).
        if let truth = witness?.float32("model_out") ?? goldens.float32("model_out") {
            func global(_ v: [Float]) -> Double {
                var (e, n) = (0.0, 0.0)
                for i in 0..<truth.count { let t = Double(truth[i]); e += (Double(v[i]) - t) * (Double(v[i]) - t); n += t * t }
                return (e / n).squareRoot()
            }
            print(String(format: "model_out, global relative error against the %@: %.3e (%@ %@)", witness == nil ? "oracle" : "fp64 witness",
                         global(velocity), report.half, half.float32("model_out").map { String(format: "%.3e", global(Array($0))) } ?? "—"))
        }
        return report
    }

    static let pace = Stage(
        model: model, name: "pace", proves: Pace.proves + " (with --variant compact, the Compact's)", sizes: ["1024", "512"],
        options: ["--renders", "--evaluations", "--map", "--variant"],
        oracle: nil, weights: ["DiT"], rule: Pace.rule
    ) { invocation in
        let compact = try invocation.compact()
        let path = try invocation.ditMap(ditMap, compactMap, compact: compact)
        let (h, w) = (invocation.height / LatentSpace.flux.factor, invocation.width / LatentSpace.flux.factor)
        let config = ZImageDiT.Config.published, capLength = 48
        let caps = HostTensor([capLength, config.capDim], floats: Pace.noise(capLength * config.capDim, seed: 1))
        return try Pace.run(invocation, title: "\(model) pace\(compact ? " compact" : "") \(invocation.size)") {
            let d = try ZImageRender.Denoiser(map: path, latentHeight: h, latentWidth: w, capLength: capLength)
            print("\(model) pace: " + d.described)
            d.caps = caps
            return d
        } evaluate: { d, e, time in
            try d.velocity(Pace.noise(config.channels * h * w, seed: UInt64(e + 2)), time: time)
        }
    }

    // MARK: - The sampler

    static let trajectory = Stage(
        model: model, name: "trajectory",
        proves: "the seed's noise bit for bit, the 8-step σ schedule, then each step's velocity from the reference's own latent at that step (the render's DiT session and its derivative, the velocity negated as the pipeline does)",
        sizes: ["512", "1024"], options: ["--map"],
        oracle: "tools/oracle/z-image.py trajectory --size {size}; trajectory-bf16 --size {size}; trajectory-witness --size {size} --step N for a step at its threshold", weights: ["DiT"],
        rule: "the noise and σ exact; each guided step by the rule against diffusers' same step in bf16, against its fp64 witness where one is written"
    ) { invocation in
        let report = Report(title: "\(model) trajectory \(invocation.size)")
        let path = try invocation.ditMap(ditMap)
        let reference = try invocation.goldens.open("goldens-trajectory-\(invocation.size).safetensors")
        let text = try invocation.goldens.open("goldens-text.safetensors")
        let half = try invocation.goldens.half("witness-bf16-trajectory-\(invocation.size).safetensors",
                                               script: "tools/oracle/z-image.py trajectory-bf16 --size \(invocation.size)")
        let manifest = try invocation.goldens.manifest("goldens-trajectory-\(invocation.size)")
        guard let initEntry = reference.entries["latent_init"], let initial = reference.float32("latent_init"),
              let caps = text.float32("cap_feats"), let capsEntry = text.entries["cap_feats"],
              let seed = manifest["seed"] as? NSNumber, let sigmas = manifest["sigmas"] as? [Double] else {
            throw Skip("\(reference.path): the trajectory's inputs are missing")
        }
        let (height, width) = (initEntry.shape[2], initEntry.shape[3])
        // The starting noise first: `torch.randn` at the manifest's seed, to the bit (`TorchNoise`).
        report.add(Judge.identical("noise (seed \(seed))", Latent.noise(.flux, height: height, width: width, seed: seed.uint64Value).values,
                                   initial))
        let schedule = FlowMatchSchedule(steps: sigmas.count - 1)
        report.add(sigmas.map(Float.init).withUnsafeBufferPointer { Judge.schedule(schedule.sigmas, $0) })
        // Then each step from the reference's own latent: each velocity is computed on the input the reference saw.
        let denoiser = try ZImageRender.Denoiser(map: path, latentHeight: height, latentWidth: width, capLength: capsEntry.shape[0])
        denoiser.caps = HostTensor(capsEntry.shape, floats: Array(caps))
        var seconds: [Double] = []
        // An evaluation's time, the first apart (it loads): worth reading only with nothing else running.
        defer { if seconds.count > 1 { print("evaluation · first \(String(format: "%.2f", seconds[0])) s · then median \(String(format: "%.3f", seconds.dropFirst().sorted()[(seconds.count - 1) / 2])) s (\(seconds.dropFirst().map { String(format: "%.3f", $0) }.joined(separator: " ")))") } }
        // A step at its threshold is judged against its fp64 witness, where one was written (`trajectory-witness`).
        try report.trajectory(reference, steps: sigmas.count - 1, half: half, columns: height * width, name: { "v_\($0)" }, witness: { step in
            invocation.goldens.optional("witness-f64-trajectory-\(invocation.size)-v\(step).safetensors")?.float32("v_\(step)")
        }) { step, latent in
            let start = Date()
            defer { seconds.append(Date().timeIntervalSince(start)) }
            // The render's own call (`RenderLoop`): its time and its sign, under the reference's Euler (the render's default).
            return try denoiser.derivative(latent, sigma: schedule.sigmas[step], solver: .euler, first: step == 0)
        }
        return report
    }

    // MARK: - The decoder

    static let vae = Stage(
        model: model, name: "vae",
        proves: "Flux's decoder in bf16 on a rectangular latent (the centered crop of the final 1024² latent), plane by plane: what a square cannot see",
        sizes: ["768x512"],
        oracle: "tools/oracle/z-image.py vae --size {size}; vae-witness --size {size}; vae-bf16 --size {size}", weights: ["VAE"],
        rule: "per plane, at the fp64 witness: no further from it than diffusers decoding in bf16, + 25 %; on a rectangle, the transposed image must deviate"
    ) { invocation in
        let report = Report(title: "\(model) vae \(invocation.size)")
        let goldens = try invocation.goldens.open("goldens-vae-\(invocation.size).safetensors")
        let witness = invocation.goldens.optional("witness-vae-f64-\(invocation.size).safetensors")
        let half = try invocation.goldens.half("witness-bf16-vae-\(invocation.size).safetensors",
                                               script: "tools/oracle/z-image.py vae-bf16 --size \(invocation.size)")
        guard let entry = goldens.entries["latent"], let latent = goldens.float32("latent"),
              let expected = goldens.float32("image") else {
            throw Skip("\(goldens.path): latent or image missing")
        }
        let (h, w) = (entry.shape[1], entry.shape[2])
        let path = try Library.processWide.component(.zImage, "vae.safetensors")
        let image = try DecoderSession<FluxVAEDecoder>(path: path, latentHeight: h, latentWidth: w).pixels(Array(latent))
        report.add(Judge.planes("image RGB", engine: image, oracle: expected, witness: witness?.float32("image"),
                                reference: half.float32("image").map(Array.init), planes: 3))
        if let m = Judge.orientation(image, expected, height: 8 * h, width: 8 * w) { report.add(m) }
        return report
    }

    // MARK: - An image to start from

    static let img2img = Stage(
        model: model, name: "img2img",
        proves: "an image to start from: Flux's encoder in bf16 on the oracle's picture (the posterior mean, shifted and scaled), the first step run at strength 0.6 and its σ, the starting latent σ·ε + (1 − σ)·z₀ from the reference's own z₀ and noise, then the app's render from the picture",
        sizes: ["512"], options: ["--png"],
        oracle: "tools/oracle/z-image.py text; img2img --size {size}; img2img-bf16 --size {size}", weights: ["text encoder", "DiT", "VAE"],
        rule: "z₀ per latent plane by the rule against diffusers encoding in bf16; the first step and σ exact; the starting latent bit for bit; the render finite, its PSNR against the reference's final latent decoded here said and not judged (a free loop in bf16)"
    ) { invocation in
        let report = Report(title: "\(model) img2img \(invocation.size)")
        let vae = try Library.processWide.component(.zImage, "vae.safetensors")
        let reference = try StartImageReference(
            report, invocation, stem: "goldens-img2img-\(invocation.size)", witness: ("witness-bf16-img2img-\(invocation.size)", "tools/oracle/z-image.py img2img-bf16 --size \(invocation.size)"),
            space: .flux,
            schedule: { FlowMatchSchedule(steps: $0).sigmas }, encode: { try FluxVAEEncoder.latent(of: $0, path: vae) })
        guard let prompt = try invocation.goldens.manifest("goldens-text")["prompt"] as? String else {
            throw Skip("goldens-text: the prompt is missing")
        }
        let model = try Model.zImage(in: Library.processWide, card: .zImage, dit: try invocation.ditMap(ditMap),
                                     encoder: Family.zImage.encoderMap(.standard))
        let render = try Engine().render(reference.request(prompt), model: model)
        if let png = invocation.extras["--png"] { try render.png().write(to: URL(fileURLWithPath: png)) }
        let (h, w) = (reference.latentHeight, reference.latentWidth)
        let decoded = try DecoderSession<FluxVAEDecoder>(path: vae, latentHeight: h, latentWidth: w).pixels(reference.final)
        reference.judge(report, render, decoded: decoded)
        return report
    }

    // MARK: - The whole render

    static let render = Stage(
        model: model, name: "render",
        proves: "the app's render (Engine.render: tokenizer, encoder, 8 Euler steps, decoder) from the reference prompt and seed: the reference's final latent and picture",
        sizes: ["512", "1024"], options: ["--variant", "--png", "--f32"],
        oracle: "tools/oracle/z-image.py trajectory --size {size}", weights: ["text encoder", "DiT", "VAE"],
        rule: "the same picture: PSNR past 25 dB against diffusers in fp32 (a bound written before the run, until a bf16 trajectory judges each step); with --variant compact, a finite picture, its PSNR said and not judged: the Compact is judged at the DiT"
    ) { invocation in
        let compact = try invocation.compact()
        let report = Report(title: "\(model) render\(compact ? " compact" : "") \(invocation.size)")
        let reference = try invocation.goldens.open("goldens-trajectory-\(invocation.size).safetensors")
        guard let prompt = try invocation.goldens.manifest("goldens-text")["prompt"] as? String,
              let seed = (try invocation.goldens.manifest("goldens-trajectory-\(invocation.size)"))["seed"] as? NSNumber,
              let expected = reference.float32("image_raw") else {
            throw Skip("\(reference.path): the prompt, seed or image is missing")
        }
        let library = Library.processWide
        let path = try invocation.ditMap(ditMap, compactMap, compact: compact)
        // The encoder of the version judged, not the library's choice.
        let model = try Model.zImage(in: library, card: .zImage, dit: path,
                                     encoder: Family.zImage.encoderMap(compact ? .compact : .standard))
        print("encoder \(Library.withoutHome((model.chain.text as? ZImageTextModule)?.map ?? "?"))")
        let started = Date()
        let steps = OSAllocatedUnfairLock(initialState: [Double]())
        let render = try Engine().render(Request(prompt, width: invocation.width, height: invocation.height,
                                                 seed: seed.uint64Value, steps: 8), model: model) { event in
            if case .step(_, _, _, _, let seconds, _, _) = event { steps.withLock { $0.append(seconds) } }
        }
        try report.rendered(render, since: started, invocation,
                            more: "; the steps " + steps.withLock { $0 }.map { String(format: "%.2f", $0) }.joined(separator: " ") + " s")
        // The pixels' bits, `[3, H, W]` f32 as the decoder gives them: what another computation of the same render compares.
        let raw = render.image.pixels.withUnsafeBytes { Data($0) }
        print("image md5 " + Insecure.MD5.hash(data: raw).map { String(format: "%02x", $0) }.joined().prefix(12))
        if let f32 = invocation.extras["--f32"] { try raw.write(to: URL(fileURLWithPath: f32)) }
        report.judgePicture(render.image.pixels, expected, compact: compact)
        return report
    }

    /// No oracle of a LoRA for Z-Image (`z-image.py lora` not written): its functions are proved against the CPU
    /// (`block-lora`); here the render that carries them.
    static let renderLoRA = Stage(
        model: model, name: "render-lora",
        proves: "the app's render with a LoRA (--lora FILE, strength 1) from the reference prompt and seed: the LoRA reaches the picture",
        sizes: ["512", "1024"], options: ["--variant", "--png", "--lora"],
        oracle: "tools/oracle/z-image.py trajectory --size {size} (the prompt and seed only)", weights: ["text encoder", "DiT", "VAE", "the LoRA"],
        rule: "the picture finite, and apart from the same render without the LoRA: PSNR between them under 40 dB (a counter-test: a LoRA that reached nothing gives the same picture)"
    ) { invocation in
        let compact = try invocation.compact()
        let report = Report(title: "\(model) render-lora\(compact ? " compact" : "") \(invocation.size)")
        guard let lora = invocation.extras["--lora"] else { throw Skip("no LoRA given: --lora FILE, a LoRA of Z-Image") }
        guard let prompt = try invocation.goldens.manifest("goldens-text")["prompt"] as? String,
              let seed = (try invocation.goldens.manifest("goldens-trajectory-\(invocation.size)"))["seed"] as? NSNumber else {
            throw Skip("goldens-text, goldens-trajectory-\(invocation.size): the prompt or the seed is missing")
        }
        let path = try invocation.ditMap(ditMap, compactMap, compact: compact)
        let model = try Model.zImage(in: Library.processWide, card: .zImage, dit: path, encoder: Family.zImage.encoderMap(compact ? .compact : .standard))
        func render(_ loras: [LoRAEntry]) throws -> Engine.Render {
            try Engine().render(Request(prompt, width: invocation.width, height: invocation.height, seed: seed.uint64Value, steps: 8, loras: loras), model: model)
        }
        let adapted = try render([LoRAEntry(lora)]), plain = try render([])
        if let png = invocation.extras["--png"] { try adapted.png().write(to: URL(fileURLWithPath: png)) }
        let (psnr, _, _, notFinite) = plain.image.pixels.withUnsafeBufferPointer { imageError(adapted.image.pixels, $0) }
        report.add(Judge.holds("the picture finite", notFinite == 0, note: notFinite == 0 ? "" : "\(notFinite) non-finite values"))
        report.add(Judge.bound("PSNR against the render without the LoRA", psnr, 40, unit: "dB"))
        return report
    }
}
