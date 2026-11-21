import Foundation
import SiliCore
import Siliconed

// **Krea 2 Turbo, stage by stage** — `siliconed check krea2 <stage>`.
//
//     text → dit → dit-lora → trajectory → vae → render
//
// What sets it apart: the text is not one encoder's output but twelve taps of Qwen3-VL (layers spread over its depth), fused
// by the DiT's own text fusion and `txt_in`; the reference pads the prompt in the middle and masks, the engine removes the
// padding, so the taps are judged against states computed WITH it. The VAE is Qwen-Image's, with Krea's weights. There is no
// separate block stage: the DiT's boundaries include block 0 inside out (attention, feed-forward, output).
//
// Every stage but the tokenizer's identifiers runs on the engine, in the arithmetic Siliconed declares (bf16 as the
// reference runs in bf16, int8 on the Compact's codes, judged beside the publisher's own int8 products). So each is judged by the rule
// with the reference's own bf16 run beside the f32 oracle (`half`); a stage without its bf16 reference is skipped, with the
// script that writes it.

enum Krea2Stages {
    static let all: [Stage] = [text, dit, ditLoRA, pace, trajectory, vae, img2img, render]

    static let model = "krea2"
    static let ditMap = Family.krea2.ditMap(.standard)
    static let compactMap = Family.krea2.ditMap(.compact)
    static let encoderMap = Family.krea2.encoderMap(.standard)

    // MARK: - The DiT under a LoRA

    /// The bars, written before the first run (19/11): the fusion's change computed on the same bf16 weights in both
    /// files, its low rank the engine's own (`LowRankRun`), within its bf16 rounding; block 0's, the Compact's int8
    /// products and f16 forms against the Standard's bf16 at the first block, where no chaos has grown — a wrong factor,
    /// basis or row gives an error of order 1.
    static let loraBars = (fusion: 1e-2, block: 5e-2)

    /// The boundaries' buffers, made and written before the DiT is planned, so that its floor counts them, and judged once
    /// its stage is closed (Z-Image's).
    static func boundaries(latentHeight: Int, latentWidth: Int, text: Int) -> Krea2Render.Denoiser.Taps {
        Krea2Render.Denoiser.tapCounts(latentHeight: latentHeight, latentWidth: latentWidth, text: text).mapValues { n in
            let b = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
            b.initialize(repeating: .nan)
            return b
        }
    }

    static let ditLoRA = Stage(
        model: model, name: "dit-lora",
        proves: "one evaluation of the DiT at σ = 1 under a user's LoRA read in its published file (strength 1, every module applied: the 28 blocks' and the text fusion's), the Compact's change against the Standard's: what the LoRA moves in the fused text, in block 0's attention, feed-forward and output, and in the velocity",
        sizes: ["512"], options: ["--map", "--lora"],
        oracle: nil, weights: ["DiT", "the Compact's DiT", "the LoRA"],
        rule: "no reference of Krea 2 under a LoRA: the change each brings (with − without) on the Compact (its shared forms, the ANE's share under --neural-engine on) against the Standard's (bf16, the engine's low rank checked against the CPU by LowRankRun); text_fused within 1e-2, block 0's boundaries within 5e-2; the velocity's said, not judged (28 blocks of int8 against bf16)"
    ) { invocation in
        let report = Report(title: "\(model) dit-lora \(invocation.size)")
        let compactPath = try invocation.ditMap(compactMap, golden: .eightBit)
        let standardPath = Library.processWide.map(ditMap)
        guard FileManager.default.fileExists(atPath: standardPath) else { throw Skip("map missing: \(standardPath) — siliconed install") }
        let folder = Library.processWide.loraFolder(.krea2)
        guard let lora = invocation.extras["--lora"]
                ?? (try? FileManager.default.contentsOfDirectory(atPath: folder))?.filter({ $0.hasSuffix(".safetensors") }).sorted().first.map({ folder + "/" + $0 })
        else { throw Skip("no LoRA in \(folder): --lora FILE") }
        let loras = try LoRAStack([LoRAEntry(lora)], on: .krea2)
        report.add(Judge.holds("the LoRA recognized, every module applied", true, note: (lora as NSString).lastPathComponent))
        let goldens = try invocation.goldens.open("krea2/goldens-krea2-dit-512.safetensors")
        let text = try invocation.goldens.open("krea2/goldens-krea2-text.safetensors")
        guard let latentEntry = goldens.entries["latent_in"], let latent = goldens.float32("latent_in"), let sigma = goldens.float32("timestep"),
              let tapsEntry = text.entries["hidden_valid"], let taps = text.float32("hidden_valid") else {
            throw Skip("\(goldens.path): the evaluation's inputs are missing")
        }
        let told = Float(bitPattern: UInt32(BFloat16Bits.rounded(sigma[0])) << 16)
        let names = ["text_fused", "block0_attn_out", "block0_ff_out", "block0_out", "model_out"]
        // A DiT under the LoRA, let go, then one under the same LoRA at strength 0: the same functions, their factors
        // zeros, so the change is the products' alone. Each stage alone, its text fused before it opens (as the render
        // fuses it), the first run's results in the second's floor — two stages at once, or results held past P, would
        // judge memory the stages do not declare.
        let zeroed = LoRAStack(entries: loras.entries.map { ($0.file, 0) })
        let hidden = HostTensor(tapsEntry.shape, floats: Array(taps))
        func run(_ loras: LoRAStack, _ path: String) throws -> [String: [Float]] {
            let taps = boundaries(latentHeight: latentEntry.shape[1], latentWidth: latentEntry.shape[2], text: tapsEntry.shape[0])
            defer { taps.values.forEach { $0.deallocate() } }
            let (velocity, fused): ([Float], [Float]) = try {
                let d = try Krea2Render.Denoiser(map: path, loras: loras, taps: hidden, latentHeight: latentEntry.shape[1],
                                                 latentWidth: latentEntry.shape[2], boundaries: true)
                return (try d.velocity(Array(latent), time: told, taps: taps), d.text!.fused)
            }()
            var out = Dictionary(uniqueKeysWithValues: taps.map { ($0.key.rawValue, Array($0.value)) })
            out["text_fused"] = fused
            out["model_out"] = velocity
            return out.filter { names.contains($0.key) }
        }
        func change(_ path: String) throws -> [String: [Float]] {
            let adapted = try run(loras, path)
            let bare = try run(zeroed, path)
            return Dictionary(uniqueKeysWithValues: names.map { n in (n, zip(adapted[n]!, bare[n]!).map { $0 - $1 }) })
        }
        let standard = try change(standardPath), compact = try change(compactPath)
        for n in names {
            let (c, s) = (compact[n]!, standard[n]!)
            var e = 0.0, m = 0.0
            for i in s.indices { let x = Double(c[i] - s[i]); e += x * x; m += Double(s[i]) * Double(s[i]) }
            let error = (e / max(m, .leastNormalMagnitude)).squareRoot()
            let size = (m / Double(s.count)).squareRoot()
            let note = String(format: "the change's rms %.3g on the Standard", size)
            switch n {
            case "model_out": report.add(Judge.bound(n + " change, said", error, .infinity, unit: "relative", note: note))
            case "text_fused": report.add(Judge.bound(n + " change", error, loraBars.fusion, unit: "relative", note: note))
            default: report.add(Judge.bound(n + " change", error, loraBars.block, unit: "relative", note: note))
            }
        }
        return report
    }

    // MARK: - The DiT's pace

    static let pace = Stage(
        model: model, name: "pace", proves: Pace.proves, sizes: ["1024", "512"], options: ["--renders", "--evaluations", "--map", "--variant"],
        oracle: nil, weights: ["DiT"], rule: Pace.rule
    ) { invocation in
        let compact = try invocation.compact()
        let path = try invocation.ditMap(compact ? compactMap : ditMap, golden: compact ? .eightBit : .standard)
        let (h, w) = (invocation.height / 8, invocation.width / 8), rows = 48
        let taps = HostTensor([rows, Krea2Text.sockets.count, Krea2Functions.textWidth],
                              floats: Pace.noise(rows * Krea2Text.sockets.count * Krea2Functions.textWidth, seed: 1))
        return try Pace.run(invocation, title: "\(model) pace\(compact ? " compact" : "") \(invocation.size)") {
            let d = try Krea2Render.Denoiser(map: path, taps: taps, latentHeight: h, latentWidth: w)
            print("krea2 pace: " + d.described)
            return d
        } evaluate: { d, e, time in
            try (d as! Krea2Render.Denoiser).velocity(Pace.noise(16 * h * w, seed: UInt64(e + 2)), time: time)
        }
    }

    // MARK: - The text

    static let text = Stage(
        model: model, name: "text",
        proves: "the prompt's identifiers, then Qwen3-VL-4B's twelve taps (bf16, the real tokens alone, judged against states computed with the reference's padding)",
        oracle: "tools/oracle/krea2.py text; text-bf16", weights: ["text encoder"],
        rule: "identifiers bit for bit; the rule against transformers in bf16"
    ) { invocation in
        let report = Report(title: "\(model) text")
        let goldens = try invocation.goldens.open("krea2/goldens-krea2-text.safetensors")
        let half = try invocation.goldens.half("krea2/witness-bf16-krea2-text.safetensors", script: "tools/oracle/krea2.py text-bf16")
        guard let prompt = try invocation.goldens.manifest("krea2/goldens-krea2-text")["prompt"] as? String,
              let paddedIds = goldens.int32("ids"), let mask = goldens.int32("mask"),
              let validEntry = goldens.entries["hidden_valid"] else {
            throw Skip("\(goldens.path): prompt, ids, mask or hidden_valid missing")
        }
        // The reference pads in the middle and masks: its valid tokens are the engine's whole prompt.
        let valid = zip(paddedIds, mask).filter { $0.1 != 0 }.map { $0.0 }
        let tokenizer = try Tokenizer(directory: try Library.processWide.component(.krea2, "tokenizer"))
        let ids = Krea2Text.identifiers(prompt, tokenizer: tokenizer)
        report.add(valid.withUnsafeBufferPointer { Judge.integers("tokens", ids, $0) })
        // The computation starts from the golden's identifiers: one thing judged at a time.
        let path = try MapCheck.use(Library.processWide.map(encoderMap))
        let sockets = Krea2Text.sockets, width = 2560, t = validEntry.shape[0]
        let hidden = try Krea2Render.encode(valid.map(Int.init), map: path, sockets: sockets)
        report.add(goldens, "Qwen3-VL taps", hidden.floats, name: "hidden_valid", half: half.float32("hidden_valid").map(Array.init),
                   columns: width)
        // Each tap alone: a deviation that grows with depth shows here.
        let taps = sockets.count
        let tap = { (values: [Float], slot: Int) -> [Float] in
            (0..<t).flatMap { row in values[((row * taps + slot) * width)..<((row * taps + slot + 1) * width)] }
        }
        if let oracle = goldens.float32("hidden_valid"), let reference = half.float32("hidden_valid") {
            let own = Array(oracle), ref = Array(reference)
            for slot in 0..<taps {
                
                let expected = tap(own, slot)
                report.add(expected.withUnsafeBufferPointer {
                    Judge.boundary("tap \(sockets[slot])", engine: tap(hidden.floats, slot), oracle: $0, half: tap(ref, slot), columns: width)
                })
            }
        }
        return report
    }

    // MARK: - The DiT

    static let dit = Stage(
        model: model, name: "dit",
        proves: "one evaluation of the DiT at σ = 1, boundary by boundary: the timestep path, the text fusion and txt_in, img_in, block 0 inside out (attention, feed-forward), blocks 1, 13 and 27, and the velocity per latent plane (with --variant compact, the Compact's int8 codes)",
        sizes: ["512", "1024", "768x512"], options: ["--map", "--variant"],
        oracle: "tools/oracle/krea2.py dit --size {size}; witness --size {size}; witness-step --size {size} --step 0; bf16 --size {size}; with --variant compact, dit, w8a8, witness and witness-step --step 0 with --variant compact at 512",
        weights: ["DiT"],
        rule: "the rule against the fp64 witness where it exists, with diffusers in bf16 beside the oracle (the W8A8 run for a Compact); model_out per latent plane at the witness; at h ≠ w, the orientation counter-test"
    ) { invocation in
        let compact = try invocation.compact()
        // `goldens-krea2-dit-compact-512`: the variant before the size.
        let stem = "\(compact ? "compact-" : "")\(invocation.size)"
        let report = Report(title: "\(model) dit\(compact ? " compact" : "") \(invocation.size)", half: compact ? "W8A8" : "bf16")
        let path = try invocation.ditMap(compact ? compactMap : ditMap, golden: compact ? .eightBit : .standard)
        let goldens = try invocation.goldens.open("krea2/goldens-krea2-dit-\(stem).safetensors")
        let text = try invocation.goldens.open("krea2/goldens-krea2-text.safetensors")
        let half = compact
            ? try invocation.goldens.int8("krea2/witness-w8a8-krea2-dit-\(stem).safetensors", script: "tools/oracle/krea2.py w8a8 --size \(invocation.size) --variant compact")
            : try invocation.goldens.half("krea2/witness-bf16-krea2-dit-\(stem).safetensors", script: "tools/oracle/krea2.py bf16 --size \(invocation.size)")
        // `witness-krea2-f64-R` stops after block 1; `-step0` is the whole evaluation in double.
        let witnesses = ["krea2/witness-krea2-f64-\(stem).safetensors", "krea2/witness-krea2-f64-\(stem)-step0.safetensors"]
            .compactMap { invocation.goldens.optional($0) }
        let exact = { (name: String) in witnesses.lazy.compactMap { $0.float32(name) }.first }
        guard let latentEntry = goldens.entries["latent_in"], let latent = goldens.float32("latent_in"),
              let sigma = goldens.float32("timestep"), let tapsEntry = text.entries["hidden_valid"],
              let taps = text.float32("hidden_valid") else {
            throw Skip("\(goldens.path): the evaluation's inputs are missing")
        }
        let (h, w) = (latentEntry.shape[1], latentEntry.shape[2])
        // The bf16 reference is told its timestep in bf16 (`timestep.to(half)`).
        let told = Float(bitPattern: UInt32(BFloat16Bits.rounded(sigma[0])) << 16)
        let buffers = boundaries(latentHeight: h, latentWidth: w, text: tapsEntry.shape[0])
        defer { buffers.values.forEach { $0.deallocate() } }
        let (velocity, fusion): ([Float], (fused: [Float], rows: [Float])) = try {
            let denoiser = try Krea2Render.Denoiser(map: path, taps: HostTensor(tapsEntry.shape, floats: Array(taps)),
                                                    latentHeight: h, latentWidth: w, boundaries: true)
            return (try denoiser.velocity(Array(latent), time: told, taps: buffers), denoiser.text!)
        }()
        var out = Dictionary(uniqueKeysWithValues: buffers.map { ($0.key.rawValue, Array($0.value)) })
        out["text_fused"] = fusion.fused
        out["txt_in_out"] = fusion.rows
        let widths: [String: Int] = ["temb_mod": Krea2Functions.d, "text_fused": Krea2Functions.textWidth]
        for tap in Krea2Transformer.Tap.allCases {
            let name = tap.rawValue
            report.add(goldens, name, out[name], witness: exact(name), half: half.float32(name).map(Array.init),
                       columns: widths[name] ?? Krea2Functions.d)
        }
        if let expected = goldens.float32("model_out") {
            if let witness = exact("model_out") {
                report.add(Judge.planes("model_out per plane", engine: velocity, oracle: expected, witness: witness,
                                        reference: half.float32("model_out").map(Array.init), planes: 16))
            } else {
                report.add(goldens, "model_out", velocity, half: half.float32("model_out").map(Array.init), columns: h * w)
            }
            if let m = Judge.orientation(velocity, expected, height: h, width: w) { report.add(m) }
        }
        return report
    }

    // MARK: - The sampler

    static let trajectory = Stage(
        model: model, name: "trajectory",
        proves: "the seed's noise bit for bit, the 8-step σ schedule, then each step's velocity from the reference's own latent at that step (the render's DiT session)",
        sizes: ["512", "1024"], options: ["--map"],
        oracle: "tools/oracle/krea2.py trajectory --size {size}; trajectory-bf16 --size {size}; witness-step --size {size} --step N for a step at its threshold", weights: ["DiT"],
        rule: "the noise and σ exact; each step by the rule against diffusers' same step in bf16, against its fp64 witness where one is written"
    ) { invocation in
        let report = Report(title: "\(model) trajectory \(invocation.size)")
        let path = try invocation.ditMap(ditMap)
        let reference = try invocation.goldens.open("krea2/goldens-krea2-trajectory-\(invocation.size).safetensors")
        let text = try invocation.goldens.open("krea2/goldens-krea2-text.safetensors")
        let half = try invocation.goldens.half("krea2/witness-bf16-krea2-trajectory-\(invocation.size).safetensors",
                                               script: "tools/oracle/krea2.py trajectory-bf16 --size \(invocation.size)")
        guard let initEntry = reference.entries["latent_init"], let initial = reference.float32("latent_init"),
              let sigmas = reference.float32("sigmas"), let tapsEntry = text.entries["hidden_valid"],
              let taps = text.float32("hidden_valid"),
              let seed = (try invocation.goldens.manifest("krea2/goldens-krea2-trajectory-\(invocation.size)"))["seed"] as? NSNumber else {
            throw Skip("\(reference.path): the trajectory's inputs are missing")
        }
        let (height, width) = (initEntry.shape[1], initEntry.shape[2])
        report.add(Judge.identical("noise (seed \(seed))", Latent.noise(.qwenImage, height: height, width: width, seed: seed.uint64Value).values,
                                   initial))
        let steps = sigmas.count - 1
        let schedule = Krea2DiT.sigmas(steps: steps)
        report.add(Judge.schedule(schedule, sigmas))
        let denoiser = try Krea2Render.Denoiser(map: path, taps: HostTensor(tapsEntry.shape, floats: Array(taps)),
                                                latentHeight: height, latentWidth: width)
        for step in 0..<steps {
            let name = "velocity_\(step)"
            guard let latent = reference.float32(step == 0 ? "latent_init" : "latent_\(step)") else { continue }
            let v = try denoiser.velocity(Array(latent), time: Krea2Render.modelTime(sigma: schedule[step], solver: .euler, first: step == 0))
            let witness = invocation.goldens.optional("krea2/witness-krea2-f64-\(invocation.size)-step\(step).safetensors")
            report.add(reference, name, v, witness: witness?.float32("model_out"), half: half.float32(name).map(Array.init),
                       columns: height * width)
        }
        return report
    }

    // MARK: - The decoder

    static let vae = Stage(
        model: model, name: "vae",
        proves: "the Qwen-Image decoder in bf16 (Krea's weights) on the final latent of the oracle's trajectory (a centered crop of the 1024² one at 768x512), plane by plane",
        sizes: ["512", "1024", "768x512"],
        oracle: "tools/oracle/krea2.py vae --size {size}; vae-witness --size {size}; vae-bf16 --size {size}", weights: ["VAE"],
        rule: "per plane, at the fp64 witness: no further from it than diffusers decoding in bf16, + 25 %; on a rectangle, the transposed image must deviate"
    ) { invocation in
        let report = Report(title: "\(model) vae \(invocation.size)")
        let goldens = try invocation.goldens.open("krea2/goldens-krea2-vae-\(invocation.size).safetensors")
        let witness = invocation.goldens.optional("krea2/witness-krea2-vae-f64-\(invocation.size).safetensors")
        let half = try invocation.goldens.half("krea2/witness-bf16-krea2-vae-\(invocation.size).safetensors",
                                               script: "tools/oracle/krea2.py vae-bf16 --size \(invocation.size)")
        guard let entry = goldens.entries["latent_final"], let latent = goldens.float32("latent_final"),
              let expected = goldens.float32("image") else {
            throw Skip("\(goldens.path): latent_final or image missing")
        }
        let (h, w) = (entry.shape[1], entry.shape[2])
        let image = try qwenImageDecoded(Array(latent), height: h, width: w, vae: try Library.processWide.component(.krea2, "vae.safetensors"))
        report.add(Judge.planes("image RGB", engine: image, oracle: expected, witness: witness?.float32("image"),
                                reference: half.float32("image").map(Array.init), planes: 3))
        if let m = Judge.orientation(image, expected, height: 8 * h, width: 8 * w) { report.add(m) }
        return report
    }

    // MARK: - The whole render

    static let img2img = Stage(
        model: model, name: "img2img",
        proves: "an image to start from: the VAE's encoder in bf16 on the oracle's picture (the posterior mean, normalized), the first step run at strength 0.6 and its σ, the starting latent σ·ε + (1 − σ)·z₀ from the reference's own z₀ and noise, then the app's render from the picture",
        sizes: ["512"], options: ["--png"],
        oracle: "tools/oracle/krea2.py text; img2img --size {size}; img2img-bf16 --size {size}", weights: ["text encoder", "DiT", "VAE"],
        rule: "z₀ per latent plane by the rule against diffusers encoding in bf16; the first step and σ exact; the starting latent bit for bit; the render finite, its PSNR against the reference's final latent decoded here said and not judged (a free loop in bf16)"
    ) { invocation in
        let report = Report(title: "\(model) img2img \(invocation.size)")
        let vae = try Library.processWide.component(.krea2, "vae.safetensors")
        let reference = try StartImageReference(
            report, invocation, stem: "krea2/goldens-krea2-img2img-\(invocation.size)",
            witness: ("krea2/witness-bf16-krea2-img2img-\(invocation.size)", "tools/oracle/krea2.py img2img-bf16 --size \(invocation.size)"),
            space: .qwenImage, schedule: { Krea2DiT.sigmas(steps: $0) }, encode: { try QwenImageVAEEncoder.latent(of: $0, path: vae) })
        guard let prompt = try invocation.goldens.manifest("krea2/goldens-krea2-text")["prompt"] as? String else {
            throw Skip("krea2/goldens-krea2-text: the prompt is missing")
        }
        let model = try Model.krea2(in: Library.processWide, card: .krea2, dit: try invocation.ditMap(ditMap),
                                    encoder: Library.processWide.map(Family.krea2.encoderMap(.standard)))
        let render = try Engine().render(reference.request(prompt), model: model)
        if let png = invocation.extras["--png"] { try render.png().write(to: URL(fileURLWithPath: png)) }
        reference.judge(report, render, decoded: try qwenImageDecoded(reference.final, height: reference.latentHeight, width: reference.latentWidth, vae: vae))
        return report
    }

    static let render = Stage(
        model: model, name: "render",
        proves: "the app's render (Engine.render: tokenizer, encoder, 8 Euler steps, decoder) from the reference prompt and seed: the reference's final picture",
        sizes: ["512", "1024"], options: ["--variant", "--png"],
        oracle: "tools/oracle/krea2.py trajectory --size {size}; vae --size {size}", weights: ["text encoder", "DiT", "VAE"],
        rule: "the same picture: PSNR past 25 dB against diffusers in fp32 (a bound written before the run, Z-Image's); with --variant compact, a finite picture, its PSNR said and not judged: the Compact is judged at the DiT"
    ) { invocation in
        let compact = try invocation.compact()
        let report = Report(title: "\(model) render\(compact ? " compact" : "") \(invocation.size)")
        let image = try invocation.goldens.open("krea2/goldens-krea2-vae-\(invocation.size).safetensors")
        guard let prompt = try invocation.goldens.manifest("krea2/goldens-krea2-text")["prompt"] as? String,
              let seed = (try invocation.goldens.manifest("krea2/goldens-krea2-trajectory-\(invocation.size)"))["seed"] as? NSNumber,
              let expected = image.float32("image") else {
            throw Skip("\(image.path): the prompt, seed or image is missing")
        }
        let path = try invocation.ditMap(compact ? compactMap : ditMap, golden: compact ? .eightBit : .standard)
        // The encoder of the version judged, not the library's choice.
        let encoder = Library.processWide.map(Family.krea2.encoderMap(compact ? .compact : .standard))
        let model = try Model.krea2(in: Library.processWide, card: .krea2, dit: path, encoder: encoder)
        let started = Date()
        let render = try Engine().render(Request(prompt, width: invocation.width, height: invocation.height,
                                                 seed: seed.uint64Value, steps: 8), model: model)
        print(String(format: "rendered in %.1f s (text %.2f, denoising %.2f, decoding %.2f)", Date().timeIntervalSince(started),
                     render.timings.text, render.timings.denoising, render.timings.decoding))
        if let png = invocation.extras["--png"] { try render.png().write(to: URL(fileURLWithPath: png)) }
        if compact { report.add(Judge.compactRender(render.image.pixels, expected)); return report }
        let (psnr, _, _, notFinite) = imageError(render.image.pixels, expected)
        report.add(Judge.bound("image PSNR", psnr, 25, unit: "dB", above: true,
                               note: notFinite > 0 ? "\(notFinite) non-finite values" : ""))
        return report
    }
}
