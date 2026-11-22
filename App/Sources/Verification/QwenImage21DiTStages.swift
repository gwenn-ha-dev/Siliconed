import Foundation
import Siliconed

// **Qwen-Image-2.1's DiT, stage by stage**: `pace`, `dit` and its `Evaluation`, `dit-lora`.
extension QwenImage21Stages {
    static let pace = Stage(
        model: model, name: "pace", proves: Pace.proves + " (with --variant compact, the Compact's)",
        sizes: ["1024", "512"], options: ["--renders", "--evaluations", "--map", "--variant"],
        oracle: nil, weights: ["DiT"], rule: Pace.rule
    ) { invocation in
        let compact = try invocation.compact()
        let map = try invocation.ditMap(ditMap, compactMap, compact: compact)
        let space = LatentSpace.qwenImage21, (h, w) = (invocation.height / space.factor, invocation.width / space.factor)
        let c = QwenImage21DiT.Config.published, textRows = 48
        let sequence = try QwenImage21DiT.Sequence(slots: [Bool](repeating: false, count: textRows), references: [], height: h, width: w)
        let text = HostTensor([textRows, c.context], floats: Pace.noise(textRows * c.context, seed: 1))
        return try Pace.run(invocation, title: "\(model) pace\(compact ? " compact" : "") \(invocation.size)") {
            let d = try QwenImage21Render.Denoiser(map: map, sequence: sequence, text: text, references: nil)
            print("\(model) pace: " + d.described)
            return d
        } evaluate: { d, e, time in
            try d.velocity(Pace.noise(c.channels * h * w, seed: UInt64(e + 2)), time: time)
        }
    }

    static let dit = Stage(
        model: model, name: "dit",
        proves: "one evaluation of the DiT at σ = 1 over the joint sequence, boundary by boundary: the layout (segments, RoPE table), the two timesteps' embedding and the shared modulation, txt_in, img_in, the sequence, layers 0, 1 and 31, norm_out, and the velocity per latent channel (with --edit N, N reference pictures in the sequence; with --variant compact, the Compact's int8 codes)",
        options: ["--edit", "--map", "--variant"], variants: edits,
        oracle: "tools/oracle/qwen-image-2.1.py dit [--edit N]; witness; bf16 [--edit N]; with --variant compact (no --edit), dit and w8a8 with --variant compact",
        weights: ["DiT"],
        rule: "the layout bit for bit, the RoPE table within 1e-3; the rule against the fp64 witness where it exists, with diffusers in bf16 beside the oracle (the W8A8 run for a Compact); norm_out on the target's rows, the only ones the DiT computes it for"
    ) { invocation in
        let compact = try invocation.compact()
        let n = invocation.edit ?? 0
        if compact, n > 0 { throw UsageError("--variant compact judges the generation (no --edit): the oracle writes no editing golden for it") }
        let tag = (n > 0 ? "-edit\(n)" : "") + (compact ? "-compact" : "")
        let report = Report(title: "\(model) dit\(n > 0 ? " edit \(n)" : "")\(compact ? " compact" : "") \(invocation.size)", half: compact ? "W8A8" : "bf16")
        let path = try invocation.ditMap(ditMap, compactMap, compact: compact)
        let stem = "qwen21/goldens-qwen21-dit\(tag)-\(invocation.size)"
        let goldens = try invocation.goldens.open(stem + ".safetensors")
        let text = try invocation.goldens.open((n > 0 ? "qwen21/goldens-qwen21-text-image-\(n)" : "qwen21/goldens-qwen21-text") + ".safetensors")
        let half = compact
            ? try invocation.goldens.int8("qwen21/witness-w8a8-qwen21-dit-compact-\(invocation.size).safetensors", script: "tools/oracle/qwen-image-2.1.py w8a8 --variant compact")
            : try invocation.goldens.half("qwen21/witness-bf16-qwen21-dit\(tag)-\(invocation.size).safetensors",
                                          script: "tools/oracle/qwen-image-2.1.py bf16\(n > 0 ? " --edit \(n)" : "")")
        // The fp64 witness is the generation's, on the publisher's weights.
        let witness = n > 0 || compact ? nil : invocation.goldens.optional("qwen21/witness-qwen21-f64-\(invocation.size).safetensors")
        let info = try invocation.goldens.manifest(stem)
        let given = try Evaluation(goldens, info: info, text: text)
        let (slots, sequence, hidden, timestep, textRows) = (given.slots, given.sequence, given.hidden, given.timestep, given.textRows)
        let dit = QwenImage21Transformer(sequence: sequence)
        let c = dit.config, (p, s, rows) = (sequence.prefixRows, sequence.targetRows, sequence.rows)

        // The layout: pure functions of `img_mask` and `img_shapes`.
        let segments = (info["segments"] as? [[Any]] ?? []).map { "\($0[0] as? Int ?? -1)-\($0[1] as? Int ?? -1)\(($0[2] as? Bool) == true ? "t" : "i")" }
        let mine = sequence.segments.map { "\($0.start)-\($0.end)\($0.isText ? "t" : "i")" }
        report.add(Judge.holds("segments", segments == mine, note: mine.joined(separator: " ")))
        report.add(Judge.integers("target_token_mask", (0..<rows).map { $0 >= p ? 1 : 0 }, goldens.int32("target_token_mask")))
        let rope = dit.rope()
        var deviation: Float = .infinity
        if let cosines = goldens.float32("rope_cos"), let sines = goldens.float32("rope_sin"), cosines.count == rope.cos.count {
            deviation = zip(rope.cos, cosines).map { abs($0 - $1) }.max() ?? 0
            deviation = max(deviation, zip(rope.sin, sines).map { abs($0 - $1) }.max() ?? 0)
        }
        report.add(Judge.bound("RoPE table", Double(deviation), 1e-3, unit: "max |Δ|", note: "the oracle's angles are fp32"))

        let (latent, references) = (given.latent, given.references)

        // One evaluation, the stream read after each pass's embedding and its layers 0, 1 and 31.
        let d = c.dim, wanted = [-1, 0, 1, c.layers - 1]
        let picture = sequence.segments.flatMap { seg in (seg.start..<seg.end).map { _ in !seg.isText } }
        let run = try blocking { () -> UncheckedSendable<(prefix: [Int: [Float]], target: [Int: [Float]], velocity: [Float], normed: [Float],
                                                           temb: [Float], modulation: [Float])> in
            let loop = try await QwenImage21Loop.open(map: path, sequence: sequence, tapped: true)
            print("stage    \(loop.described)")
            var prefix: [Int: [Float]] = [:], target: [Int: [Float]] = [:]
            loop.observe = { pass, after in
                guard wanted.contains(after) else { return }
                if pass == .prefix { prefix[after] = await loop.streamRows(p) } else { target[after] = await loop.streamRows(s) }
            }
            try await loop.prompt(text: textRows, references: references)
            // The bf16 reference is told its timestep in bf16 (`timestep.to(hidden_states.dtype)`).
            let velocity = try await loop.evaluate(latent, time: BFloat.round(timestep[0]))
            let normed = await loop.floats("normed", count: s * d)
            let temb = await loop.floats("temb", count: d) + (await loop.floats("temb0", count: d))
            let modulation = await loop.floats("modulation", count: 4 * d) + (await loop.floats("modulation0", count: 4 * d))
            loop.observe = nil
            return UncheckedSendable((prefix, target, velocity, normed, temb, modulation))
        }.value

        report.add(goldens, "temb", run.temb, witness: witness?.float32("temb"), half: half.float32("temb").map(Array.init), columns: d)
        report.add(goldens, "modulation", run.modulation, witness: witness?.float32("modulation"), half: half.float32("modulation").map(Array.init), columns: 4 * d)
        // `txt_in` reads every row of the encoder's states in the reference; the picture slots' rows are then replaced
        // by `img_in`'s, so only the text rows are a boundary of the sequence.
        func textOnly(_ rows: UnsafeBufferPointer<Float>?) -> [Float]? {
            guard let rows, rows.count == slots.count * d else { return nil }
            return QwenImage21Render.spoken(HostTensor([slots.count, d], floats: Array(rows)), slots: slots).floats
        }
        func rowsOf(_ all: [Float], _ which: (Int) -> Bool) -> [Float] { (0..<p).filter(which).flatMap { all[($0 * d)..<(($0 + 1) * d)] } }
        if let embedded = run.prefix[-1], let oracle = textOnly(goldens.float32("txt_in_out")) {
            report.add(oracle.withUnsafeBufferPointer {
                Judge.boundary("txt_in_out", engine: rowsOf(embedded, { !picture[$0] }), oracle: $0, half: textOnly(half.float32("txt_in_out")),
                               halfName: report.half, columns: d)
            })
        }
        func joint(_ after: Int) -> [Float]? { run.prefix[after].flatMap { a in run.target[after].map { a + $0 } } }
        report.add(goldens, "img_in_out", run.prefix[-1].flatMap { a in run.target[-1].map { rowsOf(a, { picture[$0] }) + $0 } },
                   witness: witness?.float32("img_in_out"), half: half.float32("img_in_out").map(Array.init), columns: d)
        for (name, after) in [("joint_in", -1), ("layer0_out", 0), ("layer1_out", 1), ("layer_last_out", c.layers - 1)] {
            report.add(goldens, name, joint(after), witness: witness?.float32(name), half: half.float32(name).map(Array.init), columns: d)
        }
        // norm_out: the target's rows, the last S of the joint sequence (the DiT drops the prefix's before norm_out).
        func targetOnly(_ all: UnsafeBufferPointer<Float>?) -> [Float]? {
            guard let all, all.count == rows * d else { return nil }
            return Array(all[(p * d)...])
        }
        if let oracle = targetOnly(goldens.float32("norm_out")) {
            let w = targetOnly(witness?.float32("norm_out"))
            report.add(oracle.withUnsafeBufferPointer { o in
                let measure = { (w: UnsafeBufferPointer<Float>?) in
                    Judge.boundary("norm_out (target)", engine: run.normed, oracle: o, witness: w, half: targetOnly(half.float32("norm_out")),
                                   halfName: report.half, columns: d)
                }
                return w.map { w in w.withUnsafeBufferPointer { measure($0) } } ?? measure(nil)
            })
        }
        // The velocity, token-major as the oracle writes it: its 64 channels are the latent's planes.
        let velocity = run.velocity
        let tokens = velocity.count == s * c.channels ? QwenImage21Render.tokens(planes: velocity) : [Float](repeating: 0, count: velocity.count)
        report.add(goldens, "model_out", tokens, witness: witness?.float32("model_out"), half: half.float32("model_out").map(Array.init),
                   columns: c.channels)
        return report
    }

    /// **An evaluation's inputs as a DiT golden gives them**: the sequence from `img_mask` and `img_shapes`, the text
    /// rows of the encoder's states (`text`'s `prompt_embeds`), the references' tokens, the target's `[64, S]` planar.
    struct Evaluation {
        let slots: [Bool]
        let sequence: QwenImage21DiT.Sequence
        let hidden: [Float]
        let timestep: [Float]
        let textRows: [Float]
        let latent: [Float]
        let references: [Float]

        init(_ goldens: Safetensors, info: [String: Any], text: Safetensors) throws {
            guard let shapes = info["img_shapes"] as? [[Int]], let last = shapes.last, let mask = goldens.int32("img_mask"),
                  let hidden = goldens.float32("hidden_states"), let timestep = goldens.float32("timestep"),
                  let embeds = text.float32("prompt_embeds"), let embedsEntry = text.entries["prompt_embeds"] else {
                throw Skip("\(goldens.path): the evaluation's inputs are missing")
            }
            let c = QwenImage21DiT.Config.published
            // `img_mask` spans the encoder's rows and the target's appended slots: the latter are not the prefix's.
            slots = mask.prefix(embedsEntry.shape[0]).map { $0 != 0 }
            sequence = try QwenImage21DiT.Sequence(slots: slots, references: shapes.dropLast().map { (height: $0[1], width: $0[2]) },
                                                   height: last[1], width: last[2])
            (self.hidden, self.timestep) = (Array(hidden), Array(timestep))
            textRows = QwenImage21Render.spoken(HostTensor([slots.count, c.context], floats: Array(embeds.prefix(slots.count * c.context))), slots: slots).floats
            let (r, s) = (sequence.referenceRows, sequence.targetRows)
            guard hidden.count == (r + s) * c.channels else { throw Skip("\(goldens.path): hidden_states is not the references and the target") }
            latent = QwenImage21Render.planes(tokens: Array(hidden[(r * c.channels)...]))
            references = Array(hidden.prefix(r * c.channels))
        }
    }

    // MARK: - The DiT under a LoRA

    static let ditLoRA = Stage(
        model: model, name: "dit-lora",
        proves: "one evaluation of the DiT at σ = 1 under a user's LoRA read in its published file (the golden's: PEFT's names, 224 modules, the SwiGLU's input in diffusers' two halves, strength 1), the low-rank path beside each product: the velocity per latent channel (with --variant compact, on the Compact's codes; under --neural-engine on, in the shared forms)",
        options: ["--map", "--lora", "--variant"],
        oracle: "tools/oracle/qwen-image-2.1.py dit; bf16; w8a8 --variant compact (the references without the LoRA); under it, dit-lora --lora FILE (the reference under it, qwen21/goldens-qwen21-dit-lora-user-{size}, whose manifest names the LoRA's file, its source and its scale)",
        weights: ["DiT", "the LoRA"],
        rule: "the file recognized as this family's, every module applied; model_out per latent channel against the fp32 reference under the LoRA: no further than the bf16 reference (the W8A8 run for a Compact) without the LoRA is from the fp32 one, + 25 % — the LoRA alone moves model_out by a quarter on its worst channel"
    ) { invocation in
        let compact = try invocation.compact()
        let stem = "qwen21/goldens-qwen21-dit-lora-user-\(invocation.size)"
        let goldens = try invocation.goldens.open(stem + ".safetensors"), info = try invocation.goldens.manifest(stem)
        guard let named = info["user_lora"] as? String else { throw Skip("\(stem): no user_lora in its manifest") }
        guard let file = invocation.extras["--lora"] else { throw Skip("the golden's LoRA is not given: --lora <\(named)>") }
        guard (file as NSString).lastPathComponent == named else { throw UsageError("--lora \(file): the golden's is \(named)") }
        let report = Report(title: "\(model) dit-lora\(compact ? " compact" : "") \(invocation.size)", half: compact ? "W8A8" : "bf16")
        let lora = try LoRAFile(file)
        let unapplied = lora.unapplied(by: LoRAStack.modules(of: .qwenImage21))
        report.add(Judge.holds("modules applied", unapplied.isEmpty && lora.family() == .qwenImage21,
                               note: "\(lora.modules.count - unapplied.count) of \(lora.modules.count)" + (unapplied.first.map { ", not \($0)" } ?? "")))
        let stack = try LoRAStack([LoRAEntry(file)], on: .qwenImage21)

        // Without the LoRA: the fp32 reference, and the bf16 (W8A8) run beside it, on the same inputs.
        let base = try invocation.goldens.open("qwen21/goldens-qwen21-dit-\(invocation.size).safetensors")
        let half = compact
            ? try invocation.goldens.int8("qwen21/witness-w8a8-qwen21-dit-compact-\(invocation.size).safetensors", script: "tools/oracle/qwen-image-2.1.py w8a8 --variant compact")
            : try invocation.goldens.half("qwen21/witness-bf16-qwen21-dit-\(invocation.size).safetensors", script: "tools/oracle/qwen-image-2.1.py bf16")
        guard let underLoRA = goldens.float32("model_out"), let alone = base.float32("model_out"), let halfAlone = half.float32("model_out"),
              let h0 = goldens.float32("hidden_states"), let h1 = base.float32("hidden_states"), Array(h0) == Array(h1) else {
            throw Skip("\(stem): model_out, or the inputs of the dit stage's goldens, missing")
        }
        let given = try Evaluation(goldens, info: info, text: try invocation.goldens.open("qwen21/goldens-qwen21-text.safetensors"))
        let path = try invocation.ditMap(ditMap, compactMap, compact: compact)
        let c = QwenImage21DiT.Config.published
        let velocity = try blocking { () -> [Float] in
            let loop = try await QwenImage21Loop.open(map: path, sequence: given.sequence, loras: stack)
            print("stage    \(loop.described)")
            try await loop.prompt(text: given.textRows, references: given.references)
            return try await loop.evaluate(given.latent, time: BFloat.round(given.timestep[0]))
        }
        let tokens = QwenImage21Render.tokens(planes: velocity)
        let reference = channelError(Array(halfAlone), alone, columns: c.channels).worst
        let moved = channelError(Array(alone), underLoRA, columns: c.channels).worst
        let ours = channelError(tokens, underLoRA, columns: c.channels)
        let threshold = max(Judge.floor, 1.25 * reference)
        report.add(Measure(boundary: "model_out", worst: ours.worst, median: ours.median, channel: ours.channel, atWitness: nil,
                           threshold: threshold, passed: ours.worst <= threshold,
                           note: String(format: "\(report.half) without the LoRA %.2e; the LoRA alone %.2e", reference, moved)))
        return report
    }
}
