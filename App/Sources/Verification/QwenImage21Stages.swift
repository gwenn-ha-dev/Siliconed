import Foundation
import Siliconed

// **Qwen-Image-2.1 Turbo, stage by stage** — `siliconed check qwen-image-2.1 <stage>`.
//
//     text → dit → trajectory → vae → render                                    a generation
//     references → text --edit N → vae-encode → dit --edit N → render --edit N   an edit, N reference pictures
//
// What sets it apart: one joint sequence under a block-causal attention — the text causal, each picture a block, the
// target last and seeing everything —, the conditions modulated by t = 0, a modulation shared by the 32 layers, and a
// schedule published as eight σ. `dit` judges one evaluation of the whole sequence, as the reference's without a K/V
// cache (the oracle found its cached step equal to it, bit for bit); `trajectory` judges the render's session, whose
// steps after the first read the conditions' kept keys and values. Only the target's rows of the output are a
// velocity, and only they have a bf16 reference (`model_out`).
//
// Every stage but the tokenizer's identifiers runs on the engine, in the arithmetic Siliconed declares. So each is judged by the rule with the reference's own bf16 run beside the f32 oracle
// (`half`); a stage without its bf16 reference is skipped, with the script that writes it.

enum QwenImage21Stages {
    static let all: [Stage] = [references, vision, text, dit, ditLoRA, pace, trajectory, encoding, vae, render]

    static let model = "qwen-image-2.1"
    static let ditMap = Family.qwenImage21.ditMap(.standard)
    static let compactMap = Family.qwenImage21.ditMap(.compact)
    static let encoderMap = Family.qwenImage21.encoderMap(.standard)
    static let tokenizer = "composants/qwen-image-2.1/tokenizer"
    static let vaeMap = "composants/qwen-image-2.1/vae.safetensors"
    /// `--edit 1`, `--edit 2`: the goldens the oracle's `all` writes beside the generation's.
    static let edits: [[String]] = [[], ["--edit", "1"], ["--edit", "2"]]
    /// The oracle's `output_resolution`: its references' side, where the product's is 1024.
    static let resolution = 512

    /// A test picture of the oracle's, read as the app reads a reference.
    static func picture(_ name: String, _ invocation: Invocation) throws -> ImageRGB {
        try ImageRGB(contentsOf: try invocation.goldens.input(name))
    }

    /// `[C, h·w]` planes to `[h·w, C]` tokens: a plane is then a column, which `Judge.boundary` judges one by one.
    static func columns(_ planes: [Float], _ channels: Int) -> [Float] {
        let plane = planes.count / channels
        var out = [Float](repeating: 0, count: planes.count)
        for t in 0..<plane { for c in 0..<channels { out[t * channels + c] = planes[c * plane + t] } }
        return out
    }

    // MARK: - The references

    static let references = Stage(
        model: model, name: "references",
        proves: "a reference picture as both encoders read it, from its file: the size, Pillow's Lanczos, the RGBA the VAE reads and the patches the vision tower reads — an opaque picture (the oracle's two) and one with transparency, at two sizes",
        oracle: "tools/oracle/qwen-image-2.1.py references; references --side 443; text --edit 2; vae-encode", weights: [],
        rule: "bit for bit"
    ) { invocation in
        let report = Report(title: "\(model) references")
        func judge(_ label: String, _ prepared: ImageRGB, image: UnsafeBufferPointer<Float>?, patches: UnsafeBufferPointer<Float>?) throws {
            report.add(Judge.identical("\(label) RGBA", QwenImage21References.rgba(prepared), image))
            // The values live in the `Pixels`' own pages: it is kept until they are copied.
            let pixels = try Qwen3VLImages.pixels(QwenImage21References.rgb8(prepared))
            report.add(Judge.identical("\(label) patches", withExtendedLifetime(pixels) { Array(pixels.values) }, patches))
        }
        // The two opaque pictures: the VAE's copy in `vae-encode`'s golden, the tower's in `text --edit 2`'s.
        let encoded = try invocation.goldens.open("qwen21/goldens-qwen21-vae-encode-\(resolution).safetensors")
        let read = try invocation.goldens.open("qwen21/goldens-qwen21-text-image-2.safetensors")
        guard let names = try invocation.goldens.manifest("qwen21/goldens-qwen21-text-image-2")["images"] as? [String],
              let all = read.float32("pixel_values") else { throw Skip("\(read.path): images or pixel_values missing") }
        var at = 0
        for (i, name) in names.enumerated() {
            let prepared = QwenImage21References.prepared(try picture(name, invocation), resolution: resolution)
            let count = prepared.height / 16 * (prepared.width / 16) * Qwen3VLImages.Pixels.patchValues
            guard at + count <= all.count else { throw Skip("\(read.path): pixel_values is short") }
            try judge("picture \(i)", prepared, image: encoded.float32("image_\(i)"), patches: UnsafeBufferPointer(rebasing: all[at..<(at + count)]))
            at += count
        }
        // The picture with transparency, at two sizes.
        for side in [512, 443] {
            let stem = "qwen21/goldens-qwen21-preprocess-alpha-\(side)"
            let golden = try invocation.goldens.open(stem + ".safetensors")
            guard let name = try invocation.goldens.manifest(stem)["image"] as? String else { throw Skip("\(golden.path): image missing") }
            let prepared = QwenImage21References.prepared(try picture(name, invocation), resolution: side)
            try judge("alpha \(side)", prepared, image: golden.float32("vae_image"), patches: golden.float32("pixel_values"))
        }
        return report
    }

    // MARK: - The text

    static let text = Stage(
        model: model, name: "text",
        proves: "Qwen3-VL-8B reads the prompt: template and identifiers, then layers 0, 17 and 35 over the real tokens and prompt_embeds, the last layer's output without the system turn (with --edit N, N reference pictures read with it: their slots and three-axis positions, the vision tower's rows at the slots — layer 0's input —, its deepstack rows added after layers 0, 1 and 2)",
        options: ["--edit"], variants: edits,
        oracle: "tools/oracle/qwen-image-2.1.py text [--edit N]; text-bf16 [--edit N]", weights: ["text encoder"],
        rule: "identifiers and positions bit for bit, the RoPE table within 1e-3; the rule against transformers in bf16"
    ) { invocation in
        let n = invocation.edit ?? 0
        let tag = n > 0 ? "-image-\(n)" : "", edit = n > 0 ? " --edit \(n)" : ""
        let report = Report(title: "\(model) text\(n > 0 ? " edit \(n)" : "")")
        let goldens = try invocation.goldens.open("qwen21/goldens-qwen21-text\(tag).safetensors")
        let half = try invocation.goldens.half("qwen21/witness-bf16-qwen21-text\(tag).safetensors", script: "tools/oracle/qwen-image-2.1.py text-bf16\(edit)")
        let info = try invocation.goldens.manifest("qwen21/goldens-qwen21-text\(tag)")
        guard let prompt = info["prompt"] as? String, let drop = info["drop_idx"] as? Int, let expected = goldens.int32("ids") else {
            throw Skip("\(goldens.path): prompt, drop_idx or ids missing")
        }
        // The pictures' grids, in patches, and the patches themselves: the golden's, one thing judged at a time
        // (`references` judges the app's own).
        var pictures: [Qwen3VLImages.Pixels] = []
        if n > 0 {
            guard let grids = goldens.int32("image_grid_thw"), let values = goldens.float32("pixel_values"), grids.count == 3 * n else {
                throw Skip("\(goldens.path): image_grid_thw or pixel_values missing")
            }
            var at = 0
            for i in 0..<n {
                let (gh, gw) = (Int(grids[3 * i + 1]), Int(grids[3 * i + 2])), count = gh * gw * Qwen3VLImages.Pixels.patchValues
                pictures.append(Qwen3VLImages.Pixels(values: Array(values[at..<(at + count)]), gridHeight: gh, gridWidth: gw))
                at += count
            }
        }
        let tokenizer = try Tokenizer(directory: Library.processWide.map(Self.tokenizer))
        let (ids, _, dropped, _) = try QwenImage21Text.identifiers(prompt, pictures: pictures.map { $0.patches / 4 }, tokenizer: tokenizer)
        report.add(Judge.integers("tokens", ids, expected))
        report.add(Judge.holds("system turn", dropped == drop, note: "\(dropped) tokens dropped"))
        // The computation starts from the golden's identifiers: one thing judged at a time.
        let golden = expected.map(Int.init), pad = try Qwen3VLChat.padIdentifier(tokenizer)
        let slots = golden.map { $0 == pad }
        let config = Qwen3Encoder.Config.qwenImage21
        if n > 0 {
            let positions = try Qwen3VLImages.rotaryPositions(slots: slots, grids: pictures.map { (height: $0.gridHeight, width: $0.gridWidth) })
            report.add(Judge.integers("positions", positions.map(\.0) + positions.map(\.1) + positions.map(\.2), goldens.int32("position_ids")))
            let table = Qwen3VLImages.rotaryTable(positions, headDim: config.headDim, theta: config.theta)
            var deviation: Float = .infinity
            if let cosines = goldens.float32("rope_cos"), let sines = goldens.float32("rope_sin"), cosines.count == table.cos.count {
                deviation = max(zip(table.cos, cosines).map { abs($0 - $1) }.max() ?? 0, zip(table.sin, sines).map { abs($0 - $1) }.max() ?? 0)
            }
            report.add(Judge.bound("RoPE table", Double(deviation), 1e-3, unit: "max |Δ|", note: "three axes"))
        }
        let path = try MapCheck.use(Library.processWide.map(encoderMap))
        let width = config.width, layers = n > 0 ? [0, 1, 2, 17, 35] : [0, 17, 35], entries = n > 0 ? [0, 1, 2, 3] : []
        let states = try QwenImage21Render.states(golden, pictures: pictures, slots: slots, map: path, taps: Set(layers), entries: Set(entries))
        for i in entries {
            report.add(goldens, "layer\(i)_in", states["layer\(i)_in"]?.floats, half: half.float32("layer\(i)_in").map(Array.init), columns: width)
        }
        for i in layers {
            report.add(goldens, "layer\(i)_out", states["layer\(i)"]?.floats, half: half.float32("layer\(i)_out").map(Array.init), columns: width)
        }
        report.add(goldens, "prompt_embeds", states["hidden"].map { Array($0.floats[(drop * width)...]) },
                   half: half.float32("prompt_embeds").map(Array.init), columns: width)
        return report
    }

    // MARK: - The vision tower

    static let vision = Stage(
        model: model, name: "vision",
        proves: "Qwen3-VL-8B's vision tower on the oracle's two reference pictures, each alone: the first block (patch embedding, resampled positions, axial RoPE), the last hidden state, the merged rows the language model reads and the three deepstack rows",
        oracle: "tools/oracle/qwen-image-2.1.py vision; vision-bf16", weights: ["text encoder"],
        rule: "the rule against transformers in bf16"
    ) { invocation in try Qwen3VLVisionCheck.judged(invocation, model: model, map: encoderMap) }

    // MARK: - The DiT

    static let pace = Stage(
        model: model, name: "pace", proves: Pace.proves + " (with --variant compact, the Compact's map in the library whatever is installed)",
        sizes: ["1024", "512"], options: ["--renders", "--evaluations", "--variant"],
        oracle: nil, weights: ["DiT"], rule: Pace.rule
    ) { invocation in
        let installed = try Model.named(model, in: Library.processWide)
        guard let denoising = installed.chain.denoising as? QwenImage21DenoisingModule else { throw Skip("\(model): no DiT installed") }
        let map = try invocation.compact() ? try invocation.ditMap(compactMap, golden: .eightBit) : denoising.map
        let (h, w) = (invocation.height / denoising.space.factor, invocation.width / denoising.space.factor)
        let c = QwenImage21DiT.Config.published, textRows = 48
        let sequence = try QwenImage21DiT.Sequence(slots: [Bool](repeating: false, count: textRows), references: [], height: h, width: w)
        let text = HostTensor([textRows, c.context], floats: Pace.noise(textRows * c.context, seed: 1))
        return try Pace.run(invocation, title: "\(model) pace \(invocation.size)") {
            try QwenImage21Render.Denoiser(map: map, sequence: sequence, text: text, references: nil)
        } evaluate: { d, e, time in
            try (d as! QwenImage21Render.Denoiser).velocity(Pace.noise(c.channels * h * w, seed: UInt64(e + 2)), time: time)
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
        let path = try invocation.ditMap(compact ? compactMap : ditMap, golden: compact ? .eightBit : .standard)
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
        let spoken = slots.indices.filter { !slots[$0] }
        func textOnly(_ rows: UnsafeBufferPointer<Float>?) -> [Float]? {
            guard let rows, rows.count == slots.count * d else { return nil }
            return spoken.flatMap { rows[($0 * d)..<(($0 + 1) * d)] }
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
        var tokens = [Float](repeating: 0, count: velocity.count)
        if velocity.count == s * c.channels {
            for t in 0..<s { for ch in 0..<c.channels { tokens[t * c.channels + ch] = velocity[ch * s + t] } }
        }
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
            textRows = slots.enumerated().filter { !$0.element }.flatMap { i, _ in embeds[(i * c.context)..<((i + 1) * c.context)] }
            let (r, s) = (sequence.referenceRows, sequence.targetRows)
            guard hidden.count == (r + s) * c.channels else { throw Skip("\(goldens.path): hidden_states is not the references and the target") }
            var latent = [Float](repeating: 0, count: s * c.channels)
            for t in 0..<s { for ch in 0..<c.channels { latent[ch * s + t] = hidden[(r + t) * c.channels + ch] } }
            self.latent = latent
            references = Array(hidden.prefix(r * c.channels))
        }
    }

    // MARK: - The DiT under a LoRA

    static let ditLoRA = Stage(
        model: model, name: "dit-lora",
        proves: "one evaluation of the DiT at σ = 1 under a user's LoRA read in its published file (the golden's: PEFT's names, 224 modules, the SwiGLU's input in diffusers' two halves, strength 1), the low-rank path beside each product: the velocity per latent channel (with --variant compact, on the Compact's codes; under the precision ane, in the shared forms)",
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
        let path = try invocation.ditMap(compact ? compactMap : ditMap, golden: compact ? .eightBit : .standard)
        let c = QwenImage21DiT.Config.published, s = given.sequence.targetRows
        let velocity = try blocking { () -> [Float] in
            let loop = try await QwenImage21Loop.open(map: path, sequence: given.sequence, loras: stack)
            print("stage    \(loop.described)")
            try await loop.prompt(text: given.textRows, references: given.references)
            return try await loop.evaluate(given.latent, time: BFloat.round(given.timestep[0]))
        }
        var tokens = [Float](repeating: 0, count: velocity.count)
        for t in 0..<s { for ch in 0..<c.channels { tokens[t * c.channels + ch] = velocity[ch * s + t] } }
        let reference = channelError(Array(halfAlone), alone, columns: c.channels).worst
        let moved = channelError(Array(alone), underLoRA, columns: c.channels).worst
        let ours = channelError(tokens, underLoRA, columns: c.channels)
        let threshold = max(Judge.floor, 1.25 * reference)
        report.add(Measure(boundary: "model_out", worst: ours.worst, median: ours.median, channel: ours.channel, atWitness: nil,
                           threshold: threshold, passed: ours.worst <= threshold,
                           note: String(format: "\(report.half) without the LoRA %.2e; the LoRA alone %.2e", reference, moved)))
        return report
    }

    // MARK: - The VAE's encoder

    static let encoding = Stage(
        model: model, name: "vae-encode",
        proves: "the VAE's encoder in bf16 on the oracle's two reference pictures, RGBA: the latent's mean, then the latent normalized in bf16 as the pipeline hands it to the DiT",
        oracle: "tools/oracle/qwen-image-2.1.py vae-encode; vae-encode-bf16", weights: ["VAE"],
        rule: "per latent plane: the rule against diffusers encoding in bf16"
    ) { invocation in
        let report = Report(title: "\(model) vae-encode \(resolution)")
        let goldens = try invocation.goldens.open("qwen21/goldens-qwen21-vae-encode-\(resolution).safetensors")
        let half = try invocation.goldens.half("qwen21/witness-bf16-qwen21-vae-encode-\(resolution).safetensors",
                                               script: "tools/oracle/qwen-image-2.1.py vae-encode-bf16")
        let path = Library.processWide.map(vaeMap)
        let channels = QwenImage21VAEDecoder.latentChannels
        for i in 0..<2 {
            guard let entry = goldens.entries["image_\(i)"], let image = goldens.float32("image_\(i)") else {
                throw Skip("\(goldens.path): image_\(i) missing")
            }
            let mean = try QwenImage21VAEEncoder.mean(of: Array(image), height: entry.shape[1], width: entry.shape[2], path: path)
            for (name, values) in [("latent_mean_raw_\(i)", mean), ("latent_\(i)", QwenImage21VAEEncoder.normalized(mean))] {
                guard let oracle = goldens.float32(name) else { report.add(Judge.missing(name)); continue }
                let expected = columns(Array(oracle), channels)
                report.add(expected.withUnsafeBufferPointer {
                    Judge.boundary(name, engine: columns(values, channels), oracle: $0,
                                   half: half.float32(name).map { columns(Array($0), channels) }, columns: channels)
                })
            }
        }
        return report
    }

    // MARK: - The decoder

    static let vae = Stage(
        model: model, name: "vae",
        proves: "the decoder in bf16 on the final latent of the oracle's trajectory, de-normalized in bf16 as the pipeline does: the RGBA image plane by plane",
        oracle: "tools/oracle/qwen-image-2.1.py vae; vae-witness; vae-bf16", weights: ["VAE"],
        rule: "per plane, at the fp64 witness: no further from it than diffusers decoding in bf16, + 25 %"
    ) { invocation in
        let report = Report(title: "\(model) vae \(invocation.size)")
        let goldens = try invocation.goldens.open("qwen21/goldens-qwen21-vae-\(invocation.size).safetensors")
        let witness = invocation.goldens.optional("qwen21/witness-qwen21-vae-f64-\(invocation.size).safetensors")
        let half = try invocation.goldens.half("qwen21/witness-bf16-qwen21-vae-\(invocation.size).safetensors",
                                               script: "tools/oracle/qwen-image-2.1.py vae-bf16")
        guard let entry = goldens.entries["vae_in"], let tokens = goldens.float32("latent_final"),
              let expected = goldens.float32("image") else {
            throw Skip("\(goldens.path): latent_final or image missing")
        }
        let (channels, h, w) = (entry.shape[0], entry.shape[1], entry.shape[2])
        // `_unpack_latents`: `[h·w, 64]` to planes, a plain transposition.
        var latent = [Float](repeating: 0, count: tokens.count)
        for t in 0..<(h * w) { for c in 0..<channels { latent[c * h * w + t] = tokens[t * channels + c] } }
        let given = QwenImage21VAEDecoder.denormalized(latent)
        let pixels = try DecoderSession<QwenImage21VAEDecoder>(path: Library.processWide.map(vaeMap), latentHeight: h, latentWidth: w).pixels(given)
        report.add(Judge.planes("image RGBA", engine: pixels, oracle: expected, witness: witness?.float32("image"),
                                reference: half.float32("image").map(Array.init), planes: 4))
        return report
    }

    // MARK: - The sampler

    static let trajectory = Stage(
        model: model, name: "trajectory",
        proves: "the seed's noise bit for bit, the published 8-step σ schedule, then each step's velocity from the reference's own latent at that step, by the render's DiT session: the first step the whole sequence, the others the target over the kept keys and values (with --edit N, N reference pictures in the sequence)",
        options: ["--edit", "--map"], variants: edits,
        oracle: "tools/oracle/qwen-image-2.1.py trajectory [--edit N]; trajectory-bf16; with --edit N, trajectory-steps-bf16 --edit N",
        weights: ["DiT"],
        rule: "the noise and σ exact; each step by the rule against diffusers' same step in bf16"
    ) { invocation in
        let n = invocation.edit ?? 0
        // An edit's size is the oracle's (its last picture's ratio), not `--size`.
        let tag = n > 0 ? "-edit\(n)-\(resolution)" : "-\(invocation.size)", edit = n > 0 ? " --edit \(n)" : ""
        let report = Report(title: "\(model) trajectory\(n > 0 ? " edit \(n) \(resolution)" : " \(invocation.size)")")
        let path = try invocation.ditMap(ditMap)
        let stem = "qwen21/goldens-qwen21-trajectory\(tag)"
        let reference = try invocation.goldens.open(stem + ".safetensors")
        let info = try invocation.goldens.manifest(stem)
        let layout = try invocation.goldens.open("qwen21/goldens-qwen21-dit\(tag).safetensors")
        let shapes = (try invocation.goldens.manifest("qwen21/goldens-qwen21-dit\(tag)"))["img_shapes"] as? [[Int]]
        let text = try invocation.goldens.open((n > 0 ? "qwen21/goldens-qwen21-text-image-\(n)" : "qwen21/goldens-qwen21-text") + ".safetensors")
        // A generation's bf16 steps are written beside its free run; an edit's by a stage of their own.
        let half = n > 0
            ? try invocation.goldens.half("qwen21/witness-bf16-qwen21-trajectory-steps\(tag).safetensors",
                                          script: "tools/oracle/qwen-image-2.1.py trajectory-steps-bf16\(edit)")
            : try invocation.goldens.half("qwen21/witness-bf16-qwen21-trajectory\(tag).safetensors", script: "tools/oracle/qwen-image-2.1.py trajectory-bf16")
        guard let tokens = reference.float32("latent_init"), let sigmas = reference.float32("sigmas"),
              let seed = info["seed"] as? NSNumber, let height = info["height"] as? Int, let width = info["width"] as? Int,
              let shapes, let mask = layout.int32("img_mask"),
              let embeds = text.float32("prompt_embeds"), let embedsEntry = text.entries["prompt_embeds"] else {
            throw Skip("\(reference.path): the trajectory's inputs are missing")
        }
        let space = LatentSpace.qwenImage21, (h, w) = (height / space.factor, width / space.factor), plane = h * w
        // Drawn `(1, 1, 64, h, w)`, recorded packed `[h·w, 64]`.
        func planar(_ packed: UnsafeBufferPointer<Float>) -> [Float] {
            var out = [Float](repeating: 0, count: packed.count)
            for t in 0..<plane { for c in 0..<space.channels { out[c * plane + t] = packed[t * space.channels + c] } }
            return out
        }
        report.add(planar(tokens).withUnsafeBufferPointer {
            Judge.identical("noise (seed \(seed))", Latent.noise(space, height: h, width: w, seed: seed.uint64Value).values, $0)
        })
        report.add(Judge.schedule(QwenImage21DiT.sigmas, sigmas))
        guard tokens.count == plane * space.channels else { throw Skip("\(reference.path): latent_init is not the target's \(h)×\(w)") }

        // Then each step from the reference's own latent, as the render runs them: one session, the first step filling
        // the conditions' keys and values, the others reading them.
        let slots = mask.prefix(embedsEntry.shape[0]).map { $0 != 0 }
        let sequence = try QwenImage21DiT.Sequence(slots: slots, references: shapes.dropLast().map { (height: $0[1], width: $0[2]) }, height: h, width: w)
        let context = embedsEntry.shape[1]
        let spoken = slots.enumerated().filter { !$0.element }.flatMap { i, _ in embeds[(i * context)..<((i + 1) * context)] }
        var references: HostTensor?
        if n > 0 {
            guard let latents = reference.float32("reference_latents"), latents.count == sequence.referenceRows * space.channels else {
                throw Skip("\(reference.path): reference_latents is not the sequence's references")
            }
            references = HostTensor([sequence.referenceRows, space.channels], floats: Array(latents))
        }
        let denoiser = try QwenImage21Render.Denoiser(map: path, sequence: sequence, text: HostTensor([sequence.textRows, context], floats: spoken),
                                                      references: references)
        for step in 0..<(sigmas.count - 1) {
            let name = "velocity_\(step)"
            guard let latent = reference.float32(step == 0 ? "latent_init" : "latent_\(step)") else { continue }
            let v = try denoiser.velocity(planar(latent), time: QwenImage21Render.modelTime(sigma: QwenImage21DiT.sigmas[step], solver: .euler, first: step == 0))
            // Token-major, as the oracle writes it: its 64 channels are the latent's planes.
            report.add(reference, name, columns(v, space.channels), half: half.float32(name).map(Array.init), columns: space.channels)
        }
        return report
    }

    // MARK: - The whole render

    static let render = Stage(
        model: model, name: "render",
        proves: "the app's render (Engine.render: tokenizer, encoder, 8 Euler steps, decoder) from the reference prompt and seed: the reference's final picture (with --edit N, an edit of the oracle's N reference pictures, read from their files: the vision tower and the VAE's encoder too)",
        options: ["--variant", "--png", "--edit"], variants: edits,
        oracle: "tools/oracle/qwen-image-2.1.py trajectory [--edit N]; vae [--edit N]; trajectory-bf16 [--edit N]", weights: ["text encoder", "DiT", "VAE"],
        rule: "the same picture: PSNR against diffusers in fp32 no more than 3 dB under diffusers' own bf16 render's (a free 8-step loop in bf16 is as far from the fp32 picture as it is); with --variant compact, a finite picture, its PSNR said and not judged: the Compact is judged at the DiT"
    ) { invocation in
        let compact = try invocation.compact()
        let n = invocation.edit ?? 0
        // An edit's size is the oracle's (its last picture's ratio), not `--size`.
        let tag = n > 0 ? "-edit\(n)-\(resolution)" : "-\(invocation.size)", edit = n > 0 ? " --edit \(n)" : ""
        let report = Report(title: "\(model) render\(n > 0 ? " edit \(n)" : "")\(compact ? " compact" : "") \(n > 0 ? "\(resolution)" : invocation.size)")
        let image = try invocation.goldens.open("qwen21/goldens-qwen21-vae\(tag).safetensors")
        // The bf16 free run is the Standard's bar; a Compact's render has none.
        let half = compact ? nil : try invocation.goldens.half("qwen21/witness-bf16-qwen21-trajectory\(tag).safetensors",
                                                               script: "tools/oracle/qwen-image-2.1.py trajectory-bf16\(edit)")
        let text = try invocation.goldens.manifest(n > 0 ? "qwen21/goldens-qwen21-text-image-\(n)" : "qwen21/goldens-qwen21-text")
        let size = try invocation.goldens.manifest("qwen21/goldens-qwen21-vae\(tag)")
        guard let prompt = text["prompt"] as? String,
              let seed = (try invocation.goldens.manifest("qwen21/goldens-qwen21-trajectory\(tag)"))["seed"] as? NSNumber,
              let rgba = image.float32("image") else {
            throw Skip("\(image.path): the prompt, seed or image is missing")
        }
        let (width, height) = n > 0 ? (size["width"] as? Int ?? 0, size["height"] as? Int ?? 0) : (invocation.width, invocation.height)
        // The reference decodes RGBA; the picture is its RGB.
        let expected = Array(rgba.prefix(rgba.count / 4 * 3))
        let path = try invocation.ditMap(compact ? compactMap : ditMap, golden: compact ? .eightBit : .standard)
        // The encoder of the version judged, not the library's choice.
        let encoder = Library.processWide.map(Family.qwenImage21.encoderMap(compact ? .compact : .standard))
        let model = try Model.qwenImage21(in: Library.processWide, card: .qwenImage21, dit: path, encoder: encoder, resolution: resolution)
        var request = Request(prompt, width: width, height: height, seed: seed.uint64Value, steps: 8)
        request.references = try (text["images"] as? [String] ?? []).map { try picture($0, invocation) }
        guard request.references.count == n else { throw Skip("the text golden names \(request.references.count) pictures for --edit \(n)") }
        let started = Date()
        // The oracle's edit is under the app's floor (624×416 for 512 px a side at least): the family's render is called
        // as the engine calls it, past the format's preflight alone.
        let render = n > 0
            ? try Family.qwenImage21.traits.render(request, Admission.judge(request, chain: model.chain), [request.seed], model, nil) { _ in }[0]
            : try Engine().render(request, model: model)
        print(String(format: "rendered in %.1f s (text %.2f, references %.2f, denoising %.2f, decoding %.2f)", Date().timeIntervalSince(started),
                     render.timings.text, render.timings.encoding, render.timings.denoising, render.timings.decoding))
        if let png = invocation.extras["--png"] { try render.png().write(to: URL(fileURLWithPath: png)) }
        guard let wide = half?.float32("image") else {
            report.add(expected.withUnsafeBufferPointer { Judge.compactRender(render.image.pixels, $0) })
            return report
        }
        let reference = Array(wide.prefix(wide.count / 4 * 3))
        let (psnr, _, _, notFinite) = expected.withUnsafeBufferPointer { imageError(render.image.pixels, $0) }
        let (bar, _, _, _) = expected.withUnsafeBufferPointer { imageError(reference, $0) }
        report.add(Judge.bound("image PSNR", psnr, bar - 3, unit: "dB", above: true,
                               note: String(format: "diffusers in bf16 %.2f dB, − 3", bar) + (notFinite > 0 ? ", \(notFinite) non-finite values" : "")))
        let (against, _, _, _) = reference.withUnsafeBufferPointer { imageError(render.image.pixels, $0) }
        print(String(format: "  against diffusers' bf16 render: %.2f dB", against))
        return report
    }
}
