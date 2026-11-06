import Foundation
import Siliconed

// **Qwen-Image-2.1 Turbo, stage by stage** — `siliconed check qwen-image-2.1 <stage>`.
//
//     text [--edit N] → vision → references → dit [--edit N] → trajectory [--edit N] → vae [--edit N]
//                                                                                              vae-encode
//
// `--edit N` (1 or 2) is editing with N reference images, the test pictures in `<goldens>/inputs/`. The
// goldens are at 512 only. What sets this model apart, and what each stage watches: the encoder reads
// the prompt AND the images (Pillow's Lanczos to the bit, a vision tower, deepstack, 3D positions); the
// DiT runs as a prefill of the conditions, then each step from their K/V cache — the full pass under
// the block-causal mask judges both; the VAE decodes RGBA.

enum QwenImageStages {
    static let all: [Stage] = [text, vision, references, dit, trajectory, vae, vaeEncode] + QwenImageWriterStages.all

    static let model = "qwen-image-2.1"
    /// Generation, then editing with one and two pictures: what the oracle's `all` writes for a stage.
    static let edits: [[String]] = [[], ["--edit", "1"], ["--edit", "2"]]
    /// The test pictures, by number of references.
    static let pictures = [1: ["surfer-raincoat-cloudy-beach.png"], 2: ["surfer-raincoat-cloudy-beach.png", "border-collie-beach.png"]]

    static func encoder() throws -> (Artifact, Tokenizer) {
        let library = Library.processWide
        return (try Artifact(path: MapCheck.use(library.map(library.encoderMap(.qwenImage21)))),
                try Tokenizer(directory: try library.component(.qwenImage21, "processor")))
    }

    // MARK: - The encoder

    static let text = Stage(
        model: model, name: "text",
        proves: "Qwen3-VL-8B reads the prompt (and with --edit N, the N pictures; with --long, a prompt past 512 tokens): template, identifiers, positions, 36 layers, prompt_embeds",
        options: ["--edit", "--long"], variants: edits, oracle: "tools/oracle/qwen-image-2.1.py text [--edit N]; text --long", weights: ["text encoder"],
        rule: "identifiers and masks bit for bit; the rule, floor 5e-4 (with images, the deep layers judged on their text rows)"
    ) { invocation in
        if let n = invocation.edit {
            guard !invocation.long else { throw UsageError("--long is generation's: no --edit") }
            return try textWithImages(n, invocation)
        }
        let stem = invocation.long ? "qwen21/goldens-qwen21-text-long" : "qwen21/goldens-qwen21-text"
        let report = Report(title: "\(model) text\(invocation.long ? " --long" : "")")
        let goldens = try invocation.goldens.open(stem + ".safetensors")
        let info = try invocation.goldens.manifest(stem)
        guard let prompt = info["prompt"] as? String, let drop = info["drop_idx"] as? Int else {
            throw Skip("\(stem).json: prompt or drop_idx missing")
        }
        let (artifact, tokenizer) = try encoder()
        let encoder = try Qwen3VLEncoder(artifact: artifact, tokenizer: tokenizer)
        let prepared = try Qwen3VLPrompt.prepare(prompt: prompt, grids: [], tokenizer: tokenizer, tokens: encoder.tokens)
        report.add(Judge.integers("tokens", prepared.ids, goldens.int32("ids")))
        report.add(Judge.holds("_drop_idx", prepared.dropCount == drop, note: "\(prepared.dropCount), the oracle's \(drop)"))
        guard report.measures.allSatisfy(\.passed) else { return report }

        // Text alone: the three mRoPE axes are the index, the 1D RoPE.
        let theta = Float(info["rope_theta"] as? Double ?? 5e6)
        let inverse = TextEncoder.inverseFrequencies(theta: theta, dim: 128)
        let positions = prepared.positions
        report.add(Judge.holds("mRoPE, text alone", positions[0] == positions[1] && positions[1] == positions[2]
                               && positions[0] == Array(0..<prepared.ids.count), note: "t = h = w = index"))
        let sections = info["mrope_section"] as? [Int] ?? [24, 20, 20]
        var cosines = [Float](), sines = [Float]()
        for s in 0..<prepared.ids.count {
            var c = [Float](repeating: 0, count: 128), n = [Float](repeating: 0, count: 128)
            for j in 0..<64 {
                let angle = Float(positions[TextEncoder.mropeAxis(frequency: j, sections: sections)][s]) * inverse[j]
                c[j] = cosf(angle); c[j + 64] = c[j]; n[j] = sinf(angle); n[j + 64] = n[j]
            }
            cosines += c; sines += n
        }
        report.add(goldens, "rope_cos", cosines, columns: 128)
        report.add(goldens, "rope_sin", sines, columns: 128)

        let probes = ["layer0_in", "layer0_out", "layer17_out", "layer35_out"]
        encoder.recordedNames = Set(probes)
        let output = try encoder.encode(prompt: prompt)
        for name in probes { report.add(goldens, name, encoder.boundaries[name], columns: output.hidden) }
        report.add(goldens, "prompt_embeds", output.embeddings, columns: output.hidden)
        report.add(Judge.integers("image_pad_mask", output.imagePadMask.map { $0 ? 1 : 0 }, goldens.int32("image_pad_mask")))
        return report
    }

    /// The prompt with N pictures: the processor's pixels (Pillow's Lanczos, then one fused division:
    /// bit for bit), the 3D positions, the language model.
    static func textWithImages(_ n: Int, _ invocation: Invocation) throws -> Report {
        let report = Report(title: "\(model) text --edit \(n)")
        guard let names = pictures[n] else { throw UsageError("--edit \(n): 1 or 2") }
        let stem = "qwen21/goldens-qwen21-text-image-\(n)"
        let goldens = try invocation.goldens.open(stem + ".safetensors")
        guard let prompt = try invocation.goldens.manifest(stem)["prompt"] as? String else { throw Skip("\(stem).json: no prompt") }
        let (artifact, tokenizer) = try encoder()
        let encoder = try Qwen3VLEncoder(artifact: artifact, tokenizer: tokenizer)
        let images = try names.map { try Qwen3VLImages.prepare(Qwen3VLImages.RGB8(contentsOf: try invocation.goldens.input($0)), resolution: 512).pixels }
        let ours = images.flatMap(\.values)
        let pixels = Judge.identical("pixel_values", ours, goldens.float32("pixel_values"))
        report.add(pixels)
        report.add(Judge.integers("image_grid_thw", images.flatMap { [1, $0.gridHeight, $0.gridWidth] }, goldens.int32("image_grid_thw")))
        let prepared = try Qwen3VLPrompt.prepare(prompt: prompt, grids: images.map { ($0.gridHeight, $0.gridWidth) },
                                                 tokenizer: tokenizer, tokens: encoder.tokens)
        report.add(Judge.integers("ids", prepared.ids, goldens.int32("ids")))
        report.add(Judge.integers("mm_token_type_ids", prepared.ids.map { $0 == encoder.tokens.image ? 1 : 0 }, goldens.int32("mm_token_type_ids")))
        report.add(Judge.integers("position_ids", prepared.positions.flatMap { $0 }, goldens.int32("position_ids")))

        // One fault at a time: the encoder runs on the oracle's pixels when ours differ.
        var inputs = images
        if !pixels.passed, let golden = goldens.float32("pixel_values") {
            var offset = 0
            inputs = inputs.map { p in
                defer { offset += p.values.count }
                return Qwen3VLImages.Pixels(values: Array(golden[offset..<(offset + p.values.count)]),
                                            gridHeight: p.gridHeight, gridWidth: p.gridWidth)
            }
        }
        let probes = ["layer0_in", "layer0_out", "layer1_in", "layer1_out", "layer2_in", "layer2_out", "layer3_in",
                      "layer17_out", "layer35_out"]
        encoder.recordedNames = Set(probes)
        let output = try encoder.encode(prompt: prompt, images: inputs)
        let h = output.hidden
        // **From layer 17 on, the text rows carry the verdict.** The image rows of `prompt_embeds` never
        // reach the DiT (the VAE latents replace them, only their effect on the following text does),
        // and their error is the vision's (~6·10⁻⁵) amplified by 36 layers: shown, not judged.
        func split(_ name: String, _ got: [Float]?, imageRow: (Int) -> Bool) {
            guard let expected = goldens.float32(name) else { return }
            guard let got, got.count == expected.count else { report.add(Judge.missing(name)); return }
            for (suffix, image) in [(" text rows", false), (" image rows", true)] {
                let rows = (0..<(got.count / h)).filter { imageRow($0) == image }
                let theirs = rows.flatMap { expected[($0 * h)..<(($0 + 1) * h)] }
                let m = theirs.withUnsafeBufferPointer {
                    Judge.boundary(name + suffix, engine: rows.flatMap { got[($0 * h)..<(($0 + 1) * h)] }, oracle: $0, columns: h)
                }
                if image { print("\(name + suffix): \(String(format: "%.3e", m.worst)) (shown, not judged)") } else { report.add(m) }
            }
        }
        let imageToken = prepared.ids.map { $0 == encoder.tokens.image }
        for name in probes {
            if name == "layer17_out" || name == "layer35_out" { split(name, encoder.boundaries[name]) { imageToken[$0] } }
            else { report.add(goldens, name, encoder.boundaries[name], columns: h) }
        }
        split("prompt_embeds", output.embeddings) { output.imagePadMask[$0] }
        report.add(Judge.integers("image_pad_mask", output.imagePadMask.map { $0 ? 1 : 0 }, goldens.int32("image_pad_mask")))
        return report
    }

    static let vision = Stage(
        model: model, name: "vision",
        proves: "the processor (Lanczos, patches, grids), the 3D positions of both edit templates, and the vision tower with its deepstack, image by image",
        oracle: "tools/oracle/qwen-image-2.1.py vision", weights: ["text encoder"],
        rule: "pixels, grids, identifiers and positions bit for bit; the rule, floor 5e-4"
    ) { invocation in
        let report = Report(title: "\(model) vision")
        let goldens = try invocation.goldens.open("qwen21/goldens-qwen21-vision.safetensors")
        let info = try invocation.goldens.manifest("qwen21/goldens-qwen21-vision")
        let (artifact, tokenizer) = try encoder()
        let encoder = try Qwen3VLEncoder(artifact: artifact, tokenizer: tokenizer)
        let config = encoder.vision
        let images = try pictures[2]!.map { try Qwen3VLImages.prepare(Qwen3VLImages.RGB8(contentsOf: try invocation.goldens.input($0)), resolution: 512).pixels }
        for (i, pixels) in images.enumerated() {
            report.add(Judge.integers("image_grid_thw_\(i)", [1, pixels.gridHeight, pixels.gridWidth], goldens.int32("image_grid_thw_\(i)")))
            report.add(Judge.identical("pixel_values_\(i)", Array(pixels.values), goldens.float32("pixel_values_\(i)")))
        }
        for n in [1, 2] {
            let prompt = try invocation.goldens.manifest("qwen21/goldens-qwen21-text-image-\(n)")["prompt"] as? String ?? ""
            let prepared = try Qwen3VLPrompt.prepare(prompt: prompt, grids: images.prefix(n).map { ($0.gridHeight, $0.gridWidth) },
                                                     tokenizer: tokenizer, tokens: encoder.tokens)
            report.add(Judge.integers("ids_\(n)", prepared.ids, goldens.int32("ids_\(n)")))
            report.add(Judge.integers("position_ids_\(n)", prepared.positions.flatMap { $0 }, goldens.int32("position_ids_\(n)")))
            let delta = (prepared.positions.flatMap { $0 }.max() ?? 0) + 1 - prepared.ids.count
            report.add(Judge.holds("rope_delta_\(n)", delta == info["rope_delta_\(n)"] as? Int, note: "\(delta)"))
        }
        // The tower on the oracle's pixels: one fault at a time.
        let tower = try Qwen3VLVision(artifact: artifact, config: config, maximumPatches: images.map(\.patches).max()!)
        tower.recordedNames = ["vit_block0_out", "vit_last_hidden", "vit_out", "deepstack0", "deepstack1", "deepstack2"]
        for (i, pixels) in images.enumerated() {
            let golden = goldens.float32("pixel_values_\(i)").map(Array.init) ?? Array(pixels.values)
            _ = try tower.encode(Qwen3VLImages.Pixels(values: golden, gridHeight: pixels.gridHeight, gridWidth: pixels.gridWidth))
            for (name, columns) in [("vit_block0_out", config.hidden), ("vit_last_hidden", config.hidden),
                                    ("deepstack0", config.output), ("deepstack1", config.output),
                                    ("deepstack2", config.output), ("vit_out", config.output)] {
                report.add(goldens, "\(name)_\(i)", tower.boundaries[name], columns: columns)
            }
        }
        return report
    }

    static let references = Stage(
        model: model, name: "references",
        proves: "a reference with transparency, prepared as the pipeline does: the VAE's RGBA input and the vision copy over white (at 512, Pillow premultiplies around its Lanczos; at 443 it copies)",
        options: ["--side"], variants: [[], ["--side", "443"]], oracle: "tools/oracle/qwen-image-2.1.py references [--side 443]", weights: [],
        rule: "bit for bit; the vision copy on its bytes (the Pil and Fast processors differ by 6e-8 in float)"
    ) { invocation in
        // `--side`: the reference's prepared side, not an image size — 443 is where the 512×384 test
        // image is copied, not resampled.
        let side = invocation.extras["--side"] ?? "512"
        guard let resolution = Int(side), resolution > 0 else { throw UsageError("--side \(side): a side in pixels (512, 443)") }
        let report = Report(title: "\(model) references \(side)")
        let goldens = try invocation.goldens.open("qwen21/goldens-qwen21-preprocess-alpha-\(side).safetensors")
        guard let vae = goldens.float32("vae_image"), let pixels = goldens.float32("pixel_values") else {
            throw Skip("\(goldens.path): vae_image or pixel_values missing")
        }
        let image = try ImageRGB(contentsOf: try invocation.goldens.input("rgba-512x384.png"))
        let prepared = QwenImage21Pipeline.preparedReference(image, resolution: resolution)
        report.add(Judge.identical("VAE input, RGBA", QwenImage21EncodingModule.rgba(prepared), vae))
        let tower = try Qwen3VLImages.pixels(QwenImage21Pipeline.rgb8(prepared))
        let values = withExtendedLifetime(tower) { Array(tower.values) }
        func bytes(_ v: [Float]) -> [Float] { v.map { ($0 * 127.5 + 127.5).rounded() } }
        report.add(bytes(Array(pixels)).withUnsafeBufferPointer { Judge.identical("vision copy over white, bytes", bytes(values), $0) })
        return report
    }

    // MARK: - The DiT

    static let dit = Stage(
        model: model, name: "dit",
        proves: "one evaluation three ways — prefill of the conditions, the step from their K/V cache, the full pass under the block-causal mask — plus the layout and the modulation; on published int8 weights, along the int8 kernel a render's map may pick (gemm_int8), and the step once on the widened weights",
        options: ["--edit", "--map", "--neural-engine", "--bf16-activations", "--bf16-attention", "--fp16-attention", "--int8-activations", "--no-accelerators", "--variant"], variants: edits,
        oracle: "tools/oracle/qwen-image-2.1.py dit [--edit N]; witness; with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention, bf16 [--edit N]; with --variant compact or light (no --edit), dit and bf16 with --variant \(Wording.publishers); with --int8-activations, w8a8 --variant compact (ComfyUI's own int8 products)", weights: ["DiT"],
        rule: "layout bit for bit; the rule, floor 5e-4, at the fp64 witness where one exists; cache and full pass ≤ 1e-5 apart; a second step the same bits; \(Wording.halfPrecision), cache and full pass too; with --int8-activations, against ComfyUI's W8A8 error + 25 %; on int8 weights, every route by the same rule"
    ) { invocation in
        let n = invocation.edit ?? 0
        let tag = n > 0 ? "-edit\(n)" : ""
        let version = try invocation.quantizedVariant()
        if version != nil, n > 0 { throw UsageError("--variant judges the generation (no --edit): the oracle writes no editing golden for a version") }
        let report = Report(title: "\(model) dit\(tag)\(version.map { " " + $0.rawValue } ?? "") \(invocation.size)")
        let file = variantStem("qwen21/goldens-qwen21-dit\(tag)-\(invocation.size)", version)
        let goldens = try invocation.goldens.open(file + ".safetensors")
        let text = try invocation.goldens.open((n > 0 ? "qwen21/goldens-qwen21-text-image-\(n)" : "qwen21/goldens-qwen21-text") + ".safetensors")
        // The fp64 witness of the evaluation without references.
        let witnessFile = n > 0 || version != nil ? nil : invocation.goldens.optional("qwen21/witness-qwen21-f64-\(invocation.size).safetensors")
        let witness = { (name: String) -> UnsafeBufferPointer<Float>? in witnessFile?.float32(name) }
        let info = try invocation.goldens.manifest(file)
        guard let shapes = info["img_shapes"] as? [[Int]], let mask = goldens.int32("img_mask"),
              let hidden = goldens.float32("hidden_states"), let timestep = goldens.float32("timestep"),
              let embeds = text.float32("prompt_embeds") else {
            throw Skip("\(goldens.path): the evaluation's inputs are missing")
        }
        let artifact = try Artifact(path: try invocation.ditMap(Library.processWide.ditMap(.qwenImage21), family: .qwenImage21, version: version))
        let grids = shapes.map { QwenImage21Sequence.Grid(height: $0[1], width: $0[2]) }
        let sequence = try QwenImage21Sequence(slots: mask.map { $0 != 0 }, images: grids)
        let (p, count) = (sequence.prefix, sequence.count)
        let sigma = timestep[0]
        let config = QwenImage21DiT.Config(header: artifact.header)

        // The layout: pure functions of `img_mask` and `img_shapes`.
        report.add(Judge.integers("image_ids", sequence.imageIds, goldens.int32("image_ids")))
        report.add(Judge.integers("target_token_mask", (0..<count).map { $0 >= p ? 1 : 0 }, goldens.int32("target_token_mask")))
        let segments = (info["segments"] as? [[Any]] ?? []).map { "\($0[0] as? Int ?? -1)-\($0[1] as? Int ?? -1)\(($0[2] as? Bool) == true ? "t" : "i")" }
        let mine = sequence.segments.map { "\($0.start)-\($0.end)\($0.isText ? "t" : "i")" }
        report.add(Judge.holds("segments", segments == mine, note: mine.joined(separator: " ")))
        report.add(Judge.holds("timestep", QwenImage21DiT.timestep(sigma: sigma).bitPattern == timestep[0].bitPattern, note: "bit for bit"))
        var rope = [Float](repeating: 0, count: count * 128)
        rope.withUnsafeMutableBufferPointer {
            QwenImage21Rope.write(into: $0.baseAddress!, positions: sequence.positions, axes: config.axes, theta: config.theta)
        }
        var ropeDeviation: Float = .infinity
        if let cosines = goldens.float32("rope_cos"), let sines = goldens.float32("rope_sin"), cosines.count == count * 64 {
            ropeDeviation = 0
            for i in 0..<(count * 64) { ropeDeviation = max(ropeDeviation, abs(rope[2 * i] - cosines[i]), abs(rope[2 * i + 1] - sines[i])) }
        }
        report.add(Judge.bound("RoPE table, max |Δ|", Double(ropeDeviation), 1e-3, note: "the oracle's angles are fp32"))

        // The latents `[64, h, w]`, unpacked from `[Σ, 64]`.
        let c = config.channels
        var latents: [[Float]] = []
        var offset = 0
        for grid in grids {
            let cells = grid.height * grid.width
            var l = [Float](repeating: 0, count: cells * c)
            for t in 0..<cells { for ch in 0..<c { l[ch * cells + t] = hidden[(offset + t) * c + ch] } }
            latents.append(l)
            offset += cells
        }
        // Published int8 weights (the Compact): the evaluation along the int8 kernel, the step once widened below (`Int8Route`).
        let int8 = Int8Route(artifact, invocation)
        let dit = try QwenImage21DiT(artifact: artifact, sequence: sequence, lora: nil, fullPass: true,
                                     accelerators: .on(neuralEngine: int8?.neuralEngine ?? EngineSettings.effective.plan.usesANE), plan: int8?.plan)
        dit.recordBoundaries = true
        let d = config.dim
        let half = try invocation.reference(file + ".safetensors", script: "tools/oracle/qwen-image-2.1.py bf16\(version.map { " --variant " + $0.rawValue } ?? "")",
                                            w8a8Script: "tools/oracle/qwen-image-2.1.py w8a8 --variant compact")
        let probe = try dit.probeModulation(sigma: sigma)
        report.add(goldens, "temb", probe.temb, witness: witness("temb"), columns: d)
        report.add(goldens, "modulation", probe.modulation, witness: witness("modulation"), columns: 4 * d)

        /// A boundary's rows `[from, to)`, from the golden and the bf16 reference alike.
        func rows(_ name: String, _ from: Int, _ to: Int, of values: [Float]?) -> [Float]? {
            guard let values, values.count >= to * d else { return nil }
            return Array(values[(from * d)..<(to * d)])
        }
        func judgeRows(_ label: String, _ name: String, _ from: Int, _ to: Int) {
            guard let expected = rows(name, from, to, of: goldens.float32(name).map(Array.init)) else { return }
            guard let got = dit.boundaries[name] else { report.add(Judge.missing(label)); return }
            report.add(expected.withUnsafeBufferPointer {
                Judge.boundary(label, engine: got, oracle: $0, half: rows(name, from, to, of: half?.values(name)), columns: d)
            })
        }
        // 1. The prefill, rows [0, p).
        let references = latents.dropLast().map { UnsafeMutablePointer<Float>.allocate(capacity: $0.count) }
        defer { references.forEach { $0.deallocate() } }
        for (i, l) in latents.dropLast().enumerated() { references[i].update(from: l, count: l.count) }
        try embeds.baseAddress.map { try dit.prefill(hidden: $0, references: references.map { UnsafePointer($0) }) }
        for name in ["joint_in", "layer0_out", "layer1_out"] { judgeRows("prefill \(name)", name, 0, p) }
        // 2. The step from the cache, rows [p, n).
        let target = latents.last!
        let cached = try target.withUnsafeBufferPointer { try dit.forward(latent: $0.baseAddress!, sigma: sigma) }
        for name in ["joint_in", "layer0_out", "layer1_out", "layer_last_out", "norm_out"] { judgeRows("step \(name)", name, p, count) }
        let cachedOut = channelsLast(cached, channels: c)
        report.add(goldens, "model_out (step)", cachedOut, name: "model_out", witness: witness("model_out"),
                   half: half?.values("model_out"), columns: c)
        // The cache is read, not consumed: a second step gives the same bits.
        let again = try target.withUnsafeBufferPointer { try dit.forward(latent: $0.baseAddress!, sigma: sigma) }
        report.add(Judge.identity("second step from the cache", differing: zip(again, cached).filter { $0.bitPattern != $1.bitPattern }.count,
                                  of: cached.count))
        // 3. The full pass under the block-causal mask, rows [0, n).
        let full = try target.withUnsafeBufferPointer { try dit.fullForward(latent: $0.baseAddress!, sigma: sigma) }
        for name in ["joint_in", "layer0_out", "layer1_out", "layer_last_out", "norm_out"] { judgeRows("full \(name)", name, 0, count) }
        let fullOut = channelsLast(full, channels: c)
        report.add(goldens, "model_out (full)", fullOut, name: "model_out", witness: witness("model_out"),
                   half: half?.values("model_out"), columns: c)
        // Under the ANE the two paths cut their rows between the engines at different heights (prefill and
        // step against the whole sequence): they differ by fp16 rounding, bounded by the bf16 bar instead.
        var apart = (bound: 1e-5, note: "")
        if invocation.halfPrecision, let bf16 = half?.values("model_out"), let golden = goldens.float32("model_out") {
            apart = (1.25 * channelError(bf16, golden, columns: c).worst, invocation.neuralEngine ? "under the ANE, the bf16 bar" : "on rounded activations, the bf16 bar")
        }
        report.add(fullOut.withUnsafeBufferPointer {
            Judge.bound("prefill + cache against full pass", channelError(cachedOut, $0, columns: c).worst, apart.bound,
                        unit: "per channel", note: apart.note)
        })
        int8?.account(report)
        if int8?.judgesWidened == true {
            // The other route of the same weights, the process's plan: the step from a fresh prefill, by the same rule.
            let widened = try QwenImage21DiT(artifact: artifact, sequence: sequence, lora: nil, fullPass: true)
            try embeds.baseAddress.map { try widened.prefill(hidden: $0, references: references.map { UnsafePointer($0) }) }
            let step = try target.withUnsafeBufferPointer { try widened.forward(latent: $0.baseAddress!, sigma: sigma) }
            report.add(goldens, "model_out (step, widened)", channelsLast(step, channels: c), name: "model_out",
                       witness: witness("model_out"), half: half?.values("model_out"), columns: c)
        }
        if invocation.neuralEngine { try NeuralEngineRun.account(report) }
        if invocation.bf16Activations { try BF16ActivationRun.account(report) }
        if invocation.bf16Attention { try BF16AttentionRun.account(report) }
        if invocation.fp16Attention { try FP16AttentionRun.account(report) }
        if invocation.int8Activations { try W8A8Run.account(report) }
        return report
    }

    /// `[64, h, w]` → `[h·w, 64]`, the oracle's packed layout.
    static func channelsLast(_ v: [Float], channels: Int) -> [Float] {
        let count = v.count / channels
        var out = [Float](repeating: 0, count: v.count)
        for ch in 0..<channels { for t in 0..<count { out[t * channels + ch] = v[ch * count + t] } }
        return out
    }

    static func planar(_ packed: UnsafeBufferPointer<Float>, channels: Int = 64) -> [Float] {
        let count = packed.count / channels
        var out = [Float](repeating: 0, count: packed.count)
        for c in 0..<channels { for t in 0..<count { out[c * count + t] = packed[t * channels + c] } }
        return out
    }

    // MARK: - The trajectory

    static let trajectory = Stage(
        model: model, name: "trajectory",
        proves: "the 8 steps: σ, timesteps and noise to the bit, then each step's velocity from the oracle's latent and encoder states; the product's free run from the real inputs is shown",
        options: ["--edit", "--map"], variants: edits,
        oracle: "tools/oracle/qwen-image-2.1.py trajectory [--edit N]", weights: ["DiT", "text encoder", "VAE"],
        rule: "schedule and noise bit for bit; the rule on each guided velocity, floor 5e-4; the free run's final latent and image are shown, not judged"
    ) { invocation in
        let n = invocation.edit ?? 0
        let tag = n > 0 ? "-edit\(n)" : ""
        let report = Report(title: "\(model) trajectory\(tag) \(invocation.size)")
        let steps = QwenImage21Pipeline.defaultSteps
        let stem = "qwen21/goldens-qwen21-trajectory\(tag)-\(invocation.size)"
        let trajectory = try invocation.goldens.open(stem + ".safetensors")
        let info = try invocation.goldens.manifest(stem)
        let textGoldens = try invocation.goldens.open((n > 0 ? "qwen21/goldens-qwen21-text-image-\(n)" : "qwen21/goldens-qwen21-text") + ".safetensors")
        guard let prompt = info["prompt"] as? String, let height = info["height"] as? Int, let width = info["width"] as? Int,
              let resolution = info["output_resolution"] as? Int, (info["steps"] as? Int) == steps else {
            throw Skip("\(stem).json: prompt, height, width, output_resolution or steps missing")
        }
        let model = try Model.qwenImage21(in: Library.processWide)
        guard let denoiser = model.chain.denoising as? QwenImage21DenoisingModule,
              let textModule = model.chain.text as? QwenImage21TextModule else { throw Skip("not the Qwen-Image-2.1 chain") }
        let (h, w) = (height / 16, width / 16)

        // The references as the product reads them: ImageIO in sRGB must give PIL's raw bytes.
        var prepared: [ImageRGB] = []
        for name in pictures[n] ?? [] {
            let url = try invocation.goldens.input(name)
            let image = try ImageRGB(contentsOf: url)
            report.add(Judge.holds("\(name): product reading = PIL bytes",
                                   QwenImage21Pipeline.rgb8(image) == (try Qwen3VLImages.RGB8(contentsOf: url))))
            prepared.append(QwenImage21Pipeline.image(QwenImage21Pipeline.reference(image, resolution: resolution)))
        }
        if n > 0, let sizes = try invocation.goldens.manifest("qwen21/goldens-qwen21-text-image-\(n)")["condition_sizes"] as? [[Int]] {
            report.add(Judge.holds("condition sizes", prepared.map { [$0.width, $0.height] } == sizes))
        }
        // σ, timesteps and noise, to the bit.
        let sigmas = denoiser.sigmas(steps: steps)
        report.add(Judge.identical("σ", sigmas, trajectory.float32("sigmas")))
        report.add(Judge.identical("timesteps", QwenImage21Pipeline.timesteps(sigmas), trajectory.float32("timesteps")))
        let noise = Latent.noise(.qwenImage21, height: h, width: w, seed: 0)
        report.add(Judge.identical("noise, seed 0", channelsLast(noise.values, channels: 64), trajectory.float32("latent_init")))

        // Guided: the oracle's encoder states and reference latents, each step from the oracle's latent.
        guard let embeds = textGoldens.float32("prompt_embeds"), let rows = textGoldens.entries["prompt_embeds"]?.shape[0],
              let mask = textGoldens.int32("image_pad_mask") else { throw Skip("\(textGoldens.path): prompt_embeds or image_pad_mask") }
        let oracleText = Conditioning(format: .qwenImage21, rows: rows, width: 4096, values: Array(embeds), tokens: rows,
                                      imageSlots: mask.map { $0 != 0 })
        var oracleReferences: [Latent] = []
        if n > 0, let all = trajectory.float32("reference_latents") {
            var offset = 0
            for image in prepared {
                let (rh, rw) = (image.height / 16, image.width / 16)
                oracleReferences.append(Latent(space: .qwenImage21, height: rh, width: rw,
                                               values: planar(UnsafeBufferPointer(rebasing: all[(offset * 64)..<((offset + rh * rw) * 64)]))))
                offset += rh * rw
            }
        }
        var sampler = QwenImage21Sampler(artifact: try Artifact(path: MapCheck.use(invocation.map ?? denoiser.map)), lora: nil)
        for i in 0..<(sigmas.count - 1) {
            var latent = i == 0 ? noise.values : planar(trajectory.float32("latent_\(i)")!)
            var velocity: [Float] = []
            try sampler.run(latent: &latent, height: h, width: w, hidden: oracleText.values, slots: oracleText.imageSlots,
                            references: oracleReferences, sigmas: [sigmas[i], sigmas[i + 1]]) { step in
                velocity = Array(UnsafeBufferPointer(start: step.velocity, count: 64 * h * w))
            }
            if let expected = trajectory.float32("velocity_\(i)") {
                report.add(Judge.boundary("v_\(i)", engine: channelsLast(velocity, channels: 64), oracle: expected, columns: 64))
            }
        }
        // Free: the product's own encoder and VAE encoder, the loop on its own. Shown.
        let context = Context(onProgress: nil)
        let text = try textModule.encoder(prompt, images: prepared, context: context)
        let references = try prepared.map { try model.chain.encoding!.encoder($0, context: context) }
        var latent = noise.values
        try sampler.run(latent: &latent, height: h, width: w, hidden: text.values, slots: text.imageSlots,
                        references: references, sigmas: sigmas) { _ in }
        let final = channelError(channelsLast(latent, channels: 64), trajectory.float32("latent_final")!, columns: 64)
        let vae = try invocation.goldens.open("qwen21/goldens-qwen21-vae\(tag)-\(invocation.size).safetensors")
        let decoded = try model.chain.decoding.decode(Latent(space: .qwenImage21, height: h, width: w, values: latent), context: context)
        let rgb = UnsafeBufferPointer(rebasing: vae.float32("image")![0..<(3 * height * width)])
        print(String(format: "free run from the real inputs: final latent %.3e (median %.3e), image PSNR %.2f dB — shown, not judged",
                     final.worst, final.median, imageError(decoded.pixels, rgb).psnr))
        return report
    }

    // MARK: - The VAE

    static let vae = Stage(
        model: model, name: "vae",
        proves: "the decoder on the trajectory's final latent (with --edit N, the edit's): unpacking and de-normalization to the bit, then the RGBA image plane by plane",
        options: ["--edit"], variants: edits, oracle: "tools/oracle/qwen-image-2.1.py vae [--edit N]; vae-witness; vae-bf16", weights: ["VAE"],
        rule: "vae_in bit for bit; the rule per plane, floor 5e-4, at the fp64 witness where one exists; a decoder in bf16 no further from it than the reference in bf16, + 25 %; on a rectangle, the transposed image must deviate"
    ) { invocation in
        let tag = invocation.edit.map { "-edit\($0)" } ?? ""
        let report = Report(title: "\(model) vae\(tag) \(invocation.size)")
        // A decoder in bf16 is judged at the fp64 witness, which only the plain latent has: an edit's decoding is skipped then.
        if !tag.isEmpty, VAEPrecision.half(plan: EngineSettings.effective.plan, publishedInBF16: QwenImage21VAEDecoder.publishedInBF16) {
            throw Skip("the decoder computes in bf16 and the edit's decoding has no fp64 witness: judged on the plain latent")
        }
        let reference = try DecoderReference.image(invocation.goldens, file: "qwen21/goldens-qwen21-vae-\(invocation.size).safetensors",
                                                   publishedInBF16: QwenImage21VAEDecoder.publishedInBF16,
                                                   script: "tools/oracle/qwen-image-2.1.py vae-bf16")
        let goldens = try invocation.goldens.open("qwen21/goldens-qwen21-vae\(tag)-\(invocation.size).safetensors")
        let witness = tag.isEmpty ? invocation.goldens.optional("qwen21/witness-qwen21-vae-f64-\(invocation.size).safetensors") : nil
        guard let tokensEntry = goldens.entries["latent_final"], let tokens = goldens.float32("latent_final"),
              let vaeIn = goldens.float32("vae_in"), let vaeInEntry = goldens.entries["vae_in"],
              let expected = goldens.float32("image") else {
            throw Skip("\(goldens.path): latent_final, vae_in or image missing")
        }
        let (h, w) = (vaeInEntry.shape[1], vaeInEntry.shape[2])
        let planar = QwenImage21VAE.unpack(Array(tokens), channels: tokensEntry.shape[1], height: h, width: w)
        let decoder = try QwenImage21VAEDecoder(path: try Library.processWide.component(.qwenImage21, "vae.safetensors"),
                                                latentHeight: h, latentWidth: w)
        // `_unpack_latents` then `z · std + mean`: exact arithmetic, the same bits.
        report.add(Judge.identical("unpack · ×std + mean", QwenImage21VAE.denormalize(planar, config: decoder.config), vaeIn))
        let (rgba, _) = try decoder.decode(latent: planar)
        report.add(Judge.planes("image RGBA", engine: rgba, oracle: expected, witness: witness?.float32("image"), reference: reference,
                                planes: 4))
        if let m = Judge.orientation(rgba, expected, height: decoder.imageHeight, width: decoder.imageWidth) { report.add(m) }
        return report
    }

    static let vaeEncode = Stage(
        model: model, name: "vae-encode",
        proves: "the encoder on both edit pictures (RGBA): the raw mean and the normalized latent, plane by plane; the product's module (RGB in, alpha added) gives the same bits",
        oracle: "tools/oracle/qwen-image-2.1.py vae-encode; vae-witness", weights: ["VAE"],
        rule: "the rule per plane, floor 5e-4, at the fp64 witness where one exists; the module bit for bit"
    ) { invocation in
        let report = Report(title: "\(model) vae-encode \(invocation.size)")
        let path = try Library.processWide.component(.qwenImage21, "vae.safetensors")
        let goldens = try invocation.goldens.open("qwen21/goldens-qwen21-vae-encode-\(invocation.size).safetensors")
        let witness = invocation.goldens.optional("qwen21/witness-qwen21-vae-f64-\(invocation.size).safetensors")
        for i in 0..<2 {
            guard let entry = goldens.entries["image_\(i)"], let image = goldens.float32("image_\(i)"),
                  let raw = goldens.float32("latent_mean_raw_\(i)"), let expected = goldens.float32("latent_\(i)") else {
                throw Skip("\(goldens.path): image_\(i) or latent_\(i) missing")
            }
            let (H, W) = (entry.shape[1], entry.shape[2])
            let encoder = try QwenImage21VAEEncoder(path: path, height: H, width: W)
            let output = try encoder.encode(rgba: Array(image))
            let plane = encoder.latentHeight * encoder.latentWidth
            report.add(Judge.planes("picture \(i) mean (raw)", engine: Array(output.moments.prefix(64 * plane)), oracle: raw, planes: 64))
            report.add(Judge.planes("picture \(i) latent", engine: output.latent, oracle: expected,
                                    witness: witness?.float32("latent_\(i)"), planes: 64))
            if let m = Judge.orientation(output.latent, expected, height: encoder.latentHeight, width: encoder.latentWidth) { report.add(m) }
            let module = try QwenImage21EncodingModule(path: path).encoder(
                ImageRGB(pixels: Array(image.prefix(3 * H * W)), height: H, width: W), context: Context(onProgress: nil))
            report.add(Judge.identity("picture \(i) module (RGB → RGBA)", differing: zip(module.values, output.latent).filter { $0.bitPattern != $1.bitPattern }.count,
                                      of: output.latent.count))
        }
        return report
    }
}
