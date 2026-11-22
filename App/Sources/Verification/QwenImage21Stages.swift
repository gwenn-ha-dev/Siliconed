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
        let path = try invocation.ditMap(ditMap, compactMap, compact: compact)
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
        try report.rendered(render, since: started, invocation, references: true)
        // The bf16 render's RGB, its RGBA's alpha dropped.
        let reference = half?.float32("image").map { Array($0.prefix($0.count / 4 * 3)) }
        expected.withUnsafeBufferPointer { report.judgePicture(render.image.pixels, $0, compact: compact, half: reference) }
        return report
    }
}
