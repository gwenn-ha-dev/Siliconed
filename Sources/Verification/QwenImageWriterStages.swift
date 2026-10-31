import CryptoKit
import Foundation
import Siliconed

// **Qwen-Image-2.1's encoder made to write** — `siliconed check qwen-image-2.1 writer [--case system|plain|describe]`.
//
// The Prompt Lab (`PromptWriter`) has Qwen3-VL-8B continue a chat: greedy decoding with a K/V cache. Three
// cases, the bench prompt « woman posing in a library »: a rewrite with a system turn, the same without
// one, and the description of a 512×512 picture. What is judged, in this order, each failure stopping the
// next (one fault at a time):
//
//   1. the head map reproduces the bits of `lm_head.weight` and the final norm in the published file;
//   2. the prompt: identifiers bit for bit (the template with and without a system turn, the image slots),
//      and for the picture its `pixel_values` and grid;
//   3. the tokens: the engine, **teacher-forced on the reference's tokens** (`Qwen3VLGenerator.Teacher`),
//      picks at every step the token the reference picked — and since a free run follows the forced path
//      until the first difference, this proves the free run's tokens too;
//   4. the logits at every step, the registry's rule (`Judge.boundary`, floor 5e-4) on the `[steps, vocabulary]`
//      matrix, a vocabulary entry being a channel.
//
// **A divergence is a tie, not a fault, when the reference's gap is within what the rule allows.** Where the
// engine picks `c` and the reference `r`, the reference's gap `g = logit_r − logit_c` is read in its own
// logits; if `g ≤ 2·floor·max(|logit_r|, |logit_c|)` the two logits are within the judge's tolerance of each
// other, and either order is a correct fp32 answer. Such a step is reported ("tie"), not failed; any other
// divergence fails. The smallest gap of each golden (top-1 minus top-2) is printed: a gap that small is where
// a tie could happen.
//
// The reference reads the Standard (bf16) map's weights widened to fp32, which is what the engine's Standard
// map holds. A Compact (int8 convrot) map rounds differently by construction: its reference is the oracle run with
// `--map` on that very map, every linear dequantized by Comfy's own function in fp32 (goldens `…-compact`, the two
// text cases: the picture's would need the vision tower dequantized too).
// The map is streamed (13.9 GB of bf16 do not fit resident on 16 GB, which is why `PromptWriter` refuses the
// Standard there): each step reads every layer again, seconds a token.

enum QwenImageWriterStages {
    static let all: [Stage] = [writer]

    static let cases = ["system", "plain", "describe"]

    /// The three texts of the cases, to the character those of `tools/oracle/qwen-image-2.1-writer.py`
    /// (the golden's manifest carries the oracle's; a difference fails the stage before anything runs).
    static let prompt = "woman posing in a library"
    static let system = "You write prompts for a text-to-image model. Rewrite the user's short prompt as one vivid, detailed "
        + "paragraph: subject, pose, clothing, setting, light, mood. Answer with the prompt only."
    static let rewritePlain = "Rewrite this short prompt for a text-to-image model as one vivid, detailed paragraph, "
        + "and answer with the prompt only: "
    static let describe = "Describe this picture as a prompt for a text-to-image model: one vivid, detailed paragraph. "
        + "Answer with the prompt only."

    static let writer = Stage(
        model: QwenImageStages.model, name: "writer",
        proves: "Qwen3-VL-8B writes: the head map is the published lm_head and final norm; the chat prompt (with or without a system turn, with a picture); greedy decoding with a K/V cache, teacher-forced on the reference's tokens — the tokens and the logits at every step",
        options: ["--case", "--map"], oracle: "tools/oracle/qwen-image-2.1-writer.py [CASE]",
        weights: ["text encoder", "writer head"],
        rule: "head, identifiers and tokens bit for bit (a step whose gap is within the rule is a tie); the logits by the rule, floor 5e-4, a vocabulary entry a channel"
    ) { invocation in
        guard invocation.extras["--case"].map(cases.contains) ?? true else {
            throw UsageError("--case \(invocation.extras["--case"] ?? ""): " + cases.joined(separator: ", "))
        }
        let library = Library.processWide
        guard let headPath = library.writerHeadMap(.qwenImage21) else {
            throw Skip("the writer's head map is not installed (\(Family.qwenImage21.writerHead ?? "")) — the Prompt Lab's installation forges it")
        }
        // The map decides the golden: the reference of a Compact (int8) map read that very map, dequantized
        // (`--map` of the oracle), so its goldens are apart (`-compact`), and the picture's case is not among them
        // (the vision tower is not dequantized).
        let mapPath = invocation.map ?? library.map(library.encoderMap(.qwenImage21))
        let variant = try MapSummary(path: mapPath).variant
        guard variant != .packed else { throw Skip("a packed (GGUF) text encoder: no oracle reads it") }
        let encoderPath = try MapCheck.use(mapPath, golden: variant)
        let suffix = variant == .standard ? "" : "-compact"
        let selected = invocation.extras["--case"].map { [$0] } ?? cases.filter { suffix.isEmpty || $0 != "describe" }
        if !suffix.isEmpty, selected.contains("describe") { throw Skip("the Compact's picture case has no reference: its vision tower is not dequantized") }
        let artifact = try Artifact(path: encoderPath)
        let head = try Artifact(path: headPath)
        let tokenizer = try Tokenizer(directory: try library.component(.qwenImage21, "processor"))
        let generator = try Qwen3VLGenerator(encoder: artifact, head: head, tokenizer: tokenizer)
        // The path the Prompt Lab runs: a Compact's layers and head resident, each step on the GPU
        // (`Qwen3VLDecoder`, the CPU's with `SILICONED_WRITER_CPU_DECODE=1`); the Standard's 16 GB do not stay
        // in memory and stream through the CPU's passes.
        if variant != .standard { try generator.load() }
        let gpu = generator.decodesOnGPU && generator.decoder != nil
        print("  decoding: " + (gpu ? "GPU, resident weights" : "CPU passes" + (generator.decoderNote.map { " (\($0))" } ?? "")))
        let report = Report(title: "\(QwenImageStages.model) writer\(invocation.extras["--case"].map { " --case " + $0 } ?? "")")
        var headChecked = false
        for name in selected {
            let stem = "qwen21/goldens-qwen21-writer-\(name)\(suffix)"
            let goldens = try invocation.goldens.open(stem + ".safetensors")
            let info = try invocation.goldens.manifest(stem)
            if !headChecked {
                report.add(headMeasure(head, published: info["head_sha256"] as? [String: String] ?? [:]))
                headChecked = true
                guard report.measures.allSatisfy(\.passed) else { return report }
            }
            try check(name, suffix: suffix, generator: generator, tokenizer: tokenizer, goldens: goldens, info: info,
                      invocation: invocation, report: report)
        }
        return report
    }

    // MARK: - The head map

    /// The two tensors of the head map, byte for byte against the sha256 of the published file's.
    static func headMeasure(_ head: Artifact, published: [String: String]) -> Measure {
        var differing: [String] = []
        for (name, origin) in [("lm_head.weight", "lm_head.weight"), ("norm.weight", "model.language_model.norm.weight")] {
            guard let entry = head.tensors[name], let pointer = head.pointer(name), let expected = published[origin] else {
                differing.append(name + " (absent)"); continue
            }
            let digest = SHA256.hash(data: Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: pointer), count: entry.bytes, deallocator: .none))
            if digest.map({ String(format: "%02x", $0) }).joined() != expected { differing.append(name) }
        }
        return Judge.holds("head map", differing.isEmpty,
                           note: differing.isEmpty ? "lm_head and norm: the published bits (sha256)" : "differ: " + differing.joined(separator: ", "))
    }

    // MARK: - One case

    static func check(_ name: String, suffix: String, generator: Qwen3VLGenerator, tokenizer: Tokenizer, goldens: Safetensors,
                      info: [String: Any], invocation: Invocation, report: Report) throws {
        let theirs = info["case"] as? String == name
        let texts: [String: String?] = ["system": info["system"] as? String, "prompt": info["prompt"] as? String,
                                        "rewrite_plain": info["rewrite_plain"] as? String, "describe": info["describe"] as? String]
        let expectedTexts: [String: String?] = [
            "system": name == "system" ? system : nil, "prompt": name == "describe" ? nil : prompt,
            "rewrite_plain": name == "plain" ? rewritePlain : nil, "describe": name == "describe" ? describe : nil]
        report.add(Judge.holds("\(name): instructions", theirs && texts == expectedTexts,
                               note: "the oracle's texts are the stage's"))
        guard let maxNew = info["max_new_tokens"] as? Int, let reference = goldens.int32("generated"),
              let logits = goldens.float32("logits"), let gaps = goldens.float32("gaps") else {
            throw Skip("goldens-qwen21-writer-\(name)\(suffix): generated, logits, gaps, max_new_tokens missing")
        }
        let steps = reference.count, vocabulary = generator.vocabulary
        guard logits.count == steps * vocabulary, gaps.count == steps else { throw Skip("goldens-qwen21-writer-\(name)\(suffix): shapes") }

        // ── the prompt ───────────────────────────────────────────────────────────────────────────
        var grids: [(Int, Int)] = [], images: [Qwen3VLImages.Pixels] = []
        if name == "describe" {
            guard let picture = info["picture"] as? String else { throw Skip("manifest: no picture") }
            let image = try Qwen3VLImages.RGB8(contentsOf: try invocation.goldens.input(picture))
            let prepared = try Qwen3VLChat.pixels(image, maxPixels: info["max_pixels"] as? Int ?? Qwen3VLChat.maxPixels)
            var pixels = prepared.pixels
            let identical = Judge.identical("\(name): pixel_values", Array(pixels.values), goldens.float32("pixel_values"))
            report.add(identical)
            report.add(Judge.integers("\(name): image_grid_thw", [1, pixels.gridHeight, pixels.gridWidth], goldens.int32("image_grid_thw")))
            if !identical.passed, let golden = goldens.float32("pixel_values"), golden.count == pixels.values.count {
                pixels = Qwen3VLImages.Pixels(values: Array(golden), gridHeight: pixels.gridHeight, gridWidth: pixels.gridWidth)
                print("  → the generator reads the ORACLE's pixel_values")
            }
            grids = [(pixels.gridHeight, pixels.gridWidth)]; images = [pixels]
        }
        let user = name == "plain" ? rewritePlain + prompt : name == "describe" ? describe : prompt
        let prepared = try Qwen3VLChat.prepare(system: name == "system" ? system : nil, user: user, grids: grids,
                                               tokenizer: tokenizer, tokens: generator.tokens)
        report.add(Judge.integers("\(name): ids", prepared.ids, goldens.int32("ids")))
        guard report.measures.allSatisfy(\.passed) else { return }

        // ── generation, teacher-forced ──────────────────────────────────────────────────────────
        let tokens = reference.map(Int.init)
        var choices: [Int] = [], engine: [Float] = []
        engine.reserveCapacity(steps * vocabulary)
        generator.teacher = .init(tokens: tokens) { _, choice, row in
            choices.append(choice); engine.append(contentsOf: row)
        }
        defer { generator.teacher = nil }
        let peak = FootprintPeak()
        let started = Date()
        let output = try generator.generate(.init(ids: prepared.ids, positions: prepared.positions, images: images),
                                            maxNewTokens: min(steps, maxNew))
        let seconds = Date().timeIntervalSince(started)
        let footprint = peak.stop()
        print(String(format: "  %@: %d prompt + %d tokens in %.0f s (vision %.1f, prefill %.1f, decode %.1f), peak %.2f GB",
                     name, output.promptTokens, output.tokens.count, seconds, generator.timings["vision"] ?? 0,
                     generator.timings["prefill"] ?? 0, generator.timings["decode"] ?? 0, Double(footprint) / 1e9))
        guard output.tokens == tokens, choices.count == steps else {
            report.add(Judge.holds("\(name): forced run", false, note: "\(output.tokens.count) tokens forced, \(steps) expected"))
            return
        }

        // ── the tokens: the engine's pick against the reference's, ties said ─────────────────────
        var ties: [String] = [], faults: [String] = []
        for step in 0..<steps where choices[step] != tokens[step] {
            let row = step * vocabulary, r = tokens[step], c = choices[step]
            let gap = logits[row + r] - logits[row + c]
            let allowed = 2 * Float(Judge.floor) * max(abs(logits[row + r]), abs(logits[row + c]))
            let line = String(format: "step %d: ours %d, reference %d, reference gap %.4f (allowed %.4f)", step, c, r, gap, allowed)
            if gap <= allowed { ties.append(line) } else { faults.append(line) }
        }
        let smallest = gaps.indices.min { gaps[$0] < gaps[$1] }
        let note = String(format: "%d of %d steps pick the reference's token; smallest reference gap %.4f at step %d",
                          steps - ties.count - faults.count, steps, smallest.map { gaps[$0] } ?? .nan, smallest ?? -1)
        report.add(Judge.holds("\(name): tokens", faults.isEmpty,
                               note: note + (ties.isEmpty ? "" : "; ties: " + ties.joined(separator: " | "))
                               + (faults.isEmpty ? "" : "; FAULTS: " + faults.joined(separator: " | "))))

        // ── the logits: the rule, then each step's figure for the reader ─────────────────────────
        report.add(Judge.boundary("\(name): logits", engine: engine, oracle: logits, columns: vocabulary))
        var worstStep = (value: 0.0, step: 0), largest: Float = 0, perStep = 0.0
        for step in 0..<steps {
            let row = Array(engine[(step * vocabulary)..<((step + 1) * vocabulary)])
            let expected = UnsafeBufferPointer(rebasing: logits[(step * vocabulary)..<((step + 1) * vocabulary)])
            var numerator = 0.0, denominator = 0.0
            for j in 0..<vocabulary {
                let d = Double(row[j]) - Double(expected[j]); numerator += d * d
                denominator += Double(expected[j]) * Double(expected[j])
                largest = max(largest, abs(row[j] - expected[j]))
            }
            perStep = max(perStep, channelError(row, expected, columns: vocabulary).worst)
            let relative = (numerator / denominator).squareRoot()
            if relative > worstStep.value { worstStep = (relative, step) }
        }
        print(String(format: "  %@: per step, relative L2 of the whole row: worst %.2e (step %d); largest |Δ logit| %.2e; "
                     + "channelError of a single step's row (each logit a channel, shown): worst %.2e",
                     name, worstStep.value, worstStep.step, largest, perStep))
        let text = Qwen3VLChat.text(tokens, tokenizer: tokenizer)
        report.add(Judge.holds("\(name): text", text == (info["text"] as? String ?? ""),
                               note: "« \(text.prefix(70))\(text.count > 70 ? "…" : "") »"))
    }
}
