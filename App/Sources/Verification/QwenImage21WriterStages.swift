import Foundation
import Siliconed

// **Qwen-Image-2.1's encoder made to write** — `siliconed check qwen-image-2.1 writer [--case system|plain|describe]`.
//
// The prompt writer has Qwen3-VL-8B continue a chat: greedy decoding with a K/V cache (`Qwen3VLGenerator`). Three
// cases on the bench prompt « woman posing in a library »: a rewrite with a system turn, the same without one, and
// the description of a 512×512 picture. What is judged, in this order, each failure stopping the next:
//
//   1. the prompt: identifiers bit for bit (the template with and without a system turn, the picture's slots), and
//      for the picture its `pixel_values`;
//   2. the logits, teacher-forced on the reference's tokens: the whole forward over the prompt and the reference's
//      answer (`continuation`), by the rule beside the reference's own bf16 run — a vocabulary entry a channel;
//   3. the tokens that forward picks: at every step the fp32 reference's, or the bf16 reference's where the two
//      differ (a tie the precision decides);
//   4. the decoding itself: `generate`, free, writes the tokens its own whole forward picks — the first one and nine
//      in ten at least, a prefill and a decode step not being the same program to the bit (a bar fixed before the
//      first run; the engine's own decode test holds the same bar).
//
// The oracle writes the fp32 goldens; `--bf16` writes the witness, teacher-forced on them.
//
// `--variant compact` is the Compact's encoder (its int8 codes, W8A16) on the `system` case: the prompt and the decoding
// are judged as above; the tokens are counted against the oracle's fp32 run on the same weights dequantized and said, not
// judged — no reference computes the int8 products of this encoder, and a Compact is judged at its DiT.

enum QwenImage21WriterStages {
    static let all: [Stage] = [writer]

    static let cases = ["system", "plain", "describe"]

    /// The cases' texts, to the character those of `tools/oracle/qwen-image-2.1-writer.py` (the golden's manifest
    /// carries the oracle's; a difference fails the stage before anything runs).
    static let prompt = "woman posing in a library"
    static let system = "You write prompts for a text-to-image model. Rewrite the user's short prompt as one vivid, detailed "
        + "paragraph: subject, pose, clothing, setting, light, mood. Answer with the prompt only."
    static let rewritePlain = "Rewrite this short prompt for a text-to-image model as one vivid, detailed paragraph, "
        + "and answer with the prompt only: "
    static let describe = "Describe this picture as a prompt for a text-to-image model: one vivid, detailed paragraph. "
        + "Answer with the prompt only."

    static let writer = Stage(
        model: QwenImage21Stages.model, name: "writer",
        proves: "Qwen3-VL-8B writes: the chat prompt (with or without a system turn, with a picture), the logits teacher-forced on the reference's tokens, and greedy decoding with a K/V cache writing what the whole forward picks",
        options: ["--case", "--variant"], oracle: "tools/oracle/qwen-image-2.1-writer.py [CASE]; qwen-image-2.1-writer.py --bf16 [CASE]; qwen-image-2.1-writer.py system --variant compact",
        weights: ["text encoder"],
        rule: "identifiers and pixels bit for bit; the logits by the rule beside the reference's bf16 run; each token the fp32 reference's or the bf16 one's; the decoding's tokens its own whole forward's, the first and nine in ten at least; with --variant compact (the system case), the identifiers and the decoding, the tokens said and not judged"
    ) { invocation in
        guard invocation.extras["--case"].map(cases.contains) ?? true else {
            throw UsageError("--case \(invocation.extras["--case"] ?? ""): " + cases.joined(separator: ", "))
        }
        let compact = try invocation.compact()
        if compact, let other = invocation.extras["--case"], other != "system" {
            throw UsageError("--variant compact judges the system case: the oracle writes no other golden for it")
        }
        let library = Library.processWide
        let tokenizer = try Tokenizer(directory: library.map(QwenImage21Stages.tokenizer))
        let file = Family.qwenImage21.encoderMap(compact ? .compact : .standard)
        let path = library.map(file)
        guard FileManager.default.fileExists(atPath: path) else { throw Skip("map missing: \(path) — siliconed install") }
        let generator = try Qwen3VLGenerator(map: path, config: .qwenImage21)
        guard generator.writes else { throw Skip("\(file) holds no lm_head: nothing writes") }
        let report = Report(title: "\(QwenImage21Stages.model) writer\(compact ? " compact" : "")\(invocation.extras["--case"].map { " --case " + $0 } ?? "")")
        if compact {
            try checkCompact(generator: generator, tokenizer: tokenizer, invocation: invocation, report: report)
            return report
        }
        for name in invocation.extras["--case"].map({ [$0] }) ?? cases {
            try check(name, generator: generator, tokenizer: tokenizer, invocation: invocation, report: report)
            guard report.measures.allSatisfy(\.passed) else { break }
        }
        return report
    }

    /// The Compact on the system case: the prompt, the tokens its whole forward picks beside the oracle's (said), and
    /// the decoding against that forward.
    static func checkCompact(generator: Qwen3VLGenerator, tokenizer: Tokenizer, invocation: Invocation, report: Report) throws {
        let stem = "qwen21/goldens-qwen21-writer-system-compact"
        let goldens = try invocation.goldens.open(stem + ".safetensors")
        let info = try invocation.goldens.manifest(stem)
        report.add(Judge.holds("system: instructions", info["system"] as? String == system && info["prompt"] as? String == prompt,
                               note: "the oracle's texts are the stage's"))
        guard let reference = goldens.int32("generated") else { throw Skip("\(goldens.path): generated missing") }
        let tokens = reference.map(Int.init)
        let prepared = try Qwen3VLChat.prepare(system: system, user: prompt, grids: [], tokenizer: tokenizer)
        report.add(Judge.integers("system: ids", prepared.ids, goldens.int32("ids")))
        guard report.measures.allSatisfy(\.passed) else { return }
        let chat = Qwen3VLGenerator.Prompt(ids: prepared.ids, slots: prepared.slots, pictures: [])
        let forced = try generator.continuation(chat, answer: tokens)
        let same = zip(forced.next, tokens).filter { $0 == $1 }.count
        report.add(Judge.holds("system: tokens", true, note: "\(same) of \(tokens.count) the fp32 run's on the dequantized weights, said and not judged"))
        try decoding("system", chat: chat, steps: tokens.count, reference: tokens, generator: generator, tokenizer: tokenizer, report: report)
    }

    static func check(_ name: String, generator: Qwen3VLGenerator, tokenizer: Tokenizer, invocation: Invocation, report: Report) throws {
        let stem = "qwen21/goldens-qwen21-writer-\(name)"
        let goldens = try invocation.goldens.open(stem + ".safetensors")
        let half = try invocation.goldens.half("qwen21/witness-bf16-qwen21-writer-\(name).safetensors",
                                               script: "tools/oracle/qwen-image-2.1-writer.py --bf16 \(name)")
        let info = try invocation.goldens.manifest(stem)
        let texts: [String: String?] = ["system": info["system"] as? String, "prompt": info["prompt"] as? String,
                                        "rewrite_plain": info["rewrite_plain"] as? String, "describe": info["describe"] as? String]
        let expected: [String: String?] = [
            "system": name == "system" ? system : nil, "prompt": name == "describe" ? nil : prompt,
            "rewrite_plain": name == "plain" ? rewritePlain : nil, "describe": name == "describe" ? describe : nil]
        report.add(Judge.holds("\(name): instructions", info["case"] as? String == name && texts == expected, note: "the oracle's texts are the stage's"))
        guard let reference = goldens.int32("generated"), let logits = goldens.float32("logits"), let witness = half.float32("logits") else {
            throw Skip("goldens-qwen21-writer-\(name): generated or logits missing")
        }
        let tokens = reference.map(Int.init), steps = tokens.count, vocabulary = Qwen3Encoder.Config.qwenImage21.vocabulary
        guard logits.count == steps * vocabulary, witness.count == logits.count else { throw Skip("goldens-qwen21-writer-\(name): shapes") }

        // ── the prompt ──
        var pictures: [Qwen3VLImages.Pixels] = []
        if name == "describe" {
            guard let picture = info["picture"] as? String else { throw Skip("manifest: no picture") }
            let pixels = try Qwen3VLChat.pixels(Qwen3VLImages.RGB8(contentsOf: try invocation.goldens.input(picture)),
                                                maxPixels: info["max_pixels"] as? Int ?? Qwen3VLChat.maxPixels)
            report.add(Judge.identical("\(name): pixel_values", withExtendedLifetime(pixels) { Array(pixels.values) }, goldens.float32("pixel_values")))
            pictures = [pixels]
        }
        let user = name == "plain" ? rewritePlain + prompt : name == "describe" ? describe : prompt
        let prepared = try Qwen3VLChat.prepare(system: name == "system" ? system : nil, user: user,
                                               grids: pictures.map { (height: $0.gridHeight, width: $0.gridWidth) }, tokenizer: tokenizer)
        report.add(Judge.integers("\(name): ids", prepared.ids, goldens.int32("ids")))
        guard report.measures.allSatisfy(\.passed) else { return }
        let chat = Qwen3VLGenerator.Prompt(ids: prepared.ids, slots: prepared.slots, pictures: pictures)

        // ── the logits and the picks, teacher-forced ──
        let started = Date()
        let forced = try generator.continuation(chat, answer: tokens)
        print(String(format: "  %@: %d prompt + %d tokens, whole forward in %.1f s", name, prepared.ids.count, steps, Date().timeIntervalSince(started)))
        report.add(Judge.boundary("\(name): logits", engine: forced.logits, oracle: logits, half: Array(witness), columns: vocabulary))
        var others: [String] = [], faults: [String] = []
        for step in 0..<steps where forced.next[step] != tokens[step] {
            let row = UnsafeBufferPointer(rebasing: witness[(step * vocabulary)..<((step + 1) * vocabulary)])
            let theirs = row.indices.max { row[$0] < row[$1] || (row[$0] == row[$1] && $0 > $1) }.map { $0 - row.startIndex } ?? -1
            let line = "step \(step): ours \(forced.next[step]), fp32 \(tokens[step]), bf16 \(theirs)"
            if forced.next[step] == theirs { others.append(line) } else { faults.append(line) }
        }
        report.add(Judge.holds("\(name): tokens", faults.isEmpty,
                               note: "\(steps - others.count - faults.count) of \(steps) the fp32 reference's"
                               + (others.isEmpty ? "" : "; the bf16 reference's: " + others.joined(separator: " | "))
                               + (faults.isEmpty ? "" : "; FAULTS: " + faults.joined(separator: " | "))))

        try decoding(name, chat: chat, steps: steps, reference: tokens, generator: generator, tokenizer: tokenizer, report: report)
    }

    /// The decoding, free: `generate` writes the tokens its own whole forward picks.
    static func decoding(_ name: String, chat: Qwen3VLGenerator.Prompt, steps: Int, reference tokens: [Int], generator: Qwen3VLGenerator,
                         tokenizer: Tokenizer, report: Report) throws {
        let started = Date()
        let output = try generator.generate(chat, limit: steps, stops: Qwen3VLChat.stops(tokenizer))
        let seconds = Date().timeIntervalSince(started), t = generator.timings
        print(String(format: "  %@: %d tokens decoded in %.1f s (vision %.1f, preparation %.1f, run %.1f)", name, output.tokens.count, seconds,
                     t.vision, t.preparation, t.run))
        print("    « \(Qwen3VLChat.text(output.tokens, tokenizer: tokenizer)) »")
        guard !output.tokens.isEmpty else { report.add(Judge.holds("\(name): decoding", false, note: "no token")); return }
        let own = try generator.continuation(chat, answer: output.tokens)
        let differing = zip(own.next, output.tokens).enumerated().filter { $0.element.0 != $0.element.1 }.map(\.offset)
        report.add(Judge.holds("\(name): decoding", differing.first != 0 && differing.count * 10 <= output.tokens.count,
                               note: "\(output.tokens.count - differing.count) of \(output.tokens.count) tokens the whole forward's"
                               + (differing.isEmpty ? "" : " (differ at \(differing))")
                               + (output.tokens == tokens ? "; the reference's answer" : "")))
    }
}
