import Foundation
import Siliconed

// **MOSS-Image's text side, stage by stage** — `siliconed check moss-image <stage>`.
//
//     tokenizer → text-delta → text-full → text → (dit → vae …, their own files)
//
// The encoder is Qwen3.5-9B (Gated DeltaNet ×3, full attention ×1, per group of four layers) and its output is the
// DiT's `prompt_embeds`: `last_hidden_state`, after the final norm, every token of the chat template. The goldens are
// the reference prompt's, one unpadded forward (`tools/oracle/moss-image.py tokenizer`, `text`, `text-witness`),
// written under `moss/` in the goldens folder. The map and the tokenizer are the installed family's
// (`store/`, `store/composants/moss-image/tokenizer`), never a download folder.
//
// `text-delta` and `text-full` stop early (one layer, four layers): they read 1/32 and 4/32 of the map, so a
// fault of a layer's arithmetic shows in seconds instead of after the whole stream. `text` runs the 32.

enum MossImageEncoderStages {
    static let all: [Stage] = [tokenizer, text, textLong]

    static let model = MossImageStages.model
    static let mapName = Family.mossImage.encoderMap(.standard)
    static let goldenStem = "moss/goldens-moss-text"

    static let tokenizer = Stage(
        model: model, name: "tokenizer",
        proves: "the chat template (no system turn, thinking block open) and the Qwen2 tokenizer against `tokenizers` on six prompts: accents, a newline and tab, JSON, digits and punctuation, a long text; cut at 4096",
        oracle: "tools/oracle/moss-image.py tokenizer", weights: ["tokenizer"], rule: "the template string and every identifier bit for bit"
    ) { invocation in
        let report = Report(title: "\(model) tokenizer")
        let goldens = try invocation.goldens.open("moss/goldens-moss-tokenizer.safetensors")
        let manifest = try invocation.goldens.manifest("moss/goldens-moss-tokenizer")
        guard let templates = manifest["templates"] as? [String: [String: Any]] else { throw Skip("goldens-moss-tokenizer.json: no templates") }
        let tokenizer = try Tokenizer(directory: try tokenizerDirectory())
        for name in goldens.entries.keys.sorted() {
            guard let entry = templates[name], var prompt = entry["prompt"] as? String else { continue }
            if prompt == "<common.LONG_PROMPT>" {
                // The long prompt is a file of the oracle, not a string of the manifest.
                let candidates = ["tools/oracle/inputs/long-prompt.txt", "public/tools/oracle/inputs/long-prompt.txt", invocation.goldens.path("inputs/long-prompt.txt")]
                guard let file = candidates.first(where: FileManager.default.fileExists(atPath:)),
                      let text = try? String(contentsOfFile: file, encoding: .utf8) else {
                    print("  \(name): the long prompt's file is not found, case left out")
                    continue
                }
                prompt = text.trimmingCharacters(in: .whitespacesAndNewlines)
            } else if let formatted = entry["formatted"] as? String {
                report.add(Judge.holds("\(name) template", Qwen35Prompt.template(prompt) == formatted,
                                       note: Qwen35Prompt.template(prompt) == formatted ? "" : "differs from the chat template's"))
            }
            let ids = Qwen35Prompt.ids(prompt, tokenizer: tokenizer)
            report.add(Judge.integers("\(name) ids (\(ids.count))", ids, goldens.int32(name)))
        }
        return report
    }

    // MARK: - The text

    static let text = Stage(
        model: model, name: "text",
        proves: "the Qwen3.5-9B encoder (32 layers, Gated DeltaNet and full attention, bf16) gives the reference's layer outputs and last_hidden_state (the DiT's prompt_embeds) on the real tokens",
        options: ["--variant"], variants: [[], ["--variant", "compact"]],
        oracle: "tools/oracle/moss-image.py text; text-witness; text-bf16", weights: ["text encoder"],
        rule: "the rule against transformers in bf16, at the fp64 witness (with --variant compact, the Compact's int8 encoder against the same references)"
    ) { invocation in try judged(invocation, title: "text", suffix: "", script: "tools/oracle/moss-image.py text-bf16") }

    /// The same judgement past one chunk of the gated delta rule: the recurrent state carried from chunk to chunk,
    /// and a short last chunk.
    static let textLong = Stage(
        model: model, name: "text-long",
        proves: "the same encoder on the long prompt: the gated delta rule's 64-token chunks and the state carried between them give the reference's layer outputs and last_hidden_state",
        options: ["--variant"], variants: [[], ["--variant", "compact"]],
        oracle: "tools/oracle/moss-image.py text --long; text-witness --long; text-bf16 --long", weights: ["text encoder"],
        rule: "the rule against transformers in bf16, at the fp64 witness"
    ) { invocation in try judged(invocation, title: "text-long", suffix: "-long", script: "tools/oracle/moss-image.py text-bf16 --long") }

    /// The encoder's layers 0, 3, 15 and 31 and its final state, on the golden's identifiers.
    static func judged(_ invocation: Invocation, title: String, suffix: String, script: String) throws -> Report {
        let report = Report(title: "\(model) \(title)")
        // `--variant compact`: the Compact's encoder (int8 convrot, W8A8 in f16). The goldens have no Compact of their
        // own for the text: the 8-bit map is taken against the Standard's, said, its measures the quantization's error
        // and the path's together.
        let compact = try invocation.compact()
        let path = try MapCheck.use(Library.processWide.map(compact ? Family.mossImage.encoderMap(.compact) : mapName),
                                    golden: compact ? .eightBit : .standard)
        if compact { print("Compact: no reference of its own for the text, judged against the Standard's (bf16 weights): the int8 weights' error counts") }
        let goldens = try invocation.goldens.open("\(goldenStem)\(suffix).safetensors")
        let witness = invocation.goldens.optional("moss/witness-moss-text\(suffix)-f64.safetensors")
        let half = try invocation.goldens.half("moss/witness-bf16-moss-text\(suffix).safetensors", script: script)
        guard let ids = goldens.int32("ids") else { throw Skip("\(goldens.path): no ids") }
        let encoder = Qwen35Encoder()
        let width = encoder.config.width
        let taps: Set<Int> = [0, 3, 15, 31]
        let out = try encoder.states(ids.map(Int.init), map: path, taps: taps)
        for (name, key) in taps.sorted().map({ ("layer\($0)_out", "layer\($0)") }) + [("final", "hidden")] {
            guard let expected = goldens.float32(name), let got = out[key]?.floats else { report.add(Judge.missing(name)); continue }
            report.add(Judge.boundary(name, engine: got, oracle: expected, witness: witness?.float32(name),
                                      half: half.float32(name).map(Array.init), columns: width))
        }
        return report
    }

    // MARK: -

    /// The installed family's `tokenizer/` folder.
    static func tokenizerDirectory() throws -> String {
        do { return try Library.processWide.component(.mossImage, "tokenizer") } catch {
            throw Skip("tokenizer missing: siliconed install moss-image")
        }
    }
}
