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
    static let all: [Stage] = [tokenizer, textDelta, textFull, text]

    static let model = MossImageStages.model
    static let mapName = Family.mossImage.encoderMap
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

    static let textDelta = Stage(
        model: model, name: "text-delta",
        proves: "the embedding rows and layer 0, a Gated DeltaNet layer, part by part: input norm, fused projection, delta rule, gated norm, mixer, MLP, output",
        options: ["--map"], oracle: "tools/oracle/moss-image.py text", weights: ["text encoder (layer 0)"],
        rule: "the rule, floor 5e-4"
    ) { invocation in
        try run(invocation, layers: 1, title: "text-delta")
    }

    static let textFull = Stage(
        model: model, name: "text-full",
        proves: "layers 0 to 3: three Gated DeltaNet layers and the first full-attention layer (gated queries, partial RoPE, output gate), part by part",
        options: ["--map"], oracle: "tools/oracle/moss-image.py text", weights: ["text encoder (layers 0–3)"],
        rule: "the rule, floor 5e-4"
    ) { invocation in
        try run(invocation, layers: 4, title: "text-full")
    }

    static let text = Stage(
        model: model, name: "text",
        proves: "the prompt through Qwen3.5-9B: all 32 layers' outputs and the final state, the DiT's `prompt_embeds`",
        options: ["--map"],
        oracle: "tools/oracle/moss-image.py text; text-witness", weights: ["text encoder"],
        rule: "the rule, floor 5e-4; the layers 0, 3, 15, 31 and the final state at the fp64 witness"
    ) { invocation in
        try run(invocation, layers: 32, title: "text")
    }

    // MARK: -

    /// The installed family's `tokenizer/` folder.
    static func tokenizerDirectory() throws -> String {
        do { return try Library.processWide.component(.mossImage, "tokenizer") } catch {
            throw Skip("tokenizer missing: siliconed install moss-image")
        }
    }

    static func run(_ invocation: Invocation, layers: Int, title: String) throws -> Report {
        let report = Report(title: "\(model) \(title)")
        let goldens = try invocation.goldens.open(goldenStem + ".safetensors")
        guard let ids32 = goldens.int32("ids") else { throw Skip("\(goldens.path): no ids") }
        let ids = ids32.map { Int($0) }
        let witness = layers == 32 ? invocation.goldens.optional("moss/witness-moss-text-f64.safetensors") : nil
        let mapPath = invocation.map ?? Library.processWide.map(mapName)
        guard FileManager.default.fileExists(atPath: mapPath) else { throw Skip("map missing: \(mapPath) — siliconed install moss-image") }
        try MapCheck.use(mapPath)

        // The prompt side first: the oracle's identifiers are the template's, through our tokenizer.
        if let directory = try? tokenizerDirectory(), let tokenizer = try? Tokenizer(directory: directory),
           let prompt = try invocation.goldens.manifest(goldenStem)["prompt"] as? String {
            report.add(Judge.integers("ids", Qwen35Prompt.ids(prompt, tokenizer: tokenizer), ids32))
        }

        let encoder = try Qwen35Encoder(artifact: try Artifact(path: mapPath), sequence: ids.count)
        var names: Set<String> = ["embed_out"]
        for l in 0..<layers {
            names.insert("layer\(l)_out")
            names.insert("layer\(l)_mixer")
        }
        names.formUnion(["layer0_input_norm", "layer0_qkv", "layer0_core", "layer0_gated",
                         "layer3_input_norm", "layer3_q_proj", "layer3_attn_gated"])
        encoder.recordedNames = names
        let started = Date()
        let hidden = try encoder.encode(ids: ids, stopAfter: layers == 32 ? nil : layers)
        if invocation.profile { print(String(format: "encoder %.1f s for %d tokens", Date().timeIntervalSince(started), ids.count)) }

        let h = encoder.config.hidden, b = encoder.boundaries
        report.add(goldens, "embed", b["embed_out"], columns: h)
        report.add(goldens, "layer0 input norm", b["layer0_input_norm"], name: "layer0_input_norm", columns: h)
        report.add(goldens, "layer0 in_proj_qkv", b["layer0_qkv"], name: "layer0_in_proj_qkv", columns: encoder.config.convChannels)
        report.add(goldens, "layer0 delta core", b["layer0_core"], name: "layer0_delta_core", columns: encoder.config.valueHeadDim)
        report.add(goldens, "layer0 delta gated norm", b["layer0_gated"], name: "layer0_delta_normed", columns: encoder.config.valueHeadDim)
        report.add(goldens, "layer0 delta mixer", b["layer0_mixer"], name: "layer0_attn_out", columns: h)
        if layers >= 4 {
            report.add(goldens, "layer3 input norm", b["layer3_input_norm"], name: "layer3_input_norm", columns: h)
            report.add(goldens, "layer3 q_proj", b["layer3_q_proj"], name: "layer3_q_proj", columns: 2 * encoder.config.attentionWidth)
            report.add(goldens, "layer3 gated attention", b["layer3_attn_gated"], name: "layer3_attn_gated", columns: encoder.config.attentionWidth)
            report.add(goldens, "layer3 attention mixer", b["layer3_mixer"], name: "layer3_attn_out", columns: h)
        }
        let witnessed: Set<Int> = [0, 3, 15, 31]
        for l in 0..<layers {
            let name = "layer\(l)_out"
            let exact = witnessed.contains(l) ? witness?.float32(name) : nil
            report.add(goldens, "layer \(l) out\(encoder.config.fullLayer[l] ? " (full)" : "")", b[name], name: name, witness: exact, columns: h)
        }
        if layers == 32 {
            report.add(goldens, "last_hidden_state", hidden, name: "final", witness: witness?.float32("final"), columns: h)
        }
        return report
    }
}
