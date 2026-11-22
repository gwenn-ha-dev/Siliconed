import Foundation
import Siliconed

// **The Qwen3 encoders' text stages**, called by each family's `text` stage: the encoder (`Qwen3Encoder`) run on the
// golden's own identifiers, one thing judged at a time, its layers judged by the rule beside the reference's bf16 run.

enum Qwen3TextStages {
    /// **Z-Image's**: Qwen3-4B's layer 33 and `cap_feats` (`hidden_states[-2]`, layer 34's output) on the real tokens.
    /// The Standard on the bf16 file against transformers in fp32 and bf16; `--variant compact` on Disty0's SDNQ int8
    /// against the same encoder dequantized, in fp32 and bf16 (`z-image.py text|text-bf16 --variant compact`).
    static func zImage(_ invocation: Invocation) throws -> Report {
        let compact = try invocation.compact()
        let report = Report(title: "z-image text" + (compact ? " compact" : ""))
        let path = try MapCheck.use(Library.processWide.map(Family.zImage.encoderMap(compact ? .compact : .standard)),
                                    golden: compact ? .eightBit : .standard)
        let goldens = try invocation.goldens.open("goldens-text.safetensors")
        let oracle = compact ? try invocation.goldens.open("goldens-text-compact.safetensors") : goldens
        let half = try invocation.goldens.half(compact ? "witness-bf16-text-compact.safetensors" : "witness-bf16-text.safetensors",
                                               script: "tools/oracle/z-image.py text-bf16" + (compact ? " --variant compact" : ""))
        guard let ids = goldens.int32("input_ids"), let mask = goldens.int32("attention_mask") else {
            throw Skip("\(goldens.path): no input_ids or attention_mask")
        }
        // Causal attention with right padding: the real tokens alone are computed, no real row reads a padded one.
        let real = (0..<mask.count).filter { mask[$0] != 0 }.map { Int(ids[$0]) }
        let encoder = Qwen3Encoder(config: .zImage, path: path)
        let width = encoder.config.width
        let started = Date()
        let states = try encoder.states(real, taps: [33])
        print(String(format: "  the encoder: %d tokens in %.2f s (weights read, functions loaded and run)", real.count, Date().timeIntervalSince(started)))
        // The Standard's goldens are padded (512 rows): the real ones only.
        let kept = real.count * width
        for (name, key) in [("layer33_out", "layer33"), ("cap_feats", "hidden")] {
            guard let expected = oracle.float32(name), let got = states[key]?.floats else { report.add(Judge.missing(name)); continue }
            report.add(Judge.boundary(name, engine: got, oracle: UnsafeBufferPointer(rebasing: expected.prefix(kept)),
                                      half: half.float32(name).map { Array($0.prefix(kept)) }, columns: width))
        }
        return report
    }

    /// **Anima's Qwen3-0.6B**: the two tokenizers against the golden's identifiers, then the encoder (every layer, its
    /// final norm: `last_hidden_state`) on the golden's own Qwen identifiers, layers 0, 13, 26 and the last hidden state
    /// into `report`. What the conditioner reads, returned: `[Lq, 1024]`.
    @discardableResult
    static func anima(_ invocation: Invocation, report: Report) throws -> HostTensor {
        let goldens = try invocation.goldens.open("anima/goldens-anima-text.safetensors")
        let half = try invocation.goldens.half("anima/witness-bf16-anima-text.safetensors", script: "tools/oracle/anima.py text-bf16")
        guard let prompt = try invocation.goldens.manifest("anima/goldens-anima-text")["prompt"] as? String,
              let qwenGolden = goldens.int32("qwen_ids"), let t5Golden = goldens.int32("t5_ids") else {
            throw Skip("\(goldens.path): no prompt or identifiers")
        }
        // The identifiers as the render reads them (uncut); the reference cuts at 512.
        let (qwenIds, t5Ids) = try AnimaRender.tokens(prompt, AnimaTextModule(
            map: "", tokenizerQwen: try Library.processWide.component(.anima, "tokenizer"),
            tokenizerT5: try Library.processWide.component(.anima, "t5_tokenizer"), adapter: ""))
        report.add(Judge.integers("Qwen tokens", qwenIds, qwenGolden))
        report.add(Judge.integers("T5 tokens", t5Ids, t5Golden))
        // The computation starts from the golden's identifiers: one thing judged at a time.
        let qwen = (0..<qwenGolden.count).map { Int(qwenGolden[$0]) }
        let path = try MapCheck.use(Library.processWide.map(Family.anima.encoderMap(.standard)))
        let encoder = Qwen3Encoder(config: .anima, path: path)
        let width = encoder.config.width, layers = [0, 13, 26]
        let states = try encoder.states(qwen, taps: Set(layers))
        for i in layers {
            report.add(goldens, "qwen layer \(i)", states["layer\(i)"]?.floats, name: "qwen_layer\(i)_out",
                       half: half.float32("qwen_layer\(i)_out").map(Array.init), columns: width)
        }
        report.add(goldens, "qwen last_hidden", states["hidden"]?.floats, name: "qwen_last_hidden",
                   half: half.float32("qwen_last_hidden").map(Array.init), columns: width)
        guard let hidden = states["hidden"] else { throw Skip("the encoder gave no hidden states") }
        return hidden
    }
}
