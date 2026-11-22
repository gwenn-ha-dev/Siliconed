import Foundation
import SiliCore

/// **Qwen3 as a text encoder** (Qwen3-4B, Qwen3-0.6B, Qwen3-VL's language model): the prompt's token identifiers in, the
/// hidden states after the last layer its configuration reads out. Transcribed from transformers' `modeling_qwen3.py`
/// (`Qwen3DecoderLayer`, `Qwen3Attention`, `Qwen3RMSNorm`, `Qwen3RotaryEmbedding`).
///
///     ids [L] ─ embed_tokens ─ layer 0 … layer last ─ hidden [L, d]          (hidden_states[-2]: last = 34)
///
///     layer:  h = x + o_proj( attn_causal( RoPE(RMS_h(q)), RoPE(RMS_h(k)), v ) )      on RMS(x) · γ₁
///             y = h + down( SiLU(gate n) ⊙ up n )                                    n = RMS(h) · γ₂
///
/// **On the engine** (`Qwen3Functions.swift`, `Qwen3Layers.swift`): one function serves every layer, its weights
/// arguments, on a palier of rows — the prompt padded to it: the attention is causal and every other op row by row, so
/// the rows past the prompt change none of its own, and only its own are read. The weights are the layers' groups, a
/// layer each, read once in order (`SiliCore.Layers`: kept where B holds them all, else through a ring). The embedding
/// table is read by the CPU from the file, a row a token; so are the picture rows, the deepstack additions, the taps and
/// the final norm — none of them is a product.
///
/// **Two arithmetics, by what the file holds** (`Qwen3Arithmetic`):
///   - **bf16 end to end** on a bf16 file, the GPU alone: transformers' own roundings, op by op (RMSNorm rounds the
///     normalized value, then the product by γ; RoPE's tables brought to bf16; SiLU computed in f32 and rounded once).
///   - **f16, W8A16** on int8 codes (SDNQ's, or ComfyUI's with its `convrot` rotation): the codes dequantized in the
///     function, the guard of f16 by its rules (`Guard.swift`) — the stream kept as (x̂, X), every product reading its
///     input under its own target, each result entering the stream with its factor (`residual(plus:factor:)`).
///
/// What a port by analogy would miss:
///
///   - **`hidden_states[-2]` is layer 34's output, not the normed last layer**: a configuration that reads it runs 35 of
///     Qwen3-4B's 36 layers and no norm after them. `last_hidden_state` is the other case: every layer and the model's
///     `norm` (`finalNorm`).
///   - **Grouped heads** (`repeat_kv`): head h reads key and value head h / (heads / kvHeads).
///   - **RMSNorm on q and k per head, before RoPE**; RoPE's angles in f32 from f32 inverse frequencies.
///   - **Qwen3-VL reading pictures**: the tower's merged rows replace the embedding at the picture's slots, its deepstack
///     rows are added there after layers 0, 1, 2…, and the positions are three (`Qwen3VLImages.rotaryTable`).
package struct Qwen3Encoder {
    package struct Config {
        package let width, heads, kvHeads, headDim, hidden, layers, vocabulary: Int
        package let theta, eps: Double
        /// The model's final `norm` applied after the last layer (`last_hidden_state`); without it the output is the last
        /// layer's raw one (`hidden_states[-2]` of a 36-layer file read to layer 34).
        package let finalNorm: Bool
        /// What the file puts before each name of the language model (`model.`, `model.language_model.`).
        package let prefix: String

        package init(width: Int, heads: Int, kvHeads: Int, headDim: Int, hidden: Int, layers: Int, vocabulary: Int = 151_936,
                     theta: Double = 1e6, eps: Double = 1e-6, finalNorm: Bool, prefix: String = "model.") {
            (self.width, self.heads, self.kvHeads, self.headDim, self.hidden) = (width, heads, kvHeads, headDim, hidden)
            (self.layers, self.vocabulary, self.theta, self.eps, self.finalNorm, self.prefix) = (layers, vocabulary, theta, eps, finalNorm, prefix)
        }

        /// q, k and v side by side, the fused product's width.
        var qkvWidth: Int { (heads + 2 * kvHeads) * headDim }
    }

    /// **The pictures a sequence reads** (Qwen3-VL): which rows are theirs, the tower's rows for them, and the
    /// three-axis RoPE table of the whole sequence.
    package struct Pictures {
        /// Each picture's rows in the sequence.
        package let runs: [Range<Int>]
        /// The tower's merged rows, `[slots, width]`, the pictures' one after the other.
        package let merged: [Float]
        /// Its deepstack rows, each `[slots, width]`: the j-th added after layer j.
        package let deepstack: [[Float]]
        /// `Qwen3VLImages.rotaryTable` of the sequence's positions, `[length, headDim]` each.
        package let cos: [Float], sin: [Float]

        package init(runs: [Range<Int>], merged: [Float], deepstack: [[Float]], cos: [Float], sin: [Float]) {
            (self.runs, self.merged, self.deepstack, self.cos, self.sin) = (runs, merged, deepstack, cos, sin)
        }
    }

    package let config: Config
    /// The file, or the index of its shards.
    package let path: String
    /// What the file puts before each name of the language model: the configuration's.
    package var prefix: String { config.prefix }

    package init(config: Config, path: String) {
        self.config = config
        self.path = path
    }

    /// **The encoder over `ids`**, every token real: `hidden` `[L, width]` (the last layer's output, normed when the
    /// configuration says so), `layer<i>` for each layer of `taps` (its output, before a deepstack addition), and
    /// `layer<i>_in` for each of `entries` (what layer i reads: the embedding for 0, a deepstack addition included).
    /// The values are those of the arithmetic's own output: bf16 values on a bf16 file, f16 times a row's factor on codes.
    package func states(_ ids: [Int], pictures: Pictures? = nil, taps: Set<Int> = [], entries: Set<Int> = []) throws -> [String: HostTensor] {
        try blocking { try await self.run(ids, pictures: pictures, taps: taps, entries: entries) }
    }

    /// `states`, from an `async` caller. What the stage hands back is made before it is planned, so that its floor counts
    /// it: made while it lived, Krea 2's twelve taps (130 MB at 1063 rows) were read as Core AI's share. The final norm is
    /// the CPU's, once the stage is let go.
    package func run(_ ids: [Int], pictures: Pictures? = nil, taps: Set<Int> = [], entries: Set<Int> = []) async throws -> [String: HostTensor] {
        let file = try Self.file(path)
        let a = try Qwen3Arithmetic(file, prefix: prefix)
        let width = config.width, n = ids.count
        let table = try table(file)
        let names = taps.map { "layer\($0)" } + entries.map { "layer\($0)_in" } + ["hidden"]
        var made = Dictionary(uniqueKeysWithValues: names.map { ($0, [Float](repeating: 0, count: n * width)) })
        do {
            let layers = try await open(file, a, rows: try Qwen3Encoder.palier(n))
            defer { layers.close() }
            try await embed(layers, a, ids, pictures: pictures, table: table, file: file)
            try await layers.forward(before: { i in
                if entries.contains(i) { await layers.values("x", rows: 0..<n, into: &made["layer\(i)_in"]!) }
            }, after: { i in
                if taps.contains(i) { await layers.values("x", rows: 0..<n, into: &made["layer\(i)"]!) }
                if let pictures, i < pictures.deepstack.count { await self.add(layers, a, pictures.deepstack[i], at: pictures.runs) }
            })
            await layers.values("x", rows: 0..<n, into: &made["hidden"]!)
        }
        var out = made.mapValues { HostTensor([n, width], floats: $0) }
        if config.finalNorm { out["hidden"] = HostTensor([n, width], floats: try finalNorm(out["hidden"]!.floats, rows: n, a, file: file)) }
        return out
    }

    /// The paliers a sequence is padded to: one function each, specialized once. Z's longest prompt (1024 tokens of its
    /// own and the template's) holds in 1088; an edit's pictures take up to a few thousand rows.
    static let paliers = [64, 128, 256, 512, 1088, 2048, 4096]

    /// The smallest palier that holds `n` rows.
    static func palier(_ n: Int) throws -> Int {
        guard let p = paliers.first(where: { $0 >= n }) else { throw ProgramError("\(n) tokens, past the longest sequence \(paliers.last!)") }
        return p
    }

    /// cos and sin `[rows, headDim]` at positions `start ..< start + rows`, as `Qwen3RotaryEmbedding` computes them in f32
    /// (`inv_freq = 1 / θ^(2i / d)` by an f32 `pow`, the angle an f32 product, `emb = cat(freqs, freqs)`), `cosSin`.
    func rope(rows: Int, start: Int = 0) -> (cos: [Float], sin: [Float]) {
        let d = config.headDim, half = d / 2
        let inverse = (0..<half).map { 1 / powf(Float(config.theta), Float(2 * $0) / Float(d)) }
        var c = [Float](repeating: 0, count: rows * d), s = c
        for r in 0..<rows {
            for i in 0..<d {
                let angle = Float(start + r) * inverse[i % half]
                (c[r * d + i], s[r * d + i]) = cosSin(Double(angle))
            }
        }
        return (c, s)
    }
}
