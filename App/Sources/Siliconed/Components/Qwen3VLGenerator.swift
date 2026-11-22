import CoreAI
import Foundation
import SiliCore

/// **Qwen3-VL that writes**: greedy decoding with a K/V cache — `Qwen3VLForConditionalGeneration.generate` with
/// `do_sample=False`, from the encoder's published file, in the encoder's arithmetic (bf16, or the Compact's int8
/// weights), the head's sum in f32.
///
///     pictures ─ vision tower (a session a picture) ─ merged rows + 3 deepstack
///     prompt but its last token ─ prefill: every layer once, its keys and values to the caches' first rows
///     the last token ─ decode, a token a run until a stop token or the limit:
///         the token's row through every layer over the cache, in place ─ norm ─ lm_head ─ argmax ─ the next token
///
/// What a port by analogy would miss:
///
///   - **The prompt's last token is the decode's first**: the prefill needs no head, and every token written comes out
///     of the same program.
///   - **A token written sits at `max(prompt positions) + 1 + k` on the three axes** (`rope_deltas`): after a picture
///     the positions run short of the cache's rows; the decode is told the difference (`shift`).
///   - **Two stop tokens** (`generation_config.eos_token_id`: `<|im_end|>`, `<|endoftext|>`): the engine's flag is the
///     first, the client's word ends on either. The stop token is kept in the output, as `generate` keeps it.
///   - **Greedy**, whatever the published `generation_config.json` samples with: ties to the lowest identifier.
///   - **The caches' room is rounded up** (256 rows): one decode asset serves every prompt of a size.
package final class Qwen3VLGenerator {
    /// The sequence to continue: its identifiers (each picture's `<|image_pad|>` repeated over its slots), which rows
    /// are a picture's, and the pictures in slot order.
    package struct Prompt {
        package let ids: [Int]
        package let slots: [Bool]
        package let pictures: [Qwen3VLImages.Pixels]
        package init(ids: [Int], slots: [Bool], pictures: [Qwen3VLImages.Pixels]) {
            (self.ids, self.slots, self.pictures) = (ids, slots, pictures)
        }
    }

    package struct Output {
        /// The tokens written, the stop token included when one was reached.
        package let tokens: [Int]
        /// Whether a stop token ended it (otherwise the limit, or the client's word).
        package let stopped: Bool
    }

    /// The seconds of the last `generate`: the pictures, the preparation (the weights read, the functions loaded), the
    /// run (prefill and tokens).
    package private(set) var timings: (vision: Double, preparation: Double, run: Double) = (0, 0, 0)

    let path: String
    let config: Qwen3Encoder.Config
    let tower: Qwen3VLVision.Config
    let encoder: Qwen3Encoder
    private let file: SafeTensors

    package static let room = 256

    package init(map path: String, config: Qwen3Encoder.Config, tower: Qwen3VLVision.Config = .eightB) throws {
        self.path = path
        (self.config, self.tower) = (config, tower)
        encoder = Qwen3Encoder(config: config, path: path)
        file = try Qwen3Encoder.file(path)
    }

    /// Whether the file holds what writing reads beyond the encoder: the final norm and the head.
    package var writes: Bool { file.tensors["lm_head.weight"] != nil && file.tensors[encoder.prefix + "norm.weight"] != nil }

    /// **Continues `prompt`** by at most `limit` tokens. `each` is handed every token as it is written and ends the
    /// run by answering `false` (a cancellation, a loop seen); a stop token ends it too. Computed exactly, whatever the
    /// Neural Engine switch: two tokens a rounding apart are two texts.
    package func generate(_ prompt: Prompt, limit: Int, stops: [Int], each: @escaping (Int) -> Bool = { _ in true }) throws -> Output {
        let started = Date()
        let pictures = try towerRows(prompt)
        timings.vision = Date().timeIntervalSince(started)
        return try blocking { try await self.decoding(prompt, pictures: pictures, limit: limit, stops: stops, each: each) }
    }

    /// **The decoding**: every layer read at every token (the stage keeps all of them where B holds them, else reads
    /// those it cannot keep again through its ring, in the loop's order), the prompt but its last token read once by the
    /// prefill — its keys and values to the cache's first rows —, then a token a pass: its row through every layer over
    /// the cache (`layerModule(.cached)`: the cache's rows and the token's own, those past it masked), its keys and values
    /// written to the cache's row, the head, the argmax. A token's calls go on the stream unwaited, one wait at the
    /// head's end: the next token's row is the CPU's.
    func decoding(_ prompt: Prompt, pictures: [String: [Float]], limit: Int, stops: [Int], each: @escaping (Int) -> Bool) async throws -> Output {
        guard let last = prompt.ids.last, prompt.ids.count > 1, limit > 0, !stops.isEmpty else {
            throw ProgramError("nothing to continue: a prompt of two tokens, a limit and a stop token at least")
        }
        var started = Date()

        // The prefill reads every token but the last; a picture's slots are never last (the template ends in text).
        let length = prompt.ids.count - 1, c = config
        let read = try Self.sequence(slots: Array(prompt.slots.dropLast()), pictures: prompt.pictures, rows: pictures, config: c, tower: tower)
        let positions = try Qwen3VLImages.rotaryPositions(slots: prompt.slots, grids: prompt.pictures.map { (height: $0.gridHeight, width: $0.gridWidth) },
                                                         merge: tower.merge)
        let rows = try Qwen3Encoder.palier(length)
        let capacity = max(rows, (length + limit + Self.room - 1) / Self.room * Self.room)
        let a = try Qwen3Arithmetic(file, prefix: encoder.prefix)
        // The row of logits and the embedding table made before the stage, so that its floor counts them (`Qwen3Head.read`).
        var logits = [Float](repeating: 0, count: c.vocabulary)
        let table = try encoder.table(file)
        let shape = Qwen3Head.Shape(rows: 1, input: false), model = encoder
        let layers = try await encoder.open(file, a, rows: rows, writing: .init(capacity: capacity), head: shape)
        defer { layers.close() }
        let head = try await Qwen3Head(shape, encoder, a, layers: layers, file: file)
        timings.preparation = Date().timeIntervalSince(started)
        started = Date()

        try await encoder.embed(layers, a, Array(prompt.ids.dropLast()), pictures: read, table: table, file: file)
        try await layers.forward(after: { i in
            if let read, i < read.deepstack.count { await model.add(layers, a, read.deepstack[i], at: read.runs) }
        })

        // A text token's position is the same on the three axes; after a picture it runs short of its row (`rope_deltas`).
        let shift = positions[length].0 - length
        var masked = [Float](repeating: -1e4, count: capacity + 1)
        masked[capacity] = 0
        for r in 0..<length { masked[r] = 0 }
        var tokens: [Int] = [], stopped = false, next = last
        let ends = Set(stops)
        for step in 0..<limit {
            let row = length + step
            try await encoder.token(layers, a, next, position: row + shift, masked: masked, table: table, file: file)
            try await head.run(layers, values: try await encoder.decode(layers, a, row: row))
            await head.read(layers, rows: 1, into: &logits)
            next = Qwen3Head.argmax(logits[...])
            tokens.append(next)
            masked[row] = 0
            if ends.contains(next) { stopped = true; break }
            if !each(next) { break }
        }
        timings.run = Date().timeIntervalSince(started)
        return Output(tokens: tokens, stopped: stopped)
    }

    /// **The whole forward over `prompt` and `answer`, teacher-forced**: what follows each of the rows that precede
    /// `answer`'s tokens — `logits` `[answer.count, vocabulary]` and their argmax. What `generate` must write, and
    /// what a reference's logits are judged on.
    package func continuation(_ prompt: Prompt, answer: [Int]) throws -> (logits: [Float], next: [Int]) {
        let pictures = try towerRows(prompt)
        return try blocking { try await self.forced(prompt, pictures: pictures, answer: answer) }
    }

    func forced(_ prompt: Prompt, pictures: [String: [Float]], answer: [Int]) async throws -> (logits: [Float], next: [Int]) {
        let ids = prompt.ids + answer.dropLast(), c = config, n = answer.count
        let read = try Self.sequence(slots: prompt.slots + [Bool](repeating: false, count: answer.count - 1), pictures: prompt.pictures,
                                     rows: pictures, config: c, tower: tower)
        let a = try Qwen3Arithmetic(file, prefix: encoder.prefix)
        let rows = try Qwen3Encoder.palier(n)
        // The logits and the embedding table made before the stage, so that its floor counts them (`Qwen3Head.read`).
        var logits = [Float](repeating: 0, count: n * c.vocabulary)
        let table = try encoder.table(file)
        let shape = Qwen3Head.Shape(rows: rows, input: true), model = encoder
        let layers = try await encoder.open(file, a, rows: try Qwen3Encoder.palier(ids.count), head: shape)
        defer { layers.close() }
        let head = try await Qwen3Head(shape, encoder, a, layers: layers, file: file)
        try await encoder.embed(layers, a, ids, pictures: read, table: table, file: file)
        try await layers.forward(after: { i in
            if let read, i < read.deepstack.count { await model.add(layers, a, read.deepstack[i], at: read.runs) }
        })
        let last = await layers.values("x", rows: (ids.count - n)..<ids.count)
        let factors = a.half ? await layers.pointer("head_xf").assumingMemoryBound(to: Float.self) : nil
        Qwen3Encoder.write(last, rows: 0..<n, width: c.width, half: a.half, to: await layers.pointer("head_x"), factors: factors)
        try await head.run(layers, bind: head.input)
        await head.read(layers, rows: n, into: &logits)
        let next = (0..<n).map { Qwen3Head.argmax(logits[($0 * c.vocabulary)..<(($0 + 1) * c.vocabulary)]) }
        return (logits, next)
    }

    /// The vision tower's rows for the prompt's pictures, `merged` and `deepstack<j>`: its own stage, done before the
    /// language model's arena is taken.
    private func towerRows(_ prompt: Prompt) throws -> [String: [Float]] {
        try Qwen3VLVision.rows(prompt.pictures, config: tower, map: path)
    }

    /// **What the language model reads of a sequence's pictures**: each picture's rows, the tower's rows, the three-axis
    /// RoPE table; nil without a picture.
    package static func sequence(slots: [Bool], pictures: [Qwen3VLImages.Pixels], rows: [String: [Float]],
                                 config: Qwen3Encoder.Config, tower: Qwen3VLVision.Config) throws -> Qwen3Encoder.Pictures? {
        guard !pictures.isEmpty else { return nil }
        let grids = pictures.map { (height: $0.gridHeight, width: $0.gridWidth) }
        let table = Qwen3VLImages.rotaryTable(try Qwen3VLImages.rotaryPositions(slots: slots, grids: grids, merge: tower.merge),
                                              headDim: config.headDim, theta: config.theta)
        var runs: [Range<Int>] = [], at = 0
        for g in grids {
            guard let start = slots[at...].firstIndex(of: true) else { throw ProgramError("a picture without its slots") }
            at = start + g.height * g.width / (tower.merge * tower.merge)
            runs.append(start..<at)
        }
        return Qwen3Encoder.Pictures(runs: runs, merged: rows["merged"] ?? [],
                                     deepstack: (0..<tower.deepstack.count).map { rows["deepstack\($0)"] ?? [] }, cos: table.cos, sin: table.sin)
    }
}

/// **An incremental UTF-8 decoder**: bytes in, the text they complete out. A byte-level BPE token can end in
/// the middle of a character (an accent is two bytes, a kanji three, an emoji four, and the tokens cut
/// where the merges fell), so the bytes of an incomplete tail wait for the next token; `finish` turns a tail
/// that never completed into U+FFFD, as `String(decoding:)` (`errors="replace"`) does. Everything else is
/// decoded by `String(decoding:)` itself, so the concatenation of what `push` and `finish` return is
/// that call's result on all the bytes, whatever the cuts.
package struct UTF8Stream {
    private var pending: [UInt8] = []

    package init() {}

    /// The text completed by `bytes`; an incomplete character at their end is kept for the next call.
    package mutating func push(_ bytes: [UInt8]) -> String {
        pending += bytes
        let keep = UTF8Stream.incompleteTail(pending)
        let head = pending.count - keep
        guard head > 0 else { return "" }
        let text = String(decoding: pending[0..<head], as: UTF8.self)
        pending.removeFirst(head)
        return text
    }

    /// What is left: a character cut short, replaced.
    package mutating func finish() -> String {
        defer { pending = [] }
        return pending.isEmpty ? "" : String(decoding: pending, as: UTF8.self)
    }

    /// The length of the longest suffix that is a proper prefix of a well-formed UTF-8 sequence (Unicode
    /// Table 3-7: `E0` wants `A0…BF` next, `ED` `80…9F`, `F0` `90…BF`, `F4` `80…8F`).
    package static func incompleteTail(_ bytes: [UInt8]) -> Int {
        guard !bytes.isEmpty else { return 0 }
        for back in 1...min(3, bytes.count) {
            let lead = bytes[bytes.count - back]
            let needed: Int
            switch lead {
            case 0xC2...0xDF: needed = 2
            case 0xE0...0xEF: needed = 3
            case 0xF0...0xF4: needed = 4
            case 0x80...0xBF: continue          // a continuation byte: look further back for its lead
            default: return 0                   // ASCII, or a byte no sequence starts with
            }
            guard back < needed else { return 0 }   // complete (or longer than needed: invalid, decoded now)
            let tail = bytes[(bytes.count - back + 1)...]
            for (offset, byte) in tail.enumerated() {
                var range: ClosedRange<UInt8> = 0x80...0xBF
                if offset == 0 {
                    switch lead {
                    case 0xE0: range = 0xA0...0xBF
                    case 0xED: range = 0x80...0x9F
                    case 0xF0: range = 0x90...0xBF
                    case 0xF4: range = 0x80...0x8F
                    default: break
                    }
                }
                guard range.contains(byte) else { return 0 }
            }
            return back
        }
        return 0
    }
}
