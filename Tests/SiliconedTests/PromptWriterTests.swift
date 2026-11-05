import XCTest
@testable import Siliconed

/// What the lab's writer computes without a byte of weights: the template's identifiers (with and without a
/// system turn), the incremental detokenization, the tie rule, the head map's forge, the resident tensors, the
/// memory need on the real shapes and the availability on 16 and 8 GB. The tokens themselves are checked
/// against transformers (W0, with the installed maps).
final class PromptWriterTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-writer-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDown() { try? FileManager.default.removeItem(at: root) }

    // ── a synthetic byte-level tokenizer: id = byte, then the control tokens, then one merge ──────

    private static let endOfText = 256, start = 257, end = 258, visionStart = 259, visionEnd = 260, imagePad = 261, acute = 262

    /// The 256 bytes are ids 0…255 (so a text's identifiers are its UTF-8 bytes), `é`'s two bytes merge into 262,
    /// every character is its own piece: nothing here depends on the real vocabulary, only on its structure.
    private func tokenizer() throws -> Tokenizer {
        var vocab: [String: Int] = [:]
        for (byte, symbol) in Tokenizer.byteSymbols.enumerated() { vocab[symbol] = byte }
        let e = Tokenizer.byteSymbols[0xC3], a = Tokenizer.byteSymbols[0xA9]
        vocab[e + a] = Self.acute
        func added(_ id: Int, _ content: String) -> [String: Any] {
            ["id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true]
        }
        let object: [String: Any] = [
            "version": "1.0",
            "added_tokens": [added(Self.endOfText, "<|endoftext|>"), added(Self.start, "<|im_start|>"), added(Self.end, "<|im_end|>"),
                             added(Self.visionStart, "<|vision_start|>"), added(Self.visionEnd, "<|vision_end|>"),
                             added(Self.imagePad, "<|image_pad|>")],
            "normalizer": NSNull(),
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": [
                ["type": "Split", "pattern": ["Regex": "(?s:.)"], "behavior": "Isolated", "invert": false],
                ["type": "ByteLevel", "add_prefix_space": false, "use_regex": false]]],
            "post_processor": ["type": "ByteLevel", "add_prefix_space": false],
            "decoder": ["type": "ByteLevel", "add_prefix_space": false],
            "model": ["type": "BPE", "dropout": NSNull(), "unk_token": NSNull(), "continuing_subword_prefix": "",
                      "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "vocab": vocab, "merges": [[e, a]]],
        ]
        let path = root.appendingPathComponent("tokenizer.json").path
        try JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes]).write(to: URL(fileURLWithPath: path))
        return try Tokenizer(file: path)
    }

    private func bytes(_ text: String) -> [Int] { text.utf8.map(Int.init) }
    private var tokens: Qwen3VLPrompt.Tokens { .init(image: Self.imagePad, visionStart: Self.visionStart, visionEnd: Self.visionEnd) }

    // ── the template ───────────────────────────────────────────────────────────────────────────

    /// The Instruct's `apply_chat_template`, as text: a system turn exactly when one is given (and then the
    /// turn is closed by `<|im_end|>\n`), the image slot with nothing between it and the text, the generation prompt.
    func testTheTemplateWithAndWithoutASystemTurn() {
        XCTAssertEqual(Qwen3VLChat.template(system: nil, user: "Hi", images: 0),
                       "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
        XCTAssertEqual(Qwen3VLChat.template(system: "Be brief.", user: "Hi", images: 0),
                       "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
        XCTAssertEqual(Qwen3VLChat.template(system: "S", user: "", images: 1),
                       "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n")
        XCTAssertEqual(Qwen3VLChat.template(instruction: "x", images: 1), Qwen3VLChat.template(system: nil, user: "x", images: 1))
        XCTAssertNil(PromptWriter.system(" \n"))
        XCTAssertEqual(PromptWriter.system("Rewrite."), "Rewrite.")
    }

    /// The identifiers: control tokens are single ids, the rest is the text's bytes, each `<|image_pad|>`
    /// is expanded to the image's merged tokens (a 4 × 4 grid of patches is 4).
    func testTheTemplateIdentifiers() throws {
        let t = try tokenizer()
        let plain = try Qwen3VLChat.prepare(user: "hi", grids: [], tokenizer: t, tokens: tokens)
        XCTAssertEqual(plain.ids, [Self.start] + bytes("user\nhi") + [Self.end] + bytes("\n") + [Self.start] + bytes("assistant\n"))
        let system = try Qwen3VLChat.prepare(system: "S", user: "hi", grids: [(4, 4)], tokenizer: t, tokens: tokens)
        let expected = [Self.start] + bytes("system\nS") + [Self.end] + bytes("\n") + [Self.start] + bytes("user\n")
            + [Self.visionStart] + Array(repeating: Self.imagePad, count: 4) + [Self.visionEnd] + bytes("hi")
            + [Self.end] + bytes("\n") + [Self.start] + bytes("assistant\n")
        XCTAssertEqual(system.ids, expected)
        XCTAssertEqual(system.positions.count, 3)
        XCTAssertTrue(system.positions.allSatisfy { $0.count == expected.count })
        // After a 2 × 2 merged grid the text resumes 2 later (not 4): the first generated token is at max + 1.
        let before = expected.firstIndex(of: Self.imagePad)!
        let after = expected.count - (before + 4)
        XCTAssertEqual(system.positions[0][before + 4], before + 2)
        XCTAssertEqual(Qwen3VLChat.firstGeneratedPosition(system.positions), before + 2 + after)
        XCTAssertLessThan(Qwen3VLChat.firstGeneratedPosition(system.positions), expected.count)
        for text in ["<|im_end|>", "<|image_pad|>", "<|endoftext|>"] {
            XCTAssertThrowsError(try Qwen3VLChat.prepare(user: "a" + text, grids: [], tokenizer: t, tokens: tokens))
            XCTAssertThrowsError(try Qwen3VLChat.prepare(system: text, user: "a", grids: [], tokenizer: t, tokens: tokens))
        }
    }

    // ── incremental detokenization ─────────────────────────────────────────────────────────────

    /// A stream cut anywhere gives what `String(decoding:)` gives on the whole: French accents (2 bytes),
    /// Japanese (3), emoji (4) and a ZWJ family (4 × 3 + 2 joiners), cut byte by byte and at every pair of cuts —
    /// and never a broken character before the end.
    func testTheStreamNeverBreaksACharacter() {
        let text = "Café déjà vu — 日本語のプロンプト 😀 👨‍👩‍👧 ça va ?"
        let all = Array(text.utf8)
        func stream(_ cuts: [Int]) -> String {
            var s = UTF8Stream(), out = ""
            for (a, b) in zip([0] + cuts, cuts + [all.count]) {
                let piece = s.push(Array(all[a..<b]))
                XCTAssertFalse(piece.contains("\u{FFFD}"), "a broken character after \(cuts)")
                out += piece
            }
            return out + s.finish()
        }
        XCTAssertEqual(stream(Array(1..<all.count)), text)
        for i in 1..<all.count { XCTAssertEqual(stream([i]), text) }
        for i in stride(from: 1, to: all.count, by: 3) { for j in stride(from: i + 1, to: all.count, by: 5) { XCTAssertEqual(stream([i, j]), text) } }
        // The emoji arrives in four pieces: nothing until the fourth.
        var s = UTF8Stream()
        let smile = Array("😀".utf8)
        XCTAssertEqual([s.push([smile[0]]), s.push([smile[1]]), s.push([smile[2]])], ["", "", ""])
        XCTAssertEqual(s.push([smile[3]]), "😀")
    }

    /// What never completes is replaced as `errors="replace"` does, at the end; an invalid byte is not
    /// held back (a lone continuation, a lead followed by a non-continuation, an overlong lead).
    func testTheStreamReplacesWhatCannotComplete() {
        var s = UTF8Stream()
        XCTAssertEqual(s.push([0xE6, 0x97]), "")
        XCTAssertEqual(s.finish(), String(decoding: [0xE6, 0x97], as: UTF8.self))
        for invalid: [UInt8] in [[0x80], [0xE6, 0x41], [0xC0, 0xAF], [0xF5, 0x80], [0xED, 0xA0, 0x80], [0xF0, 0x80, 0x80, 0x80], [0xFF]] {
            var s = UTF8Stream()
            XCTAssertEqual(s.push(invalid) + s.finish(), String(decoding: invalid, as: UTF8.self), "\(invalid)")
        }
        // Table 3-7's second-byte ranges: E0 A0…BF, ED 80…9F, F0 90…BF, F4 80…8F.
        XCTAssertEqual(UTF8Stream.incompleteTail([0xE0, 0xA0]), 2)
        XCTAssertEqual(UTF8Stream.incompleteTail([0xE0, 0x9F]), 0)
        XCTAssertEqual(UTF8Stream.incompleteTail([0xED, 0x9F]), 2)
        XCTAssertEqual(UTF8Stream.incompleteTail([0xF4, 0x90]), 0)
        XCTAssertEqual(UTF8Stream.incompleteTail([0x61, 0xF0]), 1)
        XCTAssertEqual(UTF8Stream.incompleteTail(Array("é".utf8)), 0)
    }

    /// Token by token through the tokenizer's byte table: an accent merged into one token, an emoji spread over
    /// four, the control tokens left out of the text — and the whole equals `decode`.
    func testTheTokenBytesFeedTheStream() throws {
        let t = try tokenizer()
        let table = t.tokenBytes(rows: 300)
        XCTAssertEqual(Array(table.bytes(of: Self.acute)), [0xC3, 0xA9])
        XCTAssertEqual(Array(table.bytes(of: 0x41)), [0x41])
        XCTAssertEqual(String(decoding: table.bytes(of: Self.end), as: UTF8.self), "<|im_end|>")
        XCTAssertTrue(table.bytes(of: 299).isEmpty && table.bytes(of: 5000).isEmpty)
        let text = "Café 😀 日本"
        let ids = t.encode(text) + [Self.end]
        XCTAssertTrue(ids.contains(Self.acute))
        var s = UTF8Stream(), out = ""
        for id in ids where !t.specialIdentifiers.contains(id) {
            let piece = s.push(Array(table.bytes(of: id)))
            XCTAssertFalse(piece.contains("\u{FFFD}"))
            out += piece
        }
        out += s.finish()
        XCTAssertEqual(out, text)
        XCTAssertEqual(Qwen3VLChat.text(ids, tokenizer: t), text)
        XCTAssertEqual(t.specialIdentifiers, [Self.endOfText, Self.start, Self.end, Self.visionStart, Self.visionEnd, Self.imagePad])
        // What the need charges for the table bounds what it maps.
        XCTAssertLessThanOrEqual(table.mapped, try Tokenizer.detokenizationBytes(file: root.appendingPathComponent("tokenizer.json").path, rows: 300))
    }

    // ── the tie rule ───────────────────────────────────────────────────────────────────────────

    /// The first maximum wins across blocks, as `torch.argmax`; a later equal value is only the runner-up.
    func testTheTieGoesToTheLowestIdentifier() {
        var top = Qwen3VLGenerator.Top2()
        [1, 5, 3].withUnsafeBufferPointer { top.scan($0.baseAddress!, count: 3, base: 0) }
        [5, 4, 2].withUnsafeBufferPointer { top.scan($0.baseAddress!, count: 3, base: 3) }
        XCTAssertEqual(top.id, 1)
        XCTAssertEqual(top.value - top.second, 0)
        var clear = Qwen3VLGenerator.Top2()
        [0, 2, 7, 6].withUnsafeBufferPointer { clear.scan($0.baseAddress!, count: 4, base: 10) }
        XCTAssertEqual(clear.id, 12)
        XCTAssertEqual(clear.value - clear.second, 1)
    }

    // ── cached attention (the part of a layer the encoder does not have) ─────────────────────────

    private func values(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: state >> 33)) / Float(Int32.max)
        }
    }

    private func attend(q: [Float], keys: [Float], values v: [Float], rows: Int, offset: Int, heads: Int, kvHeads: Int, d: Int) -> [Float] {
        var out = [Float](repeating: .nan, count: rows * heads * d)
        let floats = 4 * Qwen3VLGenerator.attentionBlock * (offset + rows)
        let scratch = UnsafeMutablePointer<Float>.allocate(capacity: floats)
        defer { scratch.deallocate() }
        q.withUnsafeBufferPointer { qp in keys.withUnsafeBufferPointer { kp in v.withUnsafeBufferPointer { vp in
            out.withUnsafeMutableBufferPointer { op in
                Qwen3VLGenerator.cachedAttention(q: qp.baseAddress!, rows: rows, keys: kp.baseAddress!, values: vp.baseAddress!,
                                                 offset: offset, heads: heads, kvHeads: kvHeads, headDim: d,
                                                 into: op.baseAddress!, scratch: scratch, scratchFloats: floats)
            }
        } } }
        return out
    }

    /// Causal GQA attention in Double, from scratch: query row `r` at `offset + r`, one key head per `heads / kvHeads` queries.
    func testCachedAttentionIsCausalFromItsOffset() {
        let (heads, kvHeads, d, offset, rows) = (4, 2, 8, 70, 3)
        let length = offset + rows
        let q = values(rows * heads * d, seed: 1), k = values(length * kvHeads * d, seed: 2), v = values(length * kvHeads * d, seed: 3)
        let ours = attend(q: q, keys: k, values: v, rows: rows, offset: offset, heads: heads, kvHeads: kvHeads, d: d)
        var exact = [Double](repeating: 0, count: rows * heads * d)
        for r in 0..<rows { for h in 0..<heads {
            let g = h / (heads / kvHeads), valid = offset + r + 1
            var s = (0..<valid).map { j in
                (0..<d).reduce(0.0) { $0 + Double(q[(r * heads + h) * d + $1]) * Double(k[(j * kvHeads + g) * d + $1]) } / Double(d).squareRoot()
            }
            let top = s.max()!
            s = s.map { exp($0 - top) }
            let sum = s.reduce(0, +)
            for j in 0..<valid { for c in 0..<d { exact[(r * heads + h) * d + c] += s[j] / sum * Double(v[(j * kvHeads + g) * d + c]) } }
        } }
        XCTAssertLessThan(zip(ours, exact).map { abs(Double($0) - $1) }.max()!, 1e-5)
    }

    /// The cache is the whole computation: a prompt cut in two passes (a prefill chunk, then a step) gives a single pass's rows.
    func testAChunkedPrefillThenAStepIsOnePass() {
        let (heads, kvHeads, d, length) = (4, 1, 8, 101)
        let q = values(length * heads * d, seed: 4), k = values(length * kvHeads * d, seed: 5), v = values(length * kvHeads * d, seed: 6)
        let whole = attend(q: q, keys: k, values: v, rows: length, offset: 0, heads: heads, kvHeads: kvHeads, d: d)
        let first = attend(q: Array(q[0..<(60 * heads * d)]), keys: k, values: v, rows: 60, offset: 0, heads: heads, kvHeads: kvHeads, d: d)
        let second = attend(q: Array(q[(60 * heads * d)..<(100 * heads * d)]), keys: k, values: v, rows: 40, offset: 60,
                            heads: heads, kvHeads: kvHeads, d: d)
        let step = attend(q: Array(q[(100 * heads * d)...]), keys: k, values: v, rows: 1, offset: 100, heads: heads, kvHeads: kvHeads, d: d)
        XCTAssertLessThan(zip(first + second + step, whole).map { abs($0 - $1) }.max()!, 1e-6)
    }

    // ── the head map ───────────────────────────────────────────────────────────────────────────

    private func bf16(_ count: Int, seed: UInt16) -> [UInt8] {
        // Typed step by step: Swift 6.3's checker gives up on mixed integer literals in one closure (the CI's).
        let mask: UInt16 = 0x3C00 & 0x3FFF
        return (0..<count).flatMap { (i: Int) -> [UInt8] in
            let mixed: Int = Int(seed) &* 31 &+ i &* 7
            let v: UInt16 = UInt16(truncatingIfNeeded: mixed) | mask
            return [UInt8(v & 255), UInt8(v >> 8)]
        }
    }

    /// A one-file safetensors: `[(name, shape, bytes)]`.
    private func safetensors(_ tensors: [(String, [Int], [UInt8])], to path: String) throws {
        var header: [String: Any] = [:], at = 0, data: [UInt8] = []
        for (name, shape, bytes) in tensors {
            header[name] = ["dtype": "BF16", "shape": shape, "data_offsets": [at, at + bytes.count]]
            at += bytes.count; data += bytes
        }
        var json = [UInt8](try JSONSerialization.data(withJSONObject: header))
        while json.count % 8 != 0 { json.append(0x20) }
        var out = Data()
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { out.append(contentsOf: $0) }
        out.append(contentsOf: json); out.append(contentsOf: data)
        try out.write(to: URL(fileURLWithPath: path))
    }

    private func checkpoint(tied: Bool = false) throws -> (folder: String, norm: [UInt8], head: [UInt8]) {
        let folder = root.appendingPathComponent("instruct-\(UUID())").path
        try FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)
        let norm = bf16(8, seed: 3), head = bf16(6 * 8, seed: 5)
        try safetensors([("model.language_model.layers.0.input_layernorm.weight", [8], bf16(8, seed: 1)),
                         ("lm_head.weight", [6, 8], head), ("model.language_model.norm.weight", [8], norm),
                         ("model.visual.pos_embed.weight", [2, 8], bf16(16, seed: 9))],
                        to: folder + "/model-00004-of-00004.safetensors")
        try Data(#"{"tie_word_embeddings": \#(tied), "text_config": {"hidden_size": 8, "vocab_size": 6, "num_hidden_layers": 1}}"#.utf8)
            .write(to: URL(fileURLWithPath: folder + "/config.json"))
        try Data(#"{"eos_token_id": [151645, 151643], "do_sample": true, "top_k": 20}"#.utf8)
            .write(to: URL(fileURLWithPath: folder + "/generation_config.json"))
        return (folder, norm, head)
    }

    /// **The head map**: the final norm and `lm_head`, as published (bf16, not transposed, no rounding), the stop tokens
    /// of `generation_config.json`, nothing else of the shard (layer 35 and the vision tower are not copied); a tied
    /// head is refused (it would be `embed_tokens`).
    func testTheHeadMapKeepsThePublishedBits() throws {
        let c = try checkpoint()
        let out = root.appendingPathComponent(ForgeText.qwen3VLHeadMap).path
        let tally = try ForgeText.forgeHead(folder: c.folder, to: out)
        XCTAssertEqual(tally.tensors, 2)
        XCTAssertEqual(tally.inexact, 0)
        let map = try Artifact(path: out)
        XCTAssertEqual(map.header["kind"] as? String, "qwen3-vl-lm-head")
        XCTAssertEqual(map.header["eos_token_ids"] as? [Int], [151645, 151643])
        XCTAssertEqual(map.header["map_dtype"] as? String, "bfloat16")
        XCTAssertEqual(map.order, ["norm.weight", "lm_head.weight"])
        XCTAssertEqual(map.tensors["lm_head.weight"]?.shape, [6, 8])
        XCTAssertEqual(map.tensors["lm_head.weight"]?.dtype, .bfloat16)
        let kept = Array(UnsafeRawBufferPointer(start: map.pointer("lm_head.weight"), count: c.head.count))
        XCTAssertEqual(kept, c.head)
        XCTAssertEqual(Array(UnsafeRawBufferPointer(start: map.pointer("norm.weight"), count: c.norm.count)), c.norm)
        // The decoding reads rows of it, widened.
        var row = [Float](repeating: 0, count: 16)
        try map.materializeRows("lm_head.weight", first: 2, count: 2, into: &row)
        XCTAssertEqual(row[0], Float(bitPattern: UInt32(c.head[2 * 8 * 2]) << 16 | UInt32(c.head[2 * 8 * 2 + 1]) << 24))

        let tied = try checkpoint(tied: true)
        XCTAssertThrowsError(try ForgeText.forgeHead(folder: tied.folder, to: root.appendingPathComponent("tied.silicon").path))
    }

    /// **Resident tensors are the map's bytes, from the process's pages**: the same values through every reader, the
    /// size the need charges (`Artifact.residentBytes`), gone on release; a second call reads nothing again.
    func testResidentTensorsAreTheMapsBytes() throws {
        let c = try checkpoint()
        let out = root.appendingPathComponent("head.silicon").path
        _ = try ForgeText.forgeHead(folder: c.folder, to: out)
        let map = try Artifact(path: out)
        var before = [Float](repeating: 0, count: 48)
        _ = try before.withUnsafeMutableBufferPointer { try map.materialize("lm_head.weight", into: $0) }
        let mapped = map.pointer("lm_head.weight")
        var fractions: [Double] = []
        try map.keepResident(prefixes: ["lm_head.weight", "norm.weight"]) { fractions.append($0) }
        XCTAssertEqual(fractions.last, 1)
        XCTAssertNotEqual(map.pointer("lm_head.weight"), mapped, "served from the pages, not from the mapping")
        var after = [Float](repeating: 0, count: 48)
        _ = try after.withUnsafeMutableBufferPointer { try map.materialize("lm_head.weight", into: $0) }
        XCTAssertEqual(before, after)
        XCTAssertEqual(map.residentCapacity, Artifact.residentBytes(header: map.header, names: ["lm_head.weight", "norm.weight"]))
        XCTAssertEqual(map.residentCapacity, 2 * 16384, "each tensor on its own page")
        fractions = []
        try map.keepResident(prefixes: ["lm_head.weight"]) { fractions.append($0) }
        XCTAssertTrue(fractions.isEmpty, "already kept")
        let cancelled = Cancellation(); cancelled.cancel()
        map.releaseResident()
        XCTAssertEqual(map.residentCapacity, 0)
        XCTAssertEqual(map.pointer("lm_head.weight"), mapped)
        XCTAssertThrowsError(try map.keepResident(prefixes: ["lm_head.weight"], cancellation: cancelled))
        XCTAssertEqual(map.residentCapacity, 0, "a cancelled load keeps nothing")
    }

    /// **Tensors read ahead are the map's bytes too** (`keepResidentAhead`, the kept text encoder's): served from the pages once
    /// read, the size the need charges, the prefetcher's to leave alone; given back with the artifact's keep.
    func testResidentTensorsReadAheadAreTheMapsBytes() throws {
        let c = try checkpoint()
        let out = root.appendingPathComponent("head.silicon").path
        _ = try ForgeText.forgeHead(folder: c.folder, to: out)
        let map = try Artifact(path: out)
        var before = [Float](repeating: 0, count: 48)
        _ = try before.withUnsafeMutableBufferPointer { try map.materialize("lm_head.weight", into: $0) }
        let mapped = map.pointer("lm_head.weight")
        XCTAssertNil(map.residentLoad)
        try map.keepResidentAhead(prefixes: ["lm_head.weight", "norm.weight"])
        XCTAssertTrue(map.isResident("lm_head.weight"))
        var after = [Float](repeating: 0, count: 48)
        _ = try after.withUnsafeMutableBufferPointer { try map.materialize("lm_head.weight", into: $0) }
        XCTAssertEqual(before, after)
        XCTAssertNotEqual(map.pointer("lm_head.weight"), mapped, "served from the pages, not from the mapping")
        map.waitForResidentLoad()
        XCTAssertEqual(map.residentLoad, .complete)
        XCTAssertEqual(map.residentCapacity, Artifact.residentBytes(header: map.header, names: ["lm_head.weight", "norm.weight"]))
        map.releaseResident()
        XCTAssertFalse(map.isResident("lm_head.weight"))
        XCTAssertEqual(map.pointer("lm_head.weight"), mapped)
    }

    /// The head's files are the family's own (the Turbo's `text_encoder/` configs) and two pinned byte ranges of its encoder
    /// file; the family's map is counted with its encoder and removed with it, and a family without a writer has none.
    func testTheHeadIsPinnedAndManaged() throws {
        let files = Installer.writerHead(.qwenImage21)
        XCTAssertEqual(files.map(\.path), ["text_encoder/config.json", "text_encoder/generation_config.json"])
        XCTAssertEqual(Set(files.map(\.repository.name)), ["Qwen/Qwen-Image-2.1-Turbo"])
        XCTAssertEqual(files[0].repository.revision, "d65dbc9a7e8f6b5479e33dee6030eaab2a906509")
        XCTAssertNil(files[0].sha256)
        let source = Installer.writerHeadSource
        XCTAssertEqual(source.repository.revision, files[0].repository.revision)
        XCTAssertEqual(source.sha256, "a1eee8233f3063466dd3b06e857b0a2cf13a77c543908374771d51a546beb2e0")
        XCTAssertEqual(Installer.writerHeadTensors.map(\.name), ["lm_head.weight", "model.language_model.norm.weight"])
        XCTAssertEqual(Installer.writerHeadTensors.map(\.length), [1_244_659_712, 8_192])
        XCTAssertTrue(Installer.writerHeadTensors.allSatisfy { $0.sha256.count == 64 && $0.begin + $0.length <= source.size! })
        XCTAssertEqual(Installer.writerHeadDownloadBytes, 1_244_667_904)
        for f in Family.allCases where f != .qwenImage21 { XCTAssertTrue(Installer.writerHead(f).isEmpty); XCTAssertNil(f.writerHead) }
        let library = Library(root: root.appendingPathComponent("lib"))
        try FileManager.default.createDirectory(atPath: library.map(""), withIntermediateDirectories: true)
        XCTAssertNil(library.writerHeadMap(.qwenImage21))
        try Data(repeating: 4, count: 300).write(to: URL(fileURLWithPath: library.map(ForgeText.qwen3VLHeadMap)))
        XCTAssertEqual(library.writerHeadMap(.qwenImage21), library.map(ForgeText.qwen3VLHeadMap))
        XCTAssertEqual(try XCTUnwrap(library.occupancy().families.first { $0.family == .qwenImage21 }).encoder, 300)
        try library.uninstall(.qwenImage21)
        XCTAssertNil(library.writerHeadMap(.qwenImage21))
    }

    // ── the memory need, on the real shapes ────────────────────────────────────────────────────

    private struct Shapes { let k: Int, n: Int }

    /// Qwen3-VL-8B's encoder map as `Artifact` lays it out, from its shapes: per layer q, k, v, o, gate, up, down (int8 +
    /// per-column fp32 scales, convrot 256 — or bf16) and four small norms, the embedding table, a vision tower as one bf16 block.
    private func encoderHeader(int8: Bool) -> [String: Any] {
        var tensors: [String: [String: Any]] = [:]
        var cursor = 16384   // each tensor on its own page, in the order added, as the forge lays a map out
        func add(_ name: String, _ shape: [Int], int8 weight: Bool = false) {
            let count = shape.reduce(1, *)
            var entry: [String: Any] = ["shape": shape, "offset": cursor]
            if weight && int8 {
                entry["dtype"] = "int8"; entry["bytes"] = count + 4 * shape[1]
                entry["scale"] = ["offset": count, "dtype": "float32", "shape": [shape[1]]]
                entry["rotation"] = ["kind": "convrot", "group": 256]
            } else { entry["dtype"] = "bfloat16"; entry["bytes"] = 2 * count }
            cursor += ((entry["bytes"] as! Int) + 16383) / 16384 * 16384
            tensors[name] = entry
        }
        for l in 0..<36 {
            let p = "layers.\(l)."
            add(p + "self_attn.q_proj.weight", [4096, 4096], int8: true); add(p + "self_attn.k_proj.weight", [4096, 1024], int8: true)
            add(p + "self_attn.v_proj.weight", [4096, 1024], int8: true); add(p + "self_attn.o_proj.weight", [4096, 4096], int8: true)
            add(p + "mlp.gate_proj.weight", [4096, 12288], int8: true); add(p + "mlp.up_proj.weight", [4096, 12288], int8: true)
            add(p + "mlp.down_proj.weight", [12288, 4096], int8: true)
            add(p + "input_layernorm.weight", [4096]); add(p + "post_attention_layernorm.weight", [4096])
            add(p + "self_attn.q_norm.weight", [128]); add(p + "self_attn.k_norm.weight", [128])
        }
        add("embed_tokens.weight", [151936, 4096])
        return ["page": 16384, "tensors": tensors, "last_layer": 35, "linear_weights_transposed": true,
                "config": ["hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
                           "intermediate_size": 12288, "num_hidden_layers": 36, "rope_theta": 5_000_000.0,
                           "rope_scaling": ["mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default"]],
                "vision_config": ["hidden_size": 1152, "num_heads": 16, "intermediate_size": 4304, "depth": 27, "patch_size": 16,
                                  "temporal_patch_size": 2, "spatial_merge_size": 2, "num_position_embeddings": 2304,
                                  "out_hidden_size": 4096, "deepstack_visual_indexes": [8, 16, 24], "hidden_act": "gelu_pytorch_tanh"],
                "vision_patch_embed_as_linear": "(c, t, y, x)"]
    }

    private func headHeader() -> [String: Any] {
        ["kind": "qwen3-vl-lm-head", "page": 16384, "tensors": [
            "lm_head.weight": ["shape": [151936, 4096], "dtype": "bfloat16", "bytes": 151936 * 4096 * 2],
            "norm.weight": ["shape": [4096], "dtype": "bfloat16", "bytes": 8192]]]
    }

    private func maps(int8: Bool) throws -> PromptWriterMaps {
        try XCTUnwrap(PromptWriterMaps(encoder: encoderHeader(int8: int8), head: headHeader(), tokenizerTables: 20_000_000,
                                       detokenization: 3_000_000))
    }

    /// The synthetic header is the installed Compact map's layers: 36 × (7 weights + 4 norms) summing to its 6 951 979 008 bytes.
    func testTheSyntheticHeaderIsTheInstalledMapsShapes() throws {
        let table = try XCTUnwrap(encoderHeader(int8: true)["tensors"] as? [String: [String: Any]])
        XCTAssertEqual(table.filter { $0.key.hasPrefix("layers.") }.values.reduce(0) { $0 + ($1["bytes"] as! Int) }, 6_951_979_008)
        XCTAssertEqual(table["layers.0.self_attn.q_proj.weight"]?["bytes"] as? Int, 16_793_600)
        XCTAssertEqual(table["layers.0.mlp.down_proj.weight"]?["bytes"] as? Int, 50_348_032)
    }

    /// **The writer's need at its limits** (2 048 prompt tokens, 768 new, a 1024² image), from shapes alone: the Compact's
    /// 8-bit layers resident plus the head and the larger phase — and the Mac that holds it.
    func testTheNeedOnTheRealShapes() throws {
        let compact = try maps(int8: true), standard = try maps(int8: false)
        // The layers at the map's width, each tensor on its own pages; the head bf16 as published.
        XCTAssertEqual(compact.residentHead, 151936 * 4096 * 2 + 16384)
        func rounded(_ header: [String: Any]) -> Int {
            (header["tensors"] as! [String: [String: Any]]).filter { $0.key.hasPrefix("layers.") }
                .reduce(0) { $0 + (($1.value["bytes"] as! Int) + 16383) / 16384 * 16384 }
        }
        XCTAssertEqual(compact.residentLayers, rounded(encoderHeader(int8: true)))
        XCTAssertEqual(compact.residentLayers, 6_951_979_008 + 36 * 2 * (16384 - 8192) + 36 * 2 * (16384 - 256) + 36 * 2 * (16384 - 4096),
                       "k and v (4 198 400 bytes) end a quarter into a page; the norms take a page each")
        XCTAssertGreaterThan(standard.residentLayers, 2 * compact.residentLayers * 99 / 100 * 99 / 100)
        XCTAssertGreaterThan(compact.rotationScratch, 0)
        XCTAssertEqual(standard.rotationScratch, 0)

        let need = PromptWriterNeed.atLimits(maps: compact, plan: .defaults)
        XCTAssertEqual(need.placement, .cached(layers: 36), "the need at the limits starts from every layer in the cache")
        XCTAssertEqual(compact.layerSpans.reduce(0, +), compact.residentLayers, "a layer's run is its tensors laid on pages")
        let language = try XCTUnwrap(need.phases.first { $0.kind == .text }), vision = try XCTUnwrap(need.phases.first { $0.kind == .vision })
        // 36 layers × 2 (K, V) × (2048 + 768) tokens × 1024 values × 4 B, to the page.
        XCTAssertEqual(language.bytes("K/V cache"), Qwen3VLGenerator.cacheBytes(config: compact.encoder, length: 2816))
        XCTAssertEqual(language.bytes("K/V cache"), 2 * 36 * 2816 * 1024 * 4)
        // 512 rows of activations (59 520 floats a row), the reserve (hidden × intermediate), a block of the head widened, the small ones.
        let floats = [512 * 4096, 512 * 4096, 512 * 4096, 512 * 4096, 512 * 1024, 512 * 1024, 512 * 4096, 512 * 12288, 512 * 12288,
                      512 * 12288, 512 * 128, 4096, 4096, 4096 * 12288, 4096, 8192 * 4096, 8192]
        XCTAssertEqual(language.bytes("writer arena"), floats.reduce(0) { $0 + Arena.capacity(for: $1 * 4) })
        XCTAssertLessThan(language.bytes("writer arena"), 500_000_000)
        XCTAssertEqual(vision.bytes("vision features"), Qwen3VLVision.featureBytes(tokens: 1024))
        // The weights are the cache's, not the writer's: what it holds through its life is the decoder's scratch.
        XCTAssertEqual(need.residentBytes, Qwen3VLGenerator.decoderBytes(config: compact.encoder, vocabulary: compact.vocabulary))
        XCTAssertGreaterThan(need.peak, need.residentBytes)
        XCTAssertEqual(need.peak, need.residentBytes + max(language.bytes, vision.bytes))
        XCTAssertEqual(need.cacheBytes, compact.residentLayers + compact.residentHead)
        XCTAssertEqual(need.streamedBytes, 0)

        // The Mac: 16 GB (B = 10 GiB) holds the Compact whole in the cache; the Standard's bf16 layers (13.9 GB) do not fit it
        // whole, and are streamed in part rather than refused.
        let sixteen = MemoryBudget.ceiling(physical: 16 << 30), eight = MemoryBudget.ceiling(physical: 8 << 30)
        XCTAssertEqual([eight, sixteen], [4 << 30, 10 << 30])
        XCTAssertEqual(need.availability(ceiling: sixteen), .available(need: need.peak))
        let big = PromptWriterNeed.atLimits(maps: standard, plan: .defaults)
        XCTAssertGreaterThan(big.total, sixteen)
        guard case .needsMac = big.availability(ceiling: sixteen) else { return XCTFail("every layer cached does not fit") }
        XCTAssertTrue(PromptWriterNeed.atLimits(maps: standard, ceiling: sixteen, plan: .defaults).placement.streams)
        print("PromptWriter need (GB): compact peak \(Double(need.peak) / 1e9) = resident \(Double(need.residentBytes) / 1e9) + language \(Double(language.bytes) / 1e9) / vision \(Double(vision.bytes) / 1e9); cache \(Double(need.cacheBytes) / 1e9); standard peak \(Double(big.peak) / 1e9)")
    }

    /// The lab asks `availability` again once the weights are loaded (window key, render end): the footprint then holds
    /// them, and the need charges them too. A writer that holds its weights takes them out of the base, so the answer
    /// does not change between the first request and the second (it used to ask for a 32 GB Mac).
    func testAvailabilityDoesNotCountTheHeldWeightsTwice() throws {
        let m = try maps(int8: true)
        let sixteen = MemoryBudget.ceiling(physical: 16 << 30)
        let writer = PromptWriter(maps: m, encoderPath: "", headPath: "", tokenizerPath: "", ceiling: { sixteen })
        let app = 600_000_000
        let before = writer.availability(base: app)
        guard case .available = before else { return XCTFail("16 GB holds the Compact") }
        let resident = PromptWriterNeed.atLimits(maps: m, ceiling: sixteen, plan: .defaults).residentBytes
        writer.noteResident(resident)
        XCTAssertEqual(writer.availability(base: app + resident), before, "loaded: the footprint holds the weights")
        writer.noteResident(0)
        XCTAssertNotEqual(writer.availability(base: app + resident), before, "unloaded, the same footprint is the app's")
    }

    // ── the plan, chosen in B ──────────────────────────────────────────────────────────────────

    private func plan(_ m: PromptWriterMaps, gigabytes: Int, base: Int = 600_000_000) -> PromptWriterNeed {
        PromptWriterNeed.atLimits(maps: m, base: base, ceiling: MemoryBudget.ceiling(physical: gigabytes << 30), plan: .defaults)
    }

    /// **The plan at 8, 16, 32 and 64 GB** (B = 4, 10, 22, 46 GiB), for the Compact's 8-bit layers and the Standard's bf16, with
    /// the app's 600 MB beside: the layers kept in the cache, the anonymous peak, the cache, what a token reads around it.
    func testThePlanAtEachMacSize() throws {
        let compact = try maps(int8: true), standard = try maps(int8: false)
        var table = ""
        for (name, m) in [("compact", compact), ("standard", standard)] {
            var previousCached = -1
            for gigabytes in [8, 16, 32, 64] {
                let b = MemoryBudget.ceiling(physical: gigabytes << 30)
                let chosen = plan(m, gigabytes: gigabytes)
                XCTAssertLessThanOrEqual(chosen.total, b, "\(name) \(gigabytes) GB: the plan that is taken fits B, cache included")
                XCTAssertGreaterThanOrEqual(chosen.placement.cachedLayers, previousCached, "a bigger Mac never caches less")
                previousCached = chosen.placement.cachedLayers
                // The anonymous side is the decoder's scratch and the staging buffers, none when nothing streams.
                let staging = chosen.placement.streams ? 2 * Arena.capacity(for: m.layerSpans.max()!) : 0
                XCTAssertEqual(chosen.residentBytes, Qwen3VLGenerator.decoderBytes(config: m.encoder, vocabulary: m.vocabulary) + staging)
                XCTAssertEqual(chosen.cacheBytes + chosen.streamedBytes, m.layerSpans.reduce(0, +) + m.residentHead,
                               "every byte a token reads is the cache's or the stream's")
                table += String(format: "%@ %2d GB (B %.2f): cached %2d/36 layers, streamed %2d, anonymous %.2f GB, cache %.2f GB, streamed/token %.2f GB\n",
                                name as NSString, gigabytes, Double(b) / 1e9, chosen.placement.cachedLayers, chosen.placement.streamedLayers,
                                Double(chosen.peak) / 1e9, Double(chosen.cacheBytes) / 1e9, Double(chosen.streamedBytes) / 1e9)
            }
        }
        print(table)
        // 16 GB: the Compact whole in the cache (8.2 GB of clean pages), as a render leaves its map; the Standard (14 GB) cannot.
        XCTAssertFalse(plan(compact, gigabytes: 16).placement.streams)
        XCTAssertTrue(plan(standard, gigabytes: 16).placement.streams)
        // 32 and 64: everything is cached, whatever the width.
        for m in [compact, standard] { for g in [32, 64] { XCTAssertFalse(plan(m, gigabytes: g).placement.streams) } }
        // 8 GB: the Compact is offered, its tail around the cache, where it used to be refused.
        let eight = plan(compact, gigabytes: 8)
        XCTAssertTrue(eight.placement.streams)
        XCTAssertGreaterThan(eight.placement.cachedLayers, 0)
        XCTAssertEqual(eight.availability(ceiling: MemoryBudget.ceiling(physical: 8 << 30)), .available(need: eight.peak))
    }

    /// **Offered if and only if a plan fits B**: from the smallest plan's total, any ceiling gives a writer; a byte under it,
    /// « needs a Mac with N GB », and the Mac named is one whose B holds that smallest plan.
    func testTheWriterIsOfferedExactlyWhenAPlanFits() throws {
        let m = try maps(int8: true)
        let base = 600_000_000
        let all = PromptWriterNeed.atLimits(maps: m, base: base, plan: .defaults)
        let floor = all.placed(streamedFrom: 0)
        XCTAssertLessThan(floor.total, all.total)
        XCTAssertEqual(floor.cacheBytes, m.residentHead, "the head is the cache's whatever the plan")
        XCTAssertEqual(floor.streamedBytes, m.layerSpans.reduce(0, +))
        for ceiling in [floor.total, floor.total + 1, (floor.total + all.total) / 2, all.total, all.total + 1] {
            let chosen = all.chosen(ceiling: ceiling)
            XCTAssertEqual(chosen.availability(ceiling: ceiling), .available(need: chosen.peak), "B \(ceiling)")
            XCTAssertLessThanOrEqual(chosen.total, ceiling)
            // The most layers kept that fit: one more would not.
            if chosen.placement.streams {
                XCTAssertGreaterThan(all.placed(streamedFrom: chosen.placement.streamedFrom + 1).total, ceiling)
            }
        }
        XCTAssertEqual(all.chosen(ceiling: all.total).placement, .cached(layers: 36))
        let refused = all.chosen(ceiling: floor.total - 1)
        XCTAssertEqual(refused.placement.streamedFrom, 0, "nothing smaller to take: the floor is what is refused")
        guard case let .needsMac(gigabytes, need) = refused.availability(ceiling: floor.total - 1) else { return XCTFail("a plan fits under its floor") }
        XCTAssertEqual(need, floor.total)
        XCTAssertGreaterThanOrEqual(MemoryBudget.ceiling(physical: gigabytes << 30), floor.total)
        XCTAssertEqual(gigabytes, MemoryBudget.macNeeded(for: floor.total))
        // A Mac that sells today holds the floor: every size from 8 GB (B = 4 GiB) offers the Compact.
        XCTAssertLessThan(floor.total, MemoryBudget.ceiling(physical: 8 << 30))
    }

    /// **A token's seconds from the card**, written in memory: cached bytes over the memory's bandwidth, plus streamed bytes over the
    /// disk's. No figure, no prediction; a plan that streams needs the disk's.
    func testTheTokenIsPredictedFromTheCard() throws {
        let m = try maps(int8: true)
        let all = PromptWriterNeed.atLimits(maps: m, plan: .defaults)
        var card = CapacityModel()
        XCTAssertNil(all.tokenSeconds(model: card), "a card without the memory's bandwidth predicts nothing")
        card.memoryBandwidth = 200e9
        let cached = try XCTUnwrap(all.tokenSeconds(model: card))
        XCTAssertEqual(cached, Double(m.residentLayers + m.residentHead) / 200e9, accuracy: 1e-12)
        XCTAssertEqual(cached, 0.041, accuracy: 0.002, "8.2 GB at 200 GB/s: 41 ms a token")
        let half = all.placed(streamedFrom: 18)
        XCTAssertNil(half.tokenSeconds(model: card), "a stream without the disk's rate is not guessed")
        card.diskBandwidth = 3e9
        let streamed = try XCTUnwrap(half.tokenSeconds(model: card))
        XCTAssertEqual(streamed, Double(half.cacheBytes) / 200e9 + Double(half.streamedBytes) / 3e9, accuracy: 1e-12)
        XCTAssertEqual(streamed, 1.18, accuracy: 0.05, "3.5 GB around the cache at 3 GB/s dominate: about a second a token")
        // Every layer more around the cache costs more: the prediction is monotone in what is streamed.
        let seconds = (0...36).map { all.placed(streamedFrom: $0).tokenSeconds(model: card)! }
        XCTAssertEqual(seconds, seconds.sorted(by: >), "streaming more is never faster")
        print(String(format: "token: all cached %.3f s, half %.3f s, all streamed %.3f s", seconds[36], seconds[18], seconds[0]))
    }

    /// A prompt that grows makes the need grow by its cache; a text rewrite has no vision phase; the cache is the only
    /// term that follows the prompt beyond the first 512 rows.
    func testTheNeedFollowsThePrompt() throws {
        let m = try maps(int8: true)
        let short = PromptWriterNeed(maps: m, promptTokens: 300, newTokens: 768, plan: .defaults)
        let long = PromptWriterNeed(maps: m, promptTokens: 2000, newTokens: 768, plan: .defaults)
        XCTAssertEqual(short.phases.map(\.kind), [.text])
        XCTAssertEqual(long.peak - short.peak, 2 * 36 * 1700 * 1024 * 4 + (long.phases[0].bytes("attention scores") - short.phases[0].bytes("attention scores"))
            + (long.phases[0].bytes("writer arena") - short.phases[0].bytes("writer arena"))
            + (long.phases[0].bytes("writer GEMM") - short.phases[0].bytes("writer GEMM")), "cache, scores, rows")
        XCTAssertEqual(Qwen3VLGenerator.rows(prompt: 300), 300)
        XCTAssertEqual(Qwen3VLGenerator.rows(prompt: 5000), 512)
        XCTAssertEqual(Qwen3VLGenerator.rows(prompt: 0), 1)
        var withBase = short
        withBase.base = 700_000_000
        XCTAssertEqual(withBase.peak - short.peak, 700_000_000)
    }

    // ── the head by byte range ───────────────────────────────────────────────────────────────

    /// A synthetic text encoder: three tensors, the header padded as the real one is.
    private func encoderFile() throws -> (path: String, bytes: [UInt8], header: [String: Any], headerLength: Int, ranges: [String: (Int, Int)]) {
        let a = bf16(10, seed: 1), norm = bf16(8, seed: 3), head = bf16(6 * 8, seed: 5)
        let path = root.appendingPathComponent("encoder-\(UUID()).safetensors").path
        try safetensors([("a.weight", [10], a), ("lm_head.weight", [6, 8], head), ("model.language_model.norm.weight", [8], norm)], to: path)
        let bytes = [UInt8](try Data(contentsOf: URL(fileURLWithPath: path)))
        let n = Int(bytes[0...7].enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * UInt64($1.offset)) })
        let header = try JSONSerialization.jsonObject(with: Data(bytes[8..<(8 + n)])) as! [String: Any]
        var ranges: [String: (Int, Int)] = [:]
        for (name, v) in header where name != "__metadata__" {
            let o = (v as! [String: Any])["data_offsets"] as! [Int]
            ranges[name] = (8 + n + o[0], o[1] - o[0])
        }
        return (path, bytes, header, n, ranges)
    }

    private func sha(_ bytes: ArraySlice<UInt8>) throws -> String {
        let p = root.appendingPathComponent("hash-\(UUID())").path
        try Data(bytes).write(to: URL(fileURLWithPath: p))
        return try Downloader.sha256(p)
    }

    /// **The range plan reads offsets from the header** (data starts at `8 + header length`), and refuses a header that
    /// no longer says what the pin says.
    func testTheRangePlanFollowsTheHeader() throws {
        let f = try encoderFile()
        let (hb, hl) = f.ranges["lm_head.weight"]!, (nb, nl) = f.ranges["model.language_model.norm.weight"]!
        let pins = [Downloader.PinnedRange(name: "lm_head.weight", begin: hb, length: hl, sha256: "x"),
                    Downloader.PinnedRange(name: "model.language_model.norm.weight", begin: nb, length: nl, sha256: "y")]
        let plan = try Downloader.plan(pins, headerLength: f.headerLength, header: f.header, file: "synthetic")
        XCTAssertEqual(plan.map(\.begin), [hb, nb])
        XCTAssertEqual(plan[0].shape, [6, 8])
        XCTAssertEqual(plan[0].dtype, "BF16")
        XCTAssertEqual(plan[0].last, hb + hl - 1)
        var moved = pins; moved[0] = .init(name: "lm_head.weight", begin: hb + 2, length: hl, sha256: "x")
        XCTAssertThrowsError(try Downloader.plan(moved, headerLength: f.headerLength, header: f.header, file: "synthetic"))
        var shorter = pins; shorter[1] = .init(name: pins[1].name, begin: nb, length: nl - 2, sha256: "y")
        XCTAssertThrowsError(try Downloader.plan(shorter, headerLength: f.headerLength, header: f.header, file: "synthetic"))
        XCTAssertThrowsError(try Downloader.plan([.init(name: "absent", begin: 0, length: 1, sha256: "z")],
                                                 headerLength: f.headerLength, header: f.header, file: "synthetic"))
    }

    /// **A range is used only if its bytes are the pinned ones**: one wrong bit, or a missing byte, removes the file.
    func testAWrongByteInARangeIsRefused() throws {
        let f = try encoderFile()
        let (b, l) = f.ranges["lm_head.weight"]!
        let range = Downloader.TensorRange(name: "lm_head.weight", dtype: "BF16", shape: [6, 8], begin: b, length: l,
                                           sha256: try sha(f.bytes[b..<(b + l)]))
        let good = root.appendingPathComponent("good").path
        try Data(f.bytes[b..<(b + l)]).write(to: URL(fileURLWithPath: good))
        XCTAssertNoThrow(try Downloader.verify(good, range))
        var wrong = Array(f.bytes[b..<(b + l)]); wrong[17] ^= 1
        let bad = root.appendingPathComponent("bad").path
        try Data(wrong).write(to: URL(fileURLWithPath: bad))
        XCTAssertThrowsError(try Downloader.verify(bad, range))
        XCTAssertFalse(FileManager.default.fileExists(atPath: bad), "a refused range is not kept")
        let short = root.appendingPathComponent("short").path
        try Data(f.bytes[b..<(b + l - 1)]).write(to: URL(fileURLWithPath: short))
        XCTAssertThrowsError(try Downloader.verify(short, range))
    }

    /// **The head forged from two ranged tensors** equals the one forged from the whole shard: the ranges laid in a small
    /// safetensors file beside the configs, the same bits in the map.
    func testTheHeadForgedFromRangesKeepsTheBits() throws {
        let f = try encoderFile()
        var laid: [(Downloader.TensorRange, String)] = []
        for name in ["lm_head.weight", "model.language_model.norm.weight"] {
            let (b, l) = f.ranges[name]!
            let shape = ((f.header[name] as! [String: Any])["shape"] as! [Int])
            let file = root.appendingPathComponent("range-" + name).path
            try Data(f.bytes[b..<(b + l)]).write(to: URL(fileURLWithPath: file))
            laid.append((Downloader.TensorRange(name: name, dtype: "BF16", shape: shape, begin: b, length: l, sha256: try sha(f.bytes[b..<(b + l)])), file))
        }
        let c = try checkpoint()
        let folder = root.appendingPathComponent("ranged-\(UUID())").path
        try FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)
        for name in ["config.json", "generation_config.json"] {
            try FileManager.default.copyItem(atPath: c.folder + "/" + name, toPath: folder + "/" + name)
        }
        try Downloader.assembleSafetensors(laid, to: folder + "/model.safetensors")
        let whole = root.appendingPathComponent("whole.silicon").path, ranged = root.appendingPathComponent("ranged.silicon").path
        _ = try ForgeText.forgeHead(folder: c.folder, to: whole)
        let tally = try ForgeText.forgeHead(folder: folder, to: ranged)
        XCTAssertEqual(tally.tensors, 2)
        let a = try Artifact(path: ranged)
        XCTAssertEqual(a.header["eos_token_ids"] as? [Int], [151645, 151643])
        XCTAssertEqual(a.tensors["lm_head.weight"]?.shape, [6, 8])
        // The checkpoint's bits and the ranged file's differ (other seeds): the ranged map holds the ranged file's.
        let (b, l) = f.ranges["lm_head.weight"]!
        XCTAssertEqual(Array(UnsafeRawBufferPointer(start: a.pointer("lm_head.weight"), count: l)), Array(f.bytes[b..<(b + l)]))
        let (nb, nl) = f.ranges["model.language_model.norm.weight"]!
        XCTAssertEqual(Array(UnsafeRawBufferPointer(start: a.pointer("norm.weight"), count: nl)), Array(f.bytes[nb..<(nb + nl)]))
    }
}
