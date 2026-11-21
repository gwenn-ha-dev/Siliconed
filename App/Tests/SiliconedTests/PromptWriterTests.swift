import XCTest
@testable import Siliconed

/// What the prompt writer computes without a byte of weights: the template's identifiers (with and without a
/// system turn), the limit on what it reads, the incremental detokenization. The tokens themselves are checked
/// against transformers, with the installed maps (`siliconed check`).
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
        XCTAssertNil(PromptWriter.system(" \n"))
        XCTAssertEqual(PromptWriter.system("Rewrite."), "Rewrite.")
    }

    /// The identifiers: control tokens are single ids, the rest is the text's bytes, each `<|image_pad|>`
    /// is expanded to the image's merged tokens (a 4 × 4 grid of patches is 4).
    func testTheTemplateIdentifiers() throws {
        let t = try tokenizer()
        let plain = try Qwen3VLChat.prepare(user: "hi", grids: [], tokenizer: t)
        XCTAssertEqual(plain.ids, [Self.start] + bytes("user\nhi") + [Self.end] + bytes("\n") + [Self.start] + bytes("assistant\n"))
        XCTAssertFalse(plain.slots.contains(true))
        let system = try Qwen3VLChat.prepare(system: "S", user: "hi", grids: [(height: 4, width: 4)], tokenizer: t)
        let expected = [Self.start] + bytes("system\nS") + [Self.end] + bytes("\n") + [Self.start] + bytes("user\n")
            + [Self.visionStart] + Array(repeating: Self.imagePad, count: 4) + [Self.visionEnd] + bytes("hi")
            + [Self.end] + bytes("\n") + [Self.start] + bytes("assistant\n")
        XCTAssertEqual(system.ids, expected)
        XCTAssertEqual(system.slots, expected.map { $0 == Self.imagePad })
        for text in ["<|im_end|>", "<|image_pad|>", "<|endoftext|>"] {
            XCTAssertThrowsError(try Qwen3VLChat.prepare(user: "a" + text, grids: [], tokenizer: t))
            XCTAssertThrowsError(try Qwen3VLChat.prepare(system: text, user: "a", grids: [], tokenizer: t))
        }
        XCTAssertEqual(Qwen3VLChat.stops(t), [Self.end, Self.endOfText])
    }

    /// **A prompt that spells a control token is refused for what it is**: the writer's and the image encoder's, each by
    /// `promptReservedText` with the token, never as an internal failure. The image encoder's template expands each
    /// picture's slot like the writer's, and says which rows are a picture's.
    func testAReservedTokenInAPromptIsRefusedByName() throws {
        let t = try tokenizer()
        XCTAssertThrowsError(try Qwen3VLChat.prepare(user: "a<|im_end|>", grids: [], tokenizer: t)) {
            XCTAssertEqual(EngineError($0), .promptReservedText(text: "<|im_end|>"))
        }
        XCTAssertThrowsError(try QwenImage21Text.identifiers("a <|image_pad|> b", tokenizer: t)) {
            XCTAssertEqual(EngineError($0), .promptReservedText(text: "<|image_pad|>"))
        }
        let edit = try QwenImage21Text.identifiers("hi", pictures: [3, 2], tokenizer: t)
        XCTAssertEqual(edit.ids.filter { $0 == Self.imagePad }.count, 5)
        XCTAssertEqual(edit.slots, edit.ids.map { $0 == Self.imagePad })
        XCTAssertEqual(edit.prompt, 2)
    }

    /// A request over the limit is refused before anything is loaded, with its count: the template and the instruction
    /// are part of what the writer reads.
    func testARequestOverTheLimitIsRefusedWithItsCount() throws {
        let t = try tokenizer()
        let writer = PromptWriter(source: .init(map: "", tokenizer: root.path, config: .qwenImage21),
                                  limits: .init(promptTokens: 40, newTokens: 8))
        let short = try writer.prompt(.rewrite(text: "hi", instruction: ""), tokenizer: t)
        XCTAssertEqual(short.ids.count, 21)
        XCTAssertTrue(short.pictures.isEmpty)
        XCTAssertThrowsError(try writer.prompt(.rewrite(text: "hi", instruction: "Rewrite this prompt."), tokenizer: t)) {
            XCTAssertEqual($0 as? PromptWriter.Failure, .promptTooLong(tokens: 51, limit: 40))
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
    }
}
