import XCTest
import MetalPerformanceShadersGraph
@testable import Siliconed

/// **The two pieces that escaped the second pass**.
///
/// The tokenizer parses 7 MB of JSON at startup, the decoder reads a weights file: neither fits in
/// a test target that forbids itself from reading a file. That is no reason to leave them out,
/// though, because what is risky about them is **not** the reading — it is the rule that comes
/// after.
///
///   - For the tokenizer: truncation and padding. The trap had long been documented
///     ("at 512, a long prompt loses the assistant's header entirely") and had never been run
///     against an assertion. It is extracted into `Tokenizer.pad`.
///   - For the decoder: the tile geometry. `positions` and `ramp` decide where we cut and how we
///     glue back together — hence whether a seam shows. They are pure, and the close-out only
///     judges them through a PSNR of 53 dB that says "roughly" and nothing more.
///
/// **What these tests do not replace**: the tokenizer check compares the 512 identifiers to the
/// ground truth of `transformers`, and "vae tuiles" compares an image. No assertion here says the
/// tokenizer tokenizes correctly — only that what it does *afterwards* is what is written.
final class TokenizerAndVAETests: XCTestCase {

    // ── the tokenizer: the half that does not depend on the vocabulary ─────────────────────────

    func testPaddingFillsUpToTheFixedLength() {
        // The pipeline pads to a fixed length and does not cut short sequences: the mask carries
        // the information, not the length.
        let (ids, mask) = Tokenizer.pad([1, 2, 3], maxLength: 8)
        XCTAssertEqual(ids, [1, 2, 3, Tokenizer.padToken, Tokenizer.padToken,
                             Tokenizer.padToken, Tokenizer.padToken, Tokenizer.padToken])
        XCTAssertEqual(mask, [1, 1, 1, 0, 0, 0, 0, 0])
        XCTAssertEqual(mask.reduce(0, +), 3, "the mask counts the real tokens")
    }

    func testPaddingTouchesNothingWhenLengthIsExact() {
        let (ids, mask) = Tokenizer.pad([7, 8], maxLength: 2)
        XCTAssertEqual(ids, [7, 8])
        XCTAssertEqual(mask, [1, 1])
    }

    func testTruncationIsSILENTAndCutsTheEnd() {
        // **The trap, executed.** The reference truncates without a word, and what goes is the END —
        // so `<|im_end|>\n<|im_start|>assistant\n` in full on a long prompt. The useful budget is
        // 504 tokens out of 512, and this test is here so that this behavior stays a reproduced
        // choice and not an accident that someone would one day "fix" without noticing.
        let long = Array(0..<600)
        let (ids, mask) = Tokenizer.pad(long, maxLength: 512)
        XCTAssertEqual(ids.count, 512)
        XCTAssertEqual(ids.first, 0)
        XCTAssertEqual(ids.last, 511, "it is the tail that is cut, not the head")
        XCTAssertTrue(mask.allSatisfy { $0 == 1 }, "truncated ⟹ no padding, hence full mask")
    }

    func testByteAlphabetIsABijectionOver256Bytes() {
        // The byte → symbol table of the `ByteLevel` stage. A wrong entry does not crash: it
        // encodes a non-ASCII byte as a symbol the vocabulary does not know, and the fault surfaces
        // three stages later, in a caption.
        let symbols = Tokenizer.byteSymbols
        XCTAssertEqual(symbols.count, 256)
        XCTAssertEqual(Set(symbols).count, 256, "two bytes cannot share a symbol")
        // Printable ASCII characters represent themselves — this is what makes a `tokenizer.json`
        // readable to the naked eye.
        for byte in UInt8(ascii: "!")...UInt8(ascii: "~") {
            XCTAssertEqual(symbols[Int(byte)], String(UnicodeScalar(byte)))
        }
        // The space, on the other hand, is moved: it is the famous "Ġ".
        XCTAssertNotEqual(symbols[Int(UInt8(ascii: " "))], " ")
    }

    /// **The tables read from the raw bytes**: escapes, both merge spellings, a repeated key
    /// (the last wins, as the dictionaries did), and the BPE that runs on them. Equal to the
    /// `JSONSerialization` dictionaries on Z-Image's and Qwen-Image-2.1's 151 k + 151 k entries
    /// (checked once on the files).
    func testTokenizerTablesFromRawBytes() throws {
        let json = #"""
        {"version": "1.0", "added_tokens": [{"id": 9, "content": "<|x|>", "single_word": false, "lstrip": false,
          "rstrip": false, "normalized": false, "special": true}],
         "normalizer": null,
         "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\p{L}+|\\s+"},
           "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}]},
         "post_processor": {"type": "ByteLevel", "add_prefix_space": false},
         "decoder": {"type": "ByteLevel", "add_prefix_space": false},
         "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
           "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false,
           "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4, "q\"\\": 5, "b": 6, "😀": 7},
           "merges": ["a b", ["ab", "c"], ["a", "b"]]}}
        """#
        let path = NSTemporaryDirectory() + "siliconed-tokenizer-\(getpid()).json"
        try json.write(toFile: path, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(atPath: path) }
        let t = try Tokenizer(file: path)
        XCTAssertEqual(t.id(of: "abc"), 4)
        XCTAssertEqual(t.id(of: "q\"\\"), 5)
        XCTAssertEqual(t.id(of: "b"), 6)
        XCTAssertEqual(t.id(of: "😀"), 7)
        XCTAssertNil(t.id(of: "abd"))
        XCTAssertEqual(t.rank(of: "a", "b"), 2)
        XCTAssertEqual(t.rank(of: "ab", "c"), 1)
        XCTAssertEqual(t.encode("abc<|x|>"), [4, 9])
    }

    // ── the decoder: a missing weight ────────────────────────────────────────────────────────
}
