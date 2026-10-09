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

    // ── the decoder: the tile geometry ─────────────────────────────────────────────

    func testTilesCoverTheWholePlane() {
        // **The property that matters**: not one latent pixel must escape the tiles. A hole would
        // not show on an average PSNR — it would show on the image, once.
        for L in [64, 96, 128, 160, 256] {
            for t in [32, 48, 80] where t < L {
                for o in [4, 8, 16] where t > 2 * o {
                    let p = VAE.positions(side: L, tile: t, overlap: o)
                    var covered = [Bool](repeating: false, count: L)
                    for begin in p {
                        XCTAssertGreaterThanOrEqual(begin, 0)
                        XCTAssertLessThanOrEqual(begin + t, L,
                                                 "a tile overflows: L=\(L) t=\(t) o=\(o)")
                        for i in begin..<(begin + t) { covered[i] = true }
                    }
                    XCTAssertFalse(covered.contains(false),
                                   "hole in the coverage: L=\(L) t=\(t) o=\(o) → \(p)")
                }
            }
        }
    }

    func testTilesTouchBothEdgesAndAdvance() {
        let p = VAE.positions(side: 128, tile: 48, overlap: 8)
        XCTAssertEqual(p.first, 0, "the first tile starts at the edge")
        XCTAssertEqual(p.last, 128 - 48, "the last one ends on the edge")
        for i in 0..<(p.count - 1) {
            XCTAssertLessThan(p[i], p[i + 1], "positions must advance strictly")
        }
    }

    func testTheEffectiveOverlapIsNeverBelowTheRequestedOne() {
        // The documentation of `positions` promises it: "at least the one requested — never less,
        // often more at the edge". A promise never executed is a promise we believe we keep.
        for L in [64, 96, 128, 160, 256] {
            for t in [32, 48, 80] where t < L {
                for o in [4, 8, 16] where t > 2 * o {
                    let p = VAE.positions(side: L, tile: t, overlap: o)
                    for i in 0..<(p.count - 1) {
                        let overlap = (p[i] + t) - p[i + 1]
                        XCTAssertGreaterThanOrEqual(overlap, o,
                            "overlap \(overlap) < \(o) requested: L=\(L) t=\(t) → \(p)")
                    }
                }
            }
        }
    }

    func testATileLargerThanThePlaneIsNotATile() {
        XCTAssertEqual(VAE.positions(side: 64, tile: 64, overlap: 8), [0])
        XCTAssertEqual(VAE.positions(side: 64, tile: 128, overlap: 8), [0])
    }

    func testTheFadeNeverCancelsOut() {
        // **It is a division by zero being avoided, not an elegance.** The tile weights are
        // normalized by their sum: a zero weight on both sides of an exact overlap would make 0/0
        // right in the middle of the image. Hence the start at `1/(marge+1)` and not at zero.
        let length = 48, margin = 8
        for position in 0..<length {
            let weights = VAE.ramp(position, length: length, margin: margin,
                                  freeStart: true, freeEnd: true)
            XCTAssertGreaterThan(weights, 0, "zero weight at position \(position)")
            XCTAssertLessThanOrEqual(weights, 1)
        }
    }

    func testTheFadeIsOneAtTheCenterAndDescendsTowardTheEdges() {
        let length = 48, margin = 8
        func weights(_ p: Int) -> Float {
            VAE.ramp(p, length: length, margin: margin, freeStart: true, freeEnd: true)
        }
        XCTAssertEqual(weights(length / 2), 1, accuracy: 1e-6, "a tile's core weighs full")
        for p in 0..<(margin - 1) {
            XCTAssertLessThan(weights(p), weights(p + 1), "the entry ramp must rise")
        }
        for p in (length - margin)..<(length - 1) {
            XCTAssertGreaterThan(weights(p), weights(p + 1), "the exit ramp must fall")
        }
    }

    func testAnImageEdgeIsNotFaded() {
        // The first tile has nobody on its left: blending it there would create an image that
        // darkens at its own edge, which the normalization would not make up for.
        XCTAssertEqual(VAE.ramp(0, length: 48, margin: 8, freeStart: false, freeEnd: true),
                       1, accuracy: 1e-6)
        XCTAssertEqual(VAE.ramp(47, length: 48, margin: 8, freeStart: true, freeEnd: false),
                       1, accuracy: 1e-6)
    }

    /// **The banded decoder's bands**: the cores tile the rows exactly once, in order; every
    /// window holds its core plus the margin on each side that is not an image edge, stays inside the
    /// image, and starts on an even row — the condition for the bits of the one graph.
    func testTheBandsCoverEveryRowOnceWithTheirMarginOnEvenRows() {
        for rows in [2, 6, 64, 66, 96, 128, 192, 1024, 1536] {
            for core in [1, 2, 3, 7, 32, 64, 100, 5000] {
                let (window, bands) = VAE.bands(rows: rows, core: core, margin: 1)
                var next = 0
                for (c0, c1, start) in bands {
                    XCTAssertEqual(c0, next, "rows \(rows) core \(core)")
                    XCTAssertGreaterThan(c1, c0)
                    XCTAssertEqual(start % VAE.alignment, 0, "rows \(rows) core \(core): window at \(start)")
                    XCTAssertGreaterThanOrEqual(start, 0)
                    XCTAssertLessThanOrEqual(start + window, rows)
                    XCTAssertTrue(start == 0 || c0 - start >= 1, "rows \(rows) core \(core): no margin above \(c0)")
                    XCTAssertTrue(start + window == rows || start + window - c1 >= 1,
                                  "rows \(rows) core \(core): no margin below \(c1)")
                    next = c1
                }
                XCTAssertEqual(next, rows)
            }
        }
    }

    func testWithoutMarginTheFadeIsNeutral() {
        for p in [0, 10, 47] {
            XCTAssertEqual(VAE.ramp(p, length: 48, margin: 0, freeStart: true, freeEnd: true),
                           1, accuracy: 1e-6)
        }
    }

    // ── the decoder: a missing weight ────────────────────────────────────────────────────────

    /// **An incomplete VAE throws, it does not kill the process** (B11). A file of a few bytes,
    /// which has only a convolution's weight and not its bias: the block must name the missing
    /// bias. Only the graph is built — nothing runs on the GPU.
    func testAMissingWeightThrowsAndNamesIt() throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("vae-incomplet-\(UUID().uuidString).safetensors").path
        defer { try? FileManager.default.removeItem(atPath: path) }
        let weights = [Float](repeating: 0, count: 4 * 4)
        try ForgeDiT.writeSafetensors([(name: "c.weight", dtype: "F32", shape: [4, 4, 1, 1],
                                         bytes: weights.withUnsafeBytes { Data($0) })], to: path)
        let graph = MPSGraph()
        let bricks = VAEBuildingBlocks(graph: graph, weights: try Safetensors(path: path), name: "test VAE")
        let x = graph.placeholder(shape: [1, 4, 2, 2], dataType: .float32, name: nil)
        XCTAssertThrowsError(try bricks.conv(x, "c", padding: 0)) { errorMessage in
            guard case Safetensors.Failure.missingTensor(let file, let name)? = errorMessage as? Safetensors.Failure else {
                return XCTFail("expected missingTensor, got \(errorMessage)")
            }
            XCTAssertEqual(file, "test VAE")
            XCTAssertEqual(name, "c.bias")
        }
    }
}
