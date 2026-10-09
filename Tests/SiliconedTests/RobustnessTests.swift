import CoreGraphics
import XCTest
@testable import Siliconed

/// **What a damaged file or an odd input does — a named refusal, never a trap.** Maps written by hand
/// (16-byte pages), prompts as identifiers, images drawn in memory: no byte of weights.
final class RobustnessTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-robust-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    // ── maps ────────────────────────────────────────────────────────────────────────────────

    /// `SILICON` v3, the header's length (or `length` as given), the JSON, the data at `dataOffset`.
    private func map(_ header: [String: Any], data: [Float] = [1, 2, 3, 4], length: UInt64? = nil,
                     dataOffset: Int = 256) throws -> String {
        let json = try JSONSerialization.data(withJSONObject: header)
        var bytes = Array("SILICON".utf8) + [3]
        let declared = length ?? UInt64(json.count)
        for i in 0..<8 { bytes.append(UInt8((declared >> (8 * UInt64(i))) & 0xff)) }
        bytes += json
        precondition(bytes.count <= dataOffset, "header longer than the test's layout")
        bytes += [UInt8](repeating: 0, count: dataOffset - bytes.count)
        data.withUnsafeBytes { bytes += $0 }
        let path = root.appendingPathComponent("m-\(UUID()).silicon").path
        FileManager.default.createFile(atPath: path, contents: Data(bytes))
        return path
    }

    private func header(page: Int = 16, offset: Int = 256, bytes: Int = 16, shape: [Int] = [4]) -> [String: Any] {
        ["page": page, "order": ["w"],
         "tensors": ["w": ["offset": offset, "bytes": bytes, "shape": shape, "dtype": "float32"]]]
    }

    /// The refusal reaches the door as a damaged map, named — whatever the field that lies.
    private func assertCorrupt(_ path: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try Artifact(path: path), file: file, line: line) { error in
            XCTAssertEqual(EngineError(error), .corruptMap(file: path), "\(error)", file: file, line: line)
        }
    }

    func testAValidHandWrittenMapReads() throws {
        let map = try Artifact(path: try map(header()))
        var out = [Float](repeating: 0, count: 4)
        try out.withUnsafeMutableBufferPointer { XCTAssertEqual(try map.materialize("w", into: $0), 4) }
        XCTAssertEqual(out, [1, 2, 3, 4])
    }

    /// The prologue's length with its high bit set: `Int(UInt64)` trapped, `16 + length` overflowed,
    /// and `header(_:)` asked `readData` for it.
    func testAnAberrantHeaderLengthIsRefused() throws {
        for length in [UInt64.max, UInt64(Int.max), 1 << 40, 100_000] {
            let path = try map(header(), length: length)
            assertCorrupt(path)
            XCTAssertThrowsError(try Artifact.header(path)) { XCTAssertEqual(EngineError($0), .corruptMap(file: path)) }
        }
        XCTAssertNil(Artifact.headerLength(UInt64.max, fileSize: 1000))
        XCTAssertNil(Artifact.headerLength(985, fileSize: 1000))
        XCTAssertEqual(Artifact.headerLength(984, fileSize: 1000), 984)
    }

    func testAPageOfZeroOrLessIsRefused() throws {
        assertCorrupt(try map(header(page: 0)))
        assertCorrupt(try map(header(page: -16)))
    }

    /// A negative offset that is a multiple of the page passed the alignment test and pointed
    /// before the map; a negative size passed `offset + bytes <= size`.
    func testNegativeOffsetsAndSizesAreRefused() throws {
        assertCorrupt(try map(header(offset: -16)))
        assertCorrupt(try map(header(offset: -4096)))
        assertCorrupt(try map(header(bytes: -16)))
        assertCorrupt(try map(header(offset: Int.max / 16 * 16)))      // beyond the file, without overflowing
    }

    func testAShapeWhoseCountOverflowsIsRefused() throws {
        assertCorrupt(try map(header(shape: [Int.max, 4])))
        assertCorrupt(try map(header(shape: [-1, -4])))
        XCTAssertNil(Artifact.elementCount([Int.max, 2]))
        XCTAssertNil(Artifact.elementCount([3, -1]))
        XCTAssertEqual(Artifact.elementCount([]), 1)
        XCTAssertEqual(Artifact.elementCount([3, 0]), 0)
    }

    /// Every failure after `mmap` gives the map and the descriptor back: thousands of refused
    /// opens, then a valid one still opens (a leak would end in `EMFILE`, a `cannotOpen`).
    func testARefusedMapLeaksNoDescriptor() throws {
        let bad = try map(header(page: 0))
        for _ in 0..<3000 {
            XCTAssertThrowsError(try Artifact(path: bad)) { error in
                guard case .inFile(_, .badHeader)? = error as? Artifact.Failure else {
                    return XCTFail("\(error)")
                }
            }
        }
        XCTAssertNoThrow(try Artifact(path: try map(header())))
    }

    /// `materialize` judges the destination BEFORE writing: a tensor larger than the buffer is
    /// refused (`doesNotFit`, a damaged map at the door) and not a value lands.
    func testATensorLargerThanItsDestinationIsRefusedBeforeWriting() throws {
        let path = try map(header())
        let map = try Artifact(path: path)
        var out = [Float](repeating: -7, count: 3)
        try out.withUnsafeMutableBufferPointer { buffer in
            XCTAssertThrowsError(try map.materialize("w", into: buffer)) { error in
                XCTAssertEqual(EngineError(error), .corruptMap(file: path))
                guard case .inFile(_, .doesNotFit(name: "w", count: 4, capacity: 3))? = error as? Artifact.Failure else {
                    return XCTFail("\(error)")
                }
            }
            XCTAssertThrowsError(try map.materialize("w", into: buffer.baseAddress!, capacity: 2))
        }
        XCTAssertEqual(out, [-7, -7, -7], "nothing written")
    }

    /// An engine's own mistake is not a damaged file: it folds into `internalFailure`.
    func testAMisuseIsAnInternalFailureNotADamagedMap() {
        XCTAssertEqual(EngineError(Artifact.Failure.misuse("x: 9 tokens requested, 4 reserved")),
                       .internalFailure(component: "engine", detail: "engine: x: 9 tokens requested, 4 reserved"))
    }

    // ── the prompt of Qwen-Image-2.1 ────────────────────────────────────────────────────────

    /// `<|image_pad|>` typed in a prompt (the tokenizer splits it as the slot's special token): a
    /// named refusal, where `expand`'s precondition killed the process.
    func testAnImageSlotSpelledInThePromptIsRefused() throws {
        XCTAssertThrowsError(try Qwen3VLPrompt.admit(ids: [1, 2, 9, 3], image: 9)) {
            XCTAssertEqual($0 as? Qwen3VLPrompt.Failure, .reservedText("<|image_pad|>"))
            XCTAssertEqual(EngineError($0), .promptReservedText(text: "<|image_pad|>"))
        }
        // One slot more than images, one fewer: refused, not trapped.
        XCTAssertThrowsError(try Qwen3VLPrompt.expand([1, 9, 9], image: 9, counts: [4]))
        XCTAssertThrowsError(try Qwen3VLPrompt.expand([1, 9], image: 9, counts: []))
        XCTAssertThrowsError(try Qwen3VLPrompt.expand([1], image: 9, counts: [4]))
        XCTAssertEqual(try Qwen3VLPrompt.expand([1, 2], image: 9, counts: []), [1, 2])
    }

    func testAPromptBeyondItsBoundIsRefusedAtItsBound() throws {
        let max = Qwen3VLPrompt.maxPromptTokens
        XCTAssertNoThrow(try Qwen3VLPrompt.admit(ids: Array(repeating: 1, count: max), image: 9))
        XCTAssertThrowsError(try Qwen3VLPrompt.admit(ids: Array(repeating: 1, count: max + 1), image: 9)) {
            XCTAssertEqual($0 as? Qwen3VLPrompt.Failure, .tooLong(tokens: max + 1, max: max))
            XCTAssertEqual(EngineError($0), .promptTooLong(tokens: max + 1, max: max))
        }
    }

    // ── an image in memory ──────────────────────────────────────────────────────────────────

    private func image(width: Int, height: Int) -> CGImage {
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        for y in 0..<height { for x in 0..<width where (x + y) % 3 == 0 {
            context.setFillColor(CGColor(srgbRed: CGFloat(x % 7) / 7, green: 0.5, blue: CGFloat(y % 5) / 5, alpha: 1))
            context.fill(CGRect(x: x, y: y, width: 1, height: 1))
        } }
        return context.makeImage()!
    }

    /// A pasted image whose side exceeds the bound is drawn at the bound, proportions kept.
    func testAPastedImageIsBoundedLikeAFile() throws {
        let big = try ImageRGB(cgImage: image(width: 300, height: 120), maxSide: 100)
        XCTAssertEqual([big.width, big.height], [100, 40])
        XCTAssertEqual(ImageRGB.decodedSize(width: 20_000, height: 20_000).width, ImageRGB.maximumDecodedSide)
        XCTAssertEqual(ImageRGB.decodedSize(width: 30_000, height: 7).height, 1, "never below one pixel")
        XCTAssertTrue(ImageRGB.decodedSize(width: 3072, height: 2000) == (3072, 2000))
    }

    /// Within the bound, the same bits as before the bound existed (a copy at `.none`).
    func testAnImageWithinTheBoundIsUnchanged() throws {
        let cg = image(width: 37, height: 23)
        let bounded = try ImageRGB(cgImage: cg)
        let unbounded = try ImageRGB(cgImage: cg, maxSide: .max)
        XCTAssertEqual([bounded.width, bounded.height], [37, 23])
        XCTAssertEqual(bounded.pixels.map(\.bitPattern), unbounded.pixels.map(\.bitPattern))
    }

    // ── the rest ────────────────────────────────────────────────────────────────────────────

    /// `:` alone named no file and trapped on `pieces[0]`: it is kept as written, and the render
    /// refuses it as a missing file. A valid stack reads as before.
    func testALoRAEntryWithoutAPathDoesNotTrap() {
        XCTAssertEqual(LoRAEntry.stack(":"), [LoRAEntry(":")])
        XCTAssertEqual(LoRAEntry.stack("a.silicon:0.5,::"), [LoRAEntry("a.silicon", strength: 0.5), LoRAEntry("::")])
    }

    /// Under `SILICONED_DIVISORS`: a power of two dividing both sides, or no descent.
    func testTheSpectralDivisorsMustDivideTheGrid() {
        XCTAssertTrue(Sampler.divides([1, 1], height: 0, width: 0), "no descent, no grid needed")
        XCTAssertTrue(Sampler.divides([2, 2, 1], height: 128, width: 152))
        XCTAssertTrue(Sampler.divides([4, 2], height: 256, width: 256))
        XCTAssertFalse(Sampler.divides([3], height: 192, width: 192), "not a power of two")
        XCTAssertFalse(Sampler.divides([2], height: 129, width: 128), "an odd side")
        XCTAssertFalse(Sampler.divides([4], height: 130, width: 128), "halved once, not twice")
        XCTAssertFalse(Sampler.divides([0, 2], height: 128, width: 128))
    }

    /// A golden is not installed: its message names the oracle that writes it.
    func testAMissingGoldenNamesItsOracle() {
        let qwen = MissingFile("/repo/bench/oracle/qwen21/goldens-qwen21-dit-512.safetensors", .published)
        XCTAssertEqual(qwen.nature, .golden)
        XCTAssertTrue(qwen.description.contains("tools/machine.sh tools/.venv/bin/python bench/oracle/qwen21/oracle_qwen21.py"),
                      qwen.description)
        XCTAssertFalse(qwen.description.contains("siliconed-dev install"))
        XCTAssertEqual(MissingFile.oracle(forGolden: "/repo/bench/oracle/goldens-vae-encode-512x768.safetensors"),
                       "bench/oracle/oracle_vae_encode.py")
        XCTAssertEqual(MissingFile.oracle(forGolden: "/repo/bench/oracle/goldens-trajectory-512.safetensors"),
                       "bench/oracle/oracle_trajectory.py")
        XCTAssertEqual(MissingFile("/lib/store/composants/z/vae.safetensors", .published).nature, .published)
    }

    /// Messages: the bound of a strength is said as it is (0 excluded), and a strength just inside
    /// it does not print as the bound; the largest format comes from `Format.maxSurface`.
    func testMessagesSayWhatTheBoundsAre() {
        let tiny = EngineError.strengthOutOfRange(strength: 0.001).errorDescription!
        XCTAssertTrue(tiny.contains("0.001") && tiny.contains("above 0"), tiny)
        XCTAssertTrue(EngineError.strengthTooLow(strength: 0.1, steps: 8, floor: 0.125).errorDescription!.contains("0.1 "))
        let large = EngineError.formatRefused(width: 2048, height: 2048, reason: .tooLarge).errorDescription!
        XCTAssertTrue(large.contains(String(format: "%.2f", Double(Format.maxSurface) / 1e6)), large)
        for e in EngineError.samples {
            let text = [e.errorDescription, e.recoverySuggestion].compactMap { $0 }.joined()
            for symbol in ["Library.cards()", "ModelCard.formats", "Model.named", "loras(for:)", "huggingface-cli", "is resumed"] {
                XCTAssertFalse(text.contains(symbol), "\(e.code): \(text)")
            }
        }
    }

    /// One lock for the machine, whatever the library or the user — and a lock file another user
    /// created (not writable for us) still locks: it is opened read-only.
    func testTheRenderLockIsTheMachines() throws {
        XCTAssertEqual(RenderLock.machine, "/private/tmp/siliconed-render.lock")
        let path = root.appendingPathComponent("render.lock").path
        FileManager.default.createFile(atPath: path, contents: nil, attributes: [.posixPermissions: 0o444])
        var first = try RenderLock.acquire(path)
        XCTAssertNotNil(first)
        XCTAssertThrowsError(try RenderLock.acquire(path)) { XCTAssertEqual($0 as? EngineError, .renderAlreadyRunning) }
        first = nil
        XCTAssertNotNil(try RenderLock.acquire(path))
    }
}
