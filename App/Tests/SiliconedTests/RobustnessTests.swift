import CoreGraphics
import XCTest
@testable import Siliconed

/// **What an odd input does — a named refusal, never a trap.** Prompts as identifiers, images drawn in memory: no byte
/// of weights.
final class RobustnessTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-robust-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    // ── a prompt beyond its bound ───────────────────────────────────────────────────────────

    /// Z-Image, Anima and Krea 2 truncate where their references do; the engine refuses there instead.
    func testEveryFamilysPromptIsRefusedRatherThanCut() throws {
        XCTAssertNoThrow(try Request.admitPrompt(tokens: 504, max: 504))
        XCTAssertThrowsError(try Request.admitPrompt(tokens: 505, max: 504)) {
            XCTAssertEqual($0 as? Request.Failure, .promptTooLong(tokens: 505, max: 504))
            XCTAssertEqual(EngineError($0), .promptTooLong(tokens: 505, max: 504))
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
        XCTAssertEqual(ImageRGB.decodedSize(width: 100_000, height: 7).height, 1, "never below one pixel")
        XCTAssertTrue(ImageRGB.decodedSize(width: 3072, height: 2000) == (3072, 2000))
        XCTAssertTrue(ImageRGB.decodedSize(width: 8192, height: 5000) == (8192, 5000), "the longest side a format has")
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

    /// A golden is not installed: its message says the reference oracle writes it, never « install ».
    func testAMissingGoldenIsNotInstalled() {
        let qwen = MissingFile("/repo/goldens/goldens-qwen21-dit-512.safetensors", .published)
        XCTAssertEqual(qwen.nature, .golden)
        XCTAssertTrue(qwen.description.contains("written by the reference oracle"), qwen.description)
        XCTAssertFalse(qwen.description.contains("install the model"))
        XCTAssertEqual(MissingFile("/lib/store/composants/z/vae.safetensors", .published).nature, .published)
    }

    /// Messages: the bound of a strength is said as it is (0 excluded), and a strength just inside
    /// it does not print as the bound.
    func testMessagesSayWhatTheBoundsAre() {
        let tiny = EngineError.strengthOutOfRange(strength: 0.001).errorDescription!
        XCTAssertTrue(tiny.contains("0.001") && tiny.contains("above 0"), tiny)
        XCTAssertTrue(EngineError.strengthTooLow(strength: 0.1, steps: 8, floor: 0.125).errorDescription!.contains("0.1 "))
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
        var first = try RenderLock.acquire(path) { _ in }
        XCTAssertNotNil(first)
        XCTAssertThrowsError(try RenderLock.acquire(path) { _ in }) { XCTAssertEqual($0 as? EngineError, .renderAlreadyRunning) }
        first = nil
        XCTAssertNotNil(try RenderLock.acquire(path) { _ in })
    }
}
