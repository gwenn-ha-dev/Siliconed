import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import Measurement
@testable import Siliconed

/// **`silicontrol add` and `grid` as the app reads them** (`AddOrder`), without an app: the prompt, the size and the seed in
/// their places, the lone `--`, a flag it does not take; and `silicontrol measure`'s renders, which are `add` lines.
final class AddOrderTests: XCTestCase {
    private func order(_ arguments: [String]) throws -> AddOrder { try AddOrder.parse(arguments).get() }

    func testThePromptThenASizeAndASeedInEitherOrder() throws {
        var d = try order(["a lighthouse at dusk", "832x1216", "7"])
        XCTAssertEqual(d.prompt, "a lighthouse at dusk")
        XCTAssertEqual(d.format?.width, 832); XCTAssertEqual(d.format?.height, 1216)
        XCTAssertEqual(d.seed, 7)
        d = try order(["a cat", "7", "512"])
        XCTAssertEqual(d.format?.width, 512); XCTAssertEqual(d.format?.height, 512)
        XCTAssertEqual(d.seed, 7)
        // A proportion and a resolution: the rack's two-scale Size.
        d = try order(["a cat", "3:4@2K"])
        let cell = try SizeScales.parse("3:4@2K")
        XCTAssertEqual(d.format?.width, cell.width); XCTAssertEqual(d.format?.height, cell.height)
        XCTAssertNil(d.seed)
        // The model's default and a drawn seed are the app's: nothing typed, nothing read.
        d = try order(["a cat"])
        XCTAssertNil(d.format); XCTAssertNil(d.seed)
        XCTAssertEqual(d.model, ModelCard.of(.default).id)
    }

    /// Flags anywhere, before the prompt or after it; the last value of one given twice, except those that add up.
    func testFlagsAnywhere() throws {
        let d = try order(["--steps", "4", "a cat", "1024", "--lora", "flat:0.5", "--wait", "--lora", "ink", "--steps", "6", "--no-preview"])
        XCTAssertEqual(d.prompt, "a cat")
        XCTAssertEqual(d.steps, 6)
        XCTAssertEqual(d.loras.map(\.name), ["flat", "ink"])
        XCTAssertEqual(d.loras.map(\.strength), [0.5, 1])
        XCTAssertTrue(d.wait)
        XCTAssertEqual(d.previews, false)
        XCTAssertEqual(d.format?.width, 1024)
    }

    /// **A prompt that begins with `--` goes after a lone `--`**, and the size and the seed may follow it there; before it, a word
    /// of two dashes is a flag, refused by name when `add` does not take it.
    func testTheLoneSeparator() throws {
        let d = try order(["--wait", "--batch", "2", "--", "--ar 16:9 a cat", "1024", "42"])
        XCTAssertEqual(d.prompt, "--ar 16:9 a cat")
        XCTAssertEqual(d.format?.width, 1024)
        XCTAssertEqual(d.seed, 42)
        XCTAssertEqual(d.batch, 2)
        XCTAssertTrue(d.batchGiven)
        XCTAssertTrue(d.wait)
        // After the `--`, even what reads as a flag is a word — here, one too many —, never a switch.
        XCTAssertEqual(AddOrder.parse(["a cat", "--", "--wait"]).failure, .extraArgument("--wait"))
        XCTAssertEqual(AddOrder.parse(["--ar 16:9 a cat"]).failure, .unknownFlag("--ar 16:9 a cat"))
        XCTAssertEqual(AddOrder.parse(["--ar", "16:9", "a cat"]).failure, .unknownFlag("--ar"))
        XCTAssertEqual(AddOrder.parse(["a cat", "--seed", "4"]).failure, .unknownFlag("--seed"), "the seed is a word, not a flag")
    }

    func testWhatItRefuses() {
        XCTAssertEqual(AddOrder.parse([]).failure, .promptMissing)
        XCTAssertEqual(AddOrder.parse(["--wait"]).failure, .promptMissing)
        XCTAssertEqual(AddOrder.parse(["a cat", "--steps"]).failure, .flagWithoutValue("--steps"))
        XCTAssertEqual(AddOrder.parse(["a cat", "--batch", "9"]).failure, .badValue(.batch, "9"))
        XCTAssertEqual(AddOrder.parse(["a cat", "--steps", "0"]).failure, .badValue(.steps, "0"))
        XCTAssertEqual(AddOrder.parse(["a cat", "--detail", "max"]).failure, .badValue(.detail, "max"))
        XCTAssertEqual(AddOrder.parse(["a cat", "--solver", "heun"]).failure, .badValue(.solver, "heun"))
        XCTAssertEqual(AddOrder.parse(["a cat", "--lora", "flat:nan"]).failure, .badValue(.lora, "flat:nan"))
        XCTAssertEqual(AddOrder.parse(["a cat", "--strength", "2"]).failure, .badValue(.strength, "2"))
        XCTAssertEqual(AddOrder.parse(["a cat", "1:1@3K"]).failure, .unknownResolution("3K"))
        XCTAssertEqual(AddOrder.parse(["a cat", "8x"]).failure, .unreadableFormat("8x"))
        XCTAssertEqual(AddOrder.parse(["a", "cat"]).failure, .extraArgument("cat"), "the prompt is one argument")
        XCTAssertEqual(AddOrder.parse(["a cat", "512", "7", "8"]).failure, .extraArgument("8"))
    }

    /// **`measure` sends its renders as `add` lines, and `add` reads them back whole**: a `--prompt` that begins with `--`, the
    /// size and the seed after a lone `--`, the passthrough's flags before it. The fake app reads each line as the app does.
    func testMeasureRendersAreAddLinesTheAppReads() throws {
        var read: [AddOrder] = []
        let send: Runner.Send = { argv, onLine in
            switch argv[0] {
            case "status": return (["ok": true, "queue": [], "neural_engine": false], getpid())
            case "models": return (["ok": true, "models": [["id": "fake", "ready": true]]], getpid())
            default:
                XCTAssertEqual(argv[0], "add")
                guard case .success(let d) = AddOrder.parse(Array(argv.dropFirst())) else {
                    return (["ok": false, "error": "usage", "message": "\(argv)"], getpid())
                }
                read.append(d)
                let path = URL(fileURLWithPath: d.folder!, isDirectory: true).appendingPathComponent("render-\(read.count).png")
                Self.writePNG(path)
                onLine(["type": "image", "seconds": 0.5, "steps": d.steps ?? 8, "path": path.path])
                return (["ok": true], getpid())
            }
        }
        let o = try MeasureOptions.parse(["--model", "fake", "--size", "832x1216", "--seed", "7", "--renders", "1", "--settle", "0",
                                          "--prompt", "--ar 16:9 a cat", "--", "--steps", "4"])
        _ = try Runner.run(o, send: send, log: { _ in }, emit: { _ in }, append: { _ in })
        XCTAssertEqual(read.map(\.prompt), [MeasureOptions.defaultPrompt, MeasureOptions.defaultPrompt, "--ar 16:9 a cat",
                                            MeasureOptions.defaultPrompt])
        XCTAssertEqual(read.map(\.seed), [42, 42, 7, 42])
        XCTAssertTrue(read.allSatisfy { $0.model == "fake" && $0.steps == 4 && $0.wait && $0.format?.width == 832 && $0.format?.height == 1216 })
    }

    private static func writePNG(_ url: URL) {
        var pixels = [UInt8](repeating: 128, count: 4 * 4 * 4)
        let context = pixels.withUnsafeMutableBytes {
            CGContext(data: $0.baseAddress, width: 4, height: 4, bitsPerComponent: 8, bytesPerRow: 16,
                      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        }
        let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, context.makeImage()!, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
    }

    /// **An image lost in a job of several seeds** is a warning of the job's answer, by its seed and its stage, in the engine's
    /// words; the window knows it from the engine's other warnings.
    func testALostImageIsSaidByItsSeedAndItsStage() {
        let vae = LostImage.warning(seed: 43, stage: .decoding)
        XCTAssertEqual(vae, "non_finite_image: seed 43: This render went wrong in the VAE and its image is unusable: it was not kept.")
        XCTAssertEqual(LostImage.warning(seed: 7, stage: .denoising),
                       "non_finite_image: seed 7: This render went wrong in the DiT and its image is unusable: it was not kept.")
        XCTAssertTrue(LostImage.isWarning(vae))
        XCTAssertFalse(LostImage.isWarning(bandDecodingNotice))
        XCTAssertFalse(LostImage.isWarning(sharedFileRemovedNotice + ": x.safetensors"))
    }
}

private extension Result {
    var failure: Failure? { if case .failure(let f) = self { f } else { nil } }
}
