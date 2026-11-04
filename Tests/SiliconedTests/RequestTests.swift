import CoreGraphics
import Foundation
import XCTest
@testable import Siliconed

/// **What a request refuses, and what it does not read** — the bugs from the API review.
/// The chains have empty paths: if the engine touched a file before refusing,
/// the error would be `fileMissing`, not the one we expect.
final class RequestTests: XCTestCase {

    /// These renders stop before any weight: the preflight's memory is not what they test, and the
    /// machine's state would make them depend on what else runs (`MemoryBudget`). A budget that fits all.
    override class func setUp() {
        super.setUp()
        setenv(MemoryBudget.forcingVariable, "64", 1)
    }

    private let zImage = Model(card: .zImage, chain: try! Chain(
        text: ZImageTextModule(map: "", tokenizer: ""), denoising: ZImageDenoisingModule(map: ""),
        decoding: FluxDecodingModule(path: ""), encoding: FluxEncodingModule(path: "")))
    private let anima = Model(card: .anima, chain: try! Chain(
        text: AnimaTextModule(map: "", tokenizerQwen: "", tokenizerT5: "", adapter: ""),
        denoising: AnimaDenoisingModule(map: ""), decoding: QwenImageDecodingModule(path: ""),
        encoding: QwenImageEncodingModule(path: "")))
    private let krea2 = Model(card: .krea2, chain: try! Chain(
        text: Krea2TextModule(map: "", tokenizer: "", sockets: []), denoising: Krea2DenoisingModule(map: ""),
        decoding: QwenImageDecodingModule(path: ""), encoding: QwenImageEncodingModule(path: "")))
    private let qwen = Model(card: .qwenImage21, chain: try! Chain(
        text: QwenImage21TextModule(map: "", tokenizer: ""),
        denoising: QwenImage21DenoisingModule(map: "", scheduler: .init()),
        decoding: QwenImage21DecodingModule(path: ""), encoding: QwenImage21EncodingModule(path: "")))

    /// B1: with a single step, the schedule of the three models is finite (Anima rendered σ₀ = NaN).
    /// **A request is comparable**: a SwiftUI app relaunches a preview on `onChange(of: request)`.
    /// Two requests equal at construction, then the seed, then the image, tell them apart.
    func testARequestIsComparable() {
        let a = Request("une bibliothèque", resolution: 512)
        var b = Request("une bibliothèque", resolution: 512)
        XCTAssertEqual(a, b)
        b.seed = 7
        XCTAssertNotEqual(a, b)
        b.seed = a.seed
        b.image = ImageRGB(pixels: [Float](repeating: 0, count: 3 * 512 * 512), height: 512, width: 512)
        XCTAssertNotEqual(a, b)
    }

    func testASingleStepGivesFiniteSigmas() {
        for m in [zImage, anima, krea2] {
            let σ = m.chain.denoising.sigmas(steps: 1)
            XCTAssertEqual(σ.count, 2, m.identifier)
            XCTAssertTrue(σ.allSatisfy(\.isFinite), "\(m.identifier) : \(σ)")
            XCTAssertEqual(σ.last, 0)
        }
    }

    /// B1: fewer than one step is refused before any computation, with a message about steps (no
    /// more `Range` crash at step < 0, nor a message about "strength" at step = 0).
    func testFewerThanOneStepIsRefused() {
        for m in [zImage, anima, krea2] {
            for n in [0, -3] {
                XCTAssertThrowsError(try Engine().render(Request("a 30 year old woman posing in a library", resolution: 512, steps: n), model: m)) {
                    XCTAssertEqual($0 as? EngineError, .invalidSteps(steps: n, allowed: []), "\(m.identifier) : \($0)")
                }
            }
        }
    }

    /// A reference (editing) is refused before any computation by a model that does not read any —
    /// and beyond what Qwen-Image-2.1 reads.
    func testReferencesAreRefusedBeforeAnyComputation() {
        let image = ImageRGB(pixels: [Float](repeating: 0, count: 3 * 64 * 64), height: 64, width: 64)
        for (m, n, max) in [(zImage, 1, 0), (krea2, 1, 0), (qwen, 4, 3)] {
            var r = Request("une bibliothèque", resolution: 512)
            r.references = Array(repeating: image, count: n)
            XCTAssertThrowsError(try Engine().render(r, model: m)) {
                XCTAssertEqual($0 as? EngineError, .tooManyReferences(count: n, max: max), "\(m.identifier) : \($0)")
            }
        }
        XCTAssertEqual(qwen.chain.entries.map(\.kind), [.text, .reference])
        XCTAssertEqual(zImage.chain.entries.map(\.kind), [.text, .image])
    }

    /// An empty or blank prompt is refused by the three models, before any computation.
    func testTheEmptyPromptIsRefusedByAllThree() {
        for m in [zImage, anima, krea2] {
            for prompt in ["", "  \n "] {
                XCTAssertThrowsError(try Engine().render(Request(prompt, resolution: 512), model: m)) {
                    XCTAssertEqual($0 as? EngineError, .emptyPrompt, "\(m.identifier) : \($0)")
                }
            }
        }
    }

    /// B2: building a request does not read the machine's profile.
    func testNothingReadsTheProfileAtConstruction() {
        XCTAssertEqual(Request("p", previews: true).previews, true)
    }

    /// B7: an absurd format raises `sideTooLong` instead of overflowing at the multiplication — each
    /// side is bounded before anything is multiplied.
    func testAnAbsurdFormatDoesNotOverflow() {
        XCTAssertThrowsError(try Format.check(width: 99_999_999_984, height: 99_999_999_984)) {
            XCTAssertEqual($0 as? EngineError, .formatRefused(width: 99_999_999_984, height: 99_999_999_984, reason: .sideTooLong, bound: Format.maximumSide))
        }
        XCTAssertThrowsError(try Format.check(width: 512, height: 16_000))
    }

    /// B10: a request cancelled before starting pays nothing and emits nothing, not even `start`.
    func testAnAlreadyCancelledRequestEmitsNothing() {
        let token = Cancellation()
        token.cancel()
        let receivedEvents = Box(0)
        XCTAssertThrowsError(try Engine().render(Request("a 30 year old woman posing in a library", resolution: 512), model: zImage,
                                                 cancellation: token) { _ in receivedEvents.increment() }) {
            XCTAssertEqual($0 as? EngineError, .cancelled, "\($0)")
        }
        XCTAssertEqual(receivedEvents.value, 0)
    }

    /// B4: the request fits the image as soon as it receives it — at init, at assignment, and when
    /// the format changes — and an image already at the right format stays bit-identical.
    func testTheImageIsFittedFromTheEntry() {
        // 800×600: larger than both formats (512², 768×512), so each fit crops and scales down.
        let largeImage = ImageRGB(pixels: (0..<(3 * 800 * 600)).map { (i: Int) -> Float in Float(i % 7) / 7 }, height: 600, width: 800)
        var r = Request("p", resolution: 512, image: largeImage)
        XCTAssertEqual(r.image.map { [$0.width, $0.height] }, [512, 512])
        r.width = 768
        XCTAssertEqual(r.image.map { [$0.width, $0.height] }, [768, 512])
        r.image = largeImage
        XCTAssertEqual(r.image.map { [$0.width, $0.height] }, [768, 512])
        let atFormat = largeImage.fitted(width: 512, height: 512)
        XCTAssertEqual(Request("p", resolution: 512, image: atFormat).image?.pixels, atFormat.pixels)
    }

    /// `ImageRGB(cgImage:)`: the round trip of an 8-bit image is exact.
    func testAnInMemoryImageRoundTrips() throws {
        let bytes: [Float] = [0, 1, 0.5, 0.25, 0.75, 0.125].map { ($0 * 255).rounded() / 255 * 2 - 1 }
        let image = ImageRGB(pixels: bytes, height: 1, width: 2)
        let reread = try ImageRGB(cgImage: image.cgImage())
        XCTAssertEqual(reread.width, 2); XCTAssertEqual(reread.height, 1)
        for (a, b) in zip(reread.pixels, image.pixels) { XCTAssertEqual(a, b, accuracy: 1e-6) }
    }
}

/// A counter that a `@Sendable` closure increments.
private final class Box: @unchecked Sendable {
    private let lock = NSLock()
    private var n: Int
    init(_ n: Int) { self.n = n }
    func increment() { lock.lock(); n += 1; lock.unlock() }
    var value: Int { lock.lock(); defer { lock.unlock() }; return n }
}
