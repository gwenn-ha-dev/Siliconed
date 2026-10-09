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
    private let klein = Model(card: .klein4b, chain: try! Chain(
        text: KleinTextModule(map: "", tokenizer: ""), denoising: KleinDenoisingModule(map: ""),
        decoding: Flux2DecodingModule(path: ""), encoding: Flux2EncodingModule(path: "")))

    /// B1: with a single step, the schedule of the three models is finite (Anima rendered σ₀ = NaN).
    /// **A request is comparable**: a SwiftUI app relaunches a preview on `onChange(of: request)`.
    /// Two requests equal at construction, then the seed, then the image, tell them apart.
    /// **Where a sketch stops**: σ, not a count — Z-Image's 8 steps stop after 3 evaluations
    /// (σ .800), Qwen-Image-2.1's 6 after 4 (σ .632); a null step is no evaluation; a plan sketched
    /// keeps its first evaluations.
    func testASketchStopsWhereSigmaReachesTheThreshold() {
        XCTAssertEqual(Sketch.evaluations(sigmas: FlowMatchSchedule(steps: 8).sigmas), 3)
        XCTAssertEqual(zImage.sketchEvaluations(steps: nil, width: 512, height: 512), 3)
        XCTAssertEqual(Sketch.evaluations(sigmas: [1, 0.963, 0.923, 0.837, 0.632, 0.3, 0]), 4)
        XCTAssertEqual(Sketch.evaluations(sigmas: [1, 1, 0.9, 0.5, 0]), 2)
        XCTAssertEqual(Sketch.evaluations(sigmas: [1, 0]), 1)
        let plan = zImage.chain.denoising.plan(height: 128, width: 128, steps: 8, start: 0)
        XCTAssertEqual([plan.evaluations, plan.reduced], [7, 2])
        XCTAssertEqual([plan.sketched(3).evaluations, plan.sketched(3).reduced], [3, 2])
        XCTAssertEqual([plan.sketched(1).evaluations, plan.sketched(1).reduced], [1, 1])
        XCTAssertEqual(plan.sketched(99).evaluations, 7)
    }

    /// **Qwen-Image-2.1 does not sketch** (its x̂₀ at 4 and 5 of 6 is blurred and grainy): its
    /// « automatic » stop is the plan's end, a finished image; Z-Image still stops at 3.
    func testQwenSketchesAreFinishedImages() {
        let qwen = Model(card: .qwenImage21, chain: try! Chain(
            text: QwenImage21TextModule(map: "", tokenizer: ""),
            denoising: QwenImage21DenoisingModule(map: "", turbo: "", scheduler: .init()),
            decoding: QwenImage21DecodingModule(path: ""), encoding: QwenImage21EncodingModule(path: "")))
        let d = qwen.chain.denoising
        XCTAssertFalse(d.sketches)
        XCTAssertEqual(qwen.sketchEvaluations(steps: nil, width: 512, height: 512),
                       d.plan(height: 32, width: 32, steps: d.defaultSteps, start: 0).evaluations)
        XCTAssertTrue(zImage.chain.denoising.sketches)
    }

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
        for m in [zImage, anima, krea2, klein] {
            let σ = m.chain.denoising.sigmas(steps: 1)
            XCTAssertEqual(σ.count, 2, m.identifier)
            XCTAssertTrue(σ.allSatisfy(\.isFinite), "\(m.identifier) : \(σ)")
            XCTAssertEqual(σ.last, 0)
        }
    }

    /// B1: fewer than one step is refused before any computation, with a message about steps (no
    /// more `Range` crash at step < 0, nor a message about "strength" at step = 0).
    func testFewerThanOneStepIsRefused() {
        for m in [zImage, anima, krea2, klein] {
            for n in [0, -3] {
                XCTAssertThrowsError(try Engine().render(Request("a 30 year old woman posing in a library", resolution: 512, steps: n), model: m)) {
                    XCTAssertEqual($0 as? EngineError, .invalidSteps(steps: n, allowed: []), "\(m.identifier) : \($0)")
                }
            }
        }
    }

    /// A reference (editing) is refused before any computation by a model that does not read any —
    /// and beyond what FLUX.2 [klein] reads.
    func testReferencesAreRefusedBeforeAnyComputation() {
        let image = ImageRGB(pixels: [Float](repeating: 0, count: 3 * 64 * 64), height: 64, width: 64)
        for (m, n, max) in [(zImage, 1, 0), (krea2, 1, 0), (klein, 5, 4)] {
            var r = Request("une bibliothèque", resolution: 512)
            r.references = Array(repeating: image, count: n)
            XCTAssertThrowsError(try Engine().render(r, model: m)) {
                XCTAssertEqual($0 as? EngineError, .tooManyReferences(count: n, max: max), "\(m.identifier) : \($0)")
            }
        }
        XCTAssertEqual(klein.chain.entries.map(\.kind), [.text, .image, .reference])
        XCTAssertEqual(zImage.chain.entries.map(\.kind), [.text, .image])
    }

    /// The size of a FLUX.2 reference: its area under 1024², its sides at a multiple of 16.
    func testTheSizeOfAReference() {
        let photo = ImageRGB(pixels: [Float](repeating: 0, count: 3 * 3000 * 2000), height: 2000, width: 3000)
        let r = photo.forReference()
        XCTAssertEqual([r.width, r.height], [1248, 832])
        XCTAssertLessThanOrEqual(r.width * r.height, 1024 * 1024)
        let square = ImageRGB(pixels: [Float](repeating: 0, count: 3 * 512 * 512), height: 512, width: 512)
        XCTAssertEqual(square.forReference(), square)
    }

    /// An empty or blank prompt is refused by the three models, before any computation.
    func testTheEmptyPromptIsRefusedByAllThree() {
        for m in [zImage, anima, krea2, klein] {
            for prompt in ["", "  \n "] {
                XCTAssertThrowsError(try Engine().render(Request(prompt, resolution: 512), model: m)) {
                    XCTAssertEqual($0 as? EngineError, .emptyPrompt, "\(m.identifier) : \($0)")
                }
            }
        }
    }

    /// B2: building a request or the product's denoiser does not read the machine's profile.
    func testNothingReadsTheProfileAtConstruction() {
        XCTAssertNil(Request("p").reproducible, "resolved at render time, not frozen at construction")
        XCTAssertNil(ZImageDenoisingModule.product(map: "").spectral)
        XCTAssertEqual(Request("p", previews: true).previews, true)
    }

    /// B7: an absurd format raises `tooLarge` instead of overflowing at the multiplication.
    func testAnAbsurdFormatDoesNotOverflow() {
        XCTAssertThrowsError(try Format.check(width: 99_999_999_984, height: 99_999_999_984)) {
            XCTAssertEqual($0 as? EngineError, .formatRefused(width: 99_999_999_984, height: 99_999_999_984, reason: .tooLarge))
        }
        XCTAssertThrowsError(try Format.check(width: 512, height: 16_000))
    }

    /// B9: a spectral forced below the floor (512² → 256²) is refused before the text encoder.
    func testAForcedSpectralBelowTheFloorIsRefusedBeforeAnyComputation() throws {
        let forced = Model(card: .zImage, chain: try Chain(
            text: ZImageTextModule(map: "", tokenizer: ""), denoising: ZImageDenoisingModule(map: "", spectral: 2),
            decoding: FluxDecodingModule(path: "")))
        XCTAssertThrowsError(try Engine().render(Request("a 30 year old woman posing in a library", resolution: 512), model: forced)) {
            XCTAssertEqual($0 as? EngineError, .imageTooSmall(side: 256, minimum: 512))
        }
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
        let largeImage = ImageRGB(pixels: (0..<(3 * 800 * 600)).map { Float($0 % 7) / 7 }, height: 600, width: 800)
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
