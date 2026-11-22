import Foundation
import XCTest
@testable import Siliconed

/// **The render every family runs** (`RenderLoop`), on the CPU alone: a family's part whose text, DiT and decoder are
/// fakes — the DiT's derivative zero, or NaN for the evaluations a test says, the decoder's picture zero or NaN. What is
/// judged is the loop's: a latent gone non-finite is said `.unusable` at the denoising and never decoded, the images
/// keep their index in the batch, image 0's times hold the text and the DiT's opening and the others' do not, a
/// cancellation stops the loop, and a batch left with no image fails at the stage where it was lost.
final class RenderLoopTests: XCTestCase {
    /// What the fakes do, set by each test before it runs the loop.
    fileprivate struct Script {
        /// The DiT's evaluations (counted across the batch) whose derivative is NaN.
        var nonFinite: Set<Int> = []
        /// The evaluation whose call raises the cancellation.
        var cancelAt: Int?
        /// The decoder's pictures NaN.
        var decodesNaN = false
        var cancellation: Cancellation?
        var evaluations = 0
        var decoderOpened = 0
        var decoded = 0
    }
    nonisolated(unsafe) fileprivate static var script = Script()

    /// The text and the DiT's opening, each as long: what image 0's times hold and no other's.
    fileprivate static let pause = 0.02

    override func setUp() { Self.script = Script() }

    private func model() throws -> Model {
        Model(card: .zImage, chain: try Chain(text: LoopText(), denoising: LoopDenoising(), decoding: LoopDecoding()))
    }

    private func run(seeds: [UInt64]) throws -> (renders: [Engine.Render], events: [Engine.Event]) {
        let model = try model(), request = Request("a prompt", width: 64, height: 64, steps: 2)
        let admitted = try Admission.judge(request, chain: model.chain)
        var events: [Engine.Event] = []
        let renders = try RenderLoop.run(LoopParts.self, request, admitted, seeds: seeds, model: model,
                                         cancellation: Self.script.cancellation) { events.append($0) }
        return (renders, events)
    }

    /// Two evaluations an image: the second image's DiT gives NaN, the first's and the third's zero.
    func testANonFiniteLatentIsUnusableAndNotDecoded() throws {
        Self.script.nonFinite = [2, 3]
        let (renders, events) = try run(seeds: [1, 2, 3])
        XCTAssertEqual(renders.map(\.seed), [1, 3], "the images that were finite, in their order")
        XCTAssertEqual(Self.script.evaluations, 6, "every image denoised")
        XCTAssertEqual(Self.script.decoderOpened, 1, "one decoder for the batch")
        XCTAssertEqual(Self.script.decoded, 2, "the lost image never reaches the decoder")
        var unusable: [(Int, Engine.Stage)] = [], images: [Int] = [], decoding: [Int] = []
        for event in events {
            switch event {
            case let .unusable(image, stage): unusable.append((image, stage))
            case let .image(index, _): images.append(index)
            case let .stage(.decoding, image): decoding.append(image)
            default: break
            }
        }
        XCTAssertEqual(unusable.map(\.0), [1])
        XCTAssertEqual(unusable.map(\.1), [.denoising])
        XCTAssertEqual(images, [0, 2], "each image keeps its index in the batch")
        XCTAssertEqual(decoding, [0, 2], "the decoding announced for the images decoded alone")
    }

    /// Image 0's times hold the text and the DiT's opening; another's, its own denoising and decoding alone.
    func testImageZerosTimesHoldTheTextAndTheOpening() throws {
        let (renders, _) = try run(seeds: [1, 2])
        XCTAssertEqual(renders.count, 2)
        let (first, second) = (renders[0].timings, renders[1].timings)
        XCTAssertGreaterThanOrEqual(first.tokenizer, Self.pause)
        XCTAssertGreaterThanOrEqual(first.text, first.tokenizer)
        XCTAssertGreaterThanOrEqual(first.denoising, Self.pause, "the DiT's opening is image 0's denoising")
        XCTAssertGreaterThanOrEqual(first.total, first.text + first.denoising + first.decoding)
        XCTAssertEqual(second.text, 0)
        XCTAssertEqual(second.tokenizer, 0)
        XCTAssertLessThan(second.denoising, Self.pause, "no opening in another image's denoising")
        XCTAssertGreaterThanOrEqual(second.total, second.denoising + second.decoding)
    }

    /// A cancellation raised during the denoising stops the loop at its next check: nothing more is evaluated or decoded.
    func testACancellationStopsTheLoop() throws {
        let cancellation = Cancellation()
        Self.script.cancellation = cancellation
        Self.script.cancelAt = 2
        XCTAssertThrowsError(try run(seeds: [1, 2, 3])) { XCTAssertEqual($0 as? EngineError, .cancelled) }
        XCTAssertEqual(Self.script.evaluations, 3)
        XCTAssertEqual(Self.script.decoderOpened, 0)
    }

    /// Every latent lost at the DiT: the batch fails there, no decoder opened.
    func testABatchLostAtTheDiTFailsThere() throws {
        Self.script.nonFinite = Set(0..<4)
        XCTAssertThrowsError(try run(seeds: [1, 2])) { XCTAssertEqual($0 as? EngineError, .nonFiniteImage(stage: .denoising)) }
        XCTAssertEqual(Self.script.decoderOpened, 0)
    }

    /// Every picture lost at the decoder: each said unusable there, the batch failing at the decoding.
    func testABatchLostAtTheDecoderFailsThere() throws {
        Self.script.decodesNaN = true
        var unusable: [(Int, Engine.Stage)] = []
        XCTAssertThrowsError(try {
            let model = try model(), request = Request("a prompt", width: 64, height: 64, steps: 2)
            _ = try RenderLoop.run(LoopParts.self, request, try Admission.judge(request, chain: model.chain), seeds: [1, 2], model: model,
                                   cancellation: nil) { if case let .unusable(image, stage) = $0 { unusable.append((image, stage)) } }
        }()) { XCTAssertEqual($0 as? EngineError, .nonFiniteImage(stage: .decoding)) }
        XCTAssertEqual(unusable.map(\.0), [0, 1])
        XCTAssertEqual(unusable.map(\.1), [.decoding, .decoding])
    }
}

private struct LoopText: TextModule {
    var output: TextFormat { TextFormat("loop") }
}

/// Two steps, two evaluations.
private struct LoopDenoising: DenoisingModule {
    var model: String { "loop" }
    var entry: TextFormat { TextFormat("loop") }
    var space: LatentSpace { .flux }
    var defaultSteps: Int { 2 }
    func sigmas(steps: Int) -> [Float] { [1, 0.5, 0] }
    func plan(height: Int, width: Int, steps: Int, start: Int) -> DenoisingPlan { DenoisingPlan(startStep: 0, startSigma: 1, evaluations: 2) }
}

private struct LoopDecoding: DecodingModule {
    var space: LatentSpace { .flux }
}

private struct LoopParts: RenderParts {
    typealias Decoder = LoopDecoder
    init(_ request: Request, _ admitted: Admission, model: Model) throws {}
    var encodes: Bool { false }
    mutating func tokens() throws -> Int {
        Thread.sleep(forTimeInterval: RenderLoopTests.pause)
        return 3
    }
    mutating func encode() throws -> Bool { false }
    mutating func image() throws -> Latent? { nil }
    func denoiser(height: Int, width: Int) throws -> any RenderDenoiser {
        Thread.sleep(forTimeInterval: RenderLoopTests.pause)
        return LoopDiT()
    }
    var decoderPath: String { "" }
}

private final class LoopDiT: RenderDenoiser {
    func derivative(_ latent: [Float], sigma: Float, solver: Solver, first: Bool) throws -> [Float] {
        let n = RenderLoopTests.script.evaluations
        RenderLoopTests.script.evaluations += 1
        if RenderLoopTests.script.cancelAt == n { RenderLoopTests.script.cancellation?.cancel() }
        return [Float](repeating: RenderLoopTests.script.nonFinite.contains(n) ? .nan : 0, count: latent.count)
    }
}

private struct LoopDecoder: LatentDecoder {
    let pixels: Int
    static func open(path: String, latentHeight: Int, latentWidth: Int, budget: Int) async throws -> LoopDecoder {
        RenderLoopTests.script.decoderOpened += 1
        let factor = LatentSpace.flux.factor
        return LoopDecoder(pixels: 3 * factor * latentHeight * factor * latentWidth)
    }
    static func admit(path: String, latentHeight: Int, latentWidth: Int, budget: Int) throws {}
    func pixels(_ latent: [Float]) async throws -> [Float] {
        RenderLoopTests.script.decoded += 1
        return [Float](repeating: RenderLoopTests.script.decodesNaN ? .nan : 0, count: pixels)
    }
}
