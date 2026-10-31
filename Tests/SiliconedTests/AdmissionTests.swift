import XCTest
@testable import Siliconed

/// **What the chain says of a request before any computation** (`Admission`): pure, no weight, no file — the chains have empty
/// paths, so a test that touched one would not get the error it expects. Also the one rule of the half-size start
/// (`Spectral.reducedEvaluations`) and its field on the request.
final class AdmissionTests: XCTestCase {

    private let zImage = try! Chain(
        text: ZImageTextModule(map: "", tokenizer: ""), denoising: ZImageDenoisingModule(map: ""),
        decoding: FluxDecodingModule(path: ""), encoding: FluxEncodingModule(path: ""))

    private func judge(_ request: Request, usesANE: Bool = true) throws -> Admission {
        try Admission.judge(request, chain: zImage, usesANE: usesANE)
    }

    func testTheGridStepsAndPlanOfARequest() throws {
        let a = try judge(Request("a cat", width: 1024, height: 1536))
        XCTAssertEqual([a.height, a.width], [192, 128])
        XCTAssertEqual(a.steps, zImage.denoising.defaultSteps)
        XCTAssertEqual(a.start, 0)
        XCTAssertEqual(a.plan.evaluations, a.steps)
        XCTAssertEqual(a.plan.reduced, 0, "off by default: the recipe at full size")
    }

    func testRefusalsComeInTheirOrderAndBeforeAnything() {
        XCTAssertThrowsError(try judge(Request("  \n", resolution: 512))) {
            XCTAssertEqual($0 as? Request.Failure, .emptyPrompt)
        }
        var many = Request("x", resolution: 512); many.steps = Request.maximumSteps + 1
        XCTAssertThrowsError(try judge(many)) { XCTAssertEqual($0 as? Request.Failure, .steps(Request.maximumSteps + 1)) }
        var detail = Request("x", resolution: 512); detail.detail = .more
        XCTAssertNoThrow(try judge(detail), "Z-Image offers Detail")
        var solver = Request("x", resolution: 512); solver.solver = .unipc
        XCTAssertNoThrow(try judge(solver))
    }

    func testTheNeuralEngineIsAskedAndOffered() throws {
        var request = Request("x", resolution: 512)
        XCTAssertTrue(try judge(request, usesANE: true).neuralEngine)
        XCTAssertFalse(try judge(request, usesANE: false).neuralEngine, "the plan has no ANE")
        request.neuralEngine = false
        XCTAssertFalse(try judge(request, usesANE: true).neuralEngine)
    }

    func testFasterRenderingIsTheRequestsAndReachesThePlan() throws {
        try XCTSkipIf(Spectral.forced != nil, "SILICONED_SPECTRAL forces k")
        var request = Request("x", resolution: 1024)
        let off = try judge(request)
        XCTAssertFalse(off.fasterRendering); XCTAssertEqual(off.plan.reduced, 0)
        request.fasterRendering = true
        let on = try judge(request)
        XCTAssertTrue(on.fasterRendering)
        XCTAssertEqual(on.plan.reduced, ZImageDenoisingModule.halfSizeSteps)
        // The rule's own refusals still hold: a LoRA, a format that does not halve.
        var lora = request; lora.loras = [LoRAEntry("x.lora.silicon")]
        XCTAssertEqual(try judge(lora).plan.reduced, 0)
        var small = Request("x", resolution: 512); small.fasterRendering = true
        XCTAssertEqual(try judge(small).plan.reduced, 0)
    }

    /// The rule, alone: the recipe's step count, the first step, the floor, no LoRA — and a forcing replaces the family's count.
    func testTheHalfSizeRuleAlone() throws {
        try XCTSkipIf(Spectral.forced != nil, "SILICONED_SPECTRAL forces k")
        func k(faster: Bool = true, size: Int = 128, steps: Int = 8, start: Int = 0, lora: Bool = false, evaluations: Int = 8) -> Int {
            Spectral.reducedEvaluations(faster: faster, height: size, width: size, steps: steps, start: start, withLoRA: lora,
                                        defaultSteps: 8, halfSizeSteps: 3, evaluations: evaluations)
        }
        XCTAssertEqual(k(), 3)
        XCTAssertEqual(k(faster: false), 0)
        XCTAssertEqual(k(size: 64), 0, "512²")
        XCTAssertEqual(k(steps: 12), 0, "off the recipe")
        XCTAssertEqual(k(start: 1), 0)
        XCTAssertEqual(k(lora: true), 0)
        XCTAssertEqual(k(evaluations: 3), 2, "one evaluation at least stays at full size")
    }

    func testTheRequestAndTheRenderKeepTheSwitchInTheirMetadata() {
        var request = Request("x", resolution: 1024)
        XCTAssertFalse(request.fasterRendering)
        request.fasterRendering = true
        XCTAssertNotEqual(request, Request("x", resolution: 1024))
        let text = Infotext.text(prompt: "x", steps: 8, sampler: .euler, seed: 1, width: 1024, height: 1024, model: "m",
                                 strength: nil, loras: [], civitaiModelVersion: nil, version: nil, fasterRendering: true)
        XCTAssertTrue(text.contains("Faster rendering: true"))
        let plain = Infotext.text(prompt: "x", steps: 8, sampler: .euler, seed: 1, width: 1024, height: 1024, model: "m",
                                  strength: nil, loras: [], civitaiModelVersion: nil, version: nil)
        XCTAssertFalse(plain.contains("Faster rendering"))
    }
}
