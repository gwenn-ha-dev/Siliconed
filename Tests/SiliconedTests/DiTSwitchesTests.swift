import XCTest
@testable import Siliconed

/// **The rules behind the two switches of the app's DiT box** (`DiTSwitches`): pure, no weights.
final class DiTSwitchesTests: XCTestCase {
    /// The Beta option is the master: off, nothing is faster whatever was chosen or defaulted; on, the choice, else the default.
    func testFasterRenderingFollowsBetaThenChoiceThenDefault() {
        for choice in [nil, false, true] as [Bool?] {
            for versionDefault in [false, true] {
                XCTAssertFalse(DiTSwitches.fasterRendering(beta: false, choice: choice, versionDefault: versionDefault))
            }
        }
        XCTAssertTrue(DiTSwitches.fasterRendering(beta: true, choice: true, versionDefault: false))
        XCTAssertFalse(DiTSwitches.fasterRendering(beta: true, choice: false, versionDefault: true))
        XCTAssertTrue(DiTSwitches.fasterRendering(beta: true, choice: nil, versionDefault: true))
        XCTAssertFalse(DiTSwitches.fasterRendering(beta: true, choice: nil, versionDefault: false))
    }

    /// Off on every version: a user who never chose gets the full-size recipe even with the option on. The Neural Engine is the
    /// opposite: on by default on every version, the Standard's included (2/11/2026), off where the user turns it off.
    func testNoVersionDefaultsToFasterRenderingAndEveryVersionToTheNeuralEngine() {
        for variant in Variant.allCases {
            XCTAssertFalse(variant.fasterRenderingByDefault)
            XCTAssertTrue(variant.acceleratorsByDefault)
        }
    }

    /// The box greys the switch for the same reasons the plan would not apply it.
    func testFasterRenderingApplicability() {
        func reason(_ w: Int, _ h: Int, lora: Bool = false, edit: Bool = false, recipe: Bool = true) -> DiTSwitchUnavailable? {
            DiTSwitches.fasterRenderingUnavailable(width: w, height: h, withLoRA: lora, edit: edit, atRecipeSteps: recipe)
        }
        XCTAssertNil(reason(1024, 1024))
        XCTAssertNil(reason(1536, 1024))
        XCTAssertNil(reason(1024, 1536))
        XCTAssertEqual(reason(832, 1216), .sizeUnder1024)
        XCTAssertEqual(reason(1216, 832), .sizeUnder1024)
        XCTAssertEqual(reason(512, 512), .sizeUnder1024)
        XCTAssertEqual(reason(1024, 1024, lora: true), .withLoRADetailOrEdit)
        XCTAssertEqual(reason(1024, 1024, edit: true), .withLoRADetailOrEdit)
        XCTAssertEqual(reason(1024, 1024, recipe: false), .stepsOffRecipe)
    }

    /// The box's verdict and the engine's agree: where the box says "applies", the engine's rule reduces at least one evaluation.
    func testBoxAgreesWithEngineRule() {
        for (w, h) in [(1024, 1024), (1024, 1536), (832, 1216), (2048, 2048), (1216, 832)] {
            let box = DiTSwitches.fasterRenderingUnavailable(width: w, height: h, withLoRA: false, edit: false, atRecipeSteps: true) == nil
            let k = Spectral.reducedEvaluations(faster: true, height: h / 8, width: w / 8, steps: 8, start: 0, withLoRA: false,
                                                defaultSteps: 8, halfSizeSteps: 3, evaluations: 8)
            XCTAssertEqual(box, k > 0, "\(w)x\(h)")
        }
    }

    func testNeuralEngineAvailabilityAndReason() {
        XCTAssertNil(DiTSwitches.acceleratorsUnavailable(measured: true, offered: true))
        XCTAssertEqual(DiTSwitches.acceleratorsUnavailable(measured: true, offered: false), .noAccelerator)
        // Before the first measure (after an update, say), the switch says so rather than « nothing to do »: whether the
        // Neural Engine gets rows is not known yet, and an unread map offers nothing whatever it holds.
        XCTAssertEqual(DiTSwitches.acceleratorsUnavailable(measured: false, offered: false), .notMeasured)
        XCTAssertEqual(DiTSwitches.acceleratorsUnavailable(measured: false, offered: true), .notMeasured)
        // Offered: the choice, else the default — on, the Standard's included; a choice off stays off.
        XCTAssertTrue(DiTSwitches.accelerators(offered: true, choice: nil, versionDefault: Variant.standard.acceleratorsByDefault))
        XCTAssertFalse(DiTSwitches.accelerators(offered: true, choice: false, versionDefault: Variant.standard.acceleratorsByDefault))
        XCTAssertTrue(DiTSwitches.accelerators(offered: true, choice: nil, versionDefault: Variant.compact.acceleratorsByDefault))
        // Not offered: never on, whatever was kept.
        XCTAssertFalse(DiTSwitches.accelerators(offered: false, choice: true, versionDefault: true))
    }
}
