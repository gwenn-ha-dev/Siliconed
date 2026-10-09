import XCTest
@testable import Siliconed

/// **The rule of the badly conditioned stage**, without a byte of weights: above 10× of
/// measured amplification, a stage is judged on its local error; below, on the cumulative one as
/// before; never excused when its input is the oracle's own or its output is not finite.
final class StageJudgementTests: XCTestCase {
    func testTheFluxMidBlockIsJudgedLocally() {
        // The figures of `vae-encode 512`: down_blocks.3 at 1.248e-5, mid_block at 6.069e-4 (×48.6).
        XCTAssertEqual(StageJudgement.amplification(inputError: 1.248e-5, outputError: 6.069e-4)!, 48.63, accuracy: 0.01)
        XCTAssertTrue(StageJudgement.judgesLocally(inputError: 1.248e-5, outputError: 6.069e-4))
    }

    func testAWellConditionedStageStaysCumulative() {
        // down_blocks.3 after down_blocks.2 (×3.5), and conv_out after mid_block (×0.32): a fault there
        // must still fail on the cumulative error.
        XCTAssertFalse(StageJudgement.judgesLocally(inputError: 3.568e-6, outputError: 1.248e-5))
        XCTAssertFalse(StageJudgement.judgesLocally(inputError: 6.069e-4, outputError: 1.924e-4))
        // A stage that adds a fault of its own on a clean input: ×8, still below the limit.
        XCTAssertFalse(StageJudgement.judgesLocally(inputError: 1e-4, outputError: 8e-4))
    }

    func testTheLimitIsStrict() {
        XCTAssertFalse(StageJudgement.judgesLocally(inputError: 1e-5, outputError: 1e-4))
        XCTAssertTrue(StageJudgement.judgesLocally(inputError: 1e-5, outputError: 1.0001e-4))
    }

    func testAnInputEqualToTheOraclesIsNeverExcused() {
        // `conv_in` reads the image itself: no amplification, the cumulative error is already local.
        XCTAssertNil(StageJudgement.amplification(inputError: 0, outputError: 1e-3))
        XCTAssertFalse(StageJudgement.judgesLocally(inputError: 0, outputError: 1e-3))
    }

    func testANonFiniteOutputIsNeverExcused() {
        // `channelError` turns a NaN into infinity: an infinite amplification must not send it to the
        // local error, where a NaN born upstream would vanish.
        XCTAssertFalse(StageJudgement.judgesLocally(inputError: 1e-5, outputError: .infinity))
        XCTAssertFalse(StageJudgement.judgesLocally(inputError: .infinity, outputError: .infinity))
        XCTAssertFalse(StageJudgement.judgesLocally(inputError: 1e-5, outputError: .nan))
    }
}
