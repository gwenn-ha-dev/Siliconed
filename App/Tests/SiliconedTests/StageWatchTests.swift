import Foundation
@testable import SiliCore
import XCTest
@testable import Siliconed

/// **A render's stages as the engine's told them** (`StageWatch`): each engine stage opened while a stage of the render
/// was under way is that stage's; what its plan counted above the floor, its ring, the
/// share of the Neural Engine the family said, weighted by the seconds each stage lived.
final class StageWatchTests: XCTestCase {
    private func plan(_ name: String, bytes: Int) throws -> SiliCore.Stage.Plan {
        var d = SiliCore.Stage.Declaration(name)
        d.tensors = [.init("x", bytes: bytes)]
        return try SiliCore.Stage.plan(d, files: [], budget: 1 << 40, floor: 100 << 20)
    }

    func testEachEngineStageIsTheRenderStageUnderWay() throws {
        let ledger = StageWatch()
        defer { ledger.stop() }
        let tell = SiliCore.Stages.tell
        let text = try plan("text", bytes: 64 << 20), dit = try plan("dit", bytes: 512 << 20)

        ledger.begin(.text)
        tell(.opened(text)); tell(.closed(text, seconds: 1))
        ledger.begin(.denoising)
        tell(.opened(dit))
        tell(.neural(0.25)); tell(.closed(dit, seconds: 3))

        let (stages, share) = ledger.held()
        XCTAssertEqual(stages.map(\.stage), [.text, .denoising])
        XCTAssertEqual(stages[0].counted, text.p - text.floor)
        XCTAssertEqual(stages[1].counted, dit.p - dit.floor)
        XCTAssertEqual(stages[1].neuralEngine, 0.25)
        XCTAssertEqual(share ?? 0, 0.75 / 4, accuracy: 1e-12, "3 s at 0.25 over 4 s lived")
        XCTAssertGreaterThan(stages[1].process, 0)
        XCTAssertEqual(ledger.heard.map(\.name), ["text", "dit"])
        XCTAssertEqual(ledger.heard.map(\.open), [false, false])
    }
}
