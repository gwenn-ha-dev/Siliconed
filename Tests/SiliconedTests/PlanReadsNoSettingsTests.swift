import XCTest
@testable import Siliconed

/// **No plan reads the settings.** The app asks for plans (the time foreseen, the exploration, the
/// grids, `silicontrol add`) before this Mac's map is measured; a plan that read `EngineSettings.effective`
/// froze the process on the defaults and the map was refused for the whole session — the AMX off, +20 %
/// per step, on the first launch after each update (0.15).
final class PlanReadsNoSettingsTests: XCTestCase {

    func testNoFamilyPlanReadsTheSettings() throws {
        #if DEBUG
        let modules: [any DenoisingModule] = [
            ZImageDenoisingModule(map: ""), AnimaDenoisingModule(map: ""), Krea2DenoisingModule(map: ""),
            QwenImage21DenoisingModule(map: "", scheduler: .init()), MossImageDenoisingModule(map: "", turbo: ""),
        ]
        let before = EngineSettings.forbiddenReads.load(ordering: .relaxed)
        EngineSettings.$readsForbidden.withValue(true) {
            for halfSize in [false, true] {
                Spectral.halfSizeStart = halfSize
                for m in modules {
                    for (h, w) in [(64, 64), (128, 128), (128, 192)] {
                        _ = m.plan(height: h, width: w, steps: m.defaultSteps, start: 0)
                        _ = m.plan(height: h, width: w, steps: m.defaultSteps, start: 0, withLoRA: true)
                        _ = m.plan(height: h, width: w, steps: m.defaultSteps, start: 0, withLoRA: false,
                                   schedule: .official)
                    }
                }
            }
        }
        Spectral.halfSizeStart = false
        XCTAssertEqual(EngineSettings.forbiddenReads.load(ordering: .relaxed), before,
                       "a plan read EngineSettings.effective")
        #else
        throw XCTSkip("the tripwire exists in debug builds only")
        #endif
    }

    /// The tripwire itself trips: a read inside the scope is counted, one outside is not.
    func testTheTripwireCountsAReadInsideItsScope() throws {
        #if DEBUG
        let before = EngineSettings.forbiddenReads.load(ordering: .relaxed)
        _ = EngineSettings.effective.machine
        XCTAssertEqual(EngineSettings.forbiddenReads.load(ordering: .relaxed), before)
        EngineSettings.$readsForbidden.withValue(true) { _ = EngineSettings.effective.machine }
        XCTAssertEqual(EngineSettings.forbiddenReads.load(ordering: .relaxed), before + 1)
        #else
        throw XCTSkip("the tripwire exists in debug builds only")
        #endif
    }
}
