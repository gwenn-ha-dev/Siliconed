import XCTest
@testable import Siliconed

/// **0.30's preferences, carried once** (`RetiredPreferences`): its « Exact » turns the Neural Engine off, its « Neural
/// Engine » leaves the default; a switch set since stands; the old keys go.
final class RetiredPreferencesTests: XCTestCase {
    private var defaults: UserDefaults!
    private let suite = "siliconed.tests.retired-preferences"
    private let key = "neuralEngine"

    override func setUp() {
        UserDefaults().removePersistentDomain(forName: suite)
        defaults = UserDefaults(suiteName: suite)
    }

    override func tearDown() { UserDefaults().removePersistentDomain(forName: suite) }

    func testExactTurnsTheNeuralEngineOff() {
        defaults.set("exact", forKey: "precision")
        defaults.set(240.0, forKey: "labImagesBoxHeight")
        RetiredPreferences.migrate(defaults, neuralEngineKey: key)
        XCTAssertEqual(defaults.object(forKey: key) as? Bool, false)
        XCTAssertNil(defaults.object(forKey: "precision"))
        XCTAssertNil(defaults.object(forKey: "labImagesBoxHeight"))
        // Once: the key gone, a second launch changes nothing — the switch turned on since stays on.
        defaults.set(true, forKey: key)
        RetiredPreferences.migrate(defaults, neuralEngineKey: key)
        XCTAssertEqual(defaults.object(forKey: key) as? Bool, true)
    }

    func testTheNeuralEngineChoiceKeepsTheDefault() {
        defaults.set("ane", forKey: "precision")
        RetiredPreferences.migrate(defaults, neuralEngineKey: key)
        XCTAssertNil(defaults.object(forKey: key))
        XCTAssertNil(defaults.object(forKey: "precision"))
    }

    func testASwitchSetSinceStands() {
        defaults.set("exact", forKey: "precision")
        defaults.set(true, forKey: key)
        RetiredPreferences.migrate(defaults, neuralEngineKey: key)
        XCTAssertEqual(defaults.object(forKey: key) as? Bool, true)
        XCTAssertNil(defaults.object(forKey: "precision"))
    }

    func testNothingToCarry() {
        RetiredPreferences.migrate(defaults, neuralEngineKey: key)
        XCTAssertNil(defaults.object(forKey: key))
    }
}
