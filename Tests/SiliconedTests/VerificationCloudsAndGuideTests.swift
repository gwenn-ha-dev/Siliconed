import XCTest
@testable import Siliconed
@testable import Verification

/// **Two things `check list` promises, without a weight.** The verification guide's table is the registry's
/// own output, word for word, in both languages; and the clouds' arbiter (Krea 2's ill-conditioned
/// boundaries) decides on numbers small enough to compute by hand.
final class VerificationCloudsAndGuideTests: XCTestCase {
    // MARK: - The guide's table

    private func guideTable(_ file: String) throws -> [String] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let text = try String(contentsOf: root.appendingPathComponent("docs/\(file)"), encoding: .utf8)
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        guard let start = lines.firstIndex(where: { $0.hasPrefix("| Stage | Proves") }) else { return [] }
        return Array(lines[start...].prefix { $0.hasPrefix("|") })
    }

    func testTheGuidesTableIsWhatCheckListGenerates() throws {
        let generated = CheckCommand.markdown(Registry.stages).split(separator: "\n").map(String.init)
        for file in ["VERIFY.md", "VERIFY.fr.md"] {
            let table = try guideTable(file)
            XCTAssertEqual(table.count, generated.count, "\(file): the table has another number of rows than `check list --markdown`")
            for (written, expected) in zip(table, generated) {
                XCTAssertEqual(written, expected, "\(file): regenerate the table with `siliconed check list --markdown`")
            }
        }
    }

    /// The guide clones the tag of the version being released (`tools/app.sh`): it drifted one version once.
    func testTheGuideClonesTheAppsVersion() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let app = try String(contentsOf: root.appendingPathComponent("tools/app.sh"), encoding: .utf8)
        let key = "<key>CFBundleShortVersionString</key><string>"
        let version = try XCTUnwrap(app.components(separatedBy: key).dropFirst().first?.components(separatedBy: "<").first)
        for file in ["VERIFY.md", "VERIFY.fr.md"] {
            let text = try String(contentsOf: root.appendingPathComponent("docs/\(file)"), encoding: .utf8)
            XCTAssertTrue(text.contains("git clone --branch v\(version) "), "\(file): clone the tag v\(version), the app's version")
        }
    }

    // MARK: - Clouds

    /// The truth [3, 4] has norm 5, as one channel (`columns: 1`). A draw moved by [3x, 4x] is x away from it.
    private let truth: [Float] = [3, 4]
    private func away(_ x: Float) -> [Float] { [3 + 3 * x, 4 + 4 * x] }

    private func draws(_ errors: [Float], shape: [Int] = [2, 1]) throws -> [Safetensors] {
        try errors.map { x in
            let path = FileManager.default.temporaryDirectory.appendingPathComponent("draw-\(UUID().uuidString).safetensors").path
            defer { try? FileManager.default.removeItem(atPath: path) }
            let values = shape == [2, 1] ? away(x) : [Float](repeating: 1, count: shape.reduce(1, *))
            try ForgeDiT.writeSafetensors([(name: "h", dtype: "F32", shape: shape, bytes: values.withUnsafeBytes { Data($0) })], to: path)
            return try Safetensors(path: path)
        }
    }

    /// A measure that failed at the witness: the engine 0.2 from the exact, the reference 0.08.
    private func failed(engine: Double = 0.2, reference: Double = 0.08) -> Measure {
        Measure(boundary: "h", worst: 0.3, median: 0.3, channel: 0, atWitness: (engine, reference),
                threshold: 1.25 * reference, passed: false, note: "")
    }

    private func arbitrate(_ m: Measure, oracle: [Float], engine: [Float], exact: [Float]? = nil,
                           shape: [Int] = [2, 1]) throws -> Measure {
        var clouds = Clouds(draws: try draws(oracle, shape: shape)) { seed in ["h": self.away(engine[Int(seed) - 1])] }
        let exact = exact ?? truth
        return try exact.withUnsafeBufferPointer { try clouds.arbitrate(m, exact: $0, columns: 1) }
    }

    /// Oracle cloud: its own 0.08 at the witness and draws 0.10 and 0.05, worst 0.10, allowance 1.25 · 0.10 = 0.125.
    /// Engine cloud: 0.2 at the witness and four draws, the median of the five against 0.125.
    func testAFailureAtTheWitnessPassesByTheCloudsMedian() throws {
        let m = try arbitrate(failed(), oracle: [0.10, 0.05], engine: [0.01, 0.02, 0.03, 0.9])   // sorted 0.01 0.02 0.03 0.2 0.9
        XCTAssert(m.passed)
        XCTAssertEqual(m.atWitness!.engine, 0.03, accuracy: 1e-6)
        XCTAssertEqual(m.atWitness!.reference, 0.10, accuracy: 1e-6)
        XCTAssertEqual(m.threshold, 0.125, accuracy: 1e-6)
        XCTAssertEqual(m.note, "clouds: engine median of 5 draws, oracle worst of 3")
    }

    func testAFailureStaysAFailureWhenTheMedianIsPastTheAllowance() throws {
        let m = try arbitrate(failed(), oracle: [0.10, 0.05], engine: [0.04, 0.06, 0.5, 0.9])    // sorted 0.04 0.06 0.2 0.5 0.9
        XCTAssertFalse(m.passed)
        XCTAssertEqual(m.atWitness!.engine, 0.2, accuracy: 1e-6)
    }

    /// The edge of the allowance, 0.125: a median at 0.12 passes, at 0.13 it does not.
    func testTheMedianIsJudgedAgainstOneAndAQuarterOfTheOraclesWorstDraw() throws {
        XCTAssert(try arbitrate(failed(), oracle: [0.10, 0.05], engine: [0.01, 0.02, 0.12, 0.9]).passed)    // 0.01 0.02 0.12 0.2 0.9
        XCTAssertFalse(try arbitrate(failed(), oracle: [0.10, 0.05], engine: [0.01, 0.02, 0.13, 0.9]).passed)
    }

    func testTheWorstOracleDrawSetsTheBar() throws {
        // Draws 0.10 and 0.30: the bar rises to 0.375, and a median of 0.2 now passes.
        let m = try arbitrate(failed(), oracle: [0.10, 0.30], engine: [0.2, 0.2, 0.2, 0.2])
        XCTAssert(m.passed)
        XCTAssertEqual(m.threshold, 0.375, accuracy: 1e-6)
    }

    func testAMeasureThatPassedOrHasNoWitnessIsLeftAlone() throws {
        var passed = failed()
        passed = Measure(boundary: "h", worst: 0.1, median: 0.1, channel: 0, atWitness: (0.01, 0.08), threshold: 0.1, passed: true, note: "")
        XCTAssertEqual(try arbitrate(passed, oracle: [0.1, 0.1], engine: [0, 0, 0, 0]).note, "")
        let blind = Measure(boundary: "h", worst: 0.3, median: 0.3, channel: 0, atWitness: nil, threshold: 5e-4, passed: false, note: "")
        let unchanged = try arbitrate(blind, oracle: [0.1, 0.1], engine: [0, 0, 0, 0])
        XCTAssertFalse(unchanged.passed)
        XCTAssertEqual(unchanged.note, "")
    }

    func testFewerThanTwoUsableOracleDrawsKeepTheSingleVerdict() throws {
        let one = try arbitrate(failed(), oracle: [0.1], engine: [0, 0, 0, 0])
        XCTAssertFalse(one.passed)
        XCTAssertEqual(one.note, "no oracle draws to arbitrate")
        // Two draws of another size than the truth do not count.
        let wrong = try arbitrate(failed(), oracle: [0.1, 0.1], engine: [0, 0, 0, 0], shape: [3, 1])
        XCTAssertFalse(wrong.passed)
        XCTAssertEqual(wrong.note, "no oracle draws to arbitrate")
    }
}
