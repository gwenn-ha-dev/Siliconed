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
        XCTAssertEqual(m.note, "clouds of one-ulp moves: engine median of 5 draws, oracle worst of 3")
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

    // MARK: - Clouds of summation orders

    /// The oracle's order draws (seeds 2, 5, 7) at 0.12, 0.30, 0.06 beside its ulp draws 0.10 and 0.05: its cloud holds the
    /// five and its nominal, worst 0.30, bar 0.375. The engine draws the same three seeds — never the ulp moves — and its
    /// median of four (0.2 at the witness, 0.1, 0.3, 0.35) is judged: 0.3 passes.
    func testTheOrdersJudgeWhenTheOracleDrewThem() throws {
        let orders = zip([UInt64(2), 5, 7], try draws([0.12, 0.30, 0.06])).map { (seed: $0, file: $1) }
        var asked: [UInt64] = [], moved = 0
        let byOrder: [UInt64: Float] = [2: 0.1, 5: 0.3, 7: 0.35]
        var clouds = Clouds(draws: try draws([0.10, 0.05]), orders: orders,
                            engineOrder: { seed in asked.append(seed); return ["h": self.away(byOrder[seed]!)] }) { _ in
            moved += 1
            return ["h": self.away(0)]
        }
        let m = try truth.withUnsafeBufferPointer { try clouds.arbitrate(failed(), exact: $0, columns: 1) }
        XCTAssertEqual(asked, [2, 5, 7])
        XCTAssertEqual(moved, 0)
        XCTAssertEqual(m.threshold, 0.375, accuracy: 1e-6)
        XCTAssertEqual(m.atWitness!.engine, 0.3, accuracy: 1e-6)   // sorted 0.1 0.2 0.3 0.35: the upper median
        XCTAssert(m.passed)
        XCTAssertEqual(m.note, "clouds of summation orders: engine median of 4 (nominal + 3 orders), oracle worst of 6 (nominal + 2 ulp + 3 orders)")
    }

    /// Without the engine's own order draws (a stage that cannot draw them), the oracle's are left out and the ulp moves judge.
    func testOrdersWithoutTheEnginesDrawsAreLeftOut() throws {
        let orders = zip([UInt64(1), 2], try draws([0.9, 0.9])).map { (seed: $0, file: $1) }
        var clouds = Clouds(draws: try draws([0.10, 0.05]), orders: orders) { _ in ["h": self.away(0.01)] }
        let m = try truth.withUnsafeBufferPointer { try clouds.arbitrate(failed(), exact: $0, columns: 1) }
        XCTAssertEqual(m.threshold, 0.125, accuracy: 1e-6)
        XCTAssertEqual(m.note, "clouds of one-ulp moves: engine median of 5 draws, oracle worst of 3")
    }

    /// The trajectory's way in: a step's measure (`v_2`) judged under the draws' name (`model_out`), the oracle's order draws
    /// read by `Clouds.orders` from the step's stem — `permgroup1` and up, never `permgroup0` (the identity by another path)
    /// nor `permfull` — and the engine asked for exactly those seeds.
    func testATrajectoryStepIsJudgedByTheOrdersOfItsStem() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("orders-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("krea2"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let stem = "krea2/draw-krea2-512-step2"
        for (suffix, x) in [("permgroup0", Float(5)), ("permgroup1", 0.12), ("permgroup3", 0.30), ("permfull1", 5)] {
            let path = directory.appendingPathComponent("\(stem)-\(suffix).safetensors").path
            try ForgeDiT.writeSafetensors([(name: "model_out", dtype: "F32", shape: [2, 1], bytes: away(x).withUnsafeBytes { Data($0) })],
                                          to: path)
        }
        let goldens = Goldens(directory: directory.path)
        let orders = Clouds.orders(goldens, stem)
        XCTAssertEqual(orders.map(\.seed), [1, 3])
        var asked: [UInt64] = []
        var clouds = Clouds(draws: [], orders: orders, engineOrder: { seed in
            asked.append(seed)
            return ["model_out": self.away(seed == 1 ? 0.1 : 0.3)]
        }) { _ in XCTFail("no ulp draw where the orders judge"); return [:] }
        let step = Measure(boundary: "v_2", worst: 0.3, median: 0.3, channel: 0, atWitness: (0.2, 0.08),
                           threshold: 0.1, passed: false, note: "")
        let m = try truth.withUnsafeBufferPointer { try clouds.arbitrate(step, exact: $0, columns: 1, as: "model_out") }
        XCTAssertEqual(asked, [1, 3])
        XCTAssertEqual(m.boundary, "v_2")
        XCTAssertEqual(m.threshold, 0.375, accuracy: 1e-6)                  // 1.25 · 0.30
        XCTAssertEqual(m.atWitness!.engine, 0.2, accuracy: 1e-6)            // sorted 0.1 0.2 0.3
        XCTAssert(m.passed)
        XCTAssertEqual(m.note, "clouds of summation orders: engine median of 3 (nominal + 2 orders), oracle worst of 3 (nominal + 0 ulp + 2 orders)")
    }

    // MARK: - Seed outside

    /// The trajectory's order draws: one setup (a permuted map) per seed, prepared once and released before the next,
    /// however many failing steps want it — 8 maps for 1 step as for 5, never 8 per step.
    func testOneMapIsWrittenPerSeedWhateverTheNumberOfSteps() throws {
        for steps in [1, 5] {
            var prepared: [UInt64] = [], released = 0, alive = 0, evaluated: [(Int, UInt64)] = []
            let requests = Array(repeating: Array(UInt64(1)...8), count: steps)
            let results = try SeedOutside.run(requests, prepare: { seed -> UInt64 in
                XCTAssertEqual(alive, 0, "one map on the disk at a time")
                prepared.append(seed); alive += 1
                return seed
            }, release: { _ in released += 1; alive -= 1 }, evaluate: { i, seed, map in
                XCTAssertEqual(map, seed)
                evaluated.append((i, seed))
                return ["model_out": [Float(i), Float(seed)]]
            })
            XCTAssertEqual(prepared, Array(1...8))
            XCTAssertEqual(released, 8)
            XCTAssertEqual(evaluated.count, 8 * steps)
            XCTAssertEqual(results.count, steps)
            for (i, r) in results.enumerated() {
                XCTAssertEqual(r.keys.sorted(), Array(1...8))
                XCTAssertEqual(r[3]?["model_out"], [Float(i), 3])
            }
        }
    }

    /// Requests that want different seeds: the union is prepared once each, each request gets only its own.
    func testSeedsAreTheUnionOfTheRequests() throws {
        var prepared: [UInt64] = []
        let results = try SeedOutside.run([[2, 5], [], [5, 7]], prepare: { prepared.append($0); return $0 },
                                          release: { _ in }, evaluate: { _, seed, _ in ["h": [Float(seed)]] })
        XCTAssertEqual(prepared, [2, 5, 7])
        XCTAssertEqual(results.map { $0.keys.sorted() }, [[2, 5], [], [5, 7]])
    }

    /// `judgesByOrders` is the question `arbitrate` answers: a failing measure with two order draws, not a passed one, not
    /// one without its witness, not one whose draws are of another size.
    func testJudgesByOrdersIsWhatArbitrateDoes() throws {
        let orders = zip([UInt64(1), 2], try draws([0.1, 0.1])).map { (seed: $0, file: $1) }
        let clouds = Clouds(draws: [], orders: orders, engineOrder: { _ in ["h": self.away(0.1)] }) { _ in [:] }
        try truth.withUnsafeBufferPointer { exact in
            XCTAssertTrue(clouds.judgesByOrders(failed(), exact: exact))
            let passed = Measure(boundary: "h", worst: 0.1, median: 0.1, channel: 0, atWitness: (0.01, 0.08), threshold: 0.1,
                                 passed: true, note: "")
            XCTAssertFalse(clouds.judgesByOrders(passed, exact: exact))
            XCTAssertFalse(clouds.judgesByOrders(failed(), exact: nil))
            XCTAssertFalse(clouds.judgesByOrders(failed(), exact: exact, as: "elsewhere"))
        }
        let wrong = zip([UInt64(1), 2], try draws([0.1, 0.1], shape: [3, 1])).map { (seed: $0, file: $1) }
        let other = Clouds(draws: [], orders: wrong, engineOrder: { _ in [:] }) { _ in [:] }
        try truth.withUnsafeBufferPointer { XCTAssertFalse(other.judgesByOrders(failed(), exact: $0)) }
    }
}
