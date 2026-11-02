import XCTest
@testable import Siliconed
@testable import Verification

/// **The registry of verified stages, and the judge's rule, without a weight.** `siliconed check list`,
/// the verification guide's table and the closing run are all read from the registry: a stage that
/// names no oracle, a size the parser refuses or an option it does not know would show up only when
/// someone runs that stage. And the rule decides every verdict: its edges are checked here on numbers
/// small enough to compute by hand.
final class VerificationRegistryTests: XCTestCase {
    func testEveryStageIsNamedOnceAfterAModelOfTheApp() {
        let models = Set(ModelCard.allCards.map(\.id) + ["kernels"])
        var seen: Set<String> = []
        for stage in Registry.stages {
            XCTAssert(models.contains(stage.model), "\(stage.command): '\(stage.model)' is not a model of the app")
            XCTAssert(seen.insert("\(stage.model) \(stage.name)").inserted, "\(stage.command) registered twice")
        }
        XCTAssertEqual(Set(Registry.stages.map(\.model)), models, "a model of the app has no stage")
    }

    func testEveryStageSaysWhatItProvesAndHowItIsJudged() {
        for stage in Registry.stages {
            XCTAssertFalse(stage.proves.isEmpty, stage.command)
            XCTAssertFalse(stage.rule.isEmpty, stage.command)
            // One line each: they are table cells.
            XCTAssertFalse((stage.proves + stage.rule).contains("\n"), stage.command)
        }
    }

    /// An oracle is a script a reader can run, from the repository; a stage at several sizes says which
    /// one it writes. A stage without one is judged against its definition (a kernel, a LoRA stack's algebra).
    func testEveryOracleIsAScriptOfTheRepository() {
        for stage in Registry.stages {
            guard let oracle = stage.oracle else { continue }
            XCTAssert(oracle.hasPrefix("tools/oracle/"), "\(stage.command): \(oracle)")
            if stage.sizes.count > 1 { XCTAssert(oracle.contains("{size}"), "\(stage.command): which size does \(oracle) write?") }
        }
    }

    func testAPipeInACellIsEscaped() {
        let rows = CheckCommand.markdown(Registry.stages).split(separator: "\n").dropFirst(2)
        for row in rows { XCTAssertEqual(row.replacingOccurrences(of: "\\|", with: "").filter { $0 == "|" }.count, 6, String(row)) }
    }

    func testEverySizeAndOptionParses() throws {
        for stage in Registry.stages {
            XCTAssertFalse(stage.sizes.isEmpty, stage.command)
            for size in stage.sizes {
                let invocation = try Invocation(["--size", size], accepted: stage.options, defaultSize: stage.sizes[0])
                XCTAssertEqual(invocation.size, size, "\(stage.command): '\(size)' is not how a golden's name spells it")
            }
            for option in stage.options { XCTAssert(option.hasPrefix("--"), "\(stage.command): \(option)") }
        }
    }

    func testTheMarkdownTableHasOneRowPerStage() {
        let stages = Registry.stages
        XCTAssertEqual(CheckCommand.markdown(stages).split(separator: "\n").count, stages.count + 2)
    }

    func testOptionsAreNamedAndRefusedWhenNotTheStages() throws {
        XCTAssertThrowsError(try Invocation(["1024"], accepted: [], defaultSize: "512"))
        XCTAssertThrowsError(try Invocation(["--edit", "1"], accepted: [], defaultSize: "512"))
        XCTAssertThrowsError(try Invocation(["--size", "256"], accepted: [], defaultSize: "512"))
        XCTAssertThrowsError(try Invocation(["--size"], accepted: [], defaultSize: "512"))
        let edit = try Invocation(["--edit", "2", "--size", "512x768"], accepted: ["--edit"], defaultSize: "512")
        XCTAssertEqual(edit.edit, 2)
        XCTAssertEqual([edit.width, edit.height], [512, 768])
        XCTAssertThrowsError(try Invocation(["--edit", "0"], accepted: ["--edit"], defaultSize: "512"))
    }

    // MARK: - The rule

    private func judge(_ engine: [Float], _ oracle: [Float], witness: [Float]? = nil, half: [Float]? = nil) -> Measure {
        oracle.withUnsafeBufferPointer { o in
            if let witness {
                return witness.withUnsafeBufferPointer { Judge.boundary("x", engine: engine, oracle: o, witness: $0, half: half, columns: 1) }
            }
            return Judge.boundary("x", engine: engine, oracle: o, half: half, columns: 1)
        }
    }

    func testWithoutAWitnessTheFloorDecides() {
        XCTAssert(judge([1.0004], [1]).passed)
        XCTAssertFalse(judge([1.0006], [1]).passed)
        // On a half-precision path, bf16's own distance + 25 % when it is above the floor.
        XCTAssert(judge([1.0012], [1], half: [1.001]).passed)
        XCTAssertFalse(judge([1.0013], [1], half: [1.001]).passed)
    }

    /// With a witness, it alone decides: an engine at the floor from the oracle but far from the exact
    /// fails, and one far from the oracle but no further from the exact than the oracle passes.
    func testWithAWitnessTheExactDecides() {
        XCTAssert(judge([0.997], [1.003], witness: [1]).passed)
        XCTAssertFalse(judge([1.0045], [1.003], witness: [1]).passed)
        XCTAssertEqual(judge([0.997], [1.003], witness: [1]).threshold, 1.25 * 0.003, accuracy: 1e-6)
        // The floor stays under it: a reference closer than the floor does not tighten the bar below it.
        XCTAssert(judge([1.0004], [1.00001], witness: [1]).passed)
    }

    func testPlanesHaveNoFloorAtTheWitness() {
        let exact = [Float](repeating: 1, count: 8), oracle = [Float](repeating: 1.000001, count: 8)
        let engine = [Float](repeating: 1.0001, count: 8)   // 100× the oracle, under the floor
        let m = oracle.withUnsafeBufferPointer { o in
            exact.withUnsafeBufferPointer { Judge.planes("p", engine: engine, oracle: o, witness: $0, planes: 2) }
        }
        XCTAssertFalse(m.passed, "a floor at the witness would hide a hundredfold drift")
    }

    func testAShapeMismatchFailsAndSaysIt() {
        let m = judge([1, 2], [1])
        XCTAssertFalse(m.passed)
        XCTAssert(m.note.contains("shape"))
        XCTAssertFalse([Float](repeating: 0, count: 3).withUnsafeBufferPointer {
            Judge.identical("i", [0, 0], $0).passed
        })
    }

    func testOutcomesMapToExitCodes() {
        XCTAssertEqual(Outcome.passed.exitCode, 0)
        XCTAssertEqual(Outcome.failed.exitCode, 1)
        XCTAssertEqual(Outcome.skipped("x").exitCode, 2)
    }

    func testAFailureIsNotHiddenByASkipAfterIt() {
        struct Broken: Error {}
        func kernel(_ name: String, _ run: @escaping @Sendable (Invocation) throws -> Report) -> Stage {
            Stage(model: "kernels", name: name, proves: "-", oracle: nil, weights: [], rule: "-", run: run)
        }
        let fails = kernel("fails") { _ in throw Broken() }
        let skips = kernel("skips") { _ in throw Skip("no input") }
        XCTAssertEqual(Command.run(["check", "kernels"], stages: [fails, skips]), 1)
        XCTAssertEqual(Command.run(["check", "kernels"], stages: [skips, fails]), 1)
        XCTAssertEqual(Command.run(["check", "kernels"], stages: [skips]), 2)
    }

    func testAModelAloneRunsEveryStageAtEverySize() {
        let stages = Registry.stages.filter { $0.model == "anima" }
        let runs = CheckCommand.everyStage(stages, ["--goldens", "g"])
        XCTAssertEqual(runs.map(\.0), ["anima tokenizer", "anima text", "anima dit 512", "anima dit 1024", "anima dit 768x512",
                                       "anima trajectory 512", "anima trajectory 1024",
                                       "anima vae 512", "anima vae 1024", "anima vae 768x512", "anima img2img"])
        XCTAssertEqual(runs[3].2, ["--size", "1024", "--goldens", "g"])
        XCTAssertEqual(runs[0].2, ["--goldens", "g"])
        let at1024 = CheckCommand.everyStage(stages, ["--size", "1024"])
        XCTAssertEqual(at1024.map(\.0), ["anima dit 1024", "anima trajectory 1024", "anima vae 1024"])
        XCTAssertEqual(at1024[0].2, ["--size", "1024"])
        let qwen = CheckCommand.everyStage(Registry.stages.filter { $0.model == "qwen-image-2.1" }, [])
        XCTAssertEqual(qwen.filter { $0.1.name == "dit" }.map(\.2), [[], ["--edit", "1"], ["--edit", "2"]])
        XCTAssert(qwen.map(\.0).contains("qwen-image-2.1 references --side 443"))
    }
}
