import XCTest
@testable import Siliconed

/// **The decisions behind `silicontrol rewrite`, `describe` and `set`**, without an app: the order of the writer's refusals (the Lab's
/// and the command's are one rule), the model it writes for, the arguments, and the Beta options' names.
final class RemoteWritingTests: XCTestCase {
    // ── Why the writer waits ───────────────────────────────────────────────────────────────

    func testTheWriterWaitsForTheOtherMacThenTheRenderThenTheDisk() {
        let head = WriterPause.needsHead(bytes: 1_244_659_712)
        XCTAssertEqual(WriterPause.decide(following: true, rendering: true, installed: head, hasTarget: false), .following)
        XCTAssertEqual(WriterPause.decide(following: false, rendering: true, installed: head, hasTarget: false), .rendering)
        XCTAssertEqual(WriterPause.decide(following: false, rendering: false, installed: head, hasTarget: false), head)
        XCTAssertEqual(WriterPause.decide(following: false, rendering: false, installed: nil, hasTarget: false), .notInstalled)
        XCTAssertNil(WriterPause.decide(following: false, rendering: false, installed: nil, hasTarget: true))
    }

    func testEachRefusalHasItsOwnStableCode() {
        let all: [WriterPause] = [.following, .rendering, .notInstalled, .needsMemory(gigabytes: 16), .needsHead(bytes: 1), .preparing]
        XCTAssertEqual(all.map(\.code), ["remote", "busy", "writer_absent", "writer_memory", "writer_head", "writer_preparing"])
        XCTAssertEqual(Set(all.map(\.code)).count, all.count)
    }

    // ── The model it writes for ────────────────────────────────────────────────────────────

    func testTheDefaultTargetIsTheLabs() {
        let installed: [Family] = [.zImage, .qwenImage21, .anima]
        // A choice made by hand in the Lab wins while that model is installed, then the rack's, then the first.
        XCTAssertEqual(WriterTarget.pick(chosen: .anima, rack: .qwenImage21, installed: installed), .anima)
        XCTAssertEqual(WriterTarget.pick(chosen: .krea2, rack: .qwenImage21, installed: installed), .qwenImage21)
        XCTAssertEqual(WriterTarget.pick(chosen: nil, rack: .krea2, installed: installed), .zImage)
        XCTAssertEqual(WriterTarget.pick(chosen: nil, rack: nil, installed: []), nil)
    }

    func testForNamesAnInstalledFamilyOrOneOfItsModels() {
        let installed: [Family] = [.zImage, .anima]
        let modelFamily: (String) -> Family? = { $0 == "my-imported-model" ? .anima : nil }
        XCTAssertEqual(WriterTarget.named("anima", installed: installed, modelFamily: modelFamily), .anima)
        XCTAssertEqual(WriterTarget.named("z-image", installed: installed, modelFamily: modelFamily), .zImage)
        XCTAssertEqual(WriterTarget.named("my-imported-model", installed: installed, modelFamily: modelFamily), .anima)
        // Known, but not installed: the Lab's picker does not offer it either.
        XCTAssertNil(WriterTarget.named("krea2", installed: installed, modelFamily: modelFamily))
        XCTAssertNil(WriterTarget.named("nope", installed: installed, modelFamily: modelFamily))
    }

    // ── rewrite and describe ───────────────────────────────────────────────────────────────

    func testRewriteTakesOneTextAndAnOptionalTarget() {
        XCTAssertEqual(try WriterOrder.parse(.rewrite, ["a cat"]).get(), WriterOrder(gesture: .rewrite, subject: "a cat", target: nil))
        XCTAssertEqual(try WriterOrder.parse(.rewrite, ["--for", "anima", "a cat"]).get().target, "anima")
        XCTAssertEqual(try WriterOrder.parse(.rewrite, ["a cat", "--for", "anima"]).get().subject, "a cat")
        // A text with a space is a text, even when it starts like a flag.
        XCTAssertEqual(try WriterOrder.parse(.rewrite, ["--ar 16:9 a cat"]).get().subject, "--ar 16:9 a cat")
    }

    func testRewriteRefusesWhatIsNotItsSyntax() {
        XCTAssertEqual(failure(.rewrite, []), .usage(WriterOrder.usage(.rewrite)))
        XCTAssertEqual(failure(.rewrite, ["a", "b"]), .usage(WriterOrder.usage(.rewrite)), "two texts: the quotes are missing")
        XCTAssertEqual(failure(.rewrite, ["a cat", "--for"]), .flagWithoutValue("--for"))
        XCTAssertEqual(failure(.rewrite, ["a cat", "--wait"]), .unknownFlag("--wait"))
        XCTAssertEqual(failure(.rewrite, ["   "]), .emptyText)
        XCTAssertEqual(failure(.rewrite, ["a", "--for", "x", "--for", "y"]), .usage(WriterOrder.usage(.rewrite)))
    }

    func testDescribeTakesAnImageNameOrAPath() {
        XCTAssertEqual(try WriterOrder.parse(.describe, ["i7"]).get().subject, "i7")
        XCTAssertEqual(try WriterOrder.parse(.describe, ["photo.jpg", "--for", "z-image"]).get(),
                       WriterOrder(gesture: .describe, subject: "photo.jpg", target: "z-image"))
        XCTAssertEqual(failure(.describe, []), .usage(WriterOrder.usage(.describe)))
    }

    private func failure(_ g: WriterOrder.Gesture, _ a: [String]) -> WriterOrder.Failure? {
        if case .failure(let f) = WriterOrder.parse(g, a) { return f }
        return nil
    }

    // ── set ────────────────────────────────────────────────────────────────────────────────

    func testSetReadsAModelsSwitch() throws {
        XCTAssertEqual(try SetOrder.parse(["faster-rendering", "on"]).get(), .perModel(.fasterRendering, on: true, model: nil))
        XCTAssertEqual(try SetOrder.parse(["neural-engine", "off", "--model", "krea2"]).get(), .perModel(.accelerators, on: false, model: "krea2"))
        XCTAssertEqual(try SetOrder.parse(["accelerators", "on"]).get(), .perModel(.accelerators, on: true, model: nil))
        for bad in [["neural-engine"], ["neural-engine", "maybe"], ["neural-engine", "on", "--model"], ["neural-engine", "on", "krea2"], []] {
            guard case .failure(.usage) = SetOrder.parse(bad) else { return XCTFail("\(bad)") }
        }
        guard case .failure(.usage) = SetOrder.parse(["volume", "on"]) else { return XCTFail() }
    }

    func testSetReadsTheBetaOptions() throws {
        XCTAssertEqual(try SetOrder.parse(["beta"]).get(), .betaList)
        XCTAssertEqual(try SetOrder.parse(["beta", "more-solvers", "on"]).get(), .beta(.moreSolvers, on: true))
        XCTAssertEqual(try SetOrder.parse(["beta", "size-in-two-scales", "off"]).get(), .beta(.sizeScales, on: false), "the setting's own name")
        XCTAssertEqual(try SetOrder.parse(["beta", "Negative-LoRA-Strengths", "on"]).get(), .beta(.negativeLoRA, on: true))
        XCTAssertEqual(SetOrder.parse(["beta", "dark-mode", "on"]), .failure(.unknownBeta("dark-mode")))
        guard case .failure(.usage) = SetOrder.parse(["beta", "more-solvers"]) else { return XCTFail() }
        guard case .failure(.usage) = SetOrder.parse(["beta", "more-solvers", "1"]) else { return XCTFail() }
    }

    func testSetReadsSaveEveryImage() throws {
        // Save Every Image To…: a folder turns it on, `off` turns it off, alone it is read; `./off` is a folder.
        XCTAssertEqual(try SetOrder.parse(["save-images"]).get(), .saveImagesState)
        XCTAssertEqual(try SetOrder.parse(["save-images", "off"]).get(), .saveImages(folder: nil))
        XCTAssertEqual(try SetOrder.parse(["save-images", "~/Pictures/Siliconed"]).get(), .saveImages(folder: "~/Pictures/Siliconed"))
        XCTAssertEqual(try SetOrder.parse(["save-images", "./off"]).get(), .saveImages(folder: "./off"))
        guard case .failure(.usage) = SetOrder.parse(["save-images", "on", "~/Pictures"]) else { return XCTFail() }
    }

    /// The four options of Settings › Advanced › Beta, no more: a fifth has to be added here, in the app's `AppState.beta`
    /// (a `switch` the compiler checks) and in the help.
    func testTheBetaOptionsAreTheSettingsFour() {
        XCTAssertEqual(BetaOption.allCases.map(\.rawValue), ["more-solvers", "faster-rendering", "size-scales", "negative-lora"])
        XCTAssertEqual(BetaOption.allCases.map(\.key), ["more_solvers", "faster_rendering", "size_scales", "negative_lora"])
        XCTAssertEqual(BetaOption.names, "more-solvers, faster-rendering, size-scales, negative-lora")
    }

    /// The Neural Engine is offered or not by the map, whichever the version: `set neural-engine` refuses on the same rule as the switch.
    func testTheNeuralEngineSwitchHasNothingToChooseWithoutARow() {
        XCTAssertEqual(DiTSwitches.acceleratorsUnavailable(measured: true, offered: false), .noAccelerator)
        XCTAssertNil(DiTSwitches.acceleratorsUnavailable(measured: true, offered: true))
    }
}
