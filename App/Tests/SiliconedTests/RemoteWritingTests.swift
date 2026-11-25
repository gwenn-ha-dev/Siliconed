import XCTest
@testable import Siliconed

/// **The decisions behind `silicontrol rewrite`, `describe`, `instructions` and `set`**, without an app: the order of the writer's refusals (the Lab's
/// and the command's are one rule), the model it writes for, the arguments, and the Beta options' names.
final class RemoteWritingTests: XCTestCase {
    // ── Why the writer waits ───────────────────────────────────────────────────────────────

    func testTheWriterWaitsForTheOtherMacThenTheRenderThenTheDisk() {
        XCTAssertEqual(WriterPause.decide(following: true, rendering: true, installed: false, hasTarget: false), .following)
        XCTAssertEqual(WriterPause.decide(following: false, rendering: true, installed: false, hasTarget: false), .rendering)
        XCTAssertEqual(WriterPause.decide(following: false, rendering: false, installed: false, hasTarget: true), .notInstalled)
        XCTAssertEqual(WriterPause.decide(following: false, rendering: false, installed: true, hasTarget: false), .notInstalled)
        XCTAssertNil(WriterPause.decide(following: false, rendering: false, installed: true, hasTarget: true))
    }

    func testEachRefusalHasItsOwnStableCode() {
        let all: [WriterPause] = [.following, .rendering, .notInstalled]
        XCTAssertEqual(all.map(\.code), ["remote", "busy", "writer_absent"])
    }

    // ── The model it writes for ────────────────────────────────────────────────────────────

    func testTheDefaultTargetIsTheLabs() {
        let installed: [Family] = [.zImage, .mossImage, .anima]
        // A choice made by hand in the Lab wins while that model is installed, then the rack's, then the first.
        XCTAssertEqual(WriterTarget.pick(chosen: .anima, rack: .mossImage, installed: installed), .anima)
        XCTAssertEqual(WriterTarget.pick(chosen: .krea2, rack: .mossImage, installed: installed), .mossImage)
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
        // A text that begins like a flag is one, refused by name; after a lone `--`, it is the text.
        XCTAssertEqual(failure(.rewrite, ["--ar 16:9 a cat"]), .unknownFlag("--ar 16:9 a cat"))
        XCTAssertEqual(try WriterOrder.parse(.rewrite, ["--for", "anima", "--", "--ar 16:9 a cat"]).get().subject, "--ar 16:9 a cat")
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

    /// **`instructions`**: the list alone, or one family's; one gesture shown, set, read from a file or restored — a change
    /// names its gesture, one change at a time.
    func testInstructionsListsShowsAndChangesOne() throws {
        typealias O = InstructionOrder
        XCTAssertEqual(try O.parse([]).get(), O())
        XCTAssertEqual(try O.parse(["--for", "krea2"]).get(), O(gesture: nil, target: "krea2", change: nil))
        XCTAssertEqual(try O.parse(["rewrite", "--for", "anima"]).get(), O(gesture: .rewrite, target: "anima", change: nil))
        XCTAssertEqual(try O.parse(["describe", "--set", "Mine."]).get().change, .set("Mine."))
        XCTAssertEqual(try O.parse(["--file", "mine.txt", "rewrite"]).get(), O(gesture: .rewrite, target: nil, change: .file("mine.txt")))
        XCTAssertEqual(try O.parse(["rewrite", "--restore", "--for", "z-image"]).get().change, .restore)
        XCTAssertEqual(try O.parse(["rewrite", "--set", ""]).get().change, .set(""))

        XCTAssertEqual(O.parse(["--restore"]), .failure(.usage(O.usage)), "a change names its gesture")
        XCTAssertEqual(O.parse(["rewrite", "--set", "a", "--restore"]), .failure(.usage(O.usage)))
        XCTAssertEqual(O.parse(["rewrite", "describe"]), .failure(.usage(O.usage)))
        XCTAssertEqual(O.parse(["caption"]), .failure(.usage(O.usage)))
        XCTAssertEqual(O.parse(["rewrite", "--set"]), .failure(.flagWithoutValue("--set")))
        XCTAssertEqual(O.parse(["rewrite", "--reset"]), .failure(.unknownFlag("--reset")))
    }

    private func failure(_ g: WriterOrder.Gesture, _ a: [String]) -> WriterOrder.Failure? {
        if case .failure(let f) = WriterOrder.parse(g, a) { return f }
        return nil
    }

    // ── set ────────────────────────────────────────────────────────────────────────────────

    func testSetRefusesWhatItDoesNotKnow() {
        for bad in [["volume", "on"], ["volume"], []] {
            guard case .failure(.usage) = SetOrder.parse(bad) else { return XCTFail("\(bad)") }
        }
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

    /// **`set` takes no flag**: a word of two dashes is refused by name, never read as a folder; a folder whose name begins with
    /// `--` goes after a lone `--`.
    func testSetRefusesAFlagAndTakesAFolderAfterTheSeparator() throws {
        XCTAssertEqual(SetOrder.parse(["save-images", "--x"]), .failure(.unknownFlag("--x")))
        XCTAssertEqual(SetOrder.parse(["--glass"]), .failure(.unknownFlag("--glass")))
        XCTAssertEqual(SetOrder.parse(["glass", "on", "--now"]), .failure(.unknownFlag("--now")))
        XCTAssertEqual(try SetOrder.parse(["save-images", "--", "--x"]).get(), .saveImages(folder: "--x"))
        XCTAssertEqual(try SetOrder.parse(["--", "glass", "off"]).get(), .toggle(.glass, on: false))
    }

    /// Every subcommand of `set` is read here, its own usage on a mistake, all of them for an unknown one.
    func testSetReadsTheSwitchesAndTheMemory() throws {
        for toggle in SetOrder.Toggle.allCases {
            XCTAssertEqual(try SetOrder.parse([toggle.rawValue]).get(), .toggle(toggle, on: nil))
            XCTAssertEqual(try SetOrder.parse([toggle.rawValue, "on"]).get(), .toggle(toggle, on: true))
            XCTAssertEqual(try SetOrder.parse([toggle.rawValue, "off"]).get(), .toggle(toggle, on: false))
            XCTAssertEqual(SetOrder.parse([toggle.rawValue, "yes"]), .failure(.usage("set \(toggle.rawValue) [on|off]")))
        }
        XCTAssertEqual(try SetOrder.parse(["memory"]).get(), .memory(nil))
        XCTAssertEqual(try SetOrder.parse(["memory", "default"]).get(), .memory(.standard))
        XCTAssertEqual(try SetOrder.parse(["memory", "12.5"]).get(), .memory(.gigabytes(12.5)))
        for bad in [["memory", "nan"], ["memory", "-4"], ["memory", "12", "GB"]] {
            XCTAssertEqual(SetOrder.parse(bad), .failure(.usage("set memory [GB|default]")), "\(bad)")
        }
        XCTAssertEqual(SetOrder.parse(["volume"]), .failure(.usage(SetOrder.usage)))
        XCTAssertEqual(SetOrder.usages.map(\.name), ["beta", "neural-engine", "save-images", "glass", "updates", "memory", "preview", "debug-live"])
    }

    /// The four options of Settings › Advanced › Beta, no more: a fifth has to be added here, in the app's `AppState.beta`
    /// (a `switch` the compiler checks) and in the help.
    func testTheBetaOptionsAreTheSettingsFour() {
        XCTAssertEqual(BetaOption.allCases.map(\.rawValue), ["more-solvers", "steps", "size-scales", "negative-lora"])
        XCTAssertEqual(BetaOption.allCases.map(\.key), ["more_solvers", "steps", "size_scales", "negative_lora"])
        XCTAssertEqual(BetaOption.names, "more-solvers, steps, size-scales, negative-lora")
        XCTAssertEqual(BetaOption(argument: "choose-steps"), .steps)
    }

}
