import XCTest
@testable import Siliconed

/// **The decisions behind the gestures `silicontrol` takes from the window**, without an app: the arguments, the name of a cell, a crop's frame,
/// the `--yes` guard, and which paired Mac a word names.
final class RemoteGesturesTests: XCTestCase {
    private func parse(_ command: String, _ arguments: [String]) -> GestureOrder? {
        if case .success(let order) = GestureOrder.parse(command, arguments) { return order }
        return nil
    }

    // ── The Prompt Lab ─────────────────────────────────────────────────────────────────────

    /// `lab`: what it holds, fold or unfold it, images in, a role by its word, an image out — numbered from 1.
    func testTheLabReadsItsGesturesAndItsRoles() {
        XCTAssertEqual(parse("lab", []), .lab(.status))
        XCTAssertEqual(parse("lab", ["open"]), .lab(.unfold))
        XCTAssertEqual(parse("lab", ["close"]), .lab(.fold))
        XCTAssertEqual(parse("lab", ["detach"]), .lab(.detach))
        XCTAssertEqual(parse("lab", ["attach"]), .lab(.attach))
        XCTAssertEqual(parse("lab", ["add", "a.jpg", "i7"]), .lab(.add(["a.jpg", "i7"])))
        XCTAssertEqual(parse("lab", ["role", "2", "reference"]), .lab(.role(image: 2, .reference)))
        XCTAssertEqual(parse("lab", ["role", "1", "none"]), .lab(.role(image: 1, .neutral)))
        XCTAssertEqual(parse("lab", ["role", "1", "Start"]), .lab(.role(image: 1, .start)))
        XCTAssertEqual(parse("lab", ["remove", "3"]), .lab(.remove(image: 3)))
        // Each role's word reads back as itself.
        for role in LabInputs.Role.allCases {
            XCTAssertEqual(GestureOrder.LabAction.role(named: GestureOrder.LabAction.name(role)), role)
        }
        XCTAssertEqual(GestureOrder.parse("lab", ["role", "0", "edited"]), .failure(.badValue("image", "0")))
        XCTAssertEqual(GestureOrder.parse("lab", ["role", "1", "neutral"]), .failure(.badValue("role", "neutral")))
        XCTAssertEqual(GestureOrder.parse("lab", ["remove", "x"]), .failure(.badValue("image", "x")))
        for arguments in [["add"], ["open", "now"], ["role", "1"], ["remove"], ["fold"]] {
            XCTAssertEqual(GestureOrder.parse("lab", arguments), .failure(.usage(GestureOrder.usage("lab"))), "\(arguments)")
        }
        XCTAssertEqual(GestureOrder.parse("lab", ["--yes"]), .failure(.unknownFlag("--yes")))
    }

    /// `lab rewrite` and `lab describe N`: the bar's two buttons, answered at once or, with `--wait`, once written — `--wait` belongs
    /// to them alone.
    func testTheLabWritesInItsText() {
        XCTAssertEqual(parse("lab", ["rewrite"]), .lab(.rewrite(wait: false)))
        XCTAssertEqual(parse("lab", ["rewrite", "--wait"]), .lab(.rewrite(wait: true)))
        XCTAssertEqual(parse("lab", ["--wait", "rewrite"]), .lab(.rewrite(wait: true)))
        XCTAssertEqual(parse("lab", ["describe", "2"]), .lab(.describe(image: 2, wait: false)))
        XCTAssertEqual(parse("lab", ["describe", "1", "--wait"]), .lab(.describe(image: 1, wait: true)))
        XCTAssertEqual(GestureOrder.parse("lab", ["describe", "0"]), .failure(.badValue("image", "0")))
        XCTAssertEqual(GestureOrder.parse("lab", ["describe", "i7"]), .failure(.badValue("image", "i7")))
        for arguments in [["rewrite", "a cat"], ["describe"], ["describe", "1", "2"], ["open", "--wait"], ["--wait"], ["add", "a.jpg", "--wait"]] {
            XCTAssertEqual(GestureOrder.parse("lab", arguments), .failure(.usage(GestureOrder.usage("lab"))), "\(arguments)")
        }
        XCTAssertEqual(GestureOrder.parse("lab", ["rewrite", "--for", "krea2"]), .failure(.unknownFlag("--for")))
    }

    // ── The Settings window ────────────────────────────────────────────────────────────────

    /// `settings [PANE]`, and `settings lab [rewrite|describe] [--for ID]`: Prompt Lab on one instruction, a menu not named left alone.
    func testSettingsOpensOnAPaneOrAnInstruction() {
        XCTAssertEqual(parse("settings", []), .settings(.show(nil)))
        for pane in SettingsPane.allCases { XCTAssertEqual(parse("settings", [pane.rawValue]), .settings(.show(pane))) }
        XCTAssertEqual(parse("settings", ["Macs"]), .settings(.show(.macs)))
        XCTAssertEqual(parse("settings", ["lab", "rewrite", "--for", "krea2"]), .settings(.instruction(.rewrite, target: "krea2")))
        XCTAssertEqual(parse("settings", ["lab", "describe"]), .settings(.instruction(.describe, target: nil)))
        XCTAssertEqual(parse("settings", ["lab", "--for", "anima"]), .settings(.instruction(nil, target: "anima")))
        XCTAssertEqual(SettingsPane.allCases.map(\.rawValue), ["general", "macs", "lab", "advanced"])
        XCTAssertEqual(GestureOrder.parse("settings", ["prompt-lab"]), .failure(.badValue("pane", "prompt-lab")))
        XCTAssertEqual(GestureOrder.parse("settings", ["lab", "vary"]), .failure(.badValue("gesture", "vary")))
        let line = GestureOrder.usage("settings")
        for arguments in [["general", "rewrite"], ["macs", "--for", "krea2"], ["--for", "krea2"], ["lab", "rewrite", "now"]] {
            XCTAssertEqual(GestureOrder.parse("settings", arguments), .failure(.usage(line)), "\(arguments)")
        }
        XCTAssertEqual(GestureOrder.parse("settings", ["lab", "--for"]), .failure(.flagWithoutValue("--for")))
        XCTAssertEqual(GestureOrder.parse("settings", ["--wait"]), .failure(.unknownFlag("--wait")))
        XCTAssertTrue(GestureOrder.verbs.contains("settings"))
    }

    // ── A knob's bubble, the rack's scroll ─────────────────────────────────────────────────

    /// `bracket KNOB` opens a knob's bubble by its word, `bracket lora NAME` one LoRA's, `bracket close` closes it; `rack top|dit`.
    func testBracketNamesAKnobAndRackAPlace() {
        for (word, knob) in [("seed", GestureOrder.BracketKnob.seeds), ("detail", .detail), ("Solver", .solver), ("start", .start),
                             ("loras", .compareLoRAs)] {
            XCTAssertEqual(parse("bracket", [word]), .bracket(.open(knob)), word)
        }
        XCTAssertEqual(parse("bracket", ["lora", "incase"]), .bracket(.open(.lora("incase"))))
        XCTAssertEqual(parse("bracket", ["close"]), .bracket(.close))
        XCTAssertEqual(GestureOrder.parse("bracket", ["steps"]), .failure(.badValue("knob", "steps")))
        let line = GestureOrder.usage("bracket")
        for arguments in [[], ["lora"], ["detail", "solver"], ["close", "now"], ["lora", "a", "b"]] {
            XCTAssertEqual(GestureOrder.parse("bracket", arguments), .failure(.usage(line)), "\(arguments)")
        }
        XCTAssertEqual(parse("rack", ["top"]), .rack(.top))
        XCTAssertEqual(parse("rack", ["DiT"]), .rack(.dit))
        XCTAssertEqual(GestureOrder.parse("rack", ["vae"]), .failure(.badValue("place", "vae")))
        XCTAssertEqual(GestureOrder.parse("rack", []), .failure(.usage(GestureOrder.usage("rack"))))
        XCTAssertTrue(GestureOrder.verbs.isSuperset(of: ["bracket", "rack"]))
    }

    // ── Naming a cell ──────────────────────────────────────────────────────────────────────

    func testACellIsAGridAndARank() {
        XCTAssertEqual(CellName.name(grid: 2, cell: 3), "g2c3")
        XCTAssertFalse(GestureOrder.verbs.contains("finish"), "a cell is a finished image: nothing left to finish")
        XCTAssertEqual(parse("sheet", ["g2"]), .sheet(grid: 2))
        XCTAssertEqual(GestureOrder.parse("sheet", ["t2"]), .failure(.badValue("grid", "t2")))
    }

    // ── A crop ─────────────────────────────────────────────────────────────────────────────

    func testCropArguments() {
        XCTAssertEqual(parse("crop", ["i7", "--box", "10,20,600,700"]), .crop(image: "i7", .box(Crop.Box(x: 10, y: 20, width: 600, height: 700))))
        XCTAssertEqual(parse("crop", ["--ratio", "4:5", "i7"]), .crop(image: "i7", .ratio(Crop.Ratio(4, 5))))
        XCTAssertEqual(GestureOrder.parse("crop", ["i7", "--box", "1,2,3"]), .failure(.badValue("--box", "1,2,3")))
        XCTAssertEqual(GestureOrder.parse("crop", ["i7", "--ratio", "4x5"]), .failure(.badValue("--ratio", "4x5")))
        XCTAssertEqual(GestureOrder.parse("crop", ["i7", "--ratio"]), .failure(.flagWithoutValue("--ratio")))
        // One frame, one image.
        for arguments in [["i7"], ["--box", "0,0,512,512"], ["i7", "--box", "0,0,512,512", "--ratio", "1:1"], ["i7", "i8", "--ratio", "1:1"]] {
            guard case .failure(.usage) = GestureOrder.parse("crop", arguments) else { return XCTFail("\(arguments)") }
        }
    }

    func testACropFollowsTheEditorsFrame() {
        let box = CropSpec.box(Crop.Box(x: 100, y: 0, width: 1024, height: 1024))
        XCTAssertEqual(try box.resolve(width: 1216, height: 1024).get(), Crop.Box(x: 100, y: 0, width: 1024, height: 1024))
        // Outside the image, under the floor, the whole image: the editor offers none of them.
        XCTAssertEqual(CropSpec.box(Crop.Box(x: 300, y: 0, width: 1024, height: 1024)).resolve(width: 1216, height: 1024), .failure(.outside))
        XCTAssertEqual(CropSpec.box(Crop.Box(x: -1, y: 0, width: 600, height: 600)).resolve(width: 1216, height: 1024), .failure(.outside))
        XCTAssertEqual(CropSpec.box(Crop.Box(x: 0, y: 0, width: 511, height: 600)).resolve(width: 1216, height: 1024),
                       .failure(.belowFloor(minimum: Format.minimumSide)))
        XCTAssertEqual(CropSpec.box(Crop.Box(x: 0, y: 0, width: 1216, height: 1024)).resolve(width: 1216, height: 1024), .failure(.whole))
        // A 512² image has nothing to give.
        XCTAssertEqual(CropSpec.box(Crop.Box(x: 0, y: 0, width: 512, height: 512)).resolve(width: 512, height: 512),
                       .failure(.cannotCrop(minimum: Format.minimumSide)))
    }

    func testAProportionIsTheLargestCenteredFrame() throws {
        // A 1216 × 1024 image holds a square of 1024 in its middle, and a 4:5 as high as the grain of 4:5 allows: exact, so 816 × 1020.
        let square = try CropSpec.ratio(Crop.Ratio(1, 1)).resolve(width: 1216, height: 1024).get()
        XCTAssertEqual(square, Crop.Box(x: 96, y: 0, width: 1024, height: 1024))
        let portrait = try CropSpec.ratio(Crop.Ratio(4, 5)).resolve(width: 1216, height: 1024).get()
        XCTAssertEqual(portrait.height, 1020)
        XCTAssertEqual(portrait.width * 5, portrait.height * 4)
        XCTAssertEqual(portrait.x, (1216 - portrait.width) / 2)
        // The image's own proportion is no crop; one the image cannot hold at 512 is refused.
        XCTAssertEqual(CropSpec.ratio(Crop.Ratio(1216, 1024)).resolve(width: 1216, height: 1024), .failure(.whole))
        XCTAssertEqual(CropSpec.ratio(Crop.Ratio(16, 1)).resolve(width: 1216, height: 1024), .failure(.unsupportedRatio))
    }

    // ── The history ────────────────────────────────────────────────────────────────────────

    func testHistoryOrders() {
        XCTAssertEqual(parse("remove", ["i7", "i8"]), .remove(["i7", "i8"]))
        XCTAssertEqual(GestureOrder.parse("remove", []), .failure(.usage("remove <i7> [i8 …]")))
        XCTAssertEqual(parse("undo", []), .undo(redo: false))
        XCTAssertEqual(parse("undo", ["--redo"]), .undo(redo: true))
        XCTAssertEqual(GestureOrder.parse("undo", ["i7"]), .failure(.usage("undo [--redo]")))
        XCTAssertEqual(parse("again", ["i7"]), .again("i7"))
        XCTAssertEqual(parse("again", ["~/out/a.png"]), .again("~/out/a.png"))
        XCTAssertEqual(parse("reuse", ["i7"]), .reuse("i7"))
        XCTAssertEqual(GestureOrder.parse("again", ["i7", "i8"]), .failure(.usage("again <i7|file.png>")))
        XCTAssertEqual(GestureOrder.parse("again", ["--hard", "i7"]), .failure(.unknownFlag("--hard")))
        // A path that has a space in it is a path, not a flag.
        XCTAssertEqual(parse("reuse", ["--my file.png"]), .reuse("--my file.png"))
    }

    // ── What destroys: --yes ───────────────────────────────────────────────────────────────

    func testOnlyYesMakesAnUninstallAct() {
        XCTAssertEqual(parse("uninstall", ["z-image"]), .uninstall(.model("z-image"), confirmed: false))
        XCTAssertEqual(parse("uninstall", ["z-image", "--yes"]), .uninstall(.model("z-image"), confirmed: true))
        XCTAssertEqual(parse("uninstall", ["--yes", "lora", "flat"]), .uninstall(.lora("flat"), confirmed: true))
        XCTAssertEqual(parse("uninstall", ["lora", "flat"]), .uninstall(.lora("flat"), confirmed: false))
        XCTAssertEqual(parse("uninstall", ["downloads"]), .uninstall(.downloads, confirmed: false))
        XCTAssertEqual(parse("uninstall", ["downloads", "--yes"]), .uninstall(.downloads, confirmed: true))
        for arguments in [[], ["lora"], ["a", "b"], ["lora", "a", "b"]] {
            guard case .failure(.usage) = GestureOrder.parse("uninstall", arguments) else { return XCTFail("\(arguments)") }
        }
    }

    func testTheLibraryMovesOnlyWithYes() {
        XCTAssertEqual(parse("library", []), .library(.status, confirmed: false))
        XCTAssertEqual(parse("library", ["move", "/Volumes/Disk"]), .library(.move("/Volumes/Disk"), confirmed: false))
        XCTAssertEqual(parse("library", ["move", "/Volumes/Disk", "--yes"]), .library(.move("/Volumes/Disk"), confirmed: true))
        XCTAssertEqual(parse("library", ["home"]), .library(.home, confirmed: false))
        XCTAssertEqual(parse("library", ["--yes", "home"]), .library(.home, confirmed: true))
        XCTAssertEqual(parse("library", ["open", "/Volumes/Disk/Siliconed"]), .library(.open("/Volumes/Disk/Siliconed"), confirmed: false))
        XCTAssertEqual(parse("library", ["local"]), .library(.local, confirmed: false))
        // Nothing moves in `open` and `local`: `--yes` is not theirs, and nothing is silently ignored.
        XCTAssertEqual(GestureOrder.parse("library", ["local", "--yes"]), .failure(.unknownFlag("--yes")))
        XCTAssertEqual(GestureOrder.parse("library", ["open", "/x", "--yes"]), .failure(.unknownFlag("--yes")))
        XCTAssertEqual(GestureOrder.parse("library", ["move"]), .failure(.usage(GestureOrder.usage("library"))))
        XCTAssertEqual(GestureOrder.parse("remove", ["i7", "--yes"]), .failure(.unknownFlag("--yes")))
        XCTAssertTrue(GestureOrder.takesYes("uninstall", ["x"]))
        XCTAssertFalse(GestureOrder.takesYes("library", ["status"]))
    }

    // ── Models, Macs ───────────────────────────────────────────────────────────────────────

    func testImportComputeAndMacs() {
        XCTAssertEqual(parse("import", ["a.safetensors", "b.gguf"]), .importFiles(["a.safetensors", "b.gguf"]))
        XCTAssertEqual(GestureOrder.parse("import", []), .failure(.usage(GestureOrder.usage("import"))))
        XCTAssertEqual(parse("compute", ["This"]), .compute(.thisMac))
        XCTAssertEqual(parse("compute", ["Mac mini"]), .compute(.mac("Mac mini")))
        XCTAssertEqual(parse("macs", []), .macs(.list))
        XCTAssertEqual(parse("macs", ["connect", "mini"]), .macs(.connect("mini")))
        XCTAssertEqual(parse("macs", ["disconnect", "mini"]), .macs(.disconnect("mini")))
        XCTAssertEqual(GestureOrder.parse("macs", ["pair", "mini"]), .failure(.usage(GestureOrder.usage("macs"))))
        // Every verb the parser knows has a usage line of its own.
        for verb in GestureOrder.verbs { XCTAssertTrue(GestureOrder.usage(verb).hasPrefix(verb), verb) }
    }

    func testAMacIsNamedByItsNameOrItsID() {
        let macs = [(id: "A1", name: "Mac mini"), (id: "B2", name: "Mac Studio de Léa"), (id: "C3", name: "Mac mini 2"), (id: "D4", name: "Dup"), (id: "E5", name: "dup")]
        XCTAssertEqual(MacName.match("mac mini", among: macs)?.id, "A1")
        XCTAssertEqual(MacName.match("mac studio de lea", among: macs)?.id, "B2")
        XCTAssertEqual(MacName.match("C3", among: macs)?.id, "C3")
        XCTAssertNil(MacName.match("mini", among: macs))
        // Two Macs of one name: the command cannot guess; the id names one.
        XCTAssertNil(MacName.match("dup", among: macs))
        XCTAssertEqual(MacName.match("E5", among: macs)?.id, "E5")
    }
}
