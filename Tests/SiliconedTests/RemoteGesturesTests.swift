import XCTest
@testable import Siliconed

/// **The decisions behind the gestures `silicontrol` takes from the window**, without an app: the arguments, the name of a cell, a crop's frame,
/// the `--yes` guard, and which paired Mac a word names.
final class RemoteGesturesTests: XCTestCase {
    private func parse(_ command: String, _ arguments: [String]) -> GestureOrder? {
        if case .success(let order) = GestureOrder.parse(command, arguments) { return order }
        return nil
    }

    // ── Naming a cell ──────────────────────────────────────────────────────────────────────

    func testACellIsAGridAndARankOrItsJob() {
        XCTAssertEqual(CellName("g2c3"), .cell(grid: 2, index: 3))
        XCTAssertEqual(CellName("g12c0"), .cell(grid: 12, index: 0))
        XCTAssertEqual(CellName("t5"), .job(5))
        XCTAssertEqual(CellName.name(grid: 2, cell: 3), "g2c3")
        // The names of other things, and half-written ones, name no cell.
        for word in ["g2", "g2c", "gc3", "c3", "i7", "g2c3c4", "g-1c0", "g2c-1", "", "5", "t", "tx"] { XCTAssertNil(CellName(word), word) }
        XCTAssertEqual(parse("finish", ["g2c3"]), .finish(.cell(grid: 2, index: 3)))
        XCTAssertEqual(parse("finish", ["t9"]), .finish(.job(9)))
        XCTAssertEqual(GestureOrder.parse("finish", ["i7"]), .failure(.badValue("cell", "i7")))
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
