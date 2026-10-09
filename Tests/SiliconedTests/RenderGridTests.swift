import XCTest
@testable import Siliconed

/// **The XY grid** — its cells, its refusals, its sheet's geometry, pure: what the app and
/// `silicontrol grid` queue, one render per cell, before any weight is read.
final class RenderGridTests: XCTestCase {

    private let library = "woman posing in a {library|greenhouse}"

    private func refusal(_ base: RenderGrid.Base, x: RenderGrid.Axis, y: RenderGrid.Axis? = nil,
                         limit: Int = RenderGrid.maxCells,
                         file: StaticString = #filePath, line: UInt = #line) -> EngineError? {
        do { _ = try RenderGrid(base, x: x, y: y, limit: limit); XCTFail("accepted", file: file, line: line); return nil }
        catch { return error }
    }

    /// The lot's own grid: X the prompt's group, Y two seeds — four cells, row by row.
    func testAPromptGroupBySeedsMakesFourCellsInReadingOrder() throws {
        let g = try RenderGrid(.init(prompt: library, seed: 42, steps: 8), x: .prompt(group: 0), y: .seeds([42, 7]))
        XCTAssertEqual(g.columns, 2)
        XCTAssertEqual(g.rows, 2)
        XCTAssertEqual(g.count, 4)
        XCTAssertEqual(g.cells.map(\.prompt), ["woman posing in a library", "woman posing in a greenhouse",
                                               "woman posing in a library", "woman posing in a greenhouse"])
        XCTAssertEqual(g.cells.map(\.seed), [42, 42, 7, 7])
        XCTAssertEqual(g.cells.map { [$0.column, $0.row] }, [[0, 0], [1, 0], [0, 1], [1, 1]])
        XCTAssertEqual(Set(g.cells.map(\.steps)), [8])
        XCTAssertEqual(g.values(of: g.x), [.alternative("library"), .alternative("greenhouse")])
        XCTAssertEqual(g.title, library)
    }

    /// Without Y: one row. Each axis changes only its own setting.
    func testEachAxisChangesItsSettingAndNothingElse() throws {
        let base = RenderGrid.Base(prompt: "woman posing in a library", seed: 42, steps: 8, loraStrengths: [0.8, 0.5])
        let steps = try RenderGrid(base, x: .steps([4, 8, 12]))
        XCTAssertEqual(steps.rows, 1)
        XCTAssertEqual(steps.cells.map(\.steps), [4, 8, 12])
        XCTAssertEqual(Set(steps.cells.map(\.seed)), [42])
        XCTAssertEqual(Set(steps.cells.map(\.loraStrengths)), [[0.8, 0.5]])

        let lora = try RenderGrid(base, x: .loraStrength(slot: 1, values: [0, 0.5, 1]), y: .consecutiveSeeds(count: 2))
        XCTAssertEqual(lora.cells.map(\.loraStrengths), [[0.8, 0], [0.8, 0.5], [0.8, 1], [0.8, 0], [0.8, 0.5], [0.8, 1]])
        XCTAssertEqual(lora.cells.map(\.seed), [42, 42, 42, 43, 43, 43])
        XCTAssertEqual(lora.values(of: lora.y!), [.seed(42), .seed(43)])
        XCTAssertEqual(Set(lora.cells.map(\.prompt)), ["woman posing in a library"])
    }

    /// Two groups, one per axis; escapes stay characters.
    func testTwoGroupsMakeTheTwoAxes() throws {
        let g = try RenderGrid(.init(prompt: "a {red|blue} dress in a {library|greenhouse} \\{x\\}", seed: 1, steps: 8),
                               x: .prompt(group: 1), y: .prompt(group: 0))
        XCTAssertEqual(g.cells.map(\.prompt), ["a red dress in a library {x}", "a red dress in a greenhouse {x}",
                                               "a blue dress in a library {x}", "a blue dress in a greenhouse {x}"])
        XCTAssertEqual(g.title, "a {red|blue} dress in a {library|greenhouse} {x}")
    }

    /// The refusals, each before a cell is built.
    func testWhatIsNotAGridIsRefused() {
        let base = RenderGrid.Base(prompt: library, seed: 42, steps: 8, loraStrengths: [0.8])
        let plain = RenderGrid.Base(prompt: "woman posing in a library", seed: 42, steps: 8)
        // A group left out would stack two images in each cell: refused, not multiplied.
        XCTAssertEqual(refusal(base, x: .seeds([1, 2])), .gridRefused(reason: .groupNotOnAxis))
        XCTAssertEqual(refusal(base, x: .prompt(group: 1)), .gridRefused(reason: .noSuchGroup))
        XCTAssertEqual(refusal(plain, x: .loraStrength(slot: 0, values: [1])), .gridRefused(reason: .noSuchLoRA))
        XCTAssertEqual(refusal(plain, x: .seeds([])), .gridRefused(reason: .emptyAxis))
        XCTAssertEqual(refusal(plain, x: .consecutiveSeeds(count: 0)), .gridRefused(reason: .emptyAxis))
        XCTAssertEqual(refusal(plain, x: .steps([8, 0])), .gridRefused(reason: .invalidValue))
        XCTAssertEqual(refusal(base, x: .prompt(group: 0), y: .loraStrength(slot: 0, values: [.nan])),
                       .gridRefused(reason: .invalidValue))
        // The seed is one setting, however it is listed.
        XCTAssertEqual(refusal(plain, x: .seeds([1, 2]), y: .consecutiveSeeds(count: 3)), .gridRefused(reason: .sameAxisTwice))
        XCTAssertEqual(refusal(base, x: .prompt(group: 0), y: .prompt(group: 0)), .gridRefused(reason: .sameAxisTwice))
        XCTAssertEqual(refusal(.init(prompt: "{a|b", seed: 1, steps: 8), x: .seeds([1])),
                       .promptSyntax(position: 1, reason: .unclosedGroup))
    }

    /// The ceiling: the queue's 64, with the numbers — and no million-seed array on the way.
    func testTheCeilingIsSaidWithItsNumbers() throws {
        let plain = RenderGrid.Base(prompt: "woman posing in a library", seed: 42, steps: 8)
        XCTAssertEqual(try RenderGrid(plain, x: .consecutiveSeeds(count: 8), y: .steps(Array(1...8))).count, 64)
        XCTAssertEqual(refusal(plain, x: .consecutiveSeeds(count: 9), y: .steps(Array(1...8))),
                       .tooManyGridCells(count: 72, max: 64))
        XCTAssertEqual(refusal(plain, x: .consecutiveSeeds(count: 1_000_000)), .tooManyGridCells(count: 1_000_000, max: 64))
        XCTAssertEqual(refusal(plain, x: .consecutiveSeeds(count: .max), y: .steps([1, 2])), .tooManyGridCells(count: .max, max: 64))
        XCTAssertEqual(refusal(plain, x: .steps([4, 8]), limit: 1), .tooManyGridCells(count: 2, max: 1))
        // The prompt's own product is no limit: its groups are the axes.
        let big = "{" + (1...40).map(String.init).joined(separator: "|") + "}"
        XCTAssertEqual(try RenderGrid(.init(prompt: big, seed: 1, steps: 8), x: .prompt(group: 0)).count, 40)
    }

    /// Consecutive seeds wrap past the last one, like a batch.
    func testConsecutiveSeedsCountFromTheBaseSeed() throws {
        let g = try RenderGrid(.init(prompt: "p", seed: .max, steps: 8), x: .consecutiveSeeds(count: 2))
        XCTAssertEqual(g.cells.map(\.seed), [.max, 0])
    }

    /// The command line's axes.
    func testAnAxisReadsAsTheCommandWritesIt() throws {
        XCTAssertEqual(try RenderGrid.axis("prompt"), .prompt(group: 0))
        XCTAssertEqual(try RenderGrid.axis(" prompt:2 "), .prompt(group: 1))
        XCTAssertEqual(try RenderGrid.axis("seed=42, 7,1000"), .seeds([42, 7, 1000]))
        XCTAssertEqual(try RenderGrid.axis("seeds=4"), .consecutiveSeeds(count: 4))
        XCTAssertEqual(try RenderGrid.axis("steps=6,8,10"), .steps([6, 8, 10]))
        XCTAssertEqual(try RenderGrid.axis("lora=0.4,1", loras: ["flat", "ink"]), .loraStrength(slot: 0, values: [0.4, 1]))
        XCTAssertEqual(try RenderGrid.axis("lora:ink=0.5", loras: ["flat", "ink"]), .loraStrength(slot: 1, values: [0.5]))
        XCTAssertEqual(try RenderGrid.axis("lora:2=0.5", loras: ["flat", "ink"]), .loraStrength(slot: 1, values: [0.5]))
        XCTAssertEqual(try RenderGrid.axis("STEPS=8"), .steps([8]))
        XCTAssertEqual(try RenderGrid.axis("model=z-image, qwen-image-2.1"), .models(["z-image", "qwen-image-2.1"]))
        XCTAssertEqual(try RenderGrid.axis("image"), .images)
        XCTAssertEqual(try RenderGrid.axis("loras=none,flat"), .addedLoRAs(["", "flat"], strength: 1))
        XCTAssertEqual(try RenderGrid.axis("loras:0.6=flat,ink"), .addedLoRAs(["flat", "ink"], strength: 0.6))
        for bad in ["", "model=", "model=a,,b", "loras=", "loras:x=flat", "image=1", "image:2", "prompt=1", "prompt:0", "seed", "seed=", "seed=4,,5", "seed=-1", "seeds=4,5",
                    "steps=6.5", "lora=0;4", "seed:2=4"] {
            XCTAssertThrowsError(try RenderGrid.axis(bad, loras: ["flat"]), bad) {
                XCTAssertEqual($0 as? EngineError, .gridRefused(reason: .unreadableAxis), bad)
            }
        }
        for missing in [("lora=1", [String]()), ("lora:ink=1", ["flat"]), ("lora:3=1", ["flat", "ink"])] {
            XCTAssertThrowsError(try RenderGrid.axis(missing.0, loras: missing.1)) {
                XCTAssertEqual($0 as? EngineError, .gridRefused(reason: .noSuchLoRA))
            }
        }
    }

    /// A model axis: each cell its model, at its model's own step count unless steps are an axis;
    /// a LoRA stack refused, a model named twice refused.
    func testAModelAxisLeavesTheStepsToEachModel() throws {
        let base = RenderGrid.Base(prompt: "woman posing in a library", seed: 42, steps: 8)
        let g = try RenderGrid(base, x: .models(["z-image", "qwen-image-2.1"]), y: .consecutiveSeeds(count: 2))
        XCTAssertEqual(g.cells.map(\.model), ["z-image", "qwen-image-2.1", "z-image", "qwen-image-2.1"])
        XCTAssertEqual(Set(g.cells.map(\.steps)), [nil])
        XCTAssertEqual(g.values(of: g.x), [.model("z-image"), .model("qwen-image-2.1")])
        let stepped = try RenderGrid(base, x: .models(["z-image", "klein-4b"]), y: .steps([4, 8]))
        XCTAssertEqual(stepped.cells.map(\.steps), [4, 4, 8, 8])
        XCTAssertEqual(Set(try RenderGrid(base, x: .consecutiveSeeds(count: 2)).cells.map(\.model)), [nil])

        let lora = RenderGrid.Base(prompt: "woman posing in a library", seed: 42, steps: 8, loraStrengths: [1])
        XCTAssertEqual(refusal(lora, x: .models(["z-image"])), .gridRefused(reason: .loraAcrossModels))
        XCTAssertEqual(refusal(base, x: .models([])), .gridRefused(reason: .emptyAxis))
        XCTAssertEqual(refusal(base, x: .models(["z-image", "z-image"])), .gridRefused(reason: .invalidValue))
        XCTAssertEqual(refusal(base, x: .models(["z-image"]), y: .models(["anima"])), .gridRefused(reason: .sameAxisTwice))
    }

    /// A LoRA axis: each cell adds one LoRA (or none) to the stack, at the axis's strength; the stack's
    /// strength axis is another setting; refused across models.
    func testALoRAAxisAddsOneLoRAPerCell() throws {
        let base = RenderGrid.Base(prompt: "woman posing in a library", seed: 42, steps: 8, loraStrengths: [0.8])
        let g = try RenderGrid(base, x: .addedLoRAs(["", "flat", "ink"], strength: 0.6), y: .loraStrength(slot: 0, values: [0.5, 1]))
        XCTAssertEqual(g.cells.map(\.addedLoRA), [nil, "flat", "ink", nil, "flat", "ink"])
        XCTAssertEqual(g.cells.map(\.addedStrength), [0, 0.6, 0.6, 0, 0.6, 0.6])
        XCTAssertEqual(g.cells.map(\.loraStrengths), [[0.5], [0.5], [0.5], [1], [1], [1]])
        XCTAssertEqual(g.values(of: g.x), [.addedLoRA(""), .addedLoRA("flat"), .addedLoRA("ink")])
        XCTAssertEqual(Set(try RenderGrid(base, x: .consecutiveSeeds(count: 2)).cells.map(\.addedLoRA)), [nil])
        let plain = RenderGrid.Base(prompt: "woman posing in a library", seed: 42, steps: 8)
        XCTAssertEqual(refusal(plain, x: .addedLoRAs(["flat"], strength: 1), y: .models(["z-image"])),
                       .gridRefused(reason: .loraAcrossModels))
        XCTAssertEqual(refusal(plain, x: .addedLoRAs([], strength: 1)), .gridRefused(reason: .emptyAxis))
        XCTAssertEqual(refusal(plain, x: .addedLoRAs(["flat", "flat"], strength: 1)), .gridRefused(reason: .invalidValue))
    }

    /// An image axis: as many cells as the edit has images, each starting from one alone.
    func testAnImageAxisSplitsTheEdit() throws {
        let base = RenderGrid.Base(prompt: "change her jacket to red", seed: 42, steps: 6, images: 3)
        let g = try RenderGrid(base, x: .images, y: .consecutiveSeeds(count: 2))
        XCTAssertEqual(g.columns, 3)
        XCTAssertEqual(g.cells.map(\.image), [0, 1, 2, 0, 1, 2])
        XCTAssertEqual(g.values(of: .images), [.image(0), .image(1), .image(2)])
        XCTAssertEqual(Set(g.cells.map(\.steps)), [6])
        XCTAssertEqual(Set(try RenderGrid(base, x: .consecutiveSeeds(count: 2)).cells.map(\.image)), [nil])
        XCTAssertEqual(refusal(.init(prompt: "p", seed: 1, steps: 6), x: .images), .gridRefused(reason: .emptyAxis))
    }

    // ── the sheet ──

    /// 2×2 at 512²: the cells at their size, the labels around them, everything inside the sheet,
    /// nothing overlapping.
    func testASmallSheetKeepsTheCellsAtTheirSize() throws {
        let g = try RenderGrid(.init(prompt: library, seed: 42, steps: 8), x: .prompt(group: 0), y: .seeds([42, 7]))
        let s = g.sheet(imageWidth: 512, imageHeight: 512)
        XCTAssertEqual([s.cellWidth, s.cellHeight], [512, 512])
        let bounds = CGRect(x: 0, y: 0, width: s.width, height: s.height)
        var rects = g.cells.map { s.cell(column: $0.column, row: $0.row) }
        rects += (0..<s.columns).map(s.columnLabel) + (0..<s.rows).map(s.rowLabel) + [s.title]
        for (i, r) in rects.enumerated() {
            XCTAssertTrue(bounds.contains(r), "\(r) out of \(bounds)")
            for o in rects[(i + 1)...] { XCTAssertFalse(r.intersects(o), "\(r) over \(o)") }
        }
        // Reading order: the second cell right of the first, the third under it.
        XCTAssertEqual(s.cell(column: 1, row: 0).minX, s.cell(column: 0, row: 0).maxX + CGFloat(s.gap))
        XCTAssertEqual(s.cell(column: 0, row: 1).minY, s.cell(column: 0, row: 0).maxY + CGFloat(s.gap))
        XCTAssertLessThan(s.width * s.height, 2_000_000)
    }

    /// Without Y, no row labels; the cells start at the margin.
    func testWithoutYThereIsNoRowLabel() {
        let s = RenderGrid.Sheet(columns: 3, rows: 1, imageWidth: 1024, imageHeight: 1024, rowLabels: false)
        XCTAssertEqual(s.rowLabelWidth, 0)
        XCTAssertEqual(Int(s.cell(column: 0, row: 0).minX), s.margin)
        XCTAssertEqual(s.cellWidth, 512, "a 1024² render is reduced to 512 on the sheet")
    }

    /// The largest grid stays a few tens of MB: every cell together within `maxCellsArea`, and the
    /// proportions of the render kept.
    func testTheLargestSheetStaysSmall() {
        for (w, h) in [(512, 512), (1024, 1024), (832, 1216), (1536, 1024), (1024, 1536)] {
            for (c, r) in [(8, 8), (64, 1), (1, 64), (4, 16)] {
                let s = RenderGrid.Sheet(columns: c, rows: r, imageWidth: w, imageHeight: h, rowLabels: r > 1)
                XCTAssertLessThanOrEqual(c * r * s.cellWidth * s.cellHeight, RenderGrid.Sheet.maxCellsArea, "\(w)x\(h) \(c)x\(r)")
                XCTAssertLessThanOrEqual(max(s.cellWidth, s.cellHeight), RenderGrid.Sheet.maxCellSide)
                XCTAssertEqual(Double(s.cellWidth) / Double(s.cellHeight), Double(w) / Double(h), accuracy: 0.01)
                // RGBA while it is drawn: under 80 MB.
                XCTAssertLessThan(s.width * s.height * 4, 80_000_000, "\(w)x\(h) \(c)x\(r): \(s.width)x\(s.height)")
            }
        }
        let s = RenderGrid.Sheet(columns: 8, rows: 8, imageWidth: 512, imageHeight: 512, rowLabels: true)
        XCTAssertEqual(s.cellWidth, 432)
    }
}
