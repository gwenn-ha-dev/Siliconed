import XCTest
@testable import Siliconed

/// **The crop frame** (`Crop`): which rectangles exist on an image — the floor, the proportions, the
/// edges. Pure: no image is decoded.
final class CropTests: XCTestCase {
    private func frame(_ w: Int, _ h: Int) -> Crop.Frame { Crop.Frame(width: w, height: h) }

    private func assertValid(_ b: Crop.Box, in f: Crop.Frame, ratio: Crop.Ratio?, _ note: String = "",
                             file: StaticString = #filePath, line: UInt = #line) {
        let m = Format.minimumSide
        XCTAssertGreaterThanOrEqual(b.x, 0, note, file: file, line: line)
        XCTAssertGreaterThanOrEqual(b.y, 0, note, file: file, line: line)
        XCTAssertLessThanOrEqual(b.maxX, f.width, note, file: file, line: line)
        XCTAssertLessThanOrEqual(b.maxY, f.height, note, file: file, line: line)
        XCTAssertGreaterThanOrEqual(b.width, m, note, file: file, line: line)
        XCTAssertGreaterThanOrEqual(b.height, m, note, file: file, line: line)
        if let r = ratio {
            XCTAssertEqual(b.height, r.height(forWidth: b.width), "\(note) \(b)", file: file, line: line)
        }
    }

    // ── what can be cropped ──

    func testTheFloorIsTheRenderers() {
        XCTAssertEqual(Crop.Frame(width: 1024, height: 1024).minimum, Format.minimumSide)
        XCTAssertEqual(Format.minimumSide, 512)
    }

    func testAnImageUnderTheFloorOrAtItCannotBeCropped() {
        XCTAssertTrue(frame(1024, 1024).canCrop)
        XCTAssertTrue(frame(512, 700).canCrop)        // one side to give
        XCTAssertFalse(frame(512, 512).canCrop)
        XCTAssertFalse(frame(500, 2000).canCrop)       // a side already under the floor
        XCTAssertFalse(frame(2000, 511).canCrop)
    }

    // ── proportions ──

    func testTheOfferedProportionsAreTheFormatPickers() {
        XCTAssertEqual(Crop.Ratio.offered.map(\.label), ["1:1", "7:9", "13:19", "4:7", "9:7", "19:13", "7:4"])
    }

    func testAProportionTheFloorCannotHoldIsNotSupported() {
        XCTAssertTrue(frame(1024, 1536).supports(.fixed(Crop.Ratio(4, 7))))
        XCTAssertFalse(frame(1500, 600).supports(.fixed(Crop.Ratio(4, 7))))   // 896 px of height needed
        XCTAssertTrue(frame(1500, 600).supports(.fixed(Crop.Ratio(7, 4))))
        XCTAssertTrue(frame(1500, 600).supports(.free))
        XCTAssertTrue(frame(1023, 769).supports(.original))
    }

    func testAnAwkwardProportionIsNotReduced() {
        let r = Crop.Ratio(1023, 769)
        XCTAssertEqual(r.step, 1)
        XCTAssertEqual(r.height(forWidth: 1023), 769)
        XCTAssertEqual(Crop.Ratio(1536, 1024).label, "3:2")
    }

    // ── resizing, free ──

    func testFreeSidesStopAtTheFloorAndTheEdges() {
        let f = frame(1024, 1024)
        var b = f.whole
        b = f.resize(b, edges: .right, to: CGPoint(x: 100, y: 0), ratio: nil)
        XCTAssertEqual(b, Crop.Box(x: 0, y: 0, width: 512, height: 1024))     // not 100 wide
        b = f.resize(b, edges: .left, to: CGPoint(x: -50, y: 0), ratio: nil)
        XCTAssertEqual(b.x, 0)
        b = f.resize(b, edges: [.bottom, .right], to: CGPoint(x: 5000, y: 5000), ratio: nil)
        XCTAssertEqual(b, Crop.Box(x: 0, y: 0, width: 1024, height: 1024))
        b = f.resize(b, edges: [.top, .left], to: CGPoint(x: 900, y: 900), ratio: nil)
        XCTAssertEqual(b, Crop.Box(x: 512, y: 512, width: 512, height: 512))
    }

    func testMovingKeepsTheFrameInside() {
        let f = frame(1024, 1536)
        let b = Crop.Box(x: 100, y: 100, width: 600, height: 700)
        XCTAssertEqual(f.move(b, dx: -500, dy: 5000), Crop.Box(x: 0, y: 836, width: 600, height: 700))
        XCTAssertEqual(f.move(b, dx: 5000, dy: -5000), Crop.Box(x: 424, y: 0, width: 600, height: 700))
    }

    // ── resizing, with a proportion ──

    func testACornerKeepsTheOppositeCornerAndTheProportion() {
        let f = frame(1536, 1024)
        let r = Crop.Ratio(1, 1)
        var b = Crop.Box(x: 100, y: 100, width: 600, height: 600)
        b = f.resize(b, edges: [.right, .bottom], to: CGPoint(x: 900, y: 800), ratio: r)
        XCTAssertEqual(b.x, 100); XCTAssertEqual(b.y, 100)
        XCTAssertEqual(b.width, 800)                       // the larger of the two wishes
        assertValid(b, in: f, ratio: r)
        b = f.resize(b, edges: [.left, .top], to: CGPoint(x: -300, y: -300), ratio: r)
        XCTAssertEqual(b.maxX, 900); XCTAssertEqual(b.maxY, 900)
        XCTAssertEqual(b.width, 900)                       // the top edge is the nearer wall
        assertValid(b, in: f, ratio: r)
    }

    func testASideKeepsItsOppositeAndCentersTheOtherAxis() {
        let f = frame(1536, 1536)
        let r = Crop.Ratio(7, 9)
        var b = Crop.Box(x: 200, y: 400, width: 700, height: 900)
        b = f.resize(b, edges: .right, to: CGPoint(x: 900, y: 0), ratio: r)
        XCTAssertEqual(b.x, 200)
        XCTAssertEqual(b.width, 700)
        b = f.resize(b, edges: .right, to: CGPoint(x: 200 + 560, y: 0), ratio: r)
        XCTAssertEqual(b.width, 560); XCTAssertEqual(b.height, 720)
        XCTAssertEqual(b.y, 490)                           // still centered on y = 850
        assertValid(b, in: f, ratio: r)
    }

    func testASideNeverGrowsOutOfTheOtherAxis() {
        let f = frame(1536, 1024)
        let r = Crop.Ratio(1, 1)
        // Centered near the top: the vertical room is what limits the width.
        var b = Crop.Box(x: 0, y: 0, width: 600, height: 600)
        b = f.resize(b, edges: .right, to: CGPoint(x: 1500, y: 0), ratio: r)
        assertValid(b, in: f, ratio: r)
        XCTAssertLessThanOrEqual(b.height, 1024)
    }

    func testTheProportionNeverGoesBelowTheFloor() {
        let f = frame(1536, 1536)
        for r in Crop.Ratio.offered {
            var b = f.fit(f.whole, to: r)!
            b = f.resize(b, edges: [.right, .bottom], to: CGPoint(x: 0, y: 0), ratio: r)
            assertValid(b, in: f, ratio: r, r.label)
            XCTAssertTrue(b.width == 512 || b.height == 512 || min(b.width, b.height) < 512 + max(r.num, r.den) * 2, r.label)
        }
    }

    // ── changing the proportion ──

    func testFitTakesTheLargestInsideTheFrame() {
        let f = frame(1536, 1024)
        let b = f.fit(f.whole, to: Crop.Ratio(1, 1))!
        XCTAssertEqual(b, Crop.Box(x: 256, y: 0, width: 1024, height: 1024))
        let c = f.fit(b, to: Crop.Ratio(7, 9))!
        XCTAssertEqual(c, Crop.Box(x: 373, y: 4, width: 791, height: 1017))
        assertValid(c, in: f, ratio: Crop.Ratio(7, 9))
        XCTAssertGreaterThanOrEqual(c.x, b.x); XCTAssertLessThanOrEqual(c.maxX, b.maxX)
    }

    func testFitFallsBackWhereTheFloorLeavesNoRoom() {
        let f = frame(2000, 2000)
        let small = Crop.Box(x: 900, y: 900, width: 600, height: 520)
        let b = f.fit(small, to: Crop.Ratio(4, 7))!        // 4:7 needs 896 px of height: not inside `small`
        assertValid(b, in: f, ratio: Crop.Ratio(4, 7))
    }

    func testFitRefusesWhatTheImageCannotHold() {
        let f = frame(1500, 600)
        XCTAssertNil(f.fit(f.whole, to: Crop.Ratio(4, 7)))
    }

    // ── a thousand drags ──

    func testRandomDragsNeverBreakTheRules() {
        var rng = SystemRandomNumberGenerator()
        let shapes: [Crop.Ratio?] = [nil] + Crop.Ratio.offered.map { Optional($0) } + [Crop.Ratio(1023, 769)]
        let edgeSets: [Crop.Edges] = [.left, .right, .top, .bottom, [.left, .top], [.right, .top], [.left, .bottom], [.right, .bottom]]
        for (w, h) in [(1024, 1024), (1536, 1024), (1024, 1536), (3072, 512), (700, 2000), (513, 600)] {
            let f = frame(w, h)
            for ratio in shapes {
                var box = f.whole
                if let r = ratio { guard let fitted = f.fit(box, to: r) else { continue }; box = fitted }
                for step in 0..<150 {
                    let p = CGPoint(x: Double.random(in: -200...Double(w + 200), using: &rng),
                                    y: Double.random(in: -200...Double(h + 200), using: &rng))
                    if step % 5 == 0 {
                        box = f.move(box, dx: Int.random(in: -400...400, using: &rng), dy: Int.random(in: -400...400, using: &rng))
                    } else {
                        box = f.resize(box, edges: edgeSets.randomElement(using: &rng)!, to: p, ratio: ratio)
                    }
                    assertValid(box, in: f, ratio: ratio, "\(w)×\(h) \(ratio?.label ?? "free")")
                }
            }
        }
    }
}
