import XCTest
@testable import Siliconed

/// **A series' axes and the words it varies** — where each thing that varies goes, what a new axis replaces, and the
/// groups of a prompt rewritten where they are typed. Pure: what « Generate » and « Vary This Word… » rest on.
final class SeriesLayoutTests: XCTestCase {

    private let flat = SeriesKey.loraStrength("/loras/flat.lora.silicon")

    /// Nothing placed: the columns first, then the rows, in the order present; beyond two, the excess.
    func testWhatVariesFillsTheColumnsThenTheRows() {
        XCTAssertEqual(SeriesLayout(present: [], placed: [nil, nil]).axes, [])
        let one = SeriesLayout(present: [.group(0)], placed: [nil, nil])
        XCTAssertEqual(one.columns, .group(0))
        XCTAssertNil(one.rows)
        let two = SeriesLayout(present: [.group(0), .seeds], placed: [])
        XCTAssertEqual([two.columns, two.rows], [.group(0), .seeds])
        XCTAssertEqual(two.excess, [])
        let three = SeriesLayout(present: [.group(0), .seeds, flat], placed: [nil, nil])
        XCTAssertEqual(three.axes, [.group(0), .seeds])
        XCTAssertEqual(three.excess, [flat])
    }

    /// What the hand placed keeps its place; a key placed but no longer present frees it; one axis alone is the columns.
    func testAPlacedAxisKeepsItsPlace() {
        let placed = SeriesLayout(present: [.group(0), .seeds], placed: [.seeds, nil])
        XCTAssertEqual([placed.columns, placed.rows], [.seeds, .group(0)])
        let rows = SeriesLayout(present: [.group(0), flat], placed: [nil, .group(0)])
        XCTAssertEqual([rows.columns, rows.rows], [flat, .group(0)])
        let gone = SeriesLayout(present: [.group(0)], placed: [.seeds, .group(0)])
        XCTAssertEqual(gone.columns, .group(0))
        XCTAssertNil(gone.rows)
        let twice = SeriesLayout(present: [.seeds, .detail], placed: [.seeds, .seeds])
        XCTAssertEqual([twice.columns, twice.rows], [.seeds, .detail])
        XCTAssertEqual(rows.slot(of: .group(0)), 1)
        XCTAssertNil(rows.slot(of: .seeds))
    }

    /// The bubble offers the free place first: a new axis replaces nothing unless the hand says so.
    func testANewAxisGoesWhereItReplacesNothing() {
        XCTAssertEqual(SeriesLayout(present: [], placed: []).freeSlot, 0)
        XCTAssertEqual(SeriesLayout(present: [.group(0)], placed: []).freeSlot, 1)
        let (placed, replaced) = SeriesLayout.place(flat, at: 1, present: [.group(0)], placed: [nil, nil])
        XCTAssertEqual(placed, [.group(0), flat])
        XCTAssertNil(replaced)
    }

    /// Placed where another axis is, a key replaces it and names it; moved from the other place, it frees that one.
    func testPlacingWhereAnAxisIsReplacesIt() {
        let (placed, replaced) = SeriesLayout.place(flat, at: 1, present: [.group(0), .seeds], placed: [nil, nil])
        XCTAssertEqual(placed, [.group(0), flat])
        XCTAssertEqual(replaced, .seeds)
        let moved = SeriesLayout.place(.seeds, at: 0, present: [.group(0), .seeds], placed: [nil, nil])
        XCTAssertEqual(moved.placed, [.seeds, nil])
        XCTAssertEqual(moved.replaced, .group(0))
        let same = SeriesLayout.place(.seeds, at: 1, present: [.group(0), .seeds], placed: [nil, nil])
        XCTAssertEqual(same.placed, [.group(0), .seeds])
        XCTAssertNil(same.replaced)
        let layout = SeriesLayout(present: [.group(0), .seeds], placed: [nil, nil])
        XCTAssertEqual(SeriesLayout(present: [.group(0), .seeds], placed: layout.swapped).axes, [.seeds, .group(0)])
    }

    /// The groups where they are typed: braces included, escapes skipped as the parser skips them; none over an error.
    func testTheGroupsAreFoundWhereTheyAreTyped() {
        let p = "a {red|blue} dress \\{not\\} in a {library|} \\|"
        let spans = PromptAlternatives.spans(p)
        XCTAssertEqual(spans.map { String(p[$0.range]) }, ["{red|blue}", "{library|}"])
        XCTAssertEqual(spans[0].alternatives.map { String(p[$0]) }, ["red", "blue"])
        XCTAssertEqual(spans[1].alternatives.map { String(p[$0]) }, ["library", ""])
        XCTAssertEqual(PromptAlternatives.spans("a {red|blue"), [])
        XCTAssertEqual(PromptAlternatives.spans("no group"), [])
        let escaped = "{a\\|b|c}"
        XCTAssertEqual(PromptAlternatives.spans(escaped).first?.alternatives.map { String(escaped[$0]) }, ["a\\|b", "c"])
    }

    /// A group written, replaced, collapsed: each reads back as what it says.
    func testAGroupIsWrittenAndRewrittenWhereItIs() throws {
        let g = PromptAlternatives.group(["red", "a|b", "{x}"])
        XCTAssertEqual(g, "{red|a\\|b|\\{x\\}}")
        XCTAssertEqual(try PromptAlternatives(g).groups, [["red", "a|b", "{x}"]])
        let p = "a {red|blue} dress in a {library|greenhouse}"
        XCTAssertEqual(PromptAlternatives.replacing(group: 1, in: p, with: "{attic|cellar}"), "a {red|blue} dress in a {attic|cellar}")
        XCTAssertEqual(PromptAlternatives.collapsing(group: 0, in: p), "a red dress in a {library|greenhouse}")
        XCTAssertEqual(PromptAlternatives.collapsing(group: 2, in: p), p)
        let braces = "a {\\{x\\}|y}"
        XCTAssertEqual(try PromptAlternatives(PromptAlternatives.collapsing(group: 0, in: braces)).prompts, ["a {x}"])
    }

    /// Three around the value, a step apart; held within the knob's range, never twice.
    func testABracketOffersValuesAroundTheKnob() {
        XCTAssertEqual(Bracketing.around(0.8, step: 0.2, count: 3, in: 0...1.5), [0.6, 0.8, 1.0])
        XCTAssertEqual(Bracketing.around(0.6, step: 0.15, count: 3, in: 0...1), [0.45, 0.6, 0.75])
        XCTAssertEqual(Bracketing.around(0.9, step: 0.15, count: 3, in: 0...1), [0.75, 0.9, 1.0])
        XCTAssertEqual(Bracketing.around(1.5, step: 0.2, count: 3, in: 0...1.5), [1.3, 1.5])
        XCTAssertEqual(Bracketing.around(0.8, step: 0.2, count: 4, in: 0...1.5), [0.5, 0.7, 0.9, 1.1])
        XCTAssertEqual(Bracketing.around(0.8, step: 0.2, count: 0, in: 0...1.5), [0.8])
    }

    /// **A grid fills the canvas**: the side at which every cell holds, labels and gaps included, clamped to the slider's
    /// range — and the figures the canvas lays out with (`margin`, `gap`, the labels).
    func testAGridFillsTheRoom() {
        // 2 × 2 squares, no labels: (1000 − 48 − 8) / 2 = 472 wide, (800 − 48 − 8 − 8) / 2 = 368 high.
        XCTAssertEqual(SeriesFit.side(in: CGSize(width: 1000, height: 800), columns: 2, rows: 2, aspect: 1,
                                      rowLabels: false, columnLabels: false), 368)
        // Labels on both axes: 128 beside (a label and a gap), 52 above: (800 − 48 − 52 − 8) / 2 = 346.
        XCTAssertEqual(SeriesFit.side(in: CGSize(width: 1000, height: 800), columns: 2, rows: 2, aspect: 1,
                                      rowLabels: true, columnLabels: true), 346)
        // A landscape cell's long side is its width: the height holds at 368, the width at 368 × 2 = 736 ≥ 472.
        XCTAssertEqual(SeriesFit.side(in: CGSize(width: 1000, height: 800), columns: 2, rows: 2, aspect: 2,
                                      rowLabels: false, columnLabels: false), 472)
        XCTAssertEqual(SeriesFit.side(in: CGSize(width: 100, height: 100), columns: 4, rows: 4, aspect: 1,
                                      rowLabels: false, columnLabels: false), SeriesFit.sides.lowerBound)
        XCTAssertEqual(SeriesFit.side(in: CGSize(width: 9000, height: 9000), columns: 1, rows: 1, aspect: 1,
                                      rowLabels: false, columnLabels: false), SeriesFit.sides.upperBound)
    }

    /// **The cells follow the room until the hand sizes them**: a new series fits; the room changing refits while the side
    /// is the fitted one; a side the hand set stays, and ⌘0 gives the room's back; the band under the grid coming or going
    /// moves nothing.
    func testTheCellsFollowTheRoomUntilTheHandSizesThem() {
        var (fit, side) = SeriesFit.refit(nil, run: 1, band: false, current: 220, fitting: 300)
        XCTAssertEqual(side, 300)
        XCTAssertTrue(fit.follows(300))
        (fit, side) = SeriesFit.refit(fit, run: 1, band: false, current: 300, fitting: 340)
        XCTAssertEqual(side, 340)
        // The band comes: the record follows it, the cells stay under the pointer, still following.
        (fit, side) = SeriesFit.refit(fit, run: 1, band: true, current: 340, fitting: 320)
        XCTAssertNil(side)
        XCTAssertTrue(fit.follows(340))
        // The hand sizes them: the room changing leaves them be, and ⌘0 gives back the room's side of now.
        (fit, side) = SeriesFit.refit(fit, run: 1, band: true, current: 500, fitting: 360)
        XCTAssertNil(side)
        XCTAssertFalse(fit.follows(500))
        XCTAssertEqual(fit.restore(), 360)
        XCTAssertTrue(fit.follows(360))
        // Another series fits, whatever the hand did to the last one.
        (fit, side) = SeriesFit.refit(fit, run: 2, band: false, current: 900, fitting: 250)
        XCTAssertEqual(side, 250)
        XCTAssertEqual(fit.run, 2)
    }
}
