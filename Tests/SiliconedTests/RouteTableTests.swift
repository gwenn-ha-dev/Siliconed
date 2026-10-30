import Foundation
import XCTest
@testable import Siliconed

/// **The previous map's route table** (`RouteTable.swift`, legacy: the engine reads the plan, not this) — what it still does for
/// the report and the render check until the previous map goes (E4b): a class rule ahead of the grids, the attention's flash and
/// split apart, a length nobody tabulated, a grid extrapolated, a plan become class rules. No GPU, no weights.
final class RouteTableTests: XCTestCase {
    typealias Table = RouteTable

    // MARK: synthetic grids

    private static let amxGrid = "3840×3840@1024=0.610,10240×3840@1024=0.760,3840×10240@1024=0.600,3840×3840@4096=0.640,"
        + "10240×3840@4096=0.790,3840×10240@4096=0.630,6144×6144@1024=0.700,6144×6144@4096=0.720,2048×2048@1024=0.520,2048×2048@4096=0.560"
    private static let aneGrid = "3840×3840@1024=0.550,3840×10240@1024=0.500,10240×3840@1024=0,3840×3840@4096=0.580,"
        + "3840×10240@4096=0.560,10240×3840@4096=0,6144×6144@4096=0.600,6144×16384@4096=0.520,16384×6144@4096=0,"
        + "3072×16384@4096=0.450,8192×3072@4096=0.600"
    private static let three = "3840×3840@1024=0,3840×10240@4096=1,10240×3840@4096=0,3072×16384@4096=1,8192×3072@4096=0,6144×6144@4096=0"

    private func table(amx: Bool = false, ane: Bool = false, amxShares: String? = nil, aneShares: String? = nil,
                       aneAMX: String? = nil, flash: Bool = false, split: Bool = true) -> Table {
        func grid(_ text: String?) -> Table.Grid { text.flatMap(Conductor.Shares.init(text:)).map(Table.Grid.init) ?? Table.Grid() }
        return Table(products: .init(amx: .init(enabled: amx, grid: grid(amxShares)),
                                     ane: .init(enabled: ane, gpuShare: 0.55, grid: grid(aneShares), besideAMX: grid(aneAMX))),
                     attention: .init(rules: [.init(minSequence: HeadSplitAttention.minimumSequence, flash: flash,
                                                    splitFraction: split ? 0.4 : nil), .init(flash: flash)]))
    }

    func testTheThreeEnginesTakeTheShapesTheGridNames() {
        let t = table(amx: true, ane: true, amxShares: Self.amxGrid, aneShares: Self.aneGrid, aneAMX: Self.three).products
        XCTAssertEqual(t.route(k: 3840, n: 10240, m: 4096).engines, .aneAMX)
        XCTAssertEqual(t.route(k: 3840, n: 3840, m: 4096).engines, .ane)
        XCTAssertEqual(t.route(k: 10240, n: 3840, m: 4096).engines, .amx, "the AMX's shape: the ANE's share is 0 there")
        XCTAssertEqual(t.route(k: 3840, n: 3840, m: 32).engines, .amx)
    }

    // MARK: what the table adds

    func testAClassRuleIsAheadOfTheGrids() {
        var t = table(amx: true, ane: true, amxShares: Self.amxGrid, aneShares: Self.aneGrid)
        let before = t.products.route(k: 3840, n: 3840, m: 4096)
        XCTAssertGreaterThan(before.aneRows, 0)
        XCTAssertEqual(before.engines, .ane)
        t.products.classes = [.init(engines: .gpu, k: 3840, n: 3840, minRows: 4096, maxRows: 4096),
                              .init(engines: .amx, k: 3840, n: 10240)]
        XCTAssertEqual(t.products.route(k: 3840, n: 3840, m: 4096).engines, .gpu)
        XCTAssertEqual(t.products.route(k: 3840, n: 3840, m: 4128).engines, .gpu, "another height takes the nearest measured one's")
        XCTAssertEqual(t.products.route(k: 3840, n: 10240, m: 4096).engines, .amx)
        XCTAssertEqual(t.products.forced(k: 3840, n: 10240, m: 1), .amx)
        XCTAssertEqual(t.products.forced(k: 4096, n: 4096, m: 4096), .gpu, "another shape takes the nearest shape's")
        XCTAssertEqual(t.products.forced(k: 4096, n: 12288, m: 4096), .amx, "4096×12288 is nearer 3840×10240")
        // A neighbour is ahead of a rule that names no shape; beyond reach, that rule.
        t.products.classes.append(.init(engines: .ane))
        XCTAssertEqual(t.products.forced(k: 4096, n: 4096, m: 6144), .gpu)
        XCTAssertEqual(t.products.forced(k: 64, n: 3840, m: 4096), .ane, "64 wide is no 3840×3840")
        t.products.classes.removeLast()
        XCTAssertNil(t.products.forced(k: 64, n: 3840, m: 4096), "beyond reach and no rule: the grids")
        // Without a class rule, the grids decide (the switches' tables).
        var g = t; g.products.classes = []
        XCTAssertNil(g.products.forced(k: 4096, n: 4096, m: 4096))
        // A rule cannot bring an engine the table did not bring up.
        t.products.ane.enabled = false
        t.products.classes = [.init(engines: .ane)]
        XCTAssertEqual(t.products.route(k: 3840, n: 3840, m: 4096).aneRows, 0)
    }

    func testTheAttentionKeepsFlashAndTheSplitApart() {
        let t = table(flash: true)
        let long = t.attention.route(sequence: 4096, heads: 32, headDim: 128)
        XCTAssertTrue(long.flash); XCTAssertTrue(long.splits)
        let short = t.attention.route(sequence: 512)
        XCTAssertTrue(short.flash); XCTAssertFalse(short.splits)
        // A rule for one class only.
        var u = t
        u.attention.rules.insert(.init(minSequence: 4096, maxSequence: 4096, heads: 48, headDim: 128, flash: false, splitFraction: 0.25), at: 0)
        XCTAssertEqual(u.attention.route(sequence: 4096, heads: 48, headDim: 128), .init(flash: false, splitFraction: 0.25))
        XCTAssertEqual(u.attention.route(sequence: 4096, heads: 32, headDim: 128), long)
        XCTAssertEqual(u.attention.route(sequence: 4096), long, "an omitted class never matches a rule that names one")
    }

    func testAnUntabulatedLengthTakesItsClassAtTheNearestLength() {
        // A map's table: one rule per measured length, no rule for any other.
        let a = Table.Attention(rules: [
            .init(minSequence: 4128, maxSequence: 4128, heads: 30, headDim: 128, splitFraction: 0.4),
            .init(minSequence: 1056, maxSequence: 1056, heads: 30, headDim: 128),
        ])
        XCTAssertTrue(a.route(sequence: 4128, heads: 30, headDim: 128).splits)
        XCTAssertTrue(a.route(sequence: 6176, heads: 30, headDim: 128).splits, "1024×1536 takes 4128's split")
        XCTAssertFalse(a.route(sequence: 512, heads: 30, headDim: 128).splits, "below every measured length: the shortest")
        XCTAssertFalse(a.route(sequence: 2000, heads: 30, headDim: 128).splits, "2000/1056 < 4128/2000: nearer 1056")
        XCTAssertTrue(a.route(sequence: 2200, heads: 30, headDim: 128).splits)
        XCTAssertTrue(a.route(sequence: 4128, heads: 24, headDim: 128).splits, "another class takes the nearest class's")
        XCTAssertFalse(a.route(sequence: 4128, heads: 8, headDim: 128).splits, "beyond reach")
        XCTAssertFalse(a.route(sequence: 4128).splits)
    }

    func testAGridIsExtrapolatedPastItsHeightsOnlyWhenAsked() {
        let grid = Conductor.Shares([.init(k: 3840, n: 3840, m: 1024, ratio: 0.5), .init(k: 3840, n: 3840, m: 4096, ratio: 0.8)])
        XCTAssertEqual(grid.ratio(k: 3840, n: 3840, m: 6144)!, 0.8, accuracy: 1e-12, "held at the last height, as before")
        XCTAssertEqual(grid.ratio(k: 3840, n: 3840, m: 6144, extrapolatingHeights: true)!,
                       0.5 + log(6144.0 / 1024) / log(4096.0 / 1024) * 0.3, accuracy: 1e-12)
        XCTAssertEqual(grid.ratio(k: 3840, n: 3840, m: 2048, extrapolatingHeights: true)!, grid.ratio(k: 3840, n: 3840, m: 2048)!, accuracy: 1e-12)
        XCTAssertEqual(grid.ratio(k: 3840, n: 3840, m: 512, extrapolatingHeights: true)!, 0.5 - 0.3 / 2, accuracy: 1e-12)
    }

    // MARK: a plan, as a table

    func testAPlanBecomesClassRules() {
        let keys: [Table.ClassKey] = [.product(k: 3840, n: 3840, m: 4096), .product(k: 10240, n: 3840, m: 4096),
                                      .product(k: 3840, n: 3840, m: 1024), .attention(sequence: 4096, heads: 30, headDim: 128)]
        for key in keys { XCTAssertEqual(Table.ClassKey(id: key.id), key) }
        XCTAssertNil(Table.ClassKey(id: "weights:bf16"))
        typealias CM = CapacityMeasure
        let decide = { (id: String, chosen: String?) in CM.Decision(classID: id, chosen: chosen, status: .closed, loss: 0, hotRatios: [], runs: 0) }
        let plan = CM.Plan(decisions: [decide(keys[0].id, "gpu+ane"), decide(keys[1].id, "gpu"), decide(keys[2].id, nil),
                                       decide(keys[3].id, "split"), decide("unknown", "gpu")],
                           residualLoss: 0, spent: 1, exhausted: false)
        let table = Table(plan: plan, classes: [])
        XCTAssertTrue(table.products.usesANE)
        XCTAssertFalse(table.products.usesAMX)
        XCTAssertEqual(table.products.forced(k: 3840, n: 3840, m: 4096), .ane)
        XCTAssertEqual(table.products.forced(k: 10240, n: 3840, m: 4096), .gpu)
        XCTAssertEqual(table.products.forced(k: 3840, n: 3840, m: 1024), .ane, "an open class takes its nearest decided height's")
        XCTAssertTrue(table.attention.route(sequence: 4096, heads: 30, headDim: 128).splits)
        XCTAssertTrue(table.attention.route(sequence: 4096, heads: 32, headDim: 128).splits, "the nearest class's")
        let data = try? JSONEncoder().encode(table)
        XCTAssertEqual(data.flatMap { try? JSONDecoder().decode(Table.self, from: $0) }, table)
    }
}
