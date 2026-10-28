import Foundation
import XCTest
@testable import Siliconed

/// **The route table** (`RouteTable.swift`) against what the switches it replaces made the engine do: for a
/// map's decisions, each product and each attention of the families takes the same route, the same share, the
/// same form of weight. No GPU, no weights: synthetic
/// decisions covering every switch and grid the table replaces.
final class RouteTableTests: XCTestCase {
    typealias Table = RouteTable

    // MARK: the families' classes

    /// The products of the five families' DiTs (k, n), `WeightProducts.linear` — Z-Image, Qwen-Image-2.1, Anima
    /// (self-attention, cross-attention k/v, MLP), Krea 2 (GQA k/v), MOSS-Image (fused), the text encoders',
    /// read in their `Config`s.
    private static let shapes: [(k: Int, n: Int)] = [
        (3840, 3840), (3840, 10240), (10240, 3840), (64, 3840), (2560, 3840), (3840, 64),
        (4096, 4096), (4096, 12288), (12288, 4096), (64, 4096),
        (2048, 2048), (1024, 2048), (2048, 8192), (8192, 2048),
        (6144, 6144), (6144, 1536), (6144, 16384), (16384, 6144),
        (3072, 9216), (3072, 3072), (3072, 16384), (8192, 3072), (4096, 3072),
        (2560, 4096), (2560, 1024), (2560, 9728), (9728, 2560), (4096, 1024), (4096, 8192),
    ]
    /// Their heights: 512², 1024², 1024×1536 with each family's text rows, the half-size start, the modulation
    /// and the text blocks, and the map's own two heights.
    private static let heights = [1, 11, 19, 21, 32, 192, 512, 1024, 1040, 1045, 1056, 1536, 1568, 4096, 4112, 4117, 4128, 6144, 6165, 6176]
    /// The attentions: (S, heads, headDim) — Z-Image 30×128, Qwen 32×128, Anima 16×128, Krea 2 48×128, MOSS 24×128.
    private static let attentions: [(heads: Int, headDim: Int)] = [(30, 128), (32, 128), (16, 128), (48, 128), (24, 128)]
    private static let sequences = [32, 512, 1023, 1024, 1045, 1056, 2048, 4096, 4128, 6144, 6176]

    // MARK: today's routes, from the switches

    /// What `GEMM.linear` and `Conductor` did with the flat settings: the engine's own instances, built the
    /// way `GEMM.init` built them from the settings before the table.
    private func today(_ s: EngineSettings, k: Int, n: Int, m: Int) -> (aneRows: Int, amxBesideANE: Bool, amx: Bool, amxShare: Double) {
        let engine = NeuralEngine(gpuShare: s.aneFraction, shares: s.aneShares)
        engine.besideAMX = s.aneAMX
        let rows = s.gemmANE ? engine.rows(m, k: k, n: n) : 0
        return (rows, rows > 0 && s.amx && engine.takesAMX(m, k: k, n: n), s.amx,
                s.amxShares.share(k: k, n: n, m: m) ?? s.amxFraction)
    }

    /// What `WeightProducts.prepare` decided from the flat settings (the reserve and the dtype aside).
    private func today(_ s: EngineSettings, kind: Table.WeightKind, m: Int) -> Table.WeightRoute {
        let rounds = s.gemmBF16Activations && m >= s.bf16ActivationMinimumRows
        switch kind {
        case .bf16:
            if s.gemmTensor || rounds { return .init(path: .bf16, roundsActivation: rounds) }
        case .int8, .q8:
            if kind == .int8 ? s.gemmInt8 : s.gemmQ8 { return .init(path: .native, roundsActivation: rounds && s.bf16ActivationEightBit) }
        }
        return .init(path: .widened, roundsActivation: false)
    }

    private func check(_ decisions: Diagnostic.Decisions, _ name: String, file: StaticString = #filePath, line: UInt = #line) {
        let settings = EngineSettings(applying: decisions.settings)
        let table = Table(decisions: decisions)
        XCTAssertEqual(table, settings.routes, name, file: file, line: line)
        var compared = 0
        for (k, n) in Self.shapes {
            for m in Self.heights {
                let (want, got) = (today(settings, k: k, n: n, m: m), table.products.route(k: k, n: n, m: m))
                XCTAssertEqual(got.aneRows, want.aneRows, "\(name) ANE rows \(m)×\(k)×\(n)", file: file, line: line)
                XCTAssertEqual(got.amxBesideANE, want.amxBesideANE, "\(name) AMX beside the ANE \(m)×\(k)×\(n)", file: file, line: line)
                XCTAssertEqual(got.amx, want.amx, "\(name) AMX \(m)×\(k)×\(n)", file: file, line: line)
                XCTAssertEqual(table.products.amx.share(k: k, n: n, m: m), want.amxShare, "\(name) AMX share \(m)×\(k)×\(n)", file: file, line: line)
                XCTAssertEqual(table.products.ane.share(k: k, n: n, m: m),
                               settings.aneShares.share(k: k, n: n, m: m, range: NeuralEngine.shareRange) ?? settings.aneFraction,
                               "\(name) ANE share \(m)×\(k)×\(n)", file: file, line: line)
                for kind in [Table.WeightKind.bf16, .int8, .q8] {
                    XCTAssertEqual(table.products.weight(kind, k: k, n: n, m: m), today(settings, kind: kind, m: m),
                                   "\(name) weight \(kind) \(m)×\(k)×\(n)", file: file, line: line)
                }
                compared += 1
            }
        }
        for (heads, headDim) in Self.attentions {
            for sequence in Self.sequences {
                let route = table.attention.route(sequence: sequence, heads: heads, headDim: headDim)
                XCTAssertEqual(route.flash, settings.flash, "\(name) flash \(sequence)", file: file, line: line)
                let splits = settings.headSplit && sequence >= HeadSplitAttention.minimumSequence
                XCTAssertEqual(route.splits, splits, "\(name) split \(sequence)", file: file, line: line)
                if splits { XCTAssertEqual(route.splitFraction, settings.headSplitFraction, "\(name) fraction \(sequence)", file: file, line: line) }
                // A caller that does not know its class gets the same answer.
                XCTAssertEqual(table.attention.route(sequence: sequence), route, file: file, line: line)
            }
        }
        XCTAssertGreaterThan(compared, 500, file: file, line: line)
        // The engines the process brings up.
        XCTAssertEqual(table.products.usesAMX, settings.amx, name, file: file, line: line)
        XCTAssertEqual(table.products.usesANE, settings.gemmANE, name, file: file, line: line)
        XCTAssertEqual(table.products.usesBF16Weights, settings.gemmTensor, name, file: file, line: line)
        XCTAssertEqual(table.products.usesInt8Weights, settings.gemmInt8, name, file: file, line: line)
        XCTAssertEqual(table.products.usesQ8Weights, settings.gemmQ8, name, file: file, line: line)
        XCTAssertEqual(table.products.roundsActivations, settings.gemmBF16Activations, name, file: file, line: line)
        XCTAssertEqual(table.attention.usesFlash, settings.flash, name, file: file, line: line)
        XCTAssertEqual(table.attention.usesHeadSplit, settings.headSplit, name, file: file, line: line)
        // The report's text: the table survives it.
        let data = try? JSONEncoder().encode(table)
        XCTAssertNotNil(data, file: file, line: line)
        XCTAssertEqual(data.flatMap { try? JSONDecoder().decode(Table.self, from: $0) }, table, "\(name) round trip", file: file, line: line)
    }

    // MARK: synthetic decisions

    private func decisions(amx: Bool = false, ane: Bool = false, tensor: Bool = false, bf16Act: Bool = false,
                           int8: Bool = false, q8: Bool = false, flash: Bool = false, split: Bool = true,
                           amxShares: String? = nil, aneShares: String? = nil, aneAMX: String? = nil,
                           fraction: Double? = nil) -> Diagnostic.Decisions {
        func d(_ on: Bool) -> Diagnostic.Decision { .init(on: on, passed: true, gain: on ? 1.3 : 1, gains: [1, 1]) }
        return .init(margin: 0.05, amx: d(amx), amxFraction: amx ? 0.63 : nil, flash: d(flash), headSplit: d(split), tensor: d(tensor),
                     amxShares: amx ? amxShares : nil, int8: d(int8), q8: d(q8), bf16Act: d(bf16Act), ane: d(ane),
                     aneFraction: ane ? 0.55 : nil, aneShares: ane ? aneShares : nil, headSplitFraction: fraction,
                     aneAMX: aneAMX)
    }

    private static let amxGrid = "3840×3840@1024=0.610,10240×3840@1024=0.760,3840×10240@1024=0.600,3840×3840@4096=0.640,"
        + "10240×3840@4096=0.790,3840×10240@4096=0.630,6144×6144@1024=0.700,6144×6144@4096=0.720,2048×2048@1024=0.520,2048×2048@4096=0.560"
    private static let aneGrid = "3840×3840@1024=0.550,3840×10240@1024=0.500,10240×3840@1024=0,3840×3840@4096=0.580,"
        + "3840×10240@4096=0.560,10240×3840@4096=0,6144×6144@4096=0.600,6144×16384@4096=0.520,16384×6144@4096=0,"
        + "3072×16384@4096=0.450,8192×3072@4096=0.600"
    private static let three = "3840×3840@1024=0,3840×10240@4096=1,10240×3840@4096=0,3072×16384@4096=1,8192×3072@4096=0,6144×6144@4096=0"

    func testEveryKindOfMapGivesTodaysRoutes() {
        check(decisions(), "defaults")
        check(decisions(amx: true), "AMX, scalar share")
        check(decisions(amx: true, amxShares: Self.amxGrid), "AMX grid")
        check(decisions(amx: true, amxShares: Self.amxGrid, fraction: 0.35), "AMX grid, other head share")
        check(decisions(ane: true), "ANE, scalar share")
        check(decisions(ane: true, aneShares: Self.aneGrid), "ANE grid, AMX off")
        check(decisions(amx: true, ane: true, amxShares: Self.amxGrid, aneShares: Self.aneGrid), "ANE and AMX")
        check(decisions(amx: true, ane: true, amxShares: Self.amxGrid, aneShares: Self.aneGrid, aneAMX: Self.three), "three engines")
        check(decisions(tensor: true), "bf16 weights")
        check(decisions(amx: true, tensor: true, amxShares: Self.amxGrid), "bf16 weights with the AMX map")
        check(decisions(bf16Act: true), "rounded activation")
        check(decisions(tensor: true, bf16Act: true), "bf16 weights and rounded activation")
        check(decisions(amx: true, int8: true, q8: true, amxShares: Self.amxGrid), "int8 and Q8_0 beside the AMX")
        check(decisions(ane: true, int8: true, aneShares: Self.aneGrid), "int8, ANE")
        check(decisions(flash: true), "flash and split")
        check(decisions(flash: true, split: false), "flash alone")
        check(decisions(split: false), "no split")
    }

    func testTheThreeEnginesTakeTheShapesTheGridNames() {
        let t = Table(decisions: decisions(amx: true, ane: true, amxShares: Self.amxGrid, aneShares: Self.aneGrid, aneAMX: Self.three)).products
        XCTAssertEqual(t.route(k: 3840, n: 10240, m: 4096).engines, .aneAMX)
        XCTAssertEqual(t.route(k: 3840, n: 3840, m: 4096).engines, .ane)
        XCTAssertEqual(t.route(k: 10240, n: 3840, m: 4096).engines, .amx, "the AMX's shape: the ANE's share is 0 there")
        XCTAssertEqual(t.route(k: 3840, n: 3840, m: 32).engines, .amx)
    }

    func testForcedSettingsAreATable() {
        // A check that rounds the 8-bit weights too, a profile that lowers the ANE's floor: settings, hence table.
        let s = EngineSettings(applying: ["gemm_bf16_act": true, "gemm_int8": true, "gemm_q8": true, "bf16_act_8bit": true,
                                          "bf16_act_min": 64, "gemm_ane": true, "ane_min": 256, "ane_fraction": 0.4])
        let t = s.routes.products
        XCTAssertEqual(t.weight(.int8, k: 3072, n: 3072, m: 4096), .init(path: .native, roundsActivation: true))
        XCTAssertEqual(t.weight(.int8, k: 3072, n: 3072, m: 32), .init(path: .native, roundsActivation: false))
        XCTAssertEqual(t.weight(.q8, k: 3072, n: 3072, m: 64), .init(path: .native, roundsActivation: true))
        XCTAssertEqual(t.weight(.bf16, k: 3072, n: 3072, m: 63), .init(path: .widened, roundsActivation: false))
        XCTAssertEqual(t.weight(.bf16, k: 3072, n: 3072, m: 64), .init(path: .bf16, roundsActivation: true))
        XCTAssertEqual(t.ane.minimumRows, 256)
        XCTAssertGreaterThan(t.route(k: 3840, n: 3840, m: 512).aneRows, 0)
        XCTAssertEqual(t.route(k: 3840, n: 3840, m: 128).aneRows, 0)
    }

    // MARK: what the table adds

    func testAClassRuleIsAheadOfTheGrids() {
        var t = Table(decisions: decisions(amx: true, ane: true, amxShares: Self.amxGrid, aneShares: Self.aneGrid))
        let before = t.products.route(k: 3840, n: 3840, m: 4096)
        XCTAssertGreaterThan(before.aneRows, 0)
        XCTAssertEqual(before.engines, .ane)
        t.products.classes = [.init(engines: .gpu, k: 3840, n: 3840, minRows: 4096, maxRows: 4096),
                              .init(engines: .amx, k: 3840, n: 10240)]
        XCTAssertEqual(t.products.route(k: 3840, n: 3840, m: 4096).engines, .gpu)
        XCTAssertEqual(t.products.route(k: 3840, n: 3840, m: 4128).engines, .ane, "another height keeps the grid")
        XCTAssertEqual(t.products.route(k: 3840, n: 10240, m: 4096).engines, .amx)
        XCTAssertEqual(t.products.forced(k: 3840, n: 10240, m: 1), .amx)
        XCTAssertNil(t.products.forced(k: 4096, n: 4096, m: 4096))
        // A rule cannot bring an engine the table did not bring up.
        t.products.ane.enabled = false
        t.products.classes = [.init(engines: .ane)]
        XCTAssertEqual(t.products.route(k: 3840, n: 3840, m: 4096).aneRows, 0)
    }

    func testTheAttentionKeepsFlashAndTheSplitApart() {
        let t = Table(decisions: decisions(flash: true))
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
        XCTAssertNil(table.products.forced(k: 3840, n: 3840, m: 1024), "an open class keeps the grids")
        XCTAssertTrue(table.attention.route(sequence: 4096, heads: 30, headDim: 128).splits)
        XCTAssertFalse(table.attention.route(sequence: 4096, heads: 32, headDim: 128).splits)
        let data = try? JSONEncoder().encode(table)
        XCTAssertEqual(data.flatMap { try? JSONDecoder().decode(Table.self, from: $0) }, table)
    }
}
