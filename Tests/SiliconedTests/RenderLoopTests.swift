import Foundation
import XCTest
@testable import Siliconed

/// **The render checks the map** (`ProductLedger`, `RenderLoop`, `RenderLoopVerdict`): the ledger's lines merged by class as the
/// map merges heights, the gap to the prediction (nil when the render is the prediction), the rule that keeps the map, corrects
/// the regime or asks for a measure. No GPU, no weights: a synthetic map over the catalogue's own classes, and renders simulated
/// by scaling the prediction.
final class RenderLoopTests: XCTestCase {
    typealias CC = CapacityClasses

    // MARK: a synthetic map

    private static let machine = Diagnostic.Machine(chip: "Apple M1 Pro", hardwareModel: "MacBookPro18,3", performanceCores: 8,
                                                    efficiencyCores: 2, gpuCores: 16, memoryBytes: 17_179_869_184, macOS: "27.0.1 (26A434)",
                                                    version: "0.17 (170)", build: "x+y", gpu: "Apple M1 Pro", gpuFamily: "apple7",
                                                    gpuWorkingSetBytes: 11_453_251_584)

    private var request: CC.Request { .init(family: .zImage, variant: .standard, width: 1024, height: 1024, includesEncoder: false) }

    /// The table: the AMX beside the GPU from `amx_min` rows, the flash kernel on attentions.
    private func table(weights: [RouteTable.WeightRule] = []) -> RouteTable {
        RouteTable(products: .init(weights: weights, amx: .init(enabled: true)),
                   attention: .init(rules: [.init(flash: true)]))
    }

    /// A map that has timed every class of `request` at its own height: the AMX at `amxRatio` of the GPU's time, flash at
    /// `flashRatio` of the graph's, a witness at `gpu` TFLOP/s and the engines' heat `heat`.
    private func report(amxRatio: Double = 0.8, flashRatio: Double = 0.7, gpu: Double = 3.4, heat: Double = 1,
                        classes: [CC.Class]? = nil, date: String = "2026-10-29T10:00:00Z") -> Diagnostic.Report {
        let cs = classes ?? CC.classes(for: request)!
        var seen = Set<String>(), entries: [Diagnostic.MapClass] = []
        for c in cs {
            let id = c.routeKey.id + RouteTable.ClassKey.suffix(c.weightKind ?? .bf16)
            guard seen.insert(id).inserted else { continue }
            let attention: Bool = { if case .attention = c.shape { return true } else { return false } }()
            let timings: [Diagnostic.MapClass.Timing] = attention
                ? [.init(route: "flash", ratio: flashRatio, error: 0.01)]
                : (c.routeKey.id.hasSuffix("@32") ? [] : [.init(route: "gpu+amx", ratio: amxRatio, error: 0.01)])
            entries.append(.init(id: id, weight: 0.01, chosen: attention ? "flash" : (timings.isEmpty ? "gpu" : "gpu+amx"),
                                 loss: 0.001, runs: 20, status: .closed, timings: timings))
        }
        let map = Diagnostic.MapResult(classes: entries, residualLoss: 0.005, epsilon: 0.02, spent: 50, budget: 52, exhausted: false,
                                       tail: 0.01, judged: 2, wrong: [])
        let flops = 2.0 * 2048 * 3840 * 3840
        func witness(_ engine: String, _ rate: Double) -> CapacityMap.EngineWitness {
            .init(engine: engine, before: flops / (rate * 1e12), after: flops / (rate * 1e12) * heat, errorBefore: 0.0005, errorAfter: 0.0005)
        }
        let conditions = Diagnostic.Conditions(thermal: "nominal", lowPowerMode: false, onBattery: false, swapUsedBytes: 0)
        return Diagnostic.Report(date: date, seconds: 58, machine: Self.machine, cpuFeatures: [:], before: conditions, after: conditions,
                                 products: [], lowPrecision: [], attention: [], widening: [],
                                 witness: .init(gpuBefore: gpu, gpuAfter: gpu, amxBefore: 1.9, amxAfter: 1.9),
                                 routes: table(), map: map, regime: .init(witnesses: [witness("gpu", gpu), witness("amx", 1.9)]),
                                 verdict: .init(green: true, reasons: []))
    }

    /// The render that ran the prediction, each line `scale(route)` times slower.
    private func simulated(_ predicted: [RenderLoop.Predicted], scale: (String) -> Double = { _ in 1 }) -> [ProductLedger.Tally] {
        predicted.map { .init(work: $0.work, route: $0.route, executions: $0.executions, seconds: $0.seconds * scale($0.route)) }
    }

    private func predictions(_ r: Diagnostic.Report, amxMinimumRows: Int = 512) -> [RenderLoop.Predicted] {
        RenderLoop.predictions(for: request, table: table(), report: r, amxMinimumRows: amxMinimumRows)!
    }

    // MARK: the ledger

    func testTheLedgerIsArmedForTheDenoisingAlone() {
        XCTAssertNil(ProductLedger.current)
        let ledger = ProductLedger()
        ProductLedger.recording(into: ledger) {
            XCTAssertTrue(ProductLedger.current === ledger)
            ProductLedger.attention(sequence: 4128, heads: 30, headDim: 128, route: "flash")?.finish()
            XCTAssertNil(ProductLedger.attention(sequence: 32, heads: 30, headDim: 128, route: "graph"), "under 64 rows: no class")
        }
        XCTAssertNil(ProductLedger.current)
        XCTAssertNil(ProductLedger.attention(sequence: 4128, heads: 30, headDim: 128, route: "flash"), "nothing records outside a render")
        XCTAssertEqual(ledger.tallies.count, 1)
        XCTAssertEqual(ledger.tallies[0].executions, 1)
        // A throwing body disarms it too.
        struct Boom: Error {}
        XCTAssertThrowsError(try ProductLedger.recording(into: ProductLedger()) { throw Boom() })
        XCTAssertNil(ProductLedger.current)
    }

    func testTheLedgerCountsAndSumsPerWorkAndRoute() {
        let ledger = ProductLedger()
        let w = ProductLedger.Work.product(k: 3840, n: 3840, m: 4128, kind: .bf16)
        ledger.add(w, route: "gpu+amx", seconds: 0.5); ledger.add(w, route: "gpu+amx", seconds: 0.25); ledger.add(w, route: "gpu", seconds: 1)
        let t = ledger.tallies
        XCTAssertEqual(t.count, 2)
        XCTAssertEqual(t.first { $0.route == "gpu+amx" }?.executions, 2)
        XCTAssertEqual(t.first { $0.route == "gpu+amx" }?.seconds ?? 0, 0.75, accuracy: 1e-12)
        XCTAssertEqual(t, ledger.tallies, "a fixed order")
    }

    /// The kernels record themselves: a graph attention on the GPU (kilobytes), armed and not.
    func testTheGraphAttentionRecordsItselfWhenArmedAndProductsNameTheirEngines() throws {
        let gemm = try GEMM(freezeCut: true, coexecution: false, tensorWeights: false, neuralEngine: false)
        let (sequence, heads, dim) = (64, 2, 64)
        let bytes = Arena.capacity(for: sequence * heads * dim * 4)
        var keep: [Pages] = []
        func buffer() throws -> MTLBuffer {
            let pages = Pages(bytes: bytes)
            let p = pages.floats()
            for i in 0..<(bytes / 4) { p[i] = Float(i % 7) * 0.01 }
            keep.append(pages)
            return try gemm.wrap(UnsafeMutableRawPointer(p), bytes: bytes, name: "t")
        }
        let (q, k, v, out) = (try buffer(), try buffer(), try buffer(), try buffer())
        let attention = Attention(device: gemm.device, queue: gemm.queue, heads: heads, sequence: sequence, headDim: dim)
        _ = attention.run(q: q, k: k, v: v, into: out)           // nothing armed: nothing recorded
        let ledger = ProductLedger()
        ProductLedger.recording(into: ledger) {
            _ = attention.run(q: q, k: k, v: v, into: out)
            _ = attention.run(q: q, k: k, v: v, into: out)
        }
        let t = ledger.tallies
        XCTAssertEqual(t.count, 1)
        XCTAssertEqual(t[0].work, .attention(sequence: sequence, heads: heads, headDim: dim))
        XCTAssertEqual(t[0].route, "graph")
        XCTAssertEqual(t[0].executions, 2)
        XCTAssertGreaterThan(t[0].seconds, 0)
        // A widened product: the GPU alone here (no conductor, no ANE), under any height.
        let (m, kk, n) = (128, 64, 64)
        let a = Pages(bytes: Arena.capacity(for: m * kk * 4)), w = Pages(bytes: Arena.capacity(for: kk * n * 4)), c = Pages(bytes: Arena.capacity(for: m * n * 4))
        for i in 0..<(m * kk) { a.floats()[i] = 0.01 }
        for i in 0..<(kk * n) { w.floats()[i] = 0.02 }
        _ = gemm.linear(a: try gemm.wrap(UnsafeMutableRawPointer(a.floats()), bytes: Arena.capacity(for: m * kk * 4), name: "a"),
                        b: try gemm.wrap(UnsafeMutableRawPointer(w.floats()), bytes: Arena.capacity(for: kk * n * 4), name: "w"),
                        c: try gemm.wrap(UnsafeMutableRawPointer(c.floats()), bytes: Arena.capacity(for: m * n * 4), name: "c"),
                        m: m, k: kk, n: n, weightIsTransposed: true)
        XCTAssertEqual(gemm.lastEngines, .gpu)
        withExtendedLifetime((keep, a, w, c)) {}
    }

    // MARK: classes, as the map merges them

    func testHeightsWithin64RowsAreOneClassAndOthersAreNot() {
        let t: [ProductLedger.Tally] = [
            .init(work: .product(k: 3840, n: 3840, m: 4128, kind: .bf16), route: "gpu+amx", executions: 120, seconds: 12),
            .init(work: .product(k: 3840, n: 3840, m: 4096, kind: .bf16), route: "gpu+amx", executions: 8, seconds: 0.8),
            .init(work: .product(k: 3840, n: 3840, m: 4150, kind: .bf16), route: "gpu", executions: 2, seconds: 0.3),   // 22 from 4128: chained
            .init(work: .product(k: 3840, n: 3840, m: 4300, kind: .bf16), route: "gpu", executions: 1, seconds: 0.1),   // 150 from 4150: apart
            .init(work: .product(k: 3840, n: 3840, m: 4128, kind: .int8), route: "gpu/int8", executions: 3, seconds: 0.2),   // another kind
            .init(work: .product(k: 3840, n: 10240, m: 4128, kind: .bf16), route: "gpu", executions: 1, seconds: 0.1),     // another shape
            .init(work: .attention(sequence: 4128, heads: 30, headDim: 128), route: "flash", executions: 30, seconds: 4),
            .init(work: .attention(sequence: 4096, heads: 30, headDim: 128), route: "flash", executions: 2, seconds: 0.2),  // exact: apart
        ]
        let c = RenderLoop.compare(t)
        XCTAssertEqual(c.rows.count, 6)
        let merged = c.rows.first { $0.id == "product:3840x3840@4128" }!
        XCTAssertEqual(merged.executions, 130)
        XCTAssertEqual(merged.measured, 13.1, accuracy: 1e-9)
        XCTAssertEqual(merged.routes, ["gpu+amx", "gpu"], "the route with the most seconds first")
        XCTAssertNotNil(c.rows.first { $0.id == "product:3840x3840@4128~int8" })
        XCTAssertNotNil(c.rows.first { $0.id == "product:3840x3840@4300" })
        XCTAssertEqual(c.rows.filter(\.isAttention).count, 2)
        XCTAssertEqual(c.measured, t.reduce(0) { $0 + $1.seconds }, accuracy: 1e-9, "merging loses nothing")
        XCTAssertNil(c.predicted, "no map: the measure alone")
        XCTAssertNil(c.error)
        XCTAssertEqual(c.rows.map(\.measured), c.rows.map(\.measured).sorted(by: >), "heaviest first")
    }

    func testTheMergedClassHoldsTheHeightsOfBothTheRenderAndThePrediction() {
        // The prediction at 4 128 rows and a render that ran 4 117 (another prompt's text): one class, matched.
        let p = RenderLoop.Predicted(work: .product(k: 3840, n: 3840, m: 4128, kind: .bf16), route: "gpu+amx", executions: 10, seconds: 2, provenance: .measured)
        let t = ProductLedger.Tally(work: .product(k: 3840, n: 3840, m: 4117, kind: .bf16), route: "gpu+amx", executions: 10, seconds: 2.2)
        let c = RenderLoop.compare([t], predicted: [p])
        XCTAssertEqual(c.rows.count, 1)
        XCTAssertEqual(c.rows[0].error ?? 0, 0.1, accuracy: 1e-9)
        XCTAssertEqual(c.coverage, 1)
    }

    // MARK: the gap

    func testTheGapIsNilWhenThereIsNothingToCompareAndZeroWhenTheRenderIsThePrediction() {
        let r = report()
        let p = predictions(r)
        XCTAssertFalse(p.isEmpty)
        let c = RenderLoop.compare(simulated(p), predicted: p)
        for row in c.rows { XCTAssertEqual(row.error ?? 1, 0, accuracy: 1e-9, row.id) }
        XCTAssertEqual(c.error ?? 1, 0, accuracy: 1e-12)
        XCTAssertEqual(c.productError ?? 1, 0, accuracy: 1e-12)
        XCTAssertEqual(c.attentionError ?? 1, 0, accuracy: 1e-12)
        XCTAssertEqual(c.coverage, 1, accuracy: 1e-12)
        XCTAssertEqual(c.routeAgreement, 1, accuracy: 1e-12)
        XCTAssertEqual(c.measured, c.predicted ?? 0, accuracy: 1e-9 * c.measured)
        XCTAssertEqual(RenderLoop.verdict(c, report: r), .held)
        // A uniform ×1.2 is a +20 % gap on every row and on the total.
        let slow = RenderLoop.compare(simulated(p) { _ in 1.2 }, predicted: p)
        XCTAssertEqual(slow.error ?? 0, 0.2, accuracy: 1e-9)
        for row in slow.rows { XCTAssertEqual(row.error ?? 0, 0.2, accuracy: 1e-9) }
    }

    func testThePredictionIsWhatTheMapSaysByHand() {
        // One product on the AMX: executions · FLOP / (the witness's GPU rate ÷ the map's ratio) · heat 1.
        let r = report(amxRatio: 0.8, gpu: 3.4)
        let p = predictions(r)
        let big = p.first { $0.work == .product(k: 3840, n: 3840, m: 4128, kind: .bf16) }!
        XCTAssertEqual(big.route, "gpu+amx")
        let flop = 2.0 * 3840 * 3840 * 4128
        XCTAssertEqual(big.seconds, Double(big.executions) * flop / (3.4e12 / 0.8), accuracy: 1e-9 * big.seconds)
        XCTAssertEqual(big.provenance, .measured)
        // Under `amx_min` rows the AMX is not asked: the GPU alone's rate.
        let text = p.first { $0.work == .product(k: 3840, n: 3840, m: 32, kind: .bf16) }!
        XCTAssertEqual(text.route, "gpu")
        XCTAssertEqual(text.seconds, Double(text.executions) * 2.0 * 3840 * 3840 * 32 / 3.4e12, accuracy: 1e-9 * text.seconds)
        // The engine's threshold is read, not copied: with `amx_min` above the height, the AMX goes.
        let high = predictions(r, amxMinimumRows: 5000)
        XCTAssertEqual(high.first { $0.work == big.work }?.route, "gpu")
    }

    func testTheRegimeIsAppliedOnceToTheRoutesThatKeepTheEngineBusy() {
        // The map's ratios were decided hot: undone of the regime, the prediction multiplies it back — a ratio of 0.8 at a
        // heat of 1.5 on the AMX and 1 on the GPU is a cold ratio of 0.8 · 1 / 1.5, a hot time of 0.8 of the GPU's again.
        var r = report(amxRatio: 0.8)
        r.regime!.heat = ["gpu": 1, "amx": 1.5]
        let p = predictions(r)
        let big = p.first { $0.work == .product(k: 3840, n: 3840, m: 4128, kind: .bf16) }!
        XCTAssertEqual(big.seconds, Double(big.executions) * 2.0 * 3840 * 3840 * 4128 / (3.4e12 / 0.8), accuracy: 1e-9 * big.seconds)
        // A learned factor moves the prediction and not the cold rates.
        r.regime!.learned = ["amx": 1.1]
        let learned = predictions(r).first { $0.work == big.work }!
        XCTAssertEqual(learned.seconds, big.seconds * 1.1, accuracy: 1e-9 * big.seconds)
    }

    func testAHeightNobodyTimedIsReadOffTheLineThroughTheTwoNearest() {
        // The map timed 3840×3840 at 1 024 and 4 096 rows (AMX at 0.9 and 0.6 of the GPU's time).
        func entry(_ m: Int, _ ratio: Double) -> Diagnostic.MapClass {
            .init(id: "product:3840x3840@\(m)", weight: 0.1, chosen: "gpu+amx", loss: 0, runs: 10, status: .closed,
                  timings: [.init(route: "gpu+amx", ratio: ratio, error: 0.01)])
        }
        var r = report()
        r.map!.classes = [entry(1024, 0.9), entry(4096, 0.6)]
        let wanted = CC.classes(for: request)!.filter { $0.routeKey.id == "product:3840x3840@4128" }
        let basis = RenderLoop.Basis(classes: wanted, report: r)
        // t = ln(4128/1024) / ln(4096/1024); ln ratio is linear in ln m.
        let t = log(4128.0 / 1024) / log(4.0)
        let ratio = exp((1 - t) * log(0.9) + t * log(0.6))
        XCTAssertEqual(basis.rates.measured["product:3840x3840@4128"]?["gpu+amx"] ?? 0, 3.4e12 / ratio, accuracy: 1e3)
        XCTAssertEqual(basis.provenance("product:3840x3840@4128", route: "gpu+amx"), .interpolated)
        XCTAssertEqual(basis.provenance("product:3840x3840@4128", route: "gpu"), .measured)
        // A shape the map never saw: the GPU's rate stands in, and the row says so.
        let none = RenderLoop.Basis(classes: CC.classes(for: request)!.filter { $0.routeKey.id == "product:3840x10240@4128" }, report: r)
        XCTAssertEqual(none.provenance("product:3840x10240@4128", route: "gpu+amx"), .assumed)
    }

    // MARK: the rule

    func testTheMapHoldsWithinTheTolerance() {
        let r = report()
        let p = predictions(r)
        for scale in [0.93, 1.0, 1.08] {
            let c = RenderLoop.compare(simulated(p) { _ in scale }, predicted: p)
            XCTAssertEqual(RenderLoop.verdict(c, report: r), .held, "×\(scale)")
        }
    }

    func testARenderFarOffIsAMapToMeasureAgain() {
        let r = report()
        let p = predictions(r)
        let c = RenderLoop.compare(simulated(p) { _ in 1.6 }, predicted: p)
        XCTAssertEqual(RenderLoop.verdict(c, report: r), .remeasure(.gap))
        let fast = RenderLoop.compare(simulated(p) { _ in 0.5 }, predicted: p)
        XCTAssertEqual(RenderLoop.verdict(fast, report: r), .remeasure(.gap))
    }

    func testAGPUOnlySlowdownCorrectsTheGPUsRegime() {
        let r = report()
        let p = predictions(r)
        // The GPU runs 25 % slow: every route keeps it busy, so every route does (the AMX's share waits for the GPU's).
        let c = RenderLoop.compare(simulated(p) { _ in 1.25 }, predicted: p)
        let gap = c.error ?? 0
        XCTAssertGreaterThan(gap, 0.10, "a gap the tolerance does not hold")
        XCTAssertLessThan(gap, 0.40)
        guard case let .correct(factors) = RenderLoop.verdict(c, report: r) else { return XCTFail("\(RenderLoop.verdict(c, report: r))") }
        XCTAssertEqual(factors["gpu"] ?? 0, 1.25, accuracy: 1e-3)
        XCTAssertNil(factors["amx"], "the AMX is hidden behind the GPU: nothing to tell")
        // Applied, the prediction is the render: nothing is left of the gap.
        let learned = r.applying(RenderLoop.Note(request: request, map: r.date, comparison: c, verdict: .correct(factors)))
        XCTAssertEqual(learned.regime?.learned?["gpu"] ?? 0, 1.25, accuracy: 1e-3)
        let again = RenderLoop.predictions(for: request, table: table(), report: learned)!
        let after = RenderLoop.compare(simulated(p) { _ in 1.25 }, predicted: again)
        XCTAssertLessThan(abs(after.error ?? 1), 0.02)
        XCTAssertEqual(RenderLoop.verdict(after, report: learned), .held)
    }

    func testAnAMXThatRunsHotIsCorrectedWhereItIsTheEngineTheRouteWaitsFor() {
        // The AMX's routes at 0.55 of the GPU's time: a slowdown of the AMX by 1.5 leaves them at 0.82, the choice stands.
        let r = report(amxRatio: 0.55)
        let p = predictions(r)
        let c = RenderLoop.compare(simulated(p) { $0 == "gpu+amx" ? 1.5 : 1 }, predicted: p)
        XCTAssertGreaterThan(c.error ?? 0, 0.10)
        guard case let .correct(factors) = RenderLoop.verdict(c, report: r) else { return XCTFail("\(RenderLoop.verdict(c, report: r))") }
        XCTAssertEqual(factors["amx"] ?? 0, 1.5, accuracy: 1e-3)
        XCTAssertNil(factors["gpu"])
    }

    func testACorrectionThatWouldChangeRoutesIsAMapToMeasureAgain() {
        // The AMX only just beat the GPU (0.9); at 1.5 times the heat it loses on every class that chose it.
        let r = report(amxRatio: 0.9)
        let p = predictions(r)
        let c = RenderLoop.compare(simulated(p) { $0 == "gpu+amx" ? 1.5 : 1 }, predicted: p)
        XCTAssertEqual(RenderLoop.verdict(c, report: r), .remeasure(.decisions))
    }

    func testARenderTheCatalogueDoesNotSpeakForJudgesNothing() {
        let r = report()
        let p = predictions(r)
        // A computation the catalogue does not carry takes most of the render.
        let extra = ProductLedger.Tally(work: .product(k: 77, n: 99, m: 2000, kind: .bf16), route: "gpu", executions: 10,
                                        seconds: simulated(p).reduce(0) { $0 + $1.seconds } * 2)
        let c = RenderLoop.compare(simulated(p) + [extra], predicted: p)
        XCTAssertLessThan(c.coverage, 0.8)
        XCTAssertEqual(c.error ?? 1, 0, accuracy: 1e-12, "what the map speaks for is on the prediction")
        XCTAssertEqual(RenderLoop.verdict(c, report: r), .unjudged(.uncovered))
        // Unpredicted seconds that stay small neither blame nor absolve it.
        let little = ProductLedger.Tally(work: .product(k: 77, n: 99, m: 2000, kind: .bf16), route: "gpu", executions: 1, seconds: 0.01)
        XCTAssertEqual(RenderLoop.verdict(RenderLoop.compare(simulated(p) + [little], predicted: p), report: r), .held)
        // No map classes: nothing to judge against.
        var empty = r
        empty.map!.classes = []
        XCTAssertEqual(RenderLoop.verdict(RenderLoop.compare(simulated(p), predicted: p), report: empty), .unjudged(.noMap))
    }

    func testARouteThatIsNotThePredictedOneSaysNothingOfTheRegime() {
        let r = report()
        let p = predictions(r)
        // The AMX products ran on the GPU alone (the engine refused): slower, but that is no regime.
        var t = simulated(p)
        for i in t.indices where t[i].route == "gpu+amx" { t[i].route = "gpu"; t[i].seconds *= 1.25 }
        let c = RenderLoop.compare(t, predicted: p)
        XCTAssertLessThan(c.routeAgreement, 0.6)
        XCTAssertNotNil(c.error)
        let verdict = RenderLoop.verdict(c, report: r)
        if case let .correct(f) = verdict { XCTAssertNil(f["gpu"], "no factor from the classes that did not run the predicted route") }
    }

    // MARK: what is kept

    func testACorrectionAppliesToItsOwnMapOnlyAndCompounds() {
        let r = report(date: "2026-10-29T10:00:00Z")
        let c = RenderLoop.Comparison(rows: [])
        let note = RenderLoop.Note(request: request, map: r.date, comparison: c, verdict: .correct(["gpu": 1.2]))
        let once = r.applying(note)
        XCTAssertEqual(once.regime?.learned?["gpu"] ?? 0, 1.2, accuracy: 1e-12)
        XCTAssertEqual(once.regime?.heat, r.regime?.heat, "the map's own regime is not rewritten")
        XCTAssertEqual(once.routes, r.routes, "nor its routes")
        XCTAssertEqual(once.applying(note).regime?.learned?["gpu"] ?? 0, 1.44, accuracy: 1e-9)
        // Another map's note, a held map, a map to measure again: nothing to apply.
        let other = RenderLoop.Note(request: request, map: "2026-01-01T00:00:00Z", comparison: c, verdict: .correct(["gpu": 2]))
        XCTAssertEqual(r.applying(other), r)
        for v in [RenderLoop.Verdict.held, .remeasure(.gap), .unjudged(.uncovered)] {
            XCTAssertEqual(r.applying(RenderLoop.Note(request: request, map: r.date, comparison: c, verdict: v)), r)
        }
    }

    func testTheNextLaunchActsOnTheLastNoteAndNeverTwice() {
        let r = report()
        let c = RenderLoop.Comparison(rows: [])
        func note(_ v: RenderLoop.Verdict, map: String? = nil) -> RenderLoop.Note { .init(request: request, map: map ?? r.date, comparison: c, verdict: v) }
        // No note, a held map: as it is.
        XCTAssertEqual(RenderLoop.launching(r, note: nil).map, r)
        XCTAssertFalse(RenderLoop.launching(r, note: nil).consumed)
        let held = RenderLoop.launching(r, note: note(.held))
        XCTAssertEqual(held.map, r); XCTAssertFalse(held.remeasure); XCTAssertTrue(held.consumed)
        // A correction is applied and consumed: the corrected map is kept, so the note is not applied again.
        let fixed = RenderLoop.launching(r, note: note(.correct(["gpu": 1.2])))
        XCTAssertEqual(fixed.map.regime?.learned?["gpu"] ?? 0, 1.2, accuracy: 1e-12)
        XCTAssertFalse(fixed.remeasure); XCTAssertTrue(fixed.consumed)
        // A map to measure again is measured at this launch, whatever the heat.
        let rest = RenderLoop.launching(r, note: note(.remeasure(.gap)))
        XCTAssertTrue(rest.remeasure); XCTAssertTrue(rest.consumed); XCTAssertEqual(rest.map, r)
        // A note of another map speaks of nothing that is kept.
        let stale = RenderLoop.launching(r, note: note(.remeasure(.gap), map: "2026-01-01T00:00:00Z"))
        XCTAssertFalse(stale.remeasure); XCTAssertTrue(stale.consumed)
        XCTAssertEqual(RenderLoop.launching(r, note: note(.unjudged(.uncovered))).consumed, true)
    }

    func testTheNoteRoundTripsAndStaysWithinTheLinksBytes() throws {
        let r = report()
        let p = predictions(r)
        let c = RenderLoop.compare(simulated(p) { _ in 1.15 }, predicted: p)
        let verdict = RenderLoop.verdict(c, report: r)
        let note = RenderLoop.Note(request: request, map: r.date, comparison: c, verdict: verdict)
        XCTAssertEqual(note.classes.count, RenderLoop.Note.linkClasses)
        let data = try JSONEncoder().encode(note)
        XCTAssertEqual(try JSONDecoder().decode(RenderLoop.Note.self, from: data), note)
        XCTAssertEqual(note.decision, verdict)
        print("RENDERLOOP note \(data.count) bytes, \(note.classes.count) classes")
        XCTAssertLessThan(data.count, 1800)
        // Attached to the report, which keeps its link short, and read back.
        let withNote = r.with(render: note)
        XCTAssertLessThan(try withNote.issueURL().absoluteString.utf8.count, 8000)
        XCTAssertEqual(try JSONDecoder().decode(Diagnostic.Report.self, from: withNote.json()), withNote)
        // A report without the note, and a regime without a correction, are what they were: the fields are optional.
        XCTAssertNil(try JSONDecoder().decode(Diagnostic.Report.self, from: r.json()).render)
        XCTAssertNil(try JSONDecoder().decode(Diagnostic.Report.self, from: r.json()).regime?.learned)
    }

    // MARK: as text

    func testTheTableIsOneLineOfTabsPerClassUnderAHeader() {
        let r = report()
        let p = predictions(r)
        let c = RenderLoop.compare(simulated(p) { _ in 1.1 }, predicted: p)
        let lines = RenderLoop.lines(c, title: "render classes")
        XCTAssertEqual(lines[0], "render classes")
        XCTAssertEqual(lines[1].split(separator: "\t", omittingEmptySubsequences: false).map(String.init),
                       ["#class", "route", "executions", "measured_s", "predicted_s", "error", "predicted_route", "basis"])
        let body = lines[2..<(2 + c.rows.count)]
        for line in body { XCTAssertEqual(line.split(separator: "\t", omittingEmptySubsequences: false).count, 8, line) }
        XCTAssertTrue(body.allSatisfy { !$0.hasPrefix("#") })
        XCTAssertTrue(lines.suffix(4).map { String($0.prefix(while: { $0 != "\t" })) } == ["#total", "#products", "#attention", "#coverage"])
        XCTAssertTrue(body.first!.contains("+10.0%"))
        // Without a map the prediction columns are empty and the lines still parse.
        let bare = RenderLoop.lines(RenderLoop.compare(simulated(p)), title: "t")
        XCTAssertTrue(bare[2].split(separator: "\t", omittingEmptySubsequences: false)[4].isEmpty)
    }

    // MARK: routes, with the weight's form

    func testThePredictedRouteNamesTheWeightsFormAndKeepsTheANEOutOfAProcessWithoutIt() {
        let classes = CC.classes(for: .init(family: .zImage, variant: .compact, includesEncoder: false))!
        let c = classes.first { $0.routeKey.id == "product:3840x3840@4128" }!
        XCTAssertEqual(c.weightKind, .q8)
        var t = table(weights: [.init(kind: .q8, path: .native)])
        XCTAssertEqual(CC.route(of: c, family: .zImage, table: t, weightForms: true), "gpu+amx/q8")
        XCTAssertEqual(CC.route(of: c, family: .zImage, table: t, weightForms: false), "gpu+amx")
        XCTAssertEqual(CC.route(of: c, family: .zImage, table: t, amxMinimumRows: 5000, weightForms: true), "gpu/q8")
        t = table(weights: [.init(kind: .bf16, path: .bf16, roundsActivation: true, minRows: 256), .init(kind: .bf16, path: .bf16)])
        let bf16 = CC.classes(for: .init(family: .zImage, includesEncoder: false))!.first { $0.routeKey.id == "product:3840x3840@4128" }!
        XCTAssertEqual(CC.route(of: bf16, family: .zImage, table: t, weightForms: true), "gpu/bf16act")
        let small = CC.classes(for: .init(family: .zImage, includesEncoder: false))!.first { $0.routeKey.id == "product:3840x3840@32" }!
        XCTAssertEqual(CC.route(of: small, family: .zImage, table: t, weightForms: true), "gpu/bf16", "under the rounding's rows")
        // The ANE: a class rule that names it gives the route it keeps without the ANE where the process did not bring it up.
        var ane = table()
        ane.products.ane = .init(enabled: true, minimumRows: 1024)
        ane.products.classes = [.init(engines: .ane, k: 3840, n: 3840, minRows: 4128, maxRows: 4128, withoutANE: .gpu)]
        XCTAssertEqual(CC.route(of: bf16, family: .zImage, table: ane), "gpu+ane")
        XCTAssertEqual(CC.route(of: bf16, family: .zImage, table: ane, aneBroughtUp: false), "gpu")
        // Anima's cross-attention never takes the flash kernel; its self-attention does.
        let anima = CC.classes(for: .init(family: .anima, includesEncoder: false))!.filter { if case .attention = $0.shape { return true } else { return false } }
        XCTAssertEqual(anima.count, 4, "two grids, a self- and a cross-attention each")
        for a in anima {
            guard case let .attention(_, _, _, _, _, cross) = a.shape else { continue }
            XCTAssertEqual(CC.route(of: a, family: .anima, table: table()), cross ? "graph" : "flash")
        }
    }

    func testTheRenderOfAnyFamilyMakesARequestTheCatalogueReads() {
        for family in Family.allCases {
            let r = CC.Request(family: family, variant: .standard, width: 1024, height: 1536, fasterRendering: true, steps: 8, includesEncoder: false, halfSteps: 2)
            let passes = CC.passes(r)
            XCTAssertEqual(passes.map(\.evaluations), [2, 6], "\(family): the render's own half-grid count")
        }
    }
}
