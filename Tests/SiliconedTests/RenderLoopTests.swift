import Foundation
import Testing
@testable import Siliconed

/// **The render checks the plan** (`ProductLedger`, `RenderLoop`, `RenderLoopVerdict`): the ledger's lines merged by class, the prediction
/// the plan's own assignments are timed to by its model, the gap, the rule that holds or flags it, and the note it leaves — never a change
/// of the plan. No GPU, no weights: a synthetic model over the catalogue's own classes, and renders simulated by scaling the prediction.
@Suite struct RenderLoopTests {
    typealias CC = CapacityClasses
    typealias M = CapacityModel
    let gpu = M.Kernel(.gpu, .fp32), amx = M.Kernel(.amx, .fp32), ane = M.Kernel(.ane, .fp16)

    var request: CC.Request { .init(family: .zImage, variant: .standard, width: 1024, height: 1024, includesEncoder: false) }

    /// An M1 Pro's order of lines, every engine and the attention kernels, the weights' preparation.
    func model() -> M {
        var m = M()
        m.lines[gpu] = .init(t0: 0.0003, rate: 3.5e12)
        m.lines[amx] = .init(t0: 0.0009, rate: 2.3e12)
        m.lines[ane] = .init(t0: 0.00125, rate: 4.9e12)
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0.0001, rate: 1.4e12)
        m.lines[M.Kernel(.gpu, .flash)] = .init(t0: 0.0007, rate: 1.6e12)
        m.lines[M.Kernel(.cpu, .heads)] = .init(t0: 0, rate: 1.0e12)
        m.lines[M.Kernel(.gpu, .bf16)] = .init(t0: 0.0002, rate: 2.2e12)
        m.sharing["gpu+amx"] = .init(overhead: 0.00001, keep: [.gpu: 0.95, .amx: 0.88])
        m.sharing["gpu+ane"] = .init(overhead: 0.00001, keep: [.gpu: 0.87, .ane: 0.71])
        m.sharing["gpu+amx+ane"] = .init(overhead: 0.00008, keep: [.gpu: 0.8, .amx: 0.65, .ane: 0.7])
        m.preparation[.bf16] = [.widened: 4.6e-11, .bf16: 2.6e-11]
        return m
    }

    func predictions(_ plan: CapacityPlan? = nil, aneBroughtUp: Bool = true) -> [RenderLoop.Predicted] {
        RenderLoop.predictions(for: request, plan: plan ?? CapacityPlan(model: model()), aneBroughtUp: aneBroughtUp)!
    }

    /// The render that ran the prediction, each line `scale(route)` times slower.
    func simulated(_ predicted: [RenderLoop.Predicted], scale: (String) -> Double = { _ in 1 }) -> [ProductLedger.Tally] {
        predicted.map { .init(work: $0.work, route: $0.route, executions: $0.executions, seconds: $0.seconds * scale($0.route)) }
    }

    // MARK: the ledger

    @Test func theLedgerIsArmedForTheDenoisingAlone() throws {
        #expect(ProductLedger.current == nil)
        let ledger = ProductLedger()
        ProductLedger.recording(into: ledger) {
            #expect(ProductLedger.current === ledger)
            ProductLedger.attention(sequence: 4128, heads: 30, headDim: 128, route: "flash")?.finish()
            #expect(ProductLedger.attention(sequence: 32, heads: 30, headDim: 128, route: "graph") == nil, "under 64 rows: no class")
        }
        #expect(ProductLedger.current == nil)
        #expect(ProductLedger.attention(sequence: 4128, heads: 30, headDim: 128, route: "flash") == nil, "nothing records outside a render")
        #expect(ledger.tallies.count == 1 && ledger.tallies[0].executions == 1)
        struct Boom: Error {}
        #expect(throws: Boom.self) { try ProductLedger.recording(into: ProductLedger()) { throw Boom() } }
        #expect(ProductLedger.current == nil)
    }

    @Test func theLedgerCountsAndSumsPerWorkAndRoute() {
        let ledger = ProductLedger()
        let w = ProductLedger.Work.product(k: 3840, n: 3840, m: 4128, weight: .bf16)
        ledger.add(w, route: "gpu+amx", seconds: 0.5); ledger.add(w, route: "gpu+amx", seconds: 0.25); ledger.add(w, route: "gpu", seconds: 1)
        let t = ledger.tallies
        #expect(t.count == 2)
        #expect(t.first { $0.route == "gpu+amx" }?.executions == 2)
        #expect(abs((t.first { $0.route == "gpu+amx" }?.seconds ?? 0) - 0.75) < 1e-12)
        #expect(t == ledger.tallies, "a fixed order")
        // A K-quant weight is its own class, not a bf16 one.
        #expect(ProductLedger.Work.product(k: 1, n: 1, m: 1, weight: .q4k) != .product(k: 1, n: 1, m: 1, weight: .bf16))
    }

    /// On the stream, an execution is its engines' span, not the encoder's wall: `max(GPU, ANE, wall)`, counted once, whatever the
    /// order the seconds arrive in (the ANE's thread, a drain before or after the call returns).
    @Test func aChargeIsTheLongerSideCountedOnce() {
        let ledger = ProductLedger()
        let w = ProductLedger.Work.product(k: 3840, n: 3840, m: 4128, weight: .bf16)
        let pair = ProductLedger.Charge(ledger: ledger)
        pair.addGPU(0.010); pair.credit()
        #expect(ledger.tallies.isEmpty, "nothing before the call returns: its route is not known yet")
        pair.addANE(0.012)
        pair.finish(w, route: "gpu+ane/bf16act", wall: 0.0001)
        pair.addGPU(0.004); pair.credit(); pair.credit()          // a second buffer, drained after the return; credited twice
        let alone = ProductLedger.Charge(ledger: ledger)
        alone.finish(w, route: "gpu/bf16act", wall: 0.0002)
        alone.addGPU(0.020); alone.credit()
        let t = ledger.tallies
        #expect(t.map(\.executions) == [1, 1])
        #expect(abs((t.first { $0.route == "gpu+ane/bf16act" }?.seconds ?? 0) - 0.014) < 1e-12, "the GPU's 14 ms over the ANE's 12")
        #expect(abs((t.first { $0.route == "gpu/bf16act" }?.seconds ?? 0) - 0.020) < 1e-12)
    }

    // MARK: classes

    @Test func heightsWithin64RowsAreOneClassAndOthersAreNot() {
        let t: [ProductLedger.Tally] = [
            .init(work: .product(k: 3840, n: 3840, m: 4128, weight: .bf16), route: "gpu+amx", executions: 120, seconds: 12),
            .init(work: .product(k: 3840, n: 3840, m: 4096, weight: .bf16), route: "gpu+amx", executions: 8, seconds: 0.8),
            .init(work: .product(k: 3840, n: 3840, m: 4150, weight: .bf16), route: "gpu", executions: 2, seconds: 0.3),   // 22 from 4128: chained
            .init(work: .product(k: 3840, n: 3840, m: 4300, weight: .bf16), route: "gpu", executions: 1, seconds: 0.1),   // 150 from 4150: apart
            .init(work: .product(k: 3840, n: 3840, m: 4128, weight: .int8), route: "gpu/int8", executions: 3, seconds: 0.2),   // another format
            .init(work: .product(k: 3840, n: 10240, m: 4128, weight: .bf16), route: "gpu", executions: 1, seconds: 0.1),     // another shape
            .init(work: .attention(sequence: 4128, heads: 30, headDim: 128), route: "flash", executions: 30, seconds: 4),
            .init(work: .attention(sequence: 4096, heads: 30, headDim: 128), route: "flash", executions: 2, seconds: 0.2),  // exact: apart
        ]
        let c = RenderLoop.compare(t)
        #expect(c.rows.count == 6)
        let merged = c.rows.first { $0.id == "product:3840x3840@4128" }!
        #expect(merged.executions == 130)
        #expect(abs(merged.measured - 13.1) < 1e-9)
        #expect(merged.routes == ["gpu+amx", "gpu"], "the route with the most seconds first")
        #expect(c.rows.contains { $0.id == "product:3840x3840@4128~int8" } && c.rows.contains { $0.id == "product:3840x3840@4300" })
        #expect(c.rows.filter(\.isAttention).count == 2)
        #expect(abs(c.measured - t.reduce(0) { $0 + $1.seconds }) < 1e-9, "merging loses nothing")
        #expect(c.predicted == nil && c.error == nil, "no model: the measure alone")
        #expect(c.rows.map(\.measured) == c.rows.map(\.measured).sorted(by: >), "heaviest first")
    }

    @Test func theMergedClassHoldsTheHeightsOfBothTheRenderAndThePrediction() {
        // The prediction at 4 128 rows and a render that ran 4 117 (another prompt's text): one class, matched.
        let p = RenderLoop.Predicted(work: .product(k: 3840, n: 3840, m: 4128, weight: .bf16), route: "gpu+amx", executions: 10, seconds: 2)
        let t = ProductLedger.Tally(work: .product(k: 3840, n: 3840, m: 4117, weight: .bf16), route: "gpu+amx", executions: 10, seconds: 2.2)
        let c = RenderLoop.compare([t], predicted: [p])
        #expect(c.rows.count == 1)
        #expect(abs((c.rows[0].error ?? 0) - 0.1) < 1e-9)
        #expect(c.coverage == 1)
    }

    // MARK: the prediction

    /// **The prediction is the plan's assignment timed by the model that solved it**: class by class, the solver's own seconds.
    @Test func thePredictionIsThePlansAssignmentTimedByItsModel() throws {
        let plan = CapacityPlan(model: model())
        let p = predictions(plan)
        #expect(!p.isEmpty)
        let classes = CC.classes(for: request)!.filter { $0.phase == .denoising }
        #expect(p.count == classes.count, "every class of the catalogue is predicted")
        for c in classes {
            guard case let .product(k, n, rows, format) = c.shape else { continue }
            let key = ProductKey(k: k, n: n, m: rows, weight: format, phase: c.phase)
            let a = plan.assign(key)
            let predicted = try #require(p.first { $0.work == .product(k: k, n: n, m: rows, weight: format) }, "\(key)")
            #expect(predicted.route == a.route(weight: format))
            let solved = try #require(CapacitySolver.product(assigned: a, key: key, model: model(), constraints: .init()))
            #expect(abs(predicted.seconds - Double(c.executions[.zImage]!) * solved.compute) < 1e-9 * predicted.seconds, "the call, not the weight's preparation")
        }
        // By hand: a product the plan leaves whole to the GPU is the line, the weight's widening not counted (the ledger times the call).
        let big = try #require(p.first { $0.work == .product(k: 3840, n: 3840, m: 128, weight: .bf16) } ?? p.first { if case .product = $0.work { return $0.route == "gpu" } else { return false } })
        guard case let .product(k, n, rows, _) = big.work else { return }
        let once = 0.0003 + 2.0 * Double(rows) * Double(k) * Double(n) / 3.5e12
        #expect(abs(big.seconds / Double(big.executions) - once) < 1e-9 + 1e-6 * once)
        // The prediction is the plan's, so a plan that gives the ANE rows is predicted to take them.
        #expect(p.contains { $0.route.contains("ane") })
    }

    /// A process that did not bring the Neural Engine up is predicted without it: the plan restricted, as the engine's own.
    @Test func aProcessWithoutTheANEIsPredictedWithoutIt() {
        let with = predictions(aneBroughtUp: true), without = predictions(aneBroughtUp: false)
        #expect(with.contains { $0.route.contains("ane") })
        #expect(!without.contains { $0.route.contains("ane") })
        #expect(without.reduce(0) { $0 + $1.seconds } > 0)
    }

    @Test func aForcedPlanIsPredictedAsItRuns() {
        // The plan of the checks (`SILICONED_PLAN=amx`) has no model: nothing measured, nothing to predict from.
        #expect(RenderLoop.predictions(for: request, plan: .forcing(amx: true)) == nil)
        #expect(RenderLoop.predictions(for: request, plan: .defaults) == nil)
        // A model without a kernel the plan uses leaves that class unpredicted, not wrong: a row of its own.
        var m = model()
        m.lines[ane] = nil
        let plan = CapacityPlan(model: m, allowsANE: true)
        #expect(RenderLoop.predictions(for: request, plan: plan)?.isEmpty == false)
        // A version the family does not publish is not predicted.
        #expect(RenderLoop.predictions(for: .init(family: .mossImage, variant: .light), plan: CapacityPlan(model: model())) == nil)
    }

    // MARK: the gap and the verdict

    @Test func theGapIsZeroWhenTheRenderIsThePredictionAndTheModelHolds() {
        let p = predictions()
        let c = RenderLoop.compare(simulated(p), predicted: p)
        for row in c.rows { #expect(abs(row.error ?? 1) < 1e-9, "\(row.id)") }
        #expect(abs(c.error ?? 1) < 1e-12 && abs(c.productError ?? 1) < 1e-12 && abs(c.attentionError ?? 1) < 1e-12)
        #expect(abs(c.coverage - 1) < 1e-12 && abs(c.routeAgreement - 1) < 1e-12)
        #expect(RenderLoop.verdict(c) == .held)
        // A uniform ×1.2 is a +20 % gap on every row and on the total: the model drifted.
        let slow = RenderLoop.compare(simulated(p) { _ in 1.2 }, predicted: p)
        #expect(abs((slow.error ?? 0) - 0.2) < 1e-9)
        for row in slow.rows { #expect(abs((row.error ?? 0) - 0.2) < 1e-9) }
        #expect(RenderLoop.verdict(slow) == .drift)
        // Within the tolerance, either side: held.
        for scale in [0.92, 1.08] { #expect(RenderLoop.verdict(RenderLoop.compare(simulated(p) { _ in scale }, predicted: p)) == .held) }
        #expect(RenderLoop.verdict(RenderLoop.compare(simulated(p) { _ in 0.8 }, predicted: p)) == .drift)
    }

    /// The gap is read by block: a drift of the attentions alone names the attentions.
    @Test func theGapIsReadByBlock() {
        let p = predictions()
        let c = RenderLoop.compare(simulated(p), predicted: p)
        let attention = RenderLoop.compare(p.map { x -> ProductLedger.Tally in
            var scale = 1.0
            if case .attention = x.work { scale = 1.5 }
            return .init(work: x.work, route: x.route, executions: x.executions, seconds: x.seconds * scale)
        }, predicted: p)
        #expect(abs(c.attentionError ?? 1) < 1e-12)
        #expect(abs((attention.attentionError ?? 0) - 0.5) < 1e-9 && abs(attention.productError ?? 1) < 1e-9)
        #expect((attention.error ?? 0) > 0 && (attention.error ?? 0) < 0.5)
        #expect(attention.rows.filter { $0.isAttention }.allSatisfy { abs(($0.error ?? 0) - 0.5) < 1e-9 })
    }

    @Test func aRenderTheCatalogueDoesNotSpeakForJudgesNothing() {
        let p = predictions()
        // Half of the render's seconds in classes the catalogue does not carry: coverage under 80 %, unjudged — a gap there is not the model's.
        var t = simulated(p) { _ in 3 }
        let total = t.reduce(0) { $0 + $1.seconds }
        t.append(.init(work: .product(k: 77, n: 99, m: 4128, weight: .bf16), route: "gpu", executions: 1, seconds: total * 1.5))
        let c = RenderLoop.compare(t, predicted: p)
        #expect(c.coverage < 0.8)
        #expect(RenderLoop.verdict(c) == .unjudged(.uncovered))
        let covered = RenderLoop.compare(simulated(p) + [.init(work: .product(k: 77, n: 99, m: 4128, weight: .bf16), route: "gpu", executions: 1, seconds: 0.001)],
                                         predicted: p)
        #expect(RenderLoop.verdict(covered) == .held, "an unknown class neither blames nor absolves the model")
        // No model, nothing to judge.
        #expect(RenderLoop.verdict(RenderLoop.compare(simulated(p))) == .unjudged(.noModel))
        #expect(RenderLoop.verdict(RenderLoop.compare([])) == .unjudged(.noModel))
    }

    @Test func aRouteThatIsNotThePredictedOneShowsInTheAgreement() {
        let p = predictions()
        let t = p.map { x -> ProductLedger.Tally in
            .init(work: x.work, route: x.route.contains("ane") ? "gpu" : x.route, executions: x.executions, seconds: x.seconds)
        }
        let c = RenderLoop.compare(t, predicted: p)
        #expect(c.routeAgreement < 1 && abs(c.error ?? 1) < 1e-9)
    }

    // MARK: as text, and the note

    @Test func theTableIsOneLineOfTabsPerClassUnderAHeader() {
        let p = predictions()
        let c = RenderLoop.compare(simulated(p) { _ in 1.1 }, predicted: p)
        let lines = RenderLoop.lines(c, title: "render classes")
        #expect(lines[0] == "render classes")
        let header: [String] = lines[1].split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
        #expect(header == ["#class", "route", "executions", "measured_s", "predicted_s", "error", "predicted_route"])
        let body = lines[2..<(2 + c.rows.count)]
        for line in body { #expect(line.split(separator: "\t", omittingEmptySubsequences: false).count == 7, "\(line)") }
        #expect(body.allSatisfy { !$0.hasPrefix("#") })
        let heads: [String] = lines.suffix(5).map { (line: String) -> String in String(line.prefix(while: { $0 != "\t" })) }
        #expect(heads == ["#total", "#products", "#attention", "#coverage", "#fallbacks"])
        #expect(body.first!.contains("+10.0%"))
        // Without a model the prediction columns are empty and the lines still parse.
        let bare = RenderLoop.lines(RenderLoop.compare(simulated(p)), title: "t")
        #expect(bare[2].split(separator: "\t", omittingEmptySubsequences: false)[4].isEmpty)
    }

    @Test func theNoteRoundTripsAndStaysShort() throws {
        let p = predictions()
        let c = RenderLoop.compare(simulated(p) { _ in 1.07 }, predicted: p)
        let note = RenderLoop.Note(request: request, map: "2026-10-30T10:00:00Z", comparison: c, verdict: RenderLoop.verdict(c))
        #expect(note.family == "z-image" && note.format == "1024x1024" && note.verdict == "held" && note.map == "2026-10-30T10:00:00Z")
        #expect(note.classes.count == RenderLoop.Note.linkClasses)
        #expect(abs((note.error ?? 0) - 0.07) < 1e-9)
        let data = try JSONEncoder().encode(note)
        #expect(try JSONDecoder().decode(RenderLoop.Note.self, from: data) == note)
        #expect(data.count < 2_000)
        // A note never acts: it holds no factor, and the plan answers the same after as before.
        let text = String(decoding: data, as: UTF8.self)
        #expect(!text.contains("factors") && !text.contains("correct"))
    }

    @Test func aRenderTheCatalogueReadsAndTheCheckThatBuildsItsNote() throws {
        for family in Family.allCases {
            let r = CC.Request(family: family, variant: .standard, width: 1024, height: 1536, fasterRendering: true, steps: 8, includesEncoder: false, halfSteps: 2)
            #expect(CC.passes(r).map(\.evaluations) == [2, 6], "\(family): the render's own half-grid count")
        }
        #expect(RenderLoop.Reason.noModel.rawValue == "noModel" && RenderLoop.Verdict.unjudged(.uncovered).name == "unjudged:uncovered")
        #expect(RenderLoop.Verdict.drift.name == "drift" && RenderLoop.Verdict.held.name == "held")
    }

    /// **The band is the map's own uncertainty where that is wider than 10 %**: a prediction read on the quick side of calls that disagreed
    /// by ±x % is not held to better than x %, and a map of a free Mac keeps the band it had.
    @Test func theToleranceWidensWithTheMapsOwnUncertainty() {
        var report = ReportFixtures.report()
        #expect(RenderLoop.Thresholds(uncertainty: report).tolerance == 0.10, "no regime, nothing to widen")
        var regime = Diagnostic.Regime()
        regime.spreads = ["gpu+ane 4096x3840x3840 pass 1": 0.05, "ane/fp16 1024x3840x3840": 0.03]
        report.regime = regime
        #expect(RenderLoop.Thresholds(uncertainty: report).tolerance == 0.10, "a free Mac's ±2.5 % keeps the 10 %")
        regime.spreads["gpu+ane 4096x3840x3840 pass 1"] = 0.40
        report.regime = regime
        let widened = RenderLoop.Thresholds(uncertainty: report)
        #expect(widened.tolerance == 0.20 && widened.minimumCoverage == RenderLoop.Thresholds().minimumCoverage)
        // A gap of 15 % is a drift on a free Mac's map and holds on a map that says it is ±20 % sure.
        let p = predictions()
        let off = RenderLoop.compare(simulated(p) { _ in 1.15 }, predicted: p)
        #expect(RenderLoop.verdict(off) == .drift && RenderLoop.verdict(off, thresholds: widened) == .held)
    }
}
