import Foundation
import XCTest
@testable import Siliconed

/// **The capacity map end to end, without a machine** (`CapacityMapClasses.swift`, `CapacityMeasure.plan`, `RouteTable(plan:…)`,
/// `Diagnostic.Report` of schema 14): the grid and its reach over every family, a product decided per weight form, a simulated
/// Mac whose engines have known rates producing the right table inside its budget, the regime read from the witnesses, green
/// and orange.
final class CapacityMapTests: XCTestCase {
    typealias CM = CapacityMeasure
    typealias CC = CapacityClasses

    private let all = CapacityMap.Capabilities(amx: true, ane: true, bf16Weights: true, bf16Rounding: true, int8Weights: true,
                                               q8Weights: true, flash: true, headSplit: true)
    private var grid: [CapacityMap.Entry] { CapacityMap.catalogue(caps: all) }
    /// A quarter of the grid — its biggest points at both heights and an attention at each — for the quarter of the minute
    /// the simulations of one property are given.
    private var part: [CapacityMap.Entry] {
        grid.filter { ["product:8192x8192@1024", "product:4096x16384@1024", "product:2048x2048@4096", "product:16384x4096@4096",
                       "attention:1024/32x128", "attention:4096/16x128"].contains($0.id) }
    }

    /// A fictional Mac's preparation: seconds per value, widened and copied, by form — the 8-bit weights widen slower and copy
    /// faster than bf16's.
    private let preparation: CapacityMap.Preparation = [.bf16: (0.4e-9, 0.2e-9), .int8: (0.7e-9, 0.1e-9), .q8: (0.9e-9, 0.1e-9)]

    // MARK: the grid

    func testTheGridKnowsNoModel() {
        let g = grid
        XCTAssertEqual(g.count, CapacityMap.gridShapes.count * CapacityMap.gridHeights.count
                                 + CapacityMap.gridAttention.count * CapacityMap.gridHeights.count)
        XCTAssertEqual(Set(g.map(\.id)).count, g.count, "an id names one point")
        XCTAssertTrue(g.allSatisfy { $0.weight == 1 })
        XCTAssertEqual(g, CapacityMap.catalogue(caps: all), "deterministic")
        // The same grid whatever the Mac holds: only the machine's capabilities change its routes, never its points.
        XCTAssertEqual(CapacityMap.catalogue(caps: .init()).map(\.id), g.map(\.id))
        for e in g { XCTAssertNotNil(RouteTable.ClassKey.parse(e.id), e.id) }
    }

    /// **Every product and attention that weighs in a render of any family, version and format falls within the grid's
    /// reach** — the table's nearest rule (`RouteTable.Nearest`) finds a point for it, so no model renders its work on the
    /// « GPU alone » left beyond reach. What is out of reach is the 64-wide embedders and the encoders' 1 024-wide
    /// projections at a prompt's rows: under 0.02 % of a render each, the simplest route by design (64×3840 must not
    /// borrow 3840×3840's three engines). A family added to the catalogue joins this test, not the grid.
    func testEveryFamilyIsWithinTheGridsReach() {
        let shapes = CapacityMap.gridShapes.map { (k: $0.k, n: $0.n) }
        let heads = CapacityMap.gridAttention.map { (k: $0.heads, n: $0.headDim) }
        var checked = 0
        for family in Family.allCases {
            for variant in Variant.allCases {
                for (w, h) in [(512, 512), (1024, 1024), (1024, 1536), (1536, 1024)] {
                    for faster in [true, false] {
                        let r = CC.Request(family: family, variant: variant, width: w, height: h, fasterRendering: faster)
                        guard let classes = CC.classes(for: r) else { continue }
                        let shares = CC.weights(classes, family: family, rates: CapacityMap.referenceRates)
                        var outside = 0.0
                        for (c, share) in zip(classes, shares) {
                            let reached: Bool
                            switch c.shape {
                            case let .product(k, n, _, _):
                                reached = RouteTable.Nearest.rule(among: shapes, k: k, n: n, m: 0) { ($0, 0, nil) } != nil
                            case let .attention(_, _, hs, _, dim, _):
                                reached = RouteTable.Nearest.rule(among: heads, k: hs, n: dim, m: 0) { ($0, 0, nil) } != nil
                            }
                            if !reached { outside += share }
                            XCTAssertTrue(reached || share < 0.001, "\(family) \(variant) \(w)×\(h): \(c.routeKey.id) (\(c.phase)) beyond the grid's reach")
                            checked += 1
                        }
                        XCTAssertLessThan(outside, 0.001, "\(family) \(variant) \(w)×\(h): \(outside) of the render beyond reach")
                    }
                }
            }
        }
        XCTAssertGreaterThan(checked, 500)
    }

    func testAProductIsDecidedOncePerWeightForm() {
        let e = grid.first { $0.id == "product:4096x4096@4096" }!
        let c = e.compute(preparation: preparation)
        XCTAssertEqual(c.decisions.map(\.id), ["product:4096x4096@4096", "product:4096x4096@4096~int8", "product:4096x4096@4096~q8"])
        func ids(_ v: CM.View) -> [String] { v.routes.map { e.routes[$0].id } }
        let engines = ["gpu", "gpu+amx", "gpu+ane", "gpu+ane+amx"]
        XCTAssertEqual(ids(c.decisions[0]), engines + ["gpu/bf16", "gpu/bf16act"])
        XCTAssertEqual(ids(c.decisions[1]), engines + ["gpu/int8", "gpu+amx/int8"])
        XCTAssertEqual(ids(c.decisions[2]), engines + ["gpu/q8", "gpu+amx/q8"])
        // The engines read the widened weight, the native forms copy it: their preparation, per value of the point's k×n.
        let values = 4096.0 * 4096
        XCTAssertEqual(c.decisions[1].extra[0], values * 0.7e-9, accuracy: 1e-15)
        XCTAssertEqual(c.decisions[1].extra[4], values * 0.1e-9, accuracy: 1e-15)
        // Each decision is a class of its own for the table, a third of the point's weight.
        XCTAssertEqual(c.decisionClasses.map(\.id), c.decisions.map(\.id))
        XCTAssertEqual(c.decisionClasses.map(\.weight), [1.0 / 3, 1.0 / 3, 1.0 / 3])
        let p = RouteTable.ClassKey.parse("product:4096x16384@4096~int8")
        XCTAssertEqual(p?.kind, .int8)
        XCTAssertEqual(p?.key, .product(k: 4096, n: 16384, m: 4096))
        // An attention has one decision, its own.
        let a = grid.first { if case .attention = $0.work { return true } else { return false } }!.compute()
        XCTAssertEqual(a.decisions.map(\.id), [a.id])
    }

    func testRoutesFollowTheMachineAndTheShape() {
        XCTAssertEqual(CapacityMap.productRoutes(m: 4096, caps: .init()).map(\.id), ["gpu"])
        let ids = { (m: Int) in CapacityMap.productRoutes(m: m, caps: self.all).map(\.id) }
        XCTAssertEqual(ids(4096), ["gpu", "gpu+amx", "gpu+ane", "gpu+ane+amx", "gpu/bf16", "gpu/bf16act", "gpu/int8", "gpu+amx/int8",
                                   "gpu/q8", "gpu+amx/q8"])
        XCTAssertEqual(ids(300), ["gpu", "gpu/bf16", "gpu/bf16act", "gpu/int8", "gpu/q8"], "no AMX under 512 rows, no ANE under 1 024")
        XCTAssertEqual(CapacityMap.productRoutes(m: 4096, caps: all).filter(\.efficient).map(\.id), ["gpu+ane", "gpu+ane+amx"])
        XCTAssertEqual(CapacityClasses.engines(ofRoute: "gpu+amx/int8"), ["gpu", "amx"])
        XCTAssertEqual(CapacityMap.attentionRoutes(sequence: 512, caps: all).map(\.id), ["graph", "flash"])
        XCTAssertEqual(CapacityMap.attentionRoutes(sequence: 4096, caps: all).map(\.id), ["graph", "flash", "split"])
    }

    // MARK: a simulated Mac

    /// The engines of a fictional Mac: TFLOP/s by route. The AMX helps everywhere, the ANE from 20 M weights, the three from 50 M,
    /// the bf16 weight loses, the 8-bit ones win; flash and the split win on long sequences. The engines' work alone: the
    /// preparation is the views'.
    private func seconds(_ e: CapacityMap.Entry, _ route: String, heat: [String: Double] = [:]) -> Double? {
        var rate: Double
        switch e.work {
        case let .product(k, n, m):
            let flop = 2 * Double(k) * Double(n) * Double(m), weights = k * n
            switch route {
            case "gpu": rate = 4.0
            case "gpu+amx": rate = 5.2
            case "gpu+ane": rate = weights >= 20_000_000 ? 5.5 : 4.3
            case "gpu+ane+amx": rate = weights >= 50_000_000 ? 5.8 : 5.0
            case "gpu/bf16": rate = 3.2
            case "gpu/bf16act": rate = 3.0
            case "gpu/int8", "gpu/q8": rate = 4.4
            case "gpu+amx/int8", "gpu+amx/q8": rate = 5.6
            default: return nil
            }
            let t = flop / (rate * 1e12)
            let factor = CapacityClasses.engines(ofRoute: route).map { heat[$0] ?? 1 }.max() ?? 1
            return t * factor
        case let .attention(s, h, d):
            let flop = 4 * Double(s) * Double(s) * Double(h) * Double(d)
            switch route {
            case "graph": rate = 2.0
            case "flash": rate = 2.6
            case "split": rate = s >= 2048 ? 3.0 : 1.8
            default: return nil
            }
            let factor = CapacityClasses.engines(ofRoute: route).map { heat[$0] ?? 1 }.max() ?? 1
            return flop / (rate * 1e12) * factor
        }
    }

    /// A quarter of the minute is plenty for a quarter of the grid, and keeps the suite short.
    private static var quarter: CM.Settings { var s = CM.Settings(); s.budget = 15; return s }

    private func classes(_ entries: [CapacityMap.Entry]) -> [CM.ComputeClass] { entries.map { $0.compute(preparation: preparation) } }

    private func simulate(_ entries: [CapacityMap.Entry], heat: [String: Double] = [:], noise: Double = 0.02,
                          settings: CM.Settings = CapacityMapTests.quarter, finalHeat: [String: Double] = [:],
                          drift: @escaping (Double) -> Double = { _ in 1 }) -> (CM.Plan, CapacityMeasureTests.Sim) {
        let base = entries.map { e in e.routes.map { seconds(e, $0.id, heat: heat) } }
        let sim = CapacityMeasureTests.Sim(base: base, noise: noise, drift: drift)
        let plan = CM.plan(classes: classes(entries), settings: settings,
                           finalHeat: finalHeat.isEmpty ? nil : { CM.Heat(finalHeat) }, clock: sim) { sim.probe($0, $1) }
        return (plan, sim)
    }

    /// Every decision of the grid: its point, its view.
    private func decisions(_ entries: [CapacityMap.Entry]) -> [(entry: CapacityMap.Entry, view: CM.View)] {
        zip(entries, classes(entries)).flatMap { e, c in c.decisions.map { (e, $0) } }
    }

    /// What the rule chooses from the true hot times of a decision (preparation included), and by how much the best route
    /// leads the next.
    private func best(_ e: CapacityMap.Entry, _ v: CM.View, heat: [String: Double] = [:]) -> (route: String, lead: Double) {
        let t = v.routes.map { r in seconds(e, e.routes[r].id).map { $0 * CM.Heat(heat).factor(e.routes[r]) + v.extra(r) } ?? .infinity }
        let ratios: [Double?] = t.indices.map { t[$0].isFinite ? t[$0] / t[0] : nil }
        let chosen = CM.rule(ratios: ratios, efficient: v.routes.map { e.routes[$0].efficient })!
        let others = t.indices.filter { $0 != chosen }.map { t[$0] }
        return (e.routes[v.routes[chosen]].id, (others.min() ?? .infinity) / t[chosen])
    }

    func testTheSimulatedMacGetsItsTableInsideTheBudget() {
        let g = grid
        var settings = CM.Settings(); settings.budget = 60
        let (plan, sim) = simulate(g, settings: settings)
        print("MAPSIM spent \(plan.spent) residual \(plan.residualLoss) statuses \(Dictionary(grouping: plan.decisions, by: \.status).mapValues(\.count)) runs \(plan.decisions.map(\.runs).reduce(0, +))")
        XCTAssertLessThanOrEqual(plan.spent, 60 + 1, "the budget holds (one step over at most)")
        XCTAssertEqual(plan.spent, sim.t, accuracy: 1e-9)
        let all = decisions(g)
        XCTAssertEqual(plan.decisions.map(\.classID), all.map(\.view.id), "one decision per view, in order")
        // The decisions get the route the rule picks from the true times where it leads by more than the noise.
        var checked = 0
        for ((e, v), d) in zip(all, plan.decisions) {
            let (route, lead) = best(e, v)
            guard lead > 1.05 else { continue }
            XCTAssertEqual(d.chosen, route, v.id)
            checked += 1
        }
        XCTAssertGreaterThan(checked, all.count / 2)
        XCTAssertLessThan(plan.residualLoss, CM.Settings().epsilon)
        // Deterministic: the same machine, the same plan.
        XCTAssertEqual(plan, simulate(g, settings: settings).0)
    }

    func testAWeightsFormChangesTheDecisionNotTheMeasurement() {
        // At 1 024 rows of 8192², the 8-bit weight copied and multiplied beside the AMX beats every widened route, the bf16 one
        // does not: one measurement, two routes.
        let g = part
        let (plan, _) = simulate(g)
        let bf16 = plan.decisions.first { $0.classID == "product:8192x8192@1024" }!
        let int8 = plan.decisions.first { $0.classID == "product:8192x8192@1024~int8" }!
        XCTAssertEqual(int8.chosen, "gpu+amx/int8")
        XCTAssertFalse(bf16.chosen!.contains("/"), bf16.chosen!)
        XCTAssertEqual(bf16.runs, int8.runs, "the same pairs")
    }

    func testTheTableMakesTheEngineTakeTheChosenRoutes() {
        let g = part
        let (plan, _) = simulate(g)
        let table = CapacityMap.table(plan: plan, entries: g, classes: classes(g), cuts: g.map { _ in .init() })
        let all = decisions(g)
        func chosen(_ id: String) -> String { plan.decisions.first { $0.classID == id }!.chosen! }
        for (e, v) in all {
            guard case let .product(k, n, m) = e.work, let kind = RouteTable.ClassKey.parse(v.id)?.kind else { continue }
            // One engine rule per point, its bf16 view's: the engines are the same hardware for every form.
            XCTAssertEqual(table.products.route(k: k, n: n, m: m).engines.rawValue, String(chosen(e.id).split(separator: "/")[0]), v.id)
            // One form per weight at every height of its shape: the tallest's.
            let top = g.compactMap { if case let .product(k2, n2, m2) = $0.work, k2 == k, n2 == n { return m2 } else { return nil } }.max()!
            let tall = chosen(CapacityMap.Entry.id(.product(k: k, n: n, m: top)) + RouteTable.ClassKey.suffix(kind))
            let path = table.products.weight(kind, k: k, n: n, m: m).path
            switch CapacityMap.Entry.form(tall) {
            case nil: XCTAssertEqual(path, .widened, v.id)
            case "bf16", "bf16act": XCTAssertEqual(path, .bf16, v.id)
            default: XCTAssertEqual(path, .native, v.id)
            }
        }
        // A model's shape reads its nearest point: Z-Image's 3840×10240 at 1 056 rows is 4096×16384 at 1 024.
        XCTAssertEqual(table.products.route(k: 3840, n: 10240, m: 1056).engines.rawValue,
                       String(chosen("product:4096x16384@1024").split(separator: "/")[0]))
        XCTAssertEqual(table.products.weight(.int8, k: 3840, n: 10240, m: 1056),
                       table.products.weight(.int8, k: 4096, n: 16384, m: 1024))
        // A shape no point reaches is the simplest route.
        XCTAssertEqual(table.products.route(k: 7, n: 7, m: 4096).aneRows, 0)
    }

    func testAClassThatChoseTheANEKnowsItsRouteWithoutTheANE() {
        let g = part
        let (plan, _) = simulate(g)
        let table = CapacityMap.table(plan: plan, entries: g, classes: classes(g), cuts: [])
        var seen = 0
        for d in plan.decisions where d.chosen?.hasPrefix("gpu+ane") == true {
            guard let (key, kind) = RouteTable.ClassKey.parse(d.classID), kind == .bf16, case let .product(k, n, m) = key else { continue }
            XCTAssertNotEqual(table.products.forced(k: k, n: n, m: m, aneBroughtUp: false), .ane, d.classID)
            XCTAssertNotEqual(table.products.forced(k: k, n: n, m: m, aneBroughtUp: false), .aneAMX, d.classID)
            XCTAssertEqual(table.products.forced(k: k, n: n, m: m), table.products.classes.first { $0.matches(k: k, n: n, m: m) }?.engines)
            seen += 1
        }
        XCTAssertGreaterThan(seen, 0)
    }

    func testAHotAMXLosesItsRoutes() {
        // The end witnesses say the AMX takes 2.06× longer hot (the Neo's): the table is decided there.
        let g = part
        let (cold, _) = simulate(g)
        let (hot, _) = simulate(g, finalHeat: ["amx": 2.06])
        XCTAssertTrue(cold.decisions.contains { $0.chosen?.contains("amx") == true })
        for ((e, v), d) in zip(decisions(g), hot.decisions) {
            let (route, lead) = best(e, v, heat: ["amx": 2.06])
            guard lead > 1.05 else { continue }
            XCTAssertEqual(d.chosen, route, v.id)
            XCTAssertFalse(d.chosen?.contains("amx") == true, "\(v.id): \(d.chosen ?? "-")")
        }
    }

    func testADriftingMachineGivesTheSameTable() {
        // The whole Mac slows ×1.8 during the map: the pairs cancel it, the table does not move.
        let g = part
        let steady = simulate(g).0
        let drift = simulate(g, drift: { 1 + 0.8 * min($0, 15) / 15 }).0
        for (((e, v), a), b) in zip(zip(decisions(g), steady.decisions), drift.decisions) where best(e, v).lead > 1.05 {
            XCTAssertEqual(a.chosen, b.chosen, v.id)
        }
    }

    func testThePlanSaysWhatItLeaves() {
        var tight = CM.Settings(); tight.budget = 3
        let (plan, _) = simulate(grid, settings: tight)
        XCTAssertTrue(plan.exhausted)
        XCTAssertGreaterThan(plan.residualLoss, CM.Settings().epsilon, "three seconds do not decide the grid")
        XCTAssertTrue(plan.decisions.contains { $0.status == .open || $0.status == .unmeasured })
    }

    // MARK: the regime

    func testAWitnessIsHeatOnlyBeyondItsOwnNoise() {
        // A quiet M1 Pro: ×1.02 with 2 % of error each: noise. The Neo's AMX: ×2.06: heat.
        let quiet = CapacityMap.EngineWitness(engine: "gpu", before: 1, after: 1.02, errorBefore: 0.02, errorAfter: 0.02)
        XCTAssertEqual(quiet.factor, 1)
        let hot = CapacityMap.EngineWitness(engine: "amx", before: 1, after: 2.06, errorBefore: 0.02, errorAfter: 0.02)
        XCTAssertEqual(hot.factor, 2.06, accuracy: 1e-12)
        let faster = CapacityMap.EngineWitness(engine: "ane", before: 1, after: 0.7, errorBefore: 0.01, errorAfter: 0.01)
        XCTAssertEqual(faster.factor, 0.7, accuracy: 1e-12, "a warm-up that speeds an engine up counts as well")
        let heat = CapacityMap.heat([quiet, hot, faster])
        XCTAssertEqual(heat.factors, ["gpu": 1, "amx": 2.06, "ane": 0.7])
        // A route runs at the speed of its slowest engine.
        XCTAssertEqual(heat.factor(.init("gpu+ane+amx", engines: ["gpu", "ane", "amx"])), 2.06, accuracy: 1e-12)
        XCTAssertEqual(CapacityMap.witness([hot], span: 50).drift["amx"] ?? 0, log(2.06), accuracy: 1e-12)
    }

    // MARK: green and orange

    func testVerdict() {
        func v(loss: Double = 0.01, statuses: [CM.Status] = [.closed], wrong: Bool = false, held: Bool = true,
               seconds: Double = 58) -> CapacityMap.Verdict {
            .init(residualLoss: loss, epsilon: 0.02, statuses: statuses, wrongChosen: wrong, memoryHeld: held, seconds: seconds)
        }
        XCTAssertTrue(v().green)
        XCTAssertEqual(v(loss: 0.05).reasons, ["loss"])
        XCTAssertEqual(v(statuses: [.closed, .none]).reasons, ["none"])
        XCTAssertEqual(v(wrong: true).reasons, ["wrong"])
        XCTAssertEqual(v(held: false).reasons, ["memory"])
        XCTAssertEqual(v(seconds: 75).reasons, ["time"])
        XCTAssertEqual(v(loss: 0.5, held: false).reasons, ["loss", "memory"])
        // Open classes are not a reason of their own: the loss they leave is.
        XCTAssertTrue(v(statuses: [.open, .unmeasured, .closed]).green)
    }

    func testGreenDoesNotDependOnTheHour() {
        // Heat, battery, Low Power Mode and swap are facts of the regime, never a motive.
        let r = report(green: true, thermal: "serious", lowPower: true, battery: true, swap: 3_000_000_000)
        XCTAssertTrue(r.isGreen)
        XCTAssertFalse(report(green: false).isGreen)
    }

    func testMemoryHeld() {
        let plan = Diagnostic.Plan(budget: 4 << 30, base: 100 << 20, aneAllotment: 0, need: .init(steps: ["routes": 3 << 30, "widening": 1 << 29]))
        XCTAssertTrue(Diagnostic.Memory(plan: plan, peaks: ["routes": (3 << 30) + (100 << 20), "widening": 1 << 29]).held)
        XCTAssertFalse(Diagnostic.Memory(plan: plan, peaks: ["routes": (3 << 30) + (200 << 20)]).held, "a peak over base + need is a defect")
        let over = Diagnostic.Plan(budget: 3 << 30, base: 100 << 20, aneAllotment: 0, need: .init(steps: ["routes": 3 << 30]))
        XCTAssertFalse(Diagnostic.Memory(plan: over, peaks: [:]).held, "a need over B")
    }

    func testTheMapsNeedFitsTheNeoOnlyByCuttingTheANE() throws {
        let c = grid
        let big = CapacityMap.need(entries: c, allotment: nil).steps["routes"]!
        let small = CapacityMap.need(entries: c, allotment: 0).steps["routes"]!
        XCTAssertGreaterThan(big, small)
        XCTAssertEqual(Set(CapacityMap.need(entries: c, allotment: nil).steps.keys), Set(CapacityMap.needSteps))
        let gib = 1 << 30
        let plan = try Diagnostic.Plan.choose(budget: 10 * gib, base: 16 << 20) { CapacityMap.need(entries: c, allotment: $0) }
        XCTAssertLessThanOrEqual(plan.base + plan.need.peak, plan.budget)
        print("MAPNEED whole \(big >> 20) MiB, cut \(small >> 20) MiB, plan at 10 GiB allotment \(String(describing: plan.aneAllotment))")
        // An 8 GB Mac: B = 4 GiB. Either it fits with the engines cut, or it is refused before the first product.
        do {
            let neo = try Diagnostic.Plan.choose(budget: 4 * gib, base: 600 << 20) { CapacityMap.need(entries: c, allotment: $0) }
            XCTAssertLessThanOrEqual(neo.base + neo.need.peak, neo.budget)
        } catch { XCTAssertEqual((error as? EngineError)?.code, EngineError.insufficientMemory(needed: 0, budget: 0).code) }
    }

    // MARK: the report of schema 14

    nonisolated(unsafe) private static var base: Diagnostic.Report?

    private func report(green: Bool, thermal: String = "nominal", lowPower: Bool = false, battery: Bool? = false, swap: Int = 0,
                        classes: Int = 0) -> Diagnostic.Report {
        var r = Self.base ?? makeReport()
        Self.base = r
        r.after = Diagnostic.Conditions(thermal: thermal, lowPowerMode: lowPower, onBattery: battery, swapUsedBytes: swap)
        r.verdict = .init(green: green, reasons: green ? [] : ["loss"])
        return r
    }

    private func makeReport() -> Diagnostic.Report {
                let machine = Diagnostic.Machine(chip: "Apple M1 Pro", hardwareModel: "MacBookPro18,3", performanceCores: 8, efficiencyCores: 2,
                                         gpuCores: 16, memoryBytes: 17_179_869_184, macOS: "27.0.1 (26A434)", version: "0.17 (170)", build: "x+y",
                                         gpu: "Apple M1 Pro", gpuFamily: "apple7", gpuWorkingSetBytes: 11_453_251_584)
        let before = Diagnostic.Conditions(thermal: "nominal", lowPowerMode: false, onBattery: false, swapUsedBytes: 0)
        let after = before
        let g = grid
        let (plan, _) = simulate(g)
        let table = CapacityMap.table(plan: plan, entries: g, classes: classes(g), cuts: [])
        let decided = classes(g).flatMap(\.decisionClasses)
        let total = decided.reduce(0) { $0 + $1.weight }
        let mapClasses = zip(decided, plan.decisions).map { c, d in
            Diagnostic.MapClass(id: c.id, weight: c.weight / total, chosen: d.chosen, loss: d.loss, runs: d.runs, status: d.status,
                                timings: c.routes.indices.dropFirst().compactMap { r in
                                    d.hotRatios[r].map { .init(route: c.routes[r].id, ratio: $0, error: d.spread[r] ?? 0) } })
        }
        let map = Diagnostic.MapResult(classes: mapClasses, residualLoss: plan.residualLoss, epsilon: 0.02, spent: 51.2, budget: 52.4,
                                       exhausted: plan.exhausted, judged: 3.1, wrong: [])
        let w = [CapacityMap.EngineWitness(engine: "gpu", before: 0.0123, after: 0.0125, errorBefore: 0.01, errorAfter: 0.01),
                 CapacityMap.EngineWitness(engine: "amx", before: 0.031, after: 0.032, errorBefore: 0.01, errorAfter: 0.01),
                 CapacityMap.EngineWitness(engine: "ane", before: 0.019, after: 0.0191, errorBefore: 0.01, errorAfter: 0.01)]
        return Diagnostic.Report(date: "2026-10-29T10:00:00Z", seconds: 58.3, machine: machine, cpuFeatures: ["FEAT_BF16": 1],
                                 before: before, after: after, products: [], lowPrecision: [], attention: [],
                                 widening: [], witness: .init(gpuBefore: 3.4, gpuAfter: 3.3, amxBefore: 1.9, amxAfter: 1.9),
                                 stages: ["widening": 0.2, "witness": 0.9, "routes": 54.1, "regime": 0.9, "judge": 3.1],
                                 memory: .init(plan: .init(budget: 10 << 30, base: 16 << 20, aneAllotment: nil, need: .init(steps: ["routes": 4 << 30, "widening": 1 << 29])),
                                               peaks: ["routes": 3 << 30, "widening": 1 << 28], aneBatches: 2),
                                 routes: table, map: map, regime: .init(witnesses: w),
                                 verdict: .init(green: true, reasons: []))
    }

    func testReportOfSchema14RoundTripsAndKeepsItsLinkShort() throws {
        let r = report(green: true)
        XCTAssertEqual(r.schema, 14)
        XCTAssertNil(r.decisions, "the rules are for the older reports")
        let back = try JSONDecoder().decode(Diagnostic.Report.self, from: r.json())
        XCTAssertEqual(back, r)
        XCTAssertEqual(try JSONDecoder().decode(Diagnostic.Report.self, from: r.json(pretty: false)), r)
        let link = try r.issueURL().absoluteString.utf8.count
        let json = try JSONSerialization.jsonObject(with: r.json(pretty: false)) as! [String: Any]
        print("MAPSIZES " + json.map { "\($0.key) \((try? JSONSerialization.data(withJSONObject: $0.value, options: [.fragmentsAllowed]).count) ?? 0)" }.sorted().joined(separator: " · "))
        print("MAPLINK \(link) bytes, report \(try r.json(pretty: false).count) bytes, table \(try JSONEncoder().encode(r.routes!).count) bytes")
        XCTAssertLessThan(link, 8000, "the link carries the report below the length browsers and GitHub keep")
        XCTAssertFalse(try r.issueURL().absoluteString.contains("minRows"), "the table is not in the link")
        XCTAssertTrue(r.passes)
    }

    func testAnOlderReportIsNotRead() {
        var old = report(green: true)
        old.schema = 13
        old.routes = nil; old.map = nil; old.verdict = nil
        XCTAssertFalse(old.isCurrent)
        XCTAssertFalse(old.isGreen)
        var current = report(green: true)
        current.schema = Diagnostic.schema
        XCTAssertEqual(current.isCurrent, current.fingerprint == .current)
    }

    func testTheEngineReadsTheTableOfASchema14Map() {
        let r = report(green: true)
        let read = EngineSettings(applying: [:], map: r)
        XCTAssertEqual(read.routes, r.routes)
        XCTAssertEqual(read.amx, r.routes!.products.usesAMX)
        XCTAssertEqual(read.gemmANE, r.routes!.products.usesANE)
        XCTAssertEqual(read.flash, r.routes!.attention.usesFlash)
        XCTAssertEqual(read.provenance["amx"], .map)
        // Without a table, the switches' (the defaults here).
        XCTAssertEqual(EngineSettings(applying: [:]).routes, RouteTable(settings: EngineSettings(applying: [:])))
    }

    // MARK: the pieces

    func testAFailedWeightFormFallsBackToTheNextRule() {
        // Int8 beside the AMX chosen for one class: engines and weight rule, in the table.
        let classes: [CM.ComputeClass] = [.init("product:4096x12288@4096~int8", weight: 1,
                                                routes: CapacityMap.productRoutes(m: 4096, caps: all))]
        let routes = classes[0].routes.map(\.id)
        let plan = CM.Plan(decisions: [.init(classID: classes[0].id, chosen: "gpu+amx/int8", status: .closed, loss: 0, hotRatios: routes.map { _ in 1 }, runs: 6)],
                           residualLoss: 0, spent: 1, exhausted: false)
        let t = RouteTable(plan: plan, classes: classes, base: RouteTable())
        XCTAssertEqual(t.products.route(k: 4096, n: 12288, m: 4096).engines, .amx)
        XCTAssertEqual(t.products.weight(.int8, k: 4096, n: 12288, m: 4096).path, .native)
        XCTAssertEqual(t.products.weight(.bf16, k: 4096, n: 12288, m: 4096).path, .widened, "the kind is part of the rule")
        XCTAssertTrue(t.products.usesInt8Weights)
        XCTAssertTrue(t.products.usesAMX)
        XCTAssertFalse(t.products.usesANE)
    }

    func testTheClassRuleCodableKeepsItsFallback() throws {
        let rule = RouteTable.ClassRule(engines: .ane, k: 1, n: 2, minRows: 3, maxRows: 3, withoutANE: .amx)
        XCTAssertEqual(try JSONDecoder().decode(RouteTable.ClassRule.self, from: JSONEncoder().encode(rule)), rule)
        let old = Data(#"{"engines":"gpu+ane","minRows":0}"#.utf8)
        XCTAssertNil(try JSONDecoder().decode(RouteTable.ClassRule.self, from: old).withoutANE)
    }

    // MARK: the core's two new inputs

    /// Routes gpu / amx (simple) / ane (efficient): the AMX hot-doubles at the end of the map.
    private func tinyClasses() -> [CM.ComputeClass] {
        [.init("a", weight: 1, routes: [.init("gpu", engines: ["gpu"]), .init("gpu+amx", engines: ["gpu", "amx"]),
                                        .init("gpu+ane", efficient: true, engines: ["gpu", "ane"])])]
    }

    func testTheHeatIsAskedOnceWhenTheAllocationIsOver() {
        let base: [[Double?]] = [[0.1, 0.07, nil]]
        var asked = 0
        let sim = CapacityMeasureTests.Sim(base: base, noise: 0.01)
        let plan = CM.plan(classes: tinyClasses(), finalHeat: { asked += 1; return CM.Heat(["amx": 2.0]) }, clock: sim) { sim.probe($0, $1) }
        XCTAssertEqual(asked, 1)
        XCTAssertEqual(plan.decisions[0].chosen, "gpu", "decided at the hot regime: the AMX's 0.7 became 1.4")
        XCTAssertEqual(plan.decisions[0].hotRatios[1]!, 1.4, accuracy: 0.05)
        let sim2 = CapacityMeasureTests.Sim(base: base, noise: 0.01)
        let cold = CM.plan(classes: tinyClasses(), clock: sim2) { sim2.probe($0, $1) }
        XCTAssertEqual(cold.decisions[0].chosen, "gpu+amx")
        XCTAssertEqual(cold.decisions[0].spread.count, 3)
        XCTAssertNil(cold.decisions[0].spread[2], "a hole has no dispersion")
    }

    func testAnEfficientRouteKeptAgainstAFasterEfficientOneIsNoLoss() {
        // ANE 5.5, the three 5.8 TFLOP/s: within 90 %, the rule keeps the pair — and the plan stops measuring for it.
        let routes: [CM.Route] = [.init("gpu", engines: ["gpu"]), .init("gpu+ane", efficient: true, engines: ["gpu", "ane"]),
                                  .init("gpu+ane+amx", efficient: true, engines: ["gpu", "ane", "amx"])]
        let sim = CapacityMeasureTests.Sim(base: [[0.4, 0.4 / 1.375, 0.4 / 1.45]], noise: 0.005, seed: 3)
        let plan = CM.plan(classes: [.init("a", weight: 1, routes: routes)], clock: sim) { sim.probe($0, $1) }
        XCTAssertEqual(plan.decisions[0].chosen, "gpu+ane")
        XCTAssertLessThan(plan.decisions[0].loss, CM.Settings().epsilon)
        XCTAssertLessThan(plan.decisions[0].runs, 200, "it was 805 runs of a minute's budget before")
    }
}
