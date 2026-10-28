import Foundation
import XCTest
@testable import Siliconed

/// **The capacity map end to end, without a machine** (`CapacityMapClasses.swift`, `CapacityMeasure.plan`, `RouteTable(plan:…)`,
/// `Diagnostic.Report` of schema 13): the catalogue's weights and clusters, a simulated Mac whose engines have known rates
/// producing the right table inside its budget, the regime read from the witnesses, green and orange.
final class CapacityMapTests: XCTestCase {
    typealias CM = CapacityMeasure
    typealias CC = CapacityClasses

    private let all = CapacityMap.Capabilities(amx: true, ane: true, bf16Weights: true, bf16Rounding: true, int8Weights: true,
                                               q8Weights: true, flash: true, headSplit: true)

    /// Building the catalogue reads the five families' configs five-and-twenty times: once per size.
    nonisolated(unsafe) private static var catalogues: [Int: CapacityMap.Catalogue] = [:]
    private func cat(_ maxEntries: Int) -> CapacityMap.Catalogue {
        if let c = Self.catalogues[maxEntries] { return c }
        let c = CapacityMap.catalogue(caps: all, maxEntries: maxEntries)
        Self.catalogues[maxEntries] = c
        return c
    }

    // MARK: the catalogue

    func testWeightIsTheMaximumOfTheShares() {
        // Z-Image and Qwen-Image-2.1 apart, then together: a family is never diluted — the class that measures for one of
        // its classes weighs at least what that class weighs for it alone.
        let z = CC.Request(family: .zImage), q = CC.Request(family: .qwenImage21)
        let together = CapacityMap.catalogue(requests: [z, q], caps: all, coverage: 2, maxEntries: 1000)
        func near(_ a: CapacityMap.Work, _ b: CapacityMap.Work) -> Bool {
            let r = log(CapacityMap.shapeRatio)
            func close(_ x: Int, _ y: Int) -> Bool { abs(log(Double(x) / Double(y))) <= r }
            switch (a, b) {
            case let (.product(k, n, m, kind), .product(k2, n2, m2, kind2)):
                return kind == kind2 && close(k, k2) && close(n, n2) && abs(m - m2) <= CapacityMap.heightGap
            case let (.attention(s, h, d), .attention(s2, h2, d2)):
                return close(h, h2) && close(d, d2) && abs(s - s2) <= CapacityMap.heightGap
            default: return false
            }
        }
        for family in [z, q] {
            let alone = CapacityMap.catalogue(requests: [family], caps: all, coverage: 2, maxEntries: 1000)
            for e in alone.entries where e.weight > 1e-3 {
                let standing = together.entries.filter { near($0.work, e.work) }
                XCTAssertFalse(standing.isEmpty, e.id)
                XCTAssertGreaterThanOrEqual(standing.map(\.weight).max() ?? 0, e.weight * 0.999, "\(e.id): not diluted")
            }
        }
        // Z-Image's and Qwen's neighbouring shapes are one measurement: 3840×10240 at 4 128 rows and 4096×12288 at 4 096.
        XCTAssertLessThan(together.entries.count,
                          CapacityMap.catalogue(requests: [z], caps: all, coverage: 2, maxEntries: 1000).entries.count
                            + CapacityMap.catalogue(requests: [q], caps: all, coverage: 2, maxEntries: 1000).entries.count)
        // The shares of a request sum to 1: the maximum over requests of a class is at most 1.
        XCTAssertTrue(together.entries.allSatisfy { $0.weight <= 1.000001 })
    }

    func testHeightsOfOneShapeAreOneMeasurement() {
        let c = CapacityMap.catalogue(requests: [.init(family: .zImage)], caps: all, coverage: 2, maxEntries: 1000)
        // Z-Image 1024²: 4096 and 4128 rows of the same k×n: one entry, both heights named.
        let entries = c.entries.filter { if case .product(3840, 3840, _, .bf16) = $0.work { return true } else { return false } }
        let tall = entries.filter { $0.heights.contains(4128) }
        XCTAssertEqual(tall.count, 1)
        XCTAssertTrue(tall[0].heights.contains(4096))
        XCTAssertEqual(Set(entries.map(\.id)).count, entries.count)
    }

    func testTheCutAndTheTail() {
        let c = cat(40)
        XCTAssertLessThanOrEqual(c.entries.count, CapacityMap.maxEntries)
        XCTAssertGreaterThan(c.entries.count, 15)
        XCTAssertEqual(c.entries.map(\.weight), c.entries.map(\.weight).sorted(by: >), "heaviest first")
        XCTAssertGreaterThan(c.tailShare, 0)
        XCTAssertEqual(c.tailShare + c.entries.reduce(0) { $0 + $1.weight } / c.total, 1, accuracy: 1e-12)
        XCTAssertEqual(Set(c.entries.map(\.id)).count, c.entries.count, "an id names one class")
        // The whole catalogue is kept to complete the table over: more classes than the entries.
        XCTAssertGreaterThan(c.render.count, c.entries.count)
        XCTAssertEqual(c, CapacityMap.catalogue(caps: all), "deterministic")
    }

    func testKindsAreInTheIdAndLightFoldsIntoBF16() {
        let c = CapacityMap.catalogue(requests: [.init(family: .qwenImage21, variant: .compact), .init(family: .qwenImage21, variant: .light)],
                                      caps: all, coverage: 2, maxEntries: 1000)
        XCTAssertTrue(c.entries.contains { $0.id.hasSuffix("~int8") })
        XCTAssertTrue(c.entries.contains { !$0.id.contains("~") && $0.id.hasPrefix("product:") }, "the K-quants widen: bf16's id")
        for e in c.entries { XCTAssertNotNil(RouteTable.ClassKey.parse(e.id), e.id) }
        let p = RouteTable.ClassKey.parse("product:4096x12288@4096~int8")
        XCTAssertEqual(p?.kind, .int8)
        XCTAssertEqual(p?.key, .product(k: 4096, n: 12288, m: 4096))
    }

    func testRoutesFollowTheMachineAndTheShape() {
        let none = CapacityMap.Capabilities()
        XCTAssertEqual(CapacityMap.productRoutes(kind: .bf16, m: 4096, denoising: true, caps: none).map(\.id), ["gpu"])
        let ids = { (m: Int, d: Bool, k: RouteTable.WeightKind) in
            CapacityMap.productRoutes(kind: k, m: m, denoising: d, caps: self.all).map(\.id) }
        XCTAssertEqual(ids(4096, true, .bf16), ["gpu", "gpu+amx", "gpu+ane", "gpu+ane+amx", "gpu/bf16", "gpu/bf16act"])
        XCTAssertEqual(ids(4096, false, .bf16), ["gpu", "gpu+amx", "gpu/bf16", "gpu/bf16act"], "the ANE serves the denoisers only")
        XCTAssertEqual(ids(300, true, .int8), ["gpu", "gpu/int8"], "no AMX under 512 rows")
        XCTAssertEqual(ids(1024, true, .q8), ["gpu", "gpu+amx", "gpu+ane", "gpu+ane+amx", "gpu/q8", "gpu+amx/q8"])
        let route = CapacityMap.productRoutes(kind: .bf16, m: 4096, denoising: true, caps: all)
        XCTAssertEqual(route.filter(\.efficient).map(\.id), ["gpu+ane", "gpu+ane+amx"])
        XCTAssertEqual(CapacityClasses.engines(ofRoute: "gpu+amx/int8"), ["gpu", "amx"])
        XCTAssertEqual(CapacityMap.attentionRoutes(sequence: 512, caps: all).map(\.id), ["graph", "flash"])
        XCTAssertEqual(CapacityMap.attentionRoutes(sequence: 4096, caps: all).map(\.id), ["graph", "flash", "split"])
    }

    // MARK: a simulated Mac

    /// The engines of a fictional Mac: TFLOP/s by route. The AMX helps everywhere, the ANE from 20 M weights, the three from 50 M,
    /// the bf16 weight loses, the 8-bit ones win; flash and the split win on long sequences.
    private func seconds(_ e: CapacityMap.Entry, _ route: String, heat: [String: Double] = [:]) -> Double? {
        var rate: Double
        switch e.work {
        case let .product(k, n, m, _):
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

    /// A quarter of the minute is plenty for fourteen simulated classes, and keeps the suite short.
    private static var quarter: CM.Settings { var s = CM.Settings(); s.budget = 15; return s }

    private func simulate(_ catalogue: CapacityMap.Catalogue, heat: [String: Double] = [:], noise: Double = 0.02,
                          settings: CM.Settings = CapacityMapTests.quarter, finalHeat: [String: Double] = [:]) -> (CM.Plan, CapacityMeasureTests.Sim) {
        let base = catalogue.entries.map { e in e.routes.map { seconds(e, $0.id, heat: heat) } }
        let sim = CapacityMeasureTests.Sim(base: base, noise: noise)
        let plan = CM.plan(classes: catalogue.entries.map(\.compute), settings: settings, extraWeight: catalogue.tail,
                           finalHeat: finalHeat.isEmpty ? nil : { CM.Heat(finalHeat) }, clock: sim) { sim.probe($0, $1) }
        return (plan, sim)
    }

    /// What the rule chooses from the true hot times.
    private func best(_ e: CapacityMap.Entry, heat: [String: Double] = [:]) -> String {
        let t = e.routes.map { seconds(e, $0.id) ?? .infinity }
        let ratios: [Double?] = e.routes.indices.map { r in t[r].isFinite ? t[r] * (CM.Heat(heat).factor(e.routes[r])) / t[0] : nil }
        return e.routes[CM.rule(ratios: ratios, efficient: e.routes.map(\.efficient))!].id
    }

    func testTheSimulatedMacGetsItsTableInsideTheBudget() {
        let c = cat(14)
        var settings = CM.Settings(); settings.budget = 60
        let (plan, sim) = simulate(c, settings: settings)
        print("MAPSIM spent \(plan.spent) residual \(plan.residualLoss) statuses \(Dictionary(grouping: plan.decisions, by: \.status).mapValues(\.count)) runs \(plan.decisions.map(\.runs).reduce(0, +)) tail \(c.tailShare)")
        print("MAPSIM " + zip(c.entries, plan.decisions).prefix(12).map { "\($0.id) w\(String(format: "%.3f", $0.weight / c.total)) \($1.status) loss \(String(format: "%.5f", $1.loss)) runs \($1.runs)" }.joined(separator: "\n MAPSIM "))
        XCTAssertLessThanOrEqual(plan.spent, 60 + 1, "the budget holds (one step over at most)")
        XCTAssertEqual(plan.spent, sim.t, accuracy: 1e-9)
        // The heavy classes get the route the rule picks from the true times.
        let total = c.total
        var checked = 0
        for (e, d) in zip(c.entries, plan.decisions).prefix(6) {
            XCTAssertEqual(d.chosen, best(e), e.id)
            checked += 1
        }
        XCTAssertGreaterThan(total, 0)
        XCTAssertEqual(checked, 6)
        // The loss that stays is a fraction of the render under ε, though every class is "open" (a route not yet at its three
        // ratios): the seconds went where the weight is.
        XCTAssertLessThan(plan.residualLoss, CM.Settings().epsilon)
        // And no class burnt the budget alone: an efficient route kept against a faster efficient one is no loss.
        XCTAssertLessThan(plan.decisions.map(\.runs).max() ?? 0, 400)
        // Deterministic: the same machine, the same plan.
        XCTAssertEqual(plan, simulate(c, settings: settings).0)
    }

    func testTheTableMakesTheEngineTakeTheChosenRoutes() {
        let c = cat(14)
        let (plan, _) = simulate(c)
        let table = CapacityMap.table(plan: plan, catalogue: c, cuts: c.entries.map { _ in .init() })
        var named = Set<[Int]>()
        for (e, d) in zip(c.entries, plan.decisions) {
            guard case let .product(k, n, m, kind) = e.work, let chosen = d.chosen else { continue }
            // One engine rule per height, the heaviest kind's: another kind of the same k×n@m shares its engines.
            guard named.insert([k, n, m]).inserted else { continue }
            let engines = chosen.split(separator: "/")[0]
            XCTAssertEqual(table.products.route(k: k, n: n, m: m).engines.rawValue, String(engines), e.id)
            let form = chosen.split(separator: "/").count > 1 ? String(chosen.split(separator: "/")[1]) : nil
            let path = table.products.weight(kind, k: k, n: n, m: m).path
            switch form {
            case nil: XCTAssertEqual(path, .widened, e.id)
            case "bf16", "bf16act": XCTAssertEqual(path, .bf16, e.id)
            default: XCTAssertEqual(path, .native, e.id)
            }
        }
        // A class nobody measured at a height its cluster names takes its neighbour's route, the ANE falling back to the AMX's.
        let tall = c.entries.first { $0.heights.count > 1 && $0.routes.count > 2 }!
        guard case let .product(k, n, _, _) = tall.work else { return XCTFail() }
        let chosen = plan.decisions[c.entries.firstIndex(of: tall)!].chosen!
        for h in tall.heights.dropFirst() {
            XCTAssertEqual(table.products.route(k: k, n: n, m: h).engines.rawValue, String(chosen.split(separator: "/")[0]), "\(tall.id) @ \(h)")
        }
        // A shape no measured class reaches is the simplest route.
        XCTAssertEqual(table.products.route(k: 7, n: 7, m: 4096).aneRows, 0)
    }

    func testAClassThatChoseTheANEKnowsItsRouteWithoutTheANE() {
        let c = cat(14)
        let (plan, _) = simulate(c)
        let table = CapacityMap.table(plan: plan, catalogue: c, cuts: [])
        var seen = 0
        for (e, d) in zip(c.entries, plan.decisions) where d.chosen?.hasPrefix("gpu+ane") == true {
            guard case let .product(k, n, m, _) = e.work else { continue }
            XCTAssertNotEqual(table.products.forced(k: k, n: n, m: m, aneBroughtUp: false), .ane, e.id)
            XCTAssertNotEqual(table.products.forced(k: k, n: n, m: m, aneBroughtUp: false), .aneAMX, e.id)
            XCTAssertEqual(table.products.forced(k: k, n: n, m: m), table.products.classes.first { $0.matches(k: k, n: n, m: m) }?.engines)
            seen += 1
        }
        XCTAssertGreaterThan(seen, 0)
    }

    func testAHotAMXLosesItsRoutes() {
        // The end witnesses say the AMX takes 2.06× longer hot (the Neo's): the table is decided there.
        let c = cat(14)
        let (cold, _) = simulate(c)
        let (hot, _) = simulate(c, finalHeat: ["amx": 2.06])
        XCTAssertTrue(cold.decisions.contains { $0.chosen?.contains("amx") == true })
        let total = c.total
        for (e, d) in zip(c.entries, hot.decisions) where e.weight / total > 0.04 {
            XCTAssertEqual(d.chosen, best(e, heat: ["amx": 2.06]), e.id)
            XCTAssertFalse(d.chosen?.contains("amx") == true, "\(e.id): \(d.chosen ?? "-")")
        }
    }

    func testADriftingMachineGivesTheSameTable() {
        // The whole Mac slows ×1.8 during the map: the pairs cancel it, the table does not move.
        let c = cat(14)
        let steady = simulate(c, noise: 0.02).0
        let base = c.entries.map { e in e.routes.map { seconds(e, $0.id) } }
        let sim = CapacityMeasureTests.Sim(base: base, noise: 0.02, drift: { 1 + 0.8 * min($0, 60) / 60 })
        let drift = CM.plan(classes: c.entries.map(\.compute), settings: CapacityMapTests.quarter, extraWeight: c.tail, clock: sim) { sim.probe($0, $1) }
        let total = c.total
        for (i, e) in c.entries.enumerated() where e.weight / total > 0.04 {
            XCTAssertEqual(steady.decisions[i].chosen, drift.decisions[i].chosen, e.id)
        }
    }

    func testThePlanSaysWhatItLeaves() {
        let c = cat(14)
        var tight = CM.Settings(); tight.budget = 3
        let (plan, _) = simulate(c, settings: tight)
        XCTAssertTrue(plan.exhausted)
        XCTAssertGreaterThan(plan.residualLoss, CM.Settings().epsilon, "three seconds do not decide forty classes")
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
        func v(loss: Double = 0.01, tail: Double = 0.03, statuses: [CM.Status] = [.closed], wrong: Bool = false, held: Bool = true,
               seconds: Double = 58) -> CapacityMap.Verdict {
            .init(residualLoss: loss, epsilon: 0.02, tailShare: tail, statuses: statuses, wrongChosen: wrong, memoryHeld: held, seconds: seconds)
        }
        XCTAssertTrue(v().green)
        XCTAssertEqual(v(loss: 0.05).reasons, ["loss"])
        XCTAssertEqual(v(tail: 0.2).reasons, ["tail"])
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
        let c = cat(40)
        let big = CapacityMap.need(entries: c.entries, allotment: nil).steps["routes"]!
        let small = CapacityMap.need(entries: c.entries, allotment: 0).steps["routes"]!
        XCTAssertGreaterThan(big, small)
        XCTAssertEqual(Set(CapacityMap.need(entries: c.entries, allotment: nil).steps.keys), Set(CapacityMap.needSteps))
        let gib = 1 << 30
        let plan = try Diagnostic.Plan.choose(budget: 10 * gib, base: 16 << 20) { CapacityMap.need(entries: c.entries, allotment: $0) }
        XCTAssertLessThanOrEqual(plan.base + plan.need.peak, plan.budget)
        print("MAPNEED whole \(big >> 20) MiB, cut \(small >> 20) MiB, plan at 10 GiB allotment \(String(describing: plan.aneAllotment))")
        // An 8 GB Mac: B = 4 GiB. Either it fits with the engines cut, or it is refused before the first product.
        do {
            let neo = try Diagnostic.Plan.choose(budget: 4 * gib, base: 600 << 20) { CapacityMap.need(entries: c.entries, allotment: $0) }
            XCTAssertLessThanOrEqual(neo.base + neo.need.peak, neo.budget)
        } catch { XCTAssertEqual((error as? EngineError)?.code, EngineError.insufficientMemory(needed: 0, budget: 0).code) }
    }

    // MARK: the report of schema 13

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
        let c = cat(40)
        let (plan, _) = simulate(c)
        let table = CapacityMap.table(plan: plan, catalogue: c, cuts: [])
        let mapClasses = zip(c.entries, plan.decisions).prefix(40).map { e, d in
            Diagnostic.MapClass(id: e.id, weight: e.weight / c.total, chosen: d.chosen, loss: d.loss, runs: d.runs, status: d.status,
                                timings: e.routes.indices.dropFirst().compactMap { r in
                                    d.hotRatios[r].map { .init(route: e.routes[r].id, ratio: $0, error: d.spread[r] ?? 0) } })
        }
        let map = Diagnostic.MapResult(classes: Array(mapClasses), residualLoss: plan.residualLoss, epsilon: 0.02, spent: 51.2, budget: 52.4,
                                       exhausted: plan.exhausted, tail: c.tailShare, judged: 3.1, wrong: [])
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

    func testReportOfSchema13RoundTripsAndKeepsItsLinkShort() throws {
        let r = report(green: true)
        XCTAssertEqual(r.schema, 13)
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
        old.schema = 12
        old.routes = nil; old.map = nil; old.verdict = nil
        XCTAssertFalse(old.isCurrent)
        XCTAssertFalse(old.isGreen)
        var current = report(green: true)
        current.schema = Diagnostic.schema
        XCTAssertEqual(current.isCurrent, current.fingerprint == .current)
    }

    func testTheEngineReadsTheTableOfASchema13Map() {
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
                                                routes: CapacityMap.productRoutes(kind: .int8, m: 4096, denoising: true, caps: all))]
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

    func testWhatTheMapDoesNotDecideIsStillPartOfTheRender() {
        // The same measures, a class that weighs the same, and a tail as heavy as the class: its loss is half.
        let base: [[Double?]] = [[0.1, 0.099, nil]]
        func loss(_ tail: Double) -> Double {
            var s = CM.Settings(); s.budget = 2
            let sim = CapacityMeasureTests.Sim(base: base, noise: 0.03, seed: 7)
            return CM.plan(classes: tinyClasses(), settings: s, extraWeight: tail, clock: sim) { sim.probe($0, $1) }.residualLoss
        }
        XCTAssertEqual(loss(1), loss(0) / 2, accuracy: loss(0) * 0.05)
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

    func testAMapCoversTheFamiliesItWasMadeFor() {
        var r = report(green: true)
        XCTAssertFalse(r.covers(["z-image"]), "a map without families covers none")
        r.map?.families = ["z-image", "qwen-image-2.1"]
        XCTAssertTrue(r.covers(["z-image"]))
        XCTAssertTrue(r.covers([]))
        XCTAssertFalse(r.covers(["z-image", "krea2"]))
    }
}
