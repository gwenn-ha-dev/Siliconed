import Foundation
import XCTest
@testable import Siliconed

/// **The capacity map's measuring core** (`CapacityMeasure.swift`) on simulated engines: drift, noise, holes,
/// the budget, and the Neo's own rates. No GPU, no weights, no real clock.
final class CapacityMeasureTests: XCTestCase {
    typealias CM = CapacityMeasure

    /// Simulated machine: a clock the probes advance, noise drawn from (seed, class, route, call number) so
    /// that it does not depend on the order of the measurements, a machine-wide slowness `drift(t)`.
    final class Sim: CM.Clock {
        var t = 0.0
        let seed: UInt64
        let noise: Double
        let base: [[Double?]]            // cold seconds per [class][route]; nil: a hole
        let drift: (Double) -> Double
        var calls: [[Int]]

        init(base: [[Double?]], noise: Double = 0, seed: UInt64 = 1, drift: @escaping (Double) -> Double = { _ in 1 }) {
            self.base = base; self.noise = noise; self.seed = seed; self.drift = drift
            calls = base.map { $0.map { _ in 0 } }
        }

        func now() -> Double { t }

        static func normal(_ key: UInt64) -> Double {
            func mix(_ x: UInt64) -> UInt64 {
                var z = x &+ 0x9E37_79B9_7F4A_7C15
                z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
                z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
                return z ^ (z >> 31)
            }
            let u1 = (Double(mix(key) >> 11) + 1) / 9_007_199_254_740_993
            let u2 = Double(mix(key ^ 0xABCD_EF01) >> 11) / 9_007_199_254_740_992
            return (-2 * log(u1)).squareRoot() * cos(2 * Double.pi * u2)
        }

        func probe(_ c: Int, _ r: Int) -> Double? {
            guard let b = base[c][r] else { return nil }
            calls[c][r] += 1
            let key = seed &* 1_000_003 &+ UInt64(c) &* 10_007 &+ UInt64(r) &* 101 &+ UInt64(calls[c][r])
            // The run takes b · g(its midpoint): solve d = b · g(t + d/2) by fixed point.
            let n = exp(noise * Sim.normal(key))
            var d = b * n
            for _ in 0..<30 { d = b * n * drift(t + d / 2) }
            t += d
            return d
        }
    }

    private func plan(_ classes: [CM.ComputeClass], _ sim: Sim, heat: CM.Heat = CM.Heat(), witness: CM.Witness = CM.Witness(),
                      settings: CM.Settings = CM.Settings()) -> CM.Plan {
        CM.plan(classes: classes, heat: heat, witness: witness, settings: settings, clock: sim) { sim.probe($0, $1) }
    }

    private let gpu = CM.Route("gpu", engines: ["gpu"])
    private let amx = CM.Route("gpu+amx", engines: ["gpu", "amx"])
    private let ane = CM.Route("gpu+ane", efficient: true, engines: ["ane", "gpu"])

    private func classes(_ weights: [Double]) -> [CM.ComputeClass] {
        weights.enumerated().map { CM.ComputeClass("c\($0.offset)", weight: $0.element, routes: [gpu, amx, ane]) }
    }

    // MARK: 1. drift

    /// A machine that slows to ×0.5 (time ×2) during the map: the decisions are those of the steady machine,
    /// and today's rule (margin = the drift) would have turned off every gain under ×2.
    func testDriftLeavesDecisionsAlone() {
        // [gpu, amx, ane] seconds: AMX ×1.30 faster, ANE ×1.20 faster, AMX slower, ANE equal-ish, ...
        let base: [[Double?]] = [[0.10, 0.077, 0.095], [0.08, 0.10, 0.0667], [0.06, 0.065, 0.07], [0.05, 0.06, 0.07]]
        let cls = classes([0.4, 0.3, 0.2, 0.1])
        var settings = CM.Settings(); settings.budget = 300
        let steady = plan(cls, Sim(base: base, noise: 0.02), settings: settings)
        let horizon = steady.spent
        let linear = plan(cls, Sim(base: base, noise: 0.02, drift: { 1 + $0 / horizon }), settings: settings)
        let exponential = plan(cls, Sim(base: base, noise: 0.02, drift: { exp(log(2) * $0 / horizon) }), settings: settings)
        let chosen = steady.decisions.map(\.chosen)
        XCTAssertEqual(chosen, ["gpu+amx", "gpu+ane", "gpu", "gpu"])
        XCTAssertEqual(linear.decisions.map(\.chosen), chosen)
        XCTAssertEqual(exponential.decisions.map(\.chosen), chosen)
        // The drift really was ~×2 by the end of the exponential run's map, and today's margin would have beaten it.
        XCTAssertGreaterThan(exponential.spent, 0.8 * steady.spent, "the drift was real: runs took up to twice as long")
        let legacyMargin = 0.5, gainAMX = 0.10 / 0.077
        XCTAssertFalse(gainAMX > 1 + legacyMargin, "today's rule: the AMX's ×1.30 is turned off by a ×2 drift")
    }

    /// First-order cancellation of a linear drift: no noise, the ratio is the true one to (slope × run)².
    func testLinearDriftCancelsExactly() {
        let base: [[Double?]] = [[0.1, 0.05, nil]]
        let cls = [CM.ComputeClass("c", weight: 1, routes: [gpu, amx])]
        let sim = Sim(base: base, drift: { 1 + 0.2 * $0 })
        let p = plan(cls, sim)
        XCTAssertEqual(p.decisions[0].hotRatios[1]!, 0.5, accuracy: 2e-3)
    }

    // MARK: 2. noise

    func testNoiseKeepsDecisionsAndBoundsTheLossAtTies() {
        // Clear classes (gap ≥ 15 %) and two near-ties (0.5 % and 1.5 %).
        let base: [[Double?]] = [[0.1, 0.07, 0.13], [0.1, 0.12, 0.13], [0.1, 0.095, 0.13], [0.1, 0.1, 0.0995], [0.1, 0.0985, 0.13]]
        let cls = classes([0.3, 0.3, 0.1, 0.15, 0.15])
        var settings = CM.Settings(); settings.budget = 120
        for seed in UInt64(1)...40 {
            let p = plan(cls, Sim(base: base, noise: 0.03, seed: seed), settings: settings)
            let ch = p.decisions.map(\.chosen)
            XCTAssertEqual(ch[0], "gpu+amx", "seed \(seed)")
            XCTAssertEqual(ch[1], "gpu", "seed \(seed)")
            // Ties: any pick costs at most its weight × the true gap (≤ 1.5 %), well under ε.
            for (c, rivals) in [(2, [0.1, 0.095, 0.13]), (3, [0.1, 0.1, 0.0995]), (4, [0.1, 0.0985, 0.13])] {
                let i = ["gpu", "gpu+amx", "gpu+ane"].firstIndex(of: ch[c]!)!
                let loss = cls[c].weight / 1 * (rivals[i] / rivals.min()! - 1)
                XCTAssertLessThan(loss, 0.02, "seed \(seed)")
            }
            XCTAssertLessThan(p.residualLoss, 0.05, "seed \(seed)")
        }
    }

    // MARK: 3. a hole

    /// A class with no ANE timing changes no other class's decision (its figures move a little: a route not measured at a
    /// point borrows from the others, and the allocation follows).
    func testAHoleVetoesNothing() {
        func same(_ a: CM.Decision, _ b: CM.Decision) { XCTAssertEqual(a.chosen, b.chosen); XCTAssertEqual(a.status, b.status) }
        var base: [[Double?]] = [[0.1, 0.09, 0.06], [0.08, 0.06, 0.075], [0.07, 0.08, 0.05]]
        let cls = classes([0.4, 0.3, 0.3])
        var settings = CM.Settings(); settings.budget = 200
        let whole = plan(cls, Sim(base: base, noise: 0.02), settings: settings)
        base[1][2] = nil
        let holed = plan(cls, Sim(base: base, noise: 0.02), settings: settings)
        same(whole.decisions[0], holed.decisions[0])
        same(whole.decisions[2], holed.decisions[2])
        XCTAssertEqual(whole.decisions[0].chosen, "gpu+ane")
        XCTAssertEqual(holed.decisions[1].chosen, "gpu+amx", "the other route of the holed class still decides")
        XCTAssertNil(holed.decisions[1].hotRatios[2])
        // A route judged wrong is excluded the same way.
        var wrong = cls
        wrong[0].routes[2].correct = false
        let judged = plan(wrong, Sim(base: base, noise: 0.02), settings: settings)
        XCTAssertNotEqual(judged.decisions[0].chosen, "gpu+ane")
        same(judged.decisions[2], holed.decisions[2])
    }

    // MARK: 4. budget

    func testBudgetIsHeldAndHeavyClassesGetMoreMeasures() {
        // Same gap, same noise (a 4 % advantage, a hard call); the weights differ ×10.
        let base: [[Double?]] = [[0.05, 0.049, nil], [0.05, 0.049, nil]]
        let cls = [CM.ComputeClass("heavy", weight: 0.9, routes: [gpu, amx]), CM.ComputeClass("light", weight: 0.1, routes: [gpu, amx])]
        var settings = CM.Settings(); settings.budget = 3; settings.epsilon = 1e-4
        var heavy = 0, light = 0, exhausted = 0
        for seed in UInt64(1)...12 {
            let p = plan(cls, Sim(base: base, noise: 0.06, seed: seed), settings: settings)
            XCTAssertLessThanOrEqual(p.spent, settings.budget * 1.02, "seed \(seed)")
            if p.exhausted { exhausted += 1; XCTAssertGreaterThan(p.residualLoss, 0, "what stays open is reported") }
            heavy += p.decisions[0].runs; light += p.decisions[1].runs
        }
        XCTAssertGreaterThanOrEqual(exhausted, 6, "the budget really was the limit")
        XCTAssertGreaterThan(heavy, Int(1.5 * Double(light)), "heavy \(heavy) light \(light)")
        // A starved budget falls back to the simplest route.
        settings.budget = 0.2
        let starved = plan(cls, Sim(base: base, noise: 0.06), settings: settings)
        XCTAssertLessThanOrEqual(starved.spent, 0.2 * 1.1)
        XCTAssertEqual(starved.decisions[1].chosen, "gpu")
        XCTAssertEqual(starved.decisions.map(\.status), [.open, .unmeasured])
    }

    func testDeterministic() {
        let base: [[Double?]] = [[0.1, 0.09, 0.08], [0.07, 0.08, 0.06]]
        let cls = classes([0.6, 0.4])
        XCTAssertEqual(plan(cls, Sim(base: base, noise: 0.03, seed: 7)), plan(cls, Sim(base: base, noise: 0.03, seed: 7)))
    }

    // MARK: 5. the Neo

    /// A MacBook Neo's capacity map (A18 Pro, 8 GB, schema 12): z-image and qwen-image-2.1 products, TFLOP/s per route, and the witnesses
    /// (AMX 1.538 → 0.747, GPU 1.452 → 1.394). Cold figures are simulated; the hot regime is applied by `Heat`.
    func testNeoTakesGPUPlusANEWhereItWins() {
        // (k, n, m, gpu, amx pair, ane alone or nil, ane's GPU share), from that report.
        let rows: [(Int, Int, Int, Double, Double, Double?, Double)] = [
            (3840, 10240, 1024, 1.452, 2.59, 1.401, 0.05), (10240, 3840, 1024, 1.407, 2.116, nil, 0),
            (3840, 10240, 4096, 1.475, 2.254, 3.178, 0.05), (10240, 3840, 4096, 1.308, 2.128, 3.095, 0.05),
            (4096, 12288, 1024, 1.323, 1.731, 2.527, 0.05), (12288, 4096, 1024, 1.287, 1.703, 2.557, 0.05),
            (4096, 12288, 4096, 1.295, 2.003, 4.067, 0.05), (12288, 4096, 4096, 1.204, 1.934, 2.939, 0.05),
        ]
        func seconds(_ k: Int, _ n: Int, _ m: Int, _ tflops: Double) -> Double { 2 * Double(m) * Double(k) * Double(n) / (tflops * 1e12) }
        let cls = rows.enumerated().map { i, r in
            CM.ComputeClass("p\(i)", weight: 2 * Double(r.2) * Double(r.0) * Double(r.1), routes: [gpu, amx, ane])
        }
        let base: [[Double?]] = rows.map { r in
            [seconds(r.0, r.1, r.2, r.3), seconds(r.0, r.1, r.2, r.4),
             r.5.map { seconds(r.0, r.1, r.2, $0 + r.6 * r.3) }]
        }
        // Repeat each product so that a run lasts as a real chunk of work (a layer's worth) and the budget is in seconds.
        let scaled = base.map { $0.map { $0.map { $0 * 3 } } }
        var settings = CM.Settings(); settings.budget = 60
        let heat = CM.Heat(["gpu": 1.452 / 1.394, "amx": 1.538 / 0.7468])

        let hot = plan(cls, Sim(base: scaled, noise: 0.02), heat: heat, settings: settings)
        for i in 2...7 { XCTAssertEqual(hot.decisions[i].chosen, "gpu+ane", "p\(i)") }
        XCTAssertEqual(hot.decisions[1].chosen, "gpu", "no ANE timing here: the GPU, the AMX pair loses hot")
        XCTAssertFalse(hot.decisions.contains { $0.chosen == "gpu+amx" })
        XCTAssertLessThanOrEqual(hot.spent, settings.budget * 1.02)
        XCTAssertLessThan(hot.residualLoss, 0.03)

        // Without the regime (cold), the AMX pair is what wins where the ANE has no timing: the heat is what moved it.
        let cold = plan(cls, Sim(base: scaled, noise: 0.02), settings: settings)
        XCTAssertEqual(cold.decisions[1].chosen, "gpu+amx")
        XCTAssertEqual(cold.decisions[0].chosen, "gpu+amx")
    }
}
