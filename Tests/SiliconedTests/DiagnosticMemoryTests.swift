import Foundation
import Testing
@testable import Siliconed

/// **The probes' memory need** (`MemoryNeedDiagnostic.swift`, `CapacityProbes.need`): a pure function of shapes — no weight, no
/// clock — summed before the first computation, a tier chosen under B, a peak above it a defect of the code.
@Suite struct DiagnosticMemoryTests {
    typealias Probes = Diagnostic.CapacityProbes
    let gib = 1 << 30

    @Test func theNeedShrinksWithTheTierAndAlwaysCountsTheFrameworks() {
        let needs = Probes.Sizes.tiers.map { Probes.need(sizes: $0) }
        for (big, small) in zip(needs, needs.dropFirst()) { #expect(small.peak < big.peak) }
        for need in needs { #expect(Set(need.steps.keys) == ["widening", "routes"] && need.steps.values.allSatisfy { $0 >= Diagnostic.frameworks }) }
        // The widest tier holds the chains' long k at 4 096 rows (a weight of 201 MB, its copies and truths): ~4.3 GB, plus the disk probe's staging (~0.2 GB: `CapacityDisk.probeMemoryBytes`); the second ~2.8, the third ~1.6.
        #expect((3.8e9...5.0e9).contains(Double(needs[0].peak)), "\(needs[0].peak)")
        #expect(Probes.need(sizes: .full, validating: true).peak > needs[0].peak)
    }

    /// A weight's operands are the largest of every shape: cutting a product to make it fit would measure another product.
    @Test func theOperandsAreTheLargestOfTheTiersShapes() {
        let s = Diagnostic.Operands.sizes(products: Probes.Sizes.full.products.map { ($0.m, $0.k, $0.n, WeightKind.q8) })
        #expect(s.aFloats == 4096 * 12288 && s.bFloats == 12288 * 4096 && s.cFloats == 4096 * 6144)
        let small = Diagnostic.Operands.sizes(products: Probes.Sizes.tiers[2].products.map { ($0.m, $0.k, $0.n, WeightKind.q8) })
        #expect(small.bytes < s.bytes)
    }

    @Test func theLargestTierThatFitsB() throws {
        let big = try Diagnostic.Plan.choose(budget: 10 * gib, base: 16 << 20) { Probes.need(sizes: Probes.Sizes.tiers[$0]) }
        #expect(big.tier == 0 && big.base == 16 << 20 && big.budget == 10 * gib)
        // A Neo's B (¾ · 8 GiB − 2 GiB = 4 GiB) with an app's base: the need is counted, the plan fits under B.
        for base in [16 << 20, 300 << 20, 600 << 20] {
            let neo = try Diagnostic.Plan.choose(budget: 4 * gib, base: base) { Probes.need(sizes: Probes.Sizes.tiers[$0]) }
            #expect(neo.base + neo.need.peak <= 4 * gib)
            if neo.tier > 0 { #expect(neo.base + Probes.need(sizes: Probes.Sizes.tiers[neo.tier - 1]).peak > 4 * gib, "a larger tier would not have fitted") }
        }
        // The inventory's tier is a floor: the plan never returns a larger one.
        let from1 = try Diagnostic.Plan.choose(budget: 10 * gib, base: 16 << 20, from: 1) { Probes.need(sizes: Probes.Sizes.tiers[$0]) }
        #expect(from1.tier == 1)
    }

    @Test func notEvenTheSmallestTierFitsIsRefusedBeforeTheFirstProduct() {
        #expect(throws: EngineError.self) {
            try Diagnostic.Plan.choose(budget: 1 * gib, base: 16 << 20) { Probes.need(sizes: Probes.Sizes.tiers[$0]) }
        }
        do {
            _ = try Diagnostic.Plan.choose(budget: 1 * gib, base: 16 << 20) { Probes.need(sizes: Probes.Sizes.tiers[$0]) }
        } catch let e as EngineError {
            if case let .insufficientMemory(needed, budget) = e { #expect(budget == 1 * gib && needed > budget) } else { Issue.record("\(e)") }
        } catch { Issue.record("\(error)") }
    }

    /// The memory is held when the peak stays under the base and the need, and both under B.
    @Test func theMemoryHeldIsThePeakUnderTheNeedUnderB() throws {
        let m = Diagnostic.Memory(budget: 4 * gib, base: 100 << 20, need: 2 * gib, peak: 2 * gib)
        #expect(m.held)
        var over = m; over.peak = 2 * gib + (101 << 20)
        #expect(!over.held)
        var tight = m; tight.budget = 2 * gib
        #expect(!tight.held)
        #expect(try JSONDecoder().decode(Diagnostic.Memory.self, from: JSONEncoder().encode(m)) == m)
    }
}
