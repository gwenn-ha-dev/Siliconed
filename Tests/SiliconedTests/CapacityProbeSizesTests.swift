import Foundation
import Testing
@testable import Siliconed

/// **How big the probes are, and what they are projected to cost** (`CapacityProbeSizes.swift`): the tiers nest, the minute is held by
/// the slowest Mac of the fleet from what its first witnesses say — never from its name —, and the projection is the
/// right order on the Mac the probes were tuned on. Pure.
@Suite struct CapacityProbeSizesTests {
    typealias Sizes = Diagnostic.CapacityProbes.Sizes
    typealias Rates = Diagnostic.CapacityProbes.Rates

    /// What the probes measured on the M1 Pro in the runs of the report's version (seconds per stage), against which the projection is checked.
    static let measured: [String: Double] = ["preparation": 0.18, "witnesses start": 0.5, "bandwidth": 0.04, "disk": 2.0, "solos fp32": 7.8, "stairs": 1.0,
                                              "attention": 4.0, "sharing": 9.0, "solos 8-bit and rounded": 5.3, "sharing int8": 2.1,
                                              "witnesses end": 0.43, "thin products": 2.53]
    /// The engines alone on the witness (2 048 × 3 840²: 60.4 GFLOP in 17.2, 26.2 and 12.4 ms): TFLOP/s.
    static let m1Pro = Rates(gpu: 3.51e12, amx: 2.30e12, ane: 4.87e12, cpu: 1.0e12)
    /// A Neo's order: a 5-core GPU, the AMX and the ANE of its generation (the fleet's: 1.43 · 1.52 (0.83 hot) · 4.13 TFLOP/s).
    static let neo = Rates(gpu: 1.43e12, amx: 1.52e12, ane: 4.13e12, cpu: 0.5e12)

    @Test func tiersShrinkAndTheirOperandsNest() {
        #expect(Sizes.tiers.map(\.tier) == [0, 1, 2])
        for (big, small) in zip(Sizes.tiers, Sizes.tiers.dropFirst()) {
            #expect(small.operandsFit(in: big), "tier \(small.tier) fits the operands allocated for tier \(big.tier)")
            #expect(!big.operandsFit(in: small))
            #expect(small.solo.count <= big.solo.count && small.sharing.count <= big.sharing.count)
            #expect(small.attention.max()! <= big.attention.max()!)
            #expect(small.projectedSeconds(Self.m1Pro) < big.projectedSeconds(Self.m1Pro))
        }
        // The first tier is what every Mac so far ran: 4, 30 and 309 GFLOP; 30, 60 and 121; S = 1 024 and 4 096.
        let full = Sizes.full
        #expect(full.solo.map { 2.0 * Double($0.m) * Double($0.k) * Double($0.n) / 1e9 }.map { Int($0.rounded()) } == [4, 30, 309])
        #expect(full.sharing.map { 2.0 * Double($0.m) * Double($0.k) * Double($0.n) / 1e9 }.map { Int($0.rounded()) } == [30, 60, 121])
        #expect(full.attention == [1024, 4096])
        // The attention's pair runs at the models' sequences in every tier (the half and the full grid of 1024²): a tier that measured it
        // at 2 048 took the cut by heads off every full-size attention.
        #expect(Sizes.pairSequences == [1024, 4096] && Sizes.tiers.allSatisfy { Set(Sizes.pairSequences).isSubset(of: $0.attentionRuns) })
        // Every tier keeps a small, a middle and a large product while it can: a line needs a t₀ and a rate.
        #expect(Sizes.tiers.allSatisfy { $0.solo.count >= 2 && $0.stairRows.contains(where: { $0 % 512 != 0 }) })
        #expect(Sizes.tiers.allSatisfy { $0.products.contains { $0.m == Sizes.witness.m && $0.k == Sizes.witness.k && $0.n == Sizes.witness.n } })
    }

    /// The tier a small product lies in never asks for more rows than the pair's balance needs to exist (the ANE's unit is 256).
    @Test func everyStairRowIsAMultipleOfHalfAUnit() {
        for t in Sizes.tiers { #expect(t.stairRows.allSatisfy { $0 % 128 == 0 }) }
    }

    @Test func theProjectionIsTheRightOrderOnTheMacItWasTunedOn() {
        let p = Sizes.full.projection(Self.m1Pro)
        #expect(Set(p.keys) == Set(Self.measured.keys))
        for (stage, seconds) in Self.measured.sorted(by: { $0.key < $1.key }) {
            let projected = p[stage]!
            #expect(abs(projected / seconds - 1) < 0.35 || abs(projected - seconds) < 0.1, "\(stage): projected \(projected), measured \(seconds)")
        }
        let total = p.values.reduce(0, +), measured = Self.measured.values.reduce(0, +)
        #expect(abs(total / measured - 1) < 0.2, "projected \(total) s of the \(measured) s the probes took")
    }

    @Test func theFirstTierHoldsOnTheMacItWasTunedOn() {
        #expect(Sizes.tier(for: Self.m1Pro).tier == 0)
        #expect(Sizes.tier(for: Self.m1Pro, startingAt: 1).tier == 1, "a tier is never stepped back up")
    }

    /// **The slowest Mac of the fleet holds the minute**: the Neo's rates, and a Mac three times slower still, step down until the projection,
    /// with the heat's margin, fits what the minute leaves.
    @Test func aSlowMacStepsDownUntilTheMinuteHolds() {
        let budget = Diagnostic.minute - Sizes.reserve
        let neo = Sizes.tier(for: Self.neo)
        #expect(neo.projectedSeconds(Self.neo) * Sizes.margin <= budget, "the Neo, tier \(neo.tier): \(neo.projectedSeconds(Self.neo)) s")
        #expect(neo.projectedSeconds(Self.neo) < 40, "the old map took 57.6 s there, 31 of them fixed costs")
        let slower = Rates(gpu: Self.neo.gpu / 3, amx: Self.neo.amx.map { $0 / 3 }, ane: Self.neo.ane.map { $0 / 3 }, cpu: Self.neo.cpu / 3)
        let slow = Sizes.tier(for: slower)
        #expect(slow.tier > neo.tier || neo.tier == Sizes.tiers.count - 1)
        #expect(slow.projectedSeconds(slower) * Sizes.margin <= budget || slow.tier == Sizes.tiers.count - 1)
        // Whatever the rates, the choice is the first tier that fits (the smallest when none does), and a tighter budget never picks a bigger one.
        var last = 0
        for gpu in stride(from: 4.0e12, through: 0.05e12, by: -0.1e12) {
            let r = Rates(gpu: gpu, amx: gpu * 0.6, ane: gpu * 1.4, cpu: 1e12)
            let t = Sizes.tier(for: r).tier
            #expect(t >= last, "a slower GPU never picks a bigger tier")
            last = t
        }
        #expect(last == Sizes.tiers.count - 1)
    }

    /// **The first measurement sets the tier**, the memory only its ceiling: the Neo's real rates, which an inventory's guess (half the GPU's nominal
    /// peak, the other engines below it) would have under-estimated threefold for its AMX and Neural Engine, keep the whole first tier.
    @Test func theWitnessesStepTheTierDownAndNeverUp() throws {
        // The Neo's rates project the first tier at 39 s, 63 s with the heat's margin: over the minute, so the second (26 s, 41 with the margin) — which its
        // memory would give it anyway (below).
        #expect(Sizes.tier(for: Self.neo).tier == 1)
        // The same Neo, a third of its speed: the smallest tier (the first, with its chains of 121 and 412 GFLOP, projects 71 s there; the second 36 s).
        #expect(Sizes.tier(for: Rates(gpu: 0.477e12, amx: 0.5e12, ane: 1.38e12, cpu: 0.17e12)).tier == 2)
        // …and from the tier the memory allowed, never above it.
        #expect(Sizes.tier(for: Self.m1Pro, startingAt: 2).tier == 2)
        // The memory plan takes the largest tier B holds, whatever the machine is called.
        let plan = try Diagnostic.Plan.choose(budget: 4 << 30, base: 600 << 20) { Diagnostic.CapacityProbes.need(sizes: Sizes.tiers[$0]) }
        #expect(plan.tier == 1, "a Neo's B holds the second tier: the first needs 4.3 GB for the chains' long k (a weight of 201 MB, its copies and truths)")
        // A hot Neo (SME ÷1.9, GPU ÷1.2) still holds the minute, at the second tier.
        let hot = Rates(gpu: 1.19e12, amx: 0.83e12, ane: 4.13e12, cpu: 0.4e12)
        let tier = Sizes.tier(for: hot)
        #expect(tier.tier == 1)
        #expect(tier.projectedSeconds(hot) * Sizes.margin <= Diagnostic.minute - Sizes.reserve)
    }

    /// No ANE: its stages cost nothing, its sets are not run.
    @Test func anEngineThatIsAbsentCostsNothing() {
        let without = Sizes.full.projection(Rates(gpu: 3.5e12, amx: 2.3e12, ane: nil, cpu: 1e12))
        let with = Sizes.full.projection(Self.m1Pro)
        #expect(without["stairs"] == 0)
        #expect(without["sharing"]! < with["sharing"]!)
        #expect(without["solos fp32"]! < with["solos fp32"]!)
    }
}
