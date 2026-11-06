import Foundation
import Testing
@testable import Siliconed

/// **How big the probes are, and what they are projected to cost** (`CapacityProbeSizes.swift`): the tiers nest, the memory chooses the
/// tier and the time only past the ceiling — from what the first witnesses say, never from the Mac's name —, every tier measures the
/// sets at both k's, and the projection is the right order on the Mac the probes were tuned on. Pure.
@Suite struct CapacityProbeSizesTests {
    typealias Sizes = Diagnostic.CapacityProbes.Sizes
    typealias Rates = Diagnostic.CapacityProbes.Rates

    /// What the probes measured on the M1 Pro in the runs of the report's version (seconds per stage), against which the projection is checked.
    static let measured: [String: Double] = ["preparation": 0.18, "witnesses start": 0.5, "bandwidth": 0.04, "disk": 2.0, "solos fp32": 7.8, "stairs": 1.0,
                                              "attention": 4.0, "sharing": 9.0, "solos 8-bit and rounded": 5.3, "sharing int8": 2.1,
                                              "witnesses end": 3 * 0.43 + 2 * 0.2, "thin products": 2.53,   // the end: three rounds of 0.43 s, two pauses
                                              "sharing rounded": 0]   // never run on an M1 Pro: wholly untimed (`roundedPairSeconds`)
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
        func gigaflop(m: Int, k: Int, n: Int) -> Int {
            let flop: Double = 2.0 * Double(m) * Double(k) * Double(n)
            return Int((flop / 1e9).rounded())
        }
        let solo: [Int] = full.solo.map { gigaflop(m: $0.m, k: $0.k, n: $0.n) }
        let sharing: [Int] = full.sharing.map { gigaflop(m: $0.m, k: $0.k, n: $0.n) }
        #expect(solo == [4, 30, 309])
        #expect(sharing == [30, 60, 121])
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
        // The tiles' trials came after these runs too (2/11/2026); the pulses are not counted here (`m1Pro.pulse` is 0).
        let untimed = Sizes.full.roundedEightBitSeconds(Self.m1Pro).merging(Sizes.full.roundedPairSeconds(Self.m1Pro)) { $0 + $1 }
            .merging(Sizes.full.roundedProductSeconds(Self.m1Pro)) { $0 + $1 }.merging(Sizes.full.tileTrialSeconds(Self.m1Pro)) { $0 + $1 }
            .merging(Sizes.full.quantizedSeconds(Self.m1Pro)) { $0 + $1 }
        #expect(Set(p.keys) == Set(Self.measured.keys))
        for (stage, seconds) in Self.measured.sorted(by: { $0.key < $1.key }) {
            // The attention's 4.0 s were measured before the half-precision kernels chose their form (2/11/2026), the 8-bit stages before
            // the products on a rounded activation and the rounded bf16 pair joined them (3/11/2026): the parts of the projection no M1 Pro
            // has timed yet, left out
            // until one does.
            let projected = p[stage]! - (stage == "attention" ? Sizes.full.formSeconds(Self.m1Pro) : 0) - (untimed[stage] ?? 0)
            #expect(abs(projected / seconds - 1) < 0.35 || abs(projected - seconds) < 0.1, "\(stage): projected \(projected), measured \(seconds)")
        }
        let total = p.values.reduce(0, +) - Sizes.full.formSeconds(Self.m1Pro) - untimed.values.reduce(0, +), measured = Self.measured.values.reduce(0, +)
        #expect(abs(total / measured - 1) < 0.2, "projected \(total) s of the \(measured) s the probes took")
    }

    /// **No tier takes a pair off the long products**: the sets' chains at a short and a long k in every tier, so that the plan's pairs
    /// reach a DiT's down projection (k = 10 240, 12 288) whatever the tier the memory or the ceiling left.
    @Test func everyTierMeasuresTheSetsAtAShortAndALongK() {
        let ks = Set(Sizes.full.chains.map(\.k))
        #expect(ks == [3840, 12288])
        for t in Sizes.tiers { #expect(Set(t.chains.map(\.k)) == ks, "tier \(t.tier): \(t.chains.map(\.k))") }
    }

    @Test func theFirstTierHoldsOnTheMacItWasTunedOn() {
        #expect(Sizes.tier(for: Self.m1Pro).tier == 0)
        #expect(Sizes.tier(for: Self.m1Pro, startingAt: 1).tier == 1, "a tier is never stepped back up")
    }

    /// **The time chooses no tier on the Macs of the fleet**: the M1 Pro and the Neo project the first tier, with the heat's margin, under
    /// the ceiling, and a hot Neo the second, the one its memory allows — the memory alone sets their tier. A Neo at a third of its speed steps down: its map would take
    /// more than the margin leaves, which a projection without the tiles' trials and the pulses hid (it kept the first tier at 0.6 of the
    /// ceiling, and the five maps of the fleet took 1.7 to 2.1 times their projection). A slower GPU never picks a bigger tier.
    @Test func aSlowMacStepsDownOnlyPastTheCeiling() {
        let budget = Diagnostic.ceiling - Sizes.reserve
        // A pulse's order on the fleet's Macs (17 to 20 ms: fourteen empty submissions and four loops of 4 ms).
        func pulsed(_ r: Rates) -> Rates { var r = r; r.pulse = 0.018; return r }
        let hot = Rates(gpu: 1.19e12, amx: 0.83e12, ane: 4.13e12, cpu: 0.4e12)
        let third = Rates(gpu: 0.477e12, amx: 0.5e12, ane: 1.38e12, cpu: 0.17e12)
        for (name, r) in [("M1 Pro", Self.m1Pro), ("Neo", Self.neo)].map({ ($0.0, pulsed($0.1)) }) {
            #expect(Sizes.tier(for: r).tier == 0, "\(name)")
            #expect(Sizes.full.projectedSeconds(r) * Sizes.margin <= budget, "\(name): \(Sizes.full.projectedSeconds(r)) s")
        }
        // A hot Neo at the first tier projects past the ceiling since W8A8 is priced as the rounded kernels; its B never offers that tier
        // (the second, `theWitnessesStepTheTierDownAndNeverUp`), where it holds.
        let warm = pulsed(hot)
        #expect(Sizes.tier(for: warm).tier == 1 && Sizes.tiers[1].projectedSeconds(warm) * Sizes.margin <= budget,
                "hot Neo: \(Sizes.full.projectedSeconds(warm)) s at the first tier, \(Sizes.tiers[1].projectedSeconds(warm)) at the second")
        #expect(Sizes.tier(for: warm, startingAt: 1).tier == 1, "the tier its memory allows holds")
        var old = pulsed(third); old.pulse = 0; old.tiled = false
        #expect(Sizes.full.projectedSeconds(old) * Sizes.margin <= budget, "the projection without trials nor pulses kept the first tier")
        #expect(Sizes.full.projectedSeconds(pulsed(third)) * Sizes.margin > budget && Sizes.tier(for: pulsed(third)).tier > 0,
                "a third of a Neo: \(Sizes.full.projectedSeconds(pulsed(third))) s")
        var last = 0
        for gpu in stride(from: 4.0e12, through: 0.01e12, by: -0.01e12) {
            let r = Rates(gpu: gpu, amx: gpu * 0.6, ane: gpu * 1.4, cpu: gpu / 4)
            let t = Sizes.tier(for: r).tier
            #expect(t >= last, "a slower GPU never picks a bigger tier")
            last = t
        }
        #expect(last == Sizes.tiers.count - 1)
    }

    /// **The first measurement can only step the tier down**, from the tier the memory allowed, and the memory plan takes the largest
    /// tier B holds, whatever the machine is called: a Neo's B holds the second.
    @Test func theWitnessesStepTheTierDownAndNeverUp() throws {
        #expect(Sizes.tier(for: Self.m1Pro, startingAt: 1).tier == 1, "a tier is never stepped back up")
        #expect(Sizes.tier(for: Self.m1Pro, startingAt: 2).tier == 2)
        let plan = try Diagnostic.Plan.choose(budget: 4 << 30, base: 600 << 20) { Diagnostic.CapacityProbes.need(sizes: Sizes.tiers[$0]) }
        #expect(plan.tier == 1, "a Neo's B holds the second tier: the first needs 4.3 GB for the chains' long k (a weight of 201 MB, its copies and truths)")
    }

    /// **What a stage does beside its figures, counted as the probes do it**: the steps through the guard are those of an M5 Pro's map at
    /// the first tier (schema 22: its first measures, tiles' trials and attention's forms, 427 pulses once the thin trials' third round is
    /// taken off) and of a Neo's at the second (378), W8A8's added since schema 24 (no map has counted them yet), Z order's wide candidate since 25 (three kernels, three rounds); the trials are each candidate's rounds on the shapes its kernel takes — the
    /// 8-bit weights leave out a k under a group, the rounded one a reserve too small for 64 rows —; a pulse is counted once per step.
    @Test func theTrialsAndThePulsesAreCountedAsTheProbesRunThem() {
        let r = Rates(gpu: 5.8e12, amx: 1.8e12, ane: 7.1e12, cpu: 1.9e12, pulse: 0.018)
        // W8A8 (schema 24) adds its eight shapes and its tiles' 60 trials to the 8-bit stage, six thin shapes to the thin one.
        #expect(Sizes.full.steps(r) == ["solos fp32": 87, "stairs": 8, "attention": 39, "sharing": 12, "solos 8-bit and rounded": 213 + 8 + 60 + 9,
                                        "thin products": 59 + 6, "sharing int8": 6, "sharing rounded": 3])
        #expect(Sizes.tiers[1].steps(r).values.reduce(0, +) == 451)
        let trials = MatmulTile.kernels.map { Sizes.full.tileTrials($0).count }
        #expect(trials == [4 * (3 + 2 * 7), 5 * 3 + 4 * 2 * 7, 4 * (3 + 2 * 6), 5 * 3 + 4 * 2 * 5, 5 * 3 + 4 * 2 * 6],
                "bf16, bf16act, int8, int8act, int8x8")
        var silent = r; silent.pulse = 0
        let with = Sizes.full.projection(r), without = Sizes.full.projection(silent)
        for (stage, n) in Sizes.full.steps(r) { #expect(abs(with[stage]! - without[stage]! - Double(n) * 0.018) < 1e-9, "\(stage)") }
        var untiled = silent; untiled.tiled = false
        #expect(Sizes.full.tileTrialSeconds(untiled).isEmpty && Sizes.full.steps(untiled)["solos fp32"] == 87 - 68)
        let priced = Sizes.full.tileTrialSeconds(silent)
        #expect(abs(Sizes.full.projection(silent)["solos fp32"]! - Sizes.full.projection(untiled)["solos fp32"]! - priced["solos fp32"]!) < 1e-9)
        // A thin trial of a millisecond is seventeen runs (`steadyCalls`: fifteen at most), not four.
        #expect(Sizes.steadySeconds(0.001) == 0.001 * 17 && Sizes.steadySeconds(0.2) == 0.2 * 3)
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
