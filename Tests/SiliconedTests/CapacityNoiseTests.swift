import Foundation
import Testing
@testable import Siliconed

/// **What a map is sure to, and what the plan does under it**: the rounded pair's per-call cost in k·n (`Sharing.perWeight`), the
/// figure measured again rather than left out (`CapacityFit.suspects`), the tile and the route chosen by a written order under the
/// noise (`MatmulTile.choose` on paired rounds, `CapacitySolver.canonical`, the rounded pair's floor), the fallbacks a render names (`ProductLedger.Fallback`) and the bias each stage
/// of the map removes (`Diagnostic.CapacityProbes.biases`). Pure: no GPU, no weights.
@Suite struct CapacityNoiseTests {
    typealias M = CapacityModel
    typealias CS = CapacitySolver

    // MARK: The rounded pair at every size

    /// The model an M5 Pro's map fitted for the rounded pair (its solos, its chains at m = 4 096, the widening rate): the lines the
    /// solver reads for a Standard's bf16 product, as the report carries them.
    static let pairMachine = #"""
    {"lines": {"gpu/fp32": {"rate": 5702300000000, "t0": 0},
               "gpu/bf16act": {"rate": 19367400000000, "t0": 0.000420067},
               "ane/fp16": {"perRow": 8.96888e-11, "perWeight": 7.67312e-11, "rate": 10484200000000, "rowsFrom": 256, "t0": 0}},
     "thin": {"gpu/bf16act": {"perRow": 5.1689e-12, "perWeight": 1.29223e-12, "rate": 19367400000000, "t0": 0.000155142},
              "ane/fp16": {"perRow": 7.29672e-11, "perWeight": 3.64836e-11, "rate": 10484200000000, "rowsFrom": 128, "t0": 0.000489013}},
     "aneRowUnit": 256, "noise": 0.0236823,
     "sharing": {"gpu+ane": {"kFrom": 1920, "kTo": 24576, "keep": {"ane": 0.933843, "gpu": 0.907383}, "overhead": 3.1e-05},
                 "gpu/bf16act+ane/fp16": {"kFrom": 1920, "kTo": 24576, "keep": {"ane": 0.736983, "gpu": 0.869709},
                 "keepByK": [{"k": 3840, "keep": {"ane": 0.536083, "gpu": 0.998544}}, {"k": 12288, "keep": {"ane": 0.937884, "gpu": 0.740875}}],
                 "overhead": 6.5e-06, "perWeight": {"ane": 2.5443e-11}}},
     "preparation": {"bf16": {"bf16": 1.4882e-11, "widened": 2.5443e-11}}}
    """#

    /// **The ANE pays for its whole bf16 copy at every call**, whatever its rows: at a 512² image's 1 024 rows that cost outweighs its
    /// share and the pair is far past the margin; at a 1 024² image's 4 096 it is a few percent and the pair stays. Without the term the
    /// same model hands the ANE 256 rows at 1 024 — the pair a render measured 13 to 33 % slower than the GPU alone.
    @Test func theRoundedPairPaysTheANEsCopyPerCall() throws {
        let model = try JSONDecoder().decode(M.self, from: Data(Self.pairMachine.utf8))
        let sharing = try #require(model.sharing["gpu/bf16act+ane/fp16"])
        #expect(sharing.lift(.ane, k: 12288, n: 4096) == 2.5443e-11 * 12288 * 4096)
        #expect(sharing.lift(.gpu, k: 12288, n: 4096) == 0)
        let c = CS.Constraints(sustained: false, allowsANE: true)
        for (k, n) in [(12288, 4096), (10240, 3840), (16384, 6144), (4096, 4096)] {
            let small = try #require(CS.product(ProductKey(k: k, n: n, m: 1024, weight: .bf16, phase: .denoising), model: model, constraints: c))
            #expect(small.route == "gpu/bf16act", "\(k)x\(n)@1024: \(small.route)")
            // at 4 096 rows the pair stays ahead by a few percent: a set with the ANE is kept unless proven slower (the ANE on a tie)
            let largeKey = ProductKey(k: k, n: n, m: 4096, weight: .bf16, phase: .denoising)
            let all = CS.candidates(largeKey, model: model, constraints: c)
            let pair = try #require(all.filter { $0.route == "gpu+ane/bf16act" }.map { $0.compute + $0.preparation }.min())
            let alone = try #require(all.first { $0.route == "gpu/bf16act" }.map { $0.compute + $0.preparation })
            #expect(pair < alone, "\(k)x\(n)@4096: \(pair) against \(alone)")
            let large = try #require(CS.product(largeKey, model: model, constraints: c))
            #expect(large.route == "gpu+ane/bf16act", "\(k)x\(n)@4096: \(large.route), gap \(1 - pair / alone)")
        }
        // the same map without the term: its 256 ANE rows priced the copy's widening lower. With it, the pair at the ANE's floor is
        // still a candidate (the balance gives the ANE nothing), but far past the margin
        var without = model
        without.sharing["gpu/bf16act+ane/fp16"]?.perWeight = nil
        let small = ProductKey(k: 12288, n: 4096, m: 1024, weight: .bf16, phase: .denoising)
        let candidates = CS.candidates(small, model: model, constraints: c)
        let floor = try #require(candidates.first { $0.route == "gpu+ane/bf16act" && $0.assignment.ane == 256 })
        let alone = try #require(candidates.first { $0.route == "gpu/bf16act" })
        #expect(floor.compute > 1.15 * alone.compute, "\(floor.compute) against \(alone.compute)")
        // the pair a render ran there (768 + 256 rows, 7.64 ms measured; the GPU alone 4.8 ms): the term prices it near what it cost
        let pair = ProductAssignment(path: .bf16, roundsActivation: true, gpu: 768, ane: 256)
        let priced = try #require(CS.product(assigned: pair, key: small, model: model, constraints: c))
        let unpriced = try #require(CS.product(assigned: pair, key: small, model: without, constraints: c))
        #expect(abs(priced.compute - 7.64e-3) < abs(unpriced.compute - 7.64e-3), "\(priced.compute) against \(unpriced.compute)")
        // the term survives the report's round trip
        let again = try JSONDecoder().decode(M.self, from: JSONEncoder().encode(model))
        #expect(again.sharing["gpu/bf16act+ane/fp16"]?.perWeight == [.ane: 2.5443e-11] && again.noise == model.noise)
    }

    // MARK: The line under a disturbed figure

    /// **A figure the Mac disturbed is measured again, never left out**: three points on t = 0.4 ms + W / 19.4 TFLOP/s, the middle one
    /// 60 % slow, is the one figure past the regime's spread over the line of the two others (`suspects`); measured again, the line is the
    /// truth. A calm figure 10 % off is no suspect, and neither is anything under three points (no line of the others to read it against).
    @Test func aDisturbedFigureIsMeasuredAgain() throws {
        typealias P = CapacityFit.Point
        let truth = { (w: Double) in 0.0004 + w / 19.4e12 }
        let ws = [2.0 * 1024 * 3840 * 3840, 2.0 * 2048 * 3840 * 3840, 2.0 * 4096 * 3840 * 3840]
        let disturbed = [P(flop: ws[0], seconds: truth(ws[0])), P(flop: ws[1], seconds: 1.6 * truth(ws[1])), P(flop: ws[2], seconds: truth(ws[2]))]
        #expect(CapacityFit.suspects(disturbed) == [1])
        var again = disturbed
        again[1] = P(flop: ws[1], seconds: min(disturbed[1].seconds, truth(ws[1])))
        let line = try #require(CapacityFit.line(again))
        #expect(abs(line.rate / 19.4e12 - 1) < 1e-3 && abs(line.t0 - 0.0004) < 1e-6, "\(line)")
        #expect(CapacityFit.suspects([disturbed[0], P(flop: ws[1], seconds: 1.1 * truth(ws[1])), disturbed[2]]).isEmpty)
        #expect(CapacityFit.suspects(Array(disturbed.prefix(2))).isEmpty)
    }

    // MARK: Under the noise, a written order

    /// **A tile's gap is read round by round, one margin for all**: 3 % slower than the fastest in every round is the listed one (under
    /// the 5 % margin), 12 % slower is not. A candidate whose rounds scatter gains nothing by its scatter: its fast side 10 % under the
    /// listed one's, its rounds 11 % under, 14 % over, 6 % over — the listed one, 6 % over it at the median, takes it by the margin; and
    /// rounds that fall on both sides of the margin put the candidate in doubt, to be measured again.
    @Test func aTileIsChosenByItsPairedRounds() {
        let names = ["64x32/4", "128x64/4"]
        let even = ["64x32/4": [10.3, 10.3, 10.3], "128x64/4": [10.0, 10.0, 10.0]]
        #expect(MatmulTile.choose(["64x32/4": 10.3, "128x64/4": 10.0], among: names, standard: "64x32/4", rounds: even) == "64x32/4")
        let far = ["64x32/4": [11.2, 11.2, 11.2], "128x64/4": [10.0, 10.0, 10.0]]
        #expect(MatmulTile.choose(["64x32/4": 11.2, "128x64/4": 10.0], among: names, standard: "64x32/4", rounds: far) == "128x64/4")
        let scattered = ["64x32/4": [10.0, 10.1, 10.0], "128x64/4": [9.0, 11.5, 10.6]]
        let seconds = ["64x32/4": 10.0, "128x64/4": 9.0]
        #expect(MatmulTile.choose(seconds, among: names, standard: "64x32/4", rounds: scattered) == "64x32/4")
        #expect(abs((MatmulTile.gaps(seconds, rounds: scattered)?["64x32/4"] ?? 0) - (10.0 / 10.6 - 1)) < 1e-12)
        // the listed one 3 %, 4 %, then 9 % over in its rounds: in doubt; 3 % in each, no doubt
        let straddling = ["64x32/4": [10.3, 10.4, 10.9], "128x64/4": [10.0, 10.0, 10.0]]
        #expect(MatmulTile.ambiguous(["64x32/4": 10.3, "128x64/4": 10.0], among: names, standard: "64x32/4", rounds: straddling) == ["64x32/4"])
        #expect(MatmulTile.ambiguous(["64x32/4": 10.3, "128x64/4": 10.0], among: names, standard: "64x32/4", rounds: even).isEmpty)
        // without rounds: the fast sides
        #expect(MatmulTile.choose(["64x32/4": 10.3, "128x64/4": 10.0], among: names, standard: "64x32/4") == "64x32/4")
    }

    /// The rounded pair as two maps of one M5 Pro a minute apart fitted it (3/11/2026, this code on their figures): map a with κ at k
    /// 3 840 and 12 288, map b at k 3 840 only (its probe skipped the long k's).
    static let pairMapA = #"""
    {"lines": {"gpu/fp32": {"rate": 5807640000000, "t0": 0.000176578}, "gpu/bf16act": {"rate": 24287200000000, "t0": 0.00077074},
               "ane/fp16": {"perRow": 1.09662e-10, "perWeight": 1.09286e-10, "rate": 12058900000000, "rowsFrom": 256, "t0": 0}},
     "aneRowUnit": 256, "noise": 0.0293858,
     "sharing": {"gpu+ane": {"kFrom": 1920, "kTo": 24576, "keep": {"ane": 0.933116, "gpu": 0.900993}, "overhead": 4.005e-05},
                 "gpu/bf16act+ane/fp16": {"kFrom": 1920, "kTo": 24576, "keep": {"ane": 0.873894, "gpu": 0.695924},
                 "keepByK": [{"k": 3840, "keep": {"ane": 0.758714, "gpu": 0.851893}}, {"k": 12288, "keep": {"ane": 0.989075, "gpu": 0.539955}}],
                 "overhead": 6.305e-05, "perWeight": {"ane": 2.5099e-11}}},
     "preparation": {"bf16": {"bf16": 1.4919e-11, "widened": 2.5099e-11}}}
    """#
    static let pairMapB = #"""
    {"lines": {"gpu/fp32": {"rate": 5977030000000, "t0": 0.000741352}, "gpu/bf16act": {"rate": 20526400000000, "t0": 0.000270435},
               "ane/fp16": {"perRow": 1.02333e-10, "perWeight": 7.61227e-11, "rate": 10852100000000, "rowsFrom": 256, "t0": 0}},
     "aneRowUnit": 256, "noise": 0.0110559,
     "sharing": {"gpu+ane": {"kFrom": 1920, "kTo": 24576, "keep": {"ane": 0.928173, "gpu": 0.902774}, "overhead": 2.875e-05},
                 "gpu/bf16act+ane/fp16": {"kFrom": 1920, "kTo": 7680, "keep": {"ane": 0.715841, "gpu": 0.873703},
                 "overhead": 7.37e-05, "perWeight": {"ane": 2.3016e-11}}},
     "preparation": {"bf16": {"bf16": 1.5248e-11, "widened": 2.3016e-11}}}
    """#

    /// **The pair 0.1 % either side of the GPU alone is the pair, whichever map measured what** (a set with the ANE is dropped only
    /// where proven slower, past twice the error of the difference — the ANE on a tie, decided 3/11/2026): Z-Image's `3840x10240` at a 1 024²
    /// image's 4 128 rows, which the first map priced 0.1 % under the GPU alone (the pair taken) and the second left out (the balance
    /// gave the ANE no row: no candidate, the GPU alone; offered at its floor alone, the pair's best time jumped by 4 % where the balance
    /// starts to give it rows). Each map's lines with each map's sharing, the ANE's speed set so that the pair
    /// lands 0.1 % under, then 0.1 % over the GPU alone: the pair in all eight, never the sign of the gap, at the ANE's most rows
    /// within the margin (never the rows a flat minimum falls on); a tenth under, the pair; and the GPU alone once the pair is past it.
    @Test func thePairATenthOfAPercentAwayIsThePair() throws {
        let a = try JSONDecoder().decode(M.self, from: Data(Self.pairMapA.utf8))
        let b = try JSONDecoder().decode(M.self, from: Data(Self.pairMapB.utf8))
        let key = ProductKey(k: 3840, n: 10240, m: 4128, weight: .bf16, phase: .denoising)
        let c = CS.Constraints(sustained: false, allowsANE: true)
        let pairKey = "gpu/bf16act+ane/fp16"
        func times(_ m: M) -> (pair: Double, alone: Double)? {
            let all = CS.candidates(key, model: m, constraints: c)
            guard let pair = all.filter({ $0.route == "gpu+ane/bf16act" }).map(\.compute).min(),
                  let alone = all.first(where: { $0.route == "gpu/bf16act" })?.compute else { return nil }
            return (pair, alone)
        }
        // the pair's best split `ratio` times the GPU alone, by the ANE's speed (its line's every term scaled; the GPU alone does not read it), bisected
        func tuned(_ m: M, _ ratio: Double) throws -> M {
            let ane = M.Kernel(.ane, .fp16), base = try #require(m.lines[ane])
            func scaled(_ f: Double) -> M {
                var s = m
                s.lines[ane] = .init(t0: base.t0 / f, rate: base.rate * f, perWeight: base.perWeight.map { $0 / f }, rowsFrom: base.rowsFrom,
                                     perRow: base.perRow.map { $0 / f })
                return s
            }
            var (lo, hi) = (0.05, 20.0)
            for _ in 0..<60 {
                let f = (lo * hi).squareRoot()
                guard let t = times(scaled(f)) else { lo = f; continue }
                if t.pair / t.alone > ratio { lo = f } else { hi = f }
            }
            let out = scaled(hi), t = try #require(times(out))
            #expect(abs(t.pair / t.alone - ratio) < 1e-4, "\(t)")
            return out
        }
        for lines in [a, b] {
            for sharing in [a, b] {
                var m = lines
                m.sharing[pairKey] = sharing.sharing[pairKey]
                // a tenth of a percent either side (within the error of the difference: the ANE on a tie), and a tenth under
                for ratio in [0.999, 1.001, 0.90] {
                    let t = try tuned(m, ratio)
                    let s = try #require(CS.product(key, model: t, constraints: c))
                    #expect(s.route == "gpu+ane/bf16act", "ratio \(ratio), noise \(m.noise ?? 0): \(s.route)")
                    // and within the margin, the ANE's most rows: never the rows a flat minimum falls on
                    let all = CS.candidates(key, model: t, constraints: c)
                    let fastest = try #require(all.map { $0.compute + $0.preparation }.min())
                    let width = max(MatmulTile.margin, 2 * (t.noise ?? 0))
                    let most = all.filter { $0.route == "gpu+ane/bf16act" && $0.compute + $0.preparation <= fastest * (1 + width) }.map(\.assignment.ane).max()
                    #expect(s.assignment.ane == most && (most ?? 0) > 256, "\(s.assignment.ane) against \(most ?? 0)")
                }
                // past the margin (max(5 %, twice the noise) — 5.9 % under map a's noise): the GPU alone
                let s = try #require(CS.product(key, model: try tuned(m, 1.07), constraints: c))
                #expect(s.route == "gpu/bf16act", "\(s.route)")
            }
        }
    }

    /// **Two routes closer than the noise are one, and the written order takes**: the ANE beside the GPU before the GPU alone (its watts
    /// are the fewest), the GPU alone before the AMX beside it, the widened weight before the bf16 one; past the margin the faster wins.
    @Test func underTheNoiseTheWrittenOrderDecides() throws {
        let gpu = M.Kernel(.gpu, .fp32), amx = M.Kernel(.amx, .fp32), ane = M.Kernel(.ane, .fp16), bf16 = M.Kernel(.gpu, .bf16)
        func machine(gpuRate: Double, amxRate: Double? = nil, aneRate: Double? = nil, bf16Rate: Double? = nil, noise: Double?) -> M {
            var m = M()
            m.lines[gpu] = .init(t0: 0, rate: gpuRate)
            if let amxRate { m.lines[amx] = .init(t0: 0, rate: amxRate) }
            if let aneRate { m.lines[ane] = .init(t0: 0, rate: aneRate) }
            if let bf16Rate { m.lines[bf16] = .init(t0: 0, rate: bf16Rate) }
            m.noise = noise
            return m.withFreeSharing()
        }
        let c = CS.Constraints(sustained: false, allowsANE: true)
        let key = ProductKey(k: 4096, n: 4096, m: 4096, weight: .bf16, phase: .encoding)
        // the AMX beside the GPU, 1/0.97 faster: a gain by a part in a billion without noise, none under 5 %
        let withAMX = machine(gpuRate: 3e12, amxRate: 0.0928e12, noise: nil)
        #expect(try #require(CS.product(key, model: withAMX, constraints: c)).route == "gpu+amx")
        var calm = withAMX; calm.noise = 0.01
        #expect(try #require(CS.product(key, model: calm, constraints: c)).route == "gpu")
        // the ANE beside the GPU, 3 % faster than the GPU alone once its launching is paid: the pair without noise and under it alike —
        // within the margin, the written order puts a set with the ANE before the GPU alone (a set slower than the GPU alone is never
        // a candidate: the split leaves the ANE out)
        var withANE = machine(gpuRate: 3e12, aneRate: 0.75e12, noise: nil)
        let alone = 2.0 * 4096 * 4096 * 4096 / 3e12
        withANE.sharing["gpu+ane"] = .init(overhead: alone / 1.03 - 0.03722304989866667, keep: [.gpu: 1, .ane: 1])
        let denoising = ProductKey(k: 4096, n: 4096, m: 4096, weight: .bf16, phase: .denoising)
        let pair = try #require(CS.product(denoising, model: withANE, constraints: c))
        #expect(pair.route == "gpu+ane" && abs(pair.compute * 1.03 / alone - 1) < 1e-3, "\(pair.route) \(pair.compute)")
        withANE.noise = 0.01
        #expect(try #require(CS.product(denoising, model: withANE, constraints: c)).route == "gpu+ane")
        // the bf16 weight 3 % faster than the widened one: the widened under the noise, the bf16 one past it
        var prepared = machine(gpuRate: 3e12, bf16Rate: 3.09e12, noise: 0.01)
        prepared.preparation[.bf16] = [.widened: 0, .bf16: 0]
        #expect(try #require(CS.product(key, model: prepared, constraints: c)).route == "gpu")
        prepared.lines[bf16] = .init(t0: 0, rate: 3.6e12)
        #expect(try #require(CS.product(key, model: prepared, constraints: c)).route == "gpu/bf16")
        // a thin product's floor is the thin tiles' (15 %): 10 % faster is no gain there
        prepared.lines[bf16] = .init(t0: 0, rate: 3.3e12)
        let thin = ProductKey(k: 4096, n: 4096, m: 32, weight: .bf16, phase: .encoding)
        #expect(try #require(CS.product(thin, model: prepared, constraints: c)).route == "gpu")
        #expect(try #require(CS.product(key, model: prepared, constraints: c)).route == "gpu/bf16")
    }

    // MARK: Fallbacks named

    /// **A route not taken is named**: the plan without a model or without an answer says why its product is on the GPU alone; the
    /// ledger lists each fallback once per reason, route and class; the render's table and its note carry them.
    @Test func aFallbackIsNamedNeverSilent() throws {
        let key = ProductKey(k: 4096, n: 4096, m: 4096, weight: .bf16, phase: .denoising)
        let none = CapacityPlan.defaults
        #expect(none.assign(key) == .gpuAlone(4096) && none.fallback(key) == .noModel)
        var empty = M(); empty.lines[M.Kernel(.ane, .fp16)] = .init(t0: 0, rate: 1e12)
        let unsolved = CapacityPlan(model: empty)
        #expect(unsolved.assign(key) == .gpuAlone(4096) && unsolved.fallback(key) == .unsolved)
        var solvable = M(); solvable.lines[M.Kernel(.gpu, .fp32)] = .init(t0: 0, rate: 1e12)
        let solved = CapacityPlan(model: solvable)
        _ = solved.assign(key)
        #expect(solved.fallback(key) == nil)

        let ledger = ProductLedger()
        let work = ProductLedger.Work.product(k: 4096, n: 4096, m: 4096, weight: .bf16)
        ledger.fellBack(work, planned: "gpu+ane/bf16act", reason: .aneGivenBack)
        ledger.fellBack(work, planned: "gpu+ane/bf16act", reason: .aneGivenBack)
        ledger.fellBack(work, planned: "gpu/bf16act", reason: .roundingUnavailable)
        let f = ledger.fallbacks
        #expect(f.map(\.reason) == [.roundingUnavailable, .aneGivenBack] && f.map(\.executions) == [1, 2])
        let lines = RenderLoop.lines(RenderLoop.compare([]), title: "t", fallbacks: f)
        #expect(lines.contains("#fallback\tproduct:4096x4096@4096\t2\tgpu+ane/bf16act\tane given back"), "\(lines)")
        #expect(lines.last == "#fallbacks\t\t3\t\t")
        #expect(RenderLoop.lines(RenderLoop.compare([]), title: "t").last == "#fallbacks\t\t0\t\t")
        let request = CapacityClasses.Request(family: .zImage, variant: .standard, width: 1024, height: 1024, fasterRendering: false)
        let note = RenderLoop.Note(request: request, map: "", comparison: .init(rows: []), verdict: .unjudged(.noModel), fallbacks: f)
        #expect(note.fallbacks == ["rounding unavailable": 1, "ane given back": 2])
        #expect(RenderLoop.Note(request: request, map: "", comparison: .init(rows: []), verdict: .unjudged(.noModel)).fallbacks == nil)
    }

    // MARK: Each second of the map names its bias

    /// **Every stage the probes time removes a named bias**: the stages the sizes project, the fixed costs, and the two pairs' chains.
    @Test func everyStageNamesTheBiasItRemoves() {
        let projected = Diagnostic.CapacityProbes.Sizes.full.projection(CapacityProbeSizesTests.m1Pro).keys
        let stages = Set(projected).union(["preparation", "witnesses start", "setup", "fit"])
        for stage in stages.sorted() { #expect(!(Diagnostic.CapacityProbes.biases[stage] ?? "").isEmpty, "\(stage) names no bias") }
        #expect(Set(Diagnostic.CapacityProbes.biases.keys) == stages, "a bias without a stage: \(Set(Diagnostic.CapacityProbes.biases.keys).subtracting(stages))")
        let t = Diagnostic.StageTime(seconds: 1.234567, projected: 1.1, removes: Diagnostic.CapacityProbes.biases["stairs"])
        #expect(t.seconds == 1.23 && t.removes == Diagnostic.CapacityProbes.biases["stairs"])
    }
}
