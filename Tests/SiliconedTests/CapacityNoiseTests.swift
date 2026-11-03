import Foundation
import Testing
@testable import Siliconed

/// **What a map is sure to, and what the plan does under it**: the rounded pair's per-call cost in k·n (`Sharing.perWeight`), the
/// line fitted under a disturbed figure (`CapacityFit.kept`), the tile and the route chosen by a written order under the noise
/// (`MatmulTile.choose`, `CapacitySolver.canonical`), the fallbacks a render names (`ProductLedger.Fallback`) and the bias each stage
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
    /// share and the pair leaves the candidates; at a 1 024² image's 4 096 it is a few percent and the pair stays. Without the term the
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
            let large = try #require(CS.product(ProductKey(k: k, n: n, m: 4096, weight: .bf16, phase: .denoising), model: model, constraints: c))
            #expect(large.route == "gpu+ane/bf16act", "\(k)x\(n)@4096: \(large.route)")
        }
        // the same map without the term: the pair a candidate again at 1 024, and its 256 ANE rows priced the copy's widening lower
        var without = model
        without.sharing["gpu/bf16act+ane/fp16"]?.perWeight = nil
        let small = ProductKey(k: 12288, n: 4096, m: 1024, weight: .bf16, phase: .denoising)
        #expect(!CS.candidates(small, model: model, constraints: c).contains { $0.route == "gpu+ane/bf16act" })
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

    /// **A figure the Mac disturbed is left out, not averaged in**: three points on t = 0.4 ms + W / 19.4 TFLOP/s, the middle one 60 %
    /// slow — the line is the other two's. A calm figure 10 % off stays (under the regime's spread), and two points are always kept.
    @Test func aDisturbedFigureLeavesTheLine() throws {
        typealias P = CapacityFit.Point
        let truth = { (w: Double) in 0.0004 + w / 19.4e12 }
        let ws = [2.0 * 1024 * 3840 * 3840, 2.0 * 2048 * 3840 * 3840, 2.0 * 4096 * 3840 * 3840]
        let disturbed = [P(flop: ws[0], seconds: truth(ws[0])), P(flop: ws[1], seconds: 1.6 * truth(ws[1])), P(flop: ws[2], seconds: truth(ws[2]))]
        #expect(CapacityFit.kept(disturbed) == [disturbed[0], disturbed[2]])
        let line = try #require(CapacityFit.line(disturbed))
        #expect(abs(line.rate / 19.4e12 - 1) < 1e-3 && abs(line.t0 - 0.0004) < 1e-6, "\(line)")
        let calm = [disturbed[0], P(flop: ws[1], seconds: 1.1 * truth(ws[1])), disturbed[2]]
        #expect(CapacityFit.kept(calm).count == 3)
        #expect(CapacityFit.kept(Array(disturbed.prefix(2))).count == 2)
    }

    // MARK: Under the noise, a written order

    /// **A tile within its rounds' spread of the fastest is the listed one**: 3 % faster with rounds 4 % apart is no gain; with rounds
    /// half a percent apart, the floor (5 %) still holds it; 12 % faster against a 4 % spread is.
    @Test func aTileWithinItsSpreadIsTheListedOne() {
        let names = ["64x32/4", "128x64/4"]
        #expect(MatmulTile.choose(["64x32/4": 10.3, "128x64/4": 10.0], among: names, standard: "64x32/4", spread: ["64x32/4": 0.04, "128x64/4": 0.04]) == "64x32/4")
        #expect(MatmulTile.choose(["64x32/4": 10.3, "128x64/4": 10.0], among: names, standard: "64x32/4", spread: ["64x32/4": 0.005, "128x64/4": 0.005]) == "64x32/4")
        #expect(MatmulTile.choose(["64x32/4": 11.2, "128x64/4": 10.0], among: names, standard: "64x32/4", spread: ["64x32/4": 0.04, "128x64/4": 0.04]) == "128x64/4")
        #expect(abs(MatmulTile.spread([0: [1.0, 1.1], 1: [2.0, 2.0]]) - (3.1 / 3.0 - 1)) < 1e-12)
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
