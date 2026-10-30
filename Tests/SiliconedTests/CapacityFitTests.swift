import Foundation
import Testing
@testable import Siliconed

@Suite struct CapacityFitTests {
    typealias F = CapacityFit
    typealias M = CapacityModel

    @Test func aLineIsFoundFromItsOwnPoints() throws {
        let truth = M.Line(t0: 0.0008, rate: 2.5e12)
        let pts = [4.3e9, 30e9, 300e9].map { F.Point(flop: $0, seconds: truth.seconds($0)) }
        let l = try #require(F.line(pts))
        #expect(abs(l.t0 - truth.t0) < 1e-9)
        #expect(abs(l.rate / truth.rate - 1) < 1e-9)
    }

    @Test func smallPointsWeighAsMuchAsBigOnes() throws {
        // 2 % noise on the big point must not drown t₀: the fit is on the relative error.
        let truth = M.Line(t0: 0.001, rate: 3e12)
        var pts = [4.3e9, 30e9, 300e9].map { F.Point(flop: $0, seconds: truth.seconds($0)) }
        pts[2].seconds *= 1.02
        let l = try #require(F.line(pts))
        #expect(F.relativeError(l, pts).worst < 0.02)
        #expect(abs(l.t0 - 0.001) < 0.0008)
    }

    @Test func aNegativeT0IsClampedAndTheRateRefitted() throws {
        let pts = [F.Point(flop: 1e10, seconds: 0.002), F.Point(flop: 1e11, seconds: 0.03), F.Point(flop: 1e12, seconds: 0.3)]
        let l = try #require(F.line(pts))
        #expect(l.t0 >= 0 && l.rate > 0)
    }

    @Test func oneUnsortedPointStillGivesARate() throws {
        let l = try #require(F.line([F.Point(flop: 1e10, seconds: 0.005)]))
        #expect(l.t0 == 0 && abs(l.rate - 2e12) < 1)
    }

    @Test func theAneUnitIsReadFromTheStairs() throws {
        // Rows cost by units of 512: 768 costs 1 024, 1 792 costs 2 048, 2 304 costs 2 560.
        func t(_ r: Int) -> Double { 0.002 + 3e-6 * Double((r + 511) / 512 * 512) }
        let rows = [768, 1024, 1792, 2048, 2304, 4096]
        let s = try #require(F.stairs(rows.map { ($0, t($0)) }))
        #expect(s.unit == 512)
        #expect(abs(s.t0 - 0.002) < 1e-9 && abs(s.perRow - 3e-6) < 1e-12)
    }

    @Test func aUnitOf256IsKeptWhenOddMultiplesAreTheirOwn() throws {
        func t(_ r: Int) -> Double { 0.002 + 3e-6 * Double((r + 255) / 256 * 256) }
        let s = try #require(F.stairs([768, 1024, 1792, 2048, 2304, 4096].map { ($0, t($0)) }))
        #expect(s.unit == 256)
    }

    @Test func sharingKeepAndOverheadAreRecoveredAndSeparated() throws {
        // Truth: the GPU keeps 0.8, the ANE 0.6, the set pays 3 ms once — at three sizes.
        let gpu = M.Kernel(.gpu, .fp32), ane = M.Kernel(.ane, .fp16)
        let lines: [M.Kernel: M.Line] = [gpu: .init(t0: 0.0008, rate: 3.2e12), ane: .init(t0: 0.002, rate: 5e12)]
        var truth = M()
        truth.lines = lines
        truth.sharing["gpu+ane"] = .init(overhead: 0.003, keep: [.gpu: 0.8, .ane: 0.6])
        var trials: [F.Measurements.Trial] = []
        for m in [1024, 2048, 4096] {
            let (k, n) = (3840, 3840)
            let aneRows = m / 2 / 256 * 256, gpuRows = m - aneRows
            let side = { (kernel: M.Kernel, rows: Int, keep: Double) -> F.Measurements.Side in
                let solo = lines[kernel]!.seconds(2 * Double(k) * Double(n) * Double(rows))
                let within = lines[kernel]!.t0 + (solo - lines[kernel]!.t0) / keep
                return .init(kernel: kernel.description, rows: rows, solo: solo, within: within)
            }
            let g = side(gpu, gpuRows, 0.8), a = side(ane, aneRows, 0.6)
            let wall = max(g.within!, a.within!) + 0.003
            trials.append(.init(set: "gpu+ane", shape: [m, k, n], sides: [g, a], wall: wall))
        }
        let fit = try #require(F.sharing(set: "gpu+ane", trials: trials, lines: lines))
        #expect(abs(fit.sharing.keep[.gpu]! - 0.8) < 1e-9)
        #expect(abs(fit.sharing.keep[.ane]! - 0.6) < 1e-9)
        #expect(abs(fit.sharing.overhead - 0.003) < 1e-9)
        #expect(fit.residuals.allSatisfy { abs($0) < 1e-9 })
    }

    @Test func withoutASideClockTheSideEndsWithTheSet() throws {
        let gpu = M.Kernel(.gpu, .fp32)
        let lines: [M.Kernel: M.Line] = [gpu: .init(t0: 0, rate: 3e12)]
        let solo = lines[gpu]!.seconds(2 * 1024 * 3840 * 3840)
        let trial = F.Measurements.Trial(set: "gpu+amx", shape: [1024, 3840, 3840], sides: [.init(kernel: "gpu/fp32", rows: 1024, solo: solo)], wall: solo * 1.25)
        let fit = try #require(F.sharing(set: "gpu+amx", trials: [trial], lines: lines))
        #expect(abs(fit.sharing.keep[.gpu]! - 0.8) < 1e-9)
        #expect(fit.sharing.overhead == 0)
    }

    @Test func theModelCollectsEverythingAndRoundTrips() throws {
        var m = F.Measurements()
        let l = M.Line(t0: 0.001, rate: 3e12)
        for flop in [4.3e9, 30e9, 300e9] { m.solos.append(.init(kernel: "gpu/fp32", shape: [1, 1, 1], flop: flop, seconds: l.seconds(flop))) }
        m.solos.append(.init(kernel: "gpu/q8", shape: [1, 1, 1], flop: 1e10, seconds: 0.001))
        m.wrong = ["gpu/q8"]
        m.witnesses = [.init(engine: "gpu", before: 0.010, after: 0.012), .init(engine: "ane", before: 0.02, after: 0.019)]
        m.memoryBandwidth = 2e11
        func t(_ r: Int) -> Double { 0.002 + 3e-6 * Double((r + 255) / 256 * 256) }
        m.stairs = [768, 1024, 1792, 2048, 2304, 4096].map { .init(rows: $0, seconds: t($0)) }
        let model = F.model(from: m)
        #expect(model.lines.keys.map(\.description) == ["gpu/fp32"])
        #expect(abs(model.heat[.gpu]! - 1.2) < 1e-9)
        #expect(model.heat[.ane] == 1)
        #expect(model.aneRowUnit == 256)
        #expect(model.memoryBandwidth == 2e11)
        let back = try JSONDecoder().decode(F.Measurements.self, from: JSONEncoder().encode(m))
        #expect(back == m)
    }
}

extension CapacityFitTests {
    @Test func theModelsUnitIsNeverFinerThanWhatAnEngineAllocates() throws {
        // The ANE charges per 128 rows here (linear down to the half-unit), but its rows are given in 256s.
        var m = F.Measurements()
        m.stairs = [768, 896, 1024, 1792, 1920, 2048, 2304, 4096].map { .init(rows: $0, seconds: 0.001 + 6e-6 * Double($0)) }
        #expect(F.stairs(m.stairs.map { ($0.rows, $0.seconds) })?.unit == 128)
        #expect(F.model(from: m).aneRowUnit == 256)
        #expect(F.model(from: m, allocationUnit: 128).aneRowUnit == 128)
    }

    @Test func theRawInterceptIsTheFitsOwnBeforeItsClamp() throws {
        // Exactly affine with a negative intercept: the line is clamped, the raw intercept is not.
        let pts = [4.3e9, 30e9, 300e9].map { F.Point(flop: $0, seconds: $0 / 2.3e12 - 0.001) }
        #expect(abs(try #require(F.rawIntercept(pts)) + 0.001) < 1e-9)
        #expect(try #require(F.line(pts)).t0 == 0)
        // A positive one is the line's own.
        let truth = M.Line(t0: 0.0008, rate: 2.5e12)
        let good = [4.3e9, 30e9, 300e9].map { F.Point(flop: $0, seconds: truth.seconds($0)) }
        #expect(abs(try #require(F.rawIntercept(good)) - 0.0008) < 1e-9)
        #expect(F.rawIntercept([good[0]]) == nil && F.rawIntercept([]) == nil)
    }

    /// A set that computed wrong is a trial with no side: it fills no sharing (the report excludes the set), it does not fill a free one.
    @Test func aSetThatComputedWrongFitsNothing() {
        var m = F.Measurements()
        m.trials = [.init(set: "gpu+ane", shape: [2048, 3840, 3840], pass: 2, sides: [], wall: 0, error: 0.4)]
        #expect(F.model(from: m).sharing["gpu+ane"] == nil)
    }

    @Test func measurementsRoundToTheDigitsAReportKeepsAndStayExact() {
        var m = F.Measurements()
        m.solos = [.init(kernel: "gpu/fp32", shape: [512, 2048, 2048], flop: 4_294_967_296, seconds: 0.0012195121951, error: 1.23456e-6)]
        m.witnesses = [.init(engine: "gpu", before: 0.017123456789, after: 0.018123456789)]
        m.memoryBandwidth = 187_632_361_492.06
        m.preparation = ["int8": ["widened": 1.3952839402818476e-10]]
        let r = m.rounded()
        #expect(r.solos[0].flop == 4_294_967_296, "a FLOP count is exact")
        #expect(r.solos[0].seconds == 0.0012195 && r.solos[0].error == 1.23e-6)
        #expect(r.witnesses[0].before == 0.017123 && r.memoryBandwidth == 187_630_000_000)
        #expect(r.preparation["int8"]?["widened"] == 1.3953e-10)
        #expect(r.rounded() == r)
    }

    /// **A route over its bar on the massive activation is wrong, and a wrong route has no line**: the probes name it in `wrong`
    /// (as for any product), the fit gives it none, and the plan of that model does not build the ANE's int8 program. The record of
    /// the judge survives its text.
    @Test func aRouteJudgedWrongOnTheMassiveActivationLeavesNoLineAndNoProgram() throws {
        var m = F.Measurements()
        let line = M.Line(t0: 0.001, rate: 3e12)
        for kernel in ["gpu/fp32", "gpu/int8", "ane/fp16", "ane/int8"] {
            for flop in [4.3e9, 30e9, 300e9] { m.solos.append(.init(kernel: kernel, shape: [1, 1, 1], flop: flop, seconds: line.seconds(flop))) }
        }
        #expect(CapacityPlan(model: F.model(from: m)).resources.aneInt8)
        m.wrong = ["ane/int8"]
        let model = F.model(from: m)
        #expect(model.lines[M.Kernel(.ane, .int8)] == nil && model.lines[M.Kernel(.ane, .fp16)] != nil)
        let plan = CapacityPlan(model: model)
        #expect(plan.resources.ane && !plan.resources.aneInt8)
        let massive = Diagnostic.CapacityProbes.Massive(kernel: "ane/int8", shape: [1024, 3840, 3840], columns: [548, 1280, 1925, 3829],
                                                        rowError: 3.3e-4, rowFloor: 3.5e-3, columnError: 1.4e-3, columnFloor: 3.3e-3, pass: true)
        let back = try JSONDecoder().decode(Diagnostic.CapacityProbes.Massive.self, from: JSONEncoder().encode(massive))
        #expect(back.kernel == massive.kernel && back.columns == massive.columns && back.columnError == massive.columnError && back.pass)
    }
}
