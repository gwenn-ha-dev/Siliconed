import Foundation
import Testing
@testable import Siliconed

/// **Green or orange** (`DiagnosticVerdict.swift`): the rules, one by one, each on a report whose only fault is the one tested — and
/// that a kernel the judge refused, the heat, the battery or another process's swap are never a reason. Pure.
@Suite struct DiagnosticVerdictTests {
    typealias R = ReportFixtures
    typealias Fit = CapacityFit

    func defects(_ report: Diagnostic.Report) -> [String] { report.verdict.defects.map { "\($0.kind.rawValue) \($0.text)" } }

    @Test func aCleanMeasureIsGreen() {
        let report = R.report()
        #expect(report.verdict.green && report.verdict.defects.isEmpty)
        #expect(report.verdict.reasons.isEmpty)
        // …and so is a hot, fanless, low-power run: the regime is a fact, not a defect.
        var hot = R.measurements()
        hot.witnesses = [.init(engine: "gpu", before: 0.0172, after: 0.0301), .init(engine: "amx", before: 0.0262, after: 0.0498),
                         .init(engine: "ane", before: 0.0124, after: 0.0125)]
        let serious = Diagnostic.Conditions(thermal: "serious", lowPowerMode: true, onBattery: true, swapUsedBytes: 0)
        let report2 = R.report(measurements: hot, before: R.conditions(), after: serious)
        #expect(report2.verdict.green, "\(defects(report2))")
        #expect(report2.model.heat[.amx] ?? 0 > 1.8)
    }

    // ── every announced engine has its lines ────────────────────────────────────────────────

    @Test func anAnnouncedEngineWithoutALineIsNamed() {
        let noANE = R.report(measurements: R.measurements(dropping: ["ane/fp16"]))
        #expect(!noANE.verdict.green)
        #expect(defects(noANE).contains("line ane/fp16 has no line"))
        let noAMX = R.report(measurements: R.measurements(dropping: ["amx/fp32"]))
        #expect(defects(noAMX).contains("line amx/fp32 has no line"))
        let noHeads = R.report(measurements: R.measurements(dropping: ["cpu/heads"]))
        #expect(defects(noHeads).contains("line cpu/heads has no line"))
        // The forms are optional (a Mac without Metal 4's tensor product has no bf16 or int8 kernel): no defect.
        let noForms = R.report(measurements: R.measurements(dropping: ["gpu/bf16", "gpu/int8", "gpu/flash"]))
        #expect(noForms.verdict.green, "\(defects(noForms))")
    }

    /// An engine the inventory does not announce is not asked for (a virtualised runner has neither a clusters table nor an `ane0`).
    @Test func aMacWithoutAnInventoryIsJudgedOnWhatItHas() {
        let bare = R.machine(inventory: false)
        let gpuOnly = R.measurements(dropping: ["ane/fp16", "amx/fp32"]).trials(of: ["gpu+amx", "gpu+ane", "gpu+amx+ane", "amx+ane"], removed: true)
        let report = R.report(machine: bare, measurements: gpuOnly)
        #expect(report.verdict.green, "\(defects(report))")
        let withANE = R.report(machine: R.machine(), measurements: gpuOnly)
        #expect(!withANE.verdict.green)
        let announced = Diagnostic.Verdict.announces(R.machine())
        #expect(announced.amx && announced.ane)
        #expect(Diagnostic.Verdict.announces(bare) == (amx: false, ane: false))
        #expect(Diagnostic.Verdict.required(bare).map(\.description) == ["gpu/fp32", "gpu/graph", "cpu/heads"])
        #expect(Diagnostic.Verdict.sets(R.machine()).map(CapacityModel.key) == ["gpu+amx", "gpu+ane", "gpu+amx+ane", "amx+ane", "gpu+cpu"])
        #expect(Diagnostic.Verdict.sets(bare).map(CapacityModel.key) == ["gpu+cpu"], "the attention is a set whatever the inventory")
    }

    /// **A kernel the judge refused is excluded, never a reason for orange** — except the reference engine, without which nothing is planned.
    @Test func aKernelJudgedWrongIsExcludedAndNeverAReasonForOrange() {
        let ane = R.report(measurements: R.measurements(wrong: ["ane/fp16"]), judge: R.judge(excluded: ["ane/fp16"]))
        #expect(ane.verdict.green, "\(defects(ane))")
        #expect(ane.excluded == [R.ane])
        let amx = R.report(measurements: R.measurements(wrong: ["amx/fp32"]), judge: R.judge(excluded: ["amx/fp32"]))
        #expect(amx.verdict.green, "\(defects(amx))")
        let reference = R.report(measurements: R.measurements(wrong: ["gpu/fp32"]), judge: R.judge(excluded: ["gpu/fp32"]))
        #expect(defects(reference).contains { $0.hasPrefix("line gpu/fp32 was judged wrong") })
    }

    @Test func aSetThatExistsHasItsSharing() {
        var m = R.measurements()
        m.trials.removeAll { $0.set == "gpu+ane" }
        let report = R.report(measurements: m)
        #expect(defects(report) == ["sharing gpu+ane has no sharing"])
        // Every set the solver may offer is one: the AMX beside the ANE without the GPU, and the attention's graph beside the CPU's heads.
        var noPair = R.measurements()
        noPair.trials.removeAll { $0.set == "amx+ane" }
        #expect(defects(R.report(measurements: noPair)) == ["sharing amx+ane has no sharing"])
        var noAttention = R.measurements()
        noAttention.trials.removeAll { $0.set == "gpu+cpu" }
        #expect(defects(R.report(measurements: noAttention)) == ["sharing gpu+cpu has no sharing"])
        // A set whose member was excluded does not exist: no sharing is asked of it.
        var noANE = R.measurements(wrong: ["ane/fp16"])
        noANE.trials.removeAll { $0.set.contains("ane") }
        #expect(R.report(measurements: noANE, judge: R.judge(excluded: ["ane/fp16"])).verdict.green)
    }

    // ── nothing aberrant ────────────────────────────────────────────────────────────────────

    /// A negative intercept, five times past what a healthy two-point line shows, is a line the points do not sit on.
    @Test func aLineWithANegativeInterceptIsAberrant() throws {
        var m = R.measurements()
        for i in m.solos.indices where m.solos[i].kernel == "amx/fp32" { m.solos[i].seconds = m.solos[i].flop / 2.3e12 - 0.001 }
        let points = m.solos.filter { $0.kernel == "amx/fp32" }.map { Fit.Point(flop: $0.flop, seconds: $0.seconds) }
        let raw = try #require(Fit.rawIntercept(points))
        #expect(abs(raw + 0.001) < 1e-9, "the raw fit, not the clamped one, is read")
        let report = R.report(measurements: m)
        #expect(report.model.lines[R.amx]?.t0 == 0, "the model itself carries the clamped line")
        #expect(defects(report).contains { $0.hasPrefix("aberrant amx/fp32 has a fixed cost of -1.00 ms") }, "\(defects(report))")
    }

    /// The healthy figures of the probes' three runs are far inside the rule: the two-point attention lines come out a hair under 0.
    @Test func theHealthyTwoPointLinesAreNotAberrant() {
        var m = R.measurements()
        for i in m.solos.indices where m.solos[i].kernel == "cpu/heads" { m.solos[i].seconds = m.solos[i].flop / 1.0e12 + (i % 2 == 0 ? -0.0006 : -0.0004) }
        for i in m.solos.indices where m.solos[i].kernel == "gpu/graph" { m.solos[i].seconds = m.solos[i].flop / 1.39e12 + 0.00003 * (i % 2 == 0 ? -1 : 1) }
        let report = R.report(measurements: m)
        #expect(report.verdict.green, "\(defects(report))")
    }

    @Test func aRateOutsideTheBandOfThePeakReadIsAberrant() {
        let peak = 14.0 * 256 * 1.296e9
        func with(_ kernel: String, rate: Double) -> Diagnostic.Report {
            var m = R.measurements()
            for i in m.solos.indices where m.solos[i].kernel == kernel { m.solos[i].seconds = m.solos[i].flop / rate }
            return R.report(measurements: m)
        }
        // Over 1.2 × the peak, for a kernel that computes with the shader cores' fp32 FMA alone: the clock read or the timing is wrong.
        let fast = with("gpu/fp32", rate: 1.3 * peak)
        #expect(defects(fast).contains { $0.hasPrefix("aberrant gpu/fp32 reaches 1.30 of the peak read (over 1.2)") }, "\(defects(fast))")
        #expect(with("gpu/fp32", rate: 1.15 * peak).verdict.green, "within the band")
        // The tensor forms may run on matrix units of their own: only the lower bound applies.
        #expect(with("gpu/bf16", rate: 2.5 * peak).verdict.green)
        // Under 0.05: the kernel did not run on the silicon.
        let slow = with("gpu/int8", rate: 0.03 * peak)
        #expect(defects(slow).contains { $0.hasPrefix("aberrant gpu/int8 reaches 0.03 of the peak read (under 0.05)") }, "\(defects(slow))")
        // The engines that read no peak have no bound: the ANE reached 1.06 × the GPU's fp32 peak here.
        #expect(with("ane/fp16", rate: 30e12).verdict.green)
        #expect(with("amx/fp32", rate: 0.05e12).verdict.green)
        // Without an inventory there is no peak to read it against.
        var m = R.measurements()
        for i in m.solos.indices where m.solos[i].kernel == "gpu/fp32" { m.solos[i].seconds = m.solos[i].flop / (5 * peak) }
        #expect(R.report(machine: R.machine(inventory: false), measurements: m).verdict.green)
    }

    /// A staircase read through a disturbed minute: the largest unit that explains the noise.
    @Test func aStaircaseOfTheWholeRangeIsNoStaircase() {
        var m = R.measurements()
        // The stairs of a minute another job shared (ms at 768, 896, 1 024, 1 792, 1 920, 2 048, 2 304 and 4 096 rows).
        let disturbed = [19.55, 16.52, 21.60, 26.92, 16.90, 33.59, 41.87, 52.71]
        m.stairs = [768, 896, 1024, 1792, 1920, 2048, 2304, 4096].enumerated().map { .init(rows: $1, seconds: disturbed[$0] * 1e-3) }
        let report = R.report(measurements: m)
        let unit = report.model.aneRowUnit
        #expect(unit > Diagnostic.Verdict.Rules.maximumANEUnit, "the fixture's stairs fit a unit of \(unit)")
        #expect(defects(report).contains { $0.hasPrefix("aberrant ane/fp16 has a staircase of \(unit) rows (over 1024)") }, "\(defects(report))")
        #expect(R.report().model.aneRowUnit <= Diagnostic.Verdict.Rules.maximumANEUnit && R.report().verdict.green)
    }

    @Test func aKeepAtTheFitsClampIsNotAMeasure() {
        var m = R.measurements()
        // A side that took ten times as long within the set as alone: the fit's keep clamps at 0.05.
        for i in m.trials.indices where m.trials[i].set == "gpu+ane" {
            for j in m.trials[i].sides.indices where m.trials[i].sides[j].kernel == "ane/fp16" {
                m.trials[i].sides[j].within = (m.trials[i].sides[j].within ?? 0) * 40
            }
        }
        let report = R.report(measurements: m)
        #expect(defects(report).contains { $0.hasPrefix("aberrant gpu+ane keeps 0.05 of the ane's rate") }, "\(defects(report))")
        // The fleet's engines kept 0.34–1.03 of their solo rate in a set, this Mac's 0.65–0.95: all inside.
        #expect(R.report().model.sharing.values.allSatisfy { $0.keep.values.allSatisfy { Diagnostic.Verdict.Rules.keepRange.contains($0) } })
    }

    // ── the memory, the swap, the minute ────────────────────────────────────────────────────

    @Test func theMemoryMustHold() {
        // A peak over the base and the need is a defect of the code.
        let over = R.report(memory: .init(budget: 10_737_418_240, base: 60_000_000, need: 2_000_000_000, peak: 2_300_000_000))
        #expect(!over.verdict.green && defects(over).first?.hasPrefix("memory peak 2300 MB is over the need 2060 MB") == true, "\(defects(over))")
        // A need over B is a plan that should have been refused.
        let big = R.report(memory: .init(budget: 2_000_000_000, base: 60_000_000, need: 2_970_000_000, peak: 2_400_000_000))
        #expect(defects(big).first?.hasPrefix("memory need 3030 MB is over B 2000 MB") == true, "\(defects(big))")
        #expect(R.report(memory: .init(budget: 10_737_418_240, base: 60_000_000, need: 2_970_000_000, peak: 3_030_000_000)).verdict.green,
                "a peak exactly at the base and the need holds")
    }

    @Test func aSwapThatGrewIsADefectAndOneThatDidNotIsNot() {
        let grew = R.report(before: R.conditions(swap: 1_000_000_000), after: R.conditions(swap: 1_800_000_000))
        #expect(defects(grew) == ["swap swap grew from 1000 MB to 1800 MB"])
        #expect(R.report(before: R.conditions(swap: 7_000_000_000), after: R.conditions(swap: 7_000_000_000)).verdict.green,
                "another process's swap, already there, is a fact of the regime")
        #expect(R.report(before: R.conditions(swap: 7_000_000_000), after: R.conditions(swap: 6_000_000_000)).verdict.green)
    }

    @Test func theMinuteIsHeldWithin2Percent() {
        #expect(R.report(seconds: 61.1).verdict.green)
        let late = R.report(seconds: 61.3)
        #expect(defects(late) == ["time seconds took 61.3 s of 60"])
        #expect(R.report(seconds: 57.6).verdict.green, "the Neo's old map: 57.6 s")
    }

    @Test func severalDefectsAreAllSaid() {
        var m = R.measurements(dropping: ["ane/fp16"])
        m.trials.removeAll { $0.set.contains("ane") }
        let report = R.report(measurements: m, after: R.conditions(swap: 5), seconds: 70)
        let kinds = Set(report.verdict.defects.map(\.kind))
        #expect(kinds == [.line, .swap, .time])
        #expect(report.verdict.reasons.count == 3)
        #expect(!report.isGreen)
    }
}

extension CapacityFit.Measurements {
    /// The same measurements without the trials of these sets (or only with them).
    func trials(of sets: Set<String>, removed: Bool) -> CapacityFit.Measurements {
        var m = self
        m.trials = trials.filter { removed ? !sets.contains($0.set) : sets.contains($0.set) }
        return m
    }
}
