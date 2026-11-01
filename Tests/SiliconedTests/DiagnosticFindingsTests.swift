import Foundation
import Testing
@testable import Siliconed

/// **What a measure finds** (`DiagnosticFindings.swift`): the rules, one by one, each on a report whose only fault is the one tested — and
/// that a kernel the judge refused, the heat, the battery or another process's swap are never a finding. A finished measure is a map
/// whatever it finds: each finding says what the plan does with it, and no test here asks for a grade. Pure.
@Suite struct DiagnosticFindingsTests {
    typealias R = ReportFixtures
    typealias Fit = CapacityFit

    func found(_ report: Diagnostic.Report) -> [String] { report.findings.map { "\($0.kind.rawValue) \($0.text)" } }

    @Test func aCleanMeasureFindsNothing() {
        let report = R.report()
        #expect(report.findings.isEmpty, "\(found(report))")
        // …and so does a hot, fanless, low-power run: the regime is a fact written beside the model, not a finding.
        var hot = R.measurements()
        hot.witnesses = [.init(engine: "gpu", before: 0.0172, after: 0.0301), .init(engine: "amx", before: 0.0262, after: 0.0498),
                         .init(engine: "ane", before: 0.0124, after: 0.0125)]
        let serious = Diagnostic.Conditions(thermal: "serious", lowPowerMode: true, onBattery: true, swapUsedBytes: 0)
        let report2 = R.report(measurements: hot, before: R.conditions(), after: serious)
        #expect(report2.findings.isEmpty, "\(found(report2))")
        #expect(report2.model.heat[.amx] ?? 0 > 1.8)
    }

    // ── every announced engine has its lines ────────────────────────────────────────────────

    @Test func anAnnouncedEngineWithoutALineIsNamedAndNotOffered() {
        let noANE = R.report(measurements: R.measurements(dropping: ["ane/fp16"]))
        #expect(found(noANE).contains("line ane/fp16 has no line: it is not offered to the plan"))
        #expect(noANE.findings.first { $0.subject == "ane/fp16" }?.effect == .leftOut)
        let noAMX = R.report(measurements: R.measurements(dropping: ["amx/fp32"]))
        #expect(found(noAMX).contains("line amx/fp32 has no line: it is not offered to the plan"))
        let noHeads = R.report(measurements: R.measurements(dropping: ["cpu/heads"]))
        #expect(found(noHeads).contains("line cpu/heads has no line: it is not offered to the plan"))
        // The forms are optional (a Mac without Metal 4's tensor product has no bf16 or int8 kernel): nothing to say.
        let noForms = R.report(measurements: R.measurements(dropping: ["gpu/bf16", "gpu/int8", "gpu/flash"]))
        #expect(noForms.findings.isEmpty, "\(found(noForms))")
        // The map of this Mac designates its plan all the same, and the engine it has no line for is not in it.
        let mine = R.report(machine: Diagnostic.machine(), measurements: R.measurements(dropping: ["ane/fp16"]))
        let designated = EngineSettings.designation(of: mine)
        #expect(designated.plan != nil && designated.warning == nil)
        #expect(designated.plan?.resources.ane == false)
    }

    /// An engine the inventory does not announce is not asked for (a virtualised runner has neither a clusters table nor an `ane0`).
    @Test func aMacWithoutAnInventoryIsReadOnWhatItHas() {
        let bare = R.machine(inventory: false)
        let gpuOnly = R.measurements(dropping: ["ane/fp16", "amx/fp32"]).trials(of: ["gpu+amx", "gpu+ane", "gpu+amx+ane", "amx+ane"], removed: true)
        let report = R.report(machine: bare, measurements: gpuOnly)
        #expect(report.findings.isEmpty, "\(found(report))")
        let withANE = R.report(machine: R.machine(), measurements: gpuOnly)
        #expect(!withANE.findings.isEmpty)
        let announced = Diagnostic.Findings.announces(R.machine())
        #expect(announced.amx && announced.ane)
        #expect(Diagnostic.Findings.announces(bare) == (amx: false, ane: false))
        #expect(Diagnostic.Findings.required(bare).map(\.description) == ["gpu/fp32", "gpu/graph", "cpu/heads"])
        #expect(Diagnostic.Findings.sets(R.machine()).map(CapacityModel.key) == ["gpu+amx", "gpu+ane", "gpu+amx+ane", "amx+ane", "gpu+cpu"])
        #expect(Diagnostic.Findings.sets(bare).map(CapacityModel.key) == ["gpu+cpu"], "the attention is a set whatever the inventory")
    }

    /// **A kernel the judge refused is excluded, and says nothing more** — except the reference engine, which the plan cannot do without:
    /// its defaults answer, and the finding says so.
    @Test func aKernelJudgedWrongIsExcludedAndNeverAFinding() {
        let ane = R.report(measurements: R.measurements(wrong: ["ane/fp16"]), judge: R.judge(excluded: ["ane/fp16"]))
        #expect(ane.findings.isEmpty, "\(found(ane))")
        #expect(ane.excluded == [R.ane])
        let amx = R.report(measurements: R.measurements(wrong: ["amx/fp32"]), judge: R.judge(excluded: ["amx/fp32"]))
        #expect(amx.findings.isEmpty, "\(found(amx))")
        let reference = R.report(measurements: R.measurements(wrong: ["gpu/fp32"]), judge: R.judge(excluded: ["gpu/fp32"]))
        #expect(found(reference).contains { $0.hasPrefix("line gpu/fp32 was judged wrong: the plan has no model of it") }, "\(found(reference))")
    }

    /// **A set named by its kernels and judged wrong leaves the kernels beside the GPU out** — the integers' pair (`gpu/int8+ane/int8`)
    /// as a set of engines (`gpu+ane`) does: read as engines, its name matched nothing and the ANE's integers stayed in the plan.
    @Test func aPairOfKernelsJudgedWrongLeavesTheANEsKernelOut() {
        let wrong = ["gpu/int8+ane/int8": Diagnostic.Judge(error: 1, threshold: 0.01, excluded: true)]
        #expect(Diagnostic.excluded(wrong) == [CapacityModel.Kernel(.ane, .int8)])
        let engines = ["gpu+ane": Diagnostic.Judge(error: 1, threshold: 0.01, excluded: true)]
        #expect(Diagnostic.excluded(engines) == [CapacityModel.Kernel(.ane, .fp16)])
    }

    @Test func aSetThatExistsHasItsSharingOrIsNotOffered() {
        var m = R.measurements()
        m.trials.removeAll { $0.set == "gpu+ane" }
        let report = R.report(measurements: m)
        #expect(found(report) == ["sharing gpu+ane has no sharing: the set is not offered to the plan"])
        #expect(report.findings.first?.effect == .leftOut)
        #expect(!report.model.measures([.gpu, .ane]), "the solver offers no set that was not measured")
        // Every set the solver may offer is one: the AMX beside the ANE without the GPU, and the attention's graph beside the CPU's heads.
        var noPair = R.measurements()
        noPair.trials.removeAll { $0.set == "amx+ane" }
        #expect(found(R.report(measurements: noPair)) == ["sharing amx+ane has no sharing: the set is not offered to the plan"])
        var noAttention = R.measurements()
        noAttention.trials.removeAll { $0.set == "gpu+cpu" }
        #expect(found(R.report(measurements: noAttention)) == ["sharing gpu+cpu has no sharing: the set is not offered to the plan"])
        // A set whose member was excluded does not exist: no sharing is asked of it.
        var noANE = R.measurements(wrong: ["ane/fp16"])
        noANE.trials.removeAll { $0.set.contains("ane") }
        #expect(R.report(measurements: noANE, judge: R.judge(excluded: ["ane/fp16"])).findings.isEmpty)
    }

    // ── nothing aberrant is used as it is ───────────────────────────────────────────────────

    /// A negative intercept, five times past what a healthy two-point line shows, is a line the points do not sit on: taken as 0.
    @Test func aLineWithANegativeInterceptIsBounded() throws {
        var m = R.measurements()
        for i in m.solos.indices where m.solos[i].kernel == "amx/fp32" { m.solos[i].seconds = m.solos[i].flop / 2.3e12 - 0.001 }
        let points = m.solos.filter { $0.kernel == "amx/fp32" }.map { Fit.Point(flop: $0.flop, seconds: $0.seconds) }
        let raw = try #require(Fit.rawIntercept(points))
        #expect(abs(raw + 0.001) < 1e-9, "the raw fit, not the clamped one, is read")
        let report = R.report(measurements: m)
        #expect(report.model.lines[R.amx]?.t0 == 0, "the model itself carries the clamped line")
        #expect(found(report).contains { $0.hasPrefix("aberrant amx/fp32 has a fixed cost of -1.00 ms, below zero: taken as 0") }, "\(found(report))")
        #expect(report.findings.first { $0.subject == "amx/fp32" }?.effect == .bounded)
    }

    /// The healthy figures of the probes' three runs are far inside the rule: the two-point attention lines come out a hair under 0.
    @Test func theHealthyTwoPointLinesFindNothing() {
        var m = R.measurements()
        for i in m.solos.indices where m.solos[i].kernel == "cpu/heads" { m.solos[i].seconds = m.solos[i].flop / 1.0e12 + (i % 2 == 0 ? -0.0006 : -0.0004) }
        for i in m.solos.indices where m.solos[i].kernel == "gpu/graph" { m.solos[i].seconds = m.solos[i].flop / 1.39e12 + 0.00003 * (i % 2 == 0 ? -1 : 1) }
        let report = R.report(measurements: m)
        #expect(report.findings.isEmpty, "\(found(report))")
    }

    @Test func aRateOutsideTheBandOfThePeakReadIsSaidAndHandledAsItSays() {
        let peak = 14.0 * 256 * 1.296e9
        func with(_ kernel: String, rate: Double) -> Diagnostic.Report {
            var m = R.measurements()
            for i in m.solos.indices where m.solos[i].kernel == kernel { m.solos[i].seconds = m.solos[i].flop / rate }
            return R.report(measurements: m)
        }
        // Over 1.2 × the peak, for a kernel that computes with the shader cores' fp32 FMA alone: the clock read or the timing is wrong —
        // the plan uses the rate measured, and the finding says the peak is the one in doubt.
        let fast = with("gpu/fp32", rate: 1.3 * peak)
        #expect(found(fast).contains { $0.hasPrefix("aberrant gpu/fp32 reaches 1.30 of the peak read (over 1.2)") }, "\(found(fast))")
        #expect(fast.findings.first { $0.subject == "gpu/fp32" }?.effect == .fact && !fast.excluded.contains(R.gpu))
        #expect(with("gpu/fp32", rate: 1.15 * peak).findings.isEmpty, "within the band")
        // The tensor forms may run on matrix units of their own: only the lower bound applies.
        #expect(with("gpu/bf16", rate: 2.5 * peak).findings.isEmpty)
        // Under 0.05: the kernel did not run on the silicon, and the plan does not offer it.
        let slow = with("gpu/int8", rate: 0.03 * peak)
        #expect(found(slow).contains { $0.hasPrefix("aberrant gpu/int8 reaches 0.03 of the peak read (under 0.05): it is not offered to the plan") }, "\(found(slow))")
        #expect(slow.excluded.contains(CapacityModel.Kernel(.gpu, .int8)))
        // The engines that read no peak have no bound: the ANE reached 1.06 × the GPU's fp32 peak here.
        #expect(with("ane/fp16", rate: 30e12).findings.isEmpty)
        // (the sets are read on the lines: a line 40 times slower makes the AMX's sides keep more than 1, which is another rule's)
        #expect(!found(with("amx/fp32", rate: 0.05e12)).contains { $0.hasPrefix("aberrant amx/fp32 reaches") })
        // Without an inventory there is no peak to read it against.
        var m = R.measurements()
        for i in m.solos.indices where m.solos[i].kernel == "gpu/fp32" { m.solos[i].seconds = m.solos[i].flop / (5 * peak) }
        #expect(R.report(machine: R.machine(inventory: false), measurements: m).findings.isEmpty)
    }

    /// A staircase read through a disturbed minute: the largest unit that explains the noise — bounded to four of the engine's own.
    @Test func aStaircaseOfTheWholeRangeIsBounded() {
        var m = R.measurements()
        // The stairs of a minute another job shared (ms at 768, 896, 1 024, 1 792, 1 920, 2 048, 2 304 and 4 096 rows).
        let disturbed = [19.55, 16.52, 21.60, 26.92, 16.90, 33.59, 41.87, 52.71]
        m.stairs = [768, 896, 1024, 1792, 1920, 2048, 2304, 4096].enumerated().map { .init(rows: $1, seconds: disturbed[$0] * 1e-3) }
        let report = R.report(measurements: m)
        let unit = Diagnostic.Report.fit(m, machine: R.machine(), bounded: false).aneRowUnit
        #expect(unit > Diagnostic.Findings.Rules.maximumANEUnit, "the fixture's stairs fit a unit of \(unit)")
        #expect(found(report).contains { $0.hasPrefix("aberrant ane/fp16 has a staircase of \(unit) rows (over 1024): taken as 1024") }, "\(found(report))")
        #expect(report.model.aneRowUnit == Diagnostic.Findings.Rules.maximumANEUnit, "the plan rounds the ANE's rows to the bound, not to the whole range")
        #expect(report.refit == report.model, "the fit replayed from the measurements gives the bounded model back")
        #expect(R.report().model.aneRowUnit <= Diagnostic.Findings.Rules.maximumANEUnit && R.report().findings.isEmpty)
    }

    @Test func aKeepAtTheFitsClampIsBounded() {
        var m = R.measurements()
        // A side that took forty times as long within the set as alone: the fit's keep clamps at 0.05.
        for i in m.trials.indices where m.trials[i].set == "gpu+ane" {
            for j in m.trials[i].sides.indices where m.trials[i].sides[j].kernel == "ane/fp16" {
                m.trials[i].sides[j].within = (m.trials[i].sides[j].within ?? 0) * 40
            }
        }
        let report = R.report(measurements: m)
        #expect(found(report).contains { $0.hasPrefix("aberrant gpu+ane keeps 0.05 of the ane's rate: taken as 0.10") }, "\(found(report))")
        #expect(report.model.sharing["gpu+ane"]?.keep[.ane] == Diagnostic.Findings.Rules.keepRange.lowerBound, "the plan reads the band's edge")
        #expect(report.refit == report.model)
        // The fleet's engines kept 0.34–1.03 of their solo rate in a set, this Mac's 0.65–0.95: all inside.
        #expect(R.report().model.sharing.values.allSatisfy { $0.keep.values.allSatisfy { Diagnostic.Findings.Rules.keepRange.contains($0) } })
    }

    // ── the memory, the swap, the minute: facts of the run ──────────────────────────────────

    @Test func theMemoryIsWrittenAsItWas() {
        // A peak over the base and the need is a defect of the code, written as such.
        let over = R.report(memory: .init(budget: 10_737_418_240, base: 60_000_000, need: 2_000_000_000, peak: 2_300_000_000))
        #expect(found(over).first?.hasPrefix("memory peak 2300 MB is over the need 2060 MB: a defect of the code") == true, "\(found(over))")
        // A need over B is a plan that should have been refused.
        let big = R.report(memory: .init(budget: 2_000_000_000, base: 60_000_000, need: 2_970_000_000, peak: 2_400_000_000))
        #expect(found(big).first?.hasPrefix("memory need 3030 MB is over B 2000 MB") == true, "\(found(big))")
        #expect(R.report(memory: .init(budget: 10_737_418_240, base: 60_000_000, need: 2_970_000_000, peak: 3_030_000_000)).findings.isEmpty,
                "a peak exactly at the base and the need holds")
        #expect(over.findings[0].effect == .fact)
    }

    @Test func aSwapThatGrewIsWrittenAndOneThatDidNotIsNot() {
        let grew = R.report(before: R.conditions(swap: 1_000_000_000), after: R.conditions(swap: 1_800_000_000))
        #expect(found(grew) == ["swap swap grew from 1000 MB to 1800 MB"])
        #expect(R.report(before: R.conditions(swap: 7_000_000_000), after: R.conditions(swap: 7_000_000_000)).findings.isEmpty,
                "another process's swap, already there, is a fact of the regime")
        #expect(R.report(before: R.conditions(swap: 7_000_000_000), after: R.conditions(swap: 6_000_000_000)).findings.isEmpty)
    }

    @Test func theCeilingIsWrittenBeyond2Percent() {
        #expect(R.report(seconds: 305.9).findings.isEmpty)
        let late = R.report(seconds: 306.2)
        #expect(found(late) == ["time seconds took 306.2 s of 300"])
        #expect(R.report(seconds: 70).findings.isEmpty, "a minute and more is no finding: five minutes are the ceiling")
    }

    @Test func severalFindingsAreAllSaid() {
        var m = R.measurements(dropping: ["ane/fp16"])
        m.trials.removeAll { $0.set.contains("ane") }
        let report = R.report(measurements: m, after: R.conditions(swap: 5), seconds: 310)
        #expect(Set(report.findings.map(\.kind)) == [.line, .swap, .time])
        #expect(report.findings.count == 3)
        // …and the map is the map all the same: of this Mac, current, designated, its plan made. The live Mac may add findings of its
        // own (the CI's virtual M1, with 7 GB and three cores, does): the three are said among them.
        let mine = R.report(machine: Diagnostic.machine(), measurements: m, after: R.conditions(swap: 5), seconds: 310)
        let kinds: Set<Diagnostic.Finding.Kind> = Set(mine.findings.map(\.kind))
        #expect(kinds.isSuperset(of: [.line, .swap, .time]), "\(mine.findings.map(\.detail))")
        #expect(mine.isCurrent)
        #expect(EngineSettings.designation(of: mine).plan != nil)
    }

    // ── what an app says of a finding ───────────────────────────────────────────────────────

    /// Each finding has a cause and the engines it concerns, read from the kind and the subject — never from the English of `detail`.
    @Test func eachFindingHasItsCauseAndItsEngines() {
        func causes(_ report: Diagnostic.Report) -> [(String, Diagnostic.Cause)] {
            report.findings.map { ($0.subject, $0.cause(excluding: report.excluded)) }
        }
        var noANE = R.measurements(dropping: ["ane/fp16"])
        noANE.trials.removeAll { $0.set.contains("ane") }
        let missing = R.report(measurements: noANE)
        #expect(causes(missing).contains { $0.0 == "ane/fp16" && $0.1 == .noFigure })
        #expect(missing.findings.first?.engines == [.ane] && missing.findings.first?.kernel == R.ane)

        let reference = R.report(measurements: R.measurements(wrong: ["gpu/fp32"]), judge: R.judge(excluded: ["gpu/fp32"]))
        #expect(causes(reference).contains { $0.0 == "gpu/fp32" && $0.1 == .computedWrong })

        var m = R.measurements()
        m.trials.removeAll { $0.set == "gpu+ane" }
        let alone = R.report(measurements: m)
        let sharing = alone.findings[0]
        #expect(sharing.cause(excluding: alone.excluded) == .notShared)
        #expect(sharing.kernel == nil && sharing.engines == [.gpu, .ane])

        var slow = R.measurements()
        for i in slow.solos.indices where slow.solos[i].kernel == "gpu/int8" { slow.solos[i].seconds = slow.solos[i].flop / 1e11 }
        let aberrant = R.report(measurements: slow).findings[0]
        #expect(aberrant.cause(excluding: []) == .implausible && aberrant.engines == [.gpu])

        let over = R.report(memory: .init(budget: 10_737_418_240, base: 60_000_000, need: 2_000_000_000, peak: 2_300_000_000)).findings[0]
        let big = R.report(memory: .init(budget: 2_000_000_000, base: 60_000_000, need: 2_970_000_000, peak: 2_400_000_000)).findings[0]
        #expect(over.cause(excluding: []) == .memoryOverNeed && big.cause(excluding: []) == .memoryOverBudget)
        #expect(over.engines.isEmpty && over.kernel == nil)
        let swap = R.report(before: R.conditions(swap: 1), after: R.conditions(swap: 9)).findings[0]
        let late = R.report(seconds: 310).findings[0]
        #expect(swap.cause(excluding: []) == .swapGrew && late.cause(excluding: []) == .tookTooLong)
    }

    // ── how sure each figure is, and the fast side of a noisy chain ─────────────────────────

    /// A chain whose calls agree reads as it always did; one whose calls disagree is read on its quickest third — a disturbance only adds time.
    @Test func aNoisyChainIsReadOnItsFastSide() {
        typealias P = Diagnostic.CapacityProbes
        let calm = [20.0, 20.2, 19.9, 20.1, 20.0, 20.3, 19.8, 20.1].map { $0 * 1e-3 }
        #expect(P.figure(calm) == CapacityFit.median(calm), "calls that agree: the median, bit for bit")
        // Half the calls met another program: the median is the Mac's company, the quick third is the engine's.
        let noisy = [20.0, 20.1, 19.9, 20.2, 27.0, 28.5, 26.0, 29.0, 27.5, 20.0, 26.5, 28.0].map { $0 * 1e-3 }
        #expect(P.spread(noisy) > Diagnostic.Regime.Rules.spread)
        #expect(P.figure(noisy) < CapacityFit.median(noisy))
        #expect(abs(P.figure(noisy) - 20.0e-3) < 0.2e-3, "\(P.figure(noisy))")
        // Fewer than six calls say nothing of the Mac: never reread.
        let few = [20.0, 30.0, 25.0, 40.0].map { $0 * 1e-3 }
        #expect(P.figure(few) == CapacityFit.median(few))
        // The sides of a set are read from the same calls as its wall.
        let samples = noisy.enumerated().map { (wall: $1, within: ["ane/fp16": $1 * 0.9, "gpu/fp32": 18e-3 + Double($0) * 1e-5]) }
        let used = P.quick(samples)
        #expect(used.count == 4 && used.allSatisfy { $0.wall < 21e-3 })
        #expect(P.quick(calm.map { (wall: $0, within: [:]) }).count == calm.count)
    }

    /// The uncertainty of a figure is how far the identical calls of its chains agreed, the worst of them; the plus-or-minus is half of it.
    @Test func theUncertaintyOfAFigureIsTheSpreadOfItsChains() {
        var report = R.report()
        var regime = Diagnostic.Regime()
        regime.spreads = ["gpu+ane 4096x3840x3840 pass 1": 0.40, "gpu+ane 4096x12288x4096 pass 3": 0.10, "ane/fp16 1024x3840x3840": 0.03,
                          "gpu/int8+ane/int8 4096x3840x3840 pass 1": 0.30, "gpu/fp32 512x2048x2048": 0, "stairs 768": 0.5]
        regime.foreign = ["run": 0.45]
        report.regime = regime
        let u = Dictionary(uniqueKeysWithValues: report.uncertainty.map { ($0.subject, $0) })
        #expect(u["gpu+ane"]?.spread == 0.40 && u["gpu+ane"]?.plusMinus == 0.20)
        #expect(u["ane/fp16"]?.spread == 0.03)
        #expect(u["gpu/int8+ane/int8"]?.spread == 0.30, "the integers' pair is a subject of its own")
        #expect(u["gpu/fp32"] == nil && u["stairs"] == nil, "a spread of 0 (too few calls to compare) and a step that is no kernel say nothing")
        #expect(report.uncertainty.map(\.subject) == report.uncertainty.map(\.subject).sorted())
        #expect(report.foreignCores == 0.45)
        #expect(R.report().uncertainty.isEmpty && R.report().foreignCores == nil)
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
