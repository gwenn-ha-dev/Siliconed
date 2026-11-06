import Foundation
import Testing
@testable import Siliconed

/// **The report of schema 25** (`DiagnosticReport.swift`): the JSON a user shares — sober, stable, read back whole —, the model it
/// holds being what its measurements fit to (the fleet's reports can be replayed offline), what an older schema or another build
/// designates (nothing), and what the plan of a current report excludes. Pure: no GPU, no weights.
@Suite struct DiagnosticReportTests {
    typealias R = ReportFixtures

    @Test func theSchemaIsSixteenAndAnOlderOneIsNotAMap() throws {
        #expect(Diagnostic.schema == 25)
        let report = R.report()
        #expect(report.schema == 25)
        let data = try report.json()
        #expect(Diagnostic.Report.read(data) == report)
        // Schema 14 and before were tables of routes: whatever else they hold, they are not read.
        let old = #"{"schema":14,"date":"2026-10-29T10:00:00Z","products":[],"map":{"classes":[]},"routes":{}}"#
        #expect(Diagnostic.Report.read(Data(old.utf8)) == nil)
        for text in ["", "{}", "[1]", #"{"schema":17}"#, #"{"schema":"16"}"#] { #expect(Diagnostic.Report.read(Data(text.utf8)) == nil, "\(text)") }
        // A report of a lower schema than 17 is never current, whatever the machine.
        var stale = report
        stale.schema = 16
        #expect(!stale.isCurrent)
    }

    /// A report crosses the remote control whole (`RemoteDiagnostic.finished`), in a frame of at most 1 MiB, and an app keeps it in its preferences.
    @Test func aReportStaysFarUnderAFrame() throws {
        #expect(try R.report().json(pretty: false).count < 60_000)
        #expect(try R.report().json(pretty: false).count < Frame.maximumPayload / 8)
    }

    @Test func theJSONIsStableAndReadsBackWhole() throws {
        let report = R.report()
        let a = try report.json(), b = try report.json()
        #expect(a == b)
        #expect(try JSONDecoder().decode(Diagnostic.Report.self, from: a) == report)
        #expect(try JSONDecoder().decode(Diagnostic.Report.self, from: report.json(pretty: false)) == report)
        let text = String(decoding: a, as: UTF8.self)
        #expect(text.contains("\"schema\" : 25"))
        // Sorted keys: `findings` before `judge` before `machine` before `measurements` before `model` before `stages` — and no verdict.
        let order = ["\"findings\"", "\"judge\"", "\"machine\"", "\"measurements\"", "\"model\"", "\"stages\""].map { text.range(of: $0)!.lowerBound }
        #expect(order == order.sorted())
        #expect(!text.contains("\"verdict\"") && !text.contains("\"green\""))
        // Nothing of the model-rendering diagnostics, nor of the table of routes, survives.
        for gone in ["\"models\"", "\"routes\"", "\"products\"", "\"decisions\"", "\"cuts\"", "\"map\"", "golden"] {
            #expect(!text.contains(gone), "\(gone)")
        }
    }

    /// A shared report reads as a person writes it: every dictionary keyed by what it holds, no array of alternating keys and values.
    @Test func theModelIsWrittenWithReadableKeys() throws {
        let report = R.report()
        let json = try #require(JSONSerialization.jsonObject(with: report.json()) as? [String: Any])
        let model = try #require(json["model"] as? [String: Any])
        let lines = try #require(model["lines"] as? [String: Any])
        #expect(Set(lines.keys) == Set(R.truth.keys.map(\.description)))
        let gpu = try #require(lines["gpu/fp32"] as? [String: Any])
        #expect(Set(gpu.keys) == ["t0", "rate"])
        let sharing = try #require(model["sharing"] as? [String: Any])
        #expect(Set(sharing.keys) == ["gpu+amx", "gpu+ane", "gpu+amx+ane", "amx+ane", "gpu+cpu"])
        let pair = try #require(sharing["gpu+ane"] as? [String: Any])
        #expect(Set((pair["keep"] as? [String: Any] ?? [:]).keys) == ["gpu", "ane"])
        #expect(Set((model["heat"] as? [String: Any] ?? [:]).keys) == ["gpu", "amx", "ane"])
        #expect(Set((model["crest"] as? [String: Any] ?? [:]).keys) == ["gpu"])
        let prep = try #require(model["preparation"] as? [String: [String: Double]])
        #expect(prep["int8"]?["widened"] != nil && prep["bf16"]?["bf16"] != nil)
        // No dictionary came out as an array of alternating keys and values.
        for key in ["lines", "sharing", "heat", "crest", "preparation"] { #expect(model[key] is [String: Any], "\(key)") }
        // Unknown names are refused, not dropped: a report from a build that has an engine this one lacks is not this one's.
        var bad = try #require(JSONSerialization.jsonObject(with: report.json(pretty: false)) as? [String: Any])
        var m = try #require(bad["model"] as? [String: Any])
        m["heat"] = ["npu": 1.2]
        bad["model"] = m
        let data = try JSONSerialization.data(withJSONObject: bad)
        #expect(throws: (any Error).self) { try JSONDecoder().decode(Diagnostic.Report.self, from: data) }
    }

    /// **The fit replayed from the measurements is the model**: what an offline reading of the fleet's reports relies on.
    @Test func theModelIsWhatTheMeasurementsFitTo() throws {
        let report = R.report()
        #expect(report.refit == report.model)
        // And after a trip through JSON, so that the figures a reader sees are the ones the engine planned with.
        let read = try #require(Diagnostic.Report.read(report.json()))
        #expect(read.refit == read.model && read.model == report.model)
        #expect(report.model.lines[R.gpu] != nil && report.model.lines[R.ane] != nil)
        // The lines are the ones the measurements were generated from, to the digits the report keeps.
        for (kernel, line) in R.truth {
            let fitted = try #require(report.model.lines[kernel], "\(kernel)")
            #expect(abs(fitted.rate / line.rate - 1) < 1e-3, "\(kernel)")
        }
        #expect(report.model.crest[.gpu] == Diagnostic.significant(14 * 128 * 2 * 1296e6, 6))
        #expect(report.model.heat[.gpu] ?? 0 > 1 && report.model.heat[.ane] ?? 0 >= 1)
        // Rounded: a six-digit model, five-digit measurements.
        #expect(report.model == report.model.rounded())
        #expect(report.measurements == report.measurements.rounded())
        #expect(report.measurements.solos.allSatisfy { $0.seconds == Diagnostic.significant($0.seconds, 5) })
    }

    @Test func aKernelTheJudgeRefusedLeavesTheFitAndThePlan() throws {
        let m = R.measurements(wrong: ["ane/fp16"])
        let report = R.report(measurements: m, judge: R.judge(excluded: ["ane/fp16"]))
        #expect(report.model.lines[R.ane] == nil)
        #expect(report.excluded == [R.ane])
        let plan = report.plan()
        #expect(!plan.usesANE)
        // The pair that includes it is fitted from no trial that computed wrong; the plan never offers the ANE rows.
        for m in [1024, 2048, 4096] { #expect(plan.assign(ProductKey(k: 3840, n: 3840, m: m)).ane == 0) }
        // A set judged wrong as a whole leaves the engines beside the GPU: the plan has no sharing for it, and would take it for free.
        var judge = R.judge()
        judge["gpu+amx"] = .init(error: 0.3, threshold: 1e-5, excluded: true)
        #expect(R.report(judge: judge).excluded == [R.amx])
    }

    @Test func theReportHoldsTheJudgeTheMemoryAndTheStagesWithTheirProjection() throws {
        let report = R.report()
        #expect(report.judge["ane/fp16"]?.threshold ?? 0 > Diagnostic.fp32Threshold, "the ANE is judged within 1.25 × bf16's rounding")
        #expect(report.judge["gpu/fp32"]?.excluded == false && report.passes)
        #expect(report.memory.held && report.memory.need == 2_970_000_000)
        #expect(report.stages["solos fp32"]?.projected == 2.1 && report.stages["setup"]?.projected == nil)
        var refused = R.judge()
        refused["gpu/q8"] = .init(error: 4e-2, threshold: 1e-5, excluded: true)
        #expect(!R.report(judge: refused).passes)
        // What each kernel reaches, against the peak the inventory read where it read one.
        let reach = report.reached
        let g = try #require(reach.first { $0.kernel == R.gpu })
        #expect(abs((g.fraction ?? 0) - 3.53e12 / (14 * 256 * 1.296e9)) < 1e-3)
        #expect(reach.first { $0.kernel == R.ane }?.fraction == nil, "no peak is read for the Neural Engine")
        #expect(reach.map(\.kernel.description) == reach.map(\.kernel.description).sorted { a, b in
            let (x, y) = (CapacityModel.Kernel(a)!, CapacityModel.Kernel(b)!)
            return (x.engine, x.form.rawValue) < (y.engine, y.form.rawValue)
        })
    }

    // ── what it designates ──────────────────────────────────────────────────────────────────

    /// **A report of this Mac and this build designates the plan of its model; any other designates nothing, and says why.**
    @Test func aCurrentReportDesignatesItsModelsPlan() throws {
        let report = R.report(machine: Diagnostic.machine())
        #expect(report.isCurrent)
        let taken = EngineSettings.designation(of: report)
        #expect(taken.warning == nil)
        let plan = try #require(taken.plan)
        #expect(plan.model == report.model)
        #expect(plan.identity.hasPrefix("model "))
        #expect(plan.usesANE && plan.usesAMX)
        // A process without the Neural Engine (the Standard, an encoder) restricts the same plan.
        #expect(!plan.restricted(allowingANE: false).usesANE)
        // The same report designates the same plan (the bits of a render depend on it).
        #expect(EngineSettings.designation(of: report).plan?.identity == plan.identity)
    }

    @Test func anotherFingerprintOrAnOlderSchemaIsIgnoredWithAWarning() throws {
        let elsewhere = R.report()                      // an M1 Pro of the fixture, not this Mac
        #expect(!elsewhere.isCurrent)
        let a = EngineSettings.designation(of: elsewhere)
        #expect(a.plan == nil)
        let warning = try #require(a.warning)
        #expect(warning.contains("ignored, the defaults apply") && warning.contains("schema 25"))
        var older = R.report(machine: Diagnostic.machine())
        older.schema = 14
        let b = EngineSettings.designation(of: older)
        #expect(b.plan == nil && b.warning?.contains("schema 14") == true)
        // Another build of the same Mac: an update measures again.
        var rebuilt = Diagnostic.machine()
        rebuilt.build = "another build"
        #expect(!R.report(machine: rebuilt).isCurrent)
    }

    @Test func theEnginesDefaultsStayWithoutAReport() {
        #expect(CapacityPlan.defaults.model == nil)
        #expect(CapacityPlan.defaults.assign(ProductKey(k: 3840, n: 3840, m: 4128)) == .gpuAlone(4128))
    }

    // ── the link ────────────────────────────────────────────────────────────────────────────

    @Test func theLastRendersNoteTravelsWithTheReportAndOnlyThere() throws {
        let report = R.report()
        #expect(report.render == nil)
        let request = CapacityClasses.Request(family: .zImage, includesEncoder: false)
        let note = RenderLoop.Note(request: request, map: report.date, comparison: .init(rows: []), verdict: .unjudged(.noModel))
        let carried = report.with(render: note)
        #expect(carried.render == note && report.render == nil)
        #expect(try Diagnostic.Report.read(carried.json())?.render == note)
    }
}
