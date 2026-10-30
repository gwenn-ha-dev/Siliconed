import Foundation
import Testing
@testable import Siliconed

/// **The measure in a process of its own, and the process that watches it** (`DiagnosticChild.swift`): a route whose load kills the
/// child is left out and the measure starts again without it; nothing else is a verdict on a route. The supervisor runs here against
/// real processes — `/bin/sh` scripts that say a stage and a route and then crash, or report — so that the death, the pipe and the
/// restart are the real ones, with no weights and no engine.
@Suite struct DiagnosticChildTests {
    typealias R = ReportFixtures
    typealias Child = Diagnostic.Child

    // ── what a death condemns ───────────────────────────────────────────────────────────────

    @Test func aCrashOnARouteCondemnsThatRoute() {
        let death = Child.Death(status: SIGSEGV, signaled: true, stage: .sharing, route: "ane/int8")
        #expect(Child.culprit(of: death, skipped: []) == "ane/int8")
        for crash in [SIGBUS, SIGILL, SIGABRT, SIGTRAP, SIGFPE] {
            #expect(Child.culprit(of: Child.Death(status: crash, signaled: true, route: "ane/fp16"), skipped: []) == "ane/fp16")
        }
    }

    /// Nothing but a crash is a word on a route: the user's stop, the system's need of the memory, a status the child chose.
    @Test func nothingButACrashCondemnsARoute() {
        for number in [SIGKILL, SIGTERM, SIGINT, SIGHUP] {
            #expect(Child.culprit(of: Child.Death(status: number, signaled: true, route: "ane/int8"), skipped: []) == nil, "signal \(number)")
        }
        // An exit status is the child's own word (an `error` line says what failed); 11 here is a status, not SIGSEGV.
        #expect(Child.culprit(of: Child.Death(status: 11, signaled: false, route: "ane/int8"), skipped: []) == nil)
        // No route named: nothing to leave out.
        #expect(Child.culprit(of: Child.Death(status: SIGSEGV, signaled: true, route: nil), skipped: []) == nil)
        #expect(Child.culprit(of: Child.Death(status: SIGSEGV, signaled: true, route: "not a kernel"), skipped: []) == nil)
    }

    /// The GPU's product and its attention are what everything is measured against; a route already left out cannot be the cause again.
    @Test func theReferenceIsNeverLeftOutAndNoRouteTwice() {
        for reference in ["gpu/fp32", "gpu/graph"] {
            #expect(Child.culprit(of: Child.Death(status: SIGSEGV, signaled: true, route: reference), skipped: []) == nil, "\(reference)")
        }
        #expect(Child.culprit(of: Child.Death(status: SIGSEGV, signaled: true, route: "ane/int8"), skipped: ["ane/int8"]) == nil)
        #expect(Child.culprit(of: Child.Death(status: SIGSEGV, signaled: true, route: "amx/fp32"), skipped: ["ane/int8"]) == "amx/fp32")
    }

    // ── what the child says ─────────────────────────────────────────────────────────────────

    @Test func theChildsLinesAreRead() throws {
        guard case .stage(.sharing)? = Child.Line("stage sharing") else { Issue.record("stage"); return }
        guard case .route("ane/int8")? = Child.Line("route ane/int8") else { Issue.record("route"); return }
        guard case .failure(let code, let detail)? = Child.Line("error render-already-running another render holds the machine") else {
            Issue.record("error"); return
        }
        #expect(code == "render-already-running" && detail == "another render holds the machine")
        let json = String(decoding: try R.report().json(pretty: false), as: UTF8.self)
        guard case .report(let report)? = Child.Line("report " + json) else { Issue.record("report"); return }
        #expect(report == R.report())
        #expect(Child.Line("stage not-a-stage") == nil)
        #expect(Child.Line("hello") == nil)
        #expect(Child.Line("report {}") == nil)
    }

    @Test func theRoutesToLeaveOutAreReadFromTheArguments() {
        #expect(Child.skipped(in: ["Siliconed", "--measure-capacity"]).isEmpty)
        let kernels = Child.skipped(in: ["Siliconed", "--measure-capacity", "--skip-route", "ane/int8", "--skip-route", "amx/fp32", "--skip-route", "nonsense"])
        #expect(kernels == [CapacityModel.Kernel(.ane, .int8), CapacityModel.Kernel(.amx, .fp32)])
    }

    // ── a route that died is excluded, and said ─────────────────────────────────────────────

    /// A route that died is left out of the plan like a route the judge refused, and is not a reason for orange: the Mac was not slow,
    /// the route is not there. The report says it died.
    @Test func aRouteThatDiedIsExcludedSaidAndNotOrange() throws {
        var judge = R.judge()
        judge["ane/fp16"] = .init(error: 0, threshold: 0, excluded: true, reason: .died)
        var m = R.measurements(dropping: ["ane/fp16"])
        m.trials.removeAll { $0.set.contains("ane") }
        let report = R.report(measurements: m, judge: judge)
        #expect(report.excluded.contains(R.ane))
        #expect(report.verdict.green, "\(report.verdict.reasons)")
        #expect(!report.plan().resources.ane, "no render loads a route the map has not seen load")
        #expect(report.judge["ane/fp16"]?.reason == .died)
        #expect(report.judge["gpu/fp32"]?.reason == nil)
        // …and it reads back.
        let back = try #require(Diagnostic.Report.read(try report.json()))
        #expect(back.judge["ane/fp16"]?.reason == .died && back == report)
        let text = String(decoding: try report.json(), as: UTF8.self)
        #expect(text.contains("\"reason\" : \"died\""))
        #expect(!String(decoding: try R.report().json(), as: UTF8.self).contains("\"reason\""), "a judge that excluded nothing says nothing of a reason")
    }

    // ── the supervisor against real processes ───────────────────────────────────────────────

    /// A `/bin/sh` child: `script` sees the arguments the supervisor passes (`--skip-route <kernel>`) as `$@`.
    func supervisor(_ script: String) -> Child.Supervisor {
        Child.Supervisor(executable: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", script, "sh"])
    }

    func reportFile() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("child-report-\(UUID().uuidString).json")
        try R.report().json(pretty: false).write(to: url)
        return url
    }

    /// The case of the lot: the first child announces the integers' route and dies on it; the second, told to leave it out, reports.
    @Test func aChildThatDiesOnARouteIsStartedAgainWithoutIt() throws {
        let file = try reportFile()
        defer { try? FileManager.default.removeItem(at: file) }
        let script = """
        echo "stage widening"; echo "stage engines"; echo "route gpu/fp32"; echo "route ane/int8"
        case "$*" in *ane/int8*) echo "report $(cat '\(file.path)')"; exit 0 ;; esac
        kill -SEGV $$
        """
        var stages: [Diagnostic.Stage] = [], routes: [String] = [], restarts: [[String]] = []
        let outcome = supervisor(script).run(onStage: { stages.append($0) }, onRoute: { routes.append($0) }, onRestart: { restarts.append($0) })
        #expect(outcome.report == R.report())
        #expect(outcome.failure == nil && !outcome.stopped)
        #expect(outcome.skipped == ["ane/int8"])
        #expect(outcome.deaths == [Child.Death(status: SIGSEGV, signaled: true, stage: .engines, route: "ane/int8")])
        #expect(restarts == [["ane/int8"]])
        #expect(stages == [.widening, .engines, .widening, .engines])
        #expect(routes == ["gpu/fp32", "ane/int8", "gpu/fp32", "ane/int8"])
    }

    /// A route per death, and no more than three restarts: a Mac whose every route kills is told, not looped on.
    @Test func theRestartsAreBounded() {
        let script = """
        for route in ane/fp16 amx/fp32 gpu/q8 gpu/int8 gpu/bf16; do
          case "$*" in *$route*) ;; *) echo "route $route"; kill -SEGV $$ ;; esac
        done
        """
        let outcome = supervisor(script).run()
        #expect(outcome.report == nil)
        #expect(outcome.skipped == ["ane/fp16", "amx/fp32", "gpu/q8"])
        #expect(outcome.deaths.count == 4)
        #expect(outcome.failure?.detail.contains("ended by signal \(SIGSEGV)") == true)
        #expect(outcome.failure?.detail.contains("gpu/int8") == true)
    }

    @Test func aKillOrAStatusOrTheReferenceIsAFailureNotARestart() {
        var restarts = 0
        let killed = supervisor(#"echo "route ane/int8"; kill -KILL $$"#).run(onRestart: { _ in restarts += 1 })
        #expect(killed.report == nil && killed.skipped.isEmpty && killed.deaths.count == 1 && killed.failure != nil)
        let exited = supervisor(#"echo "route ane/int8"; exit 3"#).run(onRestart: { _ in restarts += 1 })
        #expect(exited.skipped.isEmpty && exited.deaths == [Child.Death(status: 3, signaled: false, stage: nil, route: "ane/int8")])
        let reference = supervisor(#"echo "route gpu/fp32"; kill -SEGV $$"#).run(onRestart: { _ in restarts += 1 })
        #expect(reference.skipped.isEmpty && reference.failure != nil)
        #expect(restarts == 0)
    }

    /// The child's own word is kept: an `error` line is its failure, whatever the status after it.
    @Test func theChildsOwnFailureIsReturnedAsItIs() {
        let outcome = supervisor(#"echo "route ane/int8"; echo "error render-already-running another render holds the machine"; exit 1"#).run()
        #expect(outcome.failure?.code == "render-already-running" && outcome.failure?.detail == "another render holds the machine")
        #expect(outcome.deaths.isEmpty && outcome.skipped.isEmpty)
    }

    @Test func stoppingEndsTheChildAndTheRestarts() {
        let watched = supervisor(#"echo "stage widening"; sleep 30"#)
        let started = Date()
        nonisolated(unsafe) var outcome: Child.Outcome?
        let done = DispatchSemaphore(value: 0)
        Thread.detachNewThread { outcome = watched.run(); done.signal() }
        Thread.sleep(forTimeInterval: 0.4)
        watched.stop()
        #expect(done.wait(timeout: .now() + 5) == .success)
        #expect(outcome?.stopped == true && outcome?.report == nil && outcome?.failure == nil)
        #expect(Date().timeIntervalSince(started) < 10)
        // A stop before the run began is a run that never starts.
        let late = supervisor("echo hi")
        late.stop()
        #expect(late.run().stopped)
    }
}
