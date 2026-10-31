import Foundation
import Testing
@testable import Siliconed

/// **A finished measure applies to the next render** (`SettingsStore`): the plan is replaced between two renders and never during one,
/// and everything that survives a render keyed by the plan follows. Pure: each test has its own store, the process's is not touched.
@Suite struct SettingsReplacementTests {
    typealias R = ReportFixtures

    /// A report of this Mac; `excluding` leaves a kernel unmeasured, so that its model — and the plan's identity — differ.
    private func current(excluding kernel: String? = nil, date: String) -> Diagnostic.Report {
        var report = R.report(machine: Diagnostic.machine(), measurements: R.measurements(dropping: kernel.map { [$0] } ?? []))
        report.date = date
        return report
    }

    private func store() -> SettingsStore { SettingsStore(EngineSettings(applying: [:])) }

    @Test func aMeasureBetweenTwoRendersReplacesThePlan() {
        let store = store()
        let first = current(date: "2026-10-31T10:00:00Z"), second = current(excluding: "ane/fp16", date: "2026-10-31T11:00:00Z")
        #expect(store.current.plan.identity == CapacityPlan.defaults.identity)
        #expect(store.adopt(first) == .applied)
        let before = store.current
        #expect(before.mapRead == first.date && before.plan.identity.hasPrefix("model "))
        let generation = store.generation
        store.renderBegan(); store.renderEnded()
        #expect(store.adopt(second) == .applied)
        #expect(store.current.mapRead == second.date && store.current.plan.identity != before.plan.identity)
        #expect(store.generation == generation + 1)
        // The same measure again changes nothing, and counts for nothing.
        #expect(store.adopt(second) == .unchanged && store.generation == generation + 1)
    }

    @Test func aRenderUnderWayKeepsItsPlanToTheEnd() {
        let store = store()
        let first = current(date: "2026-10-31T10:00:00Z"), second = current(excluding: "ane/fp16", date: "2026-10-31T11:00:00Z")
        store.adopt(first)
        store.renderBegan()
        let seen = store.current.plan.identity
        #expect(store.adopt(second) == .deferred)
        #expect(store.current.plan.identity == seen && store.current.mapRead == first.date, "a read in the middle of the render is the render's")
        #expect(!store.applyPending(), "nothing applies while the render runs")
        store.renderEnded()
        #expect(store.current.mapRead == first.date, "the end of a render does not apply it either: the app does, before its next job")
        #expect(store.applyPending())
        #expect(store.current.mapRead == second.date && store.current.plan.identity != seen)
        #expect(!store.applyPending(), "applied once")
    }

    @Test func theLastDeferredMeasureWins() {
        let store = store()
        store.renderBegan()
        store.adopt(current(date: "2026-10-31T10:00:00Z"))
        store.adopt(current(excluding: "ane/fp16", date: "2026-10-31T11:00:00Z"))
        store.renderEnded()
        store.applyPending()
        #expect(store.current.mapRead == "2026-10-31T11:00:00Z")
    }

    @Test func aDesignatedPlanIsNotReplaced() throws {
        var settings = EngineSettings(applying: [:])
        settings.planFixedForTesting()
        let store = SettingsStore(settings)
        #expect(store.adopt(current(date: "2026-10-31T10:00:00Z")) == .unchanged)
        #expect(store.current.plan.identity == CapacityPlan.defaults.identity)
    }

    @Test func aReportOfAnotherMacChangesNothingAndSaysSo() {
        let store = store()
        store.adopt(current(date: "2026-10-31T10:00:00Z"))
        let plan = store.current.plan.identity
        #expect(store.adopt(R.report()) == .unchanged)   // the fixture's own M1 Pro, not this Mac
        #expect(store.current.plan.identity == plan)
        #expect(store.current.warnings.contains { $0.hasPrefix(EngineSettings.staleMapWarning) })
        // A current one clears the warning.
        store.adopt(current(excluding: "ane/fp16", date: "2026-10-31T12:00:00Z"))
        #expect(!store.current.warnings.contains { $0.hasPrefix(EngineSettings.staleMapWarning) })
    }

    /// What survives a render and depends on the plan is keyed by its identity: a new plan reads other entries, nothing stale is served.
    @Test func theRenderCacheKeyFollowsThePlan() {
        let a = current(date: "2026-10-31T10:00:00Z"), b = current(excluding: "ane/fp16", date: "2026-10-31T11:00:00Z")
        let (pa, pb) = (a.plan(), b.plan())
        #expect(pa.identity != pb.identity)
        #expect("plan \(pa.restricted(allowingANE: true).identity)" != "plan \(pb.restricted(allowingANE: true).identity)")
    }
}
