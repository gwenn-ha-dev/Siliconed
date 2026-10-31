import Foundation
import Testing
@testable import Siliconed

/// **Was the Mac busy while it measured?** (`CapacityRegime.swift`): the criterion on the pulses and the spreads a run keeps, each rule on a
/// regime whose only fault is the one tested, then on the cards recorded on the M1 Pro (`Fixtures/regime/`) — free Macs, and Macs beside
/// busy loops, a compilation and a GPU loop of another process — so that the rule rejects none of the first and passes none of the second.
/// Pure: no GPU, no clock.
@Suite struct RegimeTests {
    typealias Regime = Diagnostic.Regime
    typealias R = ReportFixtures

    /// A free Mac's pulse: the median thread took 1.05 × a thread alone, an empty submission waited 0.23 ms.
    func pulse(_ step: String, attempt: Int = 1, crowd: Double = 1.05, wait: Double = 0.00023) -> Regime.Pulse {
        .init(at: 1, step: step, attempt: attempt, crowd: crowd, wait: wait, late: wait * 1.2)
    }

    func regime(_ pulses: [Regime.Pulse] = [], spreads: [String: Double] = [:]) -> Regime {
        var r = Regime()
        r.pulses = pulses; r.spreads = spreads
        return r
    }

    @Test func aFreeMacIsNotDisturbed() {
        let r = regime((0..<12).map { pulse("step \($0)", crowd: 1.03 + 0.01 * Double($0 % 5), wait: 0.00022 + 0.000004 * Double($0 % 4)) },
                       spreads: ["a": 0.02, "b": 0.07, "c": 0.11])
        #expect(r.disturbances.isEmpty)
        #expect(!r.busy(r.pulses[0]))
    }

    /// The threads' crowd: the median thread over one thread alone, over 1.25.
    @Test func threadsThatLostTheirCoresAreNamed() {
        let r = regime([pulse("a"), pulse("b", crowd: 1.42), pulse("c", crowd: 1.33), pulse("d")])
        #expect(r.busy(r.pulses[1]) && !r.busy(r.pulses[0]))
        let d = r.disturbances
        #expect(d.map(\.kind) == [.threads])
        #expect(d[0].steps == ["b", "c"] && d[0].worstStep == "b" && d[0].worst == 1.42)
    }

    /// The wait is read against the quietest of the run, and against a ceiling no Mac's quiet submission is near.
    @Test func aSubmissionThatWaitedIsNamedAgainstTheQuietestOfTheRun() {
        let r = regime([pulse("a"), pulse("b", wait: 0.000545), pulse("c", wait: 0.00024), pulse("d", wait: 0.0051)])
        #expect(r.disturbances.map(\.kind) == [.gpu])
        #expect(r.disturbances[0].steps == ["b", "d"] && r.disturbances[0].worstStep == "d")
        // A Mac whose submissions all wait 0.55 ms is a slow Mac, not a busy one: only the ceiling can tell, and it is far above.
        let slow = regime((0..<6).map { pulse("s\($0)", wait: 0.00055) })
        #expect(slow.disturbances.isEmpty)
        let jammed = regime((0..<6).map { pulse("j\($0)", wait: 0.0021) })
        #expect(jammed.disturbances.map(\.kind) == [.gpu])
    }

    @Test func identicalCallsThatDifferAreNamed() {
        let r = regime([pulse("a")], spreads: ["ane/fp16 2048x3840x3840": 0.19, "amx+ane 4096x3840x3840 pass 1": 0.62, "gpu+ane pass 2": 0.10])
        let d = r.disturbances
        #expect(d.map(\.kind) == [.calls])
        #expect(d[0].steps == ["amx+ane 4096x3840x3840 pass 1", "ane/fp16 2048x3840x3840"] && d[0].worst == 0.62)
    }

    /// A step measured again until the Mac was quiet is not a disturbance: only the pulse that closed it for the last time counts.
    @Test func aStepMeasuredAgainUntilQuietIsNotKept() {
        let r = regime([pulse("a"), pulse("b", attempt: 1, crowd: 1.6, wait: 0.004), pulse("b", attempt: 2, crowd: 1.5), pulse("b", attempt: 3), pulse("c")],
                       spreads: ["b": 0.04])
        #expect(r.disturbances.isEmpty)
        // …and one that stayed busy at the third time is.
        let stuck = regime([pulse("a"), pulse("b", attempt: 1, crowd: 1.6), pulse("b", attempt: 2, crowd: 1.5), pulse("b", attempt: 3, crowd: 1.4)])
        #expect(stuck.disturbances.map(\.kind) == [.threads] && stuck.disturbances[0].steps == ["b"])
    }

    // ── the findings ────────────────────────────────────────────────────────────────────────

    func report(_ regime: Regime?) -> Diagnostic.Report {
        let base = R.report()
        let probes = Diagnostic.CapacityProbes.Result(date: base.date, seconds: base.seconds, tier: 0, measurements: base.measurements, model: base.model,
                                                      judge: base.judge, memory: base.memory, stages: base.stages, details: .init(), regime: regime ?? Regime())
        var r = Diagnostic.Report(date: base.date, seconds: base.seconds, machine: base.machine, cpuFeatures: base.cpuFeatures, settings: [:],
                                  before: base.before, after: base.after, probes: probes)
        if regime == nil { r.regime = nil }
        return r
    }

    @Test func aMeasureThatStayedBusyIsAMapThatSaysWhatItSaw() throws {
        let busy = report(regime([pulse("a"), pulse("sharing gpu+ane", crowd: 1.42)], spreads: ["ane/fp16 2048x3840x3840": 0.4]))
        #expect(busy.findings.map(\.kind) == [.disturbed, .disturbed])
        #expect(busy.findings.map(\.subject) == ["threads", "calls"])
        #expect(busy.findings.allSatisfy { $0.effect == .fastSide }, "an uncertainty: the figures are read on their quick side")
        let text = busy.findings.map(\.text)
        #expect(text[0].contains("70 % of their cores") && text[0].contains("1 step") && text[0].contains("sharing gpu+ane"))
        #expect(text[1].contains("40 %") && text[1].contains("15 %") && text[1].contains("fast side"))
        for finding in busy.findings { #expect(finding.cause(excluding: []) == .disturbed) }
        // …and it is the map all the same: current, designated, its uncertainty written beside the figure it belongs to.
        var mine = busy
        mine.machine = Diagnostic.machine()
        #expect(mine.isCurrent && EngineSettings.designation(of: mine).plan != nil)
        #expect(busy.uncertainty.first { $0.subject == "ane/fp16" }?.plusMinus == 0.2)
        // It crosses the JSON whole, the regime with it.
        let back = try #require(Diagnostic.Report.read(try busy.json()))
        #expect(back == busy && back.regime?.pulses.count == 2)
    }

    /// A free Mac's report, a report without a regime (an older one) and a step measured again find nothing: what the guard redid is not a finding.
    @Test func aFreeMacFindsNothingWhateverTheGuardRedid() {
        #expect(report(regime([pulse("a"), pulse("b")], spreads: ["a": 0.03])).findings.isEmpty)
        #expect(report(nil).findings.isEmpty)
        var redone = regime([pulse("a"), pulse("b", crowd: 1.6), pulse("b", attempt: 2)])
        redone.repeated = 1; redone.repeatedSeconds = 1.4
        #expect(report(redone).findings.isEmpty)
    }

    // ── the cards recorded on the M1 Pro ────────────────────────────────────────────────────

    struct Card: Decodable {
        var name: String
        var expect: String
        var seconds: Double
        var lines: [String: Double]
        var regime: Regime
    }

    static let cards: [Card] = {
        let folder = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/regime")
        let files = ((try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []).filter { $0.hasSuffix(".json") }.sorted()
        return files.compactMap { try? JSONDecoder().decode(Card.self, from: Data(contentsOf: folder.appendingPathComponent($0))) }
    }()

    /// **The criterion on recorded cards**: none of the free ones is called disturbed, every one of the others is.
    @Test func theCriterionSeparatesTheRecordedCards() throws {
        let cards = Self.cards
        try #require(!cards.isEmpty, "the recorded cards are Fixtures/regime/*.json")
        var wrong: [String] = []
        for card in cards {
            let found = !card.regime.disturbances.isEmpty
            if (card.expect == "busy") != found { wrong.append("\(card.name) (\(card.expect)): \(card.regime.disturbances.map { "\($0.kind.rawValue) \($0.worst)" })") }
        }
        #expect(wrong.isEmpty, "\(wrong)")
        #expect(cards.contains { $0.expect == "quiet" } && cards.contains { $0.expect == "busy" })
    }
}
