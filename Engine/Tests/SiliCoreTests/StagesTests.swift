import Foundation
import Testing
@testable import SiliCore

/// **Every observer hears a stage, until it lets go** (`Stages.observe`). An observer hears the whole process: what it
/// counts is this test's own events, the stages other tests open beside it heard too.
@Suite struct StagesTests {
    @Test func everyObserverHearsUntilCancelled() {
        final class Heard: @unchecked Sendable { var lock = NSLock(); var n = 0; func add() { lock.withLock { n += 1 } } }
        let a = Heard(), b = Heard()
        @Sendable func mine(_ e: Stages.Event) -> Bool { if case .neural(0.5) = e { true } else { false } }
        let first = Stages.observe { if mine($0) { a.add() } }
        var second: Stages.Observation? = Stages.observe { if mine($0) { b.add() } }
        Stages.computed(neuralEngine: 0.5)
        second = nil
        Stages.computed(neuralEngine: 0.5)
        first.cancel()
        Stages.computed(neuralEngine: 0.5)
        #expect(a.n == 2 && b.n == 1 && second == nil)
    }
}
