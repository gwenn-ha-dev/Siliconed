import Foundation
@testable import SiliCore
import Testing

/// **The trace's account of a run** (`Stages.occupancy`), on calls written by hand, in milliseconds: two groups of a
/// form whose stretches of either unit lie around a step of the GPU alone (`alone`).
@Suite struct Traces {
    static func ms(_ t: Double) -> UInt64 { UInt64(t * 1e6) }
    static func call(_ kind: Stages.Called.Kind, _ step: String, group: Int, place: Int, _ from: Double, _ to: Double) -> Stages.Called {
        Stages.Called(kind: kind, step: step, group: group, place: place, rows: 64, flops: 1_000_000, start: ms(from), end: ms(to))
    }

    static var calls: [Stages.Called] {
        let g1 = Block.places
        return [call(.neural, "q", group: 0, place: 0, 0, 10), call(.past, "q", group: 0, place: 0, 0, 8),
                call(.fixed, "attention", group: 0, place: 1, 12, 20),
                call(.neural, "out", group: 0, place: 2, 22, 30), call(.past, "out", group: 0, place: 2, 20, 28),
                call(.neural, "q", group: 1, place: g1, 35, 40), call(.past, "q", group: 1, place: g1, 31, 39)]
    }

    /// The ANE idle across the GPU's own step and at the next group; the GPU idle before its own step (the ANE drained
    /// first) and at the next group; each unit's busy seconds; nothing outside but the GPU's last millisecond.
    @Test func idleIsToldByWhereItLies() {
        let o = Stages.occupancy(Self.calls, start: 0, end: Self.ms(40))
        func near(_ a: Double, _ b: Double) -> Bool { abs(a - b) < 1e-9 }
        #expect(near(o.seconds, 0.040))
        #expect(near(o.ane, 0.023) && near(o.gpu, 0.032))
        #expect(near(o.aneIdle.acrossAlone, 0.012) && near(o.aneIdle.atGroups, 0.005) && o.aneIdle.between == 0 && o.aneIdle.outside == 0)
        #expect(near(o.gpuIdle.acrossAlone, 0.004) && near(o.gpuIdle.atGroups, 0.003) && o.gpuIdle.between == 0 && near(o.gpuIdle.outside, 0.001))
        #expect(o.unread == 0)
        let q = o.steps.first { $0.unit == "ane" && $0.name == "q" }
        #expect(q?.calls == 2 && q?.rows == 128 && q.map { near($0.seconds, 0.015) } == true)
    }

    /// A call whose end was not read is left out and counted; a unit with no call stood idle the whole run.
    @Test func anUnreadCallIsCountedApart() {
        var calls = Self.calls.filter { !$0.neural }
        calls[0].end = 0
        let o = Stages.occupancy(calls, start: 0, end: Self.ms(40))
        #expect(o.unread == 1)
        #expect(o.ane == 0 && abs(o.aneIdle.outside - 0.040) < 1e-9)
    }

    /// Off, a stage records nothing: the switch is read at its opening, never at a call.
    @Test func theSwitchIsOffUnlessSaid() {
        #expect(Stages.tracing == false)
    }
}
