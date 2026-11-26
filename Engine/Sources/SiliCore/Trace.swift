import Foundation
import Synchronization

/// **What a stage opened under the trace records** (`Stages.tracing`): its opening and its close piece by piece, and
/// its calls as they ran, summed into where each unit's seconds went. Nothing of it is read or written while the switch
/// was off at the stage's opening: what it records are the times the stream and the ANE's calls give anyway (a GPU
/// call's end as its wait task ends, an ANE call's own start and end), no wait added, the calls in the same order.
extension Stages {
    public enum Trace: Sendable, Equatable {
        /// A stage's opening from its start (`DispatchTime` nanoseconds): its arena, its residents read, its ring, its
        /// functions loaded.
        case opening(stage: String, start: UInt64, pieces: [Piece])
        /// A stage's close from its start: everything let go, no wait (`Stage.close`).
        case closing(stage: String, start: UInt64, pieces: [Piece])
        /// A block's cycle, its ordinal among the block's cycles from 0 (told before the cycle's `Event.cycle`), or a
        /// run of calls outside the cycles (nil: groups run once, a stage of bands' run).
        case calls(stage: String, cycle: Int?, Occupancy)
    }

    /// A piece of an opening or a close, its seconds.
    public struct Piece: Sendable, Equatable {
        public let name: String
        public let seconds: Double
    }

    /// **Where a run's seconds went, unit by unit**, from its calls on one clock: each unit's busy seconds; where each
    /// stood idle within the run — between two calls a step of the GPU alone lies between (its calls wait for every
    /// ANE call before them, and the ANE has nothing in flight until they are done), between two calls of two groups,
    /// between two calls of one stretch, before its first call and after its last —; each step's calls on each unit
    /// (`Calls.Step`); the calls whose end was not read (a GPU call not done when the run was told).
    public struct Occupancy: Sendable, Equatable {
        public let start, end: UInt64
        public let ane, gpu: Double
        public let aneIdle, gpuIdle: Idle
        public let steps: [Calls.Step]
        public let unread: Int
        public var seconds: Double { Double(end &- start) / 1e9 }

        public struct Idle: Sendable, Equatable {
            public internal(set) var acrossAlone = 0.0, atGroups = 0.0, between = 0.0, outside = 0.0
            public var total: Double { acrossAlone + atGroups + between + outside }
        }
    }

    /// **A call as the trace saw it**: its unit's kind (the ANE's at a step; the GPU's past the ANE's rows, of a step of
    /// its own, an edge's; an edge of the ANE's), its step, the group it ran in (an ordinal over the block's life) and
    /// its place there — 2 × the `gpu` steps before it, + 1 a call of a `gpu` step, so that a `gpu` step's calls lie
    /// between the stretches of the ANE around them —, its rows and FLOP, its start and its end (0: not read).
    struct Called: Equatable {
        enum Kind: String { case neural = "ane", past = "beside", fixed = "alone", edge = "edge", neuralEdge = "ane edge" }
        var kind: Kind
        var step: String
        var group: Int
        var place: Int
        var rows = 0, flops = 0
        var start: UInt64 = 0, end: UInt64 = 0
        var neural: Bool { kind == .neural || kind == .neuralEdge }
    }

    /// **The occupancy of `calls` over `start ..< end`** (`Occupancy`). Each unit's calls by their start; a gap between
    /// two of the ANE's where a call of a `gpu` step lies between their places is `acrossAlone`, one between two groups
    /// `atGroups`, any other `between`; the GPU's gap before the first call of a `gpu` step after a stretch of the ANE's
    /// is `acrossAlone`, before a group's first call `atGroups`. A unit with no call stood idle the whole run.
    static func occupancy(_ calls: [Called], start: UInt64, end: UInt64) -> Occupancy {
        let read = calls.filter { $0.end > 0 && $0.end >= $0.start }
        let fixed = read.filter { $0.kind == .fixed }.map(\.place)
        func seconds(_ a: UInt64, _ b: UInt64) -> Double { b > a ? Double(b - a) / 1e9 : 0 }
        func idle(_ unit: [Called], across: (Called, Called) -> Bool) -> (busy: Double, idle: Occupancy.Idle) {
            var idle = Occupancy.Idle()
            guard let first = unit.first, let last = unit.max(by: { $0.end < $1.end }) else {
                idle.outside = seconds(start, end)
                return (0, idle)
            }
            idle.outside = seconds(start, first.start) + seconds(last.end, end)
            var busy = 0.0, reached = first.start
            for (i, c) in unit.enumerated() {
                // Calls of one unit may overlap (two of the GPU's stream ends read late): only the time past the one
                // before counts, as busy and as a gap.
                if i > 0 {
                    let gap = seconds(reached, c.start), p = unit[i - 1]
                    if across(p, c) { idle.acrossAlone += gap } else if p.group != c.group { idle.atGroups += gap } else { idle.between += gap }
                }
                busy += seconds(max(reached, c.start), c.end)
                reached = max(reached, c.end)
            }
            return (busy, idle)
        }
        let ane = read.filter(\.neural).sorted { $0.start < $1.start }
        let gpu = read.filter { !$0.neural }.sorted { $0.start < $1.start }
        let a = idle(ane) { p, c in fixed.contains { $0 > p.place && $0 < c.place } }
        let g = idle(gpu) { p, c in c.kind == .fixed && p.kind != .fixed }
        var steps: [String: Calls.Step] = [:]
        for c in read { steps["\(c.kind.rawValue) \(c.step)", default: .init(name: c.step, unit: c.kind.rawValue)].add(rows: c.rows, seconds: seconds(c.start, c.end), flops: c.flops) }
        return Occupancy(start: start, end: end, ane: a.busy, gpu: g.busy, aneIdle: a.idle, gpuIdle: g.idle,
                         steps: steps.values.sorted { ($0.unit, $0.name) < ($1.unit, $1.name) }, unread: calls.count - read.count)
    }

    /// **A GPU call's end, as its wait task reads it** (`SerialStream.done`): written there, read once the run is told.
    final class End: Sendable {
        let value = Atomic<UInt64>(0)
    }
}
