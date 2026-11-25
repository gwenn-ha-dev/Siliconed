import Foundation
import SiliCore

// **The render's memory in time** — the process's footprint as a series, the Machine window's graph. The worker reads
// the footprint every millisecond (`FootprintWatch`, one thread a process) and `MemoryRecorder` keeps, per bucket of
// 20 ms, its lowest and highest reading: a ring slot filled then let go is a tooth a thinner series would flatten, and
// the min–max pair survives any later thinning. Beside the readings, **what the engine's stages say of it** at each
// change (`MemoryLevel`): the process's floor under the stages open, and the bytes of weights they hold. Sent to the app
// as deltas (`Engine.Event.memory`), kept there whole.

extension Engine {
    /// One bucket: `index` counts `step`s from the render's start.
    package struct MemoryBucket: Sendable, Codable, Equatable {
        package var index: Int
        package var low, high: Int
        package init(index: Int, low: Int, high: Int) { self.index = index; self.low = low; self.high = high }
    }

    /// **From `at` on, what the engine's stages hold**: the process's footprint before the first of them opened
    /// (`floor`; `nil` while none is open: the whole footprint is the process's) and the bytes of weights they hold
    /// (`Stage.Plan.weights`, and the groups kept in flight) — the rest of the footprint is their intermediates and Core
    /// AI's working memory.
    package struct MemoryLevel: Sendable, Codable, Equatable {
        package var at: Double
        package var floor: Int?
        package var weights: Int
        package init(at: Double, floor: Int?, weights: Int) { (self.at, self.floor, self.weights) = (at, floor, weights) }
    }

    /// A stage of the render begins, in seconds from its start.
    package struct MemoryMark: Sendable, Codable, Equatable {
        package var stage: Stage
        package var at: Double
        package init(stage: Stage, at: Double) { self.stage = stage; self.at = at }
    }

    /// **A series, or a part of one** (a delta carries the buckets and marks since the last one): buckets in order of
    /// index, a bucket missing where no reading came, and the stages' starts.
    package struct MemorySeries: Sendable, Codable, Equatable {
        /// A bucket's length in seconds: 20 ms as recorded, longer once thinned.
        package var step = 0.02
        package var buckets: [MemoryBucket] = []
        package var marks: [MemoryMark] = []
        package var levels: [MemoryLevel] = []
        package init() {}
        package init(step: Double = 0.02, buckets: [MemoryBucket], marks: [MemoryMark] = [], levels: [MemoryLevel] = []) {
            self.step = step; self.buckets = buckets; self.marks = marks; self.levels = levels
        }

        package var isEmpty: Bool { buckets.isEmpty && marks.isEmpty && levels.isEmpty }

        /// Seconds from the render's start to the end of the last bucket, or to the last stage's start.
        package var duration: Double { max((buckets.last.map { Double($0.index + 1) } ?? 0) * step, marks.last?.at ?? 0) }

        /// The delta that follows: deltas arrive in order, a bucket already there (the same index) takes the pair's
        /// extremes.
        package mutating func append(_ delta: MemorySeries) {
            if buckets.isEmpty { step = delta.step }
            for b in delta.buckets {
                if let last = buckets.last, last.index >= b.index {
                    if last.index == b.index { buckets[buckets.count - 1].low = min(last.low, b.low); buckets[buckets.count - 1].high = max(last.high, b.high) }
                } else {
                    buckets.append(b)
                }
            }
            marks += delta.marks
            levels += delta.levels
        }

        /// At most `count` buckets, neighbours merged with their extremes kept.
        package func thinned(to count: Int) -> MemorySeries {
            guard let last = buckets.last, count > 0, last.index >= count else { return self }
            let every = (last.index + count) / count
            var out = MemorySeries(step: step * Double(every), buckets: [], marks: marks, levels: levels)
            for b in buckets {
                let i = b.index / every
                if let l = out.buckets.last, l.index == i {
                    out.buckets[out.buckets.count - 1].low = min(l.low, b.low)
                    out.buckets[out.buckets.count - 1].high = max(l.high, b.high)
                } else {
                    out.buckets.append(MemoryBucket(index: i, low: b.low, high: b.high))
                }
            }
            return out
        }

        /// **The envelope over `columns` pixel columns** spanning `0 ... duration`: each column's lowest and highest
        /// footprint among the buckets that overlap it, `nil` where none does. A bucket wider than a column fills
        /// every column it covers, so a narrow chart and a wide one show the same teeth.
        package func envelope(columns: Int, duration: Double) -> [(low: Int, high: Int)?] {
            guard columns > 0, duration > 0 else { return [] }
            var out = [(low: Int, high: Int)?](repeating: nil, count: columns)
            let scale = Double(columns) / duration
            for b in buckets {
                let first = min(columns - 1, max(0, Int(Double(b.index) * step * scale)))
                let end = min(columns, max(first + 1, Int((Double(b.index + 1) * step * scale).rounded(.up))))
                for c in first..<end {
                    if let e = out[c] { out[c] = (min(e.low, b.low), max(e.high, b.high)) } else { out[c] = (b.low, b.high) }
                }
            }
            return out
        }
    }
}

/// **The worker's side**: every reading of the process's footprint, folded into buckets as it comes (`record` runs on
/// the ticker's thread), and the buckets sent so far forgotten. `drain` hands over what is new: the closed buckets, the
/// open one only at the end — so nothing is sent twice and nothing is lost.
package final class MemoryRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private let origin: UInt64
    private var pending: [Engine.MemoryBucket] = []
    private var marks: [Engine.MemoryMark] = []
    private var levels: [Engine.MemoryLevel] = []

    package init(origin: UInt64 = DispatchTime.now().uptimeNanoseconds) { self.origin = origin }

    /// `now` (`DispatchTime` nanoseconds) on the series' clock: seconds from the recorder's start.
    package func seconds(_ now: UInt64) -> Double { Double(now &- min(now, origin)) / 1e9 }

    /// A reading of `footprint` at `now` (`DispatchTime` nanoseconds).
    package func record(_ footprint: Int, at now: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        let index = Int(seconds(now) / Engine.MemorySeries().step)
        lock.withLock {
            if let last = pending.last, last.index >= index {
                // A late reading from another thread (`FootprintWatch.peak`) lands in the bucket it belongs to, or the last.
                pending[pending.count - 1].low = min(last.low, footprint)
                pending[pending.count - 1].high = max(last.high, footprint)
            } else {
                pending.append(Engine.MemoryBucket(index: index, low: footprint, high: footprint))
            }
        }
    }

    package func mark(_ stage: Engine.Stage, at now: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        lock.withLock { marks.append(Engine.MemoryMark(stage: stage, at: seconds(now))) }
    }

    /// What the engine's stages hold from `now` on (`Engine.MemoryLevel`).
    package func level(floor: Int?, weights: Int, at now: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        lock.withLock { levels.append(Engine.MemoryLevel(at: seconds(now), floor: floor, weights: weights)) }
    }

    /// What is new since the last call; `nil` when nothing. `final`: the open bucket too (the render is over).
    package func drain(final: Bool = false) -> Engine.MemorySeries? {
        lock.withLock {
            let take = final ? pending.count : max(0, pending.count - 1)
            guard take > 0 || !marks.isEmpty || !levels.isEmpty else { return nil }
            let delta = Engine.MemorySeries(buckets: Array(pending.prefix(take)), marks: marks, levels: levels)
            pending.removeFirst(take); marks = []; levels = []
            return delta
        }
    }
}
