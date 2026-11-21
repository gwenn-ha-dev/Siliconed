import Foundation
import XCTest
@testable import Siliconed

/// **The footprint in time** (`MemoryRecorder`, `Engine.MemorySeries`): 20 ms buckets keeping their lowest and highest
/// reading, the time counted from the render's start, the deltas that carry them to the app (nothing twice, nothing
/// lost), and the envelope that draws them.
final class MemorySeriesTests: XCTestCase {
    private let ms: UInt64 = 1_000_000
    private let origin: UInt64 = 5_000_000_000

    private func recorder() -> MemoryRecorder { MemoryRecorder(origin: origin) }

    func testABucketKeepsItsLowestAndHighestReading() {
        let r = recorder()
        for (t, f) in [(0, 100), (5, 400), (19, 250), (20, 90), (39, 95), (40, 7)] as [(UInt64, Int)] {
            r.record(f, at: origin + t * ms)
        }
        let all = r.drain(final: true)
        XCTAssertEqual(all?.buckets, [.init(index: 0, low: 100, high: 400), .init(index: 1, low: 90, high: 95),
                                      .init(index: 2, low: 7, high: 7)])
    }

    func testTimeCountsFromTheRendersStart() {
        let r = recorder()
        r.record(1, at: origin + 1_000 * ms)
        r.record(2, at: origin + 1_019 * ms)
        r.mark(.denoising, at: origin + 1_500 * ms)
        let d = r.drain(final: true)
        XCTAssertEqual(d?.buckets.map(\.index), [50], "1.000 s and 1.019 s are one bucket of 20 ms: the 50th")
        XCTAssertEqual(d?.marks, [.init(stage: .denoising, at: 1.5)])
    }

    func testNothingIsSentTwiceAndNothingIsLost() {
        let r = recorder()
        var kept = Engine.MemorySeries()
        var every: [Engine.MemoryBucket] = []
        for t in stride(from: UInt64(0), to: 200, by: 1) {
            let f = Int(t % 7) * 10 + 1
            r.record(f, at: origin + t * ms)
            every.append(.init(index: Int(t / 20), low: f, high: f))
            if t % 37 == 0, let d = r.drain() {
                XCTAssertTrue(d.buckets.allSatisfy { $0.index > (kept.buckets.last?.index ?? -1) }, "no bucket twice")
                kept.append(d)
            }
        }
        if let d = r.drain(final: true) { kept.append(d) }
        XCTAssertNil(r.drain(final: true), "all handed over")
        let expected = Dictionary(grouping: every, by: \.index).map { i, v in
            Engine.MemoryBucket(index: i, low: v.map(\.low).min()!, high: v.map(\.high).max()!)
        }.sorted { $0.index < $1.index }
        XCTAssertEqual(kept.buckets, expected)
    }

    func testTheOpenBucketWaitsForTheEnd() {
        let r = recorder()
        r.record(10, at: origin)
        r.record(30, at: origin + 25 * ms)
        XCTAssertEqual(r.drain()?.buckets, [.init(index: 0, low: 10, high: 10)])
        r.record(5, at: origin + 30 * ms)
        XCTAssertNil(r.drain(), "bucket 1 is still open")
        XCTAssertEqual(r.drain(final: true)?.buckets, [.init(index: 1, low: 5, high: 30)])
    }

    func testAMarkTravelsOnceWithTheNextDelta() {
        let r = recorder()
        r.mark(.text, at: origin)
        XCTAssertEqual(r.drain()?.marks.count, 1)
        XCTAssertNil(r.drain())
    }

    func testThinningKeepsTheTeeth() {
        let s = Engine.MemorySeries(buckets: (0..<1000).map { .init(index: $0, low: $0 == 500 ? 1 : 100, high: $0 == 501 ? 9_000 : 100) })
        let t = s.thinned(to: 100)
        XCTAssertLessThanOrEqual(t.buckets.count, 100)
        XCTAssertEqual(t.buckets.map(\.low).min(), 1)
        XCTAssertEqual(t.buckets.map(\.high).max(), 9_000)
        XCTAssertEqual(t.duration, s.duration, accuracy: t.step)
        XCTAssertEqual(s.thinned(to: 2_000), s, "nothing to thin")
    }

    func testTheEnvelopeShowsTheTeethAtAnyWidth() {
        let s = Engine.MemorySeries(buckets: (0..<100).map { .init(index: $0, low: $0 % 10 == 0 ? 10 : 50, high: 50) })
        for columns in [10, 100, 1000] {
            let e = s.envelope(columns: columns, duration: s.duration)
            XCTAssertEqual(e.count, columns)
            XCTAssertEqual(e.compactMap { $0 }.map(\.low).min(), 10, "\(columns) columns")
            XCTAssertEqual(e.compactMap { $0 }.map(\.high).max(), 50)
            XCTAssertFalse(e.contains { $0 == nil }, "a bucket wider than a column fills each it covers")
        }
        let gap = Engine.MemorySeries(buckets: [.init(index: 0, low: 1, high: 2), .init(index: 99, low: 3, high: 4)])
        XCTAssertNil(gap.envelope(columns: 10, duration: gap.duration)[5], "no reading, no column")
    }

    func testTheSeriesAndItsDeltasTravelAsJSON() throws {
        var s = Engine.MemorySeries(buckets: [.init(index: 3, low: 1, high: 2)], marks: [.init(stage: .text, at: 0.5)])
        s.step = 0.04
        let back = try JSONDecoder().decode(Engine.MemorySeries.self, from: JSONEncoder().encode(s))
        XCTAssertEqual(back, s)
    }
}
