import Foundation
import Testing
@testable import SiliCore

/// **One ticker reads for every watch** (`FootprintTicker`): each watch keeps its own peak, and one's reset or stop
/// leaves the others' alone.
@Suite struct FootprintWatchTests {
    final class Reading: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Int
        init(_ value: Int) { self.value = value }
        func set(_ v: Int) { lock.withLock { value = v } }
        var now: Int { lock.withLock { value } }
    }

    private func wait(_ done: () -> Bool) {
        let t = DispatchTime.now().uptimeNanoseconds
        while !done() && DispatchTime.now().uptimeNanoseconds - t < 2_000_000_000 { usleep(1000) }
    }

    @Test func eachWatchKeepsItsOwnPeak() {
        let a = Reading(10), b = Reading(100)
        let wa = FootprintWatch { a.now }, wb = FootprintWatch { b.now }
        a.set(50); b.set(300)
        wait { wa.peak >= 50 && wb.peak >= 300 }
        a.set(20); b.set(150)
        usleep(20_000)
        #expect(wa.peak == 50 && wb.peak == 300)
        wa.reset()
        #expect(wa.peak == 20 && wb.peak == 300)
        wa.stop()
        b.set(400)
        wait { wb.peak >= 400 }
        a.set(900)
        usleep(20_000)
        // The stopped one no longer moves the other's peak.
        #expect(wb.peak == 400)
        wb.stop()
    }
}
