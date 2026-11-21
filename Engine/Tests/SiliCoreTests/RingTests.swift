import Foundation
import SiliCore
import Testing

/// **The ring in two phases, on simulated reads** (no GPU): a place is a marker, a read a line in a log. What is
/// checked is what the loop sees — the place of each use, the reads behind it — whatever the reader's thread and the
/// test's interleave: the bars hold for every order.
@Suite struct Rings {
    /// A place: one marker weight.
    static func place(_ k: Int) -> [String: Placed] { ["w": Placed(offset: k, shape: [1], type: .float16)] }
    static let slotsBase = 1000, keptBase = 2000

    /// The reads, in order, by group; safe across the reader's thread and the test's.
    final class Log: @unchecked Sendable {
        private let lock = NSLock()
        private var reads: [(group: String, place: Int)] = []
        private var kept: [String] = []
        func read(_ g: String, _ p: [String: Placed]) { lock.withLock { reads.append((g, p["w"]!.offset)) } }
        func keep(_ g: String) { lock.withLock { kept.append(g) } }
        var all: [(group: String, place: Int)] { lock.withLock { reads } }
        func count(_ g: String) -> Int { lock.withLock { reads.filter { $0.group == g }.count } }
    }

    static func ring(depth: Int, once: [String] = [], cycle: [String], bound: [String] = [], keeps: Set<String> = [],
                     log: Log, keep: @escaping @Sendable (String, Int) -> Bool = { _, _ in false },
                     fail: String? = nil) -> Ring {
        let all = Array(Set(once + cycle)).sorted()
        let places = Dictionary(uniqueKeysWithValues: bound.map { g in (g, place(keptBase + all.firstIndex(of: g)!)) })
        return Ring(slots: (0..<depth).map { place(slotsBase + $0) }, once: once, cycle: cycle, bound: places, keeps: keeps,
                    place: { g in log.keep(g); return place(keptBase + all.firstIndex(of: g)!) }, keep: keep,
                    fill: { g, p in
                        if g == fail { throw Failure("\(g): unreadable") }
                        log.read(g, p)
                    })
    }

    /// The loop over `uses` uses: each acquired, checked, released; `after(m)` once use m is released.
    static func loop(_ ring: Ring, once: [String] = [], cycle: [String], uses: Int, after: (Int) -> Void = { _ in },
                     check: (Int, String, [String: Placed], Int?) -> Void) throws {
        for m in 0..<uses {
            let g = m < once.count ? once[m] : cycle[(m - once.count) % cycle.count]
            let (w, n) = try ring.acquire(m, g)
            check(m, g, w, n)
            if let n { ring.release(n) }
            after(m)
        }
    }

    /// **Before the signal, every use rides the ring**, read once each, never more than the slots ahead of the loop.
    @Test func beforeTheSignalEveryUseRidesTheRing() throws {
        let log = Log(), cycle = ["a", "b", "c"]
        let ring = Self.ring(depth: 2, once: ["first"], cycle: cycle, keeps: Set(cycle), log: log) { _, _ in true }
        defer { ring.stop() }
        try Self.loop(ring, once: ["first"], cycle: cycle, uses: 10) { m, g, w, n in
            #expect(n == m, "use \(m) is the ring's use \(n.map(String.init) ?? "none")")
            #expect(w["w"]!.offset == Self.slotsBase + m % 2)
            #expect(log.all.count <= m + 2, "use \(m): \(log.all.count) read")
        }
        #expect(ring.keptGroups.isEmpty)
        #expect(log.all.prefix(10).map(\.group) == ["first"] + Array((cycle + cycle + cycle).prefix(9)))
    }

    /// **Past the signal, the cycle is kept**: each group read at most once more into its own place, then read never
    /// again; two cycles past the signal, no use is the ring's.
    @Test func pastTheSignalTheCycleIsKept() throws {
        let log = Log(), cycle = ["a", "b", "c", "d"]
        let ring = Self.ring(depth: 2, cycle: cycle, keeps: Set(cycle), log: log) { _, _ in true }
        defer { ring.stop() }
        try Self.loop(ring, cycle: cycle, uses: 20, after: { m in if m == 3 { ring.signal() } }) { m, g, w, n in
            if m >= 12 {
                #expect(n == nil, "use \(m) still the ring's")
                #expect(w["w"]!.offset == Self.keptBase + cycle.firstIndex(of: g)!)
            }
        }
        #expect(Set(ring.keptGroups) == Set(cycle))
        // Read in the first cycle, in a slot or two the reader had read ahead, then once kept.
        for g in cycle { #expect(log.count(g) <= 3, "\(g) read \(log.count(g)) times") }
        #expect(log.all.filter { $0.place >= Self.keptBase }.count == 4)
    }

    /// **A policy that refuses keeps the ring**: past the signal, every use still a slot's, every use read.
    @Test func aPolicyThatRefusesKeepsTheRing() throws {
        let log = Log(), cycle = ["a", "b", "c"]
        let ring = Self.ring(depth: 2, cycle: cycle, keeps: Set(cycle), log: log) { _, _ in false }
        defer { ring.stop() }
        ring.signal()
        try Self.loop(ring, cycle: cycle, uses: 9) { m, _, _, n in #expect(n == m) }
        #expect(ring.keptGroups.isEmpty)
        #expect(log.all.count >= 9)
    }

    /// **A policy that holds two groups**: the first two asked kept, the others the ring's; a kept group never read again.
    @Test func aPolicyThatHoldsTwoGroups() throws {
        let log = Log(), cycle = ["a", "b", "c", "d"]
        final class Count: @unchecked Sendable { var n = 0; let lock = NSLock() }
        let count = Count()
        let ring = Self.ring(depth: 2, cycle: cycle, keeps: Set(cycle), log: log) { _, _ in
            count.lock.withLock { count.n += 1; return count.n <= 2 }
        }
        defer { ring.stop() }
        ring.signal()
        try Self.loop(ring, cycle: cycle, uses: 16) { _, _, _, _ in }
        let kept = ring.keptGroups
        #expect(kept.count == 2)
        for g in kept { #expect(log.all.filter { $0.group == g && $0.place >= Self.keptBase }.count == 1) }
        for g in cycle where !kept.contains(g) { #expect(log.count(g) >= 4, "\(g) read \(log.count(g)) times") }
    }

    /// **A reader waiting for a slot keeps at the signal**: one slot, held by the loop; the next use waits for it until
    /// the signal, then is kept, and the loop has it without giving the slot back.
    @Test func aReaderWaitingForASlotKeepsAtTheSignal() throws {
        let log = Log(), cycle = ["a", "b"]
        let ring = Self.ring(depth: 1, cycle: cycle, keeps: Set(cycle), log: log) { _, _ in true }
        defer { ring.stop() }
        let (_, n) = try ring.acquire(0, "a")
        #expect(n == 0)
        usleep(20_000)
        #expect(log.all.count == 1, "the reader read past a slot held")
        ring.signal()
        let done = DispatchSemaphore(value: 0)
        final class Got: @unchecked Sendable { var use: Int?? = .none }
        let got = Got()
        Thread.detachNewThread {
            got.use = try? ring.acquire(1, "b").use
            done.signal()
        }
        #expect(done.wait(timeout: .now() + 2) == .success, "the loop waited on a slot it holds")
        #expect(got.use == .some(nil))
        ring.release(0)
    }

    /// **Groups kept from the start are read once, at their first use**, in the loop's order among the ring's.
    @Test func boundGroupsAreReadOnceInTheLoopsOrder() throws {
        let log = Log(), once = ["first"], cycle = ["k", "a", "b"]
        let ring = Self.ring(depth: 2, once: once, cycle: cycle, bound: ["first", "k"], log: log)
        defer { ring.stop() }
        try Self.loop(ring, once: once, cycle: cycle, uses: 10) { _, g, w, n in
            if g == "first" || g == "k" { #expect(n == nil && w["w"]!.offset >= Self.keptBase) } else { #expect(n != nil) }
        }
        #expect(log.all.prefix(4).map(\.group) == ["first", "k", "a", "b"])
        #expect(log.count("k") == 1 && log.count("first") == 1)
        #expect(ring.keptGroups == ["first", "k"])
    }

    /// **Every group kept from the start**: no slot, each read once, the reader done once the cycle is read.
    @Test func everyGroupBoundNeedsNoSlot() throws {
        let log = Log(), cycle = ["a", "b"]
        let ring = Self.ring(depth: 0, once: ["o"], cycle: cycle, bound: ["o", "a", "b"], log: log)
        defer { ring.stop() }
        try Self.loop(ring, once: ["o"], cycle: cycle, uses: 9) { _, _, _, n in #expect(n == nil) }
        #expect(log.all.map(\.group) == ["o", "a", "b"])
    }

    /// **What is used once is never kept**: past the signal, `once` still rides the ring.
    @Test func onceGroupsAreNeverKept() throws {
        let log = Log(), once = ["o1", "o2"], cycle = ["a"]
        let ring = Self.ring(depth: 2, once: once, cycle: cycle, keeps: ["a"], log: log) { _, _ in true }
        defer { ring.stop() }
        ring.signal()
        try Self.loop(ring, once: once, cycle: cycle, uses: 5) { m, _, _, n in if m < 2 { #expect(n != nil) } }
        #expect(ring.keptGroups == ["a"])
    }

    /// **A read that fails is the loop's**: thrown at the use that waits for it at the latest (at any use the loop asks
    /// once the reader has failed).
    @Test func aFailedReadIsThrownToTheLoop() throws {
        let log = Log(), cycle = ["a", "bad"]
        let ring = Self.ring(depth: 2, cycle: cycle, log: log, fail: "bad")
        defer { ring.stop() }
        _ = try? ring.acquire(0, "a")
        #expect(throws: Failure.self) { try ring.acquire(1, "bad") }
    }
}
