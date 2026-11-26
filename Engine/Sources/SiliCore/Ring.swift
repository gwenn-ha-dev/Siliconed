import Foundation

/// **The weights B does not keep, read ahead into a ring of slots**: a thread of its own reads
/// the n-th use of the ring into slot n mod `depth` as soon as use n − depth is released, while the loop computes on the
/// others; the loop takes a use only once it is read. What is read in what order is the loop's own (`order`), known
/// before it starts: the ring knows no model, and what is acquired must be what was read.
///
/// A slot is released when the last call that read it has completed (`Block.settled`, `Layers` a layer done): the reader never writes
/// a buffer the GPU may still read, and nothing but the arena holds what it reads (its staging is the arena's).
///
/// **What it costs is the conversion, not the disk** (measured 17/11 on the M1 Pro, a transformer block in W8A16, 172.8 MiB, over
/// 171 to 270 reads; the codes transposed on one core then, by NEON tiles on every core since 18/11, `Weights.read`):
/// the file 36 ms past the system's cache (5.0 GB/s), the codes transposed to `[in, out]` and the scales rounded 280 ms on one core, for ~540 ms of the GPU's on the block. Where a GPU computes a block in less than
/// the conversion takes, the ring falls behind: hence the conversion on every core; the file's codes written once
/// transposed would be the next answer — not measured.
///
/// **The weights kept go through the same reader**: read once into a place of their own, in the loop's order among the
/// ring's uses, while the loop computes on those already read — the arena is not filled before the first call.
///
/// **Two phases** (the ring's plan changes on the way): every use of the loop is resolved by the reader, kept or a slot.
/// Until the loop's `signal`, a group not kept from the start goes through the slots; past it, each use of a group that
/// may be kept (`keeps`: the cycle's) is asked of `keep` as it comes, and a group kept is read once more into a place
/// of its own (`place`), then read never again. A reader that waits for a slot decides again when the signal comes, or
/// when a slot is released. The loop waits for nothing new: `signal` is given, never waited for.
public final class Ring: @unchecked Sendable {
    public let depth: Int
    private let slots: [[String: Placed]]
    private let once: [String], cycle: [String]
    private let condition = NSCondition()
    /// The ring's uses read and released; the slots never written yet.
    private var ringUses = 0, released = 0, unwritten: Int
    private var stopped = false, finished = false, signalled = false, peeking = true
    /// The loop's uses the ring holds, by their index in the loop's order: the ring use each is, and its group — until
    /// released (a use acquired stays, for `peek`).
    private var resolved: [Int: (use: Int, group: String)] = [:]
    /// The groups kept (from the start, or past the signal), their places; those read, in the order read.
    private var places: [String: [String: Placed]]
    private var read: Set<String> = []
    private var kept: [String] = []
    /// The groups kept past the signal so far (`kept` counts those from the start too).
    private var inFlight = 0
    private var failure: (any Error)?

    /// The reader's work, let go by `stop` once it has finished: what it captures (the arena) is then held by nothing here.
    private var work: Work?
    private struct Work {
        let keeps: Set<String>
        let place: @Sendable (String) throws -> [String: Placed]
        let keep: @Sendable (String, Int) -> Bool
        let kept: @Sendable (Int) -> Void
        let fill: @Sendable (String, [String: Placed]) throws -> Void
    }

    /// The loop's uses: each of `once` in turn, then `cycle` over and over — the order it acquires them in. `bound`:
    /// groups kept from the start, their places taken (read once, at their first use). `keeps`: the groups the reader may
    /// keep past the signal; `place(g)`, a kept group's own place, taken when kept; `keep(g, unwritten)`, whether to keep
    /// g now, the slots never written yet counted. `fill(g, place)`: g's weights read into place
    /// (`Weights.read`).
    public init(slots: [[String: Placed]], once: [String], cycle: [String], bound: [String: [String: Placed]] = [:], keeps: Set<String> = [],
                place: @escaping @Sendable (String) throws -> [String: Placed] = { g in preconditionFailure("\(g): no place") },
                keep: @escaping @Sendable (String, Int) -> Bool = { _, _ in false },
                kept onKeep: @escaping @Sendable (Int) -> Void = { _ in },
                fill: @escaping @Sendable (String, [String: Placed]) throws -> Void) {
        depth = slots.count
        self.slots = slots
        (self.once, self.cycle) = (once, cycle)
        unwritten = slots.count
        places = bound
        work = Work(keeps: keeps, place: place, keep: keep, kept: onKeep, fill: fill)
        let thread = Thread { [self] in run() }
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    /// The group of the loop's use m, nil past the last.
    public func group(_ m: Int) -> String? {
        m < once.count ? once[m] : cycle.isEmpty ? nil : cycle[(m - once.count) % cycle.count]
    }

    private enum Decision { case skip, bound([String: Placed]), keep, ring(Int), stop }

    private func run() {
        defer { condition.lock(); finished = true; condition.broadcast(); condition.unlock() }
        var m = 0
        while let g = group(m) {
            condition.lock()
            guard !stopped, let w = work else { condition.unlock(); return }
            var decision: Decision = .stop
            if read.contains(g) {
                // A group read into its place: the loop finds it there (`acquire`). Past `once`, with every group of the
                // cycle kept, nothing is left to read.
                decision = m >= once.count && Set(cycle).isSubset(of: read) ? .stop : .skip
            } else if let p = places[g] {
                decision = .bound(p)
            } else {
                while !stopped {
                    if signalled && w.keeps.contains(g) && w.keep(g, unwritten) { decision = .keep; break }
                    if ringUses - released < depth { decision = .ring(ringUses); break }
                    condition.wait()
                }
            }
            condition.unlock()
            do {
                switch decision {
                case .stop: return
                case .skip: break
                case .bound(let p):
                    try w.fill(g, p)
                    record { read.insert(g); kept.append(g) }
                case .keep:
                    let p = try w.place(g)
                    try w.fill(g, p)
                    record { places[g] = p; read.insert(g); kept.append(g); inFlight += 1 }
                    w.kept(inFlight)
                case .ring(let n):
                    try w.fill(g, slots[n % depth])
                    record {
                        if n < depth { unwritten -= 1 }
                        ringUses = n + 1
                        resolved[m] = (n, g)
                    }
                }
            } catch {
                condition.lock(); failure = error; condition.broadcast(); condition.unlock()
                return
            }
            m += 1
        }
    }

    /// A read's `then` done, under the lock, the waiters woken.
    private func record(then: () -> Void) {
        condition.lock(); then(); condition.broadcast(); condition.unlock()
    }

    /// **The loop's use m, `group`'s, once read** (waits for it): a slot the reader read it into (its ring use, to
    /// `release`), or the group's own place.
    public func acquire(_ m: Int, _ group: String) throws -> (weights: [String: Placed], use: Int?) {
        condition.lock(); defer { condition.unlock() }
        while true {
            if let failure { throw failure }
            if let r = resolved[m] {
                precondition(r.group == group, "the loop's use \(m) is \(r.group)'s, not \(group)'s")
                return (slots[r.use % depth], r.use)
            }
            if read.contains(group) { return (places[group]!, nil) }
            precondition(!finished, "\(group): the loop's use \(m), past what the reader read")
            condition.wait()
        }
    }

    /// **The loop's use m's weights, once read, left for the loop to acquire** (waits for them): the slot the reader read
    /// it into — not written again before the loop releases use m —, or the group's own place. Nil past the loop's last
    /// use, or once the ring stops or `stopPeeking` is called: a peek never outlives the arena.
    public func peek(_ m: Int) -> [String: Placed]? {
        condition.lock(); defer { condition.unlock() }
        while true {
            if failure != nil || stopped || !peeking { return nil }
            // Acquired and released already: its slot may hold another use's weights.
            if let r = resolved[m] { return r.use >= released ? slots[r.use % depth] : nil }
            guard let g = group(m) else { return nil }
            if read.contains(g) { return places[g]! }
            if finished { return nil }
            condition.wait()
        }
    }

    /// Every peek under way returns nil, and none waits again.
    public func stopPeeking() {
        condition.lock(); peeking = false; condition.broadcast(); condition.unlock()
    }

    /// **The signal**: what Core AI holds has been read (`Stage.signal`); from now on the reader may keep. Given, never
    /// waited for.
    public func signal() {
        condition.lock(); signalled = true; condition.broadcast(); condition.unlock()
    }

    /// The policy may say otherwise now (a reading changed): a reader waiting for a slot decides again.
    public func wake() {
        condition.lock(); condition.broadcast(); condition.unlock()
    }

    /// Ring use n done: its slot may take the use `depth` after it.
    public func release(_ n: Int) {
        condition.lock()
        released = max(released, n + 1)
        resolved = resolved.filter { $0.value.use >= released }
        condition.broadcast(); condition.unlock()
    }

    /// The reader stopped, and waited for: a read under way finishes first. Past it, nothing the reader holds (the
    /// arena its `fill` writes into) is held by it.
    public func stop() {
        condition.lock(); stopped = true; condition.broadcast()
        while !finished { condition.wait() }
        work = nil
        condition.unlock()
    }

    /// **Whether the reader may still keep a group in flight**: one it may keep (`keeps`) never read into a place of its
    /// own, while it reads.
    public var deciding: Bool {
        condition.lock(); defer { condition.unlock() }
        guard let w = work, !finished, !stopped else { return false }
        return !w.keeps.isSubset(of: read)
    }

    /// The groups kept so far, in the order read.
    public var keptGroups: [String] {
        condition.lock(); defer { condition.unlock() }
        return kept
    }
}
