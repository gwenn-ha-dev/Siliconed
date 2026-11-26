import Foundation

/// **A stage opened beside the one that runs.** A stage may open — its arena taken, its residents and kept groups read,
/// its functions loaded — while another runs, where it takes there the plan it would take once the other has gone, where
/// the other has nothing left to decide in flight, and where both were read on this Mac; otherwise it opens after the
/// other's close, as it would alone. Nothing but B decides it: two stages open at once are a cost in memory like any
/// other (decided 26/11/2026: with tens of GiB under B, waiting for a stage to end and its memory to go before the next
/// one opened was time lost for nothing).
///
///     the one that runs   judged at its peak (`peak`): the footprint now, what its share has left to grow to its
///                         reading, its arena not yet written, what it was promised — `holds`'s rule
///     the one beside      planned on that peak; refused there, or another form than on the other's rest: after
///     from the opening    both frozen (`frozen`): they read nothing and decide nothing in flight, the footprint holding
///                         the other's — the first's readings are those before, the second's from its thaw
///     the first closed    the second thawed: it reads from its floor beside, short by what left, and keeps no first
///                         reading nor first reserve (`trailing`)
///
/// The client says which stage an opening goes beside (`Ahead`): the next one the process opens, or the one open now. It
/// calls the second once done with the first (given what the first made): a call beside the
/// first changes no reading (both frozen), and B holds them both at their peaks.
extension Stage {
    /// **A stage's close, seen without holding it**: what a wait for it reads (`awaitClose`), never the stage itself —
    /// the last reference let go there would run its `deinit` under the wait's lock.
    final class Life: @unchecked Sendable {
        private let lock = NSLock()
        private var ended = false
        var closed: Bool { lock.withLock { ended } }
        func close() { lock.withLock { ended = true } }
    }

    /// A stage not held, and its close.
    final class WeakStage: @unchecked Sendable {
        weak var stage: Stage?
        let life: Life
        init(_ stage: Stage) { (self.stage, life) = (stage, stage.life) }
    }

    /// **An opening beside `running`**, until the stage it opened adopts it (`adopt`): let go unadopted — the opening
    /// threw —, the one that runs thaws.
    public final class Pairing: @unchecked Sendable {
        let running: WeakStage
        private let lock = NSLock()
        private var adopted = false
        init(_ running: Stage) { self.running = WeakStage(running) }

        /// The stage opened: thawed at the close of the one that runs, or now where its close began.
        func adopt(_ stage: Stage) {
            lock.withLock { adopted = true }
            guard let r = running.stage else { stage.thaw(); return }
            let open = r.lock.withLock { () -> Bool in
                guard !r.closing else { return false }
                r.besides.append(WeakStage(stage))
                return true
            }
            if !open { stage.thaw() }
        }

        deinit { if !adopted { running.stage?.thaw() } }
    }

    /// **An opening ahead**: what a thread that opens a stage ahead binds (`ahead`), the stage it goes beside said by
    /// the client — the next one the process opens (`next`: the client's last stage before it), or the one open now
    /// (`open`: at its last call) — or none (`end`). The opening waits for it (`planned`).
    public final class Ahead: @unchecked Sendable {
        enum Beside { case stage(WeakStage), after([WeakStage]), alone }
        private let condition = NSCondition()
        private var beside: Beside?
        private var cancelled = false
        public init() {}

        /// **The next stage the process opens**, not one opened ahead: the one the opening goes beside.
        public func next() { Self.lock.withLock { Self.armed = self } }

        /// **The stage open now**: the one the opening goes beside; none, alone; more than one, after them all.
        public func open() {
            let all = Stage.livesLock.withLock { Array(Stage.lives.values) }
            resolve(all.count == 1 ? .stage(all[0]) : all.isEmpty ? .alone : .after(all))
        }

        /// **None comes**: an opening that waits goes alone — or, `cancelled`, throws.
        public func end(cancelled: Bool = false) {
            Self.lock.withLock { if Self.armed === self { Self.armed = nil } }
            condition.lock()
            self.cancelled = self.cancelled || cancelled
            condition.broadcast()
            condition.unlock()
            resolve(.alone)
        }

        func resolve(_ b: Beside) {
            condition.lock()
            if beside == nil { beside = b }
            condition.broadcast()
            condition.unlock()
        }

        func wait() throws -> Beside {
            condition.lock(); defer { condition.unlock() }
            while beside == nil && !cancelled { condition.wait() }
            if cancelled { throw CancellationError() }
            return beside!
        }

        nonisolated(unsafe) private static var armed: Ahead?
        private static let lock = NSLock()

        /// A stage opened, not ahead: the opening armed goes beside it.
        static func fire(_ stage: Stage) {
            let a = lock.withLock { () -> Ahead? in defer { armed = nil }; return armed }
            a?.resolve(.stage(WeakStage(stage)))
        }
    }

    /// **The opening ahead the current task's stages open under** (`planned`), bound by the thread that opens them.
    @TaskLocal public static var ahead: Ahead?

    /// The stages open in the process.
    nonisolated(unsafe) static var lives: [ObjectIdentifier: WeakStage] = [:]
    static let livesLock = NSLock()
    func live(_ on: Bool) { Self.livesLock.withLock { Self.lives[ObjectIdentifier(self)] = on ? WeakStage(self) : nil } }

    /// **`plan`'s opening beside this stage**, judged on its peak, or nil where it opens after its close: refused on
    /// the peak, another form than on this stage's rest, a set of functions never read on this Mac (this one's, or the
    /// one opened beside: a first reading is taken alone), something this stage may still decide in flight (a group
    /// its ring may keep). Frozen first, its decisions stopped, so that nothing it takes escapes the judgment; thawed
    /// where nil.
    func beside<T>(measured: ThisMac, _ plan: (_ floor: Int) throws -> (T, Plan)) -> (value: T, plan: Plan)? {
        let froze = lock.withLock { () -> Bool in
            guard !frozen, !closing else { return false }
            frozen = true
            return true
        }
        guard froze else { return nil }
        let share = read().share
        lock.withLock { frozenShare = share }
        let key = lock.withLock { readingKey }
        guard (self.measured.reading(key, budget: self.plan.budget) ?? -1) >= 0, !decides,
              let a = try? plan(peak()), let b = try? plan(before),
              Self.key(a.1.declaration) == Self.key(b.1.declaration), a.1.kept == b.1.kept, a.1.depth == b.1.depth,
              (measured.reading(Self.key(a.1.declaration), budget: a.1.budget) ?? -1) >= 0
        else { thaw(); return nil }
        return (a.0, a.1)
    }

    /// **Whether this stage may still take something in flight**: a group of its cycle its ring may keep and has not.
    var decides: Bool {
        let keepable = plan.depth > 0 && plan.declaration.cycle.contains { !plan.kept.contains($0) && plan.sizes.rides($0) }
        guard keepable else { return false }
        return lock.withLock { ring }.map(\.deciding) ?? true
    }

    /// **Where this stage takes the process at most from now on, frozen**: the footprint now, what its share has left to
    /// grow to the larger of its reading on this Mac and the largest read in it, what is left of its reserve, what it
    /// was promised — its arena counted whole as planned (`held`): what is not allocated or not written yet counts as
    /// share to come.
    func peak() -> Int {
        let now = footprint()
        let key = lock.withLock { readingKey }
        let reading = measured.reading(key, budget: plan.budget) ?? 0
        return lock.withLock {
            let held = max(arena?.allocated ?? 0, plan.arena), current = now - plan.floor - held
            let share = max(largest, reading, current)
            return now + (share - current) + promised + left(share)
        }
    }

    /// Reading and deciding again: the stage it was opened beside closed (or the one opened beside it never came).
    /// What the watch saw meanwhile held the other's.
    func thaw() {
        lock.withLock { (frozen, frozenShare) = (false, nil) }
        watch?.reset()
    }

    /// **The stage closed** (`close`, or let go): the stages opened beside it thawed, the openings that wait for it woken.
    func ended() {
        life.close()
        let besides = lock.withLock { () -> [WeakStage] in
            closed = true
            defer { self.besides = [] }
            return self.besides
        }
        for b in besides { b.stage?.thaw() }
        Self.closings.lock(); Self.closings.broadcast(); Self.closings.unlock()
    }

    private static let closings = NSCondition()

    /// **The wait for `stages`' close**, its seconds.
    static func awaitClose(_ stages: [WeakStage]) -> Double {
        let t = DispatchTime.now().uptimeNanoseconds
        closings.lock()
        while stages.contains(where: { !$0.life.closed }) { closings.wait() }
        closings.unlock()
        return Functions.seconds(since: t)
    }
}
