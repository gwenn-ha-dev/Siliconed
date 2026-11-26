import Foundation

/// **What the stages let go leave behind, waited for only where it changes something.** A stage let go gives its heap
/// back within 0.05 to 0.5 s, then 100 to 160 MiB of Core AI's or Metal's 1 to 2 s after (18/11/2026, the M1 Pro); a
/// wait for it at each close, 2 s at most, before the next stage is planned costs 0.6 to 2.0 s at each close of a chain
/// of stages, 2.0 at a transformer's (26/11/2026, the M1 Pro, the trace). So `close` waits for nothing and says where the
/// process comes back to (`rested`); the next stage is planned on the footprint now, what trails included — its P the
/// higher by it, never the lower: what trails only leaves — and waits only where that changes its plan (another form,
/// or a refusal, than on the rest), or where its functions were never read on this Mac: a stage's reading is the
/// footprint past its floor, short by what leaves meanwhile, and a reading is kept at its largest (`ThisMac.keep`), so
/// a short one does no harm only where one was kept before it. A stage planned on what trails keeps no first reading
/// nor first reserve (`Opening.trailing`, `keep`): a function met in flight is read at a later run.
extension Stage {
    /// Under `restLock`: the footprint the process comes back to once what the last stage let go has left — the
    /// process's before it allocated (not the least of every stage's: a floor grown since, a stage still open, would make
    /// every opening wait its 2 s); nil, nothing let go is awaited.
    nonisolated(unsafe) private static var resting: Int?
    private static let restLock = NSLock()

    /// A stage let go, opened over `before`: the process comes back there.
    static func rested(on before: Int) { restLock.withLock { resting = before } }

    /// **The floor a judgment of what comes later takes** (an admission before the computation whose result the stage
    /// reads, a preparation): the footprint the process comes back to, or the footprint now.
    public static func restingFloor() -> Int {
        let now = Memory.floor()
        return min(now, restLock.withLock { resting } ?? now)
    }

    /// **The wait for what the stages let go**: until the footprint is back within `margin` of the rest, 2 s at most;
    /// past them, what is left lies in the floors that follow. What it left, against the rest.
    @discardableResult
    public static func settle() -> Settled {
        let t = DispatchTime.now().uptimeNanoseconds
        var f = Memory.footprint()
        let rest = restLock.withLock { resting } ?? f
        while f > rest + margin && Functions.seconds(since: t) < 2 { usleep(2000); f = Memory.footprint() }
        restLock.withLock { resting = nil }
        return Settled(footprint: f, before: rest, seconds: Functions.seconds(since: t), alive: Arena.alive, held: Arena.lingering)
    }

    /// **How a stage was opened** (`planned`): when its planning began, the seconds it waited for what the stages
    /// before it let go, whether its floor holds what trails of them.
    public struct Opening: Sendable {
        public var start: UInt64?
        public var waited = 0.0
        public var trailing = false
        /// Opened beside a stage that runs (`beside`); the seconds it waited for that stage to open (`Ahead`), and for the
        /// close of the one it could not go beside.
        public var beside: Pairing?
        public var ahead = 0.0, after = 0.0
        public init() {}
    }

    /// **The plan a stage opens on, and the wait only where it changes something**: `plan` chooses on a floor (the
    /// client's forms judged there, `Plan.on`, `choose`); called on the footprint now, and where what the stages before
    /// let go still trails, on the rest too: the same choice and functions read on this Mac, the plan on now, no wait;
    /// else the wait (`settle`), then the plan on the footprint it leaves.
    public static func planned<T>(measured: ThisMac = .current, _ plan: (_ floor: Int) throws -> (T, Plan)) throws -> (value: T, plan: Plan, opening: Opening) {
        var opening = Opening()
        opening.start = DispatchTime.now().uptimeNanoseconds
        // Opened ahead (`Ahead`): beside the stage the client said, where B holds them both (`beside`); else after it.
        if let ahead = Self.ahead {
            let t = DispatchTime.now().uptimeNanoseconds
            let said = try ahead.wait()
            opening.ahead = Functions.seconds(since: t)
            switch said {
            case .stage(let w):
                if let running = w.stage, let b = running.beside(measured: measured, plan) {
                    opening.beside = Pairing(running)
                    opening.trailing = true
                    return (b.value, b.plan, opening)
                }
                opening.after = awaitClose([w])
            case .after(let all):
                opening.after = awaitClose(all)
            case .alone:
                break
            }
        }
        let now = Memory.floor()
        guard let rest = restLock.withLock({ resting }), now > rest + margin else {
            restLock.withLock { resting = nil }
            let (v, p) = try plan(now)
            return (v, p, opening)
        }
        if let a = try? plan(now), let b = try? plan(rest), !waits(now: a.1, rested: b.1, measured: measured) {
            opening.trailing = true
            return (a.0, a.1, opening)
        }
        opening.waited = settle().seconds
        let (v, p) = try plan(Memory.floor())
        return (v, p, opening)
    }

    /// **Whether a stage planned on what trails waits for it**: where on the rest it would take other functions (a
    /// taller form), or where these were never read on this Mac (their first reading taken clear).
    static func waits(now: Plan, rested: Plan, measured: ThisMac) -> Bool {
        let key = Self.key(now.declaration)
        guard key == Self.key(rested.declaration), let r = measured.reading(key, budget: now.budget) else { return true }
        return r < 0
    }
}
