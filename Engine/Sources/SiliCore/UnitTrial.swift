/// **The ANE's unit, decided once** (`Block.settle`, `ThisMac.neuralUnits`): the unit a share is counted in is one of
/// `candidates`, and no formula predicts which is faster on a Mac — a call's own cost and the finer rounding of the share
/// pull the other way, by 0.4 % to 8 % a cycle, by the model and the Mac. So the block that first
/// finds its key undecided runs its cycles at each candidate in turn, in its own run (whose functions are specialized anyway), and keeps
/// the unit of the faster cycle for good.
///
/// **A unit's cycles**: the first is cold — its functions are specialized at their first call, which no later cycle pays —
/// and left out; the `measured` cycles after it are compared, each unit by its best share predicted (`Best`) — a unit is
/// worth what it does at its own best share, which a cycle run at the other's does not show.
/// One measured cycle: a block's client runs few cycles (8 for most of today's), the
/// trial takes `candidates.count × (1 + measured)` of them — 4 — and the run goes on at the winner for the rest, where
/// its pace (`ThisMac.paces`) is taken from a block's fourth cycle at the kept unit on. More measured cycles would
/// compare more fairly and leave a run of 8 cycles too few at the winner to time it. A unit's cold cycle moves
/// no share; its measured cycle runs at the share the unit before predicted best, rounded to its own units, and tells
/// what its own best share would do.
struct UnitTrial: Equatable, Sendable {
    /// The units a share may be counted in, the tallest first (256 and 128 measured slower a row: 1.075 ms at 256 against
    /// 0.982 at 512 and 0.965 at 1024 on the mini, 19/11). Both only where the shareable rows hold two units of the tallest.
    static let candidates = [1024, 512]
    static let measured = 1

    /// The candidates for a share of `rows` shareable rows: both where two units of 1024 fit, else 512 alone — a unit the
    /// share cannot split in two leaves the ANE none or all.
    static func candidates(rows: Int) -> [Int] {
        let both = candidates.filter { rows / $0 >= 2 }
        return both.isEmpty ? [candidates.last!] : both
    }

    /// What the cycle just recorded asks of the block, at its next cycle's start.
    enum Next: Equatable {
        /// Run again at this unit.
        case stay
        /// Go on at another unit.
        case move(to: Int)
        /// Decided: this unit, for good.
        case settle(Int)
    }

    /// A unit's best: the seconds a cycle is predicted to take at the best share of whole units, and that share's ANE rows
    /// (`Block.bestShare`).
    struct Best: Equatable, Sendable { let seconds: Double, rows: Int }

    let order: [Int]
    /// The unit under trial now (an index into `order`), its cycles done, each unit's best of its measured cycles.
    private var at = 0
    private var cycles: [Int: Int] = [:]
    private(set) var best: [Int: Best] = [:]

    init(_ order: [Int]) {
        precondition(order.count >= 2, "a trial compares two units at least")
        self.order = order
    }

    var unit: Int { order[at] }
    /// Whether the cycle that ran last — the one the next `record` is given — was this unit's cold one.
    var cold: Bool { cycles[unit, default: 0] == 0 }

    /// **A cycle at the unit under trial, done**, its best share predicted from its calls (`Best`): the cold one counts
    /// for nothing. The units are compared by these, each at its own best share, not by the cycles as they ran: a cycle
    /// runs at the share the cycle before set, which is the other unit's.
    mutating func record(_ cycle: Best) -> Next {
        let n = cycles[unit, default: 0]
        cycles[unit] = n + 1
        if n > 0, best[unit].map({ cycle.seconds < $0.seconds }) ?? true { best[unit] = cycle }
        guard n + 1 > Self.measured else { return .stay }
        if at + 1 < order.count {
            at += 1
            return .move(to: order[at])
        }
        // The first of the fastest: the taller unit, fewer calls, where two are as fast.
        return .settle(order.min { (best[$0]!.seconds, order.firstIndex(of: $0)!) < (best[$1]!.seconds, order.firstIndex(of: $1)!) }!)
    }

    /// **The ANE's rows `rows`, counted in units of `unit`**: the nearest whole count, the ANE and the GPU each at least
    /// one unit of `shareable` rows.
    static func units(rows: Int, of unit: Int, shareable: Int) -> Int {
        min(max((rows + unit / 2) / unit, 1), max(shareable / unit - 1, 1))
    }
}
