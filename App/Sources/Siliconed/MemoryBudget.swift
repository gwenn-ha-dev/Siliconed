import Foundation
import SiliCore

/// **B, what a render may hold on this Mac**: the user's, within `range`, else the default (¾ · RAM − 2 GiB). The
/// engine plans every render under it before the first computation and refuses what no plan fits; the app only says it.
///
/// `SILICONED_MEMORY_BUDGET_GB` gives B itself, in GiB, over anything chosen: a small Mac's budget tried on a large one
/// (10 for 16 GB of RAM, 4 for 8), and the user's B as the app hands it to its engine's process (`EngineWorker`). The
/// processes that plan — the app's engine (`silicore-worker`), the command line — read it through `machineCeiling`, once,
/// and live under it, their fuse armed by that first reading; the app reads `budget`.
package enum MemoryBudget {
    /// **What B may be set to on this Mac**: from 4 GiB to its RAM less 4 GiB — a Mac of 8 GB has no choice (4), one of
    /// 16 goes from 4 to 12, one of 32 from 4 to 28 (decided 16/11/2026). The default lies in it on every Mac.
    package static var range: ClosedRange<Int> { range(memory: Int(ProcessInfo.processInfo.physicalMemory)) }

    package static func range(memory: Int) -> ClosedRange<Int> {
        let least = 4 << 30
        return least...max(least, memory - (4 << 30))
    }

    /// B as the environment gives it, read once.
    private static let given: Int? = {
        guard let text = ProcessInfo.processInfo.environment["SILICONED_MEMORY_BUDGET_GB"], let gib = Double(text), gib > 0 else { return nil }
        return Int(gib * Double(1 << 30))
    }()

    private static let lock = NSLock()
    nonisolated(unsafe) private static var userBudget: Int?

    /// **The user's B** (the app's setting), held within `range`; `nil`: the default.
    package static var chosen: Int? {
        get { lock.withLock { userBudget } }
        set {
            let r = range
            lock.withLock { userBudget = newValue.map { min(max($0, r.lowerBound), r.upperBound) } }
        }
    }

    /// The default: ¾ · RAM − 2 GiB.
    package static var standard: Int { Int(ProcessInfo.processInfo.physicalMemory / 4 * 3) - (2 << 30) }

    /// B, as this process reads it now: the environment's, else the user's, else the default — the app's, which launches
    /// its worker under it. A plan reads `machineCeiling()`, the B its process's fuse holds.
    package static var budget: Int { given ?? chosen ?? standard }

    /// **B as the process first read it, which it keeps, and its fuse**: the engine's footprint read every millisecond
    /// on a thread of its own (`FootprintWatch`), the process stopped (status 3) the moment it holds more — whatever
    /// holds it: the arena, Core AI, a buffer of the app's. One mechanism for the app's engine and `siliconed` alike; the
    /// app itself never arms it (its engine is another process, `EngineWorker`). Armed by the first reading, which every
    /// plan makes.
    private static let armed: (budget: Int, fuse: FootprintWatch) = {
        let b = budget
        return (b, FootprintWatch(stopPast: b))
    }()

    package static func machineCeiling() -> Int { armed.budget }

    /// How the fuse's last words begin (`FootprintWatch`): what tells a stop past B from a crash (`EngineWorker`).
    package static let fuseWords = "SiliCore: the footprint"

}
