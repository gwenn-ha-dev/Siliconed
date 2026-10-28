import Foundation

/// **What a lending Mac says of itself** — the machine, its map, its diagnostic — so that a borrowing Mac
/// shows them where it would show its own: the « On: » menu, Settings › Other Macs, and the diagnostic
/// sheet in a mirror. Small, and sent whole at each join and whenever it changes (`SessionMessage.machine`).
///
/// **No `Date`, no sentence**: a map is told by its ISO date string and two flags (`isCurrent` is a
/// question about the lender's own macOS and build, which the borrower cannot ask), a diagnostic's stage
/// by its rank, a failure as a `RemoteRefusal`, what blocks a measure as a code — all said in the reader's language.

/// **A capacity map as the other end needs it**: the day it was measured, and whether it still describes
/// that Mac (`Diagnostic.Report.isCurrent`) and was taken at rest (`isProvisional`).
package struct RemoteMap: Codable, Equatable, Sendable {
    package var date: String
    package var current: Bool
    package var provisional: Bool
    package init(date: String, current: Bool, provisional: Bool) {
        self.date = date; self.current = current; self.provisional = provisional
    }
    package init(_ r: Diagnostic.Report) { self.init(date: r.date, current: r.isCurrent, provisional: r.isProvisional) }
}

/// Why the lender cannot measure now (`AppState.diagnosticBlocker`), as a code.
package enum RemoteBlocker: String, Codable, Equatable, Sendable {
    case measuring, rendering, installing
}

/// **Where the lender's diagnostic stands** (`DiagnosticPhase`, without its sentences).
package enum RemoteDiagnostic: Codable, Equatable, Sendable {
    case idle
    /// The stage under way is `Diagnostic.Stage.allCases[done]`, said by the reader in its language.
    case running(done: Int, total: Int)
    case finished(Diagnostic.Report)
    case failed(RemoteRefusal)
}

package struct RemoteMachine: Codable, Equatable, Sendable {
    /// `Diagnostic.Machine.chip`, and the memory in bytes.
    package var chip: String
    package var memoryBytes: Int
    /// B, as the lender computes it (also in the catalog: here for the menu, which has no catalog).
    package var ceiling: Int
    /// The ready models' ids.
    package var ready: [String]
    /// The map kept in the preferences, and the one this session's settings were read with.
    package var stored: RemoteMap?
    package var inUse: RemoteMap?
    /// The kept map is provisional and the lender is at rest: it is measured again before the next render.
    package var remeasuring: Bool
    /// `AppState.profileWarnings`, as the engine words them (the borrower says them in its language).
    package var warnings: [String]
    package var diagnostic: RemoteDiagnostic
    /// `nil`: the lender can measure now.
    package var blocker: RemoteBlocker?

    package init(chip: String, memoryBytes: Int, ceiling: Int, ready: [String], stored: RemoteMap? = nil, inUse: RemoteMap? = nil,
                 remeasuring: Bool = false, warnings: [String] = [], diagnostic: RemoteDiagnostic = .idle,
                 blocker: RemoteBlocker? = nil) {
        self.chip = chip; self.memoryBytes = memoryBytes; self.ceiling = ceiling; self.ready = ready
        self.stored = stored; self.inUse = inUse; self.remeasuring = remeasuring; self.warnings = warnings
        self.diagnostic = diagnostic; self.blocker = blocker
    }

    /// The map is made, current and taken at rest: the chip of « Chain » is green, as on the lender.
    package var measured: Bool { stored.map { $0.current && !$0.provisional } == true }

    /// What the menu keeps from one session to the next: all but the diagnostic, whose report is big
    /// and whose phase is the session's.
    package var forMenu: RemoteMachine {
        var m = self
        m.diagnostic = .idle; m.blocker = nil; m.warnings = []
        return m
    }
}
