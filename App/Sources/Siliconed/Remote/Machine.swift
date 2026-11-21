import Foundation

/// **What a lending Mac says of itself** — the machine and its ready models — so that a borrowing Mac shows them where it
/// would show its own: the « On: » menu and Settings › Other Macs. Small, and sent whole at each join and whenever it
/// changes (`SessionMessage.machine`). No `Date`, no sentence: all said in the reader's language.
package struct RemoteMachine: Codable, Equatable, Sendable {
    /// `Mac.chip`, and the memory in bytes.
    package var chip: String
    package var memoryBytes: Int
    /// `hw.model` ("MacBookPro18,3"): the icon the borrower draws beside « Chain ».
    package var hardwareModel: String
    /// B, as the lender computes it (also in the catalog: here for the menu, which has no catalog).
    package var ceiling: Int
    /// The ready models' ids.
    package var ready: [String]

    package init(chip: String, hardwareModel: String, memoryBytes: Int, ceiling: Int, ready: [String]) {
        self.chip = chip; self.hardwareModel = hardwareModel; self.memoryBytes = memoryBytes; self.ceiling = ceiling; self.ready = ready
    }
}
