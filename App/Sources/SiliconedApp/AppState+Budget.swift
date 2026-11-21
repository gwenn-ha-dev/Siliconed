// **`AppState`: the user's B** (Settings ▸ Macs ▸ This Mac), handed to the next engine process.

import Foundation
import Siliconed

extension AppState {
    static let memoryBudgetKey = "memoryBudget"

    /// **The user's B** (Settings ▸ Macs ▸ This Mac), in bytes, within `MemoryBudget.range`; `nil`: the default
    /// (¾ · RAM − 2 GiB). Handed to the next engine process (`EngineWorker`): a render under way keeps the B it began under.
    var memoryBudget: Int? {
        get { _ = memoryBudgetChanges; return MemoryBudget.chosen }
        set {
            MemoryBudget.chosen = newValue.flatMap { $0 == MemoryBudget.standard ? nil : $0 }
            if let b = MemoryBudget.chosen { UserDefaults.standard.set(b, forKey: Self.memoryBudgetKey) } else {
                UserDefaults.standard.removeObject(forKey: Self.memoryBudgetKey)
            }
            memoryBudgetChanges += 1
            if mirror == nil { memoryCeiling = MemoryBudget.budget }
        }
    }

    /// The user's B as it was stored, given to the engine's budget before anything reads it.
    static func restoreMemoryBudget() {
        let stored = UserDefaults.standard.integer(forKey: memoryBudgetKey)
        MemoryBudget.chosen = stored > 0 ? stored : nil
    }
}
