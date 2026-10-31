import Foundation
import Synchronization

/// **What this Mac offers a render, B — fixed by its RAM, and nothing else.**
///
///     physical RAM ── ¾ · RAM − 2 GiB ── B (`ceiling`) ─┬─ Preflight: the lean plan's need ≤ B, or refused
///                                                       ├─ the app: greyed out ahead, « needs a Mac with N GB »
///                                                       └─ MemoryPlan: the fastest plan whose need ≤ B
///
/// **B is a contract the engine imposes on itself** (12/10/2026): macOS has no reservation,
/// and reading the kernel's state to decide oscillated, read stale figures and arrived after the cliff
/// (measured); a reserve and a compression credit calibrated on six renders of one machine refused
/// renders the kernel would have served. So nothing here reads the kernel: the same render on the same
/// Mac gets the same answer and the same plan, always — give or take what the process itself holds.
/// What the others hold is theirs; the promise is that Siliconed takes at most B.
///
/// **What B covers**: Siliconed's whole process, `phys_footprint` — what it holds when the render
/// starts (the app's interface and history, the process's runtime: `MemoryNeed.Request.base`,
/// read from the process itself) plus the render's computed need (`MemoryNeed`). Not the page cache
/// of the maps it reads (clean, file-backed, nobody's footprint: `MemoryPlan`), nor the frameworks'
/// constants no shape gives (`OperatorFootprint`), which the render's check reports.
package enum MemoryBudget {

    /// **B = ¾ · RAM − 2 GiB** (decided on 12/10/2026): 8 GiB → 4, 16 → 10, 24 → 16, 32 → 22, 64 → 46.
    /// A RAM is counted in GiB (`ProcessInfo.physicalMemory` of a "16 GB" Mac is 16 GiB), and so is B.
    /// **Below 8 GiB nothing is aimed at**: the formula still answers — 6 GiB gives 2.5 (2.68 GB),
    /// which refuses Z-Image (bf16) from 1024² (3.28 GB) and takes its 512² and every Qwen-Image-2.1
    /// request (≤ 2.46 GB, 12/10) — nothing measured there; under 2⅔ GiB, 0: every render refused.
    package static func ceiling(physical: Int) -> Int { max(0, physical / 4 * 3 - 2 * gib) }

    /// **The smallest RAM whose B covers `need`**, in whole GiB — the inverse of `ceiling(physical:)`.
    package static func physicalNeeded(for need: Int) -> Int {
        var n = max(1, ((max(0, need) + 2 * gib) * 4 + 3 * gib - 1) / (3 * gib))
        while n > 1, ceiling(physical: (n - 1) * gib) >= need { n -= 1 }
        while ceiling(physical: n * gib) < need { n += 1 }
        return n
    }

    /// **The memory sizes Macs are sold with**, in GB (Apple silicon, 2020–2026). A table, on purpose: a
    /// « needs a Mac with 9 GB » answers nobody, the question is which Mac. It moves once a chip
    /// generation; a size missing here falls to the next one listed, never to a figure that is not sold.
    package static let soldSizes = [8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512]

    /// **The N of the app's « Needs a Mac with N GB of memory »**: the smallest size sold whose B covers
    /// `need` (`physicalNeeded` rounded up to `soldSizes`); beyond the largest, the raw inverse.
    package static func macNeeded(for need: Int) -> Int {
        let raw = physicalNeeded(for: need)
        return soldSizes.first { $0 >= raw } ?? raw
    }

    /// **B for this Mac**: the forcing (`SILICONED_MEMORY_BUDGET_GB`) when set, then a simulated RAM
    /// (`SILICONED_SIMULATED_RAM`, only where a process opts in: `allowSimulation`), otherwise `ceiling(physical:)` of
    /// its RAM. What the preflight refuses on, the app greys out with, and the plan is chosen in.
    package static func machineCeiling(environment: [String: String]? = nil) -> Int {
        let live = environment == nil
        let environment = environment ?? liveEnvironment([forcingVariable, simulatedRAMVariable])
        if let forced = forcedCeiling(environment) { return forced }
        // The live environment is read for a simulation only where it was asked for (`allowSimulation`).
        if (!live || simulationAllowed.load(ordering: .relaxed)), let ram = simulatedRAM(environment) {
            return ceiling(physical: ram)
        }
        return ceiling(physical: Int(ProcessInfo.processInfo.physicalMemory))
    }

    /// Whether `SILICONED_MEMORY_BUDGET_GB` replaces this Mac's B.
    package static var isForced: Bool { forcedCeiling(liveEnvironment([forcingVariable])) != nil }

    /// **The simulated RAM** (`SILICONED_SIMULATED_RAM=32G`): the RAM B is computed from, in place of
    /// this Mac's — to see which plan, or which refusal, a 24, 32, 64 or 128 GB Mac gets for a format, on
    /// a Mac that has 16. **It changes B and nothing else**: the render still runs on the real
    /// hardware, so a simulated RAM above the real one can plan a render this Mac cannot hold. A
    /// measurement process asks for it (`allowSimulation`); the app and the library do not
    /// read it, whatever the environment says — a user's Mac renders in its own B.
    package static let simulatedRAMVariable = "SILICONED_SIMULATED_RAM"

    private static let simulationAllowed = Atomic<Bool>(false)

    /// The measurement processes' switch: from now on the live environment's simulated RAM counts.
    package static func allowSimulation() { simulationAllowed.store(true, ordering: .relaxed) }

    /// Whether the live environment simulates a RAM (and this process reads it).
    package static var isSimulated: Bool {
        simulationAllowed.load(ordering: .relaxed) && !isForced
            && simulatedRAM(liveEnvironment([simulatedRAMVariable])) != nil
    }

    /// **A RAM as typed**, in bytes: `32G`, `32GB`, `32GiB`, `32` (GiB all: a "32 GB" Mac holds 32 GiB),
    /// from 1 to 1024. Anything else is refused with a warning and B stays this Mac's.
    package static func parseRAM(_ text: String) -> Int? {
        var t = text.trimmingCharacters(in: .whitespaces).lowercased().replacingOccurrences(of: ",", with: ".")
        for suffix in ["gib", "gb", "g"] where t.hasSuffix(suffix) { t = String(t.dropLast(suffix.count)); break }
        guard let value = Double(t.trimmingCharacters(in: .whitespaces)), value.isFinite, value >= 1, value <= maximumForcedGiB
        else { return nil }
        return Int(value * 1_073_741_824)
    }

    /// The simulated RAM of `environment`, in bytes, if it sets one that reads.
    package static func simulatedRAM(_ environment: [String: String]) -> Int? {
        guard let text = environment[simulatedRAMVariable] else { return nil }
        guard let bytes = parseRAM(text) else {
            Warnings.emit("\(simulatedRAMVariable)=\(text) ignored: a RAM is a number of GiB from 1 to \(Int(maximumForcedGiB)), like 32G")
            return nil
        }
        return bytes
    }

    private static let gib = 1 << 30

    /// **The forcing** (`SILICONED_MEMORY_BUDGET_GB`, GiB): B itself — "4" simulates an 8 GB Mac, "22"
    /// a 32 GB one, refused and planned as one. A tool of measurement and of the tests, read in the
    /// engine and in the app alike.
    package static let forcingVariable = "SILICONED_MEMORY_BUDGET_GB"

    /// A forcing that is not a finite number of GiB from 0 to `maximumForcedGiB` (`inf`, `nan`, `-1`,
    /// `1e30`, a word) is **refused with a warning** and B stays this Mac's: `Int(Double)` traps on
    /// anything it cannot hold.
    package static func forcedCeiling(_ environment: [String: String]) -> Int? {
        guard let text = environment[forcingVariable] else { return nil }
        guard let gib = Double(text.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: ",", with: ".")),
              gib.isFinite, gib >= 0, gib <= maximumForcedGiB else {
            Warnings.emit("\(forcingVariable)=\(text) ignored: B is a number of GiB from 0 to \(Int(maximumForcedGiB))")
            return nil
        }
        return Int(gib * 1_073_741_824)
    }

    /// The largest forcing taken: 1 TiB, beyond any Mac sold (`soldSizes`).
    package static let maximumForcedGiB = 1024.0

    /// The process's environment, read live (`getenv`: a test may set it), for these variables only.
    package static func liveEnvironment(_ names: [String]) -> [String: String] {
        var environment: [String: String] = [:]
        for name in names { if let value = getenv(name) { environment[name] = String(cString: value) } }
        return environment
    }

    /// B in one line, for a report line.
    package static func summary(_ ceiling: Int) -> String {
        let how = isForced ? " (forced: \(forcingVariable))" : isSimulated ? " (simulated: \(simulatedRAMVariable))" : ""
        return String(format: "B %.2f GB%@", Double(ceiling) / 1e9, how as NSString)
    }
}
