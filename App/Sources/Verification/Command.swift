import Foundation
import Siliconed

/// **`siliconed`'s verbs.** The executable is this dispatch and nothing else, so that the maintainer's
/// tool can carry the same verbs beside its own.
package enum Command {
    /// **`siliconed` itself**: every verb in this process. What computes is under B (`MemoryBudget`): past it, the
    /// process stops, and so does the command — no net runs it again (P is written before the start: a stop past B is
    /// a fault to find, not a cut to retry).
    package static func main(_ arguments: [String]) -> Int32 {
        _ = MemoryBudget.machineCeiling()
        CoreAI.place(.processWide)
        // The verbs that compute pay Core AI's first call before any stage reads its floor.
        if ["check", "write"].contains(arguments.first) { CoreAI.witness() }
        return run(arguments)
    }

    /// The command in this process: what `main` runs, and the tests.
    package static func run(_ arguments: [String], stages: [Stage] = Registry.stages) -> Int32 {
        switch arguments.first {
        case "check": return CheckCommand.run(Array(arguments.dropFirst()), stages: stages)
        case "write": return WriteCommand.run(Array(arguments.dropFirst()))
        case let verb? where LibraryCommand.verbs.contains(verb): return LibraryCommand.run(verb, Array(arguments.dropFirst()))
        case nil, "help", "--help", "-h": print(help); return 0
        case let verb?:
            FileHandle.standardError.write("✗ unknown command '\(verb)', see `siliconed help`\n".data(using: .utf8)!)
            return 64
        }
    }

    static var help: String {
        """
        siliconed — image generation on Apple silicon, checked against the reference implementations

        \(LibraryCommand.usage)

        \(CheckCommand.usage)

        \(WriteCommand.usage)

        Exit codes: 0 passed · 1 failed · 2 skipped (an input is missing, said) · 64 usage.
        """
    }
}
