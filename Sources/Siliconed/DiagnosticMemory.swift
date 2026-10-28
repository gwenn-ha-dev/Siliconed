// What the diagnostic holds, step by step: the gauge that reads its peak (`phys_footprint`, the kernel's
// interval maximum) and, in `MemoryNeedDiagnostic.swift`, the need the shapes give. Split from `Diagnostic.swift`.

import Foundation

extension Diagnostic {
    /// **The peak of each step of the map**, read from the process itself (`FootprintInterval`: the
    /// kernel's `ri_interval_max_phys_footprint`, reset at each mark — invariant 5, not an allocator's
    /// count). A step is `group` (a stage or a part of one: `products`, `cuts`, `aneCuts`, `lowPrecision`,
    /// `attention`, `widening`, `witness`) and `detail` (the shape): the report keeps the group's maximum,
    /// `SILICONED_DIAGNOSTIC_MEMORY=1` prints every mark on standard error.
    final class Gauge {
        struct Mark { let group, detail: String; var peak: Int, after: Int }
        private(set) var marks: [Mark] = []
        private var open: (group: String, detail: String)?
        private let verbose = ProcessInfo.processInfo.environment["SILICONED_DIAGNOSTIC_MEMORY"] == "1"
        /// What the process held when the map started: the runtime, an app's interface — B covers it too.
        let base: Int

        init() { base = Arena.processFootprint(); FootprintInterval.reset() }

        /// Closes the step open, if any, and opens the next.
        func mark(_ group: String, _ detail: String = "") {
            close()
            open = (group, detail)
        }

        /// Ends the step open. The next one's interval starts here.
        func close() {
            guard let step = open else { return }
            let mark = Mark(group: step.group, detail: step.detail, peak: FootprintInterval.peak(), after: Arena.processFootprint())
            marks.append(mark)
            if verbose {
                FileHandle.standardError.write(String(format: "memory  %@ %@  peak %.0f MB  after %.0f MB\n", step.group,
                                                      step.detail, Double(mark.peak) / 1e6, Double(mark.after) / 1e6)
                    .data(using: .utf8)!)
            }
            open = nil
            FootprintInterval.reset()
        }

        /// The maximum of each group, in bytes.
        var peaks: [String: Int] {
            Dictionary(marks.map { ($0.group, $0.peak) }, uniquingKeysWith: max)
        }

        func finish() { close(); FootprintInterval.stop() }
    }
}
