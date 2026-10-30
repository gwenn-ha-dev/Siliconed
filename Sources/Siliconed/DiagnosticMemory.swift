// What the diagnostic reads of its own memory: the process's footprint when it starts (the base B covers) and the kernel's interval
// maximum of `phys_footprint` over the run (invariant 5, not an allocator's count). The need the shapes give is in
// `MemoryNeedDiagnostic.swift`. Split from `Diagnostic.swift`.

import Foundation

extension Diagnostic {
    /// **The peak of the probes**, read from the process itself (`FootprintInterval`: the kernel's `ri_interval_max_phys_footprint`,
    /// reset when the gauge is made).
    final class Gauge {
        /// What the process held when the probes started: the runtime, an app's interface — B covers it too.
        let base: Int

        init() { base = Arena.processFootprint(); FootprintInterval.reset() }

        func finish() { FootprintInterval.stop() }
    }
}
