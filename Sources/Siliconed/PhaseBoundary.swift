import Foundation
import Metal
import os

// **The render's memory ledger**: the phase boundaries (`PhaseBoundary`) and the `phys_footprint`
// maximum each phase reads (`FootprintInterval`). The render that marks them is in `EngineRender.swift`.

// ── the engine ───────────────────────────────────────────────────────────────────────────

/// **The render's phase boundaries** — called where a phase has handed back what it owned.
///
/// Its reader is **the product's own check** (`MemoryCheck`): between `begin` and `end`
/// each mark records the phase's `phys_footprint` maximum — the kernel's interval maximum, or the
/// 10 ms sampler without it (`FootprintInterval`), reset at every mark — so a render knows which
/// phase peaked, and by how much, against its computed need.
///
/// **`nonisolated(unsafe)`, and the one invariant that makes it safe**: at most one render (or one
/// diagnostic) runs in the process at a time — renders are serialized on the engine's queue
/// (`Engine.file`) and, across processes, by `RenderLock.machine`; a check runs alone by the machine
/// token. Every access below happens on that one render's thread (`begin`, `mark`, `end`). A second
/// render running concurrently in one process would interleave two ledgers and two kernel
/// intervals: the serialization is what this relies on, not a lock.
package enum PhaseBoundary {
    /// The marks of the render in progress, `nil` outside one — one ledger, by the invariant above.
    nonisolated(unsafe) private static var marks: [(name: String, peak: Int)]?

    /// The graphics ledger when the render began: what `returned` waits for Metal to come back to.
    nonisolated(unsafe) private static var graphicsAtBegin = 0

    /// A render starts: an empty ledger, a new kernel interval.
    package static func begin() {
        marks = []
        graphicsAtBegin = VAEMemory.graphicsFootprint()
        FootprintInterval.reset()
    }

    /// **Waits until Metal has taken back what a phase released** — the graphics ledger back to its
    /// level at the render's start (within `VAEMemory.releaseTolerance`), for `VAEMemory.releaseWait`
    /// at most. Metal returns a released buffer late (`VAEMemory.finish`, which does the same for a
    /// VAE's heaps): Qwen-Image-2.1's DiT, released, still had 10 MB in the ledger when the decoder
    /// started, beside it. Called where a phase hands back everything GPU-side and nothing
    /// GPU-side crosses: after the text. After the denoiser: `returnedDenoiser()`.
    package static func returned() { wait(above: graphicsAtBegin) }

    private static let device = MTLCreateSystemDefaultDevice()

    /// **The denoiser released, waits until the graphics ledger is down to what Metal still has allocated** — the system device's
    /// `currentAllocatedSize` less our wraps over our own memory (`MetalWraps`: their pages are anonymous memory or a file's, never
    /// the ledger's), within `VAEMemory.releaseTolerance`, for `VAEMemory.releaseWait` at most. Read again at every poll.
    ///
    /// **What is still allocated, not the start's level**: the ledger also carries the Metal driver's command-buffer memory
    /// (~40–60 MB), which no allocation holds. The driver keeps it while a command queue that submitted lives, trims it ~1.2 s after
    /// the process's last command buffer, and gives it back at once when the last such queue dies — the denoiser's, now that no
    /// cache keeps a queue (`QKNormRope.shared`, `OperatorFootprint.bringUp`). The start's level held the bring-up's share, gone by
    /// then: waited back to it, the boundary waited for the trim (0.42–0.52 s on Z-Image 1024², 1.1 s on Anima 512²) while
    /// the driver's share was still there, and stopped short of the late returns once it had left (Krea 2 512², +30 MB beside the
    /// decoder). What the driver returns late — the private buffers of the tiled attention, the head split's MPSGraph temporaries
    /// on Krea 2, up to ~0.35 s after their release — is out of `currentAllocatedSize` and still in the ledger: waited for.
    /// Measured on the M5 Pro after each family's denoiser: the ledger meets the allocations within 0.03–0.17 s.
    package static func returnedDenoiser() {
        guard let device else { return returned() }
        let deadline = Date().addingTimeInterval(VAEMemory.releaseWait)
        while VAEMemory.graphicsFootprint() > device.currentAllocatedSize - MetalWraps.live + VAEMemory.releaseTolerance,
              Date() < deadline { usleep(1000) }
    }

    private static func wait(above floor: Int) {
        let deadline = Date().addingTimeInterval(VAEMemory.releaseWait)
        while VAEMemory.graphicsFootprint() > floor + VAEMemory.releaseTolerance, Date() < deadline { usleep(1000) }
    }

    package static func mark(_ name: String) {
        let peak = FootprintInterval.peak()
        // The phase's freed pages go back to the kernel before the next phase starts beside them
        // (`Arena.returnFreedPages`): what survives a boundary is what is alive, not what `free` queued.
        Arena.returnFreedPages()
        guard marks != nil else { return }
        marks?.append((name, peak))
        FootprintInterval.reset()
    }

    /// The render's marks, in order; the ledger closes.
    package static func end() -> [(name: String, peak: Int)] {
        defer { marks = nil; FootprintInterval.stop() }
        return marks ?? []
    }
}

/// **This process's `phys_footprint` maximum since the last reset**, by one of two methods — said by
/// `method`, carried by every `MemoryCheck` and by the diagnostic (`peakMethod`):
///
/// - **`kernelInterval`**: `ri_interval_max_phys_footprint` (`proc_pid_rusage`), reset by
///   `proc_reset_footprint_interval`. `phys_footprint`'s exact maximum (invariant 5), not an
///   allocator's count, for one system call: an SDPA of a few milliseconds falls between two 10 ms
///   samples, never between two kernel updates. The reset is not in a public header
///   (`libproc`'s private half): looked up by name. No thread.
/// - **`sampled10ms`**: the symbol absent, or the reset or the read failing once (then for the rest of
///   the process), a thread reads `phys_footprint` (`task_vm_info`) every 10 ms and keeps the maximum
///   since the last `reset`. A lower bound — a spike shorter than 10 ms can fall between two samples —
///   but a phase's, not the footprint at its boundary. The thread lives from the first `reset` to
///   `stop` (a render's `PhaseBoundary.begin` and `end`): never resident at rest.
///
/// Its `nonisolated(unsafe)` statics (`kernelFailed`, `sampler`, the test injection) are safe under
/// `PhaseBoundary`'s invariant — one render at a time in the process, its calls on its own thread;
/// the sampler's thread touches only its own locked state.
package enum FootprintInterval {
    private typealias Reset = @convention(c) (pid_t) -> Int32
    private static let resetFunction: Reset? = {
        guard let symbol = dlsym(dlopen(nil, RTLD_NOW), "proc_reset_footprint_interval") else { return nil }
        return unsafeBitCast(symbol, to: Reset.self)
    }()
    /// **Test injection only**: the symbol treated as absent, so the sampler runs on a Mac that has
    /// it. Not a setting of the product.
    nonisolated(unsafe) package static var symbolMissingForTesting = false
    /// The kernel's reset or read failed once: sampled from then on.
    nonisolated(unsafe) private static var kernelFailed = false
    nonisolated(unsafe) private static var sampler: Sampler?

    package static var method: PeakMethod {
        resetFunction != nil && !symbolMissingForTesting && !kernelFailed ? .kernelInterval : .sampled10ms
    }

    /// A new interval: the kernel's, or the sampler's — started here if it is not running.
    package static func reset() {
        if method == .kernelInterval, let resetFunction {
            if resetFunction(getpid()) == 0 { return }
            kernelFailed = true
        }
        if let sampler { sampler.reset() } else { sampler = Sampler() }
    }

    /// The sampler, if one runs, stops — the end of a render or of a measuring check. Nothing for the kernel.
    package static func stop() {
        sampler?.stop()
        sampler = nil
    }

    /// The interval's maximum by the method in force; `0` when nothing measured it (no sampler since
    /// the last `stop`, or the kernel's read failing — sampled from the next `reset`).
    package static var maximum: Int {
        if method == .kernelInterval {
            var r = rusage_info_v6()
            let status = withUnsafeMutablePointer(to: &r) {
                $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V6, $0) }
            }
            if status == 0 { return Int(r.ri_interval_max_phys_footprint) }
            kernelFailed = true
            return 0
        }
        return sampler?.maximum ?? 0
    }

    /// The interval's maximum, never below the footprint now.
    package static func peak() -> Int { max(maximum, Arena.processFootprint()) }

    /// `phys_footprint` every 10 ms on a thread of its own; the maximum since the last `reset`.
    private final class Sampler: @unchecked Sendable {
        private let lock = NSLock()
        private var peak: Int, running = true
        init() {
            peak = Arena.processFootprint()
            Thread.detachNewThread { [self] in
                while true {
                    let now = Arena.processFootprint()
                    lock.lock()
                    peak = max(peak, now)
                    let go = running
                    lock.unlock()
                    if !go { return }
                    usleep(10_000)
                }
            }
        }
        func reset() {
            let now = Arena.processFootprint()
            lock.lock(); peak = now; lock.unlock()
        }
        var maximum: Int { lock.lock(); defer { lock.unlock() }; return peak }
        func stop() { lock.lock(); running = false; lock.unlock() }
    }
}
