import Foundation
import Metal

// **The DiT's map made resident while the text is encoded**, where the plan reads it in place.
//
//     render:  text encoder ──────────────┐ denoising: step 1 … step n
//     warm-up:  open map · wrap · resident ┘ (the DiT takes this same Artifact)
//
// A map read in place (`Artifact.wrappedMap`) pays, at the first command buffer that names it, the kernel's wiring and GPU mapping
// of every page of the file (`Artifact.makeResident`: ~1 s for 14 GB from a warm cache, measured) — on the thread that commits,
// so inside the first step, with the GPU idle. Paid here on its own thread, beside the text stage, it is out of the denoising.
// The residency is per buffer, not per file: a second mapping of the same file pays again (measured: 1.13 s after another one's),
// so the warmed `Artifact` itself is handed to the DiT (`open`). Nothing changes in the computation: the same weights, the same
// products, the same bits.
//
// Taken only where the plan says so (`MemoryPlan.Render.warmsMap`): the wired map beside every phase before the denoising holds in B.
// One warm-up at a time (renders are serialized); what no DiT took is dropped at the render's end (`end`).

package enum MapWarmup {
    private final class Slot: @unchecked Sendable {
        let path: String
        let done = DispatchGroup()
        var artifact: Artifact?
        var seconds = 0.0
        init(path: String) { self.path = path }
    }
    private static let lock = NSLock()
    nonisolated(unsafe) private static var slot: Slot?

    /// Opens `path` and makes its wrapped map resident on the system's GPU (the device the products wrap it for: `SystemGPU`), on a
    /// thread of its own; returns at once.
    package static func begin(path: String) {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let warm = Slot(path: path)
        lock.withLock { slot = warm }
        warm.done.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            defer { warm.done.leave() }
            let began = DispatchTime.now().uptimeNanoseconds
            guard let artifact = try? Artifact(path: path), artifact.makeResident(on: device) else { return }
            warm.artifact = artifact
            warm.seconds = Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9
        }
    }

    /// **The DiT's map**: the warmed `Artifact` of `path` if one is under way (waited for), once — otherwise a new one.
    package static func open(_ path: String) throws -> Artifact {
        let warm = lock.withLock { () -> Slot? in
            guard let s = slot, s.path == path else { return nil }
            slot = nil
            return s
        }
        guard let warm else { return try Artifact(path: path) }
        let waited = DispatchTime.now().uptimeNanoseconds
        warm.done.wait()
        guard let artifact = warm.artifact else { return try Artifact(path: path) }
        MemoryPlan.note(String(format: "map made resident beside the text in %.2f s (%.2f s waited for at the denoising)",
                               warm.seconds, Double(DispatchTime.now().uptimeNanoseconds - waited) * 1e-9))
        return artifact
    }

    /// The render's end: a warm-up no DiT took (a failure, a cancellation before the denoising) goes with its pages' residency.
    package static func end() {
        let left = lock.withLock { () -> Slot? in let s = slot; slot = nil; return s }
        left?.done.wait()
    }
}
