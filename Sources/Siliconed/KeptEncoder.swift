import Foundation
import Synchronization

/// **The text encoder kept from one render to the next** — the plan's choice (`Request.encoderKept`, `MemoryPlan.choose`), held by
/// the process, one encoder at a time.
///
///     render n   plan keeps it? ─ no ─▶ what is held goes back before the first computation (`settle`); the encoder streams as before
///                     │ yes
///                text phase: held, same file? ─ yes ─▶ the layers already in the process's pages: nothing read
///                     │ no
///                the map opened, its layers read ahead into one anonymous region (`Artifact.keepResidentAhead`) while
///                the encoder computes behind the reads; held from then on
///     end        a read that did not finish (a cancellation, a failure): nothing held (`renderEnded`)
///
/// What a reading by analogy would get wrong:
///
///   - **The same bits.** The kept layers are the file's bytes, read with the stream's own `pread`; every reader reaches them through
///     `Artifact.pointer`, which serves the region in place of the stream's buffer. The products, their routes and their order
///     do not change: a prompt encoded by a kept encoder is the prompt encoded by a streamed one.
///   - **Charged where it lives, which is everywhere.** Read in the text phase, the layers stay beside the denoising, the decoding
///     and the time between renders: the need charges them to every phase (`MemoryNeed.keptEncoderTerm`), and the plan keeps them
///     only where that holds in B and the DiT's map loses nothing to them.
///   - **Never twice in the base.** The process's footprint, read as the base of the next render (`OperatorFootprint.processBase`),
///     holds the kept layers: they are taken out of it (`heldBytes`), since the need charges them itself — or gives them back first.
///     Only a finished read is held between renders (`renderEnded`), so what is taken out is exactly what the footprint holds.
///   - **Only the layers.** Not the embedding table (a prompt reads a few of its rows, through the map, as before), nor Qwen3-VL's
///     vision tower (an edit's, through the map): `prefixes`.
///   - **Not the writer's, nor a diagnostic's.** The Prompt Lab and the capacity probes plan their memory on the process's base
///     too: they give the kept encoder back before reading it (`release`), as a render gives the writer's weights back.
package enum KeptEncoder {
    /// The tensors kept: an encoder's layers — what its stream reads at every new prompt.
    package static let prefixes = ["layers."]

    private final class Held: @unchecked Sendable {
        let path: String
        let stamp: MemoryNeed.InstalledFiles
        let artifact: Artifact
        init(path: String, artifact: Artifact) {
            self.path = path; self.artifact = artifact
            stamp = MemoryNeed.InstalledFiles(paths: [path])
        }
    }
    private static let held = Mutex<Held?>(nil)

    /// **The bytes the kept encoder holds** — taken out of the process's base (`OperatorFootprint.processBase`): the plan charges them.
    package static var heldBytes: Int { held.withLock { $0?.artifact.residentCapacity ?? 0 } }

    /// The map held, if any (the report's, the tests').
    package static var heldPath: String? { held.withLock { $0?.path } }

    /// **The encoder map `request`'s text module opens**: the library's installed encoder of its family.
    package static func path(for request: MemoryNeed.Request) -> String {
        request.library.map(request.library.encoderMap(request.card.family))
    }

    /// **Before a render's first computation, once its plan is chosen**: what the plan does not keep goes back — another family's
    /// encoder, a map replaced since it was read, or all of it on a plan that keeps none.
    package static func settle(for render: MemoryPlan.Render) {
        let keep = render.request.encoderKept ? path(for: render.request) : nil
        let dropped = held.withLock { h -> Held? in
            guard let current = h, current.path != keep || current.stamp != MemoryNeed.InstalledFiles(paths: [current.path]) else {
                return nil
            }
            h = nil
            return current
        }
        if let dropped {
            MemoryPlan.note(String(format: "text encoder kept before given back (%.2f GB)", Double(dropped.artifact.residentCapacity) / 1e9))
        }
    }

    /// **A text module's encoder map**: on a plan that keeps it, the one held (read before), or the map opened and its layers read
    /// ahead into the process's pages, then held; otherwise the map, opened as before.
    package static func artifact(_ path: String) throws -> Artifact {
        guard MemoryPlan.current?.request.encoderKept == true else { return try Artifact(path: path) }
        if let current = held.withLock({ $0 }), current.path == path {
            MemoryPlan.note(String(format: "text encoder kept: %.2f GB already read", Double(current.artifact.residentCapacity) / 1e9))
            return current.artifact
        }
        release()
        let artifact = try Artifact(path: path)
        try artifact.keepResidentAhead(prefixes: prefixes)
        held.withLock { $0 = Held(path: path, artifact: artifact) }
        MemoryPlan.note(String(format: "text encoder kept: %.2f GB read once, ahead of the layers", Double(artifact.residentCapacity) / 1e9))
        return artifact
    }

    /// **A render's end**: a read that did not finish — cancelled in the text phase, or failed — is not held between renders.
    /// After a render that went through, the read is complete: every layer was waited for.
    package static func renderEnded() {
        held.withLock { h in
            guard let current = h, current.artifact.residentLoad != .complete else { return }
            h = nil
        }
    }

    /// Gives the kept encoder back (its pages unmapped at once, its reads stopped first).
    package static func release() {
        let dropped = held.withLock { h -> Held? in let current = h; h = nil; return current }
        withExtendedLifetime(dropped) {}
    }
}
