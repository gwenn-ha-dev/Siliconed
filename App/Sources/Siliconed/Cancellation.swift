import Foundation
import Synchronization

/// **A render's cancellation token: an atomic boolean, raised from wherever you like.**
///
/// It is consulted between two runs of the engine, never inside one — a program is Core AI's whole (`Programs.swift`):
///
///     between stages        each family's render: text, image, denoising, decoding, and between two images of a batch
///     between steps         each family's sampler loop, before every evaluation of its DiT
///
/// **What cannot be cancelled midway**: a run under way — an encoder's, a step's, a decoding's. Cancellation waits for
/// its end; the sessions are let go as the render's scope unwinds.
///
/// The error thrown is `EngineError.cancelled` — the one error that leaves the engine; a cancelled `Task` that throws
/// Swift's `CancellationError` folds into the same case at the door (`EngineError(_:)`).
package final class Cancellation: Sendable {
    private let flag = Atomic<Bool>(false)
    /// A token this one also obeys: raised, it raises this one — never the reverse.
    private let parent: Cancellation?

    package init() { parent = nil }

    /// A token raised by its own `cancel()` **or** by `parent`'s: what stops one part of a job
    /// (one download of an installation running beside another) without stopping the whole.
    package init(following parent: Cancellation?) { self.parent = parent }

    /// Raises the flag. Idempotent, lock-free, callable from any thread (a signal handler
    /// included: a relaxed atomic store).
    package func cancel() { flag.store(true, ordering: .relaxed) }

    package var isCancelled: Bool { flag.load(ordering: .relaxed) || parent?.isCancelled == true }

    /// Throws `EngineError.cancelled` if the flag is raised.
    package func check() throws(EngineError) {
        if isCancelled { throw .cancelled }
    }
}

extension Optional where Wrapped == Cancellation {
    /// `try cancellation.check()` on an optional token: nothing without a token.
    func check() throws(EngineError) { try self?.check() }
}
