import Foundation
import Synchronization

/// **A render's cancellation token: an atomic boolean, raised from wherever you like.**
///
/// A Krea 2 step takes half a minute at 1024²: cancelling only between steps would leave
/// the app frozen for that long. The token is therefore consulted at three levels, from coarsest
/// to finest:
///
///     between stages        `Engine.renderBatch` — text, image, denoising, decoding, and between two images of a batch
///     between steps         `Sampler.run` (Z-Image), `QwenImage21Sampler.run`, `euler` (the other families)
///     between layers        the DiTs (`DiT`, `QwenImage21DiT`, …), `TextEncoder`, `Qwen3VLEncoder`, `Qwen3VLVision`
///
/// **What cannot be cancelled midway**: the VAEs (img2img encoding and decoding) and Anima's
/// conditioner. Cancellation there waits for the end of the stage — a few seconds for a Krea 2
/// decode at 1024×1536. It is a choice: these stages are short next to denoising.
///
/// **Never lower.** A check inside a co-executed GEMM (`Conductor`) would leave the AMX thread in
/// the middle of a slice and a Metal submission in flight; between two layers, on the contrary,
/// `GEMM` has waited for its submission (`waitUntilCompleted`) and the conductor is at rest.
/// Throwing there is safe: the arenas live in the DiT, the block and the encoder, and are
/// unmapped on leaving scope (`Arena.deinit`) — `Arena.live` returns to zero, which the
/// cancellation check verifies, model by model.
///
/// The error thrown is `EngineError.cancelled` — the one public error; a cancelled `Task` that throws
/// Swift's `CancellationError` folds into the same case at the door (`EngineError(_:)`).
public final class Cancellation: Sendable {
    private let flag = Atomic<Bool>(false)

    public init() {}

    /// Raises the flag. Idempotent, lock-free, callable from any thread (a signal handler
    /// included: a relaxed atomic store).
    public func cancel() { flag.store(true, ordering: .relaxed) }

    public var isCancelled: Bool { flag.load(ordering: .relaxed) }

    /// Throws `EngineError.cancelled` if the flag is raised.
    public func check() throws(EngineError) {
        if isCancelled { throw .cancelled }
    }
}

extension Optional where Wrapped == Cancellation {
    /// `try cancellation.check()` on an optional token: nothing without a token.
    func check() throws(EngineError) { try self?.check() }
}

/// **The library's warnings — it writes neither to stdout nor to stderr.**
///
/// What used to go to standard error (a flash-kernel fallback, a non-finite speed, a refused VAE
/// tile, an ignored profile, a GPU failure) is often born deep in a loop that knows nothing of
/// the current render. It therefore goes through this single channel:
///
/// - **during a render**, the engine wires its `Context` in: the warning becomes a `.warning`
///   event, which the app displays and the CLI writes to stderr;
/// - **outside a render** (a check driving a DiT directly), it goes to the receiver the caller
///   set (`outsideRender`) — the CLI puts standard error there; without a receiver, it is lost, and
///   that is the caller's choice.
///
/// One render at a time (the `Engine` queue): a single current receiver suffices, no stack.
public enum Warnings {
    private static let current = Mutex<(@Sendable (String) -> Void)?>(nil)
    private static let outsideRenderState = Mutex<(@Sendable (String) -> Void)?>(nil)

    /// The receiver of warnings emitted outside a render.
    public static var outsideRender: (@Sendable (String) -> Void)? {
        get { outsideRenderState.withLock { $0 } }
        set { outsideRenderState.withLock { $0 = newValue } }
    }

    /// Emits a warning (one line, no trailing newline).
    static func emit(_ message: String) {
        let receiver = current.withLock { $0 } ?? outsideRender
        receiver?(message)
    }

    /// Wires `receiver` in for the duration of `body` — the engine, around a render.
    static func during<T>(_ receiver: @escaping @Sendable (String) -> Void, _ body: () throws -> T) rethrows -> T {
        let previous = current.withLock { state -> (@Sendable (String) -> Void)? in
            let p = state; state = receiver; return p
        }
        defer { current.withLock { $0 = previous } }
        return try body()
    }
}
