import Foundation

// **The `async` facade: what a SwiftUI app calls.**
//
// `render` blocks the thread that calls it — over four minutes at 1024² for Krea 2. Called
// from a `Task`, it would tie up a thread of the cooperative pool, which has only as many as
// cores. The facade therefore returns control immediately and runs the render **on the engine
// queue** (`Engine.file`, static, one render at a time for the whole process): never on the pool.
//
// Cancelling the `Task` raises the token (`withTaskCancellationHandler`); the render stops at the
// next layer boundary and the facade throws `EngineError.cancelled`. Nothing is copied: the blocking
// form goes through `renderBatch`, the `async` form through `renderBatchOnQueue`, and both call
// `execute`.
//
// **Same parameters, in the same order, everywhere**: `(request, [seeds:], model:,
// [onProgress:])` for `render`, `renderBatch` and `events`, blocking or not. A custom chain
// enters as a `Model(card:chain:)`.
//
// **Which to prefer**: a SwiftUI app takes `events` (events arrive where one iterates, in
// order); `render(…, onProgress:)` is for whoever needs only the result, or a closure called on
// the render thread. Cancelling the `Task` makes `render` and `renderBatch` throw
// `EngineError.cancelled`; the stream, for its part, ends without throwing (that is
// `AsyncThrowingStream`).

extension Engine {
    /// **Renders an image without blocking** — `onProgress` is called on the render thread (see
    /// `Event`); a UI does its `MainActor` hop there, or prefers `events(_:seeds:model:)`.
    public func render(_ request: Request, model: Model,
                       onProgress: (@Sendable (Event) -> Void)? = nil) async throws(EngineError) -> Render {
        try await renderBatch(request, seeds: [request.seed], model: model, onProgress: onProgress)[0]
    }

    /// A batch, without blocking.
    public func renderBatch(_ request: Request, seeds: [UInt64], model: Model,
                          onProgress: (@Sendable (Event) -> Void)? = nil) async throws(EngineError) -> [Render] {
        let cancellation = Cancellation()
        do {
            return try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[Render], Error>) in
                    Engine.file.async {
                        continuation.resume(with: Result {
                            try self.renderBatchOnQueue(request, seeds: seeds, model: model,
                                                        cancellation: cancellation, onProgress: onProgress)
                        })
                    }
                }
            } onCancel: {
                cancellation.cancel()
            }
        } catch {
            throw EngineError(error)
        }
    }

    /// **The render as a stream of events** — `for try await event in engine.events(…)`.
    ///
    /// The result arrives in the stream: each image is an `.image(index:, Render)`, the last
    /// element of a successful render; an error ends the stream by throwing — always an `EngineError`.
    /// Stopping iteration — or cancelling the `Task` that iterates — cancels the render, and the
    /// stream then ends without throwing (`AsyncThrowingStream`'s rule). The
    /// buffer is **unbounded** so as to lose neither a step nor an image: events are rare (a few
    /// dozen); only the previews weigh (64 KB per preview at 1024²).
    public func events(_ request: Request, seeds: [UInt64]? = nil,
                           model: Model) -> AsyncThrowingStream<Event, Error> {
        let cancellation = Cancellation()
        let (flux, continuation) = AsyncThrowingStream<Event, Error>.makeStream()
        continuation.onTermination = { _ in cancellation.cancel() }
        Engine.file.async {
            do {
                _ = try self.renderBatchOnQueue(request, seeds: seeds ?? [request.seed], model: model,
                                                cancellation: cancellation) { continuation.yield($0) }
                continuation.finish()
            } catch {
                continuation.finish(throwing: EngineError(error))
            }
        }
        return flux
    }
}
