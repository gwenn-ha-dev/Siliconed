import Darwin
import Synchronization
import Foundation
import Measurement

/// **The app's engine, in a process of its own** (`silicore-worker`, `WorkerServer`): the app asks for a render or a
/// writer's gesture and reads its events; the engine's memory, its weights and Core AI live there, never in the app.
///
/// - **B bounds the worker alone.** Whatever the worker runs, its fuse (`MemoryBudget`, armed at its start) stops it the
///   moment its footprint passes B: one reading a millisecond, from inside, the same in `siliconed`. The app is never
///   the one that stops.
/// - **A worker keeps the B it was launched with** (`MemoryBudget.budget`, handed in its environment): a call made
///   under another, with no other in hand, ends it and starts one under the new; with others in hand, it waits.
/// - **A worker that stops**, by its fuse or a crash: every call in hand fails with `EngineError.engineStopped`, and the
///   next call starts a new worker. Nothing is run again: P is written before the start and the plan refuses what passes
///   B, so a stop past B is a fault, not a cut to retry. What follows is the user's to decide — **but for a decoding**:
///   a render the fuse stopped once its DiT was done (its latents kept, `RenderLoop.keeping`) is handed once to a new
///   worker, which decodes them alone in a form that holds (the fuse marked the one it stopped under). The DiT's minutes
///   are not lost to the decoder's fault: the user decided so on 21/11/2026. Before the DiT, what a stop loses is the
///   text's seconds, and the call fails.
/// - **One piece of work at a time**: the worker runs its calls in their order. A cancelled call is still the worker's
///   until it says it is over.
/// - **One build**: each end says its protocol first (`WorkerHello`); a worker of another fails every call in hand with
///   `EngineError.engineMismatch`, and is killed.
/// - **A worker that goes silent** with a call in hand — no reply, no event, no byte for `silence` seconds — is killed:
///   its calls fail with `EngineError.engineSilent`, and the next call starts another.
package final class EngineWorker: @unchecked Sendable {
    package static let shared = EngineWorker(executable: EngineWorker.executable)

    /// **How long a worker may say nothing with a call in hand**: ten minutes. Silence, not a render's whole time: a
    /// render of minutes speaks at each stage and each step. The longest gap a call that works leaves is one step of
    /// the largest render on the smallest Mac: Z Compacte at 1024² takes 17.1 s a step on the M1 Pro's GPU alone
    /// (13/11); 2048² has four times the tokens and a quadratic attention, and an M1's GPU half the M1 Pro's cores —
    /// about five minutes. Twice that. Core AI's compilations are seconds to tens of seconds a function (the ANE's: 10
    /// to 14 s a shape), before a stage's first event.
    package static let silence: TimeInterval = 600

    let program: URL?, arguments: [String]
    let silence: TimeInterval
    /// Where each render call's latents are kept between its DiT and its decoder (`RenderLoop.keeping`), `<id>.latents`.
    private let latents: URL
    let lock = NSLock()
    var live: Live?
    var calls: [UInt32: Pending] = [:]
    private var lastID: UInt32 = 0, lastBlob: UInt32 = 0
    /// The calls go out in order, off the caller's thread: an edit's images are megabytes.
    let sending = DispatchQueue(label: "siliconed.worker.send", qos: .userInitiated)

    /// `executable` run with `arguments`; `silence`: what a worker may stay silent, a call in hand; `latents`: the folder
    /// of the latents its renders keep, this process's own.
    package init(executable: URL?, arguments: [String] = [], silence: TimeInterval = EngineWorker.silence,
                 latents: URL = FileManager.default.temporaryDirectory.appendingPathComponent("silicore-latents-\(getpid())")) {
        program = executable; self.arguments = arguments; self.silence = silence; self.latents = latents
        // A worker gone, its pipe written to: the write fails (`EPIPE`) instead of killing the app.
        signal(SIGPIPE, SIG_IGN)
    }

    /// The worker's process, while one runs.
    package var pid: pid_t? { lock.withLock { live?.pid } }

    // MARK: - The calls

    /// **A render, as the engine's stream of events** (`Engine.Event`, as `Engine.renderBatch` emits them in the worker):
    /// the images in it, an error ends it by throwing (always an `EngineError`), and stopping the iteration cancels the
    /// render — the stream then ends without throwing (`AsyncThrowingStream`).
    package func render(_ request: Request, seeds: [UInt64], model: String, library: Library) -> AsyncThrowingStream<Engine.Event, Error> {
        let (stream, continuation) = AsyncThrowingStream<Engine.Event, Error>.makeStream()
        let neuralEngine = CoreAI.neuralEngine
        let id = register { reply, blobs in
            switch reply {
            case .event(_, let e):
                do { continuation.yield(try e.event(blobs)) } catch { continuation.finish(throwing: EngineError(error)) }
            case .finished: continuation.finish()
            case .failed(_, let f): continuation.finish(throwing: f.error)
            default: break
            }
        } stopped: { continuation.finish(throwing: $0) }
        continuation.onTermination = { [weak self] _ in self?.cancel(id) }
        sending.async { [self] in
            let first = lock.withLock { () -> UInt32 in
                let f = lastBlob &+ 1
                lastBlob &+= UInt32(1 + request.references.count)
                return f
            }
            try? FileManager.default.createDirectory(at: latents, withIntermediateDirectories: true)
            let kept = latents.appendingPathComponent("\(id).latents")
            let (order, blobs) = RenderOrder.of(request, seeds: seeds, model: model, library: library,
                                                neuralEngine: neuralEngine, keeping: kept, firstBlob: first)
            lock.withLock {
                calls[id]?.kept = kept
                calls[id]?.again = { [weak self] in self?.send(id, .render(id: id, order), blobs: blobs) }
            }
            send(id, .render(id: id, order), blobs: blobs)
        }
        return stream
    }

    /// **A writer's gesture** (`PromptWriter`), its events as the writer gives them. `image`: Describe's file, as dropped.
    package func write(rewrite text: String? = nil, describe image: Data? = nil, instruction: String,
                       library: Library) -> AsyncThrowingStream<PromptWriter.Event, Error> {
        let (stream, continuation) = AsyncThrowingStream<PromptWriter.Event, Error>.makeStream()
        let id = register { reply, _ in
            switch reply {
            case .written(_, let e): continuation.yield(e)
            case .finished: continuation.finish()
            case .failed(_, let f): continuation.finish(throwing: f.error)
            default: break
            }
        } stopped: { continuation.finish(throwing: $0) }
        continuation.onTermination = { [weak self] _ in self?.cancel(id) }
        let neuralEngine = CoreAI.neuralEngine
        sending.async { [self] in
            var blobs: [(UInt32, Data)] = []
            let input: WriteOrder.Input
            if let image {
                let b = lock.withLock { lastBlob &+= 1; return lastBlob }
                blobs.append((b, image))
                input = .describe(imageBlob: b)
            } else {
                input = .rewrite(text: text ?? "")
            }
            send(id, .write(id: id, WriteOrder(input: input, instruction: instruction, library: WorkerLibrary(library),
                                               neuralEngine: neuralEngine)), blobs: blobs)
        }
        return stream
    }

    /// **What a render of `request` would compile, compiled ahead** (`WorkerCall.prepare`): the worker queues its
    /// specializations — the form's in place of what its last left, first; `background` ones after — and runs them while
    /// no call is in hand: a render asked meanwhile waits for the one under way at most. Nothing to wait for here: a
    /// refusal is said on the standard error, and the render compiles what it needs as it loads it.
    package func prepare(_ request: Request, model: String, library: Library, background: Bool = false) {
        let id = register { reply, _ in
            if case .failed(_, let f) = reply {
                FileHandle.standardError.write(Data("siliconed: \(model) not prepared: \(f.error)\n".utf8))
            }
        } stopped: { _ in }
        let neuralEngine = CoreAI.neuralEngine
        sending.async { [self] in
            let first = lock.withLock { () -> UInt32 in
                let f = lastBlob &+ 1
                lastBlob &+= UInt32(1 + request.references.count)
                return f
            }
            let (order, blobs) = RenderOrder.of(request, seeds: [], model: model, library: library, neuralEngine: neuralEngine, firstBlob: first)
            send(id, .prepare(id: id, order, background: background), blobs: blobs)
        }
    }

    /// **Whether a preparation compiles now** (`WorkerReply.preparing`), told on the worker's reading thread: it must
    /// return at once. A worker that stops ends it.
    package var preparing: (@Sendable (Bool) -> Void)? {
        get { lock.withLock { _preparing } }
        set { lock.withLock { _preparing = newValue } }
    }
    private var _preparing: (@Sendable (Bool) -> Void)?

    /// **The compilations erased by the worker** (`CoreAI.erase`), between two pieces of work: sent now, in the order of
    /// the calls — after what was asked before, before what is asked after, which compiles again what it needs. `done`
    /// is handed what it freed, in bytes, or why it failed.
    package func erase(_ done: @escaping @Sendable (Result<Int, any Error>) -> Void) {
        let freed = Mutex(0)
        let id = register { reply, _ in
            switch reply {
            case .erased(_, let bytes): freed.withLock { $0 = bytes }
            case .finished: done(.success(freed.withLock { $0 }))
            case .failed(_, let f): done(.failure(f.error))
            default: break
            }
        } stopped: { done(.failure($0)) }
        sending.async { [self] in send(id, .erase(id: id), blobs: []) }
    }

    /// Ends the worker: its standard input closed, it exits. The app's last gesture.
    package func stop() {
        lock.withLock {
            live?.close()
            live = nil
        }
    }

    struct Pending {
        let reply: @Sendable (WorkerReply, [UInt32: Data]) -> Void
        /// The worker stopped, and why.
        let stopped: @Sendable (EngineError) -> Void
        /// Its consumer stopped listening: nothing more is handed, but the worker still computes it.
        var abandoned = false
        /// The worker it was sent to: that one's end fails it, another's does not.
        var worker: ObjectIdentifier?
        /// A render's latents between its DiT and its decoder (`RenderLoop.keeping`), removed when the call ends.
        var kept: URL?
        /// The call sent again, to the next worker: a render the fuse stopped while it decoded, once (`taken`).
        var again: (@Sendable () -> Void)?
        var taken = false

        /// Whether a worker that stopped with `error` hands it on: the fuse, past a DiT whose latents are kept, once.
        func handedOn(after error: EngineError) -> Bool {
            guard case .engineStopped(_?, _) = error, !taken, !abandoned, again != nil, let kept else { return false }
            return KeptLatents.read(kept).map { !$0.images.isEmpty } ?? false
        }

        func ended() { if let kept { try? FileManager.default.removeItem(at: kept) } }
    }

    private func register(_ reply: @escaping @Sendable (WorkerReply, [UInt32: Data]) -> Void,
                          stopped: @escaping @Sendable (EngineError) -> Void) -> UInt32 {
        lock.withLock {
            lastID &+= 1
            calls[lastID] = Pending(reply: reply, stopped: stopped)
            return lastID
        }
    }

    /// The consumer is gone (the stream ended or was cancelled): a call still running is cancelled in the worker — after
    /// its own sending, on the same queue, so a call cancelled before it went out is cancelled all the same.
    private func cancel(_ id: UInt32) {
        let first = lock.withLock { () -> Bool in
            guard calls[id] != nil, calls[id]?.abandoned == false else { return false }
            calls[id]?.abandoned = true
            return true
        }
        guard first else { return }
        sending.async { [self] in
            guard let pipe = lock.withLock({ calls[id] != nil ? live?.pipe : nil }) else { return }
            pipe.send(WorkerCall.cancel(id: id))
        }
    }

    private func send(_ id: UInt32, _ call: WorkerCall, blobs: [(UInt32, Data)]) {
        let worker: Live
        do {
            worker = try lock.withLock {
                guard calls[id] != nil else { throw CancellationError() }
                let w = try launched()
                calls[id]?.worker = ObjectIdentifier(w)
                return w
            }
        } catch is CancellationError {
            return
        } catch {
            finish(id) { $0.stopped(EngineError(error)) }
            return
        }
        // A worker that died meanwhile fails its calls itself (`died`), this one with them.
        worker.pipe.send(call, blobs: blobs)
    }

    private func finish(_ id: UInt32, _ body: (Pending) -> Void) {
        guard let p = lock.withLock({ calls.removeValue(forKey: id) }) else { return }
        body(p)
    }
}
