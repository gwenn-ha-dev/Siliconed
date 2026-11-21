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
///   B, so a stop past B is a fault, not a cut to retry. What follows is the user's to decide.
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

    private let program: URL?, arguments: [String]
    private let silence: TimeInterval
    private let lock = NSCondition()
    private var live: Live?
    private var calls: [UInt32: Pending] = [:]
    private var lastID: UInt32 = 0, lastBlob: UInt32 = 0
    /// The calls go out in order, off the caller's thread: an edit's images are megabytes.
    private let sending = DispatchQueue(label: "siliconed.worker.send", qos: .userInitiated)

    /// `executable` run with `arguments`; `silence`: what a worker may stay silent, a call in hand.
    package init(executable: URL?, arguments: [String] = [], silence: TimeInterval = EngineWorker.silence) {
        program = executable; self.arguments = arguments; self.silence = silence
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
            let (order, blobs) = RenderOrder.of(request, seeds: seeds, model: model, library: library,
                                                neuralEngine: neuralEngine, firstBlob: first)
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

    private struct Pending {
        let reply: @Sendable (WorkerReply, [UInt32: Data]) -> Void
        /// The worker stopped, and why.
        let stopped: @Sendable (EngineError) -> Void
        /// Its consumer stopped listening: nothing more is handed, but the worker still computes it.
        var abandoned = false
        /// The worker it was sent to: that one's end fails it, another's does not.
        var worker: ObjectIdentifier?
    }

    private func register(_ reply: @escaping @Sendable (WorkerReply, [UInt32: Data]) -> Void,
                          stopped: @escaping @Sendable (EngineError) -> Void) -> UInt32 {
        lock.withLock {
            lastID &+= 1
            calls[lastID] = Pending(reply: reply, stopped: stopped)
            lock.broadcast()
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

    // MARK: - The process

    /// The worker's executable: beside this one (`swift build`), or in the bundle's `Helpers` (`tools/app.sh`).
    static var executable: URL? {
        guard let folder = Bundle.main.executableURL?.resolvingSymlinksInPath().deletingLastPathComponent() else { return nil }
        return [folder.appendingPathComponent("silicore-worker"), folder.appendingPathComponent("../Helpers/silicore-worker").standardizedFileURL]
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    /// The worker running, or a new one. The lock is held.
    /// **The worker and the processes it launched, killed together** (`SIGKILL`): a probe's process outlived a worker
    /// killed alone, an orphan holding its asset's memory until its own room stopped it. Its children first, while they
    /// are still its own (once it is gone they belong to launchd and no list names them).
    static func kill(tree pid: pid_t) {
        var children = [pid_t](repeating: 0, count: 64)
        let count = proc_listchildpids(pid, &children, Int32(children.count * MemoryLayout<pid_t>.size))
        for child in children.prefix(max(0, Int(count))) where child > 0 { Darwin.kill(child, SIGKILL) }
        Darwin.kill(pid, SIGKILL)
    }

    private func launched() throws -> Live {
        let budget = MemoryBudget.budget
        if let live {
            // Another B, and no call in hand but this one: the worker ends (its input closed), another starts under it.
            if live.budget == budget || calls.count > 1 { return live }
            live.close()
            self.live = nil
        }
        guard let url = program else {
            throw EngineError.engineStopped(budget: nil, detail: "no silicore-worker beside the app")
        }
        let process = Process()
        process.executableURL = url
        process.arguments = arguments
        process.environment = ProcessInfo.processInfo.environment.merging(
            ["SILICONED_MEMORY_BUDGET_GB": String(Double(budget) / Double(1 << 30))]) { $1 }
        let input = Pipe(), output = Pipe(), errors = Pipe()
        process.standardInput = input
        process.standardOutput = output
        process.standardError = errors
        let ended = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in ended.signal() }
        do { try process.run() } catch {
            throw EngineError.engineStopped(budget: nil, detail: "silicore-worker could not start: \(error)")
        }
        let worker = Live(process: process, input: input, output: output, errors: errors, budget: budget)
        live = worker
        worker.pipe.send(WorkerHello())
        // The engine's own words go where the app's go; the last ones say why it stopped. Read to their end on a thread
        // of their own, which the worker's end waits for: read as they came (`readabilityHandler`), the fuse's last
        // line could still be on its way when the calls were failed, and the stop said a crash.
        let said = DispatchSemaphore(value: 0)
        Thread.detachNewThread {
            let fd = errors.fileHandleForReading.fileDescriptor
            var buffer = [UInt8](repeating: 0, count: 64 << 10)
            while true {
                let n = Darwin.read(fd, &buffer, buffer.count)
                if n < 0, errno == EINTR { continue }
                guard n > 0 else { break }
                let bytes = Data(buffer[0..<n])
                FileHandle.standardError.write(bytes)
                worker.heard(bytes)
            }
            said.signal()
        }
        Thread.detachNewThread { [weak self] in
            self?.read(worker)
            ended.wait()
            // All it wrote is in the pipe once it ended; the bound is for a child still holding its standard error.
            _ = said.wait(timeout: .now() + 2)
            self?.died(worker)
        }
        let tick = min(5, silence / 4)
        let watch = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        watch.schedule(deadline: .now() + tick, repeating: tick)
        watch.setEventHandler { [weak self, weak worker] in
            guard let self, let worker else { return }
            self.listen(to: worker)
        }
        worker.watch = watch
        watch.resume()
        return worker
    }

    /// **A worker silent too long**, a call in hand: killed, its calls failed with why. Its clock runs only while it
    /// has a call in hand; any byte it writes on its standard output sets it back.
    private func listen(to worker: Live) {
        let now = ProcessInfo.processInfo.systemUptime
        let silent = lock.withLock { () -> Bool in
            let busy = calls.values.contains { $0.worker == ObjectIdentifier(worker) }
            return worker.silent(at: now, counting: busy, limit: silence)
        }
        guard silent else { return }
        worker.end(.engineSilent(seconds: Int(silence.rounded())))
        Self.kill(tree: worker.pid)
    }

    /// The worker's replies, until its standard output ends.
    private func read(_ worker: Live) {
        var inbox = WorkerInbox()
        var buffer = [UInt8](repeating: 0, count: 1 << 20)
        let fd = worker.output.fileHandleForReading.fileDescriptor
        var greeted = false
        while true {
            let n = Darwin.read(fd, &buffer, buffer.count)
            if n < 0, errno == EINTR { continue }
            guard n > 0 else { return }
            worker.spoke(at: ProcessInfo.processInfo.systemUptime)
            do {
                for message in try inbox.receive(Data(buffer[0..<n])) {
                    guard greeted else {
                        let theirs = WorkerHello.version(of: message)
                        guard theirs == WorkerHello.version else {
                            worker.end(.engineMismatch(app: WorkerHello.version, engine: theirs))
                            Self.kill(tree: worker.pid)
                            return
                        }
                        greeted = true
                        continue
                    }
                    let reply = try WireJSON.decode(WorkerReply.self, from: message)
                    deliver(reply, &inbox)
                }
            } catch {
                // What no build of ours writes: this worker is not trusted further, and the end says why, not a signal.
                worker.end(.internalFailure(component: "worker", detail: "a reply the app cannot read: \(error)"))
                Self.kill(tree: worker.pid)
                return
            }
        }
    }

    private func deliver(_ reply: WorkerReply, _ inbox: inout WorkerInbox) {
        let id: UInt32, last: Bool
        switch reply {
        case .event(let i, let e):
            id = i; last = false
            let blobs = inbox.take(e.blob.map { [$0] } ?? [])
            guard let p = lock.withLock({ calls[id] }), !p.abandoned else { return }
            p.reply(reply, blobs)
            return
        case .written(let i, _), .erased(let i, _): id = i; last = false
        case .finished(let i), .failed(let i, _): id = i; last = true
        }
        let p = lock.withLock { last ? calls.removeValue(forKey: id) : calls[id] }
        guard let p, !p.abandoned else { return }
        p.reply(reply, [:])
    }

    /// **The worker stopped**: every call in hand fails with why, and the next call starts another.
    private func died(_ worker: Live) {
        let pending = lock.withLock { () -> [Pending] in
            guard live === worker || live == nil else {
                // A worker ended for another B: the calls in hand are the next one's.
                let ended = calls.filter { $0.value.worker == ObjectIdentifier(worker) }
                for id in ended.keys { calls[id] = nil }
                lock.broadcast()
                return Array(ended.values)
            }
            live = nil
            let p = Array(calls.values)
            calls = [:]
            lock.broadcast()
            return p
        }
        worker.watch?.cancel()
        let error = worker.why()
        for p in pending where !p.abandoned { p.stopped(error) }
    }

    /// One worker: its process, its pipes, and what says why it ended.
    private final class Live: @unchecked Sendable {
        let process: Process
        let input: Pipe, output: Pipe, errors: Pipe
        let pipe: WorkerPipe
        let pid: pid_t
        /// The B it was launched with.
        let budget: Int
        /// What reads its silence (`listen`), from its launch to its end.
        var watch: DispatchSourceTimer?
        private let lock = NSLock()
        /// The last of what it wrote on its standard error.
        private var said = Data()
        /// When it last wrote on its standard output, or was last idle: where its silence counts from.
        private var heardAt = ProcessInfo.processInfo.systemUptime
        /// Why the app ended it, when the app did.
        private var ending: EngineError?

        init(process: Process, input: Pipe, output: Pipe, errors: Pipe, budget: Int) {
            self.process = process; self.input = input; self.output = output; self.errors = errors
            self.budget = budget
            pipe = WorkerPipe(fd: input.fileHandleForWriting.fileDescriptor)
            pid = process.processIdentifier
        }

        func heard(_ bytes: Data) {
            lock.withLock {
                said.append(bytes)
                if said.count > 16 << 10 { said = Data(said.suffix(8 << 10)) }
            }
        }

        func close() { try? input.fileHandleForWriting.close() }

        func spoke(at now: TimeInterval) { lock.withLock { heardAt = now } }

        /// Whether it has been silent past `limit`; a silence not `counting` (no call in hand) sets the clock back.
        func silent(at now: TimeInterval, counting: Bool, limit: TimeInterval) -> Bool {
            lock.withLock {
                guard counting else { heardAt = now; return false }
                return ending == nil && now - heardAt > limit
            }
        }

        /// The app ends it, and why: what its calls fail with.
        func end(_ why: EngineError) { lock.withLock { if ending == nil { ending = why } } }

        /// **Why it stopped**: the app's reason when the app ended it, its fuse (its last words say B), or anything
        /// else — a crash.
        func why() -> EngineError {
            if let ending = lock.withLock({ ending }) { return ending }
            let gib = { (b: Int) in String(format: "%.2f GiB", Double(b) / Double(1 << 30)) }
            let words = lock.withLock { String(decoding: said, as: UTF8.self) }
            let lines = words.split(separator: "\n").map(String.init)
            // The fuse's sentence (`FootprintWatch`): the process past B, stopped from inside.
            if let fused = lines.last(where: { $0.hasPrefix(MemoryBudget.fuseWords) }) {
                return .engineStopped(budget: budget, detail: "silicore-worker past B = \(gib(budget)): " + Library.withoutHome(fused))
            }
            let end = process.terminationReason == .uncaughtSignal ? "signal \(process.terminationStatus)" : "status \(process.terminationStatus)"
            return .engineStopped(budget: nil, detail: Library.withoutHome((["silicore-worker ended (\(end))"] + lines.suffix(1)).joined(separator: ": ")))
        }
    }
}
