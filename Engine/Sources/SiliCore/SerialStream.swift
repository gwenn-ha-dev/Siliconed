import CoreAI
import Dispatch
import Foundation
import Metal
import Synchronization

/// **Core AI's stream, touched by one thread at a time**: every call put on it (`Run.encode`) and every wait for it
/// (`done`) go through here, and nothing else holds a `ComputeStream`.
///
/// `ComputeStream` is not `Sendable`, and its sync point counter, read by each `encode`, is read then written back
/// without a lock by each `currentWorkCompleted` and each task Core AI puts in its order (`appendTask`): a wait that
/// read it, then wrote it back after another wait's write, lowers it, and the next wait finds it below its own point, a
/// fatal error (`ComputeStream.swift:421`, read in CoreAIRuntime 3605.6.4). The 0.32's worker died so twice in a night
/// on the M1 Pro (25/11/2026), each time in the second run of a client's blocks, and once on the M5 Pro (22/11). One
/// call and four waits at once, 20 000 times (`aStreamsWaitsSideBySide`), died within 10 000 in 4 runs of 4 with the
/// waits on the cooperative pool, and lived 4 of 4, in the same time, as here (25/11 on the M1 Pro).
///
/// So the calls are put on under a lock, and the waits run on an executor of their own, a serial queue whose every job
/// holds that lock: only Core AI's code of a wait runs there, so a job never waits for another.
///
/// **A wait's point is the calls before it**: `currentWorkCompleted` waits for « all previous work encoded to this
/// stream » when its task runs, not when `done` is asked — and the task runs once the queue gets to it, under the lock
/// the next call's encoding holds. Two calls put on back to back, the first's wait was often the second's: a
/// transformer's attention, put on right after its heads, read at 10.45 TFLOP/s on a GPU of 4.6 at most, its time
/// counted in the heads' (26/11 on the M1 Pro). So a call is put on once every job put on the queue before it has run
/// (`settle`).
public final class SerialStream: @unchecked Sendable {
    private let stream: ComputeStream
    private let lock = NSLock()
    private let executor: Waits

    public init(commandQueue: MTLCommandQueue) {
        stream = ComputeStream(commandQueue: commandQueue)
        executor = Waits(lock)
    }

    /// `body` given the stream, alone on it, every wait asked for before it at its point.
    func encode<T>(_ body: (ComputeStream) throws -> T) rethrows -> T {
        executor.settle()
        return try lock.withLock { try body(stream) }
    }

    /// **A task done once every call put on the stream so far is**, its value when (`uptimeNanoseconds`); written also
    /// into `end` where a trace reads it without waiting (`Stages.tracing`).
    func done(_ end: Stages.End?) -> Task<UInt64, Never> {
        // The waits' executor holds the lock for every step of this task: the stream is its alone while it runs.
        nonisolated(unsafe) let stream = stream
        return Task(executorPreference: executor) {
            await stream.currentWorkCompleted()
            let t = DispatchTime.now().uptimeNanoseconds
            end?.value.store(t, ordering: .releasing)
            return t
        }
    }

    /// **A task done once every call put on the stream so far is**, its value when (`uptimeNanoseconds`).
    public func done() -> Task<UInt64, Never> { done(nil) }

    /// The waits' executor: one queue, each job under the stream's lock.
    private final class Waits: TaskExecutor, @unchecked Sendable {
        private let queue = DispatchQueue(label: "silicore.stream.waits", qos: .userInitiated)
        private let lock: NSLock
        init(_ lock: NSLock) { self.lock = lock }
        func enqueue(_ job: consuming ExecutorJob) {
            let job = UnownedJob(job)
            queue.async { self.lock.withLock { job.runSynchronously(on: self.asUnownedTaskExecutor()) } }
        }
        /// Every job put on the queue so far run. Never from the queue itself, nor under the lock.
        func settle() { queue.sync {} }
    }
}
