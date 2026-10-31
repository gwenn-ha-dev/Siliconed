import Foundation

/// Splits a job by rows across the cores, with a threshold.
///
/// `concurrentPerform` costs a few tens of microseconds to launch: below a certain
/// amount of work, parallelizing slows things down. The threshold is in rows × columns, not in rows — a block of
/// 32 tokens over 10240 columns is worth a block of 4128 over 80.
package enum Parallel {
    /// The elementwise ops weighed 4.4 s out of 27 of a Z-Image 1024² evaluation on a single thread; they are cut by rows across the cores.
    package static let threads = max(2, ProcessInfo.processInfo.activeProcessorCount - 2)
    /// `SILICONED_PARALLEL`: `0` disables, otherwise the threshold. For bisecting — an operation
    /// split by rows should be bit-for-bit identical, so any deviation is a defect.
    package static let threshold: Int = {
        let requested = EngineSettings.effective.parallel
        if requested == 0 && EngineSettings.effective.provenance[.parallel] == nil { return 1 << 18 }
        return requested == 0 ? Int.max : requested
    }()

    @inline(__always)
    package static func rows(_ count: Int, width: Int, _ body: (Int, Int) -> Void) {
        guard count * width >= threshold, count >= threads * 2 else {
            body(0, count)
            return
        }
        // Each thread receives its own rows, bounded by their ends (`count · worker / threads`):
        // `body` never writes twice to the same place, and `concurrentPerform` returns when all
        // have finished — it does not escape.
        withoutActuallyEscaping(body) { body in
            nonisolated(unsafe) let body = body
            DispatchQueue.concurrentPerform(iterations: threads) { worker in
                let start = count * worker / threads
                let end = count * (worker + 1) / threads
                if start < end { body(start, end - start) }
            }
        }
    }

    /// `body(0..<iterations)`, the iterations spread over the cores, returning when all are done: for work that is already
    /// cut into independent tasks (a head, a piece of a buffer), each owning what it writes. `SILICONED_PARALLEL=0` runs them in
    /// order on the calling thread, like `rows` — the bisection switch covers every parallel loop of the engine.
    package static func perform(_ iterations: Int, _ body: (Int) -> Void) {
        guard iterations > 1, threshold != Int.max else {
            for i in 0..<iterations { body(i) }
            return
        }
        withoutActuallyEscaping(body) { body in
            nonisolated(unsafe) let body = body
            DispatchQueue.concurrentPerform(iterations: iterations) { body($0) }
        }
    }
}
