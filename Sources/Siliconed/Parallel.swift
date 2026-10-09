import Foundation

/// Splits a job by rows across the cores, with a threshold.
///
/// `concurrentPerform` costs a few tens of microseconds to launch: below a certain
/// amount of work, parallelizing slows things down. The threshold is in rows × columns, not in rows — a block of
/// 32 tokens over 10240 columns is worth a block of 4128 over 80.
package enum Parallel {
    /// Measured at stage 2: the elementwise ops weigh 4.4 s out of 27 of evaluation, on a single thread.
    package static let threads = max(2, ProcessInfo.processInfo.activeProcessorCount - 2)
    /// `SILICONED_PARALLEL`: `0` disables, otherwise the threshold. For bisecting — an operation
    /// split by rows should be bit-for-bit identical, so any deviation is a defect.
    package static let threshold: Int = {
        let requested = EngineSettings.effective.parallel
        if requested == 0 && EngineSettings.effective.provenance["parallel"] == nil { return 1 << 18 }
        return requested == 0 ? Int.max : requested
    }()

    @inline(__always)
    package static func rows(_ count: Int, width: Int, _ body: (Int, Int) -> Void) {
        guard count * width >= threshold, count >= threads * 2 else {
            body(0, count)
            return
        }
        let chunk = (count + threads - 1) / threads
        // Each thread receives its own rows: `body` never writes twice to the same place,
        // and `concurrentPerform` returns when all have finished — it does not escape.
        withoutActuallyEscaping(body) { body in
            nonisolated(unsafe) let body = body
            DispatchQueue.concurrentPerform(iterations: threads) { worker in
                let start = worker * chunk
                let end = min(start + chunk, count)
                if start < end { body(start, end - start) }
            }
        }
    }
}
