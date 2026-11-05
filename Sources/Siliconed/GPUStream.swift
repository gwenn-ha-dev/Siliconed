import Foundation
import Metal
import Synchronization

/// **The GPU's work queued without waiting, and waited for only where the CPU reads or writes** — a denoiser's evaluation as one
/// stream of command buffers, instead of a submission and a wait per product, attention and norm.
///
///     off (the default):  encode ─ commit ─ wait ─ verdict ─▶ the CPU's next step        each call its own round trip
///     on:                 encode ─ commit ─▶ encode ─ commit ─▶ …  ─ drain ─▶ the CPU      one wait where the CPU needs the bytes
///
/// One stream per command queue (`of`): every GPU component of a denoiser shares its `GEMM`'s queue, so a buffer committed later runs
/// after the earlier ones, and Metal's hazard tracking (the buffers are tracked, the default) orders a read after the write before it.
///
/// What sets it apart — what a port by analogy with a wait per call would get wrong:
///
///   - **The CPU must drain before it touches shared memory** the queued work reads or writes: the unified memory makes a CPU write
///     after a commit a race, not an error. The caller owns that rule (`drain()` before a vDSP pass, a staging, a modulation, a
///     boundary recorded, the ANE's half of a product); a call that waits on its own buffer (`settle` off) needs nothing.
///   - **Seconds are the GPU's own** (`gpuEndTime − gpuStartTime`), added at the drain, not the caller's wall: on, a call returns 0
///     and the encode is all its wall measures. `gpuSeconds` gives them back by label.
///   - **A refusal is reported at the drain**, not lost: a completed buffer's error is kept and said once, as `Attention.verdict` does.
package final class GPUStream: @unchecked Sendable {   // its state under `lock`
    /// **Where a denoiser streams** (`SILICONED_STREAM`, a prototype's switch): `off` as before; `on` the stream, the ANE's products
    /// waited for as before; `gpu` the stream with no ANE rows (each product the GPU's whole), the ANE's rendezvous gone.
    package enum Policy: String, Sendable { case off, on, gpu }
    package static let policy: Policy = {
        let p: Policy = switch ProcessInfo.processInfo.environment["SILICONED_STREAM"] {
        case "1", "on": .on
        case "gpu": .gpu
        default: .off
        }
        if ProcessInfo.processInfo.environment["SILICONED_STREAM_REPORT"] != nil { atexit { print(GPUStream.report()) } }
        return p
    }()

    private let lock = NSLock()
    private var pending: [(commands: MTLCommandBuffer, label: String, what: @Sendable () -> String)] = []
    private var seconds: [String: Double] = [:]
    private var on = false

    private static let lock = NSLock()
    nonisolated(unsafe) private static var streams: [ObjectIdentifier: GPUStream] = [:]

    /// The stream of `queue`, made on first use.
    package static func of(_ queue: MTLCommandQueue) -> GPUStream {
        lock.lock(); defer { lock.unlock() }
        let key = ObjectIdentifier(queue)
        if let s = streams[key] { return s }
        let s = GPUStream()
        streams[key] = s
        return s
    }

    /// Whether calls queue without waiting. Turning it off drains first.
    package var isOn: Bool {
        get { lock.lock(); defer { lock.unlock() }; return on }
        set {
            if !newValue { drain() }
            lock.lock(); on = newValue; lock.unlock()
        }
    }

    /// **A committed buffer, settled**: off, waited for and judged (`Attention.verdict`), its GPU seconds returned; on, kept until the
    /// next `drain`, which judges it and adds its seconds under `label`, and 0 returned. Works on a buffer committed elsewhere.
    @discardableResult
    package func settle(_ commands: MTLCommandBuffer, label: String, what: @escaping @Sendable () -> String) -> Double {
        lock.lock()
        guard on else {
            lock.unlock()
            commands.waitUntilCompleted()
            return Attention.verdict(commands, what: what())
        }
        pending.append((commands, label, what))
        lock.unlock()
        return 0
    }

    /// **Waits for everything queued**, then judges each buffer and adds its GPU seconds: after it the CPU may read and write what
    /// the queued work touched.
    package func drain() {
        lock.lock(); let queued = pending; pending = []; let ane = (event, awaited); lock.unlock()
        if let event = ane.0, ane.1 > 0 { _ = event.wait(untilSignaledValue: ane.1, timeoutMS: 600_000) }
        guard let last = queued.last else { return }
        let began = DispatchTime.now().uptimeNanoseconds
        last.commands.waitUntilCompleted()
        Self.tally.withLock { $0.drains += 1; $0.waited += Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9; $0.buffers += queued.count }
        var added: [String: Double] = [:]
        for q in queued {
            q.commands.waitUntilCompleted()   // a queue completes in order: already done, unless it was another queue's
            added[q.label, default: 0] += Attention.verdict(q.commands, what: q.what())
        }
        lock.lock(); for (k, v) in added { seconds[k, default: 0] += v }; lock.unlock()
        Self.tally.withLock { t in for (k, v) in added { t.gpu[k, default: 0] += v } }
    }

    /// **The prototype's account** (`SILICONED_STREAM_REPORT`): drains, the CPU's seconds waiting in them, buffers settled, GPU
    /// seconds by label — printed at the process's exit.
    package struct Tally: Sendable { var drains = 0; var waited = 0.0; var buffers = 0; var gpu: [String: Double] = [:] }
    package static let tally = Mutex<Tally>(Tally())
    package static func report() -> String {
        tally.withLock { t in
            let gpu = t.gpu.sorted { $0.key < $1.key }.map { String(format: "%@ %.2f s", $0.key, $0.value) }.joined(separator: " · ")
            return String(format: "stream: %d drains, %.2f s waited, %d buffers · ", t.drains, t.waited, t.buffers) + gpu
        }
    }

    // MARK: The Neural Engine beside the stream

    /// The event the GPU signals when an ANE product's operand is ready, and the ANE's thread when its rows are written.
    private var event: MTLSharedEvent?
    private var counter: UInt64 = 0
    /// The value the next command buffer must wait for (the last ANE product's rows), and the value already waited for in one.
    private var awaited: UInt64 = 0, encodedWait: UInt64 = 0
    private let neuralQueue = DispatchQueue(label: "siliconed.stream.ane", qos: .userInitiated)

    /// **An ANE product beside the queued GPU work, with no wait on the CPU**: a buffer signals once the queued work before it is done
    /// (the operand written); the ANE's thread waits for that, runs `work` (which writes its rows of the output and returns whether it
    /// computed them), and on a refusal `fallback` (the same rows on the CPU) before signalling; the next command buffer made on this
    /// queue after `neuralEngineQueued` waits for that signal on the GPU. The GPU's own rows of the product are queued by the caller
    /// between the two calls, beside the ANE's — not behind them.
    package func neuralEngine(on queue: MTLCommandQueue, work: @escaping @Sendable () -> Bool,
                              fallback: @escaping @Sendable () -> Void) -> UInt64 {
        lock.lock()
        if event == nil { event = queue.device.makeSharedEvent() }
        let event = event!
        counter += 2
        let ready = counter - 1, done = counter
        lock.unlock()
        guard let signal = queue.makeCommandBuffer() else { fatalError("Metal refused a command buffer") }
        lock.lock()
        if awaited > encodedWait { signal.encodeWaitForEvent(event, value: awaited); encodedWait = awaited }
        lock.unlock()
        signal.encodeSignalEvent(event, value: ready)
        signal.commit()
        nonisolated(unsafe) let e = event
        neuralQueue.async {
            _ = e.wait(untilSignaledValue: ready, timeoutMS: 600_000)
            let computed = work()
            GEMM.neuralEngineTally.withLock { $0.handed += 1; if !computed { $0.givenBack += 1 } }
            if !computed { fallback() }
            e.signaledValue = done
        }
        return done
    }

    /// The GPU's rows of the ANE's product are queued: from here every command buffer waits for the ANE's rows (`done`).
    package func neuralEngineQueued(_ done: UInt64) {
        lock.lock(); awaited = max(awaited, done); lock.unlock()
    }

    /// Encodes, at the start of `commands`, the wait for the last ANE product's rows if no buffer waits for them yet.
    func awaitNeuralEngine(in commands: MTLCommandBuffer) {
        lock.lock(); defer { lock.unlock() }
        guard awaited > encodedWait, let event else { return }
        commands.encodeWaitForEvent(event, value: awaited)
        encodedWait = awaited
    }

    /// The GPU seconds of the stream's buffers by label since the last call, and reset.
    package func gpuSeconds() -> [String: Double] {
        drain()
        lock.lock(); defer { lock.unlock() }
        let s = seconds
        seconds = [:]
        return s
    }
}

extension MTLCommandQueue {
    /// This queue's stream (`GPUStream.of`).
    package var stream: GPUStream { GPUStream.of(self) }
}

extension MTLCommandBuffer {
    /// **Commits and settles on its queue's stream**: off, waits and returns the GPU seconds; on, queues and returns 0.
    @discardableResult
    package func commitSettled(label: String, what: @escaping @Sendable @autoclosure () -> String) -> Double {
        commit()
        return commandQueue.stream.settle(self, label: label, what: what)
    }
}
