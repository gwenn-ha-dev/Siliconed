import Foundation
import Metal
import ObjectiveC
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
///     and the encode is all its wall measures. `gpuSeconds` gives them back by label; the render's ledger gets them by class through
///     the execution open when each buffer was settled (`charge`, `ProductLedger.Charge`), the ANE's side timed on its own thread.
///   - **A refusal is reported at the drain**, not lost: a completed buffer's error is kept and said once, as `Attention.verdict` does.
package final class GPUStream: @unchecked Sendable {   // its state under `lock`
    /// **Where a denoiser streams** (`SILICONED_STREAM`, a forcing of measurement): `on`, the default, the stream with the ANE's rows on
    /// its event; `off` each call waited for as before the stream (the CPU's norms, other bits); `gpu` the stream with no ANE rows
    /// (each product the GPU's whole). The five families' streamed blocks are judged against the oracle and keep their bits with a
    /// wait at every operation (`SILICONED_STREAM_WAIT`).
    package enum Policy: String, Sendable { case off, on, gpu }
    package static let policy: Policy = {
        let p: Policy = switch ProcessInfo.processInfo.environment["SILICONED_STREAM"] {
        case "0", "off": .off
        case "gpu": .gpu
        default: .on
        }
        if ProcessInfo.processInfo.environment["SILICONED_STREAM_REPORT"] != nil { atexit { print(GPUStream.report()) } }
        return p
    }()

    private let lock = NSLock()
    private var pending: [(commands: MTLCommandBuffer, label: String, what: @Sendable () -> String, charge: ProductLedger.Charge?,
                           waitsFor: UInt64)] = []
    private var seconds: [String: Double] = [:]
    private var on = false
    private var charged: ProductLedger.Charge?
    /// The ledger's clock on the GPU: where the last settled buffer ended, the ANE's rows each waited-for value stands for ended (host
    /// seconds, `gpuEndTime`'s base), and the value a buffer made here waits for (`awaitNeuralEngine`), until it is settled.
    private var lastEnd: CFTimeInterval = 0
    private var aneEnds: [UInt64: CFTimeInterval] = [:]
    private var waits: [ObjectIdentifier: UInt64] = [:]

    /// **The render's execution the buffers settled from here are charged to** (`ProductLedger.Charge`): set by the ledger's caller
    /// around one product or attention, while the stream queues; at the drain each buffer's GPU seconds go to it, and the ANE's rows
    /// handed meanwhile (`neuralEngine`) add their seconds on the ANE's thread. `nil`: the buffers are no class's (a norm, a SwiGLU).
    package var charge: ProductLedger.Charge? {
        get { lock.lock(); defer { lock.unlock() }; return charged }
        set { lock.lock(); charged = newValue; lock.unlock() }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static let key = UnsafeMutableRawPointer.allocate(byteCount: 1, alignment: 1)

    /// The stream of `queue`, made on first use. It rides on the queue (an associated object) and dies with it: a table by the queue's
    /// address would hand a dead queue's stream — its switch, its pending buffers — to a new queue made at the same address, and queues
    /// die now (`PhaseBoundary.returnedDenoiser`).
    package static func of(_ queue: MTLCommandQueue) -> GPUStream {
        lock.lock(); defer { lock.unlock() }
        if let s = objc_getAssociatedObject(queue, key) as? GPUStream { return s }
        let s = GPUStream()
        objc_setAssociatedObject(queue, key, s, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
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
        pending.append((commands, label, what, charged, waits.removeValue(forKey: ObjectIdentifier(commands)) ?? 0))
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
        let waited: Double = Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9
        Self.tally.withLock { $0.drains += 1; $0.waited += waited; $0.buffers += queued.count }
        var added: [String: Double] = [:]
        var charges: [ProductLedger.Charge] = []
        lock.lock(); var end = lastEnd; let ends = aneEnds; aneEnds = [:]; waits = [:]; lock.unlock()
        var behindGPU = 0.0, behindANE = 0.0
        for q in queued {
            q.commands.waitUntilCompleted()   // a queue completes in order: already done, unless it was another queue's
            let s = Attention.verdict(q.commands, what: q.what())
            added[q.label, default: 0] += s
            // **What the ledger charges is the buffer's own span**: `gpuStartTime` is when the buffer began, which can be before the
            // buffer ahead of it ended (its kernels then wait on the hazard) or before the ANE's rows it waits for were written (the
            // event); that wait is the other's time, already counted. From the later of the three to its end.
            let start = q.commands.gpuStartTime, finish = q.commands.gpuEndTime
            let afterGPU = max(start, end), afterANE = max(afterGPU, ends[q.waitsFor] ?? 0)
            let own = max(0, finish - afterANE)
            behindGPU += max(0, min(afterGPU, finish) - start); behindANE += max(0, min(afterANE, finish) - min(afterGPU, finish))
            end = max(end, finish)
            if let c = q.charge {
                c.addGPU(own)
                if charges.last !== c { charges.append(c) }
            }
        }
        lock.lock(); lastEnd = end; for (k, v) in added { seconds[k, default: 0] += v }; lock.unlock()
        Self.tally.withLock { t in
            for (k, v) in added { t.gpu[k, default: 0] += v }
            t.behindGPU += behindGPU; t.behindANE += behindANE
        }
        for c in charges { c.credit() }       // a charge credited twice adds nothing the second time
    }

    /// **The prototype's account** (`SILICONED_STREAM_REPORT`): drains, the CPU's seconds waiting in them, buffers settled, GPU
    /// seconds by label — printed at the process's exit.
    package struct Tally: Sendable { var drains = 0; var waited = 0.0; var buffers = 0; var gpu: [String: Double] = [:]; var ane = 0.0
                                    /// Seconds inside the buffers' spans spent behind the buffer ahead, or behind the ANE's rows waited for.
                                    var behindGPU = 0.0, behindANE = 0.0 }
    package static let tally = Mutex<Tally>(Tally())
    package static func report() -> String {
        tally.withLock { t in
            let gpu = t.gpu.sorted { $0.key < $1.key }.map { String(format: "%@ %.2f s", $0.key, $0.value) }.joined(separator: " · ")
            return String(format: "stream: %d drains, %.2f s waited, %d buffers · ", t.drains, t.waited, t.buffers) + gpu
                + (t.ane > 0 ? String(format: " · ANE's side %.2f s", t.ane) : "")
                + (t.buffers > 0 ? String(format: " · inside the spans: behind the GPU %.2f s, behind the ANE %.2f s", t.behindGPU, t.behindANE) : "")
                + " · rounded products \(GEMM.roundedProducts.withLock { $0 }), their operand rounded by its producer \(GEMM.producerRounded.withLock { $0 })"
                + ", operands with no fp32 written \(GEMM.fp32Unwritten.withLock { $0 })"
                + " · W8A8 activation passes \(GEMM.quantizePasses.withLock { $0 }), products on an operand quantized ahead"
                + " \(GEMM.prequantizedProducts.withLock { $0 }), operands quantized by their producer \(GEMM.producerQuantized.withLock { $0 })"
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
        let charge = charged
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
            let began = DispatchTime.now().uptimeNanoseconds
            let computed = work()
            GEMM.neuralEngineTally.withLock { $0.handed += 1; if !computed { $0.givenBack += 1 } }
            if !computed { fallback() }
            // The ANE's side, from its operand ready to its rows written: what the GPU waits for before the next buffer.
            let ended = DispatchTime.now().uptimeNanoseconds   // mach_absolute_time's base, `gpuEndTime`'s
            let side = Double(ended - began) * 1e-9
            charge?.addANE(side)
            GPUStream.tally.withLock { $0.ane += side }
            self.lock.lock(); self.aneEnds[done] = Double(ended) * 1e-9; self.lock.unlock()
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
        waits[ObjectIdentifier(commands)] = awaited
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
