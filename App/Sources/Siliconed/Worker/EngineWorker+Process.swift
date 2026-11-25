// `EngineWorker`: the worker's process — launched, listened to, read, killed, and what its death fails.

import Darwin
import Synchronization
import Foundation
import Measurement

extension EngineWorker {
    /// The worker's executable: beside this one (`swift build`), or in the bundle's `Helpers` (`tools/app.sh`).
    static var executable: URL? {
        guard let folder = Bundle.main.executableURL?.resolvingSymlinksInPath().deletingLastPathComponent() else { return nil }
        return [folder.appendingPathComponent("silicore-worker"), folder.appendingPathComponent("../Helpers/silicore-worker").standardizedFileURL]
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    /// **The worker and the processes it launched, killed together** (`SIGKILL`): a probe's process outlived a worker
    /// killed alone, an orphan holding its asset's memory until its own room stopped it. Its children first, while they
    /// are still its own (once it is gone they belong to launchd and no list names them).
    static func kill(tree pid: pid_t) {
        var children = [pid_t](repeating: 0, count: 64)
        let count = proc_listchildpids(pid, &children, Int32(children.count * MemoryLayout<pid_t>.size))
        for child in children.prefix(max(0, Int(count))) where child > 0 { Darwin.kill(child, SIGKILL) }
        Darwin.kill(pid, SIGKILL)
    }

    /// The worker running, or a new one. The lock is held.
    func launched() throws -> Live {
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
        case .preparing(let compiling):
            preparing?(compiling)
            return
        }
        let p = lock.withLock { last ? calls.removeValue(forKey: id) : calls[id] }
        if last { p?.ended() }
        guard let p, !p.abandoned else { return }
        p.reply(reply, [:])
    }

    /// **The worker stopped**: every call in hand fails with why — a decoding the fuse stopped is handed to the next
    /// worker instead (`Pending.handedOn`) —, and the next call starts another.
    private func died(_ worker: Live) {
        let error = worker.why()
        preparing?(false)
        let (ended, again) = lock.withLock { () -> ([Pending], [@Sendable () -> Void]) in
            let mine: [UInt32: Pending]
            if live === worker || live == nil {
                live = nil
                mine = calls
            } else {
                // A worker ended for another B: the calls in hand are the next one's.
                mine = calls.filter { $0.value.worker == ObjectIdentifier(worker) }
            }
            var ended: [Pending] = [], again: [@Sendable () -> Void] = []
            for (id, var p) in mine {
                if p.handedOn(after: error) {
                    (p.taken, p.worker) = (true, nil)
                    calls[id] = p
                    again.append(p.again!)
                } else {
                    calls[id] = nil
                    ended.append(p)
                }
            }
            return (ended, again)
        }
        worker.watch?.cancel()
        for p in ended {
            p.ended()
            if !p.abandoned { p.stopped(error) }
        }
        for send in again { sending.async(execute: send) }
    }

    /// One worker: its process, its pipes, and what says why it ended.
    final class Live: @unchecked Sendable {
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
