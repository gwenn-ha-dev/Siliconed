import Foundation
import SiliCore

/// **The engine's process, `silicore-worker`**: the app's renders and writer run here, never in the app (`EngineWorker`).
/// It reads its calls on its standard input and answers on its standard output (`WorkerWire.swift`), one piece of work
/// at a time, in their order; its standard error is the engine's, which the app passes on.
///
/// - **B bounds this process alone** (`MemoryBudget`): its fuse is armed before the first call, and stops it the moment
///   it holds more than B. The app lives on, and says why (`EngineWorker`).
/// - **Its life is the app's**: the end of its standard input (the app quit, or died) ends it. What it keeps between two
///   renders — the render cache's conditionings, what Core AI loaded — goes with it.
/// - **Its name is stable**: Core AI files its compilations by process name.
/// - **Its first work is Core AI's first call** (`CoreAI.witness`): every stage's floor holds it.
/// - **A preparation compiles between the calls** (`WorkerCall.prepare`): its specializations one at a time on the
///   work's queue while no call is in hand — a call that comes waits for the one under way, the others after it.
package enum WorkerServer {
    package static func run() -> Never {
        _ = MemoryBudget.machineCeiling()
        let server = Server(out: WorkerPipe(fd: STDOUT_FILENO))
        // Its version before anything else (`WorkerHello`): an app of another build refuses it on that alone.
        server.out.send(WorkerHello())
        server.witness()
        Thread.detachNewThread { server.read(STDIN_FILENO) }
        dispatchMain()
    }
}

private final class Server: @unchecked Sendable {
    let out: WorkerPipe
    /// The calls run here, one after the other: a render waits for the writer's run before it, as on the engine's queue.
    private let work = DispatchQueue(label: "silicore-worker.work", qos: .userInitiated)
    private let lock = NSLock()
    private var calls: [UInt32: Call] = [:]
    private var blob: UInt32 = 0
    /// Under `lock`: what the form's last preparation left to specialize, then the others' (`WorkerCall.prepare`), the
    /// ANE's first in each; whether one is on the queue.
    private var pending: [Functions.Specialization] = [], background: [Functions.Specialization] = [], queued = false

    /// A call in hand: its token, and what stops it once it runs (the writer's own cancel).
    final class Call: @unchecked Sendable {
        let token = Cancellation()
        private let lock = NSLock()
        private var stop: (() -> Void)?
        func cancel() {
            token.cancel()
            lock.withLock { stop }?()
        }
        func onCancel(_ action: @escaping () -> Void) {
            lock.withLock { stop = action }
            if token.isCancelled { action() }
        }
    }

    init(out: WorkerPipe) { self.out = out }

    func read(_ fd: Int32) {
        var inbox = WorkerInbox()
        var buffer = [UInt8](repeating: 0, count: 1 << 20)
        var greeted = false
        while true {
            let n = Darwin.read(fd, &buffer, buffer.count)
            if n < 0, errno == EINTR { continue }
            // The app is gone: so is its engine.
            guard n > 0 else { exit(0) }
            do {
                for message in try inbox.receive(Data(buffer[0..<n])) {
                    guard greeted else {
                        let theirs = WorkerHello.version(of: message)
                        guard theirs == WorkerHello.version else {
                            let said = theirs.map { "protocol \($0)" } ?? "no protocol"
                            FileHandle.standardError.write(Data("silicore-worker: protocol \(WorkerHello.version), the app speaks \(said): refused\n".utf8))
                            exit(66)
                        }
                        greeted = true
                        continue
                    }
                    handle(try WireJSON.decode(WorkerCall.self, from: message), &inbox)
                }
            } catch {
                FileHandle.standardError.write(Data("silicore-worker: the app sent what no build of ours sends (\(error))\n".utf8))
                exit(65)
            }
        }
    }

    private func handle(_ call: WorkerCall, _ inbox: inout WorkerInbox) {
        switch call {
        case let .render(id, order):
            let blobs = inbox.take(order.blobs), c = Call()
            lock.withLock { calls[id] = c }
            work.async { self.render(id, order, blobs, c) }
        case let .write(id, order):
            let blobs = inbox.take(order.blobs), c = Call()
            lock.withLock { calls[id] = c }
            work.async { self.write(id, order, blobs, c) }
        case let .prepare(id, order, background):
            let blobs = inbox.take(order.blobs)
            lock.withLock { calls[id] = Call() }
            work.async { self.prepare(id, order, blobs, background: background) }
        case .erase(let id):
            lock.withLock { calls[id] = Call() }
            work.async {
                self.out.send(WorkerReply.erased(id: id, bytes: CoreAI.erase()))
                self.ended(id, nil)
            }
        case .cancel(let id):
            lock.withLock { calls[id] }?.cancel()
        }
    }

    /// The process's first call of Core AI (`CoreAI.witness`), the first work of the queue: every call's stages after it.
    func witness() {
        work.async {
            CoreAI.place(.processWide)
            CoreAI.witness()
        }
    }

    private func ended(_ id: UInt32, _ failure: WorkerFailure?) {
        lock.withLock { calls[id] = nil }
        out.send(failure.map { WorkerReply.failed(id: id, $0) } ?? .finished(id: id))
        specializeNext()
    }

    /// **What a render of `order` would specialize** (`Engine.prepare`): the form's in place of what its last left, the
    /// others after theirs. The call ends there, the specializations come after it.
    private func prepare(_ id: UInt32, _ order: RenderOrder, _ blobs: [UInt32: Data], background later: Bool) {
        CoreAI.neuralEngine = order.neuralEngine
        CoreAI.place(order.library.library)
        do {
            let request = try order.request(blobs)
            let model = try Model.named(order.model, in: order.library.library)
            let all = try Engine().prepare(request, model: model)
            lock.withLock { if later { background += all } else { pending = all } }
            ended(id, nil)
        } catch {
            ended(id, WorkerFailure(error))
        }
    }

    /// **The next specialization put on the work's queue**, where no call is in hand and none is queued.
    private func specializeNext() {
        let go = lock.withLock { () -> Bool in
            guard calls.isEmpty, !(pending.isEmpty && background.isEmpty), !queued else { return false }
            queued = true
            return true
        }
        if go { work.async { self.specialize() } }
    }

    /// **One specialization**, unless a call came since it was queued; its compilation told to the app
    /// (`WorkerReply.preparing`), a failure on the standard error — the render that needs it compiles it, and says why.
    private func specialize() {
        let next = lock.withLock { () -> Functions.Specialization? in
            queued = false
            guard calls.isEmpty else { return nil }
            return pending.isEmpty ? background.isEmpty ? nil : background.removeFirst() : pending.removeFirst()
        }
        guard let next else { return }
        let out = out
        let observation = Stages.observe { event in
            switch event {
            case .compiling: out.send(WorkerReply.preparing(compiling: true))
            case .compiled: out.send(WorkerReply.preparing(compiling: false))
            default: break
            }
        }
        do { try blocking { try await Functions.specialize(next) } } catch {
            FileHandle.standardError.write(Data("silicore-worker: \(next.asset.lastPathComponent) not prepared: \(error)\n".utf8))
        }
        observation.cancel()
        specializeNext()
    }

    private func nextBlob() -> UInt32 { lock.withLock { blob &+= 1; return blob } }

    /// The app's activity (`AppState.next`) is the app's process's: the engine's asks the system the same — user
    /// initiated, latency critical — for the time of each call.
    private func busy<T>(_ work: () -> T) -> T {
        let activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .latencyCritical], reason: "Image render")
        defer { ProcessInfo.processInfo.endActivity(activity) }
        return work()
    }

    private func render(_ id: UInt32, _ order: RenderOrder, _ blobs: [UInt32: Data], _ call: Call) {
        busy { rendering(id, order, blobs, call) }
    }

    private func write(_ id: UInt32, _ order: WriteOrder, _ blobs: [UInt32: Data], _ call: Call) {
        busy { writing(id, order, blobs, call) }
    }

    private func rendering(_ id: UInt32, _ order: RenderOrder, _ blobs: [UInt32: Data], _ call: Call) {
        CoreAI.neuralEngine = order.neuralEngine
        CoreAI.place(order.library.library)
        RenderLoop.keeping = order.keeping.map { URL(fileURLWithPath: $0) }
        defer { RenderLoop.keeping = nil }
        do {
            let request = try order.request(blobs)
            let model = try Model.named(order.model, in: order.library.library)
            _ = try Engine().renderBatch(request, seeds: order.seeds, model: model, cancellation: call.token) { event in
                let (e, data) = WorkerEvent.of(event, blob: self.nextBlob())
                self.out.send(WorkerReply.event(id: id, e), blobs: data.map { [(e.blob!, $0)] } ?? [])
            }
            ended(id, nil)
        } catch {
            ended(id, WorkerFailure(error))
        }
    }

    private func writing(_ id: UInt32, _ order: WriteOrder, _ blobs: [UInt32: Data], _ call: Call) {
        CoreAI.neuralEngine = order.neuralEngine
        do {
            try call.token.check()
            CoreAI.place(order.library.library)
            guard let writer = PromptWriter(library: order.library.library) else { throw PromptWriter.Failure.notInstalled }
            let request: PromptWriter.Request
            switch order.input {
            case .rewrite(let text): request = .rewrite(text: text, instruction: order.instruction)
            case .describe(let b):
                guard let data = blobs[b] else { throw EngineError.imageUnreadable(file: "image") }
                request = .describe(image: try PromptWriterImage.rgb8(from: data), instruction: order.instruction)
            }
            call.onCancel { writer.cancel() }
            let stream = writer.write(request)
            let out = self.out
            try blocking { for try await event in stream { out.send(WorkerReply.written(id: id, event)) } }
            ended(id, nil)
        } catch {
            ended(id, WorkerFailure(error))
        }
    }
}
