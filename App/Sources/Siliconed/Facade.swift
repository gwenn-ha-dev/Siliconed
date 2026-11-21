import Foundation
import os
import SiliCore

// **The render's facade**: what a process that renders calls — the app's engine (`WorkerServer`, whose events the app
// reads through `EngineWorker`), the command line, the checks and the tests.
//
// `render` and `renderBatch` block the thread that calls them — over four minutes at 1024² for Krea 2 — and run the
// render **on the engine queue** (`Engine.file`, static, one render at a time for the whole process). Raising the token
// stops the render at the next boundary, which throws `EngineError.cancelled`. Both go through `renderBatchOnQueue`, the
// seam where a family's render runs.
//
// **Same parameters, in the same order**: `(request, [seeds:], model:, [cancellation:], [onProgress:])`. A custom chain
// enters as a `Model(card:chain:)`. `onProgress` is called on the render thread, synchronously: it must return quickly.

extension Engine {
    /// **Renders an image. Blocks, and serializes concurrent calls.** **Throws only `EngineError`**.
    package func render(_ request: Request, model: Model, cancellation: Cancellation? = nil,
                        onProgress: (@Sendable (Event) -> Void)? = nil) throws(EngineError) -> Render {
        try renderBatch(request, seeds: [request.seed], model: model, cancellation: cancellation, onProgress: onProgress)[0]
    }

    /// **A batch: the same request under several seeds.** `request.seed` is ignored.
    package func renderBatch(_ request: Request, seeds: [UInt64], model: Model, cancellation: Cancellation? = nil,
                             onProgress: (@Sendable (Event) -> Void)? = nil) throws(EngineError) -> [Render] {
        // Called from the engine's own queue (a progress closure), `sync` below would wait for itself for ever.
        dispatchPrecondition(condition: .notOnQueue(Engine.file))
        return try EngineError.boundary {
            try Engine.file.sync { try renderBatchOnQueue(request, seeds: seeds, model: model, cancellation: cancellation,
                                                          onProgress: onProgress) }
        }
    }

    /// **The render, on the engine queue**, in this order: the request's own refusals (`Admission`, pure: a request the model
    /// cannot render is refused for what it is), the preflights (`Preflight`: installed, license, format, no other render)
    /// — all before any computation. Then the family's render runs — one line here, the rest in the family's folder — and
    /// each image it hands over is stamped with what the request, not the family, says of it.
    func renderBatchOnQueue(_ request: Request, seeds: [UInt64], model: Model, cancellation: Cancellation?,
                            onProgress: (@Sendable (Event) -> Void)?) throws -> [Render] {
        dispatchPrecondition(condition: .onQueue(Engine.file))
        guard !seeds.isEmpty else { return [] }
        try cancellation.check()
        let admitted = try Admission.judge(request, chain: model.chain)
        let lock = try Preflight.run(model, width: request.width, height: request.height) { onProgress?(.warning($0)) }
        defer { withExtendedLifetime(lock) {} }
        // What the render's stages held and what the Neural Engine computed of it: the engine's stages' own account,
        // stamped on each image as the family hands it over — the event is the only result a stream's reader gets.
        let stages = StageWatch()
        defer { stages.stop() }
        // A compilation is said as it begins and ends, not at the next step: the first render of a form waits on it.
        // The held ledger and the footprint's new buckets, told together; one thread at a time, so the buckets arrive in order.
        let telling = NSLock()
        @Sendable func tell(final: Bool = false) {
            telling.withLock {
                onProgress?(.held(stages.held().stages))
                if let memory = stages.memory(final: final) { onProgress?(.memory(memory)) }
            }
        }
        stages.changed = { tell() }
        return try CoreAI.under(request.precision) { _ in
            @Sendable func stamped(_ render: Render) -> Render {
                var render = render
                render.strength = Render.strength(of: request)
                render.loraOrigins = render.loras.map(Render.LoRAOrigin.init)
                (render.footprints.stages, render.footprints.neuralEngine) = stages.held()
                return render
            }
            let renders = try model.card.family.traits.render(request, admitted, seeds, model, cancellation) { event in
                switch event {
                case .image(let index, let render):
                    onProgress?(.image(index: index, stamped(render)))
                case .stage(let stage, _):
                    stages.begin(stage)
                    onProgress?(event)
                    tell()
                case .step, .text, .encoding, .decoding:
                    onProgress?(event)
                    tell()
                default:
                    onProgress?(event)
                }
            }
            // The open bucket and what the last result left: the series ends with the render.
            tell(final: true)
            return renders.map(stamped)
        }
    }
}

/// **The engine's stages as they told them** (`SiliCore.Stages`), from the watch's start to its stop: each one's plan
/// (P, its floor, B, its ring), the footprint at its most while it lived, its seconds, the swap-outs meanwhile, and the
/// Neural Engine's share its client said. Each is the render's stage under way when it opened (`begin`): the family
/// announces a stage (`Event.stage`), then opens its engine stages — no family names its stages to the app (rule 2).
/// The footprint's peak is read on a thread of its own (`FootprintWatch`), from a stage's opening to its close (from
/// the first's, when they overlap).
package final class StageWatch: @unchecked Sendable {
    package struct Entry: Sendable, Equatable {
        package let name: String
        /// The render's stage under way when it opened; `nil` outside a render.
        package let stage: Engine.Stage?
        package let p, floor, budget, ring: Int
        /// The groups of weights its declaration has.
        package var groups = 0
        /// The process's footprint at its most while it lived (so far, while it lives).
        package var peak = 0
        package var seconds = 0.0
        /// Pages swapped out while it lived (system-wide: the Mac's, `Memory.swapouts`).
        package var swapouts: UInt64 = 0
        package var neuralEngine: Double?
        /// The groups of weights it kept, from the start and in flight (those its plan kept, while it is open).
        package var kept = 0
        package var open = true
        /// Its block's cycles as they ran, and its ANE unit weighed for the next block (`SiliCore.Stages.Cycle`, `Weighed`).
        package var cycles: [SiliCore.Stages.Cycle] = []
        package var weighed: SiliCore.Stages.Weighed?
    }

    private let lock = NSLock()
    private let watch: FootprintWatch
    private let recorder = MemoryRecorder()
    private var observation: SiliCore.Stages.Observation?
    private var entries: [Entry] = []
    private var swapsAtOpen: [Int: UInt64] = [:], openedAt: [Int: UInt64] = [:]
    private var order: [Engine.Stage] = []
    /// What Core AI compiled under each of the render's stages, in seconds, and the compilations under way.
    private var compiled: [Engine.Stage: Double] = [:], compiling: [Engine.Stage: Int] = [:]
    /// Called when a compilation begins or ends, on the thread that loads the function: it must return at once.
    /// Under `lock`: set by the render's thread, read by the ticker and by the engine's observation.
    package var changed: (@Sendable () -> Void)? {
        get { lock.withLock { _changed } }
        set { lock.withLock { _changed = newValue } }
    }
    private var _changed: (@Sendable () -> Void)?
    /// Under `lock`: while a stage is open, `changed` is called four times a second (the chain's memory graph moves with
    /// the stage, not only at the render's events); none else is called more often.
    private var ticker: DispatchSourceTimer?

    private func tick(_ on: Bool) {
        if !on { ticker?.cancel(); ticker = nil; return }
        guard ticker == nil else { return }
        let t = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        t.schedule(deadline: .now() + .milliseconds(250), repeating: .milliseconds(250))
        t.setEventHandler { [weak self] in self?.changed?() }
        ticker = t
        t.resume()
    }

    /// The footprint is read by the process's one ticker, every millisecond; each reading is also the series' (`MemoryRecorder`).
    package init() {
        let recorder = self.recorder
        watch = FootprintWatch(reading: { let f = Memory.footprint(); recorder.record(f); return f })
        observation = SiliCore.Stages.observe { [weak self] in self?.hear($0) }
    }

    /// Stops hearing; what it heard stays readable. A stage still open (a decoder's, held until its client ends) is
    /// closed as it stands: its peak, its seconds and its swap-outs so far (19/11: closed with none, `memory` read the
    /// decoder at 0.00 GiB in 0.0 s).
    package func stop() {
        observation?.cancel()
        observation = nil
        changed = nil
        lock.withLock { tick(false) }
        let peak = watch.peak, swaps = Memory.swapouts(), now = DispatchTime.now().uptimeNanoseconds
        watch.stop()
        lock.withLock {
            for i in entries.indices where entries[i].open {
                entries[i].open = false
                entries[i].peak = max(entries[i].peak, peak)
                entries[i].seconds = Double(now &- (openedAt.removeValue(forKey: i) ?? now)) / 1e9
                entries[i].swapouts = swaps &- (swapsAtOpen.removeValue(forKey: i) ?? swaps)
            }
        }
    }

    deinit { if observation != nil { stop() } }

    /// A stage of the render begins: what opens from now on is its own.
    package func begin(_ stage: Engine.Stage) {
        recorder.mark(stage)
        lock.withLock { if order.last != stage { order.removeAll { $0 == stage }; order.append(stage) } }
    }

    /// **The footprint's buckets since the last call** (`MemoryRecorder.drain`); `final`: the render is over.
    package func memory(final: Bool = false) -> Engine.MemorySeries? { recorder.drain(final: final) }

    /// The engine stages heard so far, in the order they opened; an open one's peak as it stands.
    package var heard: [Entry] {
        let peak = watch.peak
        return lock.withLock { entries.map { var e = $0; if e.open { e.peak = max(e.peak, peak) }; return e } }
    }

    private func hear(_ e: SiliCore.Stages.Event) {
        switch e {
        case .opened(let plan):
            let first = lock.withLock { () -> Bool in
                let first = !entries.contains(where: \.open)
                entries.append(Entry(name: plan.declaration.name, stage: order.last, p: plan.p, floor: plan.floor,
                                     budget: plan.budget, ring: plan.depth, groups: plan.declaration.groups.count,
                                     kept: plan.kept.count))
                swapsAtOpen[entries.count - 1] = Memory.swapouts()
                openedAt[entries.count - 1] = DispatchTime.now().uptimeNanoseconds
                tick(true)
                return first
            }
            // A stage opens on the floor it was planned on: the peak is read from there, unless another is still open.
            if first { watch.reset() }
        case .closed(let plan, let seconds, let kept):
            let peak = watch.peak, swaps = Memory.swapouts()
            lock.withLock {
                guard let i = entries.lastIndex(where: { $0.open && $0.name == plan.declaration.name }) else { return }
                entries[i].open = false
                entries[i].peak = max(entries[i].peak, peak)
                entries[i].seconds = seconds
                entries[i].kept = kept
                entries[i].swapouts = swaps &- (swapsAtOpen.removeValue(forKey: i) ?? swaps)
                openedAt[i] = nil
                if !entries.contains(where: \.open) { tick(false) }
            }
        case .kept(let n):
            lock.withLock { if let i = entries.lastIndex(where: \.open) { entries[i].kept = max(entries[i].kept, n) } }
        case .neural(let share):
            lock.withLock { if let i = entries.lastIndex(where: \.open) ?? entries.indices.last { entries[i].neuralEngine = share } }
        case .cycle(let c):
            lock.withLock { if let i = entries.lastIndex(where: \.open) ?? entries.indices.last { entries[i].cycles.append(c) } }
        case .weighed(let w):
            lock.withLock { if let i = entries.lastIndex(where: \.open) ?? entries.indices.last { entries[i].weighed = w } }
        case .compiling:
            guard lock.withLock({ () -> Bool in
                guard let stage = order.last else { return false }
                compiling[stage, default: 0] += 1
                return true
            }) else { return }
            changed?()
        case .compiled(let seconds):
            guard lock.withLock({ () -> Bool in
                guard let stage = order.last else { return false }
                compiling[stage] = max(0, compiling[stage, default: 0] - 1)
                compiled[stage, default: 0] += seconds
                return true
            }) else { return }
            changed?()
        }
    }

    /// **What each stage of the render held so far**, in their order (`Engine.Held`), and the render's Neural Engine
    /// share: each engine stage's, weighted by the seconds it lived.
    package func held() -> (stages: [Engine.Held], neuralEngine: Double?) {
        let all = heard, (stages, compiled, compiling) = lock.withLock { (order, compiled, compiling) }
        var lived = 0.0, shared = 0.0
        let held = stages.compactMap { stage -> Engine.Held? in
            let mine = all.filter { $0.stage == stage }
            guard !mine.isEmpty || compiled[stage] != nil || compiling[stage] != nil else { return nil }
            var h = Engine.Held(stage: stage)
            h.compilation = compiled[stage] ?? 0
            h.compiling = compiling[stage, default: 0] > 0
            for e in mine {
                h.counted = max(h.counted, e.p - e.floor)
                h.held = max(h.held, e.peak - e.floor)
                h.budget = max(h.budget, e.budget)
                h.process = max(h.process, e.peak)
                h.ring = max(h.ring, e.ring)
                if let n = e.neuralEngine { h.neuralEngine = n }
                if !e.open { lived += e.seconds; shared += e.seconds * (e.neuralEngine ?? 0) }
            }
            if let largest = mine.max(by: { $0.groups < $1.groups }) { (h.groups, h.kept) = (largest.groups, largest.kept) }
            return h
        }
        return (held, lived > 0 && shared > 0 ? shared / lived : nil)
    }
}
