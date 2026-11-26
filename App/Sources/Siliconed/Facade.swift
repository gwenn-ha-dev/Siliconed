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

    /// **What a render of `request` on `model` would specialize**, its assets written (`FamilyTraits.prepare`): the
    /// request's own refusals first (`Admission`), then the family's stages at the shapes its render opens them, nothing
    /// allocated, nothing computed; the ANE's functions first, the longest to compile, each stage's order kept. On the
    /// engine queue, as a render: no asset is written while a render loads it.
    package func prepare(_ request: Request, model: Model) throws(EngineError) -> [Preparation] {
        dispatchPrecondition(condition: .notOnQueue(Engine.file))
        let all = try EngineError.boundary {
            try Engine.file.sync {
                let admitted = try Admission.judge(request, chain: model.chain)
                return try CoreAI.under { try model.card.family.traits.prepare(request, admitted, model) }
            }
        }
        return all.filter { $0.unit == .neuralEngine } + all.filter { $0.unit != .neuralEngine }
    }

    /// **The render, on the engine queue**, in this order: the request's own refusals (`Admission`, pure: a request the model
    /// cannot render is refused for what it is), the preflights (`Preflight`: installed, license, format, no other render,
    /// the shared files hashed once) — all before any computation. Then the family's render runs — one line here, the rest in the family's folder — and
    /// each image it hands over is stamped with what the request, not the family, says of it.
    func renderBatchOnQueue(_ request: Request, seeds: [UInt64], model: Model, cancellation: Cancellation?,
                            onProgress: (@Sendable (Event) -> Void)?) throws -> [Render] {
        dispatchPrecondition(condition: .onQueue(Engine.file))
        guard !seeds.isEmpty else { return [] }
        try cancellation.check()
        let admitted = try Admission.judge(request, chain: model.chain)
        let lock = try Preflight.run(model, width: request.width, height: request.height, cancellation: cancellation) { onProgress?(.warning($0)) }
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
        return try CoreAI.under {
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
/// Neural Engine's share its client said. Each is the render's stage under way at its first call (`begin`,
/// `Stages.Event.calling`), and until then at its opening: the family announces a stage (`Event.stage`), then calls its
/// engine stages — opened before, beside the one that runs (`Stage.beside`: the DiT during the text, the decoder during
/// the DiT's last evaluation), or then — no family names its stages to the app (rule 2). What a stage tells is told
/// under its name: two may be open.
/// The footprint's peak is read on a thread of its own (`FootprintWatch`), from a stage's opening to its close (from
/// the first's, when they overlap).
package final class StageWatch: @unchecked Sendable {
    package struct Entry: Sendable, Equatable {
        package let name: String
        /// The render's stage under way at its first call, until then at its opening; `nil` outside a render.
        package var stage: Engine.Stage?
        /// Whether it made a call (`Stages.Event.calling`): its stage is the one under way then.
        package var called = false
        package let p, floor, budget, ring: Int
        /// The groups of weights its declaration has.
        package var groups = 0
        /// The process's footprint at its most while it lived (so far, while it lives).
        package var peak = 0
        package var seconds = 0.0
        /// Pages swapped out while it lived (system-wide: the Mac's, `Memory.swapouts`).
        package var swapouts: UInt64 = 0
        /// What Core AI compiled for its functions, in seconds, and its compilations under way: counted on it from their
        /// start to their end, said under the render's stage it is ranked in (`stage`).
        package var compilation = 0.0, compiling = 0
        package var neuralEngine: Double?
        /// The groups of weights it kept, from the start and in flight (those its plan kept, while it is open), and their
        /// bytes with the residents' and the ring's (`Stage.Plan.weights`, then `Stages.Event.kept`).
        package var kept = 0
        package var weights = 0
        package var open = true
        /// Its block's cycles as they ran (`SiliCore.Stages.Cycle`).
        package var cycles: [SiliCore.Stages.Cycle] = []
        /// What its calls executed, over their spans (`SiliCore.Stages.Work`).
        package var works: [SiliCore.Stages.Work] = []
        /// What the trace recorded (`SiliCore.Stages.Trace`): its opening and close from their starts, piece by piece;
        /// each cycle's occupancy by its ordinal; its runs of calls outside the cycles.
        package var openingStart: UInt64?, closingStart: UInt64?
        package var opening: [SiliCore.Stages.Piece] = [], closing: [SiliCore.Stages.Piece] = []
        package var traced: [Int: SiliCore.Stages.Occupancy] = [:]
        package var runs: [SiliCore.Stages.Occupancy] = []
    }

    private let lock = NSLock()
    private let watch: FootprintWatch
    private let recorder = MemoryRecorder()
    private var observation: SiliCore.Stages.Observation?
    private var entries: [Entry] = []
    private var swapsAtOpen: [Int: UInt64] = [:], openedAt: [Int: UInt64] = [:]
    private var order: [Engine.Stage] = []
    /// What Core AI compiled for no engine stage heard (`Stages.Event.compiling` naming none), under the render's stage
    /// under way, in seconds, and the compilations under way.
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

    /// Under `lock`: **what the open stages hold, told to the series** (`MemoryRecorder.level`) — the floor the first of
    /// them opened on, the weights of all; none open, the footprint is the process's.
    private func leveled() {
        let open = entries.filter(\.open)
        recorder.level(floor: open.first?.floor, weights: open.reduce(0) { $0 + $1.weights })
    }

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
                                     kept: plan.kept.count, weights: plan.weights))
                swapsAtOpen[entries.count - 1] = Memory.swapouts()
                openedAt[entries.count - 1] = DispatchTime.now().uptimeNanoseconds
                tick(true)
                leveled()
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
                leveled()
            }
        case .kept(let stage, let n, let weights):
            lock.withLock {
                if let i = entry(stage) { entries[i].kept = max(entries[i].kept, n); entries[i].weights = weights }
                leveled()
            }
        case .neural(let stage, let share):
            lock.withLock { if let i = entry(stage) { entries[i].neuralEngine = share } }
        case .calling(let stage):
            lock.withLock {
                if let i = entry(stage), !entries[i].called { (entries[i].called, entries[i].stage) = (true, order.last ?? entries[i].stage) }
            }
        case .cycle(let c):
            lock.withLock { if let i = entry(c.stage) { entries[i].cycles.append(c) } }
        case .work(let w):
            lock.withLock { if let i = entry(w.stage) { entries[i].works.append(w) } }
        case .traced(let t):
            lock.withLock {
                switch t {
                case .opening(let stage, let start, let pieces):
                    if let i = entry(stage) { (entries[i].openingStart, entries[i].opening) = (start, pieces) }
                case .closing(let stage, let start, let pieces):
                    if let i = entry(stage) { (entries[i].closingStart, entries[i].closing) = (start, pieces) }
                case .calls(let stage, let cycle, let o):
                    guard let i = entry(stage) else { return }
                    if let cycle { entries[i].traced[cycle] = o } else { entries[i].runs.append(o) }
                }
            }
        // A compilation counted on the engine stage that loads: a DiT opening beside the text may end its compilation
        // after the text has closed, its end meeting its start.
        case .compiling(let name):
            guard lock.withLock({ () -> Bool in
                if let name, let i = entry(name) { entries[i].compiling += 1; return true }
                guard let stage = order.last else { return false }
                compiling[stage, default: 0] += 1
                return true
            }) else { return }
            changed?()
        case .compiled(let name, let seconds):
            guard lock.withLock({ () -> Bool in
                if let name, let i = entry(name) {
                    entries[i].compiling = max(0, entries[i].compiling - 1)
                    entries[i].compilation += seconds
                    return true
                }
                guard let stage = order.last else { return false }
                compiling[stage] = max(0, compiling[stage, default: 0] - 1)
                compiled[stage, default: 0] += seconds
                return true
            }) else { return }
            changed?()
        }
    }

    /// Under `lock`: the engine stage named `name`, the last open one, else the last one.
    private func entry(_ name: String) -> Int? {
        entries.lastIndex { $0.open && $0.name == name } ?? entries.lastIndex { $0.name == name }
    }

    /// An occupancy as a render tells it (`Engine.Occupancy`): each step's TFLOP/s over its calls' seconds.
    static func occupancy(_ o: SiliCore.Stages.Occupancy) -> Engine.Occupancy {
        func idle(_ i: SiliCore.Stages.Occupancy.Idle) -> Engine.Occupancy.Idle {
            var x = Engine.Occupancy.Idle()
            (x.acrossAlone, x.atGroups, x.between, x.outside) = (i.acrossAlone, i.atGroups, i.between, i.outside)
            return x
        }
        var x = Engine.Occupancy(seconds: o.seconds, ane: o.ane, gpu: o.gpu)
        (x.aneIdle, x.gpuIdle, x.unread) = (idle(o.aneIdle), idle(o.gpuIdle), o.unread)
        x.steps = o.steps.map { s in
            Engine.Occupancy.Step(unit: s.unit, name: s.name, calls: s.calls, rows: s.rows, seconds: s.seconds,
                                  tflops: s.seconds > 0 ? Double(s.flops) / s.seconds / 1e12 : 0)
        }
        return x
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
            h.compilation = mine.reduce(compiled[stage] ?? 0) { $0 + $1.compilation }
            h.compiling = compiling[stage, default: 0] > 0 || mine.contains { $0.compiling > 0 }
            for e in mine {
                h.counted = max(h.counted, e.p - e.floor)
                h.held = max(h.held, e.peak - e.floor)
                h.budget = max(h.budget, e.budget)
                h.process = max(h.process, e.peak)
                h.ring = max(h.ring, e.ring)
                if let n = e.neuralEngine { h.neuralEngine = n }
                if !e.open { lived += e.seconds; shared += e.seconds * (e.neuralEngine ?? 0) }
                // Its block's cycles as they ran, each with its rates: the last block of the stage that ran one.
                if !e.cycles.isEmpty {
                    h.cycles = e.cycles.enumerated().map { n, c in
                        let r = c.rates, total = c.flops.ane + c.flops.gpu
                        return Engine.Cycle(rows: c.rows, shareable: c.shareable, unit: c.unit, zero: c.trying, seconds: c.seconds,
                                            tflops: r.block, neuralEngine: r.ane, gpu: r.gpu, end: recorder.seconds(c.end),
                                            aneFraction: total > 0 ? Double(c.flops.ane) / Double(total) : nil,
                                            occupancy: e.traced[n].map(Self.occupancy))
                    }
                }
                h.work += e.works.map { w in
                    let total = w.flops.ane + w.flops.gpu
                    return Engine.Work(tflops: w.rate, seconds: w.seconds, end: recorder.seconds(w.end),
                                       aneFraction: total > 0 ? Double(w.flops.ane) / Double(total) : 0)
                }
                if e.openingStart != nil || e.closingStart != nil || !e.runs.isEmpty {
                    var t = Engine.StageTrace(name: e.name)
                    (t.opened, t.closed) = (e.openingStart.map(recorder.seconds), e.closingStart.map(recorder.seconds))
                    t.opening = e.opening.map { Engine.StageTrace.Piece(name: $0.name, seconds: $0.seconds) }
                    t.closing = e.closing.map { Engine.StageTrace.Piece(name: $0.name, seconds: $0.seconds) }
                    t.runs = e.runs.map(Self.occupancy)
                    h.traces.append(t)
                }
            }
            if let largest = mine.max(by: { $0.groups < $1.groups }) { (h.groups, h.kept) = (largest.groups, largest.kept) }
            return h
        }
        return (held, lived > 0 && shared > 0 ? shared / lived : nil)
    }
}
