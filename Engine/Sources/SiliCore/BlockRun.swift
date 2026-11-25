import CoreAI
import Foundation
import Metal

/// **The calls**: each put on the stream ahead of the GPU (`encode`), an edge called, a group run over a pass, the
/// stage's signal.
extension Block {
    /// One GPU call put on the stream once the function's last is done: the task done when this one is, and its time.
    private func encode(_ f: String, _ io: (inputs: [String: View], outputs: [(String, View)])) async throws -> Timing {
        if let m = marks[f] { _ = await m.value }
        // A buffer Core AI sees for the first time (a part of a surface made for this call, `Arena.part`) is read out of
        // the stream's order: the first pass of a block shared with the ANE read its GPU rows of a stream an edge had just
        // written as zeros, the second the right values; the stream waited for first, 1.5e-4 of the GPU alone's, f16's
        // rounding (21/11 on the mini, `aSharedBlockIsTheGPUAlones`). Once a buffer, a few at a block's opening.
        let buffers = (io.inputs.values.map { $0 } + io.outputs.map(\.1)).map { ObjectIdentifier($0.buffer) }
        if !buffers.allSatisfy(seen.contains) {
            await settled()
            seen.formUnion(buffers)
        }
        // An argument from off a page (a factor's rows, a value each) is copied by Core AI outside the stream's order: put
        // ahead, the call read it before the call before wrote it (measured 19/11 on the M5 Pro, a transformer's f16
        // outputs apart; the same bits once such a call waits for the stream).
        if (io.inputs.values.map { $0 } + io.outputs.map(\.1)).contains(where: { $0.offset % Arena.page != 0 }) { await settled() }
        let left = try Run.encode(functions[f]!, stage.stream, inputs: io.inputs, outputs: io.outputs)
        precondition(left.isEmpty, "\(f): outputs no view holds, \(left.keys.sorted())")
        let timing = Timing(before: last, encoded: DispatchTime.now().uptimeNanoseconds, done: stage.stream.done())
        marks[f] = timing.done
        last = timing.done
        // And its result from off a page is copied back the same way, after the call: the next call reads it once it is, or
        // reads what the call before wrote (measured 19/11 on the mini, a shared form beside the ANE at 512², the rows past
        // the ANE's of a factor a row each, from byte 1024: the attention's gate and keys were the previous block's in
        // one run of two; with this wait, 6 in 6, the same bits and time on a transformer's).
        if io.outputs.contains(where: { $0.1.offset % Arena.page != 0 }) { await settled() }
        return timing
    }

    /// A GPU call's time once done: from its encoding or the call before's end, the later, to its end.
    struct Timing {
        let before: Task<UInt64, Never>?
        let encoded: UInt64
        let done: Task<UInt64, Never>
        func seconds() async -> Double {
            let end = await done.value, start = max(encoded, await before?.value ?? 0)
            return Double(end > start ? end - start : 0) / 1e9
        }
    }

    /// Every call put on the stream, done.
    public func settled() async { _ = await last?.value }

    /// **Edge `name`'s call** (`Declaration.edges`): its per-row arguments from row `row`, the group a run holds its
    /// weights' and its own tensors' (`run`'s `before`), or `group`'s held weights and its own tensors outside a run
    /// (its other weights are not held). An edge of the ANE's (`neuralEdges`) runs once every call on the stream is
    /// done, and is done when this returns.
    public func call(_ name: String, group: String? = nil, from row: Int = 0, _ bind: [String: Bind] = [:]) async throws {
        var written = Set<String>()
        let f = Self.edge(name), g = group.map { groups[$0]! } ?? held?.group
        if declaration.neuralEdges.contains(name) {
            flops.ane += signatures[f]?.flops ?? 0
            await settled()
            let io = views(f, start: row, group: g, pass: nil, step: nil, call: bind, neural: true, written: &written)
            try await Run.neural(functions[f]!, inputs: io.inputs, outputs: io.outputs)
            return
        }
        flops.gpu += signatures[f]?.flops ?? 0
        _ = try await encode(f, views(f, start: row, group: g, pass: nil, step: nil, call: bind, neural: false, written: &written))
    }

    /// **Group `group` over pass `pass`**: its weights acquired (kept, or the ring's next use), `before` (its edges: a
    /// modulation), then its form's steps in turn — a step of either unit the ANE's units of the shareable rows (each
    /// call after the ANE's call before it, and after the GPU's calls before it when the step before was the GPU's) and
    /// one GPU call past them; a step of the GPU's one call over the pass, every ANE call before it done; its weights
    /// released; the streams written swapped. `until`: the steps up to this one only (a pass whose last steps reach nothing:
    /// the keys of a prefix's last layer); a stream no step wrote is not swapped. `tapped` (the declaration's `tapped`): the
    /// ANE left, the GPU alone over every row, each step that taps in its tapped form — a check listening inside the group.
    public func run(_ group: String, _ pass: String, until: String? = nil, before: (() async throws -> Void)? = nil,
                    tapped: Bool = false) async throws {
        precondition(!tapped || declaration.tapped, "\(group): tapped, in a block not declared tapped")
        guard let g = groups[group], let p = passes[pass], let form = forms[g.form], p.serves(form) else { preconditionFailure("\(group) over \(pass)") }
        precondition(until == nil || form.steps.contains { $0.name == until }, "\(group): no step \(until!)")
        if group == declaration.cycle.first {
            let at = DispatchTime.now().uptimeNanoseconds
            // The cycle before's own seconds, what its start loaded left out; warm, its functions the ones the cycle
            // before it called (nothing moved nor climbed at its start, not the first).
            let last = begun.map { Double(at &- $0.at) * 1e-9 - ($0.moved ?? 0) - ($0.climbed ?? 0) }
            let warm = begun.map { $0.moved == nil && $0.climbed == nil && !$0.first } ?? false
            let first = begun == nil
            let climbed = try await climb()
            var moved: Double?
            if case .inFlight = share, units > 0 || zero != nil { moved = try await rebalance(last: last, warm: warm) }
            cycled(until: at)
            begun = (at, units * unit, unit, trial != nil || zero != nil, moved, climbed, first, zeroRefused)
            zeroRefused = false
        }
        let t0 = DispatchTime.now().uptimeNanoseconds
        let weights = try stage.acquire(group)
        waited += Functions.seconds(since: t0)
        held = (g, weights)
        defer { held = nil }
        let shared = !tapped && form.shared && p.shareable != nil && units > 0, neural = shared ? units * unit : 0
        let ane = unit
        // The share is sized on the block's largest shareable pass: a smaller pass would give the ANE rows it has not.
        precondition(!shared || neural <= p.shareable!.count, "\(pass): the ANE's \(neural) rows past its \(p.shareable!.count) shareable")
        let tiles = shared ? (0..<units).map { Tile(start: p.rows.lowerBound + $0 * ane, rows: ane) } : []
        let run = Self.steps(of: form, until: until)
        let copying = shared && !copies.isEmpty
        let reads = copying ? stretches(g, p, run, until: until) : []
        var copied = copying ? prepare(weights, reads[0]) : nil, stretch = 0
        var written = Set<String>()
        // The ANE's calls, made before the pass, for the steps it runs: a step's views name the tensors it writes, whose
        // two buffers turn once the pass is over (`current`). A call beside the GPU's reads nothing of the loop's state.
        let made = tiles.map { u in
            Dictionary(uniqueKeysWithValues: run.map(\.element).filter { $0.unit == .either }.map { s in
                let f = Self.neural(g.form, s.name, ane)
                return (s.name, (function: functions[f]!, io: views(f, start: u.start, group: g, pass: p, step: s, call: [:], neural: true, written: &written)))
            })
        }
        nonisolated(unsafe) let calls = made
        var record: [Timed] = []
        var launched: [(step: String, index: Int, task: Task<Double, Error>)] = []
        func drain() async throws {
            for l in launched {
                let seconds = try await l.task.value
                record.append(Timed(kind: .neural, step: l.step, phase: Self.phase(form, l.index), rows: ane, seconds: seconds,
                                    flops: signatures[Self.neural(g.form, l.step, ane)]?.flops ?? 0))
            }
            launched = []
        }
        do {
            try await before?()
            var afterGPU = true
            for (i, s) in run {
                if s.unit == .either && shared {
                    // What the ANE reads of the GPU's: every GPU call before, when the step before was the GPU's.
                    let gpu = afterGPU ? last : nil
                    flops.ane += tiles.count * (signatures[Self.neural(g.form, s.name, ane)]?.flops ?? 0)
                    for u in tiles.indices {
                        let previous = launched.last?.task, c = calls[u][s.name]!
                        let function = c.function, ready = copied
                        nonisolated(unsafe) let io = c.io
                        let task = Task { () async throws -> Double in
                            _ = try await previous?.value
                            _ = await gpu?.value
                            _ = await ready?.value
                            let t = DispatchTime.now().uptimeNanoseconds
                            try await Run.neural(function, inputs: io.inputs, outputs: io.outputs)
                            return Functions.seconds(since: t)
                        }
                        launched.append((s.name, i, task))
                    }
                } else if s.unit == .gpu {
                    let idle = !launched.isEmpty
                    try await drain()
                    // The ANE has nothing in flight until the GPU's steps are done: the codes of its next stretch copied now.
                    if copying && idle { copied = copy(after: stretch, weights, reads); stretch += 1 }
                }
                for t in Self.calls(form, s, p, neural: neural, cut: cut, limit: p.limit ?? declaration.rows) {
                    let f = Self.gpu(g.form, s.name, t.rows, tapped: tapped && s.taps)
                    flops.gpu += signatures[f]?.flops ?? 0
                    let timing = try await encode(f, views(f, start: t.start, group: g, pass: p, step: s, call: [:], neural: false, written: &written))
                    if shared, case .inFlight = share {
                        record.append(Timed(kind: s.unit == .either ? .past : .fixed, step: s.name, phase: Self.phase(form, i), rows: t.rows, timing: timing,
                                            flops: signatures[f]?.flops ?? 0))
                    }
                }
                afterGPU = s.unit == .gpu
            }
            try await drain()
        } catch {
            // Nothing left running on the ANE's rows past the pass; the group's weights given back as a pass gives them,
            // so that the ring's order of use holds for a caller that goes on.
            for l in launched { _ = try? await l.task.value }
            if weights.use != nil { await settled() }
            stage.release(weights)
            throw error
        }
        if shared, case .inFlight = share { cycleRecords.append(record) }
        // A slot of the ring is read again once released: what reads it done first.
        if weights.use != nil { await settled() }
        stage.release(weights)
        for s in written { current[s] = 1 - current[s]! }
        await passed(group, pass)
    }

    /// **The stage's signal**, given once the second group in a row over a pass is done (a turn of the loop begins anew
    /// at its first group): each function of the pass called twice, every call done (`Stage.signal`).
    private func passed(_ group: String, _ name: String) async {
        if name != pass || group == declaration.cycle.first || group == declaration.once.first { (pass, inPass) = (name, 0) }
        inPass += 1
        if inPass == 2 { await signal() }
    }

    /// Every call done, the stage's signal given (`Stage.signal`).
    func signal() async {
        await settled()
        stage.signal()
    }

    // MARK: - The ANE's copies of the codes

    /// The name of the ANE's copy of the codes `w` (`places`).
    static func copy(_ w: String) -> String { "ane.\(w)" }

    /// **The ANE's copy of the codes, one each, written while the ANE is idle**: the ANE reads codes `[in, out]` only
    /// (`[out, in]` transposed at every call, a fixed cost, 0.54 to 0.87 × the rate on the M1 Pro), the GPU's int8 × int8
    /// `[out, in]` only; the codes lie the GPU's way, by pieces `[p, out, in / p]`, and the CPU writes the codes a stretch
    /// of the ANE's steps reads `[1, in, out]` into their copy (`Weights.transposedPieces`, on every core: a block of 434 MB
    /// of codes, the widest of today's, ≈ 7 ms on the M5 Pro, 10 to 35 ms on the M1 Pro). A stretch's codes are written at
    /// the GPU step before it, every ANE call before it done (the heads and the attention of every shared form today, the
    /// GPU alone); the first stretch's at the last such step of the turn before, its copies read already. Two copies a
    /// code, one read while the other was written, cost the ANE 30 %: on the M1 Pro, a block of 28 groups by 4096 rows, its
    /// 2560 rows 11.3 to 12.8 s a step whether the other copy was written beside its calls or beside the attention, 9.0
    /// to 9.3 with one copy written before its calls, the GPU beside 10.6 every time (22/11/2026).
    ///
    /// This run's first stretch: its codes copied ahead, or copied now; a task done once they are.
    private func prepare(_ held: Stage.Held, _ first: Set<String>) -> Task<Void, Never> {
        let made = ahead?.turn == held.turn ? ahead!.names : []
        ahead = nil
        enqueue(held.weights, first.subtracting(made))
        return marker()
    }

    /// **Past stretch `k`, the ANE idle**: stretch k + 1's codes copied, a task done once they are; past the last but one,
    /// the next turn's first stretch's behind them, read already by this run's first (`Stage.upcoming`, `Ring.peek`: a
    /// slot of the ring is not read again before its turn is released).
    private func copy(after k: Int, _ held: Stage.Held, _ reads: [Set<String>]) -> Task<Void, Never> {
        precondition(k + 1 < reads.count, "the ANE idle past its \(reads.count) stretches")
        enqueue(held.weights, reads[k + 1])
        let done = marker()
        guard k + 2 == reads.count else { return done }
        let next = held.turn + 1
        let names = reads[0].subtracting(reads.dropFirst().reduce(into: Set()) { $0.formUnion($1) })
        // Only into a group of the same form: another form's codes bear other names, and its run copies its own.
        guard !names.isEmpty, let up = stage.ring?.group(next), groups[up]?.form == groups[held.group]!.form else { return done }
        ahead = (next, names)
        let stage = self.stage
        copier.async {
            guard let up = stage.upcoming(next) else { return }
            Self.write(stage, up.weights, names)
        }
        return done
    }

    /// **The codes each stretch of a run's ANE steps reads**: a stretch ends at a GPU step past an ANE step, where the
    /// run waits for the ANE's calls (`drain`).
    private func stretches(_ g: Group, _ p: Pass, _ run: [(offset: Int, element: Step)], until: String?) -> [Set<String>] {
        let key = "\(g.form) \(p.name) \(until ?? "")"
        if let s = stretches[key] { return s }
        var all: [Set<String>] = [[]], ane = false
        for (_, s) in run {
            guard s.unit == .either else {
                if ane { all.append([]); ane = false }
                continue
            }
            ane = true
            let sig = signatures[Self.neural(g.form, s.name, unit)]!, outputs = Set(sig.outputs.map(\.0))
            for (n, _, _) in sig.inputs {
                if case .weight(let w) = resolver.resolve(n, input: true, shadowed: outputs.contains(n), group: g, pass: p, step: s, call: [:]),
                   copies.contains(w) { all[all.count - 1].insert(w) }
            }
        }
        stretches[key] = all
        return all
    }

    /// A task done once every copy put on the copier so far is.
    private func marker() -> Task<Void, Never> {
        let m = DispatchWorkItem {}
        copier.async(execute: m)
        return Task { await withCheckedContinuation { c in m.notify(queue: .global(qos: .userInitiated)) { c.resume() } } }
    }

    /// The copy of `names` of `weights`, on the copier.
    private func enqueue(_ weights: [String: Placed], _ names: Set<String>) {
        guard !names.isEmpty else { return }
        let stage = self.stage
        copier.async { Self.write(stage, weights, names) }
    }

    /// Each code of `names`, `[p, out, in / p]`, written `[1, in, out]` into its copy. Of the stage, never of the
    /// block: a copy on the copier holding the block's last reference would run its `deinit` there, whose `close`
    /// waits for the copier (`copier.sync`) on the copier itself, and the process stops. The stage outlives every
    /// copy put on the copier: the block's `close` waits for them before it closes the stage.
    private static func write(_ stage: Stage, _ weights: [String: Placed], _ names: Set<String>) {
        for name in names.sorted() {
            guard let p = weights[name] ?? stage.residents[name] else { preconditionFailure("\(name): no codes to copy") }
            precondition(p.type == .int8 && p.shape.count == 3, "\(name): \(p.shape) \(p.type), not codes by pieces")
            let from = UnsafePointer(stage.arena.pointer(p.offset).assumingMemoryBound(to: Int8.self))
            Weights.transposedPieces(from, count: p.shape[0], out: p.shape[1], piece: p.shape[2],
                                     into: stage.pointer(Self.copy(name)).assumingMemoryBound(to: Int8.self))
        }
    }
}
