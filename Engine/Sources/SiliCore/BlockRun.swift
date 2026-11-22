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
        nonisolated(unsafe) let stream = stage.stream!
        let timing = Timing(before: last, encoded: DispatchTime.now().uptimeNanoseconds,
                            done: Task { await stream.currentWorkCompleted(); return DispatchTime.now().uptimeNanoseconds })
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
            await settled()
            let io = views(f, start: row, group: g, pass: nil, step: nil, call: bind, neural: true, written: &written)
            try await Run.neural(functions[f]!, inputs: io.inputs, outputs: io.outputs)
            return
        }
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
            timed(pass)
            let climbed = try await climb()
            var moved: Double?
            if case .inFlight = share, units > 0 { moved = try await rebalance() }
            cycled(until: at)
            begun = (at, units * unit, moved, climbed)
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
                record.append(Timed(kind: .neural, step: l.step, phase: Self.phase(form, l.index), rows: ane, seconds: seconds))
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
                    for u in tiles.indices {
                        let previous = launched.last?.task, c = calls[u][s.name]!
                        let function = c.function
                        nonisolated(unsafe) let io = c.io
                        let task = Task { () async throws -> Double in
                            _ = try await previous?.value
                            _ = await gpu?.value
                            let t = DispatchTime.now().uptimeNanoseconds
                            try await Run.neural(function, inputs: io.inputs, outputs: io.outputs)
                            return Functions.seconds(since: t)
                        }
                        launched.append((s.name, i, task))
                    }
                } else if s.unit == .gpu {
                    try await drain()
                }
                for t in Self.calls(form, s, p, neural: neural, cut: cut, limit: p.limit ?? declaration.rows) {
                    let f = Self.gpu(g.form, s.name, t.rows, tapped: tapped && s.taps)
                    let timing = try await encode(f, views(f, start: t.start, group: g, pass: p, step: s, call: [:], neural: false, written: &written))
                    if shared, case .inFlight = share {
                        record.append(Timed(kind: s.unit == .either ? .past : .fixed, step: s.name, phase: Self.phase(form, i), rows: t.rows, timing: timing))
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
}
