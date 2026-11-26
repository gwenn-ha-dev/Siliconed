import CoreAI
import Foundation

/// **The ANE's share followed in flight**: the calls' height climbed, the share moved to the one predicted to finish
/// first where it pays its move, the share of none among them (visited where predicted the fastest, or to measure κ).
extension Block {
    /// **The cycles its client says are left**, said before each, that one included (its loop's calls, and those it will
    /// run after): what a move at that cycle's start is weighed against (`pays`). Never said, a move must pay itself back
    /// within one cycle.
    public func expect(cycles n: Int) { left = max(n, 0) }

    /// **The calls climbed in flight**, at a turn's first group once a whole turn ran at this height: Core AI's share read
    /// now, the tallest height above whose estimate — its own reading, or a bound from its kin's, this one's share now
    /// among them (`estimated`) — holds under B beside what is kept (`Stage.rekey`) taken, its functions loaded (built and
    /// specialized where first met) and this one's let go, this height's reading kept and the stage's reading kept under
    /// the new one's from now on (`rekey`). A height the fuse stopped or with no bound is never taken. The seconds its
    /// loading took; nil, nothing climbed.
    func climb() async throws -> Double? {
        turns += 1
        guard turns == 2, height > 0 else { return nil }
        let share = max(stage.observe(), 0)
        // The forms at the share of the ANE now (a move since the opening changed the functions).
        let forms = try heights.map { try form(units: units, cut: $0.cut) }
        let known = estimated(forms, share: share, now: forms[height])
        for i in 0..<height {
            guard let estimate = known[i].bytes,
                  stage.rekey(forms[i].key, expects: estimate - share, extent: forms[i].extent) else { continue }
            (height, turns) = (i, 1)
            let t = DispatchTime.now().uptimeNanoseconds
            _ = try await stage.loading { try await load() }
            return Functions.seconds(since: t)
        }
        return nil
    }

    /// The steps `run` takes, with their places in the form: all, or up to the one named `until`.
    static func steps(of form: Form, until: String?) -> [(offset: Int, element: Step)] {
        let all = Array(form.steps.enumerated())
        guard let until, let last = form.steps.firstIndex(where: { $0.name == until }) else { return all }
        return Array(all[...last])
    }

    /// **A call of a pass beside the ANE, timed**: the ANE's (a unit); the GPU's past the ANE's rows, in a phase beside
    /// it (`phase`); the GPU's of a `gpu` step.
    struct Timed {
        enum Kind { case neural, past, fixed }
        var kind: Kind
        var step: String
        var phase: Int
        var rows: Int
        var seconds = 0.0
        var timing: Timing? = nil
        /// What the call executed (`Signature.flops`).
        var flops = 0
        /// The first call of a `gpu` step in two (`Block.first`): the rows of the ANE's first unit.
        var first = false

        /// A group's calls summed by step and unit, in their order: what `predicted` reads of them, whatever the count of
        /// calls a share makes. A step's first call apart.
        static func summed(_ record: [Timed]) -> [Timed] {
            var order: [String] = [], by: [String: Timed] = [:]
            for t in record {
                let k = "\(t.kind) \(t.step)\(t.first ? " first" : "")"
                if var s = by[k] { (s.rows, s.seconds, s.flops) = (s.rows + t.rows, s.seconds + t.seconds, s.flops + t.flops); by[k] = s }
                else { order.append(k); by[k] = Timed(kind: t.kind, step: t.step, phase: t.phase, rows: t.rows, seconds: t.seconds, flops: t.flops, first: t.first) }
            }
            return order.map { by[$0]! }
        }
    }

    /// **The share in flight moved**, at a cycle's first group: the cycle before's calls timed and smoothed into the
    /// block's (`smooth`); the share `predicted` to finish first among those admitted under B (`bestShare`, `admitted`)
    /// taken where it gains more than the noise between two warm cycles, and its gain over the cycles left more than its
    /// move costs and the noise (`pays`); its rows kept on this Mac, the GPU's functions at the new rows loaded (built and
    /// specialized where first met) and the others let go. A cold cycle moves no share: its calls carry the
    /// specialization. `last`: the cycle before's own seconds.
    func rebalance(last: Double?, warm: Bool) async throws -> Double? {
        defer { cycleRecords = [] }
        observe(last: last, warm: warm)
        // At 0 rows: the visit goes on (its cycles' calls are not timed).
        if let z = zero, let last { return try await probe(z, last: last) }
        var records: [[Timed]] = []
        for r in cycleRecords {
            var record = r
            for j in record.indices { if let t = record[j].timing { record[j].seconds = await t.seconds() } }
            records.append(record)
        }
        guard !records.isEmpty else { return nil }
        var sums = (ane: 0.0, beside: 0.0, alone: 0.0)
        for t in records.joined() {
            switch t.kind {
            case .neural: sums.ane += t.seconds
            case .past: sums.beside += t.seconds
            case .fixed: sums.alone += t.seconds
            }
        }
        var steps: [String: Stages.Calls.Step] = [:]
        for t in records.joined() {
            let unit = switch t.kind { case .neural: "ane"; case .past: "beside"; case .fixed: "alone" }
            steps["\(unit) \(t.step)", default: .init(name: t.step, unit: unit)].add(rows: t.rows, seconds: t.seconds, flops: t.flops)
        }
        lastCalls = Stages.Calls(ane: sums.ane, beside: sums.beside, alone: sums.alone,
                                 steps: steps.values.sorted { ($0.unit, $0.name) < ($1.unit, $1.name) })
        let a0 = units * unit
        // A cold cycle's calls carry their specialization: neither smoothed nor weighed.
        guard warm else { return nil }
        smooth(records, at: a0)
        try admit()
        let kappa = alone ?? 1
        func predicted(_ u: Int) -> Double { Self.predicted(smoothed, measured: a0, rows: u * unit, unit: unit, alone: kappa) }
        let best = Self.bestShare(smoothed, measured: a0, unit: unit, shareable: shareable, admitted: admitted ?? [], alone: kappa).rows / unit
        // The share of none, visited where the curve predicts it the fastest and the move pays — κ unknown, at its bound,
        // the visit measuring it —, once a block: its own κ is the visit's from then on.
        if let last, shouldVisitZero() { return try await visitZero(last: last) }
        return best != units && pays(predicted(units) - predicted(best)) ? try await move(units: best) : nil
    }

    /// **κ at most**, where this Mac's is not known yet for a block's arithmetic: the GPU at half its rate beside the
    /// ANE. Measured 1.00 on the M1 Pro, 1.24 to 1.33 on the M5 Pro (25/11/2026). At the share where both units finish
    /// together a cycle takes ≈ S·γ / (1 + ρ) + g, at none S·γ / κ + g: none wins only where κ > 1 + ρ, so a Mac whose
    /// ANE runs as fast as its GPU beside it (ρ ≥ 1: 1.69 on the M1 Pro's codes, ≈ 0.3 on the M5 Pro) never visits it,
    /// where visiting it to measure κ cost a transformer of 6 billion weights on its codes, at 4096 rows, its first run's
    /// two cycles of 16 s against 8.5 on the M1 Pro.
    static let aloneAtMost = 2.0

    /// **The share of none's cycle and the best share's**, predicted from a cycle's calls at `a0` with κ (`alone`) — not
    /// known yet, at its bound (`aloneAtMost`): the share of none at the fastest it could be, every share below the one
    /// measured on the same line (`predicted`).
    static func zeroAgainstBest(_ records: [[Timed]], measured a0: Int, unit: Int, shareable: Int, admitted: Set<Int> = [],
                                alone: Double?) -> (zero: Double, best: Double) {
        let kappa = alone ?? aloneAtMost
        return (predicted(records, measured: a0, rows: 0, unit: unit, alone: kappa),
                bestShare(records, measured: a0, unit: unit, shareable: shareable, admitted: admitted, alone: kappa).seconds)
    }

    /// **Whether the share of none is predicted the fastest** (`zeroAgainstBest`), once a block: faster than the best
    /// share's, and the move there paying over the cycles left.
    private func shouldVisitZero() -> Bool {
        guard !zeroVisited, let at = smoothedAt else { return false }
        let (zero, best) = Self.zeroAgainstBest(smoothed, measured: at, unit: unit, shareable: shareable, admitted: admitted ?? [], alone: alone)
        let now = Self.predicted(smoothed, measured: at, rows: units * unit, unit: unit, alone: alone ?? Self.aloneAtMost)
        return zero < best && pays(now - zero)
    }

    /// **κ, from a warm cycle at 0 rows** that took `zero` seconds and the smoothed calls at the share `a0` before it: the
    /// GPU's past the ANE's rows over every row at their seconds a row beside the ANE, against the cycle's own past its
    /// `gpu` steps (which run alone at any share). Nil where the cycle is no longer than those steps.
    static func aloneRate(_ records: [[Timed]], measured a0: Int, zero: Double) -> Double? {
        let fixed = records.joined().filter { $0.kind == .fixed }.reduce(0) { $0 + $1.seconds }
        let beside = predicted(records, measured: a0, rows: 0, unit: 0) - fixed
        guard zero > fixed, beside > 0 else { return nil }
        return beside / (zero - fixed)
    }

    /// **The cycle before, observed** at the next one's start: the noise, the difference
    /// between two warm cycles at one share, smoothed; the first cycle's own seconds, and the cold cost the second tells —
    /// what a move costs before any was measured (the functions of a share met first are called first as the first
    /// cycle's were) —, unless the stage's opening compiled: what a compilation costs is not what a move costs (decided
    /// 24/11/2026, `prepare` specializes them before).
    func observe(last: Double?, warm: Bool) {
        guard let last else { return }
        let rows = units * unit
        if let p = previous, p.rows == rows, p.warm, warm { noise = Self.smoothing(noise, abs(last - p.seconds)) }
        if firstCycle == nil { firstCycle = last } else if cost == nil, warm, previous?.rows == rows, !stage.compiled { cost = max((firstCycle ?? last) - last, 0) }
        previous = (last, rows, warm)
    }

    /// **Whether a move gaining `gain` seconds a cycle pays**: more than the noise, and over the cycles left (one where
    /// none was said) more than what it costs and the noise.
    func pays(_ gain: Double) -> Bool { Self.pays(gain, noise: noise, cost: cost, left: left) }

    static func pays(_ gain: Double, noise: Double?, cost: Double?, left: Int?) -> Bool {
        let n = noise ?? 0
        return gain > n && gain * Double(max(left ?? 1, 1)) > (cost ?? 0) + n
    }

    /// **The shares that hold under B**, once a block (`admitted`): each count of units
    /// estimated as `move` estimates it — its functions' reading on this Mac, or a bound from their kin's, the share now
    /// among them —, against B beside what is kept (`Stage.holds`); a share the fuse stopped or with no bound left out.
    /// Weighed before any move, so that the share followed is never one B refuses; a refusal in flight (an estimate
    /// short) takes its share out still.
    func admit() throws {
        guard admitted == nil else { return }
        let share = max(stage.observe(), 0)
        let range = Array(1..<max(shareable / unit, 2))
        let forms = try range.map { try form(units: $0, cut: cut) }
        let known = estimated(forms, share: share, now: try form(units: units, cut: cut))
        admitted = Set(zip(range, known).compactMap { n, e in e.bytes.map { stage.holds(expects: $0 - share) } == true ? n : nil })
    }

    /// **A value smoothed**: the new one weighed so that a cycle's half-life is two (1 − 2^−½ ≈ 0.29), the heat and the
    /// battery followed within a few cycles, one cycle's noise damped.
    static func smoothing(_ old: Double?, _ new: Double) -> Double {
        guard let old else { return new }
        return old + 0.29 * (new - old)
    }

    /// **A warm cycle's calls smoothed into the block's**, each step's seconds a row (the ANE's over its rows, the GPU's
    /// beside over the rows past the ANE's, the GPU's alone as they are), at the share they ran at `a0`; the ANE's rate
    /// against the GPU's beside, FLOP for FLOP. Each group's calls summed step by step first (`Timed.summed`: a share's
    /// count of calls changes, its steps do not); another shape of steps starts the smoothing over.
    func smooth(_ cycle: [[Timed]], at a0: Int) {
        let records = cycle.map(Timed.summed)
        var ane = (flops: 0, seconds: 0.0), gpu = (flops: 0, seconds: 0.0)
        for t in records.joined() {
            switch t.kind {
            case .neural: ane = (ane.flops + t.flops, ane.seconds + t.seconds)
            case .past: gpu = (gpu.flops + t.flops, gpu.seconds + t.seconds)
            case .fixed: break
            }
        }
        if ane.seconds > 0, gpu.seconds > 0, gpu.flops > 0 {
            ratio = Self.smoothing(ratio, (Double(ane.flops) / ane.seconds) / (Double(gpu.flops) / gpu.seconds))
        }
        func shape(_ r: [[Timed]]) -> [[String]] { r.map { $0.map { "\($0.kind) \($0.step) \($0.phase) \($0.first)" } } }
        guard shape(smoothed) == shape(records), let before = smoothedAt else { (smoothed, smoothedAt) = (records, a0); return }
        // Each call's seconds a row now, from the smoothed ones at the share they were taken at.
        func perRow(_ t: Timed, _ a: Int) -> Double {
            switch t.kind {
            case .neural: t.seconds / Double(max(a, 1))
            case .past: t.seconds / Double(max(t.rows, 1))
            case .fixed: t.seconds
            }
        }
        for i in records.indices {
            for j in records[i].indices {
                let new = records[i][j], old = smoothed[i][j]
                let value = Self.smoothing(perRow(old, before), perRow(new, a0))
                smoothed[i][j] = new
                switch new.kind {
                case .neural: smoothed[i][j].seconds = value * Double(max(a0, 1))
                case .past: smoothed[i][j].seconds = value * Double(max(new.rows, 1))
                case .fixed: smoothed[i][j].seconds = value
                }
            }
        }
        smoothedAt = a0
    }

    /// **The best share predicted** from a cycle's calls (`predicted`, over every share of whole units of `unit`, the
    /// first of the fastest): its seconds and its ANE rows. `admitted`: the counts of units that hold under B (`admit`);
    /// none known, every share.
    static func bestShare(_ records: [[Timed]], measured a0: Int, unit: Int, shareable: Int, admitted: Set<Int> = [],
                          alone: Double = 1) -> (seconds: Double, rows: Int) {
        let all = Array(1..<max(shareable / unit, 2)), within = all.filter(admitted.contains)
        func t(_ k: Int) -> Double { predicted(records, measured: a0, rows: k * unit, unit: unit, alone: alone) }
        let k = (within.isEmpty ? all : within).min { (t($0), $0) < (t($1), $1) }!
        return (t(k), k * unit)
    }

    /// **The share of none, visited** at a cycle's first group, the cycle before warm at the share: its seconds noted, the
    /// block moved to 0 rows (not kept on this Mac yet; refused under B, weighed again at the next opening). The seconds
    /// the loading took; nil, nothing moved.
    private func visitZero(last: Double) async throws -> Double? {
        zeroVisited = true
        let rows = units * unit
        // Refused under B: told with the cycle (`Stages.Cycle.refused`).
        guard let loaded = try await move(units: 0, keep: false) else { zeroRefused = true; return nil }
        zero = .cold(rows: rows, seconds: last)
        return loaded
    }

    /// **The share of none, measured** (`zero`), at a cycle's first group: `cold`, the cycle before was the first at 0 —
    /// dropped; `measured`, it was the second (`zeroMeasured`). The seconds the loading took; nil, nothing moved.
    private func probe(_ z: Zero, last: Double) async throws -> Double? {
        switch z {
        case .cold(let rows, let seconds):
            zero = .measured(rows: rows, seconds: seconds)
            return nil
        case .measured(let rows, let seconds):
            guard let back = zeroMeasured(at: last, share: rows, seconds) else { return nil }
            return try await move(units: back / unit)
        }
    }

    /// **A warm cycle at 0 rows taken `zero` seconds**, against `seconds` at `share` rows: κ measured from it and the
    /// smoothed calls at the share (`aloneRate`), this block's from now on and its arithmetic's on this Mac, smoothed from block
    /// to block (`ThisMac.Capacity.alone`: the next openings predict their share of none with it, not its bound). The share
    /// of none weighed against the best share the curve predicts with that κ (`bestShare`), the share left where it ran
    /// faster than that prediction: not against the share left alone — a first opening starts at half, far from where
    /// the units finish together, and its share of none won there and stayed, its cycles no longer timed (a transformer of
    /// 13 billion weights in f16 on the M5 Pro, 25/11/2026: 5.40 s at 0 against 7.63 at 2048, kept, where 1024 rows took
    /// 4.82). Kept for the next openings where it is the fastest (`flight`). The rows to go to where that pays its move
    /// over the cycles left (`pays`); nil: it stays at 0.
    func zeroMeasured(at zero: Double, share rows: Int, _ seconds: Double) -> Int? {
        self.zero = nil
        let measured = smoothedAt.flatMap { Self.aloneRate(smoothed, measured: $0, zero: zero) }
        if let measured { alone = measured }
        let predicted = smoothedAt.map { Self.bestShare(smoothed, measured: $0, unit: unit, shareable: shareable,
                                                         admitted: admitted ?? [], alone: alone ?? 1) }
        let target = predicted.flatMap { $0.seconds < seconds ? $0 : nil } ?? (seconds: seconds, rows: rows)
        let (blockKey, arithmetic, wins) = (key, arithmetic, zero <= target.seconds)
        try? ThisMac.update {
            if let measured {
                var c = $0.capacities?[arithmetic] ?? ThisMac.Capacity()
                c.alone = Self.smoothing(c.alone, measured)
                $0.capacities = ($0.capacities ?? [:]).merging([arithmetic: c]) { $1 }
            }
            if wins { $0.flight = ($0.flight ?? [:]).merging([blockKey: 0]) { $1 } }
        }
        return wins || !pays(zero - target.seconds) ? nil : target.rows
    }

    /// **The ANE's share moved to `n` units**, if what the move loads holds under B: the functions it loads, their
    /// estimate — their reading on this Mac, or a bound from their kin's, the share now among them (`estimated`) —,
    /// holding under B beside what is kept (`Stage.rekey`); a set the fuse stopped or with no bound never taken. Its rows
    /// kept on this Mac. The seconds the loading took; nil, nothing moved.
    private func move(units n: Int, keep: Bool = true) async throws -> Double? {
        let rows = n * unit, moving = try form(units: n, cut: cut), share = max(stage.observe(), 0)
        guard let estimate = estimated([moving], share: share, now: try form(units: units, cut: cut))[0].bytes,
              stage.rekey(moving.key, expects: estimate - share, extent: moving.extent) else {
            admitted?.remove(n)
            return nil
        }
        units = n
        let t = DispatchTime.now().uptimeNanoseconds
        let compiled = try await stage.loading { try await load() }
        let loaded = Functions.seconds(since: t)
        // A load that compiled says what a compilation costs, not a move.
        if !compiled { cost = Self.smoothing(cost, loaded) }
        told()
        let blockKey = key
        if keep { try? ThisMac.update { $0.flight = ($0.flight ?? [:]).merging([blockKey: rows]) { $1 } } }
        return loaded
    }

    /// **The cycle begun last, told done at `end`** (`Stages.Event.cycle`): at the next cycle's first group, or at the
    /// block's close. Its rates noted where it ran warm at the share kept — nothing loaded at its start, not the first, not
    /// the share of none's —: kept on this Mac at the block's close (`keepMeasured`), no file written in flight.
    func cycled(until end: UInt64) {
        defer { lastCalls = nil; flops = Stages.Flops() }
        if traced { told(until: end) }
        guard let b = begun else { return }
        begun = nil
        let cycle = Stages.Cycle(block: declaration.name, stage: stage.plan.declaration.name, unit: b.unit, rows: b.rows, shareable: shareable / b.unit * b.unit,
                                 trying: b.trying, seconds: Double(end &- b.at) * 1e-9, end: end, moved: b.moved, climbed: b.climbed,
                                 calls: lastCalls, refused: b.refused, flops: flops)
        Stages.tell(.cycle(cycle))
        // The last cycle beside the ANE has no calls timed (`rebalance` times them at the next cycle's start): the one before kept.
        guard !key.isEmpty, b.moved == nil, b.climbed == nil, !b.first, !b.trying, b.rows == 0 || lastCalls != nil else { return }
        warmRates = cycle.rates
    }

    /// **The trace's calls told** (`Stages.Trace.calls`): those recorded since the cycle under way began, the cycle's
    /// under its ordinal; before the first cycle — groups run once, edges — or in a block that runs none, a run of its
    /// own, to its last call's end. The stream runs ahead of the calls put on it: a run whose GPU calls are not all done
    /// waits for the next one's telling, or the close (`toldLast`), its ends read then, never waited for.
    func told(until end: UInt64) {
        let cycle: Int? = begun == nil ? nil : traceCycles
        if begun != nil { traceCycles += 1 }
        if let p = pendingTrace { pendingTrace = nil; tell(p) }
        guard !recorded.isEmpty else { return }
        let run = PendingTrace(cycle: cycle, start: begun?.at, end: begun == nil ? nil : end, records: recorded)
        recorded = []
        if run.records.contains(where: { $0.end?.value.load(ordering: .acquiring) == 0 }) { pendingTrace = run } else { tell(run) }
    }

    /// The run held for its ends, told at the block's close.
    func toldLast() {
        if let p = pendingTrace { pendingTrace = nil; tell(p) }
    }

    /// A run of calls told: a GPU call's start its encoding or the call before's end, the later (`Timing`); an end its
    /// wait has not written yet left unread.
    private func tell(_ run: PendingTrace) {
        let calls = run.records.map { r -> Stages.Called in
            var c = r.call
            if let e = r.end {
                c.end = e.value.load(ordering: .acquiring)
                c.start = max(r.encoded, r.after?.value.load(ordering: .acquiring) ?? 0)
            }
            return c
        }
        let first: UInt64 = calls.map(\.start).filter { $0 > 0 }.min() ?? 0
        let start: UInt64 = run.start ?? first
        let last: UInt64 = calls.map(\.end).max() ?? start
        let stop: UInt64 = run.end ?? max(last, start)
        // Told under the stage's name, as its opening and its close are (`Stage.Declaration.name`, not the block's).
        Stages.tell(.traced(.calls(stage: stage.plan.declaration.name, cycle: run.cycle, Stages.occupancy(calls, start: start, end: stop))))
    }

    /// **What the block measured, kept on this Mac** at its close, once a block: its last warm cycle's rates
    /// (`ThisMac.rates`, where they moved by 1 % or more), and its arithmetic's ANE rate against the GPU's, smoothed from
    /// block to block as from cycle to cycle (`ThisMac.Capacity.ratio`).
    func keepMeasured() {
        let (key, arithmetic, rates, ratio) = (key, arithmetic, warmRates, ratio)
        // Kept once: a block closed by its owner is closed again by its `deinit`.
        (warmRates, self.ratio) = (nil, nil)
        let moved = rates.map { r in ThisMac.current.rates?[key].map { $0.moved(r) } ?? true } ?? false
        guard moved || ratio != nil else { return }
        try? ThisMac.update {
            if moved, let rates { $0.rates = ($0.rates ?? [:]).merging([key: rates]) { $1 } }
            if let ratio {
                var c = $0.capacities?[arithmetic] ?? ThisMac.Capacity()
                c.ratio = Self.smoothing(c.ratio, ratio)
                $0.capacities = ($0.capacities ?? [:]).merging([arithmetic: c]) { $1 }
            }
        }
    }

    /// **A cycle's shared passes with `a` rows the ANE's, predicted from their calls at `a0`**, as `run` orders them: each
    /// phase lasts the longer of its two units' calls — the GPU's past the ANE's rows at their seconds a row, the ANE's
    /// at theirs (`neural` times them: another unit's cost a row); the `gpu` steps' calls, in turn. (A balance of the
    /// pass's sums, f = 1 / (1 + r), overlaps the ANE with every `gpu` step: 7 units of 256 at 1024² predicted 1.99 s, run
    /// in 2.60; this, 2.57 — measured 19/11 on the mini.) A `gpu` step in two (`Block.first`) and the phase after it, as
    /// one: the GPU's step then its calls of the phase, against the step's first call, then the ANE's first unit (`unit`
    /// rows: the records summed by step tell the share's rows, not a call's) over the whole phase or the step's rest, the
    /// longer, then its other units. (The unit read from the summed records, the share's 2560 rows: the whole phase seen
    /// beside the step's rest, a share of 3072 predicted the faster, run 11.39 s a cycle against 10.13 at 2560 — 26/11/2026
    /// on the M1 Pro.)
    static func predicted(_ records: [[Timed]], measured a0: Int, rows a: Int, unit: Int, neural: Double = 1, alone: Double = 1) -> Double {
        // **The GPU's seconds a row past the ANE's, against those measured at `a0`**: at 0 rows it runs every row alone, at
        // 1/κ of them (`ThisMac.Capacity.alone`); between none and the share measured, on the straight line between the two
        // — the GPU slows as the ANE beside it takes more (a transformer of 13 billion weights in f16 on the M5 Pro,
        // 25/11/2026: 1.07 ms a row alone, 1.20 beside 1024 ANE rows, 1.43 beside 2048; at 1024 the line predicts 4.85 s,
        // measured 4.82, the constant 5.40); past the share measured, as measured. κ 1 (unknown, or the M1 Pro's 1.00): as
        // measured everywhere.
        let factor = a >= a0 || a0 <= 0 ? 1 : 1 / alone + (1 - 1 / alone) * Double(a) / Double(a0)
        return records.reduce(0) { sum, r in
            var steps: [String: (kind: Timed.Kind, step: String, phase: Int, seconds: Double, rows: Int, first: Bool)] = [:]
            for c in r {
                let k = "\(c.kind) \(c.step)\(c.first ? " first" : "")"
                steps[k, default: (c.kind, c.step, c.phase, 0, 0, c.first)].seconds += c.seconds
                steps[k, default: (c.kind, c.step, c.phase, 0, 0, c.first)].rows += c.rows
            }
            // The `gpu` steps in two, by the phase they lead into: their first call's seconds, and all of theirs.
            let two = Set(r.filter(\.first).map(\.step))
            var gpu: [Int: Double] = [:], ane: [Int: Double] = [:], serial = 0.0, led: [Int: (first: Double, all: Double)] = [:]
            for s in steps.values {
                switch s.kind {
                case .neural: ane[s.phase, default: 0] += s.seconds * neural * Double(a) / Double(max(a0, 1))
                case .past: gpu[s.phase, default: 0] += s.seconds / Double(s.rows) * Double(max(s.rows - (a - a0), 0)) * factor
                case .fixed where two.contains(s.step):
                    led[s.phase + 1, default: (0, 0)].all += s.seconds
                    if s.first { led[s.phase + 1, default: (0, 0)].first += s.seconds }
                case .fixed: serial += s.seconds
                }
            }
            return sum + serial + Set(gpu.keys).union(ane.keys).union(led.keys).reduce(0) { total, k in
                let g = gpu[k] ?? 0, n = ane[k] ?? 0
                guard let step = led[k] else { return total + max(g, n) }
                // The ANE's first unit over the phase, at `a` rows.
                let lead = a > 0 ? n * Double(min(unit, a)) / Double(a) : 0
                return total + max(step.all + g, step.first + max(lead, step.all - step.first) + n - lead)
            }
        }
    }

    /// **The functions a share of `units` units and calls of `cut` rows at most load**: this Mac's key for them
    /// (`Stage.key`) and what they hold (`Block.extent`) — the edges' and the ANE's as the stage loaded them, the steps'
    /// built and named (not written).
    private func form(units: Int, cut: Int?) throws -> (key: String, extent: Stage.Extent) {
        var loaded = stage.plan.declaration.functions.filter { functions[$0.key] != nil && ($0.key.hasPrefix("edge-") || $0.key.hasPrefix("ane-")) }
        for (name, spec) in Self.steps(declaration, declaration.used, neural: units * unit, unit: unit, cut: cut) {
            nonisolated(unsafe) let m = try Self.module(spec)
            loaded[name] = Stage.Function(Functions.name(m, prefix: "block"), unit: .gpu) { m }
        }
        return (Stage.key(loaded.values.map { ($0.asset, $0.unit) }), try Self.extent(loaded, inFlight: stage.plan.declaration.inFlight))
    }

    /// **The estimates of `forms` in flight** (`Stage.estimates`): this Mac's readings, and the share read now under the
    /// functions loaded now (`now`) as one more of their kin's — a form never read is bounded from it as from any
    /// reading, never scaled below it.
    private func estimated(_ forms: [(key: String, extent: Stage.Extent)], share: Int, now: (key: String, extent: Stage.Extent)) -> [Stage.Estimate] {
        var measured = ThisMac.current
        measured.keep(reading: share, under: now.key)
        measured.extents = (measured.extents ?? [:]).merging([now.key: now.extent]) { $1 }
        return Stage.estimates(extents: forms.map { ($0.key, $0.extent) }, measured: measured, budget: stage.plan.budget)
    }

    /// **The GPU's functions of the share now**, each built, named and loaded where it is not, the others let go once
    /// their calls are done. Their shares are no larger than those P counts, at the tallest calls. Whether one compiled.
    private func load() async throws -> Bool {
        let need = Self.steps(declaration, declaration.used, neural: units * unit, unit: unit, cut: cut)
        var compiled = false
        for (name, spec) in need.sorted(by: { $0.key < $1.key }) where functions[name] == nil {
            let m = try Self.module(spec)
            let asset = try Functions.named(m, prefix: "block")
            signatures[name] = Signature(m)
            let l = try await Functions.loaded(try Functions.asset(asset) { m }, unit: .gpu, stage: stage.plan.declaration.name)
            functions[name] = l.function
            compiled = compiled || l.compiled
        }
        let gone = functions.keys.filter { !$0.hasPrefix("edge-") && !$0.hasPrefix("ane-") && need[$0] == nil }
        guard !gone.isEmpty else { return compiled }
        await settled()
        for n in gone { functions[n] = nil; marks[n] = nil }
        stage.drop(gone)
        return compiled
    }
}
