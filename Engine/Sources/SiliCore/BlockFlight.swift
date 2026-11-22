import CoreAI
import Foundation

/// **The ANE's share found in flight**: the pace timed, the calls' height climbed, the share moved to the one
/// predicted to finish first, the unit under trial settled and kept.
extension Block {
    /// **The block's pace**, at a cycle's first group over `pass`: the seconds between two starts over one pass (the pass
    /// that ran most: a pass run once, a prefix's, gives none), the first left out; their median kept on this Mac
    /// (`ThisMac.paces`) from the second on, again where it moves by 1 % or more. Taken at the unit kept only (`pace`): the
    /// starts of a trial's cycles are dropped when it settles, so that no period spans two units.
    func timed(_ pass: String) {
        guard !paceBase.isEmpty else { return }
        starts[pass, default: []].append(DispatchTime.now().uptimeNanoseconds)
        let s = starts.values.max { $0.count < $1.count }!
        guard s.count >= 4, !pace.isEmpty else { return }
        let periods = zip(s.dropFirst(2), s.dropFirst()).map { Double($0 - $1) * 1e-9 }.sorted()
        let median = periods[periods.count / 2]
        if let kept, abs(median - kept) < 0.01 * kept { return }
        kept = median
        let (key, value) = (pace, median)
        try? ThisMac.update { $0.paces = ($0.paces ?? [:]).merging([key: value]) { $1 } }
    }

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
        let forms = try heights.map { try form(units: units, unit: unit, cut: $0.cut) }
        let known = estimated(forms, share: share, now: forms[height])
        for i in 0..<height {
            guard let estimate = known[i].bytes,
                  stage.rekey(forms[i].key, expects: estimate - share, extent: forms[i].extent) else { continue }
            (height, turns) = (i, 1)
            let t = DispatchTime.now().uptimeNanoseconds
            try await stage.loading { try await load() }
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
    }

    /// **The share in flight moved**, at a cycle's first group: the cycle before's calls timed, the units `predicted`
    /// to finish first taken if they gain 3 % at least, their rows kept on this Mac, the GPU's functions at the new rows
    /// loaded (built and specialized where first met) and the others let go. **The unit under trial** (`UnitTrial`) is
    /// told that cycle's seconds — its own work, what its start loaded left out — and answers: a cold cycle moves no share
    /// (its calls carry the specialization), a measured one moves to the unit next with the share it predicted best, in
    /// rows, rounded to the unit's; the last settles on the faster (`settle`).
    func rebalance() async throws -> Double? {
        defer { cycleRecords = [] }
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
        lastCalls = Stages.Calls(ane: sums.ane, beside: sums.beside, alone: sums.alone)
        let a0 = units * unit
        func predicted(_ u: Int) -> Double { Self.predicted(records, measured: a0, rows: u * unit) }
        let (seconds, rows) = Self.bestShare(records, measured: a0, unit: unit, shareable: shareable), best = rows / unit
        let better = best != units && predicted(best) <= 0.97 * predicted(units)
        guard var t = trial else { return better ? try await move(to: unit, units: best) : nil }
        let cold = t.cold
        let next = t.record(UnitTrial.Best(seconds: seconds, rows: rows))
        trial = t
        switch next {
        case .stay:
            return cold || !better ? nil : try await move(to: unit, units: best)
        case .move(let to):
            // Refused under B (the guard of a share's move): the unit it runs at is the one kept.
            if let loaded = try await move(to: to, units: UnitTrial.units(rows: rows, of: to, shareable: shareable)) { return loaded }
            await settle(unit, best: t.best)
            return nil
        case .settle(let winner):
            // The winner at its own best share: the one its cycle predicted, a unit's multiple.
            let units = UnitTrial.units(rows: t.best[winner]!.rows, of: winner, shareable: shareable)
            var loaded: Double? = nil
            if winner != unit || units != self.units { loaded = try await move(to: winner, units: units) }
            await settle(unit, best: t.best)
            return loaded
        }
    }

    /// **A unit's best share predicted** from a cycle's calls (`predicted`, over every share of whole units of `unit`, the
    /// first of the fastest): its seconds and its ANE rows. What two units are compared by — never the raw cycle, run at
    /// the share the other unit's cycle set: the finer unit's whole advantage may be a share the taller cannot take (M1 Pro,
    /// 20/11: 2560 rows at 512, 3072 at 1024; the raw cycles, both at 3072, showed none).
    static func bestShare(_ records: [[Timed]], measured a0: Int, unit: Int, shareable: Int) -> (seconds: Double, rows: Int) {
        let k = (1..<max(shareable / unit, 2)).min { (predicted(records, measured: a0, rows: $0 * unit), $0) < (predicted(records, measured: a0, rows: $1 * unit), $1) }!
        return (predicted(records, measured: a0, rows: k * unit), k * unit)
    }

    /// **The ANE's share moved to `n` units of `target`** rows, if what the move loads holds under B: the functions it
    /// loads, their estimate — their reading on this Mac, or a bound from their kin's, the share now among them
    /// (`estimated`) —, holding under B beside what is kept (`Stage.rekey`); a set the fuse stopped or with no bound never
    /// taken. Another unit: the surfaces the ANE reads copied into that unit's, the block runs at
    /// it from here. Its rows kept on this Mac. The seconds the loading took; nil, nothing moved.
    private func move(to target: Int, units n: Int) async throws -> Double? {
        let rows = n * target, moving = try form(units: n, unit: target, cut: cut), share = max(stage.observe(), 0)
        guard let estimate = estimated([moving], share: share, now: try form(units: units, unit: unit, cut: cut))[0].bytes,
              stage.rekey(moving.key, expects: estimate - share, extent: moving.extent) else { return nil }
        if target != unit {
            await settled()
            for r in declaration.perRow where neuralRows.contains(r.name) {
                let bytes = (r.rows ?? declaration.rows) * r.rowBytes
                for name in Self.buffers(r) {
                    memcpy(stage.pointer(name + Self.surface(target == home ? nil : target)), stage.pointer(name + Self.surface(unit == home ? nil : unit)), bytes)
                }
            }
        }
        (unit, total, units) = (target, shareable / target, n)
        let t = DispatchTime.now().uptimeNanoseconds
        try await stage.loading { try await load() }
        let loaded = Functions.seconds(since: t)
        told()
        let blockKey = key
        try? ThisMac.update { $0.flight = ($0.flight ?? [:]).merging([blockKey: rows]) { $1 } }
        return loaded
    }

    /// **The cycle begun last, told done at `end`** (`Stages.Event.cycle`): at the next cycle's first group, or at the
    /// block's close.
    func cycled(until end: UInt64) {
        defer { lastCalls = nil }
        guard let b = begun else { return }
        begun = nil
        Stages.tell(.cycle(Stages.Cycle(block: declaration.name, unit: unit, rows: b.rows, shareable: total * unit,
                                        seconds: Double(end &- b.at) * 1e-9, moved: b.moved, climbed: b.climbed, calls: lastCalls)))
    }

    /// **A cycle's shared passes with `a` rows the ANE's, predicted from their calls at `a0`**, as `run` orders them: each
    /// phase lasts the longer of its two units' calls — the GPU's past the ANE's rows at their seconds a row, the ANE's
    /// at theirs (`neural` times them: another unit's cost a row); the `gpu` steps' calls, in turn. (A balance of the
    /// pass's sums, f = 1 / (1 + r), overlaps the ANE with every `gpu` step: 7 units of 256 at 1024² predicted 1.99 s, run
    /// in 2.60; this, 2.57 — measured 19/11 on the mini.)
    static func predicted(_ records: [[Timed]], measured a0: Int, rows a: Int, neural: Double = 1) -> Double {
        records.reduce(0) { sum, r in
            var steps: [String: (kind: Timed.Kind, phase: Int, seconds: Double, rows: Int)] = [:]
            for c in r {
                let k = "\(c.kind) \(c.step)"
                steps[k, default: (c.kind, c.phase, 0, 0)].seconds += c.seconds
                steps[k, default: (c.kind, c.phase, 0, 0)].rows += c.rows
            }
            var gpu: [Int: Double] = [:], ane: [Int: Double] = [:], serial = 0.0
            for s in steps.values {
                switch s.kind {
                case .neural: ane[s.phase, default: 0] += s.seconds * neural * Double(a) / Double(max(a0, 1))
                case .past: gpu[s.phase, default: 0] += s.seconds / Double(s.rows) * Double(max(s.rows - (a - a0), 0))
                case .fixed: serial += s.seconds
                }
            }
            return sum + serial + Set(gpu.keys).union(ane.keys).reduce(0) { $0 + max(gpu[$1] ?? 0, ane[$1] ?? 0) }
        }
    }

    /// **The functions a share of `units` units and calls of `cut` rows at most load**: this Mac's key for them
    /// (`Stage.key`) and what they hold (`Block.extent`) — the edges' and the ANE's as the stage loaded them, the steps'
    /// built and named (not written).
    private func form(units: Int, unit u: Int, cut: Int?) throws -> (key: String, extent: Stage.Extent) {
        // The ANE's functions loaded now: both units' while the unit is under trial.
        var loaded = stage.plan.declaration.functions.filter { functions[$0.key] != nil && ($0.key.hasPrefix("edge-") || $0.key.hasPrefix("ane-")) }
        for (name, spec) in Self.steps(declaration, declaration.used, neural: units * u, cut: cut) {
            nonisolated(unsafe) let m = try Self.module(spec, codes: codes)
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
        return Stage.estimates(extents: forms.map { ($0.key, $0.extent) }, measured: measured)
    }

    /// **The GPU's functions of the share now**, each built, named and loaded where it is not, the others let go once
    /// their calls are done. Their shares are no larger than those P counts, at the tallest calls.
    private func load() async throws {
        let need = Self.steps(declaration, declaration.used, neural: units * unit, cut: cut)
        for (name, spec) in need.sorted(by: { $0.key < $1.key }) where functions[name] == nil {
            let m = try Self.module(spec, codes: codes)
            let asset = try Functions.named(m, prefix: "block")
            signatures[name] = Signature(m)
            functions[name] = try await Functions.load(try Functions.asset(asset) { m }, unit: .gpu)
        }
        let gone = functions.keys.filter { !$0.hasPrefix("edge-") && !$0.hasPrefix("ane-") && need[$0] == nil }
        guard !gone.isEmpty else { return }
        await settled()
        for n in gone { functions[n] = nil; marks[n] = nil }
        stage.drop(gone)
    }

    /// **The unit decided**, at the trial's end: the unit the block runs at kept on this Mac for good, the other's functions
    /// let go (their surfaces stay, counted in P), the cycles' starts dropped but this one's so that the pace is taken
    /// at the unit kept alone (`pace`). `best`: each unit's best share predicted.
    private func settle(_ winner: Int, best: [Int: UnitTrial.Best]) async {
        trial = nil
        starts = starts.mapValues { Array($0.suffix(1)) }
        kept = nil
        let gone = functions.keys.filter { $0.hasPrefix("ane-") && !$0.hasSuffix("@\(winner)") }
        if !gone.isEmpty {
            await settled()
            for n in gone { functions[n] = nil; marks[n] = nil }
            stage.drop(gone)
        }
        keep(winner, best: best)
    }

    /// The unit kept on this Mac (`ThisMac.neuralUnits`) and said (`Stages.Event.weighed`).
    func keep(_ unit: Int, best: [Int: UnitTrial.Best]) {
        let key = key
        try? ThisMac.update { $0.neuralUnits = ($0.neuralUnits ?? [:]).merging([key: unit]) { $1 } }
        Stages.tell(.weighed(Stages.Weighed(block: declaration.name, seconds: best.mapValues(\.seconds), rows: best.mapValues(\.rows), kept: unit)))
    }
}
