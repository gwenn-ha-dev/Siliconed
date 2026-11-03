import Accelerate
internal import AMXLegacy
import Foundation
import Metal
import MetalPerformanceShadersGraph

// The sharing probes: the sets of engines, each side alone and then together, the integers' pair included.

extension Diagnostic.CapacityProbes.Context {
    // MARK: Sharing of the integers

    /// **The GPU's int8 kernel beside the ANE's**, both reading the integers staged in the reserve — the pair a render of a Compact makes
    /// under `ane_int8` — as the fp32 sets are measured: chains, the lines' balance (pass 1), the balance of the κ it gave (pass 2, the
    /// same shape) and a long k (pass 3, on pass 2's κ), the fit reading the passes from the second. The long k is the point: the
    /// « down » projections of the three int8 families (k = 8 192 to 16 384) were left to the ANE alone for want of a pair measured
    /// there, which cost them a third of their seconds.
    ///
    /// **Twice: beside the GPU's kernel on the fp32 activation, and on the activation rounded to bf16** (`gpu/int8act+ane/int8`). The
    /// rounded kernel runs on the GPU's tensor units at another rate and draws another power: what the ANE beside it keeps, and what it
    /// keeps of itself, is that pair's own measure — the fp32 pair's κ read for it would be a guess (each set keeps its own sharing).
    func sharingInt8(_ measurements: inout Fit.Measurements) {
        for form in Self.integerPairs { sharingInt8(&measurements, gpu: Kernel(.gpu, form)) }
    }

    /// The GPU's kernels the ANE's integers are measured beside, each pair its own sharing (`gpu/int8+ane/int8`, `gpu/int8act+ane/int8`).
    static let integerPairs: [CapacityModel.Form] = [.int8, .int8act]

    func sharingInt8(_ measurements: inout Fit.Measurements, gpu g: Kernel) {
        restoreB()
        var model = Fit.model(from: measurements, allocationUnit: NeuralEngine.rowUnit)
        let a = Kernel(.ane, .int8), rounded = g.form == .int8act
        guard model.lines[g] != nil, model.lines[a] != nil, rounded ? alone.tensorInt8 && alone.bf16Activations : both.tensorInt8,
              neural.neuralEngine != nil, let short = sizes.chains.first else { return }
        let key = CapacityModel.key([g, a])
        for (pass, shape) in [(1, short), (2, short)] + (sizes.chains.count > 1 ? [(3, sizes.chains[1])] : []) {
            if let t = trialInt8(key: key, gpu: g, shape: shape, model: model, pass: pass) {
                measurements.trials.append(t)
                log?(String(format: "  %@ pass %d  %dx%dx%d  wall %.2f ms  sides %@", key, pass, shape.m, shape.k, shape.n, t.wall * 1e3,
                            t.sides.map { String(format: "%@ %d rows line %.1f within %.1f", $0.kernel, $0.rows, $0.solo * 1e3, ($0.within ?? 0) * 1e3) }.joined(separator: " | ")))
            }
            let trials = pass == 1 ? measurements.trials : measurements.trials.filter { $0.pass >= 2 }
            if let fit = Fit.overlap(set: key, trials: trials, model: model) { model.sharing[key] = fit.sharing }
        }
    }

    func trialInt8(key: String, gpu g: Kernel, shape: (m: Int, k: Int, n: Int), model: CapacityModel, pass: Int) -> Fit.Measurements.Trial? {
        let (m, k, n) = shape
        let a = Kernel(.ane, .int8), rounded = g.form == .int8act
        // The rounded GPU side runs on `alone` (the rounding kernels), the plan's rounded copy in the reserve's tail beside the integers
        // the ANE reads; the ANE is `neural`'s either way.
        let gemm = rounded ? alone : both, buffers = rounded ? aloneB : bothB
        // The balance of a chain is asked of the model without the sets' domains: a chain at a k the pair was not measured at yet is how it gets it.
        var open = model
        for (name, var sharing) in model.sharing { sharing.kFrom = nil; sharing.kTo = nil; open.sharing[name] = sharing }
        guard let plan = int8Plan(k, n, rounded: rounded), let engine = neural.neuralEngine,
              let split = open.split(m: m, k: k, n: n, kernels: [g, a]), let aneRows = split.rows[a], let gpuRows = split.rows[g],
              aneRows > 0, gpuRows > 0 else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): the model gives a side no row"); return nil }
        // The rows the engines can run for this key: the plan's own rule (`isRunnable`) and the process's (the program loads).
        guard ProductAssignment(path: .native, roundsActivation: rounded, gpu: gpuRows, ane: aneRows).isRunnable(for: ProductKey(k: k, n: n, m: m, weight: .int8)),
              engine.accepts(rows: aneRows, k: k, n: n, int8Group: plan.group) else {
            log?("  \(key) \(m)x\(k)x\(n) pass \(pass): rows \(aneRows) not accepted by the engines"); return nil
        }
        prime(.int8, ane: true, m: m, k: k, n: n)
        stageInt8(plan, k: k, n: n)
        RouteBeacon.announce(g.description); RouteBeacon.announce(a.description)
        func once() -> (wall: Double, within: [String: Double])? {
            var result: (gpu: Double, computed: Bool, gpuWall: Double, aneWall: Double)?
            guard let wall = CapacityProbes.timed({
                result = WeightProducts.int8Product(gemm: gemm, plan: plan, reserve: buffers.b, a: buffers.a, c: buffers.c,
                                                    m: m, k: k, n: n, ane: engine, aneRows: aneRows)
                return result?.computed == true
            }), let r = result else { return nil }
            return (wall, [a.description: r.aneWall, g.description: r.gpuWall])
        }
        guard let verdict = judged(key, .int8, ane: true, m: m, k: k, n: n, { once() != nil }) else {
            log?("  \(key) \(m)x\(k)x\(n) pass \(pass): the set refused (fallback)"); return nil
        }
        guard verdict.pass else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): WRONG, error \(verdict.error)"); return .init(set: key, shape: [m, k, n], pass: pass, sides: [], wall: 0, error: verdict.error) }
        // The chain, as the fp32 sets' (`trial`): back to back, the ramp thrown away, the median of the window.
        var samples: [(wall: Double, within: [String: Double])] = []
        guard guarded("\(key) \(m)x\(k)x\(n) pass \(pass)", { () -> Bool? in
            samples = []
            let begin = CapacityProbes.now()
            var calls = 0
            while true {
                let start = CapacityProbes.now()
                guard let s = once() else { return nil }
                calls += 1
                if calls > CapacityProbes.chainWarmCalls, start - begin >= CapacityProbes.chainWarm { samples.append(s) }
                let elapsed = CapacityProbes.now() - begin
                if elapsed >= CapacityProbes.chainWarm + CapacityProbes.chainWindow, samples.count >= CapacityProbes.chainCalls { break }
                if elapsed >= CapacityProbes.chainCap, !samples.isEmpty { break }
            }
            CapacityProbes.lastSpread = max(CapacityProbes.lastSpread, CapacityProbes.spread(samples.map(\.wall)))
            return true
        }) != nil else { return nil }
        let used = CapacityProbes.quick(samples)
        let wall = Fit.median(used.map(\.wall))
        // `solo` is the line's seconds at the rows the side took (the reference of κ, as the fp32 sets').
        var sides: [Fit.Measurements.Side] = []
        for (kernel, rows) in [(g, gpuRows), (a, aneRows)] {
            guard let solo = model.alone(kernel, rows: rows, k: k, n: n) else { return nil }
            let within = used.compactMap { $0.within[kernel.description] }
            sides.append(.init(kernel: kernel.description, rows: rows, solo: solo, within: within.isEmpty ? nil : Fit.median(within)))
        }
        return .init(set: key, shape: [m, k, n], pass: pass, sides: sides, wall: wall, error: verdict.error)
    }

    // MARK: Sharing

    func rows(_ kernels: [Kernel], model: CapacityModel, m: Int, k: Int, n: Int) -> [Kernel: Int]? {
        Diagnostic.CapacityProbes.trialRows(kernels, model: model, m: m, k: k, n: n)
    }

    /// **The sets, as chains.** Every set of two or more engines that have a line, the smallest first (a set's κ is read through the
    /// sets that remain when its first side ends; none is ever taken at keep 1): the lines' balance (pass 1, κ = 1), then the balance of
    /// the κ it gave (pass 2, the same short shape) and a long k (pass 3, on pass 2's κ). The fit reads the passes from the second.
    func sharing(_ measurements: inout Fit.Measurements) {
        restoreB()
        var model = Fit.model(from: measurements, allocationUnit: NeuralEngine.rowUnit)
        let g = Kernel(.gpu, .fp32), x = Kernel(.amx, .fp32), a = Kernel(.ane, .fp16)
        let have = [g, x, a].filter { model.lines[$0] != nil }
        var sets: [[Kernel]] = []
        for size in 2...max(2, have.count) where have.count >= 2 {
            for mask in 1..<(1 << have.count) where mask.nonzeroBitCount == size {
                sets.append(have.indices.filter { mask & (1 << $0) != 0 }.map { have[$0] })
            }
        }
        let chains = sizes.chains
        for set in sets {
            let key = CapacityModel.key(set.map(\.engine))
            for (pass, shape) in [(1, chains[0]), (2, chains[0])] + (chains.count > 1 ? [(3, chains[1])] : []) {
                if let trial = trial(set, key: key, shape: shape, model: model, pass: pass) {
                    measurements.trials.append(trial)
                    log?(String(format: "  %@ pass %d  %dx%dx%d  wall %.2f ms  sides %@", key, pass, shape.m, shape.k, shape.n,
                                trial.wall * 1e3, trial.sides.map { String(format: "%@ %d rows line %.1f within %.1f", $0.kernel, $0.rows, $0.solo * 1e3, ($0.within ?? 0) * 1e3) }.joined(separator: " | ")))
                }
                let trials = pass == 1 ? measurements.trials : measurements.trials.filter { $0.pass >= 2 }
                if let fit = Fit.overlap(set: key, trials: trials, model: model) { model.sharing[key] = fit.sharing }
            }
        }
    }

    func trial(_ set: [Kernel], key: String, shape: (m: Int, k: Int, n: Int), model: CapacityModel, pass: Int,
                       aneRows forced: Int? = nil) -> Fit.Measurements.Trial? {
        let (m, k, n) = shape
        // The balance of a chain is asked of the model without the sets' domains: a chain at a k the sets were not measured at yet is how they get it.
        var open = model
        for (name, var sharing) in model.sharing { sharing.kFrom = nil; sharing.kTo = nil; open.sharing[name] = sharing }
        var planned = rows(set, model: open, m: m, k: k, n: n)
        if let forced, let p = planned { planned = p.merging([Kernel(.ane, .fp16): forced, Kernel(.gpu, .fp32): m - forced]) { $1 } }
        guard let plan = planned else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): the model gives a side no row"); return nil }
        let g = Kernel(.gpu, .fp32), x = Kernel(.amx, .fp32), a = Kernel(.ane, .fp16)
        let hasANE = set.contains(a), hasAMX = set.contains(x)
        let engine = (hasAMX ? three : neural).neuralEngine
        let gemm = hasANE ? (hasAMX ? three : neural) : both
        let buffers = hasANE ? (hasAMX ? threeB : neuralB) : bothB
        let aneRows = plan[a] ?? 0
        let first = m - aneRows
        // The GPU's rows by the conductor's quantum, the AMX the rest of the GPU's side.
        // Without the GPU (the AMX beside the ANE) the side is the AMX's whole.
        let hasGPU = set.contains(g)
        var gpuRows = hasGPU ? plan[g] ?? first : 0, amxRows = hasGPU ? 0 : first
        var fraction = 1.0
        if hasAMX && hasGPU {
            fraction = Double(plan[g]!) / Double(plan[g]! + plan[x]!)
            // The GPU's rows on MPS's tile, the AMX the rest of the side (it keeps one tile at least, and the rows its line speaks for).
            let tile = GEMM.rowQuantum, amxFewest = max(tile, model.lines[x]?.rowsFrom ?? 0)
            gpuRows = max(tile, min(first - amxFewest, Int((Double(first) * fraction / Double(tile)).rounded()) * tile))
            amxRows = first - gpuRows
            fraction = Double(gpuRows) / Double(first)
        }
        let key2 = "\(k)×\(n)"
        // The set's one call, and the sides' own clocks it leaves.
        // The plan answers this key with these exact rows until the next set is pinned.
        func configure() -> Bool {
            guard !hasANE || engine != nil else { return false }
            let key = ProductKey(k: k, n: n, m: m)
            let assignment = ProductAssignment(gpu: gpuRows, amx: amxRows, ane: aneRows)
            guard assignment.rows == m, assignment.isRunnable(for: key) else { return false }
            gemm.plan.unpinAll()
            gemm.plan.pin(key, assignment)
            return true
        }
        guard configure() else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): rows \(aneRows)/\(gpuRows)/\(amxRows) not accepted by the engines"); return nil }
        for kernel in set { RouteBeacon.announce(kernel.description) }
        func once() -> (wall: Double, within: [String: Double])? {
            let before = gemm.conductor?.perShape[key2] ?? Conductor.ShapeRecord()
            let fallbacks = engine?.fallbacks ?? 0
            guard let wall = CapacityProbes.timed({ _ = gemm.linear(a: buffers.a, b: buffers.b, c: buffers.c, m: m, k: k, n: n, weightIsTransposed: true); return true }) else { return nil }
            if let e = engine, e.fallbacks != fallbacks { return nil }
            var within: [String: Double] = [:]
            if hasANE, let sides = gemm.lastSides {
                within[a.description] = sides.ane
                if hasGPU { within[g.description] = sides.gpu }
            }
            if let c = gemm.conductor, let after = c.perShape[key2], !hasANE || hasAMX {
                let w = after.wall - before.wall
                if hasGPU { within[g.description] = w - (after.idleGPU - before.idleGPU) }
                within[x.description] = w - (after.idleAMX - before.idleAMX)
            }
            return (wall, within)
        }
        // Judged once: the set's output on the sampled rows (every quantum of the cut is visited), then warmed.
        guard let verdict = judged(key, .fp32, ane: hasANE, m: m, k: k, n: n, { once() != nil }) else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): the set refused (fallback)"); return nil }
        guard verdict.pass else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): WRONG, error \(verdict.error)"); return .init(set: key, shape: [m, k, n], pass: pass, sides: [], wall: 0, error: verdict.error) }
        // The chain: back to back, the ramp thrown away, the median of the window (`chain`); the set's clock per call.
        var samples: [(wall: Double, within: [String: Double])] = []
        guard guarded("\(key) \(m)x\(k)x\(n) pass \(pass)", { () -> Bool? in
            samples = []
            let begin = CapacityProbes.now()
            var calls = 0
            while true {
                let start = CapacityProbes.now()
                guard let s = once() else { return nil }
                calls += 1
                if calls > CapacityProbes.chainWarmCalls, start - begin >= CapacityProbes.chainWarm { samples.append(s) }
                let elapsed = CapacityProbes.now() - begin
                if elapsed >= CapacityProbes.chainWarm + CapacityProbes.chainWindow, samples.count >= CapacityProbes.chainCalls { break }
                if elapsed >= CapacityProbes.chainCap, !samples.isEmpty { break }
            }
            CapacityProbes.lastSpread = max(CapacityProbes.lastSpread, CapacityProbes.spread(samples.map(\.wall)))
            return true
        }) != nil else { return nil }
        let used = CapacityProbes.quick(samples)
        let wall = Fit.median(used.map(\.wall))
        // `solo` is the line's seconds at the rows the side took: the reference of κ (the line's own bias at these rows is in κ).
        var sides: [Fit.Measurements.Side] = []
        for kernel in set {
            let r = kernel == a ? aneRows : kernel == x ? amxRows : gpuRows
            guard let solo = model.alone(kernel, rows: r, k: k, n: n) else { return nil }
            let within = used.compactMap { $0.within[kernel.description] }
            sides.append(.init(kernel: kernel.description, rows: r, solo: solo, within: within.isEmpty ? nil : Fit.median(within)))
        }
        return .init(set: key, shape: [m, k, n], pass: pass, sides: sides, wall: wall, error: verdict.error)
    }
}

extension Diagnostic.CapacityProbes {
    /// **The rows each kernel takes of a set's trial**: the model's balanced split where it gives every kernel at least the rows its line
    /// speaks for (`Line.rowsFrom`; the ANE's unit; the GPU's tile), else each kernel its fewest and the rest by the lines' rates (the ANE's
    /// by whole units, the remainder to the GPU). A trial measures the set's κ, and the fit reads it from a chain balanced or not
    /// (`CapacityFit.overlap`): a set the balance of a short probe leaves a side out of is a set the plan never offers, even on the rows
    /// of a full-size product — the M1 Pro's GPU + AMX + ANE at tier 1, whose 2 048-row chains gave the AMX ~350 rows under its 512.
    /// `nil`: the fewest rows do not fit in `m`, or a kernel has no line.
    package static func trialRows(_ kernels: [CapacityModel.Kernel], model: CapacityModel, m: Int, k: Int, n: Int) -> [CapacityModel.Kernel: Int]? {
        guard kernels.allSatisfy({ (model.lines[$0]?.rate ?? 0) > 0 }) else { return nil }
        let fewest = Dictionary(uniqueKeysWithValues: kernels.map { kernel in
            (kernel, max(model.lines[kernel]?.rowsFrom ?? 0, kernel.engine == .ane ? model.aneRowUnit : GEMM.rowQuantum))
        })
        if let s = model.split(m: m, k: k, n: n, kernels: kernels), kernels.allSatisfy({ (s.rows[$0] ?? 0) >= fewest[$0]! }) { return s.rows }
        let left = m - fewest.values.reduce(0, +)
        guard left >= 0 else { return nil }
        let rates = kernels.map { model.lines[$0]!.rate }, total = rates.reduce(0, +)
        let receiver = kernels.first { $0.engine == .gpu } ?? kernels.first { $0.engine != .ane } ?? kernels[0]
        var rows = fewest, given = 0
        for (kernel, rate) in zip(kernels, rates) where kernel != receiver {
            var extra = Int(Double(left) * rate / total)
            if kernel.engine == .ane { extra = extra / model.aneRowUnit * model.aneRowUnit }
            rows[kernel]! += extra; given += extra
        }
        rows[receiver]! += left - given
        return rows
    }
}
