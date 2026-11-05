import Accelerate
internal import AMXLegacy
import Foundation
import Metal
import MetalPerformanceShadersGraph

// The sharing probes: the sets of engines, each side alone and then together, the pairs on a weight read as published included
// (the integers', the rounded bf16 one).

extension Diagnostic.CapacityProbes.Context {
    // MARK: Sharing of the weights read as published

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
        for form in Self.integerPairs { sharingPair(&measurements, gpu: Kernel(.gpu, form), ane: Kernel(.ane, .int8)) }
    }

    /// The GPU's kernels the ANE's integers are measured beside, each pair its own sharing (`gpu/int8+ane/int8`, `gpu/int8act+ane/int8`).
    static let integerPairs: [CapacityModel.Form] = [.int8, .int8act]

    /// **The Standard's own pair: the GPU's rounded rows on a bf16 weight read as published beside the ANE's fp16 program on the same
    /// bf16 copy** (`gpu/bf16act+ane/fp16`, `WeightProducts.bf16Product`), measured as the integers' pairs. The ×1.52 once claimed
    /// for GPU + ANE on a Standard was the rounded kernel's rate times the κ of `gpu/fp32+ane/fp16`: the GPU's tensor units beside the
    /// ANE keep what this pair's own chains say, nothing borrowed.
    func sharingRounded(_ measurements: inout Fit.Measurements) {
        sharingPair(&measurements, gpu: Self.roundedPair[0], ane: Self.roundedPair[1])
    }

    /// The rounded bf16 pair, its sharing its own.
    static let roundedPair = [Kernel(.gpu, .bf16act), Kernel(.ane, .fp16)]

    /// One pair of a weight read as published, its three passes and its fit (`sharing`'s, for a pair of kernels).
    func sharingPair(_ measurements: inout Fit.Measurements, gpu g: Kernel, ane a: Kernel) {
        restoreB()
        var model = Fit.model(from: measurements, allocationUnit: NeuralEngine.rowUnit)
        let ready = switch (g.form, a.form) {
        case (.int8, .int8): both.tensorInt8
        case (.int8act, .int8): alone.tensorInt8 && alone.bf16Activations
        case (.bf16act, .fp16): alone.bf16Activations
        default: false
        }
        guard ready, model.lines[g] != nil, model.lines[a] != nil, neural.neuralEngine != nil, let short = sizes.chains.first else { return }
        let key = CapacityModel.key([g, a])
        // The rows the last chain ran: where the model of the passes so far gives the ANE none, the next chain runs them again. Each
        // pass is a measure, not a verdict — the long k (where the ANE's per-call cost weighs least) is measured whatever the short one
        // said; a pair skipped there is a pair one map has and the next has not.
        var last: (gpu: Int, ane: Int)?
        var plan = [(1, short), (2, short)] + (sizes.chains.count > 1 ? [(3, sizes.chains[1])] : [])
        var next = 0
        while next < plan.count {
            let (pass, shape) = plan[next]; next += 1
            if let t = trialPair(key: key, gpu: g, ane: a, shape: shape, model: model, pass: pass, previous: last) {
                measurements.trials.append(t)
                if let gr = t.sides.first(where: { $0.kernel == g.description })?.rows, let ar = t.sides.first(where: { $0.kernel == a.description })?.rows {
                    last = (gr, ar)
                }
                log?(String(format: "  %@ pass %d  %dx%dx%d  wall %.2f ms  sides %@", key, pass, shape.m, shape.k, shape.n, t.wall * 1e3,
                            t.sides.map { String(format: "%@ %d rows line %.1f within %.1f", $0.kernel, $0.rows, $0.solo * 1e3, ($0.within ?? 0) * 1e3) }.joined(separator: " | ")))
            }
            refit(key, pass: pass, shape: shape, measurements: measurements, model: &model, plan: &plan, at: next)
        }
    }

    /// **After a chain, the set's fit again** — on the passes so far, from the second the chains `Fit.chains` reads — **and, after the
    /// second, a third chain at the same shape where the first two disagree beyond the map's noise** (`Fit.needsAgain`): two chains
    /// cannot say which met a disturbance, the median of three can. The third is the measure the guard would have taken had it seen
    /// the chain disturbed (a whole chain slowed evenly does not spread its calls).
    func refit(_ key: String, pass: Int, shape: (m: Int, k: Int, n: Int), measurements: Fit.Measurements, model: inout CapacityModel,
               plan: inout [(Int, (m: Int, k: Int, n: Int))], at next: Int) {
        let trials = pass == 1 ? measurements.trials : Fit.chains(set: key, trials: measurements.trials, model: model).used
        if let fit = Fit.overlap(set: key, trials: trials, model: model) { model.sharing[key] = fit.sharing }
        guard pass == 2, Fit.needsAgain(set: key, shape: [shape.m, shape.k, shape.n], trials: measurements.trials, model: model) else { return }
        log?("  \(key) \(shape.m)x\(shape.k)x\(shape.n): two chains apart beyond the map's noise — a third")
        plan.insert((Fit.Measurements.Trial.again, shape), at: next)
    }

    /// **One chain of a pair on a weight read as published**: the weight staged as a render stages it, the rows the model's balance gives
    /// and the engines accept, the route a render takes (`WeightProducts.int8Product(…ane:)`, `WeightProducts.bf16Product`) judged once
    /// against the fp64 truth at the half-precision bar, then chained.
    func trialPair(key: String, gpu g: Kernel, ane a: Kernel, shape: (m: Int, k: Int, n: Int), model: CapacityModel, pass: Int,
                   previous: (gpu: Int, ane: Int)? = nil) -> Fit.Measurements.Trial? {
        let (m, k, n) = shape
        // The balance of a chain is asked of the model without the sets' domains: a chain at a k the pair was not measured at yet is how it gets it.
        var open = model
        for (name, var sharing) in model.sharing { sharing.kFrom = nil; sharing.kTo = nil; open.sharing[name] = sharing }
        guard let engine = neural.neuralEngine else { return nil }
        let gpuRows: Int, aneRows: Int
        if let split = open.split(m: m, k: k, n: n, kernels: [g, a]), let e = split.rows[a], let r = split.rows[g], e > 0, r > 0 {
            (gpuRows, aneRows) = (r, e)
        } else if let previous, previous.gpu + previous.ane == m {
            // The model of the passes so far gives the ANE no row: the last chain's rows again (`sharingPair`).
            (gpuRows, aneRows) = previous
            log?("  \(key) \(m)x\(k)x\(n) pass \(pass): the model gives a side no row — the last chain's rows, \(aneRows) on the ANE")
        } else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): the model gives a side no row"); return nil }
        let truthKind: Truth
        let once: () -> (wall: Double, within: [String: Double])?
        func timedPair(_ run: () -> (gpu: Double, computed: Bool, gpuWall: Double, aneWall: Double)) -> (wall: Double, within: [String: Double])? {
            var result: (gpu: Double, computed: Bool, gpuWall: Double, aneWall: Double)?
            guard let wall = CapacityProbes.timed({ result = run(); return result?.computed == true }), let r = result else { return nil }
            return (wall, [a.description: r.aneWall, g.description: r.gpuWall])
        }
        if a.form == .int8 {
            let rounded = g.form == .int8act
            // The rounded GPU side runs on `alone` (the rounding kernels), the plan's rounded copy in the reserve's tail beside the integers
            // the ANE reads; the ANE is `neural`'s either way.
            let gemm = rounded ? alone : both, buffers = rounded ? aloneB : bothB
            // The rows the engines can run for this key: the plan's own rule (`isRunnable`) and the process's (the program loads).
            guard let plan = int8Plan(k, n, rounded: rounded),
                  ProductAssignment(path: .native, roundsActivation: rounded, gpu: gpuRows, ane: aneRows).isRunnable(for: ProductKey(k: k, n: n, m: m, weight: .int8)),
                  engine.accepts(rows: aneRows, k: k, n: n, int8Group: plan.group) else {
                log?("  \(key) \(m)x\(k)x\(n) pass \(pass): rows \(aneRows) not accepted by the engines"); return nil
            }
            truthKind = .int8
            prime(.int8, ane: true, m: m, k: k, n: n)
            stageInt8(plan, k: k, n: n)
            once = {
                timedPair { WeightProducts.int8Product(gemm: gemm, plan: plan, reserve: buffers.b, a: buffers.a, c: buffers.c,
                                                       m: m, k: k, n: n, ane: engine, aneRows: aneRows) }
            }
        } else {
            // The bf16 copy at the reserve's head (`b`, which the widened weight restores), the GPU's rounded rows in its tail.
            guard let rounding = WeightProducts.roundingPlan(k: k, n: n, capacity: k * n),
                  ProductAssignment(path: .bf16, roundsActivation: true, gpu: gpuRows, ane: aneRows).isRunnable(for: ProductKey(k: k, n: n, m: m)),
                  engine.accepts(rows: aneRows, k: k, n: n) else {
                log?("  \(key) \(m)x\(k)x\(n) pass \(pass): rows \(aneRows) not accepted by the engines"); return nil
            }
            truthKind = .fp32
            prime(.fp32, ane: true, m: m, k: k, n: n)
            Widen.copyBfloat16(source: operands.bf16.base, destination: operands.b.base, count: k * n)
            dirty = max(dirty, k * n)
            once = {
                // **The pair as the render runs it.** Where a denoiser streams the ANE (`GPUStream.policy` on), its rows run beside the
                // queued GPU rows, no rendezvous on the CPU: measured so. The synchronous form's chains read the GPU at κ 0.74–0.80 at
                // k = 3 840 where the stream's read 0.91–0.98, and three times their noise — a pair priced 4–14 % over the GPU alone,
                // taken or not from one map to the next, that a render ran 12 % under it.
                if GPUStream.policy == .on {
                    return timedPair { WeightProducts.bf16ProductStreamed(gemm: self.alone, rounding: (self.aloneB.b, rounding.offset, rounding.rows),
                                                                          reserve: self.aloneB.b, a: self.aloneB.a, c: self.aloneB.c,
                                                                          m: m, k: k, n: n, ane: engine, aneRows: aneRows) }
                }
                return timedPair { WeightProducts.bf16Product(gemm: self.alone, rounding: (self.aloneB.b, rounding.offset, rounding.rows),
                                                              reserve: self.aloneB.b, a: self.aloneB.a, c: self.aloneB.c,
                                                              m: m, k: k, n: n, ane: engine, aneRows: aneRows) }
            }
        }
        RouteBeacon.announce(g.description); RouteBeacon.announce(a.description)
        guard let verdict = judged(key, truthKind, ane: true, m: m, k: k, n: n, { once() != nil }) else {
            log?("  \(key) \(m)x\(k)x\(n) pass \(pass): the set refused (fallback)"); return nil
        }
        guard verdict.pass else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): WRONG, error \(verdict.error)"); return .init(set: key, shape: [m, k, n], pass: pass, sides: [], wall: 0, error: verdict.error) }
        guard let used = chained("\(key) \(m)x\(k)x\(n) pass \(pass)", once) else { return nil }
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

    /// **A set's chain**: the call back to back, the ramp thrown away, the median's window (`CapacityProbes.quick`) — the samples kept,
    /// under the guard that measures again a chain the Mac disturbed. `nil`: a call refused, or the guard gave up.
    func chained(_ step: String, _ once: () -> (wall: Double, within: [String: Double])?) -> [(wall: Double, within: [String: Double])]? {
        var samples: [(wall: Double, within: [String: Double])] = []
        guard guarded(step, { () -> Bool? in
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
        return CapacityProbes.quick(samples)
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
            var plan = [(1, chains[0]), (2, chains[0])] + (chains.count > 1 ? [(3, chains[1])] : [])
            var next = 0
            while next < plan.count {
                let (pass, shape) = plan[next]; next += 1
                if let trial = trial(set, key: key, shape: shape, model: model, pass: pass) {
                    measurements.trials.append(trial)
                    log?(String(format: "  %@ pass %d  %dx%dx%d  wall %.2f ms  sides %@", key, pass, shape.m, shape.k, shape.n,
                                trial.wall * 1e3, trial.sides.map { String(format: "%@ %d rows line %.1f within %.1f", $0.kernel, $0.rows, $0.solo * 1e3, ($0.within ?? 0) * 1e3) }.joined(separator: " | ")))
                }
                refit(key, pass: pass, shape: shape, measurements: measurements, model: &model, plan: &plan, at: next)
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
        // The chain: back to back, the ramp thrown away, the median of the window (`chained`); the set's clock per call.
        guard let used = chained("\(key) \(m)x\(k)x\(n) pass \(pass)", once) else { return nil }
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
