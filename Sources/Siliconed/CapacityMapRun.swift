import Accelerate
internal import AMXLegacy
import Foundation
import Metal
import MetalPerformanceShadersGraph

/// **The capacity map's machine layer: the probes that run one route of one class once, and the run that gives the
/// measuring core (`CapacityMeasure.plan`) its clock, its classes and its probe.** The decisions, the allocation of the
/// seconds and the table are pure (`CapacityMeasure`, `CapacityMapClasses.swift`, `RouteTable`); this file holds the
/// Metal, the Core ML, the BNNS and the fp64 CPU behind them.
///
///     Diagnostic.run
///        ├─ plan the memory           from the classes' shapes, under B, before the first product     (MemoryNeedCapacityMap)
///        ├─ widening                  the preparation rates (bf16 / int8 / Q8_0: widen, copy), 0.2 s
///        ├─ witnesses, start          a fixed product on the GPU, the AMX and the ANE: median of 5, their dispersion
///        ├─ CapacityMeasure.plan      60 s of wall for everything, minus what the end needs
///        │     probe(class, route)    ┌ first time:  the shares of its cut (balanced once, kept), one run to warm, one to judge
///        │                            │              against fp64 (a wrong route is a hole of that class, nothing more)
///        │                            └ every time:  one timed run + the weight's preparation (widen or copy), seconds
///        ├─ witnesses, end            the same products again, asked by the plan when its allocation is over → the heat
///        └─ RouteTable(plan:…)        the table, the verdict, the report
///
/// What sets it apart — what a probe written by analogy with the old stages would get wrong:
///
///   - **A probe returns a duration it has already warmed**: the first call of a route on a class builds pipelines,
///     compiles a Core ML program, wires pages. That call is an untimed run that is also the judged one; the ratio the
///     plan sees is steady state. A route that comes after its ANE engine was released and rebuilt (`generation`) warms again.
///   - **A product's time is its run plus its weight's preparation**: the widened routes (GPU, AMX, ANE, the three) pay
///     the widening of `k · n` values at the CPU's measured rate, the native ones (`gpu/bf16`, `gpu/int8`…) the copy —
///     today's rule ("tensor + copy against the better + widening") as one number per route. A constant, so it
///     weighs on ratios only through the product's own seconds, never through the dispersion.
///   - **The cuts are parameters of a route, balanced when the route is first asked** (`amxShare`, `aneBalance`), kept for
///     the class and written into the table's grids: the shares are measured where the plan decided a pair was worth a
///     measure, not on a grid nobody reads. Each balancing is time of the plan's clock.
///   - **The ANE's engine holds its surfaces until it is released**, so it is released and rebuilt before it would pass
///     the plan's allotment (`MemoryNeedCapacityMap`); the shares survive, the programs do not (`generation`).
///   - **An attention class holds one set of objects at a time** (the last class asked): MPSGraph keeps a copy of the
///     queries per shape until the process ends unless the pool that built it drains, so the objects are built, run and
///     dropped inside an `autoreleasepool` when the class changes.
///   - **Exactness is per route, once**: the first run of a route on a class is judged against the fp64 truth
///     (`exactRows`, `exactAttention`) with the checks' rule — the fp32 threshold, the ANE within 1.25× of bf16's own rounding
///     (invariant 3), bf16-activation routes against the product of the rounded activation. A truth is kept for as
///     long as the cache allows (`CapacityMap.truthCache`).
extension Diagnostic {
    // MARK: - Clock

    struct WallClock: CapacityMeasure.Clock {
        func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) * 1e-9 }
    }

    /// The seconds one execution of `body` takes, `nil` when it reports a refusal.
    private static func timed(_ body: () -> Bool) -> Double? {
        let begin = DispatchTime.now().uptimeNanoseconds
        let ok = body()
        let wall = Double(DispatchTime.now().uptimeNanoseconds - begin) * 1e-9
        return ok ? wall : nil
    }

    /// **The median of `runs` timed executions after two warm-ups, and its relative standard error** (1.253 · σ / √n,
    /// σ = 1.4826 · MAD / median): a witness's figure and what it can tell from noise.
    static func witnessTime(runs: Int = 5, _ body: () -> Double?) -> (seconds: Double, error: Double)? {
        for _ in 0..<2 { guard body() != nil else { return nil } }
        var times: [Double] = []
        for _ in 0..<runs { guard let t = body() else { return nil }; times.append(t) }
        let median = CapacityMeasure.median(times)
        let mad = 1.4826 * CapacityMeasure.median(times.map { abs($0 - median) })
        return (median, median > 0 ? 1.253 * mad / median / Double(runs).squareRoot() : 0)
    }

    // MARK: - The run

    final class MapRun {
        struct Buffers { let a, b, c: MTLBuffer }

        /// The ANE side: a pair's GEMM and a three's, each with its own engine (its own surfaces).
        final class Side {
            let neural: GEMM, neuralBuffers: Buffers
            let three: GEMM?, threeBuffers: Buffers?
            init(neural: GEMM, neuralBuffers: Buffers, three: GEMM?, threeBuffers: Buffers?) {
                self.neural = neural; self.neuralBuffers = neuralBuffers; self.three = three; self.threeBuffers = threeBuffers
            }
        }

        struct State {
            var amxShare: Double?
            var aneShare: Double?
            var amxRefused = false
            var aneRefused = false
            /// The ANE route's bound-on floor: bf16's own rounding of the product (invariant 3).
            var aneFloor: Double?
            /// Per route id: the generation it was warmed in (0: never).
            var warmed: [String: Int] = [:]
            var verdict: [String: Bool] = [:]
        }

        enum Truth: Hashable { case fp32, rounded, int8, q8 }
        struct TruthKey: Hashable { let c: Int; let truth: Truth }

        let bench: Bench
        var entries: [CapacityMap.Entry]
        let operands: Operands
        let allotment: Int?
        let alone: GEMM, both: GEMM
        let aloneBuffers: Buffers, bothBuffers: Buffers, bf16: MTLBuffer
        var caps: CapacityMap.Capabilities
        var states: [State]
        /// Seconds per value: the weight's widening and its copy, by kind.
        let preparation: CapacityMap.Preparation

        var workspace: Pages?
        var side: Side?
        var sideBatches = 0
        var generation = 1
        var held: [GridPoint] = []
        var witnessANE: (gemm: GEMM, buffers: Buffers)?

        var truths: [TruthKey: [Double]] = [:]
        var truthOrder: [TruthKey] = []
        var truthBytes = 0
        var bDirty = false
        var dirtyValues = 0
        var amxStart = 0.631
        /// The cuts already balanced, by product: the same k×n at the same rows in bf16 and in int8 is one balance, not two.
        var cutCache: [[Int]: CapacityMap.Cut] = [:]
        var aneRefusedFor = Set<[Int]>()

        var cancelled = false
        var judgeSeconds = 0.0
        /// The seconds the cuts' balancing took (the AMX's, the ANE's), and the engines' warm-ups: part of the plan's clock.
        var balanceSeconds = 0.0
        var warmSeconds = 0.0
        var wrong: [String] = []

        // The attention's pages and the objects of the class under way.
        var attentionPages: (q: Pages, k: Pages, v: Pages, out: Pages)?
        var attentionBuffers: (q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, out: MTLBuffer)?
        var attentionTruth: [Int: [Double]] = [:]
        var attentionObjects: (c: Int, graph: Attention?, flash: FlashMatrix?, split: HeadSplitAttention?)?

        init(bench: Bench, entries: [CapacityMap.Entry], operands: Operands, allotment: Int?, caps: CapacityMap.Capabilities,
             alone: GEMM, both: GEMM, widening: [Widening]) throws {
            self.bench = bench; self.entries = entries; self.operands = operands; self.allotment = allotment
            self.caps = caps; self.alone = alone; self.both = both
            states = Array(repeating: State(), count: entries.count)
            func wrap(_ g: GEMM) throws -> Buffers {
                Buffers(a: try g.wrap(operands.a.base, bytes: operands.a.capacity, name: "map.a"),
                        b: try g.wrap(operands.b.base, bytes: operands.b.capacity, name: "map.b"),
                        c: try g.wrap(operands.c.base, bytes: operands.c.capacity, name: "map.c"))
            }
            aloneBuffers = try wrap(alone); bothBuffers = try wrap(both)
            bf16 = try alone.wrap(operands.bf16.base, bytes: operands.bf16.capacity, name: "map.bf16")
            func per(_ kind: String) -> Double {
                guard let r = widening.first(where: { $0.kind == kind })?.cpu.rate, r > 0 else { return 0 }
                return 1e-9 / r
            }
            preparation = [.bf16: (per("bf16"), per("bf16 copy")), .int8: (per("int8 convrot"), per("int8 copy")),
                     .q8: (per("q8_0"), per("q8_0 copy"))]
        }

        private func wrap(_ g: GEMM) throws -> Buffers {
            Buffers(a: try g.wrap(operands.a.base, bytes: operands.a.capacity, name: "map.a"),
                    b: try g.wrap(operands.b.base, bytes: operands.b.capacity, name: "map.b"),
                    c: try g.wrap(operands.c.base, bytes: operands.c.capacity, name: "map.c"))
        }

        /// The classes the pairs will decide, once the engines that exist are known (the first guess assumed them all).
        func restrict(to entries: [CapacityMap.Entry]) {
            self.entries = entries
            states = Array(repeating: State(), count: entries.count)
        }

        // MARK: Operands

        /// `B` restored to the fp32 weight the engines multiply (the native routes stage their own form in it, over the first
        /// `dirtyValues` values).
        private func restoreB(count: Int = 0) {
            guard bDirty else { return }
            Widen.bfloat16ToFloat32(source: operands.bf16.base, destination: operands.b.base,
                                    count: min(dirtyValues, operands.bf16.capacity / 2))
            bDirty = false
        }

        // MARK: The ANE side

        private func buildSide() -> Side? {
            guard let neural = try? GEMM(freezeCut: true, coexecution: false, tensorWeights: false, neuralEngine: true, neuralEngineShare: 0.5),
                  neural.neuralEngine != nil, let nb = try? wrap(neural) else { return nil }
            neural.neuralEngine?.shares = .none
            var three: GEMM?, tb: Buffers?
            if caps.amx, let g = try? GEMM(freezeCut: true, coexecution: true, tensorWeights: false, neuralEngine: true, neuralEngineShare: 0.5),
               let b = try? wrap(g) {
                three = g; tb = b
            }
            sideBatches += 1
            generation += 1
            return Side(neural: neural, neuralBuffers: nb, three: three, threeBuffers: tb)
        }

        /// The side for a product at `point`: released and rebuilt first when the engines it holds, plus this point, would
        /// pass the plan's allotment.
        private func sideFor(_ point: GridPoint) -> Side? {
            if let allotment, side != nil, !held.contains(point), !held.isEmpty,
               Diagnostic.engineBytes(held + [point]) > allotment {
                side = nil
                held = []
            }
            if side == nil { side = buildSide() }
            if side != nil, !held.contains(point) { held.append(point) }
            return side
        }

        // MARK: Shares

        /// **The GPU's share beside the AMX that ends both together** — `measureCuts`' balance, once per class: a warm-up, then
        /// the median of three estimates at the share just found.
        private func amxShare(_ c: Int, k: Int, n: Int, m: Int) -> Double? {
            if let s = states[c].amxShare { return s }
            if states[c].amxRefused { return nil }
            if let s = cutCache[[k, n, m]]?.amx { states[c].amxShare = s; return s }
            guard caps.amx, let conductor = both.conductor else { states[c].amxRefused = true; return nil }
            let began = DispatchTime.now().uptimeNanoseconds
            defer { balanceSeconds += Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9 }
            restoreB()
            conductor.shares = .none
            let key = "\(k)×\(n)"
            var share = amxStart
            func balanced() -> Double {
                conductor.initialFraction = share
                // As in a render: the weight was just widened by the CPU, the AMX finds it in the caches it left warm.
                Widen.bfloat16ToFloat32(source: operands.bf16.base, destination: operands.b.base, count: k * n)
                let before = conductor.perShape[key] ?? Conductor.ShapeRecord()
                _ = both.linear(a: bothBuffers.a, b: bothBuffers.b, c: bothBuffers.c, m: m, k: k, n: n, weightIsTransposed: true)
                guard let after = conductor.perShape[key] else { return share }
                let wall = after.wall - before.wall
                let t = Double(Conductor.cut(m, share)) / Double(m)
                let gpu = t / max(1e-9, wall - (after.idleGPU - before.idleGPU))
                let amx = (1 - t) / max(1e-9, wall - (after.idleAMX - before.idleAMX))
                return min(Conductor.fractionRange.upperBound, max(Conductor.fractionRange.lowerBound, gpu / (gpu + amx)))
            }
            _ = balanced()
            share = balanced()
            share = [balanced(), balanced(), balanced()].sorted()[1]
            states[c].amxShare = share
            cutCache[[k, n, m], default: .init()].amx = share
            amxStart = share
            return share
        }

        /// **The GPU's share beside the ANE that ends both together** (`aneBalance`), once per class.
        private func aneShare(_ c: Int, k: Int, n: Int, m: Int) -> Double? {
            if let s = states[c].aneShare { return s }
            if states[c].aneRefused { return nil }
            if let s = cutCache[[k, n, m]]?.ane { states[c].aneShare = s; return s }
            if aneRefusedFor.contains([k, n, m]) { states[c].aneRefused = true; return nil }
            let began = DispatchTime.now().uptimeNanoseconds
            defer { balanceSeconds += Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9 }
            guard caps.ane, let side = sideFor(.init(m: m, k: k, n: n)), let engine = side.neural.neuralEngine,
                  engine.rows(m, k: k, n: n) > 0 else { states[c].aneRefused = true; return nil }
            restoreB()
            let nb = side.neuralBuffers
            let run = { _ = side.neural.linear(a: nb.a, b: nb.b, c: nb.c, m: m, k: k, n: n, weightIsTransposed: true) }
            guard let (share, _) = Diagnostic.aneBalance(side.neural, run: run, m: m, k: k, n: n) else {
                states[c].aneRefused = true; aneRefusedFor.insert([k, n, m]); return nil
            }
            states[c].aneShare = share
            cutCache[[k, n, m], default: .init()].ane = share
            return share
        }

        // MARK: Truths

        private func keep(_ key: TruthKey, _ truth: [Double]) {
            truths[key] = truth
            truthOrder.append(key)
            truthBytes += truth.count * 8
            while truthBytes > CapacityMap.truthCache, truthOrder.count > 1 {
                let old = truthOrder.removeFirst()
                truthBytes -= (truths.removeValue(forKey: old)?.count ?? 0) * 8
            }
        }

        private func truth(_ c: Int, _ kind: Truth, k: Int, n: Int, m: Int) -> [Double] {
            let key = TruthKey(c: c, truth: kind)
            if let t = truths[key] { return t }
            let began = DispatchTime.now().uptimeNanoseconds
            defer { judgeSeconds += Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9 }
            restoreB()
            let rows = sampleRows(m, count: sampledRows)
            let a = operands.a.floats()
            let result: [Double]
            switch kind {
            case .fp32:
                result = exactRows(rows, a: a, b: operands.b.floats(), k: k, n: n)
            case .rounded:
                result = roundedTruth(rows, k: k, n: n)
            case .int8:
                result = exactRows(rows, a: a, q: operands.q.base.assumingMemoryBound(to: Int8.self), scale: Double(int8Scale),
                                   group: int8Group, k: k, n: n)
            case .q8:
                result = exactRows(rows, a: a, q: operands.q.base.assumingMemoryBound(to: Int8.self),
                                   blockScales: operands.q8Scales.floats(), block: q8Block, k: k, n: n)
            }
            keep(key, result)
            return result
        }

        /// The exact product of the sampled rows of `A` rounded to bf16 (`gemm_bf16_act`'s, and the ANE's floor).
        private func roundedTruth(_ rows: [Int], k: Int, n: Int) -> [Double] {
            let a = operands.a.floats()
            var rounded = [Float](repeating: 0, count: rows.count * k)
            for (i, row) in rows.enumerated() { for j in 0..<k { rounded[i * k + j] = Half.bf16.round(a[row * k + j]) } }
            return rounded.withUnsafeBufferPointer {
                exactRows(Array(0..<rows.count), a: $0.baseAddress!, b: operands.b.floats(), k: k, n: n)
            }
        }

        /// bf16's own error on this product: the floor no fp16 path can pass under.
        private func aneFloor(_ c: Int, k: Int, n: Int, m: Int) -> Double {
            if let f = states[c].aneFloor { return f }
            let rows = sampleRows(m, count: sampledRows)
            let witness = roundedTruth(rows, k: k, n: n).map { Double(Half.bf16.round(Float($0))) }
            let exact = truth(c, .fp32, k: k, n: n, m: m)
            let floor = judge(witness.map(Float.init), exact, columns: rows.count, rate: 0, threshold: 1).error
            states[c].aneFloor = floor
            return floor
        }

        // MARK: Products

        func probe(_ c: Int, _ r: Int) -> Double? {
            if cancelled { return nil }
            do { try bench.check() } catch { cancelled = true; return nil }
            switch entries[c].work {
            case let .product(k, n, m): return product(c, r, k: k, n: n, m: m)
            case let .attention(s, heads, dim): return attention(c, r, sequence: s, heads: heads, headDim: dim)
            }
        }

        /// One run of a product's route, seconds — the engines' work alone: a weight's preparation is the view's
        /// (`CapacityMap.Entry.compute(preparation:)`), added by the core.
        private func product(_ c: Int, _ r: Int, k: Int, n: Int, m: Int) -> Double? {
            let route = entries[c].routes[r].id
            let parts = route.split(separator: "/").map(String.init)
            let engines = parts[0], form = parts.count > 1 ? parts[1] : nil
            bench.gauge.mark("routes", "\(entries[c].id) \(route)")
            let rows = sampleRows(m, count: sampledRows)
            let out = operands.c.floats()
            var truthKind = Truth.fp32
            var threshold = fp32Threshold
            var stage: (() -> Void)?
            let run: () -> Bool
            var needsGeneration = false

            switch (engines, form) {
            case ("gpu", nil):
                restoreB()
                let g = alone, b = aloneBuffers
                run = { _ = g.linear(a: b.a, b: b.b, c: b.c, m: m, k: k, n: n, weightIsTransposed: true); return true }
            case ("gpu+amx", nil):
                guard let share = amxShare(c, k: k, n: n, m: m), let conductor = both.conductor else { return nil }
                restoreB()
                let g = both, b = bothBuffers
                run = {
                    conductor.shares = .none
                    conductor.initialFraction = share
                    _ = g.linear(a: b.a, b: b.b, c: b.c, m: m, k: k, n: n, weightIsTransposed: true)
                    return true
                }
            case ("gpu+ane", nil), ("gpu+ane+amx", nil):
                let three = engines == "gpu+ane+amx"
                guard let ane = aneShare(c, k: k, n: n, m: m) else { return nil }
                let amx = three ? amxShare(c, k: k, n: n, m: m) : nil
                if three && amx == nil { return nil }
                guard let side = sideFor(.init(m: m, k: k, n: n)) else { return nil }
                let g = three ? side.three : side.neural, b = three ? side.threeBuffers : side.neuralBuffers
                guard let g, let b, let engine = g.neuralEngine else { return nil }
                restoreB()
                needsGeneration = true
                run = {
                    if three, let amx {
                        engine.gpuShare = Diagnostic.threeWayShare(ane: ane, amx: amx)
                        engine.shares = .none
                        engine.besideAMX = Conductor.Shares([.init(k: k, n: n, m: m, ratio: 1)])
                        g.conductor?.shares = Conductor.Shares([.init(k: k, n: n, m: m, ratio: (1 - amx) / amx)])
                    } else {
                        engine.gpuShare = ane
                    }
                    let before = engine.fallbacks
                    _ = g.linear(a: b.a, b: b.b, c: b.c, m: m, k: k, n: n, weightIsTransposed: true)
                    return engine.fallbacks == before
                }
                threshold = -1          // set below, from bf16's own error
            case ("gpu", "bf16"), ("gpu", "bf16act"):
                guard alone.tensor != nil, alone.tensorBF16 else { return nil }
                let g = alone, b = aloneBuffers, wb = bf16
                if form == "bf16act" {
                    guard alone.bf16Activations, let plan = WeightProducts.roundingPlan(k: k, n: n, capacity: k * n) else { return nil }
                    truthKind = .rounded
                    stage = { [self] in
                        Widen.copyBfloat16(source: operands.bf16.base, destination: operands.b.base, count: k * n)
                        bDirty = true; dirtyValues = max(dirtyValues, k * n)
                    }
                    run = { _ = g.linear(a: b.a, rounding: (b.b, plan.offset, plan.rows), bf16: b.b, c: b.c, m: m, k: k, n: n, weightIsTransposed: true); return true }
                } else {
                    run = { _ = g.linear(a: b.a, bf16: wb, c: b.c, m: m, k: k, n: n, weightIsTransposed: true); return true }
                }
            case ("gpu", "int8"), ("gpu+amx", "int8"), ("gpu", "q8"), ("gpu+amx", "q8"):
                let isInt8 = form == "int8"
                guard (isInt8 ? both.tensorInt8 : both.tensorQ8), (k * n) % 2 == 0 else { return nil }
                guard isInt8 ? k % int8Group == 0 : k % q8Block == 0 else { return nil }
                var amxColumns = 0
                if engines == "gpu+amx" {
                    guard m >= Conductor.minimumRows, let share = amxShare(c, k: k, n: n, m: m) else { return nil }
                    amxColumns = WeightProducts.amxColumns(n: n, share: share)
                }
                guard let plan = WeightProducts.int8Plan(k: k, n: n, capacity: k * n, perColumn: false, group: isInt8 ? int8Group : nil,
                                                         amxColumns: amxColumns, block: isInt8 ? nil : q8Block) else { return nil }
                truthKind = isInt8 ? .int8 : .q8
                let g = both, b = bothBuffers
                stage = { [self] in
                    Widen.copyBfloat16(source: operands.q.base, destination: operands.b.base, count: k * n / 2)
                    if isInt8 {
                        (operands.b.base + plan.scalesOffset).storeBytes(of: int8Scale, as: Float.self)
                    } else {
                        (operands.b.base + plan.scalesOffset).copyMemory(from: operands.q8Scales.base, byteCount: (k + q8Block - 1) / q8Block * n * 4)
                    }
                    bDirty = true; dirtyValues = max(dirtyValues, k * n)
                }
                run = {
                    WeightProducts.int8Product(gemm: g, plan: plan, reserve: b.b, a: b.a, c: b.c, m: m, k: k, n: n).status == 0
                }
            default:
                return nil
            }

            // The truth first (it restores `B` for the fp32 ones), then the staged weight.
            let judgedBefore = states[c].verdict[route]
            let warmedIn = states[c].warmed[route] ?? 0
            let needsWarm = warmedIn == 0 || (needsGeneration && warmedIn != generation)
            var exact: [Double]?
            if judgedBefore == nil {
                exact = truth(c, truthKind, k: k, n: n, m: m)
                if threshold < 0 { threshold = max(fp32Threshold, floorMargin * aneFloor(c, k: k, n: n, m: m)) }
            }
            if let stage { stage() } else if form == nil { restoreB() }
            if needsWarm || judgedBefore == nil {
                poison(out, rows: rows, n: n)
                let warmBegan = DispatchTime.now().uptimeNanoseconds
                defer { warmSeconds += Double(DispatchTime.now().uptimeNanoseconds - warmBegan) * 1e-9 }
                guard run() else { states[c].verdict[route] = judgedBefore ?? false; return nil }
                states[c].warmed[route] = generation
                if let exact {
                    let began = DispatchTime.now().uptimeNanoseconds
                    let measure = judge(gather(out, rows: rows, n: n), exact, columns: rows.count, rate: 0, threshold: threshold)
                    judgeSeconds += Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9
                    states[c].verdict[route] = measure.pass
                    if !measure.pass { wrong.append("\(entries[c].id) \(route)"); return nil }
                }
            }
            if states[c].verdict[route] == false { return nil }
            return Diagnostic.timed(run)
        }

        // MARK: Attention

        private func attention(_ c: Int, _ r: Int, sequence s: Int, heads h: Int, headDim d: Int) -> Double? {
            let route = entries[c].routes[r].id
            bench.gauge.mark("routes", "\(entries[c].id) \(route)")
            return autoreleasepool { () -> Double? in
                guard let buffers = try? attentionBuffersFor(maxFloats: s * h * d) else { return nil }
                if attentionObjects?.c != c { attentionObjects = (c, nil, nil, nil) }
                let out = attentionPages!.out.floats()
                let rows = sampleRows(s, count: sampledQueries)
                let run: () -> Bool
                switch route {
                case "graph":
                    if attentionObjects?.graph == nil {
                        attentionObjects?.graph = Attention(device: bench.device, queue: bench.queue, heads: h, sequence: s, headDim: d)
                    }
                    guard let g = attentionObjects?.graph else { return nil }
                    run = { _ = g.run(q: buffers.q, k: buffers.k, v: buffers.v, into: buffers.out); return true }
                case "flash":
                    if attentionObjects?.flash == nil {
                        attentionObjects?.flash = try? FlashMatrix(device: bench.device, queue: bench.queue, heads: h, sequence: s, headDim: d)
                    }
                    guard let f = attentionObjects?.flash else { return nil }
                    run = { _ = f.run(q: buffers.q, k: buffers.k, v: buffers.v, into: buffers.out); return true }
                case "split":
                    if attentionObjects?.split == nil {
                        attentionObjects?.split = try? HeadSplitAttention(device: bench.device, queue: bench.queue, heads: h, sequence: s,
                                                                         headDim: d, cpuFraction: HeadSplitAttention.referenceFraction)
                    }
                    guard let cut = attentionObjects?.split, let p = attentionPages else { return nil }
                    let (q, k, v) = (p.q.floats(), p.k.floats(), p.v.floats())
                    nonisolated(unsafe) let (qp, kp, vp, op) = (q, k, v, out)
                    run = { _ = cut.run(q: buffers.q, k: buffers.k, v: buffers.v, into: buffers.out, q: qp, k: kp, v: vp, out: op); return true }
                default:
                    return nil
                }
                let width = h * d
                let judged = states[c].verdict[route]
                let warmed = states[c].warmed[route] ?? 0
                if warmed == 0 || judged == nil {
                    for row in rows { (out + row * width).update(repeating: .nan, count: width) }
                    _ = run()
                    states[c].warmed[route] = generation
                    if judged == nil {
                        let began = DispatchTime.now().uptimeNanoseconds
                        if attentionTruth[c] == nil {
                            let p = attentionPages!
                            attentionTruth[c] = exactAttention(q: p.q.floats(), k: p.k.floats(), v: p.v.floats(), sequence: s, heads: h,
                                                               headDim: d, rows: rows)
                        }
                        let vectors = rows.count * h
                        var got = [Float](repeating: 0, count: vectors * d)
                        for (i, row) in rows.enumerated() {
                            for head in 0..<h { for x in 0..<d { got[x * vectors + i * h + head] = out[row * width + head * d + x] } }
                        }
                        let measure = judge(got, attentionTruth[c]!, columns: vectors, rate: 0, threshold: fp32Threshold)
                        judgeSeconds += Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9
                        states[c].verdict[route] = measure.pass
                        if !measure.pass { wrong.append("\(entries[c].id) \(route)") }
                    }
                }
                if states[c].verdict[route] == false { return nil }
                return Diagnostic.timed(run)
            }
        }

        private func attentionBuffersFor(maxFloats: Int) throws -> (q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, out: MTLBuffer) {
            if let b = attentionBuffers { return b }
            let floats = entries.compactMap { e -> Int? in
                if case let .attention(s, h, d) = e.work { return s * h * d } else { return nil }
            }.max() ?? maxFloats
            let pages = (q: Pages(bytes: floats * 4), k: Pages(bytes: floats * 4), v: Pages(bytes: floats * 4), out: Pages(bytes: floats * 4))
            fill(pages.q.floats(), count: floats, scale: 3, seed: 11)
            fill(pages.k.floats(), count: floats, scale: 3, seed: 12)
            fill(pages.v.floats(), count: floats, scale: 1, seed: 13)
            attentionPages = pages
            let buffers = (q: try bench.buffer(pages.q), k: try bench.buffer(pages.k), v: try bench.buffer(pages.v), out: try bench.buffer(pages.out))
            attentionBuffers = buffers
            return buffers
        }

        /// The attention's objects and pages are released before the end witnesses and the report.
        func releaseAttention() {
            autoreleasepool {
                attentionObjects = nil
                attentionBuffers = nil
                attentionPages = nil
            }
        }

        // MARK: Witnesses

        /// **One round of witnesses**: the same product (`CapacityMap.witnessShape`) on each engine, median of five. The GPU
        /// alone, the AMX alone (BNNS), the ANE's side of a pair at a fixed share. An engine that refuses is not witnessed.
        func witnesses() -> [String: (seconds: Double, error: Double)] {
            var result: [String: (seconds: Double, error: Double)] = [:]
            let (m, k, n) = CapacityMap.witnessShape
            restoreB()
            let b = aloneBuffers
            if let w = Diagnostic.witnessTime({ Diagnostic.timed { _ = self.alone.linear(a: b.a, b: b.b, c: b.c, m: m, k: k, n: n, weightIsTransposed: true); return true } }) {
                result["gpu"] = w
            }
            if caps.amx {
                var refused = false
                let w = Diagnostic.witnessTime {
                    Diagnostic.timed {
                        let status = Conductor.multiply(a: self.operands.a.base, b: self.operands.b.base, c: self.operands.c.base, rows: m, k: k, n: n,
                                                        transposeWeight: false, workspace: &self.workspace).status
                        if status != 0 { refused = true }
                        return true
                    }
                }
                if let w, !refused { result["amx"] = w } else { caps.amx = false }
            }
            if caps.ane {
                if witnessANE == nil, let g = try? GEMM(freezeCut: true, coexecution: false, tensorWeights: false, neuralEngine: true, neuralEngineShare: 0.5),
                   g.neuralEngine != nil, let buffers = try? wrap(g) {
                    g.neuralEngine?.shares = .none
                    witnessANE = (g, buffers)
                }
                if let (g, buffers) = witnessANE, let engine = g.neuralEngine, engine.rows(m, k: k, n: n) > 0 {
                    var refused = false
                    let w = Diagnostic.witnessTime {
                        let before = engine.fallbacks
                        _ = g.linear(a: buffers.a, b: buffers.b, c: buffers.c, m: m, k: k, n: n, weightIsTransposed: true)
                        if engine.fallbacks != before { refused = true }
                        return g.lastSides?.ane
                    }
                    if let w, !refused { result["ane"] = w } else { caps.ane = false }
                } else { caps.ane = false }
            }
            return result
        }
    }
}

// MARK: - The run

extension Diagnostic {
    private static func flashAvailable(_ bench: Bench) -> Bool {
        autoreleasepool { (try? FlashMatrix(device: bench.device, queue: bench.queue, heads: 1, sequence: 1024, headDim: 128)) != nil }
    }

    /// The witnesses of the two rounds, joined by engine.
    static func engineWitnesses(start: [String: (seconds: Double, error: Double)],
                                end: [String: (seconds: Double, error: Double)]) -> [CapacityMap.EngineWitness] {
        ["gpu", "amx", "ane"].compactMap { engine in
            guard let a = start[engine], let b = end[engine] else { return nil }
            return CapacityMap.EngineWitness(engine: engine, before: a.seconds, after: b.seconds, errorBefore: a.error, errorAfter: b.error)
        }
    }

    static func runQueued(cancellation: Cancellation?, onProgress: (@Sendable (_ stage: Stage) -> Void)?) throws(EngineError) -> Report {
        let started = Date()
        let clock = WallClock()
        let t0 = clock.now()
        let lock = try RenderLock.acquire(RenderLock.machine)
        defer { withExtendedLifetime(lock) {} }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw .internalFailure(component: "diagnostic", detail: "no Metal device")
        }
        // The frameworks' first use (Metal, MPSGraph, Core ML) is the process's base, as a render's (`OperatorFootprint.bringUp`).
        OperatorFootprint.bringUp()
        _ = NeuralEngine.bringUp
        let gauge = Gauge()
        let settings = EngineSettings.effective
        let limits = CapacityMap.Limits(amxRows: settings.amxMinimumRows, aneRows: settings.aneMinimumRows,
                                        roundingRows: settings.bf16ActivationMinimumRows)
        // The memory, before the first computation: the grid with every engine assumed present (the largest need).
        let everything = CapacityMap.Capabilities(amx: true, ane: true, bf16Weights: true, bf16Rounding: true, int8Weights: true,
                                                  q8Weights: true, flash: true, headSplit: true)
        let widest = CapacityMap.catalogue(caps: everything, limits: limits)
        let plan = try Plan.choose(budget: MemoryBudget.machineCeiling(), base: gauge.base) {
            CapacityMap.need(entries: widest, allotment: $0)
        }
        let before = conditions()
        var stages: [String: Double] = [:]
        do {
            let bench = Bench(device: device, queue: queue, cancellation: cancellation, gauge: gauge)

            // ── the preparation's rates ──
            onProgress?(.widening)
            gauge.mark("widening")
            var lap = clock.now()
            let widening = try measureWidening(bench)
            stages["widening"] = clock.now() - lap

            // ── the engines, the operands, the first witnesses ──
            onProgress?(.witness)
            gauge.mark("routes", "operands")
            lap = clock.now()
            let alone = try GEMM(freezeCut: true, coexecution: false, tensorWeights: true, bf16Activations: true)
            let both = try GEMM(freezeCut: true, coexecution: true, tensorWeights: false, int8Weights: true, q8Weights: true)
            let operands = Operands(Operands.sizes(products: CapacityMap.products(widest)))
            let caps = CapacityMap.Capabilities(amx: both.conductor != nil, ane: true, bf16Weights: alone.tensorBF16,
                                                bf16Rounding: alone.bf16Activations, int8Weights: both.tensorInt8, q8Weights: both.tensorQ8,
                                                flash: flashAvailable(bench), headSplit: true)
            let run = try MapRun(bench: bench, entries: widest, operands: operands, allotment: plan.aneAllotment, caps: caps,
                                 alone: alone, both: both, widening: widening)
            gauge.mark("routes", "witness")
            let first = run.witnesses()
            let witnessSeconds = clock.now() - lap
            stages["witness"] = witnessSeconds
            // What exists here: the engines that answered, the kernels that were built.
            let entries = CapacityMap.catalogue(caps: run.caps, limits: limits)
            run.restrict(to: entries)
            let classes = entries.map { $0.compute(preparation: run.preparation) }

            // ── the plan: 60 s of wall for everything, minus what the end needs ──
            var measuring = CapacityMeasure.Settings()
            let reserve = witnessSeconds + 2.0
            measuring.budget = max(5, CapacityMap.budget - (clock.now() - t0) - reserve)
            var last: [String: (seconds: Double, error: Double)] = [:]
            onProgress?(.routes)
            lap = clock.now()
            let result = CapacityMeasure.plan(classes: classes, settings: measuring, finalHeat: {
                onProgress?(.regime)
                run.releaseAttention()
                gauge.mark("routes", "witness end")
                let begin = clock.now()
                last = run.witnesses()
                stages["regime"] = clock.now() - begin
                return CapacityMap.heat(engineWitnesses(start: first, end: last))
            }, clock: clock, probe: { run.probe($0, $1) })
            stages["routes"] = clock.now() - lap - (stages["regime"] ?? 0)
            stages["judge"] = run.judgeSeconds
            stages["balance"] = run.balanceSeconds
            stages["warm"] = run.warmSeconds
            if run.cancelled { throw EngineError.cancelled }
            gauge.finish()

            // ── the table ──
            let table = CapacityMap.table(plan: result, entries: entries, classes: classes,
                                          cuts: run.states.map { .init(amx: $0.amxShare, ane: $0.aneShare) }, limits: limits)

            // ── the report ──
            // One row per decision: a point of the grid, a product's once per weight form.
            let decided = classes.flatMap(\.decisionClasses)
            let total = max(decided.reduce(0) { $0 + $1.weight }, 1e-12)
            let rows = zip(decided, result.decisions).map { c, d -> MapClass in
                let timings = c.routes.indices.compactMap { r -> MapClass.Timing? in
                    guard let ratio = d.hotRatios[r], r != c.routes.firstIndex(where: \.correct) else { return nil }
                    return .init(route: c.routes[r].id, ratio: ratio, error: d.spread.indices.contains(r) ? d.spread[r] ?? 0 : 0)
                }
                return MapClass(id: c.id, weight: c.weight / total, chosen: d.chosen, loss: d.loss, runs: d.runs,
                                status: d.status, timings: timings)
            }
            let map = MapResult(classes: rows, residualLoss: result.residualLoss, epsilon: measuring.epsilon, spent: result.spent,
                                budget: measuring.budget, exhausted: result.exhausted, judged: run.judgeSeconds, wrong: run.wrong)
            let witnesses = engineWitnesses(start: first, end: last)
            let memory = Memory(plan: plan, peaks: gauge.peaks, aneBatches: run.sideBatches)
            let seconds = Date().timeIntervalSince(started)
            let verdict = CapacityMap.Verdict(residualLoss: result.residualLoss, epsilon: measuring.epsilon,
                                              statuses: result.decisions.map(\.status), wrongChosen: false, memoryHeld: memory.held,
                                              seconds: seconds)
            let flops = 2.0 * Double(CapacityMap.witnessShape.m * CapacityMap.witnessShape.k * CapacityMap.witnessShape.n) * 1e-12
            func rate(_ w: CapacityMap.EngineWitness?, _ time: KeyPath<CapacityMap.EngineWitness, Double>) -> Double? { w.map { flops / $0[keyPath: time] } }
            let gpu = witnesses.first { $0.engine == "gpu" }, amx = witnesses.first { $0.engine == "amx" }
            let witness = Witness(gpuBefore: rate(gpu, \.before) ?? 0, gpuAfter: rate(gpu, \.after) ?? 0,
                                  amxBefore: rate(amx, \.before), amxAfter: rate(amx, \.after))
            var forced: [String: String] = [:]
            for row in settings.rows where row.source != .byDefault { forced[row.key] = "\(row.value) (\(row.source.rawValue))" }
            return Report(date: ISO8601DateFormatter().string(from: started), seconds: seconds, machine: machine(),
                          cpuFeatures: cpuFeatures(), settings: forced, before: before, after: conditions(), products: [],
                          lowPrecision: [], attention: [], widening: widening, witness: witness, stages: stages, memory: memory,
                          routes: table, map: map, regime: Regime(witnesses: witnesses), verdict: verdict)
        } catch {
            throw EngineError(error)
        }
    }
}
