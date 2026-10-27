// The fp32 products' measurements: GPU, AMX, both, and the bf16, int8, Q8_0 and ANE
// weight paths, with the cuts' grid. Split from `DiagnosticProbes.swift`.

import Accelerate
internal import AMXLegacy
import Foundation
import Metal
import MetalPerformanceShadersGraph

extension Diagnostic {
    // ── the fp32 products ───────────────────────────────────────────────────────────────────

    static func measureProducts(_ o: Operands, _ bench: Bench) throws -> [Product] {
        let alone = try GEMM(freezeCut: true, coexecution: false, tensorWeights: true, bf16Activations: true)
        let both = try GEMM(freezeCut: true, coexecution: true, tensorWeights: false, int8Weights: true, q8Weights: true)
        let neural = try GEMM(freezeCut: true, coexecution: false, tensorWeights: false, neuralEngine: true, neuralEngineShare: 0.5)
        neural.neuralEngine?.shares = .none
        func wrap(_ g: GEMM) throws -> (MTLBuffer, MTLBuffer, MTLBuffer) {
            (try g.wrap(o.a.base, bytes: o.a.capacity, name: "diagnostic.a"),
             try g.wrap(o.b.base, bytes: o.b.capacity, name: "diagnostic.b"),
             try g.wrap(o.c.base, bytes: o.c.capacity, name: "diagnostic.c"))
        }
        let (aloneBuffers, bothBuffers, neuralBuffers) = (try wrap(alone), try wrap(both), try wrap(neural))
        let bf16 = try alone.wrap(o.bf16.base, bytes: o.bf16.capacity, name: "diagnostic.bf16")
        var workspace: Pages?
        var products: [Product] = []
        for s in shapes {
            try bench.check()
            let rows = sampleRows(s.m, count: sampledRows)
            let exact = exactRows(rows, a: o.a.floats(), b: o.b.floats(), k: s.k, n: s.n)
            let c = o.c.floats()
            let flops = 2.0 * Double(s.m) * Double(s.k) * Double(s.n)
            func measure(against truth: [Double]? = nil, giveUpAbove limit: Double? = nil, _ run: () -> Void) -> (Measure, Double) {
                poison(c, rows: rows, n: s.n)
                let t = seconds(giveUpAbove: limit, run)
                return (judge(gather(c, rows: rows, n: s.n), truth ?? exact, columns: rows.count, rate: flops / t / 1e12,
                              threshold: fp32Threshold), t)
            }
            let (gpu, tGPU) = measure {
                _ = alone.linear(a: aloneBuffers.0, b: aloneBuffers.1, c: aloneBuffers.2, m: s.m, k: s.k, n: s.n,
                                 weightIsTransposed: true)
            }
            var refused = false
            let (amxMeasure, tAMX) = measure {
                let status = Conductor.multiply(a: o.a.base, b: o.b.base, c: o.c.base, rows: s.m, k: s.k, n: s.n,
                                                transposeWeight: false, workspace: &workspace).status
                if status != 0 { refused = true }
            }
            let amx = refused ? nil : amxMeasure
            // The GPU's share of the rows that equalizes the two engines' times: `T*` of the servo.
            let share = amx.map { _ in tAMX / (tGPU + tAMX) }
            var coexecution: Measure?, gpuRows: Int?
            if let share, let conductor = both.conductor {
                try bench.check()
                conductor.initialFraction = share
                coexecution = measure {
                    _ = both.linear(a: bothBuffers.0, b: bothBuffers.1, c: bothBuffers.2, m: s.m, k: s.k, n: s.n,
                                    weightIsTransposed: true)
                }.0
                gpuRows = s.m >= Conductor.minimumRows ? Conductor.cut(s.m, share) : s.m
            }
            // ── gemm_ane: the GPU on the first rows, the Neural Engine on the last ── At the share that ends both
            // together (`aneBalance`), judged against bf16's error (invariant 3), not the fp32 threshold.
            var ane: Measure?, aneShare: Double?
            if let engine = neural.neuralEngine, engine.rows(s.m, k: s.k, n: s.n) > 0 {
                try bench.check()
                func run() { _ = neural.linear(a: neuralBuffers.0, b: neuralBuffers.1, c: neuralBuffers.2, m: s.m, k: s.k, n: s.n,
                                               weightIsTransposed: true) }
                if let share = aneBalance(neural, run: run, m: s.m, k: s.k, n: s.n)?.share {
                    engine.gpuShare = share
                    let before = engine.fallbacks
                    poison(c, rows: rows, n: s.n)
                    let t = seconds(run)
                    // bf16's error: the sampled rows of A rounded to bf16, times B (bf16 values), rounded to bf16.
                    let a = o.a.floats()
                    var rounded = [Float](repeating: 0, count: rows.count * s.k)
                    for (i, row) in rows.enumerated() {
                        for j in 0..<s.k { rounded[i * s.k + j] = Half.bf16.round(a[row * s.k + j]) }
                    }
                    let witness = rounded.withUnsafeBufferPointer {
                        exactRows(Array(0..<rows.count), a: $0.baseAddress!, b: o.b.floats(), k: s.k, n: s.n)
                    }.map { Half.bf16.round(Float($0)) }
                    let floor = judge(witness, exact, columns: rows.count, rate: 0, threshold: 1).error
                    if engine.fallbacks == before {
                        ane = judge(gather(c, rows: rows, n: s.n), exact, columns: rows.count, rate: flops / t / 1e12,
                                    threshold: max(fp32Threshold, floorMargin * floor))
                        aneShare = engine.gpuShare
                    }
                }
            }
            var tensor: Measure?
            if alone.tensor != nil {
                try bench.check()
                tensor = measure(giveUpAbove: weightPathGiveUp * tGPU) {
                    _ = alone.linear(a: aloneBuffers.0, bf16: bf16, c: aloneBuffers.2, m: s.m, k: s.k, n: s.n,
                                     weightIsTransposed: true)
                }.0
            }
            // ── gemm_bf16_act: the activation rounded to bf16 just before the product ── The weight staged in the
            // head of `b`, the rounded rows in its tail, as a model's reserve holds them (`WeightProducts.roundingPlan`);
            // judged against the exact product of the rounded activation: it checks the accumulation and the
            // kernel, not what the rounding costs (the check's, against diffusers in bf16). `b` restored from `bf16`.
            var bf16Act: Measure?
            if alone.bf16Activations, let plan = WeightProducts.roundingPlan(k: s.k, n: s.n, capacity: s.k * s.n) {
                try bench.check()
                let a = o.a.floats()
                var roundedRows = [Float](repeating: 0, count: rows.count * s.k)
                for (i, row) in rows.enumerated() {
                    for j in 0..<s.k { roundedRows[i * s.k + j] = Half.bf16.round(a[row * s.k + j]) }
                }
                let truth = roundedRows.withUnsafeBufferPointer {
                    exactRows(Array(0..<rows.count), a: $0.baseAddress!, b: o.b.floats(), k: s.k, n: s.n)
                }
                Widen.copyBfloat16(source: o.bf16.base, destination: o.b.base, count: s.k * s.n)
                bf16Act = measure(against: truth, giveUpAbove: weightPathGiveUp * tGPU) {
                    _ = alone.linear(a: aloneBuffers.0, rounding: (aloneBuffers.1, plan.offset, plan.rows), bf16: aloneBuffers.1,
                                     c: aloneBuffers.2, m: s.m, k: s.k, n: s.n, weightIsTransposed: true)
                }.0
                Widen.bfloat16ToFloat32(source: o.bf16.base, destination: o.b.base, count: s.k * s.n)
            }
            // ── gemm_int8 and gemm_q8: the published 8-bit weight read as it is, alone, then beside the AMX on
            // its columns ── `WeightProducts.int8Product`, the engine's own call, the weight staged in `b` as in
            // a model's reserve of `k · n` floats; `b` restored from `bf16` after.
            func eightBit(group: Int?, block: Int?, truth: [Double]) throws -> (Measure?, Measure?, Int?) {
                var (alone, beside, columns): (Measure?, Measure?, Int?) = (nil, nil, nil)
                func run(_ plan: WeightProducts.Int8Plan) {
                    if WeightProducts.int8Product(gemm: both, plan: plan, reserve: bothBuffers.1, a: bothBuffers.0,
                                                  c: bothBuffers.2, m: s.m, k: s.k, n: s.n).status != 0 {
                        poison(c, rows: rows, n: s.n)
                    }
                }
                func stage(_ plan: WeightProducts.Int8Plan) {
                    Widen.copyBfloat16(source: o.q.base, destination: o.b.base, count: s.k * s.n / 2)
                    if let block {
                        (o.b.base + plan.scalesOffset).copyMemory(from: o.q8Scales.base, byteCount: (s.k + block - 1) / block * s.n * 4)
                    } else {
                        (o.b.base + plan.scalesOffset).storeBytes(of: int8Scale, as: Float.self)
                    }
                }
                func plan(_ amxColumns: Int) -> WeightProducts.Int8Plan? {
                    WeightProducts.int8Plan(k: s.k, n: s.n, capacity: s.k * s.n, perColumn: false, group: group,
                                            amxColumns: amxColumns, block: block)
                }
                if let p = plan(0) {
                    stage(p)
                    alone = measure(against: truth, giveUpAbove: weightPathGiveUp * tGPU) { run(p) }.0
                }
                if let share, both.conductor != nil, s.m >= Conductor.minimumRows,
                   let p = plan(WeightProducts.amxColumns(n: s.n, share: share)) {
                    try bench.check()
                    stage(p)
                    beside = measure(against: truth, giveUpAbove: weightPathGiveUp * tGPU) { run(p) }.0
                    columns = s.n - p.amxColumns
                }
                Widen.bfloat16ToFloat32(source: o.bf16.base, destination: o.b.base, count: s.k * s.n)
                return (alone, beside, columns)
            }
            let q = o.q.base.assumingMemoryBound(to: Int8.self)
            var int8: Measure?, int8Coexecution: Measure?, int8GPUColumns: Int?
            if both.tensorInt8, s.k % int8Group == 0, (s.k * s.n) % 2 == 0 {
                try bench.check()
                let truth = exactRows(rows, a: o.a.floats(), q: q, scale: Double(int8Scale), group: int8Group, k: s.k, n: s.n)
                (int8, int8Coexecution, int8GPUColumns) = try eightBit(group: int8Group, block: nil, truth: truth)
            }
            var q8: Measure?, q8Coexecution: Measure?, q8GPUColumns: Int?
            if both.tensorQ8, s.k % q8Block == 0, (s.k * s.n) % 2 == 0 {
                try bench.check()
                let truth = exactRows(rows, a: o.a.floats(), q: q, blockScales: o.q8Scales.floats(), block: q8Block, k: s.k, n: s.n)
                (q8, q8Coexecution, q8GPUColumns) = try eightBit(group: nil, block: q8Block, truth: truth)
            }
            products.append(Product(model: s.model, product: s.product, m: s.m, k: s.k, n: s.n, gpu: gpu, amx: amx,
                                    gpuShare: share, coexecution: coexecution, coexecutionGPURows: gpuRows, tensor: tensor, bf16Act: bf16Act,
                                    int8: int8, int8Coexecution: int8Coexecution, int8GPUColumns: int8GPUColumns,
                                    q8: q8, q8Coexecution: q8Coexecution, q8GPUColumns: q8GPUColumns,
                                    ane: ane, aneGPUShare: aneShare))
        }
        return products
    }

    /// **The GPU's share beside the ANE that ends both sides together**, for `run` (one `m × k → n` product
    /// through `neural`). Each side's wall is affine in its rows — the ANE reads the whole weight at every
    /// call, whatever its rows —: three runs at ½ and three at ¾, the medians give both lines, and the share
    /// where they cross, in `NeuralEngine.shareRange`. And the GPU alone's wall on all `m` rows, read off the
    /// same fit (`gpuFixed + gpuSlope · m`) — no run of its own. `nil` when the ANE fell back.
    static func aneBalance(_ neural: GEMM, run: () -> Void, m: Int, k: Int, n: Int) -> (share: Double, alone: Double)? {
        guard let engine = neural.neuralEngine else { return nil }
        func sides(_ share: Double) -> (gpu: Double, gpuRows: Double, ane: Double, aneRows: Double)? {
            engine.gpuShare = share
            run()
            var walls: [(gpu: Double, ane: Double)] = []
            for _ in 0..<3 {
                let before = engine.fallbacks
                run()
                guard let last = neural.lastSides, engine.fallbacks == before else { return nil }
                walls.append((last.gpu, last.ane))
            }
            let rows = engine.rows(m, k: k, n: n)
            return (walls.map(\.gpu).sorted()[1], Double(m - rows), walls.map(\.ane).sorted()[1], Double(rows))
        }
        guard let half = sides(0.5), let quarter = sides(0.75), half.aneRows != quarter.aneRows else { return nil }
        let gpuSlope = (half.gpu - quarter.gpu) / (half.gpuRows - quarter.gpuRows)
        let aneSlope = (half.ane - quarter.ane) / (half.aneRows - quarter.aneRows)
        let (gpuFixed, aneFixed) = (half.gpu - gpuSlope * half.gpuRows, half.ane - aneSlope * half.aneRows)
        // gpuFixed + gpuSlope · (m − r) = aneFixed + aneSlope · r
        let r = (gpuFixed + gpuSlope * Double(m) - aneFixed) / (gpuSlope + aneSlope)
        let range = NeuralEngine.shareRange
        let alone = gpuFixed + gpuSlope * Double(m)
        return (r.isFinite ? min(range.upperBound, max(range.lowerBound, 1 - r / Double(m))) : range.upperBound, alone)
    }

    /// **The GPU's share beside the ANE on its grid** (`aneCuts` → `ane_shares`): each `aneShapes`
    /// product at each of `heights`, balanced by `aneBalance`, then three walls: the pair at that share and the
    /// GPU beside the AMX at the share its grid (`cuts`) gives, each the median of three runs, and the GPU alone
    /// read off the balance's own fit (a shape among `products`: their share and rates, measured already) —
    /// the routes `CapacityRules` chooses between, shape by shape. The share is checked, not trusted: below
    /// 1 024 rows the ANE leaves the line `aneBalance` draws (Krea 2's 16 384 → 6 144, K in two parts: 768 rows
    /// at 319 ms beside a GPU done in 240, the pair slower than the GPU alone, 23/10/2026). Not judged: the
    /// products judge the ANE's arithmetic, which no share changes. Empty when no ANE program runs on this Mac.
    static func measureANECuts(_ o: Operands, _ bench: Bench, cuts amxCuts: [Cut], products: [Product],
                               threeSeconds: inout Double) throws -> [Cut] {
        let neural = try GEMM(freezeCut: true, coexecution: false, tensorWeights: false, neuralEngine: true, neuralEngineShare: 0.5)
        guard let engine = neural.neuralEngine else { return [] }
        engine.shares = .none
        let a = try neural.wrap(o.a.base, bytes: o.a.capacity, name: "diagnostic.a")
        let b = try neural.wrap(o.b.base, bytes: o.b.capacity, name: "diagnostic.b")
        let c = try neural.wrap(o.c.base, bytes: o.c.capacity, name: "diagnostic.c")
        // The AMX's route as a render takes it: its conductor frozen at its grid's share for the shape.
        let amx = amxCuts.isEmpty ? nil : try GEMM(freezeCut: true, coexecution: true, tensorWeights: false, neuralEngine: false)
        if let conductor = amx?.conductor {
            conductor.shares = Conductor.Shares(amxCuts.map { .init(k: $0.k, n: $0.n, m: $0.m, ratio: (1 - $0.share) / $0.share) })
        }
        let operands = try amx.map { g in (try g.wrap(o.a.base, bytes: o.a.capacity, name: "diagnostic.a"),
                                           try g.wrap(o.b.base, bytes: o.b.capacity, name: "diagnostic.b"),
                                           try g.wrap(o.c.base, bytes: o.c.capacity, name: "diagnostic.c")) }
        // The three at once: the ANE on its rows, the GPU beside the AMX on the rest (`ane_amx`), at the share
        // both pairs give — one wall per point, no sweep (on the M1 Pro, 26/10/2026, the derived split lands on a sweep's best).
        let three = amx == nil ? nil : try GEMM(freezeCut: true, coexecution: true, tensorWeights: false, neuralEngine: true,
                                                 neuralEngineShare: 0.5)
        if let conductor = three?.conductor, let grid = amx?.conductor?.shares { conductor.shares = grid }
        let threeOperands = try three.map { g in (try g.wrap(o.a.base, bytes: o.a.capacity, name: "diagnostic.a"),
                                                  try g.wrap(o.b.base, bytes: o.b.capacity, name: "diagnostic.b"),
                                                  try g.wrap(o.c.base, bytes: o.c.capacity, name: "diagnostic.c")) }
        /// `best`: the faster pair's wall (ms) — the three are given up past it.
        func threeWall(k: Int, n: Int, m: Int, share: Double, best: Double) -> Double? {
            guard let g = three, let side = g.neuralEngine, let (a, b, c) = threeOperands,
                  let amxShare = amx?.conductor?.shares.share(k: k, n: n, m: m) else { return nil }
            side.gpuShare = threeWayShare(ane: share, amx: amxShare)
            side.shares = .none
            side.besideAMX = Conductor.Shares([.init(k: k, n: n, m: m, ratio: 1)])
            let before = side.fallbacks, began = Date()
            defer { threeSeconds += Date().timeIntervalSince(began) }
            let t = seconds(giveUpAbove: best * 1e-3) { _ = g.linear(a: a, b: b, c: c, m: m, k: k, n: n, weightIsTransposed: true) }
            return side.fallbacks == before ? t * 1e3 : nil
        }
        var cuts: [Cut] = []
        for shape in aneShapes {
            for m in heights {
                try bench.check()
                // A product the map measured already (Z-Image's and Qwen-Image-2.1's MLP): its share and its rates.
                if let p = products.first(where: { $0.k == shape.k && $0.n == shape.n && $0.m == m }),
                   let share = p.aneGPUShare, let ane = p.ane?.rate, ane > 0, p.gpu.rate > 0 {
                    let flops = 2 * Double(m * shape.k * shape.n) * 1e-9   // TFLOP/s → milliseconds
                    let beside = amx != nil ? p.coexecution.map { flops / $0.rate } : nil
                    cuts.append(Cut(k: shape.k, n: shape.n, m: m, share: share, ane: flops / ane, gpu: flops / p.gpu.rate,
                                    amx: beside, three: beside.flatMap { threeWall(k: shape.k, n: shape.n, m: m, share: share,
                                                                                   best: min($0, flops / ane)) }))
                    continue
                }
                let run = { _ = neural.linear(a: a, b: b, c: c, m: m, k: shape.k, n: shape.n, weightIsTransposed: true) }
                guard let (share, alone) = aneBalance(neural, run: run, m: m, k: shape.k, n: shape.n) else { return [] }
                engine.gpuShare = share
                let paired = seconds(run)
                // Past the ANE's tolerance the AMX's pair cannot take the shape: its exact wall no longer matters.
                let beside = amx.flatMap { g in operands.map { (a, b, c) in
                    seconds(giveUpAbove: paired / aneTolerance) {
                        _ = g.linear(a: a, b: b, c: c, m: m, k: shape.k, n: shape.n, weightIsTransposed: true) } } }
                cuts.append(Cut(k: shape.k, n: shape.n, m: m, share: share, ane: paired * 1e3, gpu: alone * 1e3,
                                amx: beside.map { $0 * 1e3 },
                                three: beside.flatMap { threeWall(k: shape.k, n: shape.n, m: m, share: share,
                                                                  best: min($0, paired) * 1e3) }))
            }
        }
        return cuts
    }

    /// **The GPU's share on the grid** (`cuts` → `amx_shares`): each `cutShapes` product at each of
    /// `heights`, through the conductor at a frozen share; the share that would have ended both engines
    /// together is read from the time each spent waiting (`Conductor.perShape`), then measured three times at
    /// that share, the median kept. In co-execution, not from the two engines alone: side by side they do
    /// not keep their solo rates — 93 % of the sum at 4 096 rows, 70 % at 1 024 (Z-Image's up
    /// projection) —, and the share that balances them is theirs together. Not judged against the exact: the
    /// products above judge the cut's arithmetic, which no share changes.
    static func measureCuts(_ o: Operands, _ bench: Bench, start: Double) throws -> [Cut] {
        let gemm = try GEMM(freezeCut: true, coexecution: true, tensorWeights: false)
        guard let conductor = gemm.conductor else { return [] }
        conductor.shares = .none
        let a = try gemm.wrap(o.a.base, bytes: o.a.capacity, name: "diagnostic.a")
        let b = try gemm.wrap(o.b.base, bytes: o.b.capacity, name: "diagnostic.b")
        let c = try gemm.wrap(o.c.base, bytes: o.c.capacity, name: "diagnostic.c")
        var cuts: [Cut] = []
        for shape in cutShapes {
            for m in heights {
                try bench.check()
                let key = "\(shape.k)×\(shape.n)"
                var share = start
                /// One product at `share`, and the share that would have balanced it.
                func balanced() -> Double {
                    conductor.initialFraction = share
                    // As in a render, the weight was just widened by the CPU (`Artifact.materialize`): the AMX
                    // finds it in the caches the CPU left warm, the GPU does not.
                    Widen.bfloat16ToFloat32(source: o.bf16.base, destination: o.b.base, count: shape.k * shape.n)
                    let before = conductor.perShape[key] ?? Conductor.ShapeRecord()
                    _ = gemm.linear(a: a, b: b, c: c, m: m, k: shape.k, n: shape.n, weightIsTransposed: true)
                    guard let after = conductor.perShape[key] else { return share }
                    let wall = after.wall - before.wall
                    let t = Double(Conductor.cut(m, share)) / Double(m)
                    let gpu = t / max(1e-9, wall - (after.idleGPU - before.idleGPU))
                    let amx = (1 - t) / max(1e-9, wall - (after.idleAMX - before.idleAMX))
                    return min(Conductor.fractionRange.upperBound, max(Conductor.fractionRange.lowerBound, gpu / (gpu + amx)))
                }
                _ = balanced()          // the warm-up: the operators built, the pages wired
                share = balanced()
                // Three estimates at that share, their median: a 1 024-row product lasts a few milliseconds,
                // and one estimate moved ρ by ±0.05 from one point of the grid to the next.
                let estimates = [balanced(), balanced(), balanced()].sorted()
                share = estimates[1]
                cuts.append(Cut(k: shape.k, n: shape.n, m: m, share: share))
            }
        }
        return cuts
    }
}
