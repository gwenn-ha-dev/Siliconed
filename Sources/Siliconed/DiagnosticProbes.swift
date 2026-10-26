import Accelerate
internal import AMXLegacy
import Foundation
import Metal
import MetalPerformanceShadersGraph

/// **The diagnostic's measurements** (`Diagnostic.swift` says what each decides): the engine's own
/// objects on synthetic operands, each output against an fp64 truth computed on the CPU.
extension Diagnostic {
    /// What every measurement needs: the device, its queue, the token.
    final class Bench {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let cancellation: Cancellation?
        init(device: MTLDevice, queue: MTLCommandQueue, cancellation: Cancellation?) {
            self.device = device; self.queue = queue; self.cancellation = cancellation
        }
        func check() throws { if cancellation?.isCancelled == true { throw EngineError.cancelled } }
        /// `pages` seen by the GPU, without a copy.
        func buffer(_ pages: Pages) throws -> MTLBuffer {
            guard let b = device.makeBuffer(bytesNoCopy: pages.base, length: pages.capacity, options: .storageModeShared,
                                            deallocator: nil) else { throw GEMM.Failure.unalignedBuffer("diagnostic") }
            return b
        }
    }

    /// One product: a block's `[m, k] · [k, n]`.
    package struct Shape: Sendable, Equatable {
        package let model, product: String
        package let m, k, n: Int
    }

    /// The heights of the products, and the sequences of the attention: a 512² and a 1024² image's
    /// tokens — multiples of 64, so `GEMM` submits them whole (`rowSplit`).
    package static let heights = [1024, 4096]
    /// Rows of each output compared with the exact: 64 for a product, 16 queries per head for an attention.
    package static let sampledRows = 64, sampledQueries = 16
    /// Beyond this warm-up, one timed run instead of three: an engine that slow (BNNS in a type it
    /// emulates) is said slow at once, and the diagnostic keeps its budget.
    package static let longWarmUp = 0.4

    /// **The blocks measured**, from the models' `Config` defaults — the shapes a render computes:
    /// Z-Image 3840 → 10 240, 30 heads; Qwen-Image-2.1 4096 → 12 288, 32 heads; heads of 128.
    package static var blocks: [(model: String, dim: Int, hidden: Int, heads: Int, headDim: Int)] {
        let z = DiT.Config(header: [:]), q = QwenImage21DiT.Config(header: [:])
        return [(Family.zImage.rawValue, z.dim, z.hidden, z.heads, z.dim / z.heads),
                (Family.qwenImage21.rawValue, q.dim, q.hidden, q.heads, q.headDim)]
    }

    /// Every fp32 product, the first being the drift witness's: per model, per height, up then down.
    package static var shapes: [Shape] {
        blocks.flatMap { b in
            heights.flatMap { m in
                [Shape(model: b.model, product: "up", m: m, k: b.dim, n: b.hidden),
                 Shape(model: b.model, product: "down", m: m, k: b.hidden, n: b.dim)]
            }
        }
    }

    /// **The grid the AMX/GPU ratio is measured on** (`measureCuts`), two axes through one point: inner
    /// dimensions `k` from 2 048 to 16 384 at `n` = 6 144, and widths `n` from 1 536 to 16 384 at `k` = 6 144 —
    /// the span of every DiT's products today (Anima's 2 048 to Krea 2's 16 384), at both `heights`. Not a
    /// model's shapes: `Conductor.Shares` reads any product's `k` and `n` along these axes.
    package static let cutDepths = [2048, 3072, 4096, 6144, 8192, 12288, 16384]
    package static let cutWidths = [1536, 3072, 12288, 16384]
    package static let cutWidth = 6144
    package static var cutShapes: [(k: Int, n: Int)] {
        cutDepths.map { (k: $0, n: cutWidth) } + cutWidths.map { (k: cutWidth, n: $0) }
    }

    /// **The ANE's grid** (`measureANECuts` → `ane_shares`): every family's own products, from the `Config`
    /// defaults — the square of its width, its MLP up and down, Krea 2's grouped K/V and Anima's text K/V —,
    /// at both `heights`. Not the AMX's two axes: the ANE's rate jumps from one shape to the next (its share at
    /// 4 096 rows: 0.47 at 4 096 × 6 144, 0.69 at 6 144², 0.85 at 8 192 × 6 144, 23/10/2026), and a share read
    /// across those axes gave Z-Image's up projection 0.37 where it balances at 0.56: its renders 12 % slower
    /// than the AMX's. A product of another shape takes its nearest (`Conductor.Shares`).
    package static var aneShapes: [(k: Int, n: Int)] {
        let z = DiT.Config(header: [:]), q = QwenImage21DiT.Config(header: [:])
        let krea = Krea2DiT.Config(header: [:]), anima = AnimaDiT.Config(header: [:]), moss = MossImageDiT.Config(header: [:])
        var shapes: [(k: Int, n: Int)] = []
        for (dim, hidden) in [(z.dim, z.hidden), (q.dim, q.hidden), (krea.dim, krea.hidden), (anima.dim, anima.hidden)] {
            shapes += [(dim, dim), (dim, hidden), (hidden, dim)]
        }
        // MOSS-Image's FFN is one fused Linear: up and gate together, `2 · hidden` wide.
        shapes += [(moss.dim, moss.dim), (moss.dim, 2 * moss.hidden), (moss.hidden, moss.dim)]
        shapes += [(krea.dim, krea.kvWidth), (anima.textDim, anima.dim)]
        var seen = Set<[Int]>()
        return shapes.filter { seen.insert([$0.k, $0.n]).inserted }
    }

    // ── the clock and the operands ──────────────────────────────────────────────────────────

    /// **Median wall seconds of three runs after a warm-up** (pipelines built, pages wired, clocks
    /// up); one run when the warm-up alone passed `longWarmUp`.
    ///
    /// `giveUpAbove`: **a loser is not timed three times**. When the first run after the warm-up is already
    /// slower than what the rule needs the route to beat, that run is the figure: the route loses whatever two
    /// more runs say, and the map's minute goes to the routes that may win. One noisy run can only keep a
    /// winner on the safe side (the route stays off), never make a loser win: a win is still a median of three.
    package static func seconds(giveUpAbove limit: Double? = nil, _ body: () -> Void) -> Double {
        func once() -> Double {
            let begin = DispatchTime.now().uptimeNanoseconds
            body()
            return Double(DispatchTime.now().uptimeNanoseconds - begin) * 1e-9
        }
        if once() > longWarmUp { return once() }
        let first = once()
        if let limit, first > limit { return first }
        return [first, once(), once()].sorted()[1]
    }

    /// **How much slower than the GPU alone's fp32 product a weight path may run and still be timed in full**:
    /// what it saves is the weight's widening, at most a few percent of a product of 1 024 rows or more (k · n
    /// values against 2 · m · k · n flops: on the M1 Pro, 18/10/2026, 8 % at 1 024 rows). At half again the GPU's
    /// time, the widening would have to cost half a product for the path to win.
    package static let weightPathGiveUp = 1.5

    /// **Uniform in `[−scale, scale]`, the same on every machine**: xorshift by blocks of 2¹⁶ values,
    /// each seeded by its index — whatever the cores, the first `count` values of a longer fill are
    /// these.
    package static func fill(_ p: UnsafeMutablePointer<Float>, count: Int, scale: Float, seed: UInt64) {
        let block = 1 << 16
        nonisolated(unsafe) let p = p
        DispatchQueue.concurrentPerform(iterations: (count + block - 1) / block) { j in
            var state = (seed &* 0x2545_F491_4F6C_DD1D) ^ (UInt64(j + 1) &* 0x9E37_79B9_7F4A_7C15)
            if state == 0 { state = 1 }
            for i in (j * block)..<min(count, (j + 1) * block) {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17
                p[i] = scale * Float(Int32(truncatingIfNeeded: state)) / Float(Int32.max)
            }
        }
    }

    /// `count` rows of `m`, the midpoints of `count` equal slices: every quantum of the cut is visited.
    package static func sampleRows(_ m: Int, count: Int) -> [Int] {
        (0..<count).map { ((2 * $0 + 1) * m) / (2 * count) }
    }

    /// **The exact rows**: `rows` of `A [m, k] · B [k, n]` in fp64 on the CPU — `cblas_dgemm` over
    /// slices of `B` widened to double, so `B` is never held whole in fp64. Laid out `[n, rows]`:
    /// `channelError` then normalizes each row by its own norm.
    package static func exactRows(_ rows: [Int], a: UnsafePointer<Float>, b: UnsafePointer<Float>, k: Int, n: Int) -> [Double] {
        exactRows(rows, a: a, k: k, n: n, rotation: nil) { k0, count, into in vDSP_vspdp(b + k0 * n, 1, into, 1, vDSP_Length(count)) }
    }

    /// **The exact rows of an int8 product**: `rows` of `A · W`, `W = R · q · s` (convrot of `group` along
    /// K, `Widen.dequantizeRotated`'s convention), computed as `((A · R) · q) · s` in fp64 — `R`'s
    /// butterflies on the rows in Double, the integers exact. Laid out `[n, rows]`.
    package static func exactRows(_ rows: [Int], a: UnsafePointer<Float>, q: UnsafePointer<Int8>, scale: Double,
                                  group: Int, k: Int, n: Int) -> [Double] {
        exactRows(rows, a: a, k: k, n: n, rotation: group) { k0, count, into in
            vDSP_vflt8D(q + k0 * n, 1, into, 1, vDSP_Length(count))
        }.map { $0 * scale }
    }

    /// **The exact rows of a Q8_0 product**: `rows` of `A · W`, `W[r, j] = q[r, j] · s[r / block, j]`, in fp64
    /// (`Float(q) · s` is exact in both). Laid out `[n, rows]`.
    package static func exactRows(_ rows: [Int], a: UnsafePointer<Float>, q: UnsafePointer<Int8>,
                                  blockScales s: UnsafePointer<Float>, block: Int, k: Int, n: Int) -> [Double] {
        let scales = UnsafeMutablePointer<Double>.allocate(capacity: n)
        defer { scales.deallocate() }
        return exactRows(rows, a: a, k: k, n: n, rotation: nil) { k0, count, into in
            for r in k0..<(k0 + count / n) {
                let row = into + (r - k0) * n
                vDSP_vflt8D(q + r * n, 1, row, 1, vDSP_Length(n))
                vDSP_vspdp(s + r / block * n, 1, scales, 1, vDSP_Length(n))
                vDSP_vmulD(row, 1, scales, 1, row, 1, vDSP_Length(n))
            }
        }
    }

    private static func exactRows(_ rows: [Int], a: UnsafePointer<Float>, k: Int, n: Int, rotation group: Int?,
                                  widen: (_ k0: Int, _ count: Int, _ into: UnsafeMutablePointer<Double>) -> Void) -> [Double] {
        let r = rows.count
        var left = [Double](repeating: 0, count: r * k)
        left.withUnsafeMutableBufferPointer { d in
            for (i, row) in rows.enumerated() { vDSP_vspdp(a + row * k, 1, d.baseAddress! + i * k, 1, vDSP_Length(k)) }
            // x · R per group: the four-point butterflies of `Widen.hadamard` (H symmetric), then 1/√G.
            if let group {
                let norm = 1 / Double(group).squareRoot()
                for g0 in stride(from: 0, to: r * k, by: group) {
                    let v = d.baseAddress! + g0
                    var span = 1
                    while span < group {
                        for base in stride(from: 0, to: group, by: 4 * span) {
                            for j in base..<(base + span) {
                                let (a0, b0, c0, d0) = (v[j], v[j + span], v[j + 2 * span], v[j + 3 * span])
                                let (p, m, u, w) = (a0 + b0, a0 - b0, c0 + d0, c0 - d0)
                                (v[j], v[j + span], v[j + 2 * span], v[j + 3 * span]) = (p + w, p - w, m + u, u - m)
                            }
                        }
                        span *= 4
                    }
                    for j in 0..<group { v[j] *= norm }
                }
            }
        }
        let slice = max(1, (1 << 22) / n)                       // 32 MB of B in fp64 at a time
        let right = Pages(bytes: slice * n * 8)
        let widened = right.base.assumingMemoryBound(to: Double.self)
        var product = [Double](repeating: 0, count: r * n)
        var k0 = 0
        while k0 < k {
            let kc = min(slice, k - k0)
            widen(k0, kc * n, widened)
            left.withUnsafeBufferPointer { A in
                product.withUnsafeMutableBufferPointer { C in
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(r), Int32(n), Int32(kc), 1,
                                A.baseAddress! + k0, Int32(k), widened, Int32(n), k0 == 0 ? 0 : 1, C.baseAddress!, Int32(n))
                }
            }
            k0 += kc
        }
        var transposed = [Double](repeating: 0, count: r * n)
        for i in 0..<r { for j in 0..<n { transposed[j * r + i] = product[i * n + j] } }
        return transposed
    }

    /// `rows` of `c [m, n]`, laid out `[n, rows]` as `exactRows`.
    package static func gather(_ c: UnsafePointer<Float>, rows: [Int], n: Int) -> [Float] {
        var got = [Float](repeating: 0, count: rows.count * n)
        for (i, row) in rows.enumerated() { for j in 0..<n { got[j * rows.count + i] = c[row * n + j] } }
        return got
    }

    /// NaN on the sampled rows: what an engine that did not run leaves there.
    private static func poison(_ c: UnsafeMutablePointer<Float>, rows: [Int], n: Int) {
        for row in rows { (c + row * n).update(repeating: .nan, count: n) }
    }

    /// The worst row of `got` against `exact`, both `[n, rows]`.
    private static func judge(_ got: [Float], _ exact: [Double], columns: Int, rate: Double, threshold: Double) -> Measure {
        let e = exact.withUnsafeBufferPointer { channelError(got, $0, columns: columns) }
        return Measure(rate: rate, error: e.worst, threshold: threshold)
    }

    /// The fp32 operands of every product: `A` and `B` at the largest of the shapes, `C` likewise.
    /// **`B` holds bf16 values**, as a published weight does (P19): `bf16` is the same weight as the map
    /// stores it, for `TensorGEMM` — and its widening is `b` exactly, so every engine multiplies the
    /// same numbers.
    ///
    /// `q` is the int8 weight of the `gemm_int8` products, `Int8((b · 127).rounded())` — read as a convrot
    /// weight of group 256 and scale `int8Scale`; staged in `b` (the reserve, as a model's), which the
    /// bf16 copy then restores. The `gemm_q8` products read the same `q` with `q8Scales`, fp32 values of
    /// fp16 scales per `q8Block` rows and column.
    final class Operands {
        let a, b, c, bf16, q, q8Scales: Pages
        init() {
            let m = Diagnostic.heights.max()!
            let shapes = Diagnostic.shapes.map { (m: $0.m, k: $0.k, n: $0.n) }
                + (Diagnostic.cutShapes + Diagnostic.aneShapes).map { (m: m, k: $0.k, n: $0.n) }
            let (aFloats, bFloats) = (shapes.map { $0.m * $0.k }.max()!, shapes.map { $0.k * $0.n }.max()!)
            a = Pages(bytes: aFloats * 4); b = Pages(bytes: bFloats * 4); bf16 = Pages(bytes: bFloats * 2)
            c = Pages(bytes: shapes.map { $0.m * $0.n }.max()! * 4)
            Diagnostic.fill(a.floats(), count: aFloats, scale: 1, seed: 1)
            Diagnostic.fill(b.floats(), count: bFloats, scale: 1, seed: 2)
            Widen.float32ToBfloat16(source: b.base, destination: bf16.base, count: bFloats)
            Widen.bfloat16ToFloat32(source: bf16.base, destination: b.base, count: bFloats)
            let qValues = Diagnostic.shapes.map { $0.k * $0.n }.max()!
            q = Pages(bytes: qValues)
            let (bp, qp) = (b.floats(), q.base.assumingMemoryBound(to: Int8.self))
            for i in 0..<qValues { qp[i] = Int8((bp[i] * 127).rounded()) }
            let scaleCount = Diagnostic.shapes.map { ($0.k + q8Block - 1) / q8Block * $0.n }.max()!
            q8Scales = Pages(bytes: scaleCount * 4)
            let sp = q8Scales.floats()
            for i in 0..<scaleCount { sp[i] = q8Scale(i) }
        }
    }

    /// GGUF Q8_0's block along K, and its scales here: 2⁻⁷ · {1, 1.25, 1.5, 1.75}, exact in fp16, varying by
    /// block and column so that a scale read at the wrong place shows in the error.
    static let q8Block = 32
    static func q8Scale(_ i: Int) -> Float { int8Scale * (1 + Float(i % 4) / 4) }

    /// The int8 products' scale (2⁻⁷, exact) and rotation group: ComfyUI's `int8_tensorwise` convrot.
    static let int8Scale: Float = 1.0 / 128, int8Group = 256

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

    // ── fp16 and bf16 products ──────────────────────────────────────────────────────────────

    /// A half-precision type, the way the four parties name and round it.
    private enum Half: String, CaseIterable {
        case fp16, bf16
        var graph: MPSDataType { self == .fp16 ? .float16 : .bFloat16 }
        var bnns: BNNSDataType { self == .fp16 ? .float16 : BNNSDataTypeBFloat16 }
        /// The `matmul2d` kernel of operands of this type, fp32 result.
        var kernel: String { self == .fp16 ? "product_half" : "product_bfloat" }
        /// `x` rounded to the nearest of the type, to nearest even, as an fp32.
        func round(_ x: Float) -> Float { self == .fp16 ? Float(Float16(x)) : widen(narrow(x)) }
        func narrow(_ x: Float) -> UInt16 {
            if self == .fp16 { return Float16(x).bitPattern }
            let bits = x.bitPattern
            return UInt16(truncatingIfNeeded: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) >> 16)
        }
        func widen(_ h: UInt16) -> Float { self == .fp16 ? Float(Float16(bitPattern: h)) : Float(bitPattern: UInt32(h) << 16) }
    }

    /// **Half precision at the first product's shape**, three engines per type. The fp32 operands are
    /// rounded in place to the type first — the exact is then the product of exactly the values the
    /// engines read (fp16 before bf16: a value rounded to fp16, then to bf16, is a bf16).
    static func measureLowPrecision(_ o: Operands, _ bench: Bench) throws -> [LowPrecision] {
        let s = shapes[0]
        let (m, k, n) = (s.m, s.k, s.n)
        let a = o.a.floats(), b = o.b.floats(), c = o.c.floats()
        let (aHalf, bHalf, cHalf) = (Pages(bytes: m * k * 2), Pages(bytes: k * n * 2), Pages(bytes: m * n * 2))
        let (aBuffer, bBuffer, cBuffer) = (try bench.buffer(aHalf), try bench.buffer(bHalf), try bench.buffer(cHalf))
        let cFloat = try bench.buffer(o.c)
        let tensors = TensorProducts(bench)
        let flops = 2.0 * Double(m) * Double(k) * Double(n)
        var workspace: Pages?
        var results: [LowPrecision] = []
        for type in Half.allCases {
            try bench.check()
            for (values, half, count) in [(a, aHalf, m * k), (b, bHalf, k * n)] {
                nonisolated(unsafe) let (values, out) = (values, half.base.assumingMemoryBound(to: UInt16.self))
                DispatchQueue.concurrentPerform(iterations: (count + 65_535) / 65_536) { j in
                    for i in (j * 65_536)..<min(count, (j + 1) * 65_536) {
                        values[i] = type.round(values[i])
                        out[i] = type.narrow(values[i])
                    }
                }
            }
            let rows = sampleRows(m, count: sampledRows)
            let exact = exactRows(rows, a: a, b: b, k: k, n: n)
            // The floors: the exact rounded to the output's type, against the exact.
            let floorHalf = judge(exact.map { type.round(Float($0)) }, exact, columns: rows.count, rate: 0, threshold: 1).error
            let floorFloat = judge(exact.map { Float($0) }, exact, columns: rows.count, rate: 0, threshold: 1).error
            func threshold(_ floor: Double) -> Double { max(fp32Threshold, floorMargin * floor) }

            // MPSGraph: operands and result in the type.
            let h = cHalf.base.assumingMemoryBound(to: UInt16.self)
            for row in rows { (h + row * n).update(repeating: type.narrow(.nan), count: n) }
            let graph = MPSGraph()
            let x = graph.placeholder(shape: [m, k].map { NSNumber(value: $0) }, dataType: type.graph, name: nil)
            let y = graph.placeholder(shape: [k, n].map { NSNumber(value: $0) }, dataType: type.graph, name: nil)
            let z = graph.matrixMultiplication(primary: x, secondary: y, name: nil)
            let feeds = [x: MPSGraphTensorData(aBuffer, shape: [m, k].map { NSNumber(value: $0) }, dataType: type.graph),
                         y: MPSGraphTensorData(bBuffer, shape: [k, n].map { NSNumber(value: $0) }, dataType: type.graph)]
            let result = [z: MPSGraphTensorData(cBuffer, shape: [m, n].map { NSNumber(value: $0) }, dataType: type.graph)]
            let tGraph = seconds { graph.run(with: bench.queue, feeds: feeds, targetOperations: nil, resultsDictionary: result) }
            var got = [Float](repeating: 0, count: rows.count * n)
            for (i, row) in rows.enumerated() { for j in 0..<n { got[j * rows.count + i] = type.widen(h[row * n + j]) } }
            results.append(LowPrecision(engine: "mpsgraph", type: type.rawValue, output: type.rawValue, m: m, k: k, n: n,
                                        measure: judge(got, exact, columns: rows.count, rate: flops / tGraph / 1e12,
                                                       threshold: threshold(floorHalf)),
                                        floor: floorHalf, threshold: threshold(floorHalf)))

            // BNNS: operands in the type, result in fp32.
            try bench.check()
            var measure: Measure?
            if let run = bnnsProduct(type, a: aHalf.base, b: bHalf.base, c: o.c.base, m: m, k: k, n: n, workspace: &workspace) {
                poison(c, rows: rows, n: n)
                let t = seconds(run)
                measure = judge(gather(c, rows: rows, n: n), exact, columns: rows.count, rate: flops / t / 1e12,
                                threshold: threshold(floorFloat))
            }
            results.append(LowPrecision(engine: "bnns", type: type.rawValue, output: "fp32", m: m, k: k, n: n,
                                        measure: measure, floor: floorFloat, threshold: threshold(floorFloat)))

            // Metal 4's matmul2d: operands in the type, result in fp32.
            try bench.check()
            measure = nil
            if let run = tensors?.product(type.kernel, a: aBuffer, b: bBuffer, c: cFloat, m: m, k: k, n: n) {
                poison(c, rows: rows, n: n)
                let t = seconds(run)
                measure = judge(gather(c, rows: rows, n: n), exact, columns: rows.count, rate: flops / t / 1e12,
                                threshold: threshold(floorFloat))
            }
            results.append(LowPrecision(engine: "matmul2d", type: type.rawValue, output: "fp32", m: m, k: k, n: n,
                                        measure: measure, floor: floorFloat, threshold: threshold(floorFloat)))
        }
        return results
    }

    /// `BNNSMatMul` on operands of `type`, fp32 result, as `Conductor.multiply` calls it otherwise;
    /// `nil` when BNNS refuses the types.
    private static func bnnsProduct(_ type: Half, a: UnsafeMutableRawPointer, b: UnsafeMutableRawPointer,
                                    c: UnsafeMutableRawPointer, m: Int, k: Int, n: Int,
                                    workspace: inout Pages?) -> (() -> Void)? {
        func nd(_ p: UnsafeMutableRawPointer, _ rows: Int, _ columns: Int, _ t: BNNSDataType) -> BNNSNDArrayDescriptor {
            var d = BNNSNDArrayDescriptor()
            d.layout = BNNSDataLayoutRowMajorMatrix
            d.size = (columns, rows, 0, 0, 0, 0, 0, 0)
            d.data = p
            d.data_type = t
            d.data_scale = 1
            d.data_bias = 0
            return d
        }
        var (A, B, C) = (nd(a, m, k, type.bnns), nd(b, k, n, type.bnns), nd(c, m, n, .float))
        var params = BNNSFilterParameters()
        let size = amx_matmul_workspace(false, false, 1, &A, &B, &C, &params)
        guard size >= 0 else { return nil }
        if size > (workspace?.capacity ?? 0) { workspace = Pages(bytes: size) }
        nonisolated(unsafe) let space = size > 0 ? workspace?.base : nil
        return { _ = amx_matmul(false, false, 1, &A, &B, &C, space, &params) }
    }

    /// **Metal 4's tensor product** (`MetalPerformancePrimitives`, `matmul2d`): the kernel of Apple's
    /// header, 64 × 32 tiles over four simdgroups, its destination a cooperative tensor written once
    /// (`run` accumulates into a device tensor). On an M5 it runs on the GPU's neural accelerators;
    /// before, on the shader cores. Not tuned: it answers "is there a matrix engine", not "how fast can
    /// the best kernel go". `nil` before macOS 26, or on a device without Metal 4.
    private final class TensorProducts {
        private let queue: MTLCommandQueue
        private let pipelines: [String: MTLComputePipelineState]

        static let source = """
        #include <metal_stdlib>
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace metal;
        using namespace mpp::tensor_ops;

        template <typename T>
        inline void product(device T *a, device T *b, device float *c, constant uint3 &mkn, uint2 tgid) {
            const int M = mkn.x, K = mkn.y, N = mkn.z;
            // Extents innermost first: A [M, K] row-major is (K, M).
            tensor<device T, dextents<int32_t, 2>, tensor_inline> A(a, dextents<int32_t, 2>(K, M));
            tensor<device T, dextents<int32_t, 2>, tensor_inline> B(b, dextents<int32_t, 2>(N, K));
            tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
            constexpr auto descriptor = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));
            matmul2d<descriptor, execution_simdgroups<4>> op;
            auto mA = A.slice(0, tgid.y * 64);
            auto mB = B.slice(tgid.x * 32, 0);
            auto mC = C.slice(tgid.x * 32, tgid.y * 64);
            // The header's example says `get_mask` and `__remove_addrspace_t`; this SDK's compiler
            // knows neither: `is_valid_element`, and the slices' own types.
            auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0; }
            op.run(mA, mB, cT);
            cT.store(mC);
        }

        kernel void product_half(device half *a [[buffer(0)]], device half *b [[buffer(1)]],
                                 device float *c [[buffer(2)]], constant uint3 &mkn [[buffer(3)]],
                                 uint2 tgid [[threadgroup_position_in_grid]]) { product(a, b, c, mkn, tgid); }
        kernel void product_bfloat(device bfloat *a [[buffer(0)]], device bfloat *b [[buffer(1)]],
                                   device float *c [[buffer(2)]], constant uint3 &mkn [[buffer(3)]],
                                   uint2 tgid [[threadgroup_position_in_grid]]) { product(a, b, c, mkn, tgid); }
        """

        init?(_ bench: Bench) {
            guard #available(macOS 26.0, *), bench.device.supportsFamily(.metal4) else { return nil }
            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            guard let library = try? bench.device.makeLibrary(source: TensorProducts.source, options: options) else { return nil }
            var pipelines: [String: MTLComputePipelineState] = [:]
            for name in ["product_half", "product_bfloat"] {
                if let f = library.makeFunction(name: name), let p = try? bench.device.makeComputePipelineState(function: f) {
                    pipelines[name] = p
                }
            }
            queue = bench.queue
            self.pipelines = pipelines
        }

        /// `C [m, n] = A [m, k] · B [k, n]`, `m` a multiple of 64 and `n` of 32 (no edge tile).
        func product(_ kernel: String, a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int) -> (() -> Void)? {
            guard let pipeline = pipelines[kernel], m % 64 == 0, n % 32 == 0 else { return nil }
            let queue = self.queue
            return {
                guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else { return }
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(a, offset: 0, index: 0)
                encoder.setBuffer(b, offset: 0, index: 1)
                encoder.setBuffer(c, offset: 0, index: 2)
                var mkn = SIMD3<UInt32>(UInt32(m), UInt32(k), UInt32(n))
                encoder.setBytes(&mkn, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
                encoder.dispatchThreadgroups(MTLSize(width: n / 32, height: m / 64, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth * 4,
                                                                            height: 1, depth: 1))
                encoder.endEncoding()
                commands.commit()
                commands.waitUntilCompleted()
                _ = Attention.verdict(commands, what: "matmul2d \(m)×\(k)×\(n)")
            }
        }
    }

    // ── the attention ───────────────────────────────────────────────────────────────────────

    /// **Each model's heads, at both sequences, through the engine's three SDPAs.** `q` and `k` uniform
    /// in ±3, so the logits spread over ±12 and the softmax is peaked as a model's is (uniform in ±1, it
    /// would be flat, and an attention that ignored its weights would pass); `v` in ±1.
    static func measureAttention(_ bench: Bench) throws -> [AttentionCapacity] {
        var results: [AttentionCapacity] = []
        for block in blocks {
            for sequence in heights {
                try bench.check()
                let (h, d) = (block.heads, block.headDim)
                let floats = sequence * h * d
                let (q, k, v, out) = (Pages(bytes: floats * 4), Pages(bytes: floats * 4), Pages(bytes: floats * 4),
                                      Pages(bytes: floats * 4))
                fill(q.floats(), count: floats, scale: 3, seed: 11)
                fill(k.floats(), count: floats, scale: 3, seed: 12)
                fill(v.floats(), count: floats, scale: 1, seed: 13)
                let (qb, kb, vb, ob) = (try bench.buffer(q), try bench.buffer(k), try bench.buffer(v), try bench.buffer(out))
                let rows = sampleRows(sequence, count: sampledQueries)
                let exact = exactAttention(q: q.floats(), k: k.floats(), v: v.floats(), sequence: sequence, heads: h,
                                           headDim: d, rows: rows)
                let o = out.floats(), vectors = rows.count * h
                let flops = 4.0 * Double(sequence) * Double(sequence) * Double(h) * Double(d)
                func measure(_ run: () -> Void) -> Measure {
                    for row in rows { (o + row * h * d).update(repeating: .nan, count: h * d) }
                    let t = seconds(run)
                    var got = [Float](repeating: 0, count: vectors * d)
                    for (i, row) in rows.enumerated() {
                        for head in 0..<h { for x in 0..<d { got[x * vectors + i * h + head] = o[row * h * d + head * d + x] } }
                    }
                    return judge(got, exact, columns: vectors, rate: flops / t / 1e12, threshold: fp32Threshold)
                }
                let graph: Measure = try {
                    let attention = Attention(device: bench.device, queue: bench.queue, heads: h, sequence: sequence, headDim: d)
                    try bench.check()
                    return measure { _ = attention.run(q: qb, k: kb, v: vb, into: ob) }
                }()
                try bench.check()
                let flash = (try? FlashMatrix(device: bench.device, queue: bench.queue, heads: h, sequence: sequence,
                                              headDim: d)).map { kernel in measure { kernel.run(q: qb, k: kb, v: vb, into: ob) } }
                try bench.check()
                // At the reference share, not at whatever the settings say: the map must measure the same cut
                // whatever it decided last time.
                var parts: AttentionCapacity.SplitParts?
                let split = (try? HeadSplitAttention(device: bench.device, queue: bench.queue, heads: h, sequence: sequence,
                                                     headDim: d, cpuFraction: HeadSplitAttention.referenceFraction)).map { cut in
                    var best = (gpu: Double.infinity, cpu: Double.infinity, wall: Double.infinity)
                    let measured = measure {
                        let t = DispatchTime.now().uptimeNanoseconds
                        let r = cut.run(q: qb, k: kb, v: vb, into: ob, q: q.floats(), k: k.floats(), v: v.floats(), out: o)
                        let wall = Double(DispatchTime.now().uptimeNanoseconds - t) * 1e-9
                        if wall < best.wall { best = (r.gpu, r.cpu, wall) }
                    }
                    if best.wall.isFinite, cut.cpuHeads > 0 {
                        parts = .init(cpuHeads: cut.cpuHeads, gpuSeconds: best.gpu, cpuSeconds: best.cpu)
                    }
                    return measured
                }
                results.append(AttentionCapacity(model: block.model, heads: h, headDim: d, sequence: sequence,
                                                 graph: graph, flash: flash, headSplit: split, splitParts: parts))
            }
        }
        return results
    }

    /// **The exact attention of `rows` queries of every head**, fp64 on the CPU — `softmax(q·kᵀ/√Dh)·v`
    /// as the reference writes it, over all `S` keys. Laid out `[Dh, rows·H]`, the vector of query `i`
    /// at head `h` in column `i·H + h`.
    package static func exactAttention(q: UnsafePointer<Float>, k: UnsafePointer<Float>, v: UnsafePointer<Float>,
                                       sequence: Int, heads: Int, headDim: Int, rows: [Int]) -> [Double] {
        let vectors = rows.count * heads, width = heads * headDim
        let result = Pages(bytes: vectors * headDim * 8)
        nonisolated(unsafe) let (q, k, v, o) = (q, k, v, result.base.assumingMemoryBound(to: Double.self))
        let scale = 1 / Double(headDim).squareRoot()
        DispatchQueue.concurrentPerform(iterations: vectors) { c in
            let row = rows[c / heads], head = c % heads
            withUnsafeTemporaryAllocation(of: Double.self, capacity: sequence + headDim) { buffer in
                let scores = buffer.baseAddress!, sum = scores + sequence
                let query = q + row * width + head * headDim
                var maximum = -Double.infinity
                for j in 0..<sequence {
                    let key = k + j * width + head * headDim
                    var s = 0.0
                    for x in 0..<headDim { s += Double(query[x]) * Double(key[x]) }
                    scores[j] = s * scale
                    maximum = max(maximum, scores[j])
                }
                var total = 0.0
                for x in 0..<headDim { sum[x] = 0 }
                for j in 0..<sequence {
                    let p = Foundation.exp(scores[j] - maximum)
                    total += p
                    let value = v + j * width + head * headDim
                    for x in 0..<headDim { sum[x] += p * Double(value[x]) }
                }
                for x in 0..<headDim { o[x * vectors + c] = sum[x] / total }
            }
        }
        return Array(UnsafeBufferPointer(start: o, count: vectors * headDim))
    }

    // ── the widening ────────────────────────────────────────────────────────────────────────

    /// **A Z-Image w1 widened to fp32**, bf16 and int8 × bf16 scale per column, on the CPU: what the
    /// default path pays per weight, against which the copies of `gemm_tensor`, `gemm_int8` and `gemm_q8`
    /// are weighed. Both are exact (a shift; 8 + 8 significant bits in fp32's 24), so the threshold is 0:
    /// every value is compared, none may differ from the exact.
    static func measureWidening(_ bench: Bench) throws -> [Widening] {
        let block = blocks[0]
        let count = block.dim * block.hidden, columns = block.hidden
        let destination = Pages(bytes: count * 4)
        let floats = destination.floats()
        var results: [Widening] = []
        /// One engine: poisoned output, timed, every value against `exact`.
        func measure(_ exact: @escaping @Sendable (Int) -> Float, _ run: () throws -> Void) -> Measure {
            memset(destination.base, 0xFF, count * 4)
            var failed = false
            let t = seconds { do { try run() } catch { failed = true } }
            let error = failed ? Double.infinity : worstRelative(floats, count: count, exact: exact)
            return Measure(rate: Double(count) / t / 1e9, error: error, threshold: 0)
        }

        // bf16 → fp32, a shift.
        try bench.check()
        let bf16 = Pages(bytes: count * 2)
        fill(floats, count: count, scale: 1, seed: 21)
        Widen.float32ToBfloat16(source: destination.base, destination: bf16.base, count: count)
        nonisolated(unsafe) let halves = bf16.base.assumingMemoryBound(to: UInt16.self)
        let bf16Exact: @Sendable (Int) -> Float = { Float(bitPattern: UInt32(halves[$0]) << 16) }
        results.append(Widening(
            kind: "bf16", values: count,
            cpu: measure(bf16Exact) { Widen.bfloat16ToFloat32(source: bf16.base, destination: destination.base, count: count) }))

        // bf16 → bf16, the copy `gemm_tensor` makes instead: the same reads, half the writes.
        try bench.check()
        let copy = Pages(bytes: count * 2)
        memset(copy.base, 0xFF, count * 2)
        let tCopy = seconds { Widen.copyBfloat16(source: bf16.base, destination: copy.base, count: count) }
        results.append(Widening(kind: "bf16 copy", values: count,
                                cpu: Measure(rate: Double(count) / tCopy / 1e9,
                                             error: memcmp(copy.base, bf16.base, count * 2) == 0 ? 0 : .infinity,
                                             threshold: 0)))

        // int8 × bf16 scale per column, the layout of an 8-bit map's `[K, N]` weight.
        try bench.check()
        let int8 = Pages(bytes: count + 2 * columns)
        nonisolated(unsafe) let q = int8.base.assumingMemoryBound(to: Int8.self)
        nonisolated(unsafe) let scales = (int8.base + count).assumingMemoryBound(to: UInt16.self)
        for i in 0..<count { q[i] = Int8(truncatingIfNeeded: Int(floats[i] * 127)) }
        // Scales in [2⁻¹⁰, 2⁻⁶): normal products, as the forge requires (`checkSubnormal`).
        for c in 0..<columns { scales[c] = Half.bf16.narrow((1 + abs(floats[c]) * 15) / 1024) }
        let int8Exact: @Sendable (Int) -> Float = { i in
            Float(Double(q[i]) * Double(Float(bitPattern: UInt32(scales[i % columns]) << 16)))
        }
        results.append(Widening(
            kind: "int8", values: count,
            cpu: measure(int8Exact) {
                Widen.dequantize(int8.base, kind: .int8, count: count, scale: int8.base + count, scaleType: .bfloat16,
                                 layout: .columns(columns), into: floats)
            }))

        // int8 convrot (group 256, one fp32 scale), what an int8 map's default path widens on the CPU —
        // `gemm_int8` replaces it by a copy of the bytes. Exact at the bit on 4096 sampled values:
        // `Float(H·q) · (s/√G)`, `H`'s signs from its base-4 digits (`h4`'s Kronecker powers).
        try bench.check()
        let rows = block.dim, group = int8Group
        var scale = int8Scale
        memset(destination.base, 0xFF, count * 4)
        let tRotated = seconds {
            Widen.dequantizeRotated(int8.base, rows: rows, columns: columns, group: group, scale: &scale,
                                    scaleType: .float32, layout: .tensor, into: floats)
        }
        let h4: [Int32] = [1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1]
        func sign(_ r: Int, _ j: Int) -> Int32 {
            var (r, j, p) = (r, j, Int32(1))
            for _ in 0..<4 { p *= h4[(r % 4) * 4 + j % 4]; r /= 4; j /= 4 }
            return p
        }
        let factor = int8Scale * (1 / Float(group).squareRoot())
        var rotatedError = 0.0
        for t in 0..<4096 {
            let i = (t * 2_654_435_761) % count, (r, c) = (i / columns, i % columns), g0 = r / group * group
            var m: Int32 = 0
            for j in 0..<group { m += sign(r - g0, j) * Int32(q[(g0 + j) * columns + c]) }
            let e = Float(m) * factor
            if floats[i].bitPattern != e.bitPattern {
                rotatedError = max(rotatedError, floats[i].isNaN ? .infinity : Double(abs(floats[i] - e)) / max(Double(abs(e)), 1e-30))
            }
        }
        results.append(Widening(kind: "int8 convrot", values: count,
                                cpu: Measure(rate: Double(count) / tRotated / 1e9, error: rotatedError, threshold: 0)))
        // int8 → int8, the copy `gemm_int8` makes instead (`Artifact.stageInt8`): a quarter of the writes.
        let copy8 = Pages(bytes: count)
        memset(copy8.base, 0x7F, count)
        let tCopy8 = seconds { Widen.copyBfloat16(source: int8.base, destination: copy8.base, count: count / 2) }
        results.append(Widening(kind: "int8 copy", values: count,
                                cpu: Measure(rate: Double(count) / tCopy8 / 1e9,
                                             error: memcmp(copy8.base, int8.base, count) == 0 ? 0 : .infinity, threshold: 0)))

        // Q8_0: the same int8 × fp16 scales per 32 rows and column, what a GGUF Q8_0 map's default path widens
        // on the CPU — `gemm_q8` copies the bytes and widens the scales alone (`Artifact.stageInt8`).
        try bench.check()
        let blockScales = (rows + q8Block - 1) / q8Block * columns
        let halves16 = Pages(bytes: blockScales * 2)
        nonisolated(unsafe) let h = halves16.base.assumingMemoryBound(to: Float16.self)
        for i in 0..<blockScales { h[i] = Float16(q8Scale(i)) }
        let q8Exact: @Sendable (Int) -> Float = { i in Float(q[i]) * Float(h[i / columns / q8Block * columns + i % columns]) }
        results.append(Widening(
            kind: "q8_0", values: count,
            cpu: measure(q8Exact) {
                Widen.dequantize(int8.base, kind: .int8, count: count, scale: halves16.base, scaleType: .float16,
                                 layout: .blocks(columns: columns, rows: q8Block), into: floats)
            }))
        let scales32 = Pages(bytes: blockScales * 4)
        memset(copy8.base, 0x7F, count)
        let tCopyQ8 = seconds {
            Widen.copyBfloat16(source: int8.base, destination: copy8.base, count: count / 2)
            Widen.float16ToFloat32(halves16.base, count: blockScales, into: scales32.floats())
        }
        let scalesRight = (0..<blockScales).allSatisfy { scales32.floats()[$0] == Float(h[$0]) }
        results.append(Widening(kind: "q8_0 copy", values: count,
                                cpu: Measure(rate: Double(count) / tCopyQ8 / 1e9,
                                             error: memcmp(copy8.base, int8.base, count) == 0 && scalesRight ? 0 : .infinity,
                                             threshold: 0)))
        return results
    }

    /// The worst `|got − exact| / |exact|` over `count` values, over the cores; infinite for a NaN.
    private static func worstRelative(_ got: UnsafePointer<Float>, count: Int, exact: @escaping @Sendable (Int) -> Float) -> Double {
        let block = 1 << 18, jobs = (count + block - 1) / block
        let worst = Pages(bytes: jobs * 8)
        nonisolated(unsafe) let (got, w) = (got, worst.base.assumingMemoryBound(to: Double.self))
        DispatchQueue.concurrentPerform(iterations: jobs) { j in
            var m = 0.0
            for i in (j * block)..<min(count, (j + 1) * block) where got[i].bitPattern != exact(i).bitPattern {
                let e = Double(exact(i)), d = abs(Double(got[i]) - e)
                m = max(m, d.isNaN ? .infinity : d / max(abs(e), .leastNormalMagnitude))
            }
            w[j] = m
        }
        return (0..<jobs).map { w[$0] }.max() ?? 0
    }

    // ── the drift witness ───────────────────────────────────────────────────────────────────

    /// **The first product again, GPU and AMX**, on fresh operands of the same values (the same
    /// seeds): the ratio to `products[0]` is the machine's drift over the diagnostic.
    static func measureWitness(first: Product, _ bench: Bench) throws -> Witness {
        let s = shapes[0]
        let (a, b, c) = (Pages(bytes: s.m * s.k * 4), Pages(bytes: s.k * s.n * 4), Pages(bytes: s.m * s.n * 4))
        fill(a.floats(), count: s.m * s.k, scale: 1, seed: 1)
        fill(b.floats(), count: s.k * s.n, scale: 1, seed: 2)
        let gemm = try GEMM(freezeCut: true, coexecution: false)
        let (ab, bb, cb) = (try gemm.wrap(a.base, bytes: a.capacity, name: "witness.a"),
                            try gemm.wrap(b.base, bytes: b.capacity, name: "witness.b"),
                            try gemm.wrap(c.base, bytes: c.capacity, name: "witness.c"))
        let flops = 2.0 * Double(s.m) * Double(s.k) * Double(s.n)
        let gpu = seconds { _ = gemm.linear(a: ab, b: bb, c: cb, m: s.m, k: s.k, n: s.n, weightIsTransposed: true) }
        var amx: Double?
        if first.amx != nil {
            var workspace: Pages?
            amx = flops / seconds {
                _ = Conductor.multiply(a: a.base, b: b.base, c: c.base, rows: s.m, k: s.k, n: s.n, transposeWeight: false,
                                       workspace: &workspace)
            } / 1e12
        }
        return Witness(gpuBefore: first.gpu.rate, gpuAfter: flops / gpu / 1e12, amxBefore: first.amx?.rate, amxAfter: amx)
    }
}
