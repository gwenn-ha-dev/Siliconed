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
        let gauge: Gauge
        init(device: MTLDevice, queue: MTLCommandQueue, cancellation: Cancellation?, gauge: Gauge = Gauge()) {
            self.device = device; self.queue = queue; self.cancellation = cancellation; self.gauge = gauge
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
    static func poison(_ c: UnsafeMutablePointer<Float>, rows: [Int], n: Int) {
        for row in rows { (c + row * n).update(repeating: .nan, count: n) }
    }

    /// The worst row of `got` against `exact`, both `[n, rows]`.
    static func judge(_ got: [Float], _ exact: [Double], columns: Int, rate: Double, threshold: Double) -> Measure {
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
        /// **The bytes each operand maps**, from the shapes alone (`Diagnostic.need` charges these):
        /// `a` and `b` at the largest of the shapes (`m · k`, `k · n`), `c` likewise, `bf16` half of `b`,
        /// `q` the int8 weight of the products' largest, `q8Scales` one fp32 per block of rows and column.
        struct Sizes: Equatable {
            var aFloats, bFloats, cFloats, qValues, scaleCount: Int
            var bytes: Int {
                [aFloats * 4, bFloats * 4, bFloats * 2, cFloats * 4, qValues, scaleCount * 4]
                    .map { Arena.capacity(for: max(1, $0)) }.reduce(0, +)
            }
        }
        /// **The operands of the capacity map's classes** (`CapacityMap.Entry`) and of the witnesses: the largest of each.
        static func sizes(products: [(m: Int, k: Int, n: Int, kind: RouteTable.WeightKind)]) -> Sizes {
            let eightBit = products.filter { $0.kind != .bf16 }
            return Sizes(aFloats: products.map { $0.m * $0.k }.max() ?? 1, bFloats: products.map { $0.k * $0.n }.max() ?? 1,
                         cFloats: products.map { $0.m * $0.n }.max() ?? 1, qValues: max(1, eightBit.map { $0.k * $0.n }.max() ?? 1),
                         scaleCount: max(1, eightBit.map { ($0.k + Diagnostic.q8Block - 1) / Diagnostic.q8Block * $0.n }.max() ?? 1))
        }
        static var sizes: Sizes {
            let m = Diagnostic.heights.max()!
            let shapes = Diagnostic.shapes.map { (m: $0.m, k: $0.k, n: $0.n) }
                + (Diagnostic.cutShapes + Diagnostic.aneShapes).map { (m: m, k: $0.k, n: $0.n) }
            return Sizes(aFloats: shapes.map { $0.m * $0.k }.max()!, bFloats: shapes.map { $0.k * $0.n }.max()!,
                         cFloats: shapes.map { $0.m * $0.n }.max()!, qValues: Diagnostic.shapes.map { $0.k * $0.n }.max()!,
                         scaleCount: Diagnostic.shapes.map { ($0.k + q8Block - 1) / q8Block * $0.n }.max()!)
        }
        init(_ size: Sizes = Operands.sizes) {
            let (aFloats, bFloats) = (size.aFloats, size.bFloats)
            a = Pages(bytes: aFloats * 4); b = Pages(bytes: bFloats * 4); bf16 = Pages(bytes: bFloats * 2)
            c = Pages(bytes: size.cFloats * 4)
            Diagnostic.fill(a.floats(), count: aFloats, scale: 1, seed: 1)
            Diagnostic.fill(b.floats(), count: bFloats, scale: 1, seed: 2)
            Widen.float32ToBfloat16(source: b.base, destination: bf16.base, count: bFloats)
            Widen.bfloat16ToFloat32(source: bf16.base, destination: b.base, count: bFloats)
            let qValues = size.qValues
            q = Pages(bytes: qValues)
            let (bp, qp) = (b.floats(), q.base.assumingMemoryBound(to: Int8.self))
            for i in 0..<qValues { qp[i] = Int8((bp[i] * 127).rounded()) }
            let scaleCount = size.scaleCount
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
}
