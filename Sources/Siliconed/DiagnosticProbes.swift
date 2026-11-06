import Accelerate
internal import AMXLegacy
import Foundation
import Metal
import MetalPerformanceShadersGraph

/// **The diagnostic's shared means** (`Diagnostic.swift`): the engine's own objects on synthetic operands, each output against an
/// fp64 truth computed on the CPU.
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

    /// Rows of each output compared with the exact: 64 for a product, 16 queries per head for an attention.
    package static let sampledRows = 64, sampledQueries = 16
    /// Beyond this warm-up, one timed run instead of three: an engine that slow (BNNS in a type it
    /// emulates) is said slow at once, and the diagnostic keeps its budget.
    package static let longWarmUp = 0.4

    /// **The blocks measured**, from the models' `Config` defaults — the shapes a render computes:
    /// Z-Image 3840 → 10 240, 30 heads; Qwen-Image-2.1 4096 → 12 288, 32 heads; heads of 128. The widening probes size their
    /// weight on the first.
    package static var blocks: [(model: String, dim: Int, hidden: Int, heads: Int, headDim: Int)] {
        let z = DiT.Config(header: [:]), q = QwenImage21DiT.Config(header: [:])
        return [(Family.zImage.rawValue, z.dim, z.hidden, z.heads, z.dim / z.heads),
                (Family.qwenImage21.rawValue, q.dim, q.hidden, q.heads, q.headDim)]
    }

    // ── the clock and the operands ──────────────────────────────────────────────────────────

    /// **Median wall seconds of three runs after a warm-up** (pipelines built, pages wired, clocks
    /// up); one run when the warm-up alone passed `longWarmUp`.
    ///
    package static func seconds(_ body: () -> Void) -> Double {
        func once() -> Double {
            let begin = DispatchTime.now().uptimeNanoseconds
            body()
            return Double(DispatchTime.now().uptimeNanoseconds - begin) * 1e-9
        }
        if once() > longWarmUp { return once() }
        return [once(), once(), once()].sorted()[1]
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

    /// **The exact rows of a W8A8 product** (`gpu/int8x8`): each row rotated and quantized as the route does it (`W8A8.quantize`, the
    /// same fp32 arithmetic, so the codes are the route's to the bit), the codes times the integers exact in fp64, then `s_x · scale`.
    /// The truth of the integers, not of the activation: what the quantization loses is the recipe's, judged against its own reference
    /// (`--int8-activations`), not the kernel's. Laid out `[n, rows]`.
    package static func exactQuantizedRows(_ rows: [Int], a: UnsafePointer<Float>, q: UnsafePointer<Int8>, scale: Double,
                                           group: Int?, k: Int, n: Int) -> [Double] {
        var codes = [Float](repeating: 0, count: rows.count * k), scales = [Double](repeating: 0, count: rows.count)
        for (i, row) in rows.enumerated() {
            let (x, s) = W8A8.quantize(a + row * k, k: k, group: group)
            for j in 0..<k { codes[i * k + j] = Float(x[j]) }
            scales[i] = Double(s) * scale
        }
        var product = codes.withUnsafeBufferPointer {
            exactRows(Array(0..<rows.count), a: $0.baseAddress!, k: k, n: n, rotation: nil) { k0, count, into in
                vDSP_vflt8D(q + k0 * n, 1, into, 1, vDSP_Length(count))
            }
        }
        for j in 0..<n { for i in 0..<rows.count { product[j * rows.count + i] *= scales[i] } }
        return product
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

    /// **The worst channel of `got` against `exact`**, both `[n, rows]`: its error, and whether it passes `threshold` — judged on the
    /// error as it is, a NaN being infinite.
    static func judge(_ got: [Float], _ exact: [Double], columns: Int, threshold: Double) -> (pass: Bool, error: Double) {
        let e = exact.withUnsafeBufferPointer { channelError(got, $0, columns: columns) }
        return (e.worst.isFinite && e.worst <= threshold, e.worst)
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
        /// A Q4_K weight's super-blocks for the `gpu/q4kact` products (`kquantBlocks`), as a map stores them (`[N][K / 256]`), sized for
        /// the largest 8-bit product: deterministic bytes, `d` and `dmin` normal fp16 values.
        let kquant: Pages
        /// **The bytes each operand maps**, from the shapes alone (`CapacityProbes.need` charges these):
        /// `a` and `b` at the largest of the shapes (`m · k`, `k · n`), `c` likewise, `bf16` half of `b`,
        /// `q` the int8 weight of the products' largest, `q8Scales` one fp32 per block of rows and column.
        struct Sizes: Equatable {
            var aFloats, bFloats, cFloats, qValues, scaleCount: Int
            var bytes: Int {
                [aFloats * 4, bFloats * 4, bFloats * 2, cFloats * 4, qValues, scaleCount * 4, Diagnostic.kquantBytes(qValues)]
                    .map { Arena.capacity(for: max(1, $0)) }.reduce(0, +)
            }
        }
        /// **The operands of the probes' products**: the largest of each.
        static func sizes(products: [(m: Int, k: Int, n: Int, kind: WeightKind)]) -> Sizes {
            let eightBit = products.filter { $0.kind != .bf16 }
            return Sizes(aFloats: products.map { $0.m * $0.k }.max() ?? 1, bFloats: products.map { $0.k * $0.n }.max() ?? 1,
                         cFloats: products.map { $0.m * $0.n }.max() ?? 1, qValues: max(1, eightBit.map { $0.k * $0.n }.max() ?? 1),
                         scaleCount: max(1, eightBit.map { ($0.k + Diagnostic.q8Block - 1) / Diagnostic.q8Block * $0.n }.max() ?? 1))
        }
        init(_ size: Sizes) {
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
            kquant = Pages(bytes: Diagnostic.kquantBytes(qValues))
            Diagnostic.fillKQuant(kquant.base, blocks: Diagnostic.kquantBytes(qValues) / 144)
        }
    }

    /// The bytes of a Q4_K weight of `values` values (super-blocks of 256, 144 bytes each), one block more for a count not a multiple.
    static func kquantBytes(_ values: Int) -> Int { (values / 256 + 1) * 144 }

    /// **Q4_K super-blocks for the probes**: every byte of the scales and the integers from a deterministic sequence (each 6-bit scale
    /// and min, each 4-bit integer, takes every value), `d` in [2⁻⁹, 2⁻⁸) and `dmin` in [2⁻⁷, 2⁻⁶) — normal fp16 values, as the forge
    /// requires — so that a weight `d·sc·q − dmin·m` lies in about (−1, 2), the order of the other operands.
    static func fillKQuant(_ base: UnsafeMutableRawPointer, blocks: Int) {
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        var state: UInt64 = 0x51C0_4B11
        for b in 0..<blocks {
            let block = bytes + b * 144
            for i in 4..<144 {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                block[i] = UInt8(truncatingIfNeeded: state >> 33)
            }
            let d = Float16(Float(1 + (b % 7)) / 7 / 512 + 1.0 / 512).bitPattern, dmin = Float16(Float(1 + (b % 5)) / 5 / 128 + 1.0 / 128).bitPattern
            block[0] = UInt8(d & 0xFF); block[1] = UInt8(d >> 8); block[2] = UInt8(dmin & 0xFF); block[3] = UInt8(dmin >> 8)
        }
    }

    /// GGUF Q8_0's block along K, and its scales here: 2⁻⁷ · {1, 1.25, 1.5, 1.75}, exact in fp16, varying by
    /// block and column so that a scale read at the wrong place shows in the error.
    static let q8Block = 32
    static func q8Scale(_ i: Int) -> Float { int8Scale * (1 + Float(i % 4) / 4) }

    /// The int8 products' scale (2⁻⁷, exact) and rotation group: ComfyUI's `int8_tensorwise` convrot.
    static let int8Scale: Float = 1.0 / 128, int8Group = 256
}
