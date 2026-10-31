import Accelerate
import Metal
import XCTest
@testable import Siliconed

/// **The GPU decode step reads the resident weights where they lie** (`DecodeKernels`, `Qwen3VLDecoder`), on synthetic
/// data and no weights: each product against an fp64 reference of `(q · s) · R` judged channel by channel and against the
/// widened path it replaces (invariant 1), the rotation and norm kernels, the cached attention, the head and its
/// argmax (ties, gap), and a whole step of a tiny layer stack against the CPU pass `generate` runs — the same tokens.
final class Qwen3VLDecoderTests: XCTestCase {
    private var device: MTLDevice!
    private var kernels: DecodeKernels!
    /// The pages the buffers wrap.
    private var held: [Pages] = []

    override func setUpWithError() throws {
        device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        kernels = try DecodeKernels(device: device)
    }

    override func tearDown() { held = [] }

    // ── synthetic data ─────────────────────────────────────────────────────────────────────────

    private struct Random {
        var state: UInt64
        init(_ seed: UInt64) { state = seed &* 0x9E3779B97F4A7C15 &+ 1 }
        mutating func next() -> UInt64 {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return state
        }
        /// Uniform in [-1, 1).
        mutating func unit() -> Float { Float(Int32(truncatingIfNeeded: next() >> 33)) / Float(1 << 30) - 1 }
        mutating func floats(_ count: Int, scale: Float = 1) -> [Float] { (0..<count).map { _ in unit() * scale } }
    }

    private func pages(_ bytes: Int) -> Pages {
        let p = Pages(bytes: bytes)
        held.append(p)
        return p
    }

    private func buffer(_ p: Pages) throws -> MTLBuffer {
        try XCTUnwrap(device.makeBuffer(bytesNoCopy: p.base, length: p.capacity, options: .storageModeShared, deallocator: nil))
    }

    private func buffer(floats: [Float]) throws -> MTLBuffer {
        let p = pages(floats.count * 4)
        floats.withUnsafeBytes { p.base.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return try buffer(p)
    }

    private func contents(_ b: MTLBuffer, _ count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: Float.self), count: count))
    }

    /// `count` pseudo-random bytes, written at `base` by copying blocks of a seeded pool at other offsets: a matrix of 50 M
    /// values in milliseconds, not a loop of 50 M in a debug build.
    private static let bytePool: [UInt8] = {
        var rng = Random(0x5EED)
        return (0..<(1 << 20)).map { _ in UInt8(truncatingIfNeeded: rng.next() >> 40) }
    }()

    /// The same for bf16 values of magnitude below `scale`.
    private static let bf16Pool: [UInt16] = {
        var rng = Random(0xBF16)
        return (0..<(1 << 18)).map { _ in UInt16((rng.unit()).bitPattern >> 16) }
    }()

    private func fillBF16(_ base: UnsafeMutableRawPointer, count: Int, seed: UInt64, scale: Float) {
        var rng = Random(seed)
        let block = 4096
        var at = 0
        Self.bf16Pool.withUnsafeBytes { source in
            while at < count {
                let from = Int(rng.next() >> 33) % (Self.bf16Pool.count - block)
                base.advanced(by: 2 * at).copyMemory(from: source.baseAddress! + 2 * from, byteCount: 2 * min(block, count - at))
                at += block
            }
        }
        // One scale for the tensor, applied to the exponent: a power of two keeps bf16 exact.
        let bits = base.assumingMemoryBound(to: UInt16.self), shift = Int(log2(1 / scale).rounded())
        for i in 0..<count { var b = bits[i]; let e = Int((b >> 7) & 0xFF); if e > shift + 1 { b = (b & 0x807F) | UInt16(e - shift) << 7 } else { b = 0 }; bits[i] = b }
    }

    private func fillBytes(_ bytes: inout [UInt8], count: Int, seed: UInt64) {
        bytes.withUnsafeMutableBytes { fillBytes($0.baseAddress!, count: count, seed: seed) }
    }

    private func fillBytes(_ base: UnsafeMutableRawPointer, count: Int, seed: UInt64) {
        var rng = Random(seed)
        let pool = Self.bytePool
        let block = 4096
        pool.withUnsafeBytes { source in
            var at = 0
            while at < count {
                let from = Int(rng.next() >> 33) % (pool.count - block)
                base.advanced(by: at).copyMemory(from: source.baseAddress! + from, byteCount: min(block, count - at))
                at += block
            }
        }
    }

    private func hadamard(_ g: Int) -> [Double] {
        let h4: [Double] = [1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1]
        var h = h4, size = 4
        while size < g {
            var k = [Double](repeating: 0, count: size * 4 * size * 4)
            for i in 0..<size { for j in 0..<size { for a in 0..<4 { for b in 0..<4 {
                k[(i * 4 + a) * size * 4 + j * 4 + b] = h[i * size + j] * h4[a * 4 + b]
            } } } }
            h = k; size *= 4
        }
        let root: Double = Double(g).squareRoot()
        return h.map { (x: Double) -> Double in x / root }
    }

    /// An int8 `[k, n]` matrix as the map lays a tensor out: the values, then on a 16-byte boundary the per-column scales in
    /// `type`. Returns it with the scales as the kernel will read them (bf16 and fp16 round).
    private func int8Matrix(k: Int, n: Int, group: Int, seed: UInt64, type: DecodeKernels.Matrix.ScaleType = .float32)
        throws -> (matrix: DecodeKernels.Matrix, pages: Pages, scales: [Float]) {
        let scalesAt = (k * n + 15) / 16 * 16
        let width = type == .float32 ? 4 : 2
        let p = pages(scalesAt + n * width)
        fillBytes(p.base, count: k * n, seed: seed)
        var rng = Random(seed ^ 0xABCDEF)
        // The products are of order 1: |q| ~ 64, k terms.
        let nominal = 1 / (64 * Float(k).squareRoot())
        var read: [Float] = []
        for c in 0..<n {
            let s = nominal * (1.2 + rng.unit() * 0.8)
            switch type {
            case .float32:
                (p.base + scalesAt + 4 * c).storeBytes(of: s.bitPattern, as: UInt32.self); read.append(s)
            case .bfloat16:
                let bits = UInt16(s.bitPattern >> 16)
                (p.base + scalesAt + 2 * c).storeBytes(of: bits, as: UInt16.self); read.append(Float(bitPattern: UInt32(bits) << 16))
            default:
                let half = Float16(s)
                (p.base + scalesAt + 2 * c).storeBytes(of: half.bitPattern, as: UInt16.self); read.append(Float(half))
            }
        }
        let b = try buffer(p)
        return (.init(buffer: b, scales: b, scalesOffset: scalesAt, scaleType: type, format: .int8, k: k, n: n, group: group), p, read)
    }

    private func bf16Matrix(k: Int, n: Int, seed: UInt64) throws -> (matrix: DecodeKernels.Matrix, pages: Pages) {
        let p = pages(k * n * 2)
        fillBF16(p.base, count: k * n, seed: seed, scale: 1 / Float(k).squareRoot())
        return (.init(buffer: try buffer(p), format: .bfloat16, k: k, n: n), p)
    }

    // ── references ─────────────────────────────────────────────────────────────────────────────

    /// `C[m, n] = rotate(X) · (q · s)` in fp64 (`W = (q · s) · R`, so `x · W = (x · R) · q · s`, `R` symmetric).
    private func reference(q: UnsafePointer<Int8>, scales: [Float], x: [Float], m: Int, k: Int, n: Int, group: Int) -> [Double] {
        var xr = x.map(Double.init)
        if group > 0 {
            let h = hadamard(group)
            var out = [Double](repeating: 0, count: xr.count)
            vDSP_mmulD(xr, 1, h, 1, &out, 1, vDSP_Length(m * k / group), vDSP_Length(group), vDSP_Length(group))
            xr = out
        }
        var c = [Double](repeating: 0, count: m * n)
        let rows = 256
        var narrow = [Float](repeating: 0, count: rows * n), wide = [Double](repeating: 0, count: rows * n)
        for first in stride(from: 0, to: k, by: rows) {
            let count = min(rows, k - first)
            narrow.withUnsafeMutableBufferPointer { vDSP_vflt8(q + first * n, 1, $0.baseAddress!, 1, vDSP_Length(count * n)) }
            vDSP_vspdp(narrow, 1, &wide, 1, vDSP_Length(count * n))
            // vDSP_mmulD wants a dense A: the columns [first, first + count) of every row, gathered.
            var block = [Double](repeating: 0, count: m * count), part = [Double](repeating: 0, count: m * n)
            for r in 0..<m { for j in 0..<count { block[r * count + j] = xr[r * k + first + j] } }
            vDSP_mmulD(block, 1, wide, 1, &part, 1, vDSP_Length(m), vDSP_Length(n), vDSP_Length(count))
            vDSP_vaddD(c, 1, part, 1, &c, 1, vDSP_Length(m * n))
        }
        for r in 0..<m { for j in 0..<n { c[r * n + j] *= Double(scales[j]) } }
        return c
    }

    /// The widened path this one replaces (`WeightProducts`' `.widened`): the weight in fp32 (`Widen`), one sgemm.
    private func widened(q: UnsafeRawPointer, scales: [Float], x: [Float], m: Int, k: Int, n: Int, group: Int) -> [Float] {
        let w = UnsafeMutablePointer<Float>.allocate(capacity: k * n)
        defer { w.deallocate() }
        scales.withUnsafeBytes { s in
            if group > 0 {
                Widen.dequantizeRotated(q, rows: k, columns: n, group: group, scale: s.baseAddress!, scaleType: .float32,
                                        layout: .columns(n), into: w)
            } else {
                Widen.dequantizeColumns(q, rows: k, columns: n, scales: s.bindMemory(to: Float.self).baseAddress!,
                                        layout: .columns(n), columns: 0..<n, into: w)
            }
        }
        var c = [Float](repeating: 0, count: m * n)
        vDSP_mmul(x, 1, w, 1, &c, 1, vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
        return c
    }

    /// Runs `x` (one row) through [rotation], a product, and returns the partial sums added in fp64, `n` values.
    private func product(_ matrix: DecodeKernels.Matrix, x: [Float], splits: Int) throws -> [Float] {
        let input = try buffer(floats: x)
        let rotated = try buffer(floats: [Float](repeating: 0, count: matrix.k))
        let parts = try buffer(floats: [Float](repeating: .nan, count: splits * matrix.n))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let commands = try XCTUnwrap(queue.makeCommandBuffer())
        let e = try XCTUnwrap(commands.makeComputeCommandEncoder(dispatchType: .concurrent))
        var source = input
        if matrix.group > 0 {
            kernels.rotate(e, input, out: rotated, n: matrix.k, group: matrix.group)
            e.memoryBarrier(scope: .buffers)
            source = rotated
        }
        kernels.gemv(e, matrix, x: source, parts: parts, splits: splits)
        e.endEncoding(); commands.commit(); commands.waitUntilCompleted()
        XCTAssertNil(commands.error)
        let p = contents(parts, splits * matrix.n)
        return (0..<matrix.n).map { j in Float((0..<splits).reduce(0.0) { $0 + Double(p[$1 * matrix.n + j]) }) }
    }

    // ── the products ───────────────────────────────────────────────────────────────────────────

    /// **The in-place int8 product against fp64 and against the widened path**, at the shapes of the layers: q/o
    /// (4096 × 4096), gate/up (4096 × 12288), down (12288 × 4096), with the convrot of group 256 moved to the activation.
    /// Judged by output channel over several input rows. The rule is the checks' (`Judge`): no further from the exact value
    /// than the path it replaces, plus 25 % (its own noise floor under it).
    func testTheInt8ProductIsTheWidenedPathsArithmetic() throws {
        for (k, n) in [(4096, 4096), (4096, 12288), (12288, 4096), (4096, 1024)] {
            let m = 4
            let (matrix, p, scales) = try int8Matrix(k: k, n: n, group: 256, seed: UInt64(k + n))
            var rng = Random(UInt64(k * n))
            let x = rng.floats(m * k)
            let splits = DecodeKernels.splits(k: k, n: n, target: 192)
            var gpu: [Float] = []
            for r in 0..<m { gpu += try product(matrix, x: Array(x[(r * k)..<((r + 1) * k)]), splits: splits) }
            let exact = reference(q: p.base.assumingMemoryBound(to: Int8.self), scales: scales, x: x, m: m, k: k, n: n, group: 256)
            let eGPU = exact.withUnsafeBufferPointer { channelError(gpu, $0, columns: n) }
            XCTAssertLessThan(eGPU.median, 1e-6, "\(k)×\(n) median")
            // Cutting K any other way only moves the last bits.
            let other = try product(matrix, x: Array(x[0..<k]), splits: 1)
            XCTAssertLessThan(Array(exact[0..<n]).withUnsafeBufferPointer { channelError(other, $0, columns: n) }.worst, 1e-3)
            // The widened path (a fp32 weight and one sgemm) is slow at the largest shapes in a debug build: two of them judge it.
            guard n <= 4096, k <= 4096 else { XCTAssertLessThan(eGPU.worst, 5e-6, "\(k)×\(n)"); continue }
            let old = widened(q: p.base, scales: scales, x: x, m: m, k: k, n: n, group: 256)
            let eOld = exact.withUnsafeBufferPointer { channelError(old, $0, columns: n) }
            XCTAssertLessThanOrEqual(eGPU.worst, max(1.25 * eOld.worst, 1e-6), "\(k)×\(n): GPU \(eGPU.worst), widened \(eOld.worst)")
        }
    }

    /// The scale types a map can carry, and no rotation: bf16 and fp16 scales are read as published; the same arithmetic.
    func testTheScaleTypesAndNoRotation() throws {
        for type in [DecodeKernels.Matrix.ScaleType.bfloat16, .float16, .float32] {
            let (k, n, m) = (1024, 1040, 3)
            let (matrix, p, scales) = try int8Matrix(k: k, n: n, group: 0, seed: 77, type: type)
            var rng = Random(5)
            let x = rng.floats(m * k)
            var gpu: [Float] = []
            for r in 0..<m { gpu += try product(matrix, x: Array(x[(r * k)..<((r + 1) * k)]), splits: 5) }
            let exact = reference(q: p.base.assumingMemoryBound(to: Int8.self), scales: scales, x: x, m: m, k: k, n: n, group: 0)
            let e = exact.withUnsafeBufferPointer { channelError(gpu, $0, columns: n) }
            XCTAssertLessThan(e.worst, 5e-6, "\(type)")
        }
    }

    /// A bf16 layer (the Standard map's) is read as published and accumulated in fp32.
    func testTheBF16Product() throws {
        let (k, n, m) = (2048, 1032, 3)
        let (matrix, p) = try bf16Matrix(k: k, n: n, seed: 9)
        var rng = Random(6)
        let x = rng.floats(m * k)
        var gpu: [Float] = []
        for r in 0..<m { gpu += try product(matrix, x: Array(x[(r * k)..<((r + 1) * k)]), splits: DecodeKernels.splits(k: k, n: n, target: 192)) }
        var w = [Float](repeating: 0, count: k * n)
        Widen.bfloat16ToFloat32(source: p.base, destination: &w, count: k * n)
        var exact = [Double](repeating: 0, count: m * n)
        for r in 0..<m { for i in 0..<k { let xv = Double(x[r * k + i]); for j in 0..<n { exact[r * n + j] += xv * Double(w[i * n + j]) } } }
        let e = exact.withUnsafeBufferPointer { channelError(gpu, $0, columns: n) }
        XCTAssertLessThan(e.worst, 5e-6)
    }

    // ── the head and the argmax ────────────────────────────────────────────────────────────────

    private func runHead(rows v: Int, k: Int, head: MTLBuffer, x: [Float]) throws -> (logits: [Float], pick: [UInt32]) {
        let input = try buffer(floats: x), logits = try buffer(floats: [Float](repeating: .nan, count: v))
        let result = try buffer(floats: [0, 0, 0, 0])
        let commands = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
        let e = try XCTUnwrap(commands.makeComputeCommandEncoder(dispatchType: .concurrent))
        kernels.headProduct(e, head: head, rows: v, k: k, x: input, logits: logits)
        e.memoryBarrier(scope: .buffers)
        kernels.argmaxTop2(e, logits: logits, rows: v, result: result)
        e.endEncoding(); commands.commit(); commands.waitUntilCompleted()
        XCTAssertNil(commands.error)
        return (contents(logits, v), Array(UnsafeBufferPointer(start: result.contents().assumingMemoryBound(to: UInt32.self), count: 3)))
    }

    /// `lm_head` as published (`[vocabulary, hidden]` bf16) times an fp32 row, fp32 accumulation, against fp64 — on a
    /// vocabulary that is not a multiple of the 16 rows a threadgroup takes — and the argmax is the CPU's `Top2` on the same logits.
    func testTheHeadAndItsArgmax() throws {
        let (v, k, m) = (2503, 4096, 3)
        var rng = Random(31)
        let p = pages(v * k * 2)
        fillBF16(p.base, count: v * k, seed: 99, scale: 1 / 64)
        let head = try buffer(p)
        var w = [Float](repeating: 0, count: 1024 * k), narrow = [Double](repeating: 0, count: 1024 * k)
        var gpu: [Float] = [], exact = [Double](repeating: 0, count: m * v)
        let x = rng.floats(m * k)
        for r in 0..<m {
            let (logits, pick) = try runHead(rows: v, k: k, head: head, x: Array(x[(r * k)..<((r + 1) * k)]))
            gpu += logits
            var top = Qwen3VLGenerator.Top2()
            logits.withUnsafeBufferPointer { top.scan($0.baseAddress!, count: v, base: 0) }
            XCTAssertEqual(Int(pick[0]), top.id)
            XCTAssertEqual(Float(bitPattern: pick[1]), top.value)
            XCTAssertEqual(Float(bitPattern: pick[2]), top.second)
        }
        for first in stride(from: 0, to: v, by: 1024) {
            let rows = min(1024, v - first)
            Widen.bfloat16ToFloat32(source: p.base + 2 * first * k, destination: &w, count: rows * k)
            vDSP_vspdp(w, 1, &narrow, 1, vDSP_Length(rows * k))
            for r in 0..<m {
                let xr = x[(r * k)..<((r + 1) * k)].map(Double.init)
                var part = [Double](repeating: 0, count: rows)
                vDSP_mmulD(narrow, 1, xr, 1, &part, 1, vDSP_Length(rows), 1, vDSP_Length(k))
                for j in 0..<rows { exact[r * v + first + j] = part[j] }
            }
        }
        let e = exact.withUnsafeBufferPointer { channelError(gpu, $0, columns: v) }
        XCTAssertLessThan(e.worst, 5e-6)
    }

    /// **Greedy ties go to the lowest identifier, and the runner-up counts a tie** (a gap of 0), whichever thread of the
    /// scan holds which: the largest at ids 2000 and 3000 (threads 976 and 952 of 1024), the lowest id wins.
    func testTheArgmaxTieRuleAndGap() throws {
        let n = 151936
        func run(_ logits: [Float]) throws -> (id: Int, value: Float, second: Float) {
            let buf = try buffer(floats: logits), result = try buffer(floats: [0, 0, 0, 0])
            let commands = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
            let e = try XCTUnwrap(commands.makeComputeCommandEncoder())
            kernels.argmaxTop2(e, logits: buf, rows: n, result: result)
            e.endEncoding(); commands.commit(); commands.waitUntilCompleted()
            let w = result.contents().assumingMemoryBound(to: UInt32.self)
            return (Int(w[0]), Float(bitPattern: w[1]), Float(bitPattern: w[2]))
        }
        var rng = Random(3)
        var base = rng.floats(n).map { $0 * 5 }
        base[3000] = 9; base[2000] = 9; base[150_000] = 9
        var r = try run(base)
        XCTAssertEqual(r.id, 2000); XCTAssertEqual(r.value, 9); XCTAssertEqual(r.second, 9)
        base[2000] = 8.5; base[3000] = 8.75; base[150_000] = 9.25
        r = try run(base)
        XCTAssertEqual(r.id, 150_000); XCTAssertEqual(r.value - r.second, 0.5)
        base[n - 1] = 20
        r = try run(base)
        XCTAssertEqual(r.id, n - 1); XCTAssertEqual(r.value, 20); XCTAssertEqual(r.second, 9.25)
        base[0] = 20
        r = try run(base)
        XCTAssertEqual(r.id, 0); XCTAssertEqual(r.second, 20)
        var cpu = Qwen3VLGenerator.Top2()
        base.withUnsafeBufferPointer { cpu.scan($0.baseAddress!, count: n, base: 0) }
        XCTAssertEqual(cpu.id, r.id); XCTAssertEqual(cpu.second, r.second)
        // A short vocabulary (fewer logits than threads).
        let few = try run(base)
        XCTAssertEqual(few.id, 0)
    }

    // ── the rest of a layer ────────────────────────────────────────────────────────────────────

    private func oneCommand(_ body: (MTLComputeCommandEncoder) -> Void) throws {
        let commands = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
        let e = try XCTUnwrap(commands.makeComputeCommandEncoder(dispatchType: .concurrent))
        body(e)
        e.endEncoding(); commands.commit(); commands.waitUntilCompleted()
        XCTAssertNil(commands.error)
    }

    private func rotated(_ rows: [Double], group: Int) -> [Double] {
        guard group > 0 else { return rows }
        var out = [Double](repeating: 0, count: rows.count)
        let h = hadamard(group)
        vDSP_mmulD(rows, 1, h, 1, &out, 1, vDSP_Length(rows.count / group), vDSP_Length(group), vDSP_Length(group))
        return out
    }

    /// The residual of the previous product's partials, the RMSNorm, then the rotation, in one threadgroup — `Ops.rmsNorm`'s
    /// arithmetic order, judged against fp64; `x` ends as `x + Σ parts`, in fp32.
    func testNormRotateWithItsResidual() throws {
        let (n, rows, parts) = (4096, 5, 3)
        var rng = Random(11)
        for group in [256, 0] {
            var got: [Float] = [], exact: [Double] = []
            let w = rng.floats(n).map { 1 + $0 / 2 }
            for _ in 0..<rows {
                let x = rng.floats(n), p = rng.floats(parts * n, scale: 0.5)
                let xb = try buffer(floats: x), pb = try buffer(floats: p), wb = try buffer(floats: w)
                let out = try buffer(floats: [Float](repeating: 0, count: n))
                try oneCommand { kernels.normRotate($0, x: xb, parts: pb, count: parts, stride: n, weight: wb, weightOffset: 0,
                                                    out: out, n: n, group: group, eps: 1e-6) }
                let sum = (0..<n).map { i in Double(x[i]) + (0..<parts).reduce(0.0) { $0 + Double(p[$1 * n + i]) } }
                let xNew = contents(xb, n)
                XCTAssertLessThan((0..<n).map { abs(Double(xNew[$0]) - sum[$0]) }.max()!, 2e-7 * 4)
                let mean = sum.reduce(0) { $0 + $1 * $1 } / Double(n)
                let normed = (0..<n).map { sum[$0] / (mean + 1e-6).squareRoot() * Double(w[$0]) }
                got += contents(out, n); exact += rotated(normed, group: group)
            }
            let e = exact.withUnsafeBufferPointer { channelError(got, $0, columns: n) }
            XCTAssertLessThan(e.worst, 5e-6, "group \(group)")
        }
    }

    /// `silu(gate) · up` from two products' partials, then the rotation by groups of 256 (or none).
    func testSwigluRotate() throws {
        let n = 12288
        var rng = Random(21)
        for group in [256, 0] {
            var got: [Float] = [], exact: [Double] = []
            for _ in 0..<3 {
                let g = rng.floats(2 * n, scale: 3), u = rng.floats(3 * n, scale: 2)
                let gb = try buffer(floats: g), ub = try buffer(floats: u), out = try buffer(floats: [Float](repeating: 0, count: n))
                try oneCommand { kernels.swiglu($0, gate: gb, gateParts: 2, up: ub, upParts: 3, out: out, n: n, group: group) }
                let value = (0..<n).map { i -> Double in
                    let a = Double(g[i]) + Double(g[n + i]), b = (0..<3).reduce(0.0) { $0 + Double(u[$1 * n + i]) }
                    return a / (1 + exp(-a)) * b
                }
                got += contents(out, n); exact += rotated(value, group: group)
            }
            let e = exact.withUnsafeBufferPointer { channelError(got, $0, columns: n) }
            XCTAssertLessThan(e.worst, 5e-6, "group \(group)")
        }
    }

    /// The sum of the projections' partials, QK-norm and RoPE of q and k, K and V written at the cache row: the CPU's
    /// `Ops.rmsNormRope` on the summed rows, to rounding.
    func testQKVPostIsTheCPUsNormAndRope() throws {
        let (heads, kvHeads, d) = (8, 2, 128)
        var rng = Random(41)
        let a = heads * d, kv = kvHeads * d, qParts = 3, kParts = 2, vParts = 4
        let qp = rng.floats(qParts * a), kp = rng.floats(kParts * kv), vp = rng.floats(vParts * kv)
        let qw = rng.floats(d).map { 1 + $0 / 2 }, kw = rng.floats(d).map { 1 + $0 / 2 }
        // Typed: Swift 6.3's checker (the CI's) gives up on `Float($0) * 0.37 + 0.1` in a closure.
        let angles: [Float] = (0..<(d / 2)).map { (i: Int) -> Float in Float(i) * 0.37 + 0.1 }
        let freqs: [Float] = angles.flatMap { (angle: Float) -> [Float] in [cosf(angle), sinf(angle)] }
        let q = try buffer(floats: [Float](repeating: 0, count: a)), cache = try buffer(floats: [Float](repeating: .nan, count: 6 * kv))
        let row = 2
        try oneCommand {
            kernels.post($0, q: (try! buffer(floats: qp), qParts), k: (try! buffer(floats: kp), kParts), v: (try! buffer(floats: vp), vParts),
                         qNorm: try! buffer(floats: qw), kNorm: try! buffer(floats: kw), normOffsets: (0, 0),
                         freqs: try! buffer(floats: freqs), out: q, keys: (cache, row * kv * 4), values: (cache, (3 + row) * kv * 4),
                         heads: heads, kvHeads: kvHeads, headDim: d, eps: 1e-6)
        }
        func sum(_ p: [Float], _ parts: Int, _ width: Int) -> [Float] { (0..<width).map { i in (0..<parts).reduce(0) { $0 + p[$1 * width + i] } } }
        var wantQ = sum(qp, qParts, a), wantK = sum(kp, kParts, kv)
        Ops.rmsNormRope(&wantQ, weight: qw, freqs: freqs, sequence: 1, heads: heads, headDim: d, eps: 1e-6)
        Ops.rmsNormRope(&wantK, weight: kw, freqs: freqs, sequence: 1, heads: kvHeads, headDim: d, eps: 1e-6)
        let all = contents(cache, 6 * kv)
        let eq = wantQ.map(Double.init).withUnsafeBufferPointer { channelError(contents(q, a), $0, columns: d) }
        let ek = wantK.map(Double.init).withUnsafeBufferPointer { channelError(Array(all[(row * kv)..<((row + 1) * kv)]), $0, columns: d) }
        XCTAssertLessThan(eq.worst, 2e-6); XCTAssertLessThan(ek.worst, 2e-6)
        XCTAssertEqual(Array(all[((3 + row) * kv)..<((4 + row) * kv)]).map(\.bitPattern), sum(vp, vParts, kv).map(\.bitPattern),
                       "V is the sum of its partials, in the kernel's order")
        XCTAssertTrue(all[0..<kv].allSatisfy { $0.isNaN }, "nothing is written at another row")
    }

    /// One query row over a cache of `length` rows, GQA, in fp64 — lengths below the eight simdgroups, not a multiple of them,
    /// and long.
    func testCachedAttentionOfOneRow() throws {
        let (heads, kvHeads, d) = (8, 2, 128)
        for length in [1, 5, 8, 301, 2051] {
            var rng = Random(UInt64(length))
            let q = rng.floats(heads * d), keys = rng.floats(length * kvHeads * d), values = rng.floats(length * kvHeads * d)
            let out = try buffer(floats: [Float](repeating: .nan, count: heads * d))
            let qb = try buffer(floats: q), kb = try buffer(floats: keys), vb = try buffer(floats: values)
            try oneCommand { kernels.attend($0, q: qb, keys: (kb, 0), values: (vb, 0), out: out, heads: heads, kvHeads: kvHeads, length: length) }
            var exact = [Double](repeating: 0, count: heads * d)
            for h in 0..<heads {
                let g = h / (heads / kvHeads)
                var s = (0..<length).map { j in (0..<d).reduce(0.0) { $0 + Double(q[h * d + $1]) * Double(keys[(j * kvHeads + g) * d + $1]) } / Double(d).squareRoot() }
                let top = s.max()!
                s = s.map { exp($0 - top) }
                let total = s.reduce(0, +)
                for j in 0..<length { for c in 0..<d { exact[h * d + c] += s[j] / total * Double(values[(j * kvHeads + g) * d + c]) } }
            }
            let e = exact.withUnsafeBufferPointer { channelError(contents(out, heads * d), $0, columns: d) }
            XCTAssertLessThan(e.worst, 5e-6, "length \(length)")
        }
    }

    // ── one whole step ─────────────────────────────────────────────────────────────────────────

    private enum Style { case rotated, plain, bf16 }

    /// **A tiny layer stack stepped on the GPU gives the CPU pass's tokens** (`Qwen3VLGenerator.generate`'s passes, written
    /// here from the same `Ops`/`Widen`/`cachedAttention` functions): a prompt is read by the CPU pass, its K/V cache is handed
    /// to the decoder, then six steps are taken by each, every one feeding its own token to the next — same tokens, hidden
    /// rows within rounding. Three layers over two command buffers (`layersPerCommand = 2`), head dimension 128, GQA 4/2,
    /// int8 convrot, int8 without rotation, and bf16.
    func testAWholeStepIsTheCPUPassesToken() throws {
        for style in [Style.rotated, .plain, .bf16] { try wholeStep(style) }
    }

    private func wholeStep(_ style: Style) throws {
        let shape = Qwen3VLDecoder.Shape(hidden: 512, layers: 3, heads: 4, kvHeads: 2, headDim: 128, intermediate: 1024,
                                         vocabulary: 1000, eps: 1e-6)
        let (h, a, kv, i, d, v) = (shape.hidden, shape.attentionWidth, shape.kvWidth, shape.intermediate, shape.headDim, shape.vocabulary)
        var rng = Random(UInt64(h) &+ (style == .bf16 ? 5 : style == .plain ? 3 : 1))
        let group = style == .rotated ? 256 : 0

        // Weights: the decoder's matrices, and the fp32 widened copies the CPU pass multiplies by.
        var cpu: [[[Float]]] = []   // layer → [q, k, v, o, gate, up, down]
        var layers: [Qwen3VLDecoder.Layer] = []
        var seed: UInt64 = 100
        func product(_ k: Int, _ n: Int) throws -> (DecodeKernels.Matrix, [Float]) {
            seed += 1
            if style == .bf16 {
                let (m, p) = try bf16Matrix(k: k, n: n, seed: seed)
                var w = [Float](repeating: 0, count: k * n)
                Widen.bfloat16ToFloat32(source: p.base, destination: &w, count: k * n)
                return (m, w)
            }
            let (m, p, scales) = try int8Matrix(k: k, n: n, group: group, seed: seed)
            var w = [Float](repeating: 0, count: k * n)
            scales.withUnsafeBytes { s in
                if group > 0 {
                    Widen.dequantizeRotated(p.base, rows: k, columns: n, group: group, scale: s.baseAddress!, scaleType: .float32,
                                            layout: .columns(n), into: &w)
                } else {
                    Widen.dequantizeColumns(p.base, rows: k, columns: n, scales: s.bindMemory(to: Float.self).baseAddress!,
                                            layout: .columns(n), columns: 0..<n, into: &w)
                }
            }
            return (m, w)
        }
        var norms: [[[Float]]] = []   // layer → [input, post, qNorm, kNorm]
        for _ in 0..<shape.layers {
            let ms = [try product(h, a), try product(h, kv), try product(h, kv), try product(a, h),
                      try product(h, i), try product(h, i), try product(i, h)]
            layers.append(.init(q: ms[0].0, k: ms[1].0, v: ms[2].0, o: ms[3].0, gate: ms[4].0, up: ms[5].0, down: ms[6].0))
            cpu.append(ms.map(\.1))
            norms.append([rng.floats(h).map { 1 + $0 / 2 }, rng.floats(h).map { 1 + $0 / 2 },
                          rng.floats(d).map { 1 + $0 / 2 }, rng.floats(d).map { 1 + $0 / 2 }])
        }
        let finalNorm = rng.floats(h).map { 1 + $0 / 2 }
        let headPages = pages(v * h * 2)
        fillBF16(headPages.base, count: v * h, seed: 98, scale: 1 / 4)
        var headWide = [Float](repeating: 0, count: v * h)
        Widen.bfloat16ToFloat32(source: headPages.base, destination: &headWide, count: v * h)
        let table = rng.floats(v * h)

        var tuning = Qwen3VLDecoder.Tuning()
        tuning.layersPerCommand = 2
        let decoder = try Qwen3VLDecoder(device: device, shape: shape, layers: layers, head: try buffer(headPages), tuning: tuning) { out in
            for l in 0..<shape.layers {
                for (part, w) in norms[l].enumerated() { (out + Qwen3VLDecoder.normOffset(shape, layer: l, part: part)).update(from: w, count: w.count) }
            }
            (out + Qwen3VLDecoder.finalNormOffset(shape)).update(from: finalNorm, count: h)
        }

        // The CPU pass of `generate`, for `used` rows from cache row `start`.
        let prompt = 5, steps = 6, capacity = prompt + steps + 1
        let layerStride = capacity * kv
        let cacheCPU = pages(2 * shape.layers * layerStride * 4)
        func keys(_ cache: Pages, _ l: Int) -> UnsafeMutablePointer<Float> { cache.floats() + l * layerStride }
        func values(_ cache: Pages, _ l: Int) -> UnsafeMutablePointer<Float> { cache.floats() + (shape.layers + l) * layerStride }
        func matmul(_ x: [Float], rows: Int, _ w: [Float], k: Int, n: Int, transposed: Bool = false) -> [Float] {
            var c = [Float](repeating: 0, count: rows * n)
            // `transposed`: w is [n, k] (the head), so the product is w · xᵀ for one row.
            if transposed { vDSP_mmul(w, 1, x, 1, &c, 1, vDSP_Length(n), 1, vDSP_Length(k)) } else {
                vDSP_mmul(x, 1, w, 1, &c, 1, vDSP_Length(rows), vDSP_Length(n), vDSP_Length(k))
            }
            return c
        }
        func cpuPass(_ x: inout [Float], used: Int, start: Int) {
            var freqs = [Float](repeating: 0, count: used * d)
            TextEncoder.writeRope(into: &freqs, positions: [Array(start..<(start + used))], count: used, theta: 1_000_000, headDim: d, sections: nil)
            let scores = UnsafeMutablePointer<Float>.allocate(capacity: 8 * 64 * capacity)
            defer { scores.deallocate() }
            for l in 0..<shape.layers {
                var normed = [Float](repeating: 0, count: used * h)
                Ops.rmsNorm(x, weight: norms[l][0], into: &normed, rows: used, columns: h, eps: shape.eps)
                var q = matmul(normed, rows: used, cpu[l][0], k: h, n: a), k = matmul(normed, rows: used, cpu[l][1], k: h, n: kv)
                let vv = matmul(normed, rows: used, cpu[l][2], k: h, n: kv)
                Ops.rmsNormRope(&q, weight: norms[l][2], freqs: freqs, sequence: used, heads: shape.heads, headDim: d, eps: shape.eps)
                Ops.rmsNormRope(&k, weight: norms[l][3], freqs: freqs, sequence: used, heads: shape.kvHeads, headDim: d, eps: shape.eps)
                (keys(cacheCPU, l) + start * kv).update(from: k, count: used * kv)
                (values(cacheCPU, l) + start * kv).update(from: vv, count: used * kv)
                var attended = [Float](repeating: 0, count: used * a)
                Qwen3VLGenerator.cachedAttention(q: q, rows: used, keys: keys(cacheCPU, l), values: values(cacheCPU, l), offset: start,
                                                 heads: shape.heads, kvHeads: shape.kvHeads, headDim: d, into: &attended,
                                                 scratch: scores, scratchFloats: 8 * 64 * capacity)
                let o = matmul(attended, rows: used, cpu[l][3], k: a, n: h)
                var residual = x
                vDSP_vadd(x, 1, o, 1, &residual, 1, vDSP_Length(used * h))
                x = residual
                Ops.rmsNorm(x, weight: norms[l][1], into: &normed, rows: used, columns: h, eps: shape.eps)
                var gate = matmul(normed, rows: used, cpu[l][4], k: h, n: i)
                let up = matmul(normed, rows: used, cpu[l][5], k: h, n: i)
                var scratch = [Float](repeating: 0, count: used * i)
                Ops.siluGate(gate, up, into: &gate, count: used * i, scratch: &scratch)
                let down = matmul(gate, rows: used, cpu[l][6], k: i, n: h)
                residual = x
                vDSP_vadd(x, 1, down, 1, &residual, 1, vDSP_Length(used * h))
                x = residual
            }
        }
        func cpuPick(_ row: [Float]) -> Qwen3VLGenerator.Top2 {
            var last = row
            Ops.rmsNorm(row, weight: finalNorm, into: &last, rows: 1, columns: h, eps: shape.eps)
            let logits = matmul(last, rows: 1, headWide, k: h, n: v, transposed: true)
            var top = Qwen3VLGenerator.Top2()
            logits.withUnsafeBufferPointer { top.scan($0.baseAddress!, count: v, base: 0) }
            return top
        }

        // The prompt, by the CPU pass.
        var ids = (0..<prompt).map { _ in Int(rng.next() >> 33) % v }
        var x = [Float](repeating: 0, count: prompt * h)
        for (r, id) in ids.enumerated() { x.replaceSubrange((r * h)..<((r + 1) * h), with: table[(id * h)..<((id + 1) * h)]) }
        cpuPass(&x, used: prompt, start: 0)
        let cacheGPU = pages(2 * shape.layers * layerStride * 4)
        cacheGPU.base.copyMemory(from: cacheCPU.base, byteCount: cacheCPU.capacity)
        try decoder.attach(cache: cacheGPU, layerStride: layerStride)

        // The first token, from the prompt's last row.
        let lastRow = Array(x[((prompt - 1) * h)..<(prompt * h)])
        decoder.hidden.update(from: lastRow, count: h)
        var pick = try decoder.pick()
        var want = cpuPick(lastRow)
        XCTAssertEqual(pick.id, want.id, "\(style): the first token")
        XCTAssertEqual(pick.value, want.value, accuracy: 1e-4 * abs(want.value) + 1e-5)
        var gpuRows: [Float] = [], cpuRows: [Float] = []
        for s in 0..<steps {
            let token = want.id
            ids.append(token)
            // CPU
            var row = Array(table[(token * h)..<((token + 1) * h)])
            cpuPass(&row, used: 1, start: prompt + s)
            // GPU
            decoder.hidden.update(from: Array(table[(token * h)..<((token + 1) * h)]), count: h)
            TextEncoder.writeRope(into: decoder.frequencies, positions: [[prompt + s]], count: 1, theta: 1_000_000, headDim: d, sections: nil)
            pick = try decoder.step(row: prompt + s)
            gpuRows += Array(UnsafeBufferPointer(start: decoder.hidden, count: h)); cpuRows += row
            want = cpuPick(row)
            XCTAssertEqual(pick.id, want.id, "\(style): step \(s)")
            XCTAssertEqual(pick.value - pick.second, want.value - want.second, accuracy: 1e-3 * abs(want.value) + 1e-4, "\(style): the gap, step \(s)")
        }
        let e = cpuRows.map(Double.init).withUnsafeBufferPointer { channelError(gpuRows, $0, columns: h) }
        XCTAssertLessThan(e.worst, 1e-4, "\(style): the residual stream, channel by channel")
        XCTAssertGreaterThan(decoder.lastStep.gpu, 0)
    }

    /// **The decoder over the resident tensors of real map files** (`Qwen3VLDecoder.make`): a one-layer encoder map with the
    /// Compact's tensors (int8 `[k, n]`, fp32 scales per column inside the tensor, convrot 256) and a head map, written by
    /// the forge's own writer, read into resident pages and wrapped without a copy. It steps, and says why it cannot when
    /// the weights are not resident.
    func testMakeWrapsTheResidentTensorsOfAMap() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-decoder-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let (h, a, kv, i, v) = (512, 512, 256, 1024, 1000)
        var rng = Random(61)
        func int8(_ name: String, _ k: Int, _ n: Int) -> MapWriter.Tensor {
            var t = MapWriter.Tensor(name: name, shape: [k, n], dtype: .int8, scale: .init(dtype: .float32, shape: [n]), transposed: true) {
                var bytes = [UInt8](repeating: 0, count: k * n)
                self.fillBytes(&bytes, count: k * n, seed: UInt64(k * 31 + n))
                let value: Float = 1 / (64 * Float(k).squareRoot())
                let one: [UInt8] = withUnsafeBytes(of: value.bitPattern.littleEndian) { Array($0) }
                let scale: [UInt8] = (0..<n).flatMap { (_: Int) -> [UInt8] in one }
                return .init(values: bytes, scale: scale, absMax: 1)
            }
            t.rotation = 256
            return t
        }
        func bf16(_ name: String, _ shape: [Int], scale: Float = 1) -> MapWriter.Tensor {
            let values = rng.floats(shape.reduce(1, *)).map { $0 * scale }
            return .init(name: name, shape: shape, dtype: .bfloat16, transposed: nil) { values }
        }
        let p = "layers.0."
        let encoder = [int8(p + "self_attn.q_proj.weight", h, a), int8(p + "self_attn.k_proj.weight", h, kv), int8(p + "self_attn.v_proj.weight", h, kv),
                       int8(p + "self_attn.o_proj.weight", a, h), int8(p + "mlp.gate_proj.weight", h, i), int8(p + "mlp.up_proj.weight", h, i),
                       int8(p + "mlp.down_proj.weight", i, h), bf16(p + "input_layernorm.weight", [h]), bf16(p + "post_attention_layernorm.weight", [h]),
                       bf16(p + "self_attn.q_norm.weight", [128]), bf16(p + "self_attn.k_norm.weight", [128])]
        let config: OrderedJSON = .object([.init("hidden_size", .integer(h)), .init("num_attention_heads", .integer(4)),
                                           .init("num_key_value_heads", .integer(2)), .init("head_dim", .integer(128)),
                                           .init("intermediate_size", .integer(i)), .init("num_hidden_layers", .integer(1)),
                                           .init("rms_norm_eps", .real(1e-6)), .init("rope_theta", .real(1e6))])
        let encoderPath = root.appendingPathComponent("encoder.silicon").path, headPath = root.appendingPathComponent("head.silicon").path
        _ = try MapWriter.write(to: encoderPath, tensors: encoder, header: { _ in
            [.init("format", 1), .init("page", .integer(MapWriter.page)), .init("linear_weights_transposed", true),
             .init("last_layer", .integer(0)), .init("config", config), .init("order", .list(encoder.map { .string($0.name) }))]
        })
        let head = [bf16("lm_head.weight", [v, h], scale: 0.1), bf16("norm.weight", [h])]
        _ = try MapWriter.write(to: headPath, tensors: head, header: { _ in
            [.init("format", 1), .init("kind", "qwen3-vl-lm-head"), .init("page", .integer(MapWriter.page)),
             .init("linear_weights_transposed", false), .init("order", .list(head.map { .string($0.name) }))]
        })
        let map = try Artifact(path: encoderPath), headMap = try Artifact(path: headPath)
        let shape = try TextEncoder.Config(header: map.header)
        // Not resident: refused, with the reason.
        XCTAssertThrowsError(try Qwen3VLDecoder.make(encoder: map, head: headMap, config: shape, vocabulary: v)) {
            XCTAssertTrue("\($0)".contains("not resident"), "\($0)")
        }
        let prefixes = Qwen3VLGenerator.residentPrefixes(config: shape)
        try map.keepResident(prefixes: prefixes.encoder)
        try headMap.keepResident(prefixes: prefixes.head)
        let decoder = try Qwen3VLDecoder.make(encoder: map, head: headMap, config: shape, vocabulary: v)
        XCTAssertEqual(decoder.shape.layers, 1)
        // A step with a fresh cache: finite, a token inside the vocabulary, the gap is a gap.
        let cache = pages(2 * 4 * 2 * 128 * 4)
        try decoder.attach(cache: cache, layerStride: 4 * 256)
        for j in 0..<h { decoder.hidden[j] = rng.unit() }
        TextEncoder.writeRope(into: decoder.frequencies, positions: [[0]], count: 1, theta: 1_000_000, headDim: 128, sections: nil)
        let pick = try decoder.step(row: 0)
        XCTAssertTrue((0..<v).contains(pick.id)); XCTAssertTrue(pick.value.isFinite); XCTAssertGreaterThanOrEqual(pick.gap, 0)
        XCTAssertTrue((0..<h).allSatisfy { decoder.hidden[$0].isFinite })
        decoder.detachCache()
    }

    /// What the decoder charges is what it maps, from shapes: the real model's scratch is a few megabytes.
    func testTheScratchIsAFunctionOfTheShapes() {
        let real = Qwen3VLDecoder.Shape(hidden: 4096, layers: 36, heads: 32, kvHeads: 8, headDim: 128, intermediate: 12288,
                                        vocabulary: 151_936, eps: 1e-6)
        let bytes = Qwen3VLDecoder.scratchBytes(real)
        XCTAssertEqual(bytes, Arena.capacity(slices: Qwen3VLDecoder.slices(real).map { $0.count * 4 }))
        XCTAssertLessThan(bytes, 16 << 20)
        XCTAssertGreaterThan(bytes, 151_936 * 4, "the logits are in it")
        XCTAssertEqual(Qwen3VLDecoder.normCount(real), 36 * (2 * 4096 + 2 * 128) + 4096)
        // Every product is cut so that the GPU is full, and never finer than eight rows to a simdgroup.
        XCTAssertEqual(DecodeKernels.splits(k: 4096, n: 1024, target: 192), 64)
        XCTAssertEqual(DecodeKernels.splits(k: 4096, n: 4096, target: 192), 24)
        XCTAssertEqual(DecodeKernels.splits(k: 64, n: 4096, target: 192), 1)
    }
}
