@preconcurrency import Accelerate
import Foundation
import Metal
import Siliconed

// **The kernels, judged without weights** — `siliconed check kernels [NAME]`.
//
// Each is confronted with its definition: double precision, `cblas_sgemm`, the whole computation it
// slices, or (`rope`, `resample`) the reference's own tables, written by torch on the CPU. No model is
// loaded; the set runs in seconds and is what continuous integration can run.

enum KernelStages {
    static let all: [Stage] = [widen, arena, elementwise, gelu, sdpa, sdpaTiles, gemmSplit, rope, resample]

    // MARK: - The widener and the arena

    static let widen = Stage(
        model: "kernels", name: "widen",
        proves: "bf16 → fp32 widening and its inverse are exact, on the values a naive shift gets wrong",
        oracle: nil, weights: [], rule: "bit for bit"
    ) { _ in
        let report = Report(title: "kernels widen")
        // Signed zeros, subnormals, infinities, the largest finite bf16.
        let cases: [(UInt16, Float)] = [
            (0x0000, 0.0), (0x8000, -0.0), (0x3F80, 1.0), (0xBF80, -1.0),
            (0x4049, 3.140625),                       // π rounded to bf16
            (0x7F80, .infinity), (0xFF80, -.infinity),
            (0x7F7F, Float(bitPattern: 0x7F7F0000)),  // the largest finite bf16
            (0x0080, Float(bitPattern: 0x00800000)),  // the smallest normal
            (0x0001, Float(bitPattern: 0x00010000)),  // subnormal
        ]
        var output = [Float](repeating: .nan, count: cases.count)
        cases.map(\.0).withUnsafeBytes { src in output.withUnsafeMutableBytes { dst in
            Widen.bfloat16ToFloat32(source: src.baseAddress!, destination: dst.baseAddress!, count: cases.count) } }
        let wrong = zip(output, cases).filter { $0.bitPattern != $1.1.bitPattern }
        report.add(Judge.identity("special values", differing: wrong.count, of: cases.count))

        // fp32 → bf16 → fp32 is idempotent on a value that already is a bf16, as every weight is.
        let source = (0..<4096).map { Float(bitPattern: UInt32($0 &* 2_654_435_761 & 0x7F7F0000)) }
        var packed = [UInt16](repeating: 0, count: source.count), back = [Float](repeating: 0, count: source.count)
        source.withUnsafeBytes { s in packed.withUnsafeMutableBytes { d in
            Widen.float32ToBfloat16(source: s.baseAddress!, destination: d.baseAddress!, count: source.count) } }
        packed.withUnsafeBytes { s in back.withUnsafeMutableBytes { d in
            Widen.bfloat16ToFloat32(source: s.baseAddress!, destination: d.baseAddress!, count: source.count) } }
        report.add(Judge.identity("round trip", differing: zip(back, source).filter { $0.bitPattern != $1.bitPattern }.count,
                                  of: source.count))

        // The lengths the vectorized loop does not take whole.
        for count in [1, 7, 8, 9, 15] {
            let ones = [UInt16](repeating: 0x3F80, count: count)
            var got = [Float](repeating: .nan, count: count)
            ones.withUnsafeBytes { s in got.withUnsafeMutableBytes { d in
                Widen.bfloat16ToFloat32(source: s.baseAddress!, destination: d.baseAddress!, count: count) } }
            report.add(Judge.identity("length \(count)", differing: got.filter { $0 != 1 }.count, of: count))
        }
        return report
    }

    static let arena = Stage(
        model: "kernels", name: "arena",
        proves: "the arena hands out distinct page-aligned slices, refuses duplicates and growth",
        oracle: nil, weights: [], rule: "each property holds"
    ) { _ in
        let report = Report(title: "kernels arena")
        let arena = try Arena(capacity: 1 << 20)
        let a = try arena.reserve("residual", bytes: 4096)
        let b = try arena.reserve("reserve", bytes: 4096)
        report.add(Judge.holds("two distinct slices", a != b))
        report.add(Judge.holds("page alignment", Int(bitPattern: a) % Arena.alignment == 0 && Int(bitPattern: b) % Arena.alignment == 0))
        report.add(Judge.holds("found by its name", arena.pointer("reserve") == b))
        report.add(Judge.holds("refuses a duplicate", (try? arena.reserve("residual", bytes: 16)) == nil))
        report.add(Judge.holds("refuses to grow", (try? arena.reserve("oversize", bytes: 1 << 30)) == nil))
        let footprint = Arena.processFootprint()
        report.add(Judge.holds("phys_footprint readable", footprint > 0,
                               note: String(format: "%.0f MB", Double(footprint) / 1e6)))
        return report
    }

    // MARK: - Elementwise kernels against double precision

    static let elementwise = Stage(
        model: "kernels", name: "elementwise",
        proves: "the GPU's SwiGLU, gate and gated residual are as close to double precision as the CPU's; the residual rounds twice, as torch",
        oracle: nil, weights: [], rule: "GPU worst ulp ≤ max(CPU's, 1), mean ≤ CPU's + 5 %; residual bit for bit"
    ) { _ in
        let report = Report(title: "kernels elementwise")
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw GEMM.Failure.noDevice
        }
        let gpu = try ElementwiseGPU(device: device, queue: queue)
        var random = SplitMix(seed: 0x5EED_4B7E_A2)
        func buffer(_ v: [Float]) -> MTLBuffer { device.makeBuffer(bytes: v, length: v.count * 4, options: .storageModeShared)! }
        func read(_ b: MTLBuffer, _ n: Int) -> [Float] {
            Array(UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: Float.self), count: n))
        }
        /// Against the exact: worst in ulp of the output, mean of |deviation| / |exact|.
        func deviation(_ y: [Float], _ exact: [Double]) -> (worst: Double, mean: Double, unfinished: Int) {
            var worst = 0.0, sum = 0.0, unfinished = 0
            for i in y.indices {
                guard y[i].isFinite else { unfinished += 1; continue }
                if abs(exact[i]) < Double(Float.leastNormalMagnitude) { continue }
                let d = abs(Double(y[i]) - exact[i])
                worst = max(worst, d / Double(Float(abs(exact[i])).ulp))
                sum += d / abs(exact[i])
            }
            return (worst, sum / Double(y.count), unfinished)
        }
        func judge(_ name: String, cpu: [Float], gpu: [Float], exact: [Double]) {
            let c = deviation(cpu, exact), g = deviation(gpu, exact)
            report.add(Judge.bound("\(name) worst", g.unfinished == 0 ? g.worst : .infinity, max(c.worst, 1), unit: "ulp",
                                   note: String(format: "CPU %.2f ulp", c.worst)))
            report.add(Judge.bound("\(name) mean", g.mean, 1.05 * c.mean, unit: "relative",
                                   note: String(format: "CPU %.2e", c.mean)))
        }

        // SwiGLU and gate, over [−20, 20]: beyond, e^{−a} overflows identically on both sides.
        let n = 1 << 22
        let a = (0..<n).map { _ in Float(max(-20, min(20, 4 * random.normal()))) }
        let b = (0..<n).map { _ in Float(2 * random.normal()) }
        do {
            var cpu = a, scratch = [Float](repeating: 0, count: n)
            cpu.withUnsafeMutableBufferPointer { o in a.withUnsafeBufferPointer { ap in b.withUnsafeBufferPointer { bp in
                scratch.withUnsafeMutableBufferPointer { s in
                    Ops.siluGate(ap.baseAddress!, bp.baseAddress!, into: o.baseAddress!, count: n, scratch: s.baseAddress!)
                } } } }
            let ab = buffer(a), bb = buffer(b)
            _ = gpu.swiglu((ab, 0), (bb, 0), count: n)
            judge("swiglu", cpu: cpu, gpu: read(ab, n),
                  exact: (0..<n).map { (i: Int) -> Double in
                      let x = Double(a[i]), y = Double(b[i])
                      return x / (1 + exp(-x)) * y
                  })
        }
        do {
            var cpu = b, scratch = [Float](repeating: 0, count: n)
            cpu.withUnsafeMutableBufferPointer { x in a.withUnsafeBufferPointer { g in scratch.withUnsafeMutableBufferPointer { s in
                Krea2Ops.sigmoidGate(x.baseAddress!, gate: g.baseAddress!, count: n, scratch: s.baseAddress!)
            } } }
            let xb = buffer(b), gb = buffer(a)
            _ = gpu.carries((xb, 0), (gb, 0), count: n)
            judge("gate", cpu: cpu, gpu: read(xb, n), exact: (0..<n).map { (i: Int) -> Double in
                let g = Double(a[i])
                return Double(b[i]) / (1 + exp(-g))
            })
        }
        // The gated residual: rows of 6 144 at the residual's scale, with ×50 spikes.
        let (rows, d) = (512, 6144)
        let x = (0..<(rows * d)).map { i -> Float in Float(random.normal() * (i % 997 == 0 ? 250 : 5)) }
        let y = (0..<(rows * d)).map { _ in Float(3 * random.normal()) }
        let carries = (0..<d).map { _ in Float(0.3 * random.normal()) }
        var cpu = x
        cpu.withUnsafeMutableBufferPointer { xp in y.withUnsafeBufferPointer { yp in carries.withUnsafeBufferPointer { gp in
            Ops.gatedResidual(xp.baseAddress!, plus: yp.baseAddress!, gate: gp.baseAddress!, rows: rows, columns: d)
        } } }
        let xb = buffer(x)
        _ = gpu.residual((xb, 0), plus: (buffer(y), 0), carries: (buffer(carries), 0), rows: rows, columns: d)
        let got = read(xb, rows * d)
        judge("residual", cpu: cpu, gpu: got, exact: (0..<(rows * d)).map { Double(x[$0]) + Double(carries[$0 % d]) * Double(y[$0]) })
        // `x + gate * y` rounded twice, as torch: one fused `fma` would move about a quarter of the values by one ulp.
        report.add(Judge.identity("residual, two roundings", differing: zip(cpu, got).filter { $0.bitPattern != $1.bitPattern }.count,
                                  of: rows * d))
        return report
    }

    static let gelu = Stage(
        model: "kernels", name: "gelu",
        proves: "the GPU's GELU (erf) is as close to double precision as the CPU's, over one bit pattern in 64 of [−12, 12]",
        oracle: nil, weights: [], rule: "GPU worst deviation ≤ the CPU's, in ulp of the operand"
    ) { _ in
        let report = Report(title: "kernels gelu")
        try GELUJudge.run(into: report)
        return report
    }

    // MARK: - Attention, sliced against whole

    static let sdpa = Stage(
        model: "kernels", name: "sdpa",
        proves: "the attention cut by heads (GPU + CPU), by query blocks, past maxMatrix and the matrix kernel return the whole graph's result; a head width padded to 128 (the vision tower's 72) returns a float64 witness's",
        options: ["--sequence"],
        oracle: nil, weights: [],
        rule: "against the whole graph: query blocks ≤ 1e-6 per channel; head cut ≤ 1e-5 with the GPU heads bit for bit; matrix kernel ≤ 1e-5; padded width ≤ 1e-5 against float64"
    ) { invocation in
        let sequence = try invocation.integer("--sequence", 4128)
        let report = Report(title: "kernels sdpa S=\(sequence)")
        let heads = 30, headDim = 128
        let width = heads * headDim, count = sequence * width
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw GEMM.Failure.noDevice
        }
        let arena = try Arena(capacity: 5 * count * 4 + (64 << 20))
        func slot(_ name: String) throws -> (UnsafeMutablePointer<Float>, MTLBuffer) {
            let p = try arena.reserve(name, bytes: count * 4).assumingMemoryBound(to: Float.self)
            let rounded = (count * 4 + Arena.alignment - 1) / Arena.alignment * Arena.alignment
            return (p, device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(p), length: rounded,
                                         options: .storageModeShared, deallocator: nil)!)
        }
        let (q, qb) = try slot("q"), (k, kb) = try slot("k"), (v, vb) = try slot("v")
        let (reference, rb) = try slot("reference"), (got, ob) = try slot("got")
        // The amplitudes of the golden tensors (normalized q, k peaking at 8 and 11, v at 189): on
        // values too small, the softmax never saturates and the case measured does not occur.
        var state: UInt64 = 0x5D9A
        func noise(_ amplitude: Float) -> Float {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return amplitude * Float(Int32(truncatingIfNeeded: state)) / Float(Int32.max)
        }
        for i in 0..<count { q[i] = noise(8.9); k[i] = noise(11.5); v[i] = noise(188.9) }
        let gotArray = { Array(UnsafeBufferPointer(start: got, count: count)) }
        let referenceBuffer = UnsafeBufferPointer(start: reference, count: count)

        _ = Attention(device: device, queue: queue, heads: heads, sequence: sequence, headDim: headDim)
            .run(q: qb, k: kb, v: vb, into: rb)

        // By heads: the first on the GPU, the rest on the AMX and the P cores. The GPU heads must be the
        // whole graph's bits; the CPU heads differ by summation order only.
        for fraction in [0.2, 0.3, 0.4, 0.433, 0.5] {
            memset(got, 0xFF, count * 4)    // NaN: a head nobody wrote cannot pass
            let split = try HeadSplitAttention(device: device, queue: queue, heads: heads, sequence: sequence,
                                               headDim: headDim, cpuFraction: fraction)
            _ = split.run(q: qb, k: kb, v: vb, into: ob, q: q, k: k, v: v, out: got)
            let gpuWidth = split.gpuHeads * headDim
            var identical = 0, notFinite = 0
            for i in 0..<count {
                if !got[i].isFinite { notFinite += 1 }
                if i % width < gpuWidth && got[i].bitPattern == reference[i].bitPattern { identical += 1 }
            }
            let label = "heads \(split.gpuHeads) GPU / \(split.cpuHeads) CPU"
            report.add(Judge.bound(label, notFinite == 0 ? channelError(gotArray(), referenceBuffer, columns: width).worst : .infinity,
                                   1e-5, unit: "per channel"))
            report.add(Judge.identity("  its GPU heads", differing: sequence * gpuWidth - identical, of: sequence * gpuWidth))
        }
        // By query blocks: a null reassociation, each output row depends on its own query row alone.
        for chunk in [2048, 1024, 512, 256] where chunk <= sequence {
            memset(got, 0, count * 4)
            _ = Attention(device: device, queue: queue, heads: heads, sequence: sequence, headDim: headDim, queryChunk: chunk)
                .run(q: qb, k: kb, v: vb, into: ob)
            report.add(Judge.bound("query block \(chunk)", channelError(gotArray(), referenceBuffer, columns: width).worst,
                                   1e-6, unit: "per channel"))
        }
        // Past `maxMatrix`: the bound lowered to a third of this matrix slices a graph that fits whole.
        do {
            memset(got, 0, count * 4)
            let limit = max(1, heads * sequence * sequence * 4 / 3)
            _ = Attention(device: device, queue: queue, heads: heads, sequence: sequence, headDim: headDim, matrixLimit: limit)
                .run(q: qb, k: kb, v: vb, into: ob)
            report.add(Judge.bound("\(Attention.groupedSlices(heads: heads, rows: sequence, keys: sequence, limit: limit)) query slices",
                                   channelError(gotArray(), referenceBuffer, columns: width).worst, 1e-6, unit: "per channel"))
        }
        // The matrix kernel, wired behind the plan (`SILICONED_PLAN=flash`): a path a render can take must be crossed here.
        do {
            memset(got, 0, count * 4)
            _ = try FlashMatrix(device: device, queue: queue, heads: heads, sequence: sequence, headDim: headDim)
                .run(q: qb, k: kb, v: vb, into: ob)
            let values = gotArray()
            report.add(Judge.bound("matrix kernel", values.allSatisfy(\.isFinite)
                                   ? channelError(values, referenceBuffer, columns: width).worst : .infinity, 1e-5, unit: "per channel"))
        }
        // The head cut past `maxMatrix`: the GPU heads' queries in slices (a third of their matrix), the CPU
        // heads untouched — against the same cut in one SDPA (last: it rewrites `reference`). What a render over 1024×1600 on Z-Image runs.
        do {
            memset(got, 0xFF, count * 4)
            let gpuHeads = HeadSplitAttention.partition(heads: heads, fraction: HeadSplitAttention.referenceFraction).gpu.count
            let limit = max(1, gpuHeads * sequence * sequence * 4 / 3)
            let whole = try HeadSplitAttention(device: device, queue: queue, heads: heads, sequence: sequence, headDim: headDim)
            _ = whole.run(q: qb, k: kb, v: vb, into: rb, q: q, k: k, v: v, out: reference)
            let sliced = try HeadSplitAttention(device: device, queue: queue, heads: heads, sequence: sequence, headDim: headDim,
                                                matrixLimit: limit)
            _ = sliced.run(q: qb, k: kb, v: vb, into: ob, q: q, k: k, v: v, out: got)
            let values = gotArray()
            report.add(Judge.bound("head cut, \(Attention.groupedSlices(heads: gpuHeads, rows: sequence, keys: sequence, limit: limit)) query slices",
                                   values.allSatisfy(\.isFinite) ? channelError(values, referenceBuffer, columns: width).worst : .infinity,
                                   1e-6, unit: "per channel"))
        }
        // A head width that is not a multiple of 32 — the vision tower's, 16 heads of 72 — runs padded to 128 in
        // the graph (`Attention.workingHeadDim`; MPS's tile kernel did not compile at 72 on an M4 under macOS
        // 26.6, issue #11), against a float64 witness, at a 512² reference's 1,024 patches.
        do {
            let (h, d, s) = (16, 72, 1024)
            let w = h * d, n = s * w
            guard let qb = device.makeBuffer(length: n * 4, options: .storageModeShared),
                  let kb = device.makeBuffer(length: n * 4, options: .storageModeShared),
                  let vb = device.makeBuffer(length: n * 4, options: .storageModeShared),
                  let ob = device.makeBuffer(length: n * 4, options: .storageModeShared) else { throw GEMM.Failure.noDevice }
            let (q, k, v) = (qb.contents().assumingMemoryBound(to: Float.self), kb.contents().assumingMemoryBound(to: Float.self),
                             vb.contents().assumingMemoryBound(to: Float.self))
            for i in 0..<n { q[i] = noise(8.9); k[i] = noise(11.5); v[i] = noise(188.9) }
            _ = Attention(device: device, queue: queue, heads: h, sequence: s, headDim: d).run(q: qb, k: kb, v: vb, into: ob)
            let values = Array(UnsafeBufferPointer(start: ob.contents().assumingMemoryBound(to: Float.self), count: n))
            var witness = [Double](repeating: 0, count: n)
            let scale = 1 / Double(d).squareRoot()
            witness.withUnsafeMutableBufferPointer { out in
                let out = out.baseAddress!
                DispatchQueue.concurrentPerform(iterations: h) { head in
                    var scores = [Double](repeating: 0, count: s)
                    for i in 0..<s {
                        let row = q + i * w + head * d
                        var top = -Double.infinity
                        for j in 0..<s {
                            let key = k + j * w + head * d
                            var dot = 0.0
                            for c in 0..<d { dot += Double(row[c]) * Double(key[c]) }
                            scores[j] = dot * scale
                            top = max(top, scores[j])
                        }
                        var total = 0.0
                        for j in 0..<s { scores[j] = exp(scores[j] - top); total += scores[j] }
                        let target = out + i * w + head * d
                        for j in 0..<s {
                            let p = scores[j] / total, value = v + j * w + head * d
                            for c in 0..<d { target[c] += p * Double(value[c]) }
                        }
                    }
                }
            }
            let worst = witness.withUnsafeBufferPointer { values.allSatisfy(\.isFinite) ? channelError(values, $0, columns: w).worst : .infinity }
            report.add(Judge.bound("head width \(d) padded to \(Attention.workingHeadDim(d)), \(h) heads", worst, 1e-5, unit: "per channel"))
        }
        withExtendedLifetime(arena) {}
        return report
    }

    // MARK: - Attention in three stages, by tiles of query rows

    static let sdpaTiles = Stage(
        model: "kernels", name: "sdpa-tiles",
        proves: "the attention by tiles (Metal 4 matmul2d) returns the float64 attention — in three stages in fp32, and with fp16 or bf16 operands and fp32 sums in three stages and fused at every candidate tile — on a self-attention, a cross-attention over few keys, grouped k/v heads and a length that tiles nothing",
        options: ["--sequence"],
        oracle: nil, weights: [],
        rule: "against float64, per channel: fp32 ≤ 1e-5 and ≤ the graph's own error + 25 % on the golden tensors' amplitudes (saturated softmax); fp16 operands ≤ 1.25 × the error of rounding the same exact output to bf16 (invariant 3, the ANE's bar); bf16 operands ≤ 1.25 × bf16's own error (the float64 attention of the operands rounded to bf16, its output rounded too); both on the map's amplitudes"
    ) { invocation in
        let sequence = try invocation.integer("--sequence", 4128)
        let report = Report(title: "kernels sdpa-tiles S=\(sequence)")
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw GEMM.Failure.noDevice }
        // A GPU without Metal 4 (the CI's virtual one) has no such kernel to judge: skipped, as a stage without its device.
        guard device.supportsFamily(.metal4) else { throw Skip("no Metal 4 on this GPU: the tiled attention needs it") }
        var state: UInt64 = 0x5D9A
        func noise(_ amplitude: Float) -> Float {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return amplitude * Float(Int32(truncatingIfNeeded: state)) / Float(Int32.max)
        }
        func bf16(_ x: Float) -> Float {
            let bits = x.bitPattern
            return Float(bitPattern: UInt32(UInt16(truncatingIfNeeded: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) >> 16)) << 16)
        }
        // Two regimes of data. The golden tensors' amplitudes (q peaking at 8.9, k at 11.5, v at 189, as `sdpa`): the logits run to ±100, the
        // softmax saturates, and an fp16 operand's rounding (2⁻¹¹) moves a logit by 0.02 — 3× bf16's own rounding of the output, which is
        // why the fp16 kernel is not judged there: whether a model's attention is that conditioned is the DiT check's to say, against
        // diffusers in bf16. The map's amplitudes (q, k at 3, v at 1): what the diagnostic judges every Mac's kernels on.
        let regimes: [(label: String, q: Float, k: Float, v: Float, fp16: Bool)] = [("golden amplitudes", 8.9, 11.5, 188.9, false), ("map's amplitudes", 3, 3, 1, true)]
        // (queries, keys, heads, k/v heads, head dimension, label): the DiTs' self-attention, Anima's cross-attention, Krea 2's grouped heads.
        let cases = [(sequence, sequence, 30, 30, 128, "self"), (1000, 77, 24, 24, 128, "cross"), (777, 777, 12, 3, 128, "grouped k/v heads")]
        for regime in regimes {
        for (s, keys, heads, kvHeads, headDim, label) in cases {
            let width = heads * headDim, kvWidth = kvHeads * headDim
            func buffer(_ n: Int, _ amplitude: Float) -> (UnsafeMutablePointer<Float>, MTLBuffer) {
                let b = device.makeBuffer(length: n * 4, options: .storageModeShared)!
                let p = b.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<n { p[i] = noise(amplitude) }
                return (p, b)
            }
            let (q, qb) = buffer(s * width, regime.q), (k, kb) = buffer(keys * kvWidth, regime.k), (v, vb) = buffer(keys * kvWidth, regime.v)
            let rows = [0, 1, 2, 3, s / 3, s / 2, s - 4, s - 3, s - 2, s - 1].filter { $0 >= 0 && $0 < s }
            // The float64 attention of those rows, every head: [rows, width].
            let scale = 1 / Double(headDim).squareRoot()
            func attention(_ q: UnsafePointer<Float>, _ k: UnsafePointer<Float>, _ v: UnsafePointer<Float>) -> [Double] {
            var exact = [Double](repeating: 0, count: rows.count * width)
            exact.withUnsafeMutableBufferPointer { out in
                nonisolated(unsafe) let out = out.baseAddress!
                DispatchQueue.concurrentPerform(iterations: rows.count * heads) { c in
                    let r = c / heads, head = c % heads, kv = head / (heads / kvHeads)
                    var scores = [Double](repeating: 0, count: keys)
                    let query = q + rows[r] * width + head * headDim
                    var top = -Double.infinity
                    for j in 0..<keys {
                        let key = k + j * kvWidth + kv * headDim
                        var dot = 0.0
                        for x in 0..<headDim { dot += Double(query[x]) * Double(key[x]) }
                        scores[j] = dot * scale
                        top = max(top, scores[j])
                    }
                    var total = 0.0
                    for j in 0..<keys { scores[j] = exp(scores[j] - top); total += scores[j] }
                    let target = out + r * width + head * headDim
                    for j in 0..<keys {
                        let p = scores[j] / total, value = v + j * kvWidth + kv * headDim
                        for x in 0..<headDim { target[x] += p * Double(value[x]) }
                    }
                }
            }
            return exact
            }
            let exact = attention(q, k, v)
            // A bf16 path's own error: the float64 attention of the operands rounded to bf16, its output rounded too.
            let roundedOperands = [(q, s * width), (k, keys * kvWidth), (v, keys * kvWidth)].map { p, n in (0..<n).map { bf16(p[$0]) } }
            let bf16Own = roundedOperands[0].withUnsafeBufferPointer { rq in roundedOperands[1].withUnsafeBufferPointer { rk in
                roundedOperands[2].withUnsafeBufferPointer { rv in attention(rq.baseAddress!, rk.baseAddress!, rv.baseAddress!) } } }
            let bf16Floor = exact.withUnsafeBufferPointer { channelError(bf16Own.map { bf16(Float($0)) }, $0, columns: width).worst }
            func error(_ out: UnsafeMutablePointer<Float>) -> Double {
                var got = [Float](repeating: 0, count: rows.count * width)
                for (r, row) in rows.enumerated() { for c in 0..<width { got[r * width + c] = out[row * width + c] } }
                return got.allSatisfy(\.isFinite) ? exact.withUnsafeBufferPointer { channelError(got, $0, columns: width).worst } : .infinity
            }
            let floor = exact.withUnsafeBufferPointer { channelError(exact.map { bf16(Float($0)) }, $0, columns: width).worst }
            // The graph's own error on the same data: the fp32 kernel goes no further from float64 than the graph, + 25 %.
            let (graphOut, graphBuffer) = buffer(s * width, 0)
            _ = Attention(device: device, queue: queue, heads: heads, sequence: s, headDim: headDim, keySequence: keys,
                          kvHeads: kvHeads == heads ? nil : kvHeads).run(q: qb, k: kb, v: vb, into: graphBuffer)
            let graphError = error(graphOut)
            // Every form the map may choose: the three stages in each precision, and fused at each candidate tile in half and bf16.
            let forms: [(TiledAttention.Precision, AttentionTile?)] = [(.fp32, nil)]
                + [TiledAttention.Precision.fp16, .bf16].flatMap { p in ([nil] + AttentionTile.candidates.map { $0 }).map { (p, $0) } }
            for (precision, tile) in forms where precision == .fp32 || regime.fp16 {
                let operands = ["fp32", "fp16 operands", "bf16 operands"][precision == .fp32 ? 0 : precision == .fp16 ? 1 : 2]
                let name = "\(label) S=\(s)/\(keys) \(heads)/\(kvHeads) heads, \(operands)" + (tile.map { ", fused \($0)" } ?? "") + ", \(regime.label)"
                let (out, ob) = buffer(s * width, 0)
                memset(out, 0xFF, s * width * 4)            // NaN: a row nobody wrote cannot pass
                let kernel = try TiledAttention(device: device, queue: queue, heads: heads, sequence: s, headDim: headDim, keys: keys,
                                                kvHeads: kvHeads, precision: precision, tile: tile)
                _ = kernel.run(q: qb, k: kb, v: vb, into: ob)
                switch precision {
                case .fp32:
                    report.add(Judge.bound(name, error(out), max(1e-5, 1.25 * graphError), unit: "per channel",
                                           note: String(format: "the graph's own %.2e", graphError)))
                case .fp16:
                    report.add(Judge.bound(name, error(out), max(1e-5, 1.25 * floor), unit: "per channel",
                                           note: String(format: "bf16's own rounding %.2e", floor)))
                case .bf16:
                    report.add(Judge.bound(name, error(out), max(1e-5, 1.25 * bf16Floor), unit: "per channel",
                                           note: String(format: "bf16's own error %.2e", bf16Floor)))
                }
            }
        }
        }
        return report
    }

    // MARK: - The GEMM split between GPU and AMX

    static let gemmSplit = Stage(
        model: "kernels", name: "gemm-split",
        proves: "the token-wise split of a GEMM between the GPU and the AMX stitches back cblas_sgemm's product",
        oracle: nil, weights: [],
        rule: "against cblas_sgemm, per row: the GPU's rows bit for bit, the AMX's ≤ 1e-5 (another summation order over k = 3 840)"
    ) { _ in
        let report = Report(title: "kernels gemm-split")
        // The AMX brought up, the cut the plan's and frozen: the seam stays where the plan pins it.
        let plan = CapacityPlan.probing(amx: true)
        let gemm = try GEMM(plan: plan)
        guard gemm.conductor != nil else { throw Skip("no AMX conductor on this machine") }
        let m = 1024, k = 3840, n = 3840
        let arena = try Arena(capacity: (m * k + k * n + m * n) * 4 + (32 << 20))
        let a = try arena.reserve("a", bytes: m * k * 4).assumingMemoryBound(to: Float.self)
        let b = try arena.reserve("b", bytes: k * n * 4).assumingMemoryBound(to: Float.self)
        let c = try arena.reserve("c", bytes: m * n * 4).assumingMemoryBound(to: Float.self)
        // Signed pseudo-random, never periodic: on a ramp, a misplaced slice gives plausible values.
        var state: UInt64 = 0xA11CE
        func noise() -> Float {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return Float(Int32(truncatingIfNeeded: state)) / Float(Int32.max)
        }
        for i in 0..<(m * k) { a[i] = noise() }
        for i in 0..<(k * n) { b[i] = noise() }
        var truth = [Float](repeating: 0, count: m * n)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(k),
                    1, a, Int32(k), b, Int32(n), 0, &truth, Int32(n))
        let ab = try gemm.wrap(UnsafeMutableRawPointer(a), bytes: m * k * 4, name: "a")
        let bb = try gemm.wrap(UnsafeMutableRawPointer(b), bytes: k * n * 4, name: "b")
        let cb = try gemm.wrap(UnsafeMutableRawPointer(c), bytes: m * n * 4, name: "c")

        /// Rows `[from, to)`: how many are bit for bit the truth, and the worst relative error of a row.
        func rows(_ from: Int, _ to: Int) -> (exact: Int, worst: Double) {
            var exact = 0, worst = 0.0
            for row in from..<to {
                var numerator = 0.0, denominator = 0.0, identical = true
                for column in 0..<n {
                    let i = row * n + column
                    if c[i].bitPattern != truth[i].bitPattern { identical = false }
                    let d = Double(c[i]) - Double(truth[i])
                    numerator += d * d
                    denominator += Double(truth[i]) * Double(truth[i])
                }
                if identical { exact += 1 }
                let relative = denominator > 0 ? (numerator / denominator).squareRoot() : 0
                worst = max(worst, relative.isNaN ? .infinity : relative)
            }
            return (exact, worst)
        }
        func judge(_ label: String, cut: Int) {
            let gpu = rows(0, cut), amx = rows(cut, m)
            if cut > 0 { report.add(Judge.identity("\(label): GPU rows 0…\(cut)", differing: cut - gpu.exact, of: cut, unit: "row")) }
            report.add(Judge.bound("\(label): AMX rows \(cut)…\(m)", amx.worst, 1e-5, unit: "per row"))
        }
        // The AMX alone: the GPU's rows filled from the truth, so that nothing but the AMX is judged.
        memset(c, 0, m * n * 4)
        let planned = 640                      // five eighths of the rows to the GPU, on its 64-row tile
        let cut = planned
        // The AMX's side on a conductor of its own: the GPU's side here is nothing.
        let direct = Conductor(queue: gemm.queue)
        truth.withUnsafeBufferPointer { c.update(from: $0.baseAddress!, count: cut * n) }
        _ = direct.linear(a: ab, b: bb, c: cb, gpuRows: cut, amxRows: m - cut, k: k, n: n, weightIsTransposed: true,
                          multiply: { _ in 0 })
        judge("AMX alone", cut: cut)
        // The AMX with no GPU rows at all: the plan may give it the whole product.
        memset(c, 0, m * n * 4)
        _ = direct.linear(a: ab, b: bb, c: cb, gpuRows: 0, amxRows: m, k: k, n: n, weightIsTransposed: true,
                          multiply: { _ in preconditionFailure("the GPU has no rows") })
        judge("AMX only", cut: 0)
        // The GPU alone: a plan that gives it every row, whatever the height.
        memset(c, 0, m * n * 4)
        let small = 511
        _ = gemm.linear(a: ab, b: bb, c: cb, m: small, k: k, n: n, weightIsTransposed: true)
        report.add(Judge.identity("GPU alone, \(small) rows", differing: small - rows(0, small).exact, of: small, unit: "row"))
        // The whole split, three times, as the plan assigns it.
        plan.pin(ProductKey(k: k, n: n, m: m), ProductAssignment(gpu: planned, amx: m - planned))
        for pass in 1...3 {
            memset(c, 0, m * n * 4)
            let used = planned
            _ = gemm.linear(a: ab, b: bb, c: cb, m: m, k: k, n: n, weightIsTransposed: true)
            judge(String(format: "split %d (GPU %.0f %%)", pass, 100 * Double(used) / Double(m)), cut: used)
        }
        // `a`, `b`, `c` are bare pointers into the arena: it must outlive their last read.
        withExtendedLifetime(arena) {}
        return report
    }

    // MARK: - Against the reference's own tables

    static let rope = Stage(
        model: "kernels", name: "rope",
        proves: "every RoPE table of the engine is the reference's, value by value (13 tables, 5 bit for bit)",
        oracle: "tools/oracle/kernels.py rope", weights: [],
        rule: "max |Δ| per table within its bound, in units of 2⁻²⁴: 0 where the reference is bit-reproducible, 1 to 64 where torch uses SLEEF's cos, sin, pow"
    ) { invocation in
        let report = Report(title: "kernels rope")
        try RopeJudge.run(goldens: try invocation.goldens.open("goldens-rope.safetensors"), into: report)
        return report
    }

    static let resample = Stage(
        model: "kernels", name: "resample",
        proves: "the spectral schedule's nearest-exact decimation and bilinear upsampling are torch's, on a square and a rectangle",
        oracle: "tools/oracle/kernels.py resample", weights: [],
        rule: "decimation bit for bit (and the even decimation, its counter-test, must differ); bilinear within one ulp of the input's scale"
    ) { invocation in
        let report = Report(title: "kernels resample")
        for file in ["goldens-resample.safetensors", "goldens-resample-rect.safetensors"] {
            try ResampleJudge.run(goldens: try invocation.goldens.open(file), into: report)
        }
        return report
    }
}

/// SplitMix64, and a normal draw from two uniforms (Box–Muller).
struct SplitMix {
    var state: UInt64
    init(seed: UInt64) { state = seed }

    mutating func uniform() -> Double {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return Double((z ^ (z >> 31)) >> 11) / Double(1 << 53)
    }

    mutating func normal() -> Double { (-2 * log(max(uniform(), 1e-300))).squareRoot() * cos(2 * .pi * uniform()) }
}
