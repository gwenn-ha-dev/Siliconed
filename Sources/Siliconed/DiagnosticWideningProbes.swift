// The weight widening's measurements and the drift witness. Split from
// `DiagnosticProbes.swift`.

import Accelerate
internal import AMXLegacy
import Foundation
import Metal
import MetalPerformanceShadersGraph

extension Diagnostic {
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
