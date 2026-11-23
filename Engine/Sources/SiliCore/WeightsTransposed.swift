import Foundation

/// **The CPU's passes over a weight as it is read**: bf16 widened, f32 rounded to f16, `[out, in]` transposed by
/// tiles on every core.
extension Weights {
    /// **A narrow tensor (F16, F8: `SafeTensors.Tensor.Stored`) into 16-bit values, on every core**: its codes read as
    /// they lie into `staging`, transposed there as codes where `transposed` (`[out, in]` into `[in, out]`, by tiles of
    /// 16 bytes or 8 halves), then each looked up in its table (`SafeTensors.codeTable`) into `into` — to the bit what the
    /// f32 path gives (`SafeTensors.read`, then `rounded`), which widened each code on one core and transposed on one core:
    /// a transformer of 13 billion weights in fp8 « scaled » beside the ANE, read through the ring at a block's first cycles (nothing kept yet), took 38
    /// to 40 s a cycle where its bf16 took 7.9; read this way 7.8, the same outputs to the bit, its first 8 cycles 212 →
    /// 45 s (23/11/2026, the M5 Pro). The staging holds the tensor's f32s (`staging`): its codes and their transpose fit
    /// in it twice over. False: `t` is not narrow, nothing read.
    static func narrow(_ t: SafeTensors.Tensor, of file: SafeTensors, transposed: Bool, into to: UnsafeMutablePointer<UInt16>,
                       staging: UnsafeMutableRawPointer, _ rounded: (Float) -> UInt16) throws -> Bool {
        guard t.stored != .asListed else { return false }
        let n = t.shape.reduce(1, *), w = t.stored.width
        try file.readCodes(t, into: staging)
        var codes = UnsafeRawPointer(staging)
        if transposed {
            precondition(t.shape.count == 2, "\(t.shape): not a matrix to transpose")
            let (out, inn) = (t.shape[0], t.shape[1]), there = staging + n * w
            if w == 1 {
                let (f, d) = (UnsafePointer(staging.assumingMemoryBound(to: Int8.self)), there.assumingMemoryBound(to: Int8.self))
                if out % 16 == 0 && inn % 16 == 0 { transposedTiles(f, out, inn, into: d) } else { transposedRows(f, out, inn, into: d) }
            } else {
                let (f, d) = (UnsafePointer(staging.assumingMemoryBound(to: UInt16.self)), there.assumingMemoryBound(to: UInt16.self))
                if out % 8 == 0 && inn % 8 == 0 { transposedHalves(f, out, inn, into: d, rounding: false) } else { transposedRows(f, out, inn, into: d) }
            }
            codes = UnsafeRawPointer(there)
        }
        let table = SafeTensors.codeTable(t.stored, rounded), chunk = 1 << 20
        table.withUnsafeBufferPointer { values in
            nonisolated(unsafe) let (v, c, o) = (values.baseAddress!, codes, to)
            DispatchQueue.concurrentPerform(iterations: (n + chunk - 1) / chunk) { k in
                let range = k * chunk..<min((k + 1) * chunk, n)
                if w == 1 {
                    let b = c.assumingMemoryBound(to: UInt8.self)
                    for i in range { o[i] = v[Int(b[i])] }
                } else {
                    let b = c.assumingMemoryBound(to: UInt16.self)
                    for i in range { o[i] = v[Int(b[i])] }
                }
            }
        }
        return true
    }

    /// `count` bf16 widened to f32 (exact), each through `convert` (to f16: rounded to nearest once).
    static func bf16<B>(_ from: UnsafeRawPointer, _ count: Int, into to: UnsafeMutablePointer<B>, _ convert: (Float) -> B) {
        let b = from.assumingMemoryBound(to: UInt16.self)
        for i in 0..<count { to[i] = convert(Float(bitPattern: UInt32(b[i]) << 16)) }
    }

    /// `[out, in]` 16-bit values into `[in, out]`, **the rows of the result shared among the cores** by blocks of 64, each
    /// block by tiles of 8 × 8 in NEON registers (`transposedTiles`' way, at 16 bits): the bf16 path's weights, 14 GB of a
    /// transformer, which one core's 64 × 64 loop takes tens of seconds over. `out` and `inn` multiples of 8.
    /// `rounding`: each bf16 rounded to nearest in f16 as its tile is stored (`halves`), in the same pass.
    public static func transposedHalves(_ from: UnsafePointer<UInt16>, _ out: Int, _ inn: Int, into to: UnsafeMutablePointer<UInt16>,
                                        rounding: Bool) {
        precondition(out % 8 == 0 && inn % 8 == 0, "[\(out), \(inn)]: tiles of 8")
        nonisolated(unsafe) let (f, t) = (from, to)
        DispatchQueue.concurrentPerform(iterations: (inn + 63) / 64) { block in
            let i0 = block * 64, i1 = min(i0 + 64, inn)
            withUnsafeTemporaryAllocation(of: SIMD8<UInt16>.self, capacity: 16) { buffer in
                let base = buffer.baseAddress!
                for o0 in stride(from: 0, to: out, by: 8) {
                    for i in stride(from: i0, to: i1, by: 8) {
                        var a = base, b = base + 8
                        for m in 0..<8 { a[m] = UnsafeRawPointer(f + (o0 + m) * inn + i).loadUnaligned(as: SIMD8<UInt16>.self) }
                        // Three rounds of even/odd unzips swap row and column in an 8 × 8 tile (an index's six bits
                        // rotate by one each).
                        for _ in 0..<3 {
                            for m in 0..<4 {
                                let w = SIMD16(lowHalf: a[2 * m], highHalf: a[2 * m + 1])
                                b[m] = w.evenHalf; b[m + 4] = w.oddHalf
                            }
                            swap(&a, &b)
                        }
                        for m in 0..<8 { UnsafeMutableRawPointer(t + (i + m) * out + o0).storeBytes(of: rounding ? halves(a[m]) : a[m], as: SIMD8<UInt16>.self) }
                    }
                }
            }
        }
    }

    /// Eight bf16 rounded to nearest in f16 (widened to f32, exact, then narrowed: one `fcvtn`), as bits.
    @inline(__always) static func halves(_ b: SIMD8<UInt16>) -> SIMD8<UInt16> {
        let wide = unsafeBitCast(SIMD8<UInt32>(truncatingIfNeeded: b) &<< 16, to: SIMD8<Float>.self)
        return unsafeBitCast(SIMD8<Float16>(wide), to: SIMD8<UInt16>.self)
    }

    /// `count` f32 rounded to nearest in f16.
    static func half(_ from: UnsafeRawPointer, _ count: Int, into to: UnsafeMutableRawPointer) {
        let f = from.assumingMemoryBound(to: Float.self), h = to.bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { h[i] = Float16(f[i]) }
    }

    /// The same by tiles of 16 × 16 in NEON registers: 16 rows loaded, four unzips of the 256 bytes (each takes the even
    /// elements then the odd: an index's bits rotate by one, four rotations swap row and column), 16 rows stored; the
    /// rows of the result shared among the cores by blocks of 64, as `transposedRows`. `out` and `inn` multiples of 16.
    public static func transposedTiles(_ from: UnsafePointer<Int8>, _ out: Int, _ inn: Int, into to: UnsafeMutablePointer<Int8>, rowLength: Int? = nil) {
        precondition(out % 16 == 0 && inn % 16 == 0, "[\(out), \(inn)]: tiles of 16")
        nonisolated(unsafe) let (f, t) = (UnsafeRawPointer(from), UnsafeMutableRawPointer(to))
        let ld = rowLength ?? out
        DispatchQueue.concurrentPerform(iterations: (inn + 63) / 64) { block in
            let i0 = block * 64, i1 = min(i0 + 64, inn)
            withUnsafeTemporaryAllocation(of: SIMD16<UInt8>.self, capacity: 32) { buffer in
                let base = buffer.baseAddress!
                for o0 in stride(from: 0, to: out, by: 16) {
                    for i in stride(from: i0, to: i1, by: 16) {
                        var a = base, b = base + 16
                        for m in 0..<16 { a[m] = (f + (o0 + m) * inn + i).loadUnaligned(as: SIMD16<UInt8>.self) }
                        for _ in 0..<4 {
                            for m in 0..<8 {
                                let w = SIMD32(lowHalf: a[2 * m], highHalf: a[2 * m + 1])
                                b[m] = w.evenHalf; b[m + 8] = w.oddHalf
                            }
                            swap(&a, &b)
                        }
                        for m in 0..<16 { (t + (i + m) * ld + o0).storeBytes(of: a[m], as: SIMD16<UInt8>.self) }
                    }
                }
            }
        }
    }

    /// `[out, in]` into `[in, out]` as they are (int8 codes), **the rows of the result shared among the
    /// cores** by blocks of 64, each block by tiles of 64 × 64. By rows and not by columns: each core writes pages of its
    /// own, which matters on an arena's fresh pages (each first write a fault, and faults on one page wait on each other).
    /// `rowLength`: the result's row length, when it lies beside others in a wider one (the products a function fuses).
    static func transposedRows<T>(_ from: UnsafePointer<T>, _ out: Int, _ inn: Int, into to: UnsafeMutablePointer<T>, rowLength: Int? = nil) {
        nonisolated(unsafe) let (f, t) = (from, to)
        let ld = rowLength ?? out
        DispatchQueue.concurrentPerform(iterations: (inn + 63) / 64) { block in
            let i0 = block * 64, i1 = min(i0 + 64, inn)
            for o0 in stride(from: 0, to: out, by: 64) {
                for o in o0..<min(o0 + 64, out) {
                    let row = f + o * inn
                    for i in i0..<i1 { t[i * ld + o] = row[i] }
                }
            }
        }
    }

    /// `[out, in]` into `[in, out]`, by tiles of 64 × 64 (what the caches hold), each element through `convert`.
    static func transposed<A, B>(_ from: UnsafePointer<A>, _ out: Int, _ inn: Int, into to: UnsafeMutablePointer<B>, _ convert: (A) -> B) {
        for o0 in stride(from: 0, to: out, by: 64) {
            for i0 in stride(from: 0, to: inn, by: 64) {
                for o in o0..<min(o0 + 64, out) { for i in i0..<min(i0 + 64, inn) { to[i * out + o] = convert(from[o * inn + i]) } }
            }
        }
    }

    /// **Codes by pieces `[count, out, piece]` into `[count · piece, out]`**: each piece transposed into its rows, on
    /// every core (`transposedTiles` where out and piece are multiples of 16). Measured 22/11/2026, int8 `[n, k]` into
    /// `[k, n]`: 58 to 119 GB/s read and written on the M5 Pro, 25 to 46 on the M1 Pro.
    public static func transposedPieces(_ from: UnsafePointer<Int8>, count: Int, out: Int, piece: Int, into to: UnsafeMutablePointer<Int8>) {
        for i in 0..<count {
            let f = from + i * out * piece, t = to + i * piece * out
            if out % 16 == 0 && piece % 16 == 0 { transposedTiles(f, out, piece, into: t) } else { transposedRows(f, out, piece, into: t) }
        }
    }
}
