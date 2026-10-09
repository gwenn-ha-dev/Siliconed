import Accelerate
import Foundation

/// bf16 → fp32, which is a shift and not a conversion.
///
/// A bf16 *is* the sixteen high-order bits of an fp32. Widening it therefore consists in writing those
/// sixteen bits at the head of a thirty-two-bit word and setting the rest to zero — no rounding, no
/// table, no special case: infinities, NaNs and subnormals pass through intact.
/// This is what makes the reserve cheap, and it is a direct consequence of the fact that
/// the published weights are bf16 (the `float32` files hold bf16 values: 7 mantissa bits used of 23).
package enum Widen {
    /// Writes `count` fp32 values into `destination` from `count` bf16 in `source`.
    ///
    /// **Over the cores** once there are two jobs' worth, as `dequantize`: on one
    /// thread the shift ran at 10.3 G values/s on the DiT's shapes — one core cannot fill the
    /// M1 Pro's bandwidth, and a Standard map widens ~0.5 s per Z-Image evaluation that way, serial
    /// in front of each GEMM. Jobs of 128 K contiguous values (`spread`): no two cores share an output
    /// line except at a job's two ends. A shift moves bits and rounds nothing, so any split gives the
    /// same bits; small calls (scales, a table's row) stay on the caller's thread.
    ///
    /// - Parameters:
    ///   - source: the memory-mapped map, read-only. Not being 4-aligned is not a problem:
    ///             the read is done through unaligned loads.
    ///   - destination: the reserve, which the caller owns.
    package static func bfloat16ToFloat32(source: UnsafeRawPointer,
                                         destination: UnsafeMutableRawPointer,
                                         count: Int) {
        guard count >= 2 * valuesPerJob else { return shift(source, destination, first: 0, count: count) }
        nonisolated(unsafe) let (src, dst) = (source, destination)
        spread(rows: count, width: 1, unit: 16) { _, first, n in shift(src, dst, first: first, count: n) }
    }

    /// `destination[first ..< first + count]` (fp32) ← `source[…]` (bf16) shifted by 16 bits.
    /// Sixteen at a time (`ushll`/`ushll2` and four `str q` on NEON), a scalar tail.
    @inline(__always)
    private static func shift(_ source: UnsafeRawPointer, _ destination: UnsafeMutableRawPointer, first: Int, count: Int) {
        let input = source + 2 * first, output = destination + 4 * first
        var i = 0
        while i + 16 <= count {
            let packed = input.loadUnaligned(fromByteOffset: 2 * i, as: SIMD16<UInt16>.self)
            output.storeBytes(of: SIMD16<UInt32>(truncatingIfNeeded: packed) &<< 16, toByteOffset: 4 * i, as: SIMD16<UInt32>.self)
            i += 16
        }
        while i < count {
            output.storeBytes(of: UInt32(input.loadUnaligned(fromByteOffset: 2 * i, as: UInt16.self)) << 16,
                              toByteOffset: 4 * i, as: UInt32.self)
            i += 1
        }
    }

    /// The reciprocal, which the forge needs and the engine never. Round to nearest even,
    /// like `torch.Tensor.to(torch.bfloat16)`.
    @inlinable
    package static func float32ToBfloat16(source: UnsafeRawPointer,
                                         destination: UnsafeMutableRawPointer,
                                         count: Int) {
        let input = source.assumingMemoryBound(to: UInt32.self)
        let output = destination.assumingMemoryBound(to: UInt16.self)
        for i in 0..<count {
            let bits = input[i]
            let rounding = UInt32(0x7FFF) &+ ((bits >> 16) & 1)
            output[i] = UInt16(truncatingIfNeeded: (bits &+ rounding) >> 16)
        }
    }
}

// MARK: - 8-bit maps

/// **An 8-bit weight → fp32, the way its publisher defines it**: `w = Float(q) · s`, one IEEE
/// multiplication in fp32 and nothing else.
///
/// A map imported from an 8-bit checkpoint keeps the published bytes and their scales (never a
/// widened copy: a map is never wider than what it was given). The engine widens them here, where a
/// bf16 map is shifted. The product is the reference's bit for bit:
///   · int8 × bf16 or fp16: 8 + 11 significant bits fit in fp32's 24 — the product is **exact**;
///   · fp8 or int8 × fp32: the same correctly rounded multiplication as torch's
///     `q.to(float32) * scale` (torchao `_dequantize_affine_float8`, SDNQ, ComfyUI).
/// Publishers that then round the product to bf16 (torchao `dequantize()` returns its `dtype`,
/// SDNQ multiplies in the scale's bf16) lose bits we keep: the engine computes in fp32 (never below
/// the published weights' precision, nor below bf16).
///
/// No fused multiply-add anywhere: `vDSP_vmul`/`vDSP_vsmul` round each product once, and the GPU
/// kernel (`WidenGPU`) is built with fast math off, so the two paths produce the same bits —
/// **as long as no product is subnormal**: the M1's GPU flushes an fp32 product under 2⁻¹²⁶ to
/// zero, the CPU keeps it. The forge therefore refuses any nonzero scale under 2⁻¹²⁶ (int8,
/// smallest |q| = 1) or 2⁻¹¹⁷ (fp8 E4M3, smallest |q| = 2⁻⁹) — `QuantizedLayouts.checkSubnormal`.
extension Widen {
    /// Where the scale of value `(r, c)` lives.
    package enum ScaleLayout: Equatable {
        /// A single scale for the tensor (ComfyUI "scaled" fp8).
        case tensor
        /// One per column of a `[K, N]` map tensor — one per output, the published per-row scale
        /// once the forge has transposed the weight (torchao, SDNQ, `int8_rowwise`).
        case columns(Int)
        /// `[K/b, N]`: one per column and per block of `rows` along the input (GGUF Q8_0, b = 32).
        case blocks(columns: Int, rows: Int)
        /// One per row of a `[N, K]` tensor — the **published** layout, which only the forge reads.
        case rows(columns: Int)
    }

    /// Writes `count` fp32 values: the 8-bit `values` (`.int8` or `.float8_e4m3`) times their
    /// scales. `firstRow`: the row of `values[0]` in the whole tensor (to read a few rows of a table).
    ///
    /// **One pass, over the cores**. Until then this was two `vDSP` passes on one thread —
    /// `vDSP_vflt8` over the whole tensor, then `vDSP_vmul` row by row — at 5.2 G values/s on the Q8_0
    /// map's w1: the fp32 tensor (4 bytes a value, 157 MB for w1) went through memory three
    /// times, and one core cannot fill the M1 Pro's bandwidth. Here each job converts its rows and
    /// multiplies them in registers, 16 at a time, and writes each output byte once. Nothing to
    /// transpose (the map already holds `[K, N]`, unlike the packed blocks), so the lessons of
    /// `dequantizePacked` reduce to this: a job writes whole rows, contiguous, so no two cores ever
    /// share a 128-byte line except at a job's two ends; a job of `.blocks` is whole blocks, so it
    /// widens each block's scale row once, beside its 32 rows.
    ///
    /// The bits are the two passes' bits: `Float(q)` is exact (int8, and fp8 through
    /// `float8E4M3ToFloat32`, unchanged), and `Float(q) · s` is one IEEE multiplication, rounded
    /// once, operands in the same order as `vDSP_vmul` — no addition anywhere, so nothing a fused
    /// multiply-add could contract.
    package static func dequantize(_ values: UnsafeRawPointer, kind: Artifact.DType, count: Int,
                                   scale: UnsafeRawPointer?, scaleType: Artifact.DType, layout: ScaleLayout,
                                   firstRow: Int = 0, into destination: UnsafeMutablePointer<Float>) {
        guard kind == .int8 || kind == .float8_e4m3 else { preconditionFailure("dequantize: \(kind) is not an 8-bit type") }
        guard count > 0 else { return }
        nonisolated(unsafe) let (src, dst) = (values, destination)
        guard let scale else {
            // A plain cast (fp8 without a scale): the conversion alone, in chunks over the cores.
            spread(rows: count, width: 1, unit: 16) { _, first, n in convert(src, kind: kind, first: first, count: n, into: dst) }
            return
        }
        nonisolated(unsafe) let scales = scale
        /// `scale + first` widened to fp32, `n` of them, into `d`.
        @Sendable func widen(_ first: Int, _ n: Int, _ d: UnsafeMutablePointer<Float>) {
            switch scaleType {
            case .bfloat16: bfloat16ToFloat32(source: scales + 2 * first, destination: UnsafeMutableRawPointer(d), count: n)
            case .float16: float16ToFloat32(scales + 2 * first, count: n, into: d)
            case .float32: d.update(from: (scales + 4 * first).assumingMemoryBound(to: Float.self), count: n)
            default: preconditionFailure("dequantize: a scale of type \(scaleType)")
            }
        }
        switch layout {
        case .tensor:
            var s: Float = 0
            widen(0, 1, &s)
            let factor = s
            spread(rows: count, width: 1, unit: 16) { _, first, n in scaled(src, kind: kind, first: first, count: n, by: factor, into: dst) }
        case .columns(let n):
            withUnsafeTemporaryAllocation(of: Float.self, capacity: n) { buffer in
                nonisolated(unsafe) let s = buffer.baseAddress!
                widen(0, n, s)
                spread(rows: count / n, width: n, unit: 1) { _, first, rows in
                    for r in first..<(first + rows) { scaled(src, kind: kind, first: r * n, count: n, by: s, into: dst) }
                }
            }
        case .blocks(let n, let b):
            // Jobs of whole blocks in the tensor's own numbering — the first and last partial when a
            // table's rows start or end mid-block (`materializeRows`); each block's scales widened
            // once into the job's own row of floats.
            spread(rows: count / n, width: n, unit: b, offset: firstRow % b, scratch: n) { s, first, rows in
                var r = first
                while r < first + rows {
                    let block = (firstRow + r) / b, end = min(first + rows, (block + 1) * b - firstRow)
                    widen(block * n, n, s)
                    for row in r..<end { scaled(src, kind: kind, first: row * n, count: n, by: s, into: dst) }
                    r = end
                }
            }
        case .rows(let k):
            let rows = count / k
            withUnsafeTemporaryAllocation(of: Float.self, capacity: rows) { buffer in
                nonisolated(unsafe) let s = buffer.baseAddress!
                widen(firstRow, rows, s)
                spread(rows: rows, width: k, unit: 1) { _, first, rows in
                    for r in first..<(first + rows) { scaled(src, kind: kind, first: r * k, count: k, by: s[r], into: dst) }
                }
            }
        }
    }

    /// The values a job of `dequantize` aims at: 128 K (512 KB of fp32), enough to pay for a
    /// dispatch; w1 of the Z-Image DiT (39 M values) makes 120 jobs of one Q8_0 block of 32 rows.
    private static let valuesPerJob = 1 << 17

    /// Runs `body(scratch, first, rows)` over `rows` rows of `width` values, in jobs of whole
    /// `unit`s of rows (unit boundaries fall `offset` rows before each multiple of `unit`), over the
    /// cores — on the caller's thread when one job suffices. `scratch`: `scratch` floats of the job's own.
    private static func spread(rows: Int, width: Int, unit: Int, offset: Int = 0, scratch: Int = 0,
                               _ body: @escaping @Sendable (_ scratch: UnsafeMutablePointer<Float>, _ first: Int, _ rows: Int) -> Void) {
        let step = max(1, valuesPerJob / (width * unit)) * unit
        // Job j: rows [j·step − offset, (j+1)·step − offset) ∩ [0, rows).
        let jobs = (rows + offset + step - 1) / step
        let run: @Sendable (Int) -> Void = { j in
            let first = max(0, j * step - offset), end = min(rows, (j + 1) * step - offset)
            guard end > first else { return }
            withUnsafeTemporaryAllocation(of: Float.self, capacity: max(1, scratch)) { body($0.baseAddress!, first, end - first) }
        }
        if jobs <= 1 { run(0) } else { DispatchQueue.concurrentPerform(iterations: jobs, execute: run) }
    }

    /// `destination[first ..< first + count] = Float(values[…])`, exact.
    private static func convert(_ values: UnsafeRawPointer, kind: Artifact.DType, first: Int, count: Int,
                                into destination: UnsafeMutablePointer<Float>) {
        if kind == .int8 {
            vDSP_vflt8(values.assumingMemoryBound(to: Int8.self) + first, 1, destination + first, 1, vDSP_Length(count))
        } else {
            float8E4M3ToFloat32(values.assumingMemoryBound(to: UInt8.self) + first, count: count, into: destination + first)
        }
    }

    /// `destination[first + c] = Float(values[first + c]) · s[c]`, `c < count`.
    @inline(__always)
    private static func scaled(_ values: UnsafeRawPointer, kind: Artifact.DType, first: Int, count: Int,
                               by s: UnsafePointer<Float>, into destination: UnsafeMutablePointer<Float>) {
        let y = UnsafeMutableRawPointer(destination + first), f = UnsafeRawPointer(s)
        var c = 0
        if kind == .int8 {
            let q = values + first
            while c + 16 <= count {
                let v = SIMD16<Float>(SIMD16<Int32>(truncatingIfNeeded: q.loadUnaligned(fromByteOffset: c, as: SIMD16<Int8>.self)))
                y.storeBytes(of: v * f.loadUnaligned(fromByteOffset: 4 * c, as: SIMD16<Float>.self), toByteOffset: 4 * c, as: SIMD16<Float>.self)
                c += 16
            }
            let p = q.assumingMemoryBound(to: Int8.self), out = destination + first
            while c < count { out[c] = Float(p[c]) * s[c]; c += 1 }
        } else {
            convert(values, kind: kind, first: first, count: count, into: destination)
            while c + 16 <= count {
                let v = UnsafeRawPointer(y).loadUnaligned(fromByteOffset: 4 * c, as: SIMD16<Float>.self)
                y.storeBytes(of: v * f.loadUnaligned(fromByteOffset: 4 * c, as: SIMD16<Float>.self), toByteOffset: 4 * c, as: SIMD16<Float>.self)
                c += 16
            }
            let out = destination + first
            while c < count { out[c] = out[c] * s[c]; c += 1 }
        }
    }

    /// The same with one factor for every value.
    @inline(__always)
    private static func scaled(_ values: UnsafeRawPointer, kind: Artifact.DType, first: Int, count: Int,
                               by s: Float, into destination: UnsafeMutablePointer<Float>) {
        let y = UnsafeMutableRawPointer(destination + first)
        var c = 0
        if kind == .int8 {
            let q = values + first
            while c + 16 <= count {
                let v = SIMD16<Float>(SIMD16<Int32>(truncatingIfNeeded: q.loadUnaligned(fromByteOffset: c, as: SIMD16<Int8>.self)))
                y.storeBytes(of: v * s, toByteOffset: 4 * c, as: SIMD16<Float>.self)
                c += 16
            }
            let p = q.assumingMemoryBound(to: Int8.self), out = destination + first
            while c < count { out[c] = Float(p[c]) * s; c += 1 }
        } else {
            convert(values, kind: kind, first: first, count: count, into: destination)
            while c + 16 <= count {
                let v = UnsafeRawPointer(y).loadUnaligned(fromByteOffset: 4 * c, as: SIMD16<Float>.self)
                y.storeBytes(of: v * s, toByteOffset: 4 * c, as: SIMD16<Float>.self)
                c += 16
            }
            let out = destination + first
            while c < count { out[c] = out[c] * s; c += 1 }
        }
    }

    /// fp16 → fp32, exactly (every fp16 is an fp32): the hardware converts. A weight published in
    /// fp16 stays fp16 in the map; widening it is not a rounding.
    package static func float16ToFloat32(_ source: UnsafeRawPointer, count: Int, into destination: UnsafeMutablePointer<Float>) {
        guard count > 0 else { return }
        var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: source), height: 1,
                                width: vImagePixelCount(count), rowBytes: count * 2)
        var dst = vImage_Buffer(data: destination, height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
        vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
    }

    /// fp8 E4M3 "fn" → fp32, exactly. Its bits moved into an fp16 (bias 15 instead of 7, the
    /// mantissa three bits further left) read the value × 2⁻⁸ — **subnormals included**, which then
    /// land on fp16 subnormals — so the hardware converts (`vImage`), and × 256 is exact. The only
    /// code fp16 cannot carry is e4m3fn's NaN (`0x7F`/`0xFF`, no infinity): patched afterwards.
    /// Equal to `Numerics.e4m3[b]` on all 256 codes (`ForgeTests`).
    package static func float8E4M3ToFloat32(_ source: UnsafePointer<UInt8>, count: Int,
                                           into destination: UnsafeMutablePointer<Float>) {
        let chunk = 1 << 14
        var halves = [UInt16](repeating: 0, count: min(chunk, max(count, 1)))
        var nan: UInt8 = 0
        halves.withUnsafeMutableBufferPointer { h in
            var start = 0
            while start < count {
                let n = min(chunk, count - start), s = source + start, d = h.baseAddress!
                for i in 0..<n {
                    let b = UInt16(s[i])
                    d[i] = (b & 0x80) << 8 | (b & 0x7F) << 7
                    nan |= (s[i] & 0x7F) == 0x7F ? 1 : 0
                }
                var src = vImage_Buffer(data: d, height: 1, width: vImagePixelCount(n), rowBytes: n * 2)
                var dst = vImage_Buffer(data: destination + start, height: 1, width: vImagePixelCount(n), rowBytes: n * 4)
                vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
                start += n
            }
        }
        var k: Float = 256
        vDSP_vsmul(destination, 1, &k, destination, 1, vDSP_Length(count))
        if nan != 0 {
            for i in 0..<count where source[i] & 0x7F == 0x7F { destination[i] = .nan }
        }
    }
}

// MARK: - ComfyUI's int8 convrot

/// **An 8-bit weight published rotated: `W = (q · s) · R`** (ComfyUI `int8_tensorwise` with
/// `convrot`, comfy_kitchen `dequantize_int8_convrot_weight`). `R` is block-diagonal along the
/// input: one block per `G` consecutive inputs (G = 256), each the Kronecker power of
/// `H4 = [[1,1,1,−1],[1,1,−1,1],[1,−1,1,1],[−1,1,1,1]]` (the "regular" Hadamard, not Sylvester's)
/// divided by √G — symmetric and orthogonal, so the weight is `R` applied to each output's `G`
/// inputs (`Rᵀ = R`).
///
/// **Computed exactly, then rounded once.** The rotation is applied to the *integers* first:
/// `m = (±1 matrix) · q` is an integer, |m| ≤ 32 648 for G = 256 (each row of `H` sums to +16, so
/// at most 136 terms +127 and 120 terms −128), which an `Int16` holds; the four stages of 4-point
/// butterflies (one per base-4 digit of the input index) are therefore exact in any order. Then
/// `w = Float(m) · (s / √G)`, one IEEE multiplication: `s / √G` is `s` times a power of two, exact.
/// The result is the **correctly rounded** `(q · s) · R` — what the fp64 witness gives once rounded
/// to fp32 — whereas Comfy's own fp32 path (`q.float() * s`, rounded, then a 256-term matmul) lands
/// ~10× further from it. No fused multiply-add can change anything: there is one rounding.
///
/// The smallest nonzero |m| is 1, so the smallest product is `s / √G`: the forge refuses a scale
/// that would make it subnormal (`QuantizedLayouts.checkSubnormal`).
extension Widen {
    /// The largest group read: up to 256, `m` fits in an `Int16` (see above).
    package static let largestRotationGroup = 256

    /// Is `g` a group ComfyUI's construction accepts and this code reads — a power of 4, 4 to 256?
    package static func isRotationGroup(_ g: Int) -> Bool {
        var p = 4
        while p < g { p *= 4 }
        return p == g && g <= largestRotationGroup
    }

    /// `1 / √G`, a power of two.
    static func rotationNorm(_ g: Int) -> Float { 1 / Float(g).squareRoot() }

    /// The four 4-point butterflies of `H4` (unnormalized) on `a, b, c, d`:
    /// `(a+b+c−d, a+b−c+d, a−b+c+d, −a+b+c+d)`, in eight additions.
    @inline(__always)
    static func h4<V: SIMD>(_ a: inout V, _ b: inout V, _ c: inout V, _ d: inout V) where V.Scalar: FixedWidthInteger {
        let p = a &+ b, m = a &- b, u = c &+ d, v = c &- d
        a = p &+ v; b = p &- v; c = m &+ u; d = u &- m
    }

    /// **A map's rotated tensor `[K, N]` → fp32**: column `c` (output `c`) of each group of `group`
    /// rows (inputs) is rotated, then multiplied by its scale over √G. Values `int8`; scale one per
    /// column (`.columns`) or one for the tensor (`.tensor`), of any scale type. The tiles are spread
    /// over the cores (the GEMM waits for this weight: the widening is serial in front of it,
    /// `Block.swift`).
    ///
    /// **A tile is `group` rows × 4096 columns, not 64**: what cost was not the butterflies
    /// but the shape of the writes. On Qwen-Image-2.1's shapes, the same tiles *without* any
    /// butterfly (load, convert, scale, store) ran at 9.5 G values/s 64 columns wide, 17–19 at 2048
    /// or 3072, and 24–28 at 4096 — the contiguous int8 path's speed (`dequantize`): an output row
    /// of 4096 floats is one whole 16 KB page, and anything narrower writes pieces of pages from
    /// 256 rows at once. Not set conflicts (a padded scratch changes nothing), not the loops' trip
    /// counts (constant vector counts change nothing), not the job order. With the butterflies:
    /// 6.9 → 13–16 G values/s, 1.01 → 0.51 s per Qwen evaluation; their scratch (2 MB of Int16 per
    /// job) lives in L2 instead of L1 and costs the rest. Wider is worse (12288 columns: 16 jobs of
    /// 6 MB). The integers are the same in any tiling, so the bits are.
    package static func dequantizeRotated(_ values: UnsafeRawPointer, rows k: Int, columns n: Int, group: Int,
                                          scale: UnsafeRawPointer, scaleType: Artifact.DType, layout: ScaleLayout,
                                          into destination: UnsafeMutablePointer<Float>) {
        precondition(isRotationGroup(group) && k % group == 0, "rotation: group \(group) on \(k) rows")
        // The factor of each column: s / √G, exact.
        let norm = rotationNorm(group)
        var factors = [Float](repeating: 0, count: n)
        factors.withUnsafeMutableBufferPointer { f in
            let d = f.baseAddress!
            func widened(_ count: Int) {
                switch scaleType {
                case .bfloat16: bfloat16ToFloat32(source: scale, destination: UnsafeMutableRawPointer(d), count: count)
                case .float16: float16ToFloat32(scale, count: count, into: d)
                case .float32: d.update(from: scale.assumingMemoryBound(to: Float.self), count: count)
                default: preconditionFailure("dequantizeRotated: a scale of type \(scaleType)")
                }
            }
            switch layout {
            case .tensor:
                widened(1)
                for c in 1..<max(n, 1) { d[c] = d[0] }
            case .columns(let m):
                precondition(m == n)
                widened(n)
            default:
                preconditionFailure("dequantizeRotated: scales \(layout) — only per column or per tensor")
            }
            for c in 0..<n { d[c] *= norm }
        }
        let tile = 4096, tiles = (n + tile - 1) / tile, groups = k / group
        nonisolated(unsafe) let (src, dst) = (values.assumingMemoryBound(to: Int8.self), destination)
        factors.withUnsafeBufferPointer { f in
            nonisolated(unsafe) let factor = f.baseAddress!
            DispatchQueue.concurrentPerform(iterations: groups * tiles) { job in
                let g = job / tiles, c0 = (job % tiles) * tile, w = min(tile, n - c0)
                rotateTile(src + g * group * n + c0, dst + g * group * n + c0, factor + c0,
                           rows: group, width: w, stride: n)
            }
        }
    }

    /// One tile: `rows` (= G) × `width` (≤ 4096) values at `source` (row stride `stride`), rotated
    /// along the rows, times `factor[c]`, written at `destination`.
    private static func rotateTile(_ source: UnsafePointer<Int8>, _ destination: UnsafeMutablePointer<Float>,
                                   _ factor: UnsafePointer<Float>, rows g: Int, width w: Int, stride: Int) {
        guard w % 8 == 0 else { return rotateTileScalar(source, destination, factor, rows: g, width: w, stride: stride) }
        let vectors = w / 8
        withUnsafeTemporaryAllocation(of: SIMD8<Int16>.self, capacity: g * vectors) { tile in
            let t = tile.baseAddress!
            for r in 0..<g {
                let row = UnsafeRawPointer(source + r * stride)
                for v in 0..<vectors {
                    t[r * vectors + v] = SIMD8<Int16>(truncatingIfNeeded: row.loadUnaligned(fromByteOffset: 8 * v, as: SIMD8<Int8>.self))
                }
            }
            // One stage per base-4 digit of the row index: rows `base + {0,1,2,3}·span`.
            var span = 1
            while span < g {
                var base = 0
                while base < g {
                    for j in base..<(base + span) {
                        let a = t + j * vectors, b = a + span * vectors, c = b + span * vectors, d = c + span * vectors
                        for v in 0..<vectors { h4(&a[v], &b[v], &c[v], &d[v]) }
                    }
                    base += 4 * span
                }
                span *= 4
            }
            for r in 0..<g {
                let out = UnsafeMutableRawPointer(destination + r * stride)
                for v in 0..<vectors {
                    let m = SIMD8<Float>(SIMD8<Int32>(truncatingIfNeeded: t[r * vectors + v]))
                    let f = UnsafeRawPointer(factor).loadUnaligned(fromByteOffset: 32 * v, as: SIMD8<Float>.self)
                    out.storeBytes(of: m * f, toByteOffset: 32 * v, as: SIMD8<Float>.self)
                }
            }
        }
    }

    /// The same, one column at a time — a width that is not a multiple of 8 (none in the published
    /// files; kept so that any shape is right).
    private static func rotateTileScalar(_ source: UnsafePointer<Int8>, _ destination: UnsafeMutablePointer<Float>,
                                         _ factor: UnsafePointer<Float>, rows g: Int, width w: Int, stride: Int) {
        var m = [Int32](repeating: 0, count: g)
        for c in 0..<w {
            for r in 0..<g { m[r] = Int32(source[r * stride + c]) }
            hadamard(&m)
            for r in 0..<g { destination[r * stride + c] = Float(m[r]) * factor[c] }
        }
    }

    /// `m ← (unnormalized regular Hadamard) · m`, in place, exact (integers).
    package static func hadamard(_ m: inout [Int32]) {
        let g = m.count
        var span = 1
        while span < g {
            var base = 0
            while base < g {
                for j in base..<(base + span) {
                    var a = SIMD2<Int32>(m[j], 0), b = SIMD2<Int32>(m[j + span], 0)
                    var c = SIMD2<Int32>(m[j + 2 * span], 0), d = SIMD2<Int32>(m[j + 3 * span], 0)
                    h4(&a, &b, &c, &d)
                    (m[j], m[j + span], m[j + 2 * span], m[j + 3 * span]) = (a[0], b[0], c[0], d[0])
                }
                base += 4 * span
            }
            span *= 4
        }
    }
}

// MARK: - GGUF packed blocks: Q4_0…Q5_1, K-quants

/// **A packed GGUF type → fp32, ggml's arithmetic at the bit** (`ggml-quants.c`,
/// `dequantize_row_q4_0`, `_q4_1`, `_q5_0`, `_q5_1`, `_q4_K`, `_q5_K`, `_q6_K`; gguf-py's `quants.py`
/// and diffusers' `dequantize_blocks_*` are numpy/torch transcriptions of it). A block covers 32
/// (legacy) or 256 (K-quant) consecutive inputs of one output row:
///
///     Q4_0   18 B   d fp16 · qs[16]                                    w = d·(q − 8),          q 4 bits
///     Q4_1   20 B   d fp16 · m fp16 · qs[16]                           w = d·q + m,            q 4 bits
///     Q5_0   22 B   d fp16 · qh[4] · qs[16]                            w = d·(q − 16),         q 5 bits
///     Q5_1   24 B   d fp16 · m fp16 · qh[4] · qs[16]                   w = d·q + m,            q 5 bits
///     Q4_K  144 B   d fp16 · dmin fp16 · scales[12] · qs[128]           w = (d·sc)·q − dmin·m,  q 4 bits
///     Q5_K  176 B   d fp16 · dmin fp16 · scales[12] · qh[32] · qs[128]  w = (d·sc)·q − dmin·m,  q 5 bits
///     Q6_K  210 B   ql[128] · qh[64] · scales[16] int8 · d fp16          w = (d·sc)·(q − 32),     q 6 bits
///
/// What a port by analogy with Q8_0 would miss:
///
///   · **The values are not in byte order.** Q4_0…Q5_1: the low nibbles of `qs[0…15]` are values
///     0–15, the high nibbles values 16–31; Q5's fifth bit of value j is bit j of the little-endian
///     `uint32` `qh`. Q4_K/Q5_K: four groups of 64 values, each read from the same 32 bytes of `qs`
///     — the low nibbles give values 0–31 (sub-block 2g), the high nibbles values 32–63 (sub-block
///     2g+1); Q5_K's fifth bit of sub-block s is bit s of `qh[l]`. Q6_K: two halves of 128; in each,
///     value `32·s + l` takes nibble `s / 2` of `ql[32·(s mod 2) + l]` and bits `2s, 2s+1` of
///     `qh[l]`, and its scale is `scales[8·half + 2s + l/16]` (one per 16).
///   · **Q4_K/Q5_K's eight 6-bit scales and mins are packed in 12 bytes** (`get_scale_min_k4`):
///     sub-blocks 0–3 in the low 6 bits of bytes 0–3 (scale) and 4–7 (min); sub-blocks 4–7 take
///     their low 4 bits from a nibble of bytes 8–11 and their high 2 bits from the top of bytes 0–7.
///   · Q4_1/Q5_1 **add** their min, the K-quants **subtract** theirs; Q6_K's `d` is at the **end**
///     of the block, and its scales are signed.
///
/// **One rounding at most, so any order or a fused multiply-add gives the same bits.** `d`, `dmin`
/// and `m` are fp16 (11 significant bits), `sc` and the K-quants' `m` 6 bits (Q6_K: |sc| ≤ 128, 7 bits),
/// `q` at most 5 bits (Q6_K: |q − 32| ≤ 32; Q4_0, Q5_0: |q − 8| ≤ 8, |q − 16| ≤ 16): every product —
/// `d·sc·q` in any order, `dmin·m`, `d·q` — holds in 23 bits of fp32's 24, **exact**. Q4_0, Q5_0
/// and Q6_K are therefore exact; the others round once, at the addition — and a fused
/// `fma(a, q, ±b)` rounds the same exact operands once, signed zeros included. Every nonzero product
/// and sum is a multiple of 2⁻²⁴ (fp16's quantum), so none is subnormal: the GPU's flush of fp32
/// subnormals cannot differ from the CPU (`WidenGPU`). The forge refuses a non-finite fp16 scale
/// (`QuantizedLayouts.checkPacked`), so no NaN payload has to agree either.
extension Widen {
    /// `[rows, columns / b]` blocks of `kind` → fp32: `[columns, rows]` (`transposing`: the map's
    /// `[K, N]`, what the engine reads) or `[rows, columns]` (the published layout, the forge's).
    ///
    /// **The transposition is the cost, not the decoding** (on the DiT's shapes, 8 cores): K1
    /// decoded Q4_K at 16 G values/s in place but transposed at 3.4 — slower than the Q8_0 map's
    /// single-threaded path (5.2), which is why a 5 GB GGUF map rendered slower than the 7.25 GB Q8_0
    /// one. Measured, one at a time: tiles of 16 rows wrote 64 bytes per output row, **half
    /// of the M1's 128-byte line**, two cores on every line; scalar stores → 4×4 transposes in
    /// registers (×1.6–1.9); jobs that walk the **rows** first, so that consecutive jobs write
    /// further along the same 256 output rows (pages) rather than 256 new ones (+10 %); tiles of 256
    /// rows (1 KB contiguous per output row, the tile `[256][256]` in L2). With the 16-wide decoders
    /// below (19–21 G values/s in place): 6–7.6 G values/s transposed, every type ahead of Q8_0.
    /// Moving bits changes none: the blocks' arithmetic is the scalar formula's.
    package static func dequantizePacked(_ source: UnsafeRawPointer, kind: Artifact.DType, rows n: Int, columns k: Int,
                                         transposing: Bool, into destination: UnsafeMutablePointer<Float>) {
        guard let (qk, bytes) = kind.packedBlock else { preconditionFailure("dequantizePacked: \(kind)") }
        precondition(k % qk == 0, "dequantizePacked: \(k) columns for blocks of \(qk)")
        let per = k / qk
        guard n > 0, per > 0 else { return }
        nonisolated(unsafe) let (src, dst) = (source, destination)
        @Sendable @inline(__always) func decode(_ block: UnsafeRawPointer, _ out: UnsafeMutablePointer<Float>) {
            switch kind {
            case .q4_0: blockQ4_0(block, out)
            case .q4_1: blockQ4_1(block, out)
            case .q5_0: blockQ5_0(block, out)
            case .q5_1: blockQ5_1(block, out)
            case .q4_k: superBlockQ4K(block, out)
            case .q5_k: superBlockQ5K(block, out)
            default: superBlockQ6K(block, out)
            }
        }
        guard transposing else {
            // Straight: each row's blocks one after the other, rows spread over the cores.
            let tile = 64, tiles = (n + tile - 1) / tile
            DispatchQueue.concurrentPerform(iterations: tiles) { t in
                for r in (t * tile)..<min(n, (t + 1) * tile) {
                    for j in 0..<per { decode(src + (r * per + j) * bytes, dst + r * k + j * qk) }
                }
            }
            return
        }
        // Jobs of `tile` rows × `group` blocks (256 inputs), the rows varying fastest.
        let tile = 256, group = max(1, 256 / qk), width = group * qk
        let tiles = (n + tile - 1) / tile, groups = (per + group - 1) / group
        DispatchQueue.concurrentPerform(iterations: tiles * groups) { job in
            let r0 = (job % tiles) * tile, h = min(tile, n - r0)
            let j0 = (job / tiles) * group, blocks = min(group, per - j0), w = blocks * qk, c0 = j0 * qk
            withUnsafeTemporaryAllocation(of: Float.self, capacity: tile * width) { buffer in
                let y = buffer.baseAddress!                        // [h][width]: row i, input c0 + c
                for i in 0..<h {
                    let row = src + ((r0 + i) * per + j0) * bytes
                    for j in 0..<blocks { decode(row + j * bytes, y + i * width + j * qk) }
                }
                let h4 = h & ~3, out = UnsafeMutableRawPointer(dst)
                var c = 0
                while c < w {                                      // w is a multiple of 32
                    var i = 0
                    while i < h4 {
                        let p = UnsafeRawPointer(y + i * width + c)
                        let a = p.loadUnaligned(as: SIMD4<Float>.self)
                        let b = p.loadUnaligned(fromByteOffset: 4 * width, as: SIMD4<Float>.self)
                        let e = p.loadUnaligned(fromByteOffset: 8 * width, as: SIMD4<Float>.self)
                        let f = p.loadUnaligned(fromByteOffset: 12 * width, as: SIMD4<Float>.self)
                        let at = ((c0 + c) * n + r0 + i) * 4
                        out.storeBytes(of: SIMD4(a[0], b[0], e[0], f[0]), toByteOffset: at, as: SIMD4<Float>.self)
                        out.storeBytes(of: SIMD4(a[1], b[1], e[1], f[1]), toByteOffset: at + 4 * n, as: SIMD4<Float>.self)
                        out.storeBytes(of: SIMD4(a[2], b[2], e[2], f[2]), toByteOffset: at + 8 * n, as: SIMD4<Float>.self)
                        out.storeBytes(of: SIMD4(a[3], b[3], e[3], f[3]), toByteOffset: at + 12 * n, as: SIMD4<Float>.self)
                        i += 4
                    }
                    for i in h4..<h {
                        for l in 0..<4 { dst[(c0 + c + l) * n + r0 + i] = y[i * width + c + l] }
                    }
                    c += 4
                }
            }
        }
    }

    // The decoders work on 16 values at a time (`SIMD16`, NEON's four registers); the arithmetic is
    // the scalar formula's, operation for operation — exact products, one addition — so the bits are.

    @inline(__always)
    private static func bytes16(_ p: UnsafeRawPointer, _ offset: Int = 0) -> SIMD16<UInt8> {
        p.loadUnaligned(fromByteOffset: offset, as: SIMD16<UInt8>.self)
    }
    @inline(__always)
    private static func floats(_ q: SIMD16<UInt8>) -> SIMD16<Float> { SIMD16<Float>(SIMD16<Int32>(truncatingIfNeeded: q)) }
    @inline(__always)
    private static func store(_ v: SIMD16<Float>, _ y: UnsafeMutablePointer<Float>, _ at: Int) {
        UnsafeMutableRawPointer(y + at).storeBytes(of: v, as: SIMD16<Float>.self)
    }
    /// The fifth bits of Q5_0/Q5_1: bit `j` (`first` ≤ j < `first` + 16) of `qh`, at bit 4.
    @inline(__always)
    private static func fifthBits(_ qh: UInt32, _ first: UInt32) -> SIMD16<UInt8> {
        let lanes = SIMD16<UInt32>(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) &+ first
        return SIMD16<UInt8>(truncatingIfNeeded: ((SIMD16<UInt32>(repeating: qh) &>> lanes) & 1) &<< 4)
    }

    /// One Q4_0 block (18 bytes) → 32 floats: `d·(q − 8)`.
    @inline(__always)
    static func blockQ4_0(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = half(b, 0), qs = bytes16(b, 2)
        store((floats(qs & 0xF) - 8) * d, y, 0)
        store((floats(qs &>> 4) - 8) * d, y, 16)
    }

    /// One Q4_1 block (20 bytes) → 32 floats: `d·q + m`.
    @inline(__always)
    static func blockQ4_1(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = half(b, 0), m = half(b, 2), qs = bytes16(b, 4)
        store(floats(qs & 0xF) * d + m, y, 0)
        store(floats(qs &>> 4) * d + m, y, 16)
    }

    /// One Q5_0 block (22 bytes) → 32 floats: `d·(q − 16)`.
    @inline(__always)
    static func blockQ5_0(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = half(b, 0), qh = b.loadUnaligned(fromByteOffset: 2, as: UInt32.self).littleEndian, qs = bytes16(b, 6)
        store((floats(qs & 0xF | fifthBits(qh, 0)) - 16) * d, y, 0)
        store((floats(qs &>> 4 | fifthBits(qh, 16)) - 16) * d, y, 16)
    }

    /// One Q5_1 block (24 bytes) → 32 floats: `d·q + m`.
    @inline(__always)
    static func blockQ5_1(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = half(b, 0), m = half(b, 2), qh = b.loadUnaligned(fromByteOffset: 4, as: UInt32.self).littleEndian
        let qs = bytes16(b, 8)
        store(floats(qs & 0xF | fifthBits(qh, 0)) * d + m, y, 0)
        store(floats(qs &>> 4 | fifthBits(qh, 16)) * d + m, y, 16)
    }

    @inline(__always)
    private static func half(_ p: UnsafeRawPointer, _ offset: Int) -> Float {
        Float(Float16(bitPattern: p.loadUnaligned(fromByteOffset: offset, as: UInt16.self)))
    }

    /// ggml's `get_scale_min_k4(j, scales)`: sub-block `j`'s 6-bit scale and min.
    @inline(__always)
    package static func scaleMinK4(_ j: Int, _ s: UnsafePointer<UInt8>) -> (scale: UInt8, min: UInt8) {
        if j < 4 { return (s[j] & 63, s[j + 4] & 63) }
        return ((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4))
    }

    /// One Q4_K super-block (144 bytes) → 256 floats.
    @inline(__always)
    static func superBlockQ4K(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = half(b, 0), dmin = half(b, 2), scales = (b + 4).assumingMemoryBound(to: UInt8.self)
        for g in 0..<4 {
            let (s1, m1) = scaleMinK4(2 * g, scales), (s2, m2) = scaleMinK4(2 * g + 1, scales)
            let d1 = d * Float(s1), dm1 = dmin * Float(m1), d2 = d * Float(s2), dm2 = dmin * Float(m2)
            for half in 0..<2 {
                let q = bytes16(b, 16 + 32 * g + 16 * half), at = 64 * g + 16 * half
                store(d1 * floats(q & 0xF) - dm1, y, at)
                store(d2 * floats(q &>> 4) - dm2, y, at + 32)
            }
        }
    }

    /// One Q5_K super-block (176 bytes) → 256 floats.
    @inline(__always)
    static func superBlockQ5K(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = half(b, 0), dmin = half(b, 2), scales = (b + 4).assumingMemoryBound(to: UInt8.self)
        for g in 0..<4 {
            let (s1, m1) = scaleMinK4(2 * g, scales), (s2, m2) = scaleMinK4(2 * g + 1, scales)
            let d1 = d * Float(s1), dm1 = dmin * Float(m1), d2 = d * Float(s2), dm2 = dmin * Float(m2)
            let low = UInt8(2 * g), high = UInt8(2 * g + 1)
            for half in 0..<2 {
                let q = bytes16(b, 48 + 32 * g + 16 * half), qh = bytes16(b, 16 + 16 * half), at = 64 * g + 16 * half
                store(d1 * floats(q & 0xF | ((qh &>> low) & 1) &<< 4) - dm1, y, at)
                store(d2 * floats(q &>> 4 | ((qh &>> high) & 1) &<< 4) - dm2, y, at + 32)
            }
        }
    }

    /// One Q6_K super-block (210 bytes) → 256 floats.
    @inline(__always)
    static func superBlockQ6K(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = half(b, 208), sc = (b + 192).assumingMemoryBound(to: Int8.self)
        for h in 0..<2 {
            for s in 0..<4 {
                let nibble = UInt8(4 * (s / 2)), bits = UInt8(2 * s)
                for part in 0..<2 {                                 // l < 16, then l ≥ 16: one scale each
                    let ql = bytes16(b, 64 * h + 32 * (s % 2) + 16 * part), qh = bytes16(b, 128 + 32 * h + 16 * part)
                    let q = SIMD16<Int32>(truncatingIfNeeded: (ql &>> nibble) & 0xF | ((qh &>> bits) & 3) &<< 4) &- 32
                    store(d * Float(sc[8 * h + 2 * s + part]) * SIMD16<Float>(q), y, 128 * h + 32 * s + 16 * part)
                }
            }
        }
    }
}
