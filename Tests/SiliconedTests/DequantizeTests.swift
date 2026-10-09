import Accelerate
import XCTest
@testable import Siliconed

/// **The 8-bit widening spread over the cores is the scalar formula at the bit**:
/// `Widen.dequantize` against `Float(q) · s` written here one value at a time, on shapes that make
/// several jobs — whole and partial, rows not a multiple of 16 (the scalar tail), a table read from
/// the middle of a Q8_0 block — for every layout and scale type the maps carry.
final class DequantizeTests: XCTestCase {
    private var seed: UInt64 = 0x4D31_3432

    private func next() -> UInt64 {
        seed &+= 0x9E37_79B9_7F4A_7C15
        var z = seed
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// `count` codes, every one of the 256 included; fp8's NaN code too (it must stay NaN).
    private func codes(_ count: Int) -> [UInt8] { (0..<count).map { $0 < 256 ? UInt8($0) : UInt8(truncatingIfNeeded: next()) } }

    /// `count` scales of `type` as bytes, and their fp32 values. Full fp32 mantissas, so that a
    /// product rounds; one negative, one zero.
    private func scales(_ count: Int, _ type: Artifact.DType) -> (bytes: [UInt8], values: [Float]) {
        var bytes: [UInt8] = [], values: [Float] = []
        for i in 0..<count {
            var x = Float(next() % 1_000_000 + 1) * 1.37e-9
            if i == 1 { x = -x } else if i == 2 { x = 0 }
            switch type {
            case .float16:
                let h = Float16(x)
                bytes += withUnsafeBytes(of: h.bitPattern.littleEndian, Array.init); values.append(Float(h))
            case .bfloat16:
                let b = UInt16(x.bitPattern >> 16)
                bytes += withUnsafeBytes(of: b.littleEndian, Array.init); values.append(Float(bitPattern: UInt32(b) << 16))
            default:
                bytes += withUnsafeBytes(of: x.bitPattern.littleEndian, Array.init); values.append(x)
            }
        }
        return (bytes, values)
    }

    private func value(_ code: UInt8, _ kind: Artifact.DType) -> Float {
        kind == .int8 ? Float(Int8(bitPattern: code)) : Numerics.e4m3[Int(code)]
    }

    /// The engine's output for `count` values from row `firstRow`, against `expected(i)`.
    private func check(_ kind: Artifact.DType, _ q: [UInt8], count: Int, scale: [UInt8]?, _ scaleType: Artifact.DType,
                       _ layout: Widen.ScaleLayout, firstRow: Int = 0, offset: Int = 0,
                       expected: (Int) -> Float, _ label: String, file: StaticString = #filePath, line: UInt = #line) {
        var out = [Float](repeating: -7, count: count + 1)          // one guard value past the end
        q.withUnsafeBytes { v in
            out.withUnsafeMutableBufferPointer { o in
                if let scale {
                    scale.withUnsafeBytes { s in
                        Widen.dequantize(v.baseAddress! + offset, kind: kind, count: count, scale: s.baseAddress!,
                                         scaleType: scaleType, layout: layout, firstRow: firstRow, into: o.baseAddress!)
                    }
                } else {
                    Widen.dequantize(v.baseAddress! + offset, kind: kind, count: count, scale: nil,
                                     scaleType: scaleType, layout: layout, firstRow: firstRow, into: o.baseAddress!)
                }
            }
        }
        var differ = 0, first = -1
        for i in 0..<count {
            let e = expected(i)
            let same = e.isNaN ? out[i].isNaN : out[i].bitPattern == e.bitPattern
            if !same { differ += 1; if first < 0 { first = i } }
        }
        XCTAssertEqual(differ, 0, "\(label): \(differ) of \(count) differ, first at \(first)", file: file, line: line)
        XCTAssertEqual(out[count], -7, "\(label): written past the end", file: file, line: line)
    }

    /// Q8_0's `[K/32, N]` scales: 100 rows of 2 052 columns make jobs of 32 rows (three whole, one
    /// of 4), each row ending in a scalar tail (2 052 = 128·16 + 4); then 50 rows read from row 45,
    /// mid-block (jobs cut on the tensor's blocks, not the read's), and fp8 on the same layout.
    func testBlockScalesOverSeveralJobs() {
        let (k, n, b) = (100, 2052, 32)
        let q = codes(k * n)
        for type in [Artifact.DType.float16, .bfloat16, .float32] {
            let (bytes, s) = scales((k + b - 1) / b * n, type)
            for kind in [Artifact.DType.int8, .float8_e4m3] {
                check(kind, q, count: k * n, scale: bytes, type, .blocks(columns: n, rows: b),
                      expected: { i in self.value(q[i], kind) * s[(i / n / b) * n + i % n] }, "\(kind) blocks \(type)")
                let first = 45, rows = 50
                check(kind, q, count: rows * n, scale: bytes, type, .blocks(columns: n, rows: b), firstRow: first, offset: first * n,
                      expected: { i in self.value(q[first * n + i], kind) * s[((first + i / n) / b) * n + i % n] },
                      "\(kind) blocks \(type) rows \(first)..<\(first + rows)")
            }
        }
    }

    /// One scale per column (SDNQ, int8 rowwise once transposed): 300 rows of 500 make jobs of
    /// 262 rows (one whole, one of 38); and a single row (`count == n`).
    func testColumnScalesOverSeveralJobs() {
        let (k, n) = (300, 500)
        let q = codes(k * n)
        for type in [Artifact.DType.bfloat16, .float32] {
            let (bytes, s) = scales(n, type)
            for kind in [Artifact.DType.int8, .float8_e4m3] {
                check(kind, q, count: k * n, scale: bytes, type, .columns(n),
                      expected: { i in self.value(q[i], kind) * s[i % n] }, "\(kind) columns \(type)")
                check(kind, q, count: n, scale: bytes, type, .columns(n),
                      expected: { i in self.value(q[i], kind) * s[i] }, "\(kind) columns \(type), one row")
            }
        }
    }

    /// One scale for the tensor, and none (a plain fp8 cast): 140 001 values make jobs of 131 072
    /// (one whole, one of 8 929, not a multiple of 16).
    func testTensorScaleAndPlainCastOverSeveralJobs() {
        let count = 140_001
        let q = codes(count)
        let (bytes, s) = scales(4, .float32)
        for kind in [Artifact.DType.int8, .float8_e4m3] {
            check(kind, q, count: count, scale: Array(bytes[0..<4]), .float32, .tensor,
                  expected: { i in self.value(q[i], kind) * s[0] }, "\(kind) tensor")
        }
        check(.float8_e4m3, q, count: count, scale: nil, .float32, .tensor,
              expected: { i in self.value(q[i], .float8_e4m3) }, "fp8 cast")
    }

    /// One scale per row of the published `[N, K]` (the forge's layout), from row 7: 250 rows of 600
    /// make jobs of 218 rows (one whole, one of 32).
    func testRowScalesOverSeveralJobs() {
        let (rows, k) = (250, 600)
        let q = codes(rows * k)
        let (bytes, s) = scales(rows + 7, .bfloat16)
        check(.int8, q, count: rows * k, scale: bytes, .bfloat16, .rows(columns: k), firstRow: 7,
              expected: { i in self.value(q[i], .int8) * s[7 + i / k] }, "int8 rows")
    }
    /// **The bf16 shift over the cores**: every one of the 65 536 codes (NaNs, infinities,
    /// subnormals, signed zeros) against `UInt32(b) << 16`, on counts that make several jobs of
    /// 131 072 — whole, and a partial last one ending in a scalar tail —, just under the threshold
    /// (one thread), and a source starting on an odd byte (the map's tensors are not all aligned).
    func testBfloat16ShiftOverSeveralJobs() {
        for (count, offset) in [(262_144 + 3 * 16 + 7, 0), (3 * 131_072, 0), (262_143, 0), (400_009, 1), (5, 1)] {
            var bytes = [UInt8](repeating: 0, count: 2 * count + offset)
            var codes = [UInt16](repeating: 0, count: count)
            for i in 0..<count {
                let b = i < 65_536 ? UInt16(i) : UInt16(truncatingIfNeeded: next())
                codes[i] = b
                bytes[offset + 2 * i] = UInt8(b & 0xFF); bytes[offset + 2 * i + 1] = UInt8(b >> 8)
            }
            var out = [UInt32](repeating: 0xDEAD_BEEF, count: count + 1)   // one guard value past the end
            bytes.withUnsafeBytes { s in
                out.withUnsafeMutableBytes { d in
                    Widen.bfloat16ToFloat32(source: s.baseAddress! + offset, destination: d.baseAddress!, count: count)
                }
            }
            var differ = 0, first = -1
            for i in 0..<count where out[i] != UInt32(codes[i]) << 16 { differ += 1; if first < 0 { first = i } }
            XCTAssertEqual(differ, 0, "count \(count), offset \(offset): \(differ) differ, first at \(first)")
            XCTAssertEqual(out[count], 0xDEAD_BEEF, "count \(count): written past the end")
        }
    }
    /// The unnormalized regular Hadamard of size `g` (entries ±1), by Kronecker powers of `H4` —
    /// as comfy_kitchen's `_build_hadamard` builds it, independently of the engine's butterflies.
    private func signs(_ g: Int) -> [Int32] {
        let h4: [Int32] = [1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1]
        var h = h4, size = 4
        while size < g {
            var k = [Int32](repeating: 0, count: 16 * size * size)
            for i in 0..<size { for j in 0..<size { for a in 0..<4 { for b in 0..<4 {
                k[(i * 4 + a) * size * 4 + j * 4 + b] = h[i * size + j] * h4[a * 4 + b]
            } } } }
            h = k; size *= 4
        }
        return h
    }

    /// **convrot over the cores is `Float(H·q) · (s/√G)` at the bit**: the integer product
    /// computed here as an fp64 matrix product (exact: integers), then one multiplication — on maps
    /// whose tiles of 4096 columns are whole, partial (a multiple of 8 but not of the tile) and
    /// scalar (not a multiple of 8), several groups along K, every group the forge accepts, the extremes of `m` (columns of
    /// ±127/−128 following `H`'s first row), bf16 and fp32 scales, and one scale for the tensor.
    func testRotatedOverSeveralTiles() {
        for (g, k, n) in [(256, 768, 328), (256, 256, 12), (64, 192, 136), (16, 64, 40), (4, 8, 20), (16, 32, 8200), (4, 8, 4100)] {
            let h = signs(g)
            var q = codes(k * n).map { Int8(bitPattern: $0) }
            for r in 0..<g { q[r * n] = h[r] > 0 ? 127 : -128; q[r * n + 1] = h[r] > 0 ? -128 : 127 }
            // m = H·q per group, in fp64: integers under 2⁵³, so the matrix product is exact.
            var m = [Double](repeating: 0, count: k * n), block = [Double](repeating: 0, count: g * n)
            let hd = h.map(Double.init)
            for g0 in stride(from: 0, to: k, by: g) {
                for i in 0..<(g * n) { block[i] = Double(q[g0 * n + i]) }
                m.withUnsafeMutableBufferPointer { vDSP_mmulD(hd, 1, block, 1, $0.baseAddress! + g0 * n, 1, vDSP_Length(g), vDSP_Length(n), vDSP_Length(g)) }
            }
            for (type, layout) in [(Artifact.DType.bfloat16, Widen.ScaleLayout.columns(n)), (.float32, .columns(n)), (.float32, .tensor)] {
                let (bytes, s) = scales(layout == .tensor ? 1 : n, type)
                var out = [Float](repeating: -7, count: k * n + 1)          // one guard value past the end
                q.withUnsafeBytes { v in
                    bytes.withUnsafeBytes { sc in
                        out.withUnsafeMutableBufferPointer { o in
                            Widen.dequantizeRotated(v.baseAddress!, rows: k, columns: n, group: g, scale: sc.baseAddress!,
                                                    scaleType: type, layout: layout, into: o.baseAddress!)
                        }
                    }
                }
                let norm = 1 / Float(g).squareRoot()
                var differ = 0, first = -1
                for i in 0..<(k * n) where out[i].bitPattern != (Float(m[i]) * (s[layout == .tensor ? 0 : i % n] * norm)).bitPattern {
                    differ += 1; if first < 0 { first = i }
                }
                XCTAssertEqual(differ, 0, "G \(g), [\(k), \(n)], \(type) \(layout): \(differ) differ, first at \(first)")
                XCTAssertEqual(out[k * n], -7, "G \(g), [\(k), \(n)]: written past the end")
            }
        }
    }
}
