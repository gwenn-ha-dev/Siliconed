import Metal
import XCTest
@testable import Siliconed

/// **A K-quant's blocks read as published on a rounded activation** (`TensorGEMM.linear(a:kquant:…)`, `KQuantKernels`): the GPU's
/// product against fp64 on the activation rounded to bf16 and the weight's values decoded by ggml's formula (`KQuantTests`'s scalar
/// transcription, independent of the engine) — Q4_K and Q5_K, seeded blocks plus the extremes (every bit set: `sc·q` at 945 and 1953,
/// the largest the fp16 `Iq` must hold exactly; negative and subnormal scales; zero mins), a shape cut in three chunks of rows with a
/// remainder, and columns that are not a whole tile. Exactness is the route's premise: the error is fp32 accumulation's, ~1e-7, where a
/// bf16 widening of the same values would show ~1e-3. Skipped without the Metal 4 tensor kernels.
final class KQuantKernelTests: XCTestCase {
    private static func blocks(n: Int, k: Int, five: Bool, seed: UInt64) -> [UInt8] {
        let size = five ? 176 : 144, count = n * (k / 256)
        var bytes = [UInt8](repeating: 0, count: count * size), state = seed
        func next() -> UInt8 { state = state &* 6364136223846793005 &+ 1442695040888963407; return UInt8(truncatingIfNeeded: state >> 33) }
        for b in 0..<count {
            let at = b * size
            for i in 4..<size { bytes[at + i] = next() }
            switch b % 9 {
            case 0: for i in 4..<size { bytes[at + i] = 0xFF }   // sc = m = 63, q at its top
            case 1: for i in 4..<size { bytes[at + i] = 0 }
            default: break
            }
            let d: Float16 = b % 9 == 2 ? -0.004 : b % 9 == 3 ? Float16(bitPattern: 0x0003) : Float16(Float(1 + b % 7) / 3000)
            let dmin: Float16 = b % 9 == 4 ? 0 : b % 9 == 5 ? -0.01 : Float16(Float(1 + b % 5) / 700)
            bytes[at] = UInt8(d.bitPattern & 0xFF); bytes[at + 1] = UInt8(d.bitPattern >> 8)
            bytes[at + 2] = UInt8(dmin.bitPattern & 0xFF); bytes[at + 3] = UInt8(dmin.bitPattern >> 8)
        }
        return bytes
    }

    private func check(five: Bool) throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        guard let gemm = TensorGEMM(device: device, queue: queue, roundingActivations: true), gemm.readsKQuants else {
            throw XCTSkip("no Metal 4 tensor kernels for K-quants on this device")
        }
        let m = 150, k = 512, n = 96, size = five ? 176 : 144
        // The smallest reserve that lays the weight out: 64 rows a chunk, so 150 rows run as 64 + 64 + 22.
        var capacity = 16384
        while TensorGEMM.kquantLayout(k: k, n: n, capacity: capacity, five: five) == nil { capacity += 16384 }
        let layout = try XCTUnwrap(TensorGEMM.kquantLayout(k: k, n: n, capacity: capacity, five: five))
        XCTAssertEqual(layout.rows, 64)
        let raw = Self.blocks(n: n, k: k, five: five, seed: five ? 0x5A5 : 0x4A4)
        // W [k, n] by ggml's formula, block j of output column c holding k in 256·j ..< 256·(j + 1).
        var w = [Double](repeating: 0, count: k * n)
        for c in 0..<n {
            for j in 0..<(k / 256) {
                let at = (c * (k / 256) + j) * size, block = Array(raw[at..<(at + size)])
                let values = five ? KQuantTests.referenceQ5K(block) : KQuantTests.referenceQ4K(block)
                for (i, v) in values.enumerated() { w[(j * 256 + i) * n + c] = Double(v) }
            }
        }
        var state: UInt64 = 0xA11CE
        let a = (0..<(m * k)).map { _ -> Float in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return (Float(UInt32(truncatingIfNeeded: state >> 32)) / Float(UInt32.max) - 0.5) * 6
        }
        let reserve = try XCTUnwrap(device.makeBuffer(length: capacity, options: .storageModeShared))
        raw.withUnsafeBytes { reserve.contents().copyMemory(from: $0.baseAddress!, byteCount: raw.count) }
        let input = try XCTUnwrap(device.makeBuffer(bytes: a, length: 4 * a.count, options: .storageModeShared))
        let output = try XCTUnwrap(device.makeBuffer(length: 4 * m * n, options: .storageModeShared))
        _ = gemm.linear(a: input, kquant: reserve, layout: layout, c: output, m: m, k: k, n: n)
        let c = output.contents().assumingMemoryBound(to: Float.self)
        // Channel by channel: each output column against its own largest value.
        var worst = 0.0
        for col in 0..<n {
            var err = 0.0, top = 0.0
            for row in 0..<m {
                var exact = 0.0
                for i in 0..<k { exact += Double(Diagnostic.Half.bf16.round(a[row * k + i])) * w[i * n + col] }
                err = max(err, abs(Double(c[row * n + col]) - exact)); top = max(top, abs(exact))
            }
            worst = max(worst, top > 0 ? err / top : err)
        }
        XCTAssertLessThan(worst, 2e-6, "\(five ? "Q5_K" : "Q4_K"): worst column's error \(worst)")
    }

    func testQ4KBlocksAgainstFP64() throws { try check(five: false) }
    func testQ5KBlocksAgainstFP64() throws { try check(five: true) }

    /// The layout lies inside its reserve, page by page, and refuses what it cannot hold.
    func testTheLayoutFitsItsReserveOrIsRefused() throws {
        XCTAssertNil(TensorGEMM.kquantLayout(k: 3000, n: 64, capacity: 1 << 30, five: false), "k not a whole number of super-blocks")
        XCTAssertNil(TensorGEMM.kquantLayout(k: 3072, n: 3072, capacity: 3072 * 3072, five: false), "a reserve below 64 rows")
        for five in [false, true] {
            let k = 3840, n = 10240, capacity = 4 * k * n
            let l = try XCTUnwrap(TensorGEMM.kquantLayout(k: k, n: n, capacity: capacity, five: five))
            let offsets = [l.integers, l.scales, l.mins, l.sums, l.rounded]
            XCTAssertTrue(offsets.allSatisfy { $0 % 16384 == 0 })
            XCTAssertEqual(offsets, offsets.sorted())
            XCTAssertGreaterThanOrEqual(l.integers, n * (k / 256) * (five ? 176 : 144))
            XCTAssertGreaterThanOrEqual(l.scales - l.integers, 2 * k * n)
            XCTAssertGreaterThanOrEqual(l.rounded - l.sums, 4 * (k / 32) * l.rows)
            XCTAssertLessThanOrEqual(l.rounded + 2 * k * l.rows, capacity)
            XCTAssertTrue(l.rows >= 64 && l.rows % 64 == 0)
        }
    }
}
