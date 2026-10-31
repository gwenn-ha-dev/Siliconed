import XCTest
@testable import Siliconed

/// The attention at the lengths of 2048² and 2048×3072 (16,384 and 24,576 image tokens), and the share of heads the
/// CPU takes: pure functions, no Metal.
final class AttentionScalingTests: XCTestCase {
    /// The four families' heads: (name, q heads, k/v heads), all of 128.
    private let families = [("Z-Image", 30, 30), ("Qwen-Image-2.1", 32, 32), ("Anima", 16, 16), ("Krea 2", 48, 12)]

    /// Issue #11: MPS's tile SDPA did not compile at the vision tower's 72 on an M4. A width that is not a multiple of
    /// 32 runs padded to the next multiple of 128; every other width keeps its graph, and the memory need charges the
    /// padded tensors only where the graph pads.
    func testAHeadWidthOffThe32GridIsPaddedTo128AndChargedThere() {
        for width in [32, 64, 96, 128, 256, 512] { XCTAssertEqual(Attention.workingHeadDim(width), width) }
        XCTAssertEqual(Attention.workingHeadDim(72), 128)
        XCTAssertEqual(Attention.workingHeadDim(80), 128)
        XCTAssertEqual(Attention.workingHeadDim(136), 256)
        func bytes(_ headDim: Int) -> [String: Int] {
            Dictionary(uniqueKeysWithValues: MemoryNeed.attention("vision", heads: 16, rows: 1024, keys: 1024, headDim: headDim,
                                                                  causal: false, threads: 8).map { ($0.name, $0.bytes) })
        }
        XCTAssertEqual(bytes(72)["vision SDPA queries"], 1024 * 16 * 128 * 4)
        XCTAssertEqual(bytes(72)["vision head padding"], 4 * 1024 * 16 * 128 * 4)
        XCTAssertEqual(bytes(128)["vision SDPA queries"], 1024 * 16 * 128 * 4)
        XCTAssertNil(bytes(128)["vision head padding"])
    }

    func testTheSoftmaxPassIsExpToFloatPrecisionAndSumsTheRow() {
        // Arguments in [−87.3, 0] as a row after the maximum gives them, and the sizes that leave a tail of 0…7.
        for count in [1, 7, 8, 9, 63, 4096, 4101] {
            var row: [Float] = (0..<count).map { (i: Int) -> Float in
                let x = Double(i)
                let wave: Double = sin(x * 0.37) * 40
                let ripple: Double = 3 * cos(x * 1.9)
                return Float(wave + ripple)
            }
            let maximum = row.max()!, scale = Float(1 / 128.0.squareRoot())
            let expected: [Double] = row.map { (v: Float) -> Double in
                let shifted: Double = Double(v) - Double(maximum)
                return exp(Double(scale) * shifted)
            }
            let sum = row.withUnsafeMutableBufferPointer {
                HeadSplitAttention.exponentiate($0.baseAddress!, count: count, scale: scale, maximum: maximum)
            }
            var worst = 0.0
            for (got, want) in zip(row, expected) where want > 1e-30 { worst = max(worst, abs(Double(got) / want - 1)) }
            XCTAssertLessThan(worst, 5e-7, "count \(count)")
            let total: Double = expected.reduce(0, +)
            XCTAssertEqual(Double(sum), total, accuracy: 3e-6 * total)
        }
        // Far under the maximum: held at exp(−87.3), never a subnormal, a NaN or a negative.
        var far: [Float] = [0, -1000, -1e6, -87.4, -86]
        let sum = far.withUnsafeMutableBufferPointer {
            HeadSplitAttention.exponentiate($0.baseAddress!, count: 5, scale: 1, maximum: 0)
        }
        XCTAssertTrue(far.allSatisfy { $0.isFinite && $0 >= 0 && ($0 == 0 || $0 >= Float.leastNormalMagnitude) })
        XCTAssertEqual(far[0], 1)
        XCTAssertEqual(Double(sum), 1 + exp(-86), accuracy: 1e-6)
    }

    func testEveryQuerySliceStaysUnderTheMatrixBoundAndCoversTheRowsAtLongLengths() {
        for (name, heads, kv) in families {
            let group = heads / kv
            for tokens in [4096, 6144, 16384, 24576] {
                let keys = tokens + (name.hasPrefix("Qwen") ? 130 : 0)
                let fraction = HeadSplitAttention.referenceFraction
                let gpu = HeadSplitAttention.partition(heads: heads, fraction: fraction, group: group).gpu.count
                let slices = Attention.groupedSlices(heads: gpu, rows: tokens, keys: keys)
                let length = (tokens + slices - 1) / slices
                XCTAssertGreaterThanOrEqual(slices, 1, name)
                // The slices tile [0, S): the last starts inside it, and none is longer than the first.
                XCTAssertLessThan((slices - 1) * length, tokens, "\(name) \(tokens)")
                XCTAssertLessThanOrEqual(Double(gpu * length * keys) * 4, Double(Attention.maxMatrix) * 1.0001, "\(name) \(tokens)")
                // Every index a kernel or a BNNS descriptor forms fits 32 bits: rows × width of q / out, and k/v.
                XCTAssertLessThan(tokens * heads * 128, Int(Int32.max), "\(name) \(tokens) q")
                XCTAssertLessThan(keys * kv * 128, Int(Int32.max), "\(name) \(tokens) k")
                XCTAssertLessThan(HeadSplitAttention.blockRows * keys, Int(Int32.max), "\(name) \(tokens) scores")
            }
        }
    }

    func testTheCPUBlocksTileEveryRowOfEveryCPUHeadAtLongLengths() {
        for (_, heads, kv) in families {
            let cpu = HeadSplitAttention.partition(heads: heads, fraction: 0.4, group: heads / kv).cpu
            for tokens in [16384, 24576, 24577] {
                let items = HeadSplitAttention.items(cpu: cpu, sequence: tokens, rows: HeadSplitAttention.blockRows)
                XCTAssertEqual(items.reduce(0) { $0 + $1.rows }, cpu.count * tokens)
                XCTAssertEqual(items.count, cpu.count * ((tokens + 127) / 128))
            }
        }
    }

    /// An M1 Pro's figures at 4,096 tokens (Anima, 16 heads; measured 24/10/2026): graph alone 84 ms, the cut at
    /// 6 CPU heads: GPU part 59 ms, CPU part 48 ms.
    private func m1(heads: Int = 16, cpuHeads: Int = 6, cpu: Double = 0.048) -> HeadSplitAttention.Balance {
        .init(heads: heads, graphSeconds: 0.0842 * Double(heads) / 16, cpuHeads: cpuHeads,
              gpuSeconds: 0.059 * Double(heads - cpuHeads) / 10, cpuSeconds: cpu * Double(cpuHeads) / 6)
    }

    func testTheShareStaysAt04WhereTheMeasuredFiguresAreTheM1Pros() {
        let items = [m1(), m1(heads: 30, cpuHeads: 12), m1(heads: 32, cpuHeads: 13)]
        XCTAssertNil(HeadSplitAttention.balance(items, margin: 0.05))
        // The prediction at the reference share is the cut's measured time to within the contention's fit.
        let t = m1().seconds(fraction: 0.4)!
        XCTAssertEqual(t, 0.059, accuracy: 0.004)
    }

    func testTheShareFollowsTheCPUsSpeedAndOnlyBeyondTheMargin() {
        // A CPU twice as slow (fewer P cores): the GPU should keep more heads.
        let slow = HeadSplitAttention.balance([m1(cpu: 0.096)], margin: 0.05)
        XCTAssertNotNil(slow)
        XCTAssertLessThan(slow ?? 1, 0.4)
        // A CPU twice as fast: it should take more.
        let fast = HeadSplitAttention.balance([m1(cpu: 0.024)], margin: 0.05)
        XCTAssertGreaterThan(fast ?? 0, 0.4)
        // The chosen share is faster than 0.4 by the margin, by the model's own account.
        for (cpu, share) in [(0.096, slow), (0.024, fast)] {
            guard let share else { continue }
            XCTAssertGreaterThan(m1(cpu: cpu).seconds(fraction: 0.4)! / m1(cpu: cpu).seconds(fraction: share)!, 1.05)
        }
        // Unusable figures (no CPU head measured, a zero time) decide nothing.
        XCTAssertNil(HeadSplitAttention.balance([.init(heads: 16, graphSeconds: 0.08, cpuHeads: 0, gpuSeconds: 0.08, cpuSeconds: 0)],
                                                margin: 0.05))
        XCTAssertNil(HeadSplitAttention.balance([], margin: 0.05))
    }
}
