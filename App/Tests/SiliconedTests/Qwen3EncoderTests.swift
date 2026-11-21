import Foundation
import XCTest
@testable import Siliconed

/// The CPU's side of the Qwen3 encoder on the new engine: what it writes into the stream and reads back, the final norm,
/// the paliers, the head's stretches and its argmax.
final class Qwen3EncoderTests: XCTestCase {
    /// A row written into the f16 stream comes back within f16's rounding of its true value, its largest in [2¹³, 2¹⁴].
    func testTheF16StreamKeepsARowUnderItsFactor() {
        let d = 64
        let values = (0..<(2 * d)).map { i in Float(sin(Double(i))) * (i < d ? 3e5 : 1e-3) }
        let to = UnsafeMutableRawPointer.allocate(byteCount: 2 * d * 2, alignment: 16), factors = UnsafeMutablePointer<Float>.allocate(capacity: 2)
        defer { to.deallocate(); factors.deallocate() }
        Qwen3Encoder.write(values, rows: 0..<2, width: d, half: true, to: to, factors: factors)
        let h = to.assumingMemoryBound(to: Float16.self)
        for r in 0..<2 {
            let top = (0..<d).map { abs(Float(h[r * d + $0])) }.max()!
            XCTAssert(top >= 0x1p13 && top <= 0x1p14, "row \(r): largest \(top)")
        }
        let back = (0..<(2 * d)).map { Float(h[$0]) * factors[$0 / d] }
        for (a, b) in zip(values, back) { XCTAssertEqual(a, b, accuracy: abs(a) * 1e-3) }
    }

    /// A bf16 stream holds a value rounded to nearest even, and gives it back exactly.
    func testTheBF16StreamRoundsOnce() {
        let values: [Float] = [1, 1.00390625, 1 + 0x1p-8 + 0x1p-9, -3.5]
        let to = UnsafeMutableRawPointer.allocate(byteCount: 8, alignment: 16)
        defer { to.deallocate() }
        Qwen3Encoder.write(values, rows: 0..<1, width: 4, half: false, to: to, factors: nil)
        let b = to.assumingMemoryBound(to: UInt16.self)
        XCTAssertEqual((0..<4).map { Float(bitPattern: UInt32(b[$0]) << 16) }, values.map(BFloat.round))
    }

    /// The final norm in bf16 rounds the normalized value, then its product by γ.
    func testTheBF16FinalNormRoundsTwice() {
        let x: [Float] = [1, 2, 3, 4], gain: [Float] = [1.5, 1, 1, 1]
        let k = 1 / ((1 + 4 + 9 + 16) / Float(4) + 1e-6).squareRoot()
        let y = Qwen3Encoder.normed(x, rows: 1, gain: gain, eps: 1e-6, bf16: true)
        XCTAssertEqual(y[0], BFloat.round(BFloat.round(k) * 1.5))
    }

    func testPaliersAndStretches() throws {
        XCTAssertEqual(try Qwen3Encoder.palier(19), 64)
        XCTAssertEqual(try Qwen3Encoder.palier(1032), 1088)
        XCTAssertThrowsError(try Qwen3Encoder.palier(5000))
        XCTAssertEqual(Qwen3Head.stretches(151_936), 16)
        XCTAssertEqual(Qwen3Head.stretches(7), 1)
    }

    /// Greedy ties go to the lowest identifier, as `torch.argmax`.
    func testArgmaxTiesToTheLowest() {
        let logits: [Float] = [0, 3, 1, 3, -2]
        XCTAssertEqual(Qwen3Head.argmax(logits[...]), 1)
        XCTAssertEqual(Qwen3Head.argmax(logits[2...]), 1)
    }
}
