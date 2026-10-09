import XCTest
@testable import Siliconed

/// What ERNIE-Image computes without a byte of weights: its schedule and its RoPE.
final class ErnieTests: XCTestCase {
    /// `linspace(1, 0, 9)[:-1]` shifted by `shift = 4`, as the oracle recorded it
    /// (`goldens-ernie-trajectory-512`, `sigmas`): 1, 28/29, 12/13, 20/23, 4/5, 12/17, 4/7, 4/11, 0.
    func testTheScheduleIsTheSchedulers() {
        let σ = ErnieDiT.sigmas(steps: 8)
        let expectedCounts: [Double] = [1, 28.0 / 29, 12.0 / 13, 20.0 / 23, 0.8, 12.0 / 17, 4.0 / 7, 4.0 / 11, 0]
        XCTAssertEqual(σ.count, 9)
        for (a, b) in zip(σ, expectedCounts) { XCTAssertEqual(Double(a), b, accuracy: 1e-7) }
    }

    /// The pair `(j, j + 64)` rotates by two angles: `e[j/2]` and `e[32 + j/2]`. At text `t = 0`
    /// all the angles are zero — the table is the identity.
    func testRoPEWithTwoAngles() {
        var table = [Float](repeating: .nan, count: 2 * 4 * 64)
        table.withUnsafeMutableBufferPointer {
            ErnieRope.write(into: $0.baseAddress!, textRows: 1, tilesHigh: 1, tilesWide: 1, axes: [32, 48, 48], theta: 256)
        }
        // Row 1: the text, at (0, 0, 0).
        for j in 0..<64 { XCTAssertEqual(Array(table[(256 + 4 * j)..<(256 + 4 * j + 4)]), [1, 0, 1, 0]) }
        // Row 0: the image (0, 0) at (T = 1, 0, 0) — only the first axis rotates (e₀…e₁₅):
        // F[j] = e[j/2] for j < 32; F[j + 64] = e[32 + j/2], zero (axes 1 and 2 at zero).
        XCTAssertEqual(table[1], sinf(1), accuracy: 1e-7)          // sin e₀, ω₀ = 1
        XCTAssertEqual(table[4 * 31 + 1], sinf(1 / powf(256, 30 / 32)), accuracy: 1e-7)
        XCTAssertEqual(table[4 * 40 + 1], 0)                        // F[40] = e[20]: axis 1
        for j in 0..<64 { XCTAssertEqual(table[4 * j + 3], 0) }     // F[j + 64]: axes 1–2
        var x = [Float](repeating: 0, count: 128)
        x[0] = 1; x[64] = 2
        table.withUnsafeBufferPointer { t in
            x.withUnsafeMutableBufferPointer { ErnieRope.apply($0.baseAddress!, table: t.baseAddress!, rows: 1, heads: 1, headDim: 128) }
        }
        XCTAssertEqual(x[0], cosf(1) - 2 * sinf(1), accuracy: 1e-6)
        XCTAssertEqual(x[64], 2 * 1 + 1 * 0, accuracy: 1e-6)        // F[64] zero: 2·cos 0 + 1·sin 0
    }
}
