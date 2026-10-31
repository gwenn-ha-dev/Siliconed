import XCTest
@testable import Siliconed

/// Krea 2's and Qwen-Image-2.1's timestep embeddings share one frequency table, `exp(−ln(10⁴) · j / half)`. MOSS-Image's
/// computes `pow(10⁴, −j / half)`, which is **not** the same double — a unification was tried and this test is what refused it:
/// the two forms differ in the last bit on most entries, hence in the embedding's bits.
final class SinusoidTests: XCTestCase {
    func testTheSharedTableIsTheReferencesExpLogForm() {
        for half in [64, 128, 256] {
            let shared = Sinusoid.frequencies(half: half)
            XCTAssertEqual(shared.count, half)
            for j in 0..<half {
                XCTAssertEqual(shared[j], exp(-log(10000) * Double(j) / Double(half)))
            }
        }
    }

    func testMossImageKeepsItsPowBecauseTheTwoFormsDiffer() {
        let half = 128
        let shared = Sinusoid.frequencies(half: half)
        let differing = (0..<half).filter { shared[$0] != pow(10000, -Double($0) / Double(half)) }
        XCTAssertGreaterThan(differing.count, 0, "if the forms ever agree, MOSS-Image may share the table")
        let moss = MossImageExact.sinusoid(timestep: 1000, dim: 2 * half)
        let expected = Float(cos(1000 * pow(10000, -Double(7) / Double(half))))
        XCTAssertEqual(moss[7], expected)
    }
}
