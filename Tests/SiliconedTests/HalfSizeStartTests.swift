import XCTest
@testable import Siliconed

/// **The half-size start** (Z-Image's `Sampler.init(half:…)`, Anima's `denoise`): when the plan takes
/// it, and the two operators it adds — the images it changes are judged by eye, not here.
final class HalfSizeStartTests: XCTestCase {

    func testOffByDefaultAndThreeEvaluationsWhenOn() {
        let z = ZImageDenoisingModule(map: "")
        XCTAssertEqual(z.plan(height: 128, width: 128, steps: 8, start: 0, withLoRA: false, faster: false).reduced, 0)
        let plan = z.plan(height: 128, width: 128, steps: 8, start: 0, withLoRA: false, faster: true)
        XCTAssertEqual(plan.reduced, 3)
        XCTAssertEqual(plan.evaluations, 8)
        XCTAssertEqual(plan.reducedHeight, 64); XCTAssertEqual(plan.reducedWidth, 64)
        XCTAssertEqual(z.plan(height: 192, width: 128, steps: 8, start: 0, withLoRA: false, faster: true).reduced, 3)
    }

    func testNeverUnderTheFloorNorWithALoRANorOffTheRecipe() {
        let z = ZImageDenoisingModule(map: "")
        // 512², and 832×1216 whose half has a side under 512 px.
        XCTAssertEqual(z.plan(height: 64, width: 64, steps: 8, start: 0, withLoRA: false, faster: true).reduced, 0)
        XCTAssertEqual(z.plan(height: 152, width: 104, steps: 8, start: 0, withLoRA: false, faster: true).reduced, 0)
        XCTAssertEqual(z.plan(height: 128, width: 128, steps: 8, start: 0, withLoRA: true, faster: true).reduced, 0)
        XCTAssertEqual(z.plan(height: 128, width: 128, steps: 8, start: 3, withLoRA: false, faster: true).reduced, 0)
        XCTAssertEqual(z.plan(height: 128, width: 128, steps: 12, start: 0, withLoRA: false, faster: true).reduced, 0)
    }

    /// **Anima's plan**: the same rules, in 8-px cells with each half side a multiple of the ×2 patch.
    func testAnimaOffByDefaultThreeWhenOnNeverUnderTheRules() throws {
        try XCTSkipIf(Spectral.forced != nil, "SILICONED_SPECTRAL forces k")
        let anima = AnimaDenoisingModule(map: "")
        XCTAssertEqual(anima.plan(height: 128, width: 128, steps: 8, start: 0, withLoRA: false, faster: false).reduced, 0)
        let plan = anima.plan(height: 128, width: 128, steps: 8, start: 0, withLoRA: false, faster: true)
        XCTAssertEqual(plan.reduced, AnimaDenoisingModule.halfSizeSteps)
        XCTAssertEqual(plan.evaluations, 8)
        XCTAssertEqual([plan.reducedHeight, plan.reducedWidth], [64, 64])
        XCTAssertEqual(anima.plan(height: 192, width: 128, steps: 8, start: 0, withLoRA: false, faster: true).reduced,
                       AnimaDenoisingModule.halfSizeSteps)
        // 512², 832×1216 (a half side under 512 px), 1032² (a half side off the patch).
        XCTAssertEqual(anima.plan(height: 64, width: 64, steps: 8, start: 0, withLoRA: false, faster: true).reduced, 0)
        XCTAssertEqual(anima.plan(height: 152, width: 104, steps: 8, start: 0, withLoRA: false, faster: true).reduced, 0)
        XCTAssertEqual(anima.plan(height: 130, width: 130, steps: 8, start: 0, withLoRA: false, faster: true).reduced, 0)
        XCTAssertEqual(anima.plan(height: 128, width: 128, steps: 8, start: 0, withLoRA: true, faster: true).reduced, 0)
        XCTAssertEqual(anima.plan(height: 128, width: 128, steps: 8, start: 2, withLoRA: false, faster: true).reduced, 0)
        XCTAssertEqual(anima.plan(height: 128, width: 128, steps: 12, start: 0, withLoRA: false, faster: true).reduced, 0)
    }

    /// 𝓑x + σ𝓑ε: the 2×2 mean of `a + σ·b`, block by block, channel by channel.
    func testTheMeanTakesEachBlock() {
        let a: [Float] = (0..<32).map(Float.init)           // 2 channels of 4×4
        let b = [Float](repeating: 1, count: 32)
        var out = [Float](repeating: 0, count: 8)
        Resample.average(a, plus: 0.5, times: b, into: &out, channels: 2, height: 2, width: 2)
        let expected: [Float] = [3, 5, 11, 13, 19, 21, 27, 29]
        XCTAssertEqual(out, expected)
    }

    /// Normalized taps: a constant stays the constant, a ramp inside the edges stays a ramp on
    /// bilinear's grid (`src = j/2 − 1/4`), as Lanczos reproduces linear functions away from the edges.
    func testLanczosKeepsConstantsAndRamps() {
        let n = 16
        var out = [Float](repeating: 0, count: 4 * n * n)
        Resample.lanczosDouble([Float](repeating: 3, count: n * n), into: &out, channels: 1, height: n, width: n)
        XCTAssertTrue(out.allSatisfy { abs($0 - 3) < 1e-5 })
        let ramp = (0..<(n * n)).map { (i: Int) -> Float in Float(i % n) }
        Resample.lanczosDouble(ramp, into: &out, channels: 1, height: n, width: n)
        for j in 6..<(2 * n - 6) {
            XCTAssertEqual(out[5 * 2 * n + j], Float(j) / 2 - 0.25, accuracy: 0.02)
        }
    }
}
