import XCTest
@testable import Siliconed

/// **MOSS-Image's DiT, without a byte of weights**: the scheduler's sigmas against the reference's
/// (`MOSSImageScheduler.set_timesteps`, torch 2.14 on this machine) and the modules a LoRA reaches.
final class MossImageTests: XCTestCase {

    // MARK: - the sigmas

    /// The recipe's 8 steps are values of the training grid: 0.968, not the shifted `linspace`'s 0.9679.
    func testTheEightStepsAreTheTrainingGridsValues() {
        let sigmas = MossImageSchedule.sigmas(steps: 8)
        XCTAssertEqual(sigmas, [1, 0.968, 0.926, 0.871, 0.791, 0.671, 0.46300002932548523, 0.024, 0])
        // Timesteps are σ·1000 in fp32: 463.00003, not 463.
        XCTAssertEqual(MossImageSchedule.timesteps(sigmas), [1000, 968, 926, 871, 791, 671, 463.0000305175781, 24])
    }

    /// All of the 50-step schedule, to the bit (torch's values; the official recipe's).
    func testFiftyStepsToTheBit() {
        let reference: [Float] = [
            1.0, 0.9959999918937683, 0.9919999837875366, 0.9869999885559082, 0.9829999804496765, 0.9779999852180481,
            0.9729999899864197, 0.9679999947547913, 0.9629999995231628, 0.9570000171661377, 0.9520000219345093,
            0.9459999799728394, 0.9390000104904175, 0.9329999685287476, 0.9259999990463257, 0.918999969959259,
            0.9120000004768372, 0.9049999713897705, 0.8970000147819519, 0.8880000114440918, 0.8799999952316284,
            0.8709999918937683, 0.8610000014305115, 0.8510000109672546, 0.8399999737739563, 0.8289999961853027,
            0.8169999718666077, 0.8050000071525574, 0.7910000085830688, 0.7770000100135803, 0.7619999647140503,
            0.7459999918937683, 0.7289999723434448, 0.7109999656677246, 0.6919999718666077, 0.6710000038146973,
            0.6480000019073486, 0.6229999661445618, 0.597000002861023, 0.5679999589920044, 0.5359999537467957,
            0.5009999871253967, 0.46300002932548523, 0.42100003361701965, 0.37300002574920654, 0.32100000977516174,
            0.26100000739097595, 0.19300000369548798, 0.11500000208616257, 0.024000000208616257, 0.0]
        XCTAssertEqual(MossImageSchedule.sigmas(steps: 50).map(\.bitPattern), reference.map(\.bitPattern))
    }

    /// A dense schedule has fewer σ than steps: consecutive duplicates are removed after the snapping —
    /// 250 steps give 246 evaluations, 1000 give 615 (torch's counts).
    func testADenseScheduleLosesItsDuplicates() {
        for (steps, evaluations, middle) in [(250, 246, Float(0.8289999961853027)), (500, 415, 0.781000018119812),
                                             (1000, 615, 0.6919999718666077), (1500, 739, 0.6299999952316284)] {
            let sigmas = MossImageSchedule.sigmas(steps: steps)
            XCTAssertEqual(sigmas.count - 1, evaluations, "\(steps) steps")
            XCTAssertEqual(sigmas[(evaluations + 1) / 2], middle, "\(steps) steps")
            XCTAssertEqual(sigmas[evaluations - 1], 0.024)
            XCTAssertEqual(sigmas[1], 0.999)
        }
    }

    func testEverySigmaIsOnTheTrainingGridAndTheyDescend() {
        let grid = Set(MossImageSchedule.trainingGrid.map(\.bitPattern))
        for steps in [1, 2, 3, 5, 13, 28, 100] {
            let sigmas = MossImageSchedule.sigmas(steps: steps)
            XCTAssertEqual(sigmas.last, 0)
            XCTAssertEqual(sigmas.first, 1)
            XCTAssertEqual(sigmas, sigmas.sorted(by: >), "\(steps)")
            XCTAssertEqual(Set(sigmas).count, sigmas.count, "\(steps): duplicates")
            XCTAssert(sigmas.dropLast().allSatisfy { grid.contains($0.bitPattern) }, "\(steps)")
        }
        XCTAssertEqual(MossImageSchedule.sigmas(steps: 1), [1, 0])
        // 1000 on 0.001 … 1: the grid, 0.001 apart.
        XCTAssertEqual(MossImageSchedule.trainingGrid.count, 1000)
        XCTAssertEqual(MossImageSchedule.trainingGrid.first, 0.001)
        XCTAssertEqual(MossImageSchedule.trainingGrid.last, 1)
    }

    /// `torch.linspace` on this machine fuses each value's multiply-add: the unfused formula misses 239 of
    /// the training grid's 1000 values.
    func testLinspaceFusesItsMultiplyAdd() {
        let fused = MossImageSchedule.linspace(1, 0.001, 1000)
        let step = (Float(0.001) - 1) / 999
        let unfused = (0..<1000).map { $0 < 500 ? 1 + step * Float($0) : Float(0.001) - step * Float(999 - $0) }
        XCTAssertGreaterThan(zip(fused, unfused).filter { $0 != $1 }.count, 100)
        XCTAssertEqual(fused.first, 1); XCTAssertEqual(fused.last, 0.001)
    }

    // MARK: - the DiT's shapes

    /// The text's rows take a palier (128, 512, 1152, then multiples of 64) and as many more as bring the keys to a
    /// multiple of 64, where a call over the sequence ends; the sequence is the image's rows then the text's.
    func testTheTextsPalierBringsTheKeysToSixtyFour() {
        XCTAssertEqual(MossDiT.Geometry.palier(40, image: 1024), 128)
        XCTAssertEqual(MossDiT.Geometry.palier(129, image: 1024), 512)
        XCTAssertEqual(MossDiT.Geometry.palier(1200, image: 4096), 1216)
        let g = MossDiT.Geometry(height: 30, width: 34, text: 100)
        XCTAssertEqual(g.image, 1020)
        XCTAssertEqual(g.keys % 64, 0)
        XCTAssertEqual(g.palier, 128 + 4)
        XCTAssertEqual(g.textRows, 192)
        XCTAssertEqual(g.sequence, 1120)
    }
}
