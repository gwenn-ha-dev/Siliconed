import Foundation
import XCTest
@testable import Siliconed

/// **The capacity map's memory need** (`MemoryNeedDiagnostic.swift`): a pure function of shapes — no weight, no
/// engine, no GPU. The peak it is checked against is the report's (`Report.memory`), read by `diagnose` on a Mac.
final class DiagnosticMemoryTests: XCTestCase {
    private let gib = 1 << 30

    /// The operands are the largest shape of the grid: `a` 4 096 × 16 384, `b` 6 144 × 16 384 (both axes), `c` 4 096 × 16 384.
    func testOperandsAreTheGridsLargest() {
        let s = Diagnostic.Operands.sizes
        XCTAssertEqual(s.aFloats, 4096 * 16384)
        XCTAssertEqual(s.bFloats, 6144 * 16384)
        XCTAssertEqual(s.cFloats, 4096 * 16384)
        XCTAssertEqual(s.qValues, 4096 * 12288)
        XCTAssertGreaterThan(s.bytes, 1_100_000_000)
        XCTAssertLessThan(s.bytes, 1_300_000_000)
    }

    /// The batches cover the grid once, in the order it is measured; a batch fits its allotment, or is one point.
    func testBatchesCoverTheGridWithinTheAllotment() {
        for allotment in Diagnostic.Plan.allotments {
            let batches = Diagnostic.aneBatches(allotment: allotment)
            XCTAssertEqual(batches.flatMap { $0 }, Diagnostic.aneGrid)
            guard let allotment else { XCTAssertEqual(batches.count, 1); continue }
            for batch in batches where batch.count > 1 { XCTAssertLessThanOrEqual(Diagnostic.engineBytes(batch), allotment) }
        }
        let counts = Diagnostic.Plan.allotments.map { Diagnostic.aneBatches(allotment: $0).count }
        XCTAssertEqual(counts, counts.sorted(), "a smaller allotment never makes fewer engines")
    }

    /// The engine's bound is above what a share of one half takes, and it takes rows at 1 024 too (the lowest
    /// share would hand the ANE all of the product, which `NeuralEngine.rows` refuses: that bound was 0).
    func testEngineBoundCountsEveryHeight() {
        let at1024 = Diagnostic.engineBytes([.init(m: 1024, k: 3840, n: 3840)])
        let half = NeuralEngine.bytes(products: [(1024, 3840, 3840)], gpuShare: 0.5)
        XCTAssertGreaterThan(at1024, half)
        XCTAssertGreaterThan(Diagnostic.engineBytes([.init(m: 4096, k: 3840, n: 3840)]), at1024)
    }

    /// Only the ANE's term moves with the plan; the other steps are the operands' and the kernels'.
    func testOnlyTheANEStepFollowsTheAllotment() {
        let whole = Diagnostic.need(allotment: nil), none = Diagnostic.need(allotment: 0)
        for step in Diagnostic.needSteps where step != "aneCuts" { XCTAssertEqual(whole.steps[step], none.steps[step], step) }
        XCTAssertGreaterThan(whole.steps["aneCuts"]!, none.steps["aneCuts"]!)
        XCTAssertEqual(Set(whole.steps.keys), Set(Diagnostic.needSteps))
        // The ANE's grid on one engine is what swapped the MacBook Neo: it alone is above B = 4 GiB there.
        XCTAssertGreaterThan(whole.steps["aneCuts"]!, 4 * gib)
    }

    /// The plan is the fastest that fits B with the process's base; the Neo (8 GB, B = 4 GiB) gets a cut grid.
    func testPlanFitsBudget() throws {
        let big = try Diagnostic.Plan.choose(budget: 10 * gib, base: 16 << 20)
        XCTAssertNil(big.aneAllotment, "a Mac with room keeps one engine")
        for base in [16 << 20, 600 << 20] {
            let neo = try Diagnostic.Plan.choose(budget: 4 * gib, base: base)
            XCTAssertNotNil(neo.aneAllotment)
            XCTAssertLessThanOrEqual(neo.base + neo.need.peak, neo.budget)
            XCTAssertGreaterThan(Diagnostic.aneBatches(allotment: neo.aneAllotment).count, 1)
        }
        // Plans are ordered: the one chosen is the largest allotment that fits.
        let neo = try Diagnostic.Plan.choose(budget: 4 * gib, base: 16 << 20)
        let index = Diagnostic.Plan.allotments.firstIndex { $0 == neo.aneAllotment }!
        if index > 0 {
            XCTAssertGreaterThan(Diagnostic.need(allotment: Diagnostic.Plan.allotments[index - 1]).peak + (16 << 20), 4 * gib)
        }
    }

    /// A Mac whose B cannot hold even one point at a time is refused before the first product.
    func testPlanRefusesWhenNothingFits() {
        XCTAssertThrowsError(try Diagnostic.Plan.choose(budget: 1 * gib, base: 16 << 20)) { error in
            guard case EngineError.insufficientMemory(let needed, let budget)? = error as? EngineError else {
                return XCTFail("\(error)")
            }
            XCTAssertEqual(budget, 1 * gib)
            XCTAssertGreaterThan(needed, budget)
        }
    }

    /// The report carries it, and a report without it (before this field) reads back.
    func testMemoryFieldRoundTripsAndIsOptional() throws {
        let plan = try Diagnostic.Plan.choose(budget: 4 * gib, base: 16 << 20)
        let memory = Diagnostic.Memory(plan: plan, peaks: ["products": 2_500_000_000])
        let data = try JSONEncoder().encode(memory)
        XCTAssertEqual(try JSONDecoder().decode(Diagnostic.Memory.self, from: data), memory)
        XCTAssertEqual(memory.aneBatches, Diagnostic.aneBatches(allotment: plan.aneAllotment).count)
        XCTAssertEqual(memory.need, plan.need.steps)
    }
}
