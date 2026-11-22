import SiliCore
import XCTest
@testable import Siliconed

/// **Qwen3-VL's vision tower without a byte of weights**: the weights its functions read, the RoPE table, the
/// positions resampled from the table.
final class Qwen3VLVisionTests: XCTestCase {

    /// Every weight a function argument reads, in its layer's group: 27 blocks', four mergers' (each deepstack one after
    /// its block, the merger last); the embedding's resident.
    func testEveryArgumentHasItsWeight() {
        let c = Qwen3VLVision.Config.eightB, v = Qwen3VLVision(gridHeight: 2, gridWidth: 2)
        let groups = v.groups()
        XCTAssertEqual(groups.map(\.name).suffix(5), ["blocks.24", "deepstack.2", "blocks.25", "blocks.26", "merger"])
        XCTAssertEqual(groups.firstIndex { $0.name == "deepstack.0" }, 9)
        let blockNames = Set(groups.first { $0.name == "blocks.26" }!.weights.map(\.name))
        XCTAssertEqual(blockNames.count, 12)
        for a in Qwen3VLVision.blockArguments(c, rows: 4).dropFirst() { XCTAssertTrue(blockNames.contains(a.name), a.name) }
        let mergerNames = Set(groups.first { $0.name == "deepstack.2" }!.weights.map(\.name))
        for a in Qwen3VLVision.mergerArguments(c, rows: 4, deepstack: true).dropFirst() { XCTAssertTrue(mergerNames.contains(a.name), a.name) }
        let residents = Set(v.declaration().residents.map(\.name))
        for a in Qwen3VLVision.embedArguments(c, rows: 4) where a.name.hasPrefix("kernel") { XCTAssertTrue(residents.contains(a.name)) }
    }

    /// `[h | w | h | w]`: the first patch at angle 0, the second (row 0, column 1) turning the w halves only.
    func testTheAxialRopeTable() {
        let v = Qwen3VLVision(gridHeight: 2, gridWidth: 2), d = v.config.headDim
        let (cos, sin) = v.rope()
        XCTAssertEqual(cos.count, 4 * d)
        XCTAssertTrue(cos[0..<d].allSatisfy { $0 == 1 }); XCTAssertTrue(sin[0..<d].allSatisfy { $0 == 0 })
        // Patch 1 is (0, 1): the h quarter still at 0, the w quarter's first frequency 1 radian.
        XCTAssertEqual(sin[d + 0], 0); XCTAssertEqual(sin[d + d / 4], sinf(1)); XCTAssertEqual(sin[d + d / 2 + d / 4], sinf(1))
    }

    /// On a 48² table whose rows hold their own index, a grid of 48² samples each row once, exactly.
    func testThePositionsOfAGridTheTablesSize() {
        let c = Qwen3VLVision.Config.eightB, v = Qwen3VLVision(gridHeight: 48, gridWidth: 48)
        let table = (0..<(48 * 48 * c.width)).map { Float($0 / c.width) }
        var out = [UInt16](repeating: 0, count: 48 * 48 * c.width)
        table.withUnsafeBufferPointer { t in out.withUnsafeMutableBufferPointer { v.positions(table: t.baseAddress!, into: $0.baseAddress!) } }
        let order = Qwen3VLImages.patchPositions(gridHeight: 48, gridWidth: 48)
        for n in [0, 1, 2, 3, 100, 2303] {
            let expected = Float(order[n].row * 48 + order[n].column)
            XCTAssertEqual(BFloat16Bits.widened(out[n * c.width]), BFloat16Bits.widened(BFloat16Bits.rounded(expected)), "patch \(n)")
        }
    }
}
