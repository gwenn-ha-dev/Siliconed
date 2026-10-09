import XCTest
@testable import Siliconed

/// What Qwen-Image-2.1 computes without a byte of weights: its joint sequence (rows, blocks,
/// segments of the block-causal mask, RoPE positions) and its RoPE table. The expected values are
/// those the oracle recorded (`goldens-qwen21-dit-edit2-512.json`: `prefix`, `segments`).
final class QwenImage21Tests: XCTestCase {
    typealias Grid = QwenImage21Sequence.Grid

    /// Two references (26×38 then 32×32) and a 32×32 target, the edit-2 layout: 8 text positions,
    /// 247 slots, 6 text, 256 slots, 46 text, then the target's 256 slots.
    func testTheEditLayoutIsTheOracles() throws {
        let slots = [Bool](repeating: false, count: 8) + [Bool](repeating: true, count: 247)
            + [Bool](repeating: false, count: 6) + [Bool](repeating: true, count: 256)
            + [Bool](repeating: false, count: 46) + [Bool](repeating: true, count: 256)
        let s = try QwenImage21Sequence(slots: slots, images: [Grid(height: 26, width: 38), Grid(height: 32, width: 32),
                                                               Grid(height: 32, width: 32)])
        XCTAssertEqual(s.count, 3096); XCTAssertEqual(s.prefix, 2072); XCTAssertEqual(s.target, 1024)
        XCTAssertEqual(s.segments.map { [$0.start, $0.end, $0.isText ? 1 : 0] },
                       [[0, 8, 1], [8, 996, 0], [996, 1002, 1], [1002, 2026, 0], [2026, 2072, 1]])
        XCTAssertEqual(s.blockStarts, [8, 1002, 2072])
        XCTAssertEqual(s.textRows.count, 60)
        // The text after the first image reads the encoder row after its 247 slots.
        XCTAssertEqual(s.textRows[8].joint, 996); XCTAssertEqual(s.textRows[8].encoderRow, 255)
        // RoPE: text (p, p, p); the image frozen at p = 8, centred (y ∈ [−13, 13), x ∈ [−19, 19)),
        // then p += max(26, 38).
        XCTAssertEqual(s.positions[7], [7, 7, 7])
        XCTAssertEqual(s.positions[8], [8, -13, -19])
        XCTAssertEqual(s.positions[995], [8, 12, 18])
        XCTAssertEqual(s.positions[996], [46, 46, 46])
        XCTAssertEqual(s.positions[1002], [52, -16, -16])
        XCTAssertEqual(s.positions[2026], [84, 84, 84])
        XCTAssertEqual(s.positions[2072], [130, -16, -16])
        XCTAssertEqual(s.positions[3095], [130, 15, 15])
    }

    /// Two **adjacent** references stay two blocks: `image_ids` follow the token counts, not
    /// the runs of slots — otherwise they would see each other bidirectionally.
    func testAdjacentReferencesAreTwoBlocks() throws {
        let slots = [false, false] + [Bool](repeating: true, count: 4 + 4) + [false] + [Bool](repeating: true, count: 4)
        let s = try QwenImage21Sequence(slots: slots, images: [Grid(height: 4, width: 4), Grid(height: 4, width: 4),
                                                               Grid(height: 4, width: 4)])
        XCTAssertEqual(s.segments.map { [$0.start, $0.end, $0.isText ? 1 : 0] }, [[0, 2, 1], [2, 18, 0], [18, 34, 0], [34, 35, 1]])
        XCTAssertEqual(s.positions[18], [6, -2, -2])   // p = 2 + max(4, 4): the second block follows the first
    }

    func testAMismatchedLayoutIsRefused() {
        XCTAssertThrowsError(try QwenImage21Sequence(slots: [false, true, true], images: [Grid(height: 4, width: 4)]))
    }

    /// Pairs `[frame 8 | height 28 | width 28]`, `ωᵢ = 10⁴^(−2i/dim)`; negative positions rotate backwards.
    func testRoPETable() {
        var table = [Float](repeating: .nan, count: 128)
        table.withUnsafeMutableBufferPointer {
            QwenImage21Rope.write(into: $0.baseAddress!, positions: [[3, -2, 5]], axes: [16, 56, 56], theta: 10000)
        }
        XCTAssertEqual(table[0], cosf(3), accuracy: 1e-7); XCTAssertEqual(table[1], sinf(3), accuracy: 1e-7)
        XCTAssertEqual(table[2 * 1 + 1], Float(sin(3 / pow(10000, 2.0 / 16))), accuracy: 1e-7)
        XCTAssertEqual(table[2 * 8], cosf(-2), accuracy: 1e-7); XCTAssertEqual(table[2 * 8 + 1], sinf(-2), accuracy: 1e-7)
        XCTAssertEqual(table[2 * 36 + 1], sinf(5), accuracy: 1e-7)
        XCTAssertEqual(table[2 * 63 + 1], Float(sin(5 / pow(10000, 54.0 / 56))), accuracy: 1e-7)
    }

    /// `[cos | sin]`, cos first: at `t = 0` the first half is 1 and the second 0.
    func testTheSinusoidPutsCosineFirst() {
        let s = QwenImage21DiT.sinusoid(timestep: 0)
        XCTAssertEqual(s.count, 256)
        XCTAssertTrue(s[..<128].allSatisfy { $0 == 1 }); XCTAssertTrue(s[128...].allSatisfy { $0 == 0 })
        XCTAssertEqual(QwenImage21DiT.sinusoid(timestep: 1)[128], sin(1000), accuracy: 1e-12)
    }

    /// The LoRA is merged into the weight beyond `k·n / (k + n)` rows: 2 048 on `[4096, 4096]`, 3 072 on
    /// the MLP's two shapes — a 1024² generation (4 096 rows) merges everything, a 512² one (1 024) and
    /// the prefill's text rows nothing.
    func testTheLoRAMergesBeyondTheFLOPThreshold() {
        let (d, h) = (4096, 12288)
        XCTAssertFalse(LoRA.merges(rows: 2048, k: d, n: d)); XCTAssertTrue(LoRA.merges(rows: 2049, k: d, n: d))
        XCTAssertFalse(LoRA.merges(rows: 3072, k: d, n: h)); XCTAssertTrue(LoRA.merges(rows: 3073, k: d, n: h))
        XCTAssertFalse(LoRA.merges(rows: 3072, k: h, n: d)); XCTAssertTrue(LoRA.merges(rows: 3073, k: h, n: d))
        XCTAssertTrue(LoRA.merges(rows: 4096, k: d, n: h)); XCTAssertFalse(LoRA.merges(rows: 1024, k: d, n: d))
        XCTAssertFalse(LoRA.merges(rows: 40, k: d, n: d))
        XCTAssertTrue(LoRA.merges(rows: 4096, k: 64, n: d)); XCTAssertTrue(LoRA.merges(rows: 4096, k: d, n: 64))
        XCTAssertFalse(LoRA.merges(rows: 0, k: d, n: d))
    }

    /// The MLP is cut by rows only when its two `[rows, 12 288]` buffers would pass 512 MB: one chunk
    /// up to 5 461 rows (a 1024² generation, 4 096), two at 1024×1536 (6 144), equal chunks covering all.
    func testTheMLPIsCutOnlyWhenMemoryRequires() {
        let h = 12288
        XCTAssertEqual(QwenImage21DiT.mlpRows(scratch: 1024, hidden: h), 1024)
        XCTAssertEqual(QwenImage21DiT.mlpRows(scratch: 4096, hidden: h), 4096)
        XCTAssertEqual(QwenImage21DiT.mlpRows(scratch: 5461, hidden: h), 5461)
        XCTAssertEqual(QwenImage21DiT.mlpRows(scratch: 5462, hidden: h), 2731)
        XCTAssertEqual(QwenImage21DiT.mlpRows(scratch: 6144, hidden: h), 3072)
        XCTAssertEqual(QwenImage21DiT.mlpRows(scratch: 1, hidden: h), 1)
        for n in [1, 17, 2048, 4096, 6144, 6211, 9000, 16_384] {
            let rows = QwenImage21DiT.mlpRows(scratch: n, hidden: h)
            let chunks = (n + rows - 1) / rows
            XCTAssertLessThanOrEqual(2 * rows * h * 4, QwenImage21DiT.mlpBudget, "\(n) rows")
            XCTAssertEqual((0..<chunks).map { ($0 + 1) * n / chunks - $0 * n / chunks }.reduce(0, +), n)
            XCTAssertTrue((0..<chunks).allSatisfy { ($0 + 1) * n / chunks - $0 * n / chunks <= rows })
        }
    }
}
