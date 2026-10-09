import XCTest
@testable import Siliconed

/// What Qwen-Image-2.1's encoder (Qwen3-VL-8B) computes without a byte of weights: the template,
/// the image token expansion, the 3D positions, the mRoPE axes, the processor's geometry and
/// Pillow's Lanczos. The identifiers themselves are checked against the oracle (`qwen21-text`,
/// `qwen21-vision`): the tokenizer lives in the store, not in the tests.
final class Qwen3VLTests: XCTestCase {
    /// `prompt_template_t2i` / `_ti2i` of `QwenImage21Pipeline`: a space between two images,
    /// none before the prompt; an empty prompt becomes " ".
    func testTheTemplateIsThePipelines() {
        let head = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n"
        let tail = "<|im_end|>\n<|im_start|>assistant\n"
        XCTAssertEqual(Qwen3VLPrompt.template(prompt: "a cat", images: 0), head + "a cat" + tail)
        XCTAssertEqual(Qwen3VLPrompt.template(prompt: "", images: 0), head + " " + tail)
        XCTAssertEqual(Qwen3VLPrompt.template(prompt: "x", images: 2),
                       head + "<image1><|vision_start|><|image_pad|><|vision_end|> "
                       + "<image2><|vision_start|><|image_pad|><|vision_end|>x" + tail)
        XCTAssertTrue(Qwen3VLPrompt.template(prompt: "x", images: 1).hasPrefix(Qwen3VLPrompt.systemTurn))
    }

    func testEachImagePadIsExpandedToItsMergedTokens() {
        XCTAssertEqual(try Qwen3VLPrompt.expand([1, 9, 2, 9, 3], image: 9, counts: [2, 3]), [1, 9, 9, 2, 9, 9, 9, 3])
    }

    /// `get_rope_index`: text counts up; a 4×6 patch grid (2×3 merged) sits at `(s, s+r, s+c)`;
    /// the text after it resumes at `s + max(4, 6)/2 = s + 3`, not at `s + 6`.
    func testThreeDimensionalPositions() {
        let ids = [5, 5, 9, 9, 9, 9, 9, 9, 5]
        let p = Qwen3VLPrompt.positions(ids: ids, image: 9, grids: [(4, 6)])
        XCTAssertEqual(p[0], [0, 1, 2, 2, 2, 2, 2, 2, 5])
        XCTAssertEqual(p[1], [0, 1, 2, 2, 2, 3, 3, 3, 5])
        XCTAssertEqual(p[2], [0, 1, 2, 3, 4, 2, 3, 4, 5])
        // Text only: the three axes are the index — the 1D RoPE.
        let text = Qwen3VLPrompt.positions(ids: [1, 2, 3], image: 9, grids: [])
        XCTAssertEqual(text, [[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    }

    /// Interleaved (24, 20, 20): 0, 3, …, 57 and 60–63 → t; 1, 4, …, 58 → h; 2, 5, …, 59 → w.
    func testTheMRoPEAxes() {
        let axes = (0..<64).map { TextEncoder.mropeAxis(frequency: $0, sections: [24, 20, 20]) }
        XCTAssertEqual(axes.filter { $0 == 0 }.count, 24)
        XCTAssertEqual(axes.filter { $0 == 1 }.count, 20)
        XCTAssertEqual(axes.filter { $0 == 2 }.count, 20)
        XCTAssertEqual(Array(axes[0..<6]), [0, 1, 2, 0, 1, 2])
        XCTAssertEqual(Array(axes[57..<64]), [0, 1, 2, 0, 0, 0, 0])
    }

    /// The test images' sizes, as the oracle recorded them (`condition_sizes`): 1216×832 → 608×416,
    /// 1024² → 512² at R = 512; and `smart_resize` is the identity on them.
    func testTheConditionSizesAndSmartResize() {
        XCTAssertTrue(Qwen3VLImages.conditionSize(width: 1216, height: 832, resolution: 512) == (608, 416))
        XCTAssertTrue(Qwen3VLImages.conditionSize(width: 1024, height: 1024, resolution: 512) == (512, 512))
        XCTAssertTrue(Qwen3VLImages.smartResize(height: 416, width: 608) == (416, 608))
        XCTAssertTrue(Qwen3VLImages.smartResize(height: 1000, width: 1000) == (992, 992))
        XCTAssertTrue(Qwen3VLImages.smartResize(height: 100, width: 100) == (256, 256))  // min_pixels 65 536
    }

    /// Patches in 2×2 merge-block order, the time copy, and the fused normalization.
    func testPixelValuesLayout() throws {
        // 256² (min_pixels): a 16×16 patch grid; each patch's red value is its raster index.
        var bytes = [UInt8](repeating: 0, count: 256 * 256 * 3)
        for y in 0..<256 { for x in 0..<256 { bytes[(y * 256 + x) * 3] = UInt8((y / 16) * 16 + x / 16) } }
        let p = try Qwen3VLImages.pixels(Qwen3VLImages.RGB8(bytes: bytes, width: 256, height: 256))
        XCTAssertEqual(p.gridHeight, 16); XCTAssertEqual(p.gridWidth, 16)
        XCTAssertEqual(p.values.count, 256 * 1536)
        // Patch k's red plane, frame 0 then frame 1, in block order: (0,0) (0,1) (1,0) (1,1) (0,2)…
        for (k, patch) in [0, 1, 16, 17, 2, 3, 18, 19].enumerated() {
            let expected = (Float(patch) - 127.5) / 127.5
            XCTAssertEqual(p.values[k * 1536], expected)
            XCTAssertEqual(p.values[k * 1536 + 256], expected)          // the temporal copy
            XCTAssertEqual(p.values[k * 1536 + 512], (0 - 127.5) / 127.5) // green
        }
        XCTAssertThrowsError(try Qwen3VLImages.pixels(Qwen3VLImages.RGB8(bytes: [UInt8](repeating: 0, count: 300 * 300 * 3),
                                                                         width: 300, height: 300)))
    }

    /// Bilinear `align_corners=True` on 48²: the grid's corners land on the table's corners.
    func testPositionTapsHitTheCorners() {
        let t = Qwen3VLImages.positionTaps(gridHeight: 4, gridWidth: 4, side: 48)
        // Patch 0 is (0, 0): all its weight on index 0.
        XCTAssertEqual(t.indices[0], 0); XCTAssertEqual(t.weights[0], 1)
        // The last patch in block order is (3, 3) → source (47, 47).
        let last = t.indices.count - 4
        let total = (0..<4).reduce(Float(0)) { $0 + t.weights[last + $1] * (t.indices[last + $1] == 47 * 48 + 47 ? 1 : 0) }
        XCTAssertEqual(total, 1)
        for p in 0..<16 { XCTAssertEqual(t.weights[(4 * p)..<(4 * p + 4)].reduce(0, +), 1, accuracy: 1e-6) }
    }

    /// Pillow's fixed-point coefficients sum to one (2²² within rounding), and a constant image
    /// stays constant through a 2× Lanczos reduction.
    func testPillowLanczos() {
        let c = PillowResample.coefficients(input: 1216, output: 608)
        XCTAssertEqual(c.ksize, 13)
        for xx in [0, 300, 607] {
            let sum = (0..<c.bounds[xx].1).reduce(0) { $0 + Int(c.k[xx * c.ksize + $1]) }
            XCTAssertLessThan(abs(sum - (1 << 22)), 16)
        }
        let flat = Qwen3VLImages.RGB8(bytes: [UInt8](repeating: 200, count: 64 * 48 * 3), width: 64, height: 48)
        let small = PillowResample.lanczos(flat, width: 32, height: 24)
        XCTAssertEqual(Set(small.bytes), [200])
    }
}
