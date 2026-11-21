import XCTest
@testable import Siliconed

/// **An edit, from the request to the DiT's sequence, without a byte of weights**: the output's
/// format, what is refused before any computation, the order of the images and their `<imageN>`.
/// The chains have empty paths: if the engine touched a file before refusing, the error would be
/// `fileMissing`, not the one expected (`RequestTests`' convention).
final class EditSequenceTests: XCTestCase {

    private let qwen = Model(card: .qwenImage21, chain: try! Chain(
        text: QwenImage21TextModule(map: "", tokenizer: ""),
        denoising: QwenImage21DenoisingModule(map: ""),
        decoding: QwenImage21DecodingModule(path: ""), encoding: QwenImage21EncodingModule(path: "")))

    private func image(_ width: Int, _ height: Int, rgb: (Float, Float, Float) = (0, 0, 0)) -> ImageRGB {
        let plane = width * height
        return ImageRGB(pixels: [Float](repeating: rgb.0, count: plane) + [Float](repeating: rgb.1, count: plane)
                            + [Float](repeating: rgb.2, count: plane), height: height, width: width)
    }

    /// Reference sizes, from a phone's portrait to a 10:1 strip, both orientations.
    private let sizes: [(Int, Int)] = {
        var s: [(Int, Int)] = [(1216, 832), (1024, 1024), (4032, 3024), (3024, 4032), (1920, 1080), (640, 480),
                               (300, 4000), (8000, 1000), (1, 1000), (1000, 1), (512, 512), (333, 777)]
        for w in stride(from: 200, through: 6000, by: 290) {
            for h in stride(from: 150, through: 6000, by: 370) { s.append((w, h)) }
        }
        return s
    }()

    // ── the output's format ───────────────────────────────────────────────────────────────────

    /// **Qwen-Image-2.1's rule** (`editFormat`, the module's, which the developer's command line, the
    /// app and `silicontrol add --ref` all call): each side ≥ 512, a multiple of 32, the area under
    /// `Format.editSurface` — so `Format.check` always passes — and image 1’s orientation never flipped (a near-square may come out square).
    func testQwenEditFormatIsAlwaysRenderable() {
        let d = qwen.chain.denoising
        for (w, h) in sizes {
            let f = d.editFormat(referenceWidth: w, referenceHeight: h)
            XCTAssertGreaterThanOrEqual(min(f.width, f.height), Format.minimumSide, "\(w)×\(h) → \(f)")
            XCTAssertEqual(f.width % 32, 0, "\(w)×\(h) → \(f)"); XCTAssertEqual(f.height % 32, 0, "\(w)×\(h) → \(f)")
            XCTAssertLessThanOrEqual(f.width * f.height, Format.editSurface, "\(w)×\(h) → \(f)")
            XCTAssertNoThrow(try Format.check(width: f.width, height: f.height), "\(w)×\(h) → \(f)")
            if f.width != f.height { XCTAssertEqual(f.width > f.height, w > h, "\(w)×\(h) → \(f): orientation flipped") }
        }
        // The test photo and the square: what the edit renders and the oracle used.
        XCTAssertTrue(d.editFormat(referenceWidth: 1216, referenceHeight: 832) == (1248, 832))
        XCTAssertTrue(d.editFormat(referenceWidth: 4032, referenceHeight: 3024) == (1184, 896))
        XCTAssertTrue(d.editFormat(referenceWidth: 1, referenceHeight: 1) == (1024, 1024))
    }

    /// **The protocol's default** (a denoiser that does not edit): the reference's own size floored
    /// to a multiple of 16, each side raised to 512, the long side shrunk under `Format.editSurface`.
    func testTheDefaultEditFormatIsAlwaysRenderable() {
        let d = ZImageDenoisingModule(map: "")
        for (w, h) in sizes {
            let f = d.editFormat(referenceWidth: w, referenceHeight: h)
            XCTAssertEqual(f.width % 16, 0, "\(w)×\(h) → \(f)"); XCTAssertEqual(f.height % 16, 0, "\(w)×\(h) → \(f)")
            XCTAssertNoThrow(try Format.check(width: f.width, height: f.height), "\(w)×\(h) → \(f)")
        }
        XCTAssertTrue(d.editFormat(referenceWidth: 1216, referenceHeight: 832) == (1216, 832))
        XCTAssertTrue(d.editFormat(referenceWidth: 1000, referenceHeight: 1) == (992, 512))
    }

    /// Past about 4.2:1 the short side is raised to 512 keeping the ratio, then the area capped: a 5:1 reference
    /// renders 2560×512.
    func testAnElongatedReferenceKeepsItsRatio() {
        XCTAssertTrue(qwen.chain.denoising.editFormat(referenceWidth: 5000, referenceHeight: 1000) == (2560, 512))
        XCTAssertTrue(qwen.chain.denoising.editFormat(referenceWidth: 1000, referenceHeight: 5000) == (512, 2560))
    }

    // ── what is refused before any computation ──────────────────────────────────────────────

    /// Three references pass the count (the request then stops on its 6 steps, which the Turbo has no
    /// schedule for — still before any file); a fourth is refused as such.
    func testAtMostThreeReferences() {
        XCTAssertEqual(qwen.chain.denoising.maxReferences, 3)
        let reference = image(64, 64)
        var r = Request("replace the cloudy sky with a blue sky", resolution: 512, steps: 6)
        r.references = Array(repeating: reference, count: 3)
        XCTAssertThrowsError(try Engine().render(r, model: qwen)) {
            XCTAssertEqual($0 as? EngineError, .invalidSteps(steps: 6, allowed: [8]), "\($0)")
        }
        r.references.append(reference)
        XCTAssertThrowsError(try Engine().render(r, model: qwen)) {
            XCTAssertEqual($0 as? EngineError, .tooManyReferences(count: 4, max: 3), "\($0)")
        }
    }

    /// **No img2img**: its pipeline has none. A starting image is refused before any computation,
    /// with a message that says where the image goes instead (image 1, `--ref`).
    func testImageToImageIsRefusedCleanly() {
        let r = Request("a woman posing in a library", resolution: 512, image: image(64, 64))
        XCTAssertThrowsError(try Engine().render(r, model: qwen)) {
            XCTAssertEqual($0 as? EngineError, .imageToImageUnsupported(model: "qwen-image-2.1"), "\($0)")
            XCTAssertTrue("\($0)".contains("reference 1"), "\($0)")
        }
        // The Turbo's published count passes the module's own check; another count does not.
        XCTAssertNoThrow(try qwen.chain.denoising.check(steps: 8, startImage: false))
        for n in [5, 6, 7, 9] { XCTAssertThrowsError(try qwen.chain.denoising.check(steps: n, startImage: false), "\(n)") }
    }

    /// What the chain declares: the text, the references, and the encoder that sees them. ⚠️ It also
    /// declares `.image` (img2img), because it has an image encoder (the references need it) — and
    /// the denoiser refuses img2img. The app ignores `.image`; another app would offer a field
    /// that always fails.
    func testTheChainDeclaresTheEdit() {
        XCTAssertTrue(qwen.chain.text.output.readsImages)
        XCTAssertTrue(qwen.chain.text is any ImageTextModule)
        XCTAssertTrue(qwen.chain.entries.contains { $0.kind == .reference })
        XCTAssertFalse(qwen.chain.entries.contains { $0.kind == .image })
    }

    // ── the sequence ────────────────────────────────────────────────────────────────────────

    /// **The references prepared**: each at ~R² at its own ratio, sides at 32 — a latent cell is a vision patch, so a
    /// reference takes a quarter of its cells in slots. A transparent one keeps its RGBA for the VAE.
    func testAReferenceIsPreparedForBothEncoders() throws {
        let prepared = try qwen.chain.denoising.preparedReferences([image(1216, 832), image(640, 480, rgb: (1, 1, 1))])
        XCTAssertTrue(prepared.map { [$0.width, $0.height] } == [[1248, 832], [1184, 896]], "\(prepared.map { [$0.width, $0.height] })")
        XCTAssertNil(prepared[0].rgba)
        XCTAssertEqual(QwenImage21References.rgba(prepared[1]).suffix(4), [1, 1, 1, 1], "an opaque picture's alpha is +1")
        let clear = ImageRGB(pixels: [Float](repeating: 1, count: 3 * 64 * 64), height: 64, width: 64,
                             rgba: [UInt8](repeating: 0, count: 4 * 64 * 64))
        let kept = QwenImage21References.prepared(clear, resolution: 64)
        XCTAssertEqual(kept.rgba?.count, 4 * 64 * 64)
        XCTAssertEqual(QwenImage21References.rgba(kept).last, -1, "a transparent pixel's alpha is −1, not composited")
        XCTAssertEqual(kept.pixels.first, 1, "the vision copy is over white")
    }

    /// **The three positions of the encoder's rows** (`get_rope_index`): text counts on the three axes, a picture
    /// freezes the first and counts rows and columns from it, then the position advances by the grid's larger side.
    func testThePicturesPositions() throws {
        let slots = [false, false] + [Bool](repeating: true, count: 6) + [false]
        let p = try Qwen3VLImages.rotaryPositions(slots: slots, grids: [(height: 4, width: 6)])
        XCTAssertTrue(p.map { [$0.0, $0.1, $0.2] } == [[0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 3], [2, 2, 4], [2, 3, 2], [2, 3, 3], [2, 3, 4], [5, 5, 5]])
        XCTAssertThrowsError(try Qwen3VLImages.rotaryPositions(slots: slots, grids: []))
        XCTAssertThrowsError(try Qwen3VLImages.rotaryPositions(slots: slots, grids: [(height: 4, width: 8)]))
        // Frequency 1 reads the height, 2 the width, 0 and those from 60 on the time.
        let table = Qwen3VLImages.rotaryTable([(0, 1, 2)], headDim: 128, theta: 5e6)
        XCTAssertEqual(table.cos[0], 1); XCTAssertEqual(table.cos[60], 1)
        XCTAssertNotEqual(table.cos[1], 1); XCTAssertNotEqual(table.cos[2], table.cos[1])
        XCTAssertEqual(Array(table.sin[..<64]), Array(table.sin[64...]))
    }

    /// `(z − mean) / std` in bf16, every operation rounded: the decoder's de-normalization, inverted.
    func testTheReferencesNormalization() {
        let x: Float = 1.25
        let out = QwenImage21VAEEncoder.normalized([Float](repeating: x, count: 64))
        for c in [0, 15, 63] {
            let std = BFloat.round(Float(QwenImage21Published.latentsStd[c])), mean = BFloat.round(Float(QwenImage21Published.latentsMean[c]))
            XCTAssertEqual(out[c], BFloat.round(BFloat.round(x - mean) / std))
        }
        XCTAssertEqual(QwenImage21Render.tokens([1, 2, 3, 4] + [Float](repeating: 0, count: 63 * 4)).prefix(65).filter { $0 != 0 }, [1, 2])
    }

    /// The template's one slot a picture becomes the picture's slots; a prompt that spells the slot is refused.
    func testASlotAPictureBecomesItsSlots() throws {
        XCTAssertTrue(QwenImage21Text.template(prompt: "x", images: 1).contains("user\n<image1><|vision_start|><|image_pad|><|vision_end|>x"))
    }
}
