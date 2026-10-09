import XCTest
@testable import Siliconed

/// **An edit, from the request to the DiT's sequence, without a byte of weights**: the output's
/// format, what is refused before any computation, the order of the images and their `<imageN>`.
/// The chains have empty paths: if the engine touched a file before refusing, the error would be
/// `fileMissing`, not the one expected (`RequestTests`' convention).
final class EditSequenceTests: XCTestCase {

    /// These renders stop before any weight: the preflight's memory is not what they test, and the
    /// machine's state would make them depend on what else runs (`MemoryBudget`). A budget that fits all.
    override class func setUp() {
        super.setUp()
        setenv(MemoryBudget.forcingVariable, "64", 1)
    }

    private let qwen = Model(card: .qwenImage21, chain: try! Chain(
        text: QwenImage21TextModule(map: "", tokenizer: ""),
        denoising: QwenImage21DenoisingModule(map: "", turbo: "", scheduler: .init()),
        decoding: QwenImage21DecodingModule(path: ""), encoding: QwenImage21EncodingModule(path: "")))
    private let klein = Model(card: .klein4b, chain: try! Chain(
        text: KleinTextModule(map: "", tokenizer: ""), denoising: KleinDenoisingModule(map: ""),
        decoding: Flux2DecodingModule(path: ""), encoding: Flux2EncodingModule(path: "")))

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
    /// `Format.maxSurface` — so `Format.check` always passes — and image 1’s orientation never flipped (a near-square may come out square).
    func testQwenEditFormatIsAlwaysRenderable() {
        let d = qwen.chain.denoising
        for (w, h) in sizes {
            let f = d.editFormat(referenceWidth: w, referenceHeight: h)
            XCTAssertGreaterThanOrEqual(min(f.width, f.height), Format.minimumSide, "\(w)×\(h) → \(f)")
            XCTAssertEqual(f.width % 32, 0, "\(w)×\(h) → \(f)"); XCTAssertEqual(f.height % 32, 0, "\(w)×\(h) → \(f)")
            XCTAssertLessThanOrEqual(f.width * f.height, Format.maxSurface, "\(w)×\(h) → \(f)")
            XCTAssertNoThrow(try Format.check(width: f.width, height: f.height), "\(w)×\(h) → \(f)")
            if f.width != f.height { XCTAssertEqual(f.width > f.height, w > h, "\(w)×\(h) → \(f): orientation flipped") }
        }
        // The test photo and the square: what the edit renders and the oracle used.
        XCTAssertTrue(d.editFormat(referenceWidth: 1216, referenceHeight: 832) == (1248, 832))
        XCTAssertTrue(d.editFormat(referenceWidth: 4032, referenceHeight: 3024) == (1184, 896))
        XCTAssertTrue(d.editFormat(referenceWidth: 1, referenceHeight: 1) == (1024, 1024))
    }

    /// **FLUX.2 [klein]'s rule** (the protocol's default): the reference's own size under 1024², at
    /// multiples of 16, each side raised to 512. It does not cap the area: beyond a ratio of about
    /// 9:1, raising the short side to 512 puts the format over `Format.maxSurface`, and the render
    /// is refused (`EngineError.formatRefused`, `.tooLarge`) instead of being given a smaller format.
    func testKleinEditFormat() throws {
        let d = klein.chain.denoising
        for (w, h) in sizes {
            let f = d.editFormat(referenceWidth: w, referenceHeight: h)
            XCTAssertGreaterThanOrEqual(min(f.width, f.height), Format.minimumSide, "\(w)×\(h) → \(f)")
            XCTAssertEqual(f.width % 16, 0); XCTAssertEqual(f.height % 16, 0)
            if max(w, h) <= 9 * min(w, h) {
                XCTAssertNoThrow(try Format.check(width: f.width, height: f.height), "\(w)×\(h) → \(f)")
            }
        }
        XCTAssertTrue(d.editFormat(referenceWidth: 1216, referenceHeight: 832) == (1216, 832))
        XCTAssertTrue(d.editFormat(referenceWidth: 3000, referenceHeight: 2000) == (1248, 832))
        let strip = d.editFormat(referenceWidth: 10_000, referenceHeight: 1000)
        XCTAssertNoThrow(try Format.check(width: strip.width, height: strip.height), "\(strip) for a 10:1 reference")
        XCTAssertEqual(strip.height, Format.minimumSide)
    }

    /// **The app's former copy (`EditFormat`, since removed) against the engine's rule.** The
    /// copy is kept here, frozen as it was at `879475b`, to say exactly where the two differed:
    ///
    /// - FLUX.2 [klein]: the same code — identical wherever the copy respected the area cap; past
    ///   ~6:1 it did not (10:1 gave 3232×512, refused by `Format`), and the engine now caps it;
    /// - Qwen-Image-2.1: identical as long as both sides of `calculate_dimensions(1024², ratio)`
    ///   reach 512, i.e. up to a ratio of about 4.2:1. Beyond, the copy raised the short side to 512
    ///   **alone** (the ratio lost), the engine raises it keeping the ratio, then caps the area:
    ///   a 5:1 reference gave 2304×512 in the app and gives 2560×512 now. The app and the CLI
    ///   therefore rendered different formats for those references; they no longer can.
    func testTheAppsCopyMatchedTheEngineUpToAboutFourToOne() {
        func h1Qwen(_ width: Int, _ height: Int) -> (width: Int, height: Int) {
            let resolution = 1024, ratio = Double(width) / Double(height)
            let w0 = (Double(resolution * resolution) * ratio).squareRoot(), h0 = w0 / ratio
            func side(_ v: Double) -> Int { max(Format.minimumSide, Int((v / 32).rounded(.toNearestOrEven)) * 32) }
            var w = side(w0), h = side(h0)
            if w * h > Format.maxSurface {
                if w >= h { w = Format.maxSurface / h / 32 * 32 } else { h = Format.maxSurface / w / 32 * 32 }
            }
            return (w, h)
        }
        func h1Klein(_ width: Int, _ height: Int) -> (width: Int, height: Int) {
            let scale = min(1, (Double(1024 * 1024) / Double(width * height)).squareRoot())
            let w = max(64, Int(Double(width) * scale) / 16 * 16), h = max(64, Int(Double(height) * scale) / 16 * 16)
            return (max(Format.minimumSide, w), max(Format.minimumSide, h))
        }
        var differing: [String] = []
        for (w, h) in sizes {
            let kleinNow = klein.chain.denoising.editFormat(referenceWidth: w, referenceHeight: h), kleinThen = h1Klein(w, h)
            if kleinThen.width * kleinThen.height <= Format.maxSurface {
                XCTAssertTrue(kleinNow == kleinThen, "\(w)×\(h)")
            } else {   // past ~6:1 the old rule broke the area cap; the engine's long side now gives way
                XCTAssertNoThrow(try Format.check(width: kleinNow.width, height: kleinNow.height), "\(w)×\(h)")
            }
            let engine = qwen.chain.denoising.editFormat(referenceWidth: w, referenceHeight: h), copy = h1Qwen(w, h)
            let c = Qwen3VLImages.conditionSize(width: w, height: h, resolution: 1024)
            if min(c.width, c.height) >= Format.minimumSide {
                XCTAssertTrue(engine == copy, "\(w)×\(h): engine \(engine), app H1 \(copy)")
            } else if engine != copy {
                differing.append("\(w)×\(h)")
            }
        }
        XCTAssertFalse(differing.isEmpty, "the documented gap must show on the sweep's elongated sizes")
        XCTAssertTrue(qwen.chain.denoising.editFormat(referenceWidth: 5000, referenceHeight: 1000) == (2560, 512))
        XCTAssertTrue(h1Qwen(5000, 1000) == (2304, 512))
    }

    // ── what is refused before any computation ──────────────────────────────────────────────

    /// Three references pass the count (the request then stops on its 8 steps, which Viggle's turbo
    /// has no schedule for — still before any file); a fourth is refused as such.
    func testAtMostThreeReferences() {
        XCTAssertEqual(qwen.chain.denoising.maxReferences, 3)
        let reference = image(64, 64)
        var r = Request("replace the cloudy sky with a blue sky", resolution: 512, steps: 8)
        r.references = Array(repeating: reference, count: 3)
        XCTAssertThrowsError(try Engine().render(r, model: qwen)) {
            XCTAssertEqual($0 as? EngineError, .invalidSteps(steps: 8, allowed: [5, 6, 7, 9]), "\($0)")
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
        // The steps Viggle publishes pass the module's own check; another count does not.
        for n in [5, 6, 7, 9] { XCTAssertNoThrow(try qwen.chain.denoising.check(steps: n, startImage: false), "\(n)") }
        XCTAssertThrowsError(try qwen.chain.denoising.check(steps: 8, startImage: false))
    }

    /// What the chain declares: the text, the references, and the encoder that sees them. ⚠️ It also
    /// declares `.image` (img2img), because it has an image encoder (the references need it) — and
    /// the denoiser refuses img2img. The app ignores `.image`; another app would offer a field
    /// that always fails.
    func testTheChainDeclaresTheEdit() {
        XCTAssertTrue(qwen.chain.text.output.readsImages)
        XCTAssertTrue(qwen.chain.text is any ImageTextModule)
        XCTAssertFalse(klein.chain.text.output.readsImages)
        XCTAssertTrue(qwen.chain.entries.contains { $0.kind == .reference })
        XCTAssertFalse(qwen.chain.entries.contains { $0.kind == .image })
    }

    // ── the order of the images ─────────────────────────────────────────────────────────────

    /// **The order is the numbering, at each stage.** The references come out of
    /// `preparedReferences` in the request's order, each at its own ratio (at `R = 128` here, so the
    /// resampling stays small); the template numbers them `<image1>`, `<image2>`, `<image3>` in that
    /// order; the DiT's sequence puts their blocks in that order, the target last.
    func testTheOrderOfTheImagesIsKeptEndToEnd() throws {
        let module = QwenImage21DenoisingModule(map: "", turbo: "", scheduler: .init(), resolution: 128)
        let red = image(48, 32, rgb: (1, -1, -1)), green = image(32, 32, rgb: (-1, 1, -1)), blue = image(32, 64, rgb: (-1, -1, 1))
        let prepared = try module.preparedReferences([red, green, blue])
        XCTAssertEqual(prepared.map { [$0.width, $0.height] }, [[160, 96], [128, 128], [96, 192]])
        for (i, channel) in [0, 1, 2].enumerated() {
            let p = prepared[i], plane = p.width * p.height, centre = (p.height / 2) * p.width + p.width / 2
            XCTAssertEqual(p.pixels[channel * plane + centre], 1, accuracy: 1e-6, "image \(i + 1) is not where it was put")
        }

        let template = Qwen3VLPrompt.template(prompt: "put the dog of image 2 next to the woman in image 1", images: 3)
        let marks = (1...3).map { template.range(of: "<image\($0)><|vision_start|><|image_pad|><|vision_end|>")!.lowerBound }
        XCTAssertEqual(marks, marks.sorted())
        XCTAssertNil(template.range(of: "<image4>"))
        XCTAssertLessThan(marks[2], template.range(of: "put the dog")!.lowerBound, "the images come before the prompt")
        // Each `<|image_pad|>` expands to ITS image's count, in order.
        XCTAssertEqual(try Qwen3VLPrompt.expand([1, 9, 2, 9, 3, 9, 4], image: 9, counts: [1, 2, 3]),
                       [1, 9, 2, 9, 9, 3, 9, 9, 9, 4])

        // Three references (4×6, 4×4, 6×4 latent cells: 6, 4, 6 slots), then a 4×4 target (4 slots).
        typealias Grid = QwenImage21Sequence.Grid
        let t = false, s = true
        let slots = [t, t, t] + [Bool](repeating: s, count: 6) + [t] + [Bool](repeating: s, count: 4) + [t]
            + [Bool](repeating: s, count: 6) + [t, t, t, t, t] + [Bool](repeating: s, count: 4)
        let grids = [Grid(height: 4, width: 6), Grid(height: 4, width: 4), Grid(height: 6, width: 4), Grid(height: 4, width: 4)]
        let sequence = try QwenImage21Sequence(slots: slots, images: grids)
        // 3 text, 24 rows (image 1), 1 text, 16 (image 2), 1 text, 24 (image 3), 5 text, then the target.
        XCTAssertEqual(sequence.blockStarts, [3, 28, 45, 74])
        for (i, start) in sequence.blockStarts.enumerated() {
            XCTAssertEqual(sequence.imageIds[start], i)
            XCTAssertEqual(sequence.imageIds[start + grids[i].height * grids[i].width - 1], i)
        }
        XCTAssertEqual(sequence.target, 16)
        XCTAssertEqual(Array(sequence.imageIds.suffix(16)), [Int](repeating: 3, count: 16))
        XCTAssertEqual(sequence.prefix, 3 + 24 + 1 + 16 + 1 + 24 + 5)
    }
}
