import XCTest
@testable import Siliconed

/// **img2img, without a byte of weights**: the strength (`t_start`), each denoiser's plan, the
/// refusal of a strength with no evaluation, the noise blending, the cropping and the encoder wiring.
final class Img2ImgTests: XCTestCase {

    /// These renders stop before any weight: the preflight's memory is not what they test, and the
    /// machine's state would make them depend on what else runs (`MemoryBudget`). A budget that fits all.
    override class func setUp() {
        super.setUp()
        setenv(MemoryBudget.forcingVariable, "64", 1)
    }

    /// The `get_timesteps` table at N = 8: the strength is quantized by 1/8.
    func testTheStartStepFollowsGetTimesteps() {
        let expected: [(Double, Int)] = [(1, 0), (0.9, 0), (0.8, 1), (0.75, 2), (0.7, 2), (0.6, 3), (0.5, 4),
                                        (0.4, 4), (0.3, 5), (0.25, 6), (0.2, 6), (0.1, 7), (0.05, 7)]
        for (strength, start) in expected {
            XCTAssertEqual(Strength.startStep(steps: 8, strength: strength), start, "strength \(strength)")
        }
    }

    /// **The floating-point edge case**: at N = 50 and s = 0.56, `N·s` is 28.000000000000004 in
    /// double — Python truncates to 21, not 22. In `Float`, we would land on 22.
    func testTheStartStepIsComputedInDouble() {
        XCTAssertEqual(Strength.startStep(steps: 50, strength: 0.56), 21)
        XCTAssertEqual(Int(Float(50) - min(Float(50) * Float(0.56), 50)), 22, "the trap that Double avoids")
    }

    /// At strength 0.6: the three do 5 evaluations (Z-Image too since 0.8.2: no final zero step), and
    /// σ_s differs from one model to the next (Z-Image and Anima share shift 3 on the same ramp:
    /// 0.833 · 0.833 · 0.840).
    func testEachModelsPlanAtStrength60() {
        let start = Strength.startStep(steps: 8, strength: 0.6)
        let z = ZImageDenoisingModule(map: "").plan(height: 64, width: 64, steps: 8, start: start)
        let a = AnimaDenoisingModule(map: "").plan(height: 64, width: 64, steps: 8, start: start)
        let k = Krea2DenoisingModule(map: "").plan(height: 64, width: 64, steps: 8, start: start)
        XCTAssertEqual([z.evaluations, a.evaluations, k.evaluations], [5, 5, 5])
        XCTAssertEqual(z.startSigma, 0.8333333, accuracy: 1e-6)
        XCTAssertEqual(a.startSigma, 0.8333, accuracy: 1e-4)
        XCTAssertEqual(k.startSigma, 0.8403, accuracy: 1e-4)
        XCTAssertEqual([z.startStep, a.startStep, k.startStep], [3, 3, 3])
    }

    /// txt2img is start 0: nothing has changed (8 evaluations, σ = 1).
    func testStartZeroIsTxt2img() {
        let z = ZImageDenoisingModule(map: "").plan(height: 64, width: 64, steps: 8, start: 0)
        XCTAssertEqual(z.evaluations, 8); XCTAssertEqual(z.startSigma, 1)
        XCTAssertEqual(AnimaDenoisingModule(map: "").plan(height: 64, width: 64, steps: 8, start: 0).evaluations, 8)
    }

    /// **Z-Image never evaluates at half size** (since 0.8.2): at 1024² (latent 128), from any start,
    /// with or without a LoRA, every evaluation is at the full grid.
    func testZImageNeverHalves() {
        let z = ZImageDenoisingModule(map: "")
        for start in [0, 1, 3] {
            for withLoRA in [false, true] {
                let plan = z.plan(height: 128, width: 128, steps: 8, start: start, withLoRA: withLoRA, faster: false)
                XCTAssertEqual(plan.reduced, 0)
                XCTAssertEqual([plan.reducedHeight, plan.reducedWidth], [128, 128])
            }
        }
    }

    /// **Every strength above 0 keeps an evaluation**: Z-Image's last step moves since 0.8.2 (it
    /// was 0 → 0, and a strength ≤ 1/N was refused), like Anima's. Out-of-range strengths and a
    /// chain without an encoder are refused before any computation — the modules have empty paths:
    /// if the engine touched a file, the error would be different.
    func testAStrengthWithoutEvaluationIsRefused() throws {
        let image = ImageRGB(pixels: [Float](repeating: 0, count: 3 * 512 * 512), height: 512, width: 512)
        let zImage = try Chain(text: ZImageTextModule(map: "", tokenizer: ""), denoising: ZImageDenoisingModule(map: ""),
                                decoding: FluxDecodingModule(path: ""), encoding: FluxEncodingModule(path: ""))
        for strength in [0.01, 0.1] {
            let start = Strength.startStep(steps: 8, strength: strength)
            XCTAssertEqual(start, 7)
            XCTAssertEqual(ZImageDenoisingModule(map: "").plan(height: 64, width: 64, steps: 8, start: start).evaluations, 1)
            XCTAssertEqual(AnimaDenoisingModule(map: "").plan(height: 64, width: 64, steps: 8, start: start).evaluations, 1)
        }
        // Out of bounds, and a chain without an encoder: refused too.
        XCTAssertThrowsError(try Engine().render(Request("une femme dans une bibliothèque", resolution: 512, image: image, strength: 1.5),
                                                 model: Model(card: .zImage, chain: zImage))) {
            XCTAssertEqual($0 as? EngineError, .strengthOutOfRange(strength: 1.5))
        }
        let withoutEncoder = try Chain(text: ZImageTextModule(map: "", tokenizer: ""),
                                      denoising: ZImageDenoisingModule(map: ""), decoding: FluxDecodingModule(path: ""))
        XCTAssertThrowsError(try Engine().render(Request("une femme dans une bibliothèque", resolution: 512, image: image),
                                                 model: Model(card: .zImage, chain: withoutEncoder))) {
            XCTAssertEqual($0 as? EngineError, .imageToImageUnsupported(model: "z-image"))
        }
    }

    /// `σ·ε + (1 − σ)·z₀`: at σ = 1 it is the noise BIT FOR BIT (txt2img), at σ = 0 the image.
    func testTheNoiseMix() {
        let z₀ = Latent(space: .flux, height: 1, width: 2, values: [Float](repeating: 0.25, count: 32))
        let ε = Latent.noise(.flux, height: 1, width: 2, seed: 42)
        XCTAssertEqual(Latent.start(image: z₀, noise: ε, sigma: 1).values, ε.values)
        XCTAssertEqual(Latent.start(image: z₀, noise: ε, sigma: 0).values, z₀.values)
        let milieu = Latent.start(image: z₀, noise: ε, sigma: 0.5).values
        for i in 0..<32 { XCTAssertEqual(milieu[i], 0.5 * ε.values[i] + 0.5 * 0.25, accuracy: 1e-7) }
    }

    /// **Fill, then crop at the center**: the proportions are kept, the format is covered.
    func testCenteredCropKeepsProportions() {
        let landscape = ImageRGB.crop(source: (1000, 500), target: (512, 512))
        XCTAssertEqual(landscape.width, 1024); XCTAssertEqual(landscape.height, 512)
        XCTAssertEqual(landscape.x0, 256); XCTAssertEqual(landscape.y0, 0)
        let portrait = ImageRGB.crop(source: (500, 1000), target: (832, 1216))
        XCTAssertEqual(portrait.width, 832); XCTAssertEqual(portrait.height, 1664)
        XCTAssertEqual(portrait.y0, 224); XCTAssertEqual(portrait.x0, 0)
        XCTAssertEqual(Double(portrait.width) / Double(portrait.height), 0.5, accuracy: 0.001)
        // Already at the right format: the image comes back bit-identical; a black left half and a
        // white right half, cropped to a square, keeps its two halves on either side of the center.
        let alreadyFitted = ImageRGB(pixels: (0..<48).map { (i: Int) -> Float in Float(i) / 48 }, height: 4, width: 4)
        XCTAssertEqual(alreadyFitted.fitted(width: 4, height: 4).pixels, alreadyFitted.pixels)
        let colonIndex = ImageRGB(pixels: (0..<3).flatMap { _ in (0..<(16 * 32)).map { $0 % 32 < 16 ? Float(-1) : 1 } },
                            height: 16, width: 32)
        let square = colonIndex.fitted(width: 16, height: 16)
        XCTAssertEqual(square.width, 16); XCTAssertEqual(square.height, 16)
        XCTAssertEqual(square.pixels[8 * 16 + 0], -1); XCTAssertEqual(square.pixels[8 * 16 + 15], 1)
        XCTAssertTrue(square.pixels.allSatisfy { $0 >= -1 && $0 <= 1 }, "Lanczos overshoots: clamped to [-1, 1]")
    }

    /// **Crop before enlarging** (B5): the source area of an extreme ratio stays small, and the
    /// fitted image comes out at the right format without ever holding the enlarged strip.
    func testAnExtremeRatioCropsWithinTheSource() {
        let zone = ImageRGB.zoneSource(source: (3072, 4), target: (512, 512))
        XCTAssertEqual(zone.height, 4); XCTAssertEqual(zone.width, 4)
        XCTAssertEqual(zone.x0, 1534); XCTAssertEqual(zone.y0, 0)
        let landscape = ImageRGB.zoneSource(source: (1000, 500), target: (512, 512))
        XCTAssertEqual(landscape.width, 500); XCTAssertEqual(landscape.height, 500)
        XCTAssertEqual(landscape.x0, 250); XCTAssertEqual(landscape.y0, 0)
        XCTAssertEqual(ImageRGB.zoneSource(source: (640, 960), target: (512, 768)).width, 640,
                       "at the same proportions, the area is the whole image")
        let band = ImageRGB(pixels: [Float](repeating: 0.5, count: 3 * 4 * 3072), height: 4, width: 3072)
        let square = band.fitted(width: 512, height: 512)
        XCTAssertEqual(square.width, 512); XCTAssertEqual(square.height, 512)
        XCTAssertEqual(square.pixels[256 * 512 + 256], 0.5, accuracy: 1e-5)
    }

    /// An encoder from a different space than the denoiser does not wire up; the three models do.
    func testTheEncoderWiringIsChecked() throws {
        XCTAssertThrowsError(try Chain(text: ZImageTextModule(map: "", tokenizer: ""),
                                        denoising: ZImageDenoisingModule(map: ""), decoding: FluxDecodingModule(path: ""),
                                        encoding: QwenImageEncodingModule(path: ""))) {
            guard case EngineError.incompatibleChain = $0 else { return XCTFail("\($0)") }
        }
        _ = try Chain(text: AnimaTextModule(map: "", tokenizerQwen: "", tokenizerT5: "", adapter: ""),
                       denoising: AnimaDenoisingModule(map: ""), decoding: QwenImageDecodingModule(path: ""),
                       encoding: QwenImageEncodingModule(path: ""))
        _ = try Chain(text: Krea2TextModule(map: "", tokenizer: "", sockets: []),
                       denoising: Krea2DenoisingModule(map: ""), decoding: QwenImageDecodingModule(path: ""),
                       encoding: QwenImageEncodingModule(path: ""))
    }
}
