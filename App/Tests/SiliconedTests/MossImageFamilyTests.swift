import XCTest
@testable import Siliconed
@testable import Verification

/// **MOSS-Image as a family, without a byte of weights**: its variants, the names its maps have, the chain's wires
/// (a wrong module is refused before any computation) and its computed need — the encoder streamed, the DMD LoRA
/// always charged, the decoder's heaps.
final class MossImageFamilyTests: XCTestCase {

    // MARK: - what the family declares

    /// A Standard and a Compact.
    func testAStandardAndACompact() {
        XCTAssertEqual(Family.mossImage.variants, [.standard, .compact])
        XCTAssertNotEqual(Family.mossImage.ditMap(.compact), Family.mossImage.ditMap(.standard))
        XCTAssertNotEqual(Family.mossImage.encoderMap(.compact), Family.mossImage.encoderMap(.standard))
    }

    /// The names the stages and the engine agree on.
    func testTheMapsHaveOneName() {
        XCTAssertEqual(Family.mossImage.ditMap(.standard), MossImageStages.ditMap)
        XCTAssertEqual(Family.mossImage.ditKind, MossImageDiT.kind)
        XCTAssertEqual(Family.mossImage.encoderMap(.standard), "qwen3.5_9b_bf16.safetensors")
        XCTAssertEqual(MossImageStages.model, ModelCard.mossImage.id)
        XCTAssertEqual(MossImageDenoisingModule(map: "", turbo: "").model, MossImageStages.model)
    }

    func testTheLicensesAreTheModelsAndTheLoRAs() {
        let urls = ModelCard.mossImage.license.urls.map(\.absoluteString)
        XCTAssertEqual(urls.count, 2)
        XCTAssertTrue(urls[0].contains("OpenMOSS-Team/MOSS-Image-SFT/blob/f972bd4b4b9e3852133cc5da954d87bc32cd8fb9/LICENSE"), urls[0])
        XCTAssertTrue(urls[1].contains("lodestones/MOSS-Image-SFT-8-steps-dmd/blob/f95df83b5e90ae91ea412434b3f9b7f1c8c6314f/README.md"), urls[1])
        XCTAssertTrue(ModelCard.mossImage.license.commercial)
        XCTAssertFalse(ModelCard.mossImage.license.requiredFilter)
        XCTAssertEqual(ModelCard.of(.mossImage), ModelCard.mossImage)
        XCTAssertEqual(ModelCard.withID("moss-image"), ModelCard.mossImage)
    }

    // MARK: - the chain

    func testTheChainWiresAndRefusesTheWrongOnes() throws {
        let text = MossImageTextModule(map: "", tokenizer: ""), dit = MossImageDenoisingModule(map: "", turbo: "")
        let decoder = Flux2DecodingModule(path: "")
        XCTAssertNoThrow(try Chain(text: text, denoising: dit, decoding: decoder))
        XCTAssertEqual(text.output, dit.entry)
        XCTAssertEqual(dit.space, decoder.space)
        XCTAssertEqual(dit.space.channels, 128)
        XCTAssertEqual(dit.space.factor, 16)
        // A text for another DiT, a decoder for another latent: refused at the wiring, not after a minute of encoder.
        XCTAssertThrowsError(try Chain(text: ZImageTextModule(map: "", tokenizer: ""), denoising: dit, decoding: decoder))
        XCTAssertThrowsError(try Chain(text: text, denoising: Krea2DenoisingModule(map: ""), decoding: decoder))
        XCTAssertThrowsError(try Chain(text: text, denoising: dit, decoding: QwenImageDecodingModule(path: "")))
        XCTAssertThrowsError(try Chain(text: text, denoising: dit, decoding: FluxDecodingModule(path: "")))
        // Without its encoder, no image entry; with FLUX.2's, a start image (img2img), and no reference.
        XCTAssertEqual(try Chain(text: text, denoising: dit, decoding: decoder).entries.map(\.kind), [.text])
        let encoder = Flux2EncodingModule(path: "")
        XCTAssertEqual(encoder.space, dit.space)
        XCTAssertEqual(try Chain(text: text, denoising: dit, decoding: decoder, encoding: encoder).entries.map(\.kind), [.text, .image])
    }

    func testTheRecipeIsEightStepsOfEulerWithoutCFG() {
        let dit = MossImageDenoisingModule(map: "", turbo: "")
        XCTAssertEqual(dit.defaultSteps, 8)
        XCTAssertEqual(ModelCard.mossImage.defaultSteps, 8)
        XCTAssertEqual(dit.sigmas(steps: 8), [1, 0.968, 0.926, 0.871, 0.791, 0.671, 0.46300002932548523, 0.024, 0])
        XCTAssertEqual(dit.solvers, [.euler])
        XCTAssertTrue(dit.acceptsStartImage)
        XCTAssertEqual(dit.maxReferences, 0)
        XCTAssertEqual(dit.plan(height: 64, width: 64, steps: 8, start: 0).evaluations, 8)
        // A dense schedule has fewer evaluations than steps (the σ are snapped to the training grid, then deduplicated).
        XCTAssertEqual(dit.plan(height: 64, width: 64, steps: 250, start: 0).evaluations, 246)
    }

    /// 512² and up, on a multiple of 16 — the DiT's cell.
    func testTheFormatRules() throws {
        XCTAssertNoThrow(try Format.check(width: 1024, height: 1024, family: .mossImage))
        XCTAssertNoThrow(try Format.check(width: 832, height: 1216, family: .mossImage))
        XCTAssertThrowsError(try Format.check(width: 1000, height: 1024, family: .mossImage))
        XCTAssertThrowsError(try Format.check(width: 256, height: 256, family: .mossImage))
    }

    func testTheTemplateIsTheChatTemplateWithoutASystemTurn() {
        XCTAssertEqual(Qwen35Prompt.template("a cat"), "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n")
    }

    /// **The strength's N is the schedule's length** (`len(timesteps)` in diffusers' `get_timesteps`): MOSS-Image snaps
    /// its σ to the training grid and merges the duplicates, so a dense schedule has fewer evaluations than steps, and
    /// t_start counts in evaluations, never past the last one.
    func testTheStrengthCountsTheScheduleNotTheSteps() throws {
        let dit = MossImageDenoisingModule(map: "", turbo: "")
        let chain = try Chain(text: MossImageTextModule(map: "", tokenizer: ""), denoising: dit,
                              decoding: Flux2DecodingModule(path: ""), encoding: Flux2EncodingModule(path: ""))
        let picture = ImageRGB(pixels: [Float](repeating: 0, count: 3 * 512 * 512), height: 512, width: 512)
        let eight = try Admission.judge(Request("p", resolution: 512, steps: 8, image: picture, strength: 0.6), chain: chain)
        XCTAssertEqual(eight.start, 3)
        XCTAssertEqual(eight.plan.evaluations, 5)
        XCTAssertEqual(eight.plan.startSigma, dit.sigmas(steps: 8)[3])
        for steps in 1...Request.maximumSteps {
            let n = dit.sigmas(steps: steps).count - 1
            let admitted = try Admission.judge(Request("p", resolution: 512, steps: steps, image: picture, strength: 0.5),
                                               chain: chain)
            XCTAssertEqual(admitted.start, Strength.startStep(steps: n, strength: 0.5), "\(steps) steps")
            XCTAssertEqual(admitted.plan.evaluations, n - admitted.start, "\(steps) steps")
        }
    }
}
