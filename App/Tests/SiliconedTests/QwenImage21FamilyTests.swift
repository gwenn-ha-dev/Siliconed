import CryptoKit
import Foundation
import XCTest
@testable import Siliconed

/// Qwen-Image-2.1 without a map nor a reference: what the family declares, and the layout of its sequence.
final class QwenImage21FamilyTests: XCTestCase {
    /// **Qwen's `model_index.json` and `vae/config.json` are in the code, verbatim**: the literals hash, as git hashes a
    /// blob, to the ids the Hub lists at the pinned revision — and what the engine reads of them is there.
    func testThePublishedFilesAreQwensToTheByte() {
        for (path, blob, text) in QwenImage21Published.blobs {
            let bytes = Data(text.utf8)
            var hasher = Insecure.SHA1()
            hasher.update(data: Data("blob \(bytes.count)\u{0}".utf8))
            hasher.update(data: bytes)
            XCTAssertEqual(hasher.finalize().map { String(format: "%02x", $0) }.joined(), blob, path)
        }
        XCTAssertEqual(QwenImage21Published.sampleSigmas, [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568])
        XCTAssertEqual(QwenImage21Published.latentsMean.count, 64)
        XCTAssertEqual(QwenImage21Published.latentsStd.count, 64)
    }

    /// The schedule is the published one whatever is asked, and only its eight steps are accepted; no img2img.
    func testTheScheduleIsThePublishedOne() {
        let d = QwenImage21DenoisingModule(map: "")
        XCTAssertEqual(d.defaultSteps, 8)
        XCTAssertEqual(d.sigmas(steps: 8).count, 9)
        XCTAssertEqual(d.sigmas(steps: 8).last, 0)
        XCTAssertEqual(d.sigmas(steps: 8)[1], Float(0.978453))
        XCTAssertNoThrow(try d.check(steps: 8, startImage: false))
        for steps in [1, 4, 7, 9, 20] {
            XCTAssertThrowsError(try d.check(steps: steps, startImage: false)) {
                XCTAssertEqual($0 as? EngineError, .invalidSteps(steps: steps, allowed: [8]))
            }
        }
        XCTAssertThrowsError(try d.check(steps: 8, startImage: true))
        XCTAssertFalse(d.acceptsStartImage)
        XCTAssertEqual(d.plan(height: 64, width: 64, steps: 8, start: 0).evaluations, 8)
    }

    /// Each side a multiple of 32: the pipeline floors it without a word, the app refuses it.
    func testASideIsAMultipleOf32() {
        XCTAssertEqual(Format.multiple(for: .qwenImage21), 32)
        XCTAssertNoThrow(try Format.check(width: 1024, height: 1024, family: .qwenImage21))
        XCTAssertThrowsError(try Format.check(width: 848, height: 1216, family: .qwenImage21))
        for f in ModelCard.qwenImage21.formats {
            XCTAssertNoThrow(try Format.check(width: f.width, height: f.height, family: .qwenImage21), "\(f)")
        }
    }

    /// The template is the pipeline's raw string, the system turn first; an empty prompt is a space.
    func testTheTemplate() {
        XCTAssertEqual(QwenImage21Text.template(prompt: "a cat"),
                       "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n")
        XCTAssertTrue(QwenImage21Text.template(prompt: "").contains("user\n <|im_end|>"))
        XCTAssertTrue(QwenImage21Text.template(prompt: "x", images: 2)
            .contains("user\n<image1><|vision_start|><|image_pad|><|vision_end|> <image2><|vision_start|><|image_pad|><|vision_end|>x"))
    }

    /// The sequence: the text, then the target, each a segment; with references, the oracle's own layout
    /// (`goldens-qwen21-dit-edit1-512`: 8 text rows, a 26×38 picture, 25 text rows, the 26×38 target).
    func testTheSequencesLayout() throws {
        let plain = QwenImage21DiT.Sequence(text: 19, height: 32, width: 32)
        XCTAssertEqual(plain.rows, 19 + 1024)
        XCTAssertEqual(plain.segments.map { [$0.start, $0.end] }, [[0, 19]])
        let slots = [Bool](repeating: false, count: 8) + [Bool](repeating: true, count: 26 * 38 / 4) + [Bool](repeating: false, count: 25)
        let edit = try QwenImage21DiT.Sequence(slots: slots, references: [(height: 26, width: 38)], height: 26, width: 38)
        XCTAssertEqual(edit.segments.map { [$0.start, $0.end] }, [[0, 8], [8, 996], [996, 1021]])
        XCTAssertEqual(edit.segments.map(\.isText), [true, false, true])
        XCTAssertEqual(edit.rows, 1021 + 988)
        XCTAssertEqual(edit.positions.count, edit.rows)
    }

    /// `latents · std + mean` in bf16, every operation rounded.
    func testTheDecodersDenormalizationIsPerChannelInBf16() {
        let x: Float = 0.3337
        let out = QwenImage21VAEDecoder.denormalized([Float](repeating: x, count: 64))
        for c in [0, 15, 63] {
            let std = BFloat.round(Float(QwenImage21Published.latentsStd[c])), mean = BFloat.round(Float(QwenImage21Published.latentsMean[c]))
            XCTAssertEqual(out[c], BFloat.round(BFloat.round(BFloat.round(x) * std) + mean))
        }
    }

    /// The time the DiT is told: `σ · 1000` in bf16, then `/ 1000` in bf16.
    func testTheTimeTheDiTIsToldIsThePipelinesBf16() {
        XCTAssertEqual(QwenImage21Transformer.modelTime(sigma: 1), 1)
        let sigma: Float = 0.978453
        XCTAssertEqual(QwenImage21Transformer.modelTime(sigma: sigma), BFloat.round(BFloat.round(sigma * 1000) / 1000))
        XCTAssertEqual(QwenImage21Render.modelTime(sigma: sigma, solver: .dpmpp2m, first: false), BFloat.round(Float(978) / 1000))
    }
}
