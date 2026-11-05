import XCTest
@testable import Siliconed

/// **What a family adds to the general format rules** (`Format.check(width:height:family:)`), each
/// read in its reference: Qwen-Image-2.1's multiple of 32, Anima's longest side.
final class FormatFamilyTests: XCTestCase {

    /// Qwen-Image-2.1: VAE ×16, patch ×2 — a side of 16 that is not of 32 is refused, by name and
    /// with its multiple, before any computation (848×1216 used to fail after the encoder).
    func testQwenImage21WantsMultiplesOf32() {
        XCTAssertEqual(Format.multiple(for: .qwenImage21), 32)
        for (w, h) in [(848, 1216), (528, 512), (1008, 1008), (512, 528)] {
            XCTAssertNoThrow(try Format.check(width: w, height: h), "\(w)×\(h): the general rules take it")
            XCTAssertThrowsError(try Format.check(width: w, height: h, family: .qwenImage21)) {
                XCTAssertEqual($0 as? EngineError, .formatRefused(width: w, height: h, reason: .notMultiple, bound: 32))
            }
        }
        for (w, h) in [(512, 512), (1024, 1024), (832, 1216), (1024, 1536), (544, 1600), (2048, 2048), (2048, 3072)] {
            XCTAssertNoThrow(try Format.check(width: w, height: h, family: .qwenImage21), "\(w)×\(h)")
        }
    }

    /// The other families keep 16.
    func testOtherFamiliesKeep16() {
        for family in [Family.zImage, .anima, .krea2] {
            XCTAssertEqual(Format.multiple(for: family), 16)
            XCTAssertNoThrow(try Format.check(width: 848, height: 1216, family: family), "\(family)")
        }
    }

    /// Anima: the reference's RoPE table is `arange(max(max_size // patch_size))` — 128 positions with
    /// the published (128, 240, 240) / (1, 2, 2), the time axis' bound, not 240 // 2 — so 2048 px a side.
    func testAnimaLongestSideIsItsRopeTable() {
        XCTAssertEqual(AnimaDiT.Config.ropePositions(maxSize: [128, 240, 240], patchSize: [1, 2, 2]), 128)
        XCTAssertEqual(AnimaDiT.Config.publishedRopePositions, 128)
        XCTAssertEqual(Format.maximumSide(for: .anima), 2048)
        XCTAssertNoThrow(try Format.check(width: 2048, height: 768, family: .anima))
        XCTAssertNoThrow(try Format.check(width: 768, height: 2048, family: .anima))
        for (w, h) in [(2064, 752), (2560, 512), (512, 3072)] {
            XCTAssertNoThrow(try Format.check(width: w, height: h), "\(w)×\(h): the general rules take it")
            XCTAssertThrowsError(try Format.check(width: w, height: h, family: .anima)) {
                XCTAssertEqual($0 as? EngineError, .formatRefused(width: w, height: h, reason: .sideTooLong, bound: 2048))
            }
        }
        for family in [Family.zImage, .krea2, .qwenImage21] {
            XCTAssertEqual(Format.maximumSide(for: family), Format.maximumSide)
        }
    }

    /// **No area ceiling**: 2048², 2048×3072 and 3072×2048 pass the format rules of every family but
    /// Anima's side bound — B decides the rest (`FormatTableTests`). The longest side is Z-Image's RoPE
    /// table (512 positions of 16 px), and an absurd side is refused before anything is multiplied.
    func testNoAreaCeilingOnlySideBounds() {
        XCTAssertEqual(Format.maximumSide, 8192)
        XCTAssertEqual(Format.maximumSide, DiT.Config.publishedAxesLens[1] * 16)
        for family in [Family.zImage, .krea2, .qwenImage21] {
            for (w, h) in [(2048, 2048), (2048, 3072), (3072, 2048), (4096, 4096), (8192, 8192)] {
                XCTAssertNoThrow(try Format.check(width: w, height: h, family: family), "\(family) \(w)×\(h)")
            }
            XCTAssertThrowsError(try Format.check(width: 8224, height: 512, family: family)) {
                XCTAssertEqual($0 as? EngineError, .formatRefused(width: 8224, height: 512, reason: .sideTooLong, bound: 8192))
            }
        }
        XCTAssertNoThrow(try Format.check(width: 2048, height: 2048, family: .anima))
    }

    /// The map's header says the same as the constant the preflight uses (the forged Anima map's).
    func testAnimaConfigReadsItsTableFromTheHeader() {
        let header: [String: Any] = ["config": ["max_size": [128, 240, 240], "patch_size": [1, 2, 2]]]
        XCTAssertEqual(AnimaDiT.Config(header: header).ropePositions, AnimaDiT.Config.publishedRopePositions)
        XCTAssertEqual(AnimaDiT.Config(header: [:]).ropePositions, AnimaDiT.Config.publishedRopePositions)
    }

    /// Every recommended format and every edit output pass every family.
    func testOfferedFormatsPassEveryFamily() {
        for card in ModelCard.allCards {
            for f in card.formats {
                XCTAssertNoThrow(try Format.check(width: f.width, height: f.height, family: card.family), "\(card.id) \(f)")
            }
        }
        for (w, h) in [(1000, 1000), (848, 1216), (4000, 400), (400, 4000), (1920, 1080), (513, 1027), (3000, 2999)] {
            let f = QwenImage21Pipeline.outputSize(referenceWidth: w, referenceHeight: h)
            XCTAssertNoThrow(try Format.check(width: f.width, height: f.height, family: .qwenImage21), "\(w)×\(h) → \(f)")
        }
    }

    /// The sentence names the bound broken.
    func testSentencesCarryTheBound() {
        let multiple = EngineError.formatRefused(width: 848, height: 1216, reason: .notMultiple, bound: 32).errorDescription!
        XCTAssertTrue(multiple.contains("multiple of 32"), multiple)
        let side = EngineError.formatRefused(width: 2560, height: 512, reason: .sideTooLong, bound: 2048).errorDescription!
        XCTAssertTrue(side.contains("2048 px"), side)
        XCTAssertEqual(EngineError.formatRefused(width: 2560, height: 512, reason: .sideTooLong, bound: 2048).code,
                       "format_refused")
    }
}
