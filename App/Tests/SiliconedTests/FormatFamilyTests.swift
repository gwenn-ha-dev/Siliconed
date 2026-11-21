import CryptoKit
import XCTest
@testable import Siliconed

/// **What a family adds to the general format rules** (`Format.check(width:height:family:)`), each
/// read in its reference: Qwen-Image-2.1's multiple of 32, Anima's longest side.
final class FormatFamilyTests: XCTestCase {

    /// Every family names, for each version it declares, both maps it reads (`Family.ditMap`, `encoderMap`).
    func testEveryFamilyDeclaresBothMapsForEachVersion() {
        for family in Family.allCases {
            XCTAssertFalse(family.variants.isEmpty, "\(family)")
            for v in family.variants { XCTAssertNotNil(family.traits.encoderMaps[v], "\(family) \(v)") }
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
        for family in [Family.zImage, .krea2] {
            XCTAssertEqual(Format.maximumSide(for: family), Format.maximumSide)
        }
    }

    /// **No area ceiling**: 2048², 2048×3072 and 3072×2048 pass the format rules of every family but
    /// Anima's side bound — B decides the rest (`FormatTableTests`). Z-Image's longest side is its RoPE table
    /// (512 positions of 16 px), the general one's; an absurd side is refused before anything is multiplied.
    func testNoAreaCeilingOnlySideBounds() {
        XCTAssertEqual(Format.maximumSide, 8192)
        XCTAssertEqual(Format.maximumSide(for: .zImage), ZImageDiT.Config.publishedAxisPositions * 16)
        for family in [Family.zImage, .krea2] {
            for (w, h) in [(2048, 2048), (2048, 3072), (3072, 2048), (4096, 4096), (8192, 8192)] {
                XCTAssertNoThrow(try Format.check(width: w, height: h, family: family), "\(family) \(w)×\(h)")
            }
            XCTAssertThrowsError(try Format.check(width: 8224, height: 512, family: family)) {
                XCTAssertEqual($0 as? EngineError, .formatRefused(width: 8224, height: 512, reason: .sideTooLong, bound: 8192))
            }
        }
        XCTAssertNoThrow(try Format.check(width: 2048, height: 2048, family: .anima))
    }

    /// **Krea's `model_index.json` is in the code, verbatim**: the literal hashes, as git hashes a blob, to the id the Hub
    /// lists for that file at the pinned revision — and what the engine reads of it is there (the twelve layers drawn).
    func testKrea2PublishedFileIsKreasToTheByte() {
        for (path, blob, text) in Krea2Published.blobs {
            let bytes = Data(text.utf8)
            var hasher = Insecure.SHA1()
            hasher.update(data: Data("blob \(bytes.count)\u{0}".utf8))
            hasher.update(data: bytes)
            XCTAssertEqual(hasher.finalize().map { String(format: "%02x", $0) }.joined(), blob, path)
        }
        XCTAssertEqual(Krea2Text.sockets, Array(stride(from: 2, through: 35, by: 3)))
    }

    /// The published configuration holds the constant the preflight uses.
    func testAnimaConfigHoldsThePublishedTable() {
        XCTAssertEqual(AnimaDiT.Config.published.ropePositions, AnimaDiT.Config.publishedRopePositions)
        XCTAssertEqual(AnimaDiT.Config.publishedRopePositions, 128)
    }

    /// Every recommended format and every edit output pass every family.
    func testOfferedFormatsPassEveryFamily() {
        for card in ModelCard.allCards {
            for f in card.formats {
                XCTAssertNoThrow(try Format.check(width: f.width, height: f.height, family: card.family), "\(card.id) \(f)")
            }
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
