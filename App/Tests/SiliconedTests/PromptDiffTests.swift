import XCTest
@testable import Siliconed

/// **The Prompt Lab's word diff** (`PromptDiff`) — pure text, no weight.
final class PromptDiffTests: XCTestCase {
    private func p(_ old: String, _ new: String) -> [PromptDiff.Piece] { PromptDiff.pieces(from: old, to: new) }
    private func kept(_ pieces: [PromptDiff.Piece]) -> String { pieces.filter { $0.kind != .removed }.map(\.text).joined() }
    private func texts(_ pieces: [PromptDiff.Piece], _ kind: PromptDiff.Kind) -> [String] { pieces.filter { $0.kind == kind }.map(\.text) }

    func testIdenticalTextsAreOnePiece() {
        XCTAssertEqual(p("a red car", "a red car"), [.init(.same, "a red car")])
    }

    func testBothEmptyIsNothing() { XCTAssertEqual(p("", ""), []) }

    func testEmptyOldIsAllAdded() {
        XCTAssertEqual(p("", "une voiture"), [.init(.added, "une voiture")])
    }

    func testEmptyNewIsAllRemoved() {
        XCTAssertEqual(p("une voiture", ""), [.init(.removed, "une voiture")])
    }

    func testPureInsertionKeepsEverythingElseSame() {
        let r = p("a car in the rain", "a red car in the rain")
        XCTAssertEqual(r, [.init(.same, "a"), .init(.added, " red"), .init(.same, " car in the rain")])
    }

    func testPureDeletion() {
        let r = p("a red car", "a car")
        XCTAssertEqual(r, [.init(.same, "a"), .init(.removed, " red"), .init(.same, " car")])
    }

    func testReplacementShowsRemovalBeforeAddition() {
        let r = p("a red car", "a blue car")
        XCTAssertEqual(r, [.init(.same, "a"), .init(.removed, " red"), .init(.added, " blue"), .init(.same, " car")])
    }

    func testPunctuationIsItsOwnUnit() {
        let r = p("rouge, vif", "rouge vif")
        XCTAssertEqual(texts(r, .removed), [","])
        XCTAssertTrue(texts(r, .added).isEmpty)
        XCTAssertEqual(kept(r), "rouge vif")
    }

    func testSpacingIsTheNewTexts() {
        let r = p("a  red\ncar", "a red  car ")
        XCTAssertEqual(kept(r), "a red  car ")
    }

    func testFrenchAccentsAndElision() {
        let r = p("une femme à l’intérieur d’une bibliothèque", "une femme à l’intérieur d’une grande bibliothèque ancienne")
        XCTAssertEqual(texts(r, .added), [" grande", " ancienne"])
        XCTAssertTrue(texts(r, .removed).isEmpty)
        // « l’image » is one word: a changed ending marks the whole word.
        let s = p("l’image nette", "l’images nette")
        XCTAssertEqual(texts(s, .removed), ["l’image"])
        XCTAssertEqual(texts(s, .added), ["l’images"])
    }

    func testDecomposedAndComposedAccentsAreTheSameWord() {
        let composed = "caf\u{E9} noir", decomposed = "cafe\u{301} noir"
        XCTAssertEqual(p(composed, decomposed), [.init(.same, decomposed)])
    }

    func testJapaneseIsDiffedByCharacter() {
        let r = p("赤い車が走る", "青い車が走る")
        XCTAssertEqual(texts(r, .removed), ["赤"])
        XCTAssertEqual(texts(r, .added), ["青"])
        XCTAssertEqual(kept(r), "青い車が走る")
        let kana = p("ありがとう", "ありがとうございます")
        XCTAssertEqual(texts(kana, .added), ["ございます"])
    }

    func testChineseIsDiffedByCharacter() {
        let r = p("一位女士站在图书馆里。", "一位年轻女士站在古老的图书馆里。")
        XCTAssertEqual(texts(r, .added), ["年轻", "古老的"])
        XCTAssertTrue(texts(r, .removed).isEmpty)
        XCTAssertEqual(kept(r), "一位年轻女士站在古老的图书馆里。")
    }

    func testEmojiStayWhole() {
        let family = "👨‍👩‍👧", flag = "🇫🇷"
        let r = p("a \(family) smiling", "a \(flag) smiling")
        XCTAssertEqual(texts(r, .removed), [" \(family)"])
        XCTAssertEqual(texts(r, .added), [" \(flag)"])
    }

    func testReorderingShowsMovedWords() {
        let r = p("red car blue house", "blue house red car")
        XCTAssertEqual(kept(r), "blue house red car")
        // Nothing is lost: what is neither kept nor added was removed from the old text.
        let oldWords = (texts(r, .same) + texts(r, .removed)).joined(separator: " ").split(separator: " ").sorted()
        XCTAssertEqual(oldWords.count, 4)
        XCTAssertFalse(texts(r, .removed).isEmpty)
        XCTAssertFalse(texts(r, .added).isEmpty)
    }

    func testKeptPiecesAlwaysRebuildTheNewText() {
        let pairs = [("woman posing in a library", "A woman, posing in a vast old library, soft light"),
                     ("", "x"), ("x", ""), ("a b c", "c b a"), ("日本語のテキスト", "日本語の文章です")]
        for (old, new) in pairs { XCTAssertEqual(kept(p(old, new)), new, "\(old) → \(new)") }
    }
}
