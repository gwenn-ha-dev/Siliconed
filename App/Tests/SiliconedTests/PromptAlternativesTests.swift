import XCTest
@testable import Siliconed

/// **`{a|b}` in a prompt** — the parser and its expansion, pure: what the app and `silicontrol add`
/// queue, one render per prompt.
final class PromptAlternativesTests: XCTestCase {

    private func prompts(_ p: String) throws -> [String] { try PromptAlternatives(p).prompts }

    private func refusal(_ p: String, file: StaticString = #filePath, line: UInt = #line) -> EngineError? {
        do { _ = try PromptAlternatives(p); XCTFail("accepted: \(p)", file: file, line: line); return nil }
        catch { return error }
    }

    /// A prompt without a group is itself, character for character — `|` included, since outside
    /// a group it is an ordinary character.
    func testAPromptWithoutGroupIsUnchanged() throws {
        for p in ["woman posing in a library", "", "  spaces  kept  ", "a | b", "tags: a, b | c\nnew line",
                  "émoji 📚 and accents", "a backslash \\ alone, \\n stays"] {
            let a = try PromptAlternatives(p)
            XCTAssertEqual(a.prompts, [p])
            XCTAssertEqual(a.count, 1)
            XCTAssertFalse(a.hasAlternatives)
            XCTAssertEqual(a.variants.first?.choices, [])
        }
    }

    func testOneGroupGivesOnePromptPerAlternative() throws {
        let a = try PromptAlternatives("woman posing in a {library|greenhouse|train station}")
        XCTAssertEqual(a.prompts, ["woman posing in a library", "woman posing in a greenhouse",
                                   "woman posing in a train station"])
        XCTAssertEqual(a.groups, [["library", "greenhouse", "train station"]])
        XCTAssertEqual(a.variants.map(\.choices), [["library"], ["greenhouse"], ["train station"]])
    }

    /// Several groups: the cartesian product, the first group varying the slowest.
    func testSeveralGroupsMakeTheProductInReadingOrder() throws {
        XCTAssertEqual(try prompts("{a|b} x {c|d}"), ["a x c", "a x d", "b x c", "b x d"])
        let a = try PromptAlternatives("{1|2|3}{x|y}")
        XCTAssertEqual(a.count, 6)
        XCTAssertEqual(a.prompts, ["1x", "1y", "2x", "2y", "3x", "3y"])
        XCTAssertEqual(a.variants.map(\.choices), [["1", "x"], ["1", "y"], ["2", "x"], ["2", "y"], ["3", "x"], ["3", "y"]])
    }

    /// An empty alternative means « or nothing »; nothing is trimmed.
    func testAnEmptyAlternativeAndSpacesAreKept() throws {
        XCTAssertEqual(try prompts("{|red }dress"), ["dress", "red dress"])
        XCTAssertEqual(try PromptAlternatives("{|red }dress").variants.map(\.choices), [[""], ["red "]])
        XCTAssertEqual(try prompts("{a | b}"), ["a ", " b"])
        XCTAssertEqual(try prompts("x{}y"), ["xy"])
        XCTAssertEqual(try prompts("{one}"), ["one"])
        XCTAssertTrue(try PromptAlternatives("{one}").hasAlternatives)
    }

    func testEscapesGiveTheCharacters() throws {
        XCTAssertEqual(try prompts(#"a \{b\} c"#), ["a {b} c"])
        XCTAssertEqual(try prompts(#"{x\|y|z}"#), ["x|y", "z"])
        XCTAssertEqual(try prompts(#"{\{|\}}"#), ["{", "}"])
        XCTAssertEqual(try prompts(#"pipe \| outside"#), ["pipe | outside"])
        // Any other backslash stays, and a trailing one too.
        XCTAssertEqual(try prompts(#"\n \\ end\"#), [#"\n \\ end\"#])
    }

    func testSyntaxErrorsNameTheBrace() {
        XCTAssertEqual(refusal("a {b|c"), .promptSyntax(position: 3, reason: .unclosedGroup))
        XCTAssertEqual(refusal("a b|c}"), .promptSyntax(position: 6, reason: .unmatchedClose))
        XCTAssertEqual(refusal("{a|b}}"), .promptSyntax(position: 6, reason: .unmatchedClose))
        XCTAssertEqual(refusal("{a|{b|c}}"), .promptSyntax(position: 4, reason: .nestedGroup))
        // Positions count characters as a person sees them, an escape being two.
        XCTAssertEqual(refusal("📚 \\{ {"), .promptSyntax(position: 6, reason: .unclosedGroup))
        XCTAssertEqual(EngineError.promptSyntax(position: 1, reason: .nestedGroup).code, "prompt_syntax")
        XCTAssertEqual(EngineError.PromptRefusal.allCases.map(\.rawValue), ["unclosed_group", "unmatched_close", "nested_group"])
    }

    /// 64 prompts pass, 65 or more are refused with their count — before any is built.
    func testTheCeilingIsSixtyFour() throws {
        XCTAssertEqual(PromptAlternatives.maxCombinations, 64)
        let sixtyFour = String(repeating: "{a|b}", count: 6)
        XCTAssertEqual(try PromptAlternatives(sixtyFour).prompts.count, 64)
        XCTAssertEqual(refusal(String(repeating: "{a|b}", count: 7)), .tooManyPromptVariants(count: 128, max: 64))
        XCTAssertEqual(refusal("{a|b|c|d|e|f|g|h|i|j|k|l|m}{x|y|z|w|v}"), .tooManyPromptVariants(count: 65, max: 64))
        // The count saturates instead of overflowing.
        if case .tooManyPromptVariants(let n, _)? = refusal(String(repeating: "{a|b}", count: 70)) {
            XCTAssertEqual(n, .max)
        } else { XCTFail() }
        // A caller (a grid) may ask for another ceiling.
        XCTAssertThrowsError(try PromptAlternatives("{a|b|c}", limit: 2))
        XCTAssertEqual(try PromptAlternatives("{a|b|c}", limit: 3).count, 3)
    }

    /// An expanded prompt reopened in the form parses back to itself, braces included.
    func testEscapingRoundTrips() throws {
        for p in ["woman posing in a library", "a {b} c", "}{", "x | y", "{a|b}"] {
            XCTAssertEqual(try PromptAlternatives(PromptAlternatives.escaping(p)).prompts, [p], p)
        }
        XCTAssertEqual(PromptAlternatives.escaping("no braces | here"), "no braces | here")
    }
}
