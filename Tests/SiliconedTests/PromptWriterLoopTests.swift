import XCTest
@testable import Siliconed

/// The loop guard on identifiers alone: what it cuts, and what it must leave whole.
final class PromptWriterLoopTests: XCTestCase {
    /// Replays `tokens` the way the writer does and returns the length kept at the first cut, or nil.
    private func firstCut(_ tokens: [Int]) -> (at: Int, kept: Int)? {
        for n in 1...tokens.count {
            if let keep = RepetitionGuard.cut(Array(tokens[..<n])) { return (n, keep) }
        }
        return nil
    }

    func testAOneWordTagLoopIsCutToOneTurn() throws {
        // "high quality, 1girl, red dress, solo, solo, solo, …": the head (ids 1…9), then ` solo` `,` over and over.
        let head = Array(1...9)
        let tokens = head + Array(repeating: [100, 101], count: 300).flatMap { $0 }
        let cut = try XCTUnwrap(firstCut(tokens))
        XCTAssertLessThanOrEqual(cut.at, head.count + 2 * RepetitionGuard.minimumSpan)
        // The periodic stretch starts at 100, one occurrence of the cycle is kept.
        XCTAssertEqual(cut.kept, head.count + 2)
    }

    func testTheCycleReachesBackIntoTheHeadWhenItsFirstTokenMatches() throws {
        // "…dress ,  solo , solo , solo": the comma before the first solo is already the cycle's.
        let tokens = [1, 2, 3, 101, 100, 101, 100, 101] + Array(repeating: [100, 101], count: 40).flatMap { $0 }
        let cut = try XCTUnwrap(firstCut(tokens))
        XCTAssertEqual(Array(tokens[..<cut.kept]), [1, 2, 3, 101, 100])
    }

    func testALongPeriodLoopIsCut() throws {
        let sentence = Array(200..<230)   // 30 tokens
        let tokens = [1, 2, 3] + Array(repeating: sentence, count: 6).flatMap { $0 }
        let cut = try XCTUnwrap(firstCut(tokens))
        XCTAssertEqual(cut.kept, 3 + 30)
    }

    func testNormalTextsAreLeftWhole() {
        // "ha ha ha ha": a short repetition, far from the span.
        XCTAssertNil(firstCut([1, 2, 3] + Array(repeating: [50, 51], count: 3).flatMap { $0 }))
        // A tag list that shares its commas (id 7) and some tags, never a whole period over and over.
        var tags: [Int] = []
        for i in 0..<40 { tags += [10 + (i * 7) % 23, 7, 40 + i % 5, 7] }
        XCTAssertNil(firstCut(tags))
        // A line said four times, long enough to mean it, is short of the span (4 × 6 = 24 < 32).
        XCTAssertNil(firstCut(Array(repeating: [1, 2, 3, 4, 5, 6], count: 4).flatMap { $0 }))
        // Three turns of a long period.
        XCTAssertNil(firstCut(Array(repeating: Array(0..<30), count: 3).flatMap { $0 }))
        // Short and empty inputs.
        XCTAssertNil(RepetitionGuard.cut([]))
        XCTAssertNil(RepetitionGuard.cut([5]))
    }

    // The writer's goldens hold 48 tokens against transformers (`qwen21 writer`); a loop seen in a render began past them and
    // ran to the 768-token budget. What follows replays such runs on identifiers, the writer's way: one token at a time.

    func testALoopThatStartsPastTheGoldensIsCutAtItsSpan() throws {
        // 120 tokens of a description, then a 7-token phrase over and over until the budget.
        let head = Array(1000..<1120), phrase = Array(500..<507)
        let budget = PromptWriterNeed.Limits().newTokens
        let tokens = Array((head + Array(repeating: phrase, count: budget / phrase.count).flatMap { $0 }).prefix(budget))
        let cut = try XCTUnwrap(firstCut(tokens))
        // Seen once the cycle covers the span (32 tokens, here more than four turns), one turn kept.
        XCTAssertEqual(cut.at, head.count + RepetitionGuard.minimumSpan)
        XCTAssertEqual(Array(tokens[..<cut.kept]), head + phrase)
    }

    func testTheLongestPeriodIsCutAfterThreeTurnsWithinTheBudget() throws {
        let head = Array(1000..<1050), period = Array(2000..<(2000 + RepetitionGuard.maximumPeriod))
        let tokens = head + Array(repeating: period, count: 12).flatMap { $0 }
        let cut = try XCTUnwrap(firstCut(tokens))
        XCTAssertEqual(cut.at, head.count + RepetitionGuard.longRepeats * period.count)
        XCTAssertEqual(cut.kept, head.count + period.count)
        // The worst case still stops well inside the writer's budget: a page, not pages.
        XCTAssertLessThan(cut.at, PromptWriterNeed.Limits().newTokens)
    }

    func testAListOfAboutNinetyTokensStartedLateIsCut() throws {
        // Z-Image's Rewrite: 350 tokens of a drifting description, then thirteen « no visible wear on bike … , » of 7 tokens
        // over and over to the budget — a period of 91, which four turns would not have fitted.
        var head: [Int] = []
        for i in 0..<350 { head.append(3000 + (i * 37) % 251) }
        let list = (0..<13).flatMap { item in [700, 701, 702, 703, 704, 800 + item, 705] }
        let budget = PromptWriterNeed.Limits().newTokens
        let tokens = Array((head + Array(repeating: list, count: 8).flatMap { $0 }).prefix(budget))
        let cut = try XCTUnwrap(firstCut(tokens))
        XCTAssertEqual(cut.at, head.count + RepetitionGuard.longRepeats * list.count)
        XCTAssertEqual(Array(tokens[..<cut.kept]), head + list)
    }

    func testALateLoopIsCutEvenNearTheBudget() throws {
        // 600 tokens without a loop (a varied description: no period up to 64 repeats), then "solo," to the end.
        var head: [Int] = []
        for i in 0..<600 { head.append(3000 + (i * 37) % 251) }
        XCTAssertNil(firstCut(head))
        let tokens = Array((head + Array(repeating: [100, 101], count: 200).flatMap { $0 })
            .prefix(PromptWriterNeed.Limits().newTokens))
        let cut = try XCTUnwrap(firstCut(tokens))
        XCTAssertEqual(cut.at, head.count + RepetitionGuard.minimumSpan)
        XCTAssertEqual(cut.kept, head.count + 2)
    }

    func testWhatTheGuardDoesNotSeeByDesign() {
        // A period longer than the maximum, to the budget: not cut (the decode stops at `newTokens`).
        let long = Array(0..<(RepetitionGuard.maximumPeriod + 1))
        let budget = PromptWriterNeed.Limits().newTokens
        XCTAssertNil(firstCut(Array(Array(repeating: long, count: 12).flatMap { $0 }.prefix(budget))))
        // A loop that drifts by one token at each turn: never exact, never cut.
        var drifting: [Int] = Array(1000..<1060)
        for turn in 0..<70 { drifting += [500, 501, 502, 503, 504, 505, 506, 600 + turn] }
        XCTAssertNil(firstCut(Array(drifting.prefix(budget))))
    }

    func testAnUnbrokenRunOfOneTokenIsCut() throws {
        let cut = try XCTUnwrap(firstCut([9] + Array(repeating: 4, count: 100)))
        XCTAssertEqual(cut.kept, 2)
        XCTAssertEqual(cut.at, 33)
    }
}
