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

    func testAnUnbrokenRunOfOneTokenIsCut() throws {
        let cut = try XCTUnwrap(firstCut([9] + Array(repeating: 4, count: 100)))
        XCTAssertEqual(cut.kept, 2)
        XCTAssertEqual(cut.at, 33)
    }
}
