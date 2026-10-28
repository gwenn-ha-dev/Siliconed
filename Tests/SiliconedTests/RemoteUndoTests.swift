import Foundation
import XCTest
@testable import Siliconed

/// **Removal undone, history resynchronised, refusals as codes** — the messages and the trash, without a
/// network: what crosses decodes to what was encoded, and the trash keeps what a client removed, no more.
final class RemoteUndoTests: XCTestCase {

    private func roundTrip(_ m: SessionMessage) throws -> SessionMessage {
        try JSONDecoder().decode(SessionMessage.self, from: JSONEncoder().encode(m))
    }

    func testTheNewMessagesCrossAsTheyWere() throws {
        let messages: [SessionMessage] = [
            .restoreImages([3, 9]), .synced(RemoteSync(entries: [9, 3], jobs: [1, 2, 5])),
            .synced(RemoteSync(entries: [], jobs: [])),
            .refused(.app(.imagesNotRestorable)),
            .refused(.text("No.")),
        ]
        for m in messages { XCTAssertEqual(try roundTrip(m), m) }
    }

    /// An error crosses as itself, whatever its parameters: the borrowing Mac says it in its language.
    func testEveryKindOfEngineErrorCrossesAsItself() throws {
        let errors: [EngineError] = [
            .modelNotInstalled(model: "z-image"), .emptyPrompt, .diskFull(needed: 5 << 30, available: nil),
            .invalidSteps(steps: 7, allowed: [4, 8, 16]),
            .formatRefused(width: 1000, height: 1000, reason: .notMultiple, bound: 16),
            .promptSyntax(position: 3, reason: .unclosedGroup),
            .insufficientMemory(needed: 12 << 30, budget: 10 << 30),
            .strengthTooLow(strength: 0.1, steps: 8, floor: 0.2),
            .internalFailure(component: "gpu", detail: "x"),
            .libraryRefused(path: "/a", reason: .notALibrary),
        ]
        for e in errors {
            let r = RemoteRefusal.engine(e)
            XCTAssertEqual(try roundTrip(.refused(r)), .refused(r))
            let a = RemoteAnswer.refused(r), i = RemoteIssue.failed(r)
            XCTAssertEqual(try JSONDecoder().decode(RemoteAnswer.self, from: JSONEncoder().encode(a)), a)
            XCTAssertEqual(try JSONDecoder().decode(RemoteIssue.self, from: JSONEncoder().encode(i)), i)
        }
    }

    func testTheAppSentencesCrossWithTheirParameters() throws {
        let sentences: [AppRefusal] = [
            .imageGone(7), .tooManyReferences(count: 4, model: "qwen-image-2.1", max: 3), .editSurface(width: 1536, height: 1024),
            .loraNotInstalled(name: "Gwenn"), .seedCount, .enlargeModelDownload, .enlargeFailed, .gridNotCropped,
            .sketchNotCropped, .cropNeeds(side: 512), .cropSmallest(side: 512), .imagesNotRestorable,
        ]
        for s in sentences {
            let r = RemoteRefusal.app(s)
            XCTAssertEqual(try JSONDecoder().decode(RemoteRefusal.self, from: JSONEncoder().encode(r)), r)
        }
        var q = RemoteQueue()
        q.problem = .app(.imageGone(2))
        XCTAssertEqual(try JSONDecoder().decode(RemoteQueue.self, from: JSONEncoder().encode(q)), q)
    }

    func testAStringLiteralIsTheTextFallback() {
        let r: RemoteRefusal = "No."
        XCTAssertEqual(r, .text("No."))
    }

    // MARK: - The trash

    func testTheTrashGivesBackWhatItHoldsOnceAndSaysNothingOfTheRest() {
        var t = SessionTrash<String>()
        t.put(3, "c"); t.put(5, "e"); t.put(4, "d")
        XCTAssertEqual(t.take([3, 9]), [3: "c"])
        XCTAssertEqual(t.take([3]), [:], "taken once")
        XCTAssertEqual(t.ordinals, [5, 4])
        XCTAssertEqual(t.take([4, 5]), [4: "d", 5: "e"])
        XCTAssertTrue(t.ordinals.isEmpty)
    }

    func testTheTrashForgetsTheOldestRemovedFirst() {
        var t = SessionTrash<Int>(capacity: 3)
        for o in 1...5 { t.put(o, o * 10) }
        XCTAssertEqual(t.ordinals, [3, 4, 5])
        XCTAssertEqual(t.take([1, 2]), [:])
        t.put(3, 31)
        t.put(6, 60)
        XCTAssertEqual(t.ordinals, [5, 3, 6], "put again: the newest")
        XCTAssertEqual(t.take([3]), [3: 31])
    }
}
