import Foundation
import XCTest
@testable import Siliconed

/// **What a lending Mac says of itself** (`Remote/Machine.swift`): the messages cross as JSON and come back equal.
final class RemoteMachineTests: XCTestCase {
    func testMachineTravelsAsJSON() throws {
        let m = RemoteMachine(chip: "Apple M4", hardwareModel: "Mac16,10", memoryBytes: 17_179_869_184, ceiling: 10_737_418_240,
                              ready: ["z-image", "qwen-image-2.1"])
        let messages: [SessionMessage] = [.machine(m), .refused(.app(.cropOutside(width: 1024, height: 768))), .enlargeRefused(7, .app(.imageGone(7))), .refused(.engine(.renderAlreadyRunning))]
        for message in messages {
            XCTAssertEqual(try JSONDecoder().decode(SessionMessage.self, from: JSONEncoder().encode(message)), message)
        }
    }

    func testThisMacIsRead() {
        let mac = Mac.current()
        XCTAssertGreaterThan(mac.memoryBytes, 0)
        XCTAssertFalse(mac.chip.isEmpty)
        XCTAssertTrue(mac.macOS.hasPrefix("\(ProcessInfo.processInfo.operatingSystemVersion.majorVersion)."))
    }
}
