import CoreGraphics
import XCTest
@testable import Siliconed

/// **What a press on the curtain or the switch does** (issue #10), without a window: zoomed in, a drag moves the images,
/// the curtain's line moves only by its handle, the switch's click still switches; fitted, nothing changes.
final class CurtainGestureTests: XCTestCase {
    private let still = CGSize.zero
    private let tremble = CGSize(width: 2, height: -2)
    private let drag = CGSize(width: 40, height: 10)

    func testFittedNothingChanges() {
        // The curtain: a press anywhere takes the line, and its release does nothing more.
        for x: CGFloat in [0, 100, 250, 499] {
            XCTAssertEqual(CurtainGesture.role(alternates: false, zoomed: false, startX: x, lineX: 250), .line)
        }
        XCTAssertEqual(CurtainGesture.release(alternates: false, role: .line, translation: drag), .nothing)
        XCTAssertEqual(CurtainGesture.release(alternates: false, role: .line, translation: still), .nothing)
        // The switch: nothing follows the pointer, a click switches, a drag does nothing.
        XCTAssertEqual(CurtainGesture.role(alternates: true, zoomed: false, startX: 100, lineX: 500), CurtainGesture.Role.none)
        XCTAssertEqual(CurtainGesture.release(alternates: true, role: .none, translation: still), .swap)
        XCTAssertEqual(CurtainGesture.release(alternates: true, role: .none, translation: drag), .nothing)
        XCTAssertEqual(CurtainGesture.panned(drag, role: .none), .zero)
    }

    func testZoomedTheCurtainsLineMovesOnlyByItsHandle() {
        // On the line, within half the knob: the line.
        for x: CGFloat in [250, 236, 264, 235, 265] {
            XCTAssertEqual(CurtainGesture.role(alternates: false, zoomed: true, startX: x, lineX: 250), .line, "\(x)")
        }
        // Anywhere else: the images, and a click there leaves the line where it is.
        for x: CGFloat in [0, 100, 234, 266, 499] {
            XCTAssertEqual(CurtainGesture.role(alternates: false, zoomed: true, startX: x, lineX: 250), .pan, "\(x)")
        }
        XCTAssertEqual(CurtainGesture.release(alternates: false, role: .pan, translation: drag), .keepPan)
        XCTAssertEqual(CurtainGesture.release(alternates: false, role: .pan, translation: still), .nothing)
        XCTAssertEqual(CurtainGesture.release(alternates: false, role: .line, translation: drag), .nothing)
        // The line parked at an end is still taken there.
        XCTAssertEqual(CurtainGesture.role(alternates: false, zoomed: true, startX: 3, lineX: 0), .line)
    }

    func testZoomedTheSwitchClicksAndDrags() {
        XCTAssertEqual(CurtainGesture.role(alternates: true, zoomed: true, startX: 100, lineX: 500), .pan)
        // A click, even a trembling one, switches and moves nothing.
        XCTAssertEqual(CurtainGesture.release(alternates: true, role: .pan, translation: still), .swap)
        XCTAssertEqual(CurtainGesture.release(alternates: true, role: .pan, translation: tremble), .swap)
        XCTAssertEqual(CurtainGesture.panned(tremble, role: .pan), .zero)
        // A drag moves, and never switches at its end.
        XCTAssertEqual(CurtainGesture.release(alternates: true, role: .pan, translation: drag), .keepPan)
        XCTAssertEqual(CurtainGesture.panned(drag, role: .pan), drag)
    }

    func testTheClickSlop() {
        XCTAssertTrue(CurtainGesture.isClick(CGSize(width: 4, height: 0)))
        XCTAssertTrue(CurtainGesture.isClick(CGSize(width: -2.8, height: 2.8)))
        XCTAssertFalse(CurtainGesture.isClick(CGSize(width: 3, height: 3)))
        XCTAssertFalse(CurtainGesture.isClick(CGSize(width: 0, height: -4.1)))
    }
}
