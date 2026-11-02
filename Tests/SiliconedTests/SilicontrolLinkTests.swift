import XCTest
import SilicontrolHelp

/// The `silicontrol` link refuses a translocated app: what the path test sees, and what it must leave alone.
final class SilicontrolLinkTests: XCTestCase {
    func testATranslocatedBundleIsSeen() {
        XCTAssertTrue(SilicontrolLink.isTranslocatedPath(
            "/private/var/folders/xy/abc123/T/AppTranslocation/8E1A9C1B-2F35-4C59-9B8E-0D6A5E2B7C41/d/Siliconed.app"))
        XCTAssertTrue(SilicontrolLink.isTranslocatedPath(
            "/private/var/folders/xy/abc123/T/AppTranslocation/8E1A9C1B-2F35-4C59-9B8E-0D6A5E2B7C41/d/Siliconed.app/"))
    }

    func testOrdinaryPlacesAreNot() {
        XCTAssertFalse(SilicontrolLink.isTranslocatedPath("/Applications/Siliconed.app"))
        XCTAssertFalse(SilicontrolLink.isTranslocatedPath("/Volumes/Work/Downloads/Siliconed.app"))
        XCTAssertFalse(SilicontrolLink.isTranslocatedPath("/Volumes/Siliconed/Siliconed.app"))
        // A folder named like the mount, without its UUID and `d`.
        XCTAssertFalse(SilicontrolLink.isTranslocatedPath("/Volumes/Work/AppTranslocation/Siliconed.app"))
        XCTAssertFalse(SilicontrolLink.isTranslocatedPath("/Volumes/Work/AppTranslocation/not-a-uuid/d/Siliconed.app"))
        XCTAssertFalse(SilicontrolLink.isTranslocatedPath("/"))
        XCTAssertFalse(SilicontrolLink.isTranslocatedPath(""))
    }

    func testTheSystemAnswersNoForAFolderOfTheTests() {
        // The bundle running the tests is no translocated app, whichever source answers.
        XCTAssertFalse(SilicontrolLink.isTranslocated(Bundle.main.bundleURL))
        XCTAssertFalse(SilicontrolLink.isTranslocated(URL(fileURLWithPath: NSTemporaryDirectory())))
    }
}
