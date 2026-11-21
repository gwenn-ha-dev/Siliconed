import Foundation
import XCTest
@testable import Siliconed

/// **The check for a new version**: GitHub's answer read as it is written, nothing of it trusted, and the numbers
/// compared as versions. No network.
final class NewVersionTests: XCTestCase {
    private func release(tag: String = "v0.23", page: String = "https://github.com/gwenn-ha-dev/Siliconed/releases/tag/v0.23",
                         assets: String = #"[{"name": "Siliconed-0.23.dmg", "browser_download_url": "https://github.com/gwenn-ha-dev/Siliconed/releases/download/v0.23/Siliconed-0.23.dmg"}]"#,
                         extra: String = "") -> Data {
        Data(#"{"tag_name": "\#(tag)", "html_url": "\#(page)", "name": "Siliconed 0.23", \#(extra) "assets": \#(assets)}"#.utf8)
    }

    func testTheReleaseIsRead() throws {
        let v = try XCTUnwrap(NewVersion(release: release()))
        XCTAssertEqual(v.version, "0.23")
        XCTAssertEqual(v.page.absoluteString, "https://github.com/gwenn-ha-dev/Siliconed/releases/tag/v0.23")
        XCTAssertEqual(v.image?.absoluteString, "https://github.com/gwenn-ha-dev/Siliconed/releases/download/v0.23/Siliconed-0.23.dmg")
        // No image attached: the page alone.
        XCTAssertNil(try XCTUnwrap(NewVersion(release: release(assets: "[]"))).image)
        XCTAssertNil(try XCTUnwrap(NewVersion(release: release(assets: #"[{"name": "notes.txt", "browser_download_url": "https://github.com/gwenn-ha-dev/Siliconed/x"}]"#))).image)
    }

    /// A link elsewhere, a draft, a tag that is not a number: nothing offered.
    func testNothingIsTrusted() throws {
        XCTAssertNil(NewVersion(release: release(page: "https://example.com/gwenn-ha-dev/Siliconed/")))
        XCTAssertNil(NewVersion(release: release(page: "http://github.com/gwenn-ha-dev/Siliconed/releases/tag/v0.23")))
        XCTAssertNil(NewVersion(release: release(page: "https://github.com/someone/Siliconed/releases/tag/v0.23")))
        XCTAssertNil(NewVersion(release: release(extra: #""draft": true,"#)))
        XCTAssertNil(NewVersion(release: release(extra: #""prerelease": true,"#)))
        XCTAssertNil(NewVersion(release: release(tag: "nightly")))
        XCTAssertNil(NewVersion(release: release(tag: "v0.23-beta")))
        XCTAssertNil(NewVersion(release: Data("{}".utf8)))
        XCTAssertNil(NewVersion(release: Data("<html>".utf8)))
        // An image hosted elsewhere is dropped; the page stays.
        let other = #"[{"name": "Siliconed.dmg", "browser_download_url": "https://evil.example/Siliconed.dmg"}]"#
        XCTAssertNil(try XCTUnwrap(NewVersion(release: release(assets: other))).image)
    }

    func testVersionsAreComparedAsNumbers() {
        func newer(_ latest: String, _ current: String?) -> Bool {
            NewVersion(version: latest, page: URL(string: "https://github.com/gwenn-ha-dev/Siliconed/")!, image: nil).isNewer(than: current)
        }
        XCTAssertTrue(newer("0.23", "0.22"))
        XCTAssertTrue(newer("0.10", "0.9"))
        XCTAssertTrue(newer("1.0", "0.99.9"))
        XCTAssertTrue(newer("0.22.1", "0.22"))
        XCTAssertFalse(newer("0.22", "0.22"))
        XCTAssertFalse(newer("0.22", "0.22.0"))
        XCTAssertFalse(newer("0.21", "0.22"))
        XCTAssertTrue(newer("0.21", nil), "a build outside the bundle has no version: any is newer")
    }
}
