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

    /// GitHub's list of releases, made of these entries.
    private func newest(_ entries: Data...) -> NewVersion? {
        NewVersion(releases: Data("[".utf8) + entries.joined(separator: Data(",".utf8)) + Data("]".utf8))
    }

    func testTheReleaseIsRead() throws {
        let v = try XCTUnwrap(newest(release()))
        XCTAssertEqual(v.version, "0.23")
        XCTAssertEqual(v.page.absoluteString, "https://github.com/gwenn-ha-dev/Siliconed/releases/tag/v0.23")
        XCTAssertEqual(v.image?.absoluteString, "https://github.com/gwenn-ha-dev/Siliconed/releases/download/v0.23/Siliconed-0.23.dmg")
        // No image attached: the page alone.
        XCTAssertNil(try XCTUnwrap(newest(release(assets: "[]"))).image)
        XCTAssertNil(try XCTUnwrap(newest(release(assets: #"[{"name": "notes.txt", "browser_download_url": "https://github.com/gwenn-ha-dev/Siliconed/x"}]"#))).image)
    }

    /// A link elsewhere, a draft, a tag that is not a number: nothing offered.
    func testNothingIsTrusted() throws {
        XCTAssertNil(newest(release(page: "https://example.com/gwenn-ha-dev/Siliconed/")))
        XCTAssertNil(newest(release(page: "http://github.com/gwenn-ha-dev/Siliconed/releases/tag/v0.23")))
        XCTAssertNil(newest(release(page: "https://github.com/someone/Siliconed/releases/tag/v0.23")))
        XCTAssertNil(newest(release(extra: #""draft": true,"#)))
        XCTAssertNil(newest(release(extra: #""prerelease": true,"#)))
        XCTAssertNil(newest(release(tag: "nightly")))
        XCTAssertNil(newest(release(tag: "v0.23-beta")))
        XCTAssertNil(NewVersion(releases: Data("{}".utf8)))
        XCTAssertNil(NewVersion(releases: Data("<html>".utf8)))
        // An image hosted elsewhere is dropped; the page stays.
        let other = #"[{"name": "Siliconed.dmg", "browser_download_url": "https://evil.example/Siliconed.dmg"}]"#
        XCTAssertNil(try XCTUnwrap(newest(release(assets: other))).image)
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

    /// The highest version of the list, whatever GitHub's order or its « Latest »; what does not count is skipped,
    /// an entry it cannot read drops only itself.
    func testTheHighestOfTheListIsOffered() throws {
        func entry(_ v: String, _ extra: String = "") -> Data {
            release(tag: "v\(v)", page: "https://github.com/gwenn-ha-dev/Siliconed/releases/tag/v\(v)", extra: extra)
        }
        XCTAssertEqual(newest(entry("0.30"), entry("0.31"), entry("0.9"))?.version, "0.31")
        XCTAssertEqual(newest(entry("0.10"), entry("0.9"))?.version, "0.10")
        XCTAssertEqual(newest(entry("0.32", #""draft": true,"#), entry("0.31"))?.version, "0.31")
        XCTAssertEqual(newest(entry("0.32", #""prerelease": true,"#), entry("0.31"))?.version, "0.31")
        XCTAssertEqual(newest(Data(#"{"tag_name": 7}"#.utf8), entry("0.31"))?.version, "0.31")
        XCTAssertNil(newest())
    }
}
