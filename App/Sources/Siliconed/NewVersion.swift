import Foundation

/// **The latest Siliconed GitHub publishes**: its number, its page and its disk image, read from
/// `releases/latest` as GitHub writes it. Nothing in the answer is trusted: a link counts only on this
/// project's own pages of github.com, any other is dropped (the disk image then falls back to the page).
package struct NewVersion: Sendable, Equatable {
    /// The one request the check makes: no identifier, no cookie (the app sends it through an ephemeral session).
    package static let latest = URL(string: "https://api.github.com/repos/gwenn-ha-dev/Siliconed/releases/latest")!

    /// `0.23`, the tag without its `v`.
    package let version: String
    /// The release's page, with its notes.
    package let page: URL
    /// The `.dmg` attached to the release: the click downloads it directly.
    package let image: URL?

    package init(version: String, page: URL, image: URL?) {
        (self.version, self.page, self.image) = (version, page, image)
    }

    /// The release as GitHub's API returns it; `nil` for anything else (no tag, a draft, a link elsewhere).
    package init?(release data: Data) {
        struct Release: Decodable {
            struct Asset: Decodable { let name: String; let browser_download_url: String }
            let tag_name: String
            let html_url: String
            let draft: Bool?
            let prerelease: Bool?
            let assets: [Asset]?
        }
        guard let r = try? JSONDecoder().decode(Release.self, from: data), r.draft != true, r.prerelease != true,
              let page = Self.ours(r.html_url) else { return nil }
        let version = r.tag_name.hasPrefix("v") ? String(r.tag_name.dropFirst()) : r.tag_name
        guard version.first?.isNumber == true, version.allSatisfy({ $0.isNumber || $0 == "." }) else { return nil }
        let image = r.assets?.first { $0.name.lowercased().hasSuffix(".dmg") }.flatMap { Self.ours($0.browser_download_url) }
        self.init(version: version, page: page, image: image)
    }

    /// Whether this version comes after `current` — `nil` (a build outside the app's bundle) comes before any.
    package func isNewer(than current: String?) -> Bool {
        guard let current else { return true }
        return Self.number(current).lexicographicallyPrecedes(Self.number(version))
    }

    /// "0.7.1" → [0, 7, 1]: compared component by component, so 0.10 comes after 0.9; padded to three,
    /// so 0.8 and 0.8.0 are the same version.
    static func number(_ version: String) -> [Int] {
        let parts = version.split(separator: ".").map { Int($0.prefix { $0.isNumber }) ?? 0 }
        return parts + Array(repeating: 0, count: max(0, 3 - parts.count))
    }

    /// A link of this project on github.com, in https; `nil` otherwise.
    private static func ours(_ text: String) -> URL? {
        guard let url = URL(string: text), url.scheme == "https", url.host == "github.com",
              url.path.hasPrefix("/gwenn-ha-dev/Siliconed/") else { return nil }
        return url
    }
}
