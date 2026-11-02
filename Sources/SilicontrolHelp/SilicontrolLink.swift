// **Where the `silicontrol` link may point.** The app links its helper (`Contents/Helpers/silicontrol`) into
// `/usr/local/bin`; the link is worth something only while the app stays where it was.
//
// **App Translocation** breaks that silently: an app opened straight from a download (quarantined, never moved by
// the Finder) is run by Gatekeeper from a random, read-only mount, `…/AppTranslocation/<UUID>/d/Siliconed.app`,
// made again at each launch. A link into it dangles at the next launch, and `silicontrol` then finds no helper.
// The remedy is the user's: move the app (to Applications), open it from there, then install the command.
//
// Here and not in the app's target: Foundation only, and the tests reach this module (`SilicontrolLinkTests`).

import Foundation

package enum SilicontrolLink {
    /// Whether the bundle at `path` is a translocated copy, read on its path alone:
    /// a component `AppTranslocation`, then a UUID, then `d`. A folder of the user's that happens to be named
    /// `AppTranslocation` lacks the two others.
    package static func isTranslocatedPath(_ path: String) -> Bool {
        let parts = URL(fileURLWithPath: path).standardizedFileURL.pathComponents
        guard parts.count >= 4 else { return false }
        for i in 0..<(parts.count - 3) where parts[i] == "AppTranslocation" {
            if UUID(uuidString: parts[i + 1]) != nil, parts[i + 2] == "d" { return true }
        }
        return false
    }

    /// Whether the bundle at `url` runs translocated: the system's answer (`SecTranslocateIsTranslocatedURL`, in
    /// Security but in no public header, so looked up at run time), and the path when the system gives none.
    package static func isTranslocated(_ url: URL) -> Bool {
        if let system = systemAnswer(url) { return system }
        return isTranslocatedPath(url.path)
    }

    private typealias IsTranslocated = @convention(c) (CFURL, UnsafeMutablePointer<Bool>,
                                                       UnsafeMutablePointer<Unmanaged<CFError>?>?) -> DarwinBoolean

    private static func systemAnswer(_ url: URL) -> Bool? {
        guard let security = dlopen("/System/Library/Frameworks/Security.framework/Security", RTLD_LAZY) else { return nil }
        defer { dlclose(security) }
        guard let symbol = dlsym(security, "SecTranslocateIsTranslocatedURL") else { return nil }
        let call = unsafeBitCast(symbol, to: IsTranslocated.self)
        var translocated = false
        guard call(url as CFURL, &translocated, nil).boolValue else { return nil }
        return translocated
    }
}
