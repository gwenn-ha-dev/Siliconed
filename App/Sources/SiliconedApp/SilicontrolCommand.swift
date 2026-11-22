// The `silicontrol` command linked into `/usr/local/bin`.

import AppKit
import SilicontrolHelp

/// **The `silicontrol` command** ships inside the bundle (`Contents/Helpers/silicontrol`); this links it
/// into `/usr/local/bin`, like `code` for VS Code. On a fresh Mac `/usr/local/bin` may not exist and
/// belongs to root: the link is made as administrator (macOS asks for the password); if that is
/// refused or fails, the alert shows the two commands to paste into a terminal. The link follows the
/// app: a rebuilt or moved app stays the one it drives, as long as the link points into it. An app run
/// translocated by Gatekeeper (`SilicontrolLink`) gets no link: it would dangle at the next launch.
enum SilicontrolCommand {
    static let destination = "/usr/local/bin/silicontrol"
    /// The helper inside this app — what the link must point to.
    static var source: String { Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/silicontrol").path }

    enum State { case installed, otherCopy, absent }
    /// **Whether `destination` is this app's helper**: a link to another copy of Siliconed (an old one
    /// moved, a build) drives that copy's protocol, not this one's.
    static var state: State {
        guard let target = try? FileManager.default.destinationOfSymbolicLink(atPath: destination) else {
            return FileManager.default.fileExists(atPath: destination) ? .otherCopy : .absent
        }
        return URL(fileURLWithPath: target).standardizedFileURL.path == URL(fileURLWithPath: source).standardizedFileURL.path
            ? .installed : .otherCopy
    }

    @MainActor static func install() {
        let alert = NSAlert()
        guard FileManager.default.isExecutableFile(atPath: source) else {
            alert.messageText = String(localized: "The silicontrol command is missing from this app.")
            alert.informativeText = String(localized: "Build the app with tools/app.sh.")
            alert.runModal()
            return
        }
        guard !SilicontrolLink.isTranslocated(Bundle.main.bundleURL) else {
            alert.messageText = String(localized: "Move Siliconed to the Applications folder first.")
            alert.informativeText = String(localized: "macOS runs this copy from a temporary place, made again at each launch: a command linked to it would stop working. Move Siliconed to Applications, open it from there, then install the command.")
            alert.runModal()
            return
        }
        // Writable by this account (a Homebrew Mac): no password.
        let fm = FileManager.default
        if fm.isWritableFile(atPath: "/usr/local/bin") {
            try? fm.removeItem(atPath: destination)
            if (try? fm.createSymbolicLink(atPath: destination, withDestinationPath: source)) != nil {
                return installed(alert)
            }
        }
        // Otherwise, as administrator: the folder too, which a fresh Mac does not have.
        let quoted = "'" + source.replacingOccurrences(of: "'", with: "'\\''") + "'"
        let shell = "mkdir -p /usr/local/bin && ln -sf \(quoted) \(destination)"
        let script = "do shell script \"\(shell.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\""))\" with administrator privileges"
        var failure: NSDictionary?
        NSAppleScript(source: script)?.executeAndReturnError(&failure)
        guard let failure else { return installed(alert) }
        // -128: the user cancelled the password prompt — their answer, nothing to add.
        if (failure[NSAppleScript.errorNumber] as? Int) == -128 { return }
        alert.messageText = String(localized: "The silicontrol command was not installed.")
        alert.informativeText = String(localized: "To install it yourself, paste this into a terminal:")
        let command = "sudo mkdir -p /usr/local/bin && sudo ln -sf \(quoted) \(destination)"
        let field = NSTextField(wrappingLabelWithString: command)
        field.isSelectable = true
        field.font = .monospacedSystemFont(ofSize: NSFont.smallSystemFontSize, weight: .regular)
        field.frame = NSRect(x: 0, y: 0, width: 380, height: 60)
        alert.accessoryView = field
        alert.addButton(withTitle: String(localized: "Copy the Command"))
        alert.addButton(withTitle: String(localized: "Close"))
        if alert.runModal() == .alertFirstButtonReturn {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(command, forType: .string)
        }
    }

    @MainActor private static func installed(_ alert: NSAlert) {
        alert.messageText = String(localized: "The silicontrol command is installed.")
        alert.informativeText = String(localized: "In a terminal: silicontrol help")
        alert.runModal()
    }
}
