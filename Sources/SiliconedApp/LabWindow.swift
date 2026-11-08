// **The Prompt Lab in a window of its own** (issue 16): the unfolded plate, detached from the image and as large as the hand
// makes it — the text grows with the window. Folded, the lab is the line under the image again, detached or not: the window is
// only where the unfolded plate goes (`LabModel.detached`).
//
// - **Its close button folds the lab**, as the chevron does: a gesture under way stops and the writer gives its memory back. The
//   window is only ordered out, so that it comes back where the hand left it — for the session: incognito, no frame is kept
//   past it, nothing is restored (as `SettingsWindow`).
// - Esc in the main window does not fold it: it is not over the image any more.

import AppKit
import SwiftUI

@MainActor
final class LabWindow: NSObject, NSWindowDelegate {
    static let shared = LabWindow()
    private var window: NSWindow?

    var isShown: Bool { window?.isVisible == true }

    func show(_ model: LabModel) {
        if let window {
            window.makeKeyAndOrderFront(nil)
            return
        }
        let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 760, height: 520),
                         styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: true)
        w.title = String(localized: "Prompt Lab")
        w.backgroundColor = Palette.columnInk
        w.isReleasedWhenClosed = false
        w.isRestorable = false
        w.contentMinSize = NSSize(width: 520, height: 360)
        w.delegate = self
        let host = NSHostingController(rootView: LabView(model: model, inWindow: true))
        // The window is the hand's size, not the plate's: the text takes what is left.
        host.sizingOptions = []
        w.contentViewController = host
        w.setContentSize(NSSize(width: 760, height: 520))
        w.center()
        window = w
        w.makeKeyAndOrderFront(nil)
    }

    /// Out of sight, kept for the session: it comes back with its frame.
    func hide() { window?.orderOut(nil) }

    /// The close button is the chevron's fold: `fold` orders the window out.
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        LabModel.shared.fold()
        return false
    }
}
