// The main window: the state it shows (`Carrier`), its creation, its inspectors, quitting, its minimum size.

import AppKit
import Siliconed
import SilicontrolHelp
import SwiftUI

/// Which state the window shows: observed, so that the menus follow the switch.
@MainActor @Observable
final class Carrier {
    var mirror: AppState?
    /// The other Mac's `id` while its mirror is shown.
    var showing: String?
    /// The main window (or a sheet of it) is the key window: « Generate » of the Render menu answers ⌘↩ then,
    /// and only then (`MainWindow.watchKeyWindow`).
    var mainIsKey = true
}

/// The window's content: the state the carrier shows, rebuilt from scratch at each switch (`id`), so
/// that nothing of one state's views — a field being typed, a popover — passes to the other.
struct Shown: View {
    let carrier: Carrier
    let local: AppState

    var body: some View {
        let app = carrier.mirror ?? local
        MainView(app: app)
            .id(ObjectIdentifier(app))
    }
}

/// The window, created on demand and kept as long as it is open. The state (`AppState`) lives
/// as long as the process: closing then reopening the window finds the history again.
@MainActor
final class MainWindow: NSObject, NSWindowDelegate {
    static let shared = MainWindow()

    private(set) var window: NSWindow?
    /// **The state shown**: this Mac's, or another Mac's mirror (`compute(on:)`) — every window and
    /// menu reads it here, so that the switch moves them all.
    var app: AppState { carrier.mirror ?? local }
    let carrier = Carrier()
    /// This Mac's own state: the queue, the history and the library; untouched while a mirror is shown.
    private(set) lazy var local = AppState(library: {
        // What 0.30 kept under other keys (its Precision), carried before the state reads the preferences, then deleted.
        RetiredPreferences.migrate(.standard, neuralEngineKey: AppState.neuralEngineKey)
        // The user's B, before the state reads it (`AppState.memoryCeiling`) and the engine's process is launched.
        AppState.restoreMemoryBudget()
        // `SILICONED_ROOT` names the app's space, as `Library.standard` does: a library moved
        // elsewhere is followed (`Library.placed`), and never made where its disk is absent.
        let space = ProcessInfo.processInfo.environment["SILICONED_ROOT"].flatMap { $0.isEmpty ? nil : $0 }
            .map { URL(fileURLWithPath: $0) } ?? Library.standard.root
        let b = Library.placed(in: space)
        try? FileManager.default.createDirectory(atPath: Library(root: space).map(""), withIntermediateDirectories: true)
        return b
    }())
    private(set) lazy var remote = RemoteControl(app: local)
    /// The other Macs of the network (`OtherMacs.swift`).
    private(set) lazy var otherMacs = OtherMacs(local: local)

    func show() {
        if let window {
            NSApp.activate()
            window.makeKeyAndOrderFront(nil)
            return
        }
        let f = NSWindow(contentViewController: makeRoot())
        // The title is hidden but kept: the Window menu, Mission Control and `silicontrol snapshot --window` read it.
        f.title = [String(localized: "Siliconed"), Self.instanceName].filter { !$0.isEmpty }.joined(separator: " — ")
        // **No bar**: the content runs under the toolbar to the window's top edge. The system's toolbar stays,
        // transparent — it is the zone that moves the window, and the rack's header lives in it (`MainView`).
        f.styleMask = [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView]
        f.titleVisibility = .hidden
        f.titlebarAppearsTransparent = true
        f.toolbarStyle = .unified
        f.titlebarSeparatorStyle = .none
        f.setContentSize(NSSize(width: 1480, height: 940))
        f.isReleasedWhenClosed = false
        f.delegate = self
        // Incognito: no state restored at the next launch. The window's frame is kept: it says nothing of what was made.
        f.isRestorable = false
        f.center()
        f.setFrameAutosaveName("Main")
        window = f
        app.undoManager = f.undoManager
        watchKeys()
        watchKeyWindow()
        NSApp.activate()
        f.makeKeyAndOrderFront(nil)
        if RackColumnModel.shared.detached { rackPanel.show() }
        // First launch (nothing comes preinstalled): the canvas welcomes and offers the models
        // (`Welcome`, MainView.swift) — no sheet thrown at the user before they have seen the window.
    }

    /// Two instances on one Mac (`SILICONED_ROOT`, the remote Mac's test): each window's title says which it is.
    private static var instanceName: String {
        ProcessInfo.processInfo.environment["SILICONED_ROOT"].map { _ in shared.otherMacs.shownName } ?? ""
    }

    /// One host for the window's life: the switch changes the state it shows (`Shown`), never the
    /// host — a new `contentViewController` rebuilt SwiftUI's toolbar half-bridged (short, its items
    /// on the left, seen on macOS 27).
    private func makeRoot() -> NSHostingController<Shown> {
        let host = NSHostingController(rootView: Shown(carrier: carrier, local: local))
        host.sizingOptions = []
        // SwiftUI's toolbar (`.toolbar`) becomes the window's.
        host.sceneBridgingOptions = [.toolbars]
        return host
    }

    /// **The switch**: the window, its inspectors and the menus show `mac`'s state, or this Mac's (`nil`).
    /// The form follows, filtered by the other
    /// Mac's catalog when it arrives (`AppState.receive`). A queue running here goes on.
    func compute(on mac: Pairings.Borrowed?) {
        guard mac?.id != carrier.showing else { return }
        let from = app
        otherMacs.show(nil, in: nil)
        if let mac {
            let link = MirrorLink(macName: mac.name, send: { [weak self] m in self?.otherMacs.send(m, to: mac.id) },
                                  sendBlob: { [weak self] d, id in self?.otherMacs.send(blob: d, id: id, to: mac.id) },
                                  uploadPort: UploadPort(
                                    linked: { [weak self] in self?.otherMacs.isJoined(mac.id) ?? false },
                                    send: { [weak self] id, source in self?.otherMacs.send(stream: source, id: id, to: mac.id) },
                                    cancel: { [weak self] id in self?.otherMacs.cancel(stream: id, to: mac.id) }))
            let m = AppState(mirroring: link, library: local.library, subjects: local.subjects)
            m.copyForm(from: from)
            carrier.mirror = m
            carrier.showing = mac.id
            otherMacs.show(mac, in: m)
        } else {
            carrier.mirror = nil
            carrier.showing = nil
        }
        guard let window else { return }
        app.undoManager = window.undoManager
    }

    /// The project's page: each new version and its notes. The app does not update itself: `Updates` only says
    /// whether a newer version is published, and where to get it.
    static let website = URL(string: "https://gwenn-ha-dev.github.io/Siliconed/")!

    /// **The edition**: each version is named after a beach or a coast of the Celtic Atlantic — Ireland, Wales,
    /// Cornwall, Brittany, Galicia. A place name, the same in every language; the pages and the pictures of a
    /// version are that place's. Shown at the right of the window's top band and in "About", nowhere else.
    static let edition = "Rhossili"
    /// As it is shown: the place, then the word — a proper name, the same in every language.
    static let editionLine = edition + " edition"

    /// "About": what the app is, its edition, the library it reads, and where a new version is found.
    func about() {
        let style: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: NSFont.smallSystemFontSize),
                                                    .foregroundColor: Palette.secondaryInk]
        let credits = NSMutableAttributedString(
            string: String(localized: "An image engine for Apple silicon, in Swift on Core AI, verified stage by stage against the reference.\n\nLibrary: \(app.library.shownPath)")
                + "\n\n" + String(localized: "New versions:") + " ",
            attributes: style)
        var link = style
        link[.link] = Self.website
        credits.append(NSAttributedString(string: "gwenn-ha-dev.github.io/Siliconed", attributes: link))
        NSApp.orderFrontStandardAboutPanel(options: [
            .applicationName: String(localized: "Siliconed"),
            .applicationVersion: [Infotext.appVersion, Self.editionLine].compactMap { $0 }.joined(separator: " · "),
            .credits: credits,
        ])
        NSApp.activate()
    }

    /// **The « Chain » inspector** (`Inspectors.swift`).
    private(set) lazy var inspectors = Inspectors(carrier: carrier, local: local)

    func toggleChain() { inspectors.toggle(beside: window) }
    /// The Chain inspector shown, at its top (`CompilationMark`).
    func showChain() { inspectors.show(beside: window) }

    /// **The column in its own window**, when it is detached (`RackColumn.swift`).
    private(set) lazy var rackPanel = RackPanel(carrier: carrier, local: local)

    /// « Chain », at `stage`'s line: what a stage's line in the rack opens.
    func inspect(_ stage: Engine.Stage) { inspectors.show(stage: stage, beside: window) }

    @ObservationIgnored private var keyWindowObservers: [NSObjectProtocol] = []

    /// Which window is key, for the menus: the Render menu's ⌘↩ belongs to the main window only.
    private func watchKeyWindow() {
        guard keyWindowObservers.isEmpty else { return }
        let center = NotificationCenter.default
        for name in [NSWindow.didBecomeKeyNotification, NSWindow.didResignKeyNotification] {
            keyWindowObservers.append(center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.refreshKeyWindow() }
            })
        }
    }

    private func refreshKeyWindow() {
        guard let window else { return }
        let key = NSApp.keyWindow
        // No key window (the app is in the background): the menu is not reachable anyway.
        // The detached column is the main window's own: ⌘↩ typed in its prompt generates.
        carrier.mainIsKey = key == nil || key === window || key?.sheetParent === window
            || key != nil && key === rackPanel.window || key?.sheetParent != nil && key?.sheetParent === rackPanel.window
    }

    /// The pages Help opens: the docs on GitHub, in French when the app speaks it.
    enum Page { case help, remoteControl, problem }

    func open(_ page: Page) {
        let french = Bundle.main.preferredLocalizations.first?.hasPrefix("fr") ?? false
        let base = "https://github.com/gwenn-ha-dev/Siliconed/"
        let url: String = switch page {
        case .help: base + "blob/main/docs/" + (french ? "APP.fr.md" : "APP.md")
        case .remoteControl: base + "blob/main/docs/" + (french ? "REMOTE-CONTROL.fr.md" : "REMOTE-CONTROL.md")
        // GitHub's chooser of issue templates; nothing is attached.
        case .problem: base + "issues/new/choose"
        }
        if let url = URL(string: url) { NSWorkspace.shared.open(url) }
    }

    /// Takes the keyboard from the prompt (a click on an image): the arrows then walk the strip.
    func releaseTextFocus() {
        guard let w = window, w.firstResponder is NSText else { return }
        w.makeFirstResponder(nil)
    }

    /// **The keys of the strip**, as in Photos and the Finder — only when nothing is being typed (the
    /// prompt, a seed, a size; Esc leaves the field) and no sheet is open: ← → walk along it (⇧ marks the run), Home and
    /// End go to its ends, Space moves the components aside, ⌫ removes from the history (⌘Z
    /// undoes), ⌘A marks every image, Esc closes the history's grid, brings the components back, leaves the curtain or leaves only the one
    /// image shown.
    var keyMonitor: Any?


    /// Quitting: nothing to lose, it quits. Otherwise the alert says what would be lost, and offers to
    /// export what was never saved first — the export writes off the main thread, and the app quits once it has.
    func mayQuit() -> NSApplication.TerminateReply {
        let lost = local.unsaved
        guard !lost.isEmpty || local.busy else { return .terminateNow }
        let alert = NSAlert()
        if lost.isEmpty {
            alert.messageText = String(localized: "A render is in progress.")
            alert.informativeText = String(localized: "Quitting stops it, and empties the queue.")
            alert.addButton(withTitle: String(localized: "Quit"))
            alert.addButton(withTitle: String(localized: "Cancel"))
            return alert.runModal() == .alertFirstButtonReturn ? .terminateNow : .terminateCancel
        }
        alert.messageText = lost.count == 1 ? String(localized: "One image was never saved.")
                                            : String(localized: "\(lost.count) images were never saved.")
        alert.informativeText = String(localized: "Siliconed keeps nothing on disk: what is not saved disappears when the app quits.")
            + (local.busy ? "\n\n" + String(localized: "Quitting also stops the render in progress.") : "")
        alert.addButton(withTitle: lost.count == 1 ? String(localized: "Save…") : String(localized: "Export…"))
        alert.addButton(withTitle: String(localized: "Cancel"))
        alert.addButton(withTitle: String(localized: "Quit Without Saving"))
        alert.buttons[2].hasDestructiveAction = true
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            Task { NSApp.reply(toApplicationShouldTerminate: await self.local.export(lost)) }
            return .terminateLater
        case .alertThirdButtonReturn: return .terminateNow
        default: return .terminateCancel
        }
    }

    /// **The smallest the window gets** (`MinimumContent`), read on screen with the chain at its usual 400 pt and the history at
    /// its usual thumbnails (124 pt): 1140 wide leaves the room between them 616, where the bar under an image shows a
    /// few words of its legend beside its ten buttons (at 1000, 476: the prompt cut to two words — seen 14/11/2026); 680 high
    /// leaves the rack its Text box above a footer that grows with a queue and a problem's card, and an image of about 480 pt.
    static let minimumContent = NSSize(width: 1140, height: 680)

    func windowWillResize(_ sender: NSWindow, to frameSize: NSSize) -> NSSize {
        sender === window ? MinimumContent.clamp(frameSize, of: sender, to: Self.minimumContent) : frameSize
    }

    func windowWillClose(_ notification: Notification) {
        // The inspector closes by itself; only the main window tears down, and takes it with it — and the chain's panel.
        guard (notification.object as? NSWindow) === window else { return }
        inspectors.hide()
        rackPanel.hide()
        // Nothing must keep occupying this Mac's GPU with no view to show the result. Only this Mac's queue: a mirror shown
        // leaves the other Mac's queue to it (its own window, and the other clients, still follow it).
        local.unmount()
        window = nil
    }
}

/// **A window's minimum, held by its delegate.** `contentMinSize` does not hold under SwiftUI: the hosting controller sets it
/// back to zero once it lays the window out (read on macOS 27: 0 × 0 on the main window and the Prompt Lab's,
/// both set to more before), and a window could then be dragged down to its title bar (issue #7).
enum MinimumContent {
    /// The frame size a live resize asks for, grown so that its content is never smaller than `minimum`.
    @MainActor static func clamp(_ frameSize: NSSize, of window: NSWindow, to minimum: NSSize) -> NSSize {
        let content = window.contentRect(forFrameRect: NSRect(origin: .zero, size: frameSize)).size
        return NSSize(width: frameSize.width + max(0, minimum.width - content.width),
                      height: frameSize.height + max(0, minimum.height - content.height))
    }
}
