// **Siliconed.app — the product, and the compiled reference example of `docs/API.md`.**
//
// Two roles, one code. **The product**: what a user of Draw Things or ComfyUI downloads
// — Z-Image Turbo and Qwen-Image-2.1 in front, nothing preinstalled, each license accepted
// before its first byte, a diagnostic that reports the machine. **The example**: a pure client of the
// `Siliconed` library (`import`, never `@testable`), compiled by the package (executable target
// `SiliconedApp`, `swift build`), so the API cannot drift from its doc without breaking the build —
// and therefore the close-out. Everything it calls is public, exactly what another macOS 15 app
// would see.
//
// The files: `App.swift` (the app, its menus, its window), `AppState.swift` (the state and the
// calls to the library), `MainView.swift` (the window's view and the canvas), `Sidebar.swift` (the
// left column), `Rack.swift` (the chain, each module with its settings), `ModelsSheet.swift` (install,
// import, remove), `LicenseSheet.swift` (a license to accept),
// `DiagnosticSheet.swift` (« Report My Configuration »), `Problems.swift` (each `EngineError`, in the
// user's language), `Statistics.swift` (the floating statistics palette), `RemoteControl.swift` (the
// socket that `silicontrol …` drives, `docs/REMOTE-CONTROL.md`; its `help` text is the module
// `SilicontrolHelp`, shared with `silicontrol`); the phrases are in `Localizable.xcstrings` (English source, fr de es it). To launch it:
// `tools/app.sh` (builds `Siliconed.app` at the repository root, translations included, and opens
// it), or `swift run SiliconedApp` (in English, no icon). **The models live in the app's space**
// (`Library.standard`, `~/Library/Application Support/Siliconed/`), which it fills itself — install,
// import, remove: the Models sheet. `SILICONED_ROOT=<folder>` designates another one.

import AppKit
import Siliconed
import SwiftUI

@main
struct SiliconedApplication: App {
    @NSApplicationDelegateAdaptor(Delegate.self) private var delegate

    var body: some Scene {
        // The window is an NSWindow opened by the delegate (like Draw Things Headless's Generate
        // window): a single `AppState` per process, hence a single `EngineSettings.load` —
        // and nothing of what SwiftUI would retain from a scene (position, restored state).
        Settings { EmptyView() }
            .commands { AppCommands(window: .shared) }
    }
}

/// **The menus**: everything the app does from the keyboard. The shortcuts live here, not on the
/// buttons: a single place, and they work whatever has the focus.
private struct AppCommands: Commands {
    let window: MainWindow
    private var app: AppState { window.app }

    var body: some Commands {
        CommandGroup(replacing: .appInfo) {
            Button("About Siliconed") { window.about() }
            Button("Install the silicontrol command…") { SilicontrolCommand.install() }
            // The socket did not open (another Siliconed listening, a library path too long): said
            // here, greyed, where the remote control is — `silicontrol` cannot tell it from outside.
            if let reason = window.remote.failureMessage {
                Button("Remote control off: \(reason)") {}
                    .disabled(true)
            }
        }
        // The app's one setting, in its menu rather than a Settings window (whose frame SwiftUI would
        // remember: incognito).
        CommandGroup(replacing: .appSettings) {
            Toggle("Developer Mode", isOn: Binding(get: { app.developerMode }, set: { app.developerMode = $0 }))
        }
        CommandGroup(replacing: .help) {
            Button("Report My Configuration…") { app.openDiagnostic() }
        }
        CommandGroup(replacing: .newItem) {
            Button("Siliconed Window") { window.show() }
                .keyboardShortcut("n")
            Button("Import a Model or LoRA…") { app.chooseToImport() }
                .keyboardShortcut("o")
                .disabled(!app.managementPossible)
            Button("Models…") { app.managementOpen = true }
                .keyboardShortcut("m", modifiers: [.command, .shift])
        }
        CommandGroup(replacing: .saveItem) {
            Button("Save Image…") { app.selectedEntry.map(app.save) }
                .keyboardShortcut("s")
                .disabled(app.selectedEntry == nil)
            Button(app.targets.count > 1 ? "Export \(app.targets.count) Images…" : "Export Images…") {
                app.export(app.targets.count > 1 ? app.targets : app.history)
            }
            .keyboardShortcut("e", modifiers: [.command, .shift])
            .disabled(app.history.isEmpty)
        }
        CommandGroup(after: .pasteboard) {
            Divider()
            Button(app.targets.count > 1 ? "Copy \(app.targets.count) Images" : "Copy Image") { app.copyImages(app.targets) }
                .keyboardShortcut("c", modifiers: [.command, .shift])
                .disabled(app.targets.isEmpty)
            Button("Copy the silicontrol command") { app.copyCommand() }
        }
        CommandGroup(after: .toolbar) {
            // On the exploration's grid, the cells' size; otherwise the image's zoom.
            Button("Zoom In") {
                if app.showsExploration { app.exploreCellSide = AppState.clampedCellSide(app.exploreCellSide * 1.25) }
                else { app.zoom = min(app.zoom * 1.5, 8) }
            }
                .keyboardShortcut("+")
                .disabled(app.selectedImage == nil && !app.showsExploration)
            Button("Zoom Out") {
                if app.showsExploration { app.exploreCellSide = AppState.clampedCellSide(app.exploreCellSide / 1.25) }
                else { app.zoom = max(app.zoom / 1.5, 1) }
            }
                .keyboardShortcut("-")
                .disabled(app.zoom == 1 && !app.showsExploration)
            Button("Zoom to Fit") { app.zoom = 1 }
                .keyboardShortcut("0")
                .disabled(app.zoom == 1)
            Divider()
        }
        CommandMenu("Render") {
            Button("Generate") { app.render() }
                .keyboardShortcut(.return)
            Button("Redo Image (Same Seed)") { app.selectedEntry.map(app.redo) }
                .keyboardShortcut("r")
            Button("Exploration") { window.showExploration() }
                .keyboardShortcut("g", modifiers: [.command, .option])
            Divider()
            Button("Stop Render") { app.cancel() }
                .keyboardShortcut(".")
            Button("Clear Queue and Stop") { app.stopAll() }
                .keyboardShortcut(".", modifiers: [.command, .option])
        }
        CommandMenu("Image") {
            Button("Full Screen") { if app.selectedImage != nil { app.fullScreen = true } }
                .keyboardShortcut("f")
            // ← → Home End Space ⌫ ⌘A work without ⌘ when the prompt is not being typed in
            // (`MainWindow.key`); the menu keeps forms that work from anywhere.
            Button("Previous Image") { app.chooseAdjacent(-1) }
                .keyboardShortcut(.leftArrow, modifiers: [.command, .option])
            Button("Next Image") { app.chooseAdjacent(1) }
                .keyboardShortcut(.rightArrow, modifiers: [.command, .option])
            Button("Select All Images") { app.markAll() }
                .keyboardShortcut("a", modifiers: [.command, .option])
                .disabled(app.history.count < 2)
            Divider()
            Button("Reuse These Settings") { app.selectedEntry.map(app.restoreSettings) }
            Button("Edit This Image") { app.selectedEntry.map(app.placeReference) }
                .disabled(!app.editingPossible)
            Divider()
            // No ⌘⌫: it deletes the line in the prompt field. ⌫ alone, outside the field, does it.
            Button(app.targets.count > 1 ? "Remove \(app.targets.count) Images from History" : "Remove from History") {
                app.delete(app.targets)
            }
            .disabled(app.targets.isEmpty)
            Button("Clear History…") { app.clearHistory() }
                .disabled(app.history.isEmpty)
        }
        CommandGroup(after: .windowList) {
            Button("Statistics") { window.toggleStatistics() }
                .keyboardShortcut("i", modifiers: [.command, .option])
        }
    }
}

final class Delegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Launched by `swift run` (no .app bundle), a SwiftUI executable remains a background
        // process: no Dock, no foreground window. `tools/app.sh` makes the bundle.
        NSApp.setActivationPolicy(.regular)
        MainActor.assumeIsolated {
            MainWindow.shared.show()
            // The remote control (`silicontrol …`): the same queue, the same history.
            MainWindow.shared.remote.boot()
        }
    }

    /// **Incognito has a price**: the history lives only in this process. Quitting with images never
    /// saved, or with a render under way, asks first — and offers to export them.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        MainActor.assumeIsolated { MainWindow.shared.mayQuit() } ? .terminateNow : .terminateCancel
    }

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated { MainWindow.shared.remote.stop() }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
        if !hasVisibleWindows { MainActor.assumeIsolated { MainWindow.shared.show() } }
        return true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
}

/// The window, created on demand and kept as long as it is open. The state (`AppState`) lives
/// as long as the process: closing then reopening the window finds the history again.
@MainActor
final class MainWindow: NSObject, NSWindowDelegate {
    static let shared = MainWindow()

    private var window: NSWindow?
    /// The exploration's panel (`Exploration.swift`), kept once opened.
    var explorationPanel: NSPanel?
    private(set) lazy var app = AppState(library: {
        let b = ProcessInfo.processInfo.environment["SILICONED_ROOT"].flatMap { $0.isEmpty ? nil : $0 }
            .map { Library(root: URL(fileURLWithPath: $0)) } ?? .standard
        try? FileManager.default.createDirectory(atPath: b.map(""), withIntermediateDirectories: true)
        return b
    }())
    private(set) lazy var remote = RemoteControl(app: app)

    func show() {
        if let window {
            NSApp.activate()
            window.makeKeyAndOrderFront(nil)
            return
        }
        let host = NSHostingController(rootView: MainView(app: app) { [weak self] in
            self?.toggleStatistics()
        })
        host.sizingOptions = []
        // SwiftUI's toolbar (`.toolbar`) becomes the window's.
        host.sceneBridgingOptions = [.toolbars]
        let f = NSWindow(contentViewController: host)
        f.title = String(localized: "Siliconed")
        f.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        f.toolbarStyle = .unified
        f.titlebarSeparatorStyle = .automatic
        f.setContentSize(NSSize(width: 1480, height: 940))
        f.contentMinSize = NSSize(width: 1000, height: 680)
        f.isReleasedWhenClosed = false
        f.delegate = self
        // Incognito: no window position retained, no state restored at the next launch.
        f.isRestorable = false
        f.center()
        window = f
        app.undoManager = f.undoManager
        watchKeys()
        NSApp.activate()
        f.makeKeyAndOrderFront(nil)
        // First launch (nothing comes preinstalled): the canvas welcomes and offers the two models
        // (`Welcome`, MainView.swift) — no sheet thrown at the user before they have seen the window.
    }

    /// "About": what the app is, and the library it reads.
    func about() {
        let credits = NSAttributedString(
            string: String(localized: "An image engine for Apple silicon, in Swift and Metal — fp32 end to end, verified stage by stage against the reference.\n\nLibrary: \(app.library.displayPath)"),
            attributes: [.font: NSFont.systemFont(ofSize: NSFont.smallSystemFontSize),
                         .foregroundColor: NSColor.secondaryLabelColor])
        NSApp.orderFrontStandardAboutPanel(options: [
            .applicationName: String(localized: "Siliconed"),
            .credits: credits,
        ])
        NSApp.activate()
    }

    /// **The statistics palette**: floating, above the app, retaining nothing (neither
    /// position nor restored state — incognito). Same `AppState`, hence same figures.
    private var statistics: NSPanel?

    func toggleStatistics() {
        if let p = statistics, p.isVisible { p.orderOut(nil); return }
        if statistics == nil {
            let p = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 360, height: 620),
                            styleMask: [.titled, .closable, .resizable, .utilityWindow, .nonactivatingPanel],
                            backing: .buffered, defer: false)
            p.title = String(localized: "Statistics")
            p.isFloatingPanel = true
            p.hidesOnDeactivate = false
            p.isRestorable = false
            p.isReleasedWhenClosed = false
            let host = NSHostingController(rootView: Statistics(app: app))
            host.sizingOptions = []
            p.contentViewController = host
            p.setContentSize(NSSize(width: 360, height: 640))
            if let f = window?.frame {
                p.setFrameTopLeftPoint(NSPoint(x: f.maxX - 380, y: f.maxY - 60))
            } else {
                p.center()
            }
            statistics = p
        }
        statistics?.orderFront(nil)
    }

    /// Takes the keyboard from the prompt (a click on an image): the arrows then walk the strip.
    func releaseTextFocus() {
        guard let w = window, w.firstResponder is NSText else { return }
        w.makeFirstResponder(nil)
    }

    /// **The keys of the strip**, as in Photos and the Finder — only when nothing is being typed (the
    /// prompt, a seed, a size; Esc leaves the field) and no sheet is open: ← → walk along it (⇧ marks the run), Home and
    /// End go to its ends, Space shows the image full screen, ⌫ removes from the history (⌘Z
    /// undoes), ⌘A marks every image, Esc leaves only the one shown.
    private var keyMonitor: Any?

    private func watchKeys() {
        guard keyMonitor == nil else { return }
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self, let w = window, event.window === w, w.attachedSheet == nil, !app.fullScreen else { return event }
            // Esc leaves the field being typed in: the keys go back to the strip.
            if w.firstResponder is NSText {
                guard event.keyCode == 53, event.modifierFlags.intersection(.deviceIndependentFlagsMask).isEmpty else { return event }
                w.makeFirstResponder(nil)
                return nil
            }
            guard !app.history.isEmpty else { return event }
            return key(event) ? nil : event
        }
    }

    private func key(_ event: NSEvent) -> Bool {
        let mods = event.modifierFlags.intersection(.deviceIndependentFlagsMask).subtracting([.numericPad, .function])
        let shift = mods == .shift
        switch Int(event.keyCode) {
        // The curtain shown, ← → move its line (`Curtain.swift`); ⇧ still marks a run, Esc leaves.
        case 123 where mods.isEmpty && Curtain.shown(app): Curtain.shared.nudge(-1)
        case 124 where mods.isEmpty && Curtain.shown(app): Curtain.shared.nudge(1)
        case 123 where mods.isEmpty || shift: app.chooseAdjacent(-1, extending: shift)  // ←
        case 124 where mods.isEmpty || shift: app.chooseAdjacent(1, extending: shift)   // →
        case 115 where mods.isEmpty: app.chooseEnd(oldest: false)                       // Home
        case 119 where mods.isEmpty: app.chooseEnd(oldest: true)                        // End
        case 49 where mods.isEmpty:                                                     // Space
            guard app.selectedImage != nil, !(app.inProgress && app.followsRender) else { return false }
            app.fullScreen = true
        case 51 where mods.isEmpty, 117 where mods.isEmpty:                             // ⌫, ⌦
            guard !(app.inProgress && app.followsRender) else { return false }
            app.delete(app.targets)
        case 53 where mods.isEmpty:                                                     // Esc
            if app.showsOriginal { app.comparesOriginal = false; return true }
            guard app.marked.count > 1 else { return false }
            app.unmark()
        // ⌘A by its character, not its key code: code 0 is A on a QWERTY keyboard but Q on an AZERTY
        // one — read as a position, it took ⌘Q from the menu.
        case _ where mods == .command && event.charactersIgnoringModifiers?.lowercased() == "a": app.markAll()
        default: return false
        }
        return true
    }

    /// Quitting: nothing to lose, it quits. Otherwise the alert says what would be lost, and offers to
    /// export what was never saved first.
    func mayQuit() -> Bool {
        let lost = app.unsaved
        guard !lost.isEmpty || app.busy else { return true }
        let alert = NSAlert()
        if lost.isEmpty {
            alert.messageText = String(localized: "A render is in progress.")
            alert.informativeText = String(localized: "Quitting stops it, and empties the queue.")
            alert.addButton(withTitle: String(localized: "Quit"))
            alert.addButton(withTitle: String(localized: "Cancel"))
            return alert.runModal() == .alertFirstButtonReturn
        }
        alert.messageText = lost.count == 1 ? String(localized: "One image was never saved.")
                                            : String(localized: "\(lost.count) images were never saved.")
        alert.informativeText = String(localized: "Siliconed keeps nothing on disk: what is not saved disappears when the app quits.")
            + (app.busy ? "\n\n" + String(localized: "Quitting also stops the render in progress.") : "")
        alert.addButton(withTitle: lost.count == 1 ? String(localized: "Save…") : String(localized: "Export…"))
        alert.addButton(withTitle: String(localized: "Cancel"))
        alert.addButton(withTitle: String(localized: "Quit Without Saving"))
        alert.buttons[2].hasDestructiveAction = true
        switch alert.runModal() {
        case .alertFirstButtonReturn: return app.export(lost)
        case .alertThirdButtonReturn: return true
        default: return false
        }
    }

    func windowWillClose(_ notification: Notification) {
        // The palette closes by itself; only the main window tears down.
        guard (notification.object as? NSWindow) === window else { return }
        statistics?.orderOut(nil)
        // Nothing must keep occupying the GPU with no view to show the result.
        app.unmount()
        window = nil
    }
}

/// **The `silicontrol` command** ships inside the bundle (`Contents/Helpers/silicontrol`); this links it
/// into `/usr/local/bin`, like `code` for VS Code. On a fresh Mac `/usr/local/bin` may not exist and
/// belongs to root: the link is made as administrator (macOS asks for the password); if that is
/// refused or fails, the alert shows the two commands to paste into a terminal. The link follows the
/// app: a rebuilt or moved app stays the one it drives, as long as the link points into it.
enum SilicontrolCommand {
    static let destination = "/usr/local/bin/silicontrol"

    @MainActor static func install() {
        let source = Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/silicontrol").path
        let alert = NSAlert()
        guard FileManager.default.isExecutableFile(atPath: source) else {
            alert.messageText = String(localized: "The silicontrol command is missing from this app.")
            alert.informativeText = String(localized: "Build the app with tools/app.sh.")
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

/// **The Dock follows the render**: a bar under the icon while it runs, and the number of renders
/// still waiting as a badge — a 2-minute render is followed from another app. When the queue empties
/// with the app in the background, the icon bounces once.
@MainActor
enum DockTile {
    private static var bar: NSProgressIndicator?

    static func show(fraction: Double?, waiting: Int) {
        let tile = NSApp.dockTile
        tile.badgeLabel = waiting > 0 ? "\(waiting)" : nil
        if let fraction {
            if bar == nil {
                let icon = NSImageView(image: NSApp.applicationIconImage)
                let p = NSProgressIndicator(frame: NSRect(x: 12, y: 8, width: tile.size.width - 24, height: 14))
                p.style = .bar
                p.isIndeterminate = false
                p.minValue = 0; p.maxValue = 1
                icon.addSubview(p)
                tile.contentView = icon
                bar = p
            }
            bar?.doubleValue = fraction
        } else if bar != nil {
            tile.contentView = nil
            bar = nil
        }
        tile.display()
    }

    static func finished() {
        show(fraction: nil, waiting: 0)
        if !NSApp.isActive { NSApp.requestUserAttention(.informationalRequest) }
    }
}
