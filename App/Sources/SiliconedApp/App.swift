// **Siliconed.app — the product.**
//
// What a user of Draw Things or ComfyUI downloads
// — the families of `Family.allCases` in front, nothing preinstalled, each license accepted
// before its first byte, a measure of the machine that it can report. A client of the `Siliconed` library in the same package (`import`,
// never `@testable`): everything it calls is `package`.
//
// The files: `App.swift` (the app, its menus), `MainWindow.swift` (its window), `AppState.swift` and `AppState+….swift` (the state and the
// calls to the library), `MainView.swift` (the window's view and the canvas), `Sidebar.swift` (the
// left column), `Rack.swift` (the chain, each module with its settings), `ModelsSheet.swift` (install,
// import, remove), `LicenseSheet.swift` (a license to accept),
// `Problems.swift` (each `EngineError`, in the
// user's language), `Inspectors.swift` and `ChainInspector.swift` (the two floating inspectors), `RemoteControl.swift` and `RemoteControlSocket.swift` (the
// socket that `silicontrol …` drives, `docs/REMOTE-CONTROL.md`; its `help` text is the module
// `SilicontrolHelp`, shared with `silicontrol`); the phrases are in `Localizable.xcstrings` (English source, then fr de es it ja zh-Hans). To launch it:
// `tools/app.sh` (builds `Siliconed.app` at the repository root, translations included, and opens
// it), or `swift run SiliconedApp` (in English, no icon). **The models live in the app's space**
// (`Library.standard`, `~/Library/Application Support/Siliconed/`), which it fills itself — install,
// import, remove: the Models sheet —, or on another disk the user moved them to (Settings,
// `AppState+Place.swift`). `SILICONED_ROOT=<folder>` designates another space.

import AppKit
import Siliconed
import SilicontrolHelp
import SwiftUI

@main
struct SiliconedApplication: App {
    @NSApplicationDelegateAdaptor(Delegate.self) private var delegate

    var body: some Scene {
        // The window is an NSWindow opened by the delegate (like Draw Things Headless's Generate
        // window): a single `AppState` per process,
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
            Button("Check for Updates…") { Updates.shared.check() }
            // The socket did not open (another Siliconed listening, a library path too long): said
            // here, greyed, where the remote control is — `silicontrol` cannot tell it from outside.
            if let reason = window.remote.failureMessage {
                Button("Remote control off: \(reason)") {}
                    .disabled(true)
            }
        }
        // The app's settings, in a window of ours (`SettingsWindow.swift`): SwiftUI's would keep its frame.
        CommandGroup(replacing: .appSettings) {
            Button("Settings…") { SettingsWindow.shared.show() }
                .keyboardShortcut(",")
        }
        CommandGroup(replacing: .help) {
            Button("Siliconed Help") { window.open(.help) }
            Button("Remote Control Help") { window.open(.remoteControl) }
            Button("Report a Problem…") { window.open(.problem) }
        }
        CommandGroup(replacing: .newItem) {
            Button("Models…") { app.managementOpen = true }
                .keyboardShortcut("m", modifiers: [.command, .shift])
            Button("Import a Model or LoRA…") { app.chooseToImport() }
                .disabled(!app.importPossible)
        }
        CommandGroup(replacing: .saveItem) {
            // `.saveItem` holds the system's Close: replaced, ⌘W must be given back.
            Button("Close Window") { NSApp.keyWindow?.performClose(nil) }
                .keyboardShortcut("w")
            Button("Save Image…") { if let e = app.selectedEntry { Task { await app.save(e) } } }
                .keyboardShortcut("s")
                .disabled(app.selectedEntry == nil)
            Button(app.targets.count > 1 ? "Export \(app.targets.count) Images…" : "Export Images…") {
                Task { await app.export(app.targets.count > 1 ? app.targets : app.history) }
            }
            .keyboardShortcut("e", modifiers: [.command, .shift])
            .disabled(app.history.isEmpty)
        }
        CommandGroup(after: .pasteboard) {
            Divider()
            Button(app.targets.count > 1 ? "Copy \(app.targets.count) Images" : "Copy Image") { app.copyImages(app.targets) }
                .keyboardShortcut("c", modifiers: [.command, .shift])
                .disabled(app.targets.isEmpty)
            Button("Copy as Terminal Command") { app.copyCommand() }
        }
        CommandGroup(after: .toolbar) {
            // The column shows or hides, like a sidebar: a part of the window, not a window — one fold for the text and the
            // chain. By its character, never its key code: the same letter on an AZERTY keyboard. Free in the menus.
            // Detached, the key brings the column's window forward (`toggle`): it shows, it never hides.
            Button(RackColumnModel.shared.folded || RackColumnModel.shared.detached ? "Show Chain" : "Hide Chain") {
                RackColumnModel.shared.toggle()
            }
                .keyboardShortcut("r", modifiers: [.command, .option])
            Divider()
            // On a series' grid, the cells' size; otherwise the image's zoom (`AppState+Zoom.swift`, also `silicontrol zoom`).
            Button("Zoom In") { app.zoom(.zoomIn) }
                .keyboardShortcut("+")
                .disabled(!app.canZoomIn)
            Button("Zoom Out") { app.zoom(.zoomOut) }
                .keyboardShortcut("-")
                .disabled(!app.canZoomOut)
            // On a series' grid, the cells back to the size that fills the canvas, following the window again.
            Button("Zoom to Fit") { app.zoom(.fit) }
                .keyboardShortcut("0")
                .disabled(!app.canZoomToFit)
            Divider()
        }
        CommandMenu("Render") {
            // ⌘↩ is also a sheet's or a panel's own button: while another window is key the menu item steps
            // aside, so that the key reaches that window's button.
            // Something stops the render: ⌘↩ unfolds the chain, whose foot says the reason, instead of doing nothing.
            Button("Generate") { if app.canRender { app.render() } else { RackColumnModel.shared.unfold() } }
                .keyboardShortcut(.return)
                .disabled(!window.carrier.mainIsKey)
            Button("Render Again (Same Seed)") { app.selectedEntry.map(app.redo) }
                .keyboardShortcut("r")
                .disabled(app.selectedEntry == nil)
            Divider()
            Button("Stop Render") { app.cancel() }
                .keyboardShortcut(".")
                .disabled(!app.inProgress)
            Button("Clear Queue and Stop") { app.stopAll() }
                .keyboardShortcut(".", modifiers: [.command, .option])
                .disabled(!app.busy && app.awaitingLicense.isEmpty)
            Divider()
            // **Where it computes** (`OtherMacs.swift`): this Mac, or one paired that lends itself.
            // Never both at once; the menu at the top of the window says which (`ComputeOnMenu`).
            Menu("Compute On") {
                Toggle("This Mac", isOn: Binding(get: { window.carrier.mirror == nil }, set: { if $0 { window.compute(on: nil) } }))
                // As the toolbar's `ComputeOnMenu`: a Mac not on the network is said, and cannot be chosen.
                ForEach(window.otherMacs.pairings?.borrowed ?? [], id: \.id) { b in
                    let reachable = window.otherMacs.found[b.id] != nil || window.carrier.showing == b.id
                    Toggle(isOn: Binding(get: { window.carrier.showing == b.id }, set: { if $0 { window.compute(on: b) } })) {
                        if reachable { Text(verbatim: b.name) } else { Text("\(b.name) — not on the network") }
                    }
                    .disabled(!reachable)
                }
                if window.otherMacs.pairings == nil {
                    Button("Other Macs…") { SettingsWindow.shared.show(.macs) }
                }
            }
        }
        CommandMenu("Image") {
            Button(app.componentsAway ? "Bring Back the Components" : "Move the Components Aside") {
                withAnimation(MainView.away) { app.componentsAway.toggle() }
            }
                // ⌥⌘F, not ⌃⌘F: that one is macOS's Full Screen. By its character, as every shortcut here.
                .keyboardShortcut("f", modifiers: [.command, .option])
            // « C » itself is read by its character in `MainWindow.key`, when nothing is being typed.
            Button("Crop…") { app.beginCropOfShownImage() }
                .disabled(app.selectedEntry == nil || app.cropping != nil)
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
                .disabled(app.selectedEntry == nil)
            Button("Edit This Image") { app.selectedEntry.map(app.placeReference) }
                .disabled(!app.editingPossible)
            Button("Enlarge ×4") { app.selectedEntry.map(app.enlarge) }
                .disabled(app.selectedEntry.flatMap(app.enlargeRefusal) != nil || app.selectedEntry == nil)
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
            Button("Show Siliconed") { window.show() }
            // The image's details are the popover of the ⓘ under it: the components come back for it to have its button.
            Button(app.imageDetails ? "Hide Image Details" : "Image Details") {
                if app.componentsAway { withAnimation(MainView.away) { app.componentsAway = false } }
                app.imageDetails.toggle()
            }
                .keyboardShortcut("i")
                .disabled(app.selectedEntry == nil || app.shownSeries != nil || app.historyGrid)
            Button("Chain Inspector") { window.toggleChain() }
                .keyboardShortcut("i", modifiers: [.command, .option])
            Button(app.historyGrid ? "Back to the Image" : "History Grid") {
                withAnimation(HistoryGrid.motion) { app.historyGrid.toggle() }
            }
                .keyboardShortcut("g", modifiers: [.command, .option])
                .disabled(app.history.isEmpty && !app.historyGrid)
        }
    }
}

@MainActor
final class Delegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Launched by `swift run` (no .app bundle), a SwiftUI executable remains a background
        // process: no Dock, no foreground window. `tools/app.sh` makes the bundle.
        NSApp.setActivationPolicy(.regular)
        // What older engines left, and another installation's compilations, erased before the engine's first work.
        MainWindow.shared.local.compiledAtLaunch()
        // **Incognito**: nothing of the user's survives the session. The render cache (prompts'
        // conditionings) lives in the engine's process (`EngineWorker`), in memory only, and ends with it: the worker
        // exits when the app closes its standard input, at exit or at a crash alike.
        MainWindow.shared.show()
        // The remote control (`silicontrol …`): the same queue, the same history.
        MainWindow.shared.remote.boot()
        MainWindow.shared.otherMacs.boot()
        Updates.shared.launch()
    }

    /// **Incognito has a price**: the history lives only in this process. Quitting with images never
    /// saved, or with a render under way, asks first — and offers to export them.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        MainWindow.shared.mayQuit()
    }

    func applicationWillTerminate(_ notification: Notification) {
        MainWindow.shared.remote.stop()
        MainWindow.shared.otherMacs.stop()
        EngineWorker.shared.stop()
        MainWindow.shared.local.stopWatchingVolumes()
    }

    /// `siliconed://hf?…` — « Use this model » on Hugging Face, through the site's relay page; at launch too.
    /// Anything else the scheme receives is ignored.
    func application(_ application: NSApplication, open urls: [URL]) {
        for link in urls.compactMap(HubLink.init) {
            MainWindow.shared.show()
            MainWindow.shared.app.open(link)
        }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
        if !hasVisibleWindows { MainWindow.shared.show() }
        return true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
}

/// **The Dock follows the render**: a bar under the icon while it runs, and the number of renders
/// still waiting as a badge — a 2-minute render is followed from another app. The icon never bounces:
/// the bar is enough, and the app sends no notification of any kind.
///
/// The bar is drawn by the tile's own view, not an `NSProgressIndicator`: the Dock draws the tile's
/// view into an image on `display()`, and the indicator, drawn so, shows an earlier value than the one
/// it was just given (0.5 for 0.9, its own animation) — in the Dock the bar never moved.
@MainActor
enum DockTile {
    private final class Tile: NSView {
        var fraction = 0.0
        override func draw(_ dirtyRect: NSRect) {
            NSApp.applicationIconImage.draw(in: bounds)
            let track = NSRect(x: bounds.width * 0.1, y: bounds.height * 0.07, width: bounds.width * 0.8, height: bounds.height * 0.1)
            let radius = track.height / 2
            NSColor.black.withAlphaComponent(0.6).setFill()
            NSBezierPath(roundedRect: track, xRadius: radius, yRadius: radius).fill()
            let inner = track.insetBy(dx: 2, dy: 2)
            let done = NSRect(x: inner.minX, y: inner.minY, width: max(inner.height, inner.width * min(1, max(0, fraction))), height: inner.height)
            Palette.liveInk.setFill()
            NSBezierPath(roundedRect: done, xRadius: done.height / 2, yRadius: done.height / 2).fill()
        }
    }

    private static var tile: Tile?

    static func show(fraction: Double?, waiting: Int) {
        let dock = NSApp.dockTile
        dock.badgeLabel = waiting > 0 ? "\(waiting)" : nil
        if let fraction {
            if tile == nil {
                let view = Tile(frame: NSRect(origin: .zero, size: dock.size))
                dock.contentView = view
                tile = view
            }
            tile?.fraction = fraction
        } else if tile != nil {
            dock.contentView = nil
            tile = nil
        }
        dock.display()
    }

    static func finished() {
        show(fraction: nil, waiting: 0)
    }
}

/// **A presentation mode, for captures** (`SILICONED_PRESENTATION`, which the private tool that drives a test space sets):
/// the window shows a neutral name for this Mac — the variable's value, « Mac mini » — and a neutral library,
/// `~/Pictures/Siliconed`, where it would show the machine's own name and the test space's folder. Only what is shown
/// changes: the library read, the name other Macs see and pair with stay this Mac's own.
enum Presentation {
    /// The name shown for this Mac, `nil` outside the mode.
    static let macName: String? = ProcessInfo.processInfo.environment["SILICONED_PRESENTATION"].flatMap { $0.isEmpty ? nil : $0 }
    /// The library's place as shown in the mode.
    static let libraryPath = "~/Pictures/Siliconed"
}

extension Library {
    /// The root as the window shows it (`displayPath`), or the neutral one of the presentation mode.
    var shownPath: String { Presentation.macName == nil ? displayPath : Presentation.libraryPath }
}
