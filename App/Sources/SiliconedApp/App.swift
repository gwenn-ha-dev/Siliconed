// **Siliconed.app — the product.**
//
// What a user of Draw Things or ComfyUI downloads
// — the families of `Family.allCases` in front, nothing preinstalled, each license accepted
// before its first byte, a measure of the machine that it can report. A client of the `Siliconed` library in the same package (`import`,
// never `@testable`): everything it calls is `package`.
//
// The files: `App.swift` (the app, its menus, its window), `AppState.swift` and `AppState+….swift` (the state and the
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
            // On a series' grid, the cells' size; otherwise the image's zoom.
            Button("Zoom In") {
                if app.shownSeries != nil { app.exploreCellSide = AppState.clampedCellSide(app.exploreCellSide * 1.25) }
                else { app.zoom = min(app.zoom * 1.5, app.zoomLimit) }
            }
                .keyboardShortcut("+")
                .disabled(app.selectedImage == nil && app.shownSeries == nil)
            Button("Zoom Out") {
                if app.shownSeries != nil { app.exploreCellSide = AppState.clampedCellSide(app.exploreCellSide / 1.25) }
                else { app.zoom = max(app.zoom / 1.5, 1) }
            }
                .keyboardShortcut("-")
                .disabled(app.zoom == 1 && app.shownSeries == nil)
            Button("Zoom to Fit") { app.zoom = 1 }
                .keyboardShortcut("0")
                .disabled(app.zoom == 1)
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
    private var keyMonitor: Any?

    /// The modifiers that make a shortcut: Caps Lock, the numeric pad and the function flag are states
    /// of the keyboard, not of the chord.
    private static func chord(_ event: NSEvent) -> NSEvent.ModifierFlags {
        event.modifierFlags.intersection(.deviceIndependentFlagsMask).subtracting([.capsLock, .numericPad, .function])
    }

    private func watchKeys() {
        guard keyMonitor == nil else { return }
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self, let w = window, event.window === w, w.attachedSheet == nil else { return event }
            // Esc leaves the field being typed in: the keys go back to the strip, and nothing folds. The prompt's list of
            // subjects, shown, is closed first, by the field.
            if w.firstResponder is NSText {
                guard event.keyCode == 53, Self.chord(event).isEmpty else { return event }
                if (w.firstResponder as? PromptTextView)?.listShown?() == true { return event }
                w.makeFirstResponder(nil)
                return nil
            }
            if app.cropping != nil { return cropKey(event) ? nil : event }
            // ⌘V by its character (AZERTY): an image on the pasteboard goes to the Images box.
            if Self.chord(event) == .command,
               event.charactersIgnoringModifiers?.lowercased() == "v" {
                return app.pasteImage() ? nil : event
            }
            // A control that has the keyboard (a slider, a stepper, a menu, a button reached by Tab) keeps its
            // keys: ← → adjust it, Space presses it, ⌫ is not the history's.
            if w.firstResponder is NSControl { return event }
            // The grid emptied (its images all removed) still closes by Esc, as its help says.
            if app.history.isEmpty, app.historyGrid, event.keyCode == 53, Self.chord(event).isEmpty {
                withAnimation(HistoryGrid.motion) { app.historyGrid = false }
                return nil
            }
            guard !app.history.isEmpty else { return event }
            // C opens the crop, by its character: on an AZERTY keyboard the key code of C is another letter.
            if Self.chord(event).isEmpty,
               event.charactersIgnoringModifiers?.lowercased() == "c" {
                app.beginCropOfShownImage()
                return nil
            }
            return key(event) ? nil : event
        }
    }

    /// **The keys while the crop frame is up**: Return (or Enter) crops, Esc leaves; the arrows, which
    /// would walk the strip, and Space, Delete and Home/End stay out of it. ⌘-keys go on to the menu.
    private func cropKey(_ event: NSEvent) -> Bool {
        let mods = Self.chord(event)
        guard mods.isEmpty else { return false }
        switch Int(event.keyCode) {
        case 36, 76: app.commitCrop()                                   // Return, Enter
        case 53: app.cancelCrop()                                       // Esc
        case 123, 124, 125, 126, 115, 119, 49, 51, 117: break            // ← → ↓ ↑ Home End Space ⌫ ⌦
        default: return event.charactersIgnoringModifiers?.lowercased() == "c"
        }
        return true
    }

    private func key(_ event: NSEvent) -> Bool {
        let mods = Self.chord(event)
        let shift = mods == .shift
        switch Int(event.keyCode) {
        // The curtain shown, ← → move its line — the switch shown, show A or B (`Curtain.swift`); ⇧ still marks a run, Esc leaves.
        case 123 where mods.isEmpty && Curtain.shown(app): Curtain.shared.nudge(-1)
        case 124 where mods.isEmpty && Curtain.shown(app): Curtain.shared.nudge(1)
        case 123 where mods.isEmpty || shift: app.chooseAdjacent(-1, extending: shift)  // ←
        case 124 where mods.isEmpty || shift: app.chooseAdjacent(1, extending: shift)   // →
        case 115 where mods.isEmpty: app.chooseEnd(oldest: false)                       // Home
        case 119 where mods.isEmpty: app.chooseEnd(oldest: true)                        // End
        case 49 where mods.isEmpty:                                                     // Space
            withAnimation(MainView.away) { app.componentsAway.toggle() }
        case 51 where mods.isEmpty, 117 where mods.isEmpty:                             // ⌫, ⌦
            guard !(app.inProgress && app.followsRender) else { return false }
            app.delete(app.targets)
        case 53 where mods.isEmpty:                                                     // Esc
            if app.historyGrid { withAnimation(HistoryGrid.motion) { app.historyGrid = false }; return true }
            if app.componentsAway { withAnimation(MainView.away) { app.componentsAway = false }; return true }
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
