// **Siliconed.app — the product.**
//
// What a user of Draw Things or ComfyUI downloads
// — Z-Image Turbo, Qwen-Image-2.1, Anima and Krea 2 in front, nothing preinstalled, each license accepted
// before its first byte, a measure of the machine that it can report. A client of the `Siliconed` library in the same package (`import`,
// never `@testable`): everything it calls is `package`.
//
// The files: `App.swift` (the app, its menus, its window), `AppState.swift` and `AppState+….swift` (the state and the
// calls to the library), `MainView.swift` (the window's view and the canvas), `Sidebar.swift` (the
// left column), `Rack.swift` (the chain, each module with its settings), `ModelsSheet.swift` (install,
// import, remove), `LicenseSheet.swift` (a license to accept),
// `DiagnosticSheet.swift` (« Measure This Mac »), `Problems.swift` (each `EngineError`, in the
// user's language), `Statistics.swift` (the floating statistics palette), `RemoteControl.swift` and `RemoteControlSocket.swift` (the
// socket that `silicontrol …` drives, `docs/REMOTE-CONTROL.md`; its `help` text is the module
// `SilicontrolHelp`, shared with `silicontrol`); the phrases are in `Localizable.xcstrings` (English source, fr de es it). To launch it:
// `tools/app.sh` (builds `Siliconed.app` at the repository root, translations included, and opens
// it), or `swift run SiliconedApp` (in English, no icon). **The models live in the app's space**
// (`Library.standard`, `~/Library/Application Support/Siliconed/`), which it fills itself — install,
// import, remove: the Models sheet —, or on another disk the user moved them to (Settings,
// `AppState+Place.swift`). `SILICONED_ROOT=<folder>` designates another space.

import AppKit
import Siliconed
import SwiftUI

@main
struct SiliconedApplication: App {
    @NSApplicationDelegateAdaptor(Delegate.self) private var delegate

    /// Launched as the capacity measure's child (`CapacityMeasure`), the process measures and exits here.
    init() { CapacityMeasure.runIfAsked() }

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
            Button("Check for Updates…") { window.checkForUpdates() }
            // The socket did not open (another Siliconed listening, a library path too long): said
            // here, greyed, where the remote control is — `silicontrol` cannot tell it from outside.
            if let reason = window.remote.failureMessage {
                Button("Remote control off: \(reason)") {}
                    .disabled(true)
            }
        }
        // The app's settings, in a window of ours (`SettingsWindow.swift`): SwiftUI's would keep its frame.
        CommandGroup(replacing: .appSettings) {
            Button("Settings…") { SettingsWindow.shared.show(window.local) }
                .keyboardShortcut(",")
        }
        CommandGroup(replacing: .help) {
            Button("Measure This Mac…") { app.openDiagnostic() }
        }
        CommandGroup(replacing: .newItem) {
            Button("Siliconed Window") { window.show() }
                .keyboardShortcut("n")
            Button("Import a Model or LoRA…") { app.chooseToImport() }
                .keyboardShortcut("o")
                .disabled(!app.importPossible)
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
            Divider()
            // **Where it computes** (`OtherMacs.swift`): this Mac, or one paired that lends itself.
            // Never both at once; the window's subtitle says which.
            Menu("Compute On") {
                Toggle("This Mac", isOn: Binding(get: { window.carrier.mirror == nil }, set: { if $0 { window.compute(on: nil) } }))
                ForEach(window.otherMacs.pairings?.borrowed ?? [], id: \.id) { b in
                    Toggle(isOn: Binding(get: { window.carrier.showing == b.id }, set: { if $0 { window.compute(on: b) } })) {
                        Text(verbatim: b.name)
                    }
                }
                if window.otherMacs.pairings == nil {
                    Button("Other Macs…") { SettingsWindow.shared.show(window.local) }
                }
            }
        }
        CommandMenu("Image") {
            Button("Full Screen") { if app.selectedImage != nil { app.fullScreen = true } }
                .keyboardShortcut("f")
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
            // **Incognito**: nothing of the user's survives the session. The render cache (prompts'
            // conditionings, posed images' latents, Qwen's K/V) serves during it and is emptied here —
            // which covers a crash — and at exit.
            MainWindow.shared.emptyRenderCache()
            MainWindow.shared.show()
            // The remote control (`silicontrol …`): the same queue, the same history.
            MainWindow.shared.remote.boot()
            MainWindow.shared.otherMacs.boot()
        }
    }

    /// **Incognito has a price**: the history lives only in this process. Quitting with images never
    /// saved, or with a render under way, asks first — and offers to export them.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        MainActor.assumeIsolated { MainWindow.shared.mayQuit() } ? .terminateNow : .terminateCancel
    }

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated {
            MainWindow.shared.remote.stop()
            MainWindow.shared.otherMacs.stop()
            MainWindow.shared.emptyRenderCache()
            MainWindow.shared.local.removeWorkingCopy()
        }
    }

    /// `siliconed://hf?…` — « Use this model » on Hugging Face, through the site's relay page;
    /// at launch too. Anything else the scheme receives is ignored.
    func application(_ application: NSApplication, open urls: [URL]) {
        MainActor.assumeIsolated {
            for link in urls.compactMap(HubLink.init) {
                MainWindow.shared.show()
                MainWindow.shared.local.open(link)
            }
        }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
        if !hasVisibleWindows { MainActor.assumeIsolated { MainWindow.shared.show() } }
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
}

/// The window's content: the state the carrier shows, rebuilt from scratch at each switch (`id`), so
/// that nothing of one state's views — a field being typed, a popover — passes to the other.
struct Shown: View {
    let carrier: Carrier
    let local: AppState
    let toggleStatistics: () -> Void

    var body: some View {
        let app = carrier.mirror ?? local
        MainView(app: app, toggleStatistics: toggleStatistics)
            .id(ObjectIdentifier(app))
    }
}

/// The window, created on demand and kept as long as it is open. The state (`AppState`) lives
/// as long as the process: closing then reopening the window finds the history again.
@MainActor
final class MainWindow: NSObject, NSWindowDelegate {
    static let shared = MainWindow()

    private var window: NSWindow?
    /// The exploration's panel (`Exploration.swift`), kept once opened.
    var explorationPanel: NSPanel?
    /// **The state shown**: this Mac's, or another Mac's mirror (`compute(on:)`) — every window and
    /// menu reads it here, so that the switch moves them all.
    var app: AppState { carrier.mirror ?? local }
    let carrier = Carrier()
    /// This Mac's own state: the queue, the history and the library; untouched while a mirror is shown.
    private(set) lazy var local = AppState(library: {
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

    /// Empties `<root>/cache/` (incognito, `Delegate`). A failure costs nothing but disk: the next
    /// launch tries again.
    func emptyRenderCache() { try? local.library.emptyCache() }

    func show() {
        if let window {
            NSApp.activate()
            window.makeKeyAndOrderFront(nil)
            return
        }
        let f = NSWindow(contentViewController: makeRoot())
        f.title = String(localized: "Siliconed")
        f.subtitle = Self.instanceName
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
        // First launch (nothing comes preinstalled): the canvas welcomes and offers the three models
        // (`Welcome`, MainView.swift) — no sheet thrown at the user before they have seen the window.
    }

    /// Two instances on one Mac (`SILICONED_ROOT`, the remote Mac's test): each window says which it is.
    private static var instanceName: String {
        ProcessInfo.processInfo.environment["SILICONED_ROOT"].map { _ in shared.otherMacs.name } ?? ""
    }

    /// One host for the window's life: the switch changes the state it shows (`Shown`), never the
    /// host — a new `contentViewController` rebuilt SwiftUI's toolbar half-bridged (short, its items
    /// on the left, seen on macOS 27, 28/10/2026).
    private func makeRoot() -> NSHostingController<Shown> {
        let host = NSHostingController(rootView: Shown(carrier: carrier, local: local) { [weak self] in
            self?.toggleStatistics()
        })
        host.sizingOptions = []
        // SwiftUI's toolbar (`.toolbar`) becomes the window's.
        host.sceneBridgingOptions = [.toolbars]
        return host
    }

    /// **The switch**: the window, its panels and the menus show `mac`'s state, or this Mac's (`nil`).
    /// The panels close — they were the other state's —; the form follows, filtered by the other
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
            m.prompt = from.prompt; m.identifier = from.identifier; m.steps = from.steps
            m.orientation = from.orientation; m.format = from.format; m.batch = from.batch
            m.fixedSeed = from.fixedSeed; m.seed = from.seed; m.detail = from.detail; m.previews = from.previews
            carrier.mirror = m
            carrier.showing = mac.id
            otherMacs.show(mac, in: m)
        } else {
            carrier.mirror = nil
            carrier.showing = nil
        }
        statistics?.close(); statistics = nil
        explorationPanel?.close(); explorationPanel = nil
        guard let window else { return }
        window.subtitle = mac.map { String(localized: "Computing on “\($0.name)”") } ?? Self.instanceName
        app.undoManager = window.undoManager
    }

    /// The project's page: each new version and its notes. The app does not update itself: "Check for
    /// Updates" only says whether a newer version is published, and opens its page.
    static let website = URL(string: "https://gwenn-ha-dev.github.io/Siliconed/")!

    /// "About": what the app is, the library it reads, and where a new version is found.
    func about() {
        let style: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: NSFont.smallSystemFontSize),
                                                    .foregroundColor: NSColor.secondaryLabelColor]
        let credits = NSMutableAttributedString(
            string: String(localized: "An image engine for Apple silicon, in Swift and Metal — fp32, verified stage by stage against the reference; the Neural Engine beside it where you choose.\n\nLibrary: \(app.library.displayPath)")
                + "\n\n" + String(localized: "New versions:") + " ",
            attributes: style)
        var link = style
        link[.link] = Self.website
        credits.append(NSAttributedString(string: "gwenn-ha-dev.github.io/Siliconed", attributes: link))
        NSApp.orderFrontStandardAboutPanel(options: [
            .applicationName: String(localized: "Siliconed"),
            .credits: credits,
        ])
        NSApp.activate()
    }

    /// **"Check for Updates"**: only when asked — nothing leaves the Mac otherwise ("works offline").
    /// Reads the latest release published on GitHub (tags `vX.Y.Z`) through
    /// an ephemeral session (incognito: no cookie, no cache), and compares it to the running version.
    func checkForUpdates() {
        Task {
            let alert = NSAlert()
            let current = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?"
            var page = Self.website
            do {
                var request = URLRequest(url: URL(string: "https://api.github.com/repos/gwenn-ha-dev/Siliconed/releases/latest")!)
                request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
                request.timeoutInterval = 15
                let (data, response) = try await URLSession(configuration: .ephemeral).data(for: request)
                guard (response as? HTTPURLResponse)?.statusCode == 200,
                      let release = try? JSONDecoder().decode(Release.self, from: data) else {
                    throw URLError(.badServerResponse)
                }
                let latest = release.tag_name.hasPrefix("v") ? String(release.tag_name.dropFirst()) : release.tag_name
                if let url = URL(string: release.html_url) { page = url }
                if Self.number(current).lexicographicallyPrecedes(Self.number(latest)) {
                    alert.messageText = String(localized: "Siliconed \(latest) is available.")
                    alert.informativeText = String(localized: "You have version \(current). The new version is downloaded from its page, with its notes.")
                    alert.addButton(withTitle: String(localized: "Open the Page"))
                    alert.addButton(withTitle: String(localized: "Later"))
                } else {
                    alert.messageText = String(localized: "Siliconed is up to date.")
                    alert.informativeText = String(localized: "Version \(current) is the latest published.")
                    alert.addButton(withTitle: String(localized: "OK"))
                }
            } catch {
                alert.messageText = String(localized: "The check did not succeed.")
                alert.informativeText = String(localized: "GitHub could not be reached. New versions are also listed on the project's page.")
                alert.addButton(withTitle: String(localized: "Open the Page"))
                alert.addButton(withTitle: String(localized: "Cancel"))
            }
            NSApp.activate()
            if alert.runModal() == .alertFirstButtonReturn, alert.buttons.count == 2 {
                NSWorkspace.shared.open(page)
            }
        }
    }

    private struct Release: Decodable {
        let tag_name: String
        let html_url: String
    }

    /// "0.7.1" → [0, 7, 1]: compared component by component, so 0.10 comes after 0.9; padded to three,
    /// so 0.8 and 0.8.0 are the same version.
    private static func number(_ version: String) -> [Int] {
        let parts = version.split(separator: ".").map { Int($0.prefix { $0.isNumber }) ?? 0 }
        return parts + Array(repeating: 0, count: max(0, 3 - parts.count))
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
            p.title = app.mirror.map { String(localized: "Statistics on “\($0.macName)”") } ?? String(localized: "Statistics")
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

    /// The modifiers that make a shortcut: Caps Lock, the numeric pad and the function flag are states
    /// of the keyboard, not of the chord.
    private static func chord(_ event: NSEvent) -> NSEvent.ModifierFlags {
        event.modifierFlags.intersection(.deviceIndependentFlagsMask).subtracting([.capsLock, .numericPad, .function])
    }

    private func watchKeys() {
        guard keyMonitor == nil else { return }
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self, let w = window, event.window === w, w.attachedSheet == nil, !app.fullScreen else { return event }
            // Esc leaves the field being typed in: the keys go back to the strip.
            if w.firstResponder is NSText {
                // The subjects' completion list, open under the caret, takes Esc itself (`PromptField`).
                guard event.keyCode == 53, !SubjectCompletion.isOpen, Self.chord(event).isEmpty else { return event }
                w.makeFirstResponder(nil)
                return nil
            }
            if app.cropping != nil { return cropKey(event) ? nil : event }
            // ⌘V by its character (AZERTY): an image on the pasteboard goes to the Images box.
            if Self.chord(event) == .command,
               event.charactersIgnoringModifiers?.lowercased() == "v" {
                return app.pasteImage() ? nil : event
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
        let lost = local.unsaved
        guard !lost.isEmpty || local.busy else { return true }
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
            + (local.busy ? "\n\n" + String(localized: "Quitting also stops the render in progress.") : "")
        alert.addButton(withTitle: lost.count == 1 ? String(localized: "Save…") : String(localized: "Export…"))
        alert.addButton(withTitle: String(localized: "Cancel"))
        alert.addButton(withTitle: String(localized: "Quit Without Saving"))
        alert.buttons[2].hasDestructiveAction = true
        switch alert.runModal() {
        case .alertFirstButtonReturn: return local.export(lost)
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
    }
}
