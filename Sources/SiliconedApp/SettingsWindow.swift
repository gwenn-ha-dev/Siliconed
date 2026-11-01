// **The app's settings, in one window** — three tabs, named by what one comes to do; each switch with
// what it does, said under it.
//
//     Siliconed ▸ Settings… (⌘,)
//     ┌──────────────────────────────────────────────────────┐
//     │        [ General ]  Other Macs   Advanced             │
//     │ ┌──────────────────────────────────────────────────┐ │
//     │ │ Faster Rendering   the first steps at half size   │ │
//     │ │ Models             where the library is, shown in │ │
//     │ │                    the Finder; moved to another   │ │
//     │ │                    disk, opened, brought back     │ │
//     │ └──────────────────────────────────────────────────┘ │
//     └──────────────────────────────────────────────────────┘
//
//     General     what an everyday user touches: Faster Rendering for the model in the rack, the library.
//     Other Macs  lend this Mac, compute on another (`OtherMacsView.swift`).
//     Advanced    kept out of the way: This Mac (the capacity map, measured only when asked), Command
//                 Line (silicontrol, linked into /usr/local/bin), Beta (More Solvers, Negative LoRA
//                 Strengths: off by default, may change; named subjects moved to the Prompt Lab).
//                 ┌ This Mac ───── [Measure This Mac…]
//                 ├ Command Line ─ [Install…]
//                 └ Beta ───────── More Solvers · Negative LoRA Strengths
//
// - **Incognito**: an NSWindow of ours, as `MainWindow` — not SwiftUI's `Settings` scene, which would
//   keep its frame; not restorable. What it sets are the app's preferences (`AppState`'s header), the
//   same keys as before: a switch moved here keeps its value.
// - **A switch changes what it changes at once**, as from the menu it replaces: More Solvers off takes
//   the solver and a solver axis with it (`AppState.developerMode`, whose key keeps its old name).
// - How two images are compared (curtain, switch, side by side) is set under the image, where it is
//   used (`CurtainSwitch`), and remembered.
// - **The capacity map is measured from the sheet**, on the main window (`DiagnosticSheet`): this
//   window says where the session stands and opens it.

import AppKit
import SwiftUI

/// **The settings window**: one per process, opened by « Settings… ».
@MainActor
final class SettingsWindow: NSObject, NSWindowDelegate {
    static let shared = SettingsWindow()
    private var window: NSWindow?

    func show(_ app: AppState) {
        if let window {
            NSApp.activate()
            window.makeKeyAndOrderFront(nil)
            return
        }
        let view = SettingsView(app: app, macs: MainWindow.shared.otherMacs)
        let host = NSHostingController(rootView: view)
        host.sizingOptions = [.preferredContentSize]
        let w = NSWindow(contentViewController: host)
        w.title = String(localized: "Settings")
        w.styleMask = [.titled, .closable]
        w.isReleasedWhenClosed = false
        // Incognito: no frame kept, nothing restored.
        w.isRestorable = false
        w.delegate = self
        // The hosting controller sizes the window at its first layout: centered before, it would
        // grow from the left edge and run off to the right.
        host.view.layoutSubtreeIfNeeded()
        w.setContentSize(host.view.fittingSize)
        w.center()
        window = w
        NSApp.activate()
        w.makeKeyAndOrderFront(nil)
    }
}

/// **What the window shows**: three tabs, each switch with its sentence.
struct SettingsView: View {
    @Bindable var app: AppState
    let macs: OtherMacs
    /// Read again after an install: the link is the file system's, not the app's.
    @State private var command = SilicontrolCommand.state

    /// The state the window shows: the Mac that computes, whose switch and map these are.
    private var shown: AppState { MainWindow.shared.carrier.mirror ?? app }

    /// « Faster Rendering »: this Mac's preference, which rides on each job — the lender's own is never touched.
    private var fasterRendering: Binding<Bool> {
        Binding(get: { app.fasterRendering(for: app.identifier) }, set: { app.setFasterRendering($0, for: app.identifier) })
    }

    var body: some View {
        // A fixed frame: a TabView takes the size of no tab in particular, and a window that changed
        // size from one tab to the next would jump under the pointer. A tab longer than this scrolls.
        TabView {
            general.tabItem { Label("General", systemImage: "gearshape") }
            Form { OtherMacsSection(macs: macs) }
                .formStyle(.grouped)
                .tabItem { Label("Other Macs", systemImage: "macbook.and.iphone") }
            advanced.tabItem { Label("Advanced", systemImage: "slider.horizontal.3") }
        }
        .frame(width: 480, height: 540)
    }

    /// What one comes for: the speed switch, how images are compared, where the models are.
    private var general: some View {
        Form {
            Section {
                Toggle(isOn: fasterRendering) {
                    SettingLabel("Faster Rendering", "From 1024² up, the first steps at half size: faster, a slightly different image. Not with a LoRA, Detail or an edit.")
                }
            }
            Section {
                LabeledContent {
                    Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([app.library.root]) }
                        .disabled(!app.libraryReachable)
                } label: {
                    SettingLabel("Models", "Where the installed models and LoRAs live. Remove them in the Models window, not the Finder.")
                }
                Text(verbatim: app.library.displayPath).font(.caption).textSelection(.enabled)
                library
            }
        }
        .formStyle(.grouped)
    }

    /// Out of the everyday way: the capacity map that renders are planned from, the terminal, then what
    /// is still in beta.
    private var advanced: some View {
        Form {
            Section {
                LabeledContent {
                    Button("Measure This Mac…") {
                        MainWindow.shared.show()
                        shown.openDiagnostic()
                    }
                } label: {
                    SettingLabel("This Mac", "About a minute, five at most; done before the first render and after each update. A new measure applies to the next render.")
                }
                if let name = shown.mirror?.macName {
                    Text("This measures “\(name)”, the Mac that computes for you.").font(.caption)
                }
                Text(DiagnosticSheet.mapStatus(shown)).font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Section {
                LabeledContent {
                    Button(command == .installed ? "Reinstall…" : "Install…") {
                        SilicontrolCommand.install()
                        command = SilicontrolCommand.state
                    }
                } label: {
                    SettingLabel("Command Line", "silicontrol drives this app from a terminal, a script or an AI agent. See: silicontrol help")
                }
                Group {
                    switch command {
                    case .installed: Text("Installed: \(SilicontrolCommand.destination)").textSelection(.enabled)
                    case .otherCopy: Text("\(SilicontrolCommand.destination) points to another copy of Siliconed.")
                    case .absent: Text("Not installed.")
                    }
                }
                .font(.caption)
            }
            Section {
                Toggle(isOn: $app.developerMode) {
                    SettingLabel("More Solvers", "Adds the solvers Euler a, DPM++ 2M and UniPC to the DiT’s box and to the exploration’s axes. The default, Euler, is the one the models were distilled for.")
                }
                Toggle(isOn: $app.negativeLoRAStrengths) {
                    SettingLabel("Negative LoRA Strengths", "Lets a LoRA’s strength go from −1.5 to 1.5, for “slider” LoRAs. An ordinary LoRA is damaged by a negative strength.")
                }
            } header: {
                HStack(spacing: 8) {
                    BetaBadge(help: "Beta: still being settled, and may change.")
                    Text("Off by default; they may change.").font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
    }

    /// **Where the models live**: moved to another disk (the internal SSD is precious), opened there,
    /// brought back — each said with what happens (`AppState+Place.swift`).
    @ViewBuilder private var library: some View {
        if !app.libraryReachable {
            Text(app.libraryBlocker ?? "").font(.caption).foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        } else if app.libraryIsElsewhere {
            explanation("On another disk: what a render reads is copied to this Mac first, once per session, and removed when Siliconed quits. The disk must be connected to render.")
        } else {
            explanation("On this Mac. To free its disk, move the library to another one — an external disk, a network share: Siliconed then copies to this Mac only the model it renders, for the session.")
        }
        HStack {
            Button("Move Library…") { app.chooseLibraryDestination() }
                .disabled(!app.managementPossible)
                .help("Copies the library to the folder you choose, checks every file, then removes it from here. The licenses you accepted and this Mac’s settings stay on this Mac")
            Button("Open Library…") { app.chooseLibraryToOpen() }
                .disabled(app.forgeInProgress != nil || app.busy)
                .help("Uses a library that is already somewhere else, another disk’s for instance. Nothing is moved")
        }
        if app.libraryIsElsewhere {
            HStack {
                Button("Move Back to This Mac") { app.moveLibraryHome() }
                    .disabled(!app.managementPossible)
                Button("Use This Mac’s Library") { app.useLocalLibrary() }
                    .disabled(app.forgeInProgress != nil || app.busy)
                    .help("Renders with what this Mac holds — maybe nothing yet. The library on the other disk is not touched")
            }
        }
        if let title = app.forgeInProgress {
            Text(ForgeProgress.line(app.forgeJournal).map { "\(title) — \($0)" } ?? title)
                .font(.caption).monospacedDigit().foregroundStyle(.secondary)
        }
    }

    private func explanation(_ text: LocalizedStringKey) -> some View {
        Text(text).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
    }
}

/// **A setting's name, and under it what it does** — in the Settings' tabs and in the Other Macs section.
struct SettingLabel: View {
    let title: LocalizedStringKey
    let text: LocalizedStringKey
    init(_ title: LocalizedStringKey, _ text: LocalizedStringKey) { self.title = title; self.text = text }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
            Text(text).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }
}
