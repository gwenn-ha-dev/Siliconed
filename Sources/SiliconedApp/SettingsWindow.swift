// **The app's settings, in one window** — each switch with what it does, said under it.
//
//     Siliconed ▸ Settings… (⌘,)
//     ┌───────────────────────────────────────────────────────────────┐
//     │ Faster Rendering   the first steps at half size, on            │
//     │ Developer Mode     for comparing and testing: solvers,         │
//     │                    ComfyUI's schedule                          │
//     │ Experimental       unfinished, off by default: Named Subjects, │
//     │                    Negative LoRA Strengths                     │
//     │ Comparison         curtain or side by side                     │
//     │ Other Macs         lend this one, compute on another           │
//     │ This Mac           the capacity map, measured only when asked  │
//     │ Command Line       silicontrol, linked into /usr/local/bin     │
//     │ Models             where the library is, shown in the Finder;  │
//     │                    moved to another disk, opened, brought back │
//     └───────────────────────────────────────────────────────────────┘
//
// - **Incognito**: an NSWindow of ours, as `MainWindow` — not SwiftUI's `Settings` scene, which would
//   keep its frame; not restorable. What it sets are the app's preferences (`AppState`'s header), the
//   same keys as before: a switch moved here keeps its value.
// - **A switch changes what it changes at once**, as from the menu it replaces: Developer Mode off takes
//   the solver, the ComfyUI schedule and a solver axis with it (`AppState.developerMode`).
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
        let host = NSHostingController(rootView: SettingsView(app: app, curtain: .shared, macs: MainWindow.shared.otherMacs))
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

/// **What the window shows**: six sections, each switch with its sentence.
struct SettingsView: View {
    @Bindable var app: AppState
    @Bindable var curtain: Curtain
    let macs: OtherMacs
    /// Read again after an install: the link is the file system's, not the app's.
    @State private var command = SilicontrolCommand.state

    var body: some View {
        // Two columns, so that the window stays short: what the app shows, then this Mac.
        HStack(alignment: .top, spacing: 0) {
            Form {
                Section {
                    Toggle(isOn: $app.fasterRendering) {
                        titled("Faster Rendering", "At 1024² and above, the first steps at half size: a quarter to a third faster, and a slightly different image. Not with a LoRA, Detail or an edit.")
                    }
                }
                Section {
                    Toggle(isOn: $app.developerMode) {
                        titled("Developer Mode", "For comparing and testing, not for everyday renders. It shows:")
                    }
                    bullets(["the solvers Euler a, DPM++ 2M and UniPC beside Euler, in the panel and as an axis of the exploration",
                             "ComfyUI's schedule on Qwen-Image-2.1, to compare with the official one"])
                }
                Section {
                    Toggle(isOn: Binding(get: { app.subjects.enabled },
                                         set: { app.subjects.enabled = $0; if !$0 { app.referenceMode = .images } })) {
                        titled("Named Subjects", "Name a person or a thing from a photo, then cite it as @name in an edit's instruction.")
                    }
                    Toggle(isOn: $app.negativeLoRAStrengths) {
                        titled("Negative LoRA Strengths", "LoRA sliders from −1.5 to 1.5. For « slider » LoRAs (age, detail…), trained to work both ways; on an ordinary LoRA, a negative strength mostly damages the image.")
                    }
                } header: {
                    titled("Experimental", "Unfinished: off by default, and may still change.")
                }
                Section {
                    Picker(selection: $curtain.preferred) {
                        Text("Curtain").tag(true)
                        Text("Side by Side").tag(false)
                    } label: {
                        titled("Comparison", "How two images are compared in the canvas; its toolbar switches too.")
                    }
                    .pickerStyle(.segmented)
                }
                OtherMacsSection(macs: macs)
            }
            .formStyle(.grouped)
            .frame(width: 440)
            Form {
                Section {
                    LabeledContent {
                        Button("Measure This Mac…") {
                            MainWindow.shared.show()
                            app.openDiagnostic()
                        }
                    } label: {
                        titled("This Mac", "About a minute, without any model: Siliconed measures this Mac before its first render and after each update, and chooses how it computes from it. Measured again here, it applies from the next launch.")
                    }
                    Text(DiagnosticSheet.mapStatus(app)).font(.caption)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Section {
                    LabeledContent {
                        Button(command == .installed ? "Reinstall…" : "Install…") {
                            SilicontrolCommand.install()
                            command = SilicontrolCommand.state
                        }
                    } label: {
                        titled("Command Line", "silicontrol drives this app from a terminal or a script — an AI agent can use it too: queue renders and explorations, read the queue and the history, save images. In a terminal: silicontrol help")
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
                    LabeledContent {
                        Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([app.library.root]) }
                            .disabled(!app.libraryReachable)
                    } label: {
                        titled("Models", "The models and LoRAs Siliconed installed. Remove a model from the Models window, not from the Finder.")
                    }
                    Text(verbatim: app.library.displayPath).font(.caption).textSelection(.enabled)
                    library
                }
            }
            .formStyle(.grouped)
            .frame(width: 440)
        }
        .fixedSize(horizontal: false, vertical: true)
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
                .help("Copies the library to the folder you choose, checks every file, then removes it from here. The licenses you accepted and this Mac's settings stay on this Mac")
            Button("Open Library…") { app.chooseLibraryToOpen() }
                .disabled(app.forgeInProgress != nil || app.busy)
                .help("Uses a library that is already somewhere else, another disk's for instance. Nothing is moved")
        }
        if app.libraryIsElsewhere {
            HStack {
                Button("Move Back to This Mac") { app.moveLibraryHome() }
                    .disabled(!app.managementPossible)
                Button("Use This Mac's Library") { app.useLocalLibrary() }
                    .disabled(app.forgeInProgress != nil || app.busy)
                    .help("Renders with what this Mac holds — maybe nothing yet. The library on the other disk is not touched")
            }
        }
        if let title = app.forgeInProgress {
            Text(ForgeProgress.line(app.forgeJournal).map { "\(title) — \($0)" } ?? title)
                .font(.caption).monospacedDigit().foregroundStyle(.secondary)
        }
    }

    /// A setting's name, and under it what it does.
    private func titled(_ title: LocalizedStringKey, _ text: LocalizedStringKey) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
            explanation(text)
        }
    }

    private func explanation(_ text: LocalizedStringKey) -> some View {
        Text(text).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
    }

    private func bullets(_ lines: [LocalizedStringKey]) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(lines.indices, id: \.self) { i in
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(verbatim: "•")
                    Text(lines[i])
                }
            }
        }
        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
    }
}
