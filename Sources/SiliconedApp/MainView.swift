// **The app's window: the rack on the left, the image on the right**.
//
// On the left (`Sidebar.swift`), **the rack** (`Rack.swift`): the chain of the render drawn from
// `model.chain`, top to bottom in the direction of the signal — the prompt typed in its box, an edit's
// images, the encoders, the DiT with its model, size, seed, steps and LoRAs, the VAE —, wired by
// type, lit stage by stage while it runs; under it, what it will cost and « Generate ». On the
// right, the canvas: the step-by-step preview in the frame of the coming image, then the image, its
// details (the time and memory of each stage), and the history as a strip — in memory, gone with the
// app. The column follows the window: narrower, its boxes narrow and their rows wrap; too narrow,
// the split view folds it away. The sheets: the models (`ModelsSheet.swift`), a license to accept
// (`LicenseSheet.swift`), the diagnostic (`DiagnosticSheet.swift`).

import AppKit
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

struct MainView: View {
    @Bindable var app: AppState
    var toggleStatistics: () -> Void = {}

    var body: some View {
        NavigationSplitView {
            Sidebar(app: app)
                .navigationSplitViewColumnWidth(min: 320, ideal: 380, max: 520)
        } detail: {
            GraphCanvas(app: app)
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                let group = app.targets
                Button { app.copyImages(group) } label: { Label("Copy Image", systemImage: "doc.on.doc") }
                    .disabled(group.isEmpty)
                    .help(group.count > 1 ? "Copy the selected images (⇧⌘C)" : "Copy the image (PNG, with its metadata)")
                Button { app.export(group) } label: { Label("Save…", systemImage: "square.and.arrow.down") }
                    .disabled(group.isEmpty)
                    .help(group.count > 1 ? "Export the selected images to a folder (⇧⌘E)" : "Save the image (⌘S)")
                ShareLink(items: group.map(app.transferable), preview: { SharePreview($0.name) }) {
                    Label("Share…", systemImage: "square.and.arrow.up")
                }
                    .disabled(group.isEmpty)
                    .help("Share — AirDrop, Messages, Mail, Photos…")
                Button(action: toggleStatistics) {
                    Label("Statistics", systemImage: "gauge.with.dots.needle.33percent")
                }
                .help("Render statistics (⌥⌘I)")
            }
        }
        .sheet(isPresented: $app.managementOpen) { ModelManagement(app: app) }
        .sheet(item: $app.licenseRequest) { request in LicenseSheet(app: app, card: request.card) }
        .sheet(item: $app.hubOffer) { _ in HubSheet(app: app) }
        .sheet(isPresented: $app.diagnosticOpen) { DiagnosticSheet(app: app) }
        .overlay {
            if app.fullScreen, let image = app.selectedImage?.image { FullScreen(app: app, image: image) }
        }
        .animation(Motion.pick(.easeInOut(duration: 0.15)), value: app.fullScreen)
    }
}

// MARK: - The materials

/// **Reduce Motion** (System Settings ▸ Accessibility ▸ Display): the app's springs and fades are
/// dropped, the state changes at once.
enum Motion {
    static func pick(_ animation: Animation) -> Animation? {
        NSWorkspace.shared.accessibilityDisplayShouldReduceMotion ? nil : animation
    }
}

extension View {
    /// The chosen image leaves by a drag while it is fitted; zoomed in, a drag pans it.
    @ViewBuilder func draggableWhenFitted(_ fitted: Bool, _ item: PNGImage?, preview: CGImage) -> some View {
        if fitted, let item {
            draggable(item) { Image(decorative: preview, scale: 1).resizable().scaledToFit().frame(width: 160, height: 160) }
        } else {
            self
        }
    }
}

extension View {
    /// macOS 26's glass.
    func glass<S: Shape>(_ shape: S) -> some View {
        glassEffect(.regular, in: shape)
    }
}

/// The canvas background: a near-black gray in dark, a light gray in light — neutral, so that
/// the image alone carries the color.
private let canvasBackground = Color(nsColor: NSColor(name: nil) { appearance in
    appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        ? NSColor(white: 0.075, alpha: 1) : NSColor(white: 0.9, alpha: 1)
})

// MARK: - The canvas

/// **The canvas**: the image, the Prompt Lab's bar under it (unfolded, a glass plate over the image), the history in a column on the right.
private struct GraphCanvas: View {
    @Bindable var app: AppState

    var body: some View {
        HStack(spacing: 0) {
            stage
            if !app.history.isEmpty || app.explored != nil {
                Divider()
                Banner(app: app)
            }
        }
    }

    private var stage: some View {
        let lab = LabModel.shared
        return VStack(spacing: 0) {
            Group {
                if app.showsExploration, let run = app.explored {
                    ExplorationCanvas(app: app, run: run)
                } else {
                    SceneView(app: app)
                }
            }
            .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
            .clipped()
            // A click on the image folds the bar — not while the writer writes, which folding would stop.
            .simultaneousGesture(TapGesture().onEnded { if lab.unfolded, !lab.isWriting { lab.fold() } })
            // The folded line's room: it never covers the image. Unfolded, the bar rises over the image from there — in the
            // layout, it summed its height to the image's, and the window grew past the screen, the rack cut at both ends.
            Color.clear.frame(height: LabView.foldedRoom)
        }
        .overlay(alignment: .bottom) {
            LabView(model: lab).padding(.horizontal, 16).padding(.bottom, 12)
        }
        .animation(Motion.pick(.spring(duration: 0.3)), value: lab.unfolded)
        .background(canvasBackground)
        .overlay(alignment: .top) {
            if let toast = app.toast {
                Text(toast)
                    .font(.callout)
                    .padding(.horizontal, 16).padding(.vertical, 9)
                    .glass(Capsule())
                    .padding(.top, 14)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .animation(Motion.pick(.spring(duration: 0.3)), value: app.toast)
        // An image dropped on the canvas goes to the Prompt Lab, as any image: the role a newcomer takes there. A PNG
        // Siliconed wrote no longer brings its settings back by itself — « Reuse These Settings » is on its tile's menu.
        .dropDestination(for: DroppedImage.self) { items, _ in
            let images = items.filter { if case .file(let url) = $0 { !AppState.importable(url) } else { true } }
            guard !images.isEmpty else { return false }
            for image in images { lab.load(image) }
            lab.unfold()
            return true
        }
    }
}

// MARK: - Full screen

/// The chosen image over the whole window — the app's viewer, instead of a temporary file
/// opened in Preview (which would have kept it in its recent items). Escape, Space or a
/// click closes it; ← → walk along the strip.
private struct FullScreen: View {
    let app: AppState
    let image: CGImage

    var body: some View {
        ZStack {
            Color.black.opacity(0.94)
            Image(image, scale: 1, label: Text(verbatim: app.selectedEntry?.settings.prompt ?? ""))
                .resizable()
                .interpolation(.high)
                .scaledToFit()
                .padding(24)
            // The keys, invisible: the image is what VoiceOver reads (Esc and a click close it).
            Group {
                Button("Close") { app.fullScreen = false }
                    .keyboardShortcut(.cancelAction)
                Button("Previous") { app.chooseAdjacent(-1) }
                    .keyboardShortcut(.leftArrow, modifiers: [])
                Button("Next") { app.chooseAdjacent(1) }
                    .keyboardShortcut(.rightArrow, modifiers: [])
                Button("Close") { app.fullScreen = false }
                    .keyboardShortcut(.space, modifiers: [])
            }
            .opacity(0)
            .accessibilityHidden(true)
        }
        .ignoresSafeArea()
        .contentShape(Rectangle())
        .onTapGesture { app.fullScreen = false }
        .transition(.opacity)
    }
}
