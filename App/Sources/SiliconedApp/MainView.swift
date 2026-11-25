// **The app's window: three planes, each with one role**.
//
// - **The image has the window**: the canvas is the background, from edge to edge — the step-by-step preview in the frame
//   of the coming image, then the image. Fitted, it holds whole in the room the components leave, nothing over it; zoomed
//   in, it slides under them to the window's edges, and so does a series' grid. Space moves them aside: the columns and
//   the bar go, the band's few buttons stay (hiding the window's toolbar brought its title back, and kept it).
// - **Glass carries only what floats over the image**: the two columns, the top band's buttons, the bar under the image.
//   The canvas stays neutral — a colored surround would bias the reading of the image's colors. Glass composes again only
//   when something moves under it: a fitted image, a render's preview, never pass under it.
// - **On the left, the chain: where things are set** (`RackColumn.swift`, `Sidebar.swift`, `Rack.swift`). From the idea to
//   the button, top to bottom, one column: the text and the images (`LabBoxes.swift`), the encoders, the DiT with its
//   model, size, seed, steps and LoRAs, the VAE, wired by type, lit stage by stage while it runs; « Generate » pinned at
//   its foot. It folds, and its right edge drags.
// - **On the right, time**: the history as a column — in memory, gone with the app.
//
// What is read and not set is in the two inspectors (`Inspectors.swift`). The sheets: the models (`ModelsSheet.swift`), a
// license to accept (`LicenseSheet.swift`).

import AppKit
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

struct MainView: View {
    @Bindable var app: AppState

    /// A column folding, the components moving aside: very fast at first, a little slower at the very end, short in all.
    static let away = Motion.pick(.timingCurve(0.16, 0.9, 0.3, 1, duration: 0.32))

    var body: some View {
        let rack = RackColumnModel.shared
        GraphCanvas(app: app, rack: rack)
        .animation(Self.away, value: rack.folded)
        // **The top band**, the window having no bar (`MainWindow.show`): beside the lights what the rack's header held —
        // the fold, and where the chain computes —; on the right what is read and not set. An image's actions are under it
        // (`SceneView.bar`), so that the fitted image rises into the band.
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                let rack = RackColumnModel.shared
                Button { rack.toggle() } label: {
                    // The menu's words (View › Show Chain, Hide Chain): one name for one gesture.
                    Label { rack.folded ? Text("Show Chain") : Text("Hide Chain") } icon: { Image(systemName: "sidebar.left") }
                }
                .help(rack.folded ? "Show Chain (⌥⌘R)" : "Hide Chain (⌥⌘R)")
                // Over the history grid, or with the column in its own window, the fold has nothing to fold.
                .unavailable(rack.detached ? String(localized: "The chain is in its own window: Dock the Chain brings it back here")
                             : app.historyGrid ? String(localized: "The history grid takes the window: back to the image to show the chain") : nil)
                Button { rack.setDetached(!rack.detached) } label: {
                    Label { rack.detached ? Text("Dock the Chain") : Text("Detach the Chain") } icon: {
                        Image(systemName: rack.detached ? "rectangle.leadinghalf.inset.filled" : "macwindow.on.rectangle")
                    }
                }
                .help(rack.detached ? "Dock the Chain: back in the window, on the left"
                                    : "Detach the Chain: the text and the chain in a floating window of their own")
                ComputeOnMenu(app: app)
            }
            // A series' grid on the canvas: its controls at the top of the window, in the band.
            if let run = app.shownRun, !app.componentsAway, !app.historyGrid {
                ToolbarItem(placement: .principal) { SeriesBar(app: app, run: run) }
            }
            // Without the spacer the title hidden leaves the right-hand group beside the left one.
            ToolbarSpacer(.flexible)
            // While Core AI compiles, and only then: the one place the window says it (`CompilationMark`).
            if app.compilingCoreAI {
                ToolbarItem(placement: .primaryAction) { CompilationMark(app: app) }
            }
            // Its own capsule, not the button's: a name, not a control. The band's clear glass took the colour of the image under
            // it, and a pale sky washed the dusk gradient out: the capsule is the window's own night (or paper), nearly opaque,
            // the ground the gradient was drawn for, whatever passes beneath.
            ToolbarItem(placement: .primaryAction) {
                EditionMark()
            }
            .sharedBackgroundVisibility(.hidden)
            ToolbarSpacer(.fixed, placement: .primaryAction)
            ToolbarItem(placement: .primaryAction) {
                Button { MainWindow.shared.toggleChain() } label: {
                    Label("Chain Inspector", systemImage: "gauge.with.dots.needle.33percent")
                }
                .help("The chain stage by stage: module, weights, time, memory, GPU and Neural Engine (⌥⌘I)")
            }
        }
        .sheet(isPresented: $app.managementOpen) { ModelManagement(app: app) }
        .sheet(item: $app.hubOffer) { _ in HubSheet(app: app) }
        .sheet(item: $app.licenseRequest) { request in LicenseSheet(app: app, card: request.card) }
    }
}

// MARK: - Core AI compiling

/// **What Core AI compiles, in the window's band**, while it does — the one place it is said; the rack and the canvas say
/// only why no time is told. A preparation: its model, its size, the function it is at of its key's
/// (`AppState.compilingAhead`); a render: its own model and size. A click opens the Chain inspector, which says the rest
/// (the models after it, the seconds of each stage).
private struct CompilationMark: View {
    let app: AppState

    var body: some View {
        Button { MainWindow.shared.showChain() } label: {
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                Text(words).monospacedDigit().lineLimit(1)
            }
            .padding(.horizontal, 4)
        }
        .help("Core AI compiles for this Mac what the renders need, once. The Chain inspector says what.")
    }

    private var words: String {
        if let p = app.compilingAhead {
            return String(localized: "Core AI · \(app.modelName(p.model)) \(Figures.size(p.width, p.height)) · \(p.at)/\(p.of)")
        }
        return app.currentJob.map { String(localized: "Core AI · \($0.legend)") } ?? String(localized: "Core AI")
    }
}

// MARK: - The edition

/// **The edition's name, at the right of the top band**: the place in a serif italic, washed in the colors of an Atlantic
/// evening — the sea's teal into its dusk —, the word « edition » small and quiet after it. A name, not a control.
private struct EditionMark: View {
    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 5) {
            Text(verbatim: MainWindow.edition)
                .font(.system(size: 17, weight: .semibold, design: .serif).italic())
                .foregroundStyle(LinearGradient(colors: [Palette.editionSea, Palette.editionDusk],
                                                startPoint: .leading, endPoint: .trailing))
            Text(verbatim: "edition")
                .font(.system(size: 10, weight: .medium)).textCase(.uppercase).tracking(1.2)
                .foregroundStyle(Palette.secondary)
        }
        .fixedSize()
        .padding(.horizontal, 12).padding(.vertical, 6)
        .glassEffect(.regular.tint(Palette.canvas.opacity(0.85)), in: Capsule())
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(verbatim: MainWindow.editionLine))
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

    /// **A column's glass**, from the window's top edge — under the band's buttons — to its foot; without it (Settings ▸
    /// General), the column's own ink.
    func columnGlass(_ on: Bool) -> some View {
        background {
            if on { Color.clear.glassEffect(.regular, in: Rectangle()).ignoresSafeArea() } else { Palette.column.ignoresSafeArea() }
        }
    }
}

// MARK: - The canvas

/// **The canvas**: the image under the whole window; over it the chain on the left, the history on the right. Columns over
/// a canvas, no split view: in one, a column could only vanish whole, and its width was the view's.
private struct GraphCanvas: View {
    @Bindable var app: AppState
    let rack: RackColumnModel
    /// The top band's height: the room it takes from a fitted image.
    @State private var band: CGFloat = 0

    var body: some View {
        let away = app.componentsAway
        let history = !away && (!app.history.isEmpty || !app.series.isEmpty || app.busy)
        // **The room the components leave**, as the canvas's safe area: a fitted image is laid out in it, a zoomed one
        // and a scrolled grid pass beyond it, under the glass, to the window's edges.
        let room = EdgeInsets(top: away ? 0 : band, leading: away || rack.outOfWindow ? 0 : rack.width,
                              bottom: 0, trailing: history ? Banner.width(app) : 0)
        ZStack {
            Palette.canvas.ignoresSafeArea()
            stage.safeAreaPadding(room).ignoresSafeArea()
            HStack(spacing: 0) {
                RackColumn(app: app, model: rack, away: away)
                Spacer(minLength: 0)
                if history {
                    Banner(app: app).transition(.move(edge: .trailing))
                }
            }
        }
        .onGeometryChange(for: CGFloat.self) { $0.safeAreaInsets.top } action: { band = $0 }
        // **The way back from the image alone**: Space and Esc do it, this says so and does it with a click.
        .overlay(alignment: .bottom) {
            if away {
                Button { withAnimation(MainView.away) { app.componentsAway = false } } label: {
                    Label("Bring Back the Components", systemImage: "arrow.down.right.and.arrow.up.left")
                        .labelStyle(.iconOnly).imageScale(.large).padding(10)
                }
                .buttonStyle(.borderless)
                .glass(Circle())
                .padding(.bottom, 18)
                .help("Bring back the chain, the history and the bar (Space, Esc)")
                .transition(.opacity)
            }
        }
        .animation(MainView.away, value: away)
        // The history widened over the canvas, from the column's side; it narrows back the same way.
        .overlay {
            if app.historyGrid, !app.history.isEmpty {
                HistoryGrid(app: app)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
    }

    private var stage: some View {
        let lab = LabModel.shared
        return Group {
            if let run = app.shownRun {
                SeriesCanvas(app: app, run: run)
            } else {
                SceneView(app: app)
            }
        }
        .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
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
        // An image dropped on the canvas goes to the Images box, as any image: the role a newcomer takes there. A PNG
        // Siliconed wrote does not bring its settings back by itself — « Reuse These Settings » is on its tile's menu.
        .dropDestination(for: DroppedImage.self) { items, _ in
            // A `.safetensors` dropped anywhere is imported (model or LoRA), never read as an image.
            let files = items.compactMap { item -> URL? in if case .file(let url) = item, AppState.importable(url) { url } else { nil } }
            let imported = !files.isEmpty && app.perform(files)
            let items = items.filter { if case .file(let url) = $0 { !AppState.importable(url) } else { true } }
            guard !items.isEmpty else { return imported }
            for image in items { lab.load(image) }
            RackColumnModel.shared.unfold()
            return true
        }
    }
}
