// **The app's window: the rack on the left, the image on the right**.
//
// On the left (`Sidebar.swift`), **the rack** (`Rack.swift`): the chain of the render drawn from
// `model.chain`, top to bottom in the direction of the signal — the prompt typed in its box, an edit's
// images, the encoders, the DiT with its model, size, seed, steps and LoRAs, the latent, the VAE, the
// image —, wired by type, lit stage by stage while it runs; under it, what it will cost and
// « Generate ». On the right, the canvas: the step-by-step preview in the frame of the coming image,
// then the image, its details (the time and memory of each stage), and the history as a strip — in
// memory, gone with the app. The column follows the window: narrower, its boxes narrow and their
// rows wrap; too narrow, the split view folds it away. The sheets: the models (`ModelsSheet.swift`),
// a license to accept (`LicenseSheet.swift`), the diagnostic (`DiagnosticSheet.swift`).

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
                Button { app.openDiagnostic() } label: {
                    Label("Report My Configuration…", systemImage: "stethoscope")
                }
                .help("Measure this Mac, then copy the report as JSON or open it as a GitHub issue")
            }
        }
        .sheet(isPresented: $app.managementOpen) { ModelManagement(app: app) }
        .sheet(item: $app.licenseRequest) { request in LicenseSheet(app: app, card: request.card) }
        .sheet(isPresented: $app.diagnosticOpen) { DiagnosticSheet(app: app) }
        .overlay {
            if app.fullScreen, let image = app.selectedImage?.image { FullScreen(app: app, image: image) }
        }
        .animation(.easeInOut(duration: 0.15), value: app.fullScreen)
    }
}

// MARK: - The materials

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
    /// macOS 26's glass where it exists, a material elsewhere (macOS 15).
    @ViewBuilder func glass<S: Shape>(_ shape: S) -> some View {
        if #available(macOS 26, *) {
            glassEffect(.regular, in: shape)
        } else {
            background(.regularMaterial, in: shape)
        }
    }
}

/// The canvas background: a near-black gray in dark, a light gray in light — neutral, so that
/// the image alone carries the color.
private let canvasBackground = Color(nsColor: NSColor(name: nil) { appearance in
    appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        ? NSColor(white: 0.075, alpha: 1) : NSColor(white: 0.9, alpha: 1)
})

// MARK: - The canvas

private struct GraphCanvas: View {
    @Bindable var app: AppState

    var body: some View {
        VStack(spacing: 0) {
            if app.showsExploration, let run = app.explored {
                ExplorationCanvas(app: app, run: run)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                SceneView(app: app)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            if !app.history.isEmpty || app.explored != nil {
                Banner(app: app)
            }
        }
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
        .animation(.spring(duration: 0.3), value: app.toast)
        // A PNG Siliconed wrote brings back its settings — the canvas is where an image is opened.
        // Any other image keeps the canvas's meaning: an edit's reference when the model edits
        // (to edit a Siliconed image, drop it on the rack's « Edit » box, or « Edit This Image »).
        .dropDestination(for: URL.self) { urls, _ in
            let images = urls.filter { !AppState.importable($0) }
            if images.count == 1, app.restoreSettings(fromPNG: images[0]) { return true }
            guard app.editingPossible, !images.isEmpty else {
                if images.count == 1, images[0].pathExtension.lowercased() == "png" {
                    app.flash(String(localized: "No Siliconed settings in this image"))
                }
                return false
            }
            app.addReferences(images)
            return true
        }
    }
}

/// The image at the center: the preview of the running render in the frame of the coming image, or the
/// chosen image, or the invitation to begin.
private struct SceneView: View {
    @Bindable var app: AppState

    // Pinch (or ⌘+ ⌘−) to zoom, drag to pan, double-click (or ⌘0) to return. The zoom is the app's,
    // for the menu to reach it.
    private var zoom: CGFloat { app.zoom }
    @State private var offset: CGSize = .zero
    @GestureState private var pinchScale: CGFloat = 1
    @GestureState private var dragOffset: CGSize = .zero
    @State private var details = false
    /// The image's zone, the image's size in it at zoom 1, and whether the pointer is over it: what
    /// the two-finger scroll needs.
    @State private var zone: CGSize = .zero
    @State private var shown: CGSize = .zero
    @State private var hovering = false
    @State private var scrollMonitor: Any?

    var body: some View {
        ZStack {
            if let job = app.currentJob, app.followsRender || app.history.isEmpty {
                inProgress(job)
            } else if app.showsOriginal, let original = app.original, let selected = app.selectedImage,
                      let entry = app.selectedEntry {
                VStack(spacing: 14) {
                    CurtainView(app: app, a: original.image, b: selected.image, width: entry.width, height: entry.height,
                                tags: (String(localized: "Original"), String(localized: "Edited")))
                    bar(entry)
                }
                .padding(28)
            } else if !app.compared.isEmpty, let entry = app.selectedEntry {
                VStack(spacing: 14) {
                    if let pair = Curtain.pair(app), Curtain.shared.preferred {
                        CurtainView(app: app, a: pair.a.image, b: pair.b.image, width: pair.a.entry.width,
                                    height: pair.a.entry.height, tags: Curtain.tags(pair.a.entry, pair.b.entry))
                    } else {
                        Comparison(app: app)
                    }
                    bar(entry)
                }
                .padding(28)
            } else if let selected = app.selectedImage, let entry = app.selectedEntry {
                VStack(spacing: 14) {
                    image(selected.image)
                    bar(entry)
                }
                .padding(28)
            } else {
                empty
            }
        }
        .onChange(of: app.selection) { app.zoom = 1; offset = .zero }
        .onChange(of: app.inProgress) { app.zoom = 1; offset = .zero }
        .onChange(of: app.zoom) { _, z in withAnimation(.spring(duration: 0.3)) { offset = z == 1 ? .zero : clamped(offset) } }
        .onAppear(perform: watchScroll)
        .onDisappear { scrollMonitor.map(NSEvent.removeMonitor); scrollMonitor = nil }
    }

    /// **Two fingers move the image in its zone** (a mouse wheel too): the trackpad's scroll, caught
    /// while the pointer is over the image's zone; elsewhere — the rack's column — it scrolls as usual.
    private func watchScroll() {
        guard scrollMonitor == nil else { return }
        scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
            guard hovering, app.currentJob == nil || !app.followsRender, app.selectedImage != nil else { return event }
            let k: CGFloat = event.hasPreciseScrollingDeltas ? 1 : 10
            offset = clamped(CGSize(width: offset.width + event.scrollingDeltaX * k,
                                    height: offset.height + event.scrollingDeltaY * k))
            return nil
        }
    }

    /// The image may leave its place, never its zone: a quarter of the zone beyond its edges at most.
    private func clamped(_ o: CGSize) -> CGSize {
        let x = max(0, (shown.width * zoom - zone.width) / 2) + zone.width / 4
        let y = max(0, (shown.height * zoom - zone.height) / 2) + zone.height / 4
        return CGSize(width: min(max(o.width, -x), x), height: min(max(o.height, -y), y))
    }

    // ── During the render ──

    /// The frame already has the shape of the coming image; each step's preview is drawn in it, and
    /// the progress floats below.
    private func inProgress(_ job: Job) -> some View {
        let l = CGFloat(job.settings.format.width), h = CGFloat(job.settings.format.height)
        return VStack(spacing: 14) {
            ZStack {
                if let preview = app.preview {
                    Image(decorative: preview, scale: 1)
                        .resizable()
                        .interpolation(.medium)
                        .transition(.opacity)
                } else {
                    Waiting(stage: app.stage)
                }
            }
            .aspectRatio(l / h, contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .shadow(color: .black.opacity(0.35), radius: 18, y: 6)
            .frame(maxWidth: l, maxHeight: h)
            .animation(.easeInOut(duration: 0.25), value: app.preview == nil)
            ProgressIndicator(app: app)
        }
        .padding(28)
    }

    // ── The chosen image ──

    private func image(_ cg: CGImage) -> some View {
        Image(decorative: cg, scale: 1)
            .resizable()
            .interpolation(.high)
            .scaledToFit()
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .shadow(color: .black.opacity(0.35), radius: 18, y: 6)
            .frame(maxWidth: CGFloat(cg.width), maxHeight: CGFloat(cg.height))
            .scaleEffect(zoom * pinchScale)
            .offset(x: offset.width + dragOffset.width, y: offset.height + dragOffset.height)
            .gesture(zoomGesture)
            .simultaneousGesture(dragGesture)
            .onTapGesture(count: 2) { withAnimation(.spring(duration: 0.3)) { app.zoom = 1; offset = .zero } }
            .onTapGesture { MainWindow.shared.releaseTextFocus() }
            .draggableWhenFitted(zoom == 1, app.selectedEntry.map(app.transferable), preview: cg)
            .contextMenu { if let e = app.selectedEntry { MenuEntry(app: app, entry: e) } }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .onGeometryChange(for: CGSize.self) { $0.size } action: { size in
                zone = size
                let w = CGFloat(cg.width), h = CGFloat(cg.height)
                let scale = min(size.width / w, size.height / h, 1)
                shown = CGSize(width: w * scale, height: h * scale)
            }
            .contentShape(Rectangle())
            .onHover { hovering = $0 }
            .zIndex(1)
    }

    /// Under the image: what it is, and what is done with it — or, with several marked, what is done
    /// with all of them.
    @ViewBuilder private func bar(_ e: Entry) -> some View {
        let group = app.targets
        HStack(spacing: 12) {
            if group.count > 1 {
                Text("\(group.count) images selected").font(.callout).monospacedDigit()
                    .frame(minWidth: 160, alignment: .leading)
                Divider().frame(height: 22)
                if app.compared.count == 2 {
                    CurtainSwitch(app: app)
                    Divider().frame(height: 22)
                }
                Button { app.export(group) } label: { Image(systemName: "square.and.arrow.down.on.square") }
                    .help("Export the selected images to a folder (⇧⌘E)")
                ShareLink(items: group.map(app.transferable), preview: { SharePreview($0.name) }) {
                    Image(systemName: "square.and.arrow.up")
                }
                .help("Share the selected images")
                Button { app.copyImages(group) } label: { Image(systemName: "doc.on.doc") }
                    .help("Copy the selected images (⇧⌘C)")
                Button { app.delete(group) } label: { Image(systemName: "trash") }
                    .help("Remove the selected images from the history (⌫, undo with ⌘Z)")
                Button { app.unmark() } label: { Image(systemName: "xmark.circle") }
                    .help("Keep only the image shown (Esc)")
            } else {
                VStack(alignment: .leading, spacing: 1) {
                    Text(e.settings.prompt).font(.callout).lineLimit(1).truncationMode(.tail)
                    Text(e.legend).font(.caption).monospacedDigit().foregroundStyle(.secondary)
                }
                .frame(maxWidth: 420, alignment: .leading)
                Divider().frame(height: 22)
                Button { app.redo(e) } label: { Image(systemName: "arrow.clockwise") }
                    .help("Redo: same settings, same seed (⌘R)")
                if e.grid == nil {
                    Menu { VariationButtons(app: app, entry: e) } label: { Image(systemName: "square.on.square.dashed") }
                        .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
                        .disabled(!app.canVary(e))
                        .help("Variations: cousins of this image, same scene, slightly different — as many as the rack's Images")
                }
                if app.canCompareOriginal(e) {
                    Button { app.comparesOriginal.toggle() } label: {
                        Image(systemName: "square.lefthalf.filled")
                            .foregroundStyle(app.comparesOriginal ? Color.accentColor : .primary)
                    }
                    .help(app.comparesOriginal ? "Show the edited image alone (Esc)"
                                               : "Compare with the original: the image edited, under a curtain")
                }
                Button { details.toggle() } label: { Image(systemName: "info.circle") }
                    .help("Image details")
                    .popover(isPresented: $details, arrowEdge: .top) { Details(app: app, entry: e) }
                Button { app.fullScreen = true } label: { Image(systemName: "arrow.up.left.and.arrow.down.right") }
                    .help("Full screen (Space)")
            }
        }
        .buttonStyle(.borderless)
        .imageScale(.large)
        .padding(.horizontal, 16).padding(.vertical, 8)
        .glass(Capsule())
        .zIndex(2)
        .animation(.easeInOut(duration: 0.15), value: group.count > 1)
    }

    // ── Nothing yet ──

    private var empty: some View {
        VStack(spacing: 12) {
            if !app.visibleReady.isEmpty {
                Image(systemName: app.missing == nil ? "sparkles" : "shippingbox")
                    .font(.system(size: 46, weight: .light))
                    .foregroundStyle(.tertiary)
            }
            if app.visibleReady.isEmpty {
                Welcome(app: app)
            } else if let family = app.familyToInstall {
                Text("\(app.card?.name ?? family.name) isn't installed").font(.title3).foregroundStyle(.secondary)
                Button("Install…") { app.installSelectedModel() }
                    .disabled(!app.managementPossible)
                    .padding(.top, 4)
            } else {
                Text("Describe an image, then press ⌘↩").font(.title3).foregroundStyle(.secondary)
                if app.editingPossible {
                    Text("Drop an image here to edit it.").font(.callout).foregroundStyle(.tertiary)
                }
            }
        }
        .multilineTextAlignment(.center)
        .padding()
    }

    private var zoomGesture: some Gesture {
        MagnifyGesture()
            .updating($pinchScale) { value, state, _ in state = value.magnification }
            .onEnded { value in
                app.zoom = min(max(zoom * value.magnification, 1), 8)
                if app.zoom == 1 { offset = .zero }
            }
    }

    private var dragGesture: some Gesture {
        DragGesture()
            .updating($dragOffset) { value, state, _ in if zoom > 1 { state = value.translation } }
            .onEnded { value in
                guard zoom > 1 else { return }
                offset.width += value.translation.width
                offset.height += value.translation.height
            }
    }
}

/// **Several images side by side** — two to four marked in the strip, to choose between seeds or
/// settings: each fitted in its cell, its seed under it. A click keeps that one alone.
private struct Comparison: View {
    let app: AppState

    var body: some View {
        let items = app.compared
        let columns = items.count == 4 ? 2 : items.count
        Grid(horizontalSpacing: 14, verticalSpacing: 14) {
            ForEach(Array(stride(from: 0, to: items.count, by: columns)), id: \.self) { start in
                GridRow {
                    ForEach(items[start..<min(start + columns, items.count)], id: \.entry.id) { item in
                        cell(item.entry, item.image)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func cell(_ e: Entry, _ image: CGImage) -> some View {
        VStack(spacing: 6) {
            Image(decorative: image, scale: 1)
                .resizable()
                .interpolation(.high)
                .scaledToFit()
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(Color.accentColor, lineWidth: e.id == app.selectedEntry?.id ? 2 : 0))
                .shadow(color: .black.opacity(0.3), radius: 12, y: 4)
                .frame(maxWidth: CGFloat(e.width), maxHeight: CGFloat(e.height))
            Text(verbatim: e.legend).font(.caption).monospacedDigit().foregroundStyle(.secondary).lineLimit(1)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        .onTapGesture { app.show(e.id) }
        .draggable(app.transferable(e)) {
            Image(decorative: e.vignette, scale: 1).resizable().scaledToFit().frame(width: 96, height: 96)
        }
        .contextMenu { MenuEntry(app: app, entry: e) }
        .help("Click to keep this one alone")
    }
}

/// **The first launch**: nothing comes with the app, so the canvas says it in three steps and offers
/// the two models side by side — what each does, its license, its place on disk. « Install… » opens
/// the Models sheet on that model's license: nothing is downloaded before it is accepted.
private struct Welcome: View {
    let app: AppState
    /// The version chosen on each card (`Family.preselectedVariant` until the user picks another:
    /// Standard, Light on a Mac of 8 GB).
    @State private var choice: [Family: Variant] = [:]

    var body: some View {
        VStack(spacing: 22) {
            VStack(spacing: 8) {
                Text("Welcome to Siliconed").font(.largeTitle.weight(.semibold))
                Text("Choose a model to begin. Its license and the space it takes are shown before anything is downloaded.")
                    .font(.callout).foregroundStyle(.secondary)
                    .frame(maxWidth: 460)
            }
            HStack(alignment: .top, spacing: 16) {
                ForEach([Family.zImage, .qwenImage21], id: \.self) { card($0) }
            }
            if let title = app.forgeInProgress {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(title).font(.callout).foregroundStyle(.secondary)
                }
            } else {
                Button("All Models…") { app.managementOpen = true }.buttonStyle(.link)
            }
        }
        .multilineTextAlignment(.center)
    }

    private func card(_ family: Family) -> some View {
        let model = ModelCard.of(family)
        let variant = choice[family] ?? family.preselectedVariant()
        let size = family.installedSize(variant)
        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: family == .qwenImage21 ? "wand.and.stars" : "sparkles")
                    .font(.title2).foregroundStyle(Color.accentColor)
                Spacer()
                Text(verbatim: model.licenseName).font(.caption).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.tail)
            }
            Text(verbatim: model.name).font(.title3.weight(.semibold))
            Text(family == .qwenImage21
                 ? "Generates, and edits a photo by instruction: “change her jacket to red”."
                 : "Fast and photographic. The one to start with.")
                .font(.callout).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            // Standard, Compact or Light, each with its space on disk (`Family.installedSize`): the rack
            // shows another figure, the memory a render needs (`ModelCard.memoryNeed`).
            if family.variants.count > 1 {
                VersionChoice(family: family, variant: Binding(get: { variant }, set: { choice[family] = $0 }))
                    .font(.caption)
            }
            HStack {
                if family.variants.count == 1 {
                    Label("\(Figures.disk(size.dit + size.encoder + size.components)) on disk", systemImage: "internaldrive")
                        .font(.caption).foregroundStyle(.secondary)
                        .help("The space it takes on disk once installed")
                }
                Spacer()
                Button("Install…") { app.offerInstall(family, variant: variant) }
                    .buttonStyle(.borderedProminent)
                    .disabled(!app.managementPossible)
            }
        }
        .multilineTextAlignment(.leading)
        .padding(16)
        .frame(width: 270, height: family.variants.count > 1 ? 270 : 200)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Palette.surface))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Palette.hairline))
    }
}

/// Before the first preview: the frame of the coming image, the wheel and the current stage. No
/// animated glint: it duplicated the wheel, and redrew the view at 30 frames/s during the computation.
private struct Waiting: View {
    let stage: Engine.Stage?

    var body: some View {
        ZStack {
            Color.primary.opacity(0.06)
            VStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(stage?.label ?? String(localized: "Preparing…"))
                    .font(.callout).foregroundStyle(.secondary)
            }
        }
    }
}

/// **The progress of the running render**: the stage, the step, the batch, the remaining time, one segment
/// per DiT evaluation, and "Stop".
private struct ProgressIndicator: View {
    let app: AppState

    var body: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 7) {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text(label).font(.callout).lineLimit(1)
                    Spacer(minLength: 12)
                    if app.estimatedRemaining != nil {
                        TimelineView(.periodic(from: .now, by: 1)) { context in
                            Text("\(Figures.duration(app.remaining(at: context.date) ?? 0)) left")
                                .font(.callout).monospacedDigit().foregroundStyle(.secondary)
                                .contentTransition(.numericText())
                        }
                    }
                }
                if app.stage == .denoising, app.totalSteps > 0, app.totalSteps <= 60 {
                    HStack(spacing: 3) {
                        ForEach(1...app.totalSteps, id: \.self) { i in
                            Capsule()
                                .fill(i <= app.currentStep ? Palette.live : Palette.well)
                                .frame(height: 4)
                        }
                    }
                    .animation(.easeOut(duration: 0.3), value: app.currentStep)
                } else {
                    ProgressView(value: app.fraction).progressViewStyle(.linear)
                }
            }
            .frame(width: 360)
            Button(role: .destructive) { app.cancel() } label: {
                Image(systemName: "stop.fill")
            }
            .buttonStyle(.borderless)
            .imageScale(.large)
            .help("Stop (⌘.)")
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
        .glass(RoundedRectangle(cornerRadius: 16))
    }

    private var label: String {
        var pieces: [String] = []
        if let t = app.currentJob { pieces.append(t.settings.modelName) }
        if app.batchSize > 1 {
            pieces.append(String(localized: "image \(app.batchImage + 1)/\(app.batchSize)"))
        }
        if app.stage == .denoising, app.totalSteps > 0 {
            pieces.append(String(localized: "step \(app.currentStep)/\(app.totalSteps)"))
        } else {
            pieces.append(app.stage?.label ?? String(localized: "Preparing…"))
        }
        return pieces.joined(separator: " · ")
    }
}

// MARK: - An image's details

/// **What the engine did for this image**: its settings, and the time and memory of each
/// stage — the figures the CLI prints, for each history image.
private struct Details: View {
    let app: AppState
    let entry: Entry

    var body: some View {
        let r = entry.settings, c = entry.timings, m = entry.footprints
        VStack(alignment: .leading, spacing: 12) {
            Text(r.prompt).font(.callout).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 5) {
                rowLine("Model", r.modelName)
                rowLine("Size", "\(entry.width) × \(entry.height)")
                rowLine("Seed", String(entry.seed))
                if let g = entry.grid {
                    // A sheet: its cells, and what they cost together (times summed, peaks the highest).
                    rowLine("Grid", String(localized: "\(g.grid.columns) × \(g.grid.rows), cells of \(r.format.width) × \(r.format.height)"))
                    if g.empty > 0 { rowLine("Empty cells", String(g.empty)) }
                } else {
                    rowLine("Evaluations", String(localized: "\(entry.evaluations) of \(r.steps) steps"))
                }
                if r.editing { rowLine("References", String(r.references)) }
                // The variation seeds, in the order they turned the noise, each with its strength.
                if r.isVariation {
                    rowLine("Variation", r.variations.map { "\($0.seed) · \(VariationWords.label($0.strength))" }
                        .joined(separator: " → "))
                }
                ForEach(r.loras) { l in
                    rowLine("LoRA", "\((l.path as NSString).lastPathComponent) · \(Figures.strength(l.strength))")
                }
                Divider().gridCellColumns(3)
                GridRow {
                    Text("Stage").foregroundStyle(.secondary)
                    Text("Time").foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                    Text("Memory (peak)").foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                }
                .font(.caption)
                stage("Text", c.text, m.encoder)
                if c.encoding > 0 { stage("Image", c.encoding, m.encoding) }
                stage("DiT", c.denoising, m.denoising)
                stage("VAE", c.decoding, m.end)
                GridRow {
                    Text("Total").fontWeight(.semibold)
                    Text(Figures.seconds(c.total)).fontWeight(.semibold).monospacedDigit()
                    Text(verbatim: "")
                }
            }
            .font(.callout)
            HStack {
                Text("fp32 end to end · same seed, same bits")
                    .font(.caption).foregroundStyle(.tertiary)
                Spacer()
                Menu {
                    Button("Copy Prompt") { copy(r.prompt) }
                    Button("Copy Seed") { copy(String(entry.seed)) }
                    Divider()
                    Button("Reuse These Settings") { app.restoreSettings(entry) }
                } label: { Image(systemName: "ellipsis.circle") }
                .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
                .help("Copy the prompt or the seed, or reuse the settings")
            }
        }
        .padding(16)
        .frame(width: 380)
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        app.flash(String(localized: "Copied"))
    }

    private func rowLine(_ title: LocalizedStringKey, _ value: String) -> some View {
        GridRow {
            Text(title).foregroundStyle(.secondary)
            Text(value).textSelection(.enabled).gridCellColumns(2).lineLimit(2).truncationMode(.middle)
        }
    }

    private func stage(_ title: LocalizedStringKey, _ seconds: Double, _ bytes: Int) -> some View {
        GridRow {
            Text(title)
            Text(Figures.seconds(seconds)).monospacedDigit()
            Text(bytes > 0 ? Figures.memory(bytes) : "—").monospacedDigit().foregroundStyle(.secondary)
        }
    }
}

// MARK: - The history

private struct Banner: View {
    @Bindable var app: AppState

    var body: some View {
        HStack(spacing: 0) {
            ScrollViewReader { reader in
            ScrollView(.horizontal) {
                LazyHStack(spacing: 8) {
                    if let run = app.explored { ExplorationTile(app: app, run: run) }
                    // An exploration's cell shows its own preview in the grid: no second tile for it.
                    if let job = app.currentJob, app.explored?.jobs.contains(job.ordinal) != true {
                        LiveTile(app: app, job: job)
                    }
                    ForEach(app.history) { entry in
                        Thumbnail(app: app, entry: entry,
                                  marked: app.isMarked(entry.id) && !(app.inProgress && app.followsRender),
                                  shown: entry.id == app.selectedEntry?.id && app.marked.count > 1)
                            .id(entry.id)
                    }
                }
                .padding(.horizontal, 16)
            }
            .scrollIndicators(.never)
            .onChange(of: app.selection) { _, id in
                if let id { withAnimation { reader.scrollTo(id, anchor: .center) } }
            }
            }
            Divider().opacity(0.5)
            sizeControl
        }
        // The strip is as tall as its thumbnails, never more: the canvas keeps the rest.
        .frame(height: app.thumbnailSide + 20)
        .background(.black.opacity(0.12))
        .overlay(alignment: .top) { Divider().opacity(0.5) }
    }

    /// At the strip's end: how many images, and the thumbnails' size.
    private var sizeControl: some View {
        VStack(alignment: .trailing, spacing: 6) {
            Text(app.history.count == 1 ? String(localized: "1 image") : String(localized: "\(app.history.count) images"))
                .font(.caption).monospacedDigit().foregroundStyle(.secondary)
            HStack(spacing: 4) {
                Image(systemName: "photo").imageScale(.small)
                Slider(value: $app.thumbnailSide, in: 48...140).frame(width: 80).controlSize(.mini)
                Image(systemName: "photo").imageScale(.medium)
            }
            .foregroundStyle(.secondary)
            .help("Thumbnail size")
        }
        .padding(.horizontal, 14)
        .fixedSize(horizontal: true, vertical: false)
    }
}

/// **A history image in the strip**: a click shows it, ⌘-click and ⇧-click mark several (`AppState.click`),
/// a drag carries its PNG out — to the Finder, a message, another app. Marked, it wears the accent edge;
/// among several marked, the one shown wears it thicker.
private struct Thumbnail: View {
    let app: AppState
    let entry: Entry
    let marked: Bool
    let shown: Bool
    @State private var hover = false

    var body: some View {
        let side = app.thumbnailSide
        let l = Double(entry.width), h = Double(entry.height)
        Image(decorative: entry.vignette, scale: 1)
            .resizable()
            .interpolation(.medium)
            .scaledToFill()
            .frame(width: side * min(1, l / h), height: side * min(1, h / l))
            .clipShape(RoundedRectangle(cornerRadius: 5))
            .overlay(RoundedRectangle(cornerRadius: 5)
                .strokeBorder(marked ? Color.accentColor : Color.white.opacity(hover ? 0.35 : 0.12),
                              lineWidth: marked ? (shown ? 3.5 : 2.5) : 1))
            .opacity(marked || hover ? 1 : 0.8)
            .frame(height: side + 4)
            .contentShape(Rectangle())
            .onHover { hover = $0 }
            .onTapGesture {
                MainWindow.shared.releaseTextFocus()
                app.click(entry.id, modifiers: NSEvent.modifierFlags)
            }
            .draggable(app.transferable(entry)) {
                Image(decorative: entry.vignette, scale: 1).resizable().scaledToFit().frame(width: 96, height: 96)
            }
            .help(Text(verbatim: "\(entry.settings.prompt)\n\(entry.legend)"))
            .contextMenu { MenuEntry(app: app, entry: entry) }
            .accessibilityElement()
            .accessibilityLabel(Text(verbatim: entry.settings.prompt))
            .accessibilityValue(Text(verbatim: entry.legend))
            .accessibilityAddTraits(marked ? [.isButton, .isSelected] : .isButton)
            .animation(.easeOut(duration: 0.12), value: hover)
    }
}

/// **The running render, at the head of the strip**: its preview in the shape of the coming image,
/// its progress along the bottom. Chosen, the canvas follows the render again.
private struct LiveTile: View {
    let app: AppState
    let job: Job

    var body: some View {
        let side = app.thumbnailSide
        let l = Double(job.settings.format.width), h = Double(job.settings.format.height)
        let following = app.followsRender
        ZStack(alignment: .bottom) {
            Color.primary.opacity(0.08)
            if let p = app.preview {
                Image(decorative: p, scale: 1).resizable().interpolation(.medium).scaledToFill()
            } else {
                ProgressView().controlSize(.small).frame(maxHeight: .infinity)
            }
            GeometryReader { g in
                Capsule().fill(Palette.live)
                    .frame(width: max(4, g.size.width * app.fraction), height: 4)
                    .frame(maxHeight: .infinity, alignment: .bottom)
            }
            .padding(3)
        }
        .frame(width: side * min(1, l / h), height: side * min(1, h / l))
        .clipShape(RoundedRectangle(cornerRadius: 5))
        .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(following ? Palette.live : Color.white.opacity(0.12), lineWidth: following ? 2.5 : 1))
        .frame(height: side + 4)
        .contentShape(Rectangle())
        .onTapGesture { app.followsRender = true; app.showsExploration = false }
        .help("The render in progress")
    }
}

/// **The exploration, at the head of the strip**: its grid in small, how far it is; a click brings the
/// grid back to the canvas.
private struct ExplorationTile: View {
    let app: AppState
    let run: GridRun

    var body: some View {
        let side = app.thumbnailSide
        ZStack {
            Color.primary.opacity(0.08)
            Image(systemName: "square.grid.3x3").font(.system(size: side * 0.36)).foregroundStyle(.secondary)
        }
        .overlay(alignment: .bottomTrailing) {
            Text(verbatim: "\(run.entries.count)/\(run.jobs.count)").font(.caption2).monospacedDigit()
                .padding(4)
        }
        .frame(width: side, height: side)
        .clipShape(RoundedRectangle(cornerRadius: 5))
        .overlay(RoundedRectangle(cornerRadius: 5)
            .strokeBorder(app.showsExploration ? Palette.live : Color.white.opacity(0.12), lineWidth: app.showsExploration ? 2.5 : 1))
        .frame(height: side + 4)
        .contentShape(Rectangle())
        .onTapGesture { app.showsExploration = true }
        .help("The exploration's grid")
    }
}

/// **« Variations »'s two choices** — the bar under the image and the context menu.
private struct VariationButtons: View {
    let app: AppState
    let entry: Entry

    var body: some View {
        Button("Subtle Variations") { app.vary(entry, .subtle) }
        Button("Strong Variations") { app.vary(entry, .strong) }
    }
}

/// **Sending an image to be edited**: by the rack's model when it edits; otherwise by an installed model
/// that does — chosen in the rack on the way.
private struct EditButtons: View {
    let app: AppState
    let entry: Entry

    var body: some View {
        if app.editingPossible {
            Button("Edit This Image") { app.placeReference(entry) }
        } else if app.editors.count == 1, let model = app.editors.first {
            Button("Edit with \(model.name)") { app.edit(entry, with: model.id) }
        } else if !app.editors.isEmpty {
            Menu("Edit With") {
                ForEach(app.editors, id: \.id) { model in
                    Button(model.name) { app.edit(entry, with: model.id) }
                }
            }
        }
    }
}

/// An image's context menu (the displayed image like the thumbnails). On one of several marked images,
/// what concerns files acts on all of them, as in the Finder.
private struct MenuEntry: View {
    let app: AppState
    let entry: Entry

    var body: some View {
        let group = app.marked.count > 1 && app.marked.contains(entry.id) ? app.targets : [entry]
        if group.count > 1 {
            Button("Export \(group.count) Images…") { app.export(group) }
            Button("Copy \(group.count) Images") { app.copyImages(group) }
            ShareLink("Share…", items: group.map(app.transferable), preview: { SharePreview($0.name) })
            Divider()
            Button("Remove \(group.count) Images from History", role: .destructive) { app.delete(group) }
        } else {
            Button("Save…") { app.save(entry) }
            Button("Copy Image") { app.copyImage(entry) }
            ShareLink("Share…", item: app.transferable(entry), preview: SharePreview(app.exportName(entry)))
            Button("Full Screen") { app.show(entry.id); app.fullScreen = true }
            Divider()
            Button(app.busy ? "Redo (Same Seed), Queued" : "Redo (Same Seed)") { app.redo(entry) }
            if entry.grid == nil, entry.sketch == nil {
                Menu("Variations") { VariationButtons(app: app, entry: entry) }
                    .disabled(!app.canVary(entry))
            }
            Button(entry.grid == nil ? "Reuse These Settings" : "Reuse This Grid") { app.restoreSettings(entry) }
            if entry.grid == nil { EditButtons(app: app, entry: entry) }
            Divider()
            Button("Remove from History", role: .destructive) { app.delete(entry) }
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
            Image(decorative: image, scale: 1)
                .resizable()
                .interpolation(.high)
                .scaledToFit()
                .padding(24)
            Button("Close") { app.fullScreen = false }
                .keyboardShortcut(.cancelAction)
                .opacity(0)
            Button("Previous") { app.chooseAdjacent(-1) }
                .keyboardShortcut(.leftArrow, modifiers: [])
                .opacity(0)
            Button("Next") { app.chooseAdjacent(1) }
                .keyboardShortcut(.rightArrow, modifiers: [])
                .opacity(0)
            Button("Close") { app.fullScreen = false }
                .keyboardShortcut(.space, modifiers: [])
                .opacity(0)
        }
        .ignoresSafeArea()
        .contentShape(Rectangle())
        .onTapGesture { app.fullScreen = false }
        .transition(.opacity)
    }
}
