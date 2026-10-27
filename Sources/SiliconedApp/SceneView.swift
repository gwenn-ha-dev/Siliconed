// The canvas's scene: the image being drawn or chosen, the comparison with the original
// and the details of what the engine did. Split from `MainView.swift`.

import AppKit
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

/// The image at the center: the preview of the running render in the frame of the coming image, or the
/// chosen image, or the invitation to begin.
struct SceneView: View {
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
    /// Where the pointer is over the image's zone — what a zoom keeps still (`ZoomAnchor`).
    @State private var pointer: CGPoint?
    /// The offset a pinch just settled, applied once its zoom lands (`onChange(of: app.zoom)`).
    @State private var settled: CGSize?
    @State private var scrollMonitor: Any?

    var body: some View {
        ZStack {
            if let crop = app.cropping {
                CropEditor(app: app, session: crop)
            } else if let job = app.currentJob, app.followsRender || app.history.isEmpty {
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
                        // The larger's size: the smaller is stretched to it, as the curtain of an enlargement shows it.
                        let big = pair.a.entry.width >= pair.b.entry.width ? pair.a.entry : pair.b.entry
                        CurtainView(app: app, a: pair.a.image, b: pair.b.image, width: big.width,
                                    height: big.height, tags: Curtain.tags(pair.a.entry, pair.b.entry))
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
        .onChange(of: app.zoom) { z1, z2 in
            let target = settled ?? ZoomAnchor.offset(offset, from: z1, to: z2, at: pointer, in: zone)
            settled = nil
            withAnimation(.spring(duration: 0.3)) { offset = z2 == 1 ? .zero : clamped(target) }
        }
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
            .offset(x: liveOffset.width + dragOffset.width, y: liveOffset.height + dragOffset.height)
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
            .onContinuousHover { phase in
                switch phase {
                case .active(let p): hovering = true; pointer = p
                case .ended: hovering = false; pointer = nil
                }
            }
            .zIndex(1)
    }

    /// The offset as drawn: during a pinch, moved so that the point under the fingers stays there.
    private var liveOffset: CGSize {
        pinchScale == 1 ? offset : ZoomAnchor.offset(offset, from: zoom, to: zoom * pinchScale, at: pointer, in: zone)
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
                Button { app.export(group) } label: { Image(systemName: "square.and.arrow.down") }
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
                EnlargeButton(app: app, entry: e)
                Button { app.beginCrop(e) } label: { Image(systemName: "crop") }
                    .disabled(app.cropRefusal(e) != nil)
                    .help(app.cropRefusal(e) ?? String(localized: "Crop: a new image, the original stays (C)"))
                if app.canCompareOriginal(e) {
                    Button { app.comparesOriginal.toggle() } label: {
                        Image(systemName: "square.lefthalf.filled")
                            .foregroundStyle(app.comparesOriginal ? Color.accentColor : .primary)
                    }
                    .help(app.comparesOriginal ? (e.enlarged == nil ? "Show the edited image alone (Esc)" : "Show the enlarged image alone (Esc)")
                                               : e.enlarged == nil ? "Compare with the original: the image edited, under a curtain"
                                               : "Compare with the original, merely stretched to this size, under a curtain")
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
                    .foregroundStyle(.secondary)
            }
            if let absent = app.libraryBlocker {
                // The library's disk is not there: no welcome, no « Install » — never another library.
                Image(systemName: "externaldrive.badge.xmark")
                    .font(.system(size: 46, weight: .light))
                    .foregroundStyle(.secondary)
                Text(absent).font(.title3).foregroundStyle(.secondary)
                Text("Connect it to render. Settings can also go back to this Mac's library.")
                    .font(.callout).foregroundStyle(.secondary)
                Button("Settings…") { SettingsWindow.shared.show(app) }
                    .padding(.top, 4)
            } else if app.visibleReady.isEmpty {
                Welcome(app: app)
            } else if let family = app.familyToInstall {
                Text("\(app.card?.name ?? family.name) isn't installed").font(.title3).foregroundStyle(.secondary)
                Button("Install…") { app.installSelectedModel() }
                    .disabled(!app.managementPossible)
                    .padding(.top, 4)
            } else {
                Text("Describe an image, then press ⌘↩").font(.title3).foregroundStyle(.secondary)
                if app.editingPossible {
                    Text("Drop an image here to edit it.").font(.callout).foregroundStyle(.secondary)
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
                let z = min(max(zoom * value.magnification, 1), 8)
                let target = ZoomAnchor.offset(offset, from: zoom, to: z, at: pointer, in: zone)
                if z == app.zoom { offset = z == 1 ? .zero : clamped(target) } else { settled = target; app.zoom = z }
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
        // Empty while the marked images are decoded (`loadCompared`, off the main thread — a 4096²
        // enlargement takes a while): a stride by 0 would trap.
        let columns = max(1, items.count == 4 ? 2 : items.count)
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
            // Named subjects: what the model read, under what was typed — to understand a miss.
            if let model = r.modelPrompt {
                DisclosureGroup("Model Prompt") {
                    Text(model).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .font(.caption)
            }
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
                if r.editing { rowLine("References", String(r.references - r.subjectImages)) }
                if r.subjectImages > 0 { rowLine("Subject images", String(r.subjectImages)) }
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
                Text(r.neuralEngine ? "fp32, part of the DiT in fp16 on the Neural Engine · same seed, same bits"
                                    : "fp32 end to end · same seed, same bits")
                    .font(.caption).foregroundStyle(.secondary)
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
