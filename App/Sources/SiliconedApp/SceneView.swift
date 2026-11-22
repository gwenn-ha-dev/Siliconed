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

    // Pinch (or a mouse's wheel, or ⌘+ ⌘−) to zoom, drag to pan, double-click (or ⌘0) to return. The zoom is the app's,
    // for the menu to reach it.
    private var zoom: CGFloat { app.zoom }
    @State private var offset: CGSize = .zero
    @GestureState private var pinchScale: CGFloat = 1
    @GestureState private var dragOffset: CGSize = .zero
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
                staged(bar: app.componentsAway ? nil : bar(entry)) {
                    CurtainView(app: app, a: original.image, b: selected.image, width: entry.width, height: entry.height,
                                tags: (String(localized: "Original"), String(localized: "Edited")))
                }
            } else if !app.compared.isEmpty, let entry = app.selectedEntry {
                staged(bar: app.componentsAway ? nil : bar(entry)) {
                    if let pair = Curtain.pair(app), Curtain.shared.preferred {
                        // The larger's size: the smaller is stretched to it, as the curtain of an enlargement shows it.
                        let big = pair.a.entry.width >= pair.b.entry.width ? pair.a.entry : pair.b.entry
                        CurtainView(app: app, a: pair.a.image, b: pair.b.image, width: big.width,
                                    height: big.height, tags: Curtain.tags(pair.a.entry, pair.b.entry, loraName: app.loraName))
                    } else {
                        Comparison(app: app)
                    }
                }
            } else if let selected = app.selectedImage, let entry = app.selectedEntry {
                staged(bar: app.componentsAway ? nil : bar(entry)) { image(selected.image) }
            } else {
                empty
            }
        }
        // The same zoom and place from one image to the next of its size: the same detail, compared from seed to seed.
        // Another size, or an image gone, starts fitted.
        .onChange(of: app.selection) { old, new in
            let size = { (id: Entry.ID?) in app.history.first { $0.id == id }.map { [$0.width, $0.height] } }
            if size(old) == nil || size(old) != size(new) { app.zoom = 1; offset = .zero }
        }
        // A render starting or ending refits the canvas only when the canvas follows it: an image of the history looked at
        // meanwhile keeps its zoom from one job of the queue to the next.
        .onChange(of: app.inProgress) { if app.followsRender { app.zoom = 1; offset = .zero } }
        .onChange(of: app.zoom) { z1, z2 in
            let target = settled ?? ZoomAnchor.offset(offset, from: z1, to: z2, at: pointer, in: zone)
            settled = nil
            withAnimation(Motion.pick(.spring(duration: 0.3))) { offset = z2 == 1 ? .zero : clamped(target) }
        }
        .onAppear(perform: watchScroll)
        .onDisappear { scrollMonitor.map(NSEvent.removeMonitor); scrollMonitor = nil }
    }

    /// **Two fingers move the image in its zone; a mouse's wheel zooms toward the pointer** (`Wheel`, issue #10), its horizontal
    /// wheel still moves it — caught while the pointer is over the image's zone; elsewhere (the rack's column) it scrolls as usual.
    private func watchScroll() {
        guard scrollMonitor == nil else { return }
        scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
            guard hovering, app.currentJob == nil || !app.followsRender, app.selectedImage != nil else { return event }
            guard let move = Wheel.zoomedOrMoved(event, app: app) else { return nil }
            offset = clamped(CGSize(width: offset.width + move.width, height: offset.height + move.height))
            return nil
        }
    }

    /// The image may leave its place, never its zone: a quarter of the zone beyond its edges at most.
    private func clamped(_ o: CGSize) -> CGSize {
        let x = max(0, (shown.width * zoom - zone.width) / 2) + zone.width / 4
        let y = max(0, (shown.height * zoom - zone.height) / 2) + zone.height / 4
        return CGSize(width: min(max(o.width, -x), x), height: min(max(o.height, -y), y))
    }

    /// **The image and what floats under it, never laid out together**: the image takes its zone — the canvas less a
    /// fixed band at the bottom, where the bar sits —, and the bar lies over the canvas, at its foot. A bar that changes
    /// height (the progress's steps, then its line; the progress giving way to the image's bar when the render ends) moves
    /// nothing; one taller than the band covers the image's edge, the image sliding under it. Without a bar (the image
    /// alone in the window), the band goes too.
    private func staged<Content: View>(bar: (some View)?, @ViewBuilder _ content: () -> Content) -> some View {
        content()
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.bottom, bar == nil ? 0 : Self.band)
            .overlay(alignment: .bottom) { bar }
            .padding(28)
    }

    /// The bottom band the bar sits in: its height (the image's bar and the progress both ≈ 46 points) and the gap above it.
    private static let band: CGFloat = 46 + 14

    // ── During the render ──

    /// The frame already has the shape of the coming image; each step's preview is drawn in it, and
    /// the progress floats below.
    private func inProgress(_ job: Job) -> some View {
        let l = CGFloat(job.settings.format.width), h = CGFloat(job.settings.format.height)
        return staged(bar: ProgressIndicator(app: app)) {
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
            .animation(Motion.pick(.easeInOut(duration: 0.25)), value: app.preview == nil)
        }
    }

    // ── The chosen image ──

    private func image(_ cg: CGImage) -> some View {
        Image(cg, scale: 1, label: Text(verbatim: app.selectedEntry?.settings.prompt ?? ""))
            .resizable()
            .interpolation(Zoom.interpolation(zoom * pinchScale, pixels: CGFloat(cg.width), fitted: shown.width))
            .scaledToFit()
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .shadow(color: .black.opacity(0.35), radius: 18, y: 6)
            .frame(maxWidth: CGFloat(cg.width), maxHeight: CGFloat(cg.height))
            .scaleEffect(zoom * pinchScale)
            .offset(x: liveOffset.width + dragOffset.width, y: liveOffset.height + dragOffset.height)
            .gesture(zoomGesture)
            .simultaneousGesture(dragGesture)
            .onTapGesture(count: 2) { withAnimation(Motion.pick(.spring(duration: 0.3))) { app.zoom = 1; offset = .zero } }
            .onTapGesture { MainWindow.shared.releaseTextFocus() }
            .draggableWhenFitted(zoom == 1, app.selectedEntry.map(app.transferable), preview: cg)
            .contextMenu { if let e = app.selectedEntry { MenuEntry(app: app, entry: e) } }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .onGeometryChange(for: CGSize.self) { $0.size } action: { size in
                zone = size
                let w = CGFloat(cg.width), h = CGFloat(cg.height)
                let scale = min(size.width / w, size.height / h, 1)
                shown = CGSize(width: w * scale, height: h * scale)
                app.zoomLimit = Zoom.limit(pixels: w, fitted: shown.width)
            }
            .contentShape(Rectangle())
            .onContinuousHover { phase in
                switch phase {
                case .active(let p): hovering = true; pointer = p
                case .ended: hovering = false; pointer = nil
                }
            }
            // The pointer's `.ended` is not delivered to a view that goes away under it (a curtain replaces the image):
            // without this the wheel would stay caught here.
            .onDisappear { hovering = false; pointer = nil }
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
                Button { Task { await app.export(group) } } label: { Image(systemName: "square.and.arrow.down") }
                    .help("Export the selected images to a folder (⇧⌘E)")
                    .accessibilityLabel(Text("Export Images…"))
                ShareLink(items: group.map(app.transferable), preview: { SharePreview($0.name) }) {
                    Image(systemName: "square.and.arrow.up")
                }
                .help("Share the selected images")
                .accessibilityLabel(Text("Share…"))
                Button { app.copyImages(group) } label: { Image(systemName: "doc.on.doc") }
                    .help("Copy the selected images (⇧⌘C)")
                    .accessibilityLabel(Text("Copy Image"))
                Button { app.delete(group) } label: { Image(systemName: "trash") }
                    .help("Remove the selected images from the history (⌫, undo with ⌘Z)")
                    .accessibilityLabel(Text("Remove from History"))
                Button { app.unmark() } label: { Image(systemName: "xmark.circle") }
                    .help("Keep only the image shown (Esc)")
                    .accessibilityLabel(Text("Keep Only This Image"))
            } else {
                // The legend gives way first when the bar lacks room: the buttons never do.
                VStack(alignment: .leading, spacing: 1) {
                    Text(e.settings.prompt).font(.callout).lineLimit(1).truncationMode(.tail)
                    Text(e.legend).font(.caption).monospacedDigit().foregroundStyle(Palette.secondary).lineLimit(1)
                }
                .frame(minWidth: 0, maxWidth: 420, alignment: .leading)
                .layoutPriority(-1)
                Divider().frame(height: 22)
                Button { app.redo(e) } label: { Image(systemName: "arrow.clockwise") }
                    .help("Render again: same settings, same seed (⌘R)")
                    .accessibilityLabel(Text("Render Again (Same Seed)"))
                // A titled label shown as its icon: with a bare symbol, VoiceOver could read the symbol's own name
                // (« Square On A Dashed Square ») whatever the menu's accessibility label.
                Menu { VariationButtons(app: app, entry: e) } label: {
                    Label("Variations", systemImage: "square.on.square.dashed").labelStyle(.iconOnly)
                }
                    .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
                    .help("Variations: cousins of this image, same scene, slightly different — as many as the rack’s Seeds")
                    .accessibilityLabel(Text("Variations"))
                    .unavailable(app.varyRefusal(e))
                EnlargeButton(app: app, entry: e)
                Button { app.beginCrop(e) } label: { Image(systemName: "crop") }
                    .help("Crop: a new image, the original stays (C)")
                    .accessibilityLabel(Text("Crop…"))
                    .unavailable(app.cropRefusal(e))
                if app.canCompareOriginal(e) {
                    Button { app.comparesOriginal.toggle() } label: {
                        Image(systemName: "square.lefthalf.filled")
                            .foregroundStyle(app.comparesOriginal ? Palette.live : .primary)
                    }
                    .accessibilityLabel(Text("Compare with Original"))
                    .accessibilityAddTraits(app.comparesOriginal ? .isSelected : [])
                    .help(app.comparesOriginal ? (e.enlarged == nil ? "Show the edited image alone (Esc)" : "Show the enlarged image alone (Esc)")
                                               : e.enlarged == nil ? "Compare with the original: the image edited, under a curtain"
                                               : "Compare with the original, merely stretched to this size, under a curtain")
                }
                Button { app.imageDetails.toggle() } label: {
                    Image(systemName: "info.circle").foregroundStyle(app.imageDetails ? Palette.live : .primary)
                }
                    .help("Image details (⌘I)")
                    .accessibilityLabel(Text("Image details"))
                    .popover(isPresented: $app.imageDetails, arrowEdge: .top) { ImageInspector(app: app) }
                Button { withAnimation(MainView.away) { app.componentsAway = true } } label: {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                }
                    .help("The image alone in the window (Space)")
                    .accessibilityLabel(Text("Move the Components Aside"))
                // **What is done with the image**: here only, the window's top band holding none of it.
                Divider().frame(height: 22)
                Button { app.copyImages(group) } label: { Image(systemName: "doc.on.doc") }
                    .help("Copy the image (PNG, with its metadata)")
                    .accessibilityLabel(Text("Copy Image"))
                Button { Task { await app.export(group) } } label: { Image(systemName: "square.and.arrow.down") }
                    .help("Save the image (⌘S)")
                    .accessibilityLabel(Text("Save…"))
                ShareLink(items: group.map(app.transferable), preview: { SharePreview($0.name) }) {
                    Image(systemName: "square.and.arrow.up")
                }
                .help("Share — AirDrop, Messages, Mail, Photos…")
                .accessibilityLabel(Text("Share…"))
            }
        }
        .buttonStyle(.borderless)
        .imageScale(.large)
        .padding(.horizontal, 16).padding(.vertical, 8)
        .glass(Capsule())
        .zIndex(2)
        .animation(Motion.pick(.easeInOut(duration: 0.15)), value: group.count > 1)
    }

    // ── Nothing yet ──

    private var empty: some View {
        VStack(spacing: 12) {
            if !app.visibleReady.isEmpty {
                Image(systemName: app.missing == nil ? "sparkles" : "shippingbox")
                    .font(.system(size: 46, weight: .light))
                    .foregroundStyle(Palette.secondary)
            }
            if let absent = app.libraryBlocker {
                // The library's disk is not there: no welcome, no « Install » — never another library.
                Image(systemName: "externaldrive.badge.xmark")
                    .font(.system(size: 46, weight: .light))
                    .foregroundStyle(Palette.secondary)
                Text(absent).font(.title3).foregroundStyle(Palette.secondary)
                Text("Connect it to render. Settings can also go back to this Mac’s library.")
                    .font(.callout).foregroundStyle(Palette.secondary)
                Button("Settings…") { SettingsWindow.shared.show() }
                    .padding(.top, 4)
            } else if app.visibleReady.isEmpty {
                Welcome(app: app)
            } else if let family = app.familyToInstall {
                Text("\(app.card?.name ?? family.name) isn’t installed").font(.title3).foregroundStyle(Palette.secondary)
                Button("Install…") { app.installSelectedModel() }
                    .unavailable(app.managementRefusal)
                    .padding(.top, 4)
            } else {
                Text("Describe an image, then press ⌘↩").font(.title3).foregroundStyle(Palette.secondary)
                if app.editingPossible {
                    Text("Drop an image here to edit it.").font(.callout).foregroundStyle(Palette.secondary)
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
                let z = min(max(zoom * value.magnification, 1), app.zoomLimit)
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
            Image(image, scale: 1, label: Text(verbatim: e.settings.prompt))
                .resizable()
                .interpolation(.high)
                .scaledToFit()
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(Palette.live, lineWidth: e.id == app.selectedEntry?.id ? 2 : 0))
                .shadow(color: .black.opacity(0.3), radius: 12, y: 4)
                .frame(maxWidth: CGFloat(e.width), maxHeight: CGFloat(e.height))
            Text(verbatim: e.legend).font(.caption).monospacedDigit().foregroundStyle(Palette.secondary).lineLimit(1)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isButton)
        .onTapGesture { app.show(e.id) }
        .draggable(app.transferable(e)) {
            Image(decorative: e.vignette, scale: 1).resizable().scaledToFit().frame(width: 96, height: 96)
        }
        .contextMenu { MenuEntry(app: app, entry: e) }
        .help("Click to keep this one alone")
    }
}

