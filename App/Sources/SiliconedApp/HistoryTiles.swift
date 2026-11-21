// **Time, on the right of the window**: what waits, what is being made, what is made — the queue as ghost tiles above the
// render under way, then the history, newest first, each series under its tile (`StripRow`). The column widens into a grid
// over the canvas (`HistoryGrid`), and narrows back to show the image. Split from `MainView.swift`.

import AppKit
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

// MARK: - The history

/// **The history, in a column on the right of the image** (the height is what a window lacks, not the width): the newest on top,
/// and above it what is coming — the renders that wait, each a ghost in the shape of its image, then the render under way; a
/// series' cells are counted by its tile, not by ghosts. At its foot, how many images, the thumbnails' size, and the grid.
struct Banner: View {
    @Bindable var app: AppState

    var body: some View {
        // Read once for every tile: `selectedEntry` walks the history.
        let chosen = app.selectedEntry?.id
        VStack(spacing: 0) {
            ScrollViewReader { reader in
            ScrollView(.vertical) {
                LazyVStack(spacing: 8) {
                    // **The queue**: « Stop All » at its head, then the renders that wait, the next one lowest — just above
                    // the render under way, which it follows.
                    if !app.file.isEmpty {
                        Button("Stop All", role: .destructive) { app.stopAll() }
                            .buttonStyle(.chartLink).font(.caption)
                            .help("Clear the queue and stop the current render")
                        ForEach(Array(app.file.enumerated()).reversed().filter { app.gridCells[$0.element.ordinal] == nil },
                                id: \.element.id) { n, job in
                            GhostTile(app: app, job: job, place: n + 1)
                        }
                    }
                    // A series' cell shows its preview in the grid, and its tile how far the series is: no second tile for it.
                    if let job = app.currentJob, app.gridCells[job.ordinal] == nil {
                        LiveTile(app: app, job: job)
                    }
                    ForEach(app.strip) { row in
                        switch row {
                        case .series(let run):
                            SeriesTile(app: app, run: run).id(row.id)
                        case .image(let entry, let inSeries):
                            Thumbnail(app: app, entry: entry, side: inSeries ? app.thumbnailSide * 0.8 : nil,
                                      marked: app.isMarked(entry.id, chosen: chosen) && !(app.inProgress && app.followsRender),
                                      shown: entry.id == chosen && app.marked.count > 1)
                                .frame(maxWidth: .infinity)
                                .overlay(alignment: .leading) { if inSeries { SeriesRule() } }
                                .id(entry.id)
                        }
                    }
                }
                .padding(.vertical, 12)
            }
            .scrollIndicators(.never)
            .onChange(of: app.selection) { _, id in
                if let id { withAnimation(Motion.pick(.default)) { reader.scrollTo(id, anchor: .center) } }
            }
            }
            Divider()
            sizeControl
        }
        .frame(width: Self.width(app))
        // Under the band, not through it: the band's right end — the edition, the inspector — stays on the bare canvas.
        // Its top corner on the image's side is bevelled, facing the window's own rounded foot corner across the column.
        .background {
            if app.glass { Color.clear.glassEffect(.regular, in: Self.shape) } else { Palette.column.clipShape(Self.shape) }
        }
        .clipShape(Self.shape)
    }

    private static let shape = UnevenRoundedRectangle(topLeadingRadius: 14, style: .continuous)

    /// The column is as wide as its thumbnails, never more: the canvas keeps the rest.
    static func width(_ app: AppState) -> CGFloat { app.thumbnailSide + 28 }

    /// How many images the history holds — the column's foot and the grid's head.
    static func count(_ n: Int) -> String { n == 1 ? String(localized: "1 image") : String(localized: "\(n) images") }

    /// At the column's foot: how many images, the grid, and the thumbnails' size.
    private var sizeControl: some View {
        VStack(spacing: 6) {
            HStack(spacing: 6) {
                Text(Banner.count(app.history.count))
                    .font(.caption).monospacedDigit().foregroundStyle(Palette.secondary)
                Button { withAnimation(HistoryGrid.motion) { app.historyGrid = true } } label: {
                    Image(systemName: "square.grid.2x2")
                }
                .buttonStyle(.borderless)
                .disabled(app.history.isEmpty)
                .help("The history as a grid, over the image (⌥⌘G)")
                .accessibilityLabel(Text("History Grid"))
            }
            Slider(value: $app.thumbnailSide, in: 48...160).controlSize(.mini).tint(Palette.control)
                .foregroundStyle(Palette.secondary)
                .help("Thumbnail size")
                .accessibilityLabel(Text("Thumbnail size"))
        }
        .padding(.horizontal, 8).padding(.vertical, 8)
    }
}

/// **The history as a grid, over the canvas**: the column widened, the same tiles at a larger side, the same order, the same
/// selection and keys (`MainWindow.key`). A click chooses as in the column; a drag from between the tiles draws a lasso that
/// marks what it covers (⌘ or ⇧ adds to what was marked); a double click, Esc, the button or ⌥⌘G narrow it back to the
/// image. Its head says what is marked and does with it what the bar under the image does — export, copy, remove. Two
/// images marked still open the curtain, once the grid is closed.
struct HistoryGrid: View {
    @Bindable var app: AppState

    static let motion = Motion.pick(.spring(duration: 0.3))
    nonisolated private static let space = "historyGrid"

    /// Each image tile's frame in the grid, for the lasso: the tiles laid out, the lazy grid's visible ones.
    @State private var frames: [Entry.ID: CGRect] = [:]
    @State private var lasso: CGRect?
    /// What was marked as the lasso began, kept when ⌘ or ⇧ was held.
    @State private var kept: Set<Entry.ID>?
    @State private var viewport: CGFloat = 0

    var body: some View {
        let side = max(app.thumbnailSide * 1.6, 140)
        // Read once for every tile: `selectedEntry` walks the history.
        let chosen = app.selectedEntry?.id
        VStack(spacing: 0) {
            head
            ScrollViewReader { reader in
                ScrollView(.vertical) {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: side, maximum: side * 1.4), spacing: 12)], spacing: 12) {
                        ForEach(app.strip) { row in
                            switch row {
                            case .series(let run):
                                SeriesTile(app: app, run: run, side: side, opened: close).id(row.id)
                            case .image(let entry, _):
                                Thumbnail(app: app, entry: entry, side: side,
                                          marked: app.isMarked(entry.id, chosen: chosen), shown: entry.id == chosen && app.marked.count > 1,
                                          opened: { app.show(entry.id); close() })
                                    .id(entry.id)
                                    .onGeometryChange(for: CGRect.self) { $0.frame(in: .named(Self.space)) } action: { frames[entry.id] = $0 }
                                    .onDisappear { frames[entry.id] = nil }
                            }
                        }
                    }
                    // Room at the foot for the way back, which floats there.
                    .padding(.horizontal, 20).padding(.bottom, 84)
                    // The empty room under the last row takes a lasso too.
                    .frame(minHeight: viewport, alignment: .top)
                    .background { Color.clear.contentShape(Rectangle()).gesture(lassoGesture) }
                    .overlay(alignment: .topLeading) {
                        if let r = lasso {
                            Rectangle().fill(Palette.live.opacity(0.12))
                                .overlay(Rectangle().strokeBorder(Palette.live, lineWidth: 1))
                                .frame(width: r.width, height: r.height)
                                .offset(x: r.minX, y: r.minY)
                                .allowsHitTesting(false)
                        }
                    }
                    .coordinateSpace(.named(Self.space))
                }
                .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { viewport = $0 }
                .onAppear { if let id = app.selection { reader.scrollTo(id, anchor: .center) } }
                .onChange(of: app.selection) { _, id in
                    if let id, lasso == nil { withAnimation(Motion.pick(.default)) { reader.scrollTo(id, anchor: .center) } }
                }
            }
        }
        .background(Palette.canvas)
        .overlay(alignment: .bottom) { back }
    }

    /// **The way back to the image**, large and alone at the foot, where the image's bar stands when the image is shown: a
    /// small button at the end of the head was missed.
    private var back: some View {
        Button { close() } label: {
            Label("Back to the Image", systemImage: "arrow.uturn.backward")
                .font(.body.weight(.semibold))
                .padding(.horizontal, 20).padding(.vertical, 11)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .glass(Capsule())
        .padding(.bottom, 20)
        .help("Back to the image (Esc, ⌥⌘G)")
    }

    /// How many images, what is marked and what is done with it.
    private var head: some View {
        let group = app.targets
        return HStack(spacing: 12) {
            Text(Banner.count(app.history.count))
                .font(.callout).monospacedDigit().foregroundStyle(Palette.secondary)
            if app.marked.count > 1 {
                Divider().frame(height: 18)
                Text("\(group.count) images selected").font(.callout).monospacedDigit()
                Button("Export \(group.count) Images…") { Task { await app.export(group) } }
                    .help("Each image as its PNG, with its settings, into a folder (⇧⌘E)")
                Button { app.copyImages(group) } label: { Image(systemName: "doc.on.doc") }
                    .help("Copy the selected images (⇧⌘C)")
                    .accessibilityLabel(Text("Copy \(group.count) Images"))
                Button { app.delete(group) } label: { Image(systemName: "trash") }
                    .help("Remove the selected images from the history (⌫, undo with ⌘Z)")
                    .accessibilityLabel(Text("Remove from History"))
                Button("Deselect") { app.unmark() }
                    .help("Keep only the image shown (Esc)")
            } else {
                Text("Drag across the images to select several, or ⌘-click and ⇧-click; ⌘A selects them all.")
                    .font(.caption).foregroundStyle(Palette.secondary).lineLimit(1)
            }
            Spacer()
        }
        .controlSize(.small)
        .padding(.horizontal, 20).padding(.vertical, 12)
    }

    private var lassoGesture: some Gesture {
        DragGesture(minimumDistance: 4, coordinateSpace: .named(Self.space))
            .onChanged { v in
                if kept == nil {
                    MainWindow.shared.releaseTextFocus()
                    let mods = NSEvent.modifierFlags
                    kept = mods.contains(.command) || mods.contains(.shift)
                        ? (app.marked.count > 1 ? app.marked : Set([app.selection].compactMap { $0 })) : []
                }
                let r = CGRect(x: min(v.startLocation.x, v.location.x), y: min(v.startLocation.y, v.location.y),
                               width: abs(v.location.x - v.startLocation.x), height: abs(v.location.y - v.startLocation.y))
                lasso = r
                let covered = Set(frames.filter { $0.value.intersects(r) }.map(\.key))
                if !covered.isEmpty || !(kept ?? []).isEmpty { app.lasso(covered, kept: kept ?? []) }
            }
            .onEnded { _ in lasso = nil; kept = nil }
    }

    private func close() { withAnimation(Self.motion) { app.historyGrid = false } }
}

/// A double click, attached only with an action: with one, every single click waits to know it is not the first of two.
private struct DoubleClick: ViewModifier {
    let action: (() -> Void)?

    func body(content: Content) -> some View {
        if let action { content.onTapGesture(count: 2, perform: action) } else { content }
    }
}

/// **A history image in the strip**: a click shows it, ⌘-click and ⇧-click mark several (`AppState.click`),
/// a drag carries its PNG out — to the Finder, a message, another app. Marked, it wears the accent edge;
/// among several marked, the one shown wears it thicker.
private struct Thumbnail: View {
    let app: AppState
    let entry: Entry
    /// The column's side, or the grid's.
    var side: Double? = nil
    let marked: Bool
    let shown: Bool
    /// The grid's double click: the image shown, the grid closed.
    var opened: (() -> Void)? = nil
    @State private var hover = false

    var body: some View {
        let side = self.side ?? app.thumbnailSide
        let l = Double(entry.width), h = Double(entry.height)
        Image(decorative: entry.vignette, scale: 1)
            .resizable()
            .interpolation(.medium)
            .scaledToFill()
            .frame(width: side * min(1, l / h), height: side * min(1, h / l))
            .clipShape(RoundedRectangle(cornerRadius: 5))
            .overlay(RoundedRectangle(cornerRadius: 5)
                .strokeBorder(marked ? Palette.live : hover ? Color.secondary : Palette.hairline,
                              lineWidth: marked ? (shown ? 3.5 : 2.5) : 1))
            .opacity(marked || hover ? 1 : 0.88)
            // **The ✕ of the hovered tile** (issue 16): the lab's own, on this image alone even among several marked —
            // the menu acts on the group, the ✕ on what it sits on. No question asked: ⌘Z brings it back.
            .overlay(alignment: .topTrailing) {
                if hover {
                    Button { app.delete(entry) } label: {
                        Image(systemName: "xmark.circle.fill").symbolRenderingMode(.palette)
                            .foregroundStyle(.white, .black.opacity(0.55))
                    }
                    .buttonStyle(.plain).padding(3)
                    .help("Remove from History (⌘Z brings it back)")
                    .accessibilityLabel(Text("Remove from History"))
                }
            }
            .frame(height: side + 4)
            .contentShape(Rectangle())
            .onHover { hover = $0 }
            // Only where a double click means something: a tap that may be the first of two waits for the second.
            .modifier(DoubleClick(action: opened))
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
            .animation(Motion.pick(.easeOut(duration: 0.12)), value: hover)
    }
}

/// **A render that waits**, above the one under way: a ghost in the shape of its image, its place in the queue, what it will
/// take — learned on this Mac —, and its ✕. Its prompt on hover.
private struct GhostTile: View {
    let app: AppState
    let job: Job
    let place: Int

    var body: some View {
        let side = app.thumbnailSide
        let l = Double(job.settings.format.width), h = Double(job.settings.format.height)
        let shape = RoundedRectangle(cornerRadius: 5)
        ZStack {
            shape.fill(Color.primary.opacity(0.04))
            shape.strokeBorder(Palette.secondary.opacity(0.6), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
            VStack(spacing: 2) {
                Text(verbatim: "\(place)").font(.caption.weight(.semibold)).monospacedDigit()
                if let e = app.estimation(job) {
                    Text(verbatim: "≈ " + Figures.duration(e)).font(.caption2).monospacedDigit()
                }
            }
            .foregroundStyle(Palette.secondary)
        }
        .frame(width: side * min(1, l / h), height: side * min(1, h / l))
        .overlay(alignment: .topTrailing) {
            Button { app.remove(job) } label: {
                Image(systemName: "xmark.circle.fill").foregroundStyle(Palette.secondary)
            }
            .buttonStyle(.plain).padding(3)
            .help("Remove from the queue")
            .accessibilityLabel(Text("Remove from the queue"))
        }
        .frame(height: side + 4)
        .help(Text(verbatim: "\(job.settings.prompt)\n\(job.legend)"))
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text(verbatim: job.settings.prompt))
    }
}

/// **The running render, under the queue**: its preview in the shape of the coming image, its progress along the bottom, and
/// on hover its stop. Chosen, the canvas follows the render again.
private struct LiveTile: View {
    let app: AppState
    let job: Job
    @State private var hover = false

    var body: some View {
        let side = app.thumbnailSide
        let l = Double(job.settings.format.width), h = Double(job.settings.format.height)
        let following = app.followsRender
        ZStack(alignment: .bottom) {
            Color.primary.opacity(0.08)
            if let p = app.preview {
                Image(decorative: p, scale: 1).resizable().interpolation(.medium).scaledToFill()
            }
            GeometryReader { g in
                Capsule().fill(Palette.live)
                    .frame(width: max(4, g.size.width * app.fraction), height: 4)
                    .frame(maxHeight: .infinity, alignment: .bottom)
            }
            .padding(3)
        }
        .frame(width: side * min(1, l / h), height: side * min(1, h / l))
        // A batch is one job and its bar counts the whole batch: the image it is at, said on the tile.
        .overlay(alignment: .topTrailing) {
            if app.batchSize > 1 {
                Text(verbatim: "\(app.batchImage + 1)/\(app.batchSize)")
                    .font(.caption2.weight(.semibold)).monospacedDigit()
                    .foregroundStyle(.white)
                    .padding(.horizontal, 5).padding(.vertical, 1)
                    .background(Capsule().fill(Color.black.opacity(0.6)))
                    .padding(4)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 5))
        .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(following ? Palette.live : Palette.hairline, lineWidth: following ? 2.5 : 1))
        .overlay(alignment: .topLeading) {
            if hover {
                Button { app.cancel() } label: {
                    Image(systemName: "stop.circle.fill").symbolRenderingMode(.palette).foregroundStyle(.white, .black.opacity(0.55))
                }
                .buttonStyle(.plain).padding(3)
                .help("Stop (⌘.)")
                .accessibilityLabel(Text("Stop"))
            }
        }
        .frame(height: side + 4)
        .contentShape(Rectangle())
        .onHover { hover = $0 }
        .onTapGesture { app.followsRender = true; app.shownSeries = nil }
        .help("The render in progress")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("The render in progress"))
        .accessibilityValue(Text(verbatim: Figures.percent(Int(app.fraction * 100))))
        .accessibilityAddTraits(.isButton)
    }
}

/// **A series in the history**: its grid in small — each cell's thumbnail at its place —, how many cells came, and while
/// it runs, how far it is. A click opens its grid on the canvas; its chevron, always there, folds its cells, which the
/// arrows then skip; its ✕ on hover takes the series out (⌘Z brings its cells back).
private struct SeriesTile: View {
    let app: AppState
    let run: GridRun
    /// The column's side, or the grid's.
    var side: Double? = nil
    /// The history grid's click: the series' grid shown, the history grid closed.
    var opened: (() -> Void)? = nil
    @State private var hover = false

    var body: some View {
        let side = self.side ?? app.thumbnailSide
        let images = app.cells(of: run)
        let g = run.grid, f = run.settings.format
        // The mini grid in the tile's square, at the cells' proportions.
        let wide = Double(g.columns * f.width), tall = Double(g.rows * f.height)
        let (w, h) = wide >= tall ? (side, side * tall / wide) : (side * wide / tall, side)
        let shown = app.shownSeries == run.ordinal
        VStack(spacing: 1) {
            ForEach(0..<g.rows, id: \.self) { r in
                HStack(spacing: 1) {
                    ForEach(0..<g.columns, id: \.self) { c in
                        if let e = images[r * g.columns + c] {
                            Image(decorative: e.vignette, scale: 1).resizable().interpolation(.medium).scaledToFill()
                                .frame(maxWidth: .infinity, maxHeight: .infinity).clipped()
                        } else {
                            Palette.well
                        }
                    }
                }
            }
        }
        .frame(width: w, height: h)
        .overlay(alignment: .bottom) {
            if app.isRunning(run) {
                GeometryReader { geo in
                    Capsule().fill(Palette.live)
                        .frame(width: max(4, geo.size.width * progress), height: 4)
                        .frame(maxHeight: .infinity, alignment: .bottom)
                }
                .padding(3)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 5))
        .overlay(RoundedRectangle(cornerRadius: 5)
            .strokeBorder(shown ? Palette.live : Palette.bracket, lineWidth: shown ? 2.5 : 1.5))
        .overlay(alignment: .bottomTrailing) {
            Text(verbatim: "\(images.count)/\(run.jobs.count)")
                .font(.caption2.weight(.semibold)).monospacedDigit().foregroundStyle(.white)
                .padding(.horizontal, 5).padding(.vertical, 1)
                .background(Capsule().fill(Color.black.opacity(0.6)))
                .padding(4)
        }
        // Always drawn: what the tile heads is told by it, folded or not.
        .overlay(alignment: .topLeading) {
            Button { withAnimation(Motion.pick(.easeOut(duration: 0.15))) { app.toggleFold(run) } } label: {
                Image(systemName: app.foldedSeries.contains(run.ordinal) ? "chevron.right.circle.fill" : "chevron.down.circle.fill")
                    .symbolRenderingMode(.palette).foregroundStyle(.white, .black.opacity(0.55))
            }
            .buttonStyle(.plain).padding(3)
            .help(app.foldedSeries.contains(run.ordinal) ? "Unfold the cells" : "Fold the cells — the arrows skip them")
            .accessibilityLabel(Text(app.foldedSeries.contains(run.ordinal) ? "Unfold the Cells" : "Fold the Cells"))
        }
        .overlay(alignment: .topTrailing) {
            if hover {
                Button { app.removeSeries(run) } label: {
                    Image(systemName: "xmark.circle.fill").symbolRenderingMode(.palette)
                        .foregroundStyle(.white, .black.opacity(0.55))
                }
                .buttonStyle(.plain).padding(3)
                .help("Remove the series from history (⌘Z brings its images back)")
                .accessibilityLabel(Text("Remove Series from History"))
            }
        }
        .frame(height: side + 4)
        .contentShape(Rectangle())
        .onHover { hover = $0 }
        .onTapGesture {
            MainWindow.shared.releaseTextFocus()
            app.shownSeries = run.ordinal
            opened?()
        }
        .help(Text(verbatim: run.grid.title))
        .contextMenu {
            Button("Show the Grid") { app.shownSeries = run.ordinal; opened?() }
            Button("Export Sheet…") { app.exportSheet(run) }.disabled(images.isEmpty)
            Button("Copy Sheet") { app.copySheet(run) }.disabled(images.isEmpty)
            Button("Reuse This Grid") { app.restoreSettings(run) }
            Button(app.foldedSeries.contains(run.ordinal) ? "Unfold the Cells" : "Fold the Cells") { app.toggleFold(run) }
            Divider()
            if app.isRunning(run) { Button("Stop") { app.stop(run) } }
            Button("Remove Series from History", role: .destructive) { app.removeSeries(run) }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text(verbatim: run.grid.title))
        .accessibilityValue(Text(verbatim: "\(images.count)/\(run.jobs.count)"))
        .accessibilityAddTraits(shown ? [.isButton, .isSelected] : .isButton)
        .animation(Motion.pick(.easeOut(duration: 0.12)), value: hover)
    }

    /// The cells ended, and the share of the one rendering.
    private var progress: Double {
        let running = app.currentJob.map { run.jobs.contains($0.ordinal) } == true ? app.fraction : 0
        return min(1, (Double(run.issues.count) + running) / Double(max(1, run.jobs.count)))
    }
}

/// The rule along a series' cells, in the series' yellow: what the tile above heads.
private struct SeriesRule: View {
    var body: some View {
        Capsule().fill(Palette.bracket.opacity(0.55)).frame(width: 2).padding(.leading, 7).padding(.vertical, 2)
            .accessibilityHidden(true)
    }
}

/// **« Variations »'s two choices** — the bar under the image and the context menu.
struct VariationButtons: View {
    let app: AppState
    let entry: Entry

    var body: some View {
        Button("Subtle Variations") { app.vary(entry, .subtle) }
        Button("Strong Variations") { app.vary(entry, .strong) }
    }
}

/// **« Enlarge ×4 »** in the bar under the image: greyed with its reason when it cannot; while it
/// works, a spinner — or the model's download, in percent.
struct EnlargeButton: View {
    let app: AppState
    let entry: Entry

    var body: some View {
        if app.enlarging == entry.id {
            HStack(spacing: 5) {
                ProgressView().controlSize(.small)
                if let p = app.enlargeDownload { Text(Figures.percent(Int(p * 100))).font(.caption).monospacedDigit() }
            }
            .help(app.enlargeDownload == nil ? "Enlarging…" : "macOS is downloading the enlargement model")
        } else {
            Button { app.enlarge(entry) } label: { Image(systemName: "arrow.up.left.and.down.right.magnifyingglass") }
                .help(app.enlargeHelp(entry))
                .accessibilityLabel(Text("Enlarge ×4"))
                .unavailable(app.enlargeRefusal(entry))
        }
    }
}

/// **Sending an image to be edited**: by the rack's model when it edits; otherwise by an installed model
/// that does — chosen in the rack on the way.
private struct EditButtons: View {
    let app: AppState
    let entry: Entry

    var body: some View {
        if app.editingPossible {
            // The shortcut of « Send to the Prompt Lab » under a model that edits: the image goes straight to image 1.
            Button("Edit with \(app.card?.name ?? "")") { app.placeReference(entry) }
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
struct MenuEntry: View {
    let app: AppState
    let entry: Entry

    var body: some View {
        let group = app.marked.count > 1 && app.marked.contains(entry.id) ? app.targets : [entry]
        if group.count > 1 {
            Button("Export \(group.count) Images…") { Task { await app.export(group) } }
            Button("Copy \(group.count) Images") { app.copyImages(group) }
            ShareLink("Share…", items: group.map(app.transferable), preview: { SharePreview($0.name) })
            Divider()
            Button("Remove \(group.count) Images from History", role: .destructive) { app.delete(group) }
        } else {
            Button("Save…") { Task { await app.save(entry) } }
            Button("Copy Image") { app.copyImage(entry) }
            ShareLink("Share…", item: app.transferable(entry), preview: SharePreview(app.exportName(entry)))
            Divider()
            Button(app.busy ? "Render Again (Same Seed), Queued" : "Render Again (Same Seed)") { app.redo(entry) }
            Menu("Variations") { VariationButtons(app: app, entry: entry) }
                .disabled(!app.canVary(entry))
            Button("Enlarge ×4") { app.enlarge(entry) }
                .disabled(app.enlargeRefusal(entry) != nil)
            Button("Crop…") { app.beginCrop(entry) }
                .disabled(app.cropRefusal(entry) != nil)
            Button("Reuse These Settings") { app.restoreSettings(entry) }
            EditButtons(app: app, entry: entry)
            Button("Send to the Prompt Lab") { LabModel.shared.send(entry) }
            Divider()
            Button("Remove from History", role: .destructive) { app.delete(entry) }
        }
    }
}
