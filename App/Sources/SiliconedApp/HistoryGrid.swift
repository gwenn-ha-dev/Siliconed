// The history as a grid over the canvas: its lasso, its head.

import AppKit
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

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
