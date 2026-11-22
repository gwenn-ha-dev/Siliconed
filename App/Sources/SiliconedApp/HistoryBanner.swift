// **Time, on the right of the window**: what waits, what is being made, what is made — the queue as ghost tiles above the
// render under way, then the history, newest first, each series under its tile (`StripRow`). The column widens into a grid
// over the canvas (`HistoryGrid`), and narrows back to show the image.

import AppKit
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

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
                .help("The history as a grid, over the image (⌥⌘G)")
                .accessibilityLabel(Text("History Grid"))
                .unavailable(app.history.isEmpty ? String(localized: "No image in the history yet.") : nil)
            }
            Slider(value: $app.thumbnailSide, in: 48...160).controlSize(.mini).tint(Palette.control)
                .foregroundStyle(Palette.secondary)
                .help("Thumbnail size")
                .accessibilityLabel(Text("Thumbnail size"))
        }
        .padding(.horizontal, 8).padding(.vertical, 8)
    }
}
