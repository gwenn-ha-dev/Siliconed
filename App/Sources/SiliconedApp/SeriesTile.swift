// A series in the history: its tile, and the rule along its cells.

import AppKit
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

/// **A series in the history**: its grid in small — each cell's thumbnail at its place —, how many cells came, and while
/// it runs, how far it is. A click opens its grid on the canvas; its chevron, always there, folds its cells, which the
/// arrows then skip; its ✕ on hover takes the series out (⌘Z brings its cells back).
struct SeriesTile: View {
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
        .removalOnHover(hover, padding: 3, help: "Remove the series from history (⌘Z brings its images back)",
                        label: "Remove Series from History") { app.removeSeries(run) }
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
struct SeriesRule: View {
    var body: some View {
        Capsule().fill(Palette.bracket.opacity(0.55)).frame(width: 2).padding(.leading, 7).padding(.vertical, 2)
            .accessibilityHidden(true)
    }
}
