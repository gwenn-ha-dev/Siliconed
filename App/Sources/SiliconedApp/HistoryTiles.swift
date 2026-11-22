// The history's tiles: an image, a render waiting (a ghost), the render under way.

import AppKit
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

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
struct Thumbnail: View {
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
            .removalOnHover(hover, padding: 3, help: "Remove from History (⌘Z brings it back)", label: "Remove from History") {
                app.delete(entry)
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
struct GhostTile: View {
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
struct LiveTile: View {
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
