// The history strip: banner, thumbnails, live and exploration tiles and their buttons.
// Split from `MainView.swift`.

import AppKit
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

// MARK: - The history

struct Banner: View {
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
            Divider()
            sizeControl
        }
        // The strip is as tall as its thumbnails, never more: the canvas keeps the rest.
        .frame(height: app.thumbnailSide + 20)
        .background(.black.opacity(0.12))
        .overlay(alignment: .top) { Divider() }
    }

    /// At the strip's end: how many images, and the thumbnails' size.
    private var sizeControl: some View {
        VStack(alignment: .trailing, spacing: 6) {
            Text(app.history.count == 1 ? String(localized: "1 image") : String(localized: "\(app.history.count) images"))
                .font(.caption).monospacedDigit().foregroundStyle(.secondary)
            HStack(spacing: 4) {
                Image(systemName: "photo").imageScale(.small)
                Slider(value: $app.thumbnailSide, in: 48...140).frame(width: 80).controlSize(.mini)
                Image(systemName: "photo").imageScale(.large)
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
                .strokeBorder(marked ? Color.accentColor : hover ? Color.secondary : Palette.hairline,
                              lineWidth: marked ? (shown ? 3.5 : 2.5) : 1))
            .opacity(marked || hover ? 1 : 0.88)
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
            .strokeBorder(app.showsExploration ? Palette.live : Palette.hairline, lineWidth: app.showsExploration ? 2.5 : 1))
        .frame(height: side + 4)
        .contentShape(Rectangle())
        .onTapGesture { app.showsExploration = true }
        .help("The exploration's grid")
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
                .disabled(app.enlargeRefusal(entry) != nil)
                .help(app.enlargeHelp(entry))
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
struct MenuEntry: View {
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
            if entry.grid == nil, entry.sketch == nil {
                Button("Enlarge ×4") { app.enlarge(entry) }
                    .disabled(app.enlargeRefusal(entry) != nil)
            }
            if entry.grid == nil, entry.sketch == nil {
                Button("Crop…") { app.beginCrop(entry) }
                    .disabled(app.cropRefusal(entry) != nil)
            }
            Button(entry.grid == nil ? "Reuse These Settings" : "Reuse This Grid") { app.restoreSettings(entry) }
            if entry.grid == nil { EditButtons(app: app, entry: entry) }
            Divider()
            Button("Remove from History", role: .destructive) { app.delete(entry) }
        }
    }
}
