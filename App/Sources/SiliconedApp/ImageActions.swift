// An image's actions, shared by the bar under it and its context menu: Variations, Enlarge, Edit, the menu.

import AppKit
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

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
