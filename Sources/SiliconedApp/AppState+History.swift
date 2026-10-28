// **`AppState`: the history and what leaves the app** — the chosen and marked images, their decodings,
// saving, exporting, copying, removing. The state itself is in `AppState.swift`.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

extension AppState {
    // MARK: History and outputs

    var selectedEntry: Entry? {
        history.first { $0.id == selection } ?? history.first
    }

    func loadCompared() {
        comparedLoad?.cancel()
        let entries = marked.count > 1 && marked.count <= Self.maxCompared ? history.filter { marked.contains($0.id) } : []
        guard !entries.isEmpty else { compared = []; return }
        let kept = Dictionary(compared.map { ($0.entry.id, $0.image) }, uniquingKeysWith: { a, _ in a })
        let marking = marked
        comparedLoad = Task { [weak self] in
            var loaded: [(entry: Entry, image: CGImage)] = []
            for e in entries {
                let image: CGImage? = if let k = kept[e.id] { k } else { await Entry.decoded(e.png).image }
                if let image { loaded.append((e, image)) }
            }
            guard let self, !Task.isCancelled, self.marked == marking else { return }
            self.compared = loaded
        }
    }

    /// The full-size image of the chosen entry, decoded once per choice.
    func loadSelectedImage() {
        defer { loadOriginal() }
        selectedLoad?.cancel()
        guard let e = selectedEntry else { selectedImage = nil; return }
        guard selectedImage?.id != e.id else { return }
        let id = e.id, png = e.png
        selectedLoad = Task { [weak self] in
            let image = await Entry.decoded(png).image
            guard let self, !Task.isCancelled, self.selectedEntry?.id == id else { return }
            self.selectedImage = image.map { (id, $0) }
        }
    }

    /// The chosen edit's image 1, decoded only while its curtain is asked for — up to 3072 px read.
    func loadOriginal() {
        originalLoad?.cancel()
        guard comparesOriginal, let e = selectedEntry, canCompareOriginal(e) else { original = nil; return }
        guard original?.id != e.id else { return }
        let id = e.id, reference = e.references.first, source = e.enlargedFrom, width = e.width, height = e.height
        originalLoad = Task { [weak self] in
            let image = await Task.detached(priority: .userInitiated) {
                Decoded(source.flatMap { SuperResolution.stretched($0, width: width, height: height) }
                        ?? reference?.displayed(width: width, height: height))
            }.value.image
            guard let self, !Task.isCancelled, self.selectedEntry?.id == id, self.comparesOriginal else { return }
            self.original = image.map { (id, $0) }
        }
    }

    /// A history image chosen by the user: during a render, the canvas leaves the render for it.
    func show(_ id: Entry.ID) {
        selection = id
        marked = []
        anchor = id
        if inProgress { followsRender = false }
    }

    /// **A click on a thumbnail**, read with its modifiers as the Finder reads them: alone, it shows
    /// the image; with ⌘, it adds it to the marked ones or takes it out; with ⇧, it marks the run
    /// from the last plain click.
    func click(_ id: Entry.ID, modifiers: NSEvent.ModifierFlags) {
        if modifiers.contains(.shift), let a = anchor ?? selection {
            markRun(from: a, to: id)
        } else if modifiers.contains(.command) {
            var m = marked.count > 1 ? marked : Set([selection].compactMap { $0 })
            if m.contains(id), m.count > 1 {
                m.remove(id)
                if selection == id, let other = history.first(where: { m.contains($0.id) }) { reveal(other.id) }
            } else {
                m.insert(id)
                reveal(id)
            }
            marked = m.count > 1 ? m : []
            anchor = id
        } else {
            show(id)
        }
    }

    /// Shows an image without touching what is marked.
    private func reveal(_ id: Entry.ID) {
        selection = id
        if inProgress { followsRender = false }
    }

    private func markRun(from a: Entry.ID, to b: Entry.ID) {
        guard let i = history.firstIndex(where: { $0.id == a }), let j = history.firstIndex(where: { $0.id == b }) else { return }
        let run = Set(history[min(i, j)...max(i, j)].map(\.id))
        marked = run.count > 1 ? run : []
        reveal(b)
    }

    /// ⌘A in the strip: every image.
    func markAll() {
        guard history.count > 1 else { return }
        marked = Set(history.map(\.id))
        if selection == nil { selection = history.first?.id }
    }

    /// Esc: back to the one image shown.
    func unmark() { marked = [] }

    /// Whether a thumbnail wears the selection's edge: the image shown, or one of those marked.
    func isMarked(_ id: Entry.ID) -> Bool {
        marked.count > 1 ? marked.contains(id) : id == selectedEntry?.id
    }

    /// **What the image commands act on**: the marked images, in the strip's order — or the one shown.
    var targets: [Entry] {
        marked.count > 1 ? history.filter { marked.contains($0.id) } : selectedEntry.map { [$0] } ?? []
    }

    /// The neighbouring image in the strip: −1 to the left (more recent), +1 to the right (older), as
    /// the arrow keys move along it. Watching a render, it starts from the most recent image. With
    /// `extending` (⇧), the run from the anchor is marked.
    func chooseAdjacent(_ offset: Int, extending: Bool = false) {
        guard let e = selectedEntry, let i = history.firstIndex(where: { $0.id == e.id }) else { return }
        if inProgress && followsRender { show(history[offset > 0 ? i : 0].id); return }
        let j = min(max(i + offset, 0), history.count - 1)
        if extending, let a = anchor { markRun(from: a, to: history[j].id) } else { show(history[j].id) }
    }

    /// Home, End: the most recent image, the oldest.
    func chooseEnd(oldest: Bool) {
        guard let e = oldest ? history.last : history.first else { return }
        show(e.id)
    }

    func delete(_ entry: Entry) { delete([entry]) }

    /// **Removes images from the history** — undone by ⌘Z, each back at its place. The image shown
    /// passes to the nearest one left.
    func delete(_ entries: [Entry]) {
        // In a mirror the other Mac's history is the one: it removes, and says so (`removed`). No ⌘Z yet.
        if let mirror { mirror.send(.deleteImages(entries.map(\.ordinal))); return }
        let ids = Set(entries.map(\.id))
        let removed = history.enumerated().filter { ids.contains($0.element.id) }
        guard !removed.isEmpty else { return }
        let first = removed[0].offset
        history.removeAll { ids.contains($0.id) }
        marked = []
        if let s = selection, ids.contains(s) {
            selection = history.isEmpty ? nil : history[min(first, history.count - 1)].id
        }
        anchor = selection
        loadSelectedImage()
        undoManager?.registerUndo(withTarget: self) { app in
            for (i, e) in removed { app.history.insert(e, at: min(i, app.history.count)) }
            app.show(removed[0].element.id)
            app.marked = removed.count > 1 ? ids : []
            app.undoManager?.registerUndo(withTarget: app) { $0.delete(removed.map(\.element)) }
        }
        undoManager?.setActionName(removed.count == 1 ? String(localized: "Remove Image")
                                                       : String(localized: "Remove \(removed.count) Images"))
    }

    /// Everything, after a confirmation that says how many were never saved.
    func clearHistory() {
        guard !history.isEmpty else { return }
        let alert = NSAlert()
        alert.messageText = String(localized: "Clear the history?")
        let n = unsaved.count
        alert.informativeText = n == 0
            ? String(localized: "Every image has been saved. You can undo this with ⌘Z.")
            : String(localized: "\(n) of these images were never saved. You can undo this with ⌘Z while the app is open.")
        alert.addButton(withTitle: String(localized: "Clear History"))
        alert.addButton(withTitle: String(localized: "Cancel"))
        alert.buttons[0].hasDestructiveAction = true
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        delete(history)
    }

    /// Save: in the format last used (PNG the first time — it carries the prompt, the seed, the model),
    /// JPEG or HEIC from the panel's pop-up. JPEG and HEIC carry no metadata: they are plain pictures.
    func save(_ entry: Entry) {
        let panel = NSSavePanel()
        let base = exportName(entry)
        panel.allowedContentTypes = [exportFormat.type]
        panel.nameFieldStringValue = base + "." + exportFormat.fileExtension
        panel.canCreateDirectories = true
        let (accessory, kind) = Self.formatAccessory(selected: exportFormat, note: true)
        FormatFollower.attach(panel, to: kind)
        accessory.setAccessibilityLabel(String(localized: "Format:"))
        panel.accessoryView = accessory
        guard panel.runModal() == .OK, let url = panel.url else { return }
        exportFormat = ExportFormat(extension: url.pathExtension) ?? exportFormat
        do {
            try write(entry, to: url)
            flash(String(localized: "Saved: \(url.lastPathComponent)"))
        } catch {
            problem = Problem((error as? EngineError) ?? .imageWriteFailed(file: url.path))
        }
    }

    /// The panel's « Format: » pop-up, set on `selected`; with `note`, the line that says what each keeps.
    static func formatAccessory(selected: ExportFormat, note: Bool) -> (NSView, NSPopUpButton) {
        let kind = NSPopUpButton(frame: .zero, pullsDown: false)
        kind.addItems(withTitles: ExportFormat.allCases.map(\.label))
        kind.selectItem(at: ExportFormat.allCases.firstIndex(of: selected) ?? 0)
        let label = NSTextField(labelWithString: String(localized: "Format:"))
        let row = NSStackView(views: [label, kind])
        guard note else { row.edgeInsets = NSEdgeInsets(top: 10, left: 16, bottom: 10, right: 16); return (row, kind) }
        let caption = NSTextField(wrappingLabelWithString: String(localized: "PNG keeps the prompt, seed and model. JPEG and HEIC are plain pictures, with no metadata."))
        caption.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
        caption.textColor = .secondaryLabelColor
        caption.preferredMaxLayoutWidth = 300
        let column = NSStackView(views: [row, caption])
        column.orientation = .vertical
        column.alignment = .leading
        column.spacing = 6
        column.edgeInsets = NSEdgeInsets(top: 10, left: 16, bottom: 10, right: 16)
        return (column, kind)
    }

    /// **Export**: one image is « Save As »; several go into a chosen folder, each under its own name
    /// (`exportName`), as PNG, JPEG or HEIC — `true` once written.
    @discardableResult
    func export(_ entries: [Entry]) -> Bool {
        guard !entries.isEmpty else { return false }
        if entries.count == 1 {
            save(entries[0])
            return saved.contains(entries[0].id)
        }
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.prompt = String(localized: "Export")
        panel.message = String(localized: "Export \(entries.count) images to a folder. Each keeps its prompt, seed and model in its metadata.")
        let (accessory, kind) = Self.formatAccessory(selected: exportFormat, note: true)
        panel.accessoryView = accessory
        panel.isAccessoryViewDisclosed = true
        guard panel.runModal() == .OK, let folder = panel.url else { return false }
        exportFormat = ExportFormat.allCases[max(kind.indexOfSelectedItem, 0)]
        let ext = exportFormat.fileExtension
        do {
            for e in entries { try write(e, to: Self.freeURL(in: folder, base: exportName(e), ext: ext)) }
        } catch {
            problem = Problem((error as? EngineError) ?? .imageWriteFailed(file: folder.path))
            return false
        }
        flash(String(localized: "\(entries.count) images exported to \(folder.lastPathComponent)"))
        return true
    }

    /// **An image's file name**: the model, its LoRAs, the steps if they are not the model's, the
    /// format, the seed — two different settings never share a name (`silicontrol save` writes the same).
    func exportName(_ e: Entry) -> String {
        let r = e.settings
        if let g = e.grid?.grid {
            // `grid-z-image-2x2-512x512-42`: the grid, the cells' format, the base seed (the first of
            // `seeds=N`) — none when the seeds are listed: no cell need have it.
            let listed = [g.x, g.y].contains { if case .seeds? = $0 { true } else { false } }
            return (["grid", r.identifier.replacingOccurrences(of: "/", with: "-"), "\(g.columns)x\(g.rows)",
                     "\(r.format.width)x\(r.format.height)"] + (listed ? [] : ["\(e.seed)"])).joined(separator: "-")
        }
        let modelSteps = cards.first { $0.id == r.identifier }?.defaultSteps
        return ([r.identifier.replacingOccurrences(of: "/", with: "-")]
                + r.loras.map { "lora-" + RemoteControl.shortName($0.path) + ($0.strength == 1 ? "" : String(format: "@%.2f", $0.strength)) }
                + (r.steps != modelSteps ? ["p\(r.steps)"] : [])
                + (r.editing ? ["edit"] : [])
                // A sketch never takes a finished image's name.
                + (e.sketch.map { ["sketch\($0)"] } ?? [])
                + ["\(e.width)x\(e.height)", "\(e.seed)"]
                // A variation is another image of the same seed: `v<seed>x<strength in hundredths>`, in
                // order — the CLI's name for it.
                + r.variations.map { "v\($0.seed)x\(Int(($0.strength * 100).rounded()))" }).joined(separator: "-")
    }

    /// `base.png` in the folder, or `base-2.png`, `base-3.png`… never over an existing file.
    static func freeURL(in folder: URL, base: String, ext: String) -> URL {
        var url = folder.appendingPathComponent("\(base).\(ext)")
        var k = 2
        while FileManager.default.fileExists(atPath: url.path) {
            url = folder.appendingPathComponent("\(base)-\(k).\(ext)")
            k += 1
        }
        return url
    }

    /// Writes an image as its extension says (JPEG at 0.9 and HEIC at 0.85, both plain pictures; or the
    /// PNG with its metadata), and counts it as saved.
    func write(_ entry: Entry, to url: URL) throws {
        switch ExportFormat(extension: url.pathExtension) ?? .png {
        case .png:
            try PNG.write(entry.png, to: url.path)
        case .jpeg:
            guard let cg = entry.image(),
                  let jpeg = NSBitmapImageRep(cgImage: cg).representation(using: .jpeg, properties: [.compressionFactor: 0.9]) else {
                throw EngineError.imageWriteFailed(file: url.path)
            }
            try jpeg.write(to: url, options: .atomic)
        case .heic:
            guard let cg = entry.image(), let heic = ExportFormat.heic(cg, quality: 0.85) else {
                throw EngineError.imageWriteFailed(file: url.path)
            }
            try heic.write(to: url, options: .atomic)
        }
        saved.insert(entry.id)
    }

    func copyImage(_ entry: Entry) { copyImages([entry]) }

    /// One pasteboard item per image (PNG): pasted into Finder, Photos or a message, each arrives.
    func copyImages(_ entries: [Entry]) {
        guard !entries.isEmpty else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.writeObjects(entries.map { e in
            let item = NSPasteboardItem()
            item.setData(e.png, forType: .png)
            return item
        })
        flash(entries.count == 1 ? String(localized: "Image copied") : String(localized: "\(entries.count) images copied"))
    }

    /// What a drag out of the strip or the canvas carries, and what Share sends: the PNG, under its
    /// export name. Nothing is written before the drop: the receiver writes it.
    func transferable(_ e: Entry) -> PNGImage { PNGImage(data: e.png, name: exportName(e) + ".png") }

    /// The `silicontrol add` command that redoes what the form describes — the equivalent of Draw
    /// Things Headless's "Copy as curl", for the `silicontrol` command shipped in the bundle. The
    /// references go as `--ref`, in order; one that is not a file has no command form: the command
    /// says so in a comment.
    func command() -> String {
        func quotes(_ s: String) -> String { "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'" }
        var pieces = ["silicontrol", "add", "--model", identifier]
        for l in loras where !l.path.isEmpty {
            let file = ((l.path as NSString).lastPathComponent as NSString).deletingPathExtension
            pieces += ["--lora", quotes("\((file as NSString).deletingPathExtension):\(String(format: "%.2f", l.strength))")]
        }
        if steps != card?.defaultSteps { pieces += ["--steps", "\(steps)"] }
        if effectiveDetail != .normal { pieces += ["--detail", effectiveDetail.rawValue] }
        if effectiveSolver != .euler { pieces += ["--solver", effectiveSolver.rawValue] }
        if effectiveSchedule != .official { pieces += ["--schedule", effectiveSchedule.rawValue] }
        var notes: [String] = []
        for (n, reference) in references.enumerated() {
            if case .file(let url) = reference.source {
                pieces += ["--ref", quotes(url.path)]
            } else {
                notes.append(String(localized: "image \(n + 1) comes from the history: save it, then --ref"))
            }
        }
        if batch > 1 { pieces += ["--batch", "\(batch)"] }
        pieces.append(previews ? "--preview" : "--no-preview")
        let format = outputFormat
        pieces += [quotes(prompt), "\(format.width)x\(format.height)",
                     "\(fixedSeed ? seed : (selectedEntry?.seed ?? seed))"]
        // The app's own library needs no mention; another one is the app's environment.
        let root = library.local.path == Library.standard.root.path ? ""
            : "SILICONED_ROOT=" + quotes(library.local.path) + " "
        return root + pieces.joined(separator: " ") + notes.map { "\n# \($0)" }.joined()
    }

    func copyCommand() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(command(), forType: .string)
        flash(String(localized: "Command copied"))
    }

    /// A transient confirmation ("Image copied") that clears by itself.
    func flash(_ message: String) {
        toast = message
        toastTask?.cancel()
        toastTask = Task {
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled else { return }
            toast = nil
        }
    }
}

/// **The three files an image leaves as.** PNG carries the generation metadata (`PNG.data`); JPEG and
/// HEIC are written without any, on purpose: a picture to post or to send says nothing of how it was made.
enum ExportFormat: String, CaseIterable {
    case png, jpeg, heic

    var label: String { self == .jpeg ? "JPEG" : rawValue.uppercased() }
    var fileExtension: String { self == .jpeg ? "jpg" : rawValue }
    var type: UTType { self == .png ? .png : self == .jpeg ? .jpeg : .heic }

    init?(extension ext: String) {
        switch ext.lowercased() {
        case "png": self = .png
        case "jpg", "jpeg": self = .jpeg
        case "heic", "heif": self = .heic
        default: return nil
        }
    }

    /// The image as HEIC, tagged sRGB, with no metadata — `nil` if this Mac cannot encode it.
    static func heic(_ image: CGImage, quality: Double) -> Data? {
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output as CFMutableData, UTType.heic.identifier as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        return CGImageDestinationFinalize(destination) ? output as Data : nil
    }
}

/// **Keeps a save panel's file name and type with its « Format: » pop-up**: choosing HEIC renames
/// `name.png` to `name.heic` and lets the panel accept only that.
@MainActor
final class FormatFollower: NSObject {
    private weak var panel: NSSavePanel?

    /// The pop-up holds its follower: both live as long as the panel's accessory.
    static func attach(_ panel: NSSavePanel, to popup: NSPopUpButton) {
        let follower = FormatFollower()
        follower.panel = panel
        popup.target = follower
        popup.action = #selector(changed(_:))
        objc_setAssociatedObject(popup, &key, follower, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
    }

    private static nonisolated(unsafe) var key = 0

    @objc private func changed(_ popup: NSPopUpButton) {
        guard let panel else { return }
        let format = ExportFormat.allCases[max(popup.indexOfSelectedItem, 0)]
        panel.allowedContentTypes = [format.type]
        panel.nameFieldStringValue = (panel.nameFieldStringValue as NSString).deletingPathExtension + "." + format.fileExtension
    }
}
