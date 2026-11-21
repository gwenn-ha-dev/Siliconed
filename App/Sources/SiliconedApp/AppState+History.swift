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

    /// **A new image enters the history** — rendered, enlarged, cropped, or come from the other Mac: inserted at `index`
    /// (newest first), saved when the setting asks (`saveIfAsked`), and shown when `shows` says so — chosen, nothing marked.
    /// Each caller says when its image takes the canvas: a render while the canvas follows the renders, an enlargement or a
    /// crop always (the gesture asked for it), the other Mac's newest while the canvas follows.
    func admit(_ entry: Entry, image: CGImage, at index: Int = 0, shows: Bool) {
        history.insert(entry, at: index)
        saveIfAsked(entry)
        guard shows else { return }
        selectedImage = (entry.id, image)
        selection = entry.id
        marked = []; anchor = entry.id
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
        // The image already chosen, a series over it: `selection` does not change, the canvas leaves the grid all the same.
        shownSeries = nil
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

    /// The run between two images, in the column's order — a folded series' cells left out, as the arrows leave them.
    private func markRun(from a: Entry.ID, to b: Entry.ID) {
        let strip = stripImages
        guard let i = strip.firstIndex(where: { $0.id == a }), let j = strip.firstIndex(where: { $0.id == b }) else { return }
        let run = Set(strip[min(i, j)...max(i, j)].map(\.id))
        marked = run.count > 1 ? run : []
        reveal(b)
    }

    /// ⌘A in the strip: every image.
    func markAll() {
        guard history.count > 1 else { return }
        marked = Set(history.map(\.id))
        if selection == nil { selection = history.first?.id }
    }

    /// **A lasso drawn over the history grid**: what it covers is marked — added to what was (`kept`, ⌘ or ⇧ held as it
    /// began) —, the first of them in the column's order shown. Covering one image alone shows it.
    func lasso(_ covered: Set<Entry.ID>, kept: Set<Entry.ID>) {
        let all = covered.union(kept)
        guard let first = stripImages.first(where: { all.contains($0.id) })?.id else { return }
        let m = all.count > 1 ? all : []
        if marked != m { marked = m }
        if selection != first { reveal(first) }
        anchor = first
    }

    /// Esc: back to the one image shown.
    func unmark() { marked = [] }

    /// Whether a thumbnail wears the selection's edge: the image shown, or one of those marked.
    /// `chosen`: `selectedEntry?.id`, read once by a list that asks for each of its tiles (`selectedEntry` walks the history).
    func isMarked(_ id: Entry.ID, chosen: Entry.ID?) -> Bool {
        marked.count > 1 ? marked.contains(id) : id == chosen
    }

    /// **What the image commands act on**: the marked images, in the strip's order — or the one shown.
    var targets: [Entry] {
        marked.count > 1 ? history.filter { marked.contains($0.id) } : selectedEntry.map { [$0] } ?? []
    }

    /// The neighbouring image in the column: −1 up (more recent), +1 down (older), as the arrow keys move along it — a
    /// folded series' cells skipped. Watching a render, it starts from the most recent image. With `extending` (⇧), the
    /// run from the anchor is marked.
    func chooseAdjacent(_ offset: Int, extending: Bool = false) {
        let strip = stripImages
        guard let e = selectedEntry, !strip.isEmpty else { return }
        let i = strip.firstIndex { $0.id == e.id } ?? nearest(e, in: strip)
        if inProgress && followsRender { show(strip[offset > 0 ? i : 0].id); return }
        let j = min(max(i + offset, 0), strip.count - 1)
        if extending, let a = anchor { markRun(from: a, to: strip[j].id) } else { show(strip[j].id) }
    }

    /// Where an image the column does not show — a cell folded since it was chosen — stands among those it shows: at the
    /// first older one.
    private func nearest(_ e: Entry, in strip: [Entry]) -> Int {
        guard let k = history.firstIndex(where: { $0.id == e.id }) else { return 0 }
        let older = Set(history[k...].map(\.id))
        return strip.firstIndex { older.contains($0.id) } ?? strip.count - 1
    }

    /// Home, End: the most recent image, the oldest — in the column's order.
    func chooseEnd(oldest: Bool) {
        let strip = stripImages
        guard let e = oldest ? strip.last : strip.first else { return }
        show(e.id)
    }

    func delete(_ entry: Entry) { delete([entry]) }

    /// **Removes images from the history** — undone by ⌘Z, each back at its place. The image shown
    /// passes to the nearest one left.
    func delete(_ entries: [Entry]) {
        // In a mirror the other Mac's history is the one: it removes, and says so (`removed`); ⌘Z asks the images back (`removeRemotely`).
        if mirror != nil { removeRemotely(entries.map(\.ordinal)); return }
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
        let n = unsaved.count
        // Everything saved, and ⌘Z brings it back: nothing to confirm.
        if n == 0 { delete(history); return }
        let alert = NSAlert()
        alert.messageText = String(localized: "Clear the history?")
        alert.informativeText = String(localized: "\(n) of these images were never saved. You can undo this with ⌘Z while the app is open.")
        alert.addButton(withTitle: String(localized: "Clear History"))
        alert.addButton(withTitle: String(localized: "Cancel"))
        alert.buttons[0].hasDestructiveAction = true
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        delete(history)
    }

    /// Save: in the format last used (PNG the first time — it carries the prompt, the seed, the model),
    /// JPEG or HEIC from the panel's pop-up. JPEG and HEIC carry no metadata: they are plain pictures. `true` once written.
    @discardableResult
    func save(_ entry: Entry) async -> Bool {
        let panel = NSSavePanel()
        let base = exportName(entry)
        panel.allowedContentTypes = [exportFormat.type]
        panel.nameFieldStringValue = base + "." + exportFormat.fileExtension
        panel.canCreateDirectories = true
        let (accessory, kind) = Self.formatAccessory(selected: exportFormat, note: true)
        FormatFollower.attach(panel, to: kind)
        accessory.setAccessibilityLabel(String(localized: "Format:"))
        panel.accessoryView = accessory
        guard panel.runModal() == .OK, let url = panel.url else { return false }
        exportFormat = ExportFormat(extension: url.pathExtension) ?? exportFormat
        if let failed = await written([entry], as: ExportFormat(extension: url.pathExtension) ?? .png, at: { _ in url }) {
            problem = Problem(EngineError.imageWriteFailed(file: failed.path))
            return false
        }
        flash(String(localized: "Saved: \(url.lastPathComponent)"))
        return true
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
        caption.textColor = Palette.secondaryInk
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
    func export(_ entries: [Entry]) async -> Bool {
        guard !entries.isEmpty else { return false }
        if entries.count == 1 { return await save(entries[0]) }
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.prompt = String(localized: "Export")
        panel.message = String(localized: "Export \(entries.count) images to a folder.")
        let (accessory, kind) = Self.formatAccessory(selected: exportFormat, note: true)
        panel.accessoryView = accessory
        panel.isAccessoryViewDisclosed = true
        guard panel.runModal() == .OK, let folder = panel.url else { return false }
        exportFormat = ExportFormat.allCases[max(kind.indexOfSelectedItem, 0)]
        let ext = exportFormat.fileExtension
        if let failed = await written(entries, as: exportFormat, at: { Self.freeURL(in: folder, base: $0, ext: ext) }) {
            problem = Problem(EngineError.imageWriteFailed(file: failed.path))
            return false
        }
        flash(String(localized: "\(entries.count) images exported to \(folder.lastPathComponent)"))
        return true
    }

    /// **An image's file name**: the model, its LoRAs, the steps if they are not the model's, the
    /// format, the seed — two different settings never share a name (`silicontrol save` writes the same).
    func exportName(_ e: Entry) -> String {
        let r = e.settings
        let modelSteps = cards.first { $0.id == r.identifier }?.defaultSteps
        return ([r.identifier.replacingOccurrences(of: "/", with: "-")]
                + r.loras.map { "lora-" + RemoteControl.shortName($0.path) + ($0.strength == 1 ? "" : String(format: "@%.2f", $0.strength)) }
                + (r.steps != modelSteps ? ["p\(r.steps)"] : [])
                + (r.editing ? ["edit"] : [])
                + ["\(e.width)x\(e.height)", "\(e.seed)"]
                // A variation is another image of the same seed: `v<seed>x<strength in hundredths>`, in
                // order — the CLI's name for it.
                + r.variations.map { "v\($0.seed)x\(Int(($0.strength * 100).rounded()))" }).joined(separator: "-")
    }

    /// `base.png` in the folder, or `base-2.png`, `base-3.png`… never over an existing file.
    nonisolated static func freeURL(in folder: URL, base: String, ext: String) -> URL {
        var url = folder.appendingPathComponent("\(base).\(ext)")
        var k = 2
        while FileManager.default.fileExists(atPath: url.path) {
            url = folder.appendingPathComponent("\(base)-\(k).\(ext)")
            k += 1
        }
        return url
    }

    /// Writes an image as its extension says (`ExportFormat.file`), and counts it as saved — one image, for the remote control.
    func write(_ entry: Entry, to url: URL) throws {
        guard let data = (ExportFormat(extension: url.pathExtension) ?? .png).file(entry.png) else {
            throw EngineError.imageWriteFailed(file: url.path)
        }
        try PNG.write(data, to: url.path)
        saved.insert(entry.id)
    }

    /// **Images written off the main thread**: JPEG and HEIC decode and encode each one again, seconds for a lot. Each file
    /// takes its place as it is written (`place`, given the export name, sees those written before it); the images written
    /// count as saved. The file the writing failed on, if it did — those before it are written.
    private func written(_ entries: [Entry], as format: ExportFormat,
                         at place: @escaping @Sendable (String) -> URL) async -> URL? {
        let items = entries.map { (id: $0.id, png: $0.png, base: exportName($0)) }
        let (done, failed) = await Task.detached(priority: .userInitiated) { () -> ([Entry.ID], URL?) in
            var done: [Entry.ID] = []
            for item in items {
                let url = place(item.base)
                guard let data = format.file(item.png), (try? PNG.write(data, to: url.path)) != nil else { return (done, url) }
                done.append(item.id)
            }
            return (done, nil)
        }.value
        saved.formUnion(done)
        return failed
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

    /// The `silicontrol` command that redoes what « Generate » would — the equivalent of Draw Things Headless's "Copy as
    /// curl", for the `silicontrol` command shipped in the bundle: `add`, or `grid` when something varies (the series' two
    /// axes as `--x` and `--y`). The LoRAs switched on go by the name `Library.lora` matches; the edit's images as `--ref` and
    /// the starting image as `--image`, by their file or their `i` name while they are in the history. What has no command
    /// form (an image that is neither, the variations, a third thing varying) is said in a comment, never left out silently.
    func command() -> String {
        func quotes(_ s: String) -> String { "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'" }
        func word(_ image: ReferenceImage) -> String? {
            if case .file(let url) = image.source { return quotes(url.path) }
            if let o = image.ordinal, history.contains(where: { $0.ordinal == o }) { return "i\(o)" }
            return nil
        }
        let layout = seriesLayout
        let stack = activeLoRAs
        let names = stack.map { RemoteControl.shortName($0.path) }
        var pieces = ["silicontrol", layout.columns == nil ? "add" : "grid"]
        var notes: [String] = []
        func axis(_ k: SeriesKey) -> String? { gridAxis(k).map { GridWords.command($0, loras: names) } }
        if let x = layout.columns.flatMap(axis) { pieces += ["--x", quotes(x)] }
        if let y = layout.rows.flatMap(axis) { pieces += ["--y", quotes(y)] }
        for a in layout.excess.compactMap(axis) { notes.append(String(localized: "\(a) varies too: a grid takes two axes")) }
        pieces += ["--model", identifier]
        for (l, name) in zip(stack, names) { pieces += ["--lora", quotes("\(name):\(String(format: "%.2f", l.strength))")] }
        if steps != card?.defaultSteps { pieces += ["--steps", "\(steps)"] }
        if effectiveDetail != .normal { pieces += ["--detail", effectiveDetail.rawValue] }
        if effectiveSolver != .euler { pieces += ["--solver", effectiveSolver.rawValue] }
        for (n, reference) in renderReferences.enumerated() {
            if let w = word(reference) { pieces += ["--ref", w] } else {
                notes.append(String(localized: "image \(n + 1) has no file: save it, then --ref FILE"))
            }
        }
        if usedStart != nil, let start = inputs.first(where: { $0.role == .start })?.shown {
            if let w = word(start) { pieces += ["--image", w] } else {
                notes.append(String(localized: "the starting image has no file: save it, then --image FILE"))
            }
            pieces += ["--strength", String(format: "%.2f", startStrength)]
        }
        if !variations.isEmpty {
            notes.append(String(localized: "the variations have no option: save the image, then silicontrol again FILE.png"))
        }
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
        // Read out by VoiceOver: a toast lasts two seconds and is not in the focus order.
        if let element = NSApp.mainWindow ?? NSApp.keyWindow {
            NSAccessibility.post(element: element, notification: .announcementRequested,
                                 userInfo: [.announcement: message, .priority: NSAccessibilityPriorityLevel.high.rawValue])
        }
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

    /// **The file an image leaves as**: the PNG itself, its metadata kept; JPEG at 0.9 and HEIC at 0.85, decoded and encoded
    /// again, plain pictures. `nil` when it cannot be encoded.
    func file(_ png: Data) -> Data? {
        if self == .png { return png }
        guard let cg = Entry.decode(png) else { return nil }
        return self == .jpeg ? NSBitmapImageRep(cgImage: cg).representation(using: .jpeg, properties: [.compressionFactor: 0.9])
            : Self.heic(cg, quality: 0.85)
    }

    /// The image as HEIC, tagged sRGB, with no metadata — `nil` if this Mac cannot encode it.
    private static func heic(_ image: CGImage, quality: Double) -> Data? {
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
