// **`AppState`: an edit's reference images** — chosen, dragged, taken from the history, read off the
// main thread. The state itself is in `AppState.swift`.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

extension AppState {
    // MARK: Reference (editing)

    // **No img2img in the app** (decided 2026-09-25): the re-noised starting image
    // (SDEdit) does not follow an instruction prompt and keeps a style only by losing the likeness.
    // The engine and the CLI keep it (`Request.image`, `--image`); the app offers only editing
    // by reference, where the model supports it.

    private func read(_ url: URL) -> ReferenceImage? {
        guard let vignette = NSImage(contentsOf: url), vignette.isValid, let size = ReferenceImage.size(of: url) else {
            problem = Problem(EngineError.imageUnreadable(file: url.path))
            return nil
        }
        return ReferenceImage(source: .file(url), vignette: vignette, name: url.lastPathComponent,
                              width: size.width, height: size.height)
    }

    func fromHistory(_ entry: Entry) -> ReferenceImage? {
        guard let cg = entry.image() else { return nil }
        return ReferenceImage(source: .image(cg), vignette: NSImage(cgImage: entry.vignette, size: .zero),
                             name: String(localized: "image from history (seed \(String(entry.seed)))"),
                             width: entry.width, height: entry.height, ordinal: entry.ordinal)
    }

    /// The open panel: as many images as there are free slots, added after those already set.
    func chooseReferences() {
        guard canAddReference, nextReferenceRefusal() == nil else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = maxReferences - references.count > 1
        panel.prompt = String(localized: "Choose")
        panel.message = references.isEmpty
            ? String(localized: "The image to edit: the prompt says what changes (“change her jacket to red”).")
            : String(localized: "Another image the prompt can draw from (“the dog of image 2”).")
        guard panel.runModal() == .OK else { return }
        addReferences(panel.urls)
    }

    /// Files chosen or dropped, appended in their order; beyond the model's maximum, the rest is
    /// left out and the toast says so.
    func addReferences(_ urls: [URL]) {
        // Files only: a dragged thumbnail (text) must not come back as a URL to read.
        let urls = urls.filter(\.isFileURL)
        guard editingPossible, !urls.isEmpty else { return }
        let slots = max(0, maxReferences - references.count)
        // As many as this Mac's memory takes, too (`nextReferenceRefusal`, one image at a time).
        var free = 0, refusal: String?
        while free < min(slots, urls.count) {
            refusal = nextReferenceRefusal(adding: free)
            guard refusal == nil else { break }
            free += 1
        }
        references += urls.prefix(free).compactMap(read)
        if let refusal {
            flash(refusal)
        } else if urls.count > free {
            flash(String(localized: "This model reads \(maxReferences) images at most"))
        }
    }

    /// A file chosen to take the place of the reference `id`, at its number.
    func replaceReference(_ id: ReferenceImage.ID) {
        guard references.contains(where: { $0.id == id }) else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.prompt = String(localized: "Choose")
        guard panel.runModal() == .OK, let url = panel.url, let image = read(url),
              let i = references.firstIndex(where: { $0.id == id }) else { return }
        references[i] = image
    }

    func removeReference(_ id: ReferenceImage.ID) {
        references.removeAll { $0.id == id }
    }

    /// Moves a reference to the place `index` (0: image 1), the others shifting. Image 1 is the one
    /// edited: putting another image first changes what is edited, and the output's format.
    func moveReference(_ id: ReferenceImage.ID, to index: Int) {
        guard let i = references.firstIndex(where: { $0.id == id }), i != index,
              references.indices.contains(index) else { return }
        references.insert(references.remove(at: i), at: index)
    }

    /// One place left (−1, towards image 1) or right (+1).
    func moveReference(_ id: ReferenceImage.ID, by offset: Int) {
        guard let i = references.firstIndex(where: { $0.id == id }) else { return }
        moveReference(id, to: i + offset)
    }

    /// "Edit this image": a history image becomes **image 1** — the one edited, whose format the
    /// output takes. It replaces the previous image 1 (editing a result again is the usual next
    /// gesture); the other references stay.
    func placeReference(_ entry: Entry) {
        guard editingPossible, let image = fromHistory(entry) else { return }
        if references.isEmpty { references = [image] } else { references[0] = image }
        flash(String(localized: "Image 1 set: say what changes"))
    }

    /// « Edit with Qwen-Image-2.1 »: the rack takes that model, and the image becomes its image 1.
    func edit(_ entry: Entry, with model: String) {
        if identifier != model { identifier = model }
        placeReference(entry)
    }

    /// **⌘V outside a text field**: an image copied in Procreate, Photoshop, Preview or the Finder
    /// goes where a drop on the Images box would — image or subject. `false` when the pasteboard holds
    /// no image, so the key goes on to the menu.
    func pasteImage() -> Bool {
        let pasteboard = NSPasteboard.general
        let files = (pasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? [])
            .filter { UTType(filenameExtension: $0.pathExtension)?.conforms(to: .image) == true }
        // Pixels without a file: re-encoded as a PNG once, whatever came (TIFF from most apps).
        let pixels: Data? = files.isEmpty ? pasteboard.data(forType: .png) ?? NSImage(pasteboard: pasteboard)
            .flatMap { $0.cgImage(forProposedRect: nil, context: nil, hints: nil) }
            .flatMap { try? PNG.data($0) } : nil
        guard !files.isEmpty || pixels != nil else { return false }
        guard editingPossible else {
            flash(String(localized: "To start from an image, choose a model that edits"))
            return true
        }
        if subjectsActive {
            nameSubject(files.first.map { .file($0) } ?? .png(pixels!))
        } else if let pixels {
            addReference(png: pixels, name: String(localized: "pasted image"))
        } else {
            addReferences(files)
        }
        return true
    }

    /// PNG bytes dropped on the Images box: a strip image is recognized by its bytes (its name, its
    /// curtain); any other PNG is taken as it is.
    func addReference(png data: Data, name: String = String(localized: "dropped image")) {
        guard editingPossible else { return }
        guard canAddReference else { flash(String(localized: "This model reads \(maxReferences) images at most")); return }
        if let refusal = nextReferenceRefusal() { flash(refusal); return }
        if let entry = history.first(where: { $0.png == data }), let image = fromHistory(entry) {
            references.append(image)
        } else if let source = CGImageSourceCreateWithData(data as CFData, nil),
                  let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) {
            let vignette = FinishedImage.reduce(cg, side: 240) ?? cg
            references.append(ReferenceImage(source: .image(cg), vignette: NSImage(cgImage: vignette, size: .zero),
                                             name: name, width: cg.width, height: cg.height))
        } else {
            flash(String(localized: "This image can't be read"))
        }
    }
}
