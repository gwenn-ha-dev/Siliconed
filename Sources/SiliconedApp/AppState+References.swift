// **`AppState`: how an image becomes one of the render's** — a file, a history image, ⌘V, « Edit with … ». They all land
// in the Prompt Lab (`AppState.inputs`, `AppState+Inputs.swift`), read off the main thread when the render runs.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

extension AppState {
    // MARK: Reference (editing)

    // **Editing by reference** (Qwen-Image-2.1) lives here. The re-noised starting image (img2img, SDEdit) of
    // Z-Image, Anima and Krea 2 is another input, chosen in the Prompt Lab: `AppState+Start.swift`.

    /// A file as the render reads it — the file itself (`ImageRGB(contentsOf:)`, upright), its vignette and size.
    func read(_ url: URL) -> ReferenceImage? {
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

    /// A file chosen to take the place of the image `id` of the Prompt Lab, with its role.
    func replaceReference(_ id: ReferenceImage.ID) {
        guard inputIndex(shown: id) != nil else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.prompt = String(localized: "Choose")
        guard panel.runModal() == .OK, let url = panel.url, let image = read(url),
              let i = inputIndex(shown: id) else { return }
        replaceInput(inputs[i].id, with: image)
    }

    /// "Edit this image": a history image becomes **image 1** — the one edited, whose format the
    /// output takes. It replaces the previous image 1 (editing a result again is the usual next
    /// gesture); the other references stay.
    func placeReference(_ entry: Entry) {
        guard editingPossible, let image = fromHistory(entry) else { return }
        if let edited = inputs.firstIndex(where: { $0.role == .edited }) {
            replaceInput(inputs[edited].id, with: image, bytes: entry.png)
        } else {
            addInput(image, bytes: entry.png, role: .edited)
        }
        flash(String(localized: "Image 1 set: say what changes"))
    }

    /// « Edit with Qwen-Image-2.1 »: the rack takes that model, and the image becomes its image 1.
    func edit(_ entry: Entry, with model: String) {
        if identifier != model { identifier = model }
        placeReference(entry)
    }

    /// **⌘V outside a text field**: an image copied in Procreate, Photoshop, Preview or the Finder goes to the Prompt Lab,
    /// as a drop on the canvas would, and the bar unfolds on it. `false` when the pasteboard holds no image, so the key goes
    /// on to the menu.
    func pasteImage() -> Bool {
        let pasteboard = NSPasteboard.general
        let files = (pasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? [])
            .filter { UTType(filenameExtension: $0.pathExtension)?.conforms(to: .image) == true }
        // Pixels without a file: re-encoded as a PNG once, whatever came (TIFF from most apps).
        let pixels: Data? = files.isEmpty ? pasteboard.data(forType: .png) ?? NSImage(pasteboard: pasteboard)
            .flatMap { $0.cgImage(forProposedRect: nil, context: nil, hints: nil) }
            .flatMap { try? PNG.data($0) } : nil
        guard !files.isEmpty || pixels != nil else { return false }
        let lab = LabModel.shared
        if let pixels {
            lab.load(image: pixels, name: String(localized: "pasted image"))
        } else {
            for url in files { lab.load(.file(url)) }
        }
        lab.unfold()
        return true
    }

}
