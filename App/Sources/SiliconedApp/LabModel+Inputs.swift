// `LabModel`: the rack's images and their roles, and the named subjects.

import AppKit
import Siliconed
import SwiftUI

extension LabModel {
    // MARK: The images

    /// « Start From This Image »: the gesture that makes an img2img's start — the previous start, if any, goes back to neutral.
    func startFrom(_ picture: InputImage) { setRole(.start, of: picture) }

    func setRole(_ role: LabInputs.Role, of picture: InputImage) { rack.setRole(role, of: picture.id) }

    func remove(_ picture: InputImage) {
        if describing == picture.id { stop() }
        rack.removeInput(picture.id)
    }

    /// **An image comes in as bytes** (a drop of pixels, ⌘V): a history image (dragged from the column) as « Send to the Prompt
    /// Lab » brings it — its pixels, its ordinal for a mirror —, any other decoded upright: its pixels for an edit, its bytes
    /// for a start. It takes the role a newcomer takes. Decoded off the main thread, with its vignette: 51 ms for a 2048² PNG
    /// on an M5 Pro.
    func load(image data: Data, name: String = String(localized: "dropped image")) {
        arrive {
            if let entry = self.rack.history.first(where: { $0.png.count == data.count && $0.png == data }),
               let image = self.rack.fromHistory(entry) {
                self.rack.addInput(image, bytes: entry.png)
                return true
            }
            let decoded = await Task.detached(priority: .userInitiated) { () -> (image: CGImage, vignette: CGImage)? in
                guard let source = CGImageSourceCreateWithData(data as CFData, nil),
                      let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
                return (cg, FinishedImage.reduce(cg, side: 240) ?? cg)
            }.value
            guard let decoded else { return false }
            self.rack.addInput(ReferenceImage(source: .image(decoded.image), vignette: NSImage(cgImage: decoded.vignette, size: .zero),
                                              name: name, width: decoded.image.width, height: decoded.image.height), bytes: data)
            return true
        }
    }

    /// The image of a gesture: bytes as above, a file as the render reads a file (`AppState.read`) — the same pixels for an
    /// edit and for a start (`ImageRGB(data:)` reads bytes as the file).
    func load(_ dropped: DroppedImage) {
        switch dropped {
        case .png(let data): load(image: data)
        case .file(let url):
            arrive {
                guard let image = await self.rack.read(url) else { return false }
                self.rack.addInput(image)
                return true
            }
        }
    }

    func arrive(_ read: @escaping @MainActor () async -> Bool) {
        let before = arriving
        arriving = Task {
            await before?.value
            failure = await read() ? nil : String(localized: "This image could not be read.")
        }
    }

    func chooseImages() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = true
        panel.prompt = String(localized: "Choose")
        panel.message = String(localized: "Images to describe, to start from, or to name.")
        guard panel.runModal() == .OK else { return }
        for url in panel.urls { load(.file(url)) }
    }

    // MARK: The subjects

    /// The sheet opens on an image of the rack: once the subject is created, the image becomes it.
    func name(_ picture: InputImage) {
        guard let data = Self.bytes(of: picture), let draft = SubjectDraft(new: .png(data), taken: subjects.all.map(\.name)) else {
            failure = String(localized: "This image could not be read.")
            return
        }
        naming = (picture.id, subjects.all.count)
        self.draft = draft
    }

    /// The open panel, then the sheet: an image to name that is not in the zone.
    func chooseImageToName() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.prompt = String(localized: "Choose")
        panel.message = String(localized: "An image to name: a person, a garment, an object, a place or a style.")
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard let draft = SubjectDraft(new: .file(url), taken: subjects.all.map(\.name)) else {
            failure = String(localized: "This image could not be read.")
            return
        }
        naming = nil
        self.draft = draft
    }

    func edit(_ s: NamedSubjects.Subject) {
        naming = nil
        draft = SubjectDraft(editing: s, library: subjects)
    }

    /// The sheet is closed: an image it named leaves the zone — it is a subject now.
    func sheetClosed() {
        defer { naming = nil }
        guard let naming, subjects.all.count > naming.known else { return }
        rack.removeInput(naming.picture)
    }

    /// Asked first: texts that cite it are refused afterwards.
    func confirmDeletion(_ s: NamedSubjects.Subject) {
        let alert = NSAlert()
        alert.messageText = String(localized: "Delete @\(s.name)?")
        alert.informativeText = String(localized: "Prompts that cite it will be refused.")
        alert.addButton(withTitle: String(localized: "Delete"))
        alert.addButton(withTitle: String(localized: "Cancel"))
        alert.buttons[0].hasDestructiveAction = true
        if alert.runModal() == .alertFirstButtonReturn { deleteSubject(s) }
    }

    func deleteSubject(_ s: NamedSubjects.Subject) { subjects.delete(s) }

    /// `@name` inserted where the caret is (a click on a subject) — appended when the field is not on screen. A space
    /// before it when it would glue to a word.
    func cite(_ s: NamedSubjects.Subject) {
        if let field = PromptTextView.current, field.window != nil {
            field.insertMention(s.name)
            return
        }
        let glue = text.last.map { !$0.isWhitespace } ?? false
        text += (glue ? " " : "") + "@\(s.name) "
    }

    /// A subject renamed: the text's mentions of it follow.
    func renameMentions(_ old: String, to new: String) {
        var renamed = text
        for m in NamedSubjects.mentions(in: text).reversed() where NamedSubjects.key(m.name) == NamedSubjects.key(old) {
            if let r = Range(NSRange(m.range, in: text), in: renamed) { renamed.replaceSubrange(r, with: "@" + new) }
        }
        if renamed != text { text = renamed }
    }
}
