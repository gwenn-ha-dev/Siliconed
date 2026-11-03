// **Save Every Image To…** (Settings ▸ General): each new image of the history written to a folder as it comes,
// as « Save… » writes it — the PNG with its prompt, seed and model. **Off by default**: the app stays incognito
// unless the user turns this on, and then the folder is the one place it writes.
//
//     a render, a variation, an edit     ┐
//     a grid's sheet, its finished cells ├─▶ <folder>/<exportName>.png   (-2, -3… beside a different file of that name)
//     a crop, an enlargement             ┘
//
// - **What is saved is what enters the history new**: an exploration's sketches stay in their panel, as they stay
//   out of the history; an image ⌘Z brings back is not new, nor one another Mac's user rendered here (`owner`): their
//   images are theirs. Computing on another Mac, the images that come back are this Mac's, and are saved here.
// - **The same file is never written twice**: a file of that name holding the same bytes is the image already
//   saved — what happens when the other Mac sends its images again after a cut, or a redo gives the same bits.
// - **A failure is said** like a failed « Save… » (`Problem`); the switch stays on, the next image tries again.
// - The folder is a preference (`UserDefaults`), kept until the user turns the setting off. It is never created: a
//   disk that is not there is a failure, not a new folder on the startup disk.

import AppKit
import Siliconed

/// The setting, one per Mac: the local state and a mirror's both read it.
@MainActor @Observable
final class EveryImage {
    static let shared = EveryImage()
    private static let key = "saveEveryImageFolder"

    /// Where each new image goes; `nil`: off, the default.
    var folder: URL? = UserDefaults.standard.string(forKey: EveryImage.key).map { URL(fileURLWithPath: $0, isDirectory: true) } {
        didSet {
            if let folder { UserDefaults.standard.set(folder.path, forKey: Self.key) } else { UserDefaults.standard.removeObject(forKey: Self.key) }
        }
    }

    /// The panel that turns the setting on, or moves it to another folder; Cancel changes nothing.
    func choose() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.prompt = String(localized: "Choose")
        panel.message = String(localized: "Each new image will be saved here as a PNG, with its prompt, seed and model.")
        if let folder { panel.directoryURL = folder }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        folder = url
    }

    /// One writer for every image: two images of one name never race for it.
    private nonisolated static let writer = Writer()

    nonisolated func write(_ png: Data, in folder: URL, base: String) async throws(EngineError) -> URL {
        try await Self.writer.write(png, in: folder, base: base)
    }

    private actor Writer {
        func write(_ png: Data, in folder: URL, base: String) throws(EngineError) -> URL {
            var isFolder: ObjCBool = false
            guard FileManager.default.fileExists(atPath: folder.path, isDirectory: &isFolder), isFolder.boolValue else {
                throw .imageWriteFailed(file: folder.path)
            }
            var url = folder.appendingPathComponent("\(base).png")
            var k = 2
            while FileManager.default.fileExists(atPath: url.path) {
                if (try? Data(contentsOf: url)) == png { return url }
                url = folder.appendingPathComponent("\(base)-\(k).png")
                k += 1
            }
            try PNG.write(png, to: url.path)
            return url
        }
    }
}

extension AppState {
    /// **A new image of the history, saved when the setting asks** — called where one is inserted new.
    func saveIfAsked(_ entry: Entry) {
        guard let folder = EveryImage.shared.folder, mirror != nil || entry.owner == nil else { return }
        let png = entry.png, base = exportName(entry), id = entry.id
        Task {
            do {
                _ = try await EveryImage.shared.write(png, in: folder, base: base)
                saved.insert(id)
            } catch {
                problem = Problem(error)
            }
        }
    }
}
