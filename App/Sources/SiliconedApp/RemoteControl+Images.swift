// The images a command names: a starting image, an edit's image, a file.

import AppKit
import Foundation
import Measurement
import Observation
import Siliconed
import SilicontrolHelp

extension RemoteControl {
    /// **An image the remote control names as a starting image**: `i7` of the history (its PNG's bytes), or a file — read as
    /// `siliconed-dev render --image` reads it.
    func startImage(_ word: String, cwd: String) throws -> ReferenceImage {
        if let entry = image(word) { return app.startImage(from: entry) }
        return try fileImage(word, cwd: cwd)
    }

    /// **`start-image`**: the rack's starting image, as the Prompt Lab carries it and the rack's Images box retouches it —
    /// `start-image` alone says what it holds; `start-image IMAGE|i7 [--strength S]` carries one (the size set to the nearest
    /// recommended format, `carryStart`); `--strength S` alone retouches its strength; `--remove` takes it out.
    func startImage(_ action: GestureOrder.StartImageAction, cwd: String) throws -> [String: Any] {
        func refuseWithoutImg2Img() throws {
            guard app.startsFromImage else {
                let name = app.card?.name ?? app.identifier
                throw Refusal("img2img", String(localized: "\(name) doesn’t start from an image"),
                              help: app.editingPossible ? String(localized: "--ref <image> to edit it (help edit)")
                                                        : String(localized: "models that do have \"start_image\": true in \"silicontrol models\""))
            }
        }
        switch action {
        case .status:
            break
        case .carry(let word, let strength):
            try refuseWithoutImg2Img()
            app.carryStart(try startImage(word, cwd: cwd), strength: strength)
        case .strength(let s):
            guard app.startImage != nil else {
                throw Refusal("no_start_image", String(localized: "the rack holds no starting image"),
                              help: "silicontrol start-image <image|i7>")
            }
            app.setStartStrength(s)
        case .remove:
            app.removeStart()
        }
        return ["ok": true, "rack": rackSettings()]
    }

    /// The starting image the rack holds, as the answers say it — `nil` when none.
    func startJSON() -> [String: Any]? {
        guard let s = app.startImage else { return nil }
        var o: [String: Any] = ["name": s.name, "width": s.width, "height": s.height,
                                "strength": Response.number(app.startStrength), "used": app.usedStart != nil]
        if let r = app.startRedraws { o["redrawn_steps"] = r.redrawn; o["of_steps"] = r.of }
        if app.usedStart != nil { o["image_ignored"] = app.startIgnored }
        return o
    }

    /// An edit's image as the command line names it: `i7` of the history, or a file.
    func referenceImage(_ word: String, cwd: String) throws -> ReferenceImage {
        if let entry = image(word), let reference = app.fromHistory(entry) { return reference }
        return try fileImage(word, cwd: cwd)
    }

    /// An image file, from the client's folder: refused if ImageIO cannot read it. Read here, on the main thread: a job takes
    /// its place in the queue when its command arrives, and a read awaited elsewhere would let a later command pass it.
    func fileImage(_ path: String, cwd: String) throws -> ReferenceImage {
        let url = URL(fileURLWithPath: absolute(path, cwd))
        guard let image = ReferenceImage.file(url) else {
            let path = url.path
            throw Refusal("ref", String(localized: "image unreadable or missing: \(path)"))
        }
        return image
    }
}
