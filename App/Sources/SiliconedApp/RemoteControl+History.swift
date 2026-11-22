// The history's commands: `history`, `save`, `vary`, `enlarge`, and the image a command names by its ordinal.

import AppKit
import Foundation
import Measurement
import Observation
import Siliconed
import SilicontrolHelp

extension RemoteControl {
    func history(_ arguments: [String]) throws -> [String: Any] {
        var n = Int.max
        let words = try Self.flags(arguments).allWords
        guard words.count <= 1 else { throw Refusal("usage", "history [N]") }
        if let a = words.first {
            guard let v = Int(a), v >= 1 else { throw Refusal("usage", String(localized: "history [N]: N an integer ≥ 1, not \"\(a)\"")) }
            n = v
        }
        return ["ok": true, "images": app.history.prefix(n).map { json($0) }]
    }

    /// **`save i7 [i8 …] DEST [--format png|jpeg|heic]`**: « Save… » and « Export… » without the panel. One image to a file takes its format from the
    /// file's extension (.png, .jpg/.jpeg, .heic), as the panel's Format pop-up does; into a folder, each image under its export name, in `--format`
    /// (PNG by default). Only PNG keeps the prompt, the seed and the model.
    func save(_ arguments: [String], cwd: String) throws -> [String: Any] {
        // A file or a folder whose name begins with `--` goes after a lone `--`.
        let flags = try Self.flags(arguments, valued: ["--format"])
        var kind = ExportFormat.png
        if let v = flags.value("--format") {
            guard let k = ExportFormat(extension: v) else {
                throw Refusal("usage", String(localized: "--format expects png, jpeg or heic, not \"\(v)\""))
            }
            kind = k
        }
        let arguments = flags.allWords
        guard arguments.count >= 2, let destination = arguments.last else {
            throw Refusal("usage", "save <i7> [i8 …] <file.png|file.jpg|file.heic|folder> [--format png|jpeg|heic]")
        }
        let entries = try arguments.dropLast().map { a -> Entry in
            guard let e = image(a) else {
                throw Refusal("unknown_image", String(localized: "no image \"\(a)\" in the history"), help: "silicontrol history")
            }
            return e
        }
        let target = absolute(destination, cwd)
        let ext = (target as NSString).pathExtension.lowercased()
        var files: [[String: Any]] = []
        if entries.count == 1, ExportFormat(extension: ext) != nil {
            // A named file: written, or replaced — it is « Save as ».
            try createFolder((target as NSString).deletingLastPathComponent)
            try writeFile(entries[0], target)
            files.append(["image": entries[0].name, "path": target])
        } else {
            try createFolder(target)
            for e in entries { files.append(["image": e.name, "path": try write(e, in: target, as: kind)]) }
        }
        return ["ok": true, "files": files]
    }

    // ── vary ──

    /// **`vary i7 subtle|strong [--batch N]`**: « Variations » on a history image, as its menu does — N
    /// jobs (the rack's « Seeds » by default) with its settings and seed, each turned by a variation
    /// seed drawn at random.
    func vary(_ arguments: [String]) throws -> [String: Any] {
        let (e, amount, count) = try varyRequest(arguments)
        if let reason = app.libraryBlocker { throw Refusal("library_absent", reason, help: String(localized: "connect the disk, then retry")) }
        if let reason = app.installBlocker { throw Refusal("busy", reason, help: String(localized: "retry once the installation is done (silicontrol models)")) }
        guard app.canVary(e) else {
            throw Refusal("model", String(localized: "\(e.name) has no variations: an enlargement or a crop, or its model is not ready"),
                          help: "silicontrol models")
        }
        guard app.file.count + count <= AppState.maxQueueLength else {
            throw Refusal("queue_full", String(localized: "the queue is full (\(AppState.maxQueueLength) jobs)"),
                          help: "silicontrol wait, then retry")
        }
        return ["ok": true, "jobs": app.vary(e, amount, count: count).map { json($0) }]
    }

    /// What `vary` is asked: the image, the amount, how many — read the same on this Mac and on a mirror.
    func varyRequest(_ arguments: [String]) throws -> (entry: Entry, amount: Variation.Amount, count: Int) {
        let flags = try Self.flags(arguments, valued: ["--batch"])
        var count = app.batch
        if let v = flags.value("--batch") {
            guard let n = Int(v), AddOrder.batches.contains(n) else {
                throw Refusal("usage", String(localized: "--batch expects an integer from 1 to 8, not \"\(v)\""))
            }
            count = n
        }
        let words = flags.allWords
        guard words.count == 2, let amount = Variation.Amount(rawValue: words[1]) else {
            throw Refusal("usage", "vary <i7> subtle|strong [--batch N]")
        }
        guard let e = image(words[0]) else {
            throw Refusal("unknown_image", String(localized: "no image \"\(words[0])\" in the history"), help: "silicontrol history")
        }
        return (e, amount, count)
    }

    /// `enlarge i7 [--download]`: « Enlarge ×4 », answered once the enlarged image is in the history.
    func enlarge(_ arguments: [String], _ c: Connection) throws {
        let (e, download) = try enlargeRequest(arguments)
        Task {
            do {
                let result = try await app.enlarged(e, download: download)
                finish(c, ["ok": true, "image": json(result)])
            } catch EnlargeFailure.refused(let sentence) {
                finish(c, Response.failure("refused", sentence))
            } catch EnlargeFailure.downloadRequired {
                finish(c, Response.failure("download_required",
                                           String(localized: "macOS does not have Apple’s enlargement model yet"),
                                           help: "silicontrol enlarge \(e.name) --download"))
            } catch EnlargeFailure.download {
                finish(c, Response.failure("download_failed", String(localized: "The enlargement model could not be downloaded")))
            } catch {
                finish(c, Response.failure("internal", String(localized: "This image could not be enlarged")))
            }
        }
    }

    /// What `enlarge` is asked: the image, and whether macOS may download its model — read the same on this Mac and on a mirror.
    func enlargeRequest(_ arguments: [String]) throws -> (entry: Entry, download: Bool) {
        let flags = try Self.flags(arguments, switches: ["--download"])
        guard flags.allWords.count == 1, let name = flags.allWords.first else { throw Refusal("usage", "enlarge <i7> [--download]") }
        guard let e = image(name) else {
            throw Refusal("unknown_image", String(localized: "no image \"\(name)\" in the history"), help: "silicontrol history")
        }
        return (e, flags.has("--download"))
    }

    // ── Tools ──

    /// `t3` → 3; `3` too.
    static func ordinal(_ text: String, prefix: Character) -> Int? {
        Int(text.first == prefix ? String(text.dropFirst()) : text)
    }

    func image(_ name: String) -> Entry? {
        guard let n = Self.ordinal(name, prefix: "i"), name.first == "i" else { return nil }
        return app.history.first { $0.ordinal == n }
    }

    func createFolder(_ path: String) throws {
        do { try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true) } catch {
            let detail = error.localizedDescription
            throw Refusal("write", String(localized: "folder cannot be created: \(path) (\(detail))"))
        }
    }

    /// In a folder, under the app's export name (`AppState.exportName`: the model, the LoRAs, the steps
    /// if they are not the model's, the format, the seed). Never over an existing file (`-2`, `-3`…).
    func write(_ e: Entry, in folder: String, as kind: ExportFormat = .png) throws -> String {
        let path = AppState.freeURL(in: URL(fileURLWithPath: folder), base: app.exportName(e), ext: kind.fileExtension).path
        try writeFile(e, path)
        return path
    }

    private func writeFile(_ e: Entry, _ path: String) throws {
        do { try app.write(e, to: URL(fileURLWithPath: path)) } catch {
            throw Refusal("write", "\(path): \(error.localizedDescription)")
        }
    }
}
