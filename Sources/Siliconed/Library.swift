import Foundation

/// **The models' root: one folder, and every path derives from it.**
///
///     <root>/store/                       forged maps (DiT, encoders), forged LoRAs
///     <root>/store/composants/<family>/   a family's tokenizers, VAE, configs — the small
///                                         published files the engine reads as is
///     <root>/store/importes/              the imported DiTs (Civitai checkpoints), one per model
///     <root>/telechargements/             for the duration of an installation only
///
/// **Everything can be redone**: `store/` rebuilds from nothing (the installation), and
/// nothing else is read at render time — neither a Hugging Face cache nor a file brought by the
/// user, which can be thrown away once imported.
///
/// **No path is hard-coded, and nothing is read relative to the current directory**: in an app,
/// `~` is the container and the current directory is `/`. An app therefore holds its `Library`
/// (`~/Library/Application Support/Siliconed/`, or a folder chosen by the user) and passes it to
/// `Model.named(_:in:)`; `Library.standard` is its own.
///
/// **What belongs to the Mac lives in `local`**, not with the models: the
/// render cache, the accepted licenses, the remote control's socket, where the library is
/// (`LibraryPlace.swift`). `local` is `root` itself, unless the library was moved to another
/// disk: the app's own space keeps them then, and a move never carries them.
package struct Library: Sendable {
    package let root: URL
    /// This Mac's own files (`machineFiles`) — `root` unless the library lives elsewhere.
    package let local: URL

    package init(root: URL, local: URL? = nil) {
        self.root = Self.folder(root)
        self.local = Self.folder(local ?? root)
    }

    /// A folder's URL in one spelling — standardized, with its trailing slash —, so that two URLs of
    /// one folder compare equal whether the Finder, a panel or a path gave it.
    package static func folder(_ url: URL) -> URL {
        URL(fileURLWithPath: url.standardizedFileURL.path, isDirectory: true)
    }

    /// `<root>/store/<name>` — a forged map, a forged LoRA.
    package func map(_ name: String) -> String {
        root.appendingPathComponent("store").appendingPathComponent(name).path
    }

    /// **The app's socket**: the Unix socket where the app open on this library listens for
    /// `silicontrol …` (`docs/REMOTE-CONTROL.md`). It exists only while the app runs. In `local`:
    /// a socket cannot be made on every disk (exFAT, a network share), and `silicontrol` finds it
    /// without knowing where the models are.
    package var socket: String { local.appendingPathComponent("silicontrol.sock").path }

    /// `~/Library/Application Support/Siliconed`: the root as the app shows it — never the account's
    /// name, which a screenshot or a published report would carry (`withoutHome`).
    package var displayPath: String { Self.withoutHome(root.path) }

    /// **A text with the home folder written `~`**, wherever it occurs — a path, or an error message
    /// that quotes one. What the app shows and what a diagnostic publishes go through it.
    package static func withoutHome(_ text: String) -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        guard home.count > 1 else { return text }
        return text.replacingOccurrences(of: home, with: "~")
    }

    /// `<root>/store/importes/` — the imported DiTs.
    // on-disk layout, kept for existing libraries (store/importes, store/composants)
    package var imported: URL { root.appendingPathComponent("store/importes") }

    /// A file (or folder) of a family's components. Throws if it is not there: a missing
    /// tokenizer is discovered here, not after a minute of encoder.
    package func component(_ family: Family, _ relative: String) throws -> String {
        let path = root.appendingPathComponent("store/composants/\(family.rawValue)/\(relative)").path
        guard FileManager.default.fileExists(atPath: path) else { throw MissingFile(path, .published) }
        return path
    }

    /// **The process's library, for the command line** — read once.
    ///
    /// 1. `SILICONED_ROOT`, if set: it has the last word.
    /// 2. The current directory, if it has a `store/`.
    /// 3. The first folder above the executable that has a `store/` (a binary in `.build/release/`
    ///    launched from elsewhere).
    /// 4. Otherwise the app's library (`Library.standard`, or where it was moved: `placed(in:)`) —
    ///    the nominal case: the repository keeps no `store/`, so the close-out and the checks render
    ///    with the models the app installed, without keeping a second copy of them.
    ///
    /// `SILICONED_ROOT` names the app's space as the app reads it: moved, its library is followed.
    ///
    /// An app does not depend on it: it builds its own `Library(root:)`.
    package static let processWide: Library = {
        let fm = FileManager.default
        func hasStore(_ folder: URL) -> Bool {
            var isFolder: ObjCBool = false
            return fm.fileExists(atPath: folder.appendingPathComponent("store").path,
                                 isDirectory: &isFolder) && isFolder.boolValue
        }
        if let providedValue = ProcessInfo.processInfo.environment["SILICONED_ROOT"], !providedValue.isEmpty {
            return Library.placed(in: URL(fileURLWithPath: providedValue))
        }
        let current = URL(fileURLWithPath: fm.currentDirectoryPath)
        if hasStore(current) { return Library(root: current) }
        if var folder = Bundle.main.executableURL?.resolvingSymlinksInPath().deletingLastPathComponent() {
            while folder.path != "/" {
                if hasStore(folder) { return Library(root: folder) }
                folder = folder.deletingLastPathComponent()
            }
        }
        return .placed(in: Library.standard.root)
    }()

    /// **A file's size, through a symbolic link**: `stat`, never `attributesOfItem`, which describes the
    /// link itself (a few dozen bytes) — a map linked into the store, or an encoder linked from another
    /// app's folder, would weigh nothing and the memory need would come out wrong. `nil` if nothing is there.
    package static func fileSize(_ path: String) -> Int? {
        var info = stat()
        return stat(path, &info) == 0 ? Int(info.st_size) : nil
    }

    /// **The regular files under `folder`, every link followed**: a link to a file is that file, a link
    /// to a folder (the store itself, a family's components) is walked as a folder, a broken link is
    /// nothing — like a file absent. A `FileManager` enumerator does none of this: it lists a link as a
    /// link and never enters a linked folder, so a library made of links weighed nothing and a move
    /// carried none of it. Paths are spelt as `folder`'s tree spells them (a link keeps its name),
    /// each with the `stat` of what it reaches — one file reached twice (a link beside its target, a
    /// hard link) has one `st_dev`/`st_ino`. A folder reached twice (a link to a parent) is walked once.
    package static func regularFiles(under folder: String) -> [(path: String, info: stat)] {
        var found: [(path: String, info: stat)] = []
        var walked = Set<[Int]>()
        func walk(_ directory: String) {
            var d = stat()
            guard stat(directory, &d) == 0, d.st_mode & S_IFMT == S_IFDIR,
                  walked.insert([Int(d.st_dev), Int(d.st_ino)]).inserted else { return }
            for name in ((try? FileManager.default.contentsOfDirectory(atPath: directory)) ?? []).sorted() {
                let path = (directory as NSString).appendingPathComponent(name)
                var info = stat()
                guard stat(path, &info) == 0 else { continue }
                switch info.st_mode & S_IFMT {
                case S_IFDIR: walk(path)
                case S_IFREG: found.append((path, info))
                default: continue
                }
            }
        }
        walk(folder)
        return found
    }
}
