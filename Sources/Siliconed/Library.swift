import Foundation

/// **The models' root: one folder, and every path derives from it.**
///
///     <racine>/store/                        forged maps (DiT, encoders), forged LoRAs, profil.json
///     <racine>/store/composants/<famille>/   a family's tokenizers, VAE, configs — the small
///                                            published files the engine reads as is
///     <racine>/store/importes/               the imported DiTs (Civitai checkpoints), one per model
///     <racine>/telechargements/              for the duration of an installation only
///
/// **Everything can be redone**: `store/` rebuilds from nothing (the installation), and
/// nothing else is read at render time — neither a Hugging Face cache nor a file brought by the
/// user, which can be thrown away once imported.
///
/// **No path is hard-coded, and nothing is read relative to the current directory**: in an app,
/// `~` is the container and the current directory is `/`. An app therefore holds its `Library`
/// (`~/Library/Application Support/Siliconed/`, or a folder chosen by the user) and passes it to
/// `Model.named(_:in:)`; `Library.standard` is its own.
public struct Library: Sendable {
    public let root: URL

    public init(root: URL) { self.root = root.standardizedFileURL }

    /// `<racine>/store/<nom>` — a forged map, a forged LoRA.
    public func map(_ name: String) -> String {
        root.appendingPathComponent("store").appendingPathComponent(name).path
    }

    /// The machine's profile, written by the developer's bench.
    // on-disk key, kept for existing maps/profiles (file name kept)
    public var profile: String { map("profil.json") }

    /// **The app's socket**: the Unix socket where the app open on this library listens for
    /// `silicontrol …` (`docs/REMOTE-CONTROL.md`). It exists only while the app runs.
    public var socket: String { root.appendingPathComponent("silicontrol.sock").path }

    /// `~/Library/Application Support/Siliconed`: the root as the app shows it — never the account's
    /// name, which a screenshot or a published report would carry (`withoutHome`).
    public var displayPath: String { Self.withoutHome(root.path) }

    /// **A text with the home folder written `~`**, wherever it occurs — a path, or an error message
    /// that quotes one. What the app shows and what a diagnostic publishes go through it.
    public static func withoutHome(_ text: String) -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        guard home.count > 1 else { return text }
        return text.replacingOccurrences(of: home, with: "~")
    }

    /// `<racine>/store/importes/` — the imported DiTs.
    // on-disk layout, kept for existing libraries (store/importes, store/composants, profil.json)
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
    /// 2. The current directory, if it has a `store/` — the nominal case: the close-out and the
    ///    checks run from the repository root, which is also the models' root (everything lives in
    ///    the repository).
    /// 3. The repository deduced from the binary: walk up from the executable in `.build/release/` to the
    ///    first folder that has a `store/`. That is what makes the developer's command line work when launched
    ///    from elsewhere.
    /// 4. Otherwise the app's library (`Library.standard`): a repository without `store/`
    ///    renders with the models the app installed, without keeping a second copy of them.
    ///
    /// An app does not depend on it: it builds its own `Library(root:)`, and calls
    /// `EngineSettings.load(from:)` at startup so the machine's profile is its own.
    package static let processWide: Library = {
        let fm = FileManager.default
        func hasStore(_ folder: URL) -> Bool {
            var isFolder: ObjCBool = false
            return fm.fileExists(atPath: folder.appendingPathComponent("store").path,
                                 isDirectory: &isFolder) && isFolder.boolValue
        }
        if let providedValue = ProcessInfo.processInfo.environment["SILICONED_ROOT"], !providedValue.isEmpty {
            return Library(root: URL(fileURLWithPath: providedValue))
        }
        let current = URL(fileURLWithPath: fm.currentDirectoryPath)
        if hasStore(current) { return Library(root: current) }
        if var folder = Bundle.main.executableURL?.resolvingSymlinksInPath().deletingLastPathComponent() {
            while folder.path != "/" {
                if hasStore(folder) { return Library(root: folder) }
                folder = folder.deletingLastPathComponent()
            }
        }
        return .standard
    }()
}
