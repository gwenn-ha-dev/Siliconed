import Foundation

/// **A file the user already has, taken instead of downloaded**: an encoder or a VAE dropped on the app
/// whose size and sha256 are those of a file an installation fetches (the pinned ones, `Installer.encoder`
/// and `Installer.components`) is copied to the place its download would have had, and the installation,
/// finding it complete, does not download it. Exact, so nothing to trust: a file that is not byte for byte the expected
/// one is not touched here (a different encoder is another matter, and is refused as before).
///
/// What is copied is the **source**, not a map: the installation forges it as it would a download,
/// then throws it away. The app never shares a file with another tool; the copy saves the download,
/// not the disk space (on APFS the copy is a clone, which costs none until one of the two changes).
///
/// Never a DiT: a DiT dropped is a model to import (`ModelImport`). A VAE waits in `telechargements/`
/// like an encoder, never in `store/composants/`, whose folder alone makes a family look installed.
enum Adoption {
    /// One file an installation expects, for one version of one family.
    struct Expectation {
        let family: Family
        let variant: Variant
        let file: Installer.File
    }

    /// The pinned encoder files of every version of every family. **Several versions often expect the
    /// same file** (a Compact's and a Light's encoder is one), and a file may be expected by several
    /// families: all are returned, `adopt` serves each.
    static func published() -> [Expectation] {
        Family.allCases.flatMap { f in
            f.variants.flatMap { v in
                Installer.encoder(f, v).filter { $0.sha256 != nil && $0.size != nil }.map { Expectation(family: f, variant: v, file: $0) }
            }
            // A family's components are every version's: listed once, under the Standard.
            // (a VAE alone: a tokenizer or a config pinned beside it is not a file a user drops)
            + Installer.components(f).filter { $0.destination == "vae.safetensors" && $0.sha256 != nil && $0.size != nil }
                .map { Expectation(family: f, variant: .standard, file: $0) }
        }
    }

    /// What `adopt` found and did.
    struct Result {
        /// Every (family, version) that expects this very file.
        let expected: [Expectation]
        /// The places that now hold it for an installation to find; empty when every one of the
        /// versions expecting it has its encoder already, so that nothing is needed.
        let placed: [String]
        /// Of those, the ones copied by this call (the others were there, whole).
        let copied: Int
    }

    /// **The expectations a file satisfies**: the size is tested first, the hash — seconds on a
    /// large file — only if some expectation agrees on it. Empty: not a file any installation expects.
    static func matches(_ path: String, among table: [Expectation],
                        hash: (String) throws -> String = Downloader.sha256) throws -> [Expectation] {
        let size = (try FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? -1
        let sized = table.filter { $0.file.size == size }
        guard !sized.isEmpty else { return [] }
        let sha = try hash(path)
        return sized.filter { $0.file.sha256 == sha }
    }

    /// **Copies `path` where the installations that expect it will find it**, for those still lacking
    /// their encoder; `nil` when the file is none that is expected. Copied into a `.partiel` and renamed,
    /// like a download, so that a cut leaves nothing a later installation could take for whole.
    static func adopt(_ path: String, in library: Library, among table: [Expectation] = published(),
                      hash: (String) throws -> String = Downloader.sha256) throws -> Result? {
        let found = try matches(path, among: table, hash: hash)
        guard let size = found.first?.file.size else { return nil }
        let installer = Installer(library: library)
        var places: [String] = []
        let installed = { (x: Expectation) in
            x.file.destination.map { Library.exists(installer.componentsFolder(x.family) + "/" + $0) }
                ?? Library.exists(library.map(x.family.encoderMap(x.variant)))
        }
        for x in found where !installed(x) {
            // One copy per distinct place: two versions expecting one file share it.
            let place = installer.source(x.file)
            if !places.contains(place) { places.append(place) }
        }
        let fm = FileManager.default
        var copied = 0
        for place in places where Library.size(place) != size {
            try library.checkSpace(size)
            try fm.createDirectory(atPath: (place as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
            let partial = place + ".partiel"
            try? fm.removeItem(atPath: partial)
            try fm.copyItem(atPath: path, toPath: partial)
            // Held until the file has its name: `Library.emptyDownloads` leaves a locked `.partiel` alone.
            let held = Library.holdPartial(partial)
            defer { if held >= 0 { close(held) } }
            guard Library.size(partial) == size else {
                try? fm.removeItem(atPath: partial)
                throw EngineError.downloadInterrupted(file: (path as NSString).lastPathComponent)
            }
            try? fm.removeItem(atPath: place)
            try fm.moveItem(atPath: partial, toPath: place)
            copied += 1
        }
        return Result(expected: found, placed: places, copied: copied)
    }
}
