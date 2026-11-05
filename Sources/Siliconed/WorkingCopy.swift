import Darwin
import Foundation

/// **The working copy: what renders read, copied to this Mac while the app runs.**
///
///     ~/Library/Caches/Siliconed/working-copy-<pid>/store/…   the library's store/ tree, in part
///     ~/Library/Caches/Siliconed/working-copy-<pid>/.lock     held by the process that owns it
///
/// A library on another disk or a share (`Library.isElsewhere`) is never rendered from in place: on
/// 16 GB the engine reads the DiT's map again at every step (`Prefetcher`, up to 4 GB/s from the
/// internal SSD) — a USB disk or a share would set the pace of every step, and an unplugged disk
/// would pull the pages from under a `mmap` (a crash, not an error). So the files a model reads are
/// copied here first, once, and the render reads them here.
///
/// - **The copy is a library** (`library`): the same `store/` tree, so `Model.named` builds the same
///   model from it and the engine knows nothing of where the files came from. Its `local` is the
///   library's: profile, render cache and licenses stay this Mac's.
/// - **A file is up to date when its size and date are the original's** (`FileCopy.isSame`); it is
///   copied under `.partiel`, then renamed.
/// - **Several models stay** — a series across models copies all of its own before its first
///   cell —, the least recently used thrown out first when room runs short, `margin` always left free
///   on the volume; beyond, `diskFull`, before the first byte. A file a preparation needs is never
///   thrown out by it.
/// - **One per process, gone with it**: `remove()` when the app quits; a folder whose process died
///   (its `.lock` free) goes at the next launch (`removeAbandoned`).
/// - **On this Mac's own disk, no copy** (`isNeeded`): the render reads the library.
package final class WorkingCopy: @unchecked Sendable {
    package let source: Library
    package let folder: URL
    /// The copy, as a library to build models from.
    package var library: Library { Library(root: folder, local: source.local) }

    /// The room always left free on the copy's volume: 5 GB — macOS, the other apps, a render's cache.
    package static let margin = 5_000_000_000

    /// `~/Library/Caches/Siliconed`: what can be redone, where macOS expects it.
    package static var base: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("Siliconed")
    }

    /// One preparation at a time: a render waits for the copy the model's choice started.
    private let preparing = NSLock()
    private let state = NSLock()
    /// When each file was last needed (a counter): what goes first when room runs short.
    private var lastUse: [String: Int] = [:]
    private var tick = 0
    private let held: Int32

    /// **A library elsewhere, on another volume than the copy's**, is read through a copy — and one
    /// whose disk is absent: it will be another disk when it comes back.
    package static func isNeeded(for source: Library, base: URL = WorkingCopy.base) -> Bool {
        source.isElsewhere && (!source.isReachable || !Library.sameVolume(source.root, base))
    }

    /// Creates this process's copy (empty) under `base`, after removing the abandoned ones.
    package init(source: Library, base: URL = WorkingCopy.base) throws(EngineError) {
        Self.removeAbandoned(in: base)
        let folder = base.appendingPathComponent("working-copy-\(getpid())")
        try EngineError.boundary {
            if FileManager.default.fileExists(atPath: folder.path) { try FileManager.default.removeItem(at: folder) }
            try FileManager.default.createDirectory(at: folder.appendingPathComponent("store"), withIntermediateDirectories: true)
        }
        self.source = source
        self.folder = folder
        let lock = open(folder.appendingPathComponent(".lock").path, O_RDWR | O_CREAT, 0o644)
        // A lock we cannot take is no lock: the copy would look abandoned to the next process. Held or not at all.
        if lock >= 0, flock(lock, LOCK_EX | LOCK_NB) != 0 { close(lock); held = -1 } else { held = lock }
    }

    deinit { if held >= 0 { close(held) } }

    /// **Removes the copies no process holds** — those of a process that crashed or was killed.
    package static func removeAbandoned(in base: URL = WorkingCopy.base) {
        let fm = FileManager.default
        for name in (try? fm.contentsOfDirectory(atPath: base.path)) ?? [] where name.hasPrefix("working-copy-") {
            let folder = base.appendingPathComponent(name)
            let fd = open(folder.appendingPathComponent(".lock").path, O_RDONLY)
            if fd >= 0 {
                defer { close(fd) }
                guard flock(fd, LOCK_EX | LOCK_NB) == 0 else { continue }      // alive
            } else if let made = (try? fm.attributesOfItem(atPath: folder.path))?[.creationDate] as? Date,
                      Date().timeIntervalSince(made) < 60 {
                // No lock yet: a process that is creating its copy right now (`init` makes the folder, then the lock).
                continue
            }
            try? fm.removeItem(at: folder)
        }
    }

    /// Removes the whole copy — when the app quits.
    package func remove() { try? FileManager.default.removeItem(at: folder) }

    /// What the copy occupies now, in bytes.
    package var size: Int { Library.size(folder.appendingPathComponent("store").path) }

    /// **A file of the library, in the copy** — a LoRA's path, as the stack holds it. A path outside
    /// the library is read where it is.
    package func path(of original: String) -> String {
        guard let relative = relative(original) else { return original }
        return folder.appendingPathComponent(relative).path
    }

    /// **The files the models and LoRAs read**, relative to the library's root: a family's components,
    /// its encoder map and its DiT's (or the imported model's), what goes beside a map (Anima's
    /// adapter), each LoRA. Only what is there: a model not installed stays not installed in the copy.
    package func files(models: [String], loras: [String]) throws(EngineError) -> [String] {
        let cards = source.cards()
        var found = Set<String>()
        func add(_ path: String) {
            if FileManager.default.fileExists(atPath: path), let r = relative(path) { found.insert(r) }
        }
        for id in models {
            guard let card = cards.first(where: { $0.id == id }) else { throw .unknownModel(model: id) }
            let f = card.family
            let components = Installer(library: source).componentsFolder(f)
            // Through links: a linked tokenizer is copied as the file it reaches, never left out.
            for (path, _) in Library.regularFiles(under: components) {
                let name = (path as NSString).lastPathComponent
                guard name != ".DS_Store", !name.hasSuffix(".partiel") else { continue }
                add(path)
            }
            add(source.map(source.encoderMap(f)))
            let dit = card.map ?? source.map(source.ditMap(f))
            add(dit)
            add(adapterPath(fromMap: dit))
        }
        loras.forEach(add)
        return found.sorted()
    }

    /// Where a preparation stands: bytes copied of `total`, the file at hand.
    package typealias Progress = LibraryMove.Progress

    /// **Brings the copy up to date for `models` and `loras`** — nothing to do when it is; otherwise
    /// the room is made first (or `diskFull`), then each missing or changed file is copied. Then the
    /// copy must name the same versions as the library: a model is never rendered from other files.
    package func prepare(models: [String], loras: [String] = [], cancellation: Cancellation? = nil,
                         progress: (Progress) -> Void = { _ in }) throws(EngineError) {
        preparing.lock()
        defer { preparing.unlock() }
        try source.checkReachable()
        let needed = try files(models: models, loras: loras)
        let pending = needed.filter { !FileCopy.isSame(source.root.appendingPathComponent($0).path, folder.appendingPathComponent($0).path) }
        state.withLock {
            tick += 1
            for r in needed { lastUse[r] = tick }
        }
        // The other versions' maps of these families, left by an earlier choice: a stale one could
        // make the copy resolve another version than the library's (`Library.variant(of:)`).
        let families = Set(source.cards().filter { models.contains($0.id) }.map(\.family))
        for f in families {
            for v in f.variants {
                for name in [f.ditMap(v), f.encoderMap(v)] {
                    let r = "store/" + name
                    if !needed.contains(r) { forget(r) }
                }
            }
        }
        let bytes = pending.reduce(0) { $0 + Library.size(source.root.appendingPathComponent($1).path) }
        if bytes > 0 { try makeRoom(bytes, keeping: Set(needed)) }

        var done = 0
        do {
            for relative in pending {
                let from = source.root.appendingPathComponent(relative).path
                let to = folder.appendingPathComponent(relative).path
                let name = (relative as NSString).lastPathComponent
                try FileManager.default.createDirectory(atPath: (to as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
                try FileCopy.copy(from, to + ".partiel", hashed: false, cancellation: cancellation) { n in
                    done += n; progress(Progress(done: done, total: bytes, file: name))
                }
                if FileManager.default.fileExists(atPath: to) { try FileManager.default.removeItem(atPath: to) }
                try FileManager.default.moveItem(atPath: to + ".partiel", toPath: to)
            }
        } catch {
            let e = EngineError(error)
            throw source.isReachable ? e : .libraryAbsent(path: source.root.path, volume: source.volumeName)
        }
        for f in families where source.installedVariant(of: f) != nil {
            let copy = library
            guard copy.ditMap(f) == source.ditMap(f), copy.encoderMap(f) == source.encoderMap(f) else {
                throw .internalFailure(component: "working copy",
                                       detail: "\(f.rawValue): the copy resolves \(copy.variant(of: f).rawValue), the library \(source.variant(of: f).rawValue)")
            }
        }
    }

    /// **Frees room for `bytes`**, `margin` kept: the files least recently needed go first, never one of
    /// `keeping`. `diskFull` when even then they would not fit.
    private func makeRoom(_ bytes: Int, keeping: Set<String>) throws(EngineError) {
        let copy = library
        func free() -> Int? { copy.freeSpace() }
        guard var available = free(), available - bytes < Self.margin else { return }
        let held = storedFiles().filter { !keeping.contains($0) }
        let order = state.withLock { held.sorted { (lastUse[$0] ?? 0) < (lastUse[$1] ?? 0) } }
        for r in order where available - bytes < Self.margin {
            forget(r)
            available = free() ?? available
        }
        guard available - bytes >= Self.margin else {
            throw .diskFull(needed: bytes + Self.margin, available: available)
        }
    }

    /// The files in the copy now, relative to its root.
    private func storedFiles() -> [String] {
        let store = folder.appendingPathComponent("store")
        guard let e = FileManager.default.enumerator(at: store, includingPropertiesForKeys: [.isRegularFileKey]) else { return [] }
        let prefix = folder.standardizedFileURL.path + "/"
        var found: [String] = []
        while let u = e.nextObject() as? URL {
            guard (try? u.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true else { continue }
            let p = u.standardizedFileURL.path
            if p.hasPrefix(prefix) { found.append(String(p.dropFirst(prefix.count))) }
        }
        return found
    }

    private func forget(_ relative: String) {
        try? FileManager.default.removeItem(at: folder.appendingPathComponent(relative))
        _ = state.withLock { lastUse.removeValue(forKey: relative) }
    }

    /// A path of the library, relative to its root; `nil` outside it.
    private func relative(_ path: String) -> String? {
        let p = URL(fileURLWithPath: path).standardizedFileURL.path
        let prefix = source.root.path.hasSuffix("/") ? source.root.path : source.root.path + "/"
        return p.hasPrefix(prefix) ? String(p.dropFirst(prefix.count)) : nil
    }
}
