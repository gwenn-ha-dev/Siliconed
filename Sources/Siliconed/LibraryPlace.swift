import CryptoKit
import Darwin
import Foundation

// **Where the library lives, and moving it.**
//
//     <local>/library-location     one line: the library's folder, when it is not <local> itself
//     <new place>/.moving          during a move only: the library being moved there
//
// The models weigh tens of GB and the internal SSD is precious: the library may live on another
// disk or a network share. What belongs to this Mac stays in the app's space (`Library.local`,
// `machineFiles`), and the location file there says where the models are — read by the app, the
// CLI (`Library.processWide`) and `silicontrol` alike. A library elsewhere is rendered through a
// copy on this Mac (`WorkingCopy`): its disk's speed costs one wait, never a step.
//
// **A move is verified before anything is switched or deleted**: each file is copied under a
// `.partiel` name, hashed while it is read (SHA-256), read back from the new place and hashed
// again; only an identical file takes its final name, with the original's date. The new place
// becomes the library once every file is there, and only then is the old one emptied. Stopped or
// failed, the library is the old one, untouched; started again, the files already verified are
// kept (same name, size and date) — `.moving` says whose move the new folder is receiving.
//
// **Never a library in place of another**: a library whose disk is absent stays the library
// (`isReachable` false, `EngineError.libraryAbsent`), and going back to this Mac's is a gesture
// (`useLocal`).

extension Library {
    /// **What belongs to this Mac, never carried by a move**, relative to `local`: the render
    /// cache, the remote control's socket, the licenses its user accepted, and where the library is.
    package static let machineFiles = ["cache", "silicontrol.sock", "accepted-licenses.json", locationName]
    static let locationName = "library-location"

    /// **The library of the app's space `local`**: where its location file says, else `local` itself.
    /// A location whose disk is absent is still returned — `isReachable` says so.
    package static func placed(in local: URL) -> Library {
        let file = local.appendingPathComponent(locationName)
        let path = (try? String(contentsOf: file, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !path.isEmpty else { return Library(root: local) }
        return Library(root: URL(fileURLWithPath: path), local: local)
    }

    /// The models live elsewhere than this Mac's own files (another folder, disk or share).
    package var isElsewhere: Bool { root.path != local.path }

    /// **The library's folder is there** — its disk connected, its share mounted. A library elsewhere
    /// always has its `store/` (a move or an opening put it there): a bare folder at its path is an
    /// empty mount point, not the library.
    package var isReachable: Bool {
        !isElsewhere || Self.isFolder(root.appendingPathComponent("store").path)
    }

    /// `libraryAbsent` when the library's disk is not there.
    package func checkReachable() throws(EngineError) {
        guard isReachable else { throw .libraryAbsent(path: root.path, volume: volumeName) }
    }

    /// The name of the library's disk, as the Finder shows it — read from the path when the disk is
    /// absent (`/Volumes/<name>/…`); `nil` on this Mac's own disk.
    package var volumeName: String? {
        if isReachable, isElsewhere, !Self.sameVolume(root, local),
           let name = try? root.resourceValues(forKeys: [.volumeLocalizedNameKey]).volumeLocalizedName {
            return name
        }
        let parts = root.pathComponents
        return parts.count > 2 && parts[1] == "Volumes" ? parts[2] : nil
    }

    /// **Opens the library in `folder`** (or in its `Siliconed` subfolder): it becomes the library of
    /// this Mac's space, nothing moved. Refused when the folder holds none, or cannot be written to.
    package func opening(_ folder: URL) throws(EngineError) -> Library {
        let folder = Library.folder(folder)
        let candidates = [folder, folder.appendingPathComponent("Siliconed")]
        guard let found = candidates.first(where: { Self.isFolder($0.appendingPathComponent("store").path) }) else {
            throw .libraryRefused(path: folder.path, reason: .notALibrary)
        }
        if found.path == root.path { return self }
        if found.path == local.path { return try useLocal() }
        try LibraryMove.checkWritable(found)
        try Self.record(found, in: local)
        return Library(root: found, local: local)
    }

    /// **Back to this Mac's own library**, nothing moved: what is in `local` (maybe nothing yet).
    package func useLocal() throws(EngineError) -> Library {
        try Self.record(nil, in: local)
        return Library(root: local)
    }

    /// Writes where the library of `local`'s space is — `nil` (or `local`): its own, the file removed.
    static func record(_ root: URL?, in local: URL) throws(EngineError) {
        let file = local.appendingPathComponent(locationName)
        try EngineError.boundary {
            guard let root, root.standardizedFileURL.path != local.standardizedFileURL.path else {
                if exists(file.path) { try FileManager.default.removeItem(at: file) }
                return
            }
            try FileManager.default.createDirectory(at: local, withIntermediateDirectories: true)
            try Data((root.standardizedFileURL.path + "\n").utf8).write(to: file, options: .atomic)
        }
    }

    /// The library holds models — anything in `store/` but a `profil.json` that earlier builds left (no longer read).
    static func holdsModels(_ root: URL) -> Bool {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("store").path)) ?? []
        return names.contains { $0 != "profil.json" && $0 != ".DS_Store" }
    }

    static func isFolder(_ path: String) -> Bool {
        var folder: ObjCBool = false
        return FileManager.default.fileExists(atPath: path, isDirectory: &folder) && folder.boolValue
    }

    /// The first existing folder at or above `url` — what a volume question is asked of.
    static func existing(_ url: URL) -> URL {
        var u = url.standardizedFileURL
        while !exists(u.path), u.path != "/" { u = u.deletingLastPathComponent() }
        return u
    }

    /// Two paths on one volume (their first existing folders' file systems).
    package static func sameVolume(_ a: URL, _ b: URL) -> Bool {
        var sa = statfs(), sb = statfs()
        guard statfs(existing(a).path, &sa) == 0, statfs(existing(b).path, &sb) == 0 else { return false }
        return sa.f_fsid.val.0 == sb.f_fsid.val.0 && sa.f_fsid.val.1 == sb.f_fsid.val.1
    }
}

// MARK: - Moving

package enum LibraryMove {
    /// Where a move stands: bytes copied and read back of `total`, the file at hand.
    package struct Progress: Sendable {
        package let done: Int
        package let total: Int
        package let file: String
    }

    static let marker = ".moving"

    /// **The folder the library takes in `chosen`**: `chosen` itself when it is one (named
    /// `Siliconed`, or holding `store/` — a move started there), else its `Siliconed` subfolder.
    package static func destination(in chosen: URL) -> URL {
        let chosen = Library.folder(chosen)
        if chosen.lastPathComponent == "Siliconed" || Library.isFolder(chosen.appendingPathComponent("store").path) { return chosen }
        return chosen.appendingPathComponent("Siliconed")
    }

    /// The files a move carries, relative to the library's root: everything but this Mac's own and
    /// what a crash left half-written (`.partiel`). Through links (`regularFiles(under:)`): a linked
    /// map, or a `store/` that is a link, is carried as the file it reaches — skipped, the old place's
    /// emptying would take the link and the moved library would lack what it pointed to.
    static func files(of source: Library) -> [String] {
        let root = source.root
        var found: [String] = []
        let prefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
        for (path, _) in Library.regularFiles(under: root.path) {
            guard path.hasPrefix(prefix) else { continue }
            let relative = String(path.dropFirst(prefix.count))
            let name = (relative as NSString).lastPathComponent
            if name == ".DS_Store" || name == marker || relative.hasSuffix(".partiel") { continue }
            if Library.machineFiles.contains(where: { relative == $0 || relative.hasPrefix($0 + "/") }) { continue }
            found.append(relative)
        }
        return found.sorted()
    }

    /// **Refuses a destination before the first byte**: the library itself or inside it (or holding
    /// it), a FAT disk, a folder that cannot be written to, another library's models.
    static func check(_ source: Library, _ destination: URL) throws(EngineError) {
        let d = destination.standardizedFileURL.path, r = source.root.path
        if d == r || d.hasPrefix(r + "/") || r.hasPrefix(d + "/") {
            throw .libraryRefused(path: d, reason: .sameLibrary)
        }
        try checkWritable(destination)
        let marked = (try? String(contentsOfFile: destination.appendingPathComponent(marker).path, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if Library.holdsModels(destination), marked != r {
            throw .libraryRefused(path: d, reason: .alreadyALibrary)
        }
    }

    /// A FAT disk (no file over 4 GB) or a read-only folder is refused.
    static func checkWritable(_ folder: URL) throws(EngineError) {
        let existing = Library.existing(folder)
        var s = statfs()
        if statfs(existing.path, &s) == 0 {
            let type = withUnsafeBytes(of: s.f_fstypename) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
            if type == "msdos" { throw .libraryRefused(path: folder.path, reason: .fileSizeLimit) }
            if s.f_flags & UInt32(MNT_RDONLY) != 0 { throw .libraryRefused(path: folder.path, reason: .readOnly) }
        }
        if access(existing.path, W_OK) != 0 { throw .libraryRefused(path: folder.path, reason: .readOnly) }
    }

    /// **What a move would carry**, judged before the first byte: the files still to copy (those already there, identical, are not), their
    /// size, and the room the destination has. Refuses what `move` refuses first — the library unreachable, a destination `check` turns
    /// away, a disk too small — so that a dry run (`silicontrol library move` without `--yes`) and the move cannot disagree.
    package struct Plan: Sendable {
        package let pending: [String]
        package let bytes: Int
        package let room: Int?
    }

    package static func plan(_ source: Library, to destination: URL) throws(EngineError) -> Plan {
        let destination = Library.folder(destination)
        try source.checkReachable()
        try check(source, destination)
        let pending = files(of: source).filter {
            !FileCopy.isSame(source.root.appendingPathComponent($0).path, destination.appendingPathComponent($0).path)
        }
        let bytes = pending.reduce(0) { $0 + Library.size(source.root.appendingPathComponent($1).path) }
        let room = Library(root: destination).freeSpace()
        if let room, bytes > room { throw .diskFull(needed: bytes, available: room) }
        return Plan(pending: pending, bytes: bytes, room: room)
    }

    /// **Moves `source` to `destination`** (a folder from `destination(in:)`, or `source.local` to
    /// bring it back to this Mac), verified file by file; returns the library there, recorded as this
    /// Mac's. The old place is emptied last — this Mac's own files stay.
    package static func move(_ source: Library, to destination: URL, cancellation: Cancellation? = nil,
                             progress: (Progress) -> Void = { _ in }) throws(EngineError) -> Library {
        let destination = Library.folder(destination)
        let fm = FileManager.default
        let plan = try plan(source, to: destination)
        let pending = plan.pending, bytes = plan.bytes

        try EngineError.boundary {
            try fm.createDirectory(at: destination, withIntermediateDirectories: true)
            try Data((source.root.path + "\n").utf8).write(to: destination.appendingPathComponent(marker), options: .atomic)
        }
        var done = 0
        let total = 2 * bytes
        do {
            for relative in pending {
                let from = source.root.appendingPathComponent(relative).path
                let to = destination.appendingPathComponent(relative).path
                let partial = to + ".partiel"
                let name = (relative as NSString).lastPathComponent
                try fm.createDirectory(atPath: (to as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
                let start = done
                let written = try FileCopy.copy(from, partial, hashed: true, cancellation: cancellation) { n in
                    done += n; progress(Progress(done: done, total: total, file: name))
                }
                let read = try FileCopy.digest(partial, cancellation: cancellation) { n in
                    done += n; progress(Progress(done: done, total: total, file: name))
                }
                guard written == read else {
                    try? fm.removeItem(atPath: partial)
                    throw EngineError.libraryRefused(path: to, reason: .copyDamaged)
                }
                if fm.fileExists(atPath: to) { try fm.removeItem(atPath: to) }
                try fm.moveItem(atPath: partial, toPath: to)
                done = start + 2 * Library.size(from)
            }
        } catch {
            let e = EngineError(error)
            throw source.isReachable ? e : .libraryAbsent(path: source.root.path, volume: source.volumeName)
        }

        // Every file is there and verified: the new place becomes the library, then the old one empties.
        try Library.record(destination, in: source.local)
        try? fm.removeItem(at: destination.appendingPathComponent(marker))
        empty(source)
        return destination.path == source.local.path ? Library(root: source.local) : Library(root: destination, local: source.local)
    }

    /// **Empties the old place**: everything but this Mac's own files; the folder itself goes when it
    /// is not this Mac's space and nothing is left in it.
    static func empty(_ source: Library) {
        let fm = FileManager.default
        let root = source.root
        let kept = source.isElsewhere ? [] : Library.machineFiles
        for name in (try? fm.contentsOfDirectory(atPath: root.path)) ?? [] where !kept.contains(name) {
            try? fm.removeItem(at: root.appendingPathComponent(name))
        }
        if source.isElsewhere, ((try? fm.contentsOfDirectory(atPath: root.path)) ?? []).isEmpty {
            try? fm.removeItem(at: root)
        }
    }
}

// MARK: - Copying a file

/// **One file copied, outside the page cache**: tens of GB read once must not push the user's
/// apps out of memory (`F_NOCACHE` on both ends, as `Safetensors` reads a source).
enum FileCopy {
    static let chunk = 8 << 20

    /// The same file: same size, and a date within 2 s — a FAT-like disk or a share rounds dates
    /// (exFAT to 10 ms, SMB to the second); the copy took the original's.
    static func isSame(_ a: String, _ b: String) -> Bool {
        var sa = stat(), sb = stat()
        guard stat(a, &sa) == 0, stat(b, &sb) == 0, sa.st_mode & S_IFMT == S_IFREG, sb.st_mode & S_IFMT == S_IFREG else { return false }
        return sa.st_size == sb.st_size && abs(sa.st_mtimespec.tv_sec - sb.st_mtimespec.tv_sec) <= 2
    }

    /// Copies `from` to `to` (created or truncated), with `from`'s date; returns the SHA-256 of what
    /// was read when `hashed`. `advanced`: the bytes of each chunk written.
    @discardableResult
    static func copy(_ from: String, _ to: String, hashed: Bool, cancellation: Cancellation?,
                     advanced: (Int) -> Void) throws -> SHA256.Digest? {
        let input = open(from, O_RDONLY)
        guard input >= 0 else { throw posix(from) }
        defer { close(input) }
        _ = fcntl(input, F_NOCACHE, 1)
        var st = stat()
        guard fstat(input, &st) == 0 else { throw posix(from) }
        let output = open(to, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
        guard output >= 0 else { throw posix(to) }
        defer { close(output) }
        _ = fcntl(output, F_NOCACHE, 1)
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: chunk, alignment: 16384)
        defer { buffer.deallocate() }
        var hasher = hashed ? SHA256() : nil
        while true {
            try cancellation.check()
            let n = read(input, buffer, chunk)
            if n < 0 { if errno == EINTR { continue }; throw posix(from) }
            if n == 0 { break }
            hasher?.update(bufferPointer: UnsafeRawBufferPointer(start: buffer, count: n))
            var written = 0
            while written < n {
                let w = write(output, buffer + written, n - written)
                if w < 0 { if errno == EINTR { continue }; throw posix(to) }
                written += w
            }
            advanced(n)
        }
        guard fsync(output) == 0 else { throw posix(to) }
        var times = [st.st_atimespec, st.st_mtimespec]
        _ = futimens(output, &times)
        return hasher?.finalize()
    }

    /// The SHA-256 of a file, read outside the page cache — the copy just written comes back from the disk.
    static func digest(_ path: String, cancellation: Cancellation?, advanced: (Int) -> Void) throws -> SHA256.Digest {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw posix(path) }
        defer { close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: chunk, alignment: 16384)
        defer { buffer.deallocate() }
        var hasher = SHA256()
        while true {
            try cancellation.check()
            let n = read(fd, buffer, chunk)
            if n < 0 { if errno == EINTR { continue }; throw posix(path) }
            if n == 0 { break }
            hasher.update(bufferPointer: UnsafeRawBufferPointer(start: buffer, count: n))
            advanced(n)
        }
        return hasher.finalize()
    }

    private static func posix(_ path: String) -> NSError {
        NSError(domain: NSPOSIXErrorDomain, code: Int(errno), userInfo: [NSFilePathErrorKey: path])
    }
}
