import CryptoKit
import Foundation

/// **A file of several GB, from the Neo to the mini**: a model or a LoRA chosen on the borrowing Mac,
/// imported on the lending one by the path an import from its own disk takes. The bytes cross as a
/// **stream** (`Frames.swift`), never whole in memory on either side.
///
///     Neo                                       mini
///     uploadOffer(id, name, size) ───────────▶  judged before the first byte: the name, the room
///                                 ◀───────────  uploadAnswer(id, accepted | refused(code))
///     stream id: pieces of 64 KiB ───────────▶  written as they come, in a temporary folder of the
///       …, then a last piece: the SHA-256         library (`telechargements/televersements`)
///       of every byte sent
///                                 ◀───────────  uploadResult(id, verified | failed(code))
///     uploadCancel(id) ──────────────────────▶  the partial file is deleted
///
/// - **The digest is the last piece**, not a message: a message jumps the queue (`Outbox`) and would
///   overtake the bytes it vouches for. Both sides hash as the bytes pass (one pass, a few hundred KB
///   in memory); the mini checks the count against the offer's size and the digest against the trailer.
/// - **Nothing resumes.** A link that drops ends the upload on both sides and the partial file goes: a
///   piece-by-piece resumption would need a state both ends keep through a cut, for a file that is
///   usually imported once.
/// - **One upload at a time on the mini**, from one Mac: the temporary file, then the import, take the
///   disk and the forge alone (`UploadInbox.begin`).
/// - **No sentence crosses**: refusals and failures are codes, said in the reader's language.

package struct UploadOffer: Codable, Equatable, Sendable {
    package var id: UInt32
    /// The file's name, as the Neo has it: the mini keeps it (an imported LoRA is named after its file).
    package var name: String
    package var size: Int64
    package init(id: UInt32, name: String, size: Int64) { self.id = id; self.name = name; self.size = size }
}

/// Why the mini does not take the file — said before the first byte.
package enum UploadRefusal: Codable, Equatable, Sendable {
    /// Not a name this import takes, or an empty file.
    case notImportable
    /// Another upload, an installation, an import or a render is running there.
    case busy
    case noSpace(needed: Int64, available: Int64)
    case cannotWrite
}

package enum UploadAnswer: Codable, Equatable, Sendable {
    case accepted
    case refused(UploadRefusal)
}

/// How an upload failed, on either side.
package enum UploadFailure: Codable, Equatable, Sendable, Error {
    /// The bytes received are not as many as the offer said.
    case size
    /// The bytes received do not hash to what the sender hashed.
    case fingerprint
    case cannotWrite, cannotRead
    /// Stopped by the user, on either Mac.
    case cancelled
    /// The link dropped, or the other side let go.
    case interrupted
}

package enum UploadResult: Codable, Equatable, Sendable {
    /// Size and fingerprint checked; the import has begun on the mini.
    case verified
    case failed(UploadFailure)
}

// MARK: - Sending

/// **The file, read as the network takes it** (`StreamSource`): a piece of at most 64 KiB per pull, its
/// SHA-256 running; when the file is out, one last piece holds the 32-byte digest. Everything runs on the
/// channel's queue.
package final class UploadSource: StreamSource, @unchecked Sendable {
    package static let digestSize = 32

    private let url: URL
    package let size: Int64
    private let cancelled: @Sendable () -> Bool
    private let progress: @Sendable (Int64) -> Void
    private let finished: @Sendable (Result<Void, UploadFailure>) -> Void
    private let lock = NSLock()
    private var handle: FileHandle?
    private var hasher = SHA256()
    private var sent: Int64 = 0
    private var percent = -1
    private var over = false

    /// - `progress`: the bytes handed to the network, called when the whole percent changes.
    /// - `finished`: once — `success` when the digest went out, else why not. Never when the channel is lost
    ///   first: the owner sees the link go.
    package init(url: URL, size: Int64, cancelled: @escaping @Sendable () -> Bool = { false },
                 progress: @escaping @Sendable (Int64) -> Void = { _ in },
                 finished: @escaping @Sendable (Result<Void, UploadFailure>) -> Void = { _ in }) {
        self.url = url; self.size = size
        self.cancelled = cancelled; self.progress = progress; self.finished = finished
    }

    package func next(_ limit: Int) throws -> (bytes: Data, last: Bool) {
        lock.lock(); defer { lock.unlock() }
        guard !over else { throw UploadFailure.interrupted }
        if cancelled() { throw fail(.cancelled) }
        if sent == size {
            over = true
            try? handle?.close(); handle = nil
            finished(.success(()))
            return (Data(hasher.finalize()), true)
        }
        do {
            if handle == nil { handle = try FileHandle(forReadingFrom: url) }
            let want = Int(min(Int64(limit), size - sent))
            // Fewer bytes than the size read at the start: the file shrank under us.
            guard let data = try handle?.read(upToCount: want), data.count == want else { throw fail(.cannotRead) }
            hasher.update(data: data)
            sent += Int64(data.count)
            let p = Int(sent * 100 / max(size, 1))
            if p != percent { percent = p; progress(sent) }
            return (data, false)
        } catch let f as UploadFailure {
            throw f
        } catch {
            throw fail(.cannotRead)
        }
    }

    package func close() {
        lock.lock(); defer { lock.unlock() }
        try? handle?.close(); handle = nil
        over = true
    }

    private func fail(_ f: UploadFailure) -> UploadFailure {
        over = true
        try? handle?.close(); handle = nil
        finished(.failure(f))
        return f
    }
}

// MARK: - Receiving

/// **One file coming in**, written as it arrives: `<directory>/<random>/<name>`, its SHA-256 running.
/// A failure — too many bytes, a write that fails, a digest that differs, the other side letting go —
/// deletes the partial file at once.
package final class UploadReceiver: @unchecked Sendable {
    package enum Step: Equatable, Sendable {
        /// `advanced`: the whole percent changed since the last report.
        case more(received: Int64, advanced: Bool)
        /// Verified: the file, whole.
        case done(URL)
        case failed(UploadFailure)
    }

    package let offer: UploadOffer
    package let file: URL
    private let folder: URL
    private let lock = NSLock()
    private var handle: FileHandle?
    private var hasher = SHA256()
    private var received: Int64 = 0
    private var percent = -1
    private var over = false

    package init(_ offer: UploadOffer, in directory: URL) throws(UploadFailure) {
        self.offer = offer
        folder = directory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        file = folder.appendingPathComponent(offer.name)
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            guard FileManager.default.createFile(atPath: file.path, contents: nil) else { throw UploadFailure.cannotWrite }
            handle = try FileHandle(forWritingTo: file)
        } catch {
            try? FileManager.default.removeItem(at: folder)
            throw .cannotWrite
        }
    }

    /// A piece: file bytes, or — the last — the digest.
    package func piece(_ bytes: Data, last: Bool) -> Step {
        lock.lock(); defer { lock.unlock() }
        guard !over else { return .failed(.interrupted) }
        if last {
            guard received == offer.size else { return end(.size) }
            guard bytes.count == UploadSource.digestSize, Data(hasher.finalize()) == bytes else { return end(.fingerprint) }
            over = true
            do { try handle?.close() } catch { return end(.cannotWrite) }
            handle = nil
            return .done(file)
        }
        guard received + Int64(bytes.count) <= offer.size else { return end(.size) }
        do { try handle?.write(contentsOf: bytes) } catch { return end(.cannotWrite) }
        hasher.update(data: bytes)
        received += Int64(bytes.count)
        let p = Int(received * 100 / max(offer.size, 1))
        defer { percent = p }
        return .more(received: received, advanced: p != percent)
    }

    /// Lets go: the partial file is deleted. Nothing after a `done`: the file is the import's.
    package func abort() {
        lock.lock(); defer { lock.unlock() }
        guard !over else { return }
        _ = end(.interrupted)
    }

    /// The folder of a verified file goes with the file, once the import is over (`AppState.perform`).
    package var directory: URL { folder }

    private func end(_ f: UploadFailure) -> Step {
        over = true
        try? handle?.close(); handle = nil
        try? FileManager.default.removeItem(at: folder)
        return .failed(f)
    }
}

/// **The mini's one upload slot**: judged at the offer, fed by the stream, thread-safe — the stream's
/// pieces come on the network's queue, the offer and the cancellation on the main actor.
package final class UploadInbox: @unchecked Sendable {
    private struct Current {
        let identity: Data
        let receiver: UploadReceiver
        let cancelled: @Sendable () -> Bool
    }

    private let lock = NSLock()
    private var current: Current?
    private let directory: URL
    private let extensions: Set<String>
    private let available: @Sendable (URL) -> Int64?

    /// - `directory`: where the files wait (created when needed).
    /// - `extensions`: the file suffixes the import takes, lowercased.
    /// - `available`: the free bytes on the volume of a folder (`UploadStore.available`); `nil`: unknown, refused.
    package init(directory: URL, extensions: Set<String>,
                 available: @escaping @Sendable (URL) -> Int64? = UploadStore.available) {
        self.directory = directory; self.extensions = extensions; self.available = available
    }

    package var isActive: Bool { lock.withLock { current != nil } }

    /// Judges an offer and, if it takes it, opens the file. `cancelled`: asked at each piece (the mini's "Stop").
    package func begin(identity: Data, _ offer: UploadOffer, cancelled: @escaping @Sendable () -> Bool = { false }) -> UploadRefusal? {
        lock.lock(); defer { lock.unlock() }
        guard current == nil else { return .busy }
        guard offer.size > 0, let name = UploadStore.cleanName(offer.name),
              extensions.contains((name as NSString).pathExtension.lowercased()) else { return .notImportable }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let needed = UploadStore.needed(for: offer.size)
        guard let free = available(directory) else { return .cannotWrite }
        guard free >= needed else { return .noSpace(needed: needed, available: free) }
        var named = offer
        named.name = name
        guard let receiver = try? UploadReceiver(named, in: directory) else { return .cannotWrite }
        current = Current(identity: identity, receiver: receiver, cancelled: cancelled)
        return nil
    }

    /// A piece of the stream `id`; `nil` when no upload of this Mac is under way under it (a piece after a
    /// cancellation, say). The slot is free again after a `done` or a `failed`.
    package func piece(identity: Data, id: UInt32, _ bytes: Data, last: Bool) -> (offer: UploadOffer, step: UploadReceiver.Step)? {
        lock.lock(); defer { lock.unlock() }
        guard let c = current, c.identity == identity, c.receiver.offer.id == id else { return nil }
        let step: UploadReceiver.Step
        if c.cancelled() { c.receiver.abort(); step = .failed(.cancelled) }
        else { step = c.receiver.piece(bytes, last: last) }
        switch step {
        case .more: break
        case .done, .failed: current = nil
        }
        return (c.receiver.offer, step)
    }

    /// The Mac cancelled, or left: its partial file goes. `id` `nil`: whatever it was sending.
    @discardableResult
    package func abort(identity: Data, id: UInt32? = nil) -> UploadOffer? {
        lock.lock(); defer { lock.unlock() }
        guard let c = current, c.identity == identity, id == nil || c.receiver.offer.id == id else { return nil }
        c.receiver.abort()
        current = nil
        return c.receiver.offer
    }
}

// MARK: - The folder

package enum UploadStore {
    /// `telechargements/televersements` in a library: it is the library's downloads, counted and emptied
    /// as they are (`Library.emptyDownloads`).
    package static func directory(in library: Library) -> URL {
        URL(fileURLWithPath: Installer(library: library).downloads).appendingPathComponent("televersements", isDirectory: true)
    }

    /// A file name safe to write under our folder: its last component, not hidden, not empty.
    package static func cleanName(_ name: String) -> String? {
        let last = (name as NSString).lastPathComponent
        guard !last.isEmpty, !last.hasPrefix("."), !last.contains("/"), !last.contains("\0"), last.utf8.count <= 255 else { return nil }
        return last
    }

    /// **What the volume must hold** for a file of `size`: the temporary copy, and — an import being a
    /// forge, whose peak is not announced and whose output is a clone only on APFS — as much again, plus
    /// a margin for the system. Refused before the first byte, never discovered halfway.
    package static func needed(for size: Int64) -> Int64 { 2 * size + (1 << 30) }

    /// The bytes the volume holding `folder` gives to an app (purgeable space counted).
    @Sendable package static func available(_ folder: URL) -> Int64? {
        (try? folder.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage
    }

    /// Deletes what an upload left in `directory`: at the lending app's start, when nothing waits for it.
    package static func purge(_ directory: URL) {
        try? FileManager.default.removeItem(at: directory)
    }
}
