import CryptoKit
import Foundation
import Synchronization

// **The download**: a published file at its pinned commit, to its place, judged by its pinned size and digest. What is
// downloaded, and why, is the families' (`FamilyTraits.files`); when, the installer's (`Installation.swift`).

/// **A Hugging Face file, pinned, verified.** Every repository the app reads is open, and a request carries no token:
/// the app has no sign-in and reads no other tool's (20/11: a token `hf auth login` had left, sent with every request,
/// was refused with a 401 on two open repositories that serve the same files to anyone without one).
package enum Downloader {
    /// The bytes `[begin, end]` of a remote file (a header examined before anything is downloaded, `HubImport`). Only
    /// a `206` with exactly those bytes is taken (`rangeBody`): a server that ignores `Range` answers `200` with the
    /// whole file — gigabytes that are never read.
    static func range(_ url: URL, _ begin: Int, _ end: Int) throws -> Data {
        var r = URLRequest(url: url)
        r.setValue("bytes=\(begin)-\(end)", forHTTPHeaderField: "Range")
        let frozen = r, file = url.lastPathComponent
        return try blocking { try await fetchRange(frozen, begin, end, file: file) }
    }

    /// One range's request: its answer's head judged before its body is read, the request let go once refused or read.
    private static func fetchRange(_ request: URLRequest, _ begin: Int, _ end: Int, file: String) async throws -> Data {
        let (body, response) = try await URLSession.shared.bytes(for: request)
        defer { body.task.cancel() }
        return try await rangeBody(response, body, begin, end, file: file)
    }

    /// Several ranges of one file, `together` requests at a time, in the order asked: the small texts an examination
    /// reads through a file's data (`HubImport.contentRanges`).
    static func ranges(_ url: URL, _ ranges: [Range<Int>], together: Int = 8) throws -> [Data] {
        let file = url.lastPathComponent
        let requests = ranges.map { r -> URLRequest in
            var q = URLRequest(url: url)
            q.setValue("bytes=\(r.lowerBound)-\(r.upperBound - 1)", forHTTPHeaderField: "Range")
            return q
        }
        return try blocking {
            try await withThrowingTaskGroup(of: (Int, Data).self) { group in
                var out = [Data](repeating: Data(), count: requests.count), next = 0
                func add() {
                    let k = next, q = requests[k], r = ranges[k]
                    next += 1
                    group.addTask { (k, try await fetchRange(q, r.lowerBound, r.upperBound - 1, file: file)) }
                }
                while next < min(together, requests.count) { add() }
                while let (k, data) = try await group.next() {
                    out[k] = data
                    if next < requests.count { add() }
                }
                return out
            }
        }
    }

    /// **`range`'s reading of an answer**, apart so that a test can hand it one: anything but a `206` refused before a
    /// byte of its body is read (a `200` is the whole file), then exactly `end − begin + 1` bytes, the reading stopped
    /// at the first one past them.
    static func rangeBody<Body: AsyncSequence>(_ response: URLResponse, _ body: Body, _ begin: Int, _ end: Int,
                                               file: String) async throws -> Data where Body.Element == UInt8 {
        guard let h = response as? HTTPURLResponse, h.statusCode == 206 else {
            throw EngineError.downloadRefused(file: file, status: (response as? HTTPURLResponse)?.statusCode ?? 0)
        }
        let count = end - begin + 1
        var data = Data(capacity: count)
        for try await byte in body {
            guard data.count < count else { throw EngineError.downloadInterrupted(file: file) }
            data.append(byte)
        }
        guard data.count == count else { throw EngineError.downloadInterrupted(file: file) }
        return data
    }

    /// The file a download is written to until it is whole and verified, beside its final name.
    package static func partial(_ destination: String) -> String { destination + ".partiel" }
    /// The ledger of a chunked download, beside it.
    package static func ledger(_ destination: String) -> String { destination + ".morceaux.partiel" }

    /// **Downloads `file` to `destination`**: nothing bears the final name before the bytes have the pinned size and
    /// digest. A wrong size is an interrupted download, a wrong digest a corrupt one — thrown away either way.
    /// `received`: the file's bytes on disk so far, resumed ones included — every megabyte or so, then whole.
    package static func fetch(_ file: PublishedFile, to destination: String, cancellation: Cancellation? = nil,
                              received: (@Sendable (Int) -> Void)? = nil,
                              journal: @escaping @Sendable (String) -> Void) throws {
        let fm = FileManager.default
        let label = file.label, u = file.repository.url(file.path)
        try fm.createDirectory(atPath: (destination as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        journal("↓ \(label)" + String(format: " (%.2f GB)", Double(file.bytes) / 1e9))
        let begin = Date()
        if file.bytes > chunkedFrom {
            let fetched = try downloadInChunks(u, file: file, to: destination, cancellation: cancellation, received: received,
                                               journal: journal)
            let s = Date().timeIntervalSince(begin)
            journal(String(format: "  ✓ %.0f s, %.0f MB/s, digest verified", s, Double(fetched) / s / 1e6))
            return
        }
        let partial = partial(destination)
        let held = try hold(partial, label: label)
        defer { close(held) }
        let tracking = Tracker(cancellation: cancellation)
        let session = URLSession(configuration: .default, delegate: tracking, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let temporary: URL, response: URLResponse
        do {
            (temporary, response) = try blocking { try await session.download(for: URLRequest(url: u)) }
        } catch let e as URLError where e.code == .cancelled && cancellation?.isCancelled == true {
            throw EngineError.cancelled
        } catch is URLError {
            throw EngineError.downloadInterrupted(file: label)
        }
        defer { try? fm.removeItem(at: temporary) }
        guard let h = response as? HTTPURLResponse, h.statusCode == 200 else {
            throw EngineError.downloadRefused(file: label, status: (response as? HTTPURLResponse)?.statusCode ?? 0)
        }
        // Copied into the held file itself: a `.partiel` replaced by name would be a file nobody holds, which a
        // second download of the same file would take as its own.
        try FileCopy.copy(temporary.path, partial, hashed: false, cancellation: cancellation) { _ in }
        guard Library.fileSize(partial) == file.bytes else {
            try? fm.removeItem(atPath: partial)
            throw EngineError.downloadInterrupted(file: label)
        }
        guard try file.digest(of: partial) == file.digest else {
            try? fm.removeItem(atPath: partial)
            throw EngineError.downloadCorrupt(file: label)
        }
        try? fm.removeItem(atPath: destination)
        try fm.moveItem(atPath: partial, toPath: destination)
        received?(file.bytes)
    }

    // MARK: The large files, in chunks

    /// Above this size a file comes in chunks; below, in one request (tokenizers, an index).
    static let chunkedFrom = 128 << 20
    static let chunkSize = 64 << 20
    static let parallelChunks = 8
    /// Attempts in a row **without a byte** before a chunk gives up; one that received bytes starts the count again.
    static let attemptsWithoutProgress = 6

    /// **Chunk bounds by endpoints**, never `size × index`: chunk i is `[i·size/n, (i+1)·size/n)`, n the fewest chunks
    /// of at most `chunk` bytes — contiguous, covering, never empty.
    package static func chunks(size: Int, chunk: Int = chunkSize) -> [Range<Int>] {
        let n = max(1, (size + chunk - 1) / chunk)
        return (0..<n).map { i in i * size / n ..< (i + 1) * size / n }
    }

    /// The ledger's first line: the file (size, digest) and the chunk size its indices count in.
    static func ledgerHead(size: Int, digest: String, chunk: Int = chunkSize) -> String { "\(size) \(digest) \(chunk)" }

    /// **The ledger as a resumption rewrites it**: its head, then each chunk already whole, every line ended. Written
    /// whole before anything is appended: a last line a crash cut (`"…\n1"`) would otherwise join the next chunk's
    /// (`"5\n"`) into one that was never written (`15`).
    static func ledgerText(size: Int, digest: String, done: Set<Int>, chunk: Int = chunkSize) -> String {
        ([ledgerHead(size: size, digest: digest, chunk: chunk)] + done.sorted().map(String.init)).map { $0 + "\n" }.joined()
    }

    /// **The `.partiel` of a download, opened and locked**: held while the file is written and until it has its name.
    /// A second download of the same file (another installation, the command line) is refused rather than let write
    /// over it, and `Installer.clean` leaves a locked `.partiel` alone.
    static func hold(_ path: String, label: String) throws(EngineError) -> Int32 {
        let fd = open(path, O_RDWR | O_CREAT, 0o644)
        guard fd >= 0 else { throw .fileUnreadable(path: path) }
        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else { close(fd); throw .downloadInProgress(file: label) }
        return fd
    }

    /// The ledger of a chunked download: its first line names the file and the chunk size (`ledgerHead`), each next
    /// line a chunk written whole. A ledger of another file, of another chunk size, unreadable or damaged — a line that
    /// is not an index of this file's chunks — resumes nothing: the download starts over.
    package static func resumedChunks(ledger: String, size: Int, digest: String, chunk: Int = chunkSize) -> Set<Int> {
        let lines = ledger.split(separator: "\n", omittingEmptySubsequences: false)
        // Only whole lines: the last element is "" after a final newline, or a line a crash cut.
        guard lines.count >= 2, lines[0] == ledgerHead(size: size, digest: digest, chunk: chunk) else { return [] }
        let count = chunks(size: size, chunk: chunk).count
        var done = Set<Int>()
        for line in lines.dropFirst().dropLast() {
            guard let i = Int(line), (0..<count).contains(i) else { return [] }
            done.insert(i)
        }
        return done
    }

    /// **A large file in chunks of 64 MB, 8 at a time, each over a connection of its own.** One long request does not
    /// hold: measured on the Hub's CDN, a single connection had fallen to 2–3 MB/s after half an hour, then
    /// stopped for over 60 s (the request's timeout: 3.75 GB of 4.41 lost), while a fresh connection to the same file
    /// read 30 MB/s and eight 51. So: every attempt opens a new session (`ephemeral`, nothing shared — with one session
    /// HTTP/2 would carry the eight on one connection); a chunk that stalls 30 s is asked again from its first missing
    /// byte; the bytes go to their offset in `.partiel` as they arrive (no chunk held in memory); the chunks written
    /// whole are kept in a ledger beside it, so that a download cut short — the network, a stop, a crash — resumes
    /// where it was. The digest judges the whole file at the end; a corrupt one starts over. Returns the bytes this
    /// attempt fetched.
    @discardableResult
    static func downloadInChunks(_ u: URL, file: PublishedFile, to destination: String, cancellation: Cancellation?,
                                 received: (@Sendable (Int) -> Void)? = nil,
                                 journal: @escaping @Sendable (String) -> Void) throws -> Int {
        let fm = FileManager.default
        let size = file.bytes, label = file.label
        guard case .sha256(let sha) = file.digest else { throw EngineError.downloadRefused(file: label, status: 0) }
        let partial = partial(destination), ledgerPath = ledger(destination)
        let all = chunks(size: size)
        // Locked before anything of a previous attempt is read or thrown away: that attempt may still be running.
        let fd = try hold(partial, label: label)
        defer { close(fd) }
        var info = stat()
        var done: Set<Int> = []
        if fstat(fd, &info) == 0, Int(info.st_size) == size, let text = try? String(contentsOfFile: ledgerPath, encoding: .utf8) {
            done = resumedChunks(ledger: text, size: size, digest: sha)
        }
        if done.isEmpty {
            // Starting over: the old bytes go, the file stays the one held.
            guard ftruncate(fd, 0) == 0 else { throw EngineError.fileUnreadable(path: partial) }
        } else {
            journal(String(format: "  resumed: %d of %d chunks already there", done.count, all.count))
        }
        try ledgerText(size: size, digest: sha, done: done).write(toFile: ledgerPath, atomically: true, encoding: .utf8)
        guard ftruncate(fd, off_t(size)) == 0 else { throw EngineError.fileUnreadable(path: partial) }
        // Written past the page cache: tens of GB, read back once by the hash, have no business filling memory.
        _ = fcntl(fd, F_NOCACHE, 1)
        guard let ledger = FileHandle(forWritingAtPath: ledgerPath) else { throw EngineError.fileUnreadable(path: ledgerPath) }
        defer { try? ledger.close() }
        // Locked too: it is a `.partiel` to `Installer.clean`.
        guard flock(ledger.fileDescriptor, LOCK_EX | LOCK_NB) == 0 else { throw EngineError.downloadInProgress(file: label) }
        _ = try? ledger.seekToEnd()

        let run = ChunkRun(fd: fd, ledger: ledger, size: size, label: label, whole: done,
                           already: done.reduce(0) { $0 + all[$1].count }, cancellation: cancellation, progress: received,
                           journal: journal)
        // The sha256 runs behind the download, on the chunks already whole in file order: a pass after the last byte
        // cost 6 s per 10 GB.
        let hash = RunningHash(path: partial, chunks: all, run: run)
        let pending = all.indices.filter { !done.contains($0) }
        let next = Atomic<Int>(0)
        DispatchQueue.concurrentPerform(iterations: min(parallelChunks, pending.count)) { _ in
            while run.failure == nil {
                let k = next.wrappingAdd(1, ordering: .relaxed).oldValue
                guard k < pending.count else { return }
                run.fetch(u, chunk: pending[k], range: all[pending[k]])
            }
        }
        if let e = run.failure { _ = try? hash.result(); throw e }
        if cancellation?.isCancelled == true { _ = try? hash.result(); throw EngineError.cancelled }
        try? ledger.close()
        try? fm.removeItem(atPath: ledgerPath)
        guard try hash.result() == sha else { try? fm.removeItem(atPath: partial); throw EngineError.downloadCorrupt(file: label) }
        try? fm.removeItem(atPath: destination)
        try fm.moveItem(atPath: partial, toPath: destination)
        return pending.reduce(0) { $0 + all[$1].count }
    }

    /// **The sha256 of a chunked download, computed while it downloads**: chunk i is hashed once it and every chunk
    /// before it are whole (resumed chunks are whole from the start), read back from `.partiel` (at the disk's speed,
    /// many times the network's). Stops, without a digest, when the run stops.
    final class RunningHash: @unchecked Sendable {
        private let finished = DispatchSemaphore(value: 0)
        private var digest: String?, error: EngineError?

        init(path: String, chunks: [Range<Int>], run: ChunkRun) {
            DispatchQueue.global(qos: .utility).async { [self] in
                defer { finished.signal() }
                let fd = open(path, O_RDONLY)
                guard fd >= 0 else { error = .fileUnreadable(path: path); return }
                defer { close(fd) }
                _ = fcntl(fd, F_NOCACHE, 1)
                let block = 8 << 20
                let buffer = UnsafeMutableRawPointer.allocate(byteCount: block, alignment: 16384)
                defer { buffer.deallocate() }
                var h = SHA256()
                for (i, r) in chunks.enumerated() {
                    guard run.waitWhole(i) else { return }
                    var at = r.lowerBound
                    while at < r.upperBound {
                        let n = pread(fd, buffer, min(block, r.upperBound - at), off_t(at))
                        guard n > 0 else { error = .fileUnreadable(path: path); return }
                        h.update(bufferPointer: UnsafeRawBufferPointer(start: buffer, count: n))
                        at += n
                    }
                }
                digest = h.finalize().map { String(format: "%02x", $0) }.joined()
            }
        }

        /// The digest, once every chunk is hashed; `nil`: the run stopped first.
        func result() throws -> String? {
            finished.wait()
            if let error { throw error }
            return digest
        }
    }

    /// The state the chunks of one file share: the descriptor they write to, the ledger, the bytes received (for the
    /// journal) and the first failure, which stops the others.
    final class ChunkRun: @unchecked Sendable {
        let fd: Int32, ledger: FileHandle, size: Int, label: String
        let cancellation: Cancellation?, journal: @Sendable (String) -> Void
        /// Told the bytes received, every megabyte: a caller that draws them hears a few dozen times a second at most.
        let progress: (@Sendable (Int) -> Void)?
        private let lock = NSCondition()
        private var received: Int, reported = 0, told: Int, _failure: EngineError?
        /// The chunks written whole, resumed ones included — what `RunningHash` waits for.
        private var whole: Set<Int>

        init(fd: Int32, ledger: FileHandle, size: Int, label: String, whole: Set<Int>, already: Int,
             cancellation: Cancellation?, progress: (@Sendable (Int) -> Void)? = nil,
             journal: @escaping @Sendable (String) -> Void) {
            self.fd = fd; self.ledger = ledger; self.size = size; self.label = label; self.whole = whole
            self.received = already; self.told = already; self.cancellation = cancellation; self.progress = progress
            self.journal = journal
            reported = Int(already * 100 / size) / 20 * 20
            progress?(already)
        }

        var failure: EngineError? { lock.withLock { _failure } }
        func fail(_ e: EngineError) { lock.withLock { if _failure == nil { _failure = e }; lock.broadcast() } }
        var stopped: Bool { cancellation?.isCancelled == true || failure != nil }

        /// Waits until chunk `k` is whole; `false`: the run stopped first (a cancellation is seen within half a second
        /// — the token has no way to wake anyone).
        func waitWhole(_ k: Int) -> Bool {
            lock.lock(); defer { lock.unlock() }
            while !whole.contains(k) {
                if _failure != nil || cancellation?.isCancelled == true { return false }
                _ = lock.wait(until: Date().addingTimeInterval(0.5))
            }
            return true
        }

        func add(_ n: Int) {
            let (line, bytes): (String?, Int?) = lock.withLock {
                received += n
                var bytes: Int?
                if received - told >= 1 << 20 || received >= size { told = received; bytes = received }
                let pc = Int(received * 100 / size)
                guard pc >= reported + 20 else { return (nil, bytes) }
                reported = pc - pc % 20
                return ("  … \(reported) %", bytes)
            }
            if let bytes { progress?(bytes) }
            if let line { journal(line) }
        }

        /// One chunk, asked again from its first missing byte until whole.
        func fetch(_ u: URL, chunk: Int, range: Range<Int>) {
            var at = range.lowerBound, idle = 0
            while at < range.upperBound {
                if cancellation?.isCancelled == true { fail(.cancelled); return }
                if failure != nil { return }
                let attempt = ChunkAttempt(run: self, at: at, end: range.upperBound)
                let configuration = URLSessionConfiguration.ephemeral
                configuration.timeoutIntervalForRequest = 30
                let session = URLSession(configuration: configuration, delegate: attempt, delegateQueue: nil)
                var r = URLRequest(url: u)
                r.setValue("bytes=\(at)-\(range.upperBound - 1)", forHTTPHeaderField: "Range")
                session.dataTask(with: r).resume()
                attempt.finished.wait()
                session.finishTasksAndInvalidate()
                if let refused = attempt.refused { fail(.downloadRefused(file: label, status: refused)); return }
                if attempt.at > at { idle = 0 } else { idle += 1 }
                at = attempt.at
                if at < range.upperBound {
                    guard idle < Downloader.attemptsWithoutProgress else { fail(.downloadInterrupted(file: label)); return }
                    if idle > 0 { Thread.sleep(forTimeInterval: Double(1 << min(idle, 4))) }
                }
            }
            written(chunk)
        }

        /// Chunk `k` is whole: in the ledger (for a resumption), and to the running hash.
        func written(_ k: Int) {
            let line = Data("\(k)\n".utf8)
            lock.withLock {
                try? ledger.write(contentsOf: line)
                whole.insert(k)
                lock.broadcast()
            }
        }
    }

    /// One request for `[at, end)`: the bytes gathered in memory and written at their offset by 8 MB (`F_NOCACHE`: a
    /// write as small as the network hands it, past the cache, scattered a file in tiny extents, read back at
    /// 0.5 GB/s). `at` is what is on disk.
    final class ChunkAttempt: NSObject, URLSessionDataDelegate, @unchecked Sendable {
        let run: ChunkRun, end: Int
        var at: Int, refused: Int?
        let finished = DispatchSemaphore(value: 0)
        static let block = 8 << 20
        private var gathered = Data()
        init(run: ChunkRun, at: Int, end: Int) { self.run = run; self.at = at; self.end = end }

        func urlSession(_ s: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void) {
            // Only a 206 starting at `at`: a 200 would be the whole file written at this offset.
            let h = response as? HTTPURLResponse
            guard h?.statusCode == 206,
                  h?.value(forHTTPHeaderField: "Content-Range")?.hasPrefix("bytes \(at)-") == true else {
                // A server error is passing (asked again); anything else is a refusal.
                if let c = h?.statusCode, !(500..<600).contains(c) { refused = c }
                completionHandler(.cancel); return
            }
            completionHandler(.allow)
        }

        func urlSession(_ s: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
            if run.stopped { dataTask.cancel(); return }
            let n = min(data.count, end - at - gathered.count)
            gathered.append(data.prefix(n))
            run.add(n)
            if gathered.count >= Self.block, !flush() { dataTask.cancel() }
        }

        func urlSession(_ s: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            _ = flush()
            finished.signal()
        }

        /// The gathered bytes to disk, at `at`; `false`: the write failed (the run fails).
        private func flush() -> Bool {
            guard !gathered.isEmpty else { return true }
            let n = gathered.count
            var failure: Int32 = 0
            let written = gathered.withUnsafeBytes { b -> Int in
                var done = 0
                while done < n {
                    let r = pwrite(run.fd, b.baseAddress! + done, n - done, off_t(at + done))
                    if r <= 0 { failure = errno; break }
                    done += r
                }
                return done
            }
            gathered.removeAll(keepingCapacity: true)
            guard written == n else {
                run.fail(failure == ENOSPC ? .diskFull(needed: nil, available: nil) : .downloadInterrupted(file: run.label))
                return false
            }
            at += n
            return true
        }
    }

    /// **A whole file's sha256**, in hexadecimal — a download's, an import's, a library's copy read back from its new
    /// disk. Read by `pread` outside the page cache into two buffers, the next block read while this one is hashed:
    /// measured on the M1 Pro, 1.7 GB/s for read-then-hash, the hash alone ≈ 2.4. Nothing here is autoreleased: a
    /// `FileHandle.read(upToCount:)` loop without a pool kept every block alive (31 GB for one family's files).
    /// `advanced`: the bytes of each block hashed; the token is looked at between blocks.
    package static func sha256(_ path: String, cancellation: Cancellation? = nil, advanced: (Int) -> Void = { _ in }) throws -> String {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw EngineError.fileUnreadable(path: path) }
        defer { close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)
        let block = 8 << 20
        let buffers = (0..<2).map { _ in UnsafeMutableRawPointer.allocate(byteCount: block, alignment: 16384) }
        defer { buffers.forEach { $0.deallocate() } }
        /// A full block from `at`, or less at the end; −1: an error.
        @Sendable func fill(_ b: UnsafeMutableRawPointer, _ at: Int) -> Int {
            var n = 0
            while n < block {
                let r = pread(fd, b + n, block - n, off_t(at + n))
                if r < 0 { return -1 }
                if r == 0 { break }
                n += r
            }
            return n
        }
        final class Next: @unchecked Sendable { var count = 0 }
        var h = SHA256()
        var at = 0, k = 0
        var count = fill(buffers[0], 0)
        while count > 0 {
            try cancellation.check()
            let next = Next(), group = DispatchGroup()
            nonisolated(unsafe) let target = buffers[(k + 1) % 2]
            let from = at + count
            DispatchQueue.global(qos: .utility).async(group: group) { next.count = fill(target, from) }
            h.update(bufferPointer: UnsafeRawBufferPointer(start: buffers[k % 2], count: count))
            group.wait()
            advanced(count)
            (at, count, k) = (from, next.count, k + 1)
        }
        guard count == 0 else { throw EngineError.fileUnreadable(path: path) }
        return h.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Stops a one-request download when its token is raised.
    final class Tracker: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
        let cancellation: Cancellation?
        init(cancellation: Cancellation?) { self.cancellation = cancellation }
        func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
        func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didWriteData _: Int64,
                        totalBytesWritten written: Int64, totalBytesExpectedToWrite total: Int64) {
            if cancellation?.isCancelled == true { downloadTask.cancel() }
        }
    }
}
