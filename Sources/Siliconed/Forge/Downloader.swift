import CryptoKit
import Foundation
import Synchronization

// **The download**: a Hugging Face file at a pinned revision, verified by its sha256 (`Downloader`),
// and `wait`, which runs an asynchronous task from the forge's synchronous code. What is downloaded, and
// why, is in `Installation.swift`.

// MARK: - The download

/// **A Hugging Face file, at a pinned revision, verified.** The token (gated repos, like Krea 2)
/// comes from `HF_TOKEN` or from `~/.cache/huggingface/token` — the one `huggingface-cli login`
/// writes.
package enum Downloader {
    static func url(_ repository: String, revision: String, path: String) -> URL {
        URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)!)")!
    }

    /// The file's page on the Hub (`blob/`), for a human to read — not the bytes (`resolve/`).
    static func page(_ repository: String, revision: String, path: String) -> URL {
        URL(string: "https://huggingface.co/\(repository)/blob/\(revision)/\(path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)!)")!
    }

    static var token: String? {
        let env = ProcessInfo.processInfo.environment
        if let t = env["HF_TOKEN"] ?? env["HUGGING_FACE_HUB_TOKEN"], !t.isEmpty { return t }
        let path = (env["HF_HOME"] ?? NSHomeDirectory() + "/.cache/huggingface") + "/token"
        return (try? String(contentsOfFile: path, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func request(_ url: URL, method: String = "GET") -> URLRequest {
        var r = URLRequest(url: url)
        r.httpMethod = method
        if let t = token { r.setValue("Bearer \(t)", forHTTPHeaderField: "Authorization") }
        return r
    }

    /// The bytes `[begin, end]` of a remote file. Only a `206` with exactly those bytes is taken: a
    /// server that ignores `Range` answers `200` with the whole file — gigabytes read as a header.
    static func range(_ url: URL, _ begin: Int, _ end: Int) throws -> Data {
        var r = request(url)
        r.setValue("bytes=\(begin)-\(end)", forHTTPHeaderField: "Range")
        let frozen = r
        let (data, response) = try wait { try await URLSession.shared.data(for: frozen) }
        try checkRange(response, data, begin, end, file: url.lastPathComponent)
        return data
    }

    /// `range`'s judgment of a response, apart so that a test can hand it one.
    static func checkRange(_ response: URLResponse, _ data: Data, _ begin: Int, _ end: Int, file: String) throws {
        guard let h = response as? HTTPURLResponse, h.statusCode == 206 else {
            throw EngineError.downloadRefused(file: file, status: (response as? HTTPURLResponse)?.statusCode ?? 0)
        }
        guard data.count == end - begin + 1 else { throw EngineError.downloadInterrupted(file: file) }
    }

    /// The sha256 (LFS files) and size announced by the Hub, read from the response **before**
    /// redirection to the CDN.
    static func announcedFootprint(_ url: URL) throws -> (sha256: String?, size: Int?, status: Int) {
        final class NoRedirect: NSObject, URLSessionTaskDelegate {
            func urlSession(_ s: URLSession, task: URLSessionTask, willPerformHTTPRedirection r: HTTPURLResponse,
                            newRequest: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
                completionHandler(nil)
            }
        }
        let session = URLSession(configuration: .ephemeral, delegate: NoRedirect(), delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let r = request(url, method: "HEAD")
        let (_, response) = try wait { try await session.data(for: r) }
        guard let h = response as? HTTPURLResponse else { return (nil, nil, 0) }
        guard h.statusCode < 400 else {
            throw EngineError.downloadRefused(file: url.lastPathComponent, status: h.statusCode)
        }
        let etag = (h.value(forHTTPHeaderField: "X-Linked-Etag") ?? "").trimmingCharacters(in: CharacterSet(charactersIn: "\"W/"))
        let size = h.value(forHTTPHeaderField: "X-Linked-Size").flatMap(Int.init)
        return (etag.count == 64 ? etag : nil, size, h.statusCode)
    }

    /// **The sha256 a download is checked against**: the recipe's when it pins one (a Hub that
    /// announces another is refused before the first byte; one that announces none is checked
    /// against the pin alone), else the Hub's. A weight file with neither is refused, by name: an
    /// unverified checkpoint would be forged as if it had been verified.
    static func expectedSHA(path: String, pinned: String?, announced: String?, status: Int, file: String) throws -> String? {
        if let pinned {
            if let announced, announced != pinned { throw EngineError.downloadCorrupt(file: file) }
            return pinned
        }
        if announced == nil, path.hasSuffix(".safetensors") || path.hasSuffix(".gguf") {
            throw EngineError.downloadRefused(file: file, status: status)
        }
        return announced
    }

    /// Downloads `path` of `repository@revision` to `destination`. Already there and verified: nothing.
    /// `pinned`: the sha256 the recipe expects — a Hub that announces another one is refused before
    /// the first byte, and the bytes are checked against it.
    package static func download(_ repository: String, revision: String, path: String, to destination: String,
                                    pinned: String? = nil, cancellation: Cancellation? = nil,
                                    journal: @escaping @Sendable (String) -> Void) throws {
        let fm = FileManager.default
        let u = url(repository, revision: revision, path: path)
        let (announced, size, status) = try announcedFootprint(u)
        let label = "\(repository.split(separator: "/").last!)/\(path)"
        let sha = try expectedSHA(path: path, pinned: pinned, announced: announced, status: status, file: label)
        if fm.fileExists(atPath: destination) {
            let t = Library.fileSize(destination) ?? -1
            if size == nil || t == size, sha == nil || (try? sha256(destination)) == sha { return }
            try fm.removeItem(atPath: destination)
        }
        try fm.createDirectory(atPath: (destination as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        journal("↓ \(label)" + (size.map { String(format: " (%.2f GB)", Double($0) / 1e9) } ?? ""))
        let begin = Date()
        if let size, size > chunkedFrom {
            try downloadInChunks(u, size: size, sha: sha, label: label, to: destination,
                                 cancellation: cancellation, journal: journal)
            let s = Date().timeIntervalSince(begin)
            journal(String(format: "  ✓ %.0f s, %.0f MB/s%@", s, Double(size) / s / 1e6, sha == nil ? "" : ", sha256 verified"))
            return
        }
        let tracking = Tracker(label: label, cancellation: cancellation, journal: journal)
        let session = URLSession(configuration: .default, delegate: tracking, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let temporary: URL, response: URLResponse
        do {
            (temporary, response) = try wait { try await session.download(for: request(u)) }
        } catch let e as URLError where e.code == .cancelled && cancellation?.isCancelled == true {
            throw EngineError.cancelled
        } catch is URLError {
            throw EngineError.downloadInterrupted(file: label)
        }
        guard let h = response as? HTTPURLResponse, h.statusCode == 200 else {
            try? fm.removeItem(at: temporary)
            let code = (response as? HTTPURLResponse)?.statusCode
            throw EngineError.downloadRefused(file: label, status: code ?? 0)
        }
        let partial = destination + ".partiel"
        try? fm.removeItem(atPath: partial)
        try fm.moveItem(at: temporary, to: URL(fileURLWithPath: partial))
        // Held until the file has its name: `Library.emptyDownloads` leaves a locked `.partiel` alone.
        let held = Library.holdPartial(partial)
        defer { if held >= 0 { close(held) } }
        if let size, (try fm.attributesOfItem(atPath: partial)[.size] as? Int) != size {
            try? fm.removeItem(atPath: partial)
            throw EngineError.downloadInterrupted(file: label)
        }
        if let sha {
            let received = try sha256(partial)
            guard received == sha else { try? fm.removeItem(atPath: partial); throw EngineError.downloadCorrupt(file: label) }
        }
        try fm.moveItem(atPath: partial, toPath: destination)
        let s = Date().timeIntervalSince(begin)
        if let size, size > 50_000_000 {
            journal(String(format: "  ✓ %.0f s, %.0f MB/s%@", s, Double(size) / s / 1e6, sha == nil ? "" : ", sha256 verified"))
        }
    }

    // MARK: The large files, in chunks

    /// Above this size a file comes in chunks; below, in one request (configs, tokenizers).
    static let chunkedFrom = 128 << 20
    static let chunkSize = 64 << 20
    static let parallelChunks = 8
    /// Attempts in a row **without a byte** before a chunk gives up; one that received bytes starts
    /// the count again.
    static let attemptsWithoutProgress = 6

    /// **Chunk bounds by endpoints**, never `size × index`: chunk i is `[i·size/n, (i+1)·size/n)`,
    /// n the fewest chunks of at most `chunk` bytes — contiguous, covering, never empty.
    package static func chunks(size: Int, chunk: Int = chunkSize) -> [Range<Int>] {
        let n = max(1, (size + chunk - 1) / chunk)
        return (0..<n).map { i in i * size / n ..< (i + 1) * size / n }
    }

    /// The ledger's first line: the file (size, sha256) and the chunk size its indices count in.
    static func ledgerHead(size: Int, sha: String?, chunk: Int = chunkSize) -> String { "\(size) \(sha ?? "-") \(chunk)" }

    /// The ledger of a chunked download: its first line names the file and the chunk size
    /// (`ledgerHead`), each next line a chunk written whole. A ledger of another file, of another
    /// chunk size, unreadable or damaged — a line that is not an index of this file's chunks —
    /// resumes nothing: the download starts over.
    package static func resumedChunks(ledger: String, size: Int, sha: String?, chunk: Int = chunkSize) -> Set<Int> {
        let lines = ledger.split(separator: "\n", omittingEmptySubsequences: false)
        // Only whole lines: the last element is "" after a final newline, or a line a crash cut.
        guard lines.count >= 2, lines[0] == ledgerHead(size: size, sha: sha, chunk: chunk) else { return [] }
        let count = chunks(size: size, chunk: chunk).count
        var done = Set<Int>()
        for line in lines.dropFirst().dropLast() {
            guard let i = Int(line), (0..<count).contains(i) else { return [] }
            done.insert(i)
        }
        return done
    }

    /// **A large file in chunks of 64 MB, 8 at a time, each over a connection of its own.** One
    /// long request does not hold: measured 2026-10-09 on the Hub's CDN, the installer's
    /// single connection had fallen to 2–3 MB/s after half an hour, then stopped for over 60 s
    /// (the request's timeout: 3.75 GB of 4.41 lost), while a fresh connection to the same file
    /// read 30 MB/s and eight 51. So: every attempt opens a new session (`ephemeral`, nothing
    /// shared — with one session HTTP/2 would carry the eight on one connection); a chunk that
    /// stalls 30 s is asked again from its first missing byte; the bytes go to their offset in
    /// `.partiel` as they arrive (no chunk held in memory); the chunks written whole are kept in a
    /// ledger beside it (`.morceaux.partiel`, cleaned like any `.partiel`), so that a download
    /// cut short — the network, a stop, a crash — resumes where it was. The sha256 judges the
    /// whole file at the end, as before; a corrupt one starts over.
    static func downloadInChunks(_ u: URL, size: Int, sha: String?, label: String, to destination: String,
                                 cancellation: Cancellation?, journal: @escaping @Sendable (String) -> Void) throws {
        let fm = FileManager.default
        let partial = destination + ".partiel", ledgerPath = destination + ".morceaux.partiel"
        let all = chunks(size: size)
        var done: Set<Int> = []
        if (try? fm.attributesOfItem(atPath: partial)[.size] as? Int) == size,
           let text = try? String(contentsOfFile: ledgerPath, encoding: .utf8) {
            done = resumedChunks(ledger: text, size: size, sha: sha)
        }
        if done.isEmpty {
            try? fm.removeItem(atPath: partial)
            try (ledgerHead(size: size, sha: sha) + "\n").write(toFile: ledgerPath, atomically: true, encoding: .utf8)
        } else {
            journal(String(format: "  resumed: %d of %d chunks already there", done.count, all.count))
        }
        let fd = open(partial, O_RDWR | O_CREAT, 0o644)
        guard fd >= 0, ftruncate(fd, off_t(size)) == 0 else {
            if fd >= 0 { close(fd) }
            throw EngineError.fileUnreadable(path: partial)
        }
        // Held while written: `Library.emptyDownloads` leaves a locked `.partiel` alone.
        _ = flock(fd, LOCK_EX | LOCK_NB)
        // Written past the page cache, like a map (`MapWriter.write`): tens of GB of sources, read
        // back once by the hash and once by the forge, have no business filling memory meanwhile.
        _ = fcntl(fd, F_NOCACHE, 1)
        defer { close(fd) }
        guard let ledger = FileHandle(forWritingAtPath: ledgerPath) else { throw EngineError.fileUnreadable(path: ledgerPath) }
        _ = flock(ledger.fileDescriptor, LOCK_EX | LOCK_NB)
        defer { try? ledger.close() }
        _ = try? ledger.seekToEnd()

        let run = ChunkRun(fd: fd, ledger: ledger, size: size, label: label, whole: done,
                           already: done.reduce(0) { $0 + all[$1].count }, cancellation: cancellation, journal: journal)
        // The sha256 runs behind the download, on the chunks already whole in file order — the
        // pass that followed the last byte is gone (it cost 6 s per 10 GB).
        let hash = sha == nil ? nil : RunningHash(path: partial, chunks: all, run: run)
        let pending = all.indices.filter { !done.contains($0) }
        let next = Atomic<Int>(0)
        DispatchQueue.concurrentPerform(iterations: min(parallelChunks, pending.count)) { _ in
            while run.failure == nil {
                let k = next.wrappingAdd(1, ordering: .relaxed).oldValue
                guard k < pending.count else { return }
                run.fetch(u, chunk: pending[k], range: all[pending[k]])
            }
        }
        if let e = run.failure { _ = try? hash?.result(); throw e }
        if cancellation?.isCancelled == true { _ = try? hash?.result(); throw EngineError.cancelled }
        try? ledger.close()
        try? fm.removeItem(atPath: ledgerPath)
        if let sha {
            let received = try hash?.result()
            guard received == sha else { try? fm.removeItem(atPath: partial); throw EngineError.downloadCorrupt(file: label) }
        }
        try fm.moveItem(atPath: partial, toPath: destination)
    }

    /// **The sha256 of a chunked download, computed while it downloads**: chunk i is hashed once
    /// it and every chunk before it are whole (resumed chunks are whole from the start), read back
    /// from `.partiel` (at the disk's speed, many times the network's). Stops, without a digest,
    /// when the run stops.
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

    /// The state the chunks of one file share: the descriptor they write to, the ledger, the bytes
    /// received (for the journal) and the first failure, which stops the others.
    final class ChunkRun: @unchecked Sendable {
        let fd: Int32, ledger: FileHandle, size: Int, label: String
        let cancellation: Cancellation?, journal: @Sendable (String) -> Void
        private let lock = NSCondition()
        private var received: Int, reported = 0, _failure: EngineError?
        /// The chunks written whole, resumed ones included — what `RunningHash` waits for.
        private var whole: Set<Int>

        init(fd: Int32, ledger: FileHandle, size: Int, label: String, whole: Set<Int>, already: Int,
             cancellation: Cancellation?, journal: @escaping @Sendable (String) -> Void) {
            self.fd = fd; self.ledger = ledger; self.size = size; self.label = label; self.whole = whole
            self.received = already; self.cancellation = cancellation; self.journal = journal
            reported = Int(already * 100 / size) / 20 * 20
        }

        var failure: EngineError? { lock.withLock { _failure } }
        func fail(_ e: EngineError) { lock.withLock { if _failure == nil { _failure = e }; lock.broadcast() } }
        var stopped: Bool { cancellation?.isCancelled == true || failure != nil }

        /// Waits until chunk `k` is whole; `false`: the run stopped first (a cancellation is seen
        /// within half a second — the token has no way to wake anyone).
        func waitWhole(_ k: Int) -> Bool {
            lock.lock(); defer { lock.unlock() }
            while !whole.contains(k) {
                if _failure != nil || cancellation?.isCancelled == true { return false }
                _ = lock.wait(until: Date().addingTimeInterval(0.5))
            }
            return true
        }

        func add(_ n: Int) {
            let line: String? = lock.withLock {
                received += n
                let pc = Int(received * 100 / size)
                guard pc >= reported + 20 else { return nil }
                reported = pc - pc % 20
                return "  … \(reported) %"
            }
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
                var r = Downloader.request(u)
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

    /// One request for `[at, end)`: the bytes gathered in memory and written at their offset by 8 MB
    /// (`F_NOCACHE`: a write as small as the network hands it, past the cache, scattered a file in
    /// tiny extents, read back at 0.5 GB/s by the forge). `at` is what is on disk.
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
            let written = gathered.withUnsafeBytes { b -> Int in
                var done = 0
                while done < n {
                    let r = pwrite(run.fd, b.baseAddress! + done, n - done, off_t(at + done))
                    if r <= 0 { break }
                    done += r
                }
                return done
            }
            gathered.removeAll(keepingCapacity: true)
            guard written == n else { run.fail(.fileUnreadable(path: run.label)); return false }
            at += n
            return true
        }
    }

    /// A whole file's sha256. Read by `pread` into two buffers, the next block read while this one
    /// is hashed: measured on the M1 Pro, 1.7 GB/s for read-then-hash, the hash alone ≈ 2.4. (Before that,
    /// `FileHandle.read(upToCount:)` with no autorelease pool kept every chunk alive: 31 GB and 25 GB
    /// of swap for Z-Image, 2026-10-06 — nothing here is autoreleased.)
    package static func sha256(_ path: String) throws -> String {
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
            let next = Next(), group = DispatchGroup()
            nonisolated(unsafe) let target = buffers[(k + 1) % 2]
            let from = at + count
            DispatchQueue.global(qos: .utility).async(group: group) { next.count = fill(target, from) }
            h.update(bufferPointer: UnsafeRawBufferPointer(start: buffers[k % 2], count: count))
            group.wait()
            (at, count, k) = (from, next.count, k + 1)
        }
        guard count == 0 else { throw EngineError.fileUnreadable(path: path) }
        return h.finalize().map { String(format: "%02x", $0) }.joined()
    }

    final class Tracker: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
        let label: String
        let cancellation: Cancellation?
        let journal: @Sendable (String) -> Void
        var last = 0
        init(label: String, cancellation: Cancellation?, journal: @escaping @Sendable (String) -> Void) {
            self.label = label; self.cancellation = cancellation; self.journal = journal
        }
        func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
        func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didWriteData _: Int64,
                        totalBytesWritten written: Int64, totalBytesExpectedToWrite total: Int64) {
            if cancellation?.isCancelled == true { downloadTask.cancel(); return }
            guard total > 200_000_000 else { return }
            let pc = Int(written * 100 / total)
            if pc >= last + 20 { last = pc - pc % 20; journal("  … \(last) %") }
        }
    }
}

/// Waits for an asynchronous task from synchronous code (the forge and the CLI are).
private final class Box<T>: @unchecked Sendable { var r: Result<T, Error>? }

func wait<T: Sendable>(_ task: @escaping @Sendable () async throws -> T) throws -> T {
    let b = Box<T>(), s = DispatchSemaphore(value: 0)
    Task.detached {
        do { b.r = .success(try await task()) } catch { b.r = .failure(error) }
        s.signal()
    }
    s.wait()
    return try b.r!.get()
}
