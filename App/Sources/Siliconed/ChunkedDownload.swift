// A large file downloaded in chunks, in parallel, resumed from its ledger.

import CryptoKit
import Foundation
import Measurement
import Synchronization

extension Downloader {
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

    /// **What a resumption keeps of an interrupted download of `file` to `destination`**: the chunks its ledger says are
    /// whole, read as `downloadInChunks` reads them. A chunk being written when it stopped comes again whole — up to
    /// `parallelChunks` of them, though their bytes are on disk —, and a file that comes in one request resumes nothing.
    package static func resumable(_ file: PublishedFile, at destination: String) -> Int {
        guard file.bytes > chunkedFrom, case .sha256(let sha) = file.digest, Library.fileSize(partial(destination)) == file.bytes,
              let text = try? String(contentsOfFile: ledger(destination), encoding: .utf8) else { return 0 }
        let all = chunks(size: file.bytes)
        return resumedChunks(ledger: text, size: file.bytes, digest: sha).reduce(0) { $0 + all[$1].count }
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
                run.fail(EngineError.ranOutOfRoom(errno: failure) ? .diskFull(needed: nil, available: nil) : .downloadInterrupted(file: run.label))
                return false
            }
            at += n
            return true
        }
    }
}
