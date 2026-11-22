import CryptoKit
import Foundation
import Measurement
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
