// A file's sha256: a whole file's, and a chunked download's computed while it downloads.

import CryptoKit
import Foundation
import Measurement
import Synchronization

extension Downloader {
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
                digest = h.finalize().hexadecimal
            }
        }

        /// The digest, once every chunk is hashed; `nil`: the run stopped first.
        func result() throws -> String? {
            finished.wait()
            if let error { throw error }
            return digest
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
        return h.finalize().hexadecimal
    }
}
