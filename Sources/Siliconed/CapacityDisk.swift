import Darwin
import Foundation
import Synchronization

/// **The disk as a resource of the map: its rate as the weight stream reads it, and the arithmetic of a stream beside the computation.**
/// The probe (`Diagnostic.CapacityProbes`) calls `measure`; the solver reads `CapacityModel.diskBandwidth` and `pass`.
///
///     probe   a scratch file of `probeBytes` on the library's volume, written once, then read `passes` times **the way the stream reads it**:
///             visits of `visit` bytes, each cut in `TailStream.lanes` page-aligned lanes of `pread` on an `F_NOCACHE` descriptor
///             (`TailStream.start`'s own cut), `TailStream.unitSlots − 1` visits in flight (the others' buffer is being computed from)
///             rate = bytes / the fastest pass         (a disturbance — the mouse, the Activity Monitor, a neighbour's I/O — only adds time:
///                                                      the fast side is the disk's, as the chains of the engines are read)
///
///     a pass of `evaluations` evaluations, each computing C seconds and having R = bytes / rate seconds of its weights to read:
///         the read of what comes next runs under the product that is computed, so
///         T = max(C, R) · evaluations + exposed · evaluations
///         `exposed`: what no computation hides — the first visit's read, the sweep cannot start before the evaluation does
///
/// What sets it apart — what a disk model written by analogy with "the Mac streams at the SSD's rate" would get wrong:
///
///   - **the rate is the stream's, not the SSD's**: a plain 4-lane read of large pieces reaches 5–6 GB/s on the M1 Pro; the stream, which
///     reads visits of ~100 MB two at a time, gets 3.3 — the probe reads as the stream does, so that R is the stream's.
///   - **the rate is a floor on the time, not the time**: a render that reads 97.6 GB in 100 s is at 1 GB/s on a disk that reads 3.3; the
///     disk sets the pace only where R exceeds C (a half-grid evaluation of a fast Mac), and only for that excess.
///   - **the bytes are the engine's, by construction**: the units the stream reads (`Artifact.stagingUnits`), not the map's size — a
///     tensor the cache keeps is not read twice, one used in two visits is read twice.
package enum CapacityDisk {
    /// The visit the probe reads (a stream's unit is its largest tensor: 100 MB on the Compact Krea 2, 435 MB on the Standard's layers) and
    /// the scratch file: six visits, so that the two in flight are in steady state for most of the pass.
    package static let visit = 96 << 20
    package static let probeBytes = 6 * visit
    package static let passes = 3

    /// **The rate the passes give**: `bytes` over the fastest pass; `nil` when no pass has a time.
    package static func rate(bytes: Int, passes seconds: [Double]) -> Double? {
        guard bytes > 0, let fastest = seconds.filter({ $0 > 0 }).min() else { return nil }
        return Double(bytes) / fastest
    }

    /// **The seconds a pass of `evaluations` evaluations takes** when each computes `compute` seconds and reads `reading` seconds of
    /// weights beside it; `exposed` seconds of each are read before anything can be computed. All three are per evaluation.
    package static func pass(compute: Double, reading: Double, exposed: Double, evaluations: Int) -> Double {
        Double(evaluations) * (max(compute, reading) + exposed)
    }

    // MARK: - The probe

    /// **One visit read as `TailStream` reads one**: `[start, end)` of the file cut in `TailStream.lanes` lanes (a multiple of 16 KiB, at
    /// least `smallest`), each lane `pread` in pieces of `TailStream.piece`. Blocks until the visit is in `buffer`.
    /// **A short read is a failure** (`false`): a visit that returned fewer bytes than asked must not count as read, or the rate
    /// of a disk that errors would be read as a fast one.
    @discardableResult
    static func read(_ descriptor: Int32, from start: Int, to end: Int, into buffer: UnsafeMutableRawPointer) -> Bool {
        let lane = max(((end - start) / TailStream.lanes + (1 << 14) - 1) / (1 << 14) * (1 << 14), TailStream.smallest)
        var firsts: [Int] = []
        var first = start
        while first < end { firsts.append(first); first += lane }
        let failed = Atomic<Bool>(false)
        DispatchQueue.concurrentPerform(iterations: firsts.count) { index in
            var offset = firsts[index]
            let last = min(offset + lane, end)
            while offset < last {
                let length = min(TailStream.piece, last - offset)
                var done = 0
                while done < length {
                    let n = pread(descriptor, buffer + (offset - start) + done, length - done, off_t(offset + done))
                    if n <= 0 { failed.store(true, ordering: .relaxed); return }
                    done += n
                }
                offset += length
            }
        }
        return !failed.load(ordering: .relaxed)
    }

    /// **The folder whose volume the weight stream reads from**: the library's own, unless a render reads a working copy
    /// (`WorkingCopy.isNeeded`: a library on another disk or a share), in which case this Mac's cache folder — the disk the
    /// stream really meets. Created if missing, since the probe writes there.
    package static func streamFolder(for library: Library, copyBase: URL = WorkingCopy.base) -> URL {
        if WorkingCopy.isNeeded(for: library, base: copyBase) {
            try? FileManager.default.createDirectory(at: copyBase, withIntermediateDirectories: true)
            return copyBase
        }
        return library.root
    }

    /// **The memory the probe holds while it reads**: the staging slots of the visits in flight and the scratch piece it writes
    /// from — what `CapacityProbes.need` charges to the stage, so that B holds for the diagnostic itself.
    package static func probeMemoryBytes(visit unit: Int = visit) -> Int {
        Arena.capacity(for: unit) * max(1, TailStream.unitSlots - 1) + Arena.capacity(for: scratchPiece)
    }
    static let scratchPiece = 8 << 20

    /// A raised flag shared with the reads in flight.
    private final class Failure: @unchecked Sendable {
        private let flag = Atomic<Bool>(false)
        func raise() { flag.store(true, ordering: .relaxed) }
        var raised: Bool { flag.load(ordering: .relaxed) }
    }

    /// **Measures the disk under `directory`**: writes the scratch file (without the cache, then `fsync`), reads it `passes` times,
    /// removes it. `nil` when the volume cannot hold or give it. About 1 s on an SSD.
    package static func measure(in directory: URL, bytes: Int = probeBytes, visit unit: Int = visit, passes count: Int = passes) -> Double? {
        let path = directory.appendingPathComponent(".siliconed-disk-probe-\(getpid())").path
        let fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o600)
        guard fd >= 0 else { return nil }
        defer { close(fd); unlink(path) }
        _ = fcntl(fd, F_NOCACHE, 1)
        let piece = scratchPiece
        guard let scratch = mmap(nil, piece, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0), scratch != MAP_FAILED else { return nil }
        defer { munmap(scratch, piece) }
        // Incompressible bytes: a volume that compresses on write must not be read faster than it reads.
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        let words = scratch.bindMemory(to: UInt64.self, capacity: piece / 8)
        for i in 0..<(piece / 8) { state ^= state << 13; state ^= state >> 7; state ^= state << 17; words[i] = state }
        var written = 0
        while written < bytes {
            let n = pwrite(fd, scratch, min(piece, bytes - written), off_t(written))
            guard n > 0 else { return nil }
            written += n
        }
        guard fsync(fd) == 0 else { return nil }
        let reader = open(path, O_RDONLY)
        guard reader >= 0 else { return nil }
        defer { close(reader) }
        _ = fcntl(reader, F_NOCACHE, 1)
        // The buffers of the visits in flight, as the stream's slots.
        let flying = max(1, TailStream.unitSlots - 1)
        var slots: [UnsafeMutableRawPointer] = []
        for _ in 0..<flying {
            guard let slot = mmap(nil, unit, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0), slot != MAP_FAILED else {
                for s in slots { munmap(s, unit) }
                return nil
            }
            slots.append(slot)
        }
        defer { for s in slots { munmap(s, unit) } }
        let visits = (0..<((bytes + unit - 1) / unit)).map { (start: $0 * unit, end: min(bytes, ($0 + 1) * unit)) }
        let queue = DispatchQueue(label: "siliconed.disk-probe", attributes: .concurrent)
        var seconds: [Double] = []
        let failed = Failure()
        for _ in 0..<count {
            let started = DispatchTime.now().uptimeNanoseconds
            let room = DispatchSemaphore(value: flying), group = DispatchGroup()
            for (index, v) in visits.enumerated() {
                room.wait()
                let slot = slots[index % flying]
                queue.async(group: group) {
                    if !read(reader, from: v.start, to: v.end, into: slot) { failed.raise() }
                    room.signal()
                }
            }
            group.wait()
            if failed.raised { return nil }
            seconds.append(Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9)
        }
        return rate(bytes: bytes, passes: seconds)
    }
}
