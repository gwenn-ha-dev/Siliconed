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
///     per class of service (`StreamClass`)      the same passes through a queue of that class, **beside a computation holding every core** (`Beside`: a
///                                                spinning thread per logical CPU): `Reading.rate` (the fastest pass, as above) and `Reading.mean` (all the
///                                                passes together) — what a reader running beside the computation sustains
///     the class a phase reads at (`choose`)     the lowest whose mean covers  bytes / C · (1 + ε)  — the rate the phase's weights must be read at
///                                                to be hidden under its C seconds of computation — else the default class. ε is the plus-or-minus of
///                                                the predicted C: half the spread the diagnostic accepts of a chain (`Diagnostic.Regime.Rules.spread`).
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

    /// **What the passes of one class gave**: `rate` over the fastest pass (the disk's), `mean` over all of them together (total bytes over
    /// total seconds — what a stream of several thousand visits sustains: the read-ahead smooths the pass-to-pass variation, and a render
    /// reads a hundred times what a pass does).
    package struct Reading: Codable, Sendable, Equatable {
        package var rate: Double
        package var mean: Double
        package init(rate: Double, mean: Double) { self.rate = rate; self.mean = mean }
        /// The relative spread of the passes: `rate / mean − 1`.
        package var spread: Double { mean > 0 ? rate / mean - 1 : 0 }
    }

    /// **The readings of the passes**: `nil` when none has a time.
    package static func reading(bytes: Int, passes seconds: [Double]) -> Reading? {
        let timed = seconds.filter { $0 > 0 }
        guard bytes > 0, let fastest = timed.min() else { return nil }
        return Reading(rate: Double(bytes) / fastest, mean: Double(bytes) * Double(timed.count) / timed.reduce(0, +))
    }

    /// **The plus-or-minus of a predicted computation time**: half of what a chain may spread and still be believed.
    package static var computeUncertainty: Double { Diagnostic.Regime.Rules.spread / 2 }

    /// **The lowest class whose rate covers a phase** — the equation of the weight stream's priority. A phase that reads `bytes` while it
    /// computes `compute` seconds needs the disk at `bytes / compute` to hide the read; a lower class leaves the cores to the computation
    /// but reads slower. The class is the lowest whose **mean** (not its best pass) is at least that need times `1 + uncertainty`;
    /// the default class when none covers (the disk binds the phase, or the map has no reading: the stream then reads as it always did).
    package static func choose(bytes: Double, compute: Double, readings: [StreamClass: Reading], uncertainty: Double = computeUncertainty) -> StreamClass {
        guard bytes > 0, compute > 0 else { return .standard }
        let need = bytes / compute * (1 + uncertainty)
        for c in StreamClass.allCases where c != .standard {
            if let r = readings[c], r.mean >= need { return c }
        }
        return .standard
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

    /// **The computation the reads run beside**: one thread per logical CPU spinning at the engine's own class (`.userInitiated`) while the
    /// passes read, until `stop()`. The classes differ by what they yield to the computation, and a read on an idle Mac shows none of it.
    /// Measured on the M1 Pro (4 GB/s idle for every class): `background` keeps 3.4–4 GB/s with up to 6 of 8 performance cores held and
    /// falls to 0.1 GB/s when all are (starved, not slowed: the cliff is in the count of cores the computation leaves free); `utility`
    /// loses a fifth, the default nothing. The probe holds them all: **the rate a class keeps when the computation takes every core** is
    /// the one that holds whatever the phase asks of the cores — the map does not know how many a phase holds, so it does not bet on it.
    final class Beside: @unchecked Sendable {
        private let flag = Atomic<Bool>(false)
        private let done = DispatchGroup()
        init(threads: Int) {
            for _ in 0..<max(0, threads) {
                done.enter()
                let t = Thread { [self] in
                    var x: Float = 1.0001, y: Float = 0.9999
                    while !self.flag.load(ordering: .relaxed) {
                        for _ in 0..<(1 << 16) { x = x * y + 0.5; y = y * x - 0.4999 }
                    }
                    if x.isNaN && y.isNaN { print() }
                    self.done.leave()
                }
                t.qualityOfService = .userInitiated
                t.start()
            }
        }
        func stop() { flag.store(true, ordering: .relaxed); done.wait() }
    }

    /// The most a class other than the default adds to the probe once it has one pass: the passes stop when this many seconds are spent.
    package static let classBudget = 1.0

    /// **Measures the disk under `directory`**: the rate of the default class (see `measureClasses`). `nil` when the volume cannot
    /// hold or give the scratch file.
    package static func measure(in directory: URL, bytes: Int = probeBytes, visit unit: Int = visit, passes count: Int = passes) -> Double? {
        measureClasses(in: directory, bytes: bytes, visit: unit, passes: count, classes: [.standard])?[.standard]?.rate
    }

    /// **Measures the disk under `directory` at each class of service**: writes the scratch file once (without the cache, then `fsync`),
    /// reads it `count` times through a queue of the default class, then of each lower one — at most `count` passes, and from the second
    /// pass on none once `classBudget` seconds went to the class — removes it. About 1 s on an SSD for the default class, about as much for the
    /// others together on a fast one; at most `classBudget` plus one pass for each lower class on a slow one. `nil` when the volume cannot
    /// hold or give the file.
    package static func measureClasses(in directory: URL, bytes: Int = probeBytes, visit unit: Int = visit, passes count: Int = passes,
                                       classes: [StreamClass] = [.standard, .utility, .background],
                                       beside threads: Int = Diagnostic.sysctlInteger("hw.ncpu") ?? 0) -> [StreamClass: Reading]? {
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
        var readings: [StreamClass: Reading] = [:]
        let failed = Failure()
        let computation = Beside(threads: threads)
        defer { computation.stop() }
        for cls in classes {
            // The queue of the class the stream's own would be: the reads (and the lanes' `concurrentPerform`) run at its priority.
            let queue = DispatchQueue(label: "siliconed.disk-probe.\(cls.rawValue)", qos: cls.qos, attributes: .concurrent)
            var seconds: [Double] = []
            let begun = DispatchTime.now().uptimeNanoseconds
            for pass in 0..<count {
                if pass > 0, cls != .standard, Double(DispatchTime.now().uptimeNanoseconds - begun) * 1e-9 >= classBudget { break }
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
            if let r = reading(bytes: bytes, passes: seconds) { readings[cls] = r }
        }
        return readings.isEmpty ? nil : readings
    }
}
