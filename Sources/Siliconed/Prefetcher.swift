import Darwin
import Foundation
import Synchronization

/// **The class of service of the threads that read the weights** (`TailStream`, `Prefetcher`), lowest first. A concurrent `pread` at the
/// default class slows the products beside it (the GEMM of a 512² Anima evaluation 0.95 → 1.71 s); at `.background` the compute keeps the
/// performance cores, but the kernel also throttles the reads (its I/O policy, the efficiency cores). Which class a phase can afford is
/// not a constant: it is the lowest whose measured rate covers what the phase has to read (`CapacityDisk.choose`, from the map).
/// Timing only: no value is computed differently.
package enum StreamClass: String, CaseIterable, Sendable, Comparable {
    case background, utility, standard = "default"
    package var qos: DispatchQoS {
        switch self {
        case .background: return .background
        case .utility: return .utility
        case .standard: return .unspecified
        }
    }
    package var rank: Int { Self.allCases.firstIndex(of: self)! }
    package static func < (a: StreamClass, b: StreamClass) -> Bool { a.rank < b.rank }
}

package enum StreamPriority {
    /// `SILICONED_STREAM_QOS` = `default` | `utility` | `background`: **a forcing of measurement** (as `SILICONED_PLAN` is), which the map's
    /// choice does not override. Unset or unknown: the map decides.
    package static let forced: StreamClass? = StreamClass(rawValue: ProcessInfo.processInfo.environment["SILICONED_STREAM_QOS"] ?? "")
}

/// The prefetcher: a few threads that tell the kernel what the computation is going to read, before it reads it.
///
/// There is **no residency policy here**, and that is the point. The access order is the order of the
/// file — reading the map ahead is executing the model — so all that remains is a hint to
/// issue early enough. What the kernel keeps or drops is its own business, and since the pages are
/// clean and file-backed, what it drops never goes to swap.
///
/// Measured cold, over the map's 12.33 GB:
///
///     bare mmap, 1 thread            0.57 GB/s    ← what the v0 engine suffered
///     madvise(WILLNEED) 64 MB, 1 thread 1.58
///     madvise(WILLNEED) 64 MB, 4 threads 4.26     ← 71 % of the SSD ceiling
///     pread + F_NOCACHE, 4 threads   6.01         ← the ceiling, and it costs a policy
///
/// 4.26 GB/s covers both regimes: 0.75 GB/s to sustain at 1024², 3.4 at 512². The ratio is judged
/// **at the bottom** — the weights are the same at all resolutions, so it is the step that is cheapest
/// in compute that decides.
package final class Prefetcher {
    /// 64 MiB. At 256 the measurement falls back to 1.60 GB/s: a slice that is too large drowns the kernel's
    /// read-ahead queue and the touch catches up with the read instead of following it.
    package static let span = 64 << 20
    private static let defaultThreads = 4

    private let artifact: Artifact
    private let threads: Int
    private let queue = DispatchQueue(label: "siliconed.prefetch", qos: (StreamPriority.forced ?? .standard).qos, attributes: .concurrent)
    private let group = DispatchGroup()
    private let lock = NSLock()
    private var inFlight: Set<String> = []

    /// A switch, to measure what the prefetcher costs rather than assume it is null.
    package var enabled = true

    package init(artifact: Artifact, threads: Int = Prefetcher.defaultThreads) {
        self.artifact = artifact
        self.threads = threads
    }

    /// Requests the paging-in of the named tensors, without waiting. A name already in flight is not
    /// requested again: the conductor calls at every block and the ranges overlap.
    package func request(_ names: [String]) {
        guard enabled else { return }
        let wanted: [String] = {
            lock.lock(); defer { lock.unlock() }
            let fresh = names.filter { !inFlight.contains($0) }
            inFlight.formUnion(fresh)
            return fresh
        }()
        guard !wanted.isEmpty else { return }
        // Whatever path returns, the names leave the in-flight set: a name left in it is never
        // requested again (a fully staged block has no slice to page in, and used to return early here).
        defer { lock.lock(); inFlight.subtract(wanted); lock.unlock() }
        // The streamed tail is read into its staging buffers, not paged in (`TailStream`).
        let staged = artifact.stream?.stage(wanted) ?? []

        // The ranges are cut into slices then distributed: a big tensor must occupy
        // several threads, otherwise parallelism depends on the size of the tensors.
        var slices: [(UnsafeRawPointer, Int)] = []
        // A tensor kept in the process's pages (`Artifact.keepResidentAhead`) is never paged in: its read is the keep's.
        for name in wanted where !staged.contains(name) && !artifact.isResident(name) {
            guard let tensor = artifact.tensors[name], let base = artifact.mappedPointer(name) else { continue }
            var offset = 0
            while offset < tensor.bytes {
                let length = min(Self.span, tensor.bytes - offset)
                slices.append((base.advanced(by: offset), length))
                offset += length
            }
        }
        guard !slices.isEmpty else { return }

        // What the threads capture is immutable and does not hold `self`: the slices (addresses
        // in the memory-mapped map) and the page size. The addresses stay valid because
        // the prefetcher holds the `Artifact` and **waits for its threads before dying** (`deinit`):
        // a cancellation that tears down the DiT in the middle of an evaluation unmaps nothing from under them.
        // The addresses travel as integers: a raw pointer is not `Sendable`, an address is.
        let addressSlices = slices.map { (Int(bitPattern: $0.0), $0.1) }, page = artifact.page
        let workers = min(threads, addressSlices.count)
        for worker in 0..<workers {
            queue.async(group: group) { [addressSlices] in
                var index = worker
                while index < addressSlices.count {
                    let (address, length) = addressSlices[index]
                    let pointer = UnsafeRawPointer(bitPattern: address)!
                    _ = madvise(UnsafeMutableRawPointer(mutating: pointer), length, MADV_WILLNEED)
                    // `madvise` is an asynchronous hint: without the touch, the computation would pay
                    // the fault anyway. One byte per page suffices to establish it.
                    var sum: UInt64 = 0
                    var byte = 0
                    while byte < length {
                        sum &+= UInt64(pointer.load(fromByteOffset: byte, as: UInt8.self))
                        byte += page
                    }
                    // Without a consumer, the optimizer removes the loop. Four threads write here at the
                    // same time: an atomic addition, not `sink = sink &+ sum` on a bare
                    // static — it was a real data race (benign, but TSan flags it).
                    Self.sink.wrappingAdd(sum, ordering: .relaxed)
                    index += workers
                }
            }
        }
    }

    deinit { group.wait() }

    /// **The first of `layers` whose pages are handed back once computed** (`EngineSettings.mapTail`).
    /// The layers before it are the head of the map, left to the page cache from one
    /// evaluation to the next; the ones from it on are the tail, read, widened, then released first
    /// in line so that reading them does not evict the head. `layers` when nothing is released.
    package static func firstReleasedLayer(of layers: Int, tail: Double = EngineSettings.effective.mapTail) -> Int {
        layers - Int((tail * Double(layers)).rounded())
    }

    /// The names of all of a block's tensors, in file order.
    package func namesOfBlock(prefix: String) -> [String] {
        artifact.order.filter { $0.hasPrefix(prefix) }
    }

    private static let sink = Atomic<UInt64>(0)
}

/// **The tail of a map, read around the page cache**.
///
/// A map larger than what the page cache holds, read in file order at every evaluation, misses
/// every page under LRU: by the time the sweep comes back to the head, the tail has evicted it.
/// Z-Image's 12.33 GB at 512² read 12.33 GB from disk per evaluation, with 9–10 GB of it
/// resident. Hinting the kernel does not change that: `madvise(MADV_DONTNEED)`,
/// `msync(MS_DEACTIVATE / MS_KILLPAGES / MS_INVALIDATE)` on the tail after use all leave its pages
/// resident (`mincore`), and the head is evicted in their place.
///
/// So the tail never enters the cache: each of its blocks is read with `pread` on an `F_NOCACHE`
/// descriptor — the fastest path measured, 6.0 GB/s with four lanes — into a staging buffer, and the
/// engine widens from there (`Artifact.pointer`). The head stays the map's, and stays cached.
///
/// The staging buffers are **anonymous memory**, the only kind that can swap: `slots` buffers of
/// the largest block, allocated once, reused at every evaluation. A block requested while no slot
/// is free waits for the next request; read with no slot free at all, it goes through the map:
/// slower, never wrong.
///
/// **A block is a visit, not necessarily a layer.** Given as the layer's tensors in file order, a block
/// is read whole and requested by name (the first mode). Given as **visits in the order the engine
/// consumes them** (`sequential`; `Artifact.stagingUnits`) — a few tensors up to the largest one, which
/// is the buffer's size —, the stream drives itself: whenever a buffer is given back (`release`, the
/// engine's call as soon as it has widened a tensor), it reads the next visit of the sequence, wrapping
/// from the last to the first for the next evaluation. The buffers then hold the largest tensor, not
/// the largest layer, and the read of the next visit still overlaps the computation of the current one.
/// A tensor the engine uses several times in a row of the sequence (a layer's weights for each group of
/// rows, the MLP's for each chunk) appears in as many visits, and is read as many times: a visit is
/// complete when each of its names has been released as many times as it appears.
package final class TailStream {
    package let blocks: [[String]]
    package let slotBytes: Int
    /// The staging buffers allocated.
    package var slotCount: Int { slots.count }
    /// The tensors served from the map because no buffer was free when the engine asked: a sequence that
    /// is not the engine's shows here (it costs page cache, never a wrong weight).
    package private(set) var fallbacks = 0
    /// The visits read on demand rather than ahead: the engine asked for a visit the stream had not reached.
    package private(set) var demanded = 0
    /// A sequence of visits (see above): the engine's widening of a tensor releases it (`Artifact.materialize`).
    package let sequential: Bool
    /// **The buffers of a sequence**: the visit the engine widens from and two read ahead. Three, not two:
    /// a visit can be short (a norm, a table), and the read of the next one must not wait for it.
    package static let unitSlots = 3

    private enum State { case pending, staged, complete }
    private struct Staged { let slot: Int; let group: DispatchGroup }
    /// A visit's reads: `runs` in the file, laid back to back in its buffer; where each distinct name is;
    /// how many times the visit uses it.
    private struct Visit {
        let runs: [(start: Int, end: Int)]
        let offsets: [String: Int]
        let uses: [String: Int]
    }
    private let descriptor: Int32
    private let slots: [UnsafeMutableRawPointer]
    private var free: [Int]
    private var staged: [Int: Staged] = [:]                 // visit → its buffer
    private var state: [State]
    private var left: [[String: Int]]                       // visit → uses still to come
    private var done: Set<String> = []                      // released since their visit was last read
    private let visits: [Visit]
    private let visitsOf: [String: [Int]]                   // name → the visits that use it, ascending
    private var frontier = 0                                // the first visit the engine has not completed
    private var cursor = 0                                  // sequential: the next visit to read ahead
    private var evaluation = 0                              // whole sweeps of the sequence completed (the trace's)
    private let mapped: Int                                 // the map's base address, for the fallback copy
    private let lock = NSLock()
    /// One queue per class (a queue's class is fixed when it is made); `priority` says which one the next reads go to.
    private let queues = StreamClass.allCases.map { DispatchQueue(label: "siliconed.tail.\($0.rawValue)", qos: $0.qos, attributes: .concurrent) }
    private var priority: StreamClass = StreamPriority.forced ?? .standard
    private var queue: DispatchQueue { queues[priority.rank] }
    /// The bytes one sweep of the sequence reads: what an evaluation asks of the disk.
    package let sweepBytes: Int
    /// **The class the reads from now on run at.** A forcing (`SILICONED_STREAM_QOS`) keeps it. The reads already queued finish where
    /// they were; the next visit's go to the new class.
    package func setPriority(_ new: StreamClass) {
        lock.lock(); defer { lock.unlock() }
        if StreamPriority.forced == nil { priority = new }
    }
    private let reads = DispatchGroup()
    /// Measured: four lanes, slices of 64 MiB; a run under 4 MiB is one read.
    package static let lanes = 4, piece = 64 << 20, smallest = 4 << 20

    /// - Parameter buffer: the buffers' size, for a sequence of visits (`nil`: the largest visit).
    init(artifact: Artifact, blocks: [[String]], slots count: Int, sequential: Bool = false, buffer: Int? = nil) throws {
        self.blocks = blocks
        self.sequential = sequential
        let page = artifact.page
        var visits: [Visit] = [], visitsOf: [String: [Int]] = [:]
        for (b, names) in blocks.enumerated() {
            var uses: [String: Int] = [:]
            for name in names { uses[name, default: 0] += 1 }
            let distinct = Array(uses.keys)
            let tensors = distinct.compactMap { artifact.tensors[$0] }
            guard tensors.count == distinct.count, !tensors.isEmpty else {
                throw Artifact.Failure.badHeader("streamed block \(b): tensor missing from the map")
            }
            // Runs: each tensor's, page-rounded; those that touch are one read (a layer in file order is one).
            let sorted = zip(distinct, tensors).sorted { $0.1.offset < $1.1.offset }
            var runs: [(start: Int, end: Int)] = [], placed: [String: Int] = [:], used = 0
            for (name, t) in sorted {
                let end = min((t.offset + t.bytes + page - 1) / page * page, artifact.size)
                if let last = runs.last, t.offset <= last.end {
                    placed[name] = used - (last.end - t.offset)
                    if end > last.end { used += end - last.end; runs[runs.count - 1].end = end }
                } else {
                    placed[name] = used
                    runs.append((t.offset, end)); used += end - t.offset
                }
            }
            visits.append(Visit(runs: runs, offsets: placed, uses: uses))
            for name in distinct { visitsOf[name, default: []].append(b) }
        }
        self.visits = visits
        self.visitsOf = visitsOf
        state = Array(repeating: .pending, count: visits.count)
        left = visits.map(\.uses)
        let bytes = visits.map { $0.runs.reduce(0) { $0 + $1.end - $1.start } }
        let slotBytes = max(buffer ?? 0, bytes.max()!)
        self.slotBytes = slotBytes
        sweepBytes = bytes.reduce(0, +)
        mapped = Int(bitPattern: artifact.mappedPointer(artifact.order[0])!) - artifact.tensors[artifact.order[0]]!.offset
        // Mapped, not `malloc`ed: `munmap` hands the pages back at once, where malloc's large-block
        // cache could keep a freed gigabyte in the footprint through the decoding.
        var slots: [UnsafeMutableRawPointer] = []
        for _ in 0..<count {
            guard let p = mmap(nil, slotBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0), p != MAP_FAILED else {
                let code = errno
                for slot in slots { munmap(slot, slotBytes) }
                throw Artifact.Failure.cannotOpen("staging buffer of \(slotBytes) bytes", code)
            }
            slots.append(p)
        }
        let fd = open(artifact.path, O_RDONLY)
        guard fd >= 0 else {
            let code = errno
            for slot in slots { munmap(slot, slotBytes) }
            throw Artifact.Failure.cannotOpen(artifact.path, code)
        }
        _ = fcntl(fd, F_NOCACHE, 1)
        descriptor = fd
        self.slots = slots
        free = Array((0..<count).reversed())
    }

    deinit {
        reads.wait()
        if fallbacks > 0 {
            Warnings.emit("⚠ the weight stream served \(fallbacks) tensor(s) through the page cache: the engine left its order")
        }
        close(descriptor)
        for slot in slots { munmap(slot, slotBytes) }
    }

    /// Under the lock: the visit of the sequence that uses `name` next — the first from the frontier on, in
    /// the sequence's order, that is not complete and still has a use of it to come.
    private func visit(of name: String) -> Int? {
        guard let list = visitsOf[name] else { return nil }
        let from = list.firstIndex { $0 >= frontier } ?? 0
        for k in 0..<list.count {
            let b = list[(from + k) % list.count]
            switch state[b] {
            case .complete: continue
            case .pending: return b
            case .staged: if left[b][name, default: 0] > 0 { return b }
            }
        }
        return nil
    }

    /// Starts reading the blocks these names belong to, without waiting.
    ///
    /// A block that finds no free slot is **not** paged in instead: it is staged by a later request
    /// (the next layer's, once a slot is released) or, at the latest, when the engine reads it.
    /// So the returned set is every streamed name, staged or deferred — none of them is the map's.
    /// A sequence of visits ignores the names' order: it reads on from where it is (`advance`).
    func stage(_ names: [String]) -> Set<String> {
        lock.lock(); defer { lock.unlock() }
        var out: Set<String> = []
        for name in names where visitsOf[name] != nil {
            out.insert(name)
            if !sequential, let b = visit(of: name), state[b] == .pending, let slot = free.popLast() { start(visit: b, in: slot) }
        }
        if sequential { advance() }
        return out
    }

    /// Under the lock, sequential: the free buffers read the visits that come next, skipping those already read.
    private func advance() {
        var tried = 0
        while !free.isEmpty, tried < visits.count {
            let b = cursor
            cursor = (cursor + 1) % visits.count
            tried += 1
            if state[b] == .pending { start(visit: b, in: free.removeLast()) }
        }
    }

    /// Under the lock: the visit's reads, each run in `lanes` page-aligned pieces.
    private func start(visit b: Int, in slot: Int) {
        let group = DispatchGroup()
        let base = Int(bitPattern: slots[slot])
        let fd = descriptor, mapped = self.mapped
        var placed = 0
        for (start, end) in visits[b].runs {
            let into = base + placed
            placed += end - start
            let lane = max(((end - start) / Self.lanes + (1 << 14) - 1) / (1 << 14) * (1 << 14), Self.smallest)
            var first = start
            while first < end {
                let last = min(first + lane, end), from = first
                queue.async(group: group) {
                    var offset = from
                    while offset < last {
                        let length = min(Self.piece, last - offset)
                        let destination = UnsafeMutableRawPointer(bitPattern: into + offset - start)!
                        var done = 0
                        while done < length {
                            let n = pread(fd, destination + done, length - done, off_t(offset + done))
                            if n <= 0 {
                                // A failed read must not become a wrong weight: the map has the bytes.
                                memcpy(destination + done, UnsafeRawPointer(bitPattern: mapped + offset + done)!,
                                       length - done)
                                break
                            }
                            done += n
                        }
                        offset += length
                    }
                }
                first = last
            }
        }
        reads.enter()
        if StreamTrace.enabled {
            let started = StreamTrace.now(), bytes = visits[b].runs.reduce(0) { $0 + $1.end - $1.start }, evaluation = evaluation
            group.notify(queue: queue) { [reads] in
                StreamTrace.shared.read(visit: b, evaluation: evaluation, bytes: bytes, started: started, ended: StreamTrace.now())
                reads.leave()
            }
        } else {
            group.notify(queue: queue) { [reads] in reads.leave() }
        }
        staged[b] = Staged(slot: slot, group: group)
        state[b] = .staged
        left[b] = visits[b].uses
        done.subtract(visits[b].uses.keys)
    }

    /// The staged copy of a tensor once its block is read, or `nil` if it is not staged.
    func pointer(_ name: String, offset: Int) -> UnsafeRawPointer? {
        let asked = StreamTrace.enabled ? StreamTrace.now() : 0
        lock.lock()
        guard visitsOf[name] != nil else { lock.unlock(); return nil }
        var found = visit(of: name)
        if let b = found, state[b] == .pending {
            // Not requested ahead (or no slot was free then): staged now, or read from the map.
            if let slot = free.popLast() {
                start(visit: b, in: slot)
                demanded += 1
                if sequential { cursor = (b + 1) % visits.count }
            } else { found = nil }
        }
        guard let b = found, let entry = staged[b] else { fallbacks += 1; lock.unlock(); return nil }
        let place = visits[b].offsets[name]!
        let evaluation = self.evaluation
        lock.unlock()
        if StreamTrace.enabled {
            // Was the visit read when the engine asked? Then it waits for nothing; else it waits for the rest of the read.
            let ready = entry.group.wait(timeout: .now()) == .success
            if !ready { entry.group.wait() }
            StreamTrace.shared.consume(name: name, visit: b, evaluation: evaluation, asked: asked, ready: StreamTrace.now(),
                                       readBefore: ready)
        } else {
            entry.group.wait()
        }
        return UnsafeRawPointer(slots[entry.slot]) + place
    }

    /// The engine is done with these names: a visit whose uses are all released frees its buffer — and, in a
    /// sequence, the next visit is read at once.
    /// Returns `true` if every name was staged (nothing left for the map to drop).
    func release(_ names: [String]) -> Bool {
        lock.lock(); defer { lock.unlock() }
        var all = true
        for name in names {
            guard visitsOf[name] != nil else { all = false; continue }
            if StreamTrace.enabled { StreamTrace.shared.release(name: name, at: StreamTrace.now()) }
            guard let b = visit(of: name), state[b] == .staged, left[b][name, default: 0] > 0 else {
                // Released already (a layer's last call after its tensors' own): nothing to do.
                if !done.contains(name) { all = false }
                continue
            }
            left[b][name]! -= 1
            done.insert(name)
            if left[b].values.allSatisfy({ $0 == 0 }) {
                let entry = staged.removeValue(forKey: b)!
                state[b] = .complete
                entry.group.wait()
                free.append(entry.slot)
                while state[frontier] == .complete {
                    frontier = (frontier + 1) % visits.count
                    // A whole evaluation is complete: the next one starts from the first visit.
                    if frontier == 0 { evaluation += 1; for v in visits.indices where state[v] == .complete { state[v] = .pending } }
                }
                if sequential { advance() }
            }
        }
        return all
    }

    /// Whether the name belongs to the streamed tail (staged or not).
    package func contains(_ name: String) -> Bool { visitsOf[name] != nil }
}


/// **What the weight stream did, instant by instant** — for the disk's question: does the engine wait for pages, and where?
/// Off unless a tool turns it on (`enabled`); the engine pays one branch per tensor. Not a policy and not an input of the plan.
///
///     read      a visit's bytes, from the first `pread` issued to the last one back
///     consume   the engine asks for a tensor (`asked`), gets it (`ready`), has copied it and gives its buffer back (`released`)
///     waited = ready − asked          the engine (hence the GPU, which the engine feeds) stood still for the disk
///     copied = released − ready       the staging copy, also before the product, also with the GPU idle
package final class StreamTrace: @unchecked Sendable {
    package static let shared = StreamTrace()
    nonisolated(unsafe) package static var enabled = false
    package static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

    package struct Read: Sendable { package var visit: Int, evaluation: Int, bytes: Int, started: UInt64, ended: UInt64 }
    package struct Consume: Sendable {
        package var name: String, visit: Int, evaluation: Int
        package var asked: UInt64, ready: UInt64, released: UInt64 = 0
        /// The visit was entirely read when the engine asked.
        package var readBefore: Bool
        package var waited: Double { Double(ready &- asked) * 1e-9 }
        package var copied: Double { released >= ready ? Double(released - ready) * 1e-9 : 0 }
    }
    private let lock = NSLock()
    package private(set) var reads: [Read] = []
    package private(set) var consumes: [Consume] = []
    private var open: [String: Int] = [:]

    package func reset() { lock.lock(); reads = []; consumes = []; open = [:]; lock.unlock() }
    func read(visit: Int, evaluation: Int, bytes: Int, started: UInt64, ended: UInt64) {
        lock.lock(); reads.append(Read(visit: visit, evaluation: evaluation, bytes: bytes, started: started, ended: ended)); lock.unlock()
    }
    func consume(name: String, visit: Int, evaluation: Int, asked: UInt64, ready: UInt64, readBefore: Bool) {
        lock.lock()
        open[name] = consumes.count
        consumes.append(Consume(name: name, visit: visit, evaluation: evaluation, asked: asked, ready: ready, readBefore: readBefore))
        lock.unlock()
    }
    func release(name: String, at time: UInt64) {
        lock.lock()
        if let i = open.removeValue(forKey: name) { consumes[i].released = time }
        lock.unlock()
    }
}
