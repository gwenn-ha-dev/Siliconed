import Darwin
import Foundation
import Synchronization

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
    package static let defaultThreads = 4

    private let artifact: Artifact
    private let threads: Int
    private let queue = DispatchQueue(label: "siliconed.prefetch", attributes: .concurrent)
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
        for name in wanted where !staged.contains(name) {
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

    /// Waits until everything that was requested is paged in. The conductor does not call it: it
    /// requests ahead and lets the computation catch up. It is for measurements.
    package func drain() { group.wait() }

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
/// is free waits for the next request; read with no slot free at all, it goes through the map
/// (`fallbacks` counts those reads): slower, never wrong.
package final class TailStream {
    package let blocks: [[String]]
    /// Tensor reads that found no staged copy and no free slot, and went through the map.
    package private(set) var fallbacks = 0
    package let slotBytes: Int
    /// The staging buffers allocated.
    package var slotCount: Int { slots.count }
    /// Measurement: seconds the engine waited for a staged block, bytes `pread`, seconds of reads (summed over lanes).
    package private(set) var waited = 0.0
    package final class Counters: Sendable {
        package let bytes = Atomic<Int>(0), nanoseconds = Atomic<Int>(0)
    }
    package let counters = Counters()

    private struct Staged { let slot: Int; let group: DispatchGroup }
    private let descriptor: Int32
    private let slots: [UnsafeMutableRawPointer]
    private var free: [Int]
    private var staged: [String: Staged] = [:]
    private var held: [Int: Int] = [:]                      // slot → names not yet released
    private let blockOf: [String: Int]
    private let spans: [(start: Int, end: Int)]             // per block, in the file
    private let mapped: Int                                 // the map's base address, for the fallback copy
    private let lock = NSLock()
    private let queue = DispatchQueue(label: "siliconed.tail", attributes: .concurrent)
    private let reads = DispatchGroup()
    /// Measured: four lanes, slices of 64 MiB.
    private static let lanes = 4, piece = 64 << 20

    init(artifact: Artifact, blocks: [[String]], slots count: Int) throws {
        self.blocks = blocks
        let page = artifact.page
        var blockOf: [String: Int] = [:], spans: [(start: Int, end: Int)] = []
        for (b, names) in blocks.enumerated() {
            let tensors = names.compactMap { artifact.tensors[$0] }
            guard tensors.count == names.count, !tensors.isEmpty else {
                throw Artifact.Failure.badHeader("streamed block \(b): tensor missing from the map")
            }
            let start = tensors.map(\.offset).min()!
            let end = min((tensors.map { $0.offset + $0.bytes }.max()! + page - 1) / page * page, artifact.size)
            spans.append((start, end))
            for name in names { blockOf[name] = b }
        }
        self.blockOf = blockOf
        self.spans = spans
        let slotBytes = spans.map { $0.end - $0.start }.max()!
        self.slotBytes = slotBytes
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
        close(descriptor)
        for slot in slots { munmap(slot, slotBytes) }
    }

    /// Starts reading the blocks these names belong to, without waiting.
    ///
    /// A block that finds no free slot is **not** paged in instead: it is staged by a later request
    /// (the next layer's, once a slot is released) or, at the latest, when the engine reads it.
    /// So the returned set is every streamed name, staged or deferred — none of them is the map's.
    func stage(_ names: [String]) -> Set<String> {
        lock.lock(); defer { lock.unlock() }
        var out: Set<String> = []
        for name in names where blockOf[name] != nil {
            out.insert(name)
            if staged[name] == nil, let slot = free.popLast() { start(block: blockOf[name]!, in: slot) }
        }
        return out
    }

    /// Under the lock: the block's reads, in `lanes` page-aligned pieces.
    private func start(block b: Int, in slot: Int) {
        let group = DispatchGroup()
        let (start, end) = spans[b], base = Int(bitPattern: slots[slot])
        let fd = descriptor, mapped = self.mapped
        let lane = ((end - start) / Self.lanes + (1 << 14) - 1) / (1 << 14) * (1 << 14)
        var first = start
        while first < end {
            let last = min(first + lane, end), from = first
            let counters = self.counters
            queue.async(group: group) {
                let t0 = DispatchTime.now().uptimeNanoseconds
                defer {
                    counters.bytes.wrappingAdd(last - from, ordering: .relaxed)
                    counters.nanoseconds.wrappingAdd(Int(DispatchTime.now().uptimeNanoseconds - t0), ordering: .relaxed)
                }
                var offset = from
                while offset < last {
                    let length = min(Self.piece, last - offset)
                    let destination = UnsafeMutableRawPointer(bitPattern: base + offset - start)!
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
        reads.enter()
        group.notify(queue: queue) { [reads] in reads.leave() }
        for member in blocks[b] { staged[member] = Staged(slot: slot, group: group) }
        held[slot] = blocks[b].count
    }

    /// The staged copy of a tensor once its block is read, or `nil` if it is not staged.
    func pointer(_ name: String, offset: Int) -> UnsafeRawPointer? {
        guard let b = blockOf[name] else { return nil }
        lock.lock()
        if staged[name] == nil {
            // Not requested ahead (or no slot was free then): staged now, or read from the map.
            if let slot = free.popLast() { start(block: b, in: slot) } else { fallbacks += 1 }
        }
        let entry = staged[name]
        lock.unlock()
        guard let entry else { return nil }
        let t0 = Date()
        entry.group.wait()
        waited += Date().timeIntervalSince(t0)
        return UnsafeRawPointer(slots[entry.slot]) + (offset - spans[b].start)
    }

    /// The engine is done with these names: a slot whose names are all released is free again.
    /// Returns `true` if every name was staged (nothing left for the map to drop).
    func release(_ names: [String]) -> Bool {
        lock.lock(); defer { lock.unlock() }
        var all = true
        for name in names {
            guard let entry = staged.removeValue(forKey: name) else { all = false; continue }
            held[entry.slot, default: 1] -= 1
            if held[entry.slot] == 0 {
                held[entry.slot] = nil
                entry.group.wait()
                free.append(entry.slot)
            }
        }
        return all
    }

    /// Whether the name belongs to the streamed tail (staged or not).
    package func contains(_ name: String) -> Bool { blockOf[name] != nil }
}
