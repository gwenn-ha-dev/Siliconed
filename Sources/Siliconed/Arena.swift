import Darwin
import Foundation
import Metal
import Synchronization

/// **Pages of our own, outside `malloc`, given back the moment their owner dies.**
///
/// A block `free`d is not a block the system has back. On macOS 27 every `free` above the tiny and
/// small sizes — a 64 KB block as well as a 64 MB one — leaves its pages dirty and counted in
/// `phys_footprint` (the task's deferred reclamation buffer, measured: 200 MB freed, 200 MB still
/// billed, at every size from 64 KB to 64 MB), until the kernel needs them, which is to say under
/// pressure; `malloc_zone_pressure_relief` returns 0 there. Older systems kept them in malloc's
/// large-block cache instead. Either way, what a phase allocated through `malloc` outlived it.
///
/// Nothing is asked of the allocator here: the region is anonymous memory mapped for this owner alone
/// (`mach_vm_allocate`, zero-filled, page-aligned) and unmapped by `deinit` (`mach_vm_deallocate`),
/// which the kernel honours at once on every macOS — the footprint drops by what was touched. A
/// buffer whose size follows the shapes lives here or in an `Arena`, never in a Swift array or
/// `UnsafeMutablePointer.allocate`, and its size comes from a static function of the shapes that the
/// memory need can call.
package final class Pages {
    package let base: UnsafeMutableRawPointer
    /// What the region maps: `bytes` rounded up to the page (`Arena.capacity(for:)`), at least a page.
    package let capacity: Int

    /// Traps when the kernel refuses the region, as `UnsafeMutablePointer.allocate` does: a refused
    /// address range is not a state the engine can compute through.
    package init(bytes: Int) {
        let rounded = Arena.capacity(for: max(1, bytes))
        var address: mach_vm_address_t = 0
        let result = mach_vm_allocate(mach_task_self_, &address, mach_vm_size_t(rounded), VM_FLAGS_ANYWHERE)
        guard result == KERN_SUCCESS, let base = UnsafeMutableRawPointer(bitPattern: UInt(address)) else {
            fatalError("Pages: mach_vm_allocate of \(rounded) bytes refused (\(result))")
        }
        self.base = base
        self.capacity = rounded
    }

    deinit {
        mach_vm_deallocate(mach_task_self_, mach_vm_address_t(UInt(bitPattern: base)), mach_vm_size_t(capacity))
    }

    package func floats(at byteOffset: Int = 0) -> UnsafeMutablePointer<Float> {
        (base + byteOffset).assumingMemoryBound(to: Float.self)
    }
}

/// The process's only anonymous memory, and it is of fixed size.
///
/// The "zero swap" guarantee comes down to one verifiable sentence: the map is
/// a read-only mapped file — the kernel can only take its pages back by dropping them — and
/// **everything else is here**, in a single allocation sized at startup and never
/// grown. If there is no allocator, there is nothing to watch.
///
/// It hands out named slices, aligned on a page, and it refuses to grow. A
/// request that does not fit is a sizing error, not an opportunity for `realloc`.
package final class Arena {
    package struct Slot {
        package let name: String
        package let offset: Int
        package let bytes: Int
    }

    package enum Failure: Error, CustomStringConvertible {
        case exhausted(name: String, wanted: Int, free: Int, capacity: Int)
        case duplicate(String)
        case unknown(String)
        case allocationFailed(Int)

        package var description: String {
            switch self {
            case .exhausted(let name, let wanted, let free, let capacity):
                return "arena exhausted: \(name) wants \(wanted) bytes, \(free) left out of \(capacity). "
                     + "Size at startup, do not grow."
            case .duplicate(let name): return "slice already reserved: \(name)"
            case .unknown(let name): return "no slice named \(name)"
            case .allocationFailed(let bytes): return "cannot allocate \(bytes) bytes"
            }
        }
    }

    package static let alignment = 16384

    /// **What an arena asked for `bytes` holds**: rounded up to the page — what `init` maps, and what the
    /// memory need charges (`MemoryNeed`), from this one function.
    package static func capacity(for bytes: Int) -> Int {
        (bytes + alignment - 1) / alignment * alignment
    }

    /// **The arena that holds exactly these slices, reserved in order**: each starts on a page
    /// (`reserve`), so the arena is the sum of the slices rounded to the page — not a byte of margin.
    /// A reservation never written is still charged by the memory need but absent from
    /// `phys_footprint` (an untouched page is not resident): an arena sized here holds only what its
    /// owner writes.
    package static func capacity(slices bytes: [Int]) -> Int {
        bytes.reduce(0) { $0 + capacity(for: $1) }
    }

    package let capacity: Int
    private var slots: [String: Slot] = [:]
    private let base: UnsafeMutableRawPointer
    private var cursor = 0

    /// **`mmap` and not `posix_memalign`, and a measurement decided it.**
    ///
    /// With `posix_memalign`, `free` gives the pages back to the *allocator*, not to the *system*: they
    /// stay resident and counted in `phys_footprint`. Measured at 512² — denoising peak
    /// 1,137 MB, arenas destroyed (`Arena.live == 0`), `malloc_zone_pressure_relief` called, and
    /// the footprint is still **1,137 MB**. Not a byte given back. The decoder that follows could not
    /// reuse them either: `MPSGraph` allocates through Metal, not malloc, and the two
    /// reserves do not talk to each other — hence a decoding peak that carried the residue on top.
    ///
    /// The system ends up reclaiming them **under pressure**, which explains why splitting the
    /// arenas by phase still gained 2.4 GB at 1024²: the pressure was real there. But
    /// "under pressure" is exactly the situation that B forbids — we do not want
    /// room to appear *because* memory runs short.
    ///
    /// Anonymous `mmap` makes release deterministic: `munmap` unmaps, the footprint drops
    /// immediately. And the alignment comes with it — a page is 16 KiB on this machine, which is
    /// exactly `Arena.alignment`.
    package init(capacity: Int) throws {
        let rounded = Arena.capacity(for: capacity)
        let allocated = mmap(nil, rounded, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANON, -1, 0)
        guard let allocated, allocated != MAP_FAILED else {
            throw Failure.allocationFailed(rounded)
        }
        guard Int(bitPattern: allocated) % Self.alignment == 0 else {
            munmap(allocated, rounded)
            throw Failure.allocationFailed(rounded)
        }
        self.base = allocated
        self.capacity = rounded
        owner = nil
        Arena.liveCount.add(1, ordering: .relaxed)
    }

    /// **How many arenas are alive.** An arena that outlives its phase shows up nowhere
    /// else: the footprint does not distinguish "not yet given back to the system" from "still
    /// referenced", and the two are fixed in opposite ways. This counter settles it.
    /// Atomic: the arena is born and dies on the engine's queue, but the counter is read from elsewhere
    /// (the app, the cancellation check) — and a bare integer read from another thread is a race.
    package static var live: Int { liveCount.load(ordering: .relaxed) }
    private static let liveCount = Atomic<Int>(0)

    /// **Another set of slices over the same memory**: for scratch that two users never hold at the same
    /// time (Z-Image's image refiners, caption refiners and layers). Its own slice table and cursor, the
    /// owner's pages — mapped once, unmapped with the owner, which the view keeps alive.
    package init(sharing owner: Arena) {
        base = owner.base
        capacity = owner.capacity
        self.owner = owner
    }
    private let owner: Arena?

    deinit {
        guard owner == nil else { return }
        munmap(base, capacity)
        Arena.liveCount.subtract(1, ordering: .relaxed)
    }

    @discardableResult
    package func reserve(_ name: String, bytes: Int) throws -> UnsafeMutableRawPointer {
        if slots[name] != nil { throw Failure.duplicate(name) }
        let start = (cursor + Self.alignment - 1) / Self.alignment * Self.alignment
        guard start + bytes <= capacity else {
            throw Failure.exhausted(name: name, wanted: bytes, free: capacity - start, capacity: capacity)
        }
        slots[name] = Slot(name: name, offset: start, bytes: bytes)
        cursor = start + bytes
        return base.advanced(by: start)
    }

    package func pointer(_ name: String) -> UnsafeMutableRawPointer? {
        guard let slot = slots[name] else { return nil }
        return base.advanced(by: slot.offset)
    }

    /// **A slice seen from the GPU, without a copy** — for a framework that wants an `MTLBuffer`
    /// (`MPSGraphTensorData`, an `MPSNDArray`): `makeBuffer(bytesNoCopy:)` over the slice, its length
    /// rounded to the page, which stays inside the arena (every slice starts on a page, the arena
    /// ends on one). The CPU and the GPU see the same bytes (unified memory, `storageModeShared`).
    /// No deallocator: the buffer must not outlive the arena — its owner keeps both.
    package func buffer(_ name: String, device: MTLDevice) throws -> MTLBuffer {
        guard let slot = slots[name] else { throw Failure.unknown(name) }
        let length = Arena.capacity(for: max(1, slot.bytes))
        guard let buffer = device.makeBuffer(bytesNoCopy: base.advanced(by: slot.offset), length: length,
                                             options: .storageModeShared, deallocator: nil) else {
            throw Failure.allocationFailed(length)
        }
        buffer.label = name
        MetalWraps.track(buffer)
        return buffer
    }

    /// What the kernel actually bills the process for, sampled. A peak that is not sampled
    /// does not exist on this side, and a peak read from an allocator says nothing about what swaps.
    package static func processFootprint() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.phys_footprint) : 0
    }

    /// **What `free` left billed, handed back to the kernel now** — the bytes it reclaimed (0 where
    /// the system has no such buffer).
    ///
    /// On macOS 27 `malloc` does not return a freed page: it queues it in the task's **deferred
    /// reclamation buffer** (`footprint`'s "Reclaimable", inside `phys_footprint`), which the kernel
    /// drains only under pressure — at rest, never (21 MB still billed 16 s after a render).
    /// What fills it is not ours — our buffers are `Arena`s and `Pages`, unmapped at once — but the
    /// frameworks': MPSGraph's compilation churn (mlir, ~100 graphs a decoder), Metal's objects. A
    /// phase then started beside the last one's freed pages, which no formula charges, and the
    /// first render of a process found its decoding phase 15–40 MB above its need.
    ///
    /// `mach_vm_deferred_reclamation_buffer_flush` is the kernel's own drain (the routine of
    /// `mach_vm.defs`, `libmalloc` itself calls it when the ring is full): the entries `malloc` has
    /// already marked reclaimable are unmapped, as under pressure — nothing live is touched, a block
    /// `malloc` takes back before the kernel is not in the ring. Not in a public header (`libsyscall`
    /// keeps it to itself): looked up by name, like `proc_reset_footprint_interval`; absent, nothing.
    @discardableResult
    package static func returnFreedPages() -> Int {
        guard let flush = deferredReclamationFlush else { return 0 }
        var bytes: UInt64 = 0, deadline: UInt64 = 0
        return flush(mach_task_self_, UInt32.max, &bytes, &deadline) == KERN_SUCCESS ? Int(bytes) : 0
    }

    private typealias Flush = @convention(c) (mach_port_t, UInt32, UnsafeMutablePointer<UInt64>,
                                              UnsafeMutablePointer<UInt64>) -> kern_return_t
    private static let deferredReclamationFlush: Flush? = {
        guard let symbol = dlsym(dlopen(nil, RTLD_NOW), "mach_vm_deferred_reclamation_buffer_flush") else { return nil }
        return unsafeBitCast(symbol, to: Flush.self)
    }()
}
