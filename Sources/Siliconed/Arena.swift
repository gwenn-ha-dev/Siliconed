import Darwin
import Foundation
import Synchronization

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
        case allocationFailed(Int)

        package var description: String {
            switch self {
            case .exhausted(let name, let wanted, let free, let capacity):
                return "arena exhausted: \(name) wants \(wanted) bytes, \(free) left out of \(capacity). "
                     + "Size at startup, do not grow."
            case .duplicate(let name): return "slice already reserved: \(name)"
            case .allocationFailed(let bytes): return "cannot allocate \(bytes) bytes"
            }
        }
    }

    package static let alignment = 16384

    package let capacity: Int
    package private(set) var slots: [String: Slot] = [:]
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
    /// "under pressure" is exactly the situation that criterion 4 forbids — we do not want
    /// room to appear *because* memory runs short.
    ///
    /// Anonymous `mmap` makes release deterministic: `munmap` unmaps, the footprint drops
    /// immediately. And the alignment comes with it — a page is 16 KiB on this machine, which is
    /// exactly `Arena.alignment`.
    package init(capacity: Int) throws {
        let rounded = (capacity + Self.alignment - 1) / Self.alignment * Self.alignment
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
        Arena.liveCount.add(1, ordering: .relaxed)
    }

    /// **How many arenas are alive.** An arena that outlives its phase shows up nowhere
    /// else: the footprint does not distinguish "not yet given back to the system" from "still
    /// referenced", and the two are fixed in opposite ways. This counter settles it.
    /// Atomic: the arena is born and dies on the engine's queue, but the counter is read from elsewhere
    /// (the app, the cancellation check) — and a bare integer read from another thread is a race.
    package static var live: Int { liveCount.load(ordering: .relaxed) }
    private static let liveCount = Atomic<Int>(0)

    deinit {
        munmap(base, capacity)
        Arena.liveCount.subtract(1, ordering: .relaxed)
    }

    /// **Give back to the system what `free` only gave back to the allocator.**
    ///
    /// The arenas map and unmap their own pages (see `init`), but everything else a phase allocates
    /// — Swift arrays, the frameworks' buffers — goes through `malloc`, and `free` does not unmap a
    /// large block: macOS's malloc magazine keeps it in its large-block cache, for the next request
    /// of the same size. The pages therefore stay counted in `phys_footprint` — and that is
    /// exactly what we measure when we ask whether room exists for the next phase.
    ///
    /// The symptom, measured at 512² when the arenas themselves still came from `posix_memalign`:
    /// the denoising peak was 1,137 MB, and the residue **after** the arena was released 1,137 MB
    /// too. Not a byte given back.
    ///
    /// This is not an arena defect nor a leak: under pressure, the system reclaims these
    /// pages. But "under pressure" is precisely the situation that criterion 4 forbids — we do not
    /// want room to appear *because* memory runs short. An explicit call at the end
    /// of a phase costs a few milliseconds and makes the measurement honest.
    @discardableResult
    package static func releaseToSystem() -> Int {
        Int(malloc_zone_pressure_relief(nil, 0))
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

    package var used: Int { cursor }

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
}
