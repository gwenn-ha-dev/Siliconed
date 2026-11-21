import Darwin
import Foundation

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
/// buffer whose size follows the shapes lives here, never in a Swift array or
/// `UnsafeMutablePointer.allocate`, and its size comes from a static function of the shapes that a
/// memory need can call.
public final class Pages: @unchecked Sendable {
    public let base: UnsafeMutableRawPointer
    /// What the region maps: `bytes` rounded up to the page, at least a page.
    public let capacity: Int

    /// Traps when the kernel refuses the region, as `UnsafeMutablePointer.allocate` does: a refused
    /// address range is not a state the engine can compute through.
    public init(bytes: Int) {
        let rounded = Pages.capacity(for: bytes)
        var address: mach_vm_address_t = 0
        let result = mach_vm_allocate(mach_task_self_, &address, mach_vm_size_t(rounded), VM_FLAGS_ANYWHERE)
        guard result == KERN_SUCCESS, let base = UnsafeMutableRawPointer(bitPattern: UInt(address)) else {
            fatalError("Pages: mach_vm_allocate of \(rounded) bytes refused (\(result))")
        }
        self.base = base
        self.capacity = rounded
    }

    /// What a region of `bytes` maps: rounded up to the page, at least a page.
    public static func capacity(for bytes: Int) -> Int {
        let page = Int(getpagesize())
        return (max(1, bytes) + page - 1) / page * page
    }

    deinit {
        mach_vm_deallocate(mach_task_self_, mach_vm_address_t(UInt(bitPattern: base)), mach_vm_size_t(capacity))
    }

    public func floats(at byteOffset: Int = 0) -> UnsafeMutablePointer<Float> {
        (base + byteOffset).assumingMemoryBound(to: Float.self)
    }
}

