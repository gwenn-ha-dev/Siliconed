import Darwin
import Foundation

/// What the engine reads of memory: the process's `phys_footprint` — what B bounds, what Activity Monitor calls Memory —
/// and the system's swap.
public enum Memory {
    /// The process's `phys_footprint` now.
    public static func footprint() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        precondition(result == KERN_SUCCESS, "task_info: \(result)")
        return Int(info.phys_footprint)
    }

    /// **The process's graphics ledger** (`ledger_tag_graphics_footprint`): what of the footprint Metal and its driver
    /// hold, its heaps and buffers and what no allocation holds (the driver's memory for the command buffers).
    public static func graphics() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        precondition(result == KERN_SUCCESS, "task_info: \(result)")
        return Int(info.ledger_tag_graphics_footprint)
    }

    /// Pages swapped out since boot (`vm_stat`'s "Swapouts").
    public static func swapouts() -> UInt64 {
        var s = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &s) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
        }
        return status == KERN_SUCCESS ? s.swapouts : 0
    }

    /// **What a process may wire** (`mlock`), in bytes: the lower of the system's limits on one user's wired memory and
    /// on all users' (`vm.user_wire_limit`, `vm.global_user_wire_limit`: 12.2 GiB on a 16 GB M1 Pro, read 20/11).
    public static let wireLimit: Int = {
        func read(_ name: String) -> Int {
            var v: UInt64 = 0, size = MemoryLayout<UInt64>.size
            return sysctlbyname(name, &v, &size, nil, 0) == 0 ? Int(v) : 0
        }
        return min(read("vm.user_wire_limit"), read("vm.global_user_wire_limit"))
    }()

    /// **What `free` left billed, handed back to the kernel now**: the bytes it gave back (0 where the system has no such
    /// buffer). On macOS 27 `malloc` does not unmap a freed page: it queues it in the task's deferred reclamation buffer,
    /// billed in `phys_footprint` until the kernel drains it — under pressure, else never. Measured 21/11/2026 on the M5
    /// Pro, a block of 15 groups run three times in one process: 34 to 38 MiB that its stages and its client had freed were
    /// still billed when the next one read its floor (the floors 0.05, 0.08, 0.08 GiB; drained before each, 0.05 each).
    /// What fills it is the frameworks' and the client's, never the arena's (Metal's heaps, unmapped at once).
    /// `mach_vm_deferred_reclamation_buffer_flush` is the kernel's own drain, which `libmalloc` calls when the buffer is
    /// full: the entries `malloc` has marked reclaimable are unmapped, as under pressure, nothing live touched. Not in a
    /// public header: looked up by name (the former engine's `Arena.returnFreedPages`, 0.21); absent, nothing.
    @discardableResult
    public static func returnFreed() -> Int {
        guard let flush = reclamationFlush else { return 0 }
        var bytes: UInt64 = 0, deadline: UInt64 = 0
        return flush(mach_task_self_, UInt32.max, &bytes, &deadline) == KERN_SUCCESS ? Int(bytes) : 0
    }

    private typealias Flush = @convention(c) (mach_port_t, UInt32, UnsafeMutablePointer<UInt64>, UnsafeMutablePointer<UInt64>) -> kern_return_t
    private static let reclamationFlush: Flush? = {
        guard let symbol = dlsym(dlopen(nil, RTLD_NOW), "mach_vm_deferred_reclamation_buffer_flush") else { return nil }
        return unsafeBitCast(symbol, to: Flush.self)
    }()

    /// **The footprint a stage plans over** (`Stage.plan`'s floor): what `free` left billed handed back first
    /// (`returnFreed`), so that no floor holds what was freed before it.
    public static func floor() -> Int {
        returnFreed()
        return footprint()
    }

    public static func mib(_ bytes: Int) -> String { String(format: "%.1f MiB", Double(bytes) / 1_048_576) }
}

/// **The one thread that reads the footprint, every millisecond, for every watch of the process**: a reading costs
/// 0.64 µs and a thread waking 790 times a second 0.5 % of a core (M1 Pro), and a render had up to three of them — the
/// fuse's, the stage's, its caller's — reading the same footprint. It runs while a watch is registered and
/// ends with the last; the fuse's watches are read first, a tick that reads the others never delays them.
final class FootprintTicker: @unchecked Sendable {
    static let shared = FootprintTicker()
    private let lock = NSLock()
    private var watches: [ObjectIdentifier: FootprintWatch] = [:]
    private var threaded = false

    func add(_ watch: FootprintWatch) {
        lock.lock(); defer { lock.unlock() }
        watches[ObjectIdentifier(watch)] = watch
        guard !threaded else { return }
        threaded = true
        let thread = Thread { [self] in loop() }
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    func remove(_ watch: FootprintWatch) { lock.lock(); watches[ObjectIdentifier(watch)] = nil; lock.unlock() }

    private func loop() {
        while true {
            lock.lock()
            if watches.isEmpty { threaded = false; lock.unlock(); return }
            let all = watches.values.sorted { ($0.limit != nil ? 0 : 1) < ($1.limit != nil ? 0 : 1) }
            lock.unlock()
            for w in all { w.sample() }
            usleep(1000)
        }
    }
}

/// The footprint's peak — or another reading's (`reading`: a stage's share) —, read every millisecond by the process's
/// one ticker (`FootprintTicker`), each watch keeping its own peak.
public final class FootprintWatch: @unchecked Sendable {
    private let lock = NSLock()
    private var highest = 0
    private let reading: @Sendable () -> Int
    fileprivate let limit: Int?

    /// `stopPast`: past that footprint, the process says so and exits (status 3) — a measure that would pass B.
    public init(stopPast limit: Int? = nil, reading: @escaping @Sendable () -> Int = Memory.footprint) {
        self.reading = reading
        self.limit = limit
        highest = reading()
        // The ticker holds the watch until `stop`: its last reading may come after its owner let go of it.
        FootprintTicker.shared.add(self)
    }

    fileprivate func sample() {
        let f = reading()
        if let limit, f > limit {
            // What the open stages ran: never taken again on this system (`ThisMac.readings`), and said.
            let open = Stage.fused()
            FileHandle.standardError.write(Data(("SiliCore: the footprint \(Memory.mib(f)) past \(Memory.mib(limit)): stopped"
                + (open.isEmpty ? "" : ", under " + open.joined(separator: "; ")) + "\n").utf8))
            exit(3)
        }
        lock.lock(); highest = max(highest, f); lock.unlock()
    }

    /// The peak since the last reset, a last reading included.
    public var peak: Int {
        let f = reading()
        lock.lock(); defer { lock.unlock() }
        highest = max(highest, f)
        return highest
    }

    public func reset() { let f = reading(); lock.lock(); highest = f; lock.unlock() }

    public func stop() { FootprintTicker.shared.remove(self) }
}

