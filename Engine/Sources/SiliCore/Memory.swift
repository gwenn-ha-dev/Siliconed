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

    /// The footprint, settled: the lowest of `samples` readings a millisecond apart (what was given back is gone).
    public static func settled(_ samples: Int = 200) -> Int {
        var low = Int.max
        for _ in 0..<samples { low = min(low, footprint()); usleep(1000) }
        return low
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

