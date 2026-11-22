import CoreAI
import Foundation

/// **What this Mac measured, kept** (`this-mac.json`, beside `Functions.folder`): what no query gives and no paper
/// predicts, measured once here and read by every plan after.
///
///     flight    a block's ANE share in flight, in rows (`flight`: `Block`), where the next block of the same key starts
///     neuralUnits  a block's ANE unit, decided once by the first block of its key that ran long enough to try both
///               (`Block`), never changed after
///     paces     a block's seconds a cycle at its kept unit, the median of its life's (`paces`: `Block`), what a client's
///               choice among declarations of one computation reads (`Block.choose`)
///     reserves  what a stage's functions took past its first signal, at its largest reading (`Stage.signal`)
///     readings  Core AI's share under a stage's functions, its largest reading — the hint a height is chosen on
///               (`Stage.estimates`); −1, the fuse stopped the process under them, a mark that holds for the engine
///               that took it (`fusedBy`)
///     extents   what a stage's functions held beside its reading, the same names at any shapes (`Stage.Extent`): a
///               form never read is bounded from its kin's readings (`Stage.estimates(extents:)`)
///
/// **By Mac and by system**: Core AI's kernels and Metal's change with the OS, so what was measured is kept under the
/// Mac's model and the system's build (`system`), and measured again past an update; the file keeps every system's.
/// **A key carries the weights' identity**: a function's is its asset's name, which its client derives from what the
/// asset computes (its shapes); a block's ANE share and unit, the block's key, likewise.
public struct ThisMac: Codable, Sendable, Equatable {
    /// A block's ANE share in flight, its rows, by its key (`Block`); nil in a file written before.
    public var flight: [String: Int]? = nil
    /// A block's ANE unit, its rows, by its key: decided once and kept (`Block.settle`). A field of its own: the units an
    /// earlier rule kept (`units`, some never measured) are not read.
    public var neuralUnits: [String: Int]? = nil
    /// A block's seconds a cycle — between two starts of its cycle over one pass, the first cycle's left out (its
    /// functions specialized) — the median of its last block's, by its pace key (`Block.paceKey`) and, where it shares
    /// the ANE's rows, the unit it was taken at; nil in a file written before.
    public var paces: [String: Double]? = nil
    /// What Core AI's share grew past a stage's first signal, its largest reading, by the stage's key (`Stage.key`); nil in a
    /// file written before.
    public var reserves: [String: Int]? = nil
    /// Core AI's share read under a stage's functions, its largest, by the stage's key (`Stage.key`); −1: the fuse stopped
    /// the process under them, and this engine does not take them again on this system (`reading`). Nil in a file written
    /// before.
    public var readings: [String: Int]? = nil
    /// **The engine that marked a stage's functions −1**, by the stage's key: its binary's (`engine`). A mark holds for the
    /// engine that took it alone: a new one tries the functions once more, on every Mac — what stopped the fuse may have
    /// been the engine's own fault, since corrected (decided 21/11/2026, after a mark of the M1 Pro left a block's
    /// functions 2 % slower for good). A −1 with no engine beside it (a file written before) is another engine's. Nil in a file
    /// written before.
    public var fusedBy: [String: String]? = nil
    /// What the functions of a stage's reading held, by the stage's key (`Stage.Extent`); nil in a file written before.
    public var extents: [String: Stage.Extent]? = nil

    public init() {}

    /// The key of a function's share: its unit and its asset's name.
    public static func shareKey(asset: String, unit: ComputeUnitKind) -> String { (unit == .neuralEngine ? "ane " : "gpu ") + asset }

    // MARK: - The readings and the fuse's marks

    /// **The reading under `key` as `engine` takes it**: a −1 its own (`fusedBy`); another engine's −1 nil, as never read,
    /// so that the functions are tried once more.
    public func reading(_ key: String, engine: String = ThisMac.engine) -> Int? {
        guard let r = readings?[key] else { return nil }
        return r == -1 && fusedBy?[key] != engine ? nil : r
    }

    /// **A reading kept at its largest** (`Stage.keep`); over `engine`'s own −1 nothing, over another's −1 the reading,
    /// the mark gone.
    public mutating func keep(reading share: Int, under key: String, engine: String = ThisMac.engine) {
        let old = reading(key, engine: engine)
        if old == -1 { return }
        readings = (readings ?? [:]).merging([key: max(old ?? 0, max(share, 0))]) { $1 }
        fusedBy?[key] = nil
    }

    /// **The fuse stopped the process under `key`'s functions** (`Stage.fused`): −1, `engine`'s mark.
    public mutating func fuse(_ key: String, engine: String = ThisMac.engine) {
        readings = (readings ?? [:]).merging([key: -1]) { $1 }
        fusedBy = (fusedBy ?? [:]).merging([key: engine]) { $1 }
    }

    /// **This engine, as a mark of the fuse names it**: the UUID of the binary its code runs in (`LC_UUID`, read from the
    /// Mach-O header of its own code in memory). Every build is a binary of its own: an engine changed is a UUID changed,
    /// whatever the version its client gives itself (two builds of one number differ, 20/11). `""` where it has none.
    public static let engine: String = {
        let header = #dsohandle
        // mach_header_64: magic, cputype, cpusubtype, filetype, ncmds (16), sizeofcmds, flags, reserved; then the commands.
        guard header.load(as: UInt32.self) == 0xFEED_FACF else { return "" }
        let count = Int(header.load(fromByteOffset: 16, as: UInt32.self))
        var at = 32
        for _ in 0..<count {
            let (command, size) = (header.load(fromByteOffset: at, as: UInt32.self), Int(header.load(fromByteOffset: at + 4, as: UInt32.self)))
            if command == 0x1B {  // LC_UUID
                return (0..<16).map { String(format: "%02X", header.load(fromByteOffset: at + 8 + $0, as: UInt8.self)) }.joined()
            }
            guard size > 0 else { break }
            at += size
        }
        return ""
    }()

    // MARK: - The store

    /// Where it is kept: the client's to choose; nil, `this-mac.json` beside `Functions.folder`.
    nonisolated(unsafe) public static var file: URL? = nil
    public static var url: URL { file ?? Functions.folder.deletingLastPathComponent().appendingPathComponent("this-mac.json") }

    /// The Mac's model and the system's version and build: what a measure holds for.
    public static let system: String = {
        func sysctl(_ name: String) -> String {
            var size = 0
            sysctlbyname(name, nil, &size, nil, 0)
            var bytes = [CChar](repeating: 0, count: max(size, 1))
            sysctlbyname(name, &bytes, &size, nil, 0)
            return String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        }
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "\(sysctl("hw.model")) macOS \(v.majorVersion).\(v.minorVersion).\(v.patchVersion) (\(sysctl("kern.osversion")))"
    }()

    private static func all() -> [String: ThisMac] {
        (try? JSONDecoder().decode([String: ThisMac].self, from: Data(contentsOf: url))) ?? [:]
    }

    /// What this Mac measured under this system.
    public static var current: ThisMac { all()[system] ?? ThisMac() }

    /// `change` applied to what this system kept, written at once (the other systems' kept as they are): the file read,
    /// changed and written back under an exclusive `flock` on `<file>.lock`, the write atomic (a file beside, renamed).
    /// Two processes measure on one Mac (a client's worker, a command line, a test): each writes what it read back with its
    /// own change, and without the lock one's measures were lost to the other's write (19/11: a client's shares gone).
    /// A lock of its own on each opening, so the threads of one process exclude each other the same way.
    public static func update(_ change: (inout ThisMac) -> Void) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let fd = open(url.path + ".lock", O_RDWR | O_CREAT, 0o644)
        guard fd >= 0 else { throw Failure("\(url.path).lock: cannot open") }
        defer { close(fd) }
        guard flock(fd, LOCK_EX) == 0 else { throw Failure("\(url.path).lock: cannot lock") }
        defer { flock(fd, LOCK_UN) }
        var every = all()
        var mine = every[system] ?? ThisMac()
        change(&mine)
        every[system] = mine
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(every).write(to: url, options: .atomic)
    }
}
