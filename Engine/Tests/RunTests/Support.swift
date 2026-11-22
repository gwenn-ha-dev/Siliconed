import CoreAI
import Foundation
import Metal
import SiliCore
import Testing

// What the run tests share. They run on this Mac's GPU and ANE through Core AI, so they are off unless asked for:
//
//     SILICORE_METAL=1 swift test -c release --filter RunTests
//
// In release (in debug the CPU references crawl). They import the engine as a client does, without `@testable`: what
// they reach is the public API a model is written with. Their assets go under `.build/run/`.

let metalEnabled = ProcessInfo.processInfo.environment["SILICORE_METAL"] != nil

/// **Every run test, one at a time**: the suite they all extend, serial by construction. Two tests side by side would
/// share the GPU and the footprint, and a fact of memory or time read by one would be the other's.
@Suite(.serialized) struct Runs {}

/// The device, a stream on a queue of its own, and the engine's assets under `.build/run/`; the process's first call of
/// Core AI paid before any test reads a floor (`Functions.witness`), as a client's process pays it at its start.
struct Bench {
    let device: MTLDevice
    let stream: ComputeStream

    init() async throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw Failure("no Metal device") }
        self.device = device
        stream = ComputeStream(commandQueue: queue)
        let folder = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("../../.build/run").standardizedFileURL
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        Functions.folder = folder
        try await Functions.witness(device)
    }
}

/// The same values from one run to the next: a linear congruence, uniform in `range`.
struct Seeded {
    var state: UInt64
    mutating func next() -> Double {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Double(state >> 11) / Double(1 << 53)
    }
    mutating func uniform(_ n: Int, _ range: ClosedRange<Double> = -1...1) -> [Float] {
        (0..<n).map { _ in Float(range.lowerBound + (range.upperBound - range.lowerBound) * next()) }
    }
}

extension Arena {
    /// `values` rounded to f16 into the tensor at `key`; what was written, read back as f32.
    @discardableResult
    func write(_ values: [Float], at key: Int) -> [Float] {
        let p = pointer(key).bindMemory(to: Float16.self, capacity: values.count)
        for i in values.indices { p[i] = Float16(values[i]) }
        return (0..<values.count).map { Float(p[$0]) }
    }

    func write(codes: [Int8], at key: Int) {
        codes.withUnsafeBytes { pointer(key).copyMemory(from: $0.baseAddress!, byteCount: codes.count) }
    }

    func halves(_ key: Int, _ count: Int) -> [Float] {
        let p = pointer(key).bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(p[$0]) }
    }
}

/// ‖got − want‖ / ‖want‖.
func relative(_ got: [Float], _ want: [Float]) -> Double {
    var e = 0.0, n = 0.0
    for i in want.indices { let d = Double(got[i]) - Double(want[i]); e += d * d; n += Double(want[i]) * Double(want[i]) }
    return (e / max(n, 1e-30)).squareRoot()
}

/// `x [rows, k] · w [k, n]` in f64 on the CPU, w read as `w(i, o)`.
func product(_ x: [Float], rows: Int, k: Int, n: Int, _ w: (Int, Int) -> Double) -> [Float] {
    var y = [Float](repeating: 0, count: rows * n)
    for o in 0..<n {
        let column = (0..<k).map { w($0, o) }
        for r in 0..<rows {
            var acc = 0.0
            for i in 0..<k { acc += Double(x[r * k + i]) * column[i] }
            y[r * n + o] = Float(acc)
        }
    }
    return y
}

/// The median of `calls` calls' seconds, the first two dropped (the first specializes and binds).
func median(_ calls: Int, _ call: () async throws -> Void) async throws -> Double {
    var times: [Double] = []
    for _ in 0..<calls + 2 {
        let t = DispatchTime.now().uptimeNanoseconds
        try await call()
        times.append(Functions.seconds(since: t))
    }
    let kept = times.dropFirst(2).sorted()
    return kept[kept.count / 2]
}

/// What a call adds to the footprint at its peak, over the settled footprint before it: after two calls, so that what
/// Core AI keeps from one call to the next (its specialization, its own buffers) is in the footprint before.
func share(_ call: () async throws -> Void) async throws -> Int {
    try await call(); try await call()
    let watch = FootprintWatch()
    defer { watch.stop() }
    let before = Memory.settled(50)
    watch.reset()
    try await call()
    return watch.peak - before
}

extension Weights {
    /// `entries` read from `file` into `arena` as a stage reads them (`Weights.place`, then `Weights.read` through a
    /// staging), the staging in `Pages` of its own.
    static func load(_ entries: [Entry], from file: SafeTensors, into arena: Arena, codes: CodesLayout = .inOut) throws -> [String: Placed] {
        let placed = place(try layout(entries, in: file, codes: codes), in: arena)
        let staging = Pages(bytes: try Self.staging(entries, in: file, codes: codes))
        try withExtendedLifetime(staging) {
            try read(entries, from: file, into: placed, arena: arena, staging: staging.base, codes: codes)
        }
        return placed
    }
}

extension Memory {
    /// The footprint, settled: the lowest of `samples` readings a millisecond apart (what was given back is gone).
    static func settled(_ samples: Int = 200) -> Int {
        var low = Int.max
        for _ in 0..<samples { low = min(low, footprint()); usleep(1000) }
        return low
    }
}

extension Stage.Settled {
    /// The stage let go whole: the footprint back within the stage's margin of what it was before, no arena alive, no
    /// heap held.
    var within: Bool { footprint <= before + Stage.margin && alive == 0 && held == 0 }
}
