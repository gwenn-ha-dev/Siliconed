import CoreAI
import Foundation
import Metal
import Testing
@testable import SiliCore

/// **Calls encoded ahead, ordered by the arena's buffers**: what `Block` relies on to put a call on the stream while the
/// one before runs (`Run.encode`), each reading through a view of the arena what the one before wrote there. One run
/// test at a time (`Runs`): beside another, its 64 Mi values were the other's footprint.
extension Runs {
    /// Two functions in turn, x → y (`y = x + 1`) and y → x, 16 calls put on the stream unwaited but for each function's
    /// previous call (`Block`'s rule: one invocation of a function in flight), over 64 Mi values: exact, on the arena's
    /// tracked heap. The same chain on an untracked heap, said (what 18/11 saw: a call read its input before it was written).
    @Test(.enabled(if: metalEnabled))
    func callsEncodedAheadReadWhatTheCallBeforeWrote() async throws {
        let bench = try await Bench()
        let n = 64 << 20
        func function(_ name: String) async throws -> InferenceFunction {
            var m = Aicode.Module(arguments: [("x", Aicode.tensor([n], .f16))])
            let b = m.body
            m.outputs = [("y", b.binary(.add, m.argument("x"), b.scalar(1, .f16)))]
            return try await Functions.load(try Functions.asset(m, "run-ahead-\(name)"))
        }
        let f = try await function("f"), g = try await function("g")
        func chain(_ x: MTLBuffer, _ y: MTLBuffer) async throws -> [Float] {
            let p = x.contents().bindMemory(to: Float16.self, capacity: n)
            for i in 0..<n { p[i] = Float16(Float(i % 7)) }
            var last: [Int: Task<Void, Never>] = [:]
            nonisolated(unsafe) let stream = bench.stream
            let t0 = DispatchTime.now().uptimeNanoseconds
            for i in 0..<16 {
                let (fn, from, to) = i % 2 == 0 ? (f, x, y) : (g, y, x)
                await last[i % 2]?.value
                let left = try Run.encode(fn, bench.stream, inputs: ["x": View(buffer: from, type: .float16, shape: [n])],
                                          outputs: [("y", View(buffer: to, type: .float16, shape: [n]))])
                #expect(left.isEmpty)
                last[i % 2] = Task { await stream.currentWorkCompleted() }
            }
            let encoded = Functions.seconds(since: t0)
            await Run.finish(bench.stream)
            print("encode ahead: 16 calls encoded in \(String(format: "%.2f", encoded * 1000)) ms, done in \(String(format: "%.2f", Functions.seconds(since: t0) * 1000)) ms")
            return (0..<n).map { Float(p[$0]) }
        }
        let want = (0..<n).map { Float($0 % 7 + 16) }
        let arena = try Arena(bench.device, bytes: 2 * Arena.aligned(n * 2))
        let x = arena.take(n * 2), y = arena.take(n * 2)
        let got = try await chain(arena.buffer(x), arena.buffer(y))
        #expect(got == want, "first difference at \(zip(got, want).enumerated().first { $0.1.0 != $0.1.1 }?.offset ?? -1)")

        let d = MTLHeapDescriptor()
        (d.type, d.storageMode, d.hazardTrackingMode, d.size) = (.placement, .shared, .untracked, 2 * Arena.aligned(n * 2))
        let heap = bench.device.makeHeap(descriptor: d)!
        let ux = heap.makeBuffer(length: n * 2, options: .storageModeShared, offset: 0)!
        let uy = heap.makeBuffer(length: n * 2, options: .storageModeShared, offset: Arena.aligned(n * 2))!
        let untracked = try await chain(ux, uy)
        print("encode ahead on an untracked heap: \(zip(untracked, want).filter { $0 != $1 }.count) of \(n) values wrong")
    }
}
