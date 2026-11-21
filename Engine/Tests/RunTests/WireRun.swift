import Darwin
import Foundation
import Metal
import SiliCore
import Testing

/// **What the arena wires, macOS keeps wired, the GPU reads as written, and the arena lets go** (`Arena.wire`): a tensor
/// of the heap and a surface, 256 MiB each, wired then the arena let go; in another arena, the same wired, written by the
/// CPU, copied by the GPU. The bars: the engine's count up by their pages exactly and back to where it was; the process's
/// wired pages as the system counts them up by at least 90 % of each tensor as it is wired, and down by at least 90 % of
/// both as the arena goes; every value the GPU copies the one the CPU wrote.
///
/// Measured 20/11 on the M1 Pro: the system wires each page once (its whole count up by 1024.5 MiB for a tensor of 1 GiB
/// of the heap), while the process's count says twice that for a tensor of the heap, once for a surface.
extension Runs {
    @Test(.enabled(if: metalEnabled))
    func whatTheArenaWiresStaysWiredUntilItGoes() async throws {
        let bench = try await Bench()
        let n = 256 << 20
        // The process's own wired pages (`proc_pid_rusage`): the system's whole count moves by 100 to 150 MiB of other
        // processes' within one 256 MiB wiring (20/11), the process's not.
        func systemWired() -> Int {
            var info = rusage_info_v2()
            let status = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V2, $0) }
            }
            precondition(status == 0)
            return Int(info.ri_wired_size)
        }
        let engine0 = Arena.wiredBytes
        var arena: Arena? = try Arena(bench.device, bytes: Arena.aligned(n))
        let heap = arena!.take(n), surface = arena!.take(surface: [16384, n / 2 / 16384], element: 2)
        for (name, key) in [("the heap's", heap), ("the surface", surface)] {
            let before = systemWired()
            #expect(arena!.wire(key, bytes: n))
            let wired = systemWired() - before
            print("wired: \(name), the process's count up by \(Memory.mib(wired)) for \(Memory.mib(n))")
            #expect(wired >= n * 9 / 10)
        }
        #expect(Arena.wiredBytes - engine0 == 2 * n)
        let before = systemWired()
        arena = nil
        let unwired = before - systemWired()
        print("let go: the process's count down by \(Memory.mib(unwired)), the engine's at \(Memory.mib(Arena.wiredBytes - engine0))")
        #expect(unwired >= 2 * n * 9 / 10)
        #expect(Arena.wiredBytes == engine0)

        // The GPU reads what the CPU wrote past the wiring, an arena of its own: once the GPU has used a buffer, Metal
        // keeps its pages wired until it frees it (within 0.5 s of the let-go, 20/11), which no reading at once sees.
        let read = try Arena(bench.device, bytes: Arena.aligned(n))
        let queue = try #require(bench.device.makeCommandQueue())
        for key in [read.take(n), read.take(surface: [16384, n / 2 / 16384], element: 2)] {
            #expect(read.wire(key, bytes: n))
            let p = read.pointer(key).assumingMemoryBound(to: UInt32.self), count = n / 4
            func value(_ i: Int) -> UInt32 { UInt32(truncatingIfNeeded: i &* 2654435761) }
            for i in stride(from: 0, to: count, by: 1024) { p[i] = value(i) }
            let out = try #require(bench.device.makeBuffer(length: n, options: .storageModeShared))
            let cb = try #require(queue.makeCommandBuffer()), blit = try #require(cb.makeBlitCommandEncoder())
            blit.copy(from: read.buffer(key), sourceOffset: 0, to: out, destinationOffset: 0, size: n)
            blit.endEncoding()
            cb.commit()
            await cb.completed()
            let q = out.contents().assumingMemoryBound(to: UInt32.self)
            #expect(stride(from: 0, to: count, by: 1024).allSatisfy { q[$0] == value($0) })
        }
    }
}
