import CoreAI
import Foundation
import Metal
import SiliCore
import Testing

/// **A stream's waits side by side** (`SerialStream`): on a busy Mac, the waits of a block's calls pile up and run at
/// once.
extension Runs {
    /// One call put on the stream, then four waits for it at once, 20 000 times: Core AI's stream alone, its waits on
    /// the cooperative pool, died at `ComputeStream.swift:421` within 10 000 in 4 runs of 4 (25/11 on the M1 Pro, 7 s
    /// for 20 000); through `SerialStream`, every wait returns.
    @Test(.enabled(if: metalEnabled))
    func aStreamsWaitsSideBySide() async throws {
        let bench = try await Bench()
        let n = 4096
        var m = Aicode.Module(arguments: [("x", Aicode.tensor([n], .f16))])
        m.outputs = [("y", m.body.binary(.add, m.argument("x"), m.body.scalar(1, .f16)))]
        let f = try await Functions.load(try Functions.asset(m, "waits-side-by-side"))
        let arena = try Arena(bench.device, bytes: 2 * Arena.aligned(n * 2))
        let x = View(buffer: arena.buffer(arena.take(n * 2)), type: .float16, shape: [n])
        let y = View(buffer: arena.buffer(arena.take(n * 2)), type: .float16, shape: [n])
        var returned = 0
        for _ in 0..<20_000 {
            #expect(try Run.encode(f, bench.stream, inputs: ["x": x], outputs: [("y", y)]).isEmpty)
            for w in (0..<4).map({ _ in bench.stream.done() }) { _ = await w.value; returned += 1 }
        }
        #expect(returned == 80_000)
    }
}
