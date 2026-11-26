import CoreAI
import Foundation
import Metal
import SiliCore
import Testing

/// **A stage let go leaves nothing**: a bf16 stage of 3 GiB (a resident product read from its file, a tensor the CPU
/// writes) opened, called, written and closed, twice in the process. The CPU's write and the close inside one autorelease
/// pool not drained meanwhile, as the command line's main thread is while it waits on its stages: before `Arena.bytes`
/// (19/11), `contents()` autoreleased the written tensor's buffer there, and the heap stayed whole past the close (24.7
/// GiB for a 6 MiB tensor). The bars were written before the run: each close back within `Stage.margin`, nothing alive
/// nor held (`Arena.lingering` 0); the second stage's floor within 16 MiB of the first's.
///
/// What a floor reads besides the stages, measured 19/11 on the M5 Pro (a trivial function, then others, each loaded,
/// called once and let go in a fresh process), and kept out of the floors by name:
/// - **the process's first call of Core AI**: 9.1 MiB before it, 31.1 after it and its let-go, 35 to 39 after any number
///   of functions since — the runtime's own, paid once (`Functions.witness`, `Bench`), so it lies in every floor;
/// - **what Core AI keeps of a function let go, ~36 MiB, for 1 to 2 s** (still held at 1.0 s, gone at 2.0, for every
///   function): `Stage.settle` stops waiting within `Stage.margin`, so a stage opened at once counts it in its floor —
///   P the higher by it, never the lower (`settledCoreAI`).
extension Runs {
    @Test(.enabled(if: metalEnabled))
    func aStageLetGoLeavesNothing() async throws {
        let bench = try await Bench()
        try await settledCoreAI()
        let (rows, k, n, big) = (256, 2048, 2048, 3 << 30)
        var rng = Seeded(state: 5)
        let w = rng.uniform(k * n, -0.05...0.05).map { BFloat16Bits.rounded($0) }
        let file = try SafeTensors(try weightsFile("let-go.safetensors", [("w", "BF16", [k, n], w.withUnsafeBytes { Array($0) })]))
        var d = Stage.Declaration("run-let-go")
        d.tensors = [Stage.Tensor("x", bytes: rows * k * 2), Stage.Tensor("big", bytes: big), Stage.Tensor("y", bytes: rows * n * 2, .apart)]
        d.residents = [("w", "w", .bfloat16(joined: []))]
        d.functions = ["product": Stage.Function("run-let-go-\(rows)x\(k)x\(n)") {
            var m = Aicode.Module(arguments: [("x", Aicode.tensor([rows, k], .bf16)), ("w", Aicode.tensor([k, n], .bf16))])
            m.outputs = [("y", m.body.matmul(m.argument("x"), m.argument("w"), result: Aicode.tensor([rows, n], .bf16)))]
            return m
        }]
        func call(_ f: InferenceFunction, _ stage: Stage) async throws {
            try await Run.call(f, stage.stream, inputs: ["x": stage.view("x", [rows, k], .bfloat16), "w": stage.view(stage.residents["w"]!)],
                               outputs: [("y", stage.view("y", [rows, n], .bfloat16))])
        }
        var floors: [Int] = []
        for round in 0..<2 {
            let stage = try await Stage.open(bench.device, d, files: [file], budget: 1 << 40)
            floors.append(stage.plan.floor)
            try await call(stage.functions["product"]!, stage)
            let settled = autoreleasepool { () -> Stage.Settled in
                let p = stage.pointer("big")
                for i in stride(from: 0, to: big, by: Arena.page) { p.storeBytes(of: UInt8(truncatingIfNeeded: i), toByteOffset: i, as: UInt8.self) }
                stage.close()
                return Stage.settle()
            }
            print("let go \(round): floor \(Memory.mib(stage.plan.floor)), P \(Memory.mib(stage.plan.p)), \(settled.line), lingering \(Memory.mib(Arena.lingering))")
            #expect(settled.within, "round \(round): \(settled.line)")
            #expect(Arena.lingering == 0, "round \(round): \(Memory.mib(Arena.lingering)) held")
            try await settledCoreAI()
        }
        #expect(abs(floors[1] - floors[0]) <= 16 << 20, "the floors \(floors.map { Memory.mib($0) })")
    }

    /// What Core AI keeps of the functions let go, gone: 2.5 s, past the 2 s it was measured to keep them.
    func settledCoreAI() async throws { try await Task.sleep(for: .milliseconds(2500)) }
}
