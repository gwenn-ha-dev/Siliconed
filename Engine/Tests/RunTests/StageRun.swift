import CoreAI
import Foundation
import Metal
import SiliCore
import Testing

/// **A real stage under its plan, its groups kept in flight**: four groups of W8A16 weights in a cycle, the socle a ring
/// of two, B leaving room past it for the product's Core AI share (≈ 56 MiB alone, 19/11) and two groups and a half;
/// three cycles of a product per group, the signal after each cycle's second group, each product judged against the CPU
/// (the ring's slots and the kept places hold what was acquired), the footprint's peak against B, then everything let
/// go. The bars were written before the run: the peak at most B; a group kept at least; each product within 2e-3 of the
/// CPU, as the single product (`aW8A16ProductIsTheCPUs`); the stage let go back within 64 MiB of the footprint before
/// it; its reading kept on this Mac (`ThisMac.reserves`).
extension Runs {
    @Test(.enabled(if: metalEnabled))
    func aStageHoldsUnderItsP() async throws {
        let bench = try await Bench()
        let (rows, k, n) = (256, 2048, 4096)
        let names = ["a", "b", "c", "d"]
        var rng = Seeded(state: 11)
        var codes: [String: [Int8]] = [:], scales: [String: [Float]] = [:], tensors: [(String, String, [Int], [UInt8])] = []
        for g in names {
            codes[g] = (0..<n * k).map { _ in Int8(Int(rng.next() * 255) - 127) }
            scales[g] = rng.uniform(n, 0.001...0.01)
            tensors.append(("\(g).weight", "I8", [n, k], codes[g]!.withUnsafeBytes { Array($0) }))
            tensors.append(("\(g).weight_scale", "F32", [n, 1], scales[g]!.withUnsafeBytes { Array($0) }))
        }
        let file = try SafeTensors(try weightsFile("stage.safetensors", tensors))

        func declaration(_ codes: CodesLayout) -> Stage.Declaration {
            var d = Stage.Declaration("run")
            d.codes = codes
            d.tensors = [Stage.Tensor("x", bytes: rows * k * 2), Stage.Tensor("y", bytes: rows * n * 2, .apart)]
            for g in names { d.groups[g] = [("w", g, .codes)] }
            d.slot = "a"
            d.cycle = names
            d.functions = ["product": Stage.Function("run-stage-product-\(codes.rawValue)-\(rows)x\(k)x\(n)") {
                var m = Aicode.Module(arguments: [("x", Aicode.tensor([rows, k], .f16)), ("w", Aicode.tensor(codes == .inOut ? [k, n] : [n, k], .si8)),
                                                  ("w_scale", Aicode.tensor([n], .f16))])
                var constants = Aicode.ScalingConstants()
                m.outputs = [("y", m.body.w8a16(m.argument("x"), codes: m.argument("w"), scale: m.argument("w_scale"), layout: codes, constants: &constants))]
                return m
            }]
            return d
        }
        let layout = CodesLayout.inOut
        func call(_ f: InferenceFunction, _ stage: Stage, _ held: Stage.Held) async throws {
            try await Run.call(f, stage.stream, inputs: ["x": stage.view("x", [rows, k], .float16), "w": stage.view(held.weights["w"]!),
                                                         "w_scale": stage.view(held.weights["w_scale"]!)],
                               outputs: [("y", stage.view("y", [rows, n], .float16))])
        }

        let floor = Memory.footprint()
        let socle = try Stage.plan(declaration(layout), files: [file], budget: 1 << 40, floor: floor)
        let group = Arena.allocation(bytes: Arena.aligned(n * k) + Arena.aligned(n * 2))
        let budget = socle.p + Stage.margin + (64 << 20) + 5 * group / 2
        let plan = try Stage.plan(declaration(layout), files: [file], budget: budget, floor: floor)
        #expect(plan.kept.isEmpty && plan.depth == 2, "\(plan.line)")
        print("stage run: " + plan.line + (ThisMac.current.reserves?[Stage.key(plan.declaration)].map { ", a reserve of \(Memory.mib($0))" } ?? ", no reserve read"))

        let watch = FootprintWatch()
        watch.reset()
        let stage = try await Stage(bench.device, plan: plan, files: [file])
        let xs = (0..<rows * k).map { _ in Float(Float16(Float(rng.next() * 2 - 1))) }
        let x = stage.pointer("x").bindMemory(to: Float16.self, capacity: rows * k)
        for i in xs.indices { x[i] = Float16(xs[i]) }
        // The function read from the stage at each call, never held here: a function holds its last call's buffers, the
        // stage's heap among them, until it is let go (`close`).
        let checked = 4
        var worst = 0.0
        for _ in 0..<3 {
            for (i, g) in names.enumerated() {
                let held = try stage.acquire(g)
                try await call(stage.functions["product"]!, stage, held)
                stage.release(held)
                if i == 1 { stage.signal() }
                let s16 = scales[g]!.map { Double(Float16($0)) }, c = codes[g]!
                let want = product(Array(xs[0..<checked * k]), rows: checked, k: k, n: n) { i, o in Double(c[o * k + i]) * s16[o] }
                let y = stage.pointer("y").bindMemory(to: Float16.self, capacity: checked * n)
                worst = max(worst, relative((0..<checked * n).map { Float(y[$0]) }, want))
            }
        }
        let peak = watch.peak, kept = stage.kept
        watch.stop()
        let settled = stage.close()
        print("stage run: the peak \(Memory.mib(peak)) under B \(Memory.mib(budget)), \(kept.count) groups kept \(kept); the worst product \(worst); let go \(settled.line)")
        #expect(worst < 2e-3, "relative error \(worst)")
        #expect(peak <= budget, "the peak \(Memory.mib(peak)), B \(Memory.mib(budget))")
        #expect(!kept.isEmpty)
        #expect(settled.within, "\(settled.line)")
        #expect(ThisMac.current.reserves?[stage.key] != nil)
    }
}
