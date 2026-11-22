import CoreAI
import Foundation
import Metal
import SiliCore
import Testing

/// **A real stage under its plan, its groups kept in flight**: four groups of int8 codes in a cycle, the socle a ring
/// of two, B leaving room past it for the product's Core AI share (≈ 56 MiB alone, 19/11) and two groups and a half;
/// three cycles of a product per group, the signal after each cycle's second group, the footprint's peak against B, then
/// everything let go; each product's first rows judged, once the stage is gone, against the same function called on the
/// group's weights read apart (the ring's slots and the kept places hold what was acquired: the arithmetic is the single
/// product's to judge, `theCompactsProductIsTheCPUsOnEitherUnitsCodes`). The bars were written before the run: the peak
/// at most B; a group kept at least; each product the same to the bit; the stage let go back within 64 MiB of the
/// footprint before it; its reading kept on this Mac (`ThisMac.reserves`).
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

        func declaration() -> Stage.Declaration {
            var d = Stage.Declaration("run")
            d.tensors = [Stage.Tensor("x", bytes: rows * k * 2), Stage.Tensor("y", bytes: rows * n * 2, .apart)]
            for g in names { d.groups[g] = [("w", g, .codes)] }
            d.slot = "a"
            d.cycle = names
            d.functions = ["product": Stage.Function("run-stage-w8a8-\(rows)x\(k)x\(n)") {
                var m = Aicode.Module(arguments: [("x", Aicode.tensor([rows, k], .f16)), ("w", Aicode.tensor([1, n, k], .si8)),
                                                  ("w_scale", Aicode.tensor([n], .f16))])
                var constants = Aicode.ScalingConstants()
                m.outputs = [("y", m.body.w8a8(m.argument("x"), codes: m.argument("w"), scale: m.argument("w_scale"), constants: &constants))]
                return m
            }]
            return d
        }
        func call(_ f: InferenceFunction, _ stage: Stage, _ held: Stage.Held) async throws {
            try await Run.call(f, stage.stream, inputs: ["x": stage.view("x", [rows, k], .float16), "w": stage.view(held.weights["w"]!),
                                                         "w_scale": stage.view(held.weights["w_scale"]!)],
                               outputs: [("y", stage.view("y", [rows, n], .float16))])
        }

        let floor = Memory.footprint()
        let socle = try Stage.plan(declaration(), files: [file], budget: 1 << 40, floor: floor)
        let group = Arena.allocation(bytes: Arena.aligned(n * k) + Arena.aligned(n * 2))
        let budget = socle.p + Stage.keepClear + (64 << 20) + 5 * group / 2
        let plan = try Stage.plan(declaration(), files: [file], budget: budget, floor: floor)
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
        var got: [(group: String, y: [UInt16])] = []
        for _ in 0..<3 {
            for (i, g) in names.enumerated() {
                let held = try stage.acquire(g)
                try await call(stage.functions["product"]!, stage, held)
                stage.release(held)
                if i == 1 { stage.signal() }
                let y = stage.pointer("y").bindMemory(to: UInt16.self, capacity: checked * n)
                got.append((g, Array(UnsafeBufferPointer(start: y, count: checked * n))))
            }
        }
        let peak = watch.peak, kept = stage.kept
        watch.stop()
        let settled = stage.close()

        // The same function on each group's weights read apart, the same input.
        let entries: [Weights.Entry] = [("w", "a", .codes)]
        let arena = try Arena(bench.device, bytes: try Weights.size(entries, in: file) + Arena.aligned(rows * k * 2) + Arena.aligned(rows * n * 2))
        let (xk, yk) = (arena.take(rows * k * 2), arena.take(rows * n * 2))
        let xa = arena.pointer(xk).bindMemory(to: Float16.self, capacity: rows * k)
        for i in xs.indices { xa[i] = Float16(xs[i]) }
        let f = try await Functions.load(try Functions.asset(plan.declaration.functions["product"]!.asset, plan.declaration.functions["product"]!.build))
        // One place, each group read into it in turn (the groups lay out alike).
        let place = Weights.place(try Weights.layout(entries, in: file), in: arena)
        func view(_ p: Placed) -> View { View(buffer: arena.buffer(p.offset), type: p.type, shape: p.shape) }
        var differing = 0
        for (g, y) in got {
            let staging = Pages(bytes: try Weights.staging([("w", g, .codes)], in: file))
            try withExtendedLifetime(staging) { try Weights.read([("w", g, .codes)], from: file, into: place, arena: arena, staging: staging.base) }
            try await Run.call(f, bench.stream, inputs: ["x": View(buffer: arena.buffer(xk), type: .float16, shape: [rows, k]), "w": view(place["w"]!),
                                                         "w_scale": view(place["w_scale"]!)],
                               outputs: [("y", View(buffer: arena.buffer(yk), type: .float16, shape: [rows, n]))])
            let want = arena.pointer(yk).bindMemory(to: UInt16.self, capacity: checked * n)
            differing += (0..<checked * n).filter { y[$0] != want[$0] }.count
        }
        print("stage run: the peak \(Memory.mib(peak)) under B \(Memory.mib(budget)), \(kept.count) groups kept \(kept); \(differing) values of \(got.count * checked * n) differ from the product apart; let go \(settled.line)")
        #expect(differing == 0, "\(differing) values differ from the same product on the weights read apart")
        #expect(peak <= budget, "the peak \(Memory.mib(peak)), B \(Memory.mib(budget))")
        #expect(!kept.isEmpty)
        #expect(settled.within, "\(settled.line)")
        #expect(ThisMac.current.reserves?[stage.key] != nil)
    }
}
