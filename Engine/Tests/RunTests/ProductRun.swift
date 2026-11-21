import CoreAI
import Foundation
import Metal
import SiliCore
import Testing

/// A product whose parts are joined along `out` (`Recipe.product`), its weights across two shards, int8 codes with
/// SDNQ's and ComfyUI's scales or bf16, in either layout; a chain of calls put on the stream unwaited; the bf16 layout
/// measured on this Mac. The bars as `FunctionsRun`'s, written before the run: the f16 path's rounding, ~1e-3 relative;
/// bf16 throughout, its rounding of the result, ~4e-3; a wrong layout, part or scale gives errors of order 1.
extension Runs {
    @Test(.enabled(if: metalEnabled), arguments: [CodesLayout.inOut, .outIn])
    func aJoinedProductAcrossShardsIsTheCPUs(_ layout: CodesLayout) async throws {
        let bench = try await Bench()
        let (rows, k, n1, n2) = (32, 128, 64, 96), n = n1 + n2
        var rng = Seeded(state: 5)
        let c1 = (0..<n1 * k).map { _ in Int8(Int(rng.next() * 255) - 127) }, c2 = (0..<n2 * k).map { _ in Int8(Int(rng.next() * 255) - 127) }
        // SDNQ's scales in bf16 on one part, ComfyUI's in f32 on the other: both read to f16.
        let s1 = rng.uniform(n1, 0.001...0.01).map { BFloat16Bits.rounded($0) }, s2 = rng.uniform(n2, 0.001...0.01)
        let w1 = rng.uniform(n1 * k, -0.1...0.1).map { BFloat16Bits.rounded($0) }, w2 = rng.uniform(n2 * k, -0.1...0.1).map { BFloat16Bits.rounded($0) }
        let a = try weightsFile("joined-1.safetensors", [
            ("a.weight", "I8", [n1, k], c1.withUnsafeBytes { Array($0) }), ("a.scale", "BF16", [n1, 1], s1.withUnsafeBytes { Array($0) }),
            ("p.weight", "BF16", [n1, k], w1.withUnsafeBytes { Array($0) })])
        let b = try weightsFile("joined-2.safetensors", [
            ("b.weight", "I8", [n2, k], c2.withUnsafeBytes { Array($0) }), ("b.weight_scale", "F32", [n2, 1], s2.withUnsafeBytes { Array($0) }),
            ("q.weight", "BF16", [n2, k], w2.withUnsafeBytes { Array($0) })])
        let file = try SafeTensors(shards: [a, b])
        let entries: [Weights.Entry] = [("w", "a", .product(joined: ["b"])), ("v", "p", .product(joined: ["q"]))]
        let x = rng.uniform(rows * k)
        let arena = try Arena(bench.device, bytes: try Weights.size(entries, in: file, codes: layout, bfloat16: layout) + 3 * Arena.aligned(rows * n * 2) + 2 * Arena.aligned(rows * k * 2))
        let placed = try Weights.load(entries, from: file, into: arena, codes: layout, bfloat16: layout)
        let shape = layout == .inOut ? [k, n] : [n, k]
        #expect(placed["w"]!.shape == shape && placed["v"]!.shape == shape && placed["w_scale"]!.shape == [n])
        let x16 = arena.take(rows * k * 2), x16b = arena.take(rows * k * 2), y = arena.take(rows * n * 2), z = arena.take(rows * n * 2)
        let xs = arena.write(x, at: x16)
        let xb = arena.pointer(x16b).bindMemory(to: UInt16.self, capacity: rows * k)
        for i in x.indices { xb[i] = BFloat16Bits.rounded(x[i]) }
        func view(_ p: Placed) -> View { View(buffer: arena.buffer(p.offset), type: p.type, shape: p.shape) }

        var m = Aicode.Module(arguments: [("x", Aicode.tensor([rows, k], .f16)), ("w", Aicode.tensor(shape, .si8)), ("w_scale", Aicode.tensor([n], .f16)),
                                          ("xb", Aicode.tensor([rows, k], .bf16)), ("v", Aicode.tensor(shape, .bf16))])
        var constants = Aicode.ScalingConstants()
        m.outputs = [("y", m.body.w8a16(m.argument("x"), codes: m.argument("w"), scale: m.argument("w_scale"), layout: layout, constants: &constants)),
                     ("z", m.body.product(m.argument("xb"), m.argument("v"), layout: layout))]
        let f = try await Functions.load(try Functions.asset(m, "run-joined-\(layout.rawValue)"))
        try await Run.call(f, bench.stream, inputs: ["x": View(buffer: arena.buffer(x16), type: .float16, shape: [rows, k]), "w": view(placed["w"]!),
                                                     "w_scale": view(placed["w_scale"]!), "xb": View(buffer: arena.buffer(x16b), type: .bfloat16, shape: [rows, k]),
                                                     "v": view(placed["v"]!)],
                           outputs: [("y", View(buffer: arena.buffer(y), type: .float16, shape: [rows, n])),
                                     ("z", View(buffer: arena.buffer(z), type: .bfloat16, shape: [rows, n]))])

        let scale = s1.map { Double(Float16(BFloat16Bits.widened($0))) } + s2.map { Double(Float16($0)) }
        let codes = c1 + c2, weights = w1 + w2
        let wantY = product(xs, rows: rows, k: k, n: n) { i, o in Double(codes[o * k + i]) * scale[o] }
        let wantZ = product(x.map { BFloat16Bits.widened(BFloat16Bits.rounded($0)) }, rows: rows, k: k, n: n) { i, o in Double(BFloat16Bits.widened(weights[o * k + i])) }
        let zs = arena.pointer(z).bindMemory(to: UInt16.self, capacity: rows * n)
        let ey = relative(arena.halves(y, rows * n), wantY), ez = relative((0..<rows * n).map { BFloat16Bits.widened(zs[$0]) }, wantZ)
        #expect(ey < 2e-3, "W8A16 relative error \(ey)")
        #expect(ez < 4e-3, "bf16 relative error \(ez)")
    }

    /// Calls put on the stream one after the other, unwaited, each reading the value the one before returned: eight
    /// steps of `x · 2 + 1`, one wait at the end. The bar: exact (small integers in f16).
    @Test(.enabled(if: metalEnabled))
    func aChainOfCallsIsOrderedByItsValues() async throws {
        let bench = try await Bench()
        let n = 4096
        let arena = try Arena(bench.device, bytes: 2 * Arena.aligned(n * 2))
        let x = arena.take(n * 2), y = arena.take(n * 2)
        arena.write((0..<n).map { Float($0 % 7) }, at: x)
        var m = Aicode.Module(arguments: [("x", Aicode.tensor([n], .f16))])
        let b = m.body
        m.outputs = [("y", b.binary(.add, b.binary(.mul, m.argument("x"), b.scalar(2, .f16)), b.scalar(1, .f16)))]
        let f = try await Functions.load(try Functions.asset(m, "run-chain"))
        var value = Run.value(View(buffer: arena.buffer(x), type: .float16, shape: [n]))
        for _ in 0..<7 { value = try Run.encode(f, bench.stream, inputs: [:], chained: ["x": value], outputs: [])["y"]! }
        let left = try Run.encode(f, bench.stream, inputs: [:], chained: ["x": value], outputs: [("y", View(buffer: arena.buffer(y), type: .float16, shape: [n]))])
        #expect(left.isEmpty)
        await Run.finish(bench.stream)
        let got = arena.halves(y, n), want = (0..<n).map { i -> Float in (0..<8).reduce(Float(i % 7)) { v, _ in 2 * v + 1 } }
        #expect(got == want, "first difference at \(zip(got, want).enumerated().first { $0.1.0 != $0.1.1 }?.offset ?? -1)")
    }
}
