import CoreAI
import Foundation
import Metal
import SiliCore
import Testing

/// A product whose parts are joined along `out` (`Recipe.product`), its weights across two shards, int8 codes with
/// SDNQ's and ComfyUI's scales (int8 × int8, against the exact int8 product), or bf16 `[out, in]`; a chain of calls put on the stream unwaited. The bars as `FunctionsRun`'s, written before the run: the f16 path's rounding, ~1e-3 relative;
/// bf16 throughout, its rounding of the result, ~4e-3; a wrong layout, part or scale gives errors of order 1.
extension Runs {
    @Test(.enabled(if: metalEnabled))
    func aJoinedProductAcrossShardsIsTheCPUs() async throws {
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
        let arena = try Arena(bench.device, bytes: try Weights.size(entries, in: file) + 3 * Arena.aligned(rows * n * 2) + 2 * Arena.aligned(rows * k * 2))
        let placed = try Weights.load(entries, from: file, into: arena)
        let shape = [1, n, k]
        #expect(placed["w"]!.shape == shape && placed["v"]!.shape == [n, k] && placed["w_scale"]!.shape == [n])
        let x16 = arena.take(rows * k * 2), x16b = arena.take(rows * k * 2), y = arena.take(rows * n * 2), z = arena.take(rows * n * 2)
        let xs = arena.write(x, at: x16)
        let xb = arena.pointer(x16b).bindMemory(to: UInt16.self, capacity: rows * k)
        for i in x.indices { xb[i] = BFloat16Bits.rounded(x[i]) }
        func view(_ p: Placed) -> View { View(buffer: arena.buffer(p.offset), type: p.type, shape: p.shape) }

        var m = Aicode.Module(arguments: [("x", Aicode.tensor([rows, k], .f16)), ("w", Aicode.tensor(shape, .si8)), ("w_scale", Aicode.tensor([n], .f16)),
                                          ("xb", Aicode.tensor([rows, k], .bf16)), ("v", Aicode.tensor([n, k], .bf16))])
        var constants = Aicode.ScalingConstants()
        m.outputs = [("y", m.body.w8a8(m.argument("x"), codes: m.argument("w"), scale: m.argument("w_scale"), constants: &constants)),
                     ("z", m.body.linear(m.argument("xb"), m.argument("v")))]
        let f = try await Functions.load(try Functions.asset(m, "run-joined-w8a8"))
        try await Run.call(f, bench.stream, inputs: ["x": View(buffer: arena.buffer(x16), type: .float16, shape: [rows, k]), "w": view(placed["w"]!),
                                                     "w_scale": view(placed["w_scale"]!), "xb": View(buffer: arena.buffer(x16b), type: .bfloat16, shape: [rows, k]),
                                                     "v": view(placed["v"]!)],
                           outputs: [("y", View(buffer: arena.buffer(y), type: .float16, shape: [rows, n])),
                                     ("z", View(buffer: arena.buffer(z), type: .bfloat16, shape: [rows, n]))])

        let scale = s1.map { Double(Float16(BFloat16Bits.widened($0))) } + s2.map { Double(Float16($0)) }
        let codes = c1 + c2, weights = w1 + w2
        let wantY = int8Product(xs, rows: rows, k: k, n: n, scales: scale) { o, i in codes[o * k + i] }
        let wantZ = product(x.map { BFloat16Bits.widened(BFloat16Bits.rounded($0)) }, rows: rows, k: k, n: n) { i, o in Double(BFloat16Bits.widened(weights[o * k + i])) }
        let zs = arena.pointer(z).bindMemory(to: UInt16.self, capacity: rows * n)
        let ey = relative(arena.halves(y, rows * n), wantY), ez = relative((0..<rows * n).map { BFloat16Bits.widened(zs[$0]) }, wantZ)
        #expect(ey < 2e-3, "W8A8 relative error \(ey)")
        #expect(ez < 4e-3, "bf16 relative error \(ez)")
    }

    /// **The product int8 × int8 (`w8a8`) is the CPU's exact int8 product by rows**: each row of x brought to int8 codes
    /// by its peak (127 · 2⁻¹⁴ at least), the codes' sums exact in integers, times the row's scale and the codes' — on 64
    /// rows of k = 10 240 inputs (the widest product of the shared forms) laid by pieces as the recipe lays them (four of
    /// 2560: `Weights.pieces`), magnitudes
    /// from 1/16 to 8, a row of zeros and a row whose peak lies under the floor. The bars, written before the run: within
    /// 2e-3 of the exact int8 product (f16's roundings of 1/127, of the sums and of the scaled result, each within 2⁻¹¹),
    /// and at least twice as far from x's own product by the codes (W8A16: the activations' quantization, ~4e-3 for
    /// uniform rows); the row of zeros exactly zeros, every value finite.
    @Test(.enabled(if: metalEnabled))
    func theIntegerProductIsTheCPUsInt8ProductByRows() async throws {
        let bench = try await Bench()
        let (rows, k, n) = (64, 10_240, 128), (zero, small) = (5, 9)
        var rng = Seeded(state: 8)
        var x = rng.uniform(rows * k)
        for r in 0..<rows {
            let magnitude: Float = r == zero ? 0 : r == small ? 4e-3 : [1, 8, 1.0 / 16][r % 3]
            for i in 0..<k { x[r * k + i] *= magnitude }
        }
        let codes = (0..<n * k).map { _ in Int8(Int(rng.next() * 255) - 127) }
        let scales = rng.uniform(n, 0.001...0.01)
        let arena = try Arena(bench.device, bytes: Arena.aligned(rows * k * 2) + Arena.aligned(n * k) + Arena.aligned(n * 2) + Arena.aligned(rows * n * 2))
        let (xk, wk, sk, yk) = (arena.take(rows * k * 2), arena.take(n * k), arena.take(n * 2), arena.take(rows * n * 2))
        let xs = arena.write(x, at: xk), ss = arena.write(scales, at: sk)
        let (count, piece) = Weights.pieces(k)
        var pieced = [Int8](repeating: 0, count: n * k)
        for p in 0..<count { for o in 0..<n { for i in 0..<piece { pieced[(p * n + o) * piece + i] = codes[o * k + p * piece + i] } } }
        arena.write(codes: pieced, at: wk)

        var m = Aicode.Module(arguments: [("x", Aicode.tensor([1, rows, k], .f16)), ("w", Aicode.tensor([count, n, piece], .si8)),
                                          ("w_scale", Aicode.tensor([1, 1, n], .f16))])
        var constants = Aicode.ScalingConstants()
        m.outputs = [("y", m.body.w8a8(m.argument("x"), codes: m.argument("w"), scale: m.argument("w_scale"), constants: &constants))]
        let f = try await Functions.load(try Functions.asset(m, "run-w8a8-\(rows)x\(k)x\(n)"))
        try await Run.call(f, bench.stream, inputs: ["x": View(buffer: arena.buffer(xk), type: .float16, shape: [1, rows, k]),
                                                     "w": View(buffer: arena.buffer(wk), type: .int8, shape: [count, n, piece]),
                                                     "w_scale": View(buffer: arena.buffer(sk), type: .float16, shape: [1, 1, n])],
                           outputs: [("y", View(buffer: arena.buffer(yk), type: .float16, shape: [1, rows, n]))])
        let got = arena.halves(yk, rows * n)

        // The CPU's: the row's peak and its scale in f16 as the function computes them, its codes a half away from zero.
        let exact = int8Product(xs, rows: rows, k: k, n: n, scales: ss.map(Double.init)) { o, i in codes[o * k + i] }
        let activations = product(xs, rows: rows, k: k, n: n) { i, o in Double(codes[o * k + i]) * Double(ss[o]) }
        let (e8, e16) = (relative(got, exact), relative(got, activations))
        print(String(format: "w8a8 %d × %d → %d: from the int8 product %.2e, from x's own %.2e", rows, k, n, e8, e16))
        #expect(got.allSatisfy { $0.isFinite }, "a value not finite")
        #expect(got[zero * n ..< (zero + 1) * n].allSatisfy { $0 == 0 }, "the row of zeros: \(Array(got[zero * n ..< zero * n + 4]))")
        #expect(e8 < 2e-3, "W8A8 relative error \(e8)")
        #expect(e16 > 2 * e8, "from x's own product \(e16), from the int8 product \(e8)")
        let r = small, row = relative(Array(got[r * n ..< (r + 1) * n]), Array(exact[r * n ..< (r + 1) * n]))
        #expect(row < 2e-3, "the row under the floor: relative error \(row)")
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
