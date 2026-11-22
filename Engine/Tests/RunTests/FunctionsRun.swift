import CoreAI
import Foundation
import Metal
import SiliCore
import Synchronization
import Testing

/// Small functions written op by op, their weights read by a recipe into the arena, called on the GPU and judged against
/// the CPU in f64. Each bar was written before it ran: the f16 path's rounding, ~1e-3 relative, nothing that would
/// hide a wrong layout or a wrong axis (those give errors of order 1).
extension Runs {
    /// **The Compact's product on either unit's codes**: the file's codes `[out, in]` read by the `codes` recipe by pieces
    /// of k (k = 8192: two pieces of 4096), the GPU's int8 × int8 on them (`w8a8`), the ANE's q64 on their copy `[1, in,
    /// out]` (`Weights.transposedPieces`), both called on the GPU. The bars, written before the run: each within 2e-3 of
    /// the CPU's — the exact int8 product by rows for the first, x's own product by the codes (W8A16) for the second;
    /// a wrong piece, layout or scale gives errors of order 1.
    @Test(.enabled(if: metalEnabled))
    func theCompactsProductIsTheCPUsOnEitherUnitsCodes() async throws {
        let bench = try await Bench()
        let (rows, k, n) = (64, 8192, 192)
        var rng = Seeded(state: 3)
        let codes = (0..<n * k).map { _ in Int8(Int(rng.next() * 255) - 127) }
        let scales = rng.uniform(n, 0.001...0.01)
        let file = try SafeTensors(try weightsFile("compact.safetensors", [
            ("w.weight", "I8", [n, k], codes.withUnsafeBytes { Array($0) }),
            ("w.weight_scale", "F32", [n, 1], scales.withUnsafeBytes { Array($0) }),
        ]))
        let entries: [Weights.Entry] = [("w", "w", .codes)]
        let (count, piece) = try Weights.pieces(k)
        let arena = try Arena(bench.device, bytes: try Weights.size(entries, in: file) + Arena.aligned(k * n) + Arena.aligned(rows * k * 2) + 2 * Arena.aligned(rows * n * 2))
        let w = try Weights.load(entries, from: file, into: arena)
        #expect(w["w"]!.shape == [count, n, piece] && count == 2)
        let copy = arena.take(k * n), x = arena.take(rows * k * 2), y8 = arena.take(rows * n * 2), y16 = arena.take(rows * n * 2)
        Weights.transposedPieces(arena.pointer(w["w"]!.offset).assumingMemoryBound(to: Int8.self), count: count, out: n, piece: piece,
                                 into: arena.pointer(copy).assumingMemoryBound(to: Int8.self))
        let xs = arena.write(rng.uniform(rows * k), at: x)

        var m = Aicode.Module(arguments: [("x", Aicode.tensor([1, rows, k], .f16)), ("w", Aicode.tensor([count, n, piece], .si8)),
                                          ("c", Aicode.tensor([1, k, n], .si8)), ("w_scale", Aicode.tensor([1, 1, n], .f16))])
        var constants = Aicode.ScalingConstants()
        m.outputs = [("y8", m.body.w8a8(m.argument("x"), codes: m.argument("w"), scale: m.argument("w_scale"), constants: &constants)),
                     ("y16", m.body.q64(m.argument("x"), codes: m.argument("c"), scale: m.argument("w_scale"), constants: &constants))]
        let f = try await Functions.load(try Functions.asset(m, "run-compact-\(rows)x\(k)x\(n)"))
        try await Run.call(f, bench.stream, inputs: ["x": View(buffer: arena.buffer(x), type: .float16, shape: [1, rows, k]),
                                                     "w": View(buffer: arena.buffer(w["w"]!.offset), type: .int8, shape: [count, n, piece]),
                                                     "c": View(buffer: arena.buffer(copy), type: .int8, shape: [1, k, n]),
                                                     "w_scale": View(buffer: arena.buffer(w["w_scale"]!.offset), type: .float16, shape: [1, 1, n])],
                           outputs: [("y8", View(buffer: arena.buffer(y8), type: .float16, shape: [1, rows, n])),
                                     ("y16", View(buffer: arena.buffer(y16), type: .float16, shape: [1, rows, n]))])

        let s16 = scales.map { Double(Float16($0)) }
        let exact = int8Product(xs, rows: rows, k: k, n: n, scales: s16) { o, i in codes[o * k + i] }
        let own = product(xs, rows: rows, k: k, n: n) { i, o in Double(codes[o * k + i]) * s16[o] }
        let (e8, e16) = (relative(arena.halves(y8, rows * n), exact), relative(arena.halves(y16, rows * n), own))
        print(String(format: "compact %d × %d → %d: int8 × int8 by pieces %.2e, q64 on the copy %.2e", rows, k, n, e8, e16))
        #expect(e8 < 2e-3, "W8A8 relative error \(e8)")
        #expect(e16 < 2e-3, "W8A16 relative error \(e16)")
    }

    /// The RMS norm with its gain: `x · rsqrt(mean(x²) + eps) · γ`, the statistics in f32, f16 at its borders.
    @Test(.enabled(if: metalEnabled))
    func aNormIsTheCPUs() async throws {
        let bench = try await Bench()
        let (rows, d, eps) = (64, 384, 1e-5)
        var rng = Seeded(state: 5)
        let arena = try Arena(bench.device, bytes: 2 * Arena.aligned(rows * d * 2) + Arena.aligned(d * 2))
        let x = arena.take(rows * d * 2), g = arena.take(d * 2), y = arena.take(rows * d * 2)
        let xs = arena.write(rng.uniform(rows * d, -4...4), at: x), gs = arena.write(rng.uniform(d, 0.5...1.5), at: g)

        var m = Aicode.Module(arguments: [("x", Aicode.tensor([rows, d], .f16)), ("g", Aicode.tensor([d], .f16))])
        m.outputs = [("y", m.body.binary(.mul, m.body.rmsNorm(m.argument("x"), axes: [1], eps: eps), m.argument("g")))]
        let f = try await Functions.load(try Functions.asset(m, "run-norm"))
        try await Run.call(f, bench.stream, inputs: ["x": View(buffer: arena.buffer(x), type: .float16, shape: [rows, d]), "g": View(buffer: arena.buffer(g), type: .float16, shape: [d])],
                           outputs: [("y", View(buffer: arena.buffer(y), type: .float16, shape: [rows, d]))])

        var want = [Float](repeating: 0, count: rows * d)
        for r in 0..<rows {
            let ms = (0..<d).reduce(0.0) { $0 + Double(xs[r * d + $1]) * Double(xs[r * d + $1]) } / Double(d)
            for i in 0..<d { want[r * d + i] = Float(Double(xs[r * d + i]) / (ms + eps).squareRoot() * Double(gs[i])) }
        }
        let e = relative(arena.halves(y, rows * d), want)
        #expect(e < 2e-3, "relative error \(e)")
    }

    /// A tiny attention, `softmax(q · kᵀ / √d) · v` over `[1, heads, rows, d]`, through Core AI's composite.
    @Test(.enabled(if: metalEnabled))
    func aTinyAttentionIsTheCPUs() async throws {
        let bench = try await Bench()
        let (heads, rows, keys, d) = (2, 32, 48, 64)
        let scale = 1 / Double(d).squareRoot()
        var rng = Seeded(state: 7)
        let (nq, nk) = (heads * rows * d, heads * keys * d)
        let arena = try Arena(bench.device, bytes: 2 * Arena.aligned(nq * 2) + 2 * Arena.aligned(nk * 2))
        let q = arena.take(nq * 2), k = arena.take(nk * 2), v = arena.take(nk * 2), y = arena.take(nq * 2)
        let qs = arena.write(rng.uniform(nq, -2...2), at: q), ks = arena.write(rng.uniform(nk, -2...2), at: k), vs = arena.write(rng.uniform(nk), at: v)

        var m = Aicode.Module(arguments: [("q", Aicode.tensor([1, heads, rows, d], .f16)), ("k", Aicode.tensor([1, heads, keys, d], .f16)),
                                          ("v", Aicode.tensor([1, heads, keys, d], .f16))])
        m.outputs = [("y", m.body.attention([m.argument("q"), m.argument("k"), m.argument("v")], scale: scale, symbol: "sdpa_run"))]
        let f = try await Functions.load(try Functions.asset(m, "run-attention"))
        func view(_ key: Int, _ n: Int) -> View { View(buffer: arena.buffer(key), type: .float16, shape: [1, heads, n, d]) }
        try await Run.call(f, bench.stream, inputs: ["q": view(q, rows), "k": view(k, keys), "v": view(v, keys)], outputs: [("y", view(y, rows))])

        var want = [Float](repeating: 0, count: nq)
        for h in 0..<heads {
            for r in 0..<rows {
                let scores = (0..<keys).map { j in (0..<d).reduce(0.0) { $0 + Double(qs[(h * rows + r) * d + $1]) * Double(ks[(h * keys + j) * d + $1]) } * scale }
                let top = scores.max()!, e = scores.map { Foundation.exp($0 - top) }, total = e.reduce(0, +)
                for i in 0..<d { want[(h * rows + r) * d + i] = Float((0..<keys).reduce(0.0) { $0 + e[$1] / total * Double(vs[(h * keys + $1) * d + i]) }) }
            }
        }
        let e = relative(arena.halves(y, nq), want)
        #expect(e < 5e-3, "relative error \(e)")
    }
}

/// A safetensors file of the given tensors, under `.build/run/`.
func weightsFile(_ name: String, _ tensors: [(String, String, [Int], [UInt8])]) throws -> String {
    var header: [String: Any] = [:], data: [UInt8] = [], at = 0
    for (key, dtype, shape, bytes) in tensors {
        header[key] = ["dtype": dtype, "shape": shape, "data_offsets": [at, at + bytes.count]]
        data += bytes; at += bytes.count
    }
    let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    let url = Functions.folder.appendingPathComponent(name)
    try Data(withUnsafeBytes(of: UInt64(json.count).littleEndian, Array.init) + [UInt8](json) + data).write(to: url)
    return url.path
}

extension Runs {
    /// **This process's specializations erased by their folder** (`Functions.eraseSpecializations(of:)`), nothing loaded:
    /// a function loaded and let go leaves its entry under the process's folder (`Functions.specializations`); erased,
    /// the folder is gone, and the same asset loads again, specialized anew, and computes. Not `AIModelCache.deleteAll`:
    /// on 20/11 it emptied every process's folder of the user's cache, other apps' included.
    @Test(.enabled(if: metalEnabled))
    func theProcesssSpecializationsAreErasedByTheirFolder() async throws {
        let bench = try await Bench()
        let shape = [64, 96], bytes = 64 * 96 * 2
        var m = Aicode.Module(arguments: [("x", Aicode.tensor(shape, .f16))])
        m.outputs = [("y", m.body.binary(.mul, m.argument("x"), m.argument("x")))]
        let asset = try Functions.asset(m, "run-erase")
        let name = ProcessInfo.processInfo.processName
        func entries() -> [String] {
            Functions.specializations(of: name).flatMap { (try? FileManager.default.contentsOfDirectory(atPath: $0.path)) ?? [] }
        }
        do { let f = try await Functions.load(asset); withExtendedLifetime(f) {} }
        #expect(!entries().isEmpty, "no entry under \(name)")
        #expect(Functions.eraseSpecializations(of: [name]) > 0)
        #expect(entries().isEmpty)

        let arena = try Arena(bench.device, bytes: 2 * Arena.aligned(bytes))
        let x = arena.take(bytes), y = arena.take(bytes)
        let xs = arena.write({ var r = Seeded(state: 11); return r.uniform(64 * 96) }(), at: x)
        let f = try await Functions.load(asset)
        try await Run.call(f, bench.stream, inputs: ["x": View(buffer: arena.buffer(x), type: .float16, shape: shape)],
                           outputs: [("y", View(buffer: arena.buffer(y), type: .float16, shape: shape))])
        let e = relative(arena.halves(y, 64 * 96), xs.map { $0 * $0 })
        #expect(e < 2e-3, "relative error \(e)")
        #expect(!entries().isEmpty, "the asset was not specialized again under \(name)")
    }

    /// **A compilation is told, a cached function is not** (`Functions.load`, `Stages.Event.compiling`/`compiled`): a
    /// function of bytes never seen (a constant drawn now) is specialized and said so once; loaded again, Core AI's cache
    /// hands it back unsaid, and both compute the same bits.
    @Test(.enabled(if: metalEnabled))
    func aCompilationIsToldAndACachedFunctionIsNot() async throws {
        let bench = try await Bench()
        let n = 4096, c = Double(UInt32.random(in: 1...1_000_000)) / 1_000_000
        var m = Aicode.Module(arguments: [("x", Aicode.tensor([n], .f16))])
        let b = m.body
        m.outputs = [("y", b.binary(.add, b.binary(.mul, m.argument("x"), b.scalar(c, .f16)), b.scalar(c, .f16)))]
        let url = try Functions.asset(m, try Functions.named(m, prefix: "run-compiled"))
        let told = Mutex<[String]>([])
        let observation = Stages.observe { e in
            switch e {
            case .compiling: told.withLock { $0.append("compiling") }
            case .compiled(let s): told.withLock { $0.append("compiled") }; print("compiled in \(String(format: "%.3f", s)) s")
            default: break
            }
        }
        defer { observation.cancel() }
        let arena = try Arena(bench.device, bytes: 3 * Arena.aligned(n * 2))
        let x = arena.take(n * 2), y1 = arena.take(n * 2), y2 = arena.take(n * 2)
        _ = arena.write((0..<n).map { Float($0 % 97) / 97 }, at: x)
        func run(_ f: InferenceFunction, _ y: Int) async throws {
            try await Run.call(f, bench.stream, inputs: ["x": View(buffer: arena.buffer(x), type: .float16, shape: [n])],
                               outputs: [("y", View(buffer: arena.buffer(y), type: .float16, shape: [n]))])
        }
        try await run(try await Functions.load(url), y1)
        #expect(told.withLock { $0 } == ["compiling", "compiled"])
        try await run(try await Functions.load(url), y2)
        #expect(told.withLock { $0 } == ["compiling", "compiled"], "the second load was told: \(told.withLock { $0 })")
        #expect(arena.halves(y1, n) == arena.halves(y2, n))
    }
}
