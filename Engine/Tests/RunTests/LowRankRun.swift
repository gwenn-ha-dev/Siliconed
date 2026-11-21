import CoreAI
import Foundation
import Metal
import SiliCore
import Testing

/// The low-rank path beside a product (`Aicode.Block.lowRank`), called on the GPU and judged against the CPU in f64:
/// `y + (x · down) · up`, the factors written by `LowRank.write`. The bars were written before it ran: the f16 path's
/// rounding, ~1e-3 relative on the change; in f32 (bf16 factors widened) ~1e-5 on the change, the sum rounded once to
/// f16 — an error of order 1 would be a wrong layout or a wrong axis.
extension Runs {
    @Test(.enabled(if: metalEnabled), arguments: [NDArray.ScalarType.float16, .bfloat16, .float32])
    func aLowRankPathIsTheCPUs(_ element: NDArray.ScalarType) async throws {
        let bench = try await Bench()
        let (rows, inn, out, r, R) = (64, 256, 192, 12, 32)
        var rng = Seeded(state: 11)
        let down = rng.uniform(r * inn, -0.1...0.1), up = rng.uniform(out * r, -0.5...0.5)
        let b = LowRank.bytes(inn: inn, out: out, rank: R, element: element)
        let arena = try Arena(bench.device, bytes: Arena.aligned(b.down) + Arena.aligned(b.up) + Arena.aligned(rows * inn * 2) + 2 * Arena.aligned(rows * out * 2))
        let d = arena.take(b.down), u = arena.take(b.up), x = arena.take(rows * inn * 2), y = arena.take(rows * out * 2), z = arena.take(rows * out * 2)
        try LowRank.write(down: down, up: up, rank: r, inn: inn, out: out, into: arena.pointer(d), arena.pointer(u), rank: R, element: element)
        let xs = arena.write(rng.uniform(rows * inn, -4...4), at: x), ys = arena.write(rng.uniform(rows * out), at: y)

        let e: Aicode.Element = element == .float16 ? .f16 : element == .bfloat16 ? .bf16 : .f32
        var m = Aicode.Module(arguments: [("y", Aicode.tensor([rows, out], .f16)), ("x", Aicode.tensor([rows, inn], .f16)),
                                          ("down", Aicode.tensor([inn, R], e)), ("up", Aicode.tensor([R, out], e))])
        m.outputs = [("z", m.body.lowRank(m.argument("y"), m.argument("x"), down: m.argument("down"), up: m.argument("up")))]
        let f = try await Functions.load(try Functions.asset(m, "run-low-rank-\(e)"))
        func view(_ key: Int, _ t: NDArray.ScalarType, _ shape: [Int]) -> View { View(buffer: arena.buffer(key), type: t, shape: shape) }
        try await Run.call(f, bench.stream, inputs: ["y": view(y, .float16, [rows, out]), "x": view(x, .float16, [rows, inn]),
                                                     "down": view(d, element, [inn, R]), "up": view(u, element, [R, out])],
                           outputs: [("z", view(z, .float16, [rows, out]))])

        // Against the factors as written (rounded to their element, balanced in f16): the path is judged, not the rounding.
        func read(_ key: Int, _ count: Int) -> [Double] {
            let p = arena.pointer(key)
            switch element {
            case .float16: return (0..<count).map { Double(p.load(fromByteOffset: 2 * $0, as: Float16.self)) }
            case .bfloat16: return (0..<count).map { Double(Float(bitPattern: UInt32(p.load(fromByteOffset: 2 * $0, as: UInt16.self)) << 16)) }
            default: return (0..<count).map { Double(p.load(fromByteOffset: 4 * $0, as: Float.self)) }
            }
        }
        let dw = read(d, inn * R), uw = read(u, R * out)
        let delta = product(xs, rows: rows, k: inn, n: out) { i, o in (0..<R).reduce(0.0) { $0 + dw[i * R + $1] * uw[$1 * out + o] } }
        let got = zip(arena.halves(z, rows * out), ys).map { $0 - $1 }
        let err = relative(got, delta)
        #expect(err < (element == .float16 ? 3e-3 : 2e-3), "\(element): the change's relative error \(err)")
    }
}
