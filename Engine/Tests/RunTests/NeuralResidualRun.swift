import CoreAI
import Foundation
import Metal
import SiliCore
import Testing

extension Runs {
    /// The shared forms' residual with Δ under a product of factors (`residual16(_:_:plus:factors:)`): x̂ X + Δ̂ ∏ Gᵢ
    /// against the CPU in f64, the factors' product leaving f16 on the way (2¹⁴ · 2¹² past 2¹⁵) where G does not (2⁻⁷
    /// to 2⁻¹⁰), the stream's factor from 2⁻⁸ to 2⁷ a row: either term the larger, by rows. The bar, set before it ran:
    /// f16's rounding of the sum, ~1e-3 relative; the value back within [2¹³, 2¹⁴] a row.
    @Test(.enabled(if: metalEnabled))
    func aResidualUnderFactorsIsTheCPUs() async throws {
        let bench = try await Bench()
        let (rows, n) = (16, 256)
        var rng = Seeded(state: 11)
        let rowBytes = Arena.aligned(rows * 2), valueBytes = Arena.aligned(rows * n * 2)
        let arena = try Arena(bench.device, bytes: 4 * valueBytes + 6 * rowBytes)
        let x = arena.take(rows * n * 2), xf = arena.take(rows * 2), delta = arena.take(rows * n * 2), y = arena.take(rows * n * 2), yf = arena.take(rows * 2)
        let g = (0..<4).map { _ in arena.take(rows * 2) }
        let xs = arena.write((0..<rows * n).map { _ in Float((rng.next() < 0.5 ? -1 : 1) * (0x1p13 + rng.next() * 0x1p13)) }, at: x)
        let xfs = arena.write((0..<rows).map { Float(Foundation.pow(2, Double($0 - 8))) }, at: xf)
        let ds = arena.write(rng.uniform(rows * n, -60000...60000), at: delta)
        let gs = [arena.write([Float](repeating: 0x1p14, count: rows), at: g[0]), arena.write([Float](repeating: 0x1p12, count: rows), at: g[1]),
                  arena.write([Float](repeating: 0x1p-13, count: rows), at: g[2]),
                  arena.write((0..<rows).map { Float(Foundation.pow(2, Double(-10 - $0 % 4))) }, at: g[3])]

        var m = Aicode.Module(arguments: [("x", Aicode.tensor([1, rows, n], .f16)), ("x_factor", Aicode.tensor([1, rows, 1], .f16)),
                                          ("delta", Aicode.tensor([1, rows, n], .f16))] + (0..<4).map { ("g\($0)", Aicode.tensor([1, rows, 1], .f16)) })
        let (value, factor) = m.body.residual16(m.argument("x"), m.argument("x_factor"), plus: m.argument("delta"), factors: (0..<4).map { m.argument("g\($0)") })
        m.outputs = [("y", value), ("y_factor", factor)]
        let f = try await Functions.load(try Functions.asset(m, "run-residual-factors"))
        func view(_ key: Int, _ shape: [Int]) -> View { View(buffer: arena.buffer(key), type: .float16, shape: shape) }
        try await Run.call(f, bench.stream, inputs: ["x": view(x, [1, rows, n]), "x_factor": view(xf, [1, rows, 1]), "delta": view(delta, [1, rows, n]),
                                                     "g0": view(g[0], [1, rows, 1]), "g1": view(g[1], [1, rows, 1]), "g2": view(g[2], [1, rows, 1]),
                                                     "g3": view(g[3], [1, rows, 1])],
                           outputs: [("y", view(y, [1, rows, n])), ("y_factor", view(yf, [1, rows, 1]))])

        let got = arena.halves(y, rows * n), gotFactor = arena.halves(yf, rows)
        var true_ = [Float](), want = [Float]()
        for r in 0..<rows {
            let G = gs.reduce(1.0) { $0 * Double($1[r]) }
            for i in 0..<n {
                true_.append(Float(Double(got[r * n + i]) * Double(gotFactor[r])))
                want.append(Float(Double(xs[r * n + i]) * Double(xfs[r]) + Double(ds[r * n + i]) * G))
            }
            let peak = (0..<n).map { abs(got[r * n + $0]) }.max()!
            #expect(peak > 0x1p13 && peak <= 0x1p14, "row \(r): its peak \(peak)")
        }
        let e = relative(true_, want)
        #expect(e < 2e-3, "relative error \(e)")
    }
}
