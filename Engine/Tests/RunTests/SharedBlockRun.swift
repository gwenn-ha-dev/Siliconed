import CoreAI
import Foundation
import Metal
import SiliCore
import Testing

/// **A block shared with the ANE computes what the GPU alone does**: a layer of four steps (`pre` and `out` either unit's,
/// `heads` and `attention` the GPU's), 256 rows, the ANE told 128 of them, its stream made by an edge of the GPU's as a
/// family's is, against the same block on the GPU alone and the CPU's in f64. The bar was written before the run: within
/// f16's rounding (1e-3 relative). Its first run, 21/11 on the mini, read the GPU's rows of the stream as zeros (2.3 %):
/// a buffer Core AI saw for the first time (`Block.encode`).
extension Runs {
    @Test(.enabled(if: metalEnabled))
    func aSharedBlockIsTheGPUAlones() async throws {
        let bench = try await Bench()
        let measured = FileManager.default.temporaryDirectory.appendingPathComponent("this-mac-shared-\(UUID().uuidString).json")
        let before = ThisMac.file
        ThisMac.file = measured
        defer { ThisMac.file = before; try? FileManager.default.removeItem(at: measured) }

        let (rows, heads, hd) = (256, 4, 64)
        let d = heads * hd
        var rng = Seeded(state: 31)
        let x = rng.uniform(rows * d, -2...2)
        // A group's weights come from a file: one the functions never read.
        let file = try SafeTensors(try weightsFile("shared-block.safetensors", [("L.w", "BF16", [64, 64], [UInt8](repeating: 0, count: 64 * 64 * 2))]))
        func run(_ share: Block.Share) async throws -> [Float] {
            let block = try await Block.open(bench.device, Self.sharedDeclaration(rows: rows, heads: heads, hd: hd), files: [file], budget: 1 << 40, share: share)
            defer { block.close() }
            let p = await block.rows("input").bindMemory(to: Float16.self, capacity: rows * d)
            for i in 0..<rows * d { p[i] = Float16(x[i]) }
            try await block.call("enter")
            try await block.run("L", "p")
            return await block.values("y", rows: 0..<rows)
        }
        // The CPU's, in f64 from the f16 inputs: h = x / 2, q = v = h, k = h / 4, y = x + softmax(q kᵀ / √hd) v by head.
        let x16 = x.map { Double(Float(Float16($0))) }
        var want = [Float](repeating: 0, count: rows * d)
        for h in 0..<heads {
            for i in 0..<rows {
                let scores = (0..<rows).map { j in (0..<hd).reduce(0.0) { $0 + x16[i * d + h * hd + $1] / 2 * x16[j * d + h * hd + $1] / 8 } / Double(hd).squareRoot() }
                let top = scores.max()!, e = scores.map { Foundation.exp($0 - top) }, sum = e.reduce(0, +)
                for c in 0..<hd { want[i * d + h * hd + c] = Float(x16[i * d + h * hd + c] + (0..<rows).reduce(0.0) { $0 + e[$1] / sum * x16[$1 * d + h * hd + c] / 2 }) }
            }
        }
        let alone = try await run(.gpu), shared = try await run(.told(128))
        let (e, cpu) = (relative(shared, alone), relative(alone, want))
        print("shared block: relative \(e) against the GPU alone, the GPU alone \(cpu) against the CPU")
        #expect(e < 1e-3 && cpu < 1e-3, "relative \(e), \(cpu)")
    }

    /// `pre` halves x into h (either unit); `heads` writes q, k, v of h; `attention` over every key (k and v whole) writes
    /// o; `out` adds o to x (either unit). `enter` makes the stream x of the CPU's input.
    static func sharedDeclaration(rows: Int, heads: Int, hd: Int) -> Block.Declaration {
        let d = heads * hd
        var b = Block.Declaration("run-shared-block", rows: rows)
        b.perRow = ["input", "x", "h", "q", "k", "v", "o", "y"].map { .init($0, width: d, .float16) }
        b.edges["enter"] = { _ in
            var m = Aicode.Module(arguments: [("input", Aicode.tensor([rows, d], .f16))])
            m.outputs = [("x", m.body.binary(.mul, m.argument("input"), m.body.scalar(1, .f16)))]
            return m
        }
        b.groups = [Block.Group("L", form: "s", weights: [("w", "L.w", .bfloat16(joined: []))])]
        b.cycle = ["L"]
        b.cuts = false
        b.passes = [Block.Pass("p", rows: 0..<rows, shareable: 0..<rows)]
        b.forms = [Block.Form("s", steps: [.init("pre"), .init("heads", on: .gpu), .init("attention", on: .gpu, whole: ["k", "v"]), .init("out")]) { step, build in
            let r = build.rows
            switch step {
            case "pre":
                var m = Aicode.Module(arguments: [("x", Aicode.tensor([1, r, d], .f16))])
                m.outputs = [("h", m.body.binary(.mul, m.argument("x"), m.body.scalar(0.5, .f16)))]
                return m
            case "heads":
                var m = Aicode.Module(arguments: [("h", Aicode.tensor([r, d], .f16))])
                let f = m.body, h = m.argument("h"), tokens = f.reshape(h, [1, r, heads, hd])
                m.outputs = [("q", tokens), ("k", f.reshape(f.binary(.mul, h, f.scalar(0.25, .f16)), [1, r, heads, hd])), ("v", tokens)]
                return m
            case "attention":
                var m = Aicode.Module(arguments: [("q", Aicode.tensor([1, r, heads, hd], .f16)), ("k", Aicode.tensor([1, rows, heads, hd], .f16)),
                                                  ("v", Aicode.tensor([1, rows, heads, hd], .f16))])
                let f = m.body
                let o = f.attention([f.transpose(m.argument("q"), [0, 2, 1, 3]), f.transpose(m.argument("k"), [0, 2, 1, 3]), f.transpose(m.argument("v"), [0, 2, 1, 3])],
                                    scale: 1 / Double(hd).squareRoot(), symbol: "sdpa_shared")
                m.outputs = [("o", f.reshape(f.transpose(o, [0, 2, 1, 3]), [r, d]))]
                return m
            default:
                var m = Aicode.Module(arguments: [("x", Aicode.tensor([1, r, d], .f16)), ("o", Aicode.tensor([1, r, d], .f16))])
                m.outputs = [("y", m.body.binary(.add, m.argument("x"), m.argument("o")))]
                return m
            }
        }]
        return b
    }
}
