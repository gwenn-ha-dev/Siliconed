import CoreAI
import Foundation
import Metal
import SiliCore

/// **Layer 0 alone, at its inner boundaries** (`siliconed check z-image block`): the render's block functions in the
/// map's arithmetic under the process's precision (`Arithmetic.of`: `ZImageBlock`, the Standard's;
/// `ZImageShared`, the Compact's, or the Standard's weights in f16 under `ane`, on the GPU), over the oracle's own
/// stream and adaln, one call over every row of the sequence (keys = rows, every key real: diffusers masks nothing at
/// batch 1), and before each the same function with boundaries for outputs (six at most a function). The stream given
/// as the functions take it: bf16 rounded to nearest under the Standard; under the Compact x̂ and its row factors as the
/// stream's guard brings them (`ranged16` under 2¹⁴, the factors within f16's normals), from the true values.
package enum ZImageLayerZero {
    typealias Tap = ZImageDiT.BlockTap

    /// Each function's boundaries, its taps by call: the Standard's three functions, the Compact's five.
    static func taps(_ form: ZImageFunctions.Form) -> [(function: String, taps: [Tap])] {
        form == .bf16
            ? [("modulation", [.adaln]), ("pre", [.norm1, .attentionIn, .q, .k, .v, .qNormed]), ("pre", [.kNormed]),
               ("post", [.sdpa, .attention, .norm2, .ffnNorm1, .ffnIn, .w1]), ("post", [.w3, .swiglu, .feedForward, .ffnNorm2])]
            : [("modulation", [.adaln]), ("pre", [.norm1, .attentionIn, .q, .k, .v]), ("heads", [.qNormed, .kNormed]), ("attention", [.sdpa]),
               ("out", [.attention, .norm2]), ("ffn", [.ffnNorm1, .ffnIn, .w1, .w3, .swiglu, .feedForward]), ("ffn", [.ffnNorm2])]
    }

    /// A step's name: its function and its boundaries.
    @Sendable static func step(_ f: String, _ taps: [Tap]) -> String { taps.isEmpty ? f : "\(f) " + taps.map(\.rawValue).joined(separator: " ") }

    static func width(_ t: Tap) -> Int {
        let c = ZImageDiT.Config.published
        return switch t {
        case .adaln: 4 * c.dim
        case .w1, .w3, .swiglu: c.hidden
        default: c.dim
        }
    }

    /// Each boundary's values for a latent `[16, h, w]` and a caption of `capLength` tokens.
    package static func tapCounts(latentHeight: Int, latentWidth: Int, capLength: Int) throws -> [ZImageDiT.BlockTap: Int] {
        let rows = try ZImageTransformer.Geometry(latentHeight: latentHeight, latentWidth: latentWidth, capLength: capLength).sequence
        return Dictionary(uniqueKeysWithValues: Tap.allCases.map { t in (t, t == .out ? rows * ZImageBlock.d : (t == .adaln ? 1 : rows) * width(t)) })
    }

    /// **Layer 0 over `x`** (`[rows, d]`, the oracle's unified stream) at `adaln` (`[256]`): every boundary written into
    /// `taps` (`tapCounts` values each: the caller's, made before the call, so that the stage's floor counts them), true
    /// values in f32, and what ran.
    package static func run(map path: String, x: [Float], adaln: [Float], latentHeight: Int, latentWidth: Int, capLength: Int,
                            taps: [ZImageDiT.BlockTap: UnsafeMutableBufferPointer<Float>]) throws -> String {
        nonisolated(unsafe) let taps = taps
        return try blocking { try await layerZero(path, x, adaln, latentHeight, latentWidth, capLength, taps) }
    }

    private static func layerZero(_ path: String, _ x: [Float], _ adaln: [Float], _ h: Int, _ w: Int, _ capLength: Int,
                                  _ tapped: [ZImageDiT.BlockTap: UnsafeMutableBufferPointer<Float>]) async throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { throw Failure("no Metal device") }
        let file = try ZImageDiT.file(path)
        var arithmetic = Arithmetic.of(codes: file.tensors["layers.0.attention.qkv.weight"]?.dtype == "I8", precision: CoreAI.inForce)
        // The Compact's int8 × int8 products where a control asks for them (`--products int8`): the block on the GPU.
        if case .shared(.codes, let share) = arithmetic, CoreAI.integerProducts == true { arithmetic = .shared(.integer, share) }
        let (form, weights) = (ZImageFunctions.Form(arithmetic), arithmetic.weights)
        let g = try ZImageTransformer.Geometry(latentHeight: h, latentWidth: w, capLength: capLength)
        let (rows, d, c) = (g.sequence, ZImageBlock.d, ZImageDiT.Config.published)
        precondition(x.count == rows * d && adaln.count == c.adalnDim, "x of \(x.count) values for \(rows) rows, adaln of \(adaln.count)")
        // A block of layer 0 alone over every row (keys = rows): each function after the same with its boundaries for
        // outputs (tensors of their names), a step each, in the order of `taps`.
        let shared = form == .shared, element: NDArray.ScalarType = shared ? .float16 : .bfloat16
        var n = Block.Declaration("Z-Image's layer 0 (\(form)\(shared ? ", \(weights)" : ""), \(rows) rows)", rows: rows)
        n.perRow = [.init("x", width: d, element, next: "y", factor: shared ? "x_factor" : nil),
                    .init("q", width: d, element), .init("k", width: d, element), .init("v", width: d, element),
                    .init("cos", width: ZImageBlock.hd, .float32), .init("sin", width: ZImageBlock.hd, .float32)]
        if shared {
            n.perRow += [.init("x_factor", width: 1, .float16, next: "y_factor"), .init("qkv", width: 3 * d, .float16, factor: "qkv_factor"),
                         .init("qkv_factor", width: 1, .float16), .init("o", width: d, .float16), .init("mid", width: d, .float16, factor: "mid_factor"),
                         .init("mid_factor", width: 1, .float16)]
        }
        let gains = shared ? ZImageShared.gains.map { ($0, d * 2) } : [("scales", 4 * ZImageTransformer.gainBytes), ("gates", 4 * ZImageTransformer.gainBytes)]
        n.tensors = ([("adaln", c.adalnDim * 4), ("mask", rows * 2)] + gains).map { SiliCore.Stage.Tensor($0.0, bytes: $0.1) }
            + Tap.allCases.filter { $0 != .out }.map { SiliCore.Stage.Tensor($0.rawValue, bytes: ($0 == .adaln ? 1 : rows) * width($0) * 4) }
        n.groups = [Block.Group("layers.0", form: "\(form)", weights: ZImageFunctions.entries("layers.0", form: form, weights: weights))]
        n.cycle = ["layers.0"]
        if shared { n.codes = weights.layout }
        let functions = shared ? ["modulation", "pre", "heads", "attention", "out", "ffn"] : ZImageBlock.functions
        let calls = functions.flatMap { f in Self.taps(form).filter { $0.function == f } + [(f, [])] }
        let steps = calls.filter { $0.function != "modulation" }
        // The shared forms' bindings (`ZImageTransformer`'s): heads read the activations `qkv`, out writes `mid`, ffn reads it.
        let binds: [String: [String: Block.Bind]] = ["heads": ["qkv": .rows("qkv")], "out": ["y": .rows("mid"), "y_factor": .rows("mid_factor")],
                                                     "ffn": ["x": .rows("mid"), "x_factor": .rows("mid_factor")]]
        n.forms = [Block.Form("\(form)", steps: steps.map { f, taps in
            Block.Step(step(f, taps), on: .gpu, whole: ["post", "attention"].contains(f) ? ["k", "v"] : [], bind: shared ? binds[f] ?? [:] : [:])
        }, bind: shared ? [:] : ["scale1": .row("scales", 0), "gate1": .row("gates", 1), "scale2": .row("scales", 2), "gate2": .row("gates", 3)]) { name, b in
            let (f, taps) = steps.first { step($0.function, $0.taps) == name }!
            return shared ? ZImageShared.module(f, rows: b.rows, keys: rows, weights: weights, taps: taps) : ZImageBlock.module(f, rows: b.rows, keys: rows, form: form, codes: b.codes, taps: taps)
        }]
        n.passes = [Block.Pass("rows", rows: 0..<rows)]
        for (f, taps) in calls where f == "modulation" {
            n.edges[step(f, taps)] = { b in shared ? ZImageShared.modulation(weights: weights, taps: taps) : ZImageBlock.module(f, rows: 1, keys: 1, form: form, codes: b.codes, taps: taps) }
        }
        let block = try await Block.open(device, n, files: [file], budget: MemoryBudget.machineCeiling(), share: .gpu)
        defer { block.close() }

        // The inputs: the stream, adaln, the RoPE tables, the mask all zeros.
        if form == .bf16 {
            let p = await block.rows("x").bindMemory(to: UInt16.self, capacity: rows * d)
            for i in 0..<rows * d { p[i] = BFloat16Bits.rounded(x[i]) }
        } else {
            let p = await block.rows("x").bindMemory(to: Float16.self, capacity: rows * d), f = await block.rows("x_factor").bindMemory(to: Float16.self, capacity: rows)
            for r in 0..<rows {
                let top = max((0..<d).reduce(Float(0)) { max($0, abs(x[r * d + $1])) }, 0x1p-14)
                let factor = Float(sign: .plus, exponent: RowExponent.ranged(largest: Double(top), f32Factor: false), significand: 1)
                f[r] = Float16(factor)
                for i in 0..<d { p[r * d + i] = Float16(x[r * d + i] / factor) }
            }
        }
        let given = await block.pointer("adaln")
        _ = adaln.withUnsafeBytes { memcpy(given, $0.baseAddress!, $0.count) }
        let rope = g.dit.rope(rows: rows)
        let (cos, sin) = (await block.rows("cos"), await block.rows("sin"))
        _ = rope.cos.withUnsafeBytes { memcpy(cos, $0.baseAddress!, $0.count) }
        _ = rope.sin.withUnsafeBytes { memcpy(sin, $0.baseAddress!, $0.count) }

        // Each function, then its boundaries on the same inputs; the block's own output, a true value.
        try await block.run("layers.0", "rows") {
            for (f, taps) in calls where f == "modulation" { try await block.call(step(f, taps)) }
        }
        for (_, taps) in calls {
            for t in taps {
                let count = (t == .adaln ? 1 : rows) * width(t)
                guard let into = tapped[t] else { continue }
                precondition(into.count == count, "\(t): \(into.count) values for \(count)")
                memcpy(into.baseAddress!, await block.pointer(t.rawValue), count * 4)
            }
        }
        if let into = tapped[.out] {
            let y = await block.values("x", rows: 0..<rows)
            for i in 0..<rows * d { into[i] = y[i] }
        }
        return "\(form)\(shared ? " on \(weights)" : ""), codes \(block.stage.plan.codes.rawValue), " + block.stage.plan.line
    }
}
