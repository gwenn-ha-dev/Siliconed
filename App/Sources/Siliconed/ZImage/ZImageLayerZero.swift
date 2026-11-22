import CoreAI
import Foundation
import Metal
import SiliCore

/// **Layer 0 alone, at its inner boundaries** (`siliconed check z-image block`): the render's own declaration of layer 0
/// (`ZImageTransformer.declaration`, `Part.tapped`) in the map's arithmetic under the process's Neural Engine switch
/// (`Arithmetic.of`: `ZImageBlock`, the Standard's; `ZImageShared`, the Compact's, or the Standard's weights in f16 with
/// the switch on), run on the GPU alone over the sequence's pass from the oracle's own stream and adaln — the render's
/// rows, keys and mask (the keys past the sequence masked, diffusers masking none at batch 1) —, each of its steps after
/// the same function with boundaries for outputs (six at most a function). The stream given as the functions take it:
/// bf16 rounded to nearest under the Standard; under the shared forms x̂ and its row factors as the stream's guard brings
/// them (`ranged16` under 2¹⁴, the factors within f16's normals), from the true values.
package enum ZImageLayerZero {
    typealias Tap = ZImageDiT.BlockTap

    /// Each function's boundaries, its taps by call: the Standard's three functions, the shared forms' five.
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

    // MARK: - The render's layer, tapped

    /// **Layer 0's declaration tapped** (`ZImageTransformer.Part.tapped`): the render's form of `n` again, each of its
    /// steps after its function with boundaries for outputs (`taps`) — the GPU's, on the step's own bindings and whole
    /// tensors —, the modulation's with its boundary an edge beside the modulation; each boundary a per-row tensor of its
    /// name, f32, written by each call at its rows as the stream is (a tensor of the whole stage was written whole by each
    /// call, the engine's second call over the last rows of the first: 1056 rows in two calls, 17 boundaries of 21 red,
    /// 21/11), the modulation's a tensor. Layer 0 runs the form so made (`run`'s `tapped`).
    static func tap(_ n: inout Block.Declaration, form: ZImageFunctions.Form, weights: SharedWeights, keys: Int, ranks: [String: Int]) {
        let plain = n.forms[0], calls = taps(form)
        let tapped = Dictionary(uniqueKeysWithValues: calls.map { (step($0.function, $0.taps), $0) })
        let steps = plain.steps.flatMap { s in
            calls.filter { $0.function == s.name }.map { Block.Step(step(s.name, $0.taps), on: .gpu, whole: s.whole, bind: s.bind) } + [s]
        }
        n.forms.append(Block.Form(plain.name + "-tapped", steps: steps, bind: plain.bind) { name, b in
            guard let call = tapped[name] else { return try plain.module(name, b) }
            return form == .shared ? ZImageShared.module(call.function, rows: b.rows, keys: keys, weights: weights, ranks: ranks, taps: call.taps)
                : ZImageBlock.module(call.function, rows: b.rows, keys: keys, form: form, ranks: ranks, taps: call.taps)
        })
        n.perRow += Tap.allCases.filter { ![.out, .adaln].contains($0) }.map { Block.Rows($0.rawValue, width: width($0), .float32) }
        n.tensors.append(SiliCore.Stage.Tensor(Tap.adaln.rawValue, bytes: width(.adaln) * 4))
        for (f, taps) in calls where f == "modulation" {
            n.edges[step(f, taps)] = { _ in
                form == .shared ? ZImageShared.modulation(weights: weights, ranks: ranks, taps: taps)
                    : ZImageBlock.module(f, rows: 1, keys: 1, form: form, ranks: ranks, taps: taps)
            }
        }
        n.tapped = true
    }

    /// **The inputs of layer 0's pass in a check's block** (`ZImageTransformer.Part`): the oracle's stream `x` over the
    /// sequence's rows, as the functions take it; `adaln`; the RoPE tables and the sequence's mask over the keys, as
    /// `ZImageTransformer.prompt` writes them.
    static func give(_ block: Block, _ g: ZImageTransformer.Geometry, _ form: ZImageFunctions.Form, x: [Float], adaln: [Float]) async {
        let (rows, d, k) = (g.sequence, ZImageBlock.d, g.keys)
        if form == .bf16 {
            let p = await block.rows("x").bindMemory(to: UInt16.self, capacity: rows * d)
            for i in 0..<rows * d { p[i] = BFloat16Bits.rounded(x[i]) }
        } else {
            let p = await block.rows("x").bindMemory(to: Float16.self, capacity: rows * d), f = await block.rows("x_factor").bindMemory(to: Float16.self, capacity: rows)
            for r in 0..<rows {
                let (unit, factor) = RowExponent.split(x[(r * d)..<((r + 1) * d)], .ranged(f32Factor: false))
                f[r] = Float16(factor)
                for i in 0..<d { p[r * d + i] = Float16(unit[i]) }
            }
        }
        let given = await block.pointer("adaln")
        _ = adaln.withUnsafeBytes { memcpy(given, $0.baseAddress!, $0.count) }
        let rope = g.dit.rope(rows: k)
        let (cos, sin) = (await block.rows("cos"), await block.rows("sin"))
        _ = rope.cos.withUnsafeBytes { memcpy(cos, $0.baseAddress!, $0.count) }
        _ = rope.sin.withUnsafeBytes { memcpy(sin, $0.baseAddress!, $0.count) }
        let masked: UInt16 = form == .bf16 ? 0xc61c : Float16(-1e4).bitPattern
        let m = await block.pointer("mask-sequence").assumingMemoryBound(to: UInt16.self)
        for j in 0..<k { m[j] = j < g.sequence ? 0 : masked }
    }

    /// Layer 0 over the sequence's pass, its modulation (and that with its boundary, `tapped`) before it, as the render
    /// runs it (`ZImageTransformer.evaluate`).
    static func run(_ block: Block, form: ZImageFunctions.Form, tapped: Bool) async throws {
        try await block.run("layers.0", "sequence", before: {
            if tapped { for (f, taps) in taps(form) where f == "modulation" { try await block.call(step(f, taps)) } }
            try await block.call("modulation")
        }, tapped: tapped)
    }

    // MARK: - The check

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
        // The products a render takes, on the GPU (a tapped run's).
        let arithmetic = Arithmetic.of(codes: Arithmetic.codes(file, marker: ZImageDiT.codesMarker), neuralEngine: CoreAI.neuralEngineInForce)
        let (form, weights) = (ZImageFunctions.Form(arithmetic), arithmetic.weights)
        let g = try ZImageTransformer.Geometry(latentHeight: h, latentWidth: w, capLength: capLength)
        let (rows, d, c) = (g.sequence, ZImageBlock.d, ZImageDiT.Config.published)
        precondition(x.count == rows * d && adaln.count == c.adalnDim, "x of \(x.count) values for \(rows) rows, adaln of \(adaln.count)")
        let block = try await Block.open(device, ZImageTransformer.declaration(g, arithmetic, part: .tapped), files: [file],
                                         budget: MemoryBudget.machineCeiling(), share: .gpu)
        defer { block.close() }
        await give(block, g, form, x: x, adaln: adaln)
        try await run(block, form: form, tapped: true)

        for t in Tap.allCases where t != .out {
            let count = (t == .adaln ? 1 : rows) * width(t)
            guard let into = tapped[t] else { continue }
            precondition(into.count == count, "\(t): \(into.count) values for \(count)")
            memcpy(into.baseAddress!, t == .adaln ? await block.pointer(t.rawValue) : await block.rows(t.rawValue), count * 4)
        }
        // The block's own output, a true value: the stream as the pass left it.
        if let into = tapped[.out] {
            let y = await block.values("x", rows: 0..<rows)
            for i in 0..<rows * d { into[i] = y[i] }
        }
        return "\(form)\(form == .shared ? " on \(weights)" : ""), " + block.stage.plan.line
    }
}
