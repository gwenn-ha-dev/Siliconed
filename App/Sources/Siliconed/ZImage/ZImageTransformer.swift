import CoreAI
import Foundation
import Metal
import SiliCore

/// **Z-Image's DiT entire, and its loop**: its functions at fixed shapes (`ZImageFunctions`, `ZImageBlock`,
/// `ZImageShared`), declared to the engine as a block (`Block`: its stage under B, its calls bound and put ahead, the
/// ANE's share), the loop in Swift.
///
///     once a prompt     embed-caption · context_refiner ×2 (the block unmodulated: scales and gates 1), kept aside
///     an evaluation     timestep · embed-image · noise_refiner ×2 on the image's rows · the caption put back ·
///                       layers ×30 on the sequence · final
///
/// **The rows**: `[image | its padding to 32 | caption | its padding to 32]`, every row a key at batch 1, the pad tokens
/// included; then up to the keys' palier, rows computed and masked (−10⁴). The palier is the image's rows + a caption's
/// palier (`captions`), so that no prompt up to it asks for a new shape. Each pass covers its rows (`Block`'s calls):
/// the noise refiner the image's, the context refiner the caption's, the layers the sequence's.
///
/// **The residual stream** `x` is two buffers in turn, beside its row factors in the shared forms (f16 one a row; none
/// in bf16): a step reads one and writes the other (`y`).
///
/// **The Compact** is the block in its shared forms (`ZImageShared`): on the GPU alone (the Neural Engine off), or with the ANE's share
/// (`Arithmetic.shared`), the ANE units of the image's first rows in the noise refiner's and the layers' passes
/// (`Block.Share`: a control's rows, or found in flight). **The Standard beside the ANE** is the same forms on its bf16
/// weights rounded to f16 as they are read (`SharedWeights.half`); on the GPU alone, its own bf16 functions.
///
/// Measured 17/11 on the M1 Pro (Z Compact at 1024², a caption of 48 tokens, tiles 1024 + 64, every weight kept): an
/// evaluation 17.21 s on the GPU alone (0.92 × the engine before); P = floor + arena + the functions' shares 1.04 × the
/// peak. Under B = 4, 11 blocks kept and 21 through a ring of 2, 1.003 × the time at B = 10, the velocities the same to
/// the bit.
final class ZImageTransformer {
    struct Geometry: Equatable, Sendable {
        /// The latent's, the image's rows (`h/2 · w/2`) and padded to 32, the caption's length and padded, the keys'.
        let height, width, tokens, image, capLength, capPadded, keys: Int
        /// The caption's palier: its rows in the arena, embed-caption's shape.
        var caption: Int { keys - image }
        var sequence: Int { image + capPadded }
        var dit: ZImageDiT { ZImageDiT(config: .published, latentHeight: height, latentWidth: width, capLength: capLength) }

        /// The keys: the image's rows and the first caption palier the caption fits (`captions`).
        init(latentHeight: Int, latentWidth: Int, capLength: Int) throws {
            let dit = ZImageDiT(config: .published, latentHeight: latentHeight, latentWidth: latentWidth, capLength: capLength)
            (height, width, tokens, image, self.capLength, capPadded) = (latentHeight, latentWidth, dit.imageTokens, dit.imagePadded, capLength, dit.capPadded)
            guard let palier = captions.first(where: { $0 >= dit.capPadded }) else { throw EngineError.promptTooLong(tokens: capLength, max: captions.last! - 32) }
            self.keys = image + palier
        }
    }

    /// **The caption's paliers**: 512 rows, and 1088 for the longest prompt a render admits (1024 tokens and the
    /// template's 8, padded to 1056): a multiple of 64, so that a call over the sequence, rounded up to 64, never passes
    /// the keys.
    static let captions = [512, 1088]

    static var d: Int { ZImageFunctions.d }
    static var c: ZImageDiT.Config { .published }
    var c: ZImageDiT.Config { .published }

    let g: Geometry
    let arithmetic: Arithmetic
    let block: Block
    /// The ranks of the LoRAs' low-rank paths, by product (`ZImageDiT.loraRanks`); none without a LoRA.
    let ranks: [String: Int]
    var form: ZImageFunctions.Form { ZImageFunctions.Form(arithmetic) }
    var stage: SiliCore.Stage { block.stage }
    private var prompted = false

    static let allBlocks = ZImageDiT.blocks()
    /// An evaluation's blocks in the order the loop runs them; the context refiner's run once a prompt, before.
    static let cycle = (0..<c.refiners).map { "noise_refiner.\($0)" } + (0..<c.layers).map { "layers.\($0)" }
    static let captionBlocks = (0..<c.refiners).map { "context_refiner.\($0)" }

    // MARK: - The functions

    static func module(_ f: String, _ g: Geometry, _ form: ZImageFunctions.Form, _ weights: SharedWeights,
                       ranks: [String: Int] = [:]) -> Aicode.Module {
        if f == "modulation" {
            return form == .shared ? ZImageShared.modulation(weights: weights, ranks: ranks) : ZImageBlock.module(f, rows: 1, keys: 1, form: form, ranks: ranks)
        }
        // embed-image's rows, the image's; embed-caption's, its palier.
        return ZImageFunctions.module(f, height: g.height, width: g.width, rows: f == "embed-caption" ? g.caption : g.image, form: form, ranks: ranks)
    }

    // MARK: - The low-rank paths

    /// The factors' element of the edges' function `f` and of the blocks': the timestep's f32, the form's otherwise.
    static func lowRankType(_ f: String?, _ form: ZImageFunctions.Form) -> NDArray.ScalarType {
        f == "timestep" ? .float32 : form == .bf16 ? .bfloat16 : .float16
    }

    /// The edges' adapted products by the function computing each, and every block's.
    static var edgeFunctions: [(function: String, products: [LoRAProduct])] {
        let all = ZImageDiT.loraProducts(nil)
        return ZImageFunctions.adapted.sorted { $0.key < $1.key }.map { f, names in (f, all.filter { names.contains($0.name) }) }
    }

    /// **The factors' tensors**: the edges', each block's under its prefix (`<block>.`, the block's own: `Block`) —
    /// surfaces where the ANE reads them, by the block.
    static func lowRankTensors(_ ranks: [String: Int], _ form: ZImageFunctions.Form) -> [SiliCore.Stage.Tensor] {
        guard !ranks.isEmpty else { return [] }
        return edgeFunctions.flatMap { LoRAStack.tensors($0.products, ranks: ranks, element: lowRankType($0.function, form)) }
            + allBlocks.flatMap { LoRAStack.tensors(ZImageDiT.loraProducts($0), ranks: ranks, element: lowRankType(nil, form), prefix: "\($0).") }
    }

    /// **The stack's factors written into the stage's tensors** (`lowRankTensors`), before the first call.
    func writeLowRank(_ loras: LoRAStack) throws {
        guard !ranks.isEmpty else { return }
        for (f, products) in Self.edgeFunctions {
            try loras.write(products, on: .zImage, ranks: ranks, element: Self.lowRankType(f, form)) { stage.pointer($0) }
        }
        for block in Self.allBlocks {
            try loras.write(ZImageDiT.loraProducts(block), on: .zImage, ranks: ranks, element: Self.lowRankType(nil, form), prefix: "\(block).") { stage.pointer($0) }
        }
    }

    // MARK: - What the DiT needs

    /// A row of the gains', on a page of its own (`ZImageBlock.gainRow`).
    static var gainBytes: Int { ZImageBlock.gainRow * 2 }

    /// **The DiT's declaration**: its per-row tensors (the stream, its factors, q k v, the RoPE tables; in the shared
    /// forms the steps' own), its other tensors, the edges' weights read once, a group a block, the order the loop uses
    /// them in (the context refiner's once a prompt, then the cycle at every evaluation), its forms and passes.
    static func declaration(_ g: Geometry, _ arithmetic: Arithmetic, ranks: [String: Int] = [:]) -> Block.Declaration {
        let (d, cap, form, weights) = (Self.d, g.caption, ZImageFunctions.Form(arithmetic), arithmetic.weights), latent = c.channels * g.height * g.width * 4
        let element: NDArray.ScalarType = form == .bf16 ? .bfloat16 : .float16, factor = NDArray.ScalarType.float16
        var n = Block.Declaration("Z-Image's DiT (\(form), \(g.height)×\(g.width), \(g.keys) keys)", rows: g.keys)
        n.perRow = [.init("x", width: d, element, next: "y", factor: form == .bf16 ? nil : "x_factor"),
                    .init("q", width: d, element), .init("k", width: d, element), .init("v", width: d, element),
                    .init("cos", width: ZImageBlock.hd, .float32), .init("sin", width: ZImageBlock.hd, .float32)]
        if form != .bf16 { n.perRow.append(.init("x_factor", width: 1, factor, next: "y_factor")) }
        if form == .shared {
            n.perRow += [.init("qkv", width: 3 * d, .float16, factor: "qkv_factor"), .init("qkv_factor", width: 1, .float16), .init("o", width: d, .float16),
                         .init("mid", width: d, .float16, factor: "mid_factor"), .init("mid_factor", width: 1, .float16)]
        }
        let t: [(String, Int)] = [("latent", latent), ("t_freq", c.adalnDim * 4), ("adaln", c.adalnDim * 4), ("final_scale", d * 4),
                                  ("caps", cap * c.capDim * 4), ("real", cap * 4), ("mask-image", g.keys * 2), ("mask-caption", g.keys * 2),
                                  ("mask-sequence", g.keys * 2), ("saved", cap * d * 2), ("saved-factors", cap * Layout.element(factor))]
        let gains = form == .shared ? (ZImageShared.gains + ["one"]).map { ($0, d * 2) } : ["scales", "gates", "ones"].map { ($0, 4 * gainBytes) }
        n.tensors = (t + gains).map { SiliCore.Stage.Tensor($0.0, bytes: $0.1) } + [SiliCore.Stage.Tensor("velocity", bytes: latent, .apart)]
            + lowRankTensors(ranks, form)
        n.residents = ZImageFunctions.others(form: form)
        let name = "\(form)"
        n.groups = allBlocks.map { Block.Group($0, form: name, weights: ZImageFunctions.entries($0, form: form, weights: weights)) }
        (n.once, n.cycle) = (captionBlocks, cycle)
        let keys = g.keys
        // The gains: the shared forms' each a tensor (the ANE reads them so), the others' a row each of four.
        let ones: [String: Block.Bind]
        if form == .shared {
            n.forms = [Block.Form(name, steps: [.init("pre"), .init("heads", on: .gpu, bind: ["qkv": .rows("qkv")]),
                                                .init("attention", on: .gpu, whole: ["k", "v"]),
                                                .init("out", bind: ["y": .rows("mid"), "y_factor": .rows("mid_factor")]),
                                                .init("ffn", bind: ["x": .rows("mid"), "x_factor": .rows("mid_factor")])]) { step, b in
                ZImageShared.module(step, rows: b.rows, keys: keys, weights: weights, ranks: ranks, neural: b.neural)
            }]
            ones = Dictionary(uniqueKeysWithValues: ZImageShared.gains.map { ($0, .tensor("one")) })
        } else {
            func gains(_ s: String, _ g: String) -> [String: Block.Bind] { ["scale1": .row(s, 0), "gate1": .row(g, 1), "scale2": .row(s, 2), "gate2": .row(g, 3)] }
            n.forms = [Block.Form(name, steps: [.init("pre", on: .gpu), .init("post", on: .gpu, whole: ["k", "v"])],
                                  bind: gains("scales", "gates")) { step, b in
                ZImageBlock.module(step, rows: b.rows, keys: keys, form: form, ranks: ranks)
            }]
            ones = gains("ones", "ones")
        }
        n.passes = [Block.Pass("image", rows: 0..<g.image, shareable: 0..<g.image, bind: ["mask": .tensor("mask-image")]),
                    Block.Pass("caption", rows: g.image..<g.sequence, bind: ones.merging(["mask": .tensor("mask-caption")]) { a, _ in a }),
                    Block.Pass("sequence", rows: 0..<g.sequence, shareable: 0..<g.image, bind: ["mask": .tensor("mask-sequence")])]
        for f in ["timestep", "embed-image", "embed-caption", "modulation", "final"] {
            n.edges[f] = { b in module(f, g, form, weights, ranks: ranks) }
        }
        return n
    }

    // MARK: - The block

    /// **A render's DiT** of `file` at `geometry` under B, in the map's arithmetic under the Neural Engine switch
    /// (`Arithmetic.block`), the LoRAs' factors written into it.
    static func open(file: SafeTensors, geometry g: Geometry, loras: LoRAStack = .none,
                     budget: Int = MemoryBudget.machineCeiling()) async throws -> ZImageTransformer {
        let ranks = ZImageDiT.loraRanks(loras)
        let (block, arithmetic) = try await Arithmetic.block(file: file, codes: ZImageDiT.codesMarker, budget: budget) { declaration(g, $0, ranks: ranks) }
        let dit = try await ZImageTransformer(block, g, arithmetic, ranks: ranks)
        try dit.writeLowRank(loras)
        return dit
    }

    private init(_ block: Block, _ g: Geometry, _ arithmetic: Arithmetic, ranks: [String: Int]) async throws {
        (self.block, self.g, self.arithmetic, self.ranks) = (block, g, arithmetic, ranks)
        let form = ZImageFunctions.Form(arithmetic)
        let one: UInt16 = form == .bf16 ? 0x3f80 : Float16(1).bitPattern
        let count = form == .shared ? Self.d : 4 * ZImageBlock.gainRow
        let ones = await block.pointer(form == .shared ? "one" : "ones").bindMemory(to: UInt16.self, capacity: count)
        for i in 0..<count { ones[i] = one }
    }

    /// Everything let go, and the footprint waited for (`Stage.close`).
    @discardableResult
    func close() -> SiliCore.Stage.Settled { block.close() }

    // MARK: - The loop

    /// What a check reads of the loop: the oracle's boundaries, true values in f32, written as they pass into the
    /// check's own buffers (`tapCounts` values each) — made before the stage's plan, so that its floor counts them.
    typealias Taps = [ZImageDiT.Tap: UnsafeMutableBufferPointer<Float>]

    /// Each boundary's values at `g`.
    static func tapCounts(_ g: Geometry) -> [ZImageDiT.Tap: Int] {
        let d = Self.d
        var counts: [ZImageDiT.Tap: Int] = [.tEmbedder: c.adalnDim, .xEmbedder: g.tokens * d, .capEmbedder: g.capLength * d,
                                            .contextRefinerIn: (g.sequence - g.image) * d, .contextRefiner: (g.sequence - g.image) * d,
                                            .noiseRefiner: g.image * d, .unified: g.sequence * d, .finalLayer: g.tokens * 64,
                                            .freqs: g.sequence * ZImageBlock.hd]
        for tap in ZImageDiT.Tap.layers.values { counts[tap] = g.sequence * d }
        return counts
    }

    /// The bytes of the stream's factors a row; none under bf16.
    private var factorBytes: Int { form == .bf16 ? 0 : block.rowBytes("x_factor") }

    /// **Once a prompt**: the RoPE tables and masks of its length, its caption embedded and refined, kept aside.
    /// Under a ring once only: the context refiner's blocks ride it once, first (`Stage`'s order).
    func prompt(_ caps: [Float], taps: Taps? = nil) async throws {
        guard !prompted || stage.plan.depth == 0 else { throw Failure("a second prompt on a DiT whose weights pass through a ring") }
        prompted = true
        let (cap, n, k, d) = (g.caption, g.capLength, g.keys, Self.d)
        precondition(caps.count == n * c.capDim, "caps of \(caps.count) values for \(n) tokens")
        let p = await block.pointer("caps").assumingMemoryBound(to: Float.self), real = await block.pointer("real").assumingMemoryBound(to: Float.self)
        for i in 0..<cap * c.capDim { p[i] = i < caps.count ? caps[i] : 0 }
        for i in 0..<cap { real[i] = i < n ? 1 : 0 }
        let rope = g.dit.rope(rows: k)
        let (cos, sin) = (await block.rows("cos"), await block.rows("sin"))
        _ = rope.cos.withUnsafeBytes { memcpy(cos, $0.baseAddress!, $0.count) }
        _ = rope.sin.withUnsafeBytes { memcpy(sin, $0.baseAddress!, $0.count) }
        let masked: UInt16 = form == .bf16 ? 0xc61c : Float16(-1e4).bitPattern
        for (name, keys) in [("mask-image", 0..<g.image), ("mask-caption", g.image..<g.sequence), ("mask-sequence", 0..<g.sequence)] {
            let m = await block.pointer(name).assumingMemoryBound(to: UInt16.self)
            for j in 0..<k { m[j] = keys.contains(j) ? 0 : masked }
        }
        try await block.call("embed-caption", from: g.image)
        await stream(g.image..<g.image + n, into: taps?[.capEmbedder])
        await stream(g.image..<g.sequence, into: taps?[.contextRefinerIn])
        for b in Self.captionBlocks { try await block.run(b, "caption") }
        await stream(g.image..<g.sequence, into: taps?[.contextRefiner])
        memcpy(await block.pointer("saved"), await block.rows("x", from: g.image), cap * d * 2)
        if factorBytes > 0 { memcpy(await block.pointer("saved-factors"), await block.rows("x_factor", from: g.image), cap * factorBytes) }
    }

    /// **One evaluation**: the latent `[16, h, w]` at the sampler's `time` (in [0, 1], × 1000 inside) to the velocity,
    /// diffusers' `model_out` (not negated).
    func evaluate(_ latent: [Float], time: Float, taps: Taps? = nil) async throws -> [Float] {
        let (cap, d) = (g.caption, Self.d)
        let t = ZImageDiT.timestepFrequencies(time, config: c), tf = await block.pointer("t_freq").assumingMemoryBound(to: Float.self)
        for i in 0..<t.count { tf[i] = t[i] }
        let latentBytes = await block.pointer("latent")
        _ = latent.withUnsafeBytes { memcpy(latentBytes, $0.baseAddress!, $0.count) }
        try await block.call("timestep")
        if let t = taps?[.tEmbedder] { memcpy(t.baseAddress!, await block.pointer("adaln"), c.adalnDim * 4) }
        try await block.call("embed-image")
        await stream(0..<g.tokens, into: taps?[.xEmbedder])
        let modulate: () async throws -> Void = { try await self.block.call("modulation") }
        for b in Self.cycle.prefix(c.refiners) { try await block.run(b, "image", before: modulate) }
        await stream(0..<g.image, into: taps?[.noiseRefiner])
        memcpy(await block.rows("x", from: g.image), await block.pointer("saved"), cap * d * 2)
        if factorBytes > 0 { memcpy(await block.rows("x_factor", from: g.image), await block.pointer("saved-factors"), cap * factorBytes) }
        await stream(0..<g.sequence, into: taps?[.unified])
        for (i, b) in Self.cycle.dropFirst(c.refiners).enumerated() {
            try await block.run(b, "sequence", before: modulate)
            if let tap = ZImageDiT.Tap.layers[i] { await stream(0..<g.sequence, into: taps?[tap]) }
        }
        try await block.call("final")
        let velocity = Array(UnsafeBufferPointer(start: await block.pointer("velocity").assumingMemoryBound(to: Float.self), count: latent.count))
        if let t = taps?[.finalLayer] { Self.patchified(velocity, g, into: t) }
        if let t = taps?[.freqs] { await freqs(into: t) }
        return velocity
    }

    // MARK: - What a check reads

    /// Rows `range` of the residual stream, true values in f32 into `out` (`Block.values`).
    func stream(_ range: Range<Int>, into out: UnsafeMutableBufferPointer<Float>?) async {
        guard let out else { return }
        precondition(out.count == range.count * Self.d, "\(out.count) values for \(range.count) rows")
        let values = await block.values("x", rows: range)
        for i in values.indices { out[i] = values[i] }
    }

    /// The final layer's rows of the image's tokens `[tokens, 64]` from the velocity: the unpatchify undone.
    static func patchified(_ v: [Float], _ g: Geometry, into out: UnsafeMutableBufferPointer<Float>) {
        let (h, w, wp) = (g.height, g.width, g.width / 2)
        precondition(out.count == g.tokens * 64)
        for r in 0..<g.tokens {
            let (hi, wi) = (r / wp, r % wp)
            for j in 0..<64 { out[r * 64 + j] = v[(j % 16) * h * w + (2 * hi + j / 32) * w + 2 * wi + (j / 16) % 2] }
        }
    }

    /// The RoPE tables as the oracle writes them, `[sequence, 64, 2]`: each pair's cos and sin.
    func freqs(into out: UnsafeMutableBufferPointer<Float>) async {
        let cs = await block.rows("cos").assumingMemoryBound(to: Float.self), sn = await block.rows("sin").assumingMemoryBound(to: Float.self)
        let hd = ZImageBlock.hd
        precondition(out.count == g.sequence * hd)
        for r in 0..<g.sequence {
            for j in 0..<hd / 2 { out[r * hd + 2 * j] = cs[r * hd + 2 * j]; out[r * hd + 2 * j + 1] = sn[r * hd + 2 * j + 1] }
        }
    }
}
