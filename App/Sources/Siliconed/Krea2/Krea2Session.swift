import CoreAI
import Foundation
import Metal
import SiliCore

/// **Krea 2's DiT entire, and its loop**: its functions at fixed shapes (`Krea2Functions`, `Krea2Shared`), declared to the
/// engine as a block (`Block`: its stage under B, its calls bound and put ahead, the ANE's share), the loop in Swift.
///
///     once a prompt     the fused text's rows (`Krea2Fusion`), kept aside in the stream's form
///     an evaluation     timestep · embed · the text put back · blocks ×28 on the sequence · final
///
/// **The rows**: `[image | text | the text's palier]`, every row a key at batch 1. The reference orders them text first;
/// the attention reads its keys in any order and every other op is a row's, so only the RoPE tables (a row's position)
/// and the boundaries a check reads follow the rows. The keys are the image's rows and a palier of the text's (512, then
/// by 512): no prompt up to it asks for a new shape; the keys past the text are masked (−10⁴). A block is one pass over
/// the sequence (`Block`'s calls).
///
/// **The residual stream** `x` is two buffers in turn (a step reads one and writes the other, `y`): bf16 (the Standard), or
/// f16 beside a power of two a row (`x_factor`, the Compact's shared forms, the GPU's and the ANE's).
///
/// **A LoRA** lands on each block's products (`ranks`, the same in every block): its factors held all along, block i's
/// `blocks.<i>.<argument>_down` and `_up` (`Block`: the group's own tensors), written once the block is open; and on the
/// timestep's and the final layer's products, their factors `<argument>_down` and `_up`, bf16 in both forms.
///
/// **A check listening inside block 0** (`taps`): its attention and feed-forward values, written by the steps that tap
/// (`Block.Step.taps`) in a run the GPU alone makes (`Block.run`'s `tapped`).
final class Krea2Session {
    struct Geometry: Equatable, Sendable {
        /// The latent's, the image's rows (`h/2 · w/2`), the text's rows and their palier.
        let height, width, image, text, palier: Int
        var keys: Int { image + palier }
        var sequence: Int { image + text }

        init(latentHeight: Int, latentWidth: Int, text: Int) {
            (height, width) = (latentHeight, latentWidth)
            image = (latentHeight / 2) * (latentWidth / 2)
            self.text = text
            palier = Self.palier(text)
        }

        /// The text's palier: 512, then by 512.
        static func palier(_ text: Int) -> Int { max(512, (text + 511) / 512 * 512) }
    }

    static let d = Krea2Functions.d, hd = Krea2Functions.hd, kvd = Krea2Functions.kvHeads * Krea2Functions.hd, layers = 28
    /// The blocks' groups; a block's weights are `<group>.attn.wq.weight`…
    static let blocks = (0..<layers).map { "blocks.\($0)" }

    let g: Geometry
    let arithmetic: Arithmetic
    let block: Block
    /// The LoRAs' ranks by product (`ranks(_:)`): none, the functions as they are.
    let ranks: [String: Int]
    var shared: Bool { arithmetic.shared }
    var stage: SiliCore.Stage { block.stage }

    // MARK: - What the DiT needs

    /// **The LoRAs' ranks by product** (`LoRAStack.ranks`): a block's, the same in every block, and each edge's own (their
    /// names apart from the blocks'), none where nothing lands.
    static func ranks(_ loras: LoRAStack) -> [String: Int] {
        loras.ranks(blocks.indices.flatMap(loraProducts) + Krea2Functions.edgeProducts.flatMap(\.products), on: .krea2)
    }

    /// The ranks of what step `kind` computes: a block's products only.
    static func ranks(_ kind: String, _ ranks: [String: Int]) -> [String: Int] {
        let own: Set<String> = switch kind {
        case "pre": ["wq", "wk", "wv", "gate"]
        case "out": ["wo"]
        case "ffn": ["mgate", "up", "down"]
        case "post": ["gate", "wo", "mgate", "up", "down"]
        default: []
        }
        return ranks.filter { own.contains($0.key) }
    }

    /// An edge's function: the timestep's is the bf16 one in both forms; the Compact's others are the shared forms'.
    /// `ranks`: the stack's (`ranks(_:)`), the edge taking its own.
    static func module(_ f: String, _ g: Geometry, _ a: Arithmetic, ranks: [String: Int] = [:]) -> Aicode.Module {
        let own = Krea2Functions.edgeRanks(f, ranks)
        if a.shared && f != "timestep" { return Krea2Shared.module(f, rows: g.image, keys: g.keys, weights: a.weights, ranks: own) }
        return Krea2Functions.module(f, rows: g.image, keys: g.keys, ranks: own)
    }

    /// **The DiT's declaration**: its per-row tensors (the stream, q k v, the RoPE tables; in the shared forms the steps'
    /// own), its other tensors, the edges' weights read once, a group a block in the order the loop uses them, its form
    /// and its pass. `taps`: block 0's attention and feed-forward values too (a check's).
    static func declaration(_ g: Geometry, _ a: Arithmetic, ranks: [String: Int], taps: Bool) -> Block.Declaration {
        let (k, n, p, patch) = (g.keys, g.image, g.palier, Krea2Functions.patch)
        let element: NDArray.ScalarType = a.shared ? .float16 : .bfloat16
        let form = a.shared ? "shared" : "bf16"
        var d = Block.Declaration("Krea 2's DiT (\(a), \(g.height)×\(g.width), \(g.keys) keys)", rows: k)
        d.perRow = [.init("x", width: Self.d, element, next: "y", factor: a.shared ? "x_factor" : nil),
                    .init("q", width: Self.d, element), .init("k", width: kvd, element), .init("v", width: kvd, element),
                    .init("cos", width: hd, .float32), .init("sin", width: hd, .float32)]
        if a.shared {
            d.perRow += [.init("x_factor", width: 1, .float16, next: "y_factor"),
                         .init("qh", width: Self.d, .float16), .init("kh", width: kvd, .float16), .init("vh", width: kvd, .float16),
                         .init("gh", width: Self.d, .float16), .init("pre_factor", width: 1, .float16), .init("sg", width: Self.d, .float16),
                         .init("o", width: Self.d, .float16), .init("mid", width: Self.d, .float16, factor: "mid_factor"),
                         .init("mid_factor", width: 1, .float16)]
        }
        if taps { d.perRow += [.init("attention", width: Self.d, a.shared ? .float16 : .float32), .init("feed", width: Self.d, a.shared ? .float16 : .float32)] }
        let t: [(String, Int)] = [("patches", n * patch * 4), ("t_freq", Krea2Functions.timeDim * 4), ("temb", Self.d * 2), ("temb_mod", 6 * Self.d * 2),
                                  ("mask", k * 2), ("saved", p * Self.d * 2), ("saved-factors", p * 2)]
        d.tensors = t.map { SiliCore.Stage.Tensor($0.0, bytes: $0.1) } + [SiliCore.Stage.Tensor("velocity", bytes: n * patch * 4, .apart)]
        if a.shared { d.tensors += Krea2Shared.gains.map { SiliCore.Stage.Tensor($0, bytes: Self.d * 2) } }
        for (i, b) in blocks.enumerated() {
            d.tensors += LoRAStack.tensors(loraProducts(i), ranks: ranks, element: element, prefix: b + ".")
        }
        d.tensors += LoRAStack.tensors(Krea2Functions.edgeProducts.flatMap(\.products), ranks: ranks, element: .bfloat16)
        d.residents = Krea2Functions.others
        d.groups = blocks.map { Block.Group($0, form: form, weights: a.shared ? Krea2Shared.block($0 + ".", weights: a.weights) : Krea2Functions.block($0 + ".")) }
        d.cycle = blocks
        d.tapped = taps
        if a.shared {
            d.forms = [Block.Form(form, steps: [.init("pre"), .init("heads", on: .gpu), .init("attention", on: .gpu, whole: ["k", "v"]),
                                                .init("out", bind: ["y": .rows("mid"), "y_factor": .rows("mid_factor")], taps: true),
                                                .init("ffn", bind: ["x": .rows("mid"), "x_factor": .rows("mid_factor")], taps: true)]) { step, b in
                Krea2Shared.module(step + (b.tapped ? "-taps" : ""), rows: b.rows, keys: k, weights: a.weights, ranks: Self.ranks(step, ranks),
                                   neural: b.neural)
            }]
        } else {
            d.forms = [Block.Form(form, steps: [.init("pre", on: .gpu), .init("post", on: .gpu, whole: ["k", "v"], taps: true)]) { step, b in
                Krea2Functions.module(step + (b.tapped ? "-taps" : ""), rows: b.rows, keys: k, ranks: Self.ranks(step, ranks))
            }]
        }
        d.passes = [Block.Pass("sequence", rows: 0..<g.sequence, shareable: 0..<n)]
        for f in ["timestep", "embed", "final"] + (a.shared ? ["modulation"] : []) {
            d.edges[f] = { _ in module(f, g, a, ranks: ranks) }
        }
        return d
    }

    /// Block i's products as a LoRA adapts them (`LoRAProduct`).
    static func loraProducts(_ i: Int) -> [LoRAProduct] {
        Krea2Functions.products.map { name, module in
            let (inn, out) = switch name {
            case "wk", "wv": (d, kvd)
            case "up", "mgate": (d, Krea2Functions.hidden)
            case "down": (Krea2Functions.hidden, d)
            default: (d, d)
            }
            return LoRAProduct(name: name, key: blocks[i] + "." + module, inn: inn, out: out)
        }
    }

    // MARK: - Opening

    /// A product of the DiT alone, by which its file is found under a checkpoint's prefix (`MapWeights.dit`) and its
    /// element says the Compact (`Arithmetic.codes`).
    static let marker = "blocks.0.attn.wq.weight"

    /// **The DiT of `file` at `geometry` under `budget`**, in the map's arithmetic under the Neural Engine switch
    /// (`Arithmetic.block`), the LoRAs' factors and the tables written.
    static func open(file: SafeTensors, geometry: Geometry, loras: LoRAStack = .none, taps: Bool = false,
                     budget: Int) async throws -> Krea2Session {
        let ranks = Self.ranks(loras)
        let (block, arithmetic) = try await Arithmetic.block(file: file, codes: marker, budget: budget) {
            declaration(geometry, $0, ranks: ranks, taps: taps)
        }
        let session = Krea2Session(block, geometry: geometry, arithmetic: arithmetic, ranks: ranks)
        try session.writeLowRank(loras)
        await session.writeTables()
        return session
    }

    /// **What `open` would specialize** (`Arithmetic.prepare`): the same declaration, nothing allocated.
    static func prepare(file: SafeTensors, geometry: Geometry, loras: LoRAStack = .none, budget: Int) throws -> [Preparation] {
        let ranks = Self.ranks(loras)
        return try Arithmetic.prepare(file: file, codes: marker, budget: budget) { declaration(geometry, $0, ranks: ranks, taps: false) }
    }

    private init(_ block: Block, geometry: Geometry, arithmetic: Arithmetic, ranks: [String: Int]) {
        (self.block, g, self.arithmetic, self.ranks) = (block, geometry, arithmetic, ranks)
    }

    /// **The LoRAs' factors written** into every block's tensors and the edges' (`LoRAStack.write`), at the ranks the
    /// block was opened with.
    func writeLowRank(_ loras: LoRAStack) throws {
        for (i, b) in Self.blocks.enumerated() {
            try loras.write(Self.loraProducts(i), on: .krea2, ranks: ranks, element: shared ? .float16 : .bfloat16, prefix: b + ".", into: stage.pointer)
        }
        try loras.write(Krea2Functions.edgeProducts.flatMap(\.products), on: .krea2, ranks: ranks, element: .bfloat16, into: stage.pointer)
    }

    /// Everything let go, and the footprint waited for (`Stage.close`).
    @discardableResult
    func close() -> SiliCore.Stage.Settled { block.close() }

    // MARK: - The tables

    /// The RoPE tables in the rows' order, `[keys, 1, 128]` each pair's angle twice and the sine's sign (−sin, sin); the
    /// image's rows at (0, row, column), the text's at (0, 0, 0) (`Krea2RotaryPosEmbed`'s positions and angles, f64 rounded to f32); the mask.
    private func writeTables() async {
        let c = await block.rows("cos").assumingMemoryBound(to: Float.self), s = await block.rows("sin").assumingMemoryBound(to: Float.self)
        let (axes, theta) = (Krea2DiT.Config.published.axes, Krea2DiT.Config.published.theta)
        let freqs = axes.map { n in (0..<(n / 2)).map { 1 / Foundation.pow(theta, Double(2 * $0) / Double(n)) } }
        let w = g.width / 2
        for r in 0..<g.keys {
            let position = r < g.image ? [0, r / w, r % w] : [0, 0, 0]
            var j = 0
            for (axis, p) in position.enumerated() {
                for f in freqs[axis] {
                    let angle = Double(p) * f
                    let (cs, sn) = cosSin(angle)
                    c[r * Self.hd + 2 * j] = cs; c[r * Self.hd + 2 * j + 1] = cs
                    s[r * Self.hd + 2 * j] = -sn; s[r * Self.hd + 2 * j + 1] = sn
                    j += 1
                }
            }
        }
        let m = await block.pointer("mask").assumingMemoryBound(to: UInt16.self)
        let masked = shared ? Float16(-1e4).bitPattern : BFloat16Bits.rounded(-1e4)
        for j in 0..<g.keys { m[j] = j < g.sequence ? 0 : masked }
    }

    // MARK: - The text, once a prompt

    /// The fused text's rows `[T, d]` (bf16 values, `Krea2Fusion`) kept aside in the stream's form: bf16, or f16 beside a
    /// row's factor, its largest in (2¹³, 2¹⁴] (`ranged16`'s rule on the CPU; a bf16 under a power of two is exact in f16),
    /// the factor f16 within [2⁻¹³, 2¹⁵].
    func setText(_ text: [Float]) async {
        precondition(text.count == g.text * Self.d, "krea2: \(text.count) values of text for \(g.text) rows")
        let d = Self.d
        let saved = await block.pointer("saved")
        guard shared else {
            let x = saved.assumingMemoryBound(to: UInt16.self)
            for i in 0..<text.count { x[i] = BFloat16Bits.rounded(text[i]) }
            return
        }
        let x = saved.assumingMemoryBound(to: Float16.self), factors = await block.pointer("saved-factors").assumingMemoryBound(to: Float16.self)
        for r in 0..<g.text {
            // The shared forms' rule (`ranged16`): p in (2^m, 2^(m+1)] to (2¹³, 2¹⁴], e = m + 1 − 14.
            let (unit, factor) = RowExponent.split(text[(r * d)..<((r + 1) * d)], .ranged16)
            factors[r] = Float16(factor)
            for (c, v) in unit.enumerated() { x[r * d + c] = Float16(v) }
        }
    }

    // MARK: - One evaluation

    /// Each boundary's values at `g` that the DiT writes (the fused text is the fusion's, `Krea2Fusion.Text`).
    static func tapCounts(_ g: Geometry) -> [Krea2Transformer.Tap: Int] {
        var counts: [Krea2Transformer.Tap: Int] = [.temb: d, .tembMod: 6 * d, .imgIn: g.image * d]
        for tap in [Krea2Transformer.Tap.block0, .block1, .block13, .block27, .block0Attention, .block0Feed] { counts[tap] = g.sequence * d }
        return counts
    }

    /// **One evaluation**: the latent `[16, h, w]` at the told time (`Krea2Transformer.modelTime`) to the velocity, the
    /// reference's `model_out`, into `velocity` (`latent.count` values). `taps`: the boundaries a check reads too (the
    /// session opened with `taps`).
    func evaluate(_ latent: [Float], time: Float, into velocity: UnsafeMutableBufferPointer<Float>,
                  taps: Krea2Render.Denoiser.Taps? = nil) async throws {
        let (n, d, c) = (g.image, Self.d, 16)
        precondition(latent.count == c * g.height * g.width && velocity.count == latent.count)
        let tf = Krea2Transformer.timestepFrequencies(time, config: .published), p = await block.pointer("t_freq").assumingMemoryBound(to: Float.self)
        for i in 0..<tf.count { p[i] = tf[i] }
        try await block.call("timestep")
        // "c (h p1) (w p2) → (h w) (c p1 p2)": `_pack_latents`, the channel slowest.
        let (h2, w2) = (g.height / 2, g.width / 2), patches = await block.pointer("patches").assumingMemoryBound(to: Float.self)
        for i in 0..<h2 { for j in 0..<w2 { for ch in 0..<c { for p1 in 0..<2 { for p2 in 0..<2 {
            patches[(i * w2 + j) * 64 + ch * 4 + p1 * 2 + p2] = latent[(ch * g.height + 2 * i + p1) * g.width + 2 * j + p2]
        } } } } }
        try await block.call("embed")
        if let taps {
            await bfloats("temb", into: taps[.temb]!)
            await bfloats("temb_mod", into: taps[.tembMod]!)
            await block.values("x", rows: 0..<n, into: taps[.imgIn]!)
        }
        memcpy(await block.rows("x", from: n), await block.pointer("saved"), g.palier * d * 2)
        if shared { memcpy(await block.rows("x_factor", from: n), await block.pointer("saved-factors"), g.palier * 2) }
        let modulate: (() async throws -> Void)? = shared ? { try await self.block.call("modulation") } : nil
        for (i, b) in Self.blocks.enumerated() {
            let listening = taps != nil && i == 0
            try await block.run(b, "sequence", before: modulate, tapped: listening)
            guard let taps else { continue }
            if let tap = [0: Krea2Transformer.Tap.block0, 1: .block1, 13: .block13, 27: .block27][i] { await referenceOrder("x", into: taps[tap]!) }
            if listening {
                await referenceOrder("attention", into: taps[.block0Attention]!)
                await referenceOrder("feed", into: taps[.block0Feed]!)
            }
        }
        try await block.call("final")
        // `_unpack_latents`: the image rows `[h·w, c·4]` back to `[c, h·2, w·2]`.
        let y = await block.pointer("velocity").assumingMemoryBound(to: Float.self)
        for i in 0..<h2 { for j in 0..<w2 { for ch in 0..<c { for p1 in 0..<2 { for p2 in 0..<2 {
            velocity[(ch * g.height + 2 * i + p1) * g.width + 2 * j + p2] = y[(i * w2 + j) * 64 + ch * 4 + p1 * 2 + p2]
        } } } } }
    }

    // MARK: - Reading back

    private func bfloats(_ name: String, into out: UnsafeMutableBufferPointer<Float>) async {
        let p = await block.pointer(name).assumingMemoryBound(to: UInt16.self)
        for i in 0..<out.count { out[i] = Float(bitPattern: UInt32(p[i]) << 16) }
    }

    /// Per-row tensor `name`'s `[image | text]` rows as the reference orders them, `[text | image]`.
    private func referenceOrder(_ name: String, into out: UnsafeMutableBufferPointer<Float>) async {
        let split = (g.sequence - g.image) * Self.d
        await block.values(name, rows: g.image..<g.sequence, into: UnsafeMutableBufferPointer(rebasing: out[..<split]))
        await block.values(name, rows: 0..<g.image, into: UnsafeMutableBufferPointer(rebasing: out[split...]))
    }
}
