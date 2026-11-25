import Foundation
import Metal
import CoreAI
import SiliCore

/// **MOSS-Image's DiT entire, and its loop**: its functions at fixed shapes (`MossFunctions`, `MossShared`), declared to
/// the engine as a block (`Block`: its stage under B, its calls bound and put ahead, the ANE's share), the loop in Swift.
///
///     once a prompt     embed-text · text block ×2 over the text's own rows (its RoPE rows, its keys masked past its
///                       length); the RoPE tables and the mask of its length
///     an evaluation     time · embed-image · the text's rows after the image's · 42 blocks on the sequence (one
///                       modulation for all) · final
///
/// **The rows**: `[image | text | its padding]`, as the reference's sequence (image then text); the keys' length is the
/// image's rows + a palier of the text's (`Geometry.palier`), so that no prompt up to it asks for a new shape, the keys
/// past the text masked (−1e4; the reference has no mask: every real row sees every other). The text blocks' per-row
/// tensors hold the palier's rows alone (`Geometry.textRows`).
///
/// **The residual stream** `x` is two buffers in turn: a block reads one and writes the other (`y`). In bf16 the text's
/// rows are copied after the image's at every evaluation (the blocks overwrite them). In the shared forms (the Compact's)
/// the stream is f16 beside its rows' factors: the embedders' bf16 rows, `e` (the text's copied after the image's once a
/// prompt), enter it by `enter`; the gains by `gains`, once the CPU has read the gates' factors off the modulation.
///
/// **The Compact** is the block in its shared forms (`MossShared`): on the GPU alone (the Neural Engine off), or with the ANE's share
/// of the image's first rows (`Block.Share`: a control's rows, or found in flight); the Standard beside the ANE, the same
/// forms on its weights in f16 (`SharedWeights.half`).
final class MossDiT {
    struct Geometry: Sendable {
        /// The latent's cells, the image's rows, the text's tokens and its palier (rows of the text's functions, and of
        /// the sequence's keys past the image).
        let height, width, image, text, palier: Int
        var keys: Int { image + palier }
        var sequence: Int { image + text }
        /// The text blocks' per-row tensors' rows: the palier's, rounded up to 64 as their calls are.
        var textRows: Int { (palier + 63) / 64 * 64 }

        init(height: Int, width: Int, text: Int) {
            (self.height, self.width, self.text) = (height, width, text)
            image = height * width
            palier = Self.palier(text, image: image)
        }

        /// The text's rows: 128, 512 or 1152 (a prompt of `Request.maximumPromptTokens` in its template), past that its
        /// multiple of 64; and as many more as bring the keys to a multiple of 64, so that a call over the sequence,
        /// rounded up to 64, never passes them.
        static func palier(_ text: Int, image: Int) -> Int {
            let step = [128, 512, 1152].first { $0 >= text } ?? (text + 63) / 64 * 64
            return step + (64 - image % 64) % 64
        }
    }

    static let d = MossFunctions.d, heads = MossFunctions.heads, hd = MossFunctions.hd, layers = 42
    static let allBlocks = (0..<layers).map { "layers.\($0)" }
    static let textBlocks = ["text0", "text1"]

    let g: Geometry
    let arithmetic: Arithmetic
    let block: Block
    /// Whether block 0 and the head are their tapped forms (the check's boundaries), on the GPU alone.
    let tapped: Bool
    var stage: SiliCore.Stage { block.stage }
    /// The boundaries a tapped evaluation read, by the reference's names.
    private(set) var boundaries: [String: [Float]] = [:]

    /// **The ranks of a stack's low-rank paths**, by product (`MossFunctions.adapted`): the largest any block's takes,
    /// so that one function serves the 42 (`LoRAStack.rank`); none for an empty stack.
    static func ranks(_ loras: LoRAStack) -> [String: Int] { loras.ranks((0..<layers).flatMap(MossFunctions.loraProducts), on: .mossImage) }

    /// **The DiT's file**, read under the prefix it keeps the DiT's names under (a checkpoint saved whole keeps
    /// `model.diffusion_model.`), found where it puts `final_norm_video.weight`, a tensor of the DiT alone (`MapWeights.dit`).
    static func file(_ path: String) throws -> SafeTensors { try MapWeights.dit(path, marker: "final_norm_video.weight") }

    // MARK: - What the DiT declares

    /// **The DiT's declaration**: its per-row tensors (the stream, q k v, the RoPE tables, the text blocks' own; in the
    /// shared forms the steps' own and the embedders' rows; tapped, block 0's boundaries), its other tensors, the
    /// weights read once, a group a block — the text blocks' used once a prompt, the 42 at every evaluation —, its forms
    /// (block 0's tapped, the GPU's alone), its two passes and its edges.
    static func declaration(_ g: Geometry, _ arithmetic: Arithmetic, tapped: Bool, ranks: [String: Int] = [:]) -> Block.Declaration {
        let (d, hd, k, p, tp) = (Self.d, Self.hd, g.keys, g.palier, g.textRows), shared = arithmetic.shared, e = arithmetic.element
        let latent = MossFunctions.channels * g.image * 4
        var n = Block.Declaration("MOSS-Image's DiT (\(shared ? "shared" : "bf16"), \(g.height)×\(g.width), \(k) keys)", rows: k)
        n.perRow = [.init("x", width: d, e, next: "y", factor: shared ? "x_factor" : nil), .init("q", width: d, e), .init("k", width: d, e),
                    .init("v", width: d, e), .init("cos", width: hd, .float32), .init("sin", width: hd, .float32),
                    .init("text-x", width: d, .bfloat16, next: "text-y", rows: tp)]
        n.perRow += ["text-q", "text-k", "text-v"].map { Block.Rows($0, width: d, .bfloat16, rows: tp) }
            + ["text-cos", "text-sin"].map { Block.Rows($0, width: hd, .float32, rows: tp) }
        if shared {
            n.perRow += [.init("x_factor", width: 1, .float16, next: "y_factor"), .init("qkv", width: 3 * d, .float16, factor: "qkv_factor"),
                         .init("qkv_factor", width: 1, .float16), .init("attended", width: d, .float16),
                         .init("mid", width: d, .float16, factor: "mid_factor"), .init("mid_factor", width: 1, .float16), .init("e", width: d, .bfloat16)]
        }
        if tapped {
            n.perRow += [("q_norm_out", d), ("o_out", d), ("up_gate_out", 2 * MossFunctions.hidden), ("down_out", d)].map { Block.Rows($0.0, width: $0.1, .bfloat16) }
            if !shared { n.perRow.append(.init("qkv_out", width: 3 * d, .bfloat16)) }
        }
        var t: [(String, Int)] = [("latent", latent), ("t", MossFunctions.timeDim * 4), ("time", d * 2), ("modulation", 6 * MossFunctions.gainRow * 2),
                                  ("head", 2 * d * 2), ("text", p * MossFunctions.textDim * 4), ("text-mask", p * 2), ("mask", k * 2)]
        if shared { t += MossShared.gains.map { ($0, d * 2) } + MossShared.gateFactors.map { ($0, 2) } }
        if tapped { t.append(("final_norm_out", g.image * d * 2)) }
        n.tensors = t.map { SiliCore.Stage.Tensor($0.0, bytes: $0.1) } + [SiliCore.Stage.Tensor("velocity", bytes: latent, .apart)]
            + (0..<layers).flatMap { LoRAStack.tensors(MossFunctions.loraProducts($0), ranks: ranks, element: e, prefix: "layers.\($0).") }
        n.residents = MossFunctions.others
        let form = shared ? "shared" : "bf16"
        n.groups = (0..<2).map { Block.Group(textBlocks[$0], form: "text", weights: MossFunctions.textBlock($0)) }
            + allBlocks.enumerated().map { i, b in
                Block.Group(b, form: tapped && i == 0 ? form + "-tapped" : form, weights: shared ? MossShared.layer(i, weights: arithmetic.weights) : MossFunctions.layer(i))
            }
        (n.once, n.cycle) = (textBlocks, allBlocks)

        // The text blocks over their own rows; their keys and values read whole.
        n.forms = [Block.Form("text", steps: [.init("text-pre", on: .gpu), .init("text-post", on: .gpu, bind: ["k": .whole("text-k"), "v": .whole("text-v")])],
                              bind: ["x": .rows("text-x"), "y": .next("text-x"), "q": .rows("text-q"), "k": .rows("text-k"), "v": .rows("text-v"),
                                     "cos": .rows("text-cos"), "sin": .rows("text-sin"), "mask": .tensor("text-mask")]) { step, b in
            MossFunctions.module(step, rows: b.rows, keys: p)
        }]
        for tap in [false, true] where !tap || tapped {
            let name = form + (tap ? "-tapped" : "")
            if shared {
                // pre, out and ffn either unit's (the tapped form's the GPU's); heads (f32, the RoPE) and the attention the GPU's.
                let unit: Block.Unit = tap ? .gpu : .either
                n.forms.append(Block.Form(name, steps: [.init("pre", on: unit), .init("heads", on: .gpu, bind: ["qkv": .rows("qkv")]),
                                                        .init("attention", on: .gpu, whole: ["k", "v"]),
                                                        .init("out", on: unit, bind: ["y": .rows("mid"), "y_factor": .rows("mid_factor")]),
                                                        .init("ffn", on: unit, bind: ["x": .rows("mid"), "x_factor": .rows("mid_factor")])]) { step, b in
                    MossShared.module(step, rows: b.rows, keys: k, weights: arithmetic.weights, tapped: tap, ranks: ranks, neural: b.neural)
                })
            } else {
                // The modulation's six rows, each on a page of its own.
                let rows = ["shift", "scale", "gate_a", "shift_f", "scale_f", "gate_f"].enumerated().map { ($1, Block.Bind.row("modulation", $0)) }
                n.forms.append(Block.Form(name, steps: [.init("pre", on: .gpu), .init("post", on: .gpu, whole: ["k", "v"])],
                                          bind: Dictionary(uniqueKeysWithValues: rows)) { step, b in
                    MossFunctions.module(step, rows: b.rows, keys: k, tapped: tap, ranks: ranks)
                })
            }
        }
        n.passes = [Block.Pass("text", rows: 0..<p, forms: ["text"]),
                    Block.Pass("sequence", rows: 0..<g.sequence, shareable: shared ? 0..<g.image : nil, forms: [form, form + "-tapped"])]
        let (h, w) = (g.height, g.width)
        n.edges["time"] = { _ in MossFunctions.module("time", rows: 1, keys: 1) }
        n.edges["embed-image"] = { _ in MossFunctions.module("embed-image", rows: g.image, keys: 1, height: h, width: w) }
        n.edges["embed-text"] = { _ in MossFunctions.module("embed-text", rows: p, keys: 1) }
        n.edges["final"] = { _ in
            shared ? MossShared.module("final", rows: g.image, keys: k, weights: arithmetic.weights, height: h, width: w, tapped: tapped)
                : MossFunctions.module("final", rows: g.image, keys: k, height: h, width: w, tapped: tapped)
        }
        if shared {
            n.edges["enter"] = { _ in MossShared.module("enter", rows: k, keys: k, weights: arithmetic.weights) }
            n.edges["gains"] = { _ in MossShared.module("gains", rows: 1, keys: 1, weights: arithmetic.weights) }
        }
        return n
    }

    // MARK: - The block

    /// **The DiT of `file` under B** in the map's arithmetic under the Neural Engine switch (`Arithmetic.block`); the
    /// low-rank factors written.
    static func open(file: SafeTensors, geometry g: Geometry, budget: Int, tapped: Bool = false, loras: LoRAStack = .none) async throws -> MossDiT {
        let ranks = Self.ranks(loras)
        let (block, arithmetic) = try await Arithmetic.block(file: file, codes: "layers.0.proj.qkv.weight", budget: budget) {
            declaration(g, $0, tapped: tapped, ranks: ranks)
        }
        for i in 0..<layers where !ranks.isEmpty {
            try loras.write(MossFunctions.loraProducts(i), on: .mossImage, ranks: ranks, element: arithmetic.element, prefix: "layers.\(i).") {
                block.stage.pointer($0)
            }
        }
        return MossDiT(block, g, arithmetic, tapped: tapped)
    }

    /// **What `open` would specialize** (`Arithmetic.prepare`): the same declaration, nothing allocated.
    static func prepare(file: SafeTensors, geometry g: Geometry, budget: Int, loras: LoRAStack = .none) throws -> [Functions.Specialization] {
        let ranks = Self.ranks(loras)
        return try Arithmetic.prepare(file: file, codes: "layers.0.proj.qkv.weight", budget: budget) { declaration(g, $0, tapped: false, ranks: ranks) }
    }

    private init(_ block: Block, _ g: Geometry, _ arithmetic: Arithmetic, tapped: Bool) {
        (self.block, self.g, self.arithmetic, self.tapped) = (block, g, arithmetic, tapped)
    }

    /// What the DiT is: its arithmetic and its plan under B.
    var described: String { "\(arithmetic), " + stage.plan.line }

    /// Everything let go, the footprint waited for.
    @discardableResult func close() -> SiliCore.Stage.Settled { block.close() }

    // MARK: - The loop

    /// The tables of the reference's RoPE (`MossImageDiTProgram.rope`, `[sin | cos]` of 64 each) as the functions read
    /// them: `[cos | cos]` and `[−sin | sin]`, rows `rows` of the table into `cos` and `sin` from their row 0.
    private func tables(_ table: [Float], rows: Range<Int>, into cos: UnsafeMutableRawPointer, _ sin: UnsafeMutableRawPointer) {
        let c = cos.assumingMemoryBound(to: Float.self), s = sin.assumingMemoryBound(to: Float.self), half = Self.hd / 2
        for (i, r) in rows.enumerated() {
            for j in 0..<half {
                let sn = table[r * Self.hd + j], cs = table[r * Self.hd + half + j]
                c[i * Self.hd + j] = cs; c[i * Self.hd + half + j] = cs
                s[i * Self.hd + j] = -sn; s[i * Self.hd + half + j] = sn
            }
        }
    }

    /// **Once a prompt**: the text `[T, 4096]` (the encoder's last hidden state) projected and refined by the two text
    /// blocks over its own rows; the sequence's RoPE tables and mask, of its length.
    func prompt(_ text: [Float]) async throws {
        let (t, p, k) = (g.text, g.palier, g.keys)
        precondition(text.count == t * MossFunctions.textDim, "a text of \(text.count / MossFunctions.textDim) rows for \(t)")
        let input = await block.pointer("text").assumingMemoryBound(to: Float.self)
        for i in 0..<p * MossFunctions.textDim { input[i] = i < text.count ? text[i] : 0 }
        let table = MossImageDiTProgram(latentHeight: g.height, latentWidth: g.width, textLength: t).rope()
        tables(table, rows: 0..<g.sequence, into: await block.rows("cos"), await block.rows("sin"))
        tables(table, rows: g.image..<g.sequence, into: await block.rows("text-cos"), await block.rows("text-sin"))
        // The masks in the element the attention reads: the text blocks' bf16, the sequence's the stream's.
        let mask = await block.pointer("mask").assumingMemoryBound(to: UInt16.self), textMask = await block.pointer("text-mask").assumingMemoryBound(to: UInt16.self)
        let masked = MossDiT.bf16(-1e4), sequenceMasked = arithmetic.shared ? Float16(-1e4).bitPattern : masked
        for j in 0..<k { mask[j] = j < g.sequence ? 0 : sequenceMasked }
        for j in 0..<p { textMask[j] = j < t ? 0 : masked }
        try await block.call("embed-text", ["x": .rows("text-x")])
        if tapped { boundaries["text_proj_out"] = await block.values("text-x", rows: 0..<t) }
        for (i, b) in Self.textBlocks.enumerated() {
            try await block.run(b, "text")
            if tapped { boundaries["text_block\(i)_out"] = await block.values("text-x", rows: 0..<t) }
        }
        // The shared forms' embedded rows keep the text's: `enter` reads them at every evaluation.
        if arithmetic.shared { await block.copy("text-x", rows: 0..<p, to: "e", at: g.image) }
    }

    /// **One evaluation**: the latent `[128, h, w]` at the told timestep (σ · 1000) to the velocity, `model_out` (not
    /// negated). Tapped, every boundary of the reference read into `boundaries`.
    func evaluate(_ latent: [Float], timestep: Float) async throws -> [Float] {
        let (image, shared) = (g.image, arithmetic.shared)
        let s = MossImageDiTProgram.timeSinusoid(timestep), tp = await block.pointer("t").assumingMemoryBound(to: Float.self)
        for i in 0..<s.count { tp[i] = s[i] }
        let latentBytes = await block.pointer("latent")
        _ = latent.withUnsafeBytes { memcpy(latentBytes, $0.baseAddress!, $0.count) }
        // The text's rows after the image's, which the blocks overwrite in the stream.
        if !shared { await block.copy("text-x", rows: 0..<g.palier, to: "x", at: image) }
        try await block.call("time")
        try await block.call("embed-image", shared ? ["x": .rows("e")] : [:])
        if tapped {
            boundaries["time_embedding_out"] = await floats("time", count: Self.d)
            let modulation = await floats("modulation", count: 6 * MossFunctions.gainRow)
            boundaries["shared_modulation"] = (0..<6).flatMap { modulation[$0 * MossFunctions.gainRow ..< $0 * MossFunctions.gainRow + Self.d] }
            boundaries["head_modulation"] = await floats("head", count: 2 * Self.d)
            boundaries["video_embed"] = await block.values(shared ? "e" : "x", rows: 0..<image)
        }
        if shared {
            try await block.call("enter", ["x": .rows("e"), "y": .rows("x"), "y_factor": .rows("x_factor")])
            try await gains()
        }
        for (i, b) in Self.allBlocks.enumerated() {
            try await block.run(b, "sequence")
            if tapped {
                if i == 0 { await tapBlock0() }
                if [0, 1, 13, 27, 41].contains(i) { boundaries["block\(i)_out"] = await block.values("x", rows: 0..<g.sequence) }
            }
        }
        // The head's modulation, a tensor where a weight has its name.
        try await block.call("final", ["head": .tensor("head")])
        if tapped { boundaries["final_norm_out"] = await floats("final_norm_out", count: image * Self.d) }
        return Array(UnsafeBufferPointer(start: await block.pointer("velocity").assumingMemoryBound(to: Float.self), count: latent.count))
    }

    /// The modulation's rows as the shared forms read them (`MossShared.gains`): each gate over G, the least power of
    /// two at or past its row's largest magnitude, read here off the modulation's bf16 row.
    private func gains() async throws {
        let d = Self.d, modulation = await block.pointer("modulation").assumingMemoryBound(to: UInt16.self)
        for (row, name) in [(2, "gate1_factor"), (5, "gate2_factor")] {
            let top = (0..<d).reduce(Float(0)) { max($0, abs(Float(bitPattern: UInt32(modulation[row * MossFunctions.gainRow + $1]) << 16))) }
            await block.pointer(name).assumingMemoryBound(to: Float16.self)[0] = Float16(Gate.factor(largest: top))
        }
        try await block.call("gains")
    }

    /// Block 0's boundaries inside: its projections (the shared forms' `qkv` true values), q after its norm, the
    /// attention's out product, the feed-forward's two.
    private func tapBlock0() async {
        let (s, d) = (g.sequence, Self.d)
        let qkv = await block.values(arithmetic.shared ? "qkv" : "qkv_out", rows: 0..<s)
        for (i, name) in ["block0_q_out", "block0_k_out", "block0_v_out"].enumerated() {
            boundaries[name] = (0..<s).flatMap { r in qkv[(r * 3 * d + i * d)..<(r * 3 * d + (i + 1) * d)] }
        }
        for (name, tensor) in [("block0_q_norm_out", "q_norm_out"), ("block0_o_out", "o_out"), ("block0_ffn_up_gate_out", "up_gate_out"),
                               ("block0_ffn_down_out", "down_out")] {
            boundaries[name] = await block.values(tensor, rows: 0..<s)
        }
    }

    /// The first `count` values of a bf16 tensor, widened.
    private func floats(_ name: String, count: Int) async -> [Float] {
        let p = await block.pointer(name).assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { Float(bitPattern: UInt32(p[$0]) << 16) }
    }

    /// `x` rounded to nearest even in bf16, its bits.
    static func bf16(_ x: Float) -> UInt16 { let b = x.bitPattern; return UInt16(truncatingIfNeeded: (b &+ 0x7fff &+ ((b >> 16) & 1)) >> 16) }
}
