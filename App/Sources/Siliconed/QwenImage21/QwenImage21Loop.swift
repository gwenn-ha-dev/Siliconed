import CoreAI
import Foundation
import Metal
import SiliCore

/// **Qwen-Image-2.1's DiT and its loop**: its functions (`QwenImage21Functions`) at fixed shapes, declared to the engine as
/// a block (`Block`: its stage under B, its calls bound and put ahead, the ANE's share), the loop in Swift.
///
///     once a render     timestep(0) · embed-prefix · layers ×32 on the prefix's rows: each layer's keys and values
///                       of the prefix kept (`<layer>.kp`, `<layer>.vp`)
///     an evaluation     timestep(t) · embed-target · layers ×32 on the target's rows, each reading the layer's
///                       prefix keys beside the target's · final
///
/// **The prefix is computed once** (the reference's K/V cache): its rows see neither the target nor the time — they are
/// modulated by t = 0 and the attention is block-causal, the target last —, so what they give each layer's attention
/// is the same at every step. The prefix's last layer computes its keys only (its output reaches nothing: `run`'s
/// `until`), unless someone observes the passes (`observe`).
///
/// **Two passes over one block**: the prefix's (its keys and values written into the layer's own, its modulation of
/// t = 0, its block-causal mask) and the target's (its keys and values `kt`, `vt`, its modulation, its RoPE rows past the
/// prefix's, every real key visible) — the same functions, bound to other tensors.
///
/// **The rows**: the prefix's real rows `P` within a palier of 256 keys (no prompt within it asks for a new shape), the
/// target's `S = h·w` within a multiple of 64; each pass is the engine's calls over its rows. The rows past the real ones
/// are computed and masked as keys (−10⁴). The residual stream is two buffers in turn, beside their row factors under
/// the guard of f16 (f32, one a row; f16 in the shared forms).
///
/// **With the ANE's share** (the shared forms, `QwenImage21Shared`, the Compact under the precision `ane`): the ANE the
/// target's first rows by the engine's unit, told (`--neural-engine`) or found in flight; the prefix's pass is the GPU's.
///
/// **Under a LoRA**: the factors of every product it adapts in the stage's tensors, `<product>_down`, `_up` outside the
/// layers, `<layer>.<product>_…` in them, written once the stage is opened (`LoRAStack.write`); the functions at their
/// ranks (`QwenImage21Functions.Ranks`).
package final class QwenImage21Loop {
    package typealias Path = QwenImage21Functions.Path
    package typealias Ranks = QwenImage21Functions.Ranks

    package struct Geometry: Sendable {
        package let sequence: QwenImage21DiT.Sequence
        /// The prefix's real rows and its keys; the target's and its keys.
        package let prefix, prefixKeys, target, targetKeys: Int
        package var keys: Int { prefixKeys + targetKeys }
        /// The stream's rows, either pass's.
        package var rows: Int { max(prefixKeys, targetKeys) }

        package static let palier = 256, unit = 64

        package init(_ s: QwenImage21DiT.Sequence) {
            sequence = s
            prefix = s.prefixRows
            prefixKeys = (prefix + Self.palier - 1) / Self.palier * Self.palier
            target = s.targetRows
            targetKeys = (target + Self.unit - 1) / Self.unit * Self.unit
        }
    }

    package enum Pass: Sendable { case prefix, target }

    /// **The path of a DiT's arithmetic** (`Arithmetic`): the Standard's bf16; the Compact's codes on the GPU alone in
    /// W8A16 (`exact`: the stream under the guard of f16); the shared forms beside the ANE (or told 0 rows of it), on
    /// the Compact's codes or the Standard's weights in f16.
    static func path(_ a: Arithmetic) -> Path {
        switch a {
        case .standard: .bfloat
        case .shared(.codes, .gpu): .half
        case .shared: .shared
        }
    }

    static let layers = QwenImage21DiT.Config.published.layers
    static let d = QwenImage21Functions.d, hd = QwenImage21Functions.hd, heads = QwenImage21Functions.heads
    static let context = QwenImage21DiT.Config.published.context, channels = QwenImage21DiT.Config.published.channels
    /// A layer's group, and the order the loop uses them in (the prefix's pass once, then every evaluation's).
    static let groups = (0..<layers).map { "layer.\($0)" }

    let g: Geometry
    let arithmetic: Arithmetic
    let block: Block
    /// The LoRAs' ranks the functions were built at; none without a LoRA.
    let ranks: Ranks
    var path: Path { Self.path(arithmetic) }
    var stage: SiliCore.Stage { block.stage }
    private var prompted = false
    /// Called after each pass's embedding (`-1`) and after each of its layers, the stream readable (`streamRows`); when
    /// set, the prefix's last layer is computed whole.
    package var observe: ((Pass, Int) async throws -> Void)?

    // MARK: - What the DiT needs

    /// The modulation's rows of a pass: `scale1`, `gate1`, `scale2`, `gate2`, each a row of `mod` (a page of its own).
    private static func gains(_ mod: String) -> [String: Block.Bind] {
        ["scale1": .row(mod, 0), "gate1": .row(mod, 1), "scale2": .row(mod, 2), "gate2": .row(mod, 3)]
    }

    /// The rows each outer function is built for: the timestep's one, the prefix's keys, the target's.
    private static func edgeRows(_ f: String, _ g: Geometry) -> Int { f == "timestep" ? 1 : f == "embed-prefix" ? g.prefixKeys : g.targetKeys }

    /// **The factors' tensors**: the outer products' in f32, each layer's under its group (`<layer>.<product>_down`).
    static func lowRankTensors(_ ranks: Ranks, path: Path) -> [SiliCore.Stage.Tensor] {
        guard !ranks.isEmpty else { return [] }
        return LoRAStack.tensors(QwenImage21Functions.outerLoRA(), ranks: ranks, element: .float32)
            + (0..<layers).flatMap { LoRAStack.tensors(QwenImage21Functions.layerLoRA($0), ranks: ranks, element: path.scalar, prefix: groups[$0] + ".") }
    }

    /// **The DiT's declaration**: its per-row tensors (the stream, its factors, q, the layer's keys and values and the
    /// RoPE tables of each pass, the mask; in the shared forms the steps' own), its other tensors, the outer weights read
    /// once, a group a layer — used by the prefix's pass once, then by every evaluation —, its forms and passes.
    /// `tapped`: the head's norm kept beside the velocity, for the check (`QwenImage21Functions.module`).
    static func declaration(_ g: Geometry, _ arithmetic: Arithmetic, ranks: Ranks = [:], tapped: Bool = false) -> Block.Declaration {
        let path = Self.path(arithmetic), weights = arithmetic.weights, (w, e, k) = (Self.d, path.scalar, g.keys)
        var n = Block.Declaration("Qwen-Image-2.1's DiT (\(path.rawValue), \(g.target) rows, \(g.prefixKeys) prefix keys)", rows: g.rows)
        switch path {
        case .bfloat: n.perRow = [.init("x", width: w, .float32, next: "y")]
        case .half: n.perRow = [.init("x", width: w, .float16, next: "y", factor: "x_factor"), .init("x_factor", width: 1, .float32, next: "y_factor")]
        case .shared: n.perRow = [.init("x", width: w, .float16, next: "y", factor: "x_factor"), .init("x_factor", width: 1, .float16, next: "y_factor")]
        }
        n.perRow += [.init("q", width: w, e), .init("kt", width: w, e, rows: g.targetKeys), .init("vt", width: w, e, rows: g.targetKeys),
                     .init("cos", width: hd, .float32, rows: g.prefixKeys), .init("sin", width: hd, .float32, rows: g.prefixKeys),
                     .init("cos_t", width: hd, .float32, rows: g.targetKeys), .init("sin_t", width: hd, .float32, rows: g.targetKeys),
                     .init("mask", width: k, e, rows: g.prefixKeys)]
        n.perRow += groups.flatMap { l in ["kp", "vp"].map { Block.Rows("\(l).\($0)", width: w, e, rows: g.prefixKeys) } }
        if path == .shared {
            n.perRow += ["qh", "kh", "vh", "o", "mid"].map { .init($0, width: w, .float16) } + ["qkv_factor", "mid_factor"].map { .init($0, width: 1, .float16) }
        }
        let t: [(String, Int)] = [("t_freq", 256 * 4), ("temb", w * 4), ("modulation", 4 * w * 4), ("temb0", w * 4), ("modulation0", 4 * w * 4),
                                  ("final_scale", w * 4), ("text", g.prefixKeys * context * 4), ("references", g.prefixKeys * channels * 4),
                                  ("which", g.prefixKeys * 4), ("tokens", g.targetKeys * channels * 4), ("mask-target", g.targetKeys * k * Layout.element(e)),
                                  ("mod", 4 * path.gainBytes), ("mod0", 4 * path.gainBytes)]
        // The target's gains in the shared forms are tensors of their own (the ANE reads each as a surface).
        let rows = path == .shared ? QwenImage21Shared.gains.map { ($0, w * 2) } : []
        n.tensors = (t + rows).map { SiliCore.Stage.Tensor($0.0, bytes: $0.1) }
            + [SiliCore.Stage.Tensor("velocity", bytes: g.targetKeys * channels * 4, .apart)]
            + (tapped ? [SiliCore.Stage.Tensor("normed", bytes: g.targetKeys * w * 4, .apart)] : [])
            + lowRankTensors(ranks, path: path)
        n.residents = QwenImage21Functions.others()
        let form = path.rawValue
        n.groups = groups.enumerated().map { Block.Group($1, form: form, weights: QwenImage21Functions.layer($0, path: path, weights: weights)) }
        (n.once, n.cycle) = (groups, groups)
        let (pk, tk) = (g.prefixKeys, g.targetKeys)
        let keys: Set<String> = ["kp", "vp", "kt", "vt"]
        let module: @Sendable (String, Block.Build) -> Aicode.Module = { step, b in
            QwenImage21Functions.module(step, path: path, weights: weights, codes: b.codes, rows: b.rows, prefix: pk, target: tk, ranks: ranks)
        }
        if path == .shared {
            n.forms = [Block.Form(form, steps: [.init("pre"), .init("heads", on: .gpu), .init("attention", on: .gpu, whole: keys),
                                                .init("out", bind: ["y": .rows("mid"), "y_factor": .rows("mid_factor")]),
                                                .init("ffn", bind: ["x": .rows("mid"), "x_factor": .rows("mid_factor")])], module: module)]
        } else {
            n.forms = [Block.Form(form, steps: [.init("pre", on: .gpu), .init("post", on: .gpu, whole: keys)], module: module)]
        }
        let prefixPass: [String: Block.Bind] = ["k": .rows("kp"), "v": .rows("vp")]
        let targetPass: [String: Block.Bind] = ["k": .rows("kt"), "v": .rows("vt"), "cos": .rows("cos_t"), "sin": .rows("sin_t"), "mask": .tensor("mask-target")]
        n.passes = [Block.Pass("prefix", rows: 0..<g.prefix, limit: pk, bind: prefixPass.merging(gains("mod0")) { a, _ in a }),
                    Block.Pass("target", rows: 0..<g.target, shareable: 0..<g.target, limit: tk,
                               bind: path == .shared ? targetPass : targetPass.merging(gains("mod")) { a, _ in a })]
        for f in ["timestep", "embed-prefix", "embed-target", "final"] {
            let rows = edgeRows(f, g)
            n.edges[f] = { b in
                QwenImage21Functions.module(f, path: path, weights: weights, codes: b.codes, rows: rows, prefix: pk, target: tk, ranks: ranks, tapped: tapped)
            }
        }
        return n
    }

    // MARK: - Opening

    /// **The DiT of the file at `map` for `sequence`, under B**: its arithmetic read from the file (int8 codes: the Compact,
    /// on the GPU alone under `exact`, in the shared forms under `ane`, the ANE's rows a control's or found in flight;
    /// bf16: the Standard). Under `loras`, their factors written before the first call.
    package static func open(map: String, sequence: QwenImage21DiT.Sequence, loras: LoRAStack = .none, tapped: Bool = false,
                             budget: Int = MemoryBudget.machineCeiling()) async throws -> QwenImage21Loop {
        let file = try MapWeights.dit(map, marker: "img_in.weight")
        guard let probe = file.tensors["transformer_blocks.0.attn.to_q.weight"] else { throw ProgramError("\(map): not Qwen-Image-2.1's DiT") }
        guard ["I8", "BF16", "F32"].contains(probe.dtype) else {
            throw ProgramError("\(map): its products are \(probe.dtype), which no path of the DiT reads")
        }
        let g = Geometry(sequence), ranks = QwenImage21Functions.ranks(loras)
        let arithmetic = try Arithmetic.opened(codes: probe.dtype == "I8", precision: CoreAI.inForce) { declaration(g, $0, ranks: ranks, tapped: tapped) }
        guard let device = MTLCreateSystemDefaultDevice() else { throw ProgramError("no Metal device") }
        let block = try await Block.open(device, declaration(g, arithmetic, ranks: ranks, tapped: tapped), files: [file], budget: budget, share: arithmetic.share)
        let loop = QwenImage21Loop(block, g, arithmetic, ranks: ranks)
        try loop.write(loras)
        return loop
    }

    private init(_ block: Block, _ g: Geometry, _ arithmetic: Arithmetic, ranks: Ranks) {
        (self.block, self.g, self.arithmetic, self.ranks) = (block, g, arithmetic, ranks)
    }

    deinit { _ = block.close() }

    /// The factors of `loras` into their tensors (`LoRAStack.write`), each layer's under its group.
    private func write(_ loras: LoRAStack) throws {
        guard !ranks.isEmpty else { return }
        try loras.write(QwenImage21Functions.outerLoRA(), on: .qwenImage21, ranks: ranks, element: .float32) { stage.pointer($0) }
        for i in 0..<Self.layers {
            try loras.write(QwenImage21Functions.layerLoRA(i), on: .qwenImage21, ranks: ranks, element: path.scalar, prefix: Self.groups[i] + ".") { stage.pointer($0) }
        }
    }

    // MARK: - The passes

    /// **Once a render**: the prefix's tables and masks, its rows embedded — `text` the encoder's text rows in order
    /// `[textRows, 4096]`, `references` each reference's tokens `[R, 64]` —, then the 32 layers over them, each layer's
    /// keys and values kept.
    package func prompt(text: [Float], references: [Float]) async throws {
        precondition(!prompted, "one prompt a loop")
        prompted = true
        let s = g.sequence, (pk, tk) = (g.prefixKeys, g.targetKeys)
        precondition(text.count == s.textRows * Self.context && references.count == s.referenceRows * Self.channels)
        let t = await block.pointer("text").assumingMemoryBound(to: Float.self), r = await block.pointer("references").assumingMemoryBound(to: Float.self)
        let which = await block.pointer("which").assumingMemoryBound(to: Float.self)
        var row = 0, textAt = 0, imageAt = 0
        for segment in s.prefix {
            for _ in 0..<segment.rows {
                if case .text = segment {
                    _ = text.withUnsafeBufferPointer { memcpy(t + row * Self.context, $0.baseAddress! + textAt * Self.context, Self.context * 4) }
                    which[row] = 1; textAt += 1
                } else {
                    _ = references.withUnsafeBufferPointer { memcpy(r + row * Self.channels, $0.baseAddress! + imageAt * Self.channels, Self.channels * 4) }
                    imageAt += 1
                }
                row += 1
            }
        }
        // RoPE: the sequence's angles, the prefix's rows at keys [0, P), the target's at [Pk, Pk + S) (its own tables from
        // their row 0); each pair's angle twice, the sine's sign (−sin, sin); the rows past them at angle 0.
        let rope = QwenImage21Transformer(sequence: s).rope(), half = Self.hd / 2, hd = Self.hd
        let (cp, sp) = (await block.rows("cos").assumingMemoryBound(to: Float.self), await block.rows("sin").assumingMemoryBound(to: Float.self))
        let (ct, st) = (await block.rows("cos_t").assumingMemoryBound(to: Float.self), await block.rows("sin_t").assumingMemoryBound(to: Float.self))
        for key in 0..<pk { for j in 0..<hd { cp[key * hd + j] = 1; sp[key * hd + j] = 0 } }
        for key in 0..<tk { for j in 0..<hd { ct[key * hd + j] = 1; st[key * hd + j] = 0 } }
        for i in 0..<s.rows {
            let (cs, sn, key) = i < g.prefix ? (cp, sp, i) : (ct, st, i - g.prefix)
            for j in 0..<half {
                let (c, sine) = (rope.cos[i * half + j], rope.sin[i * half + j])
                cs[key * hd + 2 * j] = c; cs[key * hd + 2 * j + 1] = c
                sn[key * hd + 2 * j] = -sine; sn[key * hd + 2 * j + 1] = sine
            }
        }
        // The masks, additive: a prefix row sees the keys up to itself in a text segment, those before its picture's end
        // in a picture; the target's rows every real key. The target's keys are zero while the prefix runs.
        let ends = s.segments.flatMap { seg in (seg.start..<seg.end).map { seg.isText ? $0 + 1 : seg.end } }
        await writeMask(await block.rows("mask"), rows: pk) { i, key in key < self.g.prefix && key < (i < self.g.prefix ? ends[i] : self.g.prefix) }
        await writeMask(await block.pointer("mask-target"), rows: tk) { _, key in key < self.g.prefix || (key >= pk && key < pk + self.g.target) }

        try await timestep(0, prefix: true)
        try await block.call("embed-prefix")
        try await observe?(.prefix, -1)
        for i in 0..<Self.layers {
            // The last layer's output reaches nothing: its keys and values, the steps up to them.
            let keysOnly = i == Self.layers - 1 && observe == nil
            try await block.run(Self.groups[i], "prefix", until: keysOnly ? (path == .shared ? "heads" : "pre") : nil)
            try await observe?(.prefix, i)
        }
    }

    private func writeMask(_ base: UnsafeMutableRawPointer, rows: Int, _ visible: (Int, Int) -> Bool) async {
        let k = g.keys
        if path != .bfloat {
            let m = base.assumingMemoryBound(to: Float16.self)
            for i in 0..<rows { for key in 0..<k { m[i * k + key] = visible(i, key) ? 0 : -1e4 } }
        } else {
            let m = base.assumingMemoryBound(to: UInt16.self), masked = UInt16(Float(-1e4).bitPattern >> 16)
            for i in 0..<rows { for key in 0..<k { m[i * k + key] = visible(i, key) ? 0 : masked } }
        }
    }

    /// The modulation's rows at `time` (the sampler's, in [0, 1], × 1000 in the sinusoid): the prefix's (`prefix`, t = 0)
    /// into `mod0` and its temb and modulation beside it, else the target's into `mod`, and the final scale.
    private func timestep(_ time: Float, prefix: Bool) async throws {
        let frequencies = QwenImage21Transformer.timestepFrequencies(time), t = await block.pointer("t_freq").assumingMemoryBound(to: Float.self)
        for (i, v) in frequencies.enumerated() { t[i] = v }
        try await block.call("timestep", prefix ? ["temb": .tensor("temb0"), "modulation": .tensor("modulation0"), "mod": .tensor("mod0")] : [:])
        // The ANE reads each of the target's rows as a surface of its own `[1, 1, d]` (it read a view at an offset into
        // one wrong: Z's, measured 17/11 on the M1 Pro).
        if path == .shared, !prefix {
            let mod = await block.pointer("mod")
            for (k, gain) in QwenImage21Shared.gains.enumerated() { memcpy(await block.pointer(gain), mod + k * path.gainBytes, Self.d * 2) }
        }
    }

    /// **One evaluation**: the latent `[64, h, w]` at the told `time` to the velocity `[64, h, w]` (`model_out`).
    package func evaluate(_ latent: [Float], time: Float) async throws -> [Float] {
        precondition(prompted, "the prefix first")
        let (s, c) = (g.target, Self.channels)
        precondition(latent.count == c * s)
        try await timestep(time, prefix: false)
        // `_pack_latents`: planes to tokens, row-major; the rows past them zero.
        let tokens = await block.pointer("tokens").assumingMemoryBound(to: Float.self)
        latent.withUnsafeBufferPointer { l in for t in 0..<s { for ch in 0..<c { tokens[t * c + ch] = l[ch * s + t] } } }
        try await block.call("embed-target")
        try await observe?(.target, -1)
        for i in 0..<Self.layers {
            try await block.run(Self.groups[i], "target")
            try await observe?(.target, i)
        }
        try await block.call("final")
        let v = await block.pointer("velocity").assumingMemoryBound(to: Float.self)
        var out = [Float](repeating: 0, count: c * s)
        for t in 0..<s { for ch in 0..<c { out[ch * s + t] = v[t * c + ch] } }
        return out
    }

    // MARK: - What a check reads

    /// The stage's plan as a line, and the ANE's rows in the shared forms.
    package var described: String {
        block.stage.plan.line + (path == .shared ? "; the ANE the target's first \(block.units * block.unit) rows of \(g.target)" : "")
    }

    /// The stream's rows `0 ..< n` as true values, `[n, 4096]` f32.
    package func streamRows(_ n: Int) async -> [Float] { await block.values("x", rows: 0..<n) }

    /// A tensor of the stage as f32 values: `count` of them.
    package func floats(_ name: String, count: Int) async -> [Float] {
        Array(UnsafeBufferPointer(start: await block.pointer(name).assumingMemoryBound(to: Float.self), count: count))
    }
}
