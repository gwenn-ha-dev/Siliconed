import CoreAI
import Foundation
import Metal
import SiliCore

/// **Qwen3-VL's vision tower, one picture at a time**: the processor's patches to the rows the
/// language model reads at the picture's slots, as transformers' `Qwen3VLVisionModel` (`modeling_qwen3_vl.py`) runs it
/// in bf16 — the bf16 path, on the GPU alone: both versions' files publish the tower in bf16.
///
///     pixels [N, 1536] ─ patch_embed (a product) ─ + positions (48×48 table, bilinear) ─ 27 blocks ─ merger ─ merged [N/4, d]
///                                                                    blocks 8, 16, 24 ─ deepstack mergers ─ [N/4, d] ×3
///
///     block:   x += proj( SDPA( RoPE₂(q), RoPE₂(k), v ) )   q, k, v = qkv(LN₁(x))        16 heads of 72
///              x += fc2( GELU-tanh( fc1(LN₂(x)) ) )
///     merger:  fc2( GELU-erf( fc1( LN(x) regrouped ×4 ) ) )          the deepstack ones: LN on the regrouped 4608
///
/// **Four functions a picture's grid, the weights their arguments** (`embed`, `block`, `merger`, `deepstack`): one
/// block function serves the 27 blocks, one deepstack function the three deepstack mergers — a layers' stage a picture
/// (`SiliCore.Layers`), the blocks and the mergers its layers in the order they run, every one kept where B holds them
/// (1.15 GB: the tower is a tenth of the language model it feeds), else the blocks through the ring.
///
/// What a port by analogy would miss:
///
///   - **The patches come in 2×2 merge-block order** (`Qwen3VLImages.pixels`): four consecutive rows are one slot of the
///     language model, so a merger's regrouping is a plain reshape.
///   - **The patch embedding is a product**: the 3-D convolution's kernel is its stride, and a row of `pixels` is
///     already one patch laid `(channel, frame, y, x)`.
///   - **The learned positions are a 48×48 table resampled to the picture's grid**, bilinear with aligned corners: four
///     taps a patch, the weights f32 (`Qwen3VLImages.positionTaps`), the sum in f32 and one rounding — on the CPU, from
///     the table read once (`positions`): the grid's rows are a gather the function need not hold.
///   - **RoPE is axial and in f32**: 18 frequencies an axis (θ = 10⁴ over half the head), `[h | w | h | w]` over the 72,
///     applied in halves to q and k widened to f32, then rounded back (`apply_rotary_pos_emb_vision`); the tables are
///     constants of the grid's block function.
///   - **The LayerNorms are torch's fused kernel**: statistics, gain and bias in f32, one rounding to bf16.
///   - **The block's MLP is GELU-tanh, the mergers' GELU-erf**; the main merger normalizes the 1152 before regrouping,
///     a deepstack merger the regrouped 4608.
///   - **Each picture attends within itself** (`cu_seqlens`): a picture's own functions, no mask.
package struct Qwen3VLVision {
    package struct Config: Sendable {
        package let width, heads, hidden, depth, output, patch, merge, temporal, tableSide: Int
        package let deepstack: [Int]
        package let theta, eps: Double

        /// `vision_config` of Qwen3-VL-8B (`text_encoder/config.json` of a pipeline that ships it).
        package static let eightB = Config(width: 1152, heads: 16, hidden: 4304, depth: 27, output: 4096, patch: 16, merge: 2,
                                           temporal: 2, tableSide: 48, deepstack: [8, 16, 24], theta: 10_000, eps: 1e-6)

        package var headDim: Int { width / heads }
        package var patchValues: Int { 3 * temporal * patch * patch }
        var group: Int { merge * merge }
    }

    package let config: Config
    /// What the file puts before each name (`model.visual.`).
    package let prefix: String
    /// The picture's grid in patches, both even.
    package let gridHeight, gridWidth: Int

    /// The products' bf16 weights lie as the engine reads it when the tower opens (`Block.Build.bfloat16`).
    package init(config: Config = .eightB, prefix: String = "model.visual.", gridHeight: Int, gridWidth: Int) {
        (self.config, self.prefix, self.gridHeight, self.gridWidth) = (config, prefix, gridHeight, gridWidth)
    }

    package var patches: Int { gridHeight * gridWidth }
    /// The rows the language model reads: one a 2×2 group.
    package var slots: Int { patches / config.group }

    package enum Tap: String, CaseIterable, Sendable { case block0 = "vit_block0_out", last = "vit_last_hidden" }

    /// The outputs the language model reads: `merged`, then `deepstack<j>`.
    package static func outputs(_ config: Config) -> [String] { ["merged"] + config.deepstack.indices.map { "deepstack\($0)" } }

    // MARK: - The CPU's tables

    /// **The axial RoPE table**, `[N, 72]` cos and sin in f32 as torch computes them (`inv_freq` by an f32 `pow`, the
    /// angle an f32 product).
    package func rope() -> (cos: [Float], sin: [Float]) {
        let c = config, axis = c.headDim / 2
        let frequencies = stride(from: 0, to: axis, by: 2).map { 1 / powf(Float(c.theta), Float($0) / Float(axis)) }
        var cos: [Float] = [], sin: [Float] = []
        cos.reserveCapacity(patches * c.headDim); sin.reserveCapacity(patches * c.headDim)
        for (row, column) in Qwen3VLImages.patchPositions(gridHeight: gridHeight, gridWidth: gridWidth, merge: c.merge) {
            let angles = frequencies.map { Float(row) * $0 } + frequencies.map { Float(column) * $0 }
            for _ in 0..<2 { for a in angles { let (c, s) = cosSin(Double(a)); cos.append(c); sin.append(s) } }
        }
        return (cos, sin)
    }

    /// **The learned positions resampled to the grid**, `[N, width]` as bf16 bits: each patch's four taps of `table`
    /// (`[side², width]`, the file's bf16 widened) summed in f32 with their f32 weights, rounded once.
    package func positions(table: UnsafePointer<Float>, into out: UnsafeMutablePointer<UInt16>) {
        let c = config
        let taps = Qwen3VLImages.positionTaps(gridHeight: gridHeight, gridWidth: gridWidth, side: c.tableSide, merge: c.merge)
        nonisolated(unsafe) let (table, out) = (table, out)
        DispatchQueue.concurrentPerform(iterations: patches) { n in
            let row = out + n * c.width
            for d in 0..<c.width {
                var sum: Float = 0
                for t in 0..<4 { sum += table[taps.indices[4 * n + t] * c.width + d] * taps.weights[4 * n + t] }
                row[d] = BFloat16Bits.rounded(sum)
            }
        }
    }

    // MARK: - The functions

    typealias Argument = (name: String, shape: [Int], element: Aicode.Element)

    /// The patch embedding and the positions: `[N, 1536]` f32 in, the stream `[N, width]` out.
    static func embedArguments(_ c: Config, rows n: Int) -> [Argument] {
        [("pixels", [n, c.patchValues], .f32), ("kernel", [c.width, c.patchValues], .bf16), ("kernel_bias", [c.width], .bf16),
         ("positions", [n, c.width], .bf16)]
    }

    /// A product's weight `[out, in]`, as the file lays it (`Recipe.product`).
    static func product(_ name: String, _ out: Int, _ inn: Int) -> Argument { (name, [out, inn], .bf16) }

    static func blockArguments(_ c: Config, rows n: Int) -> [Argument] {
        [("x", [n, c.width], .bf16), ("norm1", [c.width], .bf16), ("norm1_bias", [c.width], .bf16),
         product("qkv", 3 * c.width, c.width), ("qkv_bias", [3 * c.width], .bf16),
         product("proj", c.width, c.width), ("proj_bias", [c.width], .bf16),
         ("norm2", [c.width], .bf16), ("norm2_bias", [c.width], .bf16),
         product("fc1", c.hidden, c.width), ("fc1_bias", [c.hidden], .bf16), flat(product("fc2", c.width, c.hidden)), ("fc2_bias", [c.width], .bf16)]
    }

    /// **A weight given flat**, its shape back inside the function (`Writer.a`): the block's fc2 `[1152, 4304]` bf16 lies
    /// in rows of 8608 bytes, which Core AI binds padded to 8640 (its descriptor, read 19/11) and so copies at every
    /// call; flat, as it lies.
    static func flat(_ a: Argument) -> Argument { (a.name, [a.shape.reduce(1, *)], a.element) }

    /// A merger: the main one normalizes the stream's width, a deepstack one the regrouped width.
    static func mergerArguments(_ c: Config, rows n: Int, deepstack: Bool) -> [Argument] {
        let wide = c.group * c.width, normed = deepstack ? wide : c.width
        return [("x", [n, c.width], .bf16), ("norm", [normed], .bf16), ("norm_bias", [normed], .bf16),
                product("fc1", wide, wide), ("fc1_bias", [wide], .bf16), product("fc2", c.output, wide), ("fc2_bias", [c.output], .bf16)]
    }

    /// Each function's builder on one block: its arguments by name, a product with its bias, torch's LayerNorm.
    private struct Writer {
        let b: Aicode.Block, names: [String], eps: Double
        /// The weights given flat (`flat`) and their shapes.
        var shapes: [String: [Int]] = [:]
        func a(_ name: String) -> Int {
            let i = names.firstIndex(of: name)!
            return shapes[name].map { b.reshape(i, $0) } ?? i
        }
        /// `x · Wᵀ + b`, the product rounded to bf16, then the bias added in bf16.
        func linear(_ x: Int, _ name: String) -> Int { b.binary(.add, b.linear(x, a(name)), a(name + "_bias")) }
        /// `F.layer_norm` on bf16: statistics, gain and bias in f32, one rounding.
        func layerNorm(_ x: Int, _ name: String, axis: Int) -> Int {
            let normed = b.layerNorm(b.cast(x, to: .f32), axes: [axis], epsilon: b.scalar(eps, .f32))
            return b.cast(b.binary(.add, b.binary(.mul, normed, b.cast(a(name), to: .f32)), b.cast(a(name + "_bias"), to: .f32)), to: .bf16)
        }
    }

    private static func module(_ arguments: [Argument], eps: Double, shapes: [String: [Int]] = [:]) -> (Aicode.Module, Writer) {
        let m = Aicode.Module(arguments: arguments.map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        return (m, Writer(b: m.body, names: m.argumentNames, eps: eps, shapes: shapes))
    }

    static func embedModule(_ c: Config, rows n: Int) -> Aicode.Module {
        // The patch kernel is the file's convolution `[width, 3, 2, 16, 16]` read flat `[out, in]`.
        let (made, w) = module(embedArguments(c, rows: n), eps: c.eps)
        var m = made
        let b = w.b
        let x = b.binary(.add, b.linear(b.cast(w.a("pixels"), to: .bf16), w.a("kernel")), w.a("kernel_bias"))
        // Written into the stream's current buffer, which the first block reads.
        m.outputs = [("x", b.binary(.add, x, w.a("positions")))]
        return m
    }

    /// One block on the grid's rows, its RoPE tables constants, its products' weights `[out, in]`.
    func blockModule() -> Aicode.Module {
        let c = config, n = patches, d = c.headDim
        let shapes = ["fc2": [c.width, c.hidden]]
        let (made, w) = Self.module(Self.blockArguments(c, rows: n), eps: c.eps, shapes: shapes)
        var m = made
        let (cosValues, sinValues) = rope()
        let cos = m.constant("rope_cos", cosValues.withUnsafeBytes { Array($0) }, Aicode.tensor([n, 1, d], .f32))
        let sin = m.constant("rope_sin", sinValues.withUnsafeBytes { Array($0) }, Aicode.tensor([n, 1, d], .f32))
        let b = w.b, x = w.a("x")
        let qkv = b.reshape(w.linear(w.layerNorm(x, "norm1", axis: 1), "qkv"), [n, 3, c.heads, d])
        func part(_ i: Int) -> Int { b.reshape(b.slice(qkv, axis: 1, start: i, size: 1), [n, c.heads, d]) }
        // x · cos + rotate_half(x) · sin on [N, H, D], in f32, rounded once.
        func rotated(_ t: Int) -> Int {
            let wide = b.cast(t, to: .f32)
            let half = b.concat([b.scale("mul", b.slice(wide, axis: 2, start: d / 2, size: d / 2), -1), b.slice(wide, axis: 2, start: 0, size: d / 2)], axis: 2)
            return b.cast(b.binary(.add, b.binary(.mul, wide, cos), b.binary(.mul, half, sin)), to: .bf16)
        }
        func headsFirst(_ t: Int) -> Int { b.reshape(b.transpose(t, [1, 0, 2]), [1, c.heads, n, d]) }
        let o = b.attention([headsFirst(rotated(part(0))), headsFirst(rotated(part(1))), headsFirst(part(2))],
                            scale: 1 / Double(d).squareRoot(), symbol: "sdpa_qwen3vl_vision")
        let rowsFirst = b.reshape(b.transpose(b.reshape(o, [c.heads, n, d]), [1, 0, 2]), [n, c.width])
        let x1 = b.binary(.add, x, w.linear(rowsFirst, "proj"))
        let hidden = b.gelu(w.linear(w.layerNorm(x1, "norm2", axis: 1), "fc1"), tanh: true)
        m.outputs = [("y", b.binary(.add, x1, w.linear(hidden, "fc2")))]
        return m
    }

    static func mergerModule(_ c: Config, rows n: Int, deepstack: Bool) -> Aicode.Module {
        let (made, w) = module(mergerArguments(c, rows: n, deepstack: deepstack), eps: c.eps)
        var m = made
        let b = w.b, wide = c.group * c.width
        let regrouped = deepstack ? w.layerNorm(b.reshape(w.a("x"), [n / c.group, wide]), "norm", axis: 1)
                                  : b.reshape(w.layerNorm(w.a("x"), "norm", axis: 1), [n / c.group, wide])
        // Its own rows (`<merger>.out`), the stream left as it is.
        m.outputs = [("out", w.linear(b.gelu(w.linear(regrouped, "fc1"), tanh: false), "fc2"))]
        return m
    }

    // MARK: - The weights

    /// A block's weights or a merger's, each argument by its name, the file's key under `stem`: the products `[out, in]`
    /// (`Recipe.product`, by their keys without `.weight`), the rest as they lie.
    func entries(_ stem: String, _ parts: [String: String]) -> [Weights.Entry] {
        parts.sorted { $0.key < $1.key }.flatMap { name, part -> [Weights.Entry] in
            let key = prefix + stem + "." + part
            let weight: Weights.Entry = Self.products.contains(name) ? (name, key, .product(joined: [])) : (name, key + ".weight", .bfloat16(joined: []))
            return [weight, (name + "_bias", key + ".bias", .bfloat16(joined: []))]
        }
    }

    static let blockParts = ["norm1": "norm1", "qkv": "attn.qkv", "proj": "attn.proj", "norm2": "norm2", "fc1": "mlp.linear_fc1", "fc2": "mlp.linear_fc2"]
    static let mergerParts = ["norm": "norm", "fc1": "linear_fc1", "fc2": "linear_fc2"]

    /// The products of the blocks and the mergers, by their arguments' names.
    static let products: Set<String> = ["qkv", "proj", "fc1", "fc2"]

    /// The layers in the order they run: the blocks, each deepstack merger after its block, the merger last.
    func groups() -> [Block.Group] {
        let c = config
        var all: [Block.Group] = []
        for i in 0..<c.depth {
            all.append(Block.Group("blocks.\(i)", form: "block", weights: entries("blocks.\(i)", Self.blockParts)))
            if let j = c.deepstack.firstIndex(of: i) {
                all.append(Block.Group("deepstack.\(j)", form: "deepstack", weights: entries("deepstack_merger_list.\(j)", Self.mergerParts)))
            }
        }
        return all + [Block.Group("merger", form: "merger", weights: entries("merger", Self.mergerParts))]
    }

    // MARK: - The layers

    /// **What one picture's layers need**: the pixels, the positions and the stream a row; the outputs apart (a merger's
    /// own, `<merger>.out`); the patch embedding's weights resident (the table of positions read by the CPU before, not
    /// here); the blocks and mergers; the grid's four functions.
    func declaration() -> Layers.Declaration {
        let c = config, n = patches, v = self
        var d = Layers.Declaration("Qwen3-VL's vision tower on \(gridHeight)×\(gridWidth) patches", rows: n)
        d.perRow = [.init("pixels", width: c.patchValues, .float32), .init("positions", width: c.width, .bfloat16),
                    .init("x", width: c.width, .bfloat16, next: "y")]
        d.layers = groups()
        d.tensors = d.layers.filter { $0.form != "block" }.map { .init("\($0.name).out", bytes: slots * c.output * 2, .apart) }
        d.residents = [("kernel", prefix + "patch_embed.proj.weight", .bfloat16(joined: [])), ("kernel_bias", prefix + "patch_embed.proj.bias", .bfloat16(joined: []))]
        d.kinds = ["block": { _ in v.blockModule() },
                   "merger": { _ in Self.mergerModule(c, rows: n, deepstack: false) },
                   "deepstack": { _ in Self.mergerModule(c, rows: n, deepstack: true) }]
        d.edges["embed"] = { _ in Self.embedModule(c, rows: n) }
        return d
    }

    // MARK: - The run

    /// **The pictures through the tower**, a layers' stage a picture, each planned under B: `merged` and `deepstack<j>`,
    /// each the pictures' rows one after the other; with `taps`, their streams after block 0 and the last.
    ///
    /// What the CPU holds beside the stage is made before its plan, so that its floor counts it: the positions
    /// resampled (the table widened, then let go) and the f32 arrays the outputs are read into. Made while the stage
    /// was open, they passed P by 32 MiB at 26 × 38 patches (`check qwen-image-2.1 vision`, 1.28 GiB for 1.25).
    package static func rows(_ pictures: [Qwen3VLImages.Pixels], config: Config = .eightB, prefix: String = "model.visual.",
                             map path: String, taps: Set<Tap> = []) throws -> [String: [Float]] {
        guard !pictures.isEmpty else { return [:] }
        return try blocking {
            guard let device = MTLCreateSystemDefaultDevice() else { throw Failure("no Metal device") }
            let file = try SafeTensors(path)
            var rows: [String: [Float]] = [:]
            for picture in pictures {
                let vision = Qwen3VLVision(config: config, prefix: prefix, gridHeight: picture.gridHeight, gridWidth: picture.gridWidth)
                let positions = try vision.positions(file)
                var out = vision.outputs(taps)
                let layers = try await Layers.open(device, vision.declaration(), files: [file], budget: MemoryBudget.machineCeiling())
                // The pixels live in the picture's pages: it is held until the run has copied them.
                try await vision.run(layers, pixels: UnsafeBufferPointer(picture.values), positions: positions, into: &out)
                withExtendedLifetime(picture) {}
                layers.close()
                for (name, values) in out { rows[name, default: []] += values }
            }
            return rows
        }
    }

    /// The positions of this grid, bf16 bits `[N, width]`, from the file's table (widened for the while).
    func positions(_ file: SafeTensors) throws -> [UInt16] {
        let key = prefix + "pos_embed.weight", count = try file.real(key).shape.reduce(1, *)
        // Pages, let go at once: nothing of them stays in the footprint the plan reads next.
        let table = Pages(bytes: count * 4), raw = Pages(bytes: count * 2)
        try file.readBFloat16(key, into: raw.base)
        let bits = raw.base.assumingMemoryBound(to: UInt16.self)
        for i in 0..<count { table.floats()[i] = BFloat16Bits.widened(bits[i]) }
        withExtendedLifetime(raw) {}
        var out = [UInt16](repeating: 0, count: patches * config.width)
        withExtendedLifetime(table) { out.withUnsafeMutableBufferPointer { positions(table: table.floats(), into: $0.baseAddress!) } }
        return out
    }

    /// The f32 arrays a picture's outputs (and taps) are read into, every value written now.
    func outputs(_ taps: Set<Tap>) -> [String: [Float]] {
        var out = Dictionary(uniqueKeysWithValues: Self.outputs(config).map { ($0, [Float](repeating: 0, count: slots * config.output)) })
        for t in taps { out[t.rawValue] = [Float](repeating: 0, count: patches * config.width) }
        return out
    }

    /// One picture on its layers: the pixels and positions in, the outputs (and taps) widened into `out`'s arrays, made
    /// before the stage.
    private func run(_ layers: Layers, pixels values: UnsafeBufferPointer<Float>, positions: [UInt16], into out: inout [String: [Float]]) async throws {
        let c = config, n = patches
        precondition(values.count == n * c.patchValues, "pixels: \(values.count) values for \(n) patches")
        (await layers.rows("pixels")).copyMemory(from: values.baseAddress!, byteCount: values.count * 4)
        let at = await layers.rows("positions")
        positions.withUnsafeBytes { at.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        try await layers.call("embed")
        // Widened in place into the arrays made before the stage.
        func read(_ tensor: UnsafeMutableRawPointer, into key: String) {
            let p = tensor.assumingMemoryBound(to: UInt16.self)
            out[key]!.withUnsafeMutableBufferPointer { o in for i in o.indices { o[i] = BFloat16Bits.widened(p[i]) } }
        }
        let names = layers.declaration.layers.map(\.name), taps: [String: Tap] = ["blocks.0": .block0, "blocks.\(c.depth - 1)": .last]
        try await layers.forward(after: { i in
            if let tap = taps[names[i]], out[tap.rawValue] != nil { read(await layers.rows("x"), into: tap.rawValue) }
        })
        for (name, group) in zip(Self.outputs(c), ["merger"] + c.deepstack.indices.map { "deepstack.\($0)" }) {
            read(await layers.pointer("\(group).out"), into: name)
        }
    }
}
