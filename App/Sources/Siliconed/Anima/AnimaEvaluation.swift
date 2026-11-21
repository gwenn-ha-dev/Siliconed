import CoreAI
import Foundation
import Metal
import SiliCore

/// **Anima's DiT and its loop**: its functions at fixed shapes (`AnimaFunction`, `AnimaShared`), declared to the engine
/// as a block (`Block`: its stage under B, its calls bound and put ahead, the ANE's share), the loop in Swift.
///
///     once a prompt      context ×28: each block's cross keys and values, on its weights held (`Block.Group.held`)
///     an evaluation      time · embed · blocks ×28 (the modulation, then the block's form over the image's rows) · final
///
/// **The rows**: the image's tokens, padded to the keys' palier (`Geometry.palier`), the padding computed and masked
/// (−10⁴ on its keys), never read. The cross-attention's keys: the context's rows at a palier of 256, its padding
/// masked (the reference's 512 rows, zeros past the prompt, are all attended: a context of 512 rows has no padding).
///
/// **The residual stream** `x` is two buffers in turn; under the guard of f16 (the Compact) beside its row factors, in
/// f32 for the guarded forms (`f`), in f16 for the shared forms (`x_factor`): a block reads one buffer and writes the
/// other. The padding's rows are held under a factor of 1: x̂ = 0 normalizes to 0, never to a NaN its keys would spread.
///
/// **The forms**: one a layout of the blocks' weights — the Standard's bf16, the Compact's on codes but for the
/// products it leaves in bf16 (blocks 0 and 27, the MLP of 1 to 3) —, each `pre` then `post` on the GPU. **Under the
/// precision ane**, the blocks whose ten products are codes take the shared forms (`AnimaShared`): the ANE pre, out,
/// crossout and ffn on units of the image's first rows (`Block.Share`: a control's rows, or found in flight), the GPU
/// the rest; the stream's factors pass from f32 to f16 entering a shared block and back leaving it, every factor a
/// power of two (exact), on the CPU before the block, as the gates' factors are.
package final class AnimaEvaluation {
    typealias Path = AnimaFunction.Path

    package struct Geometry: Equatable, Sendable {
        /// The latent's sides, the image's tokens and its rows padded (the self-attention's keys), the context's palier.
        package let height, width, tokens, keys, context: Int

        /// The keys' palier: every call over the image's rows, rounded up to 64 (`Block`), stays within them.
        static let palier = 256

        package init(height: Int, width: Int, context rows: Int) {
            (self.height, self.width) = (height, width)
            tokens = (height / AnimaFunction.patch) * (width / AnimaFunction.patch)
            keys = (tokens + Self.palier - 1) / Self.palier * Self.palier
            context = max(512, (rows + 255) / 256 * 256)
        }
    }

    static let layers = 28, d = AnimaFunction.d, heads = AnimaFunction.heads, hd = AnimaFunction.hd
    /// **The DiT's file**, read under the prefix it keeps the DiT's names under (`model.diffusion_model.` in the published
    /// file, `net.` in Cosmos-Predict2's own, none in some fine-tunes), found where it puts `t_embedding_norm.weight`, a
    /// tensor of the DiT alone (`MapWeights.dit`). The conditioner, in the same file, is read under it too.
    package static func file(_ path: String) throws -> SafeTensors { try MapWeights.dit(path, marker: "t_embedding_norm.weight") }
    static func block(_ i: Int) -> String { "blocks.\(i)" }
    static let blocks = (0..<layers).map(block)

    /// The guarded forms' path: the Standard's bf16, f16 otherwise.
    static func path(_ a: Arithmetic) -> Path { a == .standard ? .bf16 : .f16 }

    /// **The blocks the shared forms take**: beside the ANE (or told 0 rows of it), the Compact's whose ten products are
    /// codes (0 and 27 are float weights, the MLP of 1 to 3 too: the guarded forms'), every block of the Standard's in
    /// f16; none on the GPU alone, where the Compact is the guarded forms' — but where its products are int8 × int8
    /// (`SharedWeights.integer`: the guarded forms read the codes in W8A16 only).
    static func sharedBlocks(_ a: Arithmetic, _ quantized: [Set<String>]) -> Set<Int> {
        switch a {
        case .standard, .shared(.codes, .gpu): []
        case .shared(.half, _): Set(quantized.indices)
        case .shared(.codes, _), .shared(.integer, _): Set(quantized.indices.filter { quantized[$0] == Set(products.keys) })
        }
    }

    let g: Geometry
    let path: Path
    let block: Block
    /// Each block's products on codes, by their arguments' names; the blocks in the shared forms (none: the guarded
    /// forms alone).
    let quantized: [Set<String>]
    let sharedBlocks: Set<Int>
    /// Block 0's branches written by its post (the oracle's boundaries; a check's alone).
    let taps: Bool
    /// Whether the stream's factors are the shared forms' (f16 `x_factor`), else the guarded forms' (f32 `f`).
    private var half = false
    /// What `evaluate` kept of its boundaries, by the oracle's names.
    package var tapped: [String: [Float]] = [:]

    var element: NDArray.ScalarType { path == .bf16 ? .bfloat16 : .float16 }
    var guarded: Bool { path == .f16 }

    // MARK: - What the DiT reads

    /// The file's path: the Compact's codes, or the Standard's bf16.
    static func path(of file: SafeTensors) -> Path {
        file.tensors["blocks.1.self_attn.q_proj.weight"]?.dtype == "I8" ? .f16 : .bf16
    }

    /// Block i's products the file holds as codes (the Compact leaves blocks 0 and 27 and the MLP of blocks 1 to 3 in bf16).
    static func quantized(_ i: Int, in file: SafeTensors) -> Set<String> {
        Set(products.filter { file.tensors["blocks.\(i)." + $0.value + ".weight"]?.dtype == "I8" }.map(\.key))
    }

    /// A block's products: their arguments' names, their modules.
    static let products = ["q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.output_proj",
                           "cq": "cross_attn.q_proj", "ck": "cross_attn.k_proj", "cv": "cross_attn.v_proj", "co": "cross_attn.output_proj",
                           "m1": "mlp.layer1", "m2": "mlp.layer2"]

    /// Block i's function of `kind`: its own products on codes, the codes laid out as `codes`.
    static func function(_ kind: AnimaFunction.Kind, _ path: Path, block q: Set<String>, codes: CodesLayout, rows: Int = 1, keys: Int = 0,
                         context: Int = 0, taps: Bool = false, ranks: [String: Int] = [:]) -> AnimaFunction {
        let own = q.intersection(AnimaFunction.products(kind)), adapted = AnimaFunction.adapted(kind)
        return AnimaFunction(kind: kind, path: path, quantized: own, rows: rows, keys: keys, context: context, taps: taps,
                             codes: own.isEmpty ? .inOut : codes, ranks: ranks.filter { adapted.contains($0.key) })
    }

    // MARK: - The LoRAs

    /// The modules a LoRA adapts in a block, by their arguments' names: its products and its modulation's.
    static let adapted = products.merging(["sa1": "adaln_modulation_self_attn.1", "sa2": "adaln_modulation_self_attn.2",
                                           "ca1": "adaln_modulation_cross_attn.1", "ca2": "adaln_modulation_cross_attn.2",
                                           "ml1": "adaln_modulation_mlp.1", "ml2": "adaln_modulation_mlp.2"]) { a, _ in a }

    /// Block i's adapted products, as the LoRAs read them (the module's name, without `.weight`).
    static func loraProducts(block i: Int) -> [LoRAProduct] {
        adapted.sorted { $0.key < $1.key }.map { name, module in
            let (inn, out) = AnimaFunction.dims(name)
            return LoRAProduct(name: name, key: "blocks.\(i).\(module)", inn: inn, out: out)
        }
    }

    /// Each adapted product's rank in the functions: the same in every block (`LoRAStack.rank`), none where nothing lands.
    static func ranks(_ loras: LoRAStack) -> [String: Int] {
        var ranks: [String: Int] = [:]
        for (name, module) in adapted {
            let r = loras.rank(of: blocks.indices.map { "blocks.\($0).\(module)" }, on: .anima)
            if r > 0 { ranks[name] = r }
        }
        return ranks
    }

    static func factorElement(_ path: Path) -> NDArray.ScalarType { path == .bf16 ? .bfloat16 : .float16 }

    // MARK: - The weights

    /// A product's entry: the codes (`<key>.weight`, `.weight_scale`), or the float weight transposed in the path's element.
    private static func product(_ name: String, _ key: String, _ path: Path, quantized: Bool) -> Weights.Entry {
        switch path {
        case .bf16: (name, key + ".weight", .bfloatTransposed)
        case .f16: quantized ? (name, key, .codes) : (name, key + ".weight", .halfTransposed)
        }
    }

    private static func gain(_ name: String, _ key: String, _ path: Path) -> Weights.Entry {
        (name, key, path == .bf16 ? .bfloat16(joined: []) : .half)
    }

    /// Block i's group in the guarded forms: its modulation, its self-attention, its cross-attention's query and output,
    /// its MLP.
    static func entries(block i: Int, _ path: Path, quantized: Set<String>) -> [Weights.Entry] {
        func product(_ name: String) -> Weights.Entry { Self.product(name, "blocks.\(i)." + products[name]!, path, quantized: quantized.contains(name)) }
        let p = "blocks.\(i)."
        return modulationEntries(block: i) + ["q", "k", "v", "o", "cq", "co", "m1", "m2"].map(product)
            + [gain("q_norm", p + "self_attn.q_norm.weight", path), gain("k_norm", p + "self_attn.k_norm.weight", path),
               gain("cq_norm", p + "cross_attn.q_norm.weight", path)]
    }

    /// Block i's modulation, bf16 on both paths: the one-row function widens it.
    static func modulationEntries(block i: Int) -> [Weights.Entry] {
        var all: [Weights.Entry] = []
        for (name, key) in [("sa", "adaln_modulation_self_attn"), ("ca", "adaln_modulation_cross_attn"), ("ml", "adaln_modulation_mlp")] {
            let k: String = "blocks.\(i)." + key
            all.append((name + "1", k + ".1.weight", .bfloatTransposed))
            all.append((name + "2", k + ".2.weight", .bfloatTransposed))
        }
        return all
    }

    /// Block i's cross keys' and values' products, read once a prompt: held, whatever B.
    static func contextEntries(block i: Int, _ path: Path, quantized q: Set<String>) -> [Weights.Entry] {
        let p = "blocks.\(i).cross_attn."
        return [product("ck", p + "k_proj", path, quantized: q.contains("ck")), product("cv", p + "v_proj", path, quantized: q.contains("cv")),
                gain("ck_norm", p + "k_norm.weight", path)]
    }

    /// The time path's, the embedder's and the final layer's.
    static func others(_ path: Path) -> [Weights.Entry] {
        [("t1", "t_embedder.1.linear_1.weight", .bfloatTransposed), ("t2", "t_embedder.1.linear_2.weight", .bfloatTransposed),
         ("t_norm", "t_embedding_norm.weight", .bfloat16(joined: [])), ("f1", "final_layer.adaln_modulation.1.weight", .bfloatTransposed),
         ("f2", "final_layer.adaln_modulation.2.weight", .bfloatTransposed),
         product("xe", "x_embedder.proj.1", path, quantized: false), product("fl", "final_layer.linear", path, quantized: false)]
    }

    // MARK: - The declaration

    /// The guarded form of a block whose products on codes are `q` (and block 0's under taps, its branches besides).
    static func guardedForm(_ q: Set<String>, taps: Bool) -> String {
        let own = q.subtracting(["ck", "cv"]).sorted()
        return "guarded" + (own.isEmpty ? "" : "-" + own.joined(separator: "+")) + (taps ? "-taps" : "")
    }

    /// The context's edge of a block whose products on codes are `q`.
    static func contextEdge(_ q: Set<String>) -> String {
        let own = q.intersection(["ck", "cv"]).sorted()
        return "context" + (own.isEmpty ? "" : "-" + own.joined(separator: "+"))
    }

    /// The modulation's rows (`self`, `cross`, `mlp`: shift, 1 + scale, gate) as the guarded forms read them, a row each.
    static let rows: [String: Block.Bind] = {
        var all: [String: Block.Bind] = [:]
        for (k, sublayer) in ["self", "cross", "mlp"].enumerated() {
            for (j, gain) in ["shift", "scale", "gate"].enumerated() { all["\(gain)\(k + 1)"] = .row(sublayer, j) }
        }
        return all
    }()

    /// **The DiT's declaration**: its per-row tensors (the stream and its factors, q k v, the RoPE tables; in the shared
    /// forms the steps' own; block 0's branches under taps), its other tensors (each block's cross keys and values its
    /// own), the edges' weights read once, a group a block (its context products held), its forms, the image's pass.
    static func declaration(_ g: Geometry, _ arithmetic: Arithmetic, quantized: [Set<String>], taps: Bool, ranks: [String: Int]) -> Block.Declaration {
        let path = Self.path(arithmetic), shared = sharedBlocks(arithmetic, quantized)
        let (k, c, e) = (g.keys, g.context, factorElement(path)), (bytes, gainRow) = (Layout.element(e), AnimaFunction.gainRow)
        var n = Block.Declaration("Anima's DiT (\(path)\(shared.isEmpty ? "" : ", shared"), \(g.height)×\(g.width), \(k) keys)", rows: k)
        n.perRow = [.init("x", width: d, e, next: "y", factor: path == .f16 ? "f" : nil), .init("q", width: d, e), .init("k", width: d, e),
                    .init("v", width: d, e), .init("cos", width: hd / 2, .float32), .init("sin", width: hd / 2, .float32)]
        if path == .f16 { n.perRow.append(.init("f", width: 1, .float32, next: "f_next")) }
        if !shared.isEmpty {
            n.perRow += [.init("x_factor", width: 1, .float16, next: "y_factor"), .init("qkv_factor", width: 1, .float16),
                         .init("attended", width: d, .float16), .init("mid", width: d, .float16, factor: "mid_factor"), .init("mid_factor", width: 1, .float16),
                         .init("mid2", width: d, .float16, factor: "mid2_factor"), .init("mid2_factor", width: 1, .float16),
                         .init("cqh", width: d, .float16), .init("cq_factor", width: 1, .float16)]
                + ["qh", "kh", "vh"].map { .init($0, width: d, .float16) }
        }
        // Block 0's branches: f32 from its guarded form, f16 from the shared forms (`AnimaShared`'s taps, a run tapped).
        let listening = taps && shared.contains(0)
        if taps { n.perRow += ["attn1", "attn2", "ff"].map { .init($0, width: d, listening ? .float16 : .float32) } }
        n.tapped = listening
        let latent = AnimaFunction.channels * g.height * g.width * 4
        var t: [(String, Int)] = [("latent", latent), ("t_freq", d * 4), ("temb", 3 * d * 4), ("embedded", d * 4), ("final", 2 * gainRow * bytes),
                                  ("mask", k * bytes), ("context", c * AnimaFunction.textDim * bytes), ("ctx_mask", c * bytes)]
        t += ["self", "cross", "mlp"].map { ($0, 3 * gainRow * bytes) }
        t += blocks.flatMap { b in ["ctx_k", "ctx_v"].map { ("\(b).\($0)", c * d * bytes) } }
        if !shared.isEmpty { t += AnimaShared.gains.map { ($0, d * 2) } + AnimaShared.gateFactors.map { ($0, 2) } }
        n.tensors = t.map { SiliCore.Stage.Tensor($0.0, bytes: $0.1) } + [SiliCore.Stage.Tensor("velocity", bytes: latent, .apart)]
            + blocks.indices.flatMap { LoRAStack.tensors(loraProducts(block: $0), ranks: ranks, element: e, prefix: "\(block($0)).") }
        n.residents = others(path)
        func form(_ i: Int) -> String { shared.contains(i) ? "shared" : guardedForm(quantized[i], taps: taps && i == 0) }
        n.groups = blocks.indices.map { i in
            Block.Group(block(i), form: form(i), weights: shared.contains(i) ? AnimaShared.layer(i, weights: arithmetic.weights) : entries(block: i, path, quantized: quantized[i]),
                        held: contextEntries(block: i, path, quantized: quantized[i]))
        }
        n.cycle = blocks
        n.codes = arithmetic.weights.layout
        // The guarded forms read the f32 factors; the activations' q beside the product's weight q.
        let factors: [String: Block.Bind] = path == .f16 ? ["x_factor": .rows("f"), "y_factor": .next("f")] : [:]
        var made = Set<String>()
        for i in blocks.indices where !shared.contains(i) && made.insert(form(i)).inserted {
            let (q, tapping) = (quantized[i], taps && i == 0)
            n.forms.append(Block.Form(form(i), steps: [.init("pre", on: .gpu), .init("post", on: .gpu, whole: ["k", "v"], bind: ["q": .rows("q")])],
                                      bind: rows.merging(factors) { a, _ in a }) { step, b in
                step == "pre" ? function(.pre, path, block: q, codes: b.codes, rows: b.rows, ranks: ranks).module()
                    : function(.post, path, block: q, codes: b.codes, rows: b.rows, keys: k, context: c, taps: tapping, ranks: ranks).module()
            })
        }
        if !shared.isEmpty {
            n.forms.append(Block.Form("shared", steps: [
                .init("pre"), .init("heads", on: .gpu), .init("attention", on: .gpu, whole: ["k", "v"], bind: ["q": .rows("q"), "o": .rows("attended")]),
                .init("out", bind: ["y": .rows("mid"), "y_factor": .rows("mid_factor")], taps: listening),
                .init("cross", on: .gpu, bind: ["o": .rows("attended")]),
                .init("crossout", bind: ["x": .rows("mid"), "x_factor": .rows("mid_factor"), "y": .rows("mid2"), "y_factor": .rows("mid2_factor")],
                      taps: listening),
                .init("ffn", bind: ["x": .rows("mid2"), "x_factor": .rows("mid2_factor")], taps: listening)]) { step, b in
                AnimaShared.module(step + (b.tapped ? "-taps" : ""), rows: b.rows, keys: k, context: c, weights: arithmetic.weights, ranks: ranks,
                                   neural: b.neural)
            })
        }
        n.passes = [Block.Pass("image", rows: 0..<g.tokens, shareable: 0..<g.tokens)]
        n.edges["time"] = { _ in AnimaFunction(kind: .time, path: path).module() }
        n.edges["modulation"] = { b in function(.modulation, path, block: [], codes: b.codes, ranks: ranks).module() }
        n.edges["embed"] = { _ in AnimaFunction(kind: .embed, path: path, rows: g.tokens, height: g.height, width: g.width).module() }
        n.edges["final"] = { _ in AnimaFunction(kind: .final, path: path, rows: g.tokens, height: g.height, width: g.width).module() }
        for q in Set(quantized.map { $0.intersection(["ck", "cv"]) }) {
            n.edges[contextEdge(q)] = { b in function(.context, path, block: q, codes: b.codes, context: c, ranks: ranks).module() }
        }
        return n
    }

    // MARK: - Opening

    private init(_ block: Block, _ g: Geometry, path: Path, quantized: [Set<String>], shared: Set<Int>, taps: Bool) async {
        (self.block, self.g, self.path, self.quantized, sharedBlocks, self.taps) = (block, g, path, quantized, shared, taps)
        // The padding's rows under a factor of 1, in both buffers of each stream of factors.
        for (name, f16) in [("f", false), ("x_factor", true)] where guarded && (!f16 || !shared.isEmpty) {
            for next in [false, true] {
                let p = await block.rows(name, next: next)
                for r in 0..<g.keys {
                    if f16 { p.assumingMemoryBound(to: Float16.self)[r] = 1 } else { p.assumingMemoryBound(to: Float.self)[r] = 1 }
                }
            }
        }
    }

    /// **The DiT of the file at `path` under B** (`budget`), in the work's arithmetic (`Arithmetic.opened`): the Standard's
    /// bf16; the Compact's guarded forms under `exact`; under `ane` the shared forms (`sharedBlocks`) with the ANE's
    /// share. `say`: P's line, and the ANE's share.
    package static func open(_ path: String, geometry: Geometry, taps: Bool = false, loras: LoRAStack = .none, budget: Int,
                             say: (String) -> Void = { _ in }) async throws -> AnimaEvaluation {
        guard let device = MTLCreateSystemDefaultDevice() else { throw Failure("no Metal device") }
        let file = try Self.file(path)
        let quantized = blocks.indices.map { Self.quantized($0, in: file) }, ranks = ranks(loras)
        let codes = Self.path(of: file) == .f16, offered = try await Arithmetic.offered(codes: codes, device)
        let arithmetic = try Arithmetic.opened(codes: codes, precision: CoreAI.inForce, offered: offered) {
            declaration(geometry, $0, quantized: quantized, taps: taps, ranks: ranks)
        }
        let declaration = declaration(geometry, arithmetic, quantized: quantized, taps: taps, ranks: ranks)
        let block = try await Block.open(device, declaration, files: [file], budget: budget, share: arithmetic.share)
        say(block.stage.plan.line)
        if block.units > 0 { say("anima dit: the ANE's share \(block.units) of \(block.total) units of \(block.unit) rows") }
        let dit = await AnimaEvaluation(block, geometry, path: Self.path(arithmetic), quantized: quantized, shared: sharedBlocks(arithmetic, quantized), taps: taps)
        for i in blocks.indices {
            try loras.write(loraProducts(block: i), on: .anima, ranks: ranks, element: factorElement(Self.path(arithmetic)), prefix: "\(Self.block(i)).") {
                block.stage.pointer($0)
            }
        }
        return dit
    }

    @discardableResult
    package func close() -> SiliCore.Stage.Settled { block.close() }

    // MARK: - Writing and reading

    /// `values` at `p`, in `type` (f32, or rounded once to bf16 or f16).
    private func write(_ values: [Float], to p: UnsafeMutableRawPointer, as type: NDArray.ScalarType? = nil) {
        switch type ?? element {
        case .float32: values.withUnsafeBytes { _ = memcpy(p, $0.baseAddress!, $0.count) }
        case .bfloat16: let q = p.assumingMemoryBound(to: UInt16.self); for i in values.indices { q[i] = UInt16(BFloat.round(values[i]).bitPattern >> 16) }
        default: let q = p.assumingMemoryBound(to: Float16.self); for i in values.indices { q[i] = Float16(values[i]) }
        }
    }

    /// `count` values of tensor `name`, widened to f32, every call done.
    package func read(_ name: String, count: Int, as type: NDArray.ScalarType? = nil) async -> [Float] {
        let base = await block.pointer(name)
        switch type ?? element {
        case .float32: return Array(UnsafeBufferPointer(start: base.assumingMemoryBound(to: Float.self), count: count))
        case .bfloat16: let p = base.assumingMemoryBound(to: UInt16.self); return (0..<count).map { Float(bitPattern: UInt32(p[$0]) << 16) }
        default: let p = base.assumingMemoryBound(to: Float16.self); return (0..<count).map { Float(p[$0]) }
        }
    }

    /// The stream's true value on the image's rows `[tokens, d]` (`Block.values`: x, or x̂ · X under the guard).
    func streamValues() async -> [Float] {
        await wide()
        return await block.values("x", rows: 0..<g.tokens)
    }

    /// The stream's rows `[tokens, d]` set to `values` (a check's input): x, or x̂ and X under the guard, x̂'s rows in
    /// [2¹³, 2¹⁴] as the guard holds them.
    func setStream(_ values: [Float]) async {
        half = false
        let x = await block.rows("x")
        guard guarded else { write(values, to: x); return }
        var unit = values, factors = [Float](repeating: 1, count: g.tokens)
        for r in 0..<g.tokens {
            let top = (0..<Self.d).reduce(Float(0x1p-14)) { max($0, abs(values[r * Self.d + $1])) }
            let f = Float(sign: .plus, exponent: RowExponent.ranged16(largest: Double(top)), significand: 1)
            factors[r] = f
            for c in 0..<Self.d { unit[r * Self.d + c] = values[r * Self.d + c] / f }
        }
        write(unit, to: x)
        write(factors, to: await block.rows("f"), as: .float32)
    }

    // MARK: - The loop

    /// The edges' factors under the guard: the f32 stream's.
    private var factors: [String: Block.Bind] { guarded ? ["x_factor": .rows("f")] : [:] }

    /// **Once a prompt**: the context `[rows, 1024]` (the conditioner's) at the palier, its padding masked, and each
    /// block's cross keys and values; the self-attention's mask and the RoPE tables of the image.
    package func prompt(_ context: [Float], rows: Int) async throws {
        precondition(context.count == rows * AnimaFunction.textDim && rows <= g.context, "a context of \(rows) rows past \(g.context)")
        write(context + [Float](repeating: 0, count: (g.context - rows) * AnimaFunction.textDim), to: await block.pointer("context"))
        write((0..<g.context).map { $0 < rows ? 0 : -1e4 }, to: await block.pointer("ctx_mask"))
        write((0..<g.keys).map { $0 < g.tokens ? 0 : -1e4 }, to: await block.pointer("mask"))
        let rope = AnimaCosmosDiT(config: .published, latentHeight: g.height, latentWidth: g.width).rope()
        write(rope.cos, to: await block.rows("cos"), as: .float32)
        write(rope.sin, to: await block.rows("sin"), as: .float32)
        for (i, b) in Self.blocks.enumerated() { try await block.call(Self.contextEdge(quantized[i]), group: b) }
    }

    /// The time's function: `temb`, `embedded` and the final layer's rows, for the model's time `t` (`AnimaRender.modelTime`).
    func time(_ t: Float) async throws {
        write(AnimaCosmosDiT.timestepFrequencies(t, config: .published), to: await block.pointer("t_freq"), as: .float32)
        try await block.call("time")
    }

    /// **One evaluation**: the latent `[16, h, w]` at the model's time `t` to the velocity, diffusers' `model_out`.
    /// `taps`: the blocks whose output to keep (`tapped`), with the embedder's and block 0's branches when the DiT has taps.
    package func evaluate(_ latent: [Float], time t: Float, taps: Set<Int> = []) async throws -> [Float] {
        try await time(t)
        write(latent, to: await block.pointer("latent"), as: .float32)
        half = false
        try await block.call("embed", factors)
        if !taps.isEmpty { tapped["patch_embed_out"] = await streamValues() }
        for i in Self.blocks.indices {
            try await run(i)
            if taps.contains(i) { tapped["block\(i)_out"] = await streamValues() }
        }
        await wide()
        try await block.call("final", factors.merging(["shift": .row("final", 0), "scale": .row("final", 1)]) { a, _ in a })
        return await read("velocity", count: latent.count, as: .float32)
    }

    /// **One block** over the image's rows: its weights (kept, or the ring's next use), its modulation, then — the
    /// stream's factors and the gates' in the form's writing — its form's steps; the stream from one buffer to the other.
    func run(_ i: Int) async throws {
        let shared = sharedBlocks.contains(i)
        try await block.run(Self.block(i), "image", before: {
            try await self.block.call("modulation")
            if shared {
                await self.gains()
                try await self.halve()
            } else {
                await self.wide()
            }
        }, tapped: taps && i == 0 && shared)
        if taps && i == 0 {
            for (name, tap) in [("attn1", "block0_attn1_out"), ("attn2", "block0_attn2_out"), ("ff", "block0_ff_out")] {
                tapped[tap] = await block.values(name, rows: 0..<g.tokens)
            }
        }
    }

    /// **Block 0 alone** from the oracle's own input: the stream `x` `[tokens, d]`, `temb` `[3d]`, `embedded` `[d]` (the
    /// prompt given before); its output and branches in `tapped`. The stage's cycle then goes on at block 1.
    package func blockZero(x: [Float], temb: [Float], embedded: [Float]) async throws {
        write(temb, to: await block.pointer("temb"), as: .float32)
        write(embedded, to: await block.pointer("embedded"), as: .float32)
        await setStream(x)
        try await run(0)
        tapped["block0_out"] = await streamValues()
    }

    // MARK: - The shared forms' borders

    /// The stream's factors as the guarded forms read them (f32 `f`), from the shared forms' (f16): powers of two, exact.
    private func wide() async {
        guard half else { return }
        let from = await block.rows("x_factor").assumingMemoryBound(to: Float16.self), to = await block.rows("f").assumingMemoryBound(to: Float.self)
        for r in 0..<g.keys { to[r] = Float(from[r]) }
        half = false
    }

    /// The stream's factors as the shared forms read them (f16), from the guarded forms' (f32). A factor under f16's
    /// smallest normal — the embedded stream straight into a shared block 0, the Standard's beside the ANE: 2⁻¹⁵ at 512²
    /// — is brought to 2⁻¹⁴, the power of two it lacks moved into its row of x̂ (exact while x̂ stays among f16's normals:
    /// x̂ lies within [2¹³, 2¹⁴]); a factor past 2¹⁵ (a stream past 2²⁹) is refused. Never a factor rounded.
    private func halve() async throws {
        guard !half else { return }
        let from = await block.rows("f").assumingMemoryBound(to: Float.self), to = await block.rows("x_factor").assumingMemoryBound(to: Float16.self)
        let x = await block.rows("x").assumingMemoryBound(to: Float16.self), d = Self.d
        for r in 0..<g.keys {
            guard from[r] <= 0x1p15 else { throw Failure("anima dit: a stream factor of \(from[r]) past f16's normals") }
            if from[r] < 0x1p-14 {
                let shift = Float16(from[r] / 0x1p-14)
                for c in 0..<d { x[r * d + c] *= shift }
            }
            to[r] = Float16(max(from[r], 0x1p-14))
        }
        half = true
    }

    /// The modulation's rows (`self`, `cross`, `mlp`: shift, 1 + scale, gate, f16) as the shared forms read them: each gate
    /// over G, the least power of two at or past its row's largest magnitude (`AnimaShared.gains`).
    private func gains() async {
        let (d, w) = (Self.d, AnimaFunction.gainRow)
        func at(_ name: String) async -> UnsafeMutablePointer<Float16> { await block.pointer(name).assumingMemoryBound(to: Float16.self) }
        for (k, sublayer) in ["self", "cross", "mlp"].enumerated() {
            let rows = await at(sublayer)
            let top = (0..<d).reduce(Float(0)) { max($0, abs(Float(rows[2 * w + $1]))) }
            let factor = Gate.factor(largest: top)
            let gateFactor = await at("gate\(k + 1)_factor")
            gateFactor[0] = Float16(factor)
            let (shift, scale, gate) = (await at("shift\(k + 1)"), await at("scale\(k + 1)"), await at("gate\(k + 1)"))
            for c in 0..<d { shift[c] = rows[c]; scale[c] = rows[w + c]; gate[c] = Float16(Float(rows[2 * w + c]) / factor) }
        }
    }
}
