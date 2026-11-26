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
/// **The residual stream** `x` is two buffers in turn; under the guard of f16 beside its row factors, in f32 at the
/// edges (`f`: the embedder writes it, the final layer reads it), in f16 for the shared forms (`x_factor`): a block
/// reads one buffer and writes the other. The padding's rows are held under a factor of 1: x̂ = 0 normalizes to 0, never to a NaN its keys would spread.
///
/// **The forms**: the Standard on the GPU alone, its blocks' bf16 in guarded forms, each `pre` then `post`; every other
/// block in the shared forms (`AnimaShared`), a form a set of products on codes (the Compact's blocks 0 and 27 none, 1
/// to 3 all but the MLP, the others all; the Standard beside the ANE none): **with the Neural Engine on**, the ANE pre,
/// out, crossout and ffn on units of the image's first rows (`Block.Share`: a control's rows, or found in flight), the
/// GPU the rest. The stream's factors pass from f32 to f16 entering block 0 and back before the final layer, every
/// factor a power of two (exact), on the CPU, as the gates' factors are.
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

    /// **The blocks the shared forms take**: every block but the Standard's on the GPU alone (bf16, the guarded forms) —
    /// the Compact's beside the ANE or on the GPU alone, each product on its codes or its f16 weight (`AnimaShared`), the
    /// Standard's beside the ANE in f16. On the GPU alone the Compact's blocks of codes went 10 % faster in the shared
    /// forms than in the guarded ones, for 112 MiB more of arena (the M5 Pro at 1024², 22/11/2026).
    static func sharedBlocks(_ a: Arithmetic) -> Set<Int> { a == .standard ? [] : Set(blocks.indices) }

    let g: Geometry
    let arithmetic: Arithmetic
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

    // MARK: - Opening

    private init(_ block: Block, _ g: Geometry, _ arithmetic: Arithmetic, quantized: [Set<String>], taps: Bool) async {
        (self.block, self.g, self.arithmetic, self.quantized, self.taps) = (block, g, arithmetic, quantized, taps)
        (path, sharedBlocks) = (Self.path(arithmetic), Self.sharedBlocks(arithmetic))
        let shared = sharedBlocks
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

    /// **The DiT of the file at `path` under B** (`budget`), in the work's arithmetic (`Arithmetic.block`): the Standard's
    /// bf16; the Compact's shared forms (`sharedBlocks`), with the ANE's share when the Neural Engine is on.
    package static func open(_ path: String, geometry: Geometry, taps: Bool = false, loras: LoRAStack = .none, budget: Int) async throws -> AnimaEvaluation {
        let file = try Self.file(path)
        let quantized = blocks.indices.map { Self.quantized($0, in: file) }, ranks = ranks(loras)
        let (block, arithmetic) = try await Arithmetic.block(file: file, codes: codesMarker, budget: budget) {
            declaration(geometry, $0, quantized: quantized, taps: taps, ranks: ranks)
        }
        let dit = await AnimaEvaluation(block, geometry, arithmetic, quantized: quantized, taps: taps)
        for i in blocks.indices {
            try loras.write(loraProducts(block: i), on: .anima, ranks: ranks, element: factorElement(Self.path(arithmetic)), prefix: "\(Self.block(i)).") {
                block.stage.pointer($0)
            }
        }
        return dit
    }

    /// **What `open` would specialize** (`Arithmetic.prepare`): the same declaration, nothing allocated.
    package static func prepare(_ path: String, geometry: Geometry, loras: LoRAStack = .none, budget: Int) throws -> [Functions.Specialization] {
        let file = try Self.file(path)
        let quantized = blocks.indices.map { Self.quantized($0, in: file) }, ranks = ranks(loras)
        return try Arithmetic.prepare(file: file, codes: codesMarker, budget: budget) {
            declaration(geometry, $0, quantized: quantized, taps: false, ranks: ranks)
        }
    }

    /// What the DiT is: its arithmetic and its plan under B.
    package var described: String { "\(arithmetic), " + block.stage.plan.line }

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
            let (row, f) = RowExponent.split(values[(r * Self.d)..<((r + 1) * Self.d)], .ranged16)
            factors[r] = f
            unit.replaceSubrange((r * Self.d)..<((r + 1) * Self.d), with: row)
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

    /// The stream's factors as the edges read them (f32 `f`), from the shared forms' (f16): powers of two, exact.
    private func wide() async {
        guard half else { return }
        let from = await block.rows("x_factor").assumingMemoryBound(to: Float16.self), to = await block.rows("f").assumingMemoryBound(to: Float.self)
        for r in 0..<g.keys { to[r] = Float(from[r]) }
        half = false
    }

    /// The stream's factors as the shared forms read them (f16), from the edges' (f32). A factor under f16's
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
