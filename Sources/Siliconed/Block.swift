import Accelerate
import Metal
import Foundation

/// A Z-Image Turbo transformer block, in fp32.
///
///     scale_msa, gate_msa, scale_mlp, gate_mlp = chunk(adaLN_modulation(adaln), 4)
///     gate = tanh(gate)   ·   scale = 1 + scale          ← no shift (pitfall 3.3)
///     x = x + gate_msa · attention_norm2( attention( attention_norm1(x) · scale_msa ) )
///     x = x + gate_mlp · ffn_norm2( feed_forward( ffn_norm1(x) · scale_mlp ) )
///
/// The sandwich-norm is what makes this block impossible in fp16: the sublayer output rises to
/// 8.7·10⁵ **before** `norm2` divides it back down (pitfall 3.11).
///
/// Where it runs: the seven GEMMs on the GPU (with the AMX when co-execution is on, `Conductor`), the
/// SDPA on the GPU with a share of the heads on the CPU (`HeadSplitAttention`), the SwiGLU on
/// the GPU (`ElementwiseGPU`), the norms, RoPE and residuals on the CPU. The elementwise ops weigh
/// little in FLOPs, carry all the porting risk, and on the CPU they can be debugged.
package final class Block {
    package struct Shapes {
        package let sequence: Int
        package let dim: Int
        package let heads: Int
        package let hidden: Int
        package var headDim: Int { dim / heads }
        package init(sequence: Int, dim: Int, heads: Int, hidden: Int) {
            self.sequence = sequence; self.dim = dim; self.heads = heads; self.hidden = hidden
        }
    }

    /// What the engine exposes to the oracle: every boundary that the checks know how to judge.
    package private(set) var boundaries: [String: [Float]] = [:]
    package var recordBoundaries = false

    /// The time per phase, accumulated over all the blocks. We measure instead of deducing: the
    /// arithmetic count gave 25 s out of 48 measured, and the gap could not be guessed by anyone.
    /// The weight products' phases (`WeightProducts`) added to the block's own.
    package var timings: [String: Double] { ownTimings.merging(products.timings, uniquingKeysWith: +) }
    private var ownTimings: [String: Double] = [:]
    private func timed<T>(_ phase: String, _ body: () throws -> T) rethrows -> T {
        let started = DispatchTime.now().uptimeNanoseconds
        let result = try body()
        ownTimings[phase, default: 0] += Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
        return result
    }
    package func resetTimings() { ownTimings.removeAll(); products.resetTimings() }

    private let shapes: Shapes
    private let eps: Float
    /// The `context_refiner`s have **no** modulation: all they keep is
    /// `x = x + norm2(sublayer(norm1(x)))`, with no scale or gate.
    package var modulation = true
    private let gemm: GEMM
    /// The seven products by a weight, and their LoRA stack.
    private let products: WeightProducts
    /// The SwiGLU on the GPU, shared by the DiT's three sets of blocks — Krea 2's kernel,
    /// judged against the double. It was 3.1–3.4 % of an
    /// evaluation on the CPU, the first elementwise item of the budget; the norms stay on the CPU.
    private let elementwise: ElementwiseGPU
    /// **The block's attention, built on first use** at its one length (`shapes.sequence`): the
    /// `MPSGraph` graph freezes its shapes at construction.
    private var builtAttention: Attention?
    /// The matrix flash kernel, when `SILICONED_FLASH=1` asks for it. It is not the default: in
    /// isolation it is on a par with `MPSGraph`, and what decides is what it returns *inside the
    /// engine*, where it removes 6.1 GB of traffic per call.
    private var builtFlash: FlashMatrix?
    /// The SDPA cut by heads between the GPU and the AMX. Not taken under `flash`, nor where
    /// `HeadSplitAttention.applies` says no (`head_split` off, half precision, below
    /// `minimumSequence`, where the text blocks run at 32 tokens): MPSGraph there.
    private var builtHeadSplit: HeadSplitAttention?
    private let arena: Arena
    private let artifact: Artifact
    private var prefix: String = ""
    private let transposed: Bool

    // Arena slices, reserved once.
    private let normed, qkv, qHeads, kHeads, vHeads, merged: UnsafeMutablePointer<Float>
    private let hidden1, hidden3, scratch, weightReserve, modulationSlot: UnsafeMutablePointer<Float>
    /// **The LoRA stack, and its three slices.** They are sized for the cumulative rank of
    /// THIS request — the engine builds the DiT after receiving the stack, so there is no
    /// maximum to fix in advance. `nil` when the request carries none: no slice reserved,
    /// no GEMM encoded, and the census mentions nothing.
    package let lora: LoRA?
    /// The stack's widened weights, kept from one evaluation to the next and shared by the three
    /// sets of blocks (`CacheLoRA`, owned by the DiT).
    private let cacheLoRA: CacheLoRA?
    //
    // **A single reserve, and that is a measured decision.** `widen` reads the fp16 weight from the map
    // and writes its fp32 version — 157 MB for `w1` — serially in front of the GEMM that will read it: 0.90 s
    // per evaluation at 1024², 6.3 s on a render. So we tried two reserves and a background thread
    // that widens the *next* weight while the GPU computes the current GEMM. Measured at 1024²:
    //
    //     widening       0.90 → 0.30 s       ✔ the thread does its job
    //     gemm GPU      16.23 → 17.14 s      ✘ and the GEMM pays more than that
    //     swiglu+norm    1.04 → 1.32 s       ✘ the elementwise ops too
    //     evaluation    24.05 → 24.80 s      ✘ net loss, plus 472 MB (three blocks × 157)
    //
    // The background thread does not take free time, it takes **bandwidth**. Overlapping
    // memory traffic does not create capacity: it moves the contention into the GEMM, which is
    // more sensitive to it than the widening is. It is also the answer to the gap that the `gemmloop`
    // loop exhibited (3.45 TFLOP/s against 2.71 in the engine): the loop does *only* GEMMs.
    //
    // **The engine is limited by DRAM, not by compute.** What wins here is what
    // reduces the total traffic — not what reorders it. Fusing QKV and gate/up at the forge looked
    // like that, and was refuted: a larger `N` tiles worse in MPS than the reread input saves
    // (the forge v1 measure, then Krea 2's shapes: gate/up fused −31 to −33 %).

    /// A single instance serves the 34 blocks: the arena slices are reserved once, and
    /// `prefix` designates at each call which set of weights to read from the map.
    package init(artifact: Artifact, shapes: Shapes, eps: Float,
                gemm: GEMM, elementwise: ElementwiseGPU, arena: Arena, lora: LoRA? = nil,
                cacheLoRA: CacheLoRA? = nil) throws {
        self.elementwise = elementwise
        self.lora = lora
        self.cacheLoRA = cacheLoRA
        self.artifact = artifact
        self.transposed = artifact.linearWeightsTransposed
        self.shapes = shapes
        self.eps = eps
        self.gemm = gemm
        self.arena = arena
        // The slices, from `Block.slices` — the function `DiT.blockArenaBytes` sizes the arena with.
        var pointers: [String: UnsafeMutablePointer<Float>] = [:]
        for (name, count) in Block.slices(shapes, loraRank: lora?.maxRank ?? 0) {
            pointers[name] = try arena.reserve(name, bytes: count * 4).assumingMemoryBound(to: Float.self)
        }
        normed = pointers["normed"]!; qkv = pointers["qkv"]!
        qHeads = pointers["q"]!; kHeads = pointers["k"]!; vHeads = pointers["v"]!; merged = pointers["merged"]!
        // `h1` lies over the attention's scratch, dead by the feed-forward: `q`, `k`, `v` and `merged` are
        // adjacent in the arena and hold more floats per row than `h1` (`Block.slices`).
        hidden1 = qHeads; hidden3 = pointers["h3"]!
        guard UnsafeMutableRawPointer(hidden1 + shapes.sequence * shapes.hidden)
                <= UnsafeMutableRawPointer(merged + shapes.sequence * shapes.dim) else {
            throw Artifact.Failure.badHeader("a block's h1 does not fit over its attention's scratch")
        }
        scratch = pointers["scratch"]!; weightReserve = pointers["reserve"]!; modulationSlot = pointers["modulation"]!
        products = WeightProducts(gemm: gemm, artifact: artifact, reserve: weightReserve,
                                  capacity: shapes.dim * shapes.hidden, transposed: artifact.linearWeightsTransposed,
                                  lora: cacheLoRA, mid: pointers["loraMid"],
                                  midFloats: shapes.sequence * (lora?.maxRank ?? 0), roundsActivations: true)
        // There used to be four more slices here — `halfQ`, `halfK`, `halfV` and `sdpa` —, the
        // intermediate buffers of the CPU rearrangement around the SDPA. The transposition now living
        // in the graph, `q`, `k`, `v` go out as they are and the output comes back into
        // `merged`: four times `s·d` floats given back, i.e. 254 MB at 1024².
    }

    /// **A block's slices, in floats, in the order `init` reserves them** — one source with
    /// `DiT.blockArenaBytes`: six `s×d` (`normed`, `qkv`, `q`, `k`, `v`, `merged`), one `s×h` (`h3`; `h1`
    /// lies over `q`…`merged`, dead by the feed-forward), the `d×h` weight reserve (the largest weight of a block: `w1`, `w2`, `w3`), `4d` of scratch
    /// (the norm weights, `d`, and the modulation's LoRA, `4d`), `4d` of modulation and, under a LoRA
    /// stack of cumulative rank `r`, the `mid [s, r]` slice (sized for the widest module it touches).
    /// Each is written whole by an evaluation at the sizing length. There used to be an `x` slice
    /// too, never written — `forward` works on the caller's `x` —: 64 MB at 1024² in the unified
    /// block, as much in the image one.
    package static func slices(_ shapes: Shapes, loraRank r: Int) -> [(name: String, count: Int)] {
        let s = shapes.sequence, d = shapes.dim, h = shapes.hidden
        return [("normed", s * d), ("qkv", s * d), ("q", s * d), ("k", s * d), ("v", s * d), ("merged", s * d),
                ("h3", s * h), ("scratch", 4 * d), ("reserve", d * h), ("modulation", 4 * d)]
            + (r > 0 ? [("loraMid", s * r)] : [])
    }

    /// The `count` is checked against the table rather than trusted: a wrong shape would produce a
    /// plausible out-of-bounds read.
    private func widen(_ name: String, count: Int, into destination: UnsafeMutablePointer<Float>) throws {
        let got = try artifact.materialize(prefix + name, into: destination, capacity: count)
        guard got == count else {
            throw Artifact.Failure.badHeader("\(prefix + name): \(got) values, \(count) expected")
        }
    }

    // ── the LoRA stack ───────────────────────────────────────────────────────────────────────

    /// **The modulation's LoRA — and it is the only place where it does not go through the GPU.**
    ///
    /// `adaLN_modulation.0` is a `gemv`: a single input row, 15,360 outputs. Entrusting it
    /// to MPS would cost two submissions for 4 M FLOPs. It therefore stays on the CPU, like the main
    /// weight — but it **must** be covered: the reference LoRA targets it in its thirty
    /// layers, and forgetting it would return a half-stylized image, which looks like an
    /// artistic choice and is not one.
    private func applyLoRAModulation(adaln: UnsafePointer<Float>, adalnDim: Int, outputs: Int) throws {
        guard let cache = cacheLoRA else { return }
        let target = prefix + "adaLN_modulation.0"
        guard let (down, up, r) = try timed("lora weights", { try cache.module(target, k: adalnDim, n: outputs) })
        else { return }
        timed("lora wall") {
            var middle = [Float](repeating: 0, count: r)
            middle.withUnsafeMutableBufferPointer { m in
                Ops.gemv(weight: down, bias: nil, x: adaln, into: m.baseAddress!,
                         outputs: r, inputs: adalnDim, transposed: true)
                Ops.gemv(weight: up, bias: nil, x: m.baseAddress!, into: scratch,
                         outputs: outputs, inputs: r, transposed: true)
            }
            vDSP_vadd(modulationSlot, 1, scratch, 1, modulationSlot, 1, vDSP_Length(outputs))
        }
    }

    private func record(_ name: String, _ pointer: UnsafePointer<Float>, _ count: Int) {
        guard recordBoundaries else { return }
        boundaries[name] = Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    /// The flash kernel, or `nil` if it does not apply — too short for its
    /// tiling, `Dh` other than 128, or simply not requested. *A path that silently decides
    /// not to apply is a path of which we will not know later whether it was used*: the fallback
    /// is reported once, through the warnings.
    private func flash() -> FlashMatrix? {
        guard HeadSplitAttention.allowsFlash(sequence: shapes.sequence, heads: shapes.heads, headDim: shapes.headDim) else { return nil }
        if let existing = builtFlash { return existing }
        let sequence = shapes.sequence
        do {
            let built = try FlashMatrix(device: gemm.device, queue: gemm.queue,
                                        heads: shapes.heads, sequence: sequence,
                                        headDim: shapes.headDim, rowsPerSimdgroup: 8,
                                        dimsPerSimdgroup: 64, keysPerTile: 16, blocksPerGroup: 8)
            builtFlash = built
            return built
        } catch {
            Warnings.emit("⚠ SILICONED_FLASH: falling back to MPSGraph at S=\(sequence) — \(error)")
            return nil
        }
    }

    /// The head split, or `nil` when it does not apply.
    private func headSplit() throws -> HeadSplitAttention? {
        let sequence = shapes.sequence
        guard HeadSplitAttention.applies(sequence: sequence) else { return nil }
        if let existing = builtHeadSplit { return existing }
        let built = try HeadSplitAttention(device: gemm.device, queue: gemm.queue, heads: shapes.heads,
                                       sequence: sequence, headDim: shapes.headDim)
        builtHeadSplit = built
        return built
    }

    /// The attention, built on first call.
    private func attention() -> Attention {
        if let existing = builtAttention { return existing }
        let built = Attention(device: gemm.device, queue: gemm.queue,
                              heads: shapes.heads, sequence: shapes.sequence, headDim: shapes.headDim)
        builtAttention = built
        return built
    }

    /// `x` as `[S, D]`, modified in place, `S` the sizing length. `freqs` as `[S, Dh/2, 2]`.
    /// `adaln` as `[256]`.
    package func forward(x: UnsafeMutablePointer<Float>, freqs: UnsafePointer<Float>,
                        adaln: UnsafePointer<Float>, adalnDim: Int, prefix: String) throws {
        self.prefix = prefix
        let s = shapes.sequence
        let d = shapes.dim, h = shapes.hidden
        let heads = shapes.heads, headDim = shapes.headDim

        // ── modulation: a bare Linear, with bias, four pieces. No SiLU (pitfall 3.3).
        let scaleMSA = modulationSlot, gateMSA = modulationSlot + d
        let scaleMLP = modulationSlot + 2 * d, gateMLP = modulationSlot + 3 * d
        if modulation {
            try widen("adaLN_modulation.0.weight", count: 4 * d * adalnDim, into: weightReserve)
            var bias = [Float](repeating: 0, count: 4 * d)
            try bias.withUnsafeMutableBufferPointer {
                try widen("adaLN_modulation.0.bias", count: 4 * d, into: $0.baseAddress!)
            }
            timed("modulation") {
                Ops.gemv(weight: weightReserve, bias: bias, x: adaln, into: modulationSlot,
                         outputs: 4 * d, inputs: adalnDim, transposed: transposed)
            }
            try applyLoRAModulation(adaln: adaln, adalnDim: adalnDim, outputs: 4 * d)
            record("layer0_adaln_raw", modulationSlot, 4 * d)
            Ops.tanhInPlace(gateMSA, count: d)
            Ops.tanhInPlace(gateMLP, count: d)
            Ops.addScalar(scaleMSA, 1, count: d)
            Ops.addScalar(scaleMLP, 1, count: d)
        } else {
            // Without modulation, scales and gates equal one: we materialize them rather than
            // duplicate the path, because a second path is a second place to go wrong.
            for i in 0..<(4 * d) { modulationSlot[i] = 1 }
        }

        // ── attention ────────────────────────────────────────────────────────────────────
        try widen("attention_norm1.weight", count: d, into: scratch)
        timed("norm") { Ops.rmsNorm(x, weight: scratch, into: normed, rows: s, columns: d, eps: eps) }
        record("layer0_attn_norm1_out", normed, s * d)
        timed("scale") { Ops.scaleRows(normed, by: scaleMSA, rows: s, columns: d) }
        record("layer0_attn_in", normed, s * d)

        let a = try gemm.wrap(UnsafeMutableRawPointer(normed), bytes: shapes.sequence * d * 4, name: "normed")
        let o = try gemm.wrap(UnsafeMutableRawPointer(qkv), bytes: shapes.sequence * d * 4, name: "qkv")

        // `q`, `k`, `v` are written directly into their slice: there used to be a copy of
        // `qkv` per projection here, 1 % of an evaluation. Same bits — the same GEMM, another address.
        let qb = try gemm.wrap(UnsafeMutableRawPointer(qHeads), bytes: shapes.sequence * d * 4, name: "q")
        let kb = try gemm.wrap(UnsafeMutableRawPointer(kHeads), bytes: shapes.sequence * d * 4, name: "k")
        let vb = try gemm.wrap(UnsafeMutableRawPointer(vHeads), bytes: shapes.sequence * d * 4, name: "v")
        for (name, destination, buffer, label) in [("attention.to_q.weight", qHeads, qb, "layer0_q"),
                                                   ("attention.to_k.weight", kHeads, kb, "layer0_k"),
                                                   ("attention.to_v.weight", vHeads, vb, "layer0_v")] {
            try products.linear(prefix + name, a: a, c: buffer, m: s, k: d, n: d)
            record(label, destination, s * d)
        }

        // QK-Norm over head_dim, AFTER the split into heads and BEFORE the RoPE.
        try widen("attention.norm_q.weight", count: headDim, into: scratch)
        timed("qk-norm") { Ops.rmsNorm(qHeads, weight: scratch, into: qHeads, rows: s * heads, columns: headDim, eps: 1e-5) }
        record("layer0_q_normed", qHeads, s * d)
        try widen("attention.norm_k.weight", count: headDim, into: scratch)
        timed("qk-norm") { Ops.rmsNorm(kHeads, weight: scratch, into: kHeads, rows: s * heads, columns: headDim, eps: 1e-5) }
        record("layer0_k_normed", kHeads, s * d)

        timed("rope") {
            Ops.rope(qHeads, freqs: freqs, sequence: s, heads: heads, headDim: headDim)
            Ops.rope(kHeads, freqs: freqs, sequence: s, heads: heads, headDim: headDim)
        }
        // No more rearrangement here: `q`, `k` and `v` go out as the GEMM wrote them,
        // `[S, H·Dh]` in fp32, and the output comes back in the same layout. The transposition and
        // the precision conversion live in `Attention`'s graph — see its initialization comment
        // for what this CPU copy cost.
        let ob = try gemm.wrap(UnsafeMutableRawPointer(merged), bytes: shapes.sequence * d * 4, name: "sdpaOut")
        let flash = flash()
        let split = flash == nil ? try headSplit() : nil
        timed("sdpa wall") {
            if let flash {
                ownTimings["sdpa GPU", default: 0] += flash.run(q: qb, k: kb, v: vb, into: ob)
            } else if let split {
                let (gpu, cpu) = split.run(q: qb, k: kb, v: vb, into: ob, q: qHeads, k: kHeads, v: vHeads, out: merged)
                ownTimings["sdpa GPU", default: 0] += gpu
                ownTimings["sdpa CPU", default: 0] += cpu
            } else {
                ownTimings["sdpa GPU", default: 0] += attention().run(q: qb, k: kb, v: vb, into: ob)
            }
        }
        record("layer0_sdpa_out", merged, s * d)

        let mb = try gemm.wrap(UnsafeMutableRawPointer(merged), bytes: shapes.sequence * d * 4, name: "merged")
        try products.linear(prefix + "attention.to_out.0.weight", a: mb, c: o, m: s, k: d, n: d)
        record("layer0_attention_out", qkv, s * d)

        try widen("attention_norm2.weight", count: d, into: scratch)
        timed("norm") { Ops.rmsNorm(qkv, weight: scratch, into: normed, rows: s, columns: d, eps: eps) }
        record("layer0_attn_norm2_out", normed, s * d)
        timed("residual") { Ops.gatedResidual(x, plus: normed, gate: gateMSA, rows: s, columns: d) }

        // ── feed-forward ─────────────────────────────────────────────────────────────────
        try widen("ffn_norm1.weight", count: d, into: scratch)
        timed("norm") { Ops.rmsNorm(x, weight: scratch, into: normed, rows: s, columns: d, eps: eps) }
        record("layer0_ffn_norm1_out", normed, s * d)
        timed("scale") { Ops.scaleRows(normed, by: scaleMLP, rows: s, columns: d) }
        record("layer0_ffn_in", normed, s * d)

        let h1 = try gemm.wrap(UnsafeMutableRawPointer(hidden1), bytes: shapes.sequence * h * 4, name: "h1")
        let h3 = try gemm.wrap(UnsafeMutableRawPointer(hidden3), bytes: shapes.sequence * h * 4, name: "h3")
        try products.linear(prefix + "feed_forward.w1.weight", a: a, c: h1, m: s, k: d, n: h)
        record("layer0_w1_out", hidden1, s * h)
        try products.linear(prefix + "feed_forward.w3.weight", a: a, c: h3, m: s, k: d, n: h)
        record("layer0_w3_out", hidden3, s * h)

        timed("swiglu") { ownTimings["elementwise GPU", default: 0] += elementwise.swiglu((h1, 0), (h3, 0), count: s * h) }
        record("layer0_swiglu_out", hidden1, s * h)

        try products.linear(prefix + "feed_forward.w2.weight", a: h1, c: o, m: s, k: h, n: d)
        record("layer0_feed_forward_out", qkv, s * d)

        try widen("ffn_norm2.weight", count: d, into: scratch)
        timed("norm") { Ops.rmsNorm(qkv, weight: scratch, into: normed, rows: s, columns: d, eps: eps) }
        record("layer0_ffn_norm2_out", normed, s * d)
        timed("residual") { Ops.gatedResidual(x, plus: normed, gate: gateMLP, rows: s, columns: d) }
        record("layers_0_out", x, s * d)
    }
}
