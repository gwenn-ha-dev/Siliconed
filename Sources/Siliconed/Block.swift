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
    package internal(set) var timings: [String: Double] = [:]
    private func timed<T>(_ phase: String, _ body: () throws -> T) rethrows -> T {
        let started = DispatchTime.now().uptimeNanoseconds
        let result = try body()
        timings[phase, default: 0] += Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
        return result
    }
    package func resetTimings() { timings.removeAll() }

    private let shapes: Shapes
    private let eps: Float
    /// The `context_refiner`s have **no** modulation: all they keep is
    /// `x = x + norm2(sublayer(norm1(x)))`, with no scale or gate.
    package var modulation = true
    private let gemm: GEMM
    private var widener: WidenGPU? { gemm.widener }
    /// The SwiGLU on the GPU, shared by the DiT's three sets of blocks — Krea 2's kernel,
    /// judged against the double. It was 3.1–3.4 % of an
    /// evaluation on the CPU, the first elementwise item of the budget; the norms stay on the CPU.
    private let elementwise: ElementwiseGPU
    /// **One attention per sequence length, built on demand.** The `MPSGraph` graph freezes
    /// its shapes at construction; the spectral schedule needs two — the full sequence and
    /// that of the low evaluation. The arena slices do not move: they are
    /// sized for the longest, and a shorter sequence only occupies their beginning.
    private var attentions: [Int: Attention] = [:]
    /// The matrix flash kernel, when `SILICONED_FLASH=1` asks for it — **one per length**, like
    /// the other. It is not the default: in isolation it is on a par with `MPSGraph`, and what
    /// decides is what it returns *inside the engine*, where it removes 6.1 GB of traffic per call.
    private var flashes: [Int: FlashMatrix] = [:]
    static let useFlash = EngineSettings.effective.flash
    /// The SDPA cut by heads between the GPU and the AMX (the only path), one per
    /// length like the others. Not taken under `SILICONED_FLASH` nor in half precision, and below
    /// `HeadSplitAttention.minimumSequence`, where the text blocks run (32 tokens).
    private var headSplits: [Int: HeadSplitAttention] = [:]
    private let arena: Arena
    private let artifact: Artifact
    private var prefix: String = ""
    private let transposed: Bool

    // Arena slices, reserved once.
    private let xSlot, normed, qkv, qHeads, kHeads, vHeads, merged: UnsafeMutablePointer<Float>
    private let hidden1, hidden3, scratch, weightReserve, modulationSlot: UnsafeMutablePointer<Float>
    /// **The LoRA stack, and its three slices.** They are sized for the cumulative rank of
    /// THIS request — the engine builds the DiT after receiving the stack, so there is no
    /// maximum to fix in advance. `nil` when the request carries none: no slice reserved,
    /// no GEMM encoded, and the census mentions nothing.
    package let lora: LoRA?
    /// The stack's widened weights, kept from one evaluation to the next and shared by the three
    /// sets of blocks (`CacheLoRA`, owned by the DiT).
    private let cacheLoRA: CacheLoRA?
    private let loraMid: UnsafeMutablePointer<Float>?
    private var loraMidBuf: MTLBuffer?
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
        let s = shapes.sequence, d = shapes.dim, h = shapes.hidden
        func slot(_ name: String, _ count: Int) throws -> UnsafeMutablePointer<Float> {
            try arena.reserve(name, bytes: count * 4).assumingMemoryBound(to: Float.self)
        }
        xSlot = try slot("x", s * d)
        normed = try slot("normed", s * d)
        qkv = try slot("qkv", s * d)
        qHeads = try slot("q", s * d)
        kHeads = try slot("k", s * d)
        vHeads = try slot("v", s * d)
        merged = try slot("merged", s * d)
        hidden1 = try slot("h1", s * h)
        hidden3 = try slot("h3", s * h)
        // No longer `s × h` since the SwiGLU lives on the GPU: only the norm weights (`d`) and the
        // modulation's LoRA (`4d`).
        scratch = try slot("scratch", 4 * d)
        weightReserve = try slot("reserve", d * h)          // the largest weight of a block
        modulationSlot = try slot("modulation", 4 * d)
        // The LoRA slices, sized for the largest shapes the stack touches —
        // `w2` for the input (10240), `adaLN_modulation.0` for the output (15360). Three slices
        // for ~4 MB at rank 32, against 59 MB for the main weight's reserve alone.
        if let lora, lora.maxRank > 0 {
            loraMid = try slot("loraMid", s * lora.maxRank)
        } else {
            loraMid = nil
        }
        // There used to be four more slices here — `halfQ`, `halfK`, `halfV` and `sdpa` —, the
        // intermediate buffers of the CPU rearrangement around the SDPA. The transposition now living
        // in the graph, `q`, `k`, `v` go out as they are and the output comes back into
        // `merged`: four times `s·d` floats given back, i.e. 254 MB at 1024².
    }

    /// The `count` is checked against the table rather than trusted: a wrong shape would produce a
    /// plausible out-of-bounds read.
    private func widen(_ name: String, count: Int, into destination: UnsafeMutablePointer<Float>) throws {
        let got = try artifact.materialize(prefix + name, into: destination, capacity: count)
        guard got == count else {
            throw Artifact.Failure.badHeader("\(prefix + name): \(got) values, \(count) expected")
        }
    }

    /// **`SILICONED_WIDEN_FUSED=1`** — widen in the GEMM's command buffer rather than
    /// in its own.
    ///
    /// The two paths do the same work on the same hardware; they do not *measure* the same.
    /// Separate, the widening keeps its phase on the clock and the GEMM census stays comparable
    /// with all previous sessions — at the price of one more submission per weight, ~240 per
    /// evaluation. Fused, those submissions are saved but the returned time covers both, and
    /// "gemm GPU" stops meaning what it used to mean.
    ///
    /// The default is **separate**, because the first thing we want to know is what
    /// GPU widening costs *on its own* against the CPU's 0.94 s — and a total that does not
    /// partition is useless. Fused or separate, GPU widening is off by default on this
    /// machine (+20 % in the engine, `WidenGPU`).
    package static let fusedWiden = EngineSettings.effective.widenFused

    /// Puts the weight `name` into fp32 in the reserve, by the shortest path available.
    ///
    /// Returns a prologue to chain onto the GEMM if the widening is to blend into its command
    /// buffer, and `nil` if it is already done — separate GPU or CPU. The three paths produce the
    /// **same bits** (identical over 39.3 M values): the choice is about speed, never about
    /// correctness.
    private func prepareWeight(_ name: String, count: Int,
                               buffer: MTLBuffer) throws -> ((MTLCommandBuffer) -> Void)? {
        if let widener, let source = widener.source(prefix + name, in: artifact) {
            guard artifact.tensors[prefix + name]?.count == count else {
                throw Artifact.Failure.badHeader("\(prefix + name): unexpected shape")
            }
            if Block.fusedWiden {
                return { commands in
                    widener.encode(into: commands, source: source, destination: buffer, count: count)
                }
            }
            try timed("widening") {
                _ = try widener.run(source: source, destination: buffer, count: count)
            }
            return nil
        }
        // The fallback: an fp32 tensor in the map (`t_embedder`, `cap_embedder`), a count that
        // is not a multiple of four, or no GPU at all. This is the previous path, intact.
        widener?.countFallback(count)
        try timed("widening") { try widen(name, count: count, into: weightReserve) }
        return nil
    }


    // ── the LoRA stack ───────────────────────────────────────────────────────────────────────

    /// The Metal wrappers of the three slices, memoized **at their maximum size**.
    ///
    /// `GEMM.wrap` memoizes by address and wraps again when a longer length is asked: wrapping a
    /// small shape first would make the next, larger one wrap anew — `w2` (k = 10240) comes after
    /// `to_q` (k = 3840). So we wrap once, for the worst case — the one the slices are sized for.
    private func loraMidBuffer() throws -> MTLBuffer? {
        guard let lora, lora.maxRank > 0, let mid = loraMid else { return nil }
        if let m = loraMidBuf { return m }
        let m = try gemm.wrap(UnsafeMutableRawPointer(mid),
                              bytes: shapes.sequence * lora.maxRank * 4, name: "loraMid")
        loraMidBuf = m
        return m
    }

    /// **Adds this weight's LoRA stack to `c`, which already holds `x·W`.**
    ///
    /// Does nothing — no materialization, no submission — when no LoRA touches this
    /// module. That is the case for the four refiners with the reference LoRA, which only targets the
    /// thirty `layers.*`.
    private func applyLoRA(_ weightName: String, x: MTLBuffer, c: MTLBuffer,
                               m: Int, k: Int, n: Int) throws {
        guard let cache = cacheLoRA, let mid = try loraMidBuffer() else { return }
        let target = prefix + String(weightName.dropLast(".weight".count))
        guard let stack = try timed("lora weights", { try cache.module(target, k: k, n: n) }) else { return }
        let down = try gemm.wrap(UnsafeMutableRawPointer(stack.down), bytes: k * stack.rank * 4, name: "loraDown")
        let up = try gemm.wrap(UnsafeMutableRawPointer(stack.up), bytes: stack.rank * n * 4, name: "loraUp")
        timed("lora wall") {
            timings["lora GPU", default: 0] += gemm.lora(
                x: x, down: down, mid: mid, up: up, c: c,
                m: m, k: k, r: stack.rank, n: n)
        }
    }

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

    /// The flash kernel of this length, or `nil` if it does not apply — too short for its
    /// tiling, `Dh` other than 128, or simply not requested. *A path that silently decides
    /// not to apply is a path of which we will not know later whether it was used*: the fallback
    /// is reported once, through the warnings.
    private func flash(_ sequence: Int) -> FlashMatrix? {
        guard Block.useFlash else { return nil }
        if let existing = flashes[sequence] { return existing }
        do {
            let built = try FlashMatrix(device: gemm.device, queue: gemm.queue,
                                        heads: shapes.heads, sequence: sequence,
                                        headDim: shapes.headDim, rowsPerSimdgroup: 8,
                                        dimsPerSimdgroup: 64, keysPerTile: 16, blocksPerGroup: 8)
            flashes[sequence] = built
            return built
        } catch {
            Warnings.emit("⚠ SILICONED_FLASH: falling back to MPSGraph at S=\(sequence) — \(error)")
            return nil
        }
    }

    /// The head split of this length, or `nil` when it does not apply.
    private func headSplit(_ sequence: Int) -> HeadSplitAttention? {
        guard !Attention.useHalfPrecision, sequence >= HeadSplitAttention.minimumSequence else { return nil }
        if let existing = headSplits[sequence] { return existing }
        let built = HeadSplitAttention(device: gemm.device, queue: gemm.queue, heads: shapes.heads,
                                       sequence: sequence, headDim: shapes.headDim)
        headSplits[sequence] = built
        return built
    }

    /// The attention of this length, built on first call.
    private func attention(_ sequence: Int) -> Attention {
        if let existing = attentions[sequence] { return existing }
        let built = Attention(device: gemm.device, queue: gemm.queue,
                              heads: shapes.heads, sequence: sequence, headDim: shapes.headDim)
        attentions[sequence] = built
        return built
    }

    /// `x` as `[S, D]`, modified in place. `freqs` as `[S, Dh/2, 2]`. `adaln` as `[256]`.
    ///
    /// - Parameter sequence: the length **of this call**, at most the sizing one. The
    ///   spectral schedule evaluates the first steps on fewer tokens; the slices are
    ///   the same, only their useful part shrinks. A larger length would be a write outside the
    ///   slice, so it is refused rather than truncated.
    package func forward(x: UnsafeMutablePointer<Float>, freqs: UnsafePointer<Float>,
                        adaln: UnsafePointer<Float>, adalnDim: Int, prefix: String,
                        sequence: Int? = nil) throws {
        self.prefix = prefix
        let s = sequence ?? shapes.sequence
        let d = shapes.dim, h = shapes.hidden
        guard s <= shapes.sequence else {
            throw Artifact.Failure.misuse(
                "\(prefix): \(s) tokens requested, \(shapes.sequence) reserved")
        }
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

        let w = try gemm.wrap(UnsafeMutableRawPointer(weightReserve), bytes: d * h * 4, name: "w")
        // `q`, `k`, `v` are written directly into their slice: there used to be a copy of
        // `qkv` per projection here, 1 % of an evaluation. Same bits — the same GEMM, another address.
        let qb = try gemm.wrap(UnsafeMutableRawPointer(qHeads), bytes: shapes.sequence * d * 4, name: "q")
        let kb = try gemm.wrap(UnsafeMutableRawPointer(kHeads), bytes: shapes.sequence * d * 4, name: "k")
        let vb = try gemm.wrap(UnsafeMutableRawPointer(vHeads), bytes: shapes.sequence * d * 4, name: "v")
        for (name, destination, buffer, label) in [("attention.to_q.weight", qHeads, qb, "layer0_q"),
                                                   ("attention.to_k.weight", kHeads, kb, "layer0_k"),
                                                   ("attention.to_v.weight", vHeads, vb, "layer0_v")] {
            let prologue = try prepareWeight(name, count: d * d, buffer: w)
            timed("gemm wall") { timings["gemm GPU", default: 0] += gemm.linear(a: a, b: w, c: buffer, m: s, k: d, n: d, weightIsTransposed: transposed, before: prologue) }
            try applyLoRA(name, x: a, c: buffer, m: s, k: d, n: d)
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
        timed("sdpa wall") {
            if let flash = flash(s) {
                timings["sdpa GPU", default: 0] += flash.run(q: qb, k: kb, v: vb, into: ob)
            } else if let split = headSplit(s) {
                let (gpu, cpu) = split.run(q: qb, k: kb, v: vb, into: ob, q: qHeads, k: kHeads, v: vHeads, out: merged)
                timings["sdpa GPU", default: 0] += gpu
                timings["sdpa CPU", default: 0] += cpu
            } else {
                timings["sdpa GPU", default: 0] += attention(s).run(q: qb, k: kb, v: vb, into: ob)
            }
        }
        record("layer0_sdpa_out", merged, s * d)

        let outPrologue = try prepareWeight("attention.to_out.0.weight", count: d * d, buffer: w)
        let mb = try gemm.wrap(UnsafeMutableRawPointer(merged), bytes: shapes.sequence * d * 4, name: "merged")
        timed("gemm wall") { timings["gemm GPU", default: 0] += gemm.linear(a: mb, b: w, c: o, m: s, k: d, n: d, weightIsTransposed: transposed, before: outPrologue) }
        try applyLoRA("attention.to_out.0.weight", x: mb, c: o, m: s, k: d, n: d)
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
        let w1Prologue = try prepareWeight("feed_forward.w1.weight", count: h * d, buffer: w)
        timed("gemm wall") { timings["gemm GPU", default: 0] += gemm.linear(a: a, b: w, c: h1, m: s, k: d, n: h, weightIsTransposed: transposed, before: w1Prologue) }
        try applyLoRA("feed_forward.w1.weight", x: a, c: h1, m: s, k: d, n: h)
        record("layer0_w1_out", hidden1, s * h)
        let w3Prologue = try prepareWeight("feed_forward.w3.weight", count: h * d, buffer: w)
        timed("gemm wall") { timings["gemm GPU", default: 0] += gemm.linear(a: a, b: w, c: h3, m: s, k: d, n: h, weightIsTransposed: transposed, before: w3Prologue) }
        try applyLoRA("feed_forward.w3.weight", x: a, c: h3, m: s, k: d, n: h)
        record("layer0_w3_out", hidden3, s * h)

        timed("swiglu") { timings["elementwise GPU", default: 0] += elementwise.swiglu((h1, 0), (h3, 0), count: s * h) }
        record("layer0_swiglu_out", hidden1, s * h)

        let w2Prologue = try prepareWeight("feed_forward.w2.weight", count: d * h, buffer: w)
        timed("gemm wall") { timings["gemm GPU", default: 0] += gemm.linear(a: h1, b: w, c: o, m: s, k: h, n: d, weightIsTransposed: transposed, before: w2Prologue) }
        try applyLoRA("feed_forward.w2.weight", x: h1, c: o, m: s, k: h, n: d)
        record("layer0_feed_forward_out", qkv, s * d)

        try widen("ffn_norm2.weight", count: d, into: scratch)
        timed("norm") { Ops.rmsNorm(qkv, weight: scratch, into: normed, rows: s, columns: d, eps: eps) }
        record("layer0_ffn_norm2_out", normed, s * d)
        timed("residual") { Ops.gatedResidual(x, plus: normed, gate: gateMLP, rows: s, columns: d) }
        record("layers_0_out", x, s * d)
    }
}
