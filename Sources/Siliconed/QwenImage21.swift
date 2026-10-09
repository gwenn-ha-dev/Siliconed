import Metal
import Accelerate
import Foundation

/// The Qwen-Image-2.1 DiT — `QwenImage21Transformer2DModel`, 7.1 G parameters —, in fp32.
///
///     Qwen3-VL last layer [T, 4096] (image slots included) ─ text rows only ─ txt_in ─┐
///     reference latents [64, hᵢ, wᵢ] ─ tokens ─ img_in ──────────────────────────────┤ prefix P
///     target latent [64, h, w] ─ tokens ─ img_in ────────────────────────────────────┐│ target S
///                                                                                     ▼▼
///       joint sequence: text, each reference's h·w tokens AT ITS SLOTS, the target last
///                                                                                     │
///     t ─ sinusoid 256 [cos | sin] ─ MLP ─ temb(t) ─┬─ SiLU ─ modulation ──▶ 32 layers ─▶ target
///     0 ─ sinusoid 256 ─ MLP ─ temb(0) ─────────────┘   (conditions read the t = 0 row)
///                                                   └─ SiLU ─ norm_out.linear ─ final scale
///
/// **One layer** — a single sequence, a single set of weights, LayerNorms without affine:
///
///     x += tanh(g₁) · Wo( SDPA_blockcausal(RoPE(RMS(q)), RoPE(RMS(k)), v) )   on LN(x)·(1 + s₁)
///     x += tanh(g₂) · W_out( SiLU(W_gate n) ⊙ W_proj n )                       n = LN(x)·(1 + s₂)
///
/// **Two paths, one implementation** (`pass`): the *prefill* runs the conditions alone (text and
/// references, rows `[0, P)`) and keeps their K/V per layer; the *step* runs the target alone and
/// rereads them. The conditions never see the target and are modulated by `t = 0`: their K/V do not
/// depend on the step — computed once per render. The *full* pass (the whole sequence under the
/// block-causal mask) exists as the check of the two others.
///
/// What sets it apart from the five other targets, and what a port by analogy would miss:
///
///   - **block-causal attention** (`(q ≥ kv) ∨ same image`): text is strictly causal, each image
///     block bidirectional in itself, and everything sees what precedes it. It is done as the
///     reference's exact processor does (`QwenImage21AttnProcessor`): one SDPA per prefix segment
///     on the keys `[0, end)`, a causal triangle for a text segment, one SDPA for the target on
///     all the keys. Two **adjacent** references stay two blocks (`image_ids`, not runs of slots);
///   - **the conditions are modulated by `t = 0`** (`causal_condition`, `_select_modulation_rows`):
///     the modulation is computed for two timesteps, and the text/references take the second row —
///     in every layer AND in `norm_out`. Viggle's LoRA adapts the σ embedding and the modulation:
///     `temb(0)` changes with it, and so do the conditions' K/V;
///   - the modulation has **no shift**: `[scale₁, gate₁, scale₂, gate₂]`, gates through **tanh**;
///     `norm_out` is scale only (`QwenImage21AdaLayerNormContinuous`, `linear` → d, not 2d);
///   - the image tokens are **substituted into the encoder's sequence**: each Qwen3-VL image slot
///     stands for a 2×2 group, so each run of slots becomes `4·slots` rows in which the latent goes
///     **row-major** (`_pack_latents` is a plain flatten, the 2×2 grouping does not reorder it). The
///     encoder rows at the slots go through `txt_in` in the reference and are then overwritten:
///     here only the text rows go through it. The target's slots are appended by the pipeline;
///   - the sinusoid is **`[cos | sin]`** (cos first), over 256 channels, on `1000 · t` with
///     `t = (σ·1000)/1000` — the pipeline divides the scheduler's timestep by 1000;
///   - `txt_in` is RMSNorm (zero-centred: `+1` folded by the forge) → Linear → **GELU tanh** → Linear;
///   - the RoPE has three axes `(16, 56, 56)`, θ = 10 000, interleaved pairs: text at `(p, p, p)`
///     with `p` advancing token by token; an image freezes the frame axis at the current `p`, lays
///     out `(y, x)` **centred on zero** (`y ∈ [−(h − h/2), h/2)`), then advances `p` by `max(h, w)`.
///     The positions of a reference do not depend on where it sits in the sequence;
///   - no bias anywhere; `img_in` and `proj_out` work at 64 channels, `patch_size` 1.
///
/// The cache costs `2 · 32 · P · d` floats — 1.05 GB per 1 000 condition tokens; at 512² with two
/// references (P = 2 072), 2.2 GB. It belongs to the DiT, hence to its LoRA stack: the turbo
/// changes `temb(0)`, so a step without the LoRA (the 9-step mode's last two) needs a new prefill.
/// The step copies the cached prefix into `keys`/`values` before attending (0.64 s per evaluation
/// at P = 2 072, measured): the price of not holding `P + S` rows per layer.
package final class QwenImage21DiT {
    package struct Config {
        package let dim, heads, headDim, hidden, layers, channels, context: Int
        package let axes: [Int]
        package let theta, eps: Double

        package init(header: [String: Any]) {
            let c = header["config"] as? [String: Any] ?? [:]
            heads = c["num_attention_heads"] as? Int ?? 32
            headDim = c["attention_head_dim"] as? Int ?? 128
            dim = heads * headDim
            hidden = dim * (c["mlp_ratio"] as? Int ?? 3)
            layers = c["num_layers"] as? Int ?? 32
            channels = c["in_channels"] as? Int ?? 64
            context = c["context_in_dim"] as? Int ?? 4096
            axes = c["axes_dims_rope"] as? [Int] ?? [16, 56, 56]
            theta = 10000   // `QwenImage21Rope(theta=10000, …)`, written in the model, not in the config
            eps = c["eps"] as? Double ?? 1e-6
        }
        /// The modulation floats of one timestep: `[1 + s₁, tanh g₁, 1 + s₂, tanh g₂ | 1 + s_final]`.
        package var modulation: Int { 5 * dim }
    }

    package let config: Config
    package let sequence: QwenImage21Sequence
    package private(set) var boundaries: [String: [Float]] = [:]
    package var recordBoundaries = false
    /// The token of the render in progress, consulted between two layers (see `Cancellation`).
    package var cancellation: Cancellation?
    package private(set) var timings: [String: Double] = [:]
    package func resetTimings() { timings.removeAll() }

    private let artifact: Artifact
    private let prefetcher: Prefetcher
    private let gemm: GEMM
    private let elementwise: ElementwiseGPU
    /// The rows `x` holds: `P + S` — the prefix's residual stream in rows `[0, P)`, the target's after
    /// it, both alive during the first evaluation (`prefillPending`) and the full pass.
    private let capacity: Int
    /// The rows of the per-layer scratch (`normed`, `q`, `merged`, `o`): `P + S` with the full pass,
    /// otherwise the largest of the target and the prefill's groups (`groups`).
    private let scratch: Int
    /// **The MLP runs by row chunks** of at most this many rows (`mlpRows(scratch:hidden:)`), only
    /// as many as its two `[rows, 12 288]` buffers need to stay under `mlpBudget`. The price of a
    /// chunk: the three MLP weights widened once more — a 1024² generation in one chunk instead of
    /// two widens in 0.83 s instead of 1.31 per evaluation, and its MLP rows pass `LoRA.merges`.
    private let mlpRows: Int
    /// The bytes of `h1` + `h2` (`2 · rows · hidden · 4`) one MLP chunk may hold: a 1024² generation
    /// (4 096 rows, 403 MB) in one chunk, 1024×1536 (6 144 rows, 604 MB) in two.
    package static let mlpBudget = 512 << 20

    /// The rows of one MLP chunk for `scratch` rows: the fewest equal chunks whose two buffers fit
    /// `budget` — `ceil(scratch / chunks)`, at least one row.
    package static func mlpRows(scratch: Int, hidden: Int, budget: Int = mlpBudget) -> Int {
        var chunks = max(1, (2 * scratch * hidden * 4 + budget - 1) / budget)
        // The bytes divide evenly, the rows round up: one chunk more when the rounding passes the budget.
        while chunks < scratch, 2 * ((scratch + chunks - 1) / chunks) * hidden * 4 > budget { chunks += 1 }
        return max(1, (scratch + chunks - 1) / chunks)
    }
    /// **The prefill, by groups of whole segments**: consecutive prefix segments run through a layer
    /// together, at most `max(S, largest image) + all text rows` rows — one group for an edit with one
    /// reference, three for three. Each group's attention needs the K/V of the rows before it, which
    /// the previous groups of the same layer have just written: the same computation as one pass over
    /// `P` rows, and the scratch holds a group, not the prefix. **Not the same bits**: MPS picks its GEMM
    /// kernel by height, and another height rounds differently — measured on the edit-2 trajectory at
    /// 512, the final latent moved from 5.0·10⁻⁵ to 3.6·10⁻⁵ of the oracle (100.4 dB against 99.2). The
    /// price: each weight widened once per group (pages already in memory).
    private let groups: [(start: Int, end: Int, segments: [QwenImage21Sequence.Segment])]
    private let allowsFullPass: Bool
    private var attentions: [String: Attention] = [:]

    /// `merged` (the SDPA's output) **is** `normed`: the norm's output is consumed by q/k/v before the
    /// attention writes, and rewritten after `to_out` has read it. `q` **is** `h1`: the queries die with
    /// the attention, `h1` is born in the MLP. Same floats, fewer bytes.
    private let x, normed, q, merged, o, h1, h2, reserve, tokens, freqs: UnsafeMutablePointer<Float>
    /// The floats `reserve` holds (`d·h`, the largest weight): what `materialize` may write into it.
    private var reserveCapacity: Int { config.dim * config.hidden }
    /// `[P + S, d]`: the keys and values of a pass, at their joint row — the step copies the
    /// cached prefix into rows `[0, P)` and writes the target's after it.
    private let keys, values: UnsafeMutablePointer<Float>
    /// `[P, d]`: the joint sequence's prefix before layer 0 (`txt_in` and `img_in` outputs) — kept
    /// only for the full pass, which restarts from it at every evaluation; the prefill writes into `x`.
    private let prefixInput: UnsafeMutablePointer<Float>?
    /// Per layer `[P, d]` of K then `[P, d]` of V, post-norm and post-RoPE, as the reference caches them.
    private let cache: UnsafeMutablePointer<Float>?
    private var prefilled = false
    /// **The prefill runs inside the first evaluation**, layer by layer: layer `l` of the conditions,
    /// its K/V stored, then layer `l` of the target — each layer's weights widened and its LoRA expanded
    /// once for both, where a prefill of its own read the whole map once more (8.5 s of a 1024²
    /// generation, whose prefix is a few dozen text rows). The same GEMMs at the same heights, the same
    /// SDPAs, the K/V the cache would give back: **the same bits**. Set by `prefill`, cleared by the
    /// first `forward`.
    private var prefillPending = false
    /// The cache on disk (`CacheStorage.file`), instead of `cache`.
    private let kvFile: QwenImage21KVFile?
    package let storage: CacheStorage
    /// The conditions' K/V come from a kept file (`QwenImage21KVFile.complete`): no prefill this render.
    package var reusedPrefill: Bool { kvFile?.complete ?? false }
    /// The conditions' K/V go to a kept file this render.
    package var keepsPrefill: Bool { kvFile?.writesKept ?? false }
    /// `[t = 0 | t]`, `config.modulation` floats each.
    private let modulationSlot: UnsafeMutablePointer<Float>
    private var reserved: [UnsafeMutablePointer<Float>: Int] = [:]
    private let arenas: [Arena]

    /// **Viggle's turbo** (and any LoRA of the family), on the 224 `Linear`s of the blocks: merged into
    /// the fp32 widened weight beyond `LoRA.merges`' rows, applied after the GEMM (`gemm.lora`) below;
    /// and unmerged, in double, on the modules of the modulation path
    /// (`time_text_embed.timestep_embedder.linear_1/2`, `modulation.1`, `norm_out.linear`) — see `exactLinears`.
    package let lora: LoRA?
    /// Expanded two layers at a time, the next one in the background (`StreamedLoRA`).
    private let cacheLoRA: StreamedLoRA?
    private let loraMid: UnsafeMutablePointer<Float>?
    /// The modules computed in double, outside the GEMMs.
    static let exactModules = ["time_text_embed.timestep_embedder.linear_1", "time_text_embed.timestep_embedder.linear_2",
                               "modulation.1", "norm_out.linear"]

    /// **Where the conditions' K/V live between the prefill and the steps** — the pipeline chooses
    /// (`QwenImage21CachePolicy`). The three give the same bits: the K/V are the same floats, kept,
    /// moved through a file, or computed again.
    package enum CacheStorage: Equatable, Sendable {
        /// In an arena: `2 · 32 · P · d` floats resident for the whole render.
        case memory
        /// In an unlinked file read back at each step, without the buffer cache (`QwenImage21KVFile`).
        case file
        /// Nowhere: each step is the full pass over `P + S` rows (`fullForward`), the prefill only
        /// prepares the conditions' inputs.
        case recompute
    }

    /// - Parameters:
    ///   - fullPass: reserve for the whole sequence at once, for `fullForward` (the check).
    ///   - kept: with `storage: .file`, the render cache's file for these conditions
    ///     (`QwenImage21KVFile.kept`) — read back without a prefill if it is whole, written and kept
    ///     otherwise; the unlinked file when keeping it would leave less than `reserve` bytes free.
    package init(artifact: Artifact, sequence: QwenImage21Sequence, lora: LoRA? = nil, fullPass: Bool = false,
                 storage: CacheStorage = .memory, kept: (path: String, key: String, reserve: Int64)? = nil,
                 freezeCut: Bool = EngineSettings.effective.frozenCut) throws {
        guard let kind = artifact.header["kind"] as? String, Family(ditKind: kind) == .qwenImage21 else {
            throw Artifact.Failure.badHeader("map \(artifact.header["kind"] ?? "?"): a Qwen-Image-2.1 DiT expected")
        }
        guard artifact.linearWeightsTransposed else {
            throw Artifact.Failure.badHeader("Qwen-Image-2.1 map forged without transposed Linears")
        }
        self.artifact = artifact
        self.config = Config(header: artifact.header)
        self.sequence = sequence
        self.prefetcher = Prefetcher(artifact: artifact)
        self.gemm = try GEMM(freezeCut: freezeCut)
        self.elementwise = try ElementwiseGPU(device: gemm.device, queue: gemm.queue)
        self.allowsFullPass = fullPass || storage == .recompute
        self.storage = storage
        let p = sequence.prefix, s = sequence.target, all = sequence.count
        capacity = all
        let textRows = sequence.segments.filter(\.isText).reduce(0) { $0 + $1.end - $1.start }
        let largestSegment = sequence.segments.filter { !$0.isText }.map { $0.end - $0.start }.max() ?? 0
        let budget = max(s, largestSegment) + textRows
        var groups: [(start: Int, end: Int, segments: [QwenImage21Sequence.Segment])] = []
        for segment in sequence.segments {
            if let last = groups.last, segment.end - last.start <= budget {
                groups[groups.count - 1] = (last.start, segment.end, last.segments + [segment])
            } else {
                groups.append((segment.start, segment.end, [segment]))
            }
        }
        self.groups = groups
        scratch = allowsFullPass ? all : max(s, sequence.textRows.count, groups.map { $0.end - $0.start }.max() ?? 0)
        let d = config.dim, h = config.hidden, n = scratch
        guard d % (Arena.alignment / 4) == 0, h % (Arena.alignment / 4) == 0, config.context == d else {
            throw Artifact.Failure.badHeader("Qwen-Image-2.1: a width that does not fall on a page")
        }
        let r = lora?.maxRank ?? 0
        let largestImage = sequence.images.map { $0.height * $0.width }.max() ?? 0
        mlpRows = QwenImage21DiT.mlpRows(scratch: n, hidden: h)   // `h1` is also `q`: sized for both below
        let floats = capacity * d + 2 * n * d + 2 * all * d + max(mlpRows * h, n * d) + mlpRows * h + d * h
            + (allowsFullPass ? p * d : 0) + largestImage * config.channels + all * config.headDim
            + 2 * config.modulation + n * r
        let arena = try Arena(capacity: floats * 4 + 32 * Arena.alignment + (8 << 20))
        var sizes: [UnsafeMutablePointer<Float>: Int] = [:]
        func slot(_ arena: Arena, _ name: String, _ count: Int) throws -> UnsafeMutablePointer<Float> {
            let pointer = try arena.reserve(name, bytes: max(count, 1) * 4).assumingMemoryBound(to: Float.self)
            sizes[pointer] = max(count, 1) * 4
            return pointer
        }
        x = try slot(arena, "x", capacity * d); normed = try slot(arena, "normed", n * d)
        merged = normed; o = try slot(arena, "o", n * d)
        keys = try slot(arena, "keys", all * d); values = try slot(arena, "values", all * d)
        h1 = try slot(arena, "h1", max(mlpRows * h, n * d)); h2 = try slot(arena, "h2", mlpRows * h)
        q = h1
        reserve = try slot(arena, "reserve", d * h)
        prefixInput = allowsFullPass ? try slot(arena, "prefixInput", p * d) : nil
        tokens = try slot(arena, "tokens", largestImage * config.channels)
        freqs = try slot(arena, "freqs", all * config.headDim)
        modulationSlot = try slot(arena, "modulation", 2 * config.modulation)
        var arenas = [arena]
        if p > 0, storage == .memory {
            let cacheArena = try Arena(capacity: 2 * config.layers * p * d * 4 + Arena.alignment)
            cache = try slot(cacheArena, "cache", 2 * config.layers * p * d)
            arenas.append(cacheArena)
        } else {
            cache = nil
        }
        if p > 0, storage == .file {
            let layers = config.layers
            kvFile = try kept.flatMap {
                QwenImage21KVFile.kept(at: $0.path, key: $0.key, halfBytes: p * d * 4, layers: layers, reserve: $0.reserve)
            } ?? QwenImage21KVFile(halfBytes: p * d * 4, layers: layers)
        } else {
            kvFile = nil
        }
        self.lora = r > 0 ? lora : nil
        if let lora, r > 0 {
            loraMid = try slot(arena, "loraMid", n * r)
            cacheLoRA = try StreamedLoRA(lora: lora, layers: config.layers,
                                         retain: { !QwenImage21DiT.exactModules.contains($0) })
        } else {
            loraMid = nil; cacheLoRA = nil
        }
        self.arenas = arenas
        reserved = sizes
        QwenImage21Rope.write(into: freqs, positions: sequence.positions, axes: config.axes, theta: config.theta)
    }

    private func record(_ name: String, _ p: UnsafePointer<Float>, _ n: Int) {
        guard recordBoundaries else { return }
        boundaries[name] = Array(UnsafeBufferPointer(start: p, count: n))
    }

    private func timed<T>(_ phase: String, _ body: () throws -> T) rethrows -> T {
        let started = DispatchTime.now().uptimeNanoseconds
        defer { timings[phase, default: 0] += Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9 }
        return try body()
    }

    // ── the two entry points ────────────────────────────────────────────────────────────

    /// **The conditions, once per render**: `txt_in` on the text rows of `hidden` (the encoder's
    /// `[T, 4096]`, image slots included — those rows are not read), `img_in` on each reference
    /// (`[64, hᵢ, wᵢ]`, conditions in order), then the 32 layers on the prefix alone, keeping K/V — now,
    /// or with `intoFirstStep` inside the first `forward`, layer by layer in front of the target
    /// (`prefillPending`: the sampler's way; the DiT check judges the prefix's own boundaries).
    package func prefill(hidden: UnsafePointer<Float>, references: [UnsafePointer<Float>],
                         intoFirstStep: Bool = false) throws {
        guard references.count == sequence.images.count - 1 else {
            throw Artifact.Failure.misuse("Qwen-Image-2.1: \(references.count) references for \(sequence.images.count - 1) blocks")
        }
        // **The conditions' K/V read back from a kept file** (`reusedPrefill`): nothing to compute —
        // `txt_in`, `img_in` and the `t = 0` row serve only the prefix's own layers. Every step then
        // reads the file, the first one included: the path the later steps of a fresh render take, and
        // the fused first step gives the same bits.
        if reusedPrefill { prefilled = true; return }
        let d = config.dim, c = config.channels
        prefetcher.request(["txt_in.text_norm.weight", "txt_in.in_layer.weight", "txt_in.out_layer.weight", "img_in.weight"])
        prefetcher.request(prefetcher.namesOfBlock(prefix: "transformer_blocks.0."))
        // Text: gather the text rows, RMSNorm (+1 folded), in_layer, GELU tanh, out_layer, scatter.
        let text = sequence.textRows
        try autoreleasepool {
        if !text.isEmpty {
            for (i, row) in text.enumerated() { (normed + i * d).update(from: hidden + row.encoderRow * d, count: d) }
            try artifact.materialize("txt_in.text_norm.weight", into: reserve, capacity: reserveCapacity)
            timed("norm") { Ops.rmsNorm(normed, weight: reserve, into: normed, rows: text.count, columns: d, eps: Float(config.eps)) }
            try linearGPU("txt_in.in_layer.weight", a: normed, into: o, m: text.count, k: d, n: d)
            timed("gelu") { QwenImage21DiT.geluTanh(o, count: text.count * d) }
            try linearGPU("txt_in.out_layer.weight", a: o, into: q, m: text.count, k: d, n: d)
            for (i, row) in text.enumerated() { ((prefixInput ?? x) + row.joint * d).update(from: q + i * d, count: d) }
        }
        // References: `img_in` straight to their block of rows (a page: d floats = 16 KB).
        for (i, reference) in references.enumerated() {
            let grid = sequence.images[i], count = grid.height * grid.width
            vDSP_mtrans(reference, 1, tokens, 1, vDSP_Length(count), vDSP_Length(c))   // [C, S] → [S, C]
            try linearGPU("img_in.weight", a: tokens, into: (prefixInput ?? x) + sequence.blockStarts[i] * d,
                          m: count, k: c, n: d)
        }
        }   // autoreleasepool
        try zeroModulation()
        // Recomputing: the conditions run inside each step's full pass, from `prefixInput`.
        if sequence.prefix > 0, storage != .recompute {
            if intoFirstStep { prefillPending = true } else { try pass(.prefill, sigma: nil, latent: nil) }
        }
        prefilled = true
    }

    /// **One evaluation on the target alone**, from the cache. `latent` as `[64, h, w]`, `sigma` as
    /// the scheduler holds it. Returns the velocity `[64, h, w]`.
    package func forward(latent: UnsafePointer<Float>, sigma: Float) throws -> [Float] {
        guard prefilled else { throw Artifact.Failure.misuse("Qwen-Image-2.1: `prefill` before `forward`") }
        return try pass(storage == .recompute ? .full : .step, sigma: sigma, latent: latent)
    }

    /// **The check of the two others**: the whole joint sequence in one pass under the
    /// block-causal mask, without the cache (which it neither reads nor writes).
    package func fullForward(latent: UnsafePointer<Float>, sigma: Float) throws -> [Float] {
        guard allowsFullPass else { throw Artifact.Failure.misuse("Qwen-Image-2.1: built without `fullPass`") }
        guard prefilled else { throw Artifact.Failure.misuse("Qwen-Image-2.1: `prefill` before `fullForward`") }
        return try pass(.full, sigma: sigma, latent: latent)
    }

    // ── the pass ────────────────────────────────────────────────────────────────────────

    private enum Pass { case prefill, step, full }

    /// Rows `[first, P + S)` of the joint sequence, each held at its joint row of `x` — and, while the
    /// prefill is pending, the conditions `[0, P)` first, layer by layer, in front of the target.
    /// Within its own autorelease pool (see the pool per layer in `body`).
    @discardableResult
    private func pass(_ kind: Pass, sigma: Float?, latent: UnsafePointer<Float>?) throws -> [Float] {
        try autoreleasepool { try body(kind, sigma: sigma, latent: latent) }
    }

    private func body(_ kind: Pass, sigma: Float?, latent: UnsafePointer<Float>?) throws -> [Float] {
        let d = config.dim, c = config.channels, p = sequence.prefix, s = sequence.target
        let first = kind == .step ? p : 0, end = kind == .prefill ? p : p + s, n = end - first
        let zero = modulationSlot, timed0 = modulationSlot + config.modulation
        // The conditions' layer `l`, then the target's: the prefill inside the first step.
        let fused = kind == .step && prefillPending
        let xs = x + first * d

        if let sigma {
            let m = try modulations[sigma.bitPattern] ?? timed("modulation") { try modulation(sigmas: [sigma])[0] }
            m.withUnsafeBufferPointer { timed0.update(from: $0.baseAddress!, count: config.modulation) }
        }
        // The bands: the conditions `[first, P)` read `t = 0`, the target `[P, end)` reads `t`.
        var bands: [(row: Int, count: Int, modulation: UnsafeMutablePointer<Float>)] = []
        if first < p { bands.append((0, p - first, zero)) }
        if end > p { bands.append((p - first, end - p, timed0)) }

        if first < p || fused, let prefixInput { x.update(from: prefixInput, count: p * d) }
        if let latent {
            prefetcher.request(["img_in.weight"])
            vDSP_mtrans(latent, 1, tokens, 1, vDSP_Length(s), vDSP_Length(c))   // [C, S] → [S, C]
            try linearGPU("img_in.weight", a: tokens, into: x + p * d, m: s, k: c, n: d)
        }
        record("joint_in", xs, n * d)

        // The tail of the map is streamed around the page cache, the head left to it
        // (`TailStream`): 14.2 GB swept at every evaluation would otherwise miss every cached page.
        // **Only while the conditions' K/V live in memory**, i.e. a generation: an edit keeps its
        // K/V in a file and runs at the memory ceiling — at the worst case (1024×1536, three
        // references) the two staging buffers pushed the machine to 48 swapped pages with 19 MB
        // free, and the rule is zero. There the map keeps sweeping the cache, as before.
        // **And it is faster so**: a 1-reference edit at 1024² with the tail streamed, as the
        // budget once allowed, read less (~14 against 19.63 GB per step) yet took 28.2 s a step against
        // 26.0 — 208.1 against 195.5 s, same bits. A generation's tail is the render's to decide
        // (`MemoryPlan.ditTail`: half by default, the whole map in the lean plan); an edit's
        // stays off in the lean plan too — its map through the cache is what was timed fastest, and
        // its own K/V reader is already beside the cache.
        let layerBlocks = (0..<config.layers).map { prefetcher.namesOfBlock(prefix: "transformer_blocks.\($0).") }
        var (released, slots) = (config.layers, 0)
        if storage == .memory {
            (released, slots) = MemoryPlan.ditTail(artifact: artifact, layerBlocks: layerBlocks)
        } else {
            MemoryPlan.noteOnce("map tail none (edit: K/V in a file)")
        }
        try artifact.streamTail(blocks: slots == 0 ? [] : Array(layerBlocks[released...]), slots: slots)
        // Requested once the stream exists: requested before, a streamed block 0 would be paged in
        // through the cache, then read again by the stream.
        prefetcher.request(prefetcher.namesOfBlock(prefix: "transformer_blocks.0."))
        for layer in 0..<config.layers {
            try cancellation.check()   // between two layers, never inside one (see `Cancellation`)
            for ahead in 1...2 where layer + ahead < config.layers {
                prefetcher.request(prefetcher.namesOfBlock(prefix: "transformer_blocks.\(layer + ahead)."))
            }
            if layer == config.layers - 1, kind != .prefill {
                prefetcher.request(["norm_out.linear.weight", "proj_out.weight"])
            }
            // **An autorelease pool per layer.** Each SDPA call wraps our buffers in autoreleased
            // `MPSGraphTensorData`; on a thread whose pool never drains (the CLI's main thread, a
            // `sync` on the engine queue) they — and the `MTLBuffer`s over the arena they retain —
            // outlived the DiT: 1.0 GB of Metal still allocated after a 1024² DiT was released.
            try autoreleasepool {
                if kind == .prefill || fused {
                    for group in groups {
                        let rows = group.end - group.start
                        try layerPass(layer, kind: .prefill, first: group.start, rows: rows, bands: [(0, rows, zero)],
                                      x: x + group.start * d, segments: group.segments)
                    }
                    try timed("kv cache") { try storeCache(layer) }
                }
                if kind != .prefill {
                    try layerPass(layer, kind: kind, first: first, rows: n, bands: bands, x: xs,
                                  segments: kind == .full ? sequence.segments : [], fused: fused)
                }
            }
            // A streamed layer gives its staging buffer back; the head stays cached for the next evaluation.
            if layer >= released {
                artifact.dropFromCache(prefetcher.namesOfBlock(prefix: "transformer_blocks.\(layer)."))
            }
            if layer == 0 { record("layer0_out", xs, n * d) }
            if layer == 1 { record("layer1_out", xs, n * d) }
            if layer == config.layers - 1, kind != .prefill { record("layer_last_out", xs, n * d) }
        }
        if fused { prefillPending = false }
        // Every layer's K/V is on disk: a file being kept becomes the one the next render reads.
        if fused || kind == .prefill { kvFile?.seal() }
        guard kind != .prefill else { return [] }

        // ── norm_out: LN without affine, × (1 + scale), each band its row; then proj_out ───
        timed("norm") {
            Ops.layerNorm(xs, into: normed, rows: n, columns: d, eps: Float(config.eps))
            for band in bands {
                Ops.scaleRows(normed + band.row * d, by: band.modulation + 4 * d, rows: band.count, columns: d)
            }
        }
        record("norm_out", normed, n * d)
        let target = normed + (p - first) * d
        try linearGPU("proj_out.weight", a: target, into: o, m: s, k: d, n: c)
        var out = [Float](repeating: 0, count: c * s)
        out.withUnsafeMutableBufferPointer { vDSP_mtrans(o, 1, $0.baseAddress!, 1, vDSP_Length(c), vDSP_Length(s)) }
        record("model_out", out, out.count)
        return out
    }

    /// The prefill's K/V of `layer` (rows `[0, P)` of `keys`/`values`, all groups done) to the cache.
    private func storeCache(_ layer: Int) throws {
        let d = config.dim, p = sequence.prefix
        if let cache {
            let layerCache = cache + 2 * layer * p * d
            copyRows(keys, into: layerCache, rows: p)
            copyRows(values, into: layerCache + p * d, rows: p)
        } else if let kvFile {
            try kvFile.write(layer: layer, keys: keys, values: values)
        }
    }

    /// One layer on joint rows `[first, first + n)`, whose residual stream is `xs`; `segments`: the
    /// prefix segments among them, each attending to the keys before its end. `fused`: a step whose
    /// prefill has just run this layer — the conditions' K/V are already in `keys`/`values`.
    private func layerPass(_ layer: Int, kind: Pass, first: Int, rows n: Int,
                           bands: [(row: Int, count: Int, modulation: UnsafeMutablePointer<Float>)],
                           x xs: UnsafeMutablePointer<Float>, segments: [QwenImage21Sequence.Segment],
                           fused: Bool = false) throws {
        let d = config.dim, hh = config.hidden, p = sequence.prefix, all = sequence.count
        let prefix = "transformer_blocks.\(layer)."
        let k = keys + first * d, v = values + first * d

        // ── attention ─────────────────────────────────────────────────────────────────────
        // The prefill's last layer: the conditions' K/V are all that later steps read.
        let keysOnly = kind == .prefill && layer == config.layers - 1
        modulatedNorm(xs, rows: n, bands: bands, scale: 0)
        if !keysOnly { try linearGPU(prefix + "attn.to_q.weight", a: normed, into: q, m: n, k: d, n: d) }
        try linearGPU(prefix + "attn.to_k.weight", a: normed, into: k, m: n, k: d, n: d)
        try linearGPU(prefix + "attn.to_v.weight", a: normed, into: v, m: n, k: d, n: d)
        if !keysOnly { try headNorm(prefix + "attn.norm_q.weight", q, rows: n) }
        try headNorm(prefix + "attn.norm_k.weight", k, rows: n)
        timed("rope") {
            let table = freqs + first * config.headDim
            if !keysOnly { Ops.rope(q, freqs: table, sequence: n, heads: config.heads, headDim: config.headDim) }
            Ops.rope(k, freqs: table, sequence: n, heads: config.heads, headDim: config.headDim)
        }
        if kind == .step, !fused {
            try timed("kv cache") {
                if let cache {
                    let layerCache = cache + 2 * layer * p * d
                    copyRows(layerCache, into: keys, rows: p)
                    copyRows(layerCache + p * d, into: values, rows: p)
                } else if let kvFile {
                    try kvFile.wait(layer: layer, keys: keys, values: values)
                }
            }
        }
        if keysOnly { return }

        // Block-causal: each prefix segment on the keys `[0, end)`, the target on all of them.
        for segment in segments {
            try attend(queries: segment.start, count: segment.end - segment.start, keys: segment.end,
                       causal: segment.isText, first: first)
        }
        if kind != .prefill { try attend(queries: p, count: all - p, keys: all, causal: false, first: first) }
        // The prefix rows of `keys`/`values` are free from here: the next layer's K/V come in from
        // disk while this one finishes — after the last layer, layer 0 of the next step. Not while the
        // prefill is fused: layer `l + 1` is not on disk yet, and its prefill is about to write those rows.
        if kind == .step, let kvFile, !fused || layer == config.layers - 1 {
            kvFile.prefetch(layer: (layer + 1) % config.layers, keys: keys, values: values)
        }
        try linearGPU(prefix + "attn.to_out.0.weight", a: merged, into: o, m: n, k: d, n: d)
        try residual(xs, rows: n, bands: bands, gate: d)

        // ── MLP: W_out(SiLU(gate) ⊙ proj) ─────────────────────────────────────────────────
        modulatedNorm(xs, rows: n, bands: bands, scale: 2 * d)
        let (h1b, h2b) = (try slice(h1), try slice(h2))
        let chunks = (n + mlpRows - 1) / mlpRows
        for chunk in 0..<chunks {
            let begin = chunk * n / chunks, rows = (chunk + 1) * n / chunks - begin
            try linearGPU(prefix + "img_mlp.gate_layer.weight", a: normed + begin * d, into: h1, m: rows, k: d, n: hh)
            try linearGPU(prefix + "img_mlp.proj.weight", a: normed + begin * d, into: h2, m: rows, k: d, n: hh)
            timed("swiglu") { timings["elementwise GPU", default: 0] += elementwise.swiglu((h1b, 0), (h2b, 0), count: rows * hh) }
            try linearGPU(prefix + "img_mlp.out.weight", a: h1, into: o + begin * d, m: rows, k: hh, n: d)
        }
        try residual(xs, rows: n, bands: bands, gate: 3 * d)
    }

    // ── the modulation ──────────────────────────────────────────────────────────────────

    /// The bits of σ → the modulation of the target at that σ (`config.modulation` floats).
    package typealias ModulationTable = [UInt32: [Float]]
    package var modulations: ModulationTable = [:]

    /// **Precomputes the modulation of the σ that will be evaluated** — each modulation weight read
    /// once for the whole render.
    package func tabulateModulation(sigmas: [Float]) throws {
        let newKeys = Array(Set(sigmas.map(\.bitPattern)).subtracting(modulations.keys)).sorted()
        guard !newKeys.isEmpty else { return }
        let values = try timed("modulation") { try modulation(sigmas: newKeys.map(Float.init(bitPattern:))) }
        for (key, value) in zip(newKeys, values) { modulations[key] = value }
    }

    /// The `t = 0` row, which the conditions read in every pass.
    private func zeroModulation() throws {
        let m = try timed("modulation") { try modulation(timesteps: [0])[0] }
        m.withUnsafeBufferPointer { modulationSlot.update(from: $0.baseAddress!, count: config.modulation) }
    }

    /// The timestep the DiT receives: the scheduler's `t = σ·1000`, divided by 1000 by the pipeline.
    package static func timestep(sigma: Float) -> Float { (sigma * 1000) / 1000 }

    private func modulation(sigmas: [Float]) throws -> [[Float]] {
        try modulation(timesteps: sigmas.map(QwenImage21DiT.timestep(sigma:)))
    }

    /// `temb = linear_2(SiLU(linear_1(sinusoid)))`, `modulation.1(SiLU(temb))`,
    /// `norm_out.linear(SiLU(temb))` — **in double**, each weight read once for all the timesteps,
    /// the LoRA added unmerged. Stored as `[1 + s₁, tanh g₁, 1 + s₂, tanh g₂, 1 + s_final]`.
    private func modulation(timesteps: [Float]) throws -> [[Float]] {
        let d = config.dim
        let (_, layers, final) = try exactModulation(timesteps: timesteps)
        return timesteps.indices.map { i in
            var out = [Float](repeating: 0, count: config.modulation)
            for j in 0..<d {
                out[j] = Float(1 + layers[i][j])
                out[d + j] = Float(tanh(layers[i][d + j]))
                out[2 * d + j] = Float(1 + layers[i][2 * d + j])
                out[3 * d + j] = Float(tanh(layers[i][3 * d + j]))
                out[4 * d + j] = Float(1 + final[i][j])
            }
            return out
        }
    }

    /// `(temb, modulation.1, norm_out.linear)` per timestep, before `1 +` and `tanh`.
    private func exactModulation(timesteps: [Float]) throws -> (temb: [[Double]], layers: [[Double]], final: [[Double]]) {
        let sinusoids = timesteps.map { QwenImage21DiT.sinusoid(timestep: $0) }
        var hidden = try exactLinears("time_text_embed.timestep_embedder.linear_1", xs: sinusoids)
        for i in hidden.indices { KleinDiT.silu(&hidden[i]) }
        let temb = try exactLinears("time_text_embed.timestep_embedder.linear_2", xs: hidden)
        var enabled = temb
        for i in enabled.indices { KleinDiT.silu(&enabled[i]) }
        return (temb, try exactLinears("modulation.1", xs: enabled), try exactLinears("norm_out.linear", xs: enabled))
    }

    /// The reference's `temb` `[2, d]` and `modulation` `[2, 4d]` (row 0 at σ, row 1 at `t = 0`) — for the checks.
    package func probeModulation(sigma: Float) throws -> (temb: [Float], modulation: [Float]) {
        let (temb, layers, _) = try exactModulation(timesteps: [QwenImage21DiT.timestep(sigma: sigma), 0])
        return (temb.flatMap { $0.map(Float.init) }, layers.flatMap { $0.map(Float.init) })
    }

    /// `Krea2Exact.linears`, plus the LoRA stack **unmerged, in double**: `y += (x·down)·up`, the
    /// strength folded into `down` by `LoRA.materialize`.
    private func exactLinears(_ module: String, xs: [[Double]]) throws -> [[Double]] {
        var ys = try Krea2Exact.linears(artifact, module, xs: xs)
        guard let lora, lora.rank(module) > 0, let shape = artifact.tensors[module + ".weight"]?.shape else { return ys }
        let (k, n, r) = (shape[0], shape[1], lora.rank(module))
        var down = [Float](repeating: 0, count: k * r), up = [Float](repeating: 0, count: r * n)
        try down.withUnsafeMutableBufferPointer { db in
            try up.withUnsafeMutableBufferPointer { ub in
                _ = try lora.materialize(module, k: k, n: n, down: db.baseAddress!, up: ub.baseAddress!)
            }
        }
        for v in xs.indices {
            var mid = [Double](repeating: 0, count: r)
            for i in 0..<k where xs[v][i] != 0 {
                let xi = xs[v][i]
                for c in 0..<r { mid[c] += xi * Double(down[i * r + c]) }
            }
            for c in 0..<r {
                let m = mid[c]
                for j in 0..<n { ys[v][j] += m * Double(up[c * n + j]) }
            }
        }
        return ys
    }

    /// **`QwenImage21TemporalTimesteps(256)`, in double**: `[cos | sin]` of `1000·t·ωⱼ`,
    /// `ωⱼ = exp(−ln(10⁴)·j/128)`. The reference does it in fp32 (angles up to 1000 rad); the target
    /// is the true function, that of the fp64 witness.
    package static func sinusoid(timestep: Float, dim: Int = 256) -> [Double] {
        let half = dim / 2
        let entry = 1000 * Double(timestep)
        var out = [Double](repeating: 0, count: dim)
        for j in 0..<half {
            let angle = entry * exp(-log(10000) * Double(j) / Double(half))
            out[j] = cos(angle)
            out[half + j] = sin(angle)
        }
        return out
    }

    /// `nn.GELU(approximate="tanh")` in double, rounded once.
    static func geluTanh(_ x: UnsafeMutablePointer<Float>, count: Int) {
        let c = (2 / Double.pi).squareRoot()
        let slices = Parallel.threads * 2
        Parallel.rows(slices, width: max(1, count / slices)) { first, howMany in
            for i in (first * count / slices)..<((first + howMany) * count / slices) {
                let v = Double(x[i])
                x[i] = Float(0.5 * v * (1 + tanh(c * (v + 0.044715 * v * v * v))))
            }
        }
    }

    // ── plumbing ────────────────────────────────────────────────────────────────────────

    /// `normed = LN(x) · (1 + scale)`, the scale of each band at `scale` floats into its modulation.
    private func modulatedNorm(_ xs: UnsafeMutablePointer<Float>, rows n: Int,
                               bands: [(row: Int, count: Int, modulation: UnsafeMutablePointer<Float>)], scale: Int) {
        let d = config.dim
        timed("norm") {
            Ops.layerNorm(xs, into: normed, rows: n, columns: d, eps: Float(config.eps))
            for band in bands {
                Ops.scaleRows(normed + band.row * d, by: band.modulation + scale, rows: band.count, columns: d)
            }
        }
    }

    /// `x += tanh(gate) · o` per band, on the GPU (one `fma`).
    private func residual(_ xs: UnsafeMutablePointer<Float>, rows n: Int,
                          bands: [(row: Int, count: Int, modulation: UnsafeMutablePointer<Float>)], gate: Int) throws {
        let d = config.dim
        let (xb, ob, mb) = (try slice(xs), try slice(o), try slice(modulationSlot))
        timed("residual") {
            for band in bands {
                timings["elementwise GPU", default: 0] += elementwise.residual(
                    (xb, band.row * d), plus: (ob, band.row * d), carries: (mb, band.modulation - modulationSlot + gate),
                    rows: band.count, columns: d)
            }
        }
    }

    /// QK-Norm: RMSNorm over `head_dim`, eps 1e-6, ordinary weight.
    private func headNorm(_ name: String, _ tensor: UnsafeMutablePointer<Float>, rows: Int) throws {
        try artifact.materialize(name, into: reserve, capacity: reserveCapacity)
        timed("norm") { Ops.rmsNorm(tensor, weight: reserve, into: tensor, rows: rows * config.heads,
                                    columns: config.headDim, eps: Float(config.eps)) }
    }

    private func copyRows(_ source: UnsafePointer<Float>, into destination: UnsafeMutablePointer<Float>, rows: Int) {
        let d = config.dim
        Parallel.rows(rows, width: d) { first, howMany in
            (destination + first * d).update(from: source + first * d, count: howMany * d)
        }
    }

    /// Queries `[queries, queries + count)` of the joint sequence (held from `first`) against the keys
    /// `[0, keys)` — causal: the queries are the last of those keys. Split into query slices only
    /// beyond `Attention.maxMatrix` (exact: the softmax is per row; a causal slice keeps its own keys).
    /// A lower ceiling was measured and refuted: 1 GB or 512 MB per call cost 3 to 6 % of the SDPA at
    /// 1024² and left the Metal peak of the 1-reference edit where it was (2.7–3.3 GB either way).
    private func attend(queries: Int, count: Int, keys keyCount: Int, causal: Bool, first: Int) throws {
        let d = config.dim, heads = config.heads
        let slices = max(1, Int((Double(heads * count * keyCount) * 4 / Double(Attention.maxMatrix)).rounded(.up)))
        let kb = try slice(keys), vb = try slice(values)
        for piece in 0..<slices {
            let begin = piece * count / slices, end = (piece + 1) * count / slices
            let rows = end - begin
            let visible = causal ? keyCount - count + end : keyCount
            let key = "\(rows)×\(visible)\(causal ? "c" : "")"
            let attention: Attention
            if let a = attentions[key] { attention = a } else {
                attention = Attention(device: gemm.device, queue: gemm.queue, heads: heads, sequence: rows,
                                      headDim: config.headDim, causal: causal, keySequence: visible)
                attentions[key] = attention
            }
            let local = queries + begin - first
            let (qb, ob) = (try slice(q + local * d), try slice(merged + local * d))
            timed("sdpa wall") { timings["sdpa GPU", default: 0] += attention.run(q: qb, k: kb, v: vb, into: ob) }
        }
    }

    /// A slice of an arena seen from the GPU, wrapped **to the end of its reservation**.
    private func slice(_ p: UnsafeMutablePointer<Float>) throws -> MTLBuffer {
        guard let (begin, bytes) = reserved.first(where: { $0.key <= p && p < $0.key + $0.value / 4 }) else {
            throw Artifact.Failure.misuse("Qwen-Image-2.1: an address outside the arena")
        }
        return try gemm.wrap(UnsafeMutableRawPointer(p), bytes: bytes - (p - begin) * 4, name: "slice")
    }

    /// `C[m, n] = A[m, k] · W` (W laid out `[k, n]` by the forge), with the LoRA stack either **merged
    /// into the widened weight** (`W += down·up`, one GEMM at depth `r` accumulated into `reserve`) or
    /// accumulated into the output after the GEMM (`gemm.lora`) — whichever costs fewer FLOPs at these
    /// rows (`LoRA.merges`). `reserve` is widened afresh at every call, so a merge never outlives its
    /// GEMM: the next call, at other rows, decides again (the prefill's text rows stay applied). Measured:
    /// at 1024² the turbo's application cost 2.0 s per evaluation, ∝ rows; the merge, a constant.
    private func linearGPU(_ name: String, a: UnsafeMutablePointer<Float>, into c: UnsafeMutablePointer<Float>,
                           m: Int, k: Int, n: Int) throws {
        let got = try timed("widening") { try artifact.materialize(name, into: reserve, capacity: reserveCapacity) }
        guard got == k * n else {
            throw Artifact.Failure.badHeader("\(name): \(got) values, \(k * n) expected")
        }
        let (ab, wb, cb) = (try slice(a), try slice(reserve), try slice(c))
        let target = String(name.dropLast(".weight".count))
        var lora: (down: MTLBuffer, up: MTLBuffer, rank: Int)?
        if let cache = cacheLoRA, let stack = try timed("lora weights", { try cache.module(target, k: k, n: n) }) {
            lora = (try gemm.wrap(UnsafeMutableRawPointer(stack.down), bytes: k * stack.rank * 4, name: "loraDown"),
                    try gemm.wrap(UnsafeMutableRawPointer(stack.up), bytes: stack.rank * n * 4, name: "loraUp"), stack.rank)
        }
        let merged = lora != nil && LoRA.merges(rows: m, k: k, n: n)
        if merged, let lora {
            timed("lora wall") {
                timings["lora GPU", default: 0] += gemm.accumulatedLinear(a: lora.down, b: lora.up, c: wb, m: k, k: lora.rank, n: n)
            }
        }
        timed("gemm wall") {
            timings["gemm GPU", default: 0] += gemm.linear(a: ab, b: wb, c: cb, m: m, k: k, n: n, weightIsTransposed: true)
        }
        if !merged, let lora, let mid = loraMid {
            let mb = try slice(mid)
            timed("lora wall") {
                timings["lora GPU", default: 0] += gemm.lora(x: ab, down: lora.down, mid: mb, up: lora.up, c: cb,
                                                             m: m, k: k, r: lora.rank, n: n)
            }
        }
    }
}

/// **The joint sequence of Qwen-Image-2.1**, as `QwenImage21Transformer2DModel.forward` builds it
/// from `img_mask` and `img_shapes` — a pure function of the layout, which the tests judge.
///
/// `slots` is the pipeline's `img_mask` row: one entry per Qwen3-VL position, `true` at an image
/// slot, **the target's slots appended** (`h·w/4` of them). Each slot becomes 4 rows; the images'
/// latent tokens fill those rows in order (conditions, then the target), row-major per image.
package struct QwenImage21Sequence: Equatable {
    package struct Grid: Equatable { package let height, width: Int
        package init(height: Int, width: Int) { self.height = height; self.width = width } }
    package struct Segment: Equatable { package let start, end: Int; package let isText: Bool }
    package struct TextRow: Equatable { package let joint, encoderRow: Int }

    /// Latent grids, the conditions in order, the target last.
    package let images: [Grid]
    /// Per joint row: `-1` for text, the image's index otherwise (`image_ids`).
    package let imageIds: [Int]
    /// The text rows: their joint row and their row in the encoder's output.
    package let textRows: [TextRow]
    /// The first joint row of each image block.
    package let blockStarts: [Int]
    /// The prefix as `(start, end, isText)` runs of equal `image_ids` (`_qwenimage21_prefix_segments`).
    package let segments: [Segment]
    /// `(frame, height, width)` RoPE positions per joint row (`QwenImage21Rope.forward`).
    package let positions: [[Int]]

    package var count: Int { imageIds.count }
    package var target: Int { images.last.map { $0.height * $0.width } ?? 0 }
    package var prefix: Int { count - target }

    package enum Failure: Error, CustomStringConvertible {
        case layout(String)
        package var description: String { switch self { case .layout(let s): return "Qwen-Image-2.1 sequence: \(s)" } }
    }

    package init(slots: [Bool], images: [Grid]) throws {
        guard let last = images.last else { throw Failure.layout("no image (the target is one)") }
        let imageRows = images.map { $0.height * $0.width }
        guard 4 * slots.filter({ $0 }).count == imageRows.reduce(0, +) else {
            throw Failure.layout("\(slots.filter { $0 }.count) slots for \(imageRows.reduce(0, +)) image tokens")
        }
        guard slots.suffix(last.height * last.width / 4).allSatisfy({ $0 }) else {
            throw Failure.layout("the target's slots are not the last ones")
        }
        self.images = images
        // Expand: a text position → 1 row, a slot → 4 rows.
        var isImage: [Bool] = [], textRows: [TextRow] = []
        for (position, slot) in slots.enumerated() {
            if slot { isImage += [true, true, true, true] } else {
                textRows.append(TextRow(joint: isImage.count, encoderRow: position)); isImage.append(false)
            }
        }
        self.textRows = textRows
        // Image ids by the token counts of `img_shapes`, not by runs of slots (`build_token_metadata`).
        var ids = [Int](repeating: -1, count: isImage.count)
        var block = 0, used = 0, starts: [Int] = []
        for row in isImage.indices where isImage[row] {
            if used == 0 { starts.append(row) }
            ids[row] = block
            used += 1
            if used == imageRows[block] { block += 1; used = 0 }
        }
        imageIds = ids
        blockStarts = starts
        guard starts.count == images.count, (0..<images.count).allSatisfy({ i in
            (starts[i]..<(starts[i] + imageRows[i])).allSatisfy { ids[$0] == i } }) else {
            throw Failure.layout("an image block is not contiguous")
        }
        let prefix = isImage.count - imageRows.last!
        var segments: [Segment] = []
        var start = 0
        for index in 1...max(prefix, 1) where prefix > 0 {
            if index == prefix || ids[index] != ids[start] {
                segments.append(Segment(start: start, end: index, isText: ids[start] < 0))
                start = index
            }
        }
        self.segments = segments
        // RoPE: text advances p on all three axes; an image freezes the frame at p, centres (y, x),
        // then p += max(h, w). Text after the last image (none: the target is last) would continue.
        var positions = [[Int]](repeating: [0, 0, 0], count: isImage.count)
        var cursor = 0, p = 0
        for (i, grid) in images.enumerated() {
            let blockStart = starts[i]
            for row in cursor..<blockStart { positions[row] = [p, p, p]; p += 1 }
            let (h, w) = (grid.height, grid.width)
            for y in 0..<h {
                for x in 0..<w {
                    positions[blockStart + y * w + x] = [p, y - (h - h / 2), x - (w - w / 2)]
                }
            }
            cursor = blockStart + h * w
            p += max(h, w)
        }
        for row in cursor..<isImage.count { positions[row] = [p, p, p]; p += 1 }
        self.positions = positions
    }
}

/// **Qwen-Image-2.1's RoPE table**: `[rows, 64, 2]` as `(cos, sin)` pairs for `Ops.rope`
/// (`apply_rotary_emb_qwen(use_real=False)`: complex product on adjacent pairs). The 64 pairs are
/// `[frame 8 | height 28 | width 28]`, `ωᵢ = θ^(−2i/dim)`. The reference computes the angles in fp32
/// (`torch.outer` on fp32 frequencies, `torch.polar`); here in double, rounded once — the fp64
/// witness's function. Negative positions are what the reference's table gives at negative indices.
package enum QwenImage21Rope {
    package static func write(into out: UnsafeMutablePointer<Float>, positions: [[Int]], axes: [Int], theta: Double) {
        let ω = axes.map { dims in (0..<(dims / 2)).map { i in 1 / pow(theta, Double(2 * i) / Double(dims)) } }
        let pairs = axes.reduce(0, +) / 2
        for (row, position) in positions.enumerated() {
            var j = 0
            let line = out + row * 2 * pairs
            for (axis, frequencies) in ω.enumerated() {
                for f in frequencies {
                    let angle = Double(position[axis]) * f
                    line[2 * j] = Float(cos(angle)); line[2 * j + 1] = Float(sin(angle))
                    j += 1
                }
            }
        }
    }
}
