import Accelerate
import Foundation
import Metal

/// **Qwen3-VL's vision tower** (`Qwen3VLVisionModel`), one image at a time, in fp32. Graph and
/// pitfalls: the header of `Qwen3VL.swift`. Only `Qwen3VLEncoder` runs it, for Qwen-Image-2.1's
/// condition images; its output replaces the `<|image_pad|>` rows of the language model's input
/// (the merger) and is added to the output of that model's first three layers (the deepstack).
///
///     pixel_values [N, 1536] ─ patch_embed (Linear) ─ + Σ₄ w·pos_embed[tap] ─ x [N, 1152]
///       27 blocks ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
///         after blocks 8, 16, 24 (output of the block): x ─ deepstack merger k ─ [N/4, 4096]   │
///       final x ─ merger (LayerNorm 1152 first) ─ [N/4, 4096] ──────────────────────────────────┘
///
/// **A merger** — `Qwen3VLVisionPatchMerger`; the 4 patches of a 2×2 block are consecutive rows, so
/// the shuffle is a `view`:
///
///     final:      y = W₂ · GELU( W₁ · view₄( LN₁₁₅₂(x) ) )              W₁: 4608 → 4608, W₂: 4608 → 4096
///     deepstack:  y = W₂ · GELU( W₁ · LN₄₆₀₈( view₄(x) ) )
///
/// What a port by analogy would miss, read in `modeling_qwen3_vl.py` (transformers 5.19) and
/// `vision_utils.py`:
///
///   - **the deepstack mergers read the OUTPUT of blocks 8, 16, 24 and leave `x` untouched**: the tower
///     goes on from the same state, and the final merger reads block 26's output
///     (`Qwen3VLVisionModel.forward`; `encode`, the `deepstack.firstIndex` branch);
///   - **the QKV projection is `reshape(N, 3, heads, 72)`**: each row is `[q | k | v]`, each of them
///     heads-major (`Qwen3VLVisionAttention.forward`; `encode`, the `packed` split);
///   - **the attention is bidirectional and scaled by `72^-0.5`**, over one image's `N` patches
///     (`cu_seqlens` cuts per image; `is_causal = False`; `encode` runs `Attention(causal: false)`);
///   - **the RoPE is `cat(h, w, h, w)` over the head, rotated by `rotate_half`** — angle for dim `d` and
///     `d + 36` equal — from `inv_freq` over `head_dim / 2 = 36` (`compute_axial_rope_parameters`,
///     `recomposition_frequencies`); the table is `writeRope`, the forge's q/k permutation makes the
///     interleaved `Ops.rope` equal to `rotate_half`;
///   - **the row/column of a patch follow the 2×2 merge-block order** of `pixel_values`, not raster
///     (`get_vision_position_ids`; `Qwen3VLImages.patchPositions`), and so do the four interpolation taps
///     of its learned position (`get_vision_interpolation_indices_and_weights`; `Qwen3VLImages.positionTaps`);
///   - **the position table is resampled by `src = i · (side − 1) / max(size − 1, 1)`** (`align_corners`),
///     weights `max(1 − |src − tap|, 0)`, border-clamped taps, the four weighted rows SUMMED then added
///     to the patch embedding (`_interpolation_axis_taps_weights`; `encode`, the `rowsOfTable` loop);
///   - **every LayerNorm is eps 1e-6 with a bias, every Linear has one** (`Qwen3VLVisionBlock`,
///     `Qwen3VLVisionMLP`: `bias=True`) — unlike the language model, which has none (`linear`, `layerNorm`);
///   - **the activation is `gelu_pytorch_tanh` in the blocks and the exact `nn.GELU()` in the mergers**
///     (`Qwen3VLVisionMLP.act_fn`; `Qwen3VLVisionPatchMerger.act_fn`): `Config.init` refuses a
///     tower that declares another, `merger` uses `AnimaOps.gelu`.
///
/// The weights are read from the encoder's map (`visual.*`, 1.15 GB of bf16 before the language
/// model) block by block into one reserve, like the language model's — the tower is a tenth of
/// the map and is never resident.
package final class Qwen3VLVision {
    package struct Config {
        package let hidden, heads, headDim, intermediate, depth: Int
        package let patch, temporal, merge, gridSide, output: Int
        package let deepstack: [Int]
        /// `rope_theta` of the vision RoPE: not in the published config (`rope_parameters: null`),
        /// `transformers` standardizes it to 10⁴.
        package let theta: Float = 10_000
        package let eps: Float = 1e-6

        package init(header: [String: Any]) throws {
            guard let v = header["vision_config"] as? [String: Any],
                  let hidden = v["hidden_size"] as? Int, let heads = v["num_heads"] as? Int,
                  let intermediate = v["intermediate_size"] as? Int, let depth = v["depth"] as? Int,
                  let patch = v["patch_size"] as? Int, let temporal = v["temporal_patch_size"] as? Int,
                  let merge = v["spatial_merge_size"] as? Int, let positions = v["num_position_embeddings"] as? Int,
                  let output = v["out_hidden_size"] as? Int, let deepstack = v["deepstack_visual_indexes"] as? [Int] else {
                throw Artifact.Failure.badHeader("`vision_config` missing or incomplete: not a Qwen3-VL encoder map")
            }
            guard v["hidden_act"] as? String == "gelu_pytorch_tanh", hidden % heads == 0,
                  (hidden / heads) % 4 == 0, header["vision_patch_embed_as_linear"] as? String == "(c, t, y, x)" else {
                throw Artifact.Failure.badHeader("vision tower: activation, head width or patch layout not ported")
            }
            self.hidden = hidden; self.heads = heads; self.headDim = hidden / heads
            self.intermediate = intermediate; self.depth = depth
            self.patch = patch; self.temporal = temporal; self.merge = merge
            self.gridSide = Int(Double(positions).squareRoot())
            guard gridSide * gridSide == positions else { throw Artifact.Failure.badHeader("\(positions) positions: not a square") }
            self.output = output; self.deepstack = deepstack
        }

        /// **Qwen3-VL-8B's tower as published** (`vision_config` of the encoder's map): what the memory
        /// formula sizes without opening the map.
        package static let published = try! Config(header: [
            "vision_config": ["hidden_size": 1152, "num_heads": 16, "intermediate_size": 4304, "depth": 27,
                              "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
                              "num_position_embeddings": 2304, "out_hidden_size": 4096,
                              "deepstack_visual_indexes": [8, 16, 24], "hidden_act": "gelu_pytorch_tanh"] as [String: Any],
            "vision_patch_embed_as_linear": "(c, t, y, x)"])

        /// `in_channels · temporal · patch²` = 1536: one row of `pixel_values`.
        package var patchValues: Int { 3 * temporal * patch * patch }
        /// The width after the 2×2 shuffle, 4608.
        package var merged: Int { hidden * merge * merge }
    }

    /// One image's output: the merger's tokens and the deepstack ones, each `[tokens, output]`, in
    /// one region of `featureBytes` — they live until the language model has read them, then
    /// leave with their `Pages` instead of staying in `malloc`'s freed blocks (3 × 64 MB at 1024²).
    package struct Features {
        package let tokens: Int
        package let output: Int
        package let levels: Int
        private let storage: Pages

        init(tokens: Int, output: Int, levels: Int) {
            self.tokens = tokens; self.output = output; self.levels = levels
            storage = Pages(bytes: Qwen3VLVision.featureBytes(tokens: tokens, output: output, levels: levels))
        }

        /// The merger's tokens, `[tokens, output]`.
        package var embeddings: UnsafeMutablePointer<Float> { storage.floats() }
        /// The deepstack tokens of `level`, `[tokens, output]`.
        package func deepstack(_ level: Int) -> UnsafeMutablePointer<Float> {
            storage.floats(at: (1 + level) * tokens * output * 4)
        }
    }

    /// **What one image's `Features` holds**: the merger's tokens and `levels` deepstack ones, each
    /// `[tokens, output]` fp32, in one region rounded to the page. `tokens` = patches / 4.
    package static func featureBytes(tokens: Int, output: Int = Config.published.output,
                                     levels: Int = Config.published.deepstack.count) -> Int {
        Arena.capacity(for: (1 + levels) * tokens * output * 4)
    }

    package let config: Config
    private let artifact: Artifact
    private let gemm: GEMM
    private let arena: Arena
    private let prefetcher: Prefetcher
    private let capacity: Int

    private let pixels, x, normed, qkv, q, k, v, attended, projected, hidden: UnsafeMutablePointer<Float>
    private let mergedOut, freqs, bias, gain, shift, reserve: UnsafeMutablePointer<Float>
    private let pixelsBuf, xBuf, normedBuf, qkvBuf, qBuf, kBuf, vBuf, attendedBuf: MTLBuffer
    private let projectedBuf, hiddenBuf, mergedOutBuf: MTLBuffer
    /// The products by a weight (`WeightProducts`): the path the settings chose, as in the DiTs.
    private let products: WeightProducts

    package var recordedNames: Set<String>?
    package var cancellation: Cancellation?
    package private(set) var boundaries: [String: [Float]] = [:]

    /// - Parameter maximumPatches: the largest image's `gh · gw`; the arena is sized for it.
    package init(artifact: Artifact, config: Config, maximumPatches: Int) throws {
        let n = maximumPatches, d = config.hidden, i = config.intermediate
        let o = config.output
        let largest = Qwen3VLVision.reserves(config).largest
        // The Neural Engine serves the denoisers only: the encoders' output enters the render cache's keys.
        let gemm = try GEMM(phase: .encoding)
        let arena = try Arena(capacity: Qwen3VLVision.arenaBytes(config: config, patches: n))
        var slices: [String: UnsafeMutablePointer<Float>] = [:]
        for (name, count) in Qwen3VLVision.slices(config: config, patches: n) {
            slices[name] = try arena.reserve(name, bytes: count * 4).assumingMemoryBound(to: Float.self)
        }
        pixels = slices["pixels"]!; x = slices["x"]!
        normed = slices["normed"]!                  // also the [N/4, 4608] shuffled view
        qkv = slices["qkv"]!; q = slices["q"]!; k = slices["k"]!; v = slices["v"]!
        attended = slices["attended"]!
        projected = slices["projected"]!            // also the mergers' hidden [N/4, 4608]
        hidden = slices["hidden"]!; mergedOut = slices["mergedOut"]!; freqs = slices["freqs"]!
        bias = slices["bias"]!; gain = slices["gain"]!; shift = slices["shift"]!
        reserve = slices["reserve"]!
        pixelsBuf = try gemm.wrap(pixels, bytes: n * config.patchValues * 4, name: "pixels")
        xBuf = try gemm.wrap(x, bytes: n * d * 4, name: "x")
        normedBuf = try gemm.wrap(normed, bytes: n * d * 4, name: "normed")
        qkvBuf = try gemm.wrap(qkv, bytes: 3 * n * d * 4, name: "qkv")
        qBuf = try gemm.wrap(q, bytes: n * d * 4, name: "q")
        kBuf = try gemm.wrap(k, bytes: n * d * 4, name: "k")
        vBuf = try gemm.wrap(v, bytes: n * d * 4, name: "v")
        attendedBuf = try gemm.wrap(attended, bytes: n * d * 4, name: "attended")
        projectedBuf = try gemm.wrap(projected, bytes: n * d * 4, name: "projected")
        hiddenBuf = try gemm.wrap(hidden, bytes: n * i * 4, name: "hidden")
        mergedOutBuf = try gemm.wrap(mergedOut, bytes: (n / 4) * o * 4, name: "mergedOut")
        products = WeightProducts(gemm: gemm, artifact: artifact, reserve: reserve, capacity: largest, transposed: true)
        self.config = config
        self.artifact = artifact
        self.gemm = gemm
        self.arena = arena
        self.capacity = n
        self.prefetcher = Prefetcher(artifact: artifact)
    }

    /// The floats of the widest weight (`reserve`) and of the widest bias row (`bias`, `gain`, `shift`).
    private static func reserves(_ config: Config) -> (largest: Int, widest: Int) {
        let d = config.hidden, i = config.intermediate, m = config.merged, o = config.output
        return ([m * m, m * o, d * i, d * 3 * d, config.patchValues * d].max()!, [m, 3 * d, i, o].max()!)
    }

    /// **The bytes of the tower's arena for images of at most `patches` patches** — one source: `init`
    /// allocates it, `MemoryNeed.qwenImage21` charges it. `pixels`, the nine `[N, d]` activations
    /// (`x`, `normed`, `qkv`·3, `q`, `k`, `v`, `attended`, `projected` — counted as 7 + 3), the MLP's
    /// hidden, the merger's output, the RoPE table, three bias rows and the widest weight: its
    /// `slices`, each on its page — formerly with 32 MB of margin besides, never reserved.
    package static func arenaBytes(config: Config, patches n: Int) -> Int {
        Arena.capacity(slices: slices(config: config, patches: n).map { $0.count * 4 })
    }

    /// **The tower's slices, in floats, in the order `init` reserves them.**
    package static func slices(config: Config, patches n: Int) -> [(name: String, count: Int)] {
        let d = config.hidden, i = config.intermediate, o = config.output
        let (largest, widest) = reserves(config)
        return [("pixels", n * config.patchValues), ("x", n * d), ("normed", n * d), ("qkv", 3 * n * d),
                ("q", n * d), ("k", n * d), ("v", n * d), ("attended", n * d), ("projected", n * d),
                ("hidden", n * i), ("mergedOut", (n / 4) * o), ("freqs", n * config.headDim),
                ("bias", widest), ("gain", widest), ("shift", widest), ("reserve", largest)]
    }

    private func record(_ name: String, _ pointer: UnsafePointer<Float>, _ count: Int) {
        guard let names = recordedNames, names.contains(name) else { return }
        boundaries[name] = Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func widen(_ name: String, count: Int, into destination: UnsafeMutablePointer<Float>) throws {
        try artifact.materializeExactly(name, count: count, into: destination)
    }

    /// `y = x · W + b`, the weight stored `[input, output]` by the forge.
    private func linear(_ name: String, _ a: MTLBuffer, _ c: MTLBuffer, _ out: UnsafeMutablePointer<Float>,
                        rows: Int, inputs: Int, outputs: Int) throws {
        try products.linear(name + ".weight", a: a, c: c, m: rows, k: inputs, n: outputs)
        try widen(name + ".bias", count: outputs, into: bias)
        let b = bias
        Parallel.rows(rows, width: outputs) { first, count in
            for r in first..<(first + count) {
                let row = out + r * outputs
                vDSP_vadd(row, 1, b, 1, row, 1, vDSP_Length(outputs))
            }
        }
    }

    /// `nn.LayerNorm(columns, eps=1e-6)` with its weight and bias.
    private func layerNorm(_ name: String, _ source: UnsafePointer<Float>, into out: UnsafeMutablePointer<Float>,
                           rows: Int, columns: Int) throws {
        Ops.layerNorm(source, into: out, rows: rows, columns: columns, eps: config.eps)
        try widen(name + ".weight", count: columns, into: gain)
        try widen(name + ".bias", count: columns, into: shift)
        let (g, s) = (gain, shift)
        Parallel.rows(rows, width: columns) { first, count in
            for r in first..<(first + count) {
                let row = out + r * columns
                // torch's affine LayerNorm is its plain one times the weight plus the bias, two roundings.
                Ops.multiplyAdd(row, times: g, plus: s, count: columns)
            }
        }
    }

    /// A `Qwen3VLVisionPatchMerger`: `normed` → `mergedOut` `[N/4, output]`. `postShuffle`: the
    /// deepstack ones, whose LayerNorm covers the 4 shuffled patches (4608) instead of one (1152).
    private func merger(_ prefix: String, patches n: Int, postShuffle: Bool,
                        into destination: UnsafeMutablePointer<Float>) throws {
        let (d, m, o) = (config.hidden, config.merged, config.output)
        if postShuffle {
            try layerNorm(prefix + "norm", x, into: normed, rows: n / 4, columns: m)
        } else {
            try layerNorm(prefix + "norm", x, into: normed, rows: n, columns: d)
        }
        // `view(-1, 4608)`: the four patches of a 2×2 block are consecutive rows — no copy.
        try linear(prefix + "linear_fc1", normedBuf, projectedBuf, projected, rows: n / 4, inputs: m, outputs: m)
        AnimaOps.gelu(projected, count: (n / 4) * m)                    // EXACT GELU (`nn.GELU()`)
        try linear(prefix + "linear_fc2", projectedBuf, mergedOutBuf, mergedOut, rows: n / 4, inputs: m, outputs: o)
        destination.update(from: mergedOut, count: (n / 4) * o)
    }

    package func encode(_ image: Qwen3VLImages.Pixels) throws -> Features {
        boundaries = [:]
        let (gh, gw) = (image.gridHeight, image.gridWidth)
        let n = gh * gw, d = config.hidden, dh = config.headDim, heads = config.heads
        precondition(n <= capacity, "\(n) patches for an arena of \(capacity)")
        precondition(gh % config.merge == 0 && gw % config.merge == 0, "grid \(gh)×\(gw) not divisible by the merge")
        precondition(image.values.count == n * config.patchValues, "pixel_values: \(image.values.count) values")
        precondition(config.patchValues == Qwen3VLImages.Pixels.patchValues, "a tower of \(config.patchValues) values a patch")
        pixels.update(from: image.values.baseAddress!, count: image.values.count)
        prefetcher.request(prefetcher.namesOfBlock(prefix: "visual.blocks.0."))

        // ── patch_embed + interpolated learned positions ──────────────────────────────────────
        try linear("visual.patch_embed.proj", pixelsBuf, xBuf, x, rows: n, inputs: config.patchValues, outputs: d)
        guard artifact.tensors["visual.pos_embed.weight"]?.shape == [config.gridSide * config.gridSide, d] else {
            throw Artifact.Failure.badHeader("visual.pos_embed.weight: missing or not a [\(config.gridSide)², \(d)] table")
        }
        let taps = Qwen3VLImages.positionTaps(gridHeight: gh, gridWidth: gw, side: config.gridSide, merge: config.merge)
        let rowsOfTable = UnsafeMutablePointer<Float>.allocate(capacity: 4 * d)
        defer { rowsOfTable.deallocate() }
        for p in 0..<n {
            for t in 0..<4 {
                try artifact.materializeRows("visual.pos_embed.weight", first: taps.indices[4 * p + t], count: 1,
                                             into: rowsOfTable + t * d)
            }
            // `(pos_embed(idx) * w[:, :, None]).sum(1)`, then added to the patch embedding.
            let row = x + p * d
            for j in 0..<d {
                var sum = rowsOfTable[j] * taps.weights[4 * p]
                for t in 1..<4 { sum += rowsOfTable[t * d + j] * taps.weights[4 * p + t] }
                row[j] += sum
            }
        }
        record("vit_embed_out", x, n * d)

        // ── 2D RoPE: per patch, 18 frequencies on the row, 18 on the column ───────────────────
        let positions = Qwen3VLImages.patchPositions(gridHeight: gh, gridWidth: gw, merge: config.merge)
        Self.writeRope(into: freqs, positions: positions.map { ($0.row, $0.column) }, headDim: dh, theta: config.theta)

        let attention = Attention(device: gemm.device, queue: gemm.queue, heads: heads, sequence: n, headDim: dh,
                                  causal: false)
        let features = Features(tokens: n / 4, output: config.output, levels: config.deepstack.count)
        for block in 0..<config.depth {
            try cancellation.check()
            let p = "visual.blocks.\(block)."
            prefetcher.request(prefetcher.namesOfBlock(prefix: "visual.blocks.\(block + 1)."))
            if config.deepstack.contains(block) || block + 1 == config.depth {
                prefetcher.request(prefetcher.namesOfBlock(prefix: "visual.deepstack_merger_list."))
                prefetcher.request(prefetcher.namesOfBlock(prefix: "visual.merger."))
            }

            try layerNorm(p + "norm1", x, into: normed, rows: n, columns: d)
            try linear(p + "attn.qkv", normedBuf, qkvBuf, qkv, rows: n, inputs: d, outputs: 3 * d)
            // `reshape(N, 3, heads, 72)`: each row is [q | k | v].
            let (qq, kk, vv, packed) = (q, k, v, qkv)
            Parallel.rows(n, width: 3 * d) { first, count in
                for r in first..<(first + count) {
                    let row = packed + r * 3 * d
                    (qq + r * d).update(from: row, count: d)
                    (kk + r * d).update(from: row + d, count: d)
                    (vv + r * d).update(from: row + 2 * d, count: d)
                }
            }
            Ops.rope(q, freqs: freqs, sequence: n, heads: heads, headDim: dh)
            Ops.rope(k, freqs: freqs, sequence: n, heads: heads, headDim: dh)
            _ = attention.run(q: qBuf, k: kBuf, v: vBuf, into: attendedBuf)
            try linear(p + "attn.proj", attendedBuf, projectedBuf, projected, rows: n, inputs: d, outputs: d)
            vDSP_vadd(x, 1, projected, 1, x, 1, vDSP_Length(n * d))

            try layerNorm(p + "norm2", x, into: normed, rows: n, columns: d)
            try linear(p + "mlp.linear_fc1", normedBuf, hiddenBuf, hidden, rows: n, inputs: d, outputs: config.intermediate)
            Ops.geluTanh(hidden, count: n * config.intermediate)    // `gelu_pytorch_tanh`
            try linear(p + "mlp.linear_fc2", hiddenBuf, projectedBuf, projected, rows: n, inputs: config.intermediate, outputs: d)
            vDSP_vadd(x, 1, projected, 1, x, 1, vDSP_Length(n * d))
            if block == 0 { record("vit_block0_out", x, n * d) }

            if let level = config.deepstack.firstIndex(of: block) {
                try merger("visual.deepstack_merger_list.\(level).", patches: n, postShuffle: true,
                           into: features.deepstack(level))
                record("deepstack\(level)", features.deepstack(level), features.tokens * features.output)
            }
        }
        record("vit_last_hidden", x, n * d)
        try merger("visual.merger.", patches: n, postShuffle: false, into: features.embeddings)
        record("vit_out", features.embeddings, features.tokens * features.output)
        return features
    }
}

extension Qwen3VLVision {
    /// **`Qwen3VLVisionRotaryEmbedding`'s table**, `[patches, Dh/2, (cos, sin)]`: `inv_freq =
    /// 1/θ^(2j/(Dh/2))` for `Dh/4` frequencies in float32, `position.float() * inv_freq` in float32, the
    /// row's angles then the column's (`recomposition_frequencies`), `.cos()` / `.sin()` — with the one-bit
    /// residue of torch's SLEEF (`Ops.polar`).
    package static func writeRope(into out: UnsafeMutablePointer<Float>, positions: [(Int, Int)], headDim dh: Int,
                                  theta: Float) {
        let quarter = dh / 4
        let inverse = (0..<quarter).map { j in 1 / powf(theta, Float(2 * j) / Float(dh / 2)) }
        for (p, position) in positions.enumerated() {
            let row = out + p * dh
            for j in 0..<(2 * quarter) {
                let angle = Float(j < quarter ? position.0 : position.1) * inverse[j % quarter]
                row[2 * j] = cosf(angle); row[2 * j + 1] = sinf(angle)
            }
        }
    }
}
