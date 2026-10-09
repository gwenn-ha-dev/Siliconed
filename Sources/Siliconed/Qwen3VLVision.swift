import Accelerate
import Foundation
import Metal

/// **Qwen3-VL's vision tower** (`Qwen3VLVisionModel`), one image at a time, in fp32. Graph and
/// pitfalls: the header of `Qwen3VL.swift`.
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

        /// `in_channels · temporal · patch²` = 1536: one row of `pixel_values`.
        package var patchValues: Int { 3 * temporal * patch * patch }
        /// The width after the 2×2 shuffle, 4608.
        package var merged: Int { hidden * merge * merge }
    }

    /// One image's output: the merger's tokens and the deepstack ones, each `[tokens, output]`.
    package struct Features {
        package let embeddings: [Float]
        package let deepstack: [[Float]]
        package let tokens: Int
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
    private let projectedBuf, hiddenBuf, mergedOutBuf, reserveBuf: MTLBuffer

    package var recordedNames: Set<String>?
    package var cancellation: Cancellation?
    package private(set) var boundaries: [String: [Float]] = [:]

    /// - Parameter maximumPatches: the largest image's `gh · gw`; the arena is sized for it.
    package init(artifact: Artifact, config: Config, maximumPatches: Int,
                 freezeCut: Bool = EngineSettings.effective.frozenCut) throws {
        let n = maximumPatches, d = config.hidden, i = config.intermediate
        let m = config.merged, o = config.output
        let largest = [m * m, m * o, d * i, d * 3 * d, config.patchValues * d].max()!
        let widest = [m, 3 * d, i, o].max()!
        let gemm = try GEMM(freezeCut: freezeCut)
        let budget = (n * config.patchValues + 7 * n * d + 3 * n * d + n * i + (n / 4) * o
                      + n * config.headDim + 3 * widest + largest) * 4 + (32 << 20)
        let arena = try Arena(capacity: budget)
        func slot(_ name: String, _ count: Int) throws -> UnsafeMutablePointer<Float> {
            try arena.reserve(name, bytes: count * 4).assumingMemoryBound(to: Float.self)
        }
        pixels = try slot("pixels", n * config.patchValues)
        x = try slot("x", n * d)
        normed = try slot("normed", n * d)          // also the [N/4, 4608] shuffled view
        qkv = try slot("qkv", 3 * n * d)
        q = try slot("q", n * d); k = try slot("k", n * d); v = try slot("v", n * d)
        attended = try slot("attended", n * d)
        projected = try slot("projected", n * d)    // also the mergers' hidden [N/4, 4608]
        hidden = try slot("hidden", n * i)
        mergedOut = try slot("mergedOut", (n / 4) * o)
        freqs = try slot("freqs", n * config.headDim)
        bias = try slot("bias", widest); gain = try slot("gain", widest); shift = try slot("shift", widest)
        reserve = try slot("reserve", largest)
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
        reserveBuf = try gemm.wrap(reserve, bytes: largest * 4, name: "reserve")
        self.config = config
        self.artifact = artifact
        self.gemm = gemm
        self.arena = arena
        self.capacity = n
        self.prefetcher = Prefetcher(artifact: artifact)
    }

    private func record(_ name: String, _ pointer: UnsafePointer<Float>, _ count: Int) {
        guard let names = recordedNames, names.contains(name) else { return }
        boundaries[name] = Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func widen(_ name: String, count: Int, into destination: UnsafeMutablePointer<Float>) throws {
        let got = try artifact.materialize(name, into: destination, capacity: count)
        guard got == count else { throw Artifact.Failure.badHeader("\(name): \(got) values, \(count) expected") }
    }

    /// `y = x · W + b`, the weight stored `[input, output]` by the forge.
    private func linear(_ name: String, _ a: MTLBuffer, _ c: MTLBuffer, _ out: UnsafeMutablePointer<Float>,
                        rows: Int, inputs: Int, outputs: Int) throws {
        try widen(name + ".weight", count: inputs * outputs, into: reserve)
        _ = gemm.linear(a: a, b: reserveBuf, c: c, m: rows, k: inputs, n: outputs, weightIsTransposed: true)
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
                vDSP_vma(row, 1, g, 1, s, 1, row, 1, vDSP_Length(columns))
            }
        }
    }

    /// A `Qwen3VLVisionPatchMerger`: `normed` → `mergedOut` `[N/4, output]`. `postShuffle`: the
    /// deepstack ones, whose LayerNorm covers the 4 shuffled patches (4608) instead of one (1152).
    private func merger(_ prefix: String, patches n: Int, postShuffle: Bool) throws -> [Float] {
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
        return Array(UnsafeBufferPointer(start: mergedOut, count: (n / 4) * o))
    }

    package func encode(_ image: Qwen3VLImages.Pixels) throws -> Features {
        boundaries = [:]
        let (gh, gw) = (image.gridHeight, image.gridWidth)
        let n = gh * gw, d = config.hidden, dh = config.headDim, heads = config.heads
        precondition(n <= capacity, "\(n) patches for an arena of \(capacity)")
        precondition(gh % config.merge == 0 && gw % config.merge == 0, "grid \(gh)×\(gw) not divisible by the merge")
        precondition(image.values.count == n * config.patchValues, "pixel_values: \(image.values.count) values")
        image.values.withUnsafeBufferPointer { pixels.update(from: $0.baseAddress!, count: $0.count) }
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
        let quarter = dh / 4
        let inverse = (0..<quarter).map { j in 1 / powf(config.theta, Float(2 * j) / Float(dh / 2)) }
        let positions = Qwen3VLImages.patchPositions(gridHeight: gh, gridWidth: gw, merge: config.merge)
        for p in 0..<n {
            let row = freqs + p * dh
            for j in 0..<(2 * quarter) {
                let angle = Float(j < quarter ? positions[p].row : positions[p].column) * inverse[j % quarter]
                row[2 * j] = cosf(angle); row[2 * j + 1] = sinf(angle)
            }
        }

        let attention = Attention(device: gemm.device, queue: gemm.queue, heads: heads, sequence: n, headDim: dh,
                                  causal: false)
        var deepstack: [[Float]] = []
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
            Krea2Ops.geluTanh(hidden, count: n * config.intermediate)    // `gelu_pytorch_tanh`
            try linear(p + "mlp.linear_fc2", hiddenBuf, projectedBuf, projected, rows: n, inputs: config.intermediate, outputs: d)
            vDSP_vadd(x, 1, projected, 1, x, 1, vDSP_Length(n * d))
            if block == 0 { record("vit_block0_out", x, n * d) }

            if let level = config.deepstack.firstIndex(of: block) {
                let feature = try merger("visual.deepstack_merger_list.\(level).", patches: n, postShuffle: true)
                record("deepstack\(level)", feature, feature.count)
                deepstack.append(feature)
            }
        }
        record("vit_last_hidden", x, n * d)
        let embeddings = try merger("visual.merger.", patches: n, postShuffle: false)
        record("vit_out", embeddings, embeddings.count)
        return Features(embeddings: embeddings, deepstack: deepstack, tokens: n / 4)
    }
}
