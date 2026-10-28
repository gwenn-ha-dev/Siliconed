import Foundation

/// **MOSS-Image's DiT need** — the denoising phase, nothing but the constructors' shape functions. The
/// encoder's and the VAE's phases are their own files'; the family's formula composes the three.
///
///     denoising   MossImageDiT arena(S + T, r) + the stream's buffers + CacheLoRA(r) + a rotated weight's widening scratch
///                 + the SDPA (queries' copy; past `Attention.maxMatrix` the slices' concatenation)
///                 + the velocities the sampler holds · the GEMM's tiers and conductor workspace
///                 — or, before the first evaluation, the modulation tabulated for every timestep
///                 (the larger of the two moments)
///
/// What the formula has to know and a port by analogy with Krea 2 would miss:
///
///   - **the map is 10.2 GB in bf16**: it streams beside the page cache, the whole map
///     (`MossImageDiT.generationTail`), unless B holds it; the largest tensor is `ffn_up_gate`, 3072 × 16384;
///   - **the fused feed-forward is wide**: `h1` holds `[S + T, 16384]` (the value and the gate together), `h2`
///     the `[S + T, 8192]` SITU output — at 1024² and 512 text rows, 0.4 and 0.2 GB;
///   - **the weight reserve is `d · 2·ffn` floats** (`ffn_up_gate` widened, 201 MB), and under `gemm_bf16_act`
///     the rounded activation sits in its second half;
///   - **the shared modulation is tabulated for every timestep at once, in double**
///     (`tabulateModulation`): `shared_block_modulation` is 56.6 M values, 226 MB widened when the map does
///     not publish it in fp32 (it publishes bf16), alive beside the arena, before any attention;
///   - **a LoRA costs three things**: the `mid [S + T, r]` slice, the `CacheLoRA` of the 252 modules it may touch
///     (rank 128: 1.19 GB for the DMD LoRA, fp32), and, with two LoRAs or more, one `down` staged while the
///     cache fills. The DiT is the only stage that carries it.
extension MossImageDiT {
    /// The image's rows: one per cell of the latent grid.
    package static func imageTokens(gridHeight: Int, gridWidth: Int) -> Int { gridHeight * gridWidth }

    /// **The DiT's slices, in floats** — what `init` reserves, each on its page, at `n = S + T` rows: `x`,
    /// `normed`, `q`, `k`, `v`, `merged`, `o` (`d`), `h1` (`2·ffn`, the fused value|gate and the text blocks'
    /// QKV), `h2` (`ffn`), the RoPE table (`head_dim`); the six modulation vectors and the head's two; one
    /// reserve for the widest weight (`d · 2·ffn`); the latent's tokens and the final projection at the image's
    /// rows; the encoder's rows at the text's; under a LoRA, its `mid [n, r]`.
    package static func slices(config: Config, imageTokens s: Int, maxText: Int, rank: Int) -> [(name: String, count: Int)] {
        let n = s + maxText, d = config.dim, f = config.hidden
        return [("x", n * d), ("normed", n * d), ("q", n * d), ("k", n * d), ("v", n * d), ("merged", n * d), ("o", n * d),
                ("h1", n * 2 * f), ("h2", n * f), ("modulation", 8 * d), ("reserve", d * 2 * f),
                ("tokens", s * config.channels), ("projected", s * config.channels), ("textIn", maxText * config.textDim),
                ("freqs", n * config.headDim)]
            + (rank > 0 ? [("loraMid", n * rank)] : [])
    }

    /// **The arena `init` maps** for this grid, these text rows and this cumulative rank.
    package static func arenaBytes(config: Config, gridHeight: Int, gridWidth: Int, maxText: Int, rank: Int) -> Int {
        Arena.capacity(slices: slices(config: config, imageTokens: imageTokens(gridHeight: gridHeight, gridWidth: gridWidth),
                                      maxText: maxText, rank: rank).map { $0.count * 4 })
    }

    /// **Every module a LoRA may touch in the DiT** (`CacheLoRA`'s retained ones, `[k, n]`): the list the cache reads.
    package static func loraModules(config: Config) -> [(k: Int, n: Int)] {
        loraTargets(config: config).map { (k: $0.k, n: $0.n) }
    }

    /// **The modulation the context keeps** per timestep: `mod [6d]` and `head [2d]` in fp32.
    package static func tableBytes(config: Config, count: Int) -> Int { count * 8 * config.dim * 4 }

    /// **What the tabulation of `count` timesteps holds at once, in double**, at its widest — the
    /// `shared_block_modulation` call: the inputs and outputs flattened and accumulated, then the fp32 table
    /// they become; under a LoRA, that call's `down` and `up`.
    package static func tabulationBytes(config: Config, count: Int, rank: Int = 0) -> Int {
        let d = config.dim
        return count * (3 * d + 6 * d + 2 * d) * 8 + tableBytes(config: config, count: count)
            + Krea2Exact.loraBytes(k: d, n: 6 * d, rank: rank)
    }

    /// The timestep weights `modulation(timesteps:)` reads, `[input, output]`.
    static func timestepWeights(_ config: Config) -> [(name: String, inputs: Int, outputs: Int)] {
        [("time_embedding.0.weight", config.timeDim, config.dim), ("time_embedding.2.weight", config.dim, config.dim),
         ("shared_block_modulation.weight", config.dim, 6 * config.dim),
         ("video_head_modulation.weight", config.dim, 2 * config.dim)]
    }

    /// **The largest timestep weight a map makes `Krea2Exact.linears` widen**: one not published in fp32 is
    /// widened for the call (`shared_block_modulation`, 226 MB, in a bf16 map). A weight the header does not
    /// list is charged as if widened.
    package static func timestepWidening(header: [String: Any], config: Config) -> Int {
        let table = header["tensors"] as? [String: [String: Any]] ?? [:]
        return timestepWeights(config).map { weight -> Int in
            guard let entry = table[weight.name], entry["dtype"] as? String == Artifact.DType.float32.rawValue else {
                let shape = (table[weight.name]?["shape"] as? [Int]).flatMap { $0.count == 2 ? $0 : nil }
                return Krea2Exact.widenedBytes(inputs: shape?[0] ?? weight.inputs, outputs: shape?[1] ?? weight.outputs)
            }
            return 0
        }.max() ?? 0
    }
}

extension MemoryNeed {
    /// **MOSS-Image's denoising phase**, on a DiT map's header: its terms, one allocation each.
    ///
    /// - Parameters:
    ///   - request: the render's shape (`width`, `height`, `loraRanks`, `tail`, `threads`, `solver`, `neuralEngine`).
    ///   - textRows: the DiT's text rows (`T`), the encoder's whole sequence.
    ///   - header: the DiT map's header (config and tensor table; never a tensor).
    ///   - fileSize: the map's size, for the staging runs.
    ///   - velocities: the latent-sized vectors the sampler keeps beside the one the DiT returns (a guided step
    ///     holds the conditional prediction while it computes the other: 1; the turbo recipe, 0).
    package static func mossImageDenoising(_ request: Request, textRows t: Int, header: [String: Any], fileSize: Int = .max,
                                           velocities: Int = 1) -> Phase? {
        let config = MossImageDiT.Config(header: header)
        let unit = Artifact.largestRun(header: header, prefixes: (0..<config.layers).map(MossImageDiT.layerPrefix), fileSize: fileSize)
        guard unit > 0 else { return nil }
        let f = MemoryLayout<Float>.size
        let (gridHeight, gridWidth) = (request.height / 16, request.width / 16)
        let s = MossImageDiT.imageTokens(gridHeight: gridHeight, gridWidth: gridWidth), n = s + t
        let latent = config.channels * s * f
        let rank = request.loraRanks.reduce(0, +)
        let (d, hidden) = (config.dim, config.hidden)
        let modules = MossImageDiT.loraModules(config: config)
        let streamed = request.tail.fraction > 0 && request.tail.streamed
        var terms = [
            Term("DiT arena", MossImageDiT.arenaBytes(config: config, gridHeight: gridHeight, gridWidth: gridWidth,
                                                     maxText: t, rank: rank)),
            Term("map staging", streamed ? TailStream.unitSlots * unit : 0),
            Term("LoRA cache", rank > 0 ? CacheLoRA.arenaBytes(CacheLoRA.bytes(modules: modules.map { ($0.k, $0.n, rank) })) : 0),
            Term("LoRA down staging", request.loraRanks.count > 1
                ? LoRA.downBytes(k: modules.map(\.k).max() ?? 0, rank: request.loraRanks.max() ?? 0) : 0),
            // The velocity `forward` returns (`[128, h, w]`) and the sampler's own (`velocities`, history).
            Term("sampler", (1 + velocities) * latent + request.solver.historyBytes(count: latent / f)),
            // Widening the largest rotated block weight (the Compact's int8 convrot): beside the arena; 0 in bf16.
            Term("DiT rotation scratch", Widen.rotationScratchBytes(header: header, prefixes: (0..<config.layers).map(MossImageDiT.layerPrefix))),
        ]
        let split = MossImageDiT.headSplit(sequence: n, heads: config.heads, headDim: config.headDim)
        terms += attention("DiT", heads: config.heads, rows: n, keys: n, headDim: config.headDim, causal: false,
                           split: split, threads: request.threads)
        // The text blocks' SDPA runs at T rows, after the main one is not yet built: the larger of the two is
        // `n`'s, which this one never exceeds.
        terms += gemm("DiT", products: [(s, config.channels, d), (s, d, config.channels)],
                      bounded: [(t, config.textDim, d), (t, d, 3 * d), (n, d, d), (n, d, 2 * hidden), (n, hidden, d)],
                      rank: rank, ane: request.neuralEngine)
        let tabulating = MossImageDiT.tabulationBytes(config: config, count: Siliconed.Request.maximumSteps, rank: rank)
            + MossImageDiT.timestepWidening(header: header, config: config)
            + terms.filter { $0.name.hasPrefix("LoRA") }.reduce(0) { $0 + $1.bytes }
        terms.append(Term("modulation tabulation beyond the evaluation",
                          max(0, tabulating - terms.reduce(0) { $0 + $1.bytes })))
        return Phase(kind: .denoising, terms: terms)
    }
}
