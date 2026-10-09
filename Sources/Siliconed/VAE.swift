import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import Foundation

/// The Flux VAE decoder, in fp32, via `MPSGraph` — so without a line of MSL.
///
///     latent [16, h, w] ─ ÷0.3611 then +0.1159 ─ conv_in ─ mid(resnet, attention, resnet)
///       ─ up×4 (3 resnets, ×2 except the last) ─ GroupNorm ─ SiLU ─ conv_out ─ image [3, 8h, 8w]
///
/// The weights are published in **bf16**: reading them as fp32 would give finite and wrong numbers.
/// The VAE weighs 84 M parameters, so it has no forge: we read the published file directly.
///
/// **Z-Image's decoder, and FLUX.2's** (klein, ERNIE: `Config.flux2`, which changes only the head).
///
/// ## Memory: the head in one graph, the rest by layers and bands — the same bits
///
/// In one graph the decoder held 5.1 GB at 1024² and **7.2 GB at 1024×1536, where Z-Image swapped**
/// (27,632 pages): `MPSGraph` keeps every intermediate of a graph until it ends, at full
/// resolution. Past the head, it now runs layer by layer in our own buffers, each layer by
/// horizontal bands (`decodeBanded`): **2.1 GB at 1024², 2.9 GB at 1024×1536 — the bits of the one
/// graph**, at every band size (checked against the one graph). Two things make it exact where a
/// tiling is not: each `GroupNorm` takes its statistics over the whole tensor, in a graph of their own,
/// and every band window starts on an even row (`alignment`).
package final class VAE {
    package struct Config {
        package let scaling: Float, shift: Float
        package let groups: Int, latentChannels: Int
        /// A 1×1 `post_quant_conv` before `conv_in` — that of the FLUX.2 VAE (Flux 1 has none).
        package let postQuantConv: Bool
        package init(scaling: Float = 0.3611, shift: Float = 0.1159,
                    groups: Int = 32, latentChannels: Int = 16, postQuantConv: Bool = false) {
            self.scaling = scaling; self.shift = shift
            self.groups = groups; self.latentChannels = latentChannels
            self.postQuantConv = postQuantConv
        }

        /// **The FLUX.2 VAE**: the 32-channel Flux decoder, preceded by a `post_quant_conv`.
        /// Neither scale nor shift: the pipeline's denormalization (the `BatchNorm` on the
        /// unpacked latent) is done beforehand, by `Flux2DecodingModule`.
        package static let flux2 = Config(scaling: 1, shift: 0, latentChannels: 32, postQuantConv: true)
    }

    private typealias Register = QwenImage21VAE.Register

    private let weights: Safetensors
    private let config: Config
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let latentHeight, latentWidth: Int
    /// `nil` when decoding in one go. Otherwise a tile's grid and its overlap, in latent.
    /// A tile is square, except on an axis shorter than it: it takes the whole axis there.
    private let tile: (height: Int, width: Int, overlap: Int)?
    package var imageHeight: Int { latentHeight * 8 }
    package var imageWidth: Int { latentWidth * 8 }

    /// The elements of one window, all channels, at most (8 Mi floats = 32 MB): a band's height is
    /// chosen so that its widest tensor — input, output or residual — stays under it. At 1024², 2²³
    /// and 2²⁴ cost the same time (4.4 s) and 2²³ 140 MB less; 2²⁰ costs 7.3 s. Settable by the
    /// checks, to force bands where one piece would do, or one piece where bands would.
    nonisolated(unsafe) package static var bandElements = 1 << 23
    /// This decoding's band: `bandElements`, or less when the budget read after the denoising does not
    /// cover the render's floor (`MemoryPlan.decoderBand`). Set as `decodeBanded` starts.
    private var band = VAE.bandElements

    /// Called after the head and after each banded layer with its name — `vae-bands` reads the footprint
    /// there. Set by a single-threaded check before decoding, never by the product.
    nonisolated(unsafe) package static var layerDone: ((String) -> Void)?

    /// The channels of the four up blocks, from the bottleneck out (`block_out_channels` reversed).
    private static let levels = [512, 512, 256, 128]

    /// `latentHeight`, `latentWidth`: the grid of the `[16, h, w]` latent.
    package init(path: String, latentHeight: Int, latentWidth: Int, config: Config = Config()) throws {
        let weights = try Safetensors(path: path)
        self.weights = weights
        self.config = config
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw GEMM.Failure.noDevice
        }
        self.device = device
        self.queue = queue
        guard weights.entries["decoder.conv_in.weight"]?.shape.dropFirst().first == config.latentChannels,
              weights.entries["decoder.conv_out.weight"]?.shape.first == 3 else {
            throw Safetensors.Failure.badHeader("\(path): not a Flux VAE with \(config.latentChannels) latent channels")
        }
        // **Tiles only apply if they cut something.** A tile larger
        // than the image is not a tile, and a tile without overlap leaves a hard
        // seam: both cases are refused here rather than rendering an inexplicable image.
        let requestedTile = EngineSettings.effective.vaeTile, overlap = max(0, EngineSettings.effective.vaeOverlap)
        if requestedTile >= 16, requestedTile < max(latentHeight, latentWidth), overlap > 0,
           requestedTile > 2 * overlap {
            self.tile = (min(requestedTile, latentHeight), min(requestedTile, latentWidth), overlap)
        } else {
            if requestedTile > 0 {
                Warnings.emit("⚠ tile of \(requestedTile) refused for a latent of \(latentWidth)×\(latentHeight) "
                                       + "(overlap \(overlap)) — decoding in one piece")
            }
            self.tile = nil
        }
    }

    /// **The head: everything at the latent's resolution, in one graph** — the denormalization,
    /// `post_quant_conv` (FLUX.2), `conv_in`, the middle block and its attention, the first up block's
    /// three resnets. Its tensors are 512 channels at `h × w` (48 MB at 1024×1536): the only large
    /// term is the bottleneck attention, which `vae_bloc_requetes` slices.
    private func head(_ b: VAEBuildingBlocks, _ input: MPSGraphTensor, height H: Int, width W: Int) throws -> MPSGraphTensor {
        let graph = b.graph, config = self.config
        var x = input
        // The pipeline denormalizes BEFORE the decoder: `latents / scaling + shift`.
        if config.scaling != 1 || config.shift != 0 {
            x = graph.addition(
                graph.division(input, graph.constant(Double(config.scaling), dataType: .float32), name: nil),
                graph.constant(Double(config.shift), dataType: .float32), name: "denormalize")
        }
        if config.postQuantConv { x = try b.conv(x, "post_quant_conv", padding: 0) }

        /// The attention of the middle block, over `H·W` tokens of 512 channels.
        func attention(_ x: MPSGraphTensor, _ prefix: String, channels: Int, size: (h: Int, w: Int)) throws -> MPSGraphTensor {
            // **The attention matrix gets materialized anyway, and it is 2.52 GB**.
            //
            // A chain `matmul → ×scale → softMax → matmul` holds three full `[1, S, S]` tensors: at
            // 1024² the latent is 128², so `S = 128² = 16,384` and **each weighs 1.073 GB**.
            // `scaledDotProductAttention` is 25 % faster at strictly identical PSNR, but it does not
            // remove the matrix: it holds it on the API's side. It was measured with a probe that
            // removed the bottleneck and nothing else (it rendered a wrong image, and is gone):
            //
            //     1024²   with 7,107 MB   ·   without 4,587 MB   →   the bottleneck weighs 2,520 MB
            //      512²   with 1,356 MB   ·   without 1,375 MB   →   nothing, the matrix is 67 MB
            //
            // 2.52 GB for a 1.073 GB matrix: the API therefore holds two or three copies, not
            // zero. That is half of the decoder's footprint problem; the other half, 4.28 GB,
            // is **proportional to pixels** — the full-resolution intermediate tensors
            // that `MPSGraph` keeps alive (verified: 1,071 MB at 512², exactly ×4).
            //
            // It had already been established for the DiT: `MPSGraph.scaledDotProductAttention`
            // **materializes** — 6.1 GB of traffic for 75 ms of compute. The VAE does not escape it.
            //
            // **`SILICONED_VAE_BLOCK=n` — the `S×S` matrix in slices of `n` queries.**
            //
            // What this buys is **room**, not time: materialization goes from
            // `S×S` to `n×S`, i.e. 2,520 MB → ~160 MB at `n = 1024` and `S = 16,384` (the probe's
            // starting figure). The split had been measured on the DiT's SDPA and
            // rejected it **for time** (×0.73 to ×0.80); here the decoder only costs 4.4 s out of a
            // 140 s render, so the trade is good as soon as room runs short.
            //
            // **Correctness is not up for discussion**: splitting the queries reassociates nothing — each
            // output row depends only on its own row of scores, and the softmax is already
            // per row. A measurement verified it and found only summation-order noise
            // (worst channel 9.4·10⁻⁸). It is not an approximation, it is the same sum.
            //
            // The default stays **0**, the whole matrix: on this machine, at 1024², the peak already
            // fits in 16 GB. A smaller machine — or a larger resolution, where the
            // term is in pixels² — will set it to 1024 or 2048. It is a per-target setting.
            let normalized = try b.groupNorm(x, prefix + ".group_norm", channels: channels, h: size.h, w: size.w)
            return try b.attention(x, normalized: normalized, channels: channels, size: size,
                                   qkv: { (try b.projection($0, prefix + ".to_q"), try b.projection($0, prefix + ".to_k"),
                                           try b.projection($0, prefix + ".to_v")) },
                                   output: prefix + ".to_out.0", requestBlock: EngineSettings.effective.vaeRequestBlock)
        }

        let L = (h: H, w: W)
        x = try b.conv(x, "decoder.conv_in", padding: 1)
        x = try b.resnetFlux(x, "decoder.mid_block.resnets.0", inChannels: 512, outChannels: 512, size: L)
        x = try attention(x, "decoder.mid_block.attentions.0", channels: 512, size: L)
        x = try b.resnetFlux(x, "decoder.mid_block.resnets.1", inChannels: 512, outChannels: 512, size: L)
        for r in 0..<3 {
            x = try b.resnetFlux(x, "decoder.up_blocks.0.resnets.\(r)", inChannels: 512, outChannels: 512, size: L)
        }
        return x
    }

    /// **The whole decoder on a `[C, h, w]` latent, exactly — the head in one graph, then layer by
    /// layer, each layer by horizontal bands.**
    ///
    /// Past the head, the decoder is three times `×2 → resnet ×3`, then `GroupNorm → SiLU → conv_out`:
    /// convolutions, which are local, and `GroupNorm`s, which are **not** — each normalizes a group of
    /// channels over the whole plane. So unlike Qwen-Image-2.1's tail (`QwenImage21VAEDecoder.tail`, per-
    /// pixel RMSNorm, one banded graph for two levels), a band here cannot run more than one
    /// normalization: every `GroupNorm` needs the statistics of a tensor that the previous layer
    /// finishes only with its last band. Hence one pass per layer, in our own buffers:
    ///
    ///     statistics(x)          one graph over the whole x: the mean and variance of each of the 32 groups
    ///     y = conv(SiLU(GN(x)))  by bands, normalized with THOSE statistics — global, as in one piece
    ///
    /// and the upsampling `conv(×2(x))` by bands too. A band reads its rows plus **one** row of margin
    /// on each side (a 3×3 convolution; for the ×2, one row of the input is two of the output), none
    /// past the image's edge, where the model's own zero padding is the right one; it keeps its core.
    /// All windows of a layer have the same height (the last slides back inside the image): one graph
    /// per layer, built just before it runs and released after, with its weight constants.
    ///
    /// **In place where it can be**: a resnet's output overwrites its input (`x + conv₂(…)` reads only
    /// the residual row of the row it writes, and a band's output is copied out after it ran), so a
    /// resnet holds `x` and `h₁` and nothing else. At 1024×1536 the largest moment is the first resnet
    /// of the last level: the doubled 256 channels at full resolution (1.6 GB) and its `h₁` (0.8 GB).
    ///
    /// What one graph used to cost: `MPSGraph` keeps every intermediate of a graph
    /// until it ends — 5.1 GB sampled at 1024², **7.2 GB at 1024×1536, the peak that swapped**.
    /// The price of the cut is time: ~40 graphs instead of one, +0.9 s at 1024² (3.3 → 4.3 s).
    private func decodeBanded(_ latent: UnsafePointer<Float>, height h: Int, width w: Int) throws -> [Float] {
        band = MemoryPlan.decoderBand(standard: VAE.bandElements, minimum: MemoryPlan.fluxMinimumBand, unit: "elements")
        let channels = config.latentChannels
        let input = try register(channels, h, w)
        input.values.update(from: latent, count: channels * h * w)
        var x = try autoreleasepool { () throws -> Register in
            let graph = MPSGraph()
            let b = VAEBuildingBlocks(graph: graph, weights: weights, name: "VAE", groups: config.groups)
            let placeholder = graph.placeholder(shape: input.data.shape, dataType: .float32, name: nil)
            let output = try head(b, placeholder, height: h, width: w)
            let result = try register(VAE.levels[0], h, w)
            QwenImage21VAE.execute(graph, queue: queue, feeds: [placeholder: input.data], results: [output: result.data])
            return result
        }
        VAE.layerDone?("head")
        var size = (h: h, w: w)
        for level in 1..<VAE.levels.count {
            let (entry, exit) = (VAE.levels[level - 1], VAE.levels[level])
            x = try upsample(x, "decoder.up_blocks.\(level - 1).upsamplers.0")
            size = (2 * size.h, 2 * size.w)
            for r in 0..<3 {
                x = try resnet(x, "decoder.up_blocks.\(level).resnets.\(r)", inChannels: r == 0 ? entry : exit,
                               outChannels: exit)
            }
        }
        let image = try normConv(x, norm: "decoder.conv_norm_out", conv: "decoder.conv_out", outChannels: 3,
                                 residual: nil, into: nil)
        return Array(UnsafeBufferPointer(start: image.values, count: 3 * size.h * size.w))
    }

    /// **The decoder as one graph — the reference of the band check, never the product.** It is
    /// the decoder as it was before the bands, operation for operation, run the same way (`graph.run`): the
    /// bands are judged against it on the same latent. At 1024×1536 it holds 7.2 GB.
    package func decodeInOnePiece(latent: UnsafePointer<Float>) throws -> [Float] {
        let (H, W) = (latentHeight, latentWidth)
        let graph = MPSGraph()
        let b = VAEBuildingBlocks(graph: graph, weights: weights, name: "VAE", groups: config.groups)
        let shape = [1, config.latentChannels, H, W].map { NSNumber(value: $0) }
        let input = graph.placeholder(shape: shape, dataType: .float32, name: "input")
        var x = try head(b, input, height: H, width: W)
        var size = (h: H, w: W)
        for level in 1..<VAE.levels.count {
            let (entry, exit) = (VAE.levels[level - 1], VAE.levels[level])
            x = try b.conv(b.double(x, channels: entry, size: size), "decoder.up_blocks.\(level - 1).upsamplers.0.conv",
                           padding: 1)
            size = (2 * size.h, 2 * size.w)
            for r in 0..<3 {
                x = try b.resnetFlux(x, "decoder.up_blocks.\(level).resnets.\(r)", inChannels: r == 0 ? entry : exit,
                                     outChannels: exit, size: size)
            }
        }
        x = try b.silu(b.groupNorm(x, "decoder.conv_norm_out", channels: VAE.levels.last!, h: size.h, w: size.w))
        x = try b.conv(x, "decoder.conv_out", padding: 1)
        let count = config.latentChannels * H * W
        guard let buffer = device.makeBuffer(length: count * 4, options: .storageModeShared) else { throw GEMM.Failure.noDevice }
        buffer.contents().assumingMemoryBound(to: Float.self).update(from: latent, count: count)
        var image = [Float](repeating: 0, count: 3 * size.h * size.w)
        autoreleasepool {
            let results = graph.run(with: queue, feeds: [input: MPSGraphTensorData(buffer, shape: shape, dataType: .float32)],
                                    targetTensors: [x], targetOperations: nil)
            image.withUnsafeMutableBytes { results[x]!.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
        }
        return image
    }

    /// Flux's `ResnetBlock2D` as two banded layers; the output takes the input's buffer.
    private func resnet(_ x: Register, _ prefix: String, inChannels: Int, outChannels: Int) throws -> Register {
        let h1 = try normConv(x, norm: prefix + ".norm1", conv: prefix + ".conv1", outChannels: outChannels,
                              residual: nil, into: nil)
        return try normConv(h1, norm: prefix + ".norm2", conv: prefix + ".conv2", outChannels: outChannels,
                            residual: (x, inChannels == outChannels ? nil : prefix + ".conv_shortcut"),
                            into: x.buffer)
    }

    private func register(_ channels: Int, _ height: Int, _ width: Int, buffer: MTLBuffer? = nil) throws -> Register {
        let shape = [1, channels, height, width].map { NSNumber(value: $0) }
        if let buffer { return Register(buffer, shape: shape) }
        guard let made = device.makeBuffer(length: channels * height * width * 4, options: .storageModeShared) else {
            throw GEMM.Failure.noDevice
        }
        return Register(made, shape: shape)
    }

    /// The statistics graphs, by group size: they carry no weight, and a shape comes back for every
    /// layer of a level.
    private var statisticsGraphs: [Int: (graph: MPSGraph, input: MPSGraphTensor, mean: MPSGraphTensor, variance: MPSGraphTensor)] = [:]

    /// The mean and variance of each `GroupNorm` group over the whole of `x` — the `mean` and `variance`
    /// that `VAEBuildingBlocks.groupNorm` lays in one piece, on the same `[1, 32, C/32·H·W]` shape, so the
    /// same bits. **`x`'s buffer is read as that shape directly**, not reshaped in the graph: the
    /// `reshape` of a fed tensor materialized a full copy — at the first resnet of the last level, 1.6 GB
    /// more, and the decoder's peak at 1024×1536 went 4.23 → 2.92 GB without it. Cutting the
    /// reduction by groups of groups would spare more, and **changes the bits** (another kernel per
    /// row count): refuted.
    private func statistics(_ x: Register) throws -> (mean: Register, variance: Register) {
        let groups = config.groups, perGroup = x.shape[1] / groups * x.shape[2] * x.shape[3]
        let m = try register(1, groups, 1), v = try register(1, groups, 1)
        let shape = [1, groups, perGroup].map { NSNumber(value: $0) }
        let s = statisticsGraphs[perGroup] ?? {
            let graph = MPSGraph()
            let input = graph.placeholder(shape: shape, dataType: .float32, name: nil)
            return (graph, input, graph.mean(of: input, axes: [2], name: nil), graph.variance(of: input, axes: [2], name: nil))
        }()
        statisticsGraphs[perGroup] = s
        autoreleasepool {
            QwenImage21VAE.execute(s.graph, queue: queue, feeds: [s.input: MPSGraphTensorData(x.buffer, shape: shape, dataType: .float32)],
                                   results: [s.mean: m.data, s.variance: v.data])
        }
        return (m, v)
    }

    /// One band layer: `conv(SiLU(GroupNorm(x)))`, plus the residual — `r`, or its 1×1 `conv_shortcut`
    /// — when there is one. Statistics over the whole of `x`; the convolution by bands.
    private func normConv(_ x: Register, norm: String, conv: String, outChannels: Int,
                          residual: (input: Register, shortcut: String?)?, into target: MTLBuffer?) throws -> Register {
        let (cx, H, W) = (x.shape[1], x.shape[2], x.shape[3])
        let stats = try statistics(x)
        let cr = residual?.input.shape[1] ?? 0
        let output = try register(outChannels, H, W, buffer: target)
        let (window, bands) = VAE.bands(rows: H, core: band / (max(cx, cr, outChannels) * W), margin: 1)
        let xw = try register(cx, window, W), out = try register(outChannels, window, W)
        let rw = try residual.map { try register($0.input.shape[1], window, W) }
        try autoreleasepool {
            let graph = MPSGraph()
            let b = VAEBuildingBlocks(graph: graph, weights: weights, name: "VAE", groups: config.groups)
            let px = graph.placeholder(shape: xw.data.shape, dataType: .float32, name: nil)
            let pm = graph.placeholder(shape: stats.mean.data.shape, dataType: .float32, name: nil)
            let pv = graph.placeholder(shape: stats.variance.data.shape, dataType: .float32, name: nil)
            var y = try b.silu(b.groupNorm(px, norm, channels: cx, h: window, w: W, statistics: (pm, pv)))
            y = try b.conv(y, conv, padding: 1)
            var feeds: [MPSGraphTensor: MPSGraphTensorData] = [px: xw.data, pm: stats.mean.data, pv: stats.variance.data]
            if let residual, let rw {
                let pr = graph.placeholder(shape: rw.data.shape, dataType: .float32, name: nil)
                feeds[pr] = rw.data
                // The order of `resnetFlux`: `residual + h`.
                y = graph.addition(try residual.shortcut.map { try b.conv(pr, $0, padding: 0) } ?? pr, y, name: nil)
            }
            for (c0, c1, start) in bands {
                VAE.copyPlanes(cx, from: x.values + start * W, stride: H * W, to: xw.values, stride: window * W,
                               count: window * W)
                if let residual, let rw {
                    VAE.copyPlanes(cr, from: residual.input.values + start * W, stride: H * W, to: rw.values,
                                   stride: window * W, count: window * W)
                }
                autoreleasepool {
                    QwenImage21VAE.execute(graph, queue: queue, feeds: feeds, results: [y: out.data])
                }
                // The core: rows [c0, c1), at c0 − start in the window. Written after the band ran — which is
                // what lets the output overwrite the residual: a kept row reads only its own residual row.
                VAE.copyPlanes(outChannels, from: out.values + (c0 - start) * W, stride: window * W,
                               to: output.values + c0 * W, stride: H * W, count: (c1 - c0) * W)
            }
        }
        VAE.layerDone?(conv)
        return output
    }

    /// The ×2 (`nearest-exact`, a duplication) and its 3×3 convolution, by bands of the INPUT's rows:
    /// one input row of margin is two output rows, one more than the convolution reads.
    private func upsample(_ x: Register, _ prefix: String) throws -> Register {
        let (c, h, w) = (x.shape[1], x.shape[2], x.shape[3])
        let output = try register(c, 2 * h, 2 * w)
        let (window, bands) = VAE.bands(rows: h, core: band / (c * 4 * w), margin: 1)
        let xw = try register(c, window, w), out = try register(c, 2 * window, 2 * w)
        try autoreleasepool {
            let graph = MPSGraph()
            let b = VAEBuildingBlocks(graph: graph, weights: weights, name: "VAE", groups: config.groups)
            let px = graph.placeholder(shape: xw.data.shape, dataType: .float32, name: nil)
            let y = try b.conv(b.double(px, channels: c, size: (window, w)), prefix + ".conv", padding: 1)
            for (c0, c1, start) in bands {
                VAE.copyPlanes(c, from: x.values + start * w, stride: h * w, to: xw.values, stride: window * w,
                               count: window * w)
                autoreleasepool {
                    QwenImage21VAE.execute(graph, queue: queue, feeds: [px: xw.data], results: [y: out.data])
                }
                VAE.copyPlanes(c, from: out.values + 2 * (c0 - start) * 2 * w, stride: 2 * window * 2 * w,
                               to: output.values + 2 * c0 * 2 * w, stride: 2 * h * 2 * w, count: 2 * (c1 - c0) * 2 * w)
            }
        }
        VAE.layerDone?(prefix)
        return output
    }

    /// **Every window starts on an even row — that is what makes the bands exact.** `MPSGraph`'s 3×3
    /// convolution does not compute a row the same way depending on its parity in the tensor it is
    /// given (two-row tiles, by all appearances): windows started on any row gave other bits — 9·10⁻⁸
    /// at worst per channel, fp32 noise, but other bits — and windows started on even rows give
    /// **the bits of the one graph**, at every band size (2, 4 and 8 alike). Every height
    /// banded here is even (a side is a multiple of 16, the latent's a multiple of 2), so the last
    /// window, flush with the bottom edge, starts even too.
    static let alignment = 2

    /// The bands of `rows`: cores `[c0, c1)` of about `core` rows, each read from a window
    /// `[start, start + window)` that holds `margin` rows on each side of its core (none past the image's
    /// edge) and starts on a multiple of `alignment`. All windows have the same height.
    static func bands(rows: Int, core requested: Int, margin: Int) -> (window: Int, bands: [(c0: Int, c1: Int, start: Int)]) {
        let a = alignment
        precondition(rows % a == 0, "VAE: a banded height must be even")
        let core = max(1, requested)
        let window = min(rows, (core + 2 * margin + a - 1 + a - 1) / a * a)
        var bands: [(Int, Int, Int)] = []
        var c0 = 0
        while c0 < rows {
            let c1 = min(rows, c0 + core)
            let start = min(max(0, c0 - margin) / a * a, rows - window)
            bands.append((c0, c1, start))
            c0 = c1
        }
        return (window, bands)
    }

    /// `count` contiguous floats of each of `channels` planes, from one stride to another — a window's rows
    /// in, a core's rows out. Split over the cores: one thread copied 2–3 GB per layer at 1024².
    private static func copyPlanes(_ channels: Int, from source: UnsafeMutablePointer<Float>, stride s: Int,
                                   to target: UnsafeMutablePointer<Float>, stride t: Int, count: Int) {
        Parallel.rows(channels, width: count) { first, n in
            for c in first..<(first + n) { (target + c * t).update(from: source + c * s, count: count) }
        }
    }


    /// **The tile positions along an axis.** The first starts at 0, the last ends on the
    /// edge, and the others are spread evenly between the two. The effective overlap is
    /// therefore **at least** the one requested — never less, often more at the edge, which is the right sense
    /// for blending: better to overlap too much than to leave a seam.
    static func positions(side L: Int, tile t: Int, overlap o: Int) -> [Int] {
        guard t < L else { return [0] }
        let step = max(1, t - o)
        let n = max(2, Int((Double(L - o) / Double(step)).rounded(.up)))
        return (0..<n).map { Int((Double($0) * Double(L - t) / Double(n - 1)).rounded()) }
    }

    /// **The blend: one weight per pixel, going down toward zero on the overlapped edges.**
    ///
    /// The weight is not exactly zero at the very edge — it would be `0` everywhere if two tiles
    /// overlapped exactly over their margin, and the normalization would divide by zero. So it starts
    /// from `1/(margin+1)`.
    /// Internal and not private: it is the only bulwark against a visible seam, and a function
    /// that decides what we see must be queryable without launching a decoder.
    static func ramp(_ position: Int, length: Int, margin: Int,
                      freeStart: Bool, freeEnd: Bool) -> Float {
        guard margin > 0 else { return 1 }
        var weights: Float = 1
        if freeStart, position < margin { weights *= Float(position + 1) / Float(margin + 1) }
        if freeEnd, position >= length - margin {
            weights *= Float(length - position) / Float(margin + 1)
        }
        return weights
    }

    /// **Tiled decoding, and what it approximates.**
    ///
    /// Each tile goes through the **whole** decoder (`decodeBanded` on the tile's latent), so the peak
    /// becomes that of **one** tile.
    ///
    /// It is not exact, and for two reasons that have nothing to do with the stitching:
    ///
    ///   - the **`GroupNorm`s** normalize over the whole plane, so each tile has its own
    ///     means and variances — those are other numbers, not another split of the same ones;
    ///   - the **bottleneck attention** is global: a tile only sees its own tokens.
    ///
    /// The overlap does not fix that; it only keeps the difference from reading as a
    /// line. It is `diffusers`' compromise, and it is **measured** — the image check returns the
    /// PSNR against the single-piece decoding, and `out/diff/` shows the seam if there is one. The
    /// bands of `decodeBanded` made it useless for room; it stays a setting that changes the
    /// image, refused in a profile.
    private func decodePerTiles(_ latent: UnsafePointer<Float>,
                                  _ t: (height: Int, width: Int, overlap: Int)) throws -> [Float] {
        let channels = config.latentChannels
        // Positions are taken **per axis**: a tile that covers a whole axis has only one place there.
        let rows = VAE.positions(side: latentHeight, tile: t.height, overlap: t.overlap)
        let columns = VAE.positions(side: latentWidth, tile: t.width, overlap: t.overlap)
        let (tileHeight, tileWidth) = (t.height * 8, t.width * 8)
        let (pictureHeight, pictureWidth) = (imageHeight, imageWidth)
        let margin = t.overlap * 8

        var sum = [Float](repeating: 0, count: 3 * pictureHeight * pictureWidth)
        var weights = [Float](repeating: 0, count: pictureHeight * pictureWidth)
        var slice = [Float](repeating: 0, count: channels * t.height * t.width)

        for y0 in rows {
            for x0 in columns {
                // The latent slice, channel by channel: `h` contiguous rows of `w` values.
                for c in 0..<channels {
                    for rowLine in 0..<t.height {
                        let source = latent + (c * latentHeight + y0 + rowLine) * latentWidth + x0
                        slice.withUnsafeMutableBufferPointer {
                            ($0.baseAddress! + (c * t.height + rowLine) * t.width).update(from: source, count: t.width)
                        }
                    }
                }
                let piece = try slice.withUnsafeBufferPointer {
                    try decodeBanded($0.baseAddress!, height: t.height, width: t.width)
                }
                // The blend, additive: we accumulate `value × weight` and the weight, and divide at the
                // end. No assumption about the effective overlap — it can be larger than
                // requested at the edges, and the normalization copes on its own.
                let freeTop = y0 > 0, freeBottom = y0 + t.height < latentHeight
                let freeLeft = x0 > 0, freeRight = x0 + t.width < latentWidth
                for ly in 0..<tileHeight {
                    let py = y0 * 8 + ly
                    let wy = VAE.ramp(ly, length: tileHeight, margin: margin,
                                       freeStart: freeTop, freeEnd: freeBottom)
                    for lx in 0..<tileWidth {
                        let px = x0 * 8 + lx
                        let w = wy * VAE.ramp(lx, length: tileWidth, margin: margin,
                                               freeStart: freeLeft, freeEnd: freeRight)
                        weights[py * pictureWidth + px] += w
                        for c in 0..<3 {
                            sum[(c * pictureHeight + py) * pictureWidth + px] +=
                                w * piece[(c * tileHeight + ly) * tileWidth + lx]
                        }
                    }
                }
            }
        }
        let pixels = pictureHeight * pictureWidth
        for p in 0..<pixels where weights[p] > 0 {
            for c in 0..<3 { sum[c * pixels + p] /= weights[p] }
        }
        return sum
    }

    /// `latent` as `[16, h, w]` as the sampler returns it. Outputs `[3, 8h, 8w]` in `[-1, 1]`.
    package func decode(latent: UnsafePointer<Float>) throws -> (image: [Float], seconds: Double) {
        let started = Date()
        let image = try tile.map { try decodePerTiles(latent, $0) }
            ?? decodeBanded(latent, height: latentHeight, width: latentWidth)
        return (image, Date().timeIntervalSince(started))
    }
}
