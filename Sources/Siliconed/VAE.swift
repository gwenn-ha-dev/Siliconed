import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import Foundation

/// The Flux VAE decoder, in fp32: `MPSGraph` passes of one kernel each, plus a few copy kernels of ours (`VAEKernels`).
/// It is Z-Image Turbo's decoder (`FluxDecodingModule`); Anima and Krea 2 decode with `AnimaVAE`
/// and Qwen-Image-2.1 with `QwenImage21VAE`, which are not this network. `AutoencoderKL.decoder` in
/// diffusers (`vae.py`: `Decoder`, `UNetMidBlock2D`, `UpDecoderBlock2D`).
///
///     latent [16, h, w] ─ ÷0.3611 then +0.1159 ─ conv_in ─ mid(resnet, attention, resnet)
///       ─ up×4 (3 resnets, ×2 except the last) ─ GroupNorm ─ SiLU ─ conv_out ─ image [3, 8h, 8w]
///
/// Channels 512 → 512 → 512 → 256 → 128 (`levels`); the ×2 sits between two levels, so the first
/// resnet of a level that narrows (512 → 256, 256 → 128) carries a 1×1 `conv_shortcut`.
///
/// **One resnet** — eps 1e-6, 32 groups, no time embedding:
///
///     h = conv₁(SiLU(GN₁(x)))        x ← (x, or conv_shortcut(x) if the channels change) + conv₂(SiLU(GN₂(h)))
///
/// **The bottleneck attention** — one head of width C = 512 over the h·w pixels, scale 1/√512:
///
///     t = GN(x) as tokens [h·w, C]      x += W_o( softmax(Wq t · (Wk t)ᵀ / √C) · Wv t ) + b
///
/// What a port by analogy would miss (reference: diffusers 0.41 `vae.py`, `resnet.py`, `unet_2d_blocks.py`,
/// and `pipeline_z_image.py` for the latent):
///
///   - **the latent's scale and shift are the pipeline's, not the VAE's**: `latents / scaling_factor +
///     shift_factor`, two scalars (`pipeline_z_image.py`, before `vae.decode`), applied here as the
///     first graph of `decodeBanded`. The Qwen-Image family denormalizes per channel instead;
///   - **`post_quant_conv` is optional** in `AutoencoderKL._decode` (`use_post_quant_conv`); this decoder
///     never reads one — `decodeBanded` goes from the denormalization straight to `conv_in`;
///   - **eps is 1e-6 in every `GroupNorm`**, the resnets' and the attention's (`resnet_eps`), not
///     torch's default 1e-5 (`groupNormalize`);
///   - **the attention is a single head of width `C`, not `C/64` heads**: `attention_head_dim` is
///     the block's channel count, so `heads = in_channels // attention_head_dim = 1` and the scale is
///     `dim_head^-0.5 = 512^-0.5` (`scaledDotProduct`, explicit `scale`); its `GroupNorm` runs on the
///     input before q, k, v, and the residual is the un-normalized input (`attention`);
///   - **the mid block is resnet, attention, resnet** (`UNetMidBlock2D.forward`), and each up block holds
///     `layers_per_block + 1 = 3` resnets, then its upsampler — so the ×2 comes AFTER the third resnet of a
///     level, which is where `decodeBanded` puts it (`upsample` of `up_blocks.(level−1)` before the next level's resnets);
///   - **the upsampler duplicates first, then convolves** (`Upsample2D.forward`: nearest ×2, then the 3×3
///     `conv`), the convolution running at the doubled resolution (`upsample`);
///   - **the resnet's shortcut reads the raw input**, before the norm (`ResnetBlock2D.forward`:
///     `input_tensor` is kept, `conv_shortcut(input_tensor)`), and is added to the second convolution's output (`resnet`, `normConv`).
///
/// The weights are published in **bf16**: reading them as fp32 would give finite and wrong numbers.
/// The VAE weighs 84 M parameters, so it has no forge: we read the published file directly.
///
/// ## Memory: by layers, bands and passes of one kernel — the same bits
///
/// In one graph the decoder held 5.1 GB at 1024² and **7.2 GB at 1024×1536, where Z-Image swapped**
/// (27,632 pages): `MPSGraph` keeps every intermediate of a graph until it ends, at full
/// resolution. Past the head, it now runs layer by layer in our own buffers, each layer by
/// horizontal bands (`decodeBanded`): **2.1 GB at 1024², 2.9 GB at 1024×1536 — the bits of the one
/// graph**, at every band size. Two things make it exact where a
/// tiling is not: each `GroupNorm` takes its statistics over the whole tensor, in a graph of their own,
/// and every band window starts on an even row (`alignment`).
///
/// **Every byte in two heaps of ours** (`VAEMemory`): the registers, band windows and weights in
/// a placement heap laid out in advance, MPS's workspace in a heap we hand it — all sized by `plan`, the
/// decoder walked without a device. Every graph is **one kernel** (a norm, its affine and SiLU, a
/// convolution and its sum), the ×2 and the attention's transposes kernels of ours: MPSGraph allocated
/// the rest beside our heaps (172 MB at 1024²). The same bits as the one graph it replaced; what stays MPSGraph's is its SDPA's own copies (`VAELayout.sdpaOutside`).
package final class VAE {
    package struct Config: Hashable, Sendable {
        package let scaling: Float, shift: Float
        package let groups: Int, latentChannels: Int
        package init(scaling: Float = 0.3611, shift: Float = 0.1159, groups: Int = 32, latentChannels: Int = 16) {
            self.scaling = scaling; self.shift = shift
            self.groups = groups; self.latentChannels = latentChannels
        }
    }

    private let weights: Safetensors
    private let config: Config
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let latentHeight, latentWidth: Int
    package var imageHeight: Int { latentHeight * 8 }
    package var imageWidth: Int { latentWidth * 8 }

    /// The elements of one window, all channels, at most (8 Mi floats = 32 MB): a band's height is
    /// chosen so that its widest tensor — input, output or residual — stays under it. At 1024², 2²³
    /// and 2²⁴ cost the same time (4.4 s) and 2²³ 140 MB less; 2²⁰ costs 7.3 s.
    package static let bandElements = 1 << 23
    /// This decoding's band: `bandElements`, or the smallest proved one on a plan whose B does not hold
    /// it (`MemoryPlan.decoderBand`). Set as `decodeBanded` starts.
    private var band = VAE.bandElements

    /// Z-Image's decoder is published in bf16 (`VAEPrecision`).
    package static let publishedInBF16 = true

    /// The channels of the four up blocks, from the bottleneck out (`block_out_channels` reversed).
    private static let levels = [512, 512, 256, 128]

    /// `latentHeight`, `latentWidth`: the grid of the `[16, h, w]` latent.
    package init(path: String, latentHeight: Int, latentWidth: Int, config: Config = Config()) throws {
        let weights = try Safetensors(path: path)
        self.weights = weights
        self.config = config
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
        let (device, queue) = try SystemGPU.open()
        self.device = device
        self.queue = queue
        guard weights.entries["decoder.conv_in.weight"]?.shape.dropFirst().first == config.latentChannels,
              weights.entries["decoder.conv_out.weight"]?.shape.first == 3 else {
            throw Safetensors.Failure.badHeader("\(path): not a Flux VAE with \(config.latentChannels) latent channels")
        }
    }

    /// **The whole decoder on a `[C, h, w]` latent, exactly — the head by passes, then layer by
    /// layer, each layer by horizontal bands.**
    ///
    /// Past the head, the decoder is three times `×2 → resnet ×3`, then `GroupNorm → SiLU → conv_out`:
    /// convolutions, which are local, and `GroupNorm`s, which are **not** — each normalizes a group of
    /// channels over the whole plane. So unlike Qwen-Image-2.1's tail (`QwenImage21VAEDecoder.tail`, per-
    /// pixel RMSNorm, one banded graph for two levels), a band here cannot run more than one
    /// normalization: every `GroupNorm` needs the statistics of a tensor that the previous layer
    /// finishes only with its last band. Hence one pass per layer, in our own registers (`Bands`):
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
    /// **Every byte in two heaps of ours** (`VAEMemory`), laid out by `plan` — this same
    /// function, walked without a device — and handed back to Metal before `decode` returns.
    ///
    /// What one graph used to cost: `MPSGraph` keeps every intermediate of a graph
    /// until it ends — 5.1 GB sampled at 1024², **7.2 GB at 1024×1536, the peak that swapped**.
    /// The price of the cut is time: ~40 graphs instead of one, +0.9 s at 1024² (3.3 → 4.3 s).
    private static func decodeBanded(_ latent: UnsafePointer<Float>?, height h: Int, width w: Int, config: Config,
                                     bands: Bands) throws -> [Float] {
        let memory = bands.memory
        let input = try memory.register([1, config.latentChannels, h, w], .float32)
        if let latent, memory.isLive { input.values.update(from: latent, count: input.count) }
        // The head, at the latent's grid: the denormalization, `conv_in`, the
        // middle block and its attention, the first up block's three resnets, by passes.
        // In bf16 (`VAEMemory.half`) the denormalization is computed on the fp32 latent and rounded once, into the activations.
        var x = input
        if config.scaling != 1 || config.shift != 0 || memory.half {
            x = try bands.whole([input], input.shape, weights: []) { b, p in
                let y = b.graph.addition(b.graph.division(p[0], b.graph.constant(Double(config.scaling), dataType: .float32), name: nil),
                                         b.graph.constant(Double(config.shift), dataType: .float32), name: "denormalize")
                return memory.half ? b.graph.cast(y, to: .bFloat16, name: nil) : y
            }
            memory.free(input)
        }
        let entry = try bands.conv(x, "decoder.conv_in", outChannels: VAE.levels[0])
        memory.free(x)
        x = entry
        let c = VAE.levels[0]
        x = try bands.resnet(x, "decoder.mid_block.resnets.0", inChannels: c, outChannels: c)
        try bands.attention(x, "decoder.mid_block.attentions.0")
        x = try bands.resnet(x, "decoder.mid_block.resnets.1", inChannels: c, outChannels: c)
        for r in 0..<3 { x = try bands.resnet(x, "decoder.up_blocks.0.resnets.\(r)", inChannels: c, outChannels: c) }
        for level in 1..<VAE.levels.count {
            let (entry, exit) = (VAE.levels[level - 1], VAE.levels[level])
            let doubled = try bands.upsample(x, "decoder.up_blocks.\(level - 1).upsamplers.0")
            memory.free(x)
            x = doubled
            for r in 0..<3 {
                x = try bands.resnet(x, "decoder.up_blocks.\(level).resnets.\(r)", inChannels: r == 0 ? entry : exit,
                                     outChannels: exit)
            }
        }
        let image = try bands.normConv(x, norm: "decoder.conv_norm_out", conv: "decoder.conv_out", outChannels: 3,
                                       residual: nil, into: nil)
        memory.free(x)
        memory.host(4 * image.count)
        let pixels = memory.isLive ? image.floats() : []
        memory.free(image)
        return pixels
    }

    /// **Everything `decodeBanded` holds, for a `h × w` latent at a band of `band` elements** — the
    /// same function walked without a device: its registers' heap, the largest graph's workspace,
    /// and the image it returns. `decode` allocates exactly this before its first graph. `half`: the decoder in bf16
    /// (`VAEPrecision`).
    package static func plan(latentHeight h: Int, latentWidth w: Int, config: Config = Config(),
                             band: Int = VAE.bandElements, half: Bool = false) -> VAELayout {
        plans(VAEPlanKey(config, [h, w, band, half ? 1 : 0])) {
            walk(latentHeight: h, latentWidth: w, config: config, band: band, half: half)
        }
    }

    private static let plans = ShapeMemo<VAEPlanKey<Config>, VAELayout>()

    /// `plan`, walked every time: what the memory keeps, and what the tests compare it with.
    package static func walk(latentHeight h: Int, latentWidth w: Int, config: Config = Config(),
                             band: Int = VAE.bandElements, half: Bool = false) -> VAELayout {
        let memory = VAEMemory(half: half)
        let bands = Bands(memory: memory, weights: nil, name: "VAE", groups: config.groups, band: band)
        do { _ = try decodeBanded(nil, height: h, width: w, config: config, bands: bands) } catch { preconditionFailure("VAE plan walk refused: \(error)") }
        return memory.layout
    }

    private func decodeBanded(_ latent: UnsafePointer<Float>, height h: Int, width w: Int) throws -> [Float] {
        band = MemoryPlan.decoderBand(standard: VAE.bandElements, minimum: MemoryPlan.fluxMinimumBand, wide: MemoryPlan.fluxWideBand,
                                      unit: "elements")
        let half = VAEPrecision.half(publishedInBF16: VAE.publishedInBF16, weights: weights)
        let memory = try VAEMemory(plan: VAE.plan(latentHeight: h, latentWidth: w, config: config, band: band, half: half),
                                   device: device, queue: queue, half: half)
        let image = try autoreleasepool {
            try VAE.decodeBanded(latent, height: h, width: w, config: config,
                                 bands: Bands(memory: memory, weights: weights, name: "VAE", groups: config.groups, band: band))
        }
        memory.finish()
        return image
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

    /// `count` contiguous elements of each of `channels` planes, from one stride to another — a window's rows
    /// in, a core's rows out; offsets, strides and count in elements of the registers' type (both the same). Split over the
    /// cores: one thread copied 2–3 GB per layer at 1024².
    static func copyPlanes(_ channels: Int, from source: VAERegister, at s0: Int, stride s: Int,
                           to target: VAERegister, at t0: Int, stride t: Int, count: Int) {
        precondition(source.dataType == target.dataType, "VAE.copyPlanes: registers of two types")
        let e = source.elementSize
        let (from, to) = (source.raw + e * s0, target.raw + e * t0)
        Parallel.rows(channels, width: count * e / 4) { first, n in
            for c in first..<(first + n) { (to + e * c * t).copyMemory(from: from + e * c * s, byteCount: e * count) }
        }
    }


    /// `latent` as `[16, h, w]` as the sampler returns it. Outputs `[3, 8h, 8w]` in `[-1, 1]`.
    package func decode(latent: UnsafePointer<Float>) throws -> (image: [Float], seconds: Double) {
        let started = Date()
        let image = try decodeBanded(latent, height: latentHeight, width: latentWidth)
        return (image, Date().timeIntervalSince(started))
    }
}

// ── sizing: the banded decoding's buffers, moment by moment ──────────────────────────────────────

extension VAE {
    /// The core rows of a `normConv` band: its widest tensor (input, output or residual, `channels`)
    /// times its rows stays under `band` elements.
    static func normConvCore(band: Int, channels: Int, width: Int) -> Int { band / (channels * width) }

    /// The core rows of an upsampling band, counted on the input: its output is `4×` the input's plane.
    static func upsampleCore(band: Int, channels: Int, width: Int) -> Int { band / (channels * 4 * width) }

    /// The core rows of a stride-2 band, counted on the output: its input window is `2×` the rows at
    /// the input's `width`, the widest tensor of the band.
    static func downsampleCore(band: Int, channels: Int, width: Int) -> Int { band / (channels * 2 * width) }

    /// **One moment of `decodeBanded`**: the buffers alive (`bytes`, ours: registers and band windows),
    /// and the output pixels of the graph that runs then (`pixels`; the head's are the latent's).
    package struct Moment: Sendable, Equatable {
        package let layer: String
        package let bytes: Int
        package let pixels: Int
    }

    /// **`decodeBanded`'s buffers, walked layer by layer** without a device: the latent `input` (kept
    /// to the end), the head's result, then for each layer the input `x`, its output (a new register, or
    /// `x`'s own buffer for a resnet's second half, which keeps the size of the buffer it was made in —
    /// 512 channels through a level that narrows to 256), and the band windows (`VAE.bands` on
    /// `normConvCore` / `upsampleCore`, the functions the decoder cuts with). The last moment adds the
    /// `[Float]` copy the decoder returns. Mirrors `decodeBanded`, `resnet`, `normConv` and `upsample`;
    /// the `GroupNorm` statistics (2 × 32 floats) are left out.
    package static func bandedMoments(latentHeight h: Int, latentWidth w: Int, latentChannels: Int,
                                      band: Int = VAE.bandElements) -> [Moment] {
        let f = MemoryLayout<Float>.size
        let input = latentChannels * h * w * f
        var moments = [Moment(layer: "head", bytes: input + levels[0] * h * w * f, pixels: h * w)]
        // x: the register a layer reads — its buffer's bytes (not always its shape's) and its shape.
        var buffer = levels[0] * h * w * f, channels = levels[0], (H, W) = (h, w)
        for level in 1..<levels.count {
            let (entry, exit) = (levels[level - 1], levels[level])
            // upsample: x, its ×2 output, the input window and the output window.
            let up = bands(rows: H, core: upsampleCore(band: band, channels: channels, width: W), margin: 1).window
            let doubled = channels * 2 * H * 2 * W * f
            moments.append(Moment(layer: "decoder.up_blocks.\(level - 1).upsamplers.0",
                                  bytes: input + buffer + doubled + channels * up * W * f + channels * 2 * up * 2 * W * f,
                                  pixels: 2 * up * 2 * W))
            buffer = doubled; (H, W) = (2 * H, 2 * W)
            for r in 0..<3 {
                let cin = r == 0 ? entry : exit
                precondition(cin == channels, "VAE.bandedMoments: the walk lost the decoder's channels")
                let h1 = exit * H * W * f
                // conv1: x → h1, windows of x and of h1.
                let w1 = bands(rows: H, core: normConvCore(band: band, channels: max(cin, exit), width: W), margin: 1).window
                moments.append(Moment(layer: "decoder.up_blocks.\(level).resnets.\(r).conv1",
                                      bytes: input + buffer + h1 + cin * w1 * W * f + exit * w1 * W * f, pixels: w1 * W))
                // conv2: h1 (+ x as residual) → x's buffer; windows of h1, of the output and of the residual.
                let w2 = bands(rows: H, core: normConvCore(band: band, channels: max(exit, cin), width: W), margin: 1).window
                moments.append(Moment(layer: "decoder.up_blocks.\(level).resnets.\(r).conv2",
                                      bytes: input + buffer + h1 + 2 * exit * w2 * W * f + cin * w2 * W * f,
                                      pixels: w2 * W))
                channels = exit
            }
        }
        // conv_out: x → the image (3 channels), then its `[Float]` copy beside it.
        let image = 3 * H * W * f
        let wo = bands(rows: H, core: normConvCore(band: band, channels: max(channels, 3), width: W), margin: 1).window
        moments.append(Moment(layer: "decoder.conv_out",
                              bytes: input + buffer + image + channels * wo * W * f + 3 * wo * W * f, pixels: wo * W))
        moments.append(Moment(layer: "output", bytes: input + buffer + 2 * image, pixels: 0))
        return moments
    }
}

// ── the banded layers, decoder's and encoder's ───────────────────────────────────────────────────

extension VAE {
    /// **Flux's layers, in our registers, by passes of one kernel** — the decoder's (`decodeBanded`) and
    /// the encoder's (`VAEEncoder.encodeBanded`): a `GroupNorm`'s statistics over the whole tensor, then
    /// `SiLU(GN(x))` and its convolution (plus a residual), a bare convolution, a ×2 or a stride-2
    /// convolution, each by horizontal bands; the bottleneck's attention over the whole tensor. Every
    /// register comes from `memory`, and every graph declares its workspace before it is built
    /// (`VAEMemory.graph`): with `weights` `nil` and a planning `memory`, the same calls are the plan.
    ///
    /// **One kernel per graph**: a band graph `conv(SiLU(GN(x)))` put its normalized window, and a
    /// `conv(×2(x))` its doubled window, in Metal memory of MPSGraph's own beside our heaps (172 MB at
    /// 1024²); the head in one graph 163 MB more. Cut between the norm and its convolution, the ×2 a
    /// kernel of ours, the same bits.
    final class Bands {
        let memory: VAEMemory
        let weights: Safetensors?
        let name: String
        let groups: Int
        /// The elements of one window, all channels, at most (`VAE.bandElements`).
        let band: Int

        init(memory: VAEMemory, weights: Safetensors?, name: String, groups: Int, band: Int) {
            self.memory = memory; self.weights = weights; self.name = name; self.groups = groups; self.band = band
        }

        func blocks(_ graph: MPSGraph, _ fed: FedWeights) -> VAEBuildingBlocks {
            VAEBuildingBlocks(graph: graph, weights: weights!, name: name, groups: groups, fed: fed, dataType: memory.activation)
        }

        /// A graph of one kernel over `inputs`, its weights in registers, ready to run on every band.
        struct Pass {
            let built: (graph: MPSGraph, output: MPSGraphTensor)?
            let feeds: [MPSGraphTensor: MPSGraphTensorData]
            let weights: [VAERegister]
        }

        /// Builds a pass (running) or charges it (planning): `declared`, the floats of the weights it
        /// reads in order (`WeightList`); `workspace`, what MPS asks for it (`VAELayout.minimumAsk`).
        func pass(_ inputs: [VAERegister], weights declared: [Int], workspace: Int = VAELayout.minimumAsk / 4,
                  _ build: (VAEBuildingBlocks, [MPSGraphTensor]) throws -> MPSGraphTensor) throws -> Pass {
            let fed = FedWeights()
            let built = try memory.graph(largest: workspace) { () -> (MPSGraph, [MPSGraphTensor], MPSGraphTensor) in
                let graph = MPSGraph()
                let placeholders = inputs.map { graph.placeholder(shape: $0.nsShape, dataType: $0.dataType, name: nil) }
                return (graph, placeholders, try build(blocks(graph, fed), placeholders))
            }
            let w = try memory.weights(declared, built.map { _ in fed }, file: weights, vae: name)
            var feeds = w.feeds
            if let built { for (p, r) in zip(built.1, inputs) { feeds[p] = r.data! } }
            return Pass(built: built.map { ($0.0, $0.2) }, feeds: feeds, weights: w.registers)
        }

        func run(_ pass: Pass, into output: VAERegister) {
            if let b = pass.built { memory.execute(b.graph, feeds: pass.feeds, results: [b.output: output.data!]) }
        }

        func done(_ pass: Pass) { memory.free(pass.weights) }

        /// One pass over whole registers: built, run once, its weights freed. Returns its output.
        func whole(_ inputs: [VAERegister], _ shape: [Int], weights declared: [Int], workspace: Int = VAELayout.minimumAsk / 4,
                   _ build: (VAEBuildingBlocks, [MPSGraphTensor]) throws -> MPSGraphTensor) throws -> VAERegister {
            let output = try memory.register(shape)
            try autoreleasepool {
                let p = try pass(inputs, weights: declared, workspace: workspace, build)
                run(p, into: output)
                done(p)
            }
            return output
        }

        /// The statistics graphs, by group size: they carry no weight, and a shape comes back for every
        /// layer of a level.
        private var statisticsGraphs: [Int: (graph: MPSGraph, input: MPSGraphTensor, mean: MPSGraphTensor, variance: MPSGraphTensor)] = [:]

        /// Flux's `ResnetBlock2D` as two banded layers. The output takes the input's slot when it fits
        /// there (the decoder: channels never grow), a new register otherwise (the encoder's widening
        /// resnets), and the input is then freed.
        func resnet(_ x: VAERegister, _ prefix: String, inChannels: Int, outChannels: Int) throws -> VAERegister {
            let h1 = try normConv(x, norm: prefix + ".norm1", conv: prefix + ".conv1", outChannels: outChannels,
                                  residual: nil, into: nil)
            let inPlace = outChannels <= inChannels
            let y = try normConv(h1, norm: prefix + ".norm2", conv: prefix + ".conv2", outChannels: outChannels,
                                 residual: (x, inChannels == outChannels ? nil : prefix + ".conv_shortcut"),
                                 into: inPlace ? x : nil)
            memory.free(h1)
            if !inPlace { memory.free(x) }
            return y
        }

        /// The mean and variance of each `GroupNorm` group over the whole of `x` — the `mean` and `variance`
        /// that `VAEBuildingBlocks.groupNorm` lays in one piece, on the same `[1, 32, C/32·H·W]` shape, so the
        /// same bits. **`x`'s buffer is read as that shape directly**, not reshaped in the graph: the
        /// `reshape` of a fed tensor materialized a full copy — at the first resnet of the last level, 1.6 GB
        /// more, and the decoder's peak at 1024×1536 went 4.23 → 2.92 GB without it. Cutting the
        /// reduction by groups of groups would spare more, and **changes the bits** (another kernel per
        /// row count): refuted. MPS's reduction asks 2 MiB of workspace of its own.
        private func statistics(_ x: VAERegister) throws -> (mean: VAERegister, variance: VAERegister) {
            if memory.half { return try memory.groupStatistics(x, groups: groups) }
            let perGroup = x.shape[1] / groups * x.shape[2] * x.shape[3]
            let m = try memory.register([1, groups, 1]), v = try memory.register([1, groups, 1])
            let shape = [1, groups, perGroup].map { NSNumber(value: $0) }
            let s = memory.graph(largest: VAELayout.minimumAsk / 4) {
                () -> (graph: MPSGraph, input: MPSGraphTensor, mean: MPSGraphTensor, variance: MPSGraphTensor) in
                if let s = statisticsGraphs[perGroup] { return s }
                let graph = MPSGraph()
                let input = graph.placeholder(shape: shape, dataType: .float32, name: nil)
                let s = (graph, input, graph.mean(of: input, axes: [2], name: nil), graph.variance(of: input, axes: [2], name: nil))
                statisticsGraphs[perGroup] = s
                return s
            }
            if let s {
                memory.execute(s.graph, feeds: [s.input: MPSGraphTensorData(x.buffer!, shape: shape, dataType: .float32)],
                               results: [s.mean: m.data!, s.variance: v.data!])
            }
            return (m, v)
        }

        /// One band layer: `conv(SiLU(GroupNorm(x)))`, plus the residual — `r`, or its 1×1 `conv_shortcut`
        /// — when there is one. Statistics over the whole of `x`; then, by bands, three passes: the norm
        /// and its SiLU, the shortcut, the convolution and the sum. `target`: the register whose slot the
        /// output takes (the residual's own: a kept row reads only its own residual row, and the core is
        /// written after the band ran).
        func normConv(_ x: VAERegister, norm: String, conv: String, outChannels: Int,
                      residual: (input: VAERegister, shortcut: String?)?, into target: VAERegister?) throws -> VAERegister {
            if memory.half {
                return try normConvHalf(x, norm: norm, conv: conv, outChannels: outChannels, residual: residual, into: target)
            }
            let (cx, H, W) = (x.shape[1], x.shape[2], x.shape[3])
            let stats = try statistics(x)
            let cr = residual?.input.shape[1] ?? 0
            let output = try target.map { try memory.view($0, as: [1, outChannels, H, W]) }
                ?? memory.register([1, outChannels, H, W])
            let (window, bands) = VAE.bands(rows: H, core: VAE.normConvCore(band: band, channels: max(cx, cr, outChannels), width: W),
                                            margin: 1)
            let xw = try memory.register([1, cx, window, W]), ng = try memory.register([1, groups, cx / groups * window * W])
            let nw = try memory.register([1, cx, window, W])
            let out = try memory.register([1, outChannels, window, W])
            let rw = try residual.map { try memory.register([1, $0.input.shape[1], window, W]) }
            let sw = try residual?.shortcut.map { _ in try memory.register([1, outChannels, window, W]) }
            try autoreleasepool {
                // The norm in two passes, each tensor fed at the shape its graph reads (a `reshape` inside a
                // graph was a copy of MPSGraph's own): normalized by group, then γ, β and the SiLU by channel.
                let normalize = try pass([memory.view(xw, as: ng.shape), stats.mean, stats.variance], weights: []) { b, p in
                    b.groupNormalize(p[0], mean: p[1], variance: p[2])
                }
                let activate = try pass([memory.view(ng, as: nw.shape)], weights: WeightList.groupNorm(cx)) { b, p in
                    try b.silu(b.affine(p[0], norm))
                }
                let shortcut = try residual?.shortcut.map { name in
                    try pass([rw!], weights: WeightList.conv(outChannels, cr, kernel: 1),
                             workspace: VAELayout.reformattedWeights(outChannels * cr) / 4) { b, p in try b.conv(p[0], name, padding: 0) }
                }
                // The order of `resnetFlux`: `residual + h`.
                let convolve = try pass([nw] + ((sw ?? rw).map { [$0] } ?? []), weights: WeightList.conv(outChannels, cx)) { b, p in
                    let y = try b.conv(p[0], conv, padding: 1)
                    return p.count > 1 ? b.graph.addition(p[1], y, name: nil) : y
                }
                for (c0, c1, start) in bands where memory.isLive {
                    VAE.copyPlanes(cx, from: x, at: start * W, stride: H * W, to: xw, at: 0, stride: window * W,
                                   count: window * W)
                    if let residual, let rw {
                        VAE.copyPlanes(cr, from: residual.input, at: start * W, stride: H * W, to: rw, at: 0,
                                       stride: window * W, count: window * W)
                    }
                    run(normalize, into: ng)
                    run(activate, into: nw)
                    if let shortcut, let sw { run(shortcut, into: sw) }
                    run(convolve, into: out)
                    // The core: rows [c0, c1), at c0 − start in the window. Written after the band ran — which is
                    // what lets the output overwrite the residual: a kept row reads only its own residual row.
                    VAE.copyPlanes(outChannels, from: out, at: (c0 - start) * W, stride: window * W,
                                   to: output, at: c0 * W, stride: H * W, count: (c1 - c0) * W)
                }
                done(convolve)
                if let shortcut { done(shortcut) }
                done(activate)
                done(normalize)
            }
            if let sw { memory.free(sw) }
            if let rw { memory.free(rw) }
            memory.free(out); memory.free(nw); memory.free(ng); memory.free(xw)
            memory.free(stats.variance); memory.free(stats.mean)
            return output
        }

        /// **`normConv` in bf16** (`VAEMemory.half`): the statistics over the whole of `x` (fp32), then by bands the norm, its
        /// affine and SiLU in one kernel of ours (fp32 inside, rounded once to bf16), the shortcut, the convolution and its sum.
        private func normConvHalf(_ x: VAERegister, norm: String, conv: String, outChannels: Int,
                                  residual: (input: VAERegister, shortcut: String?)?, into target: VAERegister?) throws -> VAERegister {
            let (cx, H, W) = (x.shape[1], x.shape[2], x.shape[3])
            let stats = try statistics(x)
            let cr = residual?.input.shape[1] ?? 0
            let output = try target.map { try memory.view($0, as: [1, outChannels, H, W]) }
                ?? memory.register([1, outChannels, H, W])
            let (window, bands) = VAE.bands(rows: H, core: VAE.normConvCore(band: band, channels: max(cx, cr, outChannels), width: W),
                                            margin: 1)
            let xw = try memory.register([1, cx, window, W]), nw = try memory.register([1, cx, window, W])
            let out = try memory.register([1, outChannels, window, W])
            let rw = try residual.map { try memory.register([1, $0.input.shape[1], window, W]) }
            let sw = try residual?.shortcut.map { _ in try memory.register([1, outChannels, window, W]) }
            let affine = try memory.groupNormWeights(norm, channels: cx, file: weights, vae: name)
            try autoreleasepool {
                let shortcut = try residual?.shortcut.map { name in
                    try pass([rw!], weights: WeightList.conv(outChannels, cr, kernel: 1),
                             workspace: VAELayout.reformattedWeights(outChannels * cr) / 4) { b, p in try b.conv(p[0], name, padding: 0) }
                }
                let convolve = try pass([nw] + ((sw ?? rw).map { [$0] } ?? []), weights: WeightList.conv(outChannels, cx)) { b, p in
                    let y = try b.conv(p[0], conv, padding: 1)
                    return p.count > 1 ? b.graph.addition(p[1], y, name: nil) : y
                }
                for (c0, c1, start) in bands where memory.isLive {
                    VAE.copyPlanes(cx, from: x, at: start * W, stride: H * W, to: xw, at: 0, stride: window * W, count: window * W)
                    if let residual, let rw {
                        VAE.copyPlanes(cr, from: residual.input, at: start * W, stride: H * W, to: rw, at: 0,
                                       stride: window * W, count: window * W)
                    }
                    memory.groupNorm(xw, into: nw, stats: stats, groups: groups, affine: affine, silu: true, epsilon: 1e-6)
                    if let shortcut, let sw { run(shortcut, into: sw) }
                    run(convolve, into: out)
                    VAE.copyPlanes(outChannels, from: out, at: (c0 - start) * W, stride: window * W,
                                   to: output, at: c0 * W, stride: H * W, count: (c1 - c0) * W)
                }
                done(convolve)
                if let shortcut { done(shortcut) }
            }
            memory.free(affine)
            if let sw { memory.free(sw) }
            if let rw { memory.free(rw) }
            memory.free(out); memory.free(nw); memory.free(xw)
            memory.free(stats.variance); memory.free(stats.mean)
            return output
        }

        /// A bare 3×3 convolution (padding 1) by bands — `conv_in`.
        func conv(_ x: VAERegister, _ prefix: String, outChannels: Int) throws -> VAERegister {
            let (cx, H, W) = (x.shape[1], x.shape[2], x.shape[3])
            let output = try memory.register([1, outChannels, H, W])
            let (window, bands) = VAE.bands(rows: H, core: VAE.normConvCore(band: band, channels: max(cx, outChannels), width: W),
                                            margin: 1)
            let xw = try memory.register([1, cx, window, W]), out = try memory.register([1, outChannels, window, W])
            try autoreleasepool {
                let p = try pass([xw], weights: WeightList.conv(outChannels, cx)) { b, p in try b.conv(p[0], prefix, padding: 1) }
                for (c0, c1, start) in bands where memory.isLive {
                    VAE.copyPlanes(cx, from: x, at: start * W, stride: H * W, to: xw, at: 0, stride: window * W,
                                   count: window * W)
                    run(p, into: out)
                    VAE.copyPlanes(outChannels, from: out, at: (c0 - start) * W, stride: window * W,
                                   to: output, at: c0 * W, stride: H * W, count: (c1 - c0) * W)
                }
                done(p)
            }
            memory.free(out); memory.free(xw)
            return output
        }

        /// The ×2 (`nearest-exact`, a duplication: a kernel of ours) and its 3×3 convolution, by bands of
        /// the INPUT's rows: one input row of margin is two output rows, one more than the convolution reads.
        func upsample(_ x: VAERegister, _ prefix: String) throws -> VAERegister {
            let (c, h, w) = (x.shape[1], x.shape[2], x.shape[3])
            let output = try memory.register([1, c, 2 * h, 2 * w])
            let (window, bands) = VAE.bands(rows: h, core: VAE.upsampleCore(band: band, channels: c, width: w), margin: 1)
            let xw = try memory.register([1, c, window, w]), dw = try memory.register([1, c, 2 * window, 2 * w])
            let out = try memory.register([1, c, 2 * window, 2 * w])
            try autoreleasepool {
                let p = try pass([dw], weights: WeightList.conv(c, c)) { b, p in try b.conv(p[0], prefix + ".conv", padding: 1) }
                for (c0, c1, start) in bands where memory.isLive {
                    VAE.copyPlanes(c, from: x, at: start * w, stride: h * w, to: xw, at: 0, stride: window * w,
                                   count: window * w)
                    memory.dispatch(.double, [xw, dw], constants: [UInt32(window), UInt32(w)],
                                    grid: MTLSize(width: 2 * w, height: 2 * window, depth: c))
                    run(p, into: out)
                    VAE.copyPlanes(c, from: out, at: 2 * (c0 - start) * 2 * w, stride: 2 * window * 2 * w,
                                   to: output, at: 2 * c0 * 2 * w, stride: 2 * h * 2 * w, count: 2 * (c1 - c0) * 2 * w)
                }
                done(p)
            }
            memory.free(out); memory.free(dw); memory.free(xw)
            return output
        }

        /// **The encoders' stride-2 convolution, padding `(0, 1, 0, 1)`, by bands of the OUTPUT's rows.**
        /// Output row `o` reads input rows `2o … 2o + 2`: a window of `n` output rows reads `2n` input rows
        /// from `2·start`, and its last output row reads the graph's bottom zero padding instead of the
        /// row below the window — so it is a margin row, never kept, except in the window flush with the
        /// image's bottom, where that padding is the model's own. The windows start on even output rows
        /// (`VAE.bands`), i.e. on input rows that are multiples of 4.
        func downsample(_ x: VAERegister, _ prefix: String) throws -> VAERegister {
            let (c, H, W) = (x.shape[1], x.shape[2], x.shape[3])
            let (h, w) = (H / 2, W / 2)
            let output = try memory.register([1, c, h, w])
            let (window, bands) = VAE.bands(rows: h, core: VAE.downsampleCore(band: band, channels: c, width: W), margin: 1)
            let xw = try memory.register([1, c, 2 * window, W]), out = try memory.register([1, c, window, w])
            try autoreleasepool {
                let p = try pass([xw], weights: WeightList.conv(c, c), workspace: VAELayout.reformattedWeights(c * c * 9) / 4) { b, p in
                    try b.downsample(p[0], prefix)
                }
                for (c0, c1, start) in bands where memory.isLive {
                    VAE.copyPlanes(c, from: x, at: 2 * start * W, stride: H * W, to: xw, at: 0, stride: 2 * window * W,
                                   count: 2 * window * W)
                    run(p, into: out)
                    VAE.copyPlanes(c, from: out, at: (c0 - start) * w, stride: window * w,
                                   to: output, at: c0 * w, stride: h * w, count: (c1 - c0) * w)
                }
                done(p)
            }
            memory.free(out); memory.free(xw)
            return output
        }

        /// **The bottleneck's attention**, in place over `x` (`[1, C, h, w]`): `GroupNorm` (statistics,
        /// then the norm alone), the tokens (a transpose of ours), q, k, v, one head (MPSGraph's SDPA), the
        /// output projection, and the residual laid back by a kernel of ours — `VAEBuildingBlocks.attention`'s
        /// operations, in its order.
        func attention(_ x: VAERegister, _ prefix: String) throws {
            let (c, h, w) = (x.shape[1], x.shape[2], x.shape[3]), s = h * w
            let stats = try statistics(x)
            let n: VAERegister
            if memory.half {
                // The norm and its affine in one kernel of ours, fp32 inside, rounded once.
                n = try memory.register(x.shape)
                let affine = try memory.groupNormWeights(prefix + ".group_norm", channels: c, file: weights, vae: name)
                if memory.isLive {
                    memory.groupNorm(x, into: n, stats: stats, groups: groups, affine: affine, silu: false, epsilon: 1e-6)
                }
                memory.free(affine)
                memory.free(stats.variance); memory.free(stats.mean)
            } else {
                let g = try whole([memory.view(x, as: [1, groups, c / groups * s]), stats.mean, stats.variance],
                                  [1, groups, c / groups * s], weights: []) { b, p in
                    b.groupNormalize(p[0], mean: p[1], variance: p[2])
                }
                memory.free(stats.variance); memory.free(stats.mean)
                n = try whole([memory.view(g, as: x.shape)], x.shape, weights: WeightList.groupNorm(c)) { b, p in
                    try b.affine(p[0], prefix + ".group_norm")
                }
                memory.free(g)
            }
            let t = try memory.register([1, s, c])
            memory.dispatch(.tokens, [n, t], constants: [UInt32(c), UInt32(s)], grid: MTLSize(width: c, height: s, depth: 1))
            memory.free(n)
            let qkv = try ["to_q", "to_k", "to_v"].map { name in
                try whole([t], [1, s, c], weights: WeightList.projection(c, c)) { b, p in try b.projection(p[0], prefix + "." + name) }
            }
            memory.free(t)
            // MPSGraph's SDPA copies its queries into the workspace: `S·C` floats, asked in whole
            // 2 MiB (`VAELayout.sdpaAsk`); and four
            // more copies of its own, beside our heaps (`VAELayout.sdpaOutside`).
            memory.outside(VAELayout.sdpaOutside(tokens: s, channels: c))
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
            let a = try whole(qkv, [1, s, c], weights: [], workspace: max(VAELayout.minimumAsk / 4, VAELayout.sdpaAsk(tokens: s, channels: c))) { b, p in
                b.scaledDotProduct(p[0], p[1], p[2], channels: c, requestBlock: EngineSettings.effective.vaeRequestBlock)
            }
            memory.free(qkv)
            let o = try whole([a], [1, s, c], weights: WeightList.projection(c, c)) { b, p in try b.projection(p[0], prefix + ".to_out.0") }
            memory.free(a)
            memory.dispatch(.untokensAdd, [x, o], constants: [UInt32(c), UInt32(s)], grid: MTLSize(width: s, height: c, depth: 1))
            memory.free(o)
        }
    }
}

// ── the decoders' precision ──────────────────────────────────────────────────────────────────────

/// **Whether a decoder computes in bf16** — the precision rule: never under the published weights, so only a
/// decoder whose weights are **published in bf16** (Z-Image's Flux decoder, the Qwen-Image decoder of Anima and Krea 2,
/// Qwen-Image-2.1's; not MOSS-Image's FLUX.2, published in fp32), and only where the plan says so (`CapacityPlan.Resources.halfDecoder`:
/// the map measured MPSGraph's convolution faster on bf16 operands, `CapacityModel.Convolution`; never with the accelerators off).
///
/// In bf16, as a card computing in bf16 does: the activations and the weights bf16 between two operations, every sum and every
/// norm's statistics in fp32 (MPSGraph's convolutions and products on bf16 operands; the norms kernels of ours, fp32 inside,
/// rounded once at their output); the latent the decoder reads and the pipeline's denormalization in fp32. Judged by each decoder
/// stage against the reference decoded in bf16 (invariant 3: no further from the fp64 witness than diffusers in bf16, + 25 %).
package enum VAEPrecision {
    /// The plan a decoder runs under: the render's (`MemoryPlan.current`, under its accelerators switch), the process's outside one.
    static var plan: CapacityPlan { MemoryPlan.current?.request.ditPlan ?? EngineSettings.effective.plan }

    /// **For the memory need**: `published` the decoder's published format (each decoder's `publishedInBF16`, the family's files
    /// fixed by their sha256), `plan` the request's.
    package static func half(plan: CapacityPlan, publishedInBF16 published: Bool) -> Bool { published && plan.resources.halfDecoder }

    /// **For a decoding**: the same, its file read — a tensor of the decoder not published in bf16 keeps it in fp32.
    static func half(publishedInBF16 published: Bool, weights: Safetensors) -> Bool {
        half(plan: plan, publishedInBF16: published) && VAEBuildingBlocks.decoderInBF16(weights)
    }
}
