import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import Foundation

/// **Qwen-Image-2.1's VAE** — `AutoencoderKLQwenImage21`, decoder and encoder, in fp32, through `MPSGraph`.
///
///     decoder   latent [64, h, w] ─ ×std + mean (CPU) ─ post_quant_conv ─ conv_in (64 → 1152)
///                 ─ mid(resnet, attention, resnet) @1152
///                 ─ 5 up blocks  1152 → 1152 → 1152 → 576 → 288 → 144   (×2 after each but the last)
///                 ─ RMSNorm ─ SiLU ─ conv_out (144 → 4) ─ clamp ─ image RGBA [4, 16h, 16w]
///
///     encoder   image RGBA [4, H, W] ─ conv_in (4 → 96)
///                 ─ 5 down blocks  96 → 96 → 192 → 384 → 768 → 768   (÷2 after each but the last)
///                 ─ mid(resnet, attention, resnet) @768 ─ RMSNorm ─ SiLU ─ conv_out (768 → 128)
///                 ─ quant_conv ─ moments [mean 64 | log-variance 64] ─ (mean − m_c) / s_c (CPU)
///
/// **One block, encoder or decoder** (`is_residual`): the Wan 2.1 blocks plus a shortcut
/// **without parameters** from the block's input,
///
///     y = resample( resnet₂( resnet₁( resnet₀(x) ) ) ) + shortcut(x)
///     resnet(x) = conv₂(SiLU(RMS(conv₁(SiLU(RMS(x)))))) + conv_shortcut(x)     RMS = F.normalize · √C · γ
///     shortcut  = AvgDown3D (encoder: space-to-depth, group mean) · DupUp3D (decoder: channel repeat,
///                 depth-to-space)
///
/// ## One frame: a 2D model, exactly
///
/// As for Anima (`AnimaVAE`), an image is a video of one frame, and the decoder and encoder run it
/// with a cleared feature cache. Here the reference **itself** says so: `QwenImage21CausalConv3d`
/// subclasses `nn.Conv2d`, squeezes the time axis, pads space symmetrically, and *refuses* a cache.
/// The published kernels are 4D (`[O, I, 3, 3]`), in **fp32** — no slice to take, nothing to widen.
/// `upsample3d` / `downsample3d` skip their `time_conv` on the first chunk (published, never read).
///
/// What sets it apart from Anima's VAE (Qwen-Image, the same Wan family), and what a port by
/// analogy would miss — read in `autoencoder_kl_qwenimage21.py` and `pipeline_qwenimage21.py`
/// (diffusers `80c7ed26`):
///
///   - **16× and 64 channels**: five levels, four resamplings; `z_dim` 64, moments 128. The encoder
///     is 96 wide at the bottom (`base_dim`), the decoder **144** (`decoder_base_dim`): the two
///     halves are not mirrors, `dims = 144 × [8, 8, 8, 4, 2, 1]` on the way up;
///   - **the upsampler does not halve the channels** (`upsample_out_dim = out_dim` in the residual
///     up block): `Conv2d(C, C)`, where Anima's is `Conv2d(C, C/2)`. And it is `upsampler.resample.1`,
///     singular, not `upsamplers.0`;
///   - **every block has a parameter-free shortcut** (`is_residual`), and its **temporal** factor
///     still acts on one frame — this is the trap of the model:
///       · `AvgDown3D` pads time with a zero frame **in front** (`pad_t = 1`), so in the temporal
///         blocks (1, 2, 3) the zero frame takes the *even* output channels: **output `o` even is
///         exactly 0, `o` odd is the 2×2 mean of input channel `(o − 1)/2`**. Taking the data at
///         `t = 0` would feed the even channels — a plausible latent, and wrong;
///       · `DupUp3D` (`first_chunk=True`) keeps the **last** time slot `t = ft − 1`: output `o` at
///         `(2y + i, 2x + j)` reads input channel `⌊(o·ft·4 + (ft − 1)·4 + 2i + j) / r⌋`, `r` the
///         repeat — `o` for blocks 0–1, **`2o + 1`** for block 2, **`2o + i`** (rows alternating
///         between two channels) for block 3 (`dupUpSources`);
///       · the encoder's last block (no downsampling) has the identity for shortcut;
///   - **RGBA in, RGBA out** (`in_channels` = `out_channels` = 4). The pipeline converts every
///     reference to RGBA (alpha 255 for an opaque photo, i.e. +1 after `[-1, 1]`), and returns an
///     **RGBA** PIL image (`numpy_to_pil` of 4 channels); the PNG written by the oracle drops the alpha
///     with `convert("RGB")`, **without compositing**. (Only the vision encoder's copy of a reference
///     is composited over white — not the VAE's.) The product does the same: the RGB of the
///     decoded RGBA (`QwenImage21DecodingModule`);
///   - **no 2×2 packing** (unlike FLUX.2): `_pack_latents` is a plain flatten, `[64, h, w]` ↔
///     `[h·w, 64]` (`pack` / `unpack`), and the normalization is per channel with the 64 `latents_mean`
///     / `latents_std` of `vae/config.json`, on the raw latent;
///   - the encoder returns the **mean** (`_encode_vae_image`, `sample_mode="argmax"`): the reference
///     itself is deterministic here;
///   - the bottleneck attention is 1152 (decoder) / 768 (encoder) wide, one head, scale `1/√C`.
///
/// ## Memory: built stage by stage, run stage by stage
///
/// The decoder weighs **1.1 GB of fp32 weights**, 85 % of them in the 1152-wide low-resolution blocks,
/// and its activations are at their largest at the end (288 channels at full resolution: 1.2 GB at
/// 1024²). Holding both at once is what to avoid: each stage is a separate graph **built just before it
/// runs and released after** (its constants with it), so the full-resolution stages only carry
/// their own few MB of weights. And after the bottleneck the decoder is **purely local** (`RMS`
/// normalizes each pixel over its channels — no spatial statistic, unlike Flux's `GroupNorm`): the
/// tail — the last two upsamplings and what follows them — runs **by horizontal bands with a margin,
/// exactly** (`tail`). Measured at 1024²: one graph per stage without bands 7.1 GB, the last level in
/// bands 4.7 GB, the last two 3.3 GB (1024×1536: 6.8 → 3.5 GB). The bands render **the same bits**
/// (any band size: one fingerprint), and the margin is the tight one: with the last level alone, 3
/// rows instead of 4 fell to 96.7 dB.
package enum QwenImage21VAE {
    /// `vae/config.json`, the parts that shape the graph. Defaults: Qwen-Image-2.1 as published.
    package struct Config: Sendable, Equatable {
        package var zDim = 64, baseDim = 96, decoderBaseDim = 144, resBlocks = 2
        package var dimMult = [1, 2, 4, 8, 8]
        package var temporalDownsample = [false, true, true, true]
        package var inChannels = 4, outChannels = 4
        package var mean: [Float] = [
            0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.6510, 0.2732,
            0.7562, -0.6163, -1.0277, 3.8363, 2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
            -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555,
            0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729,
            -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632,
            0.5547, -1.7121, -0.1562, 0.0304]
        package var std: [Float] = [
            3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910, 3.0785, 3.6500, 3.9308, 3.0904,
            2.8778, 3.7675, 3.7320, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
            3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467, 3.9037, 3.2947, 5.0770, 3.5075,
            3.2700, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
            3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959, 3.8747, 3.7608, 3.5735, 3.1490,
            3.7662, 3.6746, 3.4563, 3.8161]

        package init() {}

        /// Reads the published `config.json`; refuses what this graph does not build (no residual
        /// blocks, attention outside the bottleneck, a `patch_size`).
        package init(json: Data) throws {
            guard let c = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
                throw Safetensors.Failure.badHeader("VAE config: not a JSON object")
            }
            func floats(_ key: String) -> [Float]? { (c[key] as? [NSNumber])?.map { $0.floatValue } }
            zDim = c["z_dim"] as? Int ?? zDim
            baseDim = c["base_dim"] as? Int ?? baseDim
            decoderBaseDim = c["decoder_base_dim"] as? Int ?? baseDim
            resBlocks = c["num_res_blocks"] as? Int ?? resBlocks
            dimMult = c["dim_mult"] as? [Int] ?? dimMult
            temporalDownsample = c["temperal_downsample"] as? [Bool] ?? temporalDownsample
            inChannels = c["in_channels"] as? Int ?? inChannels
            outChannels = c["out_channels"] as? Int ?? outChannels
            mean = floats("latents_mean") ?? mean
            std = floats("latents_std") ?? std
            let residual = c["is_residual"] as? Bool ?? true
            let attention = (c["attn_scales"] as? [Any])?.isEmpty ?? true
            let patch = c["patch_size"].map { $0 is NSNull } ?? true
            guard residual, attention, patch, mean.count == zDim, std.count == zDim,
                  temporalDownsample.count == dimMult.count - 1 else {
                throw Safetensors.Failure.badHeader("VAE config: not an AutoencoderKLQwenImage21 this graph builds")
            }
        }

        /// The published `vae.json` next to the weights, if there is one; the defaults otherwise.
        package static func beside(_ weightsPath: String) throws -> Config {
            let json = URL(fileURLWithPath: weightsPath).deletingLastPathComponent().appendingPathComponent("vae.json")
            guard let data = try? Data(contentsOf: json) else { return Config() }
            return try Config(json: data)
        }

        /// Pixels per latent cell: one ×2 per level but the last.
        package var factor: Int { 1 << (dimMult.count - 1) }
        /// `[96, 96, 192, 384, 768, 768]`: the encoder's widths, `base_dim × [1] + dim_mult`.
        package var encoderDims: [Int] { ([1] + dimMult).map { baseDim * $0 } }
        /// `[1152, 1152, 1152, 576, 288, 144]`: the decoder's, `decoder_base_dim × [last] + reversed`.
        package var decoderDims: [Int] { ([dimMult.last!] + dimMult.reversed()).map { decoderBaseDim * $0 } }

        /// Block `i` of the encoder: widths, and the shortcut's `(temporal, spatial)` factors.
        /// The last block downsamples neither in space nor in time.
        package func down(_ i: Int) -> (input: Int, output: Int, temporal: Int, spatial: Int) {
            let last = i == dimMult.count - 1
            return (encoderDims[i], encoderDims[i + 1], !last && temporalDownsample[i] ? 2 : 1, last ? 1 : 2)
        }

        /// Block `i` of the decoder. `temperal_upsample` is `temperal_downsample` **reversed**.
        package func up(_ i: Int) -> (input: Int, output: Int, temporal: Int, spatial: Int) {
            let last = i == dimMult.count - 1
            let temporalUp = Array(temporalDownsample.reversed())
            return (decoderDims[i], decoderDims[i + 1], !last && temporalUp[i] ? 2 : 1, last ? 1 : 2)
        }
    }

    // ── pure functions: the latent layout and the shortcuts' index arithmetic ─────────────────

    /// `_pack_latents`: planar `[C, h, w]` → the DiT's tokens `[h·w, C]` (row-major grid).
    package static func pack(_ planar: [Float], channels: Int, height: Int, width: Int) -> [Float] {
        let plane = height * width
        var tokens = [Float](repeating: 0, count: planar.count)
        for c in 0..<channels { for p in 0..<plane { tokens[p * channels + c] = planar[c * plane + p] } }
        return tokens
    }

    /// `_unpack_latents`: tokens `[h·w, C]` → planar `[C, h, w]`.
    package static func unpack(_ tokens: [Float], channels: Int, height: Int, width: Int) -> [Float] {
        let plane = height * width
        var planar = [Float](repeating: 0, count: tokens.count)
        for c in 0..<channels { for p in 0..<plane { planar[c * plane + p] = tokens[p * channels + c] } }
        return planar
    }

    /// The pipeline's de-normalization, `z · std_c + mean_c`, on the CPU: two roundings, like
    /// `torch` (a GPU graph may fuse them into one FMA). Planar `[C, h, w]`.
    package static func denormalize(_ z: [Float], config: Config) -> [Float] {
        let plane = z.count / config.zDim
        var out = z
        for c in 0..<config.zDim {
            let (s, m) = (config.std[c], config.mean[c])
            for p in (c * plane)..<((c + 1) * plane) { let scaled = z[p] * s; out[p] = scaled + m }
        }
        return out
    }

    /// `_encode_vae_image`'s normalization, `(z − mean_c) / std_c`, planar `[C, h, w]`.
    package static func normalize(_ z: ArraySlice<Float>, config: Config) -> [Float] {
        let plane = z.count / config.zDim
        var out = [Float](repeating: 0, count: z.count)
        let base = z.startIndex
        for c in 0..<config.zDim {
            let (s, m) = (config.std[c], config.mean[c])
            for p in (c * plane)..<((c + 1) * plane) { out[p] = (z[base + p] - m) / s }
        }
        return out
    }

    /// **`DupUp3D` on one frame**, as an index table: for output channel `o` and the offset `(i, j)`
    /// of its 2×2 cell (`spatial = 2`), the input channel it copies — entry `(o·s + i)·s + j`.
    ///
    /// `repeat_interleave(r)` makes channel `c'` a copy of `⌊c'/r⌋`; the view `[out, ft, s, s]` reads
    /// `c' = ((o·ft + t)·s + i)·s + j`; `first_chunk` keeps `t = ft − 1`.
    package static func dupUpSources(input: Int, output: Int, temporal ft: Int, spatial s: Int) -> [Int] {
        let r = output * ft * s * s / input
        var table: [Int] = []
        for o in 0..<output { for i in 0..<s { for j in 0..<s {
            table.append((((o * ft + ft - 1) * s + i) * s + j) / r)
        } } }
        return table
    }

    /// **`AvgDown3D` on one frame**, as membership: for output channel `o`, the `(c, i, j)` input
    /// samples of its group, `nil` for a member drawn from the zero frame the reference pads in front.
    /// The group mean divides by **all** members, zeros included. (For the tests: the graph builds
    /// the reference's own view / permute / mean, `avgDown`.)
    package static func avgDownGroups(input: Int, output: Int, temporal ft: Int, spatial s: Int)
            -> [[(c: Int, i: Int, j: Int)?]] {
        let factor = ft * s * s, g = input * factor / output
        return (0..<output).map { o in
            (0..<g).map { k in
                let flat = o * g + k
                let (c, r) = (flat / factor, flat % factor)
                let t = r / (s * s)
                return t < ft - 1 ? nil : (c, (r % (s * s)) / s, r % s)
            }
        }
    }

    // ── the graphs ──────────────────────────────────────────────────────────────────────────

    /// A graph built on demand: it reads the registers `inputs` (in order) and writes `output`.
    /// `forget` names the registers released after it ran.
    struct Stage {
        var inputs: [String]
        var output: String
        var forget: [String] = []
        /// Keep the first input under this name too (no copy): a block's input, for its shortcut.
        var keep: String? = nil
        /// For the checks' per-stage footprint.
        var label: String? = nil
        var build:(VAEBuildingBlocks, [MPSGraphTensor]) throws -> MPSGraphTensor
    }

    /// Called after each stage with its label — the checks read the footprint there. Set by a
    /// single-threaded check before decoding, never by the product.
    nonisolated(unsafe) package static var stageDone: ((String) -> Void)?

    /// A tensor between two stages, in a buffer of ours (shared memory: the banded tail reads its rows).
    struct Register {
        let data: MPSGraphTensorData
        let buffer: MTLBuffer
        init(_ buffer: MTLBuffer, shape: [NSNumber]) {
            self.buffer = buffer
            data = MPSGraphTensorData(buffer, shape: shape, dataType: .float32)
        }
        init(_ values: [Float], shape: [NSNumber], device: MTLDevice) throws {
            guard let buffer = device.makeBuffer(length: values.count * 4, options: .storageModeShared) else {
                throw GEMM.Failure.noDevice
            }
            values.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            self.init(buffer, shape: shape)
        }
        var shape: [Int] { data.shape.map(\.intValue) }
        var values: UnsafeMutablePointer<Float> { buffer.contents().assumingMemoryBound(to: Float.self) }
    }

    /// Builds, runs and releases the stages one after the other. A stage's graph — and its weight
    /// constants — die before the next is built; a register lives until a stage forgets it.
    static func run(_ stages: [Stage], weights: Safetensors, name: String, device: MTLDevice, queue: MTLCommandQueue,
                    registers initial: [String: Register]) throws -> [String: Register] {
        var registers = initial
        for stage in stages {
            try autoreleasepool {
                let graph = MPSGraph()
                let feeds = stage.inputs.map { registers[$0]! }
                let placeholders = feeds.map { graph.placeholder(shape: $0.data.shape, dataType: .float32, name: nil) }
                let b = VAEBuildingBlocks(graph: graph, weights: weights, name: name)
                let output = try stage.build(b, placeholders)
                // The result goes into a buffer of ours, which the next stage — or the banded tail — reads.
                let shape = output.shape!
                guard let buffer = device.makeBuffer(length: shape.reduce(4) { $0 * $1.intValue },
                                                     options: .storageModeShared) else { throw GEMM.Failure.noDevice }
                let result = Register(buffer, shape: shape)
                execute(graph, queue: queue, feeds: Dictionary(uniqueKeysWithValues: zip(placeholders, feeds.map(\.data))),
                        results: [output: result.data])
                if let keep = stage.keep { registers[keep] = feeds[0] }
                registers[stage.output] = result
                for f in stage.forget { registers[f] = nil }
            }
            stageDone?(stage.label ?? stage.output)
        }
        return registers
    }

    /// **`graph.run`, without MPS's heap cache.** MPS keeps the heaps of a graph's intermediates for
    /// several seconds after the run (between 3 and 15 s, measured), in case the next graph wants
    /// them: 0.5 GB per image size for this encoder — 1.4 GB of Metal still allocated when the DiT
    /// starts right after the references' encoding, at the moment it allocates its own. Duration 0:
    /// the heaps go back with the command buffer. The same graph, encoded the same way: same bits.
    static func execute(_ graph: MPSGraph, queue: MTLCommandQueue, feeds: [MPSGraphTensor: MPSGraphTensorData],
                        results: [MPSGraphTensor: MPSGraphTensorData]) {
        let commands = MPSCommandBuffer(from: queue)
        MPSSetHeapCacheDuration(commands, 0)
        graph.encode(to: commands, feeds: feeds, targetOperations: nil, resultsDictionary: results, executionDescriptor: nil)
        commands.commit()
        commands.waitUntilCompleted()
    }

    static func device() throws -> (MTLDevice, MTLCommandQueue) {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw GEMM.Failure.noDevice
        }
        return (device, queue)
    }
}

// ── the decoder ─────────────────────────────────────────────────────────────────────────────

package final class QwenImage21VAEDecoder {
    package let config: QwenImage21VAE.Config
    package let latentHeight, latentWidth: Int
    package var imageHeight: Int { latentHeight * config.factor }
    package var imageWidth: Int { latentWidth * config.factor }
    private let weights: Safetensors
    private let device: MTLDevice
    private let queue: MTLCommandQueue

    /// - Parameter path: the published `vae/diffusion_pytorch_model.safetensors` (fp32), read as is;
    ///   its `vae.json`, if it lies beside, gives the config.
    package init(path: String, latentHeight: Int, latentWidth: Int, config: QwenImage21VAE.Config? = nil) throws {
        let config = try config ?? QwenImage21VAE.Config.beside(path)
        let weights = try Safetensors(path: path)
        guard weights.entries["decoder.conv_in.weight"]?.shape == [config.decoderDims[0], config.zDim, 3, 3],
              weights.entries["decoder.conv_out.weight"]?.shape == [config.outChannels, config.decoderDims.last!, 3, 3] else {
            throw Safetensors.Failure.badHeader("\(path): not the Qwen-Image-2.1 VAE")
        }
        (device, queue) = try QwenImage21VAE.device()
        self.config = config; self.weights = weights
        self.latentHeight = latentHeight; self.latentWidth = latentWidth
    }

    /// `latent`: planar `[64, h, w]`, **normalized** (the DiT's space). Returns RGBA `[4, 16h, 16w]` in
    /// `[-1, 1]` and the seconds (the de-normalization included).
    package func decode(latent: [Float]) throws -> (rgba: [Float], seconds: Double) {
        precondition(latent.count == config.zDim * latentHeight * latentWidth, "QwenImage21VAEDecoder: latent shape")
        let started = Date()
        let z = QwenImage21VAE.denormalize(latent, config: config)
        let input = try QwenImage21VAE.Register(z, shape: [1, NSNumber(value: config.zDim), NSNumber(value: latentHeight),
                                                           NSNumber(value: latentWidth)], device: device)
        let registers = try QwenImage21VAE.run(stages(), weights: weights, name: "VAE Qwen-Image-2.1", device: device,
                                               queue: queue, registers: ["x": input])
        let rgba = try tail(x: registers["x"]!, skip: registers["skip"]!)
        return (rgba, Date().timeIntervalSince(started))
    }

    /// The decoder's stages up to the last upsampling, excluded: the bottleneck, then **one graph per
    /// resnet**, per upsampling conv and per shortcut sum. `skip` is a block's input, kept for its shortcut.
    private func stages() -> [QwenImage21VAE.Stage] {
        typealias Stage = QwenImage21VAE.Stage
        let config = self.config
        let dims = config.decoderDims
        let L = (h: latentHeight, w: latentWidth)
        var stages: [Stage] = []
        stages.append(Stage(inputs: ["x"], output: "x", label: "mid") { b, x in
            var x = try b.conv(x[0], "post_quant_conv", padding: 0)
            x = try b.conv(x, "decoder.conv_in", padding: 1)
            x = try b.resnetQwen(x, "decoder.mid_block.resnets.0", inChannels: dims[0], outChannels: dims[0])
            x = try b.bottleneck(x, "decoder.mid_block.attentions.0", channels: dims[0], size: L)
            return try b.resnetQwen(x, "decoder.mid_block.resnets.1", inChannels: dims[0], outChannels: dims[0])
        })
        var scale = 1
        for i in 0..<(config.dimMult.count - 1) {
            let block = config.up(i), prefix = "decoder.up_blocks.\(i)"
            let size = (h: scale * L.h, w: scale * L.w)
            for r in 0...config.resBlocks {
                let entry = r == 0 ? block.input : block.output
                // The block's input is kept under `skip` by its first resnet: the shortcut reads it at the end.
                stages.append(Stage(inputs: ["x"], output: "x", keep: r == 0 ? "skip" : nil, label: "\(i).resnet\(r)") { b, x in
                    try b.resnetQwen(x[0], "\(prefix).resnets.\(r)", inChannels: entry, outChannels: block.output)
                })
            }
            if i == config.dimMult.count - 1 - QwenImage21VAEDecoder.tailLevels { break }   // the last upsamplings belong to the tail
            stages.append(Stage(inputs: ["x"], output: "x", label: "\(i).upsample") { b, x in
                try b.conv(b.double(x[0], channels: block.output, size: size), "\(prefix).upsampler.resample.1", padding: 1)
            })
            stages.append(Stage(inputs: ["x", "skip"], output: "x", forget: ["skip"], label: "\(i).shortcut") { b, x in
                try b.addDupUp(x[0], skip: x[1], input: block.input, output: block.output,
                               temporal: block.temporal, size: size)
            })
            scale *= 2
        }
        return stages
    }

    /// The full-resolution pixels of a band of the tail, at most (256 rows of 1024): its intermediates —
    /// 288 channels at full resolution — scale with it. Settable by the checks, to force bands where one
    /// piece would do.
    nonisolated(unsafe) package static var bandPixels = 1 << 18

    /// **The upsamplings in the banded tail: the last two.** With the last one only, the stages at half
    /// resolution (the ×2 before it, its shortcut sum, the four resnets of the block between) ran in one
    /// piece, and they held the decoder's peak: at 1024×1536, **6.8 GB** of `phys_footprint` at the
    /// block's first resnet (graphics memory, 4.3 GB of it still owned by the process from the shortcut
    /// just before, released a second later), 4.7 GB at 1024². In the tail: **3.5 GB and 3.3 GB**, the same
    /// bits (512², 608×416, 1024², 1024×1536 at every band size down to 2 rows, one fingerprint each), the
    /// same time. A third level would band stages whose peak (2.9 GB) is already below the tail's.
    package static let tailLevels = 2

    /// **The tail, by horizontal bands — exact.** From the second-to-last upsampling to `conv_out`,
    /// everything is local: convolutions, per-pixel RMSNorm, the shortcut's copies. A full-resolution row
    /// depends on the rows within **8 rows** of it at full resolution (the last upsampling conv, the six
    /// of the last block, `conv_out`), i.e. 4 at half resolution; to which the half-resolution level adds
    /// 7 (its upsampling conv, the six of its block): 11, i.e. **6 rows of the tail's input**. Each band
    /// reads its rows plus that margin on each side (none past the image's edge, where the model's own
    /// zero padding is the right one), and keeps only its core: every kept pixel sees exactly the inputs
    /// it sees in one piece. All windows have the same height (the last one slides back inside the
    /// image): a single graph serves them all. The skip of the inner block (its input, which its shortcut
    /// reads) is born inside the band.
    ///
    /// Why: in one piece at 1024², the last level alone held **7.1 GB** (`MPSGraph` keeps a stage's
    /// intermediates, and keeps them cached after it). The objection to tiling Flux's decoder — tiles
    /// change `GroupNorm`'s statistics — does not apply: there is no statistic across pixels here.
    private func tail(x: QwenImage21VAE.Register, skip: QwenImage21VAE.Register) throws -> [Float] {
        let levels = QwenImage21VAEDecoder.tailLevels, first = config.dimMult.count - 1 - levels
        let (cx, h, w) = (x.shape[1], x.shape[2], x.shape[3]), cs = skip.shape[1]
        let factor = 1 << levels
        // The margin, back from the output: `conv_out`, the last block's resnets and its upsampling conv;
        // then, per level below, ×2 (rounded up), that block's resnets and upsampling conv.
        let convsPerBlock = 2 * (config.resBlocks + 1)
        var margin = 1 + convsPerBlock + 1
        for _ in 1..<max(1, levels) { margin = (margin + 1) / 2 + convsPerBlock + 1 }
        margin = (margin + 1) / 2
        let bandPixels = MemoryPlan.decoderBand(standard: QwenImage21VAEDecoder.bandPixels,
                                                minimum: MemoryPlan.qwenMinimumBand, unit: "pixels")
        let core = max(1, bandPixels / (factor * factor * w))
        let window = min(h, core + 2 * margin)
        let (H, W) = (factor * h, factor * w)
        let config = self.config
        var rgba = [Float](repeating: 0, count: config.outChannels * H * W)
        func buffer(_ channels: Int, _ rows: Int, _ columns: Int) throws -> QwenImage21VAE.Register {
            guard let b = device.makeBuffer(length: channels * rows * columns * 4, options: .storageModeShared) else {
                throw GEMM.Failure.noDevice
            }
            return QwenImage21VAE.Register(b, shape: [1, NSNumber(value: channels), NSNumber(value: rows), NSNumber(value: columns)])
        }
        let xw = try buffer(cx, window, w), sw = try buffer(cs, window, w)
        let out = try buffer(config.outChannels, factor * window, W)
        try autoreleasepool {
            let graph = MPSGraph()
            let b = VAEBuildingBlocks(graph: graph, weights: weights, name: "VAE Qwen-Image-2.1")
            let px = graph.placeholder(shape: xw.data.shape, dataType: .float32, name: nil)
            let ps = graph.placeholder(shape: sw.data.shape, dataType: .float32, name: nil)
            var y = px, skipped = ps
            for i in first..<(first + levels) {
                let block = config.up(i), next = config.up(i + 1), scale = 1 << (i - first)
                let size = (h: scale * window, w: scale * w)
                y = try b.conv(b.double(y, channels: block.output, size: size), "decoder.up_blocks.\(i).upsampler.resample.1",
                               padding: 1)
                y = try b.addDupUp(y, skip: skipped, input: block.input, output: block.output, temporal: block.temporal, size: size)
                skipped = y   // the next block's input, which its shortcut reads
                for r in 0...config.resBlocks {
                    y = try b.resnetQwen(y, "decoder.up_blocks.\(i + 1).resnets.\(r)",
                                         inChannels: r == 0 ? next.input : next.output, outChannels: next.output)
                }
            }
            y = try b.silu(b.rmsNorm(y, "decoder.norm_out", channels: config.up(first + levels).output))
            y = try b.conv(y, "decoder.conv_out", padding: 1)
            y = graph.clamp(y, min: graph.constant(-1, dataType: .float32), max: graph.constant(1, dataType: .float32), name: nil)

            var c0 = 0
            while c0 < h {
                let c1 = min(h, c0 + core)
                let start = min(max(0, c0 - margin), h - window)
                for (source, target, channels) in [(x, xw, cx), (skip, sw, cs)] {
                    for c in 0..<channels {
                        (target.values + c * window * w).update(from: source.values + (c * h + start) * w, count: window * w)
                    }
                }
                autoreleasepool {
                    QwenImage21VAE.execute(graph, queue: queue, feeds: [px: xw.data, ps: sw.data], results: [y: out.data])
                }
                // Keep the core: full-resolution rows [F·c0, F·c1), found at F·(c0 − start) in the window.
                rgba.withUnsafeMutableBufferPointer { target in
                    for c in 0..<config.outChannels {
                        (target.baseAddress! + (c * H + factor * c0) * W)
                            .update(from: out.values + (c * factor * window + factor * (c0 - start)) * W,
                                    count: factor * (c1 - c0) * W)
                    }
                }
                QwenImage21VAE.stageDone?("tail \(c0)-\(c1)")
                c0 = c1
            }
        }
        return rgba
    }
}

// ── the encoder ─────────────────────────────────────────────────────────────────────────────

package final class QwenImage21VAEEncoder {
    package struct Output {
        /// `[64, H/16, W/16]`, normalized: what the DiT receives (before packing).
        package let latent: [Float]
        /// `[128, H/16, W/16]`: mean then log-variance, raw.
        package let moments: [Float]
        package let seconds: Double
    }

    package let config: QwenImage21VAE.Config
    package let height, width: Int
    package var latentHeight: Int { height / config.factor }
    package var latentWidth: Int { width / config.factor }
    private let weights: Safetensors
    private let device: MTLDevice
    private let queue: MTLCommandQueue

    package init(path: String, height: Int, width: Int, config: QwenImage21VAE.Config? = nil) throws {
        let config = try config ?? QwenImage21VAE.Config.beside(path)
        let weights = try Safetensors(path: path)
        guard weights.entries["encoder.conv_in.weight"]?.shape == [config.encoderDims[0], config.inChannels, 3, 3],
              weights.entries["quant_conv.weight"]?.shape == [2 * config.zDim, 2 * config.zDim, 1, 1] else {
            throw Safetensors.Failure.badHeader("\(path): not the Qwen-Image-2.1 VAE encoder")
        }
        guard height % config.factor == 0, width % config.factor == 0 else {
            throw EngineError.formatRefused(width: width, height: height, reason: .notMultiple)
        }
        (device, queue) = try QwenImage21VAE.device()
        self.config = config; self.weights = weights; self.height = height; self.width = width
    }

    /// `rgba`: `[4, height, width]` planar in `[-1, 1]` — alpha included (+1 for an opaque image).
    package func encode(rgba: [Float]) throws -> Output {
        precondition(rgba.count == config.inChannels * height * width, "QwenImage21VAEEncoder: image shape")
        let started = Date()
        let input = try QwenImage21VAE.Register(rgba, shape: [1, NSNumber(value: config.inChannels), NSNumber(value: height),
                                                              NSNumber(value: width)], device: device)
        let out = try QwenImage21VAE.run(stages(), weights: weights, name: "VAE Qwen-Image-2.1 encoder",
                                         device: device, queue: queue, registers: ["x": input])["x"]!
        let moments = Array(UnsafeBufferPointer(start: out.values, count: out.shape.reduce(1, *)))
        let mean = moments.prefix(config.zDim * latentHeight * latentWidth)
        return Output(latent: QwenImage21VAE.normalize(mean, config: config), moments: moments,
                      seconds: Date().timeIntervalSince(started))
    }

    /// One graph per block (the block's input is a placeholder, its shortcut is internal), the head
    /// with the last: the encoder is widest at full resolution with only 96 channels.
    private func stages() -> [QwenImage21VAE.Stage] {
        let config = self.config
        let dims = config.encoderDims
        let blocks = config.dimMult.count
        var size = (h: height, w: width)
        var stages: [QwenImage21VAE.Stage] = []
        for i in 0..<blocks {
            let block = config.down(i), prefix = "encoder.down_blocks.\(i)", here = size
            stages.append(QwenImage21VAE.Stage(inputs: ["x"], output: "x", label: "block \(i)") { b, x in
                var x = x[0]
                if i == 0 { x = try b.conv(x, "encoder.conv_in", padding: 1) }
                let skip = x
                for r in 0..<config.resBlocks {
                    x = try b.resnetQwen(x, "\(prefix).resnets.\(r)", inChannels: r == 0 ? block.input : block.output,
                                         outChannels: block.output)
                }
                if block.spatial > 1 { x = try b.downsample(x, "\(prefix).downsampler.resample.1") }
                x = b.graph.addition(x, b.avgDown(skip, input: block.input, output: block.output,
                                                  temporal: block.temporal, spatial: block.spatial, size: here), name: nil)
                guard i == blocks - 1 else { return x }
                let top = (h: here.h / block.spatial, w: here.w / block.spatial)
                x = try b.resnetQwen(x, "encoder.mid_block.resnets.0", inChannels: dims.last!, outChannels: dims.last!)
                x = try b.bottleneck(x, "encoder.mid_block.attentions.0", channels: dims.last!, size: top)
                x = try b.resnetQwen(x, "encoder.mid_block.resnets.1", inChannels: dims.last!, outChannels: dims.last!)
                x = try b.silu(b.rmsNorm(x, "encoder.norm_out", channels: dims.last!))
                x = try b.conv(x, "encoder.conv_out", padding: 1)
                return try b.conv(x, "quant_conv", padding: 0)
            })
            size = (size.h / block.spatial, size.w / block.spatial)
        }
        return stages
    }
}

// ── the bricks only this VAE has ───────────────────────────────────────────────────────────

extension VAEBuildingBlocks {
    /// The Wan attention block: RMS (`images=True`, the same formula), one 1×1 `to_qkv` split q, k, v.
    fileprivate func bottleneck(_ x: MPSGraphTensor, _ prefix: String, channels c: Int,
                                size: (h: Int, w: Int)) throws -> MPSGraphTensor {
        try attention(x, normalized: rmsNorm(x, prefix + ".norm", channels: c), channels: c, size: size,
                      qkv: { (try projection($0, prefix + ".to_qkv", rows: 0..<c),
                              try projection($0, prefix + ".to_qkv", rows: c..<(2 * c)),
                              try projection($0, prefix + ".to_qkv", rows: (2 * c)..<(3 * c))) },
                      output: prefix + ".proj", requestBlock: EngineSettings.effective.vaeRequestBlock)
    }

    /// **`AvgDown3D` on one frame, written as the reference does it**: space-to-depth
    /// (`view` + `permute`), the zero frame in front when `temporal = 2`, then the mean over groups of
    /// `in·ft·s²/out` consecutive channels. `x`: `[1, in, H, W]` → `[1, out, H/s, W/s]`.
    fileprivate func avgDown(_ x: MPSGraphTensor, input: Int, output: Int, temporal ft: Int, spatial s: Int,
                             size: (h: Int, w: Int)) -> MPSGraphTensor {
        if ft == 1 && s == 1 && input == output { return x }   // a mean over groups of one
        let n = { (v: Int) in NSNumber(value: v) }
        let (h, w) = (size.h / s, size.w / s)
        var t = graph.reshape(x, shape: [1, n(input), n(h), n(s), n(w), n(s)], name: nil)
        t = graph.transpose(t, permutation: [0, 1, 3, 5, 2, 4], name: nil)          // [1, in, s, s, h, w]
        t = graph.reshape(t, shape: [1, n(input), n(s * s), n(h * w)], name: nil)
        if ft > 1 {
            let zeros = graph.constant(0, shape: [1, n(input), n((ft - 1) * s * s), n(h * w)], dataType: .float32)
            t = graph.concatTensors([zeros, t], dimension: 2, name: nil)              // t = 0 first: the padding
        }
        let g = input * ft * s * s / output
        t = graph.reshape(t, shape: [1, n(output), n(g), n(h * w)], name: nil)
        t = graph.mean(of: t, axes: [2], name: nil)
        return graph.reshape(t, shape: [1, n(output), n(h), n(w)], name: nil)
    }

    /// `main + DupUp3D(skip)`. `skip`: `[1, in, h, w]`; `main`: `[1, out, 2h, 2w]`. The source table
    /// (`QwenImage21VAE.dupUpSources`) never depends on the column offset `j` here (the repeat is even):
    /// the rows are gathered (`[1, out, h, 2, w]`, a gather skipped when it is the identity — block 3),
    /// and the columns come free by broadcasting in the sum. Pure copies: no rounding.
    fileprivate func addDupUp(_ main: MPSGraphTensor, skip: MPSGraphTensor, input: Int, output: Int,
                              temporal ft: Int, size: (h: Int, w: Int)) throws -> MPSGraphTensor {
        let s = 2
        let n = { (v: Int) in NSNumber(value: v) }
        let table = QwenImage21VAE.dupUpSources(input: input, output: output, temporal: ft, spatial: s)
        let rows = stride(from: 0, to: table.count, by: s).map { table[$0] }
        guard (0..<table.count).allSatisfy({ table[$0] == rows[$0 / s] }) else {
            throw Safetensors.Failure.badHeader("\(name): DupUp3D \(input) → \(output) depends on the column")
        }
        var g = skip
        if rows != Array(0..<rows.count) {
            let indices = rows.map { Int32($0) }.withUnsafeBytes { Data($0) }
            g = graph.gather(withUpdatesTensor: skip,
                             indicesTensor: graph.constant(indices, shape: [n(rows.count)], dataType: .int32),
                             axis: 1, batchDimensions: 0, name: nil)                  // [1, out·2, h, w]
        }
        g = graph.reshape(g, shape: [1, n(output), n(s), n(size.h), n(size.w)], name: nil)
        g = graph.transpose(g, permutation: [0, 1, 3, 2, 4], name: nil)               // [1, out, h, 2, w]
        g = graph.reshape(g, shape: [1, n(output), n(s * size.h), n(size.w), 1], name: nil)
        let m = graph.reshape(main, shape: [1, n(output), n(s * size.h), n(size.w), n(s)], name: nil)
        return graph.reshape(graph.addition(m, g, name: nil),
                             shape: [1, n(output), n(s * size.h), n(s * size.w)], name: nil)
    }
}

// ── the modules ─────────────────────────────────────────────────────────────────────────────

/// Qwen-Image-2.1's decoder: the RGBA it decodes, **alpha dropped without compositing** — what the
/// oracle's `pil.convert("RGB")` writes.
public struct QwenImage21DecodingModule: DecodingModule {
    public var path: String
    public var space: LatentSpace { .qwenImage21 }
    public var name: String { "VAE Qwen-Image-2.1" }
    public init(path: String) { self.path = path }

    public func decode(_ latent: Latent, context: Context) throws -> ImageRGB {
        let vae = try QwenImage21VAEDecoder(path: path, latentHeight: latent.height, latentWidth: latent.width)
        let (rgba, seconds) = try vae.decode(latent: latent.values)
        context.timings.decoding = seconds
        return ImageRGB(pixels: Array(rgba.prefix(3 * vae.imageHeight * vae.imageWidth)),
                        height: vae.imageHeight, width: vae.imageWidth)
    }
}

/// Qwen-Image-2.1's encoder, for the edit references and img2img: the image made **RGBA** (alpha +1,
/// as `convert("RGBA")` makes an opaque image), then `(mean − mean_c) / std_c`.
public struct QwenImage21EncodingModule: ImageEncodingModule {
    public var path: String
    public var space: LatentSpace { .qwenImage21 }
    public var name: String { "VAE Qwen-Image-2.1" }
    public init(path: String) { self.path = path }

    public func encoder(_ image: ImageRGB, context: Context) throws -> Latent {
        let encoder = try QwenImage21VAEEncoder(path: path, height: image.height, width: image.width)
        let output = try encoder.encode(rgba: QwenImage21EncodingModule.rgba(image))
        context.footprints.encoding = Arena.processFootprint()   // before the release
        return Latent(space: space, height: encoder.latentHeight, width: encoder.latentWidth, values: output.latent)
    }

    /// `[3, H, W]` → `[4, H, W]` with an opaque alpha: 255 → `255/255 · 2 − 1` = +1.
    package static func rgba(_ image: ImageRGB) -> [Float] {
        image.pixels + [Float](repeating: 1, count: image.height * image.width)
    }
}
