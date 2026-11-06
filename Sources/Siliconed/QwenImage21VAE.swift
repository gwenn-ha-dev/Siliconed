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
///   - **no 2×2 packing**: `_pack_latents` is a plain flatten, `[64, h, w]` ↔
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
///
/// **Every tensor in a heap of ours, by passes of one kernel** (`VAEMemory`), placed by
/// `QwenImage21VAEDecoder.plan` / `QwenImage21VAEEncoder.plan` — `decode` / `encode` walked without a
/// device —, each pass's weights fed from them rather than held as graph constants. A graph of several
/// operations kept its intermediates in Metal memory of MPSGraph's own (1.76 GB beside our heaps in the
/// tail at 1024², 1.6 GB in the encoder's first block): each resnet is now five passes (SiLU·RMS,
/// conv₁, SiLU·RMS, [shortcut], conv₂ + residual), the ×2, `DupUp3D`, `AvgDown3D` and the attention's
/// transposes are kernels of ours (`VAEKernels`) — **the same bits** (md5 of both halves at 1024², and
/// of the renders). What stays MPSGraph's: its SDPA's own copies, `VAELayout.sdpaOutside`.
package enum QwenImage21VAE {
    /// `vae/config.json`, the parts that shape the graph. Defaults: Qwen-Image-2.1 as published.
    package struct Config: Sendable, Hashable {
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
    /// The group mean divides by **all** members, zeros included.
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

    // ── the passes ──────────────────────────────────────────────────────────────────────────

    /// **One GPU pass**: a graph of one kernel built on demand (`build`), or a kernel of ours (`kernel`).
    /// It reads the registers `inputs` (in order) and writes `output`, of `shape` — a new register, or the
    /// first input's own (`inPlace`, kernels only). `forget` names the registers released after it ran.
    ///
    /// Why passes of one kernel: a graph of several operations put every intermediate but its
    /// largest in Metal memory of MPSGraph's own, beyond our heaps (1.76 GB in the decoder's tail at
    /// 1024², 1.6 GB in the encoder's first block). Cut where MPSGraph fused nothing across the cut —
    /// a convolution with its bias (and the residual sum), a norm with its SiLU —, each pass's only tensors
    /// are registers, and the bits are those of the fused graphs (decoder and encoder md5).
    struct Stage {
        var inputs: [String]
        var output: String
        var shape: [Int]
        /// The floats of each weight it reads, in order (`WeightList`).
        var weights: [Int] = []
        var forget: [String] = []
        /// Keep the first input under this name too (no copy): a block's input, for its shortcut.
        var keep: String? = nil
        /// For the checks' per-stage footprint.
        var label: String? = nil
        var inPlace = false
        /// What MPSGraph allocates beside our heaps for this graph, in bytes (`VAELayout.outside`).
        var outside = 0
        /// What MPS asks of the workspace heap for this graph, in floats (`VAEMemory.graph`).
        var workspace = VAELayout.minimumAsk / 4
        var build: ((VAEBuildingBlocks, [MPSGraphTensor]) throws -> MPSGraphTensor)? = nil
        var kernel: ((VAEMemory, [VAERegister], VAERegister) throws -> Void)? = nil
        /// **An RMSNorm pass** (`QwenImageRMS_norm` at `prefix`, then the SiLU if `silu`): in bf16 (`VAEMemory.half`) it runs as a
        /// kernel of ours, fp32 inside, rounded once (`VAEMemory.rmsNorm`); in fp32, `build`'s graph.
        var norm: (prefix: String, channels: Int, silu: Bool)? = nil

        /// A graph pass.
        init(_ inputs: [String], _ output: String, _ shape: [Int], weights: [Int] = [], forget: [String] = [],
             keep: String? = nil, label: String? = nil,
             build: @escaping (VAEBuildingBlocks, [MPSGraphTensor]) throws -> MPSGraphTensor) {
            self.inputs = inputs; self.output = output; self.shape = shape; self.weights = weights
            self.forget = forget; self.keep = keep; self.label = label; self.build = build
        }

        /// A kernel pass.
        init(_ inputs: [String], _ output: String, _ shape: [Int], inPlace: Bool, forget: [String] = [],
             label: String? = nil, kernel: @escaping (VAEMemory, [VAERegister], VAERegister) throws -> Void) {
            self.inputs = inputs; self.output = output; self.shape = shape; self.inPlace = inPlace
            self.forget = forget; self.label = label; self.kernel = kernel
        }
    }

    /// The graphs a run built, by pass: the tail's bands run the same passes on windows of one shape.
    final class Graphs {
        var built: [Int: (graph: MPSGraph, placeholders: [MPSGraphTensor], output: MPSGraphTensor, fed: FedWeights)] = [:]
    }

    /// Called after each labelled pass — the checks read the footprint there. Set by a
    /// single-threaded check before decoding, never by the product.
    nonisolated(unsafe) package static var stageDone: ((String) -> Void)?

    /// **Runs the passes one after the other**, every register in `memory`. A graph and its fed weights
    /// die after its pass (unless `graphs` keeps it for the next band); a register lives until no name
    /// holds it any more (replaced, or forgotten). Planning: the same walk, nothing built.
    static func run(_ stages: [Stage], weights: Safetensors?, name: String, memory: VAEMemory,
                    registers initial: [String: VAERegister], graphs: Graphs? = nil) throws -> [String: VAERegister] {
        var registers = initial
        for (index, stage) in stages.enumerated() {
            let feeds = stage.inputs.map { registers[$0]! }
            let result: VAERegister
            if memory.half, let norm = stage.norm {
                result = try memory.register(stage.shape)
                let gamma = try memory.kernelWeights([norm.prefix + ".gamma"], counts: [norm.channels], file: weights, vae: name)
                if memory.isLive { memory.rmsNorm(feeds[0], into: result, gamma: gamma[0], silu: norm.silu) }
                memory.free(gamma)
            } else if let kernel = stage.kernel {
                result = stage.inPlace ? feeds[0] : try memory.register(stage.shape)
                precondition(result.shape == stage.shape, "QwenImage21VAE: pass \(stage.label ?? stage.output) in place on another shape")
                if memory.isLive { try kernel(memory, feeds, result) }
            } else {
                result = try memory.register(stage.shape)
                memory.outside(stage.outside)
                var weightRegisters: [VAERegister] = []
                try autoreleasepool {
                    let built = try memory.graph(largest: stage.workspace) {
                        () -> (graph: MPSGraph, placeholders: [MPSGraphTensor], output: MPSGraphTensor, fed: FedWeights) in
                        if let b = graphs?.built[index] { return b }
                        let fed = FedWeights()
                        let graph = MPSGraph()
                        let placeholders = feeds.map { graph.placeholder(shape: $0.nsShape, dataType: $0.dataType, name: nil) }
                        let output = try stage.build!(VAEBuildingBlocks(graph: graph, weights: weights!, name: name, fed: fed,
                                                                        dataType: memory.activation),
                                                      placeholders)
                        precondition(output.shape?.map(\.intValue) == stage.shape,
                                     "QwenImage21VAE: stage \(stage.label ?? stage.output) is not the shape its plan charges")
                        let b = (graph, placeholders, output, fed)
                        graphs?.built[index] = b
                        return b
                    }
                    let w = try memory.weights(stage.weights, built?.fed, file: weights, vae: name)
                    weightRegisters = w.registers
                    guard let built else { return }
                    memory.execute(built.graph, feeds: w.feeds.merging(zip(built.placeholders, feeds.map { $0.data! })) { a, _ in a },
                                   results: [built.output: result.data!])
                }
                memory.free(weightRegisters)
            }
            var released = [registers[stage.output]].compactMap { $0 }
            if let keep = stage.keep { registers[keep] = feeds[0] }
            registers[stage.output] = result
            for f in stage.forget { if let r = registers.removeValue(forKey: f) { released.append(r) } }
            var freed: [VAERegister] = []
            for r in released where !registers.values.contains(where: { $0 === r }) && !freed.contains(where: { $0 === r }) {
                memory.free(r); freed.append(r)
            }
            if memory.isLive, let label = stage.label { stageDone?(label) }
        }
        return registers
    }

    // ── the passes of a block ───────────────────────────────────────────────────────────────

    /// `QwenImageResidualBlock` on `x` (`i` → `o` channels at `size`), as passes: SiLU(RMS) · conv₁ ·
    /// SiLU(RMS) · [conv_shortcut] · conv₂ + residual — `resnetQwen`'s operations.
    /// `keep`: the name under which `x` stays (a block's input, for its shortcut).
    static func resnet(_ prefix: String, _ i: Int, _ o: Int, size: (h: Int, w: Int), keep: String? = nil,
                       label: String) -> [Stage] {
        let (a, b) = ([1, i, size.h, size.w], [1, o, size.h, size.w])
        var norm1 = Stage(["x"], "h", a, weights: WeightList.rms(i), keep: keep) { k, t in
            try k.silu(k.rmsNorm(t[0], prefix + ".norm1", channels: i)) }
        norm1.norm = (prefix + ".norm1", i, true)
        var norm2 = Stage(["h"], "h", b, weights: WeightList.rms(o)) { k, t in try k.silu(k.rmsNorm(t[0], prefix + ".norm2", channels: o)) }
        norm2.norm = (prefix + ".norm2", o, true)
        var passes = [
            norm1,
            Stage(["h"], "h", b, weights: WeightList.conv(o, i)) { k, t in try k.conv(t[0], prefix + ".conv1", padding: 1) },
            norm2,
        ]
        if i != o {
            var shortcut = Stage(["x"], "r", b, weights: WeightList.conv(o, i, kernel: 1)) { k, t in
                try k.conv(t[0], prefix + ".conv_shortcut", padding: 0) }
            shortcut.workspace = VAELayout.reformattedWeights(o * i) / 4
            passes.append(shortcut)
        }
        passes.append(Stage(["h", i != o ? "r" : "x"], "x", b, weights: WeightList.conv(o, o), forget: ["h", "r"], label: label) { k, t in
            k.graph.addition(try k.conv(t[0], prefix + ".conv2", padding: 1), t[1], name: nil) })
        return passes
    }

    /// The Wan attention block on `x` (`c` channels at `size`), as passes: RMS, the tokens (a transpose of
    /// ours), q, k, v (the three rows of `to_qkv`), one head, the output projection, and the residual
    /// laid back by a kernel of ours — `bottleneck`'s operations. The two transposes were MPSGraph's
    /// own copies, beyond our heaps (`S·C` each).
    static func attention(_ prefix: String, _ c: Int, size: (h: Int, w: Int), label: String,
                          requestBlock: Int = EngineSettings.effective.vaeRequestBlock) -> [Stage] {
        let s = size.h * size.w, tokens = [1, s, c]
        let qkv = ["q", "k", "v"].enumerated().map { (n, name) in
            Stage(["t"], name, tokens, weights: WeightList.projection(c, c), forget: n == 2 ? ["t"] : []) { k, t in
                try k.projection(t[0], prefix + ".to_qkv", rows: (n * c)..<((n + 1) * c)) }
        }
        // MPSGraph's SDPA copies its queries into the workspace: `S·C` floats, asked in whole 2 MiB.
        var sdpa = Stage(["q", "k", "v"], "a", tokens, forget: ["q", "k", "v"]) { k, t in
            k.scaledDotProduct(t[0], t[1], t[2], channels: c, requestBlock: requestBlock) }
        sdpa.workspace = max(sdpa.workspace, VAELayout.sdpaAsk(tokens: s, channels: c))
        sdpa.outside = VAELayout.sdpaOutside(tokens: s, channels: c)
        var norm = Stage(["x"], "n", [1, c, size.h, size.w], weights: WeightList.rms(c)) { k, t in
            try k.rmsNorm(t[0], prefix + ".norm", channels: c) }
        norm.norm = (prefix + ".norm", c, false)
        return [norm,
                Stage(["n"], "t", tokens, inPlace: false, forget: ["n"]) { memory, t, out in
                    memory.dispatch(.tokens, [t[0], out], constants: [UInt32(c), UInt32(s)],
                                    grid: MTLSize(width: c, height: s, depth: 1)) }]
            + qkv
            + [sdpa,
               Stage(["a"], "a", tokens, weights: WeightList.projection(c, c)) { k, t in try k.projection(t[0], prefix + ".proj") },
               Stage(["x", "a"], "x", [1, c, size.h, size.w], inPlace: true, forget: ["a"], label: label) { memory, t, out in
                   memory.dispatch(.untokensAdd, [out, t[1]], constants: [UInt32(c), UInt32(s)],
                                   grid: MTLSize(width: s, height: c, depth: 1)) }]
    }

    /// A bare convolution pass on `x` (padding 1, or 0 for a 1×1) — `conv_in`, `conv_out`, `quant_conv`.
    static func conv(_ prefix: String, _ i: Int, _ o: Int, size: (h: Int, w: Int), kernel: Int = 3,
                     label: String? = nil) -> Stage {
        var pass = Stage(["x"], "x", [1, o, size.h, size.w], weights: WeightList.conv(o, i, kernel: kernel), label: label) { k, t in
            try k.conv(t[0], prefix, padding: kernel / 2) }
        if kernel == 1 { pass.workspace = VAELayout.reformattedWeights(o * i) / 4 }
        return pass
    }

    /// The ×2 of `x` (`nearest-exact`), a kernel of ours (`VAEKernels`): `c` channels at `size`.
    static func double(_ c: Int, size: (h: Int, w: Int)) -> Stage {
        Stage(["x"], "x", [1, c, 2 * size.h, 2 * size.w], inPlace: false) { memory, t, out in
            memory.dispatch(.double, [t[0], out], constants: [UInt32(size.h), UInt32(size.w)],
                            grid: MTLSize(width: 2 * size.w, height: 2 * size.h, depth: c)) }
    }

    /// `upsampler.resample.1(×2(x)) + DupUp3D(skip)`, as passes: the ×2 and the shortcut's sum are
    /// kernels of ours (`VAEKernels`), the convolution a graph. Forgets `skip`.
    static func upsample(_ prefix: String, _ block: (input: Int, output: Int, temporal: Int, spatial: Int),
                         size: (h: Int, w: Int), label: String) throws -> [Stage] {
        let c = block.output, doubled = [1, c, 2 * size.h, 2 * size.w]
        let table = dupUpSources(input: block.input, output: c, temporal: block.temporal, spatial: 2)
        guard (0..<table.count).allSatisfy({ table[$0] == table[$0 / 2 * 2] }) else {
            throw Safetensors.Failure.badHeader("VAE Qwen-Image-2.1: DupUp3D \(block.input) → \(c) depends on the column")
        }
        let r = c * block.temporal * 4 / block.input
        return [
            double(c, size: size),
            Stage(["x"], "x", doubled, weights: WeightList.conv(c, c)) { k, t in
                try k.conv(t[0], prefix + ".upsampler.resample.1", padding: 1) },
            Stage(["x", "skip"], "x", doubled, inPlace: true, forget: ["skip"], label: label) { memory, t, out in
                memory.dispatch(.dupUpAdd, [out, t[1]],
                                constants: [UInt32(size.h), UInt32(size.w), UInt32(block.temporal), UInt32(r)],
                                grid: MTLSize(width: 2 * size.w, height: 2 * size.h, depth: c)) },
        ]
    }

    static func device() throws -> (MTLDevice, MTLCommandQueue) { try SystemGPU.open() }
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

    /// - Parameter path: the published `vae/diffusion_pytorch_model.safetensors` (the Turbo's in bf16),
    ///   each tensor widened to fp32 as it is read (`VAEBuildingBlocks.weight`), computed in fp32;
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
        let bandPixels = MemoryPlan.decoderBand(standard: QwenImage21VAEDecoder.bandPixels,
                                                minimum: MemoryPlan.qwenMinimumBand, unit: "pixels")
        let half = VAEPrecision.half(publishedInBF16: QwenImage21VAEDecoder.publishedInBF16, weights: weights)
        let plan = QwenImage21VAEDecoder.plan(config: config, latentHeight: latentHeight, latentWidth: latentWidth,
                                              bandPixels: bandPixels, half: half)
        let memory = try VAEMemory(plan: plan, device: device, queue: queue, half: half)
        let rgba = try autoreleasepool {
            try QwenImage21VAEDecoder.decode(latent, config: config, latentHeight: latentHeight, latentWidth: latentWidth,
                                             bandPixels: bandPixels, weights: weights, memory: memory)
        }
        memory.finish()
        return (rgba, Date().timeIntervalSince(started))
    }

    /// **Everything a decoding holds** for an `h × w` latent at a tail band of `bandPixels`: the
    /// registers' heap, the largest stage's workspace, and the RGBA it returns — `decode` walked
    /// without a device. `decode` allocates exactly this before its first graph.
    package static func plan(config: QwenImage21VAE.Config, latentHeight h: Int, latentWidth w: Int,
                             bandPixels: Int = QwenImage21VAEDecoder.bandPixels, half: Bool = false) -> VAELayout {
        plans(VAEPlanKey(config, [h, w, bandPixels, half ? 1 : 0])) {
            walk(config: config, latentHeight: h, latentWidth: w, bandPixels: bandPixels, half: half)
        }
    }

    /// The Turbo's decoder is published in bf16 (`VAEPrecision`).
    package static let publishedInBF16 = true

    private static let plans = ShapeMemo<VAEPlanKey<QwenImage21VAE.Config>, VAELayout>()

    /// `plan`, walked every time: what the memory keeps, and what the tests compare it with.
    package static func walk(config: QwenImage21VAE.Config, latentHeight h: Int, latentWidth w: Int,
                             bandPixels: Int = QwenImage21VAEDecoder.bandPixels, half: Bool = false) -> VAELayout {
        let memory = VAEMemory(half: half)
        do {
            _ = try decode(nil, config: config, latentHeight: h, latentWidth: w, bandPixels: bandPixels, weights: nil,
                           memory: memory)
        } catch { preconditionFailure("VAE plan walk refused: \(error)") }
        return memory.layout
    }

    /// The decoding itself, live or planned: the de-normalization written into the input register (on
    /// the CPU: two roundings, like `torch`, as `QwenImage21VAE.denormalize`), the stages, the tail.
    private static func decode(_ latent: [Float]?, config: QwenImage21VAE.Config, latentHeight h: Int, latentWidth w: Int,
                               bandPixels: Int, weights: Safetensors?, memory: VAEMemory) throws -> [Float] {
        let input = try memory.register([1, config.zDim, h, w])
        if let latent, memory.isLive {
            // In bf16 (`VAEMemory.half`) the fp32 de-normalization is rounded once into the register, a plane at a time.
            let plane = h * w
            var row = [Float](repeating: 0, count: input.isHalf ? plane : 0)
            for c in 0..<config.zDim {
                let (s, m) = (config.std[c], config.mean[c])
                if input.isHalf {
                    for p in 0..<plane { let scaled = latent[c * plane + p] * s; row[p] = scaled + m }
                    row.withUnsafeBufferPointer {
                        Widen.float32ToBfloat16(source: $0.baseAddress!, destination: input.raw + 2 * c * plane, count: plane)
                    }
                } else {
                    let z = input.values
                    for p in (c * plane)..<((c + 1) * plane) { let scaled = latent[p] * s; z[p] = scaled + m }
                }
            }
        }
        let registers = try QwenImage21VAE.run(try stages(config: config, latentHeight: h, latentWidth: w), weights: weights,
                                               name: "VAE Qwen-Image-2.1", memory: memory, registers: ["x": input])
        let (x, skip) = (registers["x"]!, registers["skip"]!)
        let expected = QwenImage21VAEDecoder.tailInput(config: config)
        precondition(x.shape[1] == expected.x && skip.shape[1] == expected.skip
                     && x.shape[2] == expected.scale * h && skip.shape[3] == expected.scale * w,
                     "QwenImage21VAEDecoder: the tail's input is not the one `tailInput` charges")
        let rgba = try tail(x: x, skip: skip, config: config, bandPixels: bandPixels, weights: weights, memory: memory)
        memory.free(x); memory.free(skip)
        return rgba
    }

    /// The decoder's passes up to the last upsamplings, excluded: `post_quant_conv`, `conv_in`, the
    /// bottleneck, then each block's resnets, ×2, convolution and shortcut sum. `skip` is a block's input,
    /// kept by its first resnet for its shortcut.
    private static func stages(config: QwenImage21VAE.Config, latentHeight: Int, latentWidth: Int) throws -> [QwenImage21VAE.Stage] {
        typealias V = QwenImage21VAE
        let dims = config.decoderDims
        let L = (h: latentHeight, w: latentWidth)
        var stages: [V.Stage] = [
            V.conv("post_quant_conv", config.zDim, config.zDim, size: L, kernel: 1),
            V.conv("decoder.conv_in", config.zDim, dims[0], size: L),
        ]
        stages += V.resnet("decoder.mid_block.resnets.0", dims[0], dims[0], size: L, label: "mid.resnet0")
        stages += V.attention("decoder.mid_block.attentions.0", dims[0], size: L, label: "mid.attention")
        stages += V.resnet("decoder.mid_block.resnets.1", dims[0], dims[0], size: L, label: "mid")
        var scale = 1
        for i in 0..<(config.dimMult.count - 1) {
            let block = config.up(i), prefix = "decoder.up_blocks.\(i)"
            let size = (h: scale * L.h, w: scale * L.w)
            for r in 0...config.resBlocks {
                stages += V.resnet("\(prefix).resnets.\(r)", r == 0 ? block.input : block.output, block.output, size: size,
                                   keep: r == 0 ? "skip" : nil, label: "\(i).resnet\(r)")
            }
            if i == config.dimMult.count - 1 - QwenImage21VAEDecoder.tailLevels { break }   // the last upsamplings belong to the tail
            stages += try V.upsample(prefix, block, size: size, label: "\(i).shortcut")
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

    /// **A band of the tail** for a tail input of `tailHeight × tailWidth` (the latent ×4): the margin in
    /// rows of that input, the core each band keeps, and the window it reads (`factor · window` output
    /// rows) — one source, which `tail` runs.
    /// The margin, back from the output: `conv_out`, the last block's resnets and its upsampling conv;
    /// then, per level below, ×2 (rounded up), that block's resnets and upsampling conv.
    package static func band(config: QwenImage21VAE.Config, tailHeight h: Int, tailWidth w: Int,
                             bandPixels: Int) -> (margin: Int, core: Int, window: Int) {
        let levels = tailLevels, factor = 1 << levels
        let convsPerBlock = 2 * (config.resBlocks + 1)
        var margin = 1 + convsPerBlock + 1
        for _ in 1..<max(1, levels) { margin = (margin + 1) / 2 + convsPerBlock + 1 }
        margin = (margin + 1) / 2
        let core = max(1, bandPixels / (factor * factor * w))
        return (margin, core, min(h, core + 2 * margin))
    }

    /// **The tail's input**: `(channels of x, channels of skip, scale)` — `x` and `skip` are
    /// `[channels, scale·h, scale·w]` for an `h × w` latent once the stages before the tail are done.
    package static func tailInput(config: QwenImage21VAE.Config) -> (x: Int, skip: Int, scale: Int) {
        let last = config.dimMult.count - 1 - tailLevels
        return (config.up(last).output, config.up(last).input, 1 << last)
    }

    /// **The tail, by horizontal bands — exact.** From the second-to-last upsampling to `conv_out`,
    /// everything is local: convolutions, per-pixel RMSNorm, the shortcut's copies. A full-resolution row
    /// depends on the rows within **8 rows** of it at full resolution (the last upsampling conv, the six
    /// of the last block, `conv_out`), i.e. 4 at half resolution; to which the half-resolution level adds
    /// 7 (its upsampling conv, the six of its block): 11, i.e. **6 rows of the tail's input**. Each band
    /// reads its rows plus that margin on each side (none past the image's edge, where the model's own
    /// zero padding is the right one), and keeps only its core: every kept pixel sees exactly the inputs
    /// it sees in one piece. All windows have the same height (the last one slides back inside the
    /// image): the same passes, built once, serve them all. The skip of the inner block (its input, which
    /// its shortcut reads) is born inside the band.
    ///
    /// Why: in one piece at 1024², the last level alone held **7.1 GB** (`MPSGraph` keeps a stage's
    /// intermediates, and keeps them cached after it). The objection to tiling Flux's decoder — tiles
    /// change `GroupNorm`'s statistics — does not apply: there is no statistic across pixels here.
    private static func tail(x: VAERegister, skip: VAERegister, config: QwenImage21VAE.Config, bandPixels: Int,
                             weights: Safetensors?, memory: VAEMemory) throws -> [Float] {
        typealias V = QwenImage21VAE
        let levels = QwenImage21VAEDecoder.tailLevels, first = config.dimMult.count - 1 - levels
        let (cx, h, w) = (x.shape[1], x.shape[2], x.shape[3]), cs = skip.shape[1]
        let factor = 1 << levels
        let (margin, core, window) = QwenImage21VAEDecoder.band(config: config, tailHeight: h, tailWidth: w,
                                                                bandPixels: bandPixels)
        let (H, W) = (factor * h, factor * w)
        var passes: [V.Stage] = []
        for i in first..<(first + levels) {
            let block = config.up(i), next = config.up(i + 1), scale = 1 << (i - first)
            let size = (h: scale * window, w: scale * w), doubled = (h: 2 * size.h, w: 2 * size.w)
            passes += try V.upsample("decoder.up_blocks.\(i)", block, size: size, label: "tail \(i).shortcut")
            // The next block's input is kept for its own shortcut — but the last block has none.
            let kept = i + 1 < config.dimMult.count - 1
            for r in 0...config.resBlocks {
                passes += V.resnet("decoder.up_blocks.\(i + 1).resnets.\(r)", r == 0 ? next.input : next.output, next.output,
                                   size: doubled, keep: r == 0 && kept ? "skip" : nil, label: "tail \(i + 1).resnet\(r)")
            }
        }
        let last = config.up(first + levels).output
        var normOut = V.Stage(["x"], "h", [1, last, factor * window, W], weights: WeightList.rms(last), forget: ["x"]) { k, t in
            try k.silu(k.rmsNorm(t[0], "decoder.norm_out", channels: last)) }
        normOut.norm = ("decoder.norm_out", last, true)
        passes.append(normOut)
        passes.append(V.Stage(["h"], "x", [1, config.outChannels, factor * window, W],
                              weights: WeightList.conv(config.outChannels, last), forget: ["h"]) { k, t in
            k.graph.clamp(try k.conv(t[0], "decoder.conv_out", padding: 1),
                          min: k.graph.constant(-1, dataType: k.dataType), max: k.graph.constant(1, dataType: k.dataType), name: nil) })
        memory.host(4 * config.outChannels * H * W)
        var rgba = [Float](repeating: 0, count: memory.isLive ? config.outChannels * H * W : 0)
        // The passes' graphs are kept for the next band — when there is one. Kept, ~20 graphs hold their
        // compiled state together until the tail ends (at 512², one band: 30 MB at the decoder's peak,
        // on the first render of a process, where each dies after its pass otherwise).
        let graphs = (h + core - 1) / core > 1 ? V.Graphs() : nil
        var c0 = 0
        while c0 < h {
            let c1 = min(h, c0 + core)
            let start = min(max(0, c0 - margin), h - window)
            let xw = try memory.register([1, cx, window, w]), sw = try memory.register([1, cs, window, w])
            if memory.isLive {
                let pairs: [(VAERegister, VAERegister, Int)] = [(x, xw, cx), (skip, sw, cs)]
                for (source, target, channels) in pairs {
                    let (from, to, e) = (source.raw, target.raw, source.elementSize)
                    for c in 0..<channels {
                        (to + e * c * window * w).copyMemory(from: from + e * (c * h + start) * w, byteCount: e * window * w)
                    }
                }
            }
            let registers = try V.run(passes, weights: weights, name: "VAE Qwen-Image-2.1", memory: memory,
                                      registers: ["x": xw, "skip": sw], graphs: graphs)
            precondition(registers.count == 1, "QwenImage21VAEDecoder: the tail left \(registers.keys.sorted()) alive")
            let out = registers["x"]!
            if memory.isLive {
                // Keep the core: full-resolution rows [F·c0, F·c1), found at F·(c0 − start) in the window (widened from bf16
                // in half precision, exactly).
                rgba.withUnsafeMutableBufferPointer { target in
                    for c in 0..<config.outChannels {
                        let to = target.baseAddress! + (c * H + factor * c0) * W
                        let from = (c * factor * window + factor * (c0 - start)) * W, count = factor * (c1 - c0) * W
                        if out.isHalf {
                            Widen.bfloat16ToFloat32(source: UnsafeRawPointer(out.raw + 2 * from), destination: to, count: count)
                        } else {
                            to.update(from: out.values + from, count: count)
                        }
                    }
                }
                V.stageDone?("tail \(c0)-\(c1)")
            }
            memory.free(out)
            c0 = c1
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
            throw EngineError.formatRefused(width: width, height: height, reason: .notMultiple, bound: config.factor)
        }
        (device, queue) = try QwenImage21VAE.device()
        self.config = config; self.weights = weights; self.height = height; self.width = width
    }

    /// `rgba`: `[4, height, width]` planar in `[-1, 1]` — alpha included (+1 for an opaque image).
    package func encode(rgba: [Float]) throws -> Output {
        precondition(rgba.count == config.inChannels * height * width, "QwenImage21VAEEncoder: image shape")
        let started = Date()
        let memory = try VAEMemory(plan: QwenImage21VAEEncoder.plan(config: config, height: height, width: width),
                                   device: device, queue: queue)
        let moments = try autoreleasepool {
            try QwenImage21VAEEncoder.encode(rgba, config: config, height: height, width: width, weights: weights,
                                             memory: memory)
        }
        memory.finish()
        let mean = moments.prefix(config.zDim * latentHeight * latentWidth)
        return Output(latent: QwenImage21VAE.normalize(mean, config: config), moments: moments,
                      seconds: Date().timeIntervalSince(started))
    }

    /// **Everything an encoding holds** for an `H × W` image: the registers' heap, the largest stage's
    /// workspace, and the host copies (the moments read back, the normalized latent) — `encode`
    /// walked without a device.
    package static func plan(config: QwenImage21VAE.Config, height: Int, width: Int) -> VAELayout {
        plans(VAEPlanKey(config, [height, width])) { walk(config: config, height: height, width: width) }
    }

    private static let plans = ShapeMemo<VAEPlanKey<QwenImage21VAE.Config>, VAELayout>()

    /// `plan`, walked every time: what the memory keeps, and what the tests compare it with.
    package static func walk(config: QwenImage21VAE.Config, height: Int, width: Int) -> VAELayout {
        let memory = VAEMemory()
        do { _ = try encode(nil, config: config, height: height, width: width, weights: nil, memory: memory) } catch { preconditionFailure("VAE plan walk refused: \(error)") }
        return memory.layout
    }

    private static func encode(_ rgba: [Float]?, config: QwenImage21VAE.Config, height: Int, width: Int,
                               weights: Safetensors?, memory: VAEMemory) throws -> [Float] {
        let input = try memory.register([1, config.inChannels, height, width])
        if let rgba, memory.isLive { input.values.update(from: rgba, count: input.count) }
        let out = try QwenImage21VAE.run(try stages(config: config, height: height, width: width), weights: weights,
                                         name: "VAE Qwen-Image-2.1 encoder", memory: memory, registers: ["x": input])["x"]!
        // The moments, then the normalized latent (`QwenImage21VAE.normalize`), both read back to the host.
        memory.host(4 * out.count + 4 * out.count / 2)
        let moments = memory.isLive ? Array(UnsafeBufferPointer(start: out.values, count: out.count)) : []
        memory.free(out)
        return moments
    }

    /// Each block's passes — `conv_in` (block 0), the resnets (the first keeps the block's input as
    /// `skip`), the stride-2 convolution, and the shortcut's `AvgDown3D` summed in place by a kernel of
    /// ours (`VAEKernels`) —, then the bottleneck and the head after the last.
    private static func stages(config: QwenImage21VAE.Config, height: Int, width: Int) throws -> [QwenImage21VAE.Stage] {
        typealias V = QwenImage21VAE
        let dims = config.encoderDims
        let blocks = config.dimMult.count
        var size = (h: height, w: width)
        var stages: [V.Stage] = []
        for i in 0..<blocks {
            let block = config.down(i), prefix = "encoder.down_blocks.\(i)", here = size
            let top = (h: here.h / block.spatial, w: here.w / block.spatial)
            if i == 0 { stages.append(V.conv("encoder.conv_in", config.inChannels, dims[0], size: here)) }
            for r in 0..<config.resBlocks {
                stages += V.resnet("\(prefix).resnets.\(r)", r == 0 ? block.input : block.output, block.output, size: here,
                                   keep: r == 0 ? "skip" : nil, label: "block \(i).resnet\(r)")
            }
            if block.spatial > 1 {
                var down = V.Stage(["x"], "x", [1, block.output, top.h, top.w], weights: WeightList.conv(block.output, block.output)) { k, t in
                    try k.downsample(t[0], "\(prefix).downsampler.resample.1") }
                down.workspace = VAELayout.reformattedWeights(block.output * block.output * 9) / 4
                stages.append(down)
            }
            // The group mean divides by a power of two: exact, whether the reference's `mean` multiplies or divides.
            let (s, ft) = (block.spatial, block.temporal), g = block.input * ft * s * s / block.output
            guard g > 0, g & (g - 1) == 0, block.input * ft * s * s == g * block.output else {
                throw Safetensors.Failure.badHeader("VAE Qwen-Image-2.1 encoder: AvgDown3D \(block.input) → \(block.output) "
                                                    + "in groups of \(g)")
            }
            stages.append(V.Stage(["x", "skip"], "x", [1, block.output, top.h, top.w], inPlace: true, forget: ["skip"],
                                  label: "block \(i)") { memory, t, out in
                memory.dispatch(.avgDownAdd, [out, t[1]],
                                constants: [UInt32(top.h), UInt32(top.w), UInt32(s), UInt32(ft), UInt32(g)],
                                grid: MTLSize(width: top.w, height: top.h, depth: block.output)) })
            size = top
        }
        let c = dims.last!, z = 2 * config.zDim
        stages += V.resnet("encoder.mid_block.resnets.0", c, c, size: size, label: "mid.resnet0")
        stages += V.attention("encoder.mid_block.attentions.0", c, size: size, label: "mid.attention")
        stages += V.resnet("encoder.mid_block.resnets.1", c, c, size: size, label: "mid")
        stages.append(V.Stage(["x"], "x", [1, c, size.h, size.w], weights: WeightList.rms(c)) { k, t in
            try k.silu(k.rmsNorm(t[0], "encoder.norm_out", channels: c)) })
        stages.append(V.conv("encoder.conv_out", c, z, size: size))
        stages.append(V.conv("quant_conv", z, z, size: size, kernel: 1, label: "head"))
        return stages
    }
}

// ── the modules ─────────────────────────────────────────────────────────────────────────────

/// Qwen-Image-2.1's decoder: the RGBA it decodes, **alpha dropped without compositing** — what the
/// oracle's `pil.convert("RGB")` writes.
package struct QwenImage21DecodingModule: DecodingModule {
    package var path: String
    package var space: LatentSpace { .qwenImage21 }
    package var name: String { "VAE Qwen-Image-2.1" }
    package init(path: String) { self.path = path }

    package func decode(_ latent: Latent, context: Context) throws -> ImageRGB {
        let vae = try QwenImage21VAEDecoder(path: path, latentHeight: latent.height, latentWidth: latent.width)
        let (rgba, seconds) = try vae.decode(latent: latent.values)
        context.timings.decoding = seconds
        return ImageRGB(pixels: Array(rgba.prefix(3 * vae.imageHeight * vae.imageWidth)),
                        height: vae.imageHeight, width: vae.imageWidth)
    }
}

/// Qwen-Image-2.1's encoder, for the edit references and img2img: the image made **RGBA** (alpha +1,
/// as `convert("RGBA")` makes an opaque image; the image's own straight RGBA when it has transparency,
/// as `preprocess` hands the RGBA reference: no compositing), then `(mean − mean_c) / std_c`.
package struct QwenImage21EncodingModule: ImageEncodingModule {
    package var path: String
    package var space: LatentSpace { .qwenImage21 }
    package var name: String { "VAE Qwen-Image-2.1" }
    package init(path: String) { self.path = path }

    package func encoder(_ image: ImageRGB, context: Context) throws -> Latent {
        let encoder = try QwenImage21VAEEncoder(path: path, height: image.height, width: image.width)
        let output = try encoder.encode(rgba: QwenImage21EncodingModule.rgba(image))
        context.footprints.encoding = Arena.processFootprint()   // before the release
        return Latent(space: space, height: encoder.latentHeight, width: encoder.latentWidth, values: output.latent)
    }

    /// `[3, H, W]` → `[4, H, W]` with an opaque alpha: 255 → `255/255 · 2 − 1` = +1. With transparency,
    /// the straight `rgba` planes, each byte `b/255 · 2 − 1` like `VaeImageProcessor` (`pil_to_numpy`, then
    /// `2x − 1`).
    package static func rgba(_ image: ImageRGB) -> [Float] {
        guard let bytes = image.rgba else { return image.pixels + [Float](repeating: 1, count: image.height * image.width) }
        let plane = image.height * image.width
        var planes = [Float](repeating: 0, count: 4 * plane)
        for p in 0..<plane { for c in 0..<4 { planes[c * plane + p] = Float(bytes[p * 4 + c]) / 255 * 2 - 1 } }
        return planes
    }
}
