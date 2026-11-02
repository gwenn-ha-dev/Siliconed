import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import Foundation

/// **The encoders of both VAEs, in fp32, via `MPSGraph` — img2img's input.**
///
///     Flux (Z-Image)     image [3, H, W] ─ conv_in ─ 4 blocks (2 resnets, ÷2 except the last)
///                          ─ mid(resnet, attention, resnet) ─ GroupNorm ─ SiLU ─ conv_out → moments [32, H/8, W/8]
///     Qwen-Image         image [3, H, W] ─ conv_in ─ 11 flat layers (resnets, ÷2 at 2, 5, 8)
///     (Anima, Krea 2)      ─ mid(resnet, attention, resnet) ─ RMSNorm ─ SiLU ─ conv_out ─ quant_conv → moments
///     FLUX.2             Flux's encoder as it stands ─ conv_out (64) ─ quant_conv (1×1, 64 → 64) → moments [64, H/8, W/8]
///     (MOSS-Image)         ─ mean (32) ─ patchify 2×2 ─ (z − mean_c) / √(var_c + ε) per patched channel → [128, H/16, W/16]
///
///     moments = [mean (16 or 32) | log-variance (16 or 32)]      latent = normalize(mean)
///
/// The bricks are those of the decoders (`VAEBuildingBlocks`); the only new operation is the
/// **downsampling**: `3×3`, stride 2, asymmetric padding `(0, 1, 0, 1)` (`VAEBuildingBlocks.downsample`).
///
/// ## Qwen-Image: one frame, hence an exact 2D encoder
///
/// Same reasoning as the decoder (`AnimaVAE`), verified on the encoding path:
/// `AutoencoderKLQwenImage._encode` passes the single frame through the causal encoder with a
/// **cleared** cache, so each `CausalConv3d` pads time with two frames of zeros and only works
/// through its `t = 2` slice (`AnimaVAE.lastSlice`); `downsample3d` skips its `time_conv`
/// on the first pass (empty cache): only `resample.1` remains, a stride-2 `Conv2d`. The
/// `time_conv`s are not read. Anima and Krea 2 publish **the same weights** (bf16 versus f32,
/// bit-equal once widened): one encoder, two files.
///
/// ## FLUX.2: Flux's encoder, then the pipeline's part on the host
///
/// `AutoencoderKLFlux2`'s encoder is diffusers' `Encoder`, the very one Flux.1's VAE uses (`block_out_channels`
/// 128…512, two resnets per block, GroupNorm 32) — only `conv_out` doubles (2 × 32 channels) and a `quant_conv`
/// (1×1, 64 → 64) follows, as Qwen-Image's does: the mean also depends on the log-variance half. What a port by
/// analogy with Flux.1 would miss is after it, in the pipeline (`Flux2Pipeline._encode_vae_image`; MOSS-Image's
/// own pipeline never encodes): the mean is **patchified** 2×2 with channel `4c + 2py + px` — the inverse of
/// `Flux2VAE.unpatchify` — and only then normalized by the batch-norm's running statistics of the 128 PATCHED
/// channels, `(z − mean) / √(var + 1e-4)`. Flux.1's two scalars are not this. Done on the host
/// (`Flux2VAE.patchify`, `.normalize`): a latent's worth of arithmetic, the same roundings as torch's.
///
/// ## The mean, not a sample — a deliberate departure from diffusers
///
/// The encoder returns a diagonal Gaussian; we take its **mean** (`mode()`), and the render
/// stays deterministic for a given seed. `ZImageImg2ImgPipeline` and `QwenImageImg2ImgPipeline` draw,
/// for their part, a **sample** (`retrieve_latents(…, sample_mode="sample")`, `mean + std·ε`) with the
/// generator, before the noise — only Anima's modular block takes `argmax`, like us. The deviation
/// is `std·ε`, with a `std` that the `vae-encode` check displays (small next to the latent's
/// standard deviation). The img2img oracles take the mean too.
///
/// ## Memory: Flux's encoder by layers and bands, in two heaps of ours — the same bits
///
/// In one graph, Flux's encoder held its peak in MPS's own allocations (3.2 GB at 1024², 2.4 GB of it
/// still pending in Metal when it returned) and its weights as graph constants. It now runs as
/// Flux's decoder does (`VAE.Bands`): `conv_in`, each resnet's two `conv(SiLU(GN(x)))` with the
/// `GroupNorm` statistics over the whole tensor, each stride-2 convolution, all by horizontal bands,
/// then the bottleneck and the head by passes at the latent's grid — every register, weight and MPS
/// workspace in the heaps `plan` sizes (`VAEMemory`). **The bits of the one graph** (md5 of the moments
/// at 512² and 1024², and of the img2img render). The one graph stays for the checks only (`probes`,
/// `startingAt`: a block's output, a stage's local error) — and for Qwen-Image's family (Anima, Krea 2),
/// whose weights are not installed here to prove a cut against it.
package final class VAEEncoder {
    package enum Family: Sendable {
        /// Flux `AutoencoderKL` (Z-Image): GroupNorm, no `quant_conv`.
        case flux
        /// `AutoencoderKLQwenImage` (Anima, Krea 2): RMSNorm, causal 3D kernels, `quant_conv`.
        case qwenImage
        /// `AutoencoderKLFlux2` (MOSS-Image): Flux's encoder with 32 channels and a `quant_conv`, then the
        /// pipeline's patchify and batch-norm normalization.
        case flux2

        /// The latent channels (half of the moments).
        var channels: Int { self == .flux2 ? 32 : 16 }
        /// Pixels per cell of the denoiser's latent: the network's 8, times FLUX.2's 2×2 patch.
        var factor: Int { self == .flux2 ? 16 : 8 }
    }

    /// What an encoding returns: the normalized latent — the one the denoiser expects —, the
    /// raw moments (for the checks) and, on request, the per-block probes.
    package struct Output {
        /// `[16, H/8, W/8]` (FLUX.2: `[128, H/16, W/16]`), normalized in the denoiser's space.
        package let latent: [Float]
        /// `[32, H/8, W/8]` (FLUX.2: `[64, H/8, W/8]`, after `quant_conv`): mean then log-variance, before normalization.
        package let moments: [Float]
        package let probes: [String: [Float]]
        package let seconds: Double
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let weights: Safetensors
    /// The one graph: the checks' (`probes`, `startingAt`) and Qwen-Image's family. `nil`: Flux's
    /// encoder by bands (`encodeBanded`), the product's.
    private let graph: MPSGraph?
    private let entry: MPSGraphTensor?
    private let entryShape: [Int]
    private let targets: [String: MPSGraphTensor]
    package let family: Family
    package let height, width: Int
    /// FLUX.2's batch-norm statistics, `√(running_var + ε)` and `running_mean` per patched channel; `nil` otherwise.
    private let batchNorm: (scale: [Float], shift: [Float])?
    /// The denoiser's grid: `H/8` (FLUX.2: `H/16`, after the patch).
    package var latentGridHeight: Int { height / family.factor }
    package var latentGridWidth: Int { width / family.factor }

    /// - Parameter path: the published `vae/diffusion_pytorch_model.safetensors`, read as is.
    /// - Parameter height, width: the image, in pixels (multiples of 8).
    /// - Parameter probes: keep the block outputs (`conv_in`, `down_blocks.i`, `mid_block`) —
    ///   for the checks; they stay alive until the end of the graph.
    /// - Parameter startingAt: for the checks only — the graph starts **at this probe's output**,
    ///   which becomes the placeholder `encode` feeds (an oracle's tensor at that boundary): what
    ///   follows is the stage's **local** error, the one a badly conditioned stage is judged on
    ///  . The probes upstream of it are dropped. `nil`: the image, the product's graph.
    package init(path: String, family: Family, height: Int, width: Int, probes: Bool = false,
                 startingAt: String? = nil) throws {
        let weights = try Safetensors(path: path)
        let expectedShape = family == .qwenImage ? [96, 3, 3, 3, 3] : [128, 3, 3, 3]
        // Flux.1's and FLUX.2's encoders differ by `conv_out`'s width (2 × 16, 2 × 32) and FLUX.2's `quant_conv`.
        let flux2Shaped = family != .flux2 || (weights.entries["encoder.conv_out.weight"]?.shape == [64, 512, 3, 3]
                                               && weights.entries["quant_conv.weight"]?.shape == [64, 64, 1, 1])
        guard weights.entries["encoder.conv_in.weight"]?.shape == expectedShape, flux2Shaped else {
            throw Safetensors.Failure.badHeader("\(path): not the VAE encoder "
                                                + [.flux: "of Flux", .qwenImage: "of Qwen-Image", .flux2: "of FLUX.2"][family]!)
        }
        guard height % family.factor == 0, width % family.factor == 0 else {
            throw EngineError.formatRefused(width: width, height: height, reason: .notMultiple, bound: family.factor)
        }
        if family == .flux2 {
            let patched = 4 * family.channels
            guard let mean = weights.materialize("bn.running_mean"), let variance = weights.materialize("bn.running_var"),
                  mean.count == patched, variance.count == patched else {
                throw Safetensors.Failure.badHeader("\(path): the FLUX.2 VAE's batch-norm statistics (\(patched) channels) are missing")
            }
            batchNorm = (Flux2VAE.scale(variance: variance, epsilon: Flux2VAE.Config().batchNormEpsilon), mean)
        } else {
            batchNorm = nil
        }
        let (device, queue) = try SystemGPU.open()
        self.device = device; self.queue = queue; self.weights = weights
        self.family = family; self.height = height; self.width = width
        guard probes || startingAt != nil || family == .qwenImage else {
            graph = nil; entry = nil; targets = [:]; entryShape = [1, 3, height, width]
            return
        }
        let graph = MPSGraph()
        let input = graph.placeholder(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)],
                                      dataType: .float32, name: "image")
        var captured: [String: MPSGraphTensor] = [:]
        var entry = input
        // A probe hands its tensor back unchanged — the product's graph — except the one the graph
        // starts at, which it replaces by a placeholder of the same shape.
        let probe = { (name: String, x: MPSGraphTensor) -> MPSGraphTensor in
            guard name == startingAt else { captured[name] = x; return x }
            entry = graph.placeholder(shape: x.shape, dataType: .float32, name: name)
            captured.removeAll()
            return entry
        }
        let moments: MPSGraphTensor
        switch family {
        case .flux:
            moments = try VAEEncoder.flux(graph, weights, input, (height, width), probe: probe)
        case .qwenImage: moments = try VAEEncoder.qwen(graph, weights, input, (height, width), probe: probe)
        case .flux2:
            moments = try VAEBuildingBlocks(graph: graph, weights: weights, name: "FLUX.2 encoder", groups: 32)
                .conv(VAEEncoder.flux(graph, weights, input, (height, width), probe: probe), "quant_conv", padding: 0)
        }
        if let startingAt, entry === input {
            throw Safetensors.Failure.badHeader("\(path): no probe \(startingAt) in this encoder")
        }
        let mean = graph.sliceTensor(moments, dimension: 1, start: 0, length: family.channels, name: nil)
        // FLUX.2's latent is made on the host from the moments (`hostLatent`): the patch is a permutation.
        var targets = family == .flux2 ? ["moments": moments] : ["moments": moments, "latent": VAEEncoder.normalize(graph, mean, family)]
        if probes { targets.merge(captured) { a, _ in a } }
        self.graph = graph; self.entry = entry; self.targets = targets
        self.entryShape = (entry.shape ?? []).map(\.intValue)
    }

    /// **The normalization, the exact inverse of the decoders' denormalization.**
    /// Flux: `(z − 0.1159) × 0.3611` (`pipeline_z_image_img2img.py:330`). Qwen-Image:
    /// `(z − mean_c) / std_c` per channel (Anima's modular block; `QwenImageImg2ImgPipeline`
    /// multiplies by `1/std`, the same thing to within one ulp).
    private static func normalize(_ graph: MPSGraph, _ z: MPSGraphTensor, _ family: Family) -> MPSGraphTensor {
        switch family {
        case .flux:
            let config = VAE.Config()
            return graph.multiplication(
                graph.subtraction(z, graph.constant(Double(config.shift), dataType: .float32), name: nil),
                graph.constant(Double(config.scaling), dataType: .float32), name: "normalize")
        case .qwenImage:
            let config = AnimaVAE.Config.qwenImage
            func vector(_ v: [Float]) -> MPSGraphTensor {
                v.withUnsafeBytes { graph.constant(Data($0), shape: [1, 16, 1, 1], dataType: .float32) }
            }
            return graph.division(graph.subtraction(z, vector(config.mean), name: nil),
                                  vector(config.std), name: "normalize")
        case .flux2:
            preconditionFailure("FLUX.2's latent is normalized on the host (hostLatent)")
        }
    }

    /// **FLUX.2's latent from its moments**, as `Flux2Pipeline._encode_vae_image` makes it: the mean (the first 32
    /// planes of `[64, h, w]`), patchified 2×2, then `(z − mean) / √(var + ε)` per patched channel.
    static func hostLatent(moments: [Float], height h: Int, width w: Int, batchNorm: (scale: [Float], shift: [Float]),
                           channels: Int = 32) -> [Float] {
        let mean = Array(moments.prefix(channels * h * w))
        let patched = Flux2VAE.patchify(mean, channels: channels, patch: 2, height: h, width: w)
        return Flux2VAE.normalize(patched, scale: batchNorm.scale, shift: batchNorm.shift, plane: (h / 2) * (w / 2))
    }

    /// Flux's encoder (`diffusers/models/autoencoders/vae.py`, `Encoder`): `block_out_channels
    /// [128, 256, 512, 512]`, two resnets per block, a downsampling after each except the last.
    private static func flux(_ graph: MPSGraph, _ weights: Safetensors, _ input: MPSGraphTensor,
                             _ size: (h: Int, w: Int),
                             probe: (String, MPSGraphTensor) -> MPSGraphTensor) throws -> MPSGraphTensor {
        let b = VAEBuildingBlocks(graph: graph, weights: weights, name: "Flux encoder", groups: 32)
        var x = try b.conv(input, "encoder.conv_in", padding: 1)
        x = probe("conv_in", x)
        var t = size, entry = 128
        for (i, output) in [128, 256, 512, 512].enumerated() {
            let prefix = "encoder.down_blocks.\(i)"
            x = try b.resnetFlux(x, prefix + ".resnets.0", inChannels: entry, outChannels: output, size: t)
            x = try b.resnetFlux(x, prefix + ".resnets.1", inChannels: output, outChannels: output, size: t)
            if i < 3 {
                x = try b.downsample(x, prefix + ".downsamplers.0.conv")
                t = (t.h / 2, t.w / 2)
            }
            entry = output
            x = probe("down_blocks.\(i)", x)
        }
        let mid = "encoder.mid_block"
        x = try b.resnetFlux(x, mid + ".resnets.0", inChannels: 512, outChannels: 512, size: t)
        x = probe("mid_block.resnets.0", x)
        let normalized = try b.groupNorm(x, mid + ".attentions.0.group_norm", channels: 512, h: t.h, w: t.w)
        x = try b.attention(x, normalized: normalized, channels: 512, size: t,
                            qkv: { (try b.projection($0, mid + ".attentions.0.to_q"), try b.projection($0, mid + ".attentions.0.to_k"),
                                    try b.projection($0, mid + ".attentions.0.to_v")) },
                            output: mid + ".attentions.0.to_out.0", requestBlock: EngineSettings.effective.vaeRequestBlock)
        x = probe("mid_block.attentions.0", x)
        x = try b.resnetFlux(x, mid + ".resnets.1", inChannels: 512, outChannels: 512, size: t)
        x = probe("mid_block", x)
        x = try b.silu(b.groupNorm(x, "encoder.conv_norm_out", channels: 512, h: t.h, w: t.w))
        // `use_quant_conv` is false in Z-Image's config: `conv_out` returns the moments.
        return try b.conv(x, "encoder.conv_out", padding: 1)
    }

    /// Qwen-Image's encoder (`QwenImageEncoder3d`, `dims [96, 96, 192, 384, 384]`) at a single frame.
    /// `down_blocks` is a **flat** list: resnets at 0, 1, 3, 4, 6, 7, 9, 10, downsamplings at 2, 5, 8.
    private static func qwen(_ graph: MPSGraph, _ weights: Safetensors, _ input: MPSGraphTensor,
                             _ size: (h: Int, w: Int), probe: (String, MPSGraphTensor) -> MPSGraphTensor) throws -> MPSGraphTensor {
        let b = VAEBuildingBlocks(graph: graph, weights: weights, name: "Qwen-Image encoder")
        var x = try b.conv(input, "encoder.conv_in", padding: 1)
        x = probe("conv_in", x)
        var t = size
        let layers: [(entry: Int, output: Int)?] = [(96, 96), (96, 96), nil, (96, 192), (192, 192), nil,
                                                       (192, 384), (384, 384), nil, (384, 384), (384, 384)]
        for (i, layerPass) in layers.enumerated() {
            let prefix = "encoder.down_blocks.\(i)"
            if let layerPass {
                x = try b.resnetQwen(x, prefix, inChannels: layerPass.entry, outChannels: layerPass.output)
            } else {
                // `downsample2d` like `downsample3d`: only `resample.1` works (see the header).
                x = try b.downsample(x, prefix + ".resample.1")
                t = (t.h / 2, t.w / 2)
            }
            if [2, 5, 8, 10].contains(i) { x = probe("down_blocks.\(i)", x) }
        }
        let mid = "encoder.mid_block"
        x = try b.resnetQwen(x, mid + ".resnets.0", inChannels: 384, outChannels: 384)
        let qkv = mid + ".attentions.0.to_qkv"
        x = try b.attention(x, normalized: b.rmsNorm(x, mid + ".attentions.0.norm", channels: 384), channels: 384, size: t,
                            qkv: { (try b.projection($0, qkv, rows: 0..<384), try b.projection($0, qkv, rows: 384..<768),
                                    try b.projection($0, qkv, rows: 768..<1152)) },
                            output: mid + ".attentions.0.proj", requestBlock: EngineSettings.effective.vaeRequestBlock)
        x = try b.resnetQwen(x, mid + ".resnets.1", inChannels: 384, outChannels: 384)
        x = probe("mid_block", x)
        x = try b.silu(b.rmsNorm(x, "encoder.norm_out", channels: 384))
        x = try b.conv(x, "encoder.conv_out", padding: 1)
        // `quant_conv` mixes the 32 channels: the mean also depends on the log-variance half.
        return try b.conv(x, "quant_conv", padding: 0)
    }

    /// `image`: `[3, height, width]` planar, in `[-1, 1]` — or, for an encoder built `startingAt` a
    /// probe, that boundary's tensor.
    package func encode(image: UnsafePointer<Float>) throws -> Output {
        guard let graph, let entry else {
            let begin = Date()
            let memory = try VAEMemory(plan: VAEEncoder.plan(family: family, height: height, width: width),
                                       device: device, queue: queue)
            let bands = VAE.Bands(memory: memory, weights: weights, name: "Flux encoder", groups: 32, band: VAE.bandElements)
            let (latent, moments) = try autoreleasepool {
                try VAEEncoder.encodeBanded(image, family: family, height: height, width: width, bands: bands)
            }
            memory.finish()
            let normalized = batchNorm.map { VAEEncoder.hostLatent(moments: moments, height: height / 8, width: width / 8, batchNorm: $0) } ?? latent
            return Output(latent: normalized, moments: moments, probes: [:], seconds: Date().timeIntervalSince(begin))
        }
        let count = entryShape.reduce(1, *)
        let buffer = device.checkedBuffer(length: count * 4)
        buffer.contents().assumingMemoryBound(to: Float.self).update(from: image, count: count)
        let begin = Date()
        var readValues: [String: [Float]] = [:]
        // The results live in the pool: they are released before the caller continues.
        autoreleasepool {
            let data = MPSGraphTensorData(buffer, shape: entryShape.map { NSNumber(value: $0) }, dataType: .float32)
            let names = Array(targets.keys)
            // **Without MPS's heap cache**: `graph.run` left the encoder's whole peak in MPS's
            // heaps for 3–15 s. Duration 0: the heaps go back with the command buffer.
            let commands = MPSCommandBuffer(from: queue)
            MPSSetHeapCacheDuration(commands, 0)
            let results = graph.encode(to: commands, feeds: [entry: data], targetTensors: names.map { targets[$0]! },
                                       targetOperations: nil, executionDescriptor: nil)
            commands.commit()
            commands.waitUntilCompleted()
            for name in names {
                let tensor = targets[name]!
                let n = (tensor.shape ?? []).reduce(1) { $0 * $1.intValue }
                var values = [Float](repeating: 0, count: n)
                values.withUnsafeMutableBytes { results[tensor]!.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
                readValues[name] = values
            }
        }
        let seconds = Date().timeIntervalSince(begin)
        let moments = readValues.removeValue(forKey: "moments")!
        let latent = readValues.removeValue(forKey: "latent")
            ?? batchNorm.map { VAEEncoder.hostLatent(moments: moments, height: height / 8, width: width / 8, batchNorm: $0) } ?? []
        return Output(latent: latent, moments: moments, probes: readValues, seconds: seconds)
    }

    /// **Flux's encoder by layers and bands** (`VAE.Bands`), live or planned: `conv_in`, the four blocks
    /// (two resnets, a stride-2 convolution but in the last), then the bottleneck and the head by
    /// passes at `H/8 × W/8` — the operations of `flux` in the same order, on the same shapes.
    private static func encodeBanded(_ image: UnsafePointer<Float>?, family: Family, height H: Int, width W: Int,
                                      bands: VAE.Bands) throws -> (latent: [Float], moments: [Float]) {
        let memory = bands.memory
        let input = try memory.register([1, 3, H, W])
        if let image, memory.isLive { input.values.update(from: image, count: input.count) }
        var x = try bands.conv(input, "encoder.conv_in", outChannels: 128)
        memory.free(input)
        var entry = 128
        for (i, output) in [128, 256, 512, 512].enumerated() {
            let prefix = "encoder.down_blocks.\(i)"
            x = try bands.resnet(x, prefix + ".resnets.0", inChannels: entry, outChannels: output)
            x = try bands.resnet(x, prefix + ".resnets.1", inChannels: output, outChannels: output)
            if i < 3 {
                let half = try bands.downsample(x, prefix + ".downsamplers.0.conv")
                memory.free(x)
                x = half
            }
            entry = output
        }
        // The bottleneck and the head at `H/8 × W/8`, by passes: `flux`'s operations in its order.
        let (h, w, c) = (H / 8, W / 8, family.channels), mid = "encoder.mid_block"
        x = try bands.resnet(x, mid + ".resnets.0", inChannels: 512, outChannels: 512)
        try bands.attention(x, mid + ".attentions.0")
        x = try bands.resnet(x, mid + ".resnets.1", inChannels: 512, outChannels: 512)
        var moments = try bands.normConv(x, norm: "encoder.conv_norm_out", conv: "encoder.conv_out", outChannels: 2 * c,
                                         residual: nil, into: nil)
        memory.free(x)
        if family == .flux2 {
            // `quant_conv`, then the pipeline's part on the host (`hostLatent`): the moments, the mean's patched
            // copy and the latent, alive together.
            let projected = try bands.whole([moments], moments.shape, weights: WeightList.conv(2 * c, 2 * c, kernel: 1),
                                            workspace: VAELayout.reformattedWeights(4 * c * c) / 4) { b, p in
                try b.conv(p[0], "quant_conv", padding: 0)
            }
            memory.free(moments)
            moments = projected
            memory.host(4 * (moments.count + c * h * w + c * h * w))
            let values = memory.isLive ? Array(UnsafeBufferPointer(start: moments.values, count: moments.count)) : []
            memory.free(moments)
            return ([], values)
        }
        let latent = try bands.whole([moments], [1, c, h, w], weights: []) { b, p in
            VAEEncoder.normalize(b.graph, b.graph.sliceTensor(p[0], dimension: 1, start: 0, length: c, name: nil), family)
        }
        memory.host(4 * (moments.count + latent.count))
        let values = memory.isLive
            ? (Array(UnsafeBufferPointer(start: latent.values, count: latent.count)),
               Array(UnsafeBufferPointer(start: moments.values, count: moments.count)))
            : ([], [])
        memory.free(latent); memory.free(moments)
        return values
    }

    /// **Everything Flux's encoder holds for an `H × W` image** at a band of `band` elements — its
    /// registers' heap, the largest graph's workspace, the latent and moments it returns:
    /// `encodeBanded` walked without a device. `encode` allocates exactly this before its first graph.
    package static func plan(family: Family, height: Int, width: Int, band: Int = VAE.bandElements) -> VAELayout {
        plans(VAEPlanKey(family, [height, width, band])) { walk(family: family, height: height, width: width, band: band) }
    }

    private static let plans = ShapeMemo<VAEPlanKey<Family>, VAELayout>()

    /// `plan`, walked every time: what the memory keeps, and what the tests compare it with.
    package static func walk(family: Family, height: Int, width: Int, band: Int = VAE.bandElements) -> VAELayout {
        let memory = VAEMemory()
        let bands = VAE.Bands(memory: memory, weights: nil, name: "Flux encoder", groups: 32, band: band)
        do { _ = try encodeBanded(nil, family: family, height: height, width: width, bands: bands) } catch { preconditionFailure("VAE plan walk refused: \(error)") }
        return memory.layout
    }
}
