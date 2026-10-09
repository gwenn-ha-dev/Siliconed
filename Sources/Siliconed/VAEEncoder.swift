import Metal
import MetalPerformanceShadersGraph
import Foundation

/// **The encoders of both VAEs, in fp32, via `MPSGraph` — img2img's input.**
///
///     Flux (Z-Image)     image [3, H, W] ─ conv_in ─ 4 blocks (2 resnets, ÷2 except the last)
///                          ─ mid(resnet, attention, resnet) ─ GroupNorm ─ SiLU ─ conv_out → moments [32, H/8, W/8]
///     Qwen-Image         image [3, H, W] ─ conv_in ─ 11 flat layers (resnets, ÷2 at 2, 5, 8)
///     (Anima, Krea 2)      ─ mid(resnet, attention, resnet) ─ RMSNorm ─ SiLU ─ conv_out ─ quant_conv → moments
///
///     moments = [mean (16) | log-variance (16)]      latent = normalize(mean)
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
/// ## The mean, not a sample — a deliberate departure from diffusers
///
/// The encoder returns a diagonal Gaussian; we take its **mean** (`mode()`), and the render
/// stays deterministic for a given seed. `ZImageImg2ImgPipeline` and `QwenImageImg2ImgPipeline` draw,
/// for their part, a **sample** (`retrieve_latents(…, sample_mode="sample")`, `mean + std·ε`) with the
/// generator, before the noise — only Anima's modular block takes `argmax`, like us. The deviation
/// is `std·ε`, with a `std` that the `vae-encode` check displays (small next to the latent's
/// standard deviation). The img2img oracles take the mean too.
///
/// ## Memory
///
/// A single graph, like Flux's decoder by default (measured: cutting into stages makes the
/// peak *rise*). The bottleneck is at the same grid as the decoder's, so its `S×S` matrix weighs as much
/// (2.5 GB at 1024²): it follows the same slicing setting, `vae_bloc_requetes`.
package final class VAEEncoder {
    package enum Family: Sendable {
        /// Flux `AutoencoderKL` (Z-Image): GroupNorm, no `quant_conv`.
        case flux
        /// `AutoencoderKLQwenImage` (Anima, Krea 2): RMSNorm, causal 3D kernels, `quant_conv`.
        case qwenImage
        /// `AutoencoderKLFlux2` (FLUX.2): the 32-channel Flux encoder, followed by a `quant_conv`.
        /// The returned latent is the **raw mean** `[32, H/8, W/8]`: the 2×2 packing and the
        /// `BatchNorm` are done by `Flux2EncodingModule`, mirror of `Flux2DecodingModule`.
        case flux2

        /// The latent channels (half of the moments).
        var channels: Int { self == .flux2 ? 32 : 16 }
    }

    /// What an encoding returns: the normalized latent — the one the denoiser expects —, the
    /// raw moments (for the checks) and, on request, the per-block probes.
    package struct Output {
        /// `[16, H/8, W/8]`, normalized in the denoiser's space.
        package let latent: [Float]
        /// `[32, H/8, W/8]`: mean then log-variance, before normalization.
        package let moments: [Float]
        package let probes: [String: [Float]]
        package let seconds: Double
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let graph: MPSGraph
    private let entry: MPSGraphTensor
    private let entryShape: [Int]
    private let targets: [String: MPSGraphTensor]
    package let family: Family
    package let height, width: Int
    package var latentGridHeight: Int { height / 8 }
    package var latentGridWidth: Int { width / 8 }

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
        guard weights.entries["encoder.conv_in.weight"]?.shape == expectedShape,
              family != .flux2 || weights.entries["quant_conv.weight"] != nil else {
            throw Safetensors.Failure.badHeader("\(path): not the VAE encoder "
                                                + [.flux: "of Flux", .qwenImage: "of Qwen-Image", .flux2: "of FLUX.2"][family]!)
        }
        guard height % 8 == 0, width % 8 == 0 else {
            throw EngineError.formatRefused(width: width, height: height, reason: .notMultiple)
        }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw GEMM.Failure.noDevice
        }
        self.device = device; self.queue = queue
        self.family = family; self.height = height; self.width = width
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
        case .flux, .flux2:
            moments = try VAEEncoder.flux(graph, weights, input, (height, width), quantConv: family == .flux2, probe: probe)
        case .qwenImage: moments = try VAEEncoder.qwen(graph, weights, input, (height, width), probe: probe)
        }
        if let startingAt, entry === input {
            throw Safetensors.Failure.badHeader("\(path): no probe \(startingAt) in this encoder")
        }
        let mean = graph.sliceTensor(moments, dimension: 1, start: 0, length: family.channels, name: nil)
        var targets = ["moments": moments, "latent": VAEEncoder.normalize(graph, mean, family)]
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
        case .flux2:
            return z   // packed then normalized outside the graph (`Flux2EncodingModule`)
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
        }
    }

    /// Flux's encoder (`diffusers/models/autoencoders/vae.py`, `Encoder`): `block_out_channels
    /// [128, 256, 512, 512]`, two resnets per block, a downsampling after each except the last.
    private static func flux(_ graph: MPSGraph, _ weights: Safetensors, _ input: MPSGraphTensor,
                             _ size: (h: Int, w: Int), quantConv: Bool = false,
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
        // `use_quant_conv` is false in Z-Image's config: `conv_out` returns the moments. True
        // for FLUX.2, and the 1×1 `quant_conv` mixes mean and log-variance.
        x = try b.conv(x, "encoder.conv_out", padding: 1)
        return quantConv ? try b.conv(x, "quant_conv", padding: 0) : x
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
    package func encode(image: UnsafePointer<Float>) -> Output {
        let count = entryShape.reduce(1, *)
        let buffer = device.makeBuffer(length: count * 4, options: .storageModeShared)!
        buffer.contents().assumingMemoryBound(to: Float.self).update(from: image, count: count)
        let begin = Date()
        var readValues: [String: [Float]] = [:]
        // The results live in the pool: they are released before the caller continues.
        autoreleasepool {
            let data = MPSGraphTensorData(buffer, shape: entryShape.map { NSNumber(value: $0) }, dataType: .float32)
            let names = Array(targets.keys)
            let results = graph.run(with: queue, feeds: [entry: data],
                                       targetTensors: names.map { targets[$0]! }, targetOperations: nil)
            for name in names {
                let tensor = targets[name]!
                let n = (tensor.shape ?? []).reduce(1) { $0 * $1.intValue }
                var values = [Float](repeating: 0, count: n)
                values.withUnsafeMutableBytes { results[tensor]!.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
                readValues[name] = values
            }
        }
        let seconds = Date().timeIntervalSince(begin)
        let latent = readValues.removeValue(forKey: "latent")!, moments = readValues.removeValue(forKey: "moments")!
        return Output(latent: latent, moments: moments, probes: readValues, seconds: seconds)
    }
}
