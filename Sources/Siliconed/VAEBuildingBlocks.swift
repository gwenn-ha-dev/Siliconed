import Metal
import MetalPerformanceShadersGraph
import Foundation

/// **The bricks of both VAEs — decoders and encoders — laid on a graph.**
///
/// They used to live as nested functions in the two decoder `build`s (`VAE`, `AnimaVAE`).
/// img2img brought two encoders made of the same pieces: rather than keep four copies,
/// they are here, once. **The extraction's safeguard is numerical**: the decoders must
/// render the same numbers as before, to the hundredth of a dB (134.70 / 127.92 / 123.71 dB, and the
/// rectangles) and the same render md5s — each brick lays exactly the `MPSGraph` operations
/// it laid before, in the same order.
///
/// A brick captures only the graph and the weight file, never a stage nor a grid:
/// that is what lets it be reused in four different graphs.
struct VAEBuildingBlocks {
    let graph: MPSGraph
    let weights: Safetensors
    /// The VAE's name, for the missing-tensor error.
    let name: String
    /// The `GroupNorm` groups (Flux: 32). Irrelevant for Qwen-Image.
    var groups = 32

    // ── the weights ─────────────────────────────────────────────────────────────────────────

    /// **A missing tensor throws**, it does not stop the process: a user's file (an imported
    /// VAE, truncated, from another family) is an input like any other, and the app that
    /// loaded it must be able to say so instead of dying. All the bricks that read a weight
    /// therefore throw, and the error bubbles up to the VAEs' `init`s, which already throw.
    func values(_ name: String) throws -> (values: [Float], shape: [Int]) {
        guard let values = weights.materialize(name), let entry = weights.entries[name] else {
            throw Safetensors.Failure.missingTensor(file: self.name, name: name)
        }
        return (values, entry.shape)
    }

    func constant(_ values: [Float], _ shape: [Int]) -> MPSGraphTensor {
        values.withUnsafeBytes {
            graph.constant(Data($0), shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        }
    }

    /// A convolution's weight, in `OIHW`. A 3D kernel `[O, I, T, kh, kw]` (Qwen-Image) is
    /// reduced to its **last** temporal slice (`AnimaVAE.lastSlice`); a 2D kernel
    /// (Flux) passes as is.
    func kernel(_ name: String) throws -> MPSGraphTensor {
        let (w, shape) = try values(name)
        let (slice, kernelShape) = AnimaVAE.lastSlice(w, shape: shape)
        return constant(slice, kernelShape)
    }

    /// A weight vector (bias, gamma), reshaped so that it broadcasts.
    func vector(_ name: String, _ shape: [NSNumber]) throws -> MPSGraphTensor {
        let values = try values(name).values
        return graph.reshape(constant(values, [values.count]), shape: shape, name: nil)
    }

    // ── the operations ──────────────────────────────────────────────────────────────────────

    /// A stride-1 convolution, symmetric padding, plus its bias.
    func conv(_ x: MPSGraphTensor, _ prefix: String, padding: Int) throws -> MPSGraphTensor {
        try conv(x, prefix, step: 1, left: padding, right: padding, up: padding, down: padding)
    }

    /// **The encoders' downsampling: `3×3`, stride 2, padding `(0, 1, 0, 1)`** — a zero on the right and
    /// at the bottom, nothing on the left or top. This is Flux's `Downsample2D(padding=0)` (`F.pad(x, (0, 1,
    /// 0, 1))` then `Conv2d(stride=2)`) and Qwen-Image's `ZeroPad2d((0, 1, 0, 1))` + `Conv2d(stride=2)`.
    /// Symmetric padding would give a latent shifted by half a pixel — plausible,
    /// and wrong: the encoder checks refuse it (a counter-test proved it).
    func downsample(_ x: MPSGraphTensor, _ prefix: String) throws -> MPSGraphTensor {
        try conv(x, prefix, step: 2, left: 0, right: 1, up: 0, down: 1)
    }

    private func conv(_ x: MPSGraphTensor, _ prefix: String, step: Int,
                      left: Int, right: Int, up: Int, down: Int) throws -> MPSGraphTensor {
        let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: step, strideInY: step, dilationRateInX: 1, dilationRateInY: 1, groups: 1,
            paddingLeft: left, paddingRight: right, paddingTop: up, paddingBottom: down,
            paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW)!
        let y = try graph.convolution2D(x, weights: kernel(prefix + ".weight"), descriptor: descriptor, name: nil)
        return try graph.addition(y, vector(prefix + ".bias", [1, -1, 1, 1]), name: nil)
    }

    /// `GroupNorm(32)` (Flux): normalization is done per channel group **and** over space,
    /// then the scale and bias are per channel. `h`, `w`: the grid at this depth.
    ///
    /// `statistics`: the `[1, groups, 1]` mean and variance of the **whole** plane, computed apart
    /// (`VAE.statistics`) — what lets a band of rows be normalized as the whole tensor is (`VAE`'s
    /// banded decoder). `nil`: computed here, over `x`.
    func groupNorm(_ x: MPSGraphTensor, _ prefix: String, channels: Int, h: Int, w: Int,
                   statistics: (mean: MPSGraphTensor, variance: MPSGraphTensor)? = nil) throws -> MPSGraphTensor {
        let perGroup = channels / groups
        let grouped = graph.reshape(x, shape: [1, NSNumber(value: groups),
                                               NSNumber(value: perGroup * h * w)], name: nil)
        let (mean, variance) = statistics ?? (graph.mean(of: grouped, axes: [2], name: nil),
                                              graph.variance(of: grouped, axes: [2], name: nil))
        let normalised = graph.normalize(grouped, mean: mean, variance: variance,
                                         gamma: nil, beta: nil, epsilon: 1e-6, name: nil)
        let restored = graph.reshape(normalised, shape: [1, NSNumber(value: channels),
                                                         NSNumber(value: h), NSNumber(value: w)], name: nil)
        let gamma = try vector(prefix + ".weight", [1, -1, 1, 1])
        let beta = try vector(prefix + ".bias", [1, -1, 1, 1])
        return graph.addition(graph.multiplication(restored, gamma, name: nil), beta, name: nil)
    }

    /// `QwenImageRMS_norm`: `F.normalize(x, dim=1) · √C · γ` — the L2 norm over the channels,
    /// bounded from below at 1e-12 like `F.normalize`, and **not** a mean of squares.
    func rmsNorm(_ x: MPSGraphTensor, _ prefix: String, channels: Int) throws -> MPSGraphTensor {
        let squares = graph.reductionSum(with: graph.multiplication(x, x, name: nil), axis: 1, name: nil)
        let norm = graph.maximum(graph.squareRoot(with: squares, name: nil),
                                  graph.constant(1e-12, dataType: .float32), name: nil)
        let gamma = try graph.multiplication(vector(prefix + ".gamma", [1, -1, 1, 1]),
                                         graph.constant(Double(channels).squareRoot(), dataType: .float32),
                                         name: nil)
        return graph.multiplication(graph.division(x, norm, name: nil), gamma, name: nil)
    }

    func silu(_ x: MPSGraphTensor) -> MPSGraphTensor {
        graph.multiplication(x, graph.sigmoid(with: x, name: nil), name: nil)
    }

    /// Flux's `ResnetBlock2D` (eps 1e-6, no `temb`); `conv_shortcut` on the raw input.
    func resnetFlux(_ x: MPSGraphTensor, _ prefix: String, inChannels: Int, outChannels: Int,
                    size: (h: Int, w: Int)) throws -> MPSGraphTensor {
        var h = try silu(groupNorm(x, prefix + ".norm1", channels: inChannels, h: size.h, w: size.w))
        h = try conv(h, prefix + ".conv1", padding: 1)
        h = try silu(groupNorm(h, prefix + ".norm2", channels: outChannels, h: size.h, w: size.w))
        h = try conv(h, prefix + ".conv2", padding: 1)
        let residual = try inChannels == outChannels ? x : conv(x, prefix + ".conv_shortcut", padding: 0)
        return graph.addition(residual, h, name: nil)
    }

    /// `QwenImageResidualBlock`: the same shape, under RMSNorm, with no spatial statistic.
    func resnetQwen(_ x: MPSGraphTensor, _ prefix: String, inChannels: Int, outChannels: Int) throws -> MPSGraphTensor {
        let residual = try inChannels == outChannels ? x : conv(x, prefix + ".conv_shortcut", padding: 0)
        var h = try silu(rmsNorm(x, prefix + ".norm1", channels: inChannels))
        h = try conv(h, prefix + ".conv1", padding: 1)
        h = try silu(rmsNorm(h, prefix + ".norm2", channels: outChannels))
        h = try conv(h, prefix + ".conv2", padding: 1)
        return graph.addition(h, residual, name: nil)
    }

    /// A linear projection of the `[1, S, C]` tokens: rows `rows` of the `[O, I]` weight (or
    /// `[O, I, 1, 1]`, a 1×1 `Conv2d`), then the bias. Qwen-Image's `to_qkv` is split this way into
    /// q, k, v, in that order.
    func projection(_ tokens: MPSGraphTensor, _ name: String, rows: Range<Int>? = nil) throws -> MPSGraphTensor {
        let (w, shape) = try values(name + ".weight")
        let (b, _) = try values(name + ".bias")
        let inputs = shape[1]
        let r = rows ?? 0..<shape[0]
        let weights = r == 0..<shape[0] ? w : Array(w[(r.lowerBound * inputs)..<(r.upperBound * inputs)])
        let bias = r == 0..<shape[0] ? b : Array(b[r])
        let wt = graph.transposeTensor(constant(weights, [r.count, inputs]),
                                       dimension: 0, withDimension: 1, name: nil)
        let y = graph.matrixMultiplication(primary: tokens, secondary: wt, name: nil)
        return graph.addition(y, graph.reshape(constant(bias, [r.count]), shape: [1, 1, -1], name: nil),
                              name: nil)
    }

    /// **The bottleneck attention**: one head, `S = h·w` tokens of `C` channels, scale `1/√C`, then the
    /// output projection and the residual. `normalized` is the input already normalized (GroupNorm for
    /// Flux, RMSNorm for Qwen-Image); `qkv` projects the tokens.
    ///
    /// `requestBlock > 0` splits the `S×S` matrix into slices of queries — the same sum, row
    /// by row: that is `vae_bloc_requetes`, room against a bit of time. See the
    /// `VAE` note on the bottleneck (2,520 MB at 1024²). The Qwen-Image decoder passes 0:
    /// it was never split, and the extraction must not be what changes its bits.
    func attention(_ x: MPSGraphTensor, normalized: MPSGraphTensor, channels: Int, size: (h: Int, w: Int),
                   qkv: (MPSGraphTensor) throws -> (MPSGraphTensor, MPSGraphTensor, MPSGraphTensor),
                   output: String, requestBlock: Int) throws -> MPSGraphTensor {
        let c = NSNumber(value: channels), tokensCount = size.h * size.w
        // [1, C, H, W] → [1, H·W, C]
        let tokens = graph.transposeTensor(graph.reshape(normalized, shape: [1, c, NSNumber(value: tokensCount)],
                                                         name: nil),
                                           dimension: 1, withDimension: 2, name: nil)
        let (q, k, v) = try qkv(tokens)
        // A single head: the 4D shape is `[1, 1, S, C]`, and `scale` is passed explicitly —
        // `1/√C`, not the `1/√headDim` the API would take by default.
        let s4 = { (t: MPSGraphTensor) in graph.expandDims(t, axis: 1, name: nil) }
        let scale = 1 / Float(channels).squareRoot()
        let attended: MPSGraphTensor
        if requestBlock > 0 && requestBlock < tokensCount {
            var slices: [MPSGraphTensor] = []
            var begin = 0
            while begin < tokensCount {
                let length = min(requestBlock, tokensCount - begin)
                let qSlice = graph.sliceTensor(q, dimension: 1, start: begin, length: length, name: nil)
                let output = graph.scaledDotProductAttention(query: s4(qSlice), key: s4(k), value: s4(v),
                                                             mask: nil, scale: scale, name: nil)
                slices.append(graph.squeeze(output, axis: 1, name: nil))
                begin += length
            }
            attended = graph.concatTensors(slices, dimension: 1, name: nil)
        } else {
            attended = graph.squeeze(
                graph.scaledDotProductAttention(query: s4(q), key: s4(k), value: s4(v), mask: nil,
                                                scale: scale, name: nil),
                axis: 1, name: nil)
        }
        let out = try projection(attended, output)
        let back = graph.reshape(graph.transposeTensor(out, dimension: 1, withDimension: 2, name: nil),
                                 shape: [1, c, NSNumber(value: size.h), NSNumber(value: size.w)], name: nil)
        return graph.addition(x, back, name: nil)
    }

    /// **`nearest-exact` ×2, written as a duplication.** No `resize`: its flags (centering,
    /// alignment, rounding direction) are a convention, and MPSGraph's shifted by one pixel at
    /// each stage, i.e. 4 + 2 + 1 = 7 pixels on the final image. Each pixel becomes its 2×2
    /// block: we unfold a dimension, broadcast it, fold it back, in width then in height.
    func double(_ x: MPSGraphTensor, channels: Int, size: (h: Int, w: Int)) -> MPSGraphTensor {
        let c = NSNumber(value: channels), h = NSNumber(value: size.h), w = NSNumber(value: size.w)
        let doubledH = NSNumber(value: size.h * 2), doubledW = NSNumber(value: size.w * 2)
        var big = graph.reshape(x, shape: [1, c, h, w, 1], name: nil)
        big = graph.broadcast(big, shape: [1, c, h, w, 2], name: nil)
        big = graph.reshape(big, shape: [1, c, h, 1, doubledW], name: nil)
        big = graph.broadcast(big, shape: [1, c, h, 2, doubledW], name: nil)
        return graph.reshape(big, shape: [1, c, doubledH, doubledW], name: nil)
    }
}
