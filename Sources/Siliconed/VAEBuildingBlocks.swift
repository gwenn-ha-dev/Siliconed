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
/// **The weights a graph reads, as placeholders fed from our registers** — the order in which
/// the graph reads them, each with what to widen into its register. A graph built with them holds no
/// weight of its own: `MPSGraph` copied each constant into Metal memory of its own, which went back
/// to the system only some time after the graph died (1.16 GB of the Qwen decoder's still counted at its
/// return), and its `Data` into `malloc` before that. Fed or constant, the same values reach the same
/// kernels: the same bits (md5 of all four VAEs).
final class FedWeights {
    var items: [(placeholder: MPSGraphTensor, name: String, rows: Range<Int>?, shape: [Int])] = []
}

struct VAEBuildingBlocks {
    let graph: MPSGraph
    let weights: Safetensors
    /// The VAE's name, for the missing-tensor error.
    let name: String
    /// The `GroupNorm` groups (Flux: 32). Irrelevant for Qwen-Image.
    var groups = 32
    /// When set, every weight becomes a placeholder recorded here (`VAEMemory.weights` feeds them);
    /// `nil`: a constant of the graph (the checks' one-piece graphs).
    var fed: FedWeights? = nil
    /// The type of the activations and of the fed weights: fp32, or bf16 for a decoder in half precision (`VAEMemory.half`).
    var dataType: MPSDataType = .float32

    init(graph: MPSGraph, weights: Safetensors, name: String, groups: Int = 32, fed: FedWeights? = nil,
         dataType: MPSDataType = .float32) {
        self.graph = graph; self.weights = weights; self.name = name; self.groups = groups; self.fed = fed
        self.dataType = dataType
    }

    // ── the weights ─────────────────────────────────────────────────────────────────────────

    /// **A missing tensor throws**, it does not stop the process: a user's file (an imported
    /// VAE, truncated, from another family) is an input like any other, and the app that
    /// loaded it must be able to say so instead of dying. All the bricks that read a weight
    /// therefore throw, and the error bubbles up to the VAEs' `init`s, which already throw.
    ///
    /// `rows`: the rows of a 2D (or `[O, I, 1, 1]`) weight; a 5D kernel `[O, I, T, kh, kw]`
    /// (Qwen-Image) gives its **last** temporal slice (`AnimaVAE.lastSlice`). Returns the shape and
    /// the pieces of the source to copy, in elements, in order.
    static func layout(_ file: Safetensors, _ name: String, rows: Range<Int>?, vae: String)
            throws -> (shape: [Int], pieces: [(start: Int, count: Int)]) {
        guard let entry = file.entries[name], ["F32", "BF16", "F16"].contains(entry.dtype) else {
            throw Safetensors.Failure.missingTensor(file: vae, name: name)
        }
        var shape = entry.shape
        if shape.count == 5 {
            let (o, i, t, kh, kw) = (shape[0], shape[1], shape[2], shape[3], shape[4])
            shape = [o, i, kh, kw]
            return (shape, (0..<(o * i)).map { ((($0 * t) + (t - 1)) * kh * kw, kh * kw) })
        }
        if let rows, rows != 0..<shape[0] {
            let perRow = shape.dropFirst().reduce(1, *)
            shape[0] = rows.count
            return (shape, [(rows.lowerBound * perRow, rows.count * perRow)])
        }
        return (shape, [(0, entry.count)])
    }

    /// The weight widened to fp32 into `target` — the widening of `Safetensors.materialize`, the same
    /// values, with no `[Float]` between.
    static func widen(_ file: Safetensors, _ name: String, rows: Range<Int>?, vae: String,
                      into target: UnsafeMutablePointer<Float>) throws {
        let (_, pieces) = try layout(file, name, rows: rows, vae: vae)
        let entry = file.entries[name]!, source = file.pointer(name)!
        let size = entry.dtype == "F32" ? 4 : 2
        var target = target
        for piece in pieces {
            let from = source + piece.start * size
            switch entry.dtype {
            case "F32": UnsafeMutableRawPointer(target).copyMemory(from: from, byteCount: 4 * piece.count)
            case "BF16": Widen.bfloat16ToFloat32(source: from, destination: UnsafeMutableRawPointer(target), count: piece.count)
            default:
                let half = from.assumingMemoryBound(to: Float16.self)
                for k in 0..<piece.count { target[k] = Float(half[k]) }
            }
            target += piece.count
        }
    }

    /// **The weight as published, bf16, copied bit for bit into `target`** — a decoder in half precision reads its weights
    /// so (`VAEMemory.half`). A tensor published in another format is refused: rounding a weight would compute under it.
    static func copyHalf(_ file: Safetensors, _ name: String, rows: Range<Int>?, vae: String,
                         into target: UnsafeMutableRawPointer) throws {
        let (_, pieces) = try layout(file, name, rows: rows, vae: vae)
        let entry = file.entries[name]!, source = file.pointer(name)!
        guard entry.dtype == "BF16" else {
            throw Safetensors.Failure.badHeader("\(vae): \(name) is published in \(entry.dtype), not bf16 — the decoder computes in fp32")
        }
        var target = target
        for piece in pieces {
            target.copyMemory(from: source + 2 * piece.start, byteCount: 2 * piece.count)
            target += 2 * piece.count
        }
    }

    /// **Whether every decoder tensor of `file` is published in bf16** — what lets a decoder compute in bf16 (`VAEPrecision`).
    static func decoderInBF16(_ file: Safetensors) -> Bool {
        let decoder = file.entries.filter { $0.key.hasPrefix("decoder.") || $0.key.hasPrefix("post_quant_conv.") }
        return !decoder.isEmpty && decoder.values.allSatisfy { $0.dtype == "BF16" }
    }

    /// **A weight in the graph**: a placeholder when `fed`, otherwise a constant widened into pages of
    /// ours (`Pages`), never into `malloc` — on macOS 27 a large `malloc` block freed stays counted.
    /// `shape`: the shape the graph takes it at, as many floats as the weight's.
    private func tensor(_ name: String, rows: Range<Int>? = nil, shape: [Int]? = nil) throws -> MPSGraphTensor {
        let (natural, _) = try VAEBuildingBlocks.layout(weights, name, rows: rows, vae: self.name)
        let shape = shape ?? natural
        precondition(shape.reduce(1, *) == natural.reduce(1, *), "VAEBuildingBlocks: \(name) reshaped to another size")
        let ns = shape.map { NSNumber(value: $0) }
        if let fed {
            let p = graph.placeholder(shape: ns, dataType: dataType, name: nil)
            fed.items.append((p, name, rows, shape))
            return p
        }
        let count = shape.reduce(1, *)
        let pages = Pages(bytes: 4 * count)
        try VAEBuildingBlocks.widen(weights, name, rows: rows, vae: self.name, into: pages.floats())
        return graph.constant(VAEBuildingBlocks.data(pages, bytes: 4 * count), shape: ns, dataType: .float32)
    }

    /// `pages` as a `Data` that owns them: the graph's constant keeps the pages, and its release returns them.
    static func data(_ pages: Pages, bytes: Int) -> Data {
        Data(bytesNoCopy: pages.base, count: bytes, deallocator: .custom { _, _ in withExtendedLifetime(pages) {} })
    }

    /// A constant from host values (a VAE's normalization vectors: a few floats), in pages like the weights.
    func constant(_ values: [Float], _ shape: [Int]) -> MPSGraphTensor {
        let pages = Pages(bytes: 4 * values.count)
        values.withUnsafeBytes { UnsafeMutableRawPointer(pages.base).copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return graph.constant(VAEBuildingBlocks.data(pages, bytes: 4 * values.count),
                              shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    }

    /// A convolution's weight, in `OIHW`. A 3D kernel `[O, I, T, kh, kw]` (Qwen-Image) is
    /// reduced to its **last** temporal slice (`AnimaVAE.lastSlice`); a 2D kernel
    /// (Flux) passes as is.
    func kernel(_ name: String) throws -> MPSGraphTensor {
        try tensor(name)
    }

    /// A weight vector (bias, gamma), reshaped so that it broadcasts.
    func vector(_ name: String, _ shape: [NSNumber]) throws -> MPSGraphTensor {
        let count = try VAEBuildingBlocks.layout(weights, name, rows: nil, vae: self.name).shape.reduce(1, *)
        return graph.reshape(try tensor(name, shape: [count]), shape: shape, name: nil)
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
        return try affine(restored, prefix)
    }

    /// `GroupNorm`'s normalization alone, on `[1, groups, C/groups·H·W]` with its statistics given — the
    /// first half of `groupNorm`, a pass of its own (`VAE.Bands`).
    func groupNormalize(_ grouped: MPSGraphTensor, mean: MPSGraphTensor, variance: MPSGraphTensor) -> MPSGraphTensor {
        graph.normalize(grouped, mean: mean, variance: variance, gamma: nil, beta: nil, epsilon: 1e-6, name: nil)
    }

    /// `GroupNorm`'s scale and bias, per channel, on the normalized `[1, C, H, W]` — its second half.
    func affine(_ restored: MPSGraphTensor, _ prefix: String) throws -> MPSGraphTensor {
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
        let shape = try VAEBuildingBlocks.layout(weights, name + ".weight", rows: rows, vae: self.name).shape
        let (count, inputs) = (shape[0], shape[1])
        let wt = graph.transposeTensor(try tensor(name + ".weight", rows: rows, shape: [count, inputs]),
                                       dimension: 0, withDimension: 1, name: nil)
        let y = graph.matrixMultiplication(primary: tokens, secondary: wt, name: nil)
        return graph.addition(y, graph.reshape(try tensor(name + ".bias", rows: rows, shape: [count]), shape: [1, 1, -1], name: nil),
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
        let (q, k, v) = try qkv(tokens(normalized, channels: channels, count: size.h * size.w))
        let attended = scaledDotProduct(q, k, v, channels: channels, requestBlock: requestBlock)
        return try restore(projection(attended, output), into: x, channels: channels, size: size)
    }

    /// The attention's tokens: `[1, C, H, W]` → `[1, H·W, C]`.
    func tokens(_ normalized: MPSGraphTensor, channels: Int, count: Int) -> MPSGraphTensor {
        graph.transposeTensor(graph.reshape(normalized, shape: [1, NSNumber(value: channels), NSNumber(value: count)], name: nil),
                              dimension: 1, withDimension: 2, name: nil)
    }

    /// One head over `[1, S, C]` queries, keys and values, scale `1/√C` — the whole `S×S` matrix, or by
    /// slices of `requestBlock` queries (the same sum, row by row).
    func scaledDotProduct(_ q: MPSGraphTensor, _ k: MPSGraphTensor, _ v: MPSGraphTensor, channels: Int,
                          requestBlock: Int) -> MPSGraphTensor {
        let tokensCount = q.shape![1].intValue
        // A single head: the 4D shape is `[1, 1, S, C]`, and `scale` is passed explicitly —
        // `1/√C`, not the `1/√headDim` the API would take by default.
        let s4 = { (t: MPSGraphTensor) in self.graph.expandDims(t, axis: 1, name: nil) }
        let scale = 1 / Float(channels).squareRoot()
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
            return graph.concatTensors(slices, dimension: 1, name: nil)
        }
        return graph.squeeze(graph.scaledDotProductAttention(query: s4(q), key: s4(k), value: s4(v), mask: nil,
                                                             scale: scale, name: nil),
                             axis: 1, name: nil)
    }

    /// `x + out`, the projected `[1, S, C]` tokens laid back as `[1, C, H, W]`: the attention's residual.
    func restore(_ out: MPSGraphTensor, into x: MPSGraphTensor, channels: Int, size: (h: Int, w: Int)) -> MPSGraphTensor {
        let back = graph.reshape(graph.transposeTensor(out, dimension: 1, withDimension: 2, name: nil),
                                 shape: [1, NSNumber(value: channels), NSNumber(value: size.h), NSNumber(value: size.w)], name: nil)
        return graph.addition(x, back, name: nil)
    }

}

// ── what each brick reads ──────────────────────────────────────────────────────────────────────

/// **The floats of every weight a brick reads, in the order it reads them** — what a plan charges
/// for a graph's fed weights (`VAEMemory.weights`), from the widths alone. A graph whose bricks read
/// otherwise stops at its first run (the fed list is checked against this one, element by element).
enum WeightList {
    /// A convolution: its kernel, then its bias.
    static func conv(_ output: Int, _ input: Int, kernel k: Int = 3) -> [Int] { [output * input * k * k, output] }
    static func groupNorm(_ c: Int) -> [Int] { [c, c] }
    static func rms(_ c: Int) -> [Int] { [c] }
    static func projection(_ rows: Int, _ inputs: Int) -> [Int] { [rows * inputs, rows] }

    /// `VAEBuildingBlocks.resnetQwen`: the shortcut first, then norm1, conv1, norm2, conv2.
    static func resnetQwen(_ i: Int, _ o: Int) -> [Int] {
        (i == o ? [] : conv(o, i, kernel: 1)) + rms(i) + conv(o, i) + rms(o) + conv(o, o)
    }
}
