import CoreAI
import Foundation
import SiliCore

/// **A VAE of Flux's kind as functions** (diffusers' `AutoencoderKL` and `AutoencoderKLFlux2`): its decoder and its
/// encoder, every layer past the trunk by bands of rows (`Bands`), each band a call of a function of fixed shape — one
/// function serves every band of its layer, and every layer of its shape. Every weight is an argument, a resident of
/// the arena, a convolution's kernel flat (`Bands.flat`) and reshaped inside: the GPU runs it at the constant's rate,
/// the same bits (measured 19/11, L2), and no asset holds a weight. The plan and the pass are `FluxVAEBands.swift`.
///
///     decoder
///     head       the latent to the trunk · conv_in · mid (resnet, attention, resnet)                 whole, its plane
///     conv1      GN(x) · SiLU · conv1, by bands                                                     h, its statistics
///     conv2      GN(h) · SiLU · conv2 + shortcut(x), by bands                                       x′, its statistics
///     up         nearest ×2 · conv, by bands (each 2r output rows, twice as wide, from r input rows) y, its statistics
///     out        GN · SiLU · conv_out [· clamp], by bands                                           the picture
///     encoder
///     first      conv_in on the picture, by bands                                                   x, its statistics
///     conv1, conv2                                                                                  the decoder's
///     down       zero pad right and bottom · 3×3 at stride 2, by bands of output rows               y, its statistics
///     tail       mid (resnet, attention, resnet) · GN · SiLU · conv_out to the latent              whole
///
/// **A GroupNorm's statistics are carried from band to band** (`Bands.Plane.carry`): each band of a layer reads what
/// the bands before it gave — each group's count, mean and sum of squared deviations, `[3, G]` f32 — and gives them
/// with its own merged (Chan's: the means' spread added to the sums, never E[x²] − E[x]²); the layer that reads the
/// plane takes them from its last band. Its rows past the plane (a band past its end) are counted out by its mask.
///
/// **Two arithmetics** (`half`), the one each VAE's reference decodes in:
///   - **f16 under the guard** (`Guard.swift`; Flux's VAE, judged against diffusers in bf16):
///       - *a stored tensor is a kept value*: t in f16, F = 2^e a row (f32), its rows' largest in [2¹³, 2¹⁴] (`stream`);
///       - *a GroupNorm*'s statistics are computed in f32 from the true value; its affine is folded with the row's
///         factor (a = F · rstd · γ by row and channel, b = β − mean · rstd · γ by channel, in f32, then f16), so that
///         the band is normalized in f16 as it is read: t · a + b, its error that of f16 on |x − mean| / σ plus
///         |mean| / σ, below a bf16 input's on the same (torch's `group_norm` on bf16);
///       - *a convolution* reads its window brought under 2^e₀ by one factor, e₀ from its own kernel (in · 3 · 3 ·
///         max|w| · 2^e₀ ≤ 65 504: its result is bounded whatever it reads), and its result is multiplied out in f32
///         with its bias, where a factor cannot pass;
///       - *the trunk's attention* reads q, k, v written true in f16: a normalized row times weights.
///   - **f32** (FLUX.2's VAE, whose pipeline decodes and encodes in f32 and publishes f32 weights): every value true.
///
/// The rows past the plane are the convolution's zero padding of what it reads — after the norm and the SiLU, for the ×2
/// before the doubling, for the ↓ as it reads: a window's mask (1 inside the plane, 0 past its edges) multiplies them
/// out. A product, not a select: what lies there is a row of the plane's own, finite (a band past the plane's end is
/// written zeros, `valid`).
package struct FluxVAE: Sendable, Hashable {
    /// Its assets' prefix.
    package var name: String
    /// f16 under the guard, or f32.
    package let half: Bool
    /// The decoder's up blocks, from the trunk to the picture (three resnets each, a ×2 after each but the last), and
    /// the encoder's down blocks (two resnets each, a ↓ after each but the last).
    package let decoderWidths: [Int]
    package let encoderWidths: [Int]
    package let latent: Latent
    /// The picture clamped to [−1, 1] by the decoder (a pipeline's own clamp).
    package let clamped: Bool
    package static let groups = 32, eps = 1e-6

    /// **What the trunk reads of the latent**, and what the encoder's tail gives back.
    package enum Latent: Sendable, Hashable {
        /// `[channels, h, w]`, decoded as z / scaling + shift; the encoder gives the posterior's mean, its first
        /// `channels` moments (the caller shifts and scales it).
        case scaled(channels: Int, scaling: Double, shift: Double)
        /// `[4 · channels, h, w]` under batch-norm statistics (`bn.running_mean`, `bn.running_var`, `eps`): its 2×2
        /// cells unpatchified to `[channels, 2h, 2w]`, then a 1×1 (`post_quant_conv`); the encoder's tail a 1×1
        /// (`quant_conv`), the first `channels` moments, patchified and normalized.
        case patched(channels: Int, eps: Double)
    }

    package init(name: String, half: Bool, decoderWidths: [Int], encoderWidths: [Int], latent: Latent, clamped: Bool) {
        (self.name, self.half, self.decoderWidths, self.encoderWidths, self.latent, self.clamped) = (name, half, decoderWidths, encoderWidths, latent, clamped)
    }

    /// The trunk's channels, where the decoder's planes start.
    var trunk: Int { decoderWidths[0] }
    /// The latent's channels on the planes, and what the head reads.
    var channels: Int { switch latent { case .scaled(let c, _, _), .patched(let c, _): c } }
    var latentChannels: Int { if case .patched(let c, _) = latent { 4 * c } else { channels } }
    /// The planes' rows and columns for one of the latent's.
    var cell: Int { if case .patched = latent { 2 } else { 1 } }
    /// The picture's pixels for one of the latent's.
    package var factor: Int { 8 * cell }
    /// What a stored plane is.
    var element: NDArray.ScalarType { half ? .float16 : .float32 }
    private var stored: Aicode.Element { half ? .f16 : .f32 }

    /// A layer's kind: what its function computes, whatever its layer (its weights are arguments).
    package enum Kind: String, Sendable, Hashable { case head, conv1, conv2, up, out, first, down, tail }

    /// What a GroupNorm's plane carries from band to band: each group's count, mean and sum of squared deviations.
    static var statistics: [Int] { [3, groups] }

    /// **The function of a layer of `kind`**, `cin` channels to `cout`: its key, its module at the plan's shapes.
    func function(_ kind: Kind, cin: Int, cout: Int) -> Bands.Function {
        let vae = self
        return Bands.Function("\(name)-\(kind.rawValue)-\(cin)x\(cout)") { g in try vae.module(kind, cin: cin, cout: cout, g) }
    }

    // MARK: - The weights

    /// A weight a function reads: its argument, the file's key, its shape (a kernel's `[out, in, k, k]`, given flat:
    /// `Bands.flat`), its element.
    struct Weight {
        let name, key: String
        let full: [Int]
        let element: Aicode.Element
        var kernel: Bool { full.count == 4 }
        var shape: [Int] { kernel ? Bands.flat(full, element) : full }
        init(_ name: String, _ key: String, _ full: [Int], _ element: Aicode.Element) { (self.name, self.key, self.full, self.element) = (name, key, full, element) }
    }

    private func kernel(_ o: Int, _ i: Int, _ k: Int) -> [Int] { [o, i, k, k] }

    private func resnetWeights(_ p: String, _ q: String, _ cin: Int, _ cout: Int) -> [Weight] {
        [Weight("\(q)_gamma1", "\(p).norm1.weight", [cin], .f32), Weight("\(q)_beta1", "\(p).norm1.bias", [cin], .f32),
         Weight("\(q)_kernel1", "\(p).conv1.weight", kernel(cout, cin, 3), stored), Weight("\(q)_bias1", "\(p).conv1.bias", [cout], .f32),
         Weight("\(q)_gamma2", "\(p).norm2.weight", [cout], .f32), Weight("\(q)_beta2", "\(p).norm2.bias", [cout], .f32),
         Weight("\(q)_kernel2", "\(p).conv2.weight", kernel(cout, cout, 3), stored), Weight("\(q)_bias2", "\(p).conv2.bias", [cout], .f32)]
    }

    /// The mid block of `side` (`decoder`, `encoder`): two resnets and the attention, the projections `[in, out]`.
    private func midWeights(_ side: String, _ c: Int) -> [Weight] {
        let a = "\(side).mid_block.attentions.0"
        return resnetWeights("\(side).mid_block.resnets.0", "r0", c, c) + resnetWeights("\(side).mid_block.resnets.1", "r1", c, c)
            + [Weight("attn_gamma", "\(a).group_norm.weight", [c], .f32), Weight("attn_beta", "\(a).group_norm.bias", [c], .f32)]
            + [("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("o", "to_out.0")].flatMap { p, n in
                [Weight("attn_\(p)", "\(a).\(n).weight", [c, c], stored), Weight("attn_\(p)_bias", "\(a).\(n).bias", [c], .f32)]
            }
    }

    /// **The weights a layer of `kind` reads**, by argument: the file's key under `layer` (`decoder.up_blocks.1.resnets.2`,
    /// `decoder.up_blocks.1`, `encoder.down_blocks.0.downsamplers.0.conv`), its shape and element as the function
    /// declares it.
    func weights(_ kind: Kind, layer p: String, cin: Int, cout: Int) -> [Weight] {
        let batchNorm = [Weight("bn_mean", "bn.running_mean", [latentChannels], .f32), Weight("bn_var", "bn.running_var", [latentChannels], .f32)]
        switch kind {
        case .head:
            let c = trunk
            var w = midWeights("decoder", c) + [Weight("conv_in_bias", "decoder.conv_in.bias", [c], .f32)]
            if case .patched(let k, _) = latent {
                w += [Weight("post_quant", "post_quant_conv.weight", kernel(k, k, 1), stored), Weight("post_quant_bias", "post_quant_conv.bias", [k], .f32),
                      Weight("conv_in", "decoder.conv_in.weight", kernel(c, k, 3), stored)] + batchNorm
            } else {
                w += [Weight("conv_in", "decoder.conv_in.weight", kernel(c, latentChannels, 3), stored)]
            }
            return w
        case .conv1:
            return [Weight("gamma", "\(p).norm1.weight", [cin], .f32), Weight("beta", "\(p).norm1.bias", [cin], .f32),
                    Weight("kernel", "\(p).conv1.weight", kernel(cout, cin, 3), stored), Weight("bias", "\(p).conv1.bias", [cout], .f32)]
        case .conv2:
            return [Weight("gamma", "\(p).norm2.weight", [cout], .f32), Weight("beta", "\(p).norm2.bias", [cout], .f32),
                    Weight("kernel", "\(p).conv2.weight", kernel(cout, cout, 3), stored), Weight("bias", "\(p).conv2.bias", [cout], .f32)]
                + (cin == cout ? [] : [Weight("shortcut", "\(p).conv_shortcut.weight", kernel(cout, cin, 1), stored), Weight("shortcut_bias", "\(p).conv_shortcut.bias", [cout], .f32)])
        case .up:
            return [Weight("kernel", "\(p).upsamplers.0.conv.weight", kernel(cout, cin, 3), stored), Weight("bias", "\(p).upsamplers.0.conv.bias", [cout], .f32)]
        case .down:
            return [Weight("kernel", "\(p).weight", kernel(cout, cin, 3), stored), Weight("bias", "\(p).bias", [cout], .f32)]
        case .out:
            return [Weight("gamma", "decoder.conv_norm_out.weight", [cin], .f32), Weight("beta", "decoder.conv_norm_out.bias", [cin], .f32),
                    Weight("kernel", "decoder.conv_out.weight", kernel(3, cin, 3), stored), Weight("bias", "decoder.conv_out.bias", [3], .f32)]
        case .first:
            return [Weight("kernel", "encoder.conv_in.weight", kernel(cout, 3, 3), stored), Weight("bias", "encoder.conv_in.bias", [cout], .f32)]
        case .tail:
            let c = cin, moments = 2 * channels
            var w = midWeights("encoder", c) + [Weight("norm_gamma", "encoder.conv_norm_out.weight", [c], .f32), Weight("norm_beta", "encoder.conv_norm_out.bias", [c], .f32),
                                                 Weight("conv_out", "encoder.conv_out.weight", kernel(moments, c, 3), stored), Weight("conv_out_bias", "encoder.conv_out.bias", [moments], .f32)]
            if case .patched = latent {
                w += [Weight("quant", "quant_conv.weight", kernel(moments, moments, 1), stored), Weight("quant_bias", "quant_conv.bias", [moments], .f32)] + batchNorm
            }
            return w
        }
    }

    /// **The recipe of a weight the arena holds** (`Weights`), by the rank the file gives it: a convolution's kernel in
    /// the plane's element (f16 rounded once under the guard; f32 as it is, or bf16 widened), an attention's projection
    /// `[out, in]` as `[in, out]`, a bias or a gain in f32.
    func recipe(_ key: String, in file: SafeTensors) -> Recipe {
        switch file.tensors[key]?.shape.count {
        case 4: return half ? .half : .float
        case 2: return half ? .halfTransposed : .floatTransposed
        default: return .float
        }
    }

    // MARK: - The functions

    /// **The module of a layer of `kind` at the plan's shapes `g`**: its windows and outputs as `Bands` names them, its
    /// weights as `weights` lists them.
    func module(_ kind: Kind, cin: Int, cout: Int, _ g: Bands.Geometry) throws -> Aicode.Module {
        let weights = weights(kind, layer: "", cin: cin, cout: cout)
        var m = Aicode.Module(arguments: Bands.arguments(g) + weights.map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        let b = m.body
        var joined: [String: Int] = [:]
        // In the names' order: a dictionary's order is the process's, and a module built in it was another asset at every
        // launch (19/11: 157 assets of one decoder, its readings never found again).
        for (name, w) in g.windows.sorted(by: { $0.key < $1.key }) {
            let (v, f) = Bands.join(m, name, w)
            joined[name] = v
            if let f { joined["\(name)_factor"] = f }
        }
        // Each kernel back at its rank: `[out, in, k, k]`.
        var reshaped: [String: Int] = [:]
        for w in weights where w.kernel { reshaped[w.name] = b.reshape(m.argument(w.name), w.full) }
        func a(_ name: String) -> Int { joined[name] ?? reshaped[name] ?? m.argument(name) }
        func shape(_ v: Int) -> [Int] { Aicode.shape(b.types[v]).shape }
        let groups = Self.groups, half = self.half

        /// A row vector (`[rows]`) laid along a band's rows: `[1, 1, rows, 1]`.
        func alongRows(_ v: Int) -> Int { b.reshape(v, [1, 1, shape(v)[0], 1]) }
        /// A channel vector laid along a band's channels: `[1, C, 1, 1]`.
        func alongChannels(_ v: Int) -> Int { b.reshape(v, [1, shape(v)[0], 1, 1]) }
        /// A window's true value, f32: a stored plane's under its rows' factors, the encoder's picture as it is.
        func read(_ t: String) -> Int {
            g.windows[t]?.factors == true ? b.binary(.mul, b.cast(a(t), to: .f32), alongRows(a("\(t)_factor"))) : b.castUnlessSame(a(t), to: .f32)
        }
        /// A plane's statistics carried (`[3, G]`) as its mean and rstd, each `[G, 1]`.
        func statistics(_ carry: Int) -> (mean: Int, rstd: Int) {
            func row(_ i: Int) -> Int { b.slice(carry, axis: 0, start: i, size: 1) }
            let rstd = b.primitive("rsqrt", b.scale("add", b.binary(.divide, row(2), row(0)), Self.eps))
            return (b.reshape(row(1), [groups, 1]), b.reshape(rstd, [groups, 1]))
        }
        /// GroupNorm, SiLU and the mask on a stored window (`[1, C, rows, W]`): the affine folded, under the guard with
        /// the rows' factors in f32 (`[1, C, rows, 1]`, `[1, C, 1, 1]`), the band normalized in f16; in f32 as it is.
        func activated(_ t: String, _ gamma: String, _ beta: String) -> Int {
            let c = shape(a(gamma))[0], (mean, rstd) = statistics(a("\(t)_stats")), mask = a("\(t)_mask")
            let gain = b.binary(.mul, b.reshape(a(gamma), [groups, c / groups]), rstd)
            let offset = b.binary(.sub, b.reshape(a(beta), [groups, c / groups]), b.binary(.mul, mean, gain))
            guard half else {
                let n = b.binary(.add, b.binary(.mul, a(t), b.reshape(gain, [1, c, 1, 1])), b.reshape(offset, [1, c, 1, 1]))
                return b.binary(.mul, b.silu(n), alongRows(mask))
            }
            let scale = b.cast(b.binary(.mul, b.reshape(gain, [1, c, 1, 1]), alongRows(a("\(t)_factor"))), to: .f16)
            let n = b.binary(.add, b.binary(.mul, a(t), scale), b.cast(b.reshape(offset, [1, c, 1, 1]), to: .f16))
            return b.binary(.mul, b.silu(n), b.cast(alongRows(mask), to: .f16))
        }
        /// A convolution over `u` (f16 `[1, C, rows, W]` under the window's factor `f`), its rows padded by `rowPad`, its
        /// columns by `colPad`: its true result, f32 `[1, O, rows′, W′]`, with its bias.
        func convolved(_ u: Int, _ f: Int, _ kernel: Int, _ bias: Int, rowPad: Int, colPad: Int) -> Int {
            let s = shape(u), k = shape(kernel)
            let (rows, cols) = (s[2] + 2 * rowPad - (k[2] - 1), s[3] + 2 * colPad - (k[3] - 1))
            let y = b.convolution(u, kernel, stride: [1, 1], padding: [rowPad, colPad], dilation: [1, 1], groups: 1,
                                  result: Aicode.tensor([1, k[0], rows, cols], .f16))
            return b.binary(.add, b.binary(.mul, b.cast(y, to: .f32), f), alongChannels(bias))
        }
        /// The same on `x` (`[1, C, rows, W]`): under the guard brought under the kernel's target by one factor; in f32
        /// as it is. `stride` 2 reads x as it lies (the ↓'s padding is its window's).
        func conv(_ x: Int, _ kernel: Int, _ bias: Int, rowPad: Int = 0, colPad: Int = 1, stride: Int = 1) -> Int {
            guard half || stride != 1 else {
                let s = shape(x), k = shape(kernel)
                let (rows, cols) = (s[2] + 2 * rowPad - (k[2] - 1), s[3] + 2 * colPad - (k[3] - 1))
                let y = b.convolution(x, kernel, stride: [1, 1], padding: [rowPad, colPad], dilation: [1, 1], groups: 1,
                                      result: Aicode.tensor([1, k[0], rows, cols], .f32))
                return b.binary(.add, y, alongChannels(bias))
            }
            if stride == 1 {
                let (u, f) = b.ranged(x, over: [0, 1, 2, 3], under: b.target(kernels: [kernel]))
                return convolved(u, f, kernel, bias, rowPad: rowPad, colPad: colPad)
            }
            // The ↓: x padded with one zero column on its right, convolved without padding at stride 2.
            let (u, f) = half ? b.ranged(x, over: [0, 1, 2, 3], under: b.target(kernels: [kernel])) : (x, -1)
            let s = shape(u), k = shape(kernel), e: Aicode.Element = half ? .f16 : .f32
            let padded = b.concat([u, b.scale("mul", b.slice(u, axis: 3, start: 0, size: 1), 0)], axis: 3)
            let y = b.convolution(padded, kernel, stride: [2, 2], padding: [0, 0], dilation: [1, 1], groups: 1,
                                  result: Aicode.tensor([1, k[0], (s[2] - 3) / 2 + 1, (s[3] + 1 - 3) / 2 + 1], e))
            return half ? b.binary(.add, b.binary(.mul, b.cast(y, to: .f32), f), alongChannels(bias)) : b.binary(.add, y, alongChannels(bias))
        }
        /// **The statistics `y` carries on** (f32 `[1, C, rows, W]`, its rows inside the plane `valid`): the band's own
        /// count, mean and sum of squared deviations by group, merged with those `carried` (`[3, G]`) by Chan's rule.
        func carried(_ y: Int, valid: Int?, _ carried: Int) -> Int {
            let s = shape(y), grouped = b.reshape(y, [groups, s[1] / groups, s[2], s[3]]), per = Double(s[1] / groups * s[3])
            let mask = valid.map { b.reshape($0, [1, 1, s[2], 1]) }
            let count = valid.map { b.scale("mul", b.reshape(b.sum($0, axes: [0]), [1, 1]), per) }
                ?? m.constant("count", withUnsafeBytes(of: Float(per * Double(s[2]))) { Array($0) }, Aicode.tensor([1, 1], .f32))
            let inside = mask.map { b.binary(.mul, grouped, $0) } ?? grouped
            let mean = b.binary(.divide, b.sum(inside, axes: [1, 2, 3]), b.reshape(count, [1, 1, 1, 1]))
            let d = mask.map { b.binary(.mul, b.binary(.sub, grouped, mean), $0) } ?? b.binary(.sub, grouped, mean)
            let (m, m2) = (b.reshape(mean, [1, groups]), b.reshape(b.sum(b.binary(.mul, d, d), axes: [1, 2, 3]), [1, groups]))
            func row(_ i: Int) -> Int { b.slice(carried, axis: 0, start: i, size: 1) }
            let (n0, m0, s0) = (row(0), row(1), row(2))
            let n = b.binary(.add, n0, count), delta = b.binary(.sub, m, m0), weight = b.binary(.divide, count, n)
            let merged = b.binary(.add, m0, b.binary(.mul, delta, weight))
            let spread = b.binary(.add, b.binary(.add, s0, m2), b.binary(.mul, b.binary(.mul, delta, delta), b.binary(.mul, n0, weight)))
            return b.concat([n, merged, spread], axis: 0)
        }
        /// **`y` written as the output `name`** (f32 `[1, C, rows, W]`): its rows past the plane zeros; under the guard f16
        /// under its rows' factors, their largest in [2¹³, 2¹⁴]; in f32 as it is; its statistics carried on.
        func written(_ name: String, _ y: Int) -> [(String, Int)] {
            let o = g.outputs[name]!, valid = o.masked ? a("\(name)_valid") : nil
            let v = valid.map { b.binary(.mul, y, alongRows($0)) } ?? y
            var out: [(String, Int)]
            if half && o.factors {
                let (u, f) = b.ranged(v, over: [1, 3], under: b.stream)
                out = Bands.split(b, name, value: u, factor: b.reshape(f, [shape(v)[2]]), o)
            } else {
                out = Bands.split(b, name, value: v, factor: nil, o)
            }
            if !o.carry.isEmpty { out.append(("\(name)_carry", carried(v, valid: valid, a("\(name)_carried")))) }
            return out
        }
        /// The ×2's window, doubled: [C, r+2, 1, W, 1] + 0 → [1, C, 2r+4, 2W] (exact in either element).
        func doubled(_ u: Int, _ c: Int, _ r: Int, _ w: Int) -> Int {
            let e = Aicode.shape(b.types[u]).element
            let zeros = m.constant("zeros", [UInt8](repeating: 0, count: e.bytes(4)), Aicode.tensor([1, 1, 2, 1, 2], e))
            return b.reshape(b.binary(.add, b.reshape(u, [c, r + 2, 1, w, 1]), zeros), [1, c, 2 * r + 4, 2 * w])
        }
        /// A GroupNorm over a whole plane `[1, C, h, w]` (f32), its affine in f32.
        func groupNorm(_ x: Int, _ c: Int, _ h: Int, _ w: Int, _ gamma: String, _ beta: String) -> Int {
            let n = b.reshape(b.layerNorm(b.reshape(x, [groups, c / groups * h * w]), axes: [1], epsilon: b.scalar(Self.eps, .f32)), [1, c, h, w])
            return b.binary(.add, b.binary(.mul, n, alongChannels(a(gamma))), alongChannels(a(beta)))
        }
        /// The mid block, whole, on `x` (f32 `[1, C, h, w]`): resnet, attention, resnet; the GroupNorms' statistics over
        /// the plane.
        func midBlock(_ x: Int, _ c: Int, _ h: Int, _ w: Int) -> Int {
            func resnet(_ x: Int, _ p: String) -> Int {
                let h1 = conv(b.silu(groupNorm(x, c, h, w, "\(p)_gamma1", "\(p)_beta1")), a("\(p)_kernel1"), a("\(p)_bias1"), rowPad: 1)
                return b.binary(.add, x, conv(b.silu(groupNorm(h1, c, h, w, "\(p)_gamma2", "\(p)_beta2")), a("\(p)_kernel2"), a("\(p)_bias2"), rowPad: 1))
            }
            /// One head over every position; under the guard q, k, v written true in f16; the residual in f32.
            func attention(_ x: Int) -> Int {
                let n = h * w
                let seq = b.transpose(b.reshape(groupNorm(x, c, h, w, "attn_gamma", "attn_beta"), [c, n]), [1, 0])
                let out: Int
                if half {
                    let (unit, factor) = b.ranged(seq, under: b.target(weights: [a("attn_q"), a("attn_k"), a("attn_v")]))
                    func projected(_ p: String) -> Int {
                        b.reshape(b.cast(b.multipliedOut(b.product(unit, a("attn_\(p)")), factor, bias: a("attn_\(p)_bias")), to: .f16), [1, 1, n, c])
                    }
                    let o = b.reshape(b.attention([projected("q"), projected("k"), projected("v")], scale: 1 / Double(c).squareRoot(), symbol: "sdpa_vae"), [n, c])
                    let (ou, of) = b.ranged(o, under: b.target(weights: [a("attn_o")]))
                    out = b.multipliedOut(b.product(ou, a("attn_o")), of, bias: a("attn_o_bias"))
                } else {
                    func linear(_ x: Int, _ p: String) -> Int {
                        b.binary(.add, b.matmul(x, a("attn_\(p)"), result: Aicode.tensor([n, c], .f32)), a("attn_\(p)_bias"))
                    }
                    let o = b.reshape(b.attention(["q", "k", "v"].map { b.reshape(linear(seq, $0), [1, 1, n, c]) }, scale: 1 / Double(c).squareRoot(), symbol: "sdpa_vae"), [n, c])
                    out = linear(o, "o")
                }
                return b.binary(.add, x, b.reshape(b.transpose(out, [1, 0]), [1, c, h, w]))
            }
            return resnet(attention(resnet(x, "r0")), "r1")
        }
        /// `[c, 2, 2, h, w]`-cells ↔ planes: the 2×2 of each position of `[4c, h, w]` to `[c, 2h, 2w]`, and back.
        func unpatchified(_ z: Int, _ c: Int, _ h: Int, _ w: Int) -> Int {
            b.reshape(b.transpose(b.reshape(z, [c, 2, 2, h, w]), [0, 3, 1, 4, 2]), [1, c, 2 * h, 2 * w])
        }
        func patchified(_ y: Int, _ c: Int, _ h: Int, _ w: Int) -> Int {
            b.reshape(b.transpose(b.reshape(y, [c, h, 2, w, 2]), [0, 2, 4, 1, 3]), [4 * c, h, w])
        }
        /// The batch-norm statistics' σ, `[1, 4c, 1, 1]`.
        func sigma(_ count: Int, _ eps: Double) -> Int { b.reshape(b.primitive("sqrt", b.scale("add", a("bn_var"), eps)), [1, count, 1, 1]) }

        switch kind {
        case .head:
            let c = trunk, w0 = g.windows["latent"]!, (h, w) = (w0.rows, w0.width)
            // Whole: `[1, C, h, w]`, the plane padded on both axes; the GroupNorm's statistics over the plane.
            var x: Int
            switch latent {
            case .scaled(_, let scaling, let shift):
                let z = b.scale("add", b.scale("divide", a("latent"), scaling), shift)
                x = midBlock(conv(z, a("conv_in"), a("conv_in_bias"), rowPad: 1), c, h, w)
            case .patched(let k, let eps):
                let n = latentChannels
                let z = b.binary(.add, b.binary(.mul, a("latent"), sigma(n, eps)), b.reshape(a("bn_mean"), [1, n, 1, 1]))
                let planes = conv(unpatchified(z, k, h, w), a("post_quant"), a("post_quant_bias"), colPad: 0)
                x = midBlock(conv(planes, a("conv_in"), a("conv_in_bias"), rowPad: 1), c, 2 * h, 2 * w)
            }
            m.outputs = written("y", x)
        case .conv1:
            m.outputs = written("y", conv(activated("x", "gamma", "beta"), a("kernel"), a("bias")))
        case .conv2:
            // The shortcut's 1×1 reads x̂ · F brought under one factor, as every convolution reads its window: the rows'
            // factors differ.
            let x = read("x")
            let shortcut = cin == cout ? x : conv(x, a("shortcut"), a("shortcut_bias"), colPad: 0)
            m.outputs = written("y", b.binary(.add, shortcut, conv(activated("h", "gamma", "beta"), a("kernel"), a("bias"))))
        case .up:
            let window = g.windows["x"]!, (r, w) = (window.rows, window.width)
            // The window brought under the kernel's target, then doubled (exact): 2r + 4 rows, the convolution's 2r + 2,
            // the first and the last dropped — those of the band are the doubled 2s − 1 ..< 2s + 2r + 1.
            let x = b.binary(.mul, read("x"), alongRows(a("x_mask")))
            let y: Int
            if half {
                let (u, f) = b.ranged(x, over: [0, 1, 2, 3], under: b.target(kernels: [a("kernel")]))
                y = b.slice(convolved(doubled(u, cin, r, w), f, a("kernel"), a("bias"), rowPad: 0, colPad: 1), axis: 2, start: 1, size: 2 * r)
            } else {
                y = b.slice(conv(doubled(x, cin, r, w), a("kernel"), a("bias")), axis: 2, start: 1, size: 2 * r)
            }
            m.outputs = written("y", y)
        case .out:
            let y = conv(activated("x", "gamma", "beta"), a("kernel"), a("bias"))
            m.outputs = written("y", clamped ? b.scale("minimum", b.scale("maximum", y, -1), 1) : y)
        case .first:
            // The picture's rows past its edges, its zero padding: its mask's 0.
            m.outputs = written("y", conv(b.binary(.mul, read("x"), alongRows(a("x_mask"))), a("kernel"), a("bias")))
        case .down:
            // The window's last row, past the plane for the last band, multiplied out: the ↓'s zero padding.
            m.outputs = written("y", conv(b.binary(.mul, read("x"), alongRows(a("x_mask"))), a("kernel"), a("bias"), stride: 2))
        case .tail:
            let window = g.windows["x"]!, (c, h, w) = (cin, window.rows, window.width)
            var x = midBlock(read("x"), c, h, w)
            x = conv(b.silu(groupNorm(x, c, h, w, "norm_gamma", "norm_beta")), a("conv_out"), a("conv_out_bias"), rowPad: 1)
            switch latent {
            case .scaled(let k, _, _):
                m.outputs = written("y", b.slice(x, axis: 1, start: 0, size: k))
            case .patched(let k, let eps):
                let q = conv(x, a("quant"), a("quant_bias"), colPad: 0)
                let p = patchified(b.reshape(b.slice(q, axis: 1, start: 0, size: k), [k, h, w]), k, h / 2, w / 2)
                let n = latentChannels
                let normalized = b.binary(.divide, b.binary(.sub, b.reshape(p, [1, n, h / 2, w / 2]), b.reshape(a("bn_mean"), [1, n, 1, 1])), sigma(n, eps))
                m.outputs = written("y", normalized)
            }
        }
        return m
    }
}
