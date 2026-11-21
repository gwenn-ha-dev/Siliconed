import Foundation
import SiliCore

/// **The Wan lineage's VAE as functions** (Qwen-Image's, which Anima and Krea 2 decode with, and the wider one of
/// Qwen-Image-2.1, a 16× of the same family): the stretches where a position reads every other whole (the decoder's
/// trunk, the encoder's tail), every other layer by bands of rows (`Bands`), each band a call of a function of fixed
/// shape — one function serves every band of its layer, and every layer of its shape. Every weight is an argument, a
/// resident of the arena, a convolution's kernel flat (`Bands.flat`) and reshaped inside (the GPU runs it at the
/// constant's rate, the same bits: measured 19/11, L2). The plan and the run are `QwenImageVAELoop.swift`; the networks,
/// `QwenImageVAEDecoder`, `QwenImageVAEEncoder` and Qwen-Image-2.1's.
///
///     head    latent · its convolutions and the mid block (resnet, attention, resnet), whole            x, stored
///     tail    stored x · the last resnets, the mid block, RMS · SiLU · conv_out · quant_conv, whole       the mean, f32
///     first   a 3×3 on a stored window, no norm (the encoder's conv_in on the picture)                   by bands
///     conv1   RMS · SiLU · conv1                                                                          by bands
///     conv2   RMS · SiLU · conv2 + shortcut(x)                                                            by bands
///     up      nearest ×2 · 3×3 (+ DupUp3D of the block's input: Qwen-Image-2.1), 2r rows from r          by bands
///     down    zero column right · 3×3 at stride 2 (+ AvgDown3D of the block's input), by output rows      by bands
///     out     RMS · SiLU · conv_out · clamp to [−1, 1]                                                    pixels, f32
///
/// **One frame through a causal 3-D VAE is a 2-D one** (`autoencoder_kl_qwenimage.py`): a causal 3×3×3 kernel padded
/// two frames before sees the frame only in its last time slice, so a kernel stored `[O, I, T, kh, kw]` is read at
/// `[:, :, T − 1]` (`Recipe.halfLast`); `time_conv` is never read on one frame. Qwen-Image-2.1 publishes its kernels 2-D.
///
/// **`QwenImageRMS_norm` is `F.normalize` over the channels**, × √C × γ: no ε in a variance, no statistics across
/// positions — so a band needs nothing of another band, unlike Flux's GroupNorm. Its value is the same under a row's
/// factor (x̂ · F / ‖x̂ · F‖), so a band normalizes the stored f16 directly, in f32, the `max(‖x‖, 10⁻¹²)` taken on the
/// true norm.
///
/// **The guard of f16** (`Guard.swift`), as Flux's decoder: a stored tensor is f16 under a row's factor (its rows'
/// largest in [2¹³, 2¹⁴], `stream`); a convolution reads its window brought under 2^e₀ by one factor, e₀ from its own
/// kernel, and multiplies its result out in f32 with its bias; a sum of two terms under different factors (a shortcut,
/// a block's DupUp3D or AvgDown3D) is taken on their true values, in f32; the attention reads q, k, v written true in
/// f16 (a normalized row times weights). **A window's factor is bounded by its plane**, never its own rows: each layer
/// carries its plane's largest row factor from band to band (`x_top`), so that a band's bits do not depend on how tall
/// it is (the `vae-bands` stage; measured 18/11: 2 and 3 values of 10⁸ in anima's last block otherwise, bands of 2¹⁴
/// against 2¹²).
///
/// The window's rows past the plane are multiplied by its mask's 0 as they are joined: zeros, which the norm keeps
/// zeros and SiLU(0) is 0 — the convolution's zero padding.
package enum QwenImageVAE {
    /// A whole function's steps over `[1, C, h, w]`, in order, each named by its module's prefix in the file.
    enum Step: Hashable, Sendable {
        /// A convolution `k × k`, its padding k / 2.
        case conv(String, cin: Int, cout: Int, k: Int)
        /// y = shortcut(x) + conv₂(SiLU(RMS₂(conv₁(SiLU(RMS₁(x)))))), the shortcut a 1×1 where the channels change.
        case resnet(String, cin: Int, cout: Int)
        /// `QwenImageAttentionBlock`: one head over every position, q, k, v the chunks of one 1×1 `to_qkv`; x + proj(·).
        case attention(String, c: Int)
        /// RMS · γ, then SiLU (`norm_out`), the norm's gain by its key.
        case activate(String, c: Int)
        /// The value kept for `addKept` (Qwen-Image-2.1's last encoder block adds its input).
        case keep
        case addKept
        /// The first `n` channels (the latent's mean of `quant_conv`'s mean and log-variance).
        case take(Int)
    }

    /// The input channels a DupUp3D shortcut reads, and its temporal factor (Qwen-Image-2.1's upsampling blocks).
    struct Dup: Hashable, Sendable { let input, temporal: Int }

    /// AvgDown3D on one frame (Qwen-Image-2.1's downsampling blocks): each channel's 2×2 average, or, where the block
    /// doubles its channels, a zero channel before each average.
    enum Avg: Hashable, Sendable { case pool, doubling }

    /// A layer's kind and its shape: what its function computes, whatever its layer (its weights are arguments).
    enum Kind: Hashable, Sendable {
        case head(steps: [Step])
        case tail(steps: [Step])
        case first(cin: Int, cout: Int)
        case conv1(cin: Int, cout: Int)
        case conv2(cin: Int, cout: Int)
        case up(cin: Int, cout: Int, dup: Dup?)
        case down(c: Int, avg: Avg?)
        case out(cin: Int, planes: Int)
    }

    /// What a stored plane carries from band to band: its largest row factor (`x_top`).
    static let top = [1]

    /// An argument's name for a file key: its dots as underscores.
    static func argument(_ key: String) -> String { key.replacingOccurrences(of: ".", with: "_") }

    /// A weight a function reads: its argument, the file's key, its shape (a kernel's `[out, in, k, k]`, given flat:
    /// `Bands.flat`; a 1×1 the attention multiplies by, the product's `[in, out]`), its element.
    struct Weight {
        let name, key: String
        let full: [Int]
        let element: Aicode.Element
        var kernel: Bool { full.count == 4 }
        var shape: [Int] { kernel ? Bands.flat(full, element) : full }
        init(_ name: String, _ key: String, _ full: [Int], _ element: Aicode.Element = .f32) { (self.name, self.key, self.full, self.element) = (name, key, full, element) }
    }

    // MARK: - The weights

    /// The weights a whole function's steps read, each named by its key.
    static func weights(_ steps: [Step]) -> [Weight] {
        func v(_ key: String, _ n: Int) -> Weight { Weight(argument(key), key, [n]) }
        func k(_ key: String, _ o: Int, _ i: Int, _ size: Int) -> Weight { Weight(argument(key), key, [o, i, size, size], .f16) }
        var seen = Set<String>()
        return steps.flatMap { step -> [Weight] in
            switch step {
            case .conv(let p, let cin, let cout, let size): [k("\(p).weight", cout, cin, size), v("\(p).bias", cout)]
            case .resnet(let p, let cin, let cout):
                [v("\(p).norm1.gamma", cin), k("\(p).conv1.weight", cout, cin, 3), v("\(p).conv1.bias", cout),
                 v("\(p).norm2.gamma", cout), k("\(p).conv2.weight", cout, cout, 3), v("\(p).conv2.bias", cout)]
                    + (cin == cout ? [] : [k("\(p).conv_shortcut.weight", cout, cin, 1), v("\(p).conv_shortcut.bias", cout)])
            case .attention(let p, let c):
                [v("\(p).norm.gamma", c), Weight(argument("\(p).to_qkv.weight"), "\(p).to_qkv.weight", [c, 3 * c], .f16), v("\(p).to_qkv.bias", 3 * c),
                 Weight(argument("\(p).proj.weight"), "\(p).proj.weight", [c, c], .f16), v("\(p).proj.bias", c)]
            case .activate(let key, let c): [v(key, c)]
            case .keep, .addKept, .take: []
            }
        }.filter { seen.insert($0.name).inserted }
    }

    /// **The weights a layer of `kind` reads**, by argument: the file's key under `layer`.
    static func weights(_ kind: Kind, layer p: String) -> [Weight] {
        func k(_ key: String, _ o: Int, _ i: Int, _ size: Int = 3) -> Weight { Weight("kernel", key, [o, i, size, size], .f16) }
        switch kind {
        case .head(let steps), .tail(let steps): return weights(steps)
        case .first(let cin, let cout): return [k("\(p).weight", cout, cin), Weight("bias", "\(p).bias", [cout])]
        case .conv1(let cin, let cout): return [Weight("gamma", "\(p).norm1.gamma", [cin]), k("\(p).conv1.weight", cout, cin), Weight("bias", "\(p).conv1.bias", [cout])]
        case .conv2(let cin, let cout):
            return [Weight("gamma", "\(p).norm2.gamma", [cout]), k("\(p).conv2.weight", cout, cout), Weight("bias", "\(p).conv2.bias", [cout])]
                + (cin == cout ? [] : [Weight("shortcut", "\(p).conv_shortcut.weight", [cout, cin, 1, 1], .f16), Weight("shortcut_bias", "\(p).conv_shortcut.bias", [cout])])
        case .up(let cin, let cout, _): return [k("\(p).weight", cout, cin), Weight("bias", "\(p).bias", [cout])]
        case .down(let c, _): return [k("\(p).weight", c, c), Weight("bias", "\(p).bias", [c])]
        case .out(let cin, let planes): return [Weight("gamma", "\(p).norm_out.gamma", [cin]), k("\(p).conv_out.weight", planes, cin), Weight("bias", "\(p).conv_out.bias", [planes])]
        }
    }

    /// **The recipe of a weight the arena holds** (`Weights`): a causal kernel at its last time slice, a 2-D kernel as
    /// it is, each rounded once to f16; the attention's 1×1s as the product's `[in, out]`; a bias or a gain in f32 (a
    /// gain is `[C, 1, 1, 1]` in the file: its rank says nothing).
    static func recipe(_ key: String, in file: SafeTensors) -> Recipe {
        guard key.hasSuffix(".weight") else { return .float }
        if key.hasSuffix(".to_qkv.weight") || key.hasSuffix(".proj.weight") { return .halfTransposed }
        switch file.tensors[key]?.shape.count {
        case 5: return .halfLast(axis: 2)
        case 4: return .half
        default: return .float
        }
    }

    /// **The function of a layer of `kind`** in `network`: its key, its module at the plan's shapes.
    static func function(_ network: String, _ kind: Kind) -> Bands.Function {
        Bands.Function("\(network) \(kind)") { g in try module(kind, g) }
    }

    // MARK: - The functions

    /// **The module of a layer of `kind` at the plan's shapes `g`**: its windows and outputs as `Bands` names them, its
    /// weights as `weights` lists them.
    static func module(_ kind: Kind, _ g: Bands.Geometry) throws -> Aicode.Module {
        let weights = weights(kind, layer: "")
        var m = Aicode.Module(arguments: Bands.arguments(g) + weights.map { ($0.name, Aicode.tensor($0.shape, $0.element)) })
        let b = m.body
        func shape(_ v: Int) -> [Int] { Aicode.shape(b.types[v]).shape }
        /// A row vector (`[rows]`) laid along a band's rows: `[1, 1, rows, 1]`.
        func alongRows(_ v: Int) -> Int { b.reshape(v, [1, 1, shape(v)[0], 1]) }
        // Each window joined, times its mask (exact: its rows past the plane zeros, the others as they are).
        var joined: [String: Int] = [:]
        // In the names' order: a dictionary's order is the process's, and a module built in it was another asset at every
        // launch (19/11: 157 assets of one decoder, its readings never found again).
        for (name, w) in g.windows.sorted(by: { $0.key < $1.key }) {
            let (v, f) = Bands.join(m, name, w)
            joined[name] = w.before || w.after ? b.binary(.mul, v, b.cast(alongRows(m.argument("\(name)_mask")), to: Aicode.shape(b.types[v]).element)) : v
            if let f { joined["\(name)_factor"] = f }
        }
        // Each kernel back at its rank: `[out, in, k, k]`.
        for w in weights where w.kernel { joined[w.name] = b.reshape(m.argument(w.name), w.full) }
        func a(_ name: String) -> Int { joined[name] ?? m.argument(name) }
        // The zeros a nearest ×2 adds to broadcast (`doubled`, `dupUp`), small enough to lie in the IR as splats.
        func zeros(_ shape: [Int], _ e: Aicode.Element) -> Int { m.constant("zeros", [UInt8](repeating: 0, count: e.bytes(shape.reduce(1, *))), Aicode.tensor(shape, e)) }

        /// A channel vector laid along a tensor's channels: `[1, C, 1, 1]`.
        func alongChannels(_ v: Int) -> Int { b.reshape(v, [1, shape(v)[0], 1, 1]) }
        /// A stored band's true value, f32.
        func read(_ t: String) -> Int { b.binary(.mul, b.cast(a(t), to: .f32), alongRows(a("\(t)_factor"))) }
        /// `QwenImageRMS_norm` · γ, then SiLU, of `x` (f32 `[1, C, rows, W]`, or a stored band's f16 under `factor`):
        /// x / max(‖x‖, 10⁻¹²) · √C · γ over the channels of each position, in f32, then f16 — bounded by √C · max|γ|.
        func activated(_ x: Int, factor: Int?, gamma: Int) -> Int {
            let wide = b.castUnlessSame(x, to: .f32), c = shape(wide)[1]
            var norm = b.primitive("sqrt", b.sum(b.binary(.mul, wide, wide), axes: [1]))
            var unit = wide
            if let factor {
                // The true norm F · ‖x̂‖ against the floor; F · x̂ over it: the same value as the stored band's own.
                let f = alongRows(factor)
                norm = b.binary(.mul, norm, f)
                unit = b.binary(.mul, unit, f)
            }
            let n = b.binary(.mul, b.binary(.divide, unit, b.scale("maximum", norm, 1e-12)), alongChannels(gamma))
            return b.silu(b.cast(b.scale("mul", n, Double(c).squareRoot()), to: .f16))
        }
        /// A convolution over `u` (f16 `[1, C, rows, W]` under the window's factor `f`): its true result, f32, with its bias.
        func convolved(_ u: Int, _ f: Int, _ kernel: Int, _ bias: Int, rowPad: Int, colPad: Int, stride: Int = 1) -> Int {
            let s = shape(u), k = shape(kernel)
            let rows = (s[2] + 2 * rowPad - k[2]) / stride + 1, cols = (s[3] + 2 * colPad - k[3]) / stride + 1
            let y = b.convolution(u, kernel, stride: [stride, stride], padding: [rowPad, colPad], dilation: [1, 1], groups: 1,
                                  result: Aicode.tensor([1, k[0], rows, cols], .f16))
            return b.binary(.add, b.binary(.mul, b.cast(y, to: .f32), f), alongChannels(bias))
        }
        /// `t` brought under `e0` by the power of two a bound `top` of it takes (`ranged`'s rule, its top given).
        func scaled(_ t: Int, top: Int, under e0: Int) -> (unit: Int, factor: Int) {
            let element = Aicode.shape(b.types[t]).element, one = [1, 1, 1, 1]
            let log2 = b.scale("mul", b.primitive("log", b.scale("maximum", b.reshape(b.castUnlessSame(top, to: .f32), one), 0x1p-14)), 1 / Foundation.log(2.0))
            var e = b.binary(.sub, b.primitive("round", b.scale("add", log2, 0.5)), b.reshape(e0, one))
            let (low, high) = element == .f32 ? (-120.0, 120.0) : (-13.0, 15.0)
            e = b.scale("minimum", b.scale("maximum", e, low), high)
            func power(_ k: Int) -> Int { b.primitive("round", b.scale("pow", b.scale("maximum", k, 0), 2, scalarFirst: true)) }
            let f = b.binary(.divide, power(e), power(b.scale("mul", e, -1)))
            let unit = element == .f32 ? b.cast(b.binary(.divide, t, f), to: .f16) : b.binary(.divide, t, b.cast(f, to: .f16))
            return (unit, f)
        }
        /// A stored band's window under the kernel's target, bounded by its plane: 2¹⁴ · the plane's largest row factor
        /// (`x_top`, its writer's carry).
        func window(_ t: String, under e0: Int) -> (unit: Int, factor: Int) {
            scaled(read(t), top: b.scale("mul", a("x_top"), 0x1p14), under: e0)
        }
        /// An activated band under the kernel's target, bounded by its norm: √C · max|γ|, or SiLU's least, −0.2785.
        func normed(_ x: Int, factor: Int?, gamma: Int, under e0: Int) -> (unit: Int, factor: Int) {
            let c = Double(shape(gamma)[0])
            let bound = b.scale("maximum", b.scale("mul", b.reduceMax(b.primitive("abs", gamma), axes: [0]), c.squareRoot()), 0.2785)
            return scaled(activated(x, factor: factor, gamma: gamma), top: bound, under: e0)
        }
        /// The same on `x` (f16 or f32), brought under the kernel's target by one factor: a whole plane's (`steps`).
        func conv(_ x: Int, _ kernel: Int, _ bias: Int, rowPad: Int = 0, colPad: Int = 1) -> Int {
            let (u, f) = b.ranged(x, over: [0, 1, 2, 3], under: b.target(kernels: [kernel]))
            return convolved(u, f, kernel, bias, rowPad: rowPad, colPad: colPad)
        }
        /// A band's 3×3 over its activated window.
        func convNormed(_ x: String) -> Int {
            let (u, f) = normed(a(x), factor: a("\(x)_factor"), gamma: a("gamma"), under: b.target(kernels: [a("kernel")]))
            return convolved(u, f, a("kernel"), a("bias"), rowPad: 0, colPad: 1)
        }
        /// A band's 1×1 shortcut over its rows `x`, bounded by its plane.
        func shortcut() -> Int {
            let (u, f) = window("x", under: b.target(kernels: [a("shortcut")]))
            return convolved(u, f, a("shortcut"), a("shortcut_bias"), rowPad: 0, colPad: 0)
        }
        /// **`y` written as the output `name`** (f32 `[1, C, rows, W]`): its rows past the plane zeros; stored f16 under
        /// its rows' factors (`[rows]`), their largest in [2¹³, 2¹⁴], the plane's largest factor carried on; a plane of
        /// f32 as it is.
        func written(_ name: String, _ y: Int) -> [(String, Int)] {
            let o = g.outputs[name]!
            let v = o.masked ? b.binary(.mul, y, alongRows(a("\(name)_valid"))) : y
            guard o.factors else { return Bands.split(b, name, value: v, factor: nil, o) }
            let (u, f) = b.ranged(v, over: [1, 3], under: b.stream)
            let rows = b.reshape(f, [shape(v)[2]])
            let top = b.reduceMax(b.concat([a("\(name)_carried"), b.reduceMax(rows, axes: [0])], axis: 0), axes: [0])
            return Bands.split(b, name, value: u, factor: rows, o) + [("\(name)_carry", top)]
        }
        /// Nearest ×2 of `x` (`[1, C, r, w]`, any element): `[C, r, 1, w, 1]` + zeros `[1, 1, 2, 1, 2]` → `[1, C, 2r, 2w]`.
        func doubled(_ x: Int) -> Int {
            let s = shape(x)
            return b.reshape(b.binary(.add, b.reshape(x, [s[1], s[2], 1, s[3], 1]), zeros([1, 1, 2, 1, 2], Aicode.shape(b.types[x]).element)),
                             [1, s[1], 2 * s[2], 2 * s[3]])
        }

        /// A resnet over the whole plane (f32 `[1, C, h, w]`).
        func resnet(_ x: Int, _ p: String, _ cin: Int, _ cout: Int) -> Int {
            func k(_ s: String) -> Int { a(argument("\(p).\(s)")) }
            let shortcut = cin == cout ? x : conv(x, k("conv_shortcut.weight"), k("conv_shortcut.bias"), colPad: 0)
            let h = conv(activated(x, factor: nil, gamma: k("norm1.gamma")), k("conv1.weight"), k("conv1.bias"), rowPad: 1)
            return b.binary(.add, shortcut, conv(activated(h, factor: nil, gamma: k("norm2.gamma")), k("conv2.weight"), k("conv2.bias"), rowPad: 1))
        }
        /// The mid block's attention over every position: q, k, v written true in f16; the residual in f32.
        func attention(_ x: Int, _ p: String, _ c: Int) -> Int {
            func k(_ s: String) -> Int { a(argument("\(p).\(s)")) }
            let s = shape(x), n = s[2] * s[3]
            let wide = x, norm = b.primitive("sqrt", b.sum(b.binary(.mul, wide, wide), axes: [1]))
            let normed = b.scale("mul", b.binary(.mul, b.binary(.divide, wide, b.scale("maximum", norm, 1e-12)), alongChannels(k("norm.gamma"))), Double(c).squareRoot())
            let seq = b.transpose(b.reshape(normed, [c, n]), [1, 0])
            let qkv = k("to_qkv.weight"), bias = k("to_qkv.bias")
            let (unit, factor) = b.ranged(seq, under: b.target(weights: [qkv]))
            let projected = b.multipliedOut(b.product(unit, qkv), factor, bias: bias)
            func part(_ i: Int) -> Int { b.reshape(b.cast(b.slice(projected, axis: 1, start: i * c, size: c), to: .f16), [1, 1, n, c]) }
            let o = b.reshape(b.attention([part(0), part(1), part(2)], scale: 1 / Double(c).squareRoot(), symbol: "sdpa_vae"), [n, c])
            let (ou, of) = b.ranged(o, under: b.target(weights: [k("proj.weight")]))
            let out = b.multipliedOut(b.product(ou, k("proj.weight")), of, bias: k("proj.bias"))
            return b.binary(.add, x, b.reshape(b.transpose(out, [1, 0]), [1, c, s[2], s[3]]))
        }
        /// A whole function's steps from `x` (f32 `[1, C, h, w]`).
        func steps(_ x0: Int, _ steps: [Step]) -> Int {
            var x = x0, kept = x0
            for step in steps {
                switch step {
                case .conv(let p, _, _, let k): x = conv(x, a(argument("\(p).weight")), a(argument("\(p).bias")), rowPad: k / 2, colPad: k / 2)
                case .resnet(let p, let cin, let cout): x = resnet(x, p, cin, cout)
                case .attention(let p, let c): x = attention(x, p, c)
                case .activate(let key, _): x = b.cast(activated(x, factor: nil, gamma: a(argument(key))), to: .f32)
                case .keep: kept = x
                case .addKept: x = b.binary(.add, x, kept)
                case .take(let n): x = b.slice(x, axis: 1, start: 0, size: n)
                }
            }
            return x
        }

        switch kind {
        case .head(let list):
            m.outputs = written("y", steps(a("latent"), list))
        case .tail(let list):
            m.outputs = written("y", steps(read("x"), list))
        case .first:
            let (u, f) = window("x", under: b.target(kernels: [a("kernel")]))
            m.outputs = written("y", convolved(u, f, a("kernel"), a("bias"), rowPad: 0, colPad: 1))
        case .conv1:
            m.outputs = written("y", convNormed("x"))
        case .conv2(let cin, let c):
            // The shortcut's 1×1 reads x̂ · F brought under one factor, as every convolution reads its window.
            m.outputs = written("y", b.binary(.add, cin == c ? read("x") : shortcut(), convNormed("h")))
        case .up(_, let cout, let dup):
            // The window brought under the kernel's target, then doubled in f16 (exact): 2r + 4 rows, of which the
            // convolution's 2r + 2, the first and the last dropped — those of the band are the doubled 2s − 1 ..< 2s + 2r + 1.
            let r = g.windows["x"]!.rows
            let (u, f) = window("x", under: b.target(kernels: [a("kernel")]))
            var y = b.slice(convolved(doubled(u), f, a("kernel"), a("bias"), rowPad: 0, colPad: 1), axis: 2, start: 1, size: 2 * r)
            if let dup {
                y = b.binary(.add, y, dupUp(b, read("entry"), output: cout, temporal: dup.temporal, pair: zeros([1, 1, 1, 1, 2], .f32), doubled: doubled))
            }
            m.outputs = written("y", y)
        case .down(_, let avg):
            // `ZeroPad2d((0, 1, 0, 1))` then a 3×3 at stride 2: the bottom row past the plane is masked (zeros), the
            // right column a zero one added here (the window's first column times 0: a window holds no infinity).
            let (r, w) = (g.outputs["y"]!.rows, g.windows["x"]!.width)
            let (u, f) = window("x", under: b.target(kernels: [a("kernel")]))
            let padded = b.concat([u, b.scale("mul", b.slice(u, axis: 3, start: 0, size: 1), 0)], axis: 3)
            var y = convolved(padded, f, a("kernel"), a("bias"), rowPad: 0, colPad: 0, stride: 2)
            if let avg {
                let e = read("entry"), ce = shape(e)[1]
                let pooled = b.reshape(b.mean(b.reshape(e, [ce, r, 2, w / 2, 2]), axes: [2, 4]), [1, ce, 1, r, w / 2])
                let shortcut = avg == .pool ? pooled : b.concat([b.scale("mul", pooled, 0), pooled], axis: 2)
                y = b.binary(.add, y, b.reshape(shortcut, [1, avg == .pool ? ce : 2 * ce, r, w / 2]))
            }
            m.outputs = written("y", y)
        case .out:
            m.outputs = written("y", b.scale("minimum", b.scale("maximum", convNormed("x"), -1), 1))
        }
        return m
    }

    /// **`DupUp3D` on one frame** (`first_chunk`; Qwen-Image-2.1's upsampling blocks), on the block's input `e` (f32
    /// `[1, in, r, w]`): output channel `o` at `(2y + i, 2x + j)` reads input channel `⌊(4·ft·o + 4·(ft − 1) + 2i + j) / s⌋`
    /// at `(y, x)`, `s = 4·ft·out / in` — a selection, no arithmetic. `[1, out, 2r, 2w]`.
    static func dupUp(_ b: Aicode.Block, _ e: Int, output: Int, temporal ft: Int, pair zeros: Int, doubled: (Int) -> Int) -> Int {
        let shape = Aicode.shape(b.types[e]).shape, (input, r, w) = (shape[1], shape[2], shape[3]), s = 4 * ft * output / input
        let source = { (o: Int, i: Int) in (4 * ft * o + 4 * (ft - 1) + 2 * i) / s }
        // The channels are the same on both rows: a nearest ×2 of one channel in `step`, from the first read.
        if source(0, 0) == source(0, 1) {
            let step = input / output, first = source(0, 0)
            let picked = step == 1 ? e : b.reshape(b.slice(b.reshape(e, [1, output, step, r, w]), axis: 2, start: first, size: 1), [1, output, r, w])
            return doubled(picked)
        }
        // Two channels a pair, one a row: `[out, 2, r, w]` → `[out, r, 2, w]`, each value twice along the width.
        precondition(input == 2 * output && source(0, 0) == 0 && source(0, 1) == 1, "DupUp3D \(input) → \(output): not a pair of rows")
        let rows = b.reshape(b.transpose(b.reshape(e, [output, 2, r, w]), [0, 2, 1, 3]), [output, r, 2, w, 1])
        return b.reshape(b.binary(.add, rows, zeros), [1, output, 2 * r, 2 * w])
    }
}

extension QwenImageVAE {
    /// The bands' sizes a run is planned among, in positions of a layer's output, the preferred first: Flux's decoder
    /// measured 2¹⁴ the fastest and the lightest (18/11 on the M1 Pro), the smaller ones lighter still for a B that does
    /// not hold it.
    static let positions = [1 << 14, 1 << 13, 1 << 12, 1 << 11]
}
