import CoreAI

/// **Writing a band's function**: its arguments at its shapes, a weight given flat, a window joined, an output split
/// into its bands.
extension Bands {
    /// **A function's arguments at `g`**, the weights aside, in one order (an asset is named by its bytes): each window's
    /// bands, the band before and after, their factors, its mask and its plane's carry; each output's mask and the carry
    /// it is given.
    public static func arguments(_ g: Geometry) -> [(String, Aicode.Entry)] {
        func element(_ t: NDArray.ScalarType) -> Aicode.Element { t == .float16 ? .f16 : .f32 }
        var out: [(String, Aicode.Entry)] = []
        for (name, w) in g.windows.sorted(by: { $0.key < $1.key }) {
            let rho = w.held / w.bands
            let names = (0..<w.bands).map { "\(name)\($0)" } + (w.before ? ["\(name)_before"] : []) + (w.after ? ["\(name)_after"] : [])
            for n in names {
                out.append((n, Aicode.tensor([1, w.channels, rho, w.width], element(w.element))))
                if w.factors { out.append(("\(n)_factor", Aicode.tensor([rho], .f32))) }
            }
            if w.before || w.after { out.append(("\(name)_mask", Aicode.tensor([w.rows + (w.before ? 1 : 0) + (w.after ? 1 : 0)], .f32))) }
            if let c = w.carry { out.append((c, Aicode.tensor(w.carried, .f32))) }
        }
        for (name, o) in g.outputs.sorted(by: { $0.key < $1.key }) {
            if o.masked { out.append(("\(name)_valid", Aicode.tensor([o.rows], .f32))) }
            if !o.carry.isEmpty { out.append(("\(name)_carried", Aicode.tensor(o.carry, .f32))) }
        }
        return out
    }

    /// **The shape a weight of `shape` is given as**: its first axis and the rest flat, `[O, C · kh · kw]` for a
    /// convolution's kernel, when a row of it is a multiple of 64 bytes; else rows of 64 bytes — Core AI's GPU copies an
    /// argument whose rows are not at every call (`Layout`). The function reshapes it back. Measured 19/11 on the mini, f16
    /// (3×3 128 → 128 at 128 × 1024, 3×3 and 1×1 512 → 512 at 64 × 256): a kernel given flat runs at 0.95 – 0.96 × the
    /// same kernel a constant of its asset, the same bits; given `[O, C, 3, 3]`, copied at every call, 0.77 × and 0.32 ×.
    public static func flat(_ shape: [Int], _ e: Aicode.Element) -> [Int] {
        let n = shape.reduce(1, *), rest = n / shape[0], row = 64 / e.size
        if rest * e.size % 64 == 0 { return [shape[0], rest] }
        precondition(n % row == 0, "a weight of \(shape) in no rows of 64 bytes")
        return [n / row, row]
    }

    /// **The window `name` joined** in `m` (`Geometry.Window`): its bands, the band before's last row and the band
    /// after's first, along the rows (`[1, C, rows + halo, W]`), and their factors (`[rows + halo]` f32) under the guard;
    /// a whole read cut back to the plane's rows. The mask is the client's to apply (`<name>_mask`): what it zeroes
    /// depends on what it reads (a norm's output is not zero where its input is).
    public static func join(_ m: Aicode.Module, _ name: String, _ w: Geometry.Window) -> (value: Int, factor: Int?) {
        let b = m.body
        func rows(_ suffix: String, _ axis: Int) -> Int {
            var parts: [Int] = []
            if w.before { parts.append(b.slice(m.argument("\(name)_before\(suffix)"), axis: axis, start: w.held / w.bands - 1, size: 1)) }
            parts += (0..<w.bands).map { m.argument("\(name)\($0)\(suffix)") }
            if w.after { parts.append(b.slice(m.argument("\(name)_after\(suffix)"), axis: axis, start: 0, size: 1)) }
            let joined = parts.count == 1 ? parts[0] : b.concat(parts, axis: axis)
            return w.held > w.rows && !w.before && !w.after ? b.slice(joined, axis: axis, start: 0, size: w.rows) : joined
        }
        return (rows("", 2), w.factors ? rows("_factor", 0) : nil)
    }

    /// **The output `name` as its bands** (`Geometry.Output`): `value` `[1, C, rows, W]` and its factors `[rows]` cut in
    /// `name0`, `name0_factor`, …; a whole write's rows past the plane's added, zeros and factors of 1. A slice a band,
    /// one band too: the function's kinds of ops the same at every band's height (`Bands.extent`).
    public static func split(_ b: Aicode.Block, _ name: String, value: Int, factor: Int?, _ o: Geometry.Output) -> [(String, Int)] {
        var (v, f) = (value, factor)
        if o.held > o.rows {
            let pad = o.held - o.rows, e = Aicode.shape(b.types[v]).element
            v = b.concat([v, b.constant(Aicode.dense(Aicode.scalar(0, e), splat: true), Aicode.tensor([1, o.channels, pad, o.width], e))], axis: 2)
            if let g = f { f = b.concat([g, b.constant(Aicode.dense(Aicode.scalar(1, .f32), splat: true), Aicode.tensor([pad], .f32))], axis: 0) }
        }
        let rho = o.held / o.bands
        var out: [(String, Int)] = []
        for i in 0..<o.bands {
            out.append(("\(name)\(i)", b.slice(v, axis: 2, start: i * rho, size: rho)))
            if let f { out.append(("\(name)\(i)_factor", b.slice(f, axis: 0, start: i * rho, size: rho))) }
        }
        return out
    }
}
