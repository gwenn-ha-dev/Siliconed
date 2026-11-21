import CoreAI
import Foundation
import Metal
import SiliCore
import Testing

extension Runs {
    /// **Probe (19/11, temporary): is a view that is not what the descriptor asks copied at every call?** The shapes of
    /// a VAE band: `[1, C, R, W]` f16 read in a slab of `R + 2` rows a plane. Each case's share (what a call adds to the
    /// settled footprint) and time, against the plain tensor on its own page.
    @Test(.enabled(if: metalEnabled && ProcessInfo.processInfo.environment["SILICORE_LAYOUT_PROBE"] != nil))
    func layoutProbe() async throws {
        let bench = try await Bench()
        let (c, r, w) = (128, 256, 512), shape = [1, c, r, w], bytes = c * r * w * 2, slab = c * (r + 2) * w * 2
        let gaps = [c * (r + 2) * w, (r + 2) * w, w, 1]
        var m = Aicode.Module(arguments: [("x", Aicode.tensor(shape, .f16))])
        m.outputs = [("y", m.body.scale("mul", m.argument("x"), 2))]
        let f = try await Functions.load(try Functions.asset(m, "run-layout-probe-\(c)-\(r)-\(w)"))
        if case .ndArray(let d)? = f.descriptor.inputDescriptor(of: "x") { print("probe: x asked strides \(d.preferredStrides) ≥ \(d.minimumByteCount) B") }
        let arena = try Arena(bench.device, bytes: 2 * Arena.aligned(bytes) + 2 * Arena.aligned(slab + w * 2))
        let x = arena.take(bytes), y = arena.take(bytes), xs = arena.take(slab + w * 2), ys = arena.take(slab + w * 2)
        let values = [Float](repeating: 1.5, count: slab / 2 + w)
        _ = arena.write(Array(values.prefix(bytes / 2)), at: x)
        _ = arena.write(values, at: xs)
        func plain(_ key: Int) -> View { View(buffer: arena.buffer(key), type: .float16, shape: shape) }
        let cases: [(String, View, View)] = [
            ("plain, on its page", plain(x), plain(y)),
            ("input with gaps, offset 0", View(buffer: arena.buffer(xs), type: .float16, shape: shape, strides: gaps), plain(y)),
            ("input with gaps, one row in (off the page)", View(buffer: arena.buffer(xs), offset: w * 2, type: .float16, shape: shape, strides: gaps), plain(y)),
            ("input contiguous, one row in (off the page)", View(buffer: arena.buffer(xs), offset: w * 2, type: .float16, shape: shape), plain(y)),
            ("output with gaps, offset 0", plain(x), View(buffer: arena.buffer(ys), type: .float16, shape: shape, strides: gaps)),
            ("output with gaps, one row in", plain(x), View(buffer: arena.buffer(ys), offset: w * 2, type: .float16, shape: shape, strides: gaps)),
        ]
        // A plain view inside a buffer four times its size: at its start, from a page 1 MiB in, as input, as output.
        let big = try Arena(bench.device, bytes: 2 * Arena.aligned(4 * bytes))
        let bx = big.take(4 * bytes), by = big.take(4 * bytes)
        _ = big.write([Float](repeating: 1.5, count: 2 * bytes), at: bx)
        func bv(_ key: Int, _ offset: Int) -> View { View(buffer: big.buffer(key), offset: offset, type: .float16, shape: shape) }
        let extent: [(String, View, View)] = [
            ("in and out at the start of buffers 4 × their size", bv(bx, 0), bv(by, 0)),
            ("in at the start of a buffer 4 × its size", bv(bx, 0), plain(y)),
            ("in from a page 1 MiB into a buffer 4 × its size", bv(bx, 1 << 20), plain(y)),
            ("out from a page 1 MiB into a buffer 4 × its size", plain(x), bv(by, 1 << 20)),
            ("in from a page 64 MiB into a buffer 4 × its size", bv(bx, 64 << 20), plain(y)),
        ]
        for (name, input, output) in cases + extent {
            func call() async throws { try await Run.call(f, bench.stream, inputs: ["x": input], outputs: [("y", output)]) }
            let s = try await share(call)
            let t = try await median(5, call)
            print(String(format: "probe: %@: share %@ (tensor %@), %.2f ms", name, Memory.mib(s), Memory.mib(bytes), 1e3 * t))
        }
    }
}

extension Runs {
    /// **Probe (19/11, temporary): a VAE band's convolution under each layout.** Flux's decoder at 1024², its last level:
    /// 128 channels, bands of 16 rows (2¹⁴ positions) over 1024 columns, a 3 × 3 kernel constant. Each case's share and
    /// time: the band viewed in a `[C, H + 2, W]` slab (today's), the band a tensor of its own, the band a page-aligned
    /// view of a contiguous region (rows outermost, `[H + 2, C, W]`, transposed inside the function).
    @Test(.enabled(if: metalEnabled && ProcessInfo.processInfo.environment["SILICORE_LAYOUT_PROBE"] != nil))
    func layoutProbeConvolution() async throws {
        let bench = try await Bench()
        let (c, r, w, h) = (128, 16, 1024, 1024), e = 2
        var rng = Seeded(state: 5)
        let kernel = rng.uniform(c * c * 9, -0.03...0.03).map { Float16($0).bitPattern }
        let kernelBytes = kernel.withUnsafeBytes { Array($0) }
        func module(rowsOutermost: Bool) -> Aicode.Module {
            let inShape = rowsOutermost ? [r + 2, c, w] : [1, c, r + 2, w]
            var m = Aicode.Module(arguments: [("u", Aicode.tensor(inShape, .f16))])
            let k = m.constant("kernel", kernelBytes, Aicode.tensor([c, c, 3, 3], .f16))
            var u = m.argument("u")
            if rowsOutermost { u = m.body.reshape(m.body.transpose(u, [1, 0, 2]), [1, c, r + 2, w]) }
            var y = m.body.convolution(u, k, stride: [1, 1], padding: [0, 1], dilation: [1, 1], groups: 1, result: Aicode.tensor([1, c, r, w], .f16))
            if rowsOutermost { y = m.body.transpose(m.body.reshape(y, [c, r, w]), [1, 0, 2]) }
            m.outputs = [("y", y)]
            return m
        }
        let b = 10
        /// The slab whole as the input, the band's window sliced inside.
        func sliced() -> Aicode.Module {
            var m = Aicode.Module(arguments: [("u", Aicode.tensor([1, c, h + 2, w], .f16))])
            let k = m.constant("kernel", kernelBytes, Aicode.tensor([c, c, 3, 3], .f16))
            let u = m.body.slice(m.argument("u"), axis: 2, start: b * r, size: r + 2)
            m.outputs = [("y", m.body.convolution(u, k, stride: [1, 1], padding: [0, 1], dilation: [1, 1], groups: 1, result: Aicode.tensor([1, c, r, w], .f16)))]
            return m
        }
        /// Each band a tensor of its own `[1, C, r, W]`: the window the band's last row before, the band, the next's first
        /// row, cut and joined inside; the band out, a tensor of its own.
        func neighbours() -> Aicode.Module {
            let band = Aicode.tensor([1, c, r, w], .f16)
            var m = Aicode.Module(arguments: [("before", band), ("u", band), ("after", band)])
            let k = m.constant("kernel", kernelBytes, Aicode.tensor([c, c, 3, 3], .f16))
            let b = m.body
            let window = b.concat([b.slice(m.argument("before"), axis: 2, start: r - 1, size: 1), m.argument("u"), b.slice(m.argument("after"), axis: 2, start: 0, size: 1)], axis: 2)
            m.outputs = [("y", b.convolution(window, k, stride: [1, 1], padding: [0, 1], dilation: [1, 1], groups: 1, result: Aicode.tensor([1, c, r, w], .f16)))]
            return m
        }
        let joined = try await Functions.load(try Functions.asset(neighbours(), "run-layout-conv-neighbours"))
        let slicedInside = try await Functions.load(try Functions.asset(sliced(), "run-layout-conv-sliced"))
        let nchw = try await Functions.load(try Functions.asset(module(rowsOutermost: false), "run-layout-conv-nchw"))
        let hcw = try await Functions.load(try Functions.asset(module(rowsOutermost: true), "run-layout-conv-hcw"))
        let slab = c * (h + 2) * w * e, window = c * (r + 2) * w * e, band = c * r * w * e
        let arena = try Arena(bench.device, bytes: 4 * Arena.aligned(slab) + Arena.aligned(window) + Arena.aligned(band))
        let xs = arena.take(slab), ys = arena.take(slab), xh = arena.take(slab), yh = arena.take(slab), u = arena.take(window), y = arena.take(band)
        _ = arena.write(rng.uniform(slab / e), at: xs)
        _ = arena.write(rng.uniform(slab / e), at: xh)
        _ = arena.write(rng.uniform(window / e), at: u)
        let gaps = [c * (h + 2) * w, (h + 2) * w, w, 1]
        func v(_ key: Int, _ shape: [Int], offset: Int = 0, strides: [Int] = []) -> View {
            View(buffer: arena.buffer(key), offset: offset, type: .float16, shape: shape, strides: strides)
        }
        let cases: [(String, InferenceFunction, View, View)] = [
            ("[C, H+2, W] slab, the band a view with gaps (today)", nchw,
             v(xs, [1, c, r + 2, w], offset: b * r * w * e, strides: gaps), v(ys, [1, c, r, w], offset: (1 + b * r) * w * e, strides: gaps)),
            ("the band a tensor of its own", nchw, v(u, [1, c, r + 2, w]), v(y, [1, c, r, w])),
            ("[H+2, C, W] slab, the band a page-aligned view, transposed inside", hcw,
             v(xh, [r + 2, c, w], offset: b * r * c * w * e), v(yh, [r, c, w], offset: (1 + b * r) * c * w * e)),
        ]
        var reference: [Float] = []
        for (name, f, input, output) in cases {
            func call() async throws { try await Run.call(f, bench.stream, inputs: ["u": input], outputs: [("y", output)]) }
            let s = try await share(call)
            let t = try await median(7, call)
            print(String(format: "probe: %@: share %@ (window %@), %.3f ms", name, Memory.mib(s), Memory.mib(window), 1e3 * t))
            if name.hasPrefix("[C, H+2, W] slab, the band") {
                let p = output.buffer.contents().advanced(by: output.offset).assumingMemoryBound(to: Float16.self)
                reference = (0..<c).flatMap { ch in (0..<r * w).map { Float(p[ch * (h + 2) * w + $0]) } }
            }
        }
        do {
            let bands = try Arena(bench.device, bytes: 4 * Arena.aligned(band))
            let (before, own, after, out) = (bands.take(band), bands.take(band), bands.take(band), bands.take(band))
            let x = arena.pointer(xs).assumingMemoryBound(to: Float16.self)
            for (key, first) in [(before, b * r - r + 1), (own, b * r + 1), (after, b * r + r + 1)] {
                let p = bands.pointer(key).assumingMemoryBound(to: Float16.self)
                for ch in 0..<c { for i in 0..<(r * w) { p[ch * r * w + i] = x[ch * (h + 2) * w + first * w + i] } }
            }
            func bv(_ key: Int) -> View { View(buffer: bands.buffer(key), type: .float16, shape: [1, c, r, w]) }
            func call() async throws {
                try await Run.call(joined, bench.stream, inputs: ["before": bv(before), "u": bv(own), "after": bv(after)], outputs: [("y", bv(out))])
            }
            let s = try await share(call)
            let t = try await median(7, call)
            print(String(format: "probe: each band its own tensor, the window joined inside from three: share %@, %.3f ms, against today's %.1e",
                         Memory.mib(s), 1e3 * t, relative(bands.halves(out, c * r * w), reference)))
            // The same bands one after the other in one buffer (bands first, `[n, C, r, W]`), each a view from its page.
            let n = h / r, one = try Arena(bench.device, bytes: 2 * Arena.aligned(n * band))
            let slabIn = one.take(n * band), slabOut = one.take(n * band)
            let q = one.pointer(slabIn).assumingMemoryBound(to: Float16.self)
            for j in 0..<n { for ch in 0..<c { for i in 0..<(r * w) { q[j * c * r * w + ch * r * w + i] = x[ch * (h + 2) * w + (j * r + 1) * w + i] } } }
            func ov(_ key: Int, _ j: Int) -> View { View(buffer: one.buffer(key), offset: j * band, type: .float16, shape: [1, c, r, w]) }
            precondition(band % Arena.page == 0)
            func callOne() async throws {
                try await Run.call(joined, bench.stream, inputs: ["before": ov(slabIn, b - 1), "u": ov(slabIn, b), "after": ov(slabIn, b + 1)], outputs: [("y", ov(slabOut, b))])
            }
            let s1 = try await share(callOne)
            let t1 = try await median(7, callOne)
            let got = one.halves(slabOut, (b + 1) * c * r * w).suffix(c * r * w)
            print(String(format: "probe: the bands first in one buffer, each a view from its page, the window joined inside: share %@, %.3f ms, against today's %.1e",
                         Memory.mib(s1), 1e3 * t1, relative(Array(got), reference)))
        }
        do {
            func call() async throws { try await Run.call(slicedInside, bench.stream, inputs: ["u": v(xs, [1, c, h + 2, w])], outputs: [("y", v(y, [1, c, r, w]))]) }
            let s = try await share(call)
            let t = try await median(7, call)
            print(String(format: "probe: [C, H+2, W] slab whole in, the window sliced inside, the band its own tensor: share %@ (window %@), %.3f ms, against today's %.1e",
                         Memory.mib(s), Memory.mib(window), 1e3 * t, relative(arena.halves(y, c * r * w), reference)))
        }
    }
}

extension Runs {
    /// **Probe (19/11, temporary): a DiT tile's keys written into the whole sequence's.** Z at 1024²: 30 heads of 128, a
    /// tile of 1024 rows, 4128 keys, bf16. The tile's `qkv` `[r, 3·d]` to its keys heads first: into a view of the
    /// sequence's `[1, 30, keys, 128]` with gaps (today's), into a tensor of its own.
    @Test(.enabled(if: metalEnabled && ProcessInfo.processInfo.environment["SILICORE_LAYOUT_PROBE"] != nil))
    func layoutProbeKeys() async throws {
        let bench = try await Bench()
        let (r, heads, hd, keys, r0) = (1024, 30, 128, 4128, 2048), d = heads * hd, e = 2
        func module() -> Aicode.Module {
            var m = Aicode.Module(arguments: [("qkv", Aicode.tensor([r, 3 * d], .bf16))])
            let b = m.body
            m.outputs = [("k", b.reshape(b.transpose(b.reshape(b.slice(m.argument("qkv"), axis: 1, start: d, size: d), [r, heads, hd]), [1, 0, 2]), [1, heads, r, hd]))]
            return m
        }
        let plain = try await Functions.load(try Functions.asset(module(), "run-layout-keys"))
        let arena = try Arena(bench.device, bytes: Arena.aligned(r * 3 * d * e) + Arena.aligned(heads * keys * hd * e) + Arena.aligned(heads * r * hd * e))
        let qkv = arena.take(r * 3 * d * e), all = arena.take(heads * keys * hd * e), own = arena.take(heads * r * hd * e)
        var rng = Seeded(state: 3)
        _ = arena.write(rng.uniform(r * 3 * d), at: qkv)
        func v(_ key: Int, _ shape: [Int], offset: Int = 0, strides: [Int] = []) -> View { View(buffer: arena.buffer(key), offset: offset, type: .bfloat16, shape: shape, strides: strides) }
        let input = ["qkv": v(qkv, [r, 3 * d])]
        let cases: [(String, () async throws -> Void)] = [
            ("into the sequence's, a view with gaps (today)", { try await Run.call(plain, bench.stream, inputs: input, outputs: [("k", v(all, [1, heads, r, hd], offset: r0 * hd * e, strides: [heads * keys * hd, keys * hd, hd, 1]))]) }),
            ("into a tensor of its own", { try await Run.call(plain, bench.stream, inputs: input, outputs: [("k", v(own, [1, heads, r, hd]))]) }),
        ]
        for (name, call) in cases {
            let s = try await share(call)
            let t = try await median(7, call)
            print(String(format: "probe: keys %@: share %@, %.3f ms", name, Memory.mib(s), 1e3 * t))
        }
    }
}

extension Runs {
    /// **Probe (19/11, temporary): a DiT tile read from and written into the whole stream.** Z at 1024²: the stream
    /// `[4128, 3840]` bf16, a tile of 1024 rows from row 2048, a product by `[3840, 3840]`. The tile a view of the
    /// stream's buffer (today's), against a tensor of its own.
    @Test(.enabled(if: metalEnabled && ProcessInfo.processInfo.environment["SILICORE_LAYOUT_PROBE"] != nil))
    func layoutProbeTile() async throws {
        let bench = try await Bench()
        let (rows, r, d, r0) = (4128, 1024, 3840, 2048), e = 2
        var m = Aicode.Module(arguments: [("x", Aicode.tensor([r, d], .bf16)), ("w", Aicode.tensor([d, d], .bf16))])
        m.outputs = [("y", m.body.matmul(m.argument("x"), m.argument("w"), result: Aicode.tensor([r, d], .bf16)))]
        let f = try await Functions.load(try Functions.asset(m, "run-layout-tile"))
        let arena = try Arena(bench.device, bytes: 2 * Arena.aligned(rows * d * e) + 3 * Arena.aligned(r * d * e) + Arena.aligned(d * d * e))
        let stream = arena.take(rows * d * e), out = arena.take(rows * d * e), x = arena.take(r * d * e), y = arena.take(r * d * e), w = arena.take(d * d * e)
        var rng = Seeded(state: 7)
        _ = arena.write(rng.uniform(rows * d, -1...1), at: stream)
        _ = arena.write(rng.uniform(d * d, -0.02...0.02), at: w)
        memcpy(arena.pointer(x), arena.pointer(stream) + r0 * d * e, r * d * e)
        func v(_ key: Int, _ shape: [Int], offset: Int = 0) -> View { View(buffer: arena.buffer(key), offset: offset, type: .bfloat16, shape: shape) }
        let cases: [(String, View, View)] = [
            ("x and y views of the stream's buffers (today)", v(stream, [r, d], offset: r0 * d * e), v(out, [r, d], offset: r0 * d * e)),
            ("x a view, y its own", v(stream, [r, d], offset: r0 * d * e), v(y, [r, d])),
            ("x its own, y a view", v(x, [r, d]), v(out, [r, d], offset: r0 * d * e)),
            ("x and y their own", v(x, [r, d]), v(y, [r, d])),
        ]
        // The same tiles, each a buffer of its own placed over the stream's bytes (a placement heap aliases at will).
        let hd = MTLHeapDescriptor()
        hd.type = .placement; hd.storageMode = .shared; hd.size = 2 * Arena.aligned(rows * d * e)
        let heap = try #require(bench.device.makeHeap(descriptor: hd))
        let hs = try #require(heap.makeBuffer(length: rows * d * e, options: .storageModeShared, offset: 0))
        let ho = try #require(heap.makeBuffer(length: rows * d * e, options: .storageModeShared, offset: Arena.aligned(rows * d * e)))
        memcpy(hs.contents(), arena.pointer(stream), rows * d * e)
        precondition((r0 * d * e) % Arena.page == 0)
        let tileIn = try #require(heap.makeBuffer(length: r * d * e, options: .storageModeShared, offset: r0 * d * e))
        let tileOut = try #require(heap.makeBuffer(length: r * d * e, options: .storageModeShared, offset: Arena.aligned(rows * d * e) + r0 * d * e))
        let aliased: [(String, View, View)] = [
            ("x and y buffers of their own over the stream's bytes", View(buffer: tileIn, type: .bfloat16, shape: [r, d]), View(buffer: tileOut, type: .bfloat16, shape: [r, d])),
        ]
        for (name, input, output) in cases + aliased {
            func call() async throws { try await Run.call(f, bench.stream, inputs: ["x": input, "w": v(w, [d, d])], outputs: [("y", output)]) }
            let s = try await share(call)
            let t = try await median(7, call)
            print(String(format: "probe: tile %@: share %@, %.3f ms", name, Memory.mib(s), 1e3 * t))
        }
        let a = arena.pointer(y).assumingMemoryBound(to: UInt16.self), b = ho.contents().advanced(by: r0 * d * e).assumingMemoryBound(to: UInt16.self)
        print("probe: tile the aliased buffers' product the same bits: \((0..<(r * d)).allSatisfy { a[$0] == b[$0] })")
        _ = hs
    }
}

extension Runs {
    /// **Probe (19/11, temporary): the keys token-first.** Z at 1024²: 4128 keys, 30 heads of 128, a tile of 1024 rows
    /// from row 2048. Today the tile writes its keys heads-first into the sequence's `[1, 30, keys, 128]` through a view
    /// with gaps, and the attention reads them whole. Token-first, `[1, keys, 30, 128]`: the tile writes a block of its
    /// own pages, no transpose, and the attention transposes them inside, before the composite. Bar set before: taken if
    /// write + attention is no slower than today's, the attention's bits the same.
    @Test(.enabled(if: metalEnabled && ProcessInfo.processInfo.environment["SILICORE_LAYOUT_PROBE"] != nil))
    func layoutProbeTokenFirst() async throws {
        let bench = try await Bench()
        let (r, heads, hd, keys, r0) = (1024, 30, 128, 4128, 2048), d = heads * hd, e = 2
        precondition((r0 * d * e) % Arena.page == 0)
        func write(tokenFirst: Bool) -> Aicode.Module {
            var m = Aicode.Module(arguments: [("qkv", Aicode.tensor([r, 3 * d], .bf16))])
            let b = m.body, k = b.reshape(b.slice(m.argument("qkv"), axis: 1, start: d, size: d), [r, heads, hd])
            m.outputs = [("k", tokenFirst ? b.reshape(k, [1, r, heads, hd]) : b.reshape(b.transpose(k, [1, 0, 2]), [1, heads, r, hd]))]
            return m
        }
        func attention(tokenFirst: Bool) -> Aicode.Module {
            let kv = Aicode.tensor(tokenFirst ? [1, keys, heads, hd] : [1, heads, keys, hd], .bf16)
            var m = Aicode.Module(arguments: [("q", Aicode.tensor([1, heads, r, hd], .bf16)), ("k", kv), ("v", kv), ("mask", Aicode.tensor([1, keys], .bf16))])
            let b = m.body
            func headsFirst(_ x: Int) -> Int { tokenFirst ? b.transpose(x, [0, 2, 1, 3]) : x }
            m.outputs = [("o", b.attention([m.argument("q"), headsFirst(m.argument("k")), headsFirst(m.argument("v")), m.argument("mask")], scale: 1 / Double(hd).squareRoot(), symbol: "sdpa_tile"))]
            return m
        }
        let writeToday = try await Functions.load(try Functions.asset(write(tokenFirst: false), "run-layout-tf-write-heads"))
        let writeTF = try await Functions.load(try Functions.asset(write(tokenFirst: true), "run-layout-tf-write-tokens"))
        let attToday = try await Functions.load(try Functions.asset(attention(tokenFirst: false), "run-layout-tf-attention-heads"))
        let attTF = try await Functions.load(try Functions.asset(attention(tokenFirst: true), "run-layout-tf-attention-tokens"))
        let all = heads * keys * hd * e
        let arena = try Arena(bench.device, bytes: Arena.aligned(r * 3 * d * e) + 4 * Arena.aligned(all) + 3 * Arena.aligned(heads * r * hd * e) + Arena.aligned(keys * e))
        let qkv = arena.take(r * 3 * d * e), q = arena.take(heads * r * hd * e)
        let kh = arena.take(all), vh = arena.take(all), kt = arena.take(all), vt = arena.take(all), mask = arena.take(keys * e)
        let oh = arena.take(heads * r * hd * e), ot = arena.take(heads * r * hd * e)
        var rng = Seeded(state: 11)
        _ = arena.write(rng.uniform(r * 3 * d), at: qkv)
        _ = arena.write(rng.uniform(heads * r * hd), at: q)
        _ = arena.write(rng.uniform(all / e), at: kh)
        _ = arena.write(rng.uniform(all / e), at: vh)
        memset(arena.pointer(mask), 0, keys * e)
        // The same keys token-first, by the CPU.
        for (from, to) in [(kh, kt), (vh, vt)] {
            let a = arena.pointer(from).assumingMemoryBound(to: UInt16.self), t = arena.pointer(to).assumingMemoryBound(to: UInt16.self)
            for hh in 0..<heads { for s in 0..<keys { for x in 0..<hd { t[(s * heads + hh) * hd + x] = a[(hh * keys + s) * hd + x] } } }
        }
        func v(_ key: Int, _ shape: [Int], offset: Int = 0, strides: [Int] = []) -> View { View(buffer: arena.buffer(key), offset: offset, type: .bfloat16, shape: shape, strides: strides) }
        let tile = try #require(arena.part(kt, from: r0 * d * e, bytes: r * d * e))
        let input = ["qkv": v(qkv, [r, 3 * d])]
        let cases: [(String, () async throws -> Void)] = [
            ("write heads-first, a view with gaps (today)", { try await Run.call(writeToday, bench.stream, inputs: input, outputs: [("k", v(kh, [1, heads, r, hd], offset: r0 * hd * e, strides: [heads * keys * hd, keys * hd, hd, 1]))]) }),
            ("write token-first, a block of its own pages", { try await Run.call(writeTF, bench.stream, inputs: input, outputs: [("k", View(buffer: tile, type: .bfloat16, shape: [1, r, heads, hd]))]) }),
            ("attention heads-first (today)", { try await Run.call(attToday, bench.stream, inputs: ["q": v(q, [1, heads, r, hd]), "k": v(kh, [1, heads, keys, hd]), "v": v(vh, [1, heads, keys, hd]), "mask": v(mask, [1, keys])], outputs: [("o", v(oh, [1, heads, r, hd]))]) }),
            ("attention token-first, transposed inside", { try await Run.call(attTF, bench.stream, inputs: ["q": v(q, [1, heads, r, hd]), "k": v(kt, [1, keys, heads, hd]), "v": v(vt, [1, keys, heads, hd]), "mask": v(mask, [1, keys])], outputs: [("o", v(ot, [1, heads, r, hd]))]) }),
        ]
        for (name, call) in cases {
            let s = try await share(call)
            let t = try await median(9, call)
            print(String(format: "probe: token-first %@: share %@, %.3f ms", name, Memory.mib(s), 1e3 * t))
        }
        let a = arena.pointer(oh).assumingMemoryBound(to: UInt16.self), b = arena.pointer(ot).assumingMemoryBound(to: UInt16.self)
        let differ = (0..<(heads * r * hd)).filter { a[$0] != b[$0] }.count
        print("probe: token-first attention bits differing: \(differ) of \(heads * r * hd)")
    }
}
