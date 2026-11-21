import CoreAI
import Foundation
import Metal
import SiliCore
import Testing

/// **The facts of Core AI the engine's design rests on** (`Arena`, `Weights`, `Recipe.codes`), each on small shapes, one
/// fact a test, true on the M1 Pro and the M5 Pro. If one turns, the design it holds up is to be read again, not the test.
extension Runs {
    /// An input whose buffer does not start on a page is copied by Core AI's GPU at every call: what a call adds to the
    /// footprint grows by the tensor's size (measured 17/11: 41.4 MiB more a pass for two tensors of 41.25 placed on
    /// 8 KiB). Hence `Arena.take`: every tensor on a page. The bar: the off-page call's share at least half the weight's
    /// size above the on-page call's.
    @Test(.enabled(if: metalEnabled))
    func anInputOffThePageIsCopiedAtEveryCall() async throws {
        let bench = try await Bench()
        let (rows, k, n) = (512, 4096, 4096), wBytes = k * n * 2
        // A heap of our own, one copy of w on a page and one 8 KiB past one: the arena never places a tensor so.
        let d = MTLHeapDescriptor()
        d.type = .placement; d.storageMode = .shared; d.size = 2 * Arena.aligned(wBytes) + Arena.page
        let heap = try #require(bench.device.makeHeap(descriptor: d))
        let on = try #require(heap.makeBuffer(length: wBytes, options: .storageModeShared, offset: 0))
        let off = try #require(heap.makeBuffer(length: wBytes, options: .storageModeShared, offset: Arena.aligned(wBytes) + 8192))
        var rng = Seeded(state: 11)
        let ws = rng.uniform(k * n, -0.05...0.05).map { Float16($0) }
        ws.withUnsafeBytes { on.contents().copyMemory(from: $0.baseAddress!, byteCount: wBytes); off.contents().copyMemory(from: $0.baseAddress!, byteCount: wBytes) }
        let arena = try Arena(bench.device, bytes: Arena.aligned(rows * k * 2) + 2 * Arena.aligned(rows * n * 2))
        let x = arena.take(rows * k * 2), y0 = arena.take(rows * n * 2), y1 = arena.take(rows * n * 2)
        arena.write(rng.uniform(rows * k), at: x)

        var m = Aicode.Module(arguments: [("x", Aicode.tensor([rows, k], .f16)), ("w", Aicode.tensor([k, n], .f16))])
        m.outputs = [("y", m.body.product(m.argument("x"), m.argument("w")))]
        let f = try await Functions.load(try Functions.asset(m, "run-product-\(rows)-\(k)-\(n)"))
        func call(_ w: MTLBuffer, _ y: Int) async throws {
            try await Run.call(f, bench.stream, inputs: ["x": View(buffer: arena.buffer(x), type: .float16, shape: [rows, k]), "w": View(buffer: w, type: .float16, shape: [k, n])],
                               outputs: [("y", View(buffer: arena.buffer(y), type: .float16, shape: [rows, n]))])
        }
        let shareOn = try await share { try await call(on, y0) }
        let shareOff = try await share { try await call(off, y1) }
        #expect(arena.halves(y0, rows * n) == arena.halves(y1, rows * n))
        #expect(shareOff - shareOn > wBytes / 2, "on a page \(Memory.mib(shareOn)), off \(Memory.mib(shareOff)), w \(Memory.mib(wBytes))")
    }

    /// The file's codes `[out, in]` given to a function that reads `[in, out]` as a transposed view (strides `[1, in]`)
    /// are copied at every call, and the copy is held while the function's calls read the view (measured 17/11 on the
    /// M1 Pro, 1024 × 3840 × 10240: 277 ms a call against 24.5, +37 MiB of share; 18/11 on the M5 Pro, 512 rows: 145 ms
    /// against 2.4, +37.5 MiB of the device's, given back by the next call on the codes themselves). Hence the codes
    /// written once into the arena in the layout the function reads (`Recipe.codes`, `CodesLayout`). The bar: the view's
    /// call at least 2 × the plain call's time, the settled footprint after it at least half the codes' size above the
    /// one after the plain call, and the same product.
    @Test(.enabled(if: metalEnabled))
    func aTransposedViewIsCopiedAtEveryCall() async throws {
        let bench = try await Bench()
        let p = try await CodesBench(bench)
        let f = try await p.function()
        let tPlain = try await median(3) { try await p.call(f, codes: p.transposed, strides: []) }
        let y = p.arena.halves(p.y, p.rows * p.n)
        let plain = Memory.settled(50)
        let tView = try await median(3) { try await p.call(f, codes: p.laid, strides: [1, p.k]) }
        let viewed = Memory.settled(50)
        #expect(relative(p.arena.halves(p.y, p.rows * p.n), y) < 1e-3)
        #expect(tView > 2 * tPlain, Comment(rawValue: String(format: "a view %.1f ms, the transposed codes %.1f ms", 1e3 * tView, 1e3 * tPlain)))
        #expect(viewed - plain > p.n * p.k / 2, "settled after the transposed codes \(Memory.mib(plain)), after the view \(Memory.mib(viewed)), codes \(Memory.mib(p.n * p.k))")
    }

    /// **A function holds the buffers of its last call**, inputs and outputs, until it is called again or let go —
    /// another function's calls do not release them (measured 18/11 on the M5 Pro, a heap of 16 MiB beside its tensors:
    /// held 2 s after its arena was let go while the function lived, whatever function was called next; gone at once
    /// with the function, or when no tensor of the last call was in it; 18/11 on the M1 Pro, three stages of a model in
    /// a row: a 1 MiB output kept a 6 GiB heap past its stage). Hence `Arena.take(apart:)` for what a stage's last call
    /// writes. The bar: a heap held 2 s after its arena while its function lives and others are called; gone within 3 s
    /// once its function is called on buffers apart, or let go.
    @Test(.enabled(if: metalEnabled))
    func aFunctionHoldsItsLastCallsBuffers() async throws {
        let bench = try await Bench()
        let (rows, d) = (256, 1024), bytes = rows * d * 2
        func function(_ name: String, _ c: Double) async throws -> InferenceFunction {
            var m = Aicode.Module(arguments: [("x", Aicode.tensor([rows, d], .f16))])
            m.outputs = [("y", m.body.scale("mul", m.argument("x"), c))]
            return try await Functions.load(try Functions.asset(m, "run-\(name)-\(rows)-\(d)"))
        }
        /// One call of `f` on an arena with a heap of 16 MiB and x and y in it, or apart: the heap.
        func run(_ f: InferenceFunction, apart: Bool) async throws -> MTLHeap? {
            let tensors = 2 * Arena.aligned(bytes)
            let arena = try Arena(bench.device, bytes: tensors + (16 << 20), apart: apart ? tensors : 0)
            let filler = arena.take(16 << 20)
            let x = apart ? arena.take(apart: bytes) : arena.take(bytes), y = apart ? arena.take(apart: bytes) : arena.take(bytes)
            arena.write([Float](repeating: 1, count: rows * d), at: x)
            try await Run.call(f, bench.stream, inputs: ["x": View(buffer: arena.buffer(x), type: .float16, shape: [rows, d])],
                               outputs: [("y", View(buffer: arena.buffer(y), type: .float16, shape: [rows, d]))])
            #expect(arena.halves(y, 2) == [2, 2])
            return arena.buffer(filler).heap
        }
        func gone(_ heap: () -> MTLHeap?, within seconds: Double) async throws -> Bool {
            for _ in 0..<Int(seconds * 20) { if heap() == nil { return true }; try await Task.sleep(for: .milliseconds(50)) }
            return heap() == nil
        }
        let f = try await function("double", 2), other = try await function("other", 2)
        weak let held = try await run(f, apart: false)
        _ = try await run(other, apart: true)
        try await Task.sleep(for: .seconds(2))
        #expect(held != nil, "a heap of the last call's tensors let go with its arena while its function lived")
        weak let apart = try await run(f, apart: true)
        #expect(try await gone({ held }, within: 3), "a heap still held after its function's next call")
        #expect(try await gone({ apart }, within: 3), "a heap held although its arena's tensors were apart")
        var dropped: InferenceFunction? = try await function("dropped", 2)
        weak let last = try await run(dropped!, apart: false)
        dropped = nil
        #expect(try await gone({ last }, within: 3), "a heap still held after its function was let go")
        _ = (f, other)
    }

    /// The GPU reads a surface of the arena (an `MTLBuffer` over its bytes) as it reads the heap: the same values to the
    /// bit (measured 17/11: 68.7 ms against 69.0). Hence the ANE's operands as surfaces the GPU reads too.
    @Test(.enabled(if: metalEnabled))
    func theGPUReadsASurfaceAsItReadsTheHeap() async throws {
        let bench = try await Bench()
        let p = try SurfaceBench(bench)
        let f = try await Functions.load(try p.asset())
        try await Run.call(f, bench.stream, inputs: p.views(surfaces: false), outputs: [("y", p.view(p.heap.y, [1, p.rows, p.n]))])
        try await Run.call(f, bench.stream, inputs: p.views(surfaces: true), outputs: [("y", p.view(p.surface.y, [1, p.rows, p.n]))])
        let onHeap = p.arena.halves(p.heap.y, p.rows * p.n), onSurfaces = p.arena.halves(p.surface.y, p.rows * p.n)
        #expect(onHeap == onSurfaces)
        #expect(relative(onHeap, p.want) < 2e-3)
    }

    /// The ANE runs a function of its form (the q64 product) on the arena's surfaces, by `Run.neural`, its result within
    /// f16's rounding of the CPU's (Core AI may place it elsewhere and says nothing: the values are what is checked).
    @Test(.enabled(if: metalEnabled))
    func theANERunsOnTheArenasSurfaces() async throws {
        let bench = try await Bench()
        let p = try SurfaceBench(bench)
        let f = try await Functions.load(try p.asset(), unit: .neuralEngine)
        try await Run.neural(f, inputs: p.views(surfaces: true, whole: true), outputs: [("y", p.view(p.surface.y, [1, p.rows, p.n], whole: true))])
        let e = relative(p.arena.halves(p.surface.y, p.rows * p.n), p.want)
        #expect(e < 5e-3, "relative error \(e)")
    }
}

/// The codes of a product at its measured shape, as a file lays them (`[out, in]`) and as the arena holds them (`[in, out]`).
private struct CodesBench {
    let bench: Bench
    let rows = 512, k = 3840, n = 10240
    let arena: Arena
    let x, laid, transposed, scale, y: Int

    init(_ bench: Bench) async throws {
        self.bench = bench
        arena = try Arena(bench.device, bytes: 2 * Arena.aligned(n * k) + Arena.aligned(n * 2) + Arena.aligned(rows * k * 2) + Arena.aligned(rows * n * 2))
        laid = arena.take(n * k); transposed = arena.take(n * k); scale = arena.take(n * 2); x = arena.take(rows * k * 2); y = arena.take(rows * n * 2)
        var rng = Seeded(state: 13)
        let q = (0..<n * k).map { _ in Int8(Int(rng.next() * 255) - 127) }
        var t = [Int8](repeating: 0, count: n * k)
        for o in 0..<n { for i in 0..<k { t[i * n + o] = q[o * k + i] } }
        arena.write(codes: q, at: laid); arena.write(codes: t, at: transposed)
        arena.write(rng.uniform(n, 0.001...0.01), at: scale)
        arena.write(rng.uniform(rows * k), at: x)
    }

    /// The product of the codes `[in, out]`, read as they lie.
    func function() async throws -> InferenceFunction {
        var m = Aicode.Module(arguments: [("x", Aicode.tensor([rows, k], .f16)), ("w", Aicode.tensor([k, n], .si8)), ("w_scale", Aicode.tensor([n], .f16))])
        var constants = Aicode.ScalingConstants()
        m.outputs = [("y", m.body.w8a16(m.argument("x"), codes: m.argument("w"), scale: m.argument("w_scale"), constants: &constants))]
        return try await Functions.load(try Functions.asset(m, "run-codes-\(rows)-\(k)-\(n)"))
    }

    /// One call reading `codes` as `[in, out]`: the transposed codes as they lie or, with `strides`, a view of the file's.
    func call(_ f: InferenceFunction, codes: Int, strides: [Int]) async throws {
        let shape = [k, n]
        try await Run.call(f, bench.stream, inputs: ["x": View(buffer: arena.buffer(x), type: .float16, shape: [rows, k]),
                                                    "w": View(buffer: arena.buffer(codes), type: .int8, shape: shape, strides: strides),
                                                    "w_scale": View(buffer: arena.buffer(scale), type: .float16, shape: [n])],
                           outputs: [("y", View(buffer: arena.buffer(y), type: .float16, shape: [rows, n]))])
    }
}

/// A q64 product (the ANE's form, `[1, rows, k] · [1, k, n]`) with its operands both in the heap and as surfaces of the
/// arena, the same values in each; the CPU's result.
private struct SurfaceBench {
    let rows = 128, k = 512, n = 256
    let arena: Arena
    let heap: (x: Int, w: Int, s: Int, y: Int)
    let surface: (x: Int, w: Int, s: Int, y: Int)
    let want: [Float]

    init(_ bench: Bench) throws {
        let (xb, wb, sb, yb) = (rows * k * 2, k * n, n * 2, rows * n * 2)
        let all = [xb, wb, sb, yb].reduce(0) { $0 + Arena.aligned($1) }
        arena = try Arena(bench.device, bytes: 2 * all, apart: all)
        heap = (arena.take(xb), arena.take(wb), arena.take(sb), arena.take(yb))
        surface = (arena.take(surface: [1, rows, k], element: 2), arena.take(surface: [1, k, n], element: 1),
                   arena.take(surface: [1, 1, n], element: 2), arena.take(surface: [1, rows, n], element: 2))
        var rng = Seeded(state: 17)
        let xs = rng.uniform(rows * k), q = (0..<k * n).map { _ in Int8(Int(rng.next() * 255) - 127) }, s = rng.uniform(n, 0.001...0.01)
        for at in [heap.x, surface.x] { arena.write(xs, at: at) }
        for at in [heap.w, surface.w] { arena.write(codes: q, at: at) }
        var s16: [Float] = []
        for at in [heap.s, surface.s] { s16 = arena.write(s, at: at) }
        let x16 = arena.halves(heap.x, rows * k)
        let (r, kk, nn) = (rows, k, n)
        want = product(x16, rows: r, k: kk, n: nn) { i, o in Double(q[i * nn + o]) * Double(s16[o]) }
    }

    func asset() throws -> URL {
        var m = Aicode.Module(arguments: [("x", Aicode.tensor([1, rows, k], .f16)), ("w", Aicode.tensor([1, k, n], .si8)), ("w_scale", Aicode.tensor([1, 1, n], .f16))])
        var constants = Aicode.ScalingConstants()
        m.outputs = [("y", m.body.q64(m.argument("x"), codes: m.argument("w"), scale: m.argument("w_scale"), constants: &constants))]
        return try Functions.asset(m, "run-q64-\(rows)-\(k)-\(n)")
    }

    func view(_ key: Int, _ shape: [Int], whole: Bool = false) -> View {
        let element = shape == [1, k, n] ? NDArray.ScalarType.int8 : .float16
        return View(buffer: arena.buffer(key), type: element, shape: shape, surface: whole ? arena.surface(key) : nil)
    }

    /// The inputs from the heap or from the surfaces; `whole`: the surfaces themselves, what the ANE reads.
    func views(surfaces: Bool, whole: Bool = false) -> [String: View] {
        let at = surfaces ? surface : heap
        return ["x": view(at.x, [1, rows, k], whole: whole), "w": view(at.w, [1, k, n], whole: whole), "w_scale": view(at.s, [1, 1, n], whole: whole)]
    }
}
