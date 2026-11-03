import Accelerate
import CoreML
import CoreVideo
import Foundation
import Synchronization

/// **The Neural Engine beside the GPU on one product: a GEMM cut by rows, the ANE on the last ones.**
///
///     C[m, n] = A[m, k] · W[k, n]          W the widened fp32 weight (the reserve), A and C fp32
///
///       rows 0 ..< m − r     the GPU (`GEMM.linear`, MPS)                      ─┐ at the same time
///       rows m − r ..< m     the ANE, one Core ML program per (r, k, n):        │
///                              W · λ  ─▶ fp16      λ_j = 2^−⌊log₂ max_i |W_ij|⌋ ─┤
///                              σ · A  ─▶ fp16      σ_i = 2^⌊log₂(2¹⁵ / (max_k |A_ik| · L))⌋, per row
///                              Y = A₁₆ · W₁₆        (fp16 out)                    │
///                              C = Y / (σ · λ)      fp32, then checked finite    ─┘
///     L = max_j Σ_k |W_kj| λ_j, the largest lifted column's L1 norm: |Y_i·| ≤ max|A_i·| · σ_i · L ≤ 2¹⁵, under fp16's 65 504.
///
///     **The same product on a published int8 weight** (`ProductAssignment` with `path: .native` and `ane > 0`, `Int8Weight`): the
///     integers are the program's input, read as they are — no fp32 widening, no fp16 copy of it:
///       q · 2⁻⁶ ─▶ fp16     MIL `dequantize`, a CONSTANT scale (the one scale per column is not allowed to be an input)
///       R · A  ─▶ σ · A ─▶ fp16      convrot: the weight's rotation moves to the activation, in fp32 (`Widen`'s butterflies)
///       Y = A₁₆ · (q · 2⁻⁶)  (fp16 out)
///       C = Y · (64 · s_j / √G) / σ_i     fp32, s_j the weight's scale per column or for the tensor
///     L = max_j Σ_k |q_kj| / 64, the largest column's L1 norm: the bound that keeps Y under 2¹⁵ is `max|R·A_i| · σ_i · L`.
///
///     **The same product on a bf16 weight read as published** (`ProductAssignment` with `path: .bf16` and `ane > 0`: the GPU's rows on
///     the activation rounded to bf16, `gpu/bf16act`, the ANE's last rows beside them): the weight is the reserve's bf16 copy, not a
///     widened one — each value widened as it is read (a shift), then the same λ, L, σ and fp16 conversion as above: the same fp16
///     operand, so the same bits as the widened weight's rows.
///
/// **The rows are the plan's** (`CapacityPlan`, `ProductAssignment.ane`): the solver wrote how many the ANE takes — in units of
/// 256, or the whole product when it runs alone — where the model measures it faster than the GPU alone and the ANE's rows hold
/// the half-precision threshold; the defaults give it none. A render may still refuse it (`Request.neuralEngine`: the app's default
/// on a Standard version), and only the denoisers take it — never an encoder.
///
/// What sets it apart — what a port by analogy with the GPU/AMX cut would get wrong:
///
///   - **the ANE computes in fp16 and flushes what falls under fp16's smallest normal (6·10⁻⁵)**. A DiT's
///     weights sit around 10⁻³, their products with the activations under it: unlifted, 4 of 12 of
///     Anima's products came out at 1.85 relative error. Each weight column is lifted by a power of two
///     to an absmax in [1, 2), undone in fp32 after: exact, the weight unchanged. The activation too, by
///     a power of two per row, up or down, as far as that row's output bound allows: Anima's
///     cross-attention output peaks at 0.017 (3 % of it under the smallest normal) and came out at 1.5 ×
///     bf16's error unlifted, 0.13 × lifted by 2⁸. Per row, not per slice: Z-Image's FFN output spans
///     8.7·10⁵ down to rows a thousand times smaller, and one σ for the slice, set by the largest row,
///     sank the small ones under fp16's normals — the next RMSNorm, per row, came out 11 × bf16's error;
///   - **a non-finite in the ANE's output sends its rows back to the GPU** (`fallbacks`): fp16 overflows
///     where fp32 does not, and the image never depends on the ANE having fit;
///   - **a program has limits the GPU does not**: past K = 8 192 or a 100 MB weight its rate falls by 5,
///     and it is fast only within K = 4 096 and 48 MB (`maximumK`, `maximumWeight`): K is cut into parts, one program each, summed in fp32 — not in fp16
///     inside the program, as one would cut a sum on the GPU;
///   - **its transcendental functions are approximate** (GELU, erf, tanh ~1 %, while a product holds the
///     fp16 rounding): a program here holds products only;
///   - **a program is a few files written here** (`model.mil`, `coremldata.bin`), compiled by Core ML at
///     load and kept per shape: the weight is an INPUT, so one program serves every weight of its shape,
///     and the rows are a multiple of 256, the unit the ANE computes in;
///   - **its operands live in IOSurfaces, one set per program shape**: Core ML copies any other memory
///     into its own surfaces and frees the copies asynchronously, after the engine is gone. The surfaces
///     are the engine's, freed with it, and the memory need sums them over the shapes (`bytes`);
///   - **each output row is one engine's**: moving the cut never re-associates a sum, and the cut is
///     the plan's, frozen: the same seed gives the same bits;
///   - **an int8 weight is an INPUT of the program, and it is `dequantize`, not `cast`**: with `cast` (+ `mul`) the ANE runs
///     4.6–6.3 TFLOP/s, with `dequantize` and a scalar scale 9.4–11.1 against the fp16 input's 3.4–6.5 (M1 Pro, 30/10/2026). The scale must
///     be a constant of the program (`dequantize` refuses an input one, `constexpr_blockwise_shift_scale` crashes the loader on
///     it), so the program holds 2⁻⁶ — exact, bringing |q| ≤ 127 to |w| < 2, the dynamic of the lifted fp16 weight — and the
///     real scale goes into the host's fp32 pass, where the column lift of the fp16 route already was. The result is closer to
///     the exact product than the fp16 route's (worst column 3.22e-4 against 4.04e-4, 30/10/2026): q is exact in fp16 and the weight
///     is no longer rounded;
///   - **a convrot weight rotates the activation, not the weight** (`W = R · q · s`, `x · W = (x · R) · q · s`): the ANE's rows
///     are rotated on the CPU in fp32 before their lift, unnormalized (the 1/√G, a power of two, goes into the column factor).
///     Its peak is the rotated row's, so σ is read after the rotation;
///   - **its K parts hold 4 096 and 50 M values** (`maximumInt8Weight`): the weight is one byte a value, so the byte limit that cut
///     the fp16 route at 48 MB lets a program twice as large through — 3 840 × 10 240 is one program, not two; a part's width
///     is a multiple of the rotation's group, or the engine declines.
package final class NeuralEngine: @unchecked Sendable {
    /// The ANE's rows come in this unit: an odd multiple of 128 costs about the next multiple of 256.
    package static let rowUnit = 256
    /// Products whose ANE rows went back to the GPU (a non-finite output, or a program refused).
    package private(set) var fallbacks = 0
    /// Products the ANE took a share of.
    package private(set) var products = 0

    private let directory: URL
    /// `int8`: the weight is the program's input as integers (`dequantize`), not fp16.
    private struct Shape: Hashable { let rows, k, n: Int; var int8 = false }
    /// A program with its operands, three IOSurfaces bound once: Core ML reads and writes them in place.
    /// Bound by pointer (`MLMultiArray(dataPointer:)`), it copied every operand into its own surfaces
    /// (+206 MB over four of Anima's shapes), released ~1 s after the program, and each call cost 1.3 to
    /// 2.6 × more (22/10/2026).
    private struct Binding {
        let model: MLModel, features: MLFeatureProvider, options: MLPredictionOptions
        let x, w, y: CVPixelBuffer
    }
    private var bindings: [Shape: Binding] = [:]
    private var refused: Set<Shape> = []

    /// Per column: `λ`, `1 / λ`, then the lift's partial maxima and L1 sums (`chunks` rows of `n` each).
    private var columns: Pages?
    /// Per chunk of rows, one row of `max(widest part of K, n)` floats: the input scaled, a weight row's
    /// magnitudes, an output part widened.
    private var scratch: Pages?
    /// Per row of the ANE's slice: `σ_i`, then `1 / σ_i`.
    private var rowLifts: Pages?
    /// The chunks the CPU's passes cut rows into, on every core.
    private static let chunks = 16
    /// `L`: the largest L1 norm of a lifted weight column, the output's bound per unit of input.
    private var liftedNorm: Float = 0
    /// Seconds per phase of the ANE's side, accumulated: what the pair's wall time is made of.
    package private(set) var timings: [String: Double] = [:]
    private func timed<T>(_ phase: String, _ body: () throws -> T) rethrows -> T {
        let started = DispatchTime.now().uptimeNanoseconds
        defer { timings[phase, default: 0] += Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9 }
        return try body()
    }

    /// **The ANE's limits on one program** (M1 Pro, macOS 27): beyond K = 8 192, or a weight of more than
    /// 3 · 2²⁴ values (100 MB in fp16), its rate falls from ~1.4 to 0.2–0.6 TFLOP/s (512 × 10 240 → 3 840: 156 ms,
    /// 0.26 TFLOP/s; two programs on 5 120: 2 × 15 ms, 22/10/2026). And AT those limits it is not yet fast: Krea 2's
    /// down projection in two parts of 8 192 × 6 144 ran at 0.45 TFLOP/s (2 048 rows in 930 ms); in parts of
    /// K ≤ 4 096 and ≤ 1.5 · 2²⁴ values (48 MB), 142 ms — and 6 144² 102 → 54 ms, 12 288 → 4 096 94 → 71,
    /// 10 240 → 3 840 76 → 64 (2 048 rows each, 23/10/2026). K is cut into the fewest parts within both, one
    /// program call each, their outputs summed in fp32.
    package static let maximumK = 4096, maximumWeight = 3 << 23
    /// **The int8 input's limits** (M1 Pro, 30/10/2026): K ≤ 4 096 as the fp16 input's, but the weight's bytes no longer bind — a
    /// 4 096 × 12 288 program (50 M values, 50 MB) ran 10.8–11.4 TFLOP/s where its fp16 twin (100 MB) ran 2.8. Past K = 4 096 it
    /// halves (4 608: 6.2) like the fp16 one.
    package static let maximumInt8Weight = 3 << 24

    /// The number of parts of K: the fewest whose widest (`⌈k / p⌉`) holds both limits. `int8`: the integers' limit on the
    /// weight. `alignment`: a part's width is a multiple of it when equal parts can give one (64; a rotation's group if more).
    package static func parts(k: Int, n: Int, int8: Bool = false, alignment: Int = 64) -> Int {
        let limit = int8 ? maximumInt8Weight : maximumWeight
        func fits(_ p: Int) -> Bool { let widest = (k + p - 1) / p; return widest <= maximumK && widest * n <= limit }
        var p = 1
        while !fits(p) { p += 1 }
        // Equal parts of a multiple of 64, when a few more parts give them: an uneven K (10 240 in three) is refused.
        let unit = max(64, alignment)
        if let even = (p...(p + 8)).first(where: { k % $0 == 0 && (k / $0) % unit == 0 }) { return even }
        return p
    }

    /// The program shapes of one product's parts (bounds `i · k / p`, invariant 4).
    private static func shapes(rows: Int, k: Int, n: Int, int8: Bool = false, alignment: Int = 64) -> [Shape] {
        let p = parts(k: k, n: n, int8: int8, alignment: alignment)
        return (0..<p).map { Shape(rows: rows, k: ($0 + 1) * k / p - $0 * k / p, n: n, int8: int8) }
    }

    /// **What it holds over a phase's `products`** (`MemoryNeed.gemm`), each given with the rows the ANE takes (the plan's): each
    /// program shape's three surfaces, kept while the engine lives — a sum over the shapes, not a maximum —, and its per-column
    /// state, per-chunk rows and per-row lifts, grown to the largest product. A product the ANE takes no row of holds nothing.
    ///
    /// **`integers`: the products whose weight the ANE reads as published** (`ProductAssignment.path == .native` with `ane > 0`),
    /// each with its weight's rotation group: the int8 program's shapes — a weight surface of one byte a value, one input
    /// surface of the whole K when it is one part, the rotated row of `k` floats a chunk. They are the plan's exact answers,
    /// not a bound over both inputs: the fp16 shapes are held only for the products that read the widened weight.
    package static func bytes(products: [(rows: Int, k: Int, n: Int)],
                              integers: [(rows: Int, k: Int, n: Int, group: Int?)] = []) -> Int {
        var shapes = Set<Shape>(), (columns, scratch, lifts) = (0, 0, 0)
        for p in products where p.rows > 0 {
            let these = NeuralEngine.shapes(rows: p.rows, k: p.k, n: p.n)
            shapes.formUnion(these)
            columns = max(columns, (2 + 2 * chunks) * p.n * 4)
            lifts = max(lifts, 2 * p.rows * 4)
            scratch = max(scratch, chunks * max(these.map(\.k).max()!, p.n) * 4)
        }
        for p in integers where p.rows > 0 {
            let these = NeuralEngine.shapes(rows: p.rows, k: p.k, n: p.n, int8: true, alignment: p.group ?? 64)
            shapes.formUnion(these)
            columns = max(columns, (2 + 2 * chunks) * p.n * 4)
            lifts = max(lifts, 2 * p.rows * 4)
            scratch = max(scratch, chunks * max(p.group != nil ? p.k : these.map(\.k).max()!, p.n) * 4)
        }
        return shapes.map { s in
            [s.rows * s.k * 2, s.k * s.n * (s.int8 ? 1 : 2), s.rows * s.n * 2].map { Arena.capacity(for: $0) }.reduce(0, +)
        }.reduce(0, +)
            + (columns > 0 ? Arena.capacity(for: columns) + Arena.capacity(for: scratch) + Arena.capacity(for: lifts) : 0)
    }

    /// **Core ML's first program, run once and dropped** before a render reads its base
    /// (`OperatorFootprint.bringUp`): what the framework keeps afterwards (~6 MB on this M1 Pro) is the
    /// process's from then on, read in its base rather than carried uncharged into a phase.
    /// The engine is kept (its three surfaces, 100 KB): released, it would wait for Core ML.
    package static let bringUp: NeuralEngine = autoreleasepool {
        let (rows, k, n) = (rowUnit, 64, 64)
        let a = Pages(bytes: rows * k * 4), w = Pages(bytes: k * n * 4), c = Pages(bytes: rows * n * 4)
        let engine = NeuralEngine()
        _ = engine.product(a: a.floats(), w: w.floats(), c: c.floats(), first: 0, rows: rows, k: k, n: n)
        return engine
    }

    package init() {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("siliconed-ane-\(ProcessInfo.processInfo.processIdentifier)-\(UUID().uuidString)", isDirectory: true)
    }

    /// **Released, it waits for Core ML to let go of its surfaces**: Core ML keeps a program and its
    /// operands ~0.5 s after its last prediction, then frees them on its own queue (0.51 s, measured 22/10/2026) — the
    /// phase after the engine's (the decoder) would find them held, above its need. The surfaces are
    /// looked up by identifier until none exists (`IOSurfaceLookup`: an object's existence, not the
    /// footprint), at most `releaseWait`.
    deinit {
        let identifiers = bindings.values.flatMap { [$0.x, $0.w, $0.y] }
            .compactMap { CVPixelBufferGetIOSurface($0)?.takeUnretainedValue() }.map(IOSurfaceGetID)
        bindings.removeAll()
        let started = DispatchTime.now().uptimeNanoseconds
        while identifiers.contains(where: { IOSurfaceLookup($0) != nil }),
              Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9 < NeuralEngine.releaseWait {
            usleep(2_000)
        }
        try? FileManager.default.removeItem(at: directory)
    }

    /// The longest a released engine waits for its surfaces; beyond it, the render's memory check reports
    /// what is left.
    package static let releaseWait = 2.0

    /// `false` once a program of the product's was refused (a surface Core ML cannot read dense, a shape
    /// it does not compile): the GPU then takes the whole product from the start, not after its share.
    package func accepts(rows: Int, k: Int, n: Int) -> Bool {
        !NeuralEngine.shapes(rows: rows, k: k, n: n).contains { refused.contains($0) }
    }

    /// The same for the int8 program (`int8Group`: the weight's convrot group, `nil` for none): no part of the product refused, and
    /// the cut of K one the rotation allows (`readsInt8`).
    package func accepts(rows: Int, k: Int, n: Int, int8Group group: Int?) -> Bool {
        guard NeuralEngine.readsInt8(k: k, n: n, group: group) else { return false }
        return !NeuralEngine.shapes(rows: rows, k: k, n: n, int8: true, alignment: group ?? 64).contains { refused.contains($0) }
    }

    /// **Whether the int8 program can read a `k × n` weight rotated by `group`** (`nil`: not rotated): K a whole number of groups, and
    /// every part's bound on a group's edge — the rotation is applied to the part's slice of the row, which only an uneven cut (no equal
    /// parts of a multiple of the group within reach) can break. Pure: what the solver asks of a class before it names `ane/int8`; a
    /// program Core ML refuses to load is `accepts`'s, learned at run.
    package static func readsInt8(k: Int, n: Int, group: Int?) -> Bool {
        guard let group else { return true }
        let p = parts(k: k, n: n, int8: true, alignment: group)
        return k % group == 0 && (0...p).allSatisfy { $0 * k / p % group == 0 }
    }

    /// **A published int8 weight as the ANE's program reads it**: `[k, n]` integers, row-major, and the scale — one per column
    /// (`perColumn`, `n` floats) or one for the tensor —, fp32, as `Artifact.stageInt8` leaves them in the reserve. `group`:
    /// the convrot rotation along K that the weight was published under (`W = R · q · s`), `nil` for none.
    package struct Int8Weight: @unchecked Sendable {
        package let values: UnsafePointer<Int8>
        package let scales: UnsafePointer<Float>
        package let perColumn: Bool
        package let group: Int?
        package init(values: UnsafePointer<Int8>, scales: UnsafePointer<Float>, perColumn: Bool, group: Int?) {
            self.values = values; self.scales = scales; self.perColumn = perColumn; self.group = group
        }
    }

    /// Products whose ANE rows read an int8 weight as it is, and were computed (not given back): what a check says it ran.
    package static let int8Tally = Mutex<Int>(0)

    private enum Weight {
        case widened(UnsafePointer<Float>)
        case int8(Int8Weight)
        /// The bf16 weight as `Artifact.stage` copied it, `[k, n]`.
        case bf16(UnsafePointer<UInt16>)
    }

    /// Products whose ANE rows read a bf16 weight as published (beside the GPU's rounded rows), and were computed: what a check says it ran.
    package static let bf16Tally = Mutex<Int>(0)

    /// **Rows `first ..< first + rows` of `C = A · W`** on the ANE, `W` fp32 `[k, n]`. `false` when the rows
    /// must be computed elsewhere: the program refused, or a non-finite output (`C`'s rows then undefined).
    /// `into`: where the rows land instead of `c`'s own (`rows · n` floats, the first row at `into`) — a product that is added to an
    /// output rather than written to it (a LoRA's, `GEMM.accumulatedLinear`) keeps the output's rows untouched.
    package func product(a: UnsafePointer<Float>, w: UnsafePointer<Float>, c: UnsafeMutablePointer<Float>,
                         first: Int, rows: Int, k: Int, n: Int, into: UnsafeMutablePointer<Float>? = nil) -> Bool {
        run(a: a, weight: .widened(w), c: c, first: first, rows: rows, k: k, n: n, into: into)
    }

    /// **The same product on a bf16 weight read as published** (`[k, n]` bf16 values, the reserve's copy): read and widened row by row
    /// into the lift and the fp16 conversion — the same fp16 operand as the widened weight's, nothing widened beside it.
    package func product(a: UnsafePointer<Float>, bf16 w: UnsafePointer<UInt16>, c: UnsafeMutablePointer<Float>,
                         first: Int, rows: Int, k: Int, n: Int) -> Bool {
        let ok = run(a: a, weight: .bf16(w), c: c, first: first, rows: rows, k: k, n: n)
        if ok { NeuralEngine.bf16Tally.withLock { $0 += 1 } }
        return ok
    }

    /// **The same product on a published int8 weight**, the integers the program's input (`Int8Weight`).
    package func product(a: UnsafePointer<Float>, int8 weight: Int8Weight, c: UnsafeMutablePointer<Float>,
                         first: Int, rows: Int, k: Int, n: Int) -> Bool {
        let ok = run(a: a, weight: .int8(weight), c: c, first: first, rows: rows, k: k, n: n)
        if ok { NeuralEngine.int8Tally.withLock { $0 += 1 } }
        return ok
    }

    private func run(a: UnsafePointer<Float>, weight: Weight, c: UnsafeMutablePointer<Float>,
                     first: Int, rows: Int, k: Int, n: Int, into: UnsafeMutablePointer<Float>? = nil) -> Bool {
        products += 1
        var int8: Int8Weight?
        if case let .int8(i) = weight { int8 = i }
        let group = int8?.group
        if let int8, !accepts(rows: rows, k: k, n: n, int8Group: int8.group) { fallbacks += 1; return false }
        let shapes = NeuralEngine.shapes(rows: rows, k: k, n: n, int8: int8 != nil, alignment: group ?? 64)
        var parts: [Binding] = []
        for shape in shapes {
            guard let binding = timed("compile", { self.binding(shape) }) else {
                fallbacks += 1
                return false
            }
            parts.append(binding)
        }
        let p = shapes.count
        let bounds = (0...p).map { $0 * k / p }
        let widest = shapes.map(\.k).max()!
        // A rotated row is rotated whole in a chunk's scratch (its peak sets σ): the scratch row holds K, not a part.
        let stride = max(group != nil ? k : widest, n)
        for (pages, bytes) in [(\NeuralEngine.columns, (2 + 2 * NeuralEngine.chunks) * n * 4),
                               (\.scratch, NeuralEngine.chunks * stride * 4)]
        where (self[keyPath: pages]?.capacity ?? 0) < bytes {
            self[keyPath: pages] = Pages(bytes: bytes)
        }
        nonisolated(unsafe) let rowScratch = scratch!.floats()
        let sideStarted = DispatchTime.now().uptimeNanoseconds
        defer { timings["side", default: 0] += Double(DispatchTime.now().uptimeNanoseconds - sideStarted) * 1e-9 }
        switch weight {
        case .widened(let w): timed("lift") { liftColumns(.widened(w), k: k, n: n, scratch: rowScratch, stride: stride) }
        case .bf16(let w): timed("lift") { liftColumns(.bf16(w), k: k, n: n, scratch: rowScratch, stride: stride) }
        case .int8(let i): timed("lift") { prepareColumns(i, k: k, n: n) }
        }

        // Each row of the activation, by the power of two its peak and the output's bound allow. By chunks
        // of rows on every core.
        if (rowLifts?.capacity ?? 0) < 2 * rows * 4 { rowLifts = Pages(bytes: 2 * rows * 4) }
        nonisolated(unsafe) let sigmas = rowLifts!.floats(), inverses = sigmas + rows
        let chunks = min(rows, NeuralEngine.chunks), per = (rows + chunks - 1) / chunks
        nonisolated(unsafe) let x = a + first * k
        let norm = liftedNorm
        // A rotated row meets its σ after the rotation. One part: rotated once, scaled and converted from the scratch row in the
        // part's own input stage (`fused`); several: rotated for its peak here, and again slice by slice at each part.
        let fused = group != nil && p == 1
        if !fused {
            let finiteInput = timed("input") { () -> Bool in
                // Where a chunk's scratch row is in use (a rotation), the flags live apart; else they are the scratch's own head.
                let own = group != nil
                let apart = own ? UnsafeMutablePointer<Float>.allocate(capacity: NeuralEngine.chunks) : nil
                defer { apart?.deallocate() }
                nonisolated(unsafe) let flags = apart ?? rowScratch   // 1 where a chunk is not finite, before the rows use it
                DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                    var bad: Float = 0
                    for row in chunk * per..<min(rows, (chunk + 1) * per) {
                        var m: Float = 0
                        if let group {
                            let rotated = rowScratch + chunk * stride
                            NeuralEngine.rotate(x + row * k, into: rotated, count: k, group: group)
                            vDSP_maxmgv(rotated, 1, &m, vDSP_Length(k))
                        } else {
                            vDSP_maxmgv(x + row * k, 1, &m, vDSP_Length(k))
                        }
                        let bound = m * norm
                        if !m.isFinite { bad = 1 }
                        let sigma: Float = bound > 0 && bound.isFinite ? exp2(floor(log2(32_768 / bound))) : 1
                        (sigmas[row], inverses[row]) = (sigma, 1 / sigma)
                    }
                    flags[chunk] = bad
                }
                return !(0..<chunks).contains { flags[$0] != 0 }
            }
            guard finiteInput else { fallbacks += 1; return false }
        }
        nonisolated(unsafe) let y = into ?? c + first * n

        // Part by part: its weight rows and input columns into the surfaces, the program, its output added
        // into C's rows in fp32.
        for (i, binding) in parts.enumerated() {
            let (k0, count) = (bounds[i], bounds[i + 1] - bounds[i])
            CVPixelBufferLockBaseAddress(binding.w, [])
            CVPixelBufferLockBaseAddress(binding.x, [])
            nonisolated(unsafe) let half = CVPixelBufferGetBaseAddress(binding.x)!.assumingMemoryBound(to: Float16.self)
            timed("weight") {
                switch weight {
                case .widened(let w):
                    convertWeight(w + k0 * n, rows: count, n: n,
                                  into: CVPixelBufferGetBaseAddress(binding.w)!.assumingMemoryBound(to: Float16.self))
                case .bf16(let w):
                    convertWeight(bf16: w + k0 * n, rows: count, n: n,
                                  into: CVPixelBufferGetBaseAddress(binding.w)!.assumingMemoryBound(to: Float16.self))
                case .int8(let q):
                    copyIntegers(q.values + k0 * n, count: count * n, into: CVPixelBufferGetBaseAddress(binding.w)!)
                }
            }
            var finiteFused = true
            timed("input") {
                // Scratch rows by chunk; a fused pass keeps its flags apart (the scratch rows are in use).
                let apart = fused ? UnsafeMutablePointer<Float>.allocate(capacity: NeuralEngine.chunks) : nil
                defer { apart?.deallocate() }
                nonisolated(unsafe) let flags = apart
                DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                    let (lower, upper) = (chunk * per, min(rows, (chunk + 1) * per))
                    // Scaled in fp32 (exact: a power of two, in range), then converted, row by row.
                    let scaled = rowScratch + chunk * stride
                    var bad: Float = 0
                    for row in lower..<upper {
                        var s: Float
                        let source: UnsafePointer<Float>
                        if let group {
                            // The rotated slice, in the scratch row: this part's columns of the row, as the fused pass's whole.
                            NeuralEngine.rotate(x + row * k + k0, into: scaled, count: count, group: group)
                            source = UnsafePointer(scaled)
                            if fused {
                                var m: Float = 0
                                vDSP_maxmgv(scaled, 1, &m, vDSP_Length(count))
                                let bound = m * norm
                                if !m.isFinite { bad = 1 }
                                let sigma: Float = bound > 0 && bound.isFinite ? exp2(floor(log2(32_768 / bound))) : 1
                                (sigmas[row], inverses[row]) = (sigma, 1 / sigma)
                            }
                            s = sigmas[row]
                            vDSP_vsmul(scaled, 1, &s, scaled, 1, vDSP_Length(count))
                        } else {
                            s = sigmas[row]
                            vDSP_vsmul(x + row * k + k0, 1, &s, scaled, 1, vDSP_Length(count))
                            source = UnsafePointer(scaled)
                        }
                        var from = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: source), height: 1, width: vImagePixelCount(count),
                                                 rowBytes: count * 4)
                        var target = vImage_Buffer(data: half + row * count, height: 1, width: vImagePixelCount(count),
                                                   rowBytes: count * 2)
                        vImageConvert_PlanarFtoPlanar16F(&from, &target, 0)
                    }
                    flags?[chunk] = bad
                }
                if let flags { finiteFused = !(0..<chunks).contains { flags[$0] != 0 } }
            }
            CVPixelBufferUnlockBaseAddress(binding.x, [])
            CVPixelBufferUnlockBaseAddress(binding.w, [])
            if !finiteFused { fallbacks += 1; return false }
            do {
                try timed("predict") { _ = try binding.model.prediction(from: binding.features, options: binding.options) }
            } catch {
                fallbacks += 1
                return false
            }
            CVPixelBufferLockBaseAddress(binding.y, .readOnly)
            nonisolated(unsafe) let out = CVPixelBufferGetBaseAddress(binding.y)!.assumingMemoryBound(to: Float16.self)
            timed("output") {
                DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                    let (lower, upper) = (chunk * per, min(rows, (chunk + 1) * per))
                    guard upper > lower else { return }
                    if i == 0 {
                        let count = (upper - lower) * n
                        var source = vImage_Buffer(data: out + lower * n, height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
                        var target = vImage_Buffer(data: y + lower * n, height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
                        vImageConvert_Planar16FtoPlanarF(&source, &target, 0)
                    } else {
                        let part = rowScratch + chunk * stride
                        for row in lower..<upper {
                            var source = vImage_Buffer(data: out + row * n, height: 1, width: vImagePixelCount(n), rowBytes: n * 2)
                            var target = vImage_Buffer(data: part, height: 1, width: vImagePixelCount(n), rowBytes: n * 4)
                            vImageConvert_Planar16FtoPlanarF(&source, &target, 0)
                            vDSP_vadd(y + row * n, 1, part, 1, y + row * n, 1, vDSP_Length(n))
                        }
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(binding.y, .readOnly)
        }

        // Each column divided by its lift (or multiplied by its scale) and each row by its σ, checked finite.
        let finite = timed("output") { () -> Bool in
            nonisolated(unsafe) let u = columns!.floats() + n
            nonisolated(unsafe) let flags = rowScratch   // 1 where a chunk is not finite
            DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                let (lower, upper) = (chunk * per, min(rows, (chunk + 1) * per))
                var m: Float = 0
                for row in lower..<upper {
                    let r = y + row * n
                    vDSP_vmul(r, 1, u, 1, r, 1, vDSP_Length(n))
                    if inverses[row] != 1 { var s = inverses[row]; vDSP_vsmul(r, 1, &s, r, 1, vDSP_Length(n)) }
                }
                if upper > lower { vDSP_maxmgv(y + lower * n, 1, &m, vDSP_Length((upper - lower) * n)) }
                flags[chunk] = m.isFinite ? 0 : 1
            }
            return !(0..<chunks).contains { flags[$0] != 0 }
        }
        guard finite else { fallbacks += 1; return false }
        return true
    }

    // MARK: - the weight

    /// `λ_j = 2^−⌊log₂ max_i |W_ij|⌋` per column, and its inverse; 1 for a column of zeros. And `L`, from
    /// each column's L1 norm, in the same pass. A bf16 weight's row is widened into the chunk's scratch first, then read as an fp32 one:
    /// the same values, the same operations, the same λ and L.
    private func liftColumns(_ weight: Weight, k: Int, n: Int, scratch rowScratch: UnsafeMutablePointer<Float>, stride: Int) {
        let chunks = min(k, NeuralEngine.chunks), per = (k + chunks - 1) / chunks
        let lift = columns!.floats(), unlift = lift + n
        nonisolated(unsafe) let p = lift + 2 * n, sums = p + chunks * n
        nonisolated(unsafe) let (rowScratch, weight) = (rowScratch, weight)
        DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
            let (m, s, magnitude) = (p + chunk * n, sums + chunk * n, rowScratch + chunk * stride)
            m.update(repeating: 0, count: n)
            s.update(repeating: 0, count: n)
            for row in min(k, chunk * per)..<min(k, (chunk + 1) * per) {
                let values: UnsafePointer<Float>
                switch weight {
                case .widened(let w): values = w + row * n
                case .bf16(let w):
                    Widen.bfloat16ToFloat32(source: w + row * n, destination: magnitude, count: n)
                    values = UnsafePointer(magnitude)
                case .int8: return
                }
                vDSP_vmaxmg(m, 1, values, 1, m, 1, vDSP_Length(n))
                vDSP_vabs(values, 1, magnitude, 1, vDSP_Length(n))
                vDSP_vadd(s, 1, magnitude, 1, s, 1, vDSP_Length(n))
            }
        }
        for chunk in 1..<chunks {
            vDSP_vmax(p, 1, p + chunk * n, 1, p, 1, vDSP_Length(n))
            vDSP_vadd(sums, 1, sums + chunk * n, 1, sums, 1, vDSP_Length(n))
        }
        liftedNorm = 0
        for j in 0..<n {
            let peak = p[j]
            let l: Float = peak > 0 && peak.isFinite ? exp2(-floor(log2(peak))) : 1
            lift[j] = l; unlift[j] = 1 / l
            liftedNorm = max(liftedNorm, sums[j] * l)
        }
    }

    /// **An int8 weight's column state**, in the place `liftColumns` leaves the fp16 weight's: the host's column factor
    /// `u_j = 64 · s_j · (1 / √G)` after the program — 64 undoes the program's 2⁻⁶, `s_j` the weight's scale (one per column or
    /// for the tensor), `1 / √G` the rotation's normalization, the rotation having been left unnormalized — and `L`, the largest
    /// column's L1 norm of `q / 64` (`liftedNorm`). The sums are integers (`UInt16` lanes, flushed every 256 rows: 128 · 256 fits)
    /// over the cores, exact; `L` is read once per call, a pass over the integers that costs what their copy does.
    private func prepareColumns(_ weight: Int8Weight, k: Int, n: Int) {
        let chunks = min(k, NeuralEngine.chunks), per = (k + chunks - 1) / chunks
        let u = columns!.floats() + n
        nonisolated(unsafe) let p = columns!.floats() + 2 * n, sums = p + NeuralEngine.chunks * n
        nonisolated(unsafe) let q = weight.values
        DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
            let narrow = UnsafeMutableRawPointer(p + chunk * n).assumingMemoryBound(to: UInt16.self)
            let wide = UnsafeMutableRawPointer(sums + chunk * n).assumingMemoryBound(to: UInt32.self)
            narrow.update(repeating: 0, count: n)
            wide.update(repeating: 0, count: n)
            var pending = 0
            for row in min(k, chunk * per)..<min(k, (chunk + 1) * per) {
                let r = q + row * n
                var j = 0
                while j + 16 <= n {
                    let v = UnsafeRawPointer(r + j).loadUnaligned(as: SIMD16<Int8>.self)
                    let sign = v &>> 7
                    let magnitude = SIMD16<UInt16>(truncatingIfNeeded: SIMD16<UInt8>(truncatingIfNeeded: (v ^ sign) &- sign))
                    let at = UnsafeMutableRawPointer(narrow + j)
                    at.storeBytes(of: at.loadUnaligned(as: SIMD16<UInt16>.self) &+ magnitude, as: SIMD16<UInt16>.self)
                    j += 16
                }
                while j < n { narrow[j] &+= UInt16(r[j].magnitude); j += 1 }
                pending += 1
                if pending == 256 {
                    for j in 0..<n { wide[j] &+= UInt32(narrow[j]); narrow[j] = 0 }
                    pending = 0
                }
            }
            for j in 0..<n { wide[j] &+= UInt32(narrow[j]) }
        }
        var largest: UInt32 = 0
        for j in 0..<n {
            var total: UInt32 = 0
            for chunk in 0..<chunks { total &+= UnsafeMutableRawPointer(sums + chunk * n).assumingMemoryBound(to: UInt32.self)[j] }
            largest = max(largest, total)
        }
        liftedNorm = Float(largest) / 64
        let normalization: Float = weight.group.map { 1 / Float($0).squareRoot() } ?? 1
        for j in 0..<n { u[j] = 64 * normalization * weight.scales[weight.perColumn ? j : 0] }
    }

    /// The integers into a surface, on every core: 0.87 ms for 3 840 × 10 240 on an M1 Pro, where the fp16 route's
    /// widening of the same weight cost 2.52.
    private func copyIntegers(_ q: UnsafePointer<Int8>, count: Int, into target: UnsafeMutableRawPointer) {
        let pieces = 32, per = (count + pieces - 1) / pieces
        nonisolated(unsafe) let (q, target) = (q, target)
        DispatchQueue.concurrentPerform(iterations: pieces) { piece in
            let lower = min(count, piece * per), upper = min(count, (piece + 1) * per)
            if upper > lower { (target + lower).copyMemory(from: q + lower, byteCount: upper - lower) }
        }
    }

    /// **`x · R` along `count` values, unnormalized** (`R = H / √G`, `H` the Kronecker power of H4 that `Widen.dequantizeRotated`
    /// documents), by groups of `group` (a power of 4, 4 to 256): the four stages of 4-point butterflies, `x` read once and `y` written.
    /// The `1 / √G` is a power of two and goes into the column factor (`prepareColumns`). The same sums as the Metal
    /// `rotate_groups` and the check's `exactRows`, in the same stage order.
    package static func rotate(_ x: UnsafePointer<Float>, into y: UnsafeMutablePointer<Float>, count: Int, group: Int) {
        precondition(group >= 4 && count % group == 0, "rotation: group \(group) on \(count) values")
        var i = 0
        while i < count {
            let a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3]
            let p = a + b, m = a - b, u = c + d, w = c - d
            y[i] = p + w; y[i + 1] = p - w; y[i + 2] = m + u; y[i + 3] = u - m
            i += 4
        }
        var span = 4
        while span < group {
            var base = 0
            while base < count {
                var j = base
                while j < base + span {
                    let at = UnsafeMutableRawPointer(y + j), step = span * MemoryLayout<Float>.size
                    let a = at.loadUnaligned(as: SIMD4<Float>.self), b = (at + step).loadUnaligned(as: SIMD4<Float>.self)
                    let c = (at + 2 * step).loadUnaligned(as: SIMD4<Float>.self), d = (at + 3 * step).loadUnaligned(as: SIMD4<Float>.self)
                    let p = a + b, m = a - b, u = c + d, w = c - d
                    at.storeBytes(of: p + w, as: SIMD4<Float>.self)
                    (at + step).storeBytes(of: p - w, as: SIMD4<Float>.self)
                    (at + 2 * step).storeBytes(of: m + u, as: SIMD4<Float>.self)
                    (at + 3 * step).storeBytes(of: u - m, as: SIMD4<Float>.self)
                    j += 4
                }
                base += 4 * span
            }
            span *= 4
        }
    }

    /// `W · λ` in fp16 for `rows` rows of the weight, on every core: 0.8 ms for 2 048 × 8 192 on an M1 Pro.
    private func convertWeight(_ w: UnsafePointer<Float>, rows: Int, n: Int, into target: UnsafeMutablePointer<Float16>) {
        let chunks = 32, per = (rows + chunks - 1) / chunks
        nonisolated(unsafe) let (w, target, l) = (w, target, columns!.floats())
        DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
            for row in min(rows, chunk * per)..<min(rows, (chunk + 1) * per) {
                let s = w + row * n, t = target + row * n
                for j in 0..<n { t[j] = Float16(s[j] * l[j]) }
            }
        }
    }

    /// The same from a bf16 weight's rows, each value widened (a shift) before its lift: the fp16 values of the widened weight's.
    private func convertWeight(bf16 w: UnsafePointer<UInt16>, rows: Int, n: Int, into target: UnsafeMutablePointer<Float16>) {
        let chunks = 32, per = (rows + chunks - 1) / chunks
        nonisolated(unsafe) let (w, target, l) = (w, target, columns!.floats())
        DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
            for row in min(rows, chunk * per)..<min(rows, (chunk + 1) * per) {
                let s = w + row * n, t = target + row * n
                for j in 0..<n { t[j] = Float16(Float(bitPattern: UInt32(s[j]) << 16) * l[j]) }
            }
        }
    }

    /// One fp16 surface of `rows × columns`, unpadded (the program reads it dense); `nil` otherwise. `int8`: one byte a value
    /// (`OneComponent8`, macOS 26).
    private static func surface(rows: Int, columns: Int, int8: Bool = false) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, columns, rows,
                                  int8 ? kCVPixelFormatType_OneComponent8 : kCVPixelFormatType_OneComponent16Half, attributes,
                                  &buffer) == kCVReturnSuccess,
              let buffer, CVPixelBufferGetBytesPerRow(buffer) == columns * (int8 ? 1 : 2) else { return nil }
        return buffer
    }

    /// The shape's program bound to its own three surfaces, made once.
    private func binding(_ shape: Shape) -> Binding? {
        if let binding = bindings[shape] { return binding }
        if refused.contains(shape) { return nil }
        guard let model = program(shape),
              let x = NeuralEngine.surface(rows: shape.rows, columns: shape.k),
              let w = NeuralEngine.surface(rows: shape.k, columns: shape.n, int8: shape.int8),
              let y = NeuralEngine.surface(rows: shape.rows, columns: shape.n) else {
            refused.insert(shape)
            return nil
        }
        let dims = { (values: [Int]) in values.map { NSNumber(value: $0) } }
        do {
            let options = MLPredictionOptions()
            options.outputBackings = ["y": MLMultiArray(pixelBuffer: y, shape: dims([shape.rows, shape.n]))]
            let features = try MLDictionaryFeatureProvider(dictionary: [
                "x": MLFeatureValue(multiArray: MLMultiArray(pixelBuffer: x, shape: dims([shape.rows, shape.k]))),
                (shape.int8 ? "wq" : "w"): MLFeatureValue(multiArray: MLMultiArray(pixelBuffer: w, shape: dims([shape.k, shape.n]))),
            ])
            let binding = Binding(model: model, features: features, options: options, x: x, w: w, y: y)
            bindings[shape] = binding
            return binding
        } catch {
            refused.insert(shape)
            return nil
        }
    }

    // MARK: - the programs

    /// The shape's program, written then loaded by Core ML (its binding keeps it).
    private func program(_ shape: Shape) -> MLModel? {
        do {
            let url = directory.appendingPathComponent("\(shape.rows)x\(shape.k)x\(shape.n)\(shape.int8 ? "i8" : "").mlmodelc",
                                                       isDirectory: true)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            try NeuralEngine.programText(rows: shape.rows, k: shape.k, n: shape.n, int8: shape.int8)
                .write(to: url.appendingPathComponent("model.mil"), atomically: true, encoding: .utf8)
            try NeuralEngine.description(rows: shape.rows, k: shape.k, n: shape.n, int8: shape.int8)
                .write(to: url.appendingPathComponent("coremldata.bin"))
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            // A program Core ML refuses may take the process down rather than return an error: the measure says what it is about to load.
            RouteBeacon.announce(shape.int8 ? "ane/int8" : "ane/fp16")
            return try MLModel(contentsOf: url, configuration: configuration)
        } catch {
            return nil
        }
    }

    /// **The program**, MIL text as `coremltools` writes it for `matmul(x, w)`: fp16 `x [rows, k]` and
    /// `w [k, n]` in, fp16 `y [rows, n]` out.
    ///
    /// `int8`: the weight is `tensor<int8, [k, n]> wq`, an input too, and the program dequantizes it with MIL's `dequantize` and a
    /// constant scalar scale 2⁻⁶ before the product (measured 9.4–11.1 TFLOP/s on an M1 Pro, 30/10/2026; the per-column scale cannot be an input of
    /// `dequantize`, and `cast` + `mul` runs at 4.6–6.3). An invalid program does not return an error: it kills the process, so
    /// only this text, which was run on that Mac, is written.
    package static func programText(rows: Int, k: Int, n: Int, int8: Bool = false) -> String {
        if int8 {
            return """
            program(1.3)
            [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3600.16.1"}, {"coremlc-version", "3600.25.2"}})]
            {
                func main<ios18>(tensor<int8, [\(k), \(n)]> wq, tensor<fp16, [\(rows), \(k)]> x) {
                        fp16 sc = const()[name = string("sc"), val = fp16(0.015625)];
                        tensor<fp16, [\(k), \(n)]> w = dequantize(input = wq, scale = sc)[name = string("w")];
                        bool tx = const()[name = string("tx"), val = bool(false)];
                        bool ty = const()[name = string("ty"), val = bool(false)];
                        tensor<fp16, [\(rows), \(n)]> y = matmul(transpose_x = tx, transpose_y = ty, x = x, y = w)[name = string("y")];
                    } -> (y);
            }
            """
        }
        return """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3600.16.1"}, {"coremlc-version", "3600.25.2"}})]
        {
            func main<ios18>(tensor<fp16, [\(k), \(n)]> w, tensor<fp16, [\(rows), \(k)]> x) {
                    bool y_transpose_x_0 = const()[name = string("y_transpose_x_0"), val = bool(false)];
                    bool y_transpose_y_0 = const()[name = string("y_transpose_y_0"), val = bool(false)];
                    tensor<fp16, [\(rows), \(n)]> y = matmul(transpose_x = y_transpose_x_0, transpose_y = y_transpose_y_0, x = x, y = w)[name = string("y")];
                } -> (y);
        }
        """
    }

    /// **`coremldata.bin`**: a fixed header carrying the description's length (a little-endian `UInt64`
    /// at 0x4b), the model description (protobuf: inputs `x`, `w`, output `y`, their shapes and fp16),
    /// then a fixed trailer — byte for byte what `coremltools` 9 compiles for the same program
    /// (`NeuralEngineTests`).
    ///
    /// `int8`: the weight `wq` is `ArrayDataType.INT8` (131 080), first of the inputs as in the program's signature.
    package static func description(rows: Int, k: Int, n: Int, int8: Bool = false) -> Data {
        func varint(_ value: Int) -> [UInt8] {
            var v = UInt64(value), bytes: [UInt8] = []
            repeat {
                let low = UInt8(v & 0x7f)
                v >>= 7
                bytes.append(v == 0 ? low : low | 0x80)
            } while v != 0
            return bytes
        }
        func field(_ tag: UInt8, _ body: [UInt8]) -> [UInt8] { [tag] + varint(body.count) + body }
        /// A feature: its name, a multi-array of `dims` in fp16 (`ArrayDataType.FLOAT16` = 65 552), or of another type.
        func feature(_ tag: UInt8, _ name: String, _ dims: [Int], type: Int = 65_552) -> [UInt8] {
            let shape = field(0x0a, dims.flatMap(varint))
            let array = field(0x2a, shape + [0x10] + varint(type))
            return field(tag, field(0x0a, Array(name.utf8)) + field(0x1a, array))
        }
        let payload = (int8 ? feature(0x0a, "wq", [k, n], type: 131_080) + feature(0x0a, "x", [rows, k])
                            : feature(0x0a, "x", [rows, k]) + feature(0x0a, "w", [k, n]))
            + feature(0x52, "y", [rows, n]) + [0xa2, 0x06, 0x00]
        var header = [UInt8](repeating: 0, count: 0x53)
        header[0] = 0xf6; header[1] = 0x01; header[4] = 0x09; header[0x1c] = 0x07
        for (i, byte) in "generic".utf8.enumerated() { header[0x24 + i] = byte }
        header[0x2b] = 0x09
        withUnsafeBytes(of: UInt64(payload.count).littleEndian) { for (i, byte) in $0.enumerated() { header[0x4b + i] = byte } }
        var trailer = [UInt8](repeating: 0, count: 16)
        trailer[0] = 0xf6; trailer[1] = 0x01
        return Data(header + payload + trailer)
    }
}
