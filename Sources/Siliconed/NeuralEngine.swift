import Accelerate
import CoreML
import CoreVideo
import Foundation

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
/// A branch the capacity map turns on (`gemm_ane`, `CapacityRules`), where it measures the pair faster
/// than the GPU alone and the ANE's rows within the half-precision threshold; off by default, like
/// every compute branch. A render may still refuse it (`Request.neuralEngine`: the app's default on a
/// Standard version), and only the denoisers take it — never an encoder.
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
///     frozen (`ane_fraction`): the same seed gives the same bits.
package final class NeuralEngine: @unchecked Sendable {
    /// The ANE's rows come in this unit: an odd multiple of 128 costs about the next multiple of 256.
    package static let rowUnit = 256
    /// Below this height the GPU alone (`ane_min`, 1024 by default).
    package static let minimumRows = EngineSettings.effective.aneMinimumRows

    /// The GPU's share of the rows; the ANE takes the rest, rounded to `rowUnit`. Frozen in a render
    /// (`ane_fraction`); the map moves it to find the share that balances both sides.
    package var gpuShare: Double
    /// The GPU's share per weight shape and height (`ane_shares`, measured by the map on the cuts' grid):
    /// with a grid, it replaces `gpuShare`.
    package var shares: Conductor.Shares
    /// **The shapes where the AMX works beside the ANE** (`ane_amx`, ρ = 1 at the points the map gave the three
    /// engines, 0 elsewhere): there the GPU's side of the rows goes through the conductor, GPU + AMX, instead of
    /// the GPU alone. Engines are throughputs that add, not alternatives: the map says where they do.
    package var besideAMX = Conductor.Shares.none
    /// Whether the AMX takes part of the GPU's side of `m × k → n` (`besideAMX` read at the product, ≥ ½).
    package func takesAMX(_ m: Int, k: Int, n: Int) -> Bool { (besideAMX.ratio(k: k, n: n, m: m) ?? 0) >= 0.5 }
    /// The shares a grid may give: from the GPU alone (1) down to a twentieth of the rows.
    package static let shareRange = 0.05...1.0
    /// Products whose ANE rows went back to the GPU (a non-finite output, or a program refused).
    package private(set) var fallbacks = 0
    /// Products the ANE took a share of.
    package private(set) var products = 0

    private let directory: URL
    private struct Shape: Hashable { let rows, k, n: Int }
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

    /// The number of parts of K: the fewest whose widest (`⌈k / p⌉`) holds both limits.
    package static func parts(k: Int, n: Int) -> Int {
        func fits(_ p: Int) -> Bool { let widest = (k + p - 1) / p; return widest <= maximumK && widest * n <= maximumWeight }
        var p = 1
        while !fits(p) { p += 1 }
        // Equal parts of a multiple of 64, when a few more parts give them: an uneven K (10 240 in three) is refused.
        if let even = (p...(p + 8)).first(where: { k % $0 == 0 && (k / $0) % 64 == 0 }) { return even }
        return p
    }

    /// The program shapes of one product's parts (bounds `i · k / p`, invariant 4).
    private static func shapes(rows: Int, k: Int, n: Int) -> [Shape] {
        let p = parts(k: k, n: n)
        return (0..<p).map { Shape(rows: rows, k: ($0 + 1) * k / p - $0 * k / p, n: n) }
    }

    /// **What it holds over a phase's `products`** (`MemoryNeed.gemm`): each program shape's three
    /// surfaces, kept while the engine lives — a sum over the shapes, not a maximum —, and its per-column
    /// state, per-chunk rows and per-row lifts, grown to the largest product.
    package static func bytes(products: [(m: Int, k: Int, n: Int)], gpuShare: Double, shares: Conductor.Shares = .none) -> Int {
        let engine = NeuralEngine(gpuShare: gpuShare, shares: shares)
        var shapes = Set<Shape>(), (columns, scratch, lifts) = (0, 0, 0)
        for p in products {
            let rows = engine.rows(p.m, k: p.k, n: p.n)
            guard rows > 0 else { continue }
            let these = NeuralEngine.shapes(rows: rows, k: p.k, n: p.n)
            shapes.formUnion(these)
            columns = max(columns, (2 + 2 * chunks) * p.n * 4)
            lifts = max(lifts, 2 * rows * 4)
            scratch = max(scratch, chunks * max(these.map(\.k).max()!, p.n) * 4)
        }
        return shapes.map { s in [s.rows * s.k, s.k * s.n, s.rows * s.n].map { Arena.capacity(for: $0 * 2) }.reduce(0, +) }.reduce(0, +)
            + (columns > 0 ? Arena.capacity(for: columns) + Arena.capacity(for: scratch) + Arena.capacity(for: lifts) : 0)
    }

    /// **Core ML's first program, run once and dropped** before a render reads its base
    /// (`OperatorFootprint.bringUp`): what the framework keeps afterwards (~6 MB on this M1 Pro) is the
    /// process's from then on, read in its base rather than carried uncharged into a phase.
    /// The engine is kept (its three surfaces, 100 KB): released, it would wait for Core ML.
    package static let bringUp: NeuralEngine = autoreleasepool {
        let (rows, k, n) = (rowUnit, 64, 64)
        let a = Pages(bytes: rows * k * 4), w = Pages(bytes: k * n * 4), c = Pages(bytes: rows * n * 4)
        let engine = NeuralEngine(gpuShare: 0.5)
        _ = engine.product(a: a.floats(), w: w.floats(), c: c.floats(), first: 0, rows: rows, k: k, n: n)
        return engine
    }

    package init(gpuShare: Double, shares: Conductor.Shares = .none) {
        self.gpuShare = gpuShare; self.shares = shares
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

    /// The ANE's rows of an `m`-row product `k → n`: `0` when the GPU keeps them all. The share is the grid's
    /// at `(k, n, m)` when there is one, `gpuShare` otherwise.
    package func rows(_ m: Int, k: Int, n: Int) -> Int {
        NeuralEngine.rows(m, gpuShare: shares.share(k: k, n: n, m: m, range: NeuralEngine.shareRange) ?? gpuShare,
                          minimum: NeuralEngine.minimumRows)
    }

    /// The same from a share already read: what the route table (`RouteTable.Products.route`) asks at each product.
    package static func rows(_ m: Int, gpuShare share: Double, minimum: Int) -> Int {
        guard m >= minimum else { return 0 }
        let unit = Double(rowUnit)
        let r = Int((Double(m) * (1 - share) / unit).rounded()) * rowUnit
        return r > 0 && r < m ? r : 0
    }

    /// `false` once a program of the product's was refused (a surface Core ML cannot read dense, a shape
    /// it does not compile): the GPU then takes the whole product from the start, not after its share.
    package func accepts(rows: Int, k: Int, n: Int) -> Bool {
        !NeuralEngine.shapes(rows: rows, k: k, n: n).contains { refused.contains($0) }
    }

    /// **Rows `first ..< first + rows` of `C = A · W`** on the ANE, `W` fp32 `[k, n]`. `false` when the rows
    /// must be computed elsewhere: the program refused, or a non-finite output (`C`'s rows then undefined).
    package func product(a: UnsafePointer<Float>, w: UnsafePointer<Float>, c: UnsafeMutablePointer<Float>,
                         first: Int, rows: Int, k: Int, n: Int) -> Bool {
        products += 1
        let shapes = NeuralEngine.shapes(rows: rows, k: k, n: n)
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
        for (pages, bytes) in [(\NeuralEngine.columns, (2 + 2 * NeuralEngine.chunks) * n * 4),
                               (\.scratch, NeuralEngine.chunks * max(widest, n) * 4)]
        where (self[keyPath: pages]?.capacity ?? 0) < bytes {
            self[keyPath: pages] = Pages(bytes: bytes)
        }
        let stride = max(widest, n)
        nonisolated(unsafe) let rowScratch = scratch!.floats()
        let sideStarted = DispatchTime.now().uptimeNanoseconds
        defer { timings["side", default: 0] += Double(DispatchTime.now().uptimeNanoseconds - sideStarted) * 1e-9 }
        timed("lift") { liftColumns(w, k: k, n: n, scratch: rowScratch, stride: stride) }

        // Each row of the activation, by the power of two its peak and the output's bound allow. By chunks
        // of rows on every core.
        if (rowLifts?.capacity ?? 0) < 2 * rows * 4 { rowLifts = Pages(bytes: 2 * rows * 4) }
        nonisolated(unsafe) let sigmas = rowLifts!.floats(), inverses = sigmas + rows
        let chunks = min(rows, NeuralEngine.chunks), per = (rows + chunks - 1) / chunks
        nonisolated(unsafe) let x = a + first * k
        let norm = liftedNorm
        let finiteInput = timed("input") { () -> Bool in
            nonisolated(unsafe) let flags = rowScratch   // 1 where a chunk is not finite, before the rows use it
            DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                var bad: Float = 0
                for row in chunk * per..<min(rows, (chunk + 1) * per) {
                    var m: Float = 0
                    vDSP_maxmgv(x + row * k, 1, &m, vDSP_Length(k))
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
        nonisolated(unsafe) let y = c + first * n

        // Part by part: its weight rows and input columns into the surfaces, the program, its output added
        // into C's rows in fp32.
        for (i, binding) in parts.enumerated() {
            let (k0, count) = (bounds[i], bounds[i + 1] - bounds[i])
            CVPixelBufferLockBaseAddress(binding.w, [])
            CVPixelBufferLockBaseAddress(binding.x, [])
            nonisolated(unsafe) let half = CVPixelBufferGetBaseAddress(binding.x)!.assumingMemoryBound(to: Float16.self)
            timed("weight") {
                convertWeight(w + k0 * n, rows: count, n: n,
                              into: CVPixelBufferGetBaseAddress(binding.w)!.assumingMemoryBound(to: Float16.self))
            }
            timed("input") {
                DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                    let (lower, upper) = (chunk * per, min(rows, (chunk + 1) * per))
                    // Scaled in fp32 (exact: a power of two, in range), then converted, row by row.
                    let scaled = rowScratch + chunk * stride
                    for row in lower..<upper {
                        var s = sigmas[row]
                        vDSP_vsmul(x + row * k + k0, 1, &s, scaled, 1, vDSP_Length(count))
                        var source = vImage_Buffer(data: scaled, height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
                        var target = vImage_Buffer(data: half + row * count, height: 1, width: vImagePixelCount(count),
                                                   rowBytes: count * 2)
                        vImageConvert_PlanarFtoPlanar16F(&source, &target, 0)
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(binding.x, [])
            CVPixelBufferUnlockBaseAddress(binding.w, [])
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

        // Each column divided by its lift and each row by its σ, checked finite.
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
    /// each column's L1 norm, in the same pass.
    private func liftColumns(_ w: UnsafePointer<Float>, k: Int, n: Int, scratch rowScratch: UnsafeMutablePointer<Float>, stride: Int) {
        let chunks = min(k, NeuralEngine.chunks), per = (k + chunks - 1) / chunks
        let lift = columns!.floats(), unlift = lift + n
        nonisolated(unsafe) let p = lift + 2 * n, sums = p + chunks * n
        nonisolated(unsafe) let (rowScratch, w) = (rowScratch, w)
        DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
            let (m, s, magnitude) = (p + chunk * n, sums + chunk * n, rowScratch + chunk * stride)
            m.update(repeating: 0, count: n)
            s.update(repeating: 0, count: n)
            for row in min(k, chunk * per)..<min(k, (chunk + 1) * per) {
                vDSP_vmaxmg(m, 1, w + row * n, 1, m, 1, vDSP_Length(n))
                vDSP_vabs(w + row * n, 1, magnitude, 1, vDSP_Length(n))
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

    /// One fp16 surface of `rows × columns`, unpadded (the program reads it dense); `nil` otherwise.
    private static func surface(rows: Int, columns: Int) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, columns, rows, kCVPixelFormatType_OneComponent16Half, attributes,
                                  &buffer) == kCVReturnSuccess,
              let buffer, CVPixelBufferGetBytesPerRow(buffer) == columns * 2 else { return nil }
        return buffer
    }

    /// The shape's program bound to its own three surfaces, made once.
    private func binding(_ shape: Shape) -> Binding? {
        if let binding = bindings[shape] { return binding }
        if refused.contains(shape) { return nil }
        guard let model = program(shape),
              let x = NeuralEngine.surface(rows: shape.rows, columns: shape.k),
              let w = NeuralEngine.surface(rows: shape.k, columns: shape.n),
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
                "w": MLFeatureValue(multiArray: MLMultiArray(pixelBuffer: w, shape: dims([shape.k, shape.n]))),
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
            let url = directory.appendingPathComponent("\(shape.rows)x\(shape.k)x\(shape.n).mlmodelc", isDirectory: true)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            try NeuralEngine.programText(rows: shape.rows, k: shape.k, n: shape.n)
                .write(to: url.appendingPathComponent("model.mil"), atomically: true, encoding: .utf8)
            try NeuralEngine.description(rows: shape.rows, k: shape.k, n: shape.n)
                .write(to: url.appendingPathComponent("coremldata.bin"))
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            return try MLModel(contentsOf: url, configuration: configuration)
        } catch {
            return nil
        }
    }

    /// **The program**, MIL text as `coremltools` writes it for `matmul(x, w)`: fp16 `x [rows, k]` and
    /// `w [k, n]` in, fp16 `y [rows, n]` out.
    package static func programText(rows: Int, k: Int, n: Int) -> String {
        """
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
    package static func description(rows: Int, k: Int, n: Int) -> Data {
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
        /// A feature: its name, a multi-array of `dims` in fp16 (`ArrayDataType.FLOAT16` = 65 552).
        func feature(_ tag: UInt8, _ name: String, _ dims: [Int]) -> [UInt8] {
            let shape = field(0x0a, dims.flatMap(varint))
            let array = field(0x2a, shape + [0x10] + varint(65_552))
            return field(tag, field(0x0a, Array(name.utf8)) + field(0x1a, array))
        }
        let payload = feature(0x0a, "x", [rows, k]) + feature(0x0a, "w", [k, n]) + feature(0x52, "y", [rows, n])
            + [0xa2, 0x06, 0x00]
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
