import Foundation

/// **What the capacity map needs, computed before its first computation from its shapes** — the
/// diagnostic's `MemoryNeed`, same rule: every term is a shape function its allocator calls (`Operands.sizes`,
/// `NeuralEngine.bytes`, `Conductor.workspaceBytes`, `HeadSplitAttention.hostBytes`…), never a figure read
/// from the machine, and a peak above the need is a defect of the code.
///
///     B (`MemoryBudget.machineCeiling`) ─ the process's base ─▶ what the map may hold
///        │
///        └─ `Plan.choose`: the fastest plan whose need fits, else `insufficientMemory` — before the first product
///
///     step          alive at its peak
///     products      operands · GEMM (BNNS workspaces, bf16 / int8 / Q8_0 paths) · the ANE engine's surfaces
///                   over the eight products · the BNNS workspace of the AMX alone · the exact rows
///     cuts          operands · the conductor's workspace over the grid
///     aneCuts       operands · two ANE engines (the pair, the three), each one batch of the grid — **the
///                   plan's lever** · two conductors' workspaces
///     lowPrecision  operands · the three half-precision copies · BNNS's workspace
///     attention     operands released · q, k, v, out · MPSGraph's queries copy or the head split's arena
///     widening      one Z-Image w1 in six forms
///     witness       the first product's operands again, and the AMX's workspace
///
/// **What sets it apart** — what a count by analogy with a render's would get wrong:
///
///   - **the Neural Engine's surfaces are the map's one unbounded term.** An engine keeps the three
///     IOSurfaces of every program shape it ran until it is released (`NeuralEngine.bytes`: a sum, not a
///     maximum), and the grid has ~80 shapes: one engine for the whole of it held 3.6 GB on top of the
///     operands (4.9 GB at the peak on the M1 Pro, 28/10/2026), and the MacBook Neo's swap went from 6 MB to
///     1.56 GB. The grid is cut in batches of shapes, an engine each, released between two batches;
///   - **the engines' rows are bounded, not measured**: a share is measured, the need is asked first — the
///     ANE's rows at the share's lower bound (`NeuralEngine.shareRange`), the largest it can take;
///   - **the operands are the map's, not a product's**: sized to the largest shape of the grid
///     (16 384 × 6 144 at 4 096 rows), they are held from the first product to the last half-precision one and
///     released before the attention. Cutting a product's rows to make it fit would measure another product
///     (the GPU's rate and the ANE's share move with the height): a map that does not fit is refused;
///   - **the process's base is read, not charged**: B covers the footprint the map starts from (an app's
///     interface, the command line's runtime) with it.
extension Diagnostic {
    /// **The map's steps**, in the order they run (`Gauge` marks them by the same names).
    package static let needSteps = ["products", "cuts", "aneCuts", "lowPrecision", "attention", "widening", "witness"]

    /// One point of the ANE's grid: a product's weight shape at a height, in the order it is measured.
    package struct GridPoint: Sendable, Equatable { package let m, k, n: Int }

    package static var aneGrid: [GridPoint] {
        aneShapes.flatMap { s in heights.map { GridPoint(m: $0, k: s.k, n: s.n) } }
    }

    /// **The plan's lever**: the surface bytes one ANE engine may hold before the grid goes on with a new
    /// one. `nil`: one engine for the whole grid. A batch holds at least one point, whatever its size.
    package static func aneBatches(allotment: Int?) -> [[GridPoint]] { aneBatches(points: aneGrid, allotment: allotment) }

    /// The same over any list of points (the capacity map's classes, in the order they are asked).
    package static func aneBatches(points: [GridPoint], allotment: Int?) -> [[GridPoint]] {
        guard let allotment else { return [points] }
        var batches: [[GridPoint]] = []
        for point in points {
            if let last = batches.last, engineBytes(last + [point]) <= allotment { batches[batches.count - 1].append(point) }
            else { batches.append([point]) }
        }
        return batches
    }

    /// **An ANE engine over these points, at the most rows it can take**: the share is measured, the need is asked
    /// first. The ANE takes at most `m − 256` rows (a program of all `m` is the GPU's whole product, refused by
    /// `NeuralEngine.rows`), so each height is asked at the share that gives it exactly those: the heights are
    /// summed apart, the per-column state they share (a few MB) counted once each — over, never under.
    static func engineBytes(_ points: [GridPoint]) -> Int {
        Dictionary(grouping: points, by: \.m).map { m, group in
            let rows = (m - 1) / NeuralEngine.rowUnit * NeuralEngine.rowUnit
            return NeuralEngine.bytes(products: group.map { ($0.m, $0.k, $0.n) }, gpuShare: 1 - Double(rows) / Double(m))
        }.reduce(0, +)
    }

    /// **The Core ML programs an engine over these points loads**: one per part of K (`NeuralEngine.parts`), at
    /// the rows it can take. Each is the framework's own memory, beyond its three surfaces.
    static func programCount(_ points: [GridPoint]) -> Int {
        Set(points.map { [$0.m, $0.k, $0.n] }).reduce(0) { $0 + NeuralEngine.parts(k: $1[1], n: $1[2]) }
    }

    /// **What a loaded program holds beyond its surfaces** (Core ML 27, M1 Pro, 28/10/2026): the 22 programs of the
    /// eight products held ~180 MB over `NeuralEngine.bytes`. The framework's, not a shape's — a constant, said as one,
    /// the check (`Memory`) reports what is left of it.
    static let programBytes = 8 << 20

    /// **What the frameworks keep that no shape gives**, charged to every step (28/10/2026, M1 Pro): MPS's and
    /// MPSGraph's compiled kernels, Metal's pipelines, Core ML's runtime — 76 to 100 MB held after the half-precision
    /// graphs, the cuts' and the lowest steps within 20 MB of their need without it. `OperatorFootprint` lists them
    /// as outside our hand; here one constant, so that the plan does not count on their absence.
    static let frameworks = 128 << 20

    /// The step by step need, in bytes, **without the process's base**.
    package struct Need: Sendable, Equatable, Codable {
        package var steps: [String: Int]
        package var peak: Int { steps.values.max() ?? 0 }
    }

    package static func need(allotment: Int?) -> Need {
        let operands = Operands.sizes.bytes
        let products = shapes.map { (m: $0.m, k: $0.k, n: $0.n) }
        let grid = (cutShapes + aneShapes).map { s in heights.map { (m: $0, k: s.k, n: s.n) } }.flatMap { $0 }
        func gemm(_ products: [(m: Int, k: Int, n: Int)], int8: Bool) -> Int {
            MemoryNeed.gemm("diagnostic", products: products, amx: true, int8: int8, ane: false).reduce(0) { $0 + $1.bytes }
        }
        func multiply(_ products: [(m: Int, k: Int, n: Int)]) -> Int {
            products.map { Arena.capacity(for: Conductor.workspaceBytes(rows: $0.m, k: $0.k, n: $0.n, weightIsTransposed: true)) }.max() ?? 0
        }
        /// The sampled rows' fp64: the exact product (`exactRows`: `A`'s rows, a 32 MB slice of `B`, the
        /// product and its transpose), the results kept beside it, and the rounded rows of the ANE and bf16 paths.
        func exact(_ products: [(m: Int, k: Int, n: Int)]) -> Int {
            products.map { p in
                let r = sampledRows
                return r * p.k * 12 + (1 << 25) + r * p.n * 8 * 5 + r * p.n * 4
            }.max() ?? 0
        }
        let tail = aneBatches(allotment: allotment).map { engineBytes($0) + programCount($0) * programBytes }.max() ?? 0

        // products: GEMM `alone` (bf16 weights, bf16 activations), `both` (the conductor, int8, Q8_0), `neural` (the ANE
        // over all eight products, its share bounded below).
        let productPoints = products.map { GridPoint(m: $0.m, k: $0.k, n: $0.n) }
        let productsNeed = operands + gemm(products, int8: true) + engineBytes(productPoints) + programCount(productPoints) * programBytes
            + multiply(products) + exact(products)
        let cutsNeed = operands + gemm(cutShapes.flatMap { s in heights.map { (m: $0, k: s.k, n: s.n) } }, int8: false)
        // aneCuts: the pair's engine and the three's, a batch each; the AMX's GEMM and the three's conductor.
        let aneGrid = aneShapes.flatMap { s in heights.map { (m: $0, k: s.k, n: s.n) } }
        let aneNeed = operands + 2 * tail + 2 * gemm(aneGrid, int8: false)

        let z = shapes[0]
        let half = Arena.capacity(for: z.m * z.k * 2) + Arena.capacity(for: z.k * z.n * 2) + Arena.capacity(for: z.m * z.n * 2)
        let lowNeed = operands + half + multiply([(z.m, z.k, z.n)]) + exact([(z.m, z.k, z.n)])

        // attention: q, k, v, out; then the largest of MPSGraph's queries copy, the flash kernel (no buffer), the head split.
        let attentionNeed = blocks.flatMap { b in heights.map { s -> Int in
            let floats = s * b.heads * b.headDim
            let gpuHeads = HeadSplitAttention.partition(heads: b.heads, fraction: HeadSplitAttention.referenceFraction).gpu.count
            let split = HeadSplitAttention.hostBytes(heads: b.heads, sequence: s, headDim: b.headDim,
                                                     cpuFraction: HeadSplitAttention.referenceFraction)
                + OperatorFootprint.sdpaQueries(rows: s, heads: gpuHeads, headDim: b.headDim)
            let graph = OperatorFootprint.sdpaQueries(rows: s, heads: b.heads, headDim: b.headDim)
            let exactBytes = sampledQueries * b.heads * b.headDim * 8 * 2
            return 4 * Arena.capacity(for: floats * 4) + max(split, graph) + exactBytes
        } }.max() ?? 0

        // widening: one Z-Image w1 in fp32, bf16 twice, int8 twice, the scales.
        let w = blocks[0], count = w.dim * w.hidden, blockScales = (w.dim + q8Block - 1) / q8Block * w.hidden
        let wideningNeed = [count * 4, count * 2, count * 2, count + 2 * w.hidden, count, blockScales * 2, blockScales * 4, 1 << 20]
            .map { Arena.capacity(for: $0) }.reduce(0, +)
        // witness: the first product's operands again, the AMX's workspace.
        let witnessNeed = Arena.capacity(for: z.m * z.k * 4) + Arena.capacity(for: z.k * z.n * 4) + Arena.capacity(for: z.m * z.n * 4)
            + multiply([(z.m, z.k, z.n)])
        return Need(steps: ["products": productsNeed, "cuts": cutsNeed, "aneCuts": aneNeed, "lowPrecision": lowNeed,
                            "attention": attentionNeed, "widening": wideningNeed, "witness": witnessNeed].mapValues { $0 + frameworks })
    }

    /// **The plan the map runs under**, chosen before its first computation: the fastest — the fewest ANE engines
    /// — whose need and base fit B; `insufficientMemory` when not even one point at a time does.
    package struct Plan: Sendable, Equatable, Codable {
        package var budget: Int
        package var base: Int
        package var aneAllotment: Int?
        package var need: Need

        /// From the largest allotment (the whole grid on one engine, a release wait saved per batch) to none.
        package static let allotments: [Int?] = [nil, 1 << 30, 1 << 29, 1 << 28, 0]

        package static func choose(budget: Int, base: Int, need needFor: (Int?) -> Need = { Diagnostic.need(allotment: $0) }) throws(EngineError) -> Plan {
            var smallest = 0
            for allotment in allotments {
                let need = needFor(allotment)
                smallest = need.peak
                if base + need.peak <= budget { return Plan(budget: budget, base: base, aneAllotment: allotment, need: need) }
            }
            throw .insufficientMemory(needed: base + smallest, budget: budget)
        }
    }
}
