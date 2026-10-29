import Foundation

/// **What the capacity map needs, computed from its classes before its first product** — the same rule as ever
/// (`MemoryNeedDiagnostic.swift`): every term a shape function, the plan chosen under B, a peak above the need a
/// defect of the code. The map's classes are asked in the order the measuring core chooses (the heaviest decision
/// first), so no step can be sized to a sequence: the need is the **sum of everything that may be alive at once**.
///
///     widening   one Z-Image w1 in six forms (the preparation's rates), before anything else is allocated
///     routes     operands (the largest of every class's A, B, C, the int8 and Q8_0 weights) ·
///                the GEMMs' padding tiers and the conductor's workspace (the GPU alone, the cut, the pair, the three) ·
///                the ANE engines' surfaces — **the plan's lever**: the engine that has run a set of shapes is released and
///                rebuilt before it would pass `allotment` (a pair's and a three's engine, one batch each) —,
///                and the witnesses' own engine · the exact rows and the cache of truths ·
///                the attention's q, k, v and out at the largest class, and MPSGraph's queries copy or the split's arena
///
/// What sets it apart:
///
///   - **no step is smaller than the largest class's**: the operands are not cut to a class (a product cut to fit
///     would measure another product — the GPU's rate and the ANE's share move with the height), so a Mac whose B
///     cannot hold the operands of the heaviest decision is refused (`insufficientMemory`), as it was;
///   - **the ANE bound is asked at the most rows it can take** (`Diagnostic.engineBytes`), over the points in the
///     order of the *grid* (the order the plan will ask them is not known); a batch holds at least one point;
///   - **the cache of truths is bounded** (`truthCache`): a truth is 64 rows of fp64 over the n columns, kept so that a
///     route first run after another is judged without widening B again; evicted first-in first-out.
extension CapacityMap {
    package static let needSteps = ["widening", "routes"]
    /// Bytes of fp64 truths the map keeps (`Diagnostic.sampledRows` rows of n columns each).
    package static let truthCache = 256 << 20
    /// The witnesses' product: Z-Image's `[3840, 3840]` square at 2 048 rows — big enough for each engine to be timed
    /// on a few milliseconds, small enough to be run ten times in a second.
    package static let witnessShape = (m: 2048, k: 3840, n: 3840)

    package static func products(_ entries: [Entry]) -> [(m: Int, k: Int, n: Int, kind: RouteTable.WeightKind)] {
        entries.compactMap { e in
            // A point whose routes read an 8-bit form holds its staged weight and scales too.
            guard case let .product(k, n, m) = e.work else { return nil }
            return (m, k, n, e.routes.contains { Entry.form($0.id).map { $0 != "bf16" && $0 != "bf16act" } ?? false } ? .q8 : .bf16)
        } + [(witnessShape.m, witnessShape.k, witnessShape.n, .bf16)]
    }

    /// The ANE-eligible points of the entries (the witness's included, first).
    package static func anePoints(_ entries: [Entry]) -> [Diagnostic.GridPoint] {
        entries.compactMap { e in
            if case let .product(k, n, m) = e.work, e.routes.contains(where: { $0.id == "gpu+ane" }) { return .init(m: m, k: k, n: n) }
            return nil
        }
    }

    package static func need(entries: [Entry], allotment: Int?) -> Diagnostic.Need {
        let ps = products(entries).map { (m: $0.m, k: $0.k, n: $0.n) }
        let operands = Diagnostic.Operands.sizes(products: products(entries)).bytes
        func gemm(_ ps: [(m: Int, k: Int, n: Int)]) -> Int {
            MemoryNeed.gemm("diagnostic", products: ps, amx: true, int8: true, ane: false).reduce(0) { $0 + $1.bytes }
        }
        let multiply = ps.map {
            Arena.capacity(for: Conductor.workspaceBytes(rows: $0.m, k: $0.k, n: $0.n, weightIsTransposed: true))
        }.max() ?? 0
        let exact = ps.map { p in
            let r = Diagnostic.sampledRows
            return r * p.k * 12 + (1 << 25) + r * p.n * 8 * 5 + r * p.n * 4
        }.max() ?? 0
        let points = anePoints(entries)
        let batches = Diagnostic.aneBatches(points: points, allotment: allotment)
        let tail = batches.map { Diagnostic.engineBytes($0) + Diagnostic.programCount($0) * Diagnostic.programBytes }.max() ?? 0
        let witness = [Diagnostic.GridPoint(m: witnessShape.m, k: witnessShape.k, n: witnessShape.n)]
        let witnessANE = Diagnostic.engineBytes(witness) + Diagnostic.programCount(witness) * Diagnostic.programBytes

        // Attention: q, k, v, out at the largest class, then the largest of MPSGraph's queries copy and the head split's arena.
        let attention = entries.compactMap { e -> Int? in
            guard case let .attention(s, heads, dim) = e.work else { return nil }
            let floats = s * heads * dim
            let gpuHeads = HeadSplitAttention.partition(heads: heads, fraction: HeadSplitAttention.referenceFraction).gpu.count
            let split = HeadSplitAttention.hostBytes(heads: heads, sequence: s, headDim: dim, cpuFraction: HeadSplitAttention.referenceFraction)
                + OperatorFootprint.sdpaQueries(rows: s, heads: gpuHeads, headDim: dim)
            let graph = OperatorFootprint.sdpaQueries(rows: s, heads: heads, headDim: dim)
            return 4 * Arena.capacity(for: floats * 4) + max(split, graph) + Diagnostic.sampledQueries * heads * dim * 8 * 2
        }.max() ?? 0

        let routes = operands + 3 * gemm(ps) + 2 * tail + witnessANE + multiply + exact + truthCache + attention
        let w = Diagnostic.blocks[0], count = w.dim * w.hidden, blockScales = (w.dim + Diagnostic.q8Block - 1) / Diagnostic.q8Block * w.hidden
        let widening = [count * 4, count * 2, count * 2, count + 2 * w.hidden, count, blockScales * 2, blockScales * 4, 1 << 20]
            .map { Arena.capacity(for: $0) }.reduce(0, +)
        return .init(steps: ["widening": widening, "routes": routes].mapValues { $0 + Diagnostic.frameworks })
    }
}
