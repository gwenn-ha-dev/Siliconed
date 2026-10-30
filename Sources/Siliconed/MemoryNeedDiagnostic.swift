import Foundation

/// **What the capacity probes need, computed before the first computation from their shapes** — the diagnostic's `MemoryNeed`,
/// same rule: every term is a shape function its allocator calls (`Operands.sizes`, `NeuralEngine.bytes`, `Conductor.workspaceBytes`,
/// `HeadSplitAttention.hostBytes`…), never a figure read from the machine, and a peak above the need is a defect of the code.
/// The terms themselves are summed by `CapacityProbes.need`; this file holds what they are made of and the plan that picks the
/// probes' size under B.
///
///     B (`MemoryBudget.machineCeiling`) ─ the process's base ─▶ what the probes may hold
///        │
///        └─ `Plan.choose`: the largest tier of probe sizes whose need fits, else `insufficientMemory` — before the first product
///
/// **What sets it apart** — what a count by analogy with a render's would get wrong:
///
///   - **the Neural Engine's surfaces are the one unbounded term.** An engine keeps the three IOSurfaces of every program shape it
///     ran until it is released (`NeuralEngine.bytes`: a sum, not a maximum): the shapes of the probes are few and counted whole;
///   - **the engines' rows are bounded, not measured**: the ANE is asked first at the most rows it can take (`m − 256`: a program of
///     all `m` is the GPU's whole product, refused by `NeuralEngine.rows`), the largest it can take;
///   - **the operands are the probes', not a product's**: sized to the largest shape of the tier, they are held from the first product
///     to the last. Cutting a product's rows to make it fit would measure another product (the GPU's rate and the ANE's keep move
///     with the height): a tier that does not fit is dropped for the next one, and a Mac where not even the last fits is refused;
///   - **the process's base is read, not charged**: B covers the footprint the probes start from (an app's interface, the command
///     line's runtime) with it.

extension Diagnostic {
    /// One product of the probes as the ANE's engine sees it: a weight shape at a height.
    package struct GridPoint: Sendable, Equatable { package let m, k, n: Int }

    /// **An ANE engine over these points, at the most rows it can take**: the share is measured, the need is asked
    /// first. The ANE takes at most `m − 256` rows (a program of all `m` is the GPU's whole product, refused by
    /// `NeuralEngine.rows`), so each height is asked at the share that gives it exactly those: the heights are
    /// summed apart, the per-column state they share (a few MB) counted once each — over, never under.
    static func engineBytes(_ points: [GridPoint], integers: [GridPoint] = []) -> Int {
        Dictionary(grouping: points, by: \.m).map { m, group in
            let rows = (m - 1) / NeuralEngine.rowUnit * NeuralEngine.rowUnit
            // `integers`: the points the integers' route is probed on also hold the int8 program's shapes (the convrot group of the
            // catalogue's int8 weights, 256), beside the fp16 ones — only those: the staircase and the witness run fp16 alone.
            return NeuralEngine.bytes(products: group.map { (rows, $0.k, $0.n) },
                                      integers: group.filter { integers.contains($0) }.map { (rows, $0.k, $0.n, Widen.largestRotationGroup) })
        }.reduce(0, +)
    }

    /// **The Core ML programs an engine over these points loads**: one per part of K (`NeuralEngine.parts`), at
    /// the rows it can take. Each is the framework's own memory, beyond its three surfaces.
    static func programCount(_ points: [GridPoint], integers: [GridPoint] = []) -> Int {
        Set(points.map { [$0.m, $0.k, $0.n] }).reduce(0) { $0 + NeuralEngine.parts(k: $1[1], n: $1[2]) }
            + Set(integers.map { [$0.m, $0.k, $0.n] }).reduce(0) { $0 + NeuralEngine.parts(k: $1[1], n: $1[2], int8: true, alignment: Widen.largestRotationGroup) }
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

    /// **The plan the probes run under**, chosen before the first computation: the largest tier of sizes (`CapacityProbes.Sizes`) whose
    /// need and base fit B, from `first`; `insufficientMemory` when not even the smallest does.
    package struct Plan: Sendable, Equatable, Codable {
        package var budget: Int
        package var base: Int
        package var tier: Int
        package var need: Need

        package static func choose(budget: Int, base: Int, from first: Int = 0, need needFor: (_ tier: Int) -> Need) throws(EngineError) -> Plan {
            var smallest = 0
            for tier in first..<CapacityProbes.Sizes.tiers.count {
                let need = needFor(tier)
                smallest = need.peak
                if base + need.peak <= budget { return Plan(budget: budget, base: base, tier: tier, need: need) }
            }
            throw .insufficientMemory(needed: base + smallest, budget: budget)
        }
    }
}
