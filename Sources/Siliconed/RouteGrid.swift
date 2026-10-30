import Foundation

// ═══════════════════════════════════════════════════════════════════════════════════════════
// LEGACY — read by nothing in the engine. The grids of the previous capacity map and the numbers the conductor once held, kept
// only because that map (`CapacityMeasure`, `CapacityMapRun`, `CapacityRules`, `RouteTable`) still compiles; E4b removes them
// together with it. The engine asks the plan (`CapacityPlan`) for exact rows: no ratio, no grid, no nearest point.
// ═══════════════════════════════════════════════════════════════════════════════════════════

extension Conductor {
    /// The previous map's cut step (`amx_quantum`, 64 under a grid) and the band its ratios were held in.
    package static let legacyQuantum = 64
    package static let fractionRange = 0.30...0.95
    /// Below this, the previous map did not ask for the AMX (`amx_min`).
    package static let minimumRows = 512
    /// Where the previous cut fell for a share `T`.
    package static func cut(_ m: Int, _ T: Double) -> Int {
        let q = legacyQuantum
        return max(q, min(m - q, Int((Double(m) * T / Double(q)).rounded()) * q))
    }

    /// **The GPU's share of a product, computed from its shape and the machine's map** — never a constant of
    /// a machine or of a model:
    ///
    ///     T(k, n, m) = 1 / (1 + ρ(k, n, m))
    ///     ρ(k, n, m) = ρ(k, w, m) · ρ(w, n, m) / ρ(w, w, m)      ρ = the AMX's rate over the GPU's, side by side
    ///
    /// The map (`Diagnostic.Cut`) measures `ρ` along two axes through one square shape `w × w`: `k` at
    /// `n = w`, `n` at `k = w`, at both heights. Each axis is read linearly in `log k` or `log n` (the
    /// nearest point outside), the two composed, then read between the heights in `log m`. Any model's
    /// product enters by its own `k`, `n` and `m`: the grid is the machine's, the shapes the model's.
    ///
    /// Why two axes (measured on an M1 Pro): the AMX's rate falls with the length of its dot products `k` — the
    /// map's ρ goes from 0.52 at `k` = 2 048 to 0.36 at 16 384 (1 024 rows), and the engine's servo balances
    /// Anima's 2 048 × 8 192 at 0.63 and its 8 192 × 2 048 at 0.76, the same weight —; the width `n` may move
    /// it too (Krea 2's servo: 6 144 × 16 384 at 0.75, 6 144 × 6 144 at 0.69), and a measured axis lets it
    /// enter. This Mac's `n` axis came out flat (0.68–0.70 at `k` = 6 144), the gap with the engine not
    /// explained. Against one share for every shape, an evaluation: Anima −2 % at 512², −4 % at 1024²,
    /// Z-Image −2.7 %, Krea 2 −1.5 % at 1024², +2.5 % at 512² (its 6 144 × 16 384 at 0.677).
    ///
    /// **Pure**: the same map, shape and height give the same share — the frozen cut stays reproducible bit
    /// for bit on one machine. An empty grid gives `nil`: the scalar `amxFraction` then serves. A grid
    /// without its square point reads its `k` axis only.
    package struct Shares: Equatable, Sendable, CustomStringConvertible {
        /// `ρ` measured on the product `k × n` at height `m`.
        package struct Point: Equatable, Sendable {
            package let k, n, m: Int
            package let ratio: Double
            package init(k: Int, n: Int, m: Int, ratio: Double) { self.k = k; self.n = n; self.m = m; self.ratio = ratio }
        }
        package let points: [Point]
        package init(_ points: [Point]) {
            self.points = points.sorted { ($0.m, $0.k, $0.n) < ($1.m, $1.k, $1.n) }
        }
        package static let none = Shares([])
        package var isEmpty: Bool { points.isEmpty }

        /// `T(k, n, m)`, clamped to `range` (`Conductor.fractionRange` for the AMX's cut; the ANE's is wider).
        package func share(k: Int, n: Int, m: Int, range: ClosedRange<Double> = Conductor.fractionRange,
                           extrapolatingHeights: Bool = false) -> Double? {
            ratio(k: k, n: n, m: m, extrapolatingHeights: extrapolatingHeights)
                .map { min(range.upperBound, max(range.lowerBound, 1 / (1 + $0))) }
        }

        /// `ρ(k, n, m)` itself, read as `share` reads it, unclamped: what a grid of 0 and 1 (`ane_amx`) says of a product.
        ///
        /// `extrapolatingHeights`: past the measured heights, ρ goes on along the line in `log m` through the two
        /// nearest heights (a 6 144-row product beside a grid measured at 1 024 and 4 096), where the default holds
        /// the nearest height's ρ. The route table's lookup for classes that sit beyond their measure; the caller
        /// bounds the result (`share`'s range).
        package func ratio(k: Int, n: Int, m: Int, extrapolatingHeights: Bool = false) -> Double? {
            let heights = Array(Set(points.map(\.m))).sorted()
            guard !heights.isEmpty else { return nil }
            /// Linear in `log x` between the two points around `x`, the nearest outside.
            func between<T>(_ items: [T], _ x: Int, at: (T) -> Int, value: (T) -> Double) -> Double {
                let sorted = items.sorted { at($0) < at($1) }
                if x <= at(sorted[0]) { return value(sorted[0]) }
                if x >= at(sorted[sorted.count - 1]) { return value(sorted[sorted.count - 1]) }
                let i = sorted.firstIndex { at($0) >= x }!
                let (a, b) = (sorted[i - 1], sorted[i])
                let t = log(Double(x) / Double(at(a))) / log(Double(at(b)) / Double(at(a)))
                return value(a) + t * (value(b) - value(a))
            }
            func ratio(at h: Int) -> Double {
                let level = points.filter { $0.m == h }
                // The shape measured itself, when it was; then one axis, or two through one pivot (the AMX's
                // grid); otherwise (the ANE's grid, a family's shapes) the nearest shape in log k + log n.
                if let exact = level.first(where: { $0.k == k && $0.n == n }) { return exact.ratio }
                if level.allSatisfy({ $0.n == level[0].n }) { return between(level, k, at: \.k, value: \.ratio) }
                let pivot = level.first(where: { $0.k == $0.n })
                guard let pivot, level.allSatisfy({ $0.k == pivot.k || $0.n == pivot.n }) else {
                    func distance(_ p: Point) -> Double { abs(log(Double(p.k) / Double(k))) + abs(log(Double(p.n) / Double(n))) }
                    return level.min { distance($0) < distance($1) }!.ratio
                }
                let depth = between(level.filter { $0.n == pivot.n }, k, at: \.k, value: \.ratio)
                let width = between(level.filter { $0.k == pivot.k }, n, at: \.n, value: \.ratio)
                return depth * width / pivot.ratio
            }
            let levels = heights.map { ($0, ratio(at: $0)) }
            if extrapolatingHeights, levels.count >= 2, m < levels[0].0 || m > levels[levels.count - 1].0 {
                let (a, b) = m < levels[0].0 ? (levels[0], levels[1]) : (levels[levels.count - 2], levels[levels.count - 1])
                let t = log(Double(m) / Double(a.0)) / log(Double(b.0) / Double(a.0))
                return a.1 + t * (b.1 - a.1)
            }
            return between(levels, m, at: \.0, value: \.1)
        }

        /// `2048×6144@1024=0.601,…` — `k×n@m=ρ`, the setting's text (`amx_shares`), the profile's and the
        /// environment's.
        package var description: String {
            points.map { "\($0.k)×\($0.n)@\($0.m)=\(String(format: "%.3f", $0.ratio))" }.joined(separator: ",")
        }
        /// The text `description` writes; `nil` when a piece does not read. Empty: no grid.
        package init?(text: String) {
            var points: [Point] = []
            for piece in text.split(separator: ",") {
                let parts = piece.split(whereSeparator: { "×@=".contains($0) }).map(String.init)
                guard parts.count == 4, let k = Int(parts[0]), let n = Int(parts[1]), let m = Int(parts[2]),
                      let ratio = Double(parts[3]) else { return nil }
                points.append(Point(k: k, n: n, m: m, ratio: ratio))
            }
            self.init(points)
        }
    }

}

extension NeuralEngine {
    /// The shares a previous grid could give: from the GPU alone (1) down to a twentieth of the rows.
    static let shareRange = 0.05...1.0
    /// The ANE's rows the previous map's ratio gave an `m`-row product.
    static func rows(_ m: Int, gpuShare share: Double, minimum: Int) -> Int {
        guard m >= minimum else { return 0 }
        let unit = Double(rowUnit)
        let r = Int((Double(m) * (1 - share) / unit).rounded()) * rowUnit
        return r > 0 && r < m ? r : 0
    }
}

extension WeightProducts {
    /// The AMX's columns of `n` at the GPU's share `T` the previous map balanced: held in the band, the GPU's a multiple of 64.
    static func amxColumns(n: Int, share T: Double) -> Int {
        let ceiling = (n - 64) / 64 * 64
        guard ceiling >= 64 else { return 0 }
        let T = min(Conductor.fractionRange.upperBound, max(Conductor.fractionRange.lowerBound, T))
        return n - max(64, min(ceiling, Int((T * Double(n) / 64).rounded()) * 64))
    }
}

extension RouteTable {
    /// **The table the plan's answers make for these classes** — so that the previous render check (`RenderLoop`), which names routes
    /// through a table, speaks of what the engine really ran: one class rule, one weight rule and one attention rule per class, the
    /// ANE's rows as the ratio that gives exactly them.
    init(answers plan: CapacityPlan, classes: [CapacityClasses.Class]) {
        self.init()
        var amx = false, ane = false
        var anePoints: [Conductor.Shares.Point] = []
        for c in classes {
            switch c.shape {
            case let .product(k, n, m, format):
                let a = plan.restricted(allowingANE: c.phase == .denoising && plan.usesANE)
                    .assign(ProductKey(k: k, n: n, m: m, weight: format, phase: c.phase))
                let engines: Engines = a.usesANE ? (a.usesAMX ? .aneAMX : .ane) : (a.usesAMX ? .amx : .gpu)
                amx = amx || a.usesAMX; ane = ane || a.usesANE
                products.classes.append(.init(engines: engines, k: k, n: n, minRows: m, maxRows: m, kind: format.kind))
                if let kind = format.kind {
                    products.weights.append(.init(kind: kind, path: a.path, roundsActivation: a.roundsActivation, k: k, n: n,
                                                  minRows: m, maxRows: m))
                }
                if a.ane > 0 && a.ane < m { anePoints.append(.init(k: k, n: n, m: m, ratio: Double(a.ane) / Double(m - a.ane))) }
            case let .attention(s, keys, heads, kv, dim, cross):
                let a = plan.assign(AttentionKey(sequence: s, keys: keys, heads: heads, kvHeads: kv, headDim: dim, cross: cross),
                                    flash: CapacitySolver.triesFlash(family: Family.allCases.first { c.executions[$0] != nil } ?? .zImage,
                                                                     cross: cross))
                attention.rules.append(.init(minSequence: s, maxSequence: s, heads: heads, headDim: dim, flash: a.usesFlash,
                                             splitFraction: a.splits ? Double(a.cpuHeads) / Double(heads) : nil))
            }
        }
        products.amx = .init(enabled: amx)
        products.ane = .init(enabled: ane, gpuShare: 1, minimumRows: 1, grid: .init(Conductor.Shares(anePoints)))
    }
}
