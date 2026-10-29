import Foundation

/// **The table at the heights nobody measured, and the one form of a prepared weight** — the two things a plan alone
/// cannot give a `RouteTable`, because a render runs products at ~1 050, ~1 560, ~4 120 and ~6 170 rows where the map
/// measured two heights, and because a weight is read in one form for every evaluation that uses it.
///
///     measured heights of k×n :  m₁ ──── m₂            target m
///     hot-time ratio of route r :  ρᵣ(m₁)  ρᵣ(m₂)  →   ln ρᵣ(m) = (1−t)·ln ρᵣ(m₁) + t·ln ρᵣ(m₂),   t = ln(m/m₁) / ln(m₂/m₁)
///     route at m :  CapacityMeasure.rule( ρ(m) )         — the efficient route within 90 % of the best, else the simplest that wins
///
/// What sets it apart — what an interpolation written by analogy with the grids (`Conductor.Shares`) would get wrong:
///
///   - **The ratios are interpolated, not the choice.** The choice flips at a height; a route that wins at both
///     measured heights may lose between (its ratio crosses 1) and one that wins at neither may win (ANE and AMX
///     curves crossing). Interpolating the choice would copy a neighbour; interpolating the times and choosing again
///     reproduces what the map would have decided at that height had it measured it.
///   - **A ratio is a rate ratio**: ρᵣ = time of r over the reference's = rate(ref) / rate(r), so a linear ln ρ in
///     ln m is the difference of the two routes' log-rates — the quantity the engines' rates actually move along (the
///     affine-in-m model of `aneBalance` is the same line in a different chart). The reference's own rate drops out.
///   - **Past the last measured height the same line goes on** (two nearest anchors, `t` beyond 1), instead of
///     stopping at the last height: 6 144 rows after 4 096. With a single measured height there is no line and its
///     ratios stand. A route measured at one anchor only (a hole at the other) is not a candidate in between.
///   - **A decision of its own always wins**: an exact class in the plan keeps its figure; only a class with none is
///     filled. An open or unmeasured class is an anchor only if its ratios exist (a `none` decision has none).
///   - **A weight prepared once has one form.** `WeightRule`s are by kind, k×n and a height range; a rule that
///     reads the 8-bit weight published only from 2 048 rows would have the same tensor widened at 1 056 and kept
///     8-bit at 4 128 — two forms of one weight, prepared twice or the wrong one read. The form is the path
///     (`widened`, `bf16`, `native`); rounding the activation is the call's and may differ by height. For every
///     (phase, kind, k, n) whose rows disagree, the path of the **heaviest** rows (FLOP × executions, all families)
///     is pinned at every height of the group, ahead of the other rules.
extension RouteTable {
    /// One measured height of a class family and its routes' hot-time ratios.
    private struct Anchor {
        var x: Double
        var ids: [String]
        var efficient: [Bool]
        var ratios: [Double?]
    }

    /// **The route a class takes at a height the plan did not measure**, from the plan's decisions at the same
    /// k×n (products) or the same heads and head dim (attentions), `nil` when it has none to start from.
    package static func interpolatedRoute(for key: ClassKey, kind: WeightKind = .bf16, plan: CapacityMeasure.Plan,
                                          classes: [CapacityMeasure.ComputeClass], excludingANE: Bool = false) -> String? {
        var anchors: [Anchor] = []
        let target: Double
        switch key {
        case let .product(_, _, m): target = Double(m)
        case let .attention(s, _, _): target = Double(s)
        }
        for decision in plan.decisions where decision.status == .closed || decision.status == .open {
            guard let (other, otherKind) = ClassKey.parse(decision.classID),
                  { if case .product = key { return otherKind == kind } else { return true } }(),
                  let compute = classes.first(where: { $0.id == decision.classID }),
                  compute.routes.count == decision.hotRatios.count else { continue }
            let x: Double
            switch (key, other) {
            case let (.product(k, n, _), .product(k2, n2, m)) where k == k2 && n == n2: x = Double(m)
            case let (.attention(_, h, d), .attention(s, h2, d2)) where h == h2 && d == d2: x = Double(s)
            default: continue
            }
            anchors.append(Anchor(x: x, ids: compute.routes.map(\.id), efficient: compute.routes.map(\.efficient),
                                  ratios: zip(compute.routes, decision.hotRatios).map { route, ratio in
                                      excludingANE && route.id.hasPrefix("gpu+ane") ? nil : ratio.flatMap { $0 > 0 ? $0 : nil }
                                  }))
        }
        anchors.sort { $0.x < $1.x }
        guard let first = anchors.first else { return nil }
        var ratios = first.ratios
        if anchors.count > 1 {
            let below = anchors.last { $0.x < target }, above = anchors.first { $0.x > target }
            let (a, b): (Anchor, Anchor)
            switch (below, above) {
            case let (lo?, hi?): (a, b) = (lo, hi)
            case let (lo?, nil): (a, b) = (anchors[anchors.count - 2], lo)
            case let (nil, hi?): (a, b) = (hi, anchors[1])
            default: (a, b) = (anchors[0], anchors[1])        // every anchor at the target's own height
            }
            let t = a.x == b.x ? 0 : (log(target) - log(a.x)) / (log(b.x) - log(a.x))
            ratios = a.ids.indices.map { i in
                guard let ra = a.ratios[i], let j = b.ids.firstIndex(of: a.ids[i]), let rb = b.ratios[j] else { return nil }
                return exp((1 - t) * log(ra) + t * log(rb))
            }
            guard let index = CapacityMeasure.rule(ratios: ratios, efficient: a.efficient) else { return nil }
            return a.ids[index]
        }
        guard let index = CapacityMeasure.rule(ratios: ratios, efficient: first.efficient) else { return nil }
        return first.ids[index]
    }

    // MARK: - One form per prepared weight

    /// A weight (kind, k×n, in one phase) that the table reads in more than one path across the heights a render uses.
    package struct WeightPathConflict: Equatable, Sendable {
        package var phase: CapacityClasses.Phase
        package var kind: WeightKind
        package var k: Int
        package var n: Int
        /// The path at each height.
        package var paths: [Int: WeightPath]
    }

    /// A weight prepared once: its phase, its form and its shape.
    package struct WeightGroup: Hashable, Sendable {
        package var phase: CapacityClasses.Phase
        package var kind: WeightKind
        package var k: Int
        package var n: Int
        package init(phase: CapacityClasses.Phase, kind: WeightKind, k: Int, n: Int) {
            self.phase = phase; self.kind = kind; self.k = k; self.n = n
        }
    }

    /// Per group: its heights and the FLOP each carries over all the families.
    private static func weightGroups(_ classes: [CapacityClasses.Class]) -> [WeightGroup: [Int: Double]] {
        var groups: [WeightGroup: [Int: Double]] = [:]
        for c in classes {
            guard case let .product(k, n, rows, _) = c.shape, let kind = c.weightKind else { continue }
            let total = c.executions.keys.reduce(0.0) { $0 + c.totalFLOP($1) }
            groups[WeightGroup(phase: c.phase, kind: kind, k: k, n: n), default: [:]][rows, default: 0] += total
        }
        return groups
    }

    /// **The weights the table would read in two forms** at the heights of `classes`.
    package func weightPathConflicts(for classes: [CapacityClasses.Class]) -> [WeightPathConflict] {
        weightPathConflicts(groups: Self.weightGroups(classes))
    }

    /// The same, over groups given with the weight each height carries.
    package func weightPathConflicts(groups: [WeightGroup: [Int: Double]]) -> [WeightPathConflict] {
        groups.compactMap { group, heights in
            let paths = Dictionary(uniqueKeysWithValues: heights.keys.map {
                ($0, products.weight(group.kind, k: group.k, n: group.n, m: $0).path)
            })
            return Set(paths.values).count > 1
                ? WeightPathConflict(phase: group.phase, kind: group.kind, k: group.k, n: group.n, paths: paths) : nil
        }.sorted { ($0.phase.rawValue, $0.kind.rawValue, $0.k, $0.n) < ($1.phase.rawValue, $1.kind.rawValue, $1.k, $1.n) }
    }

    /// **Pins each conflicting weight to the path of its heaviest height** (ties: the taller), at every height of the
    /// group, keeping at each height the activation rounding the table gave when its path is the pinned one.
    package mutating func unifyWeightPaths(for classes: [CapacityClasses.Class]) {
        unifyWeightPaths(groups: Self.weightGroups(classes))
    }

    /// The same, over groups given with the weight each height carries (the capacity map's grid: its points' rows).
    package mutating func unifyWeightPaths(groups: [WeightGroup: [Int: Double]]) {
        var pins: [WeightRule] = []
        for conflict in weightPathConflicts(groups: groups) {
            let group = WeightGroup(phase: conflict.phase, kind: conflict.kind, k: conflict.k, n: conflict.n)
            guard let heights = groups[group],
                  let heaviest = heights.max(by: { ($0.value, $0.key) < ($1.value, $1.key) })?.key,
                  let path = conflict.paths[heaviest] else { continue }
            for rows in heights.keys.sorted() {
                let was = products.weight(conflict.kind, k: conflict.k, n: conflict.n, m: rows)
                pins.append(WeightRule(kind: conflict.kind, path: path, roundsActivation: was.path == path && was.roundsActivation,
                                       k: conflict.k, n: conflict.n, minRows: rows, maxRows: rows))
            }
        }
        products.weights = pins + products.weights
    }
}
