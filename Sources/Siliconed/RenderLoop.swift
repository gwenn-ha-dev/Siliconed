import Foundation

/// **The render checks the map: what each class cost, against what the map predicted for it.** Pure — the render's side
/// is the ledger (`ProductLedger`, `Engine.Render.classes`), the map's side is a schema-13 report and the route table the
/// engine ran under; this file lines them up class by class and says how far apart they are.
///
///     Render.classes ─▶ compare ◀── predict ◀── CapacityClasses.classes(request)  (the catalogue's classes of this render)
///      (k×n@m, kind, route,            │            │  route  = the table's, with the weight's form
///       executions, seconds)           │            │  seconds = executions · FLOP / rate(route) · heat(route)
///                                      ▼            └─ rate    = the witness's GPU rate ÷ the class's cold ratio (the map)
///                    rows (class, routes, executions, measured, predicted, error) · totals · coverage
///                                      │
///                                      ▼
///                    verdict:  held  ·  correct(factor per engine)  ·  remeasure  ·  unjudged        (`RenderLoopVerdict.swift`)
///
/// What sets it apart — what a comparison written by analogy with "sum the census, divide by the model" would get wrong:
///
///   - **Heights are merged as the map merges them**: within `CapacityMap.heightGap` rows (64) of a neighbour of the same
///     weight kind, `k` and `n`, one class — Z-Image's 4 096 (noise refiner) and 4 128 (layers) are one measurement at
///     4 100 ± 30 rows, and so is the prediction of the 4 117 a prompt of another length makes. The class is named by its
///     heaviest member's height. An attention is exact (S, heads, head dim), a self- and a cross-attention of Anima adding up.
///   - **The map stores ratios, not seconds.** A class's rate is the GPU witness's (the map's own, cold, at the start) divided
///     by the class's cold ratio to the GPU alone, which is the hot ratio the plan keeps undone of the regime
///     (`hot · heat(reference) / heat(route)`); a shape reads the heights of the nearest point of the map's grid within reach
///     (`RouteTable.Nearest`, the point its route comes from), a height nobody measured takes the line through the two nearest
///     measured ones, in `ln ratio` against `ln m` (what `RouteTable.interpolatedRoute` does for the choice), and a route nobody timed
///     takes the GPU alone's rate — each row says which (`Provenance`). The witness is one product (2 048×3 840×3 840): a
///     class whose GPU rate differs from it by shape (MPS runs 2.3 to 3.6 TFLOP/s over these shapes) carries that difference
///     into its row. The attention's reference rate is the map's reference ratio to a product's, not a measure of this Mac.
///   - **The predicted route is the one the engine will take**, weight form included (`CapacityClasses.route(of:…weightForms)`):
///     the ANE only if the process brought it up (`Render.neuralEngine`), the AMX only from `amx_min` rows (the engine's
///     setting, passed in), `gpu/bf16` where the table pins the bf16 weight, `gpu+amx/int8` where it pins the 8-bit one.
///   - **A class that ran and was not predicted, or the other way round, is a row of its own**: Krea 2's text fusion, the
///     adapters, a tile the catalogue does not carry. The totals count them (a render is what it ran); the error that judges
///     the map is the matched one, and `coverage` says how much of the render it speaks for.
///   - **The attentions are timed as the products are**: wall seconds around the kernel's `run` (the three kernels record).
///     An encoder's attention, the VAE, the modulation gemv and the elementwise kernels are not in the ledger — nor in the
///     catalogue — so the sums are of the classes, never of the render's denoising seconds (which hold those too).
package enum RenderLoop {
    /// How a class's rate was known: timed by the map at that height, read off the line through measured heights, or not known
    /// (the GPU alone's rate stands in).
    package enum Provenance: String, Sendable, Comparable, Codable {
        case measured, interpolated, assumed

        package static func < (a: Provenance, b: Provenance) -> Bool { a.order < b.order }
        private var order: Int { self == .measured ? 0 : self == .interpolated ? 1 : 2 }
    }

    // MARK: - The map's rates, for a render's classes

    /// **What the map says about the classes of one render**: cold rates by route, the regime to predict at, and how each rate
    /// was known.
    package struct Basis: Sendable {
        package var rates: CapacityClasses.Rates
        /// The regime a render is predicted at: the map's heat, times what earlier renders learned (`Regime.learned`).
        package var heat: CapacityMeasure.Heat
        /// The map's own heat — its classes' hot ratios were decided at it, and are undone of it.
        package var mapHeat: CapacityMeasure.Heat
        private var sources: [String: [String: Provenance]]

        package func provenance(_ classID: String, route: String) -> Provenance { sources[classID]?[route] ?? .assumed }

        private struct Anchor { var x: Double; var cold: [String: Double] }

        package init(classes: [CapacityClasses.Class], report: Diagnostic.Report) {
            let mapHeat = CapacityMeasure.Heat(report.regime?.heat ?? [:])
            var now = mapHeat.factors
            for (engine, factor) in report.regime?.learned ?? [:] { now[engine, default: 1] *= factor }
            self.mapHeat = mapHeat
            heat = CapacityMeasure.Heat(now)

            // The GPU alone's cold rate: the witness of the start. The attention's reference stands in the map's own ratio to it.
            let reference = CapacityMap.referenceRates
            let gpuRate = report.witness.gpuBefore > 0 ? report.witness.gpuBefore * 1e12 : reference.product["gpu"]!
            let attentionRate = gpuRate * reference.attention["graph"]! / reference.product["gpu"]!

            func route(_ id: String) -> CapacityMeasure.Route { .init(id, engines: CapacityClasses.engines(ofRoute: id)) }
            var groups: [String: [Anchor]] = [:]
            for entry in report.map?.classes ?? [] {
                guard let (key, kind) = RouteTable.ClassKey.parse(entry.id) else { continue }
                let (group, x, ref) = Self.group(key, kind)
                let refHeat = mapHeat.factor(route(ref))
                var cold: [String: Double] = [ref: 1]
                for t in entry.timings where t.ratio > 0 && t.ratio < 1e6 { cold[t.route] = t.ratio * refHeat / mapHeat.factor(route(t.route)) }
                groups[group, default: []].append(Anchor(x: x, cold: cold))
            }

            var measured: [String: [String: Double]] = [:]
            var sources: [String: [String: Provenance]] = [:]
            for c in classes {
                let id = c.routeKey.id
                guard measured[id] == nil else { continue }          // a self- and a cross-attention share one
                let (group, x, ref) = Self.group(c.routeKey, c.weightKind ?? .bf16)
                // The map is a grid: a render's shape reads the nearest point's heights (`RouteTable.Nearest`), as its route does.
                let near = groups[group] == nil ? Self.nearest(group, among: Array(groups.keys)) : group
                let anchors = near.flatMap { groups[$0] } ?? []
                var routes: Set<String> = [ref]
                for a in anchors { routes.formUnion(a.cold.keys) }
                let base: Double
                if case .attention = c.shape { base = attentionRate } else { base = gpuRate }
                var rates: [String: Double] = [:], how: [String: Provenance] = [:]
                for r in routes {
                    let (ratio, provenance) = Self.ratio(of: r, at: x, in: anchors, reference: ref)
                    rates[r] = base / ratio
                    how[r] = near == group ? provenance : max(provenance, .interpolated)
                }
                measured[id] = rates; sources[id] = how
            }
            self.sources = sources
            rates = CapacityClasses.Rates(product: [:], attention: Dictionary(uniqueKeysWithValues: CapacityClasses.attentionRoutes.map { ($0, attentionRate) }),
                                          measured: measured, fallback: gpuRate)
        }

        /// **The measured group nearest `group`**: the same kind (a product's weight form), the nearest shape within reach.
        private static func nearest(_ group: String, among measured: [String]) -> String? {
            func parse(_ g: String) -> (kind: String, a: Int, b: Int)? {
                let parts = g.split(separator: "|").map(String.init)
                guard parts.count >= 3, let b = Int(parts[parts.count - 1]), let a = Int(parts[parts.count - 2]) else { return nil }
                return (parts.dropLast(2).joined(separator: "|"), a, b)
            }
            guard let own = parse(group) else { return nil }
            let candidates = measured.compactMap { g in parse(g).flatMap { $0.kind == own.kind ? (g, $0.a, $0.b) : nil } }
                .sorted { $0.0 < $1.0 }
            return RouteTable.Nearest.rule(among: candidates, k: own.a, n: own.b, m: 0) { (($0.1, $0.2), 0, nil) }?.0
        }

        /// The group of classes whose heights interpolate (a product: weight kind, k, n; an attention: heads, head dim), the
        /// height in play, and the class's reference route.
        private static func group(_ key: RouteTable.ClassKey, _ kind: RouteTable.WeightKind) -> (String, Double, String) {
            switch key {
            case let .product(k, n, m): return ("p|\(kind.rawValue)|\(k)|\(n)", Double(m), CapacityClasses.productRoutes[0])
            case let .attention(s, h, d): return ("a|\(h)|\(d)", Double(s), CapacityClasses.attentionRoutes[0])
            }
        }

        /// **A route's cold ratio to the reference at height `x`**: the map's own where it timed that height, the line through
        /// the two nearest heights it timed (in `ln` of both) otherwise, one height's ratio when only one was, 1 for a route it
        /// never timed.
        private static func ratio(of route: String, at x: Double, in anchors: [Anchor], reference: String) -> (Double, Provenance) {
            if route == reference { return (1, .measured) }
            let points = anchors.compactMap { a in a.cold[route].map { (x: a.x, ratio: $0) } }
            if let exact = points.first(where: { $0.x == x }) { return (exact.ratio, .measured) }
            guard !points.isEmpty else { return (1, .assumed) }
            guard points.count > 1 else { return (points[0].ratio, .interpolated) }
            let near = points.sorted { abs(log($0.x / x)) < abs(log($1.x / x)) }.prefix(2).sorted { $0.x < $1.x }
            let (a, b) = (near[0], near[1])
            let t = log(x / a.x) / log(b.x / a.x)
            return (exp((1 - t) * log(a.ratio) + t * log(b.ratio)), .interpolated)
        }
    }

    // MARK: - The prediction

    /// One class of the catalogue with the route and the seconds the map predicts for the render.
    package struct Predicted: Sendable, Equatable {
        package var work: ProductLedger.Work
        package var route: String
        package var executions: Int
        package var seconds: Double
        package var provenance: Provenance
    }

    /// **The render's request, as the catalogue sees it**: the family and the version it ran with, its pixels, the
    /// half-grid evaluations it did (`Render.spectral`), its evaluations as the steps (a sketch or an img2img did fewer),
    /// the text encoder left out (the ledger is of the denoising). `nil` for a model that is not one of the five families.
    package static func request(of render: Engine.Render, variant: Variant) -> CapacityClasses.Request? {
        guard let family = Family(rawValue: render.model) else { return nil }
        return CapacityClasses.Request(family: family, variant: variant, width: render.image.width, height: render.image.height,
                                       fasterRendering: render.spectral > 0, promptTokens: render.tokens,
                                       steps: render.evaluations, includesEncoder: false,
                                       halfSteps: render.spectral > 0 ? render.spectral : nil)
    }

    /// **The prediction, class by class** (`CapacityClasses.predict`, one class at a time so that each keeps its own route
    /// and seconds), for the denoising of `request` under `table` and the map's rates and regime. `amxMinimumRows`: the
    /// engine's `amx_min` (`Conductor.minimumRows`); `aneBroughtUp`: the render's products could take the ANE
    /// (`Render.neuralEngine`). `nil` when the version does not exist for the family.
    package static func predictions(for request: CapacityClasses.Request, table: RouteTable, report: Diagnostic.Report,
                                    amxMinimumRows: Int = EngineSettings.defaultAMXMinimumRows,
                                    aneBroughtUp: Bool = true) -> [Predicted]? {
        guard let classes = CapacityClasses.classes(for: request) else { return nil }
        let basis = Basis(classes: classes, report: report)
        var result: [Predicted] = []
        for c in classes where c.phase == .denoising {
            let one = CapacityClasses.predict([c], family: request.family, table: table, rates: basis.rates, heat: basis.heat,
                                              phases: [.denoising], amxMinimumRows: amxMinimumRows,
                                              aneBroughtUp: aneBroughtUp, weightForms: true)
            guard let part = one.parts.first else { continue }
            let work: ProductLedger.Work
            switch c.shape {
            case let .product(k, n, rows, _): work = .product(k: k, n: n, m: rows, kind: c.weightKind ?? .bf16)
            case let .attention(s, _, heads, _, dim, _): work = .attention(sequence: s, heads: heads, headDim: dim)
            }
            result.append(Predicted(work: work, route: part.route, executions: c.executions[request.family] ?? 0,
                                    seconds: part.seconds, provenance: basis.provenance(c.routeKey.id, route: part.route)))
        }
        return result
    }

    // MARK: - The comparison

    /// **One class of the render, measured and predicted.**
    package struct Row: Sendable, Equatable {
        /// `product:KxN@M` (+ `~int8`, `~q8`) or `attention:S/HxD`, at the heaviest height of the merged class.
        package var id: String
        package var isAttention: Bool
        /// The routes the executions took, the one with the most seconds first.
        package var routes: [String]
        package var executions: Int
        /// Wall seconds of the render's executions; 0 for a class the render did not run.
        package var measured: Double
        package var predictedRoute: String?
        package var predictedExecutions: Int
        /// `nil`: the catalogue does not carry this class.
        package var predicted: Double?
        package var provenance: Provenance?

        /// `measured / predicted − 1` where both exist.
        package var error: Double? {
            guard let predicted, predicted > 0, measured > 0 else { return nil }
            return measured / predicted - 1
        }
        /// Both sides: this row speaks for the map.
        package var matched: Bool { error != nil }
        /// The route the executions mostly took is the one predicted.
        package var routeAgrees: Bool { routes.first == predictedRoute }
    }

    package struct Comparison: Sendable, Equatable {
        /// Heaviest first (the larger of its measured and predicted seconds).
        package var rows: [Row]

        /// Seconds of every class the render ran, predicted or not.
        package var measured: Double { rows.reduce(0) { $0 + $1.measured } }
        /// Seconds the catalogue predicts for the render; `nil` without a map.
        package var predicted: Double? { rows.contains { $0.predicted != nil } ? rows.reduce(0) { $0 + ($1.predicted ?? 0) } : nil }
        private func sums(_ keep: (Row) -> Bool) -> (measured: Double, predicted: Double) {
            rows.filter { $0.matched && keep($0) }.reduce((0, 0)) { ($0.0 + $1.measured, $0.1 + ($1.predicted ?? 0)) }
        }
        private func gap(_ keep: (Row) -> Bool) -> Double? {
            let s = sums(keep)
            return s.predicted > 0 ? s.measured / s.predicted - 1 : nil
        }
        /// **The gap that judges the map**: measured over predicted, on the classes that have both, minus 1.
        package var error: Double? { gap { _ in true } }
        package var productError: Double? { gap { !$0.isAttention } }
        package var attentionError: Double? { gap { $0.isAttention } }
        /// The share of the measured seconds that falls on a class the catalogue carries.
        package var coverage: Double { measured > 0 ? sums { _ in true }.measured / measured : 0 }
        /// The share of the matched seconds whose route is the predicted one.
        package var routeAgreement: Double {
            let m = rows.filter(\.matched)
            let all = m.reduce(0) { $0 + $1.measured }
            return all > 0 ? m.filter(\.routeAgrees).reduce(0) { $0 + $1.measured } / all : 0
        }
    }

    private struct Shape: Hashable {
        var kind: RouteTable.WeightKind?      // nil: an attention
        var a = 0, b = 0, c = 0                // k, n, 0  |  heads, head dim, 0
    }

    private struct Item {
        var routes: [String: (executions: Int, seconds: Double)] = [:]
        var predicted: [Predicted] = []
    }

    private static func shape(of work: ProductLedger.Work) -> (Shape, Int) {
        switch work {
        case let .product(k, n, m, kind): return (Shape(kind: kind, a: k, b: n), m)
        case let .attention(s, h, d): return (Shape(kind: nil, a: h, b: d), s)
        }
    }

    /// **The render's ledger lined up with the prediction**, heights merged within `heightGap` (`CapacityMap.heightGap`).
    /// `predicted`: empty without a map, and every row then has measured seconds alone.
    package static func compare(_ tallies: [ProductLedger.Tally], predicted: [Predicted] = [],
                                heightGap: Int = CapacityMap.heightGap) -> Comparison {
        var items: [Shape: [Int: Item]] = [:]
        for t in tallies {
            let (shape, x) = shape(of: t.work)
            var item = items[shape]?[x] ?? Item()
            let kept = item.routes[t.route] ?? (0, 0)
            item.routes[t.route] = (kept.executions + t.executions, kept.seconds + t.seconds)
            items[shape, default: [:]][x] = item
        }
        for p in predicted {
            let (shape, x) = shape(of: p.work)
            var item = items[shape]?[x] ?? Item()
            item.predicted.append(p)
            items[shape, default: [:]][x] = item
        }

        var rows: [Row] = []
        for (shape, heights) in items {
            // Chains of heights within the gap (products); an attention's S is exact.
            var clusters: [[Int]] = []
            for x in heights.keys.sorted() {
                if shape.kind != nil, let last = clusters.last?.last, x - last <= heightGap { clusters[clusters.count - 1].append(x) }
                else { clusters.append([x]) }
            }
            for cluster in clusters {
                var routes: [String: (executions: Int, seconds: Double)] = [:]
                var parts: [Predicted] = []
                for x in cluster {
                    for (route, v) in heights[x]!.routes {
                        let kept = routes[route] ?? (0, 0)
                        routes[route] = (kept.executions + v.executions, kept.seconds + v.seconds)
                    }
                    parts += heights[x]!.predicted
                }
                // The class is named by the member that weighs most, prediction and measure together.
                func weight(_ x: Int) -> Double {
                    heights[x]!.routes.values.reduce(0) { $0 + $1.seconds } + heights[x]!.predicted.reduce(0) { $0 + $1.seconds }
                }
                let heaviest = cluster.max { (weight($0), $1) < (weight($1), $0) }!
                let id: String
                if let kind = shape.kind {
                    id = RouteTable.ClassKey.product(k: shape.a, n: shape.b, m: heaviest).id + RouteTable.ClassKey.suffix(kind)
                } else {
                    id = RouteTable.ClassKey.attention(sequence: heaviest, heads: shape.a, headDim: shape.b).id
                }
                let ordered = routes.sorted { ($1.value.seconds, $0.key) < ($0.value.seconds, $1.key) }
                let main = parts.max { ($0.seconds, $1.route) < ($1.seconds, $0.route) }
                rows.append(Row(id: id, isAttention: shape.kind == nil, routes: ordered.map(\.key),
                                executions: routes.values.reduce(0) { $0 + $1.executions },
                                measured: routes.values.reduce(0) { $0 + $1.seconds },
                                predictedRoute: main?.route,
                                predictedExecutions: parts.reduce(0) { $0 + $1.executions },
                                predicted: parts.isEmpty ? nil : parts.reduce(0) { $0 + $1.seconds },
                                provenance: parts.map(\.provenance).max()))
            }
        }
        rows.sort { a, b in
            let (x, y) = (max(a.measured, a.predicted ?? 0), max(b.measured, b.predicted ?? 0))
            return x != y ? x > y : a.id < b.id
        }
        return Comparison(rows: rows)
    }

    // MARK: - As text

    /// **The comparison as a table a script can read**: a title line, then the header `#class` and one line per class, columns
    /// separated by tabs, then `#total`, `#products`, `#attention` and `#coverage` lines. Seconds are wall seconds; `error` is
    /// `measured / predicted − 1`, empty where either side is missing; `basis` says how the rate was known.
    package static func lines(_ comparison: Comparison, title: String) -> [String] {
        func s(_ x: Double?) -> String { x.map { String(format: "%.3f", $0) } ?? "" }
        func e(_ x: Double?) -> String { x.map { String(format: "%+.1f%%", $0 * 100) } ?? "" }
        var out = [title, ["#class", "route", "executions", "measured_s", "predicted_s", "error", "predicted_route", "basis"].joined(separator: "\t")]
        for r in comparison.rows {
            out.append([r.id, r.routes.joined(separator: "|"), "\(r.executions)", s(r.measured), s(r.predicted), e(r.error),
                        r.predictedRoute ?? "", r.provenance?.rawValue ?? ""].joined(separator: "\t"))
        }
        out.append(["#total", "", "\(comparison.rows.reduce(0) { $0 + $1.executions })", s(comparison.measured),
                    s(comparison.predicted), e(comparison.error), "", ""].joined(separator: "\t"))
        out.append(["#products", "", "", "", "", e(comparison.productError), "", ""].joined(separator: "\t"))
        out.append(["#attention", "", "", "", "", e(comparison.attentionError), "", ""].joined(separator: "\t"))
        out.append(["#coverage", "", "", "", "", String(format: "%.1f%%", comparison.coverage * 100), "",
                    String(format: "routes agree on %.1f%% of the matched seconds", comparison.routeAgreement * 100)].joined(separator: "\t"))
        return out
    }
}
