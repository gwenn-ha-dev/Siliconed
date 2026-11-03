import Foundation

/// **The render checks the plan: what each class cost, against what the model predicted for the assignment the engine ran.** Pure — the
/// render's side is the ledger (`ProductLedger`, `Engine.Render.classes`), the plan's side is the answer it gave each class
/// (`CapacityPlan.assign`) timed by the model that solved it (`CapacitySolver.product(assigned:…)`); this file lines them up class by
/// class and says how far apart they are.
///
///     Render.classes ─▶ compare ◀── predict ◀── CapacityClasses.classes(request)  (the catalogue's classes of this render)
///      (k×n@m, route,                │             each class: the plan's assignment, timed by the model
///       executions, seconds)         │               seconds = executions · compute   (the solver's own number, as the ledger times the call)
///                                    ▼
///                  rows (class, routes, executions, measured, predicted, error) · totals · coverage
///                                    │
///                                    ▼
///                  verdict:  held  ·  drift  ·  unjudged                              (`RenderLoopVerdict.swift`)
///
/// **The loop reports, it never steers**: the gap goes into the render's record (the note, the journal), and the plan stays what the
/// model made it — two renders of one seed under one measure are one image, whatever the first one measured. A model that is out of date is a
/// measure to take again (« Measure This Mac »), not a factor to apply between two renders.
///
/// What sets it apart — what a comparison written by analogy with "sum the census, divide by the model" would get wrong:
///
///   - **The prediction is the call's, as the ledger times it**: the product's or the attention's compute, not the weight's preparation
///     (its widening or staging is timed apart, before the call: `Render.timings`), so that both sides measure the same span.
///   - **The prediction is the plan's, not the best**: each class is timed under the assignment the engine was given — a forced plan, a
///     process without the ANE, a Standard with its fp32 products — and not under the one the solver would pick for a process that
///     could have anything. A class whose assignment the model cannot time (a kernel it excluded or never measured) is a row of its
///     own with no prediction.
///   - **Heights are merged within `heightGap` rows (64)** of a neighbour of the same weight format, `k` and `n`: Z-Image's 4 096
///     (noise refiner) and 4 128 (layers) are one class of the render for the comparison, and so is the prediction of the 4 117 a
///     prompt of another length makes. The class is named by its heaviest member's height. An attention is exact (S, heads, head
///     dim), a self- and a cross-attention of Anima adding up — the ledger cannot tell them apart.
///   - **A class that ran and was not predicted, or the other way round, is a row of its own**: Krea 2's text fusion, the
///     adapters, a tile the catalogue does not carry. The totals count them (a render is what it ran); the error that judges the
///     model is the matched one, and `coverage` says how much of the render it speaks for.
///   - **The predicted route is the one the engine will take, weight form included** (`gpu/bf16` where the plan reads the bf16 weight
///     as published, `gpu+amx/int8` where it takes columns of the 8-bit one): the ANE only if the process brought it up
///     (`Render.neuralEngine`), the plan restricted accordingly.
///   - **The attentions are timed as the products are**: wall seconds around the kernel's `run` (the three kernels record).
///     An encoder's attention, the VAE, the modulation gemv and the elementwise kernels are not in the ledger — nor in the
///     catalogue — so the sums are of the classes, never of the render's denoising seconds (which hold those too).
package enum RenderLoop {
    /// Rows within which two heights of the same k×n are one class of the comparison.
    package static let heightGap = 64

    // MARK: - The prediction

    /// One class of the catalogue with the route and the seconds the plan's assignment is predicted to take for the render.
    package struct Predicted: Sendable, Equatable {
        package var work: ProductLedger.Work
        package var route: String
        package var executions: Int
        package var seconds: Double
    }

    /// **The render's request, as the catalogue sees it**: the family and the version it ran with, its pixels, the
    /// half-grid evaluations it did (`Render.spectral`), its evaluations as the steps (a sketch or an img2img did fewer),
    /// the text encoder left out (the ledger is of the denoising). `nil` for a model that is not one of the five families.
    package static func request(of render: Engine.Render, variant: Variant) -> CapacityClasses.Request? {
        guard let family = Family(rawValue: render.model) else { return nil }
        return CapacityClasses.Request(family: family, variant: variant, width: render.image.width, height: render.image.height,
                                       fasterRendering: render.spectral > 0,
                                       promptTokens: CapacityClasses.promptTokens(family, renderTokens: render.tokens),
                                       steps: render.evaluations, includesEncoder: false,
                                       halfSteps: render.spectral > 0 ? render.spectral : nil, loraRank: render.loraRank)
    }

    /// **The prediction, class by class**, for the denoising of `request` under `plan`: the assignment the plan gives each class, timed by
    /// the plan's model. `aneBroughtUp`: the render's products could take the ANE (`Render.neuralEngine`); `accelerators`: false, the
    /// render computed in fp32 throughout (`Render.accelerators`). `nil` when the version does not exist for the family, or the plan has no
    /// model (the engine's defaults: nothing measured, nothing to predict from).
    package static func predictions(for request: CapacityClasses.Request, plan: CapacityPlan, aneBroughtUp: Bool = true,
                                    accelerators: Bool = true) -> [Predicted]? {
        guard let model = plan.model, let classes = CapacityClasses.classes(for: request) else { return nil }
        let plan = plan.restricted(accelerators: accelerators).restricted(allowingANE: aneBroughtUp)
        let constraints = plan.constraints
        var result: [Predicted] = []
        for c in classes where c.phase == .denoising {
            let executions = c.executions[request.family] ?? 0
            guard executions > 0 else { continue }
            switch c.shape {
            case let .product(k, n, rows, format):
                let key = ProductKey(k: k, n: n, m: rows, weight: format, phase: c.phase)
                guard let s = CapacitySolver.product(assigned: plan.assign(key), key: key, model: model, constraints: constraints) else { continue }
                result.append(Predicted(work: .product(k: k, n: n, m: rows, weight: format), route: s.route, executions: executions,
                                        seconds: Double(executions) * s.compute))
            case let .attention(s, keys, heads, kv, dim, cross):
                let key = AttentionKey(sequence: s, keys: keys, heads: heads, kvHeads: kv, headDim: dim, cross: cross)
                let flash = CapacitySolver.triesFlash(family: request.family, cross: cross)
                guard let a = CapacitySolver.attention(assigned: plan.assign(key, flash: flash), key: key, model: model, constraints: constraints)
                else { continue }
                result.append(Predicted(work: .attention(sequence: s, heads: heads, headDim: dim), route: a.route, executions: executions,
                                        seconds: Double(executions) * a.compute))
            }
        }
        return result
    }

    // MARK: - The comparison

    /// **One class of the render, measured and predicted.**
    package struct Row: Sendable, Equatable {
        /// `product:KxN@M` (+ `~int8`, `~q8`, `~q4k`) or `attention:S/HxD`, at the heaviest height of the merged class.
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

        /// `measured / predicted − 1` where both exist.
        package var error: Double? {
            guard let predicted, predicted > 0, measured > 0 else { return nil }
            return measured / predicted - 1
        }
        /// Both sides: this row speaks for the model.
        package var matched: Bool { error != nil }
        /// The route the executions mostly took is the one predicted.
        package var routeAgrees: Bool { routes.first == predictedRoute }
    }

    package struct Comparison: Sendable, Equatable {
        /// Heaviest first (the larger of its measured and predicted seconds).
        package var rows: [Row]

        /// Seconds of every class the render ran, predicted or not.
        package var measured: Double { rows.reduce(0) { $0 + $1.measured } }
        /// Seconds the catalogue predicts for the render; `nil` without a model.
        package var predicted: Double? { rows.contains { $0.predicted != nil } ? rows.reduce(0) { $0 + ($1.predicted ?? 0) } : nil }
        private func sums(_ keep: (Row) -> Bool) -> (measured: Double, predicted: Double) {
            rows.filter { $0.matched && keep($0) }.reduce((0, 0)) { ($0.0 + $1.measured, $0.1 + ($1.predicted ?? 0)) }
        }
        private func gap(_ keep: (Row) -> Bool) -> Double? {
            let s = sums(keep)
            return s.predicted > 0 ? s.measured / s.predicted - 1 : nil
        }
        /// **The gap that judges the model**: measured over predicted, on the classes that have both, minus 1.
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
        var weight: CapacityClasses.WeightFormat?      // nil: an attention
        var a = 0, b = 0                                // k, n  |  heads, head dim
    }

    private struct Item {
        var routes: [String: (executions: Int, seconds: Double)] = [:]
        var predicted: [Predicted] = []
    }

    private static func shape(of work: ProductLedger.Work) -> (Shape, Int) {
        switch work {
        case let .product(k, n, m, weight): return (Shape(weight: weight, a: k, b: n), m)
        case let .attention(s, h, d): return (Shape(weight: nil, a: h, b: d), s)
        }
    }

    /// **The render's ledger lined up with the prediction**, heights merged within `heightGap`.
    /// `predicted`: empty without a model, and every row then has measured seconds alone.
    package static func compare(_ tallies: [ProductLedger.Tally], predicted: [Predicted] = [], heightGap: Int = heightGap) -> Comparison {
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
                if shape.weight != nil, let last = clusters.last?.last, x - last <= heightGap { clusters[clusters.count - 1].append(x) }
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
                if let format = shape.weight {
                    id = CapacityClasses.productID(k: shape.a, n: shape.b, m: heaviest, format: format)
                } else {
                    id = CapacityClasses.attentionID(sequence: heaviest, heads: shape.a, headDim: shape.b)
                }
                let ordered = routes.sorted { ($1.value.seconds, $0.key) < ($0.value.seconds, $1.key) }
                let main = parts.max { ($0.seconds, $1.route) < ($1.seconds, $0.route) }
                rows.append(Row(id: id, isAttention: shape.weight == nil, routes: ordered.map(\.key),
                                executions: routes.values.reduce(0) { $0 + $1.executions },
                                measured: routes.values.reduce(0) { $0 + $1.seconds },
                                predictedRoute: main?.route,
                                predictedExecutions: parts.reduce(0) { $0 + $1.executions },
                                predicted: parts.isEmpty ? nil : parts.reduce(0) { $0 + $1.seconds }))
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
    /// separated by tabs, then `#total`, `#products`, `#attention` and `#coverage` lines, then one `#fallback` line per product that
    /// ran otherwise than planned (`ProductLedger.Fallback`: class, executions, the plan's route, the reason) and a `#fallbacks` count.
    /// Seconds are wall seconds; `error` is `measured / predicted − 1`, empty where either side is missing.
    package static func lines(_ comparison: Comparison, title: String, fallbacks: [ProductLedger.Fallback] = []) -> [String] {
        func s(_ x: Double?) -> String { x.map { String(format: "%.3f", $0) } ?? "" }
        func e(_ x: Double?) -> String { x.map { String(format: "%+.1f%%", $0 * 100) } ?? "" }
        var out = [title, ["#class", "route", "executions", "measured_s", "predicted_s", "error", "predicted_route"].joined(separator: "\t")]
        for r in comparison.rows {
            out.append([r.id, r.routes.joined(separator: "|"), "\(r.executions)", s(r.measured), s(r.predicted), e(r.error),
                        r.predictedRoute ?? ""].joined(separator: "\t"))
        }
        out.append(["#total", "", "\(comparison.rows.reduce(0) { $0 + $1.executions })", s(comparison.measured),
                    s(comparison.predicted), e(comparison.error), ""].joined(separator: "\t"))
        out.append(["#products", "", "", "", "", e(comparison.productError), ""].joined(separator: "\t"))
        out.append(["#attention", "", "", "", "", e(comparison.attentionError), ""].joined(separator: "\t"))
        out.append(["#coverage", "", "", "", "", String(format: "%.1f%%", comparison.coverage * 100),
                    String(format: "routes agree on %.1f%% of the matched seconds", comparison.routeAgreement * 100)].joined(separator: "\t"))
        for f in fallbacks {
            let id: String
            switch f.work {
            case let .product(k, n, m, weight): id = CapacityClasses.productID(k: k, n: n, m: m, format: weight)
            case let .attention(s, h, d): id = CapacityClasses.attentionID(sequence: s, heads: h, headDim: d)
            }
            out.append(["#fallback", id, "\(f.executions)", f.planned, f.reason.rawValue].joined(separator: "\t"))
        }
        out.append(["#fallbacks", "", "\(fallbacks.reduce(0) { $0 + $1.executions })", "", ""].joined(separator: "\t"))
        return out
    }
}
