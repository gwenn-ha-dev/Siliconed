import Foundation

/// **What a render's gap to the map's prediction does**: nothing while the map holds, a correction of the regime when
/// the gap is one an engine's heat explains, a new map when it is not — and never during the session, for the bits' sake:
/// the verdict is kept and applied at the next launch (`Diagnostic.Report.applying`).
///
///     comparison (`RenderLoop.compare`)
///        │   coverage < 80 %                                 →  unjudged       the catalogue speaks for too little of it
///        │   |gap| ≤ 10 %                                    →  held           the map is kept as it is
///        │   |gap| > 40 %                                    →  remeasure      no regime explains a render 40 % off
///        │   else, per engine, the factor that makes its routes' predicted seconds the measured ones:
///        │        gpu     the classes on the GPU alone (graph, flash, gpu/bf16…)                  f = Σ measured / Σ predicted
///        │        amx     the classes that route to the AMX, if the AMX is the engine the route waits for   f = T / heat
///        │        ane, cpu  likewise                                (a route runs at the speed of its slowest engine)
///        │      a residual gap above 10 % after it                →  remeasure      it is not the regime
///        │      a class whose route the corrected regime would change, losses adding up past the map's ε (2 %)
///        │                                                       →  remeasure      the cuts are the map's, not a factor's
///        └─ otherwise                                          →  correct(f per engine)
///
/// What sets it apart — what a threshold written by analogy with "if the render is slower than predicted, measure again" would get wrong:
///
///   - **The map is judged on what it speaks for**: the matched classes (both sides), `Comparison.error`; the seconds the catalogue
///     does not carry (a fusion, an adapter) neither blame nor absolve it, and below 80 % coverage the check says nothing
///     (`unjudged`) — measuring again would not change the render's model.
///   - **A factor is per engine, and an engine hidden behind another has none.** A route's heat is the largest of its engines'
///     (`CapacityMeasure.Heat`): a `gpu+amx` class that runs 20 % slow tells the AMX's heat only if the AMX's heat is what the
///     route waits for; if the GPU's, corrected by the GPU-alone classes, already exceeds the target, the AMX's factor is left
///     alone. A class whose route is not the predicted one (the ANE refused, a setting forced) says nothing about a regime.
///   - **The correction moves the prediction, not the routes.** The map's table was decided at the map's heat; the corrected
///     regime is checked against the map's own ratios (`CapacityMeasure.rule`, as the plan applies it): if it would pick another
///     route for classes weighing more than ε together, it is the map that is out of date, and only a measure re-cuts.
///   - **The thresholds are first guesses, to be calibrated on real renders** (10 % for the held band is five times the map's ε
///     because the map stores ratios and the GPU witness's rate — a class's own GPU rate differs by shape — and not a time per class).
extension RenderLoop {
    package enum Reason: String, Sendable, Codable, Equatable {
        /// The gap is beyond what a regime explains, or what a regime does not explain stays.
        case gap
        /// The corrected regime would change routes whose classes weigh more than ε.
        case decisions
        /// The catalogue carries too little of what the render ran (or the render left no record).
        case uncovered
        /// No map to compare with, or none of its classes can be read.
        case noMap
    }

    package enum Verdict: Sendable, Equatable {
        /// The map predicted the render within the tolerance: kept as it is.
        case held
        /// A regime that explains the gap: multiply each engine's heat by its factor, at the next launch.
        case correct([String: Double])
        /// A map to measure again at the next launch.
        case remeasure(Reason)
        /// The record cannot judge the map (not an error of the map's).
        case unjudged(Reason)

        package var name: String {
            switch self {
            case .held: return "held"
            case .correct: return "correct"
            case let .remeasure(r): return "remeasure:" + r.rawValue
            case let .unjudged(r): return "unjudged:" + r.rawValue
            }
        }
    }

    /// The thresholds of the rule (shares and gaps are fractions).
    package struct Thresholds: Sendable, Equatable {
        /// The gap within which the map holds.
        package var tolerance = 0.10
        /// The gap past which no regime is looked for.
        package var remeasureAbove = 0.40
        /// The share of the render the matched classes must carry for the check to speak.
        package var minimumCoverage = 0.80
        /// The share of the predicted seconds an engine's routes need to give it a factor of its own.
        package var identifiable = 0.05
        package init() {}
    }

    // MARK: - The rule

    /// **The verdict on a comparison**, against the map it was predicted from.
    package static func verdict(_ comparison: Comparison, report: Diagnostic.Report, table: RouteTable? = nil,
                                thresholds t: Thresholds = Thresholds()) -> Verdict {
        guard report.map?.classes.isEmpty == false else { return .unjudged(.noMap) }
        guard let gap = comparison.error, comparison.coverage >= t.minimumCoverage else { return .unjudged(.uncovered) }
        if abs(gap) <= t.tolerance { return .held }
        if abs(gap) > t.remeasureAbove { return .remeasure(.gap) }

        let factors = engineFactors(comparison, report: report, thresholds: t)
        guard !factors.isEmpty else { return .remeasure(.gap) }
        // What is left of the gap once the regime is corrected.
        let before = Basis.regime(of: report)
        var after = before
        for (engine, f) in factors { after[engine, default: 1] *= f }
        var predicted = 0.0, measured = 0.0
        for r in comparison.rows where r.matched && r.routeAgrees {
            let route = r.predictedRoute!
            predicted += r.predicted! * routeHeat(route, after) / routeHeat(route, before)
            measured += r.measured
        }
        let skipped = comparison.rows.filter { $0.matched && !$0.routeAgrees }
        predicted += skipped.reduce(0) { $0 + $1.predicted! }
        measured += skipped.reduce(0) { $0 + $1.measured }
        guard predicted > 0, abs(measured / predicted - 1) <= t.tolerance else { return .remeasure(.gap) }
        if decisionsLoss(report: report, to: after) > (report.map?.epsilon ?? 0.02) { return .remeasure(.decisions) }
        return .correct(factors)
    }

    private static func routeHeat(_ route: String, _ factors: [String: Double]) -> Double {
        CapacityClasses.engines(ofRoute: route).map { factors[$0] ?? 1 }.max() ?? 1
    }

    /// **The factor of each engine** whose routes the render timed: what its heat must be multiplied by for those routes'
    /// predicted seconds to be the measured ones. Engines in a fixed order (GPU, AMX, CPU, ANE), each seeing the factors already found.
    static func engineFactors(_ comparison: Comparison, report: Diagnostic.Report, thresholds t: Thresholds) -> [String: Double] {
        let rows = comparison.rows.filter { $0.matched && $0.routeAgrees }
        let total = rows.reduce(0) { $0 + $1.predicted! }
        guard total > 0 else { return [:] }
        let base = Basis.regime(of: report)           // the regime the prediction was made at
        var now = base                                 // … with the factors found so far
        var factors: [String: Double] = [:]
        func engines(_ r: Row) -> [String] { CapacityClasses.engines(ofRoute: r.predictedRoute!) }

        // The GPU alone: its classes' heat is the GPU's.
        let alone = rows.filter { engines($0) == ["gpu"] }
        let aloneSeconds = alone.reduce(0) { $0 + $1.predicted! }
        if aloneSeconds >= t.identifiable * total {
            let f = alone.reduce(0) { $0 + $1.measured } / aloneSeconds
            now["gpu", default: 1] *= f
            factors["gpu"] = f
        }
        // Another engine: among the sets of engines that use it, the one with the most predicted seconds. Its routes' heat
        // was the largest of the set's; it becomes that times the gap — and is the engine's if it exceeds the others'.
        for engine in ["amx", "cpu", "ane"] {
            let sets = Dictionary(grouping: rows.filter { engines($0).contains(engine) }) { engines($0).sorted() }
            guard let (set, group) = sets.max(by: { $0.value.reduce(0) { $0 + $1.predicted! } < $1.value.reduce(0) { $0 + $1.predicted! } }),
                  group.reduce(0, { $0 + $1.predicted! }) >= t.identifiable * total else { continue }
            let measured = group.reduce(0) { $0 + $1.measured }, predicted = group.reduce(0) { $0 + $1.predicted! }
            let target = (set.map { base[$0] ?? 1 }.max() ?? 1) * measured / predicted
            // Within 1 % of another engine's heat the route waits for that one: nothing to tell.
            guard target > 1.01 * (set.filter { $0 != engine }.map { now[$0] ?? 1 }.max() ?? 1) else { continue }
            factors[engine] = target / (base[engine] ?? 1)
            now[engine] = target
        }
        return factors.filter { abs($0.value - 1) > 0.01 }.mapValues { Diagnostic.significant($0, 4) }
    }

    /// **What the corrected regime would cost in routes** (fraction of a render): the map's classes whose chosen route is no
    /// longer the plan's rule's choice on their hot ratios moved from the map's regime to `after`, each by its weight times the gap.
    static func decisionsLoss(report: Diagnostic.Report, to after: [String: Double]) -> Double {
        let map = report.regime?.heat ?? [:]
        var loss = 0.0
        for entry in report.map?.classes ?? [] where entry.status == .closed {
            guard let chosen = entry.chosen, let (key, _) = RouteTable.ClassKey.parse(entry.id) else { continue }
            let ref: String
            if case .product = key { ref = CapacityClasses.productRoutes[0] } else { ref = CapacityClasses.attentionRoutes[0] }
            let routes = [ref] + entry.timings.map(\.route)
            // Hot time of each route over the reference's: stored at the map's regime, moved to the corrected one.
            let ratios: [Double?] = zip(routes, [1.0] + entry.timings.map(\.ratio)).map { route, ratio in
                ratio * (routeHeat(route, after) / routeHeat(route, map)) / (routeHeat(ref, after) / routeHeat(ref, map))
            }
            let efficient = routes.map { $0.hasPrefix("gpu+ane") }
            guard let best = CapacityMeasure.rule(ratios: ratios, efficient: efficient), routes[best] != chosen,
                  let at = routes.firstIndex(of: chosen), let kept = ratios[at], let better = ratios[best], better > 0 else { continue }
            loss += entry.weight * max(0, kept / better - 1)
        }
        return loss
    }

    // MARK: - What is kept, and what the report carries

    /// **A render's note against its map**: what `Report my configuration` carries (the heaviest classes, one array each, to stay
    /// within the link's bytes) and what the next launch applies (the verdict, and the factors of a correction).
    package struct Note: Codable, Sendable, Equatable {
        package var family: String
        package var variant: String
        package var format: String
        package var faster: Bool
        package var evaluations: Int
        /// The date of the map predicted from (`Report.date`).
        package var map: String
        package var measured: Double
        package var predicted: Double?
        package var error: Double?
        package var productError: Double?
        package var attentionError: Double?
        package var coverage: Double
        package var verdict: String
        package var factors: [String: Double]?
        package var classes: [Line]

        /// `[id, routes, executions, measured, predicted route, predicted, error]`.
        package struct Line: Codable, Sendable, Equatable {
            package var id: String
            package var route: String
            package var executions: Int
            package var measured: Double
            package var predictedRoute: String?
            package var predicted: Double?
            package var error: Double?

            package init(id: String, route: String, executions: Int, measured: Double, predictedRoute: String?, predicted: Double?, error: Double?) {
                self.id = id; self.route = route; self.executions = executions
                self.measured = Diagnostic.significant(measured, 3); self.predictedRoute = predictedRoute
                self.predicted = predicted.map { Diagnostic.significant($0, 3) }; self.error = error.map { Diagnostic.significant($0, 2) }
            }
            package init(from decoder: any Decoder) throws {
                var c = try decoder.unkeyedContainer()
                id = try c.decode(String.self); route = try c.decode(String.self); executions = try c.decode(Int.self)
                measured = try c.decode(Double.self); predictedRoute = try c.decode(String?.self)
                predicted = try c.decode(Double?.self); error = try c.decode(Double?.self)
            }
            package func encode(to encoder: any Encoder) throws {
                var c = encoder.unkeyedContainer()
                try c.encode(id); try c.encode(route); try c.encode(executions); try c.encode(measured)
                try c.encode(predictedRoute); try c.encode(predicted); try c.encode(error)
            }
        }

        /// How many classes the note keeps: the link's bytes are counted.
        package static let linkClasses = 10

        package init(request: CapacityClasses.Request, map: String, comparison: Comparison, verdict: Verdict) {
            family = request.family.rawValue; variant = request.variant.rawValue
            format = "\(request.width)x\(request.height)"; faster = request.fasterRendering; evaluations = request.steps
            self.map = map
            measured = Diagnostic.significant(comparison.measured, 4)
            predicted = comparison.predicted.map { Diagnostic.significant($0, 4) }
            error = comparison.error.map { Diagnostic.significant($0, 2) }
            productError = comparison.productError.map { Diagnostic.significant($0, 2) }
            attentionError = comparison.attentionError.map { Diagnostic.significant($0, 2) }
            coverage = Diagnostic.significant(comparison.coverage, 3)
            self.verdict = verdict.name
            if case let .correct(f) = verdict { factors = f } else { factors = nil }
            classes = comparison.rows.prefix(Self.linkClasses).map {
                Line(id: $0.id, route: $0.routes.first ?? "", executions: $0.executions, measured: $0.measured,
                     predictedRoute: $0.predictedRoute, predicted: $0.predicted, error: $0.error)
            }
        }

        /// The verdict this note keeps, read back.
        package var decision: Verdict? {
            if verdict == "held" { return .held }
            if verdict == "correct" { return factors.map { .correct($0) } }
            let parts = verdict.split(separator: ":").map(String.init)
            guard parts.count == 2, let reason = Reason(rawValue: parts[1]) else { return nil }
            return parts[0] == "remeasure" ? .remeasure(reason) : parts[0] == "unjudged" ? .unjudged(reason) : nil
        }
    }

    /// **The whole check of a render**, from its record to its note: the request it stands for, the prediction under `table`,
    /// the comparison, the verdict. `nil` for a render the catalogue does not describe (an unknown family, a version that does not
    /// exist, a sketch, a render with references or a LoRA: classes the catalogue does not carry).
    package static func check(_ render: Engine.Render, variant: Variant, references: Int = 0, report: Diagnostic.Report, table: RouteTable,
                              amxMinimumRows: Int = EngineSettings.defaultAMXMinimumRows,
                              thresholds: Thresholds = Thresholds()) -> (comparison: Comparison, verdict: Verdict, note: Note)? {
        guard !render.classes.isEmpty, render.sketch == nil, render.loras.isEmpty, references == 0,
              let request = request(of: render, variant: variant),
              let predicted = predictions(for: request, table: table, report: report, amxMinimumRows: amxMinimumRows,
                                          aneBroughtUp: render.neuralEngine) else { return nil }
        let comparison = compare(render.classes, predicted: predicted)
        let verdict = verdict(comparison, report: report, table: table, thresholds: thresholds)
        return (comparison, verdict, Note(request: request, map: report.date, comparison: comparison, verdict: verdict))
    }
}

extension RenderLoop {
    /// **What the next launch does with the stored map and the last render's note** (never in the session that rendered).
    /// `map`: the map to use, with a correction applied (`applying`) when the note made against it says `correct`; `remeasure`: the
    /// note made against it asks for a new map — only once the Mac is `atRest`, else the note waits; `consumed`: the note has been
    /// acted on and is to be dropped (a note of another map, or a held one, is dropped too: it speaks of nothing that is kept).
    package static func launching(_ map: Diagnostic.Report, note: Note?, atRest: Bool)
        -> (map: Diagnostic.Report, remeasure: Bool, consumed: Bool) {
        guard let note else { return (map, false, false) }
        guard note.map == map.date, let decision = note.decision else { return (map, false, true) }
        switch decision {
        case .correct: return (map.applying(note), false, true)
        case .remeasure: return atRest ? (map, true, true) : (map, false, false)
        case .held, .unjudged: return (map, false, true)
        }
    }
}

extension RenderLoop.Basis {
    /// The map's heat times what was learned, as a dictionary.
    static func regime(of report: Diagnostic.Report) -> [String: Double] {
        var now = report.regime?.heat ?? [:]
        for (engine, factor) in report.regime?.learned ?? [:] { now[engine, default: 1] *= factor }
        return now
    }
}

extension Diagnostic.Report {
    /// **The map with a render's verdict applied** — at the next launch, never in the session that rendered: a `correct` multiplies
    /// the learned regime by its factors (the routes stay, the prediction moves); anything else leaves the map as it is (a
    /// `remeasure` is the app's to act on, before the settings are read). Only a note made against this very map applies.
    package func applying(_ note: RenderLoop.Note) -> Diagnostic.Report {
        guard note.map == date, case let .correct(factors)? = note.decision, var regime else { return self }
        var learned = regime.learned ?? [:]
        for (engine, f) in factors { learned[engine, default: 1] = Diagnostic.significant(learned[engine, default: 1] * f, 4) }
        regime.learned = learned
        var corrected = self
        corrected.regime = regime
        return corrected
    }

    /// The same map, its last render's note attached (what « Report my configuration » sends).
    package func with(render note: RenderLoop.Note?) -> Diagnostic.Report {
        var report = self
        report.render = note
        return report
    }
}
