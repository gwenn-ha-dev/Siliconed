import Foundation

/// **What a render's gap to the plan's prediction says**: nothing while the model holds, that it drifted when it does not, that it
/// cannot judge when the catalogue speaks for too little of the render — and never a change of the plan. The verdict is written in the
/// render's record (the note, the journal, the report) so that a drifting model is *seen* by whoever reads the reports; it changes no
/// plan, because the bits of an image must not depend on what an earlier render measured. The app does not propose to measure again on
/// a drift: the model's known gap — a long render's regime read off the probes' short passes — is the same on every Mac, and a new
/// measure could not close it (the user still can, « Measure This Mac », and it applies from the next render).
///
///     comparison (`RenderLoop.compare`)
///        │   no model, or no class matched                   →  unjudged(noModel)
///        │   coverage < 80 %                                 →  unjudged(uncovered)   the catalogue speaks for too little of it
///        │   |gap| ≤ 10 %                                    →  held
///        └─ otherwise                                        →  drift                 the render is slower or faster than the model says
///
/// What sets it apart — what a threshold written by analogy with "if the render is slower than predicted, measure again" would get wrong:
///
///   - **The model is judged on what it speaks for**: the matched classes (both sides), `Comparison.error`; the seconds the catalogue
///     does not carry (a fusion, an adapter) neither blame nor absolve it, and below 80 % coverage the check says nothing
///     (`unjudged`) — measuring again would not change the render's model.
///   - **The gap is read by block as well as in all** (`productError`, `attentionError`, and each class's row in the note): the lines were
///     fitted on products and attentions of a few sizes and carry the prediction of every height the render makes. A gap that sits on
///     one class names the line to look at; the same gap everywhere is the regime (heat, a busy Mac).
///   - **The thresholds are first guesses, to be calibrated on real renders** (10 % for the held band: the lines hold a product within
///     2–6 % of a long measure, and a class's own GPU rate varies by shape).
extension RenderLoop {
    package enum Reason: String, Sendable, Codable, Equatable {
        /// The catalogue carries too little of what the render ran (or the render left no record).
        case uncovered
        /// No model to predict from: the engine ran on its defaults.
        case noModel
    }

    package enum Verdict: Sendable, Equatable {
        /// The model predicted the render within the tolerance.
        case held
        /// The render ran off the model by more than the tolerance (the note says by how much, class by class).
        case drift
        /// The record cannot judge the model (not an error of the model's).
        case unjudged(Reason)

        package var name: String {
            switch self {
            case .held: return "held"
            case .drift: return "drift"
            case let .unjudged(r): return "unjudged:" + r.rawValue
            }
        }
    }

    /// The thresholds of the rule (shares and gaps are fractions).
    package struct Thresholds: Sendable, Equatable {
        /// The gap within which the model holds.
        package var tolerance = 0.10
        /// The share of the render the matched classes must carry for the check to speak.
        package var minimumCoverage = 0.80
        package init() {}

        /// **The band a map's own uncertainty widens.** A prediction read on the quick side of identical calls that disagreed by ±x %
        /// (`Report.uncertainty`) is not held to better than x %: a render beside a busy Mac is not a drifting model. A map of a free Mac
        /// (±1–5 %) keeps the 10 %.
        package init(uncertainty report: Diagnostic.Report) {
            self.init()
            tolerance = max(tolerance, report.uncertainty.map(\.plusMinus).max() ?? 0)
        }
    }

    // MARK: - The rule

    /// **The verdict on a comparison.** The tolerance is the band's 10 %, wider where the map says its own figures are less sure
    /// (`Thresholds(uncertainty:)`).
    package static func verdict(_ comparison: Comparison, thresholds t: Thresholds = Thresholds()) -> Verdict {
        guard comparison.predicted != nil else { return .unjudged(.noModel) }
        guard let gap = comparison.error, comparison.coverage >= t.minimumCoverage else { return .unjudged(.uncovered) }
        return abs(gap) <= t.tolerance ? .held : .drift
    }

    // MARK: - What is kept, and what the report carries

    /// **A render's note against its model**: what `Report my configuration` carries (the heaviest classes, one array each) and what the
    /// app's journal writes. Never read to decide anything.
    package struct Note: Codable, Sendable, Equatable {
        package var family: String
        package var variant: String
        package var format: String
        package var faster: Bool
        package var evaluations: Int
        /// The date of the report whose model predicted (`Report.date`).
        package var map: String
        package var measured: Double
        package var predicted: Double?
        package var error: Double?
        package var productError: Double?
        package var attentionError: Double?
        package var coverage: Double
        package var verdict: String
        package var classes: [Line]
        /// **Executions that ran otherwise than planned, by reason** (`ProductLedger.Reason`'s names); absent when none did.
        package var fallbacks: [String: Int]?

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

        /// How many classes the note keeps.
        package static let linkClasses = 10

        package init(request: CapacityClasses.Request, map: String, comparison: Comparison, verdict: Verdict,
                     fallbacks: [ProductLedger.Fallback] = []) {
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
            classes = comparison.rows.prefix(Self.linkClasses).map {
                Line(id: $0.id, route: $0.routes.first ?? "", executions: $0.executions, measured: $0.measured,
                     predictedRoute: $0.predictedRoute, predicted: $0.predicted, error: $0.error)
            }
            let byReason = Dictionary(fallbacks.map { ($0.reason.rawValue, $0.executions) }, uniquingKeysWith: +)
            self.fallbacks = byReason.isEmpty ? nil : byReason
        }
    }

    /// **The whole check of a render**, from its record to its note: the request it stands for, the prediction under `plan`, the
    /// comparison, the verdict. `nil` for a render the catalogue does not describe (an unknown family, a version that does not
    /// exist, a render with references or a LoRA: classes the catalogue does not carry).
    ///
    /// `mapDate`: the report whose model `plan` was made of (`Report.date`), written in the note.
    package static func check(_ render: Engine.Render, variant: Variant, weightFormat: CapacityClasses.WeightFormat? = nil, references: Int = 0, plan: CapacityPlan = EngineSettings.effective.plan,
                              mapDate: String = "", thresholds: Thresholds = Thresholds())
        -> (comparison: Comparison, verdict: Verdict, note: Note)? {
        guard !render.classes.isEmpty, render.loras.isEmpty, references == 0,
              let request = request(of: render, variant: variant, weightFormat: weightFormat),
              CapacityClasses.classes(for: request) != nil else { return nil }
        let predicted = predictions(for: request, plan: plan, aneBroughtUp: render.neuralEngine,
                                    accelerators: render.accelerators) ?? []
        let comparison = compare(render.classes, predicted: predicted)
        let verdict = verdict(comparison, thresholds: thresholds)
        return (comparison, verdict, Note(request: request, map: mapDate, comparison: comparison, verdict: verdict,
                                                 fallbacks: render.fallbacks))
    }
}

extension Diagnostic.Report {
    /// The same report, its last render's note attached (what « Report my configuration » sends).
    package func with(render note: RenderLoop.Note?) -> Diagnostic.Report {
        var report = self
        report.render = note
        return report
    }
}
