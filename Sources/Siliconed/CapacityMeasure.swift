import Foundation

/// **The capacity map's measuring core: comparisons in interleaved pairs, decisions by expected loss, one
/// budget of wall seconds.** Pure: no Metal, no real clock, no weights. A probe (a closure that runs one
/// route of one class and says how long it took) and a clock are injected; the plan comes out.
///
///     classes (weight w = share of a render, ordered routes, route 0 the simplest)
///        │
///        │   for a class c and a route r ≠ ref (ref = the first route not judged wrong):
///        │       A B A B A …            A = ref, B = r, one step = "B then A"
///        │       ℓᵢ = ln( Bᵢ / ((Aᵢ + Aᵢ₊₁) / 2) )               the pair's log-ratio
///        │       τ̂ = median ℓᵢ,  σ̂ = 1.4826 · MAD(ℓ)  (shrunk to a prior),  SE = 1.253 σ̂ / √n  ⊕ drift residual
///        │       ln eᵣ = τ̂ + ln( heat(r) / heat(ref) )          e = hot time of r, ref = 1   (M4)
///        │
///        ├─ the rule     the first efficient route within 90 % of the best (and faster than ref), else the
///        │               simplest route that beats the one before it by more than `tie`
///        ├─ the loss     L(c) = w · Σᵣ E[(ln e_chosen − ln eᵣ − slack)⁺],  eᵣ ~ LogNormal   (closed form)
///        └─ the budget   next step → the class with the largest L, in it the route with the largest pair
///                        loss (or the chosen one when its own SE is the larger); stop at ε or at the budget
///
/// What sets it apart — what a version written by analogy with "median of three, then a margin" would get
/// wrong:
///
///   - **Why the neighbours' ratio cancels a drift.** Let `g(t)` be the machine's slowness at time `t`
///     (heat, battery, a neighbour). A run of A over `[s, s + tA]` takes `a·g(its midpoint)`; B follows over
///     `[s + tA, s + tA + tB]`, the next A (duration tA′) after it. The two A's midpoints are `s + tA/2` and
///     `s + tA + tB + tA′/2`; their mean, `s + ¾tA + ½tB + ¼tA′`, is B's midpoint `s + tA + ½tB` up to
///     `¼(tA′ − tA)`. For `g` linear, `(g(m₁) + g(m₂)) / 2 = g((m₁ + m₂) / 2)`: the ratio `Bᵢ / mean(A)` is
///     `b / a · (1 + O(g′ · ¼(tA′ − tA)))`, and `tA′ − tA ≈ g′ · tA²` — **the first order of the drift is gone
///     for any slope, what is left is `¼ (g′ tA)²`**, (slope × one run)². For a ×2 slowdown over a 60 s map
///     (g′ = 0.017/s) and runs of 0.3 s: 6·10⁻⁶. Against an exponential `g = e^{λt}` the same: `cosh(λΔ) − 1 ≈
///     (λΔ)²/2` with Δ about one run. The witness (before / after, per engine) enters only through this
///     second-order term: `½ (D · Δ / span)²`, D its |ln after/before| — a bound, not a margin. Today's rule,
///     margin = D, turns off every gain under D: the ×2 slowdown turns off anything under ×2.
///   - **A ratio is judged with the dispersion of the ratios**, not with a witness: the median and the MAD
///     of ℓᵢ (robust to the one run a neighbour's burst ruins). The median of n has a standard error of
///     about 1.253 σ/√n. Consecutive ratios share an A, which only lowers the variance of their mean
///     (`(4n − 2)/(4n²)` σ² against σ²/n), so SE is a little conservative. With n < 8 the dispersion is
///     shrunk to a prior (`noise`, 5 %) with weight 4: one ratio has no dispersion.
///   - **The expected loss is the value of the information, not a probability of being wrong.** Choosing
///     route c while route r is better by X (log-time, X ~ N(d, s²), d = ln e_c − ln e_r, s² the two SEs'
///     sum) costs `X⁺`; its mean is `s φ(d/s) + d Φ(d/s)` — `≈ |gap| · P(wrong)` when the gap is clear
///     (d ≫ s: |d|·1), and `≈ 0.4 s` at a tie, where the gap is unknown but small. Weighted by `w` it is a
///     fraction of the render. Σ over the rival routes bounds the max (a union bound, safe side). A route
///     never measured has the prior N(0, 0.5²): it is a rival, never a candidate — a hole vetoes
///     nothing, and measuring it first is what the allocation does, heavy classes first.
///   - **A decision closes at ε / N** (N = classes with a rival), so the sum stays under ε (≈ 2 % of the
///     render): heavy classes need tight figures, a light one is closed by a rough one — this is what spends
///     the seconds where the weight is.
///   - **The ANE keeps its favour**: an efficient route is taken as soon as it reaches 90 % of the best
///     speed (`efficientTolerance`) **and beats the simplest route** (today's rule: "where its pair beats the
///     GPU alone and keeps 90 % of the AMX's speed"); its 10 % slack is not a loss either (`slack`). Among
///     routes at a tie the simplest wins: a route must beat the one before it by `tie` (1 %).
///   - **The heat factor is a time multiplier per engine** (hot / cold, from witnesses; 1 when none). A
///     route runs at the speed of its slowest engine: the cuts are frozen in the map, and the engine that
///     slows down makes the others wait — `heat(route) = max` over its engines. (Re-cutting at the hot
///     regime would give less; that is the L3's to decide, with the cuts.)
///   - **A first lap that loses clearly closes the route** (`giveUp`, 1.5 × the reference's time: today's
///     `weightPathGiveUp`): one ratio is the figure, the budget goes on.
///   - **Out of budget, what is open falls back to the simplest route** unless the chosen one beats it
///     with `confidence` (95 %); the residual loss says what that costs.
///   - **A route judged wrong, or a hole (the probe says `nil`), is not a candidate for that class** — and for
///     no other: classes are independent, a missing measure is never a global veto.
///   - Deterministic: the order of steps depends on the numbers only (ties by index).
package enum CapacityMeasure {
    /// One way to run a class. The first of a class's routes is the simplest (GPU alone).
    package struct Route: Sendable, Equatable {
        package var id: String
        /// Taken as soon as it reaches `efficientTolerance` of the best speed (the ANE: the same work at a
        /// fraction of the power).
        package var efficient: Bool
        /// The engines it keeps busy: the keys of `Heat.factors`.
        package var engines: [String]
        /// The correctness check (invariant 3) passed. A wrong route is never timed nor chosen.
        package var correct: Bool

        package init(_ id: String, efficient: Bool = false, engines: [String] = [], correct: Bool = true) {
            self.id = id; self.efficient = efficient; self.engines = engines; self.correct = correct
        }
    }

    /// A kind of computation (a product k×n at m rows, an attention…) and its share of the render.
    package struct ComputeClass: Sendable, Equatable {
        package var id: String
        /// Its expected share of the render time (any scale: normalised by the sum).
        package var weight: Double
        package var routes: [Route]

        package init(_ id: String, weight: Double, routes: [Route]) {
            self.id = id; self.weight = weight; self.routes = routes
        }
    }

    /// **The regime** (M4): per engine, how much longer the hot regime takes than the cold one
    /// (`cold rate / hot rate` from the witnesses; the AMX of the Neo: 1.538 / 0.747 = 2.06).
    package struct Heat: Sendable, Equatable {
        package var factors: [String: Double]
        package init(_ factors: [String: Double] = [:]) { self.factors = factors }

        func factor(_ route: Route) -> Double { route.engines.map { factors[$0] ?? 1 }.max() ?? 1 }
    }

    /// **The before / after witnesses of the drift**, per engine: |ln(after / before)|, over `span` seconds.
    /// Only a bound on what the pairs leave (second order); no margin.
    package struct Witness: Sendable, Equatable {
        package var drift: [String: Double]
        package var span: Double
        package init(drift: [String: Double] = [:], span: Double = 60) { self.drift = drift; self.span = span }
    }

    package struct Settings: Sendable, Equatable {
        /// Total acceptable expected loss, as a fraction of the render.
        package var epsilon = 0.02
        /// Wall seconds the map may spend (the clock's).
        package var budget = 60.0
        package var efficientTolerance = 0.9
        /// A complex route must beat a simpler one by this fraction of time.
        package var tie = 0.01
        /// A first ratio above this closes the route.
        package var giveUp = 1.5
        /// Per-ratio dispersion prior (log units) and its weight in ratios.
        package var noise = 0.05
        package var noiseWeight = 4.0
        /// Prior spread of a route never measured (log units).
        package var unmeasuredSigma = 0.5
        package var confidence = 0.95
        /// Ratios per route: past it, a route is closed.
        package var maxRatios = 80
        /// A route is not closed on fewer ratios: one has no dispersion, and a lucky first lap is not a figure.
        package var minRatios = 3
        package init() {}
    }

    package enum Status: String, Sendable, Equatable {
        /// Loss under its share of ε (or sure).
        case closed
        /// Open at the end: fell back to the simplest route (unless sure).
        case open
        /// No route could be timed: the simplest route by default.
        case unmeasured
        /// No route passes its check.
        case none
    }

    package struct Decision: Sendable, Equatable {
        package var classID: String
        package var chosen: String?
        package var status: Status
        /// Expected loss of this decision, as a fraction of the render (w · Σ …).
        package var loss: Double
        /// Hot time of each route over the reference's (nil: not a candidate), in `routes` order.
        package var hotRatios: [Double?]
        /// Probe runs spent on this class.
        package var runs: Int
    }

    package struct Plan: Sendable, Equatable {
        package var decisions: [Decision]
        /// Σ of the decisions' losses: what the map may still cost, fraction of the render.
        package var residualLoss: Double
        package var spent: Double
        package var exhausted: Bool
    }

    package protocol Clock {
        /// Seconds, monotonic.
        func now() -> Double
    }

    /// Run one route of one class once and say how long it took (seconds); `nil`: it cannot run (a hole).
    package typealias Probe = (_ classIndex: Int, _ routeIndex: Int) -> Double?

    // MARK: - Statistics

    static func phi(_ x: Double) -> Double { exp(-0.5 * x * x) / (2 * Double.pi).squareRoot() }
    static func bigPhi(_ x: Double) -> Double { 0.5 * (1 + erf(x / 2.0.squareRoot())) }

    /// E[(X − slack)⁺] for X ~ N(d, s²): the expected shortfall.
    static func shortfall(mean d: Double, sd s: Double, slack: Double = 0) -> Double {
        let m = d - slack
        if s < 1e-12 { return max(m, 0) }
        let z = m / s
        return s * phi(z) + m * bigPhi(z)
    }

    static func median(_ v: [Double]) -> Double {
        let s = v.sorted(), n = s.count
        return n % 2 == 1 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2])
    }

    // MARK: - State

    private enum PairState { case open, lost, hole, capped }

    private struct Pair {
        var a: [Double] = [], b: [Double] = []
        var state = PairState.open

        /// ℓᵢ = ln(Bᵢ / mean(Aᵢ, Aᵢ₊₁)).
        var logRatios: [Double] {
            (0..<b.count).compactMap { i in
                guard i + 1 < a.count else { return nil }
                return log(b[i] / (0.5 * (a[i] + a[i + 1])))
            }
        }
    }

    /// An estimate in log units of hot time over the reference's.
    private struct Estimate {
        var mean: Double
        var sd: Double
        var measured: Bool
    }

    // MARK: - Planning

    package static func plan(classes: [ComputeClass], heat: Heat = Heat(), witness: Witness = Witness(),
                             settings: Settings = Settings(), clock: Clock, probe: Probe) -> Plan {
        let start = clock.now()
        let total = max(classes.map(\.weight).reduce(0, +), 1e-12)
        let weights = classes.map { $0.weight / total }
        let refs: [Int?] = classes.map { $0.routes.firstIndex(where: \.correct) }
        var pairs: [[Pair]] = classes.map { Array(repeating: Pair(), count: $0.routes.count) }
        var runs = Array(repeating: 0, count: classes.count)
        var longestRun = 0.0

        // Routes that are never timed: the reference, wrong ones.
        for (c, cls) in classes.enumerated() {
            for r in cls.routes.indices where !cls.routes[r].correct || r == refs[c] { pairs[c][r].state = .hole }
        }

        func run(_ c: Int, _ r: Int) -> Double? {
            guard let t = probe(c, r), t > 0 else { return nil }
            runs[c] += 1
            longestRun = max(longestRun, t)
            return t
        }

        func estimate(_ c: Int, _ r: Int) -> Estimate {
            guard let ref = refs[c] else { return Estimate(mean: 0, sd: 0, measured: false) }
            let route = classes[c].routes[r]
            let lnHeat = log(heat.factor(route) / heat.factor(classes[c].routes[ref]))
            if r == ref { return Estimate(mean: 0, sd: 0, measured: true) }
            let ls = pairs[c][r].logRatios
            guard !ls.isEmpty else { return Estimate(mean: lnHeat, sd: settings.unmeasuredSigma, measured: false) }
            let n = Double(ls.count)
            let mid = median(ls)
            let mad = 1.4826 * median(ls.map { abs($0 - mid) })
            let w = settings.noiseWeight
            let sigma = ((w * settings.noise * settings.noise + n * mad * mad) / (w + n)).squareRoot()
            var variance = pow(1.253 * sigma / n.squareRoot(), 2)
            // The second-order residue of a drift, bounded by the witnesses of the two routes' engines.
            let engines = route.engines + classes[c].routes[ref].engines
            let d = engines.map { witness.drift[$0] ?? 0 }.max() ?? 0
            if d > 0, witness.span > 0 {
                let p = pairs[c][r]
                let delta = (p.a.reduce(0, +) / Double(p.a.count) + p.b.reduce(0, +) / Double(p.b.count)) / 2
                variance += pow(0.5 * pow(d * delta / witness.span, 2), 2)
            }
            return Estimate(mean: mid + lnHeat, sd: variance.squareRoot(), measured: true)
        }

        func candidates(_ c: Int) -> [Int] {
            guard let ref = refs[c] else { return [] }
            return classes[c].routes.indices.filter { r in
                guard classes[c].routes[r].correct else { return false }
                if r == ref { return true }
                return pairs[c][r].state != .hole && !pairs[c][r].logRatios.isEmpty
            }
        }

        /// The rule, on the estimates' means.
        func choose(_ c: Int) -> Int? {
            guard let ref = refs[c] else { return nil }
            let cand = candidates(c)
            let e = Dictionary(uniqueKeysWithValues: cand.map { ($0, exp(estimate(c, $0).mean)) })
            let best = e.values.min() ?? 1
            for r in cand where classes[c].routes[r].efficient && r != ref {
                if best / e[r]! >= settings.efficientTolerance && e[r]! < 1 - settings.tie { return r }
            }
            var chosen = ref
            for r in cand where r > ref && e[r]! < e[chosen]! * (1 - settings.tie) { chosen = r }
            return chosen
        }

        /// The routes that can still be a rival: not wrong, not a hole, not lost (nor the reference).
        func rivals(_ c: Int, excluding chosen: Int) -> [Int] {
            classes[c].routes.indices.filter { r in
                r != chosen && classes[c].routes[r].correct
                    && (r == refs[c] || (pairs[c][r].state == .open || pairs[c][r].state == .capped))
            }
        }

        func pairLoss(_ c: Int, chosen: Int, rival r: Int) -> Double {
            let ec = estimate(c, chosen), er = estimate(c, r)
            let slack = classes[c].routes[chosen].efficient && !classes[c].routes[r].efficient
                ? log(1 / settings.efficientTolerance) : 0
            let sd = (ec.sd * ec.sd + er.sd * er.sd).squareRoot()
            return min(shortfall(mean: ec.mean - er.mean, sd: sd, slack: slack), 1)
        }

        func classLoss(_ c: Int, chosen: Int?) -> Double {
            guard let ch = chosen else { return 0 }
            return weights[c] * rivals(c, excluding: ch).reduce(0) { $0 + pairLoss(c, chosen: ch, rival: $1) }
        }

        let withRivals = classes.indices.filter { c in
            refs[c] != nil && classes[c].routes.indices.contains { $0 != refs[c] && classes[c].routes[$0].correct }
        }
        let threshold = settings.epsilon / Double(max(withRivals.count, 1))

        /// Every route still in play has its `minRatios` (the chosen one, and the open rivals).
        func thin(_ c: Int, chosen: Int) -> Bool {
            (rivals(c, excluding: chosen) + [chosen]).contains { r in
                r != refs[c] && pairs[c][r].state == .open && pairs[c][r].logRatios.count < settings.minRatios
            }
        }

        // The estimated cost of the next step on a pair: what the pair has taken, else the longest run seen.
        func stepCost(_ c: Int, _ r: Int) -> Double {
            let p = pairs[c][r]
            if let a = p.a.last, let b = p.b.last { return a + b }
            return 3 * longestRun
        }

        func step(_ c: Int, _ r: Int) {
            guard let ref = refs[c] else { return }
            var p = pairs[c][r]
            if p.a.isEmpty {
                guard let a = run(c, ref) else { for x in pairs[c].indices { pairs[c][x].state = .hole }; return }
                p.a.append(a)
            }
            guard let b = run(c, r) else { p.state = .hole; pairs[c][r] = p; return }
            guard let a = run(c, ref) else { p.state = .hole; pairs[c][r] = p; return }
            p.b.append(b); p.a.append(a)
            pairs[c][r] = p
            let e = estimate(c, r)
            if p.b.count == 1, exp(e.mean) > settings.giveUp { pairs[c][r].state = .lost }
            if p.b.count >= settings.maxRatios { pairs[c][r].state = .capped }
        }

        var exhausted = false
        while true {
            // The open class with the largest expected loss.
            var best: (c: Int, loss: Double, chosen: Int, r: Int)?
            for c in withRivals {
                guard let chosen = choose(c) else { continue }
                let loss = classLoss(c, chosen: chosen)
                if loss <= threshold, !thin(c, chosen: chosen) { continue }
                // In it, the rival with the largest pair loss; the chosen one when its own SE is larger.
                let open = rivals(c, excluding: chosen).filter { pairs[c][$0].state == .open }
                var target: Int
                if let rival = open.max(by: { pairLoss(c, chosen: chosen, rival: $0) < pairLoss(c, chosen: chosen, rival: $1) }) {
                    target = rival
                    if pairs[c][chosen].state == .open, estimate(c, chosen).sd > estimate(c, rival).sd { target = chosen }
                } else if pairs[c][chosen].state == .open {
                    target = chosen            // the reference is the only rival left: sharpen the chosen one
                } else { continue }
                if best == nil || loss > best!.loss { best = (c, loss, chosen, target) }
            }
            guard let pick = best else { break }
            if clock.now() - start + stepCost(pick.c, pick.r) > settings.budget { exhausted = true; break }
            step(pick.c, pick.r)
        }

        var decisions: [Decision] = []
        var residual = 0.0
        for (c, cls) in classes.enumerated() {
            guard let ref = refs[c] else {
                decisions.append(Decision(classID: cls.id, chosen: nil, status: .none, loss: 0,
                                          hotRatios: cls.routes.map { _ in nil }, runs: runs[c]))
                continue
            }
            let rule = choose(c) ?? ref
            var chosen = rule
            var loss = classLoss(c, chosen: rule)
            var status = Status.closed
            if withRivals.contains(c), loss > threshold || thin(c, chosen: rule) {
                status = .open
                // Sure enough to keep the rule's choice? P(it beats the reference by `tie`).
                let est = estimate(c, rule)
                let sure = rule != ref && est.sd > 0
                    && bigPhi((log(1 - settings.tie) - est.mean) / est.sd) >= settings.confidence
                if !sure { chosen = ref; loss = classLoss(c, chosen: ref) }
            }
            if !candidates(c).contains(where: { $0 != ref }), withRivals.contains(c) { status = .unmeasured }
            let ratios: [Double?] = cls.routes.indices.map { r in
                candidates(c).contains(r) ? exp(estimate(c, r).mean) : nil
            }
            residual += loss
            decisions.append(Decision(classID: cls.id, chosen: cls.routes[chosen].id, status: status, loss: loss,
                                      hotRatios: ratios, runs: runs[c]))
        }
        return Plan(decisions: decisions, residualLoss: residual, spent: clock.now() - start, exhausted: exhausted)
    }
}
