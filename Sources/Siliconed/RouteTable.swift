import Foundation

/// **The route table: which route each class of computation takes on this Mac** — the engine consults it at
/// the moment it runs a product or an attention, instead of reading one global switch for the whole render.
/// Pure data: no Metal, no weights; the capacity map produces it (`init(decisions:)` today, a
/// `CapacityMeasure.Plan` through `init(plan:…)`), `EngineSettings.routes` holds it, the report shows it.
///
///     RouteTable
///        ├─ products     a product k×n at m rows — a class is its shape and its height
///        │    ├─ weights   [WeightRule]   which form of the weight the product reads: widened to fp32,
///        │    │                           the bf16 / int8 / Q8_0 weight as published (+ its activation
///        │    │                           rounded to bf16), by weight kind and shape, first match wins
///        │    ├─ amx       GPU + AMX      the GPU's share: a scalar and a grid (k, n, m) interpolated in log
///        │    ├─ ane       GPU + ANE      idem, a grid of its own, the rows' unit, the height under which it sleeps
///        │    │    └─ beside AMX          the shapes where the three engines take the product at once
///        │    └─ classes   [ClassRule]    a class forced to one set of engines, ahead of the grids
///        └─ attention    an attention of S query rows, `heads` of `headDim`
///             └─ rules   [AttentionRule]  flash allowed, head split allowed (+ the CPU's share of heads)
///
/// What sets it apart — what a table written by analogy with the switches it replaces would get wrong:
///
///   - **a route is looked up where the work runs, not where the engine is built.** `route(k:n:m:)` is
///     asked at every product (`GEMM.linear`, `WeightProducts.prepare`); a switch read once at launch
///     (`static let enabled`) cannot say "the ANE for this shape, not for that one". What stays global
///     is what exists *once per process*: the engines brought up (a conductor, a `TensorGEMM`, a
///     `NeuralEngine`: `usesAMX`, `usesANE`, `usesBF16Weights`…, derived — an engine is built iff some
///     class may use it) and the format of a *prepared* weight, which is a property of the weight's
///     kind (bf16, int8, Q8_0) and shape, never of one call: a weight is prepared once for every
///     evaluation that reads it, so its rule is looked up at the height of its first use and its
///     class is `(kind, k, n)` plus the height range the rule names;
///   - **the grids are the same functions as before**: `Conductor.Shares` (ρ at measured points,
///     composed along two axes, linear in `log` between heights, the nearest point outside the
///     measured range). The table changes who asks, not the arithmetic — the same table gives the same
///     routes as the switches did, bit for bit, and the pure test says so on every kept map;
///   - **a class rule is ahead of the grids** and may only choose among the engines the table
///     brought up: a plan that rules a shape "GPU alone" says so without touching the grid that
///     serves the others;
///   - **the attention keeps the two allowances apart** (`flash`, `split`): a model's code says which
///     it implements (Z-Image and Anima take flash before the split; Qwen-Image-2.1's target, Krea 2
///     and MOSS-Image only the split), the table says which are allowed — one `flash` flag read
///     as "the route" would take the split from Qwen the day flash went on;
///   - **Codable, one string per grid**: the table goes in a report, and the grids keep the text they
///     had in the settings (`amx_shares`, `ane_shares`, `ane_amx`), three decimals.
package struct RouteTable: Codable, Sendable, Equatable {
    package static let schema = 1

    package var products: Products
    package var attention: Attention

    package init(products: Products = Products(), attention: Attention = Attention()) {
        self.products = products
        self.attention = attention
    }

    // MARK: - A grid, as text

    /// `Conductor.Shares`, written as its text.
    package struct Grid: Codable, Sendable, Equatable {
        package var shares: Conductor.Shares
        package init(_ shares: Conductor.Shares = .none) { self.shares = shares }
        package init(from decoder: any Decoder) throws {
            let text = try decoder.singleValueContainer().decode(String.self)
            guard let shares = Conductor.Shares(text: text) else {
                throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "unreadable grid « \(text) »"))
            }
            self.shares = shares
        }
        package func encode(to encoder: any Encoder) throws {
            var c = encoder.singleValueContainer()
            try c.encode(shares.description)
        }
    }

    // MARK: - Products

    /// What a weight is, for the route: the form it is published in.
    package enum WeightKind: String, Codable, Sendable { case bf16, int8, q8 }

    /// How a product reads its weight.
    package enum WeightPath: String, Codable, Sendable {
        /// Widened to fp32 on the CPU (the default path: MPS, the AMX, the ANE read it).
        case widened
        /// The bf16 weight as published (`TensorGEMM`).
        case bf16
        /// The 8-bit weight as published (int8 convrot, Q8_0).
        case native
    }

    /// The form a weight is read in at a product, resolved.
    package struct WeightRoute: Sendable, Equatable {
        package var path: WeightPath
        /// The activation is rounded to bf16 before the product (`gemm_bf16_act`).
        package var roundsActivation: Bool
    }

    /// **One rule of the weight table**: weights of `kind` whose shape and height match read `path`. A shape
    /// field left `nil` matches any; the first rule that matches wins.
    package struct WeightRule: Codable, Sendable, Equatable {
        package var kind: WeightKind
        package var path: WeightPath
        package var roundsActivation = false
        package var k: Int?
        package var n: Int?
        package var minRows = 0
        package var maxRows: Int?

        package init(kind: WeightKind, path: WeightPath, roundsActivation: Bool = false, k: Int? = nil, n: Int? = nil,
                     minRows: Int = 0, maxRows: Int? = nil) {
            self.kind = kind; self.path = path; self.roundsActivation = roundsActivation
            self.k = k; self.n = n; self.minRows = minRows; self.maxRows = maxRows
        }

        func matches(_ kind: WeightKind, k: Int, n: Int, m: Int) -> Bool {
            self.kind == kind && (self.k ?? k) == k && (self.n ?? n) == n && m >= minRows && m <= (maxRows ?? Int.max)
        }
    }

    /// The engines one product's rows are shared between.
    package enum Engines: String, Codable, Sendable {
        case gpu = "gpu"
        case amx = "gpu+amx"
        case ane = "gpu+ane"
        case aneAMX = "gpu+ane+amx"
    }

    /// **A class forced to a set of engines**, ahead of the grids.
    package struct ClassRule: Codable, Sendable, Equatable {
        package var engines: Engines
        package var k: Int?
        package var n: Int?
        package var minRows = 0
        package var maxRows: Int?
        /// What the class takes in a process that did not bring the ANE up (a model that keeps it off: the Standard's
        /// products are bf16, the ANE computes in fp16): the best route of the map **without** the ANE's. `nil`: the
        /// engines named above, whatever the process holds.
        package var withoutANE: Engines?

        package init(engines: Engines, k: Int? = nil, n: Int? = nil, minRows: Int = 0, maxRows: Int? = nil,
                     withoutANE: Engines? = nil) {
            self.engines = engines; self.k = k; self.n = n; self.minRows = minRows; self.maxRows = maxRows
            self.withoutANE = withoutANE
        }

        func matches(k: Int, n: Int, m: Int) -> Bool {
            (self.k ?? k) == k && (self.n ?? n) == n && m >= minRows && m <= (maxRows ?? Int.max)
        }
    }

    /// GPU + AMX: the engine is brought up (`enabled`), the GPU keeps `gpuShare` of the rows where the grid says
    /// nothing, and `grid` says its share per shape and height.
    package struct AMX: Codable, Sendable, Equatable {
        package var enabled = false
        package var gpuShare = 0.631
        package var grid = Grid()
        package init(enabled: Bool = false, gpuShare: Double = 0.631, grid: Grid = Grid()) {
            self.enabled = enabled; self.gpuShare = gpuShare; self.grid = grid
        }
        /// The GPU's share of `m` rows of `k → n` (the frozen cut's `T`, before any servo).
        package func share(k: Int, n: Int, m: Int) -> Double { grid.shares.share(k: k, n: n, m: m) ?? gpuShare }
    }

    /// GPU + ANE: as the AMX, plus the height under which the GPU keeps everything and the shapes where the AMX
    /// also takes part of the GPU's side (`besideAMX`: ρ ≥ ½ there).
    package struct ANE: Codable, Sendable, Equatable {
        package var enabled = false
        package var gpuShare = 0.5
        package var minimumRows = 1024
        package var grid = Grid()
        package var besideAMX = Grid()
        package init(enabled: Bool = false, gpuShare: Double = 0.5, minimumRows: Int = 1024, grid: Grid = Grid(),
                     besideAMX: Grid = Grid()) {
            self.enabled = enabled; self.gpuShare = gpuShare; self.minimumRows = minimumRows
            self.grid = grid; self.besideAMX = besideAMX
        }
        /// The GPU's share beside the ANE of `m` rows of `k → n`.
        package func share(k: Int, n: Int, m: Int) -> Double {
            grid.shares.share(k: k, n: n, m: m, range: NeuralEngine.shareRange) ?? gpuShare
        }
    }

    /// **A product's route**, resolved at `(k, n, m)`.
    package struct ProductRoute: Sendable, Equatable {
        /// Rows the ANE takes (the last ones); 0: none.
        package var aneRows: Int
        /// The AMX takes part of the GPU's side beside the ANE (meaningful when `aneRows > 0`).
        package var amxBesideANE: Bool
        /// The AMX shares the rows with the GPU when the ANE does not take the product.
        package var amx: Bool

        package var engines: Engines {
            aneRows > 0 ? (amxBesideANE ? .aneAMX : .ane) : (amx ? .amx : .gpu)
        }
    }

    package struct Products: Codable, Sendable, Equatable {
        package var weights: [WeightRule]
        package var amx: AMX
        package var ane: ANE
        package var classes: [ClassRule]

        package init(weights: [WeightRule] = [], amx: AMX = AMX(), ane: ANE = ANE(), classes: [ClassRule] = []) {
            self.weights = weights; self.amx = amx; self.ane = ane; self.classes = classes
        }

        /// **The weight rules the switches make**: the bf16 weight with its activation rounded from `minRows` rows
        /// (`gemm_bf16_act`), then the plain bf16 weight (`gemm_tensor`); the published 8-bit weights
        /// (`gemm_int8`, `gemm_q8`), rounded too only when `eightBit` says so — no map decides it, a check forces it.
        package static func weightRules(bf16: Bool, int8: Bool, q8: Bool, rounded: Bool, minRows: Int,
                                        eightBit: Bool) -> [WeightRule] {
            var rules: [WeightRule] = []
            if rounded { rules.append(.init(kind: .bf16, path: .bf16, roundsActivation: true, minRows: minRows)) }
            if bf16 { rules.append(.init(kind: .bf16, path: .bf16)) }
            for (kind, on) in [(WeightKind.int8, int8), (.q8, q8)] where on {
                if rounded && eightBit { rules.append(.init(kind: kind, path: .native, roundsActivation: true, minRows: minRows)) }
                rules.append(.init(kind: kind, path: .native))
            }
            return rules
        }

        // The engines the process brings up: each exists once, built iff some class may use it.
        package var usesAMX: Bool { amx.enabled }
        package var usesANE: Bool { ane.enabled }
        package var usesBF16Weights: Bool { weights.contains { $0.kind == .bf16 && $0.path == .bf16 && !$0.roundsActivation } }
        package var usesInt8Weights: Bool { weights.contains { $0.kind == .int8 && $0.path == .native } }
        package var usesQ8Weights: Bool { weights.contains { $0.kind == .q8 && $0.path == .native } }
        /// Some product rounds its activation to bf16 (`TensorGEMM` is built with its rounding kernels).
        package var roundsActivations: Bool { weights.contains { $0.roundsActivation } }

        /// **How a product of `kind` weights, `k × n` at `m` rows, reads its weight.**
        /// `usable` lets the caller skip a rule its process cannot run (no `TensorGEMM`, no rounding kernels): the
        /// next rule that matches then applies, as a switch that was off never stood in the way of the one below.
        package func weight(_ kind: WeightKind, k: Int, n: Int, m: Int,
                            usable: (WeightRule) -> Bool = { _ in true }) -> WeightRoute {
            if let rule = weights.first(where: { $0.matches(kind, k: k, n: n, m: m) && usable($0) }) {
                return WeightRoute(path: rule.path, roundsActivation: rule.roundsActivation)
            }
            return WeightRoute(path: .widened, roundsActivation: false)
        }

        /// **The engines a class rule forces on a product**, `nil` when none matches (the grids decide). What
        /// `GEMM.linear` asks at every product; the engine instances (a conductor, a `NeuralEngine`) hold the
        /// grids, built from this table and moved by the diagnostic, which measures at shares of its own.
        package func forced(k: Int, n: Int, m: Int, aneBroughtUp: Bool = true) -> Engines? {
            guard let rule = classes.first(where: { $0.matches(k: k, n: n, m: m) }) else { return nil }
            if !aneBroughtUp, rule.engines == .ane || rule.engines == .aneAMX { return rule.withoutANE ?? .amx }
            return rule.engines
        }

        /// **The engines of a product `k × n` at `m` rows**, resolved from the table alone: a class rule if one
        /// matches (among the engines brought up), the grids otherwise — the ANE where its rows are worth a
        /// program (`NeuralEngine.rows`), the AMX beside the GPU on the rest. What the engine's instances give,
        /// built from this table (the pure test compares the two).
        ///
        /// `aneBroughtUp: false`: a process whose products do not take the ANE (the Standard's, the encoders'): a class
        /// rule that names it gives the route it keeps without it (`ClassRule.withoutANE`), as `GEMM.linear` does.
        package func route(k: Int, n: Int, m: Int, aneBroughtUp: Bool = true) -> ProductRoute {
            var (aneOK, amxOK) = (ane.enabled && aneBroughtUp, amx.enabled)
            var beside: Bool?
            if let engines = forced(k: k, n: n, m: m, aneBroughtUp: aneBroughtUp) {
                switch engines {
                case .gpu: (aneOK, amxOK) = (false, false)
                case .amx: aneOK = false
                case .ane: amxOK = false; beside = false
                case .aneAMX: beside = true
                }
            }
            let rows = aneOK ? NeuralEngine.rows(m, gpuShare: ane.share(k: k, n: n, m: m), minimum: ane.minimumRows) : 0
            let alongside = rows > 0 && amxOK && (beside ?? ((ane.besideAMX.shares.ratio(k: k, n: n, m: m) ?? 0) >= 0.5))
            return ProductRoute(aneRows: rows, amxBesideANE: alongside, amx: amxOK)
        }
    }

    // MARK: - Attention

    /// **One rule of the attention table**: an attention of `minSequence…maxSequence` query rows (and `heads` of
    /// `headDim`, when named) is allowed `flash`, and the head split with the CPU taking `splitFraction` of the
    /// heads (`nil`: no split).
    package struct AttentionRule: Codable, Sendable, Equatable {
        package var minSequence = 0
        package var maxSequence: Int?
        package var heads: Int?
        package var headDim: Int?
        package var flash = false
        package var splitFraction: Double?

        package init(minSequence: Int = 0, maxSequence: Int? = nil, heads: Int? = nil, headDim: Int? = nil,
                     flash: Bool = false, splitFraction: Double? = nil) {
            self.minSequence = minSequence; self.maxSequence = maxSequence; self.heads = heads; self.headDim = headDim
            self.flash = flash; self.splitFraction = splitFraction
        }

        func matches(sequence: Int, heads: Int?, headDim: Int?) -> Bool {
            sequence >= minSequence && sequence <= (maxSequence ?? Int.max)
                && (self.heads == nil || self.heads == heads) && (self.headDim == nil || self.headDim == headDim)
        }
    }

    /// An attention's route: which kernels it may use. MPSGraph is what remains.
    package struct AttentionRoute: Sendable, Equatable {
        package var flash: Bool
        /// The CPU's share of heads under the split; `nil`: the split is not allowed.
        package var splitFraction: Double?
        package var splits: Bool { splitFraction != nil }
    }

    package struct Attention: Codable, Sendable, Equatable {
        package var rules: [AttentionRule]
        package init(rules: [AttentionRule] = []) { self.rules = rules }

        /// **What an attention of `sequence` query rows may use.** `heads` / `headDim`: the class, when the caller
        /// knows it; a rule that names them then matches only that class, a call that omits them only the rules
        /// that do not.
        package func route(sequence: Int, heads: Int? = nil, headDim: Int? = nil) -> AttentionRoute {
            guard let rule = rules.first(where: { $0.matches(sequence: sequence, heads: heads, headDim: headDim) }) else {
                return AttentionRoute(flash: false, splitFraction: nil)
            }
            return AttentionRoute(flash: rule.flash, splitFraction: rule.splitFraction)
        }

        package var usesFlash: Bool { rules.contains { $0.flash } }
        package var usesHeadSplit: Bool { rules.contains { $0.splitFraction != nil } }
        /// The CPU's share of heads of the first rule that splits — what a formula that does not know its
        /// class takes (`HeadSplitAttention.referenceFraction` without a split).
        package var defaultSplitFraction: Double {
            rules.lazy.compactMap(\.splitFraction).first ?? HeadSplitAttention.referenceFraction
        }
    }

    // MARK: - From today's decisions

    /// **The table today's settings are**: every switch and grid of `settings`, as the classes they decide.
    /// Reads nothing but `settings`: a forced setting (`SILICONED_*`, a profile, a check) is a setting, hence
    /// a table; the table of a process is the settings' as they stand after the last override.
    package init(settings s: EngineSettings) {
        let weights = Products.weightRules(bf16: s.gemmTensor, int8: s.gemmInt8, q8: s.gemmQ8, rounded: s.gemmBF16Activations,
                                           minRows: s.bf16ActivationMinimumRows, eightBit: s.bf16ActivationEightBit)
        products = Products(weights: weights,
                            amx: AMX(enabled: s.amx, gpuShare: s.amxFraction, grid: Grid(s.amxShares)),
                            ane: ANE(enabled: s.gemmANE, gpuShare: s.aneFraction, minimumRows: s.aneMinimumRows,
                                     grid: Grid(s.aneShares), besideAMX: Grid(s.aneAMX)))
        // Flash everywhere (the kernel falls back by itself on a length it does not tile); the split from
        // `HeadSplitAttention.minimumSequence` rows.
        let fraction: Double? = s.headSplit ? s.headSplitFraction : nil
        attention = s.flash || fraction != nil
            ? Attention(rules: [.init(minSequence: HeadSplitAttention.minimumSequence, flash: s.flash, splitFraction: fraction),
                                .init(flash: s.flash)])
            : Attention()
    }

    /// **The table a capacity map's decisions are**: the settings they write (`Decisions.settings`), as a table.
    /// Hermetic: no environment, no profile.
    package init(decisions: Diagnostic.Decisions) {
        self.init(settings: EngineSettings(applying: decisions.settings))
    }

    // MARK: - A plan, as a table

    /// **The name of a class in a `CapacityMeasure.ComputeClass`, and its routes**: what the map's plan and the
    /// table share. `product:<k>x<n>@<m>`, `attention:<S>/<heads>x<headDim>`; a route is its engines
    /// (`gpu`, `gpu+amx`, `gpu+ane`, `gpu+ane+amx`) for a product, `graph` / `flash` / `split` for an attention.
    package enum ClassKey: Equatable, Sendable {
        case product(k: Int, n: Int, m: Int)
        case attention(sequence: Int, heads: Int, headDim: Int)

        package var id: String {
            switch self {
            case let .product(k, n, m): return "product:\(k)x\(n)@\(m)"
            case let .attention(s, h, d): return "attention:\(s)/\(h)x\(d)"
            }
        }

        /// The id's tail for a weight kind: none for bf16 (the Standard's, and the K-quants' that widen), `~int8`, `~q8`.
        package static func suffix(_ kind: WeightKind) -> String { kind == .bf16 ? "" : "~" + kind.rawValue }

        /// A plan's class id: the key and the weight kind (`bf16` when the id says none).
        package static func parse(_ id: String) -> (key: ClassKey, kind: WeightKind)? {
            let halves = id.split(separator: "~", maxSplits: 1).map(String.init)
            guard let key = ClassKey(id: halves[0]) else { return nil }
            guard halves.count == 2 else { return (key, .bf16) }
            return WeightKind(rawValue: halves[1]).map { (key, $0) }
        }

        package init?(id: String) {
            let id = String(id.split(separator: "~", maxSplits: 1)[0])
            let parts = id.split(separator: ":", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { return nil }
            let numbers = parts[1].split(whereSeparator: { "x@/".contains($0) }).compactMap { Int($0) }
            switch parts[0] {
            case "product" where numbers.count == 3: self = .product(k: numbers[0], n: numbers[1], m: numbers[2])
            case "attention" where numbers.count == 3: self = .attention(sequence: numbers[0], heads: numbers[1], headDim: numbers[2])
            default: return nil
            }
        }
    }

    /// **A plan's choices, as class rules on top of `base`** (the grids and the engines brought up, which the
    /// map's cuts give, and the weight rules, which the map's preparation figures decide): one `ClassRule` per
    /// product class whose chosen route names its engines, one `AttentionRule` per attention class. A class the
    /// plan left open (`chosen == nil`) or does not know keeps what `base` gives it. The engines a rule names
    /// but `base` did not bring up are brought up with `base`'s defaults — the plan never chose an engine
    /// that was not measured.
    ///
    /// `renderClasses` (the classes a render runs, `CapacityClasses`) completes the table to the heights nobody
    /// measured and fixes the weights' forms:
    ///
    ///   - a product (or attention) class with no decision of its own takes the route the **neighbouring measured
    ///     heights** give it (`interpolatedRoute`): the routes' hot-time ratios of the same k×n (the same heads and
    ///     head dim) at the measured heights, linear in log m, extended past the last measured height by the same
    ///     line, then the map's rule (`CapacityMeasure.rule`) on them — not the last measured height's route;
    ///   - a weight prepared once is read in **one form at every height it is used** (`unifyingWeightPaths`).
    package init(plan: CapacityMeasure.Plan, classes: [CapacityMeasure.ComputeClass], base: RouteTable = RouteTable(),
                 splitFraction: Double = HeadSplitAttention.referenceFraction,
                 renderClasses: [CapacityClasses.Class] = []) {
        var table = base
        /// `chosen`: a route id, `engines` or `engines/form`; `withoutANE`: the best route of the class that does not use the ANE.
        func apply(_ key: ClassKey, _ kind: WeightKind, _ chosen: String, withoutANE: String?) {
            switch key {
            case let .product(k, n, m):
                let parts = chosen.split(separator: "/").map(String.init)
                guard let engines = Engines(rawValue: parts[0]) else { return }
                let form = parts.count > 1 ? parts[1] : nil
                if engines == .amx || engines == .aneAMX { table.products.amx.enabled = true }
                if engines == .ane || engines == .aneAMX { table.products.ane.enabled = true }
                // One engine rule per height, whatever the weight kind: the engines are the same hardware, the first
                // class to name the height (the heaviest: the plan is in that order) names them.
                if !table.products.classes.contains(where: { $0.k == k && $0.n == n && $0.minRows == m && $0.maxRows == m }) {
                    table.products.classes.append(.init(engines: engines, k: k, n: n, minRows: m, maxRows: m,
                                                        withoutANE: withoutANE.flatMap { Engines(rawValue: String($0.split(separator: "/")[0])) }))
                }
                func pin(_ path: WeightPath, rounds: Bool = false) {
                    table.products.weights.insert(.init(kind: kind, path: path, roundsActivation: rounds, k: k, n: n,
                                                        minRows: m, maxRows: m), at: 0)
                }
                switch form {
                case nil:
                    if table.products.weight(kind, k: k, n: n, m: m).path != .widened { pin(.widened) }
                case "bf16": pin(.bf16)
                case "bf16act": pin(.bf16, rounds: true)
                default: pin(.native)
                }
            case let .attention(s, heads, headDim):
                table.attention.rules.insert(.init(minSequence: s, maxSequence: s, heads: heads, headDim: headDim,
                                                   flash: chosen == "flash",
                                                   splitFraction: chosen == "split" ? splitFraction : nil), at: 0)
            }
        }
        /// The best route of a decision that does not use the ANE (the rule on its ratios with the ANE's removed).
        func withoutANE(of decision: CapacityMeasure.Decision) -> String? {
            guard let compute = classes.first(where: { $0.id == decision.classID }),
                  compute.routes.count == decision.hotRatios.count else { return nil }
            let ratios = zip(compute.routes, decision.hotRatios).map { route, ratio in
                route.id.hasPrefix("gpu+ane") ? nil : ratio
            }
            return CapacityMeasure.rule(ratios: ratios, efficient: compute.routes.map(\.efficient)).map { compute.routes[$0].id }
        }
        var decided = Set<String>()
        for decision in plan.decisions {
            decided.insert(decision.classID)
            guard let chosen = decision.chosen, let (key, kind) = ClassKey.parse(decision.classID) else { continue }
            apply(key, kind, chosen, withoutANE: chosen.hasPrefix("gpu+ane") ? withoutANE(of: decision) : nil)
        }
        var seen = Set<String>()
        for c in renderClasses {
            let key = c.routeKey, kind = c.weightKind ?? .bf16
            let id = key.id + ClassKey.suffix(kind)
            guard !decided.contains(id), seen.insert(id).inserted else { continue }
            if let chosen = Self.interpolatedRoute(for: key, kind: kind, plan: plan, classes: classes) {
                let fallback = chosen.hasPrefix("gpu+ane")
                    ? Self.interpolatedRoute(for: key, kind: kind, plan: plan, classes: classes, excludingANE: true) : nil
                apply(key, kind, chosen, withoutANE: fallback)
            }
        }
        if !renderClasses.isEmpty { table.unifyWeightPaths(for: renderClasses) }
        self = table
    }

    /// A line for the report and the render cache's key: what the table makes the engine use.
    package var summary: String {
        let p = products
        var parts = ["amx \(p.amx.enabled) \(p.amx.gpuShare) \(p.amx.grid.shares.description)"]
        if p.ane.enabled {
            parts.append("ane \(p.ane.gpuShare) \(p.ane.minimumRows) \(p.ane.grid.shares.description) amx \(p.ane.besideAMX.shares.description)")
        }
        if !p.classes.isEmpty { parts.append("classes \(p.classes.count)") }
        parts.append("weights " + p.weights.map {
            "\($0.kind.rawValue):\($0.path.rawValue)\($0.roundsActivation ? "◐" : "")≥\($0.minRows)"
        }.joined(separator: ","))
        parts.append("attention " + attention.rules.map {
            "\($0.minSequence)…\($0.maxSequence.map(String.init) ?? "∞")\($0.flash ? " flash" : "")"
                + ($0.splitFraction.map { " split \($0)" } ?? "")
        }.joined(separator: ","))
        return parts.joined(separator: " · ")
    }
}
