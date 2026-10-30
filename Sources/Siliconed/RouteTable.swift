import Foundation

/// **LEGACY — the previous capacity map's output format; the engine reads none of it.** The engine asks the plan
/// (`CapacityPlan`) for exact rows at every product and every attention. This table survives only as what the previous map
/// (`CapacityMeasure`, `CapacityMapRun`, `CapacityRules`) writes into its report and what `RenderLoop` predicts routes from, and goes
/// with it (E4b). Nothing here is consulted by `GEMM`, `WeightProducts`, `MemoryNeed` or the attentions.
///
/// It was: which route each class takes on this Mac — products by a class rule over the AMX's and the ANE's grids of ρ (interpolated
/// in log, the nearest measured point beyond), the weights' forms by rule, the attentions by (S, heads, head dim); a ratio the
/// engine rounded to its quantum and clamped to a band.
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

    /// What a weight is, for the route: the form it is published in (now `WeightKind`, `CapacityPlan.swift`).
    package typealias WeightKind = Siliconed.WeightKind

    /// How a product reads its weight (now `WeightPath`, `CapacityPlan.swift`).
    package typealias WeightPath = Siliconed.WeightPath

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
        var shape: (k: Int, n: Int)? { k.flatMap { k in n.map { (k, $0) } } }
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
        /// The weights the rule is for (a map rules each form at its point: an int8 weight read as published beside the
        /// AMX where the bf16 one is widened for the ANE); `nil`: every kind, as a rule of the settings.
        package var kind: WeightKind?

        package init(engines: Engines, k: Int? = nil, n: Int? = nil, minRows: Int = 0, maxRows: Int? = nil,
                     withoutANE: Engines? = nil, kind: WeightKind? = nil) {
            self.engines = engines; self.k = k; self.n = n; self.minRows = minRows; self.maxRows = maxRows
            self.withoutANE = withoutANE; self.kind = kind
        }

        func matches(k: Int, n: Int, m: Int) -> Bool {
            (self.k ?? k) == k && (self.n ?? n) == n && m >= minRows && m <= (maxRows ?? Int.max)
        }
        var shape: (k: Int, n: Int)? { k.flatMap { k in n.map { (k, $0) } } }
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
        ///
        /// A product no rule matches takes the **nearest shaped rule** of its kind (`Nearest`) — a height or a model the map
        /// never saw reads its weight as its nearest measured neighbour does, not widened by default.
        package func weight(_ kind: WeightKind, k: Int, n: Int, m: Int,
                            usable: (WeightRule) -> Bool = { _ in true }) -> WeightRoute {
            // Only a product no rule speaks of borrows a neighbour's: a rule that matches but cannot run falls to the next
            // that matches, as before.
            let own = weights.filter { $0.matches(kind, k: k, n: n, m: m) }
            let rule = own.first(where: usable) ?? (own.isEmpty ? Nearest.rule(
                among: weights.filter { $0.kind == kind && $0.shape != nil && usable($0) }, k: k, n: n, m: m) { ($0.shape!, $0.minRows, $0.maxRows) } : nil)
            guard let rule else { return WeightRoute(path: .widened, roundsActivation: false) }
            return WeightRoute(path: rule.path, roundsActivation: rule.roundsActivation)
        }

        /// **The engines a class rule forces on a product**, `nil` when none matches (the grids decide). What
        /// `GEMM.linear` asks at every product; the engine instances (a conductor, a `NeuralEngine`) hold the
        /// grids, built from this table and moved by the diagnostic, which measures at shares of its own.
        ///
        /// As for the weights, a product no shaped rule matches takes the **nearest shaped rule** within reach (`Nearest`):
        /// the map rules every class it measured, and a format or a model it did not see takes its measured neighbour's
        /// route — a route judged. A rule that names no shape (the map's « GPU alone ») is what is left beyond reach; the
        /// grids decide only where no rule speaks (the switches' tables have none).
        ///
        /// `kind`: the weight's — the rules of its kind when the table has some (a map's), the kind-free ones otherwise.
        package func forced(k: Int, n: Int, m: Int, kind: WeightKind = .bf16, aneBroughtUp: Bool = true) -> Engines? {
            let classes = classes.contains { $0.kind == kind } ? classes.filter { $0.kind == kind } : classes.filter { $0.kind == nil }
            let shaped = classes.filter { $0.shape != nil }
            guard let rule = shaped.first(where: { $0.matches(k: k, n: n, m: m) })
                    ?? Nearest.rule(among: shaped, k: k, n: n, m: m, { ($0.shape!, $0.minRows, $0.maxRows) })
                    ?? classes.first(where: { $0.shape == nil && $0.matches(k: k, n: n, m: m) })
            else { return nil }
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
        package func route(k: Int, n: Int, m: Int, kind: WeightKind = .bf16, aneBroughtUp: Bool = true) -> ProductRoute {
            var (aneOK, amxOK) = (ane.enabled && aneBroughtUp, amx.enabled)
            var beside: Bool?
            if let engines = forced(k: k, n: n, m: m, kind: kind, aneBroughtUp: aneBroughtUp) {
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

    // MARK: - The nearest measured class

    /// **Where a class the map did not measure takes its route**: from the measured rule whose shape is nearest — the
    /// same shape at the nearest height first, then the nearest shape, each distance in log (a ratio, the way the work
    /// scales: 3072 is as near 4096 as 3840 is near 5120). The engines' ratios move smoothly with the shape and the
    /// height; the rule of the nearest point is the map's best guess, and the render's check (predicted against
    /// measured, `RenderLoop`) says when it was wrong. **Within reach only**: a shape more than `reach` away in all (a 64-wide
    /// projection against 3840×3840) is not one the measured class speaks for, and takes no rule from it.
    package enum Nearest {
        /// The farthest shape a rule stands for: twice as large on one side, or √2 on both.
        package static let reach = log(2.0)

        static func distance(_ x: Int, _ low: Int, _ high: Int?) -> Double {
            let x = Double(max(x, 1)), low = Double(max(low, 1)), high = Double(high ?? Int.max)
            return x < low ? log(low / x) : x > high ? log(x / high) : 0
        }

        /// `describe`: a rule's shape `(k, n)` (heads and head dim for an attention) and its height range.
        static func rule<R>(among rules: [R], k: Int, n: Int, m: Int,
                            _ describe: (R) -> ((k: Int, n: Int), Int, Int?)) -> R? {
            func shape(_ r: R) -> Double {
                let s = describe(r).0
                return abs(log(Double(s.k) / Double(k))) + abs(log(Double(s.n) / Double(n)))
            }
            return rules.filter { shape($0) <= reach + 1e-9 }.min { a, b in
                let (_, la, ha) = describe(a), (_, lb, hb) = describe(b)
                let (shapeA, shapeB) = (shape(a), shape(b))
                if abs(shapeA - shapeB) > 1e-9 { return shapeA < shapeB }
                return distance(m, la, ha) < distance(m, lb, hb)
            }
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
        ///
        /// A length no rule covers takes the rule of **its class at the nearest length** (in log) when the table
        /// has one, else of the nearest class (`Nearest`): a map writes one rule per measured length, and an
        /// untabulated format (1024×1536) or model must not fall back to MPSGraph alone while its neighbours split.
        package func route(sequence: Int, heads: Int? = nil, headDim: Int? = nil) -> AttentionRoute {
            let rule = rules.first(where: { $0.matches(sequence: sequence, heads: heads, headDim: headDim) })
                ?? nearest(sequence: sequence, heads: heads, headDim: headDim)
            guard let rule else { return AttentionRoute(flash: false, splitFraction: nil) }
            return AttentionRoute(flash: rule.flash, splitFraction: rule.splitFraction)
        }

        /// The rule naming a class nearest this one — its own at the nearest length first, else the nearest heads and
        /// head dimension (`Nearest`).
        private func nearest(sequence: Int, heads: Int?, headDim: Int?) -> AttentionRule? {
            guard let heads, let headDim else { return nil }
            let named = rules.filter { $0.heads != nil && $0.headDim != nil }
            return Nearest.rule(among: named, k: heads, n: headDim, m: sequence) { (($0.heads!, $0.headDim!), $0.minSequence, $0.maxSequence) }
        }

        package var usesFlash: Bool { rules.contains { $0.flash } }
        package var usesHeadSplit: Bool { rules.contains { $0.splitFraction != nil } }
        /// The CPU's share of heads of the first rule that splits — what a formula that does not know its
        /// class takes (`HeadSplitAttention.referenceFraction` without a split).
        package var defaultSplitFraction: Double {
            rules.lazy.compactMap(\.splitFraction).first ?? HeadSplitAttention.referenceFraction
        }
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
    /// that was not measured. An attention that splits hands the CPU the share its class was balanced at
    /// (`splitFractions`, by class id), the reference share where none was.
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
                 splitFractions: [String: Double] = [:],
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
                // One engine rule per point and weight kind: each form was decided on its own figures, and a form read
                // as published beside the AMX (`gpu+amx/int8`) keeps the AMX where the bf16 view chose the ANE.
                if !table.products.classes.contains(where: { $0.k == k && $0.n == n && $0.minRows == m && $0.maxRows == m && $0.kind == kind }) {
                    table.products.classes.append(.init(engines: engines, k: k, n: n, minRows: m, maxRows: m,
                                                        withoutANE: withoutANE.flatMap { Engines(rawValue: String($0.split(separator: "/")[0])) },
                                                        kind: kind))
                }
                func pin(_ path: WeightPath, rounds: Bool = false) {
                    table.products.weights.insert(.init(kind: kind, path: path, roundsActivation: rounds, k: k, n: n,
                                                        minRows: m, maxRows: m), at: 0)
                }
                // Every decision pins its form, the widened one too: a shape no rule names borrows its nearest rule
                // (`Nearest`), and a point left unpinned would lend a neighbour's form instead of its own.
                switch form {
                case nil: pin(.widened)
                case "bf16": pin(.bf16)
                case "bf16act": pin(.bf16, rounds: true)
                default: pin(.native)
                }
            case let .attention(s, heads, headDim):
                table.attention.rules.insert(.init(minSequence: s, maxSequence: s, heads: heads, headDim: headDim,
                                                   flash: chosen == "flash",
                                                   splitFraction: chosen == "split"
                                                       ? splitFractions[key.id] ?? HeadSplitAttention.referenceFraction : nil), at: 0)
            }
        }
        /// The best route of a decision that does not use the ANE (the rule on its ratios with the ANE's removed).
        func withoutANE(of decision: CapacityMeasure.Decision) -> String? {
            guard let compute = classes.first(where: { $0.id == decision.classID }),
                  compute.routes.count == decision.hotRatios.count else { return nil }
            let ratios = zip(compute.routes, decision.hotRatios).map { route, ratio in
                route.id.hasPrefix("gpu+ane") ? nil : ratio
            }
            return CapacityMeasure.rule(ratios: ratios, efficient: compute.routes.map(\.efficient),
                                        brutal: compute.routes.map(\.brutal)).map { compute.routes[$0].id }
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
