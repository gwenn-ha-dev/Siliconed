import Foundation

/// **What the capacity map decides over: a fixed grid of the computations a diffusion transformer is made of, the same
/// on every Mac and for every model, each with its routes in the order the measuring core wants them (simplest first),
/// and the regime the table is decided at.** Pure: no Metal, no clock, no weights, no family — the machine enters
/// through `Capabilities` (which engines and kernels exist here) and through the witnesses.
///
///     products     k×n ∈ 2048² · 4096² · 8192² · 2048×8192 · 8192×2048 · 4096×16384 · 16384×4096        × m ∈ 1024 · 4096
///                  routes: gpu · gpu+amx · gpu+ane · gpu+ane+amx  ·  gpu/bf16 · gpu/bf16act · gpu/int8 · gpu+amx/int8 ·
///                          gpu/q8 · gpu+amx/q8                                — timed once, decided per weight form:
///                  views   bf16  = the engines + the bf16 forms     (`product:4096x4096@1024`)
///                          int8  = the engines + the int8 forms     (`product:4096x4096@1024~int8`)
///                          q8    = the engines + the Q8_0 forms     (`product:4096x4096@1024~q8`)
///                          each route plus its weight's preparation (widened to fp32, or copied as published)
///     attentions   heads ∈ 16 · 32, head dim 128                                                        × S ∈ 1024 · 4096
///                  routes: graph · flash · split
///     weight       1 per point, a product's split evenly between its three forms
///
/// What sets it apart — what a catalogue written by analogy with "the shapes of the model in hand" would get wrong:
///
///   - **The map knows no model.** It measures the machine over the domain the models live in, and a render reads its
///     route at its own shape from the nearest point (`RouteTable.Nearest`: the nearest k×n within ×2 on a side or √2 on
///     both, then the nearest height). The grid is the coarsest that leaves no product or attention of the five families out
///     of that reach (a pure test holds it), and a model added tomorrow falls in it as well: nothing about the map changes
///     when a model is installed, removed or added to the catalogue, and a Mac with none measures the same map.
///   - **The shapes are a lattice in log, not a list of layers**: squares (attention projections, modulations), ×4 wide
///     (an MLP's up), ×4 tall (its down), at three sizes. The ratios of the engines move smoothly with the shape in log
///     (the maps of the five families: GPU+AMX 0.70–0.92 at ~1 040 rows over eighteen shapes, no jump between neighbours),
///     so a point measured ×2 away speaks for a shape as well as the shape's own measurement within the noise of one.
///   - **A weight's form is a view of one measurement, not a class of its own.** The engines compute the same fp32 product
///     whatever the published form; only the preparation differs (seconds per value from the widening probes, `Preparation`).
///     Timing GPU, AMX and ANE once per form measured the same hardware three times (bf16 and Q8_0 at 3840×10240: 0.744 and
///     0.747); the core decides each view on the shared pairs (`CapacityMeasure.View`). K-quants (Light) widen on the CPU like
///     any weight and read the bf16 view.
///   - **Two heights, the table's line beyond them**: 1 024 rows (a 512² image, the half grid of Faster Rendering) and 4 096
///     (a 1024² image); a render at 6 144 rows takes the nearest (the routes of 4 128 held at 6 176 on the M1 Pro).
///   - **Routes are static facts of this Mac and of the shape, not results**: the AMX from `amxMinimumRows`, the ANE from
///     `aneMinimumRows`, a native weight route only if the kernel was built, the head split from
///     `HeadSplitAttention.minimumSequence`. What may be wrong is found by the probe's judge, one route of one class at a time.
///   - **The regime is read from the engines' own witnesses** (`Regime`): a witness is a fixed product timed at the start and
///     at the end of the map on the GPU, the AMX and the ANE; the ratio is that engine's heat factor — *unless* it is
///     inside what the witness's own dispersion allows (two standard errors), in which case it is 1. The noise of a
///     quiet M1 Pro (±3 %) must not flip a 1 % tie between the GPU and the AMX; the Neo's AMX (×2.06) must.
///   - **Green is a property of the map, not of the hour** (`Verdict`): complete (residual loss under ε), every chosen route
///     judged right, the peak under the need under B, the seconds in the budget. Heat, battery, Low Power Mode and another
///     process's swap are facts of the regime, written beside the map, never a reason for it to be orange.
package enum CapacityMap {
    /// Wall seconds for the whole map, on any Mac, settings and witnesses included.
    package static let budget = 60.0
    /// Rows within which two heights of the same k×n are one class of a render's ledger (`RenderLoop`).
    package static let heightGap = 64

    // MARK: - The grid

    /// The products' shapes (k × n): squares, ×4 wide and ×4 tall, a factor 2 apart in log on each side.
    package static let gridShapes: [(k: Int, n: Int)] = [(2048, 2048), (4096, 4096), (8192, 8192), (2048, 8192), (8192, 2048),
                                                         (4096, 16384), (16384, 4096)]
    /// The rows the products and the attentions are measured at.
    package static let gridHeights = [1024, 4096]
    /// The attentions' heads and head dimension.
    package static let gridAttention: [(heads: Int, headDim: Int)] = [(16, 128), (32, 128)]

    /// **The relative rates a render's prediction starts from** (`RenderLoop.Basis`): the GPU alone's rate on a product and the
    /// system kernel's on an attention, TFLOP/s of the M1 Pro (3.5 and 1.5, the old map's own figures: attention runs at 43 % of
    /// a product's rate). Only their ratio means anything; on a Mac where it differs, the prediction is wrong by that much and
    /// the render's own check (predicted against measured time) is what says so.
    package static let referenceRates = CapacityClasses.Rates(product: ["gpu": 3.5e12], attention: ["graph": 1.5e12])

    // MARK: - Machine

    /// **Which engines and kernels exist on this Mac**, as the machine layer found them (a conductor built, a Core ML
    /// program that loads, a `TensorGEMM` with its kernels, a flash kernel that compiles).
    package struct Capabilities: Sendable, Equatable, Codable {
        package var amx = false
        package var ane = false
        package var bf16Weights = false
        package var bf16Rounding = false
        package var int8Weights = false
        package var q8Weights = false
        package var flash = false
        package var headSplit = false

        package init(amx: Bool = false, ane: Bool = false, bf16Weights: Bool = false, bf16Rounding: Bool = false,
                     int8Weights: Bool = false, q8Weights: Bool = false, flash: Bool = false, headSplit: Bool = false) {
            self.amx = amx; self.ane = ane; self.bf16Weights = bf16Weights; self.bf16Rounding = bf16Rounding
            self.int8Weights = int8Weights; self.q8Weights = q8Weights; self.flash = flash; self.headSplit = headSplit
        }
    }

    /// The rows from which an engine or a form is worth asking (`EngineSettings`' defaults).
    package struct Limits: Sendable, Equatable {
        package var amxRows = EngineSettings.defaultAMXMinimumRows
        package var aneRows = 1024
        package var roundingRows = 256
        package init(amxRows: Int = 512, aneRows: Int = 1024, roundingRows: Int = 256) {
            self.amxRows = amxRows; self.aneRows = aneRows; self.roundingRows = roundingRows
        }
    }

    // MARK: - Entries

    /// What an entry computes.
    package enum Work: Sendable, Equatable {
        case product(k: Int, n: Int, m: Int)
        case attention(sequence: Int, heads: Int, headDim: Int)
    }

    /// **The seconds per value a weight's preparation takes**, by form: widened to fp32 for the engines that read it so, copied
    /// as published for its native routes (the widening probes, `Diagnostic.measureWidening`).
    package typealias Preparation = [RouteTable.WeightKind: (widened: Double, copied: Double)]

    /// **A point of the grid**: its work, its weight, its routes.
    package struct Entry: Sendable, Equatable {
        package var work: Work
        package var weight: Double
        package var routes: [CapacityMeasure.Route]

        package var id: String { Self.id(work) }

        package static func id(_ work: Work) -> String {
            switch work {
            case let .product(k, n, m): return RouteTable.ClassKey.product(k: k, n: n, m: m).id
            case let .attention(s, h, d): return RouteTable.ClassKey.attention(sequence: s, heads: h, headDim: d).id
            }
        }

        /// The weight form a route reads (`nil`: widened, the engines' own).
        static func form(_ route: String) -> String? {
            let parts = route.split(separator: "/", maxSplits: 1)
            return parts.count > 1 ? String(parts[1]) : nil
        }

        /// **The class the core measures**: an attention as it is; a product with one view per weight form — the engines and
        /// that form's own routes, each run plus its preparation (`CapacityMeasure.View`), named `id` + the form's suffix.
        package func compute(preparation: Preparation = [:]) -> CapacityMeasure.ComputeClass {
            guard case let .product(k, n, _) = work else { return .init(id, weight: weight, routes: routes) }
            let values = Double(k * n)
            let views = RouteTable.WeightKind.allCases.map { kind -> CapacityMeasure.View in
                let own = routes.indices.filter { Self.form(routes[$0].id).map { $0.hasPrefix(kind.rawValue) } ?? true }
                let rate = preparation[kind]
                let extra = own.map { r in values * (Self.form(routes[r].id) == nil ? rate?.widened ?? 0 : rate?.copied ?? 0) }
                return .init(id + RouteTable.ClassKey.suffix(kind), routes: own, extra: extra)
            }
            return .init(id, weight: weight, routes: routes, views: views)
        }
    }

    // MARK: - Routes

    /// **The routes a product may take on this Mac**, simplest first: the engines, then each weight form's own.
    package static func productRoutes(m: Int, caps: Capabilities, limits: Limits = Limits()) -> [CapacityMeasure.Route] {
        var routes: [CapacityMeasure.Route] = [.init("gpu", engines: ["gpu"])]
        let amx = caps.amx && m >= limits.amxRows
        let ane = caps.ane && m >= limits.aneRows
        if amx { routes.append(.init("gpu+amx", engines: ["gpu", "amx"])) }
        if ane { routes.append(.init("gpu+ane", efficient: true, engines: ["gpu", "ane"])) }
        if ane && amx { routes.append(.init("gpu+ane+amx", efficient: true, engines: ["gpu", "ane", "amx"])) }
        if caps.bf16Weights { routes.append(.init("gpu/bf16", engines: ["gpu"])) }
        if caps.bf16Weights && caps.bf16Rounding && m >= limits.roundingRows { routes.append(.init("gpu/bf16act", engines: ["gpu"])) }
        if caps.int8Weights { routes.append(.init("gpu/int8", engines: ["gpu"])) }
        if caps.int8Weights && amx { routes.append(.init("gpu+amx/int8", engines: ["gpu", "amx"])) }
        if caps.q8Weights { routes.append(.init("gpu/q8", engines: ["gpu"])) }
        if caps.q8Weights && amx { routes.append(.init("gpu+amx/q8", engines: ["gpu", "amx"])) }
        return routes
    }

    /// **The routes an attention class may take**: the graph, flash, the head split.
    package static func attentionRoutes(sequence: Int, caps: Capabilities) -> [CapacityMeasure.Route] {
        var routes: [CapacityMeasure.Route] = [.init("graph", engines: ["gpu"])]
        if caps.flash { routes.append(.init("flash", engines: ["gpu"])) }
        if caps.headSplit && sequence >= HeadSplitAttention.minimumSequence { routes.append(.init("split", engines: ["gpu", "cpu"])) }
        return routes
    }

    // MARK: - The catalogue

    /// **The grid's points with this Mac's routes**: the products, then the attentions, at each height, every one weighing 1.
    package static func catalogue(caps: Capabilities, limits: Limits = Limits()) -> [Entry] {
        var entries: [Entry] = []
        for m in gridHeights {
            for s in gridShapes {
                entries.append(Entry(work: .product(k: s.k, n: s.n, m: m), weight: 1, routes: productRoutes(m: m, caps: caps, limits: limits)))
            }
        }
        for s in gridHeights {
            for a in gridAttention {
                entries.append(Entry(work: .attention(sequence: s, heads: a.heads, headDim: a.headDim), weight: 1,
                                     routes: attentionRoutes(sequence: s, caps: caps)))
            }
        }
        return entries
    }

    // MARK: - The table

    /// **The GPU's share of a pair that ended both engines together**, measured for a class when its route was first asked.
    package struct Cut: Sendable, Equatable {
        package var amx: Double?
        package var ane: Double?
        package init(amx: Double? = nil, ane: Double? = nil) { self.amx = amx; self.ane = ane }
    }

    /// **The table a plan makes**: its decisions as class rules over the grids the cuts give (`cuts[i]` for `entries[i]`,
    /// `classes[i]` the class the plan measured for it), every decision pinning its weight's form at its point, and a weight
    /// read in one form at both heights of a shape (`RouteTable.unifyWeightPaths`: the taller's, the heavier of a render). A
    /// product of no point — every model's — takes the nearest point's rule within reach (`RouteTable.Nearest`), and only
    /// beyond it the simplest route.
    package static func table(plan: CapacityMeasure.Plan, entries: [Entry], classes: [CapacityMeasure.ComputeClass], cuts: [Cut],
                              limits: Limits = Limits()) -> RouteTable {
        var amxPoints: [Conductor.Shares.Point] = [], anePoints: [Conductor.Shares.Point] = []
        var seenAMX = Set<[Int]>(), seenANE = Set<[Int]>()
        var first = 0                         // the entry's first decision (its bf16 view, for a product)
        for (i, entry) in entries.enumerated() {
            defer { first += classes.indices.contains(i) ? classes[i].decisions.count : 1 }
            guard cuts.indices.contains(i), case let .product(k, n, m) = entry.work else { continue }
            if let s = cuts[i].amx, seenAMX.insert([k, n, m]).inserted { amxPoints.append(.init(k: k, n: n, m: m, ratio: (1 - s) / s)) }
            if let a = cuts[i].ane, seenANE.insert([k, n, m]).inserted {
                // When the three take the class, the ANE's side is the three-way split of the two pairs' (`Diagnostic.threeWayShare`).
                var t = a
                if plan.decisions.indices.contains(first), plan.decisions[first].chosen == "gpu+ane+amx", let s = cuts[i].amx {
                    t = Diagnostic.threeWayShare(ane: a, amx: s)
                }
                anePoints.append(.init(k: k, n: n, m: m, ratio: (1 - t) / t))
            }
        }
        var base = RouteTable()
        let shares = amxPoints.map { 1 / (1 + $0.ratio) }.sorted()
        base.products.amx = .init(enabled: false, gpuShare: shares.isEmpty ? 0.631 : shares[shares.count / 2],
                                  grid: .init(Conductor.Shares(amxPoints)))
        base.products.ane = .init(enabled: false, gpuShare: 0.5, minimumRows: limits.aneRows, grid: .init(Conductor.Shares(anePoints)))
        var table = RouteTable(plan: plan, classes: classes.flatMap(\.decisionClasses), base: base)
        var groups: [RouteTable.WeightGroup: [Int: Double]] = [:]
        for entry in entries {
            guard case let .product(k, n, m) = entry.work else { continue }
            for kind in RouteTable.WeightKind.allCases {
                groups[.init(phase: .denoising, kind: kind, k: k, n: n), default: [:]][m] = Double(m)
            }
        }
        table.unifyWeightPaths(groups: groups)
        table.products.classes.append(.init(engines: .gpu))
        return table
    }

    // MARK: - Regime

    /// **One engine's witness**: a fixed product timed at the start and at the end of the map (seconds, a median), and the
    /// relative standard error of each.
    package struct EngineWitness: Codable, Sendable, Equatable {
        package var engine: String
        package var before: Double
        package var after: Double
        /// Relative standard error of `before` and `after` (the median of n runs: 1.253 · σ / √n, σ from the MAD).
        package var errorBefore: Double
        package var errorAfter: Double

        package init(engine: String, before: Double, after: Double, errorBefore: Double = 0, errorAfter: Double = 0) {
            self.engine = engine; self.before = before; self.after = after
            self.errorBefore = errorBefore; self.errorAfter = errorAfter
        }

        /// ln(after / before): how much longer the engine takes warm than it did cold.
        package var logRatio: Double { before > 0 && after > 0 ? log(after / before) : 0 }
        /// Two standard errors of `logRatio`: what the witness cannot tell from noise.
        package var allowance: Double { 2 * (errorBefore * errorBefore + errorAfter * errorAfter).squareRoot() }

        /// The heat factor: `after / before` when it stands out of the noise, else 1.
        package var factor: Double { abs(logRatio) > allowance ? exp(logRatio) : 1 }
    }

    /// **The heat of each engine at the regime of a render** — the factors `CapacityMeasure.Heat` multiplies the routes
    /// by. Pairs are measured within seconds; the table is decided at the end-of-map regime.
    package static func heat(_ witnesses: [EngineWitness]) -> CapacityMeasure.Heat {
        .init(Dictionary(witnesses.map { ($0.engine, $0.factor) }, uniquingKeysWith: { a, _ in a }))
    }

    /// The witnesses' worst |ln after/before| per engine and their span: the second-order residue of a drift the pairs leave.
    package static func witness(_ witnesses: [EngineWitness], span: Double) -> CapacityMeasure.Witness {
        .init(drift: Dictionary(witnesses.map { ($0.engine, abs($0.logRatio)) }, uniquingKeysWith: max), span: span)
    }

    // MARK: - The verdict

    /// **Green or orange (M6)**, and why. Green: the map is complete (residual loss ≤ ε), every route
    /// the table uses was judged right, the peak is under the need and the need under B, the seconds are in the budget.
    package struct Verdict: Codable, Sendable, Equatable {
        package var green: Bool
        /// Why not, as codes: `loss`, `wrong`, `none`, `memory`, `time`.
        package var reasons: [String]

        package init(green: Bool, reasons: [String]) { self.green = green; self.reasons = reasons }

        package init(residualLoss: Double, epsilon: Double, statuses: [CapacityMeasure.Status],
                     wrongChosen: Bool, memoryHeld: Bool, seconds: Double, timeBudget: Double = CapacityMap.budget) {
            var why: [String] = []
            if residualLoss > epsilon { why.append("loss") }
            if wrongChosen { why.append("wrong") }
            if statuses.contains(.none) { why.append("none") }
            if !memoryHeld { why.append("memory") }
            if seconds > timeBudget * 1.02 { why.append("time") }
            self.init(green: why.isEmpty, reasons: why)
        }
    }
}
