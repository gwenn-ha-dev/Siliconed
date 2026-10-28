import Foundation

/// **What the capacity map decides over: the classes of a render, merged across everything this Mac may be asked to
/// render, each with its routes in the order the measuring core wants them (simplest first), its weight, and the
/// regime the table is decided at.** Pure: no Metal, no clock, no weights — the machine enters through `Capabilities`
/// (which engines and kernels exist here) and through the witnesses.
///
///     requests (5 families × Standard / Compact / Light × 512² / 1024², Faster Rendering on)
///        │        CapacityClasses.classes(for:)          every product k×n at its real height, every attention
///        ├─ share        executions · FLOP ÷ Σ over the request          (neutral rates: the FLOP share)
///        ├─ entry        (weight kind, k×n, height) clustered  |  (S, heads, head dim) clustered
///        │                  a weight = the MAX of its shares over the requests that run it — not the sum, not the mean
///        ├─ cut          heaviest first, until `coverage` of the total or `maxEntries`; the rest is the tail
///        └─ routes       product: gpu · gpu+amx · gpu+ane · gpu+ane+amx   and the weight's own forms
///                        (gpu/bf16, gpu/bf16act, gpu/int8, gpu+amx/int8, gpu/q8, gpu+amx/q8)
///                        attention: graph · flash · split
///
/// What sets it apart — what a catalogue written by analogy with "the shapes of the model in hand" would get wrong:
///
///   - **The map does not know which family will be rendered, nor in which version**, so a class's weight is the
///     **maximum of its shares** over the requests that run it. A sum would let the families that happen to share a
///     shape outweigh the one that does not; a mean would dilute the shapes only one family runs below the cut — and
///     that family would render on the simplest route. With the maximum no family is sacrificed: a class that is 5 %
///     of Krea 2's render and 0 for the four others weighs 5 %, as it will for the person who renders Krea 2.
///     The weights then add up to more than 1; only their ratios mean anything (`CapacityMeasure.plan` normalises).
///   - **Neighbouring shapes are clustered, not rounded**: Z-Image's `4 096` and `4 128` rows, Krea 2's `4 112`, MOSS-Image's
///     `4 117` are one measurement at 4 100 ± 30 rows (within `heightGap` of the heaviest member), and so are Z-Image's
///     3840×10240 and Qwen's 4096×12288 (within `shapeRatio` on k and on n, the same weight kind), measured once at the
///     heaviest member; the others get their route from the table's interpolation (`RouteTable.interpolatedRoute`) or,
///     at run time, from the nearest measured rule (`RouteTable.Nearest`) — which is also what a model the catalogue does
///     not list gets.
///   - **A weight's kind is part of the class** (the same k×n in bf16 on the Standard and in int8 on the Compact are two
///     classes: their native routes are not alternatives to each other), and **the id says it**: `product:3072x16384@4117`
///     for bf16, `product:3072x16384@4117~int8` for int8. A K-quant (Light) widens on the CPU like any weight: its
///     classes fold into the bf16 ones of the same shape, they have no native route of their own.
///   - **Routes are static facts of this Mac and of the shape, not results**: the AMX from `amxMinimumRows`, the ANE
///     only for a denoiser's rows and from `aneMinimumRows`, a native weight route only if the kernel was built,
///     the head split from `HeadSplitAttention.minimumSequence`. What may be wrong is found by the probe's judge, one
///     route of one class at a time.
///   - **The tail is part of the denominator.** Classes under the cut are not decided by a pair, but their weight stays
///     in `CapacityMeasure.plan`'s total (`extraWeight`): a loss is a fraction of the render, not of the part we measured.
///   - **The regime is read from the engines' own witnesses** (`Regime`): a witness is a fixed product timed at the start and
///     at the end of the map on the GPU, the AMX and the ANE; the ratio is that engine's heat factor — *unless* it is
///     inside what the witness's own dispersion allows (two standard errors), in which case it is 1. The noise of a
///     quiet M1 Pro (±3 %) must not flip a 1 % tie between the GPU and the AMX; the Neo's AMX (×2.06) must.
///   - **Green is a property of the map, not of the hour** (`Verdict`): complete (residual loss under ε, the tail small),
///     every chosen route judged right, the peak under the need under B, the seconds in the budget. Heat, battery, Low
///     Power Mode and another process's swap are facts of the regime, written beside the map, never a reason for it to
///     be orange.
package enum CapacityMap {
    /// Wall seconds for the whole map, on any Mac, settings and witnesses included.
    package static let budget = 60.0
    /// Rows within which two heights of the same k×n are one measurement.
    package static let heightGap = 64
    /// **The ratio within which two shapes are one measurement** (on k and on n; on the heads of an attention): 3 072, 3 840
    /// and 4 096 are one class, 9 216 to 12 288 another. The engines' ratios move with the shape in log, and smoothly: a
    /// measurement per family's exact shape spent five families' minute on 133 classes, a third of the weight under the
    /// cut. A shape the map measured for another stands for it in the table (`RouteTable.Nearest`).
    package static let shapeRatio = 1.28
    /// The classes the pairs decide: the heaviest, until this share of the total weight …
    package static let coverage = 0.96
    /// … or this many.
    package static let maxEntries = 40
    /// The tail the map may leave to interpolation and still be complete (share of the total weight).
    package static let tailLimit = 0.05

    /// **The relative cost of the classes before anything is measured** (`CapacityClasses.weights`): the GPU alone's rate on a
    /// product and the system kernel's on an attention, TFLOP/s of the M1 Pro (3.5 and 1.5, the old map's own figures: attention
    /// runs at 43 % of a product's rate, so a FLOP of attention weighs 2.3 FLOP of a product). Only their ratio means anything;
    /// on a Mac where it differs, the weights are wrong by that much and the render's own check (predicted against measured
    /// time) is what says so. Neutral rates left the attention under every product class, and the head split — 1.45× on
    /// attention — unmeasured.
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
        case product(k: Int, n: Int, m: Int, kind: RouteTable.WeightKind)
        case attention(sequence: Int, heads: Int, headDim: Int)
    }

    /// **A class of the map**: its work, its weight, its routes.
    package struct Entry: Sendable, Equatable {
        package var work: Work
        package var weight: Double
        /// The heights of the catalogue this entry stands for at its own shape (products), the entry's own first.
        package var heights: [Int]
        package var routes: [CapacityMeasure.Route]
        /// How many shapes and heights of the catalogue the entry measures for (its own included).
        package var members = 1

        package var id: String { Self.id(work) }
        package var compute: CapacityMeasure.ComputeClass { .init(id, weight: weight, routes: routes) }

        package static func id(_ work: Work) -> String {
            switch work {
            case let .product(k, n, m, kind): return RouteTable.ClassKey.product(k: k, n: n, m: m).id + RouteTable.ClassKey.suffix(kind)
            case let .attention(s, h, d): return RouteTable.ClassKey.attention(sequence: s, heads: h, headDim: d).id
            }
        }
    }

    // MARK: - Routes

    package static let attentionRoutes = CapacityClasses.attentionRoutes

    /// **The routes a product class may take on this Mac**, simplest first.
    package static func productRoutes(kind: RouteTable.WeightKind, m: Int, denoising: Bool, caps: Capabilities,
                                      limits: Limits = Limits()) -> [CapacityMeasure.Route] {
        var routes: [CapacityMeasure.Route] = [.init("gpu", engines: ["gpu"])]
        let amx = caps.amx && m >= limits.amxRows
        let ane = caps.ane && denoising && m >= limits.aneRows
        if amx { routes.append(.init("gpu+amx", engines: ["gpu", "amx"])) }
        if ane { routes.append(.init("gpu+ane", efficient: true, engines: ["gpu", "ane"])) }
        if ane && amx { routes.append(.init("gpu+ane+amx", efficient: true, engines: ["gpu", "ane", "amx"])) }
        switch kind {
        case .bf16:
            if caps.bf16Weights { routes.append(.init("gpu/bf16", engines: ["gpu"])) }
            if caps.bf16Weights && caps.bf16Rounding && m >= limits.roundingRows { routes.append(.init("gpu/bf16act", engines: ["gpu"])) }
        case .int8:
            if caps.int8Weights { routes.append(.init("gpu/int8", engines: ["gpu"])) }
            if caps.int8Weights && amx { routes.append(.init("gpu+amx/int8", engines: ["gpu", "amx"])) }
        case .q8:
            if caps.q8Weights { routes.append(.init("gpu/q8", engines: ["gpu"])) }
            if caps.q8Weights && amx { routes.append(.init("gpu+amx/q8", engines: ["gpu", "amx"])) }
        }
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

    /// Every render this Mac may be asked for, at the two formats the map can see (512² and 1024²; a taller image is the
    /// table's extrapolation). `families`: only these — a Mac with one model installed is not asked about the four others
    /// (the catalogue is then narrow enough for the minute to cover it).
    package static func requests(families: [Family] = Family.allCases) -> [CapacityClasses.Request] {
        var requests: [CapacityClasses.Request] = []
        for family in families {
            for variant in Variant.allCases {
                for side in [512, 1024] {
                    let r = CapacityClasses.Request(family: family, variant: variant, width: side, height: side)
                    if CapacityClasses.classes(for: r) != nil { requests.append(r) }
                }
            }
        }
        return requests
    }

    package struct Catalogue: Sendable, Equatable {
        /// The classes the pairs decide, heaviest first.
        package var entries: [Entry]
        /// The weight of everything under the cut (same scale as the entries').
        package var tail: Double
        /// The whole catalogue, merged — what the table is completed over (`RouteTable.init(plan:…renderClasses:)`).
        package var render: [CapacityClasses.Class]

        package var total: Double { entries.reduce(tail) { $0 + $1.weight } }
        /// The tail's share of the total weight.
        package var tailShare: Double { tail / max(total, 1e-12) }
    }

    private struct Key: Hashable {
        var kind: RouteTable.WeightKind?      // nil: an attention
        var k = 0, n = 0, m = 0, sequence = 0, heads = 0, headDim = 0
    }

    /// **Builds the catalogue**: weights as the maximum of the shares over `requests`, heights clustered, the cut applied.
    package static func catalogue(requests: [CapacityClasses.Request] = requests(), caps: Capabilities, limits: Limits = Limits(),
                                  coverage: Double = coverage, maxEntries: Int = maxEntries) -> Catalogue {
        var lists: [[CapacityClasses.Class]] = []
        // Per request: the share of each (kind, shape, height), summed over the phases and families' repeats.
        var shares: [[Key: Double]] = []
        var denoising = Set<Key>()
        for r in requests {
            guard let classes = CapacityClasses.classes(for: r) else { continue }
            lists.append(classes)
            let weights = CapacityClasses.weights(classes, family: r.family, rates: referenceRates)
            var these: [Key: Double] = [:]
            for (c, w) in zip(classes, weights) where w > 0 {
                let key: Key
                switch c.shape {
                case let .product(k, n, m, _): key = Key(kind: c.weightKind ?? .bf16, k: k, n: n, m: m)
                case let .attention(s, _, heads, _, dim, _): key = Key(kind: nil, sequence: s, heads: heads, headDim: dim)
                }
                these[key, default: 0] += w
                if c.phase == .denoising { denoising.insert(key) }
            }
            shares.append(these)
        }
        func weight(_ key: Key) -> Double { shares.map { $0[key] ?? 0 }.max() ?? 0 }

        // Neighbouring shapes, one measurement: heaviest first, a key joins the first cluster of its kind (an attention's:
        // heads and head dim) whose representative is within `shapeRatio` on each side and `heightGap` rows in height.
        var clusters: [[Key]] = []
        let tolerance = log(shapeRatio)
        func near(_ a: Key, _ b: Key) -> Bool {
            guard a.kind == b.kind else { return false }
            let (ka, na, ma, kb, nb, mb) = a.kind == nil
                ? (a.heads, a.headDim, a.sequence, b.heads, b.headDim, b.sequence) : (a.k, a.n, a.m, b.k, b.n, b.m)
            return abs(log(Double(ka) / Double(kb))) <= tolerance && abs(log(Double(na) / Double(nb))) <= tolerance
                && abs(ma - mb) <= heightGap
        }
        let keys = Set(shares.flatMap { $0.keys }).sorted { (weight($1), $0.k, $0.n, $0.m, $0.sequence, $0.heads) < (weight($0), $1.k, $1.n, $1.m, $1.sequence, $1.heads) }
        for key in keys {
            if let i = clusters.firstIndex(where: { near($0[0], key) }) { clusters[i].append(key) } else { clusters.append([key]) }
        }
        var entries: [Entry] = []
        for cluster in clusters {
            // The cluster's weight in a request is the sum of its members' shares there; the map keeps the maximum.
            let w = shares.map { these in cluster.reduce(0.0) { $0 + (these[$1] ?? 0) } }.max() ?? 0
            let rep = cluster[0]
            if let kind = rep.kind {
                let routes = productRoutes(kind: kind, m: rep.m, denoising: cluster.contains { denoising.contains($0) },
                                           caps: caps, limits: limits)
                let heights = [rep.m] + Set(cluster.filter { $0.k == rep.k && $0.n == rep.n && $0.m != rep.m }.map(\.m)).sorted()
                entries.append(Entry(work: .product(k: rep.k, n: rep.n, m: rep.m, kind: kind), weight: w, heights: heights, routes: routes,
                                     members: cluster.count))
            } else {
                entries.append(Entry(work: .attention(sequence: rep.sequence, heads: rep.heads, headDim: rep.headDim), weight: w,
                                     heights: [rep.sequence], routes: attentionRoutes(sequence: rep.sequence, caps: caps),
                                     members: cluster.count))
            }
        }
        entries.sort { ($1.weight, $0.id) < ($0.weight, $1.id) }
        let total = entries.reduce(0) { $0 + $1.weight }
        var kept: [Entry] = [], covered = 0.0
        for e in entries {
            if kept.count >= maxEntries || covered >= coverage * total { break }
            kept.append(e); covered += e.weight
        }
        return Catalogue(entries: kept, tail: total - covered, render: CapacityClasses.merge(lists))
    }

    // MARK: - The table

    /// **The GPU's share of a pair that ended both engines together**, measured for a class when its route was first asked.
    package struct Cut: Sendable, Equatable {
        package var amx: Double?
        package var ane: Double?
        package init(amx: Double? = nil, ane: Double? = nil) { self.amx = amx; self.ane = ane }
    }

    /// **The table a plan makes**: its decisions as class rules over the grids the cuts give (`cuts[i]` for `catalogue.entries[i]`), the
    /// catalogue completing it to the heights nobody measured. A product of no class — another format, another model — takes
    /// the nearest measured class's rule (`RouteTable.Nearest`), never a catch-all: « GPU alone » for every shape the catalogue
    /// did not list left a 1024×1536 render without the ANE.
    package static func table(plan: CapacityMeasure.Plan, catalogue: Catalogue, cuts: [Cut], limits: Limits = Limits()) -> RouteTable {
        var amxPoints: [Conductor.Shares.Point] = [], anePoints: [Conductor.Shares.Point] = []
        var seenAMX = Set<[Int]>(), seenANE = Set<[Int]>()
        for (i, entry) in catalogue.entries.enumerated() where cuts.indices.contains(i) {
            guard case let .product(k, n, m, _) = entry.work else { continue }
            if let s = cuts[i].amx, seenAMX.insert([k, n, m]).inserted { amxPoints.append(.init(k: k, n: n, m: m, ratio: (1 - s) / s)) }
            if let a = cuts[i].ane, seenANE.insert([k, n, m]).inserted {
                // When the three take the class, the ANE's side is the three-way split of the two pairs' (`Diagnostic.threeWayShare`).
                var t = a
                if plan.decisions[i].chosen == "gpu+ane+amx", let s = cuts[i].amx { t = Diagnostic.threeWayShare(ane: a, amx: s) }
                anePoints.append(.init(k: k, n: n, m: m, ratio: (1 - t) / t))
            }
        }
        var base = RouteTable()
        let shares = amxPoints.map { 1 / (1 + $0.ratio) }.sorted()
        base.products.amx = .init(enabled: false, gpuShare: shares.isEmpty ? 0.631 : shares[shares.count / 2],
                                  grid: .init(Conductor.Shares(amxPoints)))
        base.products.ane = .init(enabled: false, gpuShare: 0.5, minimumRows: limits.aneRows, grid: .init(Conductor.Shares(anePoints)))
        return RouteTable(plan: plan, classes: catalogue.entries.map(\.compute), base: base, renderClasses: catalogue.render)
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

    /// **Green or orange (M6)**, and why. Green: the map is complete (residual loss ≤ ε, the tail under `tailLimit`), every route
    /// the table uses was judged right, the peak is under the need and the need under B, the seconds are in the budget.
    package struct Verdict: Codable, Sendable, Equatable {
        package var green: Bool
        /// Why not, as codes: `loss`, `tail`, `wrong`, `none`, `memory`, `time`.
        package var reasons: [String]

        package init(green: Bool, reasons: [String]) { self.green = green; self.reasons = reasons }

        package init(residualLoss: Double, epsilon: Double, tailShare: Double, statuses: [CapacityMeasure.Status],
                     wrongChosen: Bool, memoryHeld: Bool, seconds: Double, timeBudget: Double = CapacityMap.budget) {
            var why: [String] = []
            if residualLoss > epsilon { why.append("loss") }
            if tailShare > CapacityMap.tailLimit { why.append("tail") }
            if wrongChosen { why.append("wrong") }
            if statuses.contains(.none) { why.append("none") }
            if !memoryHeld { why.append("memory") }
            if seconds > timeBudget * 1.02 { why.append("time") }
            self.init(green: why.isEmpty, reasons: why)
        }
    }
}
