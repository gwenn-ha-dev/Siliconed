import Foundation

/// **The solver: from a machine's model and one class of computation to the exact assignment that runs it soonest, and the time
/// it predicts.** Pure: no Metal, no clock, no weights. It *consumes* `CapacityModel` (lines, sharing, the ANE's staircase, heat,
/// preparation) and writes what the engine reads — `ProductAssignment`, `AttentionAssignment` (`CapacityPlan`) — one class at a
/// time, the integers the engines will run, never a ratio for the engine to round.
///
///     product (k, n, m, weight format, phase)
///         │   each path    widened (any set of gpu · amx · ane) · bf16 · bf16 rounded (gpu · gpu + ane) · native int8 / q8 (gpu · gpu + ane on int8)
///         │                · native rounded int8 / q8 (gpu · gpu + ane on int8)
///         │   each set     the balanced split on the lines (`CapacityModel.split`), then the integers around it on the lattice
///         ▼                the engines can run: the GPU's rows on MPS's 64-row tile, the ANE's in units of 256
///     t = preparation(path) + T(rows)               one weight is prepared per call, at every height
///         ▼
///     the ANE before the AMX                        the best with the ANE within 15 % of the best with the AMX: the AMX's dropped
///         ▼
///     the least t, the simplest set on a tie        ──▶  ProductAssignment
///     attention (S, keys, heads, kv heads, dim, cross) ──▶ graph · flash (where the caller tries it) · split (whole kv groups on the CPU)
///                                                          · tiles (three stages by tiles of rows, fp32) · tiles16 (fp16 operands: `Constraints.allowsHalf`)
///
/// What sets it apart — what a solver written by analogy with "choose the fastest route per class" would get wrong:
///
///   - **It writes rows, so it times rows.** There is no ratio between the solver and the engine, hence no rounding, clamp or
///     quantum the prediction could miss: the candidates are the integers around the continuous balance, each timed by the
///     model as it will run; an ANE count that falls on the wrong side of a step of 256 shows in the time, not as a silent loss.
///   - **One class, one answer.** The same product in the encoder and in the denoiser, in bf16 and in int8, is four classes; none
///     lends its share to another. A preparation is paid at every call (`WeightProducts.prepare` widens or stages the weight inside
///     each `linear`, whatever the height), so a widened weight and a native one are not the same price; a K-quant has its own
///     figure (`kquantPreparation`).
///   - **An engine may be alone.** The AMX alone, the ANE alone, are candidates like any set: a Mac whose GPU is the slow engine
///     is served without a line of the solver changing. The ANE alone is the whole product (the model rounds the rows it pays for
///     up to the unit).
///   - **What the caller's code cannot do is not offered.** An encoder never has the ANE or a rounded activation (its products are
///     thin — m is the prompt's tokens, under the floors of `Limits` for most prompts — and its output conditions every evaluation);
///     a process that did not bring the ANE up (a Mac whose map gives it nothing, the accelerators off) is solved without it
///     (`Constraints.allowsANE`); flash is the caller's (`flash:`), not a family's guess; the split cuts by whole kv groups, the GPU
///     keeping one at least. The version restricts nothing here: it reaches the solver through the weight formats of its classes.
///   - **The model's domain is a floor, not a bound of the engine.** Its lines were measured from ≈ 4 GFLOP: below `Limits` the AMX,
///     the ANE and the rounding are not asked about (their t₀ would be extrapolated), and the GPU takes the product. The engine
///     itself imposes none of them.
///   - **Under the noise, a written order — never the noise**: two assignments closer than the map's own noise (twice `CapacityModel.noise`,
///     the difference of two figures each that uncertain, and never under the 5 % two calm maps differ by) are one, and the first in `canonical` order is taken: without the AMX before
///     with it, with the ANE before without (on equal time the engines' efficiency decides: the ANE's watts are the fewest, the AMX's the
///     most — a margin may favour the ANE, never the AMX), fewer engines, then the finer operand (the weight widened to fp32, the weight as
///     published on an fp32 activation, its activation rounded; the integers on an fp32 activation, then rounded), then the order the
///     candidates are listed in. Two GPU forms of a thin product pay one fixed cost (`CapacityFit.thinLines`): their gap is read to the wide
///     margin, the thin one being the width of fixed costs (`oneFixedCost`). Stateless: the same map gives the same plan, and
///     a second map of the same Mac changes a choice only where the gain passes the noise. A pair beside the ANE is offered at every
///     unit of the ANE's rows, the balance's or not, the most rows first (`pairRows`): a gap either side of zero is read the same way,
///     and the ANE's share within the margin is the written order's too. A map without rounds keeps the old rule (a part in a billion).
///   - **A set is taken on what a render pays, past the error of the difference.** Its seconds against the simpler assignment's (the
///     best on a strict subset of its kernels), the set kept only where the gap passes twice √(se_S² + se_alt²): the set's standard
///     error from three chains where two disagreed (`Sharing.noise`, the MAD of the median chain), the map's σ for every other figure
///     (`passingTheirError`). A gain under it is one the next map reverses. **A set with the ANE is read the other way** (the ANE on a
///     tie, decided 3/11/2026): dropped only where proven slower by as much, taken within that error of the fastest, its most rows first.
///   - **The ANE against the AMX, and only there, a preference — not a figure of speed.** The best candidate with the ANE and
///     without the AMX may be up to `aneOverAMX` (15 %) slower than the best with the AMX: then every candidate with the AMX is
///     dropped, and the rest is solved as everywhere (the least t, the noise, the written order). Against the GPU alone, or between
///     two sets with the ANE, time alone decides: the preference is the ANE's watts and silence beside the CPU's cores, not a
///     discount on the ANE. Decided by the user on 3/11/2026 (15 %; on the M1 Pro a render with the ANE drew 31–37 W where the
///     AMX drew 38–45, 1.22–1.24×: the joules break even near 22 % slower). The rule of 30/10 (the ANE taken at 90 % of the
///     best speed, a 10 % malus on the AMX) went with the old route table on the same day, without a decision; this replaces it.
///   - **No AMX beside a native weight.** Its columns would be widened on the CPU at every call and multiplied in fp32 — a widening,
///     the cost the native path exists to avoid —, timed by a line (`amx/int8`, `amx/q8`) no probe can measure as the route runs (the
///     widening of a share of columns inside the AMX's side, beside a GPU reading integers); the AMX beside the GPU on an 8-bit weight is
///     the widened route's `gpu+amx`, whose lines, sharing and widening the map measures; and the rounded activation the GPU's tensor units
///     take cannot have it beside (the AMX reads the fp32 one). Retired from the solver on 3/11/2026 (`ProductAssignment` still runs a
///     pinned one).
///
/// The second time (E5, the overlap between blocks) enters through `Combine`: the solver gives each block its preparation,
/// its compute and what each engine is busy for, and `combine` turns the blocks into a total. `sequential` is the sum.
package enum CapacitySolver {
    typealias Kernel = CapacityModel.Kernel
    typealias Engine = CapacityModel.Engine

    /// How much slower than the best with the AMX the best with the ANE (and without the AMX) may be and still be taken.
    package static let aneOverAMX = 0.15

    // MARK: - What goes in

    /// The rows from which an engine or a form is worth asking: the model's lines start at ≈ 4 GFLOP.
    package struct Limits: Sendable, Equatable {
        /// The AMX's floor, the AMX's floor in rows (`amx_min` of earlier builds): what its predictions and its render check still read by default.
        package static let defaultAMXRows = 512
        package var amxRows = Limits.defaultAMXRows
        package var aneRows = 1024
        package var roundingRows = 256
        package init(amxRows: Int = Limits.defaultAMXRows, aneRows: Int = 1024, roundingRows: Int = 256) {
            self.amxRows = amxRows; self.aneRows = aneRows; self.roundingRows = roundingRows
        }
    }

    package struct Constraints: Sendable {
        package var limits: Limits
        /// Judge the lines at their sustained rate (`heat`): a render outlasts the cold minute.
        package var sustained: Bool
        /// Kernels not proven on this Mac (a route the judge refused), left out whatever their line says.
        package var excluded: Set<CapacityModel.Kernel>
        /// The ANE may be asked for: this process brings it up — at every version, the Standard's included (decided 2/11/2026: its
        /// programs lift each weight column and activation row by a power of two into fp16's range and hand a row that overflows back to
        /// the GPU, so fp16 keeps what bf16 holds). False: a Mac whose map gives it nothing, the accelerators off, an encoder.
        package var allowsANE: Bool
        /// **An operand may be rounded below fp32 on the GPU's tensor units**: an activation to bf16 before a product on a bf16 weight
        /// (`gpu/bf16act`) or on an 8-bit one read as published (`gpu/int8act`, `gpu/q8act`: the integers are exact in bf16, the scales
        /// applied in fp32), the attention's operands to bf16 (`tilesbf16`) — P20: bf16 between two operations, never below the published
        /// weight, as a GPU in bf16 computes them, the sums fp32 — at every version, the Standard's included (decided 2/11/2026: the fp32
        /// operand was the M1 Pro's choice of speed, not a rule). The map says where it is faster; an encoder never.
        package var allowsRounding: Bool
        /// **The activation may be quantized to int8 beside a published int8 weight** (W8A8, `gpu/int8x8`): the quant's author's own
        /// forward (ComfyUI's `int8_linear`), the multiplication in the weight's format — under the rounding's permission (the
        /// accelerators on) and the plan's W8A8 switch (`CapacityPlan.allowsQuantizing`); an encoder never.
        package var allowsQuantizing: Bool

        /// **fp16 may be asked for** — the attention on fp16 operands (`tiles16`: Q, K, V and P rounded to half, fp32 sums). Derived,
        /// never set: `allowsHalf(ane:rounding:)` is the one rule, which the plan's resources and the app's sentence read too.
        package var allowsHalf: Bool { Self.allowsHalf(ane: allowsANE, rounding: allowsRounding) }

        /// **The one source of the fp16 permission**: where the ANE may compute (its rows are fp16 already) and operands may be rounded
        /// at all (the accelerators on) — at every version, the Standard's included (decided 3/11/2026, checked against the reference
        /// run in bf16 by `--fp16-attention`). What it rests on is the attention's operands, not the residual stream P17 measured: Q and
        /// K come out of the norms, V is a projection of a normed input, P lies in [0, 1]; every sum stays fp32.
        package static func allowsHalf(ane: Bool, rounding: Bool) -> Bool { ane && rounding }

        package init(limits: Limits = .init(), sustained: Bool = true, excluded: Set<CapacityModel.Kernel> = [],
                     allowsANE: Bool = true, allowsRounding: Bool = true, allowsQuantizing: Bool? = nil) {
            self.limits = limits; self.sustained = sustained; self.excluded = excluded; self.allowsANE = allowsANE
            self.allowsRounding = allowsRounding
            self.allowsQuantizing = allowsQuantizing ?? allowsRounding
        }

        /// The constraints of a request, as a render takes them: the version restricts nothing (the ANE, the rounding and fp16 at every
        /// version, decided 2/11 and 3/11/2026) — it reaches the solver through the weight formats of its classes alone.
        package init(request: CapacityClasses.Request, limits: Limits = .init(), sustained: Bool = true,
                     excluded: Set<CapacityModel.Kernel> = []) {
            self.init(limits: limits, sustained: sustained, excluded: excluded)
        }
    }

    // MARK: - What comes out

    /// **One product's answer**: the assignment, the rows each kernel takes (a native weight beside the AMX: columns), and the
    /// seconds of one execution.
    package struct ProductSolution: Sendable, Equatable {
        package var assignment: ProductAssignment
        /// The route as the engine names it (`gpu+ane`, `gpu+amx/int8`, `gpu/bf16act`…).
        package var route: String
        package var rows: [CapacityModel.Kernel: Int]
        /// Seconds of the weight's preparation, and of the product, for one execution.
        package var preparation: Double
        package var compute: Double
        /// Seconds each engine is busy in one execution (the overlap's input).
        package var busy: [CapacityModel.Engine: Double]
        package var seconds: Double { preparation + compute }
    }

    package struct AttentionSolution: Sendable, Equatable {
        package var assignment: AttentionAssignment
        package var route: String
        /// Heads each kernel takes.
        package var rows: [CapacityModel.Kernel: Int]
        package var compute: Double
        package var busy: [CapacityModel.Engine: Double]
    }

    /// **One class of the render, as the plan runs it.**
    package struct Block: Sendable, Equatable {
        package var classID: String
        package var phase: CapacityClasses.Phase
        package var executions: Int
        /// The route as the engine names it (`gpu+ane`, `gpu+amx/int8`, `split`…).
        package var route: String
        /// Rows (columns, for an AMX beside a native weight; heads, for an attention) each kernel takes.
        package var rows: [CapacityModel.Kernel: Int]
        package var product: ProductAssignment?
        package var attention: AttentionAssignment?
        /// Seconds of all the executions: the weight's preparation, and the product or attention.
        package var preparation: Double
        package var compute: Double
        /// Seconds each engine is busy over all the executions (the overlap's input).
        package var busy: [CapacityModel.Engine: Double]
        /// **The disk's share (`CapacityStream`)**, only on the block that stands for a pass's weight stream: the seconds its reads take over
        /// all the evaluations, the seconds of computation they run beneath (the pass's own), and what no computation hides. The block
        /// computes nothing (`seconds` 0); the overlap is `combine`'s.
        package var reading = 0.0
        package var beneath = 0.0
        package var exposed = 0.0
        package var seconds: Double { preparation + compute }
    }

    package typealias Combine = @Sendable ([Block]) -> Double
    /// **The hook of the overlap between blocks (E5)**: the plan's total as a function of its blocks. The sequential one adds.
    package static let sequential: Combine = { $0.reduce(0) { $0 + $1.seconds } }

    package struct Plan: Sendable {
        package var blocks: [Block]
        /// The total, `combine` of the blocks.
        package var seconds: Double
        package func seconds(of phase: CapacityClasses.Phase) -> Double {
            blocks.filter { $0.phase == phase }.reduce(0) { $0 + $1.seconds }
        }
        /// Classes the model could not time (an attention with no line): left to the engine's defaults, counted in no total.
        package var unpredicted: [String]
        package func block(_ id: String) -> Block? { blocks.first { $0.classID == id } }
    }

    // MARK: - Entry points

    /// **One product's answer**, `nil` when the model has no line for the GPU's fp32 product: nothing can be planned without the
    /// reference engine (the caller then takes the engine's defaults).
    package static func product(_ key: ProductKey, model: CapacityModel, constraints: Constraints) -> ProductSolution? {
        ProductSolver(model: model, c: constraints).solve(key)
    }

    /// **One attention's answer.** `flash`: the caller's code tries the flash kernel before the split.
    package static func attention(_ key: AttentionKey, flash: Bool, model: CapacityModel, constraints: Constraints) -> AttentionSolution? {
        ProductSolver(model: model, c: constraints).solve(key, flash: flash)
    }

    /// **Every candidate the solver weighs for a product** — not only the one it keeps: each set of engines and each form of the weight
    /// it may offer under `constraints`, timed. Empty when the model has no line for the GPU's fp32 product. The coverage guard reads
    /// it (`CoverageGuardTests`): what the solver can offer, beside what P20 permits.
    package static func candidates(_ key: ProductKey, model: CapacityModel, constraints: Constraints) -> [ProductSolution] {
        let solver = ProductSolver(model: model, c: constraints)
        return solver.candidates(key).map { solver.solution($0, key) }
    }

    /// The same for an attention: every runnable assignment it weighs, timed.
    package static func candidates(_ key: AttentionKey, flash: Bool, model: CapacityModel, constraints: Constraints) -> [AttentionSolution] {
        ProductSolver(model: model, c: constraints).candidates(key, flash: flash).map {
            AttentionSolution(assignment: $0.0, route: $0.1.route, rows: $0.1.rows, compute: $0.1.seconds, busy: $0.1.busy)
        }
    }

    /// **The seconds the model gives an assignment someone else chose** — the plan's, a forcing's: what the engine ran, timed as the solver
    /// times its own candidates. `nil` when the model has no line for a kernel the assignment uses (it was excluded, or never measured).
    package static func product(assigned a: ProductAssignment, key: ProductKey, model: CapacityModel, constraints: Constraints) -> ProductSolution? {
        ProductSolver(model: model, c: constraints).evaluate(a, key)
    }

    /// The same for an attention.
    package static func attention(assigned a: AttentionAssignment, key: AttentionKey, model: CapacityModel,
                                  constraints: Constraints) -> AttentionSolution? {
        ProductSolver(model: model, c: constraints).evaluate(a, key)
    }

    /// **Whether a family's code tries the flash kernel** on an attention: the self-attentions of Z-Image and Anima, before the split.
    package static func triesFlash(family: Family, cross: Bool) -> Bool { !cross && (family == .zImage || family == .anima) }

    /// **Every class of a render's family, solved**: the blocks, the total. The classes' executions come from `family`.
    package static func solve(model: CapacityModel, request: CapacityClasses.Request, constraints: Constraints,
                              combine: Combine = sequential) -> Plan? {
        guard let classes = CapacityClasses.classes(for: request) else { return nil }
        return solve(model: model, classes: classes, family: request.family, constraints: constraints, combine: combine)
    }

    /// `nil` when the model has no line for the GPU's fp32 product.
    package static func solve(model: CapacityModel, classes: [CapacityClasses.Class], family: Family, constraints: Constraints,
                              combine: Combine = sequential) -> Plan? {
        let solver = ProductSolver(model: model, c: constraints)
        guard solver.line(.gpu, .fp32) != nil else { return nil }
        var blocks: [Block] = [], unpredicted: [String] = []
        for cl in classes {
            let executions = cl.executions[family] ?? 0
            guard executions > 0 else { continue }
            let n = Double(executions)
            switch cl.shape {
            case let .product(k, nn, rows, format):
                let key = ProductKey(k: k, n: nn, m: rows, weight: format, phase: cl.phase)
                guard let s = solver.solve(key) else { unpredicted.append(key.description); continue }
                blocks.append(.init(classID: cl.id, phase: cl.phase,
                                    executions: executions, route: s.route, rows: s.rows, product: s.assignment, attention: nil,
                                    preparation: n * s.preparation, compute: n * s.compute, busy: s.busy.mapValues { $0 * n }))
            case let .attention(s, keys, heads, kv, dim, cross):
                let key = AttentionKey(sequence: s, keys: keys, heads: heads, kvHeads: kv, headDim: dim, cross: cross)
                guard let a = solver.solve(key, flash: triesFlash(family: family, cross: cross)) else {
                    unpredicted.append(cl.id); continue
                }
                blocks.append(.init(classID: cl.id, phase: cl.phase, executions: executions, route: a.route, rows: a.rows,
                                    product: nil, attention: a.assignment, preparation: 0, compute: n * a.compute,
                                    busy: a.busy.mapValues { $0 * n }))
            }
        }
        return Plan(blocks: blocks, seconds: combine(blocks), unpredicted: unpredicted)
    }
}

// MARK: - The solver proper

private struct ProductSolver {
    typealias Kernel = CapacityModel.Kernel
    typealias Engine = CapacityModel.Engine
    typealias Shape = CapacityClasses.Shape

    let model: CapacityModel
    let c: CapacitySolver.Constraints

    init(model: CapacityModel, c: CapacitySolver.Constraints) { self.model = model; self.c = c }

    // MARK: Lines

    func line(_ e: Engine, _ f: CapacityModel.Form) -> Kernel? {
        let k = Kernel(e, f)
        guard let l = model.lines[k], l.rate > 0, !c.excluded.contains(k) else { return nil }
        return k
    }

    func seconds(_ rows: [Kernel: Int], m: Int, k: Int, n: Int) -> Double? {
        // A set of engines the probes never measured is no candidate: keep 1 is not a measure.
        guard model.measures(rows.filter { $0.value > 0 }.map(\.key.engine)) else { return nil }
        return model.seconds(m: m, k: k, n: n, rows: rows, sustained: c.sustained)
    }
    func busy(_ rows: [Kernel: Int], m: Int, k: Int, n: Int) -> [Engine: Double] {
        let b = model.busy(m: m, k: k, n: n, rows: rows, sustained: c.sustained)
        return Dictionary(b.map { ($0.key.engine, $0.value) }, uniquingKeysWith: +)
    }

    // MARK: The lattice

    /// The GPU's rows (multiples of MPS's 64-row tile) around the balance `g0` of `rest` rows shared with the AMX, which keeps the
    /// remainder: at least one row, so that the set is two engines.
    func gpuRows(rest: Int, balance g0: Double) -> [Int] {
        let tile = GEMM.rowQuantum
        let top = (rest - 1) / tile * tile
        guard top >= tile else { return [] }
        let base = Int((g0 / Double(tile)).rounded())
        var seen = Set<Int>(), out: [Int] = []
        for d in -2...2 {
            let g = min(top, max(tile, (base + d) * tile))
            if seen.insert(g).inserted { out.append(g) }
        }
        return out
    }

    /// **The ANE's rows a pair beside the GPU is offered at**: every unit the GPU leaves a row beside, the balance's or not, each a
    /// candidate — listed the most rows first, so that within the margin the written order gives the ANE the most rows (its watts are
    /// the fewest), never the rows a flat minimum happens to fall on. Around the balance alone, the pair was no candidate wherever the
    /// balance gave the ANE nothing — a gap of a hair over zero, and the next map's hair under it, made two plans (Z-Image's
    /// `3840x10240` at 4 128 rows on an M5 Pro: 0.1 % under the GPU alone in one map, no candidate in the next); at its floor alone its
    /// best time jumped by 4 % where the balance starts to give it 768 rows (the ANE's per-call lift, which no share of a few rows pays);
    /// and the fastest split, 512 rows in one map and 768 in the next, was a minimum 1 % deep under a 5 % margin. Over the whole
    /// lattice the pair's times move with the lines, and the gaps are the solver's to read against its margin.
    func pairRows(m: Int) -> [Int] {
        let unit = NeuralEngine.rowUnit
        return unit < m ? Array(stride(from: unit, to: m, by: unit)) : []
    }

    /// The ANE's rows of `m` (units of 256, the GPU or the AMX keeping a row at least) around the balance `r0`.
    func aneRows(m: Int, balance r0: Double) -> [Int] {
        let unit = NeuralEngine.rowUnit
        let u0 = r0 / Double(unit)
        var units = Set<Int>()
        for u in (Int(u0.rounded(.down)) - 1)...(Int(u0.rounded(.up)) + 1) where u >= 1 && u * unit < m { units.insert(u) }
        return units.sorted().map { $0 * unit }
    }

    /// **What a `loraUp` product pays beyond its product, for it adds to the output instead of writing it**: the GPU's rows read the output once
    /// more (its kernel accumulates, `beta = 1`); the CPU engines' rows land in a scratch and a pass adds it to the output — `m' · n` floats read
    /// twice and written once. At the memory's bandwidth; 0 for any other class, and where the bandwidth was not measured.
    func accumulation(_ key: ProductKey, gpu: Int, offGPU rows: Int) -> Double {
        guard key.weight == .loraUp, let bandwidth = model.memoryBandwidth, bandwidth > 0 else { return 0 }
        return 4 * Double(gpu + 3 * rows) * Double(key.n) / bandwidth
    }

    // MARK: A way to compute one product

    struct Option {
        var assignment: ProductAssignment
        var rows: [Kernel: Int]
        var seconds: Double
        func uses(_ engine: Engine) -> Bool { rows.contains { $0.key.engine == engine && $0.value > 0 } }
    }

    /// **The widened path's candidates, one per set of engines** — the GPU alone, with the AMX, with the ANE, the three, then the
    /// AMX alone, the ANE alone, the two — each the best integer split the engines can run around the balance of the lines.
    func widened(_ key: ProductKey) -> [Option] {
        let (k, n, m) = (key.k, key.n, key.m)
        guard let gpu = line(.gpu, .fp32) else { return [] }
        let amx = m >= c.limits.amxRows ? line(.amx, .fp32) : nil
        let ane = c.allowsANE && key.phase == .denoising && m >= c.limits.aneRows ? line(.ane, .fp16) : nil

        var out: [Option] = []
        func add(_ best: [Option]) { if let b = best.min(by: { $0.seconds < $1.seconds }) { out.append(b) } }
        func option(_ g: Int, _ a: Int, _ e: Int) -> Option? {
            var rows: [Kernel: Int] = [:]
            if g > 0 { rows[gpu] = g }
            if a > 0, let amx { rows[amx] = a }
            if e > 0, let ane { rows[ane] = e }
            let assignment = ProductAssignment(gpu: g, amx: a, ane: e)
            guard assignment.isRunnable(for: key), let t = seconds(rows, m: m, k: k, n: n) else { return nil }
            // A set's loss was measured on products with no short side: none is offered on one (each engine alone is, by its thin line).
            if rows.count > 1, CapacityModel.isThin(rows: m, k: k, n: n) { return nil }
            return Option(assignment: assignment, rows: rows, seconds: t + accumulation(key, gpu: g, offGPU: a + e))
        }

        if let o = option(m, 0, 0) { out.append(o) }
        if let amx, let seed = model.split(m: m, k: k, n: n, kernels: [gpu, amx], sustained: c.sustained),
           let g0 = seed.rows[gpu], seed.rows[amx] != nil {
            add(gpuRows(rest: m, balance: Double(g0)).compactMap { option($0, m - $0, 0) })
        }
        if let ane, let seed = model.split(m: m, k: k, n: n, kernels: [gpu, ane], sustained: c.sustained), let r0 = seed.rows[ane] {
            add(aneRows(m: m, balance: Double(r0)).compactMap { option(m - $0, 0, $0) })
        }
        if let amx, let ane, let seed = model.split(m: m, k: k, n: n, kernels: [gpu, amx, ane], sustained: c.sustained),
           let r0 = seed.rows[ane], let sideGPU = seed.rows[gpu], let sideAMX = seed.rows[amx] {
            let fraction = Double(sideGPU) / Double(max(1, sideGPU + sideAMX))
            var all: [Option] = []
            for a in aneRows(m: m, balance: Double(r0)) {
                let rest = m - a
                for g in gpuRows(rest: rest, balance: Double(rest) * fraction) {
                    if let o = option(g, rest - g, a) { all.append(o) }
                }
            }
            add(all)
        }
        if amx != nil, let o = option(0, m, 0) { out.append(o) }
        if ane != nil, let o = option(0, 0, m) { out.append(o) }
        if let amx, let ane, let seed = model.split(m: m, k: k, n: n, kernels: [amx, ane], sustained: c.sustained),
           let r0 = seed.rows[ane], seed.rows[amx] != nil {
            add(aneRows(m: m, balance: Double(r0)).compactMap { option(0, m - $0, $0) })
        }
        return out
    }

    /// **A native 8-bit weight**: the GPU alone — and, for an int8 weight whose scales the ANE's program takes (per column or for the tensor, never Q8_0's blocks), the ANE's last rows
    /// beside the GPU's int8 kernel, both reading the integers: `[gpu/int8, ane/int8]`, a set with its own sharing in the model. It
    /// is offered only to the denoiser (P20), from the ANE's floor, where the model has the line `ane/int8` (a route the probes'
    /// judge refused has none) and `NeuralEngine.readsInt8` for the weight's rotation (`isRunnable`).
    ///
    /// **Then the same on the activation rounded to bf16** where operands may be rounded (`allowsRounding`, the denoiser, from the
    /// rounding's floor): the GPU's rounded kernel alone (`gpu/int8act`, `gpu/q8act`) or beside the ANE's integers
    /// (`[gpu/int8act, ane/int8]`) — that pair only where the probes measured its own sharing: the rounded kernel runs on the tensor
    /// units, and the fp32 pair's κ is not its κ. No AMX beside it: the AMX reads the fp32 activation.
    ///
    /// **Then on the activation quantized to int8** (W8A8, `gpu/int8x8`), an int8 weight only, where it may be (`allowsQuantizing`, the
    /// denoiser, from the rounding's floor): the GPU's whole product, alone — no AMX (the fp32 activation), no ANE (its int8 × int8 is
    /// not built: the pair `[gpu/int8act, ane/int8]` stays the integers' only one).
    func native(_ key: ProductKey) -> [Option] {
        let (k, n, m) = (key.k, key.n, key.m)
        let int8 = key.weight == .int8
        let short = CapacityModel.isThin(rows: m, k: k, n: n)     // no set on such a product: its loss was not measured there
        // The ANE's integers need the ANE itself: a map that left its fp16 out (judged wrong) brings no engine up (`Resources.ane`).
        let ane = int8 && c.allowsANE && key.phase == .denoising && m >= c.limits.aneRows && !short && line(.ane, .fp16) != nil
            ? line(.ane, .int8) : nil
        var out: [Option] = []

        /// The GPU's `gpu` kernel beside the ANE's integers, the ANE's rows at every unit, the most first (`pairRows`), then the ANE alone (`alone`).
        func withANE(_ gpu: Kernel, rounds: Bool, alone: Bool) {
            guard let ane else { return }
            func integers(_ rows: Int) -> Option? {
                let assignment = ProductAssignment(path: .native, roundsActivation: rounds && rows < m, gpu: m - rows, ane: rows)
                guard assignment.isRunnable(for: key) else { return nil }
                var split: [Kernel: Int] = [ane: rows]
                if rows < m { split[gpu] = m - rows }
                guard let t = seconds(split, m: m, k: k, n: n) else { return nil }
                return Option(assignment: assignment, rows: split, seconds: t)
            }
            out += pairRows(m: m).reversed().compactMap(integers)
            if alone, let o = integers(m) { out.append(o) }
        }

        // On the fp32 activation.
        if let gpu = line(.gpu, int8 ? .int8 : .q8) {
            let assignment = ProductAssignment(path: .native, gpu: m)
            if assignment.isRunnable(for: key), let t = seconds([gpu: m], m: m, k: k, n: n) {
                out.append(Option(assignment: assignment, rows: [gpu: m], seconds: t))
            }
            withANE(gpu, rounds: false, alone: true)
        }
        // On the activation rounded to bf16.
        if key.phase == .denoising, c.allowsRounding, m >= c.limits.roundingRows, let gpu = line(.gpu, int8 ? .int8act : .q8act) {
            let assignment = ProductAssignment(path: .native, roundsActivation: true, gpu: m)
            if assignment.isRunnable(for: key), let t = seconds([gpu: m], m: m, k: k, n: n) {
                out.append(Option(assignment: assignment, rows: [gpu: m], seconds: t))
            }
            if let ane, model.sharing[CapacityModel.key([gpu, ane])] != nil { withANE(gpu, rounds: true, alone: false) }
        }
        // On the activation quantized to int8.
        if int8, key.phase == .denoising, c.allowsQuantizing, m >= c.limits.roundingRows, let gpu = line(.gpu, .int8x8) {
            let assignment = ProductAssignment(path: .native, quantizesActivation: true, gpu: m)
            if assignment.isRunnable(for: key), let t = seconds([gpu: m], m: m, k: k, n: n) {
                out.append(Option(assignment: assignment, rows: [gpu: m], seconds: t))
            }
        }
        return out
    }

    /// **A K-quant's blocks read as published** (`gpu/q4kact`, `KQuantKernels`): the GPU's kernel on the activation rounded to bf16,
    /// alone — where operands may be rounded, on the denoiser, from the rounding's floor, as the other rounded kernels; never on a product
    /// with a short side (no thin line times it, and a weight of 64 outputs leaves the reserve too little room for its layout). No pair: the
    /// ANE's programs read a widened weight or per-column integers, never K-quant blocks, and the route widens nothing.
    func kquant(_ key: ProductKey) -> [Option] {
        guard key.phase == .denoising, c.allowsRounding, key.m >= c.limits.roundingRows,
              !CapacityModel.isThin(rows: key.m, k: key.k, n: key.n), let gpu = line(.gpu, .q4kact) else { return [] }
        let assignment = ProductAssignment.gpuAlone(key.m, path: .native, rounds: true)
        guard assignment.isRunnable(for: key), let t = seconds([gpu: key.m], m: key.m, k: key.k, n: key.n) else { return [] }
        return [Option(assignment: assignment, rows: [gpu: key.m], seconds: t)]
    }

    /// **The rounded bf16 weight shared with the ANE**: the GPU's rounded rows (`gpu/bf16act`) beside the ANE's fp16 program on the same
    /// bf16 copy (`[gpu/bf16act, ane/fp16]`), the ANE's rows at every unit, the most first (`pairRows`) — on the denoiser, from the ANE's floor, on a product
    /// with no short side, and only where the probes measured that pair's own sharing: the rounded kernel runs on the tensor units at
    /// another rate and draws another power than `gpu/fp32`, whose κ the ×1.52 of the members' lines once borrowed. No AMX beside it:
    /// the AMX reads a widened weight and the fp32 activation, and the reserve holds the bf16 copy and its rounded rows.
    func roundedShares(_ key: ProductKey, gpu: Kernel) -> [Option] {
        let (k, n, m) = (key.k, key.n, key.m)
        guard c.allowsANE, key.phase == .denoising, m >= c.limits.aneRows, !CapacityModel.isThin(rows: m, k: k, n: n),
              let ane = line(.ane, .fp16), model.sharing[CapacityModel.key([gpu, ane])] != nil,
              model.split(m: m, k: k, n: n, kernels: [gpu, ane], sustained: c.sustained) != nil else { return [] }
        let options = pairRows(m: m).compactMap { rows -> Option? in
            let assignment = ProductAssignment(path: .bf16, roundsActivation: true, gpu: m - rows, ane: rows)
            let split: [Kernel: Int] = [gpu: m - rows, ane: rows]
            guard assignment.isRunnable(for: key), let t = seconds(split, m: m, k: k, n: n) else { return nil }
            return Option(assignment: assignment, rows: split, seconds: t)
        }
        return options.reversed()
    }

    // MARK: Preparation

    func preparation(format: CapacityClasses.WeightFormat, path: WeightPath, k: Int, n: Int) -> Double {
        let heat = c.sustained ? max(1, model.heat[.cpu] ?? 1) : 1
        let perValue: Double
        // A LoRA's operands are the stack's fp32, expanded and cached beforehand (`CacheLoRA`): nothing to prepare at the call.
        if format.isLoRA { return 0 }
        if let kind = format.kind { perValue = model.preparation[kind]?[path] ?? 0 }
        // A K-quant's blocks copied as they are (the GPU unpacks them, inside its line), or widened on the CPU.
        else if path == .native { perValue = model.kquantStaging ?? model.preparation[.q8]?[.native] ?? 0 }
        else { perValue = model.kquantPreparation ?? model.preparation[.bf16]?[.widened] ?? 0 }
        return Double(k) * Double(n) * perValue * heat
    }

    // MARK: Products

    /// **Every way the solver may compute `key`**, in the order a tie is settled (simplest first): the widened sets, the bf16 weight as
    /// published, its activation rounded, the native 8-bit weight's. What `solve` chooses among, and what the coverage guard reads.
    func candidates(_ key: ProductKey) -> [Option] {
        guard line(.gpu, .fp32) != nil else { return [] }
        let (k, n, m) = (key.k, key.n, key.m)
        var options = widened(key)
        if key.weight == .bf16 {
            if let gpu = line(.gpu, .bf16), let t = seconds([gpu: m], m: m, k: k, n: n) {
                options.append(Option(assignment: .gpuAlone(m, path: .bf16), rows: [gpu: m], seconds: t))
            }
            // Rounding the activation to bf16 trades the operand's last bits for time, within P20 (`allowsRounding`); taken where this
            // Mac's line says it is faster, and only there — on a GPU without bf16 in hardware the fp32 operand stays.
            if key.phase == .denoising, c.allowsRounding, m >= c.limits.roundingRows, let gpu = line(.gpu, .bf16act) {
                if let t = seconds([gpu: m], m: m, k: k, n: n) {
                    options.append(Option(assignment: .gpuAlone(m, path: .bf16, rounds: true), rows: [gpu: m], seconds: t))
                }
                options += roundedShares(key, gpu: gpu)
            }
        }
        if key.weight == .int8 || key.weight == .q8 { options += native(key) }
        if key.weight == .q4k { options += kquant(key) }
        return options
    }

    func solution(_ o: Option, _ key: ProductKey) -> CapacitySolver.ProductSolution {
        let prep = preparation(format: key.weight, path: o.assignment.path, k: key.k, n: key.n)
        return .init(assignment: o.assignment, route: o.assignment.route(weight: key.weight), rows: o.rows, preparation: prep,
                     compute: o.seconds, busy: busy(o.rows, m: key.m, k: key.k, n: key.n))
    }

    func solve(_ key: ProductKey) -> CapacitySolver.ProductSolution? {
        guard line(.gpu, .fp32) != nil else { return nil }
        let timed = passingTheirError(arbitrated(canonical(candidates(key)).map { o in
            (o, o.seconds + preparation(format: key.weight, path: o.assignment.path, k: key.k, n: key.n))
        }))
        guard let fastest = timed.min(by: { $0.1 < $1.1 }) else { return nil }
        let thin = CapacityModel.isThin(rows: key.m, k: key.k, n: key.n)
        // Within the margin the written order decides; a candidate with the ANE is also taken anywhere within the error of its
        // difference with the fastest (the ANE on a tie: decided 3/11/2026), the first in that order — the most rows first.
        return timed.first { o in
            o.1 <= fastest.1 * (1 + margin(thin: thin && !oneFixedCost(o.0, fastest.0)))
                || (o.0.uses(.ane) && o.1 - fastest.1 <= 2 * differenceError(o, fastest))
        }.map { solution($0.0, key) }
    }

    /// **Whether two candidates pay one fixed cost**: both on the GPU alone, whose thin lines share one `t₀` (`CapacityFit.thinLines`). The
    /// thin margin is the width of the fixed costs two maps read apart (`MatmulTile.thinMargin`); where both candidates pay the same one, it
    /// is no part of their difference, and what is left — what each streams and computes — is read to the wide margin. Two GPU forms of a
    /// text encoder's 19-row product, 14.6 % apart in one map and 15.5 % in the next, made two plans under the thin margin's 15 %.
    func oneFixedCost(_ a: Option, _ b: Option) -> Bool {
        let engines = Set(kernels(a).map(\.engine)).union(kernels(b).map(\.engine))
        guard engines == [.gpu], kernels(a).count == 1, kernels(b).count == 1, let ka = kernels(a).first, let kb = kernels(b).first,
              let ta = model.thin[ka]?.t0, let tb = model.thin[kb]?.t0 else { return false }
        return ta == tb
    }

    /// The standard error of a figure's seconds, relative: a set's own where its chains were measured three times (`Sharing.noise`),
    /// the map's σ for any other figure.
    func relativeError(_ ks: [Kernel]) -> Double {
        let sigma = CapacityFit.sigma(mapNoise: model.noise) ?? 0
        return ks.count >= 2 ? (model.sharing(ks).noise ?? sigma) : sigma
    }

    /// √(e_a² t_a² + e_b² t_b²): the standard error of the difference of two candidates' seconds.
    func differenceError(_ a: (Option, Double), _ b: (Option, Double)) -> Double {
        let (ea, eb) = (relativeError(kernels(a.0)) * a.1, relativeError(kernels(b.0)) * b.1)
        return (ea * ea + eb * eb).squareRoot()
    }

    /// The kernels `o` gives rows to, as a set's key reads them.
    func kernels(_ o: Option) -> [Kernel] { o.rows.filter { $0.value > 0 }.map(\.key) }

    /// **A set is taken only where what a render pays on it beats the simpler assignment by more than their figures' error.** A render
    /// pays a set's mean seconds, call after call; the map knows that mean to a standard error: the set's own where its chains were
    /// measured three times (`Sharing.noise`, the median chain's, from the MAD — `CapacityFit.chains`), the map's σ for any other figure
    /// (`CapacityFit.sigma(mapNoise:)`, a single figure read once). The simpler assignment is the best candidate on a strict subset of the
    /// set's kernels (the GPU alone against GPU + ANE). Each figure t ± e·t; their difference has the standard error
    /// √(e_S² t_S² + e_alt² t_alt²), and a gain under twice it is one the next map reverses more than one time in forty-four (a normal's tail past 2σ): the set's
    /// candidates are then dropped. Not twice a noise per chain (an earlier form: the integers' rounded pair dropped from
    /// every Compact under one map of two, +10 to +15 % predicted at 1 024², its noise the gap of two chains, one of them disturbed).
    ///
    /// **A set with the ANE is the other way round** (the user's decision, 3/11/2026: the ANE on a tie — the same speed for fewer
    /// watts): it is dropped only where it is proven slower than the simpler assignment, by more than twice the same error; within
    /// it, it stays, and `solve` takes it within the error of the fastest, its most rows first. Both sides are as stable from one map
    /// to the next: the boundary is the same error, read on the other side.
    func passingTheirError(_ timed: [(Option, Double)]) -> [(Option, Double)] {
        var sets: [String: [Kernel]] = [:]
        for (o, _) in timed { let ks = kernels(o); if ks.count >= 2 { sets[CapacityModel.key(ks)] = ks } }
        var out = timed
        for (set, ks) in sets {
            let mine = Set(ks)
            guard let best = timed.filter({ CapacityModel.key(kernels($0.0)) == set }).min(by: { $0.1 < $1.1 }),
                  let simpler = timed.filter({ (o: (Option, Double)) -> Bool in let ks = Set(kernels(o.0)); return !ks.isEmpty && ks.isStrictSubset(of: mine) }).min(by: { $0.1 < $1.1 })
            else { continue }
            let bound = 2 * differenceError(best, simpler)
            let dropped = ks.contains { $0.engine == .ane } ? best.1 - simpler.1 > bound : simpler.1 - best.1 <= bound
            guard dropped else { continue }
            out.removeAll { CapacityModel.key(kernels($0.0)) == set }
        }
        return out
    }

    /// **The ANE against the AMX** (`CapacitySolver.aneOverAMX`): when the best candidate with the ANE and without the AMX is within
    /// 15 % of the best with the AMX, those with the AMX are dropped; otherwise nothing is.
    func arbitrated(_ timed: [(Option, Double)]) -> [(Option, Double)] {
        guard let ane = timed.filter({ $0.0.uses(.ane) && !$0.0.uses(.amx) }).map(\.1).min(),
              let amx = timed.filter({ $0.0.uses(.amx) }).map(\.1).min(),
              ane <= amx * (1 + CapacitySolver.aneOverAMX) else { return timed }
        return timed.filter { !$0.0.uses(.amx) }
    }

    /// **The width under which two candidates are one** (relative): twice the map's noise (`CapacityModel.noise`), never under the
    /// noise between two calm maps of one Mac — the tiles' own floors, `MatmulTile.margin` for a wide product, `thinMargin` for a thin
    /// one, whose fixed costs two maps read 10 % apart either way —; a part in a billion for a map without rounds.
    func margin(thin: Bool = false) -> Double {
        model.noise.map { max(thin ? MatmulTile.thinMargin : MatmulTile.margin, 2 * $0) } ?? 1e-9
    }

    /// **The written order a tie under the noise is settled by**: without the AMX first, then with the ANE, then fewer engines; then the
    /// finer operand (`rank`); then the order `candidates` lists them in (the sort is stable).
    func canonical(_ options: [Option]) -> [Option] {
        func rank(_ a: ProductAssignment) -> Int {
            switch a.path {
            case .widened: return 0
            case .bf16: return a.roundsActivation ? 2 : 1
            case .native: return a.quantizesActivation ? 5 : a.roundsActivation ? 4 : 3
            }
        }
        return options.enumerated().sorted { x, y in
            let (a, b) = (x.element, y.element)
            if a.uses(.amx) != b.uses(.amx) { return !a.uses(.amx) }
            if a.uses(.ane) != b.uses(.ane) { return a.uses(.ane) }
            let (ea, eb) = (a.rows.filter { $0.value > 0 }.count, b.rows.filter { $0.value > 0 }.count)
            if ea != eb { return ea < eb }
            if rank(a.assignment) != rank(b.assignment) { return rank(a.assignment) < rank(b.assignment) }
            return x.offset < y.offset
        }.map(\.element)
    }

    /// **An assignment decided elsewhere, timed here**: the kernels it reads, the rows each takes, plus the weight's preparation.
    func evaluate(_ a: ProductAssignment, _ key: ProductKey) -> CapacitySolver.ProductSolution? {
        let (k, n, m) = (key.k, key.n, key.m)
        var rows: [Kernel: Int] = [:]
        switch a.path {
        case .widened:
            if a.gpu > 0 { guard let g = line(.gpu, .fp32) else { return nil }; rows[g] = a.gpu }
            if a.amx > 0 { guard let x = line(.amx, .fp32) else { return nil }; rows[x] = a.amx }
            if a.ane > 0 { guard let e = line(.ane, .fp16) else { return nil }; rows[e] = a.ane }
        case .bf16:
            guard let g = line(.gpu, a.roundsActivation ? .bf16act : .bf16) else { return nil }
            if a.gpu > 0 { rows[g] = a.gpu }
            // The ANE's last rows beside the GPU's rounded ones, its fp16 program on the bf16 copy: the kernels `roundedShares` times.
            if a.ane > 0 {
                guard let e = line(.ane, .fp16) else { return nil }
                rows[e] = a.ane
            }
        case .native:
            let int8 = key.weight == .int8
            if key.weight == .q4k {
                guard a.isRunnable(for: key), let g = line(.gpu, .q4kact) else { return nil }
                rows[g] = a.gpu
                break
            }
            if a.quantizesActivation {
                guard a.isRunnable(for: key), let g = line(.gpu, .int8x8) else { return nil }
                rows[g] = a.gpu
                break
            }
            guard let g = line(.gpu, a.roundsActivation ? (int8 ? .int8act : .q8act) : (int8 ? .int8 : .q8)) else { return nil }
            // The AMX's columns are no candidate (`native`): no line times them.
            guard a.amxColumns == 0 else { return nil }
            // The ANE's last rows beside the GPU's, both reading the integers: the same kernels `native` times its own candidates by.
            if a.gpu > 0 { rows[g] = a.gpu }
            if a.ane > 0 {
                guard key.weight == .int8, let e = line(.ane, .int8) else { return nil }
                rows[e] = a.ane
            }
        }
        guard !rows.isEmpty, let t = seconds(rows, m: m, k: k, n: n) else { return nil }
        let prep = preparation(format: key.weight, path: a.path, k: k, n: n)
        return .init(assignment: a, route: a.route(weight: key.weight), rows: rows, preparation: prep,
                     compute: t + accumulation(key, gpu: a.gpu, offGPU: a.amx + a.ane),
                     busy: busy(rows, m: m, k: k, n: n))
    }

    func evaluate(_ a: AttentionAssignment, _ key: AttentionKey) -> CapacitySolver.AttentionSolution? {
        guard a.isRunnable(for: key), let t = timed(key, a) else { return nil }
        return .init(assignment: a, route: t.route, rows: t.rows, compute: t.seconds, busy: t.busy)
    }

    // MARK: Attentions

    /// The time and rows of an attention under a kernel and a number of CPU heads (flash: the GPU's, whole).
    /// `pass: false`: without the QK-norm and RoPE's own seconds (the kernel's choice, which the pass does not change; the wake stays).
    func timed(_ key: AttentionKey, _ a: AttentionAssignment, pass withPass: Bool = true) -> (route: String, rows: [Kernel: Int], seconds: Double, busy: [Engine: Double])? {
        let per = 2 * key.sequence * key.keys * key.headDim            // 2·k with n = 1: 4·S·keys·dim per head
        var rows: [Kernel: Int]
        switch a.kernel {
        case .flash:
            guard let f = line(.gpu, .flash) else { return nil }
            rows = [f: key.heads]
        case .graph:
            guard let g = line(.gpu, .graph) else { return nil }
            rows = [g: key.heads]
        case .tiles, .tiles16, .tilesbf16:
            guard let t = gpuLine(a.kernel) else { return nil }
            rows = [t: key.heads]
        case .split:
            // The GPU's heads run the kernel the assignment names (the best of those allowed, `solve`); the pair's sharing was measured
            // beside the graph and is taken for every GPU kernel (the same two engines).
            guard let g = gpuLine(a.gpuKernel), let cpu = line(.cpu, .heads), model.measures([.gpu, .cpu]) else { return nil }
            rows = [g: key.heads - a.cpuHeads, cpu: a.cpuHeads]
        }
        guard var t = model.seconds(m: key.heads, k: per, n: 1, rows: rows, sustained: c.sustained) else { return nil }
        let ends = model.busy(m: key.heads, k: per, n: 1, rows: rows, sustained: c.sustained)
        var b = Dictionary(ends.map { ($0.key.engine, $0.value) }, uniquingKeysWith: +)
        // A self-attention follows the CPU's QK-norm and RoPE, the GPU idle: its call pays the GPU's wake (`CapacityModel.wake`), whatever
        // the kernel. A cross-attention's keys are the text's. **The GPU's side alone is late**: under a cut by heads the CPU's heads start
        // as the call does, so the wake moves the GPU's end, and the call's only where the GPU's side is the last (the overlap's own end
        // plus the wake, against the CPU's; the set's regime and overhead as `CapacityModel.seconds` applies them). Alone on the GPU, the
        // same as before: the time plus the wake.
        // The pass on the GPU (`normRopeOnGPU`) leaves it no idle: no wake.
        let wake = key.cross || a.normRopeOnGPU ? 0 : model.wake ?? 0
        if wake > 0 {
            b[.gpu, default: 0] += wake
            let running = rows.filter { $0.value > 0 }.map(\.key)
            if running.count > 1, let last = ends.map({ $0.value + ($0.key.engine == .gpu ? wake : 0) }).max() {
                let set = model.sharing(running)
                t = last * (c.sustained ? max(1, set.sustained ?? 1) : 1) + set.overhead
            } else { t += wake }
        }
        // The pass itself, where the map measured it (`CapacityModel.normRope`): the CPU's wall, or the GPU's kernel — chained to the
        // attention, its own seconds; waited for under a cut by heads, its wall. A map without it: the CPU's, priced at nothing as before;
        // the GPU's, not timed.
        var pass = 0.0
        if !key.cross, withPass || a.normRopeOnGPU {
            if let p = normRope(key, onGPU: a.normRopeOnGPU, waited: a.splits) {
                pass = p.seconds
                b[p.engine, default: 0] += p.busy
            } else if a.normRopeOnGPU { return nil }
        }
        return (a.kernel.rawValue, rows, t + pass, b)
    }

    // MARK: The QK-norm and RoPE before a self-attention

    /// The values the pass runs over: Q's `S · H · Dh` and K's `S · H_kv · Dh`.
    func normRopeValues(_ key: AttentionKey) -> Double { Double(key.sequence) * Double(key.heads + key.kvHeads) * Double(key.headDim) }

    /// The pass's seconds on a route, the engine it holds and for how long; `nil` without the map's lines (or a head dimension the
    /// kernel does not take, on the GPU).
    func normRope(_ key: AttentionKey, onGPU: Bool, waited: Bool) -> (seconds: Double, engine: Engine, busy: Double)? {
        guard let n = model.normRope else { return nil }
        let v = normRopeValues(key)
        guard onGPU else { let t = n.cpu.seconds(v); return (t, .cpu, t) }
        guard QKNormRope.runs(headDim: key.headDim) else { return nil }
        return (waited ? n.gpuWaited.seconds(v) : n.gpu.seconds(v), .gpu, n.gpu.seconds(v))
    }

    /// **The pass on the GPU where it beats the CPU's past the noise** (the attentions' rule: the first in the written order within the
    /// margin of the fastest — the CPU's pass, what every plan ran, before): the whole call weighed, the CPU's pass with the wake its idle
    /// costs, the GPU's without. The kernel chosen stays the one chosen without it: the pass changes no kernel.
    func withNormRope(_ key: AttentionKey, _ a: AttentionAssignment,
                      _ t: (route: String, rows: [Kernel: Int], seconds: Double, busy: [Engine: Double]))
        -> (AttentionAssignment, (route: String, rows: [Kernel: Int], seconds: Double, busy: [Engine: Double])) {
        guard !key.cross, model.normRope != nil else { return (a, t) }
        let gpu = a.normRope(onGPU: true)
        guard let g = timed(key, gpu), t.seconds > g.seconds * (1 + margin()) else { return (a, t) }
        return (gpu, g)
    }

    /// The line of a kernel the GPU runs (all the heads, or its share under a split); the fp16 one only where the request allows half
    /// precision, the bf16 one where it allows the rounding.
    func gpuLine(_ k: AttentionAssignment.Kernel) -> Kernel? {
        switch k {
        case .graph: return line(.gpu, .graph)
        case .flash: return line(.gpu, .flash)
        case .tiles: return line(.gpu, .tiles)
        case .tiles16: return c.allowsHalf ? line(.gpu, .tiles16) : nil
        case .tilesbf16:
            // Where fp16 is allowed too, bf16 (three fewer bits) must beat it by more than the noise (`MatmulTile.margin`): a tie at
            // the noise is the finer mantissa's.
            guard c.allowsRounding, let b = line(.gpu, .tilesbf16) else { return nil }
            if c.allowsHalf, let h = line(.gpu, .tiles16), let hr = model.lines[h]?.rate, let br = model.lines[b]?.rate,
               br <= hr * (1 + MatmulTile.margin) { return nil }
            return b
        case .split: return nil
        }
    }

    /// **Every way the solver may run `key`**, runnable and timed, in the order a tie is settled: the graph, flash, the tiled kernels,
    /// then the cut with each GPU kernel. What `solve` chooses among, and what the coverage guard reads.
    func candidates(_ key: AttentionKey, flash: Bool, pass: Bool = true) -> [(AttentionAssignment, (route: String, rows: [Kernel: Int], seconds: Double, busy: [Engine: Double]))] {
        var options: [AttentionAssignment] = [.graph]
        if flash, line(.gpu, .flash) != nil { options.append(.flash) }
        // The tiled kernels: a candidate wherever the map has its line (fp16 only where the request allows half precision, bf16 where it
        // allows the rounding, and beats fp16 where both are: `gpuLine`).
        for k in [AttentionAssignment.Kernel.tiles, .tiles16, .tilesbf16] where gpuLine(k) != nil { options.append(AttentionAssignment(kernel: k)) }
        let groups = key.kvHeads
        if key.sequence >= HeadSplitAttention.minimumSequence, groups >= 2, line(.cpu, .heads) != nil, model.measures([.gpu, .cpu]) {
            // The GPU's heads with each kernel the engine can give them: the cut composes with the best of them.
            var gpuKernels: [AttentionAssignment.Kernel] = [.graph, .tiles, .tiles16, .tilesbf16]
            if flash { gpuKernels.insert(.flash, at: 1) }
            for gpu in gpuKernels where gpuLine(gpu) != nil {
                for cpu in 1..<groups { options.append(AttentionAssignment(kernel: .split, cpuHeads: cpu * key.group, gpuKernel: gpu)) }
            }
        }
        return options.compactMap { o in o.isRunnable(for: key) ? timed(key, o, pass: pass).map { (o, $0) } : nil }
    }

    func solve(_ key: AttentionKey, flash: Bool) -> CapacitySolver.AttentionSolution? {
        // The written order (`candidates`: the graph, flash, the tiled kernels from the finest operand, then the cut with the fewest CPU heads
        // first), the first within the noise of the fastest.
        let all = candidates(key, flash: flash, pass: false)
        guard let fastest = all.map(\.1.seconds).min(), var (a, t) = all.first(where: { $0.1.seconds <= fastest * (1 + margin()) }) else { return nil }
        if model.normRope != nil, let whole = timed(key, a) { t = whole }
        // The form its map chose for the tiled kernel that runs: what its line timed.
        let tiled = a.splits ? a.gpuKernel : a.kernel
        if tiled == .tiles16 || tiled == .tilesbf16 { a.tile = model.attentionTiles[Kernel(.gpu, tiled == .tiles16 ? .tiles16 : .tilesbf16)] }
        (a, t) = withNormRope(key, a, t)
        return .init(assignment: a, route: t.route, rows: t.rows, compute: t.seconds, busy: t.busy)
    }
}
