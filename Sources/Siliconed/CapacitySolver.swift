import Foundation

/// **The solver: from a machine's model and one class of computation to the exact assignment that runs it soonest, and the time
/// it predicts.** Pure: no Metal, no clock, no weights. It *consumes* `CapacityModel` (lines, sharing, the ANE's staircase, heat,
/// preparation) and writes what the engine reads — `ProductAssignment`, `AttentionAssignment` (`CapacityPlan`) — one class at a
/// time, the integers the engines will run, never a ratio for the engine to round.
///
///     product (k, n, m, weight format, phase)
///         │   each path    widened (any set of gpu · amx · ane) · bf16 · bf16 rounded · native int8 / q8 (gpu · gpu + amx columns)
///         │   each set     the balanced split on the lines (`CapacityModel.split`), then the integers around it on the lattice
///         ▼                the engines can run: the GPU's rows on MPS's 64-row tile, the ANE's in units of 256, native columns on 64
///     t = preparation(path) + T(rows)               one weight is prepared per call, at every height
///         ▼
///     the least t, the simplest set on a tie        ──▶  ProductAssignment
///     attention (S, keys, heads, kv heads, dim, cross) ──▶ graph · flash (where the caller tries it) · split (whole kv groups on the CPU)
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
///   - **What the caller's code cannot do is not offered.** An encoder never has the ANE or a rounded activation (P20: only the
///     denoiser's fp16 rows and bf16 operands, at the variants that publish them so); a process that did not bring the ANE up (the
///     Standard) is solved without it (`Constraints.allowsANE`); flash is the caller's (`flash:`), not a family's guess; the split
///     cuts by whole kv groups, the GPU keeping one at least.
///   - **The model's domain is a floor, not a bound of the engine.** Its lines were measured from ≈ 4 GFLOP: below `Limits` the AMX,
///     the ANE and the rounding are not asked about (their t₀ would be extrapolated), and the GPU takes the product. The engine
///     itself imposes none of them.
///   - **Neutral on tie**: candidates come simplest first (the GPU, then with the AMX, with the ANE, the three, then the AMX alone,
///     the ANE alone), and a more complex one must be faster by more than a part in a billion.
///
/// The second time (E5, the overlap between blocks) enters through `Combine`: the solver gives each block its preparation,
/// its compute and what each engine is busy for, and `combine` turns the blocks into a total. `sequential` is the sum.
package enum CapacitySolver {
    typealias Kernel = CapacityModel.Kernel
    typealias Engine = CapacityModel.Engine

    // MARK: - What goes in

    /// The rows from which an engine or a form is worth asking: the model's lines start at ≈ 4 GFLOP.
    package struct Limits: Sendable, Equatable {
        /// The AMX's floor, the previous map's `amx_min`: what its predictions and its render check still read by default.
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
        /// The ANE may be asked for: this process brings it up (never the Standard's, fp32 products).
        package var allowsANE: Bool

        package init(limits: Limits = .init(), sustained: Bool = true, excluded: Set<CapacityModel.Kernel> = [],
                     allowsANE: Bool = true) {
            self.limits = limits; self.sustained = sustained; self.excluded = excluded; self.allowsANE = allowsANE
        }

        /// The constraints of a request: the ANE for the versions that publish a non-bf16 weight.
        package init(request: CapacityClasses.Request, limits: Limits = .init(), sustained: Bool = true,
                     excluded: Set<CapacityModel.Kernel> = []) {
            self.init(limits: limits, sustained: sustained, excluded: excluded, allowsANE: request.variant != .standard)
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

    func seconds(_ rows: [Kernel: Int], m: Int, k: Int, n: Int, swapped: Bool = false) -> Double? {
        // A set of engines the probes never measured is no candidate: keep 1 is not a measure.
        guard model.measures(rows.filter { $0.value > 0 }.map(\.key.engine)) else { return nil }
        return swapped ? model.seconds(m: n, k: k, n: m, rows: rows, sustained: c.sustained)
                : model.seconds(m: m, k: k, n: n, rows: rows, sustained: c.sustained)
    }
    func busy(_ rows: [Kernel: Int], m: Int, k: Int, n: Int, swapped: Bool = false) -> [Engine: Double] {
        let b = swapped ? model.busy(m: n, k: k, n: m, rows: rows, sustained: c.sustained)
                        : model.busy(m: m, k: k, n: n, rows: rows, sustained: c.sustained)
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

    /// The ANE's rows of `m` (units of 256, the GPU or the AMX keeping a row at least) around the balance `r0`.
    func aneRows(m: Int, balance r0: Double) -> [Int] {
        let unit = NeuralEngine.rowUnit
        let u0 = r0 / Double(unit)
        var units = Set<Int>()
        for u in (Int(u0.rounded(.down)) - 1)...(Int(u0.rounded(.up)) + 1) where u >= 1 && u * unit < m { units.insert(u) }
        return units.sorted().map { $0 * unit }
    }

    /// The columns the AMX takes beside a native weight (the GPU's, multiples of 64, around the balance `g0`), as
    /// `WeightProducts.amxColumns` lays them.
    func gpuColumns(n: Int, balance g0: Double) -> [Int] {
        let top = (n - 64) / 64 * 64
        guard n >= 128, top >= 64 else { return [] }
        let base = Int((g0 / 64).rounded())
        var seen = Set<Int>(), out: [Int] = []
        for d in -2...2 {
            let g = min(top, max(64, (base + d) * 64))
            if seen.insert(g).inserted { out.append(g) }
        }
        return out
    }

    // MARK: A way to compute one product

    struct Option {
        var assignment: ProductAssignment
        var rows: [Kernel: Int]
        /// The split is along the columns (an AMX beside a native weight): the dimensions of the model's product are swapped.
        var swapped = false
        var seconds: Double
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
            return Option(assignment: assignment, rows: rows, seconds: t)
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

    /// **A native 8-bit weight**: the GPU alone, or beside the AMX on the last columns (the GPU's a multiple of 64) — and, for an
    /// int8 weight whose scales the ANE's program takes (per column or for the tensor, never Q8_0's blocks), the ANE's last rows
    /// beside the GPU's int8 kernel, both reading the integers: `[gpu/int8, ane/int8]`, a set with its own sharing in the model. It
    /// is offered only to the denoiser (P20), from the ANE's floor, where the model has the line `ane/int8` (a route the probes'
    /// judge refused has none) and `NeuralEngine.readsInt8` for the weight's rotation (`isRunnable`).
    func native(_ key: ProductKey) -> [Option] {
        let (k, n, m) = (key.k, key.n, key.m)
        let form: CapacityModel.Form = key.weight == .int8 ? .int8 : .q8
        guard let gpu = line(.gpu, form) else { return [] }
        var out: [Option] = []
        func option(_ columns: Int) -> Option? {
            let assignment = ProductAssignment(path: .native, gpu: m, amxColumns: columns)
            guard assignment.isRunnable(for: key) else { return nil }
            if columns == 0 {
                guard let t = seconds([gpu: m], m: m, k: k, n: n) else { return nil }
                return Option(assignment: assignment, rows: [gpu: m], seconds: t)
            }
            guard let amx = line(.amx, form), let t = seconds([gpu: n - columns, amx: columns], m: m, k: k, n: n, swapped: true) else { return nil }
            return Option(assignment: assignment, rows: [gpu: n - columns, amx: columns], swapped: true, seconds: t)
        }
        if let o = option(0) { out.append(o) }
        if m >= c.limits.amxRows, let amx = line(.amx, form),
           let seed = model.split(m: n, k: k, n: m, kernels: [gpu, amx], sustained: c.sustained), let g0 = seed.rows[gpu] {
            let cands = gpuColumns(n: n, balance: Double(g0)).compactMap { option(n - $0) }
            if let b = cands.min(by: { $0.seconds < $1.seconds }) { out.append(b) }
        }
        if key.weight == .int8, c.allowsANE, key.phase == .denoising, m >= c.limits.aneRows, let ane = line(.ane, .int8) {
            func integers(_ rows: Int) -> Option? {
                let assignment = ProductAssignment(path: .native, gpu: m - rows, ane: rows)
                guard assignment.isRunnable(for: key) else { return nil }
                var split: [Kernel: Int] = [ane: rows]
                if rows < m { split[gpu] = m - rows }
                guard let t = seconds(split, m: m, k: k, n: n) else { return nil }
                return Option(assignment: assignment, rows: split, seconds: t)
            }
            if let seed = model.split(m: m, k: k, n: n, kernels: [gpu, ane], sustained: c.sustained), let r0 = seed.rows[ane],
               let b = aneRows(m: m, balance: Double(r0)).compactMap(integers).min(by: { $0.seconds < $1.seconds }) {
                out.append(b)
            }
            if let o = integers(m) { out.append(o) }
        }
        return out
    }

    // MARK: Preparation

    func preparation(format: CapacityClasses.WeightFormat, path: WeightPath, k: Int, n: Int) -> Double {
        let heat = c.sustained ? max(1, model.heat[.cpu] ?? 1) : 1
        let perValue: Double
        if let kind = format.kind { perValue = model.preparation[kind]?[path] ?? 0 }
        else { perValue = model.kquantPreparation ?? model.preparation[.bf16]?[.widened] ?? 0 }
        return Double(k) * Double(n) * perValue * heat
    }

    // MARK: Products

    func solve(_ key: ProductKey) -> CapacitySolver.ProductSolution? {
        guard line(.gpu, .fp32) != nil else { return nil }
        let (k, n, m) = (key.k, key.n, key.m)
        var options = widened(key)
        if key.weight == .bf16 {
            if let gpu = line(.gpu, .bf16), let t = seconds([gpu: m], m: m, k: k, n: n) {
                options.append(Option(assignment: .gpuAlone(m, path: .bf16), rows: [gpu: m], seconds: t))
            }
            if key.phase == .denoising, m >= c.limits.roundingRows, let gpu = line(.gpu, .bf16act), let t = seconds([gpu: m], m: m, k: k, n: n) {
                options.append(Option(assignment: .gpuAlone(m, path: .bf16, rounds: true), rows: [gpu: m], seconds: t))
            }
        }
        if key.weight == .int8 || key.weight == .q8 { options += native(key) }

        var best: (Option, Double, Double)?       // option, preparation, total
        for o in options {
            let prep = preparation(format: key.weight, path: o.assignment.path, k: k, n: n)
            let total = o.seconds + prep
            if best == nil || total < best!.2 * (1 - 1e-9) { best = (o, prep, total) }
        }
        guard let (o, prep, _) = best else { return nil }
        return .init(assignment: o.assignment, route: o.assignment.route(weight: key.weight), rows: o.rows, preparation: prep,
                     compute: o.seconds, busy: busy(o.rows, m: m, k: k, n: n, swapped: o.swapped))
    }

    /// **An assignment decided elsewhere, timed here**: the kernels it reads, the rows each takes, plus the weight's preparation.
    func evaluate(_ a: ProductAssignment, _ key: ProductKey) -> CapacitySolver.ProductSolution? {
        let (k, n, m) = (key.k, key.n, key.m)
        var rows: [Kernel: Int] = [:]
        var swapped = false
        switch a.path {
        case .widened:
            if a.gpu > 0 { guard let g = line(.gpu, .fp32) else { return nil }; rows[g] = a.gpu }
            if a.amx > 0 { guard let x = line(.amx, .fp32) else { return nil }; rows[x] = a.amx }
            if a.ane > 0 { guard let e = line(.ane, .fp16) else { return nil }; rows[e] = a.ane }
        case .bf16:
            guard let g = line(.gpu, a.roundsActivation ? .bf16act : .bf16) else { return nil }
            rows[g] = a.gpu
        case .native:
            let form: CapacityModel.Form = key.weight == .int8 ? .int8 : .q8
            guard let g = line(.gpu, form) else { return nil }
            if a.amxColumns > 0 {
                guard let x = line(.amx, form) else { return nil }
                rows = [g: n - a.amxColumns, x: a.amxColumns]; swapped = true
            } else {
                // The ANE's last rows beside the GPU's, both reading the integers: the same kernels `native` times its own candidates by.
                if a.gpu > 0 { rows[g] = a.gpu }
                if a.ane > 0 {
                    guard key.weight == .int8, let e = line(.ane, .int8) else { return nil }
                    rows[e] = a.ane
                }
            }
        }
        guard !rows.isEmpty, let t = seconds(rows, m: m, k: k, n: n, swapped: swapped) else { return nil }
        let prep = preparation(format: key.weight, path: a.path, k: k, n: n)
        return .init(assignment: a, route: a.route(weight: key.weight), rows: rows, preparation: prep, compute: t,
                     busy: busy(rows, m: m, k: k, n: n, swapped: swapped))
    }

    func evaluate(_ a: AttentionAssignment, _ key: AttentionKey) -> CapacitySolver.AttentionSolution? {
        guard a.isRunnable(for: key), let t = timed(key, a) else { return nil }
        return .init(assignment: a, route: t.route, rows: t.rows, compute: t.seconds, busy: t.busy)
    }

    // MARK: Attentions

    /// The time and rows of an attention under a kernel and a number of CPU heads (flash: the GPU's, whole).
    func timed(_ key: AttentionKey, _ a: AttentionAssignment) -> (route: String, rows: [Kernel: Int], seconds: Double, busy: [Engine: Double])? {
        let per = 2 * key.sequence * key.keys * key.headDim            // 2·k with n = 1: 4·S·keys·dim per head
        var rows: [Kernel: Int]
        switch a.kernel {
        case .flash:
            guard let f = line(.gpu, .flash) else { return nil }
            rows = [f: key.heads]
        case .graph:
            guard let g = line(.gpu, .graph) else { return nil }
            rows = [g: key.heads]
        case .split:
            guard let g = line(.gpu, .graph), let cpu = line(.cpu, .heads), model.measures([.gpu, .cpu]) else { return nil }
            rows = [g: key.heads - a.cpuHeads, cpu: a.cpuHeads]
        }
        guard let t = model.seconds(m: key.heads, k: per, n: 1, rows: rows, sustained: c.sustained) else { return nil }
        let b = model.busy(m: key.heads, k: per, n: 1, rows: rows, sustained: c.sustained)
        return (a.kernel.rawValue, rows, t, Dictionary(b.map { ($0.key.engine, $0.value) }, uniquingKeysWith: +))
    }

    func solve(_ key: AttentionKey, flash: Bool) -> CapacitySolver.AttentionSolution? {
        var options: [AttentionAssignment] = [.graph]
        if flash, line(.gpu, .flash) != nil { options.append(.flash) }
        let groups = key.kvHeads
        if key.sequence >= HeadSplitAttention.minimumSequence, groups >= 2, line(.cpu, .heads) != nil, line(.gpu, .graph) != nil,
           model.measures([.gpu, .cpu]) {
            for cpu in 1..<groups { options.append(AttentionAssignment(kernel: .split, cpuHeads: cpu * key.group)) }
        }
        var best: (AttentionAssignment, (route: String, rows: [Kernel: Int], seconds: Double, busy: [Engine: Double]))?
        for o in options where o.isRunnable(for: key) {
            guard let t = timed(key, o) else { continue }
            if best == nil || t.seconds < best!.1.seconds * (1 - 1e-9) { best = (o, t) }
        }
        guard let (a, t) = best else { return nil }
        return .init(assignment: a, route: t.route, rows: t.rows, compute: t.seconds, busy: t.busy)
    }
}
