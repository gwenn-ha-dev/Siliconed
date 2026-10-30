import Foundation

/// **The solver: from a machine's model and a render's classes to the `RouteTable` that renders them soonest, and the
/// time it predicts.** Pure: no Metal, no clock, no weights. It *consumes* `CapacityModel` (lines, sharing, the ANE's
/// staircase, heat, preparation) and `CapacityClasses` (the blocks of a render), and writes what the engine already reads:
/// weight rules, class rules over the AMX and ANE grids, attention rules.
///
///     class (k×n@m, kind)  ──▶  paths      widened (gpu/fp32 · amx/fp32 · ane/fp16) · bf16 / bf16act (gpu) · native int8, q8 (gpu · amx)
///                                 │            each path × each set of engines: the balanced split on the lines (`CapacityModel.split`),
///                                 │            then the integers the ENGINE will make of it (below), timed on those
///                                 ▼
///                              t = executions · (preparation(path) + T(rows))          one weight is prepared per call
///                                 ▼
///     the least t whose grid points are free ──▶ ClassRule + WeightRule + AMX / ANE grid point  ──▶  RouteTable
///     attention (S, heads, dim) ──▶ graph · flash · split (cpu heads by whole kv groups)     ──▶  AttentionRule
///
/// What sets it apart — what a solver written by analogy with "choose the fastest route per class" would get wrong:
///
///   - **It times what the engine will do, not what it asked for.** A share goes into the table as a ratio ρ with three
///     decimals; the engine turns it into rows (`Conductor.cut` on the AMX's quantum, clamped to 0.30…0.95 of the GPU;
///     `NeuralEngine.rows` on units of 256, clamped to 0.05…1). The candidates are the integers around the balance, each
///     pushed through those functions, and the block's time is read back from the finished table (`realize`), so a ρ the
///     engine reads one unit off shows in the prediction, never as a silent loss.
///   - **A preparation is paid at every call**: `WeightProducts.prepare` widens or stages the weight inside each `linear`,
///     for every execution of the class, whatever the height. It enters the time of a path; a widened weight (the ANE's and the
///     AMX's) and a native one are not the same price. A K-quant has its own figure (`kquantPreparation`).
///   - **The grids know no weight kind and no phase**: a (k, n, m) point has one ρ. Two classes that want different ones
///     (the same product in the encoder and the denoiser) are served in order of weight; the other reads the first's. In
///     the three-engine split the AMX's point sits at the GPU's side (`m − ane rows`), where `GEMM.linear` asks it.
///   - **The encoders never have the ANE, the Standard never has it, a rounded activation is a DiT's**: those members of a
///     class that shares its rule with a DiT product are timed on the route they will really take (`withoutANE`, the widened
///     GPU), not on the one the rule names.
///   - **An attention is cut by kv group**, the CPU taking whole groups (`HeadSplitAttention.partition`), the GPU keeping one
///     at least; the rule is per (S, heads, head dim), so a self- and a cross-attention of one key share it, and flash is
///     taken only where the family's code tries it (Z-Image and Anima, not the cross-attention).
///   - **Neutral on tie**: candidates come simplest first (GPU, GPU+AMX, GPU+ANE, the three), and a more complex one must be
///     faster by more than a part in a billion.
///
/// The second time (E5, the overlap between blocks) enters through `Combine`: the solver gives each block its preparation,
/// its compute and what each engine is busy for, and `combine` turns the blocks into a total. `sequential` is the sum.
package enum CapacitySolver {
    typealias Kernel = CapacityModel.Kernel
    typealias Engine = CapacityModel.Engine

    // MARK: - What goes in

    package struct Constraints: Sendable {
        package var family: Family
        package var variant: Variant
        /// Which engines and kernels exist and were built here (the map's `Capabilities`).
        package var capabilities: CapacityMap.Capabilities
        /// The rows from which the AMX, the ANE and the activation rounding are worth asking.
        package var limits: CapacityMap.Limits
        /// The AMX cut's step (`amx_quantum`): 64 under a grid, MPS's row tile — it is a setting, not in the table.
        package var amxQuantum: Int
        /// Judge the lines at their sustained rate (`heat`): a render outlasts the cold minute.
        package var sustained: Bool
        /// Kernels not proven on this Mac (a route the judge refused), left out whatever their line says.
        package var excluded: Set<CapacityModel.Kernel>

        /// The ANE: never the Standard's (fp32 products), only where the Mac has it.
        package var allowsANE: Bool { variant != .standard && capabilities.ane }

        package init(family: Family, variant: Variant, capabilities: CapacityMap.Capabilities,
                     limits: CapacityMap.Limits = .init(), amxQuantum: Int = 64, sustained: Bool = true,
                     excluded: Set<CapacityModel.Kernel> = []) {
            self.family = family; self.variant = variant; self.capabilities = capabilities; self.limits = limits
            self.amxQuantum = amxQuantum; self.sustained = sustained; self.excluded = excluded
        }

        package init(request: CapacityClasses.Request, capabilities: CapacityMap.Capabilities,
                     limits: CapacityMap.Limits = .init(), amxQuantum: Int = 64, sustained: Bool = true,
                     excluded: Set<CapacityModel.Kernel> = []) {
            self.init(family: request.family, variant: request.variant, capabilities: capabilities, limits: limits,
                      amxQuantum: amxQuantum, sustained: sustained, excluded: excluded)
        }
    }

    // MARK: - What comes out

    /// **One class of the render, as the plan runs it.**
    package struct Block: Sendable, Equatable {
        package var classID: String
        package var phase: CapacityClasses.Phase
        package var executions: Int
        /// The route as the engine names it (`gpu+ane`, `gpu+amx/int8`, `split`…), read back from the table.
        package var route: String
        /// Rows (columns, for an AMX beside a native weight; heads, for an attention) each kernel takes, as the engine cuts them.
        package var rows: [CapacityModel.Kernel: Int]
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
        package var table: RouteTable
        package var blocks: [Block]
        /// The total, `combine` of the blocks.
        package var seconds: Double
        package func seconds(of phase: CapacityClasses.Phase) -> Double {
            blocks.filter { $0.phase == phase }.reduce(0) { $0 + $1.seconds }
        }
        /// The step of the AMX's cut the table was solved for: a setting to write beside the table (`amx_quantum`).
        package var amxQuantum: Int
        /// Classes the model could not time (an attention with no line): left to the table's defaults, counted in no total.
        package var unpredicted: [String]
        package func block(_ id: String) -> Block? { blocks.first { $0.classID == id } }
    }

    // MARK: - Entry points

    package static func solve(model: CapacityModel, request: CapacityClasses.Request, constraints: Constraints,
                              combine: Combine = sequential) -> Plan? {
        guard let classes = CapacityClasses.classes(for: request) else { return nil }
        return solve(model: model, classes: classes, constraints: constraints, combine: combine)
    }

    /// `nil` when the model has no line for the GPU's fp32 product: nothing can be planned without the reference engine.
    package static func solve(model: CapacityModel, classes: [CapacityClasses.Class], constraints: Constraints,
                              combine: Combine = sequential) -> Plan? {
        var solver = RouteSolver(model: model, c: constraints)
        return solver.run(classes: classes, combine: combine)
    }

    // MARK: - Numbers of the engine, restated

    /// `Conductor.cut` with this step: the GPU's rows of `m`.
    static func cut(_ m: Int, _ T: Double, quantum q: Int) -> Int {
        max(q, min(m - q, Int((Double(m) * T / Double(q)).rounded()) * q))
    }
    /// The ratio a grid point is written with, three decimals (`Conductor.Shares.description`): what the engine reads.
    static func written(share s: Double) -> Double { ((1 - s) / s * 1000).rounded() / 1000 }
    static func share(ofRatio r: Double) -> Double { 1 / (1 + r) }
    static func clamp(_ x: Double, _ r: ClosedRange<Double>) -> Double { min(r.upperBound, max(r.lowerBound, x)) }

    /// The GPU's rows of `height` the engine will cut for a ratio ρ of the AMX grid.
    static func amxGPURows(height: Int, ratio: Double, quantum: Int) -> Int {
        cut(height, clamp(share(ofRatio: ratio), Conductor.fractionRange), quantum: quantum)
    }
    /// The ANE's rows of `m` the engine will take for a ratio ρ of the ANE grid.
    static func aneRows(m: Int, ratio: Double, minimum: Int) -> Int {
        NeuralEngine.rows(m, gpuShare: clamp(share(ofRatio: ratio), NeuralEngine.shareRange), minimum: minimum)
    }
}

// MARK: - The model's per-kernel time

extension CapacityModel {
    /// **Seconds each kernel is busy** in a product `m×k×n` with these rows (the ANE's rounded up to its unit): the sides of
    /// `seconds`, before the maximum and the set's overhead.
    package func busy(m: Int, k: Int, n: Int, rows: [Kernel: Int], sustained: Bool = false) -> [Kernel: Double] {
        let kernels = rows.keys.filter { rows[$0]! > 0 }
        let set = sharing(Array(kernels))
        var result: [Kernel: Double] = [:]
        for kernel in kernels {
            guard let r = rate(kernel, in: set, sustained: sustained), let t0 = lines[kernel]?.t0 else { continue }
            var here = rows[kernel]!
            if kernel.engine == .ane { here = (here + aneRowUnit - 1) / aneRowUnit * aneRowUnit }
            result[kernel] = t0 + 2 * Double(k) * Double(n) * Double(here) / r
        }
        return result
    }
}

// MARK: - The solver proper

private struct RouteSolver {
    typealias Kernel = CapacityModel.Kernel
    typealias Engine = CapacityModel.Engine
    typealias S = CapacitySolver
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
        swapped ? model.seconds(m: n, k: k, n: m, rows: rows, sustained: c.sustained)
                : model.seconds(m: m, k: k, n: n, rows: rows, sustained: c.sustained)
    }
    func busy(_ rows: [Kernel: Int], m: Int, k: Int, n: Int, swapped: Bool = false) -> [Engine: Double] {
        let b = swapped ? model.busy(m: n, k: k, n: m, rows: rows, sustained: c.sustained)
                        : model.busy(m: m, k: k, n: n, rows: rows, sustained: c.sustained)
        return Dictionary(b.map { ($0.key.engine, $0.value) }, uniquingKeysWith: +)
    }

    // MARK: Claims on the grids

    enum Grid: Hashable { case amx, ane }
    struct Point: Hashable { var grid: Grid; var k, n, m: Int }
    struct Claim: Equatable { var point: Point; var ratio: Double }

    // MARK: A way to compute one product

    struct Eval {
        var engines: RouteTable.Engines
        var rows: [Kernel: Int]
        /// The split is along the columns (an AMX beside a native weight): the dimensions of the model's product are swapped.
        var swapped = false
        var claims: [Claim] = []
        var usesANE: Bool { engines == .ane || engines == .aneAMX }
        var usesAMX: Bool { engines == .amx || engines == .aneAMX }
    }

    struct Option {
        var path: RouteTable.WeightPath
        var rounds: Bool
        var eval: Eval
        var seconds: Double      // one execution's compute
    }

    /// The candidate AMX cuts of a product of `height` rows, around the continuous balance `g0` GPU rows.
    func amxCuts(height: Int, g0: Double) -> [(gpu: Int, ratio: Double)] {
        let q = c.amxQuantum
        guard height >= 2 * q else { return [] }
        let base = Int((g0 / Double(q)).rounded())
        var seen = Set<Int>(), out: [(Int, Double)] = []
        for d in -2...2 {
            let g = max(q, min(height - q, (base + d) * q))
            let t = S.clamp(Double(g) / Double(height), Conductor.fractionRange)
            let ratio = S.written(share: t)
            let real = S.amxGPURows(height: height, ratio: ratio, quantum: q)
            if seen.insert(real).inserted { out.append((real, ratio)) }
        }
        return out
    }

    /// The candidate ANE row counts of `m`, in the engine's units, around the continuous balance `r0`.
    func aneCuts(m: Int, r0: Double) -> [(rows: Int, ratio: Double)] {
        let unit = NeuralEngine.rowUnit
        let u0 = r0 / Double(unit)
        var units = Set([1])
        for u in (Int(u0.rounded(.down)) - 1)...(Int(u0.rounded(.up)) + 1) where u >= 1 { units.insert(u) }
        var seen = Set<Int>(), out: [(Int, Double)] = []
        for u in units.sorted() {
            let r = u * unit
            guard r < m else { continue }
            let t = S.clamp(1 - Double(r) / Double(m), NeuralEngine.shareRange)
            let ratio = S.written(share: t)
            let real = S.aneRows(m: m, ratio: ratio, minimum: c.limits.aneRows)
            if real > 0, seen.insert(real).inserted { out.append((real, ratio)) }
        }
        return out
    }

    /// **The widened path's candidates, one per set of engines** (GPU alone, +AMX, +ANE, the three), each the best integer
    /// cut the engine can make around the balance of the lines.
    func widened(k: Int, n: Int, m: Int, ane allowedANE: Bool) -> [(Eval, Double)] {
        guard let gpu = line(.gpu, .fp32) else { return [] }
        var out: [(Eval, Double)] = []
        if let t = seconds([gpu: m], m: m, k: k, n: n) { out.append((Eval(engines: .gpu, rows: [gpu: m]), t)) }
        let amx = c.capabilities.amx && m >= c.limits.amxRows ? line(.amx, .fp32) : nil
        let ane = allowedANE && m >= c.limits.aneRows ? line(.ane, .fp16) : nil

        func best(_ candidates: [(Eval, Double)]) -> (Eval, Double)? { candidates.min { $0.1 < $1.1 } }

        if let amx, let seed = model.split(m: m, k: k, n: n, kernels: [gpu, amx], sustained: c.sustained) {
            let g0 = Double(seed.rows[gpu] ?? Int(Double(m) * Conductor.fractionRange.upperBound))
            let cands = amxCuts(height: m, g0: g0).compactMap { cut -> (Eval, Double)? in
                let rows = [gpu: cut.gpu, amx: m - cut.gpu]
                guard let t = seconds(rows, m: m, k: k, n: n) else { return nil }
                return (Eval(engines: .amx, rows: rows, claims: [.init(point: .init(grid: .amx, k: k, n: n, m: m), ratio: cut.ratio)]), t)
            }
            if let b = best(cands) { out.append(b) }
        }
        if let ane, let seed = model.split(m: m, k: k, n: n, kernels: [gpu, ane], sustained: c.sustained) {
            let cands = aneCuts(m: m, r0: Double(seed.rows[ane] ?? 0)).compactMap { cut -> (Eval, Double)? in
                let rows = [gpu: m - cut.rows, ane: cut.rows]
                guard let t = seconds(rows, m: m, k: k, n: n) else { return nil }
                return (Eval(engines: .ane, rows: rows, claims: [.init(point: .init(grid: .ane, k: k, n: n, m: m), ratio: cut.ratio)]), t)
            }
            if let b = best(cands) { out.append(b) }
        }
        if let amx, let ane, let seed = model.split(m: m, k: k, n: n, kernels: [gpu, amx, ane], sustained: c.sustained) {
            var cands: [(Eval, Double)] = []
            let sideGPU = Double(seed.rows[gpu] ?? 0), sideAMX = Double(seed.rows[amx] ?? 0)
            let fraction = sideGPU + sideAMX > 0 && sideAMX > 0 ? sideGPU / (sideGPU + sideAMX) : Conductor.fractionRange.upperBound
            for a in aneCuts(m: m, r0: Double(seed.rows[ane] ?? 0)) {
                let rest = m - a.rows
                guard rest >= c.limits.amxRows else { continue }       // the conductor does not apply below `amx_min`
                for g in amxCuts(height: rest, g0: Double(rest) * fraction) {
                    let rows = [gpu: g.gpu, amx: rest - g.gpu, ane: a.rows]
                    guard let t = seconds(rows, m: m, k: k, n: n) else { continue }
                    // The GPU's side goes through the conductor at its own height: the AMX's point is there.
                    cands.append((Eval(engines: .aneAMX, rows: rows, claims: [
                        .init(point: .init(grid: .ane, k: k, n: n, m: m), ratio: a.ratio),
                        .init(point: .init(grid: .amx, k: k, n: n, m: rest), ratio: g.ratio)]), t))
                }
            }
            if let b = best(cands) { out.append(b) }
        }
        return out
    }

    /// **A native 8-bit weight**: the GPU alone, or beside the AMX on the last columns (multiples of 64 on the GPU's side).
    func native(form: CapacityModel.Form, k: Int, n: Int, m: Int) -> [(Eval, Double)] {
        guard let gpu = line(.gpu, form) else { return [] }
        var out: [(Eval, Double)] = []
        if let t = seconds([gpu: m], m: m, k: k, n: n) { out.append((Eval(engines: .gpu, rows: [gpu: m]), t)) }
        guard c.capabilities.amx, m >= c.limits.amxRows, n >= 128, let amx = line(.amx, form),
              let seed = model.split(m: n, k: k, n: m, kernels: [gpu, amx], sustained: c.sustained) else { return out }
        let g0 = Double(seed.rows[gpu] ?? n)
        var cands: [(Eval, Double)] = []
        var seen = Set<Int>()
        let base = Int((g0 / 64).rounded())
        for d in -2...2 {
            let g = max(64, min((n - 64) / 64 * 64, (base + d) * 64))
            guard g >= 64 else { continue }
            let ratio = S.written(share: S.clamp(Double(g) / Double(n), Conductor.fractionRange))
            let columns = WeightProducts.amxColumns(n: n, share: S.share(ofRatio: ratio))
            guard columns > 0, seen.insert(columns).inserted else { continue }
            let rows = [gpu: n - columns, amx: columns]
            guard let t = seconds(rows, m: m, k: k, n: n, swapped: true) else { continue }
            cands.append((Eval(engines: .amx, rows: rows, swapped: true,
                               claims: [.init(point: .init(grid: .amx, k: k, n: n, m: m), ratio: ratio)]), t))
        }
        if let b = cands.min(by: { $0.1 < $1.1 }) { out.append(b) }
        return out
    }

    // MARK: Preparation

    func preparation(format: CapacityClasses.WeightFormat, path: RouteTable.WeightPath, k: Int, n: Int) -> Double {
        let heat = c.sustained ? max(1, model.heat[.cpu] ?? 1) : 1
        let perValue: Double
        if let kind = format.kind { perValue = model.preparation[kind]?[path] ?? 0 }
        else { perValue = model.kquantPreparation ?? model.preparation[.bf16]?[.widened] ?? 0 }
        return Double(k) * Double(n) * perValue * heat
    }

    // MARK: Product groups

    struct Member {
        var index: Int                       // the class's place in the input
        var phase: CapacityClasses.Phase
        var format: CapacityClasses.WeightFormat
        var executions: Int
    }
    struct Group {
        var k, n, m: Int
        var kind: RouteTable.WeightKind      // the key of the rules (a K-quant is ruled as bf16)
        var members: [Member]
        var weight: Double
    }

    /// Everything one group can do, simplest first, each with its time over all its members.
    struct Candidate { var option: Option; var total: Double }

    func candidates(_ g: Group) -> (list: [Candidate], bestNonANE: Option?, gpuWidened: Option?) {
        let (k, n, m) = (g.k, g.n, g.m)
        let anyDiT = g.members.contains { $0.phase != .encoding }
        var options: [Option] = []
        for (e, t) in widened(k: k, n: n, m: m, ane: c.allowsANE && anyDiT) {
            options.append(Option(path: .widened, rounds: false, eval: e, seconds: t))
        }
        let nonANE = options.filter { !$0.eval.usesANE }
        let gpuWidened = options.first { $0.eval.engines == .gpu }
        let bestNonANE = nonANE.min { $0.seconds < $1.seconds }
        let caps = c.capabilities
        let formats = Set(g.members.map(\.format))
        if formats == [.bf16], caps.bf16Weights {
            if let gpu = line(.gpu, .bf16), let t = seconds([gpu: m], m: m, k: k, n: n) {
                options.append(Option(path: .bf16, rounds: false, eval: Eval(engines: .gpu, rows: [gpu: m]), seconds: t))
            }
            if anyDiT, caps.bf16Rounding, m >= c.limits.roundingRows, let gpu = line(.gpu, .bf16act),
               let t = seconds([gpu: m], m: m, k: k, n: n) {
                options.append(Option(path: .bf16, rounds: true, eval: Eval(engines: .gpu, rows: [gpu: m]), seconds: t))
            }
        }
        if formats.count == 1, let f = formats.first, let kind = f.kind, kind != .bf16,
           kind == .int8 ? caps.int8Weights : caps.q8Weights {
            for (e, t) in native(form: kind == .int8 ? .int8 : .q8, k: k, n: n, m: m) {
                options.append(Option(path: .native, rounds: false, eval: e, seconds: t))
            }
        }
        let list = options.map { o in Candidate(option: o, total: total(o, g, bestNonANE: bestNonANE, gpuWidened: gpuWidened)) }
        return (list, bestNonANE, gpuWidened)
    }

    /// The time of all a group's members on an option; an encoding member of a rule that names the ANE, or a rounded activation,
    /// runs the route it falls back to.
    func total(_ o: Option, _ g: Group, bestNonANE: Option?, gpuWidened: Option?) -> Double {
        var sum = 0.0
        for member in g.members {
            var (path, seconds) = (o.path, o.seconds)
            if member.phase == .encoding {
                if o.eval.usesANE, let f = bestNonANE { (path, seconds) = (.widened, f.seconds) }
                else if o.rounds, let f = gpuWidened { (path, seconds) = (.widened, f.seconds) }
            }
            sum += Double(member.executions) * (seconds + preparation(format: member.format, path: path, k: g.k, n: g.n))
        }
        return sum
    }

    // MARK: Attentions

    struct AttentionMember {
        var index: Int
        var sequence, keys, heads, kv, dim: Int
        var cross: Bool
        var executions: Int
    }
    struct AttentionGroup {
        var sequence, heads, dim: Int
        var members: [AttentionMember]
    }
    struct AttentionOption: Equatable { var flash: Bool; var fraction: Double? }

    func flashFor(_ m: AttentionMember) -> Bool {
        !m.cross && (c.family == .zImage || c.family == .anima)
    }

    /// The kernel and rows an attention member runs under a rule, as `CapacityClasses.route(of:)` and `HeadSplitAttention`
    /// read it: flash first where the family tries it, then the split, else the graph.
    func attention(_ m: AttentionMember, flash: Bool, fraction: Double?) -> (route: String, rows: [Kernel: Int], k: Int)? {
        let per = 2 * m.sequence * m.keys * m.dim                      // 2·k with n = 1: 4·S·keys·dim per head
        if flash, flashFor(m), let f = line(.gpu, .flash), c.capabilities.flash { return ("flash", [f: m.heads], per) }
        guard let graph = line(.gpu, .graph) else { return nil }
        if let fraction, c.capabilities.headSplit, let cpu = line(.cpu, .heads) {
            let part = HeadSplitAttention.partition(heads: m.heads, fraction: fraction, group: max(1, m.heads / max(1, m.kv)))
            if part.cpu.count > 0 { return ("split", [graph: part.gpu.count, cpu: part.cpu.count], per) }
        }
        return ("graph", [graph: m.heads], per)
    }

    func attentionSeconds(_ m: AttentionMember, flash: Bool, fraction: Double?) -> Double? {
        guard let a = attention(m, flash: flash, fraction: fraction) else { return nil }
        return model.seconds(m: m.heads, k: a.k, n: 1, rows: a.rows, sustained: c.sustained)
    }

    func bestAttention(_ g: AttentionGroup) -> AttentionOption? {
        var options = [AttentionOption(flash: false, fraction: nil)]
        if c.capabilities.flash, line(.gpu, .flash) != nil, g.members.contains(where: flashFor) {
            options.append(.init(flash: true, fraction: nil))
        }
        let first = g.members[0]
        let groups = first.kv                                          // the kv groups: the CPU takes whole ones
        if c.capabilities.headSplit, g.sequence >= HeadSplitAttention.minimumSequence, groups >= 2,
           line(.cpu, .heads) != nil, line(.gpu, .graph) != nil {
            for cpu in 1..<groups {
                let f = Double(cpu) / Double(groups)
                options.append(.init(flash: false, fraction: f))
                if options.contains(where: { $0.flash && $0.fraction == nil }) { options.append(.init(flash: true, fraction: f)) }
            }
        }
        var best: (AttentionOption, Double)?
        for o in options {
            var sum = 0.0
            var ok = true
            for m in g.members {
                guard let t = attentionSeconds(m, flash: o.flash, fraction: o.fraction) else { ok = false; break }
                sum += Double(m.executions) * t
            }
            if ok, best == nil || sum < best!.1 * (1 - 1e-9) { best = (o, sum) }
        }
        return best?.0
    }

    // MARK: The run

    mutating func run(classes: [CapacityClasses.Class], combine: CapacitySolver.Combine) -> CapacitySolver.Plan? {
        guard line(.gpu, .fp32) != nil else { return nil }
        let family = c.family

        // Groups of products by (k, n, m, kind), attentions by (S, heads, dim): what a rule can tell apart.
        var groups: [Group] = [], groupOf: [String: Int] = [:]
        var attentions: [AttentionGroup] = [], attentionOf: [String: Int] = [:]
        for (i, cl) in classes.enumerated() {
            let executions = cl.executions[family] ?? 0
            guard executions > 0 else { continue }
            switch cl.shape {
            case let .product(k, n, rows, format):
                let kind = format.kind ?? .bf16
                let key = "\(k)x\(n)@\(rows)~\(kind.rawValue)"
                let member = Member(index: i, phase: cl.phase, format: format, executions: executions)
                if let g = groupOf[key] {
                    groups[g].members.append(member); groups[g].weight += Double(executions) * cl.flop
                } else {
                    groupOf[key] = groups.count
                    groups.append(Group(k: k, n: n, m: rows, kind: kind, members: [member], weight: Double(executions) * cl.flop))
                }
            case let .attention(s, keys, heads, kv, dim, cross):
                let key = "\(s)/\(heads)x\(dim)"
                let member = AttentionMember(index: i, sequence: s, keys: keys, heads: heads, kv: kv, dim: dim, cross: cross,
                                             executions: executions)
                if let g = attentionOf[key] { attentions[g].members.append(member) }
                else {
                    attentionOf[key] = attentions.count
                    attentions.append(AttentionGroup(sequence: s, heads: heads, dim: dim, members: [member]))
                }
            }
        }

        // The products, heaviest first: the first to claim a grid point keeps it.
        var claimed: [Point: Double] = [:]
        func free(_ claims: [Claim]) -> Bool { claims.allSatisfy { claimed[$0.point] == nil || claimed[$0.point] == $0.ratio } }
        struct Chosen { var group: Int; var option: Option; var fallback: Option?; var gpuWidened: Option? }
        var chosen: [Chosen] = []
        for gi in groups.indices.sorted(by: { groups[$0].weight > groups[$1].weight }) {
            let (list, bestNonANE, gpuWidened) = candidates(groups[gi])
            var pick: Candidate?
            for cand in list {
                if pick == nil || cand.total < pick!.total * (1 - 1e-9) {
                    if free(cand.option.eval.claims) { pick = cand }
                }
            }
            guard let pick else { continue }
            for claim in pick.option.eval.claims { claimed[claim.point] = claim.ratio }
            chosen.append(Chosen(group: gi, option: pick.option, fallback: bestNonANE, gpuWidened: gpuWidened))
        }

        // The route the rule keeps without the ANE (a process that did not bring it up): the AMX only if the plan already
        // brings the AMX up and its point at the full height is free — nothing is enabled for a fallback alone.
        let amxUp = chosen.contains { $0.option.eval.usesAMX }
        var withoutANE: [Int: RouteTable.Engines] = [:]
        for ch in chosen where ch.option.eval.usesANE {
            if amxUp, let f = ch.fallback, f.eval.engines == .amx, free(f.eval.claims) {
                for claim in f.eval.claims { claimed[claim.point] = claim.ratio }
                withoutANE[ch.group] = .amx
            } else { withoutANE[ch.group] = .gpu }
        }

        // The table.
        var table = RouteTable()
        let aneUp = chosen.contains { $0.option.eval.usesANE }
        table.products.amx = .init(enabled: amxUp || withoutANE.values.contains(.amx))
        table.products.ane = .init(enabled: aneUp, gpuShare: 1.0, minimumRows: c.limits.aneRows)
        for ch in chosen {
            let g = groups[ch.group]
            table.products.classes.append(.init(engines: ch.option.eval.engines, k: g.k, n: g.n, minRows: g.m, maxRows: g.m,
                                                withoutANE: ch.option.eval.usesANE ? withoutANE[ch.group] : nil, kind: g.kind))
            // A weight pin only where the tensor is of the kind (a K-quant is none): its form, the widened one too.
            if groups[ch.group].members.contains(where: { $0.format.kind != nil }) {
                table.products.weights.insert(.init(kind: g.kind, path: ch.option.path, roundsActivation: ch.option.rounds,
                                                    k: g.k, n: g.n, minRows: g.m, maxRows: g.m), at: 0)
            }
        }
        for kind in RouteTable.WeightKind.allCases { table.products.classes.append(.init(engines: .gpu, kind: kind)) }
        var amxPoints: [Conductor.Shares.Point] = [], anePoints: [Conductor.Shares.Point] = []
        for (p, ratio) in claimed.sorted(by: { ($0.key.m, $0.key.k, $0.key.n) < ($1.key.m, $1.key.k, $1.key.n) }) {
            let point = Conductor.Shares.Point(k: p.k, n: p.n, m: p.m, ratio: ratio)
            if p.grid == .amx { amxPoints.append(point) } else { anePoints.append(point) }
        }
        // Where the ANE is up, every product it could take says its share, the GPU's whole (ρ = 0) included: the memory need reads
        // the ANE's rows off this grid (`NeuralEngine.bytes`), and a product with no point would be charged the default half.
        if aneUp {
            for g in groups where g.m >= c.limits.aneRows && g.members.contains(where: { $0.phase != .encoding })
                && !claimed.keys.contains(Point(grid: .ane, k: g.k, n: g.n, m: g.m)) {
                anePoints.append(.init(k: g.k, n: g.n, m: g.m, ratio: 0))
            }
        }
        table.products.amx.grid = .init(Conductor.Shares(amxPoints))
        table.products.ane.grid = .init(Conductor.Shares(anePoints))

        // The attentions.
        var unpredicted: [String] = []
        for g in attentions {
            guard let o = bestAttention(g) else {
                unpredicted += g.members.map { classes[$0.index].routeKey.id }; continue
            }
            table.attention.rules.insert(.init(minSequence: g.sequence, maxSequence: g.sequence, heads: g.heads, headDim: g.dim,
                                               flash: o.flash, splitFraction: o.fraction), at: 0)
        }

        // The blocks, read back from the table.
        var blocks: [CapacitySolver.Block] = []
        for (i, cl) in classes.enumerated() {
            let executions = cl.executions[family] ?? 0
            guard executions > 0 else { continue }
            switch cl.shape {
            case let .product(k, n, m, format):
                guard let r = realize(table, k: k, n: n, m: m, format: format, phase: cl.phase),
                      let t = seconds(r.rows, m: m, k: k, n: n, swapped: r.swapped) else { continue }
                let prep = preparation(format: format, path: r.path, k: k, n: n)
                blocks.append(.init(classID: cl.routeKey.id + RouteTable.ClassKey.suffix(format.kind ?? .bf16), phase: cl.phase,
                                    executions: executions, route: r.route, rows: r.rows,
                                    preparation: Double(executions) * prep, compute: Double(executions) * t,
                                    busy: busy(r.rows, m: m, k: k, n: n, swapped: r.swapped).mapValues { $0 * Double(executions) }))
            case let .attention(s, keys, heads, kv, dim, cross):
                let a = table.attention.route(sequence: s, heads: heads, headDim: dim)
                let member = AttentionMember(index: i, sequence: s, keys: keys, heads: heads, kv: kv, dim: dim, cross: cross,
                                             executions: executions)
                guard let r = attention(member, flash: a.flash, fraction: a.splitFraction),
                      let t = model.seconds(m: heads, k: r.k, n: 1, rows: r.rows, sustained: c.sustained) else { continue }
                let b = model.busy(m: heads, k: r.k, n: 1, rows: r.rows, sustained: c.sustained)
                blocks.append(.init(classID: cl.routeKey.id, phase: cl.phase, executions: executions, route: r.route, rows: r.rows,
                                    preparation: 0, compute: Double(executions) * t,
                                    busy: Dictionary(b.map { ($0.key.engine, $0.value * Double(executions)) }, uniquingKeysWith: +)))
            }
        }
        return .init(table: table, blocks: blocks, seconds: combine(blocks), amxQuantum: c.amxQuantum, unpredicted: unpredicted)
    }

    // MARK: Reading the table as the engine does

    struct Realized {
        var path: RouteTable.WeightPath
        var rows: [Kernel: Int]
        var swapped: Bool
        var route: String
    }

    /// **What the engine makes of the table for a product**: the weight's form (`RouteTable.Products.weight`, the rounding
    /// only for a DiT), the engines (`route`, an encoder without the ANE), then the rows each takes (`Conductor.cut` at the
    /// height the engine asks the grid at, `NeuralEngine.rows`, `WeightProducts.amxColumns`).
    func realize(_ table: RouteTable, k: Int, n: Int, m: Int, format: CapacityClasses.WeightFormat,
                 phase: CapacityClasses.Phase) -> Realized? {
        let p = table.products
        let broughtUp = c.allowsANE && phase != .encoding
        let amxOn = c.capabilities.amx && m >= c.limits.amxRows
        var path = RouteTable.WeightPath.widened, rounds = false
        if let kind = format.kind {
            let w = p.weight(kind, k: k, n: n, m: m) { rule in rule.path == .widened || !(rule.roundsActivation && phase == .encoding) }
            (path, rounds) = (w.path, w.roundsActivation)
        }
        let kind = format.kind ?? .bf16
        switch path {
        case .bf16:
            guard let gpu = line(.gpu, rounds ? .bf16act : .bf16) else { return nil }
            return Realized(path: .bf16, rows: [gpu: m], swapped: false, route: rounds ? "gpu/bf16act" : "gpu/bf16")
        case .native:
            let form: CapacityModel.Form = kind == .int8 ? .int8 : .q8
            guard let gpu = line(.gpu, form) else { return nil }
            let forced = p.forced(k: k, n: n, m: m, kind: kind, aneBroughtUp: broughtUp)
            if amxOn, n >= 128, forced == .amx || forced == .aneAMX, let amx = line(.amx, form) {
                let columns = WeightProducts.amxColumns(n: n, share: p.amx.share(k: k, n: n, m: m))
                if columns > 0 {
                    return Realized(path: .native, rows: [gpu: n - columns, amx: columns], swapped: true, route: "gpu+amx/\(form.rawValue)")
                }
            }
            return Realized(path: .native, rows: [gpu: m], swapped: false, route: "gpu/\(form.rawValue)")
        case .widened:
            guard let gpu = line(.gpu, .fp32) else { return nil }
            let r = p.route(k: k, n: n, m: m, kind: kind, aneBroughtUp: broughtUp)
            if r.aneRows > 0, let ane = line(.ane, .fp16) {
                let first = m - r.aneRows
                if r.amxBesideANE, first >= c.limits.amxRows, let amx = line(.amx, .fp32) {
                    let gpuRows = S.cut(first, S.clamp(p.amx.share(k: k, n: n, m: first), Conductor.fractionRange), quantum: c.amxQuantum)
                    return Realized(path: .widened, rows: [gpu: gpuRows, amx: first - gpuRows, ane: r.aneRows], swapped: false,
                                    route: "gpu+ane+amx")
                }
                return Realized(path: .widened, rows: [gpu: first, ane: r.aneRows], swapped: false, route: "gpu+ane")
            }
            if r.amx, amxOn, let amx = line(.amx, .fp32) {
                let gpuRows = S.cut(m, S.clamp(p.amx.share(k: k, n: n, m: m), Conductor.fractionRange), quantum: c.amxQuantum)
                return Realized(path: .widened, rows: [gpu: gpuRows, amx: m - gpuRows], swapped: false, route: "gpu+amx")
            }
            return Realized(path: .widened, rows: [gpu: m], swapped: false, route: "gpu")
        }
    }
}
