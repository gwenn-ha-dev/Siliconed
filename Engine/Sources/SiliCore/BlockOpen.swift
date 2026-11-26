import CoreAI
import Foundation
import Metal

/// **A block opened**: its functions built at its calls' heights, its stage planned under B, its arguments placed; the
/// forms its key and its pace name.
extension Block {
    /// **The edges' functions**, each the GPU's or the ANE's (`Declaration.neuralEdges`). An edge of the ANE's reads no
    /// codes: the ANE reads codes `[1, in, out]` from their copy only, which the engine makes for the steps of a shared
    /// form (`places`, `copy`), never for an edge — bound to the codes themselves, `[p, out, in / p]`, it would stop at
    /// a surface of another shape (`Run.neural`) or read W transposed. Refused here, before anything is placed.
    static func edges(_ d: Declaration) throws -> [String: (module: Aicode.Module, unit: ComputeUnitKind)] {
        var modules: [String: (module: Aicode.Module, unit: ComputeUnitKind)] = [:]
        for (name, build) in d.edges {
            let neural = d.neuralEdges.contains(name)
            let m = try build(Build(rows: 0, neural: neural))
            if neural, let codes = Signature(m).inputs.first(where: { $0.2 == .int8 }) {
                throw Failure("\(d.name): edge \(name) of the ANE's reads the codes \(codes.0), which the ANE reads only from a shared form's copy")
            }
            modules[edge(name)] = (m, neural ? .neuralEngine : .gpu)
        }
        return modules
    }

    /// **The block of `d` under `budget`**: the ANE's unit (`unit`, its arithmetic's) and share (`Share`; in flight, this
    /// Mac's last), its functions built at the calls' rows and named by their bytes, the plan of the first cut whose socle
    /// holds, the stage made; the ANE's share said (`Stages.computed`). In flight, its share of none (the GPU alone, the
    /// same functions) is a point of the same curve, legitimate wherever the ANE gains nothing: predicted from this Mac's
    /// κ, or its bound where it is not known yet, and visited where it is predicted the fastest (`zero`).
    public static func open(_ device: MTLDevice, _ d: Declaration, files: [SafeTensors], budget: Int, share: Share,
                            measured: ThisMac = .current) async throws -> Block {
        let began = DispatchTime.now().uptimeNanoseconds
        let o = try opening(d, files: files, budget: budget, share: share, measured: measured)
        let named = Functions.seconds(since: began)
        var p = try Stage.planned(measured: measured) { floor in
            let j = try o.judged(on: floor, measured: measured)
            return (j, j.plans[j.height])
        }
        // The trace's plan from the functions built and named on.
        p.opening.start = began
        let (heights, height) = (p.value.heights, p.value.height)
        let stage = try await Stage(device, plan: p.plan, files: files, measured: measured, opening: p.opening)
        let block = Block(d, stage: stage, heights: heights, height: height, unit: o.unit, total: o.total, units: o.units, share: share,
                          key: o.key, arithmetic: o.arithmetic, alone: measured.capacities?[o.arithmetic]?.alone,
                          shareable: d.shareable, neuralRows: o.neuralRows,
                          copies: o.copies, named: named, signatures: o.signatures, forms: o.forms, groups: o.groups, resolver: o.resolver)
        block.told()
        return block
    }

    /// **What `open` would specialize**, its assets written: the same unit and share, chosen by the same code from the
    /// files' headers, nothing allocated or read (`Stage.specializations`, the ANE's first). Every height planned, not the
    /// one taken alone: with nothing of its kin read `open` takes the soberest (`Stage.choose`) and a climb a taller one
    /// at its second turn (`climb`), once read the opening a taller one; and the block opens later on a floor higher than
    /// the preparation's, which may leave it a lower one — a height refused here is refused there (25/11/2026, the M1 Pro:
    /// a block of 4096 rows, the ANE on, prepared at 2048 rows a call, climbed to a call a pass and compiled it, 1.06 s).
    /// In flight, at every height, the GPU's functions also at the shares a move takes first — the two beside the start
    /// and the share of none (`zero`). A share met later is specialized as it loads, as any function the cache does not
    /// hold. On the floor the process comes back to (`Stage.restingFloor`).
    public static func prepare(_ d: Declaration, files: [SafeTensors], budget: Int, share: Share,
                               measured: ThisMac = .current) throws -> [Preparation] {
        let o = try opening(d, files: files, budget: budget, share: share, measured: measured)
        let j = try o.judged(on: Stage.restingFloor(), measured: measured)
        var functions = j.plans.flatMap { Array($0.declaration.functions.values) }
        if o.inFlight && o.units > 0 {
            for cut in j.heights.map(\.cut) {
                for n in [0, o.units - 1, o.units + 1] where n == 0 || (n >= 1 && n < o.total) {
                    for (_, spec) in steps(d, d.used, neural: n * o.unit, unit: o.unit, cut: cut) {
                        nonisolated(unsafe) let m = try module(spec)
                        functions.append(Stage.Function(Functions.name(m, prefix: "block"), unit: .gpu) { m })
                    }
                }
            }
        }
        return try Stage.specializations(functions)
    }

    /// **`open`'s choices, before anything is allocated**: the unit and the share, the functions built and named, every
    /// height's socle (the floor apart: `judged`); the edges' and the ANE's assets written.
    private struct Opening {
        let forms: [String: Form], groups: [String: Group]
        let key: String, arithmetic: String
        let unit: Int, units: Int, total: Int, inFlight: Bool
        /// Every height, the tallest first, and its socle — or why it has none (a tensor its files lack).
        let socles: [(height: Height, socle: Result<Stage.Plan, Error>)]
        let budget: Int
        let signatures: [String: Signature]
        let copies: Set<String>, neuralRows: Set<String>
        let resolver: Resolver

        /// **The heights that hold on `floor`, their plans and the one taken** (`Stage.choose`): the first whose estimate
        /// — its reading, or a bound from its kin's at another height or size (`Stage.estimates`) — holds under B, the
        /// soberest where nothing of its kin was read on this Mac. None holds: the last refusal.
        func judged(on floor: Int, measured: ThisMac) throws -> (heights: [Height], plans: [Stage.Plan], height: Int) {
            var heights: [Height] = [], plans: [Stage.Plan] = [], refusal: Error? = nil
            for (height, socle) in socles {
                do {
                    plans.append(try socle.get().on(floor: floor))
                    heights.append(height)
                } catch {
                    refusal = error
                }
            }
            guard !plans.isEmpty else { throw refusal! }
            let estimates = Stage.estimates(extents: heights.map { ($0.key, $0.extent) }, measured: measured, budget: budget)
            return (heights, plans, try Stage.choose(plans, estimates))
        }
    }

    private static func opening(_ d: Declaration, files: [SafeTensors], budget: Int, share: Share, measured: ThisMac) throws -> Opening {
        let forms = Dictionary(uniqueKeysWithValues: d.forms.map { ($0.name, $0) })
        let used = d.used, shareable = d.shareable
        let surfaces = used.contains(where: \.shared) || !d.neuralEdges.isEmpty

        // The unit and the share.
        let key = shareable > 0 ? try Self.key(used, shareable: shareable) : ""
        let arithmetic = Self.arithmetic(d), capacity = measured.capacities?[arithmetic]
        var unit = Self.neuralUnit(arithmetic, shareable: shareable), units = 0
        let inFlight = share == .inFlight && shareable > 0 && !d.tapped
        switch share {
        case .gpu: break
        case .told(let rows):
            guard rows == 0 || rows < shareable else { throw Failure("\(d.name): the ANE's \(rows) rows leave the GPU none of the shareable \(shareable)") }
            if rows > 0 {
                while rows % unit != 0 && unit > 64 { unit /= 2 }
                guard rows % unit == 0 else { throw Failure("\(d.name): the ANE's \(rows) rows are not whole units of 64") }
                units = rows / unit
            }
        case .inFlight:
            let total = shareable / unit
            // A block whose share of none was kept here (`probe`) runs on the GPU alone, its ANE neither built nor placed.
            if total > 1, measured.flight?[key] != 0 {
                let rows = measured.flight?[key] ?? startRows(ratio: capacity?.ratio, shareable: shareable)
                units = Self.count(rows: rows, of: unit, shareable: shareable)
            }
        }
        let total = shareable / unit

        var modules: [String: (module: Aicode.Module, unit: ComputeUnitKind)] = [:]
        for f in used where units > 0 {
            for s in f.steps where s.unit == .either { modules[neural(f.name, s.name, unit)] = (try f.module(s.name, Build(rows: unit, neural: true)), .neuralEngine) }
        }
        modules.merge(try edges(d)) { $1 }
        var assets: [String: String] = [:]
        for (name, m) in modules { assets[name] = try Functions.named(m.module, prefix: "block") }

        let groups = Dictionary(uniqueKeysWithValues: d.groups.map { ($0.name, $0) })
        // The weights' names as the functions read them: their layout's (a product's codes and its scales).
        func layout(_ entries: [Weights.Entry]) throws -> [Weights.Slot] {
            try Stage.byFile(entries, files).flatMap { f, e in try Weights.layout(e, in: f) }
        }
        var weightNames = ["": Set(try layout(d.residents).map(\.name))], heldNames: [String: Set<String>] = [:], layouts: [String: [String]] = [:]
        for g in d.groups {
            let l = try layout(g.weights)
            (weightNames[g.name], layouts[g.name]) = (Set(l.map(\.name)), l.map { "\($0.name) \($0.shape) \($0.type)" }.sorted())
            if !g.held.isEmpty { heldNames[g.name] = Set(try layout(g.held).map(\.name)) }
        }
        let resolver = Resolver(d, forms: forms, weights: weightNames, held: heldNames)
        var signatures = modules.mapValues { Signature($0.module) }
        let tensors = places(d, forms: forms, signatures: signatures, resolver: resolver, unit: units > 0 ? unit : nil)
        let copies = Set(tensors.compactMap { $0.name.hasPrefix(copy("")) ? String($0.name.dropFirst(copy("").count)) : nil })
        let neuralRows = Set(d.perRow.map(\.name).filter { r in
            tensors.contains { if case .units = $0.place { $0.name == r || $0.name == r + "#0" } else { false } }
        })

        // Every height, the tallest first, its functions built and named (not written: the stage writes what it loads),
        // its socle, what its functions hold beside (`extent`); judged on a floor by `judged`.
        let longest = d.passes.map { ($0.rows.count + 63) / 64 * 64 }.max() ?? 0
        let cuts: [Int?] = [nil] + [4096, 2048, 1024].filter { d.cuts && $0 < longest }
        var socles: [(height: Height, socle: Result<Stage.Plan, Error>)] = []
        for cut in cuts {
            let now = steps(d, used, neural: units * unit, unit: unit, cut: cut)
            for (name, spec) in now where assets[name] == nil {
                let m = try module(spec)
                (modules[name], assets[name]) = ((m, .gpu), Functions.name(m, prefix: "block"))
            }
            var s = Stage.Declaration("\(d.name), \(cut.map { "calls of \($0) rows at most" } ?? "a call a pass")\(units > 0 ? ", the ANE by \(unit)" : "")")
            s.tensors = tensors
            s.residents = d.residents + d.groups.flatMap { g in g.held.map { ("\(g.name).\($0.name)", $0.key, $0.recipe) } }
            for g in d.groups { s.groups[g.name] = g.weights }
            s.slot = d.slot ?? slot(layouts, order: d.cycle + d.once)
            (s.once, s.cycle, s.surfaces) = (d.once, d.cycle, surfaces)
            for (name, m) in modules where name.hasPrefix("edge-") || name.hasPrefix("ane-") || now[name] != nil {
                nonisolated(unsafe) let module = m.module
                s.functions[name] = Stage.Function(assets[name]!, unit: m.unit) { module }
            }
            s.extent = try extent(s.functions, inFlight: s.inFlight)
            let socle = Result { try Stage.socle(s, files: files, budget: budget) }
            socles.append((Height(cut: cut, key: Stage.key(s), extent: s.extent!), socle))
        }
        signatures = modules.mapValues { Signature($0.module) }
        return Opening(forms: forms, groups: groups, key: key, arithmetic: arithmetic, unit: unit, units: units, total: total,
                       inFlight: inFlight, socles: socles, budget: budget, signatures: signatures,
                       copies: copies, neuralRows: neuralRows, resolver: resolver)
    }

    /// **A height of the calls**: the tallest a call may be (nil, a call a pass), the key of its functions on this Mac
    /// (`Stage.key`) and what they hold (`extent`).
    struct Height: Sendable { let cut: Int?, key: String, extent: Stage.Extent }

    /// **The ring's slots' group**: the first in `order` of the layout most groups share — a group laid out otherwise is
    /// kept (`Stage`), so that a few groups of their own layout (float weights beside codes) never take the slots'.
    static func slot(_ layouts: [String: [String]], order: [String]) -> String? {
        let counts = Dictionary(grouping: layouts.values, by: { $0 }).mapValues(\.count)
        return order.filter { layouts[$0] != nil }.max { counts[layouts[$0]!]! < counts[layouts[$1]!]! }
    }

    /// **A block's arithmetic** (`ThisMac.capacities`): what the ANE reads — the Compact's codes, or f16 weights. One
    /// Mac's capacities hold for every block of one arithmetic, whatever its model or size.
    static func arithmetic(_ d: Declaration) -> String {
        d.groups.contains { $0.weights.contains { $0.recipe == .codes } } ? "codes" : "f16"
    }

    /// **The ANE's unit, by arithmetic**: the rows of one call of an ANE function, the share counted in them — 512 on
    /// the codes, 1024 on f16, and 512 for both where the shareable rows hold fewer than two units of 1024 (a unit the
    /// share cannot split in two leaves the ANE none or all). A finer unit lets the share sit closer to where the two
    /// units finish together, for a call more a step, each call reading its weights again: on the codes the grid weighs,
    /// on f16 weights, twice their bytes, the call. Measured 25/11/2026 on the M1 Pro, 4096 shareable rows: the codes,
    /// shares told, one transformer 8.55 s a cycle at 2560 rows by 512, 9.15 at the best of 1024's grid (3072), another
    /// 4.054 at 2048 rows by 512 against 4.035 by 1024; f16, the share in flight, the latter 3.72 s by 512 against 3.55
    /// by 1024, both at 2048 rows. The M5 Pro's shares kept, 22–24/11, lay on 1024's grid (1024 rows). Fixed, not tried:
    /// one measured cycle a unit, in a block's first run on a Mac, would decide the unit of every block of its arithmetic
    /// after — tried so, the first to run left it at 1024, the other at 9.57 s.
    static func neuralUnit(_ arithmetic: String, shareable: Int) -> Int {
        arithmetic == "codes" || shareable < 2 * 1024 ? 512 : 1024
    }

    /// **The ANE's rows `rows`, counted in units of `unit`**: the nearest whole count, the ANE and the GPU each at least
    /// one unit of `shareable` rows.
    static func count(rows: Int, of unit: Int, shareable: Int) -> Int {
        min(max((rows + unit / 2) / unit, 1), max(shareable / unit - 1, 1))
    }

    /// **The ANE's rows a block never run on this Mac starts at**: where this Mac's ANE and GPU, beside it, finish
    /// together — S · ρ / (1 + ρ), ρ the ANE's rate against the GPU's for the block's arithmetic (`ThisMac.Capacity`) —;
    /// none measured yet, half. (Half cost a block's first two cycles on the M5 Pro its ANE at 6.5 s beside a GPU done in
    /// 2.8, 7.9 s a cycle against 4.8 at a quarter: a transformer of 13 billion weights at 4096 rows, 23/11/2026.)
    static func startRows(ratio: Double?, shareable: Int) -> Int {
        guard let r = ratio, r > 0 else { return shareable / 2 }
        return Int((Double(shareable) * r / (1 + r)).rounded())
    }

    /// **The block's key**: the hash of its ANE's steps at 64 rows — what they compute, not their unit — and the rows
    /// they share.
    static func key(_ used: [Form], shareable: Int) throws -> String {
        var bytes = Data()
        for f in used { for s in f.steps where s.unit == .either { bytes += try f.module(s.name, Build(rows: 64, neural: true)).trimmed().bytecode() } }
        return "block " + digest(bytes) + " of \(shareable) rows"
    }

    /// **The stage's tensors**: each per-row tensor (a stream's two buffers) on the heap, or — where an ANE's step reads
    /// or writes it, beside a `unit` — a surface the ANE takes a unit at a time; each tensor as declared, or a surface whole
    /// at the signature's shape where an ANE's step binds it.
    static func places(_ d: Declaration, forms: [String: Form], signatures: [String: Signature], resolver: Resolver, unit: Int?) -> [Stage.Tensor] {
        var neuralRows = Set<String>(), neuralTensors: [String: ([Int], NDArray.ScalarType)] = [:], neuralCodes: [String: [Int]] = [:]
        if unit != nil {
            for g in d.groups where forms[g.form]!.shared {
                for p in d.passes where p.shareable != nil && p.serves(forms[g.form]!) {
                    for s in forms[g.form]!.steps where s.unit == .either {
                        let sig = signatures[neural(g.form, s.name, unit!)]!, outputs = Set(sig.outputs.map(\.0))
                        for (input, (arg, shape, type)) in sig.inputs.map({ (true, $0) }) + sig.outputs.map({ (false, $0) }) {
                            switch resolver.resolve(arg, input: input, shadowed: input && outputs.contains(arg), group: g, pass: p, step: s, call: [:]) {
                            case .rows(let t, _, _): neuralRows.insert(t)
                            case .tensor(let t, let row):
                                precondition(row == nil, "\(s.name).\(arg): a row of \(t), which the ANE reads whole")
                                neuralTensors[t] = (shape, type)
                            case .weight(let w):
                                if type == .int8 { neuralCodes[w] = shape }
                            }
                        }
                    }
                }
            }
        }
        var tensors: [Stage.Tensor] = []
        for r in d.perRow {
            let bytes = (r.rows ?? d.rows) * r.rowBytes
            func place(_ unit: Int?) -> Stage.Place {
                if let unit, neuralRows.contains(r.name) {
                    r.width == 1 ? .units(width: unit, height: 1, element: Layout.element(r.type)) : .units(width: r.width, height: unit, element: Layout.element(r.type))
                } else { r.apart ? .apart : .heap }
            }
            for name in Self.buffers(r) { tensors.append(Stage.Tensor(name, bytes: bytes, place(unit))) }
        }
        for t in d.tensors {
            guard let (shape, type) = neuralTensors[t.name] else { tensors.append(t); continue }
            tensors.append(Stage.Tensor(t.name, surface: shape, element: Layout.element(type)))
        }
        // The ANE's copy of the codes its functions read, `[1, in, out]`: one each, rewritten while the ANE is idle (`copy`).
        for (w, shape) in neuralCodes.sorted(by: { $0.key < $1.key }) {
            tensors.append(Stage.Tensor(copy(w), surface: shape, element: 1))
        }
        return tensors
    }

    /// The ANE's share said, as the fraction of the shareable rows it computes.
    func told() {
        if total > 0 && (units > 0 || share == .inFlight) {
            Stages.computed(stage: stage.plan.declaration.name, neuralEngine: Double(units) / Double(total))
        }
    }
}
