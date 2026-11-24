import CoreAI
import Foundation
import Metal

/// **A block opened**: its functions built at its calls' heights, its unit decided, its stage planned under B, its
/// arguments placed; the forms its key and its pace name.
extension Block {
    /// **The block of `d` under `budget`**: the ANE's unit and share (`Share`; in flight, this Mac's last), its functions
    /// built at the calls' rows and named by their bytes, the plan of the first cut whose socle holds, the stage made; the
    /// ANE's share said (`Stages.computed`). In flight, its share of none (the GPU alone, the same functions) is measured
    /// once on this Mac like any other share (`zero`): a point of the same curve, legitimate wherever the ANE gains nothing.
    public static func open(_ device: MTLDevice, _ d: Declaration, files: [SafeTensors], budget: Int, share: Share,
                            measured: ThisMac = .current) async throws -> Block {
        try await opened(device, d, files: files, budget: budget, share: share, measured: measured, trying: true)
    }

    /// **The edges' functions**, each the GPU's or the ANE's (`Declaration.neuralEdges`). An edge of the ANE's reads no
    /// codes: the ANE reads codes `[1, in, out]` from their copy only, which the engine makes for the steps of a shared
    /// form (`places`, `copy`), never for an edge — bound to the codes themselves, `[p, out, in / p]`, it would stop at
    /// a surface of another shape (`Run.neural`) or read W transposed. Refused here, before anything is placed.
    static func edges(_ d: Declaration) throws -> [String: (module: Aicode.Module, unit: ComputeUnitKind)] {
        var modules: [String: (module: Aicode.Module, unit: ComputeUnitKind)] = [:]
        for (name, build) in d.edges {
            let neural = d.neuralEdges.contains(name)
            let m = try Build(rows: 0, neural: neural).module(build)
            if neural, let codes = Signature(m).inputs.first(where: { $0.2 == .int8 }) {
                throw Failure("\(d.name): edge \(name) of the ANE's reads the codes \(codes.0), which the ANE reads only from a shared form's copy")
            }
            modules[edge(name)] = (m, neural ? .neuralEngine : .gpu)
        }
        return modules
    }

    /// `trying`: the unit under trial where this Mac decided none (`UnitTrial`) — false where the plan with both units'
    /// surfaces does not hold under B: the block then runs at the first candidate, the trial left to a block that fits.
    private static func opened(_ device: MTLDevice, _ d: Declaration, files: [SafeTensors], budget: Int, share: Share,
                               measured: ThisMac, trying: Bool) async throws -> Block {
        let forms = Dictionary(uniqueKeysWithValues: d.forms.map { ($0.name, $0) })
        let used = d.used, shareable = d.shareable
        let surfaces = used.contains(where: \.shared) || !d.neuralEdges.isEmpty
        let began = DispatchTime.now().uptimeNanoseconds

        // The unit and the share.
        let key = shareable > 0 ? try Self.key(used, shareable: shareable) : ""
        let arithmetic = Self.arithmetic(d), capacity = measured.capacities?[arithmetic]
        let candidates = UnitTrial.candidates(rows: shareable)
        let decided = capacity?.unit.flatMap { candidates.contains($0) ? $0 : nil }
        var unit = decided ?? candidates[0], units = 0
        let inFlight = share == .inFlight && shareable > 0 && !d.tapped
        var trial: UnitTrial? = nil
        if inFlight && trying && decided == nil && candidates.count > 1 { trial = UnitTrial(candidates) }
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
                units = UnitTrial.units(rows: rows, of: unit, shareable: shareable)
            }
        }
        let total = shareable / unit

        // Under trial both units' functions are built, loaded together, and the unit that loses let go (`settle`).
        let tried = trial?.order ?? [unit]
        var modules: [String: (module: Aicode.Module, unit: ComputeUnitKind)] = [:]
        for f in used where units > 0 {
            for u in tried {
                for s in f.steps where s.unit == .either { modules[neural(f.name, s.name, u)] = (try Build(rows: u, neural: true).module { try f.module(s.name, $0) }, .neuralEngine) }
            }
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
        let tensors = places(d, forms: forms, signatures: signatures, resolver: resolver, unit: units > 0 ? unit : nil, alternate: trial.map { $0.order[1] })
        let copies = Set(tensors.compactMap { $0.name.hasPrefix(copy("")) ? String($0.name.dropFirst(copy("").count)) : nil })
        let neuralRows = Set(d.perRow.map(\.name).filter { r in
            tensors.contains { if case .units = $0.place { $0.name == r || $0.name == r + "#0" } else { false } }
        })

        // Every height, the tallest first, its functions built and named (not written: the stage writes what it loads),
        // its plan the same socle, what its functions hold beside (`extent`): the first whose estimate — its reading, or
        // a bound from its kin's at another height or size (`Stage.estimates`) — holds under B taken (`Stage.choose`),
        // the soberest where nothing of its kin was read on this Mac.
        let longest = d.passes.map { ($0.rows.count + 63) / 64 * 64 }.max() ?? 0
        let cuts: [Int?] = [nil] + [4096, 2048, 1024].filter { d.cuts && $0 < longest }
        var heights: [Height] = [], plans: [Stage.Plan] = [], refusal: Error? = nil
        for cut in cuts {
            let now = steps(d, used, neural: units * unit, cut: cut)
            for (name, spec) in now where assets[name] == nil {
                let m = try module(spec)
                (modules[name], assets[name]) = ((m, .gpu), Functions.name(m, prefix: "block"))
            }
            var s = Stage.Declaration("\(d.name), \(cut.map { "calls of \($0) rows at most" } ?? "a call a pass")\(units > 0 ? ", the ANE by \(tried.map(String.init).joined(separator: " or "))" : "")")
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
            do {
                plans.append(try Stage.plan(s, files: files, budget: budget, floor: Memory.floor()))
                heights.append(Height(cut: cut, key: Stage.key(s), extent: s.extent!))
            } catch {
                refusal = error
            }
        }
        guard !plans.isEmpty else {
            if trial != nil { return try await opened(device, d, files: files, budget: budget, share: share, measured: measured, trying: false) }
            throw refusal!
        }
        let estimates = Stage.estimates(extents: heights.map { ($0.key, $0.extent) }, measured: measured, budget: budget)
        let height = try Stage.choose(plans, estimates)
        signatures = modules.mapValues { Signature($0.module) }
        let named = Functions.seconds(since: began)
        let stage = try await Stage(device, plan: plans[height], files: files, measured: measured)
        let block = Block(d, stage: stage, heights: heights, height: height, unit: unit, total: total, units: units, share: share, key: key,
                          arithmetic: arithmetic, trial: trial,
                          zero: inFlight && units > 0 && measured.zeros?[key] == nil,
                          shareable: shareable, neuralRows: neuralRows,
                          copies: copies, named: named, signatures: signatures, forms: forms, groups: groups, resolver: resolver)
        if let unit = decidedAtOpen(inFlight: inFlight, decided: decided, candidates: candidates, trying: trying) { block.keep(unit, best: [:]) }
        block.told()
        return block
    }

    /// **The unit a block decides as it opens**, in flight where this Mac decided none for its arithmetic: the one unit
    /// the shareable rows allow, or the first candidate where the plan with both units' surfaces did not hold under B
    /// (`trying` false) — the unit it runs at, kept. Nil: decided already, or left to the trial.
    static func decidedAtOpen(inFlight: Bool, decided: Int?, candidates: [Int], trying: Bool) -> Int? {
        guard inFlight && decided == nil, candidates.count == 1 || !trying else { return nil }
        return candidates[0]
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
        for f in used { for s in f.steps where s.unit == .either { bytes += try Build(rows: 64, neural: true).module { try f.module(s.name, $0) }.trimmed().bytecode() } }
        return "block " + digest(bytes) + " of \(shareable) rows"
    }

    /// **The stage's tensors**: each per-row tensor (a stream's two buffers) on the heap, or — where an ANE's step reads
    /// or writes it, beside a `unit` — a surface the ANE takes a unit at a time; each tensor as declared, or a surface whole
    /// at the signature's shape where an ANE's step binds it.
    static func places(_ d: Declaration, forms: [String: Form], signatures: [String: Signature], resolver: Resolver, unit: Int?,
                       alternate: Int? = nil) -> [Stage.Tensor] {
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
            // A trial's second unit takes surfaces of its own — a surface is as tall as the view the ANE reads it by — that
            // the first's rows are copied into when the block moves (`move`).
            if let alternate, neuralRows.contains(r.name) {
                for name in Self.buffers(r) { tensors.append(Stage.Tensor(name + Self.surface(alternate), bytes: bytes, place(alternate))) }
            }
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
        if total > 0 && (units > 0 || share == .inFlight) { Stages.computed(neuralEngine: Double(units) / Double(total)) }
    }
}
