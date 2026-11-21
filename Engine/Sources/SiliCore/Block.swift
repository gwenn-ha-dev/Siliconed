import CoreAI
import CryptoKit
import Foundation
import Metal

/// **A block: a transformer's loop as its client declares it, run by the engine.** The client writes its functions
/// (`Form.module`, the edges) and says what each argument is: rows of a per-row tensor (`Rows`, a stream two buffers in
/// turn), a tensor, a weight. The engine builds every function at the shapes its calls take, names each asset by its
/// bytes, plans the stage under B (`Stage.plan`), binds each argument of each call by its name and its signature, puts the
/// calls on the stream ahead of the GPU, and shares the passes with the ANE.
///
///     a group      weights a form computes with (a layer), kept or read through the ring (`Stage`)
///     a form       steps, each a function over rows (`either` unit, or the GPU's)
///     a pass       the rows a group's steps cover, the part the ANE may take (`shareable`, from the pass's first row)
///     an edge      a function called once (an embedding, a head), its rows from a given one; the GPU's or the ANE's
///
/// **An argument is bound by its name, first match** (`resolve`): the call's binding, the step's, the pass's, the form's;
/// a step's `whole` (rows from 0); an input named as a weight of the group or a resident; a per-row tensor (the call's
/// rows; a stream's current buffer, its `next` name the other); the group's own per-row tensor or tensor (`<group>.<name>`);
/// a tensor.
///
/// **Tall calls** (`calls`): a step of the GPU's is one call over the pass, a step of either unit one call over the rows
/// past the ANE's; a call's height is its rows rounded up to 64, the rows past the pass computed and never read, so that
/// no remainder takes a call of its own (measured 19/11 on the M5 Pro, Z's Exact at 1024²: the 64-row remainder +0.13 s
/// an evaluation, tiles of 1024 beside the ANE +5 to +8 % on the GPU against one call). A height met first is built and
/// specialized then, once a Mac. Every height is planned — one call a pass, then equal calls of 4096 rows at most, 2048,
/// 1024 — and the stage takes the tallest whose P and what Core AI was read to hold under its functions on this Mac hold
/// under B (`Stage.choose`); nothing read, the soberest whose socle holds.
///
/// **What Core AI holds is read in flight** (`Stage.signal`): no share counted in P; the block gives the stage its signal
/// once the second group of a pass is done, from which the ring's reader keeps the cycle's groups while B holds them.
///
/// **Calls put ahead**: each GPU call is encoded while the one before runs (the arena tracked: `Arena`), each waited for
/// only where its bytes are read: before a function is called again (one invocation in flight: Core AI allocates another
/// working memory for a second), before the ANE reads what the GPU wrote (the ANE's call waits for the GPU's calls
/// before it), before the CPU reads (`pointer`, `values`), before a slot of the ring is read again.
///
/// **The ANE's share** (`Share`) in units of `unit` rows of the shareable rows from their start, every unit one call of
/// one ANE function a step, from its byte offset in each per-row tensor's surface (`Arena.take(units:)`): none, a
/// control's (`told`), or found in flight — each cycle's calls timed, the share moved at the next cycle to the one
/// `predicted` to finish first (phase by phase, as `run` orders the calls) if it gains 3 % at least, its rows kept on this
/// Mac (`ThisMac.flight`). **The unit is the engine's**, decided once per block key on this Mac (`ThisMac.neuralUnits`),
/// never changed after: a finer unit balances the two closer, each a call more of the ANE's. The first block of a key to
/// run finds it undecided, builds both candidates (`UnitTrial`), runs cycles at each, and keeps the faster (`settle`).
public final class Block: @unchecked Sendable {
    // MARK: - The declaration

    /// A per-row tensor: `width` values of `type` a row, every row of the block. `next`: a stream, two buffers in turn, the
    /// outputs named `next` writing the other, swapped after a pass that wrote it. `factor`: the per-row tensor its true
    /// values are multiplied by (`values`). `rows`: fewer rows than the block's, for a pass over its first rows alone (its
    /// calls rounded up to 64 within them).
    public struct Rows: Sendable {
        public var name: String
        public var width: Int
        public var type: NDArray.ScalarType
        public var next: String?
        public var factor: String?
        /// Its own rows, not the block's (a cache's); `apart`: in buffers of its own (`Stage.Place.apart`), what a stage's
        /// last call writes.
        public var rows: Int?
        public var apart: Bool
        public init(_ name: String, width: Int, _ type: NDArray.ScalarType, next: String? = nil, factor: String? = nil, rows: Int? = nil,
                    apart: Bool = false) {
            self.name = name; self.width = width; self.type = type; self.next = next; self.factor = factor; self.rows = rows; self.apart = apart
        }
        var rowBytes: Int { width * Layout.element(type) }
    }

    /// What an argument is bound to: the call's rows of a per-row tensor, or of a stream's other buffer; a per-row tensor
    /// from row 0; a tensor whole; row `i` of a tensor, a row being the argument's bytes rounded to a page.
    public enum Bind: Sendable, Equatable {
        case rows(String), next(String), whole(String), tensor(String), row(String, Int)
    }

    /// Which unit computes a step: `either`, the ANE on its units and the GPU past them; `gpu`, the GPU on every row.
    public enum Unit: Sendable, Equatable {
        case either, gpu
    }

    /// A step: its function computes each row alone (`either` unit: the ANE on its units, the GPU past them), or is the
    /// GPU's on every row (it reads across rows, or what the ANE cannot run); every call of a step after the step before's.
    public struct Step: Sendable {
        public var name: String
        public var unit: Unit
        public var whole: Set<String>
        public var bind: [String: Bind]
        /// The step has a tapped form (`Build.tapped`): a function that writes more, for a check listening inside the block.
        public var taps: Bool
        public init(_ name: String, on unit: Unit = .either, whole: Set<String> = [], bind: [String: Bind] = [:], taps: Bool = false) {
            self.name = name; self.unit = unit; self.whole = whole; self.bind = bind; self.taps = taps
        }
    }

    /// What a function is built for: its rows (a call's, a unit's; an edge's 0), the layouts of the plan's products, its
    /// int8 codes' and its bf16 weights' (read once as the block opens: `Stage.Declaration.codes`).
    public struct Build: Sendable {
        public let rows: Int
        public let codes: CodesLayout
        public let bfloat16: CodesLayout
        /// A tapped run's function of a step that taps (`Step.taps`): the module writes what the check listens to.
        public let tapped: Bool
        public init(rows: Int, codes: CodesLayout, bfloat16: CodesLayout = .outIn, tapped: Bool = false) {
            self.rows = rows; self.codes = codes; self.bfloat16 = bfloat16; self.tapped = tapped
        }
    }

    public struct Form: Sendable {
        public var name: String
        public var steps: [Step]
        public var bind: [String: Bind]
        public var module: @Sendable (_ step: String, _ build: Build) throws -> Aicode.Module
        public init(_ name: String, steps: [Step], bind: [String: Bind] = [:], module: @escaping @Sendable (String, Build) throws -> Aicode.Module) {
            self.name = name; self.steps = steps; self.bind = bind; self.module = module
        }
        var shared: Bool { steps.contains { $0.unit == .either } }
    }

    /// `held`: weights of the group kept whatever B (residents named `<group>.<name>`), bound by their own names in the
    /// group's calls — a `run`'s, or an edge's outside one (`call(_:group:)`): what a loop reads once, before its cycles.
    public struct Group: Sendable {
        public var name: String
        public var form: String
        public var weights: [Weights.Entry]
        public var held: [Weights.Entry]
        public init(_ name: String, form: String, weights: [Weights.Entry], held: [Weights.Entry] = []) {
            self.name = name; self.form = form; self.weights = weights; self.held = held
        }
    }

    /// `forms`: the forms whose groups run over it (nil: every form), the only ones built at its calls' rows.
    public struct Pass: Sendable {
        public var name: String
        public var rows: Range<Int>
        public var shareable: Range<Int>?
        /// The rows past which no call of the pass reaches (the per-row tensors it touches hold that many): nil, the block's.
        public var limit: Int?
        public var bind: [String: Bind]
        public var forms: Set<String>?
        public init(_ name: String, rows: Range<Int>, shareable: Range<Int>? = nil, limit: Int? = nil, bind: [String: Bind] = [:],
                    forms: Set<String>? = nil) {
            precondition(shareable.map { $0.lowerBound == rows.lowerBound && $0.upperBound <= rows.upperBound } ?? true,
                         "\(name): the ANE's rows from the pass's first")
            precondition(limit.map { $0 >= rows.upperBound } ?? true, "\(name): a limit under the pass's rows")
            self.name = name; self.rows = rows; self.shareable = shareable; self.limit = limit; self.bind = bind; self.forms = forms
        }
        func serves(_ f: Form) -> Bool { forms?.contains(f.name) ?? true }
    }

    public struct Declaration: Sendable {
        public var name: String
        /// Every per-row tensor's rows: the calls' rows never pass them.
        public var rows: Int
        public var perRow: [Rows] = []
        /// The tensors no file holds and no row indexes; one an ANE's step reads becomes a surface at its signature's shape.
        public var tensors: [Stage.Tensor] = []
        public var residents: [Weights.Entry] = []
        public var groups: [Group] = []
        /// The loop's use of the groups: each of `once` in turn, then `cycle` over and over (`Stage`).
        public var once: [String] = []
        public var cycle: [String] = []
        /// The group whose layout the ring's slots take (`Stage.Declaration.slot`); nil, the layout most groups share (`slot`).
        public var slot: String? = nil
        /// Whether a pass may be cut into calls of fewer rows (B too tight for one): not where a function computes across
        /// every row of the pass it writes (a layer's own attention).
        public var cuts = true
        public var forms: [Form] = []
        public var passes: [Pass] = []
        public var edges: [String: @Sendable (Build) throws -> Aicode.Module] = [:]
        /// The edges the ANE computes (`call` runs them there, every GPU call before them done): their tensors surfaces
        /// as declared, the weights surfaces.
        public var neuralEdges: Set<String> = []
        /// Passes may be run tapped (`run`'s `tapped`): the functions of the steps that tap are built in both forms, and the
        /// calls over every row of a pass the GPU's alone, at their own heights.
        public var tapped = false
        public init(_ name: String, rows: Int) { self.name = name; self.rows = rows }
    }

    /// The ANE's share: none; a control's rows (0, the shared forms on the GPU alone); found in flight.
    public enum Share: Sendable, Equatable { case gpu, told(Int), inFlight }

    // MARK: - The block

    public let declaration: Declaration
    public let stage: Stage
    /// The tallest call now: nil, a call a step and a pass (`heights`, `climb`).
    public var cut: Int? { heights[height].cut }
    /// The heights the calls may take, the tallest first, and the one taken; the turns of the cycle begun at it.
    let heights: [Height]
    private(set) var height: Int
    private var turns = 0
    /// The ANE's unit (rows), the shareable units, the ANE's now (0: the GPU alone).
    public private(set) var unit: Int, total: Int
    public private(set) var units: Int
    /// The shareable rows; the unit the surfaces of the per-row tensors the ANE reads (`neuralRows`) are named for, the
    /// others' beside them named `name@<unit>` (a trial's second unit: `places`).
    private let shareable: Int, home: Int
    private let neuralRows: Set<String>
    /// The unit under trial (`UnitTrial`; nil: decided).
    private var trial: UnitTrial?
    /// The buffers a GPU call has read or written (`encode`).
    private var seen = Set<ObjectIdentifier>()
    /// The cycle under way, told once done (`Stages.Event.cycle`): its start, the ANE's rows over it, what its start loaded;
    /// the calls of the cycle before, summed once timed (`rebalance`).
    private var begun: (at: UInt64, rows: Int, moved: Double?, climbed: Double?)?
    private var lastCalls: Stages.Calls?
    /// What the loop waited for the ring's weights, in seconds; what building and naming the functions took at open.
    public private(set) var waited = 0.0
    public let named: Double

    private let share: Share
    /// The block's key on this Mac: its ANE's steps and its shareable rows (`ThisMac.flight`, `ThisMac.neuralUnits`).
    private let key: String
    /// Its pace key (`paceKey`, `ThisMac.paces`; empty: not timed) and whether it carries the unit (`pace`), each pass's
    /// cycle starts, the pace last kept.
    private let paceBase: String, paceByUnit: Bool
    private var starts: [String: [UInt64]] = [:], kept: Double?
    /// The pace key now: none while the unit is under trial — a cycle at another unit is no pace of the unit kept —, else
    /// the key at the unit the block runs at.
    private var pace: String { paceBase.isEmpty || trial != nil ? "" : paceBase + (paceByUnit ? Self.byUnit(unit) : "") }
    private let codes: CodesLayout
    var functions: [String: InferenceFunction]
    private var signatures: [String: Signature]
    let groups: [String: Group]
    private let forms: [String: Form], passes: [String: Pass], perRow: [String: Rows]
    private let resolver: Resolver
    private var current: [String: Int] = [:]
    var held: (group: Group, weights: Stage.Held)?
    /// The last call of each function on the stream, and the last of all: each a task done when it is.
    var marks: [String: Task<UInt64, Never>] = [:], last: Task<UInt64, Never>?
    private var cycleRecords: [[Timed]] = []
    /// The pass the last groups ran over, and how many in a row (`signal`).
    private var pass: String?, inPass = 0

    /// A function's arguments and results: their shapes and types, as its module declares them.
    struct Signature {
        var inputs: [(String, [Int], NDArray.ScalarType)]
        var outputs: [(String, [Int], NDArray.ScalarType)]
        init(_ m: Aicode.Module) {
            func t(_ e: Aicode.Entry) -> ([Int], NDArray.ScalarType) { let (s, el) = Aicode.shape(e); return (s, Block.scalar(el)) }
            inputs = zip(m.argumentNames, m.body.arguments).map { n, e in let (s, ty) = t(e); return (n, s, ty) }
            outputs = m.outputs.map { o in let (s, ty) = t(m.body.types[o.value]); return (o.name, s, ty) }
        }
    }

    static func scalar(_ e: Aicode.Element) -> NDArray.ScalarType {
        switch e {
        case .f32: .float32
        case .f16: .float16
        case .bf16: .bfloat16
        case .si8: .int8
        case .si32: .int32
        default: preconditionFailure("no argument of \(e)")
        }
    }

    /// The names of the stage's functions: a step's at a GPU call's rows, the ANE's at its unit, an edge.
    static func gpu(_ form: String, _ step: String, _ rows: Int, tapped: Bool = false) -> String { "\(form).\(step)\(tapped ? "+taps" : "")-\(rows)" }
    /// An ANE function at its unit: both candidates' are loaded while the unit is under trial.
    static func neural(_ form: String, _ step: String, _ unit: Int) -> String { "ane-\(form).\(step)@\(unit)" }
    static func edge(_ name: String) -> String { "edge-\(name)" }

    // MARK: - The calls' rows

    /// **The calls over `rows`**: one, its height the rows rounded up to 64; under `cut`, as few calls of equal height at
    /// most `cut` as cover them, each rounded up to 64. No call passes `limit` (the per-row tensors' rows): the last is
    /// shortened to it.
    static func calls(_ rows: Range<Int>, cut: Int?, limit: Int) -> [Tiles.Tile] {
        guard !rows.isEmpty else { return [] }
        let k = cut.map { (rows.count + $0 - 1) / $0 } ?? 1
        let height = ((rows.count + k - 1) / k + 63) / 64 * 64
        return (0..<k).map { i in
            let start = rows.lowerBound + i * height
            return Tiles.Tile(start: start, rows: min(height, limit - start))
        }.filter { $0.start < rows.upperBound }
    }

    /// **The GPU's calls of step `s` of form `f` over pass `p`, the ANE's first `neural` rows aside**: an `either` step's
    /// past them, a `gpu` step's over the pass. (A `gpu` step in two calls beside the ANE — the rows past the ANE's
    /// while it ends, then its rows, or its rows first so that it starts its next step — let the ANE take more rows but
    /// cost the GPU more than it gave: Z at 1024², 2.37 s an evaluation at the best share against 2.31, the attention in
    /// two calls 0.49 s against 0.41; measured 19/11 on the mini.)
    static func calls(_ f: Form, _ s: Step, _ p: Pass, neural: Int, cut: Int?, limit: Int) -> [Tiles.Tile] {
        let from = s.unit == .either && f.shared && p.shareable != nil ? p.rows.lowerBound + neural : p.rows.lowerBound
        return calls(from..<p.rows.upperBound, cut: cut, limit: limit)
    }

    /// The phase of an `either` step beside the ANE: the steps between two `gpu` steps, counted by the `gpu` steps before
    /// it. A `gpu` step waits for both units.
    static func phase(_ f: Form, _ i: Int) -> Int { f.steps[..<i].filter { $0.unit == .gpu }.count }

    /// **The GPU's functions of the steps at `neural` rows of the ANE's**: every form used over every pass, by name.
    static func steps(_ d: Declaration, _ used: [Form], neural: Int, cut: Int?) -> [String: (form: Form, step: String, rows: Int, tapped: Bool)] {
        var all: [String: (form: Form, step: String, rows: Int, tapped: Bool)] = [:]
        for f in used {
            for p in d.passes where p.serves(f) {
                for s in f.steps {
                    for t in calls(f, s, p, neural: neural, cut: cut, limit: p.limit ?? d.rows) { all[gpu(f.name, s.name, t.rows)] = (f, s.name, t.rows, false) }
                    // A tapped run leaves the ANE: every row of the pass the GPU's, a step that taps in its tapped form.
                    if d.tapped {
                        for t in calls(f, s, p, neural: 0, cut: cut, limit: p.limit ?? d.rows) { all[gpu(f.name, s.name, t.rows, tapped: s.taps)] = (f, s.name, t.rows, s.taps) }
                    }
                }
            }
        }
        return all
    }

    // MARK: - Opening

    /// **The block of `d` under `budget`**: the ANE's unit and share (`Share`; in flight, this Mac's last), its functions
    /// built at the calls' rows and named by their bytes, the plan of the first cut whose socle holds, the stage made; the
    /// ANE's share said (`Stages.computed`).
    public static func open(_ device: MTLDevice, _ d: Declaration, files: [SafeTensors], budget: Int, share: Share,
                            measured: ThisMac = .current) async throws -> Block {
        try await opened(device, d, files: files, budget: budget, share: share, measured: measured, trying: true)
    }

    /// `trying`: the unit under trial where this Mac decided none (`UnitTrial`) — false where the plan with both units'
    /// surfaces does not hold under B: the block then runs at the first candidate, the trial left to a block that fits.
    private static func opened(_ device: MTLDevice, _ d: Declaration, files: [SafeTensors], budget: Int, share: Share,
                               measured: ThisMac, trying: Bool) async throws -> Block {
        let forms = Dictionary(uniqueKeysWithValues: d.forms.map { ($0.name, $0) })
        let used = Set(d.groups.map(\.form)).sorted().map { forms[$0]! }
        let sharing = used.contains(where: \.shared)
        let surfaces = sharing || !d.neuralEdges.isEmpty
        // The layouts the functions are written for and the plan lays the weights in: int8 codes `[in, out]`, bf16
        // weights as the file lies. Measured per Mac until 20/11 on one product: on the mini it chose `[out, in]` for
        // the codes, which made Z Compacte's whole render 3.5 % slower than this default.
        let codes: CodesLayout = .inOut, bfloat16: CodesLayout = .outIn
        let shareable = sharing ? d.passes.compactMap(\.shareable).map(\.count).max() ?? 0 : 0
        let began = DispatchTime.now().uptimeNanoseconds

        // The unit and the share.
        let key = shareable > 0 ? try Self.key(used, codes: codes, shareable: shareable) : ""
        let candidates = UnitTrial.candidates(rows: shareable)
        let decided = measured.neuralUnits?[key].flatMap { candidates.contains($0) ? $0 : nil }
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
            if total > 1 { units = min(max((measured.flight?[key] ?? total / 2 * unit) / unit, 1), total - 1) }
        }
        let total = shareable / unit

        // Under trial both units' functions are built, loaded together, and the unit that loses let go (`settle`).
        let tried = trial?.order ?? [unit]
        var modules: [String: (module: Aicode.Module, unit: ComputeUnitKind)] = [:]
        for f in used where units > 0 {
            for u in tried {
                for s in f.steps where s.unit == .either { modules[neural(f.name, s.name, u)] = (try f.module(s.name, Build(rows: u, codes: codes, bfloat16: bfloat16)), .neuralEngine) }
            }
        }
        for (name, build) in d.edges { modules[edge(name)] = (try build(Build(rows: 0, codes: codes, bfloat16: bfloat16)), d.neuralEdges.contains(name) ? .neuralEngine : .gpu) }
        var assets: [String: String] = [:]
        for (name, m) in modules { assets[name] = try Functions.named(m.module, prefix: "block") }

        let groups = Dictionary(uniqueKeysWithValues: d.groups.map { ($0.name, $0) })
        // The weights' names as the functions read them: their layout's (a product's codes and its scales).
        func layout(_ entries: [Weights.Entry]) throws -> [Weights.Slot] {
            try Stage.byFile(entries, files).flatMap { f, e in try Weights.layout(e, in: f, codes: codes, bfloat16: bfloat16) }
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
        let neuralRows = Set(d.perRow.map(\.name).filter { r in
            tensors.contains { if case .units = $0.place { $0.name == r || $0.name == r + "#0" } else { false } }
        })

        // Every height, the tallest first, its functions built and named (not written: the stage writes what it loads),
        // its plan the same socle: the first whose estimate holds under B taken (`Stage.choose`), the soberest where
        // nothing was read on this Mac.
        let longest = d.passes.map { ($0.rows.count + 63) / 64 * 64 }.max() ?? 0
        let cuts: [Int?] = [nil] + [4096, 2048, 1024].filter { d.cuts && $0 < longest }
        var heights: [Height] = [], plans: [Stage.Plan] = [], refusal: Error? = nil
        for cut in cuts {
            let now = steps(d, used, neural: units * unit, cut: cut)
            for (name, spec) in now where assets[name] == nil {
                let m = try spec.form.module(spec.step, Build(rows: spec.rows, codes: codes, bfloat16: bfloat16, tapped: spec.tapped))
                (modules[name], assets[name]) = ((m, .gpu), Functions.name(m, prefix: "block"))
            }
            var s = Stage.Declaration("\(d.name), \(cut.map { "calls of \($0) rows at most" } ?? "a call a pass")\(units > 0 ? ", the ANE by \(tried.map(String.init).joined(separator: " or "))" : "")")
            s.tensors = tensors
            s.residents = d.residents + d.groups.flatMap { g in g.held.map { ("\(g.name).\($0.name)", $0.key, $0.recipe) } }
            for g in d.groups { s.groups[g.name] = g.weights }
            s.slot = d.slot ?? slot(layouts, order: d.cycle + d.once)
            (s.once, s.cycle, s.surfaces, s.codes, s.bfloat16) = (d.once, d.cycle, surfaces, codes, bfloat16)
            for (name, m) in modules where name.hasPrefix("edge-") || name.hasPrefix("ane-") || now[name] != nil {
                nonisolated(unsafe) let module = m.module
                s.functions[name] = Stage.Function(assets[name]!, unit: m.unit) { module }
            }
            do {
                plans.append(try Stage.plan(s, files: files, budget: budget, floor: Memory.footprint()))
                heights.append(Height(cut: cut, rows: calls(0..<longest, cut: cut, limit: longest).map(\.rows).max() ?? 0, key: Stage.key(s)))
            } catch {
                refusal = error
            }
        }
        guard !plans.isEmpty else {
            if trial != nil { return try await opened(device, d, files: files, budget: budget, share: share, measured: measured, trying: false) }
            throw refusal!
        }
        let estimates = Stage.estimates(heights.map { ($0.key, $0.rows) }, measured: measured)
        let height = try Stage.choose(plans, estimates)
        signatures = modules.mapValues { Signature($0.module) }
        let named = Functions.seconds(since: began)
        let stage = try await Stage(device, plan: plans[height], files: files, measured: measured)
        // Timed where it runs as a render would: not under a control's share, nor tapped.
        let told: Bool = if case .told = share { true } else { false }
        let pace = told || d.tapped ? "" : try paceKey(d)
        let block = Block(d, stage: stage, heights: heights, height: height, unit: unit, total: total, units: units, share: share, key: key,
                          pace: (pace, shareable > 0 && share == .inFlight), trial: trial, shareable: shareable, neuralRows: neuralRows,
                          codes: codes, named: named, signatures: signatures, forms: forms, groups: groups, resolver: resolver)
        // One unit the shareable rows allow: decided at once.
        if inFlight && decided == nil && candidates.count == 1 { block.keep(unit, best: [:]) }
        block.told()
        return block
    }

    /// **A height of the calls**: the tallest a call may be (nil, a call a pass), the rows the tallest call covers, the
    /// key of its functions on this Mac (`Stage.key`).
    struct Height: Sendable { let cut: Int?, rows: Int, key: String }

    /// **The ring's slots' group**: the first in `order` of the layout most groups share — a group laid out otherwise is
    /// kept (`Stage`), so that a few groups of their own layout (float weights beside codes) never take the slots'.
    static func slot(_ layouts: [String: [String]], order: [String]) -> String? {
        let counts = Dictionary(grouping: layouts.values, by: { $0 }).mapValues(\.count)
        return order.filter { layouts[$0] != nil }.max { counts[layouts[$0]!]! < counts[layouts[$1]!]! }
    }

    /// **A declaration's pace key** (`ThisMac.paces`): the hash of every step of its forms at 64 rows — what it computes,
    /// whatever its share and its calls' heights — and its rows.
    public static func paceKey(_ d: Declaration) throws -> String {
        let forms = Dictionary(uniqueKeysWithValues: d.forms.map { ($0.name, $0) })
        var bytes = Data()
        for f in Set(d.groups.map(\.form)).sorted().map({ forms[$0]! }) {
            for s in f.steps { bytes += try f.module(s.name, Build(rows: 64, codes: .inOut)).trimmed().bytecode() }
        }
        return "pace " + Data(SHA256.hash(data: bytes)).prefix(8).map { String(format: "%02x", $0) }.joined() + " of \(d.rows) rows"
    }

    /// The key of a declaration's unit (`ThisMac.neuralUnits`); nil where none of its rows may go to the ANE.
    static func unitKey(_ d: Declaration) throws -> String? {
        let forms = Dictionary(uniqueKeysWithValues: d.forms.map { ($0.name, $0) })
        let used = Set(d.groups.map(\.form)).sorted().map { forms[$0]! }
        let shareable = used.contains(where: \.shared) ? d.passes.compactMap(\.shareable).map(\.count).max() ?? 0 : 0
        return shareable > 0 ? try key(used, codes: .inOut, shareable: shareable) : nil
    }

    /// What a pace key carries once its block shares the ANE's rows: the unit it was taken at.
    static func byUnit(_ unit: Int) -> String { " by \(unit)" }

    /// **Which of `alternatives` a client opens** — declarations of one computation, each its own arithmetic: the first
    /// never timed on this Mac (`ThisMac.paces`), each in turn, then the fastest. Chosen here, by system (`ThisMac`), at
    /// the cost of one opening each at its own pace; an alternative's pace is its last block's, a share in flight as far
    /// as it went.
    public static func choose(_ alternatives: [Declaration], measured: ThisMac = .current) throws -> Int {
        // A pace counts at the unit kept (`pace`): a declaration that shares the ANE's rows has none until its unit is decided.
        let paces = try alternatives.map { d -> Double? in
            var key = try paceKey(d)
            if let block = try unitKey(d) {
                guard let unit = measured.neuralUnits?[block] else { return nil }
                key += byUnit(unit)
            }
            return measured.paces?[key]
        }
        if let first = paces.firstIndex(where: { $0 == nil }) { return first }
        return paces.indices.min { paces[$0]! < paces[$1]! }!
    }

    /// **The block's key**: the hash of its ANE's steps at 64 rows — what they compute, not their unit — and the rows
    /// they share.
    static func key(_ used: [Form], codes: CodesLayout, shareable: Int) throws -> String {
        var bytes = Data()
        for f in used { for s in f.steps where s.unit == .either { bytes += try f.module(s.name, Build(rows: 64, codes: codes)).trimmed().bytecode() } }
        return "block " + Data(SHA256.hash(data: bytes)).prefix(8).map { String(format: "%02x", $0) }.joined() + " of \(shareable) rows"
    }

    /// **The stage's tensors**: each per-row tensor (a stream's two buffers) on the heap, or — where an ANE's step reads
    /// or writes it, beside a `unit` — a surface the ANE takes a unit at a time; each tensor as declared, or a surface whole
    /// at the signature's shape where an ANE's step binds it.
    static func places(_ d: Declaration, forms: [String: Form], signatures: [String: Signature], resolver: Resolver, unit: Int?,
                       alternate: Int? = nil) -> [Stage.Tensor] {
        var neuralRows = Set<String>(), neuralTensors: [String: ([Int], NDArray.ScalarType)] = [:]
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
                            case .weight: break
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
            for name in r.next == nil ? [r.name] : [r.name + "#0", r.name + "#1"] { tensors.append(Stage.Tensor(name, bytes: bytes, place(unit))) }
            // A trial's second unit takes surfaces of its own — a surface is as tall as the view the ANE reads it by — that
            // the first's rows are copied into when the block moves (`move`).
            if let alternate, neuralRows.contains(r.name) {
                for name in r.next == nil ? [r.name] : [r.name + "#0", r.name + "#1"] { tensors.append(Stage.Tensor(name + Self.surface(alternate), bytes: bytes, place(alternate))) }
            }
        }
        for t in d.tensors {
            guard let (shape, type) = neuralTensors[t.name] else { tensors.append(t); continue }
            tensors.append(Stage.Tensor(t.name, surface: shape, element: Layout.element(type)))
        }
        return tensors
    }

    private init(_ d: Declaration, stage: Stage, heights: [Height], height: Int, unit: Int, total: Int, units: Int, share: Share, key: String,
                 pace: (base: String, byUnit: Bool), trial: UnitTrial?, shareable: Int, neuralRows: Set<String>, codes: CodesLayout, named: Double,
                 signatures: [String: Signature], forms: [String: Form], groups: [String: Group], resolver: Resolver) {
        (declaration, self.stage, self.heights, self.height, self.unit, self.total, self.units, self.share) = (d, stage, heights, height, unit, total, units, share)
        (self.key, self.paceBase, self.paceByUnit, self.trial, self.shareable, self.home, self.neuralRows) = (key, pace.base, pace.byUnit, trial, shareable, unit, neuralRows)
        (self.codes, self.named, self.signatures, self.forms, self.groups, self.resolver) = (codes, named, signatures, forms, groups, resolver)
        functions = stage.functions
        passes = Dictionary(uniqueKeysWithValues: d.passes.map { ($0.name, $0) })
        perRow = resolver.perRow
        for r in d.perRow where r.next != nil { current[r.name] = 0 }
    }

    /// The ANE's share said, as the fraction of the shareable rows it computes.
    private func told() {
        if units > 0 { Stages.computed(neuralEngine: Double(units) / Double(total)) }
    }

    // MARK: - Binding

    /// What an argument resolves to: a weight by its name; a tensor (a row of it); a per-row tensor, its stream's other
    /// buffer, from row 0.
    enum Target: Equatable { case weight(String), tensor(String, row: Int?), rows(String, next: Bool, whole: Bool) }

    struct Resolver: Sendable {
        let forms: [String: Form]
        let perRow: [String: Rows]
        let nextOf: [String: String]
        let tensorNames: Set<String>
        /// Each group's weights' names, the residents' under ""; each group's held weights' (`Group.held`).
        let weightNames: [String: Set<String>]
        let heldNames: [String: Set<String>]

        init(_ d: Declaration, forms: [String: Form], weights: [String: Set<String>], held: [String: Set<String>] = [:]) {
            self.forms = forms
            perRow = Dictionary(uniqueKeysWithValues: d.perRow.map { ($0.name, $0) })
            nextOf = Dictionary(uniqueKeysWithValues: d.perRow.compactMap { r in r.next.map { ($0, r.name) } })
            tensorNames = Set(d.tensors.map(\.name))
            weightNames = weights
            heldNames = held
        }

        /// `shadowed`: an input of the name of one of the function's outputs (a product's codes, and its result): the
        /// bindings are the output's. A per-row tensor named — by the argument, a binding or `whole` — is the group's own
        /// (`<group>.<name>`) where the group has one: a layer's cache, a layer's keys.
        func resolve(_ arg: String, input: Bool, shadowed: Bool = false, group: Group?, pass: Pass?, step: Step?, call: [String: Bind]) -> Target {
            let target = named(arg, input: input, shadowed: shadowed, group: group, pass: pass, step: step, call: call)
            if case .rows(let t, let next, let whole) = target, let g = group, perRow["\(g.name).\(t)"] != nil {
                return .rows("\(g.name).\(t)", next: next, whole: whole)
            }
            return target
        }

        private func named(_ arg: String, input: Bool, shadowed: Bool, group: Group?, pass: Pass?, step: Step?, call: [String: Bind]) -> Target {
            let form = group.map { forms[$0.form]! }
            if !shadowed, let b = call[arg] ?? step?.bind[arg] ?? pass?.bind[arg] ?? form?.bind[arg] {
                switch b {
                case .rows(let t): return .rows(t, next: false, whole: false)
                case .next(let t): return .rows(t, next: true, whole: false)
                case .whole(let t): return .rows(t, next: false, whole: true)
                case .tensor(let t): return .tensor(t, row: nil)
                case .row(let t, let i): return .tensor(t, row: i)
                }
            }
            if step?.whole.contains(arg) == true { return .rows(arg, next: false, whole: true) }
            if input, let g = group, weightNames[g.name]?.contains(arg) == true { return .weight(arg) }
            if input, let g = group, heldNames[g.name]?.contains(arg) == true { return .weight("\(g.name).\(arg)") }
            if input, weightNames[""]?.contains(arg) == true { return .weight(arg) }
            if perRow[arg] != nil || group.map({ perRow["\($0.name).\(arg)"] != nil }) == true { return .rows(arg, next: false, whole: false) }
            if let s = nextOf[arg] { return .rows(s, next: true, whole: false) }
            if let g = group, tensorNames.contains("\(g.name).\(arg)") { return .tensor("\(g.name).\(arg)", row: nil) }
            if tensorNames.contains(arg) { return .tensor(arg, row: nil) }
            preconditionFailure("no binding for \(arg)\(group.map { " in \($0.name)" } ?? "")\(step.map { ", step \($0.name)" } ?? "")")
        }
    }

    /// The stage's tensor of per-row tensor `name`: a stream's current buffer or its other.
    /// A tensor the ANE reads is the surfaces' of the unit the block runs at (`surface`).
    private func tensor(_ name: String, next: Bool) -> String {
        let surface = neuralRows.contains(name) ? Self.surface(unit == home ? nil : unit) : ""
        guard let c = current[name] else { return name + surface }
        return "\(name)#\(next ? 1 - c : c)" + surface
    }

    /// The suffix of the names of a trial's second unit's surfaces (the first's have none).
    static func surface(_ unit: Int?) -> String { unit.map { "@\($0)" } ?? "" }

    /// The views of function `f`'s arguments and results over rows from `start` (the ANE's: surfaces), but those of `skip`.
    func views(_ f: String, start: Int, group: Group?, pass: Pass?, step: Step?, call: [String: Bind], neural: Bool,
               skip: Set<String> = [], written: inout Set<String>) -> (inputs: [String: View], outputs: [(String, View)]) {
        let sig = signatures[f]!, outputs = Set(sig.outputs.map(\.0))
        func view(_ arg: String, _ shape: [Int], _ type: NDArray.ScalarType, input: Bool) -> View {
            switch resolver.resolve(arg, input: input, shadowed: input && outputs.contains(arg), group: group, pass: pass, step: step, call: call) {
            case .weight(let w):
                guard let p = held?.weights.weights[w] ?? stage.residents[w] else { preconditionFailure("\(f): no weight \(w)") }
                return stage.view(p, shape)
            case .tensor(let t, let row):
                let offset = row.map { $0 * Arena.aligned(shape.reduce(1, *) * Layout.element(type)) } ?? 0
                return neural ? stage.neuralView(t, shape, type, bytes: offset) : stage.view(t, shape, type, bytes: offset)
            case .rows(let t, let next, let whole):
                guard let r = perRow[t] else { preconditionFailure("\(f).\(arg): no per-row tensor \(t)") }
                precondition(r.type == type, "\(f).\(arg): \(t) is \(r.type), the function reads \(type)")
                if !input && current[t] != nil && next { written.insert(t) }
                let name = tensor(t, next: next), offset = whole ? 0 : start * r.rowBytes
                precondition(offset + shape.reduce(1, *) * Layout.element(type) <= (r.rows ?? declaration.rows) * r.rowBytes,
                             "\(f).\(arg): past the rows of \(t)")
                return neural ? stage.neuralView(name, shape, type, bytes: offset) : stage.view(name, shape, type, bytes: offset)
            }
        }
        var inputs: [String: View] = [:]
        for (n, s, t) in sig.inputs where !skip.contains(n) { inputs[n] = view(n, s, t, input: true) }
        return (inputs, sig.outputs.filter { !skip.contains($0.0) }.map { n, s, t in (n, view(n, s, t, input: false)) })
    }

    // MARK: - Calls

    /// One GPU call put on the stream once the function's last is done: the task done when this one is, and its time.
    private func encode(_ f: String, _ io: (inputs: [String: View], outputs: [(String, View)])) async throws -> Timing {
        if let m = marks[f] { _ = await m.value }
        // A buffer Core AI sees for the first time (a part of a surface made for this call, `Arena.part`) is read out of
        // the stream's order: the first pass of a block shared with the ANE read its GPU rows of a stream an edge had just
        // written as zeros, the second the right values; the stream waited for first, 1.5e-4 of the GPU alone's, f16's
        // rounding (21/11 on the mini, `aSharedBlockIsTheGPUAlones`). Once a buffer, a few at a block's opening.
        let buffers = (io.inputs.values.map { $0 } + io.outputs.map(\.1)).map { ObjectIdentifier($0.buffer) }
        if !buffers.allSatisfy(seen.contains) {
            await settled()
            seen.formUnion(buffers)
        }
        // An argument from off a page (a factor's rows, a value each) is copied by Core AI outside the stream's order: put
        // ahead, the call read it before the call before wrote it (measured 19/11 on the M5 Pro, Z's f16 velocities apart;
        // the same bits once such a call waits for the stream).
        if (io.inputs.values.map { $0 } + io.outputs.map(\.1)).contains(where: { $0.offset % Arena.page != 0 }) { await settled() }
        let left = try Run.encode(functions[f]!, stage.stream, inputs: io.inputs, outputs: io.outputs)
        precondition(left.isEmpty, "\(f): outputs no view holds, \(left.keys.sorted())")
        nonisolated(unsafe) let stream = stage.stream!
        let timing = Timing(before: last, encoded: DispatchTime.now().uptimeNanoseconds,
                            done: Task { await stream.currentWorkCompleted(); return DispatchTime.now().uptimeNanoseconds })
        marks[f] = timing.done
        last = timing.done
        // And its result from off a page is copied back the same way, after the call: the next call reads it once it is, or
        // reads what the call before wrote (measured 19/11 on the mini, a shared form beside the ANE at 512², the rows past
        // the ANE's of a factor a row each, from byte 1024: the attention's gate and keys were the previous block's in
        // one run of two; with this wait, 6 in 6, the same bits and time on Z's).
        if io.outputs.contains(where: { $0.1.offset % Arena.page != 0 }) { await settled() }
        return timing
    }

    /// A GPU call's time once done: from its encoding or the call before's end, the later, to its end.
    struct Timing {
        let before: Task<UInt64, Never>?
        let encoded: UInt64
        let done: Task<UInt64, Never>
        func seconds() async -> Double {
            let end = await done.value, start = max(encoded, await before?.value ?? 0)
            return Double(end > start ? end - start : 0) / 1e9
        }
    }

    /// Every call put on the stream, done.
    public func settled() async { _ = await last?.value }

    /// **Edge `name`'s call** (`Declaration.edges`): its per-row arguments from row `row`, the group a run holds its
    /// weights' and its own tensors' (`run`'s `before`), or `group`'s held weights and its own tensors outside a run
    /// (its other weights are not held). An edge of the ANE's (`neuralEdges`) runs once every call on the stream is
    /// done, and is done when this returns.
    public func call(_ name: String, group: String? = nil, from row: Int = 0, _ bind: [String: Bind] = [:]) async throws {
        var written = Set<String>()
        let f = Self.edge(name), g = group.map { groups[$0]! } ?? held?.group
        if declaration.neuralEdges.contains(name) {
            await settled()
            let io = views(f, start: row, group: g, pass: nil, step: nil, call: bind, neural: true, written: &written)
            try await Run.neural(functions[f]!, inputs: io.inputs, outputs: io.outputs)
            return
        }
        _ = try await encode(f, views(f, start: row, group: g, pass: nil, step: nil, call: bind, neural: false, written: &written))
    }

    /// **Group `group` over pass `pass`**: its weights acquired (kept, or the ring's next use), `before` (its edges: a
    /// modulation), then its form's steps in turn — a step of either unit the ANE's units of the shareable rows (each
    /// call after the ANE's call before it, and after the GPU's calls before it when the step before was the GPU's) and
    /// one GPU call past them; a step of the GPU's one call over the pass, every ANE call before it done; its weights
    /// released; the streams written swapped. `until`: the steps up to this one only (a pass whose last steps reach nothing:
    /// the keys of a prefix's last layer); a stream no step wrote is not swapped. `tapped` (the declaration's `tapped`): the
    /// ANE left, the GPU alone over every row, each step that taps in its tapped form — a check listening inside the group.
    public func run(_ group: String, _ pass: String, until: String? = nil, before: (() async throws -> Void)? = nil,
                    tapped: Bool = false) async throws {
        precondition(!tapped || declaration.tapped, "\(group): tapped, in a block not declared tapped")
        guard let g = groups[group], let p = passes[pass], let form = forms[g.form], p.serves(form) else { preconditionFailure("\(group) over \(pass)") }
        precondition(until == nil || form.steps.contains { $0.name == until }, "\(group): no step \(until!)")
        if group == declaration.cycle.first {
            let at = DispatchTime.now().uptimeNanoseconds
            timed(pass)
            let climbed = try await climb()
            var moved: Double?
            if case .inFlight = share, units > 0 { moved = try await rebalance() }
            cycled(until: at)
            begun = (at, units * unit, moved, climbed)
        }
        let t0 = DispatchTime.now().uptimeNanoseconds
        let weights = try stage.acquire(group)
        waited += Functions.seconds(since: t0)
        held = (g, weights)
        defer { held = nil }
        let shared = !tapped && form.shared && p.shareable != nil && units > 0, neural = shared ? units * unit : 0
        let ane = unit
        // The share is sized on the block's largest shareable pass: a smaller pass would give the ANE rows it has not.
        precondition(!shared || neural <= p.shareable!.count, "\(pass): the ANE's \(neural) rows past its \(p.shareable!.count) shareable")
        let tiles = shared ? (0..<units).map { Tiles.Tile(start: p.rows.lowerBound + $0 * ane, rows: ane) } : []
        let run = Self.steps(of: form, until: until)
        var written = Set<String>()
        // The ANE's calls, made before the pass, for the steps it runs: a step's views name the tensors it writes, whose
        // two buffers turn once the pass is over (`current`). A call beside the GPU's reads nothing of the loop's state.
        let made = tiles.map { u in
            Dictionary(uniqueKeysWithValues: run.map(\.element).filter { $0.unit == .either }.map { s in
                let f = Self.neural(g.form, s.name, ane)
                return (s.name, (function: functions[f]!, io: views(f, start: u.start, group: g, pass: p, step: s, call: [:], neural: true, written: &written)))
            })
        }
        nonisolated(unsafe) let calls = made
        var record: [Timed] = []
        var launched: [(step: String, index: Int, task: Task<Double, Error>)] = []
        func drain() async throws {
            for l in launched {
                let seconds = try await l.task.value
                record.append(Timed(kind: .neural, step: l.step, phase: Self.phase(form, l.index), rows: ane, seconds: seconds))
            }
            launched = []
        }
        do {
            try await before?()
            var afterGPU = true
            for (i, s) in run {
                if s.unit == .either && shared {
                    // What the ANE reads of the GPU's: every GPU call before, when the step before was the GPU's.
                    let gpu = afterGPU ? last : nil
                    for u in tiles.indices {
                        let previous = launched.last?.task, c = calls[u][s.name]!
                        let function = c.function
                        nonisolated(unsafe) let io = c.io
                        let task = Task { () async throws -> Double in
                            _ = try await previous?.value
                            _ = await gpu?.value
                            let t = DispatchTime.now().uptimeNanoseconds
                            try await Run.neural(function, inputs: io.inputs, outputs: io.outputs)
                            return Functions.seconds(since: t)
                        }
                        launched.append((s.name, i, task))
                    }
                } else if s.unit == .gpu {
                    try await drain()
                }
                for t in Self.calls(form, s, p, neural: neural, cut: cut, limit: p.limit ?? declaration.rows) {
                    let f = Self.gpu(g.form, s.name, t.rows, tapped: tapped && s.taps)
                    let timing = try await encode(f, views(f, start: t.start, group: g, pass: p, step: s, call: [:], neural: false, written: &written))
                    if shared, case .inFlight = share {
                        record.append(Timed(kind: s.unit == .either ? .past : .fixed, step: s.name, phase: Self.phase(form, i), rows: t.rows, timing: timing))
                    }
                }
                afterGPU = s.unit == .gpu
            }
            try await drain()
        } catch {
            // Nothing left running on the ANE's rows past the pass; the group's weights given back as a pass gives them,
            // so that the ring's order of use holds for a caller that goes on.
            for l in launched { _ = try? await l.task.value }
            if weights.use != nil { await settled() }
            stage.release(weights)
            throw error
        }
        if shared, case .inFlight = share { cycleRecords.append(record) }
        // A slot of the ring is read again once released: what reads it done first.
        if weights.use != nil { await settled() }
        stage.release(weights)
        for s in written { current[s] = 1 - current[s]! }
        await passed(group, pass)
    }

    /// **The stage's signal**, given once the second group in a row over a pass is done (a turn of the loop begins anew
    /// at its first group): each function of the pass called twice, every call done (`Stage.signal`).
    private func passed(_ group: String, _ name: String) async {
        if name != pass || group == declaration.cycle.first || group == declaration.once.first { (pass, inPass) = (name, 0) }
        inPass += 1
        if inPass == 2 { await signal() }
    }

    /// Every call done, the stage's signal given (`Stage.signal`).
    func signal() async {
        await settled()
        stage.signal()
    }

    /// **The block's pace**, at a cycle's first group over `pass`: the seconds between two starts over one pass (the pass
    /// that ran most: a pass run once, a prefix's, gives none), the first left out; their median kept on this Mac
    /// (`ThisMac.paces`) from the second on, again where it moves by 1 % or more. Taken at the unit kept only (`pace`): the
    /// starts of a trial's cycles are dropped when it settles, so that no period spans two units.
    private func timed(_ pass: String) {
        guard !paceBase.isEmpty else { return }
        starts[pass, default: []].append(DispatchTime.now().uptimeNanoseconds)
        let s = starts.values.max { $0.count < $1.count }!
        guard s.count >= 4, !pace.isEmpty else { return }
        let periods = zip(s.dropFirst(2), s.dropFirst()).map { Double($0 - $1) * 1e-9 }.sorted()
        let median = periods[periods.count / 2]
        if let kept, abs(median - kept) < 0.01 * kept { return }
        kept = median
        let (key, value) = (pace, median)
        try? ThisMac.update { $0.paces = ($0.paces ?? [:]).merging([key: value]) { $1 } }
    }

    /// **The calls climbed in flight**, at a turn's first group once a whole turn ran at this height: Core AI's share read
    /// now, the tallest height above whose own reading — or this one scaled by its rows (`Stage.estimates`) — holds under
    /// B beside what is kept (`Stage.rekey`) taken, its functions loaded (built and specialized where first met) and this
    /// one's let go, this height's reading kept and the stage's reading kept under the new one's from now on (`rekey`).
    /// A height the fuse stopped is never taken. The seconds its loading took; nil, nothing climbed.
    private func climb() async throws -> Double? {
        turns += 1
        guard turns == 2, height > 0 else { return nil }
        let share = max(stage.observe(), 0), here = heights[height]
        // The keys at the share of the ANE now (a move since the opening changed the functions).
        let keys = try heights.map { try key(units: units, unit: unit, cut: $0.cut) }
        let known = Stage.estimates(heights.indices.map { (keys[$0], heights[$0].rows) })
        for i in 0..<height where known[i] != .fused {
            let estimate: Int = if case .read(let b) = known[i] { b } else { Int(Double(share) * Double(heights[i].rows) / Double(max(here.rows, 1))) }
            guard stage.rekey(keys[i], expects: estimate - share) else { continue }
            (height, turns) = (i, 1)
            let t = DispatchTime.now().uptimeNanoseconds
            try await stage.loading { try await load() }
            return Functions.seconds(since: t)
        }
        return nil
    }

    /// The steps `run` takes, with their places in the form: all, or up to the one named `until`.
    static func steps(of form: Form, until: String?) -> [(offset: Int, element: Step)] {
        let all = Array(form.steps.enumerated())
        guard let until, let last = form.steps.firstIndex(where: { $0.name == until }) else { return all }
        return Array(all[...last])
    }

    /// **A call of a pass beside the ANE, timed**: the ANE's (a unit); the GPU's past the ANE's rows, in a phase beside
    /// it (`phase`); the GPU's of a `gpu` step.
    struct Timed {
        enum Kind { case neural, past, fixed }
        var kind: Kind
        var step: String
        var phase: Int
        var rows: Int
        var seconds = 0.0
        var timing: Timing? = nil
    }

    /// **The share in flight moved**, at a cycle's first group: the cycle before's calls timed, the units `predicted`
    /// to finish first taken if they gain 3 % at least, their rows kept on this Mac, the GPU's functions at the new rows
    /// loaded (built and specialized where first met) and the others let go. **The unit under trial** (`UnitTrial`) is
    /// told that cycle's seconds — its own work, what its start loaded left out — and answers: a cold cycle moves no share
    /// (its calls carry the specialization), a measured one moves to the unit next with the share it predicted best, in
    /// rows, rounded to the unit's; the last settles on the faster (`settle`).
    private func rebalance() async throws -> Double? {
        defer { cycleRecords = [] }
        var records: [[Timed]] = []
        for r in cycleRecords {
            var record = r
            for j in record.indices { if let t = record[j].timing { record[j].seconds = await t.seconds() } }
            records.append(record)
        }
        guard !records.isEmpty else { return nil }
        var sums = (ane: 0.0, beside: 0.0, alone: 0.0)
        for t in records.joined() {
            switch t.kind {
            case .neural: sums.ane += t.seconds
            case .past: sums.beside += t.seconds
            case .fixed: sums.alone += t.seconds
            }
        }
        lastCalls = Stages.Calls(ane: sums.ane, beside: sums.beside, alone: sums.alone)
        let a0 = units * unit
        func predicted(_ u: Int) -> Double { Self.predicted(records, measured: a0, rows: u * unit) }
        let (seconds, rows) = Self.bestShare(records, measured: a0, unit: unit, shareable: shareable), best = rows / unit
        let better = best != units && predicted(best) <= 0.97 * predicted(units)
        guard var t = trial else { return better ? try await move(to: unit, units: best) : nil }
        let cold = t.cold
        let next = t.record(UnitTrial.Best(seconds: seconds, rows: rows))
        trial = t
        switch next {
        case .stay:
            return cold || !better ? nil : try await move(to: unit, units: best)
        case .move(let to):
            // Refused under B (the guard of a share's move): the unit it runs at is the one kept.
            if let loaded = try await move(to: to, units: UnitTrial.units(rows: rows, of: to, shareable: shareable)) { return loaded }
            await settle(unit, best: t.best)
            return nil
        case .settle(let winner):
            // The winner at its own best share: the one its cycle predicted, a unit's multiple.
            let units = UnitTrial.units(rows: t.best[winner]!.rows, of: winner, shareable: shareable)
            var loaded: Double? = nil
            if winner != unit || units != self.units { loaded = try await move(to: winner, units: units) }
            await settle(unit, best: t.best)
            return loaded
        }
    }

    /// **A unit's best share predicted** from a cycle's calls (`predicted`, over every share of whole units of `unit`, the
    /// first of the fastest): its seconds and its ANE rows. What two units are compared by — never the raw cycle, run at
    /// the share the other unit's cycle set: the finer unit's whole advantage may be a share the taller cannot take (M1 Pro,
    /// 20/11: 2560 rows at 512, 3072 at 1024; the raw cycles, both at 3072, showed none).
    static func bestShare(_ records: [[Timed]], measured a0: Int, unit: Int, shareable: Int) -> (seconds: Double, rows: Int) {
        let k = (1..<max(shareable / unit, 2)).min { (predicted(records, measured: a0, rows: $0 * unit), $0) < (predicted(records, measured: a0, rows: $1 * unit), $1) }!
        return (predicted(records, measured: a0, rows: k * unit), k * unit)
    }

    /// **The ANE's share moved to `n` units of `target`** rows, if what the move loads holds under B: the functions it
    /// loads, their reading on this Mac — never read, the share now, scaled by the GPU's rows where the GPU takes more (a
    /// function's working memory is as tall as its call) —, holding under B beside what is kept (`Stage.rekey`); a set
    /// the fuse stopped never taken. Another unit: the surfaces the ANE reads copied into that unit's, the block runs at
    /// it from here. Its rows kept on this Mac. The seconds the loading took; nil, nothing moved.
    private func move(to target: Int, units n: Int) async throws -> Double? {
        let rows = n * target, moving = try key(units: n, unit: target, cut: cut), share = max(stage.observe(), 0)
        let estimate: Int
        switch Stage.estimates([(moving, 1)])[0] {
        case .fused: return nil
        case .read(let b): estimate = b
        default: estimate = Int(Double(share) * max(1, Double(shareable - rows) / Double(max(shareable - units * unit, 1))))
        }
        guard stage.rekey(moving, expects: estimate - share) else { return nil }
        if target != unit {
            await settled()
            for r in declaration.perRow where neuralRows.contains(r.name) {
                let bytes = (r.rows ?? declaration.rows) * r.rowBytes
                for name in r.next == nil ? [r.name] : [r.name + "#0", r.name + "#1"] {
                    memcpy(stage.pointer(name + Self.surface(target == home ? nil : target)), stage.pointer(name + Self.surface(unit == home ? nil : unit)), bytes)
                }
            }
        }
        (unit, total, units) = (target, shareable / target, n)
        let t = DispatchTime.now().uptimeNanoseconds
        try await stage.loading { try await load() }
        let loaded = Functions.seconds(since: t)
        told()
        let blockKey = key
        try? ThisMac.update { $0.flight = ($0.flight ?? [:]).merging([blockKey: rows]) { $1 } }
        return loaded
    }

    /// **The cycle begun last, told done at `end`** (`Stages.Event.cycle`): at the next cycle's first group, or at the
    /// block's close.
    private func cycled(until end: UInt64) {
        defer { lastCalls = nil }
        guard let b = begun else { return }
        begun = nil
        Stages.tell(.cycle(Stages.Cycle(block: declaration.name, unit: unit, rows: b.rows, shareable: total * unit,
                                        seconds: Double(end &- b.at) * 1e-9, moved: b.moved, climbed: b.climbed, calls: lastCalls)))
    }

    /// **A cycle's shared passes with `a` rows the ANE's, predicted from their calls at `a0`**, as `run` orders them: each
    /// phase lasts the longer of its two units' calls — the GPU's past the ANE's rows at their seconds a row, the ANE's
    /// at theirs (`neural` times them: another unit's cost a row); the `gpu` steps' calls, in turn. (A balance of the
    /// pass's sums, f = 1 / (1 + r), overlaps the ANE with every `gpu` step: 7 units of 256 at 1024² predicted 1.99 s, run
    /// in 2.60; this, 2.57 — measured 19/11 on the mini.)
    static func predicted(_ records: [[Timed]], measured a0: Int, rows a: Int, neural: Double = 1) -> Double {
        records.reduce(0) { sum, r in
            var steps: [String: (kind: Timed.Kind, phase: Int, seconds: Double, rows: Int)] = [:]
            for c in r {
                let k = "\(c.kind) \(c.step)"
                steps[k, default: (c.kind, c.phase, 0, 0)].seconds += c.seconds
                steps[k, default: (c.kind, c.phase, 0, 0)].rows += c.rows
            }
            var gpu: [Int: Double] = [:], ane: [Int: Double] = [:], serial = 0.0
            for s in steps.values {
                switch s.kind {
                case .neural: ane[s.phase, default: 0] += s.seconds * neural * Double(a) / Double(max(a0, 1))
                case .past: gpu[s.phase, default: 0] += s.seconds / Double(s.rows) * Double(max(s.rows - (a - a0), 0))
                case .fixed: serial += s.seconds
                }
            }
            return sum + serial + Set(gpu.keys).union(ane.keys).reduce(0) { $0 + max(gpu[$1] ?? 0, ane[$1] ?? 0) }
        }
    }

    /// **This Mac's key for the functions a share of `units` units and calls of `cut` rows at most load** (`Stage.key`):
    /// the edges' and the ANE's as the stage loaded them, the steps' built and named (not written).
    private func key(units: Int, unit u: Int, cut: Int?) throws -> String {
        let used = Set(declaration.groups.map(\.form)).sorted().map { forms[$0]! }
        // The ANE's functions loaded now: both units' while the unit is under trial.
        let kept = stage.plan.declaration.functions.filter { functions[$0.key] != nil && ($0.key.hasPrefix("edge-") || $0.key.hasPrefix("ane-")) }.values.map { ($0.asset, $0.unit) }
        let steps = try Self.steps(declaration, used, neural: units * u, cut: cut).values.map { spec in
            (Functions.name(try spec.form.module(spec.step, Build(rows: spec.rows, codes: codes, bfloat16: stage.plan.bfloat16, tapped: spec.tapped)), prefix: "block"),
             ComputeUnitKind.gpu)
        }
        return Stage.key(kept + steps)
    }

    /// **The GPU's functions of the share now**, each built, named and loaded where it is not, the others let go once
    /// their calls are done. Their shares are no larger than those P counts, at the tallest calls.
    private func load() async throws {
        let used = Set(declaration.groups.map(\.form)).sorted().map { forms[$0]! }
        let need = Self.steps(declaration, used, neural: units * unit, cut: cut)
        for (name, spec) in need.sorted(by: { $0.key < $1.key }) where functions[name] == nil {
            let m = try spec.form.module(spec.step, Build(rows: spec.rows, codes: codes, bfloat16: stage.plan.bfloat16, tapped: spec.tapped))
            let asset = try Functions.named(m, prefix: "block")
            signatures[name] = Signature(m)
            functions[name] = try await Functions.load(try Functions.asset(asset) { m }, unit: .gpu)
        }
        let gone = functions.keys.filter { !$0.hasPrefix("edge-") && !$0.hasPrefix("ane-") && need[$0] == nil }
        guard !gone.isEmpty else { return }
        await settled()
        for n in gone { functions[n] = nil; marks[n] = nil }
        stage.drop(gone)
    }

    /// **The unit decided**, at the trial's end: the unit the block runs at kept on this Mac for good, the other's functions
    /// let go (their surfaces stay, counted in P), the cycles' starts dropped but this one's so that the pace is taken
    /// at the unit kept alone (`pace`). `best`: each unit's best share predicted.
    private func settle(_ winner: Int, best: [Int: UnitTrial.Best]) async {
        trial = nil
        starts = starts.mapValues { Array($0.suffix(1)) }
        kept = nil
        let gone = functions.keys.filter { $0.hasPrefix("ane-") && !$0.hasSuffix("@\(winner)") }
        if !gone.isEmpty {
            await settled()
            for n in gone { functions[n] = nil; marks[n] = nil }
            stage.drop(gone)
        }
        keep(winner, best: best)
    }

    /// The unit kept on this Mac (`ThisMac.neuralUnits`) and said (`Stages.Event.weighed`).
    private func keep(_ unit: Int, best: [Int: UnitTrial.Best]) {
        let key = key
        try? ThisMac.update { $0.neuralUnits = ($0.neuralUnits ?? [:]).merging([key: unit]) { $1 } }
        Stages.tell(.weighed(Stages.Weighed(block: declaration.name, seconds: best.mapValues(\.seconds), rows: best.mapValues(\.rows), kept: unit)))
    }

    // MARK: - What the CPU reads and writes

    /// The bytes of tensor `name`, every call done.
    public func pointer(_ name: String) async -> UnsafeMutableRawPointer {
        await settled()
        return stage.pointer(name)
    }

    /// The bytes of per-row tensor `name` from row `row` (a stream's current buffer, or its other), every call done.
    public func rows(_ name: String, from row: Int = 0, next: Bool = false) async -> UnsafeMutableRawPointer {
        await settled()
        return stage.pointer(tensor(name, next: next)) + row * perRow[name]!.rowBytes
    }

    /// The bytes of a row of per-row tensor `name`.
    public func rowBytes(_ name: String) -> Int { perRow[name]!.rowBytes }

    /// **Rows `range` of per-row tensor `name` copied into per-row tensor `to` from its row `row`** (a stream's current
    /// buffer each, `to`'s other under `next`), every call done: rows a pass computed set where another pass reads them.
    public func copy(_ name: String, rows range: Range<Int>, to: String, at row: Int, next: Bool = false) async {
        let (from, into) = Self.copied(perRow[name]!, rows: range, to: perRow[to]!, at: row, block: declaration.rows)
        await settled()
        memcpy(stage.pointer(tensor(to, next: next)) + into.lowerBound, stage.pointer(tensor(name, next: false)) + from.lowerBound, from.count)
    }

    /// The bytes a `copy` reads and writes: rows of the same width and type, within both tensors' rows.
    static func copied(_ a: Rows, rows range: Range<Int>, to b: Rows, at row: Int, block: Int) -> (from: Range<Int>, into: Range<Int>) {
        precondition(a.width == b.width && a.type == b.type, "\(a.name) to \(b.name): rows of another width or type")
        precondition(range.lowerBound >= 0 && range.upperBound <= a.rows ?? block && row >= 0 && row + range.count <= b.rows ?? block,
                     "\(a.name) \(range) to \(b.name) at \(row): past their rows")
        let n = range.count * a.rowBytes
        return (range.lowerBound * a.rowBytes ..< range.lowerBound * a.rowBytes + n, row * b.rowBytes ..< row * b.rowBytes + n)
    }

    /// **Rows `range` of per-row tensor `name` as true values**, in f32: its elements widened, times its factor's.
    public func values(_ name: String, rows range: Range<Int>) async -> [Float] {
        var out = [Float](repeating: 0, count: range.count * perRow[name]!.width)
        await values(name, rows: range, into: &out)
        return out
    }

    /// The same into `out`, `range.count · width` values: storage its client made before the stage was planned, so that
    /// its floor counts it (made while the stage lives, it is read as Core AI's share).
    public func values(_ name: String, rows range: Range<Int>, into out: inout [Float]) async {
        precondition(out.count == range.count * perRow[name]!.width, "\(name): \(range.count) rows into \(out.count) values")
        let (base, factor) = await sources(name, range)
        out.withUnsafeMutableBufferPointer { widen(name, base, factor, into: $0) }
    }

    /// The same into a buffer (a check's, part of one).
    public func values(_ name: String, rows range: Range<Int>, into out: UnsafeMutableBufferPointer<Float>) async {
        precondition(out.count == range.count * perRow[name]!.width, "\(name): \(range.count) rows into \(out.count) values")
        let (base, factor) = await sources(name, range)
        widen(name, base, factor, into: out)
    }

    private func sources(_ name: String, _ range: Range<Int>) async -> (UnsafeMutableRawPointer, UnsafeMutableRawPointer?) {
        let base = await rows(name, from: range.lowerBound)
        guard let f = perRow[name]!.factor else { return (base, nil) }
        return (base, await rows(f, from: range.lowerBound))
    }

    private func widen(_ name: String, _ base: UnsafeMutableRawPointer, _ factor: UnsafeMutableRawPointer?, into out: UnsafeMutableBufferPointer<Float>) {
        let r = perRow[name]!
        func value(_ p: UnsafeMutableRawPointer, _ type: NDArray.ScalarType, _ i: Int) -> Float {
            switch type {
            case .float32: p.assumingMemoryBound(to: Float.self)[i]
            case .float16: Float(p.assumingMemoryBound(to: Float16.self)[i])
            default: Float(bitPattern: UInt32(p.assumingMemoryBound(to: UInt16.self)[i]) << 16)
            }
        }
        let ft = r.factor.map { perRow[$0]!.type }
        for i in 0..<out.count { out[i] = value(base, r.type, i) * (factor.map { value($0, ft!, i / r.width) } ?? 1) }
    }

    /// Everything let go, and the footprint waited for (`Stage.close`): the functions the block loaded itself first.
    @discardableResult
    public func close() -> Stage.Settled {
        cycled(until: DispatchTime.now().uptimeNanoseconds)
        functions = [:]
        return stage.close()
    }
}
