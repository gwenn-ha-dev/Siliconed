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
/// no remainder takes a call of its own (measured 19/11 on the M5 Pro, a transformer of 6 billion bf16 weights on the
/// GPU alone at 4128 rows: the 64-row remainder +0.13 s a cycle, tiles of 1024 beside the ANE +5 to +8 % on the GPU
/// against one call). A height met first is built and specialized then, once a Mac. Every height is planned — one call
/// a pass, then equal calls of 4096 rows at most, 2048, 1024 — and the stage takes the tallest whose P and what Core AI
/// was read to hold under its functions on this Mac, or a bound from what it held under their kin at another height or
/// size (`extent`, `Stage.estimates`), hold under B (`Stage.choose`); nothing of their kin read, the soberest whose
/// socle holds.
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
    // MARK: - The block

    public let declaration: Declaration
    public let stage: Stage
    /// The tallest call now: nil, a call a step and a pass (`heights`, `climb`).
    public var cut: Int? { heights[height].cut }
    /// The heights the calls may take, the tallest first, and the one taken; the turns of the cycle begun at it.
    let heights: [Height]
    var height: Int
    var turns = 0
    /// The ANE's unit (rows), the shareable units, the ANE's now (0: the GPU alone).
    public internal(set) var unit: Int, total: Int
    public internal(set) var units: Int
    /// The shareable rows; the unit the surfaces of the per-row tensors the ANE reads (`neuralRows`) are named for, the
    /// others' beside them named `name@<unit>` (a trial's second unit: `places`).
    let shareable: Int, home: Int
    let neuralRows: Set<String>
    /// The unit under trial (`UnitTrial`; nil: decided).
    var trial: UnitTrial?
    /// The buffers a GPU call has read or written (`encode`).
    var seen = Set<ObjectIdentifier>()
    /// The cycle under way, told once done (`Stages.Event.cycle`): its start, the ANE's rows over it, what its start loaded;
    /// the calls of the cycle before, summed once timed (`rebalance`).
    var begun: (at: UInt64, rows: Int, moved: Double?, climbed: Double?)?
    var lastCalls: Stages.Calls?
    /// What the loop waited for the ring's weights, in seconds; what building and naming the functions took at open.
    public internal(set) var waited = 0.0
    public let named: Double

    let share: Share
    /// The block's key on this Mac: its ANE's steps and its shareable rows (`ThisMac.flight`, `ThisMac.neuralUnits`).
    let key: String
    /// Its pace key (`paceKey`, `ThisMac.paces`; empty: not timed) and whether it carries the unit (`pace`), each pass's
    /// cycle starts, the pace last kept.
    let paceBase: String, paceByUnit: Bool
    var starts: [String: [UInt64]] = [:], kept: Double?
    /// The pace key now: none while the unit is under trial — a cycle at another unit is no pace of the unit kept —, else
    /// the key at the unit the block runs at.
    var pace: String { paceBase.isEmpty || trial != nil ? "" : paceBase + (paceByUnit ? Self.byUnit(unit) : "") }
    var functions: [String: InferenceFunction]
    var signatures: [String: Signature]
    let groups: [String: Group]
    let forms: [String: Form], passes: [String: Pass], perRow: [String: Rows]
    let resolver: Resolver
    var current: [String: Int] = [:]
    var held: (group: Group, weights: Stage.Held)?
    /// **The ANE's copies of the codes** (`copy`): the codes its functions read, by name, one copy each; the turn of the
    /// loop whose first codes are copied or under way, and which; what each run reads in each of its stretches of the
    /// ANE (`stretches`, by form, pass and last step); the serial queue that makes them.
    let copies: Set<String>
    var ahead: (turn: Int, names: Set<String>)?
    var stretches: [String: [Set<String>]] = [:]
    let copier = DispatchQueue(label: "block.copies", qos: .userInitiated)
    /// The last call of each function on the stream, and the last of all: each a task done when it is.
    var marks: [String: Task<UInt64, Never>] = [:], last: Task<UInt64, Never>?
    var cycleRecords: [[Timed]] = []
    /// The pass the last groups ran over, and how many in a row (`signal`).
    var pass: String?, inPass = 0

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
    /// A function's name less its rows: a step's at every height one name (`gpu`), an ANE's or an edge's as it is.
    static func unsized(_ name: String) -> String {
        guard !name.hasPrefix("edge-") && !name.hasPrefix("ane-"), let dash = name.lastIndex(of: "-"),
              name[name.index(after: dash)...].allSatisfy(\.isNumber) else { return name }
        return String(name[..<dash])
    }

    // MARK: - The calls' rows

    /// **The rows a call covers** (`calls`): its first row and how many, each height a function's fixed shape.
    struct Tile: Equatable, Sendable {
        let start: Int
        let rows: Int
    }

    /// **The calls over `rows`**: one, its height the rows rounded up to 64; under `cut`, as few calls of equal height at
    /// most `cut` as cover them, each rounded up to 64. No call passes `limit` (the per-row tensors' rows): the last is
    /// shortened to it.
    static func calls(_ rows: Range<Int>, cut: Int?, limit: Int) -> [Tile] {
        guard !rows.isEmpty else { return [] }
        let k = cut.map { (rows.count + $0 - 1) / $0 } ?? 1
        let height = ((rows.count + k - 1) / k + 63) / 64 * 64
        return (0..<k).map { i in
            let start = rows.lowerBound + i * height
            return Tile(start: start, rows: min(height, limit - start))
        }.filter { $0.start < rows.upperBound }
    }

    /// **The GPU's calls of step `s` of form `f` over pass `p`, the ANE's first `neural` rows aside**: an `either` step's
    /// past them, a `gpu` step's over the pass. (A `gpu` step in two calls beside the ANE — the rows past the ANE's
    /// while it ends, then its rows, or its rows first so that it starts its next step — let the ANE take more rows but
    /// cost the GPU more than it gave: a transformer of 6 billion weights at 4128 rows, 2.37 s a cycle at the best share
    /// against 2.31, the attention in two calls 0.49 s against 0.41; measured 19/11 on the mini.)
    static func calls(_ f: Form, _ s: Step, _ p: Pass, neural: Int, cut: Int?, limit: Int) -> [Tile] {
        let from = s.unit == .either && f.shared && p.shareable != nil ? p.rows.lowerBound + neural : p.rows.lowerBound
        return calls(from..<p.rows.upperBound, cut: cut, limit: limit)
    }

    /// The phase of an `either` step beside the ANE: the steps between two `gpu` steps, counted by the `gpu` steps before
    /// it. A `gpu` step waits for both units.
    static func phase(_ f: Form, _ i: Int) -> Int { f.steps[..<i].filter { $0.unit == .gpu }.count }

    /// A GPU call's function: its form's step at its rows, tapped or not.
    typealias Spec = (form: Form, step: String, rows: Int, tapped: Bool)

    /// **The GPU's functions of the steps at `neural` rows of the ANE's**: every form used over every pass, by name.
    static func steps(_ d: Declaration, _ used: [Form], neural: Int, cut: Int?) -> [String: Spec] {
        var all: [String: Spec] = [:]
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

    /// The module of a GPU call's function.
    static func module(_ spec: Spec) throws -> Aicode.Module {
        try spec.form.module(spec.step, Build(rows: spec.rows, tapped: spec.tapped))
    }

    /// The names of per-row tensor `r`'s buffers: its own, or a stream's two.
    static func buffers(_ r: Rows) -> [String] { r.next == nil ? [r.name] : [r.name + "#0", r.name + "#1"] }

    /// **What a stage's functions hold, by names the same at any height and size** (`Stage.Extent`): each function by
    /// its name less its rows (a step's calls at every height one name), the elements of its ops' results — its
    /// intermediates and its outputs, the weights it reads aside; a result of sizes unknown, inside a composite
    /// (`Aicode.dynamic`), counts none — summed over the functions of that name, and of their largest results; their
    /// kin, each name with its unit and what it computes apart from its shapes (its ops and their results' elements).
    static func extent(_ functions: [String: Stage.Function], inFlight: Int) throws -> Stage.Extent {
        var lines = Set<String>(), elements: [String: Int] = [:]
        for (name, f) in functions {
            let walked = walk(try f.build().body)
            let base = unsized(name)
            lines.insert("\(base) \(f.unit == .neuralEngine ? "ane" : "gpu") \(walked.ops)")
            elements["\(base) results", default: 0] += walked.results
            elements["\(base) largest", default: 0] += walked.largest
        }
        return Stage.Extent(Array(lines), inFlight: inFlight, elements: elements)
    }

    /// **What a function computes apart from its shapes**: the digest of its ops and of their results' element types, in
    /// order, composites' insides included — what a kin names (`Stage.Extent`): under the same name, other ops or other
    /// element types are another kin; the same ops at other sizes — another height, another size, intermediates larger
    /// or smaller — the same kin, a reading of one bounding the other by the ratio of their results' elements
    /// (`Stage.Extent.ratio(over:)`) —; and its results' elements, summed and the largest.
    static func walk(_ body: Aicode.Block) -> (ops: String, results: Int, largest: Int) {
        var ops: [String] = [], results = 0, largest = 0
        func visit(_ b: Aicode.Block) {
            for op in b.ops {
                ops.append(op.name)
                for t in op.results where t.kind == 2 {
                    // A tensor type: its rank, its sizes, its element (`Aicode.tensor`); read here as it lies, since
                    // a composite's types may carry sizes unknown (`Aicode.dynamic`) and elements `shape` refuses.
                    let rank = if case .int(let r) = t.fields[0] { Int(r) } else { 0 }
                    let sizes = t.fields.dropFirst().prefix(rank).map { if case .signed(let d) = $0 { Int(d) } else { -1 } }
                    if case .type(let e)? = t.fields.dropFirst(rank + 1).first { ops.append("\(e.kind)") }
                    guard sizes.allSatisfy({ $0 >= 0 }) else { continue }
                    let n = sizes.reduce(1, *)
                    (results, largest) = (results + n, max(largest, n))
                }
                if let inner = op.body { visit(inner) }
            }
        }
        visit(body)
        return (digest(Data(ops.joined(separator: " ").utf8)), results, largest)
    }

    /// A hash's first 8 bytes in hexadecimal: what a key on this Mac names (`paceKey`, `key`).
    static func digest(_ bytes: Data) -> String { Data(SHA256.hash(data: bytes)).prefix(8).map { String(format: "%02x", $0) }.joined() }

    init(_ d: Declaration, stage: Stage, heights: [Height], height: Int, unit: Int, total: Int, units: Int, share: Share, key: String,
                 pace: (base: String, byUnit: Bool), trial: UnitTrial?, shareable: Int, neuralRows: Set<String>, copies: Set<String>, named: Double,
                 signatures: [String: Signature], forms: [String: Form], groups: [String: Group], resolver: Resolver) {
        (declaration, self.stage, self.heights, self.height, self.unit, self.total, self.units, self.share) = (d, stage, heights, height, unit, total, units, share)
        (self.key, self.paceBase, self.paceByUnit, self.trial, self.shareable, self.home, self.neuralRows) = (key, pace.base, pace.byUnit, trial, shareable, unit, neuralRows)
        (self.copies, self.named, self.signatures, self.forms, self.groups, self.resolver) = (copies, named, signatures, forms, groups, resolver)
        functions = stage.functions
        passes = Dictionary(uniqueKeysWithValues: d.passes.map { ($0.name, $0) })
        perRow = resolver.perRow
        for r in d.perRow where r.next != nil { current[r.name] = 0 }
    }

    /// A block its owner lets go unclosed (an error past its opening: a low rank refused) is closed all the same; one
    /// closed already is not waited for again (`Stage.close` gives what its first close waited for).
    deinit { close() }
}
