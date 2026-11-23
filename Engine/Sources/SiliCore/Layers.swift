import CoreAI
import Foundation
import Metal

/// **Layers: a stack of a model's layers over one palier of rows, through the ring** — an encoder, a language model
/// that writes, a vision tower. The client declares its per-row tensors (the stream, its factors, the RoPE tables, a
/// cache's rows), its other tensors and its residents, its layers in the loop's order (a group of weights each, and the
/// kind of layer that computes with it), each kind's function at the palier's rows, and what runs apart from the loop
/// (`edges`: an embedding, a final norm, a merger, a head, a layer over one row and a cache). It writes the rows the first layer reads
/// and reads what the last wrote; what its layers compute is its own.
///
/// The engine makes it a `Block` — a form a kind of layer, its one step on the GPU over the palier as one call, one
/// pass — and so builds every function, names its asset by its bytes, plans the stage under B (the layers that lay out as
/// the first ride the ring from the start, in the loop's order; those kept are decided past its signal, `Stage.signal`),
/// binds each argument by its name (`Block`'s rules: a weight of the layer or a resident, a per-row tensor, the layer's
/// own per-row tensor `<layer>.<name>` — a cache —, a tensor), and puts each call on the stream as soon as the same
/// function's call before is done.
///
///     forward     every layer in turn over the palier (`run`), the client's own work between two (a tap, an addition)
///     through     one row through every layer: an edge over each layer's weights in the loop's order, put on the stream
///                 unwaited, its stream handed from call to call as Core AI's values (`through`), one wait at the end
///
/// Why `Block` serves, and what this adds: a layer is a block's group over a single call, the GPU alone (an encoder's
/// palier is a few hundred rows, its attention causal over all of them). What a block's pass has no use for is a row's
/// chain through the layers — one function called once a layer, every call put on the stream before any is done: each
/// call waited for cost ~2 ms of submission and of a GPU gone idle, 52 a row (18/11, a language model of 36 layers
/// writing).
public final class Layers: @unchecked Sendable {
    public struct Declaration: Sendable {
        public var name: String
        /// The palier: what every layer's function computes over, and each per-row tensor's rows unless it says otherwise.
        public var rows: Int
        public var perRow: [Block.Rows] = []
        public var tensors: [Stage.Tensor] = []
        public var residents: [Weights.Entry] = []
        /// The layers in the loop's order, each a group of weights and its kind (`Block.Group.form`).
        public var layers: [Block.Group] = []
        /// Each kind's function at the palier's rows (`Block.Build.rows`).
        public var kinds: [String: @Sendable (Block.Build) throws -> Aicode.Module] = [:]
        public var edges: [String: @Sendable (Block.Build) throws -> Aicode.Module] = [:]
        /// The layers read again, in the same order, for as long as the stage lives (writing: every row written reads
        /// every layer): the ring's cycle; otherwise each is read once.
        public var again = false
        public init(_ name: String, rows: Int) { self.name = name; self.rows = rows }
    }

    public let declaration: Declaration
    public let block: Block
    public var stage: Stage { block.stage }

    /// The pass every layer runs over: the palier's rows, one call.
    static let pass = "palier"

    /// **The block of `d`**: a form a kind (one step, the GPU's, at the palier's rows), the layers its groups in the loop's
    /// order, the first's layout the ring's slots.
    static func block(_ d: Declaration) -> Block.Declaration {
        var b = Block.Declaration(d.name, rows: d.rows)
        (b.perRow, b.tensors, b.residents, b.groups, b.edges) = (d.perRow, d.tensors, d.residents, d.layers, d.edges)
        let order = d.layers.map(\.name)
        if d.again { b.cycle = order } else { b.once = order }
        b.slot = order.first
        b.forms = d.kinds.sorted { $0.key < $1.key }.map { kind, build in
            Block.Form(kind, steps: [.init("layer", on: .gpu)]) { _, rows in try build(rows) }
        }
        b.passes = [Block.Pass(pass, rows: 0..<d.rows)]
        // A layer computes across every row of the palier (its attention, its causal conv): one call, never cut.
        b.cuts = false
        return b
    }

    /// **The layers of `d` under `budget`**, opened (`Block.open`, the GPU alone).
    public static func open(_ device: MTLDevice, _ d: Declaration, files: [SafeTensors], budget: Int) async throws -> Layers {
        Layers(d, try await Block.open(device, block(d), files: files, budget: budget, share: .gpu))
    }

    private init(_ d: Declaration, _ block: Block) { (declaration, self.block) = (d, block) }

    // MARK: - The loop

    /// Layer `name` over the palier: its weights acquired (kept, or the ring's next), its call put on the stream, the
    /// stream's buffers swapped.
    public func run(_ name: String) async throws { try await block.run(name, Self.pass) }

    /// **Every layer in the loop's order**, `before` and `after` each (its index): what the CPU reads or adds there.
    public func forward(before: (Int) async throws -> Void = { _ in }, after: (Int) async throws -> Void = { _ in }) async throws {
        for (i, layer) in declaration.layers.enumerated() {
            try await before(i)
            try await run(layer.name)
            try await after(i)
        }
    }

    /// Edge `name`'s call, its per-row arguments from row `row` (`Block.call`).
    public func call(_ name: String, from row: Int = 0, _ bind: [String: Block.Bind] = [:]) async throws {
        try await block.call(name, from: row, bind)
    }

    /// **One row through every layer**: edge `edge` called over each layer's weights in the loop's order, every call put
    /// on the stream unwaited, its per-row arguments from row `row` (a cache's row written). The first call's stream
    /// bound by `entry`; each next one's the call before's outputs a stream names `next`, handed on as Core AI's values —
    /// a call reading, as a view of our buffer, what the call before writes there may read it before it is written
    /// (`Run.encode`) —; `bind` a layer's own. A layer read into the ring waits for its call before its slot is given
    /// back. Returns the last call's stream, as the inputs it would hand the next (`chain`, then `finish`).
    public func through(_ edge: String, row: Int, entry: [String: Block.Bind], bind: (String) -> [String: Block.Bind] = { _ in [:] }) async throws
        -> [String: InferenceFunction.AsyncValue] {
        let input = Dictionary(uniqueKeysWithValues: declaration.perRow.compactMap { r in r.next.map { ($0, r.name) } })
        var stream: [String: InferenceFunction.AsyncValue] = [:]
        for (i, layer) in declaration.layers.enumerated() {
            // A row's second layer done: what Core AI holds of a row's calls read (`Block.signal`), at every row as a pass
            // gives it at every turn, until the stage keeps. Functions this Mac never met keep from the cycle's third turn
            // (`Stage.signal`); given on the first row alone, nothing was ever kept and every row read every layer
            // through the ring (20/11, the mini, 36 bf16 layers of 8 billion parameters: 1.5 rows/s against 10.6). Each signal
            // waits for the row's calls so far: two or three rows pay it.
            if i == 2 && !stage.keeps { await block.signal() }
            let weights = try stage.acquire(layer.name)
            block.held = (block.groups[layer.name]!, weights)
            defer { block.held = nil }
            let given = stream.isEmpty ? entry.merging(bind(layer.name)) { a, _ in a } : bind(layer.name)
            let out = try block.chain(edge, from: row, given, values: stream, carry: Set(input.keys))
            stream = Dictionary(uniqueKeysWithValues: out.map { (input[$0.key]!, $0.value) })
            if weights.use != nil { await block.settled(); stage.release(weights) }
        }
        return stream
    }

    /// **Edge `edge`'s call put on the stream unwaited**, `values` its inputs as Core AI's values (`through`'s), the others
    /// bound (`bind`, then `Block`'s rules); every output held by a view.
    public func chain(_ edge: String, bind: [String: Block.Bind] = [:], values: [String: InferenceFunction.AsyncValue] = [:]) throws {
        let left = try block.chain(edge, from: 0, bind, values: values, carry: [])
        precondition(left.isEmpty, "\(edge): outputs no view holds, \(left.keys.sorted())")
    }

    /// Every call put on the stream, done.
    public func finish() async { await block.settled() }

    // MARK: - What the CPU reads and writes

    /// The bytes of tensor `name`, every call done.
    public func pointer(_ name: String) async -> UnsafeMutableRawPointer { await block.pointer(name) }

    /// The bytes of per-row tensor `name` from row `row` (a stream's current buffer), every call done.
    public func rows(_ name: String, from row: Int = 0) async -> UnsafeMutableRawPointer { await block.rows(name, from: row) }

    /// Rows `range` of per-row tensor `name` as true values, in f32 (`Block.values`).
    public func values(_ name: String, rows range: Range<Int>) async -> [Float] { await block.values(name, rows: range) }
    /// The same into `out`, made before the stage (`Block.values(_:rows:into:)`).
    public func values(_ name: String, rows range: Range<Int>, into out: inout [Float]) async {
        await block.values(name, rows: range, into: &out)
    }

    /// Everything let go, and the footprint waited for (`Stage.close`).
    @discardableResult
    public func close() -> Stage.Settled { block.close() }
}

extension Block {
    /// **Edge `name`'s call put on the stream, not waited for** (`Layers.through`): `values` its inputs as Core AI's values,
    /// the others bound from row `row`; the outputs named in `carry` held by no view, returned as values. Not ordered by
    /// the function's last call (`encode`): a row's calls of one function follow each other on the one stream, each
    /// reading the values the one before returns, and the CPU reads nothing of them before `settled`. Its outputs off a
    /// page (a cache's row) are read by the next row's calls alone, after that wait.
    func chain(_ name: String, from row: Int, _ bind: [String: Bind], values: [String: InferenceFunction.AsyncValue],
               carry: Set<String>) throws -> [String: InferenceFunction.AsyncValue] {
        let f = Self.edge(name)
        var written = Set<String>()
        let io = views(f, start: row, group: held?.group, pass: nil, step: nil, call: bind, neural: false,
                       skip: carry.union(values.keys), written: &written)
        var given = values
        for (name, v) in io.inputs {
            let key = ValueKey(buffer: ObjectIdentifier(v.buffer), offset: v.offset, type: v.type, shape: v.shape, strides: v.strides)
            if let value = self.values[key] { given[name] = value; continue }
            let value = Run.value(v)
            self.values[key] = value
            given[name] = value
        }
        let left = try Run.encode(functions[f]!, stage.stream, inputs: [:], chained: given, outputs: io.outputs)
        nonisolated(unsafe) let stream = stage.stream!
        let done = Task { await stream.currentWorkCompleted(); return DispatchTime.now().uptimeNanoseconds }
        (marks[f], last) = (done, done)
        return left
    }
}
