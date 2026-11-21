import CryptoKit
import Foundation

/// A Core AI asset written by us: MLIR bytecode (version 6) in Core AI's stable serialization dialect, `aicode`,
/// which Core AI's loader turns back into its `coreai` ops and then places, fuses and tiles. Recopied from SiliCore
/// (`CoreAIAsset.swift`, aec8f3c) without its graph: a function is written op by op on a `Block`.
///
///     asset.aimodel/ main.mlirb   the bytecode below          main.hash   its SHA-256
///                    metadata.json  {assetVersion "2.0", producer, creationDate}
///
///     main.mlirb   "ML\xEFR" · 6 · producer · sections  dialect · offsets · attributes+types · IR · resources · strings · properties
///
///     serialization.asset ⊃ dataflow.graph_v1 (names, function type, symbol)
///                             ⊃ broadcasting_batch_matmul_v1 ⊃ { primitive.batch_matmul_v1 · dataflow.yield_v1 }
///                             ⊃ silu_v1 ⊃ { constants · broadcasting_sub_v1 ⊃ {…} · primitive.exp_v1 · … · yield }
///                             ⊃ dataflow.output_v1
///
/// What a port by analogy would miss (SiliCore's findings, judged there against coreai-torch's bytes):
///   - **`aicode` is not the `coreai` dialect printed**: every attribute and type is custom-encoded (a kind, then
///     fields that name strings, attributes and types by their index), and a composite op carries its own
///     decomposition in an isolated region — down to the reshape and `broadcast_in_dims` a broadcast needs —
///     so that a runtime that does not know the composite can still run it. The loader keeps the composite;
///   - **the numbering is MLIR's** (`IRNumbering`): entries are met in pre-order (an op's name, result types,
///     properties, location; a block's arguments; isolated regions from a stack, last pushed first), then
///     sorted by use count, stably. Strings are numbered as they are written: the dialect, the op names, then
///     the entries in their order;
///   - **a property list is not a dictionary**: the asset holds an optional date (an index with a presence bit),
///     the graph its arguments' names, a zero, then three values each preceded by the attribute `false`;
///     constants hold their dense data without a type (the op's result gives it), splat or not;
///   - **a constant `Module.constant` takes of 256 elements or more is not in the IR**: it is a blob of the resource
///     section, aligned to 16 from the file's start (so a reader can map it in place), named by a key of ours. A
///     composite's own constants (`Block.constant`: a rotation's matrix) stay dense in the IR, as SiliCore writes them;
///   - **the dialect's "version" is `1 · 2 · the count of op names`**, and **the asset's date is the latest any op
///     needs**: each op says its own (`Op.since`);
///   - **the weights are arguments, never constants of the asset**: an asset is the same for any weights, which
///     we bind at run time as views of the arena. The constants it holds are its own (a rotation's matrix, the
///     scalars of a composite).

// MARK: - The bytecode's numbers

/// MLIR's prefix varint: the trailing zeros of the first byte, plus one, are its length; nine bytes past 2⁵⁶.
struct BytecodeEmitter {
    var bytes: [UInt8] = []

    mutating func byte(_ b: UInt8) { bytes.append(b) }
    mutating func append(_ b: [UInt8]) { bytes += b }

    mutating func varint(_ v: UInt64) {
        for n in 1...8 where v < UInt64(1) << UInt64(7 * n) {
            let encoded = (v << UInt64(n)) | (UInt64(1) << UInt64(n - 1))
            for i in 0..<n { bytes.append(UInt8(truncatingIfNeeded: encoded >> UInt64(8 * i))) }
            return
        }
        bytes.append(0)
        for i in 0..<8 { bytes.append(UInt8(truncatingIfNeeded: v >> UInt64(8 * i))) }
    }
    mutating func varint(_ v: Int) { varint(UInt64(v)) }
    /// Zigzag, as MLIR's `emitSignedVarInt`.
    mutating func signed(_ v: Int64) { varint(UInt64(bitPattern: (v << 1) ^ (v >> 63))) }

    mutating func section(_ id: UInt8, _ contents: [UInt8]) {
        byte(id)
        varint(contents.count)
        append(contents)
    }
}

// MARK: - aicode's entries

public enum Aicode {
    /// One attribute or type, as `aicode` encodes it: its kind, then fields.
    public struct Entry: Hashable, Sendable {
        public var kind: UInt64
        public var fields: [Field]
        public init(_ kind: UInt64, _ fields: [Field] = []) { self.kind = kind; self.fields = fields }
    }

    public indirect enum Field: Hashable, Sendable {
        case int(UInt64)
        case signed(Int64)
        case byte(UInt8)
        case string(String)
        case attribute(Entry)
        case type(Entry)
        /// A size, then the bytes.
        case blob([UInt8])
        /// A blob of the resource section, by its key.
        case resource(String)
    }

    /// One field of an op's properties.
    public enum Property: Hashable, Sendable {
        case int(UInt64)
        case attribute(Entry)
        /// An index with a presence bit: 0 when absent.
        case optional(Entry?)
    }

    // MARK: Types

    public enum Element: UInt64, Sendable { case bool = 4, f32 = 6, f16 = 7, bf16 = 8, si32 = 18, si8 = 20, si4 = 21, ui32 = 24 }

    public static func element(_ e: Element) -> Entry { Entry(e.rawValue) }
    public static func tensor(_ shape: [Int], _ e: Element) -> Entry {
        Entry(2, [.int(UInt64(shape.count))] + shape.map { .signed(Int64($0)) } + [.type(element(e))])
    }
    /// Results first, then inputs.
    public static func function(_ inputs: [Entry], _ results: [Entry]) -> Entry {
        Entry(30, [.int(UInt64(results.count))] + results.map { .type($0) } + [.int(UInt64(inputs.count))] + inputs.map { .type($0) })
    }
    /// The input's rank, sizes unknown (`?`: the smallest signed 64-bit integer).
    static func dynamic(_ rank: Int, _ e: Element) -> Entry { tensor(Array(repeating: Int.min, count: rank), e) }
    /// A tensor of a kind of element that has no `Element`: the 8-byte integer of a mean's count.
    static func kindTensor(_ shape: [Int], _ element: Entry) -> Entry {
        Entry(2, [.int(UInt64(shape.count))] + shape.map { .signed(Int64($0)) } + [.type(element)])
    }
    static let wideInteger = Entry(23)

    // MARK: Attributes

    public static func string(_ s: String) -> Entry { Entry(1, [.string(s)]) }
    /// Signless; up to 8 bits a raw byte, beyond a signed varint (MLIR's `writeAPIntWithKnownWidth`).
    public static func integer(_ v: Int64, bits: UInt64) -> Entry {
        Entry(2, [.int(bits), .int(0), bits <= 8 ? .byte(UInt8(truncatingIfNeeded: v)) : .signed(v)])
    }
    public static let no = integer(0, bits: 1)
    public static func typeAttribute(_ t: Entry) -> Entry { Entry(4, [.type(t)]) }
    public static func array(_ xs: [Entry]) -> Entry { Entry(5, [.int(UInt64(xs.count))] + xs.map { .attribute($0) }) }
    public static func dense(_ bytes: [UInt8], splat: Bool) -> Entry { Entry(6, [.byte(splat ? 1 : 0), .blob(bytes)]) }
    /// The second field is the integer 1, not the attribute `no` (they encode alike while `no` is attribute 1, as in a module with
    /// no tie in use counts; `dequantize_v2`'s dims constant outnumbers `no` and moves it to index 2).
    public static func named(_ name: String, _ value: Entry) -> Entry { Entry(9, [.string(name), .int(1), .attribute(value)]) }
    /// The location every op carries once debug information is stripped: `"-":0:0`.
    public static let location: Entry = Entry(40, [
        .attribute(Entry(31, [.attribute(Entry(30, [.string("-"), .string(""), .string("")])), .int(0), .int(0), .int(0), .int(0)])),
        .attribute(Entry(46)), .int(0), .int(0),
    ])

    /// The arguments' names.
    static func names(_ names: [String]) -> Entry {
        array(names.map { n in array([named("aicode.name", string(n))]) })
    }

    /// Dense data, little-endian: one value repeated (a single one included) is stored once, as a splat.
    public static func dense<T: FixedWidthInteger>(_ values: [T]) -> Entry {
        precondition(!values.isEmpty, "dense data of no value: a scalar's shape is `dense([UInt8](), splat: false)`")
        let splat = values.allSatisfy { $0 == values[0] }
        let stored = splat ? [values[0]] : values
        return dense(stored.flatMap { v in (0..<MemoryLayout<T>.size).map { UInt8(truncatingIfNeeded: v >> (8 * $0)) } }, splat: splat)
    }
    /// One scalar in an element type's bytes.
    static func scalar(_ x: Double, _ e: Element) -> [UInt8] {
        let bits: UInt64 = switch e {
        case .f32: UInt64(Float(x).bitPattern)
        case .f16: UInt64(Float16(x).bitPattern)
        case .bf16: UInt64(Float(x).bitPattern >> 16)  // x is exact in bf16 where it is used (0, 1)
        case .si32: UInt64(UInt32(bitPattern: Int32(x)))
        case .ui32: UInt64(UInt32(x))
        case .si8: UInt64(UInt8(bitPattern: Int8(x)))
        case .si4: UInt64(UInt8(bitPattern: Int8(x)) & 0xF)  // one byte holds a splat's one code (`dense<0>`)
        case .bool: x == 0 ? 0 : 1
        }
        let width = e.size
        return (0..<width).map { UInt8(truncatingIfNeeded: bits >> UInt64(8 * $0)) }
    }

    /// A tensor type without its trailing unit axes (`Layout.trimmed`); any other type as it is.
    public static func trimmed(_ t: Entry) -> Entry {
        guard t.kind == 2 else { return t }
        let (s, e) = shape(t)
        return tensor(Layout.trimmed(s), e)
    }

    /// A tensor type's shape and element.
    public static func shape(_ t: Entry) -> (shape: [Int], element: Element) {
        let rank = { if case .int(let r) = t.fields[0] { Int(r) } else { 0 } }()
        let dims: [Int] = t.fields.dropFirst().prefix(rank).map { if case .signed(let d) = $0 { Int(d) } else { 0 } }
        guard case .type(let e) = t.fields[rank + 1], let element = Element(rawValue: e.kind) else { preconditionFailure("not a tensor type") }
        return (dims, element)
    }

    public enum Binary: String, Sendable { case add, sub, mul, divide }

    // MARK: Ops

    public final class Op {
        public let name: String
        public var operands: [Int]
        public var results: [Entry]
        public var properties: [Property]?
        /// One block, in an isolated region (every region here is).
        public var body: Block?
        public init(_ name: String, operands: [Int] = [], results: [Entry] = [], properties: [Property]? = nil, body: Block? = nil) {
            self.name = name; self.operands = operands; self.results = results; self.properties = properties; self.body = body
        }

        /// The dates SiliCore learnt from coreai-torch's assets: the latest an asset holding the op says (its own tables,
        /// merged; an op met only beside later ones is at most their date).
        static let dates: [String: Int64] = [
            // Reductions.
            "reduce_sum_v1": 20250926, "reduce_max_v1": 20251023, "reduce_min_v1": 20251023, "argmax_v1": 20251028,
            "reduce_mean_v1": 20260317,
            // Quantized: both composites were met in assets dated by the `reshape_v2` they hold.
            "dequantize_v2": 20260204, "quantize_v2": 20260204, "primitive.blockwise_shift_scale_v2": 20260202,
            // Elementwise.
            "primitive.sqrt_v1": 20250717, "primitive.abs_v1": 20250814, "sigmoid_v1": 20250922,
            "broadcasting_equal_v1": 20250923, "primitive.equal_v1": 20250923,
            "primitive.rsqrt_v1": 20250928, "primitive.log_v1": 20250928, "primitive.sin_v1": 20250928,
            "primitive.cos_v1": 20250928, "primitive.tanh_v1": 20250928,
            "broadcasting_maximum_v1": 20250930, "primitive.maximum_v1": 20250930,
            "broadcasting_minimum_v1": 20250930, "primitive.minimum_v1": 20250930,
            "broadcasting_pow_v1": 20250930, "primitive.pow_v1": 20250930,
            "primitive.gelu_v1": 20251024, "broadcasting_where_v1": 20251024, "primitive.where_v1": 20251024,
            "primitive.broadcast_shapes_v1": 20251024, "primitive.broadcast_to_v1": 20251024,
            "broadcasting_greater_v1": 20251105, "primitive.not_v1": 20251105,
            "broadcasting_not_equal_v1": 20260112, "primitive.not_equal_v1": 20260112,
            // Shapes.
            "primitive.concat_v1": 20250903, "primitive.broadcast_in_dims_v1": 20250826, "primitive.slice_v1": 20250922,
            "primitive.slice_update_v1": 20250922, "primitive.pad_v1": 20250924,
            "primitive.gather_along_axis_v1": 20251028, "primitive.scatter_along_axis_v1": 20251030,
            // Convolution (`Convolution.swift`).
            "primitive.conv2d_v1": 20250905, "primitive.conv3d_v1": 20251023,
        ]

        static func since(_ name: String) -> Int64 {
            if let date = dates[name] { return date }
            return switch name {
            case "primitive.reshape_v2": 20260204
            case "broadcasting_batch_matmul_v1", "primitive.batch_matmul_v1": 20251202
            case _ where name.hasPrefix("broadcasting_"),
                 "primitive.add_v1", "primitive.sub_v1", "primitive.mul_v1", "primitive.divide_v1": 20250827
            // A transpose alone (its permutation a constant) is 2025-08-11: both at most that, one exactly.
            case "primitive.transpose_v1", "primitive.constant_v1": 20250811
            case "primitive.cast_v1": 20250723
            default: 20250819
            }
        }
    }

    /// A block: its arguments are values 0..<n, then each op's results in order.
    public final class Block {
        public let arguments: [Entry]
        public var ops: [Op] = []
        /// Every value's type, arguments first.
        public private(set) var types: [Entry]
        public var values: Int { types.count }
        public init(_ arguments: [Entry]) { self.arguments = arguments; types = arguments }

        @discardableResult
        public func add(_ op: Op) -> [Int] {
            ops.append(op)
            defer { types += op.results }
            return (0..<op.results.count).map { values + $0 }
        }
    }
}

extension Aicode.Element {
    /// Bytes of one element; an int4 has none of its own (`bytes(_:)`).
    public var size: Int { switch self { case .bool, .si8, .si4: 1; case .f16, .bf16: 2; default: 4 } }
    /// Bytes of `count` elements: int4 two to a byte, the first in the low nibble (coreai-torch's resources).
    public func bytes(_ count: Int) -> Int { self == .si4 ? (count + 1) / 2 : count * size }
}

// MARK: - The core composites, decomposed as coreai-torch decomposes them

extension Aicode.Block {
    func constant(_ data: Aicode.Entry, _ type: Aicode.Entry) -> Int {
        add(Aicode.Op("primitive.constant_v1", results: [type], properties: [.attribute(data)]))[0]
    }

    func yield(_ v: Int) { add(Aicode.Op("dataflow.yield_v1", operands: [v])) }

    /// `x @ w` over the last two axes, of equal ranks.
    public func matmul(_ a: Int, _ b: Int, result: Aicode.Entry) -> Int {
        let body = Aicode.Block([types[a], types[b]])
        let y = body.add(Aicode.Op("primitive.batch_matmul_v1", operands: [0, 1], results: [result]))[0]
        body.yield(y)
        return add(Aicode.Op("broadcasting_batch_matmul_v1", operands: [a, b], results: [result], body: body))[0]
    }

    /// `x · wᵀ` as coreai-torch writes `linear`: w's last two axes transposed, then the product.
    public func linear(_ x: Int, _ w: Int) -> Int {
        let (xs, element) = Aicode.shape(types[x]), ws = Aicode.shape(types[w]).shape
        var p = Array(0..<ws.count)
        p.swapAt(p.count - 2, p.count - 1)
        return matmul(x, transpose(w, p), result: Aicode.tensor(Array(xs.dropLast()) + [ws[ws.count - 2]], element))
    }

    /// An elementwise binary op with numpy broadcasting: inside, each operand that is not the result's shape is
    /// reshaped to its rank (ones prepended) when shorter, then broadcast along the axes where it is 1.
    public func binary(_ kind: Aicode.Binary, _ a: Int, _ b: Int) -> Int { broadcasting(kind.rawValue, a, b) }

    /// `broadcasting_<name>_v1(a, b)` with numpy broadcasting, for any rank (a scalar operand has rank 0), its result in
    /// the operands' element or in `result` (a comparison's bool): every elementwise op of two operands is written so.
    func broadcasting(_ name: String, _ a: Int, _ b: Int, result element: Aicode.Element? = nil) -> Int {
        let (sa, operand) = Aicode.shape(types[a]), sb = Aicode.shape(types[b]).shape
        let rank = max(sa.count, sb.count)
        let pa = Array(repeating: 1, count: rank - sa.count) + sa, pb = Array(repeating: 1, count: rank - sb.count) + sb
        let out = zip(pa, pb).map { max($0, $1) }
        let broadcasted = Aicode.tensor(out, operand), result = Aicode.tensor(out, element ?? operand)
        let body = Aicode.Block([types[a], types[b]])
        func broadcast(_ v: Int, _ s: [Int]) -> Int {
            if s == out { return v }
            var x = v, padded = s
            if s.count < rank {
                padded = Array(repeating: 1, count: rank - s.count) + s
                let shape = body.constant(Aicode.dense(padded.map { Int32($0) }), Aicode.tensor([rank], .si32))
                x = body.add(Aicode.Op("primitive.reshape_v2", operands: [x, shape], results: [Aicode.tensor(padded, operand)]))[0]
            }
            let axes = (0..<rank).filter { padded[$0] != out[$0] }
            if axes.isEmpty { return x }
            let sizes = body.constant(Aicode.dense(axes.map { UInt32(out[$0]) }), Aicode.tensor([axes.count], .ui32))
            let dims = body.constant(Aicode.dense(axes.map { Int32($0) }), Aicode.tensor([axes.count], .si32))
            return body.add(Aicode.Op("primitive.broadcast_in_dims_v1", operands: [x, sizes, dims], results: [broadcasted]))[0]
        }
        let x = broadcast(0, sa), y = broadcast(1, sb)
        let r = body.add(Aicode.Op("primitive.\(name)_v1", operands: [x, y], results: [result]))[0]
        body.yield(r)
        return add(Aicode.Op("broadcasting_\(name)_v1", operands: [a, b], results: [result], body: body))[0]
    }

    /// A conversion to `element`, rounding to nearest (a primitive: no decomposition).
    public func cast(_ x: Int, to element: Aicode.Element) -> Int {
        add(Aicode.Op("primitive.cast_v1", operands: [x], results: [Aicode.tensor(Aicode.shape(types[x]).shape, element)]))[0]
    }

    /// A conversion unless `x` is already `element`: coreai-torch writes none of the casts that change nothing.
    public func castUnlessSame(_ x: Int, to element: Aicode.Element) -> Int {
        Aicode.shape(types[x]).element == element ? x : cast(x, to: element)
    }

    /// The axes of `x` in the order `permutation` names them, the permutation a ui32 constant before it.
    public func transpose(_ x: Int, _ permutation: [Int]) -> Int {
        let (s, element) = Aicode.shape(types[x])
        let p = constant(Aicode.dense(permutation.map { UInt32($0) }), Aicode.tensor([permutation.count], .ui32))
        return add(Aicode.Op("primitive.transpose_v1", operands: [x, p], results: [Aicode.tensor(permutation.map { s[$0] }, element)]))[0]
    }

    public func exp(_ x: Int) -> Int {
        add(Aicode.Op("primitive.exp_v1", operands: [x], results: [types[x]]))[0]
    }

    /// `x · 1 / (1 + exp(0 − x))`, the scalars as one-element tensors of x's element.
    public func silu(_ x: Int) -> Int {
        let element = Aicode.shape(types[x]).element
        let body = Aicode.Block([types[x]])
        let one = Aicode.tensor([1], element)
        let c0 = body.constant(Aicode.dense(Aicode.scalar(0, element), splat: true), one)
        let c1 = body.constant(Aicode.dense(Aicode.scalar(1, element), splat: true), one)
        let e = body.exp(body.binary(.sub, c0, 0))
        let r = body.binary(.divide, c1, body.binary(.add, c1, e))
        let y = body.binary(.mul, 0, r)
        body.yield(y)
        return add(Aicode.Op("silu_v1", operands: [x], results: [types[x]], body: body))[0]
    }
}

// MARK: - The module

extension Aicode {
    /// One function, `main`, as coreai-torch writes it: named arguments, named results.
    public struct Module {
        public var body: Block
        public var argumentNames: [String]
        public var outputs: [(name: String, value: Int)]
        /// The blobs of the constants of 256 elements or more, in the order their constants were added.
        public private(set) var resources: [(key: String, bytes: [UInt8])] = []

        public init(arguments: [(String, Entry)]) {
            body = Block(arguments.map(\.1))
            argumentNames = arguments.map(\.0)
            outputs = []
        }

        /// The value of the argument `name`.
        public func argument(_ name: String) -> Int {
            guard let i = argumentNames.firstIndex(of: name) else { preconditionFailure("no argument \(name)") }
            return i
        }

        /// **The module as the engine's functions declare it**: each tensor argument and result without its trailing
        /// unit axes (`Layout.trimmed`), reshaped back where it is read and before it is returned; the module itself when
        /// none has any. The same ops, renumbered (every region is isolated: only the top block's operands move).
        public func trimmed() -> Module {
            let declared = body.arguments.map(Aicode.trimmed)
            guard declared != body.arguments || outputs.contains(where: { Aicode.trimmed(body.types[$0.value]) != body.types[$0.value] }) else { return self }
            var m = Module(arguments: Array(zip(argumentNames, declared)))
            m.resources = resources
            let b = m.body
            var map = body.arguments.indices.map { declared[$0] == body.arguments[$0] ? $0 : b.reshape($0, Aicode.shape(body.arguments[$0]).shape) }
            for op in body.ops {
                map += b.add(Op(op.name, operands: op.operands.map { map[$0] }, results: op.results, properties: op.properties, body: op.body))
            }
            m.outputs = outputs.map { o in
                let v = map[o.value], t = Aicode.trimmed(b.types[v])
                return (o.name, t == b.types[v] ? v : b.reshape(v, Aicode.shape(t).shape))
            }
            return m
        }

        /// A constant the module computes with (a rotation's matrix; the weights are arguments): written in the IR under
        /// 256 elements, a splat when every element is the same, otherwise a blob of the resource section, 16-aligned,
        /// under `key` — coreai-torch's rule (`create_elements_attr`), the key ours: the same bytes from one run to the next.
        public mutating func constant(_ key: String, _ bytes: [UInt8], _ type: Entry) -> Int {
            let (shape, element) = Aicode.shape(type)
            let width = element.size
            precondition(bytes.count == element.bytes(shape.reduce(1, *)), "constant \(key): \(bytes.count) bytes for \(shape)")
            precondition(element != .si4 || shape.reduce(1, *) >= 256, "constant \(key): \(shape) int4 is under 256 codes")
            let data: Entry
            if shape.reduce(1, *) < 256 {
                let small = bytes, first = small.prefix(width)
                let splat = stride(from: 0, to: small.count, by: width).allSatisfy { small[$0..<$0 + width].elementsEqual(first) }
                data = Aicode.dense(splat ? Array(first) : small, splat: splat)
            } else {
                precondition(!resources.contains { $0.key == key }, "resource \(key) given twice")
                resources.append((key, bytes))
                data = Entry(7, [.resource(key)])
            }
            return body.constant(data, type)
        }

        /// The bytes of `main.mlirb`. A call that carries a composite (`Aicode.Block.attention`) has the composite's graph
        /// written before `main`, as coreai-torch's assets hold them, and keeps only its symbol.
        public func bytecode() -> [UInt8] {
            typealias Op = Aicode.Op
            var graphs: [Op] = [], calls: [(op: Op, body: Aicode.Block, properties: [Aicode.Property])] = []
            for op in body.ops where op.name == "dataflow.invoke_v1" {
                guard let inner = op.body, let properties = op.properties, properties.count == 4,
                      case .attribute(let callee) = properties[1], case .attribute(let reference) = callee.fields[0],
                      case .string(let name) = reference.fields[1], case .attribute(let declaration) = properties[3] else { continue }
                let results = inner.ops.last.map { $0.operands.map { inner.types[$0] } } ?? []
                let function = Aicode.function(inner.arguments, results)
                graphs.append(Op("dataflow.graph_v1", properties: [
                    .attribute(Aicode.names(inner.arguments.indices.map { ["q", "k", "v", "m"][$0] })), .optional(declaration),
                    .attribute(Aicode.no), .attribute(Aicode.typeAttribute(function)),
                    .attribute(Aicode.integer(1, bits: 1)), .attribute(Aicode.string(name)),
                    .attribute(Aicode.integer(1, bits: 1)), .attribute(Aicode.array([])),
                    .attribute(Aicode.array([Aicode.array([])])),
                ], body: inner))
                calls.append((op, inner, properties))
            }
            for c in calls { c.op.body = nil; c.op.properties = Array(c.properties.prefix(3)) }
            defer { for c in calls { c.op.body = c.body; c.op.properties = c.properties } }

            var all: [Op] = []
            func walk(_ b: Aicode.Block) { for op in b.ops { all.append(op); if let body = op.body { walk(body) } } }
            graphs.forEach { walk($0.body!) }
            walk(body)
            let date = all.map { Op.since($0.name) }.max() ?? Op.since("dataflow.graph_v1")
            let resultTypes = outputs.map { body.types[$0.value] }
            let main = Op("dataflow.graph_v1", properties: [
                .attribute(Aicode.names(argumentNames)), .int(0),
                .attribute(Aicode.no), .attribute(Aicode.typeAttribute(Aicode.function(body.arguments, resultTypes))),
                .attribute(Aicode.no), .attribute(Aicode.string("main")),
                .attribute(Aicode.no), .attribute(Aicode.array([])),
                .attribute(Aicode.names(outputs.map(\.name))),
            ], body: body)
            body.ops.append(Op("dataflow.output_v1", operands: outputs.map(\.value)))
            defer { body.ops.removeLast() }
            let inner = Aicode.Block([]); inner.ops = graphs + [main]
            let asset = Op("serialization.asset", properties: [.int(0), .optional(Aicode.integer(date, bits: 32))], body: inner)
            return Writer(root: asset, resources: resources).bytes()
        }

    }

    /// Writes `directory` (an `.aimodel`): `main.mlirb`, its SHA-256 in `main.hash`, and `metadata.json`.
    public static func write(_ bytecode: [UInt8], to directory: URL, producer: String) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(bytecode).write(to: directory.appendingPathComponent("main.mlirb"))
        try Data(SHA256.hash(data: bytecode)).write(to: directory.appendingPathComponent("main.hash"))
        let date = ISO8601DateFormatter()
        date.formatOptions = [.withYear, .withMonth, .withDay, .withTime, .withTimeZone]
        let stamp = date.string(from: Date()).replacingOccurrences(of: "-", with: "").replacingOccurrences(of: ":", with: "")
        let metadata = "{\n  \"producer\" : \"\(producer)\",\n  \"creationDate\" : \"\(stamp)\",\n  \"assetVersion\" : \"2.0\"\n}"
        try Data(metadata.utf8).write(to: directory.appendingPathComponent("metadata.json"))
    }
}

// MARK: - Numbering and writing

/// MLIR's `IRNumbering` and `BytecodeWriter`, for one dialect and isolated single-block regions.
struct Writer {
    let root: Aicode.Op
    let resources: [(key: String, bytes: [UInt8])]

    private struct Counted<Key: Hashable> {
        var first: [Key: Int] = [:]
        var refs: [Int] = []
        var keys: [Key] = []
        /// True when the key is new.
        mutating func count(_ k: Key) -> Bool {
            if let i = first[k] { refs[i] += 1; return false }
            first[k] = keys.count; keys.append(k); refs.append(1); return true
        }
        /// Most used first, ties in the order met.
        func ordered() -> (keys: [Key], index: [Key: Int]) {
            let order = keys.indices.sorted { refs[$0] != refs[$1] ? refs[$0] > refs[$1] : $0 < $1 }
            let keys = order.map { self.keys[$0] }
            return (keys, Dictionary(uniqueKeysWithValues: keys.enumerated().map { ($1, $0) }))
        }
    }

    func bytes() -> [UInt8] {
        // Numbering: pre-order, isolated regions from a stack.
        var ops = Counted<String>(), attrs = Counted<Aicode.Entry>(), types = Counted<Aicode.Entry>()
        var used = Counted<String>()  // resources, in the order met (never sorted)
        func field(_ f: Aicode.Field) {
            switch f {
            case .attribute(let e): attribute(e)
            case .type(let e): type(e)
            case .resource(let k): _ = used.count(k)
            default: break
            }
        }
        func attribute(_ e: Aicode.Entry) { if attrs.count(e) { e.fields.forEach(field) } }
        func type(_ e: Aicode.Entry) { if types.count(e) { e.fields.forEach(field) } }
        func number(_ op: Aicode.Op) {
            _ = ops.count(op.name)
            op.results.forEach(type)
            for p in op.properties ?? [] {
                switch p {
                case .attribute(let e): attribute(e)
                case .optional(let e?): attribute(e)
                default: break
                }
            }
            attribute(Aicode.location)
        }
        number(root)
        var stack: [Aicode.Block] = root.body.map { [$0] } ?? []
        while let block = stack.popLast() {
            for t in block.arguments { attribute(Aicode.location); type(t) }
            block.ops.forEach(number)
            stack += block.ops.compactMap(\.body)
        }
        let (opOrder, opIndex) = ops.ordered()
        let (attrOrder, attrIndex) = attrs.ordered()
        let (typeOrder, typeIndex) = types.ordered()

        // Strings, as they are written.
        var strings: [String] = [], stringIndex: [String: Int] = [:]
        func str(_ s: String) -> Int {
            if let i = stringIndex[s] { return i }
            stringIndex[s] = strings.count; strings.append(s); return strings.count - 1
        }

        var dialect = BytecodeEmitter()
        dialect.varint(1)
        dialect.varint(str("aicode") << 1 | 1)
        var version = BytecodeEmitter()
        [1, 2, UInt64(opOrder.count)].forEach { version.varint($0) }
        dialect.varint(version.bytes.count); dialect.append(version.bytes)
        dialect.varint(0); dialect.varint(opOrder.count)
        for name in opOrder { dialect.varint(str(name) << 1 | 1) }

        func encode(_ e: Aicode.Entry) -> [UInt8] {
            var w = BytecodeEmitter()
            w.varint(e.kind)
            for f in e.fields {
                switch f {
                case .int(let v): w.varint(v)
                case .signed(let v): w.signed(v)
                case .byte(let b): w.byte(b)
                case .string(let s): w.varint(str(s))
                case .attribute(let a): w.varint(attrIndex[a]!)
                case .type(let t): w.varint(typeIndex[t]!)
                case .blob(let b): w.varint(b.count); w.append(b)
                case .resource(let k): w.varint(used.first[k]!)
                }
            }
            return w.bytes
        }
        var table = BytecodeEmitter(), offsets = BytecodeEmitter()
        offsets.varint(attrOrder.count); offsets.varint(typeOrder.count)
        for group in [attrOrder, typeOrder] where !group.isEmpty {
            offsets.varint(0); offsets.varint(group.count)
            for e in group { let b = encode(e); table.append(b); offsets.varint(b.count << 1 | 1) }
        }

        // IR, properties numbered as they are written (equal encodings shared).
        var properties: [[UInt8]] = [], propertyIndex: [[UInt8]: Int] = [:]
        let location = attrIndex[Aicode.location]!
        func write(_ op: Aicode.Op, _ w: inout BytecodeEmitter) {
            w.varint(opIndex[op.name]!)
            var mask: UInt8 = 0
            if !op.results.isEmpty { mask |= 0x02 }
            if !op.operands.isEmpty { mask |= 0x04 }
            if op.body != nil { mask |= 0x10 }
            if op.properties != nil { mask |= 0x40 }
            w.byte(mask)
            w.varint(location)
            if let props = op.properties {
                var p = BytecodeEmitter()
                for f in props {
                    switch f {
                    case .int(let v): p.varint(v)
                    case .attribute(let e): p.varint(attrIndex[e]!)
                    case .optional(let e): if let e { p.varint(attrIndex[e]! << 1 | 1) } else { p.varint(0) }
                    }
                }
                if propertyIndex[p.bytes] == nil { propertyIndex[p.bytes] = properties.count; properties.append(p.bytes) }
                w.varint(propertyIndex[p.bytes]!)
            }
            if !op.results.isEmpty { w.varint(op.results.count); op.results.forEach { w.varint(typeIndex[$0]!) } }
            if !op.operands.isEmpty { w.varint(op.operands.count); op.operands.forEach { w.varint($0) } }
            if let body = op.body {
                w.varint(1 << 1 | 1)
                var region = BytecodeEmitter()
                region.varint(1); region.varint(body.values)
                region.varint(body.ops.count << 1 | (body.arguments.isEmpty ? 0 : 1))
                if !body.arguments.isEmpty {
                    region.varint(body.arguments.count)
                    for t in body.arguments { region.varint(typeIndex[t]! << 1 | 1); region.varint(location) }
                    region.byte(0)
                }
                for o in body.ops { write(o, &region) }
                w.section(4, region.bytes)
            }
        }
        var ir = BytecodeEmitter()
        ir.varint(1 << 1)
        write(root, &ir)

        // Resources: each blob its alignment, its size, padding to 16 from the file's start, its bytes; the
        // section aligned the same way, so that a reader can map the data in place.
        let blobs = Dictionary(uniqueKeysWithValues: resources.map { ($0.key, $0.bytes) })
        precondition(Set(used.keys) == Set(blobs.keys), "resources given and used differ")
        var resourceOffsets = BytecodeEmitter(), resourceSection: [UInt8] = [], header = BytecodeEmitter(), length = 0
        resourceOffsets.varint(0)  // no external group
        if !used.keys.isEmpty {
            resourceOffsets.varint(0); resourceOffsets.varint(used.keys.count)
            for k in used.keys {
                let start = length
                header = BytecodeEmitter()
                header.varint(16); header.varint(blobs[k]!.count)
                while (length + header.bytes.count) % 16 != 0 { header.byte(0xcb) }
                resourceSection += header.bytes + blobs[k]!
                length += header.bytes.count + blobs[k]!.count
                resourceOffsets.varint(str(k)); resourceOffsets.varint(length - start); resourceOffsets.byte(0)
            }
        }

        var stringSection = BytecodeEmitter()
        stringSection.varint(strings.count)
        for s in strings.reversed() { stringSection.varint(s.utf8.count + 1) }
        for s in strings { stringSection.append(Array(s.utf8) + [0]) }
        var propertySection = BytecodeEmitter()
        propertySection.varint(properties.count)
        for p in properties { propertySection.varint(p.count); propertySection.append(p) }

        var file = BytecodeEmitter()
        file.append([0x4d, 0x4c, 0xef, 0x52]); file.varint(6); file.append(Array("MLIR22.0.0git".utf8) + [0])
        file.section(1, dialect.bytes)
        file.section(3, offsets.bytes)
        file.section(2, table.bytes)
        file.section(4, ir.bytes)
        file.section(6, resourceOffsets.bytes)
        if resourceSection.isEmpty { file.section(5, []) } else {
            file.byte(5 | 0x80); file.varint(length); file.varint(16)
            while file.bytes.count % 16 != 0 { file.byte(0xcb) }
            file.append(resourceSection)
        }
        file.section(0, stringSection.bytes)
        file.section(8, propertySection.bytes)
        return file.bytes
    }
}
