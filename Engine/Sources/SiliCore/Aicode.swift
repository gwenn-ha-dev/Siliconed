import Foundation

/// A Core AI asset written by us: MLIR bytecode (version 6) in Core AI's stable serialization dialect, `aicode`,
/// which Core AI's loader turns back into its `coreai` ops and then places, fuses and tiles. Recopied from the former engine
/// (`CoreAIAsset.swift`, 0.21) without its graph: a function is written op by op on a `Block`.
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
/// What a port by analogy would miss (the former engine's findings, judged there against coreai-torch's bytes):
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
///     composite's own constants (`Block.constant`: a rotation's matrix) stay dense in the IR, as the former engine wrote them;
///   - **the dialect's "version" is `1 · 2 · the count of op names`**, and **the asset's date is the latest any op
///     needs**: each op says its own (`Op.since`);
///   - **the weights are arguments, never constants of the asset**: an asset is the same for any weights, which
///     we bind at run time as views of the arena. The constants it holds are its own (a rotation's matrix, the
///     scalars of a composite).

// MARK: - aicode's entries

public enum Aicode {
    /// **The unit a function is written for**, while a block's function is built (`Block.Build.module`): true, the ANE's
    /// or either unit's. A block reads it once, where it is made (`Block.neural`). Outside a build, the ANE's way, which
    /// both units compute.
    @TaskLocal public static var neural = true

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

        /// The dates the former engine learnt from coreai-torch's assets: the latest an asset holding the op says (its own tables,
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
        /// Written for the ANE (`Aicode.neural`, read here): the guard's factors without `round` (`ranged16`). The GPU's
        /// compute them in f32, by `round`.
        public let neural = Aicode.neural
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
