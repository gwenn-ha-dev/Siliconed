import CoreAI
import Foundation
import IOSurface
import Metal

/// A tensor of the arena as a function reads or writes it: its buffer, where it starts in it, its shape and strides
/// (row-major when empty); its surface, when it is one whole (`Arena.take(surface:)`), what the ANE reads.
public struct View {
    public let buffer: MTLBuffer
    public let offset: Int
    public let type: NDArray.ScalarType
    public let shape: [Int]
    public let strides: [Int]
    public let surface: IOSurface?

    /// `shape` without its trailing unit axes, as every function declares its arguments (`Layout.trimmed`).
    public init(buffer: MTLBuffer, offset: Int = 0, type: NDArray.ScalarType, shape: [Int], strides: [Int] = [], surface: IOSurface? = nil) {
        let trimmed = Layout.trimmed(shape)
        self.buffer = buffer; self.offset = offset; self.type = type; self.shape = trimmed; self.surface = surface
        self.strides = strides.isEmpty ? [] : Array(strides.prefix(trimmed.count))
    }
}

public enum Run {

    /// One call of `function`, waited for: one invocation in flight per function (Apple's doc: a function called
    /// concurrently allocates other intermediate buffers). Its outputs each written in place in its view.
    public static func call(_ function: InferenceFunction, _ stream: SerialStream, inputs: [String: View], outputs: [(String, View)]) async throws {
        let left = try encode(function, stream, inputs: inputs, outputs: outputs)
        guard left.isEmpty else { throw Failure("outputs no view holds: \(left.keys.sorted())") }
        await finish(stream)
    }

    /// **One call of `function` put on `stream`, not waited for**: its outputs held by no view returned as Core AI's
    /// values, to give the next call as its `chained` inputs. The stream orders its work by these values, not by the
    /// buffers they lie in: a call reading, as a view of our buffer, what the call before writes there may read it before
    /// it is written (a transformer's layers chained through the arena's views gave wrong values, 18/11; chained through
    /// the returned values, right). `finish` waits for the stream. A stage's stream: the stage called (`Stage.keep`).
    public static func encode(_ function: InferenceFunction, _ stream: SerialStream, inputs: [String: View],
                              chained: [String: InferenceFunction.AsyncValue] = [:],
                              outputs: [(String, View)]) throws -> [String: InferenceFunction.AsyncValue] {
        var ins = chained
        for (name, v) in inputs { ins[name] = value(v) }
        let values = try stream.encode { try bind(function, $0, ins, outputs[...], InferenceFunction.AsyncMutableViews()) }
        Stage.call(on: stream)
        return values
    }

    /// A view borrows its value: each output's value lives in its own frame, the encode at the deepest.
    private static func bind(_ f: InferenceFunction, _ stream: ComputeStream, _ ins: [String: InferenceFunction.AsyncValue],
                             _ rest: ArraySlice<(String, View)>,
                             _ outputs: consuming InferenceFunction.AsyncMutableViews) throws -> [String: InferenceFunction.AsyncValue] {
        guard let (name, v) = rest.first else { return try f.encode(inputs: ins, states: InferenceFunction.AsyncMutableViews(), outputViews: outputs, to: stream) }
        var value = InferenceFunction.AsyncMutableValue(unsafeBuffer: v.buffer, byteOffset: v.offset, scalarType: v.type, shape: v.shape, strides: v.strides)
        var next = outputs
        next.insert(&value, for: name)
        return try bind(f, stream, ins, rest.dropFirst(), next)
    }

    /// A view as Core AI's input value.
    public static func value(_ v: View) -> InferenceFunction.AsyncValue {
        InferenceFunction.AsyncValue(unsafeBuffer: v.buffer, byteOffset: v.offset, scalarType: v.type, shape: v.shape, strides: v.strides)
    }

    /// The work put on `stream` so far, waited for.
    public static func finish(_ stream: SerialStream) async { _ = await stream.done().value }

    /// One call of the ANE's, waited for: every operand a surface of the arena at its exact shape, from its byte `offset`
    /// (a unit of `Arena.take(units:)`; on buffers, `encode` holds a call in 17 to 33 for ~600 ms).
    public static func neural(_ function: InferenceFunction, inputs: [String: View], outputs: [(String, View)]) async throws {
        let ins = inputs.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
        for (name, v) in ins + outputs {
            guard let s = v.surface, v.strides.isEmpty, v.offset + v.shape.reduce(1, *) * Layout.element(v.type) <= IOSurfaceGetAllocSize(s) else {
                preconditionFailure("\(name): not on a surface")
            }
            precondition(IOSurfaceGetWidth(s) == v.shape.last! && IOSurfaceGetHeight(s) == v.shape.dropLast().reduce(1, *), "\(name): \(v.shape) on a surface of another")
        }
        try await bind(function, ins, outputs, 0, InferenceFunction.Inputs(), InferenceFunction.MutableViews())
    }

    /// A view borrows its surface: each one a value of the frame that inserts it, the run at the deepest.
    private static func bind(_ f: InferenceFunction, _ ins: [(String, View)], _ outs: [(String, View)], _ i: Int,
                             _ inputs: consuming InferenceFunction.Inputs, _ views: consuming InferenceFunction.MutableViews) async throws {
        if i < ins.count {
            let (name, v) = ins[i], surface = v.surface!
            inputs.insert(unsafe NDArray.RawView(ioSurface: surface, byteOffset: v.offset, scalarType: v.type, shape: v.shape), for: name)
            return try await bind(f, ins, outs, i + 1, inputs, views)
        }
        if i - ins.count < outs.count {
            let (name, v) = outs[i - ins.count], surface = v.surface!
            views.insert(unsafe NDArray.MutableRawView(ioSurface: surface, byteOffset: v.offset, scalarType: v.type, shape: v.shape), for: name)
            return try await bind(f, ins, outs, i + 1, inputs, views)
        }
        _ = try await f.run(inputs: inputs, outputViews: views)
    }
}

/// **How a call's arguments lie, as Core AI asks them**: a function's descriptor says, for its specialization on this
/// Mac, the scalar type, the shape, the strides it prefers and the bytes it needs (`NDArrayDescriptor.preferredStrides`,
/// `minimumByteCount`). What differs is copied by Core AI at each call (measured 11/11: a bf16 argument whose rows are
/// not a multiple of 64 bytes, each row padded to 64 for the whole inference); the plans lay out what it asks.
public enum Layout {

    /// The row-major strides of `shape`, in elements.
    public static func rowMajor(_ shape: [Int]) -> [Int] {
        var s = [Int](repeating: 1, count: shape.count)
        for i in stride(from: shape.count - 2, through: 0, by: -1) { s[i] = s[i + 1] * shape[i + 1] }
        return s
    }

    /// **The bytes a row of Core AI's GPU takes**: its arguments' rows a multiple of 64 bytes, a shorter one padded
    /// (`preferredStrides`, read 11/11 on the M1 Pro and 19/11 on the M5 Pro: `[1, 96, 52, 304]` f16 asked strides
    /// `[1597440, 16640, 320, 1]`, `[1, 512, 1]` f16 `[16384, 32, 1]`). What a plan lays out before any function is
    /// loaded.
    public static let rowBytes = 64

    /// The strides, in elements, of `shape` laid as Core AI's GPU asks: row-major, its last dimension padded to
    /// `rowBytes` (a vector as it is).
    public static func padded(_ shape: [Int], _ type: NDArray.ScalarType) -> [Int] {
        guard shape.count >= 2 else { return rowMajor(shape) }
        let e = element(type), row = (shape.last! * e + rowBytes - 1) / rowBytes * rowBytes / e
        var s = [Int](repeating: 1, count: shape.count)
        s[shape.count - 2] = row
        for i in stride(from: shape.count - 3, through: 0, by: -1) { s[i] = s[i + 1] * shape[i + 1] }
        return s
    }

    /// `shape` without its trailing unit axes (one axis at least): how every function declares its arguments and results,
    /// every view and every surface is laid. Core AI's GPU pads a last axis to `rowBytes`: a last axis of 1 to 64 B an
    /// element, which a surface, its rows packed, cannot give (`[1, rows, 1]` f16 asked `[32 · rows, 32, 1]`, 19/11).
    public static func trimmed(_ shape: [Int]) -> [Int] {
        var s = shape
        while s.count > 1, s.last == 1 { s.removeLast() }
        return s
    }

    /// The elements of a row of `n` that starts a page of its own (`Arena.page`): the rows of a tensor its readers take
    /// one by one, each then a buffer of its own (`Stage.view`). Measured 19/11 (M5 Pro): a buffer a placement heap puts
    /// off a page, at its 256 B alignment, Core AI still copies (0.92 ms against 0.54 for the view, 0.40 its own tensor).
    public static func pageRow(_ n: Int, _ type: NDArray.ScalarType) -> Int { Arena.aligned(n * element(type)) / element(type) }

    /// The bytes of `shape` laid as `padded`.
    public static func paddedBytes(_ shape: [Int], _ type: NDArray.ScalarType) -> Int {
        shape.isEmpty ? element(type) : padded(shape, type)[0] * shape[0] * element(type)
    }

    /// The bytes of one element of `type` (a sub-byte type counted as one: none is an argument here).
    public static func element(_ type: NDArray.ScalarType) -> Int {
        switch type {
        case .float32, .int32: 4
        case .float16, .bfloat16, .int16: 2
        default: 1
        }
    }
}
