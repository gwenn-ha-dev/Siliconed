/// Reshape, slice, concat and tanh, each one primitive as the former engine wrote it (`CoreAIShapes.swift`,
/// `CoreAIElementwise.swift`, 0.21), without the constant order of coreai-torch's references: a constant is
/// placed just before its op (valid MLIR, other bytes — the order matters to a byte comparison, not to Core AI).
///
///     reshape   primitive.reshape_v2 (shape: a ui32 constant)
///     slice     primitive.slice_v1 on one axis (start, end, step: si32 vectors of the rank, the other axes whole)
///     concat    primitive.concat_v1 (the axis, an si32 scalar, is operand 0)
extension Aicode.Block {
    public func reshape(_ x: Int, _ shape: [Int]) -> Int {
        let (s, e) = Aicode.shape(types[x])
        if s == shape { return x }
        precondition(s.reduce(1, *) == shape.reduce(1, *), "reshape \(s) to \(shape)")
        return add(Aicode.Op("primitive.reshape_v2", operands: [x, integers(shape, .ui32)], results: [Aicode.tensor(shape, e)]))[0]
    }

    /// `x[start ..< start + size]` along `axis`.
    public func slice(_ x: Int, axis: Int, start: Int, size: Int) -> Int {
        let (s, e) = Aicode.shape(types[x])
        var begin = Array(repeating: 0, count: s.count), end = s, out = s
        begin[axis] = start; end[axis] = start + size; out[axis] = size
        let operands = [integers(begin, .si32), integers(end, .si32), integers(Array(repeating: 1, count: s.count), .si32)]
        return add(Aicode.Op("primitive.slice_v1", operands: [x] + operands, results: [Aicode.tensor(out, e)]))[0]
    }

    /// `x` grown at the end of each axis to `shape`, with zeros (`pad_v1`, constant): an output laid as its readers need
    /// it, the rows of a tensor read one by one each starting a page of its own (`Layout.pageRow`).
    public func zeroPadded(_ x: Int, to shape: [Int]) -> Int {
        let (s, e) = Aicode.shape(types[x])
        if s == shape { return x }
        precondition(s.count == shape.count && zip(s, shape).allSatisfy { $0 <= $1 }, "pad \(s) to \(shape)")
        let pads = constant(Aicode.dense(zip(s, shape).flatMap { [UInt32(0), UInt32($1 - $0)] }), Aicode.tensor([2 * s.count], .ui32))
        let zero = constant(Aicode.dense(Aicode.scalar(0, e), splat: true), Aicode.tensor([], e))
        return add(Aicode.Op("primitive.pad_v1", operands: [x, pads, zero], results: [Aicode.tensor(shape, e)], properties: [.attribute(Aicode.constantPadding)]))[0]
    }

    public func concat(_ xs: [Int], axis: Int) -> Int {
        var (s, e) = Aicode.shape(types[xs[0]])
        s[axis] = xs.reduce(0) { $0 + Aicode.shape(types[$1]).shape[axis] }
        let k = constant(Aicode.dense([Int32(axis)]), Aicode.tensor([], .si32))
        return add(Aicode.Op("primitive.concat_v1", operands: [k] + xs, results: [Aicode.tensor(s, e)]))[0]
    }

    public func tanh(_ x: Int) -> Int { add(Aicode.Op("primitive.tanh_v1", operands: [x], results: [types[x]]))[0] }

    /// A row `[1, n]` repeated `count` times, `[count, n]`: numpy's broadcast against zeros.
    public func repeated(_ row: Int, _ count: Int) -> Int {
        let element = Aicode.shape(types[row]).element
        let zeros = constant(Aicode.dense(Aicode.scalar(0, element), splat: true), Aicode.tensor([count, 1], element))
        return binary(.add, zeros, row)
    }
}
