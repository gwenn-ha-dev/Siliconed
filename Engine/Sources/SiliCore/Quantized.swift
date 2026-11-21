/// int8 codes and their scales, read through `coreai.dequantize`: W8A16, the codes widened and scaled in the function,
/// the products in the scale's element. Recopied from SiliCore (`CoreAIQuantized.swift`, aec8f3c), the dequantize alone.
///
///     a weight [out, in] of int8 codes, a scale a row      arguments  w (si8 [out, in])  w_scale (f16 [out])
///       x · wᵀ                                               broadcasting_batch_matmul(x, transpose(dequantize_v2(w, w_scale, 0, 0, axis 0)))
///
/// What a port by analogy would miss (SiliCore's findings):
///   - **`dequantize_v2` is a composite whose region does the work in f32**, as coreai-torch writes it: `(q − zeroPoint) ·
///     scale` in f32 then cast to the scale's element, `+ minval` there; the zero point (si8) and the minval are constants
///     of zeros the operation always takes, the axis a si32 constant. A scale of rank 1 is reshaped to `[1…n…1]` *inside*
///     the region, the shape computed from the input's own dims with an `equal` of an iota against the axis, and broadcast
///     by `broadcast_to` of a dynamically shaped operand;
///   - **the scale's element is the product's**: SiliCore measured W8A16 with the scales read in the activations' element
///     as the form Core AI's GPU fuses (12/11: 1.30 × the W8A8 kernel in bf16; with the product in f32, 4.8 ×);
///   - **a product reading scales that follow its contraction axis is wrong on Core AI's GPU** (it fuses the dequantize
///     into the product: relative error 1 to 1000): a linear's scales follow
///     its output rows, never its inputs.

extension Aicode {
    /// The constants shared between a module's dequantizes (the zeros and the axis), made by their first user.
    public struct ScalingConstants {
        struct Key: Hashable { let data: Entry; let type: Entry }
        var made: [Key: Int] = [:]
        public init() {}
    }
}

extension Aicode.Block {
    /// `broadcasting_<name>_v1` of `a` and an operand `b` of dynamic dimensions (all of them, unknown to the type),
    /// broadcast to `a`'s shape by `broadcast_to` of that shape as a constant.
    func broadcastingDynamic(_ name: String, _ a: Int, _ b: Int) -> Int {
        let (shape, element) = Aicode.shape(types[a])
        let result = Aicode.tensor(shape, element)
        let body = Aicode.Block([types[a], types[b]])
        let sizes = body.constant(Aicode.dense(shape.map { UInt32($0) }), Aicode.tensor([shape.count], .ui32))
        let y = body.add(Aicode.Op("primitive.broadcast_to_v1", operands: [1, sizes], results: [result]))[0]
        let r = body.add(Aicode.Op("primitive.\(name)_v1", operands: [0, y], results: [result]))[0]
        body.yield(r)
        return add(Aicode.Op("broadcasting_\(name)_v1", operands: [a, b], results: [result], body: body))[0]
    }

    func unary(_ name: String, _ x: Int, result: Aicode.Entry? = nil) -> Int {
        add(Aicode.Op("primitive.\(name)_v1", operands: [x], results: [result ?? types[x]]))[0]
    }

    /// The same constant once: the first user creates it.
    private func shared(_ data: Aicode.Entry, _ type: Aicode.Entry, _ constants: inout Aicode.ScalingConstants) -> Int {
        let key = Aicode.ScalingConstants.Key(data: data, type: type)
        if let v = constants.made[key] { return v }
        let v = constant(data, type)
        constants.made[key] = v
        return v
    }

    /// What `coreai.dequantize` takes besides its tensor: the scale (a scale of one element reshaped to rank 0, the
    /// axis then 0), the zero point (si8) and the minval, zeros of the scale's shape, and the axis (si32).
    private func scalingOperands(_ scale: Int, axis: Int, _ constants: inout Aicode.ScalingConstants) -> [Int] {
        let (given, element) = Aicode.shape(types[scale])
        let one = given == [1], shape = one ? [] : given
        let minval = shared(Aicode.dense(Aicode.scalar(0, element), splat: true), Aicode.tensor(shape, element), &constants)
        let zero = shared(Aicode.dense([UInt8(0)], splat: true), Aicode.tensor(shape, .si8), &constants)
        let axisValue = shared(Aicode.dense([Int32(one || shape.isEmpty ? 0 : axis)]), Aicode.tensor([], .si32), &constants)
        var s = scale
        if one {
            let empty = shared(Aicode.dense([UInt8](), splat: false), Aicode.tensor([0], .ui32), &constants)
            s = add(Aicode.Op("primitive.reshape_v2", operands: [scale, empty], results: [Aicode.tensor([], element)]))[0]
        }
        return [s, zero, minval, axisValue]
    }

    /// The shape a vector of scales is reshaped to inside the region: the input's dims, with 1 where the axis is not.
    private func scaleShape(rank: Int, dims shape: [Int], axisArgument: Int) -> Int {
        let dims = constant(Aicode.dense(shape.map { UInt32($0) }), Aicode.tensor([rank], .ui32))
        let iota = constant(Aicode.dense((0..<rank).map { Int32($0) }), Aicode.tensor([rank], .si32))
        let hit = broadcasting("equal", iota, axisArgument, result: .bool)
        let miss = unary("not", hit)
        let hit32 = cast(hit, to: .ui32), miss32 = cast(miss, to: .ui32)
        let ones = constant(Aicode.dense([UInt32](repeating: 1, count: rank)), Aicode.tensor([rank], .ui32))
        let along = binary(.mul, hit32, dims)
        let elsewhere = binary(.mul, miss32, ones)
        return binary(.add, along, elsewhere)
    }

    /// `codes` (si8) scaled: `coreai.dequantize` along `axis` (ignored for a scale of rank 0 or one element), in the
    /// scale's element.
    public func dequantize(_ codes: Int, scale: Int, axis: Int, constants: inout Aicode.ScalingConstants) -> Int {
        let (shape, _) = Aicode.shape(types[codes])
        let element = Aicode.shape(types[scale]).element
        let operands = scalingOperands(scale, axis: axis, &constants)
        let body = Aicode.Block([types[codes]] + operands.map { types[$0] })
        let y: Int
        if Aicode.shape(types[operands[0]]).shape.isEmpty {
            let x32 = body.cast(0, to: .f32), zero32 = body.cast(2, to: .f32), scale32 = body.castUnlessSame(1, to: .f32)
            let scaled = body.binary(.mul, body.binary(.sub, x32, zero32), scale32)
            y = body.binary(.add, body.castUnlessSame(scaled, to: element), 3)
        } else {
            let rank = shape.count
            let sizes = body.scaleShape(rank: rank, dims: shape, axisArgument: 4)
            let dynamic = { (e: Aicode.Element) in Aicode.dynamic(rank, e) }
            func reshape(_ v: Int, _ e: Aicode.Element) -> Int {
                body.add(Aicode.Op("primitive.reshape_v2", operands: [v, sizes], results: [dynamic(e)]))[0]
            }
            let zero = reshape(2, .si8), s = reshape(1, element), minval = reshape(3, element)
            let x32 = body.cast(0, to: .f32), zero32 = body.cast(zero, to: .f32), scale32 = body.castUnlessSame(s, to: .f32)
            let scaled = body.broadcastingDynamic("mul", body.broadcastingDynamic("sub", x32, zero32), scale32)
            y = body.broadcastingDynamic("add", body.castUnlessSame(scaled, to: element), minval)
        }
        body.yield(y)
        return add(Aicode.Op("dequantize_v2", operands: [codes] + operands, results: [Aicode.tensor(shape, element)], body: body))[0]
    }

    /// `x` as si8 codes by `scale`, along `axis`: `clamp(round(x / scale), −128, 127)`, a half rounded away from zero
    /// (`primitive.round_v1`): `coreai.quantize` as SiliCore writes it (`CoreAIQuantized.swift`), the activations' side of
    /// a W8A8 product (coreai-optimization: « activation precisions include INT8 »).
    public func quantize(_ x: Int, scale: Int, axis: Int, constants: inout Aicode.ScalingConstants) -> Int {
        let shape = Aicode.shape(types[x]).shape
        let operands = scalingOperands(scale, axis: axis, &constants)
        let body = Aicode.Block([types[x]] + operands.map { types[$0] })
        let low = Aicode.dense(Aicode.scalar(-128, .f32), splat: true), high = Aicode.dense(Aicode.scalar(127, .f32), splat: true)
        let codes: Int
        if Aicode.shape(types[operands[0]]).shape.isEmpty {
            let x32 = body.castUnlessSame(0, to: .f32), minval32 = body.castUnlessSame(3, to: .f32)
            let rounded = body.unary("round", body.binary(.divide, body.binary(.sub, x32, minval32), body.castUnlessSame(1, to: .f32)))
            let offset = body.binary(.add, rounded, body.cast(2, to: .f32))
            let lo = body.constant(low, Aicode.tensor([], .f32)), hi = body.constant(high, Aicode.tensor([], .f32))
            codes = body.cast(body.broadcasting("minimum", body.broadcasting("maximum", offset, lo, result: .f32), hi, result: .f32), to: .si8)
        } else {
            let rank = shape.count
            let sizes = body.scaleShape(rank: rank, dims: shape, axisArgument: 4)
            func reshape(_ v: Int, _ e: Aicode.Element) -> Int {
                body.add(Aicode.Op("primitive.reshape_v2", operands: [v, sizes], results: [Aicode.dynamic(rank, e)]))[0]
            }
            let scaleElement = Aicode.shape(types[scale]).element
            let zero = reshape(2, .si8), s = reshape(1, scaleElement), minval = reshape(3, scaleElement)
            let x32 = body.castUnlessSame(0, to: .f32), minval32 = body.castUnlessSame(minval, to: .f32)
            let rounded = body.unary("round", body.broadcastingDynamic("divide", body.broadcastingDynamic("sub", x32, minval32), body.castUnlessSame(s, to: .f32)))
            let offset = body.broadcastingDynamic("add", rounded, body.cast(zero, to: .f32))
            let lo = body.constant(low, Aicode.tensor([], .f32)), hi = body.constant(high, Aicode.tensor([], .f32))
            codes = body.cast(body.broadcasting("minimum", body.broadcasting("maximum", offset, lo, result: .f32), hi, result: .f32), to: .si8)
        }
        body.yield(codes)
        return add(Aicode.Op("quantize_v2", operands: [x] + operands, results: [Aicode.tensor(shape, .si8)], body: body))[0]
    }
}
