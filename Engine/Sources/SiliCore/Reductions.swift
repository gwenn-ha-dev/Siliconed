/// The RMS norm and what it is made of (sum, mean, square, rsqrt), each written as coreai-torch writes it. Recopied from
/// the former engine (`CoreAIReductions.swift`, 0.21).
///
///     reduce_sum_v1  ⊃ { constant 0 · reduce_v1 ⊃ { cast · add · yield } · yield }          f16 and bf16 add in f32
///     reduce_mean_v1 ⊃ { cast f32 · reduce_sum_v1 · shape ▸ gather_nd ▸ reduce_product (8-byte ints) ▸ cast f32 ·
///                        broadcasting_divide_v1 ⊃ { get_shape · broadcast_shapes · broadcast_to ×2 · divide } · cast }
///     norm           cast f32 · pow 2 · reduce_mean_v1 · add eps · rsqrt · mul · cast
///
/// What a port by analogy would miss (the former engine's findings):
///   - **the reduce's declared result is dynamic** (`?x?`: the input's rank, no sizes) inside the composite: types are
///     copied, never inferred;
///   - **f16 and bf16 are summed in f32 but their zero differs**: f16's initial value is a constant f32, bf16's a bf16
///     zero converted to f32 inside the region;
///   - **a norm is not a composite here**: coreai-torch writes an externalized `RMSNorm` as a graph of its own, `invoke`d by
///     `main`; ours is one graph, so a norm is the composite's own body (statistics in f32, one rounding to the element),
///     which is what the decomposition `F.rms_norm` is;
///   - **the mean's count is the product of the reduced sizes**, taken from the shape at run time through an 8-byte
///     integer (an f16 count would overflow at 65504 elements).

extension Aicode.Block {
    /// A vector of integers; empty is a dense of no bytes (the shape of a scalar), all equal a splat.
    public func integers(_ values: [Int], _ element: Aicode.Element) -> Int {
        let data: Aicode.Entry
        if values.isEmpty { data = Aicode.dense([UInt8](), splat: false) }
        else if element == .si32 { data = Aicode.dense(values.map { Int32($0) }) }
        else { data = Aicode.dense(values.map { UInt32($0) }) }
        return constant(data, Aicode.tensor([values.count], element))
    }

    public func scalar(_ x: Double, _ element: Aicode.Element) -> Int {
        constant(Aicode.dense(Aicode.scalar(x, element), splat: true), Aicode.tensor([], element))
    }

    func reshape(_ x: Int, to shape: Int, result: Aicode.Entry) -> Int {
        add(Aicode.Op("primitive.reshape_v2", operands: [x, shape], results: [result]))[0]
    }

    /// `reduce_sum_v1`: the elements added in f32 whatever their type.
    func reduceSum(_ x: Int, _ axes: Int, result: Aicode.Entry) -> Int {
        let (shape, element) = Aicode.shape(types[x])
        let body = Aicode.Block([types[x], types[axes]])
        let zero: Int
        switch element {
        case .bf16: zero = body.cast(body.scalar(0, .bf16), to: .f32)
        default: zero = body.scalar(0, .f32)
        }
        let step = Aicode.Block([Aicode.tensor([], .f32), Aicode.tensor([], element)])
        let term = element == .f32 ? 1 : step.cast(1, to: .f32)
        step.yield(step.add(Aicode.Op("primitive.add_v1", operands: [0, term], results: [Aicode.tensor([], .f32)]))[0])
        let sum = body.add(Aicode.Op("primitive.reduce_v1", operands: [0, 1, zero], results: [Aicode.dynamic(shape.count, element)], body: step))[0]
        body.yield(sum)
        return add(Aicode.Op("reduce_sum_v1", operands: [x, axes], results: [result], body: body))[0]
    }

    /// `reduce_mean_v1`: the sum in f32 divided by the count of reduced elements, read from the shape at run time.
    func reduceMean(_ x: Int, _ axes: Int, result: Aicode.Entry) -> Int {
        let (shape, element) = Aicode.shape(types[x])
        let rank = shape.count, count = Aicode.shape(types[axes]).shape[0]
        let body = Aicode.Block([types[x], types[axes]])
        let wide = element == .f32 ? 0 : body.cast(0, to: .f32)
        let sum = body.reduceSum(wide, 1, result: Aicode.dynamic(rank, .f32))
        let sizes = body.integers(shape, .ui32)
        let column = body.integers([count, 1], .si32)
        let picked = body.add(Aicode.Op("primitive.gather_nd_v1", operands: [sizes, body.reshape(1, to: column, result: Aicode.tensor([count, 1], .si32))], results: [Aicode.tensor([count], .ui32)]))[0]
        let first = body.integers([0], .si32)
        let integers = body.add(Aicode.Op("primitive.cast_v1", operands: [picked], results: [Aicode.kindTensor([count], Aicode.wideInteger)]))[0]

        let product = Aicode.Block([body.types[integers], body.types[first]])
        let one = product.constant(Aicode.dense([UInt8(1), 0, 0, 0, 0, 0, 0, 0], splat: true), Aicode.kindTensor([], Aicode.wideInteger))
        let step = Aicode.Block([Aicode.kindTensor([], Aicode.wideInteger), Aicode.kindTensor([], Aicode.wideInteger)])
        step.yield(step.add(Aicode.Op("primitive.mul_v1", operands: [0, 1], results: [Aicode.kindTensor([], Aicode.wideInteger)]))[0])
        product.yield(product.add(Aicode.Op("primitive.reduce_v1", operands: [0, 1, one], results: [Aicode.kindTensor([Int.min], Aicode.wideInteger)], body: step))[0])
        let total = body.add(Aicode.Op("reduce_product_v1", operands: [integers, first], results: [Aicode.kindTensor([1], Aicode.wideInteger)], body: product))[0]
        let divisor = body.add(Aicode.Op("primitive.cast_v1", operands: [total], results: [Aicode.tensor([1], .f32)]))[0]

        let dynamic = Aicode.dynamic(rank, .f32)
        let divide = Aicode.Block([dynamic, body.types[divisor]])
        let extent = divide.add(Aicode.Op("primitive.get_shape_v2", operands: [0], results: [Aicode.tensor([rank], .ui32)]))[0]
        let ones = divide.integers([1], .ui32)
        let common = divide.add(Aicode.Op("primitive.broadcast_shapes_v1", operands: [extent, ones], results: [Aicode.tensor([rank], .ui32)]))[0]
        let a = divide.add(Aicode.Op("primitive.broadcast_to_v1", operands: [0, common], results: [dynamic]))[0]
        let b = divide.add(Aicode.Op("primitive.broadcast_to_v1", operands: [1, common], results: [dynamic]))[0]
        divide.yield(divide.add(Aicode.Op("primitive.divide_v1", operands: [a, b], results: [dynamic]))[0])
        let mean = body.add(Aicode.Op("broadcasting_divide_v1", operands: [sum, divisor], results: [dynamic], body: divide))[0]
        body.yield(element == .f32 ? mean : body.cast(mean, to: element))
        return add(Aicode.Op("reduce_mean_v1", operands: [x, axes], results: [result], body: body))[0]
    }

    func rsqrt(_ x: Int) -> Int { add(Aicode.Op("primitive.rsqrt_v1", operands: [x], results: [types[x]]))[0] }

    /// The mean along `axes`, kept at size 1, in f32 (`reduce_mean_v1`): a GroupNorm's statistics over a band.
    public func mean(_ x: Int, axes: [Int]) -> Int {
        let shape = Aicode.shape(types[x]).shape
        let kept = Aicode.tensor(shape.indices.map { axes.contains($0) ? 1 : shape[$0] }, .f32)
        return reduceMean(castUnlessSame(x, to: .f32), integers(axes, .si32), result: kept)
    }

    /// The sum along `axes`, kept at size 1, in f32 (`reduce_sum_v1`).
    public func sum(_ x: Int, axes: [Int]) -> Int {
        let shape = Aicode.shape(types[x]).shape
        let kept = Aicode.tensor(shape.indices.map { axes.contains($0) ? 1 : shape[$0] }, .f32)
        return reduceSum(castUnlessSame(x, to: .f32), integers(axes, .si32), result: kept)
    }

    /// `x ** 2.0` as `broadcasting_pow_v1`: `binary`'s broadcast of a scalar, with `pow` for the op.
    private func square(_ x: Int, _ two: Int) -> Int {
        let (shape, element) = Aicode.shape(types[x])
        let result = Aicode.tensor(shape, element)
        let body = Aicode.Block([types[x], types[two]])
        let padded = body.integers(Array(repeating: 1, count: shape.count), .si32)
        let reshaped = body.reshape(1, to: padded, result: Aicode.tensor(Array(repeating: 1, count: shape.count), element))
        let sizes = body.integers(shape, .ui32)
        let dims = body.integers(Array(0..<shape.count), .si32)
        let wide = body.add(Aicode.Op("primitive.broadcast_in_dims_v1", operands: [reshaped, sizes, dims], results: [result]))[0]
        body.yield(body.add(Aicode.Op("primitive.pow_v1", operands: [0, wide], results: [result]))[0])
        return add(Aicode.Op("broadcasting_pow_v1", operands: [x, two], results: [result], body: body))[0]
    }

    /// `x · rsqrt(mean(x²) + eps)` over `axes`: the statistics in f32, one rounding to x's element at the end
    /// (`F.rms_norm` as coreai-torch decomposes it).
    public func rmsNorm(_ x: Int, axes: [Int], eps: Double) -> Int { rmsNorm(x, axes: axes, eps: eps, epsilon: nil) }

    /// The same norm with its eps a value of the graph (f32, broadcast against the mean's kept shape): a row's own eps,
    /// for a row held under a factor F, whose true norm reads eps / F² (the guard of f16).
    public func rmsNorm(_ x: Int, axes: [Int], epsilon: Int) -> Int { rmsNorm(x, axes: axes, eps: 0, epsilon: epsilon) }

    private func rmsNorm(_ x: Int, axes: [Int], eps: Double, epsilon given: Int?) -> Int {
        let (shape, element) = Aicode.shape(types[x])
        let kept = Aicode.tensor(shape.indices.map { axes.contains($0) ? 1 : shape[$0] }, .f32)
        // The constants are hoisted to the block's start in an order MLIR's folder fixes: f32 (no conversion before
        // them) reversed from their use, f16 and bf16 as measured.
        let two: Int, epsilon: Int, vector: Int
        if element == .f32 { epsilon = given ?? scalar(eps, .f32); vector = integers(axes, .si32); two = scalar(2, .f32) }
        else { two = scalar(2, .f32); epsilon = given ?? scalar(eps, .f32); vector = integers(axes, .si32) }
        let wide = element == .f32 ? x : cast(x, to: .f32)
        let mean = reduceMean(square(wide, two), vector, result: kept)
        let y = broadcasting("mul", wide, rsqrt(broadcasting("add", mean, epsilon)))
        return element == .f32 ? y : cast(y, to: element)
    }

    /// LayerNorm without affine over `axes`: (x − mean) · rsqrt(mean((x − mean)²) + eps), the statistics in f32, one
    /// rounding to x's element at the end; eps a value of the graph (f32), a row's own under the guard of f16.
    public func layerNorm(_ x: Int, axes: [Int], epsilon: Int) -> Int {
        let (shape, element) = Aicode.shape(types[x])
        let kept = Aicode.tensor(shape.indices.map { axes.contains($0) ? 1 : shape[$0] }, .f32)
        let wide = element == .f32 ? x : cast(x, to: .f32)
        let centered = broadcasting("sub", wide, reduceMean(wide, integers(axes, .si32), result: kept))
        let variance = reduceMean(broadcasting("mul", centered, centered), integers(axes, .si32), result: kept)
        let y = broadcasting("mul", centered, rsqrt(broadcasting("add", variance, epsilon)))
        return element == .f32 ? y : cast(y, to: element)
    }

    /// The largest element along `axes`, kept at size 1: `reduce_max_v1`, started at the lowest finite value of the
    /// element (an all-infinite row gives the largest finite, not ∞). Recopied from the former engine (`reduceExtremum`).
    public func reduceMax(_ x: Int, axes: [Int]) -> Int {
        let (shape, element) = Aicode.shape(types[x])
        let vector = integers(axes, .si32)
        let result = Aicode.tensor(shape.indices.map { axes.contains($0) ? 1 : shape[$0] }, element)
        let body = Aicode.Block([types[x], types[vector]])
        let lowest: [UInt8] = switch element {
        case .f32: withUnsafeBytes(of: UInt32(0xff7f_ffff).littleEndian) { Array($0) }
        case .f16: withUnsafeBytes(of: UInt16(0xfbff).littleEndian) { Array($0) }
        default: withUnsafeBytes(of: UInt16(0xff7f).littleEndian) { Array($0) }
        }
        let start = body.constant(Aicode.dense(lowest, splat: true), Aicode.tensor([], element))
        let scalar = Aicode.tensor([], element)
        let step = Aicode.Block([scalar, scalar])
        step.yield(step.add(Aicode.Op("primitive.maximum_v1", operands: [0, 1], results: [scalar]))[0])
        body.yield(body.add(Aicode.Op("primitive.reduce_v1", operands: [0, 1, start], results: [Aicode.dynamic(shape.count, element)], body: step))[0])
        return add(Aicode.Op("reduce_max_v1", operands: [x, vector], results: [result], body: body))[0]
    }
}
