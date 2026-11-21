/// The convolution, as coreai-torch writes `aten.convolution`. Recopied from SiliCore (`CoreAIConvolution.swift`,
/// aec8f3c), without its graph's lowering: a function calls `convolution` on its block.
///
///     x [N, C, *S] ⊛ w [O, C/g, *K]    constants · pad_v1 (zeros, if any padding) · conv2d_v1 | conv3d_v1
///     1-D: x [N, C, L]                  reshape_v2 x to [N, C, 1, L] · reshape_v2 w to [O, C/g, 1, K] · pad · conv2d_v1
///                                         · reshape_v2 back to [N, O, L']
///
/// What a port by analogy would miss (SiliCore's findings):
///   - **the padding is not the convolution's**: Core AI's convolutions take no padding, so coreai-torch writes a
///     separate `pad_v1` (a mode attribute, a zero as a rank-0 constant of the element, the amounts: two per axis,
///     the batch and channel axes first) and convolves the padded tensor; a padding of 0 on every axis writes no pad;
///   - **a 1-D convolution is a 2-D one**: there is no `conv1d`; both operands gain a unit axis before the length
///     (a reshape with a ui32 shape constant) and the result loses it;
///   - **the constants come first, in an order that depends on the ops that follow**: stride, dilation, then groups
///     (a rank-0 ui32) with no pad; groups, dilation, stride, the zero and the amounts with one; a 1-D convolution the
///     output shape, the weight's shape, then groups, dilation, stride (then the amounts, the zero) and the input's
///     shape last. Equal data and type is one constant (stride 1 and dilation 1 share theirs);
///   - **dates**: conv2d 2025-09-05, pad 2025-09-24, conv3d 2025-10-23; the asset's is the latest of its ops.
///
/// No transposed convolution and no bias: a bias is the add after it (in f32, under the guard of f16).
extension Aicode {
    /// The mode attribute of `pad_v1`: constant.
    static let constantPadding = Entry(17, [.int(1)])
}

extension Aicode.Block {
    /// `F.conv1d/2d/3d(x, w, stride, padding, dilation, groups)` with no bias, as coreai-torch writes it. `padding`: one
    /// amount per spatial axis, on both of its sides (a band of rows pads its columns alone: `[0, 1]`).
    public func convolution(_ x: Int, _ w: Int, stride: [Int], padding: [Int], dilation: [Int], groups: Int, result: Aicode.Entry) -> Int {
        let (xs, element) = Aicode.shape(types[x]), ws = Aicode.shape(types[w]).shape, out = Aicode.shape(result).shape
        let one = xs.count == 3
        var cache: [[Aicode.Entry]: Int] = [:]
        func shared(_ data: Aicode.Entry, _ type: Aicode.Entry) -> Int {
            if let v = cache[[data, type]] { return v }
            let v = constant(data, type)
            cache[[data, type]] = v
            return v
        }
        func vector(_ v: [Int]) -> Int { shared(Aicode.dense(v.map { UInt32($0) }), Aicode.tensor([v.count], .ui32)) }
        func groupsValue() -> Int { shared(Aicode.dense([UInt32(groups)]), Aicode.tensor([], .ui32)) }
        func zero() -> Int { shared(Aicode.dense(Aicode.scalar(0, element), splat: true), Aicode.tensor([], element)) }
        // 1-D: a unit axis before the length; the stride and dilation gain a 1 before theirs.
        let s = one ? [1] + stride : stride, d = one ? [1] + dilation : dilation
        let padded = padding.contains { $0 != 0 }
        let amounts = [0, 0, 0, 0] + (one ? [0, 0] : []) + padding.flatMap { [$0, $0] }

        var xin = x, win = w, outShape = 0, g = 0, dv = 0, sv = 0, zeroValue = 0, pads = 0
        if one {
            outShape = vector(out)
            let wShape = vector([ws[0], ws[1], 1, ws[2]])
            g = groupsValue(); dv = vector(d); sv = vector(s)
            if padded { pads = vector(amounts); zeroValue = zero() }
            let xShape = vector([xs[0], xs[1], 1, xs[2]])
            xin = add(Aicode.Op("primitive.reshape_v2", operands: [x, xShape], results: [Aicode.tensor([xs[0], xs[1], 1, xs[2]], element)]))[0]
            win = add(Aicode.Op("primitive.reshape_v2", operands: [w, wShape], results: [Aicode.tensor([ws[0], ws[1], 1, ws[2]], element)]))[0]
        } else if padded {
            g = groupsValue(); dv = vector(d); sv = vector(s); zeroValue = zero(); pads = vector(amounts)
        } else {
            sv = vector(s); dv = vector(d); g = groupsValue()
        }
        if padded {
            let shape = Aicode.shape(types[xin]).shape
            let grown = shape.indices.map { $0 < 2 ? shape[$0] : shape[$0] + 2 * (one ? ($0 == 2 ? 0 : padding[0]) : padding[$0 - 2]) }
            xin = add(Aicode.Op("primitive.pad_v1", operands: [xin, pads, zeroValue], results: [Aicode.tensor(grown, element)],
                                properties: [.attribute(Aicode.constantPadding)]))[0]
        }
        let kernel = "primitive.conv\(one ? 2 : xs.count - 2)d_v1"
        guard one else { return add(Aicode.Op(kernel, operands: [xin, win, sv, dv, g], results: [result]))[0] }
        let y = add(Aicode.Op(kernel, operands: [xin, win, sv, dv, g], results: [Aicode.tensor([out[0], out[1], 1, out[2]], element)]))[0]
        return add(Aicode.Op("primitive.reshape_v2", operands: [y, outShape], results: [result]))[0]
    }
}
