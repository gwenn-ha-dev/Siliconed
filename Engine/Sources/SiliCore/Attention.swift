/// Attention, called as coreai-torch's `scaled_dot_product_attention` composite. Recopied from SiliCore
/// (`CoreAIAttention.swift`, aec8f3c), without the causal mask (the attention written here is bidirectional): the mask is an
/// input, additive, the keys past the sequence's length masked out at a fixed length of keys.
///
///     asset:  graph sdpa_… (private, composite_decl)         graph main
///               q·scale ─ ·kᵀ ─ [+ mask] ─ softmax ─ ·v         invoke sdpa_… (q, k, v [, mask]) ─ output
///
/// What a port by analogy would miss (SiliCore's findings):
///   - **the composite is a second graph, not a region**: `silu_v1` and the products carry their decomposition inline,
///     attention is a private graph beside `main`, reached by `dataflow.invoke_v1`. The call carries the graph as its
///     body until `Module.bytecode()` hoists it beside `main`;
///   - **the softmax sums in f32** (a conversion of each element, the sum typed as its input), and the scale multiplies q
///     before the product with kᵀ, so that the scores keep f16's range.

extension Aicode {
    private static let causalFlag = Entry(2, [.int(1), .byte(0), .byte(0)])
    private static func si64(_ v: Int64) -> Entry { Entry(2, [.int(64), .byte(0), .signed(v)]) }
    static let symbolReferenceType = Entry(33)

    /// `composite_decl`: the composite's name, its input names, its attributes (a float32 scale, as torch passes it) and
    /// output name. `is_causal` stays false.
    static func attentionDeclaration(masked: Bool, scale: Float) -> Entry {
        let inputs = ["query", "key", "value"] + (masked ? ["attn_mask"] : [])
        let scaleAttribute = Entry(3, [.int(32), .signed(Int64(Int32(bitPattern: scale.bitPattern)))])
        let attributes = Entry(45, [.int(4), .attribute(named("is_causal", causalFlag)), .attribute(named("scale", scaleAttribute)),
                                    .attribute(named("version", si64(1))), .attribute(named("window_size", si64(0)))])
        return Entry(9, [.string("scaled_dot_product_attention"), .int(3),
                         .attribute(named("input_names", array(inputs.map(string)))), .attribute(named("op_attrs", attributes)),
                         .attribute(named("output_names", array([string("output")])))])
    }
}

extension Aicode.Block {
    /// `softmax(q · scale · kᵀ [+ mask]) · v` over `[batch, heads, rows, dim]` operands: a call to the composite, its graph
    /// the call's body. `ins`: q, k, v, and the additive mask when there is one (`[rows, keys]`, broadcast over heads).
    public func attention(_ ins: [Int], scale: Double, symbol: String) -> Int {
        let (q, element) = Aicode.shape(types[ins[0]])
        let v = Aicode.shape(types[ins[2]]).shape
        let result = Aicode.tensor(Array(q.dropLast()) + [v.last!], element)
        let graph = Aicode.Block.attentionGraph(arguments: ins.map { types[$0] }, scale: scale, result: result)
        let callee = Aicode.Entry(16, [.attribute(Aicode.Entry(14, [.int(1), .string(symbol)])), .type(Aicode.symbolReferenceType)])
        let call = Aicode.Op("dataflow.invoke_v1", operands: ins, results: [result],
                             properties: [.attribute(Aicode.array([])), .attribute(callee), .attribute(Aicode.array([])),
                                          .attribute(Aicode.attentionDeclaration(masked: ins.count > 3, scale: Float(scale)))],
                             body: graph)
        return add(call)[0]
    }

    /// `x * s` for a rank-0 `s`, as coreai-torch writes `x * 0.5`: `broadcasting_mul` of x and a `tensor<f>`, whose
    /// region reshapes the scalar to x's rank (ones) and broadcasts it along the axes longer than 1.
    func scaled(_ x: Int, by s: Int) -> Int {
        let result = types[x]
        let (dims, element) = Aicode.shape(result)
        let body = Aicode.Block([types[x], types[s]])
        let ones = Array(repeating: 1, count: dims.count)
        let shape = body.constant(Aicode.dense(ones.map { Int32($0) }), Aicode.tensor([dims.count], .si32))
        let reshaped = body.add(Aicode.Op("primitive.reshape_v2", operands: [1, shape], results: [Aicode.tensor(ones, element)]))[0]
        let along = dims.indices.filter { dims[$0] != 1 }
        var spread = reshaped
        if !along.isEmpty {
            let sizes = body.constant(Aicode.dense(along.map { UInt32(dims[$0]) }), Aicode.tensor([along.count], .ui32))
            let axes = body.constant(Aicode.dense(along.map { Int32($0) }), Aicode.tensor([along.count], .si32))
            spread = body.add(Aicode.Op("primitive.broadcast_in_dims_v1", operands: [reshaped, sizes, axes], results: [result]))[0]
        }
        let product = body.add(Aicode.Op("primitive.mul_v1", operands: [0, spread], results: [result]))[0]
        body.yield(product)
        return add(Aicode.Op("broadcasting_mul_v1", operands: [x, s], results: [result], body: body))[0]
    }

    /// `exp(x) / sum(exp(x))` over the last axis: the softmax inside the attention composite, whose region reduces in f32
    /// (a conversion of each element) and types the sum as x — the reduction's own result dynamic, as it declares it.
    func attentionSoftmax(_ x: Int, axis: Int) -> Int {
        let (dims, element) = Aicode.shape(types[x])
        let type = types[x]
        let body = Aicode.Block([type, types[axis]])
        let one = Aicode.tensor([1], .si32)
        let ones = body.constant(Aicode.dense([Int32(1)]), one)
        let axes = body.add(Aicode.Op("primitive.reshape_v2", operands: [1, ones], results: [one]))[0]
        let e = body.exp(0)

        let sum = Aicode.Block([type, one])
        let wide = Aicode.tensor([], .f32)
        // The accumulator starts at an f32 zero; for bf16 torch writes the zero in bf16 and converts it.
        var zero = sum.constant(Aicode.dense(Aicode.scalar(0, element == .bf16 ? .bf16 : .f32), splat: true), element == .bf16 ? Aicode.tensor([], .bf16) : wide)
        if element == .bf16 { zero = sum.add(Aicode.Op("primitive.cast_v1", operands: [zero], results: [wide]))[0] }
        let step = Aicode.Block([wide, Aicode.tensor([], element)])
        let converted = element == .f32 ? 1 : step.add(Aicode.Op("primitive.cast_v1", operands: [1], results: [wide]))[0]
        let added = step.add(Aicode.Op("primitive.add_v1", operands: [0, converted], results: [wide]))[0]
        step.yield(added)
        let dynamic = Aicode.Entry(2, [.int(UInt64(dims.count))] + dims.map { _ in Aicode.Field.signed(Int64.min) } + [.type(Aicode.element(element))])
        let reduced = sum.add(Aicode.Op("primitive.reduce_v1", operands: [0, 1, zero], results: [dynamic], body: step))[0]
        sum.yield(reduced)
        let total = body.add(Aicode.Op("reduce_sum_v1", operands: [e, axes], results: [type], body: sum))[0]

        let ratio = body.binary(.divide, e, total)
        body.yield(ratio)
        return add(Aicode.Op("softmax_v1", operands: [x, axis], results: [type], body: body))[0]
    }

    /// The composite's graph: `softmax(q * scale . k^T + mask) . v`, the mask optional, as `_sdpa_decompose` writes it.
    /// Its constants come first.
    static func attentionGraph(arguments: [Aicode.Entry], scale: Double, result: Aicode.Entry) -> Aicode.Block {
        let body = Aicode.Block(arguments)
        let (q, element) = Aicode.shape(arguments[0])
        let axis = body.constant(Aicode.dense([Int32(q.count - 1)]), Aicode.tensor([], .si32))
        let perm = Array(0..<q.count - 2) + [q.count - 1, q.count - 2]
        let permutation = body.constant(Aicode.dense(perm.map { UInt32($0) }), Aicode.tensor([q.count], .ui32))
        let factor = body.constant(Aicode.dense(Aicode.roundedScalar(scale, element), splat: true), Aicode.tensor([], element))
        let scaled = body.scaled(0, by: factor)
        let (k, _) = Aicode.shape(arguments[1])
        let keys = body.add(Aicode.Op("primitive.transpose_v1", operands: [1, permutation], results: [Aicode.tensor(perm.map { k[$0] }, element)]))[0]
        let scoresType = Aicode.tensor(Array(q.dropLast()) + [k[k.count - 2]], element)
        var scores = body.matmul(scaled, keys, result: scoresType)
        if arguments.count > 3 { scores = body.binary(.add, scores, 3) }
        let weights = body.attentionSoftmax(scores, axis: axis)
        let out = body.matmul(weights, 2, result: result)
        body.add(Aicode.Op("dataflow.output_v1", operands: [out]))
        return body
    }
}
