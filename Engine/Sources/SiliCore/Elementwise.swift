/// One-operand primitives, a constant's broadcast and the sigmoid, as the former engine wrote them (`CoreAIElementwise.swift`,
/// 0.21): what the guard of f16 computes a row's power of two with (`abs`, `log`, `round`, `pow`, `maximum`).
///
///     primitive   primitive.<name>_v1, its type kept (abs, log, round, sqrt, …)
///     scale       broadcasting_<name>_v1 of x and a rank-0 constant of x's element: how coreai-torch writes `x * c`
///     sigmoid     sigmoid_v1 ⊃ { 1 / (1 + exp(0 − x)) }, the scalars one-element tensors of x's element
///
/// What a port by analogy would miss (the former engine's findings): **`round_v1` rounds a half away from zero**, not to even.
extension Aicode.Block {
    /// One primitive over `x`, its type kept: `primitive.<name>_v1`.
    public func primitive(_ name: String, _ x: Int) -> Int {
        add(Aicode.Op("primitive.\(name)_v1", operands: [x], results: [types[x]]))[0]
    }

    /// `x ∘ c` (or `c ∘ x`) with `c` a rank-0 constant of x's element, rounded once to it.
    public func scale(_ name: String, _ x: Int, _ c: Double, scalarFirst: Bool = false) -> Int {
        let element = Aicode.shape(types[x]).element
        let k = constant(Aicode.dense(Aicode.scalar(c, element), splat: true), Aicode.tensor([], element))
        return scalarFirst ? broadcasting(name, k, x) : broadcasting(name, x, k)
    }

    /// An elementwise op of two operands by its name (`maximum`, `minimum`, `pow`, …), numpy's broadcasting.
    public func elementwise(_ name: String, _ a: Int, _ b: Int) -> Int { broadcasting(name, a, b) }

    /// `1 / (1 + exp(0 − x))`: ±∞ gives 1 and 0, never a NaN.
    public func sigmoid(_ x: Int) -> Int { logistic("sigmoid_v1", x) { _, r in r } }

    /// **A composite op of `x` and its sigmoid** `r = 1 / (1 + exp(0 − x))`, the scalars one-element tensors of x's
    /// element: its body yields `yield(body, r)`, x the body's argument 0 (`sigmoid`, `silu`).
    func logistic(_ name: String, _ x: Int, yield: (Aicode.Block, Int) -> Int) -> Int {
        let element = Aicode.shape(types[x]).element
        let body = Aicode.Block([types[x]])
        let one = Aicode.tensor([1], element)
        let c0 = body.constant(Aicode.dense(Aicode.scalar(0, element), splat: true), one)
        let c1 = body.constant(Aicode.dense(Aicode.scalar(1, element), splat: true), one)
        let e = body.exp(body.binary(.sub, c0, 0))
        body.yield(yield(body, body.binary(.divide, c1, body.binary(.add, c1, e))))
        return add(Aicode.Op(name, operands: [x], results: [types[x]], body: body))[0]
    }
}

/// **The core composites, decomposed as coreai-torch decomposes them**: a constant, a product, `linear`, numpy's
/// broadcasting of two operands, a cast, a transposition, `exp`, SiLU.
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
    public func silu(_ x: Int) -> Int { logistic("silu_v1", x) { body, r in body.binary(.mul, 0, r) } }
}
