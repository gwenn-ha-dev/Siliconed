/// One-operand primitives, a constant's broadcast and the sigmoid, as SiliCore writes them (`CoreAIElementwise.swift`,
/// aec8f3c): what the guard of f16 computes a row's power of two with (`abs`, `log`, `round`, `pow`, `maximum`).
///
///     primitive   primitive.<name>_v1, its type kept (abs, log, round, sqrt, …)
///     scale       broadcasting_<name>_v1 of x and a rank-0 constant of x's element: how coreai-torch writes `x * c`
///     sigmoid     sigmoid_v1 ⊃ { 1 / (1 + exp(0 − x)) }, the scalars one-element tensors of x's element
///
/// What a port by analogy would miss (SiliCore's findings): **`round_v1` rounds a half away from zero**, not to even.
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
    public func sigmoid(_ x: Int) -> Int {
        let element = Aicode.shape(types[x]).element
        let body = Aicode.Block([types[x]])
        let one = Aicode.tensor([1], element)
        let c0 = body.constant(Aicode.dense(Aicode.scalar(0, element), splat: true), one)
        let c1 = body.constant(Aicode.dense(Aicode.scalar(1, element), splat: true), one)
        let e = body.exp(body.binary(.sub, c0, 0))
        body.yield(body.binary(.divide, c1, body.binary(.add, c1, e)))
        return add(Aicode.Op("sigmoid_v1", operands: [x], results: [types[x]], body: body))[0]
    }
}
