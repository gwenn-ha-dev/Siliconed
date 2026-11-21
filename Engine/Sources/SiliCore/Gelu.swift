/// GELU, one primitive as coreai-torch writes `F.gelu` (SiliCore's `CoreAIElementwise.swift`, ae45f2e): its attribute,
/// of kind 19 and one field, says the tanh form (1, `approximate="tanh"`) or the exact one (0, erf).
extension Aicode.Block {
    public func gelu(_ x: Int, tanh: Bool) -> Int {
        add(Aicode.Op("primitive.gelu_v1", operands: [x], results: [types[x]],
                      properties: [.attribute(Aicode.Entry(19, [.int(tanh ? 1 : 0)]))]))[0]
    }
}
