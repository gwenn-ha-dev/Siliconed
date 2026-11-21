import Foundation
import Testing
@testable import SiliCore

// MARK: - The Elementwise family's bytecode, byte for byte against coreai-torch's

/// Each reference as `tools/coreai/references_elementwise.py` writes it, rebuilt with the family's builders.
private func elementwiseReference(_ name: String) -> [UInt8] {
    let T = Aicode.tensor
    let parts = name.split(separator: "_").map(String.init)
    let kind = parts[0]
    let element: Aicode.Element = switch parts.count > 1 ? parts[1] : "" {
    case "bfloat16": .bf16
    case "float32": .f32
    case "int32", "i32": .si32
    default: .f16
    }
    func shape(_ s: String) -> [Int] { s.split(separator: "x").map { Int($0)! } }
    func module(_ args: [(String, [Int], Aicode.Element)], output: String, _ build: (Aicode.Block) -> Int) -> [UInt8] {
        var m = Aicode.Module(arguments: args.map { ($0.0, T($0.1, $0.2)) })
        m.outputs = [(output, build(m.body))]
        return m.bytecode()
    }
    func unary(_ shape: [Int], output: String, _ op: @escaping (Aicode.Block, Int) -> Int) -> [UInt8] {
        module([("x", shape, element)], output: output) { op($0, 0) }
    }
    switch kind {
    case "neg": return unary([3, 5], output: "neg") { $0.scale("mul", $1, -1) }
    case "square": return unary([3, 5], output: "pow_1") { $0.scale("pow", $1, 2) }
    case "reciprocal": return unary([3, 5], output: "reciprocal") { $0.scale("divide", $1, 1, scalarFirst: true) }
    case "abs", "sqrt", "rsqrt", "log", "sin", "cos", "tanh":
        return unary([3, 5], output: kind == "abs" ? "abs_1" : kind) { $0.primitive(kind, $1) }
    case "sigmoid": return unary(parts.last == "1" ? [1] : [3, 5], output: "sigmoid") { $0.sigmoid($1) }
    case "maximum", "minimum", "pow":
        let (a, b) = (shape(parts[2]), shape(parts[3]))
        return module([("x", a, element), ("y", b, element)], output: kind == "pow" ? "pow_1" : kind) { $0.elementwise(kind, 0, 1) }
    default: preconditionFailure(name)
    }
}

@Test(arguments: coreAIElementwiseReferences.keys.sorted())
func coreAIElementwiseBytecodeEqualsTheReference(_ name: String) throws {
    let expected = [UInt8](Data(base64Encoded: coreAIElementwiseReferences[name]!)!)
    let written = elementwiseReference(name)
    let first = (0..<min(expected.count, written.count)).first { expected[$0] != written[$0] } ?? min(expected.count, written.count)
    #expect(written == expected, "\(name): \(written.count) bytes against \(expected.count), first difference at \(first)")
}
