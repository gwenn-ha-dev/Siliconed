import Foundation
import Testing
@testable import SiliCore

// MARK: - The Quantized family's bytecode, byte for byte against coreai-torch's

/// Each reference rebuilt with the family's builders (`tools/coreai/references_quantized.py`): the arguments are named
/// as the Python modules name theirs (`w`, `w_scale`, `x`, `x_scale`), the outputs as coreai-torch names them.
private func quantizedReference(_ name: String) -> [UInt8] {
    let T = Aicode.tensor
    let parts = name.split(separator: "_").map(String.init)
    let kind = parts[0], group = parts[1]
    func element(_ tag: String) -> Aicode.Element { tag == "bf16" ? .bf16 : tag == "f32" ? .f32 : .f16 }
    let e = element(parts[2])
    let dims = parts[3...].compactMap { Int($0) }
    var constants = Aicode.ScalingConstants()
    func module(_ args: [(String, [Int], Aicode.Element)], output: String, _ build: (Aicode.Block) -> Int) -> [UInt8] {
        var m = Aicode.Module(arguments: args.map { ($0.0, T($0.1, $0.2)) })
        m.outputs = [(output, build(m.body))]
        return m.bytecode()
    }
    // The group names the axis of the 2-D references; the scale's shape follows it. `tensor` is one element, `tensor0d` none.
    func scales(_ shape: [Int]) -> (axis: Int, shape: [Int]) {
        switch group {
        case "col": (shape.count - 1, [shape[shape.count - 1]])
        case "row": (0, [shape[0]])
        case "tensor": (0, [1])
        case "tensor0d": (0, [])
        case "axis2": (2, [shape[2]])
        default: preconditionFailure(name)
        }
    }
    switch kind {
    case "dequantize":
        let (axis, s) = scales(dims)
        return module([("w", dims, .si8), ("w_scale", s, e)], output: "dequantize") { $0.dequantize(0, scale: 1, axis: axis, constants: &constants) }
    case "quantize":
        let (axis, s) = scales(dims)
        return module([("x", dims, e), ("x_scale", s, e)], output: "quantize") { $0.quantize(0, scale: 1, axis: axis, constants: &constants) }
    case "roundtrip":
        let (axis, s) = scales(dims)
        return module([("x", dims, e), ("x_scale", s, e)], output: "dequantize") {
            $0.dequantize($0.quantize(0, scale: 1, axis: axis, constants: &constants), scale: 1, axis: axis, constants: &constants)
        }
    default: preconditionFailure(name)
    }
}

/// The references' names carry what they vary: `<op>_<group>_<element>_<dims>`. A dequantize by one scale only, as the
/// engine writes it (`Aicode.Block.dequantize`); a quantize by one or a vector.
@Test(arguments: coreAIQuantizedReferences.keys.sorted())
func coreAIQuantizedBytecodeEqualsTheReference(_ name: String) throws {
    let expected = [UInt8](Data(base64Encoded: coreAIQuantizedReferences[name]!)!)
    let written = quantizedReference(name)
    let first = (0..<min(expected.count, written.count)).first { expected[$0] != written[$0] } ?? min(expected.count, written.count)
    #expect(written == expected, "\(name): \(written.count) bytes against \(expected.count), first difference at \(first)")
}
