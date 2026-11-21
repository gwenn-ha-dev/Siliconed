import Foundation
import Testing
@testable import SiliCore

// MARK: - The Shapes family's bytecode, byte for byte against coreai-torch's

/// Each reference of `tools/coreai/references_shapes.py` rebuilt with the engine's builders: reshape and concat. The
/// engine's slice writes its constants just before its op and the other axes' end as their size, not torch's order and
/// `INT_MAX`: valid, other bytes, so not compared; the other ops went with the graph.
private func shapesReference(_ name: String) -> [UInt8] {
    typealias E = Aicode.Element
    /// Arguments named as torch exports them (`a_0`, `a_1`, …) unless named.
    func module(_ shapes: [([Int], E)], names: [String]? = nil, output: String, _ build: (Aicode.Block) -> Int) -> [UInt8] {
        var m = Aicode.Module(arguments: shapes.enumerated().map { (names?[$0.offset] ?? "a_\($0.offset)", Aicode.tensor($0.element.0, $0.element.1)) })
        m.outputs = [(output, build(m.body))]
        return m.bytecode()
    }
    func element(_ suffix: String) -> E { suffix.contains("f32") ? .f32 : suffix.contains("bf16") ? .bf16 : suffix.contains("i32") ? .si32 : .f16 }
    let e = element(name)
    switch name {
    // reshape
    case "reshape_3_5_to_15": return module([([3, 5], .f16)], output: "view") { $0.reshape(0, [15]) }
    case "reshape_2_3_5_to_6_5": return module([([2, 3, 5], .f16)], output: "view") { $0.reshape(0, [6, 5]) }
    case "reshape_2_3_5_to_3_10": return module([([2, 3, 5], .f16)], output: "view") { $0.reshape(0, [3, 10]) }
    case "reshape_f32_3_5_to_5_3", "reshape_bf16_3_5_to_5_3", "reshape_i32_3_5_to_5_3":
        return module([([3, 5], e)], output: "view") { $0.reshape(0, [5, 3]) }
    // concat
    case "concat_3_5_axis0": return module([([3, 5], .f16), ([2, 5], .f16)], output: "cat") { $0.concat([0, 1], axis: 0) }
    case "concat_3_5_axis1": return module([([3, 5], .f16), ([3, 2], .f16)], output: "cat") { $0.concat([0, 1], axis: 1) }
    case "concat3_3_5_axis1": return module([([3, 5], .f16), ([3, 2], .f16), ([3, 1], .f16)], output: "cat") { $0.concat([0, 1, 2], axis: 1) }
    case "concat_2_3_5_axis2": return module([([2, 3, 5], .f16), ([2, 3, 2], .f16)], output: "cat") { $0.concat([0, 1], axis: 2) }
    case "concat_f32_3_5", "concat_bf16_3_5", "concat_i32_3_5": return module([([3, 5], e), ([3, 2], e)], output: "cat") { $0.concat([0, 1], axis: 1) }
    default: preconditionFailure("no builder for \(name)")
    }
}

@Test(arguments: coreAIShapesReferences.keys.sorted())
func coreAIShapesBytecodeEqualsTheReference(_ name: String) throws {
    let expected = [UInt8](Data(base64Encoded: coreAIShapesReferences[name]!)!)
    let written = shapesReference(name)
    let first = (0..<min(expected.count, written.count)).first { expected[$0] != written[$0] } ?? min(expected.count, written.count)
    let same = written == expected
    #expect(same, "\(name): \(written.count) bytes against \(expected.count), first difference at \(first)")
}
