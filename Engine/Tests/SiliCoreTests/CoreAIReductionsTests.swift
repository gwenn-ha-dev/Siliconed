import Foundation
import Testing
@testable import SiliCore

// MARK: - The reductions' Core AI bytecode, byte for byte against coreai-torch's

/// One reference module, read from its name (`tools/coreai/references_reductions.py`):
/// `KIND[_DTYPE]_AXES_(keep|drop)_SHAPE`, `softmax[_AXIS]_DTYPE_SHAPE`, `(rms|layernorminline)_DTYPE_SHAPE[_last2]`.
private struct Reference {
    let kind: String
    let element: Aicode.Element
    let shape: [Int]
    var axes: [Int] = []
    var keep = false
    var last2 = false

    init(_ name: String) {
        var tokens = name.split(separator: "_").map(String.init)
        kind = tokens.removeFirst()
        let dtypes: [String: Aicode.Element] = ["f16": .f16, "bf16": .bf16, "f32": .f32]
        let at = tokens.firstIndex { dtypes[$0] != nil }
        element = at.map { dtypes[tokens[$0]]! } ?? .f16
        if let at { tokens.remove(at: at) }
        last2 = tokens.contains("last2")
        if let cut = tokens.firstIndex(where: { $0 == "keep" || $0 == "drop" }) {
            axes = tokens[..<cut].compactMap { Int($0) }
            keep = tokens[cut] == "keep"
            shape = tokens[(cut + 1)...].compactMap { Int($0) }
        } else {
            // softmax names its axis before the dtype, when it is not the last.
            let cut = at ?? 0
            axes = tokens[..<cut].compactMap { Int($0) }
            shape = tokens[cut...].compactMap { Int($0) }
        }
    }

    var outputName: String {
        switch kind {
        case "sum": "sum_1"
        case "max": "amax"
        case "min": "amin"
        case "softmax": "_softmax"
        case "rms": element == .f32 ? "mul" : "_to_copy_1"
        case "layernorminline": element == .f32 ? "mul_1" : "_to_copy_1"
        default: kind
        }
    }

    /// The axes a norm reduces: `F.rms_norm` names them last first, the layer norm's mean in order.
    var normAxes: [Int] {
        let r = shape.count, n = last2 ? 2 : 1
        return kind == "rms" ? (0..<n).map { r - 1 - $0 } : (0..<n).map { r - n + $0 }
    }
    static let eps = 1e-5
}

/// Each module built with the engine's ops: the RMS norm and the kept maximum (the others went with the graph).
private func written(_ name: String) -> [UInt8] {
    let r = Reference(name)
    func T(_ s: [Int], _ e: Aicode.Element) -> Aicode.Entry { Aicode.tensor(s, e) }
    var m = Aicode.Module(arguments: [("x", T(r.shape, r.element))])
    let b = m.body
    let out: Int
    switch r.kind {
    case "rms": out = b.rmsNorm(0, axes: r.normAxes, eps: Reference.eps)
    case "max" where r.keep: out = b.reduceMax(0, axes: r.axes)
    default: preconditionFailure("\(name): no such reduction in the engine")
    }
    m.outputs = [(r.outputName, out)]
    return m.bytecode()
}

@Test(arguments: coreAIReductionsReferences.keys.sorted())
func coreAIReductionsBytecodeEqualsTheReference(_ name: String) {
    let expected = [UInt8](Data(base64Encoded: coreAIReductionsReferences[name]!)!)
    let bytes = written(name)
    let first = (0..<min(expected.count, bytes.count)).first { expected[$0] != bytes[$0] } ?? min(expected.count, bytes.count)
    let equal = bytes == expected
    #expect(equal, "\(name): \(bytes.count) bytes against \(expected.count), first difference at \(first)")
}
