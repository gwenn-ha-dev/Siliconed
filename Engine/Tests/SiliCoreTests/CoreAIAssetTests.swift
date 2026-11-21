import Foundation
import Testing
@testable import SiliCore

// MARK: - The engine's Core AI bytecode, byte for byte against coreai-torch's

/// Each module as coreai-torch exports it (`tools/coreai/references.py`), built with the engine's ops.
private func reference(_ name: String) -> [UInt8] {
    let T = Aicode.tensor
    func module(_ args: [(String, [Int], Aicode.Element)], output: String, _ build: (Aicode.Block) -> Int) -> [UInt8] {
        var m = Aicode.Module(arguments: args.map { ($0.0, T($0.1, $0.2)) })
        m.outputs = [(output, build(m.body))]
        return m.bytecode()
    }
    switch name {
    case "product_3_5_7": return module([("x", [3, 5], .f16), ("w", [5, 7], .f16)], output: "mm") { $0.matmul(0, 1, result: T([3, 7], .f16)) }
    case "product_256_512_512": return module([("x", [256, 512], .f16), ("w", [512, 512], .f16)], output: "mm") { $0.matmul(0, 1, result: T([256, 512], .f16)) }
    case "product_f32_3_5_7": return module([("x", [3, 5], .f32), ("w", [5, 7], .f32)], output: "mm") { $0.matmul(0, 1, result: T([3, 7], .f32)) }
    case "affine_3_5_7":
        return module([("x", [3, 5], .f16), ("w", [5, 7], .f16), ("b", [7], .f16)], output: "add") { $0.binary(.add, $0.matmul(0, 1, result: T([3, 7], .f16)), 2) }
    case "silu_3_5": return module([("x", [3, 5], .f16)], output: "silu") { $0.silu(0) }
    case "exp_3_5": return module([("x", [3, 5], .f16)], output: "exp") { $0.exp(0) }
    case "mul_3_5": return module([("a", [3, 5], .f16), ("b", [3, 5], .f16)], output: "mul") { $0.binary(.mul, 0, 1) }
    case "add_3_5": return module([("a", [3, 5], .f16), ("b", [3, 5], .f16)], output: "add") { $0.binary(.add, 0, 1) }
    case "sub_3_5": return module([("a", [3, 5], .f16), ("b", [3, 5], .f16)], output: "sub") { $0.binary(.sub, 0, 1) }
    case "div_3_5": return module([("a", [3, 5], .f16), ("b", [3, 5], .f16)], output: "div") { $0.binary(.divide, 0, 1) }
    case "add_bf16_3_5": return module([("a", [3, 5], .bf16), ("b", [3, 5], .bf16)], output: "add") { $0.binary(.add, 0, 1) }
    case "add_i32_3_5": return module([("a", [3, 5], .si32), ("b", [3, 5], .si32)], output: "add") { $0.binary(.add, 0, 1) }
    case "add_both_3_1_1_5": return module([("a", [3, 1], .f16), ("b", [1, 5], .f16)], output: "add") { $0.binary(.add, 0, 1) }
    case "linear_3_5_7": return module([("x", [3, 5], .f16), ("w", [7, 5], .f16)], output: "mm") { $0.matmul(0, $0.transpose(1, [1, 0]), result: T([3, 7], .f16)) }
    case "transpose_7_5": return module([("w", [7, 5], .f16)], output: "permute") { $0.transpose(0, [1, 0]) }
    case "cast_bf16_f16_3_5": return module([("x", [3, 5], .bf16)], output: "_to_copy") { $0.cast(0, to: .f16) }
    case "cast_f16_f32_3_5": return module([("x", [3, 5], .f16)], output: "_to_copy") { $0.cast(0, to: .f32) }
    case "block_4_8_16":
        return module([("x", [4, 8], .f16), ("w1", [8, 16], .f16), ("w3", [8, 16], .f16), ("w2", [16, 8], .f16)], output: "mm_2") { b in
            let gate = b.silu(b.matmul(0, 1, result: T([4, 16], .f16)))
            return b.matmul(b.binary(.mul, gate, b.matmul(0, 2, result: T([4, 16], .f16))), 3, result: T([4, 8], .f16))
        }
    case "constant_3_5_7", "constant_3_64_64", "constant_f32_3_1_256", "constants_3_5_7_3":
        // torch.arange(k·n) / scale in the weight's dtype, as `references.py` builds it.
        func f16(_ n: Int, _ scale: Float) -> [UInt8] { (0..<n).flatMap { i in withUnsafeBytes(of: (Float16(Float(i)) / Float16(scale)).bitPattern.littleEndian, Array.init) } }
        func f32(_ n: Int, _ scale: Float) -> [UInt8] { (0..<n).flatMap { i in withUnsafeBytes(of: (Float(i) / scale).bitPattern.littleEndian, Array.init) } }
        let (k, n, e, keys): (Int, Int, Aicode.Element, [String]) = switch name {
        case "constant_3_5_7": (5, 7, .f16, ["w"])
        case "constant_3_64_64": (64, 64, .f16, ["resource_14104981350352482762"])
        case "constant_f32_3_1_256": (1, 256, .f32, ["resource_11366756580192985473"])
        default: (5, 7, .f16, ["w1", "w2"])
        }
        var m = Aicode.Module(arguments: [("x", T([3, k], e))])
        let w = m.constant(keys[0], e == .f16 ? f16(k * n, 64) : f32(k * n, 64), T([k, n], e))
        if name == "constants_3_5_7_3" {
            let w2 = m.constant(keys[1], f16(21, 32), T([7, 3], .f16))
            m.outputs = [("mm_1", m.body.matmul(m.body.matmul(0, w, result: T([3, 7], .f16)), w2, result: T([3, 3], .f16)))]
        } else {
            m.outputs = [("mm", m.body.matmul(0, w, result: T([3, n], e)))]
        }
        return m.bytecode()
    default: preconditionFailure(name)
    }
}

@Test(arguments: coreAIReferences.keys.sorted())
func coreAIBytecodeEqualsTheReference(_ name: String) throws {
    let expected = [UInt8](Data(base64Encoded: coreAIReferences[name]!)!)
    let written = reference(name)
    let first = (0..<min(expected.count, written.count)).first { expected[$0] != written[$0] } ?? min(expected.count, written.count)
    #expect(written == expected, "\(name): \(written.count) bytes against \(expected.count), first difference at \(first)")
}

@Test func bytecodeVarintsAreMLIRs() {
    func enc(_ v: UInt64) -> [UInt8] { var e = BytecodeEmitter(); e.varint(v); return e.bytes }
    #expect(enc(0) == [0x01])
    #expect(enc(57) == [0x73])
    #expect(enc(179) == [0xce, 0x02])
    #expect(enc(UInt64(1) << 56) == [0x00, 0, 0, 0, 0, 0, 0, 0, 1])
    var s = BytecodeEmitter(); s.signed(-1); s.signed(3)
    #expect(s.bytes == [0x03, 0x0d])
}

/// A block at rows of a multiple of 64 bytes, its weights arguments: coreai-torch's, byte for byte
/// (`tools/coreai/references_given.py`).
@Test func aBlockOfWideRowsEqualsTheReference() {
    let T = Aicode.tensor
    var m = Aicode.Module(arguments: [("x", T([4, 32], .f16)), ("w1", T([32, 64], .f16)), ("w3", T([32, 64], .f16)), ("w2", T([64, 32], .f16))])
    let b = m.body
    let gate = b.silu(b.matmul(0, 1, result: T([4, 64], .f16)))
    m.outputs = [("mm_2", b.matmul(b.binary(.mul, gate, b.matmul(0, 2, result: T([4, 64], .f16))), 3, result: T([4, 32], .f16)))]
    #expect(m.bytecode() == [UInt8](Data(base64Encoded: coreAIGivenReferences["block_4_32_64"]!)!))
}

/// An f16 or bf16 operand of an f32 op converted first, as coreai-torch writes `x + y.float()`. (Its linear, whose
/// permutation coreai-torch hoists above the conversion, is the same ops in another order: not compared.)
@Test(arguments: ["widen_add_bf16_3_5_5", "widen_mul_f16_3_5", "widen_matmul_f16_3_5_7"])
func coreAIWideningEqualsTheReference(_ name: String) {
    let T = Aicode.tensor
    let y: ([Int], Aicode.Element) = switch name {
    case "widen_add_bf16_3_5_5": ([5], .bf16)
    case "widen_mul_f16_3_5": ([3, 5], .f16)
    default: ([5, 7], .f16)
    }
    var m = Aicode.Module(arguments: [("x", T([3, 5], .f32)), (name.contains("matmul") ? "w" : "y", T(y.0, y.1))])
    let b = m.body
    let wide = b.cast(1, to: .f32)
    m.outputs = switch name {
    case "widen_add_bf16_3_5_5": [("add", b.binary(.add, 0, wide))]
    case "widen_mul_f16_3_5": [("mul", b.binary(.mul, 0, wide))]
    default: [("mm", b.matmul(0, wide, result: T([3, 7], .f32)))]
    }
    #expect(m.bytecode() == [UInt8](Data(base64Encoded: coreAIWideningReferences[name]!)!), "\(name)")
}
