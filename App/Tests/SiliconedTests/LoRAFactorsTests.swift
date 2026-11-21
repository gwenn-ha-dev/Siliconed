import CoreAI
import Foundation
import Metal
import SiliCore
import XCTest
@testable import Siliconed

/// **A stack's factors as a family's function reads them** (`LoRAStack.read`): each module landing on its product side
/// by side along the rank, its up rows at its own rows of the product, times strength · alpha / rank — the change
/// `down · up` the files' own, a fused `qkv` adapted by its thirds. The files are written here, small, with values.
final class LoRAFactorsTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-lora-factors-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    /// A safetensors of f32 tensors with their values.
    private func file(_ name: String, _ tensors: [(String, [Int], [Float])]) throws -> String {
        var header: [String: Any] = [:], data = Data()
        for (key, shape, values) in tensors {
            header[key] = ["dtype": "F32", "shape": shape, "data_offsets": [data.count, data.count + 4 * values.count]]
            data.append(contentsOf: values.withUnsafeBytes { Array($0) })
        }
        let json = try JSONSerialization.data(withJSONObject: header)
        let path = root.appendingPathComponent(name).path
        try (withUnsafeBytes(of: UInt64(json.count).littleEndian) { Data($0) } + json + data).write(to: URL(fileURLWithPath: path))
        return path
    }

    func testModulesLandSideBySideOnTheirProduct() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let (d, r) = (4, 2)
        func values(_ n: Int, _ seed: Float) -> [Float] { (0..<n).map { Float($0 % 5) * 0.5 - seed } }
        // to_q and to_k in one file (kohya's alpha 4 at rank 2: a scale of 2), to_q again in another.
        let a = try LoRAFile(try file("a.safetensors", [
            ("diffusion_model.l.to_q.lora_down.weight", [r, d], values(r * d, 1)), ("diffusion_model.l.to_q.lora_up.weight", [d, r], values(d * r, 0.25)),
            ("diffusion_model.l.to_q.alpha", [], [4]),
            ("diffusion_model.l.to_k.lora_down.weight", [r, d], values(r * d, 0.5)), ("diffusion_model.l.to_k.lora_up.weight", [d, r], values(d * r, 1)),
            ("diffusion_model.l.to_k.alpha", [], [4]),
        ]))
        let b = try LoRAFile(try file("b.safetensors", [
            ("l.to_q.lora_A.weight", [r, d], values(r * d, 2)), ("l.to_q.lora_B.weight", [d, r], values(d * r, 0.75)),
        ]))
        let modules: [String: LoRAStack.Shape] = [
            "l.to_q": .init(out: d, inn: d, product: "l.qkv", row: 0), "l.to_k": .init(out: d, inn: d, product: "l.qkv", row: d),
            "l.to_v": .init(out: d, inn: d, product: "l.qkv", row: 2 * d), "l.out": .init(out: d, inn: d),
        ]
        let stack = LoRAStack(entries: [(a, 0.5), (b, 1)])
        XCTAssertEqual(stack.rank(of: ["l.qkv"], modules: modules), 32, "three modules of rank 2, padded")
        XCTAssertEqual(stack.rank(of: ["l.out"], modules: modules), 0)

        for element in [NDArray.ScalarType.float32, .float16] {
            let products = [LoRAProduct(name: "qkv", key: "l.qkv", inn: d, out: 3 * d), LoRAProduct(name: "out", key: "l.out", inn: d, out: d)]
            let ranks = ["qkv": 32, "out": 32]
            let tensors = LoRAStack.tensors(products, ranks: ranks, element: element, prefix: "l.")
            XCTAssertEqual(tensors.map(\.name), ["l.qkv_down", "l.qkv_up", "l.out_down", "l.out_up"])
            let arena = try Arena(device, bytes: tensors.reduce(0) { $0 + Arena.aligned($1.bytes) })
            let at = Dictionary(uniqueKeysWithValues: tensors.map { ($0.name, (arena.take($0.bytes), $0.bytes)) })
            try stack.write(products, modules: modules, ranks: ranks, element: element, prefix: "l.") { arena.pointer(at[$0]!.0) }

            func floats(_ name: String) -> [Double] {
                let n = at["l." + name]!.1 / (element == .float16 ? 2 : 4), base = arena.pointer(at["l." + name]!.0)
                return (0..<n).map { element == .float16 ? Double(base.load(fromByteOffset: 2 * $0, as: Float16.self)) : Double(base.load(fromByteOffset: 4 * $0, as: Float.self)) }
            }
            // Δ [in, out] = down [in, R] · up [R, out], against Σ scale · (up_m · down_m)ᵀ at the module's rows.
            let down = floats("qkv_down"), up = floats("qkv_up"), R = 32, out = 3 * d
            var want = [Double](repeating: 0, count: d * out)
            func add(_ dn: [Float], _ u: [Float], scale: Double, row: Int) {
                for k in 0..<d { for o in 0..<d { for j in 0..<r { want[k * out + row + o] += scale * Double(u[o * r + j]) * Double(dn[j * d + k]) } } }
            }
            add(values(r * d, 1), values(d * r, 0.25), scale: 0.5 * 2, row: 0)
            add(values(r * d, 0.5), values(d * r, 1), scale: 0.5 * 2, row: d)
            add(values(r * d, 2), values(d * r, 0.75), scale: 1, row: 0)
            for k in 0..<d {
                for o in 0..<out {
                    let got = (0..<R).reduce(0.0) { $0 + down[k * R + $1] * up[$1 * out + o] }
                    XCTAssertEqual(got, want[k * out + o], accuracy: element == .float16 ? 2e-2 : 1e-5, "\(element) [\(k), \(o)]")
                }
            }
            XCTAssert(floats("out_down").allSatisfy { $0 == 0 } && floats("out_up").allSatisfy { $0 == 0 }, "nothing lands on out: zeros")
        }
    }

    /// A place smaller than what lands on it is refused, never cut.
    func testARankPastItsPlaceIsRefused() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let a = try LoRAFile(try file("c.safetensors", [
            ("m.lora_A.weight", [40, 4], [Float](repeating: 0.1, count: 160)), ("m.lora_B.weight", [4, 40], [Float](repeating: 0.1, count: 160)),
        ]))
        let products = [LoRAProduct(name: "m", key: "m", inn: 4, out: 4)]
        let tensors = LoRAStack.tensors(products, ranks: ["m": 32], element: .float32)
        let arena = try Arena(device, bytes: tensors.reduce(0) { $0 + Arena.aligned($1.bytes) })
        let at = Dictionary(uniqueKeysWithValues: tensors.map { ($0.name, arena.take($0.bytes)) })
        XCTAssertThrowsError(try LoRAStack(entries: [(a, 1)]).write(products, modules: ["m": .init(out: 4, inn: 4)], ranks: ["m": 32],
                                                                   element: .float32) { arena.pointer(at[$0]!) })
    }
}
