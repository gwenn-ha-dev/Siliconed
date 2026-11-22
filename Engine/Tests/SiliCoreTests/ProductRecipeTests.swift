import Foundation
import Metal
import Testing
@testable import SiliCore

/// The shards of one set of weights, and a product's parts joined along `out` (`Recipe.product`): what the arena
/// takes from the headers alone, and what it holds once read, the codes in either layout, counted by hand.
@Suite struct ProductRecipes {
    static func le<T>(_ values: [T]) -> [UInt8] { values.withUnsafeBytes { Array($0) } }

    /// Part a: codes `[2, 3]` with SDNQ's bf16 scales, in the first shard; part b: codes `[1, 3]` with ComfyUI's f32 scales,
    /// and bf16 weights p `[2, 3]` and q `[1, 3]`, in the second (a group cut between two shards). `dup` in both.
    static let file: SafeTensors = {
        let first = try! safetensors("shard-1.safetensors", [
            ("a.weight", "I8", [2, 3], le([Int8(1), 2, 3, 4, 5, 6])),
            ("a.scale", "BF16", [2, 1], le([BFloat16Bits.rounded(0.5), BFloat16Bits.rounded(0.25)])),
            ("p.weight", "BF16", [2, 3], le([Float](arrayLiteral: 1, 2, 3, 4, 5, 6).map(BFloat16Bits.rounded))),
            ("dup", "F32", [1], le([Float(1)]))])
        let second = try! safetensors("shard-2.safetensors", [
            ("b.weight", "I8", [1, 3], le([Int8(-7), -8, -9])),
            ("b.weight_scale", "F32", [1, 1], le([Float(2)])),
            ("q.weight", "BF16", [1, 3], le([Float](arrayLiteral: 7, 8, 9).map(BFloat16Bits.rounded))),
            ("dup", "F32", [1], le([Float(2)]))])
        return try! SafeTensors(shards: [first, second])
    }()
    static let entries: [Weights.Entry] = [("w", "a", .product(joined: ["b"])), ("v", "p", .product(joined: ["q"]))]

    @Test func everyShardsTensorIsReadFromItsOwn() throws {
        #expect(Self.file.tensors["a.weight"]?.file == 0 && Self.file.tensors["b.weight"]?.file == 1)
        #expect(try Self.file.floats("b.weight_scale") == [2])
        #expect(try Self.file.floats("a.scale") == [0.5, 0.25])
        #expect(try Self.file.floats("dup") == [1], "a key in two shards: the first's")
        #expect(try Self.file.bytes("b.weight") == Self.le([Int8(-7), -8, -9]))
        let again = try Self.file.reopened(uncached: true)
        #expect(again.paths == Self.file.paths)
        #expect(try again.floats("q.weight") == [7, 8, 9])
    }

    @Test(arguments: [CodesLayout.inOut, .outIn])
    func aJoinedProductTakesItsPartsSideBySide(_ layout: CodesLayout) throws {
        let slots = try Weights.layout(Self.entries, in: Self.file, codes: layout)
        let shape = [3, 3]
        #expect(slots.map(\.name) == ["w", "w_scale", "v"])
        #expect(slots.map(\.shape) == [shape, [3], shape] && slots.map(\.bytes) == [9, 6, 18])
        // Through the staging: a part of codes transposed, or a part's scales; the bf16 parts straight to their place.
        #expect(try Weights.staging(Self.entries, in: Self.file, codes: layout) == (layout == .inOut ? 6 : 4))
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let arena = try Arena(device, bytes: try Weights.size(Self.entries, in: Self.file, codes: layout))
        let placed = try Weights.load(Self.entries, from: Self.file, into: arena, codes: layout)
        let w = arena.pointer(placed["w"]!.offset).assumingMemoryBound(to: Int8.self)
        let s = arena.pointer(placed["w_scale"]!.offset).assumingMemoryBound(to: Float16.self)
        let v = arena.pointer(placed["v"]!.offset).assumingMemoryBound(to: UInt16.self)
        // [out, in]: a's rows then b's; [in, out]: its transpose. The bf16 weights [out, in] in either.
        let outIn: [Int8] = [1, 2, 3, 4, 5, 6, -7, -8, -9], inOut: [Int8] = [1, 4, -7, 2, 5, -8, 3, 6, -9]
        #expect((0..<9).map { w[$0] } == (layout == .outIn ? outIn : inOut))
        #expect((0..<3).map { s[$0] } == [0.5, 0.25, 2])
        #expect((0..<9).map { BFloat16Bits.widened(v[$0]) } == [1, 2, 3, 4, 5, 6, 7, 8, 9])
    }

    @Test func aProductWhosePartsDoNotJoinIsRefused() throws {
        let bad: [Weights.Entry] = [("w", "a", .product(joined: ["q"]))]
        #expect(throws: Failure.self) { try Weights.layout(bad, in: Self.file) }
        let none: [Weights.Entry] = [("w", "p", .codes)]
        #expect(throws: Failure.self) { try Weights.layout(none, in: Self.file) }
    }

    /// A stage's group across both shards is planned as one: its layout from the two headers.
    @Test func aGroupAcrossTwoShardsIsPlanned() throws {
        var d = Stage.Declaration("shards")
        d.groups = ["g": Self.entries]
        d.once = ["g"]
        let plan = try Stage.plan(d, files: [Self.file], budget: 1 << 30, floor: 0, depth: 1)
        #expect(plan.kept == ["g"] && plan.codes == .inOut)
    }
}
