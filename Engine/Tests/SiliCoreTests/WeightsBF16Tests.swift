import Foundation
import Metal
import SiliCore
import Testing

private let metal = MTLCreateSystemDefaultDevice()
private func bytes<T>(_ values: [T]) -> [UInt8] { values.withUnsafeBytes { Array($0) } }

/// The transposed bf16 recipe: a bf16 file's tensor `[out, in]` as `[in, out]`; an f32 rounded once to nearest even.
@Test func bf16RecipesSayWhatTheWeightsTakeInTheArena() throws {
    let (out, inn) = (48, 80)
    let path = try safetensors("bf16-recipes.safetensors", [
        ("w", "BF16", [out, inn], bytes([UInt16](repeating: 0x3f80, count: out * inn))),
        ("w32", "F32", [out, inn], bytes([Float](repeating: 1, count: out * inn))),
        ("gain", "BF16", [inn], bytes([UInt16](repeating: 0x3fc0, count: inn))),
    ])
    let file = try SafeTensors(path)
    let entries: [Weights.Entry] = [("w", "w", .bfloatTransposed), ("w32", "w32", .bfloatTransposed)]
    let slots = Dictionary(uniqueKeysWithValues: try Weights.layout(entries, in: file).map { ($0.name, $0) })
    #expect(slots["w"]!.shape == [inn, out] && slots["w"]!.type == .bfloat16 && slots["w"]!.bytes == 2 * out * inn)
    #expect(slots["w32"]!.shape == [inn, out] && slots["w32"]!.type == .bfloat16)
    // Through the staging: its bytes as the file holds them.
    #expect(try Weights.staging([("w", "w", .bfloatTransposed)], in: file) == 2 * out * inn)
    #expect(try Weights.staging([("w32", "w32", .bfloatTransposed)], in: file) == 4 * out * inn)
}

/// The bf16 weights read into the arena transposed, and an f32 rounded once to nearest even on the way.
@Test(.enabled(if: metal != nil)) func bf16RecipesReadTheWeightsIntoTheArena() throws {
    let (out, inn) = (48, 80)
    let w = (0..<out * inn).map { UInt16(truncatingIfNeeded: 0x3c00 + $0 % 977) }
    // 1 + 2⁻⁸ is a tie between 1 and 1 + 2⁻⁷: even, 1; 1 + 3 · 2⁻⁸ ties up to 1 + 2⁻⁶; a value past the tie rounds up.
    let reals: [Float] = [1 + 0x1p-8, 1 + 3 * 0x1p-8, 1 + 0x1p-8 + 0x1p-20, -2.5]
    let path = try safetensors("bf16-read.safetensors", [("w", "BF16", [out, inn], bytes(w)), ("r", "F32", [1, 4], bytes(reals))])
    let file = try SafeTensors(path)
    let entries: [Weights.Entry] = [("wt", "w", .bfloatTransposed), ("r", "r", .bfloatTransposed)]
    let arena = try Arena(metal!, bytes: try Weights.size(entries, in: file))
    let placed = try Weights.load(entries, from: file, into: arena)
    let wt = arena.pointer(placed["wt"]!.offset).assumingMemoryBound(to: UInt16.self)
    for o in 0..<out { for i in 0..<inn { #expect(wt[i * out + o] == w[o * inn + i]) } }
    let r = arena.pointer(placed["r"]!.offset).assumingMemoryBound(to: UInt16.self)
    #expect([r[0], r[1], r[2], r[3]] == [0x3f80, 0x3f82, 0x3f81, 0xc020])
}
