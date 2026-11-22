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

/// An F16 tensor reads as the f32s it holds, exactly: listed F32, its rows read alone widened too (a fine-tune saved
/// in fp16, Kroma's globals).
@Test func anF16TensorReadsAsTheF32ItHolds() throws {
    // 1, the smallest subnormal, the largest finite, −0.333 (rounded to f16), and a row more.
    let halves: [Float16] = [1, Float16(bitPattern: 0x0001), 65504, -0.333, 2, 0.5]
    let path = try safetensors("f16-read.safetensors", [("h", "F16", [2, 3], bytes(halves.map(\.bitPattern)))])
    let file = try SafeTensors(path)
    let t = try file.real("h")
    #expect(t.dtype == "F32" && t.shape == [2, 3] && t.end - t.start == 24)
    #expect(try file.floats("h") == halves.map(Float.init))
    var row = [Float](repeating: 0, count: 3)
    try row.withUnsafeMutableBytes { try file.read(t, at: 12, count: 12, into: $0.baseAddress!) }
    #expect(row == halves[3...].map(Float.init))
}

/// An F16 product weight joins the bf16 products, rounded once to nearest even; an f16 recipe gets the file's bits back.
@Test(.enabled(if: metal != nil)) func anF16ProductIsRoundedToBF16Once() throws {
    let (out, inn) = (4, 64)
    let w = (0..<out * inn).map { Float16(Float($0 % 97) * 0.0137 - 0.6) }
    let path = try safetensors("f16-product.safetensors", [("p.weight", "F16", [out, inn], bytes(w.map(\.bitPattern))),
                                                            ("g", "F16", [inn], bytes(w.prefix(inn).map(\.bitPattern)))])
    let file = try SafeTensors(path)
    let entries: [Weights.Entry] = [("p", "p", .product(joined: [])), ("g", "g", .half)]
    #expect(try Weights.staging(entries, in: file) == 4 * out * inn)
    let arena = try Arena(metal!, bytes: try Weights.size(entries, in: file))
    let placed = try Weights.load(entries, from: file, into: arena)
    #expect(placed["p"]!.type == .bfloat16 && placed["p"]!.shape == [out, inn])
    let p = arena.pointer(placed["p"]!.offset).assumingMemoryBound(to: UInt16.self)
    for i in 0..<(out * inn) { #expect(p[i] == BFloat16Bits.rounded(Float(w[i]))) }
    let g = arena.pointer(placed["g"]!.offset).assumingMemoryBound(to: UInt16.self)
    for i in 0..<inn { #expect(g[i] == w[i].bitPattern) }
}

/// ComfyUI's « fp8 scaled »: an F8_E4M3 weight reads as its codes times its one scale, in f32, and joins the bf16
/// products rounded once more; a scale by rows leaves it F8_E4M3, refused by name.
@Test(.enabled(if: metal != nil)) func anF8WeightReadsAsItsCodesTimesItsScale() throws {
    // 1, the smallest subnormal (2⁻⁹), 448 (the largest), −1.5, 0.
    let codes: [UInt8] = [0x38, 0x01, 0x7e, 0xbc, 0x00]
    let expected: [Float] = [1, 0x1p-9, 448, -1.5, 0].map { $0 * 0.75 }
    let (out, inn) = (2, 64)
    let w = (0..<out * inn).map { UInt8(truncatingIfNeeded: $0 * 37) & 0x7e | 0x08 }
    let path = try safetensors("f8-read.safetensors", [
        ("c.weight", "F8_E4M3", [5], codes), ("c.weight_scale", "F32", [], bytes([Float(0.75)])),
        ("p.weight", "F8_E4M3", [out, inn], w), ("p.weight_scale", "F32", [], bytes([Float(0.01)])),
        ("r.weight", "F8_E4M3", [2, 1], [0x38, 0x38]), ("r.weight_scale", "F32", [2, 1], bytes([Float(1), 2])),
    ])
    let file = try SafeTensors(path)
    #expect(try file.floats("c.weight") == expected)
    #expect(throws: (any Error).self) { try file.real("r.weight") }
    let arena = try Arena(metal!, bytes: try Weights.size([("p", "p", .product(joined: []))], in: file))
    let placed = try Weights.load([("p", "p", .product(joined: []))], from: file, into: arena)
    let p = arena.pointer(placed["p"]!.offset).assumingMemoryBound(to: UInt16.self)
    // Positive normals and zero, as e4m3 writes them: (1 + m/8) · 2^(e − 7).
    func e4m3(_ b: UInt8) -> Float { b == 0 ? 0 : (1 + Float(b & 7) / 8) * Float(sign: .plus, exponent: Int(b >> 3) - 7, significand: 1) }
    for i in 0..<(out * inn) { #expect(p[i] == BFloat16Bits.rounded(e4m3(w[i]) * 0.01)) }
}
