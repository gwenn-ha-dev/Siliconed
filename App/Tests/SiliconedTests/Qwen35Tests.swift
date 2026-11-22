import SiliCore
import XCTest
@testable import Siliconed

/// **MOSS-Image's Qwen3.5 encoder, the part that needs no weight**: the template, the paliers, the layers' groups, and
/// the chunk's masks the delta rule's inverse is built with.
final class Qwen35Tests: XCTestCase {

    func testTemplateHasNoSystemTurnAndOpensTheThinkingBlock() {
        XCTAssertEqual(Qwen35Prompt.template("woman posing in a library"),
                       "<|im_start|>user\nwoman posing in a library<|im_end|>\n<|im_start|>assistant\n<think>\n")
    }

    /// Every palier a whole number of chunks; the longest prompt a render admits (1024 tokens and the template's) holds.
    func testPaliersAreWholeChunks() throws {
        XCTAssertTrue(Qwen35Encoder.paliers.allSatisfy { $0 % Qwen35Encoder.chunkTokens == 0 })
        XCTAssertEqual(try Qwen35Encoder.palier(21), 64)
        XCTAssertEqual(try Qwen35Encoder.palier(858), 1088)
        XCTAssertEqual(try Qwen35Encoder.palier(1024 + 14), 1088)
        XCTAssertThrowsError(try Qwen35Encoder.palier(1089))
    }

    /// The stage: a group a layer, each place an argument of its kind's function, the fused parts listed in the
    /// file's order; the DeltaNet layers lay out as the slots' group, the attention layers apart.
    func testTheStageDeclaresEveryLayer() {
        let e = Qwen35Encoder(), c = e.config, d = e.declaration(rows: 64, path: .bfloat)
        XCTAssertEqual(d.layers.map(\.name), (0..<32).map { "\($0)" })
        XCTAssertEqual(d.layers.filter { $0.form == "attention" }.map(\.name), ["3", "7", "11", "15", "19", "23", "27", "31"])
        XCTAssertEqual(Set(d.kinds.keys), ["delta", "attention"])
        XCTAssertEqual(Set(d.edges.keys), ["norm"])
        for i in [0, 3] {
            XCTAssertEqual(Set(e.entries(i).map(\.name)), Set(e.arguments(full: c.isFull(i), rows: 1).dropFirst().map(\.name)), "layer \(i)")
        }
        XCTAssertEqual(e.entries(0).first { $0.name == "in" }?.recipe,
                       .product(joined: ["in_proj_z", "in_proj_b", "in_proj_a"].map { "model.language_model.layers.0.linear_attn.\($0)" }))
        XCTAssertEqual(e.entries(3).first { $0.name == "in" }?.key, "model.language_model.layers.3.self_attn.q_proj")
        XCTAssertEqual((0..<c.layers).filter(c.isFull), [3, 7, 11, 15, 19, 23, 27, 31])
    }

    /// The masks build the true inverse of I + L for a strictly lower L, block by block (as the function does, in
    /// doubles here): X₀ = I − L∘level₀, then X ← X − X (L∘level) X.
    func testTheChunkMasksInvertAUnitLowerTriangle() {
        let t = Qwen35Encoder.chunkTokens
        let masks = Qwen35Encoder.chunkMasks(t)
        XCTAssertEqual(masks.levels.count, 6)
        var seed: UInt64 = 7
        func random() -> Double { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Double(seed >> 11) / Double(1 << 53) - 0.5 }
        var lower = [Double](repeating: 0, count: t * t)
        for i in 0..<t { for j in 0..<i { lower[i * t + j] = random() * Double(masks.below[i * t + j]) } }
        func product(_ a: [Double], _ b: [Double]) -> [Double] {
            var c = [Double](repeating: 0, count: t * t)
            for i in 0..<t { for k in 0..<t where a[i * t + k] != 0 { for j in 0..<t { c[i * t + j] += a[i * t + k] * b[k * t + j] } } }
            return c
        }
        func masked(_ level: [Float]) -> [Double] { zip(lower, level).map { $0 * Double($1) } }
        var inverse = (0..<(t * t)).map { ($0 / t == $0 % t ? 1 : 0) - masked(masks.levels[0])[$0] }
        for level in masks.levels.dropFirst() {
            let step = product(product(inverse, masked(level)), inverse)
            inverse = zip(inverse, step).map { $0 - $1 }
        }
        let unit = (0..<(t * t)).map { ($0 / t == $0 % t ? 1.0 : 0) + lower[$0] }
        let identity = product(unit, inverse)
        let worst = (0..<(t * t)).map { abs(identity[$0] - ($0 / t == $0 % t ? 1 : 0)) }.max()!
        XCTAssertLessThan(worst, 1e-9)
        // The cumulative sum's mask: row j's decay counts in every column at or after j.
        XCTAssertEqual(masks.onAndAbove[0 * t + 5], 1); XCTAssertEqual(masks.onAndAbove[5 * t + 0], 0)
        XCTAssertEqual(masks.above[0 * t + 1], -1e30); XCTAssertEqual(masks.above[1 * t + 0], 0)
    }

    /// The half path: every argument a weight of the layer's entries, the products' codes by pieces of k, the stream
    /// and its factors a row.
    func testTheHalfPathDeclaresItsCodes() {
        let e = Qwen35Encoder(), c = e.config
        for i in [0, 3] {
            let names = Set(e.halfArguments(full: c.isFull(i), rows: 1).dropFirst(2).map(\.name))
            let entries = e.halfEntries(i)
            let codes = entries.filter { $0.recipe == .codes }.map(\.name)
            XCTAssertEqual(names, Set(entries.map(\.name) + codes.map { $0 + "_scale" }), "layer \(i)")
        }
        let q = e.halfArguments(full: true, rows: 1).first { $0.name == "q" }
        XCTAssertEqual(q?.shape, Weights.codesShape(inn: c.width, out: 2 * c.heads * c.headDim))
        let d = e.declaration(rows: 128, path: .half)
        XCTAssertEqual(d.perRow.map(\.name), ["x", "x_factor"])
        XCTAssertEqual(d.perRow.map(\.next), ["y", "y_factor"])
        XCTAssertEqual(d.tensors.map(\.name), ["hidden"])
    }

    /// The embedding's rotation by four 4-point transforms is the product by ConvRot's H₂₅₆ / 16.
    func testTheEmbeddingsRotationIsConvRots() {
        let (order, h) = Aicode.convrotMatrix()
        let x = (0..<order).map { Float(($0 * 37) % 11) - 5 }
        var y = x
        y.withUnsafeMutableBufferPointer { Qwen35Encoder.hadamard256($0.baseAddress!) }
        for j in 0..<order {
            let expected = (0..<order).reduce(Float(0)) { $0 + x[$1] * Float(h[$1 * order + j]) } / 16
            XCTAssertEqual(y[j], expected, accuracy: 1e-5)
        }
    }

    /// The stream as written: each row's largest in [2¹³, 2¹⁴), x̂ · X the value.
    func testTheStreamIsWrittenUnderItsFactor() {
        let values: [Float] = [3, -0.5, 1e-3, 2, 1000, -7, 0.25, 4]
        var half = [Float16](repeating: 0, count: 8), factors = [Float](repeating: 0, count: 2)
        half.withUnsafeMutableBytes { h in factors.withUnsafeMutableBufferPointer { Qwen35Encoder.writeStream(values, width: 4, to: h.baseAddress!, factors: $0.baseAddress!) } }
        for r in 0..<2 {
            let top = (0..<4).map { abs(Float(half[r * 4 + $0])) }.max()!
            XCTAssertTrue(top >= 0x1p13 && top < 0x1p14, "row \(r): \(top)")
            for j in 0..<4 { XCTAssertEqual(Float(half[r * 4 + j]) * factors[r], values[r * 4 + j], accuracy: abs(values[r * 4 + j]) * 1e-3 + 1e-6) }
        }
    }
}
