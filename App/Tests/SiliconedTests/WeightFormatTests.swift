import Foundation
import Testing
@testable import Siliconed

/// **The weights' storage is read from what is installed** (`WeightFormat`), from fabricated safetensors headers: the dtypes of the
/// matrices weighted by their values.
@Suite struct WeightFormatTests {
    private func tensor(_ dtype: String, _ shape: [Int]) -> [String: Any] { ["dtype": dtype, "shape": shape, "data_offsets": [0, 0]] }

    @Test func aConvrotInt8FileWithFloatsLeftOverIsInt8() {
        let h: [String: Any] = ["a.weight": tensor("I8", [4096, 4096]), "a.weight_scale": tensor("F32", [4096, 1]),
                                "a.comfy_quant": tensor("U8", [72]), "b.weight": tensor("I8", [4096, 4096]),
                                "embed.weight": tensor("BF16", [1000, 4096]), "norm.weight": tensor("BF16", [4096])]
        #expect(WeightFormat.read(safetensorsHeaders: [h]) == .int8)
        #expect(WeightFormat.int8.label == "int8")
    }

    @Test func aPublishedBf16FileIsBf16EvenWithAnFp32Table() {
        let h: [String: Any] = ["l.weight": tensor("BF16", [4096, 4096]), "l2.weight": tensor("BF16", [4096, 4096]),
                                "embed.weight": tensor("F32", [1000, 4096])]
        #expect(WeightFormat.read(safetensorsHeaders: [h]) == .bf16)
    }

    /// A checkpoint in shards is one file: the matrices of every shard count together.
    @Test func shardsCountTogether() {
        let first: [String: Any] = ["a.weight": tensor("F32", [1024, 1024])]
        let others: [String: Any] = ["b.weight": tensor("BF16", [4096, 4096]), "c.weight": tensor("BF16", [4096, 4096])]
        #expect(WeightFormat.read(safetensorsHeaders: [first]) == .fp32)
        #expect(WeightFormat.read(safetensorsHeaders: [first, others]) == .bf16)
    }

    @Test func aSafetensorsVAEIsReadByItsDtype() {
        let h: [String: Any] = ["__metadata__": ["format": "pt"], "conv.weight": tensor("BF16", [64, 64, 3, 3]), "conv.bias": tensor("F32", [64])]
        #expect(WeightFormat.read(safetensorsHeaders: [h]) == .bf16)
        #expect(WeightFormat.read(safetensorsHeaders: [["x": tensor("F32", [8, 8])]]) == .fp32)
    }

    @Test func noMatrixIsNoFormatAndAFormatSurvivesJSON() throws {
        #expect(WeightFormat.read(safetensorsHeaders: [["n": tensor("F32", [64])]]) == nil)
        #expect(WeightFormat.read(safetensorsHeaders: []) == nil)
        let formats = WeightFormats(dit: .int8, text: .int8, vae: .bf16)
        let back = try JSONDecoder().decode(WeightFormats.self, from: JSONEncoder().encode(formats))
        #expect(back == formats)
    }
}
