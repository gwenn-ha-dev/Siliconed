import Foundation
import Testing
@testable import Siliconed

/// **The weights' storage is read from what is installed** (`WeightFormat`), from fabricated headers: the dtypes of the matrices
/// weighted by their values, a per-block scale making GGUF's Q8_0, K-quants mixed with Q6_K making the `_M` mixtures.
@Suite struct WeightFormatTests {
    private func tensor(_ dtype: String, _ shape: [Int], scale: [String: Any]? = nil) -> [String: Any] {
        var e: [String: Any] = ["dtype": dtype, "shape": shape, "offset": 0, "bytes": 0]
        if let scale { e["scale"] = scale }
        return e
    }
    private func header(_ tensors: [String: [String: Any]]) -> [String: Any] { ["tensors": tensors] }

    @Test func aConvrotInt8MapWithFloatsLeftOverIsInt8() {
        let h = header(["a.weight": tensor("int8", [4096, 4096], scale: ["dtype": "float32", "shape": [4096], "offset": 16]),
                        "b.weight": tensor("int8", [4096, 4096], scale: ["dtype": "float32", "shape": [4096], "offset": 16]),
                        "embed.weight": tensor("float32", [1000, 4096]), "norm.weight": tensor("float32", [4096]),
                        "mod.weight": tensor("bfloat16", [64, 64])])
        #expect(WeightFormat.read(mapHeader: h) == .int8)
        #expect(WeightFormat.int8.label == "int8")
    }

    @Test func aScalePerBlockIsGGUFQ8_0() {
        let h = header(["a.weight": tensor("int8", [4096, 4096], scale: ["dtype": "float16", "shape": [128, 4096], "offset": 16, "block": 32]),
                        "n.weight": tensor("bfloat16", [4096, 4096 / 64])])
        #expect(WeightFormat.read(mapHeader: h) == .q8_0)
        #expect(WeightFormat.q8_0.label == "Q8_0")
    }

    @Test func k_quantsMixedWithQ6KAreTheMMixture() {
        let mixed = header(["a": tensor("q4_k", [4096, 4096]), "b": tensor("q4_k", [4096, 4096]), "c": tensor("q6_k", [4096, 1024])])
        #expect(WeightFormat.read(mapHeader: mixed) == .q4_k_m)
        #expect(WeightFormat.q4_k_m.label == "Q4_K_M")
        #expect(WeightFormat.read(mapHeader: header(["a": tensor("q4_k", [256, 256])])) == .q4_k)
    }

    @Test func ourFourBitMapIsSQWhateverItsInt8Layers() {
        let h = header(["a": tensor("sq5", [4096, 4096]), "b": tensor("sq4", [4096, 4096]),
                        "c": tensor("int8", [4096, 512], scale: ["dtype": "float32", "shape": [512], "offset": 16]),
                        "r": tensor("bfloat16", [4096, 4096])])
        #expect(WeightFormat.read(mapHeader: h) == .sq)
        #expect(WeightFormat.sq.label == "SQ")
        #expect(Artifact.weightFormat(header: h) == .int8)
    }

    @Test func aPublishedBf16MapIsBf16EvenWithAnFp32Embedding() {
        let h = header(["l.weight": tensor("bfloat16", [4096, 4096]), "l2.weight": tensor("bfloat16", [4096, 4096]),
                        "embed.weight": tensor("float32", [1000, 4096])])
        #expect(WeightFormat.read(mapHeader: h) == .bf16)
    }

    @Test func aSafetensorsVAEIsReadByItsDtype() {
        let h: [String: Any] = ["__metadata__": ["format": "pt"], "conv.weight": ["dtype": "BF16", "shape": [64, 64, 3, 3], "data_offsets": [0, 1]],
                                "conv.bias": ["dtype": "F32", "shape": [64], "data_offsets": [0, 1]]]
        #expect(WeightFormat.read(safetensorsHeader: h) == .bf16)
        #expect(WeightFormat.read(safetensorsHeader: ["x": ["dtype": "F32", "shape": [8, 8]]]) == .fp32)
    }

    @Test func noMatrixIsNoFormatAndAFormatSurvivesJSON() throws {
        #expect(WeightFormat.read(mapHeader: header(["n": tensor("float32", [64])])) == nil)
        #expect(WeightFormat.read(mapHeader: [:]) == nil)
        let formats = WeightFormats(dit: .int8, text: .q8_0, vae: .bf16)
        let back = try JSONDecoder().decode(WeightFormats.self, from: JSONEncoder().encode(formats))
        #expect(back == formats)
        // A lender older than the field publishes none: the traits still decode.
        let traits = RemoteTraits(maxReferences: 0, readsImages: false, acceptsDetail: false, steps: [8], text: "t", encoding: nil, decoding: "v")
        #expect(traits.weights == nil)
        let withWeights = RemoteTraits(maxReferences: 0, readsImages: false, acceptsDetail: false, steps: [8], text: "t", encoding: nil, decoding: "v", weights: formats)
        #expect(try JSONDecoder().decode(RemoteTraits.self, from: JSONEncoder().encode(withWeights)).weights == formats)
    }
}
