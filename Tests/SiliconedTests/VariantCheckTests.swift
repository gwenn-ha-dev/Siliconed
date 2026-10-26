import Foundation
import Testing
@testable import Siliconed
@testable import Verification

/// `check <model> dit --variant compact|light`: which golden, which map, and the refusal of a map of the other version.
struct VariantCheckTests {
    private func summary(_ dtypes: [String: Int]) -> MapSummary { MapSummary(path: "/tmp/x.silicon", bytes: 1, dtypes: dtypes) }

    @Test func goldenNamesCarryTheVersionBeforeTheSize() {
        #expect(variantStem("goldens-dit-512", nil) == "goldens-dit-512")
        #expect(variantStem("goldens-dit-512", .compact) == "goldens-dit-compact-512")
        #expect(variantStem("witness-bf16-dit-768x512", .light) == "witness-bf16-dit-light-768x512")
        #expect(variantStem("anima/goldens-anima-dit-512", .light) == "anima/goldens-anima-dit-light-512")
        #expect(variantStem("qwen21/goldens-qwen21-dit-512", .compact) == "qwen21/goldens-qwen21-dit-compact-512")
        // The bf16 bar is found from the golden's name by the engine's own rule (`HalfPrecisionReference`).
        #expect(variantStem("goldens-dit-512", .compact).replacingOccurrences(of: "goldens-", with: "witness-bf16-")
                == variantStem("witness-bf16-dit-512", .compact))
    }

    @Test func aMapIsTheVersionItsTensorsSay() {
        #expect(summary(["bfloat16": 300, "float32": 4]).quantizedVersion == nil)
        // A GGUF Q8_0 is kept as int8 with its block scales, an int8 convrot the same: the Compact.
        #expect(summary(["int8": 200, "bfloat16": 20]).quantizedVersion == .compact)
        // A Light has GGUF blocks, whatever 8-bit tensors Q4_K_M files also hold (Qwen's Q8_0).
        #expect(summary(["q4_k": 100, "q6_k": 3, "int8": 20, "float32": 1]).quantizedVersion == .light)
        #expect(summary(["q5_k": 1]).quantizedVersion == .light)
    }

    @Test func aLightIsNotJudgedAgainstACompactsGoldenNorTheReverse() throws {
        let light = summary(["q4_k": 100, "bfloat16": 4]), compact = summary(["int8": 100, "bfloat16": 4])
        #expect(light.mismatch(version: .light) == nil && compact.mismatch(version: .compact) == nil)
        let why = try #require(light.mismatch(version: .compact))
        #expect(why.contains("light") && why.contains("compact"))
        #expect(compact.mismatch(version: .light) != nil)
        #expect(summary(["bfloat16": 4]).mismatch(version: .compact)?.contains("Standard") == true)
    }

    @Test func theOptionIsParsedAndRefusedWhenItIsNotAVersion() throws {
        let stage = try #require(Registry.stage("z-image", "dit"))
        let compact = try Invocation(["--variant", "compact"], accepted: stage.options, defaultSize: "512")
        #expect(try compact.quantizedVariant() == .compact)
        #expect(try Invocation([], accepted: stage.options, defaultSize: "512").quantizedVariant() == nil)
        let standard = try Invocation(["--variant", "standard"], accepted: stage.options, defaultSize: "512")
        #expect(throws: UsageError.self) { try standard.quantizedVariant() }
        let nonsense = try Invocation(["--variant", "tiny"], accepted: stage.options, defaultSize: "512")
        #expect(throws: UsageError.self) { try nonsense.quantizedVariant() }
    }

    @Test func everyFamilysDitStageTakesTheOptionAndNamesItsOracle() {
        for model in ["z-image", "qwen-image-2.1", "anima", "krea2"] {
            let stage = Registry.stage(model, "dit")
            #expect(stage?.options.contains("--variant") == true, "\(model)")
            #expect(stage?.oracle?.contains("--variant") == true, "\(model)")
        }
    }

    @Test func aVersionMapHasItsOwnNameBesideTheStandards() {
        for family in Family.allCases {
            let names = Set(family.variants.map { family.ditMap($0) })
            #expect(names.count == family.variants.count, "\(family)")
        }
    }
}
