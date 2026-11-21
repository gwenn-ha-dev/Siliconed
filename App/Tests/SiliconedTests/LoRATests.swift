import Foundation
import XCTest
@testable import Siliconed

/// **A LoRA's file read as its trainer wrote it, and refused where it is not a plain LoRA.** The files are written here,
/// small: the names and shapes are the published ones, the values zeros. What each family's DiT adapts is the new
/// engine's to say (`FamilyTraits.loraModules`, not ported): until then no file is recognized, and none is applied.
final class LoRATests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-lora-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    /// A safetensors of f32 tensors — zeros, or the one value given for a scalar — with its metadata.
    private func file(_ name: String, _ tensors: [(String, [Int], Float)], metadata: [String: String] = [:]) throws -> String {
        var header: [String: Any] = [:], data = Data()
        for (key, shape, value) in tensors {
            let count = shape.reduce(1, *)
            header[key] = ["dtype": "F32", "shape": shape, "data_offsets": [data.count, data.count + 4 * count]]
            data.append(contentsOf: [Float](repeating: value, count: count).withUnsafeBytes { Array($0) })
        }
        if !metadata.isEmpty { header["__metadata__"] = metadata }
        let json = try JSONSerialization.data(withJSONObject: header)
        let path = root.appendingPathComponent(name).path
        try (withUnsafeBytes(of: UInt64(json.count).littleEndian) { Data($0) } + json + data).write(to: URL(fileURLWithPath: path))
        return path
    }

    private func peft(_ module: String, rank: Int, inn: Int, out: Int, prefix: String = "diffusion_model.") -> [(String, [Int], Float)] {
        [("\(prefix)\(module).lora_A.weight", [rank, inn], 0), ("\(prefix)\(module).lora_B.weight", [out, rank], 0)]
    }

    private func kohya(_ stem: String, rank: Int, inn: Int, out: Int, alpha: Float) -> [(String, [Int], Float)] {
        [("\(stem).lora_down.weight", [rank, inn], 0), ("\(stem).lora_up.weight", [out, rank], 0), ("\(stem).alpha", [], alpha)]
    }

    // MARK: - the formats

    func testPEFTNamesAreReadUnderEveryPrefixAndScaleOneWithoutAlpha() throws {
        for prefix in ["diffusion_model.", "transformer.", "model.diffusion_model.", ""] {
            let lora = try LoRAFile(try file("p.safetensors", peft("layers.0.attention.to_q", rank: 4, inn: 3840, out: 3840, prefix: prefix)))
            XCTAssertFalse(lora.flat)
            XCTAssertEqual(lora.modules.keys.sorted(), ["layers.0.attention.to_q"], prefix)
            XCTAssertEqual(lora.module("layers.0.attention.to_q")?.scale, 1)
            XCTAssertEqual(lora.module("layers.0.attention.to_q")?.rank, 4)
        }
    }

    func testTheAdaptersNameInsideAKeyIsRead() throws {
        let lora = try LoRAFile(try file("d.safetensors", [("m.lora_A.default.weight", [2, 8], 0), ("m.lora_B.default.weight", [8, 2], 0)]))
        XCTAssertEqual(lora.module("m")?.rank, 2)
    }

    func testTheScaleIsAlphaOverRank() throws {
        let lora = try LoRAFile(try file("k.safetensors", kohya("diffusion_model.m", rank: 4, inn: 8, out: 8, alpha: 16)))
        XCTAssertEqual(lora.module("m")?.scale, 4)
    }

    func testTheAdaptersMetadataGivesTheAlpha() throws {
        let said = ["lora_adapter_metadata": #"{"transformer.lora_alpha": 8, "transformer.r": 4, "transformer.rank_pattern": {}}"#]
        let lora = try LoRAFile(try file("m.safetensors", peft("m", rank: 4, inn: 8, out: 8), metadata: said))
        XCTAssertEqual(lora.module("m")?.scale, 2)
    }

    func testAFlatNameResolvesAgainstTheGraphsDottedOne() throws {
        let lora = try LoRAFile(try file("f.safetensors", kohya("lora_unet_blocks_3_self_attn_q_proj", rank: 2, inn: 2048, out: 2048, alpha: 2)))
        XCTAssert(lora.flat)
        XCTAssertNotNil(lora.module("blocks.3.self_attn.q_proj"))
    }

    // MARK: - the trigger words

    func testTheTriggerWordsAreWhatTheFileStates() {
        XCTAssertEqual(LoRAFile.triggers(["modelspec.trigger_phrase": "flat color,  no lineart ,"]), ["flat color", "no lineart"])
        XCTAssertEqual(LoRAFile.triggers(["trigger_word": "sks dog"]), ["sks dog"])
        // kohya: the tags every image of the dataset carries, three at most, the shortest first; a chance tag is not one.
        let kohya = ["ss_dataset_dirs": #"{"Incase v2": {"n_repeats": 8, "img_count": 47}}"#,
                     "ss_tag_frequency": #"{"Incase v2": {"incase style": 47, "breasts": 37, "1girl": 37, "solo": 29}}"#]
        XCTAssertEqual(LoRAFile.triggers(kohya), ["incase style"])
        XCTAssertEqual(LoRAFile.triggers(kohya.merging(["modelspec.trigger_phrase": "Incase Style"]) { a, _ in a }), ["Incase Style"])
        // Without the dataset's count, a single image, an empty tag, or nothing said: no word.
        XCTAssertEqual(LoRAFile.triggers(["ss_tag_frequency": #"{"d": {"x": 4}}"#]), [])
        XCTAssertEqual(LoRAFile.triggers(["ss_dataset_dirs": #"{"1_": {"img_count": 1}}"#, "ss_tag_frequency": #"{"1_": {"": 1}}"#]), [])
        XCTAssertEqual(LoRAFile.triggers(["ss_dataset_dirs": "not json", "ss_tag_frequency": "{"]), [])
        XCTAssertEqual(LoRAFile.triggers(["software": "ai-toolkit", "name": "V5"]), [])
    }

    // MARK: - what is refused

    private func refusal(_ path: String) -> EngineError.LoRARefusal? {
        do { _ = try LoRAFile(path) } catch EngineError.unsupportedLoRA(_, let reason) { return reason } catch {}
        return nil
    }

    func testWhatIsNotAPlainLoRAIsRefusedByName() throws {
        XCTAssertEqual(refusal(try file("w.safetensors", [("layers.0.weight", [4, 4], 0)])), .notALoRA)
        XCTAssertEqual(refusal(try file("lokr.safetensors", peft("m", rank: 2, inn: 4, out: 4) + [("m.lokr_w1", [2, 2], 0)])), .unsupportedKind)
        XCTAssertEqual(refusal(try file("te.safetensors", peft("m", rank: 2, inn: 4, out: 4) + [("lora_te_x.alpha", [], 1)])), .unsupportedKind)
        XCTAssertEqual(refusal(try file("half.safetensors", [("m.lora_A.weight", [2, 4], 0)])), .inconsistent)
        XCTAssertEqual(refusal(try file("ranks.safetensors", [("m.lora_A.weight", [2, 4], 0), ("m.lora_B.weight", [4, 3], 0)])), .inconsistent)
        let patterned = ["lora_adapter_metadata": #"{"transformer.lora_alpha": 8, "transformer.rank_pattern": {"m": 2}}"#]
        XCTAssertEqual(refusal(try file("pattern.safetensors", peft("m", rank: 2, inn: 4, out: 4), metadata: patterned)), .unsupportedKind)
        let dora = ["lora_adapter_metadata": #"{"transformer.lora_alpha": 8, "transformer.use_dora": true}"#]
        XCTAssertEqual(refusal(try file("dora.safetensors", peft("m", rank: 2, inn: 4, out: 4), metadata: dora)), .unsupportedKind)
    }

    /// A file is recognized by the family whose DiT has its modules (Z-Image's `to_q`, `[3840, 3840]`), and refused by
    /// its own name where no module of it is one.
    func testAFileIsRecognizedByTheFamilyItAdapts() throws {
        let z = try file("z.safetensors", peft("layers.0.attention.to_q", rank: 2, inn: 3840, out: 3840))
        XCTAssertEqual(try LoRAFile(z).family(), .zImage)
        let none = try file("none.safetensors", peft("nowhere.to_q", rank: 2, inn: 3840, out: 3840))
        XCTAssertNil(try LoRAFile(none).family())
        XCTAssertThrowsError(try LoRAStack([LoRAEntry(none)], on: .zImage)) { error in
            guard case EngineError.unsupportedLoRA(_, .noTarget) = error else { return XCTFail("\(error)") }
        }
    }

    /// The name the window shows is the card's, read from the path alone: a grid of one model's LoRA, shown while the rack
    /// holds another model whose list does not have it, names it as its own model's list does.
    func testALoRAsNameIsItsCardsWithoutTheCard() throws {
        let path = try file("incase_style_v2.safetensors", peft("layers.0.attention.to_q", rank: 2, inn: 3840, out: 3840))
        let card = LoRACard(file: try LoRAFile(path), family: .zImage)
        XCTAssertEqual(card.name, "incase style v2")
        XCTAssertEqual(LoRACard.name(path: path), card.name)
        XCTAssertEqual(LoRACard.name(path: "/Lender/loras/anima/Niji_semi-realism.lora.safetensors"), "Niji semi realism")
    }

    func testAStrengthOfZeroAdaptsNothing() throws {
        let path = try file("z.safetensors", peft("layers.0.attention.to_q", rank: 2, inn: 3840, out: 3840))
        XCTAssert(try LoRAStack([LoRAEntry(path, strength: 0)], on: .zImage).isEmpty)
    }
}
