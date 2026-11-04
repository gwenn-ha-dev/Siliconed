import CryptoKit
import Foundation
import XCTest
@testable import Siliconed

/// **The `parameters` chunk Civitai reads** (`Infotext`), and the LoRA's sha256 it credits by.
final class InfotextTests: XCTestCase {
    private let sha = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"

    private func render(model: String = "z-image", prompt: String = "woman posing in a library",
                        loras: [LoRAEntry] = [], origins: [Engine.Render.LoRAOrigin] = [],
                        strength: Double? = nil, solver: Solver = .euler, imported: Bool = false,
                        name: String? = "Z-Image Turbo") -> Engine.Render {
        var r = Engine.Render(model: model, prompt: prompt, seed: 1_303_787_694, steps: 8, loras: loras,
                              loraSummary: nil, image: ImageRGB(pixels: [Float](repeating: 0, count: 3 * 2 * 3), height: 2, width: 3), evaluations: 8, spectral: 0, strength: strength,
                              startStep: 0, startSigma: 1, timings: .init(), footprints: .init(), tokens: 1)
        r.solver = solver
        r.modelName = name
        r.isImportedModel = imported
        r.loraOrigins = origins
        return r
    }

    func testATxt2ImgWithoutLoRA() {
        XCTAssertEqual(render().parameters(version: "0.8"), """
            woman posing in a library
            Steps: 8, Sampler: Euler, CFG scale: 1, Seed: 1303787694, Size: 3x2, Model: Z-Image Turbo, \
            Civitai resources: [{"type":"checkpoint","modelVersionId":2442439}], Version: Siliconed 0.8
            """)
        // Outside the app's bundle: no version to give. Qwen's page; a developer-mode solver by its A1111 name.
        XCTAssertEqual(render(model: "qwen-image-2.1", solver: .dpmpp2m, name: "Qwen-Image-2.1 Turbo").parameters(version: nil), """
            woman posing in a library
            Steps: 8, Sampler: DPM++ 2M, CFG scale: 1, Seed: 1303787694, Size: 3x2, Model: Qwen-Image-2.1 Turbo, \
            Civitai resources: [{"type":"checkpoint","modelVersionId":3400828}], Version: Siliconed
            """)
        // Anima: its publisher's page, the same for the three variants.
        XCTAssertTrue(render(model: "anima", name: "Anima Turbo").parameters(version: nil).contains(
            #"Civitai resources: [{"type":"checkpoint","modelVersionId":3263843}]"#))
        XCTAssertEqual(Solver.allCases.map(Infotext.sampler), ["Euler", "Euler a", "DPM++ 2M", "UniPC"])
        // Krea 2: the page Civitai joins Comfy-Org's bf16 file to (19/10/2026).
        XCTAssertTrue(render(model: "krea2", name: "Krea 2 Turbo").parameters(version: nil).contains(
            #"Civitai resources: [{"type":"checkpoint","modelVersionId":3064584}]"#))
        // MOSS-Image: CivitaiOfficial's SFT, which holds the Compact's two files, and the DMD LoRA its 8 steps
        // run under (29/10/2026).
        XCTAssertTrue(render(model: "moss-image", name: "MOSS-Image").parameters(version: nil).contains(
            #"Civitai resources: [{"type":"checkpoint","modelVersionId":3409867},{"type":"lora","weight":1,"modelVersionId":3411385}], "#))
    }

    /// The weight goes in a prompt tag, the hash in `Lora hashes`, both under one sanitized name;
    /// a LoRA forged before the hash was kept is tagged and not hashed; a negative strength, which
    /// the tag cannot carry, is hashed only.
    func testLoRAsHashedAndNot() {
        let r = render(loras: [LoRAEntry("/s/pop.lora.silicon", strength: 0.7), LoRAEntry("/s/old.lora.silicon", strength: 1),
                               LoRAEntry("/s/neg.lora.silicon", strength: -0.25)],
                       origins: [.init(name: "popart01b-ce: zimgt (v2)", sha256: sha.uppercased()),
                                 .init(name: "Niji_semi_realism_v5", sha256: nil),
                                 .init(name: "neg", sha256: sha)])
        XCTAssertEqual(r.parameters(version: "0.8"), """
            woman posing in a library <lora:popart01b-ce__zimgt__v2_:0.70> <lora:Niji_semi_realism_v5:1.00>
            Steps: 8, Sampler: Euler, CFG scale: 1, Seed: 1303787694, Size: 3x2, Model: Z-Image Turbo, \
            Lora hashes: "popart01b-ce__zimgt__v2_: \(sha), neg: \(sha)", \
            Civitai resources: [{"type":"checkpoint","modelVersionId":2442439}], Version: Siliconed 0.8
            """)
        // A `Render` built without the stack (a test's): the file name stands for the name, no hash.
        let bare = render(loras: [LoRAEntry("/s/flat.lora.silicon", strength: 0.8)])
        XCTAssertTrue(bare.parameters(version: nil).hasPrefix("woman posing in a library <lora:flat.lora.silicon:0.80>\n"))
        XCTAssertFalse(bare.parameters(version: nil).contains("Lora hashes"))
    }

    /// img2img: the denoising strength; an imported fine-tune: its name (its colon `-`, quoted for
    /// its comma), and no Civitai page claimed for it.
    func testAnImg2ImgWithAnImportedModel() {
        let r = render(strength: 0.6, solver: .unipc, imported: true, name: "Real: v2, \"pro\"")
        XCTAssertEqual(r.parameters(version: "0.8"), """
            woman posing in a library
            Steps: 8, Sampler: UniPC, CFG scale: 1, Seed: 1303787694, Size: 3x2, Model: "Real- v2, \\"pro\\"", \
            Denoising strength: 0.60, Version: Siliconed 0.8
            """)
    }

    /// Through the PNG and back: the chunk is there, byte for byte, beside the app's keys, which
    /// still read back; and the named-subjects rewrite shows the typed prompt in it too.
    func testThePNGRoundTrip() throws {
        let r = render(prompt: "une femme « posant » dans une bibliothèque\nlumière douce",
                       loras: [LoRAEntry("/s/pop.lora.silicon", strength: 0.7)], origins: [.init(name: "pop", sha256: sha)])
        let text = PNG.text(try r.png())
        XCTAssertEqual(text["parameters"], r.parameters())
        XCTAssertEqual(text["prompt"], r.prompt)
        let recipe = try XCTUnwrap(Engine.Render.Recipe(metadata: text))
        XCTAssertEqual(recipe.prompt, r.prompt)
        XCTAssertEqual(recipe.unreadable, [])

        let named = NamedSubjects.metadata(r.metadata, typed: "@alice in a library", modelPrompt: r.prompt)
        XCTAssertEqual(named["parameters"], "@alice in a library <lora:pop:0.70>\n"
                       + r.parameters().split(separator: "\n", omittingEmptySubsequences: false).last!)
    }

    /// The stream against known digests: empty, `abc` (FIPS 180-2), and a file read in blocks of 7
    /// bytes against the one-shot hash.
    func testTheStreamedSHA256() throws {
        let folder = NSTemporaryDirectory() + "infotext-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: folder) }
        func file(_ name: String, _ data: Data) throws -> String {
            let path = folder + "/" + name
            try data.write(to: URL(fileURLWithPath: path))
            return path
        }
        XCTAssertEqual(try ForgeLoRA.sha256(file: file("empty", Data())),
                       "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        XCTAssertEqual(try ForgeLoRA.sha256(file: file("abc", Data("abc".utf8)), block: 2),
                       "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        let bytes = Data((0..<1000).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ 7) })
        let oneShot = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(try ForgeLoRA.sha256(file: file("bytes", bytes), block: 7), oneShot)
        XCTAssertThrowsError(try ForgeLoRA.sha256(file: folder + "/missing"))
        XCTAssertTrue(Infotext.isSHA256(oneShot))
        XCTAssertFalse(Infotext.isSHA256(String(oneShot.dropLast())))
        XCTAssertFalse(Infotext.isSHA256(String(oneShot.dropLast()) + "g"))
    }
}
