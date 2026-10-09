import XCTest
@testable import Siliconed

/// **The API's wiring** — what plugs together, what is refused, and the LoRA stack read at the edge.
///
/// No map is opened: a module only touches its files when computing, and `Chain` judges the
/// wiring on the types alone. This is what makes it testable here.
final class ChainTests: XCTestCase {

    func testTheThreeChainsOfTheRepositoryPlugIn() throws {
        _ = try Chain(text: ZImageTextModule(map: "", tokenizer: ""),
                       denoising: ZImageDenoisingModule(map: ""), decoding: FluxDecodingModule(path: ""))
        _ = try Chain(text: AnimaTextModule(map: "", tokenizerQwen: "", tokenizerT5: "", adapter: ""),
                       denoising: AnimaDenoisingModule(map: ""), decoding: QwenImageDecodingModule(path: ""))
        _ = try Chain(text: Krea2TextModule(map: "", tokenizer: "", sockets: []),
                       denoising: Krea2DenoisingModule(map: ""), decoding: QwenImageDecodingModule(path: ""))
    }

    /// A text encoded for Anima does not condition Krea 2's DiT.
    func testATextOfTheWrongFormatIsRefused() {
        XCTAssertThrowsError(try Chain(
            text: AnimaTextModule(map: "", tokenizerQwen: "", tokenizerT5: "", adapter: ""),
            denoising: Krea2DenoisingModule(map: ""), decoding: QwenImageDecodingModule(path: ""))) {
            guard case EngineError.incompatibleChain = $0 else { return XCTFail("\($0)") }
        }
    }

    /// Sixteen channels on both sides, and yet two spaces: Flux is not decoded by Qwen-Image.
    func testALatentFromAnotherSpaceIsRefused() {
        XCTAssertThrowsError(try Chain(
            text: ZImageTextModule(map: "", tokenizer: ""),
            denoising: ZImageDenoisingModule(map: ""), decoding: QwenImageDecodingModule(path: ""))) {
            XCTAssertEqual($0 as? EngineError, .incompatibleChain(output: "\(LatentSpace.flux)", input: "\(LatentSpace.qwenImage)"))
        }
    }

    func testTheLoRAStackReadsLikeSILICONED_LORA() {
        let stack = LoRAEntry.stack("store/popart.lora.silicon:0.5,store/niji.lora.silicon")
        XCTAssertEqual(stack, [LoRAEntry("store/popart.lora.silicon", strength: 0.5),
                              LoRAEntry("store/niji.lora.silicon", strength: 1)])
        // The tag is the one that file names already carried: `popart50`.
        XCTAssertEqual(stack.map(\.shortTag), ["popart50", "niji100"])
    }

    // ── a LoRA's target ──────────────────────────────────────────────────────────────

    /// The header as the LoRA forge has long written it — without a file.
    private func header(_ model: String?) -> [String: Any] {
        var h: [String: Any] = ["kind": "lora", "nom": "Niji_semi_realism_v5"]
        if let model { h["cible"] = ["modele": model, "kind": "\(model)-turbo-dit"] }
        return h
    }

    /// Each denoiser in the repo announces the model that `Model.identifiers` names — this is the
    /// target a LoRA must declare.
    func testTheDenoisersNameTheirModel() {
        XCTAssertEqual([ZImageDenoisingModule(map: "").model, AnimaDenoisingModule(map: "").model,
                        Krea2DenoisingModule(map: "").model, KleinDenoisingModule(map: "").model,
                        ErnieDenoisingModule(map: "").model,
                        QwenImage21DenoisingModule(map: "", turbo: "", scheduler: .init()).model], Model.identifiers)
    }

    func testALoRAOfTheRightTargetIsAccepted() throws {
        try LoRA.checkTarget(header("anima"), path: "niji.lora.silicon", model: "anima")
    }

    /// Anima's on Krea 2 would touch no module: refused, and the error says why.
    func testALoRAFromAnotherModelIsRefused() {
        XCTAssertThrowsError(try LoRA.checkTarget(header("anima"), path: "niji.lora.silicon",
                                                    model: "krea2")) {
            guard case let LoRA.Failure.wrongTarget(_, name, forgedFor, model) = $0 else {
                return XCTFail("\($0)")
            }
            XCTAssertEqual([name, forgedFor, model], ["Niji_semi_realism_v5", "anima", "krea2"])
        }
    }

    /// A map forged before the forge wrote its target is not guessed at: it is refused.
    func testALoRAWithoutTargetIsRefused() {
        var oldHeader = header(nil)
        oldHeader["base"] = "anima"          // what the trainer declared — not a target
        XCTAssertThrowsError(try LoRA.checkTarget(oldHeader, path: "niji.lora.silicon", model: "anima")) {
            guard case LoRA.Failure.missingTarget = $0 else { return XCTFail("\($0)") }
        }
    }

    // ── the library ──────────────────────────────────────────────────────────────────

    /// Paths derive from the root alone — everything under `store/`, without a file.
    func testTheLibraryDerivesItsPaths() {
        let b = Library(root: URL(fileURLWithPath: "/Modèles/Siliconed"))
        XCTAssertEqual(b.map("anima-turbo-dit.v0.silicon"), "/Modèles/Siliconed/store/anima-turbo-dit.v0.silicon")
        XCTAssertEqual(b.profile, "/Modèles/Siliconed/store/profil.json")
        XCTAssertEqual(b.imported.path, "/Modèles/Siliconed/store/importes")
        XCTAssertThrowsError(try b.component(.krea2, "vae.safetensors")) {
            XCTAssertEqual(($0 as? MissingFile)?.path, "/Modèles/Siliconed/store/composants/krea2/vae.safetensors")
        }
        XCTAssertEqual(adapterPath(fromMap: "/s/anima-turbo-dit.v0.silicon"), "/s/anima-turbo-dit.v0.adaptateur.safetensors")
    }

    func testNoiseRespectsTheSpaceAndTheSeed() {
        let a = Latent.noise(.qwenImage, height: 64, width: 64, seed: 42)
        XCTAssertEqual(a.values.count, 16 * 64 * 64)
        XCTAssertEqual(a.values, Latent.noise(.qwenImage, height: 64, width: 64, seed: 42).values)
        XCTAssertNotEqual(a.values, Latent.noise(.qwenImage, height: 64, width: 64, seed: 7).values)
        // A rectangle: `channels · h · w` values, drawn in the same planar order.
        let r = Latent.noise(.flux, height: 152, width: 104, seed: 42)
        XCTAssertEqual(r.values.count, 16 * 152 * 104)
        XCTAssertEqual(Array(r.values.prefix(4096)), Array(a.values.prefix(4096)))
    }

    /// The square init is only a short spelling: a single way of storing, width × height.
    func testTheSquareRequestIsAnEqualRectangle() {
        let r = Request("une femme dans une bibliothèque", resolution: 512)
        XCTAssertEqual(r.width, 512); XCTAssertEqual(r.height, 512)
        let p = Request("une femme dans une bibliothèque", width: 832, height: 1216)
        XCTAssertEqual(p.width, 832); XCTAssertEqual(p.height, 1216)
    }
}
