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

    /// Each denoiser in the repo announces the model that `Model.identifiers` names.
    func testTheDenoisersNameTheirModel() {
        XCTAssertEqual([ZImageDenoisingModule(map: "").model, AnimaDenoisingModule(map: "").model,
                        Krea2DenoisingModule(map: "").model,
                        MossImageDenoisingModule(map: "", turbo: "").model,
                        QwenImage21DenoisingModule(map: "").model], Model.identifiers)
    }

    // ── the library ──────────────────────────────────────────────────────────────────

    /// Paths derive from the root alone — everything under `store/`, without a file.
    func testTheLibraryDerivesItsPaths() {
        let b = Library(root: URL(fileURLWithPath: "/Modèles/Siliconed"))
        XCTAssertEqual(b.map("anima-turbo-v1.1.safetensors"), "/Modèles/Siliconed/store/anima-turbo-v1.1.safetensors")
        XCTAssertEqual(b.importedFolder(.anima), "/Modèles/Siliconed/store/importes/anima")
        XCTAssertThrowsError(try b.component(.krea2, "vae.safetensors")) {
            XCTAssertEqual(($0 as? MissingFile)?.path, "/Modèles/Siliconed/store/composants/krea2/vae.safetensors")
        }
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
