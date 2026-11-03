import Foundation
import XCTest
@testable import Siliconed

/// **A publisher's base model, refused at the import** (`BaseModels`) — on small files in a temporary
/// folder, tables of the test's own and headers written here: no weight, no network.
final class BaseModelsTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-base-models-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private func write(_ name: String, _ bytes: Data) throws -> String {
        let path = root.appendingPathComponent(name).path
        try bytes.write(to: URL(fileURLWithPath: path))
        return path
    }

    /// A safetensors of BF16 zeros under `names`, with `metadata`: the names a Qwen-Image-2.1 DiT is recognized by.
    private func qwenDiT(_ name: String, metadata: [String: String] = [:]) throws -> String {
        var table: [OrderedJSON.Pair] = []
        if !metadata.isEmpty {
            table.append(.init("__metadata__", .object(metadata.sorted { $0.key < $1.key }.map { .init($0.key, .string($0.value)) })))
        }
        var at = 0
        for n in ["transformer_blocks.0.img_mlp.gate_layer.weight", "txt_in.text_norm.weight"] {
            table.append(.init(n, .object([.init("dtype", .string("BF16")), .init("shape", .list([.integer(2)])),
                                           .init("data_offsets", .list([.integer(at), .integer(at + 4)]))])))
            at += 4
        }
        var json = Array(OrderedJSON.object(table).jsonText.utf8)
        while json.count % 8 != 0 { json.append(0x20) }
        var n = UInt64(json.count).littleEndian
        var file = withUnsafeBytes(of: &n) { Data($0) }
        file.append(contentsOf: json)
        file.append(Data(count: at))
        return try write(name, file)
    }

    /// The table of the test: `path`'s own size and sha256, as if it were a published base.
    private func known(_ path: String) throws -> BaseModels.Known {
        let size = try XCTUnwrap(FileManager.default.attributesOfItem(atPath: path)[.size] as? Int)
        return BaseModels.Known(family: .qwenImage21, file: .init(repository: .init(name: "a/base", revision: "r"), path: "base.safetensors",
                                                                destination: nil, sha256: try Downloader.sha256(path), size: size))
    }

    func testPublishedTableIsPinnedAndNeverTheTurbo() {
        let table = BaseModels.published()
        XCTAssertEqual(table.count, 87)
        let byFamily = Dictionary(grouping: table, by: \.family).mapValues(\.count)
        XCTAssertEqual(byFamily, [.qwenImage21: 23, .zImage: 36, .krea2: 19, .anima: 9])
        for k in table {
            XCTAssertEqual(k.file.sha256?.count, 64, k.file.path)
            XCTAssertEqual(k.file.sha256, k.file.sha256?.lowercased())
            XCTAssertEqual(k.file.repository.revision.count, 40, k.file.path)
            XCTAssertGreaterThan(k.file.size ?? 0, 1_000_000_000, k.file.path)
        }
        XCTAssertEqual(Set(table.compactMap(\.file.sha256)).count, table.count)
        // No file an installation fetches as a DiT, of any version of any family, is a base.
        let turbo = Set(Family.allCases.flatMap { f in f.variants.flatMap { Installer.dit(f, $0).compactMap(\.sha256) } })
        XCTAssertTrue(turbo.isDisjoint(with: table.compactMap(\.file.sha256)))
        // The reason the hash decides: Qwen's base shards have the Turbo's sizes to the byte, and so have the
        // Turbo files Krea 2's and Z-Image's installations fetch (Comfy-Org's bf16, unsloth's Q8_0).
        let turboSizes: Set<Int> = [9968332504, 4261951904]
        XCTAssertEqual(table.filter { turboSizes.contains($0.file.size!) }.count, 4)
        XCTAssertTrue(table.contains { $0.family == .krea2 && $0.file.size == 26283332608 })
        XCTAssertTrue(table.contains { $0.family == .zImage && $0.file.size == 7224707136 })
    }

    func testOtherSizeIsNotHashed() throws {
        let base = try known(try write("base", Data("the base's bytes".utf8)))
        let other = try write("other.safetensors", Data("a different length altogether".utf8))
        XCTAssertNoThrow(try BaseModels.refuse(other, among: [base], hash: { _ in XCTFail("hashed"); return "" }))
    }

    /// A Turbo of the base's size (a fine-tune saved like Comfy-Org's bf16): hashed, and let through.
    func testSameSizeOtherBytesPasses() throws {
        let base = try known(try write("base", Data("the base's bytes".utf8)))
        let lookalike = try write("finetune.safetensors", Data("the base's bytez".utf8))
        var hashed = 0
        XCTAssertNoThrow(try BaseModels.refuse(lookalike, among: [base], hash: { hashed += 1; return try Downloader.sha256($0) }))
        XCTAssertEqual(hashed, 1)
    }

    func testThePublishedBaseIsRefusedByName() throws {
        let path = try qwenDiT("qwen_image_2.1_bf16.safetensors")
        let table = [try known(path)]
        XCTAssertThrowsError(try BaseModels.refuse(path, among: table)) { e in
            XCTAssertEqual(e as? EngineError, .importBaseModel(file: "qwen_image_2.1_bf16.safetensors", model: "qwen-image-2.1"))
        }
        let sentence = EngineError.importBaseModel(file: "qwen_image_2.1_bf16.safetensors", model: "qwen-image-2.1").errorDescription ?? ""
        XCTAssertTrue(sentence.contains("the base model of Qwen-Image-2.1:"), sentence)
        XCTAssertTrue(sentence.contains("Siliconed renders Qwen-Image-2.1 Turbo"), sentence)
        // Each family's refusal names its own base and its own Turbo, by the catalog.
        for (f, base, turbo) in [(Family.zImage, "Z-Image", "Z-Image Turbo"), (.krea2, "Krea 2", "Krea 2 Turbo"), (.anima, "Anima", "Anima Turbo")] {
            XCTAssertThrowsError(try BaseModels.refuse(path, among: [BaseModels.Known(family: f, file: table[0].file)])) { e in
                XCTAssertEqual(e as? EngineError, .importBaseModel(file: "qwen_image_2.1_bf16.safetensors", model: ModelCard.of(f).id))
                XCTAssertEqual((e as? EngineError)?.errorDescription,
                               "qwen_image_2.1_bf16.safetensors is the base model of \(base): Siliconed renders \(turbo).")
            }
        }
    }

    /// **The header does not decide**: a file with the base's names and whatever its metadata says
    /// is not in the table, so it imports as before — a fine-tune of the Turbo or of the base alike.
    func testAHeaderAloneIsNotRefused() throws {
        let plain = try qwenDiT("finetune.safetensors", metadata: ["format": "pt"])
        let declared = try qwenDiT("declared.safetensors", metadata: ["modelspec.title": "Qwen-Image-2.1", "format": "pt"])
        for path in [plain, declared] {
            let (kind, family) = try ModelImport.recognize(path)
            XCTAssertEqual(kind, .model)
            XCTAssertEqual(family, .qwenImage21)
        }
    }

    /// The Hub's listing announces size and sha256: decided without a byte, the case of the hash ignored.
    func testHubListingIsJudgedOnWhatItAnnounces() throws {
        let gguf = try XCTUnwrap(BaseModels.published().first { $0.file.path == "qwen-image-2.1-Q4_K_M.gguf" })
        XCTAssertNotNil(BaseModels.known(size: gguf.file.size!, sha256: gguf.file.sha256!.uppercased()))
        // The Turbo's Q4_K_M, which the Light installs: 96 bytes more, another object.
        XCTAssertNil(BaseModels.known(size: 4199565120, sha256: "8940e8b522af886e0c0f5afddf720e82dd223cb9a43580b5f61c4d740452ac20"))
        XCTAssertNil(BaseModels.known(size: gguf.file.size! + 1, sha256: gguf.file.sha256!))
    }

    func testRememberedHashIsTheFilesOwn() throws {
        let path = try write("x.safetensors", Data("some bytes".utf8))
        let first = try BaseModels.Hashes.shared.sha256(path)
        XCTAssertEqual(first, try Downloader.sha256(path))
        XCTAssertEqual(try BaseModels.Hashes.shared.sha256(path), first)
    }
}
