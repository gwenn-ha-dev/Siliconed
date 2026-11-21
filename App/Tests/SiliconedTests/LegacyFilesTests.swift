import Foundation
import XCTest
@testable import Siliconed

/// **What version 0.21 left in a library is told by name and freed whole; nothing this app reads goes with it.**
final class LegacyFilesTests: XCTestCase {
    func testOnlyWhatTheEarlierVersionForgedIsFreed() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-legacy-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let library = Library(root: root)
        func write(_ relative: String, _ bytes: Int) throws {
            let url = root.appendingPathComponent("store/" + relative)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(count: bytes).write(to: url)
        }
        let forged = ["z-image-turbo-dit.v1.silicon": 100, "anima-turbo-dit.v0.adaptateur.safetensors": 20,
                      "composants/moss-image/turbo.lora.silicon": 7, "composants/z-image/dit.json": 3,
                      "composants/anima/transformer.json": 2, "composants/qwen-image-2.1/vae.json": 1,
                      "composants/qwen-image-2.1/model_index.json": 4, "composants/qwen-image-2.1/scheduler.json": 5,
                      "composants/qwen-image-2.1/processor/tokenizer.json": 6]
        let read = ["z_image_turbo_bf16.safetensors": 50, "composants/z-image/vae.safetensors": 9, "composants/z-image/tokenizer/tokenizer.json": 4,
                    "composants/z-image/variant": 1, "loras/z-image/style.safetensors": 5, "importes/z-image/mine.safetensors": 6,
                    "composants/anima/tokenizer/vae.json": 2, "composants/anima/tokenizer/scheduler.json": 3]
        for (name, bytes) in forged.merging(read, uniquingKeysWith: { a, _ in a }) { try write(name, bytes) }
        // 0.21's render cache, on this Mac: nothing reads it any more.
        let kv = root.appendingPathComponent("cache/kv/a.kv")
        try FileManager.default.createDirectory(at: kv.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(count: 10).write(to: kv)

        let found = library.legacyFiles()
        XCTAssertEqual(Set(found.map { $0.path.replacingOccurrences(of: root.path + "/store/", with: "") }), Set(forged.keys).union([kv.path]))
        XCTAssertEqual(library.occupancy().legacy, 158)
        XCTAssertEqual(try library.releaseLegacy(), 158)
        XCTAssertTrue(library.legacyFiles().isEmpty)
        for name in read.keys { XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("store/" + name).path), name) }
    }
}
