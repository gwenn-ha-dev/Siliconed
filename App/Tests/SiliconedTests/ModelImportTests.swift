import Foundation
import XCTest
import SiliCore
@testable import Siliconed

/// **A model brought by the user, copied in whole, listed by its library, refused where it is not one.** The files are
/// written here with names and shapes and no value: a header, then a hole of the data's length — the import reads no
/// further than the header. What each family's DiT reads is the new engine's to say (`FamilyTraits.ditTensors`, not
/// ported): until then no model is recognized.
final class ModelImportTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-import-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private func file(_ name: String, _ tensors: [String: [Int]], dtype: String = "BF16", prefix: String = "") throws -> String {
        var header: [String: Any] = [:], offset = 0
        let width = dtype == "F32" ? 4 : 2
        for (key, shape) in tensors.sorted(by: { $0.key < $1.key }) {
            let bytes = shape.reduce(1, *) * width
            header[prefix + key] = ["dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]]
            offset += bytes
        }
        let json = try JSONSerialization.data(withJSONObject: header)
        let path = root.appendingPathComponent(name).path
        try (withUnsafeBytes(of: UInt64(json.count).littleEndian) { Data($0) } + json).write(to: URL(fileURLWithPath: path))
        let handle = try XCTUnwrap(FileHandle(forWritingAtPath: path))
        try handle.truncate(atOffset: UInt64(8 + json.count + offset))
        try handle.close()
        return path
    }

    /// A DiT's prefix, found by a key of its own: none, a checkpoint's, or no known one.
    func testADiTsPrefixIsFoundByItsOwnKey() throws {
        let marker = "blocks.0.attn.wq.weight"
        for prefix in [""] + MapWeights.checkpointPrefixes {
            let file = try SafeTensors(try file("dit-\(prefix.count).safetensors", [marker: [4, 4], "txtfusion.x": [4]], prefix: prefix))
            XCTAssertEqual(MapWeights.checkpointPrefix(of: file, marker: marker), prefix)
            XCTAssertNotNil(try SafeTensors(file.path, under: prefix).tensors[marker])
        }
        let other = try SafeTensors(try file("other.safetensors", [marker: [4, 4]], prefix: "unet."))
        XCTAssertNil(MapWeights.checkpointPrefix(of: other, marker: marker))
    }

    /// An import replaces a file of the same name whole or not at all, and copies the file a link points to — never
    /// the link, never a partial copy left under the final name.
    func testAnImportCopiesTheFileItselfAndReplacesWhole() throws {
        let fm = FileManager.default
        let source = root.appendingPathComponent("source.safetensors").path, link = root.appendingPathComponent("link.safetensors").path
        let target = root.appendingPathComponent("in/target.safetensors").path
        try fm.createDirectory(atPath: (target as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        try Data("old".utf8).write(to: URL(fileURLWithPath: target))
        try Data("new bytes".utf8).write(to: URL(fileURLWithPath: source))
        try fm.createSymbolicLink(atPath: link, withDestinationPath: source)
        try ModelImport.copyIn(link, to: target)
        XCTAssertEqual(fm.contents(atPath: target), Data("new bytes".utf8))
        XCTAssertNil(try? fm.destinationOfSymbolicLink(atPath: target), "the file, not a link")
        XCTAssertFalse(fm.fileExists(atPath: Downloader.partial(target)))
        // The file already in place: nothing to do.
        try ModelImport.copyIn(target, to: target)
        XCTAssertEqual(fm.contents(atPath: target), Data("new bytes".utf8))
        // A source that does not read leaves the old file as it was.
        XCTAssertThrowsError(try ModelImport.copyIn(root.appendingPathComponent("absent").path, to: target))
        XCTAssertEqual(fm.contents(atPath: target), Data("new bytes".utf8))
    }

    /// **« Stop » during an import**: the copy throws `cancelled`, the library keeps the file it had under that name — never
    /// a partial one —, no `.partiel` is left, and the user's file is as it was. A token that is not raised copies whole.
    func testAStoppedImportLeavesTheLibraryAndTheUsersFileAsTheyWere() throws {
        let fm = FileManager.default
        let source = root.appendingPathComponent("users.safetensors").path
        let target = root.appendingPathComponent("in/target.safetensors").path
        try fm.createDirectory(atPath: (target as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        let bytes = Data((0..<(3 << 20)).map { UInt8(truncatingIfNeeded: $0 &* 31) })
        try bytes.write(to: URL(fileURLWithPath: source))
        try Data("old".utf8).write(to: URL(fileURLWithPath: target))
        let stop = Cancellation()
        stop.cancel()
        XCTAssertThrowsError(try ModelImport.copyIn(source, to: target, cancellation: stop)) { error in
            XCTAssertEqual(error as? EngineError, .cancelled)
        }
        XCTAssertEqual(fm.contents(atPath: target), Data("old".utf8), "the library's file under that name, untouched")
        XCTAssertFalse(fm.fileExists(atPath: Downloader.partial(target)), "nothing half-copied left")
        XCTAssertEqual(fm.contents(atPath: source), bytes, "the user's file, only read")
        // The whole import stops before it reads the file: nothing in the library.
        let library = Library(root: root.appendingPathComponent("library"))
        XCTAssertThrowsError(try ModelImport.importFile(source, in: library, cancellation: stop)) { error in
            XCTAssertEqual(error as? EngineError, .cancelled)
        }
        XCTAssertEqual(fm.contents(atPath: source), bytes)
        // Not raised, the token lets the copy through whole.
        try ModelImport.copyIn(source, to: target, cancellation: Cancellation())
        XCTAssertEqual(fm.contents(atPath: target), bytes)
        XCTAssertFalse(fm.fileExists(atPath: Downloader.partial(target)))
        XCTAssertEqual(fm.contents(atPath: source), bytes)
    }

    /// A render's LoRA is credited by its file's name and sha256 — what Civitai joins its hashes on.
    func testALoRAIsKnownByItsFilesSHA256() throws {
        let path = root.appendingPathComponent("Pop_Art.safetensors").path
        try Data("lora".utf8).write(to: URL(fileURLWithPath: path))
        let origin = Engine.Render.LoRAOrigin(LoRAEntry(path, strength: 0.7))
        XCTAssertEqual(origin.name, "Pop_Art")
        XCTAssertEqual(origin.sha256, try Downloader.sha256(path))
        XCTAssertEqual(origin.sha256?.count, 64)
        XCTAssertNil(Engine.Render.LoRAOrigin(LoRAEntry(root.appendingPathComponent("gone.safetensors").path)).sha256)
    }

    /// A publisher's undistilled model is refused for its bytes: the size first, the hash only when a size agrees.
    func testAPublishedBaseModelIsRefused() throws {
        let tuned = try file("tuned.safetensors", ["net.x": [4, 4]])

        let base = try XCTUnwrap(Family.anima.traits.baseCheckpoints.first)
        let sized = root.appendingPathComponent("sized.safetensors").path
        FileManager.default.createFile(atPath: sized, contents: nil)
        let handle = try XCTUnwrap(FileHandle(forWritingAtPath: sized))
        try handle.truncate(atOffset: UInt64(base.bytes))
        try handle.close()
        XCTAssertThrowsError(try ModelImport.refuseBase(sized, .anima, hash: { _ in base.sha256 })) { error in
            guard case EngineError.importBaseModel(_, let model) = error else { return XCTFail("\(error)") }
            XCTAssertEqual(model, ModelCard.of(.anima).id)
        }
        XCTAssertNoThrow(try ModelImport.refuseBase(sized, .anima, hash: { _ in "another" }))
        XCTAssertNoThrow(try ModelImport.refuseBase(tuned, .anima, hash: { _ in XCTFail("hashed at another size"); return "" }))
        for family in Family.allCases {
            for known in family.traits.baseCheckpoints { XCTAssertEqual(known.sha256.count, 64, known.source) }
        }
    }

    /// What is neither a model nor a LoRA is refused by name.
    func testWhatIsNotReadIsRefusedByName() throws {
        let other = try file("other.safetensors", ["some.weight": [4, 4]])
        XCTAssertThrowsError(try ModelImport.recognize(other)) { error in
            guard case EngineError.importRefused(_, let detail) = error else { return XCTFail("\(error)") }
            XCTAssertEqual(detail, "neither a LoRA nor a model of a family this app renders")
        }
    }

    /// The library lists what its folders hold: a card of the family's, under the file's name, reading its file —
    /// and removed with it.
    func testTheLibraryListsItsImportedModels() throws {
        let library = Library(root: root.appendingPathComponent("library"))
        let folder = library.importedFolder(.anima)
        try FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)
        try FileManager.default.copyItem(atPath: try file("my_model.safetensors", ["x": [1]]), toPath: folder + "/my_model.safetensors")
        let card = try XCTUnwrap(library.card("anima/my_model"))
        XCTAssertEqual(card.name, "my model")
        XCTAssertEqual(card.family, .anima)
        XCTAssertTrue(card.isImported)
        XCTAssertEqual(card.license, ModelCard.of(.anima).license)
        XCTAssertEqual(library.cards().map(\.id), ModelCard.allCards.map(\.id) + ["anima/my_model"])
        XCTAssertThrowsError(try ModelImport.remove(ModelCard.of(.anima), in: library))
        try ModelImport.remove(card, in: library)
        XCTAssertNil(library.card("anima/my_model"))
    }
}
