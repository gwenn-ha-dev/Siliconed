import Foundation
import XCTest
@testable import Siliconed

/// **« Use this model »**: the link read as the web hands it (nothing trusted), the Hub's answer read
/// as it is written, and a LoRA judged on its header alone as the import judges it whole. No network.
final class HubLinkTests: XCTestCase {
    func testTheLinkIsReadAndNothingElse() throws {
        let l = try XCTUnwrap(HubLink(URL(string: "siliconed://hf?repo_id=nokolora/ffxiv-sphene&file=ff14-sphene-original_ANI.safetensors")!))
        XCTAssertEqual(l, HubLink(repository: "nokolora/ffxiv-sphene", file: "ff14-sphene-original_ANI.safetensors"))
        XCTAssertEqual(HubLink(l.url), l, "written as read")
        XCTAssertEqual(HubLink(URL(string: "SILICONED://HF?repo_id=unsloth/Z-Image-Turbo-GGUF&file=")!)?.file, nil)
        XCTAssertEqual(HubLink(URL(string: "siliconed://hf?repo_id=a/b&file=sub%20dir/x.gguf")!)?.file, "sub dir/x.gguf")
        for bad in ["https://hf?repo_id=a/b", "siliconed://other?repo_id=a/b", "siliconed://hf", "siliconed://hf?repo_id=a",
                    "siliconed://hf?repo_id=a/b/c", "siliconed://hf?repo_id=a--x/b", "siliconed://hf?repo_id=-a/b",
                    "siliconed://hf?repo_id=a/b.", "siliconed://hf?repo_id=a/b&file=../x.safetensors",
                    "siliconed://hf?repo_id=a/b&file=/etc/x.safetensors", "siliconed://hf?repo_id=a/b&file=x/../../y.gguf",
                    "siliconed://hf?repo_id=a/b&file=x%5Cy.gguf", "siliconed://hf?repo_id=a/b&file=x.bin",
                    "siliconed://hf?repo_id=a/b&file=x%0A.gguf", "siliconed://hf?repo_id=%C3%A9/b"] {
            XCTAssertNil(HubLink(URL(string: bad)!), bad)
        }
    }

    /// `api/models/<id>?blobs=true` as the Hub writes it: LFS files only, the commit pinned, the
    /// gate and the license read, shards, VAEs and encoders never offered, the link's file first.
    func testTheHubsAnswerIsRead() throws {
        let sha = String(repeating: "ab", count: 32)
        let json = #"""
        {"id": "someone/z-lora", "sha": "0123456789abcdef0123456789abcdef01234567", "gated": "auto",
         "tags": ["lora", "base_model:Tongyi-MAI/Z-Image-Turbo", "base_model:adapter:Tongyi-MAI/Z-Image-Turbo", "license:apache-2.0"],
         "cardData": {"license": "other"},
         "siblings": [
           {"rfilename": "README.md", "size": 900},
           {"rfilename": "b-epoch2.safetensors", "size": 1000, "lfs": {"sha256": "\#(sha)", "size": 1000}},
           {"rfilename": "a-epoch10.safetensors", "size": 1000, "lfs": {"sha256": "\#(sha)", "size": 1000}},
           {"rfilename": "a-epoch9.safetensors", "size": 1000, "lfs": {"sha256": "\#(sha)", "size": 1000}},
           {"rfilename": "vae/diffusion_pytorch_model.safetensors", "size": 1000, "lfs": {"sha256": "\#(sha)", "size": 1000}},
           {"rfilename": "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "size": 1000, "lfs": {"sha256": "\#(sha)", "size": 1000}},
           {"rfilename": "../escape.safetensors", "size": 1000, "lfs": {"sha256": "\#(sha)", "size": 1000}},
           {"rfilename": "short-sha.safetensors", "size": 1000, "lfs": {"sha256": "abc", "size": 1000}},
           {"rfilename": "q4.gguf", "size": 2000, "lfs": {"sha256": "\#(sha.uppercased())", "size": 2000}}
         ]}
        """#
        let r = try HubRepository(json: Data(json.utf8))
        XCTAssertEqual(r.commit, "0123456789abcdef0123456789abcdef01234567")
        XCTAssertTrue(r.gated)
        XCTAssertEqual(r.license, "other", "the card first, the tag failing it")
        XCTAssertEqual(r.declaredFamily, .zImage)
        XCTAssertNil(r.installs)
        XCTAssertEqual(r.files.count, 6, "\(r.files.map(\.path))")
        XCTAssertEqual(r.candidates().map(\.path), ["a-epoch9.safetensors", "a-epoch10.safetensors", "b-epoch2.safetensors", "q4.gguf"])
        XCTAssertEqual(r.candidates(preferring: "q4.gguf").first, .init(path: "q4.gguf", size: 2000, sha256: sha))
        XCTAssertEqual(r.downloadURL(r.files[0]).absoluteString,
                       "https://huggingface.co/someone/z-lora/resolve/0123456789abcdef0123456789abcdef01234567/b-epoch2.safetensors")

        let base = try HubRepository(json: Data(#"{"id": "Tongyi-MAI/Z-Image-Turbo", "sha": "f332072aa78be7aecdf3ee76d5c247082da564a6", "gated": false}"#.utf8))
        XCTAssertEqual(base.installs, .zImage)
        XCTAssertFalse(base.gated)
        XCTAssertNil(try HubRepository(json: Data(#"{"id": "krea/Krea-2", "sha": "f332072aa78be7aecdf3ee76d5c247082da564a6"}"#.utf8)).installs,
                     "a family's other repository is no installation")
        XCTAssertEqual(try HubRepository(json: Data(#"{"id": "OpenMOSS-Team/MOSS-Image-SFT", "sha": "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9"}"#.utf8)).installs,
                       .mossImage)
        XCTAssertEqual(try HubRepository(json: Data(#"{"id": "someone/moss-lora", "sha": "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9", "tags": ["base_model:adapter:OpenMOSS-Team/MOSS-Image-Base"]}"#.utf8)).declaredFamily,
                       .mossImage, "a LoRA trained on MOSS-Image Base applies to the SFT")
        for bad in [#"{"id": "a/b", "sha": "main"}"#, #"{"id": "../b", "sha": "f332072aa78be7aecdf3ee76d5c247082da564a6"}"#, "[]", "<html>"] {
            XCTAssertThrowsError(try HubRepository(json: Data(bad.utf8)), bad)
        }
    }

    /// The Hub's header, the rest of the file sparse: the forge's examination says what its forge
    /// of the whole file says.
    func testAHeaderAloneIsJudgedAsTheWholeFile() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-hub-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let f32 = { (v: [Float]) in v.withUnsafeBufferPointer { Data(buffer: $0) } }
        let whole = root.appendingPathComponent("whole.safetensors").path
        try ForgeDiT.writeSafetensors([
            ("transformer.transformer_blocks.0.attn.to_q.lora_A.weight", "F32", [2, 4], f32((0..<8).map(Float.init))),
            ("transformer.transformer_blocks.0.attn.to_q.lora_B.weight", "F32", [4, 2], f32((0..<8).map(Float.init))),
        ], to: whole)
        let bytes = try Data(contentsOf: URL(fileURLWithPath: whole))
        let n = Int(bytes.prefix(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        let sparse = root.appendingPathComponent("sparse.safetensors").path
        FileManager.default.createFile(atPath: sparse, contents: bytes.prefix(8 + n))
        let h = try FileHandle(forWritingTo: URL(fileURLWithPath: sparse))
        try h.truncate(atOffset: UInt64(bytes.count)); try h.close()

        let ref: (Family) -> [String: [Int]]? = { $0 == .qwenImage21 ? ["transformer_blocks.0.attn.to_q.weight": [4, 4]] : nil }
        let e = try ForgeLoRA.examine(file: sparse, reference: ref)
        let r = try ForgeLoRA.forge(file: whole, to: root.appendingPathComponent("w.lora.silicon").path, reference: ref)
        XCTAssertEqual(e.family, r.family)
        XCTAssertEqual(e.modules, r.modules)
        XCTAssertEqual(e.ranks, r.ranks)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("sparse.lora.silicon").path))
        XCTAssertThrowsError(try ForgeLoRA.examine(file: sparse, reference: { $0 == .qwenImage21 ? ["transformer_blocks.0.attn.to_q.weight": [4, 6]] : nil }))
    }
}
