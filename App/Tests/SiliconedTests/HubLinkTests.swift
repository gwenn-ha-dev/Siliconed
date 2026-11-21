import Foundation
import XCTest
@testable import Siliconed

/// **« Use this model »**: the link read as the web hands it (nothing trusted), the Hub's answer read as it is
/// written, and what an examination fetches of a file before its download. No network.
final class HubLinkTests: XCTestCase {
    func testTheLinkIsReadAndNothingElse() throws {
        let l = try XCTUnwrap(HubLink(URL(string: "siliconed://hf?repo_id=nokolora/ffxiv-sphene&file=ff14-sphene-original_ANI.safetensors")!))
        XCTAssertEqual(l, HubLink(repository: "nokolora/ffxiv-sphene", file: "ff14-sphene-original_ANI.safetensors"))
        XCTAssertEqual(HubLink(l.url), l, "written as read")
        XCTAssertEqual(HubLink(URL(string: "SILICONED://HF?repo_id=unsloth/Z-Image-Turbo-GGUF&file=")!)?.file, nil)
        // A GGUF repository opens — and offers nothing: an import takes a .safetensors.
        XCTAssertEqual(HubLink(URL(string: "siliconed://hf?repo_id=a/b&file=sub%20dir/x.gguf")!)?.file, "sub dir/x.gguf")
        XCTAssertFalse(HubLink.isFile("sub dir/x.gguf"))
        XCTAssertTrue(HubLink.isFile("sub dir/x.safetensors"))
        for bad in ["https://hf?repo_id=a/b", "siliconed://other?repo_id=a/b", "siliconed://hf", "siliconed://hf?repo_id=a",
                    "siliconed://hf?repo_id=a/b/c", "siliconed://hf?repo_id=a--x/b", "siliconed://hf?repo_id=-a/b",
                    "siliconed://hf?repo_id=a/b.", "siliconed://hf?repo_id=a/b&file=../x.safetensors",
                    "siliconed://hf?repo_id=a/b&file=/etc/x.safetensors", "siliconed://hf?repo_id=a/b&file=x/../../y.gguf",
                    "siliconed://hf?repo_id=a/b&file=x%5Cy.gguf", "siliconed://hf?repo_id=a/b&file=x.bin",
                    "siliconed://hf?repo_id=a/b&file=x%0A.gguf", "siliconed://hf?repo_id=%C3%A9/b"] {
            XCTAssertNil(HubLink(URL(string: bad)!), bad)
        }
    }

    /// `api/models/<id>?blobs=true` as the Hub writes it: LFS `.safetensors` only, the commit pinned, the gate and the
    /// license read, shards, VAEs and encoders never offered, the link's file first.
    func testTheHubsAnswerIsRead() throws {
        let sha = String(repeating: "ab", count: 32)
        let json = #"""
        {"id": "someone/z-lora", "sha": "0123456789abcdef0123456789abcdef01234567", "gated": "auto",
         "tags": ["lora", "license:apache-2.0"],
         "cardData": {"license": "other"},
         "siblings": [
           {"rfilename": "README.md", "size": 900},
           {"rfilename": "b-epoch2.safetensors", "size": 1000, "lfs": {"sha256": "\#(sha.uppercased())", "size": 1000}},
           {"rfilename": "a-epoch10.safetensors", "size": 1000, "lfs": {"sha256": "\#(sha)", "size": 1000}},
           {"rfilename": "a-epoch9.safetensors", "size": 1000, "lfs": {"sha256": "\#(sha)", "size": 1000}},
           {"rfilename": "vae/diffusion_pytorch_model.safetensors", "size": 1000, "lfs": {"sha256": "\#(sha)", "size": 1000}},
           {"rfilename": "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "size": 1000, "lfs": {"sha256": "\#(sha)", "size": 1000}},
           {"rfilename": "../escape.safetensors", "size": 1000, "lfs": {"sha256": "\#(sha)", "size": 1000}},
           {"rfilename": "short-sha.safetensors", "size": 1000, "lfs": {"sha256": "abc", "size": 1000}},
           {"rfilename": "q4.gguf", "size": 2000, "lfs": {"sha256": "\#(sha)", "size": 2000}}
         ]}
        """#
        let r = try HubRepository(json: Data(json.utf8))
        XCTAssertEqual(r.commit, "0123456789abcdef0123456789abcdef01234567")
        XCTAssertTrue(r.gated)
        XCTAssertEqual(r.license, "other", "the card first, the tag failing it")
        XCTAssertNil(r.installs)
        XCTAssertEqual(r.files.count, 5, "\(r.files.map(\.path))")
        XCTAssertEqual(r.candidates().map(\.path), ["a-epoch9.safetensors", "a-epoch10.safetensors", "b-epoch2.safetensors"])
        XCTAssertEqual(r.candidates(preferring: "b-epoch2.safetensors").first, .init(path: "b-epoch2.safetensors", size: 1000, sha256: sha))
        let published = r.published(r.files[0])
        XCTAssertEqual(published.repository.url(published.path).absoluteString,
                       "https://huggingface.co/someone/z-lora/resolve/0123456789abcdef0123456789abcdef01234567/b-epoch2.safetensors")
        XCTAssertEqual(published.digest, .sha256(sha))
        for bad in [#"{"id": "a/b", "sha": "main"}"#, #"{"id": "../b", "sha": "f332072aa78be7aecdf3ee76d5c247082da564a6"}"#, "[]", "<html>"] {
            XCTAssertThrowsError(try HubRepository(json: Data(bad.utf8)), bad)
        }
    }

    /// A family's installation is offered from its publisher's repository — where its license is read — and from a
    /// repository that holds the very DiT a version installs; from no other.
    func testAFamilysOwnRepositoryOpensItsInstallation() throws {
        let commit = String(repeating: "0", count: 40)
        for family in Family.allCases {
            let page = try XCTUnwrap(family.traits.card.license.urls.first)
            let id = page.pathComponents[1...2].joined(separator: "/")
            XCTAssertEqual(try HubRepository(json: Data(#"{"id": "\#(id)", "sha": "\#(commit)"}"#.utf8)).installs, family, id)
            for variant in family.variants {
                let dit = try XCTUnwrap(family.files(variant).first { $0.place == family.ditMap(variant) }, "\(family) \(variant)")
                guard case .sha256(let sha) = dit.digest else { return XCTFail("\(dit.label): a DiT is an LFS object") }
                let json = #"{"id": "mirror/of-it", "sha": "\#(commit)", "siblings": [{"rfilename": "dit.safetensors", "lfs": {"sha256": "\#(sha)", "size": \#(dit.bytes)}}]}"#
                XCTAssertEqual(try HubRepository(json: Data(json.utf8)).installs, family, dit.label)
            }
        }
        XCTAssertNil(try HubRepository(json: Data(#"{"id": "someone/else", "sha": "\#(commit)"}"#.utf8)).installs)
    }

    /// What is fetched beside the header: ComfyUI's quantization texts, near ones in one request; too scattered, the
    /// file is judged once downloaded. The examined file has the published size, and only those bytes.
    func testAnExaminationFetchesTheTextsTheImportReads() throws {
        func header(_ tensors: [(String, Int, Int)]) throws -> Data {
            let o = Dictionary(uniqueKeysWithValues: tensors.map { ($0.0, ["dtype": "U8", "shape": [$0.2 - $0.1], "data_offsets": [$0.1, $0.2]] as [String: Any]) })
            let json = try JSONSerialization.data(withJSONObject: o)
            return withUnsafeBytes(of: UInt64(json.count).littleEndian) { Data($0) } + json
        }
        let head = try header([("a.weight", 0, 1000), ("a.comfy_quant", 1000, 1040), ("b.comfy_quant", 1040, 1080),
                               ("c.comfy_quant", 5_000_000, 5_000_040), ("outside.comfy_quant", 9_999_990, 10_000_050)])
        let n = head.count, size = n + 10_000_000
        XCTAssertEqual(HubImport.contentRanges(header: head, size: size), [(n + 1000)..<(n + 1080), (n + 5_000_000)..<(n + 5_000_040)])
        XCTAssertNil(HubImport.contentRanges(header: head, size: size, requests: 1), "too scattered: judged once downloaded")
        XCTAssertNil(HubImport.contentRanges(header: head, size: size, bytes: 100))
        XCTAssertEqual(HubImport.contentRanges(header: try header([("a.weight", 0, 1000)]), size: size), [])
        XCTAssertEqual(HubImport.contentRanges(header: Data([1, 2, 3]), size: size), [])

        let path = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-hub-\(UUID()).safetensors").path
        defer { try? FileManager.default.removeItem(atPath: path) }
        try HubImport.sparse(head, size: size, contents: [((n + 1000)..<(n + 1004), Data(repeating: 7, count: 4))], at: path)
        let written = try Data(contentsOf: URL(fileURLWithPath: path), options: .alwaysMapped)
        XCTAssertEqual(written.count, size)
        XCTAssertEqual(written.prefix(n), head)
        XCTAssertEqual(Array(written[(n + 998)..<(n + 1006)]), [0, 0, 7, 7, 7, 7, 0, 0])
    }

    /// A partial answer is never a header: only a `206` of exactly the bytes asked; a `200`'s body (the whole file) is
    /// refused unread, a body longer than asked stopped at its first byte past the range.
    func testOnlyTheBytesAskedAreTaken() async throws {
        let u = URL(string: "https://huggingface.co/a/b/resolve/main/x.safetensors")!
        let answer = { (status: Int) in HTTPURLResponse(url: u, statusCode: status, httpVersion: nil, headerFields: nil)! }
        // A body of `count` bytes.
        func body(_ count: Int) -> AsyncStream<UInt8> { AsyncStream { c in for _ in 0..<count { c.yield(7) }; c.finish() } }
        let whole = try await Downloader.rangeBody(answer(206), body(8), 0, 7, file: "x")
        XCTAssertEqual(whole, Data(repeating: 7, count: 8))
        let readOfRefused = Tally()
        do {
            _ = try await Downloader.rangeBody(answer(200), body(1 << 16).map { b in readOfRefused.add(); return b }, 0, 7, file: "x")
            XCTFail("a 200 taken as a range")
        } catch {}
        XCTAssertEqual(readOfRefused.count, 0, "a 200's body read")
        let readOfLonger = Tally()
        do {
            _ = try await Downloader.rangeBody(answer(206), body(1 << 16).map { b in readOfLonger.add(); return b }, 0, 7, file: "x")
            XCTFail("a body longer than the range taken")
        } catch {}
        XCTAssertEqual(readOfLonger.count, 9, "read past the first byte beyond the range")
        do {
            _ = try await Downloader.rangeBody(answer(206), body(7), 0, 7, file: "x")
            XCTFail("a short body taken")
        } catch {}
    }

    /// The file waits in a folder of its own, named by its digest, under the library's uploads: the import throws
    /// the folder away, and a download stopped finds it again.
    func testADownloadWaitsInAFolderOfItsOwn() throws {
        let library = Library(root: FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-hub-\(UUID())"))
        let r = try HubRepository(json: Data(#"{"id": "someone/z-lora", "sha": "0123456789abcdef0123456789abcdef01234567"}"#.utf8))
        let f = HubRepository.File(path: "sub/dir/x.safetensors", size: 1000, sha256: String(repeating: "cd", count: 32))
        let destination = URL(fileURLWithPath: HubImport.destination(r, f, in: library))
        XCTAssertEqual(destination.lastPathComponent, "x.safetensors")
        XCTAssertEqual(destination.deletingLastPathComponent().lastPathComponent, "hub-someone--z-lora-cdcdcdcdcdcd")
        XCTAssertEqual(destination.deletingLastPathComponent().deletingLastPathComponent().path, UploadStore.directory(in: library).path)
    }
}

/// The bytes a stream's `map` saw: the map runs in the stream's task, so its count crosses into the test under a lock.
private final class Tally: @unchecked Sendable {
    private let lock = NSLock()
    private var seen = 0
    func add() { lock.withLock { seen += 1 } }
    var count: Int { lock.withLock { seen } }
}
