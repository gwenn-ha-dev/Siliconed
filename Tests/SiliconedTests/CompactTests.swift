import XCTest
@testable import Siliconed

/// **The Compact recipes, without a byte of weights**: every third-party file is
/// pinned (repository, revision, sha256), it is the very file that was judged (the fixtures of
/// `Fixtures/quant/`), and its whole published header — laid over a sparse file, as
/// the developer's weight check does — is what the forge reads and keeps 8-bit.
final class CompactTests: XCTestCase {
    static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/quant").path

    static let compactFamilies: [Family] = [.zImage, .qwenImage21]

    /// The third-party weights of a Compact: its DiT, then its encoder's (after the config).
    private func weights(_ f: Family) -> [Installer.File] {
        Installer.dit(f, .compact) + Installer.encoder(f, .compact).dropFirst()
    }

    func testOnlyZImageAndQwenHaveACompactAndALight() {
        for f in Family.allCases {
            XCTAssertEqual(f.variants, Self.compactFamilies.contains(f) ? [.standard, .compact, .light] : [.standard], f.name)
        }
    }

    /// **The Light**: one DiT file, a third party's, pinned (revision, sha256), the
    /// very file that was surveyed (its fixture: same repository, revision, path); **no tensor under 4 bits**
    /// in its published header (`header.json`, every tensor's type); its encoder is the Compact's,
    /// file for file; its license links the publisher's files, then the DiT's and the encoder's
    /// third-party cards; its DiT within 2 % of the published bytes, and the smallest of the three.
    func testTheLightIsTheJudgedQ4KMAndCarriesNothingUnderFourBits() throws {
        let hex = try NSRegularExpression(pattern: "^[0-9a-f]+$")
        func isHex(_ s: String) -> Bool { hex.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil }
        let allowed: Set<String> = ["F32", "F16", "BF16", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q5_1", "Q5_0", "Q4_1", "Q4_0"]
        for f in Self.compactFamilies {
            let dit = Installer.dit(f, .light)
            XCTAssertEqual(dit.count, 1)
            let x = dit[0]
            XCTAssertTrue(x.path.hasSuffix("Q4_K_M.gguf"), x.path)
            XCTAssertFalse(Set(Installer.dit(f).map(\.repository.name)).contains(x.repository.name))
            XCTAssertTrue(x.repository.revision.count == 40 && isHex(x.repository.revision))
            let sha = try XCTUnwrap(x.sha256)
            XCTAssertTrue(sha.count == 64 && isHex(sha))
            let d = try fixture(x)
            XCTAssertEqual(d.manifest["family"] as? String, f.rawValue)
            XCTAssertEqual(d.manifest["format"] as? String, "gguf")
            let header = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: d.folder + "/header.json"))) as? [String: Any])
            let tensors = try XCTUnwrap(header["tensors"] as? [String: [String: Any]])
            XCTAssertGreaterThan(tensors.count, 200)
            let types = Set(tensors.values.compactMap { $0["type"] as? String })
            XCTAssertTrue(types.isSubset(of: allowed), "\(f.name): \(types.subtracting(allowed)) under the floor")
            XCTAssertTrue(types.contains("Q4_K"), f.name)

            XCTAssertEqual(Installer.encoder(f, .light).map(\.path), Installer.encoder(f, .compact).map(\.path))
            XCTAssertEqual(Installer.encoder(f, .light).map(\.sha256), Installer.encoder(f, .compact).map(\.sha256))
            XCTAssertEqual(f.encoderMap(.light), f.encoderMap(.compact))

            let standard = Installer.licenseFiles(f), light = Installer.licenseFiles(f, .light)
            XCTAssertEqual(light.prefix(standard.count).map(\.repository.name), standard.map(\.repository.name))
            XCTAssertEqual(Set(light.dropFirst(standard.count).map(\.repository.name)),
                           Set([x.repository.name, Installer.encoder(f, .light)[1].repository.name]))
            XCTAssertTrue(light.dropFirst(standard.count).allSatisfy { $0.path == "README.md" })

            let size = f.installedSize(.light), compact = f.installedSize(.compact)
            let bytes = try XCTUnwrap(d.manifest["file_bytes"] as? Int)
            XCTAssertEqual(Double(size.dit), Double(bytes), accuracy: 0.02 * Double(bytes), "\(f.name) DiT")
            XCTAssertEqual(size.encoder, compact.encoder)
            XCTAssertEqual(size.components, compact.components)
            XCTAssertLessThan(size.dit, compact.dit, "\(f.name): the Light is the smallest")
        }
    }

    /// Repository, pinned revision, sha256 — and the publisher's `config.json`, at the Standard's revision.
    func testEveryCompactFileIsPinned() throws {
        let hex = try NSRegularExpression(pattern: "^[0-9a-f]+$")
        func isHex(_ s: String) -> Bool { hex.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil }
        for f in Self.compactFamilies {
            let w = weights(f)
            XCTAssertEqual(w.count, 2, "one DiT file, one encoder file")
            let publishers = Set(Installer.dit(f).map(\.repository.name))
            for x in w {
                XCTAssertFalse(x.repository.name.isEmpty)
                XCTAssertFalse(publishers.contains(x.repository.name), "\(x.path): a third party's, not the publisher's")
                XCTAssertTrue(x.repository.revision.count == 40 && isHex(x.repository.revision), "\(x.path): pinned revision")
                let sha = try XCTUnwrap(x.sha256, "\(x.path): sha256 pinned")
                XCTAssertTrue(sha.count == 64 && isHex(sha), "\(x.path): sha256")
                XCTAssertTrue(x.path.hasSuffix(".safetensors") || x.path.hasSuffix(".gguf"))
            }
            let config = try XCTUnwrap(Installer.encoder(f, .compact).first)
            let standardConfig = try XCTUnwrap(Installer.encoder(f).first { $0.path == "text_encoder/config.json" })
            XCTAssertEqual(config.path, "text_encoder/config.json")
            XCTAssertEqual(config.repository.name, standardConfig.repository.name, "the publisher's config, not the third party's")
            XCTAssertEqual(config.repository.revision, standardConfig.repository.revision)
            // The license read before the first byte: the Standard's files, then each third party's model card.
            let standard = Installer.licenseFiles(f), compact = Installer.licenseFiles(f, .compact)
            XCTAssertEqual(compact.prefix(standard.count).map(\.repository.name), standard.map(\.repository.name))
            XCTAssertEqual(Set(compact.dropFirst(standard.count).map(\.repository.name)), Set(w.map(\.repository.name)))
            XCTAssertTrue(compact.dropFirst(standard.count).allSatisfy { $0.path == "README.md" })
            XCTAssertEqual(Installer.licenseURLs(f, .compact).count, compact.count)
        }
    }

    /// The fixture judged in H/C1 for each Compact file — same repository, revision and path.
    private func fixture(_ x: Installer.File) throws -> (folder: String, manifest: [String: Any]) {
        let slugs = try FileManager.default.contentsOfDirectory(atPath: Self.fixtures)
        for s in slugs {
            let path = Self.fixtures + "/" + s + "/manifest.json"
            guard let data = FileManager.default.contents(atPath: path),
                  let m = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            if m["repo"] as? String == x.repository.name, m["revision"] as? String == x.repository.revision,
               m["file"] as? String == x.path {
                return (Self.fixtures + "/" + s, m)
            }
        }
        throw Numerics.Failure(description: "no fixture judged \(x.repository.name)@\(x.repository.revision)/\(x.path)")
    }

    /// **The Compact is what was judged**, and its size is its file's: each recipe file has its fixture
    /// (GGUF Q8_0, convrot, encoders); `installedSize(.compact)` within 2 % of the
    /// published bytes (the Z-Image encoder 3 %: its layer 35 is not written).
    func testCompactFilesAreTheJudgedOnesAndTheirSizesFollow() throws {
        for f in Self.compactFamilies {
            let (dit, encoder) = (Installer.dit(f, .compact)[0], Installer.encoder(f, .compact)[1])
            let d = try fixture(dit), e = try fixture(encoder)
            XCTAssertEqual(d.manifest["family"] as? String, f.rawValue)
            XCTAssertEqual(e.manifest["family"] as? String, f.rawValue)
            XCTAssertEqual(e.manifest["format"] as? String, "encoder")
            let size = f.installedSize(.compact), standard = f.installedSize(.standard)
            let ditBytes = try XCTUnwrap(d.manifest["file_bytes"] as? Int), encoderBytes = try XCTUnwrap(e.manifest["file_bytes"] as? Int)
            XCTAssertEqual(Double(size.dit), Double(ditBytes), accuracy: 0.02 * Double(ditBytes), "\(f.name) DiT")
            XCTAssertEqual(Double(size.encoder), Double(encoderBytes), accuracy: 0.03 * Double(encoderBytes), "\(f.name) encoder")
            XCTAssertEqual(size.components, standard.components, "tokenizers, VAE, turbo LoRA: the Standard's")
            let total = { (s: (dit: Int, encoder: Int, components: Int)) in s.dit + s.encoder + s.components }
            XCTAssertLessThan(Double(total(size)), 0.6 * Double(total(standard)), "\(f.name): Compact is the smaller one")
        }
    }

    /// **The headers are what the forge reads**: Comfy-Org's Qwen DiT, unsloth's Qwen encoder and
    /// Disty0's Z-Image encoder, whole published headers over sparse files — recognized, normalized,
    /// every weight kept 8-bit, nothing widened. (The GGUF's header bytes are not versioned: its
    /// 521/521 names were judged on the real file.)
    func testCompactHeadersAreForgedEightBit() throws {
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-compact-\(UUID())")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        // The DiT of Qwen-Image-2.1 Compact.
        let qwenDiT = try fixture(Installer.dit(.qwenImage21, .compact)[0])
        let ditSource = try TensorSource(paths: [try sparse(qwenDiT.folder, scratch)])
        XCTAssertEqual(Recipes.family(fromNames: ditSource.names), .qwenImage21)
        let n = try Recipes.normalize(ditSource, family: .qwenImage21)
        XCTAssertEqual(n.dit.count, 297, "M114: 297/297 names")
        var kept = 0
        for (name, p) in n.dit {
            XCTAssertNil(p.dequantizedBecause, name)
            guard p.quantization != nil else { continue }
            let (t, outcome) = try ForgeDiT.tensor(name, p, family: .qwenImage21)
            XCTAssertEqual(outcome, .kept8bit, name)
            XCTAssertNotNil(t.rotation, "\(name): convrot")
            kept += 1
        }
        XCTAssertGreaterThan(kept, 200)

        // The two encoders.
        for f in Self.compactFamilies {
            let e = try fixture(Installer.encoder(f, .compact)[1])
            let source = try TensorSource(paths: [try sparse(e.folder, scratch)])
            let config = try OrderedJSON.read(e.folder + "/config.json")
            let plan = try ForgeText.plan(source, published: config, family: f)
            var eightBit = 0
            for name in plan.order {
                let published = plan.kept[name]!
                guard source.quantization(published) != nil else { continue }
                let (_, outcome) = try ForgeText.tensor(name, published: published, source: source, plan: plan)
                XCTAssertEqual(outcome, .kept8bit, "\(f.name) \(name)")
                eightBit += 1
            }
            XCTAssertGreaterThan(eightBit, 100, f.name)
        }
    }

    /// The published header over a sparse file of the published size, ComfyUI's `comfy_quant`
    /// texts written in (`QuantCheck.swift`, `sparseSource`).
    private func sparse(_ folder: String, _ scratch: URL) throws -> String {
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: folder + "/header.json"))) as! [String: Any]
        var json = try JSONSerialization.data(withJSONObject: object)
        while json.count % 8 != 0 { json.append(0x20) }
        let end = object.values.compactMap { ($0 as? [String: Any])?["data_offsets"] as? [Int] }.map { $0[1] }.max() ?? 0
        let path = scratch.appendingPathComponent((folder as NSString).lastPathComponent + ".safetensors").path
        var file = withUnsafeBytes(of: UInt64(json.count).littleEndian) { Data($0) }
        file.append(json)
        try file.write(to: URL(fileURLWithPath: path))
        XCTAssertEqual(truncate(path, off_t(8 + json.count + end)), 0)
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: folder + "/manifest.json"))) as! [String: Any]
        if let texts = manifest["comfy_quant"] as? [String: String] {
            let handle = try XCTUnwrap(FileHandle(forWritingAtPath: path))
            defer { try? handle.close() }
            for (name, text) in texts {
                let o = try XCTUnwrap((object[name] as? [String: Any])?["data_offsets"] as? [Int])
                try handle.seek(toOffset: UInt64(8 + json.count + o[0]))
                handle.write(Data(text.utf8))
            }
        }
        return path
    }
}
