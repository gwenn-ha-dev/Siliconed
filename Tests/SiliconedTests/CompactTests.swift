import CryptoKit
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

    func testEveryFamilyHasACompactAndALight() {
        // MOSS-Image has a Standard and a Compact (`MossImageFamilyTests`): no Light.
        for f in Family.allCases where f != .mossImage {
            XCTAssertEqual(f.variants, [.standard, .compact, .light], f.name)
            XCTAssertEqual(Set(Variant.allCases.map(f.ditMap)).count, 3, f.name)
        }
    }

    /// **Krea 2 installs without a byte of Krea's gated repository**: its three versions read
    /// Comfy-Org's files (the Standard's bf16 DiT and encoder, the Compact's int8 convrot), Qwen's
    /// tokenizer and VAE, our SQ 4-bit file for the Light — all pinned (revision, sha256); no
    /// `config.json` is downloaded (`Krea2Published`). The Compact's files are the
    /// judged ones (their fixtures); the Light is our own file, pinned like the others. Its license: Krea's `LICENSE.pdf` (from
    /// Comfy-Org's copy) and Comfy-Org's card, the cards of the two Qwen repositories its VAE and its
    /// tokenizer come from, then our repository's card, NOTICE and license for the Light.
    func testKrea2VersionsReadNoGatedFileAndAreTheJudgedOnes() throws {
        let hex = try NSRegularExpression(pattern: "^[0-9a-f]+$")
        func isHex(_ s: String) -> Bool { hex.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil }
        for v in Family.krea2.variants {
            let files = Installer.components(.krea2) + Installer.encoder(.krea2, v) + Installer.dit(.krea2, v)
                + Installer.licenseFiles(.krea2, v)
            XCTAssertFalse(files.contains { $0.repository.name.hasPrefix("krea/") }, v.rawValue)
            XCTAssertFalse(files.contains { $0.path.hasSuffix("config.json") || $0.path == "model_index.json" }, v.rawValue)
            for x in Installer.encoder(.krea2, v) + Installer.dit(.krea2, v) {
                XCTAssertTrue(x.repository.revision.count == 40 && isHex(x.repository.revision), x.path)
                let sha = try XCTUnwrap(x.sha256, "\(x.path): sha256 pinned")
                XCTAssertTrue(sha.count == 64 && isHex(sha), x.path)
            }
            XCTAssertEqual(Installer.licenseFiles(.krea2, v).prefix(2).map(\.path), ["LICENSE.pdf", "README.md"])
        }
        XCTAssertNotNil(Installer.publishedEncoderConfig(.krea2))
        XCTAssertEqual(Installer.licenseFiles(.krea2, .compact).count, 4, "Comfy-Org's card is the Standard's")
        // Every repository a byte is downloaded from has its card listed (Krea's three small files are in the code).
        let downloaded = Set((Installer.components(.krea2) + Installer.encoder(.krea2, .standard) + Installer.dit(.krea2, .standard))
            .map(\.repository.name))
        XCTAssertEqual(Set(Installer.licenseFiles(.krea2).map(\.repository.name)), downloaded)
        XCTAssertEqual(Installer.licenseFiles(.krea2).dropFirst().map(\.path), ["README.md", "README.md", "README.md"])
        XCTAssertEqual(Installer.licenseFiles(.krea2, .light).last?.repository.name, Installer.dit(.krea2, .light)[0].repository.name)
        XCTAssertEqual(Installer.encoder(.krea2, .light).map(\.sha256), Installer.encoder(.krea2, .compact).map(\.sha256))
        XCTAssertEqual(Family.krea2.encoderMap(.light), Family.krea2.encoderMap(.compact))
        XCTAssertNotEqual(Family.krea2.encoderMap(.compact), Family.krea2.encoderMap)

        // The judged files: Comfy-Org's convrot DiT and encoder.
        for (x, format) in [(Installer.dit(.krea2, .compact)[0], "convrot"), (Installer.encoder(.krea2, .compact)[0], "encoder")] {
            let d = try fixture(x)
            XCTAssertEqual(d.manifest["family"] as? String, "krea2", x.path)
            XCTAssertEqual(d.manifest["format"] as? String, format, x.path)
        }
        // The Light: our repository's three files after the Standard's, the weights pinned with their size.
        let light = Installer.dit(.krea2, .light)[0]
        XCTAssertEqual(light.repository.name, "gwenn-ha-dev/Krea-2-Turbo-Siliconed-SQ4")
        XCTAssertEqual(Installer.licenseFiles(.krea2, .light).suffix(3).map(\.path), ["README.md", "NOTICE", "LICENSE.pdf"])
        XCTAssertEqual(light.path, "krea-2-turbo-sq4.safetensors")
        XCTAssertEqual(light.size, 7_567_580_968)
        XCTAssertEqual(light.sha256, "8b0f96a6649a695ae349189fa4d82b3a45e172cdcccc227c8f61c343122bc4ba")

        // Sizes: the Compact under the Standard, the Light under the Compact; the encoder shared.
        let s = Family.krea2.installedSize(.standard), c = Family.krea2.installedSize(.compact), l = Family.krea2.installedSize(.light)
        XCTAssertLessThan(l.dit, c.dit)
        XCTAssertLessThan(c.dit, s.dit)
        XCTAssertLessThan(c.encoder, s.encoder)
        XCTAssertEqual(l.encoder, c.encoder)
        XCTAssertEqual(Set([s.components, c.components, l.components]).count, 1)
        let sq4 = try XCTUnwrap(light.size)
        XCTAssertEqual(Double(l.dit), Double(sq4), accuracy: 0.02 * Double(sq4))
        let convrot = try XCTUnwrap(try fixture(Installer.dit(.krea2, .compact)[0]).manifest["file_bytes"] as? Int)
        // Above the file by the bf16 tensors the forge writes in fp32 (44 of them, `quant-weights`
        // counts them; 0.65 GB measured), never under it.
        XCTAssertGreaterThan(c.dit, convrot)
        XCTAssertEqual(Double(c.dit), Double(convrot), accuracy: 0.06 * Double(convrot))
    }

    /// **Krea's three small files are in the code, verbatim**: each literal hashes, as git hashes a
    /// blob, to the id the Hub lists for that file at the revision the weights were proven against —
    /// and what the engine reads of them is there (the DiT's config, the encoder's `text_config`,
    /// the twelve layers drawn).
    func testKrea2PublishedFilesAreKreasToTheByte() throws {
        for (path, blob, text) in Krea2Published.blobs {
            let bytes = Data(text.utf8)
            var hasher = Insecure.SHA1()
            hasher.update(data: Data("blob \(bytes.count)\u{0}".utf8))
            hasher.update(data: bytes)
            XCTAssertEqual(hasher.finalize().map { String(format: "%02x", $0) }.joined(), blob, path)
        }
        XCTAssertEqual(Krea2Published.transformerConfig["num_layers"]?.integer, 28)
        XCTAssertEqual(Krea2Published.encoderConfig["text_config"]?["num_hidden_layers"]?.integer, 36)
        XCTAssertEqual(Krea2Text.sockets, Array(stride(from: 2, through: 35, by: 3)))
        let config = Krea2DiT.Config(header: ["config": try JSONSerialization.jsonObject(with: Data(Krea2Published.transformerConfigText.utf8))])
        XCTAssertEqual([config.dim, config.heads, config.kvHeads, config.hidden, config.layers, config.channels], [6144, 48, 12, 16384, 28, 64])
    }

    /// **Anima's Compact**: Bedovyy's int8 convrot of Turbo v1.1, pinned (revision, sha256), the very file judged (its fixture):
    /// 254 int8 convrot linears of group 256, the rest bf16 as published — every adaLN, blocks 0 and 27's attention, five MLPs, the
    /// text adapter. Its license links the publisher's files, then Bedovyy's card; it reads the Standard's encoder and components.
    /// **Its size is the file's**, the adapter staying bf16 (no more fp32 widening, as the Q8_0 Compact had): within 2 %.
    func testAnimaCompactIsTheJudgedConvrot() throws {
        let hex = try NSRegularExpression(pattern: "^[0-9a-f]+$")
        func isHex(_ s: String) -> Bool { hex.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil }
        let dit = Installer.dit(.anima, .compact)
        XCTAssertEqual(dit.count, 1)
        let x = dit[0]
        XCTAssertEqual(x.repository.name, "Bedovyy/Anima-INT8")
        XCTAssertEqual(x.path, "anima-turbo-v1.1-int8convrot.safetensors")
        XCTAssertTrue(x.repository.revision.count == 40 && isHex(x.repository.revision))
        XCTAssertEqual(x.sha256, "c462bea054de87ea03000e83f2b60e6d58fbcb6b925d9f607f8a470eeaad52b4")
        let d = try fixture(x)
        XCTAssertEqual(d.manifest["family"] as? String, "anima")
        XCTAssertEqual(d.manifest["format"] as? String, "convrot")
        XCTAssertEqual(d.manifest["group"] as? Int, 256)
        XCTAssertEqual(d.manifest["file_bytes"] as? Int, x.size)
        let published = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: d.folder + "/header.json"))) as? [String: [String: Any]])
        let header = published.filter { $0.key != "__metadata__" }
        XCTAssertEqual(header.count, 1193)
        let dtypes = header.values.compactMap { $0["dtype"] as? String }
        XCTAssertEqual(dtypes.filter { $0 == "I8" }.count, 254)
        XCTAssertEqual(Set(dtypes), ["I8", "F32", "U8", "BF16"], "nothing under 8 bits")
        XCTAssertTrue(header.filter { $0.key.contains(".llm_adapter.") }.allSatisfy { $0.value["dtype"] as? String == "BF16" }, "the adapter in bf16")

        XCTAssertEqual(Installer.encoder(.anima, .compact).map(\.path), Installer.encoder(.anima).map(\.path))
        XCTAssertEqual(Family.anima.encoderMap(.compact), Family.anima.encoderMap)
        let licenses = Installer.licenseFiles(.anima, .compact), publisher = Installer.licenseFiles(.anima)
        XCTAssertEqual(licenses.prefix(publisher.count).map(\.repository.name), publisher.map(\.repository.name))
        XCTAssertEqual(licenses.dropFirst(publisher.count).map(\.repository.name), [x.repository.name])
        XCTAssertEqual(licenses.dropFirst(publisher.count).map(\.path), ["README.md"])

        let size = Family.anima.installedSize(.compact), standard = Family.anima.installedSize(.standard)
        let bytes = try XCTUnwrap(d.manifest["file_bytes"] as? Int)
        XCTAssertEqual(Double(size.dit), Double(bytes), accuracy: 0.02 * Double(bytes))
        XCTAssertEqual(size.encoder, standard.encoder)
        XCTAssertEqual(size.components, standard.components)
        XCTAssertLessThan(Family.anima.installedSize(.light).dit, size.dit)
        XCTAssertLessThan(size.dit, standard.dit)
    }

    /// **Anima's Light is our own 4-bit file** (SQ): one safetensors in our repository, pinned (revision, sha256, size); it reads
    /// the Standard's encoder and components; its license links the publisher's files, then our card, NOTICE and `LICENSE.md`.
    /// Its size is the map's plus the publisher's text adapter beside it.
    func testAnimaLightIsOurSQ4() throws {
        let hex = try NSRegularExpression(pattern: "^[0-9a-f]+$")
        func isHex(_ s: String) -> Bool { hex.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil }
        let standard = Family.anima.installedSize(.standard)
        let dit = Installer.dit(.anima, .light)
        XCTAssertEqual(dit.count, 1)
        let x = dit[0]
        XCTAssertEqual(x.path, "anima-turbo-v1.1-sq4.safetensors")
        XCTAssertEqual(x.repository.name, "gwenn-ha-dev/Anima-Turbo-Siliconed-SQ4")
        XCTAssertTrue(x.repository.revision.count == 40 && isHex(x.repository.revision))
        let sha = try XCTUnwrap(x.sha256)
        XCTAssertTrue(sha.count == 64 && isHex(sha))
        XCTAssertEqual(x.size, 1_678_437_032)

        XCTAssertEqual(Installer.encoder(.anima, .light).map(\.path), Installer.encoder(.anima).map(\.path))
        XCTAssertEqual(Installer.encoder(.anima, .light).map(\.repository.name), Installer.encoder(.anima).map(\.repository.name))
        XCTAssertEqual(Family.anima.encoderMap(.light), Family.anima.encoderMap)
        let licenses = Installer.licenseFiles(.anima, .light), publisher = Installer.licenseFiles(.anima)
        XCTAssertEqual(licenses.prefix(publisher.count).map(\.repository.name), publisher.map(\.repository.name))
        XCTAssertEqual(licenses.dropFirst(publisher.count).map(\.repository.name), Array(repeating: x.repository.name, count: 3))
        XCTAssertEqual(licenses.dropFirst(publisher.count).map(\.path), ["README.md", "NOTICE", "LICENSE.md"])

        let size = Family.anima.installedSize(.light)
        XCTAssertGreaterThan(size.dit, x.size!, "the map and the adapter written beside it")
        XCTAssertLessThan(size.dit, Family.anima.installedSize(.compact).dit)
        XCTAssertEqual(size.encoder, standard.encoder)
        XCTAssertEqual(size.components, standard.components)
    }

    /// **The Light**: one DiT file, ours (the SQ 4-bit, `SQ.swift`), pinned (repository, revision, sha256, size) in a repository of
    /// ours, none of the publisher's; its encoder is the Compact's, file for file; its license links the publisher's files, then our
    /// card, NOTICE and license file, and the encoder's third-party card; its DiT the smallest of the three.
    func testTheLightIsOurSQ4() throws {
        let hex = try NSRegularExpression(pattern: "^[0-9a-f]+$")
        func isHex(_ s: String) -> Bool { hex.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil }
        let expected: [Family: (name: String, file: String, license: String)] = [
            .zImage: ("gwenn-ha-dev/Z-Image-Turbo-Siliconed-SQ4", "z-image-turbo-sq4.safetensors", "LICENSE"),
            .qwenImage21: ("gwenn-ha-dev/Qwen-Image-2.1-Turbo-Siliconed-SQ4", "qwen-image-2.1-turbo-sq4.safetensors", "LICENSE")]
        for f in Self.compactFamilies {
            let dit = Installer.dit(f, .light)
            XCTAssertEqual(dit.count, 1)
            let x = dit[0], e = try XCTUnwrap(expected[f])
            XCTAssertEqual(x.repository.name, e.name)
            XCTAssertEqual(x.path, e.file)
            XCTAssertFalse(Set(Installer.dit(f).map(\.repository.name)).contains(x.repository.name))
            XCTAssertTrue(x.repository.revision.count == 40 && isHex(x.repository.revision))
            let sha = try XCTUnwrap(x.sha256)
            XCTAssertTrue(sha.count == 64 && isHex(sha))
            XCTAssertGreaterThan(try XCTUnwrap(x.size), 1_000_000_000)

            XCTAssertEqual(Installer.encoder(f, .light).map(\.path), Installer.encoder(f, .compact).map(\.path))
            XCTAssertEqual(Installer.encoder(f, .light).map(\.sha256), Installer.encoder(f, .compact).map(\.sha256))
            XCTAssertEqual(f.encoderMap(.light), f.encoderMap(.compact))

            let standard = Installer.licenseFiles(f), light = Installer.licenseFiles(f, .light)
            XCTAssertEqual(light.prefix(standard.count).map(\.repository.name), standard.map(\.repository.name))
            let own = light.dropFirst(standard.count)
            XCTAssertEqual(own.prefix(3).map(\.repository.name), Array(repeating: x.repository.name, count: 3))
            XCTAssertEqual(own.prefix(3).map(\.path), ["README.md", "NOTICE", e.license])
            XCTAssertEqual(own.dropFirst(3).map(\.repository.name), [Installer.encoder(f, .light)[1].repository.name])
            XCTAssertEqual(own.dropFirst(3).map(\.path), ["README.md"])

            let size = f.installedSize(.light), compact = f.installedSize(.compact)
            XCTAssertEqual(Double(size.dit), Double(x.size!), accuracy: 0.02 * Double(x.size!), "\(f.name) DiT")
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

    /// **Qwen's three versions are the Turbo's**: no file of any recipe — components, encoder, DiT,
    /// license — comes from the base model's repository; the Compact's and the Light's DiTs are the
    /// third parties' Turbo files (V5a), their encoder's `config.json` the Turbo's.
    func testNoQwenRecipeReadsTheBaseModel() throws {
        for v in Family.qwenImage21.variants {
            let files = Installer.components(.qwenImage21) + Installer.encoder(.qwenImage21, v) + Installer.dit(.qwenImage21, v)
                + Installer.licenseFiles(.qwenImage21, v)
            XCTAssertFalse(files.contains { $0.repository.name == "Qwen/Qwen-Image-2.1" }, v.rawValue)
            XCTAssertTrue(Family.qwenImage21.ditMap(v).hasPrefix("qwen-image-2.1-turbo-dit"), v.rawValue)
            XCTAssertFalse(Family.supersededQwenImage21Maps.contains(Family.qwenImage21.ditMap(v)), v.rawValue)
        }
        XCTAssertEqual(Installer.dit(.qwenImage21, .compact).map(\.path), ["diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"])
        XCTAssertEqual(Installer.dit(.qwenImage21, .compact).compactMap(\.sha256), ["8144a0182964aa8403ad6dbc011db837304167d235161b9aabaa8532e967314c"])
        XCTAssertEqual(Installer.dit(.qwenImage21, .light).map(\.path), ["qwen-image-2.1-turbo-sq4.safetensors"])
        XCTAssertEqual(Installer.dit(.qwenImage21, .light).compactMap(\.sha256), ["d289920998192edc414b447f26e83230de1afd820bcad5bce559b7e4ec666e57"])
        XCTAssertEqual(Installer.encoder(.qwenImage21, .light).compactMap(\.sha256),
                       ["0cb5ed0653e2ec8065f25a6cfb2ab6a89b6c8f143a500acea19f0e36214814b7"], "the 8-bit encoder is unchanged")
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
            XCTAssertEqual(size.components, standard.components, "tokenizers, VAE: the Standard's")
            let total = { (s: (dit: Int, encoder: Int, components: Int)) in s.dit + s.encoder + s.components }
            XCTAssertLessThan(Double(total(size)), 0.6 * Double(total(standard)), "\(f.name): Compact is the smaller one")
        }
    }

    /// **The headers are what the forge reads**: the int8 convrot DiTs of Qwen (Comfy-Org's), Z-Image (Comfy-Org's) and Anima
    /// (Bedovyy's), unsloth's Qwen encoder and Disty0's Z-Image encoder, whole published headers over sparse files — recognized,
    /// normalized, every 8-bit weight kept 8-bit with its rotation, nothing widened.
    func testCompactHeadersAreForgedEightBit() throws {
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-compact-\(UUID())")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        // The DiT of Qwen-Image-2.1 Compact.
        let qwenDiT = try fixture(Installer.dit(.qwenImage21, .compact)[0])
        let ditSource = try TensorSource(paths: [try sparse(qwenDiT.folder, scratch)])
        XCTAssertEqual(Recipes.family(fromNames: ditSource.names), .qwenImage21)
        // The Turbo's: its 31 `comfy_attention.config` (ComfyUI's attention preference) are ignored,
        // never a name the strict recipe sees.
        XCTAssertTrue(ditSource.notes.contains { $0.hasPrefix("31 ComfyUI attention preference(s)") }, "\(ditSource.notes)")
        XCTAssertFalse(ditSource.names.contains { $0.contains("comfy_attention") })
        let n = try Recipes.normalize(ditSource, family: .qwenImage21)
        XCTAssertEqual(n.dit.count, 297, "297/297 names")
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

        // Z-Image's and Anima's: every name the Standard's map has (`dit.json`'s count), the fused qkv of Z-Image split with its
        // rows' scales, Anima's q and k rows permuted per head with theirs — 8-bit and rotated all the same.
        for (f, names, eightBit) in [(Family.zImage, 521, 270), (.anima, 567, 254)] {
            let fixture = try fixture(Installer.dit(f, .compact)[0])
            let source = try TensorSource(paths: [try sparse(fixture.folder, scratch)])
            XCTAssertEqual(Recipes.family(fromNames: source.names), f)
            let n = try Recipes.normalize(source, family: f)
            XCTAssertEqual(n.dit.count, names, f.name)
            var kept = 0
            for (name, p) in n.dit {
                XCTAssertNil(p.dequantizedBecause, name)
                guard p.quantization != nil else { continue }
                let (t, outcome) = try ForgeDiT.tensor(name, p, family: f)
                XCTAssertEqual(outcome, .kept8bit, "\(f.name) \(name)")
                XCTAssertEqual(t.rotation, 256, "\(f.name) \(name): convrot")
                kept += 1
            }
            XCTAssertEqual(kept, eightBit, f.name)
            if f == .anima { XCTAssertEqual(n.adapter.count, 118, "the text adapter, beside the map") }
        }

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
    /// texts and other U8 texts (`comfy_other_u8`) written in (`QuantCheck.swift`, `sparseSource`).
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
        if let quant = manifest["comfy_quant"] as? [String: String] {
            let texts = quant.merging(manifest["comfy_other_u8"] as? [String: String] ?? [:]) { a, _ in a }
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
