import Foundation
import XCTest
@testable import Siliconed

/// **A file the user already has, taken instead of downloaded** (`Adoption`) — on small files in a
/// temporary folder and a table of the test's own: no weight, no network.
final class AdoptionTests: XCTestCase {
    private var root: URL!
    private var library: Library!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-adoption-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        library = Library(root: root.appendingPathComponent("lib"))
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private func write(_ name: String, _ bytes: String) throws -> String {
        let path = root.appendingPathComponent(name).path
        try Data(bytes.utf8).write(to: URL(fileURLWithPath: path))
        return path
    }

    private func file(_ repository: String, _ bytes: String) throws -> Installer.File {
        let sample = try write("sample-\(UUID())", bytes)
        return Installer.File(repository: .init(name: repository, revision: "r"), path: "encoder.safetensors", destination: nil,
                              sha256: try Downloader.sha256(sample), size: bytes.utf8.count)
    }

    func testPublishedTableIsTheInstallationsPinnedEncodersAndVAEs() {
        let table = Adoption.published()
        XCTAssertEqual(Set(table.map(\.family)), Set(Family.allCases))
        for x in table {
            XCTAssertEqual(x.file.sha256?.count, 64)
            XCTAssertGreaterThan(x.file.size ?? 0, 50_000_000, "weights only, never a config")
        }
        // Every family's VAE, once (a component is every version's).
        let vaes = table.filter { $0.file.destination != nil }
        XCTAssertEqual(vaes.map(\.family).sorted { $0.rawValue < $1.rawValue }, Family.allCases.sorted { $0.rawValue < $1.rawValue })
        XCTAssertTrue(vaes.allSatisfy { $0.file.destination == "vae.safetensors" && $0.variant == .standard })
        // Every version of every family has its encoder's weights pinned (a sharded one, each shard).
        for f in Family.allCases {
            for v in f.variants {
                XCTAssertTrue(table.contains { $0.family == f && $0.variant == v && $0.file.destination == nil }, "\(f) \(v)")
            }
        }
        // A Compact's and a Light's encoder is one file, so one place on disk.
        let installer = Installer(library: library)
        let z = table.filter { $0.family == .zImage && $0.variant != .standard }
        XCTAssertEqual(Set(z.map(\.variant)), [.compact, .light])
        XCTAssertEqual(Set(z.map { installer.source($0.file) }).count, 1)
    }

    func testOtherSizeIsNotHashed() throws {
        let expected = try file("a/enc", "the expected bytes")
        let table = [Adoption.Expectation(family: .zImage, variant: .compact, file: expected)]
        let other = try write("other", "a different length altogether")
        let hashed = try Adoption.adopt(other, in: library, among: table, hash: { _ in XCTFail("hashed"); return "" })
        XCTAssertNil(hashed)
    }

    func testSameSizeOtherBytesIsRefusedAndNothingCopied() throws {
        let expected = try file("a/enc", "the expected bytes")
        let table = [Adoption.Expectation(family: .zImage, variant: .compact, file: expected)]
        let lookalike = try write("lookalike", "the expected bytez")
        XCTAssertNil(try Adoption.adopt(lookalike, in: library, among: table))
        XCTAssertFalse(FileManager.default.fileExists(atPath: Installer(library: library).downloads))
    }

    func testIdenticalFileIsCopiedWhereTheDownloadWouldBe() throws {
        let expected = try file("a/enc", "the expected bytes")
        let table = [Adoption.Expectation(family: .zImage, variant: .compact, file: expected),
                     Adoption.Expectation(family: .zImage, variant: .light, file: expected)]
        let mine = try write("comfy-copy.safetensors", "the expected bytes")
        let r = try XCTUnwrap(Adoption.adopt(mine, in: library, among: table))
        let place = Installer(library: library).source(expected)
        XCTAssertEqual(r.expected.count, 2)
        XCTAssertEqual(r.placed, [place], "a Compact and a Light share one copy")
        XCTAssertEqual(r.copied, 1)
        XCTAssertEqual(try String(contentsOfFile: place, encoding: .utf8), "the expected bytes")
        XCTAssertFalse(FileManager.default.fileExists(atPath: place + ".partiel"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: mine), "the user's file is left alone")
        // The installation's own verification accepts it.
        XCTAssertEqual(try Downloader.sha256(place), expected.sha256)

        // Again: already in place, nothing copied.
        let again = try XCTUnwrap(Adoption.adopt(mine, in: library, among: table))
        XCTAssertEqual(again.placed, [place])
        XCTAssertEqual(again.copied, 0)
    }

    func testALinkIsFollowedAndItsFileCopied() throws {
        let expected = try file("a/enc", "the expected bytes")
        let table = [Adoption.Expectation(family: .zImage, variant: .compact, file: expected)]
        let mine = try write("comfy-copy.safetensors", "the expected bytes")
        let link = root.appendingPathComponent("linked.safetensors").path
        try FileManager.default.createSymbolicLink(atPath: link, withDestinationPath: mine)
        let r = try XCTUnwrap(Adoption.adopt(link, in: library, among: table), "a link weighs its file")
        let place = Installer(library: library).source(expected)
        XCTAssertEqual(r.copied, 1)
        let type = try FileManager.default.attributesOfItem(atPath: place)[.type] as? FileAttributeType
        XCTAssertEqual(type, .typeRegular, "the file is copied, never the link")
        XCTAssertEqual(try Downloader.sha256(place), expected.sha256)
    }

    func testFileExpectedByTwoFamiliesIsPlacedForEach() throws {
        let one = try file("repo/one", "shared bytes")
        let two = Installer.File(repository: .init(name: "repo/two", revision: "r"), path: one.path, destination: nil,
                                 sha256: one.sha256, size: one.size)
        let table = [Adoption.Expectation(family: .zImage, variant: .compact, file: one),
                     Adoption.Expectation(family: .krea2, variant: .compact, file: two)]
        let r = try XCTUnwrap(Adoption.adopt(try write("mine", "shared bytes"), in: library, among: table))
        XCTAssertEqual(Set(r.expected.map(\.family)), [.zImage, .krea2])
        XCTAssertEqual(r.placed.count, 2)
        XCTAssertEqual(r.copied, 2)
    }

    func testNothingIsCopiedWhenEveryEncoderIsAlreadyInstalled() throws {
        let expected = try file("a/enc", "the expected bytes")
        let table = [Adoption.Expectation(family: .zImage, variant: .compact, file: expected)]
        let map = library.map(Family.zImage.encoderMap(.compact))
        try FileManager.default.createDirectory(atPath: (map as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        try Data("map".utf8).write(to: URL(fileURLWithPath: map))
        let r = try XCTUnwrap(Adoption.adopt(try write("mine", "the expected bytes"), in: library, among: table))
        XCTAssertEqual(r.expected.count, 1)
        XCTAssertTrue(r.placed.isEmpty)
        XCTAssertEqual(r.copied, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: Installer(library: library).source(expected)))
    }

    func testVAEWaitsAmongDownloadsAndTheComponentsFolderSaysInstalled() throws {
        let vae = try file("a/vae", "vae bytes")
        let component = Installer.File(repository: vae.repository, path: "vae/diffusion_pytorch_model.safetensors",
                                       destination: "vae.safetensors", sha256: vae.sha256, size: vae.size)
        let table = [Adoption.Expectation(family: .anima, variant: .standard, file: component)]
        let installer = Installer(library: library)
        let r = try XCTUnwrap(Adoption.adopt(try write("ae.safetensors", "vae bytes"), in: library, among: table))
        XCTAssertEqual(r.placed, [installer.source(component)])
        // Never under store/composants/: that folder alone makes a family look installed.
        XCTAssertFalse(FileManager.default.fileExists(atPath: installer.componentsFolder(.anima)))

        // The family's VAE installed: nothing is needed.
        try FileManager.default.createDirectory(atPath: installer.componentsFolder(.krea2), withIntermediateDirectories: true)
        try Data("vae".utf8).write(to: URL(fileURLWithPath: installer.componentsFolder(.krea2) + "/vae.safetensors"))
        let installed = [Adoption.Expectation(family: .krea2, variant: .standard, file: component)]
        let again = try XCTUnwrap(Adoption.adopt(try write("ae2.safetensors", "vae bytes"), in: library, among: installed))
        XCTAssertTrue(again.placed.isEmpty)
    }
}
