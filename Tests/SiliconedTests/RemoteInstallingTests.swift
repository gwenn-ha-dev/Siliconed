import Foundation
import XCTest
@testable import Siliconed

/// **The lender's library as the borrower reads it**: the messages as JSON, the occupancy rebuilt from
/// what was published, the engine's journal read into phases.
final class RemoteInstallingTests: XCTestCase {
    private func roundTrip(_ m: SessionMessage, file: StaticString = #filePath, line: UInt = #line) throws {
        XCTAssertEqual(try JSONDecoder().decode(SessionMessage.self, from: JSONEncoder().encode(m)), m, file: file, line: line)
    }

    private var library: RemoteLibrary {
        let plan = RemotePlan(variant: .compact, baseDiT: true, replacing: .standard, added: 9_000_000_000, freed: 12_000_000_000,
                              peak: 20_000_000_000, removesFirst: false, fits: true)
        let part = RemoteLibrary.Part(family: .zImage, components: 1, encoder: 2, dit: 3, variant: .compact, isReady: true, accepted: true,
                                      installed: .compact, preselected: .light, plans: [plan])
        return RemoteLibrary(parts: [part, RemoteLibrary.Part(family: .anima, components: 0, encoder: 0, dit: 0, variant: nil, isReady: false,
                                                              accepted: false, installed: nil, preselected: .compact, plans: [])],
                             imported: [.init(id: "z-image/my-checkpoint", name: "My checkpoint", family: .zImage, bytes: 6)],
                             loras: [.init(path: "/Library/Siliconed/store/lora/niji.lora.silicon", rawName: "Niji_v5", target: "z-image", rank: 32,
                                           smallestRank: 8, bytes: 7)],
                             downloads: 5, cache: 4, free: 100)
    }

    func testTheLibraryAndItsGesturesTravelAsJSON() throws {
        let digest = Data(repeating: 9, count: 32)
        let file = RemoteHubFile(repository: "someone/niji-lora", path: "niji.safetensors")
        let messages: [SessionMessage] = [
            .libraryNews(.library(library)),
            .libraryNews(.forge(RemoteForge(task: .installing(.krea2), phase: .downloading(file: "dit.safetensors", percent: 40, bytes: 4, total: 10, preparing: "encoder")))),
            .libraryNews(.forge(RemoteForge(task: .importing(files: ["a.safetensors", "b.gguf"]), phase: .preparing(name: "x", converted: nil)))),
            .libraryNews(.forge(RemoteForge(task: .other, phase: .starting))), .libraryNews(.forge(nil)),
            .libraryNews(.hubVerdict(file, .lora(.zImage))), .libraryNews(.hubVerdict(file, .loraNeedsFamily(.anima))),
            .libraryNews(.hubVerdict(file, .unexamined)), .libraryNews(.hubVerdict(file, .model(.qwenImage21))),
            .libraryNews(.hubVerdict(file, .refused(.text("not a LoRA")))),
            .libraryNews(.hubVerdict(file, .refused(.app(.hubFileUnknown("niji.safetensors"))))),
            .libraryAsk(.install(RemoteInstall(family: .qwenImage21, variant: .light, baseDiT: false, digest: digest))),
            .libraryAsk(.cancel), .libraryAsk(.uninstall(.mossImage)), .libraryAsk(.removeModel("z-image/my-checkpoint")),
            .libraryAsk(.removeLoRA("/Library/Siliconed/store/lora/niji.lora.silicon")), .libraryAsk(.discard),
            .libraryAsk(.hubJudge(file)), .libraryAsk(.hubInstall(file)),
            .refused(.app(.libraryBusy)), .refused(.app(.libraryUnreachable)), .refused(.app(.familyUnknown("z-image/light"))),
            .refused(.app(.modelUnknown("x/y"))), .refused(.app(.hubNeedsFamily(.krea2))), .refused(.app(.hubFileUnknown("p.gguf"))),
        ]
        for m in messages { try roundTrip(m) }
    }

    /// What the Models sheet reads is rebuilt from what crossed, with nothing of the borrower's disk.
    func testTheOccupancyIsRebuiltFromThePublishedLibrary() throws {
        let l = try JSONDecoder().decode(RemoteLibrary.self, from: JSONEncoder().encode(library))
        let o = l.occupancy()
        XCTAssertEqual(o.families.map(\.family), [.zImage, .anima])
        XCTAssertEqual(o.families[0].total, 6)
        XCTAssertEqual(o.families[0].variant, .compact)
        XCTAssertTrue(o.families[0].isReady)
        XCTAssertFalse(o.families[1].present)
        let card = try XCTUnwrap(o.imported.first?.card)
        XCTAssertEqual(card.id, "z-image/my-checkpoint")
        XCTAssertEqual(card.name, "My checkpoint")
        XCTAssertTrue(card.isImported)
        XCTAssertEqual(o.imported.first?.bytes, 6)
        let lora = try XCTUnwrap(o.loras.first?.card)
        XCTAssertEqual(lora.path, "/Library/Siliconed/store/lora/niji.lora.silicon")
        XCTAssertEqual(lora.name, "Niji v5")
        XCTAssertEqual(lora.target, "z-image")
        XCTAssertEqual(lora.rank, 32)
        XCTAssertEqual(lora.smallestRank, 8)
        XCTAssertEqual(o.downloads, 5)
        XCTAssertEqual(o.free, 100)
        XCTAssertEqual(l.part(.zImage)?.plan(.compact, baseDiT: true)?.freed, 12_000_000_000)
        XCTAssertNil(l.part(.zImage)?.plan(.light, baseDiT: true), "a plan the lender did not send is none")
    }

    func testAUniformRankIsOneRank() throws {
        var l = library
        l.loras[0].smallestRank = 32
        let lora = try XCTUnwrap(l.occupancy().loras.first?.card)
        XCTAssertEqual(lora.rank, 32)
        XCTAssertEqual(lora.smallestRank, 32)
    }

    // MARK: - The journal, read once on the lender

    func testTheJournalReadsIntoPhases() {
        typealias P = RemoteForge.Phase
        XCTAssertEqual(P(journal: []), .starting)
        XCTAssertEqual(P(journal: ["── Z-Image Turbo (compact)"]), .starting)
        let download = ["↓ dit/model.safetensors (4.00 GB)"]
        XCTAssertEqual(P(journal: download), .downloading(file: "model.safetensors", percent: nil, bytes: nil, total: 4_000_000_000, preparing: nil))
        XCTAssertEqual(P(journal: download + ["  … 40 %"]),
                       .downloading(file: "model.safetensors", percent: 40, bytes: 1_600_000_000, total: 4_000_000_000, preparing: nil))
        // The DiT downloads while the encoder is forged: both are said.
        XCTAssertEqual(P(journal: ["forging the encoder → encoder.silicon"] + download + ["  … 20 %"]),
                       .downloading(file: "model.safetensors", percent: 20, bytes: 800_000_000, total: 4_000_000_000, preparing: "encoder.silicon"))
        XCTAssertEqual(P(journal: download + ["  ✓ 30 s, 100 MB/s, sha256 verified", "forging the DiT → dit.silicon"]),
                       .preparing(name: "dit.silicon", converted: nil))
        XCTAssertEqual(P(journal: download + ["  ✓ 30 s", "forging the DiT → dit.silicon", "  … 60 %"]),
                       .preparing(name: "dit.silicon", converted: 60))
        XCTAssertEqual(P(journal: ["forging the DiT → d.silicon", "  ✓ forged d.silicon in 9 s", "✓ Z-Image Turbo installed"]), .installed)
        // A library being moved is no phase of an installation.
        XCTAssertEqual(P(journal: ["⇢ 40 file"]), .starting)
    }
}
