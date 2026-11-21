import Foundation
import XCTest
@testable import Siliconed

/// **The lender's library as the borrower reads it**: the messages as JSON, the occupancy rebuilt from what was
/// published, the plans read from the lender's disk, an installation's progress as a phase.
final class RemoteInstallingTests: XCTestCase {
    private func roundTrip(_ m: SessionMessage, file: StaticString = #filePath, line: UInt = #line) throws {
        XCTAssertEqual(try WireJSON.decode(SessionMessage.self, from: WireJSON.encode(m)), m, file: file, line: line)
    }

    private var library: RemoteLibrary {
        let plan = RemotePlan(variant: .standard, needed: 20_000_000_000, fits: false, others: [.compact], releasable: 10_000_000_000,
                              fitsReplacing: true)
        let part = RemoteLibrary.Part(family: .zImage, installed: [.compact], variant: .compact, bytes: 10_700_000_000, accepted: true,
                                      plans: [plan, RemotePlan(variant: .compact, needed: 0, fits: true, others: [], releasable: 0, fitsReplacing: false)])
        return RemoteLibrary(parts: [part, RemoteLibrary.Part(family: .anima, installed: [], variant: nil, bytes: 0, accepted: false, plans: [])],
                             imported: [.init(id: "z-image/my_checkpoint", name: "my checkpoint", family: .zImage, bytes: 6)],
                             loras: [.init(lora, bytes: 7)],
                             downloads: 5, free: 100)
    }

    private let lora = LoRACard(path: "/Lender/Siliconed/loras/z-image/Niji_style.safetensors", rawName: "Niji_style",
                                target: "z-image", rank: 32, smallestRank: 16, resolution: "1024", trainedOn: "z-image-turbo",
                                trainerName: nil, triggers: ["niji"])

    func testTheLibraryAndItsGesturesTravelAsJSON() throws {
        let digest = Data(repeating: 9, count: 32)
        let messages: [SessionMessage] = [
            .libraryNews(.library(library)),
            .libraryNews(.job(RemoteLibraryJob(task: .installing(.krea2), progress: Installer.Progress(file: "dit.safetensors", index: 2, count: 5, received: 4, total: 10)))),
            .libraryNews(.job(RemoteLibraryJob(task: .other, progress: nil))),
            .libraryNews(.job(RemoteLibraryJob(task: .importing, progress: nil))),
            .libraryNews(.job(nil)),
            .libraryNews(.confirming(true)),
            .libraryAsk(.install(RemoteInstall(family: .anima, variant: .compact, replacing: false, digest: digest))),
            .libraryAsk(.install(RemoteInstall(family: .zImage, variant: .standard, replacing: true, digest: digest))),
            .libraryAsk(.cancel), .libraryAsk(.use(.zImage, .standard)),
            .libraryAsk(.uninstall(.krea2, nil)), .libraryAsk(.uninstall(.krea2, .compact)), .libraryAsk(.discard),
            .libraryAsk(.removeModel("z-image/my_checkpoint")), .libraryAsk(.removeLoRA(lora.path)),
            .refused(.app(.libraryBusy)), .refused(.app(.familyUnknown("anima/compact"))), .refused(.app(.removalDeclined)),
        ]
        for m in messages { try roundTrip(m) }
    }

    /// The lender's user is asked before anything is deleted there: a removal, an installation that takes another
    /// version's place, downloads that are there. The rest goes without a click.
    func testOnlyWhatDeletesWaitsForAClick() {
        let digest = Data(repeating: 9, count: 32)
        XCTAssertTrue(RemoteLibraryRequest.uninstall(.anima, nil).deletes(downloadsLeft: 0))
        XCTAssertTrue(RemoteLibraryRequest.uninstall(.anima, .compact).deletes(downloadsLeft: 0))
        XCTAssertTrue(RemoteLibraryRequest.install(RemoteInstall(family: .anima, variant: .standard, replacing: true, digest: digest)).deletes(downloadsLeft: 0))
        XCTAssertFalse(RemoteLibraryRequest.install(RemoteInstall(family: .anima, variant: .standard, replacing: false, digest: digest)).deletes(downloadsLeft: 9))
        XCTAssertTrue(RemoteLibraryRequest.discard.deletes(downloadsLeft: 1))
        XCTAssertFalse(RemoteLibraryRequest.discard.deletes(downloadsLeft: 0))
        XCTAssertFalse(RemoteLibraryRequest.cancel.deletes(downloadsLeft: 9))
        XCTAssertFalse(RemoteLibraryRequest.use(.anima, .compact).deletes(downloadsLeft: 9))
        XCTAssertTrue(RemoteLibraryRequest.removeModel("z-image/my_checkpoint").deletes(downloadsLeft: 0))
        XCTAssertTrue(RemoteLibraryRequest.removeLoRA(lora.path).deletes(downloadsLeft: 0))
    }

    /// A removal names only what the lender listed itself: an imported model, never a publisher's; a LoRA by the path
    /// it published, never one of the borrower's making — outside its LoRA folders, or led out of them.
    func testARemovalNamesOnlyWhatTheLenderHolds() {
        let publisher = ModelCard.of(.zImage)
        let imported = ModelCard(id: "z-image/my_checkpoint", name: "my checkpoint", family: .zImage, map: "maps/my_checkpoint",
                                 license: publisher.license, defaultSteps: publisher.defaultSteps, space: publisher.space,
                                 formats: publisher.formats)
        XCTAssertEqual(RemoteLibraryRequest.importedModel(imported.id, among: [publisher, imported]), imported)
        XCTAssertNil(RemoteLibraryRequest.importedModel(publisher.id, among: [publisher, imported]))
        XCTAssertNil(RemoteLibraryRequest.importedModel("z-image/gone", among: [publisher, imported]))

        XCTAssertEqual(RemoteLibraryRequest.lora(lora.path, among: [lora]), lora)
        for path in ["/Volumes/Other/notes.safetensors", "/Lender/Siliconed/loras/z-image/../../store/dit.safetensors",
                     "/Lender/Siliconed/loras/z-image/./Niji_style.safetensors", "loras/z-image/Niji_style.safetensors", ""] {
            XCTAssertNil(RemoteLibraryRequest.lora(path, among: [lora]), path)
        }
    }

    /// The Models sheet of a mirror reads an occupancy: the lender's, part for part.
    func testTheOccupancyIsRebuiltFromTheLendersWord() {
        let o = library.occupancy()
        XCTAssertEqual(o.families.map(\.family), [.zImage, .anima])
        XCTAssertEqual(o.families[0], .init(family: .zImage, installed: [.compact], variant: .compact, bytes: 10_700_000_000))
        XCTAssertTrue(o.families[0].isReady)
        XCTAssertFalse(o.families[1].isReady)
        XCTAssertFalse(o.families[1].present)
        XCTAssertEqual([o.downloads, o.total], [5, 10_700_000_011])
        // A model imported on the lender is a model of the borrower's lists too, under its family's license and steps.
        let card = o.imported.first?.card
        XCTAssertEqual(card?.id, "z-image/my_checkpoint")
        XCTAssertEqual(card?.name, "my checkpoint")
        XCTAssertEqual(card?.isImported, true)
        XCTAssertEqual(card?.defaultSteps, ModelCard.of(.zImage).defaultSteps)
        XCTAssertEqual(o.imported.first?.bytes, 6)
        XCTAssertEqual(o.free, 100)
        XCTAssertEqual(library.part(.zImage)?.plan(.standard)?.fitsReplacing, true)
        XCTAssertNil(library.part(.anima)?.plan(.standard))
        // A LoRA of the lender's is the line its own sheet shows: name, target, ranks, what its file says of itself.
        XCTAssertEqual(library.loras.map(\.card), [lora])
        XCTAssertEqual(library.loras.first?.bytes, 7)
    }

    /// The lender's snapshot is read from its disk: what is whole, what each version would still download.
    func testTheSnapshotReadsTheLendersDisk() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-remote-library-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let lender = Library(root: root)
        for file in Family.anima.files(.compact) {
            let path = lender.map(file.place)
            try FileManager.default.createDirectory(atPath: (path as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
            let fd = open(path, O_RDWR | O_CREAT, 0o644)
            XCTAssertTrue(fd >= 0 && ftruncate(fd, off_t(file.bytes)) == 0)
            close(fd)
        }
        try lender.acceptLicense(ModelCard.of(.anima))
        let published = RemoteLibrary(lender.occupancy(), loras: [(lora, 7)], in: lender)
        let anima = try XCTUnwrap(published.part(.anima))
        XCTAssertEqual(anima.installed, [.compact])
        XCTAssertEqual(anima.variant, .compact)
        XCTAssertEqual(anima.bytes, Family.anima.installedSize(.compact))
        XCTAssertTrue(anima.accepted)
        XCTAssertEqual(anima.plan(.compact)?.needed, 0)
        // The Standard would download what the Compact does not already hold: its own DiT, not the shared small files.
        let shared = Set(Family.anima.files(.compact).map(\.place))
        let own = Family.anima.files(.standard).filter { !shared.contains($0.place) }.reduce(0) { $0 + $1.bytes }
        XCTAssertEqual(anima.plan(.standard)?.needed, own)
        XCTAssertEqual(anima.plan(.standard)?.others, [.compact])
        XCTAssertEqual(published.part(.zImage)?.installed, [])
        XCTAssertEqual(published.part(.zImage)?.accepted, false)
        XCTAssertEqual(published.occupancy(), lender.occupancy())
        XCTAssertEqual(published.loras, [.init(lora, bytes: 7)])
    }

    /// An installation's progress crosses as a phase: the file's name alone, never the lender's repository path.
    func testAnInstallationsProgressIsAPhase() {
        XCTAssertEqual(RemoteLibraryJob(task: .installing(.anima), progress: nil).phase, .starting)
        let p = Installer.Progress(file: "Anima/split_files/diffusion_models/anima.safetensors", index: 2, count: 5, received: 30, total: 120)
        XCTAssertEqual(RemoteLibraryJob(task: .installing(.anima), progress: p).phase,
                       .downloading(file: "anima.safetensors", index: 2, count: 5, received: 30, total: 120))
        XCTAssertEqual(p.fraction, 0.25)
        XCTAssertEqual(Installer.Progress(file: "x", index: 1, count: 1, received: 0, total: 0).fraction, 1)
    }
}
