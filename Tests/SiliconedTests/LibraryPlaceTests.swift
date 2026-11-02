import Foundation
import Synchronization
import XCTest
@testable import Siliconed

/// **Where the library lives, moving it, and the copy renders read** — on small files in a temporary
/// folder, without a byte of weights: a map is a name and a few bytes here.
final class LibraryPlaceTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-place-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        // A read-only folder made by a test must be writable again to go.
        _ = try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: root.appendingPathComponent("locked").path)
        try? FileManager.default.removeItem(at: root)
    }

    private func write(_ path: URL, _ text: String) throws {
        try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: path)
    }

    private func read(_ path: URL) -> String? { (try? Data(contentsOf: path)).map { String(decoding: $0, as: UTF8.self) } }

    /// This Mac's space with Z-Image installed (its Standard) and its own files beside the models.
    private func space() throws -> Library {
        let local = root.appendingPathComponent("space")
        let b = Library(root: local)
        try write(URL(fileURLWithPath: b.map(Family.zImage.ditMap)), "dit")
        try write(URL(fileURLWithPath: b.map(Family.zImage.encoderMap)), "encoder")
        try write(URL(fileURLWithPath: Installer(library: b).componentsFolder(.zImage)).appendingPathComponent("vae.safetensors"), "vae")
        try write(URL(fileURLWithPath: b.map("flat.lora.silicon")), "lora")
        try write(URL(fileURLWithPath: b.acceptedLicensesFile), "{}")
        try write(b.cacheFolder.appendingPathComponent("kv/a.kv"), "kv")
        return b
    }

    // ── moving ─────────────────────────────────────────────────────────────────────────────────

    /// **The models go, this Mac's own files stay**, and the space follows the library there: same
    /// bytes, same dates, the old place emptied but for the licenses and the cache.
    func testAMovedLibraryIsFollowedAndTheMachineFilesStay() throws {
        let b = try space()
        let original = try FileManager.default.attributesOfItem(atPath: b.map(Family.zImage.ditMap))[.modificationDate] as? Date
        let destination = LibraryMove.destination(in: root.appendingPathComponent("disk"))
        XCTAssertEqual(destination.lastPathComponent, "Siliconed")
        let moved = try LibraryMove.move(b, to: destination)

        XCTAssertEqual(moved.root.path, destination.path)
        XCTAssertEqual(moved.local.path, b.local.path)
        XCTAssertEqual(Library.placed(in: b.local).root.path, destination.path)
        XCTAssertEqual(read(URL(fileURLWithPath: moved.map(Family.zImage.ditMap))), "dit")
        XCTAssertEqual(read(URL(fileURLWithPath: moved.map("flat.lora.silicon"))), "lora")
        let copied = try FileManager.default.attributesOfItem(atPath: moved.map(Family.zImage.ditMap))[.modificationDate] as? Date
        XCTAssertEqual(copied, original)
        XCTAssertFalse(FileManager.default.fileExists(atPath: b.map(Family.zImage.ditMap)))
        XCTAssertFalse(FileManager.default.fileExists(atPath: Installer(library: b).componentsFolder(.zImage)))
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.appendingPathComponent(".moving").path))
        // This Mac's files: where they were, and where the moved library looks for them.
        XCTAssertEqual(moved.acceptedLicensesFile, b.acceptedLicensesFile)
        XCTAssertEqual(moved.cacheFolder.path, b.cacheFolder.path)
        XCTAssertEqual(moved.socket, b.socket)
        for kept in [b.acceptedLicensesFile, b.cacheFolder.appendingPathComponent("kv/a.kv").path] {
            XCTAssertTrue(FileManager.default.fileExists(atPath: kept), kept)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.appendingPathComponent("store/profil.json").path))
        XCTAssertTrue(moved.isElsewhere)
        XCTAssertTrue(moved.isReachable)
    }

    /// **A dry run moves nothing and says what a move would carry** (`silicontrol library move` without `--yes`): the files, their bytes, the room —
    /// and refuses what the move refuses.
    func testAPlanCountsWhatAMoveCarriesAndTouchesNothing() throws {
        let b = try space()
        let destination = LibraryMove.destination(in: root.appendingPathComponent("disk"))
        let plan = try LibraryMove.plan(b, to: destination)
        XCTAssertFalse(plan.pending.isEmpty)
        XCTAssertEqual(plan.bytes, plan.pending.reduce(0) { $0 + Library.size(b.root.appendingPathComponent($1).path) })
        XCTAssertNotNil(plan.room)
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: b.map(Family.zImage.ditMap)))
        XCTAssertThrowsError(try LibraryMove.plan(b, to: b.root.appendingPathComponent("store/inside"))) {
            guard case .libraryRefused(_, .sameLibrary) = $0 as? EngineError else { return XCTFail("\($0)") }
        }
    }

    /// **Back to this Mac**: the location file goes, and so does the emptied folder on the other disk.
    func testALibraryMovedBackComesHome() throws {
        let b = try space()
        let away = try LibraryMove.move(b, to: LibraryMove.destination(in: root.appendingPathComponent("disk")))
        let home = try LibraryMove.move(away, to: away.local)
        XCTAssertEqual(home.root.path, b.local.path)
        XCTAssertFalse(home.isElsewhere)
        XCTAssertEqual(Library.placed(in: b.local).root.path, b.local.path)
        XCTAssertFalse(FileManager.default.fileExists(atPath: b.local.appendingPathComponent("library-location").path))
        XCTAssertEqual(read(URL(fileURLWithPath: home.map(Family.zImage.ditMap))), "dit")
        XCTAssertFalse(FileManager.default.fileExists(atPath: away.root.path))
    }

    /// **Refused before the first byte**: into the library itself, over another library's models, a
    /// folder that cannot be written to.
    func testAMoveIsRefusedIntoItselfOverAnotherLibraryOrReadOnly() throws {
        let b = try space()
        XCTAssertThrowsError(try LibraryMove.move(b, to: b.root.appendingPathComponent("store/inside"))) {
            XCTAssertEqual(($0 as? EngineError)?.code, "library_refused")
            guard case .libraryRefused(_, .sameLibrary) = $0 as? EngineError else { return XCTFail("\($0)") }
        }
        let other = root.appendingPathComponent("other/Siliconed")
        try write(other.appendingPathComponent("store/someone.silicon"), "x")
        XCTAssertThrowsError(try LibraryMove.move(b, to: other)) {
            guard case .libraryRefused(_, .alreadyALibrary) = $0 as? EngineError else { return XCTFail("\($0)") }
        }
        let locked = root.appendingPathComponent("locked")
        try FileManager.default.createDirectory(at: locked, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: locked.path)
        XCTAssertThrowsError(try LibraryMove.move(b, to: locked.appendingPathComponent("Siliconed"))) {
            guard case .libraryRefused(_, .readOnly) = $0 as? EngineError else { return XCTFail("\($0)") }
        }
        // Nothing moved, nothing switched.
        XCTAssertEqual(Library.placed(in: b.local).root.path, b.local.path)
        XCTAssertEqual(read(URL(fileURLWithPath: b.map(Family.zImage.ditMap))), "dit")
    }

    /// **Stopped, the library is the old one, untouched**; started again in the same folder, the move
    /// is recognized as its own (`.moving`) and ends.
    func testAStoppedMoveLeavesTheLibraryAndResumes() throws {
        let b = try space()
        let destination = LibraryMove.destination(in: root.appendingPathComponent("disk"))
        let stop = Cancellation()
        stop.cancel()
        XCTAssertThrowsError(try LibraryMove.move(b, to: destination, cancellation: stop)) {
            XCTAssertEqual($0 as? EngineError, .cancelled)
        }
        XCTAssertEqual(Library.placed(in: b.local).root.path, b.local.path)
        XCTAssertEqual(read(URL(fileURLWithPath: b.map(Family.zImage.ditMap))), "dit")
        XCTAssertTrue(FileManager.default.fileExists(atPath: destination.appendingPathComponent(".moving").path))
        // A file already there with the original's size and date is kept, not copied again.
        let first = LibraryMove.files(of: b)[0]
        try FileManager.default.createDirectory(at: destination.appendingPathComponent(first).deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try FileCopy.copy(b.root.appendingPathComponent(first).path, destination.appendingPathComponent(first).path,
                          hashed: false, cancellation: nil) { _ in }
        let moved = try LibraryMove.move(b, to: destination)
        XCTAssertEqual(read(URL(fileURLWithPath: moved.map(Family.zImage.encoderMap))), "encoder")
    }

    /// A move carries neither this Mac's files nor what a crash left half-written.
    func testAMoveCarriesNeitherTheMachineFilesNorPartials() throws {
        let b = try space()
        try write(URL(fileURLWithPath: b.map("half.silicon.partiel")), "half")
        let files = LibraryMove.files(of: b)
        XCTAssertTrue(files.contains("store/" + Family.zImage.ditMap))
        XCTAssertFalse(files.contains { $0.hasSuffix(".partiel") || $0.hasPrefix("cache/") || $0 == "store/profil.json"
            || $0 == "accepted-licenses.json" }, "\(files)")
    }

    // ── opening, an absent disk ────────────────────────────────────────────────────────────────

    /// A folder with a library opens (itself, or its `Siliconed` subfolder); one without is refused;
    /// « this Mac's library » undoes the opening, nothing moved.
    func testOpeningAndGoingBackToThisMacsLibrary() throws {
        let b = try space()
        let elsewhere = root.appendingPathComponent("disk/Siliconed")
        try write(elsewhere.appendingPathComponent("store/other.silicon"), "x")
        XCTAssertThrowsError(try b.opening(root.appendingPathComponent("disk/empty"))) {
            guard case .libraryRefused(_, .notALibrary) = $0 as? EngineError else { return XCTFail("\($0)") }
        }
        let opened = try b.opening(root.appendingPathComponent("disk"))
        XCTAssertEqual(opened.root.path, elsewhere.path)
        XCTAssertEqual(Library.placed(in: b.local).root.path, opened.root.path)
        let back = try opened.useLocal()
        XCTAssertEqual(back.root.path, b.local.path)
        XCTAssertEqual(Library.placed(in: b.local).root.path, b.local.path)
        XCTAssertEqual(read(elsewhere.appendingPathComponent("store/other.silicon")), "x")
    }

    /// **A disk that is not there is said, never replaced**: the library stays where it is, named by its disk.
    func testAnAbsentDiskIsSaid() throws {
        let local = root.appendingPathComponent("space")
        let disk = "NoSuchDisk-\(UUID().uuidString.prefix(8))"
        try write(local.appendingPathComponent("library-location"), "/Volumes/\(disk)/Siliconed\n")
        let b = Library.placed(in: local)
        XCTAssertEqual(b.root.path, "/Volumes/\(disk)/Siliconed")
        XCTAssertFalse(b.isReachable)
        XCTAssertEqual(b.volumeName, disk)
        XCTAssertThrowsError(try b.checkReachable()) {
            XCTAssertEqual($0 as? EngineError, .libraryAbsent(path: "/Volumes/\(disk)/Siliconed", volume: disk))
        }
        // This Mac's own library is always there, even empty.
        XCTAssertTrue(Library(root: root.appendingPathComponent("nothing-yet")).isReachable)
    }

    // ── the working copy ───────────────────────────────────────────────────────────────────────

    /// **The copy holds what the model reads, and is a library**: the same versions resolve in it, a
    /// LoRA's path translates, a second preparation copies nothing, a file changed in the library is
    /// copied again — and another family stays out.
    func testTheWorkingCopyHoldsWhatTheModelReads() throws {
        let b = try space()
        try write(URL(fileURLWithPath: b.map(Family.anima.ditMap)), "anima")
        let copy = try WorkingCopy(source: b, base: root.appendingPathComponent("caches"))
        try copy.prepare(models: ["z-image"], loras: [b.map("flat.lora.silicon")])
        let c = copy.library
        XCTAssertEqual(c.local.path, b.local.path)
        XCTAssertEqual(c.ditMap(.zImage), b.ditMap(.zImage))
        XCTAssertEqual(read(URL(fileURLWithPath: c.map(Family.zImage.ditMap))), "dit")
        XCTAssertEqual(read(URL(fileURLWithPath: try c.component(.zImage, "vae.safetensors"))), "vae")
        XCTAssertEqual(copy.path(of: b.map("flat.lora.silicon")), c.map("flat.lora.silicon"))
        XCTAssertEqual(read(URL(fileURLWithPath: c.map("flat.lora.silicon"))), "lora")
        XCTAssertFalse(FileManager.default.fileExists(atPath: c.map(Family.anima.ditMap)))
        XCTAssertEqual(copy.path(of: "/elsewhere/x.lora.silicon"), "/elsewhere/x.lora.silicon")

        let copied = Mutex(0)
        try copy.prepare(models: ["z-image"]) { _ in copied.withLock { $0 += 1 } }
        XCTAssertEqual(copied.withLock { $0 }, 0, "up to date: nothing copied")
        try write(URL(fileURLWithPath: b.map(Family.zImage.ditMap)), "dit, forged again")
        try copy.prepare(models: ["z-image"]) { _ in copied.withLock { $0 += 1 } }
        XCTAssertGreaterThan(copied.withLock { $0 }, 0)
        XCTAssertEqual(read(URL(fileURLWithPath: c.map(Family.zImage.ditMap))), "dit, forged again")
    }

    /// **Another version left in the copy goes**: the library switched to the Compact, the copy reads
    /// the Compact — never the Standard it held.
    func testAnotherVersionLeftInTheCopyGoes() throws {
        let b = try space()
        let copy = try WorkingCopy(source: b, base: root.appendingPathComponent("caches"))
        try copy.prepare(models: ["z-image"])
        try FileManager.default.removeItem(atPath: b.map(Family.zImage.ditMap))
        try FileManager.default.removeItem(atPath: b.map(Family.zImage.encoderMap))
        try write(URL(fileURLWithPath: b.map(Family.zImage.ditMap(.compact))), "compact dit")
        try write(URL(fileURLWithPath: b.map(Family.zImage.encoderMap(.compact))), "compact encoder")
        XCTAssertEqual(b.variant(of: .zImage), .compact)
        try copy.prepare(models: ["z-image"])
        XCTAssertEqual(copy.library.variant(of: .zImage), .compact)
        XCTAssertFalse(FileManager.default.fileExists(atPath: copy.library.map(Family.zImage.ditMap)))
        XCTAssertFalse(FileManager.default.fileExists(atPath: copy.library.map(Family.zImage.encoderMap)))
    }

    /// **A copy no process holds goes at the next launch**; the live one stays; `remove` empties it.
    func testAbandonedCopiesGoAndTheLiveOneStays() throws {
        let b = try space()
        let base = root.appendingPathComponent("caches")
        let live = try WorkingCopy(source: b, base: base)
        let abandoned = base.appendingPathComponent("working-copy-999999")
        try write(abandoned.appendingPathComponent("store/x.silicon"), "x")
        try write(abandoned.appendingPathComponent(".lock"), "")
        WorkingCopy.removeAbandoned(in: base)
        XCTAssertFalse(FileManager.default.fileExists(atPath: abandoned.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: live.folder.path))
        live.remove()
        XCTAssertFalse(FileManager.default.fileExists(atPath: live.folder.path))
    }

    /// A library on the copy's own volume, or this Mac's own space, is read where it is.
    func testNoCopyOnTheSameVolume() throws {
        let b = try space()
        XCTAssertFalse(WorkingCopy.isNeeded(for: b, base: root.appendingPathComponent("caches")))
        let away = Library(root: root.appendingPathComponent("disk/Siliconed"), local: b.local)
        try FileManager.default.createDirectory(at: away.root.appendingPathComponent("store"), withIntermediateDirectories: true)
        XCTAssertFalse(WorkingCopy.isNeeded(for: away, base: root.appendingPathComponent("caches")))
        // A disk not there yet is another disk: its copy is ready for when it comes.
        XCTAssertTrue(WorkingCopy.isNeeded(for: Library(root: URL(fileURLWithPath: "/Volumes/NoSuchDisk-\(UUID())/Siliconed"), local: b.local),
                                           base: root.appendingPathComponent("caches")))
    }
}
