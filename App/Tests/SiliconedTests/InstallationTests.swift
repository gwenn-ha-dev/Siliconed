import CryptoKit
import Synchronization
import XCTest
@testable import Siliconed

/// **An installation is a download of pinned published files** (`Installer`, `Downloader`) — on a temporary library
/// and sparse files of the pinned sizes: no byte of weights, no network.
final class InstallationTests: XCTestCase {
    private var root: URL!
    private var library: Library { Library(root: root) }

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-install-\(UUID())")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("store"), withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    /// A file of `bytes` bytes that takes no room: a hole.
    private func lay(_ path: String, bytes: Int) throws {
        try FileManager.default.createDirectory(atPath: (path as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        let fd = open(path, O_RDWR | O_CREAT, 0o644)
        XCTAssertTrue(fd >= 0 && ftruncate(fd, off_t(bytes)) == 0, path)
        close(fd)
    }

    private func lay(_ family: Family, _ v: Variant) throws {
        for file in family.files(v) { try lay(library.map(file.place), bytes: file.bytes) }
    }

    // ── the recipe ─────────────────────────────────────────────────────────────────────────────

    /// Every file of every version is pinned whole: a commit, a size, a digest of its kind, a place inside the store.
    func testEveryPublishedFileIsPinned() {
        let hex = Set("0123456789abcdef")
        for family in Family.allCases {
            XCTAssertEqual(family.variants, [.standard, .compact], "\(family)")
            for v in family.variants {
                let files = family.files(v)
                XCTAssertFalse(files.isEmpty, "\(family) \(v)")
                XCTAssertEqual(Set(files.map(\.place)).count, files.count, "\(family) \(v): one file a place")
                for file in files {
                    XCTAssertTrue(file.repository.revision.count == 40 && file.repository.revision.allSatisfy(hex.contains), file.label)
                    XCTAssertGreaterThan(file.bytes, 0, file.label)
                    switch file.digest {
                    case .sha256(let d): XCTAssertTrue(d.count == 64 && d.allSatisfy(hex.contains), file.label)
                    case .gitBlob(let d):
                        XCTAssertTrue(d.count == 40 && d.allSatisfy(hex.contains), file.label)
                        XCTAssertLessThan(file.bytes, Downloader.chunkedFrom, "\(file.label): a git blob comes in one request")
                    }
                    XCTAssertFalse(file.place.hasPrefix("/") || file.place.contains(".."), file.place)
                    // A generic name lives in the family's folder; only a name that tells its model apart at the top.
                    XCTAssertTrue(!file.place.contains("/") || file.place.hasPrefix("composants/\(family.rawValue)/"), file.place)
                    XCTAssertEqual(file.repository.url(file.path).absoluteString,
                                   "https://huggingface.co/\(file.repository.name)/resolve/\(file.repository.revision)/\(file.path)")
                }
            }
        }
    }

    /// **What an installation lays down is what the model asks for**: with a version's files in place — and nothing
    /// else — the family is installed, the render reads that version, and its model builds.
    func testAnInstalledVersionBuildsItsModel() throws {
        for family in Family.allCases {
            for v in family.variants {
                try? FileManager.default.removeItem(at: root.appendingPathComponent("store"))
                let installer = Installer(library: library)
                XCTAssertFalse(installer.isInstalled(family, v))
                XCTAssertNotNil(family.traits.card.missing(in: library), "\(family) \(v): nothing there yet")
                try lay(family, v)
                XCTAssertTrue(installer.isInstalled(family, v), "\(family) \(v)")
                XCTAssertEqual(installer.installed(family), [v])
                XCTAssertEqual(library.variant(of: family), v)
                let places = Set(family.files(v).map(\.place))
                XCTAssertTrue(places.contains(family.ditMap(v)) && places.contains(family.encoderMap(v)), "\(family) \(v)")
                XCTAssertNil(family.traits.card.missing(in: library), "\(family) \(v)")
            }
        }
    }

    /// A shards' index comes with every shard it names.
    func testAShardedEncoderComesWhole() {
        let places = Family.zImage.files(.standard).map(\.place).filter { $0.contains("/text_encoder/") }
        XCTAssertEqual(places.count, 4)
        XCTAssertTrue(places.contains(ZImagePublished.textEncoder))
    }

    // ── the plan ───────────────────────────────────────────────────────────────────────────────

    /// The plan counts what is missing, minus what an interrupted download already wrote; a file of another size is
    /// not there.
    func testThePlanCountsWhatIsLeftToDownload() throws {
        let installer = Installer(library: library)
        let files = Family.anima.files(.compact), total = files.reduce(0) { $0 + $1.bytes }
        var plan = installer.plan(.anima, .compact)
        XCTAssertEqual([plan.files.count, plan.bytes, plan.resumed, plan.needed], [files.count, total, 0, total])

        try lay(library.map(files[0].place), bytes: files[0].bytes)
        try lay(library.map(files[1].place), bytes: files[1].bytes - 1)
        plan = installer.plan(.anima, .compact)
        XCTAssertEqual(plan.files, Array(files.dropFirst()))

        // 3 MB written of an interrupted file: a `.partiel` as long as the file, holes but for them.
        let partial = Downloader.partial(library.map(files[2].place))
        try lay(partial, bytes: files[2].bytes)
        let handle = try XCTUnwrap(FileHandle(forWritingAtPath: partial))
        try handle.write(contentsOf: Data(repeating: 7, count: 3 << 20))
        try handle.close()
        plan = installer.plan(.anima, .compact)
        XCTAssertGreaterThanOrEqual(plan.resumed, 3 << 20)
        XCTAssertLessThan(plan.resumed, 8 << 20)
        XCTAssertEqual(plan.needed, plan.bytes - plan.resumed)

        XCTAssertTrue(Installer.Plan(family: .anima, variant: .compact, files: files, bytes: 100, resumed: 40, free: 60).fits)
        XCTAssertFalse(Installer.Plan(family: .anima, variant: .compact, files: files, bytes: 100, resumed: 40, free: 59).fits)
    }

    /// Nothing is downloaded for a model whose license was not accepted — refused before the first request.
    func testNoByteBeforeTheLicense() {
        XCTAssertThrowsError(try Installer(library: library).install(.anima, variant: .compact)) {
            XCTAssertEqual($0 as? EngineError, .licenseNotAccepted(model: "anima"))
        }
        XCTAssertEqual(Library.regularFiles(under: root.path).count, 0)
    }

    /// The model's license files first, then the card of each third party the version also downloads from — once.
    func testTheLicensePagesNameEveryRepository() {
        let anima = Installer.licensePages(.anima, .compact).map(\.path)
        XCTAssertEqual(anima.count, 3)
        XCTAssertTrue(anima[0].hasSuffix("/LICENSE.md") && anima[2].hasPrefix("/Bedovyy/Anima-INT8/blob/") && anima[2].hasSuffix("/README.md"))
        XCTAssertEqual(Installer.licensePages(.anima, .standard).count, 2)
        for family in Family.allCases {
            for v in family.variants {
                let pages = Installer.licensePages(family, v)
                XCTAssertEqual(Set(pages).count, pages.count)
                XCTAssertEqual(Array(pages.prefix(family.traits.card.license.urls.count)), family.traits.card.license.urls)
                for file in family.files(v) {
                    XCTAssertTrue(pages.contains { $0.path.hasPrefix("/\(file.repository.name)/blob/\(file.repository.revision)/") }, file.label)
                }
            }
        }
    }

    // ── removing ───────────────────────────────────────────────────────────────────────────────

    /// A version goes with its own files and leaves what the other installed version reads; the family goes whole,
    /// its folder with it. A linked file loses its link, never its target.
    func testUninstallingAVersionLeavesTheOthers() throws {
        let fm = FileManager.default
        let installer = Installer(library: library)
        try lay(.anima, .standard)
        // The Compact's DiT is a link to a file elsewhere.
        let dit = try XCTUnwrap(Family.anima.files(.compact).first { $0.place == Family.anima.ditMap(.compact) })
        let elsewhere = root.appendingPathComponent("elsewhere.safetensors").path
        try lay(elsewhere, bytes: dit.bytes)
        try fm.createSymbolicLink(atPath: library.map(dit.place), withDestinationPath: elsewhere)
        XCTAssertEqual(Set(installer.installed(.anima)), [.standard, .compact])

        // A link gives back nothing of this volume, as the plan counts it: its target stays.
        let freed = try installer.uninstall(.anima, variant: .compact)
        XCTAssertEqual(freed, 0)
        XCTAssertTrue(fm.fileExists(atPath: elsewhere), "the target stays")
        XCTAssertEqual(installer.installed(.anima), [.standard])
        XCTAssertNil(Family.anima.traits.card.missing(in: library))

        try lay(Downloader.partial(library.map(dit.place)), bytes: dit.bytes)
        try installer.uninstall(.anima)
        XCTAssertEqual(installer.installed(.anima), [])
        XCTAssertFalse(fm.fileExists(atPath: library.map("composants/anima")))
        XCTAssertEqual(Library.regularFiles(under: root.appendingPathComponent("store").path).count, 0)
    }

    /// The disk may take a version only in place of the other: the plan says what the other gives back — its own
    /// files, never what the new version reads too —, and removing it for that keeps those.
    func testAVersionMayTakeTheOthersPlace() throws {
        let installer = Installer(library: library)
        try lay(.anima, .compact)
        let plan = installer.plan(.anima, .standard)
        let standard = Set(Family.anima.files(.standard).map(\.place))
        let own = Family.anima.files(.compact).filter { !standard.contains($0.place) }
        XCTAssertFalse(own.isEmpty)
        XCTAssertEqual(plan.others, [.compact])
        XCTAssertEqual(plan.releasable, own.reduce(0) { $0 + $1.bytes })
        XCTAssertEqual(installer.plan(.anima, .compact).others, [])

        func room(_ free: Int) -> Installer.Plan {
            var p = Installer.Plan(family: .anima, variant: .standard, files: plan.files, bytes: 100, resumed: 0, free: free)
            p.others = [.compact]; p.releasable = 30
            return p
        }
        XCTAssertEqual([room(100).fits, room(100).fitsReplacing], [true, false])
        XCTAssertEqual([room(70).fits, room(70).fitsReplacing], [false, true])
        XCTAssertEqual([room(69).fits, room(69).fitsReplacing], [false, false])

        // The Compact goes for the Standard: the files both read stay, though the Standard is not whole yet.
        let freed = try installer.uninstall(.anima, variant: .compact, keeping: .standard)
        XCTAssertEqual(freed, plan.releasable)
        XCTAssertEqual(Set(installer.missing(.anima, .standard).map(\.place)), Set(plan.files.map(\.place)))
        XCTAssertEqual(installer.installed(.anima), [])
    }

    /// With both versions whole on disk, the render reads the one the user picked; the occupancy counts a file both
    /// read once, and what an interrupted download left apart.
    func testBothVersionsLiveSideBySide() throws {
        let installer = Installer(library: library)
        XCTAssertEqual(library.occupancy().families.first { $0.family == .anima }, .init(family: .anima, installed: [], variant: nil, bytes: 0))
        try lay(.anima, .standard)
        try lay(.anima, .compact)
        try installer.use(.anima, variant: .standard)
        XCTAssertEqual(library.variant(of: .anima), .standard)
        try installer.use(.anima, variant: .compact)
        XCTAssertEqual(library.variant(of: .anima), .compact)

        let partial = Downloader.partial(library.map(Family.zImage.ditMap(.compact)))
        try lay(partial, bytes: 1 << 20)
        let handle = try XCTUnwrap(FileHandle(forWritingAtPath: partial))
        try handle.write(contentsOf: Data(repeating: 7, count: 1 << 20))
        try handle.close()

        let o = library.occupancy()
        let anima = try XCTUnwrap(o.families.first { $0.family == .anima })
        let places = Set((Family.anima.files(.standard) + Family.anima.files(.compact)).map(\.place))
        let unique = Dictionary((Family.anima.files(.standard) + Family.anima.files(.compact)).map { ($0.place, $0.bytes) }, uniquingKeysWith: { a, _ in a })
        XCTAssertLessThan(places.count, Family.anima.files(.standard).count + Family.anima.files(.compact).count, "the versions share files")
        XCTAssertEqual(anima.bytes, unique.values.reduce(0, +))
        XCTAssertEqual(Set(anima.installed), [.standard, .compact])
        XCTAssertEqual(anima.variant, .compact)
        XCTAssertGreaterThanOrEqual(o.downloads, 1 << 20)
        XCTAssertEqual(o.total, anima.bytes + o.downloads)
        XCTAssertEqual(o.families.first { $0.family == .zImage }?.present, false)
        try library.emptyDownloads()
        XCTAssertEqual(library.occupancy().downloads, 0)
    }

    /// What an interrupted download left goes; one being written (locked) stays.
    func testCleaningLeavesADownloadBeingWritten() throws {
        let installer = Installer(library: library)
        let dead = library.map("a.safetensors.partiel"), live = library.map("composants/anima/b.safetensors.partiel")
        try lay(dead, bytes: 10)
        try lay(live, bytes: 10)
        try lay(library.map("kept.safetensors"), bytes: 10)
        let held = open(live, O_RDONLY)
        XCTAssertEqual(flock(held, LOCK_EX | LOCK_NB), 0)
        try installer.clean()
        XCTAssertFalse(FileManager.default.fileExists(atPath: dead))
        XCTAssertTrue(FileManager.default.fileExists(atPath: live))
        close(held)
        try installer.clean()
        XCTAssertFalse(FileManager.default.fileExists(atPath: live))
        XCTAssertTrue(FileManager.default.fileExists(atPath: library.map("kept.safetensors")))
    }

    /// What a removal gives back, counted one way by the plan, the removal, the cleaning and the occupancy: a link
    /// nothing, a `.partiel` the blocks it holds (as long as its file, a hole beyond), a file its size.
    func testWhatARemovalGivesBack() throws {
        let file = library.map("a.safetensors"), partial = library.map("b.safetensors.partiel"), link = library.map("c.safetensors")
        try Data(repeating: 1, count: 5000).write(to: URL(fileURLWithPath: file))
        try lay(partial, bytes: 1 << 30)
        let handle = try XCTUnwrap(FileHandle(forWritingAtPath: partial))
        try handle.write(contentsOf: Data(repeating: 7, count: 1 << 20))
        try handle.close()
        try FileManager.default.createSymbolicLink(atPath: link, withDestinationPath: file)
        XCTAssertEqual(Library.reclaimable(file), 5000)
        XCTAssertEqual(Library.reclaimable(link), 0)
        XCTAssertGreaterThanOrEqual(Library.reclaimable(partial), 1 << 20)
        XCTAssertLessThan(Library.reclaimable(partial), 1 << 29, "the hole is not counted")
        XCTAssertEqual(Library.reclaimable(library.map("absent")), 0)
        let expected = Library.reclaimable(partial)
        XCTAssertEqual(try Installer(library: library).clean(), expected)
    }

    /// Two downloads of one file never write it together: the second is refused by name before a byte is asked —
    /// a large file's (chunks and ledger) as a small one's.
    func testASecondDownloadOfTheSameFileIsRefused() throws {
        let r = Repository("a/b", String(repeating: "0", count: 40))
        let sha = String(repeating: "a", count: 64)
        for bytes in [1000, Downloader.chunkedFrom + 1] {
            let file = PublishedFile(r, "x.safetensors", at: "x.safetensors", bytes: bytes, sha256: sha)
            let destination = library.map("x-\(bytes).safetensors")
            let held = try Downloader.hold(Downloader.partial(destination), label: file.label)
            defer { close(held) }
            XCTAssertThrowsError(try Downloader.fetch(file, to: destination, journal: { _ in })) {
                XCTAssertEqual($0 as? EngineError, .downloadInProgress(file: "b/x.safetensors"))
            }
            XCTAssertFalse(FileManager.default.fileExists(atPath: Downloader.ledger(destination)), "nothing of the other's touched")
        }
    }

    /// A resumption rewrites its ledger whole before appending: a last line a crash cut is not glued to the next
    /// chunk's into one never written ("1" + "5" was read back as chunk 15).
    func testAResumedLedgerIsRewrittenWhole() {
        let sha = String(repeating: "a", count: 64)
        let cut = "200 \(sha) 10\n0\n1"
        let done = Downloader.resumedChunks(ledger: cut, size: 200, digest: sha, chunk: 10)
        XCTAssertEqual(done, [0])
        let rewritten = Downloader.ledgerText(size: 200, digest: sha, done: done, chunk: 10)
        XCTAssertEqual(rewritten, "200 \(sha) 10\n0\n")
        XCTAssertEqual(Downloader.resumedChunks(ledger: rewritten + "5\n", size: 200, digest: sha, chunk: 10), [0, 5])
        XCTAssertEqual(Downloader.resumedChunks(ledger: cut + "5\n", size: 200, digest: sha, chunk: 10), [0, 15], "what appending alone did")
        XCTAssertEqual(Downloader.ledgerText(size: 200, digest: sha, done: [], chunk: 10), "200 \(sha) 10\n")
    }

    // ── the download's pure parts ──────────────────────────────────────────────────────────────

    /// A digest is computed in the file's kind: the sha256 of an LFS object, git's blob id of a small file.
    func testAFileIsJudgedByItsKindOfDigest() throws {
        let path = root.appendingPathComponent("abc").path
        try Data("abc".utf8).write(to: URL(fileURLWithPath: path))
        let r = Repository("a/b", String(repeating: "0", count: 40))
        let lfs = PublishedFile(r, "x", at: "x", bytes: 3, sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        let blob = PublishedFile(r, "x", at: "x", bytes: 3, gitBlob: "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f")
        XCTAssertEqual(try lfs.digest(of: path), lfs.digest)
        XCTAssertEqual(try blob.digest(of: path), blob.digest)
        XCTAssertNotEqual(try lfs.digest(of: path), blob.digest)
        XCTAssertEqual(lfs.label, "b/x")
    }

    func testDownloadChunksCoverTheFile() {
        for size in [1, 64 << 20, (64 << 20) + 1, 1_000_000_007, 9_349_769_248] {
            let c = Downloader.chunks(size: size)
            XCTAssertEqual(c.first?.lowerBound, 0)
            XCTAssertEqual(c.last?.upperBound, size)
            for (a, b) in zip(c, c.dropFirst()) { XCTAssertEqual(a.upperBound, b.lowerBound) }
            XCTAssertTrue(c.allSatisfy { !$0.isEmpty && $0.count <= 64 << 20 }, "\(size)")
            XCTAssertEqual(c.count, (size + (64 << 20) - 1) / (64 << 20))
        }
    }

    /// A resumed download keeps only the chunks of the same file; a damaged ledger starts over.
    func testDownloadResumesOnlyTheSameFile() {
        let sha = String(repeating: "a", count: 64)
        let ledger = "100 \(sha) 10\n0\n3\n7\n"                    // 10 chunks of 10 bytes
        XCTAssertEqual(Downloader.resumedChunks(ledger: ledger, size: 100, digest: sha, chunk: 10), [0, 3, 7])
        XCTAssertEqual(Downloader.resumedChunks(ledger: ledger, size: 101, digest: sha, chunk: 10), [])
        XCTAssertEqual(Downloader.resumedChunks(ledger: ledger, size: 100, digest: String(repeating: "b", count: 64), chunk: 10), [])
        XCTAssertEqual(Downloader.resumedChunks(ledger: "", size: 100, digest: sha, chunk: 10), [])
        // A line a crash cut ("1" of "12") is not a chunk; nor is a head without its newline.
        XCTAssertEqual(Downloader.resumedChunks(ledger: "100 \(sha) 10\n2\n1", size: 100, digest: sha, chunk: 10), [2])
        XCTAssertEqual(Downloader.resumedChunks(ledger: "100 \(sha) 10", size: 100, digest: sha, chunk: 10), [])
        // The head the downloader writes is the one it reads back.
        XCTAssertEqual(Downloader.resumedChunks(ledger: Downloader.ledgerHead(size: 100, digest: sha) + "\n0\n", size: 100, digest: sha), [0])
        for tail in ["2\n-1\n", "2\n10\n", "2\nx\n", "2\n9223372036854775807\n"] {
            XCTAssertEqual(Downloader.resumedChunks(ledger: "100 \(sha) 10\n" + tail, size: 100, digest: sha, chunk: 10), [], tail)
        }
        XCTAssertEqual(Downloader.resumedChunks(ledger: "100 \(sha) 20\n2\n", size: 100, digest: sha, chunk: 10), [])
    }

    /// The running hash takes each chunk once it and every one before it are whole, in any order of arrival; stopped
    /// before the end, it gives no digest.
    func testTheRunningHashWaitsForEachChunkInOrder() throws {
        let size = (3 << 20) + 777
        var generator = SystemRandomNumberGenerator()
        let data = Data((0..<size).map { _ in UInt8.random(in: 0...255, using: &generator) })
        let path = root.appendingPathComponent("download.partiel").path, ledgerPath = path + ".morceaux"
        try data.write(to: URL(fileURLWithPath: path))
        FileManager.default.createFile(atPath: ledgerPath, contents: nil)
        let ledger = try XCTUnwrap(FileHandle(forWritingAtPath: ledgerPath))
        defer { try? ledger.close() }
        let chunks = Downloader.chunks(size: size, chunk: 256 << 10)
        let run = Downloader.ChunkRun(fd: -1, ledger: ledger, size: size, label: "test", whole: [0, 3], already: 0,
                                      cancellation: nil, journal: { _ in })
        let hash = Downloader.RunningHash(path: path, chunks: chunks, run: run)
        for k in chunks.indices.reversed() where k != 0 && k != 3 { run.written(k) }
        XCTAssertEqual(try hash.result(), SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
        XCTAssertEqual(try Downloader.sha256(path), SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
        let stopping = Cancellation()
        let cut = Downloader.ChunkRun(fd: -1, ledger: ledger, size: size, label: "test", whole: [0], already: 0,
                                      cancellation: stopping, journal: { _ in })
        let partial = Downloader.RunningHash(path: path, chunks: chunks, run: cut)
        stopping.cancel()
        XCTAssertNil(try partial.result())
    }

    /// A download says where it stands at once — what a resumed one already holds —, then every megabyte and at the
    /// last byte: never once a write.
    func testADownloadTellsItsBytesByTheMegabyte() throws {
        let ledgerPath = root.appendingPathComponent("told.morceaux").path
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: ledgerPath, contents: nil)
        let ledger = try XCTUnwrap(FileHandle(forWritingAtPath: ledgerPath))
        defer { try? ledger.close() }
        let told = Mutex<[Int]>([])
        let size = (5 << 20) + 100, already = 2 << 20
        let run = Downloader.ChunkRun(fd: -1, ledger: ledger, size: size, label: "test", whole: [], already: already,
                                      cancellation: nil, progress: { n in told.withLock { $0.append(n) } }, journal: { _ in })
        XCTAssertEqual(told.withLock { $0 }, [already])
        for _ in 0..<48 { run.add(64 << 10) }   // 3 MiB in 64-KiB writes
        XCTAssertEqual(told.withLock { $0 }, [2 << 20, 3 << 20, 4 << 20, 5 << 20])
        run.add(100)
        XCTAssertEqual(told.withLock { $0 }.last, size)
    }

    /// The shared files an older app left are taken on their size: hashed once before a render (`checkShared`), one
    /// that is not the pinned bytes is removed and refused, and the family is no longer installed. A version's own DiT
    /// and encoder are never among them (Z-Image's text encoder, 8 GB, is the Standard's).
    func testASharedFileOfTheRightSizeButWrongBytesIsRemoved() throws {
        let z = Family.zImage
        XCTAssertEqual(Set(Installer(library: library).shared(z).map(\.place)),
                       ["composants/z-image/tokenizer/tokenizer.json", "composants/z-image/vae.safetensors"])
        try lay(z, .compact)
        var installer = Installer(library: library)
        installer.journal = { _ in }
        XCTAssertTrue(installer.isInstalled(z, .compact))
        var refused: [String] = []
        for _ in 0..<3 {
            do { try installer.checkShared(z) } catch {
                guard case .corruptMap(let file?) = error else { return XCTFail("\(error)") }
                refused.append((file as NSString).lastPathComponent)
            }
        }
        XCTAssertEqual(refused.sorted(), ["tokenizer.json", "vae.safetensors"])
        XCTAssertFalse(installer.isInstalled(z, .compact))
        XCTAssertEqual(installer.missing(z, .compact).map(\.place).sorted(),
                       ["composants/z-image/tokenizer/tokenizer.json", "composants/z-image/vae.safetensors"])
    }
}
