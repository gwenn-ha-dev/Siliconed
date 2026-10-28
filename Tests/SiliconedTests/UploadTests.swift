import CryptoKit
import Foundation
import Network
import XCTest
@testable import Siliconed

/// **A file going up to the other Mac** (`Remote/Upload.swift`): the messages, the stream cut into pieces
/// and put back in a file with its digest, what is refused before the first byte, what a failure leaves
/// behind (nothing), and the whole on the loopback.
final class UploadTests: XCTestCase {

    private func folder() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("upload-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return d
    }

    /// A file of `size` pseudo-random bytes, written in slices (not held whole by the test either).
    private func file(_ size: Int, in dir: URL, name: String = "model.safetensors") throws -> URL {
        let url = dir.appendingPathComponent(name)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let h = try FileHandle(forWritingTo: url)
        var g = SystemRandomNumberGenerator()
        var left = size
        while left > 0 {
            let n = min(left, 1 << 20)
            try h.write(contentsOf: Data((0..<n).map { _ in UInt8.random(in: 0...255, using: &g) }))
            left -= n
        }
        try h.close()
        return url
    }

    private func digest(_ url: URL) throws -> Data {
        var hasher = SHA256()
        let h = try FileHandle(forReadingFrom: url)
        defer { try? h.close() }
        while let d = try h.read(upToCount: 1 << 20), !d.isEmpty { hasher.update(data: d) }
        return Data(hasher.finalize())
    }

    /// The source through the outbox and the reader, as the wire would cut it: every frame, in order.
    private func frames(of source: UploadSource, id: UInt32 = 5) -> [Frame] {
        var outbox = Outbox()
        outbox.send(stream: source, id: id)
        var reader = FrameReader()
        var out: [Frame] = []
        while let f = outbox.next() {
            reader.append(f.bytes)
            while let r = try? reader.next() { out.append(r) }
        }
        return out
    }

    // MARK: - Messages

    func testTheUploadMessagesTravelAsJSON() throws {
        let messages: [SessionMessage] = [
            .uploadOffer(UploadOffer(id: 3, name: "flux-ish.safetensors", size: 23_800_000_000)),
            .uploadCancel(3),
            .uploadAnswer(id: 3, .accepted),
            .uploadAnswer(id: 3, .refused(.noSpace(needed: 49 << 30, available: 12 << 30))),
            .uploadAnswer(id: 3, .refused(.busy)),
            .uploadAnswer(id: 3, .refused(.notImportable)),
            .uploadResult(id: 3, .verified),
            .uploadResult(id: 3, .failed(.fingerprint)),
        ]
        for m in messages {
            XCTAssertEqual(try JSONDecoder().decode(SessionMessage.self, from: JSONEncoder().encode(m)), m)
        }
    }

    // MARK: - Cutting and putting back

    func testAFileIsCutIntoPiecesAndPutBackWithItsDigest() throws {
        let dir = try folder()
        // Not a multiple of 64 KiB, several pieces.
        let size = 3 * (64 << 10) * 5 + 12_345
        let url = try file(size, in: dir)
        let progress = Progress()
        let finished = Finished()
        let source = UploadSource(url: url, size: Int64(size), progress: { progress.add($0) }, finished: { finished.set($0) })
        let fs = frames(of: source)

        XCTAssertTrue(fs.allSatisfy { $0.kind == .stream })
        XCTAssertTrue(fs.dropLast().allSatisfy { $0.flags & 1 == 0 && $0.payload.count <= 4 + Outbox.pieceSize })
        XCTAssertEqual(fs.last?.flags, 1)
        XCTAssertEqual(try fs.last?.streamPiece().bytes.count, UploadSource.digestSize, "the last piece is the digest")
        XCTAssertEqual(progress.last, Int64(size))
        XCTAssertEqual(finished.value, .sent)

        let inbox = UploadInbox(directory: dir.appendingPathComponent("in"), extensions: ["safetensors"], available: { _ in 1 << 40 })
        let mac = Data([1, 2, 3])
        XCTAssertNil(inbox.begin(identity: mac, UploadOffer(id: 5, name: "model.safetensors", size: Int64(size))))
        var done: URL?
        for f in fs {
            let p = try f.streamPiece()
            let (_, step) = try XCTUnwrap(inbox.piece(identity: mac, id: p.id, p.bytes, last: p.last))
            if case .done(let u) = step { done = u }
            if case .failed(let why) = step { XCTFail("\(why)") }
        }
        let received = try XCTUnwrap(done)
        XCTAssertEqual(received.lastPathComponent, "model.safetensors")
        XCTAssertEqual(try digest(received), try digest(url))
        XCTAssertEqual(try received.resourceValues(forKeys: [.fileSizeKey]).fileSize, size)
        XCTAssertFalse(inbox.isActive, "the slot is free after a file")
    }

    func testAMessageGoesBetweenTwoPiecesOfAStream() throws {
        let dir = try folder()
        let size = 10 * (64 << 10)
        let source = UploadSource(url: try file(size, in: dir), size: Int64(size))
        var outbox = Outbox()
        outbox.send(stream: source, id: 1)
        XCTAssertEqual(outbox.next()?.kind, .stream)
        XCTAssertEqual(outbox.next()?.kind, .stream)
        outbox.send(message: Data("{}".utf8))
        XCTAssertEqual(outbox.next()?.kind, .message, "a small message waits for one piece at most")
        outbox.send(blob: Data(repeating: 7, count: 100), id: 9)
        XCTAssertEqual(outbox.next()?.kind, .piece, "a PNG goes before the rest of an upload")
        XCTAssertEqual(outbox.next()?.kind, .stream)
    }

    // MARK: - What goes wrong

    private func receive(_ size: Int, tamper: (inout [Frame]) -> Void, name: String = "m.safetensors") throws -> [UploadReceiver.Step] {
        let dir = try folder()
        let url = try file(size, in: dir)
        var fs = frames(of: UploadSource(url: url, size: Int64(size)))
        tamper(&fs)
        let inbox = UploadInbox(directory: dir.appendingPathComponent("in"), extensions: ["safetensors"], available: { _ in 1 << 40 })
        let mac = Data([9])
        XCTAssertNil(inbox.begin(identity: mac, UploadOffer(id: 5, name: name, size: Int64(size))))
        var steps: [UploadReceiver.Step] = []
        for f in fs {
            let p = try f.streamPiece()
            guard let (_, step) = inbox.piece(identity: mac, id: p.id, p.bytes, last: p.last) else { break }
            steps.append(step)
            if case .failed = step {
                XCTAssertFalse(inbox.isActive)
                let left = (try? FileManager.default.contentsOfDirectory(atPath: dir.appendingPathComponent("in").path)) ?? []
                for sub in left {
                    XCTAssertEqual((try? FileManager.default.contentsOfDirectory(atPath: dir.appendingPathComponent("in/\(sub)").path))?.count ?? 0, 0,
                                   "a failed upload leaves no file")
                }
                break
            }
        }
        return steps
    }

    func testAFlippedByteFailsTheFingerprintAndLeavesNothing() throws {
        let steps = try receive(300_000) { fs in
            var p = fs[1].payload
            p[p.startIndex + 10] ^= 1
            fs[1] = Frame(kind: .stream, payload: p)
        }
        XCTAssertEqual(steps.last, .failed(.fingerprint))
    }

    func testAMissingPieceFailsTheSize() throws {
        let steps = try receive(300_000) { fs in fs.remove(at: 2) }
        XCTAssertEqual(steps.last, .failed(.size))
    }

    func testMoreBytesThanTheOfferFailsAtOnce() throws {
        let steps = try receive(300_000) { fs in fs.insert(fs[0], at: 1) }
        XCTAssertEqual(steps.last, .failed(.size))
    }

    func testAnOfferIsJudgedBeforeTheFirstByte() throws {
        let dir = try folder()
        let mac = Data([1])
        func inbox(free: Int64?) -> UploadInbox {
            UploadInbox(directory: dir.appendingPathComponent("in"), extensions: ["safetensors", "gguf"], available: { _ in free })
        }
        let size: Int64 = 10 << 30
        let tight = inbox(free: size)
        XCTAssertEqual(tight.begin(identity: mac, UploadOffer(id: 1, name: "a.safetensors", size: size)),
                       .noSpace(needed: UploadStore.needed(for: size), available: size))
        XCTAssertFalse(tight.isActive)
        XCTAssertEqual(inbox(free: nil).begin(identity: mac, UploadOffer(id: 1, name: "a.safetensors", size: 5)), .cannotWrite)

        let ok = inbox(free: 1 << 40)
        XCTAssertEqual(ok.begin(identity: mac, UploadOffer(id: 1, name: "a.png", size: 5)), .notImportable)
        XCTAssertEqual(ok.begin(identity: mac, UploadOffer(id: 1, name: "a.safetensors", size: 0)), .notImportable)
        XCTAssertEqual(ok.begin(identity: mac, UploadOffer(id: 1, name: ".hidden.safetensors", size: 5)), .notImportable)
        XCTAssertNil(ok.begin(identity: mac, UploadOffer(id: 1, name: "../../etc/evil.safetensors", size: 5)))
        XCTAssertEqual(ok.begin(identity: Data([2]), UploadOffer(id: 2, name: "b.gguf", size: 5)), .busy, "one upload at a time")
        // A name with a path is cut to its last component: nothing is written outside the folder.
        let (offer, _) = try XCTUnwrap(ok.piece(identity: mac, id: 1, Data([1]), last: false))
        XCTAssertEqual(offer.name, "evil.safetensors")
        // Another Mac's pieces, or an unknown id, are not ours.
        XCTAssertNil(ok.piece(identity: Data([2]), id: 1, Data([1]), last: false))
        XCTAssertNil(ok.piece(identity: mac, id: 7, Data([1]), last: false))
        // The Mac leaves: the partial file goes.
        XCTAssertNotNil(ok.abort(identity: mac))
        XCTAssertFalse(ok.isActive)
        let leftovers = try FileManager.default.subpathsOfDirectory(atPath: dir.appendingPathComponent("in").path)
        XCTAssertFalse(leftovers.contains { $0.hasSuffix(".safetensors") })
    }

    func testTheMinisStopCutsTheUpload() throws {
        let dir = try folder()
        let flag = Finished()
        let inbox = UploadInbox(directory: dir.appendingPathComponent("in"), extensions: ["safetensors"], available: { _ in 1 << 40 })
        let mac = Data([1])
        XCTAssertNil(inbox.begin(identity: mac, UploadOffer(id: 1, name: "a.safetensors", size: 100), cancelled: { flag.value != nil }))
        XCTAssertEqual(inbox.piece(identity: mac, id: 1, Data(count: 10), last: false)?.step, .more(received: 10, advanced: true))
        flag.set(.success(()))
        XCTAssertEqual(inbox.piece(identity: mac, id: 1, Data(count: 10), last: false)?.step, .failed(.cancelled))
        XCTAssertFalse(inbox.isActive)
    }

    func testTheSendersStopAndAVanishedFileEndTheStream() throws {
        let dir = try folder()
        let url = try file(500_000, in: dir)
        let stop = Finished(), result = Finished()
        let source = UploadSource(url: url, size: 500_000, cancelled: { stop.value != nil }, finished: { result.set($0) })
        XCTAssertNoThrow(try source.next(1000))
        stop.set(.success(()))
        XCTAssertThrowsError(try source.next(1000))
        XCTAssertEqual(result.value, .failed(.cancelled))

        // A file that shrank is not sent as if whole.
        let small = try file(1000, in: dir, name: "s.safetensors")
        let shrunk = Finished()
        let s2 = UploadSource(url: small, size: 5000, finished: { shrunk.set($0) })
        XCTAssertThrowsError(try s2.next(64 << 10))
        XCTAssertEqual(shrunk.value, .failed(.cannotRead))
    }

    // MARK: - The loopback

    /// The whole of it, over TLS-PSK on the loopback: the offer up, the answer down, a few hundred KB as a
    /// stream with a message sent in the middle, the verdict down.
    func testAFileGoesUpOverTheNetworkAndIsVerified() throws {
        let hello = Hello(name: "x", machine: Diagnostic.Machine(chip: "Apple M4", hardwareModel: "Mac16,10", performanceCores: 4,
                                                                  efficiencyCores: 6, gpuCores: 10, memoryBytes: 16 << 30,
                                                                  macOS: "26.1", version: "0.17 (170)", build: "engine abc"))
        let host = RemoteHost(hello: hello, service: nil, loopbackOnly: true)
        let hostEvents = RemoteNetworkTests.Events<RemoteHost.Event>()
        host.start { hostEvents.add($0) }
        addTeardownBlock { host.stop() }
        let key = PairedKey(identity: Pairing.random(Pairing.identitySize), key: SymmetricKey(size: .bits256))
        host.lend(keys: [key])
        let port = try XCTUnwrap(hostEvents.next { if case .lending(let p?) = $0 { p } else { nil } })

        let dir = try folder()
        let size = 700_000
        let url = try file(size, in: dir)
        let inbox = UploadInbox(directory: dir.appendingPathComponent("in"), extensions: ["safetensors"], available: { _ in 1 << 40 })
        let steps = RemoteNetworkTests.Events<UploadReceiver.Step>()
        host.receive(streams: { identity, id, bytes, last in
            if let (_, step) = inbox.piece(identity: identity, id: id, bytes, last: last) { steps.add(step) }
        })

        let link = RemoteLink(key: key, hello: hello) { .hostPort(host: .ipv4(.loopback), port: NWEndpoint.Port(rawValue: port)!) }
        let answers = RemoteNetworkTests.Events<SessionMessage>()
        link.listen(messages: { answers.add($0) }, blobs: { _, _ in })
        let states = RemoteNetworkTests.Events<RemoteLink.State>()
        link.connect { states.add($0) }
        addTeardownBlock { link.disconnect() }
        XCTAssertNotNil(states.next { if case .connected = $0 { true } else { nil } })
        let identity = try XCTUnwrap(hostEvents.next { if case .connected(let i, _) = $0 { i } else { nil } })

        // The offer goes up; the mini judges it and answers.
        let offer = UploadOffer(id: 11, name: "model.safetensors", size: Int64(size))
        link.send(.uploadOffer(offer))
        let asked = try XCTUnwrap(hostEvents.next { if case .message(_, .uploadOffer(let o)) = $0 { o } else { nil } })
        XCTAssertEqual(asked, offer)
        XCTAssertNil(inbox.begin(identity: identity, asked))
        host.send(.uploadAnswer(id: 11, .accepted), to: identity)
        XCTAssertNotNil(answers.next { if case .uploadAnswer(11, .accepted) = $0 { true } else { nil } })

        // The stream goes, and a small message sent right after it still arrives first of what follows.
        let ended = Finished()
        link.send(stream: UploadSource(url: url, size: Int64(size), finished: { ended.set($0) }), id: 11)
        link.send(.uploadCancel(99))
        XCTAssertNotNil(hostEvents.next { if case .message(_, .uploadCancel(99)) = $0 { true } else { nil } })
        let done = try XCTUnwrap(steps.next(10) { if case .done(let u) = $0 { u } else { nil } })
        XCTAssertEqual(try digest(done), try digest(url))
        XCTAssertEqual(ended.value, .sent)
        host.send(.uploadResult(id: 11, .verified), to: identity)
        XCTAssertNotNil(answers.next { if case .uploadResult(11, .verified) = $0 { true } else { nil } })
    }

    // MARK: - Small helpers

    final class Progress: @unchecked Sendable {
        private let lock = NSLock(); private var _last: Int64 = 0
        func add(_ n: Int64) { lock.withLock { _last = n } }
        var last: Int64 { lock.withLock { _last } }
    }

    final class Finished: @unchecked Sendable {
        enum Outcome: Equatable { case sent, failed(UploadFailure) }
        private let lock = NSLock(); private var _value: Outcome?
        func set(_ r: Result<Void, UploadFailure>) {
            lock.withLock { switch r { case .success: _value = .sent; case .failure(let f): _value = .failed(f) } }
        }
        var value: Outcome? { lock.withLock { _value } }
    }
}
