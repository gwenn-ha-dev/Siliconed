import CryptoKit
import Foundation
import XCTest
@testable import Siliconed

/// **The remote Mac's core, without a network**: the pairing played by both sides in memory, a man in
/// the middle caught by the codes, frames cut anywhere, a blob never holding up a message.
final class RemoteTests: XCTestCase {

    // MARK: - Pairing

    /// Plays a pairing through; returns both sides at the point where the codes are shown.
    private func compared(_ client: inout PairingClient, _ server: inout PairingServer) throws {
        let commit = try XCTUnwrap(try server.receive(client.start(name: "Neo")))
        let nonce = try XCTUnwrap(try client.receive(commit))
        let reveal = try XCTUnwrap(try server.receive(nonce))
        XCTAssertNil(try client.receive(reveal))
    }

    func testBothSidesShowTheSameCodeAndHoldTheSameKey() throws {
        var client = PairingClient(), server = PairingServer()
        try compared(&client, &server)
        let code = try XCTUnwrap(client.code)
        XCTAssertEqual(server.code, code)
        XCTAssertEqual(code.count, 6)
        XCTAssertTrue(code.allSatisfy(\.isNumber))
        XCTAssertEqual(server.clientName, "Neo")
        XCTAssertNil(server.key, "nothing is filed before the user accepts")

        let accepted = try server.accept()
        let confirm = try XCTUnwrap(try client.receive(accepted))
        XCTAssertNil(try server.receive(confirm))
        XCTAssertTrue(server.confirmed)
        let key = try XCTUnwrap(client.key)
        XCTAssertEqual(server.key, key)
        XCTAssertEqual(key.identity.count, Pairing.identitySize)
        XCTAssertEqual(key.key.bitCount, 256)
    }

    func testEachPairingGivesAnotherCodeAndKey() throws {
        var codes = Set<String>()
        var keys: [PairedKey] = []
        for _ in 0..<4 {
            var client = PairingClient(), server = PairingServer()
            try compared(&client, &server)
            codes.insert(try XCTUnwrap(client.code))
            _ = try client.receive(try server.accept())
            keys.append(try XCTUnwrap(client.key))
        }
        XCTAssertEqual(codes.count, 4)
        XCTAssertEqual(Set(keys.map(\.identity)).count, 4)
    }

    /// **A man in the middle** pairs with each side on his own keys: the two screens show codes that
    /// differ — the user does not accept.
    func testAManInTheMiddleShowsTwoDifferentCodes() throws {
        var client = PairingClient(), server = PairingServer()
        var towardServer = PairingClient(), towardClient = PairingServer()
        // client ⇄ intruder (as a server) ; intruder (as a client) ⇄ server
        let toIntruder = try XCTUnwrap(try towardClient.receive(client.start(name: "Neo")))
        let toServer = try XCTUnwrap(try server.receive(towardServer.start(name: "Neo")))
        let n1 = try XCTUnwrap(try client.receive(toIntruder))
        let n2 = try XCTUnwrap(try towardServer.receive(toServer))
        let r1 = try XCTUnwrap(try towardClient.receive(n1))
        let r2 = try XCTUnwrap(try server.receive(n2))
        _ = try client.receive(r1)
        _ = try towardServer.receive(r2)
        XCTAssertNotEqual(client.code, server.code)
    }

    /// The server's nonce must open its commitment: one changed afterwards is caught.
    func testARevealThatDoesNotOpenTheCommitmentIsCaught() throws {
        var client = PairingClient(), server = PairingServer()
        let commit = try XCTUnwrap(try server.receive(client.start(name: "Neo")))
        _ = try client.receive(commit)
        XCTAssertThrowsError(try client.receive(.reveal(Data(repeating: 7, count: Pairing.nonceSize)))) {
            XCTAssertEqual($0 as? PairingFailure, .commitmentBroken)
        }
        XCTAssertNil(client.code)
    }

    func testAForgedProofIsRefused() throws {
        var client = PairingClient(), server = PairingServer()
        try compared(&client, &server)
        guard case .accepted(let identity, _) = try server.accept() else { return XCTFail() }
        XCTAssertThrowsError(try client.receive(.accepted(identity: identity, proof: Data(repeating: 0, count: 32)))) {
            XCTAssertEqual($0 as? PairingFailure, .badProof)
        }
        XCTAssertNil(client.key)
        XCTAssertThrowsError(try server.receive(.confirm(proof: Data(repeating: 0, count: 32)))) {
            XCTAssertEqual($0 as? PairingFailure, .badProof)
        }
        XCTAssertNil(server.key)
        XCTAssertFalse(server.confirmed)
    }

    func testTheServerAcceptsOnlyOnceTheCodesAreShown() throws {
        var client = PairingClient(), server = PairingServer()
        XCTAssertThrowsError(try server.accept())
        _ = try server.receive(client.start(name: "Neo"))
        XCTAssertThrowsError(try server.accept(), "no code yet")
    }

    func testRefusalEndsBothSides() throws {
        var client = PairingClient(), server = PairingServer()
        try compared(&client, &server)
        XCTAssertEqual(server.refuse(), .refused)
        XCTAssertThrowsError(try client.receive(.refused)) { XCTAssertEqual($0 as? PairingFailure, .refused) }
        XCTAssertNil(client.key)
        XCTAssertNil(server.key)
        XCTAssertThrowsError(try server.accept())
    }

    func testOutOfOrderAndMalformedMessagesAreRefused() throws {
        var client = PairingClient(), server = PairingServer()
        XCTAssertThrowsError(try client.receive(.reveal(Data()))) { XCTAssertEqual($0 as? PairingFailure, .unexpected) }
        XCTAssertThrowsError(try server.receive(.start(publicKey: Data([4, 1, 2]), name: "x"))) {
            XCTAssertEqual($0 as? PairingFailure, .badKey)
        }
        var s2 = PairingServer()
        _ = try s2.receive(PairingClient().start(name: "Neo"))
        XCTAssertThrowsError(try s2.receive(.nonce(Data([1])))) { XCTAssertEqual($0 as? PairingFailure, .unexpected) }
    }

    /// The code: six digits, zeros kept, and every byte of the transcript moves it.
    func testTheCodeDependsOnEveryPart() {
        let a = Data(repeating: 1, count: 65), b = Data(repeating: 2, count: 65)
        let na = Data(repeating: 3, count: 32), nb = Data(repeating: 4, count: 32)
        let code = Pairing.code(client: a, server: b, clientNonce: na, serverNonce: nb)
        XCTAssertEqual(code.count, 6)
        XCTAssertEqual(Pairing.code(client: a, server: b, clientNonce: na, serverNonce: nb), code)
        var changed = 0
        for part in 0..<4 {
            for i in [0, 31] {
                var parts = [a, b, na, nb]
                parts[part][i] ^= 1
                if Pairing.code(client: parts[0], server: parts[1], clientNonce: parts[2], serverNonce: parts[3]) != code { changed += 1 }
            }
        }
        XCTAssertEqual(changed, 8)
        XCTAssertEqual(Pairing.spaced("048213"), "048 213")
    }

    func testPairingMessagesTravelAsJSON() throws {
        let messages: [PairingMessage] = [
            .start(publicKey: Data([4, 9]), name: "Neo de Gwenn"), .commit(publicKey: Data([1]), commitment: Data([2])),
            .nonce(Data([3])), .reveal(Data([4])), .accepted(identity: Data([5]), proof: Data([6])), .confirm(proof: Data([7])), .refused,
        ]
        for m in messages {
            XCTAssertEqual(try JSONDecoder().decode(PairingMessage.self, from: JSONEncoder().encode(m)), m)
        }
    }

    // MARK: - Frames

    func testFramesSurviveAnyCut() throws {
        let frames = [Frame(kind: .message, payload: Data("{\"a\":1}".utf8)), Frame(kind: .piece, flags: 1, payload: Data([0, 0, 0, 9, 42])),
                      Frame(kind: .message, payload: Data())]
        let stream = frames.reduce(Data()) { $0 + $1.bytes }
        for cut in [1, 2, 5, 7, stream.count] {
            var reader = FrameReader(), out: [Frame] = []
            var i = 0
            while i < stream.count {
                reader.append(stream[i..<min(i + cut, stream.count)])
                while let f = try reader.next() { out.append(f) }
                i += cut
            }
            XCTAssertEqual(out, frames, "cut \(cut)")
        }
    }

    func testAHostileHeaderIsRefusedBeforeAnyAllocation() {
        var r = FrameReader()
        r.append(Data([0, 0, 0xff, 0xff, 0xff, 0xff]))
        XCTAssertThrowsError(try r.next()) { XCTAssertEqual($0 as? FrameFailure, .tooLong(0xffff_ffff)) }
        var u = FrameReader()
        u.append(Data([9, 0, 0, 0, 0, 0]))
        XCTAssertThrowsError(try u.next()) { XCTAssertEqual($0 as? FrameFailure, .unknownKind(9)) }
    }

    /// A blob goes out in pieces; a message sent meanwhile passes before the next piece.
    func testAMessageNeverWaitsBehindABlob() throws {
        var out = Outbox()
        let blob = Data((0..<(Outbox.pieceSize * 2 + 100)).map { UInt8($0 & 0xff) })
        out.send(blob: blob, id: 7)
        let first = try XCTUnwrap(out.next())
        XCTAssertEqual(first.kind, .piece)
        out.send(message: Data("progress".utf8))
        XCTAssertEqual(out.next()?.kind, .message, "the message passes before the second piece")
        var assembler = BlobAssembler()
        XCTAssertNil(try assembler.receive(first))
        var whole: (id: UInt32, data: Data)?
        while let f = out.next() { if let b = try assembler.receive(f) { whole = b } }
        XCTAssertEqual(whole?.id, 7)
        XCTAssertEqual(whole?.data, blob)
        XCTAssertTrue(out.isEmpty)
    }

    func testBlobsAreBounded() throws {
        var small = BlobAssembler(maximumSize: 10, maximumOpen: 2)
        XCTAssertThrowsError(try small.receive(Frame(kind: .piece, payload: Data([0, 0, 0, 1]) + Data(count: 11)))) {
            XCTAssertEqual($0 as? FrameFailure, .blobTooLarge(id: 1))
        }
        _ = try small.receive(Frame(kind: .piece, payload: Data([0, 0, 0, 2, 1])))
        _ = try small.receive(Frame(kind: .piece, payload: Data([0, 0, 0, 3, 1])))
        XCTAssertThrowsError(try small.receive(Frame(kind: .piece, payload: Data([0, 0, 0, 4, 1])))) {
            XCTAssertEqual($0 as? FrameFailure, .tooManyBlobs)
        }
        XCTAssertThrowsError(try small.receive(Frame(kind: .piece, payload: Data([0, 0])))) {
            XCTAssertEqual($0 as? FrameFailure, .truncatedPiece)
        }
        let empty = try small.receive(Frame(kind: .piece, flags: 1, payload: Data([0, 0, 0, 2])))
        XCTAssertEqual(empty?.data, Data([1]))
    }

    // MARK: - Hello

    func testOnlyTheSameBuildTalks() throws {
        func machine(_ version: String) -> Diagnostic.Machine {
            Diagnostic.Machine(chip: "Apple M4", hardwareModel: "Mac16,10", performanceCores: 4, efficiencyCores: 6,
                               gpuCores: 10, memoryBytes: 16 << 30, macOS: "26.1", version: version, build: "engine abc")
        }
        let mini = Hello(name: "Mac mini", machine: machine("0.17 (170)"))
        XCTAssertNil(mini.mismatch(with: Hello(name: "Neo", machine: machine("0.17 (170)"))))
        XCTAssertEqual(mini.mismatch(with: Hello(name: "Neo", machine: machine("0.16 (164)"))),
                       .appVersion(mine: "0.17 (170)", theirs: "0.16 (164)"))
        XCTAssertEqual(mini.mismatch(with: Hello(name: "Neo", machine: machine("0.17 (170)"), protocolVersion: 2)),
                       .protocolVersion(mine: Hello.protocolVersion, theirs: 2))
        XCTAssertEqual(try JSONDecoder().decode(Hello.self, from: JSONEncoder().encode(mini)), mini)
    }

    // MARK: - Statistics and licenses

    func testStatisticsAndLicensesTravelAsJSON() throws {
        let job = RemoteJob(ordinal: 3, model: "z-image", modelName: "Z-Image Turbo", prompt: "p", width: 1024, height: 1024,
                            steps: 8, defaultSteps: 8, loras: [], seeds: [42], detail: .normal, solver: .euler,
                            schedule: .official, previews: true, neuralEngine: nil)
        var t = Engine.Timings(); t.total = 12.5
        var st = RemoteStatistics()
        st.measures = RemoteMeasures(job: job, elapsed: 4.5, evaluations: 8, reduced: 2, tokens: 12, tokenizer: 0.01,
                                     encoder: 0.4, encoding: nil, textCached: false, stepSeconds: [[1.5, 1.4]], decodings: [0.9],
                                     timings: t, footprints: Engine.Footprints())
        st.speeds = ["z-image|1024×1024": RemoteSpeed(full: 1.4, reducedAverage: 0.7, text: nil, image: 0, decoding: 0.9)]
        st.estimates = [4: 31.5]
        st.images = 3; st.seconds = 40.25
        let digest = Data(repeating: 7, count: 32)
        let messages: [SessionMessage] = [
            .statistics(st), .statistics(RemoteStatistics()),
            .licensing(RemoteLicense(model: "krea2", variant: "compact", digest: digest)), .licensing(nil),
            .answerLicense(RemoteLicenseAnswer(model: "krea2", accepted: true, digest: digest)),
            .answerLicense(RemoteLicenseAnswer(model: "krea2", accepted: false, digest: nil)),
        ]
        for m in messages {
            XCTAssertEqual(try JSONDecoder().decode(SessionMessage.self, from: JSONEncoder().encode(m)), m)
        }
        // The running render's age alone is no change; anything else is.
        var later = st
        later.measures?.elapsed = 9
        XCTAssertNotEqual(later, st)
        XCTAssertEqual(later.withoutClock, st.withoutClock)
        later.images = 4
        XCTAssertNotEqual(later.withoutClock, st.withoutClock)
    }

    /// The digest an acceptance carries changes with any word the sheet states, and with nothing else.
    func testALicenseDigestFollowsTheTextShown() {
        let z = ModelCard.zImage, k = ModelCard.krea2
        XCTAssertEqual(LicenseDigest.of(z, variant: .standard), LicenseDigest.of(z, variant: .standard))
        XCTAssertEqual(LicenseDigest.of(z, variant: .standard).count, 32)
        XCTAssertNotEqual(LicenseDigest.of(z, variant: .standard), LicenseDigest.of(k, variant: .standard))
        XCTAssertNotEqual(LicenseDigest.of(k, variant: .standard), LicenseDigest.of(k, variant: .compact),
                          "a Compact's links are third parties'")
    }
}
