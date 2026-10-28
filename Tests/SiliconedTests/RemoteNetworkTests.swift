import CryptoKit
import Foundation
import Network
import XCTest
@testable import Siliconed

/// **The remote Mac's channel, on the loopback**: the two listeners and the session as they run between
/// two Macs, without Bonjour (the port is read from the listener) — paired, an intruder turned away, a
/// wrong key refused, another build refused, a session found again after a cut, a forgotten Mac shut out.
final class RemoteNetworkTests: XCTestCase {

    /// Events as they arrive, from any queue; a test waits for the first that matches.
    final class Events<T: Sendable>: @unchecked Sendable {
        private let condition = NSCondition()
        private var seen: [T] = []

        func add(_ e: T) { condition.lock(); seen.append(e); condition.broadcast(); condition.unlock() }

        /// The first event since the last one taken that matches, in at most `timeout` seconds.
        func next<R>(_ timeout: TimeInterval = 5, _ match: (T) -> R?) -> R? {
            let deadline = Date().addingTimeInterval(timeout)
            condition.lock(); defer { condition.unlock() }
            while true {
                while !seen.isEmpty {
                    if let r = match(seen.removeFirst()) { return r }
                }
                guard condition.wait(until: deadline) else { return nil }
            }
        }
    }

    private func hello(_ name: String, version: String = "0.17 (170)") -> Hello {
        Hello(name: name, machine: Diagnostic.Machine(chip: "Apple M4", hardwareModel: "Mac16,10", performanceCores: 4,
                                                      efficiencyCores: 6, gpuCores: 10, memoryBytes: 16 << 30,
                                                      macOS: "26.1", version: version, build: "engine abc"))
    }

    private func host(_ version: String = "0.17 (170)") -> (RemoteHost, Events<RemoteHost.Event>) {
        let h = RemoteHost(hello: hello("Mac mini", version: version), service: nil, loopbackOnly: true)
        let events = Events<RemoteHost.Event>()
        h.start { events.add($0) }
        addTeardownBlock { h.stop() }
        return (h, events)
    }

    private static func loopback(_ port: UInt16) -> NWEndpoint { .hostPort(host: .ipv4(.loopback), port: NWEndpoint.Port(rawValue: port)!) }

    private func randomKey() -> PairedKey {
        PairedKey(identity: Pairing.random(Pairing.identitySize), key: SymmetricKey(size: .bits256))
    }

    // MARK: - Pairing

    private func pairingPort(_ h: RemoteHost, _ events: Events<RemoteHost.Event>) throws -> UInt16 {
        h.openPairing()
        return try XCTUnwrap(events.next { if case .pairing(.waiting(let p)) = $0 { p } else { nil } })
    }

    private func attempt(_ port: UInt16) -> (RemotePairingAttempt, Events<RemotePairingAttempt.Event>) {
        let a = RemotePairingAttempt(to: Self.loopback(port), name: "Neo")
        let events = Events<RemotePairingAttempt.Event>()
        a.start { events.add($0) }
        return (a, events)
    }

    func testTwoMacsPairOverTheNetwork() throws {
        let (h, hostEvents) = host()
        let (_, neo) = attempt(try pairingPort(h, hostEvents))
        let code = try XCTUnwrap(neo.next { if case .compare(let c) = $0 { c } else { nil } })
        let shown = try XCTUnwrap(hostEvents.next { if case .pairing(.compare(let n, let c)) = $0 { (n, c) } else { nil } })
        XCTAssertEqual(shown.0, "Neo")
        XCTAssertEqual(shown.1, code)
        h.acceptPairing()
        let key = try XCTUnwrap(neo.next { if case .paired(let k) = $0 { k } else { nil } })
        let filed = try XCTUnwrap(hostEvents.next { if case .pairing(.paired(let k, _)) = $0 { k } else { nil } })
        XCTAssertEqual(key, filed)
    }

    /// The window lets one Mac in: the second finds it closed, and the first's pairing goes on.
    func testASecondMacDuringAPairingIsTurnedAway() throws {
        let (h, hostEvents) = host()
        let port = try pairingPort(h, hostEvents)
        let (_, first) = attempt(port)
        XCTAssertNotNil(first.next { if case .compare(let c) = $0 { c } else { nil } })
        let (_, intruder) = attempt(port)
        XCTAssertNotNil(intruder.next { e -> Bool? in if case .ended(.unreachable) = e { true } else { nil } })
        h.acceptPairing()
        XCTAssertNotNil(first.next { if case .paired(let k) = $0 { k } else { nil } })
    }

    func testARefusalOnTheServerEndsTheClientsAttempt() throws {
        let (h, hostEvents) = host()
        let (_, neo) = attempt(try pairingPort(h, hostEvents))
        XCTAssertNotNil(neo.next { if case .compare(let c) = $0 { c } else { nil } })
        h.refusePairing()
        XCTAssertNotNil(neo.next { e -> Bool? in if case .ended(.refused) = e { true } else { nil } })
        XCTAssertNotNil(hostEvents.next { e -> Bool? in if case .pairing(.ended(.refused)) = e { true } else { nil } })
    }

    func testAClosedWindowIsUnreachable() throws {
        let (h, hostEvents) = host()
        let port = try pairingPort(h, hostEvents)
        h.closePairing()
        XCTAssertNotNil(hostEvents.next { e -> Bool? in if case .pairing(.ended(.closed)) = e { true } else { nil } })
        let (_, neo) = attempt(port)
        XCTAssertNotNil(neo.next { e -> Bool? in if case .ended(.unreachable) = e { true } else { nil } })
    }

    // MARK: - Sessions

    /// The listener's port, read again after each restart: what Bonjour would follow.
    final class Port: @unchecked Sendable {
        private let lock = NSLock()
        private var _value: UInt16?
        var value: UInt16? { get { lock.withLock { _value } } set { lock.withLock { _value = newValue } } }
    }

    private func lending(_ h: RemoteHost, _ events: Events<RemoteHost.Event>, _ keys: [PairedKey], into port: Port) throws {
        h.lend(keys: keys)
        port.value = try XCTUnwrap(events.next { if case .lending(let p?) = $0 { p } else { nil } })
    }

    private func link(_ key: PairedKey, _ port: Port, version: String = "0.17 (170)") -> (RemoteLink, Events<RemoteLink.State>) {
        let l = RemoteLink(key: key, hello: hello("Neo", version: version)) { port.value.map(Self.loopback) }
        let states = Events<RemoteLink.State>()
        l.connect { states.add($0) }
        addTeardownBlock { l.disconnect() }
        return (l, states)
    }

    func testAPairedMacConnectsAndBothSidesKnowIt() throws {
        let (h, hostEvents) = host()
        let key = randomKey(), port = Port()
        try lending(h, hostEvents, [randomKey(), key, randomKey()], into: port)
        let (_, neo) = link(key, port)
        let theirs = try XCTUnwrap(neo.next { if case .connected(let x) = $0 { x } else { nil } })
        XCTAssertEqual(theirs.name, "Mac mini")
        let came = try XCTUnwrap(hostEvents.next { if case .connected(let i, let x) = $0 { (i, x) } else { nil } })
        XCTAssertEqual(came.0, key.identity, "the server knows which Mac came in, from its proof")
        XCTAssertEqual(came.1.name, "Neo")
    }

    /// An identity with a zero byte, as one in sixteen random ones has: refused by TLS if sent raw.
    func testAnIdentityWithAZeroByteConnects() throws {
        let (h, hostEvents) = host()
        var identity = Pairing.random(Pairing.identitySize)
        identity[identity.startIndex + 3] = 0
        let key = PairedKey(identity: identity, key: SymmetricKey(size: .bits256)), port = Port()
        try lending(h, hostEvents, [key], into: port)
        let (_, neo) = link(key, port)
        XCTAssertNotNil(neo.next { if case .connected(let x) = $0 { x } else { nil } })
    }

    func testAWrongKeyIsRefused() throws {
        let (h, hostEvents) = host()
        let key = randomKey(), port = Port()
        try lending(h, hostEvents, [key], into: port)
        let (_, neo) = link(PairedKey(identity: key.identity, key: SymmetricKey(size: .bits256)), port)
        XCTAssertEqual(neo.next { e -> RemoteLink.Refusal? in if case .refused(let r) = e { r } else { nil } }, .forgotten)
    }

    func testAnIntruderWithoutAKeyIsRefused() throws {
        let (h, hostEvents) = host()
        let port = Port()
        try lending(h, hostEvents, [randomKey()], into: port)
        let (_, intruder) = link(randomKey(), port)
        XCTAssertEqual(intruder.next { e -> RemoteLink.Refusal? in if case .refused(let r) = e { r } else { nil } }, .forgotten)
    }

    /// One paired Mac, inside its own TLS session, cannot name another's identity: its proof fails.
    func testAPairedMacCannotPassForAnother() {
        let a = randomKey(), b = randomKey(), challenge = Pairing.random(Pairing.nonceSize)
        XCTAssertTrue(SessionMessage.checks(SessionMessage.proof(a, challenge: challenge), a, challenge: challenge))
        XCTAssertFalse(SessionMessage.checks(SessionMessage.proof(b, challenge: challenge), a, challenge: challenge))
        XCTAssertFalse(SessionMessage.checks(SessionMessage.proof(a, challenge: challenge), a,
                                             challenge: Pairing.random(Pairing.nonceSize)))
    }

    func testAnotherBuildIsRefusedAndNotRetried() throws {
        let (h, hostEvents) = host("0.17 (171)")
        let key = randomKey(), port = Port()
        try lending(h, hostEvents, [key], into: port)
        let (_, neo) = link(key, port)
        XCTAssertEqual(neo.next { e -> RemoteLink.Refusal? in if case .refused(let r) = e { r } else { nil } },
                       .mismatch(.appVersion(mine: "0.17 (171)", theirs: "0.17 (170)")))
        XCTAssertNotNil(hostEvents.next { e -> Bool? in if case .turnedAway("Neo", .appVersion?) = e { true } else { nil } })
        XCTAssertNil(neo.next(1.5) { e -> Bool? in if case .connecting = e { true } else { nil } }, "a refusal is not retried")
    }

    /// The server stops (its app quits) and comes back on another port: the session is found again.
    func testASessionIsFoundAgainAfterACut() throws {
        let (h, hostEvents) = host()
        let key = randomKey(), port = Port()
        try lending(h, hostEvents, [key], into: port)
        let (_, neo) = link(key, port)
        XCTAssertNotNil(neo.next { if case .connected(let x) = $0 { x } else { nil } })
        h.stop()
        XCTAssertNotNil(neo.next { e -> Bool? in if case .waiting = e { true } else { nil } })
        try lending(h, hostEvents, [key], into: port)
        XCTAssertNotNil(neo.next(10) { if case .connected(let x) = $0 { x } else { nil } })
    }

    /// Forgotten on the server, the Mac is closed out and told so — not retried for ever.
    func testAForgottenMacIsShutOut() throws {
        let (h, hostEvents) = host()
        let key = randomKey(), other = randomKey(), port = Port()
        try lending(h, hostEvents, [key, other], into: port)
        let (_, neo) = link(key, port)
        XCTAssertNotNil(neo.next { if case .connected(let x) = $0 { x } else { nil } })
        try lending(h, hostEvents, [other], into: port)
        XCTAssertEqual(neo.next(10) { e -> RemoteLink.Refusal? in if case .refused(let r) = e { r } else { nil } }, .forgotten)
    }

    // MARK: - The mirror's messages

    /// Once joined, a request goes up and an image comes down — its message, then its PNG as a blob.
    func testAJoinedSessionCarriesMessagesAndBlobsBothWays() throws {
        let (h, hostEvents) = host()
        let key = randomKey(), port = Port()
        try lending(h, hostEvents, [key], into: port)
        let l = RemoteLink(key: key, hello: hello("Neo")) { port.value.map(Self.loopback) }
        let received = Events<SessionMessage>(), blobs = Events<Data>()
        l.listen(messages: { received.add($0) }, blobs: { _, d in blobs.add(d) })
        let states = Events<RemoteLink.State>()
        l.connect { states.add($0) }
        addTeardownBlock { l.disconnect() }
        XCTAssertNotNil(states.next { if case .connected(let x) = $0 { x } else { nil } })
        let identity = try XCTUnwrap(hostEvents.next { if case .connected(let i, _) = $0 { i } else { nil } })

        l.send(.cancel)
        XCTAssertNotNil(hostEvents.next { e -> Bool? in if case .message(identity, .cancel) = e { true } else { nil } })
        let png = Data((0..<200_000).map { UInt8($0 & 0xff) })
        h.send(.preview(7), to: identity)
        h.send(blob: png, id: 7, to: identity)
        XCTAssertEqual(received.next { e -> UInt32? in if case .preview(let id?) = e { id } else { nil } }, UInt32(7))
        XCTAssertEqual(blobs.next { d -> Data? in d }, png)
    }

    func testTheMirrorsMessagesTravelAsJSON() throws {
        let job = RemoteJob(ordinal: 3, model: "z-image", modelName: "Z-Image Turbo", prompt: "woman posing in a library",
                            width: 1024, height: 1024, steps: 8, defaultSteps: 8,
                            loras: [RemoteLoRA(path: "/l/a.silicon", strength: 0.8)], seeds: [42, 43], detail: .normal,
                            solver: .euler, schedule: .official, previews: true, neuralEngine: nil)
        var edit = job
        edit.references = [RemoteReference(origin: .history(7), name: "i7", width: 1024, height: 1024),
                           RemoteReference(origin: .file(blob: 3), name: "a.heic", width: 3024, height: 4032),
                           RemoteReference(origin: .image(blob: 4), name: "pasted", width: 640, height: 480),
                           RemoteReference(origin: .ofImage(9, index: 1), name: "b.png", width: 512, height: 768)]
        edit.variations = [Variation(seed: 7, strength: 0.25), Variation(seed: 8, strength: 0.5)]
        let quote = RemoteQuote(model: "qwen-image-2.1", width: 1024, height: 1024, first: [3024, 4032], images: 2)
        var q = RemoteQueue()
        q.current = job; q.fraction = 0.5; q.stage = "denoising"; q.remaining = 41.5
        var t = Engine.Timings(); t.total = 83.1
        let messages: [SessionMessage] = [
            .render(job), .render(edit), .quote(quote), .quoted(RemoteQuoteAnswer(quote: quote, width: 896, height: 1184, need: 3 << 30)),
            .enlarge(7, download: true), .crop(7, Crop.Box(x: 64, y: 0, width: 768, height: 1024)), .enlargeNeedsDownload(7),
            .queue(q), .cancel, .removeJob(4), .deleteImages([1, 2]), .refused("No."), .preview(nil),
            .entry(RemoteEntry(ordinal: 7, job: job, width: 1024, height: 1024, seed: 42, timings: t,
                               footprints: Engine.Footprints(), evaluations: 8, metadata: ["parameters": "x"],
                               neuralEngine: false, png: 9)),
            .catalog(MirrorCatalog(ready: ["z-image"], traits: ["z-image": RemoteTraits(
                maxReferences: 0, readsImages: false, acceptsDetail: true, comfyUI: false, steps: Array(1...50),
                text: "Qwen3", encoding: nil, decoding: "Flux VAE")], loras: [:], loraNames: [:],
                ceiling: 10 << 30, settingsRead: true, neuralEngineOffered: true, fasterRendering: true)),
        ]
        for m in messages {
            XCTAssertEqual(try JSONDecoder().decode(SessionMessage.self, from: JSONEncoder().encode(m)), m)
        }
    }

    // MARK: - What is filed

    func testPairingsKeepTheirKeysAndReplaceAMacPairedAgain() throws {
        var p = Pairings(id: "mini")
        let k1 = randomKey(), k2 = randomKey()
        p.lend(to: "Neo", k1)
        p.borrow(from: "studio", name: "Mac Studio", k1)
        p.borrow(from: "studio", name: "Mac Studio", k2)
        XCTAssertEqual(p.lendingKeys, [k1])
        XCTAssertEqual(p.borrowed.count, 1)
        XCTAssertEqual(p.key(of: p.borrowed[0]), k2)
        XCTAssertEqual(try JSONDecoder().decode(Pairings.self, from: JSONEncoder().encode(p)), p)
        XCTAssertEqual(Pairings.account(environment: [:]), "pairings")
        XCTAssertEqual(Pairings.account(environment: ["SILICONED_ROOT": "/tmp/b"]), "pairings /tmp/b")
    }
}
