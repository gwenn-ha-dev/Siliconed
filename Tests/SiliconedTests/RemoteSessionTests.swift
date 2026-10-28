import CryptoKit
import Foundation
import Network
import XCTest
@testable import Siliconed

/// **The remote Mac's sessions when they are not clean**: the same Mac joining twice, a lender that closes
/// its own sessions, a listener that dies, a join that says what the borrower already holds — on the
/// loopback, without Bonjour — and the pure rules that keep a peer's numbers from trapping.
final class RemoteSessionTests: XCTestCase {
    typealias Events = RemoteNetworkTests.Events

    private func hello(_ name: String, launch: UInt64? = nil) -> Hello {
        Hello(name: name, machine: Diagnostic.Machine(chip: "Apple M4", hardwareModel: "Mac16,10", performanceCores: 4,
                                                      efficiencyCores: 6, gpuCores: 10, memoryBytes: 16 << 30,
                                                      macOS: "26.1", version: "0.17 (170)", build: "engine abc"),
              launch: launch)
    }

    private func host(launch: UInt64? = nil) -> (RemoteHost, Events<RemoteHost.Event>) {
        let h = RemoteHost(hello: hello("Mac mini", launch: launch), service: nil, loopbackOnly: true)
        let events = Events<RemoteHost.Event>()
        h.start { events.add($0) }
        addTeardownBlock { h.stop() }
        return (h, events)
    }

    private final class Port: @unchecked Sendable {
        private let lock = NSLock()
        private var _value: UInt16?
        var value: UInt16? { get { lock.withLock { _value } } set { lock.withLock { _value = newValue } } }
    }

    private static func loopback(_ port: UInt16) -> NWEndpoint { .hostPort(host: .ipv4(.loopback), port: NWEndpoint.Port(rawValue: port)!) }

    private func randomKey() -> PairedKey {
        PairedKey(identity: Pairing.random(Pairing.identitySize), key: SymmetricKey(size: .bits256))
    }

    private func lending(_ h: RemoteHost, _ events: Events<RemoteHost.Event>, _ keys: [PairedKey], into port: Port) throws {
        h.lend(keys: keys)
        port.value = try XCTUnwrap(events.next { if case .lending(let p?) = $0 { p } else { nil } })
    }

    private func link(_ key: PairedKey, _ port: Port) -> (RemoteLink, Events<RemoteLink.State>) {
        let l = RemoteLink(key: key, hello: hello("Neo")) { port.value.map(Self.loopback) }
        let states = Events<RemoteLink.State>()
        l.connect { states.add($0) }
        addTeardownBlock { l.disconnect() }
        return (l, states)
    }

    // MARK: - Two sessions of one Mac

    /// The Mac comes back before its old session is known dead: the new one replaces it, and the old one's
    /// end must not tell the lender's owner that the Mac left — it is there, on the new session.
    func testAnOldSessionClosingDoesNotSayTheMacLeft() throws {
        let (h, events) = host()
        let key = randomKey(), port = Port()
        try lending(h, events, [key], into: port)
        let (a, aStates) = link(key, port)
        XCTAssertNotNil(aStates.next { if case .connected = $0 { true } else { nil } })
        XCTAssertNotNil(events.next { if case .connected = $0 { true } else { nil } })
        let (b, bStates) = link(key, port)
        XCTAssertNotNil(bStates.next { if case .connected = $0 { true } else { nil } })
        XCTAssertNotNil(events.next { if case .connected = $0 { true } else { nil } })
        // The old session is shut; its link would retry in a second and take the place back.
        XCTAssertNotNil(aStates.next { if case .waiting = $0 { true } else { nil } })
        a.disconnect()
        XCTAssertNil(events.next(0.8) { if case .disconnected = $0 { true } else { nil } },
                     "the old session's end is not the Mac leaving")
        b.disconnect()
        XCTAssertNotNil(events.next { if case .disconnected = $0 { true } else { nil } }, "the Mac that is there leaves once")
    }

    // MARK: - A lender that closes its own sessions

    /// A key taken off the list closes that Mac's session — and the owner is told it left, as it is for
    /// any other end: what the Mac had under way (an import, a question) is dropped with it.
    func testForgettingAMacTellsTheOwnerItLeft() throws {
        let (h, events) = host()
        let key = randomKey(), other = randomKey(), port = Port()
        try lending(h, events, [key, other], into: port)
        let (_, neo) = link(key, port)
        XCTAssertNotNil(neo.next { if case .connected = $0 { true } else { nil } })
        let identity = try XCTUnwrap(events.next { if case .connected(let i, _) = $0 { i } else { nil } })
        h.lend(keys: [other])
        XCTAssertEqual(events.next { if case .disconnected(let i) = $0 { i } else { nil } }, identity)
    }

    func testStoppingLendingTellsTheOwnerEveryMacLeft() throws {
        let (h, events) = host()
        let key = randomKey(), port = Port()
        try lending(h, events, [key], into: port)
        let (_, neo) = link(key, port)
        XCTAssertNotNil(neo.next { if case .connected = $0 { true } else { nil } })
        let identity = try XCTUnwrap(events.next { if case .connected(let i, _) = $0 { i } else { nil } })
        h.stop()
        XCTAssertEqual(events.next { if case .disconnected(let i) = $0 { i } else { nil } }, identity)
    }

    // MARK: - A listener that dies

    func testTheDelaysBeforeListeningAgainGrowAndStop() {
        XCTAssertEqual((0..<8).map { RemoteHost.relistenDelay(afterFailures: $0) }, [1, 2, 5, 10, 30, 30, 30, 30])
        XCTAssertEqual(RemoteHost.relistenDelay(afterFailures: -3), 1)
    }

    /// The network layer says the listener failed (the Wi-Fi went away, the port was taken): lending is said
    /// to be off, then listens again by itself, and a Mac gets in.
    func testALenderListensAgainAfterTheListenerFails() throws {
        let (h, events) = host()
        let key = randomKey(), port = Port()
        try lending(h, events, [key], into: port)
        h.queue.async { h.listenerFailed("the network went away") }
        XCTAssertNotNil(events.next { e -> Bool? in if case .lendingFailed = e { true } else { nil } })
        port.value = try XCTUnwrap(events.next(10) { if case .lending(let p?) = $0 { p } else { nil } },
                                   "it listens again without being asked")
        let (_, neo) = link(key, port)
        XCTAssertNotNil(neo.next { if case .connected = $0 { true } else { nil } })
    }

    func testAListenerThatFailedWithNoKeysLeftStaysOff() throws {
        let (h, events) = host()
        let key = randomKey(), port = Port()
        try lending(h, events, [key], into: port)
        h.queue.async { h.listenerFailed("x") }
        h.lend(keys: [])
        XCTAssertNil(events.next(1.6) { if case .lending(let p?) = $0 { p } else { nil } })
    }

    // MARK: - A refusal is concluded at the second

    /// One handshake that fails may be the network: the Mac waits and tries again; the second says the key is gone.
    func testForgottenIsConcludedAtTheSecondRefusalOnly() throws {
        let (h, events) = host()
        let key = randomKey(), port = Port()
        try lending(h, events, [key], into: port)
        let (_, intruder) = link(randomKey(), port)
        XCTAssertNotNil(intruder.next { e -> Bool? in if case .waiting = e { true } else { nil } }, "the first refusal is waited out")
        XCTAssertEqual(intruder.next(10) { e -> RemoteLink.Refusal? in if case .refused(let r) = e { r } else { nil } }, .forgotten)
    }

    // MARK: - What a join says it already holds

    func testAJoinNamesTheImagesHeldAfterAWelcomeSaidTheLaunch() throws {
        let (h, events) = host(launch: 77)
        let key = randomKey(), port = Port()
        try lending(h, events, [key], into: port)
        let l = RemoteLink(key: key, hello: hello("Neo")) { port.value.map(Self.loopback) }
        l.resume { [3, 1, 2] }
        let states = Events<RemoteLink.State>()
        l.connect { states.add($0) }
        addTeardownBlock { l.disconnect() }
        let welcome = try XCTUnwrap(states.next { if case .connected(let x) = $0 { x } else { nil } })
        XCTAssertEqual(welcome.launch, 77)
        let first = try XCTUnwrap(events.next { if case .connected(_, let x) = $0 { x } else { nil } })
        XCTAssertNil(first.held, "nothing is known of the lender yet: everything is sent")
        // The lender goes away and comes back.
        h.stop()
        XCTAssertNotNil(states.next { e -> Bool? in if case .waiting = e { true } else { nil } })
        try lending(h, events, [key], into: port)
        let second = try XCTUnwrap(events.next(10) { if case .connected(_, let x) = $0 { x } else { nil } })
        XCTAssertEqual(second.held, Hello.Held(launch: 77, entries: [3, 1, 2]))
    }

    func testTheImagesAlreadyHeldAreTrustedForTheSameLaunchAndOnlyThoseTheLenderHas() {
        var theirs = hello("Neo")
        theirs.held = Hello.Held(launch: 5, entries: [1, 2, 3, 9])
        XCTAssertEqual(RemoteResume.alreadyHeld(by: theirs, launch: 5, lenderHas: [2, 3, 4]), [2, 3])
        XCTAssertEqual(RemoteResume.alreadyHeld(by: theirs, launch: 6, lenderHas: [2, 3, 4]), [],
                       "another launch numbered them: they may be other images")
        XCTAssertEqual(RemoteResume.alreadyHeld(by: theirs, launch: nil, lenderHas: [2, 3, 4]), [])
        XCTAssertEqual(RemoteResume.alreadyHeld(by: hello("Neo"), launch: 5, lenderHas: [2]), [])
    }

    func testAHelloFromBeforeLaunchAndHeldStillReads() throws {
        var h = hello("Neo", launch: 9)
        h.held = Hello.Held(launch: 4, entries: [1, 2])
        XCTAssertEqual(try JSONDecoder().decode(Hello.self, from: JSONEncoder().encode(h)), h)
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(h)) as? [String: Any])
        object["launch"] = nil; object["held"] = nil
        let old = try JSONDecoder().decode(Hello.self, from: JSONSerialization.data(withJSONObject: object))
        XCTAssertNil(old.launch)
        XCTAssertNil(old.held)
        // The join of such a Hello is a join still, so that a build mismatch is said, not met with silence.
        let join = SessionMessage.join(hello: old, identity: Data([1]), proof: Data([2]))
        XCTAssertEqual(try JSONDecoder().decode(SessionMessage.self, from: JSONEncoder().encode(join)), join)
    }

    func testHeldImagesAreReadableFromAnotherThread() {
        let held = HeldImages()
        XCTAssertEqual(held.sorted, [])
        held.set([5, 1, 3])
        XCTAssertEqual(held.sorted, [1, 3, 5])
        held.set([])
        XCTAssertEqual(held.sorted, [])
    }

    // MARK: - Numbers from another Mac

    func testAQuoteWithNumbersThatWouldTrapIsNotSane() {
        func quote(_ w: Int = 1024, _ h: Int = 1024, first: [Int]? = nil, images: Int = 0) -> RemoteQuote {
            RemoteQuote(model: "z-image", width: w, height: h, first: first, images: images)
        }
        XCTAssertTrue(quote().isSane)
        XCTAssertTrue(quote(first: [3024, 4032], images: 3).isSane)
        XCTAssertFalse(quote(Int.max, 1024).isSane)
        XCTAssertFalse(quote(1024, -1).isSane)
        XCTAssertFalse(quote(0, 1024).isSane)
        XCTAssertFalse(quote(Format.maximumSide + 1, 1024).isSane)
        XCTAssertFalse(quote(images: -1).isSane)
        XCTAssertFalse(quote(images: Int.max).isSane)
        XCTAssertFalse(quote(first: [Int.max, 1024]).isSane)
        XCTAssertFalse(quote(first: [1024]).isSane)
        XCTAssertFalse(quote(first: [1024, 0]).isSane)
        XCTAssertFalse(quote(first: [1, 2, 3]).isSane)
    }

    func testACropBoxFromAnotherMacSaturatesInsteadOfTrapping() {
        XCTAssertEqual(Crop.Box(x: 10, y: 20, width: 30, height: 40).maxX, 40)
        XCTAssertEqual(Crop.Box(x: 10, y: 20, width: 30, height: 40).maxY, 60)
        let huge = Crop.Box(x: Int.max, y: Int.max, width: Int.max, height: 1)
        XCTAssertEqual(huge.maxX, Int.max)
        XCTAssertEqual(huge.maxY, Int.max)
        XCTAssertEqual(Crop.Box(x: Int.min, y: 0, width: Int.min, height: 1).maxX, Int.min)
        // …so the lender's own bounds check refuses it.
        XCTAssertFalse(huge.maxX <= 1024)
    }

    func testAnUploadOfferWithASizeThatOverflowsAsksForMoreThanAnyVolumeHas() throws {
        XCTAssertEqual(UploadStore.needed(for: 1 << 30), 3 << 30)
        XCTAssertEqual(UploadStore.needed(for: Int64.max), Int64.max)
        XCTAssertEqual(UploadStore.needed(for: Int64.max / 2), Int64.max)
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("upload-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        let inbox = UploadInbox(directory: folder, extensions: ["safetensors"], available: { _ in 100 << 30 })
        let refusal = inbox.begin(identity: Data([1]), UploadOffer(id: 1, name: "a.safetensors", size: Int64.max))
        guard case .noSpace? = refusal else { return XCTFail("\(String(describing: refusal))") }
        XCTAssertFalse(inbox.isActive)
    }

    // MARK: - The name of the other Mac, and what its install would remove

    func testANameFromAnotherMacIsFitToBeShown() {
        XCTAssertEqual(Pairing.displayName("Mac mini de Gwenn"), "Mac mini de Gwenn")
        XCTAssertEqual(Pairing.displayName("  Neo\n\tof Ana\u{0}  "), "Neoof Ana")
        XCTAssertEqual(Pairing.displayName("Neo\u{2028}Accept now"), "NeoAccept now")
        XCTAssertEqual(Pairing.displayName(String(repeating: "é", count: 500)).count, Pairing.nameLimit)
        XCTAssertEqual(Pairing.displayName(""), "Mac")
        XCTAssertEqual(Pairing.displayName("\n\n"), "Mac")
    }

    func testThePairingServerShowsTheBoundedName() throws {
        let client = PairingClient()
        var server = PairingServer()
        _ = try server.receive(client.start(name: String(repeating: "x", count: 10_000) + "\nAccept"))
        XCTAssertEqual(server.clientName?.count, Pairing.nameLimit)
    }

    func testAnInstallationThatTakesAVersionOffTheDiskAsksTheLendersUser() {
        func plan(replacing: Variant?, removesFirst: Bool) -> RemotePlan {
            RemotePlan(variant: .compact, baseDiT: true, replacing: replacing, added: 1, freed: replacing == nil ? 0 : 1, peak: 2,
                       removesFirst: removesFirst, fits: true)
        }
        XCTAssertFalse(plan(replacing: nil, removesFirst: false).deletes)
        XCTAssertTrue(plan(replacing: .standard, removesFirst: false).deletes)
        XCTAssertTrue(plan(replacing: .standard, removesFirst: true).deletes)
        XCTAssertTrue(plan(replacing: nil, removesFirst: true).deletes)
    }

    func testTheQueueFullRefusalCrossesAsACode() throws {
        let m = SessionMessage.answer(request: 3, .refused(.app(.queueFull)))
        XCTAssertEqual(try JSONDecoder().decode(SessionMessage.self, from: JSONEncoder().encode(m)), m)
    }
}
