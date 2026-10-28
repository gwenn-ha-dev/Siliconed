// **Other Macs of the network**: this Mac lent to the ones it paired, or computing on one that paired it.
// The channel only (lot 2 of `Remote/`): pairing on both sides, sessions kept up, Bonjour. What a
// session carries — the queue, the render, the history — comes with the switch of the whole app.
//
//     lending (the mini)                        borrowing (the Neo)
//       Settings › « Lend This Mac »              Settings › « Pair With a Mac… »
//         RemoteHost: _siliconed._tcp, the keys     browses _siliconed-pair._tcp
//       « Pair a Mac… » opens the window          picks the mini, shows the code
//         _siliconed-pair._tcp, one attempt         ⟶ same six digits on both screens
//       « Accept » (only here)                    the key filed under the mini's `id`
//
// - **Off by default**, and nothing listens before a pairing is accepted here: lending with no Mac
//   paired opens no port.
// - **The keys** are in the Keychain (`Pairings`), read on first need — the Settings window, or a launch
//   with lending on — not at every launch: a build signed ad hoc asks the user before reading them.
// - **This Mac's name** is the one of System Settings › General › Sharing; under `SILICONED_ROOT` (two
//   instances on one Mac, the test of the channel), the root's folder is added to tell them apart.

import AppKit
import Foundation
import Network
import Observation
import Siliconed
import SystemConfiguration

@MainActor @Observable
final class OtherMacs {
    /// What the Keychain holds; `nil` until read, or if the user denied it (`unreadable`).
    private(set) var pairings: Pairings?
    private(set) var unreadable = false

    /// « Lend This Mac », kept across launches — one per `SILICONED_ROOT`, as the Keychain item: two
    /// instances on one Mac share their preferences.
    private static let lendingKey = "lendThisMac" + (ProcessInfo.processInfo.environment["SILICONED_ROOT"].map { " " + $0 } ?? "")
    var lending: Bool = UserDefaults.standard.bool(forKey: OtherMacs.lendingKey) {
        didSet {
            UserDefaults.standard.set(lending, forKey: Self.lendingKey)
            applyLending()
        }
    }
    private(set) var lendingProblem: String?
    /// A Mac was paired to compute on: the Keychain is read at launch, so that the rack's « On: » menu
    /// and Bonjour are there from the start — whoever paired has already answered the Keychain.
    private static let borrowsKey = "borrowsAMac" + (ProcessInfo.processInfo.environment["SILICONED_ROOT"].map { " " + $0 } ?? "")
    /// The Macs lent to, connected now: identity → what they said.
    private(set) var connected: [Data: Hello] = [:]
    /// The server's pairing window, while its sheet is open.
    private(set) var window: RemoteHost.PairingWindow?

    /// The other Macs that lend themselves, as Bonjour shows them, by `id`.
    private(set) var found: [String: Found] = [:]
    /// Pairing windows open on the network, while the borrowing sheet is open.
    private(set) var pairable: [Found] = []
    /// What each Mac lent last said of its machine (`RemoteMachine.forMenu`), by `id`: shown by the « On: »
    /// menu and the Settings while it is not the one shown. Kept across launches.
    private(set) var machines: [String: RemoteMachine] = {
        UserDefaults.standard.data(forKey: OtherMacs.machinesKey).flatMap { try? JSONDecoder().decode([String: RemoteMachine].self, from: $0) } ?? [:]
    }()
    private static let machinesKey = "remoteMachines" + (ProcessInfo.processInfo.environment["SILICONED_ROOT"].map { " " + $0 } ?? "")
    /// The sessions wanted, by the other Mac's `id`.
    private(set) var links: [String: RemoteLink.State] = [:]
    /// The borrowing side's pairing, while its sheet is open.
    private(set) var attempt: Attempt?

    struct Found: Equatable, Identifiable {
        let id: String
        let name: String
        let version: String
        let endpoint: NWEndpoint
    }

    struct Attempt: Equatable {
        let with: Found
        var step: Step
        enum Step: Equatable {
            case connecting
            case compare(code: String)
            case paired
            case ended(RemotePairingAttempt.PairingEnd)
        }
    }

    let name: String
    /// Read on first need: the machine's description touches Metal and the executable.
    @ObservationIgnored private lazy var hello = Hello(name: name, machine: Diagnostic.machine(), launch: Hello.thisLaunch)
    private var host: RemoteHost?
    private var sessionBrowser: NWBrowser?
    private var pairingBrowser: NWBrowser?
    private var linkObjects: [String: RemoteLink] = [:]
    private var attemptObject: RemotePairingAttempt?
    private let book = Endpoints()
    private var awake: NSObjectProtocol?
    /// This Mac's own state, the one lent.
    private let local: AppState
    private var server: MirrorServer?
    /// The mirror shown, and the Mac it shows (`MainWindow.compute(on:)`).
    private(set) var mirrored: (id: String, app: AppState)?

    init(local: AppState) {
        self.local = local
        // Not `Host.current()`, which may wait on the DNS.
        var n = SCDynamicStoreCopyComputerName(nil, nil).map { $0 as String } ?? "Mac"
        if let root = ProcessInfo.processInfo.environment["SILICONED_ROOT"], !root.isEmpty {
            n += " (\(URL(fileURLWithPath: root).lastPathComponent))"
        }
        name = n
    }

    /// At launch: lends at once if it was on; reads the pairings if a Mac was paired to compute on.
    func boot() {
        if lending { applyLending() }
        if UserDefaults.standard.bool(forKey: Self.borrowsKey) { load() }
    }

    func stop() {
        host?.stop()
        linkObjects.values.forEach { $0.disconnect() }
        sessionBrowser?.cancel(); pairingBrowser?.cancel()
    }

    /// Reads the Keychain once.
    @discardableResult
    func load() -> Pairings? {
        if let pairings { return pairings }
        // Unreadable (the user not yet answered the Keychain's question, or said no): asked again at the
        // next need, never taken as « nothing paired » — a pairing written over it would erase the others.
        guard let p = Pairings.load() else { unreadable = true; return nil }
        unreadable = false
        pairings = p
        UserDefaults.standard.set(!p.borrowed.isEmpty, forKey: Self.borrowsKey)
        if !p.borrowed.isEmpty { browseSessions() }
        // Lending was on but the Keychain was not readable at launch: it starts now.
        if lending, host == nil { applyLending() }
        return p
    }

    private func save(_ change: (inout Pairings) -> Void) {
        guard var p = load() else {
            lendingProblem = String(localized: "The Keychain did not keep the pairing.")
            return
        }
        change(&p)
        pairings = p
        UserDefaults.standard.set(!p.borrowed.isEmpty, forKey: Self.borrowsKey)
        if !p.save() { lendingProblem = String(localized: "The Keychain did not keep the pairing.") }
    }

    private var txt: [String: String] {
        ["id": pairings?.id ?? "", "v": hello.machine.version]
    }

    // MARK: - Lending

    private func applyLending() {
        guard lending else {
            host?.stop()
            host = nil
            // The clients go with the server: what each had under way (an import, a question) ends with it.
            for identity in connected.keys { server?.left(identity) }
            server = nil
            connected = [:]
            window = nil
            lendingProblem = nil
            hold()
            return
        }
        guard let p = load() else { lendingProblem = String(localized: "The Keychain could not be read."); return }
        let h = host ?? makeHost()
        h.lend(keys: p.lendingKeys)
    }

    private func makeHost() -> RemoteHost {
        let h = RemoteHost(hello: hello, service: (name, txt))
        // Only the current host is heard: the events of one that was stopped (« Lend This Mac » off, then on)
        // are queued behind it, and would reach the new server as its own clients leaving.
        h.start { [weak self, weak h] e in
            DispatchQueue.main.async {
                MainActor.assumeIsolated { if let self, let h, self.host === h { self.handle(e) } }
            }
        }
        host = h
        server = MirrorServer(app: local, host: h) { [weak self] identity in
            self?.pairings?.lent.first { $0.identity == identity }?.name ?? self?.connected[identity]?.name
        }
        return h
    }

    private func handle(_ e: RemoteHost.Event) {
        switch e {
        case .lending:
            lendingProblem = nil
        case .lendingFailed(let why):
            lendingProblem = why
        case .pairing(let w):
            window = w
            if case .paired(let key, let client) = w {
                save { $0.lend(to: client, key) }
                pairings.map { host?.lend(keys: $0.lendingKeys) }
            }
        case .connected(let identity, let theirs):
            connected[identity] = theirs
            server?.joined(identity, theirs)
        case .disconnected(let identity):
            connected[identity] = nil
            server?.left(identity)
        case .message(let identity, let m):
            server?.handle(m, from: identity)
        case .blob(let identity, let id, let data):
            server?.received(blob: id, data, from: identity)
        case .turnedAway:
            break
        }
        hold()
    }

    /// The pairing window, from « Pair a Mac… »: one attempt, two minutes at most.
    func openPairing() {
        guard lending, load() != nil else { return }
        window = nil
        (host ?? makeHost()).openPairing()
    }

    func acceptPairing() { host?.acceptPairing() }
    func refusePairing() { host?.refusePairing() }

    func closePairing() {
        host?.closePairing()
        window = nil
    }

    /// Forgotten here: its session closes, and its key no longer opens one.
    func forget(lent: Pairings.Lent) {
        save { $0.lent.removeAll { $0.identity == lent.identity } }
        connected[lent.identity] = nil
        // The host closes that Mac's session and says nothing of it that this would not already have said.
        server?.left(lent.identity)
        if lending, let p = pairings { host?.lend(keys: p.lendingKeys) }
        hold()
    }

    /// While a Mac is connected, this one does not go to sleep on it, and App Nap does not slow it: a lender
    /// nobody sits at (its window hidden, its screen off) is the usual case, and would be napped.
    private func hold() {
        if connected.isEmpty {
            awake.map(ProcessInfo.processInfo.endActivity)
            awake = nil
        } else if awake == nil {
            awake = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .idleSystemSleepDisabled],
                                                          reason: "Another Mac computes on this one")
        }
    }

    // MARK: - Borrowing

    private func browseSessions() {
        guard sessionBrowser == nil else { return }
        sessionBrowser = browse(RemoteTransport.sessionService) { [weak self] results in
            guard let self else { return }
            found = Dictionary(results.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
            book.set(found.mapValues(\.endpoint))
            for (id, link) in linkObjects where found[id] != nil { link.retryNow() }
        }
    }

    /// The borrowing sheet opens: the pairing windows of the network are listed.
    func lookForPairable() {
        load()
        pairable = []
        pairingBrowser?.cancel()
        pairingBrowser = browse(RemoteTransport.pairingService) { [weak self] results in
            self?.pairable = results.sorted { $0.name < $1.name }
        }
    }

    func stopLookingForPairable() {
        pairingBrowser?.cancel(); pairingBrowser = nil
        attemptObject?.cancel(); attemptObject = nil
        attempt = nil
        pairable = []
    }

    private func browse(_ type: String, _ update: @escaping @MainActor ([Found]) -> Void) -> NWBrowser {
        let b = NWBrowser(for: .bonjourWithTXTRecord(type: type, domain: nil), using: NWParameters())
        let mine = pairings?.id
        b.browseResultsChangedHandler = { results, _ in
            let list = results.compactMap { r -> Found? in
                guard case .service(let n, _, _, _) = r.endpoint, case .bonjour(let t) = r.metadata,
                      let id = t["id"], !id.isEmpty, id != mine else { return nil }
                return Found(id: id, name: n, version: t["v"] ?? "", endpoint: r.endpoint)
            }
            DispatchQueue.main.async { MainActor.assumeIsolated { update(list) } }
        }
        b.start(queue: .main)
        return b
    }

    func pair(with mac: Found) {
        attemptObject?.cancel()
        attempt = Attempt(with: mac, step: .connecting)
        let a = RemotePairingAttempt(to: mac.endpoint, name: name)
        attemptObject = a
        a.start { [weak self] e in DispatchQueue.main.async { MainActor.assumeIsolated { self?.handle(e, from: mac, a) } } }
    }

    private func handle(_ e: RemotePairingAttempt.Event, from mac: Found, _ a: RemotePairingAttempt) {
        guard a === attemptObject else { return }
        switch e {
        case .compare(let code):
            attempt?.step = .compare(code: code)
        case .paired(let key):
            attempt?.step = .paired
            attemptObject = nil
            save { $0.borrow(from: mac.id, name: mac.name, key) }
            disconnect(mac.id)
            browseSessions()
        case .ended(let how):
            attempt?.step = .ended(how)
            attemptObject = nil
        }
    }

    /// Shows the other Mac in `mirror` once joined; `nil` stops showing it (the session stays if wanted).
    func show(_ b: Pairings.Borrowed?, in mirror: AppState?) {
        if let b, let mirror {
            mirrored = (b.id, mirror)
            // A session kept from Settings joined before this mirror existed, and the join's catalog went
            // nowhere: a new one, whose join is for this mirror.
            if linkObjects[b.id] != nil { disconnect(b.id) }
            connect(b, held: mirror.mirror?.held)
        } else {
            if let id = mirrored?.id { disconnect(id) }
            mirrored = nil
        }
    }

    func send(_ m: SessionMessage, to id: String) { linkObjects[id]?.send(m) }
    func send(blob data: Data, id: UInt32, to mac: String) { linkObjects[mac]?.send(blob: data, id: id) }
    func send(stream source: any StreamSource, id: UInt32, to mac: String) { linkObjects[mac]?.send(stream: source, id: id) }
    func cancel(stream id: UInt32, to mac: String) { linkObjects[mac]?.cancel(stream: id) }
    func isJoined(_ mac: String) -> Bool { if case .connected? = links[mac] { true } else { false } }

    func connect(_ b: Pairings.Borrowed, held: HeldImages? = nil) {
        guard let p = pairings, linkObjects[b.id] == nil else { return }
        browseSessions()
        let id = b.id, book = book
        let l = RemoteLink(key: p.key(of: b), hello: hello) { book.endpoint(id) }
        linkObjects[id] = l
        links[id] = .connecting
        if let held { l.resume { held.sorted } }
        l.listen(messages: { [weak self] m in
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard let s = self else { return }
                    if case .machine(let machine) = m { s.remember(machine, of: id) }
                    if s.mirrored?.id == id { s.mirrored?.app.receive(m) }
                }
            }
        }, blobs: { [weak self] blob, data in
            DispatchQueue.main.async {
                MainActor.assumeIsolated { if let s = self, s.mirrored?.id == id { s.mirrored?.app.receive(blob: blob, data) } }
            }
        })
        l.connect { [weak self] s in
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard let self, self.linkObjects[id] === l else { return }
                    self.links[id] = s
                    if case .refused = s { self.linkObjects[id] = nil }
                    if self.mirrored?.id == id { self.mirrored?.app.linkChanged(s, name: b.name) }
                }
            }
        }
    }

    private func remember(_ m: RemoteMachine, of id: String) {
        machines[id] = m.forMenu
        if let data = try? JSONEncoder().encode(machines) { UserDefaults.standard.set(data, forKey: Self.machinesKey) }
    }

    func disconnect(_ id: String) {
        linkObjects.removeValue(forKey: id)?.disconnect()
        links[id] = nil
    }

    func forget(borrowed b: Pairings.Borrowed) {
        disconnect(b.id)
        save { $0.borrowed.removeAll { $0.id == b.id } }
        machines[b.id] = nil
    }
}

/// Where Bonjour last saw each Mac, read by the links from their queues.
private final class Endpoints: @unchecked Sendable {
    private let lock = NSLock()
    private var byID: [String: NWEndpoint] = [:]
    func set(_ d: [String: NWEndpoint]) { lock.withLock { byID = d } }
    func endpoint(_ id: String) -> NWEndpoint? { lock.withLock { byID[id] } }
}
