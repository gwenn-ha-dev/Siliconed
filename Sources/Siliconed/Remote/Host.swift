import Foundation
import Network

/// **The Mac that lends itself** (the mini): its two listeners, the sessions of the Macs it paired, and
/// the pairing window. One serial queue for all; events go to the owner on it.
///
///     lend(keys:)      the session listener, one TLS-PSK key per paired Mac; restarted when the list
///                      changes (a new pairing, a Mac forgotten) — a forgotten Mac's session is closed
///     openPairing()    the pairing listener, plain TCP, for one attempt: the first Mac that comes in
///                      closes it to the others; the window ends with that attempt, or after `pairingTimeout`
///     accept/refuse    the user's answer, once both screens show the code
///
/// Nothing is filed here: a pairing ends with `.paired`, and the owner stores the key (`Pairings`)
/// and calls `lend(keys:)` again.
package final class RemoteHost: @unchecked Sendable {
    package enum Event: Sendable {
        /// The session listener is ready (its port, for a test without Bonjour), or stopped (`nil`).
        case lending(port: UInt16?)
        /// The session listener could not start or failed: lending is off.
        case lendingFailed(String)
        case pairing(PairingWindow)
        /// A paired Mac proved its key and runs the same build.
        case connected(identity: Data, hello: Hello)
        case disconnected(identity: Data)
        /// A Mac that holds a key was turned away: another build (`mismatch`), or no longer paired.
        case turnedAway(name: String, Hello.Mismatch?)
    }

    package enum PairingWindow: Equatable, Sendable {
        case waiting(port: UInt16)
        /// Both screens show `code`: the user compares and answers (`acceptPairing`, `refusePairing`).
        case compare(name: String, code: String)
        case paired(PairedKey, name: String)
        case ended(PairingEnd)
    }

    package enum PairingEnd: Equatable, Sendable {
        case closed, timedOut, refused, failed(String)
    }

    package static let pairingTimeout: TimeInterval = 120

    private let queue = DispatchQueue(label: "siliconed.remote.host")
    private let hello: Hello
    /// The Bonjour name and TXT; `nil`: not advertised (the tests reach the port directly).
    private let service: (name: String, txt: [String: String])?
    private let loopbackOnly: Bool
    private var handler: (@Sendable (Event) -> Void)?

    private var keys: [PairedKey] = []
    private var listener: NWListener?
    private var sessions: [ObjectIdentifier: Session] = [:]

    private var pairingListener: NWListener?
    private var pairingChannel: RemoteChannel?
    private var pairingServer: PairingServer?
    private var pairingClosed = true
    private var pairingDeadline: DispatchWorkItem?

    private struct Session {
        let channel: RemoteChannel
        let challenge: Data
        var identity: Data?
    }

    /// - `service`: advertised under `RemoteTransport.sessionService` and `pairingService`.
    /// - `loopbackOnly`: listens on 127.0.0.1, for the tests.
    package init(hello: Hello, service: (name: String, txt: [String: String])?, loopbackOnly: Bool = false) {
        self.hello = hello; self.service = service; self.loopbackOnly = loopbackOnly
    }

    package func start(_ handler: @escaping @Sendable (Event) -> Void) {
        queue.async { self.handler = handler }
    }

    // MARK: - Lending

    /// Listens with these keys; none stops lending. A session whose key is gone is closed.
    package func lend(keys new: [PairedKey]) {
        queue.async { [self] in
            let same = new == keys && listener != nil
            keys = new
            for (id, s) in sessions where s.identity.map({ i in !new.contains { $0.identity == i } }) ?? false {
                s.channel.close()
                sessions[id] = nil
            }
            guard !same else { return }
            listener?.cancel()
            listener = nil
            guard !new.isEmpty else { handler?(.lending(port: nil)); return }
            do {
                let l = try makeListener(RemoteTransport.server(keys: new), type: RemoteTransport.sessionService)
                listener = l
                l.newConnectionHandler = { [weak self] c in self?.admit(c) }
                l.stateUpdateHandler = { [weak self, weak l] state in
                    guard let self, let l, l === listener else { return }
                    switch state {
                    case .ready: handler?(.lending(port: l.port?.rawValue))
                    case .failed(let e):
                        listener = nil
                        handler?(.lendingFailed(e.debugDescription))
                    default: break
                    }
                }
                l.start(queue: queue)
            } catch {
                handler?(.lendingFailed("\(error)"))
            }
        }
    }

    /// Stops both listeners and closes every session.
    package func stop() {
        queue.async { [self] in
            listener?.cancel(); listener = nil
            keys = []
            for s in sessions.values { s.channel.close() }
            sessions = [:]
            endPairing(.closed)
            handler?(.lending(port: nil))
        }
    }

    private func makeListener(_ p: NWParameters, type: String) throws -> NWListener {
        if loopbackOnly { p.requiredLocalEndpoint = .hostPort(host: .ipv4(.loopback), port: .any) }
        let l = try NWListener(using: p)
        if let service {
            l.service = NWListener.Service(name: service.name, type: type, txtRecord: NWTXTRecord(service.txt))
        }
        return l
    }

    private func admit(_ c: NWConnection) {
        let channel = RemoteChannel(c, queue: queue)
        let id = ObjectIdentifier(channel)
        let challenge = Pairing.random(Pairing.nonceSize)
        sessions[id] = Session(channel: channel, challenge: challenge)
        channel.start { [weak self] event in
            guard let self else { return }
            switch event {
            case .ready:
                channel.send(SessionMessage.challenge(challenge))
            case .message(let json):
                guard var s = sessions[id] else { return }
                guard s.identity == nil else { return } // lot 3: the render's messages
                guard case .join(let theirs, let identity, let proof)? = try? JSONDecoder().decode(SessionMessage.self, from: json),
                      let key = keys.first(where: { $0.identity == identity }),
                      SessionMessage.checks(proof, key, challenge: s.challenge) else {
                    channel.close()
                    return
                }
                if let m = hello.mismatch(with: theirs) {
                    handler?(.turnedAway(name: theirs.name, m))
                    channel.send(SessionMessage.mismatch(m))
                    channel.finish()
                    return
                }
                s.identity = identity
                sessions[id] = s
                channel.send(SessionMessage.welcome(hello))
                handler?(.connected(identity: identity, hello: theirs))
            case .blob:
                channel.close()
            case .closed:
                if let identity = sessions[id]?.identity { handler?(.disconnected(identity: identity)) }
                sessions[id] = nil
            }
        }
    }

    // MARK: - Pairing

    /// Opens the window for one attempt; one already open is closed first.
    package func openPairing() {
        queue.async { [self] in
            endPairing(nil)
            pairingClosed = false
            do {
                let l = try makeListener(RemoteTransport.pairing(), type: RemoteTransport.pairingService)
                pairingListener = l
                l.newConnectionHandler = { [weak self] c in self?.pair(c) }
                l.stateUpdateHandler = { [weak self, weak l] state in
                    guard let self, let l, l === pairingListener else { return }
                    switch state {
                    case .ready: l.port.map { handler?(.pairing(.waiting(port: $0.rawValue))) }
                    case .failed(let e): endPairing(.failed(e.debugDescription))
                    default: break
                    }
                }
                l.start(queue: queue)
                let deadline = DispatchWorkItem { [weak self] in self?.endPairing(.timedOut) }
                pairingDeadline = deadline
                queue.asyncAfter(deadline: .now() + Self.pairingTimeout, execute: deadline)
            } catch {
                endPairing(.failed("\(error)"))
            }
        }
    }

    package func closePairing() {
        queue.async { [self] in endPairing(.closed) }
    }

    /// The user saw the same code on both screens.
    package func acceptPairing() {
        queue.async { [self] in
            guard let channel = pairingChannel, var server = pairingServer,
                  let accepted = try? server.accept() else { return }
            pairingServer = server
            channel.send(accepted)
        }
    }

    package func refusePairing() {
        queue.async { [self] in
            guard let channel = pairingChannel, var server = pairingServer else { return }
            channel.send(server.refuse())
            channel.finish()
            pairingChannel = nil
            endPairing(.refused)
        }
    }

    /// The first Mac in: the listener closes behind it, and the window lives as long as its attempt.
    private func pair(_ c: NWConnection) {
        guard pairingChannel == nil, !pairingClosed else { c.cancel(); return }
        let channel = RemoteChannel(c, queue: queue)
        pairingChannel = channel
        pairingServer = PairingServer()
        channel.start { [weak self] event in
            guard let self, channel === pairingChannel else { return }
            switch event {
            case .ready:
                // Cancelled before, the listener would take this connection down with it.
                pairingListener?.cancel()
                pairingListener = nil
            case .blob:
                channel.close()
            case .message(let json):
                guard var server = pairingServer,
                      let m = try? JSONDecoder().decode(PairingMessage.self, from: json) else {
                    channel.close(); return
                }
                do throws(PairingFailure) {
                    let answer = try server.receive(m)
                    pairingServer = server
                    if let answer { channel.send(answer) }
                    if server.confirmed, let key = server.key {
                        channel.finish()
                        pairingChannel = nil
                        handler?(.pairing(.paired(key, name: server.clientName ?? "")))
                        endPairing(nil)
                    } else if case .reveal = answer, let code = server.code {
                        handler?(.pairing(.compare(name: server.clientName ?? "", code: code)))
                    }
                } catch {
                    channel.close()
                    pairingChannel = nil
                    endPairing(error == .refused ? .refused : .failed("\(error)"))
                }
            case .closed(let how):
                pairingChannel = nil
                endPairing(how == .closed ? .closed : .failed("\(how)"))
            }
        }
    }

    /// Ends the window; `nil` says nothing (a new window replaces it, or `.paired` was said).
    private func endPairing(_ how: PairingEnd?) {
        guard !pairingClosed else { return }
        pairingClosed = true
        pairingDeadline?.cancel(); pairingDeadline = nil
        pairingListener?.cancel(); pairingListener = nil
        pairingChannel?.close(); pairingChannel = nil
        pairingServer = nil
        if let how { handler?(.pairing(.ended(how))) }
    }
}
