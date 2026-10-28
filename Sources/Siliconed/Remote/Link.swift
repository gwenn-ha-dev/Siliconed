import Foundation
import Network

/// **The Mac that borrows** (the Neo): a session with one paired Mac, kept up while it is wanted, and
/// the pairing attempt that gives the key.
///
/// A session that drops is tried again — 1, 2, 4, then every 8 s, or at once when the owner sees the
/// Mac come back on Bonjour (`retryNow`) — but a refusal is not: a key the other Mac no longer holds,
/// or another build, will not change by waiting. The owner says it and the user decides.
package final class RemoteLink: @unchecked Sendable {
    package enum State: Equatable, Sendable {
        case connecting
        case connected(Hello)
        /// Lost, or not found on the network: retried after `retryIn` seconds.
        case waiting(reason: String, retryIn: TimeInterval)
        case refused(Refusal)
        case off
    }

    package enum Refusal: Equatable, Sendable {
        /// The other Mac does not hold this key any more: it forgot this one. Pair again.
        case forgotten
        case mismatch(Hello.Mismatch)
    }

    package static let delays: [TimeInterval] = [1, 2, 4, 8]

    private let queue = DispatchQueue(label: "siliconed.remote.link")
    private let key: PairedKey
    private let hello: Hello
    private let endpoint: @Sendable () -> NWEndpoint?
    private var handler: (@Sendable (State) -> Void)?
    private var wanted = false
    private var channel: RemoteChannel?
    private var failures = 0
    private var retry: DispatchWorkItem?
    private var joined = false
    private var messages: (@Sendable (SessionMessage) -> Void)?
    private var blobs: (@Sendable (UInt32, Data) -> Void)?

    /// `endpoint` is asked at each attempt: the Mac's Bonjour service as last seen, `nil` if not seen.
    package init(key: PairedKey, hello: Hello, endpoint: @escaping @Sendable () -> NWEndpoint?) {
        self.key = key; self.hello = hello; self.endpoint = endpoint
    }

    package func connect(_ handler: @escaping @Sendable (State) -> Void) {
        queue.async { [self] in
            self.handler = handler
            wanted = true
            failures = 0
            attempt()
        }
    }

    /// What the other Mac sends once joined, on the link's queue. Set before `connect`.
    package func listen(messages: @escaping @Sendable (SessionMessage) -> Void, blobs: @escaping @Sendable (UInt32, Data) -> Void) {
        queue.async { [self] in self.messages = messages; self.blobs = blobs }
    }

    /// Dropped when not joined: a request made while the link is down is not kept for later.
    package func send(_ m: SessionMessage) {
        queue.async { [self] in if joined { channel?.send(m) } }
    }

    package func send(blob: Data, id: UInt32) {
        queue.async { [self] in if joined { channel?.send(blob: blob, id: id) } }
    }

    package func disconnect() {
        queue.async { [self] in
            wanted = false
            retry?.cancel(); retry = nil
            channel?.close(); channel = nil
            handler?(.off)
            handler = nil
        }
    }

    /// The Mac reappeared: no need to wait for the next attempt.
    package func retryNow() {
        queue.async { [self] in
            guard wanted, channel == nil else { return }
            retry?.cancel(); retry = nil
            attempt()
        }
    }

    private func attempt() {
        guard wanted, channel == nil else { return }
        guard let target = endpoint() else { wait("not found on the network"); return }
        handler?(.connecting)
        joined = false
        let c = RemoteChannel(NWConnection(to: target, using: RemoteTransport.client(key: key)), queue: queue)
        channel = c
        c.start { [weak self] event in
            guard let self, c === channel else { return }
            switch event {
            case .ready:
                break
            case .blob(let id, let data):
                if joined { blobs?(id, data) }
            case .message(let json):
                guard let m = try? JSONDecoder().decode(SessionMessage.self, from: json) else { c.close(); return }
                switch m {
                case .challenge(let challenge) where !joined:
                    c.send(SessionMessage.join(hello: hello, identity: key.identity,
                                               proof: SessionMessage.proof(key, challenge: challenge)))
                case .welcome(let theirs) where !joined:
                    if let mm = hello.mismatch(with: theirs) { refuse(.mismatch(mm)); return }
                    joined = true
                    failures = 0
                    handler?(.connected(theirs))
                case .mismatch(let mm):
                    refuse(.mismatch(mm))
                default:
                    if joined { messages?(m) }
                }
            case .closed(let how):
                channel = nil
                switch how {
                // A forgotten Mac's key leaves the server's listener (`RemoteHost.lend`): the handshake fails.
                case .refused: refuse(.forgotten)
                case .lost(let why): wait(why)
                case .broken(let f): wait("\(f)")
                case .closed: break
                }
            }
        }
    }

    private func refuse(_ r: Refusal) {
        wanted = false
        channel?.close(); channel = nil
        handler?(.refused(r))
    }

    private func wait(_ reason: String) {
        guard wanted else { return }
        let delay = Self.delays[min(failures, Self.delays.count - 1)]
        failures += 1
        handler?(.waiting(reason: reason, retryIn: delay))
        let w = DispatchWorkItem { [weak self] in self?.retry = nil; self?.attempt() }
        retry = w
        queue.asyncAfter(deadline: .now() + delay, execute: w)
    }
}

/// **One pairing, from the borrowing side**: connects to the other Mac's pairing window, shows the code
/// when both sides have it, and ends with the key — or without, saying why.
package final class RemotePairingAttempt: @unchecked Sendable {
    package enum Event: Equatable, Sendable {
        /// Show it; the user compares with the other screen and accepts there.
        case compare(code: String)
        case paired(PairedKey)
        case ended(PairingEnd)
    }

    package enum PairingEnd: Equatable, Sendable {
        /// The other Mac's user said no, or closed its window.
        case refused
        /// Someone stood in between, or a proof did not check: nothing is filed.
        case failed(String)
        /// The other Mac's window was not open, or another Mac got there first.
        case unreachable(String)
        case cancelled
    }

    private let queue = DispatchQueue(label: "siliconed.remote.pairing")
    private let target: NWEndpoint
    private let name: String
    private var client = PairingClient()
    private var channel: RemoteChannel?
    private var handler: (@Sendable (Event) -> Void)?
    private var over = false

    package init(to target: NWEndpoint, name: String) {
        self.target = target; self.name = name
    }

    package func start(_ handler: @escaping @Sendable (Event) -> Void) {
        queue.async { [self] in
            self.handler = handler
            let c = RemoteChannel(NWConnection(to: target, using: RemoteTransport.pairing()), queue: queue)
            channel = c
            // The attempt holds itself through its channel until it ends (`RemoteChannel` drops its
            // handler on `.closed`): the user's sheet may go, the pairing still ends cleanly.
            c.start { [self] event in
                guard !over else { return }
                switch event {
                case .ready:
                    c.send(client.start(name: name))
                case .blob:
                    end(.failed("unexpected data"))
                case .message(let json):
                    guard let m = try? JSONDecoder().decode(PairingMessage.self, from: json) else { end(.failed("unreadable message")); return }
                    do throws(PairingFailure) {
                        let answer = try client.receive(m)
                        if let answer { c.send(answer) }
                        if let key = client.key {
                            over = true
                            c.finish()
                            handler(.paired(key))
                        } else if case .reveal = m, let code = client.code {
                            handler(.compare(code: code))
                        }
                    } catch {
                        end(error == .refused ? .refused : .failed("\(error)"))
                    }
                case .closed(let how):
                    // A window that closes on us before the first message: not open, or someone else's.
                    end(client.code != nil ? .refused : .unreachable("\(how)"))
                }
            }
        }
    }

    package func cancel() {
        queue.async { [self] in
            guard !over else { return }
            channel?.send(PairingMessage.refused)
            channel?.finish()
            channel = nil
            over = true
            handler?(.ended(.cancelled))
        }
    }

    private func end(_ how: PairingEnd) {
        guard !over else { return }
        over = true
        channel?.close(); channel = nil
        handler?(.ended(how))
    }
}
