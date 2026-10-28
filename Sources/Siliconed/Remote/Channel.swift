import Foundation
import Network

/// **One connection, as frames**: messages and blobs (`Frames.swift`) over an `NWConnection` — plain
/// TCP for a pairing, TLS-PSK for a session. Everything runs on the channel's serial queue, its events
/// too; the owner hops to its own actor.
///
/// **One frame in flight at a time.** The next is taken from the `Outbox` only once the network took
/// the previous one: a message queued while a PNG goes down leaves after at most one 64 KiB piece,
/// not after the whole image — what the outbox promises holds on the wire only if the channel does
/// not hand it everything at once.
package final class RemoteChannel: @unchecked Sendable {
    package enum Event: Sendable {
        case ready
        case message(Data)
        case blob(id: UInt32, data: Data)
        case closed(End)
    }

    package enum End: Equatable, Sendable {
        /// The other side closed, or the network lost it (keepalive).
        case lost(String)
        /// The TLS handshake said no: not the key the other side holds (`RemoteTransport.isRefusal`).
        case refused
        /// The other side sent what no build of ours sends: the session ends.
        case broken(FrameFailure)
        /// Closed from here.
        case closed
    }

    package let queue: DispatchQueue
    private let connection: NWConnection
    private var handler: (@Sendable (Event) -> Void)?
    private var reader = FrameReader()
    private var blobs = BlobAssembler()
    private var outbox = Outbox()
    private var writing = false
    /// `finish()` was asked: close once the outbox is empty.
    private var finishing = false
    private var ended = false
    /// What a stream brings in (`Frames.swift`): its id, a piece, whether it is the last. Set before
    /// `start`, called on `queue`; without it the pieces are dropped (a pairing carries none).
    package var streams: (@Sendable (UInt32, Data, Bool) -> Void)?

    package init(_ connection: NWConnection, queue: DispatchQueue) {
        self.connection = connection; self.queue = queue
    }

    /// Starts the connection; every event goes to `handler`, on `queue`, `.closed` last and once.
    package func start(_ handler: @escaping @Sendable (Event) -> Void) {
        queue.async { [self] in
            self.handler = handler
            connection.stateUpdateHandler = { [weak self] state in self?.changed(state) }
            connection.start(queue: queue)
        }
    }

    /// The peer as the network names it (Bonjour service, or host and port).
    package var endpoint: NWEndpoint { connection.endpoint }

    package func send(message json: Data) {
        queue.async { [self] in
            guard !ended else { return }
            outbox.send(message: json)
            pump()
        }
    }

    package func send<M: Encodable>(_ message: M) {
        guard let json = try? JSONEncoder().encode(message) else { return }
        send(message: json)
    }

    package func send(blob: Data, id: UInt32) {
        queue.async { [self] in
            guard !ended else { return }
            outbox.send(blob: blob, id: id)
            pump()
        }
    }

    /// A stream of bytes pulled from `source` as the network takes them (an upload): after the messages
    /// and the blobs waiting, never ahead of them.
    package func send(stream source: any StreamSource, id: UInt32) {
        queue.async { [self] in
            guard !ended else { source.close(); return }
            outbox.send(stream: source, id: id)
            pump()
        }
    }

    package func cancel(stream id: UInt32) {
        queue.async { [self] in outbox.cancel(stream: id) }
    }

    /// Closes once what is queued has gone out — a refusal sent just before must reach the other side.
    package func finish() {
        queue.async { [self] in
            if outbox.isEmpty && !writing { end(.closed) } else { finishing = true }
        }
    }

    package func close() {
        queue.async { [self] in end(.closed) }
    }

    // MARK: - The connection's life

    private func changed(_ state: NWConnection.State) {
        switch state {
        case .ready:
            handler?(.ready)
            receive()
            pump()
        case .waiting(let e), .failed(let e):
            // `.waiting` is the network saying it may come back: a session is retried by its owner, so
            // waiting here would only hide the loss.
            end(RemoteTransport.isRefusal(e) ? .refused : .lost(e.debugDescription))
        case .cancelled:
            end(.closed)
        default:
            break
        }
    }

    private func end(_ how: End) {
        guard !ended else { return }
        ended = true
        outbox.closeStreams()
        connection.cancel()
        let h = handler
        handler = nil
        h?(.closed(how))
    }

    private func receive() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 256 << 10) { [weak self] data, _, complete, error in
            guard let self, !self.ended else { return }
            if let data, !data.isEmpty {
                reader.append(data)
                do throws(FrameFailure) {
                    while let f = try reader.next() {
                        switch f.kind {
                        case .message: handler?(.message(f.payload))
                        case .piece: if let b = try blobs.receive(f) { handler?(.blob(id: b.id, data: b.data)) }
                        case .stream:
                            let p = try f.streamPiece()
                            streams?(p.id, p.bytes, p.last)
                        }
                        if ended { return }
                    }
                } catch {
                    end(.broken(error))
                    return
                }
            }
            if let error { end(RemoteTransport.isRefusal(error) ? .refused : .lost(error.debugDescription)); return }
            if complete { end(.lost("closed by the other side")); return }
            receive()
        }
    }

    private func pump() {
        guard !writing, !ended, connection.state == .ready else { return }
        guard let frame = outbox.next() else {
            if finishing { end(.closed) }
            return
        }
        writing = true
        connection.send(content: frame.bytes, completion: .contentProcessed { [weak self] error in
            guard let self else { return }
            writing = false
            if let error { end(.lost(error.debugDescription)); return }
            pump()
        })
    }
}
