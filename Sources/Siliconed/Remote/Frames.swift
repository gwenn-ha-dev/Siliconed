import Foundation

/// **What travels between two Macs, cut into frames** on one TCP stream (TLS above it, once paired).
/// Two kinds: a message (a small `Codable`, in JSON) and a piece of a blob (a PNG of a few megabytes,
/// a reference image). The point is the order of sending: a blob goes out in 64 KiB pieces, and any
/// message waiting is sent between two pieces — the preview and the progress of a render never wait
/// behind an image coming down.
///
///     frame = kind u8 ‖ flags u8 ‖ length u32 (big-endian) ‖ payload
///       kind 0, message:  payload = JSON
///       kind 1, piece:    payload = blob id u32 ‖ bytes ; flags bit 0 = last piece
///
/// **The stream comes from the network**: a length is bounded before anything is allocated, a blob
/// is bounded as it grows, and an unknown kind ends the session rather than being skipped.

// MARK: - A frame

package struct Frame: Equatable, Sendable {
    package enum Kind: UInt8, Sendable { case message = 0, piece = 1 }

    package let kind: Kind
    package let flags: UInt8
    package let payload: Data

    package static let headerSize = 6
    /// A message is small; a piece is `Outbox.pieceSize` plus its id. Anything longer is refused.
    package static let maximumPayload = 1 << 20

    package init(kind: Kind, flags: UInt8 = 0, payload: Data) {
        self.kind = kind; self.flags = flags; self.payload = payload
    }

    package var bytes: Data {
        var d = Data([kind.rawValue, flags])
        d.append(contentsOf: Frame.bigEndian(UInt32(payload.count)))
        d.append(payload)
        return d
    }

    static func bigEndian(_ v: UInt32) -> [UInt8] {
        [UInt8(v >> 24), UInt8(v >> 16 & 0xff), UInt8(v >> 8 & 0xff), UInt8(v & 0xff)]
    }

    static func readBigEndian(_ d: Data, at i: Data.Index) -> UInt32 {
        d[i..<i + 4].reduce(UInt32(0)) { $0 << 8 | UInt32($1) }
    }
}

package enum FrameFailure: Error, Equatable, Sendable {
    case unknownKind(UInt8)
    case tooLong(Int)
    /// A piece shorter than its blob id.
    case truncatedPiece
    case blobTooLarge(id: UInt32)
    case tooManyBlobs
}

// MARK: - Reading

/// Frames out of the bytes as they arrive, in any cut.
package struct FrameReader: Sendable {
    private var buffer = Data()

    package init() {}

    package mutating func append(_ bytes: Data) { buffer.append(bytes) }

    /// The next whole frame, or `nil` until more bytes arrive.
    package mutating func next() throws(FrameFailure) -> Frame? {
        guard buffer.count >= Frame.headerSize else { return nil }
        let s = buffer.startIndex
        guard let kind = Frame.Kind(rawValue: buffer[s]) else { throw .unknownKind(buffer[s]) }
        let length = Int(Frame.readBigEndian(buffer, at: s + 2))
        guard length <= Frame.maximumPayload else { throw .tooLong(length) }
        guard buffer.count >= Frame.headerSize + length else { return nil }
        let payload = Data(buffer[(s + Frame.headerSize)..<(s + Frame.headerSize + length)])
        let flags = buffer[s + 1]
        buffer = Data(buffer[(s + Frame.headerSize + length)...])
        return Frame(kind: kind, flags: flags, payload: payload)
    }
}

/// Blobs put back together from their pieces, each bounded.
package struct BlobAssembler: Sendable {
    package let maximumSize: Int
    package let maximumOpen: Int
    private var open: [UInt32: Data] = [:]

    /// 64 MiB covers a 4096² PNG (about 20 MB) three times over.
    package init(maximumSize: Int = 64 << 20, maximumOpen: Int = 8) {
        self.maximumSize = maximumSize; self.maximumOpen = maximumOpen
    }

    /// A piece; returns the whole blob when this was its last piece.
    package mutating func receive(_ f: Frame) throws(FrameFailure) -> (id: UInt32, data: Data)? {
        guard f.payload.count >= 4 else { throw .truncatedPiece }
        let id = Frame.readBigEndian(f.payload, at: f.payload.startIndex)
        if open[id] == nil, open.count >= maximumOpen { throw .tooManyBlobs }
        var d = open[id] ?? Data()
        d.append(f.payload.dropFirst(4))
        guard d.count <= maximumSize else { open[id] = nil; throw .blobTooLarge(id: id) }
        if f.flags & 1 != 0 {
            open[id] = nil
            return (id, d)
        }
        open[id] = d
        return nil
    }
}

// MARK: - Sending

/// What waits to go out: messages first, in order; blobs one piece at a time, in order, between them.
package struct Outbox: Sendable {
    package static let pieceSize = 64 << 10

    private var messages: [Data] = []
    private var blobs: [(id: UInt32, data: Data, offset: Int)] = []

    package init() {}

    package var isEmpty: Bool { messages.isEmpty && blobs.isEmpty }

    package mutating func send(message json: Data) { messages.append(json) }

    package mutating func send(blob data: Data, id: UInt32) { blobs.append((id, data, 0)) }

    /// The next frame to write.
    package mutating func next() -> Frame? {
        if !messages.isEmpty { return Frame(kind: .message, payload: messages.removeFirst()) }
        guard !blobs.isEmpty else { return nil }
        let b = blobs[0]
        let end = min(b.offset + Self.pieceSize, b.data.count)
        let last = end == b.data.count
        let s = b.data.startIndex
        var payload = Data(Frame.bigEndian(b.id))
        payload.append(b.data[(s + b.offset)..<(s + end)])
        if last { blobs.removeFirst() } else { blobs[0].offset = end }
        return Frame(kind: .piece, flags: last ? 1 : 0, payload: payload)
    }
}
