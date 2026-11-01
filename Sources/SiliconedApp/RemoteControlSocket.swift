// **The remote control's plumbing**: the socket, the reads, the writes — off the main thread. The
// protocol and the commands are in `RemoteControl.swift`.

import Foundation

// MARK: - The plumbing, off the main thread

/// **A client**: its descriptor, and a write queue of its own — a long answer (`wait`) blocks
/// neither the app nor the other clients.
final class Connection: @unchecked Sendable {
    private let fd: Int32
    private let writings = DispatchQueue(label: "app.remote.connection")
    private let lock = NSLock()
    private var _broken = false

    init(fd: Int32) { self.fd = fd }

    /// The client is gone (Ctrl-C during `wait`): we stop writing to it.
    var broken: Bool { lock.withLock { _broken } }

    /// **The client has hung up**, seen without writing to it: its end of the socket is closed (it sent its line and waits, so a
    /// readable socket that reads zero bytes is a close). For a command that computes a long time before its next line
    /// (`rewrite` loading the writer): it cancels the gesture instead of finishing it for nobody.
    var isGone: Bool {
        if broken { return true }
        var poll = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
        guard Darwin.poll(&poll, 1, 0) > 0 else { return false }
        if poll.revents & Int16(POLLHUP | POLLERR | POLLNVAL) != 0 { return true }
        var byte: UInt8 = 0
        return poll.revents & Int16(POLLIN) != 0 && recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT) == 0
    }

    func send(_ rowLine: Data) {
        writings.async { [self] in
            guard !broken else { return }
            let ok = rowLine.withUnsafeBytes { raw -> Bool in
                var remaining = raw.count, begin = 0
                while remaining > 0 {
                    let n = write(fd, raw.baseAddress! + begin, remaining)
                    if n < 0 { if errno == EINTR { continue }; return false }
                    remaining -= n; begin += n
                }
                return true
            }
            if !ok { lock.withLock { _broken = true } }
        }
    }

    /// Closes after the pending writes.
    func shutDown() {
        writings.async { [self] in
            lock.withLock { _broken = true }
            close(fd)
        }
    }
}

/// **The socket**: listen, accept, read the command line. Nothing here is isolated to the main
/// thread — a `DispatchSource` handler that was would crash on the first client.
enum Socket {
    struct Failure: Error, CustomStringConvertible {
        let description: String
    }

    /// Is an app already answering on this socket?
    static func isAnswering(_ path: String) -> Bool {
        guard let fd = try? connectTo(path) else { return false }
        close(fd)
        return true
    }

    static func address(_ path: String) throws -> sockaddr_un {
        var a = sockaddr_un()
        a.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8)
        let capacity = MemoryLayout.size(ofValue: a.sun_path)
        guard bytes.count < capacity else {
            let n = bytes.count, max = capacity - 1
            throw Failure(description: String(localized: "socket path too long (\(n) bytes, \(max) at most): \(path)"))
        }
        withUnsafeMutableBytes(of: &a.sun_path) { buffer in
            buffer.copyBytes(from: bytes)
            buffer[bytes.count] = 0
        }
        a.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        return a
    }

    static func connectTo(_ path: String) throws -> Int32 {
        var a = try address(path)
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw Failure(description: String(localized: "the socket could not be opened (\(String(cString: strerror(errno))))")) }
        let r = withUnsafePointer(to: &a) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        guard r == 0 else { close(fd); throw Failure(description: String(cString: strerror(errno))) }
        return fd
    }

    /// Listens on `path`; each command read reaches `onReceive` (on a background queue).
    static func startListening(_ path: String,
                        onReceive: @escaping @Sendable (_ argv: [String], _ cwd: String, Connection) -> Void) throws -> DispatchSourceRead {
        if FileManager.default.fileExists(atPath: path) {
            guard !isAnswering(path) else {
                throw Failure(description: String(localized: "another Siliconed is already listening on \(path)"))
            }
            unlink(path)   // left by an app that died without removing it
        }
        var a = try address(path)
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw Failure(description: String(localized: "the socket could not be opened (\(String(cString: strerror(errno))))")) }
        // `bind` creates the socket file with the process's umask: 0077, so that it never exists, even
        // for an instant, with a mode another account could open (the `chmod` below only confirms).
        let previousMask = umask(0o077)
        let r = withUnsafePointer(to: &a) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        umask(previousMask)
        guard r == 0 else {
            let message = String(cString: strerror(errno))
            close(fd)
            throw Failure(description: String(localized: "the socket could not be opened (\(message))"))
        }
        chmod(path, 0o600)
        guard listen(fd, 16) == 0 else { close(fd); throw Failure(description: String(localized: "the socket could not be opened (\(String(cString: strerror(errno))))")) }
        _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK)

        // **A client that hangs up must not take the app down.** A write to a socket whose peer
        // left raises SIGPIPE, which kills the process without a crash report — measured: connect,
        // close without reading, and the app was gone. `SO_NOSIGPIPE` on each client socket did
        // not cover it; ignoring the signal process-wide turns it into `EPIPE`, which `Connection`
        // already handles.
        signal(SIGPIPE, SIG_IGN)
        let source = DispatchSource.makeReadSource(fileDescriptor: fd, queue: .global(qos: .userInitiated))
        source.setEventHandler {
            while true {
                let client = accept(fd, nil, nil)
                guard client >= 0 else { return }   // EAGAIN: nobody left waiting
                // Only this account: the file mode says so, the peer's credentials prove it.
                var uid: uid_t = 0, gid: gid_t = 0
                guard getpeereid(client, &uid, &gid) == 0, uid == getuid() else { close(client); continue }
                // Reading a command line holds a thread for up to 5 s: beyond `maxReading` clients
                // still silent, a new one is told so and closed, rather than piling up threads.
                guard reading.wait(timeout: .now()) == .success else {
                    let line = Response.rowLine(Response.failure("busy", String(localized: "too many connections at once"),
                                                                 help: "retry"))
                    _ = line.withUnsafeBytes { write(client, $0.baseAddress!, $0.count) }
                    close(client)
                    continue
                }
                DispatchQueue.global(qos: .userInitiated).async { serve(client, onReceive: onReceive) }
            }
        }
        source.setCancelHandler { close(fd) }
        source.resume()
        return source
    }

    /// Clients whose command line is still being read, at most (`reading`).
    static let maxReading = 8
    private static let reading = DispatchSemaphore(value: maxReading)

    /// The time a client has to send its whole command line, and each write to it to go through.
    static let readingDeadline: TimeInterval = 5
    static let writingTimeout: TimeInterval = 5

    /// `seconds` as a socket timeout — never zero, which would mean « none ».
    static func timeout(_ seconds: TimeInterval) -> timeval {
        let s = max(seconds, 0.001)
        return timeval(tv_sec: Int(s), tv_usec: Int32((s - s.rounded(.down)) * 1_000_000))
    }

    /// Reads the client's line (5 s at most **in all**, 1 MB at most), then hands it to `onReceive`.
    /// `SO_RCVTIMEO` bounds one `read`: a client trickling a byte every 4 s would hold its thread
    /// forever, so each read gets what is left of the deadline. The slot taken in `reading` is given
    /// back once the line is read: a long `wait` that follows holds none. `SO_SNDTIMEO`: a client
    /// that stops reading its answers is dropped (`Connection.broken`), never a queue blocked for good.
    private static func serve(_ fd: Int32, onReceive: @Sendable ([String], String, Connection) -> Void) {
        _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) & ~O_NONBLOCK)
        var enabledFlag: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabledFlag, socklen_t(MemoryLayout<Int32>.size))
        var sending = timeout(writingTimeout)
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &sending, socklen_t(MemoryLayout<timeval>.size))
        let deadline = Date().addingTimeInterval(readingDeadline)
        let connection = Connection(fd: fd)
        var received = Data()
        var buffer = [UInt8](repeating: 0, count: 65536)
        while !received.contains(0x0A), received.count < 1 << 20 {
            let left = deadline.timeIntervalSinceNow
            guard left > 0 else { break }
            var delay = timeout(left)
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &delay, socklen_t(MemoryLayout<timeval>.size))
            let n = read(fd, &buffer, buffer.count)
            if n < 0, errno == EINTR { continue }
            guard n > 0 else { break }
            received.append(contentsOf: buffer[0..<n])
        }
        reading.signal()
        let rowLine = received.split(separator: 0x0A, maxSplits: 1, omittingEmptySubsequences: false).first.map { Data($0) } ?? Data()
        guard let object = try? JSONSerialization.jsonObject(with: rowLine) as? [String: Any],
              let argv = object["argv"] as? [String] else {
            connection.send(Response.rowLine(Response.failure("protocol",
                String(localized: "expected: one JSON line {\"argv\": [\"command\", …], \"cwd\": \"/folder\"}"))))
            connection.shutDown()
            return
        }
        onReceive(argv, object["cwd"] as? String ?? FileManager.default.homeDirectoryForCurrentUser.path, connection)
    }
}

/// The JSON lines: ASCII keys, sorted — stable from one call to the next, readable by `jq`.
enum Response {
    /// A decimal as one writes it: `0.8`, not `0.80000000000000004` (`JSONSerialization` writes a
    /// `Double` with 17 digits).
    static func number(_ x: Double, _ decimals: Int = 2) -> NSDecimalNumber {
        NSDecimalNumber(string: String(format: "%.\(decimals)f", x), locale: Locale(identifier: "en_US_POSIX"))
    }

    static func rowLine(_ object: [String: Any]) -> Data {
        var d = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]))
            ?? Data("{\"ok\":false,\"error\":\"internal\",\"message\":\"response not serializable\"}".utf8)
        d.append(0x0A)
        return d
    }

    static func failure(_ code: String, _ message: String, help: String? = nil) -> [String: Any] {
        var o: [String: Any] = ["ok": false, "error": code, "message": message]
        if let help { o["hint"] = help }
        return o
    }
}
