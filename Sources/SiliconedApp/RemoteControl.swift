// **The app's remote control**: the open app listens on a Unix socket, and the `silicontrol`
// command (shipped inside Siliconed.app) drives it — the same queue, the same history, the same
// buttons as the mouse. The protocol and the commands are in `docs/REMOTE-CONTROL.md`; this file
// is its server.
//
// **The protocol**: one connection per command. The client writes ONE JSON line,
// `{"argv": ["add", "--model", "z-image", "…"], "cwd": "/client/folder"}`, then reads JSON lines
// until the connection closes. Intermediate lines carry `"type"` (an event); the last one carries
// `"ok"` (the verdict). The arguments are those of the command line, as is: all the parsing is
// here, the client is only a pipe — and `nc -U` is enough to talk to the app. The help
// (`help`, `help <topic>`, `<command> --help`) is answered here too, in the user's language:
// `{"ok": true, "help": "<text>"}`. Its pages and its routing live in `Sources/SilicontrolHelp`,
// shared with `silicontrol`, which answers them itself without launching the app.
//
// **Nobody else**: the socket is `<library>/silicontrol.sock`, mode 0600 — only the account that
// launched the app can open it. No network port. **Incognito kept**: nothing is written to disk
// unless asked (`--out`, `save`), like « Save » in the app.
//
// The plumbing (socket, reads, writes) never touches the main thread; the commands do: they read
// and change the `AppState`, like a button.

import AppKit
import Foundation
import Observation
import Siliconed
import SilicontrolHelp

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

    /// Reads the client's line (5 s at most, 1 MB at most), then hands it to `onReceive`. The slot
    /// taken in `reading` is given back once the line is read: a long `wait` that follows holds none.
    private static func serve(_ fd: Int32, onReceive: @Sendable ([String], String, Connection) -> Void) {
        _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) & ~O_NONBLOCK)
        var enabledFlag: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabledFlag, socklen_t(MemoryLayout<Int32>.size))
        var delay = timeval(tv_sec: 5, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &delay, socklen_t(MemoryLayout<timeval>.size))
        let connection = Connection(fd: fd)
        var received = Data()
        var buffer = [UInt8](repeating: 0, count: 65536)
        while !received.contains(0x0A), received.count < 1 << 20 {
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

// MARK: - The commands, on the main thread

@MainActor @Observable
final class RemoteControl {
    @ObservationIgnored let app: AppState
    @ObservationIgnored private var source: DispatchSourceRead?
    /// Why the remote control is not listening (another Siliconed, a path too long), or `nil` —
    /// shown greyed in the Siliconed menu, under « Install the silicontrol command… ».
    private(set) var failureMessage: String?

    init(app: AppState) { self.app = app }

    var path: String { app.library.socket }

    func boot() {
        do {
            source = try Socket.startListening(path) { [weak self] argv, cwd, connection in
                Task { @MainActor in self?.handle(argv, cwd: cwd, connection) }
            }
        } catch {
            failureMessage = "\(error)"
        }
    }

    /// When the app quits: the socket goes with it.
    func stop() {
        guard let source else { return }
        source.cancel()
        self.source = nil
        unlink(path)
    }

    // ── Answering ──

    private func finish(_ c: Connection, _ object: [String: Any]) {
        c.send(Response.rowLine(object))
        c.shutDown()
    }

    private func emit(_ c: Connection, _ object: [String: Any]) { c.send(Response.rowLine(object)) }

    /// An error a command raises: its code is the `error` field of the answer.
    struct Refusal: Error {
        let code: String, message: String, help: String?
        init(_ code: String, _ message: String, help: String? = nil) { self.code = code; self.message = message; self.help = help }
    }

    private func handle(_ argv: [String], cwd: String, _ c: Connection) {
        // `help`, `<command> --help`: the pages `silicontrol` also answers on its own, without touching the app.
        if let answer = RemoteControlHelp.answer(argv) { finish(c, answer); return }
        let command = argv.first ?? ""
        let arguments = Array(argv.dropFirst())
        do {
            switch command {
            case "status": finish(c, state())
            case "models": finish(c, models())
            case "install": finish(c, try install(arguments))
            case "diagnose": try diagnose(arguments, c)
            case "add": try add(arguments, cwd: cwd, c)
            case "grid": try add(arguments, cwd: cwd, c, grid: true)
            case "wait": try wait(arguments, cwd: cwd, c)
            case "follow": follow(c)
            case "cancel": finish(c, try cancel(arguments))
            case "clear": finish(c, clear())
            case "history": finish(c, try history(arguments))
            case "save": finish(c, try save(arguments, cwd: cwd))
            case "vary": finish(c, try vary(arguments))
            default:
                throw Refusal("usage", String(localized: "unknown command: \"\(command)\""),
                            help: "silicontrol help")
            }
        } catch let r as Refusal {
            // A usage mistake with no advice of its own points to its command's help.
            let help = r.help ?? (r.code == "usage" ? "silicontrol help \(command)" : nil)
            finish(c, Response.failure(r.code, r.message, help: help))
        } catch {
            finish(c, Response.failure("internal", "\(error)"))
        }
    }

    // ── The objects returned ──

    private func json(_ t: Job) -> [String: Any] {
        let r = t.settings
        return ["job": t.name, "model": r.identifier, "prompt": r.prompt,
                "width": r.format.width, "height": r.format.height, "steps": r.steps,
                "seeds": t.seeds.map { NSNumber(value: $0) }, "edit": !t.references.isEmpty,
                "references": t.references.count,
                "loras": r.loras.map { ["lora": Self.shortName($0.path), "strength": Response.number($0.strength)] }]
            .merging(gridPlace(t)) { a, _ in a }
            .merging(r.isVariation ? ["variation": Self.json(r.variations)] : [:]) { a, _ in a }
            .merging(r.detail == .normal ? [:] : ["detail": r.detail.rawValue]) { a, _ in a }
    }

    /// The variation seeds in order, each with its strength — the seed stays the origin's.
    private static func json(_ variations: [Variation]) -> [[String: Any]] {
        variations.map { ["seed": NSNumber(value: $0.seed), "strength": Response.number($0.strength)] }
    }

    /// A grid's cell: its grid, its column and its row (from 0) — empty for any other job.
    private func gridPlace(_ t: Job) -> [String: Any] {
        guard let at = app.gridPlace(of: t.ordinal), let run = app.gridRuns[at.grid] else { return [:] }
        return ["grid": run.name, "column": at.cell % run.grid.columns, "row": at.cell / run.grid.columns]
    }

    private func json(_ e: Entry, path: String? = nil) -> [String: Any] {
        let r = e.settings
        var o: [String: Any] = [
            "image": e.name, "job": "t\(e.job)", "model": r.identifier, "prompt": r.prompt,
            "width": e.width, "height": e.height, "seed": NSNumber(value: e.seed), "steps": r.steps,
            "evaluations": e.evaluations, "seconds": Response.number(e.seconds), "edit": r.editing,
            "references": r.references,
            "loras": r.loras.map { ["lora": Self.shortName($0.path), "strength": Response.number($0.strength)] }]
        if r.isVariation { o["variation"] = Self.json(r.variations) }
        if r.detail != .normal { o["detail"] = r.detail.rawValue }
        if let g = e.grid {
            // A grid's sheet: no job made it; its seconds and evaluations are its cells' sum.
            o["job"] = nil
            o["grid"] = g.name
            o["sheet"] = true
            o["columns"] = g.grid.columns
            o["rows"] = g.grid.rows
            o["empty_cells"] = g.empty
        }
        if let n = e.sketch { o["sketch"] = n }
        if let path { o["path"] = path }
        return o
    }

    /// A job's end, as the remote control writes it: `outcome`, and for an error its translated
    /// `message` and the engine's stable `code` (`EngineError.code`, `"license_not_accepted"`…).
    private static func json(_ issue: Issue, job n: Int, images: [Int]) -> [String: Any] {
        var o: [String: Any] = ["job": "t\(n)", "images": images.map { "i\($0)" }]
        switch issue {
        case .finished: o["outcome"] = "done"
        case .stopped: o["outcome"] = "stopped"
        case .removed: o["outcome"] = "removed"
        case .failed(let p):
            o["outcome"] = "error"
            o["message"] = p.text
            if let code = p.code { o["code"] = code }
        }
        return o
    }

    /// `flat` for `…/store/flat.lora.silicon`: what one types after `--lora`.
    nonisolated static func shortName(_ path: String) -> String {
        let file = (path as NSString).lastPathComponent
        return file.hasSuffix(".lora.silicon") ? String(file.dropLast(".lora.silicon".count)) : file
    }

    // ── status ──

    private func state() -> [String: Any] {
        var o: [String: Any] = ["ok": true, "library": app.library.root.path,
                                "history_images": app.history.count]
        if let t = app.currentJob {
            var e = json(t)
            e["stage"] = app.stage.map(Self.stageName) ?? NSNull()
            e["batch_image"] = app.batchImage + 1
            e["step"] = app.currentStep
            e["steps_total"] = app.totalSteps
            e["fraction"] = Response.number(app.fraction, 3)
            e["remaining_s"] = app.remaining(at: Date()).map { Response.number($0, 1) } ?? NSNull()
            o["running"] = e
        } else {
            o["running"] = NSNull()
        }
        o["queue"] = app.file.map { t -> [String: Any] in
            var e = json(t)
            e["estimate_s"] = app.estimation(t).map { Response.number($0, 1) } ?? NSNull()
            return e
        }
        // Refused by the engine until the user accepts the model's license in the app (a sheet shows it):
        // out of the queue, not lost — accepted, they go back to its head; declined, they end in "error".
        o["waiting_for_license"] = app.awaitingLicense.map { t -> [String: Any] in
            var e = json(t)
            e["state"] = "waiting_for_license"
            let license = app.cards.first { $0.id == t.settings.identifier }?.license
            e["license"] = license?.text ?? NSNull()
            e["license_urls"] = license?.urls.map(\.absoluteString) ?? NSNull()
            return e
        }
        // An installation or an import blocks renders (`busy`): say it, and where it is.
        o["installing"] = app.forgeInProgress.map { title -> [String: Any] in
            ["title": title, "last": app.forgeJournal.last ?? NSNull()]
        } ?? NSNull()
        let remaining = app.remainingInQueue(at: Date())
        o["queue_remaining_s"] = Response.number(remaining.seconds, 1)
        o["remaining_complete"] = remaining.isComplete
        return o
    }

    private static func stageName(_ item: Engine.Stage) -> String {
        switch item {
        case .text: "text"
        case .image: "image"
        case .denoising: "denoising"
        case .decoding: "decoding"
        }
    }

    // ── models ──

    /// The models the app shows — Z-Image and Qwen-Image-2.1, all of them in Developer Mode.
    private func models() -> [String: Any] {
        let b = app.library
        let list = b.cards().filter { app.isVisible($0.family) }.map { f -> [String: Any] in
            var o: [String: Any] = ["id": f.id, "name": f.name, "default_steps": f.defaultSteps,
                                    "formats": f.formats.map { "\($0.width)x\($0.height)" },
                                    "commercial_license": f.license.commercial,
                                    "license": f.license.text,
                                    // A Compact or a Light also links its third parties' model cards (`Family.licenseURLs`).
                                    "license_urls": (f.isImported ? f.license.urls : f.family.licenseURLs(b.variant(of: f.family)))
                                        .map(\.absoluteString),
                                    "license_accepted": b.isLicenseAccepted(f)]
            o["default"] = f.id == "z-image"   // the default of `add` without --model
            // Stage C: the versions a publisher's model installs in, each with its place on disk, and
            // the one on disk once it is ready (an imported model has its own DiT: no version).
            if !f.isImported {
                o["variants"] = f.family.variants.map { v -> [String: Any] in
                    let s = f.family.installedSize(v)
                    return ["variant": v.rawValue, "disk_bytes": s.dit + s.encoder + s.components]
                }
            }
            if let missingList = f.missing(in: b) {
                o["ready"] = false
                o["missing"] = missingList
                // The user installs, in the app: the license and the size are read there first.
                o["install"] = f.isImported ? "silicontrol install \(f.family.rawValue)" : "silicontrol install \(f.id)"
                // One only knows by opening the installed model's chain.
                o["edit"] = NSNull()
                o["max_references"] = NSNull()
                o["variant"] = NSNull()
            } else {
                o["ready"] = true
                o["variant"] = f.isImported ? NSNull() : b.variant(of: f.family).rawValue as Any
                let max = (try? Model.named(f.id, in: b))?.chain.denoising.maxReferences ?? 0
                o["edit"] = max > 0
                o["max_references"] = max
                o["loras"] = b.loras(for: f.id).map { ["lora": Self.shortName($0.path), "name": $0.name] }
            }
            return o
        }
        return ["ok": true, "models": list]
    }

    // ── install ──

    /// `install <id> [--standard|--compact|--light]`: **the app shows the model's license and the place it
    /// takes, and the user accepts there** — the command downloads nothing by itself. It opens the
    /// Models sheet on the model and returns at once; `models` then says "ready" once the
    /// installation is done. **Without a flag the version installed is kept** (Standard if none): a
    /// command that names no version never switches. `--standard`/`--compact`/`--light` preselect that
    /// version — the one installed is offered for replacement, like « Switch to … » in the sheet. An
    /// imported model's prerequisites (`baseDiT` false) never switch: a flag that contradicts the
    /// family's version installed is refused, with the reason.
    private func install(_ arguments: [String]) throws -> [String: Any] {
        let options = arguments.filter { $0.hasPrefix("--") }
        let flags = Variant.allCases.map { "--" + $0.rawValue }
        if let unknown = options.first(where: { !flags.contains($0) }) {
            throw Refusal("usage", String(localized: "unknown flag: \(unknown)"), help: "silicontrol help install")
        }
        let named = Variant.allCases.filter { options.contains("--" + $0.rawValue) }
        if named.count > 1 {
            throw Refusal("usage", String(localized: "one version at most: --standard, --compact or --light"), help: "silicontrol help install")
        }
        let ids = arguments.filter { !$0.hasPrefix("--") }
        guard ids.count == 1, let id = ids.first else {
            throw Refusal("usage", String(localized: "install expects one model id"), help: "silicontrol models")
        }
        let b = app.library
        guard let card = b.cards().first(where: { $0.id == id || $0.family.rawValue == id }), app.isVisible(card.family) else {
            let visible = b.cards().filter { app.isVisible($0.family) }.map(\.id).joined(separator: ", ")
            throw Refusal("model", String(localized: "unknown model: \"\(id)\""), help: String(localized: "models: \(visible)"))
        }
        let asked = named.first
        if let asked, !card.family.variants.contains(asked) {
            let version = VersionChoice.name(asked)
            throw Refusal("usage", String(localized: "\(card.name) has no \(version) version"), help: "silicontrol models")
        }
        let installed = b.installedVariant(of: card.family)
        // Nothing installed: the version the app preselects (Standard; Light on a Mac of 8 GB).
        let variant = asked ?? installed ?? (card.isImported ? .standard : card.family.preselectedVariant())
        // What an imported model lacks installs in its family's version, never another one.
        if card.isImported, let asked, asked != (installed ?? .standard) {
            let name = card.name, family = card.family.name, kept = VersionChoice.name(installed ?? .standard)
            let command = "silicontrol install \(card.family.rawValue) --\(asked.rawValue)"
            throw Refusal("usage", String(localized: "\(name) needs \(family) in the version installed (\(kept)): an imported model does not switch versions"),
                          help: String(localized: "switch the publisher's model first: \(command)"))
        }
        // Ready, and in the version asked (an imported model keeps its family's version): nothing to do.
        guard card.missing(in: b) != nil || (!card.isImported && b.variant(of: card.family) != variant) else {
            return ["ok": true, "model": card.id, "installed": true, "variant": b.variant(of: card.family).rawValue,
                    "message": String(localized: "already installed")]
        }
        guard app.managementPossible else {
            throw Refusal("busy", String(localized: "a render, an installation or the diagnostic is running"),
                        help: "silicontrol wait, then retry")
        }
        app.offerInstall(card.family, baseDiT: !card.isImported, variant: variant)
        NSApp.activate()
        return ["ok": true, "model": card.id, "installed": false, "variant": variant.rawValue,
                "message": String(localized: "the app shows the license and the space on disk: the user accepts there")]
    }

    // ── diagnose ──

    /// `diagnose [--issue]`: **« Report My Configuration », from the terminal** — the sheet opens in the
    /// app and the diagnostic runs (~15 s per visible installed model, the queue held meanwhile); the
    /// answer is the report, the same JSON the issue carries. `--issue` then does what the sheet's
    /// « Open the Issue » does: GitHub's form opens in the browser, prefilled — the user submits it.
    private func diagnose(_ arguments: [String], _ c: Connection) throws {
        let issue = arguments.contains("--issue")
        if let unknown = arguments.first(where: { $0 != "--issue" }) {
            throw Refusal("usage", String(localized: "unknown flag: \(unknown)"), help: "silicontrol help diagnose")
        }
        if let reason = app.diagnosticBlocker { throw Refusal("busy", reason, help: "silicontrol wait, then retry") }
        app.openDiagnostic()
        app.startDiagnostic()
        NSApp.activate()
        emit(c, ["type": "started", "models": app.diagnosticCards.map(\.id)])
        Task { @MainActor [weak self] in
            while let self, self.app.diagnosticRunning, !c.broken {
                try? await Task.sleep(for: .milliseconds(500))
            }
            guard let self else { return }
            guard case .finished(let report) = self.app.diagnostic,
                  let data = try? report.json(pretty: false),
                  let object = try? JSONSerialization.jsonObject(with: data) else {
                self.finish(c, Response.failure("stopped", String(localized: "the diagnostic was stopped")))
                return
            }
            var o: [String: Any] = ["ok": true, "report": object]
            if issue {
                o["issue_url_length"] = (try? report.issueURL().absoluteString.count) ?? NSNull()
                if let notice = self.app.openIssue(report) { o["message"] = notice }
            }
            self.finish(c, o)
        }
    }

    // ── add ──

    /// The arguments of the developer's render command, plus `--wait` and `--out`.
    private struct RenderOrder {
        var model = "z-image"
        var prompt: String?
        var format: (Int, Int)?
        var seed: UInt64?
        var batch = 1
        var steps: Int?
        /// `--detail more|most`: the rack's Detail choice (`Request.detail`).
        var detail = Detail.normal
        var loras: [(name: String, strength: Double)] = []
        /// The `--ref`, in their order: the first is image 1, the one edited.
        var references: [String] = []
        /// `nil`: like the app's « Preview at every step » switch.
        var previews: Bool?
        var wait = false
        var folder: String?
        /// `grid`: the axes as typed (`RenderGrid.axis`), and whether `--batch` was given (refused).
        var x: String?, y: String?
        var batchGiven = false
        /// `grid --sketch`: the exploration's cells (`Sketch`), shown in its window, not in the history.
        var sketch = false
    }

    private func parse(_ arguments: [String]) throws -> RenderOrder {
        var d = RenderOrder()
        var positional: [String] = []
        var i = 0
        func value(_ flag: String) throws -> String {
            guard i + 1 < arguments.count else { throw Refusal("usage", String(localized: "\(flag) expects a value")) }
            i += 1
            return arguments[i]
        }
        while i < arguments.count {
            let a = arguments[i]
            switch a {
            case "--model": d.model = try value(a)
            case "--steps":
                let v = try value(a)
                guard let n = Int(v), (1...Request.maximumSteps).contains(n) else {
                    throw Refusal("usage", String(localized: "--steps expects an integer from 1 to \(Request.maximumSteps), not \"\(v)\""))
                }
                d.steps = n
            case "--detail":
                let v = try value(a)
                guard let level = Detail(rawValue: v.lowercased()) else {
                    throw Refusal("usage", String(localized: "--detail expects normal, more or most, not \"\(v)\""))
                }
                d.detail = level
            case "--batch":
                let v = try value(a)
                guard let n = Int(v), (1...8).contains(n) else { throw Refusal("usage", String(localized: "--batch expects an integer from 1 to 8, not \"\(v)\"")) }
                d.batch = n
                d.batchGiven = true
            case "--x": d.x = try value(a)
            case "--y": d.y = try value(a)
            case "--lora":
                let v = try value(a)
                // `name:strength` — the last colon separates the strength (a path has none).
                if let colonIndex = v.lastIndex(of: ":"), let f = Double(v[v.index(after: colonIndex)...].replacingOccurrences(of: ",", with: ".")) {
                    d.loras.append((String(v[..<colonIndex]), f))
                } else {
                    d.loras.append((v, 1))
                }
            case "--ref": d.references.append(try value(a))
            case "--out": d.folder = try value(a)
            case "--preview": d.previews = true
            case "--no-preview": d.previews = false
            case "--wait": d.wait = true
            case "--sketch": d.sketch = true
            case "--image", "--strength":
                throw Refusal("img2img", String(localized: "the app does no img2img (--image, --strength)"),
                            help: String(localized: "--ref <image> for editing (a model with \"edit\": true)"))
            default:
                if a.hasPrefix("--") { throw Refusal("usage", String(localized: "unknown flag: \(a)")) }
                positional.append(a)
            }
            i += 1
        }
        guard let prompt = positional.first else {
            throw Refusal("usage", String(localized: "the prompt is missing"), help: "silicontrol add [--model id] \"<prompt>\" [512|1024|WxH] [seed] — silicontrol help add")
        }
        d.prompt = prompt
        for p in positional.dropFirst() {
            if d.format == nil, p.lowercased().contains("x") || p.contains("×") {
                guard let f = Format.parse(p) else { throw Refusal("format", String(localized: "unreadable format: \"\(p)\""), help: String(localized: "width x height, e.g. 832x1216")) }
                d.format = (f.width, f.height)
            } else if d.format == nil, p == "512" || p == "1024" {
                d.format = (Int(p)!, Int(p)!)
            } else if d.seed == nil, let g = UInt64(p) {
                d.seed = g
            } else {
                throw Refusal("usage", String(localized: "extra argument: \"\(p)\""),
                            help: String(localized: "the prompt is ONE argument: put it in quotes — silicontrol help add"))
            }
        }
        return d
    }

    private func absolute(_ path: String, _ cwd: String) -> String {
        URL(fileURLWithPath: (path as NSString).expandingTildeInPath, relativeTo: URL(fileURLWithPath: cwd, isDirectory: true))
            .standardizedFileURL.path
    }

    /// `add`, and `grid` (`grid: true`): the same request, checked the same way; a grid queues one
    /// job per cell (`AppState.enqueueGrid`) instead of one per alternative.
    private func add(_ arguments: [String], cwd: String, _ c: Connection, grid: Bool = false) throws {
        var d = try parse(arguments)
        // One big job at a time on 16 GB: no render while an installation or an import runs.
        if let reason = app.installBlocker { throw Refusal("busy", reason, help: String(localized: "retry once the installation is done (silicontrol models)")) }
        if !grid, d.sketch {
            throw Refusal("usage", String(localized: "--sketch belongs to « silicontrol grid »"), help: "silicontrol help grid")
        }
        if !grid, d.x != nil || d.y != nil {
            throw Refusal("usage", String(localized: "--x and --y belong to « silicontrol grid »"), help: "silicontrol help grid")
        }
        if grid, d.x == nil {
            throw Refusal("usage", String(localized: "a grid needs --x (and optionally --y)"), help: "silicontrol help grid")
        }
        if grid, d.batchGiven {
            throw Refusal("usage", String(localized: "a grid's cell is one image: --batch does not apply"),
                          help: String(localized: "--x seeds=N gives N seeds — silicontrol help grid"))
        }
        let b = app.library
        let model: Model
        do { model = try Model.named(d.model, in: b) } catch {
            let detail = Problem(error).title, readySet = app.visibleReady.map(\.id).joined(separator: ", ")
            throw Refusal("model", String(localized: "model \"\(d.model)\" unavailable: \(detail)"),
                        help: String(localized: "ready: \(readySet) — silicontrol models"))
        }
        let card = b.cards().first { $0.id == model.identifier }
        // The app renders what it shows: the other families need Developer Mode (the developer's
        // command line renders them all).
        guard card.map({ app.isVisible($0.family) }) ?? true else {
            let visible = b.cards().filter { app.isVisible($0.family) }.map(\.id).joined(separator: ", ")
            throw Refusal("model", String(localized: "model \"\(d.model)\" is shown only in Developer Mode"),
                        help: String(localized: "models: \(visible) — or turn on Developer Mode in the app's menu"))
        }
        guard let prompt = d.prompt, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Refusal("usage", String(localized: "empty prompt"))
        }
        // `{a|b}`: one job per expanded prompt, parsed like the app's field — before anything else
        // is read, so that a typo costs nothing.
        let alternatives: PromptAlternatives
        do { alternatives = try PromptAlternatives(prompt) } catch {
            let p = Problem(error)
            throw Refusal("prompt", p.title, help: p.suggestion)
        }

        // The references, in order: files, or images of the history (`i7`). The first is image 1,
        // the one edited: with no format typed, the output takes the denoiser's (`editFormat`) —
        // the app's rule and the CLI's, so that a command and a click give the same image.
        var references: [ReferenceImage] = []
        if !d.references.isEmpty {
            let max = model.chain.denoising.maxReferences
            guard max > 0 else {
                let name = model.name
                throw Refusal("edit", String(localized: "\(name) does no editing by reference (--ref)"),
                            help: String(localized: "models that do have \"edit\": true in \"silicontrol models\""))
            }
            guard d.references.count <= max else {
                let n = d.references.count, name = model.name
                throw Refusal("edit", String(localized: "\(n) references: \(name) reads \(max) at most"),
                            help: String(localized: "\"max_references\" in \"silicontrol models\""))
            }
            references = try d.references.map { r -> ReferenceImage in
                if let entry = image(r), let reference = app.fromHistory(entry) { return reference }
                let url = URL(fileURLWithPath: absolute(r, cwd))
                guard let vignette = NSImage(contentsOf: url), vignette.isValid,
                      let size = ReferenceImage.size(of: url) else {
                    let path = url.path
                    throw Refusal("ref", String(localized: "image unreadable or missing: \(path)"))
                }
                return ReferenceImage(source: .file(url), vignette: vignette, name: url.lastPathComponent,
                                      width: size.width, height: size.height)
            }
            if d.format == nil, let first = references.first {
                let f = model.chain.denoising.editFormat(referenceWidth: first.width, referenceHeight: first.height)
                let capped = AppState.cappedForEditing(RecommendedFormat(f.width, f.height))
                d.format = (capped.width, capped.height)
            }
            // The app edits up to 1024² of area (`AppState.editingSurface`); beyond is the CLI's.
            if let (w, h) = d.format, w * h > AppState.editingSurface {
                throw Refusal("format", String(localized: "the app edits up to 1024² of area, not \(String(w))x\(String(h))"),
                            help: String(localized: "leave the format out (the output follows image 1)"))
            }
        }

        let (width, height) = d.format ?? (1024, 1024)
        do { try Format.check(width: width, height: height) } catch {
            let p = Problem(error)
            throw Refusal("format", p.title, help: p.suggestion)
        }
        guard d.loras.count <= AppState.maxLoRA else {
            let n = d.loras.count, max = AppState.maxLoRA
            throw Refusal("lora", String(localized: "\(n) LoRAs: the app stacks \(max) at most"))
        }
        let stack = try d.loras.map { entry -> LoRASlot in
            guard let f = b.lora(entry.name, for: model.identifier) ?? b.lora(absolute(entry.name, cwd), for: model.identifier) else {
                let compatible = b.loras(for: model.identifier).map { Self.shortName($0.path) }
                let name = entry.name, id = model.identifier, list = compatible.joined(separator: ", ")
                throw Refusal("lora", String(localized: "LoRA \"\(name)\" not found for \(id)"),
                            help: compatible.isEmpty ? String(localized: "no LoRA installed for this model")
                                                      : String(localized: "compatible: \(list)"))
            }
            return LoRASlot(path: f.path, strength: entry.strength)
        }
        guard grid || app.file.count + alternatives.count <= AppState.maxQueueLength else {
            let max = AppState.maxQueueLength
            if alternatives.count > 1 {
                let n = alternatives.count, room = max - app.file.count
                throw Refusal("queue_full", String(localized: "\(n) jobs for the alternatives: the queue has room for \(room) (\(max) jobs)"),
                            help: "silicontrol wait, then retry")
            }
            throw Refusal("queue_full", String(localized: "the queue is full (\(max) jobs)"),
                        help: "silicontrol wait, then retry")
        }
        var folder: String?
        if let raw = d.folder {
            folder = absolute(raw, cwd)
            try createFolder(folder!)
        }

        let defaultSteps = card?.defaultSteps ?? model.chain.denoising.defaultSteps
        let steps = d.steps ?? defaultSteps
        // A step count the denoiser has no schedule for (Qwen-Image-2.1: Viggle's 5, 6, 7, 9) would
        // fail in the queue: refused here, like everything else an accepted request cannot fail on.
        if (try? model.chain.denoising.check(steps: steps, startImage: false)) == nil {
            let accepted = (1...50).filter { (try? model.chain.denoising.check(steps: $0, startImage: false)) != nil }
                .map(String.init).formatted(.list(type: .or))
            throw Refusal("steps", String(localized: "This model has no schedule for \(String(steps)) steps: \(accepted)."),
                        help: "--steps \(defaultSteps) (the model's \"default_steps\")")
        }
        let f = model.chain.denoising.space.factor
        let plan = model.chain.denoising.plan(height: height / f, width: width / f, steps: steps, start: 0,
                                                 withLoRA: !stack.isEmpty || d.detail != .normal)
        // One seed for the whole series, drawn once: the alternatives are compared at the same seeds.
        let firstNumber = d.seed ?? UInt64.random(in: 0...UInt64(UInt32.max))
        if grid {
            let format = RecommendedFormat(width, height)
            try addGrid(d, prompt: prompt, model: model, card: card, format: format, steps: steps, stack: stack,
                        references: references, seed: firstNumber, defaultSteps: defaultSteps, folder: folder, cwd: cwd, c)
            return
        }
        let variants = alternatives.variants
        let jobs = variants.map { v in
            Job(ordinal: app.nextJobNumber(),
                settings: RenderSettings(identifier: model.identifier, modelName: card?.name ?? model.name,
                                         prompt: v.prompt, format: RecommendedFormat(width, height), steps: steps,
                                         loras: stack, references: references.count, detail: d.detail),
                seeds: (0..<d.batch).map { firstNumber &+ UInt64($0) },
                references: references, previews: d.previews ?? app.previews,
                defaultSteps: defaultSteps, plan: plan)
        }
        app.enqueue(jobs)
        // The form takes what the job computes, as after a click: settings, edit images, seed — and
        // the prompt as typed, so that « Generate » would queue the same series.
        app.restoreSettings(jobs[0].settings)
        app.prompt = prompt
        app.references = references
        if let seed = d.seed { app.fixedSeed = true; app.seed = seed }

        let added = zip(jobs, variants).map { job, v -> [String: Any] in
            var o = json(job)
            o["position"] = app.currentJob?.ordinal == job.ordinal
                ? 0 : (app.file.firstIndex { $0.ordinal == job.ordinal }.map { $0 + 1 } ?? 0)
            o["estimate_s"] = app.estimation(job).map { Response.number($0, 1) } ?? NSNull()
            if alternatives.hasAlternatives { o["choices"] = v.choices }
            return o
        }
        if d.wait {
            for var o in added { o["type"] = "added"; emit(c, o) }
            followToEnd(jobs.map(\.ordinal), folder: folder, c)
        } else if jobs.count == 1 {
            // One job: the answer of a prompt without alternatives, unchanged.
            var o = added[0]
            o["ok"] = true
            finish(c, o)
        } else {
            let estimates = jobs.map { app.estimation($0) }
            let total: Any = estimates.contains { $0 == nil } ? NSNull()
                : Response.number(estimates.compactMap { $0 }.reduce(0, +), 1)
            finish(c, ["ok": true, "jobs": added, "estimate_s": total])
        }
    }

    // ── grid ──

    /// **`grid`**: the request `add` checked, its axes read and the grid checked by the library
    /// (`RenderGrid`), every step count of it checked by the denoiser, the queue's room — then one job
    /// per cell, and the sheet when the last one ends.
    private func addGrid(_ d: RenderOrder, prompt: String, model: Model, card: ModelCard?, format: RecommendedFormat,
                         steps: Int, stack: [LoRASlot], references: [ReferenceImage], seed: UInt64,
                         defaultSteps: Int, folder: String?, cwd: String, _ c: Connection) throws {
        let g: RenderGrid
        do {
            let names = d.loras.map(\.name)
            // A LoRA axis names its LoRAs as `--lora` does (a name, else a path from the client's
            // folder): their paths for the cells.
            func resolved(_ a: RenderGrid.Axis) throws(EngineError) -> RenderGrid.Axis {
                guard case let .addedLoRAs(l, f) = a else { return a }
                var paths: [String] = []
                for name in l {
                    if name.isEmpty { paths.append(""); continue }
                    guard let card = app.library.lora(name, for: model.identifier)
                            ?? app.library.lora(absolute(name, cwd), for: model.identifier) else {
                        throw .gridRefused(reason: .noSuchLoRA)
                    }
                    paths.append(card.path)
                }
                return .addedLoRAs(paths, strength: f)
            }
            let x = try resolved(RenderGrid.axis(d.x ?? "", loras: names))
            let y = try d.y.map { try resolved(RenderGrid.axis($0, loras: names)) }
            g = try RenderGrid(.init(prompt: prompt, seed: seed, steps: steps, loraStrengths: stack.map(\.strength),
                                     images: references.count), x: x, y: y)
        } catch {
            let p = Problem(error)
            throw Refusal("grid", p.title, help: p.suggestion ?? "silicontrol help grid")
        }
        // Each cell's step count against its own model's schedules (a model axis: each model's).
        for cell in g.cells {
            guard let n = cell.steps else { continue }
            let denoising = cell.model.flatMap { try? Model.named($0, in: app.library).chain.denoising } ?? model.chain.denoising
            guard (try? denoising.check(steps: n, startImage: false)) == nil else { continue }
            let accepted = (1...50).filter { (try? denoising.check(steps: $0, startImage: false)) != nil }
                .map(String.init).formatted(.list(type: .or))
            throw Refusal("steps", String(localized: "This model has no schedule for \(String(n)) steps: \(accepted)."))
        }
        guard app.file.count + g.count <= AppState.maxQueueLength else {
            let n = g.count, max = AppState.maxQueueLength, room = max - app.file.count
            throw Refusal("queue_full", String(localized: "\(n) jobs for the grid: the queue has room for \(room) (\(max) jobs)"),
                          help: "silicontrol wait, then retry")
        }
        let settings = RenderSettings(identifier: model.identifier, modelName: card?.name ?? model.name, prompt: prompt,
                                      format: format, steps: steps, loras: stack, references: references.count, detail: d.detail)
        if d.sketch { app.stopExploring() }
        let (run, jobs) = app.enqueueGrid(g, settings: settings, references: references, previews: d.previews ?? app.previews,
                                          sketch: d.sketch)
        // The form takes the grid, as after a click: settings, prompt as typed, edit images, axes.
        app.restoreSettings(settings)
        app.prompt = prompt
        app.references = references
        if let s = d.seed { app.fixedSeed = true; app.seed = s }
        app.setGridForm(g, prompt: prompt)

        let added = jobs.map { job -> [String: Any] in
            var o = json(job)
            o["position"] = app.currentJob?.ordinal == job.ordinal
                ? 0 : (app.file.firstIndex { $0.ordinal == job.ordinal }.map { $0 + 1 } ?? 0)
            o["estimate_s"] = app.estimation(job).map { Response.number($0, 1) } ?? NSNull()
            return o
        }
        if d.wait {
            for var o in added { o["type"] = "added"; emit(c, o) }
            followGrid(run.ordinal, folder: folder, c)
        } else {
            let estimates = jobs.map { app.estimation($0) }
            let total: Any = estimates.contains { $0 == nil } ? NSNull()
                : Response.number(estimates.compactMap { $0 }.reduce(0, +), 1)
            finish(c, ["ok": true, "grid": run.name, "columns": g.columns, "rows": g.rows, "jobs": added,
                       "estimate_s": total])
        }
    }

    /// **Follows a grid to its sheet**: each cell's image as it comes (written into `folder` if
    /// given), then the sheet, then the verdict. A grid already ended answers at once.
    private func followGrid(_ g: Int, folder: String?, _ c: Connection) {
        let jobs = app.gridRuns[g]?.jobs ?? app.drawingSheets[g] ?? app.endedGrids[g]?.jobs ?? []
        var tallies: [Int: [String: Any]] = [:]
        func verdict(sheet: Int?) {
            let list = jobs.compactMap { tallies[$0] }
            let allFinished = list.count == jobs.count && list.allSatisfy { $0["outcome"] as? String == "done" }
            var o: [String: Any] = ["ok": allFinished, "grid": "g\(g)", "sheet": sheet.map { "i\($0)" } ?? NSNull(), "jobs": list]
            if !allFinished {
                o["error"] = "render"
                o["message"] = String(localized: "at least one cell did not render: see \"outcome\" in \"jobs\"; the sheet marks it empty")
            }
            finish(c, o)
        }
        for n in jobs {
            guard let t = app.completed[n] else { continue }
            for k in t.images { if let e = app.history.first(where: { $0.ordinal == k }) { emitImage(e, folder: folder, c) } }
            tallies[n] = Self.json(t.issue, job: n, images: t.images)
        }
        if app.drawingSheets[g] == nil, let ended = app.endedGrids[g] {
            if let s = ended.sheet, let e = app.history.first(where: { $0.ordinal == s }) {
                emitImage(e, folder: folder, type: "sheet", c)
            }
            verdict(sheet: ended.sheet)
            return
        }
        var token: UUID?
        token = app.subscribe { [weak self] signal in
            guard let self else { return }
            if c.broken { if let token { self.app.unsubscribe(token) }; return }
            switch signal {
            case .image(let t, let e) where jobs.contains(t.ordinal):
                self.emitImage(e, folder: folder, c)
            case .finished(let t, let issue) where jobs.contains(t.ordinal):
                tallies[t.ordinal] = Self.json(issue, job: t.ordinal, images: self.app.completed[t.ordinal]?.images ?? [])
            case .sheet(let run, let entry) where run.ordinal == g:
                if let entry { self.emitImage(entry, folder: folder, type: "sheet", c) }
                if let token { self.app.unsubscribe(token) }
                verdict(sheet: entry?.ordinal)
            default:
                break
            }
        }
    }

    /// One `"type": "image"` line (or `"sheet"`), the PNG written into `folder` if given.
    private func emitImage(_ e: Entry, folder: String?, type: String = "image", _ c: Connection) {
        var path: String?
        var o: [String: Any] = [:]
        if let folder {
            do { path = try write(e, in: folder) } catch { o["write_error"] = "\(error)" }
        }
        o.merge(json(e, path: path)) { a, _ in a }
        o["type"] = type
        emit(c, o)
    }

    // ── wait ──

    private func wait(_ arguments: [String], cwd: String, _ c: Connection) throws {
        var ordinals: [Int] = []
        var grids: [Int] = []
        var folder: String?
        var i = 0
        while i < arguments.count {
            let a = arguments[i]
            if a == "--out" {
                guard i + 1 < arguments.count else { throw Refusal("usage", String(localized: "--out expects a folder")) }
                folder = absolute(arguments[i + 1], cwd)
                i += 2
                continue
            }
            // `g2`: a grid, followed to its sheet.
            if a.first == "g", let n = Int(a.dropFirst()) {
                guard app.gridRuns[n] != nil || app.drawingSheets[n] != nil || app.endedGrids[n] != nil else {
                    throw Refusal("unknown_job", String(localized: "no grid \(a) in this session"), help: "silicontrol status")
                }
                grids.append(n)
                i += 1
                continue
            }
            guard let n = Self.ordinal(a, prefix: "t") else {
                throw Refusal("usage", String(localized: "job expected (t3), not \"\(a)\""))
            }
            guard app.completed[n] != nil || app.currentJob?.ordinal == n
                    || app.file.contains(where: { $0.ordinal == n })
                    || app.awaitingLicense.contains(where: { $0.ordinal == n }) else {
                throw Refusal("unknown_job", String(localized: "no job \(a) in this session"), help: "silicontrol status")
            }
            ordinals.append(n)
            i += 1
        }
        if !grids.isEmpty {
            guard grids.count == 1, ordinals.isEmpty else {
                throw Refusal("usage", String(localized: "wait follows one grid alone: wait g2 [--out D]"))
            }
            if let folder { try createFolder(folder) }
            followGrid(grids[0], folder: folder, c)
            return
        }
        if ordinals.isEmpty {
            // With no job named: everything running or waiting now.
            ordinals = (app.currentJob.map { [$0.ordinal] } ?? []) + app.file.map(\.ordinal) + app.awaitingLicense.map(\.ordinal)
        }
        if let folder { try createFolder(folder) }
        followToEnd(ordinals, folder: folder, c)
    }

    /// Emits each image (written into `folder` if given), then the verdict when all the jobs
    /// are done. Those already done answer at once.
    private func followToEnd(_ ordinals: [Int], folder: String?, _ c: Connection) {
        var remainingSet = Set(ordinals)
        var tallies: [Int: [String: Any]] = [:]

        func image(_ e: Entry) { emitImage(e, folder: folder, c) }
        func tally(_ n: Int, _ issue: Issue, _ images: [Int]) {
            tallies[n] = Self.json(issue, job: n, images: images)
            remainingSet.remove(n)
        }
        func verdict() {
            let jobs = ordinals.compactMap { tallies[$0] }
            let allFinished = jobs.allSatisfy { $0["outcome"] as? String == "done" }
            var o: [String: Any] = ["ok": allFinished, "jobs": jobs]
            if !allFinished {
                o["error"] = "render"
                o["message"] = String(localized: "at least one job did not finish: see \"outcome\" in \"jobs\"")
            }
            finish(c, o)
        }

        for n in ordinals {
            guard let t = app.completed[n] else { continue }
            for k in t.images { if let e = app.history.first(where: { $0.ordinal == k }) { image(e) } }
            tally(n, t.issue, t.images)
        }
        if remainingSet.isEmpty { verdict(); return }

        var token: UUID?
        token = app.subscribe { [weak self] signal in
            guard let self else { return }
            if c.broken { if let token { self.app.unsubscribe(token) }; return }
            switch signal {
            case .image(let t, let e) where remainingSet.contains(t.ordinal):
                image(e)
            case .finished(let t, let issue) where remainingSet.contains(t.ordinal):
                tally(t.ordinal, issue, self.app.completed[t.ordinal]?.images ?? [])
                if remainingSet.isEmpty {
                    if let token { self.app.unsubscribe(token) }
                    verdict()
                }
            default:
                break
            }
        }
    }

    // ── follow ──

    private func follow(_ c: Connection) {
        guard app.busy || !app.awaitingLicense.isEmpty else {
            finish(c, ["ok": true, "message": String(localized: "nothing is running or waiting")])
            return
        }
        var token: UUID?
        // Idle once nothing is queued, rendering, waiting **or drawing a sheet**: the last cell of
        // a grid ends before its sheet is drawn (`finishGrid`), and `follow` promises the sheet.
        func quitIfIdle() {
            guard app.file.isEmpty, app.currentJob == nil, app.awaitingLicense.isEmpty,
                  app.gridRuns.isEmpty, app.drawingSheets.isEmpty else { return }
            if let token { app.unsubscribe(token) }
            finish(c, ["ok": true, "message": String(localized: "queue empty")])
        }
        token = app.subscribe { [weak self] signal in
            guard let self else { return }
            if c.broken { if let token { self.app.unsubscribe(token) }; return }
            switch signal {
            case .started(let t):
                var o = self.json(t); o["type"] = "start"; self.emit(c, o)
            case let .stage(t, item, image):
                self.emit(c, ["type": "stage", "job": t.name, "stage": Self.stageName(item), "batch_image": image + 1])
            case let .step(t, image, index, total, seconds):
                self.emit(c, ["type": "step", "job": t.name, "batch_image": image + 1, "step": index, "steps_total": total,
                                 "seconds": Response.number(seconds)])
            case let .image(_, e):
                var o = self.json(e); o["type"] = "image"; self.emit(c, o)
            case let .sheet(run, entry):
                var o: [String: Any] = entry.map { self.json($0) } ?? ["grid": run.name, "image": NSNull()]
                o["type"] = "sheet"
                self.emit(c, o)
                quitIfIdle()
            case let .finished(t, issue):
                var o = Self.json(issue, job: t.ordinal, images: self.app.completed[t.ordinal]?.images ?? [])
                o["type"] = "end"
                self.emit(c, o)
                // `.finished` goes out before the next one is started: the queue says whether any is left.
                quitIfIdle()
            }
        }
    }

    // ── cancel, clear ──

    private func cancel(_ arguments: [String]) throws -> [String: Any] {
        var stopped: String?, removedNames: [String] = [], alreadyFinished: [String] = []
        let ordinals = try arguments.map { a -> Int in
            guard let n = Self.ordinal(a, prefix: "t") else { throw Refusal("usage", String(localized: "job expected (t3), not \"\(a)\"")) }
            return n
        }
        if ordinals.isEmpty {
            if let t = app.currentJob { app.cancel(); stopped = t.name }
        }
        for n in ordinals {
            if app.currentJob?.ordinal == n {
                app.cancel(); stopped = "t\(n)"
            } else if let t = (app.file + app.awaitingLicense).first(where: { $0.ordinal == n }) {
                app.remove(t); removedNames.append(t.name)
            } else if app.completed[n] != nil {
                alreadyFinished.append("t\(n)")
            } else {
                throw Refusal("unknown_job", String(localized: "no job t\(n) in this session"), help: "silicontrol status")
            }
        }
        return ["ok": true, "stopped": stopped ?? NSNull(), "removed": removedNames, "already_finished": alreadyFinished]
    }

    private func clear() -> [String: Any] {
        let stopped = app.currentJob?.name
        let removedNames = (app.file + app.awaitingLicense).map(\.name)
        app.stopAll()
        return ["ok": true, "stopped": stopped ?? NSNull(), "removed": removedNames]
    }

    // ── history, save ──

    private func history(_ arguments: [String]) throws -> [String: Any] {
        var n = Int.max
        if let a = arguments.first {
            guard let v = Int(a), v >= 1 else { throw Refusal("usage", String(localized: "history [N]: N an integer ≥ 1, not \"\(a)\"")) }
            n = v
        }
        return ["ok": true, "images": app.history.prefix(n).map { json($0) }]
    }

    private func save(_ arguments: [String], cwd: String) throws -> [String: Any] {
        guard arguments.count >= 2, let destination = arguments.last else {
            throw Refusal("usage", "save <i7> [i8 …] <file.png|file.jpg|folder>")
        }
        let entries = try arguments.dropLast().map { a -> Entry in
            guard let e = image(a) else {
                throw Refusal("unknown_image", String(localized: "no image \"\(a)\" in the history"), help: "silicontrol history")
            }
            return e
        }
        let target = absolute(destination, cwd)
        let ext = (target as NSString).pathExtension.lowercased()
        var files: [[String: Any]] = []
        if entries.count == 1, ["png", "jpg", "jpeg"].contains(ext) {
            // A named file: written, or replaced — it is « Save as ».
            try createFolder((target as NSString).deletingLastPathComponent)
            try writeFile(entries[0], target)
            files.append(["image": entries[0].name, "path": target])
        } else {
            try createFolder(target)
            for e in entries { files.append(["image": e.name, "path": try write(e, in: target)]) }
        }
        return ["ok": true, "files": files]
    }

    // ── vary ──

    /// **`vary i7 subtle|strong [--batch N]`**: « Variations » on a history image, as its menu does — N
    /// jobs (the rack's « Images » by default) with its settings and seed, each turned by a variation
    /// seed drawn at random.
    private func vary(_ arguments: [String]) throws -> [String: Any] {
        var count = app.batch
        var rest = arguments
        if rest.count == 4, rest[2] == "--batch" {
            guard let n = Int(rest[3]), (1...8).contains(n) else {
                throw Refusal("usage", String(localized: "--batch expects an integer from 1 to 8, not \"\(rest[3])\""))
            }
            count = n
            rest.removeLast(2)
        }
        guard rest.count == 2, let amount = Variation.Amount(rawValue: rest[1]) else {
            throw Refusal("usage", "vary <i7> subtle|strong [--batch N]")
        }
        let arguments = rest
        guard let e = image(arguments[0]) else {
            throw Refusal("unknown_image", String(localized: "no image \"\(arguments[0])\" in the history"), help: "silicontrol history")
        }
        if let reason = app.installBlocker { throw Refusal("busy", reason, help: String(localized: "retry once the installation is done (silicontrol models)")) }
        guard app.canVary(e) else {
            throw Refusal("model", String(localized: "\(e.name) has no variations: a grid's sheet, or its model is not ready"),
                          help: "silicontrol models")
        }
        guard app.file.count + count <= AppState.maxQueueLength else {
            throw Refusal("queue_full", String(localized: "the queue is full (\(AppState.maxQueueLength) jobs)"),
                          help: "silicontrol wait, then retry")
        }
        return ["ok": true, "jobs": app.vary(e, amount, count: count).map { json($0) }]
    }

    // ── Tools ──

    /// `t3` → 3; `3` too.
    private static func ordinal(_ text: String, prefix: Character) -> Int? {
        Int(text.first == prefix ? String(text.dropFirst()) : text)
    }

    private func image(_ name: String) -> Entry? {
        guard let n = Self.ordinal(name, prefix: "i"), name.first == "i" else { return nil }
        return app.history.first { $0.ordinal == n }
    }

    private func createFolder(_ path: String) throws {
        do { try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true) } catch {
            let detail = error.localizedDescription
            throw Refusal("write", String(localized: "folder cannot be created: \(path) (\(detail))"))
        }
    }

    /// In a folder, under the app's export name (`AppState.exportName`: the model, the LoRAs, the steps
    /// if they are not the model's, the format, the seed). Never over an existing file (`-2`, `-3`…).
    private func write(_ e: Entry, in folder: String) throws -> String {
        let path = AppState.freeURL(in: URL(fileURLWithPath: folder), base: app.exportName(e), ext: "png").path
        try writeFile(e, path)
        return path
    }

    private func writeFile(_ e: Entry, _ path: String) throws {
        do { try app.write(e, to: URL(fileURLWithPath: path)) } catch {
            throw Refusal("write", "\(path): \(error.localizedDescription)")
        }
    }
}
