// **`silicontrol` — the app's command-line remote control**, shipped inside `Siliconed.app`
// (`Contents/Helpers/silicontrol`; the app's menu links it into the PATH).
//
// A pipe, and nothing more: it sends its arguments verbatim to the running app — one JSON line,
// `{"argv": […], "cwd": "…"}`, on the socket `<library>/silicontrol.sock` — and copies each JSON line
// it gets back to stdout. Parsing and checks: all in the app (`RemoteControl.swift`), so the
// command cannot say anything the app does not do, and it does not link the engine.
//
// Two things are handled here. `open` launches the app. The help (`help [topic]`, `<command> --help`,
// no argument) is answered without it — reading a manual must not open a window — from the module
// `SilicontrolHelp`, the same pages and the same routing the app answers on its socket. Any other
// command launches the app when it is not running, then waits for its socket (30 s at most).
//
// Exit codes: 0 the last line says `"ok": true` · 1 it says `"ok": false` · 2 usage ·
// 3 the app cannot be reached.

import Foundation
import SilicontrolHelp

let environment = ProcessInfo.processInfo.environment

/// The library the app opens: `SILICONED_ROOT`, else its own space in Application Support.
let library: String = {
    if let root = environment["SILICONED_ROOT"], !root.isEmpty { return root }
    let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    return support.appendingPathComponent("Siliconed").path
}()
let socketPath = library + "/silicontrol.sock"

func printJSON(_ object: [String: Any]) {
    let data = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])) ?? Data()
    print(String(decoding: data, as: UTF8.self))
}

func fail(_ code: String, _ message: String, hint: String? = nil, exit status: Int32 = 3) -> Never {
    var o: [String: Any] = ["ok": false, "error": code, "message": message]
    if let hint { o["hint"] = hint }
    printJSON(o)
    exit(status)
}

func connect() -> Int32? {
    var address = sockaddr_un()
    address.sun_family = sa_family_t(AF_UNIX)
    let bytes = Array(socketPath.utf8)
    guard bytes.count < MemoryLayout.size(ofValue: address.sun_path) else { return nil }
    withUnsafeMutableBytes(of: &address.sun_path) { $0.copyBytes(from: bytes); $0[bytes.count] = 0 }
    address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { return nil }
    let r = withUnsafePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
    }
    guard r == 0 else { close(fd); return nil }
    var yes: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))
    return fd
}

/// Sends `argv`, streams every line received (a `"help"` field prints as raw text), returns the
/// last one. `nil`: nobody answers.
@discardableResult
func send(_ argv: [String], echo: Bool = true) -> [String: Any]? {
    guard let fd = connect() else { return nil }
    defer { close(fd) }
    var request = (try? JSONSerialization.data(withJSONObject: ["argv": argv, "cwd": FileManager.default.currentDirectoryPath])) ?? Data()
    request.append(0x0A)
    guard request.withUnsafeBytes({ write(fd, $0.baseAddress!, $0.count) }) == request.count else { return nil }

    var pending = Data(), last: [String: Any]?
    var buffer = [UInt8](repeating: 0, count: 65536)
    func consume(_ line: Data) {
        guard !line.isEmpty else { return }
        let object = (try? JSONSerialization.jsonObject(with: line)) as? [String: Any]
        if echo {
            if let help = object?["help"] as? String { print(help) } else { print(String(decoding: line, as: UTF8.self)) }
        }
        last = object
    }
    while true {
        let n = read(fd, &buffer, buffer.count)
        if n < 0, errno == EINTR { continue }
        guard n > 0 else { break }
        pending.append(contentsOf: buffer[0..<n])
        while let end = pending.firstIndex(of: 0x0A) {
            consume(pending[pending.startIndex..<end])
            pending = Data(pending[pending.index(after: end)...])
        }
    }
    consume(pending)
    return last
}

/// The `Siliconed.app` this command lives in (`Siliconed.app/Contents/Helpers/silicontrol`, possibly
/// through a symlink in the PATH) — so it never opens some other, older copy.
func ownBundle() -> String? {
    var url = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    if !CommandLine.arguments[0].contains("/"), let path = Bundle.main.executablePath {
        url = URL(fileURLWithPath: path).resolvingSymlinksInPath()
    }
    while url.path != "/" {
        if url.pathExtension == "app" { return url.path }
        url.deleteLastPathComponent()
    }
    return nil
}

/// Launches the app in the background if it does not answer, and waits for its socket.
/// Returns whether it was already running.
@discardableResult
func ensureRunning() -> Bool {
    if let fd = connect() { close(fd); return true }
    let open = Process()
    open.executableURL = URL(fileURLWithPath: "/usr/bin/open")
    var arguments = ["-g"]   // without stealing the foreground
    if let root = environment["SILICONED_ROOT"], !root.isEmpty { arguments += ["--env", "SILICONED_ROOT=\(root)"] }
    arguments += ownBundle().map { [$0] } ?? ["-b", "dev.gwenn-ha.siliconed"]
    open.arguments = arguments
    do { try open.run() } catch { fail("app_not_found", "cannot run /usr/bin/open: \(error)") }
    open.waitUntilExit()
    guard open.terminationStatus == 0 else {
        fail("app_not_found", "could not launch Siliconed.app (open exited with \(open.terminationStatus))")
    }
    for _ in 0..<60 {
        if let fd = connect() { close(fd); return false }
        usleep(500_000)
    }
    // Two causes look alike from here, and neither can be told apart from this side: the open
    // app could not open its socket (its Siliconed menu says why), or the app launched predates the
    // remote control. The message names both rather than guessing.
    fail("app_not_running", "the app does not answer on \(socketPath) after 30 s",
         hint: "if Siliconed is open, its Siliconed menu says why its remote control is off; "
             + "otherwise the Siliconed.app installed may predate the remote control")
}

let argv = Array(CommandLine.arguments.dropFirst())
if let answer = RemoteControlHelp.answer(argv) {
    // Printed as `send` prints the app's answer: a page raw, a refusal as its JSON line.
    if let help = answer["help"] as? String { print(help); exit(0) }
    printJSON(answer)
    exit(2)
}
if argv.first == "open" {
    let already = ensureRunning()
    printJSON(["ok": true, "message": already ? "Siliconed was already open" : "Siliconed open", "socket": socketPath])
    exit(0)
}
ensureRunning()
guard let last = send(argv) else {
    fail("app_not_running", "the app closed the connection without an answer (\(socketPath))")
}
if last["ok"] as? Bool != true { exit(last["error"] as? String == "usage" ? 2 : 1) }
