// **Where it computes and where the models live, from the terminal**: `macs`, `compute`, `library`. Compute On is the menu's toggle
// (`MainWindow.compute(on:)`); `macs connect|disconnect` are Settings › Other Macs' buttons (`OtherMacs.connect`, `disconnect`); `library`
// is Settings › General's (`AppState.moveLibrary`, `moveLibraryHome`, `openLibrary`, `useLocalLibrary`). Never a pairing, a loan or a
// forgetting: those stay in the window, on purpose — a pairing is accepted by the person in front of both screens.

import AppKit
import Foundation
import Siliconed

extension RemoteControl {
    private var otherMacs: OtherMacs { MainWindow.shared.otherMacs }

    /// The paired Macs we compute on (`Pairings.borrowed`), as the Compute On menu lists them. Empty until the Keychain is read, which
    /// the app does at launch when a Mac was paired and otherwise never without the user (a read can ask for a password).
    private var borrowed: [Pairings.Borrowed] { otherMacs.pairings?.borrowed ?? [] }

    /// A link's state as a stable word.
    private func linkState(_ b: Pairings.Borrowed) -> String {
        switch otherMacs.links[b.id] {
        case nil, .off?: otherMacs.found[b.id] != nil ? "on_the_network" : "not_on_the_network"
        case .connecting?: "connecting"
        case .connected?: "connected"
        case .waiting?: "lost"
        case .refused(.forgotten)?: "refused_forgotten"
        case .refused(.mismatch)?: "refused_version"
        }
    }

    private func json(_ b: Pairings.Borrowed) -> [String: Any] {
        var o: [String: Any] = ["name": b.name, "id": b.id, "state": linkState(b),
                                "computing": MainWindow.shared.carrier.showing == b.id]
        if let machine = otherMacs.machines[b.id] { o["machine"] = machine.summary }
        return o
    }

    private func mac(named word: String) throws -> Pairings.Borrowed {
        guard let found = MacName.match(word, among: borrowed.map { ($0.id, $0.name) }), let b = borrowed.first(where: { $0.id == found.id }) else {
            let names = borrowed.map(\.name).joined(separator: ", ")
            throw Refusal("unknown_mac", String(localized: "no paired Mac called \"\(word)\""),
                          help: names.isEmpty ? String(localized: "pair a Mac in Settings › Other Macs, in the window") : String(localized: "paired: \(names)"))
        }
        return b
    }

    /// **`macs`**: the Macs paired to compute on and their state, the ones that borrow from this one, where the window computes.
    /// **`macs connect|disconnect NAME`**: Settings › Other Macs' buttons — the session kept or dropped.
    func macs(_ action: GestureOrder.MacsAction) throws -> [String: Any] {
        switch action {
        case .list:
            return ["ok": true, "computing_on": app.mirror?.macName ?? NSNull(),
                    "macs": borrowed.map { json($0) },
                    "lending": otherMacs.lending, "lent_to": (otherMacs.pairings?.lent ?? []).map { ["name": $0.name, "connected": otherMacs.connected[$0.identity] != nil] },
                    "pairings_read": otherMacs.pairings != nil]
        case .connect(let word):
            let b = try mac(named: word)
            // The button is « Connect » only when there is no session.
            if otherMacs.links[b.id] == nil { otherMacs.connect(b) }
            return ["ok": true, "mac": json(b)]
        case .disconnect(let word):
            let b = try mac(named: word)
            if otherMacs.links[b.id] != nil { otherMacs.disconnect(b.id) }
            return ["ok": true, "mac": json(b)]
        }
    }

    /// **`compute NAME|this`**: Render › Compute On. The window shows the other Mac's queue and history from then on, and every command follows it
    /// (the contract is in `REMOTE-CONTROL.md`). Answered once the session is up, lost, or refused — or after 15 s: the switch is made either way,
    /// as the menu makes it, and the window says it is connecting.
    func compute(_ target: GestureOrder.ComputeTarget, _ c: Connection) throws {
        let window = MainWindow.shared
        guard case .mac(let word) = target else {
            let was = app.mirror?.macName
            window.compute(on: nil)
            finish(c, ["ok": true, "computing_on": NSNull(), "changed": was != nil])
            return
        }
        let b = try mac(named: word)
        let already = window.carrier.showing == b.id
        window.compute(on: b)
        Task {
            let deadline = Date().addingTimeInterval(15)
            while !already, Date() < deadline, !c.broken {
                switch otherMacs.links[b.id] {
                case .connected?, .refused?: break
                default:
                    try? await Task.sleep(for: .milliseconds(200))
                    continue
                }
                break
            }
            let state = linkState(b)
            var o: [String: Any] = ["ok": true, "computing_on": b.name, "changed": !already, "link": state]
            if state != "connected" {
                o["message"] = String(localized: "the window computes on “\(b.name)” and is still joining it: see macs")
            }
            if state.hasPrefix("refused") {
                finish(c, Response.failure("link", String(localized: "“\(b.name)” refused the session"), help: "silicontrol macs"))
                return
            }
            finish(c, o)
        }
    }

    // MARK: - The library

    private func libraryFacts() -> [String: Any] {
        let l = app.library
        var o: [String: Any] = ["path": l.root.path, "elsewhere": l.isElsewhere, "reachable": app.libraryReachable,
                                "this_mac": l.local.path]
        if let volume = l.volumeName { o["volume"] = volume }
        if let free = l.freeSpace() { o["free_bytes"] = free }
        if app.libraryReachable { o["used_bytes"] = l.occupancy().total }
        return o
    }

    /// **`library`**: where it is. **`library move DIR [--yes]`** is « Move Library… », **`home [--yes]`** « Move Back to This Mac »: the files copied
    /// and verified one by one, the old place emptied last. Without `--yes`, the answer says what would be copied and the room there is.
    /// **`open DIR`** is « Open Library… » (nothing moved), **`local`** « Use This Mac’s Library ». Refused as Settings greys them out — a render,
    /// an installation or the measure running (`busy`), the disk absent (`library_absent`) — and on a Mac followed (`remote`): the library
    /// is the lender’s there.
    func library(_ action: GestureOrder.LibraryAction, confirmed: Bool, cwd: String, _ c: Connection) throws {
        func blocked(moving: Bool) throws {
            if let mirror = app.mirror, moving {
                throw Refusal("remote", String(localized: "the library shown is that of “\(mirror.macName)”"), help: String(localized: "Render › Compute On › This Mac, then retry"))
            }
            if moving, let reason = app.libraryBlocker { throw Refusal("library_absent", reason, help: String(localized: "connect the disk, then retry")) }
            let free = moving ? app.managementPossible : app.forgeInProgress == nil && !app.busy && !app.diagnosticRunning
            guard free else {
                throw Refusal("busy", String(localized: "a render, an installation or the diagnostic is running"), help: "silicontrol wait, then retry")
            }
        }
        switch action {
        case .status:
            finish(c, ["ok": true, "library": libraryFacts()])
        case .open(let path):
            try blocked(moving: false)
            let url = URL(fileURLWithPath: absolute(path, cwd), isDirectory: true)
            if let p = app.openLibrary(at: url) { throw Refusal(p.code ?? "library", p.text) }
            finish(c, ["ok": true, "library": libraryFacts()])
        case .local:
            try blocked(moving: false)
            if let p = app.useLocalLibrary() { throw Refusal(p.code ?? "library", p.text) }
            finish(c, ["ok": true, "library": libraryFacts()])
        case .home, .move:
            try blocked(moving: true)
            let destination: URL
            if case .move(let path) = action {
                destination = LibraryMove.destination(in: URL(fileURLWithPath: absolute(path, cwd), isDirectory: true))
            } else {
                guard app.libraryIsElsewhere else { throw Refusal("library", String(localized: "the library is already on this Mac")) }
                destination = app.library.local
            }
            let plan: LibraryMove.Plan
            do { plan = try LibraryMove.plan(app.library, to: destination) } catch {
                let p = Problem(error)
                throw Refusal(p.code ?? "library", p.text)
            }
            var summary: [String: Any] = ["destination": destination.path, "files": plan.pending.count, "bytes": plan.bytes]
            if let room = plan.room { summary["free_bytes"] = room }
            guard confirmed else {
                summary["ok"] = true
                summary["dry_run"] = true
                summary["message"] = String(localized: "nothing was moved: add --yes to move it")
                finish(c, summary)
                return
            }
            let before = app.library.root.path
            app.problem = nil
            app.moveLibrary(to: destination)
            emit(c, ["type": "started", "destination": destination.path, "files": plan.pending.count, "bytes": plan.bytes])
            Task {
                var said = -1
                while app.forgeInProgress != nil, !c.broken {
                    if let row = app.forgeJournal.last(where: { $0.hasPrefix("⇢ ") }),
                       let percent = row.dropFirst(2).split(separator: " ").first.flatMap({ Int($0) }), percent != said {
                        said = percent
                        emit(c, ["type": "progress", "percent": percent])
                    }
                    try? await Task.sleep(for: .milliseconds(500))
                }
                if app.library.root.path == before {
                    // Stopped, or refused along the way: the library stays where it was.
                    let p = app.problem
                    finish(c, Response.failure(p?.code ?? "stopped", p?.text ?? String(localized: "the move was stopped: the library stays where it was")))
                    return
                }
                finish(c, ["ok": true, "library": libraryFacts(), "moved": ["files": plan.pending.count, "bytes": plan.bytes]])
            }
        }
    }
}
