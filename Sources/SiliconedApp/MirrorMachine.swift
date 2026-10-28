// **The lending Mac's machine, its map and its diagnostic — said to the Macs that borrow it, and read
// by them** (`Remote/Machine.swift`). Both sides here; the protocol is in `Mirror.swift`.
//
//     lender                                          borrower
//       MachinePublisher: watches the map, the          receiveMachine: the sheet, the rack's chip, the
//       diagnostic, B, the ready models; sends           sidebar's warnings read them as if local
//       `machine` at each join and on each change      OtherMacs.machines: the last one of each Mac,
//       `measure`, `stopMeasure` ◀── the sheet's           kept for the « On: » menu and the Settings
//       `setFasterRendering` ◀── Settings, silicontrol
//
// - **The measure runs on the lender**, in its child process, as its own button would start it: refused
//   (`machineBusy`) while it renders, installs or measures — it is never queued behind a render, the
//   local sheet does not either. Its progress is a rank among `Diagnostic.Stage.allCases`, said here in
//   this Mac's language; its report crosses whole, and « Copy JSON » or « Open the Issue » on the
//   borrower carry the lender's report (it is that machine's).
// - **« Faster Rendering » is the lender's**: `Spectral.halfSizeStart` is its process's. The borrower's
//   switch reads the catalog's value and writes by `setFasterRendering`; the local preference is not touched.
// - **Nothing is judged here**: the borrower's `diagnosticBlocker`, `mapStatus` and chip read what was
//   published; the lender judges again what comes (`handle`).

import Foundation
import Siliconed
import SwiftUI

// MARK: - Lending side

@MainActor
final class MachinePublisher {
    private let app: AppState
    private let host: RemoteHost
    private var last: RemoteMachine?
    /// Read once: it touches Metal.
    private lazy var machine = Diagnostic.machine()

    init(app: AppState, host: RemoteHost) {
        self.app = app; self.host = host
        watch()
    }

    func snapshot() -> RemoteMachine {
        let phase: RemoteDiagnostic = switch app.diagnostic {
        case .ready: .idle
        case .running(let done, let total, _): .running(done: done, total: total)
        case .finished(let report): .finished(report)
        case .failed(let problem): .failed(RemoteRefusal(problem))
        }
        return RemoteMachine(
            chip: machine.chip, memoryBytes: machine.memoryBytes, ceiling: app.memoryCeiling, ready: app.readySet.sorted(),
            stored: app.storedMap.map(RemoteMap.init), inUse: app.mapInUse.map(RemoteMap.init),
            remeasuring: app.remeasuringProvisional, warnings: app.profileWarnings, diagnostic: phase,
            blocker: app.diagnosticBlock)
    }

    func joined(_ identity: Data) {
        let m = snapshot()
        last = m
        host.send(.machine(m), to: identity)
    }

    private func watch() {
        withObservationTracking { _ = snapshot() } onChange: { [weak self] in
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.fire() } }
        }
    }

    private func fire() {
        let m = snapshot()
        if m != last {
            last = m
            host.send(.machine(m))
        }
        watch()
    }

    /// What a borrower asks of this machine, judged as the local button or `silicontrol set` would be.
    func handle(_ m: SessionMessage, from identity: Data) {
        switch m {
        case .measure:
            guard app.diagnosticBlock == nil else { host.send(.refused(.app(.machineBusy)), to: identity); return }
            app.startDiagnostic()
        case .stopMeasure:
            app.cancelDiagnostic()
        case .setFasterRendering(let on):
            // The engine reads it at every plan: not while a render, an installation or the measure runs.
            guard !app.busy, app.forgeInProgress == nil, !app.diagnosticRunning, app.awaitingLicense.isEmpty else {
                host.send(.refused(.app(.machineBusy)), to: identity)
                return
            }
            app.fasterRendering = on
        default:
            break
        }
    }
}

// MARK: - Borrowing side

extension AppState {
    func receiveMachine(_ m: RemoteMachine) {
        mirrorMachine = m
        profileWarnings = m.warnings
        diagnostic = switch m.diagnostic {
        case .idle: .ready
        case .running(let done, let total): .running(done: done, total: total, line: Self.diagnosticLine(done))
        case .finished(let report): .finished(report)
        case .failed(let r): .failed(Problem(r))
        }
    }

    /// The stage under way, in this Mac's language.
    static func diagnosticLine(_ rank: Int) -> String {
        let stages = Diagnostic.Stage.allCases
        return diagnosticLine(stages[min(max(rank, 0), stages.count - 1)])
    }

    /// **A measure refused by the lender** while this sheet showed it starting: back to « ready ».
    func measureRefused(_ r: RemoteRefusal) {
        guard case .app(.machineBusy) = r, diagnosticRunning, mirrorMachine?.diagnostic.isRunning != true else { return }
        diagnostic = .ready
    }

    /// **The map, as the sheet and the settings say it**: this Mac's own, or the lender's as it published it.
    var mapFacts: MapFacts {
        if mirror != nil {
            return MapFacts(inUse: mirrorMachine?.inUse, stored: mirrorMachine?.stored, remeasuring: mirrorMachine?.remeasuring ?? false)
        }
        return MapFacts(inUse: mapInUse.map(RemoteMap.init), stored: storedMap.map(RemoteMap.init), remeasuring: remeasuringProvisional)
    }

    /// A current map taken at rest is kept (the chip of « Chain » is green).
    var mapMeasured: Bool { mapFacts.stored.map { $0.current && !$0.provisional } == true }

    /// This report is the map the settings were read with.
    func inUse(_ report: Diagnostic.Report) -> Bool {
        mirror != nil ? mirrorMachine?.inUse?.date == report.date : report == mapInUse
    }

    /// **« Faster Rendering »** as the window shows it: this Mac's, or the lender's in a mirror.
    var fasterRenderingShown: Bool { mirror != nil ? mirrorCatalog?.fasterRendering ?? true : fasterRendering }

    /// The switch of the lender's, from a mirror; the new value comes back in its catalog.
    func setFasterRenderingOnLender(_ on: Bool) { mirror?.send(.setFasterRendering(on)) }
}

struct MapFacts: Equatable {
    var inUse: RemoteMap?
    var stored: RemoteMap?
    var remeasuring: Bool
}

extension RemoteDiagnostic {
    var isRunning: Bool { if case .running = self { true } else { false } }
}

extension RemoteMachine {
    /// What the « On: » menu and Settings › Other Macs say of a Mac: chip, memory, B, map, models.
    var summary: String {
        [chip, Figures.memory(memoryBytes), String(localized: "Budget \(Figures.memory(ceiling))"),
         measured ? String(localized: "Measured") : String(localized: "Not measured yet"),
         String(localized: "Models ready: \(ready.count)")].joined(separator: " · ")
    }
}

extension RemoteBlocker {
    var sentence: String {
        switch self {
        case .measuring: String(localized: "A diagnostic is already running.")
        case .rendering: String(localized: "A render is running: the diagnostic can start once the queue is empty.")
        case .installing: String(localized: "An installation is running: the diagnostic can start once it is done.")
        }
    }
}

// MARK: - silicontrol

extension RemoteControl {
    /// **`set faster-rendering on|off` on a mirror**: asked of the lender, answered once its catalog says so
    /// (it refuses while it renders, installs or measures: no answer then, and the command says it).
    func setOnMirror(_ mirror: MirrorLink, _ arguments: [String], _ c: Connection) throws {
        guard arguments.count == 2, arguments[0] == "faster-rendering", ["on", "off"].contains(arguments[1]) else {
            throw Refusal("usage", "set faster-rendering on|off")
        }
        if let reason = app.mirrorMachine?.blocker?.sentence {
            throw Refusal("busy", reason, help: "silicontrol wait, then retry")
        }
        let on = arguments[1] == "on", previous = app.fasterRenderingShown
        func done() -> [String: Any] {
            ["ok": true, "faster_rendering": on, "previous": previous, "computing_on": mirror.macName]
        }
        guard previous != on else { finish(c, done()); return }
        mirror.send(.setFasterRendering(on))
        Task { @MainActor [weak self] in
            for _ in 0..<50 {
                try? await Task.sleep(for: .milliseconds(100))
                guard let self, !c.broken else { return }
                if self.app.fasterRenderingShown == on { self.finish(c, done()); return }
            }
            self?.finish(c, Response.failure("busy", String(localized: "a render, an installation or the diagnostic is running"),
                                             help: "silicontrol wait, then retry"))
        }
    }
}
