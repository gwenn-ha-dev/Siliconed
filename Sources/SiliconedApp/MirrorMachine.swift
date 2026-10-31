// **The lending Mac's machine, its map and its diagnostic — said to the Macs that borrow it, and read
// by them** (`Remote/Machine.swift`). Both sides here; the protocol is in `Mirror.swift`.
//
//     lender                                          borrower
//       MachinePublisher: watches the map, the          receiveMachine: the sheet, the rack's chip, the
//       diagnostic, B, the ready models; sends           sidebar's warnings read them as if local
//       `machine` at each join and on each change      OtherMacs.machines: the last one of each Mac,
//       `measure`, `stopMeasure` ◀── the sheet's           kept for the « On: » menu and the Settings
//
// - **The measure runs on the lender**, in its child process, as its own button would start it: refused
//   (`machineBusy`) while it renders, installs or measures — it is never queued behind a render, the
//   local sheet does not either. Its progress is a rank among `Diagnostic.Stage.allCases`, said here in
//   this Mac's language; its report crosses whole, and « Copy JSON » or « Open a GitHub Issue » on the
//   borrower carry the lender's report (it is that machine's).
// - **« Faster Rendering » is the borrower's**: it rides on each job (`RemoteJob.fasterRendering`, `Request.fasterRendering`),
//   so the lender's own preference is never touched and a redo renders the same image.
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
            chip: machine.chip, hardwareModel: machine.hardwareModel, memoryBytes: machine.memoryBytes, ceiling: app.memoryCeiling,
            ready: app.readySet.sorted(),
            stored: app.storedMap.map(RemoteMap.init), inUse: app.mapInUse.map(RemoteMap.init),
            warnings: app.profileWarnings, diagnostic: phase,
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
            return MapFacts(inUse: mirrorMachine?.inUse, stored: mirrorMachine?.stored)
        }
        return MapFacts(inUse: mapInUse.map(RemoteMap.init), stored: storedMap.map(RemoteMap.init), drift: drift)
    }

    /// A map of this Mac and this build is kept (the Mac beside « Chain » is in colour): whatever it found, it is the map.
    var mapMeasured: Bool { mapFacts.stored?.current == true }

    /// `hw.model` of the Mac whose measure plans the render — this one, or the lender in a mirror (`nil` from a lender
    /// older than the field): the icon beside « Chain ».
    var computingHardwareModel: String? {
        mirror != nil ? mirrorMachine?.hardwareModel : Self.hardwareModel
    }

    private static let hardwareModel = Diagnostic.sysctlString("hw.model")

}

struct MapFacts: Equatable {
    var inUse: RemoteMap?
    var stored: RemoteMap?
    /// The last render ran off the model in use (`RenderLoop.Verdict.drift`): this Mac's own, never a lender's.
    var drift: RenderLoop.Note?
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
