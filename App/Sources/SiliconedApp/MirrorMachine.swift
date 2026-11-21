// **The lending Mac's machine — said to the Macs that borrow it, and read by them** (`Remote/Machine.swift`).
// Both sides here; the protocol is in `Mirror.swift`.
//
//     lender                                          borrower
//       MachinePublisher: watches B and the ready      receiveMachine: the rack's chip
//       models; sends `machine` at each join and       OtherMacs.machines: the last one of each Mac,
//       on each change                                 kept for the « On: » menu and the Settings

import Foundation
import Siliconed
import SwiftUI

// MARK: - Lending side

@MainActor
final class MachinePublisher {
    private let app: AppState
    private let host: RemoteHost
    private var last: RemoteMachine?
    private let machine = Mac.current()

    init(app: AppState, host: RemoteHost) {
        self.app = app; self.host = host
        watch()
    }

    func snapshot() -> RemoteMachine {
        RemoteMachine(chip: machine.chip, hardwareModel: machine.hardwareModel, memoryBytes: machine.memoryBytes,
                      ceiling: app.memoryCeiling, ready: app.readySet.sorted())
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
}

// MARK: - Borrowing side

extension AppState {
    func receiveMachine(_ m: RemoteMachine) {
        mirrorMachine = m
    }

    /// `hw.model` of the Mac that renders — this one, or the lender in a mirror (`nil` until it said its machine):
    /// the icon beside « Chain ».
    var computingHardwareModel: String? {
        mirror != nil ? mirrorMachine?.hardwareModel : Self.hardwareModel
    }

    private static let hardwareModel = Mac.current().hardwareModel
}

extension RemoteMachine {
    /// What the « On: » menu and Settings › Other Macs say of a Mac: chip, memory, B, models.
    var summary: String {
        [chip, Figures.memory(memoryBytes), String(localized: "Limit \(Figures.memory(ceiling))"),
         String(localized: "Models ready: \(ready.count)")].joined(separator: " · ")
    }
}
