// **Other Macs, in the Settings' Macs pane**: lend this Mac, pair a Mac, the Macs paired each way — and the
// two pairing sheets, one per side, where the same six digits are compared.
//
//     OTHER MACS
//     ┌────────────────────────────────────────────────────────────┐
//     │ [■] Lend This Mac                                    (●○)  │
//     │     Paired Macs render on this one, with its models.       │
//     │     [Pair a Mac…]                                          │
//     │ [▣] Neo de Gwenn   ● connected                   [Forget]  │
//     ├────────────────────────────────────────────────────────────┤
//     │ [■] Compute on Another Mac              [Pair With a Mac…] │
//     │ [▣] Mac mini de Gwenn  ● connected   [Disconnect] [Forget] │
//     │     Apple M4 Pro · 64 GB · Budget 46 GB · Measured · …     │
//     └────────────────────────────────────────────────────────────┘
//     One Mac renders for another on the same network.
//
// A paired Mac wears its own icon (`MacIcon`, from the model it said it is); one never seen, a plain tile.
// The acceptance is on the lending side only (`Pairing.swift`: the code proves nothing typed on the
// other one); the borrowing sheet shows the code and says where to accept.

import Siliconed
import SwiftUI

struct OtherMacsSection: View {
    @Bindable var macs: OtherMacs
    @State private var hostSheet = false
    @State private var clientSheet = false

    var body: some View {
        SettingsGroup("Other Macs", footer: "One Mac renders for another on the same network.") {
            SettingRow(Text("Lend This Mac"), Text("Paired Macs render on this one, with its models."),
                       detail: "Paired Macs render on this one, with its models. Nothing connects before you accept a pairing.") {
                IconTile(symbol: "antenna.radiowaves.left.and.right", color: Palette.peach, side: 26)
            } trailing: {
                SettingSwitch("Lend This Mac", isOn: $macs.lending)
            } below: {
                if macs.lending {
                    Button("Pair a Mac…") {
                        macs.openPairing()
                        hostSheet = true
                    }
                }
                if let p = macs.lendingProblem {
                    Text(verbatim: p).font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
                }
            }
            if macs.lending {
                ForEach(macs.pairings?.lent ?? [], id: \.identity) { m in
                    let hello = macs.connected[m.identity]
                    SettingRow(Text(verbatim: m.name)) {
                        Self.icon(hello?.machine.hardwareModel)
                    } trailing: {
                        Button("Forget") { macs.forget(lent: m) }
                    } below: {
                        LinkState(text: hello != nil ? String(localized: "connected") : String(localized: "not connected"),
                                  color: hello != nil ? Palette.live : .secondary)
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            SettingRow(Text("Compute on Another Mac"), Text("Pair once, with a code shown on both screens.")) {
                IconTile(symbol: "network", color: Palette.sky, side: 26)
            } trailing: {
                Button("Pair With a Mac…") {
                    macs.lookForPairable()
                    clientSheet = true
                }
            }
            ForEach(macs.pairings?.borrowed ?? [], id: \.id) { m in
                SettingRow(Text(verbatim: m.name)) {
                    Self.icon(macs.machines[m.id]?.hardwareModel)
                } trailing: {
                    HStack {
                        if macs.links[m.id] == nil {
                            Button("Connect") { macs.connect(m) }
                        } else {
                            Button("Disconnect") { macs.disconnect(m.id) }
                        }
                        Button("Forget") { macs.forget(borrowed: m) }
                    }
                } below: {
                    VStack(alignment: .leading, spacing: 2) {
                        LinkState(text: Self.state(macs.links[m.id], found: macs.found[m.id] != nil), color: Self.dot(macs.links[m.id]))
                        if let machine = macs.machines[m.id] {
                            Text(verbatim: machine.summary).foregroundStyle(.tertiary)
                        }
                    }
                    .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .onAppear { macs.load() }
        .sheet(isPresented: $hostSheet, onDismiss: macs.closePairing) {
            HostPairingSheet(macs: macs) { hostSheet = false }
        }
        .sheet(isPresented: $clientSheet, onDismiss: macs.stopLookingForPairable) {
            ClientPairingSheet(macs: macs) { clientSheet = false }
        }
    }

    /// A paired Mac's icon, from the model it said it is; a plain tile while it has said nothing.
    @ViewBuilder private static func icon(_ hardwareModel: String?) -> some View {
        if let hardwareModel {
            MacIcon(hardwareModel: hardwareModel).frame(width: 30, height: 30)
        } else {
            IconTile(symbol: "laptopcomputer", color: .secondary, side: 26)
        }
    }

    /// The dot of a link: what runs when connected, orange on the way, red when refused.
    static func dot(_ s: RemoteLink.State?) -> Color {
        switch s {
        case .connected?: Palette.live
        case .connecting?, .waiting?: .orange
        case .refused?: Palette.failure
        case nil, .off?: .secondary
        }
    }

    static func state(_ s: RemoteLink.State?, found: Bool) -> String {
        switch s {
        case nil, .off?: found ? String(localized: "on the network") : String(localized: "not on the network")
        case .connecting?: String(localized: "connecting…")
        case .connected?: String(localized: "connected")
        case .waiting?: String(localized: "lost, trying again…")
        case .refused(.forgotten)?: String(localized: "refused: that Mac forgot this one; pair again")
        case .refused(.mismatch(let m))?: mismatch(m)
        }
    }

    static func mismatch(_ m: Hello.Mismatch) -> String {
        switch m {
        case .protocolVersion, .appVersion: String(localized: "refused: the two Macs must run the same version of Siliconed")
        }
    }
}

/// The six digits, large, as both screens write them.
private struct Code: View {
    let code: String
    var body: some View {
        Text(verbatim: Pairing.spaced(code)).font(.system(size: 40, weight: .semibold, design: .rounded)).monospacedDigit()
            .textSelection(.disabled)
    }
}

/// **The lending side**: waits for a Mac, shows its name and the code, and only here is it accepted.
private struct HostPairingSheet: View {
    let macs: OtherMacs
    let done: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            switch macs.window {
            case nil, .waiting?:
                ProgressView()
                Text("Waiting for a Mac…").font(.headline)
                Text("On the other Mac: Settings → Macs → Pair With a Mac…, then choose “\(macs.name)”.")
                    .multilineTextAlignment(.center)
                Button("Cancel", action: done).keyboardShortcut(.cancelAction)
            case .compare(let client, let code)?:
                Text("Does “\(client)” show this code?").font(.headline)
                Code(code: code)
                Text("Accept only if both screens show the same six digits.").font(.caption).foregroundStyle(.secondary)
                HStack {
                    Button("Refuse") { macs.refusePairing() }.keyboardShortcut(.cancelAction)
                    // No default key: a Return typed while the other Mac's code comes up must not accept it.
                    Button("Accept") { macs.acceptPairing() }
                }
            case .paired(_, let client)?:
                Text("Paired with “\(client)”.").font(.headline)
                Button("Done", action: done).keyboardShortcut(.defaultAction).tint(Palette.control)
            case .ended(let how)?:
                Text(Self.ended(how)).font(.headline).multilineTextAlignment(.center)
                Button("Close", action: done).keyboardShortcut(.defaultAction).tint(Palette.control)
            }
        }
        .padding(24)
        .frame(width: 380)
    }

    static func ended(_ how: RemoteHost.PairingEnd) -> String {
        switch how {
        case .closed: String(localized: "The other Mac stopped the pairing.")
        case .timedOut: String(localized: "No Mac came in two minutes.")
        case .refused: String(localized: "Pairing refused.")
        case .failed: String(localized: "The pairing failed: nothing was kept.")
        }
    }
}

/// **The borrowing side**: the Macs whose pairing window is open, then the code to compare.
private struct ClientPairingSheet: View {
    let macs: OtherMacs
    let done: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            if let a = macs.attempt {
                switch a.step {
                case .connecting:
                    ProgressView()
                    Text("Connecting to “\(a.with.name)”…").font(.headline)
                case .compare(let code):
                    Text("Check that “\(a.with.name)” shows this code, and accept there.").font(.headline)
                        .multilineTextAlignment(.center)
                    Code(code: code)
                case .paired:
                    Text("Paired with “\(a.with.name)”.").font(.headline)
                    Button("Done", action: done).keyboardShortcut(.defaultAction).tint(Palette.control)
                case .ended(let how):
                    Text(Self.ended(how, a.with.name)).font(.headline).multilineTextAlignment(.center)
                    Button("Close", action: done).keyboardShortcut(.defaultAction).tint(Palette.control)
                }
                if case .paired = a.step {} else if case .ended = a.step {} else {
                    Button("Cancel", action: done).keyboardShortcut(.cancelAction)
                }
            } else {
                Text("Choose the Mac to compute on").font(.headline)
                if macs.pairable.isEmpty {
                    ProgressView()
                    Text("On that Mac: Settings → Macs → Lend This Mac, then Pair a Mac…")
                        .multilineTextAlignment(.center).font(.caption).foregroundStyle(.secondary)
                } else {
                    ForEach(macs.pairable) { m in
                        Button { macs.pair(with: m) } label: {
                            Text(verbatim: m.name).frame(maxWidth: .infinity)
                        }
                    }
                }
                Button("Cancel", action: done).keyboardShortcut(.cancelAction)
            }
        }
        .padding(24)
        .frame(width: 380)
    }

    static func ended(_ how: RemotePairingAttempt.PairingEnd, _ name: String) -> String {
        switch how {
        case .refused: String(localized: "“\(name)” refused the pairing.")
        case .failed: String(localized: "The pairing failed: nothing was kept.")
        case .unreachable: String(localized: "“\(name)” is no longer waiting for a Mac: open its pairing window again.")
        case .cancelled: String(localized: "Pairing cancelled.")
        }
    }
}
